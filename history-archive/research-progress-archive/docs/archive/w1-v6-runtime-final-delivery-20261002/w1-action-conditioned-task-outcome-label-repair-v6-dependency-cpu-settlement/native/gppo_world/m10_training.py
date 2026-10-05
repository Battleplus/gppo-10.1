"""Self-contained M-10 PPO, graph/history, world-model and trigger experiments.

The training contract is intentionally small enough to run reproducibly on the
specified server while retaining the real M-10 message/ACK/service semantics.
All policy variants consume the same public observation and action mask; only
the encoder/history/context treatment changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .m10_environment import M10Config, M10Environment, M10Scenario, default_scenario, scenario_tape, scenario_to_dict


EVENT_NAMES = ("damage", "disconnect", "reconnect", "reserved_event_3")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 256
    update_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_weight: float = 0.01
    value_weight: float = 0.5
    grad_clip: float = 0.5
    minibatch_size: int = 128


class M10WorldModel(nn.Module):
    """Predict one-step reward/event consequences from public input and action."""

    def __init__(self, obs_dim: int, action_count: int, context_dim: int = 8):
        super().__init__()
        self.obs_dim, self.action_count, self.context_dim = obs_dim, action_count, context_dim
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim + action_count, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
        )
        self.reward_head = nn.Linear(128, 1)
        self.event_head = nn.Linear(128, 4)
        self.done_head = nn.Linear(128, 1)
        self.context_head = nn.Linear(128, context_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        one_hot = torch.nn.functional.one_hot(action.long(), self.action_count).to(obs.dtype)
        hidden = self.backbone(torch.cat((obs, one_hot), dim=-1))
        return {
            "hidden": hidden,
            "reward": self.reward_head(hidden).squeeze(-1),
            "event_logits": self.event_head(hidden),
            "done_logit": self.done_head(hidden).squeeze(-1),
            "context": self.context_head(hidden),
        }

    @torch.no_grad()
    def context_for(self, obs: torch.Tensor, *, action_mask: torch.Tensor | None = None,
                    action: int | None = None, triggered: bool = False,
                    threshold: float = 0.5, ood_limit: float = 20.0) -> tuple[torch.Tensor, bool, float]:
        """Return an action-conditional expected context over legal candidates.

        A policy decision has no action yet, so the default is an explicit uniform
        expectation over the current legal action mask.  Passing a fixed action is
        allowed only when the caller names it explicitly and proves it legal.
        Non-finite or extreme inputs use the zero-context safety fallback.
        """
        if not torch.isfinite(obs).all() or float(obs.abs().max()) > ood_limit:
            return torch.zeros((obs.shape[0], self.context_dim), device=obs.device, dtype=obs.dtype), False, 1.0
        if action is not None:
            if action_mask is not None and (action < 0 or action >= action_mask.shape[-1] or not bool(action_mask[..., action].all())):
                raise ValueError("reference action is not legal under the supplied mask")
            actions = torch.full((obs.shape[0],), int(action), dtype=torch.long, device=obs.device)
            output = self.forward(obs, actions)
        else:
            if action_mask is None:
                raise ValueError("action_mask is required for action-agnostic world context")
            legal = torch.nonzero(action_mask.bool().flatten(), as_tuple=False).flatten()
            if len(legal) == 0:
                legal = torch.tensor([self.action_count - 1], dtype=torch.long, device=obs.device)
            expanded_obs = obs.repeat(len(legal), 1)
            output = self.forward(expanded_obs, legal.to(obs.device))
        risk = float(torch.sigmoid(output["event_logits"]).max().item())
        context = output["context"].mean(dim=0, keepdim=True) if action is None else output["context"]
        active = not triggered or risk >= threshold
        return context if active else torch.zeros_like(context), active, risk


class M10ActorCritic(nn.Module):
    """PPO actor with explicit entity nodes, relations and candidate scoring."""

    node_width = 32

    def __init__(self, *, uav_count: int, task_capacity: int, action_count: int,
                 encoder: str, type_count: int, history: bool, context_dim: int = 0,
                 region_count: int = 3, target_count: int = 4, event_capacity: int = 4,
                 relation_width: int = 4):
        super().__init__()
        if encoder not in ("mlp", "graph"):
            raise ValueError("encoder must be mlp or graph")
        if type_count not in (2, 5):
            raise ValueError("type_count must be 2 or 5")
        self.uav_count, self.task_capacity, self.action_count = uav_count, task_capacity, action_count
        self.region_count, self.target_count, self.event_capacity = region_count, target_count, event_capacity
        self.relation_width = relation_width
        self.encoder_name, self.type_count, self.history, self.context_dim = encoder, type_count, history, context_dim
        self.node_counts = (uav_count, region_count, target_count, task_capacity, event_capacity)
        self.node_count = sum(self.node_counts)
        self.base_obs_dim = self.node_count * self.node_width + uav_count * task_capacity * relation_width + 2
        self.input_dim = self.base_obs_dim + context_dim
        if encoder == "mlp":
            self.encoder = nn.Sequential(nn.Linear(self.input_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
            feature_dim = 128
        else:
            self.token_encoder = nn.Sequential(nn.Linear(self.node_width, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
            self.type_embedding = nn.Embedding(type_count, 64)
            self.graph_projection = nn.Sequential(nn.Linear(64, 128), nn.Tanh())
            self.context_projection = nn.Linear(context_dim, 128) if context_dim else None
            self.edge_encoder = nn.Sequential(nn.Linear(64 * 2 + relation_width, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
            self.pair_actor = nn.Sequential(nn.Linear(64 + 128, 64), nn.Tanh(), nn.Linear(64, 1))
            self.noop_actor = nn.Linear(128, 1)
            feature_dim = 128
        self.gru = nn.GRU(128, 128, batch_first=True) if history else None
        self.actor = nn.Linear(feature_dim, action_count) if encoder == "mlp" else None
        self.critic = nn.Linear(feature_dim, 1)

    def _unpack(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        blocks = []
        start = 0
        for count in self.node_counts:
            size = count * self.node_width
            blocks.append(obs[:, start:start + size].reshape(-1, count, self.node_width))
            start += size
        relations = obs[:, start:start + self.uav_count * self.task_capacity * self.relation_width]
        relations = relations.reshape(-1, self.uav_count, self.task_capacity, self.relation_width)
        return torch.cat(blocks, dim=1), relations, obs[:, -2:]

    def _graph_representation(self, base: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, relations, _ = self._unpack(base)
        if self.type_count == 2:
            type_ids = torch.cat((
                torch.zeros(self.uav_count, dtype=torch.long, device=base.device),
                torch.ones(self.node_count - self.uav_count, dtype=torch.long, device=base.device),
            ))
        else:
            type_ids = torch.cat(tuple(torch.full((count,), index, dtype=torch.long, device=base.device)
                                       for index, count in enumerate(self.node_counts)))
        encoded = self.token_encoder(tokens) + self.type_embedding(type_ids)[None, :, :]
        pooled = encoded.mean(dim=1) + encoded.amax(dim=1)
        features = self.graph_projection(pooled)
        if self.context_projection is not None:
            features = features + self.context_projection(context)
        task_start = self.uav_count + self.region_count + self.target_count
        uav_encoded = encoded[:, :self.uav_count]
        task_encoded = encoded[:, task_start:task_start + self.task_capacity]
        pair_input = torch.cat((
            uav_encoded[:, :, None, :].expand(-1, -1, self.task_capacity, -1),
            task_encoded[:, None, :, :].expand(-1, self.uav_count, -1, -1),
            relations,
        ), dim=-1)
        pair_messages = self.edge_encoder(pair_input)
        return features, pair_messages

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        base = obs[:, :self.base_obs_dim]
        context = obs[:, self.base_obs_dim:]
        if self.encoder_name == "mlp":
            return self.encoder(torch.cat((base, context), dim=-1))
        features, _ = self._graph_representation(base, context)
        return features

    def _features(self, obs: torch.Tensor, hidden: torch.Tensor | None = None):
        base = obs[:, :self.base_obs_dim]
        context = obs[:, self.base_obs_dim:]
        if self.encoder_name == "mlp":
            features = self.encoder(torch.cat((base, context), dim=-1))
            pair_messages = None
        else:
            features, pair_messages = self._graph_representation(base, context)
        next_hidden = hidden
        if self.gru is not None:
            features, next_hidden = self.gru(features[:, None, :], hidden)
            features = features[:, 0, :]
        return features, pair_messages, next_hidden

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor | None = None):
        features, pair_messages, next_hidden = self._features(obs, hidden)
        if pair_messages is None:
            logits = self.actor(features)
        else:
            pair_logits = self.pair_actor(torch.cat((pair_messages, features[:, None, None, :].expand(-1, self.uav_count, self.task_capacity, -1)), dim=-1)).squeeze(-1)
            logits = torch.cat((pair_logits.reshape(-1, self.uav_count * self.task_capacity), self.noop_actor(features)), dim=-1)
        return logits, self.critic(features).squeeze(-1), next_hidden

    def value_only(self, obs: torch.Tensor, hidden: torch.Tensor | None = None):
        """Advance the encoder/critic without evaluating the actor head.

        This is the execution path for a continuation interval.  The value
        estimate remains available for GAE, but no actor logits or action
        distribution are computed when no new policy decision is requested.
        """
        features, _, next_hidden = self._features(obs, hidden)
        return self.critic(features).squeeze(-1), next_hidden


def masked_distribution(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    safe_mask = mask.bool().clone()
    no_action = ~safe_mask.any(dim=-1)
    if no_action.any():
        safe_mask[no_action, -1] = True
    return Categorical(logits=logits.masked_fill(~safe_mask, -1e9))


@dataclass
class Transition:
    obs: np.ndarray
    context: np.ndarray
    mask: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    episode_start: bool
    info: dict[str, Any]
    terminated: bool = False
    truncated: bool = False
    next_value: float = 0.0
    actor_decision: bool = True


def _public_vector(obs: dict[str, Any], context: np.ndarray | None = None) -> np.ndarray:
    base = np.asarray(obs["flat"], dtype=np.float32)
    extra = np.zeros(0, dtype=np.float32) if context is None else np.asarray(context, dtype=np.float32)
    return np.concatenate((base, extra))


def _make_env(seed: int, scenario_name: str = "mixed", config: M10Config | None = None,
              scenario: M10Scenario | None = None) -> M10Environment:
    cfg = config or M10Config(seed=seed)
    return M10Environment(cfg, scenario or default_scenario(scenario_name, seed=seed, split="train"))


def collect_world_dataset(*, episodes: int, seed: int, config: M10Config,
                          scenarios: Iterable[M10Scenario] | None = None,
                          split: str = "train") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tape = list(scenarios or scenario_tape(split, count=episodes, base_seed=seed))
    if len(tape) < episodes:
        raise ValueError("scenario tape is shorter than requested world episodes")
    for episode in range(episodes):
        scenario = tape[episode]
        env = _make_env(scenario.seed, "mixed", config, scenario=scenario)
        obs = env.reset()
        done = False
        while not done:
            before_events = len(env.clock.log)
            rng = np.random.default_rng(seed * 1000 + episode * 17 + int(env.clock.time))
            legal = np.flatnonzero(obs["mask"])
            action = int(rng.choice(legal))
            next_obs, reward, done, info = env.step(action)
            new_events = env.clock.log[before_events:]
            event_target = np.zeros(4, dtype=np.float32)
            names = {"damage": 0, "disconnect": 1, "reconnect": 2}
            for event in new_events:
                event_kind = event.get("kind")
                if event_kind in names:
                    event_target[names[event_kind]] = 1.0
            persistence_target = np.zeros(4, dtype=np.float32)
            # This is a deliberately simple, public-information persistence
            # baseline: repeat the most recently received semantic UAV state
            # change.  It is recorded per row so it cannot be reconstructed
            # after a split or silently tuned on the held-out tape.
            for record in reversed(env._public_event_records):  # type: ignore[attr-defined]
                field = record.get("field")
                value = float(record.get("value", 0.0))
                if field == "alive" and value < 0.5:
                    persistence_target[0] = 1.0
                    break
                if field == "connected":
                    persistence_target[1 if value < 0.5 else 2] = 1.0
                    break
            rows.append({
                "episode": episode,
                "obs": obs["flat"].astype(np.float32),
                "action": action,
                "reward": float(reward),
                "next_obs": next_obs["flat"].astype(np.float32),
                "event_target": event_target,
                "persistence_event_target": persistence_target,
                "done": float(done),
                "context_target": np.concatenate((np.asarray([float(reward)], dtype=np.float32), event_target, np.asarray([float(done)], dtype=np.float32), next_obs["flat"][-2:].astype(np.float32))),
                "split": split,
                "tape_id": scenario.tape_id,
                "label_semantics": {
                    "horizon": "next_environment_step",
                    "events": dict(zip(EVENT_NAMES, ("new_service_clock_event", "new_service_clock_event", "new_service_clock_event", "reserved"))),
                    "done": "terminated_or_time_limit_after_step",
                },
            })
            obs = next_obs
    return rows


def _world_losses(output: dict[str, torch.Tensor], rewards: torch.Tensor, events: torch.Tensor,
                  dones: torch.Tensor, context_targets: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "reward_mse": torch.nn.functional.mse_loss(output["reward"], rewards),
        "event_bce": torch.nn.functional.binary_cross_entropy_with_logits(output["event_logits"], events),
        "done_bce": torch.nn.functional.binary_cross_entropy_with_logits(output["done_logit"], dones),
        "context_mse": torch.nn.functional.mse_loss(output["context"], context_targets),
    }


def _world_metrics(output: dict[str, torch.Tensor], rewards: torch.Tensor, events: torch.Tensor,
                   dones: torch.Tensor, context_targets: torch.Tensor) -> dict[str, float]:
    losses = _world_losses(output, rewards, events, dones, context_targets)
    event_pred = (torch.sigmoid(output["event_logits"]) >= 0.5).to(events.dtype)
    done_pred = (torch.sigmoid(output["done_logit"]) >= 0.5).to(dones.dtype)
    return {
        "reward_rmse": float(losses["reward_mse"].sqrt().detach()),
        "event_bce": float(losses["event_bce"].detach()),
        "event_accuracy": float((event_pred == events).float().mean().detach()),
        "done_bce": float(losses["done_bce"].detach()),
        "done_accuracy": float((done_pred == dones).float().mean().detach()),
        "context_rmse": float(losses["context_mse"].sqrt().detach()),
    }


def _average_precision(scores: np.ndarray, targets: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ordered_targets = targets[order].astype(np.float64)
    positives = float(ordered_targets.sum())
    if positives <= 0:
        return float("nan")
    cumulative = np.cumsum(ordered_targets)
    precision = cumulative / np.arange(1, len(ordered_targets) + 1, dtype=np.float64)
    return float((precision * ordered_targets).sum() / positives)


def _binary_metrics(scores: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    predicted = scores >= threshold
    actual = targets >= 0.5
    tp = float(np.sum(predicted & actual))
    fp = float(np.sum(predicted & ~actual))
    fn = float(np.sum(~predicted & actual))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        in_bin = (scores >= left) & ((scores < right) if right < 1.0 else (scores <= right))
        if np.any(in_bin):
            ece += float(np.mean(in_bin)) * abs(float(np.mean(scores[in_bin])) - float(np.mean(actual[in_bin])))
    return {
        "precision": precision,
        "recall": recall,
        "pr_auc": _average_precision(scores, actual.astype(np.float64)),
        "brier": float(np.mean((scores - actual) ** 2)),
        "ece_10bin": ece,
        "positive_count": float(np.sum(actual)),
        "sample_count": float(len(actual)),
    }


@torch.no_grad()
def evaluate_world_model_rows(model: M10WorldModel, rows: list[dict[str, Any]], *, device: str = "cpu",
                              baseline_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Report event- and done-aware metrics for a frozen row set.

    The caller controls the split.  In particular, calibration and the
    prevalence baseline must be built from train/validation rows before this
    function is called on a final test tape.
    """
    if not rows:
        raise ValueError("world-model evaluation rows are empty")
    device_obj = torch.device(device)
    model = model.to(device_obj).eval()
    obs = torch.tensor(np.stack([row["obs"] for row in rows]), dtype=torch.float32, device=device_obj)
    actions = torch.tensor([row["action"] for row in rows], dtype=torch.long, device=device_obj)
    rewards = np.asarray([float(row["reward"]) for row in rows], dtype=np.float64)
    events = np.stack([row["event_target"] for row in rows]).astype(np.float64)
    dones = np.asarray([float(row["done"]) for row in rows], dtype=np.float64)
    output = model(obs, actions)
    reward_pred = output["reward"].detach().cpu().numpy()
    event_pred = torch.sigmoid(output["event_logits"]).detach().cpu().numpy()
    done_pred = torch.sigmoid(output["done_logit"]).detach().cpu().numpy()
    event_metrics = {name: _binary_metrics(event_pred[:, index], events[:, index]) for index, name in enumerate(EVENT_NAMES)}
    baseline_rows = baseline_rows or rows
    base_rewards = np.asarray([float(row["reward"]) for row in baseline_rows], dtype=np.float64)
    base_events = np.stack([row["event_target"] for row in baseline_rows]).astype(np.float64)
    base_dones = np.asarray([float(row["done"]) for row in baseline_rows], dtype=np.float64)
    event_rate = np.clip(base_events.mean(axis=0), 1e-5, 1.0 - 1e-5)
    done_rate = float(np.clip(base_dones.mean(), 1e-5, 1.0 - 1e-5))
    prevalence_event = np.repeat(event_rate[None, :], len(rows), axis=0)
    persistence_available = all("persistence_event_target" in row for row in rows)
    persistence_event = np.stack([row["persistence_event_target"] for row in rows]).astype(np.float64) if persistence_available else None
    split_counts: dict[str, Any] = {}
    for split in sorted({str(row.get("split", "unspecified")) for row in rows}):
        selected = [row for row in rows if str(row.get("split", "unspecified")) == split]
        split_events = np.stack([row["event_target"] for row in selected]).astype(np.float64)
        split_counts[split] = {
            "rows": len(selected), "episodes": len({row.get("tape_id") for row in selected}),
            "event_positive_count": split_events.sum(axis=0).astype(int).tolist(),
            "event_positive_rate": split_events.mean(axis=0).tolist(),
            "done_count": int(sum(float(row["done"]) >= 0.5 for row in selected)),
            "done_rate": float(np.mean([float(row["done"]) for row in selected])),
        }
    metrics = {
        "rows": len(rows),
        "reward_rmse": float(np.sqrt(np.mean((reward_pred - rewards) ** 2))),
        "reward_mean_baseline_rmse": float(np.sqrt(np.mean((base_rewards.mean() - rewards) ** 2))),
        "event": event_metrics,
        "event_bce": float(np.mean(-(events * np.log(np.clip(event_pred, 1e-7, 1 - 1e-7)) + (1 - events) * np.log(np.clip(1 - event_pred, 1e-7, 1 - 1e-7))))),
        "done": _binary_metrics(done_pred, dones),
        "done_bce": float(np.mean(-(dones * np.log(np.clip(done_pred, 1e-7, 1 - 1e-7)) + (1 - dones) * np.log(np.clip(1 - done_pred, 1e-7, 1 - 1e-7))))),
        "done_never_true_baseline": _binary_metrics(np.zeros_like(dones), dones),
        "baselines": {
            "train_prevalence_event": {name: _binary_metrics(np.full(len(rows), event_rate[index]), events[:, index]) for index, name in enumerate(EVENT_NAMES)},
            "train_prevalence_done": _binary_metrics(np.full(len(rows), done_rate), dones),
            "zero_event_rule": {name: _binary_metrics(np.zeros(len(rows)), events[:, index]) for index, name in enumerate(EVENT_NAMES)},
        },
        "split_counts": split_counts,
    }
    if persistence_event is not None:
        metrics["baselines"]["persistence_event"] = {name: _binary_metrics(persistence_event[:, index], events[:, index]) for index, name in enumerate(EVENT_NAMES)}
    return metrics


def train_world_model(rows: list[dict[str, Any]], *, seed: int, action_count: int,
                      epochs: int = 30, device: str = "cpu",
                      ood_rows: list[dict[str, Any]] | None = None) -> tuple[M10WorldModel, dict[str, Any]]:
    if not rows:
        raise ValueError("world-model dataset is empty")
    seed_everything(seed)
    device_obj = torch.device(device)
    obs = torch.tensor(np.stack([row["obs"] for row in rows]), dtype=torch.float32, device=device_obj)
    actions = torch.tensor([row["action"] for row in rows], dtype=torch.long, device=device_obj)
    rewards = torch.tensor([row["reward"] for row in rows], dtype=torch.float32, device=device_obj)
    events = torch.tensor(np.stack([row["event_target"] for row in rows]), dtype=torch.float32, device=device_obj)
    dones = torch.tensor([row["done"] for row in rows], dtype=torch.float32, device=device_obj)
    context_targets = torch.tensor(np.stack([row["context_target"] for row in rows]), dtype=torch.float32, device=device_obj)
    split_values = {str(row.get("split", "")) for row in rows}
    if {"train", "validation", "test"}.issubset(split_values):
        train_idx_list = [i for i, row in enumerate(rows) if row.get("split") == "train"]
        val_idx_list = [i for i, row in enumerate(rows) if row.get("split") == "validation"]
        test_idx_list = [i for i, row in enumerate(rows) if row.get("split") == "test"]
    else:
        episode_ids = sorted({int(row["episode"]) for row in rows})
        rng = np.random.default_rng(seed)
        rng.shuffle(episode_ids)
        n_train_episodes = max(1, int(len(episode_ids) * 0.7))
        n_val_episodes = max(n_train_episodes + 1, int(len(episode_ids) * 0.85))
        train_episodes = set(episode_ids[:n_train_episodes])
        val_episodes = set(episode_ids[n_train_episodes:n_val_episodes])
        test_episodes = set(episode_ids[n_val_episodes:])
        train_idx_list = [i for i, row in enumerate(rows) if row["episode"] in train_episodes]
        val_idx_list = [i for i, row in enumerate(rows) if row["episode"] in val_episodes]
        test_idx_list = [i for i, row in enumerate(rows) if row["episode"] in test_episodes]
    train_idx = torch.tensor(train_idx_list, dtype=torch.long, device=device_obj)
    val_idx = torch.tensor(val_idx_list, dtype=torch.long, device=device_obj)
    test_idx = torch.tensor(test_idx_list, dtype=torch.long, device=device_obj)
    train_episodes = sorted({int(rows[i]["episode"]) for i in train_idx_list})
    val_episodes = sorted({int(rows[i]["episode"]) for i in val_idx_list})
    test_episodes = sorted({int(rows[i]["episode"]) for i in test_idx_list})
    model = M10WorldModel(obs.shape[1], action_count).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.optimizer = optimizer  # type: ignore[attr-defined]
    history = []
    for epoch in range(epochs):
        model.train()
        output = model(obs[train_idx], actions[train_idx])
        losses = _world_losses(output, rewards[train_idx], events[train_idx], dones[train_idx], context_targets[train_idx])
        loss = losses["reward_mse"] + losses["event_bce"] + 0.2 * losses["done_bce"] + 0.5 * losses["context_mse"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_idx_for_loss = val_idx if len(val_idx) else train_idx
            val = model(obs[val_idx_for_loss], actions[val_idx_for_loss])
            val_losses = _world_losses(val, rewards[val_idx_for_loss], events[val_idx_for_loss], dones[val_idx_for_loss], context_targets[val_idx_for_loss])
            val_loss = val_losses["reward_mse"] + val_losses["event_bce"] + 0.2 * val_losses["done_bce"] + 0.5 * val_losses["context_mse"]
        history.append({"epoch": epoch + 1, "train_loss": float(loss.detach()), "validation_loss": float(val_loss.detach()), "context_loss": float(losses["context_mse"].detach())})
    model.eval()
    metadata = {
        "dataset_rows": len(rows), "train_rows": len(train_idx), "validation_rows": len(val_idx), "test_rows": len(test_idx),
        "train_episodes": train_episodes, "validation_episodes": val_episodes, "test_episodes": test_episodes,
        "epochs": epochs, "seed": seed, "history": history, "split_unit": "frozen_episode_tape",
        "loss_weights": {"reward_mse": 1.0, "event_bce": 1.0, "done_bce": 0.2, "context_mse": 0.5},
        "context_target": "[reward,event_0..3,done,time_fraction,event_signal]",
    }
    with torch.no_grad():
        metadata["test_metrics"] = (_world_metrics(model(obs[test_idx], actions[test_idx]), rewards[test_idx], events[test_idx], dones[test_idx], context_targets[test_idx]) if len(test_idx) else {})
        train_reward_mean = rewards[train_idx].mean()
        train_event_rate = events[train_idx].mean(dim=0).clamp(1e-5, 1 - 1e-5)
        train_done_rate = dones[train_idx].mean().clamp(1e-5, 1 - 1e-5)
        metadata["test_baseline_metrics"] = {
            "reward_rmse": float(torch.mean((rewards[test_idx] - train_reward_mean) ** 2).sqrt()),
            "event_bce": float(torch.nn.functional.binary_cross_entropy(train_event_rate.expand_as(events[test_idx]), events[test_idx])),
            "done_bce": float(torch.nn.functional.binary_cross_entropy(train_done_rate.expand_as(dones[test_idx]), dones[test_idx])),
        }
        if ood_rows:
            ood_obs = torch.tensor(np.stack([row["obs"] for row in ood_rows]), dtype=torch.float32, device=device_obj)
            ood_actions = torch.tensor([row["action"] for row in ood_rows], dtype=torch.long, device=device_obj)
            ood_rewards = torch.tensor([row["reward"] for row in ood_rows], dtype=torch.float32, device=device_obj)
            ood_events = torch.tensor(np.stack([row["event_target"] for row in ood_rows]), dtype=torch.float32, device=device_obj)
            ood_dones = torch.tensor([row["done"] for row in ood_rows], dtype=torch.float32, device=device_obj)
            ood_context = torch.tensor(np.stack([row["context_target"] for row in ood_rows]), dtype=torch.float32, device=device_obj)
            metadata["ood_metrics"] = _world_metrics(model(ood_obs, ood_actions), ood_rewards, ood_events, ood_dones, ood_context)
    return model, metadata


def _policy_input(env: M10Environment, obs: dict[str, Any], model: M10WorldModel | None,
                  *, fusion: str, device: torch.device, trigger_threshold: float,
                  force_context: bool = False) -> tuple[np.ndarray, bool, float]:
    vector, active, risk, full_vector = _policy_input_bundle(
        env, obs, model, fusion=fusion, device=device, trigger_threshold=trigger_threshold,
    )
    if force_context and model is not None and fusion == "triggered" and not active:
        # Kept for source compatibility with old callers.  New decision paths
        # use the full vector returned by _policy_input_bundle, so this branch
        # is never needed for a second inference on the same snapshot.
        return full_vector, True, risk
    return vector, active, risk


def _policy_input_bundle(env: M10Environment, obs: dict[str, Any], model: M10WorldModel | None,
                         *, fusion: str, device: torch.device, trigger_threshold: float
                         ) -> tuple[np.ndarray, bool, float, np.ndarray]:
    """Build gated and full policy vectors from one decision snapshot.

    The full vector is used when a semantic/safety/max-wait trigger forces a
    decision even if model risk is below threshold.  Returning both vectors
    prevents the historical force_context path from running the world model a
    second time on the same observation/version.
    """
    if model is None or fusion == "base":
        vector = _public_vector(obs)
        return vector, False, 0.0, vector
    with torch.no_grad():
        context, _, risk = model.context_for(
            torch.tensor(obs["flat"], dtype=torch.float32, device=device)[None, :],
            action_mask=torch.tensor(obs["mask"], dtype=torch.bool, device=device),
            triggered=False, threshold=trigger_threshold,
        )
    full_vector = _public_vector(obs, context[0].detach().cpu().numpy())
    active = fusion != "triggered" or risk >= trigger_threshold
    gated_vector = full_vector if active else _public_vector(obs)
    return gated_vector, active, risk, full_vector


def _act(policy: M10ActorCritic, vector: np.ndarray, mask: np.ndarray, hidden: torch.Tensor | None,
         device: torch.device, deterministic: bool, forced_action: int | None = None) -> tuple[int, float, float, torch.Tensor | None]:
    obs_tensor = torch.tensor(vector, dtype=torch.float32, device=device)[None, :]
    mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device)[None, :]
    logits, value, hidden = policy(obs_tensor, hidden)
    dist = masked_distribution(logits, mask_tensor)
    action = torch.argmax(dist.logits, dim=-1) if deterministic and forced_action is None else dist.sample()
    if forced_action is not None:
        if not bool(mask_tensor[0, forced_action]):
            forced_action = policy.action_count - 1
        action = torch.tensor([forced_action], dtype=torch.long, device=device)
    return int(action.item()), float(dist.log_prob(action).item()), float(value.item()), hidden


_TRIGGER_PRIORITY = (
    "initial",
    "safety",
    "confirmed_fault",
    "link_recovery",
    "task_arrival",
    "completion_or_invalidation",
    "risk",
    "max_wait",
)


def _trigger_decision(obs: dict[str, Any], *, fusion: str, risk_active: bool,
                      last_action: int | None, steps_since_replan: int,
                      max_replan_interval: int) -> tuple[bool, str, dict[str, bool]]:
    """Apply the frozen trigger contract before mutating interval state."""
    flags = {key: bool(value) for key, value in obs.get("trigger_flags", {}).items()}
    flags.setdefault("safety_forced", False)
    flags["initial"] = last_action is None
    continuation_action = obs.get("continuation_action")
    flags["safety"] = bool(
        last_action is not None and not bool(obs["mask"][last_action])
        and continuation_action != last_action
    )
    conditions = {
        "initial": flags["initial"],
        "safety": flags["safety"] or flags.get("safety_forced", False),
        "confirmed_fault": flags.get("confirmed_fault", False),
        "link_recovery": flags.get("link_recovery", False),
        "task_arrival": flags.get("task_arrival", False),
        "completion_or_invalidation": flags.get("completion_or_invalidation", False),
        "risk": bool(risk_active),
        "max_wait": steps_since_replan >= max_replan_interval,
    }
    if fusion != "triggered":
        return True, "periodic_policy", conditions
    for reason in _TRIGGER_PRIORITY:
        if conditions[reason]:
            return True, reason, conditions
    return False, "none", conditions


def collect_rollout(policy: M10ActorCritic, *, config: PPOConfig, env_config: M10Config,
                    seed: int, device: torch.device, model: M10WorldModel | None,
                    fusion: str, trigger_threshold: float,
                    scenarios: Iterable[M10Scenario] | None = None,
                    max_replan_interval: int = 3) -> list[Transition]:
    scenario_list = list(scenarios or scenario_tape("train", count=max(8, config.rollout_steps // 8), base_seed=seed))
    env = _make_env(seed, "mixed", env_config, scenario=scenario_list[0])
    obs = env.reset()
    hidden = None
    transitions: list[Transition] = []
    episode_start = True
    scenario_index = 0
    last_action: int | None = None
    steps_since_replan = max_replan_interval
    for _ in range(config.rollout_steps):
        vector, active_context, risk, full_vector = _policy_input_bundle(
            env, obs, model, fusion=fusion, device=device, trigger_threshold=trigger_threshold,
        )
        should_replan, replan_reason, trigger_conditions = _trigger_decision(
            obs, fusion=fusion, risk_active=active_context, last_action=last_action,
            steps_since_replan=steps_since_replan, max_replan_interval=max_replan_interval,
        )
        if should_replan:
            decision_vector = full_vector if fusion == "triggered" else vector
            action, log_prob, value, hidden = _act(policy, decision_vector, obs["mask"], hidden, device, deterministic=False)
            vector = decision_vector
            actor_decision = True
        else:
            # A triggered policy still owns a context-sized critic input while
            # it continues an accepted command.  The gated vector intentionally
            # omits model context when risk is below threshold, but passing that
            # shorter vector to value_only would make the context projection see
            # a zero-width tensor.  Reuse the already computed context from this
            # snapshot; do not run another model inference or submit a command.
            value_vector = full_vector if policy.context_dim else vector
            value_tensor, hidden = policy.value_only(
                torch.tensor(value_vector, dtype=torch.float32, device=device)[None, :], hidden,
            )
            action = int(last_action)  # guarded by _trigger_decision's safety condition
            log_prob = 0.0
            value = float(value_tensor.item())
            actor_decision = False
            vector = value_vector
        last_action = action
        steps_since_replan = 0 if should_replan else steps_since_replan + 1
        next_obs, reward, done, info = env.step(action, submit_command=should_replan)
        info = dict(info)
        info["ledger_public_observation"] = {
            "flat": np.asarray(obs["flat"], dtype=np.float32).copy(),
            "uavs": np.asarray(obs["uavs"]).copy(), "tasks": np.asarray(obs["tasks"]).copy(),
            "entity_ids": copy.deepcopy(obs.get("public_entity_ids", {})),
            "mask": np.asarray(obs["mask"], dtype=np.bool_).copy(),
            "version": obs.get("version"), "time": obs.get("time"),
        }
        info["ledger_next_public_observation"] = {
            "flat": np.asarray(next_obs["flat"], dtype=np.float32).copy(),
            "uavs": np.asarray(next_obs["uavs"]).copy(), "tasks": np.asarray(next_obs["tasks"]).copy(),
            "entity_ids": copy.deepcopy(next_obs.get("public_entity_ids", {})),
            "mask": np.asarray(next_obs["mask"], dtype=np.bool_).copy(),
            "version": next_obs.get("version"), "time": next_obs.get("time"),
        }
        info["ledger_tape_id"] = env.scenario.tape_id
        info["ledger_episode_id"] = env._episode_id
        info["ledger_episode_index"] = scenario_index
        info["ledger_raw_reward"] = float(reward)
        info["replan"] = bool(should_replan)
        info["replan_reason"] = replan_reason
        info["trigger_conditions"] = trigger_conditions
        info["actor_decision"] = actor_decision
        info["risk"] = float(risk)
        terminated = bool(info.get("terminated", done))
        truncated = bool(info.get("truncated", False))
        if not done and _ == config.rollout_steps - 1:
            # A rollout boundary is a bootstrap boundary, not an episode
            # termination.  The next-state value is retained for GAE.
            truncated = True
        if terminated:
            next_value = 0.0
        else:
            next_vector, _, _, next_full_vector = _policy_input_bundle(
                env, next_obs, model, fusion=fusion, device=device,
                trigger_threshold=trigger_threshold,
            )
            next_value_vector = next_full_vector if policy.context_dim else next_vector
            next_value_tensor, _ = policy.value_only(
                torch.tensor(next_value_vector, dtype=torch.float32, device=device)[None, :], hidden,
            )
            next_value = float(next_value_tensor.item())
        transitions.append(Transition(
            obs=vector,
            context=vector[policy.base_obs_dim:] if policy.context_dim else np.zeros(0, dtype=np.float32),
            mask=obs["mask"].copy(), action=action, log_prob=log_prob, value=value,
            reward=reward, done=bool(terminated or truncated), episode_start=episode_start,
            info=info, terminated=terminated, truncated=truncated, next_value=next_value,
            actor_decision=actor_decision,
        ))
        if done:
            scenario_index = (scenario_index + 1) % len(scenario_list)
            env = _make_env(scenario_list[scenario_index].seed, "mixed", env_config, scenario=scenario_list[scenario_index])
            obs = env.reset()
            hidden = None
            episode_start = True
            last_action = None
            steps_since_replan = max_replan_interval
        else:
            obs = next_obs
            episode_start = False
    return transitions


def _gae(transitions: list[Transition], gamma: float, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    rewards = np.asarray([t.reward for t in transitions], dtype=np.float32)
    values = np.asarray([t.value for t in transitions], dtype=np.float32)
    terminated = np.asarray([t.terminated for t in transitions], dtype=np.float32)
    boundaries = np.asarray([t.terminated or t.truncated for t in transitions], dtype=np.float32)
    next_values = np.asarray([t.next_value for t in transitions], dtype=np.float32)
    advantages = np.zeros_like(rewards)
    running = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        next_value = next_values[index]
        delta = rewards[index] + gamma * next_value * (1.0 - terminated[index]) - values[index]
        running = delta + gamma * gae_lambda * (1.0 - boundaries[index]) * running
        advantages[index] = running
    returns = advantages + values
    advantage_tensor = torch.tensor(advantages, dtype=torch.float32)
    return advantage_tensor, torch.tensor(returns, dtype=torch.float32)


def _evaluate_sequence(policy: M10ActorCritic, transitions: list[Transition], device: torch.device):
    log_probs, values, entropies = [], [], []
    hidden = None
    for transition in transitions:
        if transition.episode_start:
            hidden = None
        obs_tensor = torch.tensor(transition.obs, dtype=torch.float32, device=device)[None, :]
        mask_tensor = torch.tensor(transition.mask, dtype=torch.bool, device=device)[None, :]
        logits, value, hidden = policy(obs_tensor, hidden)
        dist = masked_distribution(logits, mask_tensor)
        action = torch.tensor([transition.action], dtype=torch.long, device=device)
        log_probs.append(dist.log_prob(action)[0])
        entropies.append(dist.entropy()[0])
        values.append(value[0])
    return torch.stack(log_probs), torch.stack(values), torch.stack(entropies)


def update_policy(policy: M10ActorCritic, transitions: list[Transition], *, ppo: PPOConfig,
                  device: torch.device) -> dict[str, float]:
    if not transitions:
        raise ValueError("empty PPO rollout")
    advantages, returns = _gae(transitions, ppo.gamma, ppo.gae_lambda)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    decision_indices = [index for index, transition in enumerate(transitions) if transition.actor_decision]
    if not decision_indices:
        raise ValueError("PPO rollout contains no actor decisions")
    decision_index_tensor = torch.tensor(decision_indices, dtype=torch.long, device=device)
    old_log_probs = torch.tensor([transitions[index].log_prob for index in decision_indices], dtype=torch.float32, device=device)
    advantages, returns = advantages.to(device), returns.to(device)
    last = {}
    for _ in range(ppo.update_epochs):
        new_log_probs, values, entropy = _evaluate_sequence(policy, transitions, device)
        decision_log_probs = new_log_probs[decision_index_tensor]
        decision_advantages = advantages[decision_index_tensor]
        ratio = (decision_log_probs - old_log_probs).exp()
        clipped = torch.clamp(ratio, 1.0 - ppo.clip_epsilon, 1.0 + ppo.clip_epsilon)
        policy_loss = -torch.min(ratio * decision_advantages, clipped * decision_advantages).mean()
        value_loss = torch.nn.functional.mse_loss(values, returns)
        entropy_mean = entropy[decision_index_tensor].mean()
        loss = policy_loss + ppo.value_weight * value_loss - ppo.entropy_weight * entropy_mean
        policy.optimizer.zero_grad(set_to_none=True)  # type: ignore[attr-defined]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), ppo.grad_clip)
        policy.optimizer.step()  # type: ignore[attr-defined]
        last = {"loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()), "entropy": float(entropy_mean.detach()), "approx_kl": float((old_log_probs - decision_log_probs).mean().detach()), "optimizer_steps": 1}
    last["optimizer_steps"] = ppo.update_epochs
    return last


def train_policy(*, variant: str, encoder: str, type_count: int, history: bool,
                  fusion: str, model: M10WorldModel | None, seed: int, steps: int,
                  env_config: M10Config, ppo_config: PPOConfig, device: str,
                  trigger_threshold: float = 0.5,
                  scenarios: Iterable[M10Scenario] | None = None,
                  max_replan_interval: int = 3,
                  capture_recovery_reference: bool = False,
                  ledger_callback: Any | None = None) -> tuple[M10ActorCritic, dict[str, Any]]:
    if fusion not in ("base", "world", "triggered"):
        raise ValueError("unsupported fusion")
    seed_everything(seed)
    device_obj = torch.device(device)
    context_dim = model.context_dim if model is not None and fusion != "base" else 0
    policy = M10ActorCritic(
        uav_count=env_config.uav_count, task_capacity=env_config.task_capacity,
        action_count=env_config.action_count, encoder=encoder, type_count=type_count,
        history=history, context_dim=context_dim,
        region_count=env_config.region_count, target_count=env_config.target_count,
        event_capacity=env_config.event_capacity, relation_width=env_config.relation_width,
    ).to(device_obj)
    policy.optimizer = torch.optim.Adam(policy.parameters(), lr=ppo_config.learning_rate)  # type: ignore[attr-defined]
    effective = PPOConfig(**{**asdict(ppo_config), "rollout_steps": min(ppo_config.rollout_steps, steps)})
    updates = []
    all_transitions: list[Transition] = []
    recovery_reference: dict[str, Any] | None = None
    total_steps = 0
    started = time.perf_counter()
    while total_steps < steps:
        effective = PPOConfig(**{**asdict(effective), "rollout_steps": min(effective.rollout_steps, steps - total_steps)})
        transitions = collect_rollout(policy, config=effective, env_config=env_config, seed=seed + total_steps, device=device_obj, model=model, fusion=fusion, trigger_threshold=trigger_threshold, scenarios=scenarios, max_replan_interval=max_replan_interval)
        if capture_recovery_reference:
            recovery_reference = {
                "policy_state_dict": copy.deepcopy(policy.state_dict()),
                "optimizer_state_dict": copy.deepcopy(policy.optimizer.state_dict()),  # type: ignore[attr-defined]
                "transitions": list(transitions),
                "effective_ppo_config": asdict(effective),
                "preupdate_environment_steps": total_steps,
            }
        updates.append(update_policy(policy, transitions, ppo=effective, device=device_obj))
        all_transitions.extend(transitions)
        rollout_start = total_steps
        total_steps += len(transitions)
        if ledger_callback is not None:
            ledger_callback(transitions, rollout_start, total_steps, len(updates), policy)
    elapsed = time.perf_counter() - started
    metadata = {
        "variant": variant, "encoder": encoder, "type_count": type_count, "history": history,
        "fusion": fusion, "seed": seed, "steps": total_steps,
        "rollout_updates": len(updates),
        "update_epochs": ppo_config.update_epochs,
        "optimizer_updates": int(sum(item.get("optimizer_steps", 0) for item in updates)),
        "actor_decisions": int(sum(t.actor_decision for t in all_transitions)),
        "continuation_steps": int(sum(not t.actor_decision for t in all_transitions)),
        "environment_steps": len(all_transitions),
        "elapsed_seconds": elapsed, "steps_per_second": total_steps / max(elapsed, 1e-9),
        "updates": updates, "device": str(device_obj), "env_config": asdict(env_config), "ppo_config": asdict(ppo_config),
        "trigger_threshold": trigger_threshold, "max_replan_interval": max_replan_interval,
        "replans": int(sum(bool(t.info.get("replan")) for t in all_transitions)) if all_transitions else 0,
        "replan_reasons": {
            reason: sum(1 for t in all_transitions if t.info.get("replan_reason") == reason)
            for reason in sorted({str(t.info.get("replan_reason")) for t in all_transitions})
        },
    }
    if capture_recovery_reference:
        metadata["_recovery_reference"] = recovery_reference
    return policy, metadata


def evaluate_policy(policy: M10ActorCritic, *, model: M10WorldModel | None, fusion: str,
                    env_config: M10Config, seeds: Iterable[int], device: str,
                    trigger_threshold: float = 0.5,
                    scenarios: dict[int, M10Scenario] | None = None,
                    max_replan_interval: int = 3) -> dict[str, Any]:
    device_obj = torch.device(device)
    policy.eval()
    records = []
    for seed in seeds:
        scenario = scenarios.get(seed) if scenarios else None
        env = _make_env(seed, "mixed", env_config, scenario=scenario)
        obs = env.reset()
        hidden = None
        done = False
        total_reward = 0.0
        steps = 0
        trigger_count = 0
        actor_calls = 0
        continuation_steps = 0
        replan_reasons: dict[str, int] = {}
        last_action: int | None = None
        steps_since_replan = max_replan_interval
        while not done and steps < int(env_config.horizon / env_config.decision_interval) + 2:
            vector, active, risk, full_vector = _policy_input_bundle(
                env, obs, model, fusion=fusion, device=device_obj, trigger_threshold=trigger_threshold,
            )
            should_replan, reason, _ = _trigger_decision(
                obs, fusion=fusion, risk_active=active, last_action=last_action,
                steps_since_replan=steps_since_replan, max_replan_interval=max_replan_interval,
            )
            if should_replan:
                decision_vector = full_vector if fusion == "triggered" else vector
                action, _, _, hidden = _act(policy, decision_vector, obs["mask"], hidden, device_obj, deterministic=True)
                vector = decision_vector
                actor_calls += 1
            else:
                # Keep the critic input compatible with the context-enabled
                # triggered policy while preserving the no-actor/no-submit
                # continuation semantics.
                value_vector = full_vector if policy.context_dim else vector
                _, hidden = policy.value_only(torch.tensor(value_vector, dtype=torch.float32, device=device_obj)[None, :], hidden)
                vector = value_vector
                action = int(last_action)
                continuation_steps += 1
            last_action = action
            steps_since_replan = 0 if should_replan else steps_since_replan + 1
            if should_replan and fusion == "triggered":
                trigger_count += 1
                replan_reasons[reason] = replan_reasons.get(reason, 0) + 1
            obs, reward, done, info = env.step(action, submit_command=should_replan)
            total_reward += reward
            steps += 1
        counts = info["counts"]
        records.append({"seed": seed, "return": total_reward, "steps": steps, "completed": counts["completed"], "expired": counts["expired"], "rejected": counts["rejected"], "trigger_count": trigger_count, "actor_calls": actor_calls, "continuation_steps": continuation_steps, "replan_reasons": replan_reasons, "energy_remaining": float(sum(info["energy"].values())), "task_states": info["tasks"]})
    keys = ("return", "completed", "expired", "rejected", "energy_remaining")
    summary = {key: {"mean": float(np.mean([row[key] for row in records])), "std": float(np.std([row[key] for row in records]))} for key in keys}
    return {"episodes": records, "summary": summary, "fusion": fusion}


@torch.no_grad()
def calibrate_trigger_threshold(model: M10WorldModel, rows: list[dict[str, Any]], *, device: str) -> dict[str, Any]:
    """Select a trigger threshold using validation rows only."""
    if not rows:
        raise ValueError("validation rows are required for trigger calibration")
    device_obj = torch.device(device)
    obs = torch.tensor(np.stack([row["obs"] for row in rows]), dtype=torch.float32, device=device_obj)
    mask = torch.full((len(rows), model.action_count), True, dtype=torch.bool, device=device_obj)
    actions = torch.tensor([row["action"] for row in rows], dtype=torch.long, device=device_obj)
    outputs = model(obs, actions)
    risks = torch.sigmoid(outputs["event_logits"]).max(dim=-1).values.cpu().numpy()
    targets = np.asarray([float(np.max(row["event_target"])) for row in rows], dtype=np.float32)
    candidates = [round(x, 2) for x in np.arange(0.1, 0.91, 0.05)]
    scores = []
    for threshold in candidates:
        pred = risks >= threshold
        tp = float(np.sum(pred & (targets > 0.5)))
        fp = float(np.sum(pred & (targets <= 0.5)))
        fn = float(np.sum((~pred) & (targets > 0.5)))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        scores.append({"threshold": threshold, "f1": f1, "precision": precision, "recall": recall})
    best = max(scores, key=lambda item: (item["f1"], -item["threshold"]))
    return {"selected_threshold": best["threshold"], "selection_unit": "validation_rows_only", "candidates": scores, "validation_rows": len(rows)}


def save_policy(path: Path, policy: M10ActorCritic, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": policy.state_dict(), "metadata": metadata,
               "optimizer_state_dict": policy.optimizer.state_dict() if hasattr(policy, "optimizer") else None,
               "recovery_state": {"steps": int(metadata.get("steps", 0)), "seed": metadata.get("seed"), "variant": metadata.get("variant")}}
    torch.save(payload, path)


__all__ = [
    "M10WorldModel", "M10ActorCritic", "M10Config", "PPOConfig",
    "EVENT_NAMES", "collect_world_dataset", "evaluate_world_model_rows", "train_world_model",
    "calibrate_trigger_threshold", "train_policy", "evaluate_policy", "save_policy",
]
