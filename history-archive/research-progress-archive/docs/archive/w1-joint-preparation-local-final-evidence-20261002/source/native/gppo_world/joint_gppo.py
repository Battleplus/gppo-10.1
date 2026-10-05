"""Joint preference-conditioned PPO, temporal world model, and event auxiliary.

This is a new arrival-protocol training path.  It intentionally does not load
the scalar-reward, static regret, or legacy one-step M10 world-model weights.
The Graph-5 M10 actor-critic encoder is reused, while policy, vector critic,
action-conditioned temporal model, and observed-event heads are new.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from .m10_training import M10ActorCritic, masked_distribution


EVENT_NAMES = (
    "urgent_task_first_public",
    "public_task_state_change",
    "uav_damage_first_public",
    "public_connectivity_change",
    "uav_crossed_low_energy_threshold",
)
EVENT_VERSION = "arrival-observed-events/1.0.0"
JOINT_PROTOCOL = "world-gppo-9.11-arrival-joint-pref-wm-event/0.2.0"
WORLD_FEATURE_DIM = 17  # predicted state projection 8 + vector reward 2 + consequences 2 + event probs 5
PREFERENCE_DIM = 2


@dataclass(frozen=True)
class JointTrainConfig:
    seed: int = 471101
    rollout_steps: int = 64
    policy_lr: float = 3e-4
    world_lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_weight: float = 0.01
    value_weight: float = 0.5
    grad_clip: float = 0.5
    preco_lambda: float = 0.1
    task_reward_scale: float = 0.5
    energy_reward_scale: float = 1.0
    urgent_slack_steps: float = 1.0
    low_energy_fraction: float = 0.10
    wm_minibatch_size: int = 8


def masked_normalized_preference(value: Sequence[float], *, device: torch.device) -> torch.Tensor:
    preference = torch.as_tensor(value, dtype=torch.float32, device=device)
    if preference.shape != (PREFERENCE_DIM,) or not torch.isfinite(preference).all() or bool((preference < 0).any()):
        raise ValueError("preference must be a finite nonnegative 2-vector")
    total = preference.sum()
    if float(total) <= 0:
        raise ValueError("preference must have positive mass")
    return preference / total


def _base_logits(base: M10ActorCritic, features: torch.Tensor, pair_messages: torch.Tensor) -> torch.Tensor:
    pair_logits = base.pair_actor(torch.cat((
        pair_messages,
        features[:, None, None, :].expand(-1, base.uav_count, base.task_capacity, -1),
    ), dim=-1)).squeeze(-1)
    return torch.cat((pair_logits.reshape(-1, base.uav_count * base.task_capacity), base.noop_actor(features)), dim=-1)


class JointGraphPreferencePolicy(nn.Module):
    """Graph-5/history policy with explicit preference and per-action WM input."""

    def __init__(self, config: Any, *, history: bool = True):
        super().__init__()
        if config.uav_count != 4 or config.task_capacity != 6 or config.action_count != 25:
            raise ValueError("joint arrival path requires the frozen Graph-5/25-action contract")
        self.base = M10ActorCritic(
            uav_count=config.uav_count, task_capacity=config.task_capacity,
            action_count=config.action_count, encoder="graph", type_count=5,
            history=history, context_dim=0, region_count=config.region_count,
            target_count=config.target_count, event_capacity=config.event_capacity,
            relation_width=config.relation_width,
        )
        self.preference_actor = nn.Sequential(nn.Linear(128 + PREFERENCE_DIM, 64), nn.Tanh(), nn.Linear(64, 25))
        self.candidate_actor = nn.Sequential(nn.Linear(WORLD_FEATURE_DIM, 32), nn.Tanh(), nn.Linear(32, 1))
        # This is V(s,p), not an action-value head. Candidate consequences
        # affect the actor, while the vector critic fits rollout return targets.
        self.vector_value = nn.Sequential(
            nn.Linear(128 + PREFERENCE_DIM, 128), nn.Tanh(),
            nn.Linear(128, PREFERENCE_DIM),
        )
        # The legacy scalar value head is not part of this preference protocol.
        for parameter in self.base.critic.parameters():
            parameter.requires_grad_(False)

    def encode(self, obs: torch.Tensor, hidden: torch.Tensor | None = None):
        return self.base._features(obs, hidden)

    def evaluate_encoded(self, features: torch.Tensor, pair_messages: torch.Tensor,
                         preference: torch.Tensor, candidate_features: torch.Tensor,
                         mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if preference.ndim == 1:
            preference = preference[None, :].expand(features.shape[0], -1)
        if candidate_features.shape != (features.shape[0], 25, WORLD_FEATURE_DIM):
            raise ValueError("candidate-specific world-model features must be [B,25,17]")
        logits = _base_logits(self.base, features, pair_messages)
        logits = logits + self.preference_actor(torch.cat((features, preference), dim=-1))
        logits = logits + self.candidate_actor(candidate_features).squeeze(-1)
        distribution = masked_distribution(logits, mask)
        probabilities = distribution.probs
        state_values = self.vector_value(torch.cat((features, preference), dim=-1))
        return {
            "logits": logits, "distribution": distribution,
            "probabilities": probabilities, "state_values": state_values,
            "values": state_values, "critic_values": state_values,
        }

    def evaluate_observation(self, obs: torch.Tensor, preference: torch.Tensor,
                             candidate_features: torch.Tensor, mask: torch.Tensor,
                             hidden: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        features, pair_messages, next_hidden = self.encode(obs, hidden)
        result = self.evaluate_encoded(features, pair_messages, preference, candidate_features, mask)
        result.update({"features": features, "pair_messages": pair_messages, "next_hidden": next_hidden})
        return result


class ActionConditionedTemporalWorldModel(nn.Module):
    """Shared action-conditional one-step predictor with recursive rollout API."""

    def __init__(self, *, action_count: int = 25, history_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.action_embedding = nn.Embedding(action_count, 32)
        self.relation_encoder = nn.Sequential(nn.Linear(4, 32), nn.Tanh())
        self.temporal = nn.GRUCell(history_dim + 64, hidden_dim)
        self.next_public_state = nn.Sequential(nn.Linear(hidden_dim, 128), nn.Tanh(), nn.Linear(128, history_dim))
        self.vector_reward = nn.Linear(hidden_dim, 2)
        self.task_consequence = nn.Linear(hidden_dim, 2)
        self.event_head = nn.Linear(hidden_dim, len(EVENT_NAMES))
        self.policy_feature = nn.Linear(hidden_dim, 8)
        self.action_count = action_count

    def forward(self, history: torch.Tensor, action: torch.Tensor, relation: torch.Tensor,
                hidden: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if history.ndim != 2 or history.shape[-1] != 128:
            raise ValueError("world-model history must be [B,128]")
        if relation.shape != (history.shape[0], 4):
            raise ValueError("world-model candidate relation must be [B,4]")
        if hidden is None:
            hidden = history.new_zeros((history.shape[0], 128))
        encoded_action = self.action_embedding(action.long())
        encoded_relation = self.relation_encoder(relation)
        latent = self.temporal(torch.cat((history, encoded_action, encoded_relation), dim=-1), hidden)
        return {
            "latent": latent,
            "next_state": self.next_public_state(latent),
            "vector_reward": self.vector_reward(latent),
            "task_consequence": self.task_consequence(latent),
            "event_logits": self.event_head(latent),
            "policy_feature": self.policy_feature(latent),
            "hidden": latent,
        }

    def predict_all_candidates(self, history: torch.Tensor, hidden: torch.Tensor | None,
                               observation: Mapping[str, Any], *, use_events: bool = True) -> tuple[torch.Tensor, dict[int, dict[str, torch.Tensor]]]:
        device = history.device
        mask = torch.as_tensor(observation["mask"], dtype=torch.bool, device=device).reshape(-1)
        relation_rows = torch.as_tensor(observation["graph"]["relations"], dtype=torch.float32, device=device).reshape(24, 4)
        actions = torch.arange(25, dtype=torch.long, device=device)
        relations = torch.cat((relation_rows, torch.zeros((1, 4), dtype=torch.float32, device=device)), dim=0)
        expanded_history = history.expand(25, -1)
        expanded_hidden = None if hidden is None else hidden.expand(25, -1)
        outputs = self.forward(expanded_history, actions, relations, expanded_hidden)
        features = torch.cat((
            outputs["policy_feature"], outputs["vector_reward"],
            outputs["task_consequence"], torch.sigmoid(outputs["event_logits"]),
        ), dim=-1)
        if not use_events:
            features = torch.cat((features[:, :12], torch.zeros_like(features[:, 12:])), dim=-1)
        features = torch.where(mask[:, None], features, torch.zeros_like(features))
        by_action = {int(i): {key: value[i:i + 1] for key, value in outputs.items()} for i in range(25)}
        return features[None, :, :], by_action

    def recursive_rollout(self, initial_history: torch.Tensor, actions: Sequence[int],
                          relations: Sequence[torch.Tensor] | None = None,
                          hidden: torch.Tensor | None = None) -> list[dict[str, torch.Tensor]]:
        """Multi-step prediction; each predicted state feeds the next step."""
        current = initial_history
        state = hidden
        outputs: list[dict[str, torch.Tensor]] = []
        for index, action in enumerate(actions):
            relation = current.new_zeros((current.shape[0], 4)) if relations is None else relations[index]
            result = self.forward(current, torch.full((current.shape[0],), int(action), device=current.device), relation, state)
            outputs.append(result)
            current, state = result["next_state"], result["hidden"]
        return outputs


def _field_valid(row: np.ndarray, field_index: int) -> bool:
    offset = field_index * 4
    return bool(row[offset + 1] > 0.5 and row[offset + 2] > 0.5)


def build_observed_event_labels(previous: Mapping[str, Any], following: Mapping[str, Any], *,
                                initial_energy: float, urgent_slack: float = 1.0,
                                low_energy_fraction: float = 0.10,
                                same_episode: bool = True) -> dict[str, Any]:
    """Identity-aligned labels from adjacent legal observations only.

    A positive means the control side newly saw an event/change within the
    next decision interval.  Simulator truth/event schedules are never read.
    """
    labels = np.zeros(len(EVENT_NAMES), dtype=np.float32)
    masks = np.zeros(len(EVENT_NAMES), dtype=np.bool_)
    reasons: dict[str, dict[str, int]] = {name: {"valid": 0, "identity_missing": 0, "field_invalid": 0, "episode_boundary": 0} for name in EVENT_NAMES}
    if not same_episode:
        for name in EVENT_NAMES:
            reasons[name]["episode_boundary"] = 1
        return {"labels": labels, "mask": masks, "reasons": reasons, "version": EVENT_VERSION}
    prev_ids = previous.get("public_entity_ids", {})
    next_ids = following.get("public_entity_ids", {})
    prev_task_ids = tuple(prev_ids.get("tasks", ()))
    next_task_ids = tuple(next_ids.get("tasks", ()))
    prev_task_rows = np.asarray(previous["tasks"], dtype=np.float32)
    next_task_rows = np.asarray(following["tasks"], dtype=np.float32)
    prev_tasks = {identity: prev_task_rows[index] for index, identity in enumerate(prev_task_ids) if index < len(prev_task_rows)}
    next_tasks = {identity: next_task_rows[index] for index, identity in enumerate(next_task_ids) if index < len(next_task_rows)}

    # A newly delivered, publicly valid task is marked urgent only when its
    # received deadline is no more than one fixed decision interval away.
    for identity, row in next_tasks.items():
        if not (_field_valid(row, 2) and _field_valid(row, 5)):
            reasons[EVENT_NAMES[0]]["field_invalid"] += 1
            continue
        masks[0] = True
        reasons[EVENT_NAMES[0]]["valid"] += 1
        newly_public = identity not in prev_tasks
        deadline = float(row[2 * 4])
        pending = float(row[5 * 4])
        if newly_public and pending > 0.5 and deadline - float(following["time"]) <= urgent_slack:
            labels[0] = 1.0

    shared_tasks = set(prev_tasks).intersection(next_tasks)
    if not prev_task_ids or not next_task_ids:
        reasons[EVENT_NAMES[1]]["identity_missing"] += 1
    for identity in shared_tasks:
        old, new = prev_tasks[identity], next_tasks[identity]
        fields_valid = all(_field_valid(old, field) and _field_valid(new, field) for field in (3, 5))
        if not fields_valid:
            reasons[EVENT_NAMES[1]]["field_invalid"] += 1
            continue
        masks[1] = True
        reasons[EVENT_NAMES[1]]["valid"] += 1
        if abs(float(new[3 * 4]) - float(old[3 * 4])) > 1e-6 or int(new[5 * 4] > 0.5) != int(old[5 * 4] > 0.5):
            labels[1] = 1.0

    prev_uav_ids = tuple(prev_ids.get("uavs", ()))
    next_uav_ids = tuple(next_ids.get("uavs", ()))
    prev_uavs = np.asarray(previous["uavs"], dtype=np.float32)
    next_uavs = np.asarray(following["uavs"], dtype=np.float32)
    old_uavs = {identity: prev_uavs[index] for index, identity in enumerate(prev_uav_ids) if index < len(prev_uavs)}
    new_uavs = {identity: next_uavs[index] for index, identity in enumerate(next_uav_ids) if index < len(next_uavs)}
    common_uavs = set(old_uavs).intersection(new_uavs)
    if not common_uavs:
        for idx in (2, 3, 4):
            reasons[EVENT_NAMES[idx]]["identity_missing"] += 1
    for identity in common_uavs:
        old, new = old_uavs[identity], new_uavs[identity]
        # `alive` and `connected` are public fields 3 and 4; `energy` is 2.
        if _field_valid(old, 3) and _field_valid(new, 3):
            masks[2] = True
            reasons[EVENT_NAMES[2]]["valid"] += 1
            if old[3 * 4] > 0.5 and new[3 * 4] <= 0.5:
                labels[2] = 1.0
        else:
            reasons[EVENT_NAMES[2]]["field_invalid"] += 1
        if _field_valid(old, 4) and _field_valid(new, 4):
            masks[3] = True
            reasons[EVENT_NAMES[3]]["valid"] += 1
            if int(old[4 * 4] > 0.5) != int(new[4 * 4] > 0.5):
                labels[3] = 1.0
        else:
            reasons[EVENT_NAMES[3]]["field_invalid"] += 1
        if _field_valid(old, 2) and _field_valid(new, 2):
            masks[4] = True
            reasons[EVENT_NAMES[4]]["valid"] += 1
            threshold = float(initial_energy) * low_energy_fraction
            if float(old[2 * 4]) > threshold >= float(new[2 * 4]):
                labels[4] = 1.0
        else:
            reasons[EVENT_NAMES[4]]["field_invalid"] += 1
    return {"labels": labels, "mask": masks, "reasons": reasons, "version": EVENT_VERSION}


def masked_event_bce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    valid = mask.bool()
    count = int(valid.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    return F.binary_cross_entropy_with_logits(logits[valid], labels[valid]), count


def preference_utility_targets(raw_return_estimate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map fixed-contract raw returns to nonnegative utility coordinates.

    The six-task net-arrival return is bounded by [-1, 1], and normalized
    energy consumption return by [-1, 0]. The affine maps are fixed by that
    reward contract; only this preference-geometry target is transformed.
    The returned mask marks components clipped because a bootstrapped estimate
    fell outside those known bounds. PPO advantages and critic targets remain
    in the original raw-return units (apart from positive policy-gradient
    scales), so this does not silently replace the environment reward.
    """
    if raw_return_estimate.shape[-1] != PREFERENCE_DIM:
        raise ValueError("preference return estimate must end in two objectives")
    task_raw = raw_return_estimate[..., 0]
    energy_raw = raw_return_estimate[..., 1]
    clipped = torch.stack((task_raw < -1.0, task_raw > 1.0, energy_raw < -1.0, energy_raw > 0.0), dim=-1)
    task = task_raw.clamp(-1.0, 1.0)
    energy = energy_raw.clamp(-1.0, 0.0)
    utility = torch.stack((0.5 * (task + 1.0), energy + 1.0), dim=-1)
    return utility, clipped


def preference_similarity(preference: torch.Tensor, utility_vector: torch.Tensor,
                          *, reduction: str = "mean") -> torch.Tensor:
    """PCRL paper's preference-ray similarity, applied to positive utilities.

    Psi(p,u) = -1/2 || (max_i u_i/p_i) p - u ||^2.
    Returns per-sample scores with reduction='none'; default is the batch mean.
    """
    if preference.ndim == 1:
        preference = preference.unsqueeze(0)
    if preference.shape != utility_vector.shape:
        try:
            preference = preference.expand_as(utility_vector)
        except RuntimeError as error:
            raise ValueError("preference and utility target shapes are incompatible") from error
    p = preference.clamp_min(1e-6)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    if not torch.isfinite(utility_vector).all() or bool((utility_vector < -1e-6).any()):
        raise ValueError("preference similarity expects finite nonnegative utility coordinates")
    scale = (utility_vector / p).amax(dim=-1, keepdim=True)
    per_sample = -0.5 * ((scale * p - utility_vector) ** 2).sum(dim=-1)
    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError("reduction must be 'none' or 'mean'")


def preference_similarity_coefficients(preference: torch.Tensor,
                                       raw_return_estimate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Detached dPsi/du coefficients from rollout return targets, never WM errors."""
    utility, clipped = preference_utility_targets(raw_return_estimate.detach())
    utility_leaf = utility.detach().requires_grad_(True)
    similarity_rows = preference_similarity(preference.detach(), utility_leaf, reduction="none")
    coefficients = torch.autograd.grad(similarity_rows.sum(), utility_leaf)[0].detach()
    return coefficients, utility.detach(), int(clipped.sum().item())


def _flat_gradient(gradient: torch.Tensor) -> torch.Tensor:
    return gradient.detach().reshape(-1)


def preco_logit_direction(objective_losses: Sequence[torch.Tensor], similarity: torch.Tensor,
                          logits: torch.Tensor, *, strength: float) -> tuple[torch.Tensor, float]:
    """Two-objective PreCo-inspired min-norm combination in policy-logit space."""
    if len(objective_losses) != 2:
        raise ValueError("this protocol has exactly two preference objectives")
    g0 = _flat_gradient(torch.autograd.grad(-objective_losses[0], logits, retain_graph=True)[0])
    g1 = _flat_gradient(torch.autograd.grad(-objective_losses[1], logits, retain_graph=True)[0])
    gs = _flat_gradient(torch.autograd.grad(similarity, logits, retain_graph=True)[0])
    direction0 = g0.reshape_as(logits)
    direction1 = g1.reshape_as(logits)
    similarity_direction = gs.reshape_as(logits)
    a = g0 - g1
    b = g1 + float(strength) * gs
    denominator = torch.dot(a, a)
    if float(denominator) <= 1e-20:
        weight0 = 0.5
    else:
        weight0 = float((-torch.dot(a, b) / denominator).clamp(0.0, 1.0).item())
    return weight0 * direction0 + (1.0 - weight0) * direction1 + float(strength) * similarity_direction, weight0


def finite_parameters(module: nn.Module) -> bool:
    return all(torch.isfinite(parameter).all().item() for parameter in module.parameters())


def finite_optimizer(optimizer: torch.optim.Optimizer) -> bool:
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all().item():
                return False
    return True


__all__ = [
    "EVENT_NAMES", "EVENT_VERSION", "JOINT_PROTOCOL", "WORLD_FEATURE_DIM", "PREFERENCE_DIM",
    "JointTrainConfig", "JointGraphPreferencePolicy", "ActionConditionedTemporalWorldModel",
    "build_observed_event_labels", "masked_event_bce", "masked_normalized_preference",
    "preference_utility_targets", "preference_similarity", "preference_similarity_coefficients",
    "preco_logit_direction", "finite_parameters", "finite_optimizer",
]
