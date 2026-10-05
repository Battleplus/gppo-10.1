"""Pinned-baseline PPO trainer with causally aligned latent sidecars.

Import this module only after ``GPPO-8.29@2a9bb9f`` is on ``sys.path``.  The
implementation intentionally reuses the baseline optimizer, GAE, action
submission, and checkpoint contracts; only context carriage is added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from .gppo_adapter import LatentAugmentedActorCritic, LatentContext

try:
    from ppo_allocation.random_event.trainer import (
        PPOTrainer,
        TrajectoryBuffer,
        _copy_graph_to_cpu,
        _explained_variance,
        _reset_env,
        _step_env,
        _versioned_step_env,
    )
except ImportError as exc:  # pragma: no cover - exercised by server integration
    _BASELINE_IMPORT_ERROR: ImportError | None = exc
    PPOTrainer = object  # type: ignore[assignment,misc]
    TrajectoryBuffer = object  # type: ignore[assignment,misc]
else:
    _BASELINE_IMPORT_ERROR = None


@dataclass
class LatentTrajectoryBuffer(TrajectoryBuffer):  # type: ignore[misc,valid-type]
    contexts: list[LatentContext] = field(default_factory=list)
    graph_versions: list[int] = field(default_factory=list)
    action_versions: list[int] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "LatentTrajectoryBuffer":
        return cls(
            [], [], [], [], [], [], [], [],
            contexts=[], graph_versions=[], action_versions=[]
        )

    def add(
        self,
        graph: Any,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        log_prob: float,
        value: float,
        next_value: float,
        *,
        context: LatentContext,
        graph_version: int,
        action_version: int,
    ) -> None:
        self.graphs.append(_copy_graph_to_cpu(graph))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))
        self.old_log_probs.append(float(log_prob))
        self.old_values.append(float(value))
        self.next_values.append(float(next_value))
        self.contexts.append(context)
        self.graph_versions.append(int(graph_version))
        self.action_versions.append(int(action_version))


class LatentPPOTrainer(PPOTrainer):  # type: ignore[misc,valid-type]
    """Baseline PPO with the exact rollout-time context replayed in updates."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _BASELINE_IMPORT_ERROR is not None:
            raise ImportError(
                "GPPO-8.29 baseline is required on sys.path for T-05 training"
            ) from _BASELINE_IMPORT_ERROR
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, LatentAugmentedActorCritic):
            raise TypeError("LatentPPOTrainer requires LatentAugmentedActorCritic")
        world_model = getattr(getattr(self.env, "runtime", None), "model", None)
        if world_model is not None:
            if world_model.training or any(
                parameter.requires_grad for parameter in world_model.parameters()
            ):
                raise ValueError("world model must be eval-mode and frozen before PPO construction")
            optimizer_ids = {
                id(parameter)
                for group in self.optimizer.param_groups
                for parameter in group["params"]
            }
            if optimizer_ids & {id(parameter) for parameter in world_model.parameters()}:
                raise ValueError("world-model parameters must not enter the PPO optimizer")

    def _context(self, graph: Any, action_version: int | None = None) -> LatentContext:
        assert isinstance(self.model, LatentAugmentedActorCritic)
        store = self.model.context_store
        if store is not None:
            return store.read(int(graph.graph_version), action_version=action_version)
        return LatentContext.zero(
            self.model.adapter_config.latent_dim,
            model_variant=self.model.model_variant or "GPPO",
            model_version="no-world-model",
            post_graph_version=int(graph.graph_version),
            post_action_version=int(action_version or 0),
            reason="world_model_disabled",
        )

    def collect_rollout(self, steps: int | None = None):
        steps = int(steps or self.config.rollout_steps)
        if steps <= 0:
            raise ValueError("rollout size must be positive")
        graph = self._ensure_current_graph()
        model = self.initialize(graph)
        assert isinstance(model, LatentAugmentedActorCritic)
        model.eval()
        buffer = LatentTrajectoryBuffer.empty()
        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        invalid_probabilities: list[float] = []
        gate_values: dict[str, list[float]] = {}
        latent_used = 0
        stale_retries = 0
        execution_rejections = 0

        collected = 0
        while collected < steps:
            decision = None
            if hasattr(self.env, "begin_decision"):
                decision = self.env.begin_decision()
                graph = decision.graph
            device_graph = graph.to(self.device)
            if not bool(device_graph.action_mask.any().item()):
                raise RuntimeError("graph action mask contains no legal action")
            policy_context = self._context(
                graph,
                action_version=None if decision is None else int(decision.action_version),
            )
            action, log_prob, value, diagnostics = model.act(
                device_graph,
                deterministic=False,
                context=policy_context,
                action_version=None if decision is None else int(decision.action_version),
            )
            if decision is not None:
                next_graph, reward, terminated, truncated, info = _versioned_step_env(
                    self.env, action, decision
                )
                if info.get("stale_decision", False):
                    stale_retries += 1
                    continue
                if info.get("execution_rejected", False):
                    execution_rejections += 1
                    # PPO's MDP action is the sampled command proposal; an
                    # execution-layer rejection/fail-closed NOOP is part of
                    # the environment transition caused by that proposal.
                    # Preserve proposal/log-prob/reward/next-state on-policy.
                    # Shadow did not accept an executed action, so the next
                    # context store read below fails closed to zero context.
            else:
                next_graph, reward, terminated, truncated, info = _step_env(self.env, action)
            with torch.no_grad():
                if terminated:
                    next_value = 0.0
                else:
                    next_action_version = int(getattr(self.env, "decision_version", 0))
                    next_context = self._context(next_graph, action_version=next_action_version)
                    _, next_value_tensor, _ = model(
                        next_graph.to(self.device),
                        context=next_context,
                        action_version=next_action_version,
                    )
                    next_value = float(next_value_tensor.item())
            buffer.add(
                graph,
                action,
                reward,
                terminated,
                truncated,
                log_prob,
                value,
                next_value,
                context=policy_context,
                graph_version=int(graph.graph_version),
                action_version=(
                    int(policy_context.post_action_version)
                    if decision is None
                    else int(decision.action_version)
                ),
            )
            latent_used += int(diagnostics.get("latent_adapter_used", False))
            invalid_probabilities.append(float(diagnostics["pre_mask_invalid_probability"]))
            for key, gate_mean in diagnostics.get("gate_mean", {}).items():
                gate_values.setdefault(key, []).append(float(gate_mean))

            self.total_steps += 1
            collected += 1
            self._episode_return += reward
            self._episode_length += 1
            if terminated or truncated:
                episode_returns.append(self._episode_return)
                episode_lengths.append(self._episode_length)
                self._episode_return = 0.0
                self._episode_length = 0
                graph, _ = _reset_env(self.env)
            else:
                graph = next_graph

        self._current_graph = graph
        buffer.compute_gae(self.config.gamma, self.config.gae_lambda)
        rollout_stats: dict[str, Any] = {
            "rollout_steps": len(buffer),
            "episodes_completed": len(episode_returns),
            "episode_return_mean": float(np.mean(episode_returns)) if episode_returns else float("nan"),
            "episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
            "pre_mask_invalid_probability": float(np.mean(invalid_probabilities)),
            "gate_means": {key: float(np.mean(values)) for key, values in sorted(gate_values.items())},
            "stale_retries": stale_retries,
            "execution_rejections": execution_rejections,
            "latent_adapter_use_count": latent_used,
            "latent_adapter_use_rate": latent_used / max(len(buffer), 1),
        }
        rollout_stats["gate_mean"] = (
            float(np.mean(list(rollout_stats["gate_means"].values())))
            if rollout_stats["gate_means"]
            else float("nan")
        )
        return buffer, rollout_stats

    def update(self, buffer: LatentTrajectoryBuffer) -> dict[str, Any]:
        if not (
            len(buffer.contexts)
            == len(buffer.graph_versions)
            == len(buffer.action_versions)
            == len(buffer)
        ):
            raise ValueError("every PPO transition must carry context and both decision versions")
        for index, (graph, context, graph_version, action_version) in enumerate(
            zip(
                buffer.graphs,
                buffer.contexts,
                buffer.graph_versions,
                buffer.action_versions,
            )
        ):
            if int(graph.graph_version) != graph_version:
                raise ValueError(f"buffer graph_version mismatch at transition {index}")
            if context.valid and (
                context.post_graph_version != graph_version
                or context.post_action_version != action_version
            ):
                raise ValueError(f"buffer latent version mismatch at transition {index}")
        if buffer.advantages is None or buffer.returns is None:
            buffer.compute_gae(self.config.gamma, self.config.gae_lambda)
        if len(buffer) == 0:
            raise ValueError("cannot update from an empty trajectory")
        model = self.initialize(buffer.graphs[0])
        assert isinstance(model, LatentAugmentedActorCritic)
        assert self.optimizer is not None
        model.train()

        advantages = buffer.advantages.to(self.device)
        returns = buffer.returns.to(self.device)
        old_log_probs = torch.tensor(buffer.old_log_probs, dtype=torch.float32, device=self.device)
        if self.config.normalize_advantages and len(buffer) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        metric_lists: dict[str, list[float]] = {
            key: []
            for key in (
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "pre_mask_invalid_probability",
                "grad_norm",
            )
        }
        gate_lists: dict[str, list[float]] = {}
        stop_for_kl = False
        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(len(buffer), device=self.device)
            for start in range(0, len(buffer), self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                log_probs: list[torch.Tensor] = []
                entropies: list[torch.Tensor] = []
                values: list[torch.Tensor] = []
                invalid_masses: list[torch.Tensor] = []
                minibatch_gates: dict[str, list[torch.Tensor]] = {}
                for index_tensor in indices:
                    index = int(index_tensor.item())
                    graph = buffer.graphs[index].to(self.device)
                    log_prob, entropy, value, diagnostics = model.evaluate_action(
                        graph,
                        buffer.actions[index],
                        context=buffer.contexts[index],
                        action_version=buffer.action_versions[index],
                    )
                    log_probs.append(log_prob)
                    entropies.append(entropy)
                    values.append(value)
                    invalid_masses.append(diagnostics["pre_mask_invalid_probability"])
                    for key, gate in diagnostics.get("gates", {}).items():
                        minibatch_gates.setdefault(key, []).append(gate.mean())
                new_log_probs = torch.stack(log_probs)
                entropy_tensor = torch.stack(entropies)
                new_values = torch.stack(values).reshape(-1)
                log_ratio = new_log_probs - old_log_probs[indices]
                ratio = log_ratio.exp()
                minibatch_advantages = advantages[indices]
                policy_loss = torch.maximum(
                    -minibatch_advantages * ratio,
                    -minibatch_advantages
                    * ratio.clamp(1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef),
                ).mean()
                value_loss = 0.5 * (new_values - returns[indices]).pow(2).mean()
                entropy = entropy_tensor.mean()
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_coef).float().mean()
                for key, value in {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                    "pre_mask_invalid_probability": torch.stack(invalid_masses).mean(),
                    "grad_norm": torch.as_tensor(grad_norm),
                }.items():
                    metric_lists[key].append(float(value.detach().cpu()))
                for key, values_for_key in minibatch_gates.items():
                    gate_lists.setdefault(key, []).append(
                        float(torch.stack(values_for_key).mean().detach().cpu())
                    )
                if self.config.target_kl is not None and float(approx_kl) > self.config.target_kl:
                    stop_for_kl = True
                    break
            if stop_for_kl:
                break

        with torch.no_grad():
            final_values = np.asarray(
                [
                    float(
                        model(
                            graph.to(self.device),
                            context=buffer.contexts[index],
                            action_version=buffer.action_versions[index],
                        )[1].cpu()
                    )
                    for index, graph in enumerate(buffer.graphs)
                ],
                dtype=np.float32,
            )
        stats: dict[str, Any] = {
            key: float(np.mean(values)) for key, values in metric_lists.items()
        }
        stats["explained_variance"] = _explained_variance(final_values, buffer.returns.numpy())
        stats["gate_means"] = {
            key: float(np.mean(values)) for key, values in sorted(gate_lists.items())
        }
        stats["gate_mean"] = (
            float(np.mean(list(stats["gate_means"].values())))
            if stats["gate_means"]
            else float("nan")
        )
        stats["early_stop_kl"] = stop_for_kl
        return stats

    @classmethod
    def load(
        cls,
        path: str,
        env: Any,
        device: str | None = None,
        *,
        context_store: Any | None = None,
    ):
        """Restore adapter or legacy GPPO checkpoints with fail-closed context."""

        from dataclasses import asdict  # noqa: PLC0415
        from ppo_allocation.random_event.models import (  # noqa: PLC0415
            GraphActorCritic,
            GraphModelConfig,
        )
        from ppo_allocation.random_event.trainer import PPOConfig  # noqa: PLC0415

        def base_factory(spec: dict[str, Any]) -> GraphActorCritic:
            edge_dims = {
                tuple(key.split("__")): int(value)
                for key, value in spec["edge_dims"].items()
            }
            return GraphActorCritic(
                spec["node_dims"],
                GraphModelConfig(**spec["config"]),
                edge_dims=edge_dims,
            )

        model, metadata = LatentAugmentedActorCritic.load(
            path,
            base_factory=base_factory,
            legacy_loader=GraphActorCritic.load,
            map_location=device or "cpu",
        )
        if context_store is not None:
            if model.model_variant not in (None, "GPPO") and (
                context_store.model_variant != model.model_variant
                or (
                    model.model_version is not None
                    and context_store.model_version != model.model_version
                )
            ):
                raise ValueError("checkpoint and context store model identity mismatch")
            model.context_store = context_store
            context_store.reset()
        config_values = dict(metadata.get("ppo_config", {}))
        if device is not None:
            config_values["device"] = device
        config = PPOConfig(**config_values) if config_values else PPOConfig(device=device or "cpu")
        trainer = cls(
            env=env,
            variant=metadata.get("variant", "GPPO-Adaptive"),
            config=config,
            model=model,
        )
        trainer.total_steps = int(metadata.get("total_steps", 0))
        trainer.update_count = int(metadata.get("update_count", 0))
        trainer.history = list(metadata.get("history", []))
        optimizer_state = metadata.get("optimizer_state")
        if optimizer_state is not None and not metadata.get("loaded_from_legacy_gppo"):
            assert trainer.optimizer is not None
            trainer.optimizer.load_state_dict(optimizer_state)
        metadata = {
            **dict(metadata),
            "context_restored": False,
            "context_reset_required": context_store is None and model.enabled,
            "effective_device": str(config.device),
            "ppo_config": asdict(config),
        }
        return trainer, metadata


__all__ = ["LatentPPOTrainer", "LatentTrajectoryBuffer"]
