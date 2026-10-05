"""T-05 frozen-latent adapter for the existing GPPO actor/critic.

The adapter is deliberately outside the world model.  It consumes only the
accepted Shadow result from the previous transition and adds residuals to the
existing GPPO logits/value.  A disabled, invalid, stale, or all-zero context
returns the original GPPO tensors without performing adapter arithmetic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import threading
from typing import Any, Callable, Mapping

import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class LatentAdapterConfig:
    latent_dim: int = 88
    adapter_dim: int = 32
    num_actions: int = 17

    def __post_init__(self) -> None:
        if self.latent_dim <= 0 or self.adapter_dim <= 0 or self.num_actions <= 1:
            raise ValueError("latent_dim/adapter_dim must be positive and num_actions > 1")


@dataclass(frozen=True)
class LatentContext:
    """Causally bound Shadow latent available to one later GPPO decision."""

    latent: tuple[float, ...]
    valid: bool
    model_variant: str
    model_version: str
    post_graph_version: int
    post_action_version: int
    source_episode_id: str
    source_step: int
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.model_variant or not self.model_version:
            raise ValueError("model_variant and model_version must be non-empty")
        if self.post_graph_version < 0 or self.post_action_version < 0:
            raise ValueError("post versions must be non-negative")
        if self.valid:
            if not self.source_episode_id or self.source_step < 0:
                raise ValueError("valid context requires a source transition")
            if self.fallback_reason is not None:
                raise ValueError("valid context cannot carry fallback_reason")
            if not self.latent or not all(math.isfinite(value) for value in self.latent):
                raise ValueError("valid latent must be finite and non-empty")

    @classmethod
    def zero(
        cls,
        latent_dim: int,
        *,
        model_variant: str,
        model_version: str,
        post_graph_version: int,
        post_action_version: int,
        reason: str,
    ) -> "LatentContext":
        return cls(
            latent=(0.0,) * int(latent_dim),
            valid=False,
            model_variant=model_variant,
            model_version=model_version,
            post_graph_version=int(post_graph_version),
            post_action_version=int(post_action_version),
            source_episode_id="fallback",
            source_step=0,
            fallback_reason=str(reason),
        )

    def usable_for(
        self,
        *,
        graph_version: int,
        action_version: int | None,
        latent_dim: int,
        model_variant: str | None,
        model_version: str | None = None,
    ) -> bool:
        if not self.valid or len(self.latent) != latent_dim:
            return False
        if self.post_graph_version != int(graph_version):
            return False
        if action_version is not None and self.post_action_version != int(action_version):
            return False
        if model_variant is not None and self.model_variant != model_variant:
            return False
        if model_version is not None and self.model_version != model_version:
            return False
        return any(value != 0.0 for value in self.latent)


class LatentContextStore:
    """Thread-safe, read-only-to-policy handoff with fail-closed version checks."""

    def __init__(self, config: LatentAdapterConfig, *, model_variant: str, model_version: str):
        self.config = config
        self.model_variant = str(model_variant)
        self.model_version = str(model_version)
        self._lock = threading.Lock()
        self._current = LatentContext.zero(
            config.latent_dim,
            model_variant=self.model_variant,
            model_version=self.model_version,
            post_graph_version=0,
            post_action_version=0,
            reason="episode_reset",
        )
        self._decision_versions: tuple[int, int] | None = None
        self._counters = {
            "published_valid": 0,
            "published_fallback": 0,
            "served_valid": 0,
            "served_fallback": 0,
            "stale_context": 0,
        }

    @property
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def reset(self, *, graph_version: int = 0, action_version: int = 0) -> None:
        with self._lock:
            self._current = LatentContext.zero(
                self.config.latent_dim,
                model_variant=self.model_variant,
                model_version=self.model_version,
                post_graph_version=graph_version,
                post_action_version=action_version,
                reason="episode_reset",
            )
            self._decision_versions = None

    def prepare_decision(self, graph_version: int, action_version: int) -> None:
        with self._lock:
            self._decision_versions = (int(graph_version), int(action_version))

    def publish(self, context: LatentContext) -> None:
        if context.model_variant != self.model_variant or context.model_version != self.model_version:
            raise ValueError("context model identity does not match the store")
        if len(context.latent) != self.config.latent_dim:
            raise ValueError("context latent dimension does not match adapter")
        with self._lock:
            self._current = context
            self._decision_versions = None
            self._counters["published_valid" if context.valid else "published_fallback"] += 1

    def read(self, graph_version: int, action_version: int | None = None) -> LatentContext:
        with self._lock:
            effective_action_version = action_version
            if self._decision_versions is not None:
                prepared_graph, prepared_action = self._decision_versions
                if prepared_graph != int(graph_version):
                    self._counters["stale_context"] += 1
                    self._counters["served_fallback"] += 1
                    return LatentContext.zero(
                        self.config.latent_dim,
                        model_variant=self.model_variant,
                        model_version=self.model_version,
                        post_graph_version=int(graph_version),
                        post_action_version=prepared_action,
                        reason="prepared_graph_version_mismatch",
                    )
                effective_action_version = prepared_action
            if self._current.usable_for(
                graph_version=int(graph_version),
                action_version=effective_action_version,
                latent_dim=self.config.latent_dim,
                model_variant=self.model_variant,
                model_version=self.model_version,
            ):
                self._counters["served_valid"] += 1
                return self._current
            if self._current.valid:
                self._counters["stale_context"] += 1
            self._counters["served_fallback"] += 1
            return LatentContext.zero(
                self.config.latent_dim,
                model_variant=self.model_variant,
                model_version=self.model_version,
                post_graph_version=int(graph_version),
                post_action_version=int(effective_action_version or 0),
                reason=self._current.fallback_reason or "invalid_or_stale_context",
            )


def context_from_shadow(result: Any, *, model_variant: str, latent_dim: int) -> LatentContext:
    """Convert a T-04 result without exposing its prediction/event heads."""

    latent = tuple(float(value) for value in result.latent)
    if len(latent) != int(latent_dim):
        raise ValueError("Shadow latent dimension does not match the policy adapter")
    return LatentContext(
        latent=latent,
        valid=bool(result.valid),
        model_variant=str(model_variant),
        model_version=str(result.model_version),
        post_graph_version=int(result.post_graph_version),
        post_action_version=int(result.post_action_version),
        source_episode_id=str(result.episode_id),
        source_step=int(result.step),
        fallback_reason=None if result.valid else str(result.fallback_reason),
    )


class ResidualLatentAdapter(nn.Module):
    """Bias-free residual heads; zero latent is mathematically zero residual."""

    def __init__(self, config: LatentAdapterConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.adapter_dim, bias=False),
            nn.LayerNorm(config.adapter_dim, elementwise_affine=False),
            nn.SiLU(),
        )
        self.actor = nn.Linear(config.adapter_dim, config.num_actions, bias=False)
        self.critic = nn.Linear(config.adapter_dim, 1, bias=False)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.critic.weight)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(latent)
        return self.actor(encoded), self.critic(encoded).squeeze(-1)


class LatentAugmentedActorCritic(nn.Module):
    """Drop-in GPPO wrapper retaining the original action and safety contract."""

    format_version = "gppo-frozen-latent-adapter-v1"

    def __init__(
        self,
        base_model: nn.Module,
        config: LatentAdapterConfig | None = None,
        *,
        context_store: LatentContextStore | None = None,
        enabled: bool = True,
        model_variant: str | None = None,
        model_version: str | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.adapter_config = config or LatentAdapterConfig()
        self.adapter = ResidualLatentAdapter(self.adapter_config)
        self.context_store = context_store
        self.enabled = bool(enabled)
        self.model_variant = model_variant
        self.model_version = (
            model_version
            if model_version is not None
            else (context_store.model_version if context_store is not None else None)
        )

    def _resolve_context(self, graph: Any, context: LatentContext | None) -> LatentContext | None:
        if context is not None:
            return context
        if self.context_store is None:
            return None
        return self.context_store.read(int(graph.graph_version))

    def _use_context(
        self,
        graph: Any,
        context: LatentContext | None,
        action_version: int | None,
    ) -> bool:
        return bool(
            self.enabled
            and context is not None
            and context.usable_for(
                graph_version=int(graph.graph_version),
                action_version=action_version,
                latent_dim=self.adapter_config.latent_dim,
                model_variant=self.model_variant,
                model_version=self.model_version,
            )
        )

    def forward(
        self,
        graph: Any,
        context: LatentContext | None = None,
        action_version: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        base_logits, base_value, base_diagnostics = self.base_model(graph)
        selected = self._resolve_context(graph, context)
        if not self._use_context(graph, selected, action_version):
            return base_logits, base_value, base_diagnostics
        assert selected is not None
        latent = torch.as_tensor(selected.latent, dtype=base_logits.dtype, device=base_logits.device)
        actor_residual, critic_residual = self.adapter(latent)
        if actor_residual.shape != base_logits.shape:
            raise ValueError("adapter action count does not match GPPO action count")
        # Apply residuals only to legal candidates.  Invalid raw logits remain
        # the baseline raw values for an honest pre-mask diagnostic, and their
        # final masked logits remain bit-identical to the baseline output.
        base_raw_logits = base_diagnostics.get("raw_logits", base_logits)
        augmented_raw_logits = torch.where(
            graph.action_mask, base_raw_logits + actor_residual, base_raw_logits
        )
        logits = augmented_raw_logits.masked_fill(
            ~graph.action_mask, torch.finfo(augmented_raw_logits.dtype).min
        )
        value = base_value + critic_residual
        diagnostics = dict(base_diagnostics)
        diagnostics.update(
            {
                "raw_logits": augmented_raw_logits,
                "pre_mask_invalid_probability": torch.softmax(
                    augmented_raw_logits, dim=-1
                )[~graph.action_mask].sum(),
                "latent_adapter_used": True,
                "latent_actor_residual_norm": actor_residual.norm(),
                "latent_critic_residual": critic_residual,
                "latent_model_variant": selected.model_variant,
                "latent_model_version": selected.model_version,
            }
        )
        return logits, value, diagnostics

    def distribution(
        self,
        graph: Any,
        context: LatentContext | None = None,
        action_version: int | None = None,
    ):
        logits, value, diagnostics = self(
            graph, context=context, action_version=action_version
        )
        return Categorical(logits=logits), value, diagnostics

    @torch.no_grad()
    def act(
        self,
        graph: Any,
        deterministic: bool = False,
        context: LatentContext | None = None,
        action_version: int | None = None,
    ) -> tuple[int, float, float, dict[str, Any]]:
        distribution, value, diagnostics = self.distribution(
            graph, context=context, action_version=action_version
        )
        action = torch.argmax(distribution.logits) if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action)
        clean = {
            "pre_mask_invalid_probability": float(
                diagnostics["pre_mask_invalid_probability"].detach().cpu()
            ),
            "gate_mean": {
                name: float(gate.mean().detach().cpu())
                for name, gate in diagnostics.get("gates", {}).items()
            },
            "latent_adapter_used": bool(diagnostics.get("latent_adapter_used", False)),
        }
        return int(action.item()), float(log_prob.item()), float(value.item()), clean

    def evaluate_action(
        self,
        graph: Any,
        action: torch.Tensor | int,
        context: LatentContext | None = None,
        action_version: int | None = None,
    ):
        distribution, value, diagnostics = self.distribution(
            graph, context=context, action_version=action_version
        )
        action_tensor = torch.as_tensor(action, dtype=torch.long, device=value.device)
        return distribution.log_prob(action_tensor), distribution.entropy(), value, diagnostics

    def adapter_parameters(self):
        return self.adapter.parameters()

    def save(self, path: str | Path, extra: Mapping[str, Any] | None = None) -> None:
        base_spec: dict[str, Any] = {}
        if all(hasattr(self.base_model, name) for name in ("node_dims", "edge_dims", "config")):
            base_spec = {
                "node_dims": dict(self.base_model.node_dims),
                "edge_dims": {"__".join(key): value for key, value in self.base_model.edge_dims.items()},
                "config": asdict(self.base_model.config),
            }
        payload = {
            "format": self.format_version,
            "adapter_config": asdict(self.adapter_config),
            "enabled": self.enabled,
            "model_variant": self.model_variant,
            "model_version": self.model_version,
            "base_spec": base_spec,
            "state_dict": self.state_dict(),
            "extra": dict(extra or {}),
        }
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        base_factory: Callable[[Mapping[str, Any]], nn.Module],
        legacy_loader: Callable[..., tuple[nn.Module, Mapping[str, Any]]] | None = None,
        map_location: str | torch.device = "cpu",
    ) -> tuple["LatentAugmentedActorCritic", dict[str, Any]]:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if payload.get("format") == "random-event-gppo-v1":
            if legacy_loader is None:
                raise ValueError("legacy_loader is required for a GPPO v1 checkpoint")
            base, metadata = legacy_loader(path, map_location=map_location)
            # Baseline has 16 edge actions plus one NOOP; keep the explicit
            # default contract instead of deriving topology from trainable weights.
            config = LatentAdapterConfig(num_actions=17)
            return cls(
                base,
                config,
                enabled=False,
                model_variant="GPPO",
                model_version="no-world-model",
            ), {
                **dict(metadata),
                "loaded_from_legacy_gppo": True,
                "lossless_fallback": True,
            }
        if payload.get("format") != cls.format_version:
            raise ValueError(f"unsupported policy checkpoint format: {payload.get('format')}")
        base = base_factory(payload["base_spec"])
        model = cls(
            base,
            LatentAdapterConfig(**payload["adapter_config"]),
            enabled=bool(payload["enabled"]),
            model_variant=payload.get("model_variant"),
            model_version=payload.get("model_version"),
        )
        model.load_state_dict(payload["state_dict"])
        return model, dict(payload.get("extra", {}))


def freeze_world_model(model: nn.Module) -> nn.Module:
    """Freeze and audit the WM before it is connected to a rollout hook."""

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):  # pragma: no cover
        raise RuntimeError("world-model freeze failed")
    return model


__all__ = [
    "LatentAdapterConfig",
    "LatentAugmentedActorCritic",
    "LatentContext",
    "LatentContextStore",
    "ResidualLatentAdapter",
    "context_from_shadow",
    "freeze_world_model",
]
