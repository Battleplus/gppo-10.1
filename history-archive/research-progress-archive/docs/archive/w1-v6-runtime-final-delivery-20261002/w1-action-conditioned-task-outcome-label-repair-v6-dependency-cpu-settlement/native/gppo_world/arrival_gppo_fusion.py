"""Candidate-wise arrival-consequence fusion for Graph-5 GPPO.

This module is deliberately separate from the legacy one-step context path in
``m10_training``.  The legacy path averages action-conditioned context before
the policy has selected an action; this adapter keeps one prediction row per
candidate and adds a trainable, initially zero, prior to the corresponding
action logit.  It never receives simulator state or a counterfactual label.

The adapter is an interface for the next bounded experiment.  It does not
train a model or run an environment by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .arrival_consequence_model import ArrivalModelConfig, Graph5ArrivalConsequenceModel, ArrivalPrediction
from .graph5 import GRAPH5_ACTION_COUNT, Graph5Snapshot, graph5_from_m10_observation
from .m10_training import M10ActorCritic


ARRIVAL_FUSION_FORMAT = "gppo-m10-arrival-candidate-fusion-v1"
ARRIVAL_PROTOCOL = "world-gppo-9.11-arrival/0.1.0"
FEATURE_NAMES = ("arrival_mean_norm", "arrival_logvar", "deadline_probability", "failure_probability")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CandidateFeatureContract:
    """Frozen preprocessing for per-candidate policy features."""

    feature_names: tuple[str, ...] = FEATURE_NAMES
    horizon_steps: int = 6
    action_count: int = GRAPH5_ACTION_COUNT
    illegal_fill: float = 0.0

    def __post_init__(self) -> None:
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("arrival fusion feature order is frozen")
        if self.horizon_steps <= 0 or self.action_count != GRAPH5_ACTION_COUNT:
            raise ValueError("arrival fusion requires the Graph-5/25-action contract")


@dataclass(frozen=True)
class CandidateConsequenceFeatures:
    """Dense and candidate-indexed predictions for one public snapshot."""

    actions: torch.Tensor
    values: torch.Tensor
    action_mask: torch.Tensor
    contract: CandidateFeatureContract

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.shape[1] != len(self.contract.feature_names):
            raise ValueError("candidate values must be [N,4]")
        if self.actions.ndim != 1 or self.actions.shape[0] != self.values.shape[0]:
            raise ValueError("candidate actions and values must have matching rows")
        if self.action_mask.shape != (self.contract.action_count,):
            raise ValueError("candidate action mask must contain 25 actions")
        if not torch.isfinite(self.values).all():
            raise FloatingPointError("candidate consequence features are non-finite")
        if len(set(self.actions.detach().cpu().tolist())) != int(self.actions.numel()):
            raise ValueError("candidate actions must be unique")

    def dense(self) -> torch.Tensor:
        """Return [25,4], filling illegal/unpredicted actions explicitly."""
        dense = torch.full(
            (self.contract.action_count, len(self.contract.feature_names)),
            self.contract.illegal_fill,
            dtype=self.values.dtype,
            device=self.values.device,
        )
        if self.actions.numel():
            dense[self.actions.long()] = self.values
        return dense


def _prediction_features(prediction: ArrivalPrediction, contract: CandidateFeatureContract) -> CandidateConsequenceFeatures:
    if prediction.actions.ndim != 1 or prediction.actions.numel() != prediction.arrival_mean.numel():
        raise ValueError("arrival prediction action rows are malformed")
    values = torch.stack((
        prediction.arrival_mean / float(contract.horizon_steps),
        prediction.arrival_logvar,
        torch.sigmoid(prediction.deadline_logits),
        torch.sigmoid(prediction.failure_logits),
    ), dim=-1)
    return CandidateConsequenceFeatures(
        actions=prediction.actions,
        values=values,
        action_mask=torch.zeros(contract.action_count, dtype=torch.bool, device=values.device),
        contract=contract,
    )


class FrozenArrivalConsequenceScorer:
    """Load and serve a frozen arrival model using public Graph-5 input only."""

    def __init__(self, model: Graph5ArrivalConsequenceModel, *, device: torch.device, contract: CandidateFeatureContract,
                 checkpoint_sha256: str | None = None, checkpoint_format: str = "in-memory") -> None:
        if model.format_version != Graph5ArrivalConsequenceModel.format_version:
            raise ValueError("unsupported arrival consequence model format")
        self.model = model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.contract = contract
        self.checkpoint_sha256 = checkpoint_sha256
        self.checkpoint_format = checkpoint_format

    @classmethod
    def from_checkpoint(cls, path: Path, *, device: str = "cpu", expected_sha256: str | None = None,
                        expected_protocol: str = ARRIVAL_PROTOCOL) -> "FrozenArrivalConsequenceScorer":
        actual_sha256 = sha256_file(path)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError(f"arrival checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}")
        device_obj = torch.device(device)
        checkpoint: dict[str, Any] = torch.load(path, map_location=device_obj, weights_only=False)
        if checkpoint.get("format") != "gppo-arrival-consequence-inference-v1":
            raise ValueError(
                "checkpoint is not an arrival-consequence inference artifact; "
                "the legacy joint-consequence checkpoint is not protocol-compatible"
            )
        identity = checkpoint.get("run_identity", {})
        if identity.get("protocol") != expected_protocol:
            raise ValueError(f"arrival checkpoint protocol mismatch: {identity.get('protocol')!r}")
        config_data = checkpoint.get("model_config", {})
        config = ArrivalModelConfig(
            hidden_dim=int(config_data.get("hidden_dim", 96)),
            horizon_steps=int(config_data.get("horizon_steps", 6)),
        )
        model = Graph5ArrivalConsequenceModel(config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return cls(
            model,
            device=device_obj,
            contract=CandidateFeatureContract(horizon_steps=config.horizon_steps),
            checkpoint_sha256=actual_sha256,
            checkpoint_format=str(checkpoint["format"]),
        )

    @torch.no_grad()
    def score_snapshot(self, graph: Graph5Snapshot) -> CandidateConsequenceFeatures:
        graph = graph.to(self.device)
        graph.validate()
        if not all(torch.isfinite(value).all() for value in graph.nodes.values()) or not torch.isfinite(graph.candidate_features).all():
            raise FloatingPointError("public graph contains non-finite values")
        prediction = self.model.predict_candidates(graph)
        features = _prediction_features(prediction, self.contract)
        return CandidateConsequenceFeatures(
            actions=features.actions,
            values=features.values,
            action_mask=graph.action_mask,
            contract=features.contract,
        )

    @torch.no_grad()
    def score_observation(self, observation: dict[str, Any]) -> CandidateConsequenceFeatures:
        """Score an environment public observation; no environment object is accepted."""
        graph = graph5_from_m10_observation(observation)
        return self.score_snapshot(graph)


class CandidateConsequencePrior(nn.Module):
    """Trainable candidate-specific logit prior with a neutral initialization."""

    def __init__(self, feature_dim: int = len(FEATURE_NAMES), *, initial_scale: float = 1.0) -> None:
        super().__init__()
        if feature_dim != len(FEATURE_NAMES):
            raise ValueError("arrival candidate feature width is frozen at four")
        self.projection = nn.Linear(feature_dim, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.initial_scale = float(initial_scale)

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        if candidate_features.ndim not in (2, 3) or candidate_features.shape[-1] != len(FEATURE_NAMES):
            raise ValueError("candidate_features must be [25,4] or [B,25,4]")
        if candidate_features.shape[-2] != GRAPH5_ACTION_COUNT:
            raise ValueError("candidate_features must preserve all 25 action slots")
        if not torch.isfinite(candidate_features).all():
            raise FloatingPointError("candidate features are non-finite")
        return self.projection(candidate_features).squeeze(-1) * self.initial_scale


class CandidateAwarePolicy(nn.Module):
    """A Graph-5 policy whose logits consume one feature row per candidate.

    ``base_policy`` remains the GPPO actor/critic.  The adapter is intentionally
    explicit about the extra tensor so callers cannot silently fall back to an
    action-averaged context.  A future PPO runner may optimize both the base
    policy and this small prior while the consequence scorer remains frozen.
    """

    def __init__(self, base_policy: M10ActorCritic, *, feature_dim: int = len(FEATURE_NAMES)) -> None:
        super().__init__()
        if base_policy.action_count != GRAPH5_ACTION_COUNT or base_policy.encoder_name != "graph" or base_policy.type_count != 5:
            raise ValueError("candidate fusion requires Graph-5/25-action base policy")
        self.base_policy = base_policy
        self.prior = CandidateConsequencePrior(feature_dim)
        self.action_count = base_policy.action_count
        self.base_obs_dim = base_policy.base_obs_dim
        self.context_dim = base_policy.context_dim

    def forward(self, obs: torch.Tensor, candidate_features: torch.Tensor, hidden: torch.Tensor | None = None):
        logits, value, next_hidden = self.base_policy(obs, hidden)
        prior = self.prior(candidate_features)
        if prior.ndim == 1:
            prior = prior[None, :]
        if logits.shape != prior.shape:
            raise ValueError(f"policy/candidate action shape mismatch: logits={tuple(logits.shape)}, prior={tuple(prior.shape)}")
        return logits + prior, value, next_hidden

    def value_only(self, obs: torch.Tensor, hidden: torch.Tensor | None = None):
        return self.base_policy.value_only(obs, hidden)


@torch.no_grad()
def choose_candidate_action(policy: CandidateAwarePolicy, scorer: FrozenArrivalConsequenceScorer,
                            observation: dict[str, Any], *, device: torch.device | str = "cpu",
                            hidden: torch.Tensor | None = None, deterministic: bool = True) -> tuple[int, torch.Tensor, CandidateConsequenceFeatures, torch.Tensor | None]:
    """Run one public-observation -> prediction -> policy -> legal-action step."""
    device_obj = torch.device(device)
    if policy.base_policy.action_count != GRAPH5_ACTION_COUNT:
        raise ValueError("candidate fusion requires a 25-action policy")
    features = scorer.score_observation(observation)
    public = torch.as_tensor(observation["flat"], dtype=torch.float32, device=device_obj)[None, :]
    dense = features.dense().to(device_obj)[None, :, :]
    logits, _, next_hidden = policy(public, dense, hidden)
    mask = torch.as_tensor(observation["mask"], dtype=torch.bool, device=device_obj)
    action = select_candidate_action(logits, mask, deterministic=deterministic)
    return int(action.item()), logits[0], features, next_hidden


def select_candidate_action(logits: torch.Tensor, action_mask: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
    """Select a legal action after candidate-specific logits are produced.

    Deterministic ties resolve to the lowest action index, matching the frozen
    GPPO mask convention.  The function never reads labels or simulator state.
    """
    if logits.ndim == 1:
        logits = logits[None, :]
    if action_mask.ndim == 1:
        action_mask = action_mask[None, :]
    if logits.shape != action_mask.shape or logits.shape[-1] != GRAPH5_ACTION_COUNT:
        raise ValueError("logits and action_mask must both be [B,25]")
    if not action_mask.bool().any(dim=-1).all():
        raise ValueError("each policy row needs at least one legal action")
    masked = logits.masked_fill(~action_mask.bool(), -torch.inf)
    if deterministic:
        return torch.argmax(masked, dim=-1)
    return torch.distributions.Categorical(logits=masked).sample()


def validate_arrival_policy_metadata(metadata: dict[str, Any], *, history: bool) -> None:
    """Reject policy artifacts trained under the old continuous-service contract."""
    env_config = metadata.get("env_config", {})
    if env_config.get("task_completion_mode") != "arrival_to_region":
        raise ValueError("policy artifact is not trained under arrival_to_region")
    if env_config.get("deadline_basis") != "physical_arrival":
        raise ValueError("policy artifact deadline basis is not physical_arrival")
    if int(metadata.get("action_count", env_config.get("action_count", -1))) != GRAPH5_ACTION_COUNT:
        raise ValueError("policy artifact is not Graph-5/25-action")
    if bool(metadata.get("history", False)) != bool(history):
        raise ValueError("policy history contract mismatch")


__all__ = [
    "ARRIVAL_FUSION_FORMAT", "ARRIVAL_PROTOCOL", "FEATURE_NAMES",
    "CandidateFeatureContract", "CandidateConsequenceFeatures",
    "FrozenArrivalConsequenceScorer", "CandidateConsequencePrior",
    "CandidateAwarePolicy", "choose_candidate_action", "select_candidate_action", "validate_arrival_policy_metadata",
]
