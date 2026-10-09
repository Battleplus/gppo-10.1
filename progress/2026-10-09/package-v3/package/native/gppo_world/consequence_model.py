"""Action-conditioned task-consequence prediction for the 9.11 study.

This module is intentionally separate from the frozen M-10 world model.  It
keeps candidate actions distinct, exposes uncertainty, and does not change
the historical policy, reward, threshold, or execution contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import GraphSnapshot
from .graph5 import Graph5Snapshot, GRAPH5_ACTION_COUNT, GRAPH5_GLOBAL_DIM
from .model import GraphWorldModel


def graph_to_device(graph: GraphSnapshot, device: torch.device | str) -> GraphSnapshot:
    """Copy every graph component to one device before model execution."""
    return graph.to(torch.device(device))


CONTINUOUS_TARGETS = ("travel_time", "service_progress", "energy_delta")
TARGET_NAMES = CONTINUOUS_TARGETS + ("deadline_risk",)


@dataclass(frozen=True)
class ConsequenceModelConfig:
    """Frozen design knobs for the first consequence-prediction pilot."""

    hidden_dim: int = 96
    history_dim: int = 0
    horizon_steps: int = 1
    travel_weight: float = 1.0
    service_weight: float = 1.0
    energy_weight: float = 1.0
    deadline_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.hidden_dim < 1 or self.history_dim < 0 or self.horizon_steps < 1:
            raise ValueError("model dimensions and horizon_steps must be positive")
        if min(self.travel_weight, self.service_weight, self.energy_weight, self.deadline_weight) < 0:
            raise ValueError("candidate score weights cannot be negative")


@dataclass(frozen=True)
class ConsequenceTarget:
    """One simulator-generated counterfactual label.

    Hidden simulator state may be used to generate or audit this label, but it
    is deliberately not stored as an online model feature.  ``exogenous_key``
    identifies the shared random stream used by paired factual/counterfactual
    branches.
    """

    episode_id: str
    decision_index: int
    action: int
    horizon_steps: int
    exogenous_key: str
    travel_time: float
    service_progress: float
    energy_delta: float
    deadline_risk: float
    source: str = "simulator-counterfactual"
    hidden_state_used_for_label: bool = True

    def validate(self, expected_horizon_steps: int | None = None) -> None:
        if not self.episode_id or not self.exogenous_key:
            raise ValueError("episode_id and exogenous_key are required")
        if self.decision_index < 0 or self.action < 0 or self.horizon_steps < 1:
            raise ValueError("decision_index, action, and horizon_steps must be non-negative/positive")
        if expected_horizon_steps is not None and self.horizon_steps != expected_horizon_steps:
            raise ValueError("label horizon does not match the frozen model horizon")
        if self.source != "simulator-counterfactual":
            raise ValueError("online labels must come from a simulator counterfactual branch")
        if not 0.0 <= self.deadline_risk <= 1.0:
            raise ValueError("deadline_risk must be in [0, 1]")
        values = (self.travel_time, self.service_progress, self.energy_delta)
        if not all(torch.isfinite(torch.tensor(value)) for value in values):
            raise ValueError("continuous consequence labels must be finite")

    def as_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "decision_index": self.decision_index,
            "action": self.action,
            "horizon_steps": self.horizon_steps,
            "exogenous_key": self.exogenous_key,
            "travel_time": self.travel_time,
            "service_progress": self.service_progress,
            "energy_delta": self.energy_delta,
            "deadline_risk": self.deadline_risk,
            "source": self.source,
            "hidden_state_used_for_label": self.hidden_state_used_for_label,
        }


@dataclass(frozen=True)
class ConsequencePrediction:
    """Candidate-wise predictive means, variances, and deadline-risk logits."""

    actions: torch.Tensor
    means: torch.Tensor
    logvars: torch.Tensor
    deadline_logits: torch.Tensor

    def __post_init__(self) -> None:
        n = int(self.actions.shape[0])
        if self.actions.ndim != 1 or self.means.shape != (n, len(CONTINUOUS_TARGETS)):
            raise ValueError("candidate consequence shapes are inconsistent")
        if self.logvars.shape != self.means.shape or self.deadline_logits.shape != (n,):
            raise ValueError("candidate uncertainty shapes are inconsistent")

    @property
    def deadline_risk(self) -> torch.Tensor:
        return torch.sigmoid(self.deadline_logits)

    @property
    def stddev(self) -> torch.Tensor:
        return torch.exp(0.5 * self.logvars)

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "actions": self.actions,
            "travel_time": self.means[:, 0],
            "service_progress": self.means[:, 1],
            "energy_delta": self.means[:, 2],
            "deadline_risk": self.deadline_risk,
            "continuous_logvar": self.logvars,
            "continuous_stddev": self.stddev,
            "deadline_logits": self.deadline_logits,
        }


class ActionConsequenceWorldModel(nn.Module):
    """Predict the consequence of every legal UAV-task candidate separately.

    The graph encoder is shared, while the candidate embedding remains in the
    per-action row.  This prevents the previous aggregate-context behavior
    from collapsing all candidate predictions to one summary.
    """

    format_version = "gppo-action-consequence-world-model-v1"

    def __init__(
        self,
        base_world_model: GraphWorldModel,
        config: ConsequenceModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.base_world_model = base_world_model
        self.config = config or ConsequenceModelConfig()
        c = self.config
        input_dim = base_world_model.config.graph_dim + base_world_model.config.action_dim
        if c.history_dim:
            self.history_encoder = nn.Sequential(
                nn.Linear(c.history_dim, base_world_model.config.graph_dim),
                nn.LayerNorm(base_world_model.config.graph_dim),
                nn.SiLU(),
            )
            input_dim += base_world_model.config.graph_dim
        else:
            self.history_encoder = None
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, c.hidden_dim), nn.LayerNorm(c.hidden_dim), nn.SiLU()
        )
        self.continuous_mean = nn.Linear(c.hidden_dim, len(CONTINUOUS_TARGETS))
        self.continuous_logvar = nn.Linear(c.hidden_dim, len(CONTINUOUS_TARGETS))
        self.deadline_logit = nn.Linear(c.hidden_dim, 1)

    def _candidate_embeddings(self, graph: GraphSnapshot, actions: Sequence[int]) -> torch.Tensor:
        if not actions:
            return torch.empty(
                (0, self.base_world_model.config.action_dim),
                dtype=graph.nodes["uav"].dtype,
                device=graph.nodes["uav"].device,
            )
        return torch.stack([self.base_world_model._action_embedding(graph, action) for action in actions])

    def predict_candidates(
        self,
        graph: GraphSnapshot,
        legal_actions: Iterable[int] | None = None,
        *,
        history: torch.Tensor | None = None,
    ) -> ConsequencePrediction:
        """Return one row per legal action using only currently visible input."""

        if legal_actions is None:
            actions = [
                action for action, allowed in enumerate(graph.action_mask.tolist()) if allowed
            ]
        else:
            actions = [int(action) for action in legal_actions]
        if len(actions) != len(set(actions)):
            raise ValueError("legal_actions must not contain duplicates")
        if any(action < 0 or action >= graph.num_actions for action in actions):
            raise ValueError("legal action is outside the frozen action contract")
        device = graph.nodes["uav"].device
        parameter_device = next(self.parameters()).device
        if device != parameter_device:
            raise RuntimeError(
                f"graph/model device mismatch: graph={device}, model={parameter_device}; "
                "move graph, history, targets, and model together"
            )
        for name, value in graph.nodes.items():
            if value.device != device:
                raise RuntimeError(f"graph node {name} is on {value.device}, expected {device}")
        for relation, value in graph.edge_index.items():
            if value.device != device or graph.edge_attr[relation].device != device:
                raise RuntimeError(f"graph relation {relation} is not on {device}")
        if graph.candidate_edges.device != device or graph.action_mask.device != device:
            raise RuntimeError("graph action tensors are not on the model device")
        graph_embedding = self.base_world_model.graph_encoder(graph)
        action_embeddings = self._candidate_embeddings(graph, actions)
        rows = [graph_embedding.expand(len(actions), -1), action_embeddings]
        if self.history_encoder is not None:
            if history is None or history.ndim != 1 or history.shape[0] != self.config.history_dim:
                raise ValueError("history must be a visible vector with the frozen history_dim")
            rows.append(self.history_encoder(history.to(device)).expand(len(actions), -1))
        elif history is not None:
            raise ValueError("history was supplied but history_dim is zero")
        features = torch.cat(rows, dim=-1) if actions else torch.empty((0, self.trunk[0].in_features), device=device)
        hidden = self.trunk(features)
        means = self.continuous_mean(hidden)
        logvars = self.continuous_logvar(hidden).clamp(-8.0, 4.0)
        risk = self.deadline_logit(hidden).squeeze(-1)
        return ConsequencePrediction(
            actions=torch.tensor(actions, dtype=torch.long, device=device),
            means=means,
            logvars=logvars,
            deadline_logits=risk,
        )

    def auxiliary_score(self, prediction: ConsequencePrediction) -> torch.Tensor:
        """Compute a candidate feature; caller keeps the original reward intact."""

        c = self.config
        return (
            c.service_weight * prediction.means[:, 1]
            - c.travel_weight * prediction.means[:, 0]
            - c.energy_weight * prediction.means[:, 2]
            - c.deadline_weight * prediction.deadline_risk
        )

    def combine_policy_scores(
        self,
        policy_scores: torch.Tensor,
        prediction: ConsequencePrediction,
        coefficient: float,
    ) -> torch.Tensor:
        """Add an auxiliary feature to policy scores without reward shaping."""

        if policy_scores.shape != prediction.actions.shape:
            raise ValueError("policy_scores must have one value per candidate")
        if not torch.isfinite(policy_scores).all() or not torch.isfinite(torch.tensor(coefficient)):
            raise ValueError("policy scores and coefficient must be finite")
        return policy_scores + float(coefficient) * self.auxiliary_score(prediction)


class Graph5ActionConsequenceWorldModel(nn.Module):
    """Candidate consequence model for the public M-10 Graph-5 contract.

    This is a separate architecture, not a renamed legacy checkpoint.  Each
    candidate keeps its own relation row and action embedding; NOOP receives a
    zero relation row and its own action embedding.
    """

    format_version = "gppo-m10-graph5-action-consequence-v1"

    def __init__(self, config: ConsequenceModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ConsequenceModelConfig()
        c = self.config
        action_dim = max(16, c.hidden_dim // 2)
        self.node_encoders = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(32, c.hidden_dim), nn.LayerNorm(c.hidden_dim), nn.SiLU())
            for name in ("uav", "region", "target", "task", "event")
        })
        self.graph_projection = nn.Sequential(nn.Linear(5 * c.hidden_dim, c.hidden_dim), nn.LayerNorm(c.hidden_dim), nn.SiLU())
        self.global_encoder = nn.Sequential(nn.Linear(GRAPH5_GLOBAL_DIM, c.hidden_dim), nn.LayerNorm(c.hidden_dim), nn.SiLU())
        self.relation_encoder = nn.Sequential(nn.Linear(4, action_dim), nn.LayerNorm(action_dim), nn.SiLU())
        self.action_embedding = nn.Embedding(GRAPH5_ACTION_COUNT, action_dim)
        input_dim = 2 * c.hidden_dim + action_dim + action_dim
        if c.history_dim:
            self.history_encoder = nn.Sequential(nn.Linear(c.history_dim, c.hidden_dim), nn.LayerNorm(c.hidden_dim), nn.SiLU())
            input_dim += c.hidden_dim
        else:
            self.history_encoder = None
        self.trunk = nn.Sequential(nn.Linear(input_dim, c.hidden_dim), nn.LayerNorm(c.hidden_dim), nn.SiLU())
        self.continuous_mean = nn.Linear(c.hidden_dim, len(CONTINUOUS_TARGETS))
        self.continuous_logvar = nn.Linear(c.hidden_dim, len(CONTINUOUS_TARGETS))
        self.deadline_logit = nn.Linear(c.hidden_dim, 1)

    def predict_candidates(self, graph: Graph5Snapshot, legal_actions: Iterable[int] | None = None, *, history: torch.Tensor | None = None) -> ConsequencePrediction:
        actions = [action for action, allowed in enumerate(graph.action_mask.tolist()) if allowed] if legal_actions is None else [int(action) for action in legal_actions]
        if len(actions) != len(set(actions)) or any(action < 0 or action >= GRAPH5_ACTION_COUNT for action in actions):
            raise ValueError("legal_actions must be unique and inside the Graph-5 action contract")
        device = graph.nodes["uav"].device
        if next(self.parameters()).device != device:
            raise RuntimeError("Graph-5 graph/model device mismatch")
        pooled = [self.node_encoders[name](graph.nodes[name]).mean(dim=0) for name in ("uav", "region", "target", "task", "event")]
        graph_embedding = self.graph_projection(torch.cat(pooled, dim=-1))
        global_embedding = self.global_encoder(graph.global_features)
        if actions:
            relation_rows = [graph.candidate_features[action] if action < GRAPH5_ACTION_COUNT - 1 else torch.zeros(4, device=device) for action in actions]
            features = torch.cat((graph_embedding.expand(len(actions), -1), global_embedding.expand(len(actions), -1), self.relation_encoder(torch.stack(relation_rows)), self.action_embedding(torch.tensor(actions, device=device))), dim=-1)
        else:
            features = torch.empty((0, self.trunk[0].in_features), device=device)
        if self.history_encoder is not None:
            if history is None or history.ndim != 1 or history.shape[0] != self.config.history_dim:
                raise ValueError("history is missing or has the wrong visible dimension")
            history_features = self.history_encoder(history.to(device)).expand(len(actions), -1)
            features = torch.cat((features, history_features), dim=-1)
        elif history is not None:
            raise ValueError("history was supplied but history_dim is zero")
        hidden = self.trunk(features)
        return ConsequencePrediction(torch.tensor(actions, dtype=torch.long, device=device), self.continuous_mean(hidden), self.continuous_logvar(hidden).clamp(-8.0, 4.0), self.deadline_logit(hidden).squeeze(-1))

    def auxiliary_score(self, prediction: ConsequencePrediction) -> torch.Tensor:
        c = self.config
        return c.service_weight * prediction.means[:, 1] - c.travel_weight * prediction.means[:, 0] - c.energy_weight * prediction.means[:, 2] - c.deadline_weight * prediction.deadline_risk

    def combine_policy_scores(self, policy_scores: torch.Tensor, prediction: ConsequencePrediction, coefficient: float) -> torch.Tensor:
        if policy_scores.shape != prediction.actions.shape or not torch.isfinite(policy_scores).all():
            raise ValueError("policy scores must be finite and candidate aligned")
        return policy_scores + float(coefficient) * self.auxiliary_score(prediction)


def consequence_loss(
    prediction: ConsequencePrediction,
    targets: Mapping[str, torch.Tensor],
    *,
    masks: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute separate continuous uncertainty losses and BCE deadline loss."""

    required = set(TARGET_NAMES)
    if set(targets) != required:
        raise ValueError(f"targets must contain exactly {sorted(required)}")
    masks = masks or {}
    losses: dict[str, torch.Tensor] = {}
    for index, name in enumerate(CONTINUOUS_TARGETS):
        target = targets[name].to(prediction.means.device, dtype=prediction.means.dtype)
        if target.shape != prediction.means[:, index].shape:
            raise ValueError(f"target shape mismatch for {name}")
        mask = masks.get(name, torch.ones_like(target, dtype=torch.bool)).to(torch.bool)
        if mask.shape != target.shape:
            raise ValueError(f"target mask shape mismatch for {name}")
        if not mask.any():
            continue
        error = target - prediction.means[:, index]
        nll = 0.5 * (torch.exp(-prediction.logvars[:, index]) * error.square() + prediction.logvars[:, index])
        losses[f"{name}_nll"] = nll[mask].mean()
    risk_target = targets["deadline_risk"].to(prediction.deadline_logits.device, dtype=prediction.deadline_logits.dtype)
    risk_mask = masks.get("deadline_risk", torch.ones_like(risk_target, dtype=torch.bool)).to(torch.bool)
    if risk_target.shape != prediction.deadline_logits.shape or risk_mask.shape != risk_target.shape:
        raise ValueError("deadline_risk target/mask shape is invalid")
    if risk_mask.any():
        losses["deadline_risk_bce"] = F.binary_cross_entropy_with_logits(
            prediction.deadline_logits[risk_mask], risk_target[risk_mask]
        )
    if not losses:
        raise ValueError("at least one valid consequence target is required")
    losses["total"] = torch.stack(tuple(losses.values())).mean()
    if not torch.isfinite(losses["total"]):
        raise FloatingPointError("non-finite consequence loss")
    return losses
