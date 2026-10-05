"""Candidate-wise consequence model for the arrival-to-region protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .graph5 import GRAPH5_ACTION_COUNT, GRAPH5_GLOBAL_DIM, Graph5Snapshot


@dataclass(frozen=True)
class ArrivalModelConfig:
    hidden_dim: int = 96
    horizon_steps: int = 6


@dataclass(frozen=True)
class ArrivalPrediction:
    actions: torch.Tensor
    arrival_mean: torch.Tensor
    arrival_logvar: torch.Tensor
    deadline_logits: torch.Tensor
    failure_logits: torch.Tensor


class Graph5ArrivalConsequenceModel(nn.Module):
    """Predict arrival time, physical deadline success, and execution failure.

    Inputs are only the public Graph-5 snapshot and candidate identity.  The
    simulator hidden state is used exclusively to produce offline labels.
    """

    format_version = "gppo-m10-graph5-arrival-consequence-v1"

    def __init__(self, config: ArrivalModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ArrivalModelConfig()
        h = self.config.hidden_dim
        action_dim = max(16, h // 2)
        self.node_encoders = nn.ModuleDict({name: nn.Sequential(nn.Linear(32, h), nn.LayerNorm(h), nn.SiLU()) for name in ("uav", "region", "target", "task", "event")})
        self.graph_projection = nn.Sequential(nn.Linear(5 * h, h), nn.LayerNorm(h), nn.SiLU())
        self.global_encoder = nn.Sequential(nn.Linear(GRAPH5_GLOBAL_DIM, h), nn.LayerNorm(h), nn.SiLU())
        self.relation_encoder = nn.Sequential(nn.Linear(4, action_dim), nn.LayerNorm(action_dim), nn.SiLU())
        self.action_embedding = nn.Embedding(GRAPH5_ACTION_COUNT, action_dim)
        self.trunk = nn.Sequential(nn.Linear(2 * h + 2 * action_dim, h), nn.LayerNorm(h), nn.SiLU())
        self.arrival_mean = nn.Linear(h, 1)
        self.arrival_logvar = nn.Linear(h, 1)
        self.deadline_logit = nn.Linear(h, 1)
        self.failure_logit = nn.Linear(h, 1)

    def _hidden(self, graph: Graph5Snapshot, actions: list[int]) -> torch.Tensor:
        device = graph.nodes["uav"].device
        if next(self.parameters()).device != device:
            raise RuntimeError(f"Graph/model device mismatch: graph={device}, model={next(self.parameters()).device}")
        pooled = [self.node_encoders[name](graph.nodes[name]).mean(dim=0) for name in ("uav", "region", "target", "task", "event")]
        graph_embedding = self.graph_projection(torch.cat(pooled, dim=-1))
        global_embedding = self.global_encoder(graph.global_features)
        rows = []
        for action in actions:
            relation = graph.candidate_features[action] if action < GRAPH5_ACTION_COUNT - 1 else torch.zeros(4, device=device)
            rows.append(torch.cat((graph_embedding, global_embedding, self.relation_encoder(relation), self.action_embedding(torch.tensor(action, device=device)))))
        features = torch.stack(rows) if rows else torch.empty((0, self.trunk[0].in_features), device=device)
        return self.trunk(features)

    def predict_candidates(self, graph: Graph5Snapshot, legal_actions: Iterable[int] | None = None) -> ArrivalPrediction:
        actions = [i for i, allowed in enumerate(graph.action_mask.tolist()) if allowed] if legal_actions is None else [int(i) for i in legal_actions]
        if len(actions) != len(set(actions)) or any(i < 0 or i >= GRAPH5_ACTION_COUNT for i in actions):
            raise ValueError("legal_actions must be unique and inside the Graph-5/25-action contract")
        hidden = self._hidden(graph, actions)
        return ArrivalPrediction(
            actions=torch.tensor(actions, dtype=torch.long, device=hidden.device),
            arrival_mean=self.arrival_mean(hidden).squeeze(-1),
            arrival_logvar=self.arrival_logvar(hidden).squeeze(-1).clamp(-8.0, 4.0),
            deadline_logits=self.deadline_logit(hidden).squeeze(-1),
            failure_logits=self.failure_logit(hidden).squeeze(-1),
        )


def arrival_loss(prediction: ArrivalPrediction, targets: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    device = prediction.arrival_mean.device
    losses: dict[str, torch.Tensor] = {}
    arrival_mask = masks["arrival_time"].to(device=device, dtype=torch.bool)
    if arrival_mask.any():
        target = targets["arrival_time"].to(device=device, dtype=torch.float32)
        err = target - prediction.arrival_mean
        losses["arrival_time_nll"] = (0.5 * (torch.exp(-prediction.arrival_logvar) * err.square() + prediction.arrival_logvar))[arrival_mask].mean()
    for name, logits in (("deadline", prediction.deadline_logits), ("failure", prediction.failure_logits)):
        mask = masks[name].to(device=device, dtype=torch.bool)
        if mask.any():
            losses[f"{name}_bce"] = F.binary_cross_entropy_with_logits(logits[mask], targets[name].to(device=device, dtype=torch.float32)[mask])
    if not losses:
        raise ValueError("all arrival consequence heads are masked")
    losses["total"] = torch.stack(tuple(losses.values())).mean()
    if not torch.isfinite(losses["total"]):
        raise FloatingPointError("non-finite arrival consequence loss")
    return losses
