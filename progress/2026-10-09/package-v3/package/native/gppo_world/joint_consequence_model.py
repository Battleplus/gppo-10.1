"""Small candidate-wise regressor for fixed task-set consequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .graph5 import GRAPH5_ACTION_COUNT, GRAPH5_GLOBAL_DIM, Graph5Snapshot


@dataclass(frozen=True)
class JointConsequenceModelConfig:
    hidden_dim: int = 64


class Graph5JointConsequenceModel(nn.Module):
    format_version = "gppo-m10-graph5-joint-consequence-v1"

    def __init__(self, config: JointConsequenceModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or JointConsequenceModelConfig()
        h = cfg.hidden_dim
        self.config = cfg
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(32, h), nn.LayerNorm(h), nn.SiLU())
            for name in ("uav", "region", "target", "task", "event")
        })
        self.global_encoder = nn.Sequential(nn.Linear(GRAPH5_GLOBAL_DIM, h), nn.LayerNorm(h), nn.SiLU())
        self.relation_encoder = nn.Sequential(nn.Linear(4, h // 2), nn.LayerNorm(h // 2), nn.SiLU())
        self.action_embedding = nn.Embedding(GRAPH5_ACTION_COUNT, h // 2)
        # Five pooled node types + global context + candidate relation/action
        # + public task-set size.  The task-set size is public and prevents
        # raw count labels from being confused across different denominators.
        self.trunk = nn.Sequential(nn.Linear(7 * h + 1, h), nn.LayerNorm(h), nn.SiLU())
        self.head = nn.Linear(h, 2)

    def predict_candidates(self, graph: Graph5Snapshot, task_set_size: int, legal_actions: Iterable[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        actions = [i for i, allowed in enumerate(graph.action_mask.tolist()) if allowed] if legal_actions is None else [int(i) for i in legal_actions]
        if len(actions) != len(set(actions)) or any(i < 0 or i >= GRAPH5_ACTION_COUNT or not bool(graph.action_mask[i]) for i in actions):
            raise ValueError("legal_actions must be unique and legal")
        device = graph.nodes["uav"].device
        if next(self.parameters()).device != device:
            raise RuntimeError(f"graph/model device mismatch: graph={device}, model={next(self.parameters()).device}")
        pooled = [self.encoders[name](graph.nodes[name]).mean(dim=0) for name in ("uav", "region", "target", "task", "event")]
        shared = torch.cat((torch.cat(pooled, dim=-1), self.global_encoder(graph.global_features)), dim=-1)
        rows = []
        for action in actions:
            relation = graph.candidate_features[action] if action < GRAPH5_ACTION_COUNT - 1 else torch.zeros(4, device=device)
            size = torch.tensor([float(task_set_size) / 6.0], device=device)
            rows.append(torch.cat((shared, self.relation_encoder(relation), self.action_embedding(torch.tensor(action, device=device)), size)))
        if not rows:
            return torch.empty((0,), dtype=torch.long, device=device), torch.empty((0, 2), device=device)
        hidden = self.trunk(torch.stack(rows))
        return torch.tensor(actions, dtype=torch.long, device=device), self.head(hidden)


def joint_loss(prediction: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape != targets.shape or prediction.ndim != 2 or prediction.shape[1] != 2:
        raise ValueError("joint prediction and target shapes must be [N,2]")
    mask = mask.to(device=prediction.device, dtype=torch.bool)
    if not bool(mask.any()):
        raise ValueError("all joint consequence labels are masked")
    loss = (prediction - targets.to(device=prediction.device, dtype=torch.float32)).square().mean(dim=1)[mask].mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite joint consequence loss")
    return loss


__all__ = ["Graph5JointConsequenceModel", "JointConsequenceModelConfig", "joint_loss"]
