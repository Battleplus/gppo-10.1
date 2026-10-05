"""Versioned public M-10 Graph-5 observation contract.

The adapter intentionally has no access to the environment object.  It turns
the public observation payload into five typed node tensors and 24 candidate
rows plus NOOP, so a legacy GraphSnapshot cannot be passed by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch


GRAPH5_TYPES = ("uav", "region", "target", "task", "event")
GRAPH5_NODE_DIM = 32
GRAPH5_CANDIDATE_COUNT = 24
GRAPH5_ACTION_COUNT = 25
GRAPH5_GLOBAL_DIM = 2 + GRAPH5_ACTION_COUNT


def _tensor(value: Any, *, dtype: torch.dtype, device: torch.device | str | None = None) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=dtype, device=device).detach().clone().contiguous()
    result.requires_grad_(False)
    return result


@dataclass(frozen=True)
class Graph5Snapshot:
    nodes: Mapping[str, torch.Tensor]
    candidate_features: torch.Tensor
    action_mask: torch.Tensor
    global_features: torch.Tensor
    graph_version: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", MappingProxyType({key: _tensor(value, dtype=torch.float32) for key, value in self.nodes.items()}))
        object.__setattr__(self, "candidate_features", _tensor(self.candidate_features, dtype=torch.float32))
        object.__setattr__(self, "action_mask", _tensor(self.action_mask, dtype=torch.bool))
        object.__setattr__(self, "global_features", _tensor(self.global_features, dtype=torch.float32))
        object.__setattr__(self, "graph_version", int(self.graph_version))
        self.validate()

    @property
    def num_actions(self) -> int:
        return GRAPH5_ACTION_COUNT

    def validate(self) -> None:
        if set(self.nodes) != set(GRAPH5_TYPES):
            raise ValueError(f"Graph-5 node types must be {GRAPH5_TYPES}, got {sorted(self.nodes)}")
        for name in GRAPH5_TYPES:
            value = self.nodes[name]
            if value.ndim != 2 or value.shape[1] != GRAPH5_NODE_DIM:
                raise ValueError(f"{name} nodes must be [N,{GRAPH5_NODE_DIM}]")
        if self.candidate_features.shape != (GRAPH5_CANDIDATE_COUNT, 4):
            raise ValueError("candidate_features must be [24,4]")
        if self.action_mask.shape != (GRAPH5_ACTION_COUNT,) or not bool(self.action_mask.any()):
            raise ValueError("action_mask must contain 25 actions and at least one legal action")
        if self.global_features.shape != (GRAPH5_GLOBAL_DIM,) or not torch.isfinite(self.global_features).all():
            raise ValueError(f"global_features must be a finite [{GRAPH5_GLOBAL_DIM}] public context")

    def to(self, device: torch.device | str) -> "Graph5Snapshot":
        moved = object.__new__(Graph5Snapshot)
        object.__setattr__(moved, "nodes", MappingProxyType({key: value.to(device) for key, value in self.nodes.items()}))
        object.__setattr__(moved, "candidate_features", self.candidate_features.to(device))
        object.__setattr__(moved, "action_mask", self.action_mask.to(device))
        object.__setattr__(moved, "global_features", self.global_features.to(device))
        object.__setattr__(moved, "graph_version", self.graph_version)
        moved.validate()
        return moved

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": {key: value.tolist() for key, value in self.nodes.items()},
            "candidate_features": self.candidate_features.tolist(),
            "action_mask": self.action_mask.tolist(),
            "global_features": self.global_features.tolist(),
            "graph_version": self.graph_version,
        }


def graph5_from_dict(value: Mapping[str, Any]) -> Graph5Snapshot:
    return Graph5Snapshot(
        nodes={key: value["nodes"][key] for key in GRAPH5_TYPES},
        candidate_features=value["candidate_features"],
        action_mask=value["action_mask"],
        global_features=value.get("global_features", [0.0] * GRAPH5_GLOBAL_DIM),
        graph_version=int(value["graph_version"]),
    )


def graph5_from_m10_observation(observation: Mapping[str, Any], *, counts: tuple[int, int, int, int, int] = (4, 3, 4, 6, 4)) -> Graph5Snapshot:
    graph = observation.get("graph")
    if not isinstance(graph, Mapping) or "node_features" not in graph or "relations" not in graph:
        raise ValueError("M-10 observation lacks the public graph payload")
    features = torch.as_tensor(graph["node_features"], dtype=torch.float32)
    if features.ndim != 2 or features.shape[1] != GRAPH5_NODE_DIM or sum(counts) != features.shape[0]:
        raise ValueError("M-10 Graph-5 node counts or feature width do not match the frozen contract")
    nodes: dict[str, torch.Tensor] = {}
    cursor = 0
    for name, count in zip(GRAPH5_TYPES, counts):
        nodes[name] = features[cursor:cursor + count]
        cursor += count
    relations = torch.as_tensor(graph["relations"], dtype=torch.float32)
    if relations.shape != (counts[0], counts[3], 4):
        raise ValueError("M-10 candidate relation tensor does not match 4 UAV x 6 Task x 4")
    continuation = torch.zeros(GRAPH5_ACTION_COUNT, dtype=torch.float32)
    for action in observation.get("continuation_actions", ()):
        action = int(action)
        if 0 <= action < GRAPH5_ACTION_COUNT:
            continuation[action] = 1.0
    global_features = torch.cat((
        torch.tensor([
            float(observation.get("time", 0.0)) / 18.0,
            float(observation.get("event_signal", 0.0)),
        ]),
        continuation,
    ))
    return Graph5Snapshot(
        nodes=nodes,
        candidate_features=relations.reshape(GRAPH5_CANDIDATE_COUNT, 4),
        action_mask=observation["mask"],
        global_features=global_features,
        graph_version=int(observation["version"]),
    )


__all__ = ["GRAPH5_TYPES", "GRAPH5_NODE_DIM", "GRAPH5_CANDIDATE_COUNT", "GRAPH5_ACTION_COUNT", "GRAPH5_GLOBAL_DIM", "Graph5Snapshot", "graph5_from_dict", "graph5_from_m10_observation"]
