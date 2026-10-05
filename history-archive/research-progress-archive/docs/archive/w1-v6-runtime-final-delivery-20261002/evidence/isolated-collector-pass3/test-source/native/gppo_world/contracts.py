"""Versioned causal contracts for GPPO world-model data.

The policy-facing graph is copied into an immutable snapshot at decision time.
Targets remain on :class:`Transition` and are deliberately absent from
:class:`WorldModelInput`, preventing accidental future-state export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import torch

from .registry import FEATURE_REGISTRY, SCHEMA_VERSION


Relation = tuple[str, str, str]

TRUTH_ONLY_FIELD_NAMES = frozenset(
    {
        "future_action",
        "future_action_mask",
        "future_confirmation",
        "future_graph",
        "future_observation",
        "ground_truth_event",
        "internal_drop_decision",
        "optimal_action",
        "oracle_action",
        "truth_event",
        "truth_event_occurred_at",
        "truth_occurred_at",
    }
)


def _freeze_tensor(value: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype).detach().cpu().clone().contiguous()
    tensor.requires_grad_(False)
    return tensor


def _freeze_mapping(values: Mapping[Any, torch.Tensor], *, dtype: torch.dtype | None = None):
    return MappingProxyType({key: _freeze_tensor(value, dtype=dtype) for key, value in values.items()})


def _find_truth_only(value: Any, path: str = "payload") -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).strip().lower()
            child_path = f"{path}.{key}"
            if name in TRUTH_ONLY_FIELD_NAMES:
                violations.append(child_path)
            violations.extend(_find_truth_only(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            violations.extend(_find_truth_only(child, f"{path}[{index}]"))
    return violations


@dataclass(frozen=True)
class GraphSnapshot:
    """Detached, CPU-resident copy of the GPPO decision graph."""

    nodes: Mapping[str, torch.Tensor]
    edge_index: Mapping[Relation, torch.Tensor]
    edge_attr: Mapping[Relation, torch.Tensor]
    candidate_edges: torch.Tensor
    action_mask: torch.Tensor
    graph_version: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", _freeze_mapping(self.nodes, dtype=torch.float32))
        object.__setattr__(self, "edge_index", _freeze_mapping(self.edge_index, dtype=torch.long))
        object.__setattr__(self, "edge_attr", _freeze_mapping(self.edge_attr, dtype=torch.float32))
        object.__setattr__(self, "candidate_edges", _freeze_tensor(self.candidate_edges, dtype=torch.long))
        object.__setattr__(self, "action_mask", _freeze_tensor(self.action_mask, dtype=torch.bool))
        object.__setattr__(self, "graph_version", int(self.graph_version))
        self.validate()

    @property
    def noop_action(self) -> int:
        return int(self.candidate_edges.shape[0])

    @property
    def num_actions(self) -> int:
        return self.noop_action + 1

    def to(self, device: torch.device | str) -> "GraphSnapshot":
        """Return a detached device-local copy for batched model execution."""

        moved = object.__new__(GraphSnapshot)
        object.__setattr__(moved, "nodes", MappingProxyType({key: value.to(device) for key, value in self.nodes.items()}))
        object.__setattr__(moved, "edge_index", MappingProxyType({key: value.to(device) for key, value in self.edge_index.items()}))
        object.__setattr__(moved, "edge_attr", MappingProxyType({key: value.to(device) for key, value in self.edge_attr.items()}))
        object.__setattr__(moved, "candidate_edges", self.candidate_edges.to(device))
        object.__setattr__(moved, "action_mask", self.action_mask.to(device))
        object.__setattr__(moved, "graph_version", self.graph_version)
        moved.validate()
        return moved

    def validate(self) -> None:
        expected_nodes = FEATURE_REGISTRY.node_dimensions
        if set(self.nodes) != set(expected_nodes):
            raise ValueError(f"node types must be {sorted(expected_nodes)}, got {sorted(self.nodes)}")
        for node_type, feature_dim in expected_nodes.items():
            value = self.nodes[node_type]
            if value.ndim != 2 or int(value.shape[1]) != feature_dim:
                raise ValueError(
                    f"{node_type} node tensor must be [N,{feature_dim}], got {tuple(value.shape)}"
                )
        if self.candidate_edges.ndim != 2 or self.candidate_edges.shape[1] != 2:
            raise ValueError("candidate_edges must have shape [A-1,2]")
        if self.action_mask.ndim != 1 or self.action_mask.shape[0] != self.num_actions:
            raise ValueError("action_mask must contain one bit per candidate plus NOOP")
        if not bool(self.action_mask.any().item()):
            raise ValueError("at least one action must be legal")
        for relation, feature_dim in FEATURE_REGISTRY.edge_dimensions.items():
            if relation not in self.edge_index or relation not in self.edge_attr:
                raise ValueError(f"missing relation {relation}")
            index = self.edge_index[relation]
            attr = self.edge_attr[relation]
            if index.ndim != 2 or index.shape[0] != 2:
                raise ValueError(f"edge_index[{relation}] must be [2,E]")
            if attr.ndim != 2 or attr.shape[0] != index.shape[1] or attr.shape[1] != feature_dim:
                raise ValueError(f"edge_attr[{relation}] does not match [E,{feature_dim}]")


@dataclass(frozen=True)
class EvidenceItem:
    """Evidence that was available to the policy by ``received_at``."""

    source: str
    signal_type: str
    received_at: float
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        violations = _find_truth_only(self.payload)
        if violations:
            raise ValueError(f"truth-only fields are forbidden: {', '.join(violations)}")
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "signal_type", str(self.signal_type))
        object.__setattr__(self, "received_at", float(self.received_at))
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


@dataclass(frozen=True)
class ExecutionRecord:
    """Separates a policy proposal from the action accepted by execution."""

    proposed_action: int
    executed_action: int | None
    accepted: bool
    graph_version: int
    action_version: int
    status: str
    command_id: str | None = None
    ack_id: str | None = None

    def __post_init__(self) -> None:
        if self.accepted and self.executed_action is None:
            raise ValueError("accepted execution requires executed_action")
        if not self.accepted and self.executed_action is not None:
            raise ValueError("rejected proposal cannot be recorded as executed")


@dataclass(frozen=True)
class WorldModelInput:
    """Only fields that may enter online world-model inference."""

    graph: GraphSnapshot
    evidence: tuple[EvidenceItem, ...]
    executed_action: int | None
    execution_accepted: bool
    decision_time: float
    schema_version: str


@dataclass(frozen=True)
class Transition:
    """One causally aligned GPPO transition with future values kept as targets."""

    episode_id: str
    scenario_id: str
    tape_id: str
    behavior_policy: str
    seed: int
    step: int
    decision_time: float
    next_decision_time: float
    graph_t: GraphSnapshot
    evidence_t: tuple[EvidenceItem, ...]
    execution: ExecutionRecord
    reward: float
    costs: Mapping[str, float]
    graph_tp1: GraphSnapshot
    continuation: bool
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_t", tuple(self.evidence_t))
        object.__setattr__(self, "costs", MappingProxyType({str(k): float(v) for k, v in self.costs.items()}))
        if not self.episode_id or not self.scenario_id or not self.tape_id:
            raise ValueError("episode_id, scenario_id and tape_id must be non-empty")
        if self.behavior_policy not in {"random_legal", "greedy", "gppo"}:
            raise ValueError("behavior_policy must be random_legal, greedy or gppo")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {self.schema_version!r}; expected {SCHEMA_VERSION!r}")
        if self.next_decision_time < self.decision_time:
            raise ValueError("next_decision_time cannot precede decision_time")
        future_evidence = [item for item in self.evidence_t if item.received_at > self.decision_time]
        if future_evidence:
            raise ValueError("evidence received after decision_time is forbidden")
        if self.execution.graph_version != self.graph_t.graph_version:
            raise ValueError("execution graph_version must match the decision graph")
        if self.graph_tp1.graph_version < self.graph_t.graph_version:
            raise ValueError("next graph_version cannot move backwards")
        action = self.execution.executed_action
        if action is not None:
            if not 0 <= action < self.graph_t.num_actions:
                raise ValueError("executed_action is outside the graph action space")
            if not bool(self.graph_t.action_mask[action].item()):
                raise ValueError("executed_action must be legal in the decision snapshot")

    def model_input(self) -> WorldModelInput:
        """Return a future-free view suitable for model inference/training input."""

        return WorldModelInput(
            graph=self.graph_t,
            evidence=self.evidence_t,
            executed_action=self.execution.executed_action,
            execution_accepted=self.execution.accepted,
            decision_time=float(self.decision_time),
            schema_version=self.schema_version,
        )


def snapshot_from_gppo(graph: Any) -> GraphSnapshot:
    """Adapt the baseline ``HeteroGraphState`` without importing GPPO internals."""

    required = ("nodes", "edge_index", "edge_attr", "candidate_edges", "action_mask", "graph_version")
    missing = [name for name in required if not hasattr(graph, name)]
    if missing:
        raise TypeError(f"GPPO graph is missing: {', '.join(missing)}")
    return GraphSnapshot(
        nodes=graph.nodes,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        candidate_edges=graph.candidate_edges,
        action_mask=graph.action_mask,
        graph_version=graph.graph_version,
    )
