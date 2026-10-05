"""Deterministic recorder shared by every behavior policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from .contracts import EvidenceItem, GraphSnapshot, Transition


def _tensor(value: torch.Tensor) -> list:
    return value.detach().cpu().tolist()


def _relation_key(relation: tuple[str, str, str]) -> str:
    return "/".join(relation)


def graph_to_dict(graph: GraphSnapshot) -> dict[str, Any]:
    return {
        "nodes": {name: _tensor(value) for name, value in sorted(graph.nodes.items())},
        "edge_index": {
            _relation_key(relation): _tensor(value)
            for relation, value in sorted(graph.edge_index.items())
        },
        "edge_attr": {
            _relation_key(relation): _tensor(value)
            for relation, value in sorted(graph.edge_attr.items())
        },
        "candidate_edges": _tensor(graph.candidate_edges),
        "action_mask": _tensor(graph.action_mask),
        "graph_version": graph.graph_version,
    }


def evidence_to_dict(item: EvidenceItem) -> dict[str, Any]:
    return {
        "source": item.source,
        "signal_type": item.signal_type,
        "received_at": item.received_at,
        "payload": dict(item.payload),
    }


def transition_to_dict(transition: Transition) -> dict[str, Any]:
    execution = transition.execution
    return {
        "schema_version": transition.schema_version,
        "episode_id": transition.episode_id,
        "scenario_id": transition.scenario_id,
        "tape_id": transition.tape_id,
        "behavior_policy": transition.behavior_policy,
        "seed": transition.seed,
        "step": transition.step,
        "decision_time": transition.decision_time,
        "next_decision_time": transition.next_decision_time,
        "graph_t": graph_to_dict(transition.graph_t),
        "evidence_t": [evidence_to_dict(item) for item in transition.evidence_t],
        "execution": {
            "proposed_action": execution.proposed_action,
            "executed_action": execution.executed_action,
            "accepted": execution.accepted,
            "graph_version": execution.graph_version,
            "action_version": execution.action_version,
            "status": execution.status,
            "command_id": execution.command_id,
            "ack_id": execution.ack_id,
        },
        "reward": transition.reward,
        "costs": dict(transition.costs),
        "graph_tp1": graph_to_dict(transition.graph_tp1),
        "continuation": transition.continuation,
    }


class TransitionRecorder:
    """Append-only in-memory recorder with canonical JSONL output."""

    def __init__(self) -> None:
        self._items: list[Transition] = []

    def append(self, transition: Transition) -> None:
        if self._items:
            previous = self._items[-1]
            if previous.episode_id == transition.episode_id and transition.step != previous.step + 1:
                raise ValueError("steps within an episode must be contiguous")
        self._items.append(transition)

    def extend(self, transitions: Iterable[Transition]) -> None:
        for transition in transitions:
            self.append(transition)

    @property
    def items(self) -> tuple[Transition, ...]:
        return tuple(self._items)

    def canonical_bytes(self) -> bytes:
        lines = [
            json.dumps(transition_to_dict(item), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for item in self._items
        ]
        return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def write_jsonl(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(self.canonical_bytes())
        return output
