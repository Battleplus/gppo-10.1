"""Deterministic multimodal event labels and Generic Event Segmentor (GES).

Targets are generated only from adjacent recorded observations.  The future
graph/evidence are labels and never become online model inputs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .data import EDGE_COUNTS, NODE_COUNTS, NODE_ORDER, RELATION_ORDER, STATE_DIM, TensorTransition
from .registry import FEATURE_REGISTRY


ORDINAL_CLASSES = ("DOWN", "SAME", "UP")
NOMINAL_CLASSES = ("SAME", "CHANGED")
EVIDENCE_EVENTS = ("new", "duplicate", "conflict", "confirm", "expire")
MODALITIES = ("ordinal", "nominal", "structural", "evidence")


@dataclass(frozen=True)
class StateEventSlot:
    state_index: int
    modality: str
    path: str


def state_event_slots() -> tuple[StateEventSlot, ...]:
    slots: list[StateEventSlot] = []
    offset = 0
    for node_type in NODE_ORDER:
        features = FEATURE_REGISTRY.nodes[node_type]
        for entity_index in range(NODE_COUNTS[node_type]):
            for feature_index, feature in enumerate(features):
                index = offset + entity_index * len(features) + feature_index
                if feature.event_eligible and feature.modality in {"ordinal", "nominal"}:
                    slots.append(
                        StateEventSlot(index, feature.modality, f"node/{node_type}/{entity_index}/{feature.name}")
                    )
        offset += NODE_COUNTS[node_type] * len(features)
    for relation in RELATION_ORDER:
        features = FEATURE_REGISTRY.edges[relation]
        relation_name = "/".join(relation)
        for edge_index in range(EDGE_COUNTS[relation]):
            for feature_index, feature in enumerate(features):
                index = offset + edge_index * len(features) + feature_index
                if feature.event_eligible and feature.modality in {"ordinal", "nominal"}:
                    slots.append(
                        StateEventSlot(index, feature.modality, f"edge/{relation_name}/{edge_index}/{feature.name}")
                    )
        offset += EDGE_COUNTS[relation] * len(features)
    if offset != STATE_DIM:
        raise RuntimeError(f"event slot layout covers {offset} state values, expected {STATE_DIM}")
    return tuple(slots)


STATE_EVENT_SLOTS = state_event_slots()
ORDINAL_SLOTS = tuple(slot for slot in STATE_EVENT_SLOTS if slot.modality == "ordinal")
NOMINAL_SLOTS = tuple(slot for slot in STATE_EVENT_SLOTS if slot.modality == "nominal")
STRUCTURAL_SLOT_NAMES = (
    *(f"node_count/{name}" for name in NODE_ORDER),
    *(f"relation_support/{'/'.join(relation)}" for relation in RELATION_ORDER),
    *(f"candidate_edge/{index}" for index in range(16)),
    *(f"action_support/{index}" for index in range(17)),
)


@dataclass(frozen=True)
class EventSchema:
    format_version: str
    ordinal_thresholds: Mapping[str, float]
    density_thresholds: Mapping[str, float]
    threshold_method: str
    density_method: str
    source_split: str
    source_transition_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "ordinal_thresholds": dict(sorted(self.ordinal_thresholds.items())),
            "density_thresholds": dict(sorted(self.density_thresholds.items())),
            "threshold_method": self.threshold_method,
            "density_method": self.density_method,
            "source_split": self.source_split,
            "source_transition_count": self.source_transition_count,
            "ordinal_classes": list(ORDINAL_CLASSES),
            "nominal_classes": list(NOMINAL_CLASSES),
            "evidence_events": list(EVIDENCE_EVENTS),
            "ordinal_slots": [slot.path for slot in ORDINAL_SLOTS],
            "nominal_slots": [slot.path for slot in NOMINAL_SLOTS],
            "structural_slots": list(STRUCTURAL_SLOT_NAMES),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EventSchema":
        return cls(
            format_version=str(value["format_version"]),
            ordinal_thresholds={str(k): float(v) for k, v in value["ordinal_thresholds"].items()},
            density_thresholds={str(k): float(v) for k, v in value["density_thresholds"].items()},
            threshold_method=str(value["threshold_method"]),
            density_method=str(value["density_method"]),
            source_split=str(value["source_split"]),
            source_transition_count=int(value["source_transition_count"]),
        )

    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EventLabels:
    ordinal: torch.Tensor
    nominal: torch.Tensor
    structural: torch.Tensor
    evidence: torch.Tensor
    evidence_valid: torch.Tensor
    density: Mapping[str, float]

    def event_mask(self, modality: str) -> torch.Tensor:
        if modality == "ordinal":
            return self.ordinal != 1
        if modality == "nominal":
            return self.nominal.to(torch.bool)
        if modality == "structural":
            return self.structural.to(torch.bool)
        if modality == "evidence":
            return self.evidence.to(torch.bool)
        raise KeyError(modality)

    def valid_mask(self, modality: str) -> torch.Tensor:
        if modality == "evidence":
            return self.evidence_valid
        return torch.ones_like(self.event_mask(modality), dtype=torch.bool)

    def canonical_bytes(self) -> bytes:
        value = {
            "ordinal": self.ordinal.tolist(),
            "nominal": self.nominal.tolist(),
            "structural": self.structural.tolist(),
            "evidence": self.evidence.tolist(),
            "evidence_valid": self.evidence_valid.tolist(),
            "density": {key: self.density[key] for key in MODALITIES},
        }
        return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _edge_support(graph, relation) -> tuple[tuple[int, int], ...]:
    index = graph.edge_index[relation]
    return tuple(sorted((int(index[0, i]), int(index[1, i])) for i in range(index.shape[1])))


def _structural_labels(transition: TensorTransition) -> torch.Tensor:
    current, future = transition.graph, transition.next_graph
    values: list[bool] = []
    values.extend(current.nodes[name].shape[0] != future.nodes[name].shape[0] for name in NODE_ORDER)
    values.extend(_edge_support(current, relation) != _edge_support(future, relation) for relation in RELATION_ORDER)
    for index in range(16):
        left = tuple(current.candidate_edges[index].tolist()) if index < current.candidate_edges.shape[0] else None
        right = tuple(future.candidate_edges[index].tolist()) if index < future.candidate_edges.shape[0] else None
        values.append(left != right)
    for index in range(17):
        left = bool(current.action_mask[index]) if index < current.action_mask.shape[0] else False
        right = bool(future.action_mask[index]) if index < future.action_mask.shape[0] else False
        values.append(left != right)
    return torch.tensor(values, dtype=torch.float32)


def _evidence_id(item: Mapping[str, Any]) -> str:
    payload = item.get("payload", {})
    event_id = payload.get("event_id")
    if event_id is not None:
        return str(event_id)
    canonical = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _evidence_signature(item: Mapping[str, Any]) -> str:
    # Delivery bookkeeping and lifecycle status are not semantic payload
    # conflicts.  In particular, a later confirmation must not also become a
    # conflict solely because its status changed.
    value = dict(item)
    value.pop("received_at", None)
    payload = dict(value.get("payload", {}))
    for key in ("state_version", "evidence_count", "status", "expires_at", "ttl"):
        payload.pop(key, None)
    value["payload"] = payload
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _has_expiry_contract(item: Mapping[str, Any]) -> bool:
    payload = item.get("payload", {})
    return any(key in item or key in payload for key in ("expires_at", "ttl"))


def _evidence_labels(
    current: tuple[dict, ...], future: tuple[dict, ...] | None
) -> tuple[torch.Tensor, torch.Tensor]:
    if future is None:
        return torch.zeros(len(EVIDENCE_EVENTS)), torch.zeros(len(EVIDENCE_EVENTS), dtype=torch.bool)
    current_items: dict[str, list[dict]] = defaultdict(list)
    future_items: dict[str, list[dict]] = defaultdict(list)
    for item in current:
        current_items[_evidence_id(item)].append(item)
    for item in future:
        future_items[_evidence_id(item)].append(item)
    new = any(key not in current_items for key in future_items)
    duplicate = any(
        len(items) > max(1, len(current_items.get(key, ()))) for key, items in future_items.items()
    )
    conflict = False
    for key, items in future_items.items():
        signatures = {_evidence_signature(item) for item in items}
        previous = {_evidence_signature(item) for item in current_items.get(key, ())}
        if len(signatures) > 1 or (previous and signatures != previous):
            conflict = True
            break
    confirm = any(
        key in current_items
        and str(item.get("payload", {}).get("status", "")).upper() == "CONFIRMED"
        and not any(
            str(old.get("payload", {}).get("status", "")).upper() == "CONFIRMED"
            for old in current_items[key]
        )
        for key, items in future_items.items()
        for item in items
    )
    expiry_observable = any(
        _has_expiry_contract(item)
        for items in (*current_items.values(), *future_items.values())
        for item in items
    )
    expire = expiry_observable and any(key not in future_items for key in current_items)
    labels = torch.tensor([new, duplicate, conflict, confirm, expire], dtype=torch.float32)
    # The current T-01 format has no TTL/expiry field.  Mark expiry targets
    # ineligible instead of silently turning unavailable semantics into an
    # all-negative class.
    valid = torch.tensor([True, True, True, True, expiry_observable], dtype=torch.bool)
    return labels, valid


def generate_event_labels(
    transition: TensorTransition,
    next_transition: TensorTransition | None,
    schema: EventSchema,
) -> EventLabels:
    delta = transition.target_delta
    ordinal_values: list[int] = []
    for slot in ORDINAL_SLOTS:
        value = float(delta[slot.state_index])
        threshold = schema.ordinal_thresholds[slot.path]
        ordinal_values.append(0 if value <= -threshold else (2 if value >= threshold else 1))
    nominal = torch.tensor(
        [abs(float(delta[slot.state_index])) > 1e-6 for slot in NOMINAL_SLOTS], dtype=torch.float32
    )
    structural = _structural_labels(transition)
    evidence, evidence_valid = _evidence_labels(
        transition.evidence, None if next_transition is None else next_transition.evidence
    )
    ordinal = torch.tensor(ordinal_values, dtype=torch.long)
    masks = {
        "ordinal": ordinal != 1,
        "nominal": nominal.to(torch.bool),
        "structural": structural.to(torch.bool),
        "evidence": evidence.to(torch.bool),
    }
    valid = {
        "ordinal": torch.ones_like(masks["ordinal"]),
        "nominal": torch.ones_like(masks["nominal"]),
        "structural": torch.ones_like(masks["structural"]),
        "evidence": evidence_valid,
    }
    density = {
        name: float(masks[name][valid[name]].float().mean()) if bool(valid[name].any()) else 0.0
        for name in MODALITIES
    }
    return EventLabels(ordinal, nominal, structural, evidence, evidence_valid, density)


def label_episodes(
    episodes: Iterable[list[TensorTransition]], schema: EventSchema
) -> list[list[tuple[TensorTransition, EventLabels]]]:
    result = []
    for episode in episodes:
        labeled = []
        for index, transition in enumerate(episode):
            next_transition = episode[index + 1] if index + 1 < len(episode) else None
            labeled.append((transition, generate_event_labels(transition, next_transition, schema)))
        result.append(labeled)
    return result


def freeze_event_schema(train_episodes: list[list[TensorTransition]]) -> EventSchema:
    transitions = [transition for episode in train_episodes for transition in episode]
    values: dict[str, list[float]] = {slot.path: [] for slot in ORDINAL_SLOTS}
    for transition in transitions:
        delta = transition.target_delta
        for slot in ORDINAL_SLOTS:
            absolute = abs(float(delta[slot.state_index]))
            if absolute > 1e-6:
                values[slot.path].append(absolute)
    thresholds = {
        path: max(1e-6, float(np.quantile(samples, 0.25))) if samples else 1.0
        for path, samples in values.items()
    }
    provisional = EventSchema(
        format_version="gppo-world-events/0.1.0",
        ordinal_thresholds=thresholds,
        density_thresholds={name: 1.0 for name in MODALITIES},
        threshold_method="train-only q25 of nonzero absolute delta, floor=1e-6; no-support=1.0",
        density_method="train-only q75 of positive event density, floor=one eligible slot",
        source_split="train",
        source_transition_count=len(transitions),
    )
    density_values: dict[str, list[float]] = {name: [] for name in MODALITIES}
    for episode in label_episodes(train_episodes, provisional):
        for _, labels in episode:
            for name in MODALITIES:
                if labels.density[name] > 0.0:
                    density_values[name].append(labels.density[name])
    sizes = {
        "ordinal": len(ORDINAL_SLOTS),
        "nominal": len(NOMINAL_SLOTS),
        "structural": len(STRUCTURAL_SLOT_NAMES),
        "evidence": len(EVIDENCE_EVENTS),
    }
    density_thresholds = {
        name: (
            min(1.0, max(1.0 / sizes[name], float(np.quantile(samples, 0.75))))
            if samples
            else 1.0
        )
        for name, samples in density_values.items()
    }
    return EventSchema(
        format_version=provisional.format_version,
        ordinal_thresholds=thresholds,
        density_thresholds=density_thresholds,
        threshold_method=provisional.threshold_method,
        density_method=provisional.density_method,
        source_split="train",
        source_transition_count=len(transitions),
    )


def ges_weight(density: float, threshold: float, mode: str, epsilon: float = 5e-4) -> float:
    """Paper-aligned GES: boundaries are densities greater than or equal to the threshold."""

    if mode == "none":
        return 1.0
    if density >= threshold:
        return 0.0
    if mode == "hard":
        return 1.0
    if mode == "smooth":
        ratio = min(1.0, max(epsilon, density / max(threshold, 1e-12)))
        return 1.0 / math.asinh(ratio)
    raise ValueError(f"unsupported GES mode {mode!r}")


def label_digest(labeled_episodes: Iterable[list[tuple[TensorTransition, EventLabels]]]) -> str:
    digest = hashlib.sha256()
    for episode in labeled_episodes:
        for transition, labels in episode:
            digest.update(transition.episode_id.encode("utf-8"))
            digest.update(str(transition.step).encode("ascii"))
            digest.update(labels.canonical_bytes())
    return digest.hexdigest()


def event_support_report(
    labeled_episodes: Iterable[list[tuple[TensorTransition, EventLabels]]]
) -> dict[str, Any]:
    counts = {name: Counter() for name in MODALITIES}
    density = {name: [] for name in MODALITIES}
    transition_count = 0
    for episode in labeled_episodes:
        for _, labels in episode:
            transition_count += 1
            for name in MODALITIES:
                mask = labels.event_mask(name)
                valid = labels.valid_mask(name)
                counts[name]["positive"] += int((mask & valid).sum())
                counts[name]["negative"] += int((~mask & valid).sum())
                counts[name]["valid"] += int(valid.sum())
                density[name].append(labels.density[name])
    return {
        "transition_count": transition_count,
        "modalities": {
            name: {
                **dict(counts[name]),
                "mean_density": float(np.mean(density[name])) if density[name] else 0.0,
                "max_density": float(np.max(density[name])) if density[name] else 0.0,
            }
            for name in MODALITIES
        },
    }
