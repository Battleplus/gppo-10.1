"""Pure adapter from a frozen W1 public observation to the transition contract.

This module deliberately accepts an observation and an explicit public
telemetry sidecar.  It never imports or constructs the environment, and it
does not infer measurement or receipt times from flattened model features.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Any, Callable, Iterable, Mapping
from types import MappingProxyType
from public_transition_contract import PublicField, PublicSnapshot, freeze_decision_input, public_hash, validate_no_future


class PublicAdapterError(ValueError):
    """Raised when a production observation cannot satisfy the public contract."""


@dataclass(frozen=True)
class PublicTelemetry:
    entity: str
    field: str
    value: float
    measured_at: float
    received_at: float
    sequence: int
    message_id: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PublicTelemetry":
        required = ("entity", "field", "value", "measured_at", "received_at", "sequence")
        missing = [key for key in required if key not in value]
        if missing:
            raise PublicAdapterError(f"telemetry missing fields: {','.join(missing)}")
        try:
            item = cls(
                entity=str(value["entity"]), field=str(value["field"]),
                value=float(value["value"]), measured_at=float(value["measured_at"]),
                received_at=float(value["received_at"]), sequence=int(value["sequence"]),
                message_id=str(value.get("message_id", "")),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise PublicAdapterError("invalid public telemetry record") from exc
        if not item.entity or not item.field or item.sequence < 0:
            raise PublicAdapterError("public telemetry identity is invalid")
        if not all(math.isfinite(number) for number in (item.value, item.measured_at, item.received_at)):
            raise PublicAdapterError("public telemetry must be finite")
        if item.measured_at < 0 or item.received_at < item.measured_at:
            raise PublicAdapterError("public telemetry timeline is invalid")
        return item


@dataclass(frozen=True)
class W1PublicInput:
    graph: Any
    public: PublicSnapshot
    candidate_actions: tuple[int, ...]
    continuation_policy_id: str
    graph_hash: str
    graph_builder_id: str

    def validate_identity(self) -> None:
        graph_identity = self.graph.as_dict() if hasattr(self.graph, "as_dict") else self.graph
        if public_hash(graph_identity) != self.graph_hash:
            raise PublicAdapterError("frozen graph identity changed")
        rebuilt = PublicSnapshot(
            decision_time=self.public.decision_time, fields=self.public.fields,
            legal_actions=self.public.legal_actions, history=self.public.history,
            continuation_ids=self.public.continuation_ids, source=self.public.source,
        )
        if rebuilt.input_hash != self.public.input_hash:
            raise PublicAdapterError("frozen public snapshot identity changed")
        if self.candidate_actions != self.public.legal_actions:
            raise PublicAdapterError("candidate actions changed after adaptation")
        if not self.continuation_policy_id:
            raise PublicAdapterError("continuation policy identity is required")
        validate_no_future(self.public)
        _validate_graph_contract(self.graph, tuple(index in self.public.legal_actions for index in range(25)))

    def validate_production(self) -> None:
        self.validate_identity()
        if not self.graph_builder_id.endswith("gppo_world.graph5.graph5_from_m10_observation"):
            raise PublicAdapterError("unreviewed Graph-5 builder is forbidden in production")


def _validate_graph_contract(graph: Any, expected_mask: tuple[bool, ...]) -> None:
    """Check the shape boundary without importing an environment module."""
    if hasattr(graph, "candidate_features") and hasattr(graph, "action_mask"):
        if tuple(graph.candidate_features.shape) != (24, 4) or tuple(graph.action_mask.shape) != (25,):
            raise PublicAdapterError("Graph-5 builder returned an invalid candidate contract")
        actual_mask = tuple(bool(item) for item in graph.action_mask.tolist())
        if actual_mask != expected_mask:
            raise PublicAdapterError("Graph-5 builder changed the public legal mask")
        return
    if isinstance(graph, Mapping):
        try:
            shape_nodes = tuple(graph["node_features"].shape)
            shape_relations = tuple(graph["relations"].shape)
        except (KeyError, AttributeError, TypeError) as exc:
            raise PublicAdapterError("graph payload has no inspectable Graph-5 shape") from exc
        if shape_nodes != (21, 32) or shape_relations != (4, 6, 4):
            raise PublicAdapterError("graph payload does not match the 21-node/24-candidate contract")
        return
    raise PublicAdapterError("graph builder returned an unsupported object")


def _records(records: Iterable[PublicTelemetry | Mapping[str, Any]]) -> tuple[PublicTelemetry, ...]:
    return tuple(item if isinstance(item, PublicTelemetry) else PublicTelemetry.from_mapping(item) for item in records)


def adapt_m10_public_observation(
    observation: Mapping[str, Any],
    telemetry: Iterable[PublicTelemetry | Mapping[str, Any]],
    *,
    continuation_ids: Iterable[str] | None = None,
    continuation_policy_id: str | None = None,
    max_age: float | None = None,
    source: str = "w1-m10-public-adapter",
    graph_builder: Callable[[Mapping[str, Any]], Any] | None = None,
) -> W1PublicInput:
    """Convert one already-created public observation without touching runtime state.

    ``telemetry`` must contain records that were publicly received by the
    decision boundary.  The adapter preserves all records in ``history`` but
    selects the greatest sequence for each entity/field as its current field.
    A stale record remains auditable while its ``valid`` mask is false.
    """
    if not isinstance(observation, Mapping):
        raise PublicAdapterError("observation must be a mapping")
    try:
        decision_time = float(observation["time"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise PublicAdapterError("observation needs a finite decision time") from exc
    if not math.isfinite(decision_time) or decision_time < 0:
        raise PublicAdapterError("decision time must be finite and nonnegative")
    if max_age is not None and (not math.isfinite(float(max_age)) or float(max_age) < 0):
        raise PublicAdapterError("max_age must be finite and nonnegative")

    graph_payload = observation.get("graph")
    if not isinstance(graph_payload, Mapping) or set(graph_payload) != {"node_features", "relations"}:
        raise PublicAdapterError("observation lacks a public graph payload")
    # The production caller supplies the reviewed Graph-5 builder explicitly;
    # the default keeps this package independent from environment imports.
    builder_input = MappingProxyType({
        key: copy.deepcopy(observation[key])
        for key in ("graph", "mask", "time", "version", "event_signal", "continuation_actions", "continuation_ids")
        if key in observation
    })
    if graph_builder is None:
        graph = copy.deepcopy(graph_payload)
        builder_id = "payload-only-not-production"
    else:
        try:
            graph = graph_builder(builder_input)
        except Exception as exc:
            raise PublicAdapterError("production graph builder rejected public observation") from exc
        builder_id = f"{getattr(graph_builder, '__module__', '')}.{getattr(graph_builder, '__name__', '')}"
    mask = tuple(bool(item) for item in observation.get("mask", ()))
    if len(mask) != 25:
        raise PublicAdapterError("W1 public mask must contain 25 actions")
    _validate_graph_contract(graph, mask)
    records = _records(telemetry)
    latest: dict[tuple[str, str], PublicTelemetry] = {}
    history: list[dict[str, Any]] = []
    for item in records:
        if item.received_at > decision_time:
            raise PublicAdapterError("decision input contains telemetry received in the future")
        key = (item.entity, item.field)
        history.append({
            "entity": item.entity, "field": item.field, "value": item.value,
            "measured_at": item.measured_at, "received_at": item.received_at,
            "sequence": item.sequence, "message_id": item.message_id,
            "age": max(0.0, decision_time - item.measured_at),
            "known": True,
            "valid": max_age is None or decision_time - item.measured_at <= float(max_age),
        })
        previous = latest.get(key)
        if previous is not None and item.sequence < previous.sequence:
            continue
        if previous is not None and item.sequence == previous.sequence and item != previous:
            raise PublicAdapterError("conflicting duplicate public telemetry")
        latest[key] = item

    fields: dict[str, PublicField] = {}
    for (entity, field), item in sorted(latest.items()):
        age = max(0.0, decision_time - item.measured_at)
        valid = max_age is None or age <= float(max_age)
        fields[f"{entity}.{field}"] = PublicField(
            value=item.value, known=True, valid=valid, age=age,
            measured_at=item.measured_at, received_at=item.received_at,
        )

    if continuation_ids is None:
        if "continuation_ids" in observation:
            continuation_ids = observation["continuation_ids"]
        elif observation.get("continuation_actions"):
            raise PublicAdapterError(
                "continuation actions are present without public continuation identities"
            )
        else:
            continuation_ids = ()
    continuation = tuple(str(item) for item in continuation_ids)
    if any(not item for item in continuation):
        raise PublicAdapterError("continuation identities must be non-empty")
    if continuation_policy_id is None:
        continuation_policy_id = str(observation.get("continuation_policy_id", ""))
    if not isinstance(continuation_policy_id, str) or not continuation_policy_id:
        raise PublicAdapterError("continuation policy identity is required")

    public = PublicSnapshot(
        decision_time=decision_time,
        fields=fields,
        legal_actions=tuple(index for index, legal in enumerate(mask) if legal),
        history=tuple(history),
        continuation_ids=continuation,
        source=source,
    )
    public = freeze_decision_input(public)
    validate_no_future(public)
    graph_identity = graph.as_dict() if hasattr(graph, "as_dict") else graph
    result = W1PublicInput(
        graph=graph, public=public, candidate_actions=public.legal_actions,
        continuation_policy_id=continuation_policy_id, graph_hash=public_hash(graph_identity),
        graph_builder_id=builder_id,
    )
    result.validate_identity()
    return result


__all__ = ["PublicAdapterError", "PublicTelemetry", "adapt_m10_public_observation"]
