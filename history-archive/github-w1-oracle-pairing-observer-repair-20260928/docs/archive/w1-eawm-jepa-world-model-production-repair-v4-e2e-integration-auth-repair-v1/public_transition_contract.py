"""Causal public-input contract for the W1 Graph-5 world model.

The contract is deliberately independent of the environment.  It freezes the
decision-time public payload before any candidate branch runs and keeps future
observations on the target side of a transition.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
from typing import Any, Mapping


ACTION_COUNT = 25
NOOP_ACTION = 24
CONTRACT_VERSION = "w1-public-transition/1.0.0"


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite public input")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return _canonical(value.tolist())
    raise TypeError(f"unsupported public input type: {type(value)!r}")


def public_hash(value: Any) -> str:
    payload = json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PublicField:
    value: float | int | str | bool | None
    known: bool
    valid: bool
    age: float | None
    measured_at: float | None
    received_at: float | None

    def __post_init__(self) -> None:
        if self.age is not None and (not math.isfinite(float(self.age)) or float(self.age) < 0):
            raise ValueError("field age must be finite and non-negative")
        for name in ("measured_at", "received_at"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.received_at is not None and self.measured_at is not None and self.received_at < self.measured_at:
            raise ValueError("received_at cannot precede measured_at")
        if not self.known or not self.valid:
            # Unknown is represented by the mask, never by a fabricated zero.
            object.__setattr__(self, "value", None)


@dataclass(frozen=True)
class PublicSnapshot:
    decision_time: float
    fields: Mapping[str, PublicField]
    legal_actions: tuple[int, ...]
    history: tuple[Mapping[str, Any], ...] = ()
    continuation_ids: tuple[str, ...] = ()
    source: str = "w1-public"
    input_hash: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.decision_time)):
            raise ValueError("decision_time must be finite")
        actions = tuple(int(action) for action in self.legal_actions)
        if len(actions) != len(set(actions)) or any(action < 0 or action >= ACTION_COUNT for action in actions):
            raise ValueError("legal_actions must be unique and within the 25-action contract")
        if not actions:
            raise ValueError("legal_actions cannot be empty")
        for item in self.history:
            received = item.get("received_at")
            if received is not None and float(received) > self.decision_time:
                raise ValueError("future history is forbidden in a decision snapshot")
        if any(not isinstance(item, str) or not item for item in self.continuation_ids):
            raise ValueError("continuation identities must be non-empty strings")
        object.__setattr__(self, "legal_actions", actions)
        object.__setattr__(self, "fields", dict(self.fields))
        object.__setattr__(self, "history", tuple(copy.deepcopy(self.history)))
        object.__setattr__(self, "continuation_ids", tuple(self.continuation_ids))
        if not self.input_hash:
            payload = {
                "decision_time": self.decision_time,
                "fields": {key: vars(value) for key, value in sorted(self.fields.items())},
                "legal_actions": actions,
                "history": self.history,
                "continuation_ids": self.continuation_ids,
                "source": self.source,
            }
            object.__setattr__(self, "input_hash", public_hash(payload))

    def frozen_payload(self) -> dict[str, Any]:
        return {
            "decision_time": self.decision_time,
            "fields": {key: vars(value) for key, value in sorted(self.fields.items())},
            "legal_actions": list(self.legal_actions),
            "history": copy.deepcopy(list(self.history)),
            "continuation_ids": list(self.continuation_ids),
            "source": self.source,
            "input_hash": self.input_hash,
        }


@dataclass(frozen=True)
class TransitionTarget:
    next_snapshot: PublicSnapshot | None
    observed_events: Mapping[str, float | None]
    event_valid: Mapping[str, bool]
    physical_completion: bool | None = None
    host_confirmation: bool | None = None
    executed_action: int | None = None
    accepted: bool | None = None
    continuation_id: str = ""


@dataclass(frozen=True)
class PublicTransition:
    current: PublicSnapshot
    action: int
    target: TransitionTarget
    continuation_id: str
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise ValueError("unsupported transition contract")
        if self.action not in self.current.legal_actions:
            raise ValueError("action is not legal in frozen decision input")
        if self.target.executed_action is not None and self.target.executed_action != self.action:
            raise ValueError("executed action identity mismatch")
        if self.target.next_snapshot is not None and self.target.next_snapshot.decision_time < self.current.decision_time:
            raise ValueError("target decision time precedes current decision")
        if self.target.continuation_id and self.target.continuation_id != self.continuation_id:
            raise ValueError("continuation identity mismatch")


def freeze_decision_input(snapshot: PublicSnapshot) -> PublicSnapshot:
    """Return a detached snapshot whose content cannot be changed by branches."""
    return PublicSnapshot(
        decision_time=snapshot.decision_time,
        fields=copy.deepcopy(snapshot.fields),
        legal_actions=tuple(snapshot.legal_actions),
        history=copy.deepcopy(snapshot.history),
        continuation_ids=tuple(snapshot.continuation_ids),
        source=snapshot.source,
        input_hash=snapshot.input_hash,
    )


def validate_no_future(snapshot: PublicSnapshot, *, now: float | None = None) -> None:
    boundary = snapshot.decision_time if now is None else float(now)
    for name, field in snapshot.fields.items():
        if field.received_at is not None and field.received_at > boundary:
            raise ValueError(f"future field received_at for {name}")
        if field.measured_at is not None and field.received_at is not None and field.measured_at > field.received_at:
            raise ValueError(f"measurement after receipt for {name}")
    for item in snapshot.history:
        if item.get("received_at") is not None and float(item["received_at"]) > boundary:
            raise ValueError("future history item")


__all__ = [
    "ACTION_COUNT", "NOOP_ACTION", "CONTRACT_VERSION", "PublicField", "PublicSnapshot",
    "TransitionTarget", "PublicTransition", "freeze_decision_input", "public_hash", "validate_no_future",
]
