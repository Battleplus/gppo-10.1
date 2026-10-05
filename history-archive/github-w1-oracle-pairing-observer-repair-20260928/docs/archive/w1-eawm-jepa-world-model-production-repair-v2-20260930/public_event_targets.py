"""Public event targets with explicit validity and time semantics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from public_transition_contract import PublicSnapshot, PublicTransition


EVENT_NAMES = (
    "public_field_change",
    "new_measurement_received",
    "continuation_publicly_confirmed",
    "physical_completion_observed",
    "host_confirmation_observed",
)


@dataclass(frozen=True)
class EventTarget:
    names: tuple[str, ...]
    values: tuple[float, ...]
    valid: tuple[bool, ...]
    measured_at: tuple[float | None, ...]
    received_at: tuple[float | None, ...]

    def as_arrays(self) -> tuple[list[float], list[bool]]:
        return list(self.values), list(self.valid)


def _field_pair(current: PublicSnapshot, following: PublicSnapshot, name: str):
    left, right = current.fields.get(name), following.fields.get(name)
    if left is None or right is None or not (left.known and left.valid and right.known and right.valid):
        return None
    return left, right


def build_event_target(transition: PublicTransition) -> EventTarget:
    """Create labels only from the post-decision public target and execution record.

    Unknown/missing fields remain invalid.  A later receipt is a public
    information event, not evidence that the physical state was unchanged.
    """
    current = transition.current
    target = transition.target
    following = target.next_snapshot
    values = [0.0] * len(EVENT_NAMES)
    valid = [False] * len(EVENT_NAMES)
    measured = [None] * len(EVENT_NAMES)
    received = [None] * len(EVENT_NAMES)
    if following is not None:
        current_names = set(current.fields)
        following_names = set(following.fields)
        comparable = []
        for name in sorted(current_names & following_names):
            pair = _field_pair(current, following, name)
            if pair is not None:
                comparable.append(pair)
                left, right = pair
                if left.value != right.value:
                    values[0] = 1.0
                    measured[0] = right.measured_at
                    received[0] = right.received_at
        if current_names == following_names and len(comparable) == len(current_names) and current_names:
            valid[0] = True
        # A field first visible after the action is not a negative receipt
        # label. It is positive only when its receipt is after the decision
        # boundary; otherwise its arrival timing is unknown.
        newly_received = []
        for name, right in following.fields.items():
            if not (right.known and right.valid and right.received_at is not None):
                continue
            left = current.fields.get(name)
            if right.received_at > current.decision_time and (
                left is None or left.received_at is None or right.received_at > left.received_at
            ):
                newly_received.append(right)
        if newly_received:
            valid[1] = True
            values[1] = 1.0
            measured[1] = newly_received[0].measured_at
            received[1] = max(float(item.received_at) for item in newly_received)
        elif current_names == following_names and len(comparable) == len(current_names) and current_names:
            valid[1] = True
        valid[2] = True
        values[2] = float(set(following.continuation_ids) != set(current.continuation_ids))
        received[2] = following.decision_time
    if target.physical_completion is not None:
        valid[3] = True
        values[3] = float(bool(target.physical_completion))
    if target.host_confirmation is not None:
        valid[4] = True
        values[4] = float(bool(target.host_confirmation))
    return EventTarget(tuple(EVENT_NAMES), tuple(values), tuple(valid), tuple(measured), tuple(received))


__all__ = ["EVENT_NAMES", "EventTarget", "build_event_target"]
