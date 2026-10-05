"""Strict, offline-safe task outcome label contract.

This module does not call an environment. It only validates a future recorded
branch trajectory and derives labels when the evidence is explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


NOOP_ACTION = 24
ACCEPTED = {"accepted", "awaiting_ack", "reuse_existing", "noop"}
REJECTED = {"rejected", "lost", "corrupt", "timeout", "not_executed"}
EVENT_NAMES = (
    "physical_completion_observed",
    "physical_completion_on_time",
    "host_confirmation_observed",
    "host_confirmation_on_time",
    "task_expired_observed",
)


class LabelContractError(ValueError):
    """Raised when a recorded branch cannot satisfy the causal label contract."""


@dataclass(frozen=True)
class LabelResult:
    values: tuple[float | None, ...]
    valid: tuple[bool, ...]
    reasons: tuple[str, ...]
    accepted: bool | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_names": list(EVENT_NAMES),
            "values": list(self.values),
            "valid": list(self.valid),
            "reasons": list(self.reasons),
            "accepted": self.accepted,
        }


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LabelContractError(f"{name}_not_numeric") from exc
    if not math.isfinite(number):
        raise LabelContractError(f"{name}_nonfinite")
    return number


def _accepted(feedback: Any) -> bool | None:
    if feedback is None:
        return None
    value = str(feedback).lower()
    if value in ACCEPTED:
        return True
    if value in REJECTED:
        return False
    return None


def derive_task_outcome(record: Mapping[str, Any]) -> LabelResult:
    """Derive explicit labels from one candidate's post-action recording.

    Absence of a record is always unknown. A NOOP has no target task, so all
    task-specific labels remain masked. Future observations are permitted only
    on the target side, while their identity and continuation must be fixed.
    """
    action = int(record.get("action_id", -1))
    if action < 0 or action > NOOP_ACTION:
        raise LabelContractError("action_outside_fixed_space")
    if not str(record.get("public_input_hash", "")):
        raise LabelContractError("public_input_hash_missing")
    legal = record.get("legal_actions")
    if not isinstance(legal, (list, tuple)) or action not in legal:
        raise LabelContractError("action_not_in_frozen_legal_set")
    decision_time = _finite(record.get("decision_time"), "decision_time")
    continuation_id = str(record.get("continuation_id", ""))
    if not continuation_id:
        raise LabelContractError("continuation_id_missing")

    trajectory = record.get("trajectory", ())
    if not isinstance(trajectory, (list, tuple)):
        raise LabelContractError("trajectory_not_sequence")
    for item in trajectory:
        if not isinstance(item, Mapping):
            raise LabelContractError("trajectory_item_invalid")
        if str(item.get("continuation_id", "")) != continuation_id:
            raise LabelContractError("continuation_id_mismatch")
        if "time" in item and _finite(item["time"], "trajectory_time") < decision_time:
            raise LabelContractError("post_action_time_precedes_decision")
    if trajectory and trajectory[-1].get("truncated") is True and _finite(trajectory[-1].get("time"), "truncation_time") < 18.0:
        raise LabelContractError("truncation_precedes_frozen_horizon")

    reasons: list[str] = []
    values: list[float | None] = [None] * len(EVENT_NAMES)
    valid = [False] * len(EVENT_NAMES)
    if action == NOOP_ACTION:
        return LabelResult(tuple(values), tuple(valid), ("action_is_noop_no_target_task",), True)

    task_id = str(record.get("target_task_id", ""))
    if not task_id:
        raise LabelContractError("target_task_id_missing")
    command_id = str(record.get("first_command_id", ""))
    uav_id = str(record.get("target_uav_id", ""))
    if not command_id or not uav_id:
        raise LabelContractError("first_command_identity_missing")

    first_feedback = record.get("first_feedback")
    if first_feedback is None and trajectory:
        first_info = trajectory[0].get("info", {})
        if isinstance(first_info, Mapping):
            first_feedback = first_info.get("feedback")
    accepted = _accepted(first_feedback)
    if not trajectory or not (trajectory[-1].get("terminated") is True or trajectory[-1].get("truncated") is True):
        return LabelResult(tuple(values), tuple(valid), ("fixed_continuation_not_recorded_to_terminal_or_horizon",), accepted)

    completion_records = record.get("completion_records") or {}
    if not isinstance(completion_records, Mapping):
        raise LabelContractError("completion_records_invalid")
    completion = completion_records.get(task_id)
    service = record.get("service_completion_record")
    completion_source = service if isinstance(service, Mapping) else completion
    identity = completion_source.get("execution_identity") if isinstance(completion_source, Mapping) else None
    matched = isinstance(identity, Mapping) and (
        str(identity.get("command_id")) == command_id
        and str(identity.get("uav_id")) == uav_id
        and str(completion_source.get("task_id")) == task_id
    )
    physical_time = None
    if matched:
        physical_time = completion_source.get("completed_at", completion_source.get("physical_arrival_time"))
    if physical_time is not None:
        physical_time = _finite(physical_time, "physical_completion_time")
        deadline = _finite(completion_source.get("deadline"), "deadline")
        if physical_time < decision_time:
            raise LabelContractError("physical_completion_precedes_decision")
        values[0], valid[0] = 1.0, True
        values[1], valid[1] = float(physical_time <= deadline), True
    else:
        reasons.append("matched_physical_completion_record_missing")

    task_states = record.get("task_states") or {}
    state = str(task_states.get(task_id, "")) if isinstance(task_states, Mapping) else ""
    if state == "expired":
        values[4], valid[4] = 1.0, True
    elif state:
        reasons.append("task_not_explicitly_expired")
    else:
        reasons.append("task_state_missing")

    host_time = completion_source.get("host_confirmation_time") if matched else None
    communications = record.get("communication_log") or ()
    confirmed_message = False
    if isinstance(communications, (list, tuple)):
        message_id = completion_source.get("completion_message_id") if matched else None
        for item in communications:
            if not isinstance(item, Mapping):
                continue
            if (str(item.get("message_kind", "")) in {"completion", "completion_ack"}
                    and str(item.get("status", "")) == "received"
                    and message_id is not None and item.get("message_id") == message_id):
                received_at = item.get("received_at", item.get("time"))
                if received_at is not None and _finite(received_at, "host_received_at") >= decision_time:
                    confirmed_message = True
                    break
    if host_time is not None and confirmed_message:
        host_time = _finite(host_time, "host_confirmation_time")
        deadline = _finite(completion_source.get("deadline"), "deadline")
        if host_time < decision_time:
            raise LabelContractError("host_confirmation_precedes_decision")
        values[2], valid[2] = 1.0, True
        values[3], valid[3] = float(host_time <= deadline), True
    else:
        reasons.append("host_confirmation_record_missing_or_unconfirmed")

    if accepted is None:
        reasons.append("acceptance_unknown")
    return LabelResult(tuple(values), tuple(valid), tuple(dict.fromkeys(reasons)), accepted)


__all__ = ["EVENT_NAMES", "LabelContractError", "LabelResult", "derive_task_outcome"]
