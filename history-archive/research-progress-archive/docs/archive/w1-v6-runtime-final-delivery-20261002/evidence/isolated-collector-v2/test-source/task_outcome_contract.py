"""Strict arrival-contract labels for one action plus fixed Hungarian continuation."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


CONTRACT_VERSION = "w1-action-conditioned-task-outcome/4.0.0"
EXPECTED_CONTINUATION = "hungarian-v1-fixed"
EXPECTED_COMPLETION_MODE = "arrival_to_region"
EXPECTED_DEADLINE_BASIS = "physical_arrival"
NOOP_ACTION = 24


class TaskOutcomeContractError(ValueError):
    pass


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TaskOutcomeContractError(f"{name}_not_numeric") from exc
    if not math.isfinite(result):
        raise TaskOutcomeContractError(f"{name}_nonfinite")
    return result


def _digest(value: Any, name: str) -> str:
    result = str(value or "")
    if len(result) != 64:
        raise TaskOutcomeContractError(f"{name}_invalid")
    try:
        bytes.fromhex(result)
    except ValueError as exc:
        raise TaskOutcomeContractError(f"{name}_invalid") from exc
    return result


def _masked(value: bool | None, valid: bool, source: str | None, reason: str | None) -> dict[str, Any]:
    return {"value": value, "valid": valid, "source": source, "reason": reason}


def derive_action_conditioned_task_outcome(
    decision: Mapping[str, Any], trajectory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive sequence labels without attributing continuation outcomes to the first command."""
    action = decision.get("action_id")
    legal = decision.get("legal_actions")
    if type(action) is not int or not isinstance(legal, (list, tuple)) or action not in legal:
        raise TaskOutcomeContractError("candidate_not_in_frozen_legal_set")
    input_hash = _digest(decision.get("public_input_hash"), "public_input_hash")
    continuation = str(decision.get("continuation_id", ""))
    if continuation != EXPECTED_CONTINUATION:
        raise TaskOutcomeContractError("fixed_hungarian_continuation_required")
    environment_sha256 = _digest(decision.get("environment_config_sha256"), "environment_config_sha256")
    mode = str(decision.get("task_completion_mode", ""))
    deadline_basis = str(decision.get("deadline_basis", ""))
    if mode != EXPECTED_COMPLETION_MODE or deadline_basis != EXPECTED_DEADLINE_BASIS:
        raise TaskOutcomeContractError("task_semantics_mismatch")
    if decision.get("completion_notice_available") is not True:
        raise TaskOutcomeContractError("arrival_contract_requires_completion_notice_capability")
    decision_time = _finite(decision.get("decision_time"), "decision_time")
    horizon = _finite(decision.get("horizon_time"), "horizon_time")
    if horizon <= decision_time:
        raise TaskOutcomeContractError("observation_horizon_not_after_decision")
    first = decision.get("first_command")
    if not isinstance(first, Mapping):
        raise TaskOutcomeContractError("first_command_record_missing")
    first_id = first.get("command_id")
    feedback = str(first.get("status", "unknown")).lower()
    if feedback not in {"accepted", "accepted_ack_lost", "rejected", "lost", "unknown", "noop"}:
        raise TaskOutcomeContractError("first_command_status_invalid")
    if not trajectory:
        raise TaskOutcomeContractError("post_action_trajectory_missing")

    previous_time = decision_time
    previous_step = -1
    for row in trajectory:
        if not isinstance(row, Mapping):
            raise TaskOutcomeContractError("trajectory_row_invalid")
        if str(row.get("continuation_id", "")) != continuation:
            raise TaskOutcomeContractError("continuation_identity_mismatch")
        at = _finite(row.get("time"), "trajectory_time")
        step = row.get("step")
        if type(step) is not int or step != previous_step + 1 or at <= previous_time or at > horizon:
            raise TaskOutcomeContractError("trajectory_order_or_time_invalid")
        previous_step, previous_time = step, at
    final = trajectory[-1]
    final_time = _finite(final.get("time"), "final_trajectory_time")
    terminal = final.get("terminated") is True
    horizon_reached = final.get("truncated") is True and final_time >= horizon
    if terminal and final.get("truncated") is True:
        raise TaskOutcomeContractError("ambiguous_episode_end")
    if not (terminal or horizon_reached):
        raise TaskOutcomeContractError("post_action_trajectory_not_closed")

    base = {
        "schema": CONTRACT_VERSION,
        "scope": "first_action_plus_fixed_continuation",
        "continuation_id": continuation,
        "candidate_id": str(decision.get("candidate_id", "")),
        "action_id": action,
        "environment_config_sha256": environment_sha256,
        "task_completion_mode": mode,
        "deadline_basis": deadline_basis,
        "public_input_hash": input_hash,
        "decision_time": decision_time,
        "horizon_time": horizon,
        "observation_cutoff_time": final_time,
        "first_command": {"command_id": first_id, "status": feedback},
    }
    if action == NOOP_ACTION:
        if decision.get("target_task_id") is not None or decision.get("target_uav_id") is not None:
            raise TaskOutcomeContractError("noop_must_not_have_task_target")
        return {
            **base,
            "target_task_id": None,
            "target_uav_id": None,
            "deadline": None,
            "physical_on_time_completion": _masked(None, False, None, "noop_has_no_target_task"),
            "task_expired": _masked(None, False, None, "noop_has_no_target_task"),
            "host_confirmation": _masked(None, False, None, "noop_has_no_target_task"),
            "valid_mask": {"physical_on_time_completion": False, "task_expired": False, "host_confirmation": False},
            "observation_end_status": "no_target_task",
            "initial_execution_identity": None,
            "completion_execution_identity": None,
            "completion_by_first_command": None,
            "action_specific_effect_identifiable": False,
        }

    task_id = str(decision.get("target_task_id", ""))
    uav_id = str(decision.get("target_uav_id", ""))
    if not task_id or not uav_id:
        raise TaskOutcomeContractError("candidate_task_identity_missing")
    initial = decision.get("initial_task")
    if not isinstance(initial, Mapping) or str(initial.get("task_id", "")) != task_id:
        raise TaskOutcomeContractError("initial_task_lifecycle_record_missing")
    deadline = _finite(initial.get("deadline"), "deadline")
    initial_state = str(initial.get("state", ""))
    if initial_state not in {"pending", "assigned", "serving"} or initial.get("completed_at") is not None:
        raise TaskOutcomeContractError("target_task_not_active_at_decision")
    initial_identity = initial.get("execution_identity")
    if initial_identity is not None and (
        not isinstance(initial_identity, Mapping)
        or str(initial_identity.get("task_id", "")) != task_id
    ):
        raise TaskOutcomeContractError("initial_execution_identity_mismatch")

    for row in trajectory:
        lifecycle = row.get("task_lifecycle")
        if not isinstance(lifecycle, Mapping) or str(lifecycle.get("task_id", "")) != task_id:
            raise TaskOutcomeContractError("target_task_lifecycle_missing")
        if _finite(lifecycle.get("deadline"), "trajectory_deadline") != deadline:
            raise TaskOutcomeContractError("task_deadline_changed")

    final_task = final["task_lifecycle"]
    state = str(final_task.get("state", ""))
    completed_at = final_task.get("completed_at")
    completed_at_value = None if completed_at is None else _finite(completed_at, "completed_at")
    if completed_at_value is not None and (state != "completed" or completed_at_value < decision_time or completed_at_value > final_time):
        raise TaskOutcomeContractError("completion_state_time_inconsistent")
    if state == "completed" and completed_at_value is None:
        raise TaskOutcomeContractError("completed_task_missing_physical_arrival_time")
    if state == "expired" and final_time < deadline:
        raise TaskOutcomeContractError("expiry_observed_before_deadline")
    if state not in {"pending", "assigned", "serving", "completed", "expired", "unreleased"}:
        raise TaskOutcomeContractError("task_lifecycle_state_invalid")

    completion_identity = final_task.get("execution_identity")
    if completion_identity is not None and (
        not isinstance(completion_identity, Mapping)
        or str(completion_identity.get("task_id", "")) != task_id
        or not str(completion_identity.get("command_id", ""))
        or not str(completion_identity.get("uav_id", ""))
    ):
        raise TaskOutcomeContractError("completion_execution_identity_mismatch")
    notice = final_task.get("completion_notice")
    if notice is not None and (not isinstance(notice, Mapping) or str(notice.get("task_id", "")) != task_id):
        raise TaskOutcomeContractError("completion_notice_identity_mismatch")

    physical = _masked(None, False, None, "observation_end_before_terminal_task_outcome")
    expired = _masked(None, False, None, "observation_end_before_terminal_task_outcome")
    status = "unresolved_at_observation_end"
    if completed_at_value is not None:
        if completed_at_value >= deadline:
            raise TaskOutcomeContractError("physical_arrival_not_before_deadline")
        physical = _masked(True, True, "TaskLifecycle.completed_at", None)
        physical["time"] = completed_at_value
        expired = _masked(False, True, "terminal_task_lifecycle_state", None)
        status = "arrived_on_time"
    elif state == "expired":
        physical = _masked(False, True, "terminal_task_lifecycle_state", None)
        expired = _masked(True, True, "TaskLifecycle.state", None)
        expired["time"] = deadline
        status = "expired"

    host = _masked(None, False, None, "completion_notice_receipt_not_observed")
    host_time = None if notice is None else notice.get("host_confirmation_time")
    notice_time = None if host_time is None else _finite(host_time, "host_confirmation_time")
    if notice_time is not None:
        if completed_at_value is None or notice_time < completed_at_value:
            raise TaskOutcomeContractError("host_confirmation_precedes_physical_arrival")
        observed_on_time = notice_time <= deadline
        declared = notice.get("host_confirmation_before_deadline")
        if type(declared) is not bool or declared is not observed_on_time:
            raise TaskOutcomeContractError("host_confirmation_record_inconsistent")
        host = _masked(observed_on_time, True, "completion_notice.received_at", None)
        host["time"] = notice_time
        host["message_id"] = notice.get("completion_message_id")
    elif state == "expired":
        host = _masked(False, True, "explicit_task_expiry_without_completion_notice", None)

    first_command_by_completion = None
    if completion_identity is not None and first_id is not None:
        initial_id = None if initial_identity is None else initial_identity.get("command_id")
        if initial_id is not None and str(initial_id) == str(first_id):
            first_command_by_completion = None
        elif feedback in {"accepted", "accepted_ack_lost"}:
            first_command_by_completion = str(completion_identity.get("command_id")) == str(first_id)
        elif feedback in {"rejected", "lost", "noop"}:
            first_command_by_completion = False

    return {
        **base,
        "target_task_id": task_id,
        "target_uav_id": uav_id,
        "deadline": deadline,
        "physical_on_time_completion": physical,
        "task_expired": expired,
        "host_confirmation": host,
        "valid_mask": {
            "physical_on_time_completion": bool(physical["valid"]),
            "task_expired": bool(expired["valid"]),
            "host_confirmation": bool(host["valid"]),
        },
        "observation_end_status": status,
        "initial_execution_identity": None if initial_identity is None else dict(initial_identity),
        "completion_execution_identity": None if completion_identity is None else dict(completion_identity),
        "completion_by_first_command": first_command_by_completion,
        "action_specific_effect_identifiable": bool(first_command_by_completion is True and initial_identity is None),
        "causal_note": "task outcome belongs to the legal first action followed by fixed Hungarian continuation; later execution is not relabeled as direct first-command effect",
    }
