"""Action-plus-fixed-continuation task labels from label-side audit traces."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


CONTRACT_VERSION = "w1-action-conditioned-task-outcome/3.0.0"
EXPECTED_CONTINUATION = "hungarian-v1-fixed"
EXPECTED_COMPLETION_MODE = "continuous_service_until_deadline"
EXPECTED_DEADLINE_BASIS = "physical_service"
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


def derive_action_conditioned_task_outcome(
    decision: Mapping[str, Any], trajectory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Label one task outcome under a first action followed by fixed Hungarian.

    The event heads describe the task outcome of the whole action sequence.
    First-command acceptance and the execution identity observed at completion
    are audit fields; they do not redefine the sequence outcome as a direct
    effect of the first command.
    """
    action = decision.get("action_id")
    if type(action) is not int or action not in decision.get("legal_actions", ()):
        raise TaskOutcomeContractError("candidate_not_in_frozen_legal_set")
    input_hash = str(decision.get("public_input_hash", ""))
    if len(input_hash) != 64:
        raise TaskOutcomeContractError("public_input_hash_invalid")
    try:
        bytes.fromhex(input_hash)
    except ValueError as exc:
        raise TaskOutcomeContractError("public_input_hash_invalid") from exc
    continuation = str(decision.get("continuation_id", ""))
    if continuation != EXPECTED_CONTINUATION:
        raise TaskOutcomeContractError("fixed_hungarian_continuation_required")
    environment_sha256 = str(decision.get("environment_config_sha256", ""))
    if len(environment_sha256) != 64:
        raise TaskOutcomeContractError("environment_config_sha256_invalid")
    try:
        bytes.fromhex(environment_sha256)
    except ValueError as exc:
        raise TaskOutcomeContractError("environment_config_sha256_invalid") from exc
    completion_mode = str(decision.get("task_completion_mode", ""))
    deadline_basis = str(decision.get("deadline_basis", ""))
    if completion_mode != EXPECTED_COMPLETION_MODE:
        raise TaskOutcomeContractError("task_completion_mode_mismatch")
    if deadline_basis != EXPECTED_DEADLINE_BASIS:
        raise TaskOutcomeContractError("deadline_basis_mismatch")
    if decision.get("completion_notice_available") is not False:
        raise TaskOutcomeContractError("completion_notice_availability_must_be_explicit")
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
        if (type(step) is not int or step != previous_step + 1
                or at <= previous_time or at < decision_time or at > horizon):
            raise TaskOutcomeContractError("trajectory_order_or_time_invalid")
        previous_step, previous_time = step, at
    if action == NOOP_ACTION:
        if decision.get("target_task_id") is not None:
            raise TaskOutcomeContractError("noop_must_not_have_task_target")
        return {
            "schema": CONTRACT_VERSION,
            "scope": "first_action_plus_fixed_continuation",
            "continuation_id": continuation,
            "candidate_id": str(decision.get("candidate_id", "")),
            "action_id": action,
            "environment_config_sha256": environment_sha256,
            "task_completion_mode": completion_mode,
            "deadline_basis": deadline_basis,
            "target_task_id": None,
            "target_uav_id": None,
            "public_input_hash": input_hash,
            "decision_time": decision_time,
            "horizon_time": horizon,
            "observation_cutoff_time": _finite(trajectory[-1].get("time"), "final_trajectory_time"),
            "first_command": {"command_id": first_id, "status": feedback},
            "physical_on_time_completion": {"value": None, "valid": False, "source": None, "reason": "noop_has_no_target_task"},
            "task_expired": {"value": None, "valid": False, "source": None, "reason": "noop_has_no_target_task"},
            "host_confirmation": {"value": None, "valid": False, "source": None, "reason": "completion_notice_unavailable_in_continuous_service_mode"},
            "valid_mask": {"physical_on_time_completion": False, "task_expired": False, "host_confirmation": False},
            "observation_end_status": "no_target_task",
            "completion_execution_identity": None,
            "initial_execution_identity": None,
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
    if initial_state not in {"pending", "assigned", "serving"}:
        raise TaskOutcomeContractError("target_task_not_active_at_decision")
    initial_completed = initial.get("completed_at")
    if initial_state in {"completed", "expired"} or initial_completed is not None:
        raise TaskOutcomeContractError("target_task_terminal_before_decision")

    for row in trajectory:
        lifecycle = row.get("task_lifecycle")
        if not isinstance(lifecycle, Mapping) or str(lifecycle.get("task_id", "")) != task_id:
            raise TaskOutcomeContractError("target_task_lifecycle_missing")
        row_deadline = _finite(lifecycle.get("deadline"), "trajectory_deadline")
        if row_deadline != deadline:
            raise TaskOutcomeContractError("task_deadline_changed")

    final = trajectory[-1]
    final_task = final["task_lifecycle"]
    final_time = _finite(final["time"], "final_trajectory_time")
    state = str(final_task.get("state", ""))
    completed_at = final_task.get("completed_at")
    completed_at_value = None if completed_at is None else _finite(completed_at, "completed_at")
    if completed_at_value is not None and completed_at_value < decision_time:
        raise TaskOutcomeContractError("completion_precedes_decision")
    if completed_at_value is not None and (state != "completed" or completed_at_value > final_time):
        raise TaskOutcomeContractError("completion_state_time_inconsistent")
    terminal = final.get("terminated") is True
    horizon_reached = final.get("truncated") is True and final_time >= horizon
    ended = terminal or horizon_reached
    if final.get("terminated") is True and final.get("truncated") is True:
        raise TaskOutcomeContractError("ambiguous_episode_end")

    source = "TaskLifecycle.completed_at"
    completion_identity = final_task.get("execution_identity")
    if completion_identity is not None and not isinstance(completion_identity, Mapping):
        raise TaskOutcomeContractError("completion_execution_identity_invalid")
    if completion_identity is not None and (
        str(completion_identity.get("task_id", "")) != task_id
        or not str(completion_identity.get("command_id", ""))
        or not str(completion_identity.get("uav_id", ""))
    ):
        raise TaskOutcomeContractError("completion_execution_identity_mismatch")
    initial_identity = initial.get("execution_identity")
    if initial_identity is not None and not isinstance(initial_identity, Mapping):
        raise TaskOutcomeContractError("initial_execution_identity_invalid")
    if initial_identity is not None and str(initial_identity.get("task_id", "")) != task_id:
        raise TaskOutcomeContractError("initial_execution_identity_mismatch")

    physical = {"value": None, "valid": False, "source": None, "reason": "observation_not_complete"}
    expired = {"value": None, "valid": False, "source": None, "reason": "observation_not_complete"}
    status = "observation_incomplete"
    if completed_at_value is not None:
        if completed_at_value > deadline:
            raise TaskOutcomeContractError("physical_completion_after_deadline")
        physical = {"value": True, "valid": True, "source": source, "time": completed_at_value, "reason": None}
        expired = {"value": False, "valid": True, "source": "terminal_task_lifecycle_state", "reason": None}
        status = "completed_on_time"
    elif state == "expired":
        if final_time < deadline:
            raise TaskOutcomeContractError("expiry_observed_before_deadline")
        physical = {"value": False, "valid": True, "source": "terminal_task_lifecycle_state", "reason": None}
        expired = {"value": True, "valid": True, "source": "TaskLifecycle.state", "time": deadline, "reason": None}
        status = "expired"
    elif ended:
        status = "unresolved_at_observation_end"
        physical["reason"] = "no_explicit_completion_before_end"
        expired["reason"] = "no_explicit_expiry_before_end"

    preexisting = initial_identity is not None
    completion_by_first = None
    if completion_identity is not None and first_id is not None:
        initial_id = None if initial_identity is None else initial_identity.get("command_id")
        if preexisting and initial_id is not None and str(initial_id) == str(first_id):
            completion_by_first = None
        elif feedback in {"accepted", "accepted_ack_lost"}:
            completion_by_first = str(completion_identity.get("command_id", "")) == str(first_id)
        elif feedback in {"rejected", "lost", "noop"}:
            completion_by_first = False
    return {
        "schema": CONTRACT_VERSION,
        "scope": "first_action_plus_fixed_continuation",
        "continuation_id": continuation,
        "candidate_id": str(decision.get("candidate_id", "")),
        "action_id": action,
        "environment_config_sha256": environment_sha256,
        "task_completion_mode": completion_mode,
        "deadline_basis": deadline_basis,
        "target_task_id": task_id,
        "target_uav_id": uav_id,
        "public_input_hash": input_hash,
        "decision_time": decision_time,
        "deadline": deadline,
        "horizon_time": horizon,
        "observation_cutoff_time": final_time,
        "first_command": {"command_id": first_id, "status": feedback},
        "physical_on_time_completion": physical,
        "task_expired": expired,
        "host_confirmation": {"value": None, "valid": False, "source": None, "reason": "completion_notice_unavailable_in_continuous_service_mode"},
        "valid_mask": {
            "physical_on_time_completion": bool(physical["valid"]),
            "task_expired": bool(expired["valid"]),
            "host_confirmation": False,
        },
        "observation_end_status": status,
        "completion_execution_identity": None if completion_identity is None else dict(completion_identity),
        "completion_by_first_command": completion_by_first,
        "initial_execution_identity": None if initial_identity is None else dict(initial_identity),
        "action_specific_effect_identifiable": bool(completion_by_first is True and not preexisting),
        "causal_note": "outcome belongs to the complete first-action-plus-continuation sequence; completion command identity is audit only",
    }
