"""Pure schema validation for conditional action-outcome records.

This module intentionally imports no runtime, environment, or model code.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable


SCHEMA = "w1-action-outcome-record/1.0.0"
ACTION_COUNT = 25
NOOP = 24
COMMAND_STATES = {
    "accepted",
    "accepted_then_ack_lost",
    "command_transport_lost",
    "executor_rejected",
    "resource_unavailable",
    "noop",
    "unknown",
}
TASK_STATES = {"completed", "expired", "unfinished", "interrupted", "damaged", "unknown"}
FORBIDDEN_INPUT_KEYS = {
    "hidden_state",
    "hidden_fault",
    "future_observation",
    "future_result",
    "communication_delta",
    "feedback",
    "completion_records",
    "exogenous_events",
}


class ContractError(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


def masked(value: Any, valid: bool, source: str | None, *, name: str) -> dict[str, Any]:
    if valid:
        if value is None or not source:
            raise ContractError(f"valid {name} requires value and source")
    elif value is not None:
        raise ContractError(f"invalid {name} must be null, not a filled value")
    return {"value": value, "valid": bool(valid), "source": source}


def unknown_labels(task_ids: Iterable[str]) -> dict[str, Any]:
    tasks = {
        task_id: {
            "state": masked(None, False, None, name=f"task[{task_id}].state"),
            "physical_on_time": masked(None, False, None, name=f"task[{task_id}].physical_on_time"),
            "completion_uav": masked(None, False, None, name=f"task[{task_id}].completion_uav"),
            "host_on_time": masked(None, False, None, name=f"task[{task_id}].host_on_time"),
            "host_confirmation_time": masked(None, False, None, name=f"task[{task_id}].host_confirmation_time"),
        }
        for task_id in task_ids
    }
    return {
        "first_command_state": masked(None, False, None, name="first_command_state"),
        "target_task": {
            "task_id": None,
            "physical_on_time": masked(None, False, None, name="target.physical_on_time"),
            "completed_by_current_uav": masked(None, False, None, name="target.completed_by_current_uav"),
            "completion_time": masked(None, False, None, name="target.completion_time"),
            "host_on_time": masked(None, False, None, name="target.host_on_time"),
        },
        "tasks": tasks,
        "energy_to_terminal": masked(None, False, None, name="energy_to_terminal"),
        "remaining_utility": masked(None, False, None, name="remaining_utility"),
        "terminal_time": masked(None, False, None, name="terminal_time"),
    }


def _check_masked(field: dict[str, Any], name: str, *, numeric: bool = False) -> None:
    if set(field) != {"value", "valid", "source"}:
        raise ContractError(f"{name} has invalid masked-field shape")
    valid = field["valid"]
    if not isinstance(valid, bool):
        raise ContractError(f"{name}.valid must be boolean")
    if valid:
        if field["value"] is None or not isinstance(field["source"], str) or not field["source"]:
            raise ContractError(f"valid {name} requires value and source")
        if numeric:
            _finite(field["value"], name)
    elif field["value"] is not None or field["source"] is not None:
        raise ContractError(f"unknown {name} must keep value and source null")


def _walk_forbidden(value: Any, path: str = "input") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_INPUT_KEYS:
                raise ContractError(f"forbidden input key {path}.{key}")
            _walk_forbidden(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden(child, f"{path}[{index}]")


def validate_record(record: dict[str, Any]) -> None:
    required = {
        "schema",
        "decision_id",
        "source",
        "input",
        "candidate",
        "legal_actions",
        "continuation_id",
        "provenance",
        "labels",
        "training_eligible",
        "eligibility_reasons",
    }
    if set(record) != required:
        raise ContractError(f"record fields differ: {sorted(set(record) ^ required)}")
    if record["schema"] != SCHEMA:
        raise ContractError("record schema mismatch")
    source = record["source"]
    decision_time = _finite(source["decision_time"], "decision_time")
    if source.get("split_role") != "development":
        raise ContractError("current records must remain development-only")
    if source.get("independent_validation") is not False:
        raise ContractError("development records cannot be independent validation")
    if not isinstance(record["training_eligible"], bool):
        raise ContractError("training_eligible must be boolean")
    if record["training_eligible"]:
        raise ContractError("existing development diagnostic records must not be training eligible")

    public_input = record["input"]
    _walk_forbidden(public_input)
    if _finite(public_input["current_public"]["observation_time"], "observation_time") > decision_time:
        raise ContractError("current public input is after decision")
    for item in public_input["public_history"]:
        if _finite(item["observation_time"], "history time") > decision_time:
            raise ContractError("future public history entered input")
    for item in public_input["own_action_history"]:
        if _finite(item["decision_time"], "own action time") >= decision_time:
            raise ContractError("current or future action entered input history")

    legal_actions = record["legal_actions"]
    if not isinstance(legal_actions, list) or not legal_actions:
        raise ContractError("legal_actions must be a nonempty list")
    if len(set(legal_actions)) != len(legal_actions) or any(
        not isinstance(action, int) or isinstance(action, bool) or not 0 <= action < ACTION_COUNT
        for action in legal_actions
    ):
        raise ContractError("invalid legal action set")
    candidate = record["candidate"]
    action = candidate["action"]
    if action not in legal_actions:
        raise ContractError("candidate is not legal")
    if action == NOOP:
        if candidate != {"action": NOOP, "kind": "NOOP", "uav_id": None, "task_id": None}:
            raise ContractError("NOOP descriptor must not contain a target")
    elif candidate.get("kind") != "assignment" or not candidate.get("uav_id") or not candidate.get("task_id"):
        raise ContractError("assignment descriptor is incomplete")

    continuation_id = record["continuation_id"]
    if not isinstance(continuation_id, str) or not continuation_id:
        raise ContractError("continuation_id is required")
    provenance = record["provenance"]
    if provenance["is_executed"] is False and provenance["label_scope"] != "unknown_unexecuted":
        raise ContractError("unexecuted candidate cannot inherit factual labels")
    if provenance["is_true_branch"] and continuation_id != "hungarian-v1-fixed":
        raise ContractError("sealed true branches require frozen Hungarian continuation identity")
    if not provenance["is_true_branch"] and continuation_id.startswith("hungarian-v1"):
        raise ContractError("factual trajectory cannot claim Hungarian branch continuation")

    labels = record["labels"]
    _check_masked(labels["first_command_state"], "first_command_state")
    if labels["first_command_state"]["valid"] and labels["first_command_state"]["value"] not in COMMAND_STATES:
        raise ContractError("unknown command-state category")
    for name in ("energy_to_terminal", "remaining_utility", "terminal_time"):
        _check_masked(labels[name], name, numeric=True)
    target = labels["target_task"]
    for name in ("physical_on_time", "completed_by_current_uav", "completion_time", "host_on_time"):
        _check_masked(target[name], f"target_task.{name}", numeric=name == "completion_time")
    if action == NOOP:
        if target["task_id"] is not None or any(target[name]["valid"] for name in target if name != "task_id"):
            raise ContractError("NOOP target labels must be masked, not zero-filled")
    elif target["task_id"] != candidate["task_id"]:
        raise ContractError("target-task identity differs from candidate")

    for task_id, task in labels["tasks"].items():
        for name in ("state", "physical_on_time", "completion_uav", "host_on_time", "host_confirmation_time"):
            _check_masked(task[name], f"tasks[{task_id}].{name}", numeric=name == "host_confirmation_time")
        if task["state"]["valid"] and task["state"]["value"] not in TASK_STATES:
            raise ContractError("unknown task-state category")

    if not provenance["is_executed"]:
        def any_valid(value: Any) -> bool:
            if isinstance(value, dict):
                if set(value) == {"value", "valid", "source"}:
                    return bool(value["valid"])
                return any(any_valid(child) for child in value.values())
            return False
        if any_valid(labels):
            raise ContractError("unexecuted candidate labels must remain unknown")


def assert_same_continuation(records: Iterable[dict[str, Any]]) -> str:
    values = {record["continuation_id"] for record in records}
    if len(values) != 1:
        raise ContractError(f"mixed continuation identities: {sorted(values)}")
    return next(iter(values))
