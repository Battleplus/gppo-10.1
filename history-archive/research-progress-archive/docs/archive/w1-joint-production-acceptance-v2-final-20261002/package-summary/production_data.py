"""Native W1 public-prefix collector for the frozen EAWM/JEPA matrix.

Only controller-received telemetry and the public Graph-5 observation enter a
window input.  A deep-copied environment branch is used solely to produce
post-decision targets and audit records.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import hmac
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from infra_io import durable_append_jsonl, durable_atomic_json
from public_history import CausalPublicHistory, history_vector as _causal_history_vector, public_state_vector
from public_transition_contract import PublicField, PublicSnapshot, PublicTransition, TransitionTarget
from public_event_targets import build_event_target
from task_outcome_contract import EXPECTED_CONTINUATION, derive_action_conditioned_task_outcome
from runtime_config_contract import validate_config_before_collection, write_runtime_environment_record
from sequence_data_contract import (
    HORIZON_HEADS, SEQUENCE_EVIDENCE_SCHEMA, SequenceDataContractError,
    validate_sequence_window,
)
from transparent_utility import transparent_horizon_components, transparent_utility_components

WINDOW_SCHEMA = "w1-world-model-complete-window/3.0.0"
NOOP_ACTION = 24
DECISION_MIN_TIME = 4.0
_SYNTHETIC_CONTEXT_SCHEMA = "w1-world-model-synthetic-integration-config/1.0.0"
_SYNTHETIC_CONTEXT_KEY = b"w1-world-model-synthetic-integration-v1"


class ProductionDataError(RuntimeError):
    pass


def _hash(value: Any) -> str:
    """Hash structured values or raw file bytes without a text re-encoding step."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return hashlib.sha256(bytes(value)).hexdigest()
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=lambda item: item.tolist() if hasattr(item, "tolist") else item)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _native_imports(source_root: Path) -> tuple[Any, ...]:
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    native = source_root / "native"
    if str(native) not in sys.path:
        sys.path.insert(0, str(native))
    from gppo_world.graph5 import graph5_from_m10_observation
    from gppo_world.joint_consequence_baseline import public_joint_action_score
    from gppo_world.joint_training import _vector_reward
    from gppo_world.m10_environment import M10Config, M10Environment, scenario_from_dict
    from gppo_world.telemetry import Telemetry
    from classical_baselines import ClassicalSelector, PublicDecisionAdapter
    return (graph5_from_m10_observation, public_joint_action_score,
            (M10Config, M10Environment, scenario_from_dict), Telemetry,
            PublicDecisionAdapter, ClassicalSelector, _vector_reward)


def _received_telemetry(env: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    view = getattr(env, "view", None)
    if view is None:
        raise ProductionDataError("PUBLIC_VIEW_MISSING")
    for store in (getattr(view, "_uavs", None), getattr(view, "_task_values", None)):
        latest = getattr(store, "_latest", None)
        if latest is None:
            raise ProductionDataError("RECEIVED_TELEMETRY_STORE_MISSING")
        for message in latest.values():
            records.append({
                "entity": str(message.entity), "field": str(message.field),
                "value": float(message.value), "measured_at": float(message.measured_at),
                "received_at": float(message.received_at), "sequence": int(message.sequence),
                "message_id": str(getattr(message, "message_id", "")),
            })
    records.sort(key=lambda item: (item["entity"], item["field"], item["sequence"]))
    return records


def _continuation_ids(_env: Any, observation: Mapping[str, Any]) -> tuple[str, ...]:
    # Derive continuation identity only from the public observation sidecar.
    ids = observation.get("public_entity_ids", {})
    uavs, tasks = tuple(ids.get("uavs", ())), tuple(ids.get("tasks", ()))
    identities = []
    for action in observation.get("continuation_actions", ()):
        action = int(action)
        if action < 0 or action >= 24 or action // 6 >= len(uavs) or action % 6 >= len(tasks):
            raise ProductionDataError("PUBLIC_CONTINUATION_IDENTITY_MISSING")
        identities.append(f"{uavs[action // 6]}::{tasks[action % 6]}")
    return tuple(sorted(identities))


def _history_vector(observation: Mapping[str, Any]) -> list[float]:
    try:
        return _causal_history_vector(observation)
    except (TypeError, ValueError) as exc:
        raise ProductionDataError("CAUSAL_PUBLIC_HISTORY_INVALID") from exc


def _nodes(graph: Any) -> dict[str, list[list[float]]]:
    return {str(name): _jsonable(value) for name, value in graph.nodes.items()}


def _active_count(observation: Mapping[str, Any]) -> float:
    return float(len(tuple(observation.get("continuation_actions", ()))))


def _outcome_target(action: int, observation: Mapping[str, Any], next_observation: Mapping[str, Any],
                    info: Mapping[str, Any], vector_reward: Any,
                    transparent_components: tuple[float, float]) -> tuple[list[Any], list[bool]]:
    feedback_value = info.get("feedback")
    feedback = str(feedback_value).lower() if feedback_value is not None else "unknown"
    if feedback in {"accepted", "awaiting_ack", "reuse_existing", "noop", "ack_lost_after_accept"}:
        accepted: bool | None = True
    elif feedback in {"rejected", "lost", "corrupt", "timeout", "expired", "not_executed"}:
        accepted = False
    else:
        accepted = None
    before_flat = np.asarray(observation.get("flat"), dtype=np.float32).reshape(-1)
    after_flat = np.asarray(next_observation.get("flat"), dtype=np.float32).reshape(-1)
    state_changed = None
    if before_flat.shape == after_flat.shape and before_flat.size and np.isfinite(before_flat).all() and np.isfinite(after_flat).all():
        state_changed = float(not np.array_equal(before_flat, after_flat))
    active_delta = _active_count(next_observation) - _active_count(observation)
    task_residual = float(vector_reward[0]) - float(transparent_components[0])
    energy_residual = float(vector_reward[1]) - float(transparent_components[1])
    if int(action) == NOOP_ACTION:
        # A NOOP has no target task. Preserve aggregate task/energy accounting while
        # masking task-specific acceptance, state-change and completion-time
        # labels instead of manufacturing zeros.
        values: list[Any] = [None, None, None, energy_residual, task_residual, active_delta]
        valid = [False, False, False, math.isfinite(energy_residual), math.isfinite(task_residual), True]
    else:
        values = [None if accepted is None else float(accepted), state_changed, None, energy_residual, task_residual, active_delta]
        valid = [accepted is not None, state_changed is not None, False, math.isfinite(energy_residual), math.isfinite(task_residual), True]
    return values, valid


def _public_snapshot(observation: Mapping[str, Any], telemetry: list[Mapping[str, Any]],
                     continuation_ids: tuple[str, ...]) -> PublicSnapshot:
    decision_time = float(observation["time"])
    fields = {}
    history = []
    for record in telemetry:
        entity, name = str(record["entity"]), str(record["field"])
        measured_at, received_at = float(record["measured_at"]), float(record["received_at"])
        age = max(0.0, decision_time - measured_at)
        fields[f"{entity}.{name}"] = PublicField(
            float(record["value"]), True, True, age, measured_at, received_at,
        )
        history.append({**dict(record), "known": True, "valid": True, "age": age})
    legal = tuple(index for index, value in enumerate(observation["mask"]) if bool(value))
    return PublicSnapshot(decision_time, fields, legal, tuple(history), continuation_ids,
                          source="w1-production-public-adapter")


def _event_target(current_observation: Mapping[str, Any], next_observation: Mapping[str, Any],
                  current_env: Any, next_env: Any, info: Mapping[str, Any],
                  action: int, continuation_id: str):
    current = _public_snapshot(current_observation, _received_telemetry(current_env),
                               _continuation_ids(current_env, current_observation))
    following = _public_snapshot(next_observation, _received_telemetry(next_env),
                                 _continuation_ids(next_env, next_observation))
    feedback = info.get("feedback")
    accepted = None if feedback is None or str(feedback).lower() == "unknown" else str(feedback).lower() in {
        "accepted", "awaiting_ack", "reuse_existing", "noop",
    }
    transition = PublicTransition(
        current=current, action=int(action), continuation_id=continuation_id,
        target=TransitionTarget(
            next_snapshot=following, observed_events={}, event_valid={},
            physical_completion=None, host_confirmation=None,
            executed_action=int(action), accepted=accepted,
            continuation_id=continuation_id,
        ),
    )
    target = build_event_target(transition)
    return target.as_arrays()


def _task_lifecycle_record(env: Any, task_id: str | None, *, include_communication: bool = True) -> dict[str, Any] | None:
    """Copy private lifecycle identity for labels and audit, never model inputs."""
    if task_id is None:
        return None
    task = getattr(getattr(env, "clock", None), "tasks", {}).get(task_id)
    if task is None:
        raise ProductionDataError("LABEL_TARGET_TASK_MISSING_FROM_LIFECYCLE")
    execution = getattr(env, "execution", None)
    token = getattr(execution, "task_tokens", {}).get(task_id) if execution is not None else None
    command = None
    if token is not None and execution is not None:
        command = next((candidate for candidate in execution.commands.values()
                        if candidate.task_id == task_id and candidate.token == token), None)
    identity = None if command is None else {
        "command_id": str(command.command_id), "task_id": str(command.task_id),
        "uav_id": str(command.uav_id), "token": int(command.token),
    }
    state = getattr(task.state, "value", task.state)
    completion_record = getattr(env, "_completion_records", {}).get(task_id)
    communication = getattr(env, "_communication_log", ())
    related_ids = set()
    if isinstance(completion_record, Mapping):
        for key in ("completion_message_id", "completion_notice_id", "message_id"):
            value = completion_record.get(key)
            if value is not None:
                related_ids.add(str(value))
    communication_audit = [
        _jsonable(item) for item in communication
        if str(item.get("task_id", "")) == str(task_id)
        or any(str(item.get(key, "")) in related_ids for key in ("identity", "message_id", "notice_id"))
    ] if include_communication else []
    return {
        "task_id": str(task.task_id), "state": str(state),
        "deadline": float(task.deadline),
        "completed_at": None if task.completed_at is None else float(task.completed_at),
        "assigned_uav": task.assigned_uav, "service": float(task.service),
        "execution_identity": identity,
        "completion_notice": None if completion_record is None else _jsonable(completion_record),
        "completion_notice_communication": communication_audit,
    }


def _all_task_lifecycle_records(env: Any) -> list[dict[str, Any]]:
    task_ids = tuple(sorted(getattr(getattr(env, "clock", None), "tasks", {})))
    return [_task_lifecycle_record(env, task_id, include_communication=False) for task_id in task_ids]


def _all_counts(info: Mapping[str, Any] | None = None) -> dict[str, int]:
    raw = {} if info is None else info.get("counts", {})
    if not isinstance(raw, Mapping):
        raise ProductionDataError("LABEL_COUNTS_INVALID")
    counts = {"completed": 0, "expired": 0, "rejected": 0}
    for key, value in raw.items():
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ProductionDataError("LABEL_COUNTS_INVALID") from exc
        if number < 0 or number != value:
            raise ProductionDataError("LABEL_COUNTS_INVALID")
        counts[str(key)] = number
    return counts


def _public_pending_task_count(graph: Any, *, action: int, own_on_time: bool, task_capacity: int) -> int:
    """Count decision-time visible pending tasks for the stated ideal surrogate."""
    try:
        task_nodes = np.asarray(graph.nodes.get("task", graph.nodes.get("Task", [])), dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProductionDataError("PUBLIC_TASK_NODES_INVALID") from exc
    if task_nodes.ndim != 2 or task_nodes.shape[1] < 23:
        return 0
    pending = []
    required_values = (0, 4, 8, 12, 20)
    for row in task_nodes:
        visible = all(row[offset + 1] > 0.5 and row[offset + 2] > 0.5 for offset in required_values)
        pending.append(bool(visible and row[20] > 0.5))
    if action != NOOP_ACTION and own_on_time and action % 6 < len(pending):
        pending[action % 6] = False
    remaining_capacity = max(0, task_capacity - int(action != NOOP_ACTION and own_on_time))
    return min(sum(pending), remaining_capacity)


class ProductionDataCollector:
    def __init__(self, root: Path, output: Path, matrix: Mapping[str, Any], ledger: Any,
                 boundary: Any, source_rows: list[Mapping[str, Any]] | None = None):
        self.root = Path(root)
        self.output = Path(output)
        self.matrix = dict(matrix)
        self.boundary = boundary
        self.parent_split = self._load_parent_split()
        self.ledger = ledger
        self.environment_construction_sequence = 0
        (self.graph5_from_observation, self.transparent_score, env_types,
         self.telemetry_type, self.public_adapter_type,
        self.classical_selector_type, self.vector_reward) = _native_imports(self.root)
        self.M10Config, self.M10Environment, self.scenario_from_dict = env_types
        self.environment_config_contract = json.loads(
            (self.root / "environment-config-contract.json").read_text(encoding="utf-8")
        )
        if self.environment_config_contract.get("schema") != "w1-environment-config-contract/1.0.0":
            raise ProductionDataError("ENVIRONMENT_CONFIG_CONTRACT_SCHEMA_INVALID")
        self.environment_config: Any | None = None
        self.environment_config_identity: dict[str, Any] | None = None
        self.tapes = source_rows if source_rows is not None else self._load_tapes()

    def _load_parent_split(self) -> dict[str, Any]:
        split_file = self.matrix.get("parent_split_file")
        if not isinstance(split_file, str) or Path(split_file).name != split_file:
            raise ProductionDataError("FROZEN_PARENT_SPLIT_PATH_INVALID")
        split_path = self.root / split_file
        if not split_path.is_file():
            raise ProductionDataError("FROZEN_PARENT_SPLIT_MISSING")
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ProductionDataError("FROZEN_PARENT_SPLIT_SCHEMA_INVALID")
        proposal_shape = (
            isinstance(payload.get("parents"), list)
            and all(isinstance(row, Mapping) and "proposed_split" in row for row in payload["parents"])
        )
        legacy_shape = payload.get("schema") == "w1-action-conditioned-task-outcome-parent-split/1.0.0"
        if not legacy_shape and not proposal_shape:
            raise ProductionDataError("FROZEN_PARENT_SPLIT_SCHEMA_INVALID")
        if payload.get("attempt", self.matrix.get("attempt")) != self.matrix.get("attempt"):
            raise ProductionDataError("FROZEN_PARENT_SPLIT_ATTEMPT_MISMATCH")
        inputs_path = self.root / "runtime-inputs.json"
        if inputs_path.is_file():
            inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
            payload_tape_sha = payload.get("source_train_tapes_sha256", payload.get("source_train_tape_sha256"))
            if payload_tape_sha != inputs.get("source_run", {}).get("train_tape_sha256"):
                raise ProductionDataError("FROZEN_PARENT_SPLIT_TAPE_IDENTITY_MISMATCH")
        parents = payload.get("parents")
        expected_count = self.matrix.get("parent_count")
        if not isinstance(parents, list) or type(expected_count) is not int or len(parents) != expected_count:
            raise ProductionDataError("FROZEN_PARENT_SPLIT_COUNT_MISMATCH")
        names = [str(row.get("parent", "")) for row in parents if isinstance(row, Mapping)]
        if len(names) != len(parents) or len(set(names)) != len(names) or any(not name for name in names):
            raise ProductionDataError("FROZEN_PARENT_SPLIT_IDENTITIES_INVALID")
        if proposal_shape:
            allowed_splits = {"train", "model_selection", "prediction_confirmation"}
            normalized = []
            integration = self._synthetic_split_profile()
            expected_counts = (integration["counts"] if integration is not None else {
                "train": 24, "model_selection": 8, "prediction_confirmation": 8,
            })
            expected_roles = [role for role in ("train", "model_selection", "prediction_confirmation")
                              for _ in range(expected_counts[role])]
            if len(expected_roles) != len(parents):
                raise ProductionDataError("PROPOSED_PARENT_SPLIT_COUNTS_INVALID")
            seen_scenarios: set[str] = set()
            seen_structures: set[str] = set()
            seen_exogenous: set[str] = set()
            for offset, row in enumerate(parents):
                split = str(row.get("proposed_split", ""))
                if integration is None:
                    expected_parent = f"train-{64 + offset:04d}"
                    if (split not in allowed_splits or row.get("repeat", 0) != 0
                            or str(row.get("parent", "")) != expected_parent
                            or type(row.get("tape_index")) is not int or row["tape_index"] != 64 + offset
                            or split != expected_roles[offset]
                            or row.get("generator_split", "train") != "train"
                            or row.get("condition", "W1") != "W1"
                            or row.get("overlaps_prior_registered_scenario", False) is not False
                            or row.get("overlaps_prior_registered_structure", False) is not False):
                        raise ProductionDataError("PROPOSED_PARENT_SPLIT_ASSIGNMENT_INVALID")
                else:
                    parent = str(row.get("parent", ""))
                    exogenous_key = str(row.get("exogenous_key", ""))
                    if (split not in allowed_splits or split != expected_roles[offset]
                            or row.get("repeat", 0) != 0 or not parent.startswith("synthetic-")
                            or exogenous_key != f"synthetic|{parent}|repeat-0"
                            or exogenous_key in seen_exogenous):
                        raise ProductionDataError("SYNTHETIC_PARENT_SPLIT_ASSIGNMENT_INVALID")
                    seen_exogenous.add(exogenous_key)
                for digest_name in ("scenario_sha256", "structural_sha256"):
                    digest = str(row.get(digest_name, ""))
                    try:
                        valid_digest = len(digest) == 64 and len(bytes.fromhex(digest)) == 32
                    except ValueError:
                        valid_digest = False
                    if not valid_digest:
                        if integration is None or digest_name != "structural_sha256":
                            raise ProductionDataError("PROPOSED_PARENT_SPLIT_DIGEST_INVALID")
                    elif integration is not None:
                        seen = seen_scenarios if digest_name == "scenario_sha256" else seen_structures
                        if digest.lower() in seen:
                            raise ProductionDataError("SYNTHETIC_PARENT_IDENTITY_NOT_UNIQUE")
                        seen.add(digest.lower())
                normalized.append({**dict(row), "collection_split": split})
            parents = normalized
            split_counts = {split: sum(row["collection_split"] == split for row in parents)
                            for split in sorted(allowed_splits)}
            declared_counts = payload.get("split_counts")
            if isinstance(declared_counts, Mapping) and dict(declared_counts) != split_counts:
                raise ProductionDataError("FROZEN_PARENT_SPLIT_COUNTS_INVALID")
            if split_counts != expected_counts:
                raise ProductionDataError("PROPOSED_PARENT_SPLIT_COUNTS_INVALID")
            matrix_splits = self.matrix.get("splits")
            if not isinstance(matrix_splits, Mapping) or set(matrix_splits) != allowed_splits:
                raise ProductionDataError("MATRIX_PARENT_SPLIT_REFERENCE_MISMATCH")
            for split_name, value in matrix_splits.items():
                expected_rows = [row for row in parents if row["collection_split"] == split_name]
                expected_names = [str(row["parent"]) for row in expected_rows]
                if integration is not None and not isinstance(value, list):
                    raise ProductionDataError("SYNTHETIC_MATRIX_PARENT_LIST_REQUIRED")
                if isinstance(value, str):
                    if value != f"{split_file}:parents":
                        raise ProductionDataError("MATRIX_PARENT_SPLIT_REFERENCE_MISMATCH")
                elif isinstance(value, list):
                    matrix_names = [str(row.get("parent", "")) if isinstance(row, Mapping) else str(row)
                                    for row in value]
                    if matrix_names != expected_names:
                        raise ProductionDataError("MATRIX_PARENT_SPLIT_REFERENCE_MISMATCH")
                    if integration is not None:
                        for matrix_row, split_row in zip(value, expected_rows):
                            if any(matrix_row.get(field) != split_row.get(field)
                                   for field in ("parent", "scenario_sha256", "exogenous_key", "proposed_split")):
                                raise ProductionDataError("SYNTHETIC_MATRIX_PARENT_IDENTITY_MISMATCH")
                else:
                    raise ProductionDataError("MATRIX_PARENT_SPLIT_REFERENCE_MISMATCH")
            expected_repeats = {split: 1 for split in allowed_splits}
            if integration is not None:
                self._validate_synthetic_matrix(expected_counts, integration)
        else:
            if self._synthetic_split_profile() is not None:
                raise ProductionDataError("SYNTHETIC_INTEGRATION_REQUIRES_PROPOSAL_SPLIT")
            parents = [{**dict(row), "collection_split": "label_qualification"} for row in parents]
            if self.matrix.get("splits") != {"label_qualification": f"{split_file}:parents"}:
                raise ProductionDataError("MATRIX_PARENT_SPLIT_REFERENCE_MISMATCH")
            expected_repeats = {"label_qualification": 1}
        if self.matrix.get("repeats") != expected_repeats:
            raise ProductionDataError("MATRIX_REPEAT_CONTRACT_MISMATCH")
        return {**dict(payload), "parents": parents, "split_counts": {
            split: sum(row["collection_split"] == split for row in parents)
            for split in expected_repeats
        }}

    def _synthetic_split_profile(self) -> dict[str, Any] | None:
        """Authenticate the bounded integration-only 2/2/8 collection profile."""
        contract_path = self.root / "launch-contract.json"
        if not contract_path.is_file():
            if getattr(getattr(self, "boundary", None), "integration_context", None) is not None:
                raise ProductionDataError("SYNTHETIC_LAUNCH_CONTRACT_MISSING")
            return None
        launch = json.loads(contract_path.read_text(encoding="utf-8"))
        integration_test = launch.get("integration_test") is True
        boundary = getattr(self, "boundary", None)
        context = getattr(boundary, "integration_context", None)
        if not integration_test:
            if context is not None:
                raise ProductionDataError("SYNTHETIC_CONTEXT_WITHOUT_INTEGRATION_LAUNCH")
            return None
        if launch.get("attempt") != self.matrix.get("attempt"):
            raise ProductionDataError("SYNTHETIC_LAUNCH_ATTEMPT_MISMATCH")
        if not isinstance(context, Mapping) or context.get("mode") != "synthetic_test":
            raise ProductionDataError("SYNTHETIC_EXECUTION_CONTEXT_MISSING")
        config = context.get("integration_config")
        if not isinstance(config, Mapping) or config.get("schema") != _SYNTHETIC_CONTEXT_SCHEMA:
            raise ProductionDataError("SYNTHETIC_INTEGRATION_CONFIG_INVALID")
        canonical = json.dumps(dict(config), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        authentication = hmac.new(_SYNTHETIC_CONTEXT_KEY, canonical, hashlib.sha256).hexdigest()
        if (context.get("integration_config_sha256") != digest
                or not hmac.compare_digest(str(context.get("authentication", "")), authentication)):
            raise ProductionDataError("SYNTHETIC_INTEGRATION_CONFIG_AUTHENTICATION_FAILED")
        required = {"schema", "training_window_count", "selection_window_count",
                    "confirmation_window_count", "maximum_epochs", "maximum_train_candidate_rows"}
        if set(config) != required:
            raise ProductionDataError("SYNTHETIC_INTEGRATION_CONFIG_FIELDS_INVALID")
        try:
            limits = {key: int(config[key]) for key in required - {"schema"}}
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProductionDataError("SYNTHETIC_INTEGRATION_CONFIG_BOUNDS_INVALID") from exc
        if (any(type(config[key]) is not int or limits[key] <= 0 for key in limits)
                or {key: limits[key] for key in ("training_window_count", "selection_window_count",
                                                  "confirmation_window_count")} != {
                    "training_window_count": 2, "selection_window_count": 2,
                    "confirmation_window_count": 8,
                }
                or limits["maximum_epochs"] != 1
                or limits["maximum_train_candidate_rows"] > 50):
            raise ProductionDataError("SYNTHETIC_INTEGRATION_CONFIG_OUT_OF_BOUNDS")
        return {"counts": {"train": 2, "model_selection": 2, "prediction_confirmation": 8},
                "limits": limits}

    def _validate_synthetic_matrix(self, counts: Mapping[str, int], integration: Mapping[str, Any]) -> None:
        matrix = self.matrix
        total = sum(counts.values())
        bounds = matrix.get("bounds", {})
        if (matrix.get("parent_count") != total or not isinstance(bounds, Mapping)
                or bounds.get("parents") != total):
            raise ProductionDataError("SYNTHETIC_MATRIX_PARENT_COUNT_MISMATCH")
        expected_fields = {
            "world_model_training_windows": "training_window_count",
            "world_model_selection_windows": "selection_window_count",
            "world_model_confirmation_windows": "confirmation_window_count",
            "world_model_maximum_epochs": "maximum_epochs",
            "world_model_maximum_train_candidate_rows": "maximum_train_candidate_rows",
        }
        for field, profile_field in expected_fields.items():
            if matrix.get(field) != integration["limits"][profile_field]:
                raise ProductionDataError("SYNTHETIC_MATRIX_LIMIT_MISMATCH:" + field)
        if matrix.get("world_model_maximum_updates_per_variant_seed") != (
                integration["limits"]["training_window_count"] * integration["limits"]["maximum_epochs"]):
            raise ProductionDataError("SYNTHETIC_MATRIX_UPDATE_LIMIT_MISMATCH")
        if matrix.get("repeats") != {split: 1 for split in counts}:
            raise ProductionDataError("SYNTHETIC_MATRIX_REPEAT_CONTRACT_MISMATCH")
        if (not isinstance(bounds, Mapping)
                or bounds.get("parents") != total
                or bounds.get("candidate_count_per_window_upper_including_noop") != 25
                or bounds.get("branch_steps_per_branch_upper") != 14
                or bounds.get("prefix_steps_per_parent_upper") != 4):
            raise ProductionDataError("SYNTHETIC_MATRIX_COLLECTION_BOUNDS_INVALID")
        gate_path = self.root / "data-gate.json"
        if not gate_path.is_file():
            raise ProductionDataError("SYNTHETIC_DATA_GATE_MISSING")
        data_gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if data_gate.get("split_window_counts") != dict(counts):
            raise ProductionDataError("SYNTHETIC_DATA_GATE_PARENT_COUNTS_MISMATCH")
        prediction_gates = matrix.get("prediction_gates")
        if (not isinstance(prediction_gates, Mapping)
                or prediction_gates.get("confirmation_parents") != counts["prediction_confirmation"]):
            raise ProductionDataError("SYNTHETIC_PREDICTION_GATE_PARENT_COUNT_MISMATCH")
        if integration["limits"]["maximum_train_candidate_rows"] < counts["train"]:
            raise ProductionDataError("SYNTHETIC_TRAINING_ROW_CAP_TOO_SMALL")

    def _checked_environment_config(self) -> Any:
        values = self.environment_config_contract.get("config")
        digest = self.environment_config_contract.get("config_sha256")
        if not isinstance(values, Mapping) or not isinstance(digest, str):
            raise ProductionDataError("ENVIRONMENT_CONFIG_CONTRACT_INCOMPLETE")
        if self.environment_config is None:
            config = self.M10Config(**dict(values))
            try:
                checked = validate_config_before_collection(config, values, expected_sha256=digest)
            except ValueError as exc:
                raise ProductionDataError(str(exc)) from exc
            if (checked["task_completion_mode"] != "arrival_to_region"
                    or checked["deadline_basis"] != "physical_arrival"):
                raise ProductionDataError("TASK_OUTCOME_CONFIG_SEMANTICS_NOT_SUPPORTED")
            checked["completion_notice_mode"] = str(config.completion_notice_mode)
            checked["completion_notice_available"] = True
            checked["completion_notice_source"] = "M10Environment._emit_completion_notices and delivered completion_notice records"
            if not self.output.is_dir():
                raise ProductionDataError("OUTPUT_DIRECTORY_NOT_INITIALIZED_BEFORE_CONFIG_RECORD")
            write_runtime_environment_record(self.output / "environment.json", config)
            self.environment_config = config
            self.environment_config_identity = checked
        return self.environment_config

    def _source_root(self) -> Path:
        inputs = json.loads((self.root / "runtime-inputs.json").read_text(encoding="utf-8"))
        source = inputs["source_run"]
        path = Path(source["wsl_root"] if sys.platform != "win32" else source["windows_root"])
        if not path.is_absolute():
            raise ProductionDataError("SOURCE_ROOT_NOT_ABSOLUTE")
        return path

    def _load_tapes(self) -> list[Mapping[str, Any]]:
        inputs = json.loads((self.root / "runtime-inputs.json").read_text(encoding="utf-8"))
        source_root = self._source_root()
        path = source_root / inputs["source_run"]["train_tape_file"]
        if not path.is_file():
            raise ProductionDataError("TRAIN_TAPE_MISSING")
        digest = _hash(path.read_bytes())
        if digest != inputs["source_run"]["train_tape_sha256"]:
            raise ProductionDataError("TRAIN_TAPE_DIGEST_MISMATCH")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ProductionDataError("TRAIN_TAPE_INVALID")
        return rows

    def _row_for(self, identity: Mapping[str, Any], repeat: int) -> Mapping[str, Any]:
        prefix = str(identity.get("exogenous_key", "")).rsplit("|repeat-", 1)[0]
        source_key = prefix + "|repeat-0"
        runtime_key = prefix + f"|repeat-{repeat}"
        candidates = [row for row in self.tapes if str(row.get("exogenous_key")) == source_key]
        if len(candidates) != 1:
            raise ProductionDataError(f"TAPE_IDENTITY_NOT_UNIQUE:{source_key}")
        row = candidates[0]
        if str(row.get("parent", "")) != str(identity.get("parent", "")):
            raise ProductionDataError(f"TAPE_PARENT_IDENTITY_MISMATCH:{identity.get('parent')}")
        if str(row.get("scenario_sha256", "")) != str(identity.get("scenario_sha256", "")):
            raise ProductionDataError(f"SCENARIO_DIGEST_MISMATCH:{identity.get('parent')}")
        return {**dict(row), "exogenous_key": runtime_key,
                "source_exogenous_key": source_key}

    def _account(self, name: str, amounts: Mapping[str, int], fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return self.ledger.call(name, dict(amounts), self.boundary.environment, fn, *args, **kwargs)

    def _construct_environment(self, config: Any, scenario: Any, exogenous_key: str,
                               identity: Mapping[str, Any], repeat: int, split: str) -> Any:
        self.environment_construction_sequence += 1
        sequence = self.environment_construction_sequence
        evidence_path = self.output / "environment-construction.jsonl"
        common = {
            "sequence": sequence,
            "parent": str(identity.get("parent", "")),
            "repeat": repeat,
            "split": split,
            "exogenous_key": exogenous_key,
        }
        durable_append_jsonl(evidence_path, {**common, "event": "attempted"})
        env = self._account(
            f"{split}.reset", {"resets_upper": 1}, self.M10Environment,
            config, scenario, exogenous_key=exogenous_key,
        )
        durable_append_jsonl(evidence_path, {**common, "event": "constructed"})
        return env

    def _collect_unit(self, identity: Mapping[str, Any], repeat: int, split: str) -> dict[str, Any]:
        row = self._row_for(identity, repeat)
        scenario = self.scenario_from_dict(row["scenario"])
        config = self._checked_environment_config()
        env = self._construct_environment(
            config, scenario, row["exogenous_key"], identity, repeat, split,
        )
        observation = self._account(f"{split}.reset_call", {}, env.reset)
        public_history = CausalPublicHistory()
        public_history.append(observation)
        adapter = self.public_adapter_type()
        previous_counts = {"completed": 0, "expired": 0}
        previous_evidence_counts = {"completed": 0, "expired": 0, "rejected": 0}
        previous_energy = float(config.uav_count * config.initial_energy)
        prefix_steps = 0
        prefix_label_steps: list[dict[str, Any]] = []
        done = False
        while float(observation.get("time", 0.0)) < DECISION_MIN_TIME and prefix_steps < 18:
            _public_prefix, effective_mask = adapter.prepare(observation)
            if len(effective_mask) != 25 or not bool(effective_mask[NOOP_ACTION]):
                raise ProductionDataError("PREFIX_NOOP_OUTSIDE_SHARED_FILTER")
            adapter.commit(NOOP_ACTION)
            observation_before = observation
            lifecycle_before = _all_task_lifecycle_records(env)
            counts_before = dict(previous_evidence_counts)
            energy_before = float(previous_energy)
            observation, scalar_reward, done, info = self._account(
                f"{split}.prefix_step", {"environment_steps": 1, "public_rule_decisions": 1},
                env.step, NOOP_ACTION,
            )
            public_history.append(observation)
            vector_reward, _consequence, previous_counts, previous_energy = self.vector_reward(
                info, previous_counts, previous_energy, config,
            )
            evidence_counts_after = _all_counts(info)
            prefix_label_steps.append({
                "step": prefix_steps, "action_id": NOOP_ACTION,
                "time_before": float(observation_before.get("time", 0.0)),
                "time_after": float(observation.get("time", info.get("time", 0.0))),
                "public_observation_before": _jsonable(observation_before),
                "public_observation_after": _jsonable(observation),
                "scalar_environment_reward": float(scalar_reward),
                "vector_reward": _jsonable(vector_reward),
                "counts_before": counts_before, "counts_after": evidence_counts_after,
                "energy_before": energy_before, "energy_after": float(previous_energy),
                "task_lifecycle_before": lifecycle_before,
                "task_lifecycle_after": _all_task_lifecycle_records(env),
                "terminated": bool(info.get("terminated")),
                "truncated": bool(info.get("truncated")),
            })
            previous_evidence_counts = evidence_counts_after
            prefix_steps += 1
            if done:
                break
        if done:
            return self._no_opportunity(identity, repeat, split, "episode_ended_before_decision", prefix_label_steps)
        if float(observation.get("time", 0.0)) < DECISION_MIN_TIME:
            return self._no_opportunity(identity, repeat, split, "prefix_ended_before_decision", prefix_label_steps)
        observation, effective_mask = adapter.prepare(observation)
        observation = dict(observation)
        observation["mask"] = list(effective_mask)
        observation = public_history.attach_current(observation)
        legal = [index for index, allowed in enumerate(effective_mask) if bool(allowed)]
        non_noop = [index for index in legal if index != NOOP_ACTION]
        if len(non_noop) < 2:
            return self._no_opportunity(identity, repeat, split, "fewer_than_two_legal_non_noop", prefix_label_steps)
        if str(self.matrix.get("candidate_continuation_id", "")) != EXPECTED_CONTINUATION:
            raise ProductionDataError("TASK_OUTCOME_REQUIRES_FROZEN_HUNGARIAN_CONTINUATION")
        snapshot_payload = {"observation": observation, "telemetry": _received_telemetry(env), "continuation": _continuation_ids(env, observation)}
        parent_hash = self._account(
            f"{split}.snapshot_capture", {"snapshot_captures": 1}, _hash, snapshot_payload,
        )
        graph = self.graph5_from_observation(observation)
        nodes = _nodes(graph)
        history = _history_vector(observation)
        relation_rows = _jsonable(graph.candidate_features)
        if not isinstance(relation_rows, list) or len(relation_rows) != 24:
            raise ProductionDataError("GRAPH_CANDIDATE_RELATION_SHAPE_INVALID")
        relations = [relation_rows[action] if action < NOOP_ACTION else [0.0] * 4
                     for action in legal]
        continuation = _continuation_ids(env, observation)
        candidate_ids = [f"action-{action}" for action in legal]
        scores = []
        first_step_scores = []
        transparent_components = []
        transparent_horizon_pairs = []
        transparent_horizon_metadata = []
        preference_config = self.matrix.get("preference_configuration", {})
        confirmation_preference = tuple(float(value) for value in preference_config.get("task_confirmation", ()))
        task_scale = float(preference_config.get("task_component_scale", float("nan")))
        if confirmation_preference != (0.8, 0.2) or task_scale != 0.5:
            raise ProductionDataError("PREDICTION_SCORING_PREFERENCE_MISMATCH")
        for action in legal:
            scored = self.transparent_score(graph, action)
            if not isinstance(scored, Mapping) or "score" not in scored:
                raise ProductionDataError("TRANSPARENT_SCORE_CONTRACT_INVALID")
            own_on_time = bool(scored.get("own_on_time_public", False))
            ideal_continuation_completions = _public_pending_task_count(
                graph, action=action, own_on_time=own_on_time,
                task_capacity=int(config.task_capacity),
            )
            components = transparent_utility_components(
                scored, task_capacity=config.task_capacity,
                initial_total_energy=float(config.uav_count * config.initial_energy),
            )
            horizon_components = transparent_horizon_components(
                scored, task_capacity=config.task_capacity,
                initial_total_energy=float(config.uav_count * config.initial_energy),
                ideal_continuation_completions=ideal_continuation_completions,
            )
            first_step_score = task_scale * confirmation_preference[0] * components[0] + confirmation_preference[1] * components[1]
            horizon_score = task_scale * confirmation_preference[0] * horizon_components[0] + confirmation_preference[1] * horizon_components[1]
            if not math.isfinite(first_step_score) or not math.isfinite(horizon_score):
                raise ProductionDataError("TRANSPARENT_SCORE_NONFINITE")
            first_step_scores.append(first_step_score)
            scores.append(horizon_score)
            transparent_components.append(components)
            transparent_horizon_pairs.append(horizon_components)
            transparent_horizon_metadata.append({
                "source": "frozen_public_graph5_and_public_action_score",
                "uses_private_truth": False,
                "public_pending_task_count": int(ideal_continuation_completions),
                "ideal_continuation_completions": int(ideal_continuation_completions),
                "ideal_continuation_energy_cost": 0.0,
                "assumption": "Every counted decision-time public pending task completes on time during ideal continuation at zero additional energy cost; this is an optimistic surrogate, not simulator truth or a learned forecast.",
            })
        candidates = []
        for action, candidate_id, transparent, first_transparent, components, horizon_components, horizon_metadata in zip(
            legal, candidate_ids, scores, first_step_scores, transparent_components,
            transparent_horizon_pairs, transparent_horizon_metadata,
        ):
            branch, branch_adapter = self._account(
                f"{split}.branch_snapshot_copy", {"branch_snapshot_copies": 1},
                lambda parent_env, parent_adapter: (deepcopy(parent_env), deepcopy(parent_adapter)),
                env, adapter,
            )
            public_ids = observation.get("public_entity_ids", {})
            task_ids = tuple(public_ids.get("tasks", ()))
            uav_ids = tuple(public_ids.get("uavs", ()))
            target_task_id = None if action == NOOP_ACTION else (
                task_ids[action % 6] if action % 6 < len(task_ids) else None
            )
            target_uav_id = None if action == NOOP_ACTION else (
                uav_ids[action // 6] if action // 6 < len(uav_ids) else None
            )
            if action != NOOP_ACTION and (target_task_id is None or target_uav_id is None):
                raise ProductionDataError("LEGAL_CANDIDATE_PUBLIC_IDENTITY_MISSING")
            initial_task = _task_lifecycle_record(branch, target_task_id)
            first_lifecycle_before = _all_task_lifecycle_records(branch)
            first_counts_before = dict(previous_evidence_counts)
            first_energy_before = float(previous_energy)
            branch_adapter.commit(int(action))
            next_observation, reward, terminated, info = self._account(
                f"{split}.candidate_branch", {
                    "branches": 1, "environment_steps": 1,
                    "forced_first_actions": 1, "candidate_scans": 1,
                },
                branch.step, int(action),
            )
            next_observation, next_effective_mask = branch_adapter.prepare(next_observation)
            next_observation = dict(next_observation)
            next_observation["mask"] = list(next_effective_mask)
            branch_hash = _hash({"observation": observation, "telemetry": _received_telemetry(env), "continuation": _continuation_ids(env, observation)})
            if branch_hash != parent_hash:
                raise ProductionDataError("PARENT_PUBLIC_STATE_MUTATED_BY_BRANCH")
            target_graph = self.graph5_from_observation(next_observation)
            vector_reward, _consequence, branch_reward_counts, branch_energy = self.vector_reward(
                info, previous_counts, previous_energy, config,
            )
            first_counts_after = _all_counts(info)
            sequence_steps = [{
                "step": 0, "kind": "first_action", "action_id": int(action),
                "continuation_id": EXPECTED_CONTINUATION,
                "time_before": float(observation["time"]),
                "time_after": float(info.get("time", next_observation["time"])),
                "scalar_environment_reward": float(reward),
                "vector_reward": _jsonable(vector_reward),
                "counts_before": first_counts_before, "counts_after": first_counts_after,
                "energy_before": first_energy_before, "energy_after": float(branch_energy),
                "task_lifecycle_before": first_lifecycle_before,
                "task_lifecycle_after": _all_task_lifecycle_records(branch),
                "terminated": bool(info.get("terminated")),
                "truncated": bool(info.get("truncated")),
                "command_id": info.get("command_id"),
                "feedback": str(info.get("feedback", "unknown")),
            }]
            scalar_rewards = [float(reward)]
            vector_rewards = [np.asarray(vector_reward, dtype=np.float64).reshape(2)]
            event_labels, event_valid = _event_target(
                observation, next_observation, env, branch, info, action,
                str(self.matrix["candidate_continuation_id"]),
            )
            first_info = info
            feedback = str(first_info.get("feedback", "unknown")).lower()
            if feedback in {"accepted", "awaiting_ack", "reuse_existing"}:
                first_status = "accepted"
            elif feedback == "ack_lost_after_accept":
                first_status = "accepted_ack_lost"
            elif feedback == "command_lost":
                first_status = "lost"
            elif feedback in {"unknown", "none"}:
                first_status = "unknown"
            elif feedback == "noop":
                first_status = "noop"
            else:
                first_status = "rejected"
            trajectory = [{
                "step": 0, "time": float(first_info.get("time", next_observation["time"])),
                "continuation_id": EXPECTED_CONTINUATION,
                "task_lifecycle": _task_lifecycle_record(branch, target_task_id),
                "terminated": bool(first_info.get("terminated")),
                "truncated": bool(first_info.get("truncated")),
                "command_id": first_info.get("command_id"),
                "feedback": feedback,
            }]
            selector = self.classical_selector_type("hungarian", preference=(0.8, 0.2))
            continuation_decisions = 0
            current_observation = next_observation
            current_mask = next_effective_mask
            branch_done = bool(terminated)
            branch_evidence_counts = dict(first_counts_after)
            max_continuation_decisions = int(math.ceil(
                max(0.0, float(config.horizon) - float(next_observation["time"]))
                / float(config.decision_interval)
            ))
            frozen_continuation_cap = int(self.matrix["bounds"]["continuation_decisions_per_branch_upper"])
            if max_continuation_decisions > frozen_continuation_cap:
                raise ProductionDataError("DERIVED_CONTINUATION_BOUND_EXCEEDS_FROZEN_MATRIX")
            while not branch_done:
                if continuation_decisions >= frozen_continuation_cap:
                    raise ProductionDataError("FIXED_CONTINUATION_EXCEEDED_NATIVE_STEP_BOUND")
                decision_observation = dict(current_observation)
                decision_observation["mask"] = list(current_mask)
                continuation_action, diagnostic = selector.choose(decision_observation, branch_adapter.memory)
                step_lifecycle_before = _all_task_lifecycle_records(branch)
                step_counts_before = dict(branch_evidence_counts)
                step_energy_before = float(branch_energy)
                step_time_before = float(current_observation["time"])
                branch_adapter.commit(int(continuation_action))
                following, continuation_scalar_reward, branch_done, continuation_info = self._account(
                    f"{split}.candidate_continuation",
                    {"environment_steps": 1, "public_rule_decisions": 1},
                    branch.step, int(continuation_action),
                )
                continuation_vector_reward, _consequence, branch_reward_counts, branch_energy = self.vector_reward(
                    continuation_info, branch_reward_counts, branch_energy, config,
                )
                branch_evidence_counts = _all_counts(continuation_info)
                sequence_steps.append({
                    "step": continuation_decisions + 1, "kind": "hungarian_continuation",
                    "action_id": int(continuation_action),
                    "continuation_id": EXPECTED_CONTINUATION,
                    "time_before": step_time_before,
                    "time_after": float(continuation_info.get("time", following["time"])),
                    "scalar_environment_reward": float(continuation_scalar_reward),
                    "vector_reward": _jsonable(continuation_vector_reward),
                    "counts_before": step_counts_before, "counts_after": dict(branch_evidence_counts),
                    "energy_before": step_energy_before, "energy_after": float(branch_energy),
                    "task_lifecycle_before": step_lifecycle_before,
                    "task_lifecycle_after": _all_task_lifecycle_records(branch),
                    "terminated": bool(continuation_info.get("terminated")),
                    "truncated": bool(continuation_info.get("truncated")),
                    "command_id": continuation_info.get("command_id"),
                    "feedback": str(continuation_info.get("feedback", "unknown")),
                    "selector_diagnostic": _jsonable(diagnostic),
                })
                scalar_rewards.append(float(continuation_scalar_reward))
                vector_rewards.append(np.asarray(continuation_vector_reward, dtype=np.float64).reshape(2))
                continuation_decisions += 1
                trajectory.append({
                    "step": continuation_decisions,
                    "time": float(continuation_info.get("time", following["time"])),
                    "continuation_id": EXPECTED_CONTINUATION,
                    "task_lifecycle": _task_lifecycle_record(branch, target_task_id),
                    "terminated": bool(continuation_info.get("terminated")),
                    "truncated": bool(continuation_info.get("truncated")),
                    "command_id": continuation_info.get("command_id"),
                    "feedback": str(continuation_info.get("feedback", "unknown")),
                    "selector": _jsonable(diagnostic),
                })
                if not branch_done:
                    current_observation, current_mask = branch_adapter.prepare(following)
            if not branch_done:
                raise ProductionDataError("FIXED_CONTINUATION_DID_NOT_REACH_NATIVE_END")
            decision_context = {
                "candidate_id": candidate_id, "action_id": int(action),
                "legal_actions": legal, "target_task_id": target_task_id,
                "target_uav_id": target_uav_id, "public_input_hash": parent_hash,
                "continuation_id": EXPECTED_CONTINUATION,
                "environment_config_sha256": str(self.environment_config_identity["config_sha256"]),
                "task_completion_mode": str(self.environment_config_identity["task_completion_mode"]),
                "deadline_basis": str(self.environment_config_identity["deadline_basis"]),
                "completion_notice_available": bool(self.environment_config_identity["completion_notice_available"]),
                "decision_time": float(observation["time"]),
                "horizon_time": float(config.horizon),
                "first_command": {"command_id": first_info.get("command_id"), "status": first_status},
                "initial_task": initial_task,
            }
            task_outcome_target = self.task_outcome_target(decision_context, trajectory)
            vector_return = np.sum(np.stack(vector_rewards), axis=0, dtype=np.float64)
            scalar_return = float(math.fsum(scalar_rewards))
            utility_weights = {
                "task_component_scale": task_scale,
                "task_preference": confirmation_preference[0],
                "energy_preference": confirmation_preference[1],
            }
            true_utility = (
                task_scale * confirmation_preference[0] * float(vector_return[0])
                + confirmation_preference[1] * float(vector_return[1])
            )
            sequence_return = {
                "scalar_environment_reward_sum": scalar_return,
                "vector_reward_sum": vector_return.tolist(),
                "true_utility": true_utility,
                "utility_weights": utility_weights,
                "normalization": {
                    "task_capacity": int(config.task_capacity),
                    "initial_total_energy": float(config.uav_count * config.initial_energy),
                },
                "task_counts_before": dict(first_counts_before),
                "task_counts_after": dict(branch_evidence_counts),
                "energy_before": first_energy_before,
                "energy_after": float(branch_energy),
                "step_count": len(sequence_steps),
                "transparent_horizon_components": list(horizon_components),
                "transparent_horizon_baseline": horizon_metadata,
            }
            outcome, outcome_valid = _outcome_target(
                action, observation, next_observation, info, vector_return, horizon_components,
            )
            horizon_values = [task_outcome_target[head]["value"] for head in HORIZON_HEADS]
            horizon_valid = [bool(task_outcome_target[head]["valid"]) for head in HORIZON_HEADS]
            candidates.append({
                "candidate_id": candidate_id, "action": int(action), "transparent_scores": float(transparent),
                "first_step_transparent_score": float(first_transparent),
                "transparent_components": list(components),
                "transparent_horizon_components": list(horizon_components),
                "transparent_horizon_baseline": horizon_metadata,
                "first_step_scalar_environment_reward": float(reward),
                "first_step_vector_reward": _jsonable(vector_rewards[0]),
                "scalar_environment_reward": scalar_return,
                "vector_reward": vector_return.tolist(),
                "true_utility": true_utility,
                "target_nodes": _nodes(target_graph), "outcome_target": outcome, "outcome_valid": outcome_valid,
                "event_labels": event_labels, "event_valid": event_valid,
                "task_outcome_target": task_outcome_target,
                "horizon_task_outcome_target": horizon_values,
                "horizon_task_outcome_valid": horizon_valid,
                "sequence_label_evidence": {
                    "schema": SEQUENCE_EVIDENCE_SCHEMA, "steps": sequence_steps,
                },
                "sequence_return": sequence_return,
                "post_action_trajectory": trajectory,
                "state_target": public_state_vector(next_observation).tolist(), "state_valid": [True] * 128,
                "branch_status": str(first_info.get("feedback", "unknown")),
                "branch_terminated": bool(branch_done), "branch_info": _jsonable(first_info),
            })
        frozen = {
            "schema": WINDOW_SCHEMA, "split": split, "parent": str(identity["parent"]), "repeat": int(repeat),
            "window_id": f"{split}:{identity['parent']}:repeat-{repeat}:step-{prefix_steps}",
            "status": "complete", "scenario_sha256": str(identity["scenario_sha256"]),
            "source_exogenous_key": str(row["source_exogenous_key"]),
            "runtime_exogenous_key": str(row["exogenous_key"]),
            "input_hash": _hash({"nodes": nodes, "history": history, "relations": relations, "legal": legal, "continuation": continuation}),
            "input_hash_scope": "graph5_nodes_history_relations_legal_actions_continuation",
            "public_input_hash": parent_hash,
            "public_input_hash_scope": "decision_observation_received_telemetry_public_continuation_ids",
            "decision_input": {
                "observation": _jsonable(observation),
                "telemetry": _received_telemetry(env),
                "continuation_ids": list(continuation),
                "frozen_before_candidate_branches": True,
            },
            "prefix_label_evidence": {
                "schema": SEQUENCE_EVIDENCE_SCHEMA, "steps": prefix_label_steps,
            },
            "environment_config_identity": _jsonable(self.environment_config_identity),
            "continuation_id": str(self.matrix["candidate_continuation_id"]), "candidate_ids": candidate_ids,
            "actions": legal, "transparent_scores": scores,
            "first_step_transparent_scores": first_step_scores,
            "transparent_score_weights": {
                "task_component_scale": task_scale,
                "task_preference": confirmation_preference[0],
                "energy_preference": confirmation_preference[1],
            },
            "current_nodes": nodes, "history": history,
            "transparent_components": [list(value) for value in transparent_components],
            "transparent_horizon_components": [list(value) for value in transparent_horizon_pairs],
            "true_utility": [row["true_utility"] for row in candidates],
            "scalar_environment_reward_return": [row["scalar_environment_reward"] for row in candidates],
            "vector_reward": [row["vector_reward"] for row in candidates],
            "relations": relations, "target_nodes": {name: [row["target_nodes"][name] for row in candidates] for name in nodes},
            "jepa_valid": [[True] * 64 for _ in candidates],
            "state_target": [row["state_target"] for row in candidates], "state_valid": [row["state_valid"] for row in candidates],
            "outcome_target": [row["outcome_target"] for row in candidates], "outcome_valid": [row["outcome_valid"] for row in candidates],
            "event_labels": [row["event_labels"] for row in candidates], "event_valid": [row["event_valid"] for row in candidates],
            "task_outcome_target": [row["task_outcome_target"] for row in candidates],
            "horizon_task_outcome_target": [row["horizon_task_outcome_target"] for row in candidates],
            "horizon_task_outcome_valid": [row["horizon_task_outcome_valid"] for row in candidates],
            "sequence_label_evidence": [row["sequence_label_evidence"] for row in candidates],
            "sequence_return": [row["sequence_return"] for row in candidates],
            "candidate_branch_audit": [{key: value for key, value in row.items() if key not in {
                "target_nodes", "state_target", "state_valid", "outcome_target", "outcome_valid",
                "event_labels", "event_valid", "task_outcome_target", "horizon_task_outcome_target",
                "horizon_task_outcome_valid", "sequence_label_evidence", "sequence_return",
            }} for row in candidates],
        }
        try:
            validate_sequence_window(frozen)
        except SequenceDataContractError as exc:
            raise ProductionDataError(f"SEQUENCE_DATA_CONTRACT_FAILED:{exc}") from exc
        return frozen

    @staticmethod
    def task_outcome_target(decision: Mapping[str, Any], trajectory: list[Mapping[str, Any]]) -> dict[str, Any]:
        """Production collector entry point for label-side lifecycle records."""
        return derive_action_conditioned_task_outcome(decision, trajectory)

    @staticmethod
    def _no_opportunity(identity: Mapping[str, Any], repeat: int, split: str, reason: str,
                        prefix_steps: list[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        return {"schema": WINDOW_SCHEMA, "split": split, "parent": str(identity["parent"]), "repeat": int(repeat),
                "window_id": f"{split}:{identity['parent']}:repeat-{repeat}", "status": "no_opportunity",
                "scenario_sha256": str(identity["scenario_sha256"]), "input_hash": "", "continuation_id": "",
                "candidate_ids": [], "reason": reason,
                "prefix_label_evidence": {"schema": SEQUENCE_EVIDENCE_SCHEMA, "steps": _jsonable(prefix_steps or [])}}

    def collect(self) -> dict[str, Any]:
        split_names = tuple(self.matrix.get("splits", {}))
        splits = {
            split: [row for row in self.parent_split["parents"] if row.get("collection_split") == split]
            for split in split_names
        }
        identities = []
        for split, parent_rows in splits.items():
            for identity in parent_rows:
                repeats = int(self.matrix.get("repeats", {}).get(split, 0))
                for repeat in range(repeats):
                    identities.append((split, identity, repeat))
        output_path = self.output / "world-model-windows.jsonl"
        if output_path.exists():
            raise ProductionDataError("WINDOW_OUTPUT_ALREADY_EXISTS_NO_RETRY")
        rows = []
        for split, identity, repeat in identities:
            row = self._collect_unit(identity, repeat, split)
            durable_append_jsonl(output_path, row)
            rows.append(row)
        qualified = [row for row in rows if row["split"] == "label_qualification" and row["status"] == "complete"]
        if not qualified:
            qualified = [row for row in rows if row["status"] == "complete"]
        parents_with_valid_outcomes = {
            row["parent"] for row in rows if row["status"] == "complete"
            if any(label.get("valid_mask", {}).get("physical_on_time_completion")
                   or label.get("valid_mask", {}).get("task_expired")
                   for label in row.get("task_outcome_target", []))
        }
        expected_parents = {str(row["parent"]) for row in self.parent_split["parents"]}
        observed_parents = {str(row["parent"]) for row in rows}
        expected_split_counts = {
            split: len(parent_rows) for split, parent_rows in splits.items()
        }
        observed_split_counts = {
            split: sum(row["split"] == split for row in rows)
            for split in split_names
        }
        complete_split_counts = {
            split: sum(row["split"] == split and row["status"] == "complete" for row in rows)
            for split in split_names
        }
        return {"coverage_pass": observed_parents == expected_parents and len(qualified) == len(expected_parents)
                    and observed_split_counts == expected_split_counts
                    and complete_split_counts == expected_split_counts
                    and parents_with_valid_outcomes == expected_parents,
                "qualified_parent_count": len({row["parent"] for row in qualified}),
                "parents_with_valid_physical_outcome_count": len(parents_with_valid_outcomes),
                "expected_parent_count": len(expected_parents),
                "missing_parent_ids": sorted(expected_parents - parents_with_valid_outcomes),
                "expected_split_counts": expected_split_counts,
                "observed_split_counts": observed_split_counts,
                "complete_split_counts": complete_split_counts,
                "windows": len(rows), "complete_windows": sum(row["status"] == "complete" for row in rows),
                "no_opportunity_windows": sum(row["status"] == "no_opportunity" for row in rows),
                "path": str(output_path), "source_tape_identity_verified": True,
                "decision_input_frozen_before_branches": True, "future_observation_in_input": False}


__all__ = ["ProductionDataCollector", "ProductionDataError", "WINDOW_SCHEMA"]
