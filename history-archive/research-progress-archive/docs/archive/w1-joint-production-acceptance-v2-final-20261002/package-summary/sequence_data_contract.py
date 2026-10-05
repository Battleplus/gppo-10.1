"""Pure validation for collected W1 sequence labels and return accounting."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


WINDOW_SCHEMA = "w1-world-model-complete-window/3.0.0"
SEQUENCE_EVIDENCE_SCHEMA = "w1-sequence-label-evidence/1.0.0"
EXPECTED_CONTINUATION = "hungarian-v1-fixed"
HORIZON_HEADS = ("physical_on_time_completion", "task_expired", "host_confirmation")
OUTCOME_DIM = 6


class SequenceDataContractError(ValueError):
    pass


def _fail(code: str) -> None:
    raise SequenceDataContractError(code)


def _mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _finite(value: Any, code: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SequenceDataContractError(code) from exc
    if not math.isfinite(result):
        _fail(code)
    return result


def _integer(value: Any, code: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(code)
    return value


def _close(actual: Any, expected: float, code: str, *, atol: float = 1e-6) -> None:
    value = _finite(actual, code)
    if not math.isclose(value, expected, rel_tol=1e-6, abs_tol=atol):
        _fail(code)


def _two_vector(value: Any, code: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise SequenceDataContractError(code) from exc
    if result.shape != (2,) or not np.isfinite(result).all():
        _fail(code)
    return result


def _validate_label_records(rows: Any, code: str) -> list[Mapping[str, Any]]:
    if not isinstance(rows, list):
        _fail(code)
    parsed = []
    seen = set()
    for row in rows:
        item = _mapping(row, code)
        task_id = str(item.get("task_id", ""))
        if not task_id or task_id in seen:
            _fail(code)
        seen.add(task_id)
        state = str(item.get("state", ""))
        if state not in {"pending", "assigned", "serving", "completed", "expired", "unreleased"}:
            _fail(code)
        deadline = _finite(item.get("deadline"), code)
        completed_at = item.get("completed_at")
        if completed_at is not None and _finite(completed_at, code) < 0:
            _fail(code)
        if state == "completed" and completed_at is None:
            _fail(code)
        if state != "completed" and completed_at is not None:
            _fail(code)
        if deadline < 0:
            _fail(code)
        parsed.append(item)
    return parsed


def _counts(value: Any, code: str) -> dict[str, int]:
    item = _mapping(value, code)
    if not {"completed", "expired", "rejected"}.issubset(item):
        _fail(code)
    result = {}
    for key, count in item.items():
        if type(count) is not int or count < 0:
            _fail(code)
        result[str(key)] = count
    return result


def _validate_step(row: Any, expected_step: int, *, kind: str, horizon: float) -> tuple[float, np.ndarray]:
    item = _mapping(row, "SEQUENCE_STEP_INVALID")
    if item.get("step") != expected_step or item.get("kind") != kind:
        _fail("SEQUENCE_STEP_ORDER_INVALID")
    if item.get("continuation_id") != EXPECTED_CONTINUATION:
        _fail("SEQUENCE_CONTINUATION_IDENTITY_MISMATCH")
    action = _integer(item.get("action_id"), "SEQUENCE_ACTION_INVALID")
    if action > 24:
        _fail("SEQUENCE_ACTION_INVALID")
    before_time = _finite(item.get("time_before"), "SEQUENCE_TIME_INVALID")
    after_time = _finite(item.get("time_after"), "SEQUENCE_TIME_INVALID")
    if after_time <= before_time or after_time > horizon:
        _fail("SEQUENCE_TIME_ORDER_INVALID")
    scalar = _finite(item.get("scalar_environment_reward"), "SEQUENCE_SCALAR_REWARD_INVALID")
    vector = _two_vector(item.get("vector_reward"), "SEQUENCE_VECTOR_REWARD_INVALID")
    counts_before = _counts(item.get("counts_before"), "SEQUENCE_COUNTS_INVALID")
    counts_after = _counts(item.get("counts_after"), "SEQUENCE_COUNTS_INVALID")
    energy_before = _finite(item.get("energy_before"), "SEQUENCE_ENERGY_INVALID")
    energy_after = _finite(item.get("energy_after"), "SEQUENCE_ENERGY_INVALID")
    if energy_before < 0 or energy_after < 0:
        _fail("SEQUENCE_ENERGY_INVALID")
    lifecycle_before = _validate_label_records(item.get("task_lifecycle_before"), "SEQUENCE_LIFECYCLE_INVALID")
    lifecycle_after = _validate_label_records(item.get("task_lifecycle_after"), "SEQUENCE_LIFECYCLE_INVALID")
    before_ids = {str(task["task_id"]) for task in lifecycle_before}
    after_ids = {str(task["task_id"]) for task in lifecycle_after}
    if before_ids != after_ids:
        _fail("SEQUENCE_LIFECYCLE_IDENTITY_CHANGED")
    if any(task.get("completed_at") is not None
           and float(task["completed_at"]) > after_time for task in lifecycle_after):
        _fail("SEQUENCE_COMPLETION_TIME_AFTER_OBSERVATION")
    if type(item.get("terminated")) is not bool or type(item.get("truncated")) is not bool:
        _fail("SEQUENCE_TERMINAL_FLAGS_INVALID")
    if item["terminated"] and item["truncated"]:
        _fail("SEQUENCE_TERMINAL_FLAGS_AMBIGUOUS")
    return scalar, vector


def _contains_forbidden_input_key(value: Any) -> bool:
    forbidden = {
        "task_lifecycle", "task_lifecycle_before", "task_lifecycle_after",
        "completion_records", "execution_identity", "sequence_label_evidence",
        "prefix_label_evidence", "sequence_return", "vector_reward",
        "scalar_environment_reward", "horizon_task_outcome_target",
        "horizon_task_outcome_valid", "task_outcome_target",
    }
    if isinstance(value, Mapping):
        return any(str(key) in forbidden or _contains_forbidden_input_key(item)
                   for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_input_key(item) for item in value)
    return False


def _validate_public_input(window: Mapping[str, Any]) -> float:
    decision = _mapping(window.get("decision_input"), "DECISION_INPUT_MISSING")
    if decision.get("frozen_before_candidate_branches") is not True:
        _fail("DECISION_INPUT_NOT_FROZEN")
    if _contains_forbidden_input_key(decision):
        _fail("LABEL_DATA_LEAKED_INTO_DECISION_INPUT")
    observation = _mapping(decision.get("observation"), "DECISION_OBSERVATION_MISSING")
    decision_time = _finite(observation.get("time"), "DECISION_TIME_INVALID")
    mask = observation.get("mask")
    if not isinstance(mask, (list, tuple)) or len(mask) != 25 or not bool(mask[24]):
        _fail("DECISION_MASK_INVALID")
    telemetry = decision.get("telemetry")
    if not isinstance(telemetry, list):
        _fail("DECISION_TELEMETRY_INVALID")
    for record in telemetry:
        item = _mapping(record, "DECISION_TELEMETRY_INVALID")
        measured = _finite(item.get("measured_at"), "DECISION_TELEMETRY_TIME_INVALID")
        received = _finite(item.get("received_at"), "DECISION_TELEMETRY_TIME_INVALID")
        if measured > received or received > decision_time:
            _fail("FUTURE_PUBLIC_INPUT")
    return decision_time


def _validate_horizon_labels(window: Mapping[str, Any], count: int) -> None:
    labels = window.get("task_outcome_target")
    values = window.get("horizon_task_outcome_target")
    masks = window.get("horizon_task_outcome_valid")
    if not all(isinstance(rows, list) and len(rows) == count for rows in (labels, values, masks)):
        _fail("HORIZON_TASK_TARGET_SHAPE_INVALID")
    for source, target_row, mask_row in zip(labels, values, masks):
        source = _mapping(source, "HORIZON_TASK_SOURCE_INVALID")
        if not isinstance(target_row, list) or len(target_row) != len(HORIZON_HEADS):
            _fail("HORIZON_TASK_TARGET_SHAPE_INVALID")
        if not isinstance(mask_row, list) or len(mask_row) != len(HORIZON_HEADS):
            _fail("HORIZON_TASK_MASK_SHAPE_INVALID")
        source_masks = _mapping(source.get("valid_mask"), "HORIZON_TASK_SOURCE_INVALID")
        for index, head in enumerate(HORIZON_HEADS):
            record = _mapping(source.get(head), "HORIZON_TASK_SOURCE_INVALID")
            valid = record.get("valid")
            value = record.get("value")
            if type(valid) is not bool or type(mask_row[index]) is not bool or mask_row[index] is not valid:
                _fail("HORIZON_TASK_MASK_MISMATCH")
            if source_masks.get(head) is not valid:
                _fail("HORIZON_TASK_MASK_MISMATCH")
            if valid:
                if type(value) is not bool or target_row[index] not in (0, 1, 0.0, 1.0):
                    _fail("HORIZON_TASK_VALUE_INVALID")
                if float(target_row[index]) != float(value):
                    _fail("HORIZON_TASK_VALUE_MISMATCH")
            elif value is not None or target_row[index] is not None:
                _fail("HORIZON_UNKNOWN_MUST_REMAIN_NULL")
        if source.get("target_task_id") is None and any(mask_row):
            _fail("NOOP_HORIZON_TARGET_MUST_BE_MASKED")


def validate_sequence_window(window: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a schema-3.0.0 collector window without importing model code."""
    window = _mapping(window, "WINDOW_INVALID")
    if window.get("schema") != WINDOW_SCHEMA:
        _fail("WINDOW_SCHEMA_UNSUPPORTED")
    if window.get("status") != "complete":
        _fail("WINDOW_NOT_COMPLETE")
    if window.get("continuation_id") != EXPECTED_CONTINUATION:
        _fail("FIXED_CONTINUATION_REQUIRED")
    config = _mapping(window.get("environment_config_identity"), "ENVIRONMENT_CONFIG_IDENTITY_MISSING")
    if config.get("task_completion_mode") != "arrival_to_region" or config.get("deadline_basis") != "physical_arrival":
        _fail("ENVIRONMENT_TASK_SEMANTICS_INVALID")
    input_hash = str(window.get("input_hash", ""))
    if len(input_hash) != 64:
        _fail("PUBLIC_INPUT_HASH_INVALID")
    try:
        bytes.fromhex(input_hash)
    except ValueError as exc:
        raise SequenceDataContractError("PUBLIC_INPUT_HASH_INVALID") from exc
    public_input_hash = str(window.get("public_input_hash", ""))
    if len(public_input_hash) != 64:
        _fail("PUBLIC_INPUT_HASH_INVALID")
    try:
        bytes.fromhex(public_input_hash)
    except ValueError as exc:
        raise SequenceDataContractError("PUBLIC_INPUT_HASH_INVALID") from exc
    if (window.get("input_hash_scope") != "graph5_nodes_history_relations_legal_actions_continuation"
            or window.get("public_input_hash_scope") != "decision_observation_received_telemetry_public_continuation_ids"):
        _fail("PUBLIC_INPUT_HASH_SCOPE_INVALID")
    if not str(window.get("source_exogenous_key", "")) or not str(window.get("runtime_exogenous_key", "")):
        _fail("SOURCE_IDENTITY_MISSING")
    decision_time = _validate_public_input(window)
    action_ids = window.get("actions")
    candidate_ids = window.get("candidate_ids")
    audit = window.get("candidate_branch_audit")
    if not isinstance(action_ids, list) or not action_ids or len(set(action_ids)) != len(action_ids):
        _fail("CANDIDATE_ACTIONS_INVALID")
    count = len(action_ids)
    if not all(isinstance(rows, list) and len(rows) == count for rows in (
        candidate_ids, audit, window.get("sequence_label_evidence"),
        window.get("sequence_return"), window.get("transparent_horizon_components"),
        window.get("transparent_components"), window.get("transparent_scores"),
        window.get("first_step_transparent_scores"),
        window.get("true_utility"), window.get("outcome_target"), window.get("outcome_valid"),
        window.get("state_target"), window.get("state_valid"),
        window.get("event_labels"), window.get("event_valid"), window.get("jepa_valid"),
    )):
        _fail("CANDIDATE_ARRAY_SHAPE_INVALID")
    observation = window["decision_input"]["observation"]
    legal = {index for index, allowed in enumerate(observation["mask"]) if bool(allowed)}
    if set(action_ids) != legal:
        _fail("CANDIDATES_DO_NOT_MATCH_FROZEN_LEGAL_SET")
    if window.get("candidate_ids") != [f"action-{action}" for action in action_ids]:
        _fail("CANDIDATE_ID_ACTION_MISMATCH")
    for state, state_mask, event, event_mask, jepa_mask in zip(
        window["state_target"], window["state_valid"], window["event_labels"],
        window["event_valid"], window["jepa_valid"],
    ):
        try:
            state_vector = np.asarray(state, dtype=np.float64).reshape(-1)
            event_vector = np.asarray(event, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise SequenceDataContractError("IMMEDIATE_TARGET_INVALID") from exc
        if (state_vector.shape != (128,) or not np.isfinite(state_vector).all()
                or not isinstance(state_mask, list) or len(state_mask) != 128
                or not isinstance(event_mask, list) or len(event_mask) != 5
                or event_vector.shape != (5,) or not np.isfinite(event_vector).all()
                or not isinstance(jepa_mask, list) or len(jepa_mask) != 64):
            _fail("IMMEDIATE_TARGET_SHAPE_INVALID")
        if any(type(value) is not bool for value in (*state_mask, *event_mask, *jepa_mask)):
            _fail("IMMEDIATE_TARGET_MASK_INVALID")
    try:
        task_capacity = int(config["config"]["task_capacity"])
        initial_total_energy = float(config["config"]["uav_count"] * config["config"]["initial_energy"])
        horizon_time = float(config["config"]["horizon"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SequenceDataContractError("REWARD_NORMALIZATION_INVALID") from exc
    if (task_capacity <= 0 or not math.isfinite(initial_total_energy) or initial_total_energy <= 0
            or not math.isfinite(horizon_time) or horizon_time <= decision_time):
        _fail("REWARD_NORMALIZATION_INVALID")
    score_weights = _mapping(window.get("transparent_score_weights"), "TRANSPARENT_SCORE_WEIGHTS_INVALID")
    score_task_scale = _finite(score_weights.get("task_component_scale"), "TRANSPARENT_SCORE_WEIGHTS_INVALID")
    score_task_preference = _finite(score_weights.get("task_preference"), "TRANSPARENT_SCORE_WEIGHTS_INVALID")
    score_energy_preference = _finite(score_weights.get("energy_preference"), "TRANSPARENT_SCORE_WEIGHTS_INVALID")
    if not math.isclose(score_task_preference + score_energy_preference, 1.0, abs_tol=1e-6):
        _fail("TRANSPARENT_SCORE_WEIGHTS_INVALID")
    prefix_evidence = _mapping(window.get("prefix_label_evidence"), "PREFIX_EVIDENCE_INVALID")
    if prefix_evidence.get("schema") != SEQUENCE_EVIDENCE_SCHEMA or not isinstance(prefix_evidence.get("steps"), list):
        _fail("PREFIX_EVIDENCE_INVALID")
    prefix_steps = prefix_evidence["steps"]
    prefix_end = None
    for index, raw_step in enumerate(prefix_steps):
        step = _mapping(raw_step, "PREFIX_STEP_INVALID")
        if step.get("step") != index or step.get("action_id") != 24:
            _fail("PREFIX_STEP_ORDER_INVALID")
        before = _finite(step.get("time_before"), "PREFIX_TIME_INVALID")
        after = _finite(step.get("time_after"), "PREFIX_TIME_INVALID")
        if after <= before or after > decision_time:
            _fail("PREFIX_TIME_ORDER_INVALID")
        if prefix_end is not None and not math.isclose(before, prefix_end, abs_tol=1e-6):
            _fail("PREFIX_TIME_DISCONTINUITY")
        _finite(step.get("scalar_environment_reward"), "PREFIX_REWARD_INVALID")
        _two_vector(step.get("vector_reward"), "PREFIX_REWARD_INVALID")
        _counts(step.get("counts_before"), "PREFIX_COUNTS_INVALID")
        _counts(step.get("counts_after"), "PREFIX_COUNTS_INVALID")
        _finite(step.get("energy_before"), "PREFIX_ENERGY_INVALID")
        _finite(step.get("energy_after"), "PREFIX_ENERGY_INVALID")
        _validate_label_records(step.get("task_lifecycle_before"), "PREFIX_LIFECYCLE_INVALID")
        _validate_label_records(step.get("task_lifecycle_after"), "PREFIX_LIFECYCLE_INVALID")
        _mapping(step.get("public_observation_before"), "PREFIX_PUBLIC_OBSERVATION_MISSING")
        _mapping(step.get("public_observation_after"), "PREFIX_PUBLIC_OBSERVATION_MISSING")
        prefix_end = after
    if prefix_steps and not math.isclose(prefix_end, decision_time, abs_tol=1e-6):
        _fail("PREFIX_DOES_NOT_END_AT_DECISION")
    _validate_horizon_labels(window, count)
    vector_returns = window.get("vector_reward")
    scalar_returns = window.get("scalar_environment_reward_return")
    if not isinstance(vector_returns, list) or len(vector_returns) != count:
        _fail("SEQUENCE_RETURN_SHAPE_INVALID")
    if not isinstance(scalar_returns, list) or len(scalar_returns) != count:
        _fail("SEQUENCE_RETURN_SHAPE_INVALID")
    outcome_targets = window["outcome_target"]
    outcome_masks = window["outcome_valid"]
    for index, (action, raw_audit, raw_evidence, raw_return) in enumerate(zip(
        action_ids, audit, window["sequence_label_evidence"], window["sequence_return"],
    )):
        candidate = _mapping(raw_audit, "CANDIDATE_AUDIT_INVALID")
        if candidate.get("action") != action or candidate.get("candidate_id") != candidate_ids[index]:
            _fail("CANDIDATE_AUDIT_IDENTITY_MISMATCH")
        task_source = _mapping(window["task_outcome_target"][index], "HORIZON_TASK_SOURCE_INVALID")
        if (task_source.get("candidate_id") != candidate_ids[index]
                or task_source.get("action_id") != action
                or task_source.get("continuation_id") != EXPECTED_CONTINUATION
                or task_source.get("public_input_hash") != public_input_hash
                or task_source.get("legal_actions") != action_ids
                or task_source.get("environment_config_sha256") != config.get("config_sha256")):
            _fail("HORIZON_TASK_SOURCE_IDENTITY_MISMATCH")
        public_ids = observation.get("public_entity_ids", {})
        if action == 24:
            if task_source.get("target_task_id") is not None or task_source.get("target_uav_id") is not None:
                _fail("FIRST_ACTION_TARGET_IDENTITY_MISMATCH")
        else:
            try:
                expected_uav = public_ids["uavs"][action // 6]
                expected_task = public_ids["tasks"][action % 6]
            except (KeyError, IndexError, TypeError) as exc:
                raise SequenceDataContractError("FIRST_ACTION_TARGET_IDENTITY_MISSING") from exc
            if (task_source.get("target_uav_id") != expected_uav
                    or task_source.get("target_task_id") != expected_task):
                _fail("FIRST_ACTION_TARGET_IDENTITY_MISMATCH")
        evidence = _mapping(raw_evidence, "SEQUENCE_EVIDENCE_INVALID")
        if evidence.get("schema") != SEQUENCE_EVIDENCE_SCHEMA or not isinstance(evidence.get("steps"), list) or not evidence["steps"]:
            _fail("SEQUENCE_EVIDENCE_INVALID")
        steps = evidence["steps"]
        if steps[0].get("kind") != "first_action" or steps[0].get("action_id") != action:
            _fail("FIRST_ACTION_MEMBERSHIP_INVALID")
        first_command = _mapping(task_source.get("first_command"), "FIRST_COMMAND_SOURCE_MISSING")
        if first_command.get("command_id") != steps[0].get("command_id"):
            _fail("FIRST_COMMAND_SOURCE_MISMATCH")
        scalars = []
        vectors = []
        previous_time = decision_time
        previous_counts = None
        previous_energy = None
        previous_lifecycle = None
        for step_index, raw_step in enumerate(steps):
            kind = "first_action" if step_index == 0 else "hungarian_continuation"
            scalar, vector = _validate_step(raw_step, step_index, kind=kind, horizon=horizon_time)
            step = raw_step
            if not math.isclose(float(step["time_before"]), previous_time, abs_tol=1e-6):
                _fail("SEQUENCE_TIME_DISCONTINUITY")
            if previous_counts is not None and step["counts_before"] != previous_counts:
                _fail("SEQUENCE_COUNTS_DISCONTINUITY")
            if previous_energy is not None and not math.isclose(float(step["energy_before"]), previous_energy, rel_tol=1e-6, abs_tol=1e-6):
                _fail("SEQUENCE_ENERGY_DISCONTINUITY")
            if previous_lifecycle is not None and step["task_lifecycle_before"] != previous_lifecycle:
                _fail("SEQUENCE_LIFECYCLE_DISCONTINUITY")
            cb, ca = step["counts_before"], step["counts_after"]
            expected_task = ((ca["completed"] - cb["completed"]) - (ca["expired"] - cb["expired"])) / float(task_capacity)
            expected_energy = -max(0.0, float(step["energy_before"]) - float(step["energy_after"])) / initial_total_energy
            _close(vector[0], expected_task, "STEP_TASK_REWARD_MISMATCH")
            _close(vector[1], expected_energy, "STEP_ENERGY_REWARD_MISMATCH")
            scalars.append(scalar)
            vectors.append(vector)
            previous_time = float(step["time_after"])
            previous_counts = step["counts_after"]
            previous_energy = float(step["energy_after"])
            previous_lifecycle = step["task_lifecycle_after"]
        if not (steps[-1].get("terminated") is True or steps[-1].get("truncated") is True):
            _fail("SEQUENCE_TRACE_NOT_CLOSED")
        if prefix_steps:
            prefix_last = prefix_steps[-1]
            if (prefix_last["counts_after"] != steps[0]["counts_before"]
                    or not math.isclose(float(prefix_last["energy_after"]), float(steps[0]["energy_before"]), abs_tol=1e-6)
                    or prefix_last["task_lifecycle_after"] != steps[0]["task_lifecycle_before"]):
                _fail("PREFIX_BRANCH_BOUNDARY_MISMATCH")
        vector_sum = np.sum(np.stack(vectors), axis=0, dtype=np.float64)
        scalar_sum = float(math.fsum(scalars))
        record = _mapping(raw_return, "SEQUENCE_RETURN_INVALID")
        normalization = _mapping(record.get("normalization"), "REWARD_NORMALIZATION_INVALID")
        recorded_energy_normalizer = _finite(
            normalization.get("initial_total_energy"), "REWARD_NORMALIZATION_INVALID",
        )
        if (normalization.get("task_capacity") != task_capacity
                or not math.isclose(recorded_energy_normalizer, initial_total_energy,
                                    rel_tol=1e-6, abs_tol=1e-6)):
            _fail("REWARD_NORMALIZATION_INVALID")
        _close(record.get("scalar_environment_reward_sum"), scalar_sum, "SEQUENCE_SCALAR_SUM_MISMATCH")
        returned_vector = _two_vector(record.get("vector_reward_sum"), "SEQUENCE_VECTOR_SUM_INVALID")
        if not np.allclose(returned_vector, vector_sum, rtol=1e-6, atol=1e-6):
            _fail("SEQUENCE_VECTOR_SUM_MISMATCH")
        _close(scalar_returns[index], scalar_sum, "SEQUENCE_SCALAR_SUM_MISMATCH")
        if not np.allclose(_two_vector(vector_returns[index], "SEQUENCE_VECTOR_SUM_INVALID"), vector_sum, rtol=1e-6, atol=1e-6):
            _fail("SEQUENCE_VECTOR_SUM_MISMATCH")
        weights = _mapping(record.get("utility_weights"), "SEQUENCE_UTILITY_WEIGHTS_INVALID")
        task_scale = _finite(weights.get("task_component_scale"), "SEQUENCE_UTILITY_WEIGHTS_INVALID")
        task_preference = _finite(weights.get("task_preference"), "SEQUENCE_UTILITY_WEIGHTS_INVALID")
        energy_preference = _finite(weights.get("energy_preference"), "SEQUENCE_UTILITY_WEIGHTS_INVALID")
        if not math.isclose(task_preference + energy_preference, 1.0, abs_tol=1e-6):
            _fail("SEQUENCE_UTILITY_WEIGHTS_INVALID")
        utility = task_scale * task_preference * float(vector_sum[0]) + energy_preference * float(vector_sum[1])
        _close(record.get("true_utility"), utility, "SEQUENCE_UTILITY_MISMATCH")
        _close(window["true_utility"][index], utility, "SEQUENCE_UTILITY_MISMATCH")
        components = _two_vector(record.get("transparent_horizon_components"), "TRANSPARENT_HORIZON_COMPONENTS_INVALID")
        if not np.allclose(_two_vector(window["transparent_horizon_components"][index], "TRANSPARENT_HORIZON_COMPONENTS_INVALID"), components, rtol=1e-6, atol=1e-6):
            _fail("TRANSPARENT_HORIZON_COMPONENTS_MISMATCH")
        expected_horizon_score = (score_task_scale * score_task_preference * float(components[0])
                                  + score_energy_preference * float(components[1]))
        _close(window["transparent_scores"][index], expected_horizon_score, "TRANSPARENT_SCORE_MISMATCH")
        first_components = _two_vector(window["transparent_components"][index], "TRANSPARENT_COMPONENTS_INVALID")
        expected_first_step_score = (score_task_scale * score_task_preference * float(first_components[0])
                                     + score_energy_preference * float(first_components[1]))
        _close(window["first_step_transparent_scores"][index], expected_first_step_score, "FIRST_STEP_TRANSPARENT_SCORE_MISMATCH")
        baseline = _mapping(record.get("transparent_horizon_baseline"), "TRANSPARENT_HORIZON_BASELINE_MISSING")
        assumption = str(baseline.get("assumption", "")).lower()
        if (baseline.get("source") != "frozen_public_graph5_and_public_action_score"
                or baseline.get("uses_private_truth") is not False
                or "public pending task" not in assumption
                or "on time" not in assumption
                or "zero additional energy" not in assumption):
            _fail("TRANSPARENT_HORIZON_BASELINE_PROVENANCE_INVALID")
        _integer(baseline.get("ideal_continuation_completions"), "TRANSPARENT_IDEAL_CONTINUATION_INVALID")
        if baseline["ideal_continuation_completions"] > task_capacity:
            _fail("TRANSPARENT_IDEAL_CONTINUATION_INVALID")
        if baseline.get("ideal_continuation_energy_cost") != 0.0:
            _fail("TRANSPARENT_IDEAL_CONTINUATION_INVALID")
        expected_baseline = np.asarray((
            first_components[0] + baseline["ideal_continuation_completions"] / float(task_capacity),
            first_components[1],
        ), dtype=np.float64)
        if not np.allclose(components, expected_baseline, rtol=1e-6, atol=1e-6):
            _fail("TRANSPARENT_HORIZON_COMPONENTS_MISMATCH")
        residuals = outcome_targets[index]
        masks = outcome_masks[index]
        if not isinstance(residuals, list) or len(residuals) != OUTCOME_DIM or not isinstance(masks, list) or len(masks) != OUTCOME_DIM:
            _fail("OUTCOME_TARGET_SHAPE_INVALID")
        for target_value, valid in zip(residuals, masks):
            if type(valid) is not bool:
                _fail("OUTCOME_TARGET_MASK_INVALID")
            if valid:
                _finite(target_value, "OUTCOME_TARGET_VALUE_INVALID")
            elif target_value is not None:
                _fail("OUTCOME_UNKNOWN_MUST_REMAIN_NULL")
        if masks[3] is not True or masks[4] is not True:
            _fail("SEQUENCE_RESIDUAL_MUST_BE_VALID")
        _close(residuals[3], float(vector_sum[1] - components[1]), "ENERGY_RESIDUAL_MISMATCH")
        _close(residuals[4], float(vector_sum[0] - components[0]), "TASK_RESIDUAL_MISMATCH")
    return {
        "schema": WINDOW_SCHEMA,
        "candidate_count": count,
        "prefix_steps": len(prefix_steps),
        "sequence_steps": [len(item["steps"]) for item in window["sequence_label_evidence"]],
        "input_time": decision_time,
        "returns_recomputed": True,
        "horizon_masks_checked": True,
    }


__all__ = [
    "EXPECTED_CONTINUATION", "HORIZON_HEADS", "OUTCOME_DIM", "SEQUENCE_EVIDENCE_SCHEMA",
    "SequenceDataContractError", "WINDOW_SCHEMA", "validate_sequence_window",
]
