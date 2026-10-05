"""Native W1 public-prefix collector for the frozen EAWM/JEPA matrix.

Only controller-received telemetry and the public Graph-5 observation enter a
window input.  A deep-copied environment branch is used solely to produce
post-decision targets and audit records.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
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

WINDOW_SCHEMA = "w1-world-model-complete-window/2.0.0"
NOOP_ACTION = 24
DECISION_MIN_TIME = 4.0


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
    if feedback in {"accepted", "awaiting_ack", "reuse_existing", "noop"}:
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


def _task_lifecycle_record(env: Any, task_id: str | None) -> dict[str, Any] | None:
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
    return {
        "task_id": str(task.task_id), "state": str(state),
        "deadline": float(task.deadline),
        "completed_at": None if task.completed_at is None else float(task.completed_at),
        "assigned_uav": task.assigned_uav, "service": float(task.service),
        "execution_identity": identity,
    }


class ProductionDataCollector:
    def __init__(self, root: Path, output: Path, matrix: Mapping[str, Any], ledger: Any,
                 boundary: Any, source_rows: list[Mapping[str, Any]] | None = None):
        self.root = Path(root)
        self.output = Path(output)
        self.matrix = matrix
        self.ledger = ledger
        self.boundary = boundary
        (self.graph5_from_observation, self.transparent_score, env_types,
         self.telemetry_type, self.public_adapter_type,
         self.classical_selector_type, self.vector_reward) = _native_imports(self._source_root())
        self.M10Config, self.M10Environment, self.scenario_from_dict = env_types
        self.environment_config_contract = json.loads(
            (self.root / "environment-config-contract.json").read_text(encoding="utf-8")
        )
        if self.environment_config_contract.get("schema") != "w1-environment-config-contract/1.0.0":
            raise ProductionDataError("ENVIRONMENT_CONFIG_CONTRACT_SCHEMA_INVALID")
        self.environment_config: Any | None = None
        self.environment_config_identity: dict[str, Any] | None = None
        self.tapes = source_rows if source_rows is not None else self._load_tapes()

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
            if (checked["task_completion_mode"] != "continuous_service_until_deadline"
                    or checked["deadline_basis"] != "physical_service"):
                raise ProductionDataError("TASK_OUTCOME_CONFIG_SEMANTICS_NOT_SUPPORTED")
            checked["completion_notice_mode"] = str(config.completion_notice_mode)
            checked["completion_notice_available"] = False
            checked["completion_notice_unavailable_reason"] = (
                "M10Environment._emit_completion_notices returns outside arrival_to_region mode"
            )
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
        if str(row.get("scenario_sha256", "")) != str(identity.get("scenario_sha256", "")):
            raise ProductionDataError(f"SCENARIO_DIGEST_MISMATCH:{identity.get('parent')}")
        return {**dict(row), "exogenous_key": runtime_key,
                "source_exogenous_key": source_key}

    def _account(self, name: str, amounts: Mapping[str, int], fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return self.ledger.call(name, dict(amounts), self.boundary.environment, fn, *args, **kwargs)

    def _collect_unit(self, identity: Mapping[str, Any], repeat: int, split: str) -> dict[str, Any]:
        from production_policy import transparent_utility_components
        row = self._row_for(identity, repeat)
        scenario = self.scenario_from_dict(row["scenario"])
        config = self._checked_environment_config()
        env = self._account(f"{split}.reset", {"resets_upper": 1}, self.M10Environment, config, scenario, exogenous_key=row["exogenous_key"])
        observation = self._account(f"{split}.reset_call", {}, env.reset)
        public_history = CausalPublicHistory()
        public_history.append(observation)
        adapter = self.public_adapter_type()
        previous_counts = {"completed": 0, "expired": 0}
        previous_energy = float(config.uav_count * config.initial_energy)
        prefix_steps = 0
        done = False
        while float(observation.get("time", 0.0)) < DECISION_MIN_TIME and prefix_steps < 18:
            _public_prefix, effective_mask = adapter.prepare(observation)
            if len(effective_mask) != 25 or not bool(effective_mask[NOOP_ACTION]):
                raise ProductionDataError("PREFIX_NOOP_OUTSIDE_SHARED_FILTER")
            adapter.commit(NOOP_ACTION)
            observation, _, done, info = self._account(f"{split}.prefix_step", {"environment_steps": 1, "public_rule_decisions": 1}, env.step, NOOP_ACTION)
            public_history.append(observation)
            _vector, _consequence, previous_counts, previous_energy = self.vector_reward(
                info, previous_counts, previous_energy, config,
            )
            prefix_steps += 1
            if done:
                break
        if done:
            return self._no_opportunity(identity, repeat, split, "episode_ended_before_decision")
        if float(observation.get("time", 0.0)) < DECISION_MIN_TIME:
            return self._no_opportunity(identity, repeat, split, "prefix_ended_before_decision")
        observation, effective_mask = adapter.prepare(observation)
        observation = dict(observation)
        observation["mask"] = list(effective_mask)
        observation = public_history.attach_current(observation)
        legal = [index for index, allowed in enumerate(effective_mask) if bool(allowed)]
        non_noop = [index for index in legal if index != NOOP_ACTION]
        if len(non_noop) < 2:
            return self._no_opportunity(identity, repeat, split, "fewer_than_two_legal_non_noop")
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
        transparent_components = []
        preference_config = self.matrix.get("preference_configuration", {})
        confirmation_preference = tuple(float(value) for value in preference_config.get("task_confirmation", ()))
        task_scale = float(preference_config.get("task_component_scale", float("nan")))
        if confirmation_preference != (0.8, 0.2) or task_scale != 0.5:
            raise ProductionDataError("PREDICTION_SCORING_PREFERENCE_MISMATCH")
        for action in legal:
            scored = self.transparent_score(graph, action)
            if not isinstance(scored, Mapping) or "score" not in scored:
                raise ProductionDataError("TRANSPARENT_SCORE_CONTRACT_INVALID")
            components = transparent_utility_components(
                scored, task_capacity=config.task_capacity,
                initial_total_energy=float(config.uav_count * config.initial_energy),
            )
            score = task_scale * confirmation_preference[0] * components[0] + confirmation_preference[1] * components[1]
            if not math.isfinite(score):
                raise ProductionDataError("TRANSPARENT_SCORE_NONFINITE")
            scores.append(score)
            transparent_components.append(components)
        candidates = []
        for action, candidate_id, transparent, components in zip(legal, candidate_ids, scores, transparent_components):
            branch = deepcopy(env)
            branch_adapter = deepcopy(adapter)
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
            vector_reward, _consequence, _counts, _energy = self.vector_reward(
                info, previous_counts, previous_energy, config,
            )
            outcome, outcome_valid = _outcome_target(
                action, observation, next_observation, info, vector_reward, components,
            )
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
            while not branch_done:
                if continuation_decisions >= 18:
                    raise ProductionDataError("FIXED_CONTINUATION_EXCEEDED_NATIVE_STEP_BOUND")
                decision_observation = dict(current_observation)
                decision_observation["mask"] = list(current_mask)
                continuation_action, diagnostic = selector.choose(decision_observation, branch_adapter.memory)
                branch_adapter.commit(int(continuation_action))
                following, _continuation_reward, branch_done, continuation_info = self._account(
                    f"{split}.candidate_continuation",
                    {"environment_steps": 1, "public_rule_decisions": 1},
                    branch.step, int(continuation_action),
                )
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
            preference = (0.8, 0.2)
            true_utility = 0.5 * preference[0] * float(vector_reward[0]) + preference[1] * float(vector_reward[1])
            candidates.append({
                "candidate_id": candidate_id, "action": int(action), "transparent_scores": float(transparent),
                "transparent_components": list(components),
                "scalar_environment_reward": float(reward),
                "vector_reward": _jsonable(vector_reward),
                "true_utility": true_utility,
                "target_nodes": _nodes(target_graph), "outcome_target": outcome, "outcome_valid": outcome_valid,
                "event_labels": event_labels, "event_valid": event_valid,
                "task_outcome_target": task_outcome_target,
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
            "decision_input": {
                "observation": _jsonable(observation),
                "telemetry": _received_telemetry(env),
                "continuation_ids": list(continuation),
                "frozen_before_candidate_branches": True,
            },
            "environment_config_identity": _jsonable(self.environment_config_identity),
            "continuation_id": str(self.matrix["candidate_continuation_id"]), "candidate_ids": candidate_ids,
            "actions": legal, "transparent_scores": scores, "current_nodes": nodes, "history": history,
            "transparent_components": [list(value) for value in transparent_components],
            "true_utility": [row["true_utility"] for row in candidates],
            "relations": relations, "target_nodes": {name: [row["target_nodes"][name] for row in candidates] for name in nodes},
            "jepa_valid": [[True] * 64 for _ in candidates],
            "state_target": [row["state_target"] for row in candidates], "state_valid": [row["state_valid"] for row in candidates],
            "outcome_target": [row["outcome_target"] for row in candidates], "outcome_valid": [row["outcome_valid"] for row in candidates],
            "event_labels": [row["event_labels"] for row in candidates], "event_valid": [row["event_valid"] for row in candidates],
            "task_outcome_target": [row["task_outcome_target"] for row in candidates],
            "candidate_branch_audit": [{key: value for key, value in row.items() if key not in {"target_nodes", "state_target", "state_valid", "outcome_target", "outcome_valid", "event_labels", "event_valid", "task_outcome_target"}} for row in candidates],
        }
        return frozen

    @staticmethod
    def task_outcome_target(decision: Mapping[str, Any], trajectory: list[Mapping[str, Any]]) -> dict[str, Any]:
        """Production collector entry point for label-side lifecycle records."""
        return derive_action_conditioned_task_outcome(decision, trajectory)

    @staticmethod
    def _no_opportunity(identity: Mapping[str, Any], repeat: int, split: str, reason: str) -> dict[str, Any]:
        return {"schema": WINDOW_SCHEMA, "split": split, "parent": str(identity["parent"]), "repeat": int(repeat),
                "window_id": f"{split}:{identity['parent']}:repeat-{repeat}", "status": "no_opportunity",
                "scenario_sha256": str(identity["scenario_sha256"]), "input_hash": "", "continuation_id": "",
                "candidate_ids": [], "reason": reason}

    def collect(self) -> dict[str, Any]:
        splits = self.matrix.get("splits", {})
        identities = []
        for split in ("train", "model_selection", "prediction_confirmation"):
            for identity in splits.get(split, ()):
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
        prediction = [row for row in rows if row["split"] == "prediction_confirmation" and row["status"] == "complete"]
        return {"pass": len({row["parent"] for row in prediction}) == 8,
                "prediction_parent_count": len({row["parent"] for row in prediction}),
                "windows": len(rows), "complete_windows": sum(row["status"] == "complete" for row in rows),
                "no_opportunity_windows": sum(row["status"] == "no_opportunity" for row in rows),
                "path": str(output_path), "source_tape_identity_verified": True,
                "decision_input_frozen_before_branches": True, "future_observation_in_input": False}


__all__ = ["ProductionDataCollector", "ProductionDataError", "WINDOW_SCHEMA"]
