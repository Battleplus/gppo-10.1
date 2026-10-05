"""Native bottom-boundary hooks for the W1 production adapter.

The adapter owns contracts and stage ordering.  This module owns only the
reviewed native environment/model/optimizer/checkpoint calls that are allowed
to occur after an externally approved run.  Tests replace these methods at the
bottom boundary; they do not replace the adapter or the production stages.
"""
from __future__ import annotations

import time
import json
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

from infra_io import durable_append_jsonl


class RuntimeHookError(RuntimeError):
    pass


def _policy_sampling_identity(parent: str, repeat: int, policy_seed: int) -> tuple[str, int]:
    key = json.dumps({"parent": str(parent), "repeat": int(repeat),
                      "policy_seed": int(policy_seed)}, sort_keys=True,
                     separators=(",", ":"), ensure_ascii=False)
    derived = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)
    return key, derived


def task_outcome_labels(tasks: Any, final_info: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Audit terminal physical-arrival and host-receipt labels independently."""
    import math

    records = final_info.get("completion_records", {})
    states = final_info.get("tasks", {})
    final_time = float(final_info.get("time", float("nan")))
    if not math.isfinite(final_time) or not isinstance(records, Mapping) or not isinstance(states, Mapping):
        raise RuntimeHookError("TASK_OUTCOME_TERMINAL_RECORD_INVALID")
    outcomes = []
    for task in tasks:
        task_id = str(task["task_id"])
        deadline = float(task["deadline"])
        if not math.isfinite(deadline):
            raise RuntimeHookError("TASK_OUTCOME_DEADLINE_INVALID")
        record = records.get(task_id, {})
        if not isinstance(record, Mapping):
            raise RuntimeHookError("TASK_OUTCOME_COMPLETION_RECORD_INVALID")
        state = str(states.get(task_id, "unknown"))
        physical_time = record.get("physical_arrival_time")
        host_time = record.get("host_confirmation_time")
        if physical_time is not None:
            physical_time = float(physical_time)
            if not math.isfinite(physical_time) or physical_time > final_time:
                raise RuntimeHookError("TASK_OUTCOME_PHYSICAL_TIME_INVALID")
            physical_label = {
                "value": bool(physical_time < deadline), "valid": True,
                "source": "completion_records.physical_arrival_time", "reason": None,
            }
        elif state == "expired":
            physical_label = {
                "value": False, "valid": True,
                "source": "terminal_task_state.expired", "reason": None,
            }
        else:
            physical_label = {
                "value": None, "valid": False, "source": None,
                "reason": "terminal_physical_arrival_not_observed",
            }
        if host_time is not None:
            host_time = float(host_time)
            if not math.isfinite(host_time) or host_time > final_time:
                raise RuntimeHookError("TASK_OUTCOME_HOST_TIME_INVALID")
            host_label = {
                "value": bool(host_time <= deadline), "valid": True,
                "source": "completion_records.host_confirmation_time", "reason": None,
            }
        elif state == "expired":
            host_label = {
                "value": False, "valid": True,
                "source": "terminal_task_state.expired_without_confirmation", "reason": None,
            }
        else:
            host_label = {
                "value": None, "valid": False, "source": None,
                "reason": "host_confirmation_not_observed_before_episode_end",
            }
        outcomes.append({
            "task_id": task_id, "deadline": deadline, "terminal_state": state,
            "physical_on_time_completion": physical_label,
            "host_confirmation": host_label,
        })
    return outcomes


class NativeRuntimeHooks:
    def __init__(self, adapter: Any):
        self.adapter = adapter
        self.root = Path(adapter.root)
        self.source_root = self._source_root()
        self._m10_config_value: Any | None = None
        self._scenario_rows_value: list[Mapping[str, Any]] | None = None
        self._g1_binding_value: Mapping[str, Any] | None = None
        self._environment_construction_sequence = 0
        self._policy_checkpoint_hashes: dict[str, str] = {}
        self._policy_checkpoint_state_hashes: dict[str, str] = {}
        if str(self.source_root) not in __import__("sys").path:
            __import__("sys").path.insert(0, str(self.source_root))
        if str(self.root) not in __import__("sys").path:
            __import__("sys").path.insert(0, str(self.root))

    def _source_root(self) -> Path:
        path = (self.root / "native").resolve()
        if not path.is_dir():
            raise RuntimeHookError("PACKAGE_NATIVE_ROOT_MISSING")
        return path

    def _native(self) -> tuple[Any, ...]:
        from gppo_world.graph5 import graph5_from_m10_observation
        from gppo_world.joint_consequence_baseline import public_joint_action_score
        from gppo_world.joint_gppo import JointGraphPreferencePolicy, JointTrainConfig
        from gppo_world.m10_environment import M10Config, M10Environment, scenario_from_dict
        from production_data import _public_pending_task_count
        return (graph5_from_m10_observation, public_joint_action_score,
                JointGraphPreferencePolicy, JointTrainConfig,
                M10Config, M10Environment, scenario_from_dict,
                _public_pending_task_count)

    def _m10_config(self) -> Any:
        if self._m10_config_value is not None:
            return self._m10_config_value
        _, _, _, _, config_type, *_ = self._native()
        contract = json.loads((self.root / "environment-config-contract.json").read_text(encoding="utf-8"))
        values = contract.get("config")
        digest = contract.get("config_sha256")
        if contract.get("schema") != "w1-environment-config-contract/1.0.0" or not isinstance(values, Mapping) or not isinstance(digest, str):
            raise RuntimeHookError("ENVIRONMENT_CONFIG_CONTRACT_INVALID")
        from runtime_config_contract import validate_config_before_collection
        config = config_type(**dict(values))
        try:
            identity = validate_config_before_collection(config, values, expected_sha256=digest)
        except ValueError as exc:
            raise RuntimeHookError(str(exc)) from exc
        if identity["task_completion_mode"] != "arrival_to_region" or identity["deadline_basis"] != "physical_arrival":
            raise RuntimeHookError("TASK_OUTCOME_CONFIG_SEMANTICS_NOT_SUPPORTED")
        self._m10_config_value = config
        return config

    def _construct_environment(self, environment_type: Any, config: Any,
                               scenario: Any, exogenous_key: str) -> Any:
        operation = lambda: environment_type(
            config, scenario, exogenous_key=exogenous_key,
        )
        self._environment_construction_sequence += 1
        sequence = self._environment_construction_sequence
        output = getattr(self.adapter, "output", None)
        evidence_dir = Path(output).resolve() if output is not None else self.root / "run-once"
        base = {
            "sequence": sequence, "exogenous_key": str(exogenous_key),
            "scenario_id": str(getattr(scenario, "tape_id", getattr(scenario, "name", "unknown"))),
        }
        durable_append_jsonl(evidence_dir / "environment-construction.jsonl",
                             {**base, "event": "attempted"})
        try:
            integration_context = getattr(self.adapter.boundary, "integration_context", {})
            if (isinstance(integration_context, Mapping)
                    and integration_context.get("mode") == "synthetic_test"):
                construct = getattr(self.adapter.boundary, "construct_environment", None)
                if not callable(construct):
                    raise RuntimeHookError("SYNTHETIC_ENVIRONMENT_CONSTRUCTOR_BOUNDARY_MISSING")
                result = construct(environment_type, config, scenario, exogenous_key)
            else:
                result = self.adapter.boundary.environment(operation)
        except BaseException:
            raise
        durable_append_jsonl(evidence_dir / "environment-construction.jsonl",
                             {**base, "event": "constructed"})
        return result

    @staticmethod
    def _file_sha256(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _scenario_rows(self) -> list[Mapping[str, Any]]:
        if self._scenario_rows_value is not None:
            return self._scenario_rows_value
        inputs = json.loads((self.root / "runtime-inputs.json").read_text(encoding="utf-8"))
        source = inputs.get("source_run", {})
        relative = Path(str(source.get("train_tape_file", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.name:
            raise RuntimeHookError("TRAIN_TAPE_PATH_INVALID")
        path = self.source_root / "source-evidence" / relative
        if not path.is_file():
            raise RuntimeHookError("STAGED_TRAIN_TAPE_MISSING")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != str(source.get("train_tape_sha256", "")):
            raise RuntimeHookError("STAGED_TRAIN_TAPE_DIGEST_MISMATCH")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise RuntimeHookError("TRAIN_TAPE_EMPTY")
        self._scenario_rows_value = rows
        return self._scenario_rows_value

    def _row(self, exogenous_key: str) -> Mapping[str, Any]:
        prefix = str(exogenous_key).rsplit("|repeat-", 1)[0]
        source_key = prefix + "|repeat-0"
        rows = [row for row in self._scenario_rows() if str(row.get("exogenous_key")) == source_key]
        if len(rows) != 1:
            raise RuntimeHookError(f"SCENARIO_IDENTITY_NOT_UNIQUE:{source_key}")
        return rows[0]

    def world_model_loader(self, variant: str, seed: int) -> Any:
        from production_world import _file_sha256, _load_checkpoint, _model_config
        from w1_graph_jepa import W1GraphJEPA
        if variant != "G1" or int(seed) not in (8201, 8202, 8203):
            raise RuntimeHookError(f"UNSUPPORTED_FROZEN_WORLD_MODEL:{variant}:{seed}")
        binding_path = self.root / "g1-model-binding.json"
        if self._g1_binding_value is None:
            if not binding_path.is_file():
                raise RuntimeHookError("G1_MODEL_BINDING_MISSING")
            self._g1_binding_value = json.loads(binding_path.read_text(encoding="utf-8"))
        binding = self._g1_binding_value
        if binding.get("schema") != "w1-g1-model-binding/1.0.0":
            raise RuntimeHookError("G1_MODEL_BINDING_SCHEMA_INVALID")
        routes = binding.get("routes")
        if not isinstance(routes, list):
            raise RuntimeHookError("G1_MODEL_BINDING_ROUTES_INVALID")
        route_rows = [row for row in routes if isinstance(row, Mapping) and int(row.get("world_seed", -1)) == int(seed)]
        if len(route_rows) != 1:
            raise RuntimeHookError(f"G1_MODEL_BINDING_ROUTE_MISMATCH:{seed}")
        route = route_rows[0]
        paired_policy_seed = {8201: 8301, 8202: 8302, 8203: 8303}[int(seed)]
        architecture = self.adapter.matrix.get("world_model_architecture")
        route_architecture = route.get("architecture")
        if (int(route.get("policy_seed", -1)) != paired_policy_seed
                or not isinstance(route_architecture, Mapping)
                or dict(route_architecture) != dict(architecture or {})):
            raise RuntimeHookError(f"G1_MODEL_BINDING_IDENTITY_MISMATCH:{seed}")
        relative = Path(str(route.get("checkpoint_relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.name:
            raise RuntimeHookError("G1_MODEL_BINDING_PATH_INVALID")
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError as exc:
            raise RuntimeHookError("G1_MODEL_BINDING_PATH_ESCAPES_PACKAGE") from exc
        expected_hash = str(route.get("checkpoint_sha256", ""))
        if len(expected_hash) != 64 or not path.is_file() or _file_sha256(path) != expected_hash:
            raise RuntimeHookError(f"G1_MODEL_CHECKPOINT_DIGEST_MISMATCH:{seed}")
        payload = _load_checkpoint(path, "cpu")
        if not isinstance(payload, Mapping) or set(payload) != {"metadata", "state_dict"}:
            raise RuntimeHookError(f"G1_MODEL_CHECKPOINT_PAYLOAD_INVALID:{seed}")
        metadata = route.get("checkpoint_metadata")
        if (not isinstance(metadata, Mapping) or payload["metadata"] != dict(metadata)
                or metadata.get("schema") != "w1-world-model-checkpoint/2.0.0"
                or metadata.get("variant") != "G1"
                or int(metadata.get("seed", -1)) != int(seed)
                or metadata.get("event_loss_enabled") is not False
                or dict(metadata.get("architecture", {})) != dict(route_architecture)
                or type(metadata.get("best_epoch")) is not int
                or int(metadata.get("best_epoch", 0)) <= 0
                or metadata.get("continuation_id") != self.adapter.matrix.get("candidate_continuation_id")):
            raise RuntimeHookError(f"G1_MODEL_CHECKPOINT_METADATA_MISMATCH:{seed}")
        model = W1GraphJEPA(_model_config(self.adapter.matrix), event_count=5, device="cpu")
        model.load_state_dict(payload["state_dict"])
        model.eval()
        model.requires_grad_(False)
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeHookError("G1_MODEL_FREEZE_FAILED")
        return model

    def policy_factory(self, method: str, seed: int, _config: Mapping[str, Any]) -> Any:
        _, _, policy_type, *_ = self._native()
        import torch
        torch.manual_seed(int(seed))
        return policy_type(self._m10_config(), history=True)

    def optimizer_factory(self, policy: Any, _method: str, _seed: int, config: Mapping[str, Any]) -> Any:
        import torch
        return torch.optim.Adam(policy.parameters(), lr=float(config["learning_rate"]))

    def checkpoint_writer(self, path: str, policy: Any, optimizer: Any, route: Any, summary: Mapping[str, Any]) -> str:
        import torch
        from production_policy import state_dict_sha256
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # ``train_policy_routes`` passes the prior wrapper, whereas task
        # evaluation restores the base policy before rebuilding that wrapper.
        # Persist the base identity so save/load are inverse operations.
        policy_state = policy.base.state_dict() if hasattr(policy, "base") else policy.state_dict()
        state_hash = state_dict_sha256(policy_state)
        torch.save({"schema": "w1-policy-checkpoint/1.0.0", "method": route.method,
                    "seed": route.seed, "state_dict": policy_state,
                    "state_dict_sha256": state_hash,
                    "optimizer_state_dict": optimizer.state_dict(), "summary": dict(summary)}, target)
        resolved = str(target.resolve())
        self._policy_checkpoint_state_hashes[resolved] = state_hash
        return str(target)

    def policy_loader(self, method: str, seed: int, path: str) -> Any:
        import torch
        from production_policy import state_dict_sha256
        _, _, policy_type, *_ = self._native()
        resolved = str(Path(path).resolve())
        expected_hashes = getattr(self.adapter, "policy_checkpoint_hashes", {})
        expected_hash = expected_hashes.get(resolved) if isinstance(expected_hashes, Mapping) else None
        if expected_hash is None:
            expected_hash = self._policy_checkpoint_hashes.get(resolved)
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise RuntimeHookError("POLICY_CHECKPOINT_EXPECTED_HASH_MISSING")
        if not Path(resolved).is_file() or self._file_sha256(resolved) != expected_hash:
            raise RuntimeHookError(f"POLICY_CHECKPOINT_DIGEST_MISMATCH:{method}:{seed}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "w1-policy-checkpoint/1.0.0" or payload.get("method") != method or int(payload.get("seed", -1)) != int(seed):
            raise RuntimeHookError(f"POLICY_CHECKPOINT_IDENTITY_MISMATCH:{method}:{seed}")
        observed_state_hash = state_dict_sha256(payload["state_dict"])
        if payload.get("state_dict_sha256") != observed_state_hash:
            raise RuntimeHookError(f"POLICY_CHECKPOINT_STATE_DIGEST_MISMATCH:{method}:{seed}")
        expected_state_hashes = getattr(self.adapter, "policy_checkpoint_state_hashes", {})
        expected_state_hash = expected_state_hashes.get(resolved) if isinstance(expected_state_hashes, Mapping) else None
        if expected_state_hash is None:
            expected_state_hash = self._policy_checkpoint_state_hashes.get(resolved)
        if expected_state_hash is not None and observed_state_hash != expected_state_hash:
            raise RuntimeHookError(f"POLICY_CHECKPOINT_STATE_IDENTITY_MISMATCH:{method}:{seed}")
        policy = policy_type(self._m10_config(), history=True)
        policy.load_state_dict(payload["state_dict"])
        policy.eval()
        return policy

    def input_builder(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        import math
        import torch
        from public_history import history_vector
        flat = torch.as_tensor(observation["flat"], dtype=torch.float32).reshape(1, -1)
        relations = torch.as_tensor(observation["graph"]["relations"], dtype=torch.float32).reshape(24, 4)
        preference_value = observation.get("policy_preference")
        if not isinstance(preference_value, (list, tuple)) or len(preference_value) != 2:
            raise RuntimeHookError("POLICY_PREFERENCE_MISSING")
        preference = tuple(float(value) for value in preference_value)
        if not all(math.isfinite(value) for value in preference):
            raise RuntimeHookError("POLICY_PREFERENCE_NONFINITE")
        candidates = torch.zeros((25, 17), dtype=torch.float32)
        candidates[:24, :4] = relations
        return {
            "obs_tensor": flat,
            "world_history": torch.as_tensor(history_vector(observation), dtype=torch.float32).reshape(1, 128),
            "preference": torch.tensor([preference], dtype=torch.float32),
            "candidate_features": candidates,
            "mask": torch.as_tensor(observation["mask"], dtype=torch.bool).reshape(1, 25),
        }

    def transparent_scorer(self, observation: Mapping[str, Any], _inputs: Mapping[str, Any]) -> Any:
        import torch
        from transparent_utility import transparent_horizon_components
        from production_data import _public_pending_task_count
        graph_builder, scorer, *_ = self._native()
        graph = graph_builder(observation)
        config = self._m10_config()
        preference = torch.as_tensor(_inputs["preference"], dtype=torch.float32).reshape(-1)
        if preference.shape != (2,) or not bool(torch.isfinite(preference).all()):
            raise RuntimeHookError("TRANSPARENT_PREFERENCE_INVALID")
        scores = torch.zeros(25, dtype=torch.float32)
        component_rows = {}
        for action, allowed in enumerate(graph.action_mask.tolist()):
            if allowed:
                diagnostic = scorer(graph, action)
                ideal_continuation = _public_pending_task_count(
                    graph, action=action,
                    own_on_time=bool(diagnostic.get("own_on_time_public", False)),
                    task_capacity=int(config.task_capacity),
                )
                components = transparent_horizon_components(
                    diagnostic, task_capacity=int(config.task_capacity),
                    initial_total_energy=float(config.uav_count * config.initial_energy),
                    ideal_continuation_completions=ideal_continuation,
                )
                component_rows[str(action)] = {
                    "task": float(components[0]), "energy": float(components[1]),
                    "ideal_continuation_completions": int(ideal_continuation),
                    "task_score": float(diagnostic["score"]),
                    "energy_cost": float(diagnostic["energy_cost"]),
                }
                scores[action] = (
                    0.5 * float(preference[0]) * components[0]
                    + float(preference[1]) * components[1]
                )
        return {"scores": scores, "components": component_rows}

    def world_predictor(self, _method: str, model: Any, observation: Mapping[str, Any], actions: Any, _mask: Any) -> Mapping[str, Any]:
        import torch
        from w1_graph_jepa import expand_candidate_batch
        from public_history import history_vector
        graph_builder, *_ = self._native()
        graph = graph_builder(observation)
        action_tensor = torch.as_tensor(actions, dtype=torch.long)
        relation_rows = torch.as_tensor(graph.candidate_features, dtype=torch.float32).reshape(24, 4)
        relations = torch.cat((relation_rows, torch.zeros((1, 4))), dim=0)[action_tensor]
        nodes = {name: value.unsqueeze(0) for name, value in graph.nodes.items()}
        history = torch.as_tensor(history_vector(observation), dtype=torch.float32).reshape(1, 128)
        expanded_nodes, expanded_history = expand_candidate_batch(nodes, history, action_tensor, relations)
        with torch.no_grad():
            output = model.predict_candidates(expanded_nodes, expanded_history, action_tensor, relations)
        return {"candidate_actions": action_tensor, "outcomes": output["outcome"]}

    def ppo_update_fn(self, policy: Any, transitions: Any, optimizer: Any, config: Mapping[str, Any], device: Any, *, event_group: str) -> Mapping[str, Any]:
        _, _, _, config_type, *_ = self._native()
        from gppo_world.joint_training import ppo_preference_update
        train_config = config_type(
            seed=0, rollout_steps=int(config["rollout_steps"]), policy_lr=float(config["learning_rate"]),
            gamma=float(config["gamma"]), gae_lambda=float(config["gae_lambda"]),
            clip_epsilon=float(config["clip_epsilon"]), entropy_weight=float(config["entropy_weight"]),
            value_weight=float(config["value_weight"]), grad_clip=float(config["gradient_clip_norm"]),
        )
        # Keep the PriorConditionedPolicy wrapper in the replay evaluation.
        # Its augmented candidate tensor contains the behavior-time prior;
        # stripping it would recompute a different log-probability for G1/G2.
        return ppo_preference_update(policy, transitions, optimizer, train_config, device, event_group=event_group)

    def route_runner(self, context: Any) -> Mapping[str, Any]:
        """Train one GPPO route with the frozen world model, if any.

        This is deliberately a small production orchestration loop.  The old
        ``joint_training.run_group`` entry point jointly optimized a world
        model, so routing through it would violate the frozen-world-model
        contract even when its world-update budget was set to zero.
        """
        import numpy as np
        import torch
        from classical_baselines import PublicDecisionAdapter
        from gppo_world.joint_training import _mask_safe, _vector_reward
        from production_policy import canonical_sha256
        _, _, _, _, _, environment_type, scenario_from_dict, _ = self._native()
        config = self._m10_config()

        train_rows = self._scenario_rows()
        scenarios = []
        for row in self.adapter.matrix.get("splits", {}).get("train", ()):
            key = str(row.get("exogenous_key"))
            source = next((item for item in train_rows
                           if str(item.get("exogenous_key")) == key), None)
            if source is None:
                raise RuntimeHookError(f"TRAIN_SCENARIO_NOT_FOUND:{key}")
            scenarios.append((row, source))
        if len(scenarios) != 24:
            raise RuntimeHookError("TRAIN_SCENARIO_COUNT_MISMATCH")

        device = torch.device("cpu")
        transitions = []
        total_updates = 0
        rollout_count = context.route.steps // context.route.rollout_steps
        if rollout_count <= 0 or context.route.steps % context.route.rollout_steps:
            raise RuntimeHookError("POLICY_ROLLOUT_SCHEDULE_INVALID")
        updates_per_rollout = context.route.optimizer_updates // rollout_count
        if updates_per_rollout <= 0 or context.route.optimizer_updates % rollout_count:
            raise RuntimeHookError("POLICY_UPDATE_SCHEDULE_INVALID")
        scenario_index = 0
        episode_index = 0
        env = None
        observation = None
        public_adapter = None
        prepared_decision = None
        policy_hidden = None
        replay_decision_flats: list[np.ndarray] = []
        recurrent_hidden_replay_groups = 0
        recurrent_hidden_replay_observations = 0
        recurrent_hidden_replay_traces: list[dict[str, Any]] = []
        replay_hidden_check_pending = False
        replay_hidden_expected: torch.Tensor | None = None
        previous_counts = {"completed": 0, "expired": 0}
        previous_energy = float(config.uav_count * config.initial_energy)
        raw_decisions: list[dict[str, Any]] = []
        decision_traces: list[dict[str, Any]] = []
        ppo_update_traces: list[dict[str, Any]] = []

        def start_episode(index: int):
            from public_history import CausalPublicHistory
            row, source = scenarios[index % len(scenarios)]
            scenario = scenario_from_dict(source["scenario"])
            runtime_env = self._construct_environment(
                environment_type, config, scenario, str(row["exogenous_key"]),
            )
            obs = context.calls.call(
                "environment", f"policy_reset:{context.route.method}:{context.route.seed}",
                {"resets_upper": 1}, runtime_env.reset,
            )
            history = CausalPublicHistory()
            history.append(obs)
            return runtime_env, obs, PublicDecisionAdapter(), history

        current_preference = context.schedule.training_preference(episode_index, context.route.seed)
        env, observation, public_adapter, causal_history = start_episode(scenario_index)
        while context.decision_policy.decision_count < context.route.steps:
            if observation is None or public_adapter is None:
                raise RuntimeHookError("POLICY_OBSERVATION_MISSING")
            if prepared_decision is None:
                public_observation, effective_mask = public_adapter.prepare(observation)
                public_observation = causal_history.attach_current(public_observation)
                if "public_history" not in public_observation or "public_history_schema" not in public_observation:
                    raise RuntimeHookError("CAUSAL_HISTORY_ATTACH_FAILED")
            else:
                public_observation, effective_mask = prepared_decision
                prepared_decision = None
            decision_observation = dict(public_observation)
            decision_observation["mask"] = effective_mask
            decision_observation["policy_preference"] = current_preference
            source_row, scenario_source = scenarios[scenario_index % len(scenarios)]
            scenario_value = scenario_source.get("scenario", {})
            decision_observation["decision_identity"] = {
                "method": context.route.method, "seed": context.route.seed,
                "parent": str(source_row.get("parent", "policy-training")),
                "repeat": 0, "exogenous_key": str(source_row.get("exogenous_key")),
                "scenario_id": str(scenario_value.get("tape_id", source_row.get("parent", "policy-training"))),
            }
            if replay_hidden_check_pending:
                hidden_matches = (
                    policy_hidden is None and replay_hidden_expected is None
                ) or (
                    policy_hidden is not None and replay_hidden_expected is not None
                    and torch.equal(policy_hidden, replay_hidden_expected)
                )
                recurrent_hidden_replay_traces.append({
                    "decision_step": int(context.decision_policy.decision_count),
                    "expected_hidden_sha256": None if replay_hidden_expected is None
                    else canonical_sha256(replay_hidden_expected),
                    "observed_hidden_sha256": None if policy_hidden is None
                    else canonical_sha256(policy_hidden),
                    "matches": bool(hidden_matches),
                })
                if not hidden_matches:
                    raise RuntimeHookError("POLICY_HIDDEN_REPLAY_NOT_USED_AT_NEXT_DECISION")
                replay_hidden_check_pending = False
                replay_hidden_expected = None
            decision = context.decision_policy.select(
                decision_observation, decision_step=context.decision_policy.decision_count,
                hidden=policy_hidden,
            )
            replay_decision_flats.append(
                np.asarray(decision_observation["flat"], dtype=np.float32).reshape(-1).copy()
            )
            raw_decisions.append({
                "decision_step": int(context.decision_policy.decision_count - 1),
                "candidate_count": int(decision.trace["candidate_count"]),
                "base_argmax_action": int(decision.trace["base_argmax_action"]),
                "prior_adjusted_argmax_action": int(decision.trace["prior_adjusted_argmax_action"]),
                "prior_changed": bool(decision.trace["prior_changed"]),
                "argmax_changed": bool(decision.trace["argmax_changed"]),
                "selected_action_without_prior_same_rng": decision.trace["selected_action_without_prior_same_rng"],
                "sampled_action_changed": (
                    decision.action != decision.trace["selected_action_without_prior_same_rng"]
                    if decision.trace["sampled_action_change_evaluated"] else None
                ),
                "selected_action": int(decision.action),
                "noop_selected": int(decision.action) == 24,
            })
            decision_traces.append({**dict(decision.trace),
                                    "training_preference": list(current_preference)})
            values = decision.values
            if values is None:
                raise RuntimeHookError("POLICY_VALUE_MISSING")
            mask = _mask_safe(np.asarray(effective_mask, dtype=np.bool_))
            if not bool(mask[int(decision.action)]):
                raise RuntimeHookError("POLICY_SELECTED_ILLEGAL_ACTION")
            public_adapter.commit(int(decision.action))
            old_obs = decision_observation
            next_obs, raw_reward, done, info = context.calls.call(
                "environment", f"policy_step:{context.route.method}:{context.route.seed}",
                {"environment_steps": 1}, env.step, int(decision.action),
            )
            # The returned state is appended once before it can be used for
            # bootstrap or the next action; rollout replay stores that exact input.
            causal_history.append(next_obs)
            vector_reward, consequence, previous_counts, previous_energy = _vector_reward(
                info, previous_counts, previous_energy, config,
            )
            transition = {
                "obs": np.asarray(old_obs["flat"], dtype=np.float32).copy(),
                "next_obs": np.asarray(next_obs["flat"], dtype=np.float32).copy(),
                "observation_dict": old_obs, "next_observation_dict": next_obs,
                "mask": mask.copy(), "action": int(decision.action),
                "old_log_prob": float(decision.log_probability),
                "old_values": values.numpy().astype(np.float32),
                "next_values": np.zeros(2, dtype=np.float32),
                "vector_reward": vector_reward,
                "task_consequence": consequence,
                "raw_environment_reward": float(raw_reward),
                "event_label": np.zeros(5, dtype=np.float32),
                "event_mask": np.zeros(5, dtype=np.bool_),
                "state_target_valid": bool(np.isfinite(next_obs["flat"]).all()),
                "vector_reward_valid": bool(np.isfinite(vector_reward).all()),
                "task_consequence_valid": bool(np.isfinite(consequence).all()),
                "censor_reason": None,
                "policy_hidden_before": np.zeros(128, dtype=np.float32) if policy_hidden is None
                    else policy_hidden.detach().cpu().numpy().reshape(-1).copy(),
                "policy_hidden_after": np.zeros(128, dtype=np.float32) if decision.next_hidden is None
                    else decision.next_hidden.detach().cpu().numpy().reshape(-1).copy(),
                "world_hidden_before": np.zeros(128, dtype=np.float32),
                "candidate_features": decision.candidate_features[0].detach().cpu().numpy().copy(),
                "preference": np.asarray(current_preference, dtype=np.float32),
                "terminated": bool(info.get("terminated", False)),
                "truncated": bool(info.get("truncated", False)),
                "environment_truncated": bool(info.get("truncated", False)),
                "policy_version": total_updates, "world_version": 0,
                "scenario_id": str(getattr(env.scenario, "tape_id", scenarios[scenario_index][0]["parent"])),
                "time": float(old_obs.get("time", 0.0)), "action_legal": True,
            }
            if transition["truncated"] and not transition["terminated"]:
                next_public, next_effective_mask = public_adapter.prepare(next_obs)
                next_decision_observation = dict(next_public)
                next_decision_observation["mask"] = next_effective_mask
                next_decision_observation["policy_preference"] = current_preference
                next_decision_observation = causal_history.attach_current(next_decision_observation)
                transition["next_values"] = context.decision_policy.bootstrap_value(
                    next_decision_observation, hidden=decision.next_hidden,
                ).numpy().astype(np.float32)
                # The pending prepare belongs to the terminal observation and
                # must not be committed because no action is submitted.
                public_adapter.reset()
            if transitions and not transitions[-1]["terminated"] and not transitions[-1]["truncated"]:
                transitions[-1]["next_values"] = transition["old_values"].copy()
            transitions.append(transition)
            observation = next_obs
            policy_hidden = decision.next_hidden
            if done:
                episode_index += 1
                scenario_index = episode_index % len(scenarios)
                current_preference = context.schedule.training_preference(episode_index, context.route.seed)
                env, observation, public_adapter, causal_history = start_episode(scenario_index)
                prepared_decision = None
                policy_hidden = None
                replay_decision_flats = []
                previous_counts = {"completed": 0, "expired": 0}
                previous_energy = float(config.uav_count * config.initial_energy)
            if len(transitions) == context.route.rollout_steps:
                last = transitions[-1]
                if not (last["terminated"] or last["truncated"]):
                    next_public, next_effective_mask = public_adapter.prepare(observation)
                    next_decision_observation = dict(next_public)
                    next_decision_observation["mask"] = next_effective_mask
                    next_decision_observation["policy_preference"] = current_preference
                    next_decision_observation = causal_history.attach_current(next_decision_observation)
                    last["next_values"] = context.decision_policy.bootstrap_value(
                        next_decision_observation, hidden=policy_hidden,
                    ).numpy().astype(np.float32)
                    last["truncated"] = True
                    last["rollout_boundary"] = True
                    # The same prepared public observation becomes the first
                    # decision of the next rollout after the PPO update.
                    # Preserve the causal history attached to the bootstrap
                    # observation when it becomes the next rollout prefix.
                    prepared_decision = (next_decision_observation, next_effective_mask)
                rollout_index = (context.decision_policy.decision_count // context.route.rollout_steps) - 1
                for update_in_rollout in range(updates_per_rollout):
                    update_metrics = context.update_policy(transitions, device=device)
                    if not isinstance(update_metrics, Mapping):
                        raise RuntimeHookError("PPO_UPDATE_METRICS_MISSING")
                    ppo_update_traces.append({
                        "method": context.route.method, "seed": context.route.seed,
                        "rollout_index": rollout_index,
                        "update_index": total_updates,
                        "update_in_rollout": update_in_rollout,
                        "metrics": dict(update_metrics),
                    })
                    total_updates += 1
                replay_hidden = None
                with torch.no_grad():
                    for replay_flat in replay_decision_flats:
                        replay_obs = torch.as_tensor(replay_flat, dtype=torch.float32).reshape(1, -1)
                        replay_encoded = context.calls.call(
                            "model", f"{context.route.method.lower()}_recurrent_hidden_replay",
                            {"encode_sample_evaluations": 1},
                            context.decision_policy.policy.encode, replay_obs, replay_hidden,
                        )
                        if not isinstance(replay_encoded, (tuple, list)) or len(replay_encoded) != 3:
                            raise RuntimeHookError("POLICY_HIDDEN_REPLAY_OUTPUT_INVALID")
                        replay_hidden = replay_encoded[2]
                policy_hidden = replay_hidden
                replay_hidden_expected = None if replay_hidden is None else replay_hidden.detach().clone()
                replay_hidden_check_pending = True
                recurrent_hidden_replay_groups += 1
                recurrent_hidden_replay_observations += len(replay_decision_flats)
                transitions = []
        if transitions:
            raise RuntimeHookError("POLICY_ROLLOUT_REMAINDER")
        return {
            "environment_steps": context.route.steps,
            "policy_optimizer_updates": total_updates,
            "world_model_updates": 0,
            "world_model_frozen": context.route.world_model_variant is not None,
            "route": f"{context.route.method}:{context.route.seed}",
            "decision_summary": context.decision_policy.decision_summary(),
            "raw_decisions": raw_decisions,
            "decision_traces": decision_traces,
            "ppo_update_traces": ppo_update_traces,
            "recurrent_hidden_replay_groups": recurrent_hidden_replay_groups,
            "recurrent_hidden_replay_observations": recurrent_hidden_replay_observations,
            "recurrent_hidden_replay_traces": recurrent_hidden_replay_traces,
        }

    def episode_runner(self, context: Any) -> Mapping[str, Any]:
        import numpy as np
        import torch
        from public_history import CausalPublicHistory
        from classical_baselines import ClassicalSelector, PublicDecisionAdapter
        from gppo_world.joint_training import _vector_reward
        from production_policy import canonical_sha256
        _, _, _, _, _, environment_type, scenario_from_dict, _ = self._native()
        row = self._row(context.exogenous_key)
        config = self._m10_config()
        env = self._construct_environment(
            environment_type, config, scenario_from_dict(row["scenario"]), context.exogenous_key,
        )
        observation = context.reset_environment(env.reset)
        public_history = CausalPublicHistory()
        public_history.append(observation)
        opportunity = False
        actions, vector_rewards, utility, steps = [], [], 0.0, 0
        public_adapter = PublicDecisionAdapter()
        hungarian = ClassicalSelector("hungarian", preference=context.task_preference) if context.decision_policy is None else None
        policy_sampling_key = None
        policy_sampling_seed = None
        if context.decision_policy is not None:
            policy_sampling_key, policy_sampling_seed = _policy_sampling_identity(
                context.parent, context.repeat, int(context.seed),
            )
            torch.manual_seed(policy_sampling_seed)
        policy_hidden = None
        hungarian_costs: list[dict[str, Any]] = []
        previous_counts = {"completed": 0, "expired": 0}
        initial_energy = float(config.uav_count * config.initial_energy)
        previous_energy = initial_energy
        final_info = None
        done = False
        while steps < context.max_steps and not done:
            wall_started = context.clock.perf_counter()
            cpu_started = context.clock.process_time()
            public_observation, effective_mask = public_adapter.prepare(observation)
            public_observation = public_history.attach_current(public_observation)
            opportunity = opportunity or bool(torch.as_tensor(effective_mask, dtype=torch.bool)[:24].any())
            decision_observation = dict(public_observation)
            decision_observation["mask"] = effective_mask
            decision_observation["policy_preference"] = context.task_preference
            decision_observation["decision_identity"] = {
                "method": context.method, "seed": context.seed,
                "parent": context.parent, "repeat": context.repeat,
                "exogenous_key": context.exogenous_key,
                "scenario_id": str(row.get("scenario_id", row.get("parent", context.parent))),
                "policy_sampling_key": policy_sampling_key,
                "policy_sampling_seed": policy_sampling_seed,
            }
            if hungarian is not None:
                action, diagnostic = hungarian.choose(decision_observation, public_adapter.memory)
                context.calls.charge("hungarian_public_rule_decision", {"public_rule_decisions": 1})
                cpu_seconds = float(context.clock.process_time() - cpu_started)
                wall_seconds = float(context.clock.perf_counter() - wall_started)
                if not all(math.isfinite(value) and value >= 0 for value in (cpu_seconds, wall_seconds)):
                    raise RuntimeHookError("HUNGARIAN_DECISION_CLOCK_INVALID")
                legal_actions = [index for index, legal in enumerate(effective_mask) if legal]
                ids = public_observation.get("public_entity_ids", {})
                candidate_ids = []
                for legal_action in legal_actions:
                    if legal_action == 24:
                        candidate_ids.append({"action_id": 24, "candidate_id": "NOOP"})
                    else:
                        uav_index, task_index = divmod(legal_action, 6)
                        candidate_ids.append({
                            "action_id": legal_action,
                            "candidate_id": f"{ids['uavs'][uav_index]}|{ids['tasks'][task_index]}",
                            "uav_id": str(ids["uavs"][uav_index]),
                            "task_id": str(ids["tasks"][task_index]),
                        })
                public_input_sha256 = canonical_sha256({
                    "flat": decision_observation.get("flat"),
                    "relations": decision_observation.get("graph", {}).get("relations"),
                    "mask": effective_mask, "preference": list(context.task_preference),
                    "public_history": decision_observation.get("public_history"),
                    "public_history_schema": decision_observation.get("public_history_schema"),
                    "public_entity_ids": ids,
                })
                trace = {
                    "method": "H", "seed": None, "parent": context.parent,
                    "repeat": context.repeat, "exogenous_key": context.exogenous_key,
                    "scenario_id": decision_observation["decision_identity"]["scenario_id"],
                    "decision_step": steps, "public_input_sha256": public_input_sha256,
                    "public_history_identity": {
                        "schema": decision_observation.get("public_history_schema"),
                        "sha256": canonical_sha256(decision_observation.get("public_history")),
                    },
                    "legal_actions": legal_actions,
                    "legal_candidate_identities": candidate_ids,
                    "diagnostic": diagnostic, "action_probabilities": None,
                    "selected_action": int(action), "policy_sampling_key": None,
                    "policy_sampling_seed": None,
                    "sampled_action_change_evaluated": False,
                }
                hungarian_costs.append({
                    "method": "H", "parent": context.parent, "repeat": context.repeat,
                    "seed": None, "decision_step": steps,
                    "candidate_count": int(sum(bool(value) for value in effective_mask)),
                    "cpu_seconds": cpu_seconds, "wall_seconds": wall_seconds,
                    "world_model_forwards": 0, "world_model_sample_evaluations": 0,
                    "selected_action": int(action), "noop_selected": int(action) == 24,
                    "has_non_noop_candidate": any(value != 24 for value in legal_actions),
                    "g1_cost_eligible": False,
                    "sampled_action_change_evaluated": False,
                    "cost_scope": "shared public preparation/history + public features + Hungarian assignment + action selection",
                })
            else:
                decision = context.decision_policy.select(
                    decision_observation, decision_step=steps, hidden=policy_hidden,
                    clock_start=(wall_started, cpu_started),
                )
                action, trace = decision.action, decision.trace
                policy_hidden = decision.next_hidden
            public_adapter.commit(int(action))
            observation, scalar_reward, done, info = context.step_environment(env.step, int(action))
            public_history.append(observation)
            if bool(done) != (bool(info.get("terminated")) or bool(info.get("truncated"))):
                raise RuntimeHookError("TASK_TERMINATION_MISMATCH")
            if bool(info.get("terminated")) and bool(info.get("truncated")):
                raise RuntimeHookError("TASK_TERMINATION_AMBIGUOUS")
            vector_reward, _consequence, previous_counts, previous_energy = _vector_reward(
                info, previous_counts, previous_energy, config,
            )
            if not bool(np.isfinite(vector_reward).all()):
                raise RuntimeHookError("TASK_VECTOR_REWARD_NONFINITE")
            utility += context.utility_discount ** steps * (
                context.task_component_scale * context.task_preference[0] * float(vector_reward[0])
                + context.task_preference[1] * float(vector_reward[1])
            )
            reward_vector = np.asarray(vector_reward, dtype=np.float32).tolist()
            vector_rewards.append(reward_vector)
            actions.append({**dict(trace), "preference": list(context.task_preference),
                            "vector_reward": reward_vector,
                            "scalar_environment_reward": float(scalar_reward)})
            final_info = info
            steps += 1
        if not done or final_info is None:
            raise RuntimeHookError("TASK_REACHED_NON_NATIVE_TRUNCATION")
        records = final_info.get("completion_records", {})
        tasks = tuple(row["scenario"].get("tasks", ()))
        if not tasks:
            raise RuntimeHookError("TASK_SCENARIO_HAS_NO_TASKS")
        lifecycle_outcomes = task_outcome_labels(tasks, final_info)
        physical_known = [row["physical_on_time_completion"]["value"] for row in lifecycle_outcomes
                          if row["physical_on_time_completion"]["valid"]]
        host_known = [row["host_confirmation"]["value"] for row in lifecycle_outcomes
                      if row["host_confirmation"]["valid"]]
        outcome_valid = all(row["physical_on_time_completion"]["valid"] for row in lifecycle_outcomes)
        unknown_reasons = sorted({row["physical_on_time_completion"]["reason"] for row in lifecycle_outcomes
                                  if not row["physical_on_time_completion"]["valid"]})
        config_contract = json.loads((self.root / "environment-config-contract.json").read_text(encoding="utf-8"))
        decision_summary = {} if context.decision_policy is None else context.decision_policy.decision_summary()
        return {
            "steps": steps, "opportunity": opportunity,
            "outcome_valid": outcome_valid,
            "outcome_unknown_reason": None if outcome_valid else ";".join(unknown_reasons),
            "utility_valid": True,
            "utility": utility, "actions": actions,
            "vector_rewards": vector_rewards,
            "lifecycle_outcomes": lifecycle_outcomes,
            "decision_costs": hungarian_costs,
            "preference": list(context.task_preference),
            "environment_config_sha256": config_contract.get("config_sha256"),
            "scenario_id": str(row.get("scenario_id", getattr(env.scenario, "tape_id", context.parent))),
            "physical_rate": None if not physical_known else sum(bool(value) for value in physical_known) / len(physical_known),
            "physical_outcome_valid_count": len(physical_known),
            "host_rate": None if not host_known else sum(bool(value) for value in host_known) / len(host_known),
            "host_outcome_valid_count": len(host_known),
            "counts": dict(previous_counts), "energy_used": initial_energy - previous_energy,
            "terminated": bool(final_info.get("terminated")),
            "truncated": bool(final_info.get("truncated")),
            "policy_sampling_key": policy_sampling_key,
            "policy_sampling_seed": policy_sampling_seed,
            "argmax_changed_count": decision_summary.get("argmax_changed_count", 0),
            "sampled_action_changed_count": decision_summary.get("sampled_action_changed_count", 0),
            "sampled_action_comparison_count": decision_summary.get("sampled_action_comparison_count", 0),
            "sampled_action_comparison_status": decision_summary.get("sampled_action_comparison_status", "not_evaluated"),
        }

    def train_policy_routes(self, *, matrix: Mapping[str, Any], request: Mapping[str, Any],
                            ledger: Any, boundary: Any | None = None,
                            output_dir: str | Path | None = None,
                            clock: Any = time) -> dict[str, Any]:
        from production_policy import train_policy_routes
        result = train_policy_routes(
            matrix=matrix, request=request, ledger=ledger,
            boundary=self.adapter.boundary if boundary is None else boundary,
            world_model_loader=self.world_model_loader,
            policy_factory=self.policy_factory,
            optimizer_factory=self.optimizer_factory,
            route_runner=self.route_runner,
            ppo_update_fn=self.ppo_update_fn,
            checkpoint_writer=self.checkpoint_writer,
            input_builder=self.input_builder,
            transparent_scorer=self.transparent_scorer,
            world_predictor=self.world_predictor,
            output_dir=output_dir, clock=clock,
        )
        hashes = {}
        state_hashes = {}
        for route in result.get("routes", ()):
            resolved = str(Path(route["checkpoint"]).resolve())
            hashes[resolved] = str(route["checkpoint_sha256"])
            state_hashes[resolved] = str(route["checkpoint_state_sha256"])
        self._policy_checkpoint_hashes = hashes
        self._policy_checkpoint_state_hashes = state_hashes
        try:
            self.adapter.policy_checkpoint_hashes = dict(hashes)
            self.adapter.policy_checkpoint_state_hashes = dict(state_hashes)
        except (AttributeError, TypeError):
            pass
        return result

    def evaluate_task_confirmation(self, *, matrix: Mapping[str, Any], request: Mapping[str, Any],
                                   ledger: Any, checkpoints: Mapping[tuple[str, int], str],
                                   boundary: Any | None = None,
                                   output_dir: str | Path | None = None,
                                   clock: Any = time, selector: Any | None = None) -> dict[str, Any]:
        from production_policy import evaluate_task_confirmation
        return evaluate_task_confirmation(
            matrix=matrix, request=request, ledger=ledger,
            boundary=self.adapter.boundary if boundary is None else boundary,
            policy_loader=self.policy_loader,
            world_model_loader=self.world_model_loader,
            episode_runner=self.episode_runner,
            input_builder=self.input_builder,
            transparent_scorer=self.transparent_scorer,
            world_predictor=self.world_predictor,
            checkpoints=checkpoints, output_dir=output_dir,
            clock=clock, selector=selector,
        )


__all__ = ["NativeRuntimeHooks", "RuntimeHookError", "task_outcome_labels"]
