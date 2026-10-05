"""Synthetic end-to-end fixture for the frozen GPPO task pipeline."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from typing import Any

import numpy as np


SOURCE_PACKAGE = Path(__file__).resolve().parent
POLICY_SEEDS = (8301, 8302, 8303)
WORLD_SEEDS = (8201, 8202, 8203)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _synthetic_scenario(parent: str, index: int) -> dict[str, Any]:
    tasks = []
    for task_index in range(6):
        tasks.append({
            "task_id": f"task-{task_index}", "arrival": 0.0,
            "x": float((task_index * 3 + index) % 9) / 3.0,
            "y": float((task_index * 5 + 2 * index) % 8) / 3.0,
            "deadline": 18.0, "service": 1.0 + (task_index % 3) * 0.25,
            "priority": 1.0 + (task_index % 2) * 0.5,
            "region_id": task_index % 3, "target_id": task_index % 4,
        })
    return {
        "name": f"synthetic-{parent}", "seed": 9000 + index,
        "split": "synthetic_test", "tape_id": f"synthetic-{parent}-{index}",
        "tasks": tasks, "events": [], "communication": None,
    }


def _field_row(values: tuple[float, ...]) -> list[float]:
    row: list[float] = []
    for value in values:
        row.extend((float(value), 1.0, 1.0, 0.0))
    return row


class SyntheticM10Environment:
    """M10-shaped deterministic task environment with no native constructor."""

    def __init__(self, config: Any, scenario: Any, exogenous_key: str):
        self.config = config
        self.scenario = scenario
        self.exogenous_key = str(exogenous_key)
        self.time = 0
        self.energy: list[float] = []
        self.completed: set[str] = set()
        self.completion_records: dict[str, dict[str, Any]] = {}
        self.reset()

    def reset(self):
        self.time = 0
        self.energy = [float(self.config.initial_energy)] * int(self.config.uav_count)
        self.completed = set()
        self.completion_records = {}
        return self._observation()

    def _observation(self) -> dict[str, Any]:
        task_specs = tuple(self.scenario.tasks)
        uav_rows = []
        for uav_index in range(4):
            uav_rows.append(_field_row((
                float(uav_index) + self.time * 0.001, float(uav_index % 2), self.energy[uav_index],
                1.0, 1.0, 1.0,
            )))
        task_rows = []
        for task in task_specs:
            pending = task.task_id not in self.completed and self.time < float(task.deadline)
            task_rows.append(_field_row((
                float(task.x), float(task.y), float(task.deadline),
                max(0.0, float(task.service)), float(task.priority),
                float(pending), float(task.region_id), float(task.target_id),
            )))
        mask = [False] * 25
        for action in range(24):
            task = task_specs[action % 6]
            mask[action] = task.task_id not in self.completed and self.time < float(task.deadline)
        mask[24] = True

        uav_matrix = np.asarray(uav_rows, dtype=np.float32)
        task_matrix = np.asarray(task_rows, dtype=np.float32)
        uav_nodes = np.pad(uav_matrix, ((0, 0), (0, 32 - uav_matrix.shape[1])))
        task_nodes = np.pad(task_matrix, ((0, 0), (0, 32 - task_matrix.shape[1])))
        relations = np.zeros((4, 6, 4), dtype=np.float32)
        for uav_index in range(4):
            for task_index, task in enumerate(task_specs):
                distance = float(np.hypot(
                    uav_index - float(task.x), uav_index % 2 - float(task.y),
                ))
                relations[uav_index, task_index] = (
                    distance / 10.0, 1.0,
                    float(mask[uav_index * 6 + task_index]), 0.0,
                )
        regions = np.zeros((3, 32), dtype=np.float32)
        targets = np.zeros((4, 32), dtype=np.float32)
        events = np.zeros((4, 32), dtype=np.float32)
        node_features = np.concatenate(
            (uav_nodes, regions, targets, task_nodes, events), axis=0,
        )
        flat = np.concatenate((
            uav_nodes.reshape(-1), regions.reshape(-1), targets.reshape(-1),
            task_nodes.reshape(-1), events.reshape(-1), relations.reshape(-1),
            np.asarray((self.time / float(self.config.horizon), 0.0), dtype=np.float32),
        )).astype(np.float32)
        return {
            "flat": flat, "uavs": uav_rows, "tasks": task_rows,
            "mask": mask, "time": float(self.time), "version": self.time,
            "public_entity_ids": {
                "uavs": [f"uav-{index}" for index in range(4)],
                "tasks": [task.task_id for task in task_specs],
            },
            "trigger_flags": {}, "event_signal": 0.0,
            "continuation_actions": [],
            "graph": {"node_features": node_features, "relations": relations},
        }

    def step(self, action: int):
        if type(action) is not int or not 0 <= action < 25:
            raise ValueError("synthetic action outside 4x6+NOOP space")
        self.time += 1
        if action != 24:
            self.energy[action // 6] = max(0.0, self.energy[action // 6] - 0.01)
            task = self.scenario.tasks[action % 6]
            if task.task_id not in self.completed and self.time < float(task.deadline):
                self.completed.add(task.task_id)
                self.completion_records[task.task_id] = {
                    "physical_arrival_time": float(self.time),
                    "host_confirmation_time": float(self.time),
                }
        else:
            self.energy[0] = max(0.0, self.energy[0] - 0.005)
        terminated = False
        truncated = self.time >= int(self.config.horizon)
        expired = max(0, len(self.scenario.tasks) - len(self.completed)) if truncated else 0
        task_states = {
            task.task_id: ("completed" if task.task_id in self.completed else "expired")
            for task in self.scenario.tasks
        } if truncated else {
            task.task_id: ("completed" if task.task_id in self.completed else "pending")
            for task in self.scenario.tasks
        }
        info = {
            "time": float(self.time),
            "counts": {"completed": len(self.completed), "expired": expired},
            "energy": {f"uav-{index}": value for index, value in enumerate(self.energy)},
            "tasks": task_states, "completion_records": self.completion_records,
            "terminated": terminated, "truncated": truncated,
        }
        return self._observation(), 0.0, bool(terminated or truncated), info


class SyntheticPipelineBoundary:
    integration_context = {"mode": "synthetic_test"}

    def __init__(self):
        self.real_environment_constructor_calls = 0
        self.synthetic_environment_returns = 0
        self.world_model_loader_calls = 0
        self.world_model_fixture_returns = 0
        self.world_model_loader_operation_calls = 0
        self.policy_checkpoint_loads = 0
        self.environment_operations = 0
        self.world_model_fixture_state_hashes: dict[str, str] = {}
        self.world_model_fixture_state_consistent = True
        self._matrix: dict[str, Any] | None = None

    def bind_matrix(self, matrix: dict[str, Any]) -> None:
        self._matrix = matrix

    def construct_environment(self, _environment_type: Any, config: Any,
                              scenario: Any, exogenous_key: str):
        self.synthetic_environment_returns += 1
        return SyntheticM10Environment(config, scenario, exogenous_key)

    def environment(self, operation, *args, **kwargs):
        if getattr(operation, "__name__", "") == "<lambda>":
            raise RuntimeError("synthetic boundary blocked native environment construction")
        self.environment_operations += 1
        return operation(*args, **kwargs)

    model = environment
    optimizer = environment

    def checkpoint(self, operation, *args, **kwargs):
        if getattr(operation, "__name__", "") == "world_model_loader":
            self.world_model_loader_calls += 1
            owner = getattr(operation, "__self__", None)
            matrix = getattr(getattr(owner, "adapter", None), "matrix", self._matrix)
            if not isinstance(matrix, dict):
                raise RuntimeError("synthetic world model lacks its architecture")
            import torch
            from production_world import _model_config
            from production_policy import state_dict_sha256
            from w1_graph_jepa import W1GraphJEPA
            if len(args) != 2 or str(args[0]) != "G1":
                raise RuntimeError("unexpected synthetic world-model loader identity")
            seed = int(args[1])
            cpu_rng = torch.random.default_generator.get_state()
            try:
                torch.random.default_generator.manual_seed(seed)
                model = W1GraphJEPA(_model_config(matrix), event_count=5, device="cpu")
            finally:
                torch.random.default_generator.set_state(cpu_rng)
            model.cpu()
            state_hash = state_dict_sha256(model.state_dict())
            seed_key = str(seed)
            prior_hash = self.world_model_fixture_state_hashes.setdefault(seed_key, state_hash)
            self.world_model_fixture_state_consistent &= prior_hash == state_hash
            if prior_hash != state_hash:
                raise RuntimeError("synthetic world-model initialization changed across stages")
            self.world_model_fixture_returns += 1
            return model
        if getattr(operation, "__name__", "") == "policy_loader":
            self.policy_checkpoint_loads += 1
        return operation(*args, **kwargs)

    def telemetry(self) -> dict[str, Any]:
        import torch
        return {
            "real_environment_constructor_calls": self.real_environment_constructor_calls,
            "synthetic_environment_returns": self.synthetic_environment_returns,
            "world_model_loader_calls": self.world_model_loader_calls,
            "world_model_fixture_returns": self.world_model_fixture_returns,
            "world_model_loader_operation_calls": self.world_model_loader_operation_calls,
            "policy_checkpoint_loads": self.policy_checkpoint_loads,
            "environment_operations": self.environment_operations,
            "world_model_fixture_initial_state_hashes": dict(sorted(self.world_model_fixture_state_hashes.items())),
            "world_model_fixture_state_consistent": self.world_model_fixture_state_consistent,
            "cuda_initialized": bool(torch.cuda.is_initialized()),
            "world_model_device": "cpu",
        }


def make_pipeline_fixture(base: Path) -> tuple[Path, dict[str, Any], dict[str, Any], SyntheticPipelineBoundary]:
    """Copy the package without frozen G1 files and replace its tape with synthetic scenarios."""
    base = Path(base).resolve()
    base.mkdir(parents=True, exist_ok=True)
    root = base / "fixture-package"

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = {"frozen-models", "tapes-train.json"}.intersection(names)
        return ignored

    shutil.copytree(SOURCE_PACKAGE, root, ignore=ignore)
    matrix = _read_json(SOURCE_PACKAGE / "experiment-matrix.json")
    request = _read_json(SOURCE_PACKAGE / "RESOURCE_REQUEST.json")
    matrix["e2e_fixture"] = {
        "enabled": True, "mode": "synthetic_test",
        "policy_training_steps_per_method_seed": 128,
        "policy_rollout_steps": 64,
        "policy_optimizer_updates_per_method_seed": 2,
    }
    request["stages"]["conditional_policy_training"]["policy_optimizer_updates"] = 18

    source_rows = []
    all_splits = (*matrix["splits"]["train"], *matrix["splits"]["task_confirmation"])
    seen = set()
    for index, row in enumerate(all_splits):
        key = str(row["exogenous_key"])
        if key in seen:
            continue
        seen.add(key)
        parent = str(row["parent"])
        scenario = _synthetic_scenario(parent, index)
        digest = _canonical_sha256(scenario)
        row["scenario_sha256"] = digest
        row["structural_sha256"] = _canonical_sha256({
            "task_count": len(scenario["tasks"]),
            "coordinates": [[task["x"], task["y"]] for task in scenario["tasks"]],
        })
        source_rows.append({
            "parent": parent, "exogenous_key": key,
            "scenario_id": scenario["tape_id"], "scenario": scenario,
        })

    tape_path = root / "native" / "source-evidence" / "tapes-synthetic.json"
    tape_path.parent.mkdir(parents=True, exist_ok=True)
    tape_payload = json.dumps(source_rows, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True, allow_nan=False).encode("utf-8")
    tape_path.write_bytes(tape_payload)
    runtime_inputs_path = root / "runtime-inputs.json"
    runtime_inputs = _read_json(runtime_inputs_path)
    runtime_inputs["source_run"]["train_tape_file"] = "tapes-synthetic.json"
    runtime_inputs["source_run"]["train_tape_sha256"] = hashlib.sha256(tape_payload).hexdigest()
    runtime_inputs["source_run"]["windows_root"] = str(tape_path.parent.resolve())
    runtime_inputs["task_confirmation_data_read"] = False
    runtime_inputs_path.write_text(json.dumps(runtime_inputs, sort_keys=True, indent=2) + "\n",
                                   encoding="utf-8")
    (root / "experiment-matrix.json").write_text(
        json.dumps(matrix, sort_keys=True, indent=2, ensure_ascii=True) + "\n", encoding="utf-8",
    )
    (root / "RESOURCE_REQUEST.json").write_text(
        json.dumps(request, sort_keys=True, indent=2, ensure_ascii=True) + "\n", encoding="utf-8",
    )
    boundary = SyntheticPipelineBoundary()
    boundary.bind_matrix(matrix)
    return root, matrix, request, boundary


__all__ = ["SyntheticPipelineBoundary", "SyntheticM10Environment", "make_pipeline_fixture"]


class _FakeClock:
    def __init__(self):
        self.wall = 0.0
        self.cpu = 0.0

    def perf_counter(self) -> float:
        return self.wall

    def process_time(self) -> float:
        return self.cpu

    def advance(self, wall: float, cpu: float) -> None:
        self.wall += wall
        self.cpu += cpu


class _FakeCalls:
    def __init__(self, clock: _FakeClock):
        self.clock = clock
        self.names: list[str] = []

    def charge(self, name: str, _amounts: dict[str, int]) -> None:
        self.names.append(name)
        self.clock.advance(0.75, 0.25)


class _OneStepBoundary(SyntheticPipelineBoundary):
    def construct_environment(self, environment_type, config, scenario, exogenous_key):
        environment = super().construct_environment(environment_type, config, scenario, exogenous_key)
        native_step = environment.step

        def step(action: int):
            observation, reward, _done, info = native_step(action)
            info = dict(info)
            info.update({
                "time": float(environment.time), "terminated": False, "truncated": True,
                "counts": {"completed": 0, "expired": len(scenario.tasks)},
                "tasks": {task.task_id: "expired" for task in scenario.tasks},
            })
            return observation, reward, True, info

        environment.step = step
        return environment


class PolicyPreparationFixtureTests(unittest.TestCase):
    @staticmethod
    def _purge_native_modules() -> None:
        for name in tuple(sys.modules):
            if name == "gppo_world" or name.startswith("gppo_world."):
                del sys.modules[name]

    def test_fixture_replaces_formal_tape_and_excludes_frozen_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root, matrix, request, boundary = make_pipeline_fixture(Path(directory))
            tape_dir = root / "native" / "source-evidence"
            rows = json.loads((tape_dir / "tapes-synthetic.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 32)
            self.assertTrue(all(row["scenario"]["split"] == "synthetic_test" for row in rows))
            self.assertFalse((tape_dir / "tapes-train.json").exists())
            self.assertFalse((root / "frozen-models").exists())
            self.assertEqual(matrix["e2e_fixture"]["policy_training_steps_per_method_seed"], 128)
            self.assertEqual(matrix["e2e_fixture"]["policy_optimizer_updates_per_method_seed"], 2)
            self.assertEqual(
                request["stages"]["conditional_policy_training"]["policy_optimizer_updates"], 18,
            )
            self.assertEqual(boundary.synthetic_environment_returns, 0)
            self.assertEqual(boundary.world_model_loader_calls, 0)
            self.assertFalse((Path(directory) / "run-once").exists())

    def test_task_audit_uses_strict_physical_and_inclusive_host_deadlines(self):
        package_path = str(SOURCE_PACKAGE)
        native_path = str(SOURCE_PACKAGE / "native")
        for path in (package_path, native_path):
            if path not in sys.path:
                sys.path.insert(0, path)
        runtime_hooks = importlib.import_module("runtime_hooks")
        labels = runtime_hooks.task_outcome_labels(
            [
                {"task_id": "equal", "deadline": 5.0},
                {"task_id": "missing", "deadline": 5.0},
            ],
            {
                "time": 6.0,
                "completion_records": {
                    "equal": {"physical_arrival_time": 5.0, "host_confirmation_time": 5.0},
                },
                "tasks": {"equal": "completed", "missing": "unknown"},
            },
        )
        self.assertFalse(labels[0]["physical_on_time_completion"]["value"])
        self.assertTrue(labels[0]["host_confirmation"]["value"])
        self.assertFalse(labels[1]["physical_on_time_completion"]["valid"])
        self.assertIsNone(labels[1]["physical_on_time_completion"]["value"])
        self.assertFalse(labels[1]["host_confirmation"]["valid"])

    def test_unknown_utility_recomputation_uses_the_recorded_utility(self):
        package_path = str(SOURCE_PACKAGE)
        if package_path not in sys.path:
            sys.path.insert(0, package_path)
        production_policy = importlib.import_module("production_policy")
        episodes = [
            {"method": method, "seed": seed, "parent": "synthetic-parent", "repeat": 0,
             "utility": None, "utility_valid": False, "outcome_valid": False,
             "opportunity": False}
            for method in ("G0", "T", "G1") for seed in POLICY_SEEDS
        ]
        episodes.append({
            "method": "H", "seed": None, "parent": "synthetic-parent", "repeat": 0,
            "utility": None, "utility_valid": False, "outcome_valid": False,
            "opportunity": False,
        })
        metrics = production_policy.recompute_task_metrics(
            episodes, [{"parent": "synthetic-parent", "repeat": 0}],
        )
        self.assertEqual(metrics["unknown_utility_episode_count"], 10)
        self.assertIsNone(metrics["g1_minus_t_macro_utility"])
        self.assertEqual(metrics["per_parent_repeat"][0]["status"], "utility_unknown")

    def test_gppo_update_handles_masks_and_replays_each_behavior_logprob(self):
        import numpy as np
        import torch

        package_path = str(SOURCE_PACKAGE)
        native_path = str(SOURCE_PACKAGE / "native")
        for path in (package_path, native_path):
            if path not in sys.path:
                sys.path.insert(0, path)
        runtime_hooks = importlib.import_module("runtime_hooks")
        production_policy = importlib.import_module("production_policy")

        with tempfile.TemporaryDirectory() as directory:
            root, matrix, request, boundary = make_pipeline_fixture(Path(directory))
            boundary.bind_matrix(matrix)
            self._purge_native_modules()
            adapter = type("Adapter", (), {"root": root, "matrix": matrix,
                                            "boundary": boundary, "output": Path(directory)})()
            hooks = runtime_hooks.NativeRuntimeHooks(adapter)
            torch.manual_seed(771)
            base_policy = hooks.policy_factory("T", 8301, matrix["policy_configuration"])
            policy = production_policy.PriorConditionedPolicy(base_policy)
            optimizer = hooks.optimizer_factory(
                policy, "T", 8301, matrix["policy_configuration"],
            )
            schedule = production_policy.policy_schedule(
                matrix, request,
            )
            decision_policy = production_policy.DecisionPriorPolicy(
                method="T", seed=8301, policy=policy,
                prior_configuration=schedule.prior_configuration,
                input_builder=hooks.input_builder,
                transparent_scorer=hooks.transparent_scorer,
                world_models=(), world_predictor=None,
                calls=production_policy.CallAccounting(None, boundary),
                parent="synthetic-update", repeat=0,
            )
            env = SyntheticM10Environment(
                hooks._m10_config(),
                importlib.import_module("gppo_world.m10_environment").scenario_from_dict(
                    _synthetic_scenario("synthetic-update", 1),
                ),
                "synthetic-update|repeat-0",
            )
            observation = env.reset()
            from public_history import CausalPublicHistory
            history = CausalPublicHistory()
            history.append(observation)
            masks = []
            partial_mask = [False] * 25
            for action in (0, 7, 24):
                partial_mask[action] = True
            noop_mask = [False] * 24 + [True]
            masks.extend((partial_mask, partial_mask, noop_mask))
            transitions = []
            for index, mask in enumerate(masks):
                decision_observation = dict(observation)
                decision_observation = history.attach_current(decision_observation)
                decision_observation["mask"] = mask
                decision_observation["policy_preference"] = (0.5, 0.5)
                decision_observation["decision_identity"] = {
                    "parent": "synthetic-update", "repeat": 0,
                    "exogenous_key": "synthetic-update|repeat-0", "scenario_id": "synthetic-update",
                }
                decision = decision_policy.select(
                    decision_observation, decision_step=index,
                    hidden=torch.zeros((1, 1, 128), dtype=torch.float32),
                )
                probabilities = torch.as_tensor(decision.trace["action_probabilities"])
                mask_tensor = torch.as_tensor(mask, dtype=torch.bool)
                self.assertTrue(torch.isfinite(probabilities).all())
                self.assertTrue(torch.all(probabilities[~mask_tensor] == 0))
                if index == 2:
                    self.assertEqual(float(probabilities[24]), 1.0)
                else:
                    prior = decision.candidate_features[0, :, 17]
                    self.assertTrue(bool((prior[mask_tensor] != 0).any()),
                                    decision.trace["final_prior_by_action"])
                transitions.append({
                    "obs": np.asarray(decision_observation["flat"], dtype=np.float32).copy(),
                    "policy_hidden_before": np.zeros(128, dtype=np.float32),
                    "mask": np.asarray(mask, dtype=np.bool_),
                    "preference": np.asarray((0.5, 0.5), dtype=np.float32),
                    "candidate_features": decision.candidate_features[0].detach().cpu().numpy().copy(),
                    "action": decision.action, "old_log_prob": decision.log_probability,
                    "old_values": decision.values.numpy().astype(np.float32),
                    "next_values": np.zeros(2, dtype=np.float32),
                    "vector_reward": np.asarray((0.1 - 0.05 * index, -0.01), dtype=np.float32),
                    "terminated": index == 2, "truncated": False,
                })
            preupdate_errors = []
            with torch.no_grad():
                for transition in transitions:
                    obs = torch.as_tensor(transition["obs"], dtype=torch.float32).reshape(1, -1)
                    hidden = torch.as_tensor(transition["policy_hidden_before"], dtype=torch.float32).reshape(1, 1, -1)
                    features, pair, _ = policy.encode(obs, hidden)
                    evaluated = policy.evaluate_encoded(
                        features, pair,
                        torch.as_tensor(transition["preference"], dtype=torch.float32).reshape(1, 2),
                        torch.as_tensor(transition["candidate_features"], dtype=torch.float32).unsqueeze(0),
                        torch.as_tensor(transition["mask"], dtype=torch.bool).reshape(1, 25),
                    )
                    action = torch.tensor([transition["action"]], dtype=torch.long)
                    replayed = float(evaluated["distribution"].log_prob(action)[0])
                    preupdate_errors.append(abs(replayed - transition["old_log_prob"]))
            self.assertLessEqual(max(preupdate_errors), 1e-6)

            from production_policy import state_dict_sha256
            before_hash = state_dict_sha256(policy.state_dict())
            update_config = dict(matrix["policy_configuration"], rollout_steps=3)
            metrics = hooks.ppo_update_fn(
                policy, transitions, optimizer, update_config,
                torch.device("cpu"), event_group="T",
            )
            self.assertTrue(all(np.isfinite(float(value)) for value in metrics.values()))
            self.assertGreaterEqual(metrics["grad_norm"], 0.0)
            self.assertNotEqual(before_hash, state_dict_sha256(policy.state_dict()))

    def test_hungarian_decision_clock_includes_ledger_charge(self):
        with tempfile.TemporaryDirectory() as directory:
            root, matrix, _request, _unused_boundary = make_pipeline_fixture(Path(directory))
            self._purge_native_modules()
            package_path = str(SOURCE_PACKAGE)
            native_path = str(SOURCE_PACKAGE / "native")
            for path in (package_path, native_path):
                if path not in sys.path:
                    sys.path.insert(0, path)
            runtime_hooks = importlib.import_module("runtime_hooks")

            boundary = _OneStepBoundary()
            boundary.bind_matrix(matrix)
            output = Path(directory) / "probe-output"
            output.mkdir()
            adapter = type("Adapter", (), {
                "root": root, "matrix": matrix, "boundary": boundary, "output": output,
            })()
            hooks = runtime_hooks.NativeRuntimeHooks(adapter)
            clock = _FakeClock()
            calls = _FakeCalls(clock)
            source_row = matrix["splits"]["task_confirmation"][0]
            context = type("EpisodeContext", (), {
                "method": "H", "seed": None, "parent": source_row["parent"],
                "repeat": 0, "exogenous_key": source_row["exogenous_key"],
                "max_steps": 1, "decision_policy": None,
                "task_preference": (0.8, 0.2), "utility_discount": 0.99,
                "task_component_scale": 0.5, "calls": calls, "clock": clock,
                "reset_environment": lambda self, operation, *args, **kwargs: operation(*args, **kwargs),
                "step_environment": lambda self, operation, *args, **kwargs: operation(*args, **kwargs),
            })()
            result = hooks.episode_runner(context)
            self.assertEqual(calls.names, ["hungarian_public_rule_decision"])
            self.assertEqual(result["steps"], 1)
            self.assertEqual(len(result["decision_costs"]), 1)
            self.assertAlmostEqual(result["decision_costs"][0]["wall_seconds"], 0.75)
            self.assertAlmostEqual(result["decision_costs"][0]["cpu_seconds"], 0.25)
            self.assertEqual(boundary.real_environment_constructor_calls, 0)
            self.assertEqual(boundary.synthetic_environment_returns, 1)


if __name__ == "__main__":
    unittest.main()
