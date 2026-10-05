from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from runtime_backend import ProductionRuntimeAdapter, RuntimeBackendError, W1RuntimeBackend, _verify_production_source_tree
from runtime_hooks import NativeRuntimeHooks
from production_policy import policy_schedule
from production_data import ProductionDataCollector
from public_history import CausalPublicHistory
from budget_ledger import BudgetLedger


ROOT = Path(__file__).resolve().parent
REQUEST = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
MATRIX = json.loads((ROOT / "experiment-matrix.json").read_text(encoding="utf-8"))


class Ledger:
    def __init__(self):
        self.stage = None
        self.calls = []

    def select(self, stage):
        self.stage = stage

    def call(self, name, amounts, function, *args, **kwargs):
        self.calls.append((self.stage, name, dict(amounts)))
        return function(*args, **kwargs)


class Boundary:
    def environment(self, operation, *args, **kwargs):
        return operation(*args, **kwargs)

    def model(self, operation, *args, **kwargs):
        return operation(*args, **kwargs)

    def optimizer(self, operation, *args, **kwargs):
        return operation(*args, **kwargs)

    def checkpoint(self, operation, *args, **kwargs):
        return operation(*args, **kwargs)


class Hooks:
    def __init__(self):
        self.calls = []

    def _record(self, name):
        self.calls.append(name)

    def world_model_loader(self, variant, seed): self._record(f"world:{variant}:{seed}"); return object()
    def policy_factory(self, method, seed, config): self._record(f"policy:{method}:{seed}"); return object()
    def optimizer_factory(self, *args): self._record("optimizer"); return object()
    def route_runner(self, context): self._record(f"route:{context.route.method}:{context.route.seed}"); return {}
    def ppo_update_fn(self, *args, **kwargs): self._record("ppo"); return {}
    def checkpoint_writer(self, path, *args): self._record(f"checkpoint:{path}"); return path
    def input_builder(self, observation): self._record("input"); return {}
    def transparent_scorer(self, observation, inputs): self._record("transparent"); return []
    def world_predictor(self, *args): self._record("predictor"); return {}
    def policy_loader(self, method, seed, path): self._record(f"policy-load:{method}:{seed}"); return object()
    def episode_runner(self, context): self._record(f"episode:{context.method}"); return {"steps": 0, "opportunity": False, "utility": None}


class AdapterWiringTests(unittest.TestCase):
    def setUp(self):
        self.output = Path(tempfile.mkdtemp())
        self.ledger = Ledger()
        self.hooks = Hooks()
        self.adapter = ProductionRuntimeAdapter(ROOT, self.output, REQUEST, MATRIX, self.ledger, Boundary(), self.hooks)

    def test_contract_expands_task_units_and_uses_current_policy_seeds(self):
        self.adapter._validate_frozen_contract()
        self.assertEqual((8301, 8302, 8303), (8301, 8302, 8303))

    @patch("runtime_backend._write_json")
    @patch("production_data.ProductionDataCollector")
    @patch("production_world.train_select_world_models")
    @patch("production_world.load_complete_windows")
    @patch("production_world.evaluate_prediction_confirmation")
    @patch("production_policy.train_policy_routes")
    @patch("production_policy.evaluate_task_confirmation")
    def test_formal_adapter_reaches_all_stage_functions(self, task_fn, policy_fn, eval_fn, load_windows,
                                                         train_fn, collector_type, _write_json):
        collector_type.return_value.collect.return_value = {"pass": True, "prediction_parent_count": 8}
        load_windows.return_value = []
        world_run = SimpleNamespace(routes=tuple({"variant": v, "seed": s} for v in ("G1", "G2") for s in (8201, 8202, 8203)), models={("G1", 8201): object()})
        train_fn.return_value = world_run
        eval_fn.return_value = SimpleNamespace(trace_path=self.output / "trace", metrics_path=self.output / "metrics", metrics={
            "valid_parent_count": 8,
            "g1_minus_transparent_macro_regret": 0.0,
            "g2_minus_g1_macro_regret": 0.0,
            "g2_minus_g1_event_brier": 0.0,
            "parents_g1_no_worse_than_transparent": 0,
            "parents_g2_no_worse_than_g1": 0,
        })
        policy_fn.return_value = {"routes": [], "policies": {}, "checkpoints": {}}
        task_fn.return_value = {"task_episodes": 312}

        self.assertTrue(self.adapter.collect_world_model_data()["pass"])
        self.assertEqual(len(self.adapter.train_world_models()["routes"]), 6)
        metrics = self.adapter.evaluate_world_models()
        self.assertEqual(metrics["valid_parent_count"], 8)
        self.adapter.prediction_run = SimpleNamespace(metrics=metrics)
        self.adapter.prediction_passed = True
        self.adapter.train_policies()
        self.adapter.evaluate_tasks()
        collector_type.assert_called_once()
        train_fn.assert_called_once()
        eval_fn.assert_called_once()
        policy_fn.assert_called_once()
        task_fn.assert_called_once()

    def test_no_placeholder_stage_errors_remain(self):
        source = (ROOT / "runtime_backend.py").read_text(encoding="utf-8")
        for marker in ("WORLD_MODEL_COLLECTOR_NOT_CONFIGURED", "WORLD_MODEL_TRAINER_NOT_CONFIGURED",
                       "WORLD_MODEL_EVALUATOR_NOT_CONFIGURED", "POLICY_TRAINER_NOT_CONFIGURED",
                       "TASK_EVALUATOR_NOT_CONFIGURED"):
            self.assertNotIn(marker, source)

    def test_policy_route_uses_frozen_world_model_path(self):
        source = (ROOT / "runtime_hooks.py").read_text(encoding="utf-8")
        self.assertNotIn("from gppo_world.joint_training import run_group", source)
        self.assertIn("world_model_updates", source)
        self.assertIn("world_model_frozen", source)
        self.assertIn("ppo_preference_update(policy, transitions", source)

    def test_runtime_verifies_complete_production_source_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "native" / "gppo_world"
            package.mkdir(parents=True)
            first = package / "__init__.py"
            second = package / "dependency.py"
            first.write_text("", encoding="utf-8")
            second.write_text("VALUE = 1\n", encoding="utf-8")
            manifest = {
                "native/gppo_world/__init__.py": __import__("hashlib").sha256(first.read_bytes()).hexdigest(),
                "native/gppo_world/dependency.py": __import__("hashlib").sha256(second.read_bytes()).hexdigest(),
            }
            self.assertEqual(_verify_production_source_tree(root, manifest), 2)
            second.write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeBackendError, "PRODUCTION_SOURCE_TREE_DIGEST_MISMATCH"):
                _verify_production_source_tree(root, manifest)
            second.write_text("VALUE = 1\n", encoding="utf-8")
            (package / "unmanifested.py").write_text("VALUE = 3\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeBackendError, "PRODUCTION_SOURCE_TREE_FILE_SET_MISMATCH"):
                _verify_production_source_tree(root, manifest)


def _field_row(values):
    row = []
    for value in values:
        row.extend((float(value), 1.0, 1.0, 0.0))
    return row


def _public_observation(time_value=0.0):
    uavs = [_field_row((float(index), 0.0, 10.0, 1.0, 1.0, 1.0)) for index in range(4)]
    tasks = [_field_row((float(index), 1.0, 18.0, 1.0, 1.0, 1.0, 0.0, 0.0)) for index in range(6)]
    mask = [False] * 25
    mask[0] = mask[7] = mask[24] = True
    return {
        "flat": [0.0] * 802, "graph": {"relations": [[[0.0] * 4 for _ in range(6)] for _ in range(4)]},
        "uavs": uavs, "tasks": tasks, "mask": mask, "time": float(time_value), "version": int(time_value),
        "public_entity_ids": {"uavs": [f"uav-{i}" for i in range(4)], "tasks": [f"task-{i}" for i in range(6)]},
        "trigger_flags": {}, "event_signal": 0.0, "continuation_actions": [],
    }


class _FakeConfig:
    uav_count = 4
    task_capacity = 6
    initial_energy = 10.0


class _FakeEnvironment:
    def __init__(self, _config, scenario, exogenous_key=None):
        self.scenario = scenario
        self.exogenous_key = exogenous_key
        self.index = 0

    def reset(self):
        self.index = 0
        return _public_observation(0.0)

    def step(self, _action):
        self.index += 1
        observation = _public_observation(float(self.index))
        done = self.index >= 3
        info = {
            "counts": {"completed": 0, "expired": 0},
            "energy": {f"uav-{i}": 10.0 for i in range(4)},
            "completion_records": {}, "time": float(self.index),
            "terminated": done, "truncated": False,
        }
        return observation, 123.0, done, info


class _RouteCalls:
    def __init__(self):
        self.amounts = {}

    def call(self, _kind, _name, amounts, operation, *args, **kwargs):
        for key, value in amounts.items():
            self.amounts[key] = self.amounts.get(key, 0) + value
        return operation(*args, **kwargs)


class _RouteDecisionPolicy:
    def __init__(self):
        self.decision_count = 0
        self.bootstrap_count = 0
        self.preferences = []

    def select(self, observation, *, decision_step, hidden=None):
        self.preferences.append(tuple(observation["policy_preference"]))
        self.decision_count += 1
        return SimpleNamespace(
            action=24, log_probability=0.0,
            values=torch.tensor((0.1, 0.2)),
            next_hidden=torch.full((1, 1, 128), float(decision_step + 1)),
            candidate_features=torch.zeros((1, 25, 18)), trace={"selected_action": 24}, cost=None,
        )

    def bootstrap_value(self, _observation, *, hidden=None):
        self.bootstrap_count += 1
        self.last_bootstrap_hidden = hidden
        return torch.tensor((0.3, 0.4))


class NativeHookTests(unittest.TestCase):
    def _hooks(self):
        hooks = NativeRuntimeHooks.__new__(NativeRuntimeHooks)
        hooks.source_root = Path(r"E:\Z博士\runs\w1-light-repaired-fair-rerun-v2-nativefs-once")
        if str(hooks.source_root) not in sys.path:
            sys.path.insert(0, str(hooks.source_root))
        if str(hooks.source_root / "native") not in sys.path:
            sys.path.insert(0, str(hooks.source_root / "native"))
        hooks.adapter = SimpleNamespace(boundary=Boundary(), matrix={
            "splits": {"train": [
                {"parent": f"train-{index:04d}", "exogenous_key": f"key-{index}"}
                for index in range(24)
            ]}
        })
        hooks._scenario_rows = lambda: [
            {"exogenous_key": f"key-{index}", "scenario": {"tape_id": f"tape-{index}"}}
            for index in range(24)
        ]
        hooks._row = lambda key: {"scenario": {
            "tape_id": key,
            "tasks": [{"task_id": f"task-{i}", "deadline": 18.0} for i in range(6)],
        }}
        hooks._native = lambda: (None, None, None, None, _FakeConfig, _FakeEnvironment,
                                 lambda value: SimpleNamespace(tape_id=value["tape_id"]))
        return hooks

    def test_policy_and_task_environment_construction_crosses_bottom_boundary(self):
        calls = []
        hooks = self._hooks()
        hooks.adapter.boundary = Boundary()
        hooks.adapter.boundary.environment = lambda operation, *args, **kwargs: (
            calls.append((operation, args, kwargs)) or operation(*args, **kwargs)
        )
        scenario = SimpleNamespace(tape_id="boundary-test")
        env = hooks._construct_environment(_FakeEnvironment, _FakeConfig(), scenario, "test|repeat-0")
        self.assertIsInstance(env, _FakeEnvironment)
        self.assertEqual(len(calls), 1)
        self.assertTrue(callable(calls[0][0]))
        self.assertEqual(env.exogenous_key, "test|repeat-0")

    def test_route_runner_bootstraps_rollout_and_keeps_world_frozen(self):
        hooks = self._hooks()
        calls = _RouteCalls()
        decision = _RouteDecisionPolicy()
        batches = []
        context = SimpleNamespace(
            route=SimpleNamespace(method="G1", seed=8301, steps=4, rollout_steps=2,
                                  optimizer_updates=2, world_model_variant="G1"),
            calls=calls, decision_policy=decision,
            update_policy=lambda rows, **_kwargs: batches.append(rows),
            schedule=policy_schedule(MATRIX, REQUEST),
        )
        summary = hooks.route_runner(context)
        self.assertEqual(summary["world_model_updates"], 0)
        self.assertTrue(summary["world_model_frozen"])
        self.assertEqual(decision.decision_count, 4)
        self.assertEqual(decision.bootstrap_count, 2)
        self.assertEqual(len(batches), 2)
        self.assertTrue(torch.allclose(
            torch.as_tensor(batches[0][-1]["next_values"]), torch.tensor((0.3, 0.4)),
        ))
        self.assertTrue(batches[0][-1]["rollout_boundary"])
        self.assertEqual(calls.amounts["environment_steps"], 4)
        self.assertEqual(decision.preferences,
                         [(0.2, 0.8), (0.2, 0.8), (0.2, 0.8), (0.5, 0.5)])
        np.testing.assert_allclose(
            np.asarray([tuple(row["preference"]) for batch in batches for row in batch], dtype=np.float64),
            np.asarray(decision.preferences, dtype=np.float64), rtol=0.0, atol=1e-6,
        )

    def test_environment_truncation_bootstraps_from_final_public_state(self):
        hooks = self._hooks()

        class TruncatedEnvironment(_FakeEnvironment):
            def step(self, _action):
                self.index += 1
                observation = _public_observation(float(self.index))
                info = {
                    "counts": {"completed": 0, "expired": 0},
                    "energy": {f"uav-{i}": 10.0 for i in range(4)},
                    "completion_records": {}, "time": float(self.index),
                    "terminated": False, "truncated": True,
                }
                return observation, 0.0, True, info

        hooks._native = lambda: (None, None, None, None, _FakeConfig, TruncatedEnvironment,
                                 lambda value: SimpleNamespace(tape_id=value["tape_id"]))
        calls = _RouteCalls()
        decision = _RouteDecisionPolicy()
        batches = []
        context = SimpleNamespace(
            route=SimpleNamespace(method="G0", seed=8301, steps=1, rollout_steps=1,
                                  optimizer_updates=1, world_model_variant=None),
            calls=calls, decision_policy=decision,
            update_policy=lambda rows, **_kwargs: batches.append(rows),
            schedule=policy_schedule(MATRIX, REQUEST),
        )
        hooks.route_runner(context)
        self.assertEqual(decision.bootstrap_count, 1)
        transition = batches[0][0]
        self.assertTrue(transition["environment_truncated"])
        self.assertFalse(transition["terminated"])
        self.assertTrue(torch.allclose(torch.as_tensor(transition["next_values"]), torch.tensor((0.3, 0.4))))

    def test_episode_opportunity_is_accumulated_after_reset(self):
        hooks = self._hooks()

        class DelayedEnvironment(_FakeEnvironment):
            def reset(self):
                self.index = 0
                observation = _public_observation(0.0)
                observation["mask"] = [False] * 24 + [True]
                return observation

            def step(self, _action):
                self.index += 1
                observation = _public_observation(float(self.index))
                if self.index == 1:
                    observation["mask"] = [False] * 24 + [True]
                done = self.index >= 3
                info = {
                    "counts": {"completed": 0, "expired": 0},
                    "energy": {f"uav-{i}": 10.0 for i in range(4)},
                    "completion_records": {}, "time": float(self.index),
                    "terminated": done, "truncated": False,
                }
                return observation, 0.0, done, info

        hooks._native = lambda: (None, None, None, None, _FakeConfig, DelayedEnvironment,
                                 lambda value: SimpleNamespace(tape_id=value["tape_id"]))

        class Context:
            method, seed, parent, repeat, exogenous_key, max_steps = "H", None, "p", 0, "key-0", 3
            decision_policy = None
            task_preference = (0.8, 0.2)
            utility_discount = 0.99
            task_component_scale = 0.5
            def reset_environment(self, operation): return operation()
            def step_environment(self, operation, action): return operation(action)

        result = hooks.episode_runner(Context())
        self.assertTrue(result["opportunity"])

    def test_repeat_identity_reuses_frozen_scenario_but_changes_exogenous_key(self):
        source = {
            "exogenous_key": "frozen|train|train-0048|repeat-0",
            "scenario_sha256": "a" * 64,
            "scenario": {"tape_id": "train-0048"},
        }
        collector = ProductionDataCollector.__new__(ProductionDataCollector)
        collector.tapes = [source]
        identity = {
            "parent": "train-0048", "exogenous_key": source["exogenous_key"],
            "scenario_sha256": "a" * 64,
        }
        row = collector._row_for(identity, 2)
        self.assertEqual(row["source_exogenous_key"], source["exogenous_key"])
        self.assertEqual(row["exogenous_key"], "frozen|train|train-0048|repeat-2")
        self.assertEqual(row["scenario"], source["scenario"])

        hooks = self._hooks()
        hooks._scenario_rows = lambda: [source]
        recovered = NativeRuntimeHooks._row(hooks, "frozen|train|train-0048|repeat-2")
        self.assertEqual(recovered["exogenous_key"], source["exogenous_key"])

    def test_task_runner_uses_receding_hungarian_and_carries_hidden(self):
        hooks = self._hooks()

        class Context:
            method, seed, parent, repeat, exogenous_key, max_steps = "H", None, "p", 0, "key-0", 3
            decision_policy = None
            task_preference = (0.8, 0.2)
            utility_discount = 0.99
            task_component_scale = 0.5
            def reset_environment(self, operation): return operation()
            def step_environment(self, operation, action): return operation(action)

        result = hooks.episode_runner(Context())
        self.assertEqual(result["steps"], 3)
        self.assertTrue(all(row["diagnostic"]["method"] == "hungarian" for row in result["actions"]))
        self.assertEqual(result["preference"], [0.8, 0.2])
        self.assertEqual(result["utility"], 0.0)
        self.assertTrue(all(row["scalar_environment_reward"] == 123.0 for row in result["actions"]))

        class PolicyDecision:
            def __init__(self): self.hiddens = []
            def select(self, _observation, *, decision_step, hidden=None):
                self.hiddens.append(hidden)
                return SimpleNamespace(action=24, trace={"selected_action": 24},
                                       next_hidden=torch.full((1, 1, 128), float(decision_step + 1)))

        policy_decision = PolicyDecision()
        policy_context = Context()
        policy_context.method = "G0"
        policy_context.seed = 8301
        policy_context.decision_policy = policy_decision
        hooks.episode_runner(policy_context)
        self.assertIsNone(policy_decision.hiddens[0])
        self.assertIsNotNone(policy_decision.hiddens[1])

    def test_world_predictor_uses_shared_history_and_batches_legal_candidates(self):
        hooks = NativeRuntimeHooks.__new__(NativeRuntimeHooks)
        graph = SimpleNamespace(
            candidate_features=torch.ones((24, 4)),
            nodes={"uavs": torch.zeros((4, 8)), "tasks": torch.zeros((6, 8))},
        )
        hooks._native = lambda: (lambda _observation: graph, None, None, None, None, None, None)
        observation = _public_observation(4.0)
        history = CausalPublicHistory()
        history.append(observation)
        frozen_observation = history.attach_current(observation)
        actions = torch.tensor((0, 24), dtype=torch.long)

        class Predictor:
            def predict_candidates(self, nodes, input_history, candidate_actions, relations):
                self.input_history = input_history
                self.actions = candidate_actions
                self.relations = relations
                self.nodes = nodes
                values = candidate_actions.to(torch.float32).unsqueeze(1)
                return {"outcome": values.expand(-1, 6)}

        predictor = Predictor()
        result = hooks.world_predictor("G1", predictor, frozen_observation, actions, torch.tensor(observation["mask"]))
        self.assertTrue(torch.equal(predictor.input_history[0], torch.tensor(frozen_observation["public_history"])))
        self.assertEqual(result["candidate_actions"].tolist(), [0, 24])
        self.assertEqual(tuple(result["outcomes"].shape), (2, 6))


class CollectorAccountingTests(unittest.TestCase):
    def test_production_collector_separates_rule_and_forced_candidate_counts(self):
        class EmptyStore:
            _latest = {}

        class FakeEnvironment:
            def __init__(self, _config, scenario, exogenous_key=None):
                self.scenario = scenario
                self.exogenous_key = exogenous_key
                self.index = 0
                self.view = SimpleNamespace(_uavs=EmptyStore(), _task_values=EmptyStore())
            def reset(self):
                self.index = 0
                return _public_observation(0.0)
            def step(self, action):
                self.index += 1
                observation = _public_observation(float(self.index))
                info = {"counts": {"completed": 0, "expired": 0}, "energy": {},
                        "completion_records": {}, "time": float(self.index), "feedback": "accepted"}
                return observation, 0.0, False, info

        class FakeAdapter:
            def prepare(self, observation): return observation, observation["mask"]
            def commit(self, _action): pass

        source = {"scenario": {"tape_id": "test"}, "exogenous_key": "source|repeat-0",
                  "scenario_sha256": "a" * 64}
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        ledger = BudgetLedger(Path(temporary.name) / "ledger.sqlite", REQUEST)
        self.addCleanup(ledger.close)
        ledger.select("world_model_data_collection")
        collector = ProductionDataCollector.__new__(ProductionDataCollector)
        collector.root = Path(".")
        collector.output = Path(".")
        collector.matrix = {"preference_configuration": {"task_confirmation": [0.8, 0.2],
                                                          "task_component_scale": 0.5},
                            "candidate_continuation_id": "one-step-public-transition-v1"}
        collector.ledger = ledger
        collector.boundary = Boundary()
        collector.tapes = [source]
        collector.M10Config = lambda: SimpleNamespace(uav_count=4, initial_energy=10.0, task_capacity=6)
        collector.M10Environment = FakeEnvironment
        collector.scenario_from_dict = lambda value: value
        collector.public_adapter_type = FakeAdapter
        collector.graph5_from_observation = lambda _obs: SimpleNamespace(
            action_mask=torch.tensor(_public_observation(4.0)["mask"]),
            candidate_features=torch.zeros((24, 4)),
            nodes={"uavs": np.zeros((4, 8), dtype=np.float32), "tasks": np.zeros((6, 8), dtype=np.float32)},
        )
        collector.transparent_score = lambda _graph, _action: {"score": 1.0, "energy_cost": 0.1}
        collector.vector_reward = lambda _info, counts, energy, _config: ([1.0, -0.1], None, counts, energy)
        result = collector._collect_unit({"parent": "train-test", "repeat": 0,
                                          "exogenous_key": "source|repeat-0", "scenario_sha256": "a" * 64},
                                         0, "train")
        self.assertEqual(result["status"], "complete")
        totals = ledger.snapshot()["totals"]
        self.assertEqual(totals, {"resets_upper": 1, "public_rule_decisions": 4, "snapshot_captures": 1,
                                  "branches": 3, "environment_steps": 7,
                                  "forced_first_actions": 3, "candidate_scans": 3})


if __name__ == "__main__":
    unittest.main()
