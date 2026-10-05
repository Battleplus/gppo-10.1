from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import runner
from task_pipeline import (
    TaskPipelineError, _DirectBoundary, _metric_matches, _validated_checkpoints,
    run_pipeline, validate_task_configuration,
)


ROOT = Path(__file__).resolve().parent


def read_json(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


class TaskPipelineContractTests(unittest.TestCase):
    def test_frozen_task_configuration_has_only_task_worker_stages(self):
        validate_task_configuration(read_json("experiment-matrix.json"), read_json("RESOURCE_REQUEST.json"))

    def test_collection_stage_is_rejected_before_pipeline_entry(self):
        matrix = read_json("experiment-matrix.json")
        request = copy.deepcopy(read_json("RESOURCE_REQUEST.json"))
        request["stages"]["label_qualification"] = {"environment_steps": 1}
        with self.assertRaisesRegex(TaskPipelineError, "TASK_STAGE_SET_MISMATCH"):
            validate_task_configuration(matrix, request)

    def test_reduced_policy_fixture_requires_opt_in_and_exact_ledger_budget(self):
        matrix = copy.deepcopy(read_json("experiment-matrix.json"))
        request = copy.deepcopy(read_json("RESOURCE_REQUEST.json"))
        matrix["e2e_fixture"] = {
            "enabled": True,
            "mode": "synthetic_test",
            "policy_training_steps_per_method_seed": 64,
            "policy_rollout_steps": 64,
            "policy_optimizer_updates_per_method_seed": 1,
        }
        request["stages"]["conditional_policy_training"]["policy_optimizer_updates"] = 9
        with self.assertRaisesRegex(TaskPipelineError, "TASK_INTEGRATION_FIXTURE_REJECTED"):
            validate_task_configuration(matrix, request)
        validate_task_configuration(matrix, request, allow_integration=True)

        request["stages"]["conditional_policy_training"]["policy_optimizer_updates"] = 1152
        with self.assertRaisesRegex(TaskPipelineError, "TASK_STAGE_LIMIT_MISMATCH"):
            validate_task_configuration(matrix, request, allow_integration=True)

    def test_pipeline_formal_default_rejects_fixture_before_runtime(self):
        matrix = copy.deepcopy(read_json("experiment-matrix.json"))
        request = copy.deepcopy(read_json("RESOURCE_REQUEST.json"))
        matrix["e2e_fixture"] = {
            "enabled": True,
            "mode": "synthetic_test",
            "policy_training_steps_per_method_seed": 64,
            "policy_rollout_steps": 64,
            "policy_optimizer_updates_per_method_seed": 1,
        }
        request["stages"]["conditional_policy_training"]["policy_optimizer_updates"] = 9
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run-once"
            with self.assertRaisesRegex(TaskPipelineError, "TASK_INTEGRATION_FIXTURE_REJECTED"):
                run_pipeline(ROOT, output, matrix, request, ledger=None)
            self.assertFalse(output.exists())

    def test_fixture_budgets_must_be_reduced_and_update_aligned(self):
        matrix = copy.deepcopy(read_json("experiment-matrix.json"))
        request = copy.deepcopy(read_json("RESOURCE_REQUEST.json"))
        matrix["e2e_fixture"] = {
            "enabled": True,
            "mode": "synthetic_test",
            "policy_training_steps_per_method_seed": 128,
            "policy_rollout_steps": 64,
            "policy_optimizer_updates_per_method_seed": 1,
        }
        request["stages"]["conditional_policy_training"]["policy_optimizer_updates"] = 9
        with self.assertRaisesRegex(TaskPipelineError, "TASK_FIXTURE_POLICY_BUDGET_INVALID"):
            validate_task_configuration(matrix, request, allow_integration=True)
        matrix["e2e_fixture"]["policy_training_steps_per_method_seed"] = 2048
        with self.assertRaisesRegex(TaskPipelineError, "TASK_FIXTURE_POLICY_BUDGET_INVALID"):
            validate_task_configuration(matrix, request, allow_integration=True)

    def test_synthetic_checkpoint_routes_are_bound_without_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            routes = []
            checkpoints = {}
            for method in ("G0", "T", "G1"):
                for seed, world_seed in ((8301, 8201), (8302, 8202), (8303, 8203)):
                    path = output / f"{method}-{seed}.synthetic"
                    path.write_text("synthetic path marker", encoding="utf-8")
                    key = (method, seed)
                    route = {
                        "method": method, "seed": seed,
                        "world_model_variant": "G1" if method == "G1" else None,
                        "world_model_seed": world_seed if method == "G1" else None,
                        "training_steps": 2048, "optimizer_updates": 128,
                        "checkpoint": str(path),
                        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    routes.append(route)
                    checkpoints[key] = str(path)

            actual, hashes = _validated_checkpoints({"routes": routes, "checkpoints": checkpoints}, output)
            self.assertEqual(len(actual), 9)
            expected_path = str((output / "G1-8302.synthetic").resolve())
            self.assertEqual(actual[("G1", 8302)], expected_path)
            self.assertEqual(hashes[expected_path], hashlib.sha256((output / "G1-8302.synthetic").read_bytes()).hexdigest())
            reduced_routes = copy.deepcopy(routes)
            for route in reduced_routes:
                route["training_steps"] = 64
                route["optimizer_updates"] = 1
            _validated_checkpoints(
                {"routes": reduced_routes, "checkpoints": checkpoints}, output,
                expected_steps=64, expected_updates=1,
            )
            with self.assertRaisesRegex(TaskPipelineError, "ROUTE_BUDGET_MISMATCH"):
                _validated_checkpoints(
                    {"routes": reduced_routes, "checkpoints": checkpoints}, output,
                )
            routes[0]["checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(TaskPipelineError, "CHECKPOINT_DIGEST_MISMATCH"):
                _validated_checkpoints({"routes": routes, "checkpoints": checkpoints}, output)

    def test_checkpoint_path_escape_and_world_seed_mispairing_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            output, external = Path(directory), Path(outside) / "checkpoint.synthetic"
            external.write_text("synthetic marker", encoding="utf-8")
            routes = []
            checkpoints = {}
            for method in ("G0", "T", "G1"):
                for seed, world_seed in ((8301, 8201), (8302, 8202), (8303, 8203)):
                    path = external if (method, seed) == ("G1", 8301) else output / f"{method}-{seed}.synthetic"
                    if path != external:
                        path.write_text("synthetic marker", encoding="utf-8")
                    paired_seed = world_seed
                    if (method, seed) == ("G1", 8302):
                        paired_seed = 8203
                    route = {
                        "method": method, "seed": seed,
                        "world_model_variant": "G1" if method == "G1" else None,
                        "world_model_seed": paired_seed if method == "G1" else None,
                        "training_steps": 2048, "optimizer_updates": 128,
                        "checkpoint": str(path),
                        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    routes.append(route)
                    checkpoints[(method, seed)] = str(path)
            with self.assertRaisesRegex(TaskPipelineError, "WORLD_BINDING_MISMATCH"):
                _validated_checkpoints({"routes": routes, "checkpoints": checkpoints}, output)

            routes[7]["world_model_seed"] = 8202
            with self.assertRaisesRegex(TaskPipelineError, "CHECKPOINT_PATH_INVALID"):
                _validated_checkpoints({"routes": routes, "checkpoints": checkpoints}, output)

    def test_hook_metrics_must_match_raw_reward_recomputation(self):
        metrics = {
            "g1_minus_g0_macro_utility": 0.2,
            "g1_minus_t_macro_utility": 0.1,
            "g1_minus_h_macro_utility": 0.03,
        }
        self.assertTrue(_metric_matches(metrics, metrics))
        independent = dict(metrics, g1_minus_t_macro_utility=0.1001)
        self.assertFalse(_metric_matches(metrics, independent))
        self.assertFalse(_metric_matches(metrics, dict(metrics, g1_minus_t_macro_utility=float("nan"))))

    def test_formal_boundary_executes_only_the_supplied_bottom_operation(self):
        boundary = _DirectBoundary()
        observed = []
        result = boundary.environment(lambda value: observed.append(value) or value, "synthetic")
        self.assertEqual(result, "synthetic")
        self.assertEqual(observed, ["synthetic"])

    def test_runner_has_no_collection_or_world_training_entrypoint(self):
        tree = ast.parse((ROOT / "runner.py").read_text(encoding="utf-8"))
        imported = set()
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called.add(node.func.id)
        self.assertNotIn("production_data", imported)
        self.assertNotIn("joint_pipeline", imported)
        self.assertIn("run_pipeline", called)
        self.assertIn("task_contract", imported)

    def test_pipeline_keeps_training_before_confirmation(self):
        tree = ast.parse((ROOT / "task_pipeline.py").read_text(encoding="utf-8"))
        selected = [
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "select"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ]
        self.assertLess(selected.index("conditional_policy_training"),
                        selected.index("conditional_task_confirmation"))


if __name__ == "__main__":
    unittest.main()
