from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from production_backend import ABRuntimeBackend
from production_pipeline import execute_pipeline
from runtime_adapter import ProductionRuntimeAdapter
import runner
from unittest import mock


def record(parent: str, role: str, action: int, decision: str, target: float,
           transparent: float) -> dict:
    return {
        "sample_id": f"{decision}:{action}", "parent": parent, "repeat": 0,
        "decision_id": decision, "split_role": role, "action": action,
        "legal_actions": [0, 1], "flat": [0.0] * 770, "history": [0.0] * 32,
        "remaining_utility": target, "current_public_score": 0.0,
        "transparent_history_score": transparent,
        "continuation_id": "hungarian-v1-fixed", "is_true_branch": True,
        "training_eligible": True,
    }


class ProductionBoundary:
    """Bottom-only substitute: no environment/model/optimizer is constructed."""

    def __init__(self, gate: str = "pass"):
        self.gate = gate
        self.task_calls = 0
        self.training_calls: list[tuple[str, int]] = []
        self.task_model_map = None

    def zero_step(self):
        return {"pass": True, "runtime_modules_loaded": True,
                "environment_constructed": False, "model_constructed": False}

    def collect_records(self, output: Path, _matrix: dict):
        rows = []
        for role, parents in (("train", ["train-test"]), ("model_selection", ["select-test"]),
                              ("prediction_evaluation", [f"eval-{index}" for index in range(8)])):
            for parent in parents:
                decision = f"{role}-{parent}-decision"
                rows.extend([record(parent, role, 0, decision, 0.0, 1.0),
                             record(parent, role, 1, decision, 1.0, 0.0)])
        return rows

    def train_variant(self, variant: str, seed: int, _train, _selection):
        self.training_calls.append((variant, seed))
        return {"variant": variant, "seed": seed, "updates": 1}, {"variant": variant, "seed": seed}

    def predict_variant(self, variant: str, row, _models):
        if self.gate == "fail":
            values = [1.0, 0.0]
        elif variant == "A":
            values = [1.0, 0.0]
        else:
            values = [0.0, 1.0]
        return [values[row.action]] * 3

    def task_comparison(self, models, _matrix):
        self.task_calls += 1
        self.task_model_map = models
        return {"summaries": [{"identity": {"arm": arm}, "discounted_utility": 0.0}
                              for arm in ("hungarian", "transparent_one_shot", "A_one_shot", "B_one_shot")],
                "task_calls": 4, "pairing": [], "test_bottom_boundary": True}


class ProductionBackendCompletionTests(unittest.TestCase):
    root = Path(__file__).resolve().parent

    def run_case(self, mode: str):
        output = Path(tempfile.mkdtemp(prefix="production-ab-")) / "run-once"
        request = json.loads((self.root / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
        matrix = json.loads((self.root / "experiment-matrix.json").read_text(encoding="utf-8"))
        probe = ProductionBoundary(mode)
        factory = lambda root, out, req, mat, account, _boundary: ProductionRuntimeAdapter(
            root, out, req, mat, account, boundary=probe)
        backend = ABRuntimeBackend(self.root, output, request, matrix, adapter_factory=factory)
        result = execute_pipeline(backend, output)
        backend.close()
        return result, output, probe

    def test_production_backend_routes_six_models_and_task_after_both_gates(self):
        result, output, probe = self.run_case("pass")
        self.assertEqual(result.status, "complete")
        self.assertTrue(result.task_stage_executed)
        self.assertEqual(len(probe.training_calls), 6)
        self.assertEqual(probe.task_calls, 1)
        self.assertEqual(set(probe.task_model_map), {"A", "B"})
        self.assertEqual(len(probe.task_model_map["A"]), 3)
        self.assertEqual(len(probe.task_model_map["B"]), 3)
        self.assertTrue((output / "per-candidate-predictions.jsonl").is_file())
        self.assertTrue(json.loads((output / "settlement.json").read_text())["production_backend"])

    def test_prediction_gate_failure_keeps_task_calls_zero(self):
        result, output, probe = self.run_case("fail")
        self.assertEqual(result.status, "prediction_gate_stop")
        self.assertFalse(result.task_stage_executed)
        self.assertEqual(probe.task_calls, 0)
        self.assertEqual(json.loads((output / "status.json").read_text())["task_calls"], 0)

    def test_formal_runner_identity_reaches_production_backend(self):
        output = Path(tempfile.mkdtemp(prefix="production-runner-")) / "run-once"
        probe = ProductionBoundary("fail")
        factory = lambda root, out, req, mat, account, _boundary: ProductionRuntimeAdapter(
            root, out, req, mat, account, boundary=probe)
        with mock.patch.object(runner, "OUT", output):
            code = runner.main(adapter_factory=factory)
        self.assertEqual(code, 0)
        self.assertTrue((output / "production-result.json").is_file())
        self.assertEqual(json.loads((output / "status.json").read_text())["task_calls"], 0)


if __name__ == "__main__":
    unittest.main()
