from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from production_backend import ABRuntimeBackend
from production_pipeline import execute_pipeline
from runtime_adapter import ProductionRuntimeAdapter, RuntimeAdapterError


ROOT = Path(__file__).resolve().parent


def record(parent: str, role: str, decision: str, action: int, legal=(0, 1), target=None) -> dict:
    return {
        "sample_id": f"{decision}:{action}", "parent": parent, "repeat": 0,
        "decision_id": decision, "split_role": role, "action": action,
        "legal_actions": list(legal), "flat": [0.0] * 770, "history": [0.0] * 32,
        "remaining_utility": float(action if target is None else target),
        "current_public_score": 0.0, "transparent_history_score": 0.0,
        "continuation_id": "hungarian-v1-fixed", "is_true_branch": True,
        "training_eligible": True,
    }


class FakeModel:
    def __init__(self, seed_offset: float):
        self.seed_offset = seed_offset

    def __call__(self, features, **kwargs):
        # Works with both the list fallback and a torch tensor.  The last 25
        # fields encode the candidate action identity.
        rows = features.tolist() if hasattr(features, "tolist") else features
        return [[float(max(range(25), key=lambda i: row[-25 + i])) + self.seed_offset] for row in rows]


class Boundary:
    def __init__(self, records=None):
        self.records = records
        self.model_calls = []
        self.train_calls = []

    def model(self, operation, *args, **kwargs):
        self.model_calls.append((operation, args, kwargs))
        return operation(*args, **kwargs)

    def environment(self, operation, *args, **kwargs):
        return operation(*args, **kwargs)

    def optimizer(self, operation, *args, **kwargs):
        return operation(*args, **kwargs)

    def checkpoint(self, operation, *args, **kwargs):
        return operation(*args, **kwargs)

    def zero_step(self):
        return {"pass": True, "environment_constructed": False, "model_constructed": False}

    def collect_records(self, _output, _matrix):
        return list(self.records or [])

    def train_variant(self, variant, seed, _train, _selection):
        self.train_calls.append((variant, seed))
        return FakeModel(float(seed) / 10000.0), {"variant": variant, "seed": seed}


class ImplementationRepairTests(unittest.TestCase):
    def adapter(self, output: Path, boundary: Boundary | None = None, account=None):
        request = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
        matrix = json.loads((ROOT / "experiment-matrix.json").read_text(encoding="utf-8"))
        calls = []
        def accounted(name, amounts, function, *args, **kwargs):
            calls.append((name, dict(amounts), args, kwargs))
            return function(*args, **kwargs)
        adapter = ProductionRuntimeAdapter(ROOT, output, request, matrix,
                                           account or accounted,
                                           boundary=boundary or Boundary())
        adapter._test_calls = calls
        return adapter

    def test_keyword_arguments_reach_operation_and_account_once(self):
        output = Path(tempfile.mkdtemp())
        boundary = Boundary()
        adapter = self.adapter(output, boundary)
        observed = []
        def loader(path, *, map_location, weights_only):
            observed.append((path, map_location, weights_only))
            return "loaded"
        result = adapter._accounted("checkpoint.load", {"model_initializations_or_loads": 1},
                                    loader, "model.pt", map_location="cpu", weights_only=True)
        self.assertEqual(result, "loaded")
        self.assertEqual(observed, [("model.pt", "cpu", True)])
        self.assertEqual(len(adapter._test_calls), 1)
        self.assertEqual(adapter._test_calls[0][1], {"model_initializations_or_loads": 1})

    def test_prediction_loop_batches_each_window_and_maps_by_action(self):
        output = Path(tempfile.mkdtemp())
        rows = []
        for index in range(24):
            parent = f"eval-{index:02d}"
            decision = f"prediction:{parent}:r0:s0"
            legal = (1, 0) if index % 2 else (0, 1)
            for action in legal:
                rows.append(record(parent, "prediction_evaluation", decision, action, legal=legal))
        (output / "learning-records.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        boundary = Boundary()
        adapter = self.adapter(output, boundary)
        adapter._models = {"A": [FakeModel(0.0), FakeModel(0.1), FakeModel(0.2)],
                           "B": [FakeModel(1.0), FakeModel(1.1), FakeModel(1.2)]}
        result = adapter.evaluate_predictions()
        self.assertEqual(result["trace_rows"], 24 * 2 * 2)
        self.assertEqual(len(adapter._test_calls), 24 * 2 * 3)
        self.assertTrue(all(call[1]["model_batch_forwards"] == 1 for call in adapter._test_calls))
        self.assertTrue(all(call[1]["model_sample_evaluations"] == 2 for call in adapter._test_calls))
        traces = [json.loads(line) for line in (output / "per-candidate-predictions.jsonl").read_text().splitlines()]
        self.assertEqual(len(traces), 96)
        by_action = {(row["variant"], row["decision_id"], row["action"]): row for row in traces}
        self.assertAlmostEqual(by_action[("A", "prediction:eval-01:r0:s0", 0)]["ensemble_prediction"], 0.1)
        self.assertAlmostEqual(by_action[("A", "prediction:eval-01:r0:s0", 1)]["ensemble_prediction"], 1.1)

    def confirmation_records(self, count: int):
        selection = json.loads((ROOT / "new-prediction-parent-selection.json").read_text(encoding="utf-8"))
        rows = []
        for item in selection["parents"][:count]:
            parent = item["parent"]
            decision = f"prediction_evaluation:{parent}:r0:s0"
            rows.extend([record(parent, "prediction_evaluation", decision, 0),
                         record(parent, "prediction_evaluation", decision, 1)])
        return rows

    def test_confirmation_gate_zero_seven_eight_parent_cases(self):
        for count in (0, 7):
            with self.subTest(parent_count=count):
                output = Path(tempfile.mkdtemp())
                boundary = Boundary(self.confirmation_records(count))
                adapter = self.adapter(output, boundary)
                result = adapter.collect_confirmation_labels()
                self.assertFalse(result["pass"])
                self.assertEqual(boundary.train_calls, [])

        output = Path(tempfile.mkdtemp())
        boundary = Boundary(self.confirmation_records(8))
        adapter = self.adapter(output, boundary)
        result = adapter.collect_confirmation_labels()
        self.assertTrue(result["pass"])
        self.assertEqual(result["new_confirmation_parents"], 8)
        self.assertEqual(result["missing_parents"], [])

    def test_confirmation_gate_blocks_formal_pipeline_before_training_or_tasks(self):
        selection = json.loads((ROOT / "new-prediction-parent-selection.json").read_text(encoding="utf-8"))
        for count in (0, 7):
            output = Path(tempfile.mkdtemp()) / "run-once"
            request = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
            matrix = json.loads((ROOT / "experiment-matrix.json").read_text(encoding="utf-8"))
            rows = []
            for item in selection["parents"][:count]:
                decision = f"prediction_evaluation:{item['parent']}:r0:s0"
                rows.extend([record(item["parent"], "prediction_evaluation", decision, 0),
                             record(item["parent"], "prediction_evaluation", decision, 1)])
            boundary = Boundary(rows)
            factory = lambda root, out, req, mat, account, _boundary: ProductionRuntimeAdapter(
                root, out, req, mat, account, boundary=boundary)
            backend = ABRuntimeBackend(ROOT, output, request, matrix, adapter_factory=factory)
            result = execute_pipeline(backend, output)
            backend.close()
            self.assertEqual(result.status, "confirmation_stop")
            self.assertEqual(boundary.train_calls, [])
            self.assertEqual(backend.task_calls, 0)

    def test_prediction_nan_and_candidate_mismatch_stop(self):
        output = Path(tempfile.mkdtemp())
        rows = [record("eval-00", "prediction_evaluation", "d", 0),
                record("eval-00", "prediction_evaluation", "d", 1)]
        (output / "learning-records.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        class BadModel:
            def __call__(self, features, **_kwargs):
                return [[float("nan")] for _ in (features.tolist() if hasattr(features, "tolist") else features)]
        adapter = self.adapter(output, Boundary())
        adapter._models = {"A": [BadModel(), BadModel(), BadModel()],
                           "B": [BadModel(), BadModel(), BadModel()]}
        with self.assertRaises(RuntimeAdapterError):
            adapter.evaluate_predictions()

        mismatch = Path(tempfile.mkdtemp())
        bad_rows = [record("eval-00", "prediction_evaluation", "d", 0, legal=(0, 1, 2))]
        (mismatch / "learning-records.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in bad_rows), encoding="utf-8")
        adapter = self.adapter(mismatch, Boundary())
        adapter._models = {"A": [FakeModel(0.0)] * 3, "B": [FakeModel(0.0)] * 3}
        with self.assertRaises(Exception):
            adapter.evaluate_predictions()


if __name__ == "__main__":
    unittest.main()
