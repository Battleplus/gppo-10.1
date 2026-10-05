from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from production_backend import FakeABBackend, TechnicalStop
from production_pipeline import execute_pipeline


class ProductionPipelineTests(unittest.TestCase):
    def run_case(self, mode: str):
        root = Path(tempfile.mkdtemp(prefix="ab-pipeline-test-"))
        output = root / "run-once"
        backend = FakeABBackend(root, output, mode=mode)
        result = execute_pipeline(backend, output)
        return result, output

    def test_prediction_gate_failure_has_zero_task_calls(self):
        result, output = self.run_case("gate_fail")
        self.assertEqual(result.status, "prediction_gate_stop")
        self.assertFalse(result.task_stage_executed)
        self.assertEqual(__import__("json").loads((output / "status.json").read_text())["task_calls"], 0)

    def test_effect_gate_pass_but_transparent_condition_stops(self):
        result, output = self.run_case("pass")
        self.assertEqual(result.status, "complete")
        self.assertTrue(result.task_stage_executed)
        self.assertEqual(__import__("json").loads((output / "task-comparison.json").read_text())["task_calls"], 32)

    def test_reuse_failure_stops_before_confirmation(self):
        result, output = self.run_case("reuse_fail")
        self.assertEqual(result.status, "technical_stop")
        self.assertFalse((output / "first-label.json").exists())

    def test_nan_is_fail_closed(self):
        root = Path(tempfile.mkdtemp(prefix="ab-pipeline-test-"))
        output = root / "run-once"
        backend = FakeABBackend(root, output, mode="nan")
        with self.assertRaises(TechnicalStop):
            execute_pipeline(backend, output)


if __name__ == "__main__":
    unittest.main()
