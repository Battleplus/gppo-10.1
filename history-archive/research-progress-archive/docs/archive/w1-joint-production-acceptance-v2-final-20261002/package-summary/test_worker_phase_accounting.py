import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from budget_ledger import BudgetError, BudgetLedger


class WorkerPhaseAccountingTests(unittest.TestCase):
    def test_short_stage_is_recorded_even_without_monitor_poll(self):
        request = {"stages": {name: {"complete_process_cpu_seconds": 10.0, "wall_seconds": 10.0}
                              for name in ("staging_and_zero_step_gate", "collect", "train", "settlement_and_verified_export")}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            ledger = BudgetLedger(path / "ledger.sqlite3", request)
            with patch.object(ledger, "_process_cpu", side_effect=[(0.1, "synthetic"), (0.3, "synthetic"), (0.4, "synthetic")]):
                ledger.select("collect")
                ledger.select("train")
                ledger.close()
            rows = [json.loads(line) for line in (path / "worker-phase-accounting.jsonl").read_text().splitlines()]
            self.assertEqual([row["stage"] for row in rows], ["staging_and_zero_step_gate", "collect", "train"])
            self.assertAlmostEqual(sum(row["cpu_seconds"] for row in rows), 0.4)

    def test_stage_cpu_cap_is_not_borrowed(self):
        request = {"stages": {"collect": {"complete_process_cpu_seconds": 0.01}, "train": {"complete_process_cpu_seconds": 100}}}
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(Path(directory) / "ledger.sqlite3", request)
            with patch.object(ledger, "_process_cpu", side_effect=[(0.1, "synthetic"), (0.3, "synthetic")]):
                ledger.select("collect")
                with self.assertRaisesRegex(BudgetError, "WORKER_PHASE_CAP_EXCEEDED:collect"):
                    ledger.select("train")
            ledger.connection.close()

    def test_nonfinite_cpu_is_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(Path(directory) / "ledger.sqlite3", {"stages": {"collect": {}}})
            with patch.object(ledger, "_process_cpu", return_value=(math.nan, "synthetic")):
                with self.assertRaisesRegex(BudgetError, "NONFINITE"):
                    ledger.select("collect")
            ledger.connection.close()


if __name__ == "__main__":
    unittest.main()
