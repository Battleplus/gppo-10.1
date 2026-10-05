import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from budget_ledger import BudgetError, BudgetLedger
from phase_handshake import request_boundary


def request(stages=("staging_and_zero_step_gate", "collect", "train")):
    return {"stages": {name: {"complete_process_cpu_seconds": 10.0, "wall_seconds": 10.0}
                       for name in stages}}


class WorkerPhaseAccountingTests(unittest.TestCase):
    def fake_handshake(self, row, **_kwargs):
        return {"status": "acknowledged", "sequence": row["protocol_sequence"],
                "protocol_committed": True,
                "transition_committed": row["event"] == "transition"}

    def test_stage_entry_and_same_stage_snapshot_preserve_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = BudgetLedger(root / "ledger.sqlite3", request())
            with patch("phase_handshake.request_boundary", side_effect=self.fake_handshake) as handshake:
                with patch.object(ledger, "_process_cpu", side_effect=[
                    (0.3, "synthetic"), (0.4, "synthetic"),
                    (0.6, "synthetic"), (0.7, "synthetic"),
                ]):
                    ledger.select("staging_and_zero_step_gate")
                    entered_cpu = ledger.phase_started_cpu
                    sample = ledger.snapshot_phase(reason="repeated_poll")
                    self.assertEqual(ledger.phase_started_cpu, entered_cpu)
                    ledger.select("staging_and_zero_step_gate")
                    ledger.select("collect")
                    ledger.close()

            rows = [json.loads(line) for line in (root / "worker-phase-accounting.jsonl").read_text().splitlines()]
            self.assertEqual([row["event"] for row in rows], [
                "stage_enter", "snapshot", "transition", "snapshot",
            ])
            self.assertEqual([row["cpu_start_seconds"] for row in rows], [0.0, 0.0, 0.0, 0.6])
            self.assertEqual(rows[0]["cpu_seconds"], 0.0)
            self.assertAlmostEqual(rows[0]["stage_elapsed_cpu_seconds"], 0.3)
            self.assertAlmostEqual(sample["cpu_seconds"], 0.4)
            self.assertAlmostEqual(rows[2]["cpu_seconds"], 0.6)
            self.assertEqual(handshake.call_count, 2)
            self.assertEqual([call.args[0]["event"] for call in handshake.call_args_list], [
                "stage_enter", "transition",
            ])

    def test_stage_cpu_cap_is_not_borrowed_or_advanced_on_rejection(self):
        limits = request()
        limits["stages"]["collect"]["complete_process_cpu_seconds"] = 0.01
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(Path(directory) / "ledger.sqlite3", limits)
            with patch("phase_handshake.request_boundary", side_effect=self.fake_handshake) as handshake:
                with patch.object(ledger, "_process_cpu", side_effect=[
                    (0.1, "synthetic"), (0.2, "synthetic"), (0.4, "synthetic"),
                ]):
                    ledger.select("collect")
                    self.assertEqual(ledger.stage, "collect")
                    with self.assertRaisesRegex(BudgetError, "WORKER_PHASE_CAP_EXCEEDED:collect"):
                        ledger.select("train")
                    self.assertEqual(ledger.stage, "collect")
            self.assertEqual(handshake.call_count, 2)
            ledger.close(snapshot=False)

    def test_nonfinite_cpu_is_stop_before_any_handshake(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(Path(directory) / "ledger.sqlite3", request())
            with patch("phase_handshake.request_boundary") as handshake:
                with patch.object(ledger, "_process_cpu", return_value=(math.nan, "synthetic")):
                    with self.assertRaisesRegex(BudgetError, "NONFINITE"):
                        ledger.select("collect")
                handshake.assert_not_called()
            ledger.close(snapshot=False)

    def test_failure_snapshot_and_close_do_not_start_a_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = BudgetLedger(root / "ledger.sqlite3", request())
            with patch("phase_handshake.request_boundary", side_effect=self.fake_handshake) as handshake:
                with patch.object(ledger, "_process_cpu", side_effect=[
                    (0.1, "synthetic"), (0.2, "synthetic"), (0.3, "synthetic"),
                ]):
                    ledger.select("collect")
                    stage_origin = ledger.phase_started_cpu
                    sample = ledger.snapshot_phase(reason="failure_settlement")
                    self.assertEqual(ledger.phase_started_cpu, stage_origin)
                    ledger.close(snapshot=False)
            self.assertEqual(handshake.call_count, 2)
            self.assertEqual(sample["event"], "snapshot")
            self.assertEqual(sample["supervisor_handshake"]["scope"], "no_transition_snapshot")
            rows = [json.loads(line) for line in (root / "worker-phase-accounting.jsonl").read_text().splitlines()]
            self.assertEqual([row["event"] for row in rows], ["stage_enter", "transition", "snapshot"])

    def test_missing_supervisor_socket_is_not_a_production_bypass(self):
        with patch.dict(os.environ, {"W1_PHASE_ACCOUNTING_SOCKET": ""}):
            with self.assertRaisesRegex(RuntimeError, "PHASE_ACCOUNTING_SOCKET_REQUIRED"):
                request_boundary({"event": "stage_enter", "protocol_sequence": 1})


if __name__ == "__main__":
    unittest.main()
