from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import native_launch
import runner
from budget_ledger import BudgetLedger


class LearningActivityEvidenceTests(unittest.TestCase):
    def test_failed_learning_operation_is_unknown_not_false(self):
        request = {"stages": {"learning": {"model_initializations_or_loads": 1}},
                   "totals": {"model_initializations_or_loads": 1}}
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(Path(directory) / "test.sqlite3", request)
            try:
                ledger.select("learning")
                def failing_operation():
                    raise RuntimeError("synthetic failure after operation entry")
                with self.assertRaises(RuntimeError):
                    ledger.call("synthetic.initialize", {"model_initializations_or_loads": 1}, failing_operation)
                evidence = runner.learning_activity_evidence(ledger)
                self.assertIsNone(evidence["model_initialized"])
                self.assertFalse(evidence["training_started"])
                self.assertEqual(evidence["learning_call_evidence"]["failed"]["model_initializations_or_loads"], 1)
            finally:
                ledger.close()

    def test_completed_activity_survives_later_failure_without_model_compute(self):
        resources = {"model_initializations_or_loads": 1, "checkpoint_loads": 1,
                     "world_backward_calls": 1}
        request = {"stages": {"learning": resources}, "totals": resources}
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(Path(directory) / "test.sqlite3", request)
            try:
                ledger.select("learning")
                for resource in resources:
                    ledger.call("synthetic." + resource, {resource: 1}, lambda: None)
                evidence = runner.learning_activity_evidence(ledger)
                for flag in ("model_initialized", "checkpoint_loaded", "training_started"):
                    self.assertTrue(evidence[flag])
                self.assertEqual(evidence["learning_call_evidence"]["complete"], resources)
                output = Path(directory) / "isolated-output"
                output.mkdir()
                stop = runner._settlement(ledger, status="technical_stop", attempt="synthetic-accounting",
                                          output=output, error=RuntimeError("synthetic later failure"))
                self.assertEqual(stop["status"], "technical_stop")
                self.assertTrue((output / "resource-settlement.json").is_file())
            finally:
                ledger.close()


@unittest.skipIf(os.name == "nt", "runner durable bootstrap requires native Linux directory fsync")
class RunnerBootstrapTests(unittest.TestCase):
    def test_output_and_initial_state_exist_before_sqlite_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run-once"
            runner.initialize_output(output, "attempt-fixture")
            observed = {}

            class LedgerSpy:
                def __init__(self, path, request):
                    observed["path"] = path
                    observed["status_exists"] = (path.parent / "status.json").is_file()
                    observed["output_is_dir"] = path.parent.is_dir()

                def close(self):
                    pass

            with patch.object(runner, "BudgetLedger", LedgerSpy):
                ledger = runner.initialize_ledger(output, {"stages": {}, "totals": {}})

            self.assertIsInstance(ledger, LedgerSpy)
            self.assertTrue(observed["status_exists"])
            self.assertTrue(observed["output_is_dir"])
            self.assertEqual(observed["path"], output / "budget.sqlite3")

    def test_existing_output_rejects_duplicate_worker_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run-once"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                runner.initialize_output(output, "attempt-fixture")

    def test_contract_read_failure_is_settled_after_output_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run-once"
            (root / "RESOURCE_REQUEST.json").write_text(json.dumps({
                "attempt": "attempt-fixture", "status": "NOT_APPROVED",
            }), encoding="utf-8")
            with patch.object(runner, "ROOT", root), patch.object(runner, "OUTPUT", output), \
                    patch.dict(os.environ, {"W1_VERIFIED_ATTEMPT": "attempt-fixture"}, clear=False):
                self.assertEqual(runner.main(), 1)
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            settlement = json.loads((output / "resource-settlement.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "technical_stop")
            self.assertEqual(settlement["status"], "technical_stop")
            self.assertEqual(settlement["attempt"], "attempt-fixture")
            self.assertIsNone(settlement["ledger"])

    def test_existing_attempt_is_not_modified_after_duplicate_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run-once"
            output.mkdir()
            marker = output / "status.json"
            marker.write_text("preserve", encoding="utf-8")
            with patch.object(runner, "ROOT", root), patch.object(runner, "OUTPUT", output), \
                    patch.dict(os.environ, {"W1_VERIFIED_ATTEMPT": "attempt-fixture"}, clear=False):
                self.assertEqual(runner.main(), 1)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_native_launcher_passes_verified_attempt_to_supervisor_worker(self):
        observed = {}
        supervisor = ModuleType("supervise")

        def fake_supervisor_main(root, runner_name):
            observed["attempt"] = os.environ.get("W1_VERIFIED_ATTEMPT")
            observed["runner"] = runner_name
            return 17

        supervisor.main = fake_supervisor_main
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(native_launch, "ROOT", Path(directory)), \
                patch.object(native_launch, "verify_native_launch", return_value="attempt-fixture"), \
                patch.object(native_launch.sys, "argv", ["native_launch.py"]), \
                patch.dict(os.environ, {"W1_VERIFIED_ATTEMPT": ""}, clear=False), \
                patch.dict("sys.modules", {"supervise": supervisor}):
            self.assertEqual(native_launch.main(), 17)
            self.assertTrue((Path(directory) / "runtime-output" / "native-launcher-accounting.json").is_file())
        self.assertEqual(observed, {"attempt": "attempt-fixture", "runner": "runner.py"})

    def test_missing_verified_identity_records_preoutput_stop_without_attempt_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run-once"
            with patch.object(runner, "ROOT", root), patch.object(runner, "OUTPUT", output), \
                    patch.dict(os.environ, {"W1_VERIFIED_ATTEMPT": ""}, clear=False):
                self.assertEqual(runner.main(), 1)
            evidence = json.loads((root / "worker-initialization-failure.json").read_text(encoding="utf-8"))
            self.assertEqual(evidence["stage"], "runner_output_initialization")
            self.assertFalse(output.exists())
            self.assertFalse((root / "execution.lock").exists())
            self.assertFalse((root / "budget.sqlite3").exists())

    def test_environment_construction_uses_success_evidence_not_reserved_resets(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            evidence = runner.environment_construction_evidence(output)
            self.assertFalse(evidence["environment_constructed"])
            self.assertEqual(evidence["environment_construction_status"], "not_started")

            marker = output / "environment-construction.jsonl"
            marker.write_text(json.dumps({"sequence": 1, "event": "attempted"}) + "\n", encoding="utf-8")
            evidence = runner.environment_construction_evidence(output)
            self.assertIsNone(evidence["environment_constructed"])
            self.assertEqual(evidence["environment_construction_status"], "indeterminate_after_construction_attempt")

            with marker.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"sequence": 1, "event": "constructed"}) + "\n")
            evidence = runner.environment_construction_evidence(output)
            self.assertTrue(evidence["environment_constructed"])
            self.assertEqual(evidence["environment_construction_status"], "constructed")

    def test_invalid_environment_construction_evidence_is_not_reported_as_false(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "environment-construction.jsonl").write_text("not-json\n", encoding="utf-8")
            evidence = runner.environment_construction_evidence(output)
            self.assertIsNone(evidence["environment_constructed"])
            self.assertEqual(evidence["environment_construction_status"], "evidence_unavailable")

    def test_settlement_error_is_technical_stop_and_never_a_success_status(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            observed = {}

            class PendingLedger:
                def assert_settled(self):
                    raise RuntimeError("pending calls")

                def snapshot(self):
                    return {"pending_calls": 1, "failed_calls": 0, "totals": {}}

            settlement = runner._settlement(
                PendingLedger(), status="label_coverage_pass", attempt="attempt-fixture", output=output,
            )
            observed.update(settlement)
            self.assertEqual(observed["status"], "technical_stop")
            self.assertEqual(observed["ledger"]["pending_calls"], 1)
            self.assertIsNotNone(observed["ledger_settlement_error"])
            persisted = json.loads((output / "resource-settlement.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"], "technical_stop")


if __name__ == "__main__":
    unittest.main()
