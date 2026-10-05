"""Real Linux supervisor, socket, process, timeout, log, and export checks."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

from infra_io import controlled_export, durable_atomic_json

PACKAGE = Path(__file__).resolve().parent


@unittest.skipUnless(sys.platform.startswith("linux"), "requires native Linux processes and Unix sockets")
class SupervisedPhaseIntegrationTests(unittest.TestCase):
    def make_fixture(self, parent, worker_code, *, root_name=None, wall_cap=20):
        root = Path(parent) / (root_name or "a" * 215)
        root.mkdir(parents=True)
        for name in ("supervise.py", "phase_handshake.py", "linux_process_scope.py",
                     "infra_io.py", "worker_bootstrap.py", "budget_ledger.py"):
            shutil.copyfile(PACKAGE / name, root / name)
        (root / "fixture_worker.py").write_text(textwrap.dedent(worker_code), encoding="utf-8")
        request = json.loads((PACKAGE / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
        request.pop("gpu", None)
        request["totals"].update(wall_seconds=30, complete_process_cpu_seconds=30,
                                 active_storage_bytes=128 * 1024 * 1024,
                                 all_resident_rss_bytes=1024 * 1024 * 1024)
        for stage in request["stages"].values():
            stage["wall_seconds"] = wall_cap
            stage["complete_process_cpu_seconds"] = 20
            stage["active_storage_bytes"] = 64 * 1024 * 1024
        request["accounting_reserves"]["cross_system_wall_seconds"] = 5
        (root / "RESOURCE_REQUEST.json").write_text(json.dumps(request), encoding="utf-8")
        (root / "experiment-matrix.json").write_text('{"world_model_device":"cpu"}', encoding="utf-8")
        return root

    def invoke_supervisor(self, root, *, worker_input=None):
        script = textwrap.dedent("""
            import json, os
            import supervise
            root = os.getcwd()
            code = supervise.main(root, 'fixture_worker.py', sample_interval=0.01,
                                  worker_input=""" + repr(worker_input) + """)
            print(json.dumps({'returncode': code}))
        """)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script], cwd=root, env=env,
            text=True, capture_output=True, timeout=45, check=False,
        )
        status_path = root / "supervisor-status.json"
        state = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else None
        return completed, state

    def test_short_subprocesses_and_stage_boundaries_through_real_supervisor(self):
        worker = """
            from pathlib import Path
            import json, subprocess, sys
            from budget_ledger import BudgetLedger
            root = Path.cwd()
            output = root / 'run-once'
            output.mkdir()
            request = json.loads((root / 'RESOURCE_REQUEST.json').read_text())
            ledger = BudgetLedger(output / 'budget.sqlite3', request)
            def burn(seconds):
                code = 'import time; end=time.process_time()+' + repr(seconds) + '\\nwhile time.process_time()<end: pass'
                subprocess.run([sys.executable, '-B', '-c', code], check=True)
            ledger.select('label_qualification')
            burn(0.03)
            ledger.select('world_model_training_and_selection')
            burn(0.03)
            ledger.select('prediction_confirmation')
            burn(0.03)
            ledger.select('settlement_and_verified_export')
            ledger.close()
        """
        with tempfile.TemporaryDirectory(prefix="w1-supervised-cpu-") as directory:
            root = self.make_fixture(directory, worker)
            completed, state = self.invoke_supervisor(root)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIsNotNone(state, completed.stdout + completed.stderr)
            self.assertEqual(state["status"], "complete", state)
            rows = [json.loads(line) for line in (root / "run-once" / "worker-phase-accounting.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["stage"] for row in rows], [
                "staging_and_zero_step_gate", "label_qualification",
                "world_model_training_and_selection", "prediction_confirmation",
                "settlement_and_verified_export"])
            for row in rows[:-1]:
                self.assertEqual(row["supervisor_handshake"]["status"], "acknowledged")
            for row in rows[1:4]:
                self.assertGreater(row["cpu_seconds"], 0.025)
            self.assertGreater(state["cpu_seconds"], sum(row["cpu_seconds"] for row in rows))
            evidence_dir = os.environ.get("W1_TEST_EVIDENCE_DIR")
            if evidence_dir:
                durable_atomic_json(Path(evidence_dir) / "supervised-phase-integration.json", {
                    "supervisor": state,
                    "worker_phases": rows,
                    "real_environment_calls": 0,
                    "model_calls": 0,
                    "synthetic_cpu_subprocesses": 3,
                    "commanded_cpu_work_seconds": 0.09,
                    "hardware_guard_only_injected": False,
                })

    def test_real_supervised_boundaries_at_long_final_path_and_export(self):
        worker = """
            from pathlib import Path
            import json, os, subprocess, sys
            from budget_ledger import BudgetLedger
            root = Path.cwd()
            output = root / 'run-once'; output.mkdir()
            request = json.loads((root / 'RESOURCE_REQUEST.json').read_text())
            ledger = BudgetLedger(output / 'budget.sqlite3', request)
            for stage in ('label_qualification', 'world_model_training_and_selection',
                          'prediction_confirmation', 'settlement_and_verified_export'):
                ledger.select(stage)
                if stage != 'settlement_and_verified_export':
                    subprocess.run([sys.executable, '-B', '-c',
                                    'import time; end=time.process_time()+0.025\\nwhile time.process_time()<end: pass'],
                                   check=True)
            ledger.close()
            (output / 'fixture-export.json').write_text(json.dumps({'verified': True}))
        """
        with tempfile.TemporaryDirectory(prefix="w1-phase-parent-") as parent:
            root = self.make_fixture(parent, worker)
            self.assertGreater(len(os.fsencode(root / ".phase-accounting.sock")), 108)
            completed, state = self.invoke_supervisor(root)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIsNotNone(state, completed.stdout + completed.stderr)
            self.assertEqual(state["status"], "complete", state)
            rows = [json.loads(line) for line in (root / "run-once" / "worker-phase-accounting.jsonl").read_text().splitlines()]
            self.assertEqual([row["stage"] for row in rows], [
                "staging_and_zero_step_gate", "label_qualification",
                "world_model_training_and_selection", "prediction_confirmation",
                "settlement_and_verified_export"])
            for row in rows[:-1]:
                self.assertEqual(row["supervisor_handshake"]["status"], "acknowledged")
            self.assertTrue(json.loads((root / "run-once" / "fixture-export.json").read_text())["verified"])
            self.assertTrue((root / "supervisor-console.log").is_file())
            export = Path(parent) / "verified-export"
            result = controlled_export(root / "run-once", export)
            self.assertTrue(result["verified"])
            self.assertEqual(json.loads((export / "fixture-export.json").read_text()), {"verified": True})

    def test_failure_without_worker_console_preserves_startup_exception_and_missing_log(self):
        with tempfile.TemporaryDirectory(prefix="w1-phase-missing-") as parent:
            root = self.make_fixture(parent, "raise RuntimeError('fixture should not run')")
            (root / "worker_bootstrap.py").unlink()
            completed, state = self.invoke_supervisor(root)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(state["status"], "stopped")
            events = [json.loads(line) for line in (root / "supervisor-console.log").read_text().splitlines()]
            self.assertEqual(events[0]["event"], "supervisor_starting")
            error = next(row for row in events if row["event"] == "supervisor_exception")
            self.assertIn("WORKER_BOOTSTRAP_MISSING", error["error"])
            self.assertFalse((root / "console.log").exists())

    def test_nonzero_worker_exit_is_durable_and_stops_without_retry(self):
        with tempfile.TemporaryDirectory(prefix="w1-phase-exit-") as parent:
            root = self.make_fixture(parent, "raise SystemExit(7)")
            completed, state = self.invoke_supervisor(root)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(state["status"], "stopped")
            self.assertEqual(state["returncode"], 7)
            self.assertEqual(state["stop_reason"], "Worker exited 7")
            events = [json.loads(line) for line in (root / "supervisor-console.log").read_text().splitlines()]
            self.assertTrue(any(row.get("event") == "worker_exit" and row.get("returncode") == 7
                                for row in events))

    def test_timeout_kills_worker_and_leaves_no_child_running(self):
        worker = """
            import time
            from pathlib import Path
            Path('entered').write_text('yes')
            time.sleep(30)
        """
        with tempfile.TemporaryDirectory(prefix="w1-phase-timeout-") as parent:
            root = self.make_fixture(parent, worker, wall_cap=1)
            completed, state = self.invoke_supervisor(root)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(state["status"], "stopped")
            self.assertTrue((root / "entered").is_file())
            self.assertIn("wall shutdown reserve", state["stop_reason"])
            self.assertIsNotNone(state["returncode"])
            with self.assertRaises(ProcessLookupError):
                os.kill(state["pid"], 0)

    def test_optional_worker_stdin_is_delivered_once_and_not_logged(self):
        worker = """
            import os
            from pathlib import Path
            value = os.read(0, 128)
            Path('received.bin').write_bytes(value)
        """
        secret = "fixture-token-7c91"
        with tempfile.TemporaryDirectory(prefix="w1-phase-stdin-") as parent:
            root = self.make_fixture(parent, worker)
            completed, state = self.invoke_supervisor(root, worker_input=secret)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(state["status"], "complete", state)
            self.assertEqual((root / "received.bin").read_text(), secret)
            self.assertNotIn(secret, (root / "supervisor-console.log").read_text())


if __name__ == "__main__":
    unittest.main()
