import copy
import math
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from metered_joint_entry import child_cpu, closure_accounting, failure_accounting, registration_handoff
import metered_joint_entry


REQUEST = {"totals": {"complete_process_cpu_seconds": 100.0, "wall_seconds": 100.0},
           "stages": {"settlement_and_verified_export": {"complete_process_cpu_seconds": 20.0, "wall_seconds": 20.0}}}
INNER = {"status": "complete", "native_cpu_total_seconds": 10.0,
         "settlement_stage_complete_process_cpu_seconds": 1.0,
         "settlement_stage_elapsed_wall_seconds": 1.0}


class OuterAccountingTests(unittest.TestCase):
    def calculate(self, **kwargs):
        values = {"own_cpu": 0.2, "waited_cpu": 10.4, "elapsed_wall": 12.0,
                  "wrapper_wall": 0.2, "request": REQUEST, "child_returncode": 0}
        values.update(kwargs)
        return closure_accounting(INNER, **values)

    def test_waited_lifetime_not_added_to_nested_snapshot(self):
        row = self.calculate()
        self.assertAlmostEqual(row["native_cpu_total_seconds"], 13.6)
        self.assertAlmostEqual(row["inner_snapshot_to_reap_tail_cpu_seconds"], 0.4)
        self.assertAlmostEqual(row["settlement_stage_complete_process_cpu_seconds"], 4.6)

    def test_final_tail_or_registration_overrun_cannot_be_succeeded(self):
        request = copy.deepcopy(REQUEST)
        request["stages"]["settlement_and_verified_export"]["complete_process_cpu_seconds"] = 4.0
        row = self.calculate(request=request)
        self.assertEqual(row["status"], "technical_stop")
        self.assertFalse(row["final_resource_pass"])

    def test_abnormal_exit_is_stop_even_when_resource_is_below_cap(self):
        self.assertEqual(self.calculate(child_returncode=-9)["status"], "technical_stop")

    def test_nonfinite_and_inverted_snapshot_rejected(self):
        for value in (math.nan, math.inf, -1.0):
            with self.assertRaisesRegex(RuntimeError, "VALUE_INVALID"):
                self.calculate(waited_cpu=value)
        with self.assertRaisesRegex(RuntimeError, "EXCEEDS_WAITED"):
            self.calculate(waited_cpu=9.0)

    def test_real_wait_captures_cpu_after_child_final_snapshot(self):
        code = """
import time
print(time.process_time(), flush=True)
deadline = time.process_time() + 0.08
value = 1
while time.process_time() < deadline:
    value = (value * 1103515245 + 12345) & 0x7fffffff
"""
        before = child_cpu()
        child = subprocess.run([sys.executable, "-B", "-c", code],
                               capture_output=True, text=True, timeout=5)
        self.assertEqual(child.returncode, 0, child.stderr)
        snapshot = float(child.stdout.strip())
        lifetime = child_cpu() - before
        inner = dict(INNER, native_cpu_total_seconds=snapshot)
        row = closure_accounting(inner, own_cpu=0.1, waited_cpu=lifetime,
            elapsed_wall=1.0, wrapper_wall=0.0, request=REQUEST, child_returncode=0)
        self.assertGreaterEqual(row["inner_snapshot_to_reap_tail_cpu_seconds"], 0.07)
        self.assertAlmostEqual(row["native_cpu_total_seconds"], lifetime + 0.1 + 3.0)

    def test_unresolved_exit_is_lower_bound_not_complete_settlement(self):
        row = failure_accounting(attempt="synthetic-stop", error="timeout",
            waited_cpu=0.2, own_cpu=0.1, elapsed_wall=1.0,
            child_returncode=None, timed_out=True,
            cleanup={"status": "unresolved", "error": "owned child still alive"})
        self.assertIsNone(row["native_cpu_total_seconds"])
        self.assertFalse(row["accounting_complete"])
        self.assertFalse(row["stage_attribution_complete"])
        self.assertFalse(row["final_resource_pass"])
        self.assertEqual(row["registration_terminal_status"], "unknown")
        self.assertEqual(row["reaped_child_cpu_delta_lower_bound_seconds"], 0.2)

    def test_handoff_binds_actual_inner_pid_and_packet_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "synthetic-root"
            checked = {"manifest_sha256": "a"*64, "hashes_sha256": "b"*64}
            path = Path(str(root) + "-registration-handoff.json")
            handoff = {"attempt": "synthetic", "registration_job_id": "synthetic-job",
                       "outer_pid": os.getpid(), "inner_pid": 123, **checked}
            path.write_text(json.dumps(handoff), encoding="utf-8")
            self.assertEqual(registration_handoff(root, checked, "synthetic", 123), "synthetic-job")
            with self.assertRaisesRegex(RuntimeError, "IDENTITY_MISMATCH"):
                registration_handoff(root, checked, "synthetic", 124)
            with self.assertRaisesRegex(RuntimeError, "IDENTITY_MISMATCH"):
                registration_handoff(root, checked, "other-attempt", 123)

    def test_actual_outer_timeout_reaps_child_and_records_failed_lower_bound(self):
        from manifest_contract import write_identity_files
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "synthetic-accounting-entry"
            root.mkdir()
            request = copy.deepcopy(REQUEST)
            request.update(attempt="synthetic-accounting-timeout", runner_ready=True)
            request["totals"]["wall_seconds"] = 6.0
            files = {"RESOURCE_REQUEST.json": request,
                "launch-contract.json": {"attempt": request["attempt"], "native_execution_root": str(root)},
                "cpu-accounting-contract.json": {"outer_closing_cpu_reserve_seconds": 3,
                                                 "outer_closing_wall_reserve_seconds": 5}}
            for name, value in files.items():
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            (root / "joint_remote_native.py").write_text("""
import hashlib, json, os, time
from pathlib import Path
root = Path(__file__).resolve().parent
digest = lambda name: hashlib.sha256((root / name).read_bytes()).hexdigest()
request = json.loads((root / 'RESOURCE_REQUEST.json').read_text())
handoff = {'attempt': request['attempt'], 'registration_job_id': 'synthetic-ledger-job',
           'inner_pid': os.getpid(), 'outer_pid': os.getppid(),
           'manifest_sha256': digest('execution-manifest.json'), 'hashes_sha256': digest('hashes.json')}
Path(str(root) + '-registration-handoff.json').write_text(json.dumps(handoff))
time.sleep(30)
""", encoding="utf-8")
            write_identity_files(root, attempt=request["attempt"])
            terminal = []
            with (patch.object(metered_joint_entry, "ROOT", root),
                  patch.object(sys, "argv", ["entry", "--authorization-file", str(root.parent / "synthetic.json"),
                                            "--real-name", "testperson"]),
                  patch.object(metered_joint_entry, "enforce_closing_limits"),
                  patch("server_registration.append_terminal", side_effect=lambda *args, **kwargs: terminal.append((args, kwargs)))):
                self.assertEqual(metered_joint_entry.main(), 1)
            stop = json.loads(Path(str(root) + "-outer-stop.json").read_text())
            self.assertTrue(stop["child_wait_timed_out"])
            self.assertLess(stop["child_returncode"], 0)
            self.assertEqual(stop["cleanup"]["status"], "reaped")
            self.assertEqual(stop["registration_terminal_status"], "FAILED")
            self.assertIsNone(stop["native_cpu_total_seconds"])
            self.assertFalse(stop["accounting_complete"])
            self.assertEqual(len(terminal), 1)
            self.assertEqual(terminal[0][0], ("synthetic-ledger-job", "FAILED"))


if __name__ == "__main__":
    unittest.main()
