import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import resource  # noqa: F401
except ModuleNotFoundError:
    resource_stub = ModuleType("resource")
    resource_stub.RUSAGE_SELF = 0
    resource_stub.RUSAGE_CHILDREN = 1
    resource_stub.getrusage = lambda _who: SimpleNamespace(ru_utime=0.0, ru_stime=0.0)
    sys.modules["resource"] = resource_stub

import joint_remote_native as remote


class _Child:
    pid = 12345
    returncode = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class JointRemoteNativeSettlementTests(unittest.TestCase):
    def test_failure_settlement_overwrites_complete_status_and_keeps_cpu(self):
        status = {"status": "complete", "automatic_retry": False}
        with patch.object(remote, "_usage_cpu", side_effect=[1.25, 0.90]), patch.object(
            remote.time, "monotonic", return_value=12.0
        ):
            settled = remote._record_failure_settlement(
                status,
                error="RuntimeError: synthetic export failure",
                started=10.0,
                child_cpu_start=0.40,
            )
        self.assertIs(settled, status)
        self.assertEqual(status["status"], "technical_stop")
        self.assertEqual(status["error"], "RuntimeError: synthetic export failure")
        self.assertAlmostEqual(status["settlement_process_self_cpu_seconds"], 1.25)
        self.assertAlmostEqual(status["settlement_reaped_children_cpu_seconds"], 0.50)
        self.assertAlmostEqual(status["settlement_process_cpu_seconds"], 1.75)
        self.assertAlmostEqual(status["settlement_wall_seconds"], 2.0)
        self.assertFalse(status["final_resource_pass"])

    def test_controlled_export_failure_persists_technical_stop(self):
        with tempfile.TemporaryDirectory(prefix="w1-joint-native-settlement-") as directory:
            root = Path(directory)
            (root / "remote-runtime-identity.json").write_text("{}\n", encoding="utf-8")
            contract = {
                "attempt": "attempt-fixture",
                "native_execution_root": root.as_posix(),
                "runtime_identity_sha256": "runtime-digest",
                "resource_request_sha256": "request-digest",
            }
            request = {
                "runner_ready": True,
                "totals": {
                    "wall_seconds": 60.0,
                    "complete_process_cpu_seconds": 60.0,
                    "aggregate_native_plus_verified_export_bytes": 100000,
                },
                "stages": {
                    "staging_and_zero_step_gate": {
                        "wall_seconds": 30.0,
                        "complete_process_cpu_seconds": 30.0,
                    },
                    "settlement_and_verified_export": {
                        "wall_seconds": 30.0,
                        "complete_process_cpu_seconds": 30.0,
                    },
                },
            }
            checked = {
                "contract": contract,
                "request": request,
                "identity": {"manifest_sha256": "manifest", "hashes_sha256": "hashes"},
            }
            supervisor_status = {
                "pid": 22222,
                "supervisor_pid": 99999,
                "returncode": 0,
                "stop_reason": None,
            }
            launcher_snapshot = {
                "native_launcher_complete_process_cpu_snapshot_seconds": 0.0,
                "native_launcher_process_cpu_seconds": 0.0,
                "native_launcher_reaped_children_cpu_seconds": 0.0,
            }

            def bind_worker(path, supervisor_pid, child, timeout_seconds):
                _write_json(path / "supervisor-status.json", supervisor_status)
                _write_json(path / "runtime-output" / "native-launcher-accounting.json", launcher_snapshot)
                return supervisor_status

            fake_manifest = ModuleType("manifest_contract")
            fake_manifest.verify_external_authorization = lambda *args, **kwargs: checked
            fake_manifest.sha256_file = lambda path: "runtime-digest"
            fake_inputs = ModuleType("joint_inputs")
            fake_inputs.verify_joint_inputs = lambda path: {"status": "pass"}
            fake_io = ModuleType("infra_io")
            fake_io.require_native_linux_filesystem = lambda path: None
            fake_io.durable_atomic_json = _write_json

            def fail_export(source, destination):
                raise RuntimeError("synthetic export failure")

            fake_io.controlled_export = fail_export
            fake_registration = ModuleType("server_registration")
            fake_registration.register_before_launch = lambda **kwargs: {"job_id": "job-fixture"}
            fake_registration.append_running = lambda *args, **kwargs: None
            fake_registration.append_terminal = lambda *args, **kwargs: None
            argv = [
                "joint_remote_native.py", "--authorization-file", str(root / "authorization.json"),
                "--real-name", "lihua", "--name-id", "lihua",
            ]
            with (
                patch.object(remote, "__file__", str(root / "joint_remote_native.py")),
                patch.object(remote.runpy, "run_path", return_value={
                    "verify_process_isolation": lambda: None,
                    "verify": lambda value: {"status": "pass"},
                }),
                patch.object(remote, "derive_name_id", return_value="lihua"),
                patch.object(remote, "registration_details", return_value={}),
                patch.object(remote, "wait_for_worker_binding", bind_worker),
                patch.object(remote, "wait_for_gpu_binding", return_value=[]),
                patch.object(remote.subprocess, "Popen", return_value=_Child()),
                patch.object(remote.os, "getpgrp", return_value=99999, create=True),
                patch.object(remote.sys, "argv", argv),
                patch.object(remote.sys, "stdin", io.StringIO("token\n")),
                patch.dict(sys.modules, {
                    "manifest_contract": fake_manifest,
                    "joint_inputs": fake_inputs,
                    "infra_io": fake_io,
                    "server_registration": fake_registration,
                }),
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                result = remote.main()

            self.assertEqual(result, 1)
            persisted = json.loads((root / "runtime-output" / "execution-final-status.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"], "technical_stop")
            self.assertIn("synthetic export failure", persisted["error"])
            self.assertIn("settlement_process_cpu_seconds", persisted)
            printed = json.loads(stdout.getvalue())
            self.assertEqual(printed["status"], "technical_stop")

    def test_final_settlement_overrun_is_checked_before_success_registration(self):
        with tempfile.TemporaryDirectory(prefix="w1-joint-native-final-overrun-") as directory:
            root = Path(directory)
            (root / "remote-runtime-identity.json").write_text("{}\n", encoding="utf-8")
            contract = {
                "attempt": "attempt-fixture",
                "native_execution_root": root.as_posix(),
                "runtime_identity_sha256": "runtime-digest",
                "resource_request_sha256": "request-digest",
            }
            request = {
                "runner_ready": True,
                "totals": {
                    "wall_seconds": 60.0,
                    "complete_process_cpu_seconds": 0.5,
                    "aggregate_native_plus_verified_export_bytes": 100000,
                },
                "stages": {
                    "staging_and_zero_step_gate": {
                        "wall_seconds": 30.0,
                        "complete_process_cpu_seconds": 30.0,
                    },
                    "settlement_and_verified_export": {
                        "wall_seconds": 30.0,
                        "complete_process_cpu_seconds": 30.0,
                    },
                },
            }
            checked = {
                "contract": contract,
                "request": request,
                "identity": {"manifest_sha256": "manifest", "hashes_sha256": "hashes"},
            }
            supervisor_status = {
                "pid": 22222,
                "supervisor_pid": 99999,
                "returncode": 0,
                "stop_reason": None,
            }
            launcher_snapshot = {
                "native_launcher_complete_process_cpu_snapshot_seconds": 0.0,
                "native_launcher_process_cpu_seconds": 0.0,
                "native_launcher_reaped_children_cpu_seconds": 0.0,
            }
            terminal_events = []

            def bind_worker(path, supervisor_pid, child, timeout_seconds):
                _write_json(path / "supervisor-status.json", supervisor_status)
                _write_json(path / "runtime-output" / "native-launcher-accounting.json", launcher_snapshot)
                return supervisor_status

            fake_manifest = ModuleType("manifest_contract")
            fake_manifest.verify_external_authorization = lambda *args, **kwargs: checked
            fake_manifest.sha256_file = lambda path: "runtime-digest"
            fake_inputs = ModuleType("joint_inputs")
            fake_inputs.verify_joint_inputs = lambda path: {"status": "pass"}
            fake_io = ModuleType("infra_io")
            fake_io.require_native_linux_filesystem = lambda path: None
            fake_io.durable_atomic_json = _write_json
            fake_io.controlled_export = lambda source, destination: {"verified": True}
            fake_registration = ModuleType("server_registration")
            fake_registration.register_before_launch = lambda **kwargs: {"job_id": "job-fixture"}
            fake_registration.append_running = lambda *args, **kwargs: None

            def append_terminal(job_id, event, **kwargs):
                terminal_events.append(event)

            fake_registration.append_terminal = append_terminal
            argv = [
                "joint_remote_native.py", "--authorization-file", str(root / "authorization.json"),
                "--real-name", "lihua", "--name-id", "lihua",
            ]
            with (
                patch.object(remote, "__file__", str(root / "joint_remote_native.py")),
                patch.object(remote.runpy, "run_path", return_value={
                    "verify_process_isolation": lambda: None,
                    "verify": lambda value: {"status": "pass"},
                }),
                patch.object(remote, "derive_name_id", return_value="lihua"),
                patch.object(remote, "registration_details", return_value={}),
                patch.object(remote, "wait_for_worker_binding", bind_worker),
                patch.object(remote, "wait_for_gpu_binding", return_value=[]),
                patch.object(remote.subprocess, "Popen", return_value=_Child()),
                patch.object(remote.os, "getpgrp", return_value=99999, create=True),
                patch.object(remote, "_usage_cpu", side_effect=[0.0, 0.0, 0.0, 0.0, 1.0]),
                patch.object(remote.sys, "argv", argv),
                patch.object(remote.sys, "stdin", io.StringIO("token\n")),
                patch.dict(sys.modules, {
                    "manifest_contract": fake_manifest,
                    "joint_inputs": fake_inputs,
                    "infra_io": fake_io,
                    "server_registration": fake_registration,
                }),
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                result = remote.main()

            self.assertEqual(result, 1)
            self.assertEqual(terminal_events, ["FAILED"])
            persisted = json.loads(Path(str(root) + "-final-settlement.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"], "technical_stop")
            self.assertEqual(persisted["error"], "FINAL_RESOURCE_LIMIT_EXCEEDED")
            self.assertEqual(json.loads(stdout.getvalue())["status"], "technical_stop")


if __name__ == "__main__":
    unittest.main()
