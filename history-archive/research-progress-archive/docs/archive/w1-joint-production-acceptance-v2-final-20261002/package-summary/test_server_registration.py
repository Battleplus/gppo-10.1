"""Synthetic tests for the native server registration contract.

All ledger writes use temporary files and all process records are fabricated
under a temporary proc_root. These tests do not start processes or query GPUs.
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path
import tempfile
import unittest

from registration_identity import derive_name_id
import server_registration as registration


def _details() -> dict:
    return {
        "ai": "codex-synthetic-test",
        "task_name": "synthetic contract fixture",
        "purpose": "verify append-only registration behavior",
        "workdir": "/synthetic/workdir",
        "training_files": ["train.py", "config.yaml"],
        "dataset": "synthetic-data",
        "model": "synthetic-model",
        "command_redacted": "python train.py --config config.yaml",
        "gpu_request": {"ids": [1], "count": 1, "vram_estimate_gb": "unknown"},
        "cpu_cores_estimate": "unknown",
        "ram_estimate_gb": "unknown",
        "disk_growth_estimate_gb": "unknown",
        "duration_estimate": "unknown",
        "estimate_basis": "synthetic test fixture; no resource estimate performed",
        "log_path": "/synthetic/logs/run.log",
        "checkpoint_path": "/synthetic/checkpoints",
        "resource_snapshot": "synthetic fixture",
    }


def _register(path: Path) -> dict:
    return registration._register_before_launch(
        path,
        real_name="合成测试用户",
        name_id=derive_name_id("合成测试用户"),
        details=_details(),
        attempt="synthetic-attempt-001",
        manifest_sha256="a" * 64,
        hashes_sha256="b" * 64,
        resource_request_sha256="c" * 64,
        budget={"totals": {"wall_seconds": 20, "gpu_devices": 1}},
    )


def _proc_stat(pid: int, ppid: int, pgid: int, starttime: int) -> str:
    tail = ["S", str(ppid), str(pgid), *(["0"] * 16), str(starttime)]
    return f"{pid} (synthetic python worker) {' '.join(tail)}\n"


def _write_proc(proc_root: Path, pid: int, ppid: int, pgid: int,
                uid: int, starttime: int) -> None:
    process_dir = proc_root / str(pid)
    process_dir.mkdir(parents=True)
    (process_dir / "stat").write_text(
        _proc_stat(pid, ppid, pgid, starttime), encoding="ascii",
    )
    (process_dir / "status").write_text(
        f"Name:\tsynthetic\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n",
        encoding="ascii",
    )


@unittest.skipUnless(os.name == "posix" and registration.sys.platform.startswith("linux"),
                     "native Linux file-lock and /proc contract")
class ServerRegistrationContractTests(unittest.TestCase):
    def test_registered_event_is_read_back_and_history_is_append_only(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-server-registration-") as directory:
            path = Path(directory) / "dengji.txt"
            registered = _register(path)
            self.assertEqual(registered["status"], "REGISTERED")
            self.assertEqual(registered["binding"]["attempt"], "synthetic-attempt-001")
            self.assertEqual(registered["binding"]["manifest_sha256"], "a" * 64)
            self.assertEqual(registered["binding"]["hashes_sha256"], "b" * 64)
            self.assertEqual(registered["binding"]["budget"]["totals"]["wall_seconds"], 20)
            before = path.read_bytes()
            same = registration._verify_registered(
                path, registered["job_id"], "合成测试用户", derive_name_id("合成测试用户"),
            )
            self.assertEqual(same, registered)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)

    def test_missing_or_shared_account_identity_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-server-registration-") as directory:
            path = Path(directory) / "dengji.txt"
            with self.assertRaises(registration.RegistrationError):
                registration._register_before_launch(
                    path, real_name="", name_id=derive_name_id("合成测试用户"), details=_details(),
                    attempt="synthetic-attempt-001", manifest_sha256="a" * 64,
                    hashes_sha256="b" * 64, budget={"wall_seconds": 1},
                )
            with self.assertRaises(registration.RegistrationError):
                registration._register_before_launch(
                    path, real_name="admin123", name_id=derive_name_id("合成测试用户"), details=_details(),
                    attempt="synthetic-attempt-001", manifest_sha256="a" * 64,
                    hashes_sha256="b" * 64, budget={"wall_seconds": 1},
                )
            with self.assertRaisesRegex(registration.RegistrationError, "does not match"):
                registration._register_before_launch(
                    path, real_name="合成测试用户", name_id="wrongidentity", details=_details(),
                    attempt="synthetic-attempt-001", manifest_sha256="a" * 64,
                    hashes_sha256="b" * 64, budget={"wall_seconds": 1},
                )
            self.assertFalse(path.exists())

    def test_running_binding_uses_synthetic_proc_and_injected_gpu_query(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-server-registration-") as directory:
            root = Path(directory)
            ledger = root / "dengji.txt"
            proc_root = root / "proc"
            proc_root.mkdir()
            registered = _register(ledger)
            uid = registered["linux_uid"]
            _write_proc(proc_root, 4100, 1, 4100, uid, 100)
            _write_proc(proc_root, 4101, 4100, 4101, uid, 101)
            query_count = 0

            def readonly_gpu_query(process_pids):
                nonlocal query_count
                query_count += 1
                self.assertEqual(process_pids, (4100, 4101))
                return [1]

            running = registration._append_running(
                ledger, registered["job_id"], pid=4100, pgid=4100,
                worker_pids=[4101], actual_gpu_ids=[1],
                readonly_gpu_query=readonly_gpu_query, proc_root=proc_root,
            )
            self.assertEqual(running["event"], "RUNNING")
            self.assertEqual(running["worker_pids"], [4101])
            self.assertEqual(running["actual_gpu_ids"], [1])
            self.assertEqual(running["worker_pgids"], {"4101": 4101})
            self.assertEqual(running["process_starttimes"], {"4100": 100, "4101": 101})
            self.assertGreaterEqual(query_count, 2)
            self.assertEqual(len(ledger.read_text(encoding="utf-8").splitlines()), 2)

    def test_running_rejects_owner_process_group_and_gpu_mismatches(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-server-registration-") as directory:
            root = Path(directory)
            proc_root = root / "proc"
            proc_root.mkdir()
            registered = _register(root / "dengji.txt")
            uid = registered["linux_uid"]
            _write_proc(proc_root, 4200, 1, 4200, uid, 100)
            _write_proc(proc_root, 4201, 4200, 4200, uid + 1, 101)
            process_event = {
                **registered,
                "pid": 4200,
                "pgid": 4200,
                "worker_pids": [4201],
            }
            with self.assertRaises(registration.RegistrationError):
                registration._verify_process_binding(process_event, proc_root=proc_root)
            with self.assertRaises(registration.RegistrationError):
                registration._query_and_verify_gpu_ids([1], lambda _pids: [0], (4200, 4201))

    def test_running_rejects_empty_gpu_binding_for_gpu_worker(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-server-registration-") as directory:
            root = Path(directory)
            ledger = root / "dengji.txt"
            proc_root = root / "proc"
            proc_root.mkdir()
            registered = _register(ledger)
            uid = registered["linux_uid"]
            _write_proc(proc_root, 4250, 1, 4250, uid, 100)
            _write_proc(proc_root, 4251, 4250, 4251, uid, 101)
            with self.assertRaisesRegex(registration.RegistrationError, "nonempty actual GPU binding"):
                registration._append_running(
                    ledger, registered["job_id"], pid=4250, pgid=4250,
                    worker_pids=[4251], actual_gpu_ids=[],
                    readonly_gpu_query=lambda _pids: [], proc_root=proc_root,
                )

    def test_terminal_event_appends_and_public_functions_have_no_path_override(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-server-registration-") as directory:
            root = Path(directory)
            ledger = root / "dengji.txt"
            proc_root = root / "proc"
            proc_root.mkdir()
            registered = _register(ledger)
            uid = registered["linux_uid"]
            _write_proc(proc_root, 4300, 1, 4300, uid, 100)
            registration._append_running(
                ledger, registered["job_id"], pid=4300, pgid=4300, worker_pids=[],
                actual_gpu_ids=[], readonly_gpu_query=lambda _pids: [], proc_root=proc_root,
            )
            terminal = registration._append_terminal(
                ledger, registered["job_id"], "SUCCEEDED", exit_code=0,
                result_path="/synthetic/result",
            )
            self.assertEqual(terminal["status"], "SUCCEEDED")
            self.assertEqual(len(ledger.read_text(encoding="utf-8").splitlines()), 3)
        self.assertNotIn("path", inspect.signature(registration.register_before_launch).parameters)
        self.assertNotIn("path", inspect.signature(registration.append_running).parameters)
        self.assertNotIn("path", inspect.signature(registration.append_terminal).parameters)

    def test_failure_can_be_recorded_before_a_worker_starts(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-server-registration-") as directory:
            ledger = Path(directory) / "dengji.txt"
            registered = _register(ledger)
            failed = registration._append_terminal(
                ledger, registered["job_id"], "FAILED", exit_code=1,
                reason_redacted="native preflight failed before worker start",
            )
            self.assertEqual(failed["event"], "FAILED")
            self.assertEqual(len(ledger.read_text(encoding="utf-8").splitlines()), 2)

    def test_credential_like_payload_is_rejected_before_append(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-server-registration-") as directory:
            path = Path(directory) / "dengji.txt"
            details = _details()
            details["command_redacted"] = "python train.py --token raw-secret-value"
            with self.assertRaises(registration.RegistrationError):
                registration._register_before_launch(
                path, real_name="合成测试用户", name_id=derive_name_id("合成测试用户"),
                details=details, attempt="synthetic-attempt-001",
                    manifest_sha256="a" * 64, hashes_sha256="b" * 64,
                    budget={"wall_seconds": 1},
                )
            self.assertFalse(path.exists())


class GpuBindingWaitTests(unittest.TestCase):
    def test_gpu_binding_waits_for_exact_requested_set(self):
        observations = iter([[], [1]])
        queried = []

        def query(process_pids):
            queried.append(process_pids)
            return next(observations)

        actual = registration.wait_for_gpu_binding(
            (5000, 5001), [1], timeout_seconds=1.0,
            readonly_gpu_query=query, poll_seconds=0.0,
        )
        self.assertEqual(actual, [1])
        self.assertEqual(queried, [(5000, 5001), (5000, 5001)])

        with self.assertRaisesRegex(registration.RegistrationError, "GPU_REGISTRATION_BINDING_TIMEOUT"):
            registration.wait_for_gpu_binding(
                (5000, 5001), [1], timeout_seconds=0.0,
                readonly_gpu_query=lambda _pids: [], poll_seconds=0.0,
            )


class RegistrationIdentityBindingTests(unittest.TestCase):
    def test_registration_requires_canonical_name_id(self):
        with self.assertRaisesRegex(registration.RegistrationError, "does not match"):
            registration._validate_real_name(
                "合成测试用户", "wrongidentity", "isolated-account",
            )
        registration._validate_real_name(
            "合成测试用户", derive_name_id("合成测试用户"), "isolated-account",
        )


if __name__ == "__main__":
    unittest.main()
