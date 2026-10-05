from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import infra_io
import launch_once
import supervise


NATIVE_TEST_PARENT = Path("/home/asus/w1-oracle-runner-tests")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def request() -> dict:
    counters = {
        "environment_steps": 0,
        "resets": 0,
        "candidate_scans": 0,
        "public_rule_decisions": 0,
        "snapshot_captures": 0,
        "branch_snapshot_copies": 0,
        "branches": 0,
        "forced_first_actions": 0,
        "model_initializations_or_loads": 0,
        "encode_forwards": 0,
        "actor_forwards": 0,
        "world_forwards": 0,
        "policy_updates": 0,
        "world_updates": 0,
        "offline_predictor_updates": 0,
    }
    cap = {
        **counters,
        "wall_seconds": 20,
        "complete_process_cpu_seconds": 20,
        "artifact_bytes": 16 * 1024 * 1024,
    }
    return {
        "status": "NOT_APPROVED",
        "attempt": "fake-once",
        "stages": {
            "staging_and_zero_step_gate": cap,
            "single_unit_dynamic_gate": cap,
            "settlement": cap,
            "verified_export": cap,
        },
        "totals": {
            **cap,
            "all_resident_rss_bytes": 1024 * 1024 * 1024,
            "aggregate_native_plus_verified_export_bytes": 32 * 1024 * 1024,
        },
        "shutdown_reserve": {
            "wall_seconds_within_each_dynamic_stage": 1,
            "complete_process_cpu_seconds_within_each_dynamic_stage": 1,
            "artifact_bytes_within_active_limit": 1024 * 1024,
        },
    }


class NativeCase(unittest.TestCase):
    def setUp(self):
        NATIVE_TEST_PARENT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=NATIVE_TEST_PARENT)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()


class FileTests(NativeCase):
    def test_native_atomic_updates_allow_concurrent_readers(self):
        infra_io.require_native_linux_filesystem(self.root)
        target = self.root / "status.json"
        errors = []
        finished = threading.Event()

        def reader():
            while not finished.is_set():
                try:
                    json.loads(target.read_text(encoding="utf-8"))
                except FileNotFoundError:
                    pass
                except BaseException as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for thread in threads:
            thread.start()
        for value in range(40):
            infra_io.durable_atomic_json(target, {"value": value})
        finished.set()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(infra_io.read_json(target), {"value": 39})

    def test_abnormal_reader_exit_does_not_block_append(self):
        path = self.root / "events.jsonl"
        infra_io.durable_append_jsonl(path, {"sequence": 1})
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import os,sys; f=open(sys.argv[1],'rb'); f.read(); os._exit(7)",
                str(path),
            ],
            check=False,
        )
        infra_io.durable_append_jsonl(path, {"sequence": 2})
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([row["sequence"] for row in rows], [1, 2])

    def test_status_classification_distinguishes_stale_and_exited(self):
        now = 100.0
        running = {"status": "running", "supervisor_pid": 4, "updated_monotonic": 99.5}
        self.assertEqual(
            infra_io.classify_status(
                running, now_monotonic=now, alive=lambda _pid: True
            ),
            "running",
        )
        self.assertEqual(
            infra_io.classify_status(
                {**running, "updated_monotonic": 1.0},
                now_monotonic=now,
                alive=lambda _pid: True,
            ),
            "stale_while_process_alive",
        )
        self.assertEqual(
            infra_io.classify_status(
                running, now_monotonic=now, alive=lambda _pid: False
            ),
            "exited_without_final_status",
        )


class LaunchTests(NativeCase):
    def mini_source(self):
        source = self.root / "source"
        native = self.root / "native"
        export = self.root / "export"
        source.mkdir()
        token = "fake-explicit-token"
        write_json(source / "RESOURCE_REQUEST.json", request())
        write_json(
            source / "launch-contract.json",
            {
                "attempt": "fake-once",
                "external_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                "native_execution_root": str(native),
                "windows_export_root": str(export),
                "native_python": sys.executable,
            },
        )
        (source / "payload.txt").write_text("payload", encoding="utf-8")
        files = {
            name: infra_io.sha256_file(source / name)
            for name in (
                "RESOURCE_REQUEST.json",
                "launch-contract.json",
                "payload.txt",
            )
        }
        manifest = {"attempt": "fake-once", "files": files}
        write_json(source / "execution-manifest.json", manifest)
        write_json(
            source / "hashes.json",
            {
                "execution_manifest_sha256": infra_io.sha256_file(
                    source / "execution-manifest.json"
                )
            },
        )
        return source, native, export, token

    def test_invalid_token_creates_no_attempt(self):
        source, native, export, _token = self.mini_source()
        with self.assertRaisesRegex(RuntimeError, "token mismatch"):
            launch_once.orchestrate("wrong", root=source)
        self.assertFalse(native.exists())
        self.assertFalse(export.exists())

    def test_fake_worker_runs_once_then_verified_export(self):
        source, native, export, token = self.mini_source()
        starts = []

        def run_native(_python, staged, _token):
            starts.append(staged)
            (staged / "worker-finished.txt").write_text("done", encoding="utf-8")
            write_json(
                staged / "supervisor-status.json",
                {"status": "complete", "wall_seconds": 0.1, "cpu_seconds": 0.1},
            )
            return 0, None

        result = launch_once.orchestrate(
            token, root=source, run_native=run_native
        )
        self.assertEqual(result, 0)
        self.assertEqual(len(starts), 1)
        self.assertTrue((export / "EXPORT_COMPLETE.json").is_file())
        completion = infra_io.read_json(export / "EXPORT_COMPLETE.json")
        self.assertTrue(completion["verified"])

    def test_export_failure_does_not_relaunch_worker(self):
        source, native, export, token = self.mini_source()
        starts = []

        def run_native(_python, staged, _token):
            starts.append(staged)
            return 0, None

        def fail_export(_source, destination):
            destination.mkdir()
            raise OSError(5, "injected export failure")

        result = launch_once.orchestrate(
            token,
            root=source,
            run_native=run_native,
            exporter=fail_export,
        )
        self.assertEqual(result, 1)
        self.assertEqual(len(starts), 1)
        status = infra_io.read_json(native / "export-status.json")
        self.assertEqual(status["status"], "failed")
        self.assertFalse(status["worker_relaunched"])


class SupervisorTests(NativeCase):
    def run_supervisor(self, body: str):
        write_json(self.root / "RESOURCE_REQUEST.json", request())
        (self.root / "fake_runner.py").write_text(body, encoding="utf-8")
        available = sorted(os.sched_getaffinity(0))
        if len(available) < 4:
            self.skipTest("four CPU cores required")
        calls = 0

        def idle():
            nonlocal calls
            calls += 1
            base = 100 * calls
            return {cpu: (base, base) for cpu in available}

        with (
            mock.patch.object(
                supervise,
                "machine_guard",
                return_value={
                    "available_ram_bytes": 2**40,
                    "disk_free_bytes": 2**40,
                },
            ),
            mock.patch.object(supervise, "cpu_idle", side_effect=idle),
        ):
            return supervise.main(
                self.root, "fake_runner.py", sample_interval=0.02
            )

    def test_worker_nonzero_exit_is_a_final_stop(self):
        result = self.run_supervisor("raise SystemExit(9)\n")
        self.assertEqual(result, 1)
        status = infra_io.read_json(self.root / "supervisor-status.json")
        self.assertEqual(status["status"], "stopped")
        self.assertIn("Worker exited 9", status["stop_reason"])
        self.assertFalse(status["automatic_retry"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
