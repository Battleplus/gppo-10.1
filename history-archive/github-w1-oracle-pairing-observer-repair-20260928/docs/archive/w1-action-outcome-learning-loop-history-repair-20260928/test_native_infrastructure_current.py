from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import infra_io
import supervise


NATIVE_TEST_PARENT = Path("/home/asus/w1-infra-tests")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def request() -> dict:
    cap = {
        "wall_seconds": 20,
        "complete_process_cpu_seconds": 20,
        "artifact_bytes": 16 * 1024 * 1024,
    }
    return {
        "status": "NOT_APPROVED",
        "attempt": "fake-once",
        "stages": {"staging_and_zero_step_gate": cap},
        "totals": {**cap, "all_resident_rss_bytes": 1024 * 1024 * 1024},
        "shutdown_reserve": {
            "wall_seconds_within_each_dynamic_stage": 1,
            "complete_process_cpu_seconds_within_each_dynamic_stage": 1,
        },
    }


class NativeCase(unittest.TestCase):
    def setUp(self):
        NATIVE_TEST_PARENT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=NATIVE_TEST_PARENT)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def make_supervisor_root(self, body: str) -> Path:
        write_json(self.root / "RESOURCE_REQUEST.json", request())
        (self.root / "fake_runner.py").write_text(body, encoding="utf-8")
        return self.root

    def supervisor(self, root: Path, **kwargs) -> int:
        available = sorted(os.sched_getaffinity(0))
        if len(available) < 4:
            self.skipTest("Four CPU cores required by frozen supervisor")
        calls = 0

        def idle():
            nonlocal calls
            calls += 1
            base = 100 * calls
            return {cpu: (base, base) for cpu in available}

        with mock.patch.object(supervise, "machine_guard", return_value={"available_ram_bytes": 2**40, "disk_free_bytes": 2**40}), mock.patch.object(supervise, "cpu_idle", side_effect=idle):
            return supervise.main(root, "fake_runner.py", sample_interval=.02, **kwargs)


class FileStateTests(NativeCase):
    def test_infrastructure_modules_do_not_import_torch_or_project_runtime(self):
        import ast

        package = Path(infra_io.__file__).resolve().parent
        prohibited = {"torch", "gppo_world", "runner", "rl_adapters"}
        for name in ("infra_io.py", "launch_once.py", "native_launch.py", "supervise.py"):
            tree = ast.parse((package / name).read_text(encoding="utf-8"), name)
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            self.assertTrue(prohibited.isdisjoint(imported), (name, imported & prohibited))
        self.assertNotIn("torch", sys.modules)

    def test_native_mount_and_atomic_concurrent_readers(self):
        identity = infra_io.require_native_linux_filesystem(self.root)
        self.assertNotIn(identity["filesystem_type"].lower(), {"9p", "drvfs"})
        target = self.root / "status.json"
        errors = []
        done = threading.Event()

        def reader():
            while not done.is_set():
                try:
                    json.loads(target.read_text(encoding="utf-8"))
                except FileNotFoundError:
                    pass
                except BaseException as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(6)]
        for thread in threads:
            thread.start()
        for value in range(100):
            infra_io.durable_atomic_json(target, {"value": value})
        done.set()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(infra_io.read_json(target), {"value": 99})

    def test_append_stream_reader_abnormal_exit_releases_handle(self):
        progress = self.root / "progress.jsonl"
        infra_io.durable_append_jsonl(progress, {"seq": 1})
        subprocess.run(
            [sys.executable, "-c", "import os,sys; f=open(sys.argv[1],'rb'); f.read(); os._exit(7)", str(progress)],
            check=False,
        )
        infra_io.durable_append_jsonl(progress, {"seq": 2})
        rows = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["seq"] for row in rows], [1, 2])

    def test_fsync_failure_is_bounded_and_visible(self):
        target = self.root / "failed.json"
        policy = infra_io.FileRetryPolicy(attempts=2, max_elapsed_seconds=.2, delay_seconds=.01)
        with mock.patch.object(infra_io.os, "fsync", side_effect=OSError(28, "No space left")) as fsync:
            with self.assertRaises(OSError):
                infra_io.durable_atomic_json(target, {"x": 1}, policy)
        self.assertEqual(fsync.call_count, 2)
        self.assertFalse(target.exists())

    def test_status_classification(self):
        now = 100.0
        self.assertEqual(infra_io.classify_status({"status": "running", "supervisor_pid": 4, "updated_monotonic": 99.5}, now_monotonic=now, alive=lambda _: True), "running")
        self.assertEqual(infra_io.classify_status({"status": "running", "supervisor_pid": 4, "updated_monotonic": 1}, now_monotonic=now, alive=lambda _: True), "stale_while_process_alive")
        self.assertEqual(infra_io.classify_status({"status": "running", "supervisor_pid": 4, "updated_monotonic": 99.5}, now_monotonic=now, alive=lambda _: False), "exited_without_final_status")
        self.assertEqual(infra_io.classify_status({"status": "complete"}), "complete")


class SupervisorTests(NativeCase):
    NORMAL = """import json,time\nfrom pathlib import Path\nr=Path(__file__).parent\n(r/'run-once').mkdir()\n(r/'run-once'/'activity.json').write_text(json.dumps({'stage':'staging_and_zero_step_gate'})+'\\n')\ntime.sleep(.12)\n(r/'run-once'/'status.json').write_text(json.dumps({'ledger':{'reserved':{},'call_status_counts':{'verified':1,'unknown':1,'pending':1}}})+'\\n')\n"""

    def test_worker_normal_exit_and_unresolved_ledger_preserved(self):
        root = self.make_supervisor_root(self.NORMAL)
        self.assertEqual(self.supervisor(root), 0)
        status = infra_io.read_json(root / "supervisor-status.json")
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["cumulative_cost"]["current_call_status_counts"]["unknown"], 1)
        self.assertEqual(status["cumulative_cost"]["current_call_status_counts"]["pending"], 1)

    def test_worker_nonzero_exit_stops(self):
        root = self.make_supervisor_root("import sys; sys.exit(9)\n")
        self.assertEqual(self.supervisor(root), 1)
        status = infra_io.read_json(root / "supervisor-status.json")
        self.assertEqual(status["status"], "stopped")
        self.assertIn("Worker exited 9", status["stop_reason"])

    def test_injected_status_write_failure_stops_worker_and_finalizes(self):
        root = self.make_supervisor_root("import time; time.sleep(5)\n")
        calls = 0

        def fail_once(path, value):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise PermissionError(13, "injected status replace failure")
            infra_io.durable_atomic_json(path, value)

        self.assertEqual(self.supervisor(root, status_writer=fail_once), 1)
        status = infra_io.read_json(root / "supervisor-status.json")
        self.assertEqual(status["status"], "stopped")
        self.assertIn("PermissionError", status["stop_reason"])
        self.assertFalse(infra_io.process_alive(status["pid"]))

    def test_supervisor_signal_interrupt_stops_worker(self):
        root = self.make_supervisor_root("import time; time.sleep(20)\n")
        driver = self.root / "driver.py"
        package = Path(supervise.__file__).resolve().parent
        driver.write_text(
            "import os,supervise,sys\n"
            "supervise.machine_guard=lambda *a,**k:{'available_ram_bytes':2**40,'disk_free_bytes':2**40}\n"
            "c=[0]\n"
            "def idle():\n c[0]+=1; b=100*c[0]; return {x:(b,b) for x in os.sched_getaffinity(0)}\n"
            "supervise.cpu_idle=idle\n"
            "raise SystemExit(supervise.main(sys.argv[1],'fake_runner.py',sample_interval=.02))\n",
            encoding="utf-8",
        )
        environment = {**os.environ, "PYTHONPATH": str(package)}
        process = subprocess.Popen([sys.executable, str(driver), str(root)], env=environment)
        deadline = time.time() + 5
        while not (root / "supervisor-status.json").exists() and time.time() < deadline:
            time.sleep(.02)
        os.kill(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
        self.assertNotEqual(process.returncode, 0)
        status = infra_io.read_json(root / "supervisor-status.json")
        self.assertEqual(status["status"], "stopped")
        self.assertIn("Supervisor received signal", status["stop_reason"])


class StagingExportTests(NativeCase):
    def test_controlled_export_hashes_payload(self):
        source = self.root / "source"
        destination = self.root / "archive"
        (source / "run-once").mkdir(parents=True)
        (source / "run-once" / "status.json").write_text('{"ok":true}\n', encoding="utf-8")
        result = infra_io.controlled_export(source, destination)
        self.assertTrue(result["verified"])
        manifest = infra_io.read_json(destination / "export-manifest.json")
        infra_io.verify_manifest(destination, {key: value["sha256"] for key, value in manifest["files"].items()})
        self.assertTrue((destination / "EXPORT_COMPLETE.json").is_file())

    def mini_source(self) -> tuple[Path, Path]:
        source = self.root / "sealed"
        native = self.root / "native-attempt"
        source.mkdir()
        (source / "payload.txt").write_text("payload", encoding="utf-8")
        files = {"payload.txt": infra_io.sha256_file(source / "payload.txt")}
        manifest = {"attempt": "fake-once", "files": files}
        write_json(source / "execution-manifest.json", manifest)
        write_json(source / "hashes.json", {"execution_manifest_sha256": infra_io.sha256_file(source / "execution-manifest.json")})
        return source, native

    def test_initialized_native_staging_requires_status_and_verifies(self):
        source, native = self.mini_source()
        native.mkdir()
        write_json(native / "launcher-status.json", {"status": "running"})
        manifest = infra_io.read_json(source / "execution-manifest.json")
        result = infra_io.stage_sealed_package_into_initialized_root(source, native, manifest)
        self.assertEqual(result["pre_copy_status_files"], ["launcher-status.json"])
        self.assertEqual((native / "payload.txt").read_text(encoding="utf-8"), "payload")
        self.assertEqual(result["filesystem"]["filesystem_type"], "ext4")

    def test_export_settlement_append_updates_verified_manifest(self):
        source = self.root / "source"
        destination = self.root / "archive"
        source.mkdir()
        (source / "payload.txt").write_text("payload", encoding="utf-8")
        infra_io.controlled_export(source, destination)
        settlement = source / "export-status.json"
        write_json(settlement, {"status": "verified"})
        complete = infra_io.append_verified_export_file(
            settlement, destination, "export-status.json"
        )
        self.assertEqual(complete["final_settlement_included"], "export-status.json")
        manifest = infra_io.read_json(destination / "export-manifest.json")
        self.assertEqual(
            manifest["files"]["export-status.json"]["sha256"],
            infra_io.sha256_file(settlement),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
