import os
import ctypes
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from launch_once import _child_process_cpu_seconds, run_wsl
from wsl_stage_and_launch import (
    arm_wall_deadline,
    build_native_cpu_scope_breakdown,
    disarm_wall_deadline,
    run_native,
)


@unittest.skipUnless(os.name == "posix", "Linux process accounting test")
class LinuxProcessAccountingTests(unittest.TestCase):
    def test_known_cpu_tree_is_partitioned_without_double_counting(self):
        with tempfile.TemporaryDirectory(prefix="w1-launch-cpu-tree-") as directory:
            root = Path(directory)
            launcher_script = root / "native_cpu_tree.py"
            launcher_script.write_text(
                "import json,resource,subprocess,sys,time\n"
                "process_module_entry=time.monotonic()\n"
                "def burn(seconds):\n"
                " end=time.process_time()+seconds; value=0\n"
                " while time.process_time()<end: value+=1\n"
                " return value\n"
                "burn(0.06)\n"
                "worker=\"import time; end=time.process_time()+0.12; x=0; exec('while time.process_time()<end: x+=1')\"\n"
                "subprocess.run([sys.executable,'-B','-c',worker],check=True)\n"
                "def usage():\n"
                " s=resource.getrusage(resource.RUSAGE_SELF); c=resource.getrusage(resource.RUSAGE_CHILDREN)\n"
                " return s.ru_utime+s.ru_stime,c.ru_utime+c.ru_stime\n"
                "supervisor_started=time.monotonic()\n"
                "self_cpu,child_cpu=usage(); sample=time.monotonic()\n"
                "supervisor={'cpu_seconds':self_cpu+child_cpu,'cpu_sampling_started_monotonic':supervisor_started,'cpu_sampled_monotonic':sample,'status':'complete'}\n"
                "( __import__('pathlib').Path('supervisor-status.json')).write_text(json.dumps(supervisor))\n"
                "self_cpu,child_cpu=usage(); sample=time.monotonic()\n"
                "launcher={'native_launcher_process_cpu_seconds':self_cpu,'native_launcher_reaped_children_cpu_seconds':child_cpu,'native_launcher_complete_process_cpu_snapshot_seconds':self_cpu+child_cpu,'process_module_entry_monotonic':process_module_entry,'sampled_monotonic':sample,'stage':'supervisor_returned'}\n"
                "( __import__('pathlib').Path('native-launcher-accounting.json')).write_text(json.dumps(launcher))\n",
                encoding="utf-8",
            )
            code, interruption, error, started, tree_cpu, sample = run_native(
                sys.executable, root, "test-token", "unused-auth-path", launcher_script.name,
                timeout_seconds=5.0,
            )
            supervisor = __import__("json").loads((root / "supervisor-status.json").read_text())
            launcher = __import__("json").loads((root / "native-launcher-accounting.json").read_text())

        breakdown = build_native_cpu_scope_breakdown(tree_cpu, supervisor, launcher, sample)
        self.assertEqual(code, 0)
        self.assertIsNone(interruption)
        self.assertIsNone(error)
        self.assertTrue(started)
        self.assertEqual(breakdown["status"], "reconciled")
        self.assertEqual(breakdown["errors"], [])
        self.assertGreater(launcher["native_launcher_process_cpu_seconds"], 0.02)
        self.assertGreater(launcher["native_launcher_reaped_children_cpu_seconds"], 0.08)
        self.assertGreater(breakdown["supervisor_sample_to_native_launcher_snapshot_cpu_seconds"], 0)
        self.assertLessEqual(abs(breakdown["native_launcher_snapshot_to_parent_reap_cpu_seconds"]), 0.05)
        self.assertAlmostEqual(
            tree_cpu,
            supervisor["cpu_seconds"]
            + breakdown["supervisor_sample_to_native_launcher_snapshot_cpu_seconds"]
            + breakdown["native_launcher_snapshot_to_parent_reap_cpu_seconds"],
            places=6,
        )

    def test_nested_cpu_snapshot_timestamps_must_be_finite_ordered_and_in_outer_scope(self):
        tree_sample = {"sample_started_monotonic": 10.0, "sample_ended_monotonic": 14.0}
        supervisor = {
            "cpu_seconds": 0.07,
            "cpu_sampling_started_monotonic": 11.0,
            "cpu_sampled_monotonic": 12.0,
        }
        launcher = {
            "native_launcher_process_cpu_seconds": 0.08,
            "native_launcher_reaped_children_cpu_seconds": 0.01,
            "native_launcher_complete_process_cpu_snapshot_seconds": 0.09,
            "process_module_entry_monotonic": 10.5,
            "sampled_monotonic": 13.0,
        }
        valid = build_native_cpu_scope_breakdown(0.1, supervisor, launcher, tree_sample)
        self.assertEqual(valid["status"], "reconciled")
        self.assertEqual(valid["errors"], [])

        invalid_cases = (
            ({**supervisor, "cpu_sampling_started_monotonic": None}, launcher, "supervisor start timestamp unavailable", tree_sample),
            ({**supervisor, "cpu_sampled_monotonic": 9.0}, launcher, "SUPERVISOR_CPU_SNAPSHOT_CLOCK_ORDER_INVALID", tree_sample),
            (supervisor, {**launcher, "sampled_monotonic": None}, "native-launcher sample timestamp unavailable", tree_sample),
            (supervisor, {**launcher, "process_module_entry_monotonic": 14.1}, "NATIVE_LAUNCHER_ENTRY_AFTER_OUTER_TREE", tree_sample),
            (supervisor, {**launcher, "sampled_monotonic": 10.4}, "NATIVE_LAUNCHER_CPU_SNAPSHOT_CLOCK_ORDER_INVALID", tree_sample),
            (supervisor, {**launcher, "sampled_monotonic": 11.5}, "NESTED_CPU_SNAPSHOT_TIMESTAMPS_OUT_OF_ORDER", tree_sample),
            (supervisor, launcher, "NESTED_CPU_SNAPSHOT_TIMESTAMPS_OUT_OF_ORDER",
             {"sample_started_monotonic": 10.0, "sample_ended_monotonic": 12.5}),
        )
        for bad_supervisor, bad_launcher, expected_error, bad_sample in invalid_cases:
            with self.subTest(expected_error=expected_error):
                breakdown = build_native_cpu_scope_breakdown(
                    0.1, bad_supervisor, bad_launcher, bad_sample,
                )
                self.assertEqual(breakdown["status"], "failed")
                self.assertIn(expected_error, breakdown["errors"])

    def test_native_exception_retains_outer_cpu_measurement(self):
        with tempfile.TemporaryDirectory(prefix="w1-launch-exception-") as directory:
            root = Path(directory)
            (root / "native_exception.py").write_text("raise RuntimeError('fixture failure')\n", encoding="utf-8")
            code, interruption, error, started, cpu, sample = run_native(
                sys.executable, root, "test-token", "unused-auth-path", "native_exception.py",
                timeout_seconds=5.0,
            )
        self.assertNotEqual(code, 0)
        self.assertTrue(started)
        self.assertGreaterEqual(cpu, 0.0)
        self.assertTrue(sample["wait_completed"])
        self.assertIsNone(error)

    def test_native_startup_failure_is_measured_without_worker(self):
        with tempfile.TemporaryDirectory(prefix="w1-launch-startup-") as directory:
            code, interruption, error, started, cpu, sample = run_native(
                "/definitely/missing/python", Path(directory), "test-token", "unused-auth-path", "missing.py",
                timeout_seconds=5.0,
            )
        self.assertEqual(code, 1)
        self.assertFalse(started)
        self.assertIsNotNone(error)
        self.assertIn("FileNotFoundError", error)
        self.assertGreaterEqual(cpu, 0.0)
        self.assertFalse(sample["wait_completed"])

    def test_startup_scope_is_explicitly_launcher_only(self):
        scope = build_native_cpu_scope_breakdown(
            0.02,
            None,
            {
                "native_launcher_process_cpu_seconds": 0.02,
                "native_launcher_reaped_children_cpu_seconds": 0.0,
                "native_launcher_complete_process_cpu_snapshot_seconds": 0.02,
                "process_module_entry_monotonic": 1.1,
                "sampled_monotonic": 2.0,
                "stage": "native_preflight",
            },
            {"sample_started_monotonic": 1.0, "sample_ended_monotonic": 2.0},
        )
        self.assertEqual(scope["status"], "launcher_only_early_exit")
        self.assertEqual(scope["errors"], [])

    def test_missing_inner_snapshots_preserve_tree_only_stop_classification(self):
        scope = build_native_cpu_scope_breakdown(
            0.01, None, None,
            {"sample_started_monotonic": 1.0, "sample_ended_monotonic": 2.0},
        )
        self.assertEqual(scope["status"], "tree_only_timeout_or_early_exit")
        self.assertEqual(scope["errors"], [])

    def test_native_child_wall_timeout_terminates_its_process_group(self):
        with tempfile.TemporaryDirectory(prefix="w1-launch-timeout-") as directory:
            root = Path(directory)
            (root / "hung_native.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
            code, interruption, error, started, child_cpu, sample = run_native(
                sys.executable, root, "test-token", "unused-auth-path", "hung_native.py",
                timeout_seconds=0.1,
            )
        self.assertEqual(code, 124)
        self.assertIn("wall limit", interruption)
        self.assertIn("TimeoutError", error)
        self.assertTrue(started)
        self.assertGreaterEqual(child_cpu, 0.0)
        self.assertTrue(sample["wait_completed"])
        self.assertEqual(sample["termination"], "forced_process_group_stop")

    def test_linux_wall_alarm_interrupts_synchronous_stage_work(self):
        arm_wall_deadline(0.03)
        started = time.monotonic()
        try:
            with self.assertRaisesRegex(TimeoutError, "stage wall deadline"):
                time.sleep(1)
        finally:
            disarm_wall_deadline()
        self.assertLess(time.monotonic() - started, 0.5)


class WindowsProcessAccountingTests(unittest.TestCase):
    def test_get_process_times_counts_kernel_and_user_ticks(self):
        class FakeGetProcessTimes:
            def __call__(self, handle, creation, exit_time, kernel, user):
                kernel._obj.low = 5_000_000
                user._obj.low = 2_500_000
                return 1

        child = SimpleNamespace(_handle=123)
        api = SimpleNamespace(GetProcessTimes=FakeGetProcessTimes())
        with patch("launch_once.os.name", "nt"), patch(
            "ctypes.WinDLL", return_value=api, create=True
        ):
            cpu = _child_process_cpu_seconds(child, None)
        self.assertAlmostEqual(cpu, 0.75)

    def test_timeout_terminates_once_and_does_not_retry(self):
        class FakeChild:
            pid = 778899

            def __init__(self):
                self.returncode = None
                self.communicate_calls = 0
                self.wait_calls = 0

            def communicate(self, input=None, timeout=None):
                self.communicate_calls += 1
                if self.communicate_calls == 1:
                    raise subprocess.TimeoutExpired(["wsl.exe"], timeout)
                return b"", None

            def wait(self, timeout=None):
                self.wait_calls += 1
                self.returncode = -15
                return self.returncode

            def poll(self):
                return self.returncode

            def kill(self):
                self.returncode = -9

        child = FakeChild()
        with (
            patch("launch_once.os.name", "nt"),
            patch("launch_once.subprocess.Popen", return_value=child) as popen,
            patch("launch_once.subprocess.run") as taskkill,
            patch("launch_once._child_process_cpu_seconds", return_value=0.01),
        ):
            code, timed_out, child_cpu = run_wsl(["wsl.exe", "--exec", "python"], "", timeout_seconds=0.02)
        self.assertEqual(code, 124)
        self.assertTrue(timed_out)
        self.assertEqual(child_cpu, 0.01)
        popen.assert_called_once()
        self.assertEqual(taskkill.call_count, 1)
        self.assertEqual(taskkill.call_args.args[0][:4], ["taskkill", "/PID", str(child.pid), "/T"])
        self.assertEqual(taskkill.call_args.args[0][4], "/F")
        self.assertEqual(child.wait_calls, 1)
        self.assertEqual(child.communicate_calls, 2)


if __name__ == "__main__":
    unittest.main()
