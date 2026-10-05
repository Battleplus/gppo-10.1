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
from wsl_stage_and_launch import arm_wall_deadline, disarm_wall_deadline, run_native


@unittest.skipUnless(os.name == "posix", "Linux process accounting test")
class LinuxProcessAccountingTests(unittest.TestCase):
    def test_outer_wait4_cpu_contains_supervisor_self_and_reaped_worker(self):
        worker = "import time; end=time.process_time()+0.12; x=0; exec('while time.process_time()<end: x+=1')"
        supervisor = (
            "import resource,subprocess,sys; "
            "subprocess.run([sys.executable,'-B','-c',sys.argv[1]],check=True); "
            "s=resource.getrusage(resource.RUSAGE_SELF); c=resource.getrusage(resource.RUSAGE_CHILDREN); "
            "print(s.ru_utime+s.ru_stime+c.ru_utime+c.ru_stime)"
        )
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        child = subprocess.Popen(
            [sys.executable, "-B", "-c", supervisor, worker],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        supervisor_total = float(child.communicate(timeout=5)[0].strip())
        outer = resource.getrusage(resource.RUSAGE_CHILDREN)
        outer_total = (outer.ru_utime + outer.ru_stime) - (before.ru_utime + before.ru_stime)
        self.assertGreater(supervisor_total, 0.08)
        self.assertGreaterEqual(outer_total, supervisor_total)
        self.assertLess(outer_total - supervisor_total, 0.05)

    def test_native_child_wall_timeout_terminates_its_process_group(self):
        with tempfile.TemporaryDirectory(prefix="w1-launch-timeout-") as directory:
            root = Path(directory)
            (root / "hung_native.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
            code, interruption, error, started, child_cpu = run_native(
                sys.executable, root, "test-token", "unused-auth-path", "hung_native.py",
                timeout_seconds=0.1,
            )
        self.assertEqual(code, 124)
        self.assertIn("wall limit", interruption)
        self.assertIn("TimeoutError", error)
        self.assertTrue(started)
        self.assertGreaterEqual(child_cpu, 0.0)

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
