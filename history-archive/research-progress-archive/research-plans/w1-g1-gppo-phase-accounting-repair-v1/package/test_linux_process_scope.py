import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import unittest

from linux_process_scope import (
    ProcessScopeError,
    enable_subreaper,
    monotonic_cpu_delta,
    proc_tree,
    reap_owned_children,
    tree_cpu_ticks,
)


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux /proc and prctl")
class LinuxProcessScopeTests(unittest.TestCase):
    def test_waited_grandchild_ticks_survive_identity_disappearance_and_error_exit(self):
        enable_subreaper()
        clock_ticks = os.sysconf("SC_CLK_TCK")
        before = tree_cpu_ticks(proc_tree(os.getpid()))
        grandchild = textwrap.dedent("""
            import time
            end = time.process_time() + 0.22
            value = 1
            while time.process_time() < end:
                value = (value * 1103515245 + 12345) & 0x7fffffff
        """)
        parent = textwrap.dedent("""
            import subprocess, sys, time
            code = sys.argv[1]
            child = subprocess.Popen([sys.executable, '-c', code])
            print(child.pid, flush=True)
            result = child.wait()
            time.sleep(0.2)
            raise SystemExit(7 if result == 0 else 8)
        """)
        process = subprocess.Popen(
            [sys.executable, "-c", parent, grandchild],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            child_pid = int(process.stdout.readline().strip())
            deadline = time.monotonic() + 5
            waited_ticks = 0
            while time.monotonic() < deadline:
                rows = proc_tree(process.pid)
                waited_ticks = rows[process.pid]["waited_ticks"]
                if child_pid not in rows and waited_ticks >= max(1, int(clock_ticks * 0.10)):
                    break
                time.sleep(0.01)
            self.assertNotIn(child_pid, rows)
            self.assertGreaterEqual(waited_ticks, max(1, int(clock_ticks * 0.10)))
            self.assertEqual(process.wait(timeout=5), 7)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
        after = tree_cpu_ticks(proc_tree(os.getpid()))
        self.assertGreaterEqual(after - before, int(clock_ticks * 0.12))

    def test_reaper_kills_only_post_baseline_adopted_children(self):
        code = textwrap.dedent("""
            import signal, subprocess, sys, textwrap, time
            from linux_process_scope import enable_subreaper, reap_owned_children

            unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])
            enable_subreaper()
            grandchild = textwrap.dedent('''
                import signal, time
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                time.sleep(20)
            ''')
            parent = textwrap.dedent('''
                import subprocess, sys, time
                child = subprocess.Popen([sys.executable, '-c', sys.argv[1]])
                print(child.pid, flush=True)
                time.sleep(20)
            ''')
            owned = subprocess.Popen([sys.executable, '-c', parent, grandchild],
                                     stdout=subprocess.PIPE, text=True,
                                     start_new_session=True)
            adopted_pid = int(owned.stdout.readline().strip())
            owned.kill()
            owned.wait(timeout=2)
            result = reap_owned_children(timeout_seconds=0.8)
            if unrelated.poll() is not None:
                raise AssertionError('baseline child was touched')
            if adopted_pid not in result['signaled_pids']:
                raise AssertionError('adopted descendant was not signaled')
            unrelated.terminate()
            unrelated.wait(timeout=2)
            print('reaped')
        """)
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("reaped", completed.stdout)

    def test_nonmonotonic_snapshot_delta_fails_closed(self):
        with self.assertRaisesRegex(ProcessScopeError, "NONMONOTONIC"):
            monotonic_cpu_delta(101, 100, 100)
        with self.assertRaisesRegex(ValueError, "TICK_VALUE_INVALID"):
            monotonic_cpu_delta(True, 100, 100)


if __name__ == "__main__":
    unittest.main()
