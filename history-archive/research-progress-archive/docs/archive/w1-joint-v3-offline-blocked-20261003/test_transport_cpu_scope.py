import json
import os
import subprocess
import sys
import textwrap
import unittest


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux /proc and prctl")
class TransportCpuScopeTests(unittest.TestCase):
    def test_waited_short_lived_grandchild_and_root_tail_are_inclusive_once(self):
        clock_ticks = os.sysconf("SC_CLK_TCK")
        grandchild = textwrap.dedent("""
            import time
            end = time.process_time() + 0.36
            value = 1
            while time.process_time() < end:
                value = (value * 1103515245 + 12345) & 0x7fffffff
        """)
        root = textwrap.dedent("""
            import json, subprocess, sys, time
            from transport_cpu_scope import TransportCpuScope

            scope = TransportCpuScope()
            worker = (
                "import subprocess,sys; "
                "child=subprocess.Popen([sys.executable,'-c',sys.argv[1]]); "
                "sys.exit(child.wait())"
            )
            process = scope.spawn_remote_process(
                [sys.executable, '-c', worker, sys.argv[1]],
                purpose='ssh_exec', stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate(timeout=8)
            if process.returncode != 0:
                raise RuntimeError(stderr.decode('utf-8', errors='replace'))
            before_tail = scope.scope_report()['process_tree_cpu_seconds_measured']
            tail_end = time.process_time() + 0.08
            value = 1
            while time.process_time() < tail_end:
                value = (value * 1664525 + 1013904223) & 0xffffffff
            after_tail = scope.scope_report()['process_tree_cpu_seconds_measured']
            tail_delta = after_tail - before_tail
            if tail_delta < 0.06:
                raise RuntimeError('TAIL_CPU_DELTA_BELOW_TEST_THRESHOLD')
            report = scope.finalize(reap_timeout_seconds=1.0)
            print(json.dumps({'report': report, 'tail_cpu_delta_seconds': tail_delta}, sort_keys=True))
        """)
        completed = subprocess.run(
            [sys.executable, "-c", root, grandchild],
            capture_output=True, text=True, timeout=12,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        report = result["report"]
        self.assertEqual(report["process_tree_scope_status"], "verified_complete_at_snapshot")
        self.assertEqual(report["cleanup_status"], "reaped")
        self.assertGreaterEqual(report["root_reaped_descendant_cpu_ticks"], int(clock_ticks * 0.24))
        self.assertGreaterEqual(report["process_tree_cpu_seconds_measured"], 0.24)
        self.assertGreaterEqual(result["tail_cpu_delta_seconds"], 0.06)
        self.assertEqual(
            report["process_tree_cpu_seconds_charged_once"],
            report["process_tree_cpu_seconds_measured"],
        )
        self.assertEqual(report["remote_transport_scope_status"], "incomplete_requires_host_scope")
        self.assertIsNone(report["authoritative_remote_transport_cpu_seconds"])
        self.assertIn("sshd session/channel process", " ".join(report["unmeasured_remote_transport_components"]))
        self.assertIn("host-admin", report["required_host_admin_condition"])

    def test_sftp_server_child_is_owned_by_the_same_root(self):
        scope_script = textwrap.dedent("""
            import json, subprocess, sys, textwrap, time
            from transport_cpu_scope import TransportCpuScope

            scope = TransportCpuScope()
            code = textwrap.dedent('''
                import sys, time
                started = time.process_time()
                value = 1
                while time.process_time() - started < 0.07:
                    value = (value * 3 + 1) & 0x7fffffff
                elapsed = time.process_time() - started
                print('TRANSPORT_CPU_STUB_COMPLETED=' + str(elapsed), file=sys.stderr, flush=True)
            ''')
            server = scope.spawn_sftp_server(
                [sys.executable, '-c', code], stderr=subprocess.PIPE,
            )
            server.stdin.close()
            returncode = scope.wait(server, timeout=5)
            server_stderr = server.stderr.read().decode('utf-8', errors='replace')
            if returncode != 0:
                raise RuntimeError('SFTP CPU stub failed: ' + server_stderr)
            marker = next((line for line in server_stderr.splitlines()
                           if line.startswith('TRANSPORT_CPU_STUB_COMPLETED=')), None)
            if marker is None:
                raise RuntimeError('SFTP CPU stub completion evidence missing: ' + server_stderr)
            stub_cpu_seconds = float(marker.split('=', 1)[1])
            report = scope.finalize(reap_timeout_seconds=1.0)
            print(json.dumps({
                'report': report,
                'server_returncode': returncode,
                'server_stderr': server_stderr,
                'stub_cpu_seconds': stub_cpu_seconds,
            }, sort_keys=True))
        """)
        completed = subprocess.run(
            [sys.executable, "-c", scope_script],
            capture_output=True, text=True, timeout=8,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        report = result["report"]
        self.assertEqual(report["process_tree_scope_status"], "verified_complete_at_snapshot")
        self.assertEqual(result["server_returncode"], 0)
        self.assertIn("TRANSPORT_CPU_STUB_COMPLETED=", result["server_stderr"])
        self.assertGreater(result["stub_cpu_seconds"], 0.05)
        waited_cpu_seconds = report["root_reaped_descendant_cpu_ticks"] / report["clock_ticks_per_second"]
        self.assertGreaterEqual(waited_cpu_seconds, result["stub_cpu_seconds"] - 0.03)
        self.assertGreater(report["process_tree_cpu_seconds_measured"], result["stub_cpu_seconds"])
        self.assertEqual(report["authoritative_remote_transport_cpu_seconds"], None)
        self.assertEqual(report["remote_transport_scope_status"], "incomplete_requires_host_scope")


if __name__ == "__main__":
    unittest.main()
