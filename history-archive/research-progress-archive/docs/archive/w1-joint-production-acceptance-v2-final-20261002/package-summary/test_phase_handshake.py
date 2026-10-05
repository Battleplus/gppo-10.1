import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

from phase_handshake import StageServer


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux Unix sockets")
class StageHandshakeTests(unittest.TestCase):
    def send_request(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            socket_path = str(Path(directory) / "phase.sock")
            request = {"stages": {
                "staging_and_zero_step_gate": {
                    "complete_process_cpu_seconds": 1000,
                    "wall_seconds": 1000,
                },
                "label_qualification": {
                    "complete_process_cpu_seconds": 1000,
                    "wall_seconds": 1000,
                },
            }}
            child_pid = {}

            def tree_reader(root_pid):
                return {root_pid: {}, child_pid["pid"]: {}}

            server = StageServer(socket_path, request, tree_reader=tree_reader)
            code = textwrap.dedent("""
                import json, os, socket, sys, time
                path, mode = sys.argv[1:]
                end = time.monotonic()
                start = end - 0.1
                row = {
                    'pid': os.getpid(),
                    'stage': 'staging_and_zero_step_gate',
                    'next_stage': 'label_qualification',
                    'cpu_start_seconds': 0.0,
                    'cpu_end_seconds': 0.1,
                    'cpu_seconds': 0.1,
                    'wall_start_monotonic': start,
                    'wall_end_monotonic': end,
                    'wall_seconds': end - start,
                }
                if mode == 'nan':
                    row['cpu_end_seconds'] = float('nan')
                elif mode == 'boolean':
                    row['cpu_end_seconds'] = True
                elif mode == 'reversed_wall':
                    row['wall_start_monotonic'] = end + 1.0
                    row['wall_seconds'] = 0.0
                channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                channel.connect(path)
                channel.sendall(json.dumps(row).encode() + b'\\n')
                response = b''
                while not response.endswith(b'\\n'):
                    response += channel.recv(4096)
                print(json.loads(response)['status'])
            """)
            process = subprocess.Popen(
                [sys.executable, "-c", code, socket_path, mode],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            child_pid["pid"] = process.pid
            rows = []
            error = None
            try:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline and process.poll() is None:
                    try:
                        rows.extend(server.process_pending(process.pid))
                    except RuntimeError as exc:
                        error = str(exc)
                        break
                    time.sleep(0.005)
                stdout, stderr = process.communicate(timeout=2)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=2)
                server.close()
            return rows, error, stdout, stderr

    def test_valid_boundary_is_acknowledged_and_advances_stage(self):
        rows, error, stdout, stderr = self.send_request("valid")
        self.assertIsNone(error, stderr)
        self.assertEqual(stdout.strip(), "acknowledged")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["stage"], "staging_and_zero_step_gate")
        self.assertEqual(rows[0]["next_stage"], "label_qualification")

    def test_nonfinite_boolean_and_reversed_values_are_rejected(self):
        for mode, expected in (
            ("nan", "PHASE_VALUE_INVALID:cpu_end_seconds"),
            ("boolean", "PHASE_VALUE_INVALID:cpu_end_seconds"),
            ("reversed_wall", "PHASE_WALL_CLOCK_REVERSED"),
        ):
            with self.subTest(mode=mode):
                rows, error, stdout, stderr = self.send_request(mode)
                self.assertEqual(rows, [])
                self.assertIn(expected, error or "")
                self.assertEqual(stdout.strip(), "rejected", stderr)


if __name__ == "__main__":
    unittest.main()
