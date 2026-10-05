import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

if sys.platform.startswith("linux"):
    from phase_handshake import StageServer
    from linux_process_scope import proc_tree
else:
    StageServer = None
    proc_tree = None


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux Unix sockets")
class StageHandshakeTests(unittest.TestCase):
    def run_exchange(self, mode, *, server_hook=None, peer_offset=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            socket_path = root / "phase.sock"
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
            server = StageServer(socket_path, request, tree_reader=proc_tree,
                                 journal_path=root / "server-protocol.jsonl")
            if server_hook is not None:
                server._send_response = server_hook(server._send_response)
            code = textwrap.dedent(r"""
                import hashlib, json, os, socket, sys, time
                from phase_handshake import request_boundary as _request_boundary
                path, mode = sys.argv[1:]
                protocol_path = os.environ.get('W1_TEST_PROTOCOL_PATH')

                def producer_event(event, **fields):
                    if protocol_path:
                        def safe(value):
                            if isinstance(value, float) and (value != value or value in (float('inf'), float('-inf'))):
                                return repr(value)
                            if isinstance(value, dict):
                                return {str(key): safe(item) for key, item in value.items()}
                            if isinstance(value, (tuple, list)):
                                return [safe(item) for item in value]
                            return value
                        with open(protocol_path, 'a', encoding='utf-8') as handle:
                            handle.write(json.dumps(safe({'event': event, **fields}), sort_keys=True) + '\n')

                def request_boundary(row, **kwargs):
                    kwargs.setdefault('protocol_path', protocol_path)
                    return _request_boundary(row, **kwargs)

                def make_row(event, sequence, *, cpu_start=0.0, wall_start=None,
                             next_stage=None):
                    end_cpu = time.process_time()
                    end_wall = time.monotonic()
                    start_wall = end_wall - 0.01 if wall_start is None else wall_start
                    return {
                        'event': event,
                        'protocol_sequence': sequence,
                        'stage': 'staging_and_zero_step_gate',
                        'next_stage': next_stage or 'staging_and_zero_step_gate',
                        'pid': os.getpid(),
                        'cpu_start_seconds': cpu_start,
                        'cpu_end_seconds': end_cpu,
                        'cpu_seconds': 0.0 if event == 'stage_enter' else end_cpu - cpu_start,
                        'wall_start_monotonic': start_wall,
                        'wall_end_monotonic': end_wall,
                        'wall_seconds': 0.0 if event == 'stage_enter' else end_wall - start_wall,
                        'scope': 'synthetic test process',
                        **({'stage_elapsed_cpu_seconds': end_cpu - cpu_start,
                            'stage_elapsed_wall_seconds': end_wall - start_wall}
                           if event == 'stage_enter' else {}),
                    }

                def raw_request(row):
                    encoded = json.dumps(row, sort_keys=True, separators=(',', ':'),
                                         ensure_ascii=False, allow_nan=True).encode()
                    digest = hashlib.sha256(encoded).hexdigest()
                    message = {'action': 'request', 'sequence': row.get('protocol_sequence'),
                               'payload_sha256': digest, 'payload': row}
                    producer_event('request', sequence=row.get('protocol_sequence'),
                                   payload_sha256=digest, payload=row)
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
                        channel.connect(path)
                        channel.sendall(json.dumps(message, allow_nan=True).encode() + b'\n')
                        reply = b''
                        while not reply.endswith(b'\n'):
                            reply += channel.recv(4096)
                        value = json.loads(reply)
                        producer_event('server_accepted' if value.get('status') == 'accepted' else 'rejected',
                                       sequence=row.get('protocol_sequence'),
                                       payload_sha256=digest, response=value)
                        return value

                start = time.monotonic()
                entry = make_row('stage_enter', 1, wall_start=start)
                if mode == 'bad_nan':
                    entry['cpu_end_seconds'] = float('nan')
                    print(raw_request(entry))
                elif mode == 'bad_boolean':
                    entry['cpu_end_seconds'] = True
                    print(raw_request(entry))
                elif mode == 'bad_wall':
                    entry['wall_start_monotonic'] = entry['wall_end_monotonic'] + 1
                    entry['wall_seconds'] = 0
                    print(raw_request(entry))
                elif mode == 'peer_spoof':
                    request_boundary(entry)
                elif mode == 'valid':
                    request_boundary(entry)
                    transition = make_row('transition', 2, cpu_start=0.0,
                                          wall_start=start,
                                          next_stage='label_qualification')
                    print(request_boundary(transition)['status'])
                elif mode == 'duplicate':
                    first = request_boundary(entry)
                    second = request_boundary(entry)
                    print(first['status'], second['idempotent_replay'])
                elif mode == 'duplicate_old':
                    request_boundary(entry)
                    transition = make_row('transition', 2, cpu_start=0.0,
                                          wall_start=start,
                                          next_stage='label_qualification')
                    request_boundary(transition)
                    replay = request_boundary(entry)
                    print(replay['status'], replay['idempotent_replay'])
                elif mode == 'snapshot':
                    request_boundary(entry)
                    sample = make_row('snapshot', 2, cpu_start=0.0, wall_start=start)
                    request_boundary(sample)
                    transition = make_row('transition', 3, cpu_start=0.0,
                                          wall_start=start,
                                          next_stage='label_qualification')
                    print(request_boundary(transition)['status'])
                elif mode == 'conflict':
                    request_boundary(entry)
                    changed = dict(entry)
                    changed['scope'] = 'conflicting payload'
                    request_boundary(changed)
                elif mode == 'delay_accept':
                    try:
                        request_boundary(entry, timeout_seconds=0.08)
                    except RuntimeError as exc:
                        print(str(exc))
                elif mode == 'drop_final_ack':
                    request_boundary(entry)
                    transition = make_row('transition', 2, cpu_start=0.0,
                                          wall_start=start,
                                          next_stage='label_qualification')
                    try:
                        request_boundary(transition, timeout_seconds=0.15)
                    except RuntimeError as exc:
                        print(str(exc))
                elif mode == 'exit_before_confirm':
                    reply = raw_request(entry)
                    if reply.get('status') != 'accepted':
                        raise RuntimeError('expected accepted request')
                    os._exit(17)
                elif mode == 'exit_after_confirmation':
                    request_boundary(entry)
                    transition = make_row('transition', 2, cpu_start=0.0,
                                          wall_start=start,
                                          next_stage='label_qualification')
                    encoded = json.dumps(transition, sort_keys=True, separators=(',', ':'),
                                         ensure_ascii=False).encode()
                    digest = hashlib.sha256(encoded).hexdigest()
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
                        channel.connect(path)
                        envelope = {'action': 'request', 'sequence': 2,
                                    'payload_sha256': digest, 'payload': transition}
                        producer_event('request', sequence=2, payload_sha256=digest,
                                       payload=transition)
                        channel.sendall(json.dumps(envelope).encode() + b'\n')
                        reply = b''
                        while not reply.endswith(b'\n'):
                            reply += channel.recv(4096)
                        producer_event('server_accepted', sequence=2,
                                       payload_sha256=digest, response=json.loads(reply))
                        producer_event('worker_confirm_sent', sequence=2,
                                       payload_sha256=digest)
                        channel.sendall(json.dumps({'action': 'worker_confirm', 'sequence': 2,
                                                    'payload_sha256': digest}).encode() + b'\n')
                        os._exit(0)
            """)
            process = subprocess.Popen(
                [sys.executable, "-B", "-c", code, str(server.path), mode],
                cwd=Path(__file__).resolve().parent,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env={**os.environ, "W1_PHASE_ACCOUNTING_SOCKET": str(server.path),
                     "W1_TEST_PROTOCOL_PATH": str(root / "worker-protocol.jsonl")},
            )
            rows = []
            error = None
            try:
                deadline = time.monotonic() + 4
                while time.monotonic() < deadline and process.poll() is None:
                    try:
                        rows.extend(server.process_pending(process.pid + peer_offset))
                    except RuntimeError as exc:
                        error = str(exc)
                        break
                    time.sleep(0.002)
                stdout, stderr = process.communicate(timeout=2)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=2)
                server.close()
            audit = [json.loads(line) for line in (root / "server-protocol.jsonl").read_text().splitlines()]
            producer_path = root / "worker-protocol.jsonl"
            producer_audit = ([json.loads(line) for line in producer_path.read_text().splitlines()]
                              if producer_path.exists() else [])
            evidence_dir = os.environ.get("W1_TEST_EVIDENCE_DIR")
            if evidence_dir:
                from infra_io import durable_atomic_json

                evidence_root = Path(evidence_dir)
                evidence_root.mkdir(parents=True, exist_ok=True)
                evidence_path = evidence_root / "phase-protocol-test-evidence.json"
                evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else {
                    "schema": "phase-protocol-test-evidence/1.0.0", "scenarios": [],
                }
                source_files = (Path(__file__).resolve(), Path(__file__).with_name("phase_handshake.py"),
                                Path(__file__).with_name("budget_ledger.py"))
                evidence["scenarios"].append({
                    "scenario": mode,
                    "source_sha256": {item.name: hashlib.sha256(item.read_bytes()).hexdigest()
                                      for item in source_files},
                    "expected_assertion": {
                        "valid": "entry does not move CPU origin; transition is committed",
                        "duplicate": "same sequence and payload are idempotent",
                        "conflict": "same sequence with changed payload is rejected",
                        "delay_accept": "accepted request remains ambiguous and pending",
                        "drop_final_ack": "server commit is durable; worker result is ambiguous",
                        "exit_before_confirm": "no transition commit occurs",
                        "exit_after_confirmation": "transition commit remains durable",
                    }.get(mode, "request is rejected without transition mutation"),
                    "subprocess": {"returncode": process.returncode,
                                   "stdout": stdout, "stderr": stderr},
                    "worker_protocol_records": producer_audit,
                    "server_protocol_records": audit,
                    "observed": {"returned_rows": rows,
                                 "server_error": error,
                                 "server_stage": server.stage,
                                 "pending_sequence": None if server.pending_request is None
                                 else server.pending_request.get("sequence"),
                                 "rejection": server.last_rejection},
                })
                durable_atomic_json(evidence_path, evidence)
            return rows, error, stdout, stderr, process.returncode, server, audit

    def test_long_requested_path_uses_private_short_socket_and_cleans_it(self):
        with tempfile.TemporaryDirectory(prefix="w1-phase-long-") as directory:
            long_root = Path(directory) / ("final-" + "x" * 80) / ("attempt-" + "y" * 70)
            long_root.mkdir(parents=True)
            requested = long_root / ".phase-accounting.sock"
            self.assertGreater(len(os.fsencode(requested)), 108)
            server = StageServer(requested, {"stages": {}}, tree_reader=proc_tree)
            runtime_dir = server.runtime_dir
            socket_path = server.path
            try:
                self.assertLess(len(os.fsencode(socket_path)), 108)
                self.assertNotEqual(socket_path, requested)
                self.assertEqual(stat.S_IMODE(runtime_dir.stat().st_mode), 0o700)
                self.assertEqual(stat.S_IMODE(socket_path.stat().st_mode), 0o600)
            finally:
                server.close()
            self.assertFalse(socket_path.exists())
            self.assertFalse(runtime_dir.exists())

    def test_stage_enter_preserves_zero_origin_and_committed_transition_advances(self):
        rows, error, stdout, stderr, returncode, server, audit = self.run_exchange("valid")
        self.assertIsNone(error, stderr)
        self.assertEqual(returncode, 0)
        self.assertEqual(stdout.strip(), "acknowledged")
        self.assertEqual([row["event"] for row in rows], ["stage_enter", "transition"])
        self.assertEqual(rows[0]["cpu_seconds"], 0.0)
        self.assertFalse(rows[0]["transition_committed"])
        self.assertTrue(rows[1]["transition_committed"])
        self.assertEqual(rows[1]["stage"], "staging_and_zero_step_gate")
        self.assertEqual(rows[1]["next_stage"], "label_qualification")
        self.assertEqual(server.stage, "label_qualification")
        self.assertTrue(any(item["event"] == "accept" and not item["worker_confirmed"]
                            for item in audit))
        self.assertTrue(any(item["event"] == "worker_confirm"
                            for item in audit))
        self.assertEqual(sum(item["cpu_seconds"] for item in rows), rows[1]["cpu_seconds"])

    def test_invalid_numeric_values_and_peer_identity_are_rejected_and_durable(self):
        for mode, expected in (
            ("bad_nan", "PHASE_VALUE_INVALID:cpu_end_seconds"),
            ("bad_boolean", "PHASE_VALUE_INVALID:cpu_end_seconds"),
            ("bad_wall", "PHASE_WALL_CLOCK_REVERSED"),
            ("peer_spoof", "PHASE_PEER_IDENTITY_MISMATCH"),
        ):
            with self.subTest(mode=mode):
                rows, error, stdout, stderr, _code, _server, audit = self.run_exchange(
                    mode, peer_offset=1 if mode == "peer_spoof" else 0,
                )
                self.assertEqual(rows, [])
                if mode == "peer_spoof":
                    self.assertIn(expected, stdout + (error or ""))
                else:
                    self.assertIn(expected, stdout + (error or ""))
                self.assertTrue(any(item["event"] == "reject" and expected in item["error"]
                                    for item in audit))

    def test_identical_repeat_is_idempotent_and_conflicting_repeat_fails_closed(self):
        rows, error, stdout, stderr, returncode, server, audit = self.run_exchange("duplicate")
        self.assertIsNone(error, stderr)
        self.assertEqual(returncode, 0)
        self.assertEqual(stdout.strip(), "acknowledged True")
        self.assertEqual([row["event"] for row in rows], ["stage_enter"])
        self.assertTrue(any(item["event"] == "duplicate_commit" and item["idempotent"]
                            for item in audit))

        rows, error, stdout, stderr, returncode, server, audit = self.run_exchange("duplicate_old")
        self.assertIsNone(error, stderr)
        self.assertEqual(returncode, 0)
        self.assertEqual(stdout.strip(), "acknowledged True")
        self.assertEqual(server.stage, "label_qualification")
        self.assertEqual([row["event"] for row in rows], ["stage_enter", "transition"])
        self.assertTrue(any(item["event"] == "duplicate_commit" and item["sequence"] == 1
                            for item in audit))

        rows, error, stdout, stderr, _code, server, audit = self.run_exchange("conflict")
        self.assertIn("PHASE_SEQUENCE_PAYLOAD_CONFLICT", error or stdout)
        self.assertEqual(server.stage, "staging_and_zero_step_gate")
        self.assertTrue(any(item["event"] == "reject"
                            and item["error"] == "PHASE_SEQUENCE_PAYLOAD_CONFLICT"
                            for item in audit))

    def test_same_stage_snapshot_updates_sampling_baseline_only(self):
        rows, error, stdout, stderr, returncode, server, audit = self.run_exchange("snapshot")
        self.assertIsNone(error, stderr)
        self.assertEqual(returncode, 0)
        self.assertEqual(stdout.strip(), "acknowledged")
        self.assertEqual([row["event"] for row in rows], ["stage_enter", "snapshot", "transition"])
        self.assertEqual(rows[1]["cpu_seconds"], 0.0)
        self.assertFalse(rows[1]["transition_committed"])
        self.assertEqual(server.stage, "label_qualification")
        self.assertAlmostEqual(rows[2]["worker_reported_cpu_seconds"],
                               rows[2]["worker_cpu_end_seconds"])
        self.assertEqual([item["event"] for item in audit if item["event"] == "commit"],
                         ["commit", "commit", "commit"])

    def test_delayed_accept_ack_leaves_durable_pending_ambiguity_without_retry(self):
        def delay_accept(original):
            def send(channel, value):
                if value.get("status") == "accepted":
                    time.sleep(0.2)
                return original(channel, value)
            return send

        rows, error, stdout, stderr, _code, server, audit = self.run_exchange(
            "delay_accept", server_hook=delay_accept,
        )
        self.assertIn("PHASE_ACCOUNTING_OUTCOME_AMBIGUOUS", stdout)
        self.assertEqual(rows, [])
        self.assertEqual(server.stage, "staging_and_zero_step_gate")
        self.assertIsNotNone(server.pending_request)
        self.assertEqual(sum(item["event"] == "request" for item in audit), 1)
        self.assertTrue(any(item["event"] == "accept" for item in audit))
        self.assertTrue(any(item["event"] == "confirmation_ambiguous" for item in audit))

    def test_lost_final_ack_keeps_worker_ambiguity_after_durable_server_commit(self):
        def drop_transition_commit(original):
            def send(channel, value):
                if value.get("status") == "committed" and value.get("sequence") == 2:
                    raise BrokenPipeError("injected final ACK loss")
                return original(channel, value)
            return send

        rows, error, stdout, stderr, _code, server, audit = self.run_exchange(
            "drop_final_ack", server_hook=drop_transition_commit,
        )
        self.assertIn("PHASE_ACCOUNTING_OUTCOME_AMBIGUOUS", stdout)
        self.assertEqual(server.stage, "label_qualification")
        self.assertIsNone(server.pending_request)
        self.assertEqual([row["event"] for row in rows], ["stage_enter", "transition"])
        self.assertTrue(any(item["event"] == "commit" and item["sequence"] == 2
                            and item["transition_committed"] for item in audit))
        self.assertTrue(any(item["event"] == "final_ack_ambiguous" and item["sequence"] == 2
                            for item in audit))

    def test_worker_exit_before_confirm_keeps_accepted_request_pending(self):
        rows, error, stdout, stderr, returncode, server, audit = self.run_exchange("exit_before_confirm")
        self.assertEqual(returncode, 17)
        self.assertEqual(rows, [])
        self.assertEqual(server.stage, "staging_and_zero_step_gate")
        self.assertIsNotNone(server.pending_request)
        self.assertTrue(any(item["event"] == "accept" for item in audit))
        self.assertTrue(any(item["event"] == "confirmation_ambiguous" for item in audit))
        self.assertFalse(any(item["event"] == "commit" for item in audit))

    def test_worker_exit_after_confirm_leaves_durable_commit_and_ambiguous_final_ack(self):
        rows, error, stdout, stderr, returncode, server, audit = self.run_exchange("exit_after_confirmation")
        self.assertEqual(returncode, 0)
        self.assertEqual(server.stage, "label_qualification")
        self.assertTrue(any(item["event"] == "worker_confirm" and item["sequence"] == 2
                            for item in audit))
        self.assertTrue(any(item["event"] == "commit" and item["sequence"] == 2
                            and item["transition_committed"] for item in audit))
        self.assertTrue(any(item["event"] == "final_ack_ambiguous" and item["sequence"] == 2
                            for item in audit))


if __name__ == "__main__":
    unittest.main()
