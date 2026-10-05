"""Production registration routing tests with only stream and operation boundaries."""
import contextlib
import io
import json
import sys
import unittest
from unittest.mock import Mock, patch

import acceptance_entry
import launch_joint_once
from registration_start_handshake import register_then_wait_for_start_ack
from test_registration_start_handshake import (
    BUDGET, HASHES, JOB_ID, MANIFEST, NAME_ID, REAL_NAME, REGISTRATION_PATH,
    QueueChannel, ack_line, registered_job, registered_receipt,
)


class RegistrationProductionWiringTests(unittest.TestCase):
    def test_production_registration_rejects_ack_before_gpu_or_workload(self):
        events = []
        running, gpu, workload = Mock(), Mock(), Mock()

        def before_start(job):
            return register_then_wait_for_start_ack(
                register=lambda: job, real_name=REAL_NAME, name_id=NAME_ID,
                registration_path=REGISTRATION_PATH, manifest_sha256=MANIFEST,
                hashes_sha256=HASHES, budget=BUDGET, stdin=object(), stdout=io.StringIO(),
                read_line_with_timeout=lambda *_args: ack_line(job_id="wrong-job"),
            )

        with self.assertRaisesRegex(RuntimeError, "START_ACK_BINDING_MISMATCH"):
            acceptance_entry.run_registered_once(
                register=registered_job, report_before_start=before_start,
                mark_cpu_running=running, initialize_gpu_and_bind=gpu,
                run_workload=workload,
                append_terminal=lambda _job, event, *_args: events.append(event),
            )
        running.assert_not_called()
        gpu.assert_not_called()
        workload.assert_not_called()
        self.assertEqual(events, ["FAILED"])

    def test_production_registration_orders_ack_before_operations(self):
        events = []

        def before_start(job):
            result = register_then_wait_for_start_ack(
                register=lambda: job, real_name=REAL_NAME, name_id=NAME_ID,
                registration_path=REGISTRATION_PATH, manifest_sha256=MANIFEST,
                hashes_sha256=HASHES, budget=BUDGET, stdin=object(), stdout=io.StringIO(),
                read_line_with_timeout=lambda *_args: (events.append("ACK") or ack_line()),
            )
            return result

        acceptance_entry.run_registered_once(
            register=lambda: (events.append("REGISTERED") or registered_job()),
            report_before_start=before_start,
            mark_cpu_running=lambda _job: events.append("RUNNING"),
            initialize_gpu_and_bind=lambda _job: events.append("GPU_BOUNDARY"),
            run_workload=lambda _job: events.append("WORKLOAD_BOUNDARY"),
            append_terminal=lambda _job, event, *_args: events.append(event),
        )
        self.assertEqual(events, ["REGISTERED", "ACK", "RUNNING", "GPU_BOUNDARY", "WORKLOAD_BOUNDARY", "SUCCEEDED"])

    def test_production_controller_keeps_remote_stdin_open_until_ack(self):
        events = []

        class Channel(QueueChannel):
            def shutdown_write(self):
                events.append("SHUTDOWN_WRITE")

            def recv_exit_status(self):
                return 0

        channel = Channel()
        channel.deliver((json.dumps(registered_receipt()) + "\n").encode())

        class Input:
            def __init__(self):
                self.channel = channel

            def write(self, value):
                events.append("ACK_SENT" if '"START_ACK"' in value else "TOKEN_SENT")

            def flush(self):
                events.append("INPUT_FLUSH")

        class Output:
            def __init__(self):
                self.channel = channel

            def read(self):
                events.append("FINAL_OUTPUT_READ")
                return b'{"status":"synthetic_complete"}\n'

        class LocalOutput(io.StringIO):
            def flush(self):
                events.append("LOCAL_RECEIPT_FLUSH")

        client = Mock()
        client.exec_command.return_value = (Input(), Output(), io.BytesIO())
        with patch.object(sys, "stdin", io.StringIO("START_ACK:" + JOB_ID + "\n")), contextlib.redirect_stdout(LocalOutput()):
            result = launch_joint_once.execute_registered(
                client, ["synthetic-worker"], timeout=1, input_text="synthetic-token\n",
                real_name=REAL_NAME, name_id=NAME_ID, manifest_sha256=MANIFEST,
                hashes_sha256=HASHES, budget=BUDGET,
            )
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["registered_receipt"]["job_id"], JOB_ID)
        self.assertLess(events.index("TOKEN_SENT"), events.index("LOCAL_RECEIPT_FLUSH"))
        self.assertLess(events.index("LOCAL_RECEIPT_FLUSH"), events.index("ACK_SENT"))
        self.assertLess(events.index("ACK_SENT"), events.index("SHUTDOWN_WRITE"))
        self.assertLess(events.index("SHUTDOWN_WRITE"), events.index("FINAL_OUTPUT_READ"))


if __name__ == "__main__":
    unittest.main()
