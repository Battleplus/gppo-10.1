"""Synthetic tests for the REGISTERED receipt and START_ACK boundary."""
from __future__ import annotations

import json
import queue
import threading
import unittest

from registration_start_handshake import (
    ACK_SCHEMA,
    RECEIPT_SCHEMA,
    RegistrationStartHandshakeError,
    budget_binding_sha256,
    controller_receive_report_and_ack,
    register_then_wait_for_start_ack,
)


MANIFEST = "a" * 64
HASHES = "b" * 64
REAL_NAME = "hzy"
NAME_ID = "hzy"
REGISTRATION_PATH = "/media/abc_disk/admin123/lbh/dengji.txt"
BUDGET = {"totals": {"wall_seconds": 180, "complete_process_cpu_seconds": 360}}
JOB_ID = "20261003-hzy-deadbeef"


def registered_job():
    return {
        "job_id": JOB_ID,
        "event": "REGISTERED",
        "status": "REGISTERED",
        "real_name": REAL_NAME,
        "name_id": NAME_ID,
        "binding": {
            "manifest_sha256": MANIFEST,
            "hashes_sha256": HASHES,
            "budget": BUDGET,
            "budget_sha256": budget_binding_sha256(BUDGET),
        },
    }


def registered_receipt():
    return {
        "schema": RECEIPT_SCHEMA,
        "event": "REGISTERED",
        "job_id": JOB_ID,
        "real_name": REAL_NAME,
        "name_id": NAME_ID,
        "registration_path": REGISTRATION_PATH,
        "manifest_sha256": MANIFEST,
        "hashes_sha256": HASHES,
        "budget_sha256": budget_binding_sha256(BUDGET),
    }


def ack_line(*, job_id=JOB_ID, manifest=MANIFEST, hashes=HASHES, extra=None):
    value = {
        "schema": ACK_SCHEMA,
        "event": "START_ACK",
        "job_id": job_id,
        "manifest_sha256": manifest,
        "hashes_sha256": hashes,
    }
    if extra:
        value.update(extra)
    return json.dumps(value, sort_keys=True) + "\n"


class QueueChannel:
    """The recv_ready/recv surface used by an SSH channel, backed by a queue."""

    def __init__(self):
        self.data = queue.Queue()
        self.closed = False

    def deliver(self, data):
        self.data.put(data)

    def recv_ready(self):
        return not self.data.empty()

    def recv(self, size):
        try:
            data = self.data.get_nowait()
        except queue.Empty:
            return b""
        if len(data) > size:
            self.data.put(data[size:])
            return data[:size]
        return data

    def exit_status_ready(self):
        return self.closed


class ChannelReceiptWriter:
    def __init__(self, channel, events):
        self.channel = channel
        self.events = events
        self.pending = ""

    def write(self, value):
        self.pending += value
        self.events.append("RECEIPT_WRITE")
        return len(value)

    def flush(self):
        self.events.append("RECEIPT_FLUSH")
        self.channel.deliver(self.pending.encode("utf-8"))
        self.pending = ""


class QueueAckInput:
    def __init__(self, inbound, events):
        self.inbound = inbound
        self.events = events

    def write(self, value):
        self.events.append("ACK_SENT")
        self.inbound.put(value)
        return len(value)

    def flush(self):
        pass


class LocalOutput:
    def __init__(self, events):
        self.events = events
        self.value = ""

    def write(self, value):
        self.value += value
        self.events.append("LOCAL_REPORT")
        return len(value)

    def flush(self):
        self.events.append("LOCAL_FLUSH")


class RegistrationStartHandshakeTests(unittest.TestCase):
    def server_kwargs(self, *, events, stdin, stdout, **updates):
        values = {
            "register": lambda: (events.append("REGISTERED") or registered_job()),
            "real_name": REAL_NAME,
            "name_id": NAME_ID,
            "registration_path": REGISTRATION_PATH,
            "manifest_sha256": MANIFEST,
            "hashes_sha256": HASHES,
            "budget": BUDGET,
            "stdin": stdin,
            "stdout": stdout,
            "timeout_seconds": 1.0,
        }
        values.update(updates)
        return values

    def test_registered_report_ack_running_order_over_channel(self):
        events = []
        channel = QueueChannel()
        inbound = queue.Queue()
        remote_stdout = ChannelReceiptWriter(channel, events)
        remote_stdin = QueueAckInput(inbound, events)
        local_stdout = LocalOutput(events)
        server_result = {}

        def server_readline(_stream, timeout):
            try:
                value = inbound.get(timeout=timeout)
            except queue.Empty as exc:
                raise TimeoutError from exc
            events.append("ACK_RECEIVED")
            return value

        def server():
            try:
                register_then_wait_for_start_ack(
                    **self.server_kwargs(
                        events=events, stdin=object(), stdout=remote_stdout,
                        read_line_with_timeout=server_readline,
                    )
                )
                events.append("RUNNING")
            except BaseException as exc:  # surfaced to the test thread
                server_result["error"] = exc

        thread = threading.Thread(target=server, daemon=True)
        thread.start()
        receipt = controller_receive_report_and_ack(
            channel=channel,
            remote_stdin=remote_stdin,
            local_stdout=local_stdout,
            real_name=REAL_NAME,
            name_id=NAME_ID,
            registration_path=REGISTRATION_PATH,
            manifest_sha256=MANIFEST,
            hashes_sha256=HASHES,
            budget=BUDGET,
            reporter=lambda _receipt: events.append("REPORT_CALLBACK"),
            acknowledger=lambda _receipt: events.append("ACK_DECISION") or True,
            timeout_seconds=1.0,
        )
        thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", server_result)
        self.assertEqual(receipt["job_id"], JOB_ID)
        self.assertIn('"event":"REGISTERED"', local_stdout.value)
        self.assertEqual(
            events,
            ["REGISTERED", "RECEIPT_WRITE", "RECEIPT_FLUSH", "LOCAL_REPORT",
             "LOCAL_FLUSH", "REPORT_CALLBACK", "ACK_DECISION", "ACK_SENT",
             "ACK_RECEIVED", "RUNNING"],
        )

    def test_server_rejects_bad_binding_eof_and_timeout_before_running(self):
        cases = (
            (ack_line(job_id="other-job"), "START_ACK_BINDING_MISMATCH"),
            (ack_line(extra={"approval": True}), "START_ACK_FIELDS_INVALID"),
            (EOFError(), "START_ACK_EOF"),
            (TimeoutError(), "START_ACK_TIMEOUT"),
        )
        for supplied, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                events = []
                output = []

                class Writer:
                    def write(self, value):
                        output.append(value)
                    def flush(self):
                        pass

                def reader(_stream, _timeout):
                    if isinstance(supplied, BaseException):
                        raise supplied
                    return supplied

                with self.assertRaisesRegex(RegistrationStartHandshakeError, expected_error):
                    register_then_wait_for_start_ack(
                        **self.server_kwargs(
                            events=events, stdin=object(), stdout=Writer(),
                            read_line_with_timeout=reader,
                        )
                    )
                self.assertEqual(events, ["REGISTERED"])
                self.assertEqual(len(output), 1)

    def test_server_rejects_registration_binding_before_receipt(self):
        events = []
        output = []
        wrong = registered_job()
        wrong["binding"]["hashes_sha256"] = "c" * 64

        class Writer:
            def write(self, value):
                output.append(value)
            def flush(self):
                pass

        with self.assertRaisesRegex(
            RegistrationStartHandshakeError, "REGISTERED_RECORD_BINDING_MISMATCH"
        ):
            register_then_wait_for_start_ack(
                **self.server_kwargs(
                    events=events, stdin=object(), stdout=Writer(),
                    register=lambda: (events.append("REGISTERED") or wrong),
                )
            )
        self.assertEqual(events, ["REGISTERED"])
        self.assertEqual(output, [])

    def test_controller_reports_receipt_but_sends_no_ack_on_refusal(self):
        events = []
        channel = QueueChannel()
        channel.deliver((json.dumps(registered_receipt()) + "\n").encode())
        local_stdout = LocalOutput(events)
        writes = []

        class Stdin:
            def write(self, value):
                writes.append(value)
            def flush(self):
                pass

        with self.assertRaisesRegex(RegistrationStartHandshakeError, "START_NOT_ACKNOWLEDGED"):
            controller_receive_report_and_ack(
                channel=channel,
                remote_stdin=Stdin(),
                local_stdout=local_stdout,
                real_name=REAL_NAME,
                name_id=NAME_ID,
                registration_path=REGISTRATION_PATH,
                manifest_sha256=MANIFEST,
                hashes_sha256=HASHES,
                budget=BUDGET,
                acknowledger=lambda receipt: events.append("ACK_REFUSED") or False,
                reporter=lambda receipt: events.append("REPORT_CALLBACK"),
                timeout_seconds=1.0,
            )
        self.assertIn('"event":"REGISTERED"', local_stdout.value)
        self.assertEqual(events, ["LOCAL_REPORT", "LOCAL_FLUSH", "REPORT_CALLBACK", "ACK_REFUSED"])
        self.assertEqual(writes, [])

    def test_controller_rejects_mismatched_receipt_without_reporting(self):
        channel = QueueChannel()
        receipt = registered_receipt()
        receipt["manifest_sha256"] = "d" * 64
        channel.deliver((json.dumps(receipt) + "\n").encode())
        local_stdout = LocalOutput([])

        with self.assertRaisesRegex(
            RegistrationStartHandshakeError, "REGISTERED_RECEIPT_BINDING_MISMATCH"
        ):
            controller_receive_report_and_ack(
                channel=channel,
                remote_stdin=object(),
                local_stdout=local_stdout,
                real_name=REAL_NAME,
                name_id=NAME_ID,
                registration_path=REGISTRATION_PATH,
                manifest_sha256=MANIFEST,
                hashes_sha256=HASHES,
                budget=BUDGET,
                acknowledger=lambda _receipt: True,
                timeout_seconds=1.0,
            )
        self.assertEqual(local_stdout.value, "")

    def test_controller_receipt_timeout_and_eof_are_rejected(self):
        for closed, expected_error in ((False, "REGISTERED_RECEIPT_TIMEOUT"),
                                       (True, "REGISTERED_RECEIPT_EOF")):
            with self.subTest(expected_error=expected_error):
                channel = QueueChannel()
                channel.closed = closed
                with self.assertRaisesRegex(RegistrationStartHandshakeError, expected_error):
                    controller_receive_report_and_ack(
                        channel=channel,
                        remote_stdin=object(),
                        local_stdout=LocalOutput([]),
                        real_name=REAL_NAME,
                        name_id=NAME_ID,
                        registration_path=REGISTRATION_PATH,
                        manifest_sha256=MANIFEST,
                        hashes_sha256=HASHES,
                        budget=BUDGET,
                        acknowledger=lambda _receipt: True,
                        timeout_seconds=0.02,
                    )

    def test_receipt_contains_only_registered_identity_and_binding_digests(self):
        events = []
        output = []

        class Writer:
            def write(self, value):
                output.append(value)
            def flush(self):
                pass

        def reader(_stream, _timeout):
            return ack_line()

        register_then_wait_for_start_ack(
            **self.server_kwargs(
                events=events, stdin=object(), stdout=Writer(),
                read_line_with_timeout=reader,
            )
        )
        receipt = json.loads(output[0])
        self.assertEqual(set(receipt), {
            "schema", "event", "job_id", "real_name", "name_id", "registration_path",
            "manifest_sha256", "hashes_sha256", "budget_sha256",
        })
        self.assertEqual(receipt["budget_sha256"], budget_binding_sha256(BUDGET))
        self.assertFalse(any("token" in key.lower() or "password" in key.lower() for key in receipt))


if __name__ == "__main__":
    unittest.main()
