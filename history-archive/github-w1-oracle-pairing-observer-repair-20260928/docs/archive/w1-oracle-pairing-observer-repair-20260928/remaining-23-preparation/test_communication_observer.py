from __future__ import annotations

import copy
import csv
import json
import sys
import unittest
from pathlib import Path

from communication_observer import (
    InsufficientPairingEvidence,
    PairingMismatch,
    RecordingCommunication,
    verify_primitive_pairing,
)


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "native"))
from gppo_world.m10_communication import formal_three_condition_profile


def indexed_input(role):
    index = json.loads((ROOT / "input-index.json").read_text(encoding="utf-8"))
    raw = next(row["path"] for row in index["inputs"] if row["role"] == role)
    if len(raw) >= 3 and raw[1:3] == ":\\" and Path("/mnt").is_dir():
        return Path("/mnt") / raw[0].lower() / raw[3:].replace("\\", "/")
    return Path(raw)


class Delegate:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def _call(self, method, kwargs, result):
        self.calls.append((method, copy.deepcopy(kwargs)))
        if self.fail:
            raise RuntimeError("injected primitive failure")
        return result

    def telemetry(self, **kwargs):
        return self._call("telemetry", kwargs, {"dropped": False, "jitter": 0.0})

    def command_delivered(self, **kwargs):
        return self._call("command_delivered", kwargs, True)

    def ack_delivered(self, **kwargs):
        return self._call("ack_delivered", kwargs, True)

    def renewal(self, **kwargs):
        return self._call("renewal", kwargs, {"dropped": False, "delay": 0.0, "duplicate": False})


def call(arm, ordinal, method, identity, result, **parameters):
    return {
        "ordinal": ordinal,
        "branch": arm,
        "callsite": f"communication.{method}",
        "method": method,
        "arguments": {"identity": identity, **parameters},
        "status": "returned",
        "result": result,
    }


def summary(arm, calls):
    return {"identity": {"arm": arm}, "primitive_calls": calls}


class ObserverTests(unittest.TestCase):
    def test_real_frozen_communication_profile_is_transparent(self):
        direct = formal_three_condition_profile("W1")
        observer = RecordingCommunication(formal_three_condition_profile("W1"))
        observer.start_branch("action-03")
        invocations = [
            ("telemetry", {"seed": 9101, "identity": "packet-a", "now": 5.0}),
            ("command_delivered", {"seed": 9101, "identity": "command-a"}),
            ("ack_delivered", {"seed": 9101, "identity": "command-a|ack"}),
            ("renewal", {"seed": 9101, "identity": "command-a|renew|00006"}),
        ]
        expected = [getattr(direct, method)(**kwargs) for method, kwargs in invocations]
        observed = [getattr(observer, method)(**kwargs) for method, kwargs in invocations]
        self.assertEqual(observed, expected)
        self.assertEqual([row["method"] for row in observer.calls], [row[0] for row in invocations])
        self.assertEqual([row["arguments"] for row in observer.calls], [row[1] for row in invocations])

    def test_observer_preserves_return_identity_count_and_order(self):
        delegate = Delegate()
        observer = RecordingCommunication(delegate)
        marker = observer.start_branch("action-00")
        telemetry = observer.telemetry(seed=7, identity="packet-a", now=1.0)
        command = observer.command_delivered(seed=7, identity="command-a")
        ack = observer.ack_delivered(seed=7, identity="command-a")
        renewal = observer.renewal(seed=7, identity="renewal-a")
        self.assertEqual(telemetry, {"dropped": False, "jitter": 0.0})
        self.assertIs(command, True)
        self.assertIs(ack, True)
        self.assertEqual(renewal["delay"], 0.0)
        self.assertEqual([item[0] for item in delegate.calls], [
            "telemetry", "command_delivered", "ack_delivered", "renewal"
        ])
        records = observer.records_since(marker)
        self.assertEqual(len(records), len(delegate.calls))
        self.assertEqual([item["ordinal"] for item in records], [0, 1, 2, 3])
        self.assertTrue(all(item["branch"] == "action-00" for item in records))

    def test_full_identity_and_parameters_are_preserved(self):
        observer = RecordingCommunication(Delegate())
        observer.start_branch("action-06")
        kwargs = {"seed": 925710000, "identity": "exogenous[3]:key|packet:x", "now": 4.0}
        observer.telemetry(**kwargs)
        self.assertEqual(observer.calls[0]["arguments"], kwargs)
        self.assertRegex(
            observer.calls[0]["callsite"],
            r"^test_communication_observer\.py:\d+:test_full_identity_and_parameters_are_preserved$",
        )

    def test_exception_is_recorded_and_rethrown(self):
        observer = RecordingCommunication(Delegate(fail=True))
        observer.start_branch("action-01")
        with self.assertRaisesRegex(RuntimeError, "injected primitive failure"):
            observer.command_delivered(seed=1, identity="command-x")
        self.assertEqual(observer.calls[0]["status"], "raised")
        self.assertEqual(observer.calls[0]["exception_type"], "RuntimeError")

    def test_repeated_calls_are_preserved(self):
        calls = [
            call("a", 0, "telemetry", "same", {"dropped": False}, seed=1, now=0.0),
            call("a", 1, "telemetry", "same", {"dropped": False}, seed=1, now=0.0),
        ]
        other = [
            call("b", 0, "telemetry", "same", {"dropped": False}, seed=1, now=0.0),
            call("b", 1, "telemetry", "same", {"dropped": False}, seed=1, now=0.0),
            call("b", 2, "command_delivered", "branch-b", True, seed=1),
        ]
        result = verify_primitive_pairing("p", 0, [summary("a", calls), summary("b", other)])
        self.assertEqual(len(result["repeated_common_call_keys"]), 1)
        self.assertEqual(result["multiplicity_mismatch_count"], 0)

    def test_true_result_mismatch_stops(self):
        left = [call("a", 0, "command_delivered", "same", True, seed=1)]
        right = [
            call("b", 0, "command_delivered", "same", False, seed=1),
            call("b", 1, "ack_delivered", "branch-b", True, seed=1),
        ]
        with self.assertRaises(PairingMismatch):
            verify_primitive_pairing("p", 0, [summary("a", left), summary("b", right)])

    def test_branch_only_call_does_not_shift_common_result(self):
        left = [
            call("a", 0, "telemetry", "common", {"dropped": False}, seed=1, now=0.0),
            call("a", 1, "command_delivered", "only-a", True, seed=1),
        ]
        right = [
            call("b", 0, "command_delivered", "only-b", False, seed=1),
            call("b", 1, "telemetry", "common", {"dropped": False}, seed=1, now=0.0),
        ]
        result = verify_primitive_pairing("p", 0, [summary("a", left), summary("b", right)])
        self.assertEqual(result["common_call_keys"], 1)
        self.assertEqual(result["action_specific_call_keys"], 2)

    def test_parameter_difference_is_reported_not_merged(self):
        left = [
            call("a", 0, "telemetry", "same", {"dropped": False}, seed=1, now=0.0),
            call("a", 1, "ack_delivered", "common-ack", True, seed=1),
            call("a", 2, "command_delivered", "only-a", True, seed=1),
        ]
        right = [
            call("b", 0, "telemetry", "same", {"dropped": False}, seed=2, now=0.0),
            call("b", 1, "ack_delivered", "common-ack", True, seed=1),
            call("b", 2, "command_delivered", "only-b", True, seed=1),
        ]
        result = verify_primitive_pairing("p", 0, [summary("a", left), summary("b", right)])
        self.assertEqual(result["parameter_conflict_count"], 1)

    def test_missing_fields_and_empty_common_set_do_not_pass(self):
        with self.assertRaises(InsufficientPairingEvidence):
            verify_primitive_pairing("p", 0, [summary("a", []), summary("b", [])])
        incomplete = [{"ordinal": 0, "branch": "a", "method": "telemetry"}]
        with self.assertRaises(InsufficientPairingEvidence):
            verify_primitive_pairing("p", 0, [summary("a", incomplete), summary("b", incomplete)])

    def test_command_renewal_and_ack_full_identities_do_not_collide(self):
        calls = [
            call("a", 0, "command_delivered", "cmd-2|version-1|action-0", True, seed=1),
            call("a", 1, "renewal", "cmd-2|renew|00003", {"dropped": False}, seed=1),
            call("a", 2, "ack_delivered", "cmd-2|renew|00003|ack|0", True, seed=1),
        ]
        keys = {
            json.dumps({"method": item["method"], "arguments": item["arguments"]}, sort_keys=True)
            for item in calls
        }
        self.assertEqual(len(keys), 3)

    def test_all_135_saved_telemetry_cases_are_classified_as_downstream(self):
        path = indexed_input("oracle_root_cause_mismatch_table")
        self.assertTrue(path.is_file())
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row["link"] == "telemetry"]
        self.assertEqual(len(rows), 135)
        self.assertTrue(all(row["root_cause"] == "downstream_stage_overcomparison" for row in rows))
        self.assertTrue(all(row["external_fate_status"] == "saved=pass;offline=pass" for row in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
