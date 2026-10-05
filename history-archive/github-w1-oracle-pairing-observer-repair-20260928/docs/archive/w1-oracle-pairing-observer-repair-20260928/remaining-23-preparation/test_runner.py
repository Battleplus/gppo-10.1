from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from accounting import Ledger
from classical_baselines import ClassicalSelector, PublicDecisionAdapter
from communication_observer import RecordingCommunication, verify_primitive_pairing
from runner import (
    InsufficientTechnicalCoverage,
    PARENTS,
    _candidate_scan,
    execute_matrix,
    execute_unit,
)


ROOT = Path(__file__).resolve().parent


def field(value):
    return [float(value), 1.0, 1.0, 0.0]


def observation(time_value: int, actions=(0, 6, 24)):
    uavs = []
    for index in range(4):
        values = (
            float(index * 5),
            0.0,
            9.0,
            1.0,
            1.0,
            1.0,
        )
        uavs.append(sum((field(value) for value in values), []))
    tasks = []
    for index in range(6):
        values = (
            0.0,
            float(index),
            12.0 + index,
            1.0,
            1.0,
            0.0,
            float(index % 3),
            float(index),
        )
        tasks.append(sum((field(value) for value in values), []))
    mask = [False] * 25
    for action in actions:
        mask[action] = True
    return {
        "uavs": uavs,
        "tasks": tasks,
        "mask": mask,
        "time": float(time_value),
        "version": time_value,
        "public_entity_ids": {
            "uavs": [f"uav-{index}" for index in range(4)],
            "tasks": [f"task-{index}" for index in range(6)],
        },
        "continuation_actions": [],
    }


class FakeCommunication:
    name = "W1-light"

    def __init__(self):
        self.order = []

    def telemetry(self, **kwargs):
        self.order.append(("telemetry", dict(kwargs)))
        return {"dropped": False, "outage": False, "duplicate": False, "jitter": 0.0, "loss": False}

    def command_delivered(self, **kwargs):
        self.order.append(("command_delivered", dict(kwargs)))
        return True

    def ack_delivered(self, **kwargs):
        self.order.append(("ack_delivered", dict(kwargs)))
        return True

    def renewal(self, **kwargs):
        self.order.append(("renewal", dict(kwargs)))
        return {"dropped": False, "delay": 0.0, "duplicate": False}


class FakeEnv:
    def __init__(self, key: str, *, opportunity=True, rejected=False, missing_label=False, fail_step=False):
        self._exogenous_key = key
        self.opportunity = opportunity
        self.rejected = rejected
        self.missing_label = missing_label
        self.fail_step = fail_step
        self.step_count = 0
        self.first_action = None
        self.communication = RecordingCommunication(FakeCommunication())

    def reset(self):
        actions = (0, 6, 24) if self.opportunity else (24,)
        return observation(0, actions)

    def step(self, action):
        if self.fail_step:
            raise RuntimeError("injected step failure")
        if self.first_action is None:
            self.first_action = int(action)
        self.step_count += 1
        common_identity = f"exogenous|packet|common|{self.step_count}"
        self.communication.telemetry(seed=7, identity=common_identity, now=float(self.step_count))
        self.communication.command_delivered(
            seed=7,
            identity=f"exogenous|packet|command|v{self.step_count}|action-{action}",
        )
        done = self.step_count >= 2
        completed = int(done and self.first_action == 6 and not self.rejected)
        feedback = (
            "rejected_busy"
            if self.rejected and self.first_action == 6 and self.step_count == 1
            else "noop"
            if action == 24
            else "accepted"
        )
        tasks = {
            f"task-{index}": (
                "completed" if index == 0 and completed else "available"
            )
            for index in range(6)
        }
        completion_records = {}
        if completed:
            completion_records["task-0"] = {
                "physical_arrival_time": 2.0,
                "host_confirmation_time": 2.0,
                "physical_arrival_before_deadline": True,
                "host_confirmation_before_deadline": True,
                "uav_id": "uav-1",
            }
        common = {
            "link": "telemetry",
            "identity": f"uav|uav-0|x|{self.step_count}|{self.step_count:.9f}",
            "message_id": f"uav|uav-0|x|{self.step_count}|{self.step_count:.9f}",
            "delivery_ordinal": 0,
            "status": "sent",
            "received_at": float(self.step_count + 1),
        }
        command = {
            "link": "command",
            "message_id": f"command|action-{self.first_action}",
            "status": "sent",
        }
        info = {
            "feedback": feedback,
            "counts": {"completed": completed, "expired": 0},
            "energy": {
                f"uav-{index}": 9.0 - (0.1 if index == self.first_action // 6 else 0.0)
                for index in range(4)
            },
            "completion_records": completion_records,
            "tasks": tasks,
            "time": float(self.step_count),
            "terminated": done,
            "truncated": False,
            "command_submitted": True,
            "command_id": f"cmd-{self.first_action}-{self.step_count}",
            "active_continuations": [],
            "feedback_log": [feedback],
            "communication_delta": [common, command],
            "new_events": [],
        }
        if self.missing_label:
            info.pop("completion_records")
        return observation(self.step_count, (24,)), 0.0, done, info


def selector_factory():
    return ClassicalSelector("hungarian", (0.8, 0.2))


def request():
    return json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ledgers = []

    def tearDown(self):
        for ledger in self.ledgers:
            try:
                ledger.close()
            except Exception:
                pass
        self.temporary.cleanup()

    def ledger(self):
        ledger = Ledger(self.root / "budget.sqlite3", request())
        self.ledgers.append(ledger)
        return ledger

    def run_unit(self, env_factory):
        output = self.root / "evidence"
        output.mkdir()
        ledger = self.ledger()
        created = []

        def runtime(_scenario, key):
            env = env_factory(key)
            created.append(env)
            return env

        result = execute_unit(
            parent="validation-0000",
            repeat=0,
            scenario={},
            stage="remaining_fixed_units",
            ledger=ledger,
            output=output,
            runtime_factory=runtime,
            adapter_factory=PublicDecisionAdapter,
            selector_factory=selector_factory,
            initial_fleet_energy=36.0,
        )
        return result, ledger, created, output

    def test_real_public_interface_fake_environment_end_to_end(self):
        (selection, summaries, pairing), ledger, created, _output = self.run_unit(
            lambda key: FakeEnv(key)
        )
        self.assertEqual(selection["status"], "OPPORTUNITY")
        self.assertEqual(selection["branch_actions"], [0, 6, 24])
        self.assertEqual(len(summaries), 3)
        self.assertGreater(pairing["common_call_keys"], 0)
        self.assertGreater(pairing["action_specific_call_keys"], 0)
        self.assertEqual(created[0].step_count, 0, "source snapshot was mutated")
        snapshot = ledger.assert_settled()
        self.assertEqual(snapshot["reserved"]["resets"], 1)
        self.assertEqual(snapshot["reserved"]["environment_steps"], 6)
        self.assertEqual(snapshot["reserved"]["branches"], 3)
        self.assertEqual(snapshot["reserved"]["public_rule_decisions"], 4)

    def test_legal_noop_is_branch_25_not_26(self):
        public = observation(0, tuple(range(25)))

        class Memory:
            @staticmethod
            def candidates(_obs):
                return tuple(range(25))

        scan = _candidate_scan(public, Memory())
        self.assertEqual(len(scan["nonnoop"]), 24)
        self.assertEqual(len(scan["candidates"]), 25)
        self.assertEqual(scan["candidates"][-1], 24)

    def test_rejected_action_is_retained(self):
        (selection, summaries, _pairing), _ledger, _created, _output = self.run_unit(
            lambda key: FakeEnv(key, rejected=True)
        )
        by_action = {row["identity"]["action"]: row for row in summaries}
        self.assertIn(6, selection["branch_actions"])
        self.assertEqual(by_action[6]["first_acceptance_class"], "executor_rejected")

    def test_missing_label_stops_without_filling_zero(self):
        with self.assertRaisesRegex(Exception, "required info fields"):
            self.run_unit(lambda key: FakeEnv(key, missing_label=True))

    def test_environment_exception_is_unknown_and_not_retried(self):
        output = self.root / "unknown"
        output.mkdir()
        ledger = self.ledger()
        calls = []

        def runtime(_scenario, key):
            env = FakeEnv(key, opportunity=False, fail_step=True)
            calls.append(env)
            return env

        with self.assertRaisesRegex(RuntimeError, "injected step failure"):
            execute_unit(
                parent="validation-0000",
                repeat=0,
                scenario={},
                stage="remaining_fixed_units",
                ledger=ledger,
                output=output,
                runtime_factory=runtime,
                adapter_factory=PublicDecisionAdapter,
                selector_factory=selector_factory,
                initial_fleet_energy=36.0,
            )
        self.assertEqual(calls[0].step_count, 0)
        self.assertEqual(ledger.snapshot()["call_status_counts"].get("unknown"), 1)

    def test_message_pairing_is_branch_order_invariant(self):
        (_selection, summaries, pairing), _ledger, _created, _output = self.run_unit(
            lambda key: FakeEnv(key)
        )
        reversed_pairing = verify_primitive_pairing(
            "validation-0000", 0, list(reversed(summaries))
        )
        self.assertEqual(
            pairing["common_call_keys"],
            reversed_pairing["common_call_keys"],
        )
        self.assertEqual(reversed_pairing["common_result_mismatches"], 0)

    def test_matrix_skips_reused_unit_and_executes_exactly_23(self):
        tape_rows = [
            {
                "parent": parent,
                "condition": "W1",
                "scenario": {"communication": {"name": "W1-light"}},
            }
            for parent in PARENTS
        ]
        output = self.root / "matrix"
        constructed = []

        def runtime(_scenario, key):
            constructed.append(key)
            return FakeEnv(key)

        status = execute_matrix(
            tape_rows=tape_rows,
            output=output,
            request=request(),
            runtime_factory=runtime,
            adapter_factory=PublicDecisionAdapter,
            selector_factory=selector_factory,
            initial_fleet_energy=36.0,
        )
        self.assertEqual(len(constructed), 23)
        self.assertEqual(len(set(constructed)), 23)
        self.assertNotIn(
            "w1-action-consequence-oracle-ceiling-v1|validation-0000|repeat-0",
            constructed,
        )
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["reused_units"], 1)
        self.assertEqual(status["new_units"], 23)
        self.assertFalse(status["first_unit_reexecuted"])
        analysis = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(len(analysis["units"]), 24)
        self.assertEqual(analysis["matrix"]["reused_units"], 1)
        self.assertEqual(analysis["matrix"]["new_units"], 23)
        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "complete")

    def test_reuse_failure_stops_before_environment_construction(self):
        tape_rows = [
            {
                "parent": parent,
                "condition": "W1",
                "scenario": {"communication": {"name": "W1-light"}},
            }
            for parent in PARENTS
        ]
        bad_reuse = self.root / "bad-reuse"
        bad_reuse.mkdir()
        (bad_reuse / "branch-summaries.jsonl").write_text("{}\n", encoding="utf-8")
        constructed = []

        def runtime(_scenario, key):
            constructed.append(key)
            return FakeEnv(key)

        with self.assertRaisesRegex(Exception, "hash mismatch"):
            execute_matrix(
                tape_rows=tape_rows,
                output=self.root / "reuse-stop",
                request=request(),
                runtime_factory=runtime,
                adapter_factory=PublicDecisionAdapter,
                selector_factory=selector_factory,
                initial_fleet_energy=36.0,
                reuse_dir=bad_reuse,
            )
        self.assertEqual(constructed, [])
        status = json.loads(
            (self.root / "reuse-stop" / "status.json").read_text(encoding="utf-8")
        )
        self.assertEqual(status["status"], "technical_stop_no_retry")
        self.assertEqual(status["ledger"]["call_status_counts"], {})

    def test_no_opportunity_units_remain_in_joint_denominator(self):
        tape_rows = [
            {
                "parent": parent,
                "condition": "W1",
                "scenario": {"communication": {"name": "W1-light"}},
            }
            for parent in PARENTS
        ]
        output = self.root / "single-gate"
        constructed = []

        def runtime(_scenario, key):
            constructed.append(key)
            return FakeEnv(key, opportunity=False)

        status = execute_matrix(
            tape_rows=tape_rows,
            output=output,
            request=request(),
            runtime_factory=runtime,
            adapter_factory=PublicDecisionAdapter,
            selector_factory=selector_factory,
            initial_fleet_energy=36.0,
        )
        self.assertEqual(len(constructed), 23)
        self.assertEqual(status["status"], "complete")
        self.assertTrue(status["oracle_materiality_evaluated"])
        analysis = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(len(analysis["units"]), 24)
        self.assertEqual(
            sum(row["status"] == "NO_OPPORTUNITY" for row in analysis["units"]),
            23,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
