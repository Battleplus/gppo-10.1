"""Regression tests for the decision-time public-input boundary repair.

These tests use the production collector, adapter, feature builder, labeler,
validator, persistence code, and ledger with a controlled fake environment.
They never construct the real environment or a model.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path

from history_regression_harness import ControlledFakeEnvironment, build_backend
from learning_schema import LearningContractError, load_records
from runtime_backend import (
    HORIZON,
    INITIAL_FLEET_ENERGY,
    PREFERENCE,
    freeze_decision_input,
)
from transparent_baselines import current_public_scores, qualifies


ROOT = Path(__file__).resolve().parent


def digest(value) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


class HistoryBoundaryRepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="w1-history-boundary-")
        self.addCleanup(self.temporary.cleanup)
        self.counter = 0

    def backend(self, mode: str = "normal"):
        self.counter += 1
        output = Path(self.temporary.name) / f"run-{self.counter}"
        backend = build_backend(ROOT, output, mode=mode)
        self.addCleanup(backend.close)
        return backend

    def capture_decision(self, backend):
        parent = "fake-train-00"
        repeat = 0
        split_role = "train"
        key = f"w1-action-outcome-learning-v1|{split_role}|{parent}|repeat-{repeat}"
        env = backend._new_environment(parent, key)
        observation = backend.account_call(
            "environment.reset", {"resets_upper": 1}, env.reset
        )
        adapter = backend.runtime["adapter"]()
        counts = {"completed": 0, "expired": 0}
        energy = INITIAL_FLEET_ENERGY
        for decision_step in range(HORIZON):
            public_obs, _mask = adapter.prepare(observation)
            backend.account_call("candidate.scan", {"candidate_scans": 1}, lambda: None)
            candidates = tuple(int(value) for value in adapter.memory.candidates(public_obs))
            qualified, transparent_scores = qualifies(
                public_obs, adapter.memory, minimum_step=4, decision_step=decision_step
            )
            if qualified:
                decision_input = freeze_decision_input(
                    adapter,
                    public_obs,
                    candidates,
                    current_public_scores(public_obs, candidates),
                    transparent_scores,
                )
                capture = (env, adapter, decision_input, counts, energy, decision_step, key)
                return capture, {
                    "parent": parent,
                    "repeat": repeat,
                    "split_role": split_role,
                    "decision_id": f"{split_role}:{parent}:r{repeat}:s{decision_step}",
                }
            selector = backend.runtime["selector"]("hungarian", PREFERENCE)
            action, _diagnostic = backend.account_call(
                "hungarian.prefix",
                {"public_rule_decisions": 1},
                selector.choose,
                public_obs,
                adapter.memory,
            )
            adapter.commit(int(action))
            observation, _reward, done, info = backend._step(env, int(action))
            counts = {
                "completed": int(info["counts"]["completed"]),
                "expired": int(info["counts"]["expired"]),
            }
            energy = sum(float(value) for value in info["energy"].values())
            self.assertFalse(done)
        self.fail("controlled environment produced no qualifying decision")

    @staticmethod
    def run_branch(backend, capture, identity, action):
        return backend._run_label_branch(capture, action=action, **identity)

    def test_continuation_cannot_change_frozen_features_or_parent(self) -> None:
        backend = self.backend()
        capture, identity = self.capture_decision(backend)
        parent_digest = digest(capture)
        input_digest = capture[2].sha256
        records = []
        summaries = []
        for action in capture[2].legal_actions:
            record, summary = self.run_branch(backend, capture, identity, action)
            records.append(record)
            summaries.append(summary)
        self.assertEqual(digest(capture), parent_digest)
        self.assertEqual(capture[2].sha256, input_digest)
        self.assertEqual({row["decision_input_sha256"] for row in summaries}, {input_digest})
        self.assertEqual(len({tuple(row["flat"]) for row in records}), 1)
        self.assertEqual(len({tuple(row["history"]) for row in records}), 1)
        self.assertGreater(len({row["final_time"] for row in summaries}), 1)

    def test_branch_order_does_not_change_per_action_result(self) -> None:
        backend = self.backend()
        capture, identity = self.capture_decision(backend)
        actions = tuple(capture[2].legal_actions)

        def execute(order):
            result = {}
            for action in order:
                record, summary = self.run_branch(backend, capture, identity, action)
                result[action] = {
                    "record": record,
                    "utility": summary["discounted_utility"],
                    "final_time": summary["final_time"],
                    "acceptance": summary["first_acceptance_class"],
                    "input": summary["decision_input_sha256"],
                }
            return result

        self.assertEqual(execute(actions), execute(reversed(actions)))

    def test_candidate_copies_do_not_share_nested_mutables(self) -> None:
        backend = self.backend()
        capture, _identity = self.capture_decision(backend)
        left = copy.deepcopy(capture)
        right = copy.deepcopy(capture)
        self.assertIsNot(left[0], right[0])
        self.assertIsNot(left[1].memory.history, right[1].memory.history)
        self.assertIsNot(left[1].memory.history[-1], right[1].memory.history[-1])
        self.assertIsNot(left[1].memory.history[-1]["uavs"], right[1].memory.history[-1]["uavs"])
        self.assertIsNot(left[1].memory.history[-1]["uavs"][0], right[1].memory.history[-1]["uavs"][0])
        left[1].memory.history[-1]["uavs"][0][0] = 999.0
        self.assertNotEqual(
            left[1].memory.history[-1]["uavs"][0][0],
            right[1].memory.history[-1]["uavs"][0][0],
        )
        self.assertNotEqual(
            left[1].memory.history[-1]["uavs"][0][0],
            capture[1].memory.history[-1]["uavs"][0][0],
        )
        self.assertIsInstance(capture[2].flat, tuple)
        self.assertIsInstance(capture[2].history_features, tuple)

    def test_future_public_observation_is_still_rejected(self) -> None:
        backend = self.backend()
        adapter = backend.runtime["adapter"]()
        fake = ControlledFakeEnvironment(backend.runtime["recording_communication"])
        fake.time = 4
        current, _mask = adapter.prepare(fake._observation())
        fake.time = 5
        injected_future = fake._observation()
        adapter.memory.history.insert(-1, injected_future)
        candidates = tuple(adapter.memory.candidates(current))
        scores = {action: 0.0 for action in candidates}
        with self.assertRaisesRegex(LearningContractError, "future observation"):
            freeze_decision_input(adapter, current, candidates, scores, scores)

    def test_later_received_old_measurement_does_not_enter_frozen_input(self) -> None:
        backend = self.backend(mode="old_measurement")
        capture, _identity = self.capture_decision(backend)
        env, adapter, frozen, *_rest = copy.deepcopy(capture)
        before = frozen.public_input_json
        adapter.commit(0)
        later_observation, _reward, _done, _info = env.step(0)
        self.assertGreater(float(later_observation["time"]), frozen.decision_time)
        self.assertGreater(float(later_observation["uavs"][0][3]), 0.0)
        adapter.prepare(later_observation)
        self.assertEqual(frozen.public_input_json, before)
        decoded = json.loads(before)
        self.assertTrue(
            all(float(row["time"]) <= frozen.decision_time for row in decoded["public_history"])
        )
        self.assertNotIn(float(later_observation["time"]), [
            float(row["time"]) for row in decoded["public_history"]
        ])

    def test_normal_rejected_unknown_and_no_opportunity_contracts(self) -> None:
        expected = {
            "normal": "accepted",
            "rejected": "executor_rejected",
            "unknown": "unknown",
        }
        for mode, acceptance in expected.items():
            with self.subTest(mode=mode):
                backend = self.backend(mode=mode)
                records, unit = backend._collect_unit("train", f"fake-{mode}", 0)
                self.assertEqual(unit["status"], "LABELED")
                self.assertTrue(records)
                self.assertEqual(
                    unit["explanation_labels"][0]["first_acceptance_class"], acceptance
                )
        backend = self.backend(mode="no_opportunity")
        records, unit = backend._collect_unit("train", "fake-none", 0)
        self.assertEqual(records, [])
        self.assertEqual(unit["status"], "NO_OPPORTUNITY")

    def test_first_label_persists_reloads_and_validates(self) -> None:
        backend = self.backend()
        data_path = backend.collect_fixed_continuation_labels()
        records = load_records(data_path)
        self.assertGreater(len(records), 0)
        records[0].validate()
        self.assertTrue((backend.output / "data-units.jsonl").is_file())
        self.assertTrue(json.loads((backend.output / "data-gate.json").read_text())["pass"])
        settlement = backend.ledger.assert_settled()
        self.assertEqual(settlement["pending_calls"], 0)
        self.assertEqual(settlement["failed_calls"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
