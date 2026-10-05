from __future__ import annotations

import json
import math
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

import torch

import legacy_runtime_backend as runtime_module
from legacy_runtime_backend import AuthorizedRuntimeBackend


ROOT = Path(__file__).resolve().parent


class DurationClock:
    def __init__(self, durations):
        values = []
        start = 0.0
        for duration in durations:
            values.extend((start, start + duration))
            start += duration + 1.0
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class FakeLedger:
    def __init__(self):
        self.calls = []

    def call(self, name, amounts, function, *args, **kwargs):
        self.calls.append((name, dict(amounts)))
        return function(*args, **kwargs)

    def assert_settled(self):
        return None


class FakeModel:
    def __call__(self, features):
        rows = features.tolist()
        return torch.tensor(
            [[float(max(range(25), key=lambda index: row[-25 + index]))] for row in rows],
            dtype=torch.float32,
        )


class FakeMemory:
    def __init__(self):
        self.history = []

    def candidates(self, _observation):
        return (0, 1)


class FakeAdapter:
    def __init__(self):
        self.memory = FakeMemory()

    def prepare(self, observation):
        self.memory.history.append(observation)
        return observation, [True] * 25

    def commit(self, action):
        return action


class FakeSelector:
    def choose(self, _observation, _memory):
        return 0, {"method": "hungarian"}


class FakeEnvironment:
    def __init__(self):
        self.communication = object()

    @staticmethod
    def observation():
        return {"time": 4.0, "flat": [0.0] * 770}

    def reset(self):
        return self.observation()

    def step(self, _action):
        return self.observation(), 0.0, True, {}


class FakeObserver:
    def start_branch(self, branch):
        self.branch = branch
        return 0

    def records_since(self, _marker):
        return [{
            "branch": self.branch,
            "status": "returned",
            "method": "probe",
            "arguments": {"identity": "shared", "parameter": 1},
            "result": {"fate": "same"},
            "ordinal": 0,
        }]


class TaskCostAttributionTests(unittest.TestCase):
    parents = ("task-parent-a", "task-parent-b")
    repeats = 2

    def run_case(self, *, a_cpu, b_cpu, a_wall, b_wall, opportunity=True):
        units = len(self.parents) * self.repeats
        cpu_clock = DurationClock([value for _ in range(units) for value in (a_cpu, b_cpu)])
        wall_clock = DurationClock([value for _ in range(units) for value in (a_wall, b_wall)])
        output = Path(tempfile.mkdtemp(prefix="task-cost-attribution-"))
        matrix = {
            "groups": {"task_evaluation": [{"parent": parent} for parent in self.parents]},
            "task_repeats": self.repeats,
            "task_methods": ["hungarian", "transparent_one_shot", "A_one_shot", "B_one_shot"],
        }
        backend = AuthorizedRuntimeBackend(
            ROOT,
            output,
            {},
            matrix,
            wall_clock=wall_clock,
            cpu_clock=cpu_clock,
        )
        backend.ledger = FakeLedger()
        backend.bottom_environment_factory = lambda _parent, _key: FakeEnvironment()
        observer = FakeObserver()
        backend.runtime = {
            "adapter": FakeAdapter,
            "require_observer": lambda _communication: observer,
            "selector": lambda _method, _preference: FakeSelector(),
            "label_episode": self.label_episode,
        }
        models = {"A": [FakeModel()] * 3, "B": [FakeModel()] * 3}
        qualified = (lambda *_args, **_kwargs: (True, {0: 0.0, 1: 1.0})) if opportunity else (
            lambda *_args, **_kwargs: (False, {0: 0.0, 1: 1.0})
        )
        with mock.patch.object(runtime_module, "qualifies", side_effect=qualified):
            result = backend.run_task_comparison(models)
        return result, backend.ledger, output

    @staticmethod
    def label_episode(rows, *, identity, **_kwargs):
        utility = {
            "hungarian": 0.0,
            "transparent_one_shot": 0.0,
            "A_one_shot": 0.02,
            "B_one_shot": 0.02,
        }[identity["arm"]]
        return {
            "identity": identity,
            "discounted_utility": utility,
            "actions": [int(row["action"]) for row in rows],
        }

    def test_fast_a_cannot_hide_slow_b(self):
        result, ledger, output = self.run_case(
            a_cpu=0.001,
            b_cpu=0.015,
            a_wall=0.002,
            b_wall=0.060,
        )
        costs = result["decision_cost"]["by_method"]
        self.assertTrue(costs["A_one_shot"]["cpu_mean_at_most_0.010_seconds"])
        self.assertFalse(costs["B_one_shot"]["cpu_mean_at_most_0.010_seconds"])
        self.assertFalse(costs["B_one_shot"]["wall_p95_at_most_0.050_seconds"])
        self.assertFalse(result["frozen_gate"]["pass"])
        self.assertEqual(costs["B_one_shot"]["sample_count"], 4)
        self.assertEqual(sum(name == "model.forward.task" for name, _ in ledger.calls), 24)
        self.assertTrue((output / "task-decision-costs.jsonl").is_file())

    def test_slow_a_is_not_charged_to_fast_b(self):
        result, ledger, _output = self.run_case(
            a_cpu=0.020,
            b_cpu=0.002,
            a_wall=0.080,
            b_wall=0.004,
        )
        costs = result["decision_cost"]["by_method"]
        self.assertFalse(costs["A_one_shot"]["cpu_mean_at_most_0.010_seconds"])
        self.assertFalse(costs["A_one_shot"]["wall_p95_at_most_0.050_seconds"])
        self.assertTrue(costs["B_one_shot"]["cpu_mean_at_most_0.010_seconds"])
        self.assertTrue(costs["B_one_shot"]["wall_p95_at_most_0.050_seconds"])
        self.assertTrue(result["frozen_gate"]["pass"])
        self.assertEqual(sum(name == "model.forward.task" for name, _ in ledger.calls), 24)

    def test_records_recompute_by_parent_and_repeat_without_changing_results(self):
        first, _ledger, output = self.run_case(
            a_cpu=0.003,
            b_cpu=0.004,
            a_wall=0.005,
            b_wall=0.006,
        )
        second, _ledger, _output = self.run_case(
            a_cpu=0.008,
            b_cpu=0.009,
            a_wall=0.010,
            b_wall=0.011,
        )
        records = first["decision_cost"]["records"]
        identities = {
            (row["method"], row["parent"], row["repeat"], row["decision_step"])
            for row in records
        }
        self.assertEqual(len(records), 8)
        self.assertEqual(len(identities), 8)
        self.assertTrue(all(row["candidate_count"] == 2 for row in records))
        grouped = defaultdict(list)
        for row in records:
            grouped[(row["method"], row["parent"], row["repeat"])].append(row)
        self.assertEqual(len(grouped), 8)
        self.assertTrue(all(len(rows) == 1 for rows in grouped.values()))
        for method in ("A_one_shot", "B_one_shot"):
            method_rows = [row for row in records if row["method"] == method]
            cpu_mean = sum(row["cpu_seconds"] for row in method_rows) / len(method_rows)
            walls = sorted(row["wall_seconds"] for row in method_rows)
            wall_p95 = walls[min(len(walls) - 1, math.ceil(0.95 * len(walls)) - 1)]
            summary = first["decision_cost"]["by_method"][method]
            self.assertAlmostEqual(cpu_mean, summary["cpu_mean_seconds"])
            self.assertAlmostEqual(wall_p95, summary["wall_p95_seconds"])
        self.assertEqual(
            [(row["identity"], row["discounted_utility"], row["actions"]) for row in first["summaries"]],
            [(row["identity"], row["discounted_utility"], row["actions"]) for row in second["summaries"]],
        )
        persisted = [json.loads(line) for line in (output / "task-decision-costs.jsonl").read_text().splitlines()]
        self.assertEqual(persisted, records)

    def test_no_opportunity_is_not_evaluated_and_does_not_forward(self):
        result, ledger, _output = self.run_case(
            a_cpu=0.001,
            b_cpu=0.001,
            a_wall=0.001,
            b_wall=0.001,
            opportunity=False,
        )
        costs = result["decision_cost"]["by_method"]
        for method in ("A_one_shot", "B_one_shot"):
            self.assertEqual(costs[method]["status"], "not_evaluated")
            self.assertFalse(costs[method]["evaluated"])
            self.assertEqual(costs[method]["sample_count"], 0)
            self.assertIsNone(costs[method]["cpu_mean_at_most_0.010_seconds"])
            self.assertIsNone(costs[method]["wall_p95_at_most_0.050_seconds"])
        self.assertFalse(result["frozen_gate"]["pass"])
        self.assertEqual(sum(name == "model.forward.task" for name, _ in ledger.calls), 0)


if __name__ == "__main__":
    unittest.main()
