"""Deterministic bottom-boundary substitutions for WSL wiring tests."""
from __future__ import annotations


def _row(parent, role, action, decision, target, transparent):
    return {
        "sample_id": f"{decision}:{action}", "parent": parent, "repeat": 0,
        "decision_id": decision, "split_role": role, "action": action,
        "legal_actions": [0, 1], "flat": [0.0] * 770, "history": [0.0] * 32,
        "remaining_utility": target, "current_public_score": 0.0,
        "transparent_history_score": transparent,
        "continuation_id": "hungarian-v1-fixed", "is_true_branch": True,
        "training_eligible": True,
    }


class IntegrationBoundary:
    def __init__(self, mode: str = "pass"):
        self.mode = mode
        self.task_calls = 0

    def environment(self, operation, *args, **kwargs):
        return operation(*args, **kwargs)

    def model(self, operation, *args, **kwargs):
        return operation(*args, **kwargs)

    def optimizer(self, operation, *args, **kwargs):
        return operation(*args, **kwargs)

    def checkpoint(self, operation, *args, **kwargs):
        return operation(*args, **kwargs)

    class _Observer:
        def __init__(self):
            self.records = []

        def start_branch(self, branch):
            self.branch = branch
            return len(self.records)

        def records_since(self, _marker):
            return [{
                "branch": self.branch,
                "status": "returned",
                "method": "probe",
                "arguments": {"identity": "shared-task-call", "parameter": 1},
                "result": {"fate": "same"},
                "ordinal": 0,
            }]

    class _Memory:
        def candidates(self, _observation):
            return (0, 1)

        def get(self, kind, entity, field):
            return {"value": 1.0, "valid": True, "age": 0.0}

    class _Adapter:
        def __init__(self):
            self.memory = IntegrationBoundary._Memory()

        def prepare(self, observation):
            return observation, [True] * 25

        def commit(self, action):
            return action

        def decide(self, observation, selector):
            action, diagnostic = selector.choose(observation, self.memory)
            return {"action": action, "diagnostic": diagnostic}

    class _Selector:
        def choose(self, _observation, _memory):
            return 0, {"method": "hungarian"}

    class _Environment:
        def __init__(self):
            self.communication = object()
            self.step_count = 0

        @staticmethod
        def observation(step):
            return {
                "time": float(step),
                "uavs": [[1.0, 1.0, 30.0] * 4 for _ in range(4)],
                "tasks": [[2.0, 2.0, 100.0] * 6 for _ in range(6)],
                "public_entity_ids": {"uavs": [0, 1, 2, 3], "tasks": [0, 1, 2, 3, 4, 5]},
            }

        def reset(self):
            self.step_count = 0
            return self.observation(0)

        def step(self, _action):
            self.step_count += 1
            done = self.step_count >= 5
            return self.observation(self.step_count), 0.0, done, {}

    def configure_task_runtime(self, legacy):
        legacy.bottom_environment_factory = lambda _parent, _key: self._Environment()
        legacy.runtime["adapter"] = self._Adapter
        legacy.runtime["recording_communication"] = lambda communication: communication
        observer = self._Observer()
        legacy.runtime["require_observer"] = lambda _communication: observer
        legacy.runtime["selector"] = lambda _method, _preference: self._Selector()

        def label_episode(rows, *, identity, **_kwargs):
            arm = identity["arm"]
            utility = {"hungarian": 0.0, "transparent_one_shot": 0.1,
                       "A_one_shot": 0.2, "B_one_shot": 0.3}[arm]
            return {"identity": identity, "discounted_utility": utility,
                    "rows": len(rows)}

        legacy.runtime["label_episode"] = label_episode

    def zero_step(self):
        return {"pass": True, "runtime_modules_loaded": True,
                "environment_constructed": False, "model_constructed": False}

    def collect_records(self, _output, _matrix):
        rows = []
        for role, parents in (("train", ["train-test"]), ("model_selection", ["select-test"]),
                              ("prediction_evaluation", [f"eval-{i}" for i in range(8)])):
            for parent in parents:
                decision = f"{role}-{parent}-decision"
                rows.extend([_row(parent, role, 0, decision, 0.0, 1.0),
                             _row(parent, role, 1, decision, 1.0, 0.0)])
        return rows

    def train_variant(self, variant, seed, _train, _selection):
        return {"variant": variant, "seed": seed}, {"variant": variant, "seed": seed}

    def predict_variant(self, variant, row, _models):
        if self.mode == "gate_fail" or variant == "A":
            values = [1.0, 0.0]
        else:
            values = [0.0, 1.0]
        return [values[row.action]] * 3
