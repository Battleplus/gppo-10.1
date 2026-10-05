"""Controlled fake environment for the production label-collection path."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

from budget_ledger import BudgetLedger
from runtime_backend import AuthorizedRuntimeBackend


def _runtime_components(root: Path) -> dict[str, Any]:
    inputs = json.loads((root / "runtime-inputs.json").read_text(encoding="utf-8"))
    platform = "linux" if os.name == "posix" else "windows"
    oracle_root = Path(inputs["platforms"][platform]["oracle_runner"])
    if str(oracle_root) not in sys.path:
        sys.path.insert(0, str(oracle_root))
    from classical_baselines import ClassicalSelector, PublicDecisionAdapter
    from communication_observer import (
        RecordingCommunication,
        require_observer,
        verify_primitive_pairing,
    )
    from labeling import label_episode

    return {
        "selector": ClassicalSelector,
        "adapter": PublicDecisionAdapter,
        "recording_communication": RecordingCommunication,
        "require_observer": require_observer,
        "verify_pairing": verify_primitive_pairing,
        "label_episode": label_episode,
    }


class FakePrimitiveCommunication:
    @staticmethod
    def _fate(identity: str) -> bool:
        return int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8], 16) % 20 != 0

    def telemetry(self, **kwargs):
        return self._fate(str(kwargs["identity"]))

    def command_delivered(self, **kwargs):
        return self._fate(str(kwargs["identity"]))

    def ack_delivered(self, **kwargs):
        return self._fate(str(kwargs["identity"]))

    def renewal(self, **kwargs):
        return self._fate(str(kwargs["identity"]))


def _row(values: list[float], *, age: float = 0.0) -> list[float]:
    result: list[float] = []
    for value in values:
        result.extend((float(value), 1.0, 1.0, float(age)))
    return result


class ControlledFakeEnvironment:
    """Public-contract fake; it never imports or constructs the real environment."""

    def __init__(self, recording_communication, *, mode: str = "normal") -> None:
        self.time = 0
        self.mode = mode
        self.branch_action: int | None = None
        self.branch_steps = 0
        self.communication = recording_communication(FakePrimitiveCommunication())

    def __deepcopy__(self, memo):
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        copied.time = self.time
        copied.mode = self.mode
        copied.branch_action = self.branch_action
        copied.branch_steps = self.branch_steps
        copied.communication = copy.deepcopy(self.communication, memo)
        return copied

    def _observation(self) -> dict[str, Any]:
        age = 2.0 if self.mode == "old_measurement" and self.time >= 4 else 0.0
        uavs = [
            _row([float(index), 0.0, 9.0 - 0.01 * self.time, 1.0, 1.0, 1.0], age=age)
            for index in range(4)
        ]
        tasks = [
            _row([2.0 + index, float(index), 12.0 + index, 0.0, 1.0, 0.0, 0.0, float(index)])
            for index in range(6)
        ]
        mask = [False] * 25
        mask[24] = True
        if self.mode != "no_opportunity" and self.time >= 4 and self.branch_action is None:
            mask[0] = True
            mask[1] = True
        flat = [0.0] * 770
        flat[0] = float(self.time)
        return {
            "flat": flat,
            "uavs": uavs,
            "tasks": tasks,
            "mask": mask,
            "time": float(self.time),
            "version": int(self.time),
            "public_entity_ids": {
                "uavs": [f"uav-{index}" for index in range(4)],
                "tasks": [f"task-{index}" for index in range(6)],
            },
            "trigger_flags": [],
            "event_signal": 0.0,
            "continuation_actions": [],
        }

    def reset(self):
        return self._observation()

    def step(self, action: int):
        branch = self.communication.branch is not None
        if branch and self.branch_action is None:
            self.branch_action = int(action)
        if branch:
            self.branch_steps += 1
            self.communication.command_delivered(
                identity=f"command:action-{action:02d}:step-{self.branch_steps}",
                loss_rate=0.05,
            )
        self.time += 1
        if branch:
            self.communication.telemetry(
                identity=f"telemetry:time-{self.time}", loss_rate=0.05
            )
        duration = {0: 2, 1: 3, 24: 1}.get(self.branch_action, 2)
        done = bool(branch and self.branch_steps >= duration)
        completed = bool(done and self.mode not in {"rejected", "unknown"} and self.branch_action != 24)
        completed_task = f"task-{int(self.branch_action or 0) % 6}"
        tasks = {f"task-{index}": "pending" for index in range(6)}
        completion_records: dict[str, dict[str, Any]] = {}
        if completed:
            tasks[completed_task] = "completed"
            completion_records[completed_task] = {
                "physical_arrival_time": float(self.time),
                "host_confirmation_time": float(self.time),
                "physical_arrival_before_deadline": True,
                "host_confirmation_before_deadline": True,
                "uav_id": "uav-0",
            }
        if not branch or action == 24:
            feedback = "noop"
        elif self.mode == "rejected" and self.branch_steps == 1:
            feedback = "executor_rejected"
        elif self.mode == "unknown" and self.branch_steps == 1:
            feedback = "unclassified_feedback"
        else:
            feedback = "accepted"
        info = {
            "feedback": feedback,
            "counts": {"completed": int(completed), "expired": 0},
            "energy": {f"uav-{index}": 9.0 - 0.01 * self.time for index in range(4)},
            "completion_records": completion_records,
            "tasks": tasks,
            "time": float(self.time),
            "terminated": done,
            "truncated": False,
            "command_submitted": action != 24,
            "command_id": None if action == 24 else f"command-{action}-{self.time}",
            "active_continuations": [],
            "feedback_log": [],
            "communication_delta": [],
            "new_events": [],
        }
        return self._observation(), 0.0, done, info


def compact_matrix() -> dict[str, Any]:
    def group(prefix: str, count: int) -> list[dict[str, str]]:
        return [{"parent": f"{prefix}-{index:02d}"} for index in range(count)]

    return {
        "groups": {
            "train": group("fake-train", 18),
            "model_selection": group("fake-select", 6),
            "prediction_evaluation": group("fake-eval", 6),
            "task_evaluation": [],
        },
        "data_repeats": {"train": 1, "model_selection": 1, "prediction_evaluation": 1},
    }


def build_backend(root: Path, output: Path, *, mode: str = "normal", matrix=None):
    output.mkdir(parents=True, exist_ok=False)
    request = json.loads((root / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
    backend = AuthorizedRuntimeBackend(root, output, request, matrix or compact_matrix())
    backend.ledger = BudgetLedger(output / "budget.sqlite3", request)
    backend.ledger.select("label_collection")
    backend.runtime = _runtime_components(root)

    def new_environment(self, _parent: str, _key: str):
        return ControlledFakeEnvironment(
            self.runtime["recording_communication"], mode=mode
        )

    backend._new_environment = types.MethodType(new_environment, backend)
    return backend
