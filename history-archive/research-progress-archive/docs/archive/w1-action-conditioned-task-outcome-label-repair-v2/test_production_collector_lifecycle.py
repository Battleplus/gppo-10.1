import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
RUNTIME_ROOT = ROOT / "research-plans" / "w1-light-repaired-fair-rerun-infra-repair-v1"
LIFECYCLE_SOURCE = ROOT / "runs" / "w1-light-repaired-fair-rerun-v2-nativefs-once" / "native" / "gppo_world"
sys.path.insert(0, str(RUNTIME_ROOT))
sys.path.insert(0, str(PACKAGE))

from classical_baselines import ClassicalSelector, PublicDecisionAdapter  # noqa: E402
from production_data import ProductionDataCollector  # noqa: E402

import runpy  # noqa: E402

LIFECYCLE = runpy.run_path(str(LIFECYCLE_SOURCE / "task_lifecycle.py"))
TaskLifecycle = LIFECYCLE["TaskLifecycle"]
TaskState = LIFECYCLE["TaskState"]


class FakeLedger:
    def __init__(self):
        self.calls = []

    def call(self, name, amounts, _environment, fn, *args, **kwargs):
        self.calls.append((name, dict(amounts)))
        return fn(*args, **kwargs)


class FakeEnvironment:
    def __init__(self, config, scenario, *, exogenous_key):
        self.config = config
        self.profile = dict(scenario["profile"])
        self.exogenous_key = exogenous_key
        self.clock = SimpleNamespace(time=0.0, tasks={})
        deadline = float(self.profile.get("deadline", 12.0))
        service = float(self.profile.get("required_service", 2.0))
        for index in range(6):
            task = TaskLifecycle(
                f"task-{index}", 0.0, deadline, service,
                completion_mode=config.task_completion_mode,
            )
            task.advance(0.0)
            self.clock.tasks[task.task_id] = task
        self.first_action = None
        self.first_status = None
        self.active_action = None
        self.command_serial = 0
        self.execution = SimpleNamespace(commands={}, task_tokens={})
        self.view = SimpleNamespace(
            _uavs=SimpleNamespace(_latest={}),
            _task_values=SimpleNamespace(_latest={}),
        )

    def _bind_command(self, command_id, action):
        task_id = f"task-{int(action) % 6}"
        uav_id = f"uav-{int(action) // 6}"
        token = len(self.execution.commands) + 1
        self.execution.commands[command_id] = SimpleNamespace(
            command_id=command_id, task_id=task_id, uav_id=uav_id, token=token,
        )
        self.execution.task_tokens[task_id] = token
        return task_id, uav_id

    @staticmethod
    def _field_row(values, now):
        row = []
        for value in values:
            row.extend((float(value), 1.0, 1.0, 0.0))
        return row

    def _observation(self):
        now = self.clock.time
        uavs = []
        for index in range(4):
            assigned = any(task.assigned_uav == f"uav-{index}" for task in self.clock.tasks.values())
            uavs.append(self._field_row((float(index), 0.0, 10.0, 1.0, 1.0, 0.0 if assigned else 1.0), now))
        task_rows = []
        for index, task in enumerate(self.clock.tasks.values()):
            public_pending = task.state == TaskState.PENDING
            if index == 0 and self.profile.get("hide_preexisting_execution") and self.first_action is None:
                public_pending = True
            task_rows.append(self._field_row((
                2.0, 0.0, task.deadline, max(0.0, task.required_service - task.service),
                1.0, 1.0 if public_pending else 0.0, 0.0, 0.0,
            ), now))
        mask = [False] * 25
        task_zero = self.clock.tasks["task-0"]
        public_task_zero_pending = not self.profile.get("no_legal_task") and (task_zero.state == TaskState.PENDING or (
            self.profile.get("hide_preexisting_execution") and self.first_action is None
        ))
        if public_task_zero_pending:
            for uav_index in range(4):
                mask[uav_index * 6] = True
        mask[24] = True
        continuation = [] if self.active_action is None else [self.active_action]
        flat = np.zeros(128, dtype=np.float32)
        flat[0] = now
        return {
            "flat": flat,
            "graph": {},
            "uavs": uavs,
            "tasks": task_rows,
            "mask": mask,
            "time": now,
            "version": int(now),
            "public_entity_ids": {
                "uavs": [f"uav-{i}" for i in range(4)],
                "tasks": [f"task-{i}" for i in range(6)],
            },
            "trigger_flags": {},
            "event_signal": 0.0,
            "continuation_actions": continuation,
        }

    def reset(self):
        return self._observation()

    def _advance_one(self, action, *, continuation):
        old_time = self.clock.time
        task = self.clock.tasks["task-0"]
        command_id = None
        feedback = "noop" if action == 24 else "accepted"

        if old_time < 4.0:
            if old_time == 3.0 and self.profile.get("preexisting_execution"):
                task.assign("uav-0", old_time)
                task.state = TaskState.SERVING
                task.service = float(self.profile.get("preexisting_service", 1.0))
                self._bind_command("cmd-existing", 0)
            new_time = old_time + 1.0
            if task.state in (TaskState.ASSIGNED, TaskState.SERVING):
                task.provide_service(task.assigned_uav, old_time, new_time, 1.0)
            else:
                task.advance(new_time)
            for other_id, other in self.clock.tasks.items():
                if other_id != "task-0":
                    other.advance(new_time)
            self.clock.time = new_time
            return self._observation(), 0.0, False, {
                "time": new_time, "feedback": "noop", "command_id": None,
                "terminated": False, "truncated": False,
                "counts": {"completed": 0, "expired": 0},
            }

        if not continuation:
            self.first_action = int(action)
            command_id = "cmd-first"
            if action == 24:
                self.first_status = "noop"
                feedback = "noop"
            else:
                self.first_status = str(self.profile.get("first_status", "accepted"))
                feedback = self.first_status
                if feedback in {"accepted", "accepted_ack_lost"}:
                    task_id, uav_id = self._bind_command(command_id, action)
                    if task.state == TaskState.PENDING:
                        task.assign(uav_id, old_time)
                        self.active_action = int(action)
                elif feedback == "reuse_existing":
                    pass
            action_first = True
        else:
            action_first = False
            if self.profile.get("reassign_after_first") and old_time == 5.0 and task.state in (TaskState.ASSIGNED, TaskState.SERVING):
                task.interrupt(old_time)
                self.active_action = None
                self.execution.task_tokens.pop("task-0", None)
            if action != 24 and task.state == TaskState.PENDING:
                self.command_serial += 1
                command_id = f"cmd-hungarian-{self.command_serial}"
                _, uav_id = self._bind_command(command_id, action)
                task.assign(uav_id, old_time)
                self.active_action = int(action)
                feedback = "accepted"

        new_time = old_time + 1.0
        if task.state in (TaskState.ASSIGNED, TaskState.SERVING):
            task.provide_service(task.assigned_uav, old_time, new_time, 1.0)
        else:
            task.advance(new_time)
        for other_id, other in self.clock.tasks.items():
            if other_id != "task-0":
                other.advance(new_time)
        if task.state in (TaskState.COMPLETED, TaskState.EXPIRED):
            self.active_action = None
        self.clock.time = new_time

        early_end = bool(action_first and self.profile.get("early_termination"))
        terminated = all(value.state in (TaskState.COMPLETED, TaskState.EXPIRED)
                         for value in self.clock.tasks.values()) or early_end
        truncated = new_time >= self.config.horizon and not terminated
        info = {
            "time": new_time,
            "feedback": feedback,
            "command_id": command_id,
            "terminated": terminated,
            "truncated": truncated,
            "counts": {"completed": 0, "expired": 0},
        }
        return self._observation(), 0.0, bool(terminated or truncated), info

    def step(self, action):
        return self._advance_one(int(action), continuation=self.first_action is not None)


class FakeGraph:
    def __init__(self):
        self.nodes = {"UAV": [[0.0] * 4 for _ in range(4)], "Task": [[0.0] * 4 for _ in range(6)]}
        self.candidate_features = [[0.0] * 4 for _ in range(24)]


class ProductionCollectorLifecycleTests(unittest.TestCase):
    def collect_window(self, profile):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {"parent": "fixture-parent", "scenario_sha256": "a" * 64,
                        "exogenous_key": "fixture-parent|repeat-0"}
            collector = ProductionDataCollector.__new__(ProductionDataCollector)
            collector.root = root
            collector.output = root
            collector.matrix = {
                "candidate_continuation_id": "hungarian-v1-fixed",
                "preference_configuration": {"task_confirmation": [0.8, 0.2], "task_component_scale": 0.5},
                "splits": {"train": [], "model_selection": [], "prediction_confirmation": [identity]},
                "repeats": {"train": 0, "model_selection": 0, "prediction_confirmation": 1},
            }
            collector.ledger = FakeLedger()
            collector.boundary = SimpleNamespace(environment=object())
            collector.M10Environment = FakeEnvironment
            config_contract = json.loads((PACKAGE / "environment-config-contract.json").read_text(encoding="utf-8"))
            collector.environment_config_contract = config_contract
            collector.M10Config = lambda **values: SimpleNamespace(**values)
            collector.environment_config = None
            collector.scenario_from_dict = lambda value: value
            collector._row_for = lambda _identity, _repeat: {
                "scenario": {"profile": dict(profile)},
                "exogenous_key": "fixture-parent|repeat-0",
                "source_exogenous_key": "fixture-parent|repeat-0",
            }
            collector.graph5_from_observation = lambda _observation: FakeGraph()
            collector.transparent_score = lambda _graph, _action: {"score": 0.0, "energy_cost": 0.0}
            collector.public_adapter_type = PublicDecisionAdapter
            collector.classical_selector_type = ClassicalSelector
            collector.vector_reward = lambda _info, counts, energy, _config: ([0.0, 0.0], None, counts, energy)
            collector.tapes = []

            fake_policy = SimpleNamespace(
                transparent_utility_components=lambda *_args, **_kwargs: (0.0, 0.0),
            )
            with patch.dict(sys.modules, {"production_policy": fake_policy}):
                summary = collector.collect()
            output = json.loads((root / "world-model-windows.jsonl").read_text(encoding="utf-8").splitlines()[0])
            output["_test_environment_record"] = json.loads(
                (root / "environment.json").read_text(encoding="utf-8")
            )
            return summary, output, collector.ledger.calls

    def candidate(self, output, action):
        for index, row in enumerate(output["candidate_branch_audit"]):
            if row["action"] == action:
                label = output["task_outcome_target"][index]
                self.assertEqual(label["candidate_id"], row["candidate_id"])
                self.assertEqual(label["action_id"], action)
                self.assertEqual(label["continuation_id"], output["continuation_id"])
                return {**row, "task_outcome_target": label}
        raise AssertionError(f"candidate action {action} was not persisted")

    def test_real_collector_persists_first_action_and_fixed_hungarian_outcomes(self):
        _summary, output, calls = self.collect_window({"deadline": 6.0, "required_service": 1.0})
        candidate = self.candidate(output, 0)
        label = candidate["task_outcome_target"]
        self.assertEqual(label["scope"], "first_action_plus_fixed_continuation")
        self.assertEqual(label["target_task_id"], "task-0")
        self.assertEqual(label["target_uav_id"], "uav-0")
        self.assertTrue(label["physical_on_time_completion"]["valid"])
        self.assertTrue(label["physical_on_time_completion"]["value"])
        self.assertEqual(label["physical_on_time_completion"]["source"], "TaskLifecycle.completed_at")
        self.assertEqual(label["deadline"], 6.0)
        self.assertEqual(label["first_command"]["status"], "accepted")
        self.assertEqual(label["continuation_id"], "hungarian-v1-fixed")
        self.assertEqual(label["environment_config_sha256"], "c77c9978c0d1d54e73e4e3022649ccfb7a2e80674662bda395cb9f9ce46a9119")
        self.assertFalse(label["host_confirmation"]["valid"])
        self.assertIn("prediction_confirmation.candidate_continuation", {name for name, _ in calls})
        self.assertTrue(output["decision_input"]["frozen_before_candidate_branches"])
        self.assertNotIn("task_lifecycle", output["decision_input"]["observation"])
        self.assertEqual(len(output["task_outcome_target"]), len(output["candidate_branch_audit"]))
        input_hashes = set()
        for branch_row, task_label in zip(output["candidate_branch_audit"], output["task_outcome_target"]):
            self.assertEqual(task_label["candidate_id"], branch_row["candidate_id"])
            self.assertEqual(task_label["action_id"], branch_row["action"])
            self.assertEqual(task_label["continuation_id"], output["continuation_id"])
            input_hashes.add(task_label["public_input_hash"])
        self.assertEqual(len(input_hashes), 1)
        environment_record = output["_test_environment_record"]
        self.assertEqual(environment_record["config_sha256"], output["environment_config_identity"]["config_sha256"])
        self.assertEqual(environment_record["task_completion_mode"], "continuous_service_until_deadline")
        self.assertEqual(environment_record["deadline_basis"], "physical_service")

    def test_no_opportunity_window_is_persisted_without_zero_labels(self):
        summary, output, calls = self.collect_window({"no_legal_task": True})
        self.assertEqual(summary["no_opportunity_windows"], 1)
        self.assertEqual(output["status"], "no_opportunity")
        self.assertNotIn("task_outcome_target", output)
        self.assertFalse(any(name.endswith(".candidate_branch") for name, _ in calls))

    def test_rejected_and_lost_first_commands_keep_later_sequence_completion(self):
        for first_status in ("rejected", "command_lost"):
            with self.subTest(first_status=first_status):
                _summary, output, _calls = self.collect_window({
                    "deadline": 12.0, "required_service": 1.0, "first_status": first_status,
                })
                label = self.candidate(output, 0)["task_outcome_target"]
                self.assertTrue(label["physical_on_time_completion"]["valid"])
                self.assertTrue(label["physical_on_time_completion"]["value"])
                self.assertFalse(label["completion_by_first_command"])
                self.assertEqual(label["first_command"]["status"],
                                 "lost" if first_status == "command_lost" else "rejected")

    def test_hungarian_reassignment_completion_is_not_attributed_to_first_command(self):
        _summary, output, _calls = self.collect_window({
            "deadline": 12.0, "required_service": 3.0, "reassign_after_first": True,
        })
        label = self.candidate(output, 0)["task_outcome_target"]
        self.assertTrue(label["physical_on_time_completion"]["value"])
        self.assertNotEqual(label["completion_execution_identity"]["command_id"],
                            label["first_command"]["command_id"])
        self.assertFalse(label["action_specific_effect_identifiable"])

    def test_private_preexisting_execution_is_audit_only(self):
        _summary, output, _calls = self.collect_window({
            "deadline": 12.0, "required_service": 3.0,
            "preexisting_execution": True, "hide_preexisting_execution": True,
            "preexisting_service": 1.0, "first_status": "rejected",
        })
        label = self.candidate(output, 0)["task_outcome_target"]
        self.assertEqual(label["initial_execution_identity"]["command_id"], "cmd-existing")
        self.assertTrue(label["physical_on_time_completion"]["value"])
        self.assertEqual(label["completion_execution_identity"]["command_id"], "cmd-existing")
        self.assertEqual(label["first_command"]["status"], "rejected")
        self.assertFalse(label["completion_by_first_command"])
        self.assertFalse(label["action_specific_effect_identifiable"])

    def test_expiry_and_early_termination_have_distinct_masks(self):
        _summary, expired_output, _calls = self.collect_window({
            "deadline": 6.0, "required_service": 10.0,
        })
        expired = self.candidate(expired_output, 0)["task_outcome_target"]
        self.assertEqual(expired["task_expired"], {
            "value": True, "valid": True, "source": "TaskLifecycle.state",
            "time": 6.0, "reason": None,
        })
        self.assertFalse(expired["physical_on_time_completion"]["value"])

        _summary, early_output, _calls = self.collect_window({
            "deadline": 12.0, "required_service": 10.0, "early_termination": True,
        })
        early = self.candidate(early_output, 0)["task_outcome_target"]
        self.assertEqual(early["observation_end_status"], "unresolved_at_observation_end")
        self.assertFalse(early["valid_mask"]["physical_on_time_completion"])
        self.assertFalse(early["valid_mask"]["task_expired"])
        self.assertIsNone(early["physical_on_time_completion"]["value"])


if __name__ == "__main__":
    unittest.main()
