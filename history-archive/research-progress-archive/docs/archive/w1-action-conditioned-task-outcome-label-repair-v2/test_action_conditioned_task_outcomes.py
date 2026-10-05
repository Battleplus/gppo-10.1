"""Production collector label adapter exercised with lifecycle-only fixtures."""
import ast
import copy
import runpy
from pathlib import Path
import sys
import unittest

PACKAGE = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE.parents[1] / "runs" / "w1-light-repaired-fair-rerun-v2-nativefs-once" / "native" / "gppo_world"
sys.path.insert(0, str(PACKAGE))

from production_data import ProductionDataCollector  # noqa: E402
from task_outcome_contract import EXPECTED_CONTINUATION, TaskOutcomeContractError  # noqa: E402

LIFECYCLE = runpy.run_path(str(SOURCE_ROOT / "task_lifecycle.py"))
TaskLifecycle = LIFECYCLE["TaskLifecycle"]
TaskState = LIFECYCLE["TaskState"]


def lifecycle(task_state="pending", *, deadline=12.0, required_service=2.0,
              assigned_uav=None, completed_at=None, service=0.0, last_time=None):
    task = TaskLifecycle("task-0", 0.0, deadline, required_service)
    if last_time is not None:
        task.advance(last_time)
    if task_state in {"assigned", "serving"}:
        task.assign(assigned_uav or "uav-0", task.last_time if task.last_time != float("-inf") else 3.0)
        if task_state == "serving":
            task.state = TaskState.SERVING
            task.service = service
    if task_state == "completed":
        task.completed_at = completed_at
        task.state = TaskState.COMPLETED
    return task


def execution(command_id, task_id="task-0", uav_id="uav-0", token=1):
    return {"command_id": command_id, "task_id": task_id, "uav_id": uav_id, "token": token}


def snapshot(task, identity=None):
    return {
        "task_id": task.task_id, "state": task.state.value, "deadline": task.deadline,
        "completed_at": task.completed_at, "assigned_uav": task.assigned_uav,
        "service": task.service, "execution_identity": identity,
    }


def decision(*, status="accepted", task=None, initial_identity=None, initial_task=None, action=0):
    if task is None:
        task = lifecycle(last_time=4.0)
    return {
        "candidate_id": f"action-{action}", "action_id": action,
        "legal_actions": [0, 1, 24], "target_task_id": None if action == 24 else "task-0",
        "target_uav_id": None if action == 24 else "uav-0",
        "public_input_hash": "a" * 64, "continuation_id": EXPECTED_CONTINUATION,
        "environment_config_sha256": "b" * 64,
        "task_completion_mode": "continuous_service_until_deadline",
        "deadline_basis": "physical_service",
        "completion_notice_available": False,
        "decision_time": 4.0, "horizon_time": 18.0,
        "first_command": {"command_id": "cmd-first", "status": status},
        "initial_task": None if action == 24 else (initial_task or snapshot(task, initial_identity)),
    }


def row(step, time, task, identity=None, *, terminated=False, truncated=False):
    return {
        "step": step, "time": time, "continuation_id": EXPECTED_CONTINUATION,
        "task_lifecycle": snapshot(task, identity),
        "terminated": terminated, "truncated": truncated,
    }


class ActionConditionedTaskOutcomeTests(unittest.TestCase):
    def test_first_command_accepted_and_completes_by_deadline(self):
        task = lifecycle(deadline=6.0, required_service=2.0, last_time=4.0)
        frozen_decision = decision(task=task, status="accepted")
        task.assign("uav-0", 4.0)
        task.provide_service("uav-0", 4.0, 6.0, 1.0)
        result = ProductionDataCollector.task_outcome_target(
            frozen_decision,
            [row(0, 6.0, task, execution("cmd-first"), terminated=True)],
        )
        self.assertEqual(result["observation_end_status"], "completed_on_time")
        self.assertEqual(result["physical_on_time_completion"], {
            "value": True, "valid": True, "source": "TaskLifecycle.completed_at",
            "time": 6.0, "reason": None,
        })
        self.assertTrue(result["completion_by_first_command"])
        self.assertFalse(result["task_expired"]["value"])
        self.assertFalse(result["host_confirmation"]["valid"])

    def test_rejected_or_lost_first_command_can_have_later_sequence_outcome(self):
        for status in ("rejected", "lost"):
            with self.subTest(status=status):
                task = lifecycle(last_time=4.0)
                task.assign("uav-1", 5.0)
                task.provide_service("uav-1", 5.0, 7.0, 1.0)
                result = ProductionDataCollector.task_outcome_target(
                    decision(status=status),
                    [row(0, 5.0, lifecycle(last_time=5.0)),
                     row(1, 7.0, task, execution("cmd-hungarian", uav_id="uav-1"), terminated=True)],
                )
                self.assertTrue(result["physical_on_time_completion"]["valid"])
                self.assertTrue(result["physical_on_time_completion"]["value"])
                self.assertIs(result["completion_by_first_command"], False)
                self.assertFalse(result["action_specific_effect_identifiable"])

    def test_unknown_first_command_status_does_not_assert_command_attribution(self):
        task = lifecycle(deadline=6.0, required_service=2.0, last_time=4.0)
        frozen_decision = decision(status="unknown", task=task)
        task.assign("uav-0", 4.0)
        task.provide_service("uav-0", 4.0, 6.0, 1.0)
        result = ProductionDataCollector.task_outcome_target(
            frozen_decision,
            [row(0, 6.0, task, execution("cmd-first"), terminated=True)],
        )
        self.assertTrue(result["physical_on_time_completion"]["valid"])
        self.assertIsNone(result["completion_by_first_command"])

    def test_later_reassignment_is_sequence_result_not_first_command_completion(self):
        task = lifecycle(deadline=12.0, required_service=2.0, last_time=4.0)
        task.assign("uav-0", 4.0)
        initial = snapshot(task, execution("cmd-first"))
        frozen_decision = decision(status="accepted", task=task, initial_task=initial)
        task.advance(5.0)
        after_first = row(0, 5.0, task, execution("cmd-first"))
        task.interrupt(5.0)
        task.assign("uav-1", 6.0)
        after_reassign = row(1, 6.0, task, execution("cmd-hungarian", uav_id="uav-1", token=2))
        task.provide_service("uav-1", 6.0, 8.0, 1.0)
        result = ProductionDataCollector.task_outcome_target(
            frozen_decision,
            [after_first, after_reassign,
             row(2, 8.0, task, execution("cmd-hungarian", uav_id="uav-1", token=2), terminated=True)],
        )
        self.assertEqual(result["completion_execution_identity"]["command_id"], "cmd-hungarian")
        self.assertFalse(result["completion_by_first_command"])
        self.assertFalse(result["action_specific_effect_identifiable"])
        self.assertTrue(result["physical_on_time_completion"]["value"])

    def test_preexisting_execution_is_recorded_and_not_attributed_to_candidate(self):
        task = lifecycle(deadline=12.0, required_service=1.0, last_time=3.0)
        task.assign("uav-0", 3.0)
        initial = snapshot(task, execution("cmd-existing", token=1))
        frozen_decision = decision(status="rejected", task=task, initial_task=initial)
        task.provide_service("uav-0", 4.0, 5.0, 1.0)
        result = ProductionDataCollector.task_outcome_target(
            frozen_decision,
            [row(0, 5.0, task, execution("cmd-existing"), terminated=True)],
        )
        self.assertEqual(result["initial_execution_identity"]["command_id"], "cmd-existing")
        self.assertTrue(result["physical_on_time_completion"]["value"])
        self.assertFalse(result["action_specific_effect_identifiable"])

    def test_expiry_is_explicit_and_not_a_completion(self):
        task = lifecycle(deadline=9.0, required_service=4.0, last_time=4.0)
        frozen_decision = decision(task=task)
        task.assign("uav-0", 8.0)
        task.provide_service("uav-0", 8.0, 9.0, 1.0)
        result = ProductionDataCollector.task_outcome_target(
            frozen_decision, [row(0, 9.0, task, execution("cmd-first"), terminated=True)],
        )
        self.assertEqual(result["observation_end_status"], "expired")
        self.assertEqual(result["physical_on_time_completion"]["value"], False)
        self.assertEqual(result["task_expired"]["value"], True)
        self.assertTrue(result["task_expired"]["valid"])
        self.assertFalse(result["host_confirmation"]["valid"])

    def test_early_episode_end_and_horizon_before_deadline_remain_unknown(self):
        task = lifecycle(deadline=12.0, required_service=5.0, last_time=4.0)
        frozen_decision = decision(task=task)
        task.assign("uav-0", 4.0)
        task.provide_service("uav-0", 4.0, 6.0, 1.0)
        early = ProductionDataCollector.task_outcome_target(
            frozen_decision, [row(0, 6.0, task, execution("cmd-first"), terminated=True)],
        )
        self.assertFalse(early["physical_on_time_completion"]["valid"])
        self.assertFalse(early["task_expired"]["valid"])
        self.assertEqual(early["observation_end_status"], "unresolved_at_observation_end")

        late_deadline = lifecycle(deadline=20.0, required_service=30.0, last_time=4.0)
        frozen_late_decision = decision(task=late_deadline)
        late_deadline.assign("uav-0", 4.0)
        late_deadline.provide_service("uav-0", 4.0, 18.0, 1.0)
        at_horizon = ProductionDataCollector.task_outcome_target(
            frozen_late_decision, [row(0, 18.0, late_deadline, execution("cmd-first"), truncated=True)],
        )
        self.assertFalse(at_horizon["physical_on_time_completion"]["valid"])
        self.assertFalse(at_horizon["task_expired"]["valid"])

    def test_noop_masks_task_heads_and_keeps_host_unknown(self):
        result = ProductionDataCollector.task_outcome_target(
            decision(action=24, status="noop"), [
                {"step": 0, "time": 5.0, "continuation_id": EXPECTED_CONTINUATION,
                 "task_lifecycle": None, "terminated": False, "truncated": False},
            ],
        )
        self.assertFalse(result["physical_on_time_completion"]["valid"])
        self.assertFalse(result["task_expired"]["valid"])
        self.assertFalse(result["host_confirmation"]["valid"])

    def test_incomplete_or_out_of_bounds_post_action_trace_is_rejected(self):
        task = lifecycle(last_time=4.0)
        context = decision(task=task)
        valid_terminal = lifecycle(deadline=6.0, required_service=1.0, last_time=4.0)
        valid_terminal.assign("uav-0", 4.0)
        valid_terminal.provide_service("uav-0", 4.0, 6.0, 1.0)
        row0 = row(0, 5.0, lifecycle(last_time=5.0))
        terminal = row(1, 6.0, valid_terminal, execution("cmd-first"), terminated=True)
        with self.assertRaisesRegex(TaskOutcomeContractError, "trajectory_order_or_time_invalid"):
            ProductionDataCollector.task_outcome_target(context, [row0, {**terminal, "step": 2}])
        with self.assertRaisesRegex(TaskOutcomeContractError, "trajectory_order_or_time_invalid"):
            ProductionDataCollector.task_outcome_target(context, [row0, {**terminal, "time": 19.0}])
        with self.assertRaisesRegex(TaskOutcomeContractError, "post_action_trajectory_missing"):
            ProductionDataCollector.task_outcome_target(decision(action=24, status="noop"), [])

    def test_public_input_is_not_mutated_or_augmented_with_lifecycle(self):
        task = lifecycle(last_time=4.0)
        frozen = decision(task=task)
        before = copy.deepcopy(frozen)
        task.assign("uav-0", 4.0)
        task.provide_service("uav-0", 4.0, 6.0, 1.0)
        ProductionDataCollector.task_outcome_target(
            before, [row(0, 6.0, task, execution("cmd-first"), terminated=True)],
        )
        self.assertEqual(before, frozen)
        self.assertNotIn("task_lifecycle", before)

    def test_production_collector_requires_fixed_hungarian_and_wires_label_path(self):
        source = (PACKAGE / "production_data.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        collector = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ProductionDataCollector")
        collect_unit = next(node for node in collector.body if isinstance(node, ast.FunctionDef) and node.name == "_collect_unit")
        source_text = ast.get_source_segment(source, collect_unit)
        self.assertIn("TASK_OUTCOME_REQUIRES_FROZEN_HUNGARIAN_CONTINUATION", source_text)
        self.assertIn("self.classical_selector_type(\"hungarian\"", source_text)
        self.assertIn("self.task_outcome_target(decision_context, trajectory)", source_text)
        self.assertIn("\"task_outcome_target\": [row[\"task_outcome_target\"]", source_text)


if __name__ == "__main__":
    unittest.main()
