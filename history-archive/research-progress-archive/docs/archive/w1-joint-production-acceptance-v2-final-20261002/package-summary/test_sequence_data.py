import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE / "native"))

import production_data  # noqa: E402
from runtime_config_contract import canonical_sha256  # noqa: E402
from sequence_data_contract import (  # noqa: E402
    SequenceDataContractError, validate_sequence_window, _validate_label_records, _validate_step,
)
from task_outcome_contract import derive_action_conditioned_task_outcome  # noqa: E402
from test_action_conditioned_task_outcomes import decision, execution, lifecycle, row  # noqa: E402
from test_production_collector_lifecycle import ProductionCollectorLifecycleTests  # noqa: E402
from transparent_utility import (  # noqa: E402
    transparent_horizon_components, transparent_utility_components,
)


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(production_data._jsonable(payload), sort_keys=True) + "\n")


def write_environment_record(path, config):
    effective = dict(vars(config))
    record = {
        "schema": "w1-runtime-environment-record/1.0.0",
        "config": effective,
        "config_sha256": canonical_sha256(effective),
        "task_completion_mode": effective.get("task_completion_mode"),
        "deadline_basis": effective.get("deadline_basis"),
    }
    Path(path).write_text(json.dumps(record), encoding="utf-8")
    return record


def collect_fixture(profile=None):
    with patch.object(production_data, "durable_append_jsonl", append_jsonl), patch.object(
        production_data, "write_runtime_environment_record", write_environment_record,
    ):
        return ProductionCollectorLifecycleTests().collect_window(profile or {})


class SequenceDataTests(unittest.TestCase):
    def test_lifecycle_rows_are_returned_and_invalid_completion_rejected(self):
        rows = [
            {"task_id": "active", "state": "serving", "deadline": 8.0, "completed_at": None},
            {"task_id": "done", "state": "completed", "deadline": 8.0, "completed_at": 5.0},
            {"task_id": "expired", "state": "expired", "deadline": 4.0, "completed_at": None},
        ]
        self.assertEqual(_validate_label_records(rows, "LIFECYCLE_INVALID"), rows)
        self.assertEqual(_validate_label_records([], "LIFECYCLE_INVALID"), [])
        for changed in (
            {**rows[1], "completed_at": None},
            {**rows[0], "completed_at": 5.0},
            {**rows[0], "deadline": -1.0},
        ):
            with self.assertRaisesRegex(SequenceDataContractError, "LIFECYCLE_INVALID"):
                _validate_label_records([changed], "LIFECYCLE_INVALID")

    def test_step_checks_actual_lifecycle_identities_and_completion_times(self):
        task = {"task_id": "target", "state": "serving", "deadline": 8.0, "completed_at": None}
        step = {"step": 0, "kind": "branch", "continuation_id": "hungarian-v1-fixed",
                "action_id": 0, "time_before": 4.0, "time_after": 5.0,
                "scalar_environment_reward": 0.0, "vector_reward": [0.0, 0.0],
                "counts_before": {"completed": 0, "expired": 0, "rejected": 0},
                "counts_after": {"completed": 0, "expired": 0, "rejected": 0},
                "energy_before": 36.0, "energy_after": 35.0,
                "task_lifecycle_before": [task], "task_lifecycle_after": [dict(task)],
                "terminated": False, "truncated": False}
        _validate_step(step, 0, kind="branch", horizon=9.0)
        changed_id = copy.deepcopy(step)
        changed_id["task_lifecycle_after"][0]["task_id"] = "other"
        with self.assertRaisesRegex(SequenceDataContractError, "SEQUENCE_LIFECYCLE_IDENTITY_CHANGED"):
            _validate_step(changed_id, 0, kind="branch", horizon=9.0)
        future = copy.deepcopy(step)
        future["task_lifecycle_after"][0].update(state="completed", completed_at=6.0)
        with self.assertRaisesRegex(SequenceDataContractError, "SEQUENCE_COMPLETION_TIME_AFTER_OBSERVATION"):
            _validate_step(future, 0, kind="branch", horizon=9.0)

    def test_frozen_proposal_split_supports_all_three_parent_roles(self):
        instance = production_data.ProductionDataCollector.__new__(production_data.ProductionDataCollector)
        instance.root = PACKAGE
        instance.matrix = json.loads((PACKAGE / "experiment-matrix.json").read_text(encoding="utf-8"))
        frozen = instance._load_parent_split()
        self.assertEqual(frozen["split_counts"], {
            "train": 24, "model_selection": 8, "prediction_confirmation": 8,
        })
        self.assertTrue(all("proposed_split" in row for row in frozen["parents"]))

    def test_real_collector_fixture_persists_recomputable_sequence_returns(self):
        _summary, window, _calls = collect_fixture({"deadline": 6.0, "required_service": 1.0})
        audit = validate_sequence_window(window)
        self.assertEqual(audit["schema"], "w1-world-model-complete-window/3.0.0")
        self.assertEqual(len(window["prefix_label_evidence"]["steps"]), 4)
        self.assertEqual(len(window["horizon_task_outcome_target"][0]), 3)
        self.assertEqual(len(window["outcome_target"][0]), 6)
        self.assertNotEqual(window["input_hash"], window["public_input_hash"])
        self.assertEqual(
            window["task_outcome_target"][0]["public_input_hash"],
            window["public_input_hash"],
        )
        for index, (steps, result) in enumerate(zip(window["sequence_label_evidence"], window["sequence_return"])):
            self.assertEqual(result["step_count"], len(steps["steps"]))
            self.assertEqual(result["scalar_environment_reward_sum"], sum(
                item["scalar_environment_reward"] for item in steps["steps"]
            ))
            self.assertEqual(result["vector_reward_sum"], window["vector_reward"][index])
            for step in steps["steps"]:
                self.assertIn("task_lifecycle_before", step)
                self.assertIn("task_lifecycle_after", step)
                self.assertIn("counts_before", step)
                self.assertIn("counts_after", step)
                self.assertIn("energy_before", step)
                self.assertIn("energy_after", step)

    def test_validator_rejects_bad_return_future_input_and_fabricated_unknown(self):
        _summary, window, _calls = collect_fixture()
        changed_return = copy.deepcopy(window)
        changed_return["sequence_return"][0]["vector_reward_sum"][0] += 0.25
        with self.assertRaisesRegex(SequenceDataContractError, "SEQUENCE_VECTOR_SUM_MISMATCH"):
            validate_sequence_window(changed_return)

        future_input = copy.deepcopy(window)
        future_input["decision_input"]["telemetry"].append({
            "entity": "uav-0", "field": "energy", "value": 9.0,
            "measured_at": 4.1, "received_at": 4.1, "sequence": 1,
        })
        with self.assertRaisesRegex(SequenceDataContractError, "FUTURE_PUBLIC_INPUT"):
            validate_sequence_window(future_input)

        noop = window["actions"].index(24)
        fabricated_unknown = copy.deepcopy(window)
        fabricated_unknown["horizon_task_outcome_target"][noop][0] = 0
        with self.assertRaisesRegex(SequenceDataContractError, "HORIZON_UNKNOWN_MUST_REMAIN_NULL"):
            validate_sequence_window(fabricated_unknown)

        wrong_first_action = copy.deepcopy(window)
        wrong_first_action["sequence_label_evidence"][0]["steps"][0]["action_id"] = 24
        with self.assertRaisesRegex(SequenceDataContractError, "FIRST_ACTION_MEMBERSHIP_INVALID"):
            validate_sequence_window(wrong_first_action)

    def test_expired_execution_identity_cannot_be_attributed_as_first_completion(self):
        target = lifecycle(deadline=8.0, required_service=4.0, last_time=4.0)
        frozen_decision = decision(task=target, status="accepted")
        target.assign("uav-0", 4.0)
        target.advance(9.0)
        outcome = derive_action_conditioned_task_outcome(
            frozen_decision,
            [row(0, 9.0, target, execution("cmd-first"), terminated=True)],
        )
        self.assertTrue(outcome["task_expired"]["valid"])
        self.assertIsNone(outcome["completion_execution_identity"])
        self.assertIsNone(outcome["completion_by_first_command"])
        self.assertFalse(outcome["action_specific_effect_identifiable"])

    def test_public_horizon_surrogate_adds_only_ideal_public_completions(self):
        diagnostic = {"score": 42.0, "energy_cost": 12.0}
        first_step = transparent_utility_components(
            diagnostic, task_capacity=6, initial_total_energy=36.0,
        )
        horizon = transparent_horizon_components(
            diagnostic, task_capacity=6, initial_total_energy=36.0,
            ideal_continuation_completions=2,
        )
        self.assertAlmostEqual(horizon[0], first_step[0] + 2.0 / 6.0)
        self.assertAlmostEqual(horizon[1], first_step[1])


if __name__ == "__main__":
    unittest.main()
