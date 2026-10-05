from __future__ import annotations

import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from outcome_contract import ContractError, SCHEMA, assert_same_continuation, masked, unknown_labels, validate_record
from predictor import CandidatePrediction, FakeOutcomePredictor, masked_binary_cross_entropy, masked_smooth_l1
from prior_adapter import apply_candidate_prior, predict_and_apply, utility_score
from experiment_matrix import build_matrix, derive_budget
from learning import ModelSpec, TrainingConfig
from learning_schema import (
    INPUT_DIM,
    LearningContractError,
    LearningRecord,
    assert_parent_disjoint,
    history_features,
    materialize_features,
)
from one_shot_policy import OneShotOutcomeSelector
from prediction_evaluation import compare_all, ensemble_mean, prediction_gate
from pipeline import execute_pipeline


ROOT = Path(__file__).resolve().parent


def base_record(action: int = 0, *, executed: bool = True) -> dict:
    candidate = {
        "action": action,
        "kind": "NOOP" if action == 24 else "assignment",
        "uav_id": None if action == 24 else "uav-0",
        "task_id": None if action == 24 else "task-0",
    }
    labels = unknown_labels(["task-0"])
    labels["target_task"]["task_id"] = candidate["task_id"]
    if executed:
        labels["first_command_state"] = masked("noop" if action == 24 else "accepted", True, "synthetic", name="command")
        labels["tasks"]["task-0"] = {
            "state": masked("completed", True, "synthetic", name="state"),
            "physical_on_time": masked(True, True, "synthetic", name="physical"),
            "completion_uav": masked("uav-0", True, "synthetic", name="uav"),
            "host_on_time": masked(None, False, None, name="host"),
            "host_confirmation_time": masked(None, False, None, name="host_time"),
        }
        labels["energy_to_terminal"] = masked(1.5, True, "synthetic", name="energy")
        labels["remaining_utility"] = masked(0.2, True, "synthetic", name="utility")
        labels["terminal_time"] = masked(12.0, True, "synthetic", name="terminal")
        if action != 24:
            labels["target_task"] = {
                "task_id": "task-0",
                "physical_on_time": masked(True, True, "synthetic", name="target.physical"),
                "completed_by_current_uav": masked(True, True, "synthetic", name="target.uav"),
                "completion_time": masked(3.0, True, "synthetic", name="target.time"),
                "host_on_time": masked(None, False, None, name="target.host"),
            }
    return {
        "schema": SCHEMA,
        "decision_id": "synthetic",
        "source": {
            "decision_time": 4.0,
            "split_role": "development",
            "independent_validation": False,
        },
        "input": {
            "current_public": {"observation_time": 4.0},
            "public_history": [{"observation_time": 3.0}],
            "own_action_history": [{"decision_time": 3.0, "action": 24}],
        },
        "candidate": candidate,
        "legal_actions": [action, 24] if action != 24 else [24],
        "continuation_id": "factual-policy:synthetic",
        "provenance": {
            "is_executed": executed,
            "is_true_branch": False,
            "label_scope": "factual_trajectory" if executed else "unknown_unexecuted",
        },
        "labels": labels,
        "training_eligible": False,
        "eligibility_reasons": ["synthetic_test"],
    }


def prediction(action: int, utility: float, continuation: str = "fixed") -> CandidatePrediction:
    return CandidatePrediction(
        action=action,
        continuation_id=continuation,
        command_state_probabilities={"noop": 1.0} if action == 24 else {"accepted": 1.0},
        target_physical_on_time_probability=None if action == 24 else 0.5,
        target_completed_by_current_uav_probability=None if action == 24 else 0.5,
        energy_to_terminal=1.0,
        remaining_utility=utility,
        target_completion_time=None,
        valid={
            "command_state": True,
            "target_physical_on_time": action != 24,
            "target_completed_by_current_uav": action != 24,
            "energy_to_terminal": True,
            "remaining_utility": True,
            "target_completion_time": False,
        },
    )


class ContractTests(unittest.TestCase):
    def test_valid_factual_record(self):
        validate_record(base_record())

    def test_future_or_hidden_input_rejected(self):
        record = base_record()
        record["input"]["public_history"][0]["observation_time"] = 5.0
        with self.assertRaises(ContractError):
            validate_record(record)
        record = base_record()
        record["input"]["hidden_fault"] = "damage-at-7"
        with self.assertRaises(ContractError):
            validate_record(record)

    def test_unexecuted_labels_are_unknown(self):
        record = base_record(executed=False)
        validate_record(record)
        record["labels"]["remaining_utility"] = masked(0.0, True, "fabricated", name="utility")
        with self.assertRaises(ContractError):
            validate_record(record)

    def test_noop_target_is_masked_not_zero(self):
        record = base_record(24)
        validate_record(record)
        record["labels"]["target_task"]["physical_on_time"] = masked(False, True, "fabricated", name="target")
        with self.assertRaises(ContractError):
            validate_record(record)

    def test_rejection_does_not_erase_eventual_completion(self):
        record = base_record()
        record["labels"]["first_command_state"] = masked("executor_rejected", True, "synthetic", name="command")
        validate_record(record)
        self.assertEqual(record["labels"]["tasks"]["task-0"]["state"]["value"], "completed")

    def test_continuations_cannot_merge(self):
        left = base_record()
        right = base_record()
        right["continuation_id"] = "hungarian-v1-fixed"
        right["provenance"]["is_true_branch"] = True
        right["provenance"]["label_scope"] = "fixed_hungarian_true_branch"
        with self.assertRaises(ContractError):
            assert_same_continuation([left, right])


class PredictorAndPriorTests(unittest.TestCase):
    def test_masked_losses_preserve_unknown(self):
        self.assertAlmostEqual(masked_binary_cross_entropy([0.8, 0.1], [1, 1], [True, False]), -math.log(0.8))
        self.assertEqual(masked_smooth_l1([2.0, 99.0], [1.0, -99.0], [True, False]), 0.5)
        self.assertIsNone(masked_smooth_l1([1.0], [2.0], [False]))

    def test_zero_coefficient_restores_base_logits_and_choice(self):
        mask = [False] * 25
        mask[0] = mask[1] = mask[24] = True
        base = [float(index) / 10.0 for index in range(25)]
        predictions = [prediction(0, 100), prediction(1, 50), prediction(24, -100)]
        final, _ = apply_candidate_prior(base, mask, predictions, coefficient=0.0, expected_continuation_id="fixed")
        for index in (0, 1, 24):
            self.assertEqual(final[index], base[index])
        self.assertEqual(max((i for i in range(25) if mask[i]), key=final.__getitem__), 24)

    def test_illegal_action_cannot_be_unmasked(self):
        mask = [False] * 25
        mask[0] = mask[24] = True
        base = [0.0] * 25
        base[1] = float("-inf")
        final, _ = apply_candidate_prior(base, mask, [prediction(0, 0), prediction(24, 0)],
                                         coefficient=1.0, expected_continuation_id="fixed")
        self.assertEqual(final[1], float("-inf"))

    def test_prediction_set_must_equal_legal_candidates(self):
        mask = [False] * 25
        mask[0] = mask[1] = True
        with self.assertRaises(ValueError):
            apply_candidate_prior([0.0] * 25, mask, [prediction(0, 0)],
                                  coefficient=1.0, expected_continuation_id="fixed")

    def test_continuation_mismatch_rejected(self):
        mask = [False] * 25
        mask[0] = True
        with self.assertRaises(ValueError):
            apply_candidate_prior([0.0] * 25, mask, [prediction(0, 1, "other")],
                                  coefficient=1.0, expected_continuation_id="fixed")

    def test_direct_and_component_utility_cannot_double_count(self):
        value = prediction(0, 1.0)
        object.__setattr__(value, "diagnostics", {"component_utility_score": 1.0})
        with self.assertRaises(ValueError):
            utility_score(value)

    def test_fake_predictor_connects_to_selection_and_diagnostics(self):
        mask = [False] * 25
        mask[0] = mask[1] = mask[24] = True
        fake = FakeOutcomePredictor("fixed", {0: 0.1, 1: 0.5, 24: 0.0})
        selected, diagnostics = predict_and_apply(fake, {"decision_id": "d1"}, [0.0] * 25, mask, coefficient=1.0)
        self.assertEqual(selected, 1)
        self.assertEqual(fake.calls, [("d1", (0, 1, 24))])
        self.assertEqual([item.action for item in diagnostics if item.legal], [0, 1, 24])


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.availability = json.loads((ROOT / "data-availability.json").read_text(encoding="utf-8"))

    def test_export_is_development_grouped_and_not_training_eligible(self):
        self.assertTrue(self.availability["development_only"])
        self.assertEqual(self.availability["independent_validation_parents"], 0)
        self.assertEqual(self.availability["training_eligible_records"], 0)
        self.assertEqual(self.availability["factual"]["factual_decisions"], 774)

    def test_full_export_has_expected_continuation_boundaries(self):
        self.assertEqual(self.availability["sealed_oracle"]["oracle_branch_records"], 144)
        self.assertEqual(self.availability["factual"]["factual_unexecuted_candidate_records"], 3321)
        self.assertIn("hungarian-v1-fixed", self.availability["continuation_identities"])

    def test_resource_request_is_unapproved_and_internally_bounded(self):
        request = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
        self.assertEqual(request["status"], "NOT_APPROVED")
        matrix = json.loads((ROOT / "experiment-matrix.json").read_text(encoding="utf-8"))
        self.assertEqual(request["derived_dynamic_budget"], derive_budget(matrix))
        totals = request["derived_dynamic_budget"]["active_call_totals"]
        self.assertEqual(totals["environment_steps"], 42480)
        self.assertEqual(totals["offline_predictor_updates"], 5700)
        self.assertEqual(totals["branches"], 2200)

    def test_preparation_has_not_created_execution_output(self):
        self.assertFalse((ROOT / "run-once").exists())
        self.assertEqual(json.loads((ROOT / "runtime-inputs.json").read_text())["preparation_calls"], {
            "environment": 0,
            "model_initializations": 0,
            "model_forwards": 0,
            "training_updates": 0,
        })


def learning_record(parent="p0", role="train", action=0, decision="d0", utility=1.0,
                    current=0.0, transparent=0.0):
    return LearningRecord.from_mapping({
        "sample_id": f"{decision}:{action}",
        "parent": parent,
        "repeat": 0,
        "decision_id": decision,
        "split_role": role,
        "action": action,
        "legal_actions": [0, 1],
        "flat": [0.0] * 770,
        "history": [0.0] * 32,
        "remaining_utility": utility,
        "current_public_score": current,
        "transparent_history_score": transparent,
        "continuation_id": "hungarian-v1-fixed",
        "is_true_branch": True,
        "training_eligible": True,
    })


class LearningLoopTests(unittest.TestCase):
    def test_exact_feature_and_model_shapes(self):
        self.assertEqual(INPUT_DIM, 827)
        self.assertEqual(len(materialize_features([0.0] * 770, [0.0] * 32, 24)), 827)
        ModelSpec().validate()
        TrainingConfig().validate()

    def test_history_rejects_future_observation(self):
        observation = {
            "time": 2.0,
            "uavs": [[0.0] * 24 for _ in range(4)],
            "tasks": [[0.0] * 32 for _ in range(6)],
            "mask": [False] * 24 + [True],
            "continuation_actions": [],
            "event_signal": 0.0,
        }
        with self.assertRaises(LearningContractError):
            history_features([observation], 1.0)

    def test_factual_or_wrong_continuation_cannot_enter_training(self):
        row = learning_record()
        values = dict(row.__dict__)
        values["is_true_branch"] = False
        with self.assertRaises(LearningContractError):
            LearningRecord(**values).validate()
        values["is_true_branch"] = True
        values["continuation_id"] = "factual-policy:PPO"
        with self.assertRaises(LearningContractError):
            LearningRecord(**values).validate()

    def test_training_admission_flags_require_json_booleans(self):
        row = dict(learning_record().__dict__)
        row["is_true_branch"] = "false"
        with self.assertRaises(LearningContractError):
            LearningRecord.from_mapping(row)
        row["is_true_branch"] = True
        row["training_eligible"] = 1
        with self.assertRaises(LearningContractError):
            LearningRecord.from_mapping(row)

    def test_task_model_choice_rejects_nonfinite_and_records_complete_choice(self):
        from runtime_backend import AuthorizedRuntimeBackend, TechnicalStop

        class FakeOutput:
            def __init__(self, values):
                self.values = values

            def __getitem__(self, _key):
                return self.values

        class NoGrad:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        fake_torch = types.SimpleNamespace(
            float32="float32",
            tensor=lambda values, dtype=None: (values, dtype),
            no_grad=lambda: NoGrad(),
        )
        backend = AuthorizedRuntimeBackend.__new__(AuthorizedRuntimeBackend)
        backend.decision_costs = []
        backend.account_call = lambda _name, _amounts, fn, *args: fn(*args)
        memory = types.SimpleNamespace(history=[])
        observation = {"time": 1.0, "flat": [0.0] * 770}
        finite_models = [lambda _features: FakeOutput([0.0, 1.0])] * 3
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            scores, action = backend._model_choice(finite_models, observation, memory, (0, 1))
        self.assertEqual(scores, {0: 0.0, 1: 1.0})
        self.assertEqual(action, 1)
        self.assertEqual(len(backend.decision_costs), 1)
        self.assertGreaterEqual(backend.decision_costs[0]["wall_seconds"], 0.0)
        nonfinite_models = [lambda _features: FakeOutput([0.0, math.nan])] * 3
        with mock.patch.dict(sys.modules, {"torch": fake_torch}), self.assertRaises(TechnicalStop):
            backend._model_choice(nonfinite_models, observation, memory, (0, 1))

    def test_parent_splits_are_disjoint(self):
        with self.assertRaises(LearningContractError):
            assert_parent_disjoint([
                learning_record(parent="same", role="train"),
                learning_record(parent="same", role="model_selection", decision="d1"),
            ])

    def test_incomplete_candidate_labels_are_rejected(self):
        from learning_schema import validate_decision_groups
        with self.assertRaises(LearningContractError):
            validate_decision_groups([learning_record(action=0)])

    def test_prediction_metrics_and_gate_use_candidate_regret(self):
        rows = [
            learning_record(action=0, utility=0.0, current=0.1, transparent=0.2),
            learning_record(action=1, utility=1.0, current=0.0, transparent=0.0),
        ]
        comparison = compare_all(rows, [0.0, 1.0])
        self.assertTrue(prediction_gate(comparison)["pass"])
        self.assertEqual(ensemble_mean([[0.0, 1.0]] * 3), [0.0, 1.0])

    def test_one_shot_selection_then_hungarian(self):
        class Memory:
            def candidates(self, _observation):
                return (0, 1, 24)

        calls = []
        selector = OneShotOutcomeSelector(
            hungarian_choose=lambda _obs, _memory: (0, {"method": "hungarian"}),
            consequence_scores=lambda _obs, _memory, actions: calls.append(tuple(actions)) or {0: 0.0, 1: 1.0, 24: -1.0},
            mode="fake",
        )
        self.assertEqual(selector.choose({}, Memory())[0], 1)
        self.assertEqual(selector.choose({}, Memory())[0], 0)
        self.assertEqual(calls, [(0, 1, 24)])
        self.assertEqual(selector.prediction_calls, 1)

    def test_matrix_groups_are_frozen_and_disjoint(self):
        matrix = json.loads((ROOT / "experiment-matrix.json").read_text(encoding="utf-8"))
        groups = [
            {row["parent"] for row in matrix["groups"][role]}
            for role in ("train", "model_selection", "prediction_evaluation", "task_evaluation")
        ]
        self.assertEqual([len(group) for group in groups], [24, 8, 8, 8])
        self.assertEqual(len(set().union(*groups)), 48)

    def test_fake_pipeline_reaches_conditional_task_stage_without_real_model(self):
        class Backend:
            def __init__(self, data_path):
                self.data_path = data_path
                self.stages = []
                self.task_called = False
                self.settlement = None

            def select_stage(self, stage):
                self.stages.append(stage)

            def account_call(self, _name, _amounts, function, *args):
                return function(*args)

            def zero_step_gate(self):
                return {"pass": True}

            def collect_fixed_continuation_labels(self):
                return self.data_path

            def run_task_comparison(self, models):
                self.task_called = True
                self.models = list(models)
                return {"executed": True}

            def settle(self, status, _details):
                self.settlement = status

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for role, parent in (("train", "p-train"), ("model_selection", "p-select"),
                                 ("prediction_evaluation", "p-eval")):
                rows.extend([
                    learning_record(parent=parent, role=role, action=0, decision=f"d-{role}", utility=0.0,
                                    current=0.1, transparent=0.2).__dict__,
                    learning_record(parent=parent, role=role, action=1, decision=f"d-{role}", utility=1.0,
                                    current=0.0, transparent=0.0).__dict__,
                ])
            data = root / "records.jsonl"
            data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            backend = Backend(data)
            with mock.patch("pipeline.train_seed", return_value={"ok": True}), \
                 mock.patch("pipeline.load_checkpoint", side_effect=[
                     ("m1", {"seed": 7101}), ("m2", {"seed": 7102}), ("m3", {"seed": 7103})
                 ]), \
                 mock.patch("pipeline.predict_checkpoint", return_value=[0.0, 1.0]):
                result = execute_pipeline(backend, root / "output")
            self.assertEqual(result.status, "complete")
            self.assertTrue(backend.task_called)
            self.assertEqual(backend.models, ["m1", "m2", "m3"])
            self.assertEqual(backend.stages[-1], "conditional_task_comparison")


if __name__ == "__main__":
    unittest.main()
