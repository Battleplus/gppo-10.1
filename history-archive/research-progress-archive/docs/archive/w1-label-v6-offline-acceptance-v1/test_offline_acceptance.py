"""Offline corruption checks; no environment, model or checkpoint imports."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

from analyze_v6 import ATTEMPT, bucket, summary, validate_candidate


class OfflineAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2] / "runs" / ATTEMPT / "run-once/world-model-windows.jsonl"
        cls.rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]

    def test_saved_qualification_evidence_and_incomplete_attribution(self):
        result, candidates = summary(self.rows)
        self.assertTrue(result["coverage_pass_original_rule"])
        self.assertEqual(len(candidates), 108)
        self.assertEqual(result["label_counts"]["physical_on_time_completion"], {"positive": 82, "negative": 18, "unknown": 8})
        self.assertEqual(result["invalid_attribution_count"], 9)
        self.assertEqual(result["parents_with_same_task_physical_difference"], 7)
        self.assertEqual(result["same_task_groups_with_physical_difference"], 13)

    def test_unknown_cannot_be_filled_as_negative(self):
        self.assertEqual(bucket({"valid": False, "value": None}), "unknown")
        with self.assertRaisesRegex(ValueError, "unknown_not_null"):
            bucket({"valid": False, "value": False})

    def test_future_public_receipt_rejected(self):
        rows = deepcopy(self.rows)
        rows[0]["decision_input"]["telemetry"][0]["received_at"] = 999.0
        # Recompute the input sidecar to ensure a valid digest cannot mask future information.
        from analyze_v6 import canonical
        public = rows[0]["decision_input"]
        raw_hash = canonical({"observation": public["observation"], "telemetry": public["telemetry"], "continuation": public["continuation_ids"]})
        for label in rows[0]["task_outcome_target"]:
            label["public_input_hash"] = raw_hash
        with self.assertRaisesRegex(ValueError, "future_public_telemetry"):
            summary(rows)

    def test_candidate_join_rejected(self):
        window = deepcopy(self.rows[0])
        label = window["task_outcome_target"][0]
        label["candidate_id"] = "wrong-candidate"
        with self.assertRaisesRegex(ValueError, "candidate_join"):
            validate_candidate(window, label, window["candidate_branch_audit"][0])

    def test_noop_task_outcome_rejected(self):
        window = deepcopy(self.rows[0])
        label = window["task_outcome_target"][-1]
        label["target_task_id"] = "task-0"
        with self.assertRaisesRegex(ValueError, "noop_identity"):
            validate_candidate(window, label, window["candidate_branch_audit"][-1])

    def test_receipt_after_cutoff_rejected(self):
        window = deepcopy(self.rows[0])
        index = next(i for i, l in enumerate(window["task_outcome_target"]) if l["host_confirmation"]["value"] is True)
        label, audit = window["task_outcome_target"][index], window["candidate_branch_audit"][index]
        audit["post_action_trajectory"][-1]["task_lifecycle"]["completion_notice"]["host_confirmation_time"] = label["observation_cutoff_time"] + 1
        with self.assertRaisesRegex(ValueError, "receipt_not_observed"):
            validate_candidate(window, label, audit)

    def test_nan_time_rejected(self):
        window = deepcopy(self.rows[0])
        audit = window["candidate_branch_audit"][0]
        audit["post_action_trajectory"][0]["time"] = float("nan")
        with self.assertRaisesRegex(ValueError, "trajectory_time"):
            validate_candidate(window, window["task_outcome_target"][0], audit)

    def test_no_learning_or_environment_modules_loaded(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)
        self.assertFalse(any(name.startswith("gppo_world") for name in sys.modules))


if __name__ == "__main__":
    unittest.main()
