from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from production_world import TRACE_SCHEMA, _trace_decision, recompute_prediction_metrics


class PersistedMetricTests(unittest.TestCase):
    def test_mae_cannot_cancel_and_calibration_and_pairwise_are_recomputed(self):
        candidates = []
        targets = ([1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                   [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0])
        for action, target in enumerate(targets):
            predictions = {}
            for variant, probability in (("G1", 0.5), ("G2", 0.1 if action == 0 else 0.9)):
                by_model = {}
                for key in ("ensemble", "8201", "8202", "8203"):
                    outcome = [0.0] * 6
                    outcome[4] = target[4]
                    by_model[key] = {
                        "outcome": outcome,
                        "public_state": [0.0] * 128,
                        "event_probability": [probability] * 5,
                    }
                predictions[variant] = by_model
            candidates.append({
                "candidate_id": f"candidate-{action}", "action_id": action,
                "transparent_score": 0.0, "outcome_target": list(target),
                "true_utility": float(target[4]),
                "outcome_valid": [True] * 6, "state_target": [0.0] * 128,
                "state_valid": [True] * 128,
                "event_labels": [0.0 if action == 0 else 1.0] * 5,
                "event_valid": [True] * 5, "predictions": predictions,
            })
        window = {
            "schema": TRACE_SCHEMA, "split": "prediction_confirmation",
            "parent": "parent-0", "repeat": 0, "window_id": "window-0",
            "scenario_sha256": "0" * 64, "continuation_id": "one-step-public-transition-v1",
            "input_hash": "1" * 64, "candidate_count": 2,
            "candidate_rows": candidates, "decisions": {}, "status": "complete",
        }
        methods = ["transparent", "G1", "G2"] + [
            f"{variant}_seed_{seed}" for variant in ("G1", "G2") for seed in (8201, 8202, 8203)
        ]
        for method in methods:
            window["decisions"][method] = _trace_decision(window, method, (8201, 8202, 8203))
        path = Path(tempfile.mkdtemp()) / "trace.jsonl"
        path.write_text(json.dumps(window, sort_keys=True) + "\n", encoding="utf-8")
        metrics = recompute_prediction_metrics(path, expected_parent_count=1)
        self.assertGreater(metrics["parent_macro_absolute_outcome_mae"]["G1:ensemble"], 0.0)
        self.assertGreater(metrics["parent_macro_outcome_rmse"]["G1:ensemble"], 0.0)
        self.assertEqual(metrics["parent_macro_pairwise_direction_accuracy"]["G1"], 1.0)
        self.assertIn("G2:ensemble", metrics["event_nll_parent_macro"])
        self.assertIn("G2:ensemble", metrics["event_ece_10bin_parent_macro"])


if __name__ == "__main__":
    unittest.main()
