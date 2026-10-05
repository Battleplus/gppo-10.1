"""Zero-call tests for trace-based metric reproduction."""

from __future__ import annotations

import unittest

from metrics import compare_trace, evaluate_trace
from trace_schema import PredictionTraceRow, stable_input_hash


class MetricTests(unittest.TestCase):
    def rows(self):
        digest = stable_input_hash({"d": "metric"})
        values = []
        for variant, scores in (("A", (0.0, 1.0)), ("B", (1.0, 0.0))):
            for action, score in enumerate(scores):
                values.append(PredictionTraceRow(
                    parent="p", repeat=0, decision_id="d", action=action,
                    legal_actions=(0, 1), continuation_id="hungarian-v1-fixed",
                    public_input_sha256=digest, true_utility=float(action), target_valid=True,
                    baseline_current_public=0.0, baseline_transparent=0.5,
                    seed_predictions=(score, score, score), ensemble_prediction=score,
                    variant=variant,
                ))
        return values

    def test_selected_action_tie_rule_and_regret(self):
        result = evaluate_trace(self.rows()[:2])
        self.assertEqual(result["selections"]["d"]["action"], 1)
        self.assertEqual(result["parent_macro_selected_regret"], 0.0)

    def test_compare_has_two_variants_and_baselines(self):
        result = compare_trace(self.rows())
        self.assertEqual(set(result), {"A", "B", "current_public", "transparent_history"})
        self.assertEqual(result["A"]["records"], 2)
        self.assertEqual(result["B"]["records"], 2)

    def test_compare_rejects_cross_variant_window_mismatch(self):
        rows = self.rows()
        rows[-1] = PredictionTraceRow(
            parent="other", repeat=0, decision_id="d", action=1,
            legal_actions=(0, 1), continuation_id="hungarian-v1-fixed",
            public_input_sha256=rows[-1].public_input_sha256, true_utility=1.0, target_valid=True,
            baseline_current_public=0.0, baseline_transparent=0.5,
            seed_predictions=(0.0, 0.0, 0.0), ensemble_prediction=0.0,
            variant="B",
        )
        with self.assertRaises(ValueError):
            compare_trace(rows)
