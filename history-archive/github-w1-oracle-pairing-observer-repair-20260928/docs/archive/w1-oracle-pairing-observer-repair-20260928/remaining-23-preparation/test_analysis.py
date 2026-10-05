from __future__ import annotations

import json
import unittest
from pathlib import Path

from analyze_oracle import AnalysisError, PARENTS, REPEATS, analyze
from reuse_audit import audit_reuse, read_jsonl


ROOT = Path(__file__).resolve().parent
REUSE = ROOT / "reused-first-unit"


class AnalysisTests(unittest.TestCase):
    def inputs(self):
        reuse = audit_reuse(ROOT, REUSE)
        candidates = read_jsonl(REUSE / "candidate-decisions.jsonl")
        summaries = read_jsonl(REUSE / "branch-summaries.jsonl")
        pairing = read_jsonl(REUSE / "communication-pairing.jsonl")
        for parent in PARENTS:
            for repeat in REPEATS:
                if (parent, repeat) == ("validation-0000", 0):
                    continue
                candidates.append({
                    "parent": parent,
                    "repeat": repeat,
                    "stage": "remaining_fixed_units",
                    "status": "NO_OPPORTUNITY",
                    "reason": "fixture",
                })
        return candidates, summaries, pairing, reuse

    def test_exact_24_unit_merge_counts_reused_unit_once(self):
        candidates, summaries, pairing, reuse = self.inputs()
        result = analyze(candidates, summaries, pairing, reused_audit=reuse)
        self.assertEqual(len(result["units"]), 24)
        self.assertEqual(result["matrix"]["reused_units"], 1)
        self.assertEqual(result["matrix"]["new_units"], 23)
        self.assertTrue(result["matrix"]["reused_unit_counted_once"])
        self.assertEqual(result["labels"]["unknown_task_labels"], 7)
        self.assertTrue(result["labels"]["unknowns_preserved_not_imputed"])

    def test_duplicate_reused_unit_is_rejected(self):
        candidates, summaries, pairing, reuse = self.inputs()
        candidates.append(dict(candidates[0]))
        with self.assertRaisesRegex(AnalysisError, "duplicate"):
            analyze(candidates, summaries, pairing, reused_audit=reuse)

    def test_missing_unit_is_not_filled_with_zero(self):
        candidates, summaries, pairing, reuse = self.inputs()
        candidates.pop()
        with self.assertRaisesRegex(AnalysisError, "incomplete"):
            analyze(candidates, summaries, pairing, reused_audit=reuse)


if __name__ == "__main__":
    unittest.main(verbosity=2)
