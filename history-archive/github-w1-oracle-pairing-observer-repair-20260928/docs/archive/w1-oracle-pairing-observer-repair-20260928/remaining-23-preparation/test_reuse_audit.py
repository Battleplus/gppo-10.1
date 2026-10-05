from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reuse_audit import ReuseAuditError, audit_reuse


ROOT = Path(__file__).resolve().parent


class ReuseAuditTests(unittest.TestCase):
    def test_completed_unit_recomputes_exactly_and_preserves_unknowns(self):
        result = audit_reuse(ROOT, ROOT / "reused-first-unit")
        self.assertTrue(result["passed"])
        self.assertEqual(result["counted_research_samples"], 1)
        self.assertEqual(result["branch_count"], 7)
        self.assertEqual(result["step_record_count"], 90)
        self.assertLessEqual(result["maximum_absolute_utility_error"], 1e-12)
        self.assertTrue(result["saved_environment_rewards_all_finite"])
        self.assertTrue(result["all_branches_native_terminal_and_untruncated"])
        self.assertEqual(result["unknown_host_confirmation_labels"], 7)
        self.assertFalse(result["unknown_labels_affect_utility_recomputation"])

    def test_hash_mismatch_stops_before_parsing(self):
        with tempfile.TemporaryDirectory() as temporary:
            reuse = Path(temporary)
            (reuse / "branch-summaries.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ReuseAuditError, "hash mismatch"):
                audit_reuse(ROOT, reuse)


if __name__ == "__main__":
    unittest.main(verbosity=2)
