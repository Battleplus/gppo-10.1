"""Preparation-only contract checks; no runtime, checkpoint, or torch call."""

from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class PreparationTests(unittest.TestCase):
    def test_resource_request_is_not_approved(self):
        request = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
        self.assertEqual(request["status"], "NOT_APPROVED")
        self.assertEqual(request["prior_decision"], "PREDICTION_GATE_NOT_PASSED")
        self.assertEqual(request["variants"]["auxiliary_weight"], 0.25)
        self.assertEqual(request["variants"]["beta"], 1.0)
        self.assertEqual(request["totals"]["branches"], 600)
        self.assertEqual(request["totals"]["checkpoint_writes"], 6)
        self.assertEqual(request["totals"]["offline_predictor_updates"], 28800)
        self.assertEqual(request["stage_limits"]["conditional_task_comparison"]["task_parent_count"], 8)

    def test_hash_manifest_identity(self):
        manifest = json.loads((ROOT / "execution-manifest.json").read_text(encoding="utf-8"))
        digest = hashlib.sha256((ROOT / "hashes.json").read_bytes()).hexdigest()
        self.assertEqual(manifest["hashes_sha256"], digest)

    def test_preparation_code_has_no_environment_import(self):
        for path in (ROOT / "objective.py", ROOT / "trace_schema.py", ROOT / "metrics.py", ROOT / "freeze_package.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module.split(".")[0])
            self.assertNotIn("m10_environment", imports)
            self.assertNotIn("subprocess", imports)
        launcher = (ROOT / "launch_once.py").read_text(encoding="utf-8")
        self.assertIn("wslpath", launcher)

    def test_new_parent_selection_is_fixed_and_disjoint(self):
        selection = json.loads((ROOT / "new-prediction-parent-selection.json").read_text(encoding="utf-8"))
        self.assertEqual(selection["repeats"], 3)
        self.assertEqual([row["tape_index"] for row in selection["parents"]], list(range(8)))
        self.assertEqual(selection["excluded_prior_matrix_parent_count"], 48)

    def test_old_labels_are_audit_only(self):
        audit = json.loads((ROOT / "data-reuse-audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["reused_training_records"], 219)
        self.assertEqual(audit["reused_model_selection_records"], 68)
        self.assertFalse(audit["old_prediction_evaluation_used_for_training"])


if __name__ == "__main__":
    unittest.main()
