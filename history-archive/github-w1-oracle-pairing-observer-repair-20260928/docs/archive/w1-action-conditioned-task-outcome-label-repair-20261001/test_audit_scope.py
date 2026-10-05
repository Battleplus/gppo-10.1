import json
import tempfile
import unittest
from pathlib import Path

from audit_saved_windows import audit_windows


class AuditScopeTests(unittest.TestCase):
    def test_scope_and_missing_task_heads_are_counted(self):
        row = {
            "split": "prediction_confirmation", "parent": "p0", "status": "complete",
            "candidate_branch_audit": [{"branch_info": {
                "counts": {"completed": 0, "expired": 0, "rejected": 0},
                "terminated": False, "truncated": False,
                "completion_records": {}, "communication_log": [],
                "tasks": {"task-0": "assigned"},
            }}],
            "outcome_target": [[None, None, None, 0.1, 0.2, 0.0]],
            "outcome_valid": [[False, False, False, True, True, True]],
            "event_valid": [[True, True, True, False, False]],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "windows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = audit_windows(path)
        self.assertEqual(result["rows_by_split"]["prediction_confirmation"], 1)
        self.assertEqual(result["nonempty_completion_record_branches"], 0)
        self.assertEqual(result["legacy_missing_event_valid_counts"], {
            "physical_completion_observed": 0,
            "host_confirmation_observed": 0,
        })
        self.assertEqual(result["task_state_summaries"]["prediction_confirmation:assigned"], 1)


if __name__ == "__main__":
    unittest.main()
