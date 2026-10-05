import unittest

from task_outcome_label_contract import LabelContractError, derive_task_outcome


def valid_record(**overrides):
    record = {
        "action_id": 1,
        "public_input_hash": "a" * 64,
        "legal_actions": [1, 24],
        "target_task_id": "task-0",
        "target_uav_id": "uav-0",
        "first_command_id": "command-0",
        "continuation_id": "hungarian-v1",
        "decision_time": 4.0,
        "trajectory": [{"time": 18.0, "continuation_id": "hungarian-v1", "truncated": True, "info": {"feedback": "accepted"}}],
        "completion_records": {
            "task-0": {
                "physical_arrival_time": 7.0,
                "deadline": 9.0,
                "task_id": "task-0",
                "execution_identity": {"command_id": "command-0", "uav_id": "uav-0"},
                "host_confirmation_time": 8.0,
                "completion_message_id": "completion-0",
            }
        },
        "communication_log": [{
            "message_kind": "completion", "status": "received",
            "message_id": "completion-0", "received_at": 8.0,
        }],
        "task_states": {"task-0": "completed"},
    }
    record.update(overrides)
    return record


class TaskOutcomeContractTests(unittest.TestCase):
    def test_explicit_completion_and_confirmation_are_valid(self):
        result = derive_task_outcome(valid_record())
        self.assertEqual(result.valid, (True, True, True, True, False))
        self.assertEqual(result.values[:4], (1.0, 1.0, 1.0, 1.0))

    def test_missing_records_stay_unknown(self):
        result = derive_task_outcome(valid_record(completion_records={}, communication_log=[], task_states={}, trajectory=[]))
        self.assertEqual(result.valid, (False, False, False, False, False))
        self.assertIsNone(result.values[0])
        self.assertIn("fixed_continuation_not_recorded_to_terminal_or_horizon", result.reasons)

    def test_noop_masks_all_task_labels(self):
        result = derive_task_outcome({"action_id": 24, "public_input_hash": "a" * 64,
                                      "legal_actions": [1, 24], "continuation_id": "hungarian-v1", "decision_time": 4.0})
        self.assertEqual(result.valid, (False, False, False, False, False))
        self.assertIn("action_is_noop_no_target_task", result.reasons)

    def test_future_or_mismatched_continuation_is_rejected(self):
        with self.assertRaises(LabelContractError):
            derive_task_outcome(valid_record(trajectory=[{"time": 3.9, "continuation_id": "hungarian-v1"}]))
        with self.assertRaises(LabelContractError):
            derive_task_outcome(valid_record(trajectory=[{"time": 5.0, "continuation_id": "other"}]))

    def test_partial_continuation_keeps_even_observed_completion_unknown(self):
        record = valid_record(trajectory=[{"time": 5.0, "continuation_id": "hungarian-v1", "info": {"feedback": "accepted"}}])
        result = derive_task_outcome(record)
        self.assertEqual(result.valid, (False, False, False, False, False))

    def test_expiry_requires_explicit_task_state(self):
        result = derive_task_outcome(valid_record(task_states={"task-0": "expired"}, completion_records={}, communication_log=[]))
        self.assertEqual(result.valid, (False, False, False, False, True))
        self.assertEqual(result.values[4], 1.0)

    def test_other_command_completion_is_not_attributed_to_first_action(self):
        record = valid_record()
        record["completion_records"]["task-0"]["execution_identity"]["command_id"] = "later-command"
        result = derive_task_outcome(record)
        self.assertFalse(result.valid[0])
        self.assertFalse(result.valid[2])

    def test_service_completion_source_is_supported(self):
        record = valid_record(completion_records={}, communication_log=[],
                              service_completion_record={
                                  "task_id": "task-0", "completed_at": 7.0, "deadline": 9.0,
                                  "execution_identity": {"command_id": "command-0", "uav_id": "uav-0"},
                              })
        result = derive_task_outcome(record)
        self.assertEqual(result.valid, (True, True, False, False, False))


if __name__ == "__main__":
    unittest.main()
