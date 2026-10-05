"""Pure source and lifecycle checks; no W1 environment is constructed."""
import runpy
import unittest

from source_contract_check import SOURCE_ROOT, check_source


class SourceContractTests(unittest.TestCase):
    def test_default_mode_has_no_completion_notice_path(self):
        result = check_source()
        self.assertEqual(result["default_mode"], "continuous_service_until_deadline")
        self.assertFalse(result["completion_notice_enabled_in_default_mode"])
        self.assertFalse(result["physical_completed_at_in_step_info"])
        self.assertTrue(result["actual_collector_uses_default_config"])
        self.assertTrue(result["policy_and_task_hooks_use_default_config"])
        self.assertEqual(result["source_run_recorded_mode"], "arrival_to_region")

    def test_lifecycle_records_physical_completion_without_environment(self):
        module = runpy.run_path(str(SOURCE_ROOT / "task_lifecycle.py"))
        task = module["TaskLifecycle"]("task", arrival=0.0, deadline=9.0, required_service=2.0)
        task.assign("uav", 4.0)
        task.provide_service("uav", 4.0, 6.0, 1.0)
        self.assertEqual(task.state, module["TaskState"].COMPLETED)
        self.assertEqual(task.completed_at, 6.0)

    def test_lifecycle_expiry_does_not_invent_completion(self):
        module = runpy.run_path(str(SOURCE_ROOT / "task_lifecycle.py"))
        task = module["TaskLifecycle"]("task", arrival=0.0, deadline=9.0, required_service=4.0)
        task.assign("uav", 8.0)
        task.provide_service("uav", 8.0, 9.0, 1.0)
        self.assertEqual(task.state, module["TaskState"].EXPIRED)
        self.assertIsNone(task.completed_at)


if __name__ == "__main__":
    unittest.main()
