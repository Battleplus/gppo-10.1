import unittest

from launch_once import verify_completed_run_accounting


REQUEST = {
    "totals": {"wall_seconds": 753, "complete_process_cpu_seconds": 551},
    "accounting_reserves": {
        "windows_post_preflight_process_cpu_seconds": 1,
        "cross_system_wall_seconds": 2,
    },
}
CONTRACT = {"attempt": "attempt-v5"}


def status(**updates):
    payload = {
        "attempt": "attempt-v5",
        "status": "verified",
        "combined_budget_cpu_upper_bound_seconds": 48.3,
        "combined_budget_wall_seconds": 40.0,
        "combined_budget_wall_upper_bound_seconds": 72.0,
        "windows_preflight_wall_seconds": 10.0,
        "linux_wall_snapshot_seconds": 30.0,
        "export_stage_wall_snapshot_seconds": 8.0,
        "export_stage_wall_cap_seconds": 38.0,
        "native_process_tree_cpu_seconds": 10.0,
        "supervisor_complete_process_cpu_seconds": 10.01,
        "linux_launcher_process_cpu_seconds": 0.5,
        "export_stage_cpu_snapshot_seconds": 3.0,
        "export_stage_cpu_cap_seconds": 39.0,
        "windows_preflight_process_tree_cpu_seconds": 0.8,
    }
    payload.update(updates)
    return payload


class LaunchAccountingTests(unittest.TestCase):
    def test_tree_cpu_is_counted_once_and_supervisor_value_is_only_a_check(self):
        result = verify_completed_run_accounting(
            REQUEST, CONTRACT, status(),
            windows_preflight_cpu_seconds=0.8,
            windows_preflight_process_cpu_seconds=0.4,
            windows_total_cpu_seconds=0.8,
            windows_post_preflight_child_cpu_seconds=0.05,
            windows_total_wall_seconds=41.0,
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["native_process_tree_cpu_seconds"], 10.0)
        self.assertEqual(result["supervisor_complete_process_cpu_seconds_for_consistency_only"], 10.01)
        self.assertEqual(result["windows_post_preflight_wsl_process_cpu_seconds"], 0.05)
        self.assertAlmostEqual(result["complete_process_cpu_upper_bound_seconds"], 48.3)
        self.assertNotAlmostEqual(result["complete_process_cpu_upper_bound_seconds"], 58.31)

    def test_tree_counter_disagreement_fails_closed(self):
        with self.assertRaisesRegex(Exception, "LINUX_PROCESS_TREE_CPU_MEASUREMENT_MISMATCH"):
            verify_completed_run_accounting(
                REQUEST, CONTRACT,
                status(supervisor_complete_process_cpu_seconds=10.2),
                windows_preflight_cpu_seconds=0.8,
                windows_preflight_process_cpu_seconds=0.4,
                windows_total_cpu_seconds=0.8,
                windows_post_preflight_child_cpu_seconds=0.05,
                windows_total_wall_seconds=41.0,
            )

    def test_stage_caps_and_upper_bound_composition_are_verified(self):
        with self.assertRaisesRegex(Exception, "EXPORT_STAGE_CPU_SNAPSHOT_OVER_CAP"):
            verify_completed_run_accounting(
                REQUEST, CONTRACT,
                status(export_stage_cpu_snapshot_seconds=39.1),
                windows_preflight_cpu_seconds=0.8,
                windows_preflight_process_cpu_seconds=0.4,
                windows_total_cpu_seconds=0.8,
                windows_post_preflight_child_cpu_seconds=0.05,
                windows_total_wall_seconds=41.0,
            )
        with self.assertRaisesRegex(Exception, "LINUX_CPU_UPPER_BOUND_COMPOSITION_MISMATCH"):
            verify_completed_run_accounting(
                REQUEST, CONTRACT,
                status(combined_budget_cpu_upper_bound_seconds=40.0),
                windows_preflight_cpu_seconds=0.8,
                windows_preflight_process_cpu_seconds=0.4,
                windows_total_cpu_seconds=0.8,
                windows_post_preflight_child_cpu_seconds=0.05,
                windows_total_wall_seconds=41.0,
            )

    def test_preflight_identity_wall_budget_and_attempt_fail_closed(self):
        with self.assertRaisesRegex(Exception, "WINDOWS_PREFLIGHT_CPU_IDENTITY_MISMATCH"):
            verify_completed_run_accounting(
                REQUEST, CONTRACT,
                status(),
                windows_preflight_cpu_seconds=0.9,
                windows_preflight_process_cpu_seconds=0.4,
                windows_total_cpu_seconds=0.8,
                windows_post_preflight_child_cpu_seconds=0.05,
                windows_total_wall_seconds=41.0,
            )
        with self.assertRaisesRegex(Exception, "COMBINED_WALL_UPPER_BOUND_EXCEEDED"):
            verify_completed_run_accounting(
                REQUEST, CONTRACT,
                status(
                    combined_budget_wall_seconds=730.0,
                    windows_preflight_wall_seconds=700.0,
                    combined_budget_wall_upper_bound_seconds=762.0,
                ),
                windows_preflight_cpu_seconds=0.8,
                windows_preflight_process_cpu_seconds=0.4,
                windows_total_cpu_seconds=0.8,
                windows_post_preflight_child_cpu_seconds=0.05,
                windows_total_wall_seconds=754.0,
            )
        with self.assertRaisesRegex(Exception, "EXPORTED_ACCOUNTING_ATTEMPT_MISMATCH"):
            verify_completed_run_accounting(
                REQUEST, CONTRACT,
                status(attempt="other"),
                windows_preflight_cpu_seconds=0.8,
                windows_preflight_process_cpu_seconds=0.4,
                windows_total_cpu_seconds=0.8,
                windows_post_preflight_child_cpu_seconds=0.05,
                windows_total_wall_seconds=41.0,
            )

    def test_windows_post_preflight_cpu_and_actual_wall_caps_are_enforced(self):
        result = verify_completed_run_accounting(
            REQUEST, CONTRACT, status(),
            windows_preflight_cpu_seconds=0.8,
            windows_preflight_process_cpu_seconds=0.4,
            windows_total_cpu_seconds=1.6,
            windows_post_preflight_child_cpu_seconds=0.05,
            windows_total_wall_seconds=754.0,
        )
        self.assertEqual(result["status"], "technical_stop")
        self.assertIn("windows_post_preflight_cpu_reserve_exceeded", result["failures"])
        self.assertIn("cross_system_wall_reserve_exceeded", result["failures"])
        self.assertIn("complete_wall_budget_exceeded", result["failures"])


if __name__ == "__main__":
    unittest.main()
