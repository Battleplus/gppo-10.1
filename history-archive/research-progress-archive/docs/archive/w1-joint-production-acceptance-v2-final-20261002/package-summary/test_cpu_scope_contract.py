import unittest

from launch_once import verify_completed_run_accounting
from cpu_scope_contract import (
    CPU_SCOPE_SCHEMA,
    SAME_SCOPE_TOLERANCE_SECONDS,
    build_native_cpu_scope_breakdown,
    settlement_stage_remaining,
    settlement_wall_remaining,
    summarize_failed_export_cpu_evidence,
    validate_exported_cpu_scope,
    monitor_cpu_estimate,
)


def v6_synthetic_evidence():
    supervisor = {
        "status": "complete",
        "cpu_seconds": 11.283655,
        "cpu_process_self_seconds_cumulative": 1.2,
        "cpu_reaped_children_seconds_cumulative": 10.083655,
        "cpu_sampling_started_monotonic": 18396.7,
        "cpu_sampled_monotonic": 18417.870748527,
    }
    launcher = {
        "status": "supervisor_returned",
        "stage": "supervisor_and_worker",
        "native_launcher_process_cpu_seconds": 1.250638,
        "native_launcher_reaped_children_cpu_seconds": 10.033869,
        "native_launcher_complete_process_cpu_snapshot_seconds": 11.284507,
        "process_module_entry_monotonic": 18396.641737761,
        "sampled_monotonic": 18417.876281118,
    }
    sample = {
        "scope": "synthetic waited subprocess tree CPU fixture",
        "root_pid": 499,
        "sample_started_monotonic": 18396.582297516,
        "sample_ended_monotonic": 18418.132522405,
        "wait_completed": True,
        "termination": "normal_wait_or_launch_failure",
    }
    return supervisor, launcher, sample


class CpuScopeContractTests(unittest.TestCase):
    def test_monitor_charges_live_delta_and_reaped_short_child_once(self):
        result = monitor_cpu_estimate(
            {
                "supervisor": {"first_ticks": 100, "last_ticks": 160, "live": True},
                "worker": {"first_ticks": 200, "last_ticks": 240, "live": True},
            },
            100,
            0.30,
        )
        self.assertAlmostEqual(result["observed_process_tree_seconds"], 1.0)
        self.assertAlmostEqual(result["reaped_children_seconds"], 0.30)
        self.assertAlmostEqual(result["charged_seconds"], 1.30)
        self.assertTrue(result["scope_verified"])
        self.assertEqual(result["source"], "disjoint(live_process_deltas, reaped_children)")

    def test_monitor_reaped_snapshot_covers_unobserved_short_child(self):
        result = monitor_cpu_estimate(
            {"worker": {"first_ticks": 200, "last_ticks": 240, "live": True}},
            100,
            0.75,
        )
        self.assertAlmostEqual(result["observed_process_tree_seconds"], 0.4)
        self.assertAlmostEqual(result["charged_seconds"], 1.15)
        self.assertTrue(result["scope_verified"])

    def test_monitor_fails_closed_when_observed_process_disappears(self):
        result = monitor_cpu_estimate(
            {"short_child": {"first_ticks": 20, "last_ticks": 60, "live": False}},
            100,
            0.75,
        )
        self.assertAlmostEqual(result["observed_process_tree_seconds"], 0.0)
        self.assertAlmostEqual(result["charged_seconds"], 0.75)
        self.assertFalse(result["scope_verified"])
        self.assertEqual(result["unresolved_process_identities"], ["short_child"])

    def test_measured_post_snapshot_tail_is_settlement_and_tree_is_charged_once(self):
        supervisor, launcher, sample = v6_synthetic_evidence()
        scope = build_native_cpu_scope_breakdown(
            11.546687, supervisor, launcher, sample,
        )

        self.assertEqual(scope["schema"], CPU_SCOPE_SCHEMA)
        self.assertEqual(scope["status"], "reconciled")
        self.assertAlmostEqual(scope["native_launcher_exit_settlement_cpu_seconds"], 0.26218)
        self.assertAlmostEqual(
            scope["native_launcher_snapshot_to_parent_reap_wall_seconds"],
            sample["sample_ended_monotonic"] - launcher["sampled_monotonic"],
        )
        self.assertEqual(scope["native_process_tree_cpu_seconds_charged_once"], 11.546687)
        self.assertEqual(scope["native_launcher_exit_settlement_stage"], "settlement_and_verified_export")
        self.assertTrue(scope["native_launcher_exit_settlement_inclusive_in_tree"])

        cpu_left = settlement_stage_remaining(
            stage_cap=39.0, export_snapshot=0.278310513,
            exit_tail=scope["native_launcher_exit_settlement_cpu_seconds"],
        )
        wall_left = settlement_wall_remaining(
            stage_cap=38.0, export_snapshot=2.714298496,
            exit_tail=scope["native_launcher_snapshot_to_parent_reap_wall_seconds"],
        )
        self.assertAlmostEqual(cpu_left, 39.0 - 0.278310513 - 0.26218)
        self.assertAlmostEqual(
            wall_left,
            38.0 - 2.714298496 - scope["native_launcher_snapshot_to_parent_reap_wall_seconds"],
        )

        exported = {
            "native_process_tree_cpu_seconds": 11.546687,
            "supervisor_complete_process_cpu_seconds": 11.283655,
            "native_launcher_cpu_scope_breakdown": scope,
        }
        validated = validate_exported_cpu_scope(exported)
        self.assertAlmostEqual(validated["exit_tail_cpu_seconds"], 0.26218)

    def test_complete_budget_composition_uses_tree_once_and_debits_settlement_tail(self):
        supervisor, launcher, sample = v6_synthetic_evidence()
        scope = build_native_cpu_scope_breakdown(11.546687, supervisor, launcher, sample)
        cpu_remaining = 39.0 - 0.278310513 - 0.26218
        wall_tail = sample["sample_ended_monotonic"] - launcher["sampled_monotonic"]
        wall_remaining = 38.0 - 2.714298496 - wall_tail
        cpu_upper = 0.5 + 4.0 + 11.546687 + cpu_remaining + 1.0
        wall_upper = 3.0 + 28.0 + wall_remaining + 2.0
        export_status = {
            "attempt": "attempt-1",
            "status": "verified",
            "combined_budget_cpu_upper_bound_seconds": cpu_upper,
            "combined_budget_wall_seconds": 31.0,
            "combined_budget_wall_upper_bound_seconds": wall_upper,
            "windows_preflight_wall_seconds": 3.0,
            "linux_wall_snapshot_seconds": 28.0,
            "export_stage_wall_snapshot_seconds": 2.714298496,
            "export_stage_wall_cap_seconds": 38.0,
            "export_stage_wall_remaining_seconds": wall_remaining,
            "native_process_tree_cpu_seconds": 11.546687,
            "supervisor_complete_process_cpu_seconds": 11.283655,
            "native_launcher_cpu_scope_breakdown": scope,
            "linux_launcher_process_cpu_seconds": 4.0,
            "export_stage_cpu_snapshot_seconds": 0.278310513,
            "export_stage_cpu_cap_seconds": 39.0,
            "export_stage_cpu_remaining_seconds": cpu_remaining,
            "settlement_stage_elapsed_wall_seconds": 2.714298496 + wall_tail,
            "settlement_stage_complete_process_cpu_seconds": 0.278310513 + 0.26218,
            "windows_preflight_process_tree_cpu_seconds": 0.5,
        }
        result = verify_completed_run_accounting(
            {"totals": {"wall_seconds": 753, "complete_process_cpu_seconds": 551},
             "accounting_reserves": {"windows_post_preflight_process_cpu_seconds": 1,
                                     "cross_system_wall_seconds": 2}},
            {"attempt": "attempt-1"}, export_status,
            windows_preflight_cpu_seconds=0.5,
            windows_preflight_process_cpu_seconds=0.2,
            windows_total_cpu_seconds=0.3,
            windows_post_preflight_child_cpu_seconds=0.05,
            windows_total_wall_seconds=32.0,
        )
        self.assertEqual(result["status"], "pass")
        self.assertAlmostEqual(result["complete_process_cpu_upper_bound_seconds"], cpu_upper)
        self.assertAlmostEqual(result["export_stage_finalization_cpu_upper_bound_seconds"], cpu_remaining)
        self.assertAlmostEqual(result["cross_system_wall_seconds"], 1.0)
        self.assertNotAlmostEqual(result["complete_process_cpu_upper_bound_seconds"], cpu_upper + 11.283655)

    def test_same_scope_mismatch_over_fixed_tolerance_fails(self):
        supervisor, launcher, sample = v6_synthetic_evidence()
        supervisor["cpu_seconds"] += SAME_SCOPE_TOLERANCE_SECONDS + 0.001
        scope = build_native_cpu_scope_breakdown(
            11.546687, supervisor, launcher, sample,
        )
        self.assertEqual(scope["status"], "failed")
        self.assertIn("SUPERVISOR_SAME_SCOPE_COMPONENT_MISMATCH", scope["errors"])

        launcher["native_launcher_complete_process_cpu_snapshot_seconds"] += SAME_SCOPE_TOLERANCE_SECONDS + 0.001
        scope = build_native_cpu_scope_breakdown(
            11.546687, v6_synthetic_evidence()[0], launcher, sample,
        )
        self.assertIn("NATIVE_LAUNCHER_SAME_SCOPE_COMPONENT_MISMATCH", scope["errors"])

    def test_timeout_without_inner_snapshot_retains_waited_tree_cpu(self):
        sample = {
            "sample_started_monotonic": 10.0,
            "sample_ended_monotonic": 12.0,
            "wait_completed": True,
            "termination": "forced_process_group_stop",
        }
        scope = build_native_cpu_scope_breakdown(0.421, None, None, sample)
        self.assertEqual(scope["status"], "tree_only_timeout_or_early_exit")
        self.assertEqual(scope["native_process_tree_cpu_seconds_charged_once"], 0.421)
        self.assertIsNone(scope["native_launcher_exit_settlement_cpu_seconds"])

    def test_exception_and_export_failure_keep_known_subprocess_cpu(self):
        supervisor, launcher, sample = v6_synthetic_evidence()
        supervisor.update(cpu_seconds=0.10, cpu_process_self_seconds_cumulative=0.05,
                          cpu_reaped_children_seconds_cumulative=0.05,
                          cpu_sampling_started_monotonic=10.1, cpu_sampled_monotonic=10.5)
        launcher.update(native_launcher_process_cpu_seconds=0.06,
                        native_launcher_reaped_children_cpu_seconds=0.05,
                        native_launcher_complete_process_cpu_snapshot_seconds=0.11,
                        process_module_entry_monotonic=10.05, sampled_monotonic=10.6,
                        status="exception")
        sample.update(sample_started_monotonic=10.0, sample_ended_monotonic=10.8)
        scope = build_native_cpu_scope_breakdown(0.16, supervisor, launcher, sample)
        self.assertEqual(scope["status"], "reconciled")
        failed_export = {
            "attempt": "attempt-1",
            "status": "failed",
            "error": "synthetic export failure",
            "native_process_tree_cpu_seconds": 0.16,
            "supervisor_complete_process_cpu_seconds": 0.10,
            "native_launcher_cpu_scope_breakdown": scope,
            "settlement_stage_complete_process_cpu_seconds": 0.4,
            "combined_budget_cpu_upper_bound_seconds": 2.0,
            "combined_budget_wall_upper_bound_seconds": 3.0,
        }
        retained = summarize_failed_export_cpu_evidence(
            failed_export, expected_attempt="attempt-1",
        )
        self.assertEqual(retained["status"], "technical_stop")
        self.assertTrue(retained["attempt_matches"])
        self.assertEqual(retained["measured_cpu_evidence"]["native_process_tree_cpu_seconds"], 0.16)
        self.assertFalse(retained["automatic_retry"])

    def test_settlement_stage_caps_reject_tail_overrun(self):
        with self.assertRaisesRegex(ValueError, "SETTLEMENT_STAGE_CPU_CAP_EXCEEDED"):
            settlement_stage_remaining(stage_cap=1.0, export_snapshot=0.8, exit_tail=0.21)
        with self.assertRaisesRegex(ValueError, "SETTLEMENT_STAGE_WALL_CAP_EXCEEDED"):
            settlement_wall_remaining(stage_cap=1.0, export_snapshot=0.8, exit_tail=0.21)


if __name__ == "__main__":
    unittest.main()
