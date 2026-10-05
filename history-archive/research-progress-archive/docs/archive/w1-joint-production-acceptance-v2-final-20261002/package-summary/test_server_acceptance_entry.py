"""Local policy tests for the one-shot server acceptance boundary."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import acceptance_entry as acceptance
import launch_server_acceptance as launcher


ROOT = Path(__file__).resolve().parent


def request_fixture() -> dict:
    return json.loads((ROOT / "SERVER_ACCEPTANCE_REQUEST.json").read_text(encoding="utf-8"))


def authorization_fixture(request: dict, token: str) -> dict:
    return {
        "schema": acceptance.AUTH_SCHEMA,
        "status": "APPROVED",
        "engineering_job_id": request["engineering_job_id"],
        "remote_execution_root": request["remote_execution_root"],
        "evidence_directory": request["evidence_directory"],
        "server_acceptance_request_sha256": "1" * 64,
        "execution_manifest_sha256": "2" * 64,
        "hashes_sha256": "3" * 64,
        "authorization_id": "synthetic-authorization",
        "one_time_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "formal_research_attempt_created": False,
        "automatic_retry": False,
    }


class ServerAcceptanceRequestTests(TestCase):
    def test_request_remains_not_approved_and_fixes_four_test_scopes(self):
        request = request_fixture()
        acceptance.validate_request(request)
        self.assertEqual(request["status"], "NOT_APPROVED")
        self.assertFalse(request["automatic_retry"])
        self.assertEqual(request["limits"]["terminalization_reserve_cpu_seconds"], 3)
        self.assertEqual(request["limits"]["terminalization_reserve_wall_seconds"], 5)
        self.assertIn("excludes descendants", request["resource_measurement_scope"]["root_process_rss"])
        self.assertEqual(request["scope"]["supervised_phase_sync"]["commanded_cpu_work_seconds"], 0.09)

    def test_request_rejects_mutable_retry_and_test_scope(self):
        request = request_fixture()
        request["automatic_retry"] = True
        with self.assertRaisesRegex(acceptance.AcceptanceError, "NONRESEARCH_ONE_SHOT"):
            acceptance.validate_request(request)
        request = request_fixture()
        request["scope"]["supervised_phase_sync"]["synthetic_cpu_subprocesses"] = 4
        with self.assertRaisesRegex(acceptance.AcceptanceError, "TEST_SCOPE_MISMATCH"):
            acceptance.validate_request(request)

    def test_json_duplicate_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"status":"APPROVED","status":"NOT_APPROVED"}', encoding="utf-8")
            with self.assertRaisesRegex(acceptance.AcceptanceError, "JSON_DUPLICATE_FIELD"):
                acceptance._read_json(path)


class ServerAcceptanceAuthorizationTests(TestCase):
    def test_authorization_binds_request_package_and_token(self):
        request = request_fixture()
        authorization = authorization_fixture(request, "one-time-secret")
        authorization["server_acceptance_request_sha256"] = "a" * 64
        authorization_id = acceptance.validate_authorization(
            authorization, request=request, request_sha256="a" * 64,
            manifest_sha256="2" * 64, hashes_sha256="3" * 64,
            token=None, require_token=False,
        )
        self.assertEqual(authorization_id, "synthetic-authorization")
        with self.assertRaisesRegex(acceptance.AcceptanceError, "TOKEN_MISMATCH"):
            acceptance.validate_authorization(
                authorization, request=request, request_sha256="a" * 64,
                manifest_sha256="2" * 64, hashes_sha256="3" * 64,
                token="wrong-token",
            )

    def test_authorization_rejects_added_secret_fields_and_binding_drift(self):
        request = request_fixture()
        authorization = authorization_fixture(request, "token")
        authorization["token"] = "must-not-be-stored"
        with self.assertRaisesRegex(acceptance.AcceptanceError, "AUTHORIZATION_FIELDS_INVALID"):
            acceptance.validate_authorization(
                authorization, request=request, request_sha256="1" * 64,
                manifest_sha256="2" * 64, hashes_sha256="3" * 64,
                token=None, require_token=False,
            )
        authorization.pop("token")
        with self.assertRaisesRegex(acceptance.AcceptanceError, "BINDING_MISMATCH"):
            acceptance.validate_authorization(
                authorization, request=request, request_sha256="f" * 64,
                manifest_sha256="2" * 64, hashes_sha256="3" * 64,
                token=None, require_token=False,
            )


class ServerAcceptanceBudgetTests(TestCase):
    def test_exact_call_totals_include_750_sample_evaluations(self):
        limits = request_fixture()["limits"]["calls"]
        meter = acceptance.CallBudgetMeter(limits)
        meter.charge({
            "model_initializations_or_loads": 24,
            "world_batch_forwards": 150,
            "world_backward_calls": 18,
            "world_optimizer_updates": 18,
            "checkpoint_writes": 12,
            "checkpoint_loads": 12,
            "world_sample_evaluations": 750,
        })
        meter.record_cuda_context()
        meter.verify()
        self.assertEqual(meter.totals["world_sample_evaluations"], 750)

    def test_missing_sample_evaluation_call_fails_exact_total(self):
        limits = request_fixture()["limits"]["calls"]
        meter = acceptance.CallBudgetMeter(limits)
        meter.charge({
            "model_initializations_or_loads": 24,
            "world_batch_forwards": 150,
            "world_backward_calls": 18,
            "world_optimizer_updates": 18,
            "checkpoint_writes": 12,
            "checkpoint_loads": 12,
            "world_sample_evaluations": 749,
        })
        meter.record_cuda_context()
        with self.assertRaisesRegex(acceptance.AcceptanceError, "world_sample_evaluations"):
            meter.verify()

    def test_resource_meter_uses_entry_wall_and_full_process_cpu_and_closing_charge(self):
        limits = request_fixture()["limits"]
        now = [12.0]
        meter = acceptance.ResourceMeter(
            limits, started_wall=10.0, clock=lambda: now[0],
            cpu_provider=lambda: 5.0, rss_provider=lambda: 64,
        )
        sample = meter.check_closing_bound()
        self.assertEqual(sample["wall_seconds"], 2.0)
        self.assertEqual(sample["complete_process_cpu_seconds"], 5.0)
        self.assertEqual(sample["root_process_rss_high_water_bytes"], 64)
        self.assertIn("excludes descendants", sample["rss_measurement_scope"])
        self.assertIn("waited/reaped descendants", sample["complete_process_cpu_scope"])
        self.assertEqual(sample["charged_upper_wall_seconds"], 7.0)
        self.assertEqual(sample["charged_upper_complete_process_cpu_seconds"], 8.0)

    def test_resource_meter_rejects_negative_or_nonfinite_samples(self):
        limits = request_fixture()["limits"]
        for cpu_value in (-0.1, float("nan"), float("inf"), "not-a-number", True):
            meter = acceptance.ResourceMeter(
                limits, started_wall=0.0, clock=lambda: 1.0,
                cpu_provider=lambda value=cpu_value: value, rss_provider=lambda: 1,
            )
            with self.assertRaisesRegex(acceptance.AcceptanceError, "RESOURCE_SAMPLE_INVALID"):
                meter.sample()
        meter = acceptance.ResourceMeter(
            limits, started_wall=2.0, clock=lambda: 1.0,
            cpu_provider=lambda: 1.0, rss_provider=lambda: 1,
        )
        with self.assertRaisesRegex(acceptance.AcceptanceError, "RESOURCE_SAMPLE_INVALID"):
            meter.sample()
        meter = acceptance.ResourceMeter(
            limits, started_wall=0.0, clock=lambda: "not-a-time",
            cpu_provider=lambda: 1.0, rss_provider=lambda: 1,
        )
        with self.assertRaisesRegex(acceptance.AcceptanceError, "RESOURCE_SAMPLE_INVALID"):
            meter.sample()

    def test_cpu_meter_adds_waited_child_usage(self):
        fake_resource = SimpleNamespace(
            RUSAGE_SELF=1,
            RUSAGE_CHILDREN=2,
            getrusage=lambda selector: (
                SimpleNamespace(ru_utime=1.0, ru_stime=0.25)
                if selector == 1 else SimpleNamespace(ru_utime=0.5, ru_stime=0.125)
            ),
        )
        with patch.dict(sys.modules, {"resource": fake_resource}):
            self.assertEqual(acceptance._cpu_seconds(), 1.875)


class ServerAcceptanceGpuTests(TestCase):
    def test_framework_high_water_checks_do_not_query_nvidia_or_run_forwards(self):
        gpu = request_fixture()["gpu"]
        process_queries = []
        memory = {"allocated": 1024, "reserved": 2048}
        torch = SimpleNamespace(cuda=SimpleNamespace(
            max_memory_allocated=lambda _device: memory["allocated"],
            max_memory_reserved=lambda _device: memory["reserved"],
        ))
        meter = acceptance.GpuResourceMeter(
            gpu,
            lambda *_args, **_kwargs: process_queries.append("nvidia-query"),
            expected_uuid="GPU-fixture",
            pid_provider=lambda: 123,
        )
        meter.bind_torch(torch)
        self.assertEqual(meter.check_framework_memory(), {
            "cuda_max_memory_allocated_bytes": 1024,
            "cuda_max_memory_reserved_bytes": 2048,
        })
        memory["reserved"] = gpu["peak_allocated_memory_bytes"] + 1
        with self.assertRaisesRegex(acceptance.AcceptanceError, "FRAMEWORK_MEMORY_LIMIT"):
            meter.check_framework_memory()
        self.assertEqual(process_queries, [])

    def test_phase_snapshots_require_exclusive_owned_process_under_ceiling(self):
        gpu = request_fixture()["gpu"]
        snapshots = []
        snapshot_value = {
            "gpu_physical_device": 1,
            "gpu_uuid": "GPU-fixture",
            "gpu_owned_compute_process_pids": [123],
            "gpu_owned_process_memory_bytes": 4096,
            "gpu_exclusive_allocation_observed": True,
        }

        def snapshot(*_args, **_kwargs):
            snapshots.append(dict(snapshot_value))
            return dict(snapshot_value)

        meter = acceptance.GpuResourceMeter(
            gpu, snapshot, expected_uuid="GPU-fixture", pid_provider=lambda: 123,
        )
        meter.check_process_snapshot("joint_pipeline", "start")
        meter.check_process_snapshot("joint_pipeline", "end")
        self.assertEqual(len(snapshots), 2)
        snapshot_value["gpu_owned_compute_process_pids"] = [123, 456]
        with self.assertRaisesRegex(acceptance.AcceptanceError, "EXCLUSIVITY_OR_MEMORY"):
            meter.check_process_snapshot("joint_pipeline", "end")

    def test_gpu_memory_guard_runs_only_at_production_call_boundaries(self):
        import budget_ledger

        events = []
        limits = request_fixture()["limits"]
        call_meter = acceptance.CallBudgetMeter(limits["calls"])
        resource_meter = acceptance.ResourceMeter(
            limits, started_wall=0.0, clock=lambda: 1.0,
            cpu_provider=lambda: 0.0, rss_provider=lambda: 1,
        )
        base_call = lambda _ledger, _name, _amounts, operation, *args, **kwargs: operation(*args, **kwargs)
        with patch.object(budget_ledger.BudgetLedger, "call", base_call):
            restore = acceptance.install_realtime_call_guard(
                call_meter, resource_meter,
                gpu_memory_check=lambda: events.append("gpu-boundary"),
            )
            try:
                value = budget_ledger.BudgetLedger.call(
                    object(), "forward", {"world_batch_forwards": 1},
                    lambda: (events.append("operation") or "result"),
                )
                budget_ledger.BudgetLedger.call(
                    object(), "non-model", {}, lambda: events.append("non-model-operation"),
                )
            finally:
                restore()
        self.assertEqual(value, "result")
        self.assertEqual(events, ["gpu-boundary", "operation", "gpu-boundary",
                                  "non-model-operation"])


class ServerAcceptanceRegistrationTests(TestCase):
    def test_register_running_gpu_binding_then_success_is_ordered(self):
        events = []
        result = {"ok": True}
        job, returned = acceptance.run_registered_once(
            register=lambda: (events.append("REGISTERED") or {"job_id": "job-1"}),
            mark_cpu_running=lambda _job: events.append("CPU_RUNNING"),
            initialize_gpu_and_bind=lambda _job: events.append("GPU_BOUND"),
            run_workload=lambda _job: (events.append("WORKLOAD") or result),
            append_terminal=lambda _job, event, _code, _reason: events.append(event),
        )
        self.assertEqual(job["job_id"], "job-1")
        self.assertIs(returned, result)
        self.assertEqual(events, ["REGISTERED", "CPU_RUNNING", "GPU_BOUND", "WORKLOAD", "SUCCEEDED"])

    def test_workload_timeout_records_expired_and_never_succeeds(self):
        events = []
        with self.assertRaisesRegex(TimeoutError, "fixture"):
            acceptance.run_registered_once(
                register=lambda: {"job_id": "job-2"},
                mark_cpu_running=lambda _job: events.append("CPU_RUNNING"),
                initialize_gpu_and_bind=lambda _job: events.append("GPU_BOUND"),
                run_workload=lambda _job: (_ for _ in ()).throw(TimeoutError("fixture")),
                append_terminal=lambda _job, event, _code, _reason: events.append(event),
            )
        self.assertEqual(events, ["CPU_RUNNING", "GPU_BOUND", "EXPIRED"])

    def test_registration_failure_does_not_create_terminal_event(self):
        events = []
        with self.assertRaisesRegex(RuntimeError, "registration"):
            acceptance.run_registered_once(
                register=lambda: (_ for _ in ()).throw(RuntimeError("registration")),
                mark_cpu_running=lambda _job: events.append("CPU_RUNNING"),
                initialize_gpu_and_bind=lambda _job: events.append("GPU_BOUND"),
                run_workload=lambda _job: None,
                append_terminal=lambda *_args: events.append("TERMINAL"),
            )
        self.assertEqual(events, [])

    def test_manifest_covers_export_and_true_terminal_result(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            (evidence / "joint-controlled-export").mkdir()
            (evidence / "joint-controlled-export" / "metadata.json").write_text("{}\n", encoding="utf-8")
            (evidence / "acceptance-result.json").write_text('{"run_status":"passed"}\n', encoding="utf-8")
            result = {"engineering_job_id": "engineering-fixture", "run_status": "passed"}
            manifest = acceptance._manifest_artifacts(evidence, result)
            self.assertTrue(manifest["verified"])
            self.assertIn("joint-controlled-export/metadata.json", manifest["files"])
            self.assertIn("acceptance-result.json", manifest["files"])
            self.assertNotIn("acceptance-evidence-manifest.json", manifest["files"])


class ServerAcceptanceLauncherTests(TestCase):
    def test_unmeasured_remote_transport_cpu_blocks_overall_complete(self):
        status, aggregate_budget_pass = launcher._completion_gate(
            remote_exit_code=0,
            remote_status="complete",
            evidence_received=True,
            controller_budget_pass=True,
            server_budget_pass=True,
            aggregate_budget_arithmetic_pass=True,
            remote_transport_cpu_measured=False,
        )
        self.assertEqual(status, "technical_stop")
        self.assertFalse(aggregate_budget_pass)

    def test_verified_remote_transport_cpu_allows_successful_settlement(self):
        status, aggregate_budget_pass = launcher._completion_gate(
            remote_exit_code=0,
            remote_status="complete",
            evidence_received=True,
            controller_budget_pass=True,
            server_budget_pass=True,
            aggregate_budget_arithmetic_pass=True,
            remote_transport_cpu_measured=True,
        )
        self.assertEqual(status, "complete")
        self.assertTrue(aggregate_budget_pass)

    def test_structure_only_never_connects_or_reads_stdin_token(self):
        output = io.StringIO()
        with (patch.object(launcher, "verify_structure", return_value={
                    "identity": {"manifest_sha256": "a" * 64, "hashes_sha256": "b" * 64},
                    "request_sha256": "c" * 64,
                }),
              patch.object(launcher.controller, "connect", side_effect=AssertionError("SSH used")),
              patch.object(launcher, "_read_one_time_token", side_effect=AssertionError("token read")),
              contextlib.redirect_stdout(output)):
            for flag in ("--structure-only", "--preflight-only"):
                code = launcher.main([flag])
                self.assertEqual(code, 0)
                payload = json.loads(output.getvalue().splitlines()[-1])
                self.assertFalse(payload["formal_authorization_validated"])
                self.assertFalse(payload["authorization_consumed"])
                self.assertFalse(payload["worker_started"])

    def test_remote_target_preflight_rejects_existing_engineering_path(self):
        checked = {"contract": {"native_python": "/python"},
                   "request": {"remote_execution_root": "/run-once", "evidence_directory": "/evidence"}}
        result = {"exit_code": 0, "stdout": json.dumps({
            "/run-once": {"exists": False, "symlink": False},
            "/evidence": {"exists": True, "symlink": False},
        }), "stderr": ""}
        with patch.object(launcher.controller, "execute", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "PATH_ALREADY_EXISTS"):
                launcher.remote_target_preflight(object(), checked)


if __name__ == "__main__":
    import unittest
    unittest.main()
