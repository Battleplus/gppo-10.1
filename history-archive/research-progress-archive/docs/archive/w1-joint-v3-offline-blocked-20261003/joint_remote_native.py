"""Remote one-shot supervisor, registration binding, and controlled export."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import runpy
import shutil
import subprocess
import sys
import time
from registration_identity import derive_name_id
from server_registration import RegistrationError, wait_for_gpu_binding

GPU_QUERY_CPU_SECONDS = 0.0


def _gpu_query(argv):
    global GPU_QUERY_CPU_SECONDS
    before = _usage_cpu(resource.RUSAGE_CHILDREN)
    try:
        return subprocess.run(argv, check=False, capture_output=True, text=True, timeout=10)
    finally:
        GPU_QUERY_CPU_SECONDS += max(0.0, _usage_cpu(resource.RUSAGE_CHILDREN) - before)


def _usage_cpu(which) -> float:
    usage = resource.getrusage(which)
    return float(usage.ru_utime + usage.ru_stime)


def require_outer_accounting_parent(root):
    parent = os.getppid()
    if os.environ.get("W1_OUTER_ACCOUNTING_PID") != str(parent):
        raise RuntimeError("WAITED_LIFETIME_ACCOUNTING_PARENT_REQUIRED")
    command = (Path("/proc") / str(parent) / "cmdline").read_bytes().split(b"\0")
    if str(root / "metered_joint_entry.py").encode() not in command:
        raise RuntimeError("OUTER_ACCOUNTING_ENTRY_IDENTITY_MISMATCH")


def _record_failure_settlement(
    status: dict,
    *,
    error: str,
    started: float,
    child_cpu_start: float | None,
) -> dict:
    """Record a fail-closed status with the CPU and wall work seen so far."""
    own_cpu = _usage_cpu(resource.RUSAGE_SELF)
    children_total = _usage_cpu(resource.RUSAGE_CHILDREN)
    child_cpu = children_total
    status.update(
        status="technical_stop",
        error=error,
        settlement_process_self_cpu_seconds=own_cpu,
        settlement_reaped_children_cpu_seconds=child_cpu,
        settlement_process_cpu_seconds=own_cpu + child_cpu,
        settlement_wall_seconds=max(0.0, time.monotonic() - started),
        final_resource_pass=False,
        automatic_retry=False,
    )
    return status


def gpu_ids_for_pids(pids):
    """Return GPUs reported by nvidia-smi for the supplied process IDs only."""
    registered = {int(pid) for pid in pids}
    if not registered:
        return []
    gpu_rows = _gpu_query(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
    )
    if gpu_rows.returncode:
        raise RuntimeError("GPU_REGISTRATION_QUERY_FAILED")
    uuid_to_index = {}
    for line in gpu_rows.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            uuid_to_index[parts[1]] = int(parts[0])
    process_rows = _gpu_query(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
    )
    if process_rows.returncode:
        raise RuntimeError("GPU_REGISTRATION_PROCESS_QUERY_FAILED")
    owned = set()
    for line in process_rows.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if (len(parts) == 2 and parts[0] in uuid_to_index and parts[1].isdigit()
                and int(parts[1]) in registered):
            owned.add(uuid_to_index[parts[0]])
    return sorted(owned)


def registration_details(root, request, runtime_snapshot):
    gpu = request.get("gpu") if isinstance(request.get("gpu"), dict) else {}
    gpu_ids = [int(gpu["physical_device"])] if isinstance(gpu.get("physical_device"), int) else []
    totals = request.get("totals", {})
    return {
        "task_name": "W1 complete-action-consequence G1/G2 qualification",
        "purpose": "Collect complete fixed-continuation labels and run frozen G1/G2 prediction evaluation.",
        "workdir": str(root),
        "training_files": ["joint_pipeline.py", "production_world.py", "runner.py"],
        "dataset": "frozen source evidence and approved W1 parent split",
        "model": "G1/G2 action-conditioned Graph-JEPA routes",
        "command_redacted": "python -B joint_remote_native.py --authorization-file <external-authorization>",
        "gpu_request": {
            "ids": gpu_ids, "count": len(gpu_ids),
            "vram_estimate_gb": str(round(int(gpu.get("peak_allocated_memory_bytes", 0)) / 1024**3, 2)),
        },
        "cpu_cores_estimate": "4",
        "ram_estimate_gb": "4",
        "disk_growth_estimate_gb": str(round(int(totals.get("aggregate_native_plus_verified_export_bytes", 0)) / 1024**3, 2)),
        "duration_estimate": f"<= {totals.get('wall_seconds', 'unknown')} seconds",
        "estimate_basis": "Frozen RESOURCE_REQUEST totals and four-core native supervisor guard; benchmark is still required before final approval.",
        "log_path": str(root / "console.log"),
        "checkpoint_path": str(root / "run-once" / "world-model-checkpoints"),
        "resource_snapshot": {
            "captured_before_worker_launch": True,
            "runtime_preflight": runtime_snapshot,
            "cpu_count": os.cpu_count(),
            "disk_free_bytes": shutil.disk_usage(root).free,
            "memory_available_bytes": next(
                int(line.split()[1]) * 1024
                for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
                if line.startswith("MemAvailable:")
            ),
        },
    }


def wait_for_worker_binding(root: Path, supervisor_pid: int, child: subprocess.Popen,
                            timeout_seconds: float) -> dict:
    """Wait for production supervisor status to expose the actual runner PID."""
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    status_path = root / "supervisor-status.json"
    while time.monotonic() < deadline:
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                status = None
            if isinstance(status, dict):
                runner_pid = status.get("pid")
                recorded_supervisor = status.get("supervisor_pid")
                if (isinstance(runner_pid, int) and runner_pid > 0
                        and recorded_supervisor == supervisor_pid):
                    return status
        if child.poll() is not None:
            break
        time.sleep(0.1)
    raise RuntimeError("PRODUCTION_RUNNER_PID_BINDING_UNAVAILABLE")


def wait_for_learning_binding(root, child, pids, requested_gpu_ids, request, started):
    """The root's registered ancestry covers CPU collection before a CUDA context exists."""
    if not requested_gpu_ids:
        return []
    deadline = started + request["totals"]["wall_seconds"]
    while time.monotonic() < deadline:
        path = root / "supervisor-status.json"
        state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if state.get("stage") in ("world_model_training_and_selection", "prediction_confirmation"):
            return wait_for_gpu_binding(pids, requested_gpu_ids,
                min(30.0, max(0.0, deadline - time.monotonic())), readonly_gpu_query=gpu_ids_for_pids)
        if child.poll() is not None:
            # Data-gate stop never constructs a model or a CUDA context.
            return None
        if state.get("wall_seconds", 0.0) > request["stages"]["label_qualification"]["wall_seconds"] + request["stages"]["staging_and_zero_step_gate"]["wall_seconds"]:
            raise RegistrationError("GPU_REGISTRATION_BINDING_TIMEOUT")
        time.sleep(0.1)
    raise RegistrationError("GPU_REGISTRATION_BINDING_TIMEOUT")


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-file", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--real-name")
    parser.add_argument("--name-id")
    args = parser.parse_args()
    if not args.preflight_only:
        if not args.real_name:
            raise RuntimeError("REAL_NAME_REQUIRED_BEFORE_SERVER_WORKLOAD")
        generated_name_id = derive_name_id(args.real_name)
        if args.name_id is not None and args.name_id != generated_name_id:
            raise RuntimeError("NAME_ID_DOES_NOT_MATCH_DERIVED_REAL_NAME")
        args.name_id = generated_name_id
    root = Path(__file__).resolve().parent
    probe = runpy.run_path(str(root / "remote_runtime_preflight.py"))
    probe["verify_process_isolation"]()
    runtime = probe["verify"](json.loads((root / "remote-runtime-identity.json").read_text(encoding="utf-8")))
    sys.path.insert(0, str(root))
    from manifest_contract import verify_external_authorization, sha256_file
    from joint_inputs import verify_joint_inputs
    from infra_io import durable_atomic_json, controlled_export, require_native_linux_filesystem
    token = None if args.preflight_only else sys.stdin.readline().rstrip("\r\n")
    checked = verify_external_authorization(root, args.authorization_file, token=token,
                                            preflight_only=args.preflight_only)
    contract, request = checked["contract"], checked["request"]
    if sha256_file(root / "remote-runtime-identity.json") != contract["runtime_identity_sha256"]:
        raise RuntimeError("REMOTE_RUNTIME_BINDING_MISMATCH")
    if contract.get("integration_test") or root.as_posix() != contract["native_execution_root"]:
        raise RuntimeError("FORMAL_REMOTE_ENTRY_IDENTITY_OR_TEST_BACKEND_REJECTED")
    require_native_linux_filesystem(root)
    source = verify_joint_inputs(root)
    for path in ("execution.lock", "run-once", "supervisor-status.json", "runtime-output"):
        if (root / path).exists():
            raise RuntimeError("REMOTE_ATTEMPT_ALREADY_CONSUMED:" + path)
    if args.preflight_only:
        print(json.dumps({"status": "pass", "runtime": runtime, "source": source,
                          "staging_started": False, "worker_started": False,
                          "authorization_consumed": False}))
        return 0
    if not request["runner_ready"]:
        raise RuntimeError("PRODUCTION_ENGINEERING_ACCEPTANCE_REQUIRED")
    require_outer_accounting_parent(root)
    if not isinstance(args.real_name, str) or not isinstance(args.name_id, str):
        raise RuntimeError("REAL_NAME_AND_NAME_ID_REQUIRED_BEFORE_SERVER_WORKLOAD")

    from server_registration import append_running, append_terminal, register_before_launch
    registration_job = None
    registration_terminal = False
    registration_error = None
    child = None
    child_before = None
    child_cpu_start = None
    terminal_event = None
    status = {"schema": "w1-joint-native-final/1.0.0", "attempt": contract["attempt"],
              "status": "technical_stop", "automatic_retry": False}

    def finish_registration(event, exit_code, reason):
        nonlocal registration_terminal, registration_error
        if registration_job is None or registration_terminal:
            return
        if os.environ.get("W1_OUTER_ACCOUNTING_PID") == str(os.getppid()):
            status["registration_job_id"] = registration_job["job_id"]
            status["requested_terminal_event"] = event
            status["registration_terminal_deferred_to_waiting_parent"] = True
            registration_terminal = True
            return
        try:
            append_terminal(registration_job["job_id"], event, exit_code=exit_code,
                            reason_redacted=reason, result_path=str(root) + "-export")
            registration_terminal = True
        except BaseException as exc:
            registration_error = type(exc).__name__ + ": " + str(exc)

    try:
        budget_binding = {"totals": request.get("totals", {}), "stages": request.get("stages", {})}
        registration_job = register_before_launch(
            real_name=args.real_name, name_id=args.name_id,
            details=registration_details(root, request, runtime), attempt=contract["attempt"],
            manifest_sha256=checked["identity"]["manifest_sha256"],
            hashes_sha256=checked["identity"]["hashes_sha256"],
            resource_request_sha256=contract["resource_request_sha256"], budget=budget_binding,
        )
        durable_atomic_json(Path(str(root) + "-registration-handoff.json"), {
            "attempt": contract["attempt"], "registration_job_id": registration_job["job_id"],
            "outer_pid": os.getppid(), "inner_pid": os.getpid(),
            "manifest_sha256": checked["identity"]["manifest_sha256"],
            "hashes_sha256": checked["identity"]["hashes_sha256"],
        })
        from registration_start_handshake import register_then_wait_for_start_ack
        register_then_wait_for_start_ack(
            register=lambda: registration_job, real_name=args.real_name, name_id=args.name_id,
            registration_path="/media/abc_disk/admin123/lbh/dengji.txt",
            manifest_sha256=checked["identity"]["manifest_sha256"],
            hashes_sha256=checked["identity"]["hashes_sha256"], budget=budget_binding,
            stdin=sys.stdin, stdout=sys.stdout,
            timeout_seconds=min(30.0, request["stages"]["staging_and_zero_step_gate"]["wall_seconds"]),
        )
        # Register the remote root before creating any supervisor or worker.
        append_running(registration_job["job_id"], pid=os.getpid(), pgid=os.getpgrp(),
                       worker_pids=[], actual_gpu_ids=[], readonly_gpu_query=gpu_ids_for_pids,
                       log_path=str(root / "console.log"))
        (root / "runtime-output").mkdir()
        child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        child_cpu_start = child_before.ru_utime + child_before.ru_stime
        staging_cpu_usage = resource.getrusage(resource.RUSAGE_SELF)
        staging_cpu = staging_cpu_usage.ru_utime + staging_cpu_usage.ru_stime + _usage_cpu(resource.RUSAGE_CHILDREN)
        gpu_queries_before_worker = GPU_QUERY_CPU_SECONDS
        staging_wall = time.monotonic() - started
        stage_cap = request["stages"]["staging_and_zero_step_gate"]
        if staging_cpu > stage_cap["complete_process_cpu_seconds"] or staging_wall > stage_cap["wall_seconds"]:
            raise RuntimeError("REMOTE_NATIVE_STAGING_BUDGET_EXCEEDED")
        env = {**os.environ, "W1_VERIFIED_ATTEMPT": contract["attempt"],
               "W1_VERIFIED_MANIFEST_SHA256": checked["identity"]["manifest_sha256"],
               "W1_VERIFIED_HASHES_SHA256": checked["identity"]["hashes_sha256"],
               "PYTHONNOUSERSITE": "1"}
        child = subprocess.Popen(
            [sys.executable, "-I", "-B", str(root / "worker_bootstrap.py"),
             str(os.getpid()), str(root / "joint_supervisor_entry.py")],
            cwd=root, env=env, stdin=subprocess.DEVNULL,
        )
        binding_timeout = min(
            30.0, max(0.0, request["totals"]["wall_seconds"] - (time.monotonic() - started))
        )
        try:
            supervisor_status = wait_for_worker_binding(
                root, child.pid, child,
                binding_timeout,
            )
        except RuntimeError as exc:
            if str(exc) == "PRODUCTION_RUNNER_PID_BINDING_UNAVAILABLE":
                terminal_event = "EXPIRED"
            raise
        runner_pid = supervisor_status["pid"]
        gpu_request = request.get("gpu")
        requested_gpu_ids = []
        if isinstance(gpu_request, dict) and isinstance(gpu_request.get("physical_device"), int):
            requested_gpu_ids = [gpu_request["physical_device"]]
        try:
            actual_gpu_ids = wait_for_learning_binding(
                root, child, (os.getpid(), child.pid, runner_pid), requested_gpu_ids, request, started)
        except RegistrationError as exc:
            if str(exc) == "GPU_REGISTRATION_BINDING_TIMEOUT":
                terminal_event = "EXPIRED"
            raise
        if actual_gpu_ids is not None:
            append_running(registration_job["job_id"], pid=os.getpid(), pgid=os.getpgrp(),
                           worker_pids=[child.pid, runner_pid], actual_gpu_ids=actual_gpu_ids,
                           readonly_gpu_query=gpu_ids_for_pids, log_path=str(root / "console.log"),
                           restarted=True)
        else:
            status["gpu_binding"] = "not_evaluated_no_learning_context_before_child_exit"
        try:
            child.wait(timeout=max(0.0, request["totals"]["wall_seconds"] - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            terminal_event = "EXPIRED"
            raise
        waited = resource.getrusage(resource.RUSAGE_CHILDREN)
        inclusive_cpu = waited.ru_utime + waited.ru_stime - child_before.ru_utime - child_before.ru_stime
        supervisor = json.loads((root / "supervisor-status.json").read_text(encoding="utf-8"))
        snapshot = json.loads((root / "runtime-output" / "native-launcher-accounting.json").read_text(encoding="utf-8"))
        lower = snapshot["native_launcher_complete_process_cpu_snapshot_seconds"]
        post_launch_query_cpu = GPU_QUERY_CPU_SECONDS - gpu_queries_before_worker
        tail = inclusive_cpu - post_launch_query_cpu - lower
        if tail < -0.05:
            raise RuntimeError("CPU_ORDERING_CONTRACT_FAILED")
        same_scope = abs(lower - snapshot["native_launcher_process_cpu_seconds"] - snapshot["native_launcher_reaped_children_cpu_seconds"])
        if same_scope > 0.05:
            raise RuntimeError("CPU_SAME_SCOPE_COMPONENT_MISMATCH")
        status.update(status="complete" if child.returncode == 0 else "technical_stop",
                      supervisor=supervisor, staging_cpu_seconds=staging_cpu,
                      staging_wall_seconds=staging_wall,
                      native_child_inclusive_cpu_seconds=inclusive_cpu,
                      post_snapshot_interval_cpu_seconds=max(0.0, tail),
                      gpu_query_cpu_before_worker_seconds=gpu_queries_before_worker,
                      gpu_query_cpu_after_worker_launch_seconds=post_launch_query_cpu,
                      gpu_query_accounting_rule="waited query helpers measured separately; prelaunch charged to staging, binding queries charged to staging",
                      snapshot_cpu_seconds=lower,
                      cpu_same_scope_component_difference_seconds=same_scope,
                       cpu_rule="inclusive child counted once; nested snapshots are diagnostic",
                       individual_tail_operation_cpu="not_measured",
                       worker_returncode=supervisor.get("returncode"))
        if max(0.0, tail) > request["stages"]["settlement_and_verified_export"]["complete_process_cpu_seconds"]:
            raise RuntimeError("SETTLEMENT_TAIL_CPU_BUDGET_EXCEEDED")
        settlement_started = time.monotonic()
        settlement_cpu_started = _usage_cpu(resource.RUSAGE_SELF)
        durable_atomic_json(root / "runtime-output" / "execution-final-status.json", status)
        export_start = time.monotonic()
        export_cpu_start = _usage_cpu(resource.RUSAGE_SELF)
        complete = controlled_export(root, Path(str(root) + "-export"))
        own_cpu = _usage_cpu(resource.RUSAGE_SELF)
        export_cpu = own_cpu - export_cpu_start
        export_wall = time.monotonic() - export_start
        settlement_cpu = own_cpu - settlement_cpu_started
        settlement_wall = time.monotonic() - settlement_started
        settlement_cap = request["stages"]["settlement_and_verified_export"]
        cpu_total = own_cpu + _usage_cpu(resource.RUSAGE_CHILDREN)
        status.update(native_cpu_total_seconds=cpu_total,
                       native_wall_seconds=time.monotonic() - started,
                       export=complete, export_cpu_seconds=export_cpu, export_wall_seconds=export_wall,
                       settlement_stage_complete_process_cpu_seconds=settlement_cpu + max(0.0, tail),
                       settlement_stage_elapsed_wall_seconds=settlement_wall,
                       settlement_status_write_included=True)
        if (settlement_cpu + max(0.0, tail) > settlement_cap["complete_process_cpu_seconds"]
                or settlement_wall > settlement_cap["wall_seconds"]
                or cpu_total > request["totals"]["complete_process_cpu_seconds"]
                or status["native_wall_seconds"] > request["totals"]["wall_seconds"]):
            status["status"] = "technical_stop"
            status["error"] = "FINAL_RESOURCE_LIMIT_EXCEEDED"
        final_settlement_path = Path(str(root) + "-final-settlement.json")
        durable_atomic_json(final_settlement_path, status)

        # Include final-settlement and registration work in the final resource
        # decision; a successful child cannot turn an over-budget attempt into
        # a successful remote result.
        final_own_cpu = _usage_cpu(resource.RUSAGE_SELF)
        final_wall = time.monotonic() - started
        final_cpu_total = final_own_cpu + _usage_cpu(resource.RUSAGE_CHILDREN)
        status.update(native_cpu_total_seconds=final_cpu_total, native_wall_seconds=final_wall,
                      settlement_stage_complete_process_cpu_seconds=(
                          final_own_cpu - settlement_cpu_started + max(0.0, tail)),
                      settlement_stage_elapsed_wall_seconds=time.monotonic() - settlement_started)
        if (final_cpu_total > request["totals"]["complete_process_cpu_seconds"]
                or final_wall > request["totals"]["wall_seconds"]):
            status["status"] = "technical_stop"
            status["error"] = "FINAL_RESOURCE_LIMIT_EXCEEDED"
        durable_atomic_json(final_settlement_path, status)
        post_own_cpu = _usage_cpu(resource.RUSAGE_SELF)
        post_write_cpu = post_own_cpu + _usage_cpu(resource.RUSAGE_CHILDREN)
        post_write_wall = time.monotonic() - started
        post_settlement_cpu = post_own_cpu - settlement_cpu_started
        post_settlement_wall = time.monotonic() - settlement_started
        status.update(native_cpu_total_seconds=post_write_cpu,
                      native_wall_seconds=post_write_wall,
                      settlement_stage_complete_process_cpu_seconds=(
                          post_settlement_cpu + max(0.0, tail)),
                      settlement_stage_elapsed_wall_seconds=post_settlement_wall)
        if (post_settlement_cpu + max(0.0, tail) > settlement_cap["complete_process_cpu_seconds"]
                or post_settlement_wall > settlement_cap["wall_seconds"]
                or post_write_cpu > request["totals"]["complete_process_cpu_seconds"]
                or post_write_wall > request["totals"]["wall_seconds"]):
            status.update(status="technical_stop", error="FINAL_RESOURCE_LIMIT_EXCEEDED",
                          native_cpu_total_seconds=post_write_cpu,
                          native_wall_seconds=post_write_wall,
                          settlement_stage_complete_process_cpu_seconds=post_settlement_cpu + max(0.0, tail),
                          settlement_stage_elapsed_wall_seconds=post_settlement_wall,
                          final_resource_pass=False)
            durable_atomic_json(final_settlement_path, status)
        if status["status"] == "complete" and child.returncode == 0:
            terminal_event = "SUCCEEDED"
        elif supervisor.get("stop_reason") == "External stop request":
            terminal_event = "CANCELLED"
        else:
            terminal_event = "FAILED"
        finish_registration(terminal_event, 0 if terminal_event == "SUCCEEDED" else 1,
                            "completed" if terminal_event == "SUCCEEDED" else terminal_event.lower())
        durable_atomic_json(final_settlement_path, status)
        if registration_error is not None:
            status["registration_error"] = registration_error
            status["status"] = "technical_stop"
            # The terminal registration already failed, so persist that
            # terminal error after adding its diagnostic to the status.
            durable_atomic_json(final_settlement_path, status)
        print(json.dumps(status, ensure_ascii=False))
        return 0 if status["status"] == "complete" else 1
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                try:
                    child.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    status["child_cleanup_status"] = "unresolved_after_sigkill"
        _record_failure_settlement(
            status,
            error=type(exc).__name__ + ": " + str(exc),
            started=started,
            child_cpu_start=child_cpu_start,
        )
        exception_event = terminal_event or (
            "CANCELLED" if (root / "STOP_REQUESTED.json").exists() else "FAILED"
        )
        finish_registration(exception_event, 1,
                            "expired" if exception_event == "EXPIRED" else exception_event.lower())
        if registration_error is not None:
            status["registration_error"] = registration_error
        if (root / "runtime-output").is_dir():
            durable_atomic_json(root / "runtime-output" / "execution-final-status.json", status)
        else:
            durable_atomic_json(root / "registration-failure.json", status)
        try:
            durable_atomic_json(Path(str(root) + "-final-settlement.json"), status)
        except BaseException as settlement_error:
            status["final_settlement_write_error"] = (
                type(settlement_error).__name__ + ": " + str(settlement_error)
            )
        print(json.dumps(status, ensure_ascii=False))
        return 1
    finally:
        if registration_job is not None and not registration_terminal:
            finish_registration("FAILED", 1, "technical_stop")


if __name__ == "__main__":
    raise SystemExit(main())
