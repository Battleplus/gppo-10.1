"""Linux-only staging, native supervision, settlement, and controlled export."""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

PROCESS_STARTED_WALL = time.monotonic()
PROCESS_STARTED_CPU = time.process_time()

import infra_io
from dependency_probe import probe as probe_runtime_dependencies
from manifest_contract import PackageContractError, read_object, sha256_file, verify_external_authorization
from infra_io import (
    append_verified_export_file,
    controlled_export,
    durable_atomic_json,
    filesystem_identity,
    read_json,
    require_native_linux_filesystem,
    sha256_file,
    stage_sealed_package_into_initialized_root,
    tree_manifest,
    verify_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--authorization-file", required=True)
    parser.add_argument("--native-root", required=True)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--argument-probe", required=True)
    parser.add_argument("--windows-preflight-wall-seconds", type=float, required=True)
    parser.add_argument("--windows-preflight-cpu-seconds", type=float, required=True)
    parser.add_argument("--staging-stage-wall-seconds", type=float, required=True)
    parser.add_argument("--label-stage-wall-seconds", type=float, required=True)
    parser.add_argument("--export-stage-wall-seconds", type=float, required=True)
    parser.add_argument("--total-wall-seconds", type=float, required=True)
    parser.add_argument("--cross-system-wall-reserve-seconds", type=float, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dependency-only", action="store_true")
    return parser.parse_args()


def stage_cap(request: dict, stage: str, wall: float, cpu: float) -> None:
    limits = request["stages"][stage]
    if wall > limits["wall_seconds"] or cpu > limits["complete_process_cpu_seconds"]:
        raise RuntimeError(f"{stage} infrastructure resource limit")


NATIVE_CPU_SCOPE_TOLERANCE_SECONDS = 0.05


def build_native_cpu_scope_breakdown(
    native_tree_cpu_seconds: float,
    supervisor_status: dict | None,
    launcher_accounting: dict | None,
    process_tree_sample: dict,
    *,
    tolerance_seconds: float = NATIVE_CPU_SCOPE_TOLERANCE_SECONDS,
) -> dict:
    """Reconcile nested snapshots without adding overlapping CPU scopes.

    The WSL staging process's RUSAGE_CHILDREN delta is authoritative. The
    supervisor and native-launcher files are time-stamped snapshots inside
    that tree; their differences partition the interval after the supervisor
    sample and the native launcher's final accounting write/exit tail.
    """
    tree = float(native_tree_cpu_seconds)
    if not math.isfinite(tree) or tree < 0:
        raise ValueError("Native process-tree CPU is invalid")
    if not math.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("CPU scope tolerance is invalid")

    supervisor_cpu = None
    supervisor_sampled = None
    if isinstance(supervisor_status, dict):
        raw = supervisor_status.get("cpu_seconds")
        if type(raw) in (int, float) and math.isfinite(float(raw)) and float(raw) >= 0:
            supervisor_cpu = float(raw)
        raw_sample = supervisor_status.get("cpu_sampled_monotonic")
        if type(raw_sample) in (int, float) and math.isfinite(float(raw_sample)):
            supervisor_sampled = float(raw_sample)

    launcher_self_cpu = None
    launcher_children_cpu = None
    launcher_complete_cpu = None
    launcher_sampled = None
    launcher_entry = None
    launcher_stage = "unavailable"
    if isinstance(launcher_accounting, dict):
        values = (
            launcher_accounting.get("native_launcher_process_cpu_seconds"),
            launcher_accounting.get("native_launcher_reaped_children_cpu_seconds"),
            launcher_accounting.get("native_launcher_complete_process_cpu_snapshot_seconds"),
        )
        if all(type(value) in (int, float) and math.isfinite(float(value)) and float(value) >= 0 for value in values):
            launcher_self_cpu, launcher_children_cpu, launcher_complete_cpu = map(float, values)
            launcher_sampled = launcher_accounting.get("sampled_monotonic")
            launcher_entry = launcher_accounting.get("process_module_entry_monotonic")
            launcher_stage = str(launcher_accounting.get("stage", "unknown"))
            if abs(launcher_complete_cpu - launcher_self_cpu - launcher_children_cpu) > 1e-6:
                raise ValueError("Native launcher CPU component sum mismatch")

    errors = []
    completion_delta = None
    exit_tail = None
    if launcher_complete_cpu is not None:
        if launcher_complete_cpu - tree > tolerance_seconds:
            errors.append("launcher_snapshot_exceeds_outer_tree")
        exit_tail = tree - launcher_complete_cpu
        if abs(exit_tail) > tolerance_seconds:
            errors.append("native_launcher_accounting_write_or_exit_tail_exceeds_tolerance")
    if supervisor_cpu is not None and launcher_complete_cpu is not None:
        completion_delta = launcher_complete_cpu - supervisor_cpu
        if completion_delta < -tolerance_seconds:
            errors.append("launcher_snapshot_precedes_supervisor_snapshot")
    elif supervisor_cpu is not None and supervisor_cpu - tree > tolerance_seconds:
        errors.append("supervisor_snapshot_exceeds_outer_tree")

    unexplained_after_supervisor = None
    if supervisor_cpu is not None:
        unexplained_after_supervisor = tree - supervisor_cpu
        if completion_delta is not None and exit_tail is not None:
            if abs(unexplained_after_supervisor - completion_delta - exit_tail) > 1e-6:
                errors.append("native_cpu_scope_decomposition_mismatch")

    start = process_tree_sample.get("sample_started_monotonic")
    end = process_tree_sample.get("sample_ended_monotonic")
    for label, value in (("process-tree start", start), ("process-tree end", end)):
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            errors.append(f"{label} timestamp unavailable")
    if type(start) in (int, float) and type(end) in (int, float) and end < start:
        errors.append("native_process_tree_sample_clock_order_invalid")

    def valid_timestamp(value: object) -> bool:
        return type(value) in (int, float) and math.isfinite(float(value))

    supervisor_started = None
    if isinstance(supervisor_status, dict):
        supervisor_started = supervisor_status.get("cpu_sampling_started_monotonic")
    if supervisor_cpu is not None:
        if not valid_timestamp(supervisor_started):
            errors.append("supervisor_sampling_start_timestamp_missing_or_invalid")
        if not valid_timestamp(supervisor_sampled):
            errors.append("supervisor_sample_timestamp_missing_or_invalid")
        if valid_timestamp(supervisor_started) and valid_timestamp(supervisor_sampled):
            if float(supervisor_started) > float(supervisor_sampled):
                errors.append("supervisor_snapshot_clock_order_invalid")
            if valid_timestamp(start) and float(supervisor_started) < float(start):
                errors.append("supervisor_sampling_starts_before_outer_tree")
            if valid_timestamp(end) and float(supervisor_sampled) > float(end):
                errors.append("supervisor_snapshot_after_outer_tree")

    if launcher_complete_cpu is not None:
        if not valid_timestamp(launcher_entry):
            errors.append("launcher_entry_timestamp_missing_or_invalid")
        if not valid_timestamp(launcher_sampled):
            errors.append("launcher_sample_timestamp_missing_or_invalid")
        if valid_timestamp(launcher_entry) and valid_timestamp(launcher_sampled):
            if float(launcher_entry) > float(launcher_sampled):
                errors.append("launcher_snapshot_clock_order_invalid")
            if valid_timestamp(start) and float(launcher_entry) < float(start):
                errors.append("launcher_entry_before_outer_tree")
            if valid_timestamp(end) and float(launcher_entry) > float(end):
                errors.append("launcher_entry_after_outer_tree")
            if valid_timestamp(end) and float(launcher_sampled) > float(end):
                errors.append("launcher_snapshot_after_outer_tree")

    if supervisor_cpu is not None and launcher_complete_cpu is not None:
        required_times = (start, launcher_entry, supervisor_started, supervisor_sampled, launcher_sampled, end)
        if all(valid_timestamp(value) for value in required_times):
            if not (
                float(start) <= float(launcher_entry)
                <= float(supervisor_started)
                <= float(supervisor_sampled)
                <= float(launcher_sampled)
                <= float(end)
            ):
                errors.append("nested_cpu_snapshot_timestamps_out_of_order_or_outside_outer_tree")

    if errors:
        breakdown_status = "failed"
    elif supervisor_cpu is None and launcher_complete_cpu is not None:
        breakdown_status = "launcher_only_early_exit"
    elif launcher_complete_cpu is None:
        breakdown_status = "tree_only_timeout_or_early_exit"
    else:
        breakdown_status = "reconciled"
    return {
        "schema": "w1-native-cpu-scope-breakdown/1.0.0",
        "status": breakdown_status,
        "authoritative_total": "native_process_tree_cpu_seconds",
        "native_process_tree_cpu_seconds": tree,
        "native_process_tree_sample": process_tree_sample,
        "supervisor_complete_process_cpu_seconds_nested_snapshot": supervisor_cpu,
        "supervisor_cpu_sampled_monotonic": supervisor_sampled,
        "native_launcher_process_cpu_seconds_nested_component": launcher_self_cpu,
        "native_launcher_reaped_children_cpu_seconds_nested_component": launcher_children_cpu,
        "native_launcher_complete_process_cpu_snapshot_seconds": launcher_complete_cpu,
        "native_launcher_process_module_entry_monotonic": launcher_entry if valid_timestamp(launcher_entry) else None,
        "native_launcher_cpu_sampled_monotonic": launcher_sampled if valid_timestamp(launcher_sampled) else None,
        "native_launcher_stage": launcher_stage,
        "supervisor_sample_to_native_launcher_snapshot_cpu_seconds": completion_delta,
        "native_launcher_snapshot_to_parent_reap_cpu_seconds": exit_tail,
        "outer_tree_minus_supervisor_snapshot_cpu_seconds": unexplained_after_supervisor,
        "tolerance_seconds_for_only_unmeasured_launcher_write_exit_tail": tolerance_seconds,
        "scope_definitions": {
            "outer_tree": "WSL staging process RUSAGE_CHILDREN delta, sampled before native entry spawn through wait or forced process-group termination; includes native entry and waited descendants",
            "supervisor": "native entry RUSAGE_SELF plus reaped RUSAGE_CHILDREN, sampled after worker reap/cleanup and before final artifact scan, cumulative settlement and final status serialization",
            "native_launcher": "native entry RUSAGE_SELF and RUSAGE_CHILDREN after supervise.main returns, before writing this accounting file and exiting; direct self CPU is nested inside both inclusive totals",
            "budget_rule": "charge outer_tree once; nested supervisor and native-launcher values explain scope and sampling differences and are never added",
        },
        "errors": errors,
    }


def _wall_alarm(_signum, _frame) -> None:
    raise TimeoutError("stage wall deadline reached")


def arm_wall_deadline(seconds: float) -> None:
    if not math.isfinite(seconds) or seconds <= 0:
        raise TimeoutError("no stage wall budget remains")
    signal.signal(signal.SIGALRM, _wall_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)


def disarm_wall_deadline() -> None:
    signal.setitimer(signal.ITIMER_REAL, 0)


def validate_linux_contract(args: argparse.Namespace, token: str):
    if os.name != "posix" or not Path("/proc/self/mountinfo").is_file():
        raise RuntimeError("Linux staging entry requires a real Linux process")
    source = Path(args.source)
    authorization_path = Path(args.authorization_file)
    native_root = Path(args.native_root)
    export_root = Path(args.export_root)
    if not all(path.is_absolute() for path in (source, native_root, export_root)):
        raise RuntimeError("All WSL paths must be absolute")
    if not str(source).startswith("/mnt/") or not str(export_root).startswith("/mnt/"):
        raise RuntimeError("Source and export must be explicit WSL mounted paths")
    if not authorization_path.is_absolute() or not str(authorization_path).startswith("/mnt/"):
        raise RuntimeError("External authorization must use an explicit WSL mounted path")
    if str(native_root).startswith("/mnt/"):
        raise RuntimeError("Native active root cannot use a mounted Windows filesystem")
    try:
        verified = verify_external_authorization(
            source, authorization_path, token=token, preflight_only=args.preflight_only,
        )
    except PackageContractError as exc:
        raise RuntimeError(f"AUTHORIZATION_OR_PACKAGE_CONTRACT_ERROR: {exc}") from exc
    identity = verified["identity"]
    manifest = identity["manifest"]
    request = verified["request"]
    contract = verified["contract"]
    if contract.get("accounting_reserves") != request.get("accounting_reserves"):
        raise RuntimeError("Cross-system resource reserves mismatch")
    if not (args.attempt == manifest.get("attempt") == request.get("attempt") == contract.get("attempt")):
        raise RuntimeError("Attempt identity mismatch in Linux")
    if str(source.resolve()) != contract["wsl_source_root"]:
        raise RuntimeError("Linux source root identity mismatch")
    if str(native_root) != contract["native_execution_root"] or str(export_root) != contract["windows_export_root"]:
        raise RuntimeError("Linux launch path identity mismatch")
    if args.argument_probe != contract["argument_probe"]:
        raise RuntimeError("Structured argument probe changed")
    expected_limits = (
        (args.staging_stage_wall_seconds, request["stages"]["staging_and_zero_step_gate"]["wall_seconds"]),
        (args.label_stage_wall_seconds, request["stages"]["label_qualification"]["wall_seconds"]),
        (args.export_stage_wall_seconds, request["stages"]["settlement_and_verified_export"]["wall_seconds"]),
        (args.total_wall_seconds, request["totals"]["wall_seconds"]),
        (args.cross_system_wall_reserve_seconds, request["accounting_reserves"]["cross_system_wall_seconds"]),
    )
    if any(float(actual) != float(expected) for actual, expected in expected_limits):
        raise RuntimeError("Linux launcher resource limits differ from the frozen request")
    if Path(sys.executable).resolve() != Path(contract["native_python"]).resolve():
        raise RuntimeError("Unexpected WSL Python interpreter")
    dependency = probe_runtime_dependencies(
        expected_python=contract["native_python"], expected_root=source,
    )
    if filesystem_identity(authorization_path.parent)["filesystem_type"].lower() not in {"9p", "drvfs"}:
        raise RuntimeError("External authorization must remain on the Windows mounted filesystem")
    if filesystem_identity(source)["filesystem_type"].lower() not in {"9p", "drvfs"}:
        raise RuntimeError("Windows source is not exposed through the expected WSL mount")
    if filesystem_identity(export_root.parent)["filesystem_type"].lower() not in {"9p", "drvfs"}:
        raise RuntimeError("Windows export parent is not exposed through the expected WSL mount")
    require_native_linux_filesystem(native_root.parent)
    if native_root.exists():
        raise RuntimeError("Native attempt path already exists; no retry")
    if export_root.exists():
        raise RuntimeError("Export path already exists; no overwrite or retry")
    for value in (args.windows_preflight_wall_seconds, args.windows_preflight_cpu_seconds):
        if not math.isfinite(value) or value < 0:
            raise RuntimeError("Invalid Windows preflight timing")
    return source, native_root, export_root, manifest, contract, request, dependency


def run_native(
    python: str,
    root: Path,
    token: str,
    authorization_file: str,
    entry: str,
    *,
    timeout_seconds: float,
    inject_interrupt: bool = False,
    extra_args: tuple[str, ...] = (),
) -> tuple[int, str | None, str | None, bool, float, dict]:
    environment = dict(os.environ)
    environment["W1_EXTERNAL_ATTEMPT_TOKEN"] = token
    environment["W1_EXTERNAL_AUTHORIZATION_FILE"] = authorization_file
    if entry == "integration_native_launch.py":
        environment["W1_TEST_ONLY_INTEGRATION"] = "1"
    interruption = None
    child = None
    previous = {}
    sample_started = time.monotonic()
    child_cpu_start = resource.getrusage(resource.RUSAGE_CHILDREN)

    def child_cpu_seconds() -> float:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        return max(
            0.0,
            (usage.ru_utime + usage.ru_stime)
            - (child_cpu_start.ru_utime + child_cpu_start.ru_stime),
        )
    def finished(cpu: float) -> tuple[int, str | None, str | None, bool, float, dict]:
        returncode = int(child.returncode) if child is not None and child.returncode is not None else 1
        return (
            returncode,
            interruption,
            None,
            child is not None,
            cpu,
            {
                "scope": "WSL staging process RUSAGE_CHILDREN delta; child and waited descendants",
                "root_pid": child.pid if child is not None else None,
                "sample_started_monotonic": sample_started,
                "sample_ended_monotonic": time.monotonic(),
                "wait_completed": bool(child is not None and child.returncode is not None),
                "termination": "forced_process_group_stop" if interruption and "wall limit" in interruption else "normal_wait_or_launch_failure",
            },
        )
    def stop_child() -> None:
        if child is None or child.poll() is not None:
            return
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=2.0)

    try:
        child = subprocess.Popen(
            [python, "-B", str(root / entry), *extra_args],
            cwd=root,
            env=environment,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            shell=False,
        )

        def forward(sig, _frame):
            nonlocal interruption
            interruption = f"Linux launcher received signal {sig}"
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                pass

        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, forward)
        if inject_interrupt:
            time.sleep(0.25)
            interruption = "integration-test injected Linux launcher interruption"
            os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=timeout_seconds)
            return finished(child_cpu_seconds())
        except subprocess.TimeoutExpired:
            interruption = "native child exceeded frozen label-stage wall limit"
            stop_child()
            code, _, _, started, cpu, sample = finished(child_cpu_seconds())
            return 124, interruption, "TimeoutError: native child wall limit", started, cpu, sample
    except BaseException as exc:
        stop_child()
        _, _, _, started, cpu, sample = finished(child_cpu_seconds())
        return (
            1,
            interruption,
            type(exc).__name__ + ": " + str(exc),
            started,
            cpu,
            sample,
        )
    finally:
        stop_child()
        environment.pop("W1_EXTERNAL_ATTEMPT_TOKEN", None)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def safe_status(path: Path, value: dict, fallback: Path) -> None:
    try:
        durable_atomic_json(path, value)
    except BaseException as exc:
        fallback_value = dict(value)
        fallback_value["primary_status_write_error"] = type(exc).__name__ + ": " + str(exc)
        durable_atomic_json(fallback, fallback_value)


def maybe_install_integration_fault(contract: dict, native_root: Path) -> None:
    if not contract.get("integration_test") or contract.get("integration_fault") != "staging_fsync":
        return
    original = infra_io._fsync_directory

    def faulting(directory: Path) -> None:
        if Path(directory).resolve() == native_root.resolve():
            raise OSError("integration-test injected native-root fsync failure")
        original(directory)

    infra_io._fsync_directory = faulting


def export_attempt(contract: dict, native_root: Path, export_root: Path):
    if contract.get("integration_test") and contract.get("integration_fault") == "export":
        raise OSError("integration-test injected export failure")
    return controlled_export(native_root, export_root)


def _legacy_main() -> int:
    linux_wall_start = time.monotonic()
    linux_cpu_start = time.process_time()
    args = parse_args()
    token = sys.stdin.buffer.read().decode("utf-8", errors="strict")
    native_root = Path(args.native_root)
    fallback = native_root.parent / f".{args.attempt}.launcher-failure.json"
    stage = "linux_preflight"
    try:
        source, native_root, export_root, manifest, contract, request, dependency = validate_linux_contract(args, token)
        if args.preflight_only:
            print(json.dumps({"schema": "w1-linux-preflight/2.0.0", "status": "preflight_pass", "attempt": args.attempt, "staging_started": False, "worker_started": False}, sort_keys=True))
            return 0
        native_root.mkdir(parents=False, exist_ok=False)
        status_path = native_root / "launcher-status.json"
        status = {
            "schema": "w1-linux-launcher-status/1.0.0",
            "attempt": args.attempt,
            "status": "running",
            "stage": "staging",
            "windows_preflight_wall_seconds": args.windows_preflight_wall_seconds,
            "windows_preflight_cpu_seconds": args.windows_preflight_cpu_seconds,
            "linux_wall_seconds": time.monotonic() - linux_wall_start,
            "linux_cpu_seconds": time.process_time() - linux_cpu_start,
            "argument_probe": args.argument_probe,
            "automatic_retry": False,
            "worker_started": False,
        }
        durable_atomic_json(status_path, status)
        maybe_install_integration_fault(contract, native_root)
        stage = "staging"
        staging_started = time.monotonic()
        staging_cpu_started = time.process_time()
        staged = stage_sealed_package_into_initialized_root(source, native_root, manifest)
        staging_wall = time.monotonic() - staging_started
        staging_cpu = time.process_time() - staging_cpu_started
        combined_staging_wall = args.windows_preflight_wall_seconds + staging_wall
        combined_staging_cpu = args.windows_preflight_cpu_seconds + staging_cpu
        stage_cap(
            request,
            "staging_and_zero_step_gate",
            combined_staging_wall,
            combined_staging_cpu,
        )
        durable_atomic_json(native_root / "staging-record.json", {
            "schema": "native-staging-record/2.0.0",
            "attempt": args.attempt,
            "staging_wall_seconds": staging_wall,
            "staging_cpu_seconds": staging_cpu,
            "windows_preflight_wall_seconds": args.windows_preflight_wall_seconds,
            "windows_preflight_cpu_seconds": args.windows_preflight_cpu_seconds,
            "combined_staging_wall_seconds": combined_staging_wall,
            "combined_staging_cpu_seconds": combined_staging_cpu,
            "cross_process_start_gap": "unavailable",
            **staged,
        })
        status.update(stage="native_supervision", native_launcher_start_attempted=True)
        durable_atomic_json(status_path, status)
        stage = "native_supervision"
        entry = "native_launch.py" if args.dependency_only else contract.get("native_entry", "native_launch.py")
        python = contract.get("native_child_python", contract["native_python"])
        native_returncode, interruption, launch_error, native_launcher_started, native_child_cpu, native_cpu_sample = run_native(
            python,
            native_root,
            token,
            args.authorization_file,
            entry,
            inject_interrupt=bool(
                contract.get("integration_test")
                and contract.get("integration_fault") == "native_interrupt"
            ),
            extra_args=("--dependency-only",) if args.dependency_only else (),
        )
        token = ""
        if args.dependency_only and not (native_root / "supervisor-status.json").exists():
            durable_atomic_json(native_root / "supervisor-status.json", {
                "schema": "dependency-only-native-supervisor-evidence/1.0.0",
                "status": "not_started_dependency_gate_passed",
                "worker_started": False,
                "environment_constructed": False,
                "model_initialized": False,
                "checkpoint_loaded": False,
                "training_started": False,
                "cpu_seconds": native_child_cpu,
            })
            durable_atomic_json(native_root / "zero-step-settlement.json", {
                "schema": "w1-runtime-dependency-zero-step-settlement/1.0.0",
                "attempt": args.attempt,
                "status": "pass" if native_returncode == 0 and launch_error is None else "technical_stop",
                "dynamic_calls": {
                    "environment_steps": 0,
                    "resets": 0,
                    "candidate_branches": 0,
                    "rule_decisions": 0,
                    "model_initializations_or_loads": 0,
                    "optimizer_updates": 0,
                    "batch_forwards": 0,
                    "model_sample_evaluations": 0,
                    "checkpoint_writes": 0,
                },
                "worker_started": False,
                "environment_constructed": False,
                "model_initialized": False,
                "checkpoint_loaded": False,
                "training_started": False,
                "native_dependency_probe_cpu_seconds": native_child_cpu,
                "dependency_probe_synthetic_kernel_matmuls": dependency.get("synthetic_kernel_matmuls", 0),
                "dependency_probe_synthetic_kernel_backwards": dependency.get("synthetic_kernel_backwards", 0),
                "dependency_probe_synthetic_optimizer_updates": dependency.get("synthetic_optimizer_updates", 0),
                "dependency_probe_model_forwards": dependency.get("model_forwards", 0),
            })
        durable_atomic_json(native_root / "infrastructure-settlement.json", {
            "schema": "native-infrastructure-settlement/2.0.0",
            "attempt": args.attempt,
            "native_returncode": native_returncode,
            "launcher_interruption": interruption,
            "launch_error": launch_error,
            "native_launcher_started": native_launcher_started,
            "native_child_cpu_seconds": native_child_cpu,
            "worker_relaunched": False,
            "automatic_retry": False,
            "historical_credit": 0,
            "staging_wall_seconds": staging_wall,
            "staging_cpu_seconds": staging_cpu,
            "export_status_at_copy": "pending",
        })
        stage = "settlement_and_verified_export"
        export_started = time.monotonic()
        export_cpu_started = time.process_time()
        export_result = None
        export_error = None
        try:
            export_result = export_attempt(contract, native_root, export_root)
        except BaseException as exc:
            export_error = type(exc).__name__ + ": " + str(exc)
        export_wall = time.monotonic() - export_started
        export_cpu = time.process_time() - export_cpu_started
        supervisor_evidence = None
        combined_budget_cpu = None
        try:
            stage_cap(request, "settlement_and_verified_export", export_wall, export_cpu)
            if export_result is None:
                raise RuntimeError("Verified export result unavailable")
            supervisor_evidence = read_json(native_root / "supervisor-status.json")
            linux_wall = time.monotonic() - linux_wall_start
            linux_cpu = time.process_time() - linux_cpu_start
            combined_budget_wall = args.windows_preflight_wall_seconds + linux_wall
            supervisor_cpu = float(supervisor_evidence["cpu_seconds"])
            if not math.isfinite(supervisor_cpu) or supervisor_cpu < 0:
                raise RuntimeError("Supervisor complete-process CPU evidence is invalid")
            combined_budget_cpu = (
                args.windows_preflight_cpu_seconds + linux_cpu + native_child_cpu + supervisor_cpu
            )
            native_bytes = sum(row["bytes"] for row in tree_manifest(native_root).values())
            aggregate_bytes = native_bytes + int(export_result["payload_bytes"])
            if combined_budget_wall > request["totals"]["wall_seconds"] or combined_budget_cpu > request["totals"]["complete_process_cpu_seconds"]:
                raise RuntimeError("Combined Linux infrastructure limit exceeded")
            if aggregate_bytes > request["totals"]["aggregate_native_plus_verified_export_bytes"]:
                raise RuntimeError("Aggregate native plus export storage reserve reached")
        except BaseException as exc:
            export_error = export_error or type(exc).__name__ + ": " + str(exc)
        export_status = {
            "schema": "native-export-status/2.0.0",
            "attempt": args.attempt,
            "status": "verified" if export_error is None else "failed",
            "result": export_result,
            "error": export_error,
            "export_wall_seconds": export_wall,
            "export_cpu_seconds": export_cpu,
            "linux_wall_seconds": time.monotonic() - linux_wall_start,
            "linux_cpu_seconds": time.process_time() - linux_cpu_start,
            "supervisor_complete_process_cpu_seconds": (
                float(supervisor_evidence["cpu_seconds"])
                if isinstance(supervisor_evidence, dict)
                and isinstance(supervisor_evidence.get("cpu_seconds"), (int, float))
                else "unavailable"
            ),
            "combined_budget_wall_seconds": (
                args.windows_preflight_wall_seconds + (time.monotonic() - linux_wall_start)
            ),
            "combined_budget_cpu_seconds": (
                combined_budget_cpu if export_error is None else "unavailable after failed total check"
            ),
            "native_launcher_process_cpu_seconds": native_child_cpu,
            "combined_budget_scope": "Windows preflight process CPU plus Linux staging/export process CPU plus native launcher process CPU plus supervisor complete-process CPU; Windows launcher verifies its post-preflight CPU and total wall after WSL returns",
            "cross_process_start_gap": "measured by Windows launcher after WSL returns",
            "windows_post_preflight_cpu_seconds": "measured by Windows launcher after WSL returns",
            "worker_relaunched": False,
            "automatic_retry": False,
        }
        durable_atomic_json(native_root / "export-status.json", export_status)
        if export_error is None:
            try:
                export_result = append_verified_export_file(
                    native_root / "export-status.json",
                    export_root,
                    "export-status.json",
                )
            except BaseException as exc:
                export_error = type(exc).__name__ + ": " + str(exc)
                export_status["status"] = "failed"
                export_status["error"] = export_error
                durable_atomic_json(native_root / "export-status.json", export_status)
        supervisor_evidence = (
            read_json(native_root / "supervisor-status.json")
            if (native_root / "supervisor-status.json").is_file()
            else None
        )
        status.update(
            status="complete" if native_returncode == 0 and launch_error is None and export_error is None else "stopped",
            stage="complete" if export_error is None else "export_failed",
            linux_wall_seconds=time.monotonic() - linux_wall_start,
            linux_cpu_seconds=time.process_time() - linux_cpu_start,
            worker_returncode=native_returncode,
            native_launcher_started=native_launcher_started,
            worker_started=bool(
                isinstance(supervisor_evidence, dict)
                and isinstance(supervisor_evidence.get("pid"), int)
            ),
            launch_error=launch_error,
            export_error=export_error,
        )
        durable_atomic_json(status_path, status)
        return 0 if status["status"] == "complete" else 1
    except BaseException as exc:
        failure = {
            "schema": "w1-linux-launcher-status/1.0.0",
            "attempt": args.attempt,
            "status": "technical_stop",
            "stage": stage,
            "error": type(exc).__name__ + ": " + str(exc),
            "linux_wall_seconds": time.monotonic() - linux_wall_start,
            "linux_cpu_seconds": time.process_time() - linux_cpu_start,
            "windows_preflight_wall_seconds": args.windows_preflight_wall_seconds,
            "windows_preflight_cpu_seconds": args.windows_preflight_cpu_seconds,
            "worker_started": False,
            "automatic_retry": False,
        }
        if native_root.exists():
            safe_status(native_root / "launcher-status.json", failure, fallback)
        else:
            try:
                durable_atomic_json(fallback, failure)
            except BaseException:
                pass
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1
    finally:
        token = ""


def main() -> int:
    linux_wall_start = PROCESS_STARTED_WALL
    linux_cpu_start = PROCESS_STARTED_CPU
    args = parse_args()
    token = ""
    native_root = Path(args.native_root)
    fallback = native_root.parent / f".{args.attempt}.launcher-failure.json"
    stage = "linux_preflight"
    native_child_cpu = 0.0
    export_timer_armed = False
    try:
        if not args.preflight_only:
            arm_wall_deadline(
                args.staging_stage_wall_seconds - args.windows_preflight_wall_seconds
                - (time.monotonic() - linux_wall_start)
            )
        token = sys.stdin.buffer.read().decode("utf-8", errors="strict")
        source, native_root, export_root, manifest, contract, request, dependency = validate_linux_contract(args, token)
        if args.preflight_only:
            print(json.dumps({
                "schema": "w1-linux-preflight/2.0.0",
                "status": "preflight_pass",
                "attempt": args.attempt,
                "staging_started": False,
                "worker_started": False,
                "dependency_probe": {
                    "python": dependency.get("python_executable"),
                    "torch": dependency.get("torch_version"),
                    "cpu_only": dependency.get("cpu_only"),
                    "synthetic_kernel_matmuls": dependency.get("synthetic_kernel_matmuls"),
                    "synthetic_kernel_backwards": dependency.get("synthetic_kernel_backwards"),
                    "synthetic_optimizer_updates": dependency.get("synthetic_optimizer_updates"),
                    "model_forwards": dependency.get("model_forwards"),
                    "remote_runtime_binding": dependency.get("remote_runtime_binding"),
                },
            }, sort_keys=True))
            return 0

        native_root.mkdir(parents=False, exist_ok=False)
        status_path = native_root / "launcher-status.json"
        status = {
            "schema": "w1-linux-launcher-status/1.0.0",
            "attempt": args.attempt,
            "status": "running",
            "stage": "staging_and_zero_step_gate",
            "windows_preflight_wall_seconds": args.windows_preflight_wall_seconds,
            "windows_preflight_process_tree_cpu_seconds": args.windows_preflight_cpu_seconds,
            "linux_wall_seconds": time.monotonic() - linux_wall_start,
            "linux_process_cpu_seconds": time.process_time() - linux_cpu_start,
            "argument_probe": args.argument_probe,
            "automatic_retry": False,
            "worker_started": False,
        }
        durable_atomic_json(status_path, status)
        maybe_install_integration_fault(contract, native_root)
        staged = stage_sealed_package_into_initialized_root(source, native_root, manifest)
        staging_wall = time.monotonic() - linux_wall_start
        staging_cpu = time.process_time() - linux_cpu_start
        combined_staging_wall = args.windows_preflight_wall_seconds + staging_wall
        combined_staging_cpu = args.windows_preflight_cpu_seconds + staging_cpu
        stage_cap(request, "staging_and_zero_step_gate", combined_staging_wall, combined_staging_cpu)
        durable_atomic_json(native_root / "staging-record.json", {
            "schema": "native-staging-record/3.0.0",
            "attempt": args.attempt,
            "staging_wall_seconds": staging_wall,
            "staging_linux_process_cpu_seconds": staging_cpu,
            "windows_preflight_wall_seconds": args.windows_preflight_wall_seconds,
            "windows_preflight_process_tree_cpu_seconds": args.windows_preflight_cpu_seconds,
            "combined_staging_wall_seconds": combined_staging_wall,
            "combined_staging_process_cpu_seconds": combined_staging_cpu,
            "cross_process_start_gap": "included in stage wall upper bound",
            **staged,
        })
        disarm_wall_deadline()

        status.update(stage="label_qualification", native_launcher_start_attempted=True)
        durable_atomic_json(status_path, status)
        label_started = time.monotonic()
        label_parent_cpu_started = time.process_time()
        native_returncode, interruption, launch_error, native_started, native_child_cpu, native_cpu_sample = run_native(
            contract.get("native_child_python", contract["native_python"]),
            native_root,
            token,
            args.authorization_file,
            contract.get("native_entry", "native_launch.py"),
            timeout_seconds=float(request["stages"]["label_qualification"]["wall_seconds"]),
            inject_interrupt=bool(
                contract.get("integration_test")
                and contract.get("integration_fault") == "native_interrupt"
            ),
        )
        token = ""
        label_wall = time.monotonic() - label_started
        label_launcher_cpu = time.process_time() - label_parent_cpu_started
        supervisor_path = native_root / "supervisor-status.json"
        supervisor_evidence = read_json(supervisor_path) if supervisor_path.is_file() else None
        supervisor_cpu = (
            float(supervisor_evidence["cpu_seconds"])
            if isinstance(supervisor_evidence, dict)
            and isinstance(supervisor_evidence.get("cpu_seconds"), (int, float))
            and math.isfinite(float(supervisor_evidence["cpu_seconds"]))
            else None
        )
        launcher_accounting_path = native_root / "native-launcher-accounting.json"
        launcher_accounting = read_json(launcher_accounting_path) if launcher_accounting_path.is_file() else None
        cpu_scope = build_native_cpu_scope_breakdown(
            native_child_cpu, supervisor_evidence, launcher_accounting, native_cpu_sample,
        )
        label_error = None
        if cpu_scope["errors"]:
            label_error = "Native CPU scope reconciliation failed: " + "; ".join(cpu_scope["errors"])
        if label_wall > request["stages"]["label_qualification"]["wall_seconds"]:
            label_error = label_error or "label stage wall cap exceeded"
        label_complete_cpu = native_child_cpu + label_launcher_cpu
        if label_complete_cpu > request["stages"]["label_qualification"]["complete_process_cpu_seconds"]:
            label_error = label_error or "label stage complete-process CPU cap exceeded"
        launch_error = launch_error or label_error

        stage = "settlement_and_verified_export"
        export_started = time.monotonic()
        export_cpu_started = time.process_time()
        arm_wall_deadline(float(request["stages"][stage]["wall_seconds"]))
        export_timer_armed = True
        durable_atomic_json(native_root / "infrastructure-settlement.json", {
            "schema": "native-infrastructure-settlement/3.0.0",
            "attempt": args.attempt,
            "native_returncode": native_returncode,
            "launcher_interruption": interruption,
            "launch_error": launch_error,
            "native_launcher_started": native_started,
            "native_process_tree_cpu_seconds": native_child_cpu,
            "label_stage_wsl_launcher_process_cpu_seconds": label_launcher_cpu,
            "label_stage_complete_process_cpu_seconds": label_complete_cpu,
            "cpu_scope_breakdown": cpu_scope,
            "worker_relaunched": False,
            "automatic_retry": False,
            "historical_credit": 0,
            "combined_staging_wall_seconds": combined_staging_wall,
            "combined_staging_process_cpu_seconds": combined_staging_cpu,
        })

        export_result = None
        export_error = None
        if native_returncode != 0 or launch_error:
            export_error = launch_error or f"Native process exited {native_returncode}"
        try:
            export_result = export_attempt(contract, native_root, export_root)
        except BaseException as exc:
            export_error = export_error or (type(exc).__name__ + ": " + str(exc))

        export_wall_snapshot = time.monotonic() - export_started
        export_cpu_snapshot = time.process_time() - export_cpu_started
        export_wall_cap = float(request["stages"][stage]["wall_seconds"])
        export_cpu_cap = float(request["stages"][stage]["complete_process_cpu_seconds"])
        export_wall_remaining = max(0.0, export_wall_cap - export_wall_snapshot)
        export_cpu_remaining = max(0.0, export_cpu_cap - export_cpu_snapshot)
        if export_wall_snapshot > export_wall_cap or export_cpu_snapshot > export_cpu_cap:
            export_error = export_error or "settlement/export stage cap exceeded before finalization"

        linux_wall_snapshot = time.monotonic() - linux_wall_start
        linux_cpu_snapshot = time.process_time() - linux_cpu_start
        combined_wall_upper = (
            args.windows_preflight_wall_seconds + linux_wall_snapshot
            + export_wall_remaining + args.cross_system_wall_reserve_seconds
        )
        complete_cpu_upper = (
            args.windows_preflight_cpu_seconds + linux_cpu_snapshot + native_child_cpu
            + export_cpu_remaining
            + float(request["accounting_reserves"]["windows_post_preflight_process_cpu_seconds"])
        )
        supervisor_status = supervisor_evidence.get("status") if isinstance(supervisor_evidence, dict) else "unavailable"
        if export_result is None:
            export_error = export_error or "Verified export result unavailable"
        if combined_wall_upper > request["totals"]["wall_seconds"]:
            export_error = export_error or "Combined wall upper bound exceeds frozen total"
        if complete_cpu_upper > request["totals"]["complete_process_cpu_seconds"]:
            export_error = export_error or "Complete-process CPU upper bound exceeds frozen total"
        native_bytes = sum(row["bytes"] for row in tree_manifest(native_root).values())
        aggregate_bytes = native_bytes + int(export_result["payload_bytes"]) if export_result else native_bytes
        if aggregate_bytes > request["totals"]["aggregate_native_plus_verified_export_bytes"]:
            export_error = export_error or "Aggregate native plus export storage reserve reached"

        export_status = {
            "schema": "native-export-status/4.0.0",
            "attempt": args.attempt,
            "status": "verified" if export_error is None else "failed",
            "result": export_result,
            "error": export_error,
            "staging_wall_seconds": staging_wall,
            "combined_staging_wall_seconds": combined_staging_wall,
            "combined_staging_process_cpu_seconds": combined_staging_cpu,
            "label_stage_wall_seconds": label_wall,
            "label_stage_wsl_launcher_process_cpu_seconds": label_launcher_cpu,
            "label_stage_native_process_tree_cpu_seconds": native_child_cpu,
            "label_stage_complete_process_cpu_seconds": label_complete_cpu,
            "native_process_tree_cpu_seconds": native_child_cpu,
            "supervisor_complete_process_cpu_seconds": supervisor_cpu if supervisor_cpu is not None else "unavailable",
            "native_launcher_cpu_scope_breakdown": cpu_scope,
            "native_launcher_process_cpu_seconds": cpu_scope["native_launcher_process_cpu_seconds_nested_component"],
            "native_launcher_complete_process_cpu_snapshot_seconds": cpu_scope["native_launcher_complete_process_cpu_snapshot_seconds"],
            "supervisor_status": supervisor_status,
            "export_stage_wall_snapshot_seconds": export_wall_snapshot,
            "export_stage_wall_cap_seconds": export_wall_cap,
            "export_stage_cpu_snapshot_seconds": export_cpu_snapshot,
            "export_stage_cpu_cap_seconds": export_cpu_cap,
            "linux_launcher_process_cpu_seconds": linux_cpu_snapshot,
            "windows_preflight_process_tree_cpu_seconds": args.windows_preflight_cpu_seconds,
            "windows_preflight_wall_seconds": args.windows_preflight_wall_seconds,
            "linux_wall_snapshot_seconds": linux_wall_snapshot,
            "combined_budget_wall_seconds": args.windows_preflight_wall_seconds + linux_wall_snapshot,
            "combined_budget_wall_upper_bound_seconds": combined_wall_upper,
            "combined_budget_cpu_upper_bound_seconds": complete_cpu_upper,
            "automatic_retry": False,
            "worker_relaunched": False,
        }
        durable_atomic_json(native_root / "export-status.json", export_status)
        if export_error is None:
            try:
                export_result = append_verified_export_file(
                    native_root / "export-status.json", export_root, "export-status.json",
                )
                export_status["result"] = export_result
            except BaseException as exc:
                export_error = type(exc).__name__ + ": " + str(exc)
                export_status.update(status="failed", error=export_error)
                durable_atomic_json(native_root / "export-status.json", export_status)
        export_wall_final = time.monotonic() - export_started
        export_cpu_final = time.process_time() - export_cpu_started
        if export_wall_final > export_wall_cap or export_cpu_final > export_cpu_cap:
            export_error = export_error or "settlement/export final stage cap exceeded"
            export_status.update(status="failed", error=export_error)
            durable_atomic_json(native_root / "export-status.json", export_status)
        disarm_wall_deadline()
        export_timer_armed = False
        status.update(
            status="complete" if native_returncode == 0 and not launch_error and not export_error else "stopped",
            stage="complete" if not export_error else "export_failed",
            linux_wall_seconds=time.monotonic() - linux_wall_start,
            linux_process_cpu_seconds=time.process_time() - linux_cpu_start,
            export_stage_wall_seconds=export_wall_final,
            export_stage_process_cpu_seconds=export_cpu_final,
            worker_returncode=native_returncode,
            native_launcher_started=native_started,
            worker_started=bool(isinstance(supervisor_evidence, dict) and isinstance(supervisor_evidence.get("pid"), int)),
            launch_error=launch_error,
            export_error=export_error,
            combined_budget_wall_upper_bound_seconds=combined_wall_upper,
            combined_budget_cpu_upper_bound_seconds=complete_cpu_upper,
            automatic_retry=False,
        )
        durable_atomic_json(status_path, status)
        final_wall_upper = args.windows_preflight_wall_seconds + time.monotonic() - linux_wall_start + args.cross_system_wall_reserve_seconds
        if final_wall_upper > request["totals"]["wall_seconds"]:
            status.update(status="stopped", stage="wall_budget_failed", stop_reason="final combined wall upper bound exceeded")
            durable_atomic_json(status_path, status)
            return 1
        return 0 if status["status"] == "complete" else 1
    except BaseException as exc:
        disarm_wall_deadline()
        failure = {
            "schema": "w1-linux-launcher-status/3.0.0",
            "attempt": args.attempt,
            "status": "technical_stop",
            "stage": stage,
            "error": type(exc).__name__ + ": " + str(exc),
            "linux_wall_seconds": time.monotonic() - linux_wall_start,
            "linux_process_cpu_seconds": time.process_time() - linux_cpu_start,
            "windows_preflight_wall_seconds": args.windows_preflight_wall_seconds,
            "windows_preflight_process_tree_cpu_seconds": args.windows_preflight_cpu_seconds,
            "worker_started": False,
            "automatic_retry": False,
        }
        if native_root.exists():
            safe_status(native_root / "launcher-status.json", failure, fallback)
        else:
            try:
                durable_atomic_json(fallback, failure)
            except BaseException:
                pass
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1
    finally:
        if export_timer_armed:
            disarm_wall_deadline()
        token = ""


if __name__ == "__main__":
    raise SystemExit(main())
