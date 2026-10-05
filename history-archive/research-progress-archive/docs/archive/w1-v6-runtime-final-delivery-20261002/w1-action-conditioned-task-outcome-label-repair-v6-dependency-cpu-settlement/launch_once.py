"""Windows entrypoint and read-only preflight for the sealed package."""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from pathlib import PurePosixPath

PROCESS_STARTED_WALL = time.monotonic()
PROCESS_STARTED_CPU = time.process_time()

from manifest_contract import PackageContractError, verify_external_authorization

ROOT = Path(__file__).resolve().parent


def verify_authorization(root: Path, authorization_file: Path, *, token: str | None, preflight_only: bool) -> tuple[dict, dict, dict]:
    if os.name != "nt":
        raise PackageContractError("WINDOWS_ENTRY_REQUIRES_WINDOWS")
    verified = verify_external_authorization(
        root, authorization_file, token=token, preflight_only=preflight_only,
    )
    return verified["identity"]["manifest"], verified["contract"], verified["request"]


def windows_to_wsl(wsl_exe: str, distribution: str, path: Path) -> str:
    result = subprocess.run([wsl_exe, "--distribution", distribution, "--exec", "/usr/bin/wslpath", "-a", "-u", str(path)], stdin=subprocess.DEVNULL, capture_output=True, check=False)
    if result.returncode:
        raise PackageContractError("WSPATH_CONVERSION_FAILED")
    converted = result.stdout.decode("utf-8", errors="strict").strip()
    if not converted.startswith("/mnt/") or "\x00" in converted:
        raise PackageContractError("WSPATH_RESULT_INVALID")
    return converted


def _children_cpu_snapshot():
    if os.name == "nt":
        return None
    import resource

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def _child_process_cpu_seconds(child: subprocess.Popen, before: float) -> float:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class FILETIME(ctypes.Structure):
            _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

        creation, exit_time, kernel, user = FILETIME(), FILETIME(), FILETIME(), FILETIME()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetProcessTimes.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME),
        )
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        if not kernel32.GetProcessTimes(
            wintypes.HANDLE(int(child._handle)),
            ctypes.byref(creation), ctypes.byref(exit_time),
            ctypes.byref(kernel), ctypes.byref(user),
        ):
            raise OSError(ctypes.get_last_error(), "GetProcessTimes failed for WSL child")

        def seconds(value: FILETIME) -> float:
            ticks = (value.high << 32) | value.low
            return ticks / 10_000_000.0

        return seconds(kernel) + seconds(user)

    import resource

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime - before


def run_preflight_probe(wsl_exe: str, distribution: str, native_python: str,
                        source_wsl: str, authorization_windows: Path,
                        *, timeout_seconds: float) -> dict:
    command = [
        wsl_exe, "--distribution", distribution, "--exec", native_python, "-B",
        f"{source_wsl}/launcher_preflight_probe.py",
        "--authorization-windows-path", str(authorization_windows),
        "--expected-python", native_python,
    ]
    children_before = _children_cpu_snapshot()
    options = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "shell": False,
    }
    if os.name == "nt":
        options["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        options["start_new_session"] = True
    child = subprocess.Popen(command, **options)
    try:
        stdout, _stderr = child.communicate(timeout=max(0.001, timeout_seconds))
    except subprocess.TimeoutExpired as exc:
        _terminate_process_tree(child, grace_seconds=0.5)
        try:
            child.communicate(timeout=0.5)
        except subprocess.TimeoutExpired:
            child.kill()
            try:
                child.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        raise PackageContractError("RUNTIME_DEPENDENCY_PROBE_TIMEOUT") from exc
    child_cpu = _child_process_cpu_seconds(child, children_before)
    if child.returncode:
        raise PackageContractError("RUNTIME_DEPENDENCY_PROBE_FAILED")
    try:
        payload = json.loads(stdout.decode("utf-8", errors="strict").strip().splitlines()[-1])
    except (UnicodeDecodeError, json.JSONDecodeError, IndexError) as exc:
        raise PackageContractError("RUNTIME_DEPENDENCY_PROBE_OUTPUT_INVALID") from exc
    cpu = payload.get("process_tree_cpu_seconds")
    dependency = payload.get("dependency_probe")
    if not isinstance(dependency, dict):
        raise PackageContractError("RUNTIME_DEPENDENCY_PROBE_PAYLOAD_MISSING")
    if (dependency.get("synthetic_kernel_matmuls") != 1
            or dependency.get("synthetic_kernel_backwards") != 0
            or dependency.get("synthetic_optimizer_updates") != 0
            or dependency.get("model_forwards") != 0):
        raise PackageContractError("RUNTIME_DEPENDENCY_PROBE_SYNTHETIC_COUNT_MISMATCH")
    if (payload.get("status") != "pass" or payload.get("model_initialized")
            or payload.get("checkpoint_loaded") or payload.get("training_started")
            or type(cpu) not in (int, float) or not math.isfinite(cpu) or cpu < 0):
        raise PackageContractError("RUNTIME_DEPENDENCY_PROBE_CONTRACT_FAILED")
    payload["wsl_executable_process_cpu_seconds"] = child_cpu
    return payload


def build_command(contract: dict, source_wsl: str, authorization_wsl: str,
                  preflight_wall: float, preflight_cpu: float, *,
                  request: dict,
                  preflight_only: bool, dependency_only: bool) -> list[str]:
    command = [contract["wsl_executable"], "--distribution", contract["wsl_distribution"], "--exec", contract["native_python"], "-B", f"{source_wsl}/{contract['linux_stage_entry']}", "--source", source_wsl, "--authorization-file", authorization_wsl, "--native-root", contract["native_execution_root"], "--export-root", contract["windows_export_root"], "--attempt", contract["attempt"], "--argument-probe", contract["argument_probe"], "--windows-preflight-wall-seconds", repr(preflight_wall), "--windows-preflight-cpu-seconds", repr(preflight_cpu)]
    if preflight_only:
        command.append("--preflight-only")
    if dependency_only:
        command.append("--dependency-only")
    command.extend((
        "--staging-stage-wall-seconds", str(request["stages"]["staging_and_zero_step_gate"]["wall_seconds"]),
        "--label-stage-wall-seconds", str(request["stages"]["label_qualification"]["wall_seconds"]),
        "--export-stage-wall-seconds", str(request["stages"]["settlement_and_verified_export"]["wall_seconds"]),
        "--total-wall-seconds", str(request["totals"]["wall_seconds"]),
        "--cross-system-wall-reserve-seconds", str(request["accounting_reserves"]["cross_system_wall_seconds"]),
    ))
    return command


def _terminate_process_tree(child: subprocess.Popen, *, grace_seconds: float = 0.5) -> None:
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=False, timeout=grace_seconds,
            )
        except subprocess.TimeoutExpired:
            child.kill()
    else:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        child.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            child.kill()
        else:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        child.wait(timeout=grace_seconds)


def run_wsl(command: list[str], token: str, *, timeout_seconds: float) -> tuple[int, bool, float]:
    children_before = _children_cpu_snapshot()
    options = {"stdin": subprocess.PIPE, "stdout": None, "stderr": None, "shell": False}
    if os.name == "nt":
        options["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        options["start_new_session"] = True
    child = subprocess.Popen(command, **options)
    try:
        child.communicate(token.encode("utf-8"), timeout=max(0.001, timeout_seconds))
        return int(child.returncode), False, _child_process_cpu_seconds(child, children_before)
    except subprocess.TimeoutExpired:
        _terminate_process_tree(child)
        try:
            child.communicate(timeout=0.5)
        except subprocess.TimeoutExpired:
            child.kill()
            try:
                child.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        return 124, True, _child_process_cpu_seconds(child, children_before)


def windows_path_from_wsl_mount(value: str) -> Path:
    parts = PurePosixPath(value).parts
    if (not PurePosixPath(value).is_absolute() or len(parts) < 4
            or parts[1] != "mnt" or len(parts[2]) != 1 or not parts[2].isalpha()):
        raise PackageContractError("EXPORT_PATH_IS_NOT_AN_EXPLICIT_WINDOWS_MOUNT")
    return Path(f"{parts[2].upper()}:\\").joinpath(*parts[3:])


def verify_completed_run_accounting(
    request: dict, contract: dict, export_status: dict, *,
    windows_preflight_cpu_seconds: float, windows_preflight_process_cpu_seconds: float,
    windows_total_cpu_seconds: float, windows_post_preflight_child_cpu_seconds: float,
    windows_total_wall_seconds: float,
) -> dict:
    if export_status.get("attempt") != contract.get("attempt"):
        raise PackageContractError("EXPORTED_ACCOUNTING_ATTEMPT_MISMATCH")
    if export_status.get("status") != "verified":
        raise PackageContractError("EXPORTED_ACCOUNTING_STATUS_NOT_VERIFIED")
    linux_cpu_upper = export_status.get("combined_budget_cpu_upper_bound_seconds")
    linux_wall = export_status.get("combined_budget_wall_seconds")
    combined_wall_upper = export_status.get("combined_budget_wall_upper_bound_seconds")
    preflight_wall = export_status.get("windows_preflight_wall_seconds")
    linux_wall_snapshot = export_status.get("linux_wall_snapshot_seconds")
    export_wall_snapshot = export_status.get("export_stage_wall_snapshot_seconds")
    export_wall_cap = export_status.get("export_stage_wall_cap_seconds")
    native_tree_cpu = export_status.get("native_process_tree_cpu_seconds")
    supervisor_cpu = export_status.get("supervisor_complete_process_cpu_seconds")
    cpu_scope = export_status.get("native_launcher_cpu_scope_breakdown")
    linux_launcher_cpu = export_status.get("linux_launcher_process_cpu_seconds")
    export_cpu_snapshot = export_status.get("export_stage_cpu_snapshot_seconds")
    export_cpu_cap = export_status.get("export_stage_cpu_cap_seconds")
    reported_preflight_cpu = export_status.get("windows_preflight_process_tree_cpu_seconds")
    values = (linux_cpu_upper, linux_wall, combined_wall_upper, preflight_wall,
              linux_wall_snapshot, export_wall_snapshot, export_wall_cap,
              native_tree_cpu, supervisor_cpu,
              linux_launcher_cpu, export_cpu_snapshot, export_cpu_cap,
              reported_preflight_cpu, windows_preflight_cpu_seconds,
              windows_preflight_process_cpu_seconds, windows_total_cpu_seconds,
              windows_post_preflight_child_cpu_seconds,
              windows_total_wall_seconds)
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
        raise PackageContractError("COMPLETED_RUN_ACCOUNTING_VALUE_INVALID")
    post_preflight_process_cpu = windows_total_cpu_seconds - windows_preflight_process_cpu_seconds
    post_preflight_cpu = post_preflight_process_cpu + windows_post_preflight_child_cpu_seconds
    cross_system_wall = windows_total_wall_seconds - linux_wall
    if post_preflight_cpu < 0 or cross_system_wall < 0:
        raise PackageContractError("COMPLETED_RUN_ACCOUNTING_CLOCK_ORDER_INVALID")
    reserves = request.get("accounting_reserves", {})
    windows_post_reserve = reserves.get("windows_post_preflight_process_cpu_seconds", -1)
    if abs(reported_preflight_cpu - windows_preflight_cpu_seconds) > 0.02:
        raise PackageContractError("WINDOWS_PREFLIGHT_CPU_IDENTITY_MISMATCH")
    if not isinstance(cpu_scope, dict) or cpu_scope.get("schema") != "w1-native-cpu-scope-breakdown/1.0.0":
        raise PackageContractError("NATIVE_CPU_SCOPE_BREAKDOWN_MISSING")
    if cpu_scope.get("status") != "reconciled" or cpu_scope.get("errors") != []:
        raise PackageContractError("NATIVE_CPU_SCOPE_BREAKDOWN_NOT_RECONCILED")
    scope_tree = cpu_scope.get("native_process_tree_cpu_seconds")
    scope_supervisor = cpu_scope.get("supervisor_complete_process_cpu_seconds_nested_snapshot")
    identity_values = (scope_tree, scope_supervisor)
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) < 0 for value in identity_values):
        raise PackageContractError("NATIVE_CPU_SCOPE_IDENTITY_VALUE_INVALID")
    if abs(float(scope_tree) - native_tree_cpu) > 1e-6:
        raise PackageContractError("NATIVE_CPU_SCOPE_TREE_IDENTITY_MISMATCH")
    if abs(float(scope_supervisor) - supervisor_cpu) > 1e-6:
        raise PackageContractError("NATIVE_CPU_SCOPE_SUPERVISOR_IDENTITY_MISMATCH")
    launcher_self_cpu = cpu_scope.get("native_launcher_process_cpu_seconds_nested_component")
    launcher_children_cpu = cpu_scope.get("native_launcher_reaped_children_cpu_seconds_nested_component")
    launcher_snapshot_cpu = cpu_scope.get("native_launcher_complete_process_cpu_snapshot_seconds")
    completion_delta = cpu_scope.get("supervisor_sample_to_native_launcher_snapshot_cpu_seconds")
    launcher_exit_tail = cpu_scope.get("native_launcher_snapshot_to_parent_reap_cpu_seconds")
    scope_values = (launcher_self_cpu, launcher_children_cpu, launcher_snapshot_cpu, completion_delta, launcher_exit_tail)
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in scope_values):
        raise PackageContractError("NATIVE_CPU_SCOPE_COMPONENT_INVALID")
    if launcher_self_cpu < 0 or launcher_children_cpu < 0 or launcher_snapshot_cpu < 0:
        raise PackageContractError("NATIVE_CPU_SCOPE_COMPONENT_NEGATIVE")
    if abs(launcher_snapshot_cpu - launcher_self_cpu - launcher_children_cpu) > 1e-6:
        raise PackageContractError("NATIVE_LAUNCHER_CPU_COMPONENT_SUM_MISMATCH")
    if completion_delta < -0.05 or launcher_exit_tail < -0.05 or launcher_exit_tail > 0.05:
        raise PackageContractError("NATIVE_CPU_SCOPE_TAIL_OUTSIDE_FROZEN_TOLERANCE")
    if abs(native_tree_cpu - supervisor_cpu - completion_delta - launcher_exit_tail) > 1e-6:
        raise PackageContractError("NATIVE_CPU_SCOPE_DECOMPOSITION_MISMATCH")
    wall_remaining = export_wall_cap - export_wall_snapshot
    if wall_remaining < 0:
        raise PackageContractError("EXPORT_STAGE_WALL_SNAPSHOT_OVER_CAP")
    expected_wall_upper = (
        preflight_wall + linux_wall_snapshot + wall_remaining
        + reserves.get("cross_system_wall_seconds", -1)
    )
    if abs(combined_wall_upper - expected_wall_upper) > 0.02:
        raise PackageContractError("WALL_UPPER_BOUND_COMPOSITION_MISMATCH")
    if combined_wall_upper > request["totals"]["wall_seconds"]:
        raise PackageContractError("COMBINED_WALL_UPPER_BOUND_EXCEEDED")
    if export_cpu_snapshot > export_cpu_cap:
        raise PackageContractError("EXPORT_STAGE_CPU_SNAPSHOT_OVER_CAP")
    finalization_remaining = export_cpu_cap - export_cpu_snapshot
    expected_complete_upper = (
        windows_preflight_cpu_seconds + linux_launcher_cpu + native_tree_cpu
        + finalization_remaining + windows_post_reserve
    )
    if abs(linux_cpu_upper - expected_complete_upper) > 0.02:
        raise PackageContractError("LINUX_CPU_UPPER_BOUND_COMPOSITION_MISMATCH")
    complete_cpu_upper = linux_cpu_upper
    measured = {
        "attempt": contract["attempt"],
        "windows_preflight_process_cpu_seconds": windows_preflight_process_cpu_seconds,
        "windows_preflight_process_tree_cpu_seconds": windows_preflight_cpu_seconds,
        "windows_post_preflight_process_cpu_observed_seconds": post_preflight_cpu,
        "windows_post_preflight_process_cpu_self_seconds": post_preflight_process_cpu,
        "windows_post_preflight_wsl_process_cpu_seconds": windows_post_preflight_child_cpu_seconds,
        "windows_post_preflight_process_cpu_upper_bound_seconds": windows_post_reserve,
        "native_process_tree_cpu_seconds": native_tree_cpu,
        "supervisor_complete_process_cpu_seconds_for_consistency_only": supervisor_cpu,
        "native_launcher_process_cpu_seconds_nested_component": launcher_self_cpu,
        "native_launcher_complete_process_cpu_snapshot_seconds": launcher_snapshot_cpu,
        "supervisor_to_native_launcher_completion_cpu_seconds": completion_delta,
        "native_launcher_snapshot_to_parent_reap_cpu_seconds": launcher_exit_tail,
        "native_cpu_scope_breakdown": cpu_scope,
        "linux_launcher_process_cpu_seconds": linux_launcher_cpu,
        "linux_complete_process_cpu_upper_bound_seconds": linux_cpu_upper,
        "complete_process_cpu_upper_bound_seconds": complete_cpu_upper,
        "linux_combined_wall_seconds": linux_wall,
        "windows_total_wall_seconds": windows_total_wall_seconds,
        "cross_system_wall_seconds": cross_system_wall,
        "cpu_budget_seconds": request["totals"]["complete_process_cpu_seconds"],
        "wall_budget_seconds": request["totals"]["wall_seconds"],
        "windows_post_preflight_cpu_reserve_seconds": windows_post_reserve,
        "export_stage_finalization_cpu_upper_bound_seconds": finalization_remaining,
        "cross_system_wall_reserve_seconds": reserves.get("cross_system_wall_seconds"),
    }
    failures = []
    if post_preflight_cpu > windows_post_reserve:
        failures.append("windows_post_preflight_cpu_reserve_exceeded")
    if cross_system_wall > reserves.get("cross_system_wall_seconds", -1):
        failures.append("cross_system_wall_reserve_exceeded")
    if complete_cpu_upper > measured["cpu_budget_seconds"]:
        failures.append("complete_process_cpu_budget_exceeded")
    if windows_total_wall_seconds > measured["wall_budget_seconds"]:
        failures.append("complete_wall_budget_exceeded")
    measured["status"] = "pass" if not failures else "technical_stop"
    measured["failures"] = failures
    return measured


def main() -> int:
    started = PROCESS_STARTED_WALL
    process_cpu_started = PROCESS_STARTED_CPU
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-token-file")
    parser.add_argument("--authorization-file", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dependency-only", action="store_true")
    args = parser.parse_args()
    try:
        authorization_path = Path(args.authorization_file).resolve()
        token = ""
        if args.preflight_only and args.dependency_only:
            raise PackageContractError("PREFLIGHT_AND_DEPENDENCY_ONLY_ARE_EXCLUSIVE")
        if not args.preflight_only:
            if not args.attempt_token_file:
                raise PackageContractError("TOKEN_FILE_REQUIRED")
            token_path = Path(args.attempt_token_file)
            if not token_path.is_file() or token_path.resolve().is_relative_to(ROOT.resolve()):
                raise PackageContractError("TOKEN_FILE_INVALID")
            token = token_path.read_text(encoding="utf-8").strip()
        manifest, contract, request = verify_authorization(ROOT, authorization_path, token=token, preflight_only=args.preflight_only)
        source_wsl = contract["wsl_source_root"]
        if (not source_wsl.startswith("/mnt/")
                or os.path.normcase(str(ROOT.resolve()))
                != os.path.normcase(str(Path(contract["windows_source_root"]).resolve()))):
            raise PackageContractError("WSL_SOURCE_IDENTITY_MISMATCH")
        stage0_wall_remaining = (
            request["stages"]["staging_and_zero_step_gate"]["wall_seconds"]
            - (time.monotonic() - started)
        )
        if stage0_wall_remaining <= 0:
            raise PackageContractError("WINDOWS_PREFLIGHT_EXHAUSTED_STAGING_STAGE_WALL")
        dependency = run_preflight_probe(
            contract["wsl_executable"], contract["wsl_distribution"],
            contract["native_python"], source_wsl, authorization_path,
            timeout_seconds=stage0_wall_remaining,
        )
        preflight_wall = time.monotonic() - started
        preflight_process_cpu = time.process_time() - process_cpu_started
        preflight_cpu = (
            preflight_process_cpu + dependency["process_tree_cpu_seconds"]
            + dependency["wsl_executable_process_cpu_seconds"]
        )
        staging_limits = request["stages"]["staging_and_zero_step_gate"]
        if preflight_wall > staging_limits["wall_seconds"] or preflight_cpu > staging_limits["complete_process_cpu_seconds"]:
            raise PackageContractError("WINDOWS_PREFLIGHT_EXHAUSTED_STAGING_STAGE_BUDGET")
        command = build_command(contract, source_wsl, dependency["authorization_wsl_path"], preflight_wall,
                                         preflight_cpu, preflight_only=args.preflight_only,
                                         dependency_only=args.dependency_only, request=request)
        timeout_seconds = 120.0
        if not args.preflight_only:
            total_wall = float(request["totals"]["wall_seconds"])
            cross_reserve = float(request["accounting_reserves"]["cross_system_wall_seconds"])
            timeout_seconds = total_wall - cross_reserve - preflight_wall
            if timeout_seconds <= 0:
                raise PackageContractError("NO_WALL_BUDGET_REMAINS_BEFORE_WSL_LAUNCH")
        wsl_code, wsl_timed_out, wsl_child_cpu = run_wsl(command, token, timeout_seconds=timeout_seconds)
        code = wsl_code
        windows_total_wall = time.monotonic() - started
        windows_total_cpu = time.process_time() - process_cpu_started
        accounting = None
        if not args.preflight_only and not args.dependency_only and code == 0:
            export_status_path = windows_path_from_wsl_mount(contract["windows_export_root"]) / "export-status.json"
            try:
                export_status = json.loads(export_status_path.read_text(encoding="utf-8"))
                accounting = verify_completed_run_accounting(
                    request, contract, export_status,
                    windows_preflight_cpu_seconds=preflight_cpu,
                    windows_preflight_process_cpu_seconds=preflight_process_cpu,
                    windows_total_cpu_seconds=windows_total_cpu,
                    windows_post_preflight_child_cpu_seconds=wsl_child_cpu,
                    windows_total_wall_seconds=windows_total_wall,
                )
                windows_total_cpu = time.process_time() - process_cpu_started
                windows_total_wall = time.monotonic() - started
                accounting = verify_completed_run_accounting(
                    request, contract, export_status,
                    windows_preflight_cpu_seconds=preflight_cpu,
                    windows_preflight_process_cpu_seconds=preflight_process_cpu,
                    windows_total_cpu_seconds=windows_total_cpu,
                    windows_post_preflight_child_cpu_seconds=wsl_child_cpu,
                    windows_total_wall_seconds=windows_total_wall,
                )
            except BaseException as exc:
                accounting = {"status": "technical_stop", "error": f"{type(exc).__name__}: {exc}"}
            if accounting.get("status") != "pass":
                code = 1
        dependency_summary = dependency.get("dependency_probe", {})
        print(json.dumps({
            "status": "preflight_pass" if args.preflight_only and code == 0 else "completed" if code == 0 else "technical_stop",
            "attempt": manifest["attempt"],
            "wsl_returncode": wsl_code,
            "launcher_returncode": code,
            "wsl_timed_out": wsl_timed_out,
            "dependency_probe": {
                "torch": dependency_summary.get("torch_version"),
                "python": dependency_summary.get("python_executable"),
                "cpu_only": dependency_summary.get("cpu_only"),
                "synthetic_kernel_matmuls": dependency_summary.get("synthetic_kernel_matmuls"),
                "synthetic_kernel_backwards": dependency_summary.get("synthetic_kernel_backwards"),
                "synthetic_optimizer_updates": dependency_summary.get("synthetic_optimizer_updates"),
                "model_forwards": dependency_summary.get("model_forwards"),
                "remote_runtime_binding": dependency_summary.get("remote_runtime_binding"),
            },
            "staging_started": False if args.preflight_only else "unknown",
            "worker_started": False if args.preflight_only or args.dependency_only else "unknown",
            "elapsed_wall_seconds": windows_total_wall,
            "windows_process_cpu_seconds": windows_total_cpu,
            "windows_wsl_child_process_cpu_seconds": wsl_child_cpu,
            "complete_resource_accounting": accounting,
        }, sort_keys=True))
        return code
    except BaseException as exc:
        print(json.dumps({"schema": "w1-windows-launcher-failure/2.0.0", "status": "technical_stop", "stage": "windows_preflight", "error": f"{type(exc).__name__}: {exc}", "staging_started": False, "worker_started": False, "automatic_retry": False, "elapsed_wall_seconds": time.monotonic() - started}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
