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
    dependency = probe_runtime_dependencies(expected_python=contract["native_python"])
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
) -> tuple[int, str | None, str | None, bool, float]:
    environment = dict(os.environ)
    environment["W1_EXTERNAL_ATTEMPT_TOKEN"] = token
    environment["W1_EXTERNAL_AUTHORIZATION_FILE"] = authorization_file
    if entry == "integration_native_launch.py":
        environment["W1_TEST_ONLY_INTEGRATION"] = "1"
    interruption = None
    child = None
    previous = {}
    child_cpu_start = resource.getrusage(resource.RUSAGE_CHILDREN)

    def child_cpu_seconds() -> float:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        return max(
            0.0,
            (usage.ru_utime + usage.ru_stime)
            - (child_cpu_start.ru_utime + child_cpu_start.ru_stime),
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
            return int(child.wait(timeout=timeout_seconds)), interruption, None, True, child_cpu_seconds()
        except subprocess.TimeoutExpired:
            interruption = "native child exceeded frozen label-stage wall limit"
            stop_child()
            return 124, interruption, "TimeoutError: native child wall limit", True, child_cpu_seconds()
    except BaseException as exc:
        stop_child()
        return (
            1,
            interruption,
            type(exc).__name__ + ": " + str(exc),
            child is not None,
            child_cpu_seconds(),
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
        native_returncode, interruption, launch_error, native_launcher_started, native_child_cpu = run_native(
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
            print(json.dumps({"schema": "w1-linux-preflight/2.0.0", "status": "preflight_pass", "attempt": args.attempt, "staging_started": False, "worker_started": False}, sort_keys=True))
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
        native_returncode, interruption, launch_error, native_started, native_child_cpu = run_native(
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
        supervisor_path = native_root / "supervisor-status.json"
        supervisor_evidence = read_json(supervisor_path) if supervisor_path.is_file() else None
        supervisor_cpu = (
            float(supervisor_evidence["cpu_seconds"])
            if isinstance(supervisor_evidence, dict)
            and isinstance(supervisor_evidence.get("cpu_seconds"), (int, float))
            and math.isfinite(float(supervisor_evidence["cpu_seconds"]))
            else None
        )
        label_error = None
        if supervisor_cpu is not None and abs(native_child_cpu - supervisor_cpu) > 0.05:
            label_error = "Linux parent child-tree CPU differs from supervisor self-plus-child CPU"
        if label_wall > request["stages"]["label_qualification"]["wall_seconds"]:
            label_error = label_error or "label stage wall cap exceeded"
        if native_child_cpu > request["stages"]["label_qualification"]["complete_process_cpu_seconds"]:
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
            "supervisor_complete_process_cpu_seconds_for_consistency_only": supervisor_cpu,
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
            "schema": "native-export-status/3.0.0",
            "attempt": args.attempt,
            "status": "verified" if export_error is None else "failed",
            "result": export_result,
            "error": export_error,
            "staging_wall_seconds": staging_wall,
            "combined_staging_wall_seconds": combined_staging_wall,
            "combined_staging_process_cpu_seconds": combined_staging_cpu,
            "label_stage_wall_seconds": label_wall,
            "native_process_tree_cpu_seconds": native_child_cpu,
            "supervisor_complete_process_cpu_seconds": supervisor_cpu if supervisor_cpu is not None else "unavailable",
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
