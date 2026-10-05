"""Linux-only staging, native supervision, settlement, and controlled export."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import infra_io
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
    parser.add_argument("--native-root", required=True)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--argument-probe", required=True)
    parser.add_argument("--windows-preflight-wall-seconds", type=float, required=True)
    parser.add_argument("--windows-preflight-cpu-seconds", type=float, required=True)
    return parser.parse_args()


def stage_cap(request: dict, stage: str, wall: float, cpu: float) -> None:
    limits = request["stages"][stage]
    if wall > limits["wall_seconds"] or cpu > limits["complete_process_cpu_seconds"]:
        raise RuntimeError(f"{stage} infrastructure resource limit")


def validate_linux_contract(args: argparse.Namespace, token: str):
    if os.name != "posix" or not Path("/proc/self/mountinfo").is_file():
        raise RuntimeError("Linux staging entry requires a real Linux process")
    source = Path(args.source)
    native_root = Path(args.native_root)
    export_root = Path(args.export_root)
    if not all(path.is_absolute() for path in (source, native_root, export_root)):
        raise RuntimeError("All WSL paths must be absolute")
    if not str(source).startswith("/mnt/") or not str(export_root).startswith("/mnt/"):
        raise RuntimeError("Source and export must be explicit WSL mounted paths")
    if str(native_root).startswith("/mnt/"):
        raise RuntimeError("Native active root cannot use a mounted Windows filesystem")
    hashes = read_json(source / "hashes.json")
    if sha256_file(source / "execution-manifest.json") != hashes["execution_manifest_sha256"]:
        raise RuntimeError("Execution manifest SHA-256 mismatch in Linux")
    manifest = read_json(source / "execution-manifest.json")
    verify_manifest(source, manifest["files"])
    request = read_json(source / "RESOURCE_REQUEST.json")
    contract = read_json(source / "launch-contract.json")
    if request.get("status") != "NOT_APPROVED":
        raise RuntimeError("Frozen request status changed")
    if not (args.attempt == manifest.get("attempt") == request.get("attempt") == contract.get("attempt")):
        raise RuntimeError("Attempt identity mismatch in Linux")
    if str(source.resolve()) != contract["wsl_source_root"]:
        raise RuntimeError("Linux source root identity mismatch")
    if str(native_root) != contract["native_execution_root"] or str(export_root) != contract["windows_export_root"]:
        raise RuntimeError("Linux launch path identity mismatch")
    if args.argument_probe != contract["argument_probe"]:
        raise RuntimeError("Structured argument probe changed")
    if Path(sys.executable).resolve() != Path(contract["native_python"]).resolve():
        raise RuntimeError("Unexpected WSL Python interpreter")
    supplied = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if not token or not hmac.compare_digest(supplied, contract["external_token_sha256"]):
        raise RuntimeError("Native launch authorization missing or invalid")
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
    return source, native_root, export_root, manifest, contract, request


def run_native(
    python: str,
    root: Path,
    token: str,
    entry: str,
    *,
    inject_interrupt: bool = False,
) -> tuple[int, str | None, str | None, bool]:
    environment = dict(os.environ)
    environment["W1_EXTERNAL_ATTEMPT_TOKEN"] = token
    interruption = None
    child = None
    previous = {}
    try:
        child = subprocess.Popen(
            [python, "-B", str(root / entry)],
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
        return int(child.wait()), interruption, None, True
    except BaseException as exc:
        return 1, interruption, type(exc).__name__ + ": " + str(exc), child is not None
    finally:
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


def main() -> int:
    linux_wall_start = time.monotonic()
    linux_cpu_start = time.process_time()
    args = parse_args()
    token = sys.stdin.buffer.read().decode("utf-8", errors="strict")
    native_root = Path(args.native_root)
    fallback = native_root.parent / f".{args.attempt}.launcher-failure.json"
    stage = "linux_preflight"
    try:
        source, native_root, export_root, manifest, contract, request = validate_linux_contract(args, token)
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
        entry = contract.get("native_entry", "native_launch.py")
        python = contract.get("native_child_python", contract["native_python"])
        native_returncode, interruption, launch_error, native_launcher_started = run_native(
            python,
            native_root,
            token,
            entry,
            inject_interrupt=bool(
                contract.get("integration_test")
                and contract.get("integration_fault") == "native_interrupt"
            ),
        )
        token = ""
        durable_atomic_json(native_root / "infrastructure-settlement.json", {
            "schema": "native-infrastructure-settlement/2.0.0",
            "attempt": args.attempt,
            "native_returncode": native_returncode,
            "launcher_interruption": interruption,
            "launch_error": launch_error,
            "native_launcher_started": native_launcher_started,
            "worker_relaunched": False,
            "automatic_retry": False,
            "historical_credit": 0,
            "staging_wall_seconds": staging_wall,
            "staging_cpu_seconds": staging_cpu,
            "export_status_at_copy": "pending",
        })
        stage = "verified_export"
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
            stage_cap(request, "verified_export", export_wall, export_cpu)
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
                args.windows_preflight_cpu_seconds + linux_cpu + supervisor_cpu
            )
            native_bytes = sum(row["bytes"] for row in tree_manifest(native_root).values())
            aggregate_bytes = native_bytes + int(export_result["payload_bytes"])
            if combined_budget_wall > request["totals"]["wall_seconds"] or combined_budget_cpu > request["totals"]["complete_process_cpu_seconds"]:
                raise RuntimeError("Combined Linux infrastructure limit exceeded")
            if aggregate_bytes > request["totals"]["aggregate_native_plus_verified_export_bytes"] - request["shutdown_reserve"]["artifact_bytes_within_active_limit"]:
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
            "combined_budget_scope": "Windows preflight CPU plus Linux launcher CPU plus supervisor complete-process CPU; wall is Windows preflight plus Linux entry through export-status creation",
            "cross_process_start_gap": "unavailable",
            "windows_post_preflight_cpu_seconds": "unavailable in Linux child; Windows console reports launcher process CPU separately",
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
        run_status_path = native_root / "run-once" / "status.json"
        run_status = read_json(run_status_path) if run_status_path.is_file() else None
        final_settlement = {
            "schema": "w1-final-settlement/1.0.0",
            "attempt": args.attempt,
            "overall_status": status["status"],
            "launcher_stage": status["stage"],
            "worker_returncode": native_returncode,
            "worker_status": run_status.get("status") if isinstance(run_status, dict) else "unavailable",
            "worker_error": run_status.get("error") if isinstance(run_status, dict) else "unavailable",
            "supervisor_status": (
                supervisor_evidence.get("status")
                if isinstance(supervisor_evidence, dict)
                else "unavailable"
            ),
            "supervisor_final_resource_pass": (
                supervisor_evidence.get("final_resource_pass")
                if isinstance(supervisor_evidence, dict)
                else "unavailable"
            ),
            "controlled_export_status": export_status["status"],
            "launcher_status_sha256": sha256_file(status_path),
            "supervisor_status_sha256": (
                sha256_file(native_root / "supervisor-status.json")
                if (native_root / "supervisor-status.json").is_file()
                else None
            ),
            "worker_status_sha256": sha256_file(run_status_path) if run_status_path.is_file() else None,
            "automatic_retry": False,
            "worker_relaunched": False,
        }
        final_settlement_path = native_root / "FINAL_SETTLEMENT.json"
        durable_atomic_json(final_settlement_path, final_settlement)
        if export_error is None:
            try:
                export_result = append_verified_export_file(
                    final_settlement_path,
                    export_root,
                    "FINAL_SETTLEMENT.json",
                )
            except BaseException as exc:
                export_error = type(exc).__name__ + ": " + str(exc)
                status.update(status="stopped", stage="export_failed", export_error=export_error)
                durable_atomic_json(status_path, status)
                final_settlement.update(
                    overall_status="stopped",
                    launcher_stage="export_failed",
                    controlled_export_status="failed",
                    final_settlement_export_error=export_error,
                    launcher_status_sha256=sha256_file(status_path),
                )
                durable_atomic_json(final_settlement_path, final_settlement)
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


if __name__ == "__main__":
    raise SystemExit(main())
