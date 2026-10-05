"""Authorized one-shot staging, native execution, and controlled export."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import signal
import subprocess
import time
from pathlib import Path

from infra_io import (
    controlled_export,
    durable_atomic_json,
    read_json,
    sha256_file,
    stage_sealed_package,
    tree_manifest,
    verify_manifest,
)

ROOT = Path(__file__).resolve().parent


def verify_source(token: str, root: Path = ROOT) -> tuple[dict, dict, dict]:
    """Validate frozen identities and authorization before creating state."""

    hashes = read_json(root / "hashes.json")
    manifest_path = root / "execution-manifest.json"
    if sha256_file(manifest_path) != hashes["execution_manifest_sha256"]:
        raise RuntimeError("Execution manifest SHA-256 mismatch")
    manifest = read_json(manifest_path)
    verify_manifest(root, manifest["files"])
    contract = read_json(root / "launch-contract.json")
    request = read_json(root / "RESOURCE_REQUEST.json")
    if request.get("status") != "NOT_APPROVED":
        raise RuntimeError("Frozen request status changed")
    if not (manifest.get("attempt") == request.get("attempt") == contract.get("attempt")):
        raise RuntimeError("Attempt identity mismatch")
    supplied = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(supplied, contract["external_token_sha256"]):
        raise RuntimeError("Exact external one-shot token mismatch")
    return manifest, contract, request


def _stage_cap(request: dict, stage: str, wall: float, cpu: float) -> None:
    limits = request["stages"][stage]
    if wall > limits["wall_seconds"] or cpu > limits["complete_process_cpu_seconds"]:
        raise RuntimeError(f"{stage} infrastructure resource limit")


def _run_native(python: str, root: Path, token: str) -> tuple[int, str | None]:
    environment = dict(os.environ)
    environment["W1_EXTERNAL_ATTEMPT_TOKEN"] = token
    child = subprocess.Popen(
        [python, "-B", str(root / "native_launch.py")],
        cwd=root,
        env=environment,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    interrupted: str | None = None
    previous = {}

    def forward(sig, _frame):
        nonlocal interrupted
        interrupted = f"launcher received signal {sig}"
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, forward)
    try:
        return child.wait(), interrupted
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def orchestrate(
    token: str,
    *,
    root: Path = ROOT,
    run_native=_run_native,
    exporter=controlled_export,
) -> int:
    manifest, contract, request = verify_source(token, root)
    native_root = Path(contract["native_execution_root"])
    export_root = Path(contract["windows_export_root"])
    if native_root.exists():
        raise RuntimeError("Native attempt path already exists; no retry")
    if export_root.exists():
        raise RuntimeError("Export path already exists; no overwrite or retry")

    infra_cpu_started = time.process_time()
    staging_started = time.monotonic()
    staged = stage_sealed_package(root, native_root, manifest)
    staging_wall = time.monotonic() - staging_started
    staging_cpu = time.process_time() - infra_cpu_started
    _stage_cap(request, "staging_and_zero_step_gate", staging_wall, staging_cpu)
    durable_atomic_json(
        native_root / "staging-record.json",
        {
            "schema": "native-staging-record/1.0.0",
            "attempt": contract["attempt"],
            "staging_wall_seconds": staging_wall,
            "staging_cpu_seconds": staging_cpu,
            **staged,
        },
    )

    native_returncode = 1
    interruption = None
    launch_error = None
    try:
        native_returncode, interruption = run_native(
            contract["native_python"], native_root, token
        )
    except BaseException as exc:
        launch_error = type(exc).__name__ + ": " + str(exc)

    durable_atomic_json(
        native_root / "infrastructure-settlement.json",
        {
            "schema": "native-infrastructure-settlement/1.0.0",
            "attempt": contract["attempt"],
            "native_returncode": native_returncode,
            "launcher_interruption": interruption,
            "launch_error": launch_error,
            "worker_relaunched": False,
            "automatic_retry": False,
            "historical_credit": 0,
            "staging_wall_seconds": staging_wall,
            "staging_cpu_seconds": staging_cpu,
            "export_status_at_copy": "pending",
        },
    )

    export_started = time.monotonic()
    export_cpu_started = time.process_time()
    export_error = None
    try:
        export_result = exporter(native_root, export_root)
    except BaseException as exc:
        export_result = None
        export_error = type(exc).__name__ + ": " + str(exc)
    export_wall = time.monotonic() - export_started
    export_cpu = time.process_time() - export_cpu_started
    try:
        _stage_cap(request, "verified_export", export_wall, export_cpu)
        if export_result is None:
            raise RuntimeError("Verified export result unavailable")
        supervisor_status = read_json(native_root / "supervisor-status.json")
        total_wall = (
            staging_wall
            + float(supervisor_status["wall_seconds"])
            + export_wall
        )
        total_cpu = (
            staging_cpu
            + float(supervisor_status["cpu_seconds"])
            + export_cpu
        )
        if total_wall > request["totals"]["wall_seconds"]:
            raise RuntimeError("Combined wall limit exceeded")
        if total_cpu > request["totals"]["complete_process_cpu_seconds"]:
            raise RuntimeError("Combined complete-process CPU limit exceeded")
        native_bytes = sum(
            row["bytes"] for row in tree_manifest(native_root).values()
        )
        aggregate_bytes = native_bytes + int(export_result["payload_bytes"])
        aggregate_limit = int(
            request["totals"]["aggregate_native_plus_verified_export_bytes"]
        )
        aggregate_reserve = int(
            request["shutdown_reserve"]["artifact_bytes_within_active_limit"]
        )
        if aggregate_bytes > aggregate_limit - aggregate_reserve:
            raise RuntimeError("Aggregate native plus export storage reserve reached")
    except BaseException as exc:
        export_error = export_error or type(exc).__name__ + ": " + str(exc)

    durable_atomic_json(
        native_root / "export-status.json",
        {
            "schema": "native-export-status/1.0.0",
            "attempt": contract["attempt"],
            "status": "verified" if export_error is None else "failed",
            "result": export_result,
            "error": export_error,
            "export_wall_seconds": export_wall,
            "export_cpu_seconds": export_cpu,
            "combined_wall_seconds": locals().get("total_wall"),
            "combined_complete_process_cpu_seconds": locals().get("total_cpu"),
            "aggregate_native_plus_export_bytes": locals().get("aggregate_bytes"),
            "worker_relaunched": False,
            "automatic_retry": False,
        },
    )
    if launch_error or export_error or native_returncode != 0:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-token", required=True)
    args = parser.parse_args()
    return orchestrate(args.attempt_token)


if __name__ == "__main__":
    raise SystemExit(main())
