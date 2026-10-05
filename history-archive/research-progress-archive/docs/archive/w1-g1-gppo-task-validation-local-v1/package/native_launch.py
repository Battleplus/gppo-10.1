"""Entry used only from the sealed package staged on native WSL storage."""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

from infra_io import durable_atomic_json, require_native_linux_filesystem
from manifest_contract import PackageContractError, sha256_file, verify_external_authorization
from dependency_probe import probe as probe_runtime_dependencies
from cpu_scope_contract import NATIVE_LAUNCHER_ACCOUNTING_RELATIVE_PATH

ROOT = Path(__file__).resolve().parent
PROCESS_MODULE_ENTRY_MONOTONIC = time.monotonic()


def verify_native_launch(
    root: Path = ROOT,
    *,
    allow_integration: bool = False,
    preflight_only: bool = False,
) -> str:
    require_native_linux_filesystem(root)
    token = os.environ.pop("W1_EXTERNAL_ATTEMPT_TOKEN", "")
    authorization_file = os.environ.pop("W1_EXTERNAL_AUTHORIZATION_FILE", "")
    if not authorization_file:
        raise RuntimeError("External authorization file path missing")
    try:
        verified = verify_external_authorization(
            root, Path(authorization_file), token=token, preflight_only=preflight_only,
        )
    except PackageContractError as exc:
        raise RuntimeError(f"AUTHORIZATION_OR_PACKAGE_CONTRACT_ERROR: {exc}") from exc
    manifest = verified["identity"]["manifest"]
    request = verified["request"]
    contract = verified["contract"]
    if contract.get("runtime_mode") == "remote_joint" and not preflight_only:
        raise RuntimeError("REMOTE_JOINT_REQUIRES_FROZEN_REMOTE_ENTRY")
    if contract.get("integration_test") and not allow_integration:
        raise RuntimeError("FORMAL_ENTRY_REJECTS_INTEGRATION_BACKEND")
    if request.get("status") != "NOT_APPROVED":
        raise RuntimeError("Frozen request status changed")
    if contract.get("resource_request_sha256") != sha256_file(root / "RESOURCE_REQUEST.json"):
        raise RuntimeError("Resource request identity mismatch")
    if not (request.get("attempt") == contract.get("attempt") == manifest.get("attempt")):
        raise RuntimeError("Attempt identity mismatch")
    if contract.get("runtime_mode") == "remote_joint":
        from joint_inputs import verify_joint_inputs

        verify_joint_inputs(root)
        dependency = None
    else:
        dependency = probe_runtime_dependencies(expected_python=contract["native_python"], expected_root=root)
    for name in ("run-once", "supervisor-status.json", "execution.lock", "runtime-output"):
        if (root / name).exists():
            raise RuntimeError("Attempt evidence already exists; retry prohibited")
    if not preflight_only:
        (root / "runtime-dependency-probe.json").write_text(__import__("json").dumps(dependency, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract["attempt"]


def main() -> int:
    raise RuntimeError('HISTORICAL_ENTRY_DISABLED_USE_RUN_G1_TASK_VALIDATION')
    dependency_only = "--dependency-only" in sys.argv[1:]
    preflight_only = "--preflight-only" in sys.argv[1:]
    stage = "native_preflight"
    result = "exception"
    error = None
    try:
        if dependency_only and preflight_only:
            raise RuntimeError("PREFLIGHT_AND_DEPENDENCY_ONLY_ARE_EXCLUSIVE")
        attempt = verify_native_launch(preflight_only=preflight_only)
        if preflight_only:
            result = "preflight_complete"
            runtime_mode = json.loads((ROOT / "launch-contract.json").read_text(encoding="utf-8")).get("runtime_mode")
            print(json.dumps({
                "schema": "w1-native-preflight/1.0.0",
                "status": "preflight_pass",
                "attempt": attempt,
                "runtime_mode": runtime_mode,
                "preflight_scope": (
                    "package_identity_and_joint_inputs_only"
                    if runtime_mode == "remote_joint"
                    else "package_identity_and_runtime_dependency_probe"
                ),
                "staging_started": False,
                "worker_started": False,
            }, sort_keys=True))
            return 0
        if dependency_only:
            result = "dependency_only_complete"
            print(json.dumps({"schema": "w1-native-dependency-only/1.0.0", "status": "pass", "worker_started": False, "environment_constructed": False, "model_initialized": False, "checkpoint_loaded": False, "training_started": False}, sort_keys=True))
            return 0
        from supervise import main as supervise_main

        # The supervisor passes this verified, non-secret identity to the worker.
        os.environ["W1_VERIFIED_ATTEMPT"] = attempt
        stage = "supervisor_and_worker"
        return_code = supervise_main(ROOT, "runner.py")
        result = "supervisor_returned"
        return return_code
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if not preflight_only:
            try:
                self_usage = resource.getrusage(resource.RUSAGE_SELF)
                child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                self_cpu = self_usage.ru_utime + self_usage.ru_stime
                children_cpu = child_usage.ru_utime + child_usage.ru_stime
                payload = {
                    "schema": "w1-native-launcher-cpu-accounting/1.0.0",
                    "status": result,
                    "stage": stage,
                    "error": error,
                    "pid": os.getpid(),
                    "process_module_entry_monotonic": PROCESS_MODULE_ENTRY_MONOTONIC,
                    "sampled_monotonic": time.monotonic(),
                    "native_launcher_process_cpu_seconds": self_cpu,
                    "native_launcher_reaped_children_cpu_seconds": children_cpu,
                    "native_launcher_complete_process_cpu_snapshot_seconds": self_cpu + children_cpu,
                    "scope": "RUSAGE_SELF plus reaped RUSAGE_CHILDREN, cumulative since native Python process start; this snapshot precedes only its own accounting-file write and process exit",
                    "supervisor_is_same_native_process": True,
                    "secret_material_saved": False,
                }
                accounting_path = ROOT / NATIVE_LAUNCHER_ACCOUNTING_RELATIVE_PATH
                if accounting_path.exists():
                    raise RuntimeError("Native launcher accounting output already exists; refusing overwrite")
                accounting_path.parent.mkdir(parents=True, exist_ok=True)
                durable_atomic_json(accounting_path, payload)
            except BaseException as accounting_error:
                # Preserve the primary failure; the outer RUSAGE_CHILDREN sample remains authoritative.
                print(
                    f"native launcher CPU accounting write failed: {type(accounting_error).__name__}: {accounting_error}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    raise SystemExit(main())
