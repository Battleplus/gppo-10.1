"""Freeze the preparation identity without executing the experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from manifest_contract import sha256_file, write_identity_files


ROOT = Path(__file__).resolve().parent
PACKAGE_NAME = "w1-eawm-jepa-world-model-production-repair-v4-e2e-integration-auth-repair-v1"
ATTEMPT = f"{PACKAGE_NAME}-once"
SOURCE_WINDOWS = rf"E:\Z博士\research-plans\{PACKAGE_NAME}"
SOURCE_WSL = f"/mnt/e/Z博士/research-plans/{PACKAGE_NAME}"
NATIVE_ROOT = f"/home/asus/{ATTEMPT}"
EXPORT_ROOT = f"/mnt/e/Z博士/.codex-exports/{ATTEMPT}"
NATIVE_PYTHON = "/home/asus/w1-action-relative-auxiliary-runtime-dependency-repair-v1-runtime/bin/python"
DEFAULT_AUTHORIZATION = rf"E:\Z博士\.codex-private\{PACKAGE_NAME}\authorization.json"


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def require_ready_sources() -> None:
    required = {
        "runtime_backend.py",
        "runner.py",
        "world_model_pipeline.py",
        "launch_once.py",
        "wsl_stage_and_launch.py",
        "native_launch.py",
        "supervise.py",
        "worker_bootstrap.py",
        "budget_ledger.py",
        "infra_io.py",
        "dependency_probe.py",
        "RESOURCE_REQUEST.json",
        "experiment-matrix.json",
        "parent-split.json",
        "production-wiring.md",
    }
    missing = sorted(name for name in required if not (ROOT / name).is_file())
    if missing:
        raise RuntimeError("PREPARATION_PACKAGE_FILES_MISSING:" + ",".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-output", type=Path, default=Path(DEFAULT_AUTHORIZATION))
    args = parser.parse_args()
    require_ready_sources()
    authorization_path = args.authorization_output.resolve()
    if authorization_path.is_relative_to(ROOT.resolve()):
        raise RuntimeError("AUTHORIZATION_MUST_BE_OUTSIDE_PACKAGE")
    request_path = ROOT / "RESOURCE_REQUEST.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("status") != "NOT_APPROVED" or not isinstance(request.get("attempt"), str):
        raise RuntimeError("RESOURCE_REQUEST_STATUS_OR_ATTEMPT_INVALID")
    request["attempt"] = ATTEMPT
    write_json(request_path, request)
    contract = {
        "schema": "w1-eawm-jepa-launch-contract/2.1.0",
        "attempt": ATTEMPT,
        "authorization_status": "NOT_APPROVED",
        "one_shot": True,
        "integration_test": False,
        "integration_fault": None,
        "wsl_executable": "wsl.exe",
        "wsl_distribution": "Ubuntu-24.04",
        "windows_source_root": SOURCE_WINDOWS,
        "wsl_source_root": SOURCE_WSL,
        "native_execution_root": NATIVE_ROOT,
        "windows_export_root": EXPORT_ROOT,
        "native_python": NATIVE_PYTHON,
        "native_child_python": NATIVE_PYTHON,
        "linux_stage_entry": "wsl_stage_and_launch.py",
        "native_entry": "native_launch.py",
        "argument_probe": "中文 path with spaces 'quoted'",
        "resource_request_sha256": sha256_file(request_path),
    }
    write_json(ROOT / "launch-contract.json", contract)
    authorization_path.parent.mkdir(parents=True, exist_ok=True)
    authorization = {
        "schema": "w1-external-launch-authorization/2.0.0",
        "attempt": ATTEMPT,
        "execution_manifest_sha256": "pending-freeze",
        "hashes_sha256": "pending-freeze",
        "resource_request_sha256": sha256_file(request_path),
        "token_sha256": None,
        "status": "NOT_APPROVED",
    }
    command = (
        "# Unique launch commands\n\n"
        "Read-only structure and identity preflight. This accepts the frozen `NOT_APPROVED` status, "
        "does not grant approval, reads no token, and does not create an attempt:\n\n"
        "```powershell\n"
        f"python \"{SOURCE_WINDOWS}\\launch_once.py\" --authorization-file \"{authorization_path}\" --preflight-only\n"
        "```\n\n"
        "Formal start template only. Obtain a new external authorization and one-time token bound to this final identity; "
        "the resource request remains `NOT_APPROVED`:\n\n"
        "```powershell\n"
        f"python \"{SOURCE_WINDOWS}\\launch_once.py\" --attempt-token-file \"<new-external-token-file>\" --authorization-file \"<new-approved-authorization-file>\"\n"
        "```\n"
    )
    (ROOT / "unique-launch-command.md").write_text(command, encoding="utf-8")
    identity = write_identity_files(ROOT, attempt=ATTEMPT, status="NOT_APPROVED")
    authorization.update(
        execution_manifest_sha256=identity["execution_manifest_sha256"],
        hashes_sha256=identity["hashes_sha256"],
    )
    write_json(authorization_path, authorization)
    serialized = authorization_path.read_text(encoding="utf-8")
    round_trip = json.loads(serialized)
    if not isinstance(round_trip, dict) or serialized[serialized.find("}") + 1 :].strip():
        raise RuntimeError("AUTHORIZATION_JSON_ROUND_TRIP_INVALID")
    if round_trip != authorization:
        raise RuntimeError("AUTHORIZATION_JSON_ROUND_TRIP_MISMATCH")
    from manifest_contract import verify_external_authorization

    verify_external_authorization(ROOT, authorization_path, token=None, preflight_only=True)
    print(json.dumps({**identity, "attempt": ATTEMPT, "authorization_status": "NOT_APPROVED",
                      "resource_request_sha256": sha256_file(request_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
