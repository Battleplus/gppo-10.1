"""Freeze and verify the label-contract preparation package."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from pathlib import PurePosixPath

from manifest_contract import (
    PackageContractError,
    sha256_file,
    verify_package,
    write_identity_files,
)
from verify_runtime_inputs import verify_runtime_inputs


ROOT = Path(__file__).resolve().parent
ATTEMPT = "w1-action-conditioned-task-outcome-label-qualification-v5-budget-accounting-once"
NATIVE_PYTHON = "/home/asus/w1-action-relative-auxiliary-runtime-dependency-repair-v1-runtime/bin/python"


def mounted_wsl_path_to_windows(path: str) -> tuple[str, str]:
    """Translate only an explicit /mnt/<drive> package path to its Windows path."""
    value = PurePosixPath(path)
    parts = value.parts
    if (not value.is_absolute() or len(parts) < 4 or parts[1] != "mnt"
            or len(parts[2]) != 1 or not parts[2].isalpha()):
        raise RuntimeError("PACKAGE_SOURCE_NOT_ON_EXPLICIT_WINDOWS_MOUNT")
    drive = parts[2].lower()
    relative = "\\".join(parts[3:])
    return drive, f"{drive.upper()}:\\{relative}"


def _launch_paths(root: Path) -> tuple[str, str, str]:
    if os.name == "nt":
        root = root.resolve()
        if not root.drive:
            raise RuntimeError("PACKAGE_SOURCE_WINDOWS_DRIVE_MISSING")
        drive = root.drive[0].lower()
        relative_source = root.relative_to(Path(root.anchor)).as_posix()
        wsl_source = f"/mnt/{drive}/{relative_source}"
        workspace = root.parents[1]
        relative_workspace = workspace.relative_to(Path(root.anchor)).as_posix()
        wsl_workspace = f"/mnt/{drive}/{relative_workspace}"
        windows_source = str(root)
    elif os.name == "posix":
        wsl_source = root.resolve().as_posix()
        drive, windows_source = mounted_wsl_path_to_windows(wsl_source)
        wsl_workspace = root.parents[1].resolve().as_posix()
        workspace_drive, windows_workspace = mounted_wsl_path_to_windows(wsl_workspace)
        if workspace_drive != drive:
            raise RuntimeError("PACKAGE_AND_WORKSPACE_DRIVES_DIFFER")
    else:
        raise RuntimeError("UNSUPPORTED_FREEZE_HOST")
    return windows_source, wsl_source, f"{wsl_workspace}/runs/{ATTEMPT}"


def freeze() -> dict[str, Any]:
    request = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
    matrix = json.loads((ROOT / "experiment-matrix.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "parent-split.json").read_text(encoding="utf-8"))
    if request.get("status") != "NOT_APPROVED":
        raise RuntimeError("RESOURCE_REQUEST_MUST_REMAIN_NOT_APPROVED")
    if not (request.get("attempt") == matrix.get("attempt") == split.get("attempt") == ATTEMPT):
        raise RuntimeError("ATTEMPT_IDENTITY_MISMATCH_BEFORE_FREEZE")
    if [row.get("parent") for row in split.get("parents", [])] != [f"train-{i:04d}" for i in range(56, 64)]:
        raise RuntimeError("FROZEN_PARENT_IDENTITY_MISMATCH")
    runtime_identity = verify_runtime_inputs(ROOT)
    if runtime_identity.get("status") != "pass":
        raise RuntimeError("SOURCE_RUNTIME_IDENTITY_CHECK_FAILED")
    if matrix.get("source_semantics") != {
        "task_completion_mode": "arrival_to_region",
        "deadline_basis": "physical_arrival",
        "arrival_radius": 0.0,
    }:
        raise RuntimeError("TASK_SEMANTICS_IDENTITY_MISMATCH")
    reserves = request.get("accounting_reserves", {})
    if (reserves.get("windows_post_preflight_process_cpu_seconds") != 1
            or reserves.get("cross_system_wall_seconds") != 2):
        raise RuntimeError("CROSS_SYSTEM_ACCOUNTING_RESERVE_MISMATCH")
    stage_wall = sum(row["wall_seconds"] for row in request["stages"].values())
    stage_cpu = sum(row["complete_process_cpu_seconds"] for row in request["stages"].values())
    if stage_wall + reserves["cross_system_wall_seconds"] != request["totals"]["wall_seconds"]:
        raise RuntimeError("WALL_BUDGET_DOES_NOT_CLOSE")
    if stage_cpu + reserves["windows_post_preflight_process_cpu_seconds"] != request["totals"]["complete_process_cpu_seconds"]:
        raise RuntimeError("CPU_BUDGET_DOES_NOT_CLOSE")
    request_sha = sha256_file(ROOT / "RESOURCE_REQUEST.json")
    windows_source, wsl_source, wsl_export = _launch_paths(ROOT)
    contract = {
        "schema": "w1-eawm-jepa-launch-contract/2.1.0",
        "attempt": ATTEMPT,
        "authorization_status": "NOT_APPROVED",
        "one_shot": True,
        "automatic_retry": False,
        "integration_test": False,
        "integration_fault": None,
        "wsl_executable": "wsl.exe",
        "wsl_distribution": "Ubuntu-24.04",
        "native_python": NATIVE_PYTHON,
        "native_child_python": NATIVE_PYTHON,
        "native_entry": "native_launch.py",
        "linux_stage_entry": "wsl_stage_and_launch.py",
        "windows_source_root": windows_source,
        "wsl_source_root": wsl_source,
        "native_execution_root": f"/home/asus/{ATTEMPT}",
        "windows_export_root": wsl_export,
        "resource_request_sha256": request_sha,
        "accounting_reserves": reserves,
        "argument_probe": "W1 标签采集 path with spaces 'single' and \"double\" quotes",
        "purpose": "One finite, label-only task-outcome qualification; no model training.",
    }
    (ROOT / "launch-contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    identity = write_identity_files(ROOT, attempt=ATTEMPT, status="NOT_APPROVED")
    return {"frozen": True, "attempt": ATTEMPT, **identity, "dynamic_attempt_created": False}


def verify() -> dict[str, Any]:
    identity = verify_package(ROOT)
    request = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
    contract = json.loads((ROOT / "launch-contract.json").read_text(encoding="utf-8"))
    attempt = identity["manifest"].get("attempt")
    if request.get("status") != "NOT_APPROVED" or contract.get("authorization_status") != "NOT_APPROVED":
        raise PackageContractError("PREPARATION_APPROVAL_STATUS_CHANGED")
    if not (attempt == request.get("attempt") == contract.get("attempt") == ATTEMPT):
        raise PackageContractError("FROZEN_ATTEMPT_IDENTITY_MISMATCH")
    if contract.get("resource_request_sha256") != sha256_file(ROOT / "RESOURCE_REQUEST.json"):
        raise PackageContractError("FROZEN_RESOURCE_REQUEST_DIGEST_MISMATCH")
    if contract.get("accounting_reserves") != request.get("accounting_reserves"):
        raise PackageContractError("FROZEN_ACCOUNTING_RESERVE_MISMATCH")
    reserves = request["accounting_reserves"]
    stage_wall = sum(row["wall_seconds"] for row in request["stages"].values())
    stage_cpu = sum(row["complete_process_cpu_seconds"] for row in request["stages"].values())
    if stage_wall + reserves["cross_system_wall_seconds"] != request["totals"]["wall_seconds"]:
        raise PackageContractError("FROZEN_WALL_BUDGET_DOES_NOT_CLOSE")
    if stage_cpu + reserves["windows_post_preflight_process_cpu_seconds"] != request["totals"]["complete_process_cpu_seconds"]:
        raise PackageContractError("FROZEN_CPU_BUDGET_DOES_NOT_CLOSE")
    return {
        "verified": True,
        "attempt": attempt,
        "execution_manifest_sha256": identity["manifest_sha256"],
        "hashes_sha256": identity["hashes_sha256"],
        "content_file_count": len(identity["manifest"]["files"]),
        "resource_request_status": request["status"],
        "dynamic_attempt_created": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = verify() if args.verify else freeze()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
