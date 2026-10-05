"""Freeze the complete runner-ready package with deterministic SHA-256 IDs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXCLUDED = {"execution-manifest.json", "hashes.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def package_files() -> dict[str, str]:
    result = {}
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT).as_posix()
        if (
            not path.is_file()
            or path.name in EXCLUDED
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
        ):
            continue
        result[relative] = sha256(path)
    return result


def main() -> None:
    request = json.loads(
        (ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8")
    )
    contract = json.loads(
        (ROOT / "launch-contract.json").read_text(encoding="utf-8")
    )
    files = package_files()
    manifest = {
        "schema": "w1-oracle-remaining-23-manifest/1.0.0",
        "status": "RUNNER_READY_NOT_APPROVED",
        "attempt": request["attempt"],
        "resource_request_status": request["status"],
        "created_date": "2026-09-28",
        "dynamic_execution_authorized": False,
        "unique_entry": "launch_once.py",
        "native_entry": "native_launch.py",
        "worker_entry": "runner.py",
        "exact_external_token_sha256": contract["external_token_sha256"],
        "expected_stack": {
            "platform": "Linux CPU",
            "python": "3.11.16",
            "numpy": "1.26.0",
            "active_filesystem": "WSL native ext4",
        },
        "preparation_consumption": {
            "environment_steps": 0,
            "model_initializations_or_loads": 0,
            "model_forwards": 0,
            "training_updates": 0,
        },
        "files": files,
    }
    atomic_json(ROOT / "execution-manifest.json", manifest)
    hashes = {
        "schema": "w1-oracle-remaining-23-hashes/1.0.0",
        "execution_manifest_sha256": sha256(ROOT / "execution-manifest.json"),
        "resource_request_sha256": sha256(ROOT / "RESOURCE_REQUEST.json"),
        "launch_contract_sha256": sha256(ROOT / "launch-contract.json"),
        "file_count": len(files),
        "files": files,
    }
    atomic_json(ROOT / "hashes.json", hashes)


if __name__ == "__main__":
    main()
