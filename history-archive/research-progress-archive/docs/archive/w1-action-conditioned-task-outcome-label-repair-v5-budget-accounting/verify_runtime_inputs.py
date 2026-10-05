"""Check the frozen source tape and production source modules before any reset."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


class RuntimeInputError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_runtime_inputs(package_root: Path) -> dict:
    inputs = json.loads((package_root / "runtime-inputs.json").read_text(encoding="utf-8"))
    source = inputs["source_run"]
    root_value = source["windows_root"] if os.name == "nt" else source["wsl_root"]
    source_root = Path(root_value)
    if not source_root.is_absolute() or not source_root.is_dir():
        raise RuntimeInputError("SOURCE_ROOT_MISSING_OR_NOT_ABSOLUTE")
    manifest_path = source_root / "execution-manifest.json"
    if sha256_file(manifest_path) != source["execution_manifest_sha256"]:
        raise RuntimeInputError("SOURCE_EXECUTION_MANIFEST_DIGEST_MISMATCH")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict):
        raise RuntimeInputError("SOURCE_EXECUTION_MANIFEST_SCHEMA_INVALID")
    checked = {}
    required = dict(inputs.get("production_modules", {}))
    required[source["train_tape_file"]] = source["train_tape_sha256"]
    for relative, expected in required.items():
        actual = manifest_files.get(relative)
        if actual != expected:
            raise RuntimeInputError("SOURCE_MANIFEST_FILE_IDENTITY_MISMATCH:" + relative)
        path = source_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeInputError("SOURCE_FILE_DIGEST_MISMATCH:" + relative)
        checked[relative] = expected
    if sys.platform != "win32" and not str(source_root).startswith("/mnt/"):
        raise RuntimeInputError("SOURCE_ROOT_NOT_EXPLICIT_WSL_MOUNT")
    return {
        "schema": "w1-action-conditioned-task-outcome-runtime-input-check/1.0.0",
        "status": "pass",
        "source_root": str(source_root),
        "source_execution_manifest_sha256": source["execution_manifest_sha256"],
        "train_tape_sha256": source["train_tape_sha256"],
        "verified_files": checked,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    print(json.dumps(verify_runtime_inputs(root), ensure_ascii=False, sort_keys=True))
