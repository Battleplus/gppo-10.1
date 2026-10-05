"""Shared, acyclic package identity contract for every launch layer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT_SCHEMA = "w1-package-identity-contract/2.0.0"
HASHES_SCHEMA = "w1-eawm-jepa-world-model-hashes/2.0.0"
MANIFEST_SCHEMA = "w1-eawm-jepa-world-model-manifest/2.0.0"


class PackageContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PackageContractError(f"{label}_MISSING") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageContractError(f"{label}_INVALID_JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PackageContractError(f"{label}_NOT_OBJECT")
    return value


def _files(value: dict[str, Any], label: str) -> dict[str, str]:
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise PackageContractError(f"{label}_FILES_INVALID")
    output: dict[str, str] = {}
    for relative, digest in files.items():
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise PackageContractError(f"{label}_PATH_INVALID: {relative!r}")
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
            raise PackageContractError(f"{label}_DIGEST_INVALID: {relative}")
        output[relative] = digest
    return output


def verify_package(root: Path, *, expected_manifest_sha256: str | None = None, expected_hashes_sha256: str | None = None) -> dict[str, Any]:
    """Verify the package using the same one-way contract everywhere.

    `hashes.json` -> execution manifest digest -> manifest/files. The manifest
    never hashes `hashes.json`, so no circular dependency exists.
    """
    root = root.resolve()
    hashes_path = root / "hashes.json"
    manifest_path = root / "execution-manifest.json"
    hashes = read_object(hashes_path, "HASHES")
    if hashes.get("schema") != HASHES_SCHEMA:
        raise PackageContractError("HASHES_SCHEMA_INVALID")
    actual_hashes = sha256_file(hashes_path)
    if expected_hashes_sha256 and actual_hashes != expected_hashes_sha256:
        raise PackageContractError("HASHES_OUTER_DIGEST_MISMATCH")
    manifest_digest = hashes.get("execution_manifest_sha256")
    if not isinstance(manifest_digest, str) or len(manifest_digest) != 64:
        raise PackageContractError("HASHES_EXECUTION_MANIFEST_DIGEST_MISSING")
    actual_manifest = sha256_file(manifest_path)
    if actual_manifest != manifest_digest or (expected_manifest_sha256 and actual_manifest != expected_manifest_sha256):
        raise PackageContractError("EXECUTION_MANIFEST_DIGEST_MISMATCH")
    manifest = read_object(manifest_path, "EXECUTION_MANIFEST")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise PackageContractError("EXECUTION_MANIFEST_SCHEMA_INVALID")
    manifest_files = _files(manifest, "EXECUTION_MANIFEST")
    hash_files = _files(hashes, "HASHES")
    if manifest_files != hash_files:
        raise PackageContractError("MANIFEST_HASH_FILE_SET_MISMATCH")
    for relative, expected in manifest_files.items():
        candidate = root / relative
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise PackageContractError(f"PACKAGE_FILE_DIGEST_MISMATCH: {relative}")
    return {"hashes": hashes, "manifest": manifest, "manifest_sha256": actual_manifest, "hashes_sha256": actual_hashes}


def build_content_files(root: Path, *, excluded: set[str] | None = None) -> dict[str, str]:
    excluded = excluded or {"hashes.json", "execution-manifest.json"}
    result: dict[str, str] = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name not in excluded and not path.name.endswith(".pyc"):
            result[path.name] = sha256_file(path)
    if not result:
        raise PackageContractError("PACKAGE_HAS_NO_CONTENT_FILES")
    return result


def write_identity_files(root: Path, *, attempt: str, status: str = "NOT_APPROVED") -> dict[str, str]:
    """Write the shared schema; hashes is written last and hashes the manifest."""
    files = build_content_files(root)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "contract_schema": CONTRACT_SCHEMA,
        "attempt": attempt,
        "status": status,
        "files": files,
        "frozen_entrypoint": "launch_once.py",
    }
    (root / "execution-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = {
        "schema": HASHES_SCHEMA,
        "contract_schema": CONTRACT_SCHEMA,
        "execution_manifest_sha256": sha256_file(root / "execution-manifest.json"),
        "files": files,
    }
    (root / "hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_package(root)
    return {"execution_manifest_sha256": sha256_file(root / "execution-manifest.json"), "hashes_sha256": sha256_file(root / "hashes.json")}
