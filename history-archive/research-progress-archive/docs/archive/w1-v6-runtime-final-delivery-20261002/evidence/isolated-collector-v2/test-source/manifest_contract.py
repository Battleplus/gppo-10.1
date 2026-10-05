"""Shared, acyclic package identity contract for every launch layer."""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import re
from typing import Any


CONTRACT_SCHEMA = "w1-package-identity-contract/2.0.0"
HASHES_SCHEMA = "w1-eawm-jepa-world-model-hashes/2.0.0"
MANIFEST_SCHEMA = "w1-eawm-jepa-world-model-manifest/2.0.0"
AUTHORIZATION_SCHEMA = "w1-external-launch-authorization/2.0.0"
LAUNCH_CONTRACT_SCHEMA = "w1-eawm-jepa-launch-contract/2.1.0"
AUTHORIZATION_FIELDS = frozenset({
    "schema",
    "attempt",
    "execution_manifest_sha256",
    "hashes_sha256",
    "resource_request_sha256",
    "token_sha256",
    "status",
})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


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
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except FileNotFoundError as exc:
        raise PackageContractError(f"{label}_MISSING") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageContractError(f"{label}_INVALID_JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PackageContractError(f"{label}_NOT_OBJECT")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PackageContractError(f"JSON_DUPLICATE_FIELD: {key}")
        value[key] = item
    return value


def _validate_authorization_fields(authorization: dict[str, Any]) -> None:
    missing = sorted(AUTHORIZATION_FIELDS - authorization.keys())
    if missing:
        raise PackageContractError("AUTHORIZATION_FIELDS_MISSING: " + ",".join(missing))
    unexpected = sorted(authorization.keys() - AUTHORIZATION_FIELDS)
    if unexpected:
        raise PackageContractError("AUTHORIZATION_FIELDS_UNEXPECTED: " + ",".join(unexpected))

    string_fields = (
        "schema",
        "attempt",
        "execution_manifest_sha256",
        "hashes_sha256",
        "resource_request_sha256",
        "status",
    )
    for field in string_fields:
        if not isinstance(authorization[field], str) or not authorization[field]:
            raise PackageContractError(f"AUTHORIZATION_FIELD_TYPE_INVALID: {field}")
    if authorization["schema"] != AUTHORIZATION_SCHEMA:
        raise PackageContractError("AUTHORIZATION_SCHEMA_INVALID")
    if authorization["status"] not in {"NOT_APPROVED", "APPROVED"}:
        raise PackageContractError("AUTHORIZATION_STATUS_INVALID")
    if not authorization["attempt"].strip():
        raise PackageContractError("AUTHORIZATION_ATTEMPT_INVALID")
    for field in ("execution_manifest_sha256", "hashes_sha256", "resource_request_sha256"):
        if _SHA256_PATTERN.fullmatch(authorization[field]) is None:
            raise PackageContractError(f"AUTHORIZATION_DIGEST_INVALID: {field}")

    token_digest = authorization["token_sha256"]
    if token_digest is not None and (
        not isinstance(token_digest, str) or _SHA256_PATTERN.fullmatch(token_digest) is None
    ):
        raise PackageContractError("AUTHORIZED_TOKEN_DIGEST_INVALID")
    if authorization["status"] == "NOT_APPROVED" and token_digest is not None:
        raise PackageContractError("UNAPPROVED_AUTHORIZATION_HAS_TOKEN_DIGEST")
    if authorization["status"] == "APPROVED" and token_digest is None:
        raise PackageContractError("AUTHORIZED_TOKEN_DIGEST_MISSING")


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


def verify_external_authorization(root: Path, authorization_file: Path, *,
                                  token: str | None, preflight_only: bool) -> dict[str, Any]:
    """Verify the out-of-package approval and one-time token at each launch layer."""
    root = Path(root).resolve()
    authorization_path = Path(authorization_file).resolve()
    if authorization_path.is_relative_to(root):
        raise PackageContractError("AUTHORIZATION_FILE_MUST_BE_OUTSIDE_PACKAGE")
    authorization = read_object(authorization_path, "AUTHORIZATION")
    _validate_authorization_fields(authorization)
    identity = verify_package(
        root,
        expected_manifest_sha256=authorization.get("execution_manifest_sha256"),
        expected_hashes_sha256=authorization.get("hashes_sha256"),
    )
    request = read_object(root / "RESOURCE_REQUEST.json", "RESOURCE_REQUEST")
    contract = read_object(root / "launch-contract.json", "LAUNCH_CONTRACT")
    if contract.get("schema") != LAUNCH_CONTRACT_SCHEMA:
        raise PackageContractError("LAUNCH_CONTRACT_SCHEMA_INVALID")
    if request.get("status") != "NOT_APPROVED":
        raise PackageContractError("RESOURCE_REQUEST_STATUS_CHANGED")
    attempt = request.get("attempt")
    manifest_attempt = identity["manifest"].get("attempt")
    contract_attempt = contract.get("attempt")
    if not all(isinstance(item, str) and item for item in (attempt, manifest_attempt, contract_attempt)):
        raise PackageContractError("ATTEMPT_IDENTITY_TYPE_INVALID")
    if not (attempt == manifest_attempt == contract_attempt == authorization["attempt"]):
        raise PackageContractError("ATTEMPT_IDENTITY_MISMATCH")
    request_digest = sha256_file(root / "RESOURCE_REQUEST.json")
    if contract.get("resource_request_sha256") != request_digest or authorization.get("resource_request_sha256") != request_digest:
        raise PackageContractError("RESOURCE_REQUEST_DIGEST_MISMATCH")
    status = authorization.get("status")
    if preflight_only:
        if status not in {"NOT_APPROVED", "APPROVED"}:
            raise PackageContractError("AUTHORIZATION_STATUS_INVALID")
    else:
        token_digest = authorization.get("token_sha256")
        if status != "APPROVED" or not isinstance(token, str) or not token:
            raise PackageContractError("APPROVED_AUTHORIZATION_AND_TOKEN_REQUIRED")
        if not isinstance(token_digest, str) or len(token_digest) != 64:
            raise PackageContractError("AUTHORIZED_TOKEN_DIGEST_INVALID")
        supplied = hashlib.sha256(token.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(supplied, token_digest):
            raise PackageContractError("EXTERNAL_TOKEN_MISMATCH")
    return {"authorization": authorization, "contract": contract, "identity": identity, "request": request}


def build_content_files(root: Path, *, excluded: set[str] | None = None) -> dict[str, str]:
    excluded = excluded if excluded is not None else {"hashes.json", "execution-manifest.json"}
    allowed_subtrees = {"native"}
    ignored_directories = {"__pycache__", ".git"}
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        parts = Path(relative).parts
        if any(part in ignored_directories for part in parts) or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise PackageContractError(f"PACKAGE_SYMLINK_NOT_ALLOWED: {relative}")
        if path.is_dir():
            if parts[0] not in allowed_subtrees:
                raise PackageContractError(f"PACKAGE_SUBTREE_NOT_ALLOWLISTED: {relative}")
            continue
        if not path.is_file():
            raise PackageContractError(f"PACKAGE_NON_REGULAR_FILE_NOT_ALLOWED: {relative}")
        if relative in excluded or (len(parts) == 1 and path.name in excluded):
            continue
        if len(parts) > 1 and parts[0] not in allowed_subtrees:
            raise PackageContractError(f"PACKAGE_FILE_OUTSIDE_ALLOWLISTED_SUBTREES: {relative}")
        result[relative] = sha256_file(path)
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
