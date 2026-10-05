"""Freeze and verify the label-contract preparation package."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
HASHES_PATH = ROOT / "hashes.json"
MANIFEST_PATH = ROOT / "contract-manifest.json"
EXCLUDED = {"hashes.json", "contract-manifest.json"}
CONTRACT_ID = "w1-action-conditioned-task-outcome-label-repair-v2"
ATTEMPT = "w1-action-conditioned-task-outcome-label-qualification-v2-once"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _payload_files() -> dict[str, dict[str, Any]]:
    files = {}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(part == "__pycache__" for part in path.relative_to(ROOT).parts):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in EXCLUDED or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise RuntimeError(f"symlink_not_allowed:{relative}")
        payload = path.read_bytes()
        files[relative] = {"bytes": len(payload), "sha256": _sha256(payload)}
    return files


def freeze() -> dict[str, Any]:
    hashes = {"schema": "w1-label-contract-package-hashes/1.0.0", "files": _payload_files()}
    hashes_bytes = _json_bytes(hashes)
    HASHES_PATH.write_bytes(hashes_bytes)
    request_sha = _sha256((ROOT / "RESOURCE_REQUEST.json").read_bytes())
    manifest = {
        "schema": "w1-label-contract-package-manifest/1.0.0",
        "contract_id": CONTRACT_ID,
        "proposed_attempt": ATTEMPT,
        "status": "FROZEN_PREPARATION_NOT_APPROVED",
        "resource_request_status": "NOT_APPROVED",
        "resource_request_sha256": request_sha,
        "hashes_sha256": _sha256(hashes_bytes),
        "content_file_count": len(hashes["files"]),
        "dynamic_attempt_created": False,
    }
    MANIFEST_PATH.write_bytes(_json_bytes(manifest))
    return manifest


def verify() -> dict[str, Any]:
    hashes_bytes = HASHES_PATH.read_bytes()
    hashes = json.loads(hashes_bytes)
    manifest = json.loads(MANIFEST_PATH.read_bytes())
    if hashes.get("schema") != "w1-label-contract-package-hashes/1.0.0":
        raise RuntimeError("hash_schema_mismatch")
    if manifest.get("schema") != "w1-label-contract-package-manifest/1.0.0":
        raise RuntimeError("manifest_schema_mismatch")
    if manifest.get("hashes_sha256") != _sha256(hashes_bytes):
        raise RuntimeError("hashes_digest_mismatch")
    if manifest.get("resource_request_sha256") != _sha256((ROOT / "RESOURCE_REQUEST.json").read_bytes()):
        raise RuntimeError("resource_request_digest_mismatch")
    actual = _payload_files()
    if actual != hashes.get("files"):
        expected = hashes.get("files", {})
        changed = sorted(name for name in set(actual) | set(expected) if actual.get(name) != expected.get(name))
        raise RuntimeError("content_identity_mismatch:" + ",".join(changed))
    if manifest.get("content_file_count") != len(actual):
        raise RuntimeError("content_file_count_mismatch")
    return {
        "verified": True,
        "manifest_sha256": _sha256(MANIFEST_PATH.read_bytes()),
        "hashes_sha256": _sha256(hashes_bytes),
        "content_file_count": len(actual),
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
