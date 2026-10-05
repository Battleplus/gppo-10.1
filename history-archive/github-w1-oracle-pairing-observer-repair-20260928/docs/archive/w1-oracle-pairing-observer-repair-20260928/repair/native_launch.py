"""Entry used only from the sealed package staged on native WSL storage."""
from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path

from infra_io import read_json, require_native_linux_filesystem, sha256_file, verify_manifest

ROOT = Path(__file__).resolve().parent


def verify_native_launch(root: Path = ROOT) -> str:
    require_native_linux_filesystem(root)
    hashes = read_json(root / "hashes.json")
    if sha256_file(root / "execution-manifest.json") != hashes["execution_manifest_sha256"]:
        raise RuntimeError("Staged execution manifest SHA-256 mismatch")
    manifest = read_json(root / "execution-manifest.json")
    verify_manifest(root, manifest["files"])
    request = read_json(root / "RESOURCE_REQUEST.json")
    contract = read_json(root / "launch-contract.json")
    if request.get("status") != "NOT_APPROVED":
        raise RuntimeError("Frozen request status changed")
    if not (request.get("attempt") == contract.get("attempt") == manifest.get("attempt")):
        raise RuntimeError("Attempt identity mismatch")
    token = os.environ.pop("W1_EXTERNAL_ATTEMPT_TOKEN", "")
    supplied = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if not token or not hmac.compare_digest(supplied, contract["external_token_sha256"]):
        raise RuntimeError("Native launch authorization missing or invalid")
    for name in ("run-once", "supervisor-status.json", "execution.lock"):
        if (root / name).exists():
            raise RuntimeError("Attempt evidence already exists; retry prohibited")
    return contract["attempt"]


def main() -> int:
    verify_native_launch()
    from supervise import main as supervise_main

    return supervise_main(ROOT, "runner.py")


if __name__ == "__main__":
    raise SystemExit(main())
