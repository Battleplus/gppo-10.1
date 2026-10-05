"""Entry used only from the sealed package staged on native WSL storage."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from infra_io import require_native_linux_filesystem
from manifest_contract import PackageContractError, sha256_file, verify_external_authorization
from dependency_probe import probe as probe_runtime_dependencies

ROOT = Path(__file__).resolve().parent


def verify_native_launch(root: Path = ROOT, *, allow_integration: bool = False) -> str:
    require_native_linux_filesystem(root)
    token = os.environ.pop("W1_EXTERNAL_ATTEMPT_TOKEN", "")
    authorization_file = os.environ.pop("W1_EXTERNAL_AUTHORIZATION_FILE", "")
    if not authorization_file:
        raise RuntimeError("External authorization file path missing")
    try:
        verified = verify_external_authorization(
            root, Path(authorization_file), token=token, preflight_only=False,
        )
    except PackageContractError as exc:
        raise RuntimeError(f"AUTHORIZATION_OR_PACKAGE_CONTRACT_ERROR: {exc}") from exc
    manifest = verified["identity"]["manifest"]
    request = verified["request"]
    contract = verified["contract"]
    if contract.get("integration_test") and not allow_integration:
        raise RuntimeError("FORMAL_ENTRY_REJECTS_INTEGRATION_BACKEND")
    if request.get("status") != "NOT_APPROVED":
        raise RuntimeError("Frozen request status changed")
    if contract.get("resource_request_sha256") != sha256_file(root / "RESOURCE_REQUEST.json"):
        raise RuntimeError("Resource request identity mismatch")
    if not (request.get("attempt") == contract.get("attempt") == manifest.get("attempt")):
        raise RuntimeError("Attempt identity mismatch")
    dependency = probe_runtime_dependencies(expected_python=contract["native_python"])
    for name in ("run-once", "supervisor-status.json", "execution.lock"):
        if (root / name).exists():
            raise RuntimeError("Attempt evidence already exists; retry prohibited")
    (root / "runtime-dependency-probe.json").write_text(__import__("json").dumps(dependency, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract["attempt"]


def main() -> int:
    dependency_only = "--dependency-only" in sys.argv[1:]
    attempt = verify_native_launch()
    if dependency_only:
        print(json.dumps({"schema": "w1-native-dependency-only/1.0.0", "status": "pass", "worker_started": False, "environment_constructed": False, "model_initialized": False, "checkpoint_loaded": False, "training_started": False}, sort_keys=True))
        return 0
    from supervise import main as supervise_main

    # The supervisor passes this verified, non-secret identity to its worker.
    os.environ["W1_VERIFIED_ATTEMPT"] = attempt
    return supervise_main(ROOT, "runner.py")


if __name__ == "__main__":
    raise SystemExit(main())
