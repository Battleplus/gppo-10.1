"""Native entry reserved for a hash-bound synthetic integration fixture."""
from __future__ import annotations

import json
import os
from pathlib import Path

from dependency_probe import probe as probe_runtime_dependencies
from infra_io import require_native_linux_filesystem
from manifest_contract import verify_external_authorization
from supervise import main as supervise_main

ROOT = Path(__file__).resolve().parent


def main() -> int:
    require_native_linux_filesystem(ROOT)
    token = os.environ.pop("W1_EXTERNAL_ATTEMPT_TOKEN", "")
    authorization_file = os.environ.pop("W1_EXTERNAL_AUTHORIZATION_FILE", "")
    if os.environ.pop("W1_TEST_ONLY_INTEGRATION", "") != "1":
        raise RuntimeError("TEST_ONLY_ENTRY_REQUIRES_INTEGRATION_MARKER")
    verified = verify_external_authorization(
        ROOT, Path(authorization_file), token=token, preflight_only=False,
    )
    contract = verified["contract"]
    request = verified["request"]
    attempt = verified["identity"]["manifest"]["attempt"]
    if (contract.get("integration_test") is not True
            or contract.get("native_entry") != "integration_native_launch.py"
            or not attempt.endswith("-integration-test")
            or request.get("status") != "NOT_APPROVED"):
        raise RuntimeError("TEST_ONLY_IDENTITY_CONTRACT_INVALID")
    probe_runtime_dependencies(expected_python=contract["native_python"])
    for name in ("run-once", "supervisor-status.json", "execution.lock"):
        if (ROOT / name).exists():
            raise RuntimeError("TEST_ATTEMPT_ALREADY_EXISTS_NO_RETRY")
    return supervise_main(ROOT, "integration_worker.py")


if __name__ == "__main__":
    raise SystemExit(main())
