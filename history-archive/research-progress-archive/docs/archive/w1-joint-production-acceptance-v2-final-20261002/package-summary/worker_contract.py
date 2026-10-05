"""The same worker identity check for production and controlled integration."""
from pathlib import Path
import json
from manifest_contract import verify_package


def verify_worker_contract(root, attempt, manifest_sha256, hashes_sha256, *, allow_integration=False):
    root = Path(root)
    identity = verify_package(root, expected_manifest_sha256=manifest_sha256,
                              expected_hashes_sha256=hashes_sha256)
    request = json.loads((root / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
    matrix = json.loads((root / "experiment-matrix.json").read_text(encoding="utf-8"))
    contract = json.loads((root / "launch-contract.json").read_text(encoding="utf-8"))
    if not attempt or {identity["manifest"]["attempt"], request["attempt"], matrix["attempt"], contract["attempt"]} != {attempt}:
        raise RuntimeError("WORKER_EXECUTION_IDENTITY_MISMATCH")
    if contract.get("integration_test") and not allow_integration:
        raise RuntimeError("FORMAL_WORKER_REJECTS_TEST_SUBSTITUTES")
    if request["status"] != "NOT_APPROVED":
        raise RuntimeError("FROZEN_REQUEST_CHANGED")
    return {"identity": identity, "request": request, "matrix": matrix, "contract": contract}
