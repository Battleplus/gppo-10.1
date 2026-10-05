"""Validate portable source evidence without borrowing the original workspace."""
from __future__ import annotations

import json
from pathlib import Path
from manifest_contract import PackageContractError, sha256_file
from verify_runtime_inputs import verify_runtime_inputs


def verify_joint_inputs(root: Path) -> dict:
    root = Path(root).resolve()
    verify_runtime_inputs(root)
    spec = json.loads((root / "runtime-inputs.json").read_text(encoding="utf-8"))
    source = spec["source_run"]
    evidence = root / "native" / "source-evidence"
    manifest_path = evidence / "execution-manifest.json"
    if sha256_file(manifest_path) != source["execution_manifest_sha256"]:
        raise PackageContractError("PORTABLE_SOURCE_MANIFEST_MISMATCH")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tape = evidence / source["train_tape_file"]
    if (sha256_file(tape) != source["train_tape_sha256"]
            or manifest["files"].get(source["train_tape_file"]) != source["train_tape_sha256"]):
        raise PackageContractError("PORTABLE_SOURCE_TAPE_MISMATCH")
    for relative, digest in spec["source_modules"].items():
        if manifest["files"].get(relative) != digest or sha256_file(root / relative) != digest:
            raise PackageContractError("PORTABLE_SOURCE_MODULE_MISMATCH:" + relative)
    split = json.loads((root / "parent-split.json").read_text(encoding="utf-8"))
    tapes = json.loads(tape.read_text(encoding="utf-8"))
    indexed = {row["exogenous_key"]: row for row in tapes}
    found = set()
    roles = {"train": 0, "model_selection": 0, "prediction_confirmation": 0}
    for identity in split["parents"]:
        row = indexed.get(identity["exogenous_key"])
        if row is None or row["scenario_sha256"] != identity["scenario_sha256"]:
            raise PackageContractError("PORTABLE_PARENT_IDENTITY_MISMATCH:" + identity["parent"])
        if identity["scenario_sha256"] in found:
            raise PackageContractError("PARENT_CONTENT_OVERLAP")
        found.add(identity["scenario_sha256"])
        roles[identity["proposed_split"]] += 1
    return {"status": "pass", "source_tape_sha256": sha256_file(tape),
            "portable_source": str(evidence), "parent_roles": roles,
            "workspace_imports_required": False, "environment_calls": 0}
