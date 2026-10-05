"""One shared generator for the deliverable and independently identified tests."""
import argparse
import json
from pathlib import Path
from manifest_contract import write_identity_files, verify_package, sha256_file
from joint_inputs import verify_joint_inputs


def freeze(root: Path):
    request = json.loads((root / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
    contract = json.loads((root / "launch-contract.json").read_text(encoding="utf-8"))
    matrix = json.loads((root / "experiment-matrix.json").read_text(encoding="utf-8"))
    split = json.loads((root / "parent-split.json").read_text(encoding="utf-8"))
    if request["status"] != "NOT_APPROVED" or contract["authorization_status"] != "NOT_APPROVED":
        raise RuntimeError("FREEZE_CANNOT_APPROVE_A_RESEARCH_REQUEST")
    if len({request["attempt"], contract["attempt"], matrix["attempt"], split["attempt"]}) != 1:
        raise RuntimeError("FREEZE_ATTEMPT_CROSS_BINDING_MISMATCH")
    for resource, reserve in (("wall_seconds", "cross_system_wall_seconds"),
                              ("complete_process_cpu_seconds", "windows_post_preflight_process_cpu_seconds")):
        if sum(stage[resource] for stage in request["stages"].values()) + request["accounting_reserves"][reserve] != request["totals"][resource]:
            raise RuntimeError("BUDGET_NOT_CLOSED:" + resource)
    if matrix["prediction_gates"] != json.loads((root / "prediction-gates.json").read_text(encoding="utf-8")):
        raise RuntimeError("PREDICTION_GATE_CONFIGURATION_DIFFERENCE")
    if not contract.get("integration_test"):
        inputs = verify_joint_inputs(root)
        if inputs["parent_roles"] != {"train": 24, "model_selection": 8, "prediction_confirmation": 8}:
            raise RuntimeError("PRODUCTION_PARENT_SPLIT_CHANGED")
    contract["resource_request_sha256"] = sha256_file(root / "RESOURCE_REQUEST.json")
    (root / "launch-contract.json").write_text(json.dumps(contract, ensure_ascii=False, sort_keys=True,
                                                          indent=2)+"\n", encoding="utf-8")
    return write_identity_files(root, attempt=request["attempt"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = verify_package(args.root) if args.verify else freeze(args.root)
    if args.verify:
        result = {"status": "pass", "manifest_sha256": result["manifest_sha256"],
                  "hashes_sha256": result["hashes_sha256"], "content_files": len(result["manifest"]["files"])}
    print(json.dumps(result, ensure_ascii=False))
