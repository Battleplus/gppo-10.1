"""Retarget only execution identities; do not regenerate the research proposal."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD_ATTEMPT = "w1-task-outcome-g1-g2-joint-v1-once"
NEW_ATTEMPT = "w1-task-outcome-g1-g2-joint-v2-once"


def main():
    for name in ("RESOURCE_REQUEST.json", "launch-contract.json", "experiment-matrix.json", "parent-split.json"):
        path = ROOT / name
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["attempt"] not in (OLD_ATTEMPT, NEW_ATTEMPT):
            raise RuntimeError("UNEXPECTED_SOURCE_ATTEMPT:" + name)
        value["attempt"] = NEW_ATTEMPT
        if name == "launch-contract.json":
            value["native_execution_root"] = "/home/user1/" + NEW_ATTEMPT
            value["native_entry"] = "metered_joint_entry.py"
        path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    path = ROOT / "runtime-inputs.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["source_run"]["wsl_root"] = "/home/user1/" + NEW_ATTEMPT + "/native/source-evidence"
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
