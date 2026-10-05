"""Check unchanged label research and preserve the original frozen identity."""
import hashlib
import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
PLANS = EVIDENCE.parent
OLD = PLANS / "w1-action-conditioned-task-outcome-label-repair-v5-budget-accounting"
NEW = PLANS / "w1-action-conditioned-task-outcome-label-repair-v6-dependency-cpu-settlement"
EXPECTED_OLD = {
    "execution-manifest.json": "e4703e4be26f5e3d2bffd40f7c9c768d52c3c4a2465fcf9d3719dce58fe5edf4",
    "hashes.json": "ebc75ab39bd93a73b17909c73912bdf0a62c90918ce9641976d776db46d4b5ec",
    "RESOURCE_REQUEST.json": "8be3d85904c904f2e6003bc8a19dac81177e68c0f4df70bd997a33afaa428162",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


old_identity = {name: digest(OLD / name) for name in EXPECTED_OLD}
if old_identity != EXPECTED_OLD:
    raise RuntimeError("OLD_FROZEN_OUTER_IDENTITY_CHANGED")
old_manifest = json.loads((OLD / "execution-manifest.json").read_text(encoding="utf-8"))
old_files = old_manifest["files"]
if any(not (OLD / name).is_file() or digest(OLD / name) != expected for name, expected in old_files.items()):
    raise RuntimeError("OLD_FROZEN_CONTENT_CHANGED")
comparisons = {}
for name in ("experiment-matrix.json", "parent-split.json", "RESOURCE_REQUEST.json"):
    previous = json.loads((OLD / name).read_text(encoding="utf-8"))
    current = json.loads((NEW / name).read_text(encoding="utf-8"))
    previous.pop("attempt", None)
    current.pop("attempt", None)
    comparisons[name] = previous == current
    if not comparisons[name]:
        raise RuntimeError("RESEARCH_CONFIGURATION_CHANGED:" + name)
for name in ("environment-config-contract.json", "task_outcome_contract.py",
             "public_history.py", "public_transition_contract.py", "public_event_targets.py"):
    comparisons[name] = digest(OLD / name) == digest(NEW / name)
    if not comparisons[name]:
        raise RuntimeError("LABEL_OR_INPUT_SEMANTICS_CHANGED:" + name)
summary = {"old_outer_identity": old_identity, "old_content_file_count": len(old_files),
           "old_frozen_files_verified": True, "comparisons": comparisons,
           "research_budget_changed": False, "research_semantics_changed": False,
           "attempt_changed_only_in_research_json": True,
           "formal_attempt_created": False, "resource_request_status": "NOT_APPROVED"}
(EVIDENCE / "research-identity-comparison.json").write_text(
    json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
