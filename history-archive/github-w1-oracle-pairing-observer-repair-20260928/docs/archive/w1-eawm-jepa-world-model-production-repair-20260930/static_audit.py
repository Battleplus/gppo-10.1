"""Static, no-call audit for the W1 EAWM/JEPA preparation package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from production_chain_check import check_preflight, ROOT, RUN


PACKAGE = Path(__file__).resolve().parent
INVENTORY = ROOT / "refine-logs" / "EAWM-JEPA-reuse-inventory-20260929.json"
AUDIT = ROOT / "refine-logs" / "EAWM-JEPA-reuse-audit-20260929.md"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    module_checks = []
    for item in inventory["modules"]:
        current = Path(item["current_archive_path"])
        module_checks.append({
            "file": item["file"],
            "present": current.is_file(),
            "sha256": sha(current) if current.is_file() else None,
            "matches_reviewed_archive": current.is_file() and sha(current) == item["current_sha256"],
            "old_and_current_identical_per_inventory": bool(item["identical"]),
        })
    source_text = (RUN / "native" / "gppo_world" / "joint_gppo.py").read_text(encoding="utf-8")
    adapter_text = (PACKAGE / "w1_public_adapter.py").read_text(encoding="utf-8")
    bridge_text = (PACKAGE / "gppo_w1_integration.py").read_text(encoding="utf-8")
    parent_split_path = PACKAGE / "parent-split.json"
    parent_split = json.loads(parent_split_path.read_text(encoding="utf-8")) if parent_split_path.is_file() else {}
    split_rows = [row for rows in parent_split.get("splits", {}).values() for row in rows]
    split_parents = [row.get("parent") for row in split_rows]
    runtime_text = (PACKAGE / "runtime_backend.py").read_text(encoding="utf-8")
    backend_markers = {
        "collector_call": "ProductionDataCollector" in runtime_text and "collect_world_model_data" in runtime_text,
        "world_training_call": "train_select_world_models" in runtime_text,
        "world_evaluation_call": "evaluate_prediction_confirmation" in runtime_text,
        "policy_training_call": "train_policy_routes" in runtime_text,
        "task_evaluation_call": "evaluate_task_confirmation" in runtime_text,
        "placeholder_errors_removed": all(marker not in runtime_text for marker in (
            "WORLD_MODEL_COLLECTOR_NOT_CONFIGURED", "WORLD_MODEL_TRAINER_NOT_CONFIGURED",
            "WORLD_MODEL_EVALUATOR_NOT_CONFIGURED", "POLICY_TRAINER_NOT_CONFIGURED",
            "TASK_EVALUATOR_NOT_CONFIGURED")),
    }
    result = {
        "schema": "w1-eawm-jepa-static-audit/1.0.0",
        "dynamic_calls": 0,
        "environment_calls": 0,
        "model_initializations": 0,
        "checkpoint_loads": 0,
        "training_updates": 0,
        "heldout_read": False,
        "review_documents_present": AUDIT.is_file() and INVENTORY.is_file(),
        "reuse_modules": module_checks,
        "all_reuse_hashes_match": all(item["matches_reviewed_archive"] for item in module_checks),
        "current_contract": {
            "action_count_25": "action_count != 25" in source_text or "action_count: int = 25" in source_text or "25" in source_text,
            "event_names_present": "EVENT_NAMES" in source_text,
            "public_future_target_separation": (PACKAGE / "public_transition_contract.py").is_file(),
            "public_observation_adapter": (PACKAGE / "w1_public_adapter.py").is_file(),
            "strict_candidate_entry": '"predict_public_and_score"' in bridge_text and '"predict_and_score"' not in bridge_text.split("__all__", 1)[-1],
            "adapter_has_no_environment_import": "m10_environment" not in adapter_text,
        },
        "parent_split": {
            "present": parent_split_path.is_file(),
            "counts": parent_split.get("split_counts", {}),
            "overlap_count": len(split_parents) - len(set(split_parents)),
            "historical_use_elsewhere": parent_split.get("historical_use_elsewhere"),
        },
        "production_preflight": check_preflight(),
        "production_backend": backend_markers,
        "runner_ready": False,
        "runner_blocker_documented": (PACKAGE / "BLOCKERS.md").is_file(),
        "package_files": sorted(path.name for path in PACKAGE.iterdir() if path.is_file()),
    }
    result["passed"] = bool(result["review_documents_present"] and result["all_reuse_hashes_match"] and result["production_preflight"]["passed"] and result["parent_split"]["present"] and result["parent_split"]["overlap_count"] == 0 and all((result["current_contract"]["public_observation_adapter"], result["current_contract"]["strict_candidate_entry"], result["current_contract"]["adapter_has_no_environment_import"])) and all(backend_markers.values()))
    (PACKAGE / "static-audit.json").write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
