"""Freeze the bounded relative-utility comparison without running it."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from learning_schema import load_records
from manifest_contract import build_content_files
from freeze_wiring import freeze_identity


ROOT = Path(__file__).resolve().parent
TAPE = Path(r"E:\Z博士\research-plans\w1-light-repaired-fair-rerun-infra-repair-v1\tapes-train.json")
OLD_MATRIX = ROOT.parent / "w1-action-outcome-prior-prototype-v1" / "experiment-matrix.json"
RUN_RECORDS = Path(r"E:\Z博士\runs\w1-action-outcome-learning-loop-history-repair-v1-once\run-once\learning-records.jsonl")
ATTEMPT = "w1-action-relative-auxiliary-wiring-contract-repair-v1-once"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def source_index() -> dict:
    paths = (TAPE, OLD_MATRIX, RUN_RECORDS)
    return {
        "schema": "w1-action-relative-auxiliary-input-index/1.0.0",
        "inputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths],
        "read_only_analysis": {"environment_calls": 0, "reset_or_step_calls": 0, "model_initializations": 0, "model_forwards": 0, "training_updates": 0, "checkpoint_loads": 0},
        "excluded": ["sealed old prediction-evaluation labels", "sealed task-comparison parents and outcomes", "checkpoints and budget SQLite"],
    }


def audit_existing_records() -> dict:
    records = load_records(RUN_RECORDS)
    matrix = json.loads(OLD_MATRIX.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    parents: dict[str, set[str]] = {}
    continuations: set[str] = set()
    rows = 0
    for line in RUN_RECORDS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        rows += 1
        role = str(value["split_role"])
        counts[role] = counts.get(role, 0) + 1
        parents.setdefault(role, set()).add(str(value["parent"]))
        continuations.add(str(value["continuation_id"]))
    expected = {"train": 219, "model_selection": 68, "prediction_evaluation": 111}
    if {key: counts.get(key, 0) for key in expected} != expected:
        raise RuntimeError(f"reusable label counts changed: {counts}")
    if continuations != {"hungarian-v1-fixed"}:
        raise RuntimeError(f"unexpected continuation identities: {continuations}")
    matrix_parents = {
        role: {str(item["parent"]) for item in matrix["groups"][role]}
        for role in ("train", "model_selection", "prediction_evaluation")
    }
    record_parents = {role: {str(row.parent) for row in records if row.split_role == role} for role in matrix_parents}
    if any(record_parents[role] != matrix_parents[role] for role in matrix_parents):
        raise RuntimeError("reused label parents do not match the frozen prior matrix")
    return {
        "schema": "w1-action-relative-auxiliary-data-reuse/1.0.0",
        "reused_training_records": counts["train"],
        "reused_model_selection_records": counts["model_selection"],
        "old_prediction_evaluation_records": counts["prediction_evaluation"],
        "reused_training_parents": len(parents["train"]),
        "reused_model_selection_parents": len(parents["model_selection"]),
        "old_prediction_evaluation_parents": len(parents["prediction_evaluation"]),
        "reused_continuation": "hungarian-v1-fixed",
        "validated_record_count": len(records),
        "complete_candidate_windows_validated": True,
        "valid_targets_and_training_flags_validated": True,
        "parent_identity_matches_prior_matrix": True,
        "old_prediction_evaluation_used_for_training": False,
        "training_eligible_unknown_targets": 0,
        "historical_policy_use_of_new_parents": "not established by the available manifests; claim limited to disjointness from this package's train/model-selection/diagnostic sets",
    }


def select_parents() -> dict:
    tape = json.loads(TAPE.read_text(encoding="utf-8"))
    old = json.loads(OLD_MATRIX.read_text(encoding="utf-8"))
    excluded: set[str] = set()
    for group in old["groups"].values():
        excluded.update(str(item["parent"]) for item in group)
    eligible = sorted((row for row in tape if str(row["parent"]) not in excluded), key=lambda row: int(row["tape_index"]))
    chosen = eligible[:8]
    if len(chosen) != 8 or len({row["parent"] for row in chosen}) != 8:
        raise RuntimeError("could not select eight new prediction parents")
    return {
        "schema": "w1-action-relative-auxiliary-new-prediction-parents/1.0.0",
        "selection_rule": "sort frozen train tape by tape_index after excluding all 48 parents in the prior matrix; take the first eight before observing outcomes",
        "source_tape_sha256": sha256(TAPE),
        "excluded_prior_matrix_parent_count": len(excluded),
        "parents": [{key: row[key] for key in ("parent", "tape_index", "generator_split", "condition", "name", "scenario_sha256", "structural_sha256", "exogenous_key")} for row in chosen],
        "repeats": 3,
        "diagnostic_parent_status": "none of the eight parents is in the prior prediction-evaluation set",
        "task_comparison_status": "the sealed task-comparison parents remain excluded and unexecuted",
        "sealed_task_parents": sorted(str(item["parent"]) for item in old["groups"]["task_evaluation"]),
        "sealed_task_repeats": 3,
        "independence_limit": "historical use by other strategy runs is not proven absent by the available manifests",
    }


def derive_budget(selection: dict) -> dict:
    units = len(selection["parents"]) * selection["repeats"]
    task_units = len(selection["sealed_task_parents"]) * selection["sealed_task_repeats"]
    max_candidates = 25
    # The first two stages reuse sealed labels. These are technical ceilings for
    # one new confirmation pass, derived from the prior label-unit rate.
    return {
        "schema": "w1-action-relative-auxiliary-derived-budget/1.0.0",
        "new_prediction_units": units,
        "max_candidates_per_unit": max_candidates,
        "stage_dynamic_counts": {
            "new_prediction_confirmation": {
                "environment_steps": units * 468,
                "resets_upper": units,
                "branches": units * max_candidates,
                "branch_snapshot_copies": units * max_candidates,
                "forced_first_actions": units * max_candidates,
                "snapshot_captures": units,
                "candidate_scans": units * 18,
                "public_rule_decisions": units * 443,
            },
            "supervised_training_and_selection": {
                "model_initializations_or_loads": 6,
                "checkpoint_writes": 6,
                "model_batch_forwards": 38400,
                "model_sample_evaluations": 172200,
                "offline_predictor_updates": 28800,
            },
            "independent_prediction_evaluation": {
                "model_initializations_or_loads": 6,
                "model_batch_forwards": units * 6,
                "model_sample_evaluations": units * max_candidates * 6,
            },
            "conditional_task_comparison": {
                "task_parent_count": len(selection["sealed_task_parents"]),
                "task_episodes": task_units * 4,
                "environment_steps": task_units * 4 * 18,
                "resets_upper": task_units * 4,
                "candidate_scans": task_units * 4 * 18,
                "public_rule_decisions": task_units * 4 * 18,
                "model_batch_forwards": task_units * 6,
                "model_sample_evaluations": task_units * 6 * max_candidates,
            },
        },
        "historical_credit": 0,
        "no_budget_borrowing": True,
    }


def resource_request(derived: dict) -> dict:
    gib = 1024 ** 3
    stage = {
        "staging_and_zero_step_gate": {"wall_seconds": 600, "complete_process_cpu_seconds": 1200, "artifact_bytes": 256 * 1024 ** 2},
        "data_reuse_audit": {"wall_seconds": 300, "complete_process_cpu_seconds": 600, "artifact_bytes": 256 * 1024 ** 2, "environment_steps": 0, "model_forwards": 0, "training_updates": 0},
        "new_prediction_confirmation": {"wall_seconds": 7200, "complete_process_cpu_seconds": 14400, "artifact_bytes": 12 * gib, **derived["stage_dynamic_counts"]["new_prediction_confirmation"]},
        "supervised_training_and_selection": {"wall_seconds": 12000, "complete_process_cpu_seconds": 24000, "artifact_bytes": 18 * gib, **derived["stage_dynamic_counts"]["supervised_training_and_selection"]},
        "independent_prediction_evaluation": {"wall_seconds": 1200, "complete_process_cpu_seconds": 2400, "artifact_bytes": 18 * gib, **derived["stage_dynamic_counts"]["independent_prediction_evaluation"]},
        "conditional_task_comparison": {"wall_seconds": 1800, "complete_process_cpu_seconds": 3600, "artifact_bytes": 20 * gib, **derived["stage_dynamic_counts"]["conditional_task_comparison"]},
        "settlement": {"wall_seconds": 600, "complete_process_cpu_seconds": 1200, "artifact_bytes": 24 * gib},
        "verified_export": {"wall_seconds": 600, "complete_process_cpu_seconds": 1200, "artifact_bytes": 24 * gib},
    }
    totals = {
        "environment_steps": stage["new_prediction_confirmation"]["environment_steps"] + stage["conditional_task_comparison"]["environment_steps"],
        "resets_upper": stage["new_prediction_confirmation"]["resets_upper"] + stage["conditional_task_comparison"]["resets_upper"],
        "branches": stage["new_prediction_confirmation"]["branches"],
        "branch_snapshot_copies": stage["new_prediction_confirmation"]["branch_snapshot_copies"],
        "forced_first_actions": stage["new_prediction_confirmation"]["forced_first_actions"],
        "snapshot_captures": stage["new_prediction_confirmation"]["snapshot_captures"],
        "candidate_scans": stage["new_prediction_confirmation"]["candidate_scans"] + stage["conditional_task_comparison"]["candidate_scans"],
        "public_rule_decisions": stage["new_prediction_confirmation"]["public_rule_decisions"] + stage["conditional_task_comparison"]["public_rule_decisions"],
        "model_initializations_or_loads": 12,
        "checkpoint_writes": 6,
        "model_batch_forwards": 38400 + stage["independent_prediction_evaluation"]["model_batch_forwards"] + stage["conditional_task_comparison"]["model_batch_forwards"],
        "model_sample_evaluations": 172200 + stage["independent_prediction_evaluation"]["model_sample_evaluations"] + stage["conditional_task_comparison"]["model_sample_evaluations"],
        "offline_predictor_updates": 28800,
        "wall_seconds": 24300,
        "complete_process_cpu_seconds": 48600,
        "artifact_bytes": 24 * gib,
        "aggregate_native_plus_verified_export_bytes": 48 * gib,
        "all_resident_rss_bytes": 4 * gib,
    }
    return {
        "schema": "w1-action-relative-auxiliary-resource-request/1.0.0",
        "status": "NOT_APPROVED",
        "attempt": ATTEMPT,
        "prior_decision": "PREDICTION_GATE_NOT_PASSED",
        "purpose": "One fixed-continuation A/B objective comparison with one new prediction confirmation and a gated conditional task comparison.",
        "variants": {"A": "absolute remaining utility SmoothL1", "B": "absolute SmoothL1 plus window-centered SmoothL1", "auxiliary_weight": 0.25, "beta": 1.0},
        "stage_limits": stage,
        "totals": totals,
        "progression": {
            "data_reuse_failure": "stop before any new collection",
            "confirmation_failure": "settle and stop before training",
            "prediction_gate": {
                "primary": "B minus A parent-macro selected regret <= -0.005 and B is no worse in at least 4 of 8 parents",
                "secondary": "report B minus transparent regret; it is not silently substituted for the primary comparison",
                "basis": "0.005 is one half of the frozen 0.01 task-materiality threshold, used as a predeclared prediction-stage effect size",
                "uncertainty": "report parent-level percentile bootstrap interval and per-parent/per-repeat values; eight parents support development interpretation only",
            },
            "task_comparison": "run only after the frozen prediction gate; task utility and CPU/wall standards remain separate",
        },
        "model_contract": {"architecture": "827-128-64-1 tanh MLP", "training_seeds": [7101, 7102, 7103], "windows_per_update": 1, "maximum_epochs": 100, "continuation_id": "hungarian-v1-fixed", "window_weighting": "equal per complete decision window for A and B", "selection_forward_accounting": "one counted forward per complete selection window per epoch"},
        "storage_contract": {"active_state": "WSL native ext4 only", "export": "single SHA-256 verified export after settlement"},
        "automatic_retry": False,
        "automatic_extension": False,
        "historical_credit": 0,
        "unused_credit_transfer": False,
    }


def package_hashes() -> dict:
    """Return the content digest mapping used by the shared identity writer."""
    return build_content_files(ROOT)


def main() -> int:
    reuse = audit_existing_records()
    selection = select_parents()
    derived = derive_budget(selection)
    write_json(ROOT / "data-reuse-audit.json", reuse)
    write_json(ROOT / "new-prediction-parent-selection.json", selection)
    write_json(ROOT / "derived-budget.json", derived)
    write_json(ROOT / "RESOURCE_REQUEST.json", resource_request(derived))
    write_json(ROOT / "input-index.json", source_index())
    checks = {
        "reused_train_records_219": reuse["reused_training_records"] == 219,
        "reused_selection_records_68": reuse["reused_model_selection_records"] == 68,
        "reused_records_schema_validated": reuse["validated_record_count"] == 398 and reuse["complete_candidate_windows_validated"],
        "reused_parent_identity_matches_matrix": reuse["parent_identity_matches_prior_matrix"],
        "old_prediction_records_excluded": reuse["old_prediction_evaluation_used_for_training"] is False,
        "new_parent_count_8": len(selection["parents"]) == 8,
        "new_parents_disjoint_from_prior_matrix": selection["excluded_prior_matrix_parent_count"] == 48,
        "new_confirmation_repeats_3": selection["repeats"] == 3,
        "sealed_task_parent_binding_present": len(selection["sealed_task_parents"]) == 8 and selection["sealed_task_repeats"] == 3,
        "resource_request_not_approved": resource_request(derived)["status"] == "NOT_APPROVED",
        "research_gate_preserved": True,
        "no_environment_or_model_calls_in_preparation": True,
    }
    write_json(ROOT / "checks.json", checks)
    # Use the same acyclic identity writer as every production and test entry.
    freeze_identity(ROOT, ATTEMPT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
