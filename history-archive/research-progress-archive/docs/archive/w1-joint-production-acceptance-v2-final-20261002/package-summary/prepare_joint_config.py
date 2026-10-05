"""Generate the registered joint proposal; never creates an execution attempt."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ATTEMPT = "w1-task-outcome-g1-g2-joint-v2-once"
REMOTE_ROOT = "/home/user1/" + ATTEMPT

def write(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                      indent=2, allow_nan=False) + "\n", encoding="utf-8")

def build():
    plans = ROOT.parent
    proposal = json.loads((plans / "w1-label-v6-offline-acceptance-v1" / "parent-split-proposal.json").read_text(encoding="utf-8"))
    split = {**proposal, "schema": "w1-action-conditioned-task-outcome-parent-split/1.0.0",
             "attempt": ATTEMPT}
    write("parent-split.json", split)
    old = json.loads((plans / "w1-eawm-jepa-world-model-training-v1-preparation" / "experiment-matrix.json").read_text(encoding="utf-8"))
    matrix = {key: old[key] for key in ("world_model_optimizer", "world_model_loss", "world_model_model_selection")}
    matrix.update({"schema": "w1-joint-sequence-supervision-matrix/1.0.0", "attempt": ATTEMPT,
        "parent_count": 40, "parent_split_file": "parent-split.json",
        "candidate_continuation_id": "hungarian-v1-fixed", "maximum_legal_candidates_per_window": 25,
        "splits": {role: [row for row in split["parents"] if row["proposed_split"] == role]
                   for role in ("train", "model_selection", "prediction_confirmation")},
        "repeats": {role: 1 for role in ("train", "model_selection", "prediction_confirmation")},
        "bounds": {"parents": 40, "prefix_steps_per_parent_upper": 4,
                   "candidate_count_per_window_upper_including_noop": 25,
                   "branch_steps_per_branch_upper": 14,
                   "continuation_decisions_per_branch_upper": 13},
        "preference_configuration": {"task_confirmation": [0.8, 0.2], "task_component_scale": 0.5},
        "source_semantics": {"task_completion_mode": "arrival_to_region", "deadline_basis": "physical_arrival", "arrival_radius": 0.0},
        "world_model_architecture": {**old["world_model_architecture"], "horizon_task_outcome_dim": 3},
        "world_model_device": "cuda:0", "world_model_seeds": [8201, 8202, 8203],
        "world_model_variants": {"G1": "event_loss_disabled", "G2": "event_loss_enabled"},
        "world_model_training_windows": 24, "world_model_selection_windows": 8,
        "world_model_confirmation_windows": 8, "world_model_maximum_epochs": 20,
        "world_model_batch_size": 25, "world_model_maximum_train_candidate_rows": 600,
        "world_model_maximum_updates_per_variant_seed": 480,
        "world_model_batch_organization": "one complete window per optimizer update; equal window scheduling",
        "transition_horizon": "next public observation for JEPA/state; complete fixed continuation for returns and task outcomes",
        "transparent_utility_configuration": {"scope": "public_only_ideal_continuation_surrogate",
            "residual_target": "full continuation vector return minus public transparent horizon components"},
        "gppo_calls": 0, "task_comparison_calls": 0})
    matrix["world_model_loss"]["beta_horizon_task_outcome"] = 1.0
    write("experiment-matrix.json", matrix)
    # Counts follow the registered 24/8/8 whole-window design, not old resource caps.
    calls = {"environment_steps": 14160, "resets_upper": 40, "branches": 1000,
             "public_rule_decisions": 13160, "candidate_scans": 1000,
             "forced_first_actions": 1000, "snapshot_captures": 40, "branch_snapshot_copies": 1000}
    storage = 2 * 1024**3
    stages = {
        "staging_and_zero_step_gate": {"wall_seconds": 180, "complete_process_cpu_seconds": 80},
        "label_qualification": {**calls, "wall_seconds": 828, "complete_process_cpu_seconds": 445},
        "world_model_training_and_selection": {"model_initializations_or_loads": 6,
            "world_optimizer_updates": 2880, "world_batch_forwards": 6720,
            "world_sample_evaluations": 168000, "world_backward_calls": 2880, "checkpoint_writes": 6,
            "wall_seconds": 1800, "complete_process_cpu_seconds": 3600},
        "prediction_confirmation": {"model_initializations_or_loads": 6,
            "world_batch_forwards": 48, "world_sample_evaluations": 1200,
            "checkpoint_loads": 6, "wall_seconds": 180, "complete_process_cpu_seconds": 360},
        "settlement_and_verified_export": {"wall_seconds": 120, "complete_process_cpu_seconds": 80}}
    for stage in stages.values():
        stage["active_storage_bytes"] = storage
    totals = {**calls, "model_initializations_or_loads": 12, "world_optimizer_updates": 2880,
              "world_batch_forwards": 6768, "world_sample_evaluations": 169200,
              "checkpoint_writes": 6, "checkpoint_loads": 6,
              "world_backward_calls": 2880,
              "wall_seconds": sum(s["wall_seconds"] for s in stages.values()) + 30,
              "complete_process_cpu_seconds": sum(s["complete_process_cpu_seconds"] for s in stages.values()) + 30,
              "all_resident_rss_bytes": 4*1024**3, "active_storage_bytes": storage,
              "aggregate_native_plus_verified_export_bytes": 4*1024**3,
              "gpu_memory_bytes": 8*1024**3, "gpu_devices": 1,
              "gppo_updates": 0, "task_comparison_calls": 0}
    write("RESOURCE_REQUEST.json", {"schema": "w1-joint-data-world-model-request/1.0.0", "attempt": ATTEMPT,
        "status": "NOT_APPROVED", "runner_ready": False,
        "research_training": True, "policy_or_task_comparison": False,
        "stages": stages, "totals": totals,
        "accounting_reserves": {"windows_post_preflight_process_cpu_seconds": 30, "cross_system_wall_seconds": 30},
        "gpu": {"physical_device": 1, "logical_device": "cuda:0", "exclusive_allocation_required": True,
                "free_memory_minimum_bytes": 9*1024**3, "peak_allocated_memory_bytes": 8*1024**3,
                "aggregate_gpu_process_accounting": "owned worker only; no DDP; one GPU"},
        "stops": ["identity_or_runtime_mismatch", "data_gate_not_passed", "nonfinite", "budget_failure", "technical_failure_no_retry"],
        "budget_basis": "collection from inclusive v6 measurement scaled by max bounds and factor3; learning times require synthetic production benchmark before runner_ready"})
    write("data-gate.json", {"schema": "w1-sequence-data-admission/1.0.0", "all_registered_parents_required": True,
        "split_window_counts": {"train": 24, "model_selection": 8, "prediction_confirmation": 8},
        "physical_and_expiry_valid_min_per_window": 1,
        "train_physical_positive_min_parents": 4, "train_expiry_positive_min_parents": 4,
        "train_host_valid_min_parents": 8,
        "reason": "minimum outcome variation spanning parents; engineering support for a small exploratory learner, not statistical power proof",
        "unknown": "preserve null/mask false; NOOP no target; missing/no opportunity never zero or replacement",
        "complete_return_required_all_candidates": True})
    write("prediction-gates.json", {"schema": "w1-joint-world-model-prospective-gates/1.0.0",
        "confirmation_parents": 8, "g1_vs_transparent": {"regret_improvement_minimum": 0.005,
            "noninferior_parent_minimum": 6, "positive_improvement_seed_minimum": 2},
        "g2_vs_g1": {"regret_improvement_minimum": 0.005, "noninferior_parent_minimum": 6,
            "positive_improvement_seed_minimum": 2},
        "inference": "exploratory within-study held-out parents; report paired parent differences and bootstrap uncertainty; no population efficacy claim",
        "next_gppo": "separate application only; no branch runs GPPO in this protocol"})
    matrix["prediction_gates"] = json.loads((ROOT / "prediction-gates.json").read_text(encoding="utf-8"))
    write("experiment-matrix.json", matrix)
    inputs = json.loads((ROOT / "runtime-inputs.json").read_text(encoding="utf-8"))
    inputs["source_run"]["wsl_root"] = REMOTE_ROOT + "/native/source-evidence"
    inputs["purpose"] = "portable source provenance checked by joint_inputs.verify_joint_inputs"
    write("runtime-inputs.json", inputs)
    write("launch-contract.json", {"schema": "w1-eawm-jepa-launch-contract/2.1.0", "attempt": ATTEMPT,
        "authorization_status": "NOT_APPROVED", "one_shot": True, "automatic_retry": False,
        "integration_test": False, "runtime_mode": "remote_joint", "native_entry": "joint_remote_native.py",
        "frozen_windows_entry": "launch_joint_once.py", "native_python": "/home/user1/.venvs/w1-runtime-v1/bin/python",
        "host": "172.17.27.173", "user": "user1", "native_execution_root": REMOTE_ROOT,
        "resource_request_sha256": hashlib.sha256((ROOT/"RESOURCE_REQUEST.json").read_bytes()).hexdigest(),
        "runtime_identity_sha256": "af32685059d7c3107ab6d0d2f34bc0e9e71c118ec8179d394a216798d7b62cb1",
        "accounting_reserves": {"windows_post_preflight_process_cpu_seconds": 30, "cross_system_wall_seconds": 30}})
    return {"attempt": ATTEMPT, "formal_attempt_created": False, "status": "NOT_APPROVED"}

if __name__ == "__main__":
    print(json.dumps(build(), sort_keys=True))
