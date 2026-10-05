"""Derive a NOT_APPROVED data-only proposal; never execute or stage a worker."""
from __future__ import annotations

import json
import math
from pathlib import Path

from analyze_v6 import PACKAGE, canonical, read_json, sha, write_json


def main():
    output = Path(__file__).resolve().parent
    workspace = output.parents[1]
    analysis = read_json(output / "analysis.json")
    tape_path = workspace / "runs/w1-light-repaired-fair-rerun-v2-nativefs-once/tapes-train.json"
    expected = "f2e6f7b2d7590989e305be5d7c6faf685369e15563a7b2fefb18c0468e7aa510"
    if sha(tape_path.read_bytes()) != expected:
        raise ValueError("SOURCE_TRAIN_TAPE_IDENTITY_MISMATCH")
    tapes = read_json(tape_path)
    prior_path = workspace / "research-plans/w1-eawm-jepa-world-model-production-repair-v4-e2e-integration/parent-split.json"
    prior = read_json(prior_path)
    registered = [r for split in prior["splits"].values() for r in split]
    qualification_path = workspace / "research-plans" / PACKAGE / "parent-split.json"
    registered += read_json(qualification_path)["parents"]
    prior_scenarios = {r["scenario_sha256"] for r in registered}
    prior_structures = {r["structural_sha256"] for r in registered}
    names = [f"train-{i:04d}" for i in range(64, 104)]
    parents = []
    for index, parent in enumerate(names):
        found = [r for r in tapes if r["parent"] == parent]
        if len(found) != 1:
            raise ValueError("SOURCE_PARENT_NOT_UNIQUE:" + parent)
        row = found[0]
        if row.get("condition") != "W1" or row.get("generator_split") != "train":
            raise ValueError("SOURCE_DISTRIBUTION_MISMATCH:" + parent)
        if canonical(row["scenario"]) != row["scenario_sha256"]:
            raise ValueError("SCENARIO_CONTENT_DIGEST_MISMATCH:" + parent)
        role = "train" if index < 24 else "model_selection" if index < 32 else "prediction_confirmation"
        parents.append({"parent": parent, "proposed_split": role, "repeat": 0,
                        **{k: row[k] for k in ("tape_index", "scenario_sha256", "structural_sha256", "exogenous_key")},
                        "overlaps_prior_registered_scenario": row["scenario_sha256"] in prior_scenarios,
                        "overlaps_prior_registered_structure": row["structural_sha256"] in prior_structures,
                        "historical_use_elsewhere": "not_proven_absent; do not call globally unseen"})
    if any(r["overlaps_prior_registered_scenario"] or r["overlaps_prior_registered_structure"] for r in parents):
        raise ValueError("PROPOSED_IDENTITIES_OVERLAP_PRIOR_REGISTERED_MATRIX_NO_SUBSTITUTION")
    if len({r["scenario_sha256"] for r in parents}) != 40 or len({r["structural_sha256"] for r in parents}) != 40:
        raise ValueError("PROPOSED_PARENT_IDENTITIES_NOT_DISTINCT")
    parent_payload = {"status": "NOT_APPROVED_PROPOSAL", "selection_rule": "First fixed contiguous train tape parents 64..103 after previous registered matrix and v6 qualification parents; assign 24/8/8 by order, without reading outcomes or choosing by difficulty.",
                      "source_train_tape_sha256": expected, "source_registered_split_sha256": sha(prior_path.read_bytes()),
                      "source_v6_qualification_split_sha256": sha(qualification_path.read_bytes()),
                      "sealed_task_results_read": False, "new_independence_claim": "Within-study parent-disjoint proposal; historical global novelty is unproven.", "parents": parents}
    write_json(output / "parent-split-proposal.json", parent_payload)
    ratio = max(14160 / 1106, 1000 / 108, 40 / 8)
    infra = analysis["infrastructure_settlement"]
    scope = infra["cpu_scope_breakdown"]["native_process_tree_sample"]
    label_cpu = math.ceil(3 * infra["label_stage_complete_process_cpu_seconds"] * ratio)
    label_wall = math.ceil(3 * (scope["sample_ended_monotonic"] - scope["sample_started_monotonic"]) * ratio)
    counts = {"environment_steps": 14160, "resets_upper": 40, "branches": 1000,
              "public_rule_decisions": 13160, "candidate_scans": 1000,
              "forced_first_actions": 1000, "snapshot_captures": 40,
              "branch_snapshot_copies": 1000}
    zero = {"model_initializations_or_loads": 0, "model_forwards": 0,
            "checkpoint_loads_or_writes": 0, "optimizer_updates": 0,
            "gppo_training": 0, "task_comparison": 0}
    request = {"schema": "w1-task-outcome-sequence-data-proposal/1.0.0", "status": "NOT_APPROVED",
               "submission_status": "NOT_EXECUTABLE_PENDING_PRODUCTION_REPAIR_AND_ACCEPTANCE",
               "proposed_attempt": "w1-action-conditioned-task-outcome-sequence-supervision-v1-proposed-once",
               "runner_ready": False, "runtime": "CPU label collector; remote GPU training is outside this proposal",
               "purpose": "Complete branch-level multi-objective and causal audit supervision; acquire disjoint candidate-data roles, not train models or establish world-model benefit.",
               "matrix": {"parents": 40, "train": 24, "model_selection": 8, "prediction_confirmation": 8,
                          "repeats_per_parent": 1, "windows_upper": 40, "legal_candidates_upper": 25,
                          "prefix_steps_upper": 4, "branch_steps_upper": 14, "horizon_seconds": 18,
                          "continuation_id": "hungarian-v1-fixed", "no_opportunity": "Retain missing with reason, no replacement or zero imputation."},
               "parent_split_proposal_sha256": sha((output / "parent-split-proposal.json").read_bytes()),
               "totals": {**counts, **zero, "wall_seconds": 20 + label_wall + 38 + 2,
                          "complete_process_cpu_seconds": 20 + label_cpu + 39 + 1,
                          "all_resident_rss_bytes": 2147483648, "active_storage_bytes": 536870912,
                          "aggregate_native_plus_export_bytes": 1073741824},
               "stages": {"staging_and_zero_step_gate": {"wall_seconds": 20, "complete_process_cpu_seconds": 20},
                          "label_collection": {**counts, **zero, "wall_seconds": label_wall, "complete_process_cpu_seconds": label_cpu},
                          "settlement_and_verified_export": {"wall_seconds": 38, "complete_process_cpu_seconds": 39}},
               "reserves": {"windows_post_preflight_cpu_seconds": 1, "cross_system_wall_seconds": 2},
               "derivation": {"environment_steps": "40*4 + 40*25*14 = 14160",
                              "public_rule_decisions": "40*4 + 40*25*13 = 13160; forced first actions counted separately",
                              "resource_scale": "max(14160/1106 steps, 1000/108 branches, 40/8 resets); multiply measured inclusive v6 label-stage CPU and native-tree wall by that scale and by 3, then ceil. No supervisor CPU is added to the inclusive tree.",
                              "scale": ratio, "measured_label_cpu_seconds": infra["label_stage_complete_process_cpu_seconds"],
                              "measured_tree_wall_seconds": scope["sample_ended_monotonic"] - scope["sample_started_monotonic"],
                              "memory_storage": "Retain explicit 2 GiB RSS, 512 MiB active and 1 GiB aggregate hard caps; these are limits, not evidence that extended records fit. Source payload and 40 windows must fit; serialize/size-test production fixtures before submission, never raise caps silently."},
               "required_new_records": ["Initial target lifecycle and execution identity, label-side only.",
                                        "Per continuation step: real scalar reward, independent multi-objective reward components, all-task outcome counts and aggregate energy before/after; include forced first action.",
                                        "Target and global terminal lifecycle, observation cutoff, native terminated/truncated, original reward/preference and each fixed-continuation action.",
                                        "Separate one-step public-state target from horizon task label; explicit target time/horizon and masks.",
                                        "Completion identity must refer to observed physical completion, not just the last accepted execution token."],
               "preconditions_before_new_approval": ["Implement and test attribution and aggregate-return records in an independent successor; keep all v6 evidence intact.",
                                                    "Fix CPU scope/serialization tail accounting without weakening the old tolerance or double-counting; rederive bounds if measured scopes change.",
                                                    "Keep immutable package inputs separate from runtime accounting output; validate final isolated staged collector closure and controlled export.",
                                                    "Do not inspect proposed prediction-confirmation outcomes during development. Historical usage identity audit remains required.",
                                                    "Freeze a new executable manifest, NOT_APPROVED request, one-shot external authorization and unique command only after final tests."],
               "stops": ["Any identity, timeline, non-finite, ledger, budget or export violation stops and seals, without repair/retry.",
                         "No-opportunity remains missing; no outcome-based replacement or extra repeat.",
                         "Complete collection does not authorize world-model or GPPO training."]}
    proposed = workspace / "runs" / request["proposed_attempt"]
    if proposed.exists():
        raise ValueError("PROPOSED_ATTEMPT_PATH_ALREADY_EXISTS")
    request["proposed_attempt_path_absent_at_preparation"] = True
    write_json(output / "RESOURCE_REQUEST-proposal.json", request)
    print(json.dumps({"status": request["status"], "runner_ready": False,
                      "parent_identity_count": len(parents), "global_historical_unseen": False,
                      "totals": request["totals"], "formal_attempt_created": False}))


if __name__ == "__main__":
    main()
