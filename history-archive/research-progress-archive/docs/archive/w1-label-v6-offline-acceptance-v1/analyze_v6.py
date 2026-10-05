"""Standard-library, read-only analysis of the consumed v6 qualification data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import time


ATTEMPT = "w1-action-conditioned-task-outcome-label-qualification-v6-dependency-cpu-settlement-once"
PACKAGE = "w1-action-conditioned-task-outcome-label-repair-v6-dependency-cpu-settlement"
HEADS = ("physical_on_time_completion", "task_expired", "host_confirmation")
EXPECTED = {
    "execution-manifest.json": "228e69e4086fa6ee409eccd4adf6c0bc20a55e31d3bc1cf6a332944bb97d52b6",
    "hashes.json": "e19e65073dab11105730170d4fc1621fe9c01b31c05ca4a3de4b76aa6233aba9",
}
DATA_SHA = "30b773cf5e8870b8c4abed9f2daa2939d09cfdd7b02359bf2ede910d518115fc"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> str:
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8"))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): sha(p.read_bytes())
            for p in sorted(root.rglob("*")) if p.is_file()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), name)
    return float(value)


def bucket(label):
    valid, value = label["valid"], label["value"]
    require(type(valid) is bool, "mask_not_boolean")
    require(type(value) is bool if valid else value is None, "unknown_not_null_or_invalid_label")
    return "positive" if value is True else "negative" if value is False else "unknown"


def validate_candidate(window, label, audit):
    """Recheck persisted evidence independently, without inventing initial state."""
    require(label["candidate_id"] == audit["candidate_id"], "candidate_join")
    require(label["action_id"] == audit["action"], "action_join")
    require(label["continuation_id"] == window["continuation_id"] == "hungarian-v1-fixed", "continuation_join")
    decision = finite(label["decision_time"], "decision_time")
    require(decision == window["decision_input"]["observation"]["time"], "decision_time_join")
    horizon = finite(label["horizon_time"], "horizon_time")
    trajectory = audit["post_action_trajectory"]
    require(bool(trajectory), "missing_trajectory")
    previous = decision
    for index, row in enumerate(trajectory):
        at = finite(row["time"], "trajectory_time")
        require(row["step"] == index and previous < at <= horizon, "trajectory_order")
        require(row["continuation_id"] == label["continuation_id"], "trajectory_continuation")
        if label["action_id"] != 24:
            task = row["task_lifecycle"]
            require(task["task_id"] == label["target_task_id"], "task_join")
            require(task["deadline"] == label["deadline"], "deadline_changed")
        previous = at
    end = trajectory[-1]
    require(bool(end["terminated"]) != bool(end["truncated"]), "episode_end")
    require(end["terminated"] or end["time"] == horizon, "truncation_not_at_horizon")
    require(end["time"] == label["observation_cutoff_time"], "cutoff_join")
    for head in HEADS:
        bucket(label[head])
        require(label[head]["valid"] == label["valid_mask"][head], "mask_join")
    if label["action_id"] == 24:
        require(label["target_task_id"] is None and label["target_uav_id"] is None, "noop_identity")
        require(all(not label[h]["valid"] and label[h]["reason"] == "noop_has_no_target_task" for h in HEADS), "noop_mask")
        return
    final_task = end["task_lifecycle"]
    completed = final_task["completed_at"]
    state = final_task["state"]
    if completed is not None:
        completed = finite(completed, "completion_time")
        require(state == "completed" and decision <= completed < label["deadline"] and completed <= end["time"], "completion_evidence")
        require(label[HEADS[0]]["value"] is True and label[HEADS[1]]["value"] is False, "completion_label")
        require(label[HEADS[0]]["time"] == completed, "completion_time_join")
    elif state == "expired":
        require(end["time"] >= label["deadline"], "expiry_time")
        require(label[HEADS[0]]["value"] is False and label[HEADS[1]]["value"] is True, "expiry_label")
    else:
        require(not label[HEADS[0]]["valid"] and not label[HEADS[1]]["valid"], "unresolved_mask")
    notice = final_task["completion_notice"]
    receipt = None if notice is None else notice.get("host_confirmation_time")
    if receipt is not None:
        receipt = finite(receipt, "receipt_time")
        require(completed is not None and completed <= receipt <= end["time"], "receipt_not_observed_in_window")
        require(label[HEADS[2]]["value"] is (receipt <= label["deadline"]), "receipt_label")
        require(label[HEADS[2]]["time"] == receipt, "receipt_time_join")
    elif state == "expired":
        require(label[HEADS[2]]["value"] is False and label[HEADS[2]]["valid"], "expired_host_mask")
    else:
        require(label[HEADS[2]]["value"] is None and not label[HEADS[2]]["valid"], "missing_receipt_mask")


def summary(rows):
    totals = {h: Counter() for h in HEADS}
    reasons = {h: Counter() for h in HEADS}
    status_counts = Counter()
    by_status = defaultdict(Counter)
    candidates, windows, invalid_attributions = [], [], []
    same_task_groups = 0
    informative_same_task_groups = 0
    raw_direct = corrected_direct = 0
    for window in rows:
        require(window["status"] == "complete", "analysis_requires_complete_row")
        require(window["schema"] == "w1-world-model-complete-window/2.0.0", "window_schema")
        require(window["split"] == "label_qualification", "window_split")
        require(len(window["actions"]) == len(set(window["actions"])), "duplicate_action")
        require(len(window["actions"]) == len(window["task_outcome_target"]) == len(window["candidate_branch_audit"]), "candidate_count")
        public = window["decision_input"]
        observation = public["observation"]
        decision = finite(observation["time"], "public_decision_time")
        require(public["frozen_before_candidate_branches"] is True, "input_not_frozen")
        require(window["actions"] == [i for i, v in enumerate(observation["mask"]) if v], "legal_set")
        public_hash = canonical({"observation": observation, "telemetry": public["telemetry"], "continuation": public["continuation_ids"]})
        require(window["input_hash"] == canonical({"nodes": window["current_nodes"], "history": window["history"], "relations": window["relations"], "legal": window["actions"], "continuation": public["continuation_ids"]}), "feature_identity")
        for record in public["telemetry"]:
            require(finite(record["measured_at"], "measurement_time") <= finite(record["received_at"], "receipt_time") <= decision, "future_public_telemetry")
        counts = {h: Counter() for h in HEADS}
        per_task = defaultdict(list)
        accepted = lost = 0
        for index, (label, audit) in enumerate(zip(window["task_outcome_target"], window["candidate_branch_audit"])):
            require(window["candidate_ids"][index] == label["candidate_id"], "candidate_order")
            require(window["actions"][index] == label["action_id"], "action_order")
            require(label["public_input_hash"] == public_hash, "raw_public_identity")
            require(label["environment_config_sha256"] == window["environment_config_identity"]["config_sha256"], "config_join")
            validate_candidate(window, label, audit)
            first_status = label["first_command"]["status"]
            status_counts[first_status] += 1
            accepted += first_status in ("accepted", "accepted_ack_lost")
            lost += first_status == "lost"
            for head in HEADS:
                key = bucket(label[head])
                counts[head][key] += 1
                totals[head][key] += 1
                if key == "unknown":
                    reasons[head][label[head]["reason"]] += 1
            by_status[first_status][bucket(label[HEADS[0]])] += 1
            final_task = audit["post_action_trajectory"][-1]["task_lifecycle"]
            physical_completed = label[HEADS[0]]["value"] is True
            direct = label["completion_by_first_command"]
            identifiable = label["action_specific_effect_identifiable"]
            raw_direct += identifiable is True
            # Audit-only correction: command identity cannot prove a completion
            # if the task never completed. Primary labels remain untouched.
            corrected_identifiable = identifiable is True and physical_completed
            corrected_direct += corrected_identifiable
            if identifiable is True and not physical_completed:
                invalid_attributions.append({"parent": window["parent"], "candidate_id": label["candidate_id"], "action_id": label["action_id"], "physical_completed": False, "expired": label[HEADS[1]]["value"], "original_completion_by_first_command": direct, "original_action_specific_effect_identifiable": identifiable, "derived_completion_by_first_command": False, "derived_action_specific_effect_identifiable": False, "reason": "identity_match_without_physical_completion"})
            if label["target_task_id"] is not None:
                per_task[label["target_task_id"]].append(label)
            vector = audit["vector_reward"]
            utility = 0.4 * finite(vector[0], "task_reward") + 0.2 * finite(vector[1], "energy_reward")
            require(math.isclose(utility, audit["true_utility"], abs_tol=1e-12), "first_step_utility")
            require(window["true_utility"][index] == audit["true_utility"], "utility_join")
            candidates.append({"parent": window["parent"], "repeat": window["repeat"], "window_id": window["window_id"], "candidate_id": label["candidate_id"], "action": label["action_id"], "task": label["target_task_id"], "uav": label["target_uav_id"], "first_status": first_status, "physical": label[HEADS[0]]["value"], "expired": label[HEADS[1]]["value"], "host": label[HEADS[2]]["value"], "physical_valid": label[HEADS[0]]["valid"], "host_valid": label[HEADS[2]]["valid"], "host_unknown_reason": label[HEADS[2]]["reason"], "physical_time": None if final_task is None else final_task["completed_at"], "host_time": label[HEADS[2]].get("time"), "deadline": label["deadline"], "decision_time": decision, "cutoff": label["observation_cutoff_time"], "raw_direct_completion": direct, "raw_identifiable": identifiable, "derived_identifiable_with_completion_evidence": corrected_identifiable, "first_step_true_utility": utility, "continuation_true_utility": None, "public_input_sha256": public_hash, "continuation_id": label["continuation_id"]})
        task_differences = []
        for task, labels in sorted(per_task.items()):
            valid_values = {l[HEADS[0]]["value"] for l in labels if l[HEADS[0]]["valid"]}
            informative = len(valid_values) > 1
            same_task_groups += 1
            informative_same_task_groups += informative
            task_differences.append({"task_id": task, "candidate_count": len(labels), "physical_difference": informative, "host_values": sorted({str(l[HEADS[2]]["value"]) for l in labels}), "actions": [l["action_id"] for l in labels]})
        utility_values = window["true_utility"]
        windows.append({"parent": window["parent"], "window_id": window["window_id"], "repeat": window["repeat"], "candidate_count": len(window["actions"]), "accepted": accepted, "lost": lost, "labels": {h: dict(counts[h]) for h in HEADS}, "same_task_comparisons": task_differences, "same_task_physical_difference": any(t["physical_difference"] for t in task_differences), "first_step_utility_min": min(utility_values), "first_step_utility_max": max(utility_values), "first_step_task_component_values": sorted({a["vector_reward"][0] for a in window["candidate_branch_audit"]}), "complete_continuation_utility_available": False, "public_telemetry_count": len(public["telemetry"]), "maximum_measurement_time": max(t["measured_at"] for t in public["telemetry"]), "maximum_receipt_time": max(t["received_at"] for t in public["telemetry"])})
    expected_parents = {f"train-{i:04d}" for i in range(56, 64)}
    coverage = len(rows) == 8 and {w["parent"] for w in rows} == expected_parents and all(any(l["valid_mask"][HEADS[0]] or l["valid_mask"][HEADS[1]] for l in w["task_outcome_target"]) for w in rows)
    return {"coverage_pass_original_rule": coverage, "parent_count": len(rows), "candidate_count": len(candidates), "label_counts": {h: dict(totals[h]) for h in HEADS}, "unknown_reasons": {h: dict(reasons[h]) for h in HEADS}, "first_status_counts": dict(status_counts), "physical_by_first_status": {s: dict(v) for s, v in by_status.items()}, "same_task_group_count": same_task_groups, "same_task_groups_with_physical_difference": informative_same_task_groups, "parents_with_same_task_physical_difference": sum(w["same_task_physical_difference"] for w in windows), "original_identifiable_count": raw_direct, "identifiable_count_requiring_physical_completion": corrected_direct, "invalid_attribution_count": len(invalid_attributions), "invalid_attributions": invalid_attributions, "windows": windows}, candidates


def verify_export(root):
    manifest = read_json(root / "export-manifest.json")
    bad = []
    for name, record in manifest["files"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), "export_path_escape")
        if not path.is_file() or sha(path.read_bytes()) != record["sha256"] or path.stat().st_size != record["bytes"]:
            bad.append(name)
    completion = read_json(root / "EXPORT_COMPLETE.json")
    return {"manifest_sha256": sha((root / "export-manifest.json").read_bytes()), "completion_binds_manifest": completion["export_manifest_sha256"] == sha((root / "export-manifest.json").read_bytes()), "manifest_file_count": len(manifest["files"]), "completion_file_count": completion["file_count"], "digest_mismatches": bad, "all_payload_digests_match": not bad, "original_completion_record": completion}


def ledger_summary(root):
    path = root / "run-once/budget.sqlite3"
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        tables = db.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall()
        result = {"tables": {name: schema for name, schema in tables}}
        calls = db.execute("SELECT stage,name,amounts,status,started,finished FROM calls").fetchall()
        totals, stages, statuses, names = Counter(), defaultdict(Counter), Counter(), Counter()
        for stage, name, amounts, status, started, finished in calls:
            values = json.loads(amounts)
            require(all(type(v) is int and v >= 0 for v in values.values()), "ledger_amount")
            totals.update(values)
            stages[stage].update(values)
            statuses[status] += 1
            names[name] += 1
            require(finished is not None and finished >= started, "ledger_unclosed_call")
        result.update({"call_count": len(calls), "totals": dict(totals), "stages": {s: dict(v) for s, v in stages.items()}, "statuses": dict(statuses), "operation_counts": dict(names)})
    return result


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    root = args.workspace / "runs" / ATTEMPT
    package = args.workspace / "research-plans" / PACKAGE
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    before = {"package": snapshot(package), "attempt": snapshot(root)}
    for name, digest in EXPECTED.items():
        require(before["package"][name] == digest and before["attempt"][name] == digest, "frozen_outer_identity")
    path = root / "run-once/world-model-windows.jsonl"
    require(sha(path.read_bytes()) == DATA_SHA, "data_identity")
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
    result, candidates = summary(rows)
    split = read_json(package / "parent-split.json")
    inputs = read_json(package / "runtime-inputs.json")
    source_root = Path(inputs["source_run"]["windows_root"])
    tape_path = source_root / inputs["source_run"]["train_tape_file"]
    require(sha(tape_path.read_bytes()) == inputs["source_run"]["train_tape_sha256"] == split["source_train_tapes_sha256"], "source_tape_identity")
    tape = read_json(tape_path)
    frozen_parents = {p["parent"]: p for p in split["parents"]}
    for window in rows:
        parent = frozen_parents[window["parent"]]
        source = [r for r in tape if r["parent"] == window["parent"] and r["exogenous_key"] == window["source_exogenous_key"]]
        require(len(source) == 1, "source_parent_not_unique")
        require(canonical(source[0]["scenario"]) == parent["scenario_sha256"] == window["scenario_sha256"], "source_scenario_identity")
        require(source[0]["structural_sha256"] == parent["structural_sha256"], "source_structure_identity")
        require(window["runtime_exogenous_key"] == window["source_exogenous_key"] == parent["exogenous_key"], "exogenous_identity")
        require(window["repeat"] == 0, "frozen_repeat")
    construction = [json.loads(s) for s in (root / "run-once/environment-construction.jsonl").read_text(encoding="utf-8").splitlines()]
    require(Counter(r["event"] for r in construction) == {"attempted": 8, "constructed": 8}, "environment_construction_counts")
    require({r["parent"] for r in construction} == set(frozen_parents), "environment_construction_parent_identity")
    result["provenance"] = {"source_train_tape_sha256": sha(tape_path.read_bytes()), "parent_identity_verified": True,
                            "scenario_content_and_structural_identity_verified": True,
                            "environment_construction_events": dict(Counter(r["event"] for r in construction)),
                            "evaluation_type": "production_W1_M10_simulation_not_real_flight_or_synthetic_model_ground_truth",
                            "formal_entry_integration_test_flag": read_json(package / "launch-contract.json")["integration_test"]}
    expected_config = read_json(package / "environment-config-contract.json")
    runtime = read_json(root / "run-once/environment.json")
    require(runtime["config"] == expected_config["config"], "actual_config_mismatch")
    require(canonical(runtime["config"]) == runtime["config_sha256"] == expected_config["config_sha256"], "config_digest")
    frozen_hashes = read_json(package / "hashes.json")["files"]
    result["frozen_content_mismatches"] = {where: [name for name, digest in frozen_hashes.items() if records.get(name) != digest] for where, records in before.items()}
    result["export"] = verify_export(root)
    result["ledger"] = ledger_summary(root)
    result["resource_settlement"] = read_json(root / "run-once/resource-settlement.json")
    require(result["ledger"]["totals"] == result["resource_settlement"]["ledger"]["totals"], "ledger_settlement_mismatch")
    require(result["ledger"]["statuses"] == {"complete": result["ledger"]["call_count"]}, "ledger_pending_or_failed")
    request = read_json(package / "RESOURCE_REQUEST.json")
    result["dynamic_budget_checks"] = {name: {"actual": amount, "limit": request["totals"][name], "within_limit": amount <= request["totals"][name]} for name, amount in result["ledger"]["totals"].items()}
    result["stage_dynamic_budget_checks"] = {stage: {name: {"actual": amount, "limit": request["stages"][stage][name], "within_limit": amount <= request["stages"][stage][name]} for name, amount in amounts.items()} for stage, amounts in result["ledger"]["stages"].items()}
    result["infrastructure_settlement"] = read_json(root / "infrastructure-settlement.json")
    result["supervisor_status"] = read_json(root / "supervisor-status.json")
    result["source_data_sha256"] = DATA_SHA
    result["source_data_bytes"] = path.stat().st_size
    result["environment_config_sha256"] = runtime["config_sha256"]
    result["independent_confirmation_parent_count"] = 0
    result["validation_scope"] = "independent_persisted_evidence_checks_not_a_reexecution_of_collector_or_environment"
    result["limits"] = ["Initial lifecycle snapshot is not separately persisted; do not fabricate it to rerun the full production derivation.", "Prefix individual observations are not persisted; source establishes history semantics but exact prefix history cannot be reconstructed.", "Complete branch reward/counts/energy are absent; one-step true_utility is not terminal sequence utility.", "Command and final execution identities do not by themselves establish direct physical completion."]
    after = {"package": snapshot(package), "attempt": snapshot(root)}
    require(after == before, "source_changed_during_offline_analysis")
    result["source_identity_unchanged"] = True
    result["offline_calls"] = {"environment_constructions": 0, "environment_steps": 0, "model_initializations": 0, "model_forwards": 0, "checkpoint_reads_writes": 0, "optimizer_updates": 0, "formal_attempts": 0}
    result["offline_cost"] = {"process_cpu_seconds": time.process_time() - started_cpu, "wall_seconds": time.perf_counter() - started_wall, "scope": "analysis_process_only_before_output_serialization_not_whole_previous_turn"}
    write_json(output / "analysis.json", result)
    write_json(output / "source-identities-before.json", before)
    write_json(output / "source-identities-after.json", after)
    write_json(output / "attribution-erratum.json", {"source_data_sha256": DATA_SHA, "original_data_changed": False, "primary_labels_changed": False, "derivation": "For the listed expired rows only, first-command physical completion and identifiability are false because no completion occurred; do not treat the final execution token as completion evidence.", "rows": result["invalid_attributions"]})
    with (output / "candidate-table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    print(json.dumps({k: result[k] for k in ("coverage_pass_original_rule", "parent_count", "candidate_count", "label_counts", "same_task_groups_with_physical_difference", "parents_with_same_task_physical_difference", "invalid_attribution_count", "frozen_content_mismatches", "export", "source_identity_unchanged", "offline_calls")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
