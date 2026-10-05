"""Read-only post-run settlement for the sealed finite learning attempt."""

from __future__ import annotations

import collections
import hashlib
import json
import math
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN = Path(r"E:\Z博士\runs\w1-action-outcome-learning-loop-history-repair-v1-once")
ATTEMPT = "w1-action-outcome-learning-loop-history-repair-v1-once"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def export_check() -> dict:
    manifest_path = RUN / "export-manifest.json"
    complete = read_json(RUN / "EXPORT_COMPLETE.json")
    manifest = read_json(manifest_path)
    mismatches = []
    for relative, expected in manifest["files"].items():
        path = RUN / relative
        actual = {
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256(path) if path.is_file() else None,
        }
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    manifest_hash = sha256(manifest_path)
    return {
        "manifest_entries": len(manifest["files"]),
        "manifest_sha256": manifest_hash,
        "completion_manifest_sha256": complete["export_manifest_sha256"],
        "completion_file_count": complete["file_count"],
        "completion_payload_bytes": complete["payload_bytes"],
        "completion_verified": complete["verified"],
        "entry_mismatches": mismatches,
        "pass": (
            not mismatches
            and manifest_hash == complete["export_manifest_sha256"]
            and bool(complete["verified"])
        ),
    }


def unit_summary() -> dict:
    status = collections.Counter()
    by_role = collections.Counter()
    first_acceptance = collections.Counter()
    first_feedback = collections.Counter()
    task_status = collections.Counter()
    task_physical = collections.Counter()
    task_host = collections.Counter()
    parents_with_labels: dict[str, set[str]] = collections.defaultdict(set)
    branches = 0
    units = 0
    pairing_failures = 0
    input_digest_mismatches = 0
    terminated = 0
    truncated = 0
    path = RUN / "run-once" / "data-units.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            units += 1
            role = str(row["split_role"])
            unit_status = str(row["status"])
            status[unit_status] += 1
            by_role[f"{role}:{unit_status}"] += 1
            summaries = row.get("explanation_labels", [])
            if summaries:
                parents_with_labels[role].add(str(row["parent"]))
            pairing = row.get("pairing")
            if pairing is not None and any(
                int(pairing.get(name, 0)) != 0
                for name in (
                    "common_result_mismatches",
                    "parameter_conflict_count",
                    "multiplicity_mismatch_count",
                )
            ):
                pairing_failures += 1
            for summary in summaries:
                branches += 1
                first_acceptance[str(summary.get("first_acceptance_class"))] += 1
                first_feedback[str(summary.get("first_feedback"))] += 1
                terminated += int(bool(summary.get("terminated")))
                truncated += int(bool(summary.get("truncated")))
                if summary.get("decision_input_sha256") != row.get("decision_input_sha256"):
                    input_digest_mismatches += 1
                for task in summary.get("tasks", {}).values():
                    task_status[str(task.get("status"))] += 1
                    task_physical[str(task.get("physical_arrival_before_deadline"))] += 1
                    task_host[str(task.get("host_confirmation_before_deadline"))] += 1
    return {
        "units": units,
        "unit_status": dict(sorted(status.items())),
        "unit_status_by_role": dict(sorted(by_role.items())),
        "parents_with_labels": {
            role: len(parents) for role, parents in sorted(parents_with_labels.items())
        },
        "branches": branches,
        "first_acceptance_class": dict(sorted(first_acceptance.items())),
        "first_feedback": dict(sorted(first_feedback.items())),
        "task_status": dict(sorted(task_status.items())),
        "task_physical_arrival_before_deadline": dict(sorted(task_physical.items())),
        "task_host_confirmation_before_deadline": dict(sorted(task_host.items())),
        "terminated_branches": terminated,
        "truncated_branches": truncated,
        "pairing_failures": pairing_failures,
        "decision_input_digest_mismatches": input_digest_mismatches,
    }


def ledger_summary(settlement: dict) -> dict:
    database = RUN / "run-once" / "budget.sqlite3"
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT id, stage, name, amounts, status FROM calls ORDER BY id"
    ).fetchall()
    connection.close()
    call_status = collections.Counter(row[4] for row in rows)
    calls_by_stage = collections.Counter(row[1] for row in rows)
    counters_by_stage: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for _identifier, stage, _name, amounts, _status in rows:
        counters_by_stage[stage].update(json.loads(amounts))

    training_rows = [row for row in rows if row[1] == "supervised_training_and_selection"]
    seeds = [7101, 7102, 7103]
    per_seed = []
    updates = forwards = samples = initializations = 0
    for _identifier, _stage, name, amounts, _status in training_rows:
        values = json.loads(amounts)
        updates += int(values.get("offline_predictor_updates", 0))
        forwards += int(values.get("model_batch_forwards", 0))
        samples += int(values.get("model_sample_evaluations", 0))
        initializations += int(values.get("model_initializations_or_loads", 0))
        if name == "checkpoint.write":
            seed = seeds[len(per_seed)]
            per_seed.append({
                "seed": seed,
                "optimizer_updates": updates,
                "batch_forwards_including_selection": forwards,
                "sample_evaluations_including_selection": samples,
                "model_initializations": initializations,
                "checkpoint": f"run-once/checkpoints/seed-{seed}.pt",
                "checkpoint_sha256": sha256(
                    RUN / "run-once" / "checkpoints" / f"seed-{seed}.pt"
                ),
            })
            updates = forwards = samples = initializations = 0

    return {
        "database_sha256": sha256(database),
        "calls": len(rows),
        "call_status": dict(sorted(call_status.items())),
        "calls_by_stage": dict(sorted(calls_by_stage.items())),
        "counters_by_stage": {
            stage: dict(sorted(values.items()))
            for stage, values in sorted(counters_by_stage.items())
        },
        "settled_totals": settlement["ledger"]["totals"],
        "pending_calls": settlement["ledger"]["pending_calls"],
        "failed_calls": settlement["ledger"]["failed_calls"],
        "training_by_seed": per_seed,
    }


def result_summary(data_gate: dict, prediction: dict, settlement: dict, units: dict) -> dict:
    comparison = prediction["comparison"]
    learned = comparison["learned_three_seed_mean"]
    current = comparison["current_public"]
    transparent = comparison["transparent_history"]

    def relative_reduction(value: float, baseline: float) -> float:
        return (baseline - value) / baseline

    return {
        "attempt": ATTEMPT,
        "decision": "PREDICTION_GATE_NOT_PASSED",
        "technical_stop": False,
        "data_gate": data_gate,
        "label_audit": units,
        "training": {
            "completed_seeds": [7101, 7102, 7103],
            "checkpoint_writes": settlement["ledger"]["totals"]["checkpoint_writes"],
            "optimizer_updates": settlement["ledger"]["totals"]["offline_predictor_updates"],
        },
        "prediction_evaluation": prediction,
        "derived_prediction_differences": {
            "mae_reduction_vs_current_public": current["parent_macro_mae"] - learned["parent_macro_mae"],
            "mae_relative_reduction_vs_current_public": relative_reduction(
                learned["parent_macro_mae"], current["parent_macro_mae"]
            ),
            "mae_reduction_vs_transparent_history": transparent["parent_macro_mae"] - learned["parent_macro_mae"],
            "mae_relative_reduction_vs_transparent_history": relative_reduction(
                learned["parent_macro_mae"], transparent["parent_macro_mae"]
            ),
            "selected_regret_change_vs_current_public": (
                learned["parent_macro_selected_regret"] - current["parent_macro_selected_regret"]
            ),
            "selected_regret_change_vs_transparent_history": (
                learned["parent_macro_selected_regret"]
                - transparent["parent_macro_selected_regret"]
            ),
            "top1_accuracy_change_vs_current_public": (
                learned["parent_macro_top1_accuracy"] - current["parent_macro_top1_accuracy"]
            ),
            "top1_accuracy_change_vs_transparent_history": (
                learned["parent_macro_top1_accuracy"]
                - transparent["parent_macro_top1_accuracy"]
            ),
        },
        "task_comparison": {
            "status": "NOT_EVALUATED",
            "reason": "frozen prediction gate did not pass",
            "utility_delta_vs_hungarian": None,
            "utility_delta_vs_transparent": None,
            "parent_coverage": None,
        },
        "decision_cost": {
            "status": "NOT_EVALUATED",
            "reason": "conditional task comparison and its full decision-cost instrumentation did not run",
            "cpu_mean_seconds": None,
            "wall_p95_seconds": None,
            "cpu_10ms_pass": None,
            "wall_50ms_pass": None,
        },
    }


def resource_summary(request: dict, supervisor: dict, export: dict, complete: dict) -> dict:
    totals = request["totals"]
    native_bytes = int(supervisor["artifact_bytes"])
    export_bytes = int(complete["payload_bytes"])
    aggregate = native_bytes + export_bytes
    return {
        "attempt": ATTEMPT,
        "final_resource_pass": supervisor["final_resource_pass"],
        "persisted_combined_budget": {
            "wall_seconds": export["combined_budget_wall_seconds"],
            "wall_limit_seconds": totals["wall_seconds"],
            "complete_process_cpu_seconds": export["combined_budget_cpu_seconds"],
            "complete_process_cpu_limit_seconds": totals["complete_process_cpu_seconds"],
            "peak_rss_upper_bound_bytes": supervisor["peak_rss_upper_bound_bytes"],
            "rss_limit_bytes": totals["all_resident_rss_bytes"],
            "native_artifact_bytes": native_bytes,
            "active_artifact_limit_bytes": totals["artifact_bytes"],
            "verified_export_payload_bytes": export_bytes,
            "aggregate_native_plus_export_bytes": aggregate,
            "aggregate_limit_bytes": totals["aggregate_native_plus_verified_export_bytes"],
        },
        "dynamic_usage": read_json(RUN / "run-once" / "settlement.json")["ledger"]["totals"],
        "dynamic_limits": {
            key: totals[key]
            for key in read_json(RUN / "run-once" / "settlement.json")["ledger"]["totals"]
        },
        "controlled_export": {
            "status": export["status"],
            "verified": complete["verified"],
            "file_count": complete["file_count"],
            "export_manifest_sha256": complete["export_manifest_sha256"],
        },
        "outer_windows_launcher_wall_seconds_terminal_only": 250.76809070000036,
        "outer_windows_launcher_wall_persisted": False,
    }


def report(result: dict, resources: dict, ledger: dict, export: dict) -> str:
    compare = result["prediction_evaluation"]["comparison"]
    learned = compare["learned_three_seed_mean"]
    current = compare["current_public"]
    transparent = compare["transparent_history"]
    delta = result["derived_prediction_differences"]
    data = result["data_gate"]
    usage = resources["dynamic_usage"]
    seed_rows = ", ".join(
        f"{row['seed']}: {row['optimizer_updates']} updates"
        for row in ledger["training_by_seed"]
    )
    return f"""# W1 action-outcome learning-loop final report

## Decision

**Prediction gate not passed.** This was an expected gated research stop, not a
technical stop. Label collection, coverage admission, three-seed training, and
independent prediction evaluation completed. Conditional task comparison did
not run.

## Label coverage and training

The data gate passed with parent coverage train/model-selection/prediction of
{data['parent_coverage']['train']}/{data['parent_coverage']['model_selection']}/{data['parent_coverage']['prediction_evaluation']}
against required coverage {data['minimum_parent_coverage']['train']}/{data['minimum_parent_coverage']['model_selection']}/{data['minimum_parent_coverage']['prediction_evaluation']}.
The frozen splits contain {data['records']['train']} training, {data['records']['model_selection']}
model-selection, and {data['records']['prediction_evaluation']} prediction-evaluation
candidate labels. All {result['label_audit']['branches']} persisted branch input
digests match their decision snapshots, and no communication-pairing failure was
recorded. The branch explanations retain
{result['label_audit']['task_status'].get('completed', 0)} completed,
{result['label_audit']['task_status'].get('expired', 0)} expired, and
{result['label_audit']['task_status'].get('unknown', 0)} unknown task statuses.
Host-confirmation labels retain
{result['label_audit']['task_host_confirmation_before_deadline'].get('True', 0)} true,
{result['label_audit']['task_host_confirmation_before_deadline'].get('False', 0)} false,
and {result['label_audit']['task_host_confirmation_before_deadline'].get('None', 0)}
unknown values rather than imputing them. First-action evidence also retains
{result['label_audit']['first_acceptance_class'].get('command_transport_lost', 0)}
transport losses and {result['label_audit']['first_acceptance_class'].get('unknown', 0)}
unknown acceptance classes.

All three seeds produced checkpoints. Training used {usage['offline_predictor_updates']}
optimizer updates in total ({seed_rows}); early stopping remained frozen.

## Independent prediction evaluation

Across {learned['parents']} prediction parents, {learned['decisions']} decisions,
and {learned['records']} candidates:

- learned MAE: {learned['parent_macro_mae']:.9f}; current-public MAE:
  {current['parent_macro_mae']:.9f}; transparent-history MAE:
  {transparent['parent_macro_mae']:.9f};
- learned MAE is lower by {delta['mae_reduction_vs_current_public']:.9f}
  ({delta['mae_relative_reduction_vs_current_public']:.2%}) versus current-public
  and by {delta['mae_reduction_vs_transparent_history']:.9f}
  ({delta['mae_relative_reduction_vs_transparent_history']:.2%}) versus transparent-history;
- learned selected-action regret is {learned['parent_macro_selected_regret']:.9f},
  versus {current['parent_macro_selected_regret']:.9f} for current-public and
  {transparent['parent_macro_selected_regret']:.9f} for transparent-history;
- learned top-1 accuracy is {learned['parent_macro_top1_accuracy']:.3f}, versus
  {current['parent_macro_top1_accuracy']:.3f} for both baselines.

Thus both frozen MAE checks passed, but the selected-action-regret check failed:
regret increased by {delta['selected_regret_change_vs_transparent_history']:.9f}
relative to the transparent baseline. Better scalar prediction error did not
improve candidate choice.

## Downstream stages

Task utility relative to Hungarian and transparent one-shot selection is **not
evaluated** because the prediction gate failed. The 10 ms CPU-mean and 50 ms
wall-p95 full decision-cost standards are also **not evaluated** because the
conditional task stage and its cost instrumentation did not run.

No result here establishes GPPO plus world-model value. The target and use both
remain one action followed by frozen Hungarian continuation.

## Settlement

The ledger settled with zero pending and zero failed calls. Dynamic use was
{usage['environment_steps']} environment steps, {usage['resets_upper']} resets,
{usage['branches']} branches, and {usage['public_rule_decisions']} public-rule
decisions. Persisted combined wall/CPU were
{resources['persisted_combined_budget']['wall_seconds']:.3f}/
{resources['persisted_combined_budget']['complete_process_cpu_seconds']:.3f}
seconds. Peak RSS upper bound was
{resources['persisted_combined_budget']['peak_rss_upper_bound_bytes']} bytes.
All were below frozen limits.

The controlled export is verified. Its final manifest covers
{resources['controlled_export']['file_count']} files and has SHA-256
`{resources['controlled_export']['export_manifest_sha256']}`. Independent
post-run verification found {len(export['entry_mismatches'])} file mismatches.
The Windows console additionally reported an outer wall interval of
250.7680907 seconds; the launcher explicitly marks that number as terminal-only
and unavailable in native artifacts.
"""


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    request = read_json(RUN / "RESOURCE_REQUEST.json")
    final = read_json(RUN / "FINAL_SETTLEMENT.json")
    status = read_json(RUN / "run-once" / "status.json")
    data_gate = read_json(RUN / "run-once" / "data-gate.json")
    prediction = read_json(RUN / "run-once" / "prediction-evaluation.json")
    settlement = read_json(RUN / "run-once" / "settlement.json")
    supervisor = read_json(RUN / "supervisor-status.json")
    export_status = read_json(RUN / "export-status.json")
    complete = read_json(RUN / "EXPORT_COMPLETE.json")
    exported = export_check()
    units = unit_summary()
    ledger = ledger_summary(settlement)
    result = result_summary(data_gate, prediction, settlement, units)
    resources = resource_summary(request, supervisor, export_status, complete)
    global_dynamic_within_limits = all(
        int(value) <= int(request["totals"][name])
        for name, value in ledger["settled_totals"].items()
    )
    stage_counters_within_limits = all(
        int(value) <= int(request["stage_limits"][stage][name])
        for stage, counters in ledger["counters_by_stage"].items()
        for name, value in counters.items()
    )
    checks = {
        "attempt_identity": status["attempt"] == ATTEMPT == final["attempt"],
        "final_overall_complete": final["overall_status"] == "complete",
        "worker_prediction_gate_stop": status["status"] == "prediction_gate_stop",
        "not_technical_stop": "technical_stop" not in status["status"],
        "data_gate_pass": bool(data_gate["pass"]),
        "prediction_gate_failed": not bool(prediction["gate"]["pass"]),
        "task_comparison_not_executed": not bool(status["task_comparison_executed"]),
        "ledger_settled": ledger["pending_calls"] == 0 and ledger["failed_calls"] == 0,
        "input_digests_match": units["decision_input_digest_mismatches"] == 0,
        "pairing_pass": units["pairing_failures"] == 0,
        "resource_pass": bool(supervisor["final_resource_pass"]),
        "global_dynamic_counters_within_limits": global_dynamic_within_limits,
        "stage_dynamic_counters_within_limits": stage_counters_within_limits,
        "verified_export": bool(exported["pass"]),
        "no_retry": not bool(final["automatic_retry"]) and not bool(status["automatic_retry"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"post-run integrity check failed: {checks}")

    write_json(ROOT / "result-summary.json", result)
    write_json(ROOT / "ledger-summary.json", ledger)
    write_json(ROOT / "resource-settlement.json", resources)
    write_json(ROOT / "checks.json", checks)
    write_json(ROOT / "input-index.json", {
        "attempt": ATTEMPT,
        "sealed_attempt_root": str(RUN),
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (
                RUN / "FINAL_SETTLEMENT.json",
                RUN / "EXPORT_COMPLETE.json",
                RUN / "export-manifest.json",
                RUN / "export-status.json",
                RUN / "supervisor-status.json",
                RUN / "run-once" / "status.json",
                RUN / "run-once" / "data-gate.json",
                RUN / "run-once" / "prediction-evaluation.json",
                RUN / "run-once" / "settlement.json",
                RUN / "run-once" / "budget.sqlite3",
                RUN / "run-once" / "data-units.jsonl",
                RUN / "run-once" / "learning-records.jsonl",
            )
        ],
        "postrun_calls": {
            "environment": 0,
            "resets": 0,
            "model_initializations_or_loads": 0,
            "model_forwards": 0,
            "checkpoint_loads": 0,
            "training_updates": 0,
        },
    })
    (ROOT / "report.md").write_text(
        report(result, resources, ledger, exported), encoding="utf-8"
    )
    output_names = (
        "analyze_postrun.py",
        "checks.json",
        "input-index.json",
        "ledger-summary.json",
        "report.md",
        "resource-settlement.json",
        "result-summary.json",
    )
    write_json(ROOT / "hashes.json", {
        "schema": "w1-action-outcome-learning-loop-postrun-hashes/1.0.0",
        "files": [
            {"path": name, "bytes": (ROOT / name).stat().st_size, "sha256": sha256(ROOT / name)}
            for name in output_names
        ],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
