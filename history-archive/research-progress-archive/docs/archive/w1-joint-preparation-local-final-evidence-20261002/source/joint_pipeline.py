"""Collection admission followed by six frozen world-model routes; no GPPO."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from infra_io import durable_atomic_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def data_admission(records, matrix, contract):
    from sequence_data_contract import validate_sequence_window
    expected = {(role, row["parent"], repeat) for role, parents in matrix["splits"].items()
                for row in parents for repeat in range(matrix["repeats"][role])}
    found, complete, failures = set(), [], []
    classes = {"physical_positive": set(), "expiry_positive": set(), "host_valid": set()}
    for row in records:
        key = (row.get("split"), row.get("parent"), row.get("repeat"))
        if key not in expected or key in found:
            raise RuntimeError("UNREGISTERED_OR_DUPLICATE_WINDOW:" + str(key))
        found.add(key)
        if row.get("status") != "complete":
            failures.append({"unit": key, "reason": row.get("reason", "no_opportunity")})
            continue
        validate_sequence_window(row)
        labels = row["task_outcome_target"]
        valid = sum(bool(label["physical_on_time_completion"]["valid"])
                    or bool(label["task_expired"]["valid"]) for label in labels)
        if valid < contract["physical_and_expiry_valid_min_per_window"]:
            failures.append({"unit": key, "reason": "insufficient_valid_physical_outcomes"})
        complete.append(row)
        if row["split"] == "train":
            for label in labels:
                for field, bucket in (("physical_on_time_completion", "physical_positive"),
                                      ("task_expired", "expiry_positive")):
                    if label[field]["valid"] and label[field]["value"] is True:
                        classes[bucket].add(row["parent"])
                if label["host_confirmation"]["valid"]:
                    classes["host_valid"].add(row["parent"])
    for key in sorted(expected - found):
        failures.append({"unit": key, "reason": "missing_registered_window"})
    for bucket, threshold in (("physical_positive", "train_physical_positive_min_parents"),
                              ("expiry_positive", "train_expiry_positive_min_parents"),
                              ("host_valid", "train_host_valid_min_parents")):
        if len(classes[bucket]) < contract[threshold]:
            failures.append({"reason": "training_target_support:" + bucket,
                             "actual": len(classes[bucket]), "required": contract[threshold]})
    return {"status": "data_gate_pass" if not failures else "data_gate_stop",
            "coverage_pass": not failures, "expected_window_count": len(expected),
            "complete_windows": len(complete), "failures": failures,
            "training_target_parent_support": {key: sorted(value) for key, value in classes.items()},
            "gppo_calls": 0, "task_comparison_calls": 0}


def run_pipeline(root: Path, output: Path, matrix: dict, ledger: Any, boundary: Any):
    from production_data import ProductionDataCollector
    ledger.select("label_qualification")
    durable_atomic_json(output / "activity.json", {"stage": "label_qualification", "run": "complete_sequence_collection"})
    collected = ProductionDataCollector(root, output, matrix, ledger, boundary).collect()
    rows = [json.loads(line) for line in Path(collected["path"]).read_text(encoding="utf-8").splitlines()]
    admitted = data_admission(rows, matrix, read(root / "data-gate.json"))
    durable_atomic_json(output / "data-admission.json", admitted)
    if not admitted["coverage_pass"]:
        return {"status": "data_gate_stop", "data": admitted,
                "world_model_training": "not_evaluated", "prediction": "not_evaluated",
                "gppo_task_utility": "not_evaluated", "cost": "not_evaluated"}
    # Import the learning backend only after the pure data admission has passed.
    from production_world import train_select_world_models, evaluate_world_models, recompute_prediction_metrics
    durable_atomic_json(output / "activity.json", {"stage": "world_model_training_and_selection", "run": "G1_G2_six_routes"})
    training = train_select_world_models(
        [row for row in rows if row["split"] == "train"],
        [row for row in rows if row["split"] == "model_selection"],
        matrix, output, ledger, boundary)
    durable_atomic_json(output / "activity.json", {"stage": "prediction_confirmation", "run": "checkpoint_restore_and_candidate_prediction"})
    prediction = evaluate_world_models(
        [row for row in rows if row["split"] == "prediction_confirmation"],
        training.routes, matrix, output, ledger, boundary)
    recalculated = recompute_prediction_metrics(prediction.trace_path,
        expected_parent_count=matrix["world_model_confirmation_windows"],
        expected_seeds=matrix["world_model_seeds"])
    if prediction.metrics != recalculated:
        raise RuntimeError("PERSISTED_TRACE_METRIC_RECOMPUTATION_MISMATCH")
    return {"status": "prediction_evaluation_complete", "data": admitted,
            "training_routes": training.routes, "prediction": recalculated,
            "gppo_task_utility": "not_evaluated", "practical_cost": "not_evaluated",
            "gppo_calls": 0, "task_comparison_calls": 0,
            "research_success": False}
