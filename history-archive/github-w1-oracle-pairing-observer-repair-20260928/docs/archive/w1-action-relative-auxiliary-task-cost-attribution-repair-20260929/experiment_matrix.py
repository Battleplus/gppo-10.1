"""Freeze parent groups and derive every execution ceiling from the matrix."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


HORIZON = 18
MAX_BRANCHES = 25
QUALIFYING_WINDOWS_PER_UNIT = 1
TRAIN_PARENTS = 24
SELECTION_PARENTS = 8
PREDICTION_EVALUATION_PARENTS = 8
TASK_EVALUATION_PARENTS = 8
DATA_REPEATS = {"train": 2, "model_selection": 2, "prediction_evaluation": 3}
TASK_REPEATS = 3
TASK_METHODS = ("hungarian", "transparent_one_shot", "A_one_shot", "B_one_shot")
TRAINING_SEEDS = (7101, 7102, 7103)
BATCH_SIZE = 64
MAX_EPOCHS = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_matrix(tape_path: Path) -> dict[str, Any]:
    rows = json.loads(tape_path.read_text(encoding="utf-8"))
    needed = TRAIN_PARENTS + SELECTION_PARENTS + PREDICTION_EVALUATION_PARENTS + TASK_EVALUATION_PARENTS
    if len(rows) < needed:
        raise ValueError("frozen training tape is too short")
    selected = rows[:needed]
    parents = [str(row["parent"]) for row in selected]
    if len(set(parents)) != needed:
        raise ValueError("selected tape prefix contains duplicate parents")
    if any(row.get("condition") != "W1" or row["scenario"]["communication"]["name"] != "W1-light" for row in selected):
        raise ValueError("selected parent is outside repaired W1-light contract")

    cursor = 0
    groups: dict[str, list[dict[str, Any]]] = {}
    definitions = (
        ("train", TRAIN_PARENTS),
        ("model_selection", SELECTION_PARENTS),
        ("prediction_evaluation", PREDICTION_EVALUATION_PARENTS),
        ("task_evaluation", TASK_EVALUATION_PARENTS),
    )
    for role, count in definitions:
        group = []
        for row in selected[cursor:cursor + count]:
            group.append({
                "parent": row["parent"],
                "tape_index": int(row["tape_index"]),
                "scenario_sha256": row["scenario_sha256"],
                "structural_sha256": row["structural_sha256"],
                "generator_split": row["generator_split"],
            })
        groups[role] = group
        cursor += count
    return {
        "schema": "w1-action-outcome-learning-matrix/1.0.0",
        "source_tape": str(tape_path),
        "source_tape_sha256": sha256(tape_path),
        "selection_rule": "first 48 unique rows of the already frozen randomized training tape; chosen before outcomes",
        "groups": groups,
        "data_repeats": DATA_REPEATS,
        "task_repeats": TASK_REPEATS,
        "training_seeds": list(TRAINING_SEEDS),
        "decision_rule": {
            "minimum_decision_step": 4,
            "minimum_legal_non_noop_candidates": 2,
            "require_nonidentical_transparent_scores": True,
            "maximum_windows_per_parent_repeat": QUALIFYING_WINDOWS_PER_UNIT,
            "missing_opportunity": "preserve; do not replace",
        },
        "continuation_id": "hungarian-v1-fixed",
        "task_methods": list(TASK_METHODS),
    }


def derive_budget(matrix: dict[str, Any]) -> dict[str, Any]:
    groups = matrix["groups"]
    data_units = sum(len(groups[role]) * DATA_REPEATS[role] for role in DATA_REPEATS)
    max_windows = data_units * QUALIFYING_WINDOWS_PER_UNIT
    max_branches = max_windows * MAX_BRANCHES
    data_environment_steps = data_units * (HORIZON + QUALIFYING_WINDOWS_PER_UNIT * MAX_BRANCHES * HORIZON)
    data_rule_decisions = data_units * (HORIZON + QUALIFYING_WINDOWS_PER_UNIT * MAX_BRANCHES * (HORIZON - 1))
    task_units = len(groups["task_evaluation"]) * TASK_REPEATS
    task_episodes = task_units * len(TASK_METHODS)
    task_environment_steps = task_episodes * HORIZON
    task_rule_decisions = task_environment_steps

    max_train_records = len(groups["train"]) * DATA_REPEATS["train"] * MAX_BRANCHES
    max_selection_records = len(groups["model_selection"]) * DATA_REPEATS["model_selection"] * MAX_BRANCHES
    max_prediction_records = len(groups["prediction_evaluation"]) * DATA_REPEATS["prediction_evaluation"] * MAX_BRANCHES
    updates_per_seed = math.ceil(max_train_records / BATCH_SIZE) * MAX_EPOCHS
    selection_batches_per_seed = math.ceil(max_selection_records / BATCH_SIZE) * MAX_EPOCHS
    prediction_batches_per_seed = math.ceil(max_prediction_records / BATCH_SIZE)
    task_learned_batch_calls = task_units * len(TRAINING_SEEDS)
    model_batch_forwards = (
        updates_per_seed * len(TRAINING_SEEDS)
        + selection_batches_per_seed * len(TRAINING_SEEDS)
        + prediction_batches_per_seed * len(TRAINING_SEEDS)
        + task_learned_batch_calls
    )
    model_sample_forwards = (
        max_train_records * MAX_EPOCHS * len(TRAINING_SEEDS)
        + max_selection_records * MAX_EPOCHS * len(TRAINING_SEEDS)
        + max_prediction_records * len(TRAINING_SEEDS)
        + task_units * MAX_BRANCHES * len(TRAINING_SEEDS)
    )
    return {
        "matrix_counts": {
            "data_parent_repeat_units": data_units,
            "maximum_label_windows": max_windows,
            "maximum_label_branches": max_branches,
            "task_parent_repeat_units": task_units,
            "task_episodes": task_episodes,
            "maximum_train_records": max_train_records,
            "maximum_model_selection_records": max_selection_records,
            "maximum_prediction_evaluation_records": max_prediction_records,
        },
        "active_call_totals": {
            "environment_steps": data_environment_steps + task_environment_steps,
            "resets_upper": data_units + task_episodes,
            "candidate_scans": data_units * HORIZON + task_episodes * HORIZON,
            "public_rule_decisions": data_rule_decisions + task_rule_decisions,
            "snapshot_captures": max_windows,
            "branch_snapshot_copies": max_branches,
            "branches": max_branches,
            "forced_first_actions": max_branches,
            "model_initializations_or_loads": 6,
            "model_batch_forwards": model_batch_forwards,
            "model_sample_evaluations": model_sample_forwards,
            "offline_predictor_updates": updates_per_seed * len(TRAINING_SEEDS),
            "checkpoint_writes": 3,
        },
        "stage_dynamic_counts": {
            "label_collection": {
                "environment_steps": data_environment_steps,
                "resets_upper": data_units,
                "candidate_scans": data_units * HORIZON,
                "public_rule_decisions": data_rule_decisions,
                "snapshot_captures": max_windows,
                "branch_snapshot_copies": max_branches,
                "branches": max_branches,
                "forced_first_actions": max_branches,
                "model_initializations_or_loads": 0,
                "model_batch_forwards": 0,
                "offline_predictor_updates": 0,
            },
            "supervised_training_and_selection": {
                "environment_steps": 0,
                "resets_upper": 0,
                "model_initializations_or_loads": 3,
                "model_batch_forwards": (
                    updates_per_seed * len(TRAINING_SEEDS)
                    + selection_batches_per_seed * len(TRAINING_SEEDS)
                ),
                "model_sample_evaluations": (
                    max_train_records * MAX_EPOCHS * len(TRAINING_SEEDS)
                    + max_selection_records * MAX_EPOCHS * len(TRAINING_SEEDS)
                ),
                "offline_predictor_updates": updates_per_seed * len(TRAINING_SEEDS),
                "checkpoint_writes": 3,
            },
            "independent_prediction_evaluation": {
                "environment_steps": 0,
                "resets_upper": 0,
                "model_initializations_or_loads": 3,
                "model_batch_forwards": prediction_batches_per_seed * len(TRAINING_SEEDS),
                "model_sample_evaluations": max_prediction_records * len(TRAINING_SEEDS),
                "offline_predictor_updates": 0,
            },
            "conditional_task_comparison": {
                "environment_steps": task_environment_steps,
                "resets_upper": task_episodes,
                "candidate_scans": task_episodes * HORIZON,
                "public_rule_decisions": task_rule_decisions,
                "model_initializations_or_loads": 0,
                "model_batch_forwards": task_learned_batch_calls,
                "model_sample_evaluations": task_units * MAX_BRANCHES * len(TRAINING_SEEDS),
                "offline_predictor_updates": 0,
            },
        },
    }


def main() -> int:
    root = Path(__file__).resolve().parent
    tape = Path(r"E:\Z博士\research-plans\w1-light-repaired-fair-rerun-infra-repair-v1\tapes-train.json")
    matrix = build_matrix(tape)
    budget = derive_budget(matrix)
    (root / "experiment-matrix.json").write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "derived-budget.json").write_text(
        json.dumps(budget, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
