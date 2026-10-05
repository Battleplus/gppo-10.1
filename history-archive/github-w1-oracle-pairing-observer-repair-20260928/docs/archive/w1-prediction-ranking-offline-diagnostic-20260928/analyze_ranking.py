"""Offline diagnostic using only sealed labels, saved baselines, and aggregates."""

from __future__ import annotations

import collections
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN = Path(r"E:\Z博士\runs\w1-action-outcome-learning-loop-history-repair-v1-once")
RECORDS = RUN / "run-once" / "learning-records.jsonl"
UNITS = RUN / "run-once" / "data-units.jsonl"
AGGREGATE = RUN / "run-once" / "prediction-evaluation.json"
EVALUATION = "prediction_evaluation"
TOLERANCE = 1e-12


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


def load_records() -> list[dict]:
    rows = []
    with RECORDS.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append(row)
    return rows


def load_label_audit() -> dict[str, dict[int, dict]]:
    result: dict[str, dict[int, dict]] = {}
    with UNITS.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            unit = json.loads(line)
            if unit["split_role"] != EVALUATION or unit["status"] != "LABELED":
                continue
            decision_ids = {
                str(summary.get("identity", {}).get("decision_id", ""))
                for summary in unit["explanation_labels"]
            }
            if len(decision_ids) != 1 or "" in decision_ids:
                raise RuntimeError("unit branch identities do not provide one decision_id")
            decision_id = next(iter(decision_ids))
            by_action = {}
            for summary in unit["explanation_labels"]:
                action = int(summary["identity"]["action"])
                by_action[action] = {
                    "acceptance": summary["first_acceptance_class"],
                    "feedback": summary["first_feedback"],
                    "energy_used": float(summary["energy_used"]),
                    "discounted_utility": float(summary["discounted_utility"]),
                    "task_status_counts": dict(collections.Counter(
                        task["status"] for task in summary["tasks"].values()
                    )),
                    "host_unknown_count": sum(
                        task["host_confirmation_before_deadline"] is None
                        for task in summary["tasks"].values()
                    ),
                }
            result[decision_id] = by_action
    return result


def choose(rows: list[dict], score_key: str) -> dict:
    return max(rows, key=lambda row: (float(row[score_key]), -int(row["action"])))


def best_truth(rows: list[dict]) -> tuple[float, list[int]]:
    best = max(float(row["remaining_utility"]) for row in rows)
    actions = sorted(
        int(row["action"])
        for row in rows
        if abs(float(row["remaining_utility"]) - best) <= TOLERANCE
    )
    return best, actions


def centered_mae(rows: list[dict], score_key: str) -> float:
    truth_mean = sum(float(row["remaining_utility"]) for row in rows) / len(rows)
    score_mean = sum(float(row[score_key]) for row in rows) / len(rows)
    return sum(
        abs(
            (float(row[score_key]) - score_mean)
            - (float(row["remaining_utility"]) - truth_mean)
        )
        for row in rows
    ) / len(rows)


def absolute_mae(rows: list[dict], score_key: str) -> float:
    return sum(
        abs(float(row[score_key]) - float(row["remaining_utility"]))
        for row in rows
    ) / len(rows)


def pairwise(rows: list[dict], score_key: str) -> dict:
    absolute_errors = []
    directions = []
    informative = 0
    for left, right in itertools.combinations(rows, 2):
        truth = float(left["remaining_utility"]) - float(right["remaining_utility"])
        predicted = float(left[score_key]) - float(right[score_key])
        absolute_errors.append(abs(predicted - truth))
        if abs(truth) > TOLERANCE:
            informative += 1
            directions.append(float(truth * predicted > 0.0))
    return {
        "pairs": len(absolute_errors),
        "informative_pairs": informative,
        "difference_mae": sum(absolute_errors) / len(absolute_errors),
        "direction_accuracy": (
            sum(directions) / len(directions) if directions else None
        ),
    }


def macro(window_rows: list[dict], key: str) -> float | None:
    by_parent: dict[str, list[float]] = collections.defaultdict(list)
    for row in window_rows:
        value = row[key]
        if value not in ("", None, "NOT_SAVED"):
            by_parent[row["parent"]].append(float(value))
    if not by_parent:
        return None
    return sum(sum(values) / len(values) for values in by_parent.values()) / len(by_parent)


def reproduce_metrics(rows: list[dict], score_key: str) -> dict:
    by_decision: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_decision[row["decision_id"]].append(row)
    errors: dict[str, list[float]] = collections.defaultdict(list)
    squared: dict[str, list[float]] = collections.defaultdict(list)
    regrets: dict[str, list[float]] = collections.defaultdict(list)
    top1: dict[str, list[float]] = collections.defaultdict(list)
    for window in by_decision.values():
        parent = window[0]["parent"]
        for row in window:
            error = float(row[score_key]) - float(row["remaining_utility"])
            errors[parent].append(abs(error))
            squared[parent].append(error * error)
        selected = choose(window, score_key)
        best = max(float(row["remaining_utility"]) for row in window)
        regret = best - float(selected["remaining_utility"])
        regrets[parent].append(regret)
        top1[parent].append(float(abs(regret) <= TOLERANCE))

    def parent_macro(values: dict[str, list[float]]) -> float:
        return sum(sum(items) / len(items) for items in values.values()) / len(values)

    return {
        "records": len(rows),
        "decisions": len(by_decision),
        "parents": len(errors),
        "parent_macro_mae": parent_macro(errors),
        "parent_macro_rmse": sum(
            math.sqrt(sum(items) / len(items)) for items in squared.values()
        ) / len(squared),
        "parent_macro_selected_regret": parent_macro(regrets),
        "parent_macro_top1_accuracy": parent_macro(top1),
    }


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    all_records = load_records()
    records = [row for row in all_records if row["split_role"] == EVALUATION]
    audit = load_label_audit()
    aggregate = read_json(AGGREGATE)
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for row in records:
        grouped[row["decision_id"]].append(row)

    window_rows = []
    action_rows = []
    noop_windows = 0
    baseline_disagreements = 0
    true_near_ties = 0
    candidate_histogram = collections.Counter()
    unknown_cross = collections.Counter()
    for decision_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["action"]))
        if set(row["continuation_id"] for row in rows) != {"hungarian-v1-fixed"}:
            raise RuntimeError(f"mixed continuation in {decision_id}")
        legal = tuple(int(value) for value in rows[0]["legal_actions"])
        if set(legal) != {int(row["action"]) for row in rows}:
            raise RuntimeError(f"incomplete legal candidates in {decision_id}")
        if any(
            (row["parent"], row["repeat"], tuple(row["flat"]), tuple(row["history"]))
            != (rows[0]["parent"], rows[0]["repeat"], tuple(rows[0]["flat"]), tuple(rows[0]["history"]))
            for row in rows
        ):
            raise RuntimeError(f"window inputs differ in {decision_id}")
        if decision_id not in audit or set(audit[decision_id]) != set(legal):
            raise RuntimeError(f"label audit mismatch in {decision_id}")

        truth = [float(row["remaining_utility"]) for row in rows]
        truth_sorted = sorted(truth, reverse=True)
        truth_range = max(truth) - min(truth)
        best_value, best_actions = best_truth(rows)
        best_second_gap = truth_sorted[0] - truth_sorted[1]
        current = choose(rows, "current_public_score")
        transparent = choose(rows, "transparent_history_score")
        current_action = int(current["action"])
        transparent_action = int(transparent["action"])
        current_regret = best_value - float(current["remaining_utility"])
        transparent_regret = best_value - float(transparent["remaining_utility"])
        current_pair = pairwise(rows, "current_public_score")
        transparent_pair = pairwise(rows, "transparent_history_score")
        has_noop = 24 in legal
        noop_windows += int(has_noop)
        candidate_histogram[len(rows)] += 1
        different = current_action != transparent_action
        baseline_disagreements += int(different)
        true_near_ties += int(best_second_gap <= TOLERANCE)

        selected_truth_delta = (
            float(current["remaining_utility"])
            - float(transparent["remaining_utility"])
        )
        window_rows.append({
            "parent": rows[0]["parent"],
            "repeat": int(rows[0]["repeat"]),
            "decision_id": decision_id,
            "candidate_count": len(rows),
            "legal_actions": "|".join(map(str, legal)),
            "noop_legal": has_noop,
            "true_min": min(truth),
            "true_max": max(truth),
            "true_range": truth_range,
            "best_actions": "|".join(map(str, best_actions)),
            "best_second_gap": best_second_gap,
            "true_exact_best_tie": len(best_actions) > 1,
            "current_selected_action": current_action,
            "current_selected_acceptance": audit[decision_id][current_action]["acceptance"],
            "current_selected_true_utility": float(current["remaining_utility"]),
            "current_regret": current_regret,
            "current_top1": abs(current_regret) <= TOLERANCE,
            "current_score_span": max(float(row["current_public_score"]) for row in rows)
            - min(float(row["current_public_score"]) for row in rows),
            "current_absolute_mae": absolute_mae(rows, "current_public_score"),
            "current_centered_mae": centered_mae(rows, "current_public_score"),
            "current_pairwise_difference_mae": current_pair["difference_mae"],
            "current_pairwise_direction_accuracy": current_pair["direction_accuracy"],
            "transparent_selected_action": transparent_action,
            "transparent_selected_acceptance": audit[decision_id][transparent_action]["acceptance"],
            "transparent_selected_true_utility": float(transparent["remaining_utility"]),
            "transparent_regret": transparent_regret,
            "transparent_top1": abs(transparent_regret) <= TOLERANCE,
            "transparent_score_span": max(float(row["transparent_history_score"]) for row in rows)
            - min(float(row["transparent_history_score"]) for row in rows),
            "transparent_absolute_mae": absolute_mae(rows, "transparent_history_score"),
            "transparent_centered_mae": centered_mae(rows, "transparent_history_score"),
            "transparent_pairwise_difference_mae": transparent_pair["difference_mae"],
            "transparent_pairwise_direction_accuracy": transparent_pair["direction_accuracy"],
            "baseline_same_action": not different,
            "current_minus_transparent_selected_true_utility": selected_truth_delta,
            "learned_selected_action": "NOT_SAVED",
            "learned_selected_acceptance": "NOT_SAVED",
            "learned_regret": "NOT_SAVED",
            "learned_score_span": "NOT_SAVED",
            "learned_centered_mae": "NOT_SAVED",
            "learned_pairwise_difference_mae": "NOT_SAVED",
            "learned_pairwise_direction_accuracy": "NOT_SAVED",
            "learned_vs_transparent_same_action": "NOT_SAVED",
        })
        for row in rows:
            action = int(row["action"])
            label = audit[decision_id][action]
            if label["task_status_counts"].get("unknown", 0):
                unknown_cross[(
                    label["task_status_counts"].get("unknown", 0),
                    label["host_unknown_count"],
                )] += 1
            action_rows.append({
                "parent": row["parent"],
                "repeat": int(row["repeat"]),
                "decision_id": decision_id,
                "action": action,
                "is_noop": action == 24,
                "true_remaining_utility": float(row["remaining_utility"]),
                "current_public_score": float(row["current_public_score"]),
                "transparent_history_score": float(row["transparent_history_score"]),
                "learned_seed_7101_prediction": "NOT_SAVED",
                "learned_seed_7102_prediction": "NOT_SAVED",
                "learned_seed_7103_prediction": "NOT_SAVED",
                "learned_ensemble_prediction": "NOT_SAVED",
                "first_acceptance_class": label["acceptance"],
                "first_feedback": label["feedback"],
                "task_unknown_count": label["task_status_counts"].get("unknown", 0),
                "host_confirmation_unknown_count": label["host_unknown_count"],
                "continuation_id": row["continuation_id"],
            })

    fieldnames = list(window_rows[0])
    with (ROOT / "window-comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(window_rows)
    with (ROOT / "candidate-evidence.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(action_rows[0]))
        writer.writeheader()
        writer.writerows(action_rows)

    parents: dict[tuple[str, int], list[dict]] = collections.defaultdict(list)
    for row in window_rows:
        parents[(row["parent"], row["repeat"])].append(row)
    parent_rows = []
    for (parent, repeat), rows in sorted(parents.items()):
        parent_rows.append({
            "parent": parent,
            "repeat": repeat,
            "windows": len(rows),
            "current_regret": sum(float(row["current_regret"]) for row in rows) / len(rows),
            "transparent_regret": sum(float(row["transparent_regret"]) for row in rows) / len(rows),
            "learned_regret": "NOT_SAVED",
            "learned_minus_transparent_regret": "NOT_SAVED",
            "current_minus_transparent_regret": sum(
                float(row["current_regret"]) - float(row["transparent_regret"])
                for row in rows
            ) / len(rows),
            "baseline_different_action_windows": sum(not row["baseline_same_action"] for row in rows),
        })
    with (ROOT / "parent-repeat-regret.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(parent_rows[0]))
        writer.writeheader()
        writer.writerows(parent_rows)

    parent_summary_rows = []
    by_parent_windows: dict[str, list[dict]] = collections.defaultdict(list)
    for row in window_rows:
        by_parent_windows[row["parent"]].append(row)
    parent_metric_names = (
        "true_range", "best_second_gap",
        "current_absolute_mae", "current_centered_mae",
        "current_pairwise_difference_mae", "current_pairwise_direction_accuracy",
        "current_score_span", "current_regret",
        "transparent_absolute_mae", "transparent_centered_mae",
        "transparent_pairwise_difference_mae", "transparent_pairwise_direction_accuracy",
        "transparent_score_span", "transparent_regret",
    )
    for parent, rows in sorted(by_parent_windows.items()):
        output = {"parent": parent, "windows": len(rows)}
        for name in parent_metric_names:
            output[name] = sum(float(row[name]) for row in rows) / len(rows)
        output.update({
            "learned_absolute_mae": "AGGREGATE_ONLY",
            "learned_centered_mae": "NOT_SAVED",
            "learned_pairwise_difference_mae": "NOT_SAVED",
            "learned_pairwise_direction_accuracy": "NOT_SAVED",
            "learned_score_span": "NOT_SAVED",
            "learned_regret": "NOT_SAVED",
        })
        parent_summary_rows.append(output)
    with (ROOT / "parent-summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(parent_summary_rows[0]))
        writer.writeheader()
        writer.writerows(parent_summary_rows)

    train_windows: dict[str, list[dict]] = collections.defaultdict(list)
    for row in all_records:
        if row["split_role"] == "train":
            train_windows[row["decision_id"]].append(row)
    training_candidate_histogram = collections.Counter(len(rows) for rows in train_windows.values())
    training_candidate_rows_by_window_size = {
        str(size): int(size * windows)
        for size, windows in sorted(training_candidate_histogram.items())
    }

    reproduced_current = reproduce_metrics(records, "current_public_score")
    reproduced_transparent = reproduce_metrics(records, "transparent_history_score")
    saved_current = aggregate["comparison"]["current_public"]
    saved_transparent = aggregate["comparison"]["transparent_history"]
    metric_names = (
        "parent_macro_mae",
        "parent_macro_rmse",
        "parent_macro_selected_regret",
        "parent_macro_top1_accuracy",
    )
    baseline_metric_match = all(
        abs(reproduced_current[name] - saved_current[name]) <= 1e-15
        and abs(reproduced_transparent[name] - saved_transparent[name]) <= 1e-15
        for name in metric_names
    )

    baseline_different_rows = [row for row in window_rows if not row["baseline_same_action"]]
    baseline_total_regret = sum(float(row["transparent_regret"]) for row in window_rows)
    worst_baseline_window = max(window_rows, key=lambda row: float(row["transparent_regret"]))
    learned_saved = aggregate["comparison"]["learned_three_seed_mean"]
    repeat_counts = collections.Counter(row["parent"] for row in window_rows)
    equal_windows_per_parent = len(set(repeat_counts.values())) == 1
    summary = {
        "schema": "w1-prediction-ranking-offline-diagnostic/1.0.0",
        "decision_preserved": "PREDICTION_GATE_NOT_PASSED",
        "evaluation_parents_now_development_evidence": True,
        "records": len(records),
        "windows": len(window_rows),
        "parents": len({row["parent"] for row in window_rows}),
        "repeats_per_parent": dict(collections.Counter(row["parent"] for row in window_rows)),
        "candidate_count_histogram": dict(sorted(candidate_histogram.items())),
        "noop_legal_windows": noop_windows,
        "noop_true_best_windows": sum(
            "24" in row["best_actions"].split("|") for row in window_rows
        ),
        "baseline_noop_selected_windows": sum(
            int(row["transparent_selected_action"]) == 24 for row in window_rows
        ),
        "true_exact_best_tie_windows": true_near_ties,
        "exploratory_near_tie_counts_do_not_change_frozen_top1": {
            "gap_at_most_0.001": sum(float(row["best_second_gap"]) <= 0.001 for row in window_rows),
            "gap_at_most_0.005": sum(float(row["best_second_gap"]) <= 0.005 for row in window_rows),
        },
        "current_vs_transparent": {
            "same_action_windows": len(window_rows) - baseline_disagreements,
            "different_action_windows": baseline_disagreements,
            "different_action_details": [
                {
                    key: row[key]
                    for key in (
                        "parent", "repeat", "decision_id",
                        "current_selected_action", "transparent_selected_action",
                        "current_selected_acceptance", "transparent_selected_acceptance",
                        "current_selected_true_utility", "transparent_selected_true_utility",
                        "current_regret", "transparent_regret",
                        "current_minus_transparent_selected_true_utility",
                    )
                }
                for row in baseline_different_rows
            ],
        },
        "window_macro_diagnostics": {
            name: macro(window_rows, name)
            for name in (
                "true_range",
                "best_second_gap",
                "current_score_span",
                "transparent_score_span",
                "current_absolute_mae",
                "transparent_absolute_mae",
                "current_centered_mae",
                "transparent_centered_mae",
                "current_pairwise_difference_mae",
                "transparent_pairwise_difference_mae",
                "current_pairwise_direction_accuracy",
                "transparent_pairwise_direction_accuracy",
                "current_regret",
                "transparent_regret",
                "learned_score_span",
                "learned_centered_mae",
                "learned_pairwise_difference_mae",
                "learned_pairwise_direction_accuracy",
                "learned_regret",
            )
        },
        "frozen_aggregate_metrics": aggregate,
        "baseline_metric_reproduction": {
            "current_public": reproduced_current,
            "transparent_history": reproduced_transparent,
            "matches_saved_to_1e-15": baseline_metric_match,
        },
        "aggregate_only_learned_inferences": {
            "valid_because_each_parent_has_three_windows": equal_windows_per_parent,
            "learned_exact_best_windows": (
                learned_saved["parent_macro_top1_accuracy"] * len(window_rows)
                if equal_windows_per_parent else None
            ),
            "baseline_exact_best_windows": (
                saved_transparent["parent_macro_top1_accuracy"] * len(window_rows)
                if equal_windows_per_parent else None
            ),
            "learned_total_regret": (
                learned_saved["parent_macro_selected_regret"] * len(window_rows)
                if equal_windows_per_parent else None
            ),
            "baseline_total_regret": baseline_total_regret,
            "learned_minus_baseline_total_regret": (
                (learned_saved["parent_macro_selected_regret"]
                 - saved_transparent["parent_macro_selected_regret"])
                * len(window_rows)
                if equal_windows_per_parent else None
            ),
            "cannot_locate_to_windows_or_actions": True,
        },
        "learned_candidate_prediction_availability": {
            "ensemble": False,
            "per_seed": False,
            "evidence": "no per-candidate learned predictions were persisted; only aggregate metrics and checkpoints exist",
            "checkpoint_use_in_this_analysis": False,
            "model_forwards_in_this_analysis": 0,
        },
        "unknown_interpretation": {
            "remaining_utility": "finite saved reward-sequence target for every candidate; unknown host-confirmation fields were not substituted into it",
            "host_confirmation": "unknown values block host-confirmation-specific conclusions",
            "first_acceptance": "unknown acceptance blocks command-state interpretation for that branch but does not erase the saved scalar utility",
            "observed_unknown_cross_counts": [
                {
                    "task_unknown_count": key[0],
                    "host_confirmation_unknown_count": key[1],
                    "branches": value,
                }
                for key, value in sorted(unknown_cross.items())
            ],
        },
        "training_objective_review": {
            "loss": "candidate-row Smooth L1 on absolute remaining utility",
            "ranking_or_centered_term": False,
            "candidate_weighting": "each candidate is one training row; windows with more legal candidates contribute more rows",
            "training_windows": len(train_windows),
            "training_candidate_count_histogram": dict(sorted(training_candidate_histogram.items())),
            "training_candidate_rows_by_window_size": training_candidate_rows_by_window_size,
            "features": "770 current-public values + 32 frozen public-history features + 25-way candidate one-hot",
            "parent_split_isolation": "enforced and passed; train/model-selection/prediction parents are disjoint",
            "continuation": "hungarian-v1-fixed for every saved evaluation record",
        },
        "regret_concentration_available_for_baselines_only": {
            "baseline_total_regret": baseline_total_regret,
            "by_selected_acceptance": {
                acceptance: {
                    "windows": len(rows),
                    "regret_sum": sum(float(row["transparent_regret"]) for row in rows),
                }
                for acceptance, rows in sorted(
                    (
                        key,
                        [row for row in window_rows if row["transparent_selected_acceptance"] == key],
                    )
                    for key in {row["transparent_selected_acceptance"] for row in window_rows}
                )
            },
            "largest_window": {
                "parent": worst_baseline_window["parent"],
                "repeat": worst_baseline_window["repeat"],
                "decision_id": worst_baseline_window["decision_id"],
                "regret": worst_baseline_window["transparent_regret"],
                "share_of_total": (
                    float(worst_baseline_window["transparent_regret"]) / baseline_total_regret
                    if baseline_total_regret else None
                ),
                "selected_action": worst_baseline_window["transparent_selected_action"],
                "selected_acceptance": worst_baseline_window["transparent_selected_acceptance"],
                "noop_selected": int(worst_baseline_window["transparent_selected_action"]) == 24,
            },
            "learned_concentration": "NOT_SAVED",
        },
        "analysis_calls": {
            "environment": 0,
            "reset_or_step": 0,
            "checkpoint_loads": 0,
            "model_initializations": 0,
            "model_forwards": 0,
            "training_updates": 0,
        },
    }
    write_json(ROOT / "diagnostic-summary.json", summary)
    checks = {
        "evaluation_records_111": len(records) == 111,
        "evaluation_windows_24": len(window_rows) == 24,
        "evaluation_parents_8": len({row["parent"] for row in window_rows}) == 8,
        "all_windows_same_continuation": all(
            row["continuation_id"] == "hungarian-v1-fixed" for row in action_rows
        ),
        "legal_candidate_sets_complete": all(
            int(row["candidate_count"]) == len(row["legal_actions"].split("|"))
            for row in window_rows
        ),
        "baseline_metrics_reproduced": baseline_metric_match,
        "current_and_transparent_same_windows_and_labels": True,
        "learned_predictions_not_claimed_available": all(
            row["learned_ensemble_prediction"] == "NOT_SAVED" for row in action_rows
        ),
        "no_environment_model_or_training_calls": all(
            value == 0 for value in summary["analysis_calls"].values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"offline diagnostic integrity failure: {checks}")
    write_json(ROOT / "checks.json", checks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
