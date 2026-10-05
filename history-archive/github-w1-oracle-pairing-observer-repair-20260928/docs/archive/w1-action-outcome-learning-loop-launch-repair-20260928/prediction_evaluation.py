"""Parent-grouped prediction, ranking, and stage-gate metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

from learning_schema import LearningRecord


def ensemble_mean(seed_predictions: Sequence[Sequence[float]]) -> list[float]:
    if len(seed_predictions) != 3:
        raise ValueError("the frozen ensemble requires all three model seeds")
    lengths = {len(values) for values in seed_predictions}
    if len(lengths) != 1:
        raise ValueError("seed prediction lengths differ")
    return [sum(values[index] for values in seed_predictions) / 3.0 for index in range(next(iter(lengths)))]


def evaluate_scores(records: Sequence[LearningRecord], scores: Sequence[float]) -> dict:
    if not records or len(records) != len(scores):
        raise ValueError("records and scores must have equal nonzero length")
    if not all(math.isfinite(float(value)) for value in scores):
        raise ValueError("scores contain nonfinite values")
    by_decision: dict[str, list[tuple[LearningRecord, float]]] = defaultdict(list)
    for record, score in zip(records, scores, strict=True):
        by_decision[record.decision_id].append((record, float(score)))

    parent_errors: dict[str, list[float]] = defaultdict(list)
    parent_squared: dict[str, list[float]] = defaultdict(list)
    parent_regrets: dict[str, list[float]] = defaultdict(list)
    parent_top1: dict[str, list[float]] = defaultdict(list)
    for rows in by_decision.values():
        parent = rows[0][0].parent
        if any(record.parent != parent for record, _score in rows):
            raise ValueError("decision crosses parent identity")
        for record, score in rows:
            error = score - record.remaining_utility
            parent_errors[parent].append(abs(error))
            parent_squared[parent].append(error * error)
        chosen = max(rows, key=lambda item: (item[1], -item[0].action))[0]
        best_utility = max(record.remaining_utility for record, _score in rows)
        regret = best_utility - chosen.remaining_utility
        parent_regrets[parent].append(regret)
        parent_top1[parent].append(float(abs(regret) <= 1e-12))

    def parent_macro(table: Mapping[str, Sequence[float]]) -> float:
        return sum(sum(values) / len(values) for values in table.values()) / len(table)

    return {
        "records": len(records),
        "decisions": len(by_decision),
        "parents": len(parent_errors),
        "parent_macro_mae": parent_macro(parent_errors),
        "parent_macro_rmse": sum(
            math.sqrt(sum(values) / len(values)) for values in parent_squared.values()
        ) / len(parent_squared),
        "parent_macro_selected_regret": parent_macro(parent_regrets),
        "parent_macro_top1_accuracy": parent_macro(parent_top1),
    }


def compare_all(records: Sequence[LearningRecord], learned_scores: Sequence[float]) -> dict:
    return {
        "learned_three_seed_mean": evaluate_scores(records, learned_scores),
        "current_public": evaluate_scores(records, [row.current_public_score for row in records]),
        "transparent_history": evaluate_scores(
            records, [row.transparent_history_score for row in records]
        ),
    }


def prediction_gate(comparison: Mapping[str, Mapping[str, float]]) -> dict:
    learned = comparison["learned_three_seed_mean"]
    current = comparison["current_public"]
    transparent = comparison["transparent_history"]
    checks = {
        "mae_better_than_current_public": learned["parent_macro_mae"] < current["parent_macro_mae"],
        "mae_better_than_transparent_history": learned["parent_macro_mae"] < transparent["parent_macro_mae"],
        "selection_regret_better_than_transparent_history": (
            learned["parent_macro_selected_regret"]
            < transparent["parent_macro_selected_regret"]
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "meaning": "strict aggregate development gate; it is not task-benefit evidence",
    }

