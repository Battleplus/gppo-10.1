"""Independent metrics reconstructed from the per-candidate trace."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Mapping

from trace_schema import PredictionTraceRow


def _macro(values: Mapping[str, list[float]]) -> float:
    if not values:
        raise ValueError("no parent values")
    return sum(sum(items) / len(items) for items in values.values()) / len(values)


def evaluate_trace(rows: Iterable[PredictionTraceRow], *, score: str = "ensemble_prediction") -> dict:
    selected_score = {"ensemble_prediction", "baseline_current_public", "baseline_transparent"}
    if score not in selected_score:
        raise ValueError("unsupported score field")
    grouped: dict[str, list[PredictionTraceRow]] = defaultdict(list)
    for row in rows:
        row.validate()
        grouped[row.decision_id].append(row)
    if not grouped:
        raise ValueError("empty trace")
    parent_abs: dict[str, list[float]] = defaultdict(list)
    parent_sq: dict[str, list[float]] = defaultdict(list)
    parent_regret: dict[str, list[float]] = defaultdict(list)
    parent_top1: dict[str, list[float]] = defaultdict(list)
    selections = {}
    for decision_id, group in grouped.items():
        if any(row.variant != next(iter(group)).variant for row in group):
            raise ValueError("mixed variants in one metric group")
        if {row.action for row in group} != set(group[0].legal_actions):
            raise ValueError(f"{decision_id}: incomplete candidate set")
        score_values = [(row, float(getattr(row, score))) for row in group]
        for row, value in score_values:
            if not row.target_valid or row.true_utility is None:
                raise ValueError("metric target is unknown")
            error = value - row.true_utility
            parent_abs[row.parent].append(abs(error))
            parent_sq[row.parent].append(error * error)
        chosen = max(score_values, key=lambda item: (item[1], -item[0].action))[0]
        best = max(row.true_utility for row in group if row.true_utility is not None)
        regret = best - float(chosen.true_utility)
        parent_regret[chosen.parent].append(regret)
        parent_top1[chosen.parent].append(float(abs(regret) <= 1e-12))
        selections[decision_id] = {
            "action": chosen.action,
            "candidate_actions": list(group[0].legal_actions),
            "regret": regret,
            "best_utility": best,
            "variant": chosen.variant,
            "noop": chosen.action == 24,
            "tie_rule": "max(score, -action)",
        }
    return {
        "score": score,
        "records": sum(len(group) for group in grouped.values()),
        "decisions": len(grouped),
        "parents": len(parent_abs),
        "parent_macro_mae": _macro(parent_abs),
        "parent_macro_rmse": sum(math.sqrt(sum(values) / len(values)) for values in parent_sq.values()) / len(parent_sq),
        "parent_macro_selected_regret": _macro(parent_regret),
        "parent_macro_top1_accuracy": _macro(parent_top1),
        "selections": selections,
    }


def compare_trace(rows: Iterable[PredictionTraceRow]) -> dict:
    values = list(rows)
    variants = sorted({row.variant for row in values})
    if set(variants) != {"A", "B"}:
        raise ValueError("paired trace requires both A and B variants")
    by_variant: dict[str, dict[str, list[PredictionTraceRow]]] = {"A": defaultdict(list), "B": defaultdict(list)}
    for row in values:
        by_variant[row.variant][row.decision_id].append(row)
    if set(by_variant["A"]) != set(by_variant["B"]):
        raise ValueError("A and B do not contain the same decision windows")
    for decision_id in by_variant["A"]:
        left_group = sorted(by_variant["A"][decision_id], key=lambda row: row.action)
        right_group = sorted(by_variant["B"][decision_id], key=lambda row: row.action)
        if len(left_group) != len(right_group) or [row.action for row in left_group] != [row.action for row in right_group]:
            raise ValueError(f"A/B candidate set mismatch for {decision_id}")
        for left, right in zip(left_group, right_group, strict=True):
            identity = (left.parent, left.repeat, left.action, left.legal_actions, left.continuation_id, left.public_input_sha256, left.true_utility, left.target_valid, left.baseline_current_public, left.baseline_transparent)
            other = (right.parent, right.repeat, right.action, right.legal_actions, right.continuation_id, right.public_input_sha256, right.true_utility, right.target_valid, right.baseline_current_public, right.baseline_transparent)
            if identity != other:
                raise ValueError(f"A/B trace identity mismatch for {decision_id}")
    result = {variant: evaluate_trace([row for row in values if row.variant == variant]) for variant in variants}
    result["current_public"] = evaluate_trace([row for row in values if row.variant == variants[0]], score="baseline_current_public")
    result["transparent_history"] = evaluate_trace([row for row in values if row.variant == variants[0]], score="baseline_transparent")
    return result
