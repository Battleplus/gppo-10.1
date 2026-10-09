"""Parent-macro prediction gate; no risk certification and no threshold fitting."""
from collections import defaultdict
from statistics import mean
import math
from consequence_contract import SEEDS

MAJOR_LOSS = -.1


def window_metrics(predicted, truth, actions, baseline):
    if len(set(actions)) != len(actions) or set(predicted) != set(actions) or set(truth) != set(actions):
        raise ValueError("METRIC_CANDIDATE_IDENTITY")
    if baseline not in actions or any(not math.isfinite(v) for v in (*predicted.values(), *truth.values())):
        raise ValueError("METRIC_INVALID")
    if abs(truth[baseline]) > 1e-7: raise ValueError("BASELINE_DELTA_NOT_ZERO")
    selected = max(actions, key=lambda a: (predicted[a], -a))
    best = max(truth.values())
    pairs = [(a, b) for i, a in enumerate(actions) for b in actions[i + 1:] if truth[a] != truth[b]]
    ranking = mean(float((predicted[a] - predicted[b]) * (truth[a] - truth[b]) > 0) for a, b in pairs) if pairs else None
    return {"selected": selected, "regret": best - truth[selected], "transparent_regret": best,
            "improvement": truth[selected], "major_loss": truth[selected] <= MAJOR_LOSS,
            "top1": truth[selected] == best, "pairwise_accuracy": ranking, "non_noop": selected != 24,
            "utility_mae": mean(abs(predicted[a] - truth[a]) for a in actions)}


def summarize(rows):
    parents = defaultdict(list)
    for r in rows: parents[r["parent"]].append(r)
    if not parents: return {"status": "unavailable", "reason": "no_opportunity_windows"}
    per_parent = {p: {k: mean(float(r[k]) for r in rs) for k in
                     ("regret", "transparent_regret", "improvement", "major_loss", "top1", "non_noop", "utility_mae")}
                  for p, rs in parents.items()}
    improvement = mean(p["improvement"] for p in per_parent.values())
    worst = sorted(per_parent, key=lambda p: (per_parent[p]["improvement"], p))
    return {"status": "evaluated", "parent_count": len(parents), "window_count": len(rows), "parents": per_parent,
            "macro_improvement": improvement, "major_loss_rate": mean(p["major_loss"] for p in per_parent.values()),
            "non_noop_coverage": mean(p["non_noop"] for p in per_parent.values()), "worst_parent": worst[0],
            "worst_two_mean_loss": -mean(per_parent[p]["improvement"] for p in worst[:2]),
            "pairwise_accuracy": mean(r["pairwise_accuracy"] for r in rows if r["pairwise_accuracy"] is not None)
                                  if any(r["pairwise_accuracy"] is not None for r in rows) else None}


def decide_gate(ensemble, by_seed, *, identity_complete, independent, cpu_mean_ms, wall_p95_ms,
                opportunity_parent_count, valid_label_fraction, expected_parent_count=24):
    evaluated = ensemble.get("status") == "evaluated" and all(by_seed.get(s, {}).get("status") == "evaluated" for s in SEEDS)
    conditions = {"identity_complete": identity_complete is True, "independent_evaluation": independent is True,
        "all_three_seeds": evaluated, "regret_improved": evaluated and ensemble["macro_improvement"] > 0,
        "two_seeds_improve": evaluated and sum(by_seed[s]["macro_improvement"] > 0 for s in SEEDS) >= 2,
        "major_loss_not_increased": evaluated and ensemble["major_loss_rate"] == 0,
        "opportunity_coverage": opportunity_parent_count >= expected_parent_count / 2,
        "nonzero_non_noop": evaluated and ensemble["non_noop_coverage"] > 0,
        "valid_utility_labels": valid_label_fraction >= .9}
    return {"conditions": conditions, "pass": all(conditions.values()),
            "stop": None if all(conditions.values()) else "offline_prediction_gate_stop",
            "risk_certification": False,
            "major_loss_rule": "any ensemble-selected paired delta<=-0.1 stops; not a general safety guarantee",
            "practical_cost_report_only": {"cpu_mean_ms": cpu_mean_ms, "wall_p95_ms": wall_p95_ms,
                "pass": cpu_mean_ms is not None and wall_p95_ms is not None and cpu_mean_ms<=10 and wall_p95_ms<=50}}
