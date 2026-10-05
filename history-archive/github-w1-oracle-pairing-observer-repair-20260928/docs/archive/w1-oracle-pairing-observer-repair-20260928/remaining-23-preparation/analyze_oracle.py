"""Frozen joint analysis for one reused unit plus 23 newly executed units."""

from __future__ import annotations

import math


PARENTS = tuple(f"validation-{index:04d}" for index in range(8))
REPEATS = (0, 1, 2)
REUSED_UNIT = ("validation-0000", 0)
MATERIALITY = 0.01
ACCEPTED = frozenset(("accepted", "accepted_then_ack_lost"))


class AnalysisError(RuntimeError):
    pass


def _task_counts(summary: dict) -> dict[str, int]:
    tasks = summary.get("tasks")
    if not isinstance(tasks, dict) or len(tasks) != 6:
        raise AnalysisError("branch summary must contain exactly six tasks")
    return {
        "physical_on_time": sum(
            row.get("physical_arrival_before_deadline") is True
            for row in tasks.values()
        ),
        "host_on_time": sum(
            row.get("host_confirmation_before_deadline") is True
            for row in tasks.values()
        ),
        "unknown": sum(
            row.get("status") in {"unknown", "censored", "unfinished"}
            for row in tasks.values()
        ),
    }


def _best(rows: list[dict]) -> tuple[dict, list[int]]:
    utilities = [float(row["discounted_utility"]) for row in rows]
    if not utilities or not all(math.isfinite(value) for value in utilities):
        raise AnalysisError("oracle candidates lack finite utility")
    best = max(utilities)
    tied = sorted(
        int(row["identity"]["action"])
        for row in rows
        if abs(float(row["discounted_utility"]) - best) <= 1e-12
    )
    winner = min(
        (row for row in rows if int(row["identity"]["action"]) in tied),
        key=lambda row: int(row["identity"]["action"]),
    )
    return winner, tied


def analyze(
    candidate_rows: list[dict],
    summaries: list[dict],
    pairing_rows: list[dict],
    *,
    reused_audit: dict,
) -> dict:
    expected = {(parent, repeat) for parent in PARENTS for repeat in REPEATS}
    candidates: dict[tuple[str, int], dict] = {}
    for row in candidate_rows:
        key = (row.get("parent"), int(row.get("repeat", -1)))
        if key not in expected or key in candidates:
            raise AnalysisError("duplicate or unexpected candidate unit")
        if row.get("status") not in {"OPPORTUNITY", "NO_OPPORTUNITY"}:
            raise AnalysisError("invalid candidate status")
        candidates[key] = row
    if set(candidates) != expected:
        raise AnalysisError("fixed 8x3 candidate matrix is incomplete")
    if not reused_audit.get("passed") or reused_audit.get("counted_research_samples") != 1:
        raise AnalysisError("reused unit audit is not a one-sample pass")
    if candidates[REUSED_UNIT].get("stage") != "single_unit_dynamic_gate":
        raise AnalysisError("reused unit identity was not preserved")

    indexed: dict[tuple[str, int, int], dict] = {}
    unknown_task_labels = 0
    for summary in summaries:
        identity = summary.get("identity", {})
        key = (
            identity.get("parent"),
            int(identity.get("repeat", -1)),
            int(identity.get("action", -1)),
        )
        if key[:2] not in expected or key in indexed:
            raise AnalysisError("duplicate or unexpected branch summary")
        if not math.isfinite(float(summary.get("discounted_utility", math.nan))):
            raise AnalysisError("branch utility missing or nonfinite")
        if not summary.get("terminated") or summary.get("truncated"):
            raise AnalysisError("branch is not native-terminal and untruncated")
        counts = _task_counts(summary)
        unknown_task_labels += counts["unknown"]
        indexed[key] = summary

    pairing = {}
    for row in pairing_rows:
        key = (row.get("parent"), int(row.get("repeat", -1)))
        if key in pairing or key not in expected:
            raise AnalysisError("duplicate or unexpected pairing row")
        if (
            int(row.get("common_call_keys", 0)) <= 0
            or int(row.get("action_specific_call_keys", 0)) <= 0
            or int(row.get("common_result_mismatches", -1)) != 0
            or int(row.get("parameter_conflict_count", -1)) != 0
            or int(row.get("multiplicity_mismatch_count", -1)) != 0
        ):
            raise AnalysisError("primitive pairing evidence failed")
        pairing[key] = row

    units = []
    opportunity_count = 0
    opportunity_parents = set()
    both_accepted_material_units = 0
    acceptance_mismatch_units = 0
    for parent in PARENTS:
        for repeat in REPEATS:
            key = (parent, repeat)
            candidate = candidates[key]
            source = "reused_observer_gate" if key == REUSED_UNIT else "remaining_23_attempt"
            if candidate["status"] == "NO_OPPORTUNITY":
                if any(item[:2] == key for item in indexed) or key in pairing:
                    raise AnalysisError("no-opportunity unit has branch or pairing rows")
                units.append({
                    "parent": parent,
                    "repeat": repeat,
                    "source": source,
                    "status": "NO_OPPORTUNITY",
                    "primary_gain": 0.0,
                    "reason": candidate.get("reason"),
                })
                continue

            opportunity_count += 1
            opportunity_parents.add(parent)
            actions = tuple(int(value) for value in candidate["branch_actions"])
            if len(actions) != len(set(actions)) or not 2 <= len(actions) <= 25:
                raise AnalysisError("invalid branch action set")
            rows = [indexed.get((parent, repeat, action)) for action in actions]
            if any(row is None for row in rows):
                raise AnalysisError("opportunity has missing branch labels")
            branch_rows = [row for row in rows if row is not None]
            nonnoop = [row for row in branch_rows if int(row["identity"]["action"]) != 24]
            if not nonnoop:
                raise AnalysisError("primary oracle has no non-NOOP candidates")
            baseline_action = int(candidate["hungarian_action"])
            baseline = indexed.get((parent, repeat, baseline_action))
            if baseline is None:
                raise AnalysisError("Hungarian baseline branch is missing")
            primary, primary_ties = _best(nonnoop)
            secondary, secondary_ties = _best(branch_rows)
            gain = float(primary["discounted_utility"]) - float(baseline["discounted_utility"])
            if gain < -1e-12:
                raise AnalysisError("oracle gain is negative despite containing baseline")
            baseline_counts = _task_counts(baseline)
            primary_counts = _task_counts(primary)
            baseline_accepted = baseline["first_acceptance_class"] in ACCEPTED
            winner_accepted = primary["first_acceptance_class"] in ACCEPTED
            both_accepted = baseline_accepted and winner_accepted
            acceptance_mismatch = baseline_accepted != winner_accepted
            acceptance_mismatch_units += int(acceptance_mismatch)
            if both_accepted and gain >= MATERIALITY:
                both_accepted_material_units += 1
            paired = pairing.get(key)
            if paired is None:
                raise AnalysisError("opportunity lacks primitive pairing analysis")
            units.append({
                "parent": parent,
                "repeat": repeat,
                "source": source,
                "status": "LABELED",
                "public_state_sha256": candidate["public_state_sha256"],
                "decision_step": int(candidate["decision_step"]),
                "hungarian_action": baseline_action,
                "hungarian_assignment": candidate["action_descriptors"][str(baseline_action)],
                "primary_winner_action": int(primary["identity"]["action"]),
                "primary_winner_assignment": candidate["action_descriptors"][
                    str(primary["identity"]["action"])
                ],
                "primary_tied_actions": primary_ties,
                "secondary_winner_action": int(secondary["identity"]["action"]),
                "secondary_tied_actions": secondary_ties,
                "primary_gain": gain,
                "primary_gain_nonnegative_by_construction": True,
                "hungarian_acceptance": baseline["first_acceptance_class"],
                "winner_acceptance": primary["first_acceptance_class"],
                "both_first_commands_accepted": both_accepted,
                "acceptance_mismatch": acceptance_mismatch,
                "delta_energy": float(primary["energy_used"]) - float(baseline["energy_used"]),
                "delta_physical_on_time": primary_counts["physical_on_time"] - baseline_counts["physical_on_time"],
                "delta_host_on_time": primary_counts["host_on_time"] - baseline_counts["host_on_time"],
                "hungarian_unknown_task_labels": baseline_counts["unknown"],
                "winner_unknown_task_labels": primary_counts["unknown"],
                "common_primitive_call_keys": int(paired["common_call_keys"]),
                "action_specific_primitive_call_keys": int(paired["action_specific_call_keys"]),
            })

    parent_results = []
    passing_parents = 0
    for parent in PARENTS:
        rows = [row for row in units if row["parent"] == parent]
        if len(rows) != 3:
            raise AnalysisError("parent does not contain exactly three repeats")
        gains = [float(row["primary_gain"]) for row in rows]
        mean_gain = sum(gains) / 3.0
        positive_repeats = sum(value > 0 for value in gains)
        parent_pass = mean_gain >= MATERIALITY and positive_repeats >= 2
        passing_parents += int(parent_pass)
        labeled = [row for row in rows if row["status"] == "LABELED"]
        state_hashes = {row["public_state_sha256"] for row in labeled}
        parent_results.append({
            "parent": parent,
            "three_repeat_mean_gain": mean_gain,
            "positive_repeat_count": positive_repeats,
            "parent_pass": parent_pass,
            "winner_actions_descriptive_only": [
                {
                    "repeat": row["repeat"],
                    "public_state_sha256": row["public_state_sha256"],
                    "hungarian_action": row["hungarian_action"],
                    "hungarian_assignment": row["hungarian_assignment"],
                    "winner_action": row["primary_winner_action"],
                    "winner_assignment": row["primary_winner_assignment"],
                    "ties": row["primary_tied_actions"],
                }
                for row in labeled
            ],
            "winner_comparability": (
                "same_public_state_hash_only"
                if len(state_hashes) <= 1
                else "different_public_states_do_not_compare_action_numbers"
            ),
        })

    macro_gain = sum(row["three_repeat_mean_gain"] for row in parent_results) / len(PARENTS)
    coverage_pass = opportunity_count >= 12 and len(opportunity_parents) >= 4
    materiality_pass = macro_gain >= MATERIALITY and passing_parents >= 4
    acceptance_independent_support = both_accepted_material_units >= 1
    gate_pass = coverage_pass and materiality_pass and acceptance_independent_support
    if not coverage_pass:
        decision = "INSUFFICIENT_COVERAGE"
    elif gate_pass:
        decision = "PASS_PREPARE_INDEPENDENT_PARENT_PREDICTOR_STUDY_ONLY"
    else:
        decision = "FAIL_FIRST_WINDOW_HUNGARIAN_CONTINUATION_CEILING"

    return {
        "schema": "w1-action-consequence-oracle-ceiling-analysis/3.0.0",
        "decision": decision,
        "gate_pass": gate_pass,
        "matrix": {
            "fixed_units": 24,
            "reused_units": 1,
            "new_units": 23,
            "reused_unit_counted_once": True,
            "historical_failed_attempts_included_as_samples": False,
        },
        "coverage": {
            "opportunity_units": opportunity_count,
            "opportunity_parents": len(opportunity_parents),
            "required_units": 12,
            "required_parents": 4,
            "pass": coverage_pass,
        },
        "materiality": {
            "threshold": MATERIALITY,
            "eight_parent_macro_mean_gain": macro_gain,
            "passing_parents": passing_parents,
            "required_passing_parents": 4,
            "both_accepted_material_units": both_accepted_material_units,
            "acceptance_mismatch_units": acceptance_mismatch_units,
            "acceptance_mismatch_alone_is_insufficient": True,
            "pass": materiality_pass and acceptance_independent_support,
        },
        "labels": {
            "unknown_task_labels": unknown_task_labels,
            "unknowns_preserved_not_imputed": True,
            "utility_independent_of_host_confirmation_unknown": True,
        },
        "structural_fact": (
            "Primary oracle gain is nonnegative by construction because the "
            "Hungarian action is included in the maximized candidate set. "
            "This is not repeat stability or predictor learnability evidence."
        ),
        "parents": parent_results,
        "units": units,
        "interpretation_limit": (
            "A failing result closes only the first qualifying public window "
            "with frozen Hungarian continuation on these development parents. "
            "A passing result is an optimistic hindsight ceiling that may include "
            "unpredictable random luck and is not predictor performance."
        ),
        "historical_10ms_cost_failure_preserved": True,
    }
