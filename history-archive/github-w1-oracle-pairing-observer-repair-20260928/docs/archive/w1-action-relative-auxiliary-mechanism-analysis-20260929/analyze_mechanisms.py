"""Offline mechanism analysis for the sealed relative-auxiliary experiment.

This script reads saved JSON/JSONL only. It does not import the environment,
torch, model code, checkpoints, optimizers, or any execution entrypoint.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ATTEMPT = "w1-action-relative-auxiliary-task-cost-attribution-repair-v1-once"
ROOT = Path(r"E:\Z博士\.codex-exports") / ATTEMPT / "run-once"
OUT = Path(__file__).resolve().parent
TRACE_PATH = ROOT / "per-candidate-predictions.jsonl"
UNITS_PATH = ROOT / "data-units.jsonl"
RECORDS_PATH = ROOT / "learning-records.jsonl"
GAMMA = 0.99
TASK_TERM_WEIGHT = 0.5 * 0.8
ENERGY_TERM_WEIGHT = 0.2
HORIZON = 18.0
UAV_STRIDE = 32
TASK_STRIDE = 32
TASK_OFFSET = (4 + 3 + 4) * 32


class AnalysisError(RuntimeError):
    pass


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except Exception as exc:
                    raise AnalysisError(f"{path}:{line_number}: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose(rows: list[dict], score_name: str) -> dict:
    return min(rows, key=lambda row: (-float(row[score_name]), int(row["action"])))


def branch_components(label: dict) -> dict:
    completion = 0.0
    expiry = 0.0
    energy = 0.0
    signed_task = 0.0
    for index, step in enumerate(label["steps"]):
        vector = step["vector_reward"]
        task_value = GAMMA**index * TASK_TERM_WEIGHT * float(vector[0])
        energy_value = GAMMA**index * ENERGY_TERM_WEIGHT * float(vector[1])
        signed_task += task_value
        if task_value > 0:
            completion += task_value
        elif task_value < 0:
            expiry += task_value
        energy += energy_value
    reconstructed = signed_task + energy
    saved = float(label["discounted_utility"])
    if not math.isclose(reconstructed, saved, rel_tol=0.0, abs_tol=2e-12):
        raise AnalysisError(
            f"utility decomposition mismatch for {label['identity']}: {reconstructed} != {saved}"
        )
    tasks = label.get("tasks", {})
    return {
        "completion_component": completion,
        "expiry_component": expiry,
        "signed_task_component": signed_task,
        "energy_component": energy,
        "utility": saved,
        "energy_used": float(label["energy_used"]),
        "first_feedback": str(label["first_feedback"]),
        "first_acceptance_class": str(label["first_acceptance_class"]),
        "final_completed_count": int(label["final_counts"]["completed"]),
        "final_expired_count": int(label["final_counts"]["expired"]),
        "completed_tasks": sorted(
            task_id for task_id, value in tasks.items() if value.get("state") == "completed"
        ),
        "expired_tasks": sorted(
            task_id for task_id, value in tasks.items() if value.get("state") == "expired"
        ),
        "unknown_host_tasks": sorted(
            task_id for task_id, value in tasks.items() if value.get("status") == "unknown"
        ),
    }


def field(flat: list[float], base: int, index: int) -> dict:
    offset = base + 4 * index
    value, known, valid, age = map(float, flat[offset : offset + 4])
    return {
        "value": value,
        "known": known > 0.5,
        "valid": valid > 0.5,
        "age": age,
    }


def action_public_features(record: dict, action: int) -> dict | None:
    if action == 24:
        return None
    flat = record["flat"]
    uav_slot, task_slot = divmod(action, 6)
    uav_base = uav_slot * UAV_STRIDE
    task_base = TASK_OFFSET + task_slot * TASK_STRIDE
    ux = field(flat, uav_base, 0)
    uy = field(flat, uav_base, 1)
    energy = field(flat, uav_base, 2)
    tx = field(flat, task_base, 0)
    ty = field(flat, task_base, 1)
    deadline = field(flat, task_base, 2)
    distance = math.dist((ux["value"], uy["value"]), (tx["value"], ty["value"]))
    now = float(flat[-2]) * HORIZON
    position_age = max(ux["age"], uy["age"])
    task_position_age = max(tx["age"], ty["age"])
    result = {
        "action": action,
        "uav_id": f"uav-{uav_slot}",
        "task_id": f"task-{task_slot}",
        "decision_time": now,
        "distance": distance,
        "deadline": deadline["value"],
        "arrival_slack_using_current_distance": deadline["value"] - now - distance,
        "energy": energy["value"],
        "energy_margin_at_0.35_per_distance": energy["value"] - 0.35 * distance,
        "uav_position_age": position_age,
        "uav_energy_age": energy["age"],
        "task_position_age": task_position_age,
        "deadline_age": deadline["age"],
        "transparent_distance_with_uav_position_age": distance + position_age,
        "transparent_conservative_energy": max(0.0, energy["value"] - 0.05 * energy["age"]),
        "all_required_fields_known_valid": all(
            item["known"] and item["valid"] for item in (ux, uy, energy, tx, ty, deadline)
        ),
    }
    signature_payload = {
        key: result[key]
        for key in (
            "action",
            "distance",
            "deadline",
            "energy",
            "uav_position_age",
            "uav_energy_age",
            "task_position_age",
            "deadline_age",
        )
    }
    result["feature_signature_sha256"] = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result


def macro_regret(rows: list[dict]) -> float:
    by_parent: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_parent[row["parent"]].append(float(row["regret"]))
    return sum(sum(values) / len(values) for values in by_parent.values()) / len(by_parent)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise AnalysisError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    traces = list(read_jsonl(TRACE_PATH))
    trace_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in traces:
        trace_groups[(row["variant"], row["decision_id"])].append(row)

    b_decisions = sorted(decision for variant, decision in trace_groups if variant == "B")
    disagreements: list[dict] = []
    for decision_id in b_decisions:
        rows = trace_groups[("B", decision_id)]
        learned = choose(rows, "ensemble_prediction")
        transparent = choose(rows, "baseline_transparent")
        if int(learned["action"]) != int(transparent["action"]):
            disagreements.append(
                {
                    "decision_id": decision_id,
                    "parent": learned["parent"],
                    "repeat": int(learned["repeat"]),
                    "learned_action": int(learned["action"]),
                    "transparent_action": int(transparent["action"]),
                }
            )
    if len(disagreements) != 10:
        raise AnalysisError(f"expected 10 learned/transparent disagreements, got {len(disagreements)}")
    focus_ids = {row["decision_id"] for row in disagreements}

    units: dict[str, dict] = {}
    for unit in read_jsonl(UNITS_PATH):
        if unit.get("split_role") != "prediction_evaluation" or unit.get("status") != "LABELED":
            continue
        labels = unit.get("explanation_labels", [])
        if not labels:
            continue
        decision_id = labels[0]["identity"]["decision_id"]
        if decision_id in focus_ids:
            units[decision_id] = unit
    if set(units) != focus_ids:
        raise AnalysisError(f"missing data units: {sorted(focus_ids - set(units))}")

    records_by_decision: dict[str, list[dict]] = defaultdict(list)
    for record in read_jsonl(RECORDS_PATH):
        if record.get("split_role") == "prediction_evaluation" and record.get("decision_id") in focus_ids:
            records_by_decision[record["decision_id"]].append(record)
    if set(records_by_decision) != focus_ids:
        raise AnalysisError(f"missing learning records: {sorted(focus_ids - set(records_by_decision))}")

    window_rows: list[dict] = []
    for disagreement in disagreements:
        decision_id = disagreement["decision_id"]
        b_rows = trace_groups[("B", decision_id)]
        learned_trace = choose(b_rows, "ensemble_prediction")
        transparent_trace = choose(b_rows, "baseline_transparent")
        best = max(float(row["true_utility"]) for row in b_rows)
        unit = units[decision_id]
        labels = {int(row["identity"]["action"]): row for row in unit["explanation_labels"]}
        records = {int(row["action"]): row for row in records_by_decision[decision_id]}
        learned_action = int(learned_trace["action"])
        transparent_action = int(transparent_trace["action"])
        learned_label = branch_components(labels[learned_action])
        transparent_label = branch_components(labels[transparent_action])
        learned_features = action_public_features(records[learned_action], learned_action)
        transparent_features = action_public_features(records[transparent_action], transparent_action)
        feature_hash = hashlib.sha256(
            json.dumps(
                {
                    "flat": records[learned_action]["flat"],
                    "history": records[learned_action]["history"],
                    "legal": records[learned_action]["legal_actions"],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if feature_hash != learned_trace["public_input_sha256"]:
            raise AnalysisError(f"prediction feature hash mismatch: {decision_id}")
        history = records[learned_action]["history"]
        continuation_counts = [history[offset] for offset in (6, 14, 22, 30)]
        learned_delta = learned_label["utility"] - transparent_label["utility"]
        window_rows.append(
            {
                "decision_id": decision_id,
                "parent": disagreement["parent"],
                "repeat": disagreement["repeat"],
                "decision_step": int(unit["decision_step"]),
                "candidate_actions": json.dumps(learned_trace["legal_actions"], separators=(",", ":")),
                "frozen_decision_input_sha256": unit["decision_input_sha256"],
                "prediction_feature_input_sha256": learned_trace["public_input_sha256"],
                "learned_action": learned_action,
                "transparent_action": transparent_action,
                "learned_true_utility": learned_label["utility"],
                "transparent_true_utility": transparent_label["utility"],
                "learned_minus_transparent_utility": learned_delta,
                "direction": "learned_better" if learned_delta > 1e-12 else "transparent_better" if learned_delta < -1e-12 else "tie",
                "learned_regret": best - learned_label["utility"],
                "transparent_regret": best - transparent_label["utility"],
                "learned_completion_component": learned_label["completion_component"],
                "transparent_completion_component": transparent_label["completion_component"],
                "delta_completion_component": learned_label["completion_component"] - transparent_label["completion_component"],
                "learned_expiry_component": learned_label["expiry_component"],
                "transparent_expiry_component": transparent_label["expiry_component"],
                "delta_expiry_component": learned_label["expiry_component"] - transparent_label["expiry_component"],
                "learned_energy_component": learned_label["energy_component"],
                "transparent_energy_component": transparent_label["energy_component"],
                "delta_energy_component": learned_label["energy_component"] - transparent_label["energy_component"],
                "learned_energy_used": learned_label["energy_used"],
                "transparent_energy_used": transparent_label["energy_used"],
                "learned_acceptance": learned_label["first_acceptance_class"],
                "transparent_acceptance": transparent_label["first_acceptance_class"],
                "learned_first_feedback": learned_label["first_feedback"],
                "transparent_first_feedback": transparent_label["first_feedback"],
                "learned_final_completed": learned_label["final_completed_count"],
                "transparent_final_completed": transparent_label["final_completed_count"],
                "learned_final_expired": learned_label["final_expired_count"],
                "transparent_final_expired": transparent_label["final_expired_count"],
                "learned_completed_tasks": json.dumps(learned_label["completed_tasks"], separators=(",", ":")),
                "transparent_completed_tasks": json.dumps(transparent_label["completed_tasks"], separators=(",", ":")),
                "learned_expired_tasks": json.dumps(learned_label["expired_tasks"], separators=(",", ":")),
                "transparent_expired_tasks": json.dumps(transparent_label["expired_tasks"], separators=(",", ":")),
                "learned_unknown_host_tasks": json.dumps(learned_label["unknown_host_tasks"], separators=(",", ":")),
                "transparent_unknown_host_tasks": json.dumps(transparent_label["unknown_host_tasks"], separators=(",", ":")),
                "learned_public_features": json.dumps(learned_features, sort_keys=True, separators=(",", ":")),
                "transparent_public_features": json.dumps(transparent_features, sort_keys=True, separators=(",", ":")),
                "prior_history_continuation_counts_newest_first": json.dumps(continuation_counts, separators=(",", ":")),
                "current_confirmed_continuation": "unavailable_not_persisted",
                "pairing_common_result_mismatches": int(unit["pairing"]["common_result_mismatches"]),
                "pairing_parameter_conflicts": int(unit["pairing"]["parameter_conflict_count"]),
                "pairing_multiplicity_mismatches": int(unit["pairing"]["multiplicity_mismatch_count"]),
            }
        )

    # Exact full public input hashes and reduced action signatures across repeats.
    by_parent = defaultdict(list)
    for row in window_rows:
        by_parent[row["parent"]].append(row)
    for row in window_rows:
        peers = [item for item in by_parent[row["parent"]] if item is not row]
        row["same_full_public_input_hash_in_another_repeat"] = any(
            item["frozen_decision_input_sha256"] == row["frozen_decision_input_sha256"] for item in peers
        )
        current_features = json.loads(row["learned_public_features"])
        current_signature = None if current_features is None else current_features["feature_signature_sha256"]
        row["same_learned_action_feature_signature_in_another_repeat"] = any(
            item["learned_action"] == row["learned_action"]
            and json.loads(item["learned_public_features"])["feature_signature_sha256"] == current_signature
            for item in peers
            if json.loads(item["learned_public_features"]) is not None
        )

    evidence_rows: list[dict] = []
    for row in sorted(window_rows, key=lambda item: (item["parent"], item["repeat"])):
        for side in ("learned", "transparent"):
            features = json.loads(row[f"{side}_public_features"])
            evidence_rows.append(
                {
                    "decision_id": row["decision_id"],
                    "parent": row["parent"],
                    "repeat": row["repeat"],
                    "method": side,
                    "action": row[f"{side}_action"],
                    "uav_id": None if features is None else features["uav_id"],
                    "task_id": None if features is None else features["task_id"],
                    "true_utility": row[f"{side}_true_utility"],
                    "completion_component": row[f"{side}_completion_component"],
                    "expiry_component": row[f"{side}_expiry_component"],
                    "energy_component": row[f"{side}_energy_component"],
                    "energy_used": row[f"{side}_energy_used"],
                    "first_feedback": row[f"{side}_first_feedback"],
                    "acceptance_class": row[f"{side}_acceptance"],
                    "decision_time": None if features is None else features["decision_time"],
                    "distance": None if features is None else features["distance"],
                    "deadline": None if features is None else features["deadline"],
                    "arrival_slack": None if features is None else features["arrival_slack_using_current_distance"],
                    "energy": None if features is None else features["energy"],
                    "uav_position_age": None if features is None else features["uav_position_age"],
                    "uav_energy_age": None if features is None else features["uav_energy_age"],
                    "task_position_age": None if features is None else features["task_position_age"],
                    "deadline_age": None if features is None else features["deadline_age"],
                    "required_fields_known_valid": None if features is None else features["all_required_fields_known_valid"],
                    "prior_history_continuation_counts_newest_first": row[
                        "prior_history_continuation_counts_newest_first"
                    ],
                    "current_confirmed_continuation": row["current_confirmed_continuation"],
                }
            )

    seed_rows: list[dict] = []
    seed_summary: list[dict] = []
    seed_disagreements: dict[str, list[str]] = {}
    for variant in ("A", "B"):
        decisions = sorted(decision for value, decision in trace_groups if value == variant)
        decision_actions: dict[str, list[int]] = defaultdict(list)
        for predictor in ("seed:7101", "seed:7102", "seed:7103", "ensemble"):
            selected_rows = []
            for decision_id in decisions:
                rows = trace_groups[(variant, decision_id)]
                if predictor == "ensemble":
                    selected = choose(rows, "ensemble_prediction")
                else:
                    seed = int(predictor.split(":", 1)[1])
                    try:
                        seed_index = list(rows[0]["seed_ids"]).index(seed)
                    except ValueError as exc:
                        raise AnalysisError(f"missing seed {seed} in {decision_id}") from exc
                    selected = min(
                        rows,
                        key=lambda row: (-float(row["seed_predictions"][seed_index]), int(row["action"])),
                    )
                best = max(float(row["true_utility"]) for row in rows)
                result = {
                    "variant": variant,
                    "predictor": predictor,
                    "decision_id": decision_id,
                    "parent": selected["parent"],
                    "repeat": int(selected["repeat"]),
                    "action": int(selected["action"]),
                    "true_utility": float(selected["true_utility"]),
                    "regret": best - float(selected["true_utility"]),
                    "focus_disagreement_window": decision_id in focus_ids,
                }
                seed_rows.append(result)
                selected_rows.append(result)
                if predictor.startswith("seed:"):
                    decision_actions[decision_id].append(int(selected["action"]))
            seed_summary.append(
                {
                    "variant": variant,
                    "predictor": predictor,
                    "parent_macro_regret": macro_regret(selected_rows),
                    "top1_accuracy": sum(row["regret"] <= 1e-12 for row in selected_rows) / len(selected_rows),
                    "decisions": len(selected_rows),
                }
            )
        seed_disagreements[variant] = sorted(
            decision_id for decision_id, actions in decision_actions.items() if len(set(actions)) > 1
        )

    reversals = {}
    for parent, rows in sorted(by_parent.items()):
        directions = {row["direction"] for row in rows}
        if "learned_better" in directions and "transparent_better" in directions:
            reversals[parent] = [
                {
                    "repeat": row["repeat"],
                    "direction": row["direction"],
                    "learned_action": row["learned_action"],
                    "transparent_action": row["transparent_action"],
                    "delta": row["learned_minus_transparent_utility"],
                    "same_full_public_input_hash_in_another_repeat": row[
                        "same_full_public_input_hash_in_another_repeat"
                    ],
                }
                for row in sorted(rows, key=lambda item: item["repeat"])
            ]

    focus_repeat_rows: list[dict] = []
    for decision_id in b_decisions:
        rows = trace_groups[("B", decision_id)]
        if rows[0]["parent"] not in {"train-0005", "train-0006", "train-0007"}:
            continue
        learned = choose(rows, "ensemble_prediction")
        transparent = choose(rows, "baseline_transparent")
        focus_repeat_rows.append(
            {
                "decision_id": decision_id,
                "parent": learned["parent"],
                "repeat": int(learned["repeat"]),
                "learned_action": int(learned["action"]),
                "transparent_action": int(transparent["action"]),
                "learned_true_utility": float(learned["true_utility"]),
                "transparent_true_utility": float(transparent["true_utility"]),
                "learned_minus_transparent_utility": float(learned["true_utility"])
                - float(transparent["true_utility"]),
                "same_action": int(learned["action"]) == int(transparent["action"]),
            }
        )

    write_csv(OUT / "disagreement-windows.csv", sorted(window_rows, key=lambda row: (row["parent"], row["repeat"])))
    write_csv(OUT / "window-evidence.csv", evidence_rows)
    write_csv(OUT / "focus-repeat-comparison.csv", sorted(focus_repeat_rows, key=lambda row: (row["parent"], row["repeat"])))
    write_csv(OUT / "seed-selections.csv", seed_rows)
    summary = {
        "schema": "w1-relative-auxiliary-mechanism-analysis/1.0.0",
        "attempt": ATTEMPT,
        "analysis_boundary": {
            "environment_calls": 0,
            "model_initializations": 0,
            "checkpoint_loads": 0,
            "model_forwards": 0,
            "training_updates": 0,
            "sealed_task_set_read": False,
        },
        "source_hashes": {
            str(TRACE_PATH): sha256(TRACE_PATH),
            str(UNITS_PATH): sha256(UNITS_PATH),
            str(RECORDS_PATH): sha256(RECORDS_PATH),
        },
        "disagreement_windows": len(window_rows),
        "learned_better_windows": sum(row["direction"] == "learned_better" for row in window_rows),
        "transparent_better_windows": sum(row["direction"] == "transparent_better" for row in window_rows),
        "tie_windows": sum(row["direction"] == "tie" for row in window_rows),
        "sum_learned_minus_transparent_utility": sum(
            float(row["learned_minus_transparent_utility"]) for row in window_rows
        ),
        "all_common_communication_fates_paired": all(
            row["pairing_common_result_mismatches"] == 0
            and row["pairing_parameter_conflicts"] == 0
            and row["pairing_multiplicity_mismatches"] == 0
            for row in window_rows
        ),
        "current_confirmed_continuation_identity_available": False,
        "seed_summary": seed_summary,
        "seed_disagreement_windows": seed_disagreements,
        "direction_reversals_within_disagreement_windows": reversals,
        "limitations": [
            "The readable current public observation and continuation identity were not persisted; flat/history vectors and their hash were persisted.",
            "The saved per-step task vector is net (success-expired)/6. Positive and negative signed components cannot distinguish simultaneous success and expiry in one step.",
            "Exact public-input hashes differ across repeats; reduced-feature resemblance is descriptive and is not an independent causal test.",
        ],
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
