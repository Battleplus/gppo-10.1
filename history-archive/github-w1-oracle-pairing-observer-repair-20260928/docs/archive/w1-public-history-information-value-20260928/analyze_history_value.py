"""Offline public-history information-value audit.

Reads sealed JSONL and source text only. It does not import runtime, environment,
model, checkpoint, or training modules.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
RUN = Path("E:/Z博士/runs/w1-light-repaired-fair-rerun-v2-nativefs-once")
STEPS = RUN / "run-once/evaluation/steps.jsonl"
EPISODES = RUN / "run-once/evaluation/episodes.jsonl"
MECHANISM = Path("E:/Z博士/research-plans/w1-history-consequence-mechanism-v1")
POSTRUN = Path("E:/Z博士/research-plans/w1-light-repaired-fair-rerun-v2-postrun-analysis")

U_FIELDS = ("x", "y", "energy", "alive", "connected", "idle")
T_FIELDS = ("x", "y", "deadline", "remaining_service", "priority", "pending", "region_id", "target_id")
SPEED = 1.0
IDLE_POWER = 0.05
TRAVEL_POWER = 0.35
DECISION_INTERVAL = 1.0
TOL = 1e-5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite value: {value!r}")
    return result


def public_field(row: list[Any], index: int, now: float) -> dict[str, Any]:
    value, known, valid, age = map(finite, row[4 * index:4 * index + 4])
    if age < -TOL:
        raise ValueError("future public measurement")
    return {
        "value": value,
        "known": known > 0.5,
        "valid": valid > 0.5,
        "age": max(0.0, age),
        "measured_at": now - age,
    }


def time_key(value: float) -> int:
    return int(round(finite(value) * 1_000_000))


def episode_key(row: dict[str, Any]) -> str:
    return json.dumps(row["key"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def decode_continuations(obs: dict[str, Any]) -> dict[str, str]:
    ids = obs["public_entity_ids"]
    tasks = ids["tasks"]
    result: dict[str, str] = {}
    for raw in obs["continuation_actions"]:
        action = int(raw)
        uav_index, task_index = divmod(action, 6)
        if not (0 <= uav_index < len(ids["uavs"]) and 0 <= task_index < len(tasks)):
            raise ValueError("invalid public continuation identity")
        uav, task = ids["uavs"][uav_index], tasks[task_index]
        if uav in result and result[uav] != task:
            raise ValueError("multiple public continuations for one UAV")
        result[uav] = task
    return result


def entity_row(obs: dict[str, Any], kind: str, entity: str) -> list[Any] | None:
    ids = obs["public_entity_ids"][kind]
    try:
        index = ids.index(entity)
    except ValueError:
        return None
    return obs[kind][index]


def get_field(obs: dict[str, Any], kind: str, entity: str, name: str) -> dict[str, Any] | None:
    row = entity_row(obs, kind, entity)
    if row is None:
        return None
    names = U_FIELDS if kind == "uavs" else T_FIELDS
    return public_field(row, names.index(name), finite(obs["time"]))


def read_groups() -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    with STEPS.open(encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, 1):
            row = json.loads(line)
            row["_source_line"] = source_line
            groups[episode_key(row)].append(row)
    if len(groups) != 56:
        raise ValueError(f"expected 56 episodes, got {len(groups)}")
    for rows in groups.values():
        rows.sort(key=lambda item: int(item["step"]))
        seen_steps: set[int] = set()
        last_time = -math.inf
        for row in rows:
            step = int(row["step"])
            now = finite(row["observation"]["time"])
            if step in seen_steps or now < last_time - TOL:
                raise ValueError("duplicate step or reversed time")
            seen_steps.add(step)
            last_time = now
    return groups


def measurement_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    result: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in rows:
        obs = row["observation"]
        visible_at = finite(obs["time"])
        for kind, names in (("uavs", U_FIELDS), ("tasks", T_FIELDS)):
            for entity, values in zip(obs["public_entity_ids"][kind], obs[kind]):
                for index, name in enumerate(names):
                    item = public_field(values, index, visible_at)
                    if not (item["known"] and item["valid"]):
                        continue
                    key = (kind, entity, name, time_key(item["measured_at"]))
                    old = result.get(key)
                    candidate = {
                        "value": item["value"],
                        "measured_at": item["measured_at"],
                        "visible_at": visible_at,
                        "source_line": row["_source_line"],
                    }
                    if old is None or visible_at < old["visible_at"] - TOL:
                        result[key] = candidate
                    elif abs(old["value"] - item["value"]) > TOL:
                        raise ValueError("same measurement identity changed value")
    return result


def exact_label(index: dict[tuple[str, str, str, int], dict[str, Any]], kind: str, entity: str,
                field_name: str, target_time: float) -> dict[str, Any] | None:
    return index.get((kind, entity, field_name, time_key(target_time)))


def age_bin(age: float) -> str:
    if age <= TOL:
        return "0"
    if age <= 1.0 + TOL:
        return "(0,1]"
    if age <= 2.0 + TOL:
        return "(1,2]"
    return ">2"


def project_toward(start: tuple[float, float], target: tuple[float, float], elapsed: float) -> tuple[float, float]:
    distance = math.dist(start, target)
    if distance <= TOL or elapsed <= 0:
        return start
    fraction = min(1.0, SPEED * elapsed / distance)
    return tuple(a + fraction * (b - a) for a, b in zip(start, target))  # type: ignore[return-value]


def continuation_is_continuous(rows: list[dict[str, Any]], upto: int, start_time: float,
                               uav: str, task: str) -> tuple[bool, str]:
    relevant: list[float] = []
    for row in rows[:upto + 1]:
        obs = row["observation"]
        now = finite(obs["time"])
        if now <= start_time + TOL:
            continue
        relevant.append(now)
        if decode_continuations(obs).get(uav) != task:
            return False, "continuation_not_continuous"
    target_time = finite(rows[upto]["observation"]["time"])
    if target_time <= start_time + TOL:
        return decode_continuations(rows[upto]["observation"]).get(uav) == task, "zero_age"
    if not relevant or relevant[0] - start_time > DECISION_INTERVAL + TOL:
        return False, "history_gap"
    previous = relevant[0]
    for current in relevant[1:]:
        if current - previous > DECISION_INTERVAL + TOL:
            return False, "history_gap"
        previous = current
    return True, "continuous"


def metric_error(prediction: Any, truth: Any) -> float:
    if isinstance(truth, tuple):
        return math.dist(tuple(map(float, prediction)), truth)
    return abs(float(prediction) - float(truth))


def final_completion_records(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        for task, record in row.get("completion_records", {}).items():
            result[task] = dict(record)
    return result


def make_sample(*, target: str, parent: str, method: str, repeat: int, source_line: int,
                now: float, truth: Any, label_lines: list[int], label_visible_at: list[float],
                age: float, confirmed: bool, predictions: dict[str, Any], c_reason: str) -> dict[str, Any]:
    errors = {name: metric_error(value, truth) for name, value in predictions.items() if value is not None}
    return {
        "target": target,
        "parent": parent,
        "method": method,
        "repeat": repeat,
        "source_line": source_line,
        "time": now,
        "truth": truth,
        "label_source_lines": label_lines,
        "label_visible_at": label_visible_at,
        "receipt_delay": max(label_visible_at) - now if label_visible_at else None,
        "age": age,
        "age_bin": age_bin(age),
        "confirmed_continuation": confirmed,
        "predictions": predictions,
        "errors": errors,
        "c_reason": c_reason,
    }


def analyze() -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    groups = read_groups()
    samples: list[dict[str, Any]] = []
    availability = collections.Counter()
    refusals = collections.Counter()
    action_outcomes = collections.Counter()
    evidence_rows: list[dict[str, Any]] = []

    for rows in groups.values():
        labels = measurement_index(rows)
        final_records = final_completion_records(rows)
        key = rows[0]["key"]
        parent, repeat, method = str(key[0]), int(key[1]), str(key[2])
        for index, row in enumerate(rows):
            obs = row["observation"]
            now = finite(obs["time"])
            ids = obs["public_entity_ids"]
            active = decode_continuations(obs)

            for uav in ids["uavs"]:
                availability["position_candidates"] += 1
                availability["energy_candidates"] += 1
                x_label = exact_label(labels, "uavs", uav, "x", now)
                y_label = exact_label(labels, "uavs", uav, "y", now)
                energy_label = exact_label(labels, "uavs", uav, "energy", now)
                x = get_field(obs, "uavs", uav, "x")
                y = get_field(obs, "uavs", uav, "y")
                energy = get_field(obs, "uavs", uav, "energy")
                current_task = active.get(uav)
                task_x = get_field(obs, "tasks", current_task, "x") if current_task else None
                task_y = get_field(obs, "tasks", current_task, "y") if current_task else None

                c_position = None
                c_energy = None
                c_reason = "no_confirmed_continuation"
                c_start = None
                if current_task and x and y and x["known"] and x["valid"] and y["known"] and y["valid"]:
                    if abs(x["measured_at"] - y["measured_at"]) > TOL:
                        c_reason = "asynchronous_xy"
                    elif not (task_x and task_y and task_x["known"] and task_x["valid"] and task_y["known"] and task_y["valid"]):
                        c_reason = "unknown_target"
                    else:
                        c_start = x["measured_at"]
                        continuous, c_reason = continuation_is_continuous(rows, index, c_start, uav, current_task)
                        if continuous:
                            c_position = project_toward(
                                (x["value"], y["value"]),
                                (task_x["value"], task_y["value"]),
                                now - c_start,
                            )
                            if energy and energy["known"] and energy["valid"] and energy["measured_at"] >= c_start - TOL:
                                energy_continuous, energy_reason = continuation_is_continuous(
                                    rows, index, energy["measured_at"], uav, current_task
                                )
                                if energy_continuous:
                                    c_energy = max(0.0, energy["value"] - TRAVEL_POWER * (now - energy["measured_at"]))
                                else:
                                    refusals["energy:" + energy_reason] += 1
                            else:
                                refusals["energy_before_verified_interval"] += 1
                if c_position is None:
                    refusals["position:" + c_reason] += 1

                if x_label and y_label and x and y and all((x["known"], x["valid"], y["known"], y["valid"])):
                    availability["position_labels"] += 1
                    prediction = (x["value"], y["value"])
                    samples.append(make_sample(
                        target="position", parent=parent, method=method, repeat=repeat,
                        source_line=row["_source_line"], now=now,
                        truth=(x_label["value"], y_label["value"]),
                        label_lines=[x_label["source_line"], y_label["source_line"]],
                        label_visible_at=[x_label["visible_at"], y_label["visible_at"]],
                        age=max(x["age"], y["age"]), confirmed=current_task is not None,
                        predictions={"current_observation": prediction, "existing_search": prediction,
                                     "public_history": c_position}, c_reason=c_reason,
                    ))
                    if c_position is not None:
                        availability["position_history_outputs"] += 1

                if energy_label and energy and energy["known"] and energy["valid"]:
                    availability["energy_labels"] += 1
                    samples.append(make_sample(
                        target="energy", parent=parent, method=method, repeat=repeat,
                        source_line=row["_source_line"], now=now, truth=energy_label["value"],
                        label_lines=[energy_label["source_line"]], label_visible_at=[energy_label["visible_at"]],
                        age=energy["age"], confirmed=current_task is not None,
                        predictions={"current_observation": energy["value"],
                                     "existing_search": max(0.0, energy["value"] - IDLE_POWER * energy["age"]),
                                     "public_history": c_energy}, c_reason=c_reason,
                    ))
                    if c_energy is not None:
                        availability["energy_history_outputs"] += 1

                if current_task:
                    availability["remaining_candidates"] += 1
                    record = final_records.get(current_task)
                    if record and str(record.get("uav_id")) == uav and record.get("physical_arrival_time") is not None:
                        arrival = finite(record["physical_arrival_time"])
                        if arrival >= now - TOL and x and y and task_x and task_y and all(
                            item["known"] and item["valid"] for item in (x, y, task_x, task_y)
                        ):
                            availability["remaining_labels"] += 1
                            distance = math.dist((x["value"], y["value"]), (task_x["value"], task_y["value"]))
                            c_remaining = None if c_position is None else math.dist(
                                c_position, (task_x["value"], task_y["value"])
                            ) / SPEED
                            samples.append(make_sample(
                                target="remaining_time", parent=parent, method=method, repeat=repeat,
                                source_line=row["_source_line"], now=now, truth=max(0.0, arrival - now),
                                label_lines=[row["_source_line"]], label_visible_at=[arrival],
                                age=max(x["age"], y["age"]), confirmed=True,
                                predictions={"current_observation": distance / SPEED,
                                             "existing_search": distance / SPEED + max(x["age"], y["age"]),
                                             "public_history": c_remaining}, c_reason=c_reason,
                            ))
                            if c_remaining is not None:
                                availability["remaining_history_outputs"] += 1
                        else:
                            availability["remaining_censored_or_invalid"] += 1
                    else:
                        availability["remaining_censored_or_other_uav"] += 1

            if int(row["action"]) != 24:
                availability["nonnoop_actions"] += 1
                if not bool(row["command_submitted"]):
                    raise ValueError("saved non-NOOP without command submission")
                feedback = str(row["feedback"])
                action_outcomes["acceptance:" + feedback] += 1
                uav_index, task_index = divmod(int(row["action"]), 6)
                uav = ids["uavs"][uav_index]
                task = ids["tasks"][task_index]
                completion = final_records.get(task)
                if feedback != "accepted":
                    completion_class = "not_accepted"
                elif completion is None:
                    completion_class = "accepted_no_saved_completion"
                elif str(completion.get("uav_id")) == uav:
                    completion_class = "accepted_completed_same_uav"
                else:
                    completion_class = "accepted_completed_other_uav"
                action_outcomes["completion:" + completion_class] += 1
                if len(evidence_rows) < 20 or feedback != "accepted":
                    evidence_rows.append({
                        "source_line": row["_source_line"], "parent": parent, "repeat": repeat,
                        "method": method, "time": now, "uav": uav, "task": task,
                        "feedback": feedback, "completion_class": completion_class,
                    })

    if availability["nonnoop_actions"] != 367:
        raise ValueError("non-NOOP index changed")
    return samples, {"availability": dict(availability), "refusals": dict(refusals),
                     "action_outcomes": dict(action_outcomes)}, evidence_rows


def aggregate_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    units = {"position": "distance", "energy": "energy", "remaining_time": "seconds"}
    for target in ("position", "energy", "remaining_time"):
        target_rows = [row for row in samples if row["target"] == target]
        common_rows = [row for row in target_rows if row["predictions"].get("public_history") is not None]
        slices: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "all", target_rows)]
        for name in sorted({row["parent"] for row in target_rows}):
            slices.append(("parent", name, [row for row in target_rows if row["parent"] == name]))
        for name in sorted({row["method"] for row in target_rows}):
            slices.append(("method", name, [row for row in target_rows if row["method"] == name]))
        for name in ("0", "(0,1]", "(1,2]", ">2"):
            slices.append(("age_bin", name, [row for row in target_rows if row["age_bin"] == name]))
        for value in (False, True):
            slices.append(("confirmed_continuation", str(value).lower(),
                           [row for row in target_rows if row["confirmed_continuation"] is value]))

        for slice_type, slice_value, rows in slices:
            for estimator in ("current_observation", "existing_search"):
                errors = [row["errors"][estimator] for row in rows]
                if not errors:
                    continue
                output.append(summary_row(target, "all_valid_labels", estimator, slice_type,
                                          slice_value, rows, errors, len(rows), units[target]))

        common_slices: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "all", common_rows)]
        for name in sorted({row["parent"] for row in common_rows}):
            common_slices.append(("parent", name, [row for row in common_rows if row["parent"] == name]))
        for name in sorted({row["method"] for row in common_rows}):
            common_slices.append(("method", name, [row for row in common_rows if row["method"] == name]))
        for name in ("0", "(0,1]", "(1,2]", ">2"):
            common_slices.append(("age_bin", name, [row for row in common_rows if row["age_bin"] == name]))
        for slice_type, slice_value, rows in common_slices:
            for estimator in ("current_observation", "existing_search", "public_history"):
                errors = [row["errors"][estimator] for row in rows]
                if not errors:
                    continue
                output.append(summary_row(target, "common_public_history_rows", estimator, slice_type,
                                          slice_value, rows, errors, len(target_rows), units[target]))
    return output


def summary_row(target: str, scope: str, estimator: str, slice_type: str, slice_value: str,
                rows: list[dict[str, Any]], errors: list[float], denominator: int, unit: str) -> dict[str, Any]:
    return {
        "target": target,
        "scope": scope,
        "estimator": estimator,
        "slice_type": slice_type,
        "slice_value": slice_value,
        "label_rows": denominator,
        "estimated_rows": len(errors),
        "abstained_rows": denominator - len(errors),
        "parent_coverage": len({row["parent"] for row in rows}),
        "mean_absolute_error": sum(errors) / len(errors),
        "root_mean_squared_error": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "unit": unit,
        "interval_coverage": "not_applicable_point_estimate",
        "interval_width": "not_applicable_point_estimate",
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    samples, diagnostics, action_evidence = analyze()
    results = aggregate_rows(samples)
    write_csv(ROOT / "estimation-results.csv", results, list(results[0]))

    availability = diagnostics["availability"]
    action = diagnostics["action_outcomes"]
    availability_rows = [
        {
            "object": "decision_time_uav_position",
            "candidate_records": availability.get("position_candidates", 0),
            "valid_labels": availability.get("position_labels", 0),
            "history_outputs_on_valid_labels": availability.get("position_history_outputs", 0),
            "label_source": "later/current public x+y with measured_at exactly decision time",
            "label_kind": "actual telemetry measurement",
            "time_alignment": "measurement t; first public visibility at t or later",
            "missing_or_censoring": "one/both exact-time telemetry fields never visible before episode end",
            "valid_evaluation": "yes_with_delivery_selection_bias",
        },
        {
            "object": "decision_time_uav_energy",
            "candidate_records": availability.get("energy_candidates", 0),
            "valid_labels": availability.get("energy_labels", 0),
            "history_outputs_on_valid_labels": availability.get("energy_history_outputs", 0),
            "label_source": "later/current public energy with measured_at exactly decision time",
            "label_kind": "actual telemetry measurement",
            "time_alignment": "measurement t; first public visibility at t or later",
            "missing_or_censoring": "exact-time telemetry never visible before episode end",
            "valid_evaluation": "yes_with_delivery_selection_bias",
        },
        {
            "object": "confirmed_execution_remaining_travel_time",
            "candidate_records": availability.get("remaining_candidates", 0),
            "valid_labels": availability.get("remaining_labels", 0),
            "history_outputs_on_valid_labels": availability.get("remaining_history_outputs", 0),
            "label_source": "saved physical_arrival_time for same public task and UAV",
            "label_kind": "execution result",
            "time_alignment": "arrival time minus decision time",
            "missing_or_censoring": "no arrival, different completing UAV, invalid geometry, or end censoring",
            "valid_evaluation": "yes_for_same_uav_completed_continuations_only",
        },
        {
            "object": "executed_action_acceptance",
            "candidate_records": availability.get("nonnoop_actions", 0),
            "valid_labels": sum(value for key, value in action.items() if key.startswith("acceptance:")),
            "history_outputs_on_valid_labels": 0,
            "label_source": "saved post-step feedback",
            "label_kind": "execution result",
            "time_alignment": "after selected action at the same decision",
            "missing_or_censoring": "none for saved non-NOOP actions; unavailable as decision input",
            "valid_evaluation": "label_available_but_no_prefrozen_estimator",
        },
        {
            "object": "executed_action_eventual_same_uav_completion",
            "candidate_records": availability.get("nonnoop_actions", 0),
            "valid_labels": action.get("completion:accepted_completed_same_uav", 0),
            "history_outputs_on_valid_labels": 0,
            "label_source": "accepted feedback plus final physical completion record identity",
            "label_kind": "execution result",
            "time_alignment": "selected action to eventual physical arrival",
            "missing_or_censoring": "rejected/lost, no completion, or completion by another UAV",
            "valid_evaluation": "positive_attribution_only_not_counterfactual",
        },
    ]
    write_csv(ROOT / "input-label-availability.csv", availability_rows, list(availability_rows[0]))

    overall = [row for row in results if row["slice_type"] == "overall"]
    summary = {
        "schema": "w1-public-history-information-value/1.0.0",
        "development_only": True,
        "episodes": 56,
        "decisions": 774,
        "parents": 8,
        "methods": 7,
        "constants": {"travel_speed": SPEED, "idle_power": IDLE_POWER, "travel_power": TRAVEL_POWER},
        "diagnostics": diagnostics,
        "overall_results": overall,
        "environment_calls": 0,
        "model_initializations_or_loads": 0,
        "model_forwards": 0,
        "training_updates": 0,
        "new_attempts": 0,
    }
    (ROOT / "analysis-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "sample-evidence.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in action_evidence), encoding="utf-8"
    )

    inputs = [
        STEPS, EPISODES,
        RUN / "public_controller.py", RUN / "classical_baselines.py", RUN / "rl_adapters.py",
        RUN / "runner.py", RUN / "environment.json", RUN / "native/gppo_world/service_clock.py",
        RUN / "native/gppo_world/m10_environment.py",
        MECHANISM / "summary.json", MECHANISM / "decision-evidence.jsonl",
        POSTRUN / "prediction-fusion-analysis.json",
    ]
    index = {
        "schema": "w1-public-history-information-value-input-index/1.0.0",
        "inputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in inputs],
        "excluded": [
            "defective-runtime records",
            "heldout/test results",
            "checkpoints and model outputs as executable inputs",
            "oracle branch outcomes as estimation samples",
        ],
        "calls": {"environment": 0, "model": 0, "training": 0},
    }
    (ROOT / "input-index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
