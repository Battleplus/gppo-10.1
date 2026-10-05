"""Offline development-data availability export.

Reads sealed JSONL and configuration files only. It imports no runtime, model,
checkpoint, optimizer, runner, or environment module.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from outcome_contract import NOOP, SCHEMA, canonical, digest, masked, unknown_labels, validate_record


ROOT = Path(__file__).resolve().parent
FAIR = Path("E:/Z博士/runs/w1-light-repaired-fair-rerun-v2-nativefs-once")
FAIR_STEPS = FAIR / "run-once/evaluation/steps.jsonl"
FAIR_EPISODES = FAIR / "run-once/evaluation/episodes.jsonl"
TAPES = FAIR / "tapes-light-dev.json"
ORACLE = Path("E:/Z博士/runs/w1-action-consequence-oracle-remaining-23-v1-once")
ORACLE_SOURCES = (
    (ORACLE / "reused-first-unit/branch-summaries.jsonl", ORACLE / "reused-first-unit/candidate-decisions.jsonl"),
    (ORACLE / "run-once/branch-summaries.jsonl", ORACLE / "run-once/candidate-decisions.jsonl"),
)
GAMMA = 0.99
TASK_SCALE = 0.5
TASK_CAPACITY = 6.0
FLEET_INITIAL_ENERGY = 36.0


class ExportError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            yield line_number, json.loads(line)


def episode_key(row: dict[str, Any]) -> str:
    return canonical(row["key"])


def public_payload(observation: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "time",
        "uavs",
        "tasks",
        "mask",
        "continuation_actions",
        "public_entity_ids",
        "trigger_flags",
        "event_signal",
        "version",
    )
    payload = {name: observation[name] for name in allowed if name in observation}
    if len(payload.get("mask", [])) != 25:
        raise ExportError("public mask does not have 25 actions")
    return payload


def action_descriptor(action: int, observation: dict[str, Any]) -> dict[str, Any]:
    if action == NOOP:
        return {"action": NOOP, "kind": "NOOP", "uav_id": None, "task_id": None}
    uav_index, task_index = divmod(action, 6)
    ids = observation["public_entity_ids"]
    try:
        return {
            "action": action,
            "kind": "assignment",
            "uav_id": ids["uavs"][uav_index],
            "task_id": ids["tasks"][task_index],
        }
    except IndexError as error:
        raise ExportError(f"legal action {action} cannot resolve public identity") from error


def command_class(feedback: str) -> str:
    return {
        "accepted": "accepted",
        "ack_lost_after_accept": "accepted_then_ack_lost",
        "command_lost": "command_transport_lost",
        "resource_unavailable": "resource_unavailable",
        "noop": "noop",
    }.get(feedback, "executor_rejected" if feedback.startswith("rejected") else "unknown")


def task_labels(
    task_specs: list[dict[str, Any]], completion_records: dict[str, dict[str, Any]], source: str,
) -> dict[str, Any]:
    result = {}
    for task in task_specs:
        task_id = task["task_id"]
        deadline = float(task["deadline"])
        record = completion_records.get(task_id)
        physical = None if record is None else record.get("physical_arrival_time")
        host = None if record is None else record.get("host_confirmation_time")
        uav = None if record is None else record.get("uav_id")
        physical_valid = isinstance(physical, (int, float)) and not isinstance(physical, bool) and math.isfinite(float(physical))
        host_valid = isinstance(host, (int, float)) and not isinstance(host, bool) and math.isfinite(float(host))
        if physical_valid:
            state = "completed"
        else:
            state = "unknown"
        physical_label_valid = physical_valid
        uav_valid = isinstance(uav, str) and bool(uav)
        result[task_id] = {
            "state": masked(state, True, source, name=f"tasks[{task_id}].state"),
            "physical_on_time": masked(bool(float(physical) <= deadline) if physical_valid else None,
                                       physical_label_valid, source if physical_label_valid else None,
                                       name="physical_on_time"),
            "completion_uav": masked(str(uav) if isinstance(uav, str) and uav else None,
                                      uav_valid, source if uav_valid else None, name="completion_uav"),
            "host_on_time": masked(bool(float(host) <= deadline) if host_valid else None,
                                   host_valid, source if host_valid else None, name="host_on_time"),
            "host_confirmation_time": masked(float(host) if host_valid else None,
                                             host_valid, source if host_valid else None,
                                             name="host_confirmation_time"),
        }
    return result


def remaining_utility(rows: list[dict[str, Any]], index: int, preference: list[float]) -> float:
    utility = 0.0
    for offset, row in enumerate(rows[index:]):
        vector = row["vector_reward"]
        utility += GAMMA ** offset * (
            TASK_SCALE * float(preference[0]) * float(vector[0]) + float(preference[1]) * float(vector[1])
        )
    if not math.isfinite(utility):
        raise ExportError("remaining utility is nonfinite")
    return utility


def factual_labels(
    rows: list[dict[str, Any]], index: int, candidate: dict[str, Any], task_specs: list[dict[str, Any]],
    final_records: dict[str, dict[str, Any]], source_path: Path,
) -> dict[str, Any]:
    row = rows[index]
    source = f"{source_path}:{row['_line']}..{rows[-1]['_line']}"
    task_result = task_labels(task_specs, final_records, source)
    before_energy = FLEET_INITIAL_ENERGY if index == 0 else float(rows[index - 1]["energy"])
    final_energy = float(rows[-1]["energy"])
    preference = list(row["key"][4])
    labels = {
        "first_command_state": masked(command_class(str(row["feedback"])), True,
                                      f"{source_path}:{row['_line']}:feedback", name="first_command_state"),
        "target_task": {
            "task_id": candidate["task_id"],
            "physical_on_time": masked(None, False, None, name="target.physical_on_time"),
            "completed_by_current_uav": masked(None, False, None, name="target.completed_by_current_uav"),
            "completion_time": masked(None, False, None, name="target.completion_time"),
            "host_on_time": masked(None, False, None, name="target.host_on_time"),
        },
        "tasks": task_result,
        "energy_to_terminal": masked(before_energy - final_energy, True, source, name="energy_to_terminal"),
        "remaining_utility": masked(remaining_utility(rows, index, preference), True, source,
                                    name="remaining_utility"),
        "terminal_time": masked(None, False, None, name="terminal_time"),
    }
    if candidate["kind"] == "assignment":
        task_id = candidate["task_id"]
        task = task_result[task_id]
        record = final_records.get(task_id, {})
        physical = record.get("physical_arrival_time")
        physical_valid = isinstance(physical, (int, float)) and not isinstance(physical, bool) and math.isfinite(float(physical))
        labels["target_task"] = {
            "task_id": task_id,
            "physical_on_time": dict(task["physical_on_time"]),
            "completed_by_current_uav": masked(
                task["completion_uav"]["value"] == candidate["uav_id"] if task["completion_uav"]["valid"] else None,
                task["completion_uav"]["valid"], source if task["completion_uav"]["valid"] else None,
                name="target.completed_by_current_uav",
            ),
            "completion_time": masked(float(physical) - float(row["observation"]["time"]) if physical_valid else None,
                                      physical_valid, source if physical_valid else None, name="target.completion_time"),
            "host_on_time": dict(task["host_on_time"]),
        }
    return labels


def load_factual() -> tuple[list[dict[str, Any]], list[dict[str, Any]], collections.Counter[str]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for line_number, row in read_jsonl(FAIR_STEPS):
        row["_line"] = line_number
        groups[episode_key(row)].append(row)
    if len(groups) != 56:
        raise ExportError(f"expected 56 factual episodes, found {len(groups)}")
    saved_episodes = {
        canonical(row["key"]): row
        for _, row in read_jsonl(FAIR_EPISODES)
    }
    if len(saved_episodes) != 56 or set(saved_episodes) != set(groups):
        raise ExportError("factual episode summaries do not match step groups")
    tape_by_parent = {row["parent"]: row for row in json.loads(TAPES.read_text(encoding="utf-8"))}
    contexts: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    counts: collections.Counter[str] = collections.Counter()

    for key, rows in groups.items():
        rows.sort(key=lambda item: int(item["step"]))
        parent, repeat, method = str(rows[0]["key"][0]), int(rows[0]["key"][1]), str(rows[0]["key"][2])
        task_specs = tape_by_parent[parent]["scenario"]["tasks"]
        task_ids = [task["task_id"] for task in task_specs]
        rebuilt_episode_utility = remaining_utility(rows, 0, list(rows[0]["key"][4]))
        if not math.isclose(rebuilt_episode_utility, float(saved_episodes[key]["utility"]), abs_tol=1e-9):
            raise ExportError("frozen utility reconstruction differs from saved episode")
        counts["factual_episode_utility_reconstructions_verified"] += 1
        final_records: dict[str, dict[str, Any]] = {}
        for row in rows:
            final_records.update(row.get("completion_records", {}))
        history: list[dict[str, Any]] = []
        own_actions: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            observation = row["observation"]
            now = float(observation["time"])
            context_id = digest({"episode": key, "step": row["step"], "public": public_payload(observation)})
            contexts.append({
                "schema": "w1-public-decision-context/1.0.0",
                "context_id": context_id,
                "source": {"path": str(FAIR_STEPS), "line": row["_line"]},
                "decision_time": now,
                "public_payload": public_payload(observation),
                "public_history": list(history),
                "own_action_history": list(own_actions),
            })
            legal = [action for action, value in enumerate(observation["mask"]) if bool(value)]
            if not legal:
                raise ExportError("decision has no legal action")
            decision_id = f"factual:{parent}:r{repeat}:{method}:s{row['step']}"
            for action in legal:
                candidate = action_descriptor(action, observation)
                executed = action == int(row["action"])
                labels = factual_labels(rows, index, candidate, task_specs, final_records, FAIR_STEPS) if executed else unknown_labels(task_ids)
                labels["target_task"]["task_id"] = candidate["task_id"]
                record = {
                    "schema": SCHEMA,
                    "decision_id": decision_id,
                    "source": {
                        "kind": "saved_factual_trajectory",
                        "path": str(FAIR_STEPS),
                        "line": row["_line"],
                        "parent": parent,
                        "repeat": repeat,
                        "method": method,
                        "step": int(row["step"]),
                        "decision_time": now,
                        "split_role": "development",
                        "independent_validation": False,
                    },
                    "input": {
                        "current_public": {"context_id": context_id, "observation_time": now,
                                           "payload_sha256": digest(public_payload(observation))},
                        "public_history": list(history),
                        "own_action_history": list(own_actions),
                    },
                    "candidate": candidate,
                    "legal_actions": legal,
                    "continuation_id": f"factual-policy:{method}:saved-run-v2",
                    "provenance": {
                        "is_executed": executed,
                        "is_true_branch": False,
                        "label_scope": "factual_trajectory" if executed else "unknown_unexecuted",
                    },
                    "labels": labels,
                    "training_eligible": False,
                    "eligibility_reasons": [
                        "development_parent_only",
                        "factual_continuation_not_fixed_across_methods",
                        "unexecuted_candidates_unknown" if not executed else "single_factual_action_only",
                    ],
                }
                validate_record(record)
                records.append(record)
                counts["factual_candidate_records"] += 1
                counts["factual_executed_candidate_records" if executed else "factual_unexecuted_candidate_records"] += 1
                if executed:
                    counts[f"factual_first_command:{labels['first_command_state']['value']}"] += 1
                    for task in labels["tasks"].values():
                        counts[f"factual_task_state:{task['state']['value']}"] += 1
                        if not task["host_on_time"]["valid"]:
                            counts["factual_host_unknown_tasks"] += 1
            history.append({"context_id": context_id, "observation_time": now,
                            "payload_sha256": digest(public_payload(observation))})
            own_actions.append({"action": int(row["action"]), "decision_time": now,
                                "submitted": bool(row["command_submitted"])})
            counts["factual_decisions"] += 1
    return contexts, records, counts


def oracle_task_labels(tasks: dict[str, dict[str, Any]], source: str) -> dict[str, Any]:
    result = {}
    for task_id, task in tasks.items():
        physical = task.get("physical_arrival_time")
        host = task.get("host_confirmation_time")
        uav = task.get("uav_id")
        result[task_id] = {
            "state": masked(str(task.get("state", "unknown")), True, source, name="state"),
            "physical_on_time": masked(task.get("physical_arrival_before_deadline"),
                                       isinstance(task.get("physical_arrival_before_deadline"), bool),
                                       source if isinstance(task.get("physical_arrival_before_deadline"), bool) else None,
                                       name="physical_on_time"),
            "completion_uav": masked(uav if isinstance(uav, str) and uav else None,
                                      isinstance(uav, str) and bool(uav), source if isinstance(uav, str) and uav else None,
                                      name="completion_uav"),
            "host_on_time": masked(task.get("host_confirmation_before_deadline"),
                                   isinstance(task.get("host_confirmation_before_deadline"), bool),
                                   source if isinstance(task.get("host_confirmation_before_deadline"), bool) else None,
                                   name="host_on_time"),
            "host_confirmation_time": masked(float(host) if isinstance(host, (int, float)) and not isinstance(host, bool) else None,
                                             isinstance(host, (int, float)) and not isinstance(host, bool),
                                             source if isinstance(host, (int, float)) and not isinstance(host, bool) else None,
                                             name="host_confirmation_time"),
        }
    return result


def load_oracle() -> tuple[list[dict[str, Any]], collections.Counter[str]]:
    records: list[dict[str, Any]] = []
    counts: collections.Counter[str] = collections.Counter()
    for summaries_path, decisions_path in ORACLE_SOURCES:
        decisions = {
            (str(row["parent"]), int(row["repeat"])): row
            for _, row in read_jsonl(decisions_path)
        }
        for line_number, summary in read_jsonl(summaries_path):
            identity = summary["identity"]
            parent, repeat, action = str(identity["parent"]), int(identity["repeat"]), int(identity["action"])
            decision = decisions[(parent, repeat)]
            assignment = identity["assignment"]
            candidate = {
                "action": action,
                "kind": assignment["kind"],
                "uav_id": assignment.get("uav_id"),
                "task_id": assignment.get("task_id"),
            }
            source = f"{summaries_path}:{line_number}"
            tasks = oracle_task_labels(summary["tasks"], source)
            target = {
                "task_id": candidate["task_id"],
                "physical_on_time": masked(None, False, None, name="target.physical_on_time"),
                "completed_by_current_uav": masked(None, False, None, name="target.completed_by_current_uav"),
                "completion_time": masked(None, False, None, name="target.completion_time"),
                "host_on_time": masked(None, False, None, name="target.host_on_time"),
            }
            if action != NOOP:
                task = tasks[candidate["task_id"]]
                raw_task = summary["tasks"][candidate["task_id"]]
                physical = raw_task.get("physical_arrival_time")
                physical_valid = isinstance(physical, (int, float)) and not isinstance(physical, bool)
                target = {
                    "task_id": candidate["task_id"],
                    "physical_on_time": dict(task["physical_on_time"]),
                    "completed_by_current_uav": masked(
                        task["completion_uav"]["value"] == candidate["uav_id"] if task["completion_uav"]["valid"] else None,
                        task["completion_uav"]["valid"], source if task["completion_uav"]["valid"] else None,
                        name="target.completed_by_current_uav",
                    ),
                    "completion_time": masked(float(physical) - float(decision["time"]) if physical_valid else None,
                                              physical_valid, source if physical_valid else None,
                                              name="target.completion_time"),
                    "host_on_time": dict(task["host_on_time"]),
                }
            labels = {
                "first_command_state": masked(command_class(str(summary["first_feedback"])), True, source,
                                              name="first_command_state"),
                "target_task": target,
                "tasks": tasks,
                "energy_to_terminal": masked(float(summary["energy_used"]), True, source,
                                             name="energy_to_terminal"),
                "remaining_utility": masked(float(summary["discounted_utility"]), True, source,
                                            name="remaining_utility"),
                "terminal_time": masked(float(summary["final_time"]), True, source, name="terminal_time"),
            }
            record = {
                "schema": SCHEMA,
                "decision_id": f"oracle:{parent}:r{repeat}",
                "source": {
                    "kind": "sealed_true_branch",
                    "path": str(summaries_path),
                    "line": line_number,
                    "parent": parent,
                    "repeat": repeat,
                    "method": "hungarian-prefix-and-continuation",
                    "step": int(decision["decision_step"]),
                    "decision_time": float(decision["time"]),
                    "split_role": "development",
                    "independent_validation": False,
                },
                "input": {
                    "current_public": {
                        "context_id": str(decision["public_state_sha256"]),
                        "observation_time": float(decision["time"]),
                        "payload_sha256": str(decision["public_state_sha256"]),
                    },
                    "public_history": [],
                    "own_action_history": [],
                },
                "candidate": candidate,
                "legal_actions": [int(value) for value in decision["effective_legal_actions"]],
                "continuation_id": "hungarian-v1-fixed",
                "provenance": {"is_executed": True, "is_true_branch": True,
                               "label_scope": "fixed_hungarian_true_branch"},
                "labels": labels,
                "training_eligible": False,
                "eligibility_reasons": [
                    "development_parent_only",
                    "sealed_first-window_oracle_route",
                    "failed_prefrozen_parent_coverage_gate",
                    "public_history_not_materialized_in_branch_summary",
                ],
            }
            validate_record(record)
            records.append(record)
            counts["oracle_branch_records"] += 1
            counts[f"oracle_first_command:{labels['first_command_state']['value']}"] += 1
            counts["oracle_host_unknown_tasks"] += sum(
                not task["host_on_time"]["valid"] for task in tasks.values()
            )
            for task in tasks.values():
                counts[f"oracle_task_state:{task['state']['value']}"] += 1
    return records, counts


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical(value) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contexts, factual_records, factual_counts = load_factual()
    oracle_records, oracle_counts = load_oracle()
    all_records = factual_records + oracle_records
    continuation_counts = collections.Counter(record["continuation_id"] for record in all_records)
    availability = {
        "schema": "w1-action-outcome-data-availability/1.0.0",
        "development_only": True,
        "independent_validation_parents": 0,
        "training_eligible_records": sum(record["training_eligible"] for record in all_records),
        "factual": dict(sorted(factual_counts.items())),
        "sealed_oracle": dict(sorted(oracle_counts.items())),
        "continuation_identities": dict(sorted(continuation_counts.items())),
        "records": len(all_records),
        "contexts": len(contexts),
        "label_boundary": {
            "factual_executed": "labels apply only to the saved factual continuation",
            "factual_unexecuted": "all labels unknown",
            "oracle": "labels apply only to hungarian-v1-fixed at the sealed first qualifying window",
        },
        "decision": "NOT_TRAINING_READY",
        "reasons": [
            "no independent parent split",
            "general legal candidates lack fixed-continuation labels",
            "factual continuations differ by controller and visited state",
            "sealed oracle covers a failed first-window mechanism and is development-only",
        ],
        "calls": {"environment": 0, "model": 0, "checkpoint": 0, "training": 0},
    }
    write_jsonl(output / "public-contexts.jsonl", contexts)
    write_jsonl(output / "diagnostic-records.jsonl", all_records)
    (output / "data-availability.json").write_text(json.dumps(availability, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    print(json.dumps(availability, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
