"""Leakage-safe labels for candidate actions with task-set consequences.

The simulator supplies one record per legal candidate from the same public
prefix.  This module only builds labels from explicitly supplied branch
outcomes; it never infers a task denominator from a candidate branch and never
turns an unobserved outcome into failure.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .graph5 import graph5_from_dict


JOINT_CONSEQUENCE_SCHEMA = "gppo-arrival-joint-consequence/v1"
DEFAULT_CONTINUATION_CONTROLLER = "frozen_public_controller_v1"


@dataclass(frozen=True)
class JointConsequenceSummary:
    complete: bool
    new_on_time_count: int | None
    deadline_failure_count: int | None
    censored_task_ids: tuple[str, ...]
    observed_task_count: int


def _outcomes(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = record.get("joint_task_outcomes", record.get("task_outcomes"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("joint_task_outcomes is required; candidate-only legacy labels cannot define a task-set target")
    result: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping) or not item.get("task_id"):
            raise ValueError("each joint task outcome must have a task_id")
        task_id = str(item["task_id"])
        if task_id in result:
            raise ValueError(f"duplicate task outcome {task_id}")
        observed = item.get("deadline_observed")
        on_time = item.get("arrival_before_deadline_physical")
        if not isinstance(observed, bool):
            raise ValueError(f"deadline_observed must be boolean for {task_id}")
        if observed and not isinstance(on_time, bool):
            raise ValueError(f"observed deadline outcome must be boolean for {task_id}")
        if not observed and on_time is not None:
            raise ValueError(f"censored outcome cannot carry a deadline label for {task_id}")
        result[task_id] = item
    return result


def _task_set(record: Mapping[str, Any], outcomes: Mapping[str, Any]) -> tuple[str, ...]:
    raw = record.get("joint_task_set")
    if not isinstance(raw, list) or not raw:
        raise ValueError("joint_task_set must be explicit and non-empty")
    ids = tuple(str(item) for item in raw)
    if len(ids) != len(set(ids)):
        raise ValueError("joint_task_set contains duplicate task IDs")
    if set(ids) != set(outcomes):
        raise ValueError("joint_task_set must equal the task IDs in every branch outcome")
    return ids


def validate_candidate_group(records: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Validate candidate identity, legality and fixed task-set membership."""
    items = tuple(records)
    if not items:
        raise ValueError("empty candidate group")
    identities = {
        (str(item.get("parent_episode_id")), str(item.get("prefix_id")), str(item.get("repeat_id", "0")))
        for item in items
    }
    if len(identities) != 1:
        raise ValueError("all candidates must share one parent episode and prefix")
    actions: set[int] = set()
    task_set: tuple[str, ...] | None = None
    normalized: list[dict[str, Any]] = []
    for item in items:
        if "graph5_t" not in item or "target" not in item:
            raise ValueError("candidate record must contain graph5_t and target")
        graph = graph5_from_dict(item["graph5_t"])
        action = int(item["target"]["action"])
        if action in actions:
            raise ValueError(f"duplicate candidate action {action}")
        if action < 0 or action >= graph.num_actions or not bool(graph.action_mask[action]):
            raise ValueError(f"candidate action {action} is not legal in the public Graph-5 mask")
        outcomes = _outcomes(item)
        current_set = _task_set(item, outcomes)
        if task_set is None:
            task_set = current_set
        elif current_set != task_set:
            raise ValueError("all candidates in a prefix must use the same ordered joint_task_set")
        actions.add(action)
        normalized.append(dict(item))
    return tuple(normalized)


def _summarize(outcomes: Mapping[str, Mapping[str, Any]], task_set: tuple[str, ...]) -> JointConsequenceSummary:
    censored = tuple(task_id for task_id in task_set if not bool(outcomes[task_id]["deadline_observed"]))
    observed = len(task_set) - len(censored)
    if censored:
        return JointConsequenceSummary(False, None, None, censored, observed)
    on_time = sum(bool(outcomes[task_id]["arrival_before_deadline_physical"]) for task_id in task_set)
    failures = len(task_set) - on_time
    return JointConsequenceSummary(True, on_time, failures, (), observed)


def build_joint_target_record(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    observation_window_steps: int,
    continuation_controller: str = DEFAULT_CONTINUATION_CONTROLLER,
    reference_action: int | None = None,
) -> dict[str, Any]:
    """Build one candidate label against a preselected legal reference action."""
    if observation_window_steps <= 0:
        raise ValueError("observation_window_steps must be positive")
    candidate_action = int(candidate["target"]["action"])
    reference_action_value = int(reference["target"]["action"])
    if candidate_action == reference_action_value:
        if dict(candidate) != dict(reference):
            raise ValueError("same-action candidate/reference must be the same branch record")
        candidate_group = validate_candidate_group((candidate,))
    else:
        candidate_group = validate_candidate_group((candidate, reference))
    candidate_outcomes = _outcomes(candidate)
    reference_outcomes = _outcomes(reference)
    task_set = _task_set(candidate, candidate_outcomes)
    if _task_set(reference, reference_outcomes) != task_set:
        raise ValueError("candidate/reference task sets differ")
    ref_action = int(reference["target"]["action"]) if reference_action is None else int(reference_action)
    if ref_action != int(reference["target"]["action"]):
        raise ValueError("reference_action does not match the supplied reference branch")
    candidate_repeat = str(candidate.get("repeat_id", "0"))
    reference_repeat = str(reference.get("repeat_id", "0"))
    if candidate_repeat != reference_repeat:
        raise ValueError("candidate/reference branches must share one exogenous repeat")
    candidate_summary = _summarize(candidate_outcomes, task_set)
    reference_summary = _summarize(reference_outcomes, task_set)
    delta_mask = candidate_summary.complete and reference_summary.complete
    target = {
        "schema": JOINT_CONSEQUENCE_SCHEMA,
        "parent_episode_id": str(candidate["parent_episode_id"]),
        "prefix_id": str(candidate["prefix_id"]),
        "repeat_id": candidate_repeat,
        "action": candidate_action,
        "reference_action": ref_action,
        "fixed_task_set": list(task_set),
        "observation_window_steps": int(observation_window_steps),
        "continuation_controller": str(continuation_controller),
        "new_on_time_count": candidate_summary.new_on_time_count,
        "deadline_failure_count": candidate_summary.deadline_failure_count,
        "delta_new_on_time_count_vs_reference": (
            candidate_summary.new_on_time_count - reference_summary.new_on_time_count if delta_mask else None
        ),
        "delta_deadline_failure_count_vs_reference": (
            candidate_summary.deadline_failure_count - reference_summary.deadline_failure_count if delta_mask else None
        ),
        "label_source": "simulator-counterfactual",
    }
    return {
        "schema": JOINT_CONSEQUENCE_SCHEMA,
        "parent_episode_id": target["parent_episode_id"],
        "prefix_id": target["prefix_id"],
        "repeat_id": candidate_repeat,
        "exogenous_key": str(candidate.get("exogenous_key", "")),
        "graph5_t": candidate["graph5_t"],
        "joint_task_set": list(task_set),
        "joint_task_outcomes": [dict(candidate_outcomes[task_id]) for task_id in task_set],
        "reference_task_outcomes": [dict(reference_outcomes[task_id]) for task_id in task_set],
        "target": target,
        "label_masks": {
            "new_on_time_count": candidate_summary.complete,
            "deadline_failure_count": candidate_summary.complete,
            "delta_vs_reference": delta_mask,
        },
        "censoring": {
            "candidate_task_ids": list(candidate_summary.censored_task_ids),
            "reference_task_ids": list(reference_summary.censored_task_ids),
            "candidate_observed_task_count": candidate_summary.observed_task_count,
            "reference_observed_task_count": reference_summary.observed_task_count,
        },
        "label_provenance": {
            "label_source": "simulator-counterfactual",
            "hidden_state_online": False,
            "shared_exogenous_randomness": candidate.get("exogenous_key") == reference.get("exogenous_key"),
            "candidate_branch_digest": candidate.get("label_provenance", {}).get("branch_trace_sha256"),
            "prefix_trace_digest": candidate.get("label_provenance", {}).get("prefix_trace_sha256"),
            "reference_branch_digest": reference.get("label_provenance", {}).get("branch_trace_sha256"),
        },
        "branch_ledger": candidate.get("branch_ledger", {}),
    }


def build_group_targets(
    records: Iterable[Mapping[str, Any]],
    *,
    reference_action: int,
    observation_window_steps: int,
    continuation_controller: str = DEFAULT_CONTINUATION_CONTROLLER,
) -> list[dict[str, Any]]:
    """Convert an existing generator group into candidate-wise joint labels."""
    valid = validate_candidate_group(records)
    references = [item for item in valid if int(item["target"]["action"]) == reference_action]
    if len(references) != 1:
        raise ValueError("reference_action must identify exactly one legal branch in the prefix group")
    reference = references[0]
    return [
        build_joint_target_record(
            item,
            reference,
            observation_window_steps=observation_window_steps,
            continuation_controller=continuation_controller,
            reference_action=reference_action,
        )
        for item in valid
    ]


def group_records(records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, str], tuple[dict[str, Any], ...]]:
    groups: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (str(record.get("parent_episode_id")), str(record.get("prefix_id")), str(record.get("repeat_id", "0")))
        groups[key].append(dict(record))
    return {key: validate_candidate_group(value) for key, value in groups.items()}
