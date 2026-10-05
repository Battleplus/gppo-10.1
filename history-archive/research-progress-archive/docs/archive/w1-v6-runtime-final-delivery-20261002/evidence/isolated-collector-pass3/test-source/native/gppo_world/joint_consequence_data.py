"""Loader and audit helpers for the joint task-set consequence pilot."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .graph5 import Graph5Snapshot, graph5_from_dict
from .joint_consequence import JOINT_CONSEQUENCE_SCHEMA, _outcomes, _task_set


@dataclass(frozen=True)
class JointConsequenceExample:
    graph: Graph5Snapshot
    action: int
    task_set_size: int
    new_on_time_count: float | None
    deadline_failure_count: float | None
    mask: bool
    parent_episode_id: str
    prefix_id: str
    repeat_id: str
    reference_action: int


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def example_from_dict(record: Mapping[str, Any], *, expected_horizon_steps: int | None = None) -> JointConsequenceExample:
    if record.get("schema") != JOINT_CONSEQUENCE_SCHEMA:
        raise ValueError("joint consequence schema mismatch")
    forbidden = {"hidden_state", "simulator_state", "future_graph", "next_graph", "future_events"}
    if forbidden.intersection(record):
        raise ValueError("hidden or future fields are forbidden in model input")
    graph = graph5_from_dict(record["graph5_t"])
    target = record["target"]
    action = int(target["action"])
    if not bool(graph.action_mask[action]):
        raise ValueError("candidate action is not legal in the public Graph-5 mask")
    horizon = int(target.get("observation_window_steps", target.get("prediction_horizon_steps", 0)))
    if expected_horizon_steps is not None and horizon != expected_horizon_steps:
        raise ValueError("joint consequence horizon mismatch")
    outcomes = _outcomes(record)
    task_set = _task_set(record, outcomes)
    mask = bool(record.get("label_masks", {}).get("new_on_time_count", False))
    value = target.get("new_on_time_count")
    failures = target.get("deadline_failure_count")
    if mask and (value is None or failures is None):
        raise ValueError("unmasked joint labels must have both aggregate targets")
    if not mask and (value is not None or failures is not None):
        raise ValueError("masked joint labels must use null aggregate targets")
    return JointConsequenceExample(
        graph=graph,
        action=action,
        task_set_size=len(task_set),
        new_on_time_count=None if value is None else float(value),
        deadline_failure_count=None if failures is None else float(failures),
        mask=mask,
        parent_episode_id=str(record["parent_episode_id"]),
        prefix_id=str(record["prefix_id"]),
        repeat_id=str(record["repeat_id"]),
        reference_action=int(target["reference_action"]),
    )


def load_joint_jsonl(path: str | Path, *, expected_horizon_steps: int | None = None) -> list[JointConsequenceExample]:
    examples = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            examples.append(example_from_dict(json.loads(line), expected_horizon_steps=expected_horizon_steps))
        except (TypeError, KeyError, ValueError, IndexError) as exc:
            raise ValueError(f"invalid joint consequence record at line {line_number}: {exc}") from exc
    if not examples:
        raise ValueError(f"empty joint consequence split: {path}")
    return examples


def audit_joint_manifest(manifest: Mapping[str, Any], data_dir: str | Path) -> dict[str, Any]:
    errors: list[str] = []
    files = manifest.get("files", {})
    observed: dict[tuple[str, str], str] = {}
    records_by_split: dict[str, int] = {}
    for split in ("train", "validation"):
        spec = files.get(split)
        if not isinstance(spec, Mapping):
            errors.append(f"missing manifest file for {split}")
            continue
        path = (Path(data_dir) / str(spec.get("path", ""))).resolve()
        try:
            path.relative_to(Path(data_dir).resolve())
        except ValueError:
            errors.append(f"split path escapes data directory: {path}")
            continue
        if not path.is_file():
            errors.append(f"missing split file: {path}")
            continue
        actual_hash = file_sha256(path)
        if actual_hash != str(spec.get("sha256")):
            errors.append(f"sha256 mismatch for {split}")
        examples = load_joint_jsonl(path, expected_horizon_steps=6)
        records_by_split[split] = len(examples)
        for item in examples:
            key = (item.parent_episode_id, item.prefix_id)
            old = observed.setdefault(key, split)
            if old != split:
                errors.append(f"parent/prefix crosses splits: {key}")
    return {"passed": not errors, "errors": errors, "records_by_split": records_by_split}


__all__ = ["JointConsequenceExample", "audit_joint_manifest", "example_from_dict", "file_sha256", "load_joint_jsonl"]
