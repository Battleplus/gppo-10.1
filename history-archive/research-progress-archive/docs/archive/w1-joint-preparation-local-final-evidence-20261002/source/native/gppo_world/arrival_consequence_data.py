"""Strict loader and audit helpers for the arrival consequence dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .graph5 import graph5_from_dict


@dataclass(frozen=True)
class ArrivalExample:
    graph: Any
    target: dict[str, Any]
    masks: dict[str, bool]
    parent_episode_id: str
    prefix_id: str
    action: int


def load_arrival_jsonl(path: str | Path, expected_horizon: int) -> list[ArrivalExample]:
    out: list[ArrivalExample] = []
    identities: set[tuple[str, str, int]] = set()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if any(key in raw for key in ("hidden_state", "simulator_state", "future_graph", "next_graph")):
                raise ValueError(f"forbidden future/hidden input at {path}:{line_no}")
            graph = graph5_from_dict(raw["graph5_t"])
            target = dict(raw["target"])
            parent, prefix, action = str(raw["parent_episode_id"]), str(raw["prefix_id"]), int(target["action"])
            identity = (parent, prefix, action)
            if identity in identities:
                raise ValueError(f"duplicate candidate identity {identity}")
            identities.add(identity)
            if int(target["prediction_horizon_steps"]) != expected_horizon:
                raise ValueError(f"horizon mismatch at {path}:{line_no}")
            if not bool(graph.action_mask[action]):
                raise ValueError(f"candidate is not legal at {path}:{line_no}")
            masks = dict(raw.get("label_masks", {}))
            out.append(ArrivalExample(graph, target, {"arrival_time": bool(masks.get("arrival_time", False)), "deadline": bool(masks.get("arrival_before_deadline_physical", False)), "failure": bool(masks.get("execution_or_energy_failure", False))}, parent, prefix, action))
    if not out:
        raise ValueError(f"empty split: {path}")
    return out


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_arrival_manifest(manifest: dict[str, Any], data_dir: str | Path) -> dict[str, Any]:
    errors: list[str] = []
    observed: dict[tuple[str, str], set[str]] = {}
    details: dict[str, Any] = {}
    for split in ("train", "validation", "test", "ood"):
        spec = manifest.get("files", {}).get(split)
        if not isinstance(spec, dict):
            errors.append(f"missing files.{split}")
            continue
        path = Path(data_dir) / str(spec["path"])
        if not path.is_file():
            errors.append(f"missing {path}")
            continue
        actual_sha = file_sha256(path)
        if actual_sha != str(spec["sha256"]):
            errors.append(f"sha256 mismatch for {split}")
        examples = load_arrival_jsonl(path, int(manifest["prediction_horizon_steps"]))
        if len(examples) != int(spec["records"]):
            errors.append(f"record count mismatch for {split}")
        for item in examples:
            observed.setdefault((item.parent_episode_id, item.prefix_id), set()).add(split)
        details[split] = {"records": len(examples), "parents": len({x.parent_episode_id for x in examples}), "prefixes": len({(x.parent_episode_id, x.prefix_id) for x in examples}), "arrival_valid": sum(x.masks["arrival_time"] for x in examples), "deadline_valid": sum(x.masks["deadline"] for x in examples), "failure_valid": sum(x.masks["failure"] for x in examples)}
    overlaps = [key for key, splits in observed.items() if len(splits) > 1]
    if overlaps:
        errors.append(f"parent/prefix groups cross splits: {len(overlaps)}")
    return {"passed": not errors, "errors": errors, "splits": details, "cross_split_parent_prefix": len(overlaps)}
