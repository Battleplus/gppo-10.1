"""Strict loader and split audit for action-consequence supervision."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .consequence_model import ConsequenceTarget
from .data import graph_from_dict
from .dataset import sha256_file
from .graph5 import graph5_from_dict


CONSEQUENCE_SPLITS = ("train", "validation", "test", "ood")
FORBIDDEN_ONLINE_FIELDS = frozenset(
    {
        "graph_tp1",
        "next_graph",
        "future_graph",
        "hidden_state",
        "simulator_state",
        "future_events",
        "full_environment_state",
    }
)


@dataclass(frozen=True)
class ConsequenceExample:
    graph: Any
    target: ConsequenceTarget
    history: torch.Tensor | None = None
    parent_episode_id: str = ""
    prefix_id: str = ""
    masks: Mapping[str, bool] | None = None


def _identity_text(record: Mapping[str, Any], *, parent_episode_id: str, prefix_id: str) -> str:
    target = record["target"]
    identity = {
        "parent_episode_id": parent_episode_id,
        "prefix_id": prefix_id,
        "episode_id": str(target["episode_id"]),
        "decision_index": int(target["decision_index"]),
        "action": int(target["action"]),
        "exogenous_key": str(target["exogenous_key"]),
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def record_identity_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    identities = []
    for record in records:
        parent = str(record.get("parent_episode_id", record.get("episode_id", "")))
        prefix = str(record.get("prefix_id", ""))
        if not parent or not prefix:
            raise ValueError("each record requires parent_episode_id and prefix_id")
        identities.append(_identity_text(record, parent_episode_id=parent, prefix_id=prefix))
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate consequence record identity")
    payload = "\n".join(sorted(identities)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def example_from_dict(
    record: Mapping[str, Any],
    *,
    expected_horizon_steps: int | None = None,
    strict_identity: bool = False,
) -> ConsequenceExample:
    forbidden = sorted(FORBIDDEN_ONLINE_FIELDS.intersection(record))
    if forbidden:
        raise ValueError(f"future or hidden fields are forbidden in online input: {forbidden}")
    graph_key = "graph5_t" if "graph5_t" in record else "graph_t"
    if graph_key not in record or "target" not in record:
        raise ValueError("each consequence record requires graph5_t (or legacy graph_t) and target")
    parent_episode_id = str(record.get("parent_episode_id", record.get("episode_id", "")))
    prefix_id = str(record.get("prefix_id", ""))
    if strict_identity and (not parent_episode_id or not prefix_id):
        raise ValueError("strict consequence records require parent_episode_id and prefix_id")
    if not parent_episode_id:
        raise ValueError("parent_episode_id is required")
    if not prefix_id:
        prefix_id = f"{parent_episode_id}:{record['target'].get('decision_index', '')}"
    graph = graph5_from_dict(record[graph_key]) if graph_key == "graph5_t" else graph_from_dict(dict(record[graph_key]))
    target = ConsequenceTarget(**dict(record["target"]))
    target.validate(expected_horizon_steps=expected_horizon_steps)
    if target.action >= graph.num_actions or not bool(graph.action_mask[target.action].item()):
        raise ValueError("counterfactual action must be legal in the public graph snapshot")
    history_value = record.get("history")
    history = None
    if history_value is not None:
        history = torch.tensor(history_value, dtype=torch.float32)
        if history.ndim != 1 or not torch.isfinite(history).all():
            raise ValueError("history must be a finite one-dimensional visible vector")
    masks_value = record.get("label_masks", {})
    if not isinstance(masks_value, Mapping):
        raise ValueError("label_masks must be an object")
    masks = {name: bool(masks_value.get(name, True)) for name in ("travel_time", "service_progress", "energy_delta", "deadline_risk")}
    if target.episode_id != str(record.get("episode_id", target.episode_id)):
        raise ValueError("record and target episode identities disagree")
    return ConsequenceExample(
        graph=graph,
        target=target,
        history=history,
        parent_episode_id=parent_episode_id,
        prefix_id=prefix_id,
        masks=masks,
    )


def load_consequence_jsonl(
    path: str | Path,
    *,
    expected_horizon_steps: int | None = None,
    strict_identity: bool = False,
) -> list[ConsequenceExample]:
    examples: list[ConsequenceExample] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                examples.append(
                    example_from_dict(
                        json.loads(line),
                        expected_horizon_steps=expected_horizon_steps,
                        strict_identity=strict_identity,
                    )
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"invalid consequence record at {path}:{line_number}: {exc}") from exc
    if not examples:
        raise ValueError(f"consequence split is empty: {path}")
    identities = [(item.parent_episode_id, item.prefix_id, item.target.action) for item in examples]
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate parent/prefix/action identity in {path}")
    return examples


def audit_consequence_manifest(manifest: Mapping[str, Any], data_dir: str | Path) -> dict[str, Any]:
    errors: list[str] = []
    episodes = list(manifest.get("episodes", ()))
    groups: dict[tuple[str, str, int], set[str]] = {}
    for item in episodes:
        split = str(item.get("split"))
        if split not in CONSEQUENCE_SPLITS:
            errors.append(f"unknown split {split!r}")
            continue
        try:
            key = (str(item["scenario_id"]), str(item["tape_id"]), int(item["seed"]))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid episode grouping: {exc}")
            continue
        groups.setdefault(key, set()).add(split)
    overlaps = {
        "/".join((scenario, tape, str(seed))): sorted(splits)
        for (scenario, tape, seed), splits in groups.items()
        if len(splits) > 1
    }
    if overlaps:
        errors.append("scenario/tape/seed groups cross splits")
    files = manifest.get("files", {})
    if not isinstance(files, Mapping):
        errors.append("manifest.files must be a mapping keyed by split")
        files = {}
    resolved_dir = Path(data_dir)
    file_hashes: dict[str, str] = {}
    file_records: dict[str, int] = {}
    file_identity_hashes: dict[str, str] = {}
    observed_groups: dict[tuple[str, str], set[str]] = {}
    expected_horizon = manifest.get("prediction_horizon_steps")
    for split in CONSEQUENCE_SPLITS:
        spec = files.get(split)
        if not isinstance(spec, Mapping):
            errors.append(f"manifest.files[{split!r}] must be an object with path/sha256/records/identity_sha256")
            errors.append(f"missing manifest file for {split}")
            continue
        required = ("path", "sha256", "records", "identity_sha256")
        missing = [key for key in required if key not in spec]
        if missing:
            errors.append(f"manifest.files[{split!r}] missing fields: {missing}")
            continue
        relative = str(spec["path"])
        path = (resolved_dir / relative).resolve()
        try:
            path.relative_to(resolved_dir.resolve())
        except ValueError:
            errors.append(f"split path escapes data directory: {relative}")
            continue
        if not path.is_file():
            errors.append(f"missing split file: {path}")
            continue
        file_hashes[split] = sha256_file(path)
        if str(spec["sha256"]) != file_hashes[split]:
            errors.append(f"sha256 mismatch for {split}")
        raw_records: list[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, Mapping):
                        raise ValueError("record is not an object")
                    example_from_dict(
                        raw,
                        expected_horizon_steps=int(expected_horizon) if expected_horizon is not None else None,
                        strict_identity=True,
                    )
                    raw_records.append(raw)
                    group = (str(raw["parent_episode_id"]), str(raw["prefix_id"]))
                    observed_groups.setdefault(group, set()).add(split)
                except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    errors.append(f"invalid {split} record at line {line_number}: {exc}")
        file_records[split] = len(raw_records)
        if int(spec["records"]) != len(raw_records):
            errors.append(f"record count mismatch for {split}")
        try:
            identity_hash = record_identity_sha256(raw_records)
            file_identity_hashes[split] = identity_hash
            if str(spec["identity_sha256"]) != identity_hash:
                errors.append(f"record identity hash mismatch for {split}")
        except ValueError as exc:
            errors.append(f"identity audit failed for {split}: {exc}")
    observed_overlaps = {
        f"{parent}/{prefix}": sorted(splits)
        for (parent, prefix), splits in observed_groups.items()
        if len(splits) > 1
    }
    if observed_overlaps:
        errors.append("parent episode/prefix groups cross splits")
    return {
        "passed": not errors,
        "errors": errors,
        "episode_count": len(episodes),
        "group_count": len(groups),
        "split_overlap_count": len(overlaps),
        "split_overlaps": overlaps,
        "file_sha256": file_hashes,
        "file_records": file_records,
        "file_identity_sha256": file_identity_hashes,
        "observed_split_overlaps": observed_overlaps,
    }


def target_tensors(examples: Iterable[ConsequenceExample], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    targets = list(examples)
    if not targets:
        raise ValueError("at least one consequence example is required")
    return {
        name: torch.tensor([getattr(item.target, name) for item in targets], dtype=torch.float32, device=device)
        for name in ("travel_time", "service_progress", "energy_delta", "deadline_risk")
    }
