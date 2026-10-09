"""Tensor loading for versioned GPPO world-model JSONL datasets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import torch

from .contracts import GraphSnapshot, TRUTH_ONLY_FIELD_NAMES
from .dataset import SPLITS, audit_manifest, sha256_file
from .registry import FEATURE_REGISTRY, SCHEMA_VERSION


COST_NAMES = (
    "uncovered",
    "distance",
    "load_gap",
    "switches",
    "recovery_delay",
    "constraint_violation",
    "total",
)
NODE_ORDER = ("uav", "region", "target")
RELATION_ORDER = tuple(FEATURE_REGISTRY.edges)
NODE_COUNTS = {"uav": 4, "region": 4, "target": 3}
EDGE_COUNTS = {
    ("uav", "can_serve", "region"): 16,
    ("region", "served_by", "uav"): 16,
    ("region", "adjacent", "region"): 8,
    ("target", "located_in", "region"): 3,
    ("region", "contains", "target"): 3,
    ("uav", "tracks", "target"): 12,
    ("target", "tracked_by", "uav"): 12,
    ("uav", "communicates", "uav"): 12,
}
STATE_DIM = sum(NODE_COUNTS[name] * FEATURE_REGISTRY.node_dimensions[name] for name in NODE_ORDER) + sum(
    EDGE_COUNTS[relation] * FEATURE_REGISTRY.edge_dimensions[relation] for relation in RELATION_ORDER
)


def graph_from_dict(value: dict) -> GraphSnapshot:
    return GraphSnapshot(
        nodes={name: torch.tensor(tensor, dtype=torch.float32) for name, tensor in value["nodes"].items()},
        edge_index={
            tuple(name.split("/")): torch.tensor(tensor, dtype=torch.long)
            for name, tensor in value["edge_index"].items()
        },
        edge_attr={
            tuple(name.split("/")): torch.tensor(tensor, dtype=torch.float32)
            for name, tensor in value["edge_attr"].items()
        },
        candidate_edges=torch.tensor(value["candidate_edges"], dtype=torch.long),
        action_mask=torch.tensor(value["action_mask"], dtype=torch.bool),
        graph_version=int(value["graph_version"]),
    )


def state_vector(graph: GraphSnapshot) -> torch.Tensor:
    nodes = [graph.nodes[name].reshape(-1) for name in NODE_ORDER]
    edges = [graph.edge_attr[relation].reshape(-1) for relation in RELATION_ORDER]
    return torch.cat([*nodes, *edges], dim=0)


@dataclass(frozen=True)
class TensorTransition:
    episode_id: str
    step: int
    graph: GraphSnapshot
    next_graph: GraphSnapshot
    evidence: tuple[dict, ...]
    action: int
    reward: float
    costs: torch.Tensor
    continuation: float
    scenario_id: str = ""
    decision_time: float = 0.0
    action_version: int = 0
    execution_accepted: bool = True
    execution_status: str = "committed"

    @property
    def target_delta(self) -> torch.Tensor:
        return state_vector(self.next_graph) - state_vector(self.graph)


def transition_from_dict(value: dict) -> TensorTransition:
    execution = value["execution"]
    action = execution["executed_action"]
    return TensorTransition(
        episode_id=str(value["episode_id"]),
        step=int(value["step"]),
        graph=graph_from_dict(value["graph_t"]),
        next_graph=graph_from_dict(value["graph_tp1"]),
        evidence=tuple(dict(item) for item in value.get("evidence_t", ())),
        action=17 if action is None else int(action),
        reward=float(value["reward"]),
        costs=torch.tensor([float(value["costs"].get(name, 0.0)) for name in COST_NAMES]),
        continuation=float(bool(value["continuation"])),
        scenario_id=str(value.get("scenario_id", "")),
        decision_time=float(value.get("decision_time", 0.0)),
        action_version=int(execution.get("action_version", 0)),
        execution_accepted=bool(execution.get("accepted", False)),
        execution_status=str(execution.get("status", "")),
    )


def load_jsonl(path: str | Path) -> list[TensorTransition]:
    transitions: list[TensorTransition] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                transitions.append(transition_from_dict(json.loads(line)))
    return transitions


def group_episodes(transitions: Iterable[TensorTransition]) -> list[list[TensorTransition]]:
    episodes: list[list[TensorTransition]] = []
    current: list[TensorTransition] = []
    current_id: str | None = None
    closed_ids: set[str] = set()
    for transition in transitions:
        if transition.episode_id != current_id:
            if current:
                episodes.append(current)
                closed_ids.add(str(current_id))
            if transition.episode_id in closed_ids:
                raise ValueError(f"episode {transition.episode_id} is non-contiguous in the file")
            current = []
            current_id = transition.episode_id
        if transition.step != len(current):
            raise ValueError(f"non-contiguous episode {transition.episode_id}")
        current.append(transition)
    if current:
        episodes.append(current)
    return episodes


def _truth_only_paths(value, path: str = "root") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).strip().lower() in TRUTH_ONLY_FIELD_NAMES:
                found.append(child_path)
            found.extend(_truth_only_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_truth_only_paths(child, f"{path}[{index}]"))
    return found


def audit_training_inputs(manifest_path: str | Path, dataset_dir: str | Path) -> dict:
    """Revalidate T-01 hashes, group isolation and causal online fields before training."""

    manifest_file = Path(manifest_path).resolve()
    dataset_root = Path(dataset_dir).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    structural = audit_manifest(manifest)
    errors = list(structural["errors"])
    seen_episode_split: dict[str, str] = {}
    actual_groups = {split: set() for split in SPLITS}
    observed_counts = {split: 0 for split in SPLITS}
    truth_only_count = 0
    illegal_execution_count = 0
    graph_version_mismatch_count = 0
    for split in SPLITS:
        path = dataset_root / f"{split}.jsonl"
        record = manifest.get("files", {}).get(split, {})
        if not path.is_file():
            errors.append(f"missing {split} JSONL")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != record.get("sha256"):
            errors.append(f"{split} sha256 mismatch")
        expected_steps: dict[str, int] = {}
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                observed_counts[split] += 1
                episode_id = str(row["episode_id"])
                previous_split = seen_episode_split.setdefault(episode_id, split)
                if previous_split != split:
                    errors.append(f"episode {episode_id} occurs in {previous_split} and {split}")
                if not episode_id.startswith(f"{split}/"):
                    errors.append(f"{split}:{line_number} episode prefix mismatch")
                expected_step = expected_steps.get(episode_id, 0)
                if int(row["step"]) != expected_step:
                    errors.append(f"{split}:{line_number} non-contiguous episode step")
                expected_steps[episode_id] = expected_step + 1
                if row.get("schema_version") != SCHEMA_VERSION:
                    errors.append(f"{split}:{line_number} schema mismatch")
                decision_time = float(row["decision_time"])
                if float(row["next_decision_time"]) < decision_time:
                    errors.append(f"{split}:{line_number} next decision precedes current decision")
                actual_groups[split].add((str(row["scenario_id"]), str(row["tape_id"]), int(row["seed"])))
                for evidence in row.get("evidence_t", []):
                    if float(evidence["received_at"]) > decision_time:
                        errors.append(f"{split}:{line_number} future evidence")
                    paths = _truth_only_paths(evidence.get("payload", {}), "evidence.payload")
                    truth_only_count += len(paths)
                    errors.extend(f"{split}:{line_number} forbidden {item}" for item in paths)
                execution = row["execution"]
                if int(execution["graph_version"]) != int(row["graph_t"]["graph_version"]):
                    graph_version_mismatch_count += 1
                    errors.append(f"{split}:{line_number} execution graph version mismatch")
                action = execution.get("executed_action")
                accepted = bool(execution.get("accepted"))
                mask = row["graph_t"]["action_mask"]
                legal = action is not None and 0 <= int(action) < len(mask) and bool(mask[int(action)])
                if (accepted and not legal) or (not accepted and action is not None):
                    illegal_execution_count += 1
                    errors.append(f"{split}:{line_number} invalid executed action")
        if observed_counts[split] != int(record.get("transitions", -1)):
            errors.append(f"{split} transition count mismatch")
    overlaps = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            shared = actual_groups[left] & actual_groups[right]
            if shared:
                overlaps[f"{left}/{right}"] = len(shared)
    if overlaps:
        errors.append("actual scenario/tape/seed groups cross splits")
    checkpoint = manifest.get("gppo_behavior_checkpoint", {})
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    checkpoint_ok = checkpoint_path.is_file() and sha256_file(checkpoint_path) == checkpoint.get("sha256")
    if not checkpoint_ok:
        errors.append("GPPO behavior checkpoint missing or hash mismatch")
    return {
        "passed": not errors,
        "errors": errors,
        "manifest_path": str(manifest_file),
        "manifest_sha256": sha256_file(manifest_file),
        "manifest_audit": structural,
        "observed_transitions": observed_counts,
        "actual_split_overlap_count": sum(overlaps.values()),
        "actual_split_overlaps": overlaps,
        "truth_only_online_field_count": truth_only_count,
        "illegal_execution_count": illegal_execution_count,
        "graph_version_mismatch_count": graph_version_mismatch_count,
        "gppo_checkpoint_verified": checkpoint_ok,
        "gppo_checkpoint_sha256": checkpoint.get("sha256"),
    }


def apply_predicted_delta(graph: GraphSnapshot, delta: torch.Tensor) -> GraphSnapshot:
    """Build the next autoregressive graph while retaining fixed topology."""

    offset = 0
    nodes = {}
    for name in NODE_ORDER:
        current = graph.nodes[name]
        count = current.numel()
        nodes[name] = (current + delta[offset : offset + count].reshape_as(current)).clamp(0.0, 1.0)
        offset += count
    attrs = {}
    for relation in RELATION_ORDER:
        current = graph.edge_attr[relation]
        count = current.numel()
        attrs[relation] = (current + delta[offset : offset + count].reshape_as(current)).clamp(0.0, 1.0)
        offset += count
    if offset != int(delta.numel()):
        raise ValueError(f"delta has {delta.numel()} values, expected {offset}")
    return GraphSnapshot(
        nodes=nodes,
        edge_index=graph.edge_index,
        edge_attr=attrs,
        candidate_edges=graph.candidate_edges,
        action_mask=graph.action_mask,
        graph_version=graph.graph_version + 1,
    )
