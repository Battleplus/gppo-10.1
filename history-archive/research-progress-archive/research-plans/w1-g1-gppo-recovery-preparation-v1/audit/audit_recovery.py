#!/usr/bin/env python3
"""Offline recovery audit for the W1 G1 GPPO artifacts.

This file deliberately uses only the Python standard library.  Checkpoints are
read as ZIP members and passed through a fail-closed Unpickler whose only
globals are inert metadata constructors; tensor bytes come directly from ZIP
storage members.  No torch module is imported and no checkpoint code executes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
import sqlite3
import struct
import sys
import zipfile
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path(__file__).resolve().parents[2] / "w1-g1-gppo-phase-accounting-repair-v1"
DEFAULT_OUT = Path(__file__).resolve().parent
RECOVERY_PACKAGE = Path(__file__).resolve().parents[1] / "package"


@dataclass(frozen=True)
class StorageType:
    name: str


@dataclass(frozen=True)
class StorageRef:
    storage_type: StorageType
    key: str
    location: str
    element_count: int


@dataclass(frozen=True)
class TensorRef:
    storage: StorageRef
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


def inert_rebuild_tensor_v2(storage: StorageRef, storage_offset: int,
                            shape: tuple[int, ...], stride: tuple[int, ...],
                            *_ignored: Any) -> TensorRef:
    return TensorRef(storage, int(storage_offset), tuple(map(int, shape)),
                     tuple(map(int, stride)))


class RestrictedCheckpointUnpickler(pickle.Unpickler):
    """Decode primitive checkpoint data; refuse every non-inert global."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return inert_rebuild_tensor_v2
        if module == "torch" and name.endswith("Storage"):
            return StorageType(name)
        raise pickle.UnpicklingError(f"refused pickle global: {module}.{name}")

    def persistent_load(self, persistent_id: Any) -> StorageRef:
        if (not isinstance(persistent_id, tuple) or len(persistent_id) != 5
                or persistent_id[0] != "storage"
                or not isinstance(persistent_id[1], StorageType)):
            raise pickle.UnpicklingError("refused non-storage persistent id")
        _, storage_type, key, location, element_count = persistent_id
        if not isinstance(key, str) or not isinstance(location, str):
            raise pickle.UnpicklingError("invalid storage reference")
        return StorageRef(storage_type, key, location, int(element_count))


DTYPES = {
    "FloatStorage": ("torch.float32", 4, "f"),
    "DoubleStorage": ("torch.float64", 8, "d"),
    "HalfStorage": ("torch.float16", 2, None),
    "BFloat16Storage": ("torch.bfloat16", 2, None),
    "LongStorage": ("torch.int64", 8, "q"),
    "IntStorage": ("torch.int32", 4, "i"),
    "ShortStorage": ("torch.int16", 2, "h"),
    "CharStorage": ("torch.int8", 1, "b"),
    "ByteStorage": ("torch.uint8", 1, "B"),
    "BoolStorage": ("torch.bool", 1, "B"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def tensor_bytes(tensor: TensorRef, archive: zipfile.ZipFile,
                 archive_prefix: str) -> tuple[bytes, str]:
    storage = tensor.storage
    dtype = DTYPES.get(storage.storage_type.name)
    if dtype is None:
        raise ValueError(f"unsupported tensor storage: {storage.storage_type.name}")
    dtype_name, item_size, _struct_format = dtype
    member = f"{archive_prefix}/data/{storage.key}"
    raw_storage = archive.read(member)
    if len(raw_storage) != storage.element_count * item_size:
        raise ValueError(f"storage byte count mismatch: {member}")

    expected_stride = []
    running = 1
    for dimension in reversed(tensor.shape):
        expected_stride.append(running)
        running *= dimension
    expected_stride.reverse()
    contiguous = all(dimension <= 1 or stride == expected
                     for dimension, stride, expected in
                     zip(tensor.shape, tensor.stride, expected_stride))
    if not contiguous:
        raise ValueError(f"non-contiguous policy state tensor: shape={tensor.shape}")
    count = math.prod(tensor.shape)
    begin = tensor.storage_offset * item_size
    end = begin + count * item_size
    if begin < 0 or end > len(raw_storage):
        raise ValueError(f"tensor view exceeds storage: {member}")
    return raw_storage[begin:end], dtype_name


def checkpoint_facts(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        byteorder_names = [name for name in names if name.endswith("/byteorder")]
        if len(byteorder_names) != 1 or archive.read(byteorder_names[0]) != b"little":
            raise ValueError("unsupported or missing checkpoint byte order")
        data_pickle_names = [name for name in names if name.endswith("/data.pkl")]
        if len(data_pickle_names) != 1:
            raise ValueError(f"expected one data.pkl, found {len(data_pickle_names)}")
        pickle_name = data_pickle_names[0]
        pickle_bytes = archive.read(pickle_name)
        if len(pickle_bytes) > 16 * 1024 * 1024:
            raise ValueError("checkpoint metadata exceeds offline parser limit")
        payload = RestrictedCheckpointUnpickler(__import__("io").BytesIO(pickle_bytes)).load()
        if not isinstance(payload, dict):
            raise ValueError("checkpoint root is not a mapping")
        state = payload.get("state_dict")
        if not isinstance(state, (dict, OrderedDict)):
            raise ValueError("state_dict missing or invalid")

        digest = hashlib.sha256()
        state_rows = []
        for name, tensor in sorted(state.items(), key=lambda item: str(item[0])):
            if not isinstance(tensor, TensorRef):
                raise ValueError(f"non-tensor state value: {name}")
            raw, dtype_name = tensor_bytes(tensor, archive, pickle_name.rsplit("/", 1)[0])
            header = json.dumps({"name": str(name), "dtype": dtype_name,
                                 "shape": list(tensor.shape)}, sort_keys=True,
                                separators=(",", ":")).encode("utf-8")
            digest.update(len(header).to_bytes(8, "big"))
            digest.update(header)
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
            state_rows.append({"name": str(name), "dtype": dtype_name,
                               "shape": list(tensor.shape), "storage_key": tensor.storage.key})

        optimizer = payload.get("optimizer_state_dict")
        if not isinstance(optimizer, dict):
            raise ValueError("optimizer_state_dict missing or invalid")
        optimizer_slots = optimizer.get("state", {})
        optimizer_groups = optimizer.get("param_groups", [])
        optimizer_lrs = [group.get("lr") for group in optimizer_groups if isinstance(group, dict)]
        optimizer_group_parameter_counts = [len(group.get("params", [])) for group in optimizer_groups
                                            if isinstance(group, dict)]
        step_values = []
        for slot in optimizer_slots.values():
            step = slot.get("step") if isinstance(slot, dict) else None
            if not isinstance(step, TensorRef):
                continue
            raw, dtype_name = tensor_bytes(step, archive, pickle_name.rsplit("/", 1)[0])
            dtype_def = next((row for name, row in DTYPES.items()
                              if row[0] == dtype_name), None)
            if dtype_def and dtype_def[2] and len(raw) == dtype_def[1]:
                step_values.append(struct.unpack("<" + dtype_def[2], raw)[0])

        return {
            "archive_sha256": sha256_file(path),
            "archive_bytes": path.stat().st_size,
            "byteorder": "little",
            "pickle_globals_allowed": [
                "collections.OrderedDict", "torch._utils._rebuild_tensor_v2 (inert stub)",
                "torch.*Storage (metadata token only)",
            ],
            "top_level_keys": sorted(str(key) for key in payload.keys()),
            "schema": payload.get("schema"),
            "method": payload.get("method"),
            "seed": payload.get("seed"),
            "stored_state_sha256": payload.get("state_dict_sha256"),
            "recomputed_state_sha256": digest.hexdigest(),
            "state_hash_matches_stored": payload.get("state_dict_sha256") == digest.hexdigest(),
            "state_tensor_count": len(state_rows),
            "state_tensors": state_rows,
            "optimizer_state_slot_count": len(optimizer_slots),
            "optimizer_step_values": sorted(set(step_values)),
            "optimizer_lrs": optimizer_lrs,
            "optimizer_group_parameter_counts": optimizer_group_parameter_counts,
            "summary_route": payload.get("summary", {}).get("route"),
            "summary_environment_steps": payload.get("summary", {}).get("environment_steps"),
            "summary_optimizer_updates": payload.get("summary", {}).get("policy_optimizer_updates"),
            "summary_world_model_updates": payload.get("summary", {}).get("world_model_updates"),
            "summary_world_model_frozen": payload.get("summary", {}).get("world_model_frozen"),
            "summary_raw_decision_count": len(payload.get("summary", {}).get("raw_decisions", [])),
            "summary_recurrent_replay_groups": payload.get("summary", {}).get("recurrent_hidden_replay_groups"),
        }


def parse_route_identity(name: str) -> tuple[str, int] | None:
    match = re.search(r"(?<![A-Za-z0-9])(G0|T|G1):(\d{4})(?!\d)", name)
    return (match.group(1), int(match.group(2))) if match else None


def classify_call(name: str) -> str:
    lowered = name.lower()
    if "prior" in lowered:
        return "prior"
    if "ppo_update" in lowered:
        return "update"
    if "checkpoint" in lowered or "load_frozen_world_model" in lowered:
        return "checkpoint"
    if ("_policy_encode" in lowered or "_policy_action_select" in lowered
            or "bootstrap" in lowered or "recurrent_hidden_replay" in lowered):
        return "actor_history_bootstrap"
    if name.startswith("policy_step:") or name.startswith("policy_reset:"):
        return "environment"
    return "other"


def read_calls(path: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='calls'"
        ).fetchone()
        if table is None:
            raise ValueError("SQLite calls table missing")
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(calls)")}
        required = {"id", "stage", "name", "amounts", "status", "started", "finished", "error"}
        if not required <= columns:
            raise ValueError(f"SQLite calls schema missing fields: {sorted(required - columns)}")
        rows = [dict(row) for row in connection.execute(
            "SELECT id,stage,name,amounts,status,started,finished,error FROM calls ORDER BY id"
        )]
        for row in rows:
            row["amounts"] = json.loads(row["amounts"] or "{}")
            row["category"] = classify_call(row["name"])
            row["duration"] = (float(row["finished"]) - float(row["started"])
                               if row["finished"] is not None else None)
            explicit = parse_route_identity(row["name"])
            row["route_explicit"] = explicit
        return rows
    finally:
        connection.close()


def route_timings(calls: list[dict[str, Any]]) -> dict[str, Any]:
    starts: dict[tuple[str, int], dict[str, Any]] = {}
    ends: dict[tuple[str, int], dict[str, Any]] = {}
    current: tuple[str, int] | None = None
    route_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in calls:
        name = row["name"]
        if name.startswith("initialize_policy:"):
            current = parse_route_identity(name)
            if current:
                starts[current] = row
        identity = row["route_explicit"]
        owner = identity or current
        if owner:
            route_rows[owner].append(row)
        if name.startswith("save_policy_checkpoint:"):
            identity = parse_route_identity(name)
            if identity:
                ends[identity] = row
            if identity == current:
                current = None

    summaries: dict[str, Any] = {}
    for identity, rows in route_rows.items():
        start_row = starts.get(identity)
        end_row = ends.get(identity)
        if start_row is None:
            continue
        starts_at = float(start_row["started"])
        if end_row is not None and end_row["finished"] is not None:
            ends_at = float(end_row["finished"])
            end_boundary = "save_policy_checkpoint.finished"
        else:
            all_times = [float(row["started"]) for row in rows]
            all_times.extend(float(row["finished"]) for row in rows if row["finished"] is not None)
            ends_at = max(all_times)
            end_boundary = "latest_started_or_finished_call; no checkpoint boundary"
        elapsed = max(0.0, ends_at - starts_at)
        category_data: dict[str, dict[str, Any]] = {}
        intervals = []
        for row in rows:
            category = row["category"]
            slot = category_data.setdefault(category, {"calls": 0, "finished_calls": 0,
                                                        "pending_calls": 0, "wall_seconds": 0.0})
            slot["calls"] += 1
            if row["finished"] is not None:
                slot["finished_calls"] += 1
                duration = max(0.0, float(row["finished"]) - float(row["started"]))
                slot["wall_seconds"] += duration
                intervals.append((float(row["started"]), float(row["finished"])))
            else:
                slot["pending_calls"] += 1
        union_seconds = 0.0
        right_edge = starts_at
        for begin, finish in sorted(intervals):
            begin, finish = max(begin, starts_at), min(finish, ends_at)
            if finish <= begin:
                continue
            if finish > right_edge:
                union_seconds += finish - max(begin, right_edge)
                right_edge = finish
        call_union_inside = min(elapsed, union_seconds)
        gap = max(0.0, elapsed - call_union_inside)
        env_steps = sum(int(row["amounts"].get("environment_steps", 0)) for row in rows)
        updates_started = sum(1 for row in rows if "ppo_update" in row["name"])
        updates_complete = sum(1 for row in rows if "ppo_update" in row["name"]
                               and row["finished"] is not None)
        route_name = f"{identity[0]}:{identity[1]}"
        summaries[route_name] = {
            "start_epoch_seconds": starts_at,
            "end_epoch_seconds": ends_at,
            "start_utc": datetime.fromtimestamp(starts_at, timezone.utc).isoformat(),
            "end_utc": datetime.fromtimestamp(ends_at, timezone.utc).isoformat(),
            "end_boundary": end_boundary,
            "route_wall_seconds_including_unattributed_gaps": elapsed,
            "call_interval_union_seconds": call_union_inside,
            "unattributed_intercall_wall_seconds": gap,
            "unattributed_gap_is_not_pure_wait_or_cpu": True,
            "call_categories": category_data,
            "environment_steps": env_steps,
            "ppo_update_calls_started": updates_started,
            "ppo_update_calls_finished": updates_complete,
            "seconds_per_environment_step_including_gaps": elapsed / env_steps if env_steps else None,
            "seconds_per_completed_update_including_gaps": (
                elapsed / updates_complete if updates_complete else None),
            "seconds_per_started_update_including_gaps": (
                elapsed / updates_started if updates_started else None),
            "checkpoint_saved": end_row is not None,
        }
    return summaries


def route_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in (
        "method", "seed", "world_model_variant", "world_model_seed", "training_steps",
        "optimizer_updates", "rollout_steps", "updates_per_rollout", "checkpoint",
        "checkpoint_sha256", "checkpoint_state_sha256", "ppo_update_count",
    )} | {
        "summary_route": row.get("summary", {}).get("route"),
        "summary_environment_steps": row.get("summary", {}).get("environment_steps"),
        "summary_optimizer_updates": row.get("summary", {}).get("policy_optimizer_updates"),
        "summary_world_model_updates": row.get("summary", {}).get("world_model_updates"),
        "summary_world_model_frozen": row.get("summary", {}).get("world_model_frozen"),
        "summary_raw_decision_count": row.get("summary", {}).get("decision_summary", {}).get("raw_decision_count"),
        "summary_recurrent_replay_groups": row.get("summary", {}).get("recurrent_hidden_replay_groups"),
    }


def starts_before_route(load_row: dict[str, Any], calls: list[dict[str, Any]],
                        method: str, seed: int) -> bool:
    route_name = f"initialize_policy:{method}:{seed}"
    starts = [row for row in calls if row["name"] == route_name]
    return bool(starts and load_row["finished"] is not None
                and float(load_row["finished"]) <= float(starts[0]["started"]))


def audit(source: Path) -> dict[str, Any]:
    package = source / "package"
    run_once = source / "verified-export" / "run-once"
    checkpoint_dir = run_once / "policy-checkpoints"
    route_path = run_once / "policy-training-routes.jsonl"
    sqlite_path = run_once / "budget.sqlite3"
    export_hashes_path = source / "verified-export" / "export-hashes.json"
    matrix_path = package / "experiment-matrix.json"
    binding_path = package / "g1-model-binding.json"
    status_path = run_once / "status.json"
    settlement_path = run_once / "resource-settlement.json"
    resource_history_path = source / "verified-export" / "resource-history.jsonl"
    resource_request_path = RECOVERY_PACKAGE / "RESOURCE_REQUEST.json"

    routes = [json.loads(line) for line in route_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    export_hashes = load_json(export_hashes_path)
    matrix = load_json(matrix_path)
    binding = load_json(binding_path)
    status = load_json(status_path)
    settlement = load_json(settlement_path)
    resource_request = load_json(resource_request_path)
    request_stages = resource_request["stages"]
    training_request = request_stages["conditional_policy_training"]
    task_request = request_stages["conditional_task_confirmation"]
    total_request = resource_request["totals"]
    accounting_reserves = resource_request["accounting_reserves"]
    stage_wall_sum = sum(int(stage["wall_seconds"]) for stage in request_stages.values())
    stage_cpu_sum = sum(int(stage["complete_process_cpu_seconds"])
                        for stage in request_stages.values())
    request_budget_arithmetic_pass = (
        stage_wall_sum + int(accounting_reserves["cross_system_wall_seconds"])
        == int(total_request["wall_seconds"])
        and stage_cpu_sum + int(accounting_reserves["windows_post_preflight_process_cpu_seconds"])
        == int(total_request["complete_process_cpu_seconds"])
    )
    calls = read_calls(sqlite_path)
    timelines = route_timings(calls)
    policy_config = matrix.get("policy_configuration", {})
    policy_seeds = matrix.get("policy_seeds", [])
    world_seeds = matrix.get("world_model_seeds", [])

    checkpoint_reports = []
    for row in routes:
        filename = Path(row["checkpoint"]).name
        path = checkpoint_dir / filename
        facts = checkpoint_facts(path)
        rel_export_path = f"run-once/policy-checkpoints/{filename}"
        expected_file_hash = export_hashes.get(rel_export_path)
        binding_route = next((route for route in binding.get("routes", [])
                              if row.get("method") == "G1"
                              and route.get("policy_seed") == row.get("seed")), None)
        save_name = f"save_policy_checkpoint:{row['method']}:{row['seed']}"
        save_rows = [call for call in calls if call["name"] == save_name]
        ledger_save = (len(save_rows) == 1 and save_rows[0]["status"] == "complete"
                       and save_rows[0]["finished"] is not None
                       and int(save_rows[0]["amounts"].get("checkpoint_writes", 0)) == 1)
        file_hash_ok = facts["archive_sha256"] == row.get("checkpoint_sha256") == expected_file_hash
        state_hash_ok = facts["recomputed_state_sha256"] == facts["stored_state_sha256"] == row.get("checkpoint_state_sha256")
        identity_ok = facts["schema"] == "w1-policy-checkpoint/1.0.0" and facts["method"] == row.get("method") and facts["seed"] == row.get("seed")
        full_schedule_ok = (row.get("training_steps") == 2048 and row.get("optimizer_updates") == 128
                            and row.get("ppo_update_count") == 128
                            and row.get("rollout_steps") == 64 and row.get("updates_per_rollout") == 4
                            and facts["summary_environment_steps"] == 2048
                            and facts["summary_optimizer_updates"] == 128
                            and facts["optimizer_step_values"] == [128.0]
                            and facts["optimizer_lrs"] == [policy_config.get("learning_rate")]
                            and ((row.get("method") == "G1"
                                  and row.get("world_model_variant") == "G1"
                                  and row.get("world_model_seed") == {8301: 8201, 8302: 8202, 8303: 8203}.get(row.get("seed")))
                                 or (row.get("method") in {"G0", "T"}
                                     and row.get("world_model_variant") is None
                                     and row.get("world_model_seed") is None)))
        checkpoint_reports.append({
            **route_summary(row),
            **facts,
            "export_manifest_file_sha256": expected_file_hash,
            "file_hash_matches_route_and_export_manifest": file_hash_ok,
            "state_hash_matches_checkpoint_and_route": state_hash_ok,
            "schema_method_seed_match": identity_ok,
            "schedule_is_complete_2048_steps_128_updates": full_schedule_ok,
            "completed_checkpoint_write_ledger_row": ledger_save,
            "optimizer_lr_matches_frozen_policy_configuration": facts["optimizer_lrs"] == [policy_config.get("learning_rate")],
            "state_architecture_signature": sha256_json([
                {key: tensor[key] for key in ("name", "dtype", "shape")}
                for tensor in facts["state_tensors"]
            ]),
            "world_binding": ({
                "source_binding_schema": binding.get("schema"),
                "policy_seed": binding_route.get("policy_seed"),
                "world_seed": binding_route.get("world_seed"),
                "relative_path": binding_route.get("checkpoint_relative_path"),
                "expected_sha256": binding_route.get("checkpoint_sha256"),
                "observed_sha256": sha256_file(package / binding_route["checkpoint_relative_path"]),
                "checkpoint_seed": binding_route.get("checkpoint_metadata", {}).get("seed"),
                "checkpoint_variant": binding_route.get("checkpoint_metadata", {}).get("variant"),
                "binding_ok": (binding_route.get("world_seed") == row.get("world_model_seed")
                               and binding_route.get("world_seed") == 8201
                               and binding_route.get("checkpoint_metadata", {}).get("seed") == 8201
                               and binding_route.get("checkpoint_metadata", {}).get("variant") == "G1"
                               and sha256_file(package / binding_route["checkpoint_relative_path"])
                               == binding_route.get("checkpoint_sha256")),
            } if binding_route else None),
        })

    expected_routes = {(method, seed) for method in ("G0", "T", "G1") for seed in (8301, 8302, 8303)}
    observed_routes = {(row.get("method"), row.get("seed")) for row in routes}
    checkpoint_files = sorted(path.name for path in checkpoint_dir.glob("*.pt"))
    partial_rows = [row for row in calls if row["name"] == "ppo_update:G1:8302"]
    g1_8302_steps = sum(int(row["amounts"].get("environment_steps", 0)) for row in calls
                        if row["name"] == "policy_step:G1:8302")
    g1_8302_updates_complete = sum(row["finished"] is not None and row["status"] == "complete"
                                  for row in partial_rows)
    g1_8302_updates_pending = sum(row["finished"] is None or row["status"] == "pending"
                                 for row in partial_rows)
    paired_world_loads = {}
    for policy_seed, world_seed in zip((8301, 8302, 8303), (8201, 8202, 8203)):
        load_rows = [row for row in calls if row["name"] == f"load_frozen_world_model:G1:{world_seed}"]
        paired_world_loads[f"G1:{policy_seed}"] = {
            "world_seed": world_seed,
            "call_count": len(load_rows),
            "wall_seconds": sum(float(row["duration"]) for row in load_rows if row["duration"] is not None),
            "occurs_before_policy_route_boundary": bool(
                load_rows and starts_before_route(load_rows[0], calls, "G1", policy_seed)),
        }

    matrix_contract_ok = (
        matrix.get("policy_training_steps_per_method_seed") == 2048
        and policy_config.get("rollout_steps") == 64
        and policy_config.get("maximum_optimizer_updates_per_method_seed") == 128
        and policy_seeds == [8301, 8302, 8303]
        and world_seeds == [8201, 8202, 8203]
        and matrix.get("task_episode_count") == 240
    )

    full_g1 = timelines.get("G1:8301")
    partial_g1 = timelines.get("G1:8302")
    measured_g1_rates = [row["route_wall_seconds_including_unattributed_gaps"] / row["environment_steps"]
                         for row in (full_g1, partial_g1) if row and row.get("environment_steps")]
    slowest_per_step = max(measured_g1_rates) if measured_g1_rates else None
    missing_g1_routes = [
        {"method": "G1", "policy_seed": 8302, "world_seed": 8202,
         "current_steps": g1_8302_steps, "complete_updates": g1_8302_updates_complete,
         "pending_updates": g1_8302_updates_pending,
         "required_action": "restart this 2048-step/128-update route from initialization; no exact mid-route state bundle"},
        {"method": "G1", "policy_seed": 8303, "world_seed": 8203,
         "current_steps": 0, "complete_updates": 0, "pending_updates": 0,
         "required_action": "run this 2048-step/128-update route from initialization"},
    ]
    additional_steps = 2 * 2048
    additional_updates = 2 * 128
    base_train_wall = slowest_per_step * additional_steps if slowest_per_step is not None else None
    safety_factor = 1.5
    rounded_train_wall = (math.ceil(base_train_wall * safety_factor / 60) * 60
                          if base_train_wall is not None else None)
    stage_tail = settlement.get("unconfirmed_stage_tail", {})
    source_stage_cpu = stage_tail.get("cpu_seconds")
    source_stage_wall = stage_tail.get("stage_wall_seconds")
    observed_stage_cpu_wall_ratio = (source_stage_cpu / source_stage_wall
                                     if source_stage_cpu is not None and source_stage_wall else None)
    estimated_train_cpu = (math.ceil(rounded_train_wall * observed_stage_cpu_wall_ratio / 60) * 60
                           if rounded_train_wall is not None and observed_stage_cpu_wall_ratio is not None else None)

    budget_ledger_source = (package / "budget_ledger.py").read_text(encoding="utf-8")
    used_start = budget_ledger_source.find("def _used(")
    call_start = budget_ledger_source.find("def call(", used_start)
    snapshot_start = budget_ledger_source.find("    def snapshot(", call_start)
    used_text = budget_ledger_source[used_start:call_start] if used_start >= 0 and call_start > used_start else ""
    call_text = budget_ledger_source[call_start:snapshot_start] if call_start >= 0 and snapshot_start > call_start else ""
    insert_at = call_text.find("INSERT INTO calls")
    scan_before_insert = "self._used(self.stage, resource)" in call_text[:insert_at] if insert_at >= 0 else False
    scan_source_review = {
        "source_file_sha256": sha256_file(package / "budget_ledger.py"),
        "used_function_line": next((n for n, line in enumerate(budget_ledger_source.splitlines(), 1)
                                     if "def _used(" in line), None),
        "call_function_line": next((n for n, line in enumerate(budget_ledger_source.splitlines(), 1)
                                     if "def call(" in line), None),
        "scans_all_calls_in_current_stage": "SELECT amounts FROM calls WHERE stage=?" in used_text,
        "json_decodes_every_scanned_amount_payload": "json.loads(payload)" in used_text,
        "repeats_scan_for_each_charged_resource": "for resource, amount in amounts.items()" in call_text
            and "self._used(self.stage, resource)" in call_text,
        "scan_occurs_before_started_timestamp_insert": scan_before_insert,
        "assessment": "A confirmed mechanism consistent with growing inter-call gaps; elapsed gap cannot be wholly attributed to this scan because it has no dedicated timer and includes other Python/runtime/scheduler/I/O work.",
    }

    by_method = {}
    call_totals = defaultdict(lambda: {"count": 0, "finished_count": 0, "wall_seconds": 0.0})
    for row in calls:
        identity = row["route_explicit"]
        if identity:
            call_totals[(identity[0], row["category"])] ["count"] += 1
            if row["finished"] is not None:
                call_totals[(identity[0], row["category"])] ["finished_count"] += 1
                call_totals[(identity[0], row["category"])] ["wall_seconds"] += max(0.0, row["duration"])
    for method in ("G0", "T", "G1"):
        call_rows = [row for row in calls if row["name"].startswith((
            f"policy_step:{method}:", f"policy_reset:{method}:", f"ppo_update:{method}:",
            f"initialize_policy:{method}:", f"initialize_policy_optimizer:{method}:",
            f"save_policy_checkpoint:{method}:"))]
        # Model/prior calls lack a seed in their name; route ownership comes from boundary intervals above.
        per_method = defaultdict(lambda: {"count": 0, "finished_count": 0, "wall_seconds": 0.0})
        for route_name, info in timelines.items():
            if route_name.startswith(method + ":"):
                for category, counts in info["call_categories"].items():
                    slot = per_method[category]
                    slot["count"] += counts["calls"]
                    slot["finished_count"] += counts["finished_calls"]
                    slot["wall_seconds"] += counts["wall_seconds"]
        by_method[method] = dict(per_method)

    timeline_span = max(float(row["finished"] if row["finished"] is not None else row["started"])
                        for row in calls) - min(float(row["started"]) for row in calls)
    total_call_duration = sum(float(row["duration"]) for row in calls if row["duration"] is not None)
    stage_wall = stage_tail.get("stage_wall_seconds")
    stage_cpu = stage_tail.get("cpu_seconds")
    structure_signatures = {row["state_architecture_signature"] for row in checkpoint_reports}
    optimizer_param_group_review = {
        "checkpoint_count": len(checkpoint_reports),
        "one_group_each": len(checkpoint_reports) == 7
            and all(len(row["optimizer_lrs"]) == 1 for row in checkpoint_reports),
        "group_parameter_counts": [row["optimizer_group_parameter_counts"] for row in checkpoint_reports],
        "group_sizes_equal_state_tensor_counts": all(
            row["optimizer_group_parameter_counts"] == [row["state_tensor_count"]]
            for row in checkpoint_reports
        ),
        "optimizer_state_slot_counts": [row["optimizer_state_slot_count"] for row in checkpoint_reports],
        "all_populated_optimizer_slots_at_step_128": all(
            row["optimizer_step_values"] == [128.0] for row in checkpoint_reports
        ),
        "learning_rates": [row["optimizer_lrs"] for row in checkpoint_reports],
        "all_learning_rates_match_frozen_configuration": all(
            row["optimizer_lrs"] == [policy_config.get("learning_rate")]
            for row in checkpoint_reports
        ),
        "parameter_names_for_numeric_optimizer_ids_available": False,
        "assessment": "Each final checkpoint has one 35-ID group and 33 populated slots; the restricted offline parser cannot map the two unpopulated IDs to parameter names. Evaluation restores policy weights only, and new training routes construct fresh Adam optimizers.",
    }
    route_wall_trend = {
        name: {"wall_seconds": timelines[name]["route_wall_seconds_including_unattributed_gaps"],
               "unattributed_intercall_wall_seconds": timelines[name]["unattributed_intercall_wall_seconds"]}
        for name in ("G0:8301", "G0:8302", "G0:8303", "T:8301", "T:8302", "T:8303", "G1:8301", "G1:8302")
        if name in timelines
    }

    # Training calls are the available timing proxy.  The task-stage episode budget
    # is retained from the source request and is not asserted to be measured here.
    decision_proxy = {}
    for method in ("G0", "T", "G1"):
        if method == "G1":
            names = ["g1_policy_encode", "g1_candidate_prior_forward", "g1_policy_action_select"]
        else:
            prefix = method.lower()
            names = [f"{prefix}_policy_encode", f"{prefix}_policy_action_select"]
        components = {}
        count_by_name = {}
        for name in names:
            matched = [row for row in calls if row["name"] == name and row["finished"] is not None]
            seconds = sum(max(0.0, row["duration"]) for row in matched)
            count = len(matched)
            components[name] = {"calls": count, "wall_seconds": seconds,
                                "mean_wall_ms_per_call": seconds * 1000 / count if count else None}
            count_by_name[name] = count
        measured_counts = set(count_by_name.values())
        decision_proxy[method] = {
            "components": components,
            "all_components_have_equal_call_counts": len(measured_counts) == 1,
            "proxy_wall_ms_per_decision": sum(
                component["mean_wall_ms_per_call"] or 0.0 for component in components.values()),
            "proxy_source_limit": "SQLite training component wall times; not an end-to-end task decision timer and no CPU field",
        }
    # The frozen design has 24 parent/repeat units per policy seed and 3 seeds per method.
    steps_per_method_max = 8 * 3 * 3 * int(matrix.get("task_episode_max_steps", 18))
    evaluation_decision_seconds = sum(
        decision_proxy[method]["proxy_wall_ms_per_decision"] * steps_per_method_max / 1000
        for method in ("G0", "T", "G1"))

    resume_payload_keys = set()
    if checkpoint_reports:
        resume_payload_keys = set(checkpoint_reports[0].get("top_level_keys", []))
    missing_resume_state = [
        "Python/NumPy/Torch CPU and accelerator RNG states",
        "environment internal state and active episode position",
        "public-history/recurrent hidden state at interruption",
        "current rollout transitions, rewards, masks, values and GAE buffer",
        "episode/scenario cursor and exact sampling-generator state",
    ]
    g1_8302_timing = timelines.get("G1:8302")
    result = {
        "schema": "w1-g1-gppo-offline-recovery-audit/1.0.0",
        "audit_scope": "offline static audit only; no model/environment initialization, torch import/load, forward pass, training, or formal attempt",
        "source_root": str(source),
        "recovery_request_budget": {
            "attempt": resource_request.get("attempt"),
            "status": resource_request.get("status"),
            "budget_basis": resource_request.get("budget_basis"),
            "training_stage_caps_seconds": {
                "wall": int(training_request["wall_seconds"]),
                "cpu": int(training_request["complete_process_cpu_seconds"]),
            },
            "task_confirmation_stage_caps_seconds": {
                "wall": int(task_request["wall_seconds"]),
                "cpu": int(task_request["complete_process_cpu_seconds"]),
                "episodes": int(task_request["task_episodes"]),
            },
            "global_caps_seconds": {
                "wall": int(total_request["wall_seconds"]),
                "cpu": int(total_request["complete_process_cpu_seconds"]),
            },
            "stage_and_reserve_arithmetic_pass": request_budget_arithmetic_pass,
        },
        "inputs": {
            "policy_routes_sha256": sha256_file(route_path),
            "sqlite_sha256": sha256_file(sqlite_path),
            "export_hashes_sha256": sha256_file(export_hashes_path),
            "experiment_matrix_sha256": sha256_file(matrix_path),
            "task_input_check_sha256": sha256_file(run_once / "task-input-check.json"),
            "execution_manifest_sha256": sha256_file(package / "execution-manifest.json"),
            "budget_ledger_sha256": scan_source_review["source_file_sha256"],
            "g1_model_binding_sha256": sha256_file(binding_path),
            "runtime_hooks_sha256": sha256_file(package / "runtime_hooks.py"),
            "production_policy_sha256": sha256_file(package / "production_policy.py"),
            "resource_history_sha256": sha256_file(resource_history_path),
            "resource_settlement_sha256": sha256_file(settlement_path),
            "recovery_resource_request_sha256": sha256_file(resource_request_path),
        },
        "configuration": {
            "matrix_schema": matrix.get("schema"),
            "policy_seeds": policy_seeds,
            "world_model_seeds": world_seeds,
            "training_steps_per_route": matrix.get("policy_training_steps_per_method_seed"),
            "rollout_steps": policy_config.get("rollout_steps"),
            "updates_per_route": policy_config.get("maximum_optimizer_updates_per_method_seed"),
            "updates_per_rollout": (128 // (2048 // 64)) if matrix_contract_ok else None,
            "task_episode_count": matrix.get("task_episode_count"),
            "task_episode_max_steps": matrix.get("task_episode_max_steps"),
            "matrix_contract_pass": matrix_contract_ok,
            "g1_pairing_in_source_schedule": [
                {"policy_seed": ps, "world_seed": ws} for ps, ws in zip(policy_seeds, world_seeds)
            ],
            "source_chain": {
                "schedule_module": "package/production_policy.py::policy_schedule and PolicySchedule.routes",
                "checkpoint_writer_module": "package/runtime_hooks.py::checkpoint_writer",
                "route_output": "verified-export/run-once/policy-training-routes.jsonl",
                "task_input_check_worker_manifest_sha256": load_json(run_once / "task-input-check.json").get("worker_manifest_sha256"),
                "g1_binding_source_manifest_sha256": binding.get("source_manifest_sha256"),
            },
            "training_prior_configuration": matrix.get("prior_configuration"),
            "policy_configuration_embedded_in_checkpoint": False,
            "policy_configuration_source_chain": "frozen experiment-matrix.json -> production_policy.py policy_schedule/routes -> policy-training-routes.jsonl -> checkpoint writer; optimizer param-group lr is independently checked",
        },
        "checkpoint_writer_contract_static": {
            "source_file": "package/runtime_hooks.py",
            "source_function": "checkpoint_writer",
            "serialized_root_keys": ["schema", "method", "seed", "state_dict", "state_dict_sha256", "optimizer_state_dict", "summary"],
            "state_source": "policy.base.state_dict() where wrapped, otherwise policy.state_dict()",
            "optimizer_source": "optimizer.state_dict()",
            "no_rng_environment_or_rollout_snapshot_in_writer": True,
            "source_file_sha256": sha256_file(package / "runtime_hooks.py"),
        },
        "checkpoint_count": len(checkpoint_reports),
        "checkpoint_directory_files": checkpoint_files,
        "checkpoint_directory_file_count": len(checkpoint_files),
        "all_seven_checkpoint_state_shapes_match": len(checkpoint_reports) == 7 and len(structure_signatures) == 1
            and all(row["state_tensor_count"] == 35 for row in checkpoint_reports),
        "shared_state_architecture_signature": next(iter(structure_signatures)) if len(structure_signatures) == 1 else None,
        "checkpoint_reports": checkpoint_reports,
        "all_nine_final_routes_present": observed_routes == expected_routes,
        "all_checkpoint_integrity_checks_pass": all(
            report["file_hash_matches_route_and_export_manifest"]
            and report["state_hash_matches_checkpoint_and_route"]
            and report["schema_method_seed_match"]
            and report["schedule_is_complete_2048_steps_128_updates"]
            and report["completed_checkpoint_write_ledger_row"]
            and (report["world_binding"] is None or report["world_binding"]["binding_ok"])
            for report in checkpoint_reports
        ),
        "optimizer_param_group_review": optimizer_param_group_review,
        "recovery_8302": {
            "method": "G1",
            "policy_seed": 8302,
            "world_seed": 8202,
            "sqlite_environment_steps": g1_8302_steps,
            "sqlite_complete_ppo_updates": g1_8302_updates_complete,
            "sqlite_pending_ppo_updates": g1_8302_updates_pending,
            "intermediate_or_final_checkpoint_exists": any(name.startswith("G1-seed-8302") for name in checkpoint_files),
            "observed_route_wall_seconds_to_latest_ledger_event": g1_8302_timing.get("route_wall_seconds_including_unattributed_gaps") if g1_8302_timing else None,
            "observed_route_unattributed_intercall_wall_seconds": g1_8302_timing.get("unattributed_intercall_wall_seconds") if g1_8302_timing else None,
            "checkpoint_payload_keys": sorted(resume_payload_keys),
            "missing_exact_resume_state": missing_resume_state,
            "exact_resume_possible_from_available_artifacts": False,
            "reason": "No intermediate checkpoint; the seven files are final G0/T routes and G1:8301. Existing final policy checkpoints are not G1:8302. Writer persists policy/optimizer plus summary, not in-flight training/environment/RNG state.",
        },
        "sqlite_timing": {
            "table": "calls",
            "call_count": len(calls),
            "complete_call_count": sum(row["status"] == "complete" for row in calls),
            "pending_call_count": sum(row["status"] == "pending" for row in calls),
            "ledger_call_duration_sum_seconds": total_call_duration,
            "ledger_first_to_last_event_seconds": timeline_span,
            "unattributed_intercall_wall_seconds": max(0.0, timeline_span - total_call_duration),
            "outer_stage_wall_seconds": stage_wall,
            "outer_stage_cpu_seconds": stage_cpu,
            "outer_stage_cpu_wall_ratio_for_rough_budget_only": observed_stage_cpu_wall_ratio,
            "outer_stage_measurement_status": stage_tail.get("measurement_status"),
            "cpu_limit": "SQLite calls table contains only wall timestamps; no per-call or per-route CPU split is available.",
            "gap_limit": "Unattributed inter-call wall may include Python work, instrumentation, scheduler/IO effects, and waiting; it is not asserted to be pure waiting.",
            "by_method_category_call_wall_seconds": by_method,
            "paired_frozen_world_model_loads_before_policy_route": paired_world_loads,
            "route_wall_trend": route_wall_trend,
            "budget_ledger_scan_source_review": scan_source_review,
            "route_timelines": timelines,
        },
        "retraining_budget": {
            "routes": missing_g1_routes,
            "full_route_steps": 2048,
            "full_route_updates": 128,
            "additional_routes": 2,
            "additional_environment_steps": additional_steps,
            "additional_optimizer_updates": additional_updates,
            "base_wall_estimate_seconds": base_train_wall,
            "request_application_wall_basis_seconds": base_train_wall * safety_factor
                if base_train_wall is not None else None,
            "request_application_wall_headroom_to_cap_seconds": (
                int(training_request["wall_seconds"]) - base_train_wall * safety_factor
                if base_train_wall is not None else None
            ),
            "request_application_cpu_point_estimate_seconds": None,
            "request_application_cpu_basis_note": "The request states a 2600-second CPU cap estimated from the stage ratio but provides no exact CPU point estimate.",
            "observed_slowest_g1_route_wall_seconds_per_step": slowest_per_step,
            "safety_factor": safety_factor,
            "independent_wall_estimate_rounded_up_to_minute": rounded_train_wall,
            "independent_cpu_estimate_basis": "independent audit estimate only: minute-rounded independent wall estimate x incomplete source-stage CPU/wall ratio; no call- or route-level CPU exists",
            "independent_cpu_estimate_rounded_up_to_minute": estimated_train_cpu,
            "request_training_stage_caps_seconds": {
                "wall": int(training_request["wall_seconds"]),
                "cpu": int(training_request["complete_process_cpu_seconds"]),
            },
            "independent_rounded_estimate_headroom_to_request_caps_seconds": {
                "wall": int(training_request["wall_seconds"]) - rounded_train_wall
                    if rounded_train_wall is not None else None,
                "cpu": int(training_request["complete_process_cpu_seconds"]) - estimated_train_cpu
                    if estimated_train_cpu is not None else None,
            },
            "independent_rounded_estimate_fits_recovery_request_training_stage_caps": (
                rounded_train_wall <= int(training_request["wall_seconds"])
                and estimated_train_cpu <= int(training_request["complete_process_cpu_seconds"])
            )
                if rounded_train_wall is not None and estimated_train_cpu is not None else None,
            "historical_original_policy_stage_caps_seconds": {"wall": 3600, "cpu": 6000},
        },
        "evaluation_budget": {
            "task_episode_budget": matrix.get("task_episode_count"),
            "derivation": "8 parent scenarios x 3 repeats x (3 policy methods x 3 seeds + 1 shared H episode)",
            "policy_episode_count": 8 * 3 * 3 * 3,
            "shared_h_episode_count": 8 * 3,
            "max_steps_per_policy_method": steps_per_method_max,
            "decision_wall_proxy_by_method": decision_proxy,
            "estimated_decision_component_wall_seconds_all_policy_methods_at_max_steps": evaluation_decision_seconds,
            "planned_wall_budget_seconds": int(task_request["wall_seconds"]),
            "planned_cpu_budget_seconds": int(task_request["complete_process_cpu_seconds"]),
            "proxy_limit_status": "component-only lower-bound proxy; excludes H decision timing, environment, feature/transparent scoring, ledger, logging, and stage overhead; does not establish that the requested cap is sufficient",
            "cpu_note": "The task request remains 1600 CPU seconds; training/evaluation SQLite decision timings are wall-only and cannot recalibrate CPU.",
            "budget_note": "240 episodes are task confirmation, separate from the two G1 policy retraining routes. They must run once after all nine policy routes are available; this audit does not execute them.",
        },
        "status_evidence": {
            "run_status": status.get("status"),
            "run_settlement_status": status.get("settlement_status"),
            "worker_final_settlement_missing": status.get("worker_final_settlement_missing"),
            "settlement_status": settlement.get("settlement_status"),
            "worker_returncode": settlement.get("worker_returncode"),
        },
        "torch_imported_by_audit": "torch" in sys.modules,
    }
    return result


def render_note(result: dict[str, Any]) -> str:
    full = result["sqlite_timing"]["route_timelines"].get("G1:8301", {})
    partial = result["sqlite_timing"]["route_timelines"].get("G1:8302", {})
    retrain = result["retraining_budget"]
    evaluation = result["evaluation_budget"]
    request = result["recovery_request_budget"]
    training_caps = request["training_stage_caps_seconds"]
    task_caps = request["task_confirmation_stage_caps_seconds"]
    global_caps = request["global_caps_seconds"]
    integrity = result["all_checkpoint_integrity_checks_pass"]
    return f"""# G1 GPPO 离线恢复审计

审计输入来自 phase-accounting-repair 导出的 7 个策略 checkpoint、route JSONL、只读 SQLite、冻结配置和资源结算。审计器仅用 Python 标准库；pickle 仅允许原始映射、tensor 元数据 stub 与 Storage 标记，tensor 状态哈希从 ZIP 原始 storage bytes 重算。没有导入 torch、加载模型、创建环境、执行前向或训练。

- 7 个文件对应 G0×3、T×3、G1 seed 8301。完整性（schema、method/seed、文件哈希、state hash、保存账本、2048 steps/128 updates）通过：`{integrity}`。
- G1 seed 8301 的配对世界模型是 8201；绑定文件、8201 checkpoint 路径和 SHA-256 与 route 配对一致。
- G1 seed 8302 只有 512 steps、28 个完成更新和 1 个 pending 更新，没有任何中间/最终 G1:8302 checkpoint。writer 仅保存 policy、optimizer 和 route summary，缺 RNG、环境、公开历史及 rollout buffer 等精确续训状态，因此不能从该断点精确续训；公平补跑应从初始 seed 8302 重跑完整路由。seed 8303 也需要完整路由。
- G1:8301 route wall 为 {full.get('route_wall_seconds_including_unattributed_gaps', 0):.3f}s；G1:8302 partial wall 到账本最后事件为 {partial.get('route_wall_seconds_including_unattributed_gaps', 0):.3f}s。两者均含未归属的 route 间调用间隔；SQLite 只有 wall，不可把间隔说成纯等待或拆成 CPU。
- 两路补跑需要 {retrain['additional_environment_steps']} steps、{retrain['additional_optimizer_updates']} updates。本次申请 budget_basis 的精确 wall 外推为 {retrain['request_application_wall_basis_seconds']:.3f}s（慢速观测 G1 {retrain['observed_slowest_g1_route_wall_seconds_per_step']:.11f}s/step × 4096 steps × {retrain['safety_factor']:.1f}），对应 {training_caps['wall']}s wall cap，余量 {retrain['request_application_wall_headroom_to_cap_seconds']:.3f}s。独立审计另作整分钟粗估：{retrain['independent_wall_estimate_rounded_up_to_minute']} wall / {retrain['independent_cpu_estimate_rounded_up_to_minute']} CPU 秒；这是保守取整和旧 stage CPU/wall 比的交叉估计，不是本次申请 budget_basis。CPU cap 为 {training_caps['cpu']}s，申请没有给出精确 CPU 点估计。
- 本次申请的 task confirmation cap 为 {task_caps['wall']} wall / {task_caps['cpu']} CPU 秒，240 episodes：8×3×(9 个策略实例+共享 H)，不是训练步或 PPO rollout 数。训练 SQLite 的 decision component wall proxy 在最大策略步数下约 {evaluation['estimated_decision_component_wall_seconds_all_policy_methods_at_max_steps']:.1f}s；这是下界代理，不含 H 决策、环境、特征与透明评分、账本、日志和阶段开销，不能证明该 cap 充足。没有逐调用 CPU 计时，也不能据此校准 CPU。全局 cap 为 {global_caps['wall']} wall / {global_caps['cpu']} CPU 秒；按 stage 与 accounting reserve 求和一致：{request['stage_and_reserve_arithmetic_pass']}。

机器结果见 `recovery-audit.json`。重新运行：`python audit_recovery.py`。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = audit(args.source.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "recovery-audit.json"
    note_path = args.out / "recovery-audit.zh-CN.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    note_path.write_text(render_note(result), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "note": str(note_path),
                      "checkpoints": result["checkpoint_count"],
                      "checkpoint_integrity_pass": result["all_checkpoint_integrity_checks_pass"],
                      "torch_imported": result["torch_imported_by_audit"],
                      "g1_full_wall_seconds": result["sqlite_timing"]["route_timelines"].get("G1:8301", {}).get("route_wall_seconds_including_unattributed_gaps"),
                      "g1_partial_wall_seconds": result["sqlite_timing"]["route_timelines"].get("G1:8302", {}).get("route_wall_seconds_including_unattributed_gaps"),
                      "independent_train_wall_estimate_seconds": result["retraining_budget"]["independent_wall_estimate_rounded_up_to_minute"],
                      "independent_train_cpu_estimate_seconds": result["retraining_budget"]["independent_cpu_estimate_rounded_up_to_minute"],
                      "estimated_evaluation_decision_wall_seconds": result["evaluation_budget"]["estimated_decision_component_wall_seconds_all_policy_methods_at_max_steps"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
