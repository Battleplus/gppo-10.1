"""Bind recorded environment identity to the effective runtime config."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from infra_io import durable_atomic_json


class RuntimeConfigError(ValueError):
    pass


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_config_before_collection(
    runtime_config: Any,
    frozen_payload: Mapping[str, Any],
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    effective = asdict(runtime_config) if is_dataclass(runtime_config) else dict(vars(runtime_config))
    if canonical_sha256(dict(frozen_payload)) != expected_sha256:
        raise RuntimeConfigError("FROZEN_ENVIRONMENT_CONFIG_DIGEST_MISMATCH")
    if effective != dict(frozen_payload):
        raise RuntimeConfigError("EFFECTIVE_ENVIRONMENT_CONFIG_MISMATCH")
    if not effective.get("task_completion_mode") or not effective.get("deadline_basis"):
        raise RuntimeConfigError("TASK_SEMANTICS_MISSING_FROM_ENVIRONMENT_CONFIG")
    return {
        "schema": "w1-effective-environment-identity/1.0.0",
        "config": effective,
        "config_sha256": canonical_sha256(effective),
        "task_completion_mode": effective["task_completion_mode"],
        "deadline_basis": effective["deadline_basis"],
    }


def write_runtime_environment_record(path: Path, runtime_config: Any) -> dict[str, Any]:
    """Write provenance from the exact config object passed to the environment."""
    effective = asdict(runtime_config) if is_dataclass(runtime_config) else dict(vars(runtime_config))
    record = {
        "schema": "w1-runtime-environment-record/1.0.0",
        "config": effective,
        "config_sha256": canonical_sha256(effective),
        "task_completion_mode": effective.get("task_completion_mode"),
        "deadline_basis": effective.get("deadline_basis"),
    }
    durable_atomic_json(path, record)
    return record
