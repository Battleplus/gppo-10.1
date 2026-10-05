"""One-shot, approval-gated server acceptance for the synthetic G1/G2 path."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import sys
import tempfile
import time
import traceback
import unittest


ROOT = Path(__file__).resolve().parent
REQUEST_PATH = ROOT / "SERVER_ACCEPTANCE_REQUEST.json"
AUTH_SCHEMA = "w1-server-acceptance-authorization/1.0.0"
REQUEST_SCHEMA = "w1-server-synthetic-acceptance-request/1.0.0"
MODEL_COUNTERS = (
    "model_initializations_or_loads",
    "world_batch_forwards",
    "world_backward_calls",
    "world_optimizer_updates",
    "checkpoint_writes",
    "checkpoint_loads",
    "world_sample_evaluations",
)
AUTHORIZATION_FIELDS = frozenset({
    "schema", "status", "engineering_job_id", "remote_execution_root",
    "evidence_directory", "server_acceptance_request_sha256",
    "execution_manifest_sha256", "hashes_sha256", "authorization_id",
    "one_time_token_sha256", "formal_research_attempt_created", "automatic_retry",
})
EXPECTED_COUNTERS = {
    "model_initializations_or_loads": 24,
    "world_batch_forwards": 150,
    "world_backward_calls": 18,
    "world_optimizer_updates": 18,
    "checkpoint_writes": 12,
    "checkpoint_loads": 12,
    "world_sample_evaluations": 750,
}


class AcceptanceError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise AcceptanceError("JSON_DUPLICATE_FIELD:" + key)
            value[key] = item
        return value

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise AcceptanceError("JSON_OBJECT_REQUIRED:" + path.name)
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_request(request: dict) -> None:
    if request.get("schema") != REQUEST_SCHEMA:
        raise AcceptanceError("SERVER_ACCEPTANCE_REQUEST_SCHEMA_INVALID")
    if request.get("status") != "NOT_APPROVED":
        raise AcceptanceError("SERVER_ACCEPTANCE_REQUEST_MUST_REMAIN_NOT_APPROVED")
    if (request.get("formal_research_attempt_created") is not False
            or request.get("research_training") is not False
            or request.get("automatic_retry") is not False):
        raise AcceptanceError("SERVER_ACCEPTANCE_MUST_BE_NONRESEARCH_ONE_SHOT")
    smoke_adapter = request.get("smoke_adapter") is True
    if smoke_adapter:
        if not isinstance(request.get("engineering_job_id"), str) or not request["engineering_job_id"].startswith("w1-gpu-smoke-test-"):
            raise AcceptanceError("SMOKE_ENGINEERING_JOB_ID_INVALID")
        if not isinstance(request.get("remote_execution_root"), str) or not request["remote_execution_root"].startswith("/home/user1/w1-gpu-smoke-test-"):
            raise AcceptanceError("SMOKE_REMOTE_ROOT_INVALID")
        if not isinstance(request.get("evidence_directory"), str) or not request["evidence_directory"].startswith("/home/user1/w1-gpu-smoke-test-"):
            raise AcceptanceError("SMOKE_EVIDENCE_ROOT_INVALID")
    else:
        if request.get("engineering_job_id") != "w1-joint-production-acceptance-v2-engineering-once":
            raise AcceptanceError("ENGINEERING_JOB_ID_MISMATCH")
        if request.get("remote_execution_root") != "/home/user1/w1-joint-production-acceptance-v2-engineering-once":
            raise AcceptanceError("ENGINEERING_REMOTE_ROOT_MISMATCH")
        if request.get("evidence_directory") != "/home/user1/w1-joint-production-acceptance-v2-engineering-evidence":
            raise AcceptanceError("ENGINEERING_EVIDENCE_ROOT_MISMATCH")
    gpu = request.get("gpu", {})
    if (gpu.get("physical_device") != 1 or gpu.get("logical_device") != "cuda:0"
            or (not smoke_adapter and gpu.get("exclusive_allocation_required") is not True)
            or gpu.get("peak_allocated_memory_bytes") != 8 * 1024**3
            or gpu.get("free_memory_minimum_bytes") != 9 * 1024**3):
        raise AcceptanceError("SERVER_ACCEPTANCE_GPU_CONTRACT_MISMATCH")
    limits = request.get("limits", {})
    expected_limits = {
        "wall_seconds": 180,
        "complete_process_cpu_seconds": 360,
        "server_wall_seconds": 150,
        "server_complete_process_cpu_seconds": 300,
        "remote_preflight_and_transport_cpu_allowance_seconds": 30,
        "controller_wall_seconds": 30,
        "controller_complete_process_cpu_seconds": 30,
        "controller_closing_wall_reserve_seconds": 5,
        "controller_closing_cpu_reserve_seconds": 2,
        "rss_peak_bytes": 4 * 1024**3,
        "active_storage_bytes": 512 * 1024**2,
        "aggregate_storage_bytes": 1024**3,
        "terminalization_reserve_wall_seconds": 5,
        "terminalization_reserve_cpu_seconds": 3,
        "cpu_scope_regression_wall_seconds": 30,
        "cpu_scope_regression_cpu_seconds": 10,
        "supervised_phase_sync_wall_seconds": 30,
        "supervised_phase_sync_cpu_seconds": 10,
    }
    for key, expected in expected_limits.items():
        if limits.get(key) != expected:
            raise AcceptanceError("SERVER_ACCEPTANCE_LIMIT_MISMATCH:" + key)
    if limits.get("calls") != {
        "model_initializations_or_loads": 24,
        "world_batch_forwards": 150,
        "world_backward_calls": 18,
        "world_optimizer_updates": 18,
        "checkpoint_writes": 12,
        "checkpoint_loads": 12,
        "world_sample_evaluations": 750,
        "cuda_context_initializations": 1,
    }:
        raise AcceptanceError("SERVER_ACCEPTANCE_CALL_LIMITS_MISMATCH")
    scope = request.get("scope", {})
    if (scope.get("joint_pipeline", {}).get("parents") != {
            "train": 2, "model_selection": 2, "prediction_confirmation": 8}
            or scope.get("joint_pipeline", {}).get("maximum_epochs") != 1
            or scope.get("sequence_world_regression", {}).get("parents") != {
                "train": 1, "model_selection": 1, "prediction_confirmation": 8}
            or scope.get("sequence_world_regression", {}).get("device") != "cpu"
            or scope.get("supervised_phase_sync", {}).get("test_id") != (
                "test_supervised_phase_integration.SupervisedPhaseIntegrationTests."
                "test_short_subprocesses_and_stage_boundaries_through_real_supervisor")
            or scope.get("supervised_phase_sync", {}).get("synthetic_cpu_subprocesses") != 3
            or scope.get("supervised_phase_sync", {}).get("commanded_cpu_work_seconds") != 0.09):
        raise AcceptanceError("SERVER_ACCEPTANCE_TEST_SCOPE_MISMATCH")
    if request.get("resource_measurement_scope", {}).get("root_process_rss") != (
            "root process /proc/self/status VmRSS and VmHWM only; sampled checks; excludes descendants"):
        raise AcceptanceError("SERVER_ACCEPTANCE_RSS_SCOPE_MISMATCH")


def validate_authorization(
    authorization: dict,
    *,
    request: dict,
    request_sha256: str,
    manifest_sha256: str,
    hashes_sha256: str,
    token: str | None,
    require_token: bool = True,
) -> str:
    if set(authorization) != AUTHORIZATION_FIELDS:
        raise AcceptanceError("SERVER_ACCEPTANCE_AUTHORIZATION_FIELDS_INVALID")
    expected = {
        "schema": AUTH_SCHEMA,
        "status": "APPROVED",
        "engineering_job_id": request["engineering_job_id"],
        "remote_execution_root": request["remote_execution_root"],
        "evidence_directory": request["evidence_directory"],
        "server_acceptance_request_sha256": request_sha256,
        "execution_manifest_sha256": manifest_sha256,
        "hashes_sha256": hashes_sha256,
        "formal_research_attempt_created": False,
        "automatic_retry": False,
    }
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise AcceptanceError("SERVER_ACCEPTANCE_AUTHORIZATION_BINDING_MISMATCH:" + key)
    job_id = authorization.get("authorization_id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise AcceptanceError("SERVER_ACCEPTANCE_AUTHORIZATION_ID_REQUIRED")
    token_digest = authorization.get("one_time_token_sha256")
    if (not isinstance(token_digest, str) or len(token_digest) != 64
            or any(character not in "0123456789abcdef" for character in token_digest)):
        raise AcceptanceError("SERVER_ACCEPTANCE_TOKEN_DIGEST_INVALID")
    if token is None:
        if require_token:
            raise AcceptanceError("SERVER_ACCEPTANCE_ONE_TIME_TOKEN_REQUIRED")
    elif not isinstance(token, str) or not token or not hmac.compare_digest(
            hashlib.sha256(token.encode("utf-8")).hexdigest(), token_digest):
        raise AcceptanceError("SERVER_ACCEPTANCE_TOKEN_MISMATCH")
    return job_id


def _implementation_identity(request: dict) -> None:
    identities = request.get("implementation_sha256")
    if not isinstance(identities, dict) or not identities:
        raise AcceptanceError("SERVER_ACCEPTANCE_IMPLEMENTATION_IDENTITIES_MISSING")
    for relative, expected in identities.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise AcceptanceError("SERVER_ACCEPTANCE_IMPLEMENTATION_PATH_INVALID:" + str(relative))
        if not isinstance(expected, str) or len(expected) != 64 or _sha256(path) != expected:
            raise AcceptanceError("SERVER_ACCEPTANCE_IMPLEMENTATION_DIGEST_MISMATCH:" + str(relative))


def _cpu_seconds() -> float:
    import resource
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime)


def _rss_bytes() -> int:
    values = {}
    for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {"VmRSS", "VmHWM"}:
            values[key] = int(value.strip().split()[0]) * 1024
    if "VmHWM" not in values:
        raise AcceptanceError("PROCESS_RSS_UNAVAILABLE")
    return max(values.get("VmRSS", 0), values["VmHWM"])


def _finite_nonnegative_number(value) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


def _tree_bytes(root: Path | None) -> int:
    if root is None or not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        info = path.lstat()
        if path.is_symlink():
            raise AcceptanceError("SYMLINK_IN_ACCEPTANCE_EVIDENCE_TREE")
        if path.is_file():
            total += info.st_size
    return total


class ResourceMeter:
    def __init__(self, limits: dict, *, started_wall: float | None = None,
                 clock=None, cpu_provider=None, rss_provider=None):
        self.limits = limits
        self.clock = clock or time.monotonic
        self.cpu_provider = cpu_provider or _cpu_seconds
        self.rss_provider = rss_provider or _rss_bytes
        self.started_wall = self.clock() if started_wall is None else started_wall
        self.evidence_root: Path | None = None
        self.work_root: Path | None = None
        self.root_process_rss_high_water_bytes = 0
        self.peak_active_storage_bytes = 0
        self.peak_aggregate_storage_bytes = 0
        self.last_storage_sample = 0.0
        self.phase_start: tuple[str, float, float] | None = None
        self.phases: list[dict] = []

    def bind_paths(self, evidence_root: Path, work_root: Path) -> None:
        self.evidence_root = evidence_root
        self.work_root = work_root
        self.last_storage_sample = 0.0
        self.check(force_storage=True, reserve_terminal=True)

    def sample(self, *, force_storage: bool = False) -> dict:
        now = self.clock()
        if not _finite_nonnegative_number(now) or not _finite_nonnegative_number(self.started_wall):
            raise AcceptanceError("SERVER_ACCEPTANCE_RESOURCE_SAMPLE_INVALID")
        wall = now - self.started_wall
        cpu = self.cpu_provider()
        rss = self.rss_provider()
        if (not _finite_nonnegative_number(wall)
                or not _finite_nonnegative_number(cpu)
                or isinstance(rss, bool) or not isinstance(rss, int) or rss < 0):
            raise AcceptanceError("SERVER_ACCEPTANCE_RESOURCE_SAMPLE_INVALID")
        self.root_process_rss_high_water_bytes = max(self.root_process_rss_high_water_bytes, rss)
        if force_storage or now - self.last_storage_sample >= 0.5:
            active = _tree_bytes(self.work_root)
            aggregate = _tree_bytes(self.evidence_root)
            self.peak_active_storage_bytes = max(self.peak_active_storage_bytes, active)
            self.peak_aggregate_storage_bytes = max(self.peak_aggregate_storage_bytes, aggregate)
            self.last_storage_sample = now
        else:
            active = self.peak_active_storage_bytes
            aggregate = self.peak_aggregate_storage_bytes
        return {
            "wall_seconds": wall,
            "complete_process_cpu_seconds": cpu,
            "root_process_rss_high_water_bytes": self.root_process_rss_high_water_bytes,
            "rss_measurement_scope": "root process /proc/self/status VmRSS and VmHWM only; sampled checks; excludes descendants",
            "complete_process_cpu_scope": "RUSAGE_SELF plus RUSAGE_CHILDREN, including waited/reaped descendants",
            "active_storage_peak_bytes": self.peak_active_storage_bytes,
            "aggregate_storage_peak_bytes": self.peak_aggregate_storage_bytes,
            "current_active_storage_bytes": active,
            "current_aggregate_storage_bytes": aggregate,
        }

    def check(self, *, force_storage: bool = False, reserve_terminal: bool = False) -> dict:
        snapshot = self.sample(force_storage=force_storage)
        wall_limit = self.limits["server_wall_seconds"]
        cpu_limit = self.limits["server_complete_process_cpu_seconds"]
        if reserve_terminal:
            wall_limit -= self.limits["terminalization_reserve_wall_seconds"]
            cpu_limit -= self.limits["terminalization_reserve_cpu_seconds"]
        comparisons = (
            ("wall_seconds", wall_limit),
            ("complete_process_cpu_seconds", cpu_limit),
            ("root_process_rss_high_water_bytes", self.limits["rss_peak_bytes"]),
            ("active_storage_peak_bytes", self.limits["active_storage_bytes"]),
            ("aggregate_storage_peak_bytes", self.limits["aggregate_storage_bytes"]),
        )
        for field, limit in comparisons:
            if snapshot[field] > limit:
                raise AcceptanceError("SERVER_ACCEPTANCE_RESOURCE_LIMIT_EXCEEDED:" + field)
        return snapshot

    def check_closing_bound(self) -> dict:
        snapshot = self.sample(force_storage=True)
        charged = {
            "wall_seconds": snapshot["wall_seconds"] + self.limits["terminalization_reserve_wall_seconds"],
            "complete_process_cpu_seconds": (
                snapshot["complete_process_cpu_seconds"]
                + self.limits["terminalization_reserve_cpu_seconds"]
            ),
        }
        if charged["wall_seconds"] > self.limits["server_wall_seconds"]:
            raise AcceptanceError("SERVER_ACCEPTANCE_RESOURCE_LIMIT_EXCEEDED:wall_seconds_closing_upper")
        if charged["complete_process_cpu_seconds"] > self.limits["server_complete_process_cpu_seconds"]:
            raise AcceptanceError(
                "SERVER_ACCEPTANCE_RESOURCE_LIMIT_EXCEEDED:complete_process_cpu_seconds_closing_upper"
            )
        for field, limit in (
            ("root_process_rss_high_water_bytes", self.limits["rss_peak_bytes"]),
            ("active_storage_peak_bytes", self.limits["active_storage_bytes"]),
            ("aggregate_storage_peak_bytes", self.limits["aggregate_storage_bytes"]),
        ):
            if snapshot[field] > limit:
                raise AcceptanceError("SERVER_ACCEPTANCE_RESOURCE_LIMIT_EXCEEDED:" + field)
        return {**snapshot, "closing_reserve_wall_seconds": self.limits["terminalization_reserve_wall_seconds"],
                "closing_reserve_cpu_seconds": self.limits["terminalization_reserve_cpu_seconds"],
                "charged_upper_wall_seconds": charged["wall_seconds"],
                "charged_upper_complete_process_cpu_seconds": charged["complete_process_cpu_seconds"]}

    def begin_phase(self, name: str) -> None:
        if self.phase_start is not None:
            raise AcceptanceError("RESOURCE_METER_PHASE_ALREADY_OPEN")
        self.check(force_storage=True, reserve_terminal=True)
        self.phase_start = (name, self.clock(), self.cpu_provider())

    def end_phase(self, *, wall_limit: float, cpu_limit: float) -> dict:
        if self.phase_start is None:
            raise AcceptanceError("RESOURCE_METER_PHASE_NOT_OPEN")
        name, wall_start, cpu_start = self.phase_start
        phase = {"name": name, "wall_seconds": self.clock() - wall_start,
                 "complete_process_cpu_seconds": self.cpu_provider() - cpu_start}
        self.phase_start = None
        self.check(force_storage=True, reserve_terminal=True)
        if phase["wall_seconds"] > wall_limit or phase["complete_process_cpu_seconds"] > cpu_limit:
            raise AcceptanceError("SERVER_ACCEPTANCE_PHASE_LIMIT_EXCEEDED:" + name)
        self.phases.append(phase)
        return phase


class CallBudgetMeter:
    def __init__(self, limits: dict):
        self.limits = limits
        self.totals: dict[str, int] = {}

    def charge(self, amounts: dict) -> None:
        if not isinstance(amounts, dict):
            raise AcceptanceError("MODEL_CALL_AMOUNTS_INVALID")
        for name, amount in amounts.items():
            if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                raise AcceptanceError("MODEL_CALL_AMOUNT_INVALID:" + str(name))
            next_value = self.totals.get(name, 0) + amount
            if name in self.limits and next_value > self.limits[name]:
                raise AcceptanceError("SERVER_ACCEPTANCE_CALL_LIMIT_EXCEEDED:" + name)
            self.totals[name] = next_value

    def record_cuda_context(self) -> None:
        self.charge({"cuda_context_initializations": 1})

    def verify(self) -> None:
        for name, expected in EXPECTED_COUNTERS.items():
            if self.totals.get(name, 0) != expected:
                raise AcceptanceError("SERVER_ACCEPTANCE_CALL_TOTAL_MISMATCH:" + name)
        if self.totals.get("cuda_context_initializations") != 1:
            raise AcceptanceError("CUDA_CONTEXT_INITIALIZATION_COUNT_MISMATCH")


class GpuResourceMeter:
    def __init__(self, gpu_contract: dict, snapshot, *, expected_uuid: str,
                 pid_provider=None):
        self.gpu_contract = gpu_contract
        self.snapshot = snapshot
        self.expected_uuid = expected_uuid
        self.pid_provider = pid_provider or os.getpid
        self.torch = None
        self.framework_peak_allocated_bytes = 0
        self.framework_peak_reserved_bytes = 0
        self.owned_process_memory_peak_bytes = 0
        self.phase_snapshots: list[dict] = []

    def bind_torch(self, torch) -> None:
        self.torch = torch

    def check_framework_memory(self) -> dict:
        if self.torch is None:
            raise AcceptanceError("SERVER_ACCEPTANCE_CUDA_MEMORY_PROBE_NOT_BOUND")
        allocated = self.torch.cuda.max_memory_allocated(0)
        reserved = self.torch.cuda.max_memory_reserved(0)
        values = (allocated, reserved)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in values):
            raise AcceptanceError("SERVER_ACCEPTANCE_CUDA_MEMORY_SAMPLE_INVALID")
        ceiling = self.gpu_contract["peak_allocated_memory_bytes"]
        if allocated > ceiling or reserved > ceiling:
            raise AcceptanceError("SERVER_ACCEPTANCE_GPU_FRAMEWORK_MEMORY_LIMIT_EXCEEDED")
        self.framework_peak_allocated_bytes = max(self.framework_peak_allocated_bytes, allocated)
        self.framework_peak_reserved_bytes = max(self.framework_peak_reserved_bytes, reserved)
        return {
            "cuda_max_memory_allocated_bytes": allocated,
            "cuda_max_memory_reserved_bytes": reserved,
        }

    def check_process_snapshot(self, phase: str, boundary: str) -> dict:
        owned_pid = self.pid_provider()
        snapshot = self.snapshot(
            self.gpu_contract, {owned_pid}, require_free_memory=False,
        )
        pids = snapshot.get("gpu_owned_compute_process_pids")
        owned = snapshot.get("gpu_owned_process_memory_bytes")
        if (snapshot.get("gpu_physical_device") != self.gpu_contract["physical_device"]
                or snapshot.get("gpu_uuid") != self.expected_uuid
                or snapshot.get("gpu_exclusive_allocation_observed") is not True
                or not isinstance(pids, list)
                or any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in pids)
                or owned_pid not in pids
                or set(pids) - {owned_pid}
                or isinstance(owned, bool) or not isinstance(owned, int) or owned < 0
                or owned > self.gpu_contract["peak_allocated_memory_bytes"]):
            raise AcceptanceError("SERVER_ACCEPTANCE_GPU_EXCLUSIVITY_OR_MEMORY_CHECK_FAILED")
        self.owned_process_memory_peak_bytes = max(self.owned_process_memory_peak_bytes, owned)
        result = {
            "phase": phase,
            "boundary": boundary,
            "gpu_physical_device": snapshot["gpu_physical_device"],
            "gpu_uuid": snapshot["gpu_uuid"],
            "gpu_owned_compute_process_pids": list(snapshot["gpu_owned_compute_process_pids"]),
            "gpu_owned_process_memory_bytes": owned,
            "gpu_exclusive_allocation_observed": True,
        }
        self.phase_snapshots.append(result)
        return result

    def evidence(self) -> dict:
        return {
            "ceiling_bytes": self.gpu_contract["peak_allocated_memory_bytes"],
            "framework_peak_allocated_bytes": self.framework_peak_allocated_bytes,
            "framework_peak_reserved_bytes": self.framework_peak_reserved_bytes,
            "observed_owned_process_memory_peak_bytes": self.owned_process_memory_peak_bytes,
            "framework_samples": "torch.cuda.max_memory_allocated/reserved at production call boundaries; no extra model forwards",
            "process_samples": list(self.phase_snapshots),
        }


def install_realtime_call_guard(call_meter: CallBudgetMeter, resource_meter: ResourceMeter,
                                *, gpu_memory_check=None):
    import budget_ledger
    original = budget_ledger.BudgetLedger.call

    def guarded(ledger, name, amounts, operation, *args, **kwargs):
        call_meter.charge(dict(amounts))
        is_model_call = any(counter in amounts for counter in MODEL_COUNTERS)
        resource_meter.check(reserve_terminal=True)
        if is_model_call and gpu_memory_check is not None:
            gpu_memory_check()
        try:
            return original(ledger, name, amounts, operation, *args, **kwargs)
        finally:
            if is_model_call and gpu_memory_check is not None:
                gpu_memory_check()
            resource_meter.check(reserve_terminal=True)

    budget_ledger.BudgetLedger.call = guarded
    return lambda: setattr(budget_ledger.BudgetLedger, "call", original)


def _registration_details(root: Path, request: dict, runtime: dict, gpu_snapshot: dict,
                          evidence_root: Path) -> dict:
    gpu = request["gpu"]
    return {
        "ai": "Codex",
        "task_name": "W1 G1/G2 synthetic production acceptance",
        "purpose": "Bounded engineering acceptance only; no formal research attempt or research effect claim.",
        "workdir": str(root),
        "training_files": ["production_data.py", "production_world.py", "joint_pipeline.py",
                           "test_joint_pipeline.py", "test_sequence_world.py"],
        "dataset": "synthetic integration fixtures only",
        "model": "G1/G2 production world-model routes on synthetic inputs",
        "command_redacted": "python -I -B acceptance_entry.py --authorization-file <external-acceptance-authorization>",
        "gpu_request": {"ids": [gpu["physical_device"]], "count": 1, "vram_estimate_gb": "8.0"},
        "cpu_cores_estimate": "4",
        "ram_estimate_gb": "4",
        "disk_growth_estimate_gb": "1.0",
        "duration_estimate": "<= 180 seconds",
        "estimate_basis": "Independent NOT_APPROVED synthetic acceptance request; one bounded invocation; no automatic retry.",
        "log_path": str(evidence_root / "acceptance-result.json"),
        "checkpoint_path": str(evidence_root / "joint-controlled-export"),
        "resource_snapshot": {
            "runtime_identity_only_preflight": runtime,
            "gpu_before_registration": gpu_snapshot,
            "package_attempt": request["attempt"] if "attempt" in request else request["acceptance_id"],
            "formal_research_attempt_created": False,
            "evidence_directory": str(evidence_root),
        },
    }


def _manifest_artifacts(evidence_root: Path, result: dict) -> dict:
    files = {}
    for path in sorted(evidence_root.rglob("*")):
        if path.name == "acceptance-evidence-manifest.json":
            continue
        if path.is_symlink():
            raise AcceptanceError("SYMLINK_IN_ACCEPTANCE_EVIDENCE_TREE")
        if path.is_file():
            relative = path.relative_to(evidence_root).as_posix()
            files[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest = {
        "schema": "w1-server-synthetic-acceptance-evidence/1.0.0",
        "verified": True,
        "engineering_job_id": result["engineering_job_id"],
        "formal_research_attempt_created": False,
        "engineering_job_consumed": True,
        "run_status": result["run_status"],
        "files": files,
    }
    _write_json(evidence_root / "acceptance-evidence-manifest.json", manifest)
    return manifest


def _run_bounded_tests(evidence_root: Path, work_root: Path, request: dict,
                       resource_meter: ResourceMeter, call_meter: CallBudgetMeter,
                       gpu_meter: GpuResourceMeter) -> dict:
    import test_cpu_scope_contract
    import test_joint_pipeline
    import test_sequence_world
    import test_supervised_phase_integration

    previous_tempdir = tempfile.tempdir
    tempfile.tempdir = str(work_root)
    previous_environment = {
        key: os.environ.get(key)
        for key in ("W1_TEST_EVIDENCE_DIR", "W1_SYNTHETIC_TEST_DEVICE",
                    "W1_PHASE_ACCOUNTING_SOCKET")
    }
    os.environ["W1_TEST_EVIDENCE_DIR"] = str(evidence_root)
    os.environ["W1_SYNTHETIC_TEST_DEVICE"] = "cuda"
    os.environ.pop("W1_PHASE_ACCOUNTING_SOCKET", None)
    originals = {
        "controlled_export": test_joint_pipeline.controlled_export,
        "recompute_prediction_metrics": test_sequence_world.production_world.recompute_prediction_metrics,
    }

    def export_and_preserve(package, destination):
        result = originals["controlled_export"](package, destination)
        preserved = evidence_root / "joint-controlled-export"
        shutil.copytree(destination, preserved)
        resource_meter.check(force_storage=True, reserve_terminal=True)
        return result

    def recompute_and_preserve(trace_path, *args, **kwargs):
        result = originals["recompute_prediction_metrics"](trace_path, *args, **kwargs)
        shutil.copy2(trace_path, evidence_root / "sequence-world-trace.jsonl")
        _write_json(evidence_root / "sequence-world-metrics.json", result)
        return result

    test_joint_pipeline.controlled_export = export_and_preserve
    test_sequence_world.production_world.recompute_prediction_metrics = recompute_and_preserve
    restore_call_guard = install_realtime_call_guard(
        call_meter, resource_meter, gpu_memory_check=gpu_meter.check_framework_memory,
    )
    cases = (
        ("cpu_scope_regression",
         test_cpu_scope_contract.CpuScopeContractTests,
         "test_monitor_charges_live_delta_and_reaped_short_child_once"),
        ("supervised_phase_sync",
         test_supervised_phase_integration.SupervisedPhaseIntegrationTests,
         "test_short_subprocesses_and_stage_boundaries_through_real_supervisor"),
        ("joint_pipeline",
         test_joint_pipeline.JointProductionPipelineTests,
         "test_actual_collection_training_restore_metrics_ledger_export"),
        ("sequence_world_regression",
         test_sequence_world.SequenceWorldIntegrationTests,
         "test_synthetic_cpu_train_checkpoint_restore_and_confirmation_accounting"),
    )
    phase_limits = {
        "cpu_scope_regression": (
            request["limits"]["cpu_scope_regression_wall_seconds"],
            request["limits"]["cpu_scope_regression_cpu_seconds"],
        ),
        "supervised_phase_sync": (
            request["limits"]["supervised_phase_sync_wall_seconds"],
            request["limits"]["supervised_phase_sync_cpu_seconds"],
        ),
    }
    outcomes = []
    try:
        for phase, test_class, method in cases:
            resource_meter.begin_phase(phase)
            gpu_start = gpu_meter.check_process_snapshot(phase, "start")
            output = __import__("io").StringIO()
            result = unittest.TextTestRunner(stream=output, verbosity=2).run(
                unittest.TestSuite([test_class(method)])
            )
            (evidence_root / (phase + ".log")).write_text(output.getvalue(), encoding="utf-8")
            gpu_end = gpu_meter.check_process_snapshot(phase, "end")
            gpu_framework = gpu_meter.check_framework_memory()
            phase_result = {
                "phase": phase,
                "test_id": test_class.__module__ + "." + test_class.__name__ + "." + method,
                "tests_run": result.testsRun,
                "successful": result.wasSuccessful() and result.testsRun == 1 and not result.skipped,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped),
                "gpu_resources": {"start": gpu_start, "end": gpu_end,
                                  "framework_memory": gpu_framework},
            }
            if phase in phase_limits:
                phase_result["resources"] = resource_meter.end_phase(
                    wall_limit=phase_limits[phase][0], cpu_limit=phase_limits[phase][1],
                )
            else:
                phase_result["resources"] = resource_meter.end_phase(
                    wall_limit=request["limits"]["wall_seconds"],
                    cpu_limit=request["limits"]["complete_process_cpu_seconds"],
                )
            resource_meter.check(force_storage=True, reserve_terminal=True)
            outcomes.append(phase_result)
            if not phase_result["successful"]:
                raise AcceptanceError("SERVER_ACCEPTANCE_TEST_FAILED:" + phase)
        call_meter.verify()
        joint = _read_json(evidence_root / "joint-success.json")
        if (joint.get("device") != "cuda" or joint.get("status") != "prediction_evaluation_complete"
                or joint.get("real_environment_calls") != 0
                or joint.get("synthetic_environment_constructions") != 12
                or joint.get("controlled_export_verified") is not True):
            raise AcceptanceError("JOINT_PIPELINE_EVIDENCE_INVALID")
        sequence = _read_json(evidence_root / "sequence-world-metrics.json")
        if sequence.get("coverage_pass") is not True or sequence.get("gppo_executed") is not False:
            raise AcceptanceError("SEQUENCE_WORLD_EVIDENCE_INVALID")
        supervised = _read_json(evidence_root / "supervised-phase-integration.json")
        if (supervised.get("supervisor", {}).get("status") != "complete"
                or supervised.get("real_environment_calls") != 0
                or supervised.get("model_calls") != 0
                or supervised.get("synthetic_cpu_subprocesses") != 3
                or supervised.get("commanded_cpu_work_seconds") != 0.09
                or any(row.get("supervisor_handshake", {}).get("status") != "acknowledged"
                       for row in supervised.get("worker_phases", [])[:-1])):
            raise AcceptanceError("SUPERVISED_PHASE_SYNC_EVIDENCE_INVALID")
        return {"tests": outcomes, "calls": dict(sorted(call_meter.totals.items())),
                "joint_pipeline": joint, "sequence_world": sequence,
                "supervised_phase_sync": supervised,
                "gpu_resources": gpu_meter.evidence()}
    finally:
        restore_call_guard()
        test_joint_pipeline.controlled_export = originals["controlled_export"]
        test_sequence_world.production_world.recompute_prediction_metrics = originals["recompute_prediction_metrics"]
        tempfile.tempdir = previous_tempdir
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_registered_once(*, register, mark_cpu_running, initialize_gpu_and_bind,
                        run_workload, append_terminal):
    job = register()
    terminal_written = False
    try:
        mark_cpu_running(job)
        initialize_gpu_and_bind(job)
        result = run_workload(job)
        append_terminal(job, "SUCCEEDED", 0, None)
        terminal_written = True
        return job, result
    except BaseException as exc:
        if not terminal_written:
            failure_event = "EXPIRED" if isinstance(exc, TimeoutError) else "FAILED"
            try:
                append_terminal(job, failure_event, 1, type(exc).__name__)
            except BaseException as terminal_error:
                raise AcceptanceError("SERVER_REGISTRATION_FAILURE_TERMINAL_WRITE_FAILED:" +
                                      type(terminal_error).__name__) from exc
        raise


def _require_evidence_root(path: Path, *, approved_root: str | None = None) -> Path:
    resolved = path.resolve()
    if resolved != Path(approved_root or "/home/user1/w1-joint-production-acceptance-v2-engineering-evidence"):
        raise AcceptanceError("ENGINEERING_EVIDENCE_ROOT_MUST_MATCH_APPROVED_REQUEST")
    if resolved.is_relative_to(ROOT.resolve()) or resolved.exists():
        raise AcceptanceError("ENGINEERING_EVIDENCE_ROOT_MUST_BE_EXTERNAL_AND_UNUSED")
    resolved.mkdir(parents=False, exist_ok=False, mode=0o700)
    return resolved


def _install_deadline(request: dict, started: float):
    def expired(_signum, _frame):
        raise TimeoutError("SERVER_ACCEPTANCE_WALL_DEADLINE")

    previous = signal.signal(signal.SIGALRM, expired)
    remaining = request["limits"]["server_wall_seconds"] - (time.monotonic() - started)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, remaining))
    return previous


def _remaining_wall_request(request: dict, remaining: float | None) -> dict:
    if remaining is None:
        return request
    cap = request["limits"]["server_wall_seconds"]
    if (not math.isfinite(remaining) or remaining <= request["limits"]["terminalization_reserve_wall_seconds"]
            or remaining > cap):
        raise AcceptanceError("REMAINING_WALL_ALLOWANCE_INVALID")
    # Runtime narrowing only; no mutation or increase of frozen resource request.
    return {**request, "limits": {**request["limits"], "server_wall_seconds": remaining}}


def _bootstrap_runtime(request: dict) -> dict:
    """Validate isolated interpreter paths before admitting this package."""
    probe = __import__("runpy").run_path(str(ROOT / "remote_runtime_preflight.py"))
    probe["verify_process_isolation"]()
    runtime_spec = _read_json(ROOT / "remote-runtime-identity.json")
    if _sha256(ROOT / "remote-runtime-identity.json") != request["runtime"]["runtime_identity_sha256"]:
        raise AcceptanceError("SERVER_ACCEPTANCE_RUNTIME_IDENTITY_BINDING_MISMATCH")
    runtime_snapshot = probe["verify"](runtime_spec)
    smoke_adapter = request.get("smoke_adapter") is True

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "native"))
    import linux_process_scope
    if Path(linux_process_scope.__file__).resolve() != ROOT / "linux_process_scope.py":
        raise AcceptanceError("BOOTSTRAP_MODULE_ORIGIN_MISMATCH")
    return runtime_snapshot


def _run_server_acceptance(parsed, request: dict, started: float,
                           resource_meter: ResourceMeter, closing_state: dict) -> int:
    runtime_snapshot = _bootstrap_runtime(request)
    from linux_process_scope import enable_subreaper, reap_owned_children
    enable_subreaper()
    smoke_adapter = request.get("smoke_adapter") is True
    from registration_identity import derive_name_id
    from manifest_contract import sha256_file, verify_package
    import registration_identity

    derived_name_id = derive_name_id(parsed.real_name)
    if parsed.name_id is not None and parsed.name_id != derived_name_id:
        raise AcceptanceError("NAME_ID_DOES_NOT_MATCH_DERIVED_REAL_NAME")
    name_id = derived_name_id
    _implementation_identity(request)
    identity = verify_package(ROOT)
    contract = _read_json(ROOT / "launch-contract.json")
    if (contract.get("runtime_identity_sha256") != request["runtime"]["runtime_identity_sha256"]
            or contract.get("native_python") != request["runtime"]["python_executable"]):
        raise AcceptanceError("SERVER_ACCEPTANCE_LAUNCH_RUNTIME_MISMATCH")
    auth_path = parsed.authorization_file.resolve()
    if auth_path.is_relative_to(ROOT.resolve()) or not auth_path.is_file():
        raise AcceptanceError("SERVER_ACCEPTANCE_AUTHORIZATION_MUST_BE_EXTERNAL")
    token = sys.stdin.readline().rstrip("\r\n")
    authorization = _read_json(auth_path)
    authorization_id = validate_authorization(
        authorization,
        request=request,
        request_sha256=sha256_file(REQUEST_PATH),
        manifest_sha256=identity["manifest_sha256"],
        hashes_sha256=identity["hashes_sha256"],
        token=token,
    )
    token = None
    evidence_root = _require_evidence_root(parsed.evidence_dir, approved_root=request["evidence_directory"])

    from infra_io import require_native_linux_filesystem
    require_native_linux_filesystem(ROOT)
    require_native_linux_filesystem(evidence_root)
    if not smoke_adapter:
        from server_registration import append_running, append_terminal, register_before_launch
    else:
        append_running = append_terminal = register_before_launch = None
    from joint_remote_native import gpu_ids_for_pids
    supervise = __import__("runpy").run_path(str(ROOT / "supervise.py"))
    gpu_snapshot = supervise["gpu_snapshot"]
    initial_gpu = gpu_snapshot(request["gpu"], set(), require_free_memory=True)
    if initial_gpu["gpu_physical_device"] != request["gpu"]["physical_device"]:
        raise AcceptanceError("SERVER_ACCEPTANCE_PHYSICAL_GPU_MISMATCH")
    if initial_gpu["gpu_free_memory_bytes"] < request["gpu"]["free_memory_minimum_bytes"]:
        raise AcceptanceError("SERVER_ACCEPTANCE_FREE_GPU_MEMORY_BELOW_MINIMUM")
    gpu_meter = GpuResourceMeter(
        request["gpu"], gpu_snapshot, expected_uuid=initial_gpu["gpu_uuid"],
    )

    evidence_dir = evidence_root
    work_root = evidence_dir / "work"
    work_root.mkdir(mode=0o700)
    limits = request["limits"]
    resource_meter.bind_paths(evidence_dir, work_root)
    call_meter = CallBudgetMeter(limits["calls"])
    result_payload = {
        "schema": "w1-server-synthetic-acceptance-result/1.0.0",
        "engineering_job_id": request["engineering_job_id"],
        "authorization_id": authorization_id,
        "formal_research_attempt_created": False,
        "engineering_job_consumed": True,
        "automatic_retry": False,
        "run_status": "failed",
        "resource_request_status": "NOT_APPROVED",
        "research_success": False,
        "gpu_smoke_test": True,
        "registration_scope": "explicitly_skipped_by_smoke_test_authorization",
        "gpu_exclusive_allocation_scope": "not_authoritatively_proven",
        "transport_cpu_scope": "not_measured_for_full_ssh_sftp_lifetime",
        "name_id": name_id,
        "package_identity": {
            "execution_manifest_sha256": identity["manifest_sha256"],
            "hashes_sha256": identity["hashes_sha256"],
            "server_acceptance_request_sha256": sha256_file(REQUEST_PATH),
        },
        "runtime_preflight": runtime_snapshot,
        "initial_gpu_snapshot": initial_gpu,
        "started_monotonic": started,
    }
    registration_state = {"job": None, "terminal_event": None}

    def register():
        if smoke_adapter:
            result_payload["registration_skipped"] = True
            result_payload["registration_skip_reason"] = "explicit_gpu_smoke_test_scope"
            return {"job_id": request["engineering_job_id"]}
        details = _registration_details(ROOT, request, runtime_snapshot, initial_gpu, evidence_dir)
        job = register_before_launch(
            real_name=parsed.real_name,
            name_id=name_id,
            details=details,
            attempt=request["engineering_job_id"],
            manifest_sha256=identity["manifest_sha256"],
            hashes_sha256=identity["hashes_sha256"],
            resource_request_sha256=sha256_file(REQUEST_PATH),
            budget={"totals": limits, "calls": limits["calls"]},
        )
        result_payload["registration_job_id"] = job["job_id"]
        registration_state["job"] = job
        return job

    def mark_cpu_running(job):
        if smoke_adapter:
            result_payload["cpu_running_registration_skipped"] = True
            return
        append_running(job["job_id"], pid=os.getpid(), pgid=os.getpgrp(), worker_pids=[],
                       actual_gpu_ids=[], readonly_gpu_query=gpu_ids_for_pids,
                       log_path=str(evidence_dir / "acceptance-result.json"))

    def initialize_gpu_and_bind(job):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(request["gpu"]["physical_device"])
        import torch
        if (torch.__version__ != request["runtime"]["torch"]
                or torch.version.cuda != request["runtime"]["cuda"]
                or not torch.cuda.is_available()):
            raise AcceptanceError("SERVER_ACCEPTANCE_CUDA_RUNTIME_MISMATCH")
        torch.cuda.set_device(0)
        torch.cuda.init()
        gpu_meter.bind_torch(torch)
        call_meter.record_cuda_context()
        context_probe = torch.empty((1,), device="cuda:0")
        torch.cuda.synchronize()
        del context_probe
        gpu_meter.check_framework_memory()
        current_gpu = gpu_meter.check_process_snapshot("initialization", "post_cuda_context")
        resource_meter.check(reserve_terminal=True)
        if not smoke_adapter:
            append_running(job["job_id"], pid=os.getpid(), pgid=os.getpgrp(), worker_pids=[],
                           actual_gpu_ids=[request["gpu"]["physical_device"]],
                           readonly_gpu_query=gpu_ids_for_pids,
                           log_path=str(evidence_dir / "acceptance-result.json"), restarted=True)
        result_payload["cuda_context_initializations"] = 1
        result_payload["gpu_after_context"] = current_gpu

    def run_workload(_job):
        results = _run_bounded_tests(
            evidence_dir, work_root, request, resource_meter, call_meter, gpu_meter,
        )
        result_payload.update({
            "run_status": "passed_pending_terminal_registration",
            "test_results": results["tests"],
            "model_call_totals": results["calls"],
            "joint_pipeline_evidence": {
                "device": results["joint_pipeline"]["device"],
                "synthetic_environment_constructions": results["joint_pipeline"]["synthetic_environment_constructions"],
                "real_environment_calls": results["joint_pipeline"]["real_environment_calls"],
                "status": results["joint_pipeline"]["status"],
                "controlled_export_verified": results["joint_pipeline"]["controlled_export_verified"],
            },
            "sequence_world_evidence": {
                "coverage_pass": results["sequence_world"]["coverage_pass"],
                "gppo_executed": results["sequence_world"]["gppo_executed"],
                "scope": "direct production_world regression, CPU, 1/1/8",
            },
            "supervised_phase_sync_evidence": {
                "status": results["supervised_phase_sync"]["supervisor"]["status"],
                "real_environment_calls": results["supervised_phase_sync"]["real_environment_calls"],
                "model_calls": results["supervised_phase_sync"]["model_calls"],
                "synthetic_cpu_subprocesses": results["supervised_phase_sync"]["synthetic_cpu_subprocesses"],
                "commanded_cpu_work_seconds": results["supervised_phase_sync"]["commanded_cpu_work_seconds"],
            },
            "resource_phases": list(resource_meter.phases),
            "gpu_resource_evidence": results["gpu_resources"],
        })
        result_payload["resources_before_terminal_event"] = resource_meter.check_closing_bound()
        _write_json(evidence_dir / "acceptance-result.json", result_payload)
        _manifest_artifacts(evidence_dir, result_payload)
        resource_meter.check_closing_bound()
        return result_payload

    def append_terminal_event(job, event, exit_code, reason):
        if smoke_adapter:
            result_payload["terminal_registration_skipped"] = True
            result_payload["terminal_registration_skip_reason"] = "explicit_gpu_smoke_test_scope"
            registration_state["terminal_event"] = event
            return
        reap_owned_children(timeout_seconds=2.0)
        resource_meter.check_closing_bound()
        from metered_joint_entry import enforce_closing_limits
        import resource
        own = resource.getrusage(resource.RUSAGE_SELF)
        enforce_closing_limits(own.ru_utime + own.ru_stime)
        closing_state["enforced"] = True
        append_terminal(job["job_id"], event, exit_code=exit_code,
                        reason_redacted=reason, result_path=str(evidence_dir))
        registration_state["terminal_event"] = event

    try:
        job, result = run_registered_once(
            register=register,
            mark_cpu_running=mark_cpu_running,
            initialize_gpu_and_bind=initialize_gpu_and_bind,
            run_workload=run_workload,
            append_terminal=append_terminal_event,
        )
        result_payload["run_status"] = "passed"
        result_payload["registration_terminal_event"] = "SUCCEEDED"
        result_payload["terminal_registration_completed"] = True
        result_payload["resources_after_terminal_event"] = resource_meter.check_closing_bound()
        _write_json(evidence_dir / "acceptance-result.json", result_payload)
        _manifest_artifacts(evidence_dir, result_payload)
        final = resource_meter.check_closing_bound()
        print(json.dumps({
            "status": "complete",
            "engineering_job_id": request["engineering_job_id"],
            "registration_job_id": job["job_id"],
            "formal_research_attempt_created": False,
            "engineering_job_consumed": True,
            "evidence_manifest_sha256": _sha256(evidence_dir / "acceptance-evidence-manifest.json"),
            "resources": final,
        }, sort_keys=True))
        return 0
    except BaseException as exc:
        result_payload["run_status"] = "technical_stop"
        result_payload["exception_chain"] = traceback.format_exc()
        result_payload["error_type"] = type(exc).__name__
        result_payload["registration_terminal_event"] = registration_state["terminal_event"]
        result_payload["gpu_resource_evidence"] = gpu_meter.evidence()
        result_payload["resources_at_stop"] = resource_meter.sample(force_storage=True)
        if evidence_dir.exists():
            _write_json(evidence_dir / "acceptance-result.json", result_payload)
            _manifest_artifacts(evidence_dir, result_payload)
        print(json.dumps({
            "status": "technical_stop",
            "error_type": type(exc).__name__,
            "engineering_job_id": request["engineering_job_id"],
            "registration_job_id": result_payload.get("registration_job_id"),
            "formal_research_attempt_created": False,
            "engineering_job_consumed": registration_state["job"] is not None,
            "resources": result_payload.get("resources_at_stop"),
            "evidence_manifest_sha256": _sha256(evidence_dir / "acceptance-evidence-manifest.json")
                if (evidence_dir / "acceptance-evidence-manifest.json").is_file() else None,
        }, sort_keys=True))
        return 1


def _main(args=None) -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-only", action="store_true")
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--real-name")
    parser.add_argument("--name-id")
    parser.add_argument("--remaining-wall-seconds", type=float)
    parser.add_argument("--evidence-dir", type=Path)
    parsed = parser.parse_args(args)
    request = _read_json(REQUEST_PATH)
    validate_request(request)
    if parsed.bootstrap_only:
        runtime_snapshot = _bootstrap_runtime(request)
        import importlib
        origins = {}
        for name in ("linux_process_scope", "registration_identity", "manifest_contract",
                     "infra_io", "joint_remote_native", "supervise"):
            module = importlib.import_module(name)
            origin = Path(module.__file__).resolve()
            if origin != ROOT / (name + ".py"):
                raise AcceptanceError("BOOTSTRAP_MODULE_ORIGIN_MISMATCH:" + name)
            origins[name] = str(origin)
        print(json.dumps({"status": "bootstrap_pass", "modules": origins,
                          "runtime": runtime_snapshot, "worker_started": False,
                          "model_initializations": 0, "checkpoint_calls": 0}))
        return 0
    if parsed.authorization_file is None or parsed.evidence_dir is None:
        raise AcceptanceError("AUTHORIZATION_AND_EVIDENCE_REQUIRED")
    if not parsed.real_name or parsed.real_name != parsed.real_name.strip():
        raise AcceptanceError("REAL_NAME_REQUIRED_BEFORE_SERVER_WORKLOAD")
    if sys.platform != "linux":
        raise AcceptanceError("SERVER_ACCEPTANCE_REQUIRES_NATIVE_LINUX")
    metered_request = _remaining_wall_request(request, parsed.remaining_wall_seconds)
    resource_meter = ResourceMeter(metered_request["limits"], started_wall=started)
    previous_sigalrm = _install_deadline(metered_request, started)
    closing_state = {"enforced": False}
    try:
        return _run_server_acceptance(parsed, request, started, resource_meter, closing_state)
    finally:
        if not closing_state["enforced"]:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_sigalrm)


def main(argv=None) -> int:
    try:
        return _main(argv)
    except BaseException as exc:
        print(json.dumps({"status": "rejected", "error_type": type(exc).__name__,
                          "exception_chain": traceback.format_exc(),
                          "formal_research_attempt_created": False,
                          "engineering_job_consumed": False}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
