"""Authenticated local worker that delegates the actual run to runner.main()."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent


def _isolated_reexec() -> None:
    cache = Path(sys.prefix) / ".w1-no-bytecode-cache"
    if (not sys.flags.isolated or not sys.flags.dont_write_bytecode
            or not sys.flags.no_user_site or sys.pycache_prefix != str(cache)
            or cache.exists()):
        os.execv(sys.executable, [sys.executable, "-I", "-B", "-X",
                                  "pycache_prefix=" + str(cache),
                                  str(Path(__file__).resolve())])


def _read_token() -> str:
    raw = sys.stdin.readline(4097)
    if len(raw) > 4096:
        raise RuntimeError("WORKER_AUTHORIZATION_TOKEN_TOO_LARGE")
    if raw.endswith("\n"):
        raw = raw[:-1]
    if raw.endswith("\r"):
        raw = raw[:-1]
    if not raw or sys.stdin.read(1):
        raise RuntimeError("WORKER_AUTHORIZATION_TOKEN_INPUT_INVALID")
    return raw


def _install_lazy_allocator_cap(torch, gpu_request, total_gpu_bytes: int) -> None:
    """Apply the allocator cap at first CUDA initialization, after data admission."""
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA_INITIALIZED_BEFORE_LOCAL_WORKER_GUARD")
    cap = int(gpu_request["peak_allocated_memory_bytes"])
    fraction = cap / total_gpu_bytes
    original = torch.cuda._lazy_init
    state = {"configured": False}

    def bounded_lazy_init():
        original()
        if not state["configured"]:
            state["configured"] = True
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            torch.cuda.reset_peak_memory_stats(0)

    torch.cuda._lazy_init = bounded_lazy_init


def _execution_evidence(runtime: dict, gpu_start: dict, torch, started: float,
                        gpu_request: dict, runner_status: int | None,
                        error: str | None) -> dict:
    initialized = bool(torch.cuda.is_initialized())
    allocator = None
    if initialized:
        torch.cuda.synchronize(0)
        allocator = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            "current_allocated_bytes": int(torch.cuda.memory_allocated(0)),
            "current_reserved_bytes": int(torch.cuda.memory_reserved(0)),
            "device_name": torch.cuda.get_device_name(0),
            "device_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        }
    allocator_limit = int(gpu_request["peak_allocated_memory_bytes"])
    allocator_limit_pass = (allocator is None or
                            (allocator["peak_allocated_bytes"] <= allocator_limit
                             and allocator["peak_reserved_bytes"] <= allocator_limit))
    gpu_end = None
    from local_runtime import gpu_snapshot
    gpu_end = gpu_snapshot()
    if gpu_end["gpu_uuid"] != gpu_start["gpu_uuid"]:
        raise RuntimeError("LOCAL_GPU_UUID_CHANGED_DURING_WORKER")
    return {
        "schema": "w1-local-research-runtime-execution/1.0.0",
        "attempt": os.environ.get("W1_VERIFIED_ATTEMPT"),
        "runtime": runtime,
        "gpu_start_inventory": gpu_start,
        "gpu_end_inventory": gpu_end,
        "cuda_initialized": initialized,
        "allocator_measurement": allocator,
        "allocator_limit_bytes": allocator_limit,
        "allocator_limit_pass": allocator_limit_pass,
        "gpu_process_memory_or_exclusivity_measured": False,
        "gpu_limit_enforcement": "PyTorch per-process caching allocator cap installed on first CUDA initialization",
        "runner_returncode": runner_status,
        "runner_exception": error,
        "worker_elapsed_wall_seconds": time.monotonic() - started,
        "automatic_retry": False,
    }


def main() -> int:
    _isolated_reexec()
    sys.path.insert(0, str(ROOT))
    started = time.monotonic()
    token = None
    runner_status = None
    error = None
    runtime = None
    gpu_start = None
    gpu_request = None
    torch = None
    output = ROOT / "run-once"
    try:
        from manifest_contract import verify_external_authorization
        from local_runtime import gpu_snapshot, verify_runtime
        from worker_contract import verify_worker_contract
        from infra_io import durable_atomic_json

        authorization_path = Path(os.environ["W1_EXTERNAL_AUTHORIZATION_FILE"]).resolve()
        token = _read_token()
        verified = verify_external_authorization(
            ROOT, authorization_path, token=token, preflight_only=False)
        token_digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        if token_digest != os.environ.get("W1_EXTERNAL_AUTHORIZATION_TOKEN_SHA256"):
            raise RuntimeError("WORKER_AUTHORIZATION_DIGEST_BINDING_MISMATCH")
        token = None
        attempt = os.environ.get("W1_VERIFIED_ATTEMPT", "")
        identity = verified["identity"]
        request = verified["request"]
        contract = verified["contract"]
        if contract.get("native_execution_root") != str(ROOT):
            raise RuntimeError("WORKER_LOCAL_ROOT_IDENTITY_MISMATCH")
        runtime = verify_runtime(request)
        if runtime["cuda_initialized"]:
            raise RuntimeError("WORKER_PREFLIGHT_INITIALIZED_CUDA")
        worker = verify_worker_contract(
            ROOT, attempt, identity["manifest_sha256"], identity["hashes_sha256"])
        if worker["matrix"].get("world_model_device") != "cuda:0":
            raise RuntimeError("WORKER_FROZEN_MODEL_DEVICE_MISMATCH")
        gpu_request = request["gpu"]
        gpu_start = gpu_snapshot()
        if (gpu_start["gpu_physical_device"] != gpu_request["physical_device"]
                or gpu_start["free_bytes"] < gpu_request["free_memory_minimum_bytes"]
                or gpu_request["logical_device"] != "cuda:0"):
            raise RuntimeError("WORKER_LOCAL_GPU_PRECONDITION_FAILED")

        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError("WORKER_TORCH_IMPORT_INITIALIZED_CUDA")
        _install_lazy_allocator_cap(torch, gpu_request, gpu_start["total_bytes"])

        from runner import main as production_runner_main
        runner_status = production_runner_main(boundary=None)
        evidence = _execution_evidence(runtime, gpu_start, torch, started,
                                       gpu_request, runner_status, None)
        if not output.is_dir():
            raise RuntimeError("PRODUCTION_RUNNER_DID_NOT_CREATE_OUTPUT")
        durable_atomic_json(output / "local-runtime-execution.json", evidence)
        if not evidence["allocator_limit_pass"]:
            return 1
        return int(runner_status)
    except BaseException as exc:
        error = traceback.format_exc()
        traceback.print_exc()
        try:
            from infra_io import durable_atomic_json
            if runtime is not None and gpu_start is not None and torch is not None and gpu_request is not None:
                try:
                    evidence = _execution_evidence(runtime, gpu_start, torch, started,
                                                   gpu_request, runner_status, error)
                except BaseException:
                    evidence = {
                        "schema": "w1-local-research-runtime-execution/1.0.0",
                        "attempt": os.environ.get("W1_VERIFIED_ATTEMPT"),
                        "runtime": runtime,
                        "gpu_start_inventory": gpu_start,
                        "allocator_limit_bytes": int(gpu_request["peak_allocated_memory_bytes"]),
                        "allocator_limit_pass": False,
                        "gpu_measurement_error": traceback.format_exc(),
                        "runner_returncode": runner_status,
                        "runner_exception": error,
                        "automatic_retry": False,
                    }
            else:
                evidence = None
            if output.is_dir():
                if evidence is None:
                    durable_atomic_json(output / "local-runtime-execution.json", {
                        "schema": "w1-local-research-runtime-execution/1.0.0",
                        "attempt": os.environ.get("W1_VERIFIED_ATTEMPT"),
                        "runner_returncode": runner_status,
                        "runner_exception": error,
                        "automatic_retry": False,
                    })
                else:
                    durable_atomic_json(output / "local-runtime-execution.json", evidence)
            else:
                durable_atomic_json(ROOT / "worker-initialization-failure.json", {
                    "schema": "w1-worker-initialization-failure/1.0.0",
                    "attempt": os.environ.get("W1_VERIFIED_ATTEMPT", "unavailable"),
                    "stage": "local_worker_preflight",
                    "status": "technical_stop",
                    "traceback": error,
                    "environment_constructed": False,
                    "model_initialized": False,
                    "checkpoint_loaded": False,
                    "training_started": False,
                    "automatic_retry": False,
                })
        except BaseException:
            traceback.print_exc()
        return 1
    finally:
        token = None


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
    raise SystemExit(exit_code)
