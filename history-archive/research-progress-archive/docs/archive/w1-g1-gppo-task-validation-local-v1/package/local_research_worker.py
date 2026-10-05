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


def main() -> int:
    _isolated_reexec()
    sys.path.insert(0, str(ROOT))
    started = time.monotonic()
    token = None
    runner_status = None
    error = None
    runtime = None
    torch = None
    output = ROOT / "run-once"
    try:
        from manifest_contract import verify_external_authorization
        from local_runtime import verify_runtime
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
        if worker["matrix"].get("world_model_device") != "cpu":
            raise RuntimeError("WORKER_FROZEN_MODEL_DEVICE_MISMATCH")
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError("WORKER_TORCH_IMPORT_INITIALIZED_CUDA")
        if os.sched_getaffinity(0) != {0}:
            raise RuntimeError("WORKER_FIXED_CPU_AFFINITY_MISMATCH")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        from runner import main as production_runner_main
        runner_status = production_runner_main(boundary=None)
        evidence = {"schema":"w1-g1-gppo-cpu-execution/1.0.0","runtime":runtime,
            "runner_returncode":runner_status,"device":"cpu","affinity":[0],"threads":1,
            "cuda_initialized":torch.cuda.is_initialized(),"automatic_retry":False,
            "worker_elapsed_wall_seconds":time.monotonic()-started,"gpu_used":False}
        if evidence['cuda_initialized']:raise RuntimeError("TASK_CPU_RUN_INITIALIZED_CUDA")
        if not output.is_dir():
            raise RuntimeError("PRODUCTION_RUNNER_DID_NOT_CREATE_OUTPUT")
        durable_atomic_json(output / "local-runtime-execution.json", evidence)
        return int(runner_status)
    except BaseException as exc:
        error = traceback.format_exc()
        traceback.print_exc()
        try:
            from infra_io import durable_atomic_json
            evidence = {
                "schema": "w1-g1-gppo-cpu-execution/1.0.0",
                "attempt": os.environ.get("W1_VERIFIED_ATTEMPT"),
                "runtime": runtime,
                "device": "cpu",
                "runner_returncode": runner_status,
                "runner_exception": error,
                "cuda_initialized": bool(torch.cuda.is_initialized()) if torch is not None else None,
                "automatic_retry": False,
            }
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
