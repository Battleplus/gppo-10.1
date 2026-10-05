"""One-shot local WSL entry for the frozen real W1 research package."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import secrets
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
ENTRY_STARTED_WALL = time.monotonic()
ENTRY_STARTED_SELF = resource.getrusage(resource.RUSAGE_SELF)
ENTRY_STARTED_CHILDREN = resource.getrusage(resource.RUSAGE_CHILDREN)


def _isolated_reexec() -> None:
    cache = Path(sys.prefix) / ".w1-no-bytecode-cache"
    if (not sys.flags.isolated or not sys.flags.dont_write_bytecode
            or not sys.flags.no_user_site or sys.pycache_prefix != str(cache)
            or cache.exists()):
        os.execv(sys.executable, [sys.executable, "-I", "-B", "-X",
                                  "pycache_prefix=" + str(cache),
                                  str(Path(__file__).resolve()), *sys.argv[1:]])


def _token_from_stdin() -> str:
    raw = sys.stdin.readline(4097)
    if len(raw) > 4096:
        raise RuntimeError("AUTHORIZATION_TOKEN_INPUT_TOO_LARGE")
    if raw.endswith("\n"):
        raw = raw[:-1]
    if raw.endswith("\r"):
        raw = raw[:-1]
    if not raw or sys.stdin.read(1):
        raise RuntimeError("AUTHORIZATION_TOKEN_INPUT_INVALID")
    return raw


def _attempt_is_unconsumed(root: Path, consumed_path: Path) -> bool:
    return not consumed_path.exists() and not any(
        (root / name).exists()
        for name in ("run-once", "execution.lock", "supervisor-status.json",
                     "worker-initialization-failure.json", "STOP_REQUESTED.json")
    )


def _write_consumed_marker(path: Path, *, attempt: str, identity: dict,
                           authorization_path: Path, authorization: dict) -> None:
    value = {
        "schema": "w1-local-authorization-consumption/1.0.0",
        "attempt": attempt,
        "execution_manifest_sha256": identity["manifest_sha256"],
        "hashes_sha256": identity["hashes_sha256"],
        "authorization_file": str(authorization_path),
        "authorization_token_sha256": authorization["token_sha256"],
        "consumed_unix_time": time.time(),
        "supervisor_started": False,
    }
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _finite_offset(name: str) -> float:
    raw = os.environ.get(name, "0")
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError("LOCAL_ACCOUNTING_OFFSET_INVALID:" + name) from exc
    if not math.isfinite(value) or value < 0:
        raise RuntimeError("LOCAL_ACCOUNTING_OFFSET_INVALID:" + name)
    return value


def main() -> int:
    _isolated_reexec()
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-file", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--check-token", action="store_true",
                        help="validate an approved token without consuming it or starting the supervisor")
    parser.add_argument("--source-preflight", action="store_true")
    args = parser.parse_args()
    if args.source_preflight and (not args.preflight_only or args.check_token):
        raise RuntimeError("SOURCE_PREFLIGHT_IS_STRUCTURE_ONLY")
    if args.check_token and not args.preflight_only:
        raise RuntimeError("CHECK_TOKEN_REQUIRES_PREFLIGHT")

    from manifest_contract import read_object, verify_external_authorization
    from local_runtime import gpu_snapshot, verify_runtime

    authorization_path = Path(args.authorization_file).resolve()
    if authorization_path.is_relative_to(ROOT):
        raise RuntimeError("AUTHORIZATION_FILE_MUST_BE_OUTSIDE_PACKAGE")
    token = _token_from_stdin() if (not args.preflight_only or args.check_token) else None
    verified = verify_external_authorization(
        ROOT, authorization_path, token=token,
        preflight_only=args.preflight_only and not args.check_token)
    identity = verified["identity"]
    request = verified["request"]
    contract = verified["contract"]
    attempt = request.get("attempt")
    if not isinstance(attempt, str) or not attempt:
        raise RuntimeError("LOCAL_ATTEMPT_IDENTITY_INVALID")
    expected_root = contract.get("native_source_mount") if args.source_preflight else contract.get("native_execution_root")
    if expected_root != str(ROOT):
        raise RuntimeError("LOCAL_NATIVE_EXECUTION_ROOT_MISMATCH")
    if contract.get("native_entry") != Path(__file__).name:
        raise RuntimeError("LOCAL_NATIVE_ENTRY_IDENTITY_MISMATCH")
    runtime_identity_path = ROOT / "local-runtime-identity.json"
    runtime_identity_digest = hashlib.sha256(runtime_identity_path.read_bytes()).hexdigest()
    if (contract.get("runtime_mode") != "local_wsl_g1_gppo_cpu"
            or contract.get("native_python") != sys.executable
            or contract.get("runtime_identity_sha256") != runtime_identity_digest):
        raise RuntimeError("LOCAL_RUNTIME_CONTRACT_BINDING_MISMATCH")
    if contract.get("integration_test") is not False or request.get("research_training") is not True:
        raise RuntimeError("LOCAL_FORMAL_RESEARCH_SCOPE_INVALID")
    matrix = read_object(ROOT / "experiment-matrix.json", "EXPERIMENT_MATRIX")
    if matrix.get("attempt") != attempt or matrix.get("world_model_device") != "cpu":
        raise RuntimeError("LOCAL_MATRIX_IDENTITY_OR_DEVICE_MISMATCH")
    from task_contract import verify_task_inputs
    verify_task_inputs(ROOT)
    from worker_contract import verify_worker_contract
    verify_worker_contract(ROOT, attempt, identity['manifest_sha256'], identity['hashes_sha256'])
    runtime = verify_runtime(request)
    gpu = {"used": False, "device": "cpu"}
    if not set(matrix['cpu_execution']['affinity']).issubset(os.sched_getaffinity(0)):
        raise RuntimeError("FROZEN_CPU_AFFINITY_UNAVAILABLE")

    consumed_path = ROOT.parent / "authorization-consumed.json"
    unconsumed = _attempt_is_unconsumed(ROOT, consumed_path)
    if args.source_preflight:
        native_root=Path(contract['native_execution_root'])
        unconsumed=unconsumed and not native_root.parent.exists()
    if args.preflight_only:
        if args.check_token and verified["authorization"].get("status") != "APPROVED":
            raise RuntimeError("TOKEN_CHECK_REQUIRES_APPROVED_AUTHORIZATION")
        print(json.dumps({
            "status": "local_read_only_preflight_pass",
            "attempt": attempt,
            "manifest_sha256": identity["manifest_sha256"],
            "hashes_sha256": identity["hashes_sha256"],
            "resource_request_sha256": hashlib.sha256(
                (ROOT / "RESOURCE_REQUEST.json").read_bytes()).hexdigest(),
            "runtime": runtime,
            "gpu": gpu,
            "attempt_unconsumed": unconsumed,
            "staging_started": False,
            "worker_started": False,
            "authorization_consumed": False,
            "cuda_initialized": False,
            "token_validated": bool(args.check_token),
        }, sort_keys=True), flush=True)
        return 0 if unconsumed else 1
    if not unconsumed:
        raise RuntimeError("LOCAL_ATTEMPT_ALREADY_CONSUMED_OR_STARTED")
    if verified["authorization"].get("status") != "APPROVED":
        raise RuntimeError("FORMAL_APPROVED_AUTHORIZATION_REQUIRED")

    wall_offset = _finite_offset("W1_LOCAL_STAGING_WALL_SECONDS")
    cpu_offset = _finite_offset("W1_LOCAL_STAGING_CPU_SECONDS")
    wall_offset += max(0.0, time.monotonic() - ENTRY_STARTED_WALL)
    current_self = resource.getrusage(resource.RUSAGE_SELF)
    current_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    prior_entry_cpu = ((ENTRY_STARTED_SELF.ru_utime + ENTRY_STARTED_SELF.ru_stime)
                       + (ENTRY_STARTED_CHILDREN.ru_utime + ENTRY_STARTED_CHILDREN.ru_stime))
    current_entry_cpu = ((current_self.ru_utime + current_self.ru_stime)
                         + (current_children.ru_utime + current_children.ru_stime))
    cpu_offset += max(0.0, current_entry_cpu - prior_entry_cpu)
    if not token:
        raise RuntimeError("FORMAL_AUTHORIZATION_TOKEN_REQUIRED")
    _write_consumed_marker(consumed_path, attempt=attempt, identity=identity,
                           authorization_path=authorization_path,
                           authorization=verified["authorization"])
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    os.environ["W1_VERIFIED_ATTEMPT"] = attempt
    os.environ["W1_VERIFIED_MANIFEST_SHA256"] = identity["manifest_sha256"]
    os.environ["W1_VERIFIED_HASHES_SHA256"] = identity["hashes_sha256"]
    os.environ["W1_EXTERNAL_AUTHORIZATION_FILE"] = str(authorization_path)
    os.environ["W1_EXTERNAL_AUTHORIZATION_TOKEN_SHA256"] = token_hash
    os.environ["W1_LOCAL_STAGING_WALL_SECONDS"] = str(wall_offset)
    os.environ["W1_LOCAL_STAGING_CPU_SECONDS"] = str(cpu_offset)

    from supervise import main as supervise

    result = supervise(
        ROOT,
        runner=str(ROOT / "local_research_worker.py"),
        sample_interval=0.5,
        worker_input=token + "\n",
    )
    token = None
    try:
        consumed = json.loads(consumed_path.read_text(encoding="utf-8"))
        consumed["supervisor_started"] = True
        tmp = consumed_path.with_name(consumed_path.name + ".tmp-" + secrets.token_hex(4))
        tmp.write_text(json.dumps(consumed, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, consumed_path)
    except BaseException:
        traceback.print_exc()
    status_path = ROOT / "supervisor-status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else None
    print(json.dumps({"status": None if status is None else status.get("status"),
                      "returncode": result, "attempt": attempt,
                      "supervisor_status": str(status_path),
                      "authorization_consumed": True,
                      "resource_status": None if status is None else {
                          key: status.get(key) for key in (
                              "wall_seconds", "cpu_seconds", "final_resource_pass",
                              "cpu_scope_complete", "gpu_allocator_peak_bytes")}},
                     sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
    raise SystemExit(exit_code)
