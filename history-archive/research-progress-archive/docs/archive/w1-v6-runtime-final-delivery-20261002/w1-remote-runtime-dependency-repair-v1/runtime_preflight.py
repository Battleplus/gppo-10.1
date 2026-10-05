"""Read-only exact-runtime validation. Never allocates a tensor or starts a job."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


class RuntimeContractError(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_record_payload(distribution, prefix):
    record = Path(distribution._path) / "RECORD"
    with record.open(newline="", encoding="utf-8") as stream:
        for relative, encoded_hash, size in csv.reader(stream):
            # These installation-generated caches are unreachable under the enforced prefix.
            if Path(relative).suffix == ".pyc" and "__pycache__" in Path(relative).parts:
                continue
            if Path(relative).suffix == ".pyc" and not encoded_hash:
                raise RuntimeContractError("UNHASHED_LOADABLE_BYTECODE_NOT_ALLOWED:" + relative)
            if not encoded_hash:
                continue
            path = Path(distribution.locate_file(relative)).resolve()
            if not path.is_relative_to(prefix):
                raise RuntimeContractError("DEPENDENCY_FILE_OUTSIDE_VENV:" + relative)
            algorithm, expected = encoded_hash.split("=", 1)
            if algorithm != "sha256":
                raise RuntimeContractError("DEPENDENCY_HASH_ALGORITHM_UNSUPPORTED")
            if size and path.stat().st_size != int(size):
                raise RuntimeContractError("DEPENDENCY_PAYLOAD_SIZE_MISMATCH:" + relative)
            actual = base64.urlsafe_b64encode(bytes.fromhex(sha256_file(path))).rstrip(b"=").decode()
            if actual != expected:
                raise RuntimeContractError("DEPENDENCY_PAYLOAD_DIGEST_MISMATCH:" + relative)


def verify_process_isolation():
    if not sys.flags.isolated or not sys.flags.dont_write_bytecode or not sys.flags.no_user_site:
        raise RuntimeContractError("REQUIRED_ISOLATED_NO_BYTECODE_FLAGS_MISSING")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeContractError("PYTHONNOUSERSITE_REQUIRED")
    cache = Path(sys.prefix) / ".w1-no-bytecode-cache"
    if sys.pycache_prefix != str(cache) or cache.exists():
        raise RuntimeContractError("BYTECODE_CACHE_READ_ISOLATION_REQUIRED")
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    compact = f"python{sys.version_info.major}{sys.version_info.minor}.zip"
    allowed = {str(Path(sys.base_prefix) / "lib" / item) for item in (version, compact, version + "/lib-dynload")}
    allowed.add(str(Path(sys.prefix) / "lib" / version / "site-packages"))
    if set(sys.path) - allowed:
        raise RuntimeContractError("UNEXPECTED_PYTHON_SEARCH_PATH")


def verify(spec):
    if spec.get("schema") != "w1-remote-runtime-identity/1.0.0":
        raise RuntimeContractError("RUNTIME_IDENTITY_SCHEMA_INVALID")
    if spec.get("bytecode_cache_policy") != {"prefix": str(Path(spec["python_prefix"]) / ".w1-no-bytecode-cache"),
                                            "must_remain_absent": True, "writes_disabled": True}:
        raise RuntimeContractError("BYTECODE_CACHE_POLICY_MISSING_OR_INVALID")
    if os.path.abspath(sys.executable) != spec["python_executable"] or sys.prefix != spec["python_prefix"]:
        raise RuntimeContractError("EXPLICIT_PYTHON_IDENTITY_MISMATCH")
    if sha256_file(Path(sys.executable).resolve()) != spec["python_binary_sha256"]:
        raise RuntimeContractError("PYTHON_BINARY_DIGEST_MISMATCH")
    if any(os.environ.get(key) for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONHOME")):
        raise RuntimeContractError("IMPLICIT_PATH_NOT_ALLOWED")
    installed = {d.metadata["Name"].lower().replace("_", "-"): d for d in importlib.metadata.distributions()}
    expected = {k.lower().replace("_", "-"): v for k, v in spec["installed_distribution_identities"].items()}
    if set(installed) != set(expected):
        raise RuntimeContractError("DEPENDENCY_SET_MISMATCH")
    for name, identity in expected.items():
        distribution = installed[name]
        if distribution.version != identity["version"]:
            raise RuntimeContractError("DEPENDENCY_VERSION_MISMATCH:" + name)
        record = Path(distribution._path) / "RECORD"
        if identity["record_sha256"] is not None and sha256_file(record) != identity["record_sha256"]:
            raise RuntimeContractError("DEPENDENCY_RECORD_MISMATCH:" + name)
        if identity["record_sha256"] is not None:
            verify_record_payload(distribution, sys.prefix)
    for path, digest in spec["critical_files"].items():
        if sha256_file(path) != digest:
            raise RuntimeContractError("RUNTIME_FILE_DIGEST_MISMATCH:" + path)
    completed = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.free", "--format=csv,noheader,nounits"],
                               text=True, capture_output=True, timeout=10, check=True)
    rows = [line.split(", ") for line in completed.stdout.strip().splitlines()]
    if len(rows) != 2 or any(row[:3] != expected_row for row, expected_row in zip(rows, spec["gpu_identities"])):
        raise RuntimeContractError("GPU_OR_DRIVER_IDENTITY_MISMATCH")
    return {"status": "pass", "python_executable": sys.executable,
            "formal_attempt_created": False, "staging_started": False, "worker_started": False,
            "model_calls": 0, "tensor_calls": 0, "optimizer_updates": 0,
            "runtime_identity_only": True, "exclusive_resource_allocation_proven": False,
            "gpu_snapshot": completed.stdout}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", required=True)
    parser.add_argument("--identity-sha256", required=True)
    args = parser.parse_args()
    try:
        content = Path(args.identity).read_bytes()
        if hashlib.sha256(content).hexdigest() != args.identity_sha256:
            raise RuntimeContractError("RUNTIME_IDENTITY_OUTER_DIGEST_MISMATCH")
        verify_process_isolation()
        result = verify(json.loads(content))
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                  "formal_attempt_created": False, "staging_started": False, "worker_started": False}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
