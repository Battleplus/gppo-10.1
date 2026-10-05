"""Bounded synthetic runtime check; no research input, model or checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-python", required=True)
    parser.add_argument("--evidence-root", required=True)
    args = parser.parse_args()
    evidence = Path(args.evidence_root)
    evidence.mkdir(parents=True, exist_ok=True)
    output = evidence / "synthetic-probe.json"
    if output.exists():
        raise SystemExit("PROBE_EVIDENCE_ALREADY_EXISTS_NO_RETRY")
    started = time.monotonic()
    cpu_started = time.process_time()
    result = {
        "schema": "w1-remote-runtime-synthetic-probe/1.0.0",
        "status": "started", "formal_attempt_created": False,
        "research_data_read": False, "model_initializations": 0,
        "model_forwards": 0, "checkpoint_reads": 0, "checkpoint_writes": 0,
        "synthetic_counts": {"tensor_forward_checks": 0, "backwards": 0,
                             "optimizer_initializations": 0, "optimizer_updates": 0},
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "python_prefix": sys.prefix, "python_base_prefix": sys.base_prefix,
        "python_path": sys.path, "utc_started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {key: os.environ.get(key) for key in
                        ("PYTHONPATH", "PYTHONHOME", "PYTHONNOUSERSITE", "LD_LIBRARY_PATH", "LD_PRELOAD", "CUDA_VISIBLE_DEVICES")},
    }
    write_json(output, result)
    try:
        if os.path.abspath(sys.executable) != args.expected_python:
            raise RuntimeError("PYTHON_EXECUTABLE_IDENTITY_MISMATCH")
        if sys.prefix == sys.base_prefix or os.path.abspath(sys.prefix) != str(Path(args.expected_python).parent.parent):
            raise RuntimeError("PYTHON_VENV_IDENTITY_MISMATCH")
        if any(os.environ.get(key) for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD")):
            raise RuntimeError("IMPLICIT_LIBRARY_OR_PROJECT_PATH_NOT_ALLOWED")
        import numpy
        import torch

        result.update(torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
                      torch_path=torch.__file__, numpy_version=numpy.__version__, numpy_path=numpy.__file__,
                      torch_build=torch.__config__.show(), cuda_available=torch.cuda.is_available())
        if torch.__version__ != "2.5.1+cu121" or numpy.__version__ != "1.26.4" or torch.version.cuda != "12.1":
            raise RuntimeError("DEPENDENCY_VERSION_MISMATCH")
        for package_file in (torch.__file__, numpy.__file__):
            if not Path(package_file).is_relative_to(sys.prefix):
                raise RuntimeError("PACKAGE_OUTSIDE_FROZEN_VENV")
        if not result["cuda_available"] or torch.cuda.device_count() != 2:
            raise RuntimeError("EXPECTED_TWO_CUDA_DEVICES_NOT_AVAILABLE")
        result["devices"] = []
        for index in range(2):
            properties = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            result["devices"].append({"index": index, "name": properties.name,
                                      "capability": [properties.major, properties.minor],
                                      "total_bytes": total, "free_bytes_before_probe": free})
            if "2080 Ti" not in properties.name or free < 1024 ** 3:
                raise RuntimeError("DEVICE_IDENTITY_OR_MINIMUM_PROBE_MEMORY_FAILED")
        result["tensor_checks"] = []
        for device in ("cpu", "cuda:0", "cuda:1"):
            tensor = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4)
            if device == "cuda:1":
                tensor = tensor.detach().requires_grad_(True)
                optimizer = torch.optim.SGD([tensor], lr=0.001)
                result["synthetic_counts"]["optimizer_initializations"] += 1
            product = tensor @ tensor.T
            result["synthetic_counts"]["tensor_forward_checks"] += 1
            if tuple(product.shape) != (4, 4) or not bool(torch.isfinite(product).all()):
                raise RuntimeError("NONFINITE_OR_INVALID_TENSOR_FORWARD")
            checksum = float(product.sum().item())
            if checksum != 3680.0:
                raise RuntimeError("TENSOR_FORWARD_CHECKSUM_MISMATCH")
            item = {"device": device, "shape": list(product.shape), "checksum": checksum}
            if device == "cuda:1":
                before = tensor.detach().clone()
                product.mean().backward()
                result["synthetic_counts"]["backwards"] += 1
                if tensor.grad is None or not bool(torch.isfinite(tensor.grad).all()):
                    raise RuntimeError("NONFINITE_SYNTHETIC_GRADIENT")
                optimizer.step()
                result["synthetic_counts"]["optimizer_updates"] += 1
                if not bool(torch.isfinite(tensor).all()) or bool(torch.equal(tensor, before)):
                    raise RuntimeError("SYNTHETIC_OPTIMIZER_UPDATE_FAILED")
                item["backward_and_update"] = "pass"
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            result["tensor_checks"].append(item)
        maps = Path("/proc/self/maps").read_text().splitlines()
        loaded = sorted({line.split()[-1] for line in maps if any(
            name in line for name in ("libcuda", "libcublas", "libcusparse", "libnvJitLink", "libcudnn", "libnvrtc"))})
        result["loaded_cuda_libraries"] = loaded
        if any("/usr/local/cuda" in path or "/home/user1/.local/" in path for path in loaded):
            raise RuntimeError("MIXED_SYSTEM_OR_USER_CUDA_LIBRARIES")
        result["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used",
                                               "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout
        lock = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True, check=True).stdout
        lock_path = evidence / "requirements.lock.txt"
        lock_path.write_text(lock, encoding="utf-8")
        result["requirements_lock_sha256"] = digest(lock_path)
        installed = {}
        for distribution in importlib.metadata.distributions():
            name = distribution.metadata["Name"]
            record = Path(distribution._path) / "RECORD"
            installed[name] = {"version": distribution.version,
                               "record_sha256": digest(record) if record.is_file() else None}
        result["installed_distribution_identities"] = installed
        result["python_binary_sha256"] = digest(Path(sys.executable).resolve())
        result["runtime_probe_sha256"] = digest(Path(__file__))
        result["status"] = "pass"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        result["wall_seconds"] = time.monotonic() - started
        result["process_cpu_seconds"] = time.process_time() - cpu_started
        result["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
