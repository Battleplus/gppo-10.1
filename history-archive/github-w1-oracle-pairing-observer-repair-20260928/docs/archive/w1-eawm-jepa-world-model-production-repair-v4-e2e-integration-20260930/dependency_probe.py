"""Read-only dependency probe for the sealed WSL runtime."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path


def probe(*, expected_python: str | None = None, cpu_only: bool = True) -> dict:
    expected_venv = None
    if expected_python:
        expected_path = Path(expected_python)
        expected_venv = expected_path.parent.parent
        if Path(sys.executable).resolve() != expected_path.resolve():
            raise RuntimeError(f"PYTHON_IDENTITY_MISMATCH:{sys.executable}")
        if Path(sys.prefix).resolve() != expected_venv.resolve():
            raise RuntimeError(f"PYTHON_PREFIX_MISMATCH:{sys.prefix}")
    import numpy
    import torch

    cuda = torch.version.cuda
    if cpu_only and cuda is not None:
        raise RuntimeError(f"CPU_ONLY_CONTRACT_FAILED:{cuda}")
    torch.manual_seed(0)
    witness = torch.arange(16, dtype=torch.float32, device="cpu").reshape(4, 4)
    witness_result = witness @ witness.T
    if tuple(witness_result.shape) != (4, 4) or not torch.isfinite(witness_result).all():
        raise RuntimeError("CPU_KERNEL_WITNESS_FAILED")
    result = {
        "schema": "w1-runtime-dependency-probe/1.0.0",
        "python_executable_raw": sys.executable,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_expected_executable": expected_python,
        "python_prefix": sys.prefix,
        "python_base_prefix": sys.base_prefix,
        "python_venv_root": str(expected_venv) if expected_venv else None,
        "python_venv_active": sys.prefix != sys.base_prefix,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_build": torch.__config__.show(),
        "torch_cpu_kernel_witness": {
            "device": str(witness_result.device),
            "shape": list(witness_result.shape),
            "checksum": float(witness_result.sum().item()),
        },
        "numpy_version": numpy.__version__,
        "cpu_only": cpu_only,
        "model_initialized": False,
        "checkpoint_loaded": False,
        "training_started": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-python")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = probe(expected_python=args.expected_python)
    except BaseException as exc:
        print(json.dumps({"schema": "w1-runtime-dependency-probe/1.0.0", "status": "failed", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    result["status"] = "pass"
    print(json.dumps(result, sort_keys=True) if args.json else json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
