"""Freeze installed runtime identities from completed synthetic probe evidence."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_preflight import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", default="runtime-identity-v2.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if Path(args.output_name).name != args.output_name:
        raise SystemExit("IDENTITY_OUTPUT_MUST_BE_A_FILE_NAME")
    output = root / args.output_name
    if output.exists():
        raise SystemExit("RUNTIME_IDENTITY_ALREADY_FROZEN")
    probe = json.loads((root / "synthetic-probe.json").read_text())
    if probe["status"] != "pass" or probe["python_executable"] != sys.executable:
        raise SystemExit("SUCCESSFUL_SAME_RUNTIME_PROBE_REQUIRED")
    import torch
    import numpy
    critical = {Path(torch.__file__), Path(numpy.__file__), Path(torch._C.__file__), Path(sys.prefix) / "pyvenv.cfg"}
    for path in probe["loaded_cuda_libraries"]:
        if path.startswith(sys.prefix):
            critical.add(Path(path))
    # Hash all bundled CUDA and torch native libraries, including lazily loaded dependencies.
    for relative in ("torch/lib", "nvidia"):
        library_root = Path(torch.__file__).parent.parent / relative
        critical.update(path for path in library_root.rglob("*.so*") if path.is_file())
    identity = {
        "schema": "w1-remote-runtime-identity/1.0.0",
        "purpose": "runtime_candidate_only_not_training_authorization",
        "python_executable": sys.executable, "python_prefix": sys.prefix,
        "python_version": probe["python_version"], "python_binary_sha256": probe["python_binary_sha256"],
        "torch_version": probe["torch_version"], "torch_cuda_version": probe["torch_cuda_version"],
        "numpy_version": probe["numpy_version"],
        "synthetic_probe_sha256": sha256_file(root / "synthetic-probe.json"),
        "requirements_lock_sha256": sha256_file(root / "requirements.lock.txt"),
        "installed_distribution_identities": probe["installed_distribution_identities"],
        "critical_files": {str(path): sha256_file(path) for path in sorted(critical)},
        "gpu_identities": [[part.strip() for part in line.split(",")][1:4]
                           for line in probe["nvidia_smi"].strip().splitlines()],
        "required_environment": {"PYTHONNOUSERSITE": "1", "PYTHONPATH": None,
                                 "PYTHONHOME": None, "LD_LIBRARY_PATH": None, "LD_PRELOAD": None},
        "python_flags": ["-I", "-B"], "exclusive_gpu_allocation": "not_proven",
        "bytecode_cache_policy": {"prefix": str(Path(sys.prefix) / ".w1-no-bytecode-cache"),
                                  "must_remain_absent": True, "writes_disabled": True},
        "default_python_repaired": False, "formal_attempt_created": False,
    }
    identity["python_flags"] += ["-X", "pycache_prefix=" + identity["bytecode_cache_policy"]["prefix"]]
    output.write_text(json.dumps(identity, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"runtime_identity_sha256": sha256_file(output), "critical_file_count": len(critical),
                      "formal_attempt_created": False}))


if __name__ == "__main__":
    main()
