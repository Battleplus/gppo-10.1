"""Read-only dependency probe for the sealed WSL runtime."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_file(path_value: str, expected: str, label: str) -> Path:
    path = Path(path_value).resolve()
    if not path.is_file() or _sha256(path) != expected:
        raise RuntimeError(f"RUNTIME_FILE_IDENTITY_MISMATCH:{label}")
    return path


_PROJECT_ROOT_MODULES = frozenset({
    "production_data", "transparent_utility", "infra_io", "public_history",
    "public_transition_contract", "public_event_targets", "task_outcome_contract",
    "runtime_config_contract", "classical_baselines", "public_controller", "dependency_probe",
    "verify_runtime_inputs",
})
_REMOTE_BINDING_ARTIFACTS = frozenset({
    "runtime_identity_v2", "delivery_hashes", "final_readonly_preflight",
})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _is_project_module(name: str) -> bool:
    return name in _PROJECT_ROOT_MODULES or name == "gppo_world" or name.startswith("gppo_world.")


def _verify_project_module_origins(root: Path) -> dict[str, str]:
    expected_root = root.resolve()
    paths = {}
    for name, module in sorted(sys.modules.items()):
        if not _is_project_module(name):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is not None:
            module_paths = [Path(module_file).resolve()]
        else:
            spec = getattr(module, "__spec__", None)
            locations = getattr(spec, "submodule_search_locations", None)
            if locations is None:
                raise RuntimeError(f"PROJECT_IMPORT_ORIGIN_UNVERIFIABLE:{name}")
            module_paths = [Path(value).resolve() for value in locations]
            if not module_paths or any(not path.is_dir() for path in module_paths):
                raise RuntimeError(f"PROJECT_IMPORT_ORIGIN_UNVERIFIABLE:{name}")
        for path in module_paths:
            if not path.is_relative_to(expected_root):
                raise RuntimeError(f"PROJECT_IMPORT_OUTSIDE_STAGED_ROOT:{name}:{path}")
            if module_file is not None and not path.is_file():
                raise RuntimeError(f"PROJECT_IMPORT_ORIGIN_NOT_FILE:{name}:{path}")
            if module_file is not None and path.suffix == ".pyc":
                raise RuntimeError(f"PROJECT_IMPORT_BYTECODE_NOT_ALLOWED:{name}:{path}")
        paths[name] = os.pathsep.join(str(path) for path in module_paths)
    return paths


def _prioritize_project_paths(root: Path) -> None:
    project_paths = [str((root / "native").resolve()), str(root.resolve())]
    for value in project_paths:
        while value in sys.path:
            sys.path.remove(value)
    sys.path[:0] = project_paths


def _validate_remote_runtime_binding(root: Path) -> dict:
    path = root / "remote-runtime-audit-binding.json"
    try:
        binding = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"REMOTE_RUNTIME_BINDING_INVALID:{type(exc).__name__}") from exc
    if not isinstance(binding, dict) or binding.get("schema") != "w1-remote-runtime-audit-binding/1.0.0":
        raise RuntimeError("REMOTE_RUNTIME_BINDING_SCHEMA_INVALID")
    if (binding.get("scope") != "future-audit-only"
            or binding.get("remote_execution_during_v6") is not False
            or binding.get("remote_torch_import_during_v6") is not False
            or binding.get("training_authorization") is not False
            or binding.get("v6_cpu_runtime_modified") is not False
            or binding.get("v6_resource_request_modified") is not False):
        raise RuntimeError("REMOTE_RUNTIME_BINDING_SCOPE_INVALID")
    artifacts = binding.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != _REMOTE_BINDING_ARTIFACTS:
        raise RuntimeError("REMOTE_RUNTIME_BINDING_ARTIFACTS_INVALID")
    for name, record in artifacts.items():
        if (not isinstance(record, dict)
                or not isinstance(record.get("research_plans_relative_path"), str)
                or not record["research_plans_relative_path"]
                or Path(record["research_plans_relative_path"]).is_absolute()
                or ".." in Path(record["research_plans_relative_path"]).parts
                or not isinstance(record.get("sha256"), str)
                or _SHA256.fullmatch(record["sha256"]) is None):
            raise RuntimeError("REMOTE_RUNTIME_BINDING_ARTIFACT_IDENTITY_INVALID:" + name)
    facts = binding.get("preflight_facts")
    if (not isinstance(facts, dict) or facts.get("status") != "pass"
            or facts.get("exit_code") != 0
            or facts.get("local_remote_file_identity_equal") is not True
            or facts.get("before_after_file_identity_equal") is not True
            or facts.get("file_count_including_outer_manifest") != 27
            or facts.get("runtime_identity_only") is not True
            or facts.get("torch_imported") is not False
            or facts.get("staging_started") is not False
            or facts.get("worker_started") is not False
            or facts.get("formal_attempt_created") is not False
            or facts.get("exclusive_resource_allocation_proven") is not False):
        raise RuntimeError("REMOTE_RUNTIME_BINDING_PREFLIGHT_FACTS_INVALID")
    return {
        "schema": binding["schema"],
        "scope": binding["scope"],
        "artifact_count": len(artifacts),
        "remote_execution_during_v6": False,
        "remote_torch_import_during_v6": False,
        "training_authorization": False,
    }


def _verify_runtime_package_files(root: Path, numpy, torch) -> dict:
    spec_path = root / "runtime-environment-spec.json"
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"RUNTIME_ENVIRONMENT_SPEC_INVALID:{type(exc).__name__}") from exc
    if not isinstance(spec, dict) or spec.get("schema") != "w1-runtime-environment-spec/2.0.0":
        raise RuntimeError("RUNTIME_ENVIRONMENT_SPEC_SCHEMA_INVALID")
    if (spec.get("interpreter") != os.path.abspath(sys.executable)
            or spec.get("python_prefix") != str(Path(sys.prefix).resolve())
            or spec.get("python_version") != platform.python_version()):
        raise RuntimeError("FROZEN_PYTHON_IDENTITY_MISMATCH")
    if spec.get("include_system_site_packages") is not True:
        raise RuntimeError("FROZEN_VENV_SYSTEM_SITE_SETTING_MISMATCH")
    pyvenv = Path(sys.prefix) / "pyvenv.cfg"
    if _sha256(Path(sys.executable).resolve()) != spec.get("python_binary_sha256"):
        raise RuntimeError("FROZEN_PYTHON_BINARY_DIGEST_MISMATCH")
    if _sha256(pyvenv) != spec.get("pyvenv_cfg_sha256"):
        raise RuntimeError("FROZEN_PYVENV_CONFIG_DIGEST_MISMATCH")
    if torch.__version__ != spec.get("packages", {}).get("torch"):
        raise RuntimeError("FROZEN_TORCH_VERSION_MISMATCH")
    if numpy.__version__ != spec.get("packages", {}).get("numpy"):
        raise RuntimeError("FROZEN_NUMPY_VERSION_MISMATCH")
    sources = spec.get("package_sources")
    if not isinstance(sources, dict) or set(sources) != {"numpy", "torch"}:
        raise RuntimeError("FROZEN_RUNTIME_PACKAGE_SOURCES_INVALID")
    critical = {}
    for name, module, extension in (
        ("numpy", numpy, numpy.core._multiarray_umath),
        ("torch", torch, torch._C),
    ):
        source = sources[name]
        if not isinstance(source, dict):
            raise RuntimeError("FROZEN_RUNTIME_PACKAGE_SOURCE_INVALID:" + name)
        init_path = Path(module.__file__).resolve()
        extension_path = Path(extension.__file__).resolve()
        if str(init_path) != source.get("import_path"):
            raise RuntimeError("FROZEN_PACKAGE_IMPORT_PATH_MISMATCH:" + name)
        if str(extension_path) != source.get("extension_path"):
            raise RuntimeError("FROZEN_PACKAGE_EXTENSION_PATH_MISMATCH:" + name)
        _check_file(str(init_path), source.get("init_sha256", ""), name + "_init")
        _check_file(str(extension_path), source.get("extension_sha256", ""), name + "_extension")
        critical[name] = {
            "version": module.__version__,
            "import_path": str(init_path),
            "extension_path": str(extension_path),
            "init_sha256": _sha256(init_path),
            "extension_sha256": _sha256(extension_path),
        }
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("FROZEN_RUNTIME_IS_NOT_CPU_ONLY")
    required_distributions = {}
    lock_path = root / spec.get("lock_file", "runtime-dependency-lock.txt")
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#") or value.startswith("python=="):
            continue
        if "==" not in value:
            raise RuntimeError("RUNTIME_DEPENDENCY_LOCK_LINE_INVALID")
        name, version = value.split("==", 1)
        required_distributions[name.lower().replace("_", "-")] = version
    for name, version in required_distributions.items():
        try:
            installed_version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed_version = None
        if installed_version != version:
            raise RuntimeError("FROZEN_DEPENDENCY_VERSION_MISMATCH:" + name)
    return {
        "schema": spec["schema"],
        "python_executable": os.path.abspath(sys.executable),
        "python_version": platform.python_version(),
        "python_binary_sha256": spec["python_binary_sha256"],
        "pyvenv_cfg_sha256": spec["pyvenv_cfg_sha256"],
        "include_system_site_packages": True,
        "packages": critical,
        "locked_distribution_count": len(required_distributions),
        "cuda_build": torch.version.cuda,
    }


def probe(*, expected_python: str | None = None, cpu_only: bool = True,
          expected_root: Path | None = None) -> dict:
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

    project_module_paths = {}
    if expected_root is not None:
        root = Path(expected_root).resolve()
        # Reject any same-name module imported before this probe, then ensure
        # all future project imports resolve from the staged package first.
        _verify_project_module_origins(root)
        _prioritize_project_paths(root)
        from production_data import ProductionDataCollector, _native_imports
        from verify_runtime_inputs import verify_runtime_inputs

        verify_runtime_inputs(root)
        _native_imports(root)
        module_names = (
            "production_data", "transparent_utility", "infra_io", "public_history",
            "public_transition_contract", "public_event_targets", "task_outcome_contract",
            "runtime_config_contract", "classical_baselines", "public_controller", "gppo_world",
            "gppo_world.graph5", "gppo_world.joint_consequence_baseline",
            "gppo_world.joint_training", "gppo_world.m10_environment", "gppo_world.telemetry",
        )
        import importlib
        for name in module_names:
            importlib.import_module(name)
        project_module_paths = _verify_project_module_origins(root)
        if not all(name in project_module_paths for name in module_names):
            raise RuntimeError("PROJECT_IMPORT_CLOSURE_INCOMPLETE")
        if ProductionDataCollector.__module__ != "production_data":
            raise RuntimeError("PRODUCTION_COLLECTOR_IMPORT_IDENTITY_INVALID")
        runtime_identity = _verify_runtime_package_files(root, numpy, torch)
        remote_runtime_binding = _validate_remote_runtime_binding(root)
    else:
        runtime_identity = None
        remote_runtime_binding = None

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
        "project_module_paths": project_module_paths,
        "runtime_identity": runtime_identity,
        "remote_runtime_binding": remote_runtime_binding,
        "cpu_only": cpu_only,
        "synthetic_kernel_matmuls": 1,
        "synthetic_kernel_backwards": 0,
        "synthetic_optimizer_updates": 0,
        "model_forwards": 0,
        "model_initialized": False,
        "checkpoint_loaded": False,
        "training_started": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-python")
    parser.add_argument("--expected-root")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = probe(expected_python=args.expected_python,
                       expected_root=Path(args.expected_root) if args.expected_root else None)
    except BaseException as exc:
        print(json.dumps({"schema": "w1-runtime-dependency-probe/1.0.0", "status": "failed", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    result["status"] = "pass"
    print(json.dumps(result, sort_keys=True) if args.json else json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
