"""Hash the Python sources used by a controlled synthetic pipeline witness."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_sources(root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        result[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return result


def _map_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tree_manifest(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size, "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2,
                               ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def before(package: Path, snapshot_path: Path) -> dict[str, Any]:
    sources = _python_sources(package)
    value = {
        "schema": "w1-controlled-synthetic-source-snapshot/1.0.0",
        "package_root": str(package.resolve()),
        "python_source_count": len(sources),
        "python_sources": sources,
        "python_sources_sha256": _map_sha256(sources),
    }
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(snapshot_path, value)
    return value


def after(package: Path, archive: Path, native_copy: Path,
          snapshot_path: Path, pipeline_exit_code: int,
          audit_exit_code: int, process_accounting_path: Path) -> dict[str, Any]:
    before_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    sources_after = _python_sources(package)
    fixture_root = archive / "fixture-package"
    fixture_sources = _python_sources(fixture_root)
    module_map = {}
    evidence_path = archive / "controlled-run-evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    runtime_provenance_path = archive / "verified-export" / "runtime-module-provenance.json"
    runtime_provenance = (
        json.loads(runtime_provenance_path.read_text(encoding="utf-8"))
        if runtime_provenance_path.is_file() else {}
    )
    recorded_modules = runtime_provenance.get("modules", {})
    relative_paths = {
        name: relative for name, relative in recorded_modules.items()
        if not name.endswith(".relative")
        and isinstance(recorded_modules.get(name + ".relative"), str)
        for relative in (recorded_modules[name + ".relative"],)
    }
    for name, actual_path in evidence.get("execution", {}).get("runtime_provenance", {}).get("module_files", {}).items():
        if not isinstance(actual_path, str):
            continue
        parts = Path(actual_path).parts
        if "fixture-package" in parts:
            relative_paths.setdefault(name, Path(*parts[parts.index("fixture-package") + 1:]).as_posix())
    for name, relative in relative_paths.items():
        fixture_file = fixture_root / relative
        source_file = package / relative
        fixture_hash = _sha256(fixture_file) if fixture_file.is_file() else None
        source_hash = _sha256(source_file) if source_file.is_file() else None
        module_map[name] = {
            "relative_path": relative,
            "staged_fixture_path": str(fixture_file),
            "staged_fixture_sha256": fixture_hash,
            "final_package_path": str(source_file),
            "final_package_sha256": source_hash,
            "matches_final_package": fixture_hash is not None and fixture_hash == source_hash,
        }
    source_unchanged = before_snapshot.get("python_sources") == sources_after
    fixture_matches_package = fixture_sources == sources_after
    modules_match = bool(module_map) and all(
        row["matches_final_package"] for row in module_map.values()
    )
    provenance = {
        "schema": "w1-controlled-synthetic-runtime-source-provenance/1.0.0",
        "python_executable": evidence.get("execution", {}).get("runtime_provenance", {}).get("python_executable"),
        "python_version": evidence.get("execution", {}).get("runtime_provenance", {}).get("python_version"),
        "execution_flags": ["-I", "-B", "-X", "pycache_prefix=/tmp/w1-task-pipeline-pycache-20261004"],
        "environment": "env -i; PATH, HOME, TMPDIR, CUDA_VISIBLE_DEVICES, and single-thread BLAS settings explicitly set",
        "source_snapshot_before_sha256": before_snapshot["python_sources_sha256"],
        "source_snapshot_after_sha256": _map_sha256(sources_after),
        "source_snapshot_unchanged": source_unchanged,
        "python_source_count": len(sources_after),
        "fixture_python_source_count": len(fixture_sources),
        "fixture_python_sources_match_final_package": fixture_matches_package,
        "loaded_module_relative_paths": module_map,
        "loaded_modules_match_final_package": modules_match,
        "all_provenance_checks_pass": source_unchanged and fixture_matches_package and modules_match,
    }
    process_accounting = json.loads(process_accounting_path.read_text(encoding="utf-8"))
    invocation = [
        "wsl.exe", "-d", "Ubuntu-24.04", "--exec", "/usr/bin/env", "-i",
        "PATH=/usr/bin:/bin", "HOME=/home/asus", "TMPDIR=/tmp",
        "CUDA_VISIBLE_DEVICES=", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "/home/asus/.venvs/w1-light-repaired-fair-rerun-py31116/bin/python",
        "-I", "-B", "-X", "pycache_prefix=/tmp/w1-task-pipeline-pycache-20261004",
        "-u",
        "/mnt/e/Z博士/research-plans/w1-g1-gppo-task-validation-local-v1/run_controlled_synthetic_pipeline.py",
        "/tmp/w1-task-pipeline-isolated-20261004", "--archive-root",
        "/mnt/e/Z博士/research-plans/w1-g1-gppo-task-validation-local-v1/controlled-synthetic-task-pipeline-evidence-isolated-20261004",
    ]
    evidence["execution"]["run_provenance"] = {
        "invocation_argv": invocation,
        "driver_exit_code": pipeline_exit_code,
        "post_run_audit_exit_code": audit_exit_code,
        "post_run_audit_argv": [
            "wsl.exe", "-d", "Ubuntu-24.04", "--exec", "/usr/bin/env", "-i",
            "PATH=/usr/bin:/bin", "HOME=/home/asus", "TMPDIR=/tmp",
            "/home/asus/.venvs/w1-light-repaired-fair-rerun-py31116/bin/python",
            "-I", "-B", "-X", "pycache_prefix=/tmp/w1-task-pipeline-pycache-20261004",
            "-u",
            "/mnt/e/Z博士/research-plans/w1-g1-gppo-task-validation-local-v1/audit_controlled_synthetic_pipeline.py",
            str(archive.resolve()),
        ],
        "process_accounting": process_accounting,
        "first_attempt": {
            "exit_code": 1,
            "verification_failure": [
                "hidden_replay_observations_match_ledger_encode_charges",
                "reduced_route_budgets",
            ],
            "pipeline_completed": True,
            "nine_policy_routes": True,
            "raw_evidence_retained": False,
            "raw_evidence_note": "The first WSL /tmp tree was not copied before the distro session ended and could not be recovered; only the observed command exit and verifier failure list are retained.",
        },
    }
    evidence["execution"]["module_provenance"] = {
        "module_relative_paths": {
            name: row["relative_path"] for name, row in module_map.items()
        },
        "all_loaded_modules_match_final_package": modules_match,
    }
    evidence["source_provenance"] = provenance
    _write_json(evidence_path, evidence)
    _write_json(archive / "isolated-runtime-provenance.json", provenance)
    _write_json(archive / "process-accounting.json", process_accounting)

    if native_copy.exists():
        raise FileExistsError(f"Native persistent evidence copy already exists: {native_copy}")
    native_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(archive, native_copy)
    archive_manifest = _tree_manifest(archive)
    native_manifest = _tree_manifest(native_copy)
    if archive_manifest != native_manifest:
        raise IOError("PERSISTENT_NATIVE_EVIDENCE_COPY_HASH_MISMATCH")
    manifest_document = {
        "schema": "w1-controlled-synthetic-evidence-copy-manifest/1.0.0",
        "archive_root": str(archive.resolve()),
        "native_copy_root": str(native_copy.resolve()),
        "verified": True,
        "file_count": len(archive_manifest),
        "files": archive_manifest,
        "tree_manifest_sha256": _map_sha256(archive_manifest),
    }
    _write_json(archive.parent / (archive.name + ".tree-manifest.json"), manifest_document)
    _write_json(native_copy.parent / (native_copy.name + ".tree-manifest.json"), manifest_document)
    loaded_digests = {
        row["relative_path"]: row["staged_fixture_sha256"]
        for row in module_map.values()
        if row.get("relative_path") and row.get("staged_fixture_sha256")
    }
    normalized = {
        "schema": "w1-controlled-synthetic-final-evidence/1.0.0",
        "verification": {
            "all_pass": bool(evidence.get("verification", {}).get("all_pass")
                              and provenance["all_provenance_checks_pass"]
                              and audit_exit_code == 0
                              and archive_manifest == native_manifest),
            "pipeline_checks": evidence.get("verification", {}).get("checks", {}),
            "source_provenance_checks": provenance,
            "persistent_native_copy_verified": archive_manifest == native_manifest,
        },
        "loaded_production_source_sha256": loaded_digests,
        "archive_evidence_path": str(archive.resolve()),
        "native_evidence_path": str(native_copy.resolve()),
        "evidence_tree_manifest_sha256": manifest_document["tree_manifest_sha256"],
        "run_provenance": evidence["execution"]["run_provenance"],
    }
    normalized_path = package.parent / "synthetic-pipeline-final-evidence.json"
    _write_json(normalized_path, normalized)
    return {
        "archive_root": str(archive.resolve()),
        "native_copy_root": str(native_copy.resolve()),
        "source_provenance": provenance,
        "final_evidence_path": str(normalized_path.resolve()),
        "pipeline_exit_code": pipeline_exit_code,
        "audit_exit_code": audit_exit_code,
        "process_accounting": process_accounting,
        "copy_verified": archive_manifest == native_manifest,
        "tree_manifest_sha256": manifest_document["tree_manifest_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("before", "after"))
    parser.add_argument("package_root", type=Path)
    parser.add_argument("snapshot_path", type=Path)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--native-copy-root", type=Path)
    parser.add_argument("--pipeline-exit-code", type=int)
    parser.add_argument("--audit-exit-code", type=int)
    parser.add_argument("--process-accounting-path", type=Path)
    args = parser.parse_args()
    if args.mode == "before":
        result = before(args.package_root, args.snapshot_path)
    else:
        if args.archive_root is None or args.native_copy_root is None:
            parser.error("after mode requires --archive-root and --native-copy-root")
        if (args.pipeline_exit_code is None or args.audit_exit_code is None
                or args.process_accounting_path is None):
            parser.error("after mode requires pipeline exit code and process accounting")
        result = after(args.package_root, args.archive_root,
                       args.native_copy_root, args.snapshot_path,
                       args.pipeline_exit_code, args.audit_exit_code,
                       args.process_accounting_path)
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
