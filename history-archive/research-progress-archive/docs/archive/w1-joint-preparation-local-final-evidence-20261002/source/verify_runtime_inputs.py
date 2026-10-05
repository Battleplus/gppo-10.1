"""Check the frozen source tape and production source modules before any reset."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from pathlib import Path


class RuntimeInputError(RuntimeError):
    pass


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _mapping(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise RuntimeInputError(label + "_NOT_OBJECT")
    return value


def _digest(value, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RuntimeInputError(label + "_DIGEST_INVALID")
    return value


def _relative_file(value, label: str) -> str:
    if (not isinstance(value, str) or not value or Path(value).is_absolute()
            or ".." in Path(value).parts):
        raise RuntimeInputError(label + "_PATH_INVALID")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _function_fingerprint(path: Path, function_name: str) -> str:
    """Compare a reviewed function without depending on file-level append-only text."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise RuntimeInputError("DERIVED_FUNCTION_SOURCE_INVALID:" + function_name) from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return ast.dump(node, annotate_fields=True, include_attributes=False)
    raise RuntimeInputError("DERIVED_FUNCTION_MISSING:" + function_name)


def verify_runtime_inputs(package_root: Path) -> dict:
    try:
        inputs = json.loads((package_root / "runtime-inputs.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeInputError(f"RUNTIME_INPUTS_INVALID:{type(exc).__name__}") from exc
    inputs = _mapping(inputs, "RUNTIME_INPUTS")
    if inputs.get("schema") != "w1-action-conditioned-task-outcome-runtime-inputs/2.0.0":
        raise RuntimeInputError("RUNTIME_INPUTS_SCHEMA_INVALID")
    source = _mapping(inputs.get("source_run"), "SOURCE_RUN")
    required_source_fields = {
        "windows_root", "wsl_root", "execution_manifest_sha256",
        "train_tape_file", "train_tape_sha256",
    }
    if required_source_fields - source.keys():
        raise RuntimeInputError("SOURCE_RUN_FIELDS_MISSING")
    source_manifest_digest = _digest(source.get("execution_manifest_sha256"), "SOURCE_MANIFEST")
    tape_file = _relative_file(source.get("train_tape_file"), "TRAIN_TAPE")
    tape_digest = _digest(source.get("train_tape_sha256"), "TRAIN_TAPE")
    portable = inputs.get("purpose") == "portable source provenance checked by joint_inputs.verify_joint_inputs"
    root_value = str(package_root.resolve() / "native" / "source-evidence") if portable else (
        source["windows_root"] if os.name == "nt" else source["wsl_root"]
    )
    if not isinstance(root_value, str) or not root_value:
        raise RuntimeInputError("SOURCE_ROOT_IDENTITY_INVALID")
    source_root = Path(root_value)
    if not source_root.is_absolute() or not source_root.is_dir():
        raise RuntimeInputError("SOURCE_ROOT_MISSING_OR_NOT_ABSOLUTE")
    manifest_path = source_root / "execution-manifest.json"
    if sha256_file(manifest_path) != source_manifest_digest:
        raise RuntimeInputError("SOURCE_EXECUTION_MANIFEST_DIGEST_MISMATCH")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeInputError("SOURCE_EXECUTION_MANIFEST_INVALID") from exc
    manifest = _mapping(manifest, "SOURCE_EXECUTION_MANIFEST")
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict):
        raise RuntimeInputError("SOURCE_EXECUTION_MANIFEST_SCHEMA_INVALID")
    checked = {}
    source_modules = _mapping(inputs.get("source_modules"), "SOURCE_MODULES")
    staged = _mapping(inputs.get("staged_modules"), "STAGED_MODULES")
    derived = _mapping(inputs.get("derived_modules"), "DERIVED_MODULES")
    if (not source_modules or not staged or set(source_modules) & set(derived)
            or set(source_modules) | set(derived) != set(staged)):
        raise RuntimeInputError("STAGED_PRODUCTION_MODULE_SET_MISMATCH")
    for relative, expected in source_modules.items():
        _relative_file(relative, "SOURCE_MODULE")
        _digest(expected, "SOURCE_MODULE")
    for relative, expected in staged.items():
        _relative_file(relative, "STAGED_MODULE")
        _digest(expected, "STAGED_MODULE")
    required_source = dict(source_modules)
    required_source[tape_file] = tape_digest
    for relative, expected in required_source.items():
        actual = manifest_files.get(relative)
        if actual != expected:
            raise RuntimeInputError("SOURCE_MANIFEST_FILE_IDENTITY_MISMATCH:" + relative)
        path = package_root / relative if portable and relative in source_modules else source_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeInputError("SOURCE_FILE_DIGEST_MISMATCH:" + relative)
        checked["source:" + relative] = expected
    for relative, expected in staged.items():
        path = package_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeInputError("STAGED_MODULE_DIGEST_MISMATCH:" + relative)
        checked["staged:" + relative] = expected
    for relative, record in derived.items():
        if not isinstance(record, dict):
            raise RuntimeInputError("DERIVED_MODULE_RECORD_INVALID:" + relative)
        if (record.get("kind") != "derived_pure_function"
                or record.get("source_manifest_membership") is not False
                or relative in manifest_files):
            raise RuntimeInputError("DERIVED_MODULE_PROVENANCE_INVALID:" + relative)
        if record.get("sha256") != staged.get(relative):
            raise RuntimeInputError("DERIVED_MODULE_STAGED_DIGEST_MISMATCH:" + relative)
        provenance = _mapping(record.get("derived_from"), "DERIVED_MODULE_SOURCE")
        package_artifact = _relative_file(
            provenance.get("artifact_package_path"), "DERIVED_PACKAGE_ARTIFACT",
        )
        artifact_path = package_root / package_artifact
        artifact_digest = _digest(provenance.get("artifact_sha256"), "DERIVED_SOURCE")
        symbol = provenance.get("symbol")
        if (symbol != "transparent_utility_components"
                or not artifact_path.is_file()
                or sha256_file(artifact_path) != artifact_digest
                or relative != "transparent_utility.py"):
            raise RuntimeInputError("DERIVED_MODULE_SOURCE_RELATION_INVALID:" + relative)
        staged_path = package_root / relative
        if _function_fingerprint(staged_path, symbol) != _function_fingerprint(artifact_path, symbol):
            raise RuntimeInputError("DERIVED_FUNCTION_EQUIVALENCE_MISMATCH:" + symbol)
        extension_symbols = record.get("package_extension_symbols", [])
        if not isinstance(extension_symbols, list) or any(
                not isinstance(item, str) or not item for item in extension_symbols):
            raise RuntimeInputError("DERIVED_EXTENSION_SYMBOLS_INVALID:" + relative)
        for extension in extension_symbols:
            _function_fingerprint(staged_path, extension)
        extension_test = record.get("package_extension_equivalence_test")
        extension_test_digest = record.get("package_extension_equivalence_test_sha256")
        if extension_symbols:
            extension_test = _relative_file(extension_test, "DERIVED_EXTENSION_TEST")
            extension_test_digest = _digest(extension_test_digest, "DERIVED_EXTENSION_TEST")
            if (not (package_root / extension_test).is_file()
                    or sha256_file(package_root / extension_test) != extension_test_digest):
                raise RuntimeInputError("DERIVED_EXTENSION_TEST_IDENTITY_MISMATCH:" + relative)
        equivalence_test = _relative_file(provenance.get("equivalence_test"), "DERIVED_EQUIVALENCE_TEST")
        equivalence_digest = _digest(provenance.get("equivalence_test_sha256"), "DERIVED_EQUIVALENCE_TEST")
        equivalence_path = package_root / equivalence_test
        if not equivalence_path.is_file() or sha256_file(equivalence_path) != equivalence_digest:
            raise RuntimeInputError("DERIVED_EQUIVALENCE_TEST_IDENTITY_MISMATCH:" + relative)
        checked["derived:" + relative] = record["sha256"]
    if not portable and sys.platform != "win32" and not str(source_root).startswith("/mnt/"):
        raise RuntimeInputError("SOURCE_ROOT_NOT_EXPLICIT_WSL_MOUNT")
    return {
        "schema": "w1-action-conditioned-task-outcome-runtime-input-check/2.0.0",
        "status": "pass",
        "source_root": str(source_root),
        "project_import_root": str(package_root.resolve()),
        "source_execution_manifest_sha256": source_manifest_digest,
        "train_tape_sha256": tape_digest,
        "source_module_count": len(source_modules),
        "derived_module_count": len(derived),
        "verified_files": checked,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    print(json.dumps(verify_runtime_inputs(root), ensure_ascii=False, sort_keys=True))
