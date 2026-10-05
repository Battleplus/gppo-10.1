"""Isolated Windows-to-WSL test for the production collector and settlement path."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from typing import Any


PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / "native"))
sys.path.insert(0, str(PACKAGE))

TEST_ATTEMPT = "w1-label-v6-isolated-collector-settlement-test-once"
NATIVE_PYTHON = "/home/asus/w1-action-relative-auxiliary-runtime-dependency-repair-v1-runtime/bin/python"
TEST_PROFILES = frozenset({"label-qualification"})


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _require_wsl_mount(path: Path, label: str) -> None:
    if not path.is_absolute() or not str(path).startswith("/mnt/"):
        raise RuntimeError(f"{label}_MUST_BE_AN_EXPLICIT_WSL_MOUNT_PATH")


def _copy_test_source(package_root: Path, test_source: Path) -> None:
    ignored = shutil.ignore_patterns(
        "__pycache__", "*.pyc", "run-once", "execution.lock", "supervisor-status.json",
        "resource-history.jsonl", "console.log", "supervisor-console.log",
        "native-launcher-accounting.json", "worker-initialization-failure.json",
    )
    shutil.copytree(package_root, test_source, ignore=ignored)


def _copy_remote_binding_sources(source_parent: Path, target_parent: Path, paths: dict[str, str]) -> dict[str, str]:
    checked = {}
    for relative in paths.values():
        source = source_parent / relative
        target = target_parent / relative
        if not source.is_file():
            raise RuntimeError("REMOTE_BINDING_FIXTURE_SOURCE_MISSING:" + relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_digest, target_digest = _sha256(source), _sha256(target)
        if source_digest != target_digest:
            raise RuntimeError("REMOTE_BINDING_FIXTURE_COPY_MISMATCH:" + relative)
        checked[relative] = source_digest
    return checked


def _apply_micro_matrix(test_source: Path, attempt: str) -> dict[str, Any]:
    split_path = test_source / "parent-split.json"
    split = _read_json(split_path)
    if len(split.get("parents", [])) != 8:
        raise RuntimeError("FROZEN_SOURCE_PARENT_COUNT_NOT_EIGHT")
    first_parent = split["parents"][0]
    split["parents"] = [first_parent]
    _write_json(split_path, split)

    matrix_path = test_source / "experiment-matrix.json"
    matrix = _read_json(matrix_path)
    matrix["parent_count"] = 1
    _write_json(matrix_path, matrix)

    request_path = test_source / "RESOURCE_REQUEST.json"
    request = _read_json(request_path)
    request["attempt"] = attempt
    request["status"] = "NOT_APPROVED"
    request["matrix"]["parent_count"] = 1
    request["matrix"]["repeats_per_parent"] = 1
    _write_json(request_path, request)
    return {"parent_count": 1, "parent": first_parent.get("parent"), "repeat_count": 1}


def _set_test_attempt(test_source: Path) -> None:
    for name in ("RESOURCE_REQUEST.json", "experiment-matrix.json", "parent-split.json"):
        path = test_source / name
        payload = _read_json(path)
        payload["attempt"] = TEST_ATTEMPT
        if name == "RESOURCE_REQUEST.json":
            payload["status"] = "NOT_APPROVED"
        _write_json(path, payload)


def _worker_command(python: str, staged_root: Path, export_root: Path) -> tuple[list[str], dict[str, str]]:
    command = [
        python,
        "-I",
        "-B",
        str(staged_root / "test_isolated_collector_settlement.py"),
        "--staged-worker",
        "--export-root",
        str(export_root),
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPATH", None)
    return command, environment


def _orchestrate(package_root: Path, evidence_root: Path) -> dict[str, Any]:
    package_root, evidence_root = package_root.resolve(), evidence_root.resolve()
    _require_wsl_mount(package_root, "PACKAGE_ROOT")
    _require_wsl_mount(evidence_root, "EVIDENCE_ROOT")
    if not package_root.is_dir() or not (package_root / "freeze_contract.py").is_file():
        raise RuntimeError("PRODUCTION_PACKAGE_SOURCE_MISSING")
    if evidence_root.exists() and any(evidence_root.iterdir()):
        raise RuntimeError("EVIDENCE_DIRECTORY_MUST_BE_EMPTY_NO_OVERWRITE")
    evidence_root.mkdir(parents=True, exist_ok=True)
    test_source = evidence_root / "test-source"
    _copy_test_source(package_root, test_source)
    _set_test_attempt(test_source)

    # Load the production generator and staging modules from the isolated test copy.
    sys.path[:] = [
        item for item in sys.path
        if not (item and _within(Path(item), package_root))
    ]
    sys.path.insert(0, str(test_source / "native"))
    sys.path.insert(0, str(test_source))
    import freeze_contract
    import infra_io
    import manifest_contract

    for module in (freeze_contract, infra_io, manifest_contract):
        if not _within(Path(module.__file__), test_source):
            raise RuntimeError("TEST_SOURCE_PRODUCTION_MODULE_IMPORT_ESCAPED")
    freeze_contract.ROOT = test_source
    freeze_contract.ATTEMPT = TEST_ATTEMPT
    remote_binding_sources = _copy_remote_binding_sources(
        package_root.parent, evidence_root, freeze_contract.REMOTE_BINDING_PATHS,
    )
    freeze_result = freeze_contract.freeze()
    micro_matrix = _apply_micro_matrix(test_source, TEST_ATTEMPT)
    contract_path = test_source / "launch-contract.json"
    contract = _read_json(contract_path)
    contract["resource_request_sha256"] = manifest_contract.sha256_file(test_source / "RESOURCE_REQUEST.json")
    contract["purpose"] = "Isolated one-parent fake-environment production-chain regression; no formal attempt."
    _write_json(contract_path, contract)
    test_identity_written = manifest_contract.write_identity_files(
        test_source, attempt=TEST_ATTEMPT, status="NOT_APPROVED",
    )
    identity = manifest_contract.verify_package(test_source)
    if identity["manifest"]["attempt"] != TEST_ATTEMPT:
        raise RuntimeError("PRODUCTION_GENERATOR_TEST_ATTEMPT_MISMATCH")
    if identity["manifest_sha256"] == freeze_result["execution_manifest_sha256"]:
        raise RuntimeError("MICRO_MATRIX_IDENTITY_WAS_NOT_REGENERATED")
    if identity["hashes_sha256"] != test_identity_written["hashes_sha256"]:
        raise RuntimeError("PRODUCTION_GENERATOR_TEST_HASHES_IDENTITY_MISMATCH")
    runtime_inputs = _read_json(test_source / "runtime-inputs.json")
    source_code_identity = {
        relative: {
            "source_sha256": _sha256(package_root / relative),
            "test_source_sha256": _sha256(test_source / relative),
            "match": _sha256(package_root / relative) == _sha256(test_source / relative),
        }
        for relative in runtime_inputs["staged_modules"]
    }
    if not all(item["match"] for item in source_code_identity.values()):
        raise RuntimeError("TEST_COPY_CHANGED_RUNTIME_STAGED_SOURCE")

    with tempfile.TemporaryDirectory(prefix="w1-label-v6-isolated-stage-", dir="/tmp") as temp_name:
        native_parent = Path(temp_name)
        native_remote_bindings = _copy_remote_binding_sources(evidence_root, native_parent, freeze_contract.REMOTE_BINDING_PATHS)
        staged_root = native_parent / "staged"
        stage_record = infra_io.stage_sealed_package(
            test_source, staged_root, identity["manifest"],
        )
        _require_wsl_mount(evidence_root, "EVIDENCE_ROOT")
        if str(stage_record["filesystem"]["filesystem_type"]).lower() in {"9p", "drvfs", "ntfs", "ntfs3"}:
            raise RuntimeError("STAGED_ROOT_IS_NOT_NATIVE_LINUX_FILESYSTEM")

        export_root = evidence_root / "controlled-export"
        command, environment = _worker_command(NATIVE_PYTHON, staged_root, export_root)
        process = subprocess.run(
            command,
            cwd=staged_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
            check=False,
        )
        (evidence_root / "worker-stdout.txt").write_text(process.stdout, encoding="utf-8")
        (evidence_root / "worker-stderr.txt").write_text(process.stderr, encoding="utf-8")
        try:
            worker_result = json.loads(process.stdout.splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            worker_result = {"stdout_last_line_not_json": process.stdout.splitlines()[-1:]}
        result = {
            "schema": "w1-isolated-production-collector-evidence/1.0.0",
            "status": "pass" if process.returncode == 0 else "failed",
            "production_freeze_result": freeze_result,
            "production_identity_writer_result": test_identity_written,
            "test_source_readonly_verification": {
                "manifest_hashes_verified_after_micro_matrix": True,
                "test_identity_differs_from_formal_source_identity": True,
            },
            "remote_binding_sources_sha256": remote_binding_sources,
            "native_remote_binding_sources_sha256": native_remote_bindings,
            "test_runtime_source_identity_matches_package": source_code_identity,
            "production_generator_modules": {
                module.__name__: str(Path(module.__file__).resolve())
                for module in (freeze_contract, infra_io, manifest_contract)
            },
            "test_attempt": TEST_ATTEMPT,
            "production_stage_result": stage_record,
            "test_overlay": {
                "applied_before_production_identity_write_and_staging": True,
                **micro_matrix,
                "reason": "one controlled fixture unit; not the formal eight-parent matrix",
            },
            "worker_argv": command,
            "worker_exit_code": process.returncode,
            "worker_stderr_sha256": hashlib.sha256(process.stderr.encode("utf-8")).hexdigest(),
            "worker": worker_result,
            "real_environment_called": False,
            "real_model_calls": 0,
            "checkpoint_calls": 0,
            "formal_attempt_created": False,
            "evidence_root": str(evidence_root),
            "native_staged_root": str(staged_root),
            "controlled_export_root": str(export_root),
        }
        _write_json(evidence_root / "integration-evidence.json", result)
        if process.returncode != 0:
            raise RuntimeError(f"STAGED_PRODUCTION_WORKER_FAILED:{process.returncode}")
        return result


def _worker(export_root: Path) -> dict[str, Any]:
    staged_root = Path(__file__).resolve().parent
    export_root = export_root.resolve()
    if not str(staged_root).startswith("/tmp/"):
        raise RuntimeError("WORKER_PACKAGE_MUST_BE_UNDER_NATIVE_TEMP_ROOT")
    _require_wsl_mount(export_root, "CONTROLLED_EXPORT_ROOT")

    from test_production_collector_lifecycle import FakeEnvironment

    import runner
    import manifest_contract
    import production_data
    import infra_io
    import budget_ledger
    import verify_runtime_inputs
    import classical_baselines
    import task_outcome_contract
    import runtime_config_contract
    import public_history
    import public_transition_contract
    import public_event_targets
    from gppo_world import m10_environment
    from gppo_world import graph5, joint_training, task_lifecycle, telemetry

    modules = (
        runner, manifest_contract, production_data, infra_io, budget_ledger, verify_runtime_inputs,
        classical_baselines, task_outcome_contract, runtime_config_contract,
        public_history, public_transition_contract, public_event_targets,
        m10_environment, graph5, joint_training, task_lifecycle, telemetry,
    )
    module_origins = {module.__name__: str(Path(module.__file__).resolve()) for module in modules}
    if any(not _within(Path(path), staged_root) for path in module_origins.values()):
        raise RuntimeError("PROJECT_MODULE_IMPORTED_OUTSIDE_NATIVE_STAGED_ROOT")

    dependency_root = Path(NATIVE_PYTHON).resolve().parent.parent
    import numpy
    import torch
    runtime_spec = _read_json(staged_root / "runtime-environment-spec.json")
    dependency_origins = {"numpy": str(Path(numpy.__file__).resolve()), "torch": str(Path(torch.__file__).resolve())}
    dependency_checks = {}
    for package_name in ("numpy", "torch"):
        expected = runtime_spec["package_sources"][package_name]
        observed = Path(dependency_origins[package_name])
        if observed.as_posix() != str(expected["import_path"]):
            raise RuntimeError(f"FROZEN_{package_name.upper()}_IMPORT_PATH_MISMATCH")
        digest = _sha256(observed)
        if digest != expected["init_sha256"]:
            raise RuntimeError(f"FROZEN_{package_name.upper()}_IMPORT_DIGEST_MISMATCH")
        extension = Path(expected["extension_path"])
        if not extension.is_file() or _sha256(extension) != expected["extension_sha256"]:
            raise RuntimeError(f"FROZEN_{package_name.upper()}_EXTENSION_IDENTITY_MISMATCH")
        dependency_checks[package_name] = {
            "observed_path": str(observed),
            "expected_path": expected["import_path"],
            "init_sha256": digest,
            "extension_path": str(extension),
            "extension_sha256": _sha256(extension),
            "origin": expected["origin"],
        }

    request = _read_json(staged_root / "RESOURCE_REQUEST.json")
    attempt = str(request["attempt"])
    if attempt != TEST_ATTEMPT or request.get("status") != "NOT_APPROVED":
        raise RuntimeError("STAGED_TEST_IDENTITY_OR_APPROVAL_STATUS_INVALID")
    staged_identity = manifest_contract.verify_package(staged_root)
    if staged_identity["manifest"].get("attempt") != TEST_ATTEMPT:
        raise RuntimeError("STAGED_PACKAGE_MANIFEST_ATTEMPT_MISMATCH")
    runner.ROOT = staged_root
    runner.OUTPUT = staged_root / "run-once"
    os.environ["W1_VERIFIED_ATTEMPT"] = attempt

    class CountingFakeEnvironment(FakeEnvironment):
        counts = {"constructed": 0, "reset": 0, "step": 0}

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            type(self).counts["constructed"] += 1

        def reset(self):
            type(self).counts["reset"] += 1
            return super().reset()

        def step(self, action):
            type(self).counts["step"] += 1
            observation, reward, done, info = super().step(action)
            # Keep the production _vector_reward contract intact while the
            # environment boundary remains a deterministic fake.
            uav_count = int(getattr(self.config, "uav_count", 4))
            initial_energy = float(getattr(self.config, "initial_energy", 100.0))
            info["energy"] = {f"uav-{index}": initial_energy for index in range(uav_count)}
            return observation, reward, done, info

    m10_environment.M10Environment = CountingFakeEnvironment

    target_contexts: list[dict[str, Any]] = []
    original_target = production_data.ProductionDataCollector.task_outcome_target

    def capture_production_target(decision, trajectory):
        target_contexts.append({
            "decision": copy.deepcopy(decision),
            "trajectory": copy.deepcopy(trajectory),
        })
        return original_target(decision, trajectory)

    production_data.ProductionDataCollector.task_outcome_target = staticmethod(capture_production_target)

    forward_calls = {"torch_module_calls": 0}
    original_call_impl = torch.nn.Module._call_impl

    def count_module_call(module, *args, **kwargs):
        forward_calls["torch_module_calls"] += 1
        return original_call_impl(module, *args, **kwargs)

    torch.nn.Module._call_impl = count_module_call
    try:
        code = runner.main()
    finally:
        torch.nn.Module._call_impl = original_call_impl
    if code != 0:
        raise RuntimeError(f"PRODUCTION_RUNNER_RETURNED:{code}")

    output = runner.OUTPUT
    windows_path = output / "world-model-windows.jsonl"
    persisted = [json.loads(line) for line in windows_path.read_text(encoding="utf-8").splitlines()]
    if len(persisted) != 1 or persisted[0].get("status") != "complete":
        raise RuntimeError("PRODUCTION_COLLECTOR_DID_NOT_PERSIST_ONE_COMPLETE_WINDOW")
    window = persisted[0]
    labels = window.get("task_outcome_target", [])
    if not labels or len(labels) != len(target_contexts):
        raise RuntimeError("PERSISTED_LABEL_CONTEXT_COUNT_MISMATCH")

    from task_outcome_contract import derive_action_conditioned_task_outcome

    for index, (label, context) in enumerate(zip(labels, target_contexts)):
        reread_contract_label = derive_action_conditioned_task_outcome(
            context["decision"], context["trajectory"],
        )
        if reread_contract_label != label:
            raise RuntimeError(f"PERSISTED_LABEL_CONTRACT_RECHECK_MISMATCH:{index}")
        if label["candidate_id"] != window["candidate_ids"][index]:
            raise RuntimeError(f"PERSISTED_LABEL_CANDIDATE_ID_MISMATCH:{index}")
        if label["continuation_id"] != window["continuation_id"]:
            raise RuntimeError(f"PERSISTED_LABEL_CONTINUATION_ID_MISMATCH:{index}")
    if len(set(label["public_input_hash"] for label in labels)) != 1:
        raise RuntimeError("FROZEN_DECISION_INPUT_IDENTITY_DIFFERS_ACROSS_CANDIDATES")

    sqlite_path = output / "budget.sqlite3"
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        call_rows = connection.execute(
            "SELECT stage,name,amounts,status FROM calls ORDER BY id"
        ).fetchall()
    finally:
        connection.close()
    status_counts: dict[str, int] = {}
    ledger_totals: dict[str, int] = {}
    for _stage, _name, amounts_json, status in call_rows:
        status_counts[status] = status_counts.get(status, 0) + 1
        for name, amount in json.loads(amounts_json).items():
            ledger_totals[name] = ledger_totals.get(name, 0) + int(amount)
    settlement = _read_json(output / "resource-settlement.json")
    if status_counts.get("pending", 0) != 0 or status_counts.get("failed", 0) != 0:
        raise RuntimeError("PRODUCTION_LEDGER_HAS_UNSETTLED_OR_FAILED_CALLS")
    if settlement["ledger"]["pending_calls"] != 0 or settlement["ledger"]["totals"] != ledger_totals:
        raise RuntimeError("PRODUCTION_LEDGER_SETTLEMENT_DOES_NOT_MATCH_SQLITE")
    if settlement["model_calls"] != {
        "initializations_or_loads": 0,
        "checkpoint_loads_or_writes": 0,
        "forwards": 0,
        "optimizer_updates": 0,
    } or forward_calls["torch_module_calls"] != 0:
        raise RuntimeError("MODEL_OR_CHECKPOINT_CALL_OCCURRED_IN_LABEL_TEST")

    completion = infra_io.controlled_export(output, export_root)
    exported_manifest = _read_json(export_root / "export-manifest.json")
    for relative, identity in exported_manifest["files"].items():
        exported_path = export_root / relative
        if not exported_path.is_file() or _sha256(exported_path) != identity["sha256"]:
            raise RuntimeError("CONTROLLED_EXPORT_FILE_IDENTITY_MISMATCH:" + relative)
    exported_window = export_root / "world-model-windows.jsonl"
    exported_rows = [json.loads(line) for line in exported_window.read_text(encoding="utf-8").splitlines()]
    if exported_rows != persisted:
        raise RuntimeError("CONTROLLED_EXPORT_WINDOW_BYTES_CHANGED")

    model_artifacts = [
        path.name for path in output.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt"}
    ]
    if model_artifacts:
        raise RuntimeError("UNEXPECTED_MODEL_CHECKPOINT_ARTIFACT")
    status = _read_json(output / "status.json")
    return {
        "status": "pass",
        "attempt": attempt,
        "runner_exit_code": code,
        "runner_status": status,
        "coverage_summary": _read_json(output / "label-coverage-summary.json"),
        "window_count": len(persisted),
        "candidate_count": len(labels),
        "first_persisted_candidate": {
            "candidate_id": labels[0]["candidate_id"],
            "action_id": labels[0]["action_id"],
            "continuation_id": labels[0]["continuation_id"],
            "label_contract_reloaded_and_rechecked": True,
            "public_input_hash": labels[0]["public_input_hash"],
        },
        "environment_fake_counts": dict(CountingFakeEnvironment.counts),
        "environment_backend": "test_production_collector_lifecycle.FakeEnvironment; lower environment boundary only",
        "sqlite_call_count": len(call_rows),
        "sqlite_call_status_counts": status_counts,
        "sqlite_ledger_totals": ledger_totals,
        "settlement_status": settlement["status"],
        "settlement_matches_sqlite": True,
        "torch_module_forward_calls": forward_calls["torch_module_calls"],
        "model_and_checkpoint_calls": settlement["model_calls"],
        "controlled_export": completion,
        "export_file_count": len(exported_manifest["files"]),
        "export_manifest_sha256": _sha256(export_root / "export-manifest.json"),
        "export_complete_sha256": _sha256(export_root / "EXPORT_COMPLETE.json"),
        "module_origins": module_origins,
        "dependency_origins": dependency_checks,
        "project_modules_all_from_staged_root": True,
        "dependencies_all_from_frozen_runtime": True,
        "source_tape_identity_verified_by_production_runner": True,
        "test_overlay_applied_after_staging": True,
        "real_environment_called": False,
        "formal_attempt_created": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orchestrate", action="store_true")
    parser.add_argument("--staged-worker", action="store_true")
    parser.add_argument("--package-root")
    parser.add_argument("--evidence-root")
    parser.add_argument("--export-root")
    args = parser.parse_args()
    if args.staged_worker:
        if not args.export_root:
            raise RuntimeError("WORKER_EXPORT_ROOT_REQUIRED")
        result = _worker(Path(args.export_root))
    elif args.orchestrate:
        if not args.package_root or not args.evidence_root:
            raise RuntimeError("ORCHESTRATOR_ROOTS_REQUIRED")
        result = _orchestrate(Path(args.package_root), Path(args.evidence_root))
    else:
        raise RuntimeError("SELECT_ORCHESTRATOR_OR_STAGED_WORKER_MODE")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException as exc:
        print(json.dumps({
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "formal_attempt_created": False,
            "real_environment_called": False,
        }, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        raise
    else:
        raise SystemExit(exit_code)
