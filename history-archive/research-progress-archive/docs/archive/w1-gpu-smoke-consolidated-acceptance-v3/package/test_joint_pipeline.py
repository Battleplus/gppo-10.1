"""Production pipeline integration; only the environment constructor is replaced."""
from __future__ import annotations
import copy
import hashlib
import hmac
import io
import json
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import sys
import tempfile
import time
import traceback
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "native"))
from budget_ledger import BudgetLedger
from freeze_joint_package import freeze
from infra_io import controlled_export, durable_atomic_json
from joint_pipeline import run_pipeline
from worker_contract import verify_worker_contract
from runner import initialize_output, _settlement
import runner
from test_production_collector_lifecycle import FakeEnvironment


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def _traceback_text(exc):
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _ledger_counters(path):
    if not path.is_file():
        return None
    connection = None
    try:
        connection = sqlite3.connect(path)
        rows = connection.execute("SELECT amounts,status FROM calls ORDER BY id").fetchall()
        totals, statuses = {}, {}
        for amounts_json, status in rows:
            statuses[status] = statuses.get(status, 0) + 1
            for name, amount in json.loads(amounts_json).items():
                totals[name] = totals.get(name, 0) + int(amount)
        return {"available": True, "totals": totals, "call_status_counts": statuses}
    except BaseException as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if connection is not None:
            connection.close()


def _copy_tree(source, destination):
    if source.is_dir() and not (destination / "EXPORT_COMPLETE.json").is_file():
        shutil.copytree(source, destination, dirs_exist_ok=True)


def _install_training_progress_capture(production_world, output):
    models, routes = {}, {}

    def persist():
        durable_atomic_json(output / "world-model-training-progress.json", {
            "schema": "w1-test-world-model-training-progress/1.0.0",
            "routes": [routes[key] for key in sorted(routes)],
        })

    original_model_for = production_world._model_for
    original_train = production_world.train_world_model
    original_selection = production_world._selection_metrics
    original_save = production_world._save_checkpoint
    original_load = production_world._load_checkpoint

    def capture_model_for(config, seed, device, ledger, boundary):
        model = original_model_for(config, seed, device, ledger, boundary)
        models[id(model)] = {"seed": int(seed)}
        return model

    def capture_train(model, batches, **kwargs):
        losses = original_train(model, batches, **kwargs)
        route = models[id(model)]
        route["variant"] = "G2" if kwargs["event_enabled"] else "G1"
        key = f"{route['variant']}:{route['seed']}"
        progress = routes.setdefault(key, {"variant": route["variant"], "seed": route["seed"],
            "loss_by_epoch": [], "selection_metrics_by_epoch": []})
        progress["loss_by_epoch"].append({"epoch": len(progress["loss_by_epoch"]) + 1, **losses})
        persist()
        return losses

    def capture_selection(model, windows, device, expected_parent_count):
        metrics = original_selection(model, windows, device, expected_parent_count)
        route = models[id(model)]
        key = f"{route['variant']}:{route['seed']}"
        progress = routes[key]
        progress["selection_metrics_by_epoch"].append({
            "epoch": len(progress["loss_by_epoch"]),
            "selection_regret": metrics[0],
            "selection_outcome_mae": metrics[1],
        })
        persist()
        return metrics

    def capture_save(payload, path):
        result = original_save(payload, path)
        metadata = payload["metadata"]
        key = f"{metadata['variant']}:{metadata['seed']}"
        progress = routes.setdefault(key, {"variant": metadata["variant"],
            "seed": int(metadata["seed"]), "loss_by_epoch": [],
            "selection_metrics_by_epoch": []})
        progress.setdefault("checkpoint_writes", []).append({
            "path": Path(path).relative_to(output).as_posix(),
            "architecture": metadata["architecture"],
            "event_loss_enabled": metadata["event_loss_enabled"],
            "selection_regret": metadata["selection_regret"],
            "selection_outcome_mae_tiebreak": metadata["selection_outcome_mae_tiebreak"],
        })
        persist()
        return result

    def capture_load(path, device):
        payload = original_load(path, device)
        metadata = payload["metadata"]
        key = f"{metadata['variant']}:{metadata['seed']}"
        progress = routes.setdefault(key, {"variant": metadata["variant"],
            "seed": int(metadata["seed"]), "loss_by_epoch": [],
            "selection_metrics_by_epoch": []})
        progress.setdefault("checkpoint_restores", []).append({
            "path": Path(path).relative_to(output).as_posix(),
            "event_loss_enabled": metadata["event_loss_enabled"],
            "device": device,
        })
        persist()
        return payload

    patchers = [
        patch.object(production_world, "_model_for", side_effect=capture_model_for),
        patch.object(production_world, "train_world_model", side_effect=capture_train),
        patch.object(production_world, "_selection_metrics", side_effect=capture_selection),
        patch.object(production_world, "_save_checkpoint", side_effect=capture_save),
        patch.object(production_world, "_load_checkpoint", side_effect=capture_load),
    ]
    for item in patchers:
        item.start()
    return lambda: [item.stop() for item in reversed(patchers)]


class BottomEnvironment(FakeEnvironment):
    def __init__(self, config, scenario, *, exogenous_key):
        profile = {"deadline": scenario.tasks[0].deadline, "arrival_time": 5.5,
                   "required_service": 1.0, "no_legal_task": scenario.name.endswith("no-opportunity")}
        super().__init__(config, {"profile": profile}, exogenous_key=exogenous_key)

    def _advance_one(self, action, *, continuation):
        observation, _, done, info = super()._advance_one(action, continuation=continuation)
        counts = {"completed": sum(task.completed_at is not None for task in self.clock.tasks.values()),
                  "expired": sum(getattr(task.state, "value", task.state) == "expired" for task in self.clock.tasks.values())}
        info["counts"] = counts
        info["energy"] = {f"uav-{i}": self.config.initial_energy - 0.01*self.clock.time for i in range(4)}
        # This is the fixture environment's scalar reward, not a fabricated model target.
        scalar = float(counts["completed"] - counts["expired"])
        return observation, scalar, done, info


class BottomBoundary:
    def __init__(self, context, *, device):
        self.integration_context = context
        self.real_environment_constructions = 0
        self.synthetic_environment_constructions = 0
        self.device = device

    def environment(self, operation, *args, **kwargs):
        if getattr(operation, "__name__", "") == "M10Environment":
            self.synthetic_environment_constructions += 1
            return BottomEnvironment(*args, **kwargs)
        return operation(*args, **kwargs)

    @staticmethod
    def calculate(operation, *args, **kwargs):
        return operation(*args, **kwargs)
    model = calculate
    optimizer = calculate
    backward = calculate
    checkpoint = calculate


def prepare_test_copy(base, *, no_opportunity=False, device="cpu"):
    package = base / "package"
    # Dynamic attempt/export state is evidence, never part of a test package copy.
    shutil.copytree(
        ROOT,
        package,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "run-once", "runtime-output",
            "execution.lock", "supervisor-status.json", "resource-history.jsonl",
        ),
    )
    matrix = json.loads((package / "experiment-matrix.json").read_text(encoding="utf-8"))
    attempt = "synthetic-joint-production-pipeline"
    roles = {"train": 2, "model_selection": 2, "prediction_confirmation": 8}
    parents, tapes = [], []
    for role, count in roles.items():
        for index in range(count):
            parent = f"synthetic-{role}-{index}"
            deadline = 5.0 if index == 0 else 9.0
            scenario = {"name": parent + ("-no-opportunity" if no_opportunity else ""), "seed": index,
                "split": "synthetic", "tasks": [{"task_id": f"task-{i}", "arrival": 0.0,
                    "x": 2.0, "y": 0.0, "deadline": deadline, "service": 1.0,
                    "priority": 1.0} for i in range(6)]}
            digest = hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest()
            key = f"synthetic|{parent}|repeat-0"
            identity = {"parent": parent, "scenario_sha256": digest, "exogenous_key": key,
                        "proposed_split": role}
            parents.append(identity)
            tapes.append({**identity, "scenario": scenario})
    matrix.update(attempt=attempt, parent_count=12,
        splits={role: [p for p in parents if p["proposed_split"] == role] for role in roles},
        world_model_device=device, world_model_training_windows=2, world_model_selection_windows=2,
        world_model_confirmation_windows=8, world_model_maximum_epochs=1,
        world_model_maximum_train_candidate_rows=50, world_model_maximum_updates_per_variant_seed=2)
    matrix["bounds"]["parents"] = 12
    write(package / "experiment-matrix.json", matrix)
    split = {"schema": "w1-action-conditioned-task-outcome-parent-split/1.0.0", "attempt": attempt,
             "parents": parents}
    write(package / "parent-split.json", split)
    tape = package / "native" / "source-evidence" / "synthetic-tapes.json"
    write(tape, tapes)
    split["source_train_tapes_sha256"] = hashlib.sha256(tape.read_bytes()).hexdigest()
    write(package / "parent-split.json", split)
    inputs = json.loads((package / "runtime-inputs.json").read_text(encoding="utf-8"))
    source_manifest = tape.parent / "execution-manifest.json"
    write(source_manifest, {"schema": "w1-synthetic-source-provenance/1.0.0",
        "origin": "synthetic engineering fixture, not research data",
        "files": {**inputs["source_modules"], tape.name: hashlib.sha256(tape.read_bytes()).hexdigest()}})
    inputs["source_run"].update(wsl_root=str(tape.parent), windows_root=str(tape.parent),
                               train_tape_file=tape.name, train_tape_sha256=hashlib.sha256(tape.read_bytes()).hexdigest(),
                               execution_manifest_sha256=hashlib.sha256(source_manifest.read_bytes()).hexdigest())
    write(package / "runtime-inputs.json", inputs)
    request = json.loads((package / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
    request["attempt"] = attempt
    write(package / "RESOURCE_REQUEST.json", request)
    contract = json.loads((package / "launch-contract.json").read_text(encoding="utf-8"))
    contract.update(attempt=attempt, integration_test=True)
    write(package / "launch-contract.json", contract)
    gate = json.loads((package / "data-gate.json").read_text(encoding="utf-8"))
    gate.update(split_window_counts=roles, train_physical_positive_min_parents=1,
                train_expiry_positive_min_parents=1, train_host_valid_min_parents=1)
    write(package / "data-gate.json", gate)
    identity = freeze(package)
    config = {"schema": "w1-world-model-synthetic-integration-config/1.0.0",
        "training_window_count": 2, "selection_window_count": 2, "confirmation_window_count": 8,
        "maximum_epochs": 1, "maximum_train_candidate_rows": 50}
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    context = {"mode": "synthetic_test", "integration_config": config,
        "integration_config_sha256": hashlib.sha256(payload).hexdigest(),
        "authentication": hmac.new(b"w1-world-model-synthetic-integration-v1", payload, hashlib.sha256).hexdigest()}
    return package, matrix, request, identity, context


class JointProductionPipelineTests(unittest.TestCase):
    def test_formal_packet_rejects_explicit_test_boundary_before_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            attempt = "synthetic-formal-boundary-rejection"
            write(package / "RESOURCE_REQUEST.json", {"status": "NOT_APPROVED", "attempt": attempt})
            write(package / "experiment-matrix.json", {"attempt": attempt, "world_model_device": "cuda:0"})
            write(package / "launch-contract.json", {"attempt": attempt, "integration_test": False})
            from manifest_contract import write_identity_files
            identity = write_identity_files(package, attempt=attempt)
            boundary = BottomBoundary({"mode": "synthetic_test"}, device="cpu")
            with (patch.object(runner, "ROOT", package),
                  patch.object(runner, "OUTPUT", package / "run-once"),
                  patch.dict(os.environ, {"W1_VERIFIED_ATTEMPT": attempt,
                    "W1_VERIFIED_MANIFEST_SHA256": identity["execution_manifest_sha256"],
                    "W1_VERIFIED_HASHES_SHA256": identity["hashes_sha256"]})):
                self.assertEqual(runner.main(boundary=boundary), 1)
            status = json.loads((package / "run-once/status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["exception"], "TEST_BOUNDARY_REQUIRES_INDEPENDENT_INTEGRATION_CONTRACT")
            self.assertEqual(boundary.synthetic_environment_constructions, 0)
            self.assertFalse(status["model_initialized"])

    def execute_case(self, *, no_opportunity=False):
        started_wall = time.monotonic()
        started_cpu = time.process_time()
        device = os.environ.get("W1_SYNTHETIC_TEST_DEVICE", "cpu")
        evidence_value = os.environ.get("W1_TEST_EVIDENCE_DIR")
        evidence = Path(evidence_value) if evidence_value else None
        failure = failure_tb = None
        failure_traceback = None
        runner_traceback = ""
        export_error = None
        artifact_error = None
        package = output = export_destination = None
        original_root = original_output = None
        previous_environment = {}
        runner_patched = False
        export_attempted = False
        exit_code = None
        result = settlement = complete = None
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            try:
                package, matrix, request, identity, context = prepare_test_copy(base,
                    no_opportunity=no_opportunity, device=device)
                verify_worker_contract(package, matrix["attempt"], identity["execution_manifest_sha256"],
                    identity["hashes_sha256"], allow_integration=True)
                output = package / "run-once"
                export_destination = base / "verified-export"
                boundary = BottomBoundary(context, device=device)
                original_root, original_output = runner.ROOT, runner.OUTPUT
                bindings = {"W1_VERIFIED_ATTEMPT": matrix["attempt"],
                    "W1_VERIFIED_MANIFEST_SHA256": identity["execution_manifest_sha256"],
                    "W1_VERIFIED_HASHES_SHA256": identity["hashes_sha256"]}
                previous_environment = {name: os.environ.get(name) for name in bindings}
                runner.ROOT, runner.OUTPUT = package, output
                runner_patched = True
                os.environ.update(bindings)
                import builtins
                original_import = builtins.__import__
                capture_state = {"installed": False, "cleanup": None}

                def import_with_training_capture(name, globals=None, locals=None,
                                                 fromlist=(), level=0):
                    module = original_import(name, globals, locals, fromlist, level)
                    if name == "production_world" and not capture_state["installed"]:
                        capture_state["cleanup"] = _install_training_progress_capture(module, output)
                        capture_state["installed"] = True
                    return module

                runner_errors = io.StringIO()
                builtins.__import__ = import_with_training_capture
                try:
                    with redirect_stderr(runner_errors):
                        exit_code = runner.main(boundary=boundary)
                finally:
                    builtins.__import__ = original_import
                    if capture_state["cleanup"] is not None:
                        capture_state["cleanup"]()
                runner_traceback = runner_errors.getvalue()
                if package.is_dir():
                    export_attempted = True
                    complete = controlled_export(package, export_destination)
                if (output / "label-coverage-summary.json").is_file():
                    result = json.loads((output / "label-coverage-summary.json").read_text(encoding="utf-8"))
                if (output / "resource-settlement.json").is_file():
                    settlement = json.loads((output / "resource-settlement.json").read_text(encoding="utf-8"))
                self.assertEqual(exit_code, 0)
                self.assertIsNotNone(result)
                self.assertIsNotNone(settlement)
                self.assertIsNone(settlement["ledger_settlement_error"])
                self.assertEqual(settlement["ledger"]["pending_calls"], 0)
                self.assertEqual(boundary.real_environment_constructions, 0)
                self.assertEqual(boundary.synthetic_environment_constructions, 12)
                self.assertTrue(complete["verified"])
                exported = json.loads((export_destination / "run-once" / "resource-settlement.json").read_text(encoding="utf-8"))
                self.assertEqual(exported, settlement)
            except BaseException as exc:
                failure = exc
                failure_tb = exc.__traceback__
                failure_traceback = _traceback_text(exc)
            finally:
                if package is not None and package.is_dir() and not export_attempted:
                    export_attempted = True
                    try:
                        complete = controlled_export(package, export_destination)
                    except BaseException as exc:
                        export_error = _traceback_text(exc)
                        if failure is None:
                            failure, failure_tb = exc, exc.__traceback__
                            failure_traceback = export_error
                if runner_patched:
                    runner.ROOT, runner.OUTPUT = original_root, original_output
                for name, value in previous_environment.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
                if evidence is not None:
                    try:
                        evidence.mkdir(parents=True, exist_ok=True)
                        if export_destination is not None:
                            _copy_tree(export_destination, evidence / "joint-controlled-export")
                        persisted_export = evidence / "joint-controlled-export" / "EXPORT_COMPLETE.json"
                        if output is not None and not persisted_export.is_file():
                            _copy_tree(output, evidence / "joint-run-once")
                    except BaseException as exc:
                        artifact_error = _traceback_text(exc)
                        if failure is None:
                            failure, failure_tb = exc, exc.__traceback__
                            failure_traceback = artifact_error
                    try:
                        counters = _ledger_counters(output / "budget.sqlite3") if output is not None else None
                    except BaseException as exc:
                        counters = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
                    try:
                        durable_atomic_json(evidence / "joint-pipeline-test-diagnostics.json", {
                            "schema": "w1-joint-pipeline-test-diagnostics/1.0.0",
                            "status": "technical_stop" if failure is not None else "complete",
                            "profile": {"device": device, "training_windows": 2,
                                "selection_windows": 2, "confirmation_windows": 8,
                                "seeds": 3, "maximum_epochs": 1,
                                "maximum_train_candidate_rows": 50},
                            "exit_code": exit_code,
                            "first_original_traceback": runner_traceback or failure_traceback,
                            "test_failure_traceback": failure_traceback,
                            "runner_traceback": runner_traceback,
                            "export_error": export_error,
                            "artifact_persistence_error": artifact_error,
                            "partial_compute_counters": counters,
                            "artifacts": {
                                "run_once": bool((evidence / "joint-run-once").is_dir() or
                                    (evidence / "joint-controlled-export" / "run-once").is_dir()),
                                "training_summary": bool(output is not None and
                                    (output / "world-model-training-summary.json").is_file()),
                                "training_progress": bool(output is not None and
                                    (output / "world-model-training-progress.json").is_file()),
                                "prediction_trace": bool(output is not None and
                                    (output / "prediction-trace.jsonl").is_file()),
                                "prediction_metrics": bool(output is not None and
                                    (output / "prediction-metrics.json").is_file()),
                                "ledger": bool(output is not None and (output / "budget.sqlite3").is_file()),
                                "controlled_export": bool(export_destination is not None and export_destination.is_dir()),
                            },
                        })
                    except BaseException as exc:
                        if failure is None:
                            failure, failure_tb = exc, exc.__traceback__
                            failure_traceback = _traceback_text(exc)
            if failure is not None:
                raise failure.with_traceback(failure_tb)
        report = {"scope": "synthetic bottom-environment production pipeline; full model architecture, 1 epoch, 2/2/8 parents",
            "device": device, "no_opportunity": no_opportunity, "status": result["status"],
            "ledger": settlement["ledger"], "real_environment_calls": 0,
            "synthetic_environment_constructions": 12,
            "model_calls_are_synthetic_and_nonzero": not no_opportunity,
            "wall_seconds": time.monotonic()-started_wall, "cpu_seconds": time.process_time()-started_cpu,
            "checkpoint_restore_executed": not no_opportunity, "controlled_export_verified": True,
            "production_runner_main_executed": True,
            "formal_matrix_tested": False, "gppo_calls": 0}
        if evidence is not None:
            durable_atomic_json(evidence/ ("joint-no-opportunity.json" if no_opportunity else "joint-success.json"), report)
        return report

    def test_data_gate_stops_learning_without_opportunity(self):
        report = self.execute_case(no_opportunity=True)
        self.assertEqual(report["status"], "data_gate_stop")
        totals = report["ledger"]["totals"]
        for key in ("model_initializations_or_loads", "world_batch_forwards", "world_backward_calls", "world_optimizer_updates", "checkpoint_loads"):
            self.assertEqual(totals.get(key, 0), 0)

    def test_actual_collection_training_restore_metrics_ledger_export(self):
        report = self.execute_case()
        self.assertEqual(report["status"], "prediction_evaluation_complete")
        totals = report["ledger"]["totals"]
        self.assertEqual(totals["model_initializations_or_loads"], 12)
        self.assertEqual(totals["world_optimizer_updates"], 12)
        self.assertEqual(totals["world_backward_calls"], 12)
        self.assertEqual(totals["checkpoint_writes"], 6)
        self.assertEqual(totals["checkpoint_loads"], 6)
        self.assertEqual(totals["world_batch_forwards"], 84)
        self.assertEqual(totals["world_sample_evaluations"], 420)


if __name__ == "__main__":
    unittest.main()
