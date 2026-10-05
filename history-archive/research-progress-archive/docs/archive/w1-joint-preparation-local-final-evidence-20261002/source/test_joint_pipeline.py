"""Production pipeline integration; only the environment constructor is replaced."""
from __future__ import annotations
import copy
import hashlib
import hmac
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "native"))
from budget_ledger import BudgetLedger
from freeze_joint_package import freeze
from infra_io import controlled_export, durable_atomic_json
from joint_pipeline import run_pipeline
from worker_contract import verify_worker_contract
from runner import initialize_output, _settlement
from test_production_collector_lifecycle import FakeEnvironment


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


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
    inputs["source_run"].update(wsl_root=str(tape.parent), windows_root=str(tape.parent),
                               train_tape_file=tape.name, train_tape_sha256=hashlib.sha256(tape.read_bytes()).hexdigest())
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
    def execute_case(self, *, no_opportunity=False):
        started_wall = time.monotonic()
        started_cpu = time.process_time()
        device = os.environ.get("W1_SYNTHETIC_TEST_DEVICE", "cpu")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            package, matrix, request, identity, context = prepare_test_copy(base,
                no_opportunity=no_opportunity, device=device)
            verify_worker_contract(package, matrix["attempt"], identity["execution_manifest_sha256"],
                identity["hashes_sha256"], allow_integration=True)
            output = package / "run-once"
            initialize_output(output, matrix["attempt"])
            ledger = BudgetLedger(output / "budget.sqlite3", request)
            boundary = BottomBoundary(context, device=device)
            try:
                result = run_pipeline(package, output, matrix, ledger, boundary)
                settlement = _settlement(ledger, status=result["status"], attempt=matrix["attempt"], output=output)
                self.assertIsNone(settlement["ledger_settlement_error"])
                self.assertEqual(ledger.snapshot()["pending_calls"], 0)
                self.assertEqual(boundary.real_environment_constructions, 0)
                self.assertEqual(boundary.synthetic_environment_constructions, 12)
                durable_atomic_json(output / "status.json", result)
                ledger.close()
                complete = controlled_export(package, base / "verified-export")
                self.assertTrue(complete["verified"])
                exported = json.loads((base / "verified-export" / "run-once" / "resource-settlement.json").read_text(encoding="utf-8"))
                self.assertEqual(exported, settlement)
            except BaseException:
                ledger.close()
                raise
        report = {"scope": "synthetic bottom-environment production pipeline; full model architecture, 1 epoch, 2/2/8 parents",
            "device": device, "no_opportunity": no_opportunity, "status": result["status"],
            "ledger": settlement["ledger"], "real_environment_calls": 0,
            "synthetic_environment_constructions": 12,
            "model_calls_are_synthetic_and_nonzero": not no_opportunity,
            "wall_seconds": time.monotonic()-started_wall, "cpu_seconds": time.process_time()-started_cpu,
            "checkpoint_restore_executed": not no_opportunity, "controlled_export_verified": True,
            "formal_matrix_tested": False, "gppo_calls": 0}
        evidence = os.environ.get("W1_TEST_EVIDENCE_DIR")
        if evidence:
            write(Path(evidence)/("joint-no-opportunity.json" if no_opportunity else "joint-success.json"), report)
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


if __name__ == "__main__":
    unittest.main()
