"""One-shot collection, data admission, and G1/G2 learning worker; no GPPO."""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any

from budget_ledger import BudgetLedger
from infra_io import durable_atomic_json
from production_data import ProductionDataCollector
from verify_runtime_inputs import verify_runtime_inputs


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "run-once"


class DirectBoundary:
    def environment(self, operation, *args, **kwargs):
        return operation(*args, **kwargs)

    model = environment
    optimizer = environment
    backward = environment
    checkpoint = environment


def initialize_output(output: Path, attempt: str) -> None:
    output = Path(output)
    output.mkdir(parents=False, exist_ok=False)
    durable_atomic_json(output / "status.json", {
        "schema": "w1-task-outcome-label-qualification-status/1.0.0",
        "attempt": attempt,
        "status": "initializing",
        "stage": "staging_and_zero_step_gate",
        "automatic_retry": False,
        "environment_constructed": False,
        "environment_construction_status": "not_started",
        "model_initialized": False,
        "checkpoint_loaded": False,
        "training_started": False,
    })


def initialize_ledger(output: Path, request: dict[str, Any]) -> BudgetLedger:
    output = Path(output)
    if not output.is_dir() or not (output / "status.json").is_file():
        raise RuntimeError("OUTPUT_AND_INITIAL_STATUS_MUST_PRECEDE_SQLITE")
    return BudgetLedger(output / "budget.sqlite3", request)


def environment_construction_evidence(output: Path) -> dict[str, Any]:
    path = Path(output) / "environment-construction.jsonl"
    if not path.is_file():
        return {
            "environment_constructed": False,
            "environment_construction_status": "not_started",
            "environment_construction_attempts": 0,
            "environment_construction_successes": 0,
        }
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        pending: set[int] = set()
        attempted = 0
        constructed = 0
        for row in rows:
            sequence = row.get("sequence")
            if type(sequence) is not int or sequence < 1:
                raise ValueError("construction sequence invalid")
            if row.get("event") == "attempted":
                if sequence in pending or sequence != attempted + 1:
                    raise ValueError("construction attempt order invalid")
                pending.add(sequence)
                attempted += 1
            elif row.get("event") == "constructed":
                if sequence not in pending or sequence != constructed + 1:
                    raise ValueError("construction success without attempt")
                pending.remove(sequence)
                constructed += 1
            else:
                raise ValueError("construction event invalid")
        if pending:
            status = "indeterminate_after_construction_attempt"
            did_construct = None
        elif attempted == 0:
            status = "not_started"
            did_construct = False
        elif constructed == attempted:
            status = "constructed"
            did_construct = True
        else:
            status = "construction_failed"
            did_construct = constructed > 0
        return {
            "environment_constructed": did_construct,
            "environment_construction_status": status,
            "environment_construction_attempts": attempted,
            "environment_construction_successes": constructed,
        }
    except BaseException as exc:
        return {
            "environment_constructed": None,
            "environment_construction_status": "evidence_unavailable",
            "environment_construction_error": f"{type(exc).__name__}: {exc}",
        }


def _activity(stage: str, run: str) -> None:
    durable_atomic_json(OUTPUT / "activity.json", {"stage": stage, "run": run})


def learning_activity_evidence(ledger: BudgetLedger | None) -> dict[str, Any]:
    resources = {"model_initialized": "model_initializations_or_loads",
                 "checkpoint_loaded": "checkpoint_loads",
                 "training_started": "world_backward_calls"}
    if ledger is None:
        return {**{flag: False for flag in resources}, "learning_call_evidence": {}}
    amounts_by_status = {status: {} for status in ("pending", "failed", "complete")}
    for payload, status in ledger.connection.execute("SELECT amounts,status FROM calls"):
        for resource, amount in json.loads(payload).items():
            target = amounts_by_status[status]
            target[resource] = target.get(resource, 0) + amount
    flags = {}
    for flag, resource in resources.items():
        completed = amounts_by_status["complete"].get(resource, 0)
        unresolved = sum(amounts_by_status[status].get(resource, 0) for status in ("pending", "failed"))
        flags[flag] = True if completed else (None if unresolved else False)
    return {**flags, "learning_call_evidence": amounts_by_status}


def _settlement(ledger: BudgetLedger | None, *, status: str,
                attempt: str, output: Path, error: BaseException | None = None) -> dict[str, Any]:
    snapshot = None
    settlement_error = None
    if ledger is not None:
        try:
            snapshot = ledger.assert_settled()
        except BaseException as exc:
            settlement_error = f"{type(exc).__name__}: {exc}"
            snapshot = ledger.snapshot()
    effective_status = "technical_stop" if settlement_error is not None else status
    value = {
        "schema": "w1-task-outcome-label-qualification-settlement/1.0.0",
        "status": effective_status,
        "attempt": attempt,
        "ledger": snapshot,
        "ledger_settlement_error": settlement_error,
        "error": None if error is None else f"{type(error).__name__}: {error}",
        "model_calls": None if snapshot is None else {
            key: snapshot["totals"].get(key, 0) for key in (
                "model_initializations_or_loads", "checkpoint_writes", "checkpoint_loads",
                "world_batch_forwards", "world_sample_evaluations", "world_backward_calls", "world_optimizer_updates")},
        "automatic_retry": False,
    }
    durable_atomic_json(Path(output) / "resource-settlement.json", value)
    return value


def main() -> int:
    attempt = os.environ.get("W1_VERIFIED_ATTEMPT", "")
    ledger = None
    failure = None
    output_preexisted = OUTPUT.exists()
    output_initialized = False
    try:
        if not attempt:
            raise RuntimeError("VERIFIED_ATTEMPT_IDENTITY_MISSING")
        initialize_output(OUTPUT, attempt)
        output_initialized = True
        request = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
        matrix = json.loads((ROOT / "experiment-matrix.json").read_text(encoding="utf-8"))
        contract = json.loads((ROOT / "launch-contract.json").read_text(encoding="utf-8"))
        if request.get("status") != "NOT_APPROVED":
            raise RuntimeError("FROZEN_RESOURCE_REQUEST_STATUS_CHANGED")
        if not (attempt == matrix.get("attempt") == contract.get("attempt")):
            raise RuntimeError("ATTEMPT_IDENTITY_MISMATCH_IN_WORKER")
        if request.get("attempt") != attempt:
            raise RuntimeError("RESOURCE_REQUEST_ATTEMPT_IDENTITY_MISMATCH_IN_WORKER")
        _activity("staging_and_zero_step_gate", "runtime_input_validation")
        from worker_contract import verify_worker_contract
        from joint_inputs import verify_joint_inputs
        verify_worker_contract(ROOT, attempt, os.environ["W1_VERIFIED_MANIFEST_SHA256"],
                               os.environ["W1_VERIFIED_HASHES_SHA256"])
        if matrix.get("world_model_device") != "cuda:0":
            raise RuntimeError("FORMAL_WORKER_REQUIRES_FROZEN_REMOTE_GPU")
        runtime_identity = verify_joint_inputs(ROOT)
        durable_atomic_json(OUTPUT / "runtime-input-check.json", runtime_identity)
        ledger = initialize_ledger(OUTPUT, request)
        ledger.select("label_qualification")
        _activity("label_qualification", "production_collector")
        from joint_pipeline import run_pipeline
        summary = run_pipeline(ROOT, OUTPUT, matrix, ledger, DirectBoundary())
        construction = environment_construction_evidence(OUTPUT)
        durable_atomic_json(OUTPUT / "label-coverage-summary.json", summary)
        _activity("settlement_and_verified_export", "label_settlement")
        result_status = summary["status"]
        settlement = _settlement(ledger, status=result_status, attempt=attempt, output=OUTPUT)
        if settlement["ledger_settlement_error"] is not None:
            raise RuntimeError("LEDGER_SETTLEMENT_FAILED: " + settlement["ledger_settlement_error"])
        durable_atomic_json(OUTPUT / "status.json", {
            "schema": "w1-task-outcome-label-qualification-status/1.0.0",
            "attempt": attempt,
            "status": result_status,
            "stage": "complete",
            "data_admission": summary.get("data"),
            "prediction": summary.get("prediction", "not_evaluated"),
            **construction,
            "task_comparison_executed": False,
            **learning_activity_evidence(ledger),
            "automatic_retry": False,
            "settlement_status": settlement["status"],
        })
        return 0
    except BaseException as exc:
        failure = exc
        traceback.print_exc()
        if output_initialized or (not output_preexisted and OUTPUT.is_dir()):
            try:
                construction = environment_construction_evidence(OUTPUT)
                _activity("settlement_and_verified_export", "technical_stop_settlement")
                _settlement(ledger, status="technical_stop", attempt=attempt or "unavailable", output=OUTPUT, error=exc)
                durable_atomic_json(OUTPUT / "status.json", {
                    "schema": "w1-task-outcome-label-qualification-status/1.0.0",
                    "attempt": attempt,
                    "status": "technical_stop",
                    "stage": "worker",
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    **construction,
                    **learning_activity_evidence(ledger),
                    "automatic_retry": False,
                })
            except BaseException:
                traceback.print_exc()
        elif not output_preexisted:
            try:
                durable_atomic_json(ROOT / "worker-initialization-failure.json", {
                    "schema": "w1-worker-initialization-failure/1.0.0",
                    "attempt": attempt or "unavailable",
                    "status": "technical_stop",
                    "stage": "runner_output_initialization",
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "environment_constructed": False,
                    "environment_construction_status": "not_started",
                    "model_initialized": False,
                    "checkpoint_loaded": False,
                    "training_started": False,
                    "automatic_retry": False,
                })
            except BaseException:
                traceback.print_exc()
        return 1
    finally:
        if ledger is not None:
            ledger.close()


if __name__ == "__main__":
    raise SystemExit(main())
