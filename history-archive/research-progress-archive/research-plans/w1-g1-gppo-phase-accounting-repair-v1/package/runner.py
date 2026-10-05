"""One-shot G1-prior policy training and paired task-validation worker."""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any

from budget_ledger import BudgetLedger
from infra_io import durable_atomic_json


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "run-once"


def initialize_output(output: Path, attempt: str) -> None:
    output = Path(output)
    output.mkdir(parents=False, exist_ok=False)
    durable_atomic_json(output / "status.json", {
        "schema": "w1-g1-gppo-task-validation-status/1.0.0",
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
    if ledger is None:
        return {
            "model_initialized": False, "checkpoint_loaded": False,
            "training_started": False, "policy_training_started": False,
            "world_training_started": False, "learning_call_evidence": {},
        }
    amounts_by_status = {status: {} for status in ("pending", "failed", "complete")}
    for payload, status in ledger.connection.execute("SELECT amounts,status FROM calls"):
        for resource, amount in json.loads(payload).items():
            target = amounts_by_status[status]
            target[resource] = target.get(resource, 0) + amount
    def activity(resource: str) -> bool | None:
        completed = amounts_by_status["complete"].get(resource, 0)
        unresolved = sum(amounts_by_status[status].get(resource, 0) for status in ("pending", "failed"))
        return True if completed else (None if unresolved else False)

    policy_started = activity("policy_backward_calls")
    world_started = activity("world_backward_calls")
    training_started = True if policy_started is True or world_started is True else (
        None if policy_started is None or world_started is None else False
    )
    return {
        "model_initialized": activity("model_initializations_or_loads"),
        "checkpoint_loaded": activity("checkpoint_loads"),
        "training_started": training_started,
        "policy_training_started": policy_started,
        "world_training_started": world_started,
        "learning_call_evidence": amounts_by_status,
    }


def _persist_first_error(output: Path, *, attempt: str, error: BaseException,
                         stage: str | None, trace: str | None = None) -> dict[str, Any]:
    path = Path(output) / "first-error.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    value = {
        "schema": "w1-worker-first-error/1.0.0",
        "attempt": attempt or "unavailable",
        "status": "technical_stop",
        "stage": stage,
        "exception_type": type(error).__name__,
        "exception": str(error),
        "traceback": trace or "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
        "recorded_before_settlement": True,
        "automatic_retry": False,
    }
    durable_atomic_json(path, value)
    return value


def _settlement(ledger: BudgetLedger | None, *, status: str,
                attempt: str, output: Path, error: BaseException | None = None,
                error_traceback: str | None = None) -> dict[str, Any]:
    snapshot = None
    phase_snapshot = None
    phase_snapshot_recorded = False
    phase_snapshot_attempted = False
    worker_cpu_snapshot = None
    settlement_error = None
    if ledger is not None:
        try:
            if error is not None:
                _persist_first_error(output, attempt=attempt, error=error,
                                     stage=getattr(ledger, "stage", "") or None, trace=error_traceback)
                phase_snapshot_attempted = hasattr(ledger, "snapshot_phase")
                phase_snapshot = ledger.snapshot_phase(reason="failure_settlement")
                phase_snapshot_recorded = phase_snapshot is not None
                if not getattr(ledger, "stage", ""):
                    try:
                        measured_cpu, scope = ledger._process_cpu()
                        worker_cpu_snapshot = {
                            "cpu_seconds": measured_cpu,
                            "scope": scope,
                            "attribution_status": "worker_process_lifetime_unstaged",
                            "included_in_supervisor_total": True,
                            "phase_origin_status": "no_worker_stage_confirmed",
                        }
                    except BaseException as cpu_exc:
                        worker_cpu_snapshot = {
                            "cpu_seconds": None,
                            "scope": None,
                            "attribution_status": "measurement_failed",
                            "measurement_error": f"{type(cpu_exc).__name__}: {cpu_exc}",
                            "included_in_supervisor_total": None,
                            "phase_origin_status": "no_worker_stage_confirmed",
                        }
            elif "settlement_and_verified_export" in getattr(ledger, "request", {}).get("stages", {}):
                ledger.select("settlement_and_verified_export")
            snapshot = ledger.assert_settled()
        except BaseException as exc:
            settlement_error = f"{type(exc).__name__}: {exc}"
            if error is None:
                _persist_first_error(output, attempt=attempt, error=exc,
                                     stage=getattr(ledger, "stage", "") or None,
                                     trace=traceback.format_exc())
            if not phase_snapshot_attempted and hasattr(ledger, "snapshot_phase"):
                try:
                    phase_snapshot_attempted = True
                    phase_snapshot = ledger.snapshot_phase(reason="failure_settlement")
                    phase_snapshot_recorded = phase_snapshot is not None
                except BaseException as snapshot_exc:
                    settlement_error += "; phase snapshot failed: " + type(snapshot_exc).__name__ + ": " + str(snapshot_exc)
            try:
                snapshot = ledger.snapshot()
            except BaseException as snapshot_exc:
                settlement_error += "; ledger snapshot failed: " + type(snapshot_exc).__name__ + ": " + str(snapshot_exc)
    effective_status = "technical_stop" if settlement_error is not None else status
    value = {
        "schema": "w1-g1-gppo-task-validation-settlement/1.0.0",
        "status": effective_status,
        "attempt": attempt,
        "ledger": snapshot,
        "ledger_settlement_error": settlement_error,
        "phase_snapshot": phase_snapshot,
        "phase_snapshot_recorded": phase_snapshot_recorded,
        "worker_cpu_snapshot": worker_cpu_snapshot,
        "phase_rows_are_non_additive_views": True,
        "failure_settlement_attempted": error is not None or settlement_error is not None,
        "error": None if error is None else f"{type(error).__name__}: {error}",
        "model_calls": None if snapshot is None else {
            key: snapshot["totals"].get(key, 0) for key in (
                "model_initializations_or_loads", "checkpoint_writes", "checkpoint_loads",
                "policy_backward_calls", "policy_optimizer_updates", "world_batch_forwards",
                "world_sample_evaluations", "world_backward_calls", "world_optimizer_updates")},
        "automatic_retry": False,
    }
    durable_atomic_json(Path(output) / "resource-settlement.json", value)
    return value


def main(*, boundary=None) -> int:
    attempt = os.environ.get("W1_VERIFIED_ATTEMPT", "")
    ledger = None
    output_preexisted = OUTPUT.exists()
    output_initialized = False
    settlement_attempted = False
    phase_snapshot_recorded = False
    failure_settlement_attempted = False
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
        worker = verify_worker_contract(
            ROOT, attempt, os.environ["W1_VERIFIED_MANIFEST_SHA256"],
            os.environ["W1_VERIFIED_HASHES_SHA256"], allow_integration=boundary is not None,
        )
        if boundary is not None and (not contract.get("integration_test")
                or getattr(boundary, "integration_context", {}).get("mode") != "synthetic_test"):
            raise RuntimeError("TEST_BOUNDARY_REQUIRES_INDEPENDENT_INTEGRATION_CONTRACT")
        if matrix.get("world_model_device") != "cpu":
            raise RuntimeError("TASK_WORKER_REQUIRES_FROZEN_CPU")
        if (contract.get("runtime_mode") != "local_wsl_g1_gppo_cpu"
                or contract.get("cpu_affinity") != matrix.get("cpu_execution", {}).get("affinity")):
            raise RuntimeError("TASK_RUNTIME_IDENTITY_MISMATCH")
        from task_contract import verify_task_inputs
        task_identity = verify_task_inputs(ROOT)
        durable_atomic_json(OUTPUT / "task-input-check.json", {
            "worker_manifest_sha256": worker["identity"]["manifest_sha256"],
            "hashes_sha256": worker["identity"]["hashes_sha256"],
            "task_inputs": task_identity,
        })
        ledger = initialize_ledger(OUTPUT, request)
        ledger.select("staging_and_zero_step_gate")
        _activity("conditional_policy_training", "task_pipeline")
        from task_pipeline import run_pipeline
        summary = run_pipeline(ROOT, OUTPUT, matrix, request, ledger, boundary=boundary)
        construction = environment_construction_evidence(OUTPUT)
        durable_atomic_json(OUTPUT / "task-run-summary.json", {
            key: value for key, value in summary.items() if key != "independent_metrics"
        })
        _activity("settlement_and_verified_export", "task_settlement")
        result_status = summary["status"]
        settlement_attempted = True
        settlement = _settlement(ledger, status=result_status, attempt=attempt, output=OUTPUT)
        phase_snapshot_recorded = settlement["phase_snapshot_recorded"]
        failure_settlement_attempted = settlement["failure_settlement_attempted"]
        if settlement["ledger_settlement_error"] is not None:
            raise RuntimeError("LEDGER_SETTLEMENT_FAILED: " + settlement["ledger_settlement_error"])
        durable_atomic_json(OUTPUT / "status.json", {
            "schema": "w1-g1-gppo-task-validation-status/1.0.0",
            "attempt": attempt,
            "status": result_status,
            "stage": "complete",
            "task_gate": summary.get("task_gate"),
            "cost_gate": summary.get("cost_gate"),
            "research_success": summary.get("research_success"),
            **construction,
            "task_comparison_executed": summary.get("task_comparison_executed", False),
            **learning_activity_evidence(ledger),
            "automatic_retry": False,
            "settlement_status": settlement["status"],
        })
        return 0
    except BaseException as exc:
        error_traceback = traceback.format_exc()
        if output_initialized or (not output_preexisted and OUTPUT.is_dir()):
            try:
                stage = ledger.stage or None if ledger is not None else None
                _persist_first_error(OUTPUT, attempt=attempt, error=exc,
                                     stage=stage, trace=error_traceback)
            except BaseException:
                traceback.print_exc()
            traceback.print_exc()
            try:
                construction = environment_construction_evidence(OUTPUT)
                if not settlement_attempted:
                    settlement_attempted = True
                    try:
                        _activity("settlement_and_verified_export", "technical_stop_settlement")
                    except BaseException:
                        pass
                    settlement = _settlement(
                        ledger, status="technical_stop", attempt=attempt or "unavailable",
                        output=OUTPUT, error=exc, error_traceback=error_traceback,
                    )
                    phase_snapshot_recorded = settlement["phase_snapshot_recorded"]
                    failure_settlement_attempted = settlement["failure_settlement_attempted"]
                first_error = json.loads((OUTPUT / "first-error.json").read_text(encoding="utf-8"))
                durable_atomic_json(OUTPUT / "status.json", {
                    "schema": "w1-g1-gppo-task-validation-status/1.0.0",
                    "attempt": attempt,
                    "status": "technical_stop",
                    "stage": first_error.get("stage"),
                    "exception_type": first_error["exception_type"],
                    "exception": first_error["exception"],
                    "first_error_file": "first-error.json",
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
            ledger.close(snapshot=not (phase_snapshot_recorded or failure_settlement_attempted))


if __name__ == "__main__":
    raise SystemExit(main())
