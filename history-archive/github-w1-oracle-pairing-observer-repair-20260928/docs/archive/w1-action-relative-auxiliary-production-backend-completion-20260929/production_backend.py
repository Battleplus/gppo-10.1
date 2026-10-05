"""Production A/B backend and its explicit bottom-boundary test seam."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from budget_ledger import BudgetLedger
from infra_io import durable_atomic_json
from runtime_adapter import DirectBoundary, ProductionRuntimeAdapter, RuntimeAdapterError


class TechnicalStop(RuntimeError):
    pass


class ABRuntimeBackend:
    """Real backend used by the formal runner.

    Storage is created before SQLite opens. The adapter is the only boundary
    that may touch environments, models, optimizers and checkpoints.
    """

    def __init__(self, root: Path, output: Path, request: dict, matrix: dict,
                 *, adapter_factory: Callable[..., Any] | None = None,
                 boundary: Any | None = None):
        self.root, self.output = root, output
        self.request, self.matrix = request, matrix
        if output.exists():
            raise TechnicalStop("OUTPUT_ALREADY_EXISTS_NO_RETRY")
        output.mkdir(parents=True, exist_ok=False)
        self.ledger = BudgetLedger(output / "budget.sqlite3", request)
        self.task_calls = 0
        factory = adapter_factory or ProductionRuntimeAdapter
        try:
            self.adapter = factory(root, output, request, matrix, self.ledger.call,
                                   boundary or DirectBoundary())
        except Exception:
            self.ledger.close()
            raise
        if hasattr(self.adapter, "bind_ledger"):
            self.adapter.bind_ledger(self.ledger)

    def _stage(self, name: str) -> None:
        if name not in self.request.get("stage_limits", {}):
            raise TechnicalStop(f"UNKNOWN_STAGE:{name}")
        self.ledger.select(name)
        self._write_json(self.output / "activity.json", {"stage": name, "backend": "production"})

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        try:
            durable_atomic_json(path, value)
        except PermissionError:
            # Windows unit tests cannot fsync a directory. The formal WSL
            # path uses durable_atomic_json and therefore retains the fsync
            # contract.
            path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    def zero_step_gate(self) -> dict[str, Any]:
        self._stage("staging_and_zero_step_gate")
        if self.request.get("status") != "NOT_APPROVED":
            raise TechnicalStop("RESOURCE_REQUEST_STATUS_CHANGED")
        expected = self.request.get("attempt")
        contract = json.loads((self.root / "launch-contract.json").read_text(encoding="utf-8"))
        manifest = json.loads((self.root / "execution-manifest.json").read_text(encoding="utf-8"))
        if not (expected and expected == contract.get("attempt") == manifest.get("attempt")):
            raise TechnicalStop("ATTEMPT_IDENTITY_MISMATCH")
        try:
            result = self.adapter.zero_step_gate()
        except RuntimeAdapterError as exc:
            raise TechnicalStop(str(exc)) from exc
        if not result.get("pass") or result.get("environment_constructed") or result.get("model_constructed"):
            raise TechnicalStop("ZERO_STEP_GATE_CONTRACT_FAILED")
        return result

    def audit_reused_labels(self) -> dict[str, Any]:
        self._stage("data_reuse_audit")
        value = json.loads((self.root / "data-reuse-audit.json").read_text(encoding="utf-8"))
        exact = value.get("reused_training_records") == 219 and value.get("reused_model_selection_records") == 68
        if not exact or value.get("old_prediction_evaluation_used_for_training") is not False:
            return {"pass": False, **value, "reason": "reused label contract mismatch"}
        return {"pass": True, **value}

    def collect_confirmation_labels(self) -> dict[str, Any]:
        self._stage("new_prediction_confirmation")
        try:
            return self.adapter.collect_confirmation_labels()
        except RuntimeAdapterError as exc:
            raise TechnicalStop(str(exc)) from exc

    def train_variants(self) -> dict[str, Any]:
        self._stage("supervised_training_and_selection")
        try:
            return self.adapter.train_variants()
        except RuntimeAdapterError as exc:
            raise TechnicalStop(str(exc)) from exc

    def evaluate_predictions(self) -> dict[str, Any]:
        self._stage("independent_prediction_evaluation")
        try:
            return self.adapter.evaluate_predictions()
        except RuntimeAdapterError as exc:
            raise TechnicalStop(str(exc)) from exc

    def run_task_comparison(self) -> dict[str, Any]:
        self._stage("conditional_task_comparison")
        self.task_calls += 1
        try:
            return self.adapter.run_task_comparison()
        except RuntimeAdapterError as exc:
            raise TechnicalStop(str(exc)) from exc

    def settle(self, status: str, details: dict[str, Any]) -> None:
        self._stage("settlement")
        self._write_json(self.output / "settlement.json", {
            "status": status, "task_calls": self.task_calls,
            "ledger": self.ledger.assert_settled(), "automatic_retry": False,
            "automatic_extension": False, "production_backend": True,
        })

    def close(self) -> None:
        self.ledger.close()


class BoundaryProbe:
    """Bottom-boundary replacement used only by integration tests."""

    def __init__(self, *, fail: str | None = None):
        self.fail = fail
        self.calls: list[str] = []

    def _call(self, kind: str, operation, *args, **kwargs):
        self.calls.append(kind)
        if self.fail == kind:
            raise RuntimeAdapterError(f"INJECTED_{kind.upper()}_FAILURE")
        return operation(*args, **kwargs)

    def environment(self, operation, *args, **kwargs):
        return self._call("environment", operation, *args, **kwargs)

    def model(self, operation, *args, **kwargs):
        return self._call("model", operation, *args, **kwargs)

    def optimizer(self, operation, *args, **kwargs):
        return self._call("optimizer", operation, *args, **kwargs)

    def checkpoint(self, operation, *args, **kwargs):
        return self._call("checkpoint", operation, *args, **kwargs)
