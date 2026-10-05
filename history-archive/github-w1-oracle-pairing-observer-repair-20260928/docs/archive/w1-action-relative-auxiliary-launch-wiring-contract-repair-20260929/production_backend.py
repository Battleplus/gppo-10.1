"""Production backend boundary and explicit zero-call integration backend."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from budget_ledger import BudgetLedger
from infra_io import durable_atomic_json


class TechnicalStop(RuntimeError):
    pass


class ABRuntimeBackend:
    """Real backend boundary.

    Runtime modules are imported only after the identity gate.  The adapter is
    deliberately explicit: a package without the frozen runtime adapter stops
    before environment construction instead of silently substituting a fake.
    """

    def __init__(self, root: Path, output: Path, request: dict, matrix: dict):
        self.root, self.output = root, output
        self.request, self.matrix = request, matrix
        self.ledger = BudgetLedger(output / "budget.sqlite3", request)
        self.task_calls = 0

    def _stage(self, name: str) -> None:
        self.ledger.select(name)
        (self.output / "activity.json").write_text(json.dumps({"stage": name}) + "\n", encoding="utf-8")

    def zero_step_gate(self) -> dict[str, Any]:
        self._stage("staging_and_zero_step_gate")
        if self.request.get("status") != "NOT_APPROVED":
            raise TechnicalStop("frozen request status changed")
        # The dynamic adapter is a sealed input of a future approved package.
        # Do not import it or construct an environment in this preparation run.
        adapter = self.root / "runtime_adapter.py"
        if not adapter.is_file():
            raise TechnicalStop("REAL_RUNTIME_ADAPTER_NOT_ATTACHED")
        return {"pass": True, "environment_constructed": False, "model_constructed": False}

    def audit_reused_labels(self) -> dict[str, Any]:
        self._stage("data_reuse_audit")
        path = self.root / "data-reuse-audit.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        return {"pass": bool(value.get("reused_training_records", 0) == 219 and value.get("reused_model_selection_records", 0) == 68), **value}

    def collect_confirmation_labels(self) -> dict[str, Any]:
        self._stage("new_prediction_confirmation")
        raise TechnicalStop("REAL_CONFIRMATION_ADAPTER_NOT_ATTACHED")

    def train_variants(self) -> dict[str, Any]:
        self._stage("supervised_training_and_selection")
        raise TechnicalStop("REAL_TRAINING_ADAPTER_NOT_ATTACHED")

    def evaluate_predictions(self) -> dict[str, Any]:
        self._stage("independent_prediction_evaluation")
        raise TechnicalStop("REAL_PREDICTION_ADAPTER_NOT_ATTACHED")

    def run_task_comparison(self) -> dict[str, Any]:
        self._stage("conditional_task_comparison")
        self.task_calls += 1
        raise TechnicalStop("REAL_TASK_ADAPTER_NOT_ATTACHED")

    def settle(self, status: str, details: dict[str, Any]) -> None:
        self._stage("settlement")
        (self.output / "settlement.json").write_text(json.dumps({"status": status, "task_calls": self.task_calls, "ledger": self.ledger.assert_settled(), "automatic_retry": False}, indent=2) + "\n", encoding="utf-8")

    def close(self) -> None:
        self.ledger.close()


class FakeABBackend:
    """Integration-only backend; enabled only by an explicit test contract."""

    def __init__(self, root: Path, output: Path, mode: str = "pass"):
        self.root, self.output, self.mode = root, output, mode
        self.events: list[str] = []
        self.task_calls = 0

    def _stage(self, name: str) -> None:
        self.events.append(name)
        value = {"stage": name, "run": "fake-ab-backend"}
        try:
            durable_atomic_json(self.output / "activity.json", value)
        except PermissionError:
            # Direct unit tests run on Windows; the real entry runs this write
            # in Linux native storage where fsync is required.
            (self.output / "activity.json").write_text(json.dumps(value) + "\n", encoding="utf-8")

    def zero_step_gate(self) -> dict[str, Any]:
        self._stage("staging_and_zero_step_gate")
        if self.mode == "reuse_fail":
            return {"pass": False, "environment_constructed": False, "model_constructed": False}
        return {"pass": True, "environment_constructed": False, "model_constructed": False}

    def audit_reused_labels(self) -> dict[str, Any]:
        self._stage("data_reuse_audit")
        if self.mode == "reuse_fail":
            return {"pass": False, "reason": "reuse audit injected failure"}
        return {"pass": True, "train_records": 219, "model_selection_records": 68}

    def collect_confirmation_labels(self) -> dict[str, Any]:
        self._stage("new_prediction_confirmation")
        if self.mode == "confirmation_fail":
            return {"pass": False, "coverage": {"observed": 0, "required": 8}}
        (self.output / "first-label.json").write_text(json.dumps({"sample_id": "fake:first", "target": 1.25, "unknown": False}) + "\n", encoding="utf-8")
        return {"pass": True, "parent_coverage": 8, "missing_windows": 0, "first_label_persisted": True}

    def train_variants(self) -> dict[str, Any]:
        self._stage("supervised_training_and_selection")
        if self.mode == "nan":
            raise TechnicalStop("NaN/Inf in fake training output")
        rows = []
        for variant in ("A", "B"):
            for seed in (7101, 7102, 7103):
                rows.append({"variant": variant, "seed": seed, "checkpoint": f"fake-{variant}-{seed}.pt", "updates": 3})
        (self.output / "training.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        return {"variants": ["A", "B"], "seeds": [7101, 7102, 7103], "independent_outputs": True, "rows": rows}

    def evaluate_predictions(self) -> dict[str, Any]:
        self._stage("independent_prediction_evaluation")
        if self.mode == "nan":
            raise TechnicalStop("NaN/Inf in prediction output")
        trace = []
        for variant in ("A", "B"):
            for parent in range(8):
                trace.append({"variant": variant, "parent": f"confirm-{parent}", "repeat": 0, "window": f"w-{parent}", "candidate": 0, "prediction": float(parent), "target": float(parent + (0.01 if variant == "A" else 0.0)), "public_input_sha256": hashlib.sha256(f"{parent}".encode()).hexdigest(), "continuation_id": "hungarian-v1-fixed"})
        (self.output / "per-candidate-predictions.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in trace), encoding="utf-8")
        if self.mode == "gate_fail":
            b_minus_a, non_worse, strict = -0.001, 2, False
        else:
            b_minus_a, non_worse, strict = -0.006, 6, True
        metrics = {"b_minus_a_parent_macro_regret": b_minus_a, "parents_b_no_worse": non_worse, "b_strictly_better_than_transparent": strict, "mae": {"A": 0.10, "B": 0.11}, "regret": {"A": 0.020, "B": 0.014}, "top1": {"A": 0.50, "B": 0.625}, "trace_rows": len(trace), "trace_complete": True, "independent_recomputation": True}
        (self.output / "prediction-metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return metrics

    def run_task_comparison(self) -> dict[str, Any]:
        self._stage("conditional_task_comparison")
        self.task_calls += 1
        rows = [{"arm": arm, "parent": f"task-{parent}", "utility": 1.0 + 0.01 * parent} for parent in range(8) for arm in ("hungarian", "transparent_one_shot", "A_one_shot", "B_one_shot")]
        (self.output / "task-comparison.json").write_text(json.dumps({"task_calls": len(rows), "arms": rows}, indent=2) + "\n", encoding="utf-8")
        return {"task_calls": len(rows), "arms": 4, "parent_count": 8, "pairing_verified": True}

    def settle(self, status: str, details: dict[str, Any]) -> None:
        self._stage("settlement")
        (self.output / "settlement.json").write_text(json.dumps({"status": status, "task_calls": self.task_calls, "ledger_closed": True, "automatic_retry": False}, indent=2) + "\n", encoding="utf-8")
