"""Single-pass production orchestration for the frozen A/B comparison.

The backend owns environment/model details.  This module owns stage order,
gates, accounting boundaries, and the invariant that a failed prediction gate
cannot call the task-comparison stage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


class Backend(Protocol):
    def zero_step_gate(self) -> dict[str, Any]: ...
    def audit_reused_labels(self) -> dict[str, Any]: ...
    def collect_confirmation_labels(self) -> dict[str, Any]: ...
    def train_variants(self) -> dict[str, Any]: ...
    def evaluate_predictions(self) -> dict[str, Any]: ...
    def run_task_comparison(self) -> dict[str, Any]: ...
    def settle(self, status: str, details: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class PipelineResult:
    status: str
    prediction_gate_passed: bool
    task_stage_executed: bool
    details: dict[str, Any]


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gate_passes(metrics: dict[str, Any]) -> tuple[bool, bool]:
    effect = float(metrics["b_minus_a_parent_macro_regret"])
    no_worse = int(metrics["parents_b_no_worse"]) >= 4
    b_vs_transparent = bool(metrics["b_strictly_better_than_transparent"])
    return effect <= -0.005 and no_worse, b_vs_transparent


def execute_pipeline(backend: Backend, output: Path) -> PipelineResult:
    if not output.is_dir():
        raise RuntimeError("backend must initialize output storage before pipeline")
    details: dict[str, Any] = {"stage": "zero_step_gate"}
    gate = backend.zero_step_gate()
    details["zero_step_gate"] = gate
    if not gate.get("pass", False):
        result = PipelineResult("technical_stop", False, False, details)
        backend.settle(result.status, details)
        _write(output / "status.json", {"status": result.status, "task_calls": 0})
        return result

    details["stage"] = "data_reuse_audit"
    reuse = backend.audit_reused_labels()
    details["data_reuse"] = reuse
    if not reuse.get("pass", False):
        result = PipelineResult("data_reuse_stop", False, False, details)
        backend.settle(result.status, details)
        _write(output / "status.json", {"status": result.status, "task_calls": 0})
        return result

    details["stage"] = "new_prediction_confirmation"
    confirmation = backend.collect_confirmation_labels()
    details["confirmation"] = confirmation
    if not confirmation.get("pass", False):
        result = PipelineResult("confirmation_stop", False, False, details)
        backend.settle(result.status, details)
        _write(output / "status.json", {"status": result.status, "task_calls": 0})
        return result

    details["stage"] = "supervised_training_and_selection"
    details["training"] = backend.train_variants()
    details["stage"] = "independent_prediction_evaluation"
    metrics = backend.evaluate_predictions()
    details["prediction_metrics"] = metrics
    effect_gate, task_gate = _gate_passes(metrics)
    details["prediction_gate"] = {
        "effect_gate_pass": effect_gate,
        "task_stage_condition_pass": task_gate,
        "threshold": -0.005,
        "minimum_non_worse_parents": 4,
        "b_strictly_better_than_transparent_required": True,
    }
    _write(output / "prediction-gate.json", details["prediction_gate"])
    if not effect_gate:
        result = PipelineResult("prediction_gate_stop", False, False, details)
        backend.settle(result.status, details)
        _write(output / "status.json", {"status": result.status, "task_calls": 0})
        return result
    if not task_gate:
        result = PipelineResult("prediction_effect_gate_passed_task_condition_failed", True, False, details)
        backend.settle(result.status, details)
        _write(output / "status.json", {"status": result.status, "task_calls": 0})
        return result

    details["stage"] = "conditional_task_comparison"
    task = backend.run_task_comparison()
    details["task_comparison"] = task
    result = PipelineResult("complete", True, True, details)
    backend.settle(result.status, details)
    _write(output / "status.json", {"status": result.status, "task_calls": int(task.get("task_calls", 0))})
    return result
