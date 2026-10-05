"""Stage ordering and frozen research gates for the W1 EAWM/JEPA run."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


PREDICTION_PARENTS_REQUIRED = 8
REGRET_MARGIN_G1_OVER_TRANSPARENT = 0.005
REGRET_MARGIN_G2_OVER_G1 = 0.0025
EVENT_BRIER_MARGIN_G2_OVER_G1 = 0.01
PARENTS_NO_WORSE_REQUIRED = 4


class PipelineBackend(Protocol):
    def zero_step_gate(self) -> dict[str, Any]: ...
    def collect_world_model_data(self) -> dict[str, Any]: ...
    def train_world_models(self) -> dict[str, Any]: ...
    def evaluate_world_models(self) -> dict[str, Any]: ...
    def train_policies(self) -> dict[str, Any]: ...
    def evaluate_tasks(self) -> dict[str, Any]: ...
    def settle(self, status: str, evidence: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class PipelineResult:
    status: str
    policy_training_executed: bool
    task_comparison_executed: bool
    evidence: dict[str, Any]


def prediction_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    required = (
        "valid_parent_count", "g1_minus_transparent_macro_regret",
        "g2_minus_g1_macro_regret", "g2_minus_g1_event_brier",
        "parents_g1_no_worse_than_transparent", "parents_g2_no_worse_than_g1",
    )
    missing = [name for name in required if name not in metrics]
    if missing:
        raise ValueError("prediction metrics missing: " + ",".join(missing))
    coverage = int(metrics["valid_parent_count"]) == PREDICTION_PARENTS_REQUIRED
    g1 = (
        float(metrics["g1_minus_transparent_macro_regret"]) <= -REGRET_MARGIN_G1_OVER_TRANSPARENT
        and int(metrics["parents_g1_no_worse_than_transparent"]) >= PARENTS_NO_WORSE_REQUIRED
    )
    g2 = (
        float(metrics["g2_minus_g1_macro_regret"]) <= -REGRET_MARGIN_G2_OVER_G1
        and float(metrics["g2_minus_g1_event_brier"]) <= -EVENT_BRIER_MARGIN_G2_OVER_G1
        and int(metrics["parents_g2_no_worse_than_g1"]) >= PARENTS_NO_WORSE_REQUIRED
    )
    return {
        "pass": bool(coverage and g1 and g2),
        "coverage_pass": coverage, "g1_transparent_gate_pass": g1,
        "g2_event_increment_gate_pass": g2,
        "thresholds": {
            "valid_parents": PREDICTION_PARENTS_REQUIRED,
            "g1_regret_margin": REGRET_MARGIN_G1_OVER_TRANSPARENT,
            "g2_regret_margin": REGRET_MARGIN_G2_OVER_G1,
            "g2_event_brier_margin": EVENT_BRIER_MARGIN_G2_OVER_G1,
            "parents_no_worse": PARENTS_NO_WORSE_REQUIRED,
        },
    }


def execute_pipeline(backend: PipelineBackend) -> PipelineResult:
    evidence: dict[str, Any] = {}
    preflight = backend.zero_step_gate()
    evidence["zero_step_gate"] = preflight
    if not preflight.get("pass"):
        result = PipelineResult("technical_stop", False, False, evidence)
        backend.settle(result.status, evidence)
        return result
    data = backend.collect_world_model_data()
    evidence["data"] = data
    if not data.get("pass") or int(data.get("prediction_parent_count", 0)) != PREDICTION_PARENTS_REQUIRED:
        result = PipelineResult("data_coverage_stop", False, False, evidence)
        backend.settle(result.status, evidence)
        return result
    evidence["world_model_training"] = backend.train_world_models()
    metrics = backend.evaluate_world_models()
    evidence["prediction_metrics"] = metrics
    gate = prediction_gate(metrics)
    evidence["prediction_gate"] = gate
    if not gate["pass"]:
        result = PipelineResult("prediction_gate_stop", False, False, evidence)
        backend.settle(result.status, evidence)
        return result
    evidence["policy_training"] = backend.train_policies()
    evidence["task_comparison"] = backend.evaluate_tasks()
    result = PipelineResult("complete", True, True, evidence)
    backend.settle(result.status, evidence)
    return result


__all__ = ["PipelineBackend", "PipelineResult", "prediction_gate", "execute_pipeline"]
