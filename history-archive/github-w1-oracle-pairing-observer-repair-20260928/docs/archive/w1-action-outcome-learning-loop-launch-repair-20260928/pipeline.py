"""Fail-closed stage orchestration for the one-shot learning experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from learning import TrainingConfig, load_checkpoint, predict_checkpoint, train_seed
from learning_schema import LearningRecord, assert_parent_disjoint, load_records
from prediction_evaluation import compare_all, ensemble_mean, prediction_gate


class PipelineBackend(Protocol):
    def select_stage(self, stage: str) -> None: ...
    def account_call(self, name: str, amounts: dict[str, int], function, *args): ...
    def zero_step_gate(self) -> dict[str, Any]: ...
    def collect_fixed_continuation_labels(self) -> Path: ...
    def run_task_comparison(self, models: Sequence[object]) -> dict[str, Any]: ...
    def settle(self, status: str, details: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class PipelineResult:
    status: str
    prediction_gate_passed: bool
    task_comparison_executed: bool
    details: dict[str, Any]


def execute_pipeline(backend: PipelineBackend, output: Path) -> PipelineResult:
    """Execute all authorized stages once; no stage retries or budget borrowing."""

    output.mkdir(parents=True, exist_ok=False)
    details: dict[str, Any] = {"zero_step_gate": backend.zero_step_gate()}
    if not details["zero_step_gate"].get("pass", False):
        result = PipelineResult("technical_stop", False, False, details)
        backend.settle(result.status, details)
        return result

    backend.select_stage("label_collection")
    data_path = backend.collect_fixed_continuation_labels()
    records = load_records(data_path)
    assert_parent_disjoint(records)
    by_role = {
        role: [record for record in records if record.split_role == role]
        for role in ("train", "model_selection", "prediction_evaluation")
    }
    if any(not values for values in by_role.values()):
        details["data_contract"] = "missing one or more required split roles"
        result = PipelineResult("data_contract_stop", False, False, details)
        backend.settle(result.status, details)
        return result

    backend.select_stage("supervised_training_and_selection")
    config = TrainingConfig()
    checkpoint_paths: list[Path] = []
    training_rows = []
    for seed in config.seeds:
        checkpoint = output / "checkpoints" / f"seed-{seed}.pt"
        training_rows.append(train_seed(
            by_role["train"], by_role["model_selection"], seed=seed,
            output_path=checkpoint, config=config, account_call=backend.account_call,
        ))
        checkpoint_paths.append(checkpoint)
    details["training"] = training_rows

    backend.select_stage("independent_prediction_evaluation")
    seed_predictions = []
    models = []
    for seed, checkpoint in zip(config.seeds, checkpoint_paths, strict=True):
        model, metadata = load_checkpoint(
            checkpoint, expected_seed=seed, account_call=backend.account_call
        )
        models.append(model)
        if metadata["seed"] != seed:
            raise RuntimeError("selected checkpoint seed mismatch")
        seed_predictions.append(predict_checkpoint(
            model,
            by_role["prediction_evaluation"],
            account_call=backend.account_call,
        ))
    learned = ensemble_mean(seed_predictions)
    comparison = compare_all(by_role["prediction_evaluation"], learned)
    gate = prediction_gate(comparison)
    details["prediction_evaluation"] = comparison
    details["prediction_gate"] = gate
    (output / "prediction-evaluation.json").write_text(
        json.dumps({"comparison": comparison, "gate": gate}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not gate["pass"]:
        result = PipelineResult("prediction_gate_stop", False, False, details)
        backend.settle(result.status, details)
        return result

    backend.select_stage("conditional_task_comparison")
    details["task_comparison"] = backend.run_task_comparison(models)
    result = PipelineResult("complete", True, True, details)
    backend.settle(result.status, details)
    return result
