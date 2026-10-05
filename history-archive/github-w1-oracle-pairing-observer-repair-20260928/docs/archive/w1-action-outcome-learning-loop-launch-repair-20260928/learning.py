"""Minimal trainable remaining-utility model with lazy PyTorch imports."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from learning_schema import INPUT_DIM, LearningRecord, materialize_features


MODEL_SCHEMA = "w1-action-outcome-mlp/1.0.0"


@dataclass(frozen=True)
class ModelSpec:
    input_dim: int = INPUT_DIM
    hidden_dims: tuple[int, int] = (128, 64)
    output_dim: int = 1
    activation: str = "tanh"
    continuation_id: str = "hungarian-v1-fixed"
    primary_target: str = "remaining_frozen_utility_to_native_termination"

    def validate(self) -> None:
        if (self.input_dim, self.hidden_dims, self.output_dim, self.activation) != (
            827, (128, 64), 1, "tanh"
        ):
            raise ValueError("model architecture differs from the frozen minimal learner")
        if self.continuation_id != "hungarian-v1-fixed":
            raise ValueError("model continuation identity changed")


@dataclass(frozen=True)
class TrainingConfig:
    seeds: tuple[int, ...] = (7101, 7102, 7103)
    batch_size: int = 64
    max_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    early_stopping_patience: int = 12
    minimum_delta: float = 0.0

    def validate(self) -> None:
        if self.seeds != (7101, 7102, 7103):
            raise ValueError("training seeds changed")
        if self.batch_size != 64 or self.max_epochs != 100:
            raise ValueError("training budget changed")
        if self.learning_rate != 3e-4 or self.weight_decay != 0.0:
            raise ValueError("optimizer contract changed")
        if self.early_stopping_patience != 12 or self.minimum_delta != 0.0:
            raise ValueError("selection contract changed")


def build_model(spec: ModelSpec | None = None):
    """Construct the real network only when an authorized execution calls it."""

    import torch.nn as nn

    frozen = spec or ModelSpec()
    frozen.validate()
    return nn.Sequential(
        nn.Linear(frozen.input_dim, frozen.hidden_dims[0]),
        nn.Tanh(),
        nn.Linear(frozen.hidden_dims[0], frozen.hidden_dims[1]),
        nn.Tanh(),
        nn.Linear(frozen.hidden_dims[1], frozen.output_dim),
    )


def _tensors(records: Sequence[LearningRecord], device: str):
    import torch

    features = [materialize_features(row.flat, row.history, row.action) for row in records]
    labels = [row.remaining_utility for row in records]
    return (
        torch.tensor(features, dtype=torch.float32, device=device),
        torch.tensor(labels, dtype=torch.float32, device=device)[:, None],
    )


def _direct_call(_name: str, _amounts: Mapping[str, int], function: Callable, *args):
    return function(*args)


def _predict_batches(
    model,
    records: Sequence[LearningRecord],
    batch_size: int,
    device: str,
    *,
    account_call: Callable | None = None,
) -> list[float]:
    import torch

    features, _labels = _tensors(records, device)
    result: list[float] = []
    model.eval()
    caller = account_call or _direct_call
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = features[start:start + batch_size]
            output = caller(
                "model.forward",
                {"model_batch_forwards": 1, "model_sample_evaluations": len(batch)},
                model,
                batch,
            )
            result.extend(float(value) for value in output[:, 0].cpu())
    return result


def train_seed(
    train_records: Sequence[LearningRecord],
    selection_records: Sequence[LearningRecord],
    *,
    seed: int,
    output_path: Path,
    config: TrainingConfig | None = None,
    spec: ModelSpec | None = None,
    device: str = "cpu",
    account_call: Callable | None = None,
) -> dict[str, Any]:
    """Train one seed and select its checkpoint only on model-selection parents."""

    import torch

    frozen_config = config or TrainingConfig()
    frozen_spec = spec or ModelSpec()
    frozen_config.validate()
    frozen_spec.validate()
    if seed not in frozen_config.seeds:
        raise ValueError("seed is outside the frozen set")
    if {row.split_role for row in train_records} != {"train"}:
        raise ValueError("training loader received a non-training split")
    if {row.split_role for row in selection_records} != {"model_selection"}:
        raise ValueError("selection loader received a non-selection split")
    if {row.parent for row in train_records} & {row.parent for row in selection_records}:
        raise ValueError("training and model-selection parents overlap")

    caller = account_call or _direct_call
    torch.manual_seed(seed)
    model = caller(
        "model.initialize",
        {"model_initializations_or_loads": 1},
        build_model,
        frozen_spec,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=frozen_config.learning_rate, weight_decay=frozen_config.weight_decay
    )
    train_x, train_y = _tensors(train_records, device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_state = None
    best_mae = math.inf
    best_epoch = -1
    stale = 0
    updates = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(frozen_config.max_epochs):
        model.train()
        order = torch.randperm(len(train_records), generator=generator)
        losses: list[float] = []
        for start in range(0, len(train_records), frozen_config.batch_size):
            indexes = order[start:start + frozen_config.batch_size].to(device)
            batch = train_x[indexes]
            prediction = caller(
                "model.forward.training",
                {"model_batch_forwards": 1, "model_sample_evaluations": len(batch)},
                model,
                batch,
            )
            loss = torch.nn.functional.smooth_l1_loss(prediction, train_y[indexes])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            caller(
                "optimizer.step",
                {"offline_predictor_updates": 1},
                optimizer.step,
            )
            updates += 1
            losses.append(float(loss.detach().cpu()))
        selection_predictions = _predict_batches(
            model, selection_records, frozen_config.batch_size, device,
            account_call=account_call,
        )
        selection_mae = sum(
            abs(value - row.remaining_utility)
            for value, row in zip(selection_predictions, selection_records, strict=True)
        ) / len(selection_records)
        history.append({
            "epoch": epoch,
            "training_loss": sum(losses) / len(losses),
            "selection_mae": selection_mae,
            "optimizer_updates": updates,
        })
        if selection_mae < best_mae - frozen_config.minimum_delta:
            best_mae = selection_mae
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= frozen_config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no selectable checkpoint")
    metadata = {
        "schema": MODEL_SCHEMA,
        "seed": seed,
        "model_spec": asdict(frozen_spec),
        "training_config": asdict(frozen_config),
        "best_epoch": best_epoch,
        "selection_mae": best_mae,
        "optimizer_updates": updates,
        "train_parents": sorted({row.parent for row in train_records}),
        "selection_parents": sorted({row.parent for row in selection_records}),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    caller(
        "checkpoint.write",
        {"checkpoint_writes": 1},
        torch.save,
        {"metadata": metadata, "state_dict": best_state},
        output_path,
    )
    return {**metadata, "history": history}


def load_checkpoint(
    path: Path,
    *,
    expected_seed: int,
    device: str = "cpu",
    account_call: Callable | None = None,
):
    import torch

    payload = torch.load(path, map_location=device, weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("schema") != MODEL_SCHEMA or metadata.get("seed") != expected_seed:
        raise ValueError("checkpoint identity mismatch")
    spec_data = metadata.get("model_spec", {})
    if isinstance(spec_data.get("hidden_dims"), list):
        spec_data["hidden_dims"] = tuple(spec_data["hidden_dims"])
    spec = ModelSpec(**spec_data)
    spec.validate()
    caller = account_call or _direct_call
    model = caller(
        "model.load",
        {"model_initializations_or_loads": 1},
        build_model,
        spec,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, metadata


def predict_checkpoint(
    model,
    records: Sequence[LearningRecord],
    *,
    batch_size: int = 64,
    device: str = "cpu",
    account_call: Callable | None = None,
) -> list[float]:
    return _predict_batches(
        model, records, batch_size, device, account_call=account_call
    )
