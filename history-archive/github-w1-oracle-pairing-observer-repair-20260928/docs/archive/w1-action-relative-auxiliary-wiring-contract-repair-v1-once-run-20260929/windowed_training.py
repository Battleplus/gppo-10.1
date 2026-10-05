"""Lazy-torch training adapter for the two frozen objective variants."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from objective import Window, group_rows, window_loss_torch


MODEL_SCHEMA = "w1-action-outcome-windowed-mlp/1.0.0"


@dataclass(frozen=True)
class ModelSpec:
    input_dim: int = 827
    hidden_dims: tuple[int, int] = (128, 64)
    output_dim: int = 1
    activation: str = "tanh"
    continuation_id: str = "hungarian-v1-fixed"

    def validate(self) -> None:
        if (self.input_dim, self.hidden_dims, self.output_dim, self.activation) != (827, (128, 64), 1, "tanh"):
            raise ValueError("model architecture differs from the frozen W1 contract")
        if self.continuation_id != "hungarian-v1-fixed":
            raise ValueError("continuation identity changed")


@dataclass(frozen=True)
class TrainingConfig:
    seeds: tuple[int, ...] = (7101, 7102, 7103)
    windows_per_update: int = 1
    max_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    early_stopping_patience: int = 12
    minimum_delta: float = 0.0
    auxiliary_weight: float = 0.25
    beta: float = 1.0

    def validate(self) -> None:
        if self.seeds != (7101, 7102, 7103):
            raise ValueError("training seeds changed")
        if self.windows_per_update != 1 or self.max_epochs != 100:
            raise ValueError("window update contract changed")
        if self.learning_rate != 3e-4 or self.weight_decay != 0.0:
            raise ValueError("optimizer contract changed")
        if self.early_stopping_patience != 12 or self.minimum_delta != 0.0:
            raise ValueError("selection contract changed")
        if not math.isclose(self.auxiliary_weight, 0.25, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("auxiliary coefficient is not the frozen value")
        if self.beta <= 0 or not math.isfinite(self.beta):
            raise ValueError("invalid SmoothL1 beta")


def build_model(spec: ModelSpec | None = None):
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


def _field(record: Any, name: str):
    return getattr(record, name) if hasattr(record, name) else record[name]


def _features(record: Any):
    from learning_schema import materialize_features

    return materialize_features(_field(record, "flat"), _field(record, "history"), _field(record, "action"))


def _window_records(records: Sequence[Any]) -> tuple[Window, ...]:
    from objective import CandidateRow

    rows = []
    for record in records:
        rows.append(CandidateRow(
            decision_id=str(_field(record, "decision_id")),
            parent=str(_field(record, "parent")),
            repeat=int(_field(record, "repeat")),
            action=int(_field(record, "action")),
            legal_actions=tuple(int(value) for value in _field(record, "legal_actions")),
            target=float(_field(record, "remaining_utility")),
            continuation_id=str(_field(record, "continuation_id")),
            target_valid=True,
            noop=int(_field(record, "action")) == 24,
        ))
    return group_rows(rows)


def _tensor_window(model, window: Window, lookup: Mapping[tuple[str, int], Any], device: str):
    import torch

    features = [_features(lookup[(window.decision_id, row.action)]) for row in window.rows]
    targets = [row.target for row in window.rows]
    return (
        torch.tensor(features, dtype=torch.float32, device=device),
        torch.tensor(targets, dtype=torch.float32, device=device),
    )


def _selection_mae(model, windows: Sequence[Window], lookup: Mapping[tuple[str, int], Any], device: str, caller: Callable) -> float:
    import torch

    values = []
    model.eval()
    with torch.no_grad():
        for window in windows:
            features, targets = _tensor_window(model, window, lookup, device)
            predictions = caller("model.forward.selection", {"model_batch_forwards": 1, "model_sample_evaluations": len(window.rows)}, model, features)[:, 0]
            values.append(float(torch.abs(predictions - targets).mean().cpu()))
    if not values:
        raise ValueError("selection set is empty")
    return sum(values) / len(values)


def train_variant(
    train_records: Sequence[Any],
    selection_records: Sequence[Any],
    *,
    variant: str,
    seed: int,
    output_path: Path,
    config: TrainingConfig | None = None,
    spec: ModelSpec | None = None,
    device: str = "cpu",
    account_call: Callable | None = None,
) -> dict[str, Any]:
    """Train one seed using complete windows and equal window weighting.

    This function is only imported/called by a future approved runner.  The
    preparation tests exercise the pure objective and never reach this body.
    """

    import torch

    frozen_config = config or TrainingConfig()
    frozen_spec = spec or ModelSpec()
    frozen_config.validate()
    frozen_spec.validate()
    if variant not in {"absolute", "centered_auxiliary"}:
        raise ValueError("unknown objective variant")
    if seed not in frozen_config.seeds:
        raise ValueError("seed outside frozen set")

    train_windows = _window_records(train_records)
    selection_windows = _window_records(selection_records)
    train_lookup = {(str(_field(row, "decision_id")), int(_field(row, "action"))): row for row in train_records}
    selection_lookup = {(str(_field(row, "decision_id")), int(_field(row, "action"))): row for row in selection_records}
    if {window.rows[0].parent for window in train_windows} & {window.rows[0].parent for window in selection_windows}:
        raise ValueError("train and model-selection parents overlap")

    caller = account_call or (lambda _name, _amounts, function, *args: function(*args))
    torch.manual_seed(seed)
    model = caller("model.initialize", {"model_initializations_or_loads": 1}, build_model, frozen_spec).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=frozen_config.learning_rate, weight_decay=frozen_config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_state = None
    best_mae = math.inf
    best_epoch = -1
    stale = 0
    updates = 0
    history = []
    for epoch in range(frozen_config.max_epochs):
        model.train()
        order = torch.randperm(len(train_windows), generator=generator)
        losses = []
        for position in order.tolist():
            window = train_windows[position]
            features, targets = _tensor_window(model, window, train_lookup, device)
            prediction = caller("model.forward.training", {"model_batch_forwards": 1, "model_sample_evaluations": len(window.rows)}, model, features)[:, 0]
            total, absolute, centered_value = window_loss_torch(
                prediction, targets, variant=variant,
                auxiliary_weight=frozen_config.auxiliary_weight,
                beta=frozen_config.beta,
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            caller("optimizer.step", {"offline_predictor_updates": 1}, optimizer.step)
            updates += 1
            losses.append((float(total.detach().cpu()), float(absolute.cpu()), float(centered_value.cpu())))
        selection_mae = _selection_mae(model, selection_windows, selection_lookup, device, caller)
        history.append({"epoch": epoch, "training_total": sum(item[0] for item in losses) / len(losses), "training_absolute": sum(item[1] for item in losses) / len(losses), "training_centered": sum(item[2] for item in losses) / len(losses), "selection_mae": selection_mae, "optimizer_updates": updates})
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
        raise RuntimeError("no selectable checkpoint")
    metadata = {"schema": MODEL_SCHEMA, "variant": variant, "seed": seed, "model_spec": asdict(frozen_spec), "training_config": asdict(frozen_config), "best_epoch": best_epoch, "selection_mae": best_mae, "optimizer_updates": updates, "train_parents": sorted({window.rows[0].parent for window in train_windows}), "selection_parents": sorted({window.rows[0].parent for window in selection_windows})}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    caller("checkpoint.write", {"checkpoint_writes": 1}, torch.save, {"metadata": metadata, "state_dict": best_state}, output_path)
    return {**metadata, "history": history}
