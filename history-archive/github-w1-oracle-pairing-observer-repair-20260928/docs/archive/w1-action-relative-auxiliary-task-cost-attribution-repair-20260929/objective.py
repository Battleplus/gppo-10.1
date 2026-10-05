"""Window-balanced absolute and centered utility objectives.

This module is deliberately independent of the environment and PyTorch.  The
future runner can call :func:`window_loss_torch` after importing torch, while
the contract and all preparation tests remain executable without a model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


NOOP_ACTION = 24
CONTINUATION_ID = "hungarian-v1-fixed"


@dataclass(frozen=True)
class CandidateRow:
    """The fields needed to define one candidate inside one decision window."""

    decision_id: str
    parent: str
    repeat: int
    action: int
    legal_actions: tuple[int, ...]
    target: float
    continuation_id: str = CONTINUATION_ID
    target_valid: bool = True
    noop: bool = False

    def validate(self) -> None:
        if not self.decision_id or not self.parent:
            raise ValueError("window identities are required")
        if type(self.repeat) is not int or self.repeat < 0:
            raise ValueError("repeat must be a nonnegative integer")
        if type(self.action) is not int or not 0 <= self.action <= NOOP_ACTION:
            raise ValueError("action outside frozen range")
        if not self.legal_actions or len(set(self.legal_actions)) != len(self.legal_actions):
            raise ValueError("legal candidate set is empty or duplicated")
        if any(type(value) is not int for value in self.legal_actions):
            raise ValueError("legal candidate set contains non-integers")
        if self.action not in self.legal_actions:
            raise ValueError("candidate is not legal in its window")
        if self.continuation_id != CONTINUATION_ID:
            raise ValueError("continuation identity mismatch")
        if self.target_valid and not math.isfinite(float(self.target)):
            raise ValueError("valid target must be finite")
        if self.noop != (self.action == NOOP_ACTION):
            raise ValueError("noop flag does not match action")


@dataclass(frozen=True)
class Window:
    decision_id: str
    rows: tuple[CandidateRow, ...]

    def validate(self) -> None:
        if not self.rows or self.rows[0].decision_id != self.decision_id:
            raise ValueError("window identity or emptiness is invalid")
        for row in self.rows:
            row.validate()
            if row.decision_id != self.decision_id:
                raise ValueError("window crosses decision identities")
        first = self.rows[0]
        identity = (first.parent, first.repeat, first.legal_actions, first.continuation_id)
        if any((row.parent, row.repeat, row.legal_actions, row.continuation_id) != identity for row in self.rows):
            raise ValueError("window rows do not share public/candidate contract")
        if {row.action for row in self.rows} != set(first.legal_actions):
            raise ValueError("window does not contain the complete legal candidate set")
        if any(not row.target_valid for row in self.rows):
            raise ValueError("invalid/unknown target cannot enter objective")


def group_rows(rows: Iterable[CandidateRow]) -> tuple[Window, ...]:
    grouped: dict[str, list[CandidateRow]] = {}
    for row in rows:
        grouped.setdefault(row.decision_id, []).append(row)
    windows = tuple(Window(key, tuple(sorted(value, key=lambda item: item.action))) for key, value in sorted(grouped.items()))
    for window in windows:
        window.validate()
    return windows


def smooth_l1_scalar(prediction: float, target: float, beta: float = 1.0) -> float:
    if not math.isfinite(float(prediction)) or not math.isfinite(float(target)):
        raise ValueError("SmoothL1 received a nonfinite value")
    if not math.isfinite(float(beta)) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    error = abs(float(prediction) - float(target))
    return 0.5 * error * error / beta if error < beta else error - 0.5 * beta


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def centered(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        raise ValueError("cannot center an empty sequence")
    mean = _mean([float(value) for value in values])
    return tuple(float(value) - mean for value in values)


def window_loss(
    predictions: Sequence[float],
    targets: Sequence[float],
    *,
    variant: str,
    auxiliary_weight: float = 0.0,
    beta: float = 1.0,
) -> dict[str, float]:
    """Return one-window losses; callers average these values across windows.

    Variant ``absolute`` is A. Variant ``centered_auxiliary`` is B.  The
    auxiliary term is computed after centering both candidate vectors within
    the same complete legal window.  It cannot use a candidate subset.
    """

    if len(predictions) == 0 or len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal nonzero length")
    if variant not in {"absolute", "centered_auxiliary"}:
        raise ValueError("unknown objective variant")
    if not math.isfinite(float(auxiliary_weight)) or auxiliary_weight < 0:
        raise ValueError("auxiliary weight must be finite and nonnegative")
    absolute = _mean([smooth_l1_scalar(prediction, target, beta) for prediction, target in zip(predictions, targets, strict=True)])
    centered_loss = _mean([
        smooth_l1_scalar(prediction, target, beta)
        for prediction, target in zip(centered(predictions), centered(targets), strict=True)
    ])
    applied_weight = auxiliary_weight if variant == "centered_auxiliary" else 0.0
    return {
        "absolute": absolute,
        "centered": centered_loss,
        "auxiliary_weight": applied_weight,
        "total": absolute + applied_weight * centered_loss,
    }


def aggregate_window_loss(
    windows: Sequence[Window],
    predictions: Mapping[str, Sequence[float]],
    *,
    variant: str,
    auxiliary_weight: float = 0.0,
    beta: float = 1.0,
) -> dict[str, float]:
    """Average loss equally over windows, never over candidate rows."""

    if not windows:
        raise ValueError("at least one window is required")
    values = []
    for window in windows:
        window.validate()
        scores = predictions.get(window.decision_id)
        if scores is None or len(scores) != len(window.rows):
            raise ValueError(f"missing predictions for {window.decision_id}")
        values.append(window_loss(
            scores,
            [row.target for row in window.rows],
            variant=variant,
            auxiliary_weight=auxiliary_weight,
            beta=beta,
        ))
    return {key: _mean([item[key] for item in values]) for key in values[0]}


def window_loss_torch(predictions, targets, *, variant: str, auxiliary_weight: float, beta: float = 1.0):
    """Torch equivalent used only by an authorized future training runner."""

    import torch
    import torch.nn.functional as functional

    if predictions.ndim != 1 or targets.ndim != 1 or predictions.shape != targets.shape:
        raise ValueError("torch predictions and targets must be matching vectors")
    if predictions.numel() == 0:
        raise ValueError("empty torch window")
    absolute = functional.smooth_l1_loss(predictions, targets, beta=beta, reduction="mean")
    centered_loss = functional.smooth_l1_loss(predictions - predictions.mean(), targets - targets.mean(), beta=beta, reduction="mean")
    if variant == "absolute":
        return absolute, absolute.detach(), centered_loss.detach()
    if variant == "centered_auxiliary":
        if auxiliary_weight < 0 or not math.isfinite(float(auxiliary_weight)):
            raise ValueError("invalid auxiliary weight")
        return absolute + float(auxiliary_weight) * centered_loss, absolute.detach(), centered_loss.detach()
    raise ValueError("unknown objective variant")
