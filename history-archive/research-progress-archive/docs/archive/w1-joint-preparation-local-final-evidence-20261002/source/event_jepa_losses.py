"""Masked, independently switchable JEPA/state/event losses."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def _finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"non-finite {name}")


def masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, *, beta: float = 1.0) -> torch.Tensor:
    if prediction.shape != target.shape or valid.shape != prediction.shape:
        raise ValueError("continuous target and validity shapes differ")
    valid = valid.bool()
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    value = F.smooth_l1_loss(prediction[valid], target[valid], beta=beta)
    _finite("state loss", value)
    return value


def masked_event_bce(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if logits.shape != labels.shape or valid.shape != logits.shape:
        raise ValueError("event target and validity shapes differ")
    valid = valid.bool()
    if not bool(valid.any()):
        return logits.sum() * 0.0
    value = F.binary_cross_entropy_with_logits(logits[valid], labels[valid])
    _finite("event loss", value)
    return value


def jepa_objective(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    state_prediction: torch.Tensor | None = None,
    state_target: torch.Tensor | None = None,
    state_valid: torch.Tensor | None = None,
    outcome_prediction: torch.Tensor | None = None,
    outcome_target: torch.Tensor | None = None,
    outcome_valid: torch.Tensor | None = None,
    horizon_task_outcome_logits: torch.Tensor | None = None,
    horizon_task_outcome_target: torch.Tensor | None = None,
    horizon_task_outcome_valid: torch.Tensor | None = None,
    event_logits: torch.Tensor | None = None,
    event_labels: torch.Tensor | None = None,
    event_valid: torch.Tensor | None = None,
    beta_state: float = 1.0,
    beta_outcome: float = 1.0,
    beta_horizon_task_outcome: float = 1.0,
    beta_event: float = 0.0,
    beta_collapse: float = 0.04,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if predicted.shape != target.shape or valid.shape != predicted.shape:
        raise ValueError("JEPA target shapes differ")
    jepa = masked_smooth_l1(predicted, target.detach(), valid)
    state = predicted.sum() * 0.0
    outcome = predicted.sum() * 0.0
    event = predicted.sum() * 0.0
    horizon_task_outcome = predicted.sum() * 0.0
    if state_prediction is not None:
        if state_target is None or state_valid is None:
            raise ValueError("state target and mask are required together")
        state = masked_smooth_l1(state_prediction, state_target, state_valid)
    if outcome_prediction is not None:
        if outcome_target is None or outcome_valid is None:
            raise ValueError("outcome target and mask are required together")
        outcome = masked_smooth_l1(outcome_prediction, outcome_target, outcome_valid)
    if event_logits is not None:
        if event_labels is None or event_valid is None:
            raise ValueError("event labels and mask are required together")
        event = masked_event_bce(event_logits, event_labels, event_valid)
    if horizon_task_outcome_logits is not None:
        if horizon_task_outcome_target is None or horizon_task_outcome_valid is None:
            raise ValueError("horizon task-outcome target and mask are required together")
        horizon_task_outcome = masked_event_bce(
            horizon_task_outcome_logits,
            horizon_task_outcome_target,
            horizon_task_outcome_valid,
        )
    centered = predicted - predicted.mean(dim=0, keepdim=True)
    collapse = F.relu(1.0 - (centered.var(dim=0, unbiased=False) + 1e-4).sqrt()).mean()
    total = (
        jepa
        + float(beta_state) * state
        + float(beta_outcome) * outcome
        + float(beta_horizon_task_outcome) * horizon_task_outcome
        + float(beta_event) * event
        + float(beta_collapse) * collapse
    )
    _finite("total objective", total)
    return total, {
        "jepa": jepa,
        "state": state,
        "outcome": outcome,
        "horizon_task_outcome": horizon_task_outcome,
        "event": event,
        "collapse": collapse,
    }


__all__ = ["masked_smooth_l1", "masked_event_bce", "jepa_objective"]
