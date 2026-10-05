"""Explainable candidate-logit prior using predicted frozen remaining utility."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from predictor import CandidateOutcomePredictor, CandidatePrediction


NEG_INF = float("-inf")


@dataclass(frozen=True)
class PriorDiagnostic:
    action: int
    legal: bool
    base_score: float
    prior_score: float | None
    coefficient: float
    final_score: float
    continuation_id: str


def utility_score(prediction: CandidatePrediction) -> float:
    if not prediction.valid.get("remaining_utility", False) or prediction.remaining_utility is None:
        raise ValueError(f"action {prediction.action} lacks a valid remaining-utility prediction")
    if prediction.diagnostics.get("component_utility_score") is not None:
        raise ValueError("direct and component utility scores cannot be combined; that would double count")
    return float(prediction.remaining_utility)


def apply_candidate_prior(
    base_logits: Sequence[float],
    valid_mask: Sequence[bool],
    predictions: Sequence[CandidatePrediction],
    *,
    coefficient: float,
    expected_continuation_id: str,
) -> tuple[list[float], list[PriorDiagnostic]]:
    if len(base_logits) != 25 or len(valid_mask) != 25:
        raise ValueError("W1 action space requires 25 logits and mask entries")
    if not math.isfinite(float(coefficient)) or coefficient < 0:
        raise ValueError("prior coefficient must be finite and nonnegative")
    by_action = {prediction.action: prediction for prediction in predictions}
    if len(by_action) != len(predictions):
        raise ValueError("duplicate candidate predictions")
    legal_actions = {action for action, legal in enumerate(valid_mask) if legal}
    if set(by_action) != legal_actions:
        raise ValueError("predictions must match the legal candidate set exactly")

    final: list[float] = []
    diagnostics: list[PriorDiagnostic] = []
    for action, (base, legal) in enumerate(zip(base_logits, valid_mask, strict=True)):
        base = float(base)
        if math.isnan(base) or base == float("inf") or (legal and base == NEG_INF):
            raise ValueError("legal base logits must be finite; illegal logits may be negative infinity")
        if not legal:
            value = NEG_INF
            prior = None
            continuation = expected_continuation_id
        else:
            prediction = by_action.get(action)
            if prediction is None:
                raise ValueError(f"missing prediction for legal action {action}")
            if prediction.continuation_id != expected_continuation_id:
                raise ValueError("prediction continuation differs from policy prior contract")
            prior = utility_score(prediction)
            value = base + float(coefficient) * prior
            continuation = prediction.continuation_id
        final.append(value)
        diagnostics.append(PriorDiagnostic(
            action=action,
            legal=bool(legal),
            base_score=base,
            prior_score=prior,
            coefficient=float(coefficient),
            final_score=value,
            continuation_id=continuation,
        ))
    return final, diagnostics


def predict_and_apply(
    predictor: CandidateOutcomePredictor,
    public_input: Mapping[str, Any],
    base_logits: Sequence[float],
    valid_mask: Sequence[bool],
    *,
    coefficient: float,
) -> tuple[int, list[PriorDiagnostic]]:
    legal_actions = [index for index, valid in enumerate(valid_mask) if valid]
    predictions = predictor.predict(public_input, legal_actions)
    final, diagnostics = apply_candidate_prior(
        base_logits,
        valid_mask,
        predictions,
        coefficient=coefficient,
        expected_continuation_id=predictor.continuation_id,
    )
    selected = max((index for index, valid in enumerate(valid_mask) if valid), key=final.__getitem__)
    return selected, diagnostics
