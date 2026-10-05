"""Validation-only calibration and risk diagnostics for T-04 Shadow mode."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import functional as F

from .data import TensorTransition, state_vector
from .model import EventAwareGraphWorldModel


@dataclass(frozen=True)
class ShadowCalibration:
    format_version: str
    source_split: str
    source_transition_count: int
    state_change_temperature: float
    continuation_temperature: float
    state_variance_scale: float
    reward_variance_scale: float
    cost_variance_scale: float
    input_mean: tuple[float, ...]
    input_std: tuple[float, ...]
    ood_score_threshold: float
    uncertainty_threshold: float
    latency_p95_budget_ms: float = 25.0
    latency_p99_budget_ms: float = 50.0
    timeout_ms: float = 50.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ShadowCalibration":
        return cls(
            **{
                **value,
                "input_mean": tuple(float(item) for item in value["input_mean"]),
                "input_std": tuple(float(item) for item in value["input_std"]),
            }
        )

    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _episodes(items: Iterable[list[TensorTransition]]) -> list[list[TensorTransition]]:
    return [list(episode) for episode in items]


@torch.no_grad()
def _collect_predictions(model: EventAwareGraphWorldModel, episodes) -> dict[str, np.ndarray]:
    model.eval()
    values: dict[str, list] = {
        "state_logit": [],
        "state_target": [],
        "state_error": [],
        "state_variance": [],
        "reward_error": [],
        "reward_variance": [],
        "cost_error": [],
        "cost_variance": [],
        "continuation_logit": [],
        "continuation_target": [],
    }
    for episode in episodes:
        hidden = None
        for transition in episode:
            output, hidden = model.step(
                transition.graph,
                transition.action,
                hidden,
                sample=False,
                evidence=transition.evidence,
            )
            values["state_logit"].extend(output["state_change_logit"].tolist())
            values["state_target"].extend((transition.target_delta.abs() > 1e-6).float().tolist())
            values["state_error"].extend((transition.target_delta - output["state_delta_soft"]).tolist())
            values["state_variance"].extend(output["state_logvar"].exp().tolist())
            values["reward_error"].append(float(transition.reward - output["reward"]))
            values["reward_variance"].append(float(output["reward_logvar"].exp()))
            values["cost_error"].extend((transition.costs - output["costs"]).tolist())
            values["cost_variance"].extend(output["cost_logvar"].exp().tolist())
            values["continuation_logit"].append(float(output["continuation_logit"]))
            values["continuation_target"].append(float(transition.continuation))
    return {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}


def _fit_temperature(logit: np.ndarray, target: np.ndarray) -> float:
    logits = torch.tensor(logit, dtype=torch.float64)
    targets = torch.tensor(target, dtype=torch.float64)
    candidates = np.geomspace(0.25, 8.0, 121)
    losses = [
        float(F.binary_cross_entropy_with_logits(logits / float(value), targets))
        for value in candidates
    ]
    return float(candidates[int(np.argmin(losses))])


def _variance_scale(error: np.ndarray, variance: np.ndarray) -> float:
    ratio = float(np.mean(np.square(error)) / max(float(np.mean(variance)), 1e-12))
    return min(100.0, max(1e-3, ratio))


def input_ood_score(graph, calibration: ShadowCalibration) -> float:
    value = state_vector(graph).detach().cpu().numpy().astype(np.float64)
    mean = np.asarray(calibration.input_mean)
    std = np.asarray(calibration.input_std)
    return float(np.mean(np.abs((value - mean) / std)))


def prediction_uncertainty(output: dict[str, torch.Tensor], calibration: ShadowCalibration) -> float:
    state = float((output["state_logvar"].exp() * calibration.state_variance_scale).mean().sqrt())
    reward = float((output["reward_logvar"].exp() * calibration.reward_variance_scale).sqrt())
    cost = float((output["cost_logvar"].exp() * calibration.cost_variance_scale).mean().sqrt())
    return state + 0.25 * reward + 0.10 * cost


@torch.no_grad()
def fit_shadow_calibration(
    model: EventAwareGraphWorldModel,
    train_episodes: list[list[TensorTransition]],
    validation_episodes: list[list[TensorTransition]],
) -> ShadowCalibration:
    train_states = np.stack(
        [state_vector(item.graph).numpy() for episode in train_episodes for item in episode]
    ).astype(np.float64)
    mean = train_states.mean(axis=0)
    std = train_states.std(axis=0)
    std = np.maximum(std, 1e-3)
    predictions = _collect_predictions(model, _episodes(validation_episodes))
    provisional = ShadowCalibration(
        format_version="gppo-shadow-calibration/0.1.0",
        source_split="validation (temperatures/risk thresholds) + train (input moments)",
        source_transition_count=sum(len(episode) for episode in validation_episodes),
        state_change_temperature=_fit_temperature(
            predictions["state_logit"], predictions["state_target"]
        ),
        continuation_temperature=_fit_temperature(
            predictions["continuation_logit"], predictions["continuation_target"]
        ),
        state_variance_scale=_variance_scale(
            predictions["state_error"], predictions["state_variance"]
        ),
        reward_variance_scale=_variance_scale(
            predictions["reward_error"], predictions["reward_variance"]
        ),
        cost_variance_scale=_variance_scale(
            predictions["cost_error"], predictions["cost_variance"]
        ),
        input_mean=tuple(mean.tolist()),
        input_std=tuple(std.tolist()),
        ood_score_threshold=0.0,
        uncertainty_threshold=0.0,
    )
    ood_scores = [
        input_ood_score(item.graph, provisional)
        for episode in validation_episodes
        for item in episode
    ]
    uncertainties = []
    model.eval()
    for episode in validation_episodes:
        hidden = None
        for item in episode:
            output, hidden = model.step(
                item.graph, item.action, hidden, sample=False, evidence=item.evidence
            )
            uncertainties.append(prediction_uncertainty(output, provisional))
    return ShadowCalibration(
        **{
            **provisional.to_dict(),
            "ood_score_threshold": float(np.quantile(ood_scores, 0.995)),
            "uncertainty_threshold": float(np.quantile(uncertainties, 0.995)),
        }
    )


def expected_calibration_error(target: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(target), 1)
    result = 0.0
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        )
        if selected.any():
            result += selected.sum() / total * abs(float(probability[selected].mean() - target[selected].mean()))
    return float(result)


def _binary_calibration(target, logit, temperature: float) -> dict[str, float]:
    raw = 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))
    calibrated = 1.0 / (1.0 + np.exp(-np.clip(logit / temperature, -40.0, 40.0)))
    return {
        "raw_ece": expected_calibration_error(target, raw),
        "calibrated_ece": expected_calibration_error(target, calibrated),
        "raw_brier": float(np.mean(np.square(raw - target))),
        "calibrated_brier": float(np.mean(np.square(calibrated - target))),
    }


@torch.no_grad()
def evaluate_shadow_calibration(model, episodes, calibration: ShadowCalibration) -> dict[str, Any]:
    predictions = _collect_predictions(model, _episodes(episodes))
    state = _binary_calibration(
        predictions["state_target"],
        predictions["state_logit"],
        calibration.state_change_temperature,
    )
    continuation = _binary_calibration(
        predictions["continuation_target"],
        predictions["continuation_logit"],
        calibration.continuation_temperature,
    )
    errors = []
    risks = []
    profile_errors: dict[str, list[float]] = {}
    model.eval()
    for episode in episodes:
        hidden = None
        for item in episode:
            output, hidden = model.step(
                item.graph, item.action, hidden, sample=False, evidence=item.evidence
            )
            error = float((item.target_delta - output["state_delta_soft"]).abs().mean())
            risk = prediction_uncertainty(output, calibration)
            errors.append(error)
            risks.append(risk)
            profile_errors.setdefault(item.scenario_id, []).append(error)
    order = np.argsort(np.asarray(risks))
    sorted_error = np.asarray(errors)[order]
    coverages = (0.25, 0.50, 0.75, 1.0)
    risk_coverage = {
        str(coverage): float(sorted_error[: max(1, int(len(sorted_error) * coverage))].mean())
        for coverage in coverages
    }
    return {
        "state_change": state,
        "continuation": continuation,
        "risk_coverage_state_mae": risk_coverage,
        "low_risk_half_beats_full": risk_coverage["0.5"] < risk_coverage["1.0"],
        "profile_state_mae": {
            name: {"mean": float(np.mean(value)), "count": len(value)}
            for name, value in sorted(profile_errors.items())
        },
        "transition_count": len(errors),
    }
