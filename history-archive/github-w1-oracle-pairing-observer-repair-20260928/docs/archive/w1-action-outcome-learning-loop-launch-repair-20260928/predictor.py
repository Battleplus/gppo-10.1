"""Model-independent candidate outcome predictor interface and masked losses."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class OutcomeHeadSpec:
    binary_heads: tuple[str, ...] = (
        "command_accepted",
        "target_physical_on_time",
        "target_completed_by_current_uav",
    )
    categorical_heads: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: {
        "command_state": (
            "accepted",
            "accepted_then_ack_lost",
            "command_transport_lost",
            "executor_rejected",
            "resource_unavailable",
            "noop",
            "unknown",
        )
    })
    continuous_heads: tuple[str, ...] = (
        "energy_to_terminal",
        "remaining_utility",
    )
    censored_heads: tuple[str, ...] = ("target_completion_time",)
    units: Mapping[str, str] = field(default_factory=lambda: {
        "energy_to_terminal": "native_energy",
        "remaining_utility": "frozen_discounted_utility",
        "target_completion_time": "seconds_from_decision",
    })
    normalization: Mapping[str, str] = field(default_factory=lambda: {
        "energy_to_terminal": "divide_by_36_frozen_fleet_initial_energy",
        "remaining_utility": "none_pending_training_evidence",
        "target_completion_time": "divide_by_18_frozen_horizon",
    })
    loss_weights: Mapping[str, str] = field(default_factory=lambda: {
        "all_heads": "UNSET_REQUIRES_PREFROZEN_EXPERIMENT_PARAMETER",
    })


@dataclass(frozen=True)
class CandidatePrediction:
    action: int
    continuation_id: str
    command_state_probabilities: Mapping[str, float]
    target_physical_on_time_probability: float | None
    target_completed_by_current_uav_probability: float | None
    energy_to_terminal: float | None
    remaining_utility: float | None
    target_completion_time: float | None
    valid: Mapping[str, bool]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.action < 25:
            raise ValueError("action out of range")
        if not self.continuation_id:
            raise ValueError("continuation_id is required")
        total = sum(float(value) for value in self.command_state_probabilities.values())
        if self.command_state_probabilities and not math.isclose(total, 1.0, abs_tol=1e-8):
            raise ValueError("command probabilities must sum to one")
        for value in self.command_state_probabilities.values():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("invalid probability")
        for name in ("target_physical_on_time_probability", "target_completed_by_current_uav_probability"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"invalid {name}")
        for name in ("energy_to_terminal", "remaining_utility", "target_completion_time"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"nonfinite {name}")
        field_values = {
            "command_state": self.command_state_probabilities or None,
            "target_physical_on_time": self.target_physical_on_time_probability,
            "target_completed_by_current_uav": self.target_completed_by_current_uav_probability,
            "energy_to_terminal": self.energy_to_terminal,
            "remaining_utility": self.remaining_utility,
            "target_completion_time": self.target_completion_time,
        }
        for name, value in field_values.items():
            valid = bool(self.valid.get(name, False))
            if valid != (value is not None):
                raise ValueError(f"{name} value and valid mask disagree")
        if self.action == 24 and any((
            self.target_physical_on_time_probability is not None,
            self.target_completed_by_current_uav_probability is not None,
            self.target_completion_time is not None,
        )):
            raise ValueError("NOOP target outputs must be masked")


class CandidateOutcomePredictor(Protocol):
    continuation_id: str

    def predict(self, public_input: Mapping[str, Any], candidate_actions: Sequence[int]) -> list[CandidatePrediction]:
        """Return one structured prediction per candidate without changing public state."""


class FakeOutcomePredictor:
    """Deterministic test double; it is not a trained or deployable model."""

    def __init__(self, continuation_id: str, utilities: Mapping[int, float]) -> None:
        self.continuation_id = continuation_id
        self.utilities = dict(utilities)
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def predict(self, public_input: Mapping[str, Any], candidate_actions: Sequence[int]) -> list[CandidatePrediction]:
        actions = tuple(int(value) for value in candidate_actions)
        self.calls.append((str(public_input.get("decision_id", "unknown")), actions))
        result = []
        for action in actions:
            is_noop = action == 24
            result.append(CandidatePrediction(
                action=action,
                continuation_id=self.continuation_id,
                command_state_probabilities={"noop": 1.0} if is_noop else {"accepted": 1.0},
                target_physical_on_time_probability=None if is_noop else 0.5,
                target_completed_by_current_uav_probability=None if is_noop else 0.5,
                energy_to_terminal=0.0,
                remaining_utility=float(self.utilities[action]),
                target_completion_time=None,
                valid={
                    "command_state": True,
                    "target_physical_on_time": not is_noop,
                    "target_completed_by_current_uav": not is_noop,
                    "energy_to_terminal": True,
                    "remaining_utility": True,
                    "target_completion_time": False,
                },
                diagnostics={"provider": "fake", "not_accuracy_evidence": True},
            ))
        return result


def masked_binary_cross_entropy(probabilities: Sequence[float], labels: Sequence[float], mask: Sequence[bool]) -> float | None:
    values = []
    for probability, label, valid in zip(probabilities, labels, mask, strict=True):
        if not valid:
            continue
        probability = min(max(float(probability), 1e-12), 1.0 - 1e-12)
        label = float(label)
        values.append(-(label * math.log(probability) + (1.0 - label) * math.log(1.0 - probability)))
    return None if not values else sum(values) / len(values)


def masked_smooth_l1(predictions: Sequence[float], labels: Sequence[float], mask: Sequence[bool], beta: float = 1.0) -> float | None:
    if beta <= 0:
        raise ValueError("beta must be positive")
    values = []
    for prediction, label, valid in zip(predictions, labels, mask, strict=True):
        if not valid:
            continue
        error = abs(float(prediction) - float(label))
        if not math.isfinite(error):
            raise ValueError("continuous loss received nonfinite input")
        values.append(0.5 * error * error / beta if error < beta else error - 0.5 * beta)
    return None if not values else sum(values) / len(values)
