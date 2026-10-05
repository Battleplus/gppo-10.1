"""Pure normalization for the frozen transparent scorer's vector components."""
from __future__ import annotations

import math
from typing import Any, Mapping


class TransparentUtilityError(ValueError):
    pass


def transparent_utility_components(
    diagnostic: Mapping[str, Any], *, task_capacity: int, initial_total_energy: float,
) -> tuple[float, float]:
    if task_capacity <= 0 or not math.isfinite(initial_total_energy) or initial_total_energy <= 0:
        raise TransparentUtilityError("TRANSPARENT_UTILITY_NORMALIZATION_INVALID")
    score = float(diagnostic.get("score", float("nan")))
    energy_cost = float(diagnostic.get("energy_cost", float("nan")))
    if not math.isfinite(score) or not math.isfinite(energy_cost) or energy_cost < 0:
        raise TransparentUtilityError("TRANSPARENT_UTILITY_DIAGNOSTIC_INVALID")
    task_component = (score + 0.1 * energy_cost) / (100.0 * task_capacity)
    energy_component = -energy_cost / initial_total_energy
    if not math.isfinite(task_component) or not math.isfinite(energy_component):
        raise TransparentUtilityError("TRANSPARENT_UTILITY_COMPONENT_NONFINITE")
    return task_component, energy_component


def transparent_horizon_components(
    diagnostic: Mapping[str, Any], *, task_capacity: int, initial_total_energy: float,
    ideal_continuation_completions: int,
) -> tuple[float, float]:
    """Extend the frozen public first-action score with an explicit ideal surrogate.

    The continuation term assumes the requested number of additional public,
    pending tasks all complete on time with zero additional energy cost. It is
    an optimistic public-information reference, not a simulator prediction.
    """
    if type(ideal_continuation_completions) is not int or not 0 <= ideal_continuation_completions <= task_capacity:
        raise TransparentUtilityError("TRANSPARENT_IDEAL_CONTINUATION_COUNT_INVALID")
    task_component, energy_component = transparent_utility_components(
        diagnostic, task_capacity=task_capacity,
        initial_total_energy=initial_total_energy,
    )
    task_component += ideal_continuation_completions / float(task_capacity)
    if not math.isfinite(task_component):
        raise TransparentUtilityError("TRANSPARENT_UTILITY_COMPONENT_NONFINITE")
    return task_component, energy_component
