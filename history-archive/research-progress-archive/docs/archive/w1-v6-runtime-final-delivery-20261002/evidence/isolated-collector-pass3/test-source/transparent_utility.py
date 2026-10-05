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
