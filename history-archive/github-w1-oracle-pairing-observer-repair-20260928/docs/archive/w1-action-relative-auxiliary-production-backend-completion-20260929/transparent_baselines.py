"""Two frozen, public-only candidate consequence estimates."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


TASK_WEIGHT = 0.8 * 0.5 / 6.0
ENERGY_WEIGHT = 0.2 / 36.0
GAMMA = 0.99


def _field(row: Sequence[float], index: int, now: float) -> dict:
    value, known, valid, age = map(float, row[index * 4:index * 4 + 4])
    return {
        "value": value,
        "known": known > 0.5,
        "valid": valid > 0.5,
        "age": age,
        "measured_at": now - age,
    }


def _score(distance: float, deadline: float, now: float, energy: float) -> float:
    arrival = now + distance
    on_time = arrival <= deadline and 0.35 * distance <= energy
    discount = GAMMA ** max(0, math.ceil((arrival if on_time else deadline) - now) - 1)
    task = TASK_WEIGHT * discount * (1.0 if on_time else -1.0)
    energy_term = ENERGY_WEIGHT * 0.35 * distance
    return task - energy_term


def current_public_scores(observation: Mapping, candidates: Sequence[int]) -> dict[int, float]:
    now = float(observation["time"])
    result: dict[int, float] = {}
    for action in candidates:
        if action == 24:
            result[action] = 0.0
            continue
        uav, task = divmod(action, 6)
        ux, uy, energy = (_field(observation["uavs"][uav], index, now) for index in (0, 1, 2))
        tx, ty, deadline = (_field(observation["tasks"][task], index, now) for index in (0, 1, 2))
        values = (ux, uy, energy, tx, ty, deadline)
        if not all(value["known"] and value["valid"] for value in values):
            raise ValueError("legal candidate lacks current public baseline fields")
        distance = math.dist((ux["value"], uy["value"]), (tx["value"], ty["value"]))
        result[action] = _score(distance, deadline["value"], now, energy["value"])
    return result


def transparent_history_scores(observation: Mapping, memory, candidates: Sequence[int]) -> dict[int, float]:
    now = float(observation["time"])
    ids = observation["public_entity_ids"]
    result: dict[int, float] = {}
    for action in candidates:
        if action == 24:
            result[action] = 0.0
            continue
        uav_slot, task_slot = divmod(action, 6)
        uav = ids["uavs"][uav_slot]
        task = ids["tasks"][task_slot]
        ux = memory.get("uavs", uav, "x")
        uy = memory.get("uavs", uav, "y")
        energy = memory.get("uavs", uav, "energy")
        tx = memory.get("tasks", task, "x")
        ty = memory.get("tasks", task, "y")
        deadline = memory.get("tasks", task, "deadline")
        values = (ux, uy, energy, tx, ty, deadline)
        if not all(value is not None and value["valid"] for value in values):
            raise ValueError("legal candidate lacks transparent-history fields")
        position_age = max(float(ux["age"]), float(uy["age"]))
        distance = math.dist((ux["value"], uy["value"]), (tx["value"], ty["value"])) + position_age
        conservative_energy = max(0.0, float(energy["value"]) - 0.05 * float(energy["age"]))
        result[action] = _score(distance, float(deadline["value"]), now, conservative_energy)
    return result


def qualifies(observation: Mapping, memory, *, minimum_step: int, decision_step: int) -> tuple[bool, dict[int, float]]:
    candidates = tuple(int(value) for value in memory.candidates(observation))
    nonnoop = tuple(value for value in candidates if value != 24)
    scores = transparent_history_scores(observation, memory, candidates)
    distinct = len({round(scores[action], 12) for action in nonnoop}) >= 2
    return decision_step >= minimum_step and len(nonnoop) >= 2 and distinct, scores

