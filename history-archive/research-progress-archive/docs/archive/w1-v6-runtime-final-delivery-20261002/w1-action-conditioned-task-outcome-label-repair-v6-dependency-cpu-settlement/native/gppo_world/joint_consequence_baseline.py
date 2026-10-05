"""Simple public-information joint opportunity-cost baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .graph5 import GRAPH5_ACTION_COUNT, GRAPH5_CANDIDATE_COUNT, Graph5Snapshot


@dataclass(frozen=True)
class PublicJointScoreConfig:
    horizon_steps: int = 6
    decision_interval: float = 1.0
    speed: float = 1.0
    energy_per_distance: float = 0.35
    own_success_weight: float = 100.0
    opportunity_cost_weight: float = 20.0
    lateness_weight: float = 0.2
    energy_weight: float = 0.1


def _field(row, index: int) -> tuple[float, bool]:
    base = index * 4
    return float(row[base]), bool(float(row[base + 2]) > 0.5)


def _public_task(snapshot: Graph5Snapshot, task_index: int) -> dict[str, Any]:
    row = snapshot.nodes["task"][task_index]
    x, x_ok = _field(row, 0)
    y, y_ok = _field(row, 1)
    deadline, deadline_ok = _field(row, 2)
    priority, priority_ok = _field(row, 4)
    pending, pending_ok = _field(row, 5)
    return {"x": x, "y": y, "deadline": deadline, "priority": priority, "pending": pending > 0.5, "known": all((x_ok, y_ok, deadline_ok, priority_ok, pending_ok))}


def _public_uav(snapshot: Graph5Snapshot, uav_index: int) -> dict[str, Any]:
    row = snapshot.nodes["uav"][uav_index]
    x, x_ok = _field(row, 0)
    y, y_ok = _field(row, 1)
    energy, energy_ok = _field(row, 2)
    alive, alive_ok = _field(row, 3)
    connected, connected_ok = _field(row, 4)
    idle, idle_ok = _field(row, 5)
    return {"x": x, "y": y, "energy": energy, "available": all((x_ok, y_ok, energy_ok, alive_ok, connected_ok, idle_ok)) and energy > 0 and alive > 0.5 and connected > 0.5 and idle > 0.5}


def _candidate(snapshot: Graph5Snapshot, action: int) -> dict[str, Any]:
    if action == GRAPH5_ACTION_COUNT - 1:
        return {"legal": bool(snapshot.action_mask[action]), "noop": True}
    if not 0 <= action < GRAPH5_CANDIDATE_COUNT:
        raise ValueError("action outside Graph-5/25-action contract")
    uav_index, task_index = divmod(action, 6)
    relation = snapshot.candidate_features[action]
    distance = float(relation[0]) * 10.0
    return {"legal": bool(snapshot.action_mask[action]), "noop": False, "uav_index": uav_index, "task_index": task_index, "distance": distance, "visible": float(relation[1]) > 0.5}


def public_joint_action_score(snapshot: Graph5Snapshot, action: int, config: PublicJointScoreConfig | None = None) -> dict[str, float | int | bool]:
    """Score one action using only public graph features.

    The opportunity term counts other public pending tasks that become
    unreachable by the remaining public UAVs when the selected UAV is
    occupied.  It is an auditable heuristic, not a simulator truth or Q-value.
    """
    cfg = config or PublicJointScoreConfig()
    candidate = _candidate(snapshot, int(action))
    if not candidate["legal"]:
        raise ValueError("cannot score an illegal action")
    if candidate["noop"]:
        return {"action": int(action), "score": 0.0, "opportunity_cost": 0, "own_on_time_public": False, "estimated_arrival": 0.0, "energy_cost": 0.0}
    now = float(snapshot.global_features[0]) * 18.0
    task = _public_task(snapshot, int(candidate["task_index"]))
    uav = _public_uav(snapshot, int(candidate["uav_index"]))
    distance = float(candidate["distance"])
    estimated_arrival = now + distance / cfg.speed
    own_on_time = bool(task["known"] and uav["available"] and estimated_arrival <= float(task["deadline"]))
    opportunity_cost = 0
    for other_index in range(6):
        if other_index == int(candidate["task_index"]):
            continue
        other = _public_task(snapshot, other_index)
        if not other["known"] or not other["pending"]:
            continue
        reachable_all = []
        reachable_without_selected = []
        for uav_index in range(4):
            other_uav = _public_uav(snapshot, uav_index)
            relation = snapshot.candidate_features[uav_index * 6 + other_index]
            if not bool(snapshot.action_mask[uav_index * 6 + other_index]) or not other_uav["available"] or float(relation[1]) <= 0.5:
                continue
            arrival = now + float(relation[0]) * 10.0 / cfg.speed
            reachable_all.append(arrival <= float(other["deadline"]))
            if uav_index != int(candidate["uav_index"]):
                reachable_without_selected.append(arrival <= float(other["deadline"]))
        if any(reachable_all) and not any(reachable_without_selected):
            opportunity_cost += 1
    slack = float(task["deadline"]) - estimated_arrival if task["known"] else -float("inf")
    energy_cost = distance * cfg.energy_per_distance
    score = (cfg.own_success_weight if own_on_time else 0.0) - cfg.opportunity_cost_weight * opportunity_cost + cfg.lateness_weight * max(-slack, 0.0) * -1.0 - cfg.energy_weight * energy_cost
    return {"action": int(action), "score": float(score), "opportunity_cost": int(opportunity_cost), "own_on_time_public": own_on_time, "estimated_arrival": float(estimated_arrival), "energy_cost": float(energy_cost)}


def select_public_joint_action(snapshot: Graph5Snapshot, config: PublicJointScoreConfig | None = None) -> int:
    cfg = config or PublicJointScoreConfig()
    legal = [action for action, allowed in enumerate(snapshot.action_mask.tolist()) if allowed]
    if not legal:
        raise ValueError("public snapshot has no legal action")
    scored = [public_joint_action_score(snapshot, action, cfg) for action in legal]
    return int(max(scored, key=lambda item: (float(item["score"]), -int(item["action"])))["action"])

