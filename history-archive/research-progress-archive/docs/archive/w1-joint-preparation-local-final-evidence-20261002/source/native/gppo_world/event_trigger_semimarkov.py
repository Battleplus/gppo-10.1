"""Decision-level semi-Markov helpers for the event-trigger-aware pilot.

This module is deliberately separate from the historical arrival runners.  A
triggered continuation is an environment step, but it is not a PPO action
sample.  A decision window therefore stores one discounted vector return and
one ``k``-step bootstrap discount for the interval between two real policy
decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class DecisionWindow:
    """One real policy decision and its completed semi-Markov interval."""

    reward_sum: np.ndarray
    k: int
    bootstrap_discount: float
    value: np.ndarray
    next_value: np.ndarray
    terminated: bool
    truncated: bool = False
    rollout_boundary: bool = False


def discounted_window(rewards: Sequence[Sequence[float]] | np.ndarray, gamma: float) -> tuple[np.ndarray, int, float]:
    """Return ``R_k``, ``k`` and ``gamma**k`` for one decision window."""
    rows = np.asarray(rewards, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 1:
        raise ValueError("a decision window needs at least one vector reward")
    if not np.isfinite(rows).all() or not (0.0 < gamma <= 1.0):
        raise ValueError("finite rewards and 0 < gamma <= 1 are required")
    discounts = np.power(float(gamma), np.arange(rows.shape[0], dtype=np.float64))[:, None]
    result = (rows * discounts).sum(axis=0)
    k = int(rows.shape[0])
    return result.astype(np.float32), k, float(gamma ** k)


def decision_level_gae(windows: Sequence[DecisionWindow], gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE over real decisions, using ``gamma**k`` at each edge.

    ``rollout_boundary`` cuts the recursive GAE chain but still permits the
    value bootstrap in the window's own delta.  True termination never
    bootstraps.  Truncation and a training-window boundary are not terminal.
    """
    if not windows:
        raise ValueError("empty decision window sequence")
    if not (0.0 < gamma <= 1.0 and 0.0 <= gae_lambda <= 1.0):
        raise ValueError("invalid discount parameters")
    rewards = np.stack([np.asarray(item.reward_sum, dtype=np.float64) for item in windows])
    values = np.stack([np.asarray(item.value, dtype=np.float64) for item in windows])
    next_values = np.stack([np.asarray(item.next_value, dtype=np.float64) for item in windows])
    discounts = np.asarray([float(item.bootstrap_discount) for item in windows], dtype=np.float64)
    terminated = np.asarray([bool(item.terminated) for item in windows], dtype=np.float64)
    boundary = np.asarray([
        bool(item.terminated or item.truncated or item.rollout_boundary) for item in windows
    ], dtype=np.float64)
    advantages = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1], dtype=np.float64)
    for index in range(len(windows) - 1, -1, -1):
        delta = rewards[index] + discounts[index] * next_values[index] * (1.0 - terminated[index]) - values[index]
        running = delta + discounts[index] * float(gae_lambda) * (1.0 - boundary[index]) * running
        advantages[index] = running
    return advantages.astype(np.float32), (advantages + values).astype(np.float32)


def validate_window_contract(windows: Iterable[DecisionWindow], *, gamma: float) -> dict[str, object]:
    rows = list(windows)
    if not rows:
        raise ValueError("no windows")
    for row in rows:
        if row.k < 1:
            raise AssertionError("k must be positive")
        if abs(float(row.bootstrap_discount) - gamma ** row.k) > 1e-7:
            raise AssertionError("bootstrap discount does not equal gamma**k")
        if row.terminated and np.any(np.asarray(row.next_value) != 0.0):
            raise AssertionError("terminal window must not bootstrap")
    return {
        "windows": len(rows),
        "k_values": [int(row.k) for row in rows],
        "terminal_windows": sum(bool(row.terminated) for row in rows),
        "truncated_windows": sum(bool(row.truncated) for row in rows),
        "rollout_boundary_windows": sum(bool(row.rollout_boundary) for row in rows),
    }

