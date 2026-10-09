"""Shared causal history encoder for collection, training, and task inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


HISTORY_DIM = 128
HISTORY_DECAY = 0.5
HISTORY_SCHEMA = "w1-causal-public-history/1.0.0"


def public_state_vector(observation: Mapping[str, Any]) -> np.ndarray:
    from public_prefix import make_public_prefix, serialize_public_prefix
    flat = serialize_public_prefix(make_public_prefix(dict(observation)))['flat']
    if flat.size < HISTORY_DIM or not np.isfinite(flat[:HISTORY_DIM]).all():
        raise ValueError("public observation must contain 128 finite state values")
    return flat[:HISTORY_DIM].copy()


@dataclass
class CausalPublicHistory:
    """A per-episode exponentially weighted summary with a fixed public schema."""

    decay: float = HISTORY_DECAY
    state: np.ndarray | None = None
    observations: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("history decay must be in [0,1)")
        if self.state is not None:
            value = np.asarray(self.state, dtype=np.float32).reshape(-1)
            if value.shape != (HISTORY_DIM,) or not np.isfinite(value).all():
                raise ValueError("history state must be 128 finite values")
            self.state = value.copy()

    def reset(self) -> None:
        self.state = None
        self.observations = 0

    def append(self, observation: Mapping[str, Any]) -> np.ndarray:
        current = public_state_vector(observation)
        self.state = current if self.state is None else self.decay * self.state + (1.0 - self.decay) * current
        self.observations += 1
        return self.state.copy()

    def attach_current(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        if self.state is None:
            raise ValueError("cannot attach an empty episode history")
        result = dict(observation)
        result["public_history"] = self.state.copy().tolist()
        result["public_history_schema"] = HISTORY_SCHEMA
        return result


def history_vector(observation: Mapping[str, Any]) -> list[float]:
    value = observation.get("public_history")
    if value is None or observation.get("public_history_schema") != HISTORY_SCHEMA:
        raise ValueError("observation lacks shared causal public history")
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (HISTORY_DIM,) or not np.isfinite(vector).all():
        raise ValueError("causal public history must be 128 finite values")
    return vector.tolist()


__all__ = ["CausalPublicHistory", "HISTORY_DECAY", "HISTORY_DIM", "HISTORY_SCHEMA",
           "history_vector", "public_state_vector"]
