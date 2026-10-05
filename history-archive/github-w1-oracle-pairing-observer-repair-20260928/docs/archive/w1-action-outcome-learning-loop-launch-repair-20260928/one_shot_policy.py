"""One intervention followed by the frozen Hungarian continuation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


@dataclass
class OneShotOutcomeSelector:
    hungarian_choose: Callable[[Mapping, object], tuple[int, Mapping]]
    consequence_scores: Callable[[Mapping, object, Sequence[int]], Mapping[int, float]]
    mode: str
    used: bool = False
    prediction_calls: int = 0

    def choose(self, observation: Mapping, memory: object) -> tuple[int, dict]:
        candidates = tuple(int(value) for value in memory.candidates(observation))
        nonnoop = tuple(value for value in candidates if value != 24)
        if not self.used and len(nonnoop) >= 2:
            scores = dict(self.consequence_scores(observation, memory, candidates))
            self.prediction_calls += 1
            if set(scores) != set(candidates):
                raise ValueError("one-shot scores must cover the exact legal candidate set")
            selected = max(candidates, key=lambda action: (float(scores[action]), -action))
            self.used = True
            return selected, {
                "mode": self.mode,
                "phase": "one_shot_consequence_choice",
                "scores": scores,
                "continuation_id": "hungarian-v1-fixed",
            }
        action, diagnostic = self.hungarian_choose(observation, memory)
        return int(action), {
            "mode": self.mode,
            "phase": "hungarian_prefix" if not self.used else "hungarian_continuation",
            "hungarian": diagnostic,
            "continuation_id": "hungarian-v1-fixed",
        }

    def assert_complete(self) -> None:
        if self.prediction_calls > 1:
            raise RuntimeError("predictor was used more than once")

