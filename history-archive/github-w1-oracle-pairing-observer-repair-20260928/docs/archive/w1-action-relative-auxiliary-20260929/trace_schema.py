"""Auditable per-candidate prediction output contract.

The dynamic runner must persist this table during the already-counted
prediction forwards.  It is intentionally JSONL and model-agnostic here so
preparation tests do not import a runtime, checkpoint, or torch.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from objective import CONTINUATION_ID, NOOP_ACTION


def stable_input_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PredictionTraceRow:
    parent: str
    repeat: int
    decision_id: str
    action: int
    legal_actions: tuple[int, ...]
    continuation_id: str
    public_input_sha256: str
    true_utility: float | None
    target_valid: bool
    baseline_current_public: float
    baseline_transparent: float
    seed_predictions: tuple[float, ...]
    ensemble_prediction: float
    variant: str = "A"
    seed_ids: tuple[int, ...] = (7101, 7102, 7103)

    def validate(self) -> None:
        if not self.parent or not self.decision_id:
            raise ValueError("trace identity is required")
        if type(self.repeat) is not int or self.repeat < 0:
            raise ValueError("invalid repeat")
        if not 0 <= self.action <= NOOP_ACTION or self.action not in self.legal_actions:
            raise ValueError("trace action is not legal")
        if not self.legal_actions or len(set(self.legal_actions)) != len(self.legal_actions):
            raise ValueError("invalid legal candidate set")
        if self.continuation_id != CONTINUATION_ID:
            raise ValueError("trace continuation mismatch")
        if self.variant not in {"A", "B"}:
            raise ValueError("trace variant must be A or B")
        if self.seed_ids != (7101, 7102, 7103):
            raise ValueError("trace seed identity/order mismatch")
        if len(self.public_input_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.public_input_sha256.lower()):
            raise ValueError("public input hash must be SHA-256")
        if self.target_valid:
            if self.true_utility is None or not math.isfinite(float(self.true_utility)):
                raise ValueError("valid trace target must be finite")
        elif self.true_utility is not None:
            raise ValueError("unknown target cannot carry an imputed value")
        for value in (self.baseline_current_public, self.baseline_transparent, self.ensemble_prediction):
            if not math.isfinite(float(value)):
                raise ValueError("trace score is nonfinite")
        if len(self.seed_predictions) != 3 or not all(math.isfinite(float(value)) for value in self.seed_predictions):
            raise ValueError("exactly three finite seed predictions are required")
        expected = sum(self.seed_predictions) / 3.0
        if not math.isclose(expected, self.ensemble_prediction, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("ensemble is not the arithmetic seed mean")


def group_trace(rows: Iterable[PredictionTraceRow]) -> dict[tuple[str, str], list[PredictionTraceRow]]:
    grouped: dict[tuple[str, str], list[PredictionTraceRow]] = {}
    for row in rows:
        row.validate()
        grouped.setdefault((row.variant, row.decision_id), []).append(row)
    for (_variant, decision_id), group in grouped.items():
        first = group[0]
        if any((row.parent, row.repeat, row.legal_actions, row.continuation_id, row.public_input_sha256) !=
               (first.parent, first.repeat, first.legal_actions, first.continuation_id, first.public_input_sha256)
               for row in group):
            raise ValueError(f"{decision_id}: shared window identity differs")
        if {row.action for row in group} != set(first.legal_actions):
            raise ValueError(f"{decision_id}: incomplete candidate trace")
    return grouped


def write_trace(path: Path, rows: Iterable[PredictionTraceRow]) -> None:
    ordered = list(rows)
    group_trace(ordered)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in ordered), encoding="utf-8")


def read_trace(path: Path) -> list[PredictionTraceRow]:
    rows: list[PredictionTraceRow] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            value["legal_actions"] = tuple(value["legal_actions"])
            value["seed_predictions"] = tuple(value["seed_predictions"])
            value["seed_ids"] = tuple(value.get("seed_ids", (7101, 7102, 7103)))
            row = PredictionTraceRow(**value)
            row.validate()
            rows.append(row)
        except Exception as exc:
            raise ValueError(f"{path}:{line_number}: invalid prediction trace: {exc}") from exc
    group_trace(rows)
    if not rows:
        raise ValueError("prediction trace is empty")
    return rows
