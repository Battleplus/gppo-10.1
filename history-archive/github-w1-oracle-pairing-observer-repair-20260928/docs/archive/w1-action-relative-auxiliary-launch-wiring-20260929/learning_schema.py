"""Training-record contract for fixed-Hungarian action outcome learning."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FLAT_DIM = 770
HISTORY_DIM = 32
ACTION_DIM = 25
INPUT_DIM = FLAT_DIM + HISTORY_DIM + ACTION_DIM
CONTINUATION_ID = "hungarian-v1-fixed"
SPLIT_ROLES = ("train", "model_selection", "prediction_evaluation")


class LearningContractError(ValueError):
    pass


def _finite_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise LearningContractError(f"{name} must have length {length}")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise LearningContractError(f"{name} contains nonfinite values")
    return result


def history_features(public_history: Sequence[Mapping[str, Any]], now: float) -> tuple[float, ...]:
    """Summarize up to four prior public observations without future access.

    Each lag contributes: time gap, legal count, UAV known/valid counts,
    task known/valid counts, continuation count, and public event signal.
    Missing lags are zero padded. The current observation is not part of this
    function and post-action feedback cannot enter it.
    """

    rows: list[float] = []
    for observation in list(public_history)[-4:][::-1]:
        observed_at = float(observation["time"])
        if observed_at > float(now) + 1e-9:
            raise LearningContractError("public history contains a future observation")
        uavs = observation.get("uavs", [])
        tasks = observation.get("tasks", [])
        if len(uavs) != 4 or len(tasks) != 6:
            raise LearningContractError("history observation violates 4x6 contract")

        def counts(table: Sequence[Sequence[float]]) -> tuple[int, int]:
            known = valid = 0
            for row in table:
                if len(row) % 4:
                    raise LearningContractError("public field row is not value/known/valid/age")
                for offset in range(0, len(row), 4):
                    known += float(row[offset + 1]) > 0.5
                    valid += float(row[offset + 2]) > 0.5
            return known, valid

        u_known, u_valid = counts(uavs)
        t_known, t_valid = counts(tasks)
        rows.extend((
            float(now) - observed_at,
            float(sum(bool(item) for item in observation.get("mask", []))),
            float(u_known),
            float(u_valid),
            float(t_known),
            float(t_valid),
            float(len(observation.get("continuation_actions", []))),
            float(observation.get("event_signal", 0.0)),
        ))
    rows.extend([0.0] * (HISTORY_DIM - len(rows)))
    return _finite_vector(rows, HISTORY_DIM, "history_features")


def action_one_hot(action: int) -> tuple[float, ...]:
    if type(action) is not int or not 0 <= action < ACTION_DIM:
        raise LearningContractError("action must be an integer in [0,24]")
    result = [0.0] * ACTION_DIM
    result[action] = 1.0
    return tuple(result)


def materialize_features(flat: Sequence[float], history: Sequence[float], action: int) -> tuple[float, ...]:
    result = (
        _finite_vector(flat, FLAT_DIM, "flat")
        + _finite_vector(history, HISTORY_DIM, "history")
        + action_one_hot(action)
    )
    if len(result) != INPUT_DIM:
        raise AssertionError("feature dimension drift")
    return result


@dataclass(frozen=True)
class LearningRecord:
    sample_id: str
    parent: str
    repeat: int
    decision_id: str
    split_role: str
    action: int
    legal_actions: tuple[int, ...]
    flat: tuple[float, ...]
    history: tuple[float, ...]
    remaining_utility: float
    current_public_score: float
    transparent_history_score: float
    continuation_id: str = CONTINUATION_ID
    is_true_branch: bool = True
    training_eligible: bool = True

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "LearningRecord":
        for name in ("is_true_branch", "training_eligible"):
            if type(row.get(name)) is not bool:
                raise LearningContractError(f"{name} must be a JSON boolean")
        record = cls(
            sample_id=str(row["sample_id"]),
            parent=str(row["parent"]),
            repeat=int(row["repeat"]),
            decision_id=str(row["decision_id"]),
            split_role=str(row["split_role"]),
            action=int(row["action"]),
            legal_actions=tuple(int(value) for value in row["legal_actions"]),
            flat=_finite_vector(row["flat"], FLAT_DIM, "flat"),
            history=_finite_vector(row["history"], HISTORY_DIM, "history"),
            remaining_utility=float(row["remaining_utility"]),
            current_public_score=float(row["current_public_score"]),
            transparent_history_score=float(row["transparent_history_score"]),
            continuation_id=str(row["continuation_id"]),
            is_true_branch=row["is_true_branch"],
            training_eligible=row["training_eligible"],
        )
        record.validate()
        return record

    def validate(self) -> None:
        if not all((self.sample_id, self.parent, self.decision_id)):
            raise LearningContractError("record identities cannot be empty")
        if self.split_role not in SPLIT_ROLES:
            raise LearningContractError("invalid split role")
        if self.continuation_id != CONTINUATION_ID:
            raise LearningContractError("training labels require fixed Hungarian continuation")
        if not self.is_true_branch or not self.training_eligible:
            raise LearningContractError("factual or diagnostic records cannot enter training")
        if self.action not in self.legal_actions or len(set(self.legal_actions)) != len(self.legal_actions):
            raise LearningContractError("candidate legality is inconsistent")
        if not self.legal_actions or any(not 0 <= action < ACTION_DIM for action in self.legal_actions):
            raise LearningContractError("invalid legal candidate set")
        for value in (self.remaining_utility, self.current_public_score, self.transparent_history_score):
            if not math.isfinite(value):
                raise LearningContractError("target and baseline scores must be finite")
        materialize_features(self.flat, self.history, self.action)


def load_records(path: Path, *, expected_role: str | None = None) -> list[LearningRecord]:
    records: list[LearningRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = LearningRecord.from_mapping(json.loads(line))
        except Exception as exc:
            raise LearningContractError(f"{path}:{line_number}: {exc}") from exc
        if expected_role is not None and record.split_role != expected_role:
            raise LearningContractError(f"{path}:{line_number}: split role mismatch")
        records.append(record)
    if not records:
        raise LearningContractError("empty learning dataset")
    assert_parent_disjoint(records)
    validate_decision_groups(records)
    return records


def assert_parent_disjoint(records: Iterable[LearningRecord]) -> None:
    roles: dict[str, set[str]] = {}
    for record in records:
        roles.setdefault(record.parent, set()).add(record.split_role)
    overlap = {parent: values for parent, values in roles.items() if len(values) != 1}
    if overlap:
        raise LearningContractError(f"parents cross split roles: {overlap}")


def validate_decision_groups(records: Iterable[LearningRecord]) -> None:
    groups: dict[str, list[LearningRecord]] = {}
    for record in records:
        groups.setdefault(record.decision_id, []).append(record)
    for decision_id, rows in groups.items():
        actions = {row.action for row in rows}
        expected = set(rows[0].legal_actions)
        if len(actions) != len(rows) or actions != expected:
            raise LearningContractError(f"{decision_id}: candidate label set is incomplete or duplicated")
        identity = (
            rows[0].parent,
            rows[0].repeat,
            rows[0].split_role,
            rows[0].flat,
            rows[0].history,
            rows[0].legal_actions,
        )
        if any(
            (row.parent, row.repeat, row.split_role, row.flat, row.history, row.legal_actions) != identity
            for row in rows[1:]
        ):
            raise LearningContractError(f"{decision_id}: candidate records do not share one public state")
