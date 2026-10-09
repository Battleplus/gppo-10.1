"""Small, dependency-free contracts used by the S1-R acceptance harness.

The original S1 collector mixed episode termination with task completion and
used a second, different path for stale retries.  These contracts are kept
free of the training environment so the accounting rules can be regression
tested locally and reused by the server runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, MutableMapping, Optional, Set


@dataclass(frozen=True)
class SubmissionDecision:
    status: str
    accepted: bool
    executed_action: int
    charged_uid: Optional[Hashable]
    energy_before: float
    energy_after: float
    reason: Optional[str] = None


class EnergyLedger:
    """One execution constraint gate for initial submissions and retries."""

    def __init__(self, energy: Mapping[Hashable, float], *, cost: float = 1.0):
        self.energy: MutableMapping[Hashable, float] = dict(energy)
        self.cost = float(cost)
        self._accepted_submission_ids: Set[Hashable] = set()

    def submit(
        self,
        *,
        submission_id: Hashable,
        action: int,
        uid: Optional[Hashable],
        legal: bool,
        noop_action: int,
        stale: bool = False,
    ) -> SubmissionDecision:
        """Validate and charge exactly once for the action actually accepted.

        A stale or duplicate submission is rejected before charging.  A retry
        must call this method again with its own submission id and endpoint;
        no state from the old endpoint is reused.
        """
        before = float(self.energy.get(uid, 0.0)) if uid is not None else 0.0
        if stale:
            return SubmissionDecision("stale_rejected", False, noop_action, None, before, before, "stale")
        if submission_id in self._accepted_submission_ids:
            return SubmissionDecision("duplicate_rejected", False, noop_action, None, before, before, "duplicate")
        if action == noop_action:
            return SubmissionDecision("noop", True, noop_action, None, before, before)
        if not legal:
            return SubmissionDecision("illegal_rejected", False, noop_action, None, before, before, "illegal")
        if uid is None or before < self.cost:
            return SubmissionDecision("energy_rejected", False, noop_action, None, before, before, "energy")
        after = before - self.cost
        self.energy[uid] = after
        self._accepted_submission_ids.add(submission_id)
        return SubmissionDecision("accepted", True, action, uid, before, after)


@dataclass(frozen=True)
class EpisodeOutcome:
    episode_ended: bool
    task_completed: bool
    task_failed: bool
    infeasible: bool
    deadline_exceeded: bool
    end_reason: str


def classify_episode(
    *,
    terminated: bool,
    truncated: bool,
    task_completed: bool,
    infeasible: bool = False,
    deadline_exceeded: bool = False,
) -> EpisodeOutcome:
    """Classify an episode without treating ``terminated`` as completion."""
    ended = bool(terminated or truncated)
    completed = bool(task_completed and not infeasible and not deadline_exceeded)
    failed = bool(ended and not completed and not infeasible and not deadline_exceeded)
    if infeasible:
        reason = "infeasible"
    elif deadline_exceeded:
        reason = "deadline_timeout"
    elif truncated:
        reason = "timeout"
    elif terminated and completed:
        reason = "completed"
    elif terminated:
        reason = "terminated_incomplete"
    else:
        reason = "running"
    return EpisodeOutcome(ended, completed, failed, bool(infeasible), bool(deadline_exceeded), reason)


def count_reassignments(
    before: Mapping[Hashable, Optional[Hashable]],
    after: Mapping[Hashable, Optional[Hashable]],
    initial: Mapping[Hashable, Optional[Hashable]],
) -> int:
    """Count real allocation changes, excluding each region's first assignment."""
    count = 0
    for region in set(before) | set(after):
        old, new, first = before.get(region), after.get(region), initial.get(region)
        if old is not None and new is not None and old != new and old != first:
            count += 1
    return count


def low_confidence_confirmed(
    *, event_id: str, confidence: float, confirmed_ids: Set[str], occurred_at: float,
    observed_at: float, decision_time: float,
) -> bool:
    """Require detector confirmation and causal timing for a weak event."""
    return (
        confidence < 0.6
        and event_id in confirmed_ids
        and occurred_at <= observed_at <= decision_time
    )
