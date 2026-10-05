# Action-Conditioned Task Outcome Label Availability Report

## Decision

The current problem is a production data-contract and collection-horizon
failure, not evidence that a correctly supervised model failed to distinguish
actions. The sealed data cannot decide the latter.

## Scope audit

The export has 56 complete windows: 24 train windows, 8 model-selection
windows, and 24 prediction-confirmation windows. The latter are 8 parents with
3 repeats each. Candidate rows are 333, 108, and 330 respectively. They must
not be pooled as one evaluation set.

In prediction confirmation, event dimensions 3 and 4
(`physical_completion_observed` and `host_confirmation_observed`) have zero
valid labels. Every candidate branch has `counts.completed=0` and
`counts.expired=0`, no terminal or truncated flag, no completion record, and no
completion communication message. Task states are limited to assigned,
pending, and unreleased.

## Root cause

The production collector calls `branch.step` once after each candidate and then
writes the window. Its `_event_target` passes `physical_completion=None` and
`host_confirmation=None` to the public transition contract. The existing
one-step record cannot observe later physical service completion or expiry.
The frozen default environment mode is `continuous_service_until_deadline`,
where physical completion exists in `TaskLifecycle.completed_at`; the
`completion_records` and completion-notice host confirmation path belong to a
different `arrival_to_region` mode. Merely extending the branch would not make
host confirmation available in the unchanged mode. The absence of a record is
not a negative label.

## Consequence

The saved event Brier result concerns public continuation-related events. It
cannot establish action-conditioned completion, deadline, expiry, or host
confirmation prediction. No offline reconstruction is legally possible from
the sealed export.

The prior mechanism replay shows G2 selected NOOP in 19/24 prediction windows.
All 19 of those NOOP choices have zero regret against the saved one-step
candidate utilities. The 16/24 Top-1 count is an action-identity result under
the frozen tie rule, not evidence that three NOOP choices lost utility. This
does not establish terminal task benefit, which was not evaluated.

The next step is first a static and bottom-boundary fixture check of whether
the unchanged mode can expose a valid host confirmation target at all. Only if
that check passes and independent parent identities are frozen should a
separately authorized label-only collection be proposed. Until then, do not
change the world-model loss, train GPPO, or request a task comparison.
