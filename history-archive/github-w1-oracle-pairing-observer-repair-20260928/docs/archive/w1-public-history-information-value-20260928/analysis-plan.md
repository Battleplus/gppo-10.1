# Analysis plan: public-history information value

Date: 2026-09-28

This is a post-hoc development analysis of the repaired W1-light run. It reads the frozen 56 evaluation episodes (eight development parents by seven methods), the frozen public controller, and the already completed mechanism audit. It does not construct an environment, import a model entry point, load a checkpoint, run inference, or train.

## Question and boundary

The question is whether legal public history adds verifiable state information beyond the complete current public observation, its `known`/`valid`/`age` fields, and the existing transparent search corrections. This analysis can assess state-estimation information and label availability. It cannot establish counterfactual action value, candidate ranking, task utility, or a world-model benefit.

Only repaired-runtime W1-light development records are used. The older defective runtime, held-out/test results, and oracle branches are excluded from estimation. The completed first-window oracle is referenced only as a fixed historical limit.

## Time and input contract

For a decision at time `t`, estimator inputs are restricted to public observations recorded at times `<= t`, including their public entity identities, field values, `known`, `valid`, measurement age, and public `continuation_actions`. The action and feedback produced after the decision at `t`, its communication delta, future observations, hidden event schedule, and environment state are excluded from inputs.

Public history is replayed in saved episode order. A field sample has measurement time `observation.time - field.age`. Repeated copies of the same measurement time are deduplicated by entity and field. Unknown and invalid values are never converted to numeric zero.

## Labels

### Current position and energy

For target time `t`, a label is valid only when a current or later saved public observation in the same episode contains a known, valid measurement whose original `measured_at` equals `t`. The value is an actual telemetry measurement made at the target time. A later measurement with `measured_at > t` is not used to label the past. Values inferred from motion equations are never labels.

Position requires both x and y labels at `t`. Energy requires the energy label at `t`. The first saved observation that exposes each exact-time measurement supplies its receipt time. Final states never subsequently exposed in a saved public observation remain unavailable.

### Remaining travel time

At a decision with one public ACK-known continuation for a UAV, the label is `physical_arrival_time - t` only when the saved completion record identifies the same task and UAV and arrival is after `t`. Missing completion, a different completing UAV, and termination before a usable record are censored and excluded, with counts reported.

### Executed action result

For each of the 367 saved non-NOOP choices, acceptance is the saved post-step feedback (`accepted`, `command_lost`, or `resource_unavailable`). Eventual completion is attributed only when an accepted action's selected task has a saved physical completion record with the same UAV. This is label-availability accounting, not a fitted acceptance or completion predictor.

## Frozen estimators

No parameter is fitted or selected from results. Constants come from the frozen environment contract: speed `1.0`, idle power `0.05`, and travel power `0.35`.

### A. Current observation

- Position: latest known, valid public `(x, y)` used at the decision.
- Energy: latest known, valid public energy used at the decision.
- Remaining time: Euclidean distance from the public position to the public continuation target divided by speed.

### B. Existing search estimate

- Position: no separate current-position correction exists, so the estimate equals A and is reported as such.
- Energy: `max(0, public_energy - 0.05 * energy_age)`.
- Remaining time: `public_distance / speed + max(x_age, y_age)`, matching the frozen search planner's conservative active-travel calculation.

### C. Public-history integration

C emits only when x/y share a measurement time, the current continuation target and its position are public and valid, and the same UAV/task continuation is visible in every saved public observation strictly after the position measurement and through `t`. The saved cadence must have no gap larger than the frozen one-second decision interval. This treats the continuation as publicly confirmed; a submitted command alone is insufficient.

- Position: move the last public position toward the public target at speed `1.0` for the continuously confirmed elapsed duration, capped at the target.
- Energy: subtract travel power `0.35` for the continuously confirmed elapsed duration. The estimate is emitted only when the energy measurement time is no earlier than the start of the verified continuation interval; otherwise C abstains.
- Remaining time: distance from C's projected position to the public target divided by speed.

C never writes estimates back to observations and never changes age, masks, ACK, leases, versions, or execution eligibility. It produces point estimates, not intervals; interval coverage and width are therefore reported as not applicable.

## Fair comparisons and slices

For every target, A and B are evaluated on their full valid-label set. The incremental A/B/C comparison uses the exact same rows on which C emits and a valid label exists. Metrics are MAE and RMSE in native units: distance units for position, energy units for energy, and seconds for remaining travel.

Results are reported overall and grouped by parent, controller, field-age bin (`0`, `(0,1]`, `(1,2]`, `>2` seconds), and presence of a public confirmed continuation. Controller groups describe different visited states and are not causal method comparisons. Multiple rows from one parent are not treated as independent parents.

Coverage reports total decisions, candidate entity-time labels, valid labels, C outputs, refusals, eight-parent coverage, and receipt-delay selection. Exact-time labels preferentially exist when telemetry is eventually delivered, so conclusions do not extend to permanently lost or disconnected states.

## Decision rule

The main judgment is qualitative and fixed before computation:

- `verifiable_increment`: C has meaningful coverage across multiple parents and consistently lowers error versus both A and B on the common rows;
- `transparent_sufficient`: B matches or beats C, so these records do not show a need for learned history;
- `no_observed_increment`: C has adequate coverage but does not improve the common-row errors;
- `insufficient_labels`: labels or C-eligible coverage are too sparse to distinguish the above.

No new numerical pass threshold is invented. Any positive result supports at most discussion of a separately authorized small validation. The historical utility margin `0.01`, CPU mean `10 ms`, and wall p95 `50 ms` remain future system criteria and are not applied to state error.

## Failure handling

Malformed identity, reversed time, future measurement in estimator input, nonfinite numeric data, duplicate episode-step identity, estimator/label row mismatch, or any imported environment/model/checkpoint/training entry causes the offline script to fail. Missing labels and C preconditions are counted and preserved rather than imputed.
