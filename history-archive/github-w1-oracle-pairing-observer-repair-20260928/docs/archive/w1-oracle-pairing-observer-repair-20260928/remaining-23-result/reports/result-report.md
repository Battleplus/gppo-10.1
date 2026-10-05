# W1 action-consequence oracle ceiling: complete 24-unit result

Date: 2026-09-28

This is a post-run report for the one-shot attempt `w1-action-consequence-oracle-remaining-23-v1-once`. The run reused the previously accepted `validation-0000/repeat-0` observer-gate unit exactly once and executed only the other 23 fixed units. It did not initialize or call a model and did not train.

## Decision

The frozen gate failed: `FAIL_FIRST_WINDOW_HUNGARIAN_CONTINUATION_CEILING`.

All 24 fixed units and all eight development parents had a qualifying opportunity, so the coverage gate passed. The eight-parent macro mean hindsight gain was `0.013226693424422117`, above the per-parent materiality threshold of `0.01`. That aggregate is insufficient under the frozen rule: only 2 of 8 parents passed, while at least 4 were required.

| Parent | Mean gain over 3 repeats | Positive repeats | Parent pass |
| --- | ---: | ---: | --- |
| validation-0000 | 0.002642066847514150 | 1 | no |
| validation-0001 | 0.004418626930532317 | 1 | no |
| validation-0002 | 0.041278567031083310 | 3 | yes |
| validation-0003 | 0.004241392603908545 | 2 | no |
| validation-0004 | 0.000417347870779683 | 1 | no |
| validation-0005 | 0.003719455471962563 | 2 | no |
| validation-0006 | 0.004027490183335962 | 2 | no |
| validation-0007 | 0.045068600456260410 | 2 | yes |

Across 24 units, 14 had a positive hindsight gain and 3 reached gain `>= 0.01` with both first commands accepted. One unit had an acceptance-eligibility mismatch. These counts do not override the parent-level gate.

## What this answers

For the first qualifying public decision window followed by the frozen Hungarian continuation, perfect hindsight over the enumerated legal first actions did not show sufficiently broad material value across the eight development parents. This specific action-consequence target therefore does not justify automatically starting predictor training.

The result does not show that every later decision window lacks value, that all action-consequence targets are unhelpful, or that a world-model class is invalid. The oracle gain is nonnegative by construction because the Hungarian action is included in the maximized candidate set. The hindsight winner may also exploit random outcomes that were not predictable from public history.

## Labels

The joint matrix contains 144 candidate branches: 137 new branches plus 7 branches from the reused gate. Each branch has six task records, for 864 candidate-branch task labels. Of these, 182 host-confirmation labels are `unknown`; 682 are known. Unknown means native termination occurred before the host confirmation was observed. It is neither a physical task failure nor an unresolved accounting call, and it was not imputed or dropped.

The frozen utility is reconstructed from the complete finite reward sequence and does not depend on the unknown host-confirmation field. Therefore the 24-unit primary utility analysis remains defined, while host-confirmation summaries retain the unknown category.

## Execution integrity

- Reused units: 1; newly executed units: 23; fixed matrix: 24.
- `validation-0000/repeat-0` was not re-executed.
- New environment steps: 1,598; resets: 23; public rule decisions: 1,484; branches: 137.
- Model initializations, model forwards, policy updates, world updates, and offline predictor updates: 0.
- Ledger calls: 3,586 verified; pending: 0; unknown: 0.
- Automatic retries and worker relaunches: 0.
- Worker exit code: 0; final resource gate: passed.

The controlled Windows export was independently checked after completion. All 68 manifest entries matched both byte count and SHA-256; there were zero missing or mismatched files.

## Recommendation

Stop this completed oracle-ceiling line at the frozen decision point. Do not automatically train, expand the sample, change the threshold, or start another experiment. A future proposal would need a separately justified action target or decision window and a new authorization; this result alone does not supply that justification.

Historical technical stops, all prior consumption, the historical accounting gaps, and the 10 ms FULL cost failure remain preserved.
