# Post-run Independent Review

**Attempt:** `w1-task-outcome-g1-g2-joint-local-v1-once`  
**Evidence:** `verified-export/run-once` and its export hash manifest

## Conclusion

The exported run completed prediction evaluation with runner return code 0. I found no inconsistency in the audited collection records, checkpoint identities, masked prediction metrics, or trace-stored action regrets. The saved outcome reports G1 passing its prospective comparison gate and G2 failing its comparison against G1. These are exploratory results on eight within-study held-out parents, not a population efficacy claim.

Resource settlement is `measured_limits_pass_with_disclosed_gaps`; `full_resource_acceptance` is false. Native-controller wall time is 978.273764 seconds and Windows-launcher wall time is 963.911949 seconds, a 14.361815-second cross-clock difference whose source remains unresolved. Both clock-domain values are retained without correction.

## Audit Findings

- The 40 collection windows match the frozen parent split and tape identity: 24 training, 8 model-selection, and 8 prediction-confirmation parents. All windows completed and all 40 constructed events were accounted for. Data-admission support matched the 24 training parents for physical positives, expiry positives, and valid host labels.
- Across 547 collection candidate branches, each window has one NOOP with null targets and false lifecycle masks. The 20 unknown host-label cases were masked. All 4,689 continuation steps use `hungarian-v1-fixed`; sequence-return sums match branch scalar returns with no mismatches.
- Collection ledger totals are 5,396 environment steps, 40 resets, 547 branches, and 4,849 public-rule decisions. The collection phase records 114.878 CPU seconds and 217.177 wall seconds.
- All 27 exported files match the SHA-256 values in the export hash manifest. The six route best epochs are G1 seeds 8201/8202/8203: 6/4/14, and G2 seeds 8201/8202/8203: 8/2/3. Each checkpoint byte hash matches its training-summary metadata. Checkpoint bytes were hashed only; no checkpoint was loaded during this review.
- Read-only SQLite inspection found 26,519 ledger calls, all `complete`, with no pending or failed calls. The ledger records six checkpoint writes and six prediction-phase checkpoint loads. Exported settlement totals independently report six loads and six writes.
- The eight prediction-confirmation traces contain 101 candidate rows. I recomputed event Brier scores over valid masks using head order `public_field_change`, `new_measurement_received`, `continuation_publicly_confirmed`, `physical_completion_observed`, `host_confirmation_observed`; all stored per-parent and parent-macro values match. The first three heads have valid labels; the physical-completion-observed and host-confirmation-observed event heads are masked. The `public_field_change` event score is supported on only two confirmation parents.
- I recomputed the three horizon task-outcome Brier heads in order `physical_on_time_completion`, `task_expired`, `host_confirmation`, respecting their per-row validity masks. All stored per-parent and parent-macro values match.
- Recomputing trace-stored decision regret as the maximum candidate `true_utility` minus the selected candidate's utility reproduces all 72 stored decisions with zero error. The selected actions for G1 are `[1, 1, 1, 0, 1, 1, 1, 1]`; for G2, `[1, 1, 1, 18, 1, 1, 1, 1]`; the transparent baseline selects NOOP action 24 in all eight windows. G1 minus transparent macro regret is -0.183035; G2 minus G1 is +0.050385, consistent with the saved G1-pass/G2-fail outcome.
- GPPO was not executed. The transparent baseline is the recorded optimistic public-information surrogate, not a full Hungarian continuation policy.

## Scope and Limitations

This was an artifact-only review. I read exported JSON/JSONL and SQLite in read-only mode and hashed checkpoint bytes. I did not read the old sealed confirmation set, load a model or checkpoint, run inference or forwards, construct the environment, or train. No package files were modified. The existing startup review at `local-independent-review.md` is preserved.

The final resource record retains disclosed gaps, including unavailable WDDM per-process GPU-memory/exclusivity accounting, unmeasured WSL bridge CPU, the native controller's final write/exit tail, and the absence of hard delegated CPU-limit enforcement. The wall-clock discrepancy above remains unresolved. Prediction evidence is limited to eight within-study held-out parents; global novelty is not established.
