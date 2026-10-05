# Production implementation repair

This package is an independent repair of the sealed runtime-dependency repair
package. The previous zero-step attempt and all of its native/export evidence
remain sealed and are not reused.

## Keyword argument accounting

`ProductionRuntimeAdapter._accounted` and `AuthorizedRuntimeBackend.account_call`
now accept and forward both `*args` and `**kwargs` through the ledger and
bottom boundary. A zero-environment test passes `map_location="cpu"` and
`weights_only=True` to a fake loader and checks unchanged delivery with one
ledger call.

## Window batch prediction

`evaluate_predictions` groups complete prediction records by decision window,
sorts candidates by action identity, and performs one batch forward per window,
variant, and seed. Sample charges use the actual candidate count. Outputs are
checked for finite values and exact batch length, then mapped back by action
identity before the per-candidate trace is written. Missing candidates,
identity mismatches, NaN/Inf and wrong output shapes stop immediately.

The upper bound remains 24 windows x 2 variants x 3 seeds = 144 batch forwards.
No task or training budget changed.

The Windows-to-WSL integration boundary was also aligned with the frozen new
confirmation parents and the production batch-call contract. The production
loop itself is retained in the test; only its environment and model-compute
bottom boundaries are substituted.

## New-confirmation coverage

The collector reads the frozen `new-prediction-parent-selection.json` instead
of the old prediction-evaluation group. Training and model-selection rows are
reused only from the hash-verified historical file; old prediction-evaluation
rows are excluded. Confirmation passes only with at least one complete valid
window for every one of the eight selected new parents. Missing parents remain
missing and stop the pipeline before training, model loading, or task calls.

The actual reused file was checked directly:

- path: `E:\Z博士\runs\w1-action-outcome-learning-loop-history-repair-v1-once\run-once\learning-records.jsonl`
- SHA-256: `c502768a0483b485c78e9d23ee0a63e9a5595717c3524d38a243f167ccdaeca4`
- 398 records: train 219, model-selection 68, old prediction-evaluation 111
- fixed-Hungarian continuation, complete candidate windows, finite targets and
  parent identities matched the frozen prior matrix

## Scope

Research targets, A/B objectives, model architecture, seeds, gates, budgets and
the prior `PREDICTION_GATE_NOT_PASSED` decision are unchanged. This package has
not run an environment, model initialization, checkpoint load, optimizer
update, training step or real prediction forward.
