# W1 action-outcome learning-loop final report

## Decision

**Prediction gate not passed.** This was an expected gated research stop, not a
technical stop. Label collection, coverage admission, three-seed training, and
independent prediction evaluation completed. Conditional task comparison did
not run.

## Label coverage and training

The data gate passed with parent coverage train/model-selection/prediction of
24/8/8
against required coverage 18/6/6.
The frozen splits contain 219 training, 68
model-selection, and 111 prediction-evaluation
candidate labels. All 398 persisted branch input
digests match their decision snapshots, and no communication-pairing failure was
recorded. The branch explanations retain
1750 completed,
104 expired, and
534 unknown task statuses.
Host-confirmation labels retain
1683 true,
67 false,
and 638
unknown values rather than imputing them. First-action evidence also retains
10
transport losses and 13
unknown acceptance classes.

All three seeds produced checkpoints. Training used 468
optimizer updates in total (7101: 304 updates, 7102: 80 updates, 7103: 84 updates); early stopping remained frozen.

## Independent prediction evaluation

Across 8 prediction parents, 24 decisions,
and 111 candidates:

- learned MAE: 0.046080960; current-public MAE:
  0.202324152; transparent-history MAE:
  0.209299444;
- learned MAE is lower by 0.156243192
  (77.22%) versus current-public
  and by 0.163218483
  (77.98%) versus transparent-history;
- learned selected-action regret is 0.007620270,
  versus 0.006897738 for current-public and
  0.006897738 for transparent-history;
- learned top-1 accuracy is 0.375, versus
  0.500 for both baselines.

Thus both frozen MAE checks passed, but the selected-action-regret check failed:
regret increased by 0.000722533
relative to the transparent baseline. Better scalar prediction error did not
improve candidate choice.

## Downstream stages

Task utility relative to Hungarian and transparent one-shot selection is **not
evaluated** because the prediction gate failed. The 10 ms CPU-mean and 50 ms
wall-p95 full decision-cost standards are also **not evaluated** because the
conditional task stage and its cost instrumentation did not run.

No result here establishes GPPO plus world-model value. The target and use both
remain one action followed by frozen Hungarian continuation.

## Settlement

The ledger settled with zero pending and zero failed calls. Dynamic use was
3529 environment steps, 88 resets,
398 branches, and 3131 public-rule
decisions. Persisted combined wall/CPU were
230.819/
163.572
seconds. Peak RSS upper bound was
771072000 bytes.
All were below frozen limits.

The controlled export is verified. Its final manifest covers
80 files and has SHA-256
`d4c4eb8d2cf609b20eec4932d9a7a98b50750163048d32bc8d9cf6ecd3f2dab6`. Independent
post-run verification found 0 file mismatches.
The Windows console additionally reported an outer wall interval of
250.7680907 seconds; the launcher explicitly marks that number as terminal-only
and unavailable in native artifacts.
