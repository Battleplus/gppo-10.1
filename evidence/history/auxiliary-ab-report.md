# W1 Relative Auxiliary Objective: One-Shot Run Report

Date: 2026-09-29

Attempt: `w1-action-relative-auxiliary-task-cost-attribution-repair-v1-once`

## Frozen identity

- Preparation package: `E:\Z博士\research-plans\w1-action-relative-auxiliary-task-cost-attribution-repair-v1`
- `execution-manifest.json` SHA-256: `2e88549872a672bc7e426e5be046aa48e26e14a2be4b40114e8db4bc0f5988fa`
- `hashes.json` SHA-256: `9a932841da3f42714cae17d60cb18d726ed9d7f668a33937efe326bc14fdc95e`
- Launches: one formal launch; no dependency-only launch and no automatic retry

## Decision

The run completed technically and stopped at the frozen prediction gate. The final research status is `prediction_gate_stop`.

The centered auxiliary objective B did not improve parent-macro selected-action regret over the controlled A baseline. Both were `0.024674480730902813`, so the A-minus-B improvement was `0.0`, below the required `0.005`. B was non-worse than A in 8/8 parents, and B was strictly better than the transparent-history baseline (`0.03541961257778265`), but all three conditions were required. The four-arm task comparison was therefore not entered.

This is a prediction-gate failure, not a technical stop. It does not establish task utility, decision cost, or a GPPO/world-model advantage.

## Data and training

- New prediction-confirmation coverage: 8 frozen parents, 3 repeats per parent, 24 complete windows
- Legal candidate labels: 113 unique candidate outcomes; all utility targets used in the trace are valid
- Continuation identity: `hungarian-v1-fixed`
- Reused training input: 219 validated records from 24 parents
- Reused model-selection input: 68 validated records from 8 parents
- Training completed for A and B with seeds 7101, 7102, and 7103
- Checkpoints written: 6; optimizer updates: 4,704
- Prediction trace: 226 rows, comprising 113 A rows and 113 B rows
- Independent metric recomputation recorded by the production pipeline: passed

The run did not persist a separate `coverage-gate.json`. Coverage above was reconstructed from the saved candidate trace and is also consistent with the run advancing into six-model training. This missing standalone artifact is an evidence-format limitation; no file is claimed where none exists.

## Prediction results

| Method | Parent-macro MAE | Parent-macro RMSE | Parent-macro regret | Top-1 |
|---|---:|---:|---:|---:|
| A: absolute SmoothL1 | 0.0517300082 | 0.0588967992 | 0.0246744807 | 0.291666667 |
| B: absolute + centered auxiliary | 0.0517428573 | 0.0589179488 | 0.0246744807 | 0.291666667 |
| Current public baseline | 0.2016934174 | 0.2115283482 | 0.0358551045 | 0.166666667 |
| Transparent-history baseline | 0.2039515126 | 0.2135753544 | 0.0354196126 | 0.208333333 |

Frozen gate results:

- B relative to A regret improvement at least 0.005: **failed** (`0.0`)
- At least 4/8 parents B non-worse than A: **passed** (`8/8`)
- B regret strictly below transparent-history regret: **passed**
- Combined task-stage condition: **failed**

The source `prediction-gate.json` field `task_stage_condition_pass=true` names the transparent-baseline subcondition. It does not override `effect_gate_pass=false`; the production result correctly records `prediction_gate_passed=false` and skips the task stage.

The B model's MAE and RMSE were also marginally higher than A's, while Top-1 and selected regret were identical. No threshold or seed was changed after observing these results.

An independent recomputation from the saved candidate rows reproduced all four parent-macro regret values. A and B selected the same action in all 24 windows, so the auxiliary term produced no action-ranking change in this confirmation set.

## Task and cost stages

- Task calls: 0
- Four-arm task comparison: **not evaluated**
- B utility relative to Hungarian: **not evaluated**
- B utility relative to transparent selection: **not evaluated**
- A decision cost: **not evaluated**
- B decision cost and the 10 ms CPU / 50 ms wall gates: **not evaluated**

No `task-comparison.json` or `task-decision-costs.jsonl` exists because the task stage was correctly skipped. An empty cost sample was not treated as passing.

## Resource settlement

The ledger closed with zero pending and zero failed calls. Recorded dynamic totals were 1,020 environment steps, 24 resets, 113 branches, 907 rule decisions, 6,416 model batch forwards, 28,804 model sample evaluations, 12 model initializations/loads, 4,704 optimizer updates, and 6 checkpoint writes.

The supervisor completed with return code 0, about 731.45 CPU seconds, 341.68 wall seconds, and 639,823,872 bytes peak RSS. Combined launcher-through-export accounting was about 732.93 CPU seconds and 348.75 wall seconds. All frozen global and stage limits passed.

## Export and evidence boundary

The controlled export verified 87 files and 67,293,665 payload bytes. The final `export-manifest.json` SHA-256 is `06028c6b2f96af987a3e0af7880a4a45374362168d1651436665dcbf9b60860a`.

The earlier SHA stored inside `export-status.json` describes the first export pass before that status file was appended. `EXPORT_COMPLETE.json` binds the final 87-file manifest above.

This postrun package keeps small reports, summaries, the 135 KB candidate trace, and hashes. It excludes checkpoints, `budget.sqlite3`, the 60 MB `data-units.jsonl`, raw logs, authorization material, and credentials.
