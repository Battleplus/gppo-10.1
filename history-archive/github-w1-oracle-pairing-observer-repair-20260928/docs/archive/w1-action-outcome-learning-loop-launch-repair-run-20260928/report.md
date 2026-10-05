# W1 action-outcome learning loop execution report

## Decision

**Technical stop.** The one authorized attempt stopped during the first label
collection unit. It was not retried, repaired in place, or resumed. Training,
independent prediction evaluation, conditional task comparison, and decision
cost evaluation were not entered and are therefore **not evaluated**.

## First failing boundary

The first unit was `train-1534 / repeat-0`. After seven Hungarian prefix steps,
the first qualifying decision window was reached and one candidate branch ran
to its native terminal. Before that branch could be written as a label,
`learning_schema.history_features()` rejected its history with:

`LearningContractError: public history contains a future observation`

The frozen call path builds `prior_history` from the branch adapter after the
continuation has run, then validates it against the captured decision-time
`public_obs["time"]`. The contract correctly stopped when later branch
observations appeared newer than that decision time. This report records the
failure; it does not repair the code or reinterpret future observations as
legal input.

## Stage outcomes

| Stage | Outcome | Evidence |
|---|---|---|
| Label collection | Technical stop | 0 persisted labels, 0 covered parents |
| Coverage admission | Not evaluated | Label collection did not complete |
| Three-seed training | Not evaluated | No model initialization or optimizer update |
| Prediction baselines and MAE | Not evaluated | Prediction stage not entered |
| Selected-action regret | Not evaluated | Prediction stage not entered |
| Conditional task comparison | Not evaluated | Prediction gate not reached |
| Decision cost | Not evaluated | Learned ensemble was never run |

The executed branch is not counted as a label because no
`learning-records.jsonl` or `data-units.jsonl` was committed before the stop.
It also cannot be treated as a failed task sample.

## Actual consumption

The SQLite ledger contains 36 calls and all 36 are `complete`: 1 reset, 12
environment steps, 1 candidate branch, 1 snapshot, 1 branch copy, 1 forced
first action, 8 candidate scans, and 11 public rule decisions. There are zero
pending and zero failed calls. Model initialization/load, model forward,
optimizer update, sample evaluation, and checkpoint write counts are all zero.

The supervisor stopped with return code 1 because the worker returned the
contract error. No observed counter exceeded a resource ceiling. The
supervisor nevertheless records `final_resource_pass=false` and
`cumulative_resource_acceptance=false` because the run stopped technically;
complete resource acceptance must therefore not be claimed. Combined recorded budget
time was 16.243135 wall seconds and 4.827071 complete CPU seconds; the outer
Windows launcher wall interval was 18.723807 seconds. Peak RSS upper bound was
606,052,352 bytes. Final native size was 331,003 bytes and the Windows export
contains 64 files totaling 338,431 bytes.

## Settlement

The native evidence remains under
`/home/asus/w1-runs/w1-action-outcome-learning-loop-launch-repair-v1-once`.
The controlled Windows export is at
`E:\Z博士\runs\w1-action-outcome-learning-loop-launch-repair-v1-once`.
`EXPORT_COMPLETE.json` reports `verified=true`; its final export manifest
contains `export-status.json`. The exported `launcher-status.json` is the
pre-final `running/native_supervision` snapshot because the Linux launcher
published its final `stopped` status after controlled export. The native final
launcher status is authoritative, and the exported supervisor and worker
statuses independently record the stop. This stale exported launcher-status is
an infrastructure evidence limitation; it does not justify a retry. The prior attempt, the old incorrect Windows
mapping, historical negative results, unknowns, and cost failures remain
untouched.

The unique result classification is **technical stop**. Later scientific gates
were not evaluated and no method-performance conclusion is updated.
