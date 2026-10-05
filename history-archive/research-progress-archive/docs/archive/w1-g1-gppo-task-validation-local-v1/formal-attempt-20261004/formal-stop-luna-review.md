# Formal Attempt Stop Review

Review date: 2026-10-04  
Scope: retained artifacts plus read-only inspection of the frozen phase-accounting code. No run, model, or environment was started for this review.

## Finding

The single authorized attempt was consumed and launched, then stopped in `staging_and_zero_step_gate` with `PHASE_ACCOUNTING_REJECTED:PHASE_CPU_BASELINE_DISCONTINUITY`. Automatic retry is recorded as false. The preflight exit 0 was a separate read-only check; the formal launch exited 1.

Evidence places the stop at the accounting handshake while `task_pipeline` was selecting the next phase, `conditional_policy_training`. The worker status still has `training_started=false`, `model_initialized=false`, `checkpoint_loaded=false`, and `environment_constructed=false`. The task input check records zero model loads. A read-only query of `verified-export/run-once/budget.sqlite3` found the `calls` table with zero rows. No training, model, or environment call is evidenced for this attempt.

## Accounting Trace

The directly observed stop is `PHASE_CPU_BASELINE_DISCONTINUITY`: the supervisor log reports that rejection, its retained status reports zero accepted phase rows, and the worker trace places it at the attempted transition to `conditional_policy_training`. The rejected socket request itself was not retained.

The frozen source and the persisted first snapshot support this source-derived sequence:

1. `runner.py` first calls `ledger.select("staging_and_zero_step_gate")`. At that point the ledger's stage is empty, so `_finish_phase` labels the snapshot as `staging_and_zero_step_gate` with the same `next_stage`.
2. `budget_ledger.py` skips `request_boundary` when `next_stage == stage`, labels the snapshot `final_same_stage_snapshot`, then assigns its measured CPU end to the worker's `phase_started_cpu`. The retained snapshot records a CPU end of `2.866153` seconds.
3. `StageServer` initializes `last_worker_cpu_end` to `0.0` and advances it only after an accepted boundary. The supervisor retained no accepted phase rows, so this baseline remained zero.
4. When `task_pipeline.py` selects `conditional_policy_training`, the source path constructs and sends a cross-stage boundary using the worker's current local baseline, while the supervisor still expects zero. The source guard rejects a difference greater than `1e-6` as `PHASE_CPU_BASELINE_DISCONTINUITY`.

This is a source-derived reconstruction, not a claim that the rejected numeric payload was captured. The saved `2.866153` value is from the persisted initial same-stage snapshot; it supports the later baseline sequence through the frozen assignments and calls. The exact rejected socket payload is absent. The worker's `activity.json` records entry into technical-stop settlement, but `resource-settlement.json` was not produced and `status.json` remains `initializing`; the worker therefore has no retained final resource settlement.

The formal attempt has a consumed-authorization marker and a started supervisor. The separate preflight record exited 0 with no staging or worker start. The retained worker status is its initial status, not a successful completion status.

## CPU Scope

The native controller reports `0.982874` seconds of SELF CPU and `9.470093` seconds of kernel-waited children CPU, for `10.452967` seconds total. Those two components are additive once at the controller scope. Its settlement explicitly says nested supervisor and worker totals are reported for diagnosis and are not added again.

The nested supervisor record reports `9.692891` seconds: `0.021221` SELF, `3.006529` waited children, and a `6.665141` second staging offset. Its scope is incomplete and includes the controller-reported staging offset. The worker's phase snapshot reports `2.866153` seconds of SELF plus waited descendants. These are nested views; do not add either to the controller's `10.452967` seconds. The controller's separate phase figures are `4.072863` seconds for staging and `0.068655` seconds for settlement; they are phase records, not additional totals to add to the controller aggregate.

The launch record says the Windows CPU reserve and measured wall checks passed, and the native controller says its measured controller limits passed. The supervisor's `final_resource_pass=false` is recorded on a stopped run; the frozen predicate also requires there to be no supervisor failure. This is not evidence by itself that a CPU or wall cap was exceeded. Full resource acceptance remains false because the records retain gaps for Windows/WSL bridge CPU and cross-clock reconciliation, controller final write/exit tail, and sampled CPU enforcement without delegated cgroup hard enforcement.

## Conclusion

This attempt stopped before training because the phase-accounting handshake rejected a CPU baseline as discontinuous. Frozen-source inspection and the persisted initial snapshot support a specific synchronization defect in the first-stage baseline path, with the limitation that the rejected socket payload itself was not retained. The SQLite ledger confirms zero calls. No model or environment work should be inferred from the consumed authorization or the successful preflight.
