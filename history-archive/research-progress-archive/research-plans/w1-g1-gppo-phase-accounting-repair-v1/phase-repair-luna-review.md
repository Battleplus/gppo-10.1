# Luna Phase Accounting Review

Verdict: `ready_for_freeze=true`; `blocking_findings=[]`. This is a protocol and CPU-accounting review, not model or scientific readiness approval.

The stable-2 production driver exited 0 and reports all eight cases passed with `source_unchanged=true`. I independently reconciled the worker and supervisor journals, CPU partitions, final source hashes, and each export manifest. Every committed-stage CPU sum plus its final tail equals supervisor CPU within 1 microsecond; supervisor CPU plus the separately measured outer startup/exit tail equals outer waited CPU.

| Case | Committed CPU | Final tail | Supervisor CPU | Outer waited CPU | Tail attribution |
|---|---:|---:|---:|---:|---|
| delayed_ack_timeout | 0.000000 | 14.045267 | 14.045267 | 14.275015 | staging_and_zero_step_gate |
| drop_final_ack | 13.590768 | 0.400195 | 13.990963 | 14.252839 | UNATTRIBUTED |
| duplicate | 13.530955 | 0.484409 | 14.015364 | 14.252356 | conditional_policy_training |
| entry_rejection | 0.000000 | 13.068627 | 13.068627 | 13.339231 | UNATTRIBUTED |
| exit_after_confirmation | 13.860761 | 0.190118 | 14.050879 | 14.288955 | conditional_policy_training |
| exit_before_confirmation | 0.000000 | 14.093866 | 14.093866 | 14.350477 | staging_and_zero_step_gate |
| first_rejection | 0.000000 | 13.939523 | 13.939523 | 14.210491 | staging_and_zero_step_gate |
| normal | 13.549796 | 0.362857 | 13.912653 | 14.146826 | conditional_task_confirmation |

The code review found the intended accounting behavior in [budget_ledger.py](E:/Z博士/research-plans/w1-g1-gppo-phase-accounting-repair-v1/package/budget_ledger.py#L36), [phase_handshake.py](E:/Z博士/research-plans/w1-g1-gppo-phase-accounting-repair-v1/package/phase_handshake.py#L74), [runner.py](E:/Z博士/research-plans/w1-g1-gppo-phase-accounting-repair-v1/package/runner.py#L156), and [supervise.py](E:/Z博士/research-plans/w1-g1-gppo-phase-accounting-repair-v1/package/supervise.py#L561): snapshots keep a fixed origin, only acknowledged transitions move it, the server rejects conflicting sequence/payload identities, lost commit ACKs remain ambiguous, and the supervisor assigns an open tail only when the worker's durable after-ACK stage matches the server commit. Otherwise it records the tail as unassigned and stops. Phase rows are non-additive views included once in the supervisor CPU total.

All eight worker first errors are retained. The two deliberate hard-exit cases use supervisor fallback settlement, preserve the worker-exit error, and mark worker settlement missing; a worker traceback is unavailable for those hard exits. The three short-lived CPU subprocesses in the normal case were waited and recorded at 0.031478, 0.028488, and 0.040908 CPU seconds (0.100874 total).

The source-before/source-after map covers 210 files and has zero mismatches against the current package. Each case's 151 Python module hashes match the reviewed package, and all exported file sizes and SHA-256 values match their manifests. The complete relative-path Python hash map is in [phase-repair-luna-review.json](phase-repair-luna-review.json).

The replay used production worker/runner/supervisor code with synthetic CPU work and deliberate protocol failures. It recorded zero research model calls, zero environment calls, no CUDA initialization, and no formal attempt; third-party Torch was imported in workers. No model or scientific behavior was exercised. Freeze can proceed subject to the planned post-freeze eight-case replay against the unchanged Python source hashes.

