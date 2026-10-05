# Luna independent resource/export review

Reviewer: `/root/v6_settlement_independent_review`, luna_worker (gpt-6-luna, max).
Same-family read-only review; acceptance is provisional. No test, worker,
environment, model or attempt was executed; SQLite was opened read-only.

The authoritative WSL launcher state is `stopped`, stage `export_failed`,
worker_started=true and worker_returncode=0. Supervisor PID 499 and worker
PID 501 are absent. Do not rerun this consumed attempt. Windows retains an
earlier running/worker_started=false snapshot; do not edit it to match.

Inclusive native tree CPU is 11.546687s. Launcher snapshot self+children is
11.284507s; supervisor snapshot is 11.283655s. The launcher snapshot to reap
difference is 0.262180s, exceeding the frozen 0.05s reconciliation tolerance.
The supervisor-to-launcher difference is only 0.000852s. These are nested
measurements, not independently chargeable CPU costs. The stop is accounting
scope consistency failure, not demonstrated budget exhaustion.

Measured staging CPU/wall is 3.059655/7.085735s (caps 20/20); label-stage
CPU/wall is 11.568853/21.550428s (caps 491/693); export CPU/wall is
0.295859/2.734651s (caps 39/38). Global conservative upper bounds are
54.630130 CPU seconds and 68.645344 wall seconds (caps 551/753).
Peak RSS upper bound is 522,784,768 bytes, below 2 GiB.

SQLite integrity_check is ok: 1,238 complete calls, no pending or failed calls.
Measured dynamic amounts agree with settlement (1,106 environment steps,
108 branches and 998 rule decisions, eight resets). Historical cumulative
cost acceptance remains false/null with explicit monitoring gaps; do not
replace these unknowns with an all-history resource pass.

Windows payload: 119 manifest files and 17,887,782 bytes all match their
digests and lengths; EXPORT_COMPLETE binds the manifest correctly. Native
has the final failed `export-status.json` not included in that copy, and its
final launcher-status differs from the early copied version. Therefore the
copy is complete for its own payload, not a verified successful final run.
Native plus copied payload totals 35,779,918 bytes, below 1 GiB. Supervisor's
artifact scan occurred before final serialization; its byte count is not the
final directory byte count.

The data-only supplement proposal's arithmetic is consistent: 40 parents,
14,160 steps, 1,000 branches, 13,160 rule decisions; scaled draft label caps
445 CPU/828 wall and global caps 505 CPU/888 wall. It is NOT_APPROVED and
runner_ready=false. CPU-tail repair, isolated successor acceptance and
historical use checks remain required. This does not approve a new execution.

Main-thread corroboration is separately saved in
`native-final-status-evidence-v2.json`; the initial audit-tool decoding error
and its incomplete evidence are retained rather than hidden.
