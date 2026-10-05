# Luna Final Artifact Review

**Conclusion:** The formal entry completed with exit code 0 and produced the expected `task_gain_gate_stop` result. The 240-episode matrix and nine-checkpoint set are complete and internally consistent, but G1 fails the utility, stability, and cost gates. Cumulative resource acceptance is also false and must remain distinct from the successful formal process exit.

The nine logical policy routes are exactly seven final reused checkpoints (G0×3, T×3, G1:8301) plus the two new G1 checkpoints for seeds 8302 and 8303. I independently hashed the nine checkpoint files and matched their route records and the restricted checkpoint audit; all routes report 2048 training steps and 128 optimizer updates, with archive and state hashes marked complete. The static checkpoint audit reports `torch_imported=false`; no model or environment was loaded during this review.

The task file has 240 valid rows: 72 each for G0, G1, and T, plus 24 shared H episodes, covering eight parents and three repeats. Each parent-repeat unit has all nine policy identities and one H episode; there are no invalid utility or outcome rows. Frozen Python 3.11.16 independent metrics exactly match the post-export recomputation artifact, with 240 reward recomputations at maximum absolute error 0.0.

G1's macro utility is 0.2368450101, below G0 and T at 0.2456369283 and H at 0.3360456706. The recorded deltas are -0.0087919182 versus G0/T and -0.0992006606 versus H. Only one of three seed-level G1−T differences is positive (seed 8301); the other two are -0.0048890524 and -0.0368625314. G1 is no worse than T on three of eight parents, below the required six, and the paired-parent bootstrap 95% lower bound is -0.0233240962. Coverage passes; all three method comparisons and the parent/seed stability checks fail.

The G1 cost result is 51.073376 ms mean CPU and 150.704827 ms wall p95 over 1,014 decision-cost records; 693 were cost-eligible active decisions. I recomputed the CPU mean and nearest-rank wall p95 from the raw decision-cost rows and matched the reported values. Both cost criteria fail.

Ledger evidence is consistent at two different scopes. The new SQLite ledger has 28,702 complete calls and no failed or pending calls; settlement records all 240 task episodes and 256 optimizer updates complete. The old attempt still has one pending PPO update/backward call and 64 pending actor plus 64 pending encode sample evaluations. The cumulative audit correctly retains those as unknown: updates are 1,180 complete and 1,181 reserved, and `old_pending_update_not_assumed_complete=true`.

The formal runner returned 0 and the new launch passed its measured wall check at 1,175.926 seconds. Separately, cumulative launch wall is 4,805.325 seconds against the 4,670-second global cap, so `full_resource_acceptance=false`; the audit also lists Windows/WSL CPU, controller exit-tail, and outside-entry lifetime gaps. Do not describe the overall cumulative resource contract as passed.

The Windows 3.14.4 post-run audit is a separate offline check. It records 22 floating-point differences with maximum absolute error 8.326672684688674e-17 and stopped at `POST_EXPORT_METRICS_DISAGREE`; it did not restart the experiment or change production evidence. This does not change the formal runner's exit code 0 or the exact frozen 3.11.16 independent recomputation above.

Review method was standard-library-only and read-only apart from these two review files. No torch import/load, model or environment initialization, forward pass, training, new attempt, or task-result rewrite occurred.
