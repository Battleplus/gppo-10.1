# W1 action-outcome finite learning-loop result

This archive records the one authorized execution of
`w1-action-outcome-learning-loop-history-repair-v1-once` on 2026-09-28.

The run was not a technical stop. The label coverage gate passed, three fixed
training seeds completed, and independent prediction evaluation ran. The
learned ensemble reduced parent-macro MAE relative to both public baselines but
did not improve selected-action regret relative to the transparent baseline.
The frozen prediction gate therefore failed and the conditional three-arm task
comparison did not run.

The archive includes the final report, prediction evaluation, resource and
ledger summaries, final settlement, controlled-export records, integrity
checks, and a read-only analysis script. The verified Windows export remains at
the local run path.

This is not a complete backup. It excludes the 188,468,514-byte branch audit
JSONL, 1,792,623-byte learning-record JSONL, SQLite budget ledger, three model
checkpoints, raw console logs, and authorization token. The included
`input-index.json`, `ledger-summary.json`, `export-manifest.json`, and
`hashes.json` identify the relevant local evidence. No credential or token is
included.

Historical negative results, technical stops, unknown labels, and the earlier
10 ms cost failure remain in force. This fixed-Hungarian-continuation result is
not evidence for GPPO plus a world model.
