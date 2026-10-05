# W1 relative-utility auxiliary objective archive

This archive contains the preparation package for the bounded A/B comparison
of absolute remaining-utility SmoothL1 (A) and the same objective plus a
window-centered auxiliary term (B). The prior decision remains
`PREDICTION_GATE_NOT_PASSED` and this package is `NOT_APPROVED`.

No environment, reset/step, checkpoint, model forward, optimizer update, or
training call was made during preparation. The package reuses 219 train and
68 model-selection labels only after schema, complete-window, target, flag,
continuation, and parent identity validation. Old prediction-evaluation and
task-comparison records remain excluded. Eight new prediction parents and the
eight sealed task parents are recorded separately; historical use by unrelated
strategy runs is not claimed absent.

The archive includes the objective implementation, lazy training adapter,
paired per-candidate trace schema, independent metric reconstruction, tests,
resource request, manifest, and unique fail-closed entrypoint. It excludes
checkpoints, SQLite ledgers, raw branch logs, credentials, and launch tokens.
This is a preparation archive, not a complete artifact backup.
