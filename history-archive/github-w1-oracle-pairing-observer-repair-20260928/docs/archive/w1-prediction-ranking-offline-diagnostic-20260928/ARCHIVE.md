# Offline prediction-ranking diagnostic archive

This directory archives the bounded offline diagnosis of the sealed
`w1-action-outcome-learning-loop-history-repair-v1-once` prediction evaluation.
The frozen result remains `PREDICTION_GATE_NOT_PASSED`; the task-comparison
stage was not executed.

The run did not persist per-candidate predictions for the three learned seeds
or their ensemble. Consequently, this archive cannot reconstruct learned
window-centered error, pairwise ranking accuracy, score span, selected actions,
per-window regret, acceptance state, or seed-to-ensemble ranking changes.
Checkpoints were not loaded and no model forward, training, or environment call
was performed for this diagnosis.

The eight prediction-evaluation parent scenarios were used for this diagnosis
and must be treated as development evidence for any future method change. They
must not be described as a fresh confirmation set for such a change.

This is a small evidence archive, not a complete artifact backup. It excludes
checkpoints, SQLite ledgers, large raw branch logs, credentials, and launch
tokens. `input-index.json` identifies the sealed source records, and
`hashes.json` authenticates the generated diagnostic deliverables.

For staged-diff hygiene, the archive copy removes one trailing blank line from
each of three Markdown reports; their prose is unchanged. The archive-local
`hashes.json` was regenerated after that normalization. The original delivery
directory retains its own unchanged hash identity.
