# W1 action-outcome learning-loop history repair archive

This directory archives the small, reviewable materials for the independent
history-boundary repair package prepared on 2026-09-28.

The sealed prior attempt stopped after 12 environment steps, 1 reset, 1 branch,
11 public-rule decisions, and 36 settled ledger calls. Its evidence remains
unchanged. The failure was caused by constructing candidate input from public
history after Hungarian continuation had appended post-decision observations.

The repair freezes all shared public input before any candidate branch and
uses the immutable snapshot for candidate records. Branch continuation remains
available only to label and audit construction. The existing future-observation
guard remains strict. An additive final settlement file resolves the earlier
exported-running versus native-stopped status ambiguity without rewriting old
artifacts.

The archived tests include a real Windows-to-WSL entry integration that runs
the production collector with a controlled fake environment, persists and
reloads a label, settles, and completes a verified export. No real W1
environment, model initialization, checkpoint load, model forward, or training
was performed while preparing this package.

`RESOURCE_REQUEST.json` remains `NOT_APPROVED`. The future attempt
`w1-action-outcome-learning-loop-history-repair-v1-once` was not created.

This Git archive is not a complete artifact backup. It excludes the local
24,011,472-byte `diagnostic-records.jsonl`, the local 3,083,757-byte
`public-contexts.jsonl`, all SQLite databases, checkpoints, raw dynamic logs,
and the external launch token. Their local identities and relevant source
indexes are recorded in the included manifests where applicable.
