# W1 EAWM/JEPA run summary

This directory contains the small, reviewable archive for the one authorized run
of `w1-eawm-jepa-world-model-production-repair-v4-e2e-integration-auth-repair-v1-once`.
It contains summaries and identity evidence only. The native run output, candidate
trace, checkpoints, SQLite ledger, authorization, and token remain outside Git in
the controlled export and private authorization locations.

The run completed through settlement and verified controlled export. Coverage was
8/8 prediction parents, but the frozen prediction gate stopped the run before any
GPPO policy or task call. No retry, scene replacement, threshold change, or task
evaluation occurred.

Local controlled export:

`E:\Z博士\.codex-exports\w1-eawm-jepa-world-model-production-repair-v4-e2e-integration-auth-repair-v1-once`

The package identity and resource request remain frozen and are recorded in
`export-identity.json`. See `result-summary.json` and `resource-settlement.json`
for the reproducible aggregate result and ledger totals.
