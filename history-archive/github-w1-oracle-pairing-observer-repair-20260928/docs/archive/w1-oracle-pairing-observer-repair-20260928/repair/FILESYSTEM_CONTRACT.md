# Native filesystem contract

The future attempt, if separately authorized, stages sealed files to:

`/home/asus/w1-runs/w1-action-consequence-oracle-observer-single-unit-gate-v1-once`

The active ledger, branch evidence, observer records, status, and temporary files remain on native ext4. After worker exit and final settlement, one verified export may be written to:

`/mnt/e/Z博士/runs/w1-action-consequence-oracle-observer-single-unit-gate-v1-once`

Both paths must be absent before launch. Existing paths cause a zero-call stop. No worker restart, output overwrite, or environment-call retry is allowed. File operations alone use the bounded retry contract in `RESOURCE_REQUEST.json`.

The observer records are ordinary active artifacts and count toward the same stage and aggregate storage limits. Export failure preserves native evidence and does not restart the worker.
