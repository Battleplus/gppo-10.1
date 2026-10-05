# Production backend completion

This independent preparation package completes the A/B production backend
without running an environment, initializing a real model, loading a
checkpoint, or training. The prior contract-repair package and its stopped
attempt remain unchanged.

`runner.py` reads the attempt from `RESOURCE_REQUEST.json` and cross-checks it
against the launch contract and execution manifest. `ABRuntimeBackend` creates
the native output directory before opening SQLite, owns stage selection and
settlement, and delegates dynamic work to `runtime_adapter.py`.

The adapter reuses the repaired W1 collector and Hungarian continuation for
confirmation labels and task pairing. A/B training uses `windowed_training.py`
with the frozen three seeds and objectives. Prediction writes the complete
per-candidate trace through `trace_schema.py` and recomputes metrics through
`metrics.py`. Only environment, model, optimizer and checkpoint boundaries are
replaceable in integration tests.

The package remains `NOT_APPROVED`. A successful production-backend test is
not a dynamic experiment result and the final `--preflight-only` check does
not create an attempt, lock, worker, environment or model.
