# Task cost attribution repair validation

Package: `E:\Z博士\research-plans\w1-action-relative-auxiliary-task-cost-attribution-repair-v1`

Attempt identity: `w1-action-relative-auxiliary-task-cost-attribution-repair-v1-once`

## Frozen identity

- execution-manifest.json SHA-256: `2e88549872a672bc7e426e5be046aa48e26e14a2be4b40114e8db4bc0f5988fa`
- hashes.json SHA-256: `9a932841da3f42714cae17d60cb18d726ed9d7f668a33937efe326bc14fdc95e`
- frozen content files: 58
- RESOURCE_REQUEST status: `NOT_APPROVED`

All 58 content digests matched after the final preflight. The manifest digest
also matched the digest recorded by hashes.json.

## Regression

Command:

```powershell
python -m pytest -q test_task_cost_attribution.py test_production_implementation_repairs.py test_production_backend_completion.py test_windows_wsl_production_chain.py
```

Result: `13 passed, 2 subtests passed in 36.53s`.

The four cost-specific cases use controlled clocks and bottom-boundary model
and environment substitutes. They verify A-fast/B-slow, A-slow/B-fast,
parent/repeat grouping and independent statistic recomputation, no opportunity,
unchanged actions and utilities, and unchanged model-forward counts.

## Luna review

The custom `luna_worker` completed a read-only review. It confirmed the raw
record fields, method-specific summaries, B-only cost gate, explicit
not-evaluated state, preserved timing boundary, and unchanged three-forward
count. Its initial observation that the test did not explicitly recompute
statistics by parent/repeat was addressed before freezing. No review blocker
remains.

## Final read-only preflight

Command:

```powershell
python -B E:\Z博士\research-plans\w1-action-relative-auxiliary-task-cost-attribution-repair-v1\launch_once.py --authorization-file E:\Z博士\.codex-tmp\w1-action-relative-auxiliary-task-cost-attribution-repair-v1.authorization.json --preflight-only
```

Exit code: 0.

WSL output:

```json
{"attempt":"w1-action-relative-auxiliary-task-cost-attribution-repair-v1-once","schema":"w1-linux-preflight/2.0.0","staging_started":false,"status":"preflight_pass","worker_started":false}
```

Windows summary reported `preflight_pass`, WSL return code 0, CPU-only torch
`2.8.0+cpu`, and Python `/usr/bin/python3.12`. No staging, worker, native
attempt, Windows run directory, or export directory was created.

## Research identity

`experiment-matrix.json`, `derived-budget.json`,
`new-prediction-parent-selection.json`, and `data-reuse-audit.json` are byte
identical to the predecessor package. `RESOURCE_REQUEST.json` is identical
apart from the new attempt name. Research targets, data, model, seeds, gates,
and all resource ceilings are unchanged.
