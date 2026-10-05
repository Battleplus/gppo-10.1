# Independent source review

The `luna_worker` review of the preceding package found the hard blockers:
hard-coded attempt identity, SQLite opened before output creation, missing
`runtime_adapter.py`, four `NOT_ATTACHED` stage methods, and tests replacing
the production backend with `FakeABBackend`.

This package removes those paths. The attempt is read from the frozen request
and cross-checked with the manifest and launch contract. The output directory
is created by `ABRuntimeBackend` before `BudgetLedger` opens. The bridge loads
from the hash-verified package root rather than a mounted duplicate.

The production backend test instantiates `ABRuntimeBackend` and
`ProductionRuntimeAdapter`; only bottom-boundary operations are substituted.
It verifies six A/B seed routes, trace creation, metric recomputation, gate
failure with zero task calls, and gate success reaching the task stage. The
task routing assertion verifies that `A_one_shot` receives A's three models and
`B_one_shot` receives B's three models. The Windows-to-WSL integration now
executes the production legacy four-arm task loop with only bottom environment,
communication and model boundaries substituted; it no longer shortcuts the
adapter task method.

The formal package is still `NOT_APPROVED`; no dynamic run was performed. No
research configuration or budget changed.
