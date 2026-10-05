# Production Wiring

This package is prepared for the frozen W1 EAWM/JEPA matrix. The formal path is:

`launch_once.py` -> `wsl_stage_and_launch.py` -> `native_launch.py` -> `supervise.py` -> `runner.py` -> `W1RuntimeBackend` -> `ProductionRuntimeAdapter`.

`W1RuntimeBackend` owns stage selection, SQLite ledger construction, exception settlement, and the research gates. `ProductionRuntimeAdapter` owns the stage calls and delegates only bottom-level environment, model, optimizer, and checkpoint operations to `BottomBoundary`; production uses direct native operations and integration tests replace only those low-level operations.

The world-model stages call `production_data.py` for decision-prefix collection, `production_world.py` for the six G1/G2 routes and persisted candidate traces, and `world_model_pipeline.py` for the frozen prediction gates. G1-versus-transparent and G2-versus-G1 gates are computed separately; the frozen branch is all-or-none: only when both pass does the policy stage train all twelve GPPO routes and execute the complete five-method task matrix. If either fails, task results and costs are `not_evaluated`; no partial matrix or post-result arm changes are permitted. The full resource request therefore remains the worst-case matrix. Hungarian uses the same parent/repeat units and is written once per unit for paired comparison.

All model inputs are frozen before candidate branches. Future observations and branch execution results are target-side only. The production collector calls the strict `PublicTransition`/`build_event_target` path; first-seen fields with no receipt provably after the decision remain unknown rather than a negative receipt label. Missing physical/host confirmation remains masked unknown. `RESOURCE_REQUEST.json` remains `NOT_APPROVED`; no method in this package authorizes execution by itself.
