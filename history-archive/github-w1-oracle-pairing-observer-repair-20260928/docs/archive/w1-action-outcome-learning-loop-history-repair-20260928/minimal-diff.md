# Minimal repair diff

## Production change

Only `runtime_backend.py` changes the research execution path:

- add immutable `DecisionInputSnapshot` and `freeze_decision_input()`;
- freeze all shared public feature dependencies in `_collect_unit()` before
  snapshot capture and before any branch executes;
- pass the frozen snapshot through the existing deep-copy branch capture;
- build each candidate record from the snapshot rather than the advanced branch
  adapter;
- record and recheck the decision-input SHA-256 around every branch.

The future-observation guard in `learning_schema.py` is unchanged.

## Settlement change

`wsl_stage_and_launch.py` adds `FINAL_SETTLEMENT.json` after the launcher writes
its final status, then appends that new file to the already verified controlled
export. Consumers can use it to distinguish final launcher, supervisor, worker,
resource, and export states without rewriting any earlier hashed file. This is
an additive settlement record, not a supervisor redesign.

## Test-only additions

`history_regression_harness.py`, `integration_history_worker.py`,
`integration_history_native_launch.py`, `test_history_boundary_repair.py`, and
`test_history_entry_integration.py` exist only to run the production collector
against a controlled fake environment. They import the frozen public adapter,
communication observer, labeler, feature builder, validator, ledger, and
persistence code. They do not import or construct the real environment or any
model.

Attempt identity, source/export paths, external token hash, documentation, and
freeze indexes are updated for the independent package. No other research
implementation file changes.
