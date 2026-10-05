# Completion evidence

## Scope

This is a new preparation package. The prior contract-repair package and its
stopped attempt were not edited or reused as an execution directory. No real
environment, reset/step, model initialization, checkpoint load, optimizer
update or training was performed by this completion work.

## Production path

The formal path is `launch_once.py` -> `wsl_stage_and_launch.py` ->
`native_launch.py` -> `supervise.py` -> `runner.py` -> `ABRuntimeBackend` ->
`ProductionRuntimeAdapter`. The runner reads `request["attempt"]` and checks
the manifest and launch contract; no attempt string is hard-coded in the
worker.

`ABRuntimeBackend` creates `run-once` before opening `budget.sqlite3`. It owns
stage selection, per-call accounting and settlement. `runtime_adapter.py`
contains concrete collection, A/B training, checkpoint loading, prediction
trace, metric recomputation and task-comparison routes. The task route passes
separate three-seed ensembles to `A_one_shot` and `B_one_shot`; `NOT_ATTACHED`
is not present in the production path.

## Tests

- `test_production_backend_completion.py`: 3 passed. The formal runner, real
  backend and adapter execute six A/B seed routes, write a complete trace,
  independently recompute metrics, route both named model ensembles into task
  comparison, and keep task calls at zero when the prediction gate fails.
- `test_windows_wsl_production_chain.py`: 1 passed. The actual Windows entry,
  WSL staging, native supervisor, production backend, settlement and verified
  export ran with a controlled bottom boundary. No environment or real model
  was constructed.
- Python compilation and source scan completed; no `NOT_ATTACHED` or old
  attempt literal remains in the production path.

The WSL integration result is a wiring result only. It is not an experiment
result and does not authorize dynamic execution.

## Identity

The final package is `NOT_APPROVED`. `execution-manifest.json`, `hashes.json`,
`RESOURCE_REQUEST.json` and `launch-contract.json` are frozen together. The
authorization file is outside the package and binds both outer digests and the
resource-request digest. A final read-only preflight is recorded outside the
package after freezing.

Final identity is recorded in the external authorization file and the package-
external preflight evidence, avoiding a self-referential digest in this report.
