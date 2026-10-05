# E2E Engineering Acceptance

This package is an independent copy of `w1-eawm-jepa-world-model-production-repair-v3`.
The v3 package and all earlier stop evidence remain unchanged.

## Actual production path

`execute_pipeline` called the following production functions in order:

1. `ProductionDataCollector.collect`
2. `train_select_world_models`
3. `evaluate_prediction_confirmation`
4. `train_policy_routes`
5. `evaluate_task_confirmation`
6. persisted metric recomputation and `W1RuntimeBackend.settle`

The evidence is in `e2e-evidence-20260930-v18`. Each path has its own output,
SQLite ledger, prediction trace, checkpoint files, task trace where applicable,
settlement, and export manifest.

| path | prediction gate | policy training | task comparison | result |
|---|---:|---:|---:|---|
| `prediction_fail` | fail | 0 routes | 0 calls | `prediction_gate_stop` |
| `g1_pass_g2_fail` | G1 fixture pass, G2 fixture fail | 0 routes | 0 calls | `prediction_gate_stop` |
| `all_pass` | fixture pass | 12 routes | 312 episodes, including 24 H | `task_utility_gate_stop` |

The last row is a routing result, not a task-effect claim. Its prediction
metrics and task metrics are real outputs of the production metric code, while
the gate branch is explicitly marked `fixture_only`.

## Bottom-boundary fixtures

Only environment construction, model initialization/forward, optimizer,
checkpoint I/O, and the configured host filesystem sync limitation were
substituted. The collector, training/selection loops, prediction trace,
policy routes, task runner, Hungarian arm, metrics, accounting, settlement and
export manifest were not replaced. The fixture uses 1 world-model epoch and
8 policy steps per route (4-step rollouts, 2 updates per route), and a
deterministic public observation with 25-action masks. These differences are
recorded in every `e2e-path-evidence.json`; no formal W1 matrix result is
claimed. Synthetic operations are counted separately from formal dynamic calls
(`formal_dynamic_calls=0`).

The all-pass fixture proved checkpoint save/load identity, policy replay,
causal history across rollout boundaries, task/H routing, trace persistence,
cost accounting and settlement. The task gate stopped on the produced utility
metrics, as required by the frozen protocol.

## Production fixes found during acceptance

- the adapter gate is carried into `execute_pipeline`, and the backend mirrors
  the gate state into the adapter before policy routing;
- rollout-boundary bootstrap now stores the history-attached observation;
- policy checkpoints save the base policy state before task-time wrapper
  reconstruction;
- settlement evidence converts tuple-keyed internal maps to JSON-safe keys;
- training rejects unknown window splits and excludes confirmation rows.

## Runner readiness

`runner_ready=true` for engineering execution. The package-level WSL contract
and the independent Windows -> Ubuntu-24.04 -> native staging -> supervisor
fixture both passed. The configured interpreter is Python 3.12.3 with
CPU-only `torch 2.8.0+cpu`; the cross-system test used a temporary
integration identity and did not consume formal authorization. This is an
execution-readiness result only: the reduced fixture matrix does not claim the
formal W1 effect or cost result.

The unique formal entry and resource request are frozen for a later approved
run, but no dynamic budget approval is requested by this package.
