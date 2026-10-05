# W1 EAWM/JEPA Authorization Repair

Status: `READ_ONLY_PREFLIGHT_PASSED_NOT_APPROVED` (2026-09-30). This is a new
package identity based on v4. The v4 package, failed attempt, and earlier stop
evidence remain unchanged.

## Authorization repair

The failed external authorization contained one JSON object followed by two
trailing characters. Its parsed fields had the expected names and types, and
the schema, status, attempt, manifest, hashes, and resource-request bindings
matched the prior v4 identity. The token digest was a string and is not shown.
The trailing content caused the Windows entry to stop before staging.

The new external authorization is written with the standard JSON serializer,
read back, and passed through the production verifier. It is `NOT_APPROVED`,
contains no token digest, and is bound to this package's attempt and frozen
hashes. The shared verifier now rejects missing or unknown fields, incorrect
types, invalid digests, duplicate fields, trailing JSON content, and identity
or token mismatches. Tests assert rejection before WSL, staging, or worker
startup. `--preflight-only` is structural and read-only; it does not grant
formal approval.

## End-to-end evidence

The production `execute_pipeline` worker called, in one process:

1. `ProductionDataCollector.collect`;
2. `train_select_world_models` (G1/G2, three seeds, checkpoint save/load);
3. `evaluate_prediction_confirmation` (candidate traces and independent metrics);
4. `train_policy_routes` (conditional GPPO routes);
5. `evaluate_task_confirmation` (including the frozen Hungarian H arm);
6. metric recomputation, `W1RuntimeBackend.settle`, SQLite accounting, and controlled export.

Unchanged predecessor evidence is in
`E:\Z博士\research-plans\w1-eawm-jepa-world-model-production-repair-v4-e2e-integration\e2e-evidence-20260930-v18`.
Only environment construction,
model/optimizer calls, and checkpoint I/O are counted bottom-boundary fixtures;
collector, validation, training loops, prediction, gates, GPPO routing, task
evaluation, Hungarian, metrics, settlement, and export are production code.
The fixture uses one world-model epoch, eight policy steps per route, four-step
rollouts, and two updates; it does not represent the formal W1 matrix.

| Path | Result | Policy calls | Task calls |
| --- | --- | ---: | ---: |
| `prediction_fail` | `prediction_gate_stop` | 0 | 0 |
| `g1_pass_g2_fail` | `prediction_gate_stop` | 0 | 0 |
| `all_pass` | policy/checkpoint/task stages reached, then `task_utility_gate_stop` | 12 routes | 1 production task stage / 312 fixture episodes |

Gate outcomes are selected by deterministic fixtures solely to exercise the
pre-registered branches. The prediction and task numbers are outputs of the
production metric code, not research-effect evidence. `formal_dynamic_calls`
for environment, model, optimizer, and checkpoint are all zero; fixture calls
are listed separately in the evidence.

## Regression coverage

The predecessor v4 passed 54 tests, static audit, budget validation, and Python
compilation. Those are historical engineering results; the authorization
repair tests for this package are reported separately.
Coverage includes finite GPPO/PreCo loss and gradients with partial illegal
masks, NOOP, and nonzero priors; behavior/replay log-prob identity; causal
history across rollout boundaries; episode reset and terminated/truncated
bootstrap; frozen world-model parameters; separate true utility and transparent
score; Hungarian dispatch; unknown/no-opportunity handling; candidate traces,
cost accounting, checkpoint identity, and export identity.

## Cross-system entry

Evidence outside the package is at
`E:\Z博士\research-plans\w1-eawm-jepa-world-model-production-repair-v4-e2e-integration-cross-system-evidence-20260930-v4`.
The actual Windows -> Ubuntu-24.04 -> native staging -> supervisor -> integration
worker -> settlement/export returned code 0. The temporary test identity was
unchanged before and after the run, its token was not persisted, and no formal
attempt was created. The predecessor's `launch_once.py --preflight-only`
command returned code 0. The final frozen identity also passed the Windows ->
Ubuntu-24.04 read-only preflight with staging and worker both false; the
external verification record is beside this package. The dependency probe
reported Python 3.12.3 and CPU-only
`torch 2.8.0+cpu`, without starting staging or a worker.

`runner_ready=true` therefore means engineering entry and controlled production
orchestration are runnable. It does not mean that formal W1 effects, task gains,
or cost thresholds passed. Formal execution still needs external one-time
authorization bound to this package's manifest, hashes, and resource request;
the request remains `NOT_APPROVED`.

## Not evaluated

No real environment, formal model, formal checkpoint, or training call ran. No
sealed confirmation set was read, and no formal dynamic budget approval was
requested. Formal label coverage, G1/G2 prediction gates, GPPO task utility,
10 ms CPU mean, and 50 ms wall p95 are all `not evaluated`. Fixture success must
not be described as world-model effectiveness or task improvement.

## Identity

Attempt: `w1-eawm-jepa-world-model-production-repair-v4-e2e-integration-auth-repair-v1-once`.
The unique formal entry is `launch_once.py`; the exact command is in
`unique-launch-command.md`. `freeze_runner_package.py` generates the final
manifest, hashes, and pending external authorization. `RESOURCE_REQUEST.json`
remains `NOT_APPROVED`; no formal run authorization is asserted here.
