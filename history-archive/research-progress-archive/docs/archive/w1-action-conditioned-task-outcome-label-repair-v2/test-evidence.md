# Production Label-Pipeline Test Evidence

Run from WSL Ubuntu-24.04 so temporary output and directory `fsync` use a
Linux-native filesystem. The collector, config validator, task-outcome label
validator, JSONL persistence, public adapter, and Hungarian continuation are
production code. Only the bottom environment interaction is a deterministic
fake environment backed by the repository's `TaskLifecycle`; Graph-5 feature
construction, transparent score, vector reward, and policy-component import
are controlled test substitutes. No real W1 environment is constructed.

Command:

```text
wsl.exe -d Ubuntu-24.04 -- bash -lc "cd '/mnt/e/Z博士/research-plans/w1-action-conditioned-task-outcome-label-repair-v2' && python3 -B -m unittest discover -s . -p 'test_*.py' -v"
```

Latest result: 22 tests, exit code 0 (`Ran 22 tests in 0.246s`, `OK`). The suite
exercises first-command acceptance, rejection and loss; completion under the
first action and under a later Hungarian reassignment; pre-existing execution;
explicit expiry; unresolved early termination and horizon cutoff; NOOP
masking; a persisted no-opportunity window without fabricated labels;
public-input immutability; fixed continuation identity; persistence and reread
of the first label; runtime config digest and `environment.json` emission from
the exact config object; and refusal of a config mismatch before the
environment constructor is called.

The persisted label is checked against the candidate audit row by both
`candidate_id` and `action_id`. All labels retain the same decision-input hash
and continuation identity. The saved strategy observation contains no
`task_lifecycle` object. Invalid step gaps, time beyond the horizon, and an
empty post-action trace fail with `TaskOutcomeContractError`.

This is synthetic integration evidence for the production collector contract.
It is not real-label coverage, a model run, or evidence of a task or prediction
effect. This preparation used 0 real environment steps/resets, 0 model
initializations/loads, 0 checkpoint operations, 0 model forwards, and 0
optimizer updates.
