# Production Label-Pipeline and Budget-Accounting Test Evidence

Run on Ubuntu-24.04 through the package's verified native CPU Python:

```text
wsl.exe --distribution Ubuntu-24.04 --cd '/mnt/e/Z博士/research-plans/w1-action-conditioned-task-outcome-label-repair-v5-budget-accounting' --exec /home/asus/w1-action-relative-auxiliary-runtime-dependency-repair-v1-runtime/bin/python -B -m unittest discover -s . -p 'test_*.py' -v
```

Result: 45 tests, exit code 0 (`Ran 45 tests in 0.683s`, `OK`). The production
collector lifecycle tests use a deterministic fake environment backed by the
repository's `TaskLifecycle`; they exercise the production public adapter,
feature/label wiring, fixed Hungarian continuation, contract validation,
JSONL persistence and reread. They cover first-command acceptance, rejection
and loss; completion under the first action, later reassignment and pre-existing
execution; on-time arrival, explicit expiry, unresolved early end and missing
completion notice; NOOP and no-opportunity handling; frozen decision inputs;
candidate/action/continuation identities; and refusal of a runtime-config
mismatch before environment construction.

Runner bootstrap tests verify first-status ordering, early-stop settlement,
duplicate-start rejection, and that failed identity checks do not create an
attempt directory, lock, or SQLite file. These tests use temporary native paths.
v4's exact-deadline lifecycle test and construction-evidence checks remain in
the suite unchanged.

Five accounting tests verify that complete native process-tree CPU is counted
once, supervisor CPU is used only as a consistency check, preflight identity and
stage upper bounds are checked, and wall/CPU reserves fail closed. Five
zero-dynamic launcher tests exercise a synthetic Linux parent/supervisor/worker
process tree (about 0.12 CPU seconds in the grandchild), terminate a sleeping
native child at its wall deadline, interrupt synchronous stage work at its
Linux alarm, exercise the actual Windows `taskkill /T /F` timeout branch with a
controlled subprocess double, and validate the Windows `GetProcessTimes`
kernel/user tick conversion using mocked API structures. No test invokes the W1 environment or creates a model,
attempt, lock, SQLite database, checkpoint, or training run.

The independent process diagnostic observed outer waited-child CPU of
0.246027 seconds and supervisor self-plus-child CPU of 0.242640 seconds; the
0.003387-second difference is within the frozen 0.05-second consistency
tolerance. The small difference is process-exit/reporting overhead, not an
additional CPU charge.

No real W1 environment was constructed. Real environment steps/resets,
candidate branches, model initialization/loading, checkpoint operations,
model forwards, and optimizer updates were all zero. Synthetic lifecycle
fixture steps are not counted as real dynamic calls and are not evidence of
real label coverage or model/task effects.
