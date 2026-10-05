# Addendum: Controlled Outer Timeout

This addendum updates the test-coverage statement in `root-independent-cpu-review-final.md`. It does not replace that review or change its source-level conclusions about hard enforcement, durable closeout, or uninterruptible I/O.

## Newly Covered Path

`test_metered_joint_entry.py::test_actual_outer_timeout_reaps_child_and_records_failed_lower_bound` now calls the production `metered_joint_entry.main()` against a temporary synthetic package. It creates matching identity files and lets production `verify_package()` run. It launches a real Python child whose fixture writes an attempt/PID/hash-bound registration handoff and sleeps; with a six-second budget and five-second close reserve, the actual outer wait expires, then production code kills and waits for the child, runs subreaper cleanup, validates the handoff, invokes the terminal callback, and persists `-outer-stop.json`.

The assertions check timeout state, a negative child return code, cleanup status `reaped`, one `FAILED` terminal callback, null lifetime total, and `accounting_complete=false`. The CPU value is deliberately not promoted from a partial reaped-child delta to a complete total.

The captured WSL run at `actual-outer-timeout.json` / `.stderr` exits 0 and reports all 8 `test_metered_joint_entry` tests passing in 1.17 seconds. The run reports `ssh_connected=false` and `formal_attempt_started=false`.

## Remaining Test Boundaries

This is a controlled integration test of the outer timeout path, so the earlier statement that no `metered_joint_entry.main()` timeout test existed is superseded. It still does not call the real server ledger: `server_registration.append_terminal` is replaced with a callback that records its arguments, so server-side append/readback durability and terminal-state reconciliation are unverified. The real child is a short synthetic sleeper, not `joint_remote_native.py` with a supervisor, worker tree, export, or research environment; the test does not exercise unresolved grandchild cleanup.

`enforce_closing_limits()` is patched out to avoid changing the test runner's `RLIMIT_CPU` or installing its five-second alarm. The test therefore does not verify actual CPU/wall closeout signals. Nor can this ordinary child test establish behavior when a process or filesystem operation remains in uninterruptible kernel I/O. Those limits do not weaken the evidence that the production outer timeout handler can kill/reap this controlled child and write a fail-closed lower-bound record.

The supported coverage claim is now: **the production outer `main()` timeout path is exercised with a real synthetic child, real timeout/kill/wait and cleanup, a validated handoff, and a simulated terminal callback; actual server-ledger persistence, nested research process cleanup under unresolved descendants, OS closeout limits, and D-state behavior remain untested.**
