# Independent CPU Scope Review

Review target: the current source under `research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v2`, especially `linux_process_scope.py`, `supervise.py`, `phase_handshake.py`, `joint_remote_native.py`, `metered_joint_entry.py`, and `launch_joint_once.py`. This is a source review, not a formal or remote execution.

## Conclusion

The normal successful Linux path has a defensible lifetime accounting design. Stable owner-checked `/proc` snapshots sum each live process's `utime+stime` and kernel-waited `cutime+cstime`; subreapers identify and clean post-baseline adopted descendants; final `RUSAGE_SELF`/`RUSAGE_CHILDREN` and the outer wait close the accounting after inner snapshots. The nested snapshots are diagnostic and are not added twice. A real short-lived subprocess fixture burns CPU after printing its own snapshot, and the parent's waited lifetime captures that tail.

This is **eventual settlement**, not a kernel-hard CPU cap. The source sets `cpu_scope_hard_enforcement=false`; without a delegated cgroup, the supervisor checks CPU at polling intervals and can detect a cap crossing only after it occurs. Exception paths now preserve an incomplete lower bound and use bounded waits, but D-state I/O and separate durable stores prevent a guarantee that every timeout leaves a complete, mutually consistent terminal record.

## Lifetime Coverage

- `linux_process_scope.py:48-87,117-175,178-203` parses self plus waited CPU ticks, validates UID/starttime/process-tree identity, retries structural races, and rejects nonmonotonic totals. Reaped child ticks are counted through a living parent's waited counters, not added as both live and reaped work.
- `linux_process_scope.py:234-272,309-355` verifies subreaper mode and records the pre-existing direct-child identities before work starts. Cleanup signals only new adopted descendants after PID/starttime checks. Unstable scans or unresolved live processes fail rather than reporting cleanup success.
- `supervise.py:174-176,201-209,228-260` enables subreaping before the worker, samples the process tree and waited-child usage, then cleans descendants and reads final `RUSAGE_SELF`/`RUSAGE_CHILDREN`. It labels the scope incomplete at the broader wrapper level and keeps hard enforcement false.
- `metered_joint_entry.py:133-160` enables a second subreaper, waits for `joint_remote_native.py`, reaps adopted descendants, and computes waited CPU from a pre-launch `RUSAGE_CHILDREN` baseline. `closure_accounting()` (`metered_joint_entry.py:30-68`) charges waited lifetime once, disjoint parent self CPU, and the declared closing reserve; the inner total is used to measure the snapshot-to-reap tail, not added again.
- `joint_remote_native.py:383-415` now synchronizes native total and settlement CPU/wall values from the same post-write snapshot. The outer waited tail after that snapshot remains part of settlement accounting.
- The outer normal path writes a durable `finalization_pending` receipt before appending the terminal registration event, then writes the final receipt (`metered_joint_entry.py:163-180`). This avoids writing a final `complete` receipt before the terminal append. The two stores are still not one transaction; a later write failure can leave a pending receipt alongside an already-terminal ledger event.

The stage handshake (`phase_handshake.py:66-149`) authenticates the worker and rejects a boundary with descendant processes. It provides synchronous accounting boundaries, not a kernel execution barrier. A process-tree snapshot is not an atomic kernel snapshot; the code records that limitation and fails closed on unstable or nonmonotonic observations.

## Enforcement

`supervise.py:201-225` samples, checks `guard()` (`supervise.py:129-138`), writes state, and sleeps for the configured interval (default 0.5 seconds). There is no delegated cgroup CPU controller in this package. CPU can cross a cap between checks, and final `RUSAGE` can reject an observed overrun only retrospectively. Do not describe this as immediate or hard enforcement.

The accurate distinction is: **the normal-path lifetime settlement can include short-lived descendants and CPU after the inner snapshot; sampled enforcement can still overshoot before detection.**

## Failure and Closeout

The outer failure schema (`metered_joint_entry.py:83-92,182-212`) now records a reaped-child CPU lower bound, outer self snapshot, elapsed wall, child result/timeout, cleanup errors, null lifetime total, incomplete stage/lifetime flags, `final_resource_pass=false`, and registration terminal status. It bounds the post-kill child wait and descendant cleanup; the identity-bound registration handoff includes attempt, outer PID, actual child PID, manifest hash, and hashes hash (`metered_joint_entry.py:95-105`, `joint_remote_native.py:253-265`). An unresolved cleanup remains an incomplete technical stop; the partial `RUSAGE_CHILDREN` delta must not be presented as a full lifetime total or assigned to a stage.

The inner worker shutdown also has a bounded wait after `SIGKILL` and marks `unresolved_after_sigkill` (`joint_remote_native.py:433-444`). That status does not prove the worker disappeared. If a task is in uninterruptible kernel I/O, signals and bounded waits may not make it exit by the deadline; the outer scope must retain `accounting_complete=false` when it cannot reap the process tree.

Two closeout boundaries remain:

- Registration append, final receipt replacement, and failure-marker replacement are separate durable operations. The pre-registration `finalization_pending` receipt improves the record ordering, but there is no cross-store atomic transaction. If the terminal append succeeds and the final receipt write then fails or is interrupted, the ledger may contain `SUCCEEDED` while the outer marker says `technical_stop` and the final receipt remains pending. The marker preserves the known terminal event; this is diagnosable but not a mutually committed final state. Do not call it fully closed without reconciliation.
- `metered_joint_entry.py` installs its CPU/wall closeout signals before the terminal append and final write (`metered_joint_entry.py:71-80,163-212`). Its whole-second `RLIMIT_CPU=floor(own)+2` is paired with a 3-second charged reserve, and its 5-second alarm calls `os._exit(124)`. These are conservative closeout controls, not proof an fsync or registration append completes. A delivered signal can bypass cleanup, and a signal cannot ensure progress through D-state I/O.

The handoff is written immediately after `REGISTERED`, but the ledger append and handoff file cannot be atomic. If writing the handoff fails in that interval, the outer correctly leaves registration status unknown rather than guessing; a terminal event is not guaranteed for that attempt.

## Attribution and Controller Reserve

CPU differences reconcile scopes; they do not retrospectively identify individual operations. `joint_remote_native.py:325-348` records the inclusive waited total, an inner launcher snapshot, and `individual_tail_operation_cpu="not_measured"`. `cpu_scope_contract.py:241-247` defines the snapshot-to-reap tail as an inclusive settlement interval whose individual operations are not measured. The historical `0.274925`-second discrepancy is explicitly marked in `cpu-accounting-contract.json` as not retrospectively identifiable without preserved boundaries. Keep it unattributed; the snapshots do not prove it came from a particular fsync, registration append, export, or shutdown operation.

The Windows-side controller reserves two CPU seconds and five wall seconds (`launch_joint_once.py:212-234`) and charges the full transport reserve. Its close cost is explicitly unmeasured. There is no controller `RLIMIT_CPU` or close deadline; `windows-final-settlement.json` is written with `Path.write_text()` after the resource decision, JSON is printed, and `client.close()` runs afterward in `finally`. The reserve is a conservative accounting allowance, not measured or enforced time, and the JSON write is not fsync-backed. It cannot guarantee that the settlement file or connection close completes within the reserve.

## Verification and Limits

The saved WSL evidence records 33 focused regression tests passing (`final-cpu-wiring.json` / `.stderr`, exit code 0), including real post-snapshot CPU-tail accounting, unresolved lower-bound behavior, handoff identity checks, CPU-scope logic, phase handshake, supervised integration, and final settlement. The later 19-test affected-module run also exits 0 (`final-termination-status.json` / `.stderr`). Neither run connects by SSH or starts a formal attempt.

The handoff tests validate valid and mismatched attempt/PID data, but no controlled `metered_joint_entry.main()` timeout/exception test drives the full outer cleanup and FAILED append against a server ledger. Server registration persistence, D-state I/O, and interrupted fsync remain untested. The CPU scope integration fixture uses synthetic CPU subprocesses and injected hardware guards; it is not model or production-environment evidence.

Supported claim: normal successful Linux runs can be charged by waited child lifetime plus disjoint parent self CPU and explicit closeout reserves, with short-lived descendants and snapshot tail included once.

Blocked claim: the source and tests do not establish hard CPU enforcement, guaranteed completion of every exception closeout, cross-store atomic terminal settlement, operation-level attribution of historical deltas, or durable close within the Windows/controller reserves. No research result or formal acceptance is claimed here.
