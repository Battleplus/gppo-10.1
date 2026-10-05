# Real-entry integration evidence

The tests start from the real Windows `launch_once.py`, invoke the explicit
`Ubuntu-24.04` distribution, execute native Linux staging and directory fsync,
run the real Linux supervisor around a zero-call fake worker, settle, and invoke
the real controlled export. They do not directly call the Linux entry as a
substitute for the Windows entry.

The test source directory includes Chinese characters, spaces, and a single
quote. A separately frozen argument probe contains both single and double
quotes and is checked by the Linux entry. Subprocess calls use argument arrays
with `shell=False`; the external token is passed on stdin.

Validated cases:

- successful native ext4 staging, supervisor completion, concurrent SQLite
  reading/writing, zero pending rows, and verified per-file Windows export;
- wrong token rejected before WSL staging;
- repeat launch rejected before a second worker, with one invocation retained;
- injected native-root directory-fsync failure recorded during staging;
- injected Linux child-process start failure recorded in infrastructure settlement;
- injected export failure retaining native evidence without worker restart;
- injected supervisor interruption producing a stopped supervisor and no restart.
- combined staging wall/CPU enforcement and total CPU accounting that includes
  the supervisor worker subtree;
- export of `export-status.json` inside the same verified manifest, with finite
  combined wall/CPU evidence.

The successful fake worker records zero environment calls, zero model calls,
zero training updates, one worker invocation, 12 completed SQLite call rows,
zero pending rows, and concurrent reader coverage.

The current-package native-infrastructure suite is also run against this
package under the frozen Linux Python. Its 12 checks cover concurrent atomic
readers, bounded fsync failure, status classification, interrupted and failed
workers, unresolved ledger preservation, initialized-root staging, verified
export, and appending final settlement to the verified export manifest. The
separate legacy output is retained only as historical evidence and is not
counted as validation of current source.

The final local run contains 41 passing current-package checks (31 schema,
pipeline, freeze, and zero-call checks plus 10 repaired-launch checks). The 12
current native-infrastructure checks also pass. All integration workers report
zero environment calls, model calls, checkpoint loads, and training updates.

Passing these tests establishes launch wiring and failure accounting only. It
does not authorize or execute the learning experiment.
