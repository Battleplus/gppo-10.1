# Filesystem contract

The Windows entry performs identity and token checks, converts the source with
`wslpath`, and invokes `Ubuntu-24.04` with an argument array. It never resolves,
creates, copies, or fsyncs `/home` paths. The Linux entry validates its actual
OS, interpreter, source mount, target mount, and frozen path identities before
copying the first package file. A Windows preflight exception emits a structured
technical-stop record to stderr before returning nonzero; it cannot create an
attempt because authorization and identity have not passed.

An authorized execution stages the complete sealed package to
`/home/asus/w1-runs/w1-action-outcome-learning-loop-launch-repair-v1-once` and keeps all
active state, the SQLite call ledger, labels, checkpoints, reports, and
temporary files on that native ext4 path. The Windows destination
`/mnt/e/Z博士/runs/w1-action-outcome-learning-loop-launch-repair-v1-once` is created only by
the Linux launcher after the worker exits and the supervisor settles. Export is
single-shot and verifies SHA-256 per file. Export failure preserves the native
evidence and never restarts the worker.

Only staging, atomic state publication, settlement, and export file operations
may use the inherited bounded three-attempt/two-second retry. Environment
steps, resets, branches, model calls, optimizer updates, and task episodes are
never retried.

The Linux launcher writes `launcher-status.json` before the first package copy.
It includes Windows preflight time in the staging gate. Its total wall check
combines that preflight with the Linux entry lifetime; its total CPU check adds
Windows preflight process CPU, Linux launcher process CPU, and the supervisor's
complete-process CPU for the worker subtree. `export-status.json` records these
combined values and is appended to the same verified export manifest. The
Windows wrapper also enforces and prints its full outer wall interval. Windows
post-preflight process CPU and the cross-process start gap are unavailable to
the Linux child and remain explicitly marked unavailable.
