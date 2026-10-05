# v6 CPU Accounting Closure

The frozen CPU budget is unchanged. The authoritative complete-process CPU
measurement remains the WSL staging process's `RUSAGE_CHILDREN` delta for the
native launch process and all waited descendants. It is charged once.

The native entry and supervisor run in the same Linux process. Their values are
therefore nested diagnostics, not additive budget components:

- `supervise.py` records `RUSAGE_SELF + RUSAGE_CHILDREN` after worker reap and
  process-group cleanup, before its final artifact scan, cumulative-history
  read, and final status write.
- `native_launch.py` records its own `RUSAGE_SELF` and reaped-child values
  after `supervise.main` returns, before writing
  `native-launcher-accounting.json` and exiting.
- The parent WSL launcher records the outer `RUSAGE_CHILDREN` value after the
  native process is waited or forcibly stopped.

The exported `w1-native-cpu-scope-breakdown/1.0.0` record decomposes the
outer-minus-supervisor difference into the supervisor-to-native-launcher
snapshot interval and the native-launcher accounting-write/exit tail. The
existing 0.05-second tail check is retained; it is an evidence consistency
check, not additional budget or permission to widen a limit. Nested values are
never added to the authoritative outer total.

The v5 record contained only the outer value (`2.811616` seconds) and the
supervisor snapshot (`2.536691` seconds), with no timestamps or launcher
snapshot. Its `0.274925`-second difference therefore cannot be assigned to
startup, worker, settlement, export, or process exit after the fact. v6 records
those boundaries for future runs and fails closed if the decomposition is
missing, inconsistent, or exceeds the frozen tail tolerance. Exceptions,
timeouts, and export failures retain measured CPU evidence and do not retry.
