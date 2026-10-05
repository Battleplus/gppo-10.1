# Budget Accounting Erratum: v4 to v5

The v4 preparation package had correct arithmetic totals, but its implementation
did not measure the whole startup and settlement path within those totals. The
v4 package and its preflight evidence remain unchanged. This independent v5
package repairs measurement and enforcement only; it does not change the label
matrix, parent identities, task semantics, or dynamic limits.

## Root causes

- Linux staging timing began after Linux authorization, package-hash, filesystem,
  and runtime checks. The 20-second staging/zero-step allocation therefore did
  not cover the full path before supervision.
- The Linux exporter added the WSL parent's `RUSAGE_CHILDREN` CPU to the
  supervisor's `RUSAGE_SELF + RUSAGE_CHILDREN` CPU. `native_launch.py` calls the
  supervisor in the same process, and Linux child usage includes waited-for
  descendants when each intermediate process reaps them. Those two values
  describe the same process tree and must not be added.
- Export-stage CPU was sampled before writing and appending the final status
  artifact, so the recorded total omitted part of its own settlement.
- The Windows launcher waited on `wsl.exe` with unbounded `communicate()`, so
  the 753-second wall cap was checked after return but did not stop a blocked
  launcher.

## v5 accounting

- The read-only Windows preflight plus its measured WSL helper process tree is
  counted in the 20-second staging allocation. Linux module startup, contract
  validation, runtime probing, and native staging are counted before the worker
  may start. The stage has a Linux alarm and a post-stage wall/CPU check.
- The outer WSL launcher's `RUSAGE_CHILDREN` value is the one authoritative
  complete CPU measurement for the native launcher, supervisor, worker, and
  waited descendants. The supervisor value is retained only as a consistency
  check with a 0.05-second tolerance.
- Settlement timing begins before the settlement record and covers verified
  export, status writes, and status append. The status stores an upper bound
  using the unused remainder of the 39-second stage cap; the launcher then
  checks actual stage use before returning.
- `run_native()` has the frozen label-stage timeout and terminates its process
  group on timeout. The Windows `wsl.exe` wait is bounded by the remaining
  global wall allowance after reserving two seconds for cross-system return.
  There is no retry path.

## Budget effect

Global limits are unchanged: 2,832 environment steps, 8 resets, 200 branches,
753 wall seconds, and 551 complete-process CPU seconds, with the same memory and
storage ceilings. CPU stage allocations are rebalanced from
`20 + 491 + 38 + 1 native-launcher reserve + 1 Windows reserve` to
`20 + 491 + 39 + 1 Windows reserve`. The old native-launcher reserve was
double-counting CPU already included in the process-tree measurement; that one
second is moved to settlement/export to include final accounting writes. Wall
allocations are unchanged at `20 + 693 + 38 + 2` seconds.

The 0.05-second counter tolerance and finalization upper bound are accounting
checks only. They do not permit extra environment calls or borrowing between
stages. `RESOURCE_REQUEST.json` remains `NOT_APPROVED`.
