# Current joint preparation status

This directory is an unaccepted development package, not a frozen executable
research release. `RESOURCE_REQUEST.json` remains `NOT_APPROVED` and
`runner_ready=false`. The development package identities were regenerated after
the current source and provenance changes; they are valid for this development
snapshot but are not an execution approval.

Development snapshot identities are recorded in the package-external
`diagnostic-work/w1-joint-preparation-local-evidence-20261002/delivery.json`.
This document cannot include the digest of the manifest that hashes it.

The scope is complete-sequence label collection and data admission followed by
six G1/G2 training/selection routes, checkpoint restoration, and held-out-parent
prediction evaluation. GPPO training and task comparison are excluded. Existing
research results, consumed attempts, authorizations, and tokens are not reused.

## Static corrections on 2026-10-02

- Lifecycle validation now appends every valid task row inside its loop, so
  before/after identity and completion-time checks inspect the actual records.
- Worker success and failure reports use completed versus failed/pending ledger
  calls for learning activity. A failed call without a completed predecessor
  yields unknown activity rather than falsely claiming no model activity.
- Settlement takes an explicit output directory. The integration fixture cannot
  write settlement into the original development package through a module global.
- Configuration generation and the current matrix explicitly allocate thirteen
  continuation decisions after the first action, within the existing fourteen
  branch-step bound and unchanged proposed call budget.
- The collector accepts a separate, authenticated 2/2/8, one-epoch synthetic
  profile for integration testing. The proposed research matrix remains 24/8/8.
- The earlier max-of-overlapping-snapshots monitor was found to undercount a
  mixture of live and unobserved reaped child CPU. It is superseded by the
  fail-closed live-delta/reaped implementation below; the earlier tests are not
  proof of complete CPU accounting.
- Registration identity is now checked before SSH/password or remote probes;
  the Windows entry derives a deterministic lowercase `name_id` from a Chinese
  name (or reuses a lowercase English name). State transitions use a per-ledger
  transition lock, and the remote RUNNING update binds the actual runner PID in
  addition to the supervisor wrapper. Synthetic Linux registration and proc
  fixtures have executed; actual server `/proc`, GPU and SSH paths have not.

The local identity, CPU-contract, and package-contract regressions for these
changes have executed successfully. They do not establish acceptance of the
native Linux `/proc`/GPU/SSH path, the complete production pipeline, or the
formal GPU matrix.

## Additional production-path repairs on 2026-10-02

- The native remote entry now records a fail-closed `technical_stop` settlement
  with self CPU, reaped-child CPU, and wall evidence on initialization, worker,
  binding, or export exceptions. Settlement timing includes the first final-status
  write, and the final resource check runs before terminal registration so an
  over-budget run cannot be registered as `SUCCEEDED`.
- GPU jobs wait for the registered root/supervisor/runner PID set to own exactly
  the requested GPU set before the second `RUNNING` event. A runner or GPU
  binding deadline is recorded as `EXPIRED`; an empty binding is rejected when a
  GPU worker was requested. Registration also enforces the deterministic
  `derive_name_id(real_name)` identity for direct callers.
- CPU supervision charges disjoint live-process tick deltas plus reaped child
  CPU. A disappeared process identity whose ownership cannot be proven is a
  fail-closed technical stop, and a final resource overrun changes supervisor
  status from `complete` to `stopped` with a nonzero return.
- Synthetic test copies exclude dynamic attempt directories and files. Linux
  tests use production collection, training, checkpoint and prediction code
  with a bottom-environment fixture, CPU tensors, one epoch and a 2/2/8 matrix.
  Synthetic model/checkpoint calls are nonzero and recorded outside the package.
  No real W1 environment, SSH workload, research model/checkpoint or formal
  attempt was started.
- The launcher propagation unit test now isolates its accounting output in a
  temporary directory. Earlier residue is preserved outside the development
  package. The tail-mismatch fixture keeps its charged-once total consistent so
  it reaches the intended settlement-tail rejection check.
- The documented Windows command uses the frozen SSH-controller interpreter.
  Local structural preflight is explicitly distinguished from the remote probe.

## Acceptance still outstanding

1. Verify the newly wired server registration call sites on native Linux. The
   v1.2 rule requires the user's real name, locked REGISTERED append/readback,
   and a current RUNNING PID/PGID/runner-PID/GPU binding. No server process was
   run in this continuation.
2. Finish native process integration and verify the fail-closed CPU scope on the
   actual server. The local synthetic CPU and settlement regressions are green,
   but no server process has been run in this continuation.
3. The local CPU synthetic pipeline has run. The target-server GPU runtime,
   actual registration, Windows SSH entry and GPU benchmark remain unverified.
4. Verify source provenance, full return reconstruction, unknown/mask behavior,
   parent separation, data-gate stopping, persisted trace metric recomputation,
   phase budgets, export identities, and failure paths on the final source.
5. Derive defensible learning time limits from production benchmark evidence;
   the current learning time proposal is not yet a final executable application.
6. Finish docs, tests, formal freeze, external NOT_APPROVED binding, and original
   directory read-only preflight in that order. Evidence stays outside the package.

## Newly observed dependency evidence

The dependency-closure regression initially rejected `transparent_utility.py`:
`runtime-inputs.json` froze the earlier derived-file digest, while the current
production code also exposes `transparent_horizon_components`. The package now
keeps the earlier helper as `native/source-evidence/transparent_utility_base.py`,
compares the reused function by AST, and records the new helper plus its
equivalence test as a package-local extension. The staged-module check and its
pre-construction import-error regression now pass after this provenance repair.

The shared input validator now checks portable source evidence from this package
instead of borrowing an absolute historical workspace. Both joint preflight and
the dependency closure probe call it; it still checks manifest membership, tape,
source modules, derived functions and equivalence-test digests. The Linux CPU
probe is a local preparation check, not verification of the remote CUDA runtime.

## Explicit remaining accounting blockers

- Terminal registration runs after the last native CPU/wall sample. Its cost and
  the final failure-persistence tail still require an outer measured scope or
  an enforceable reserve before formal acceptance.
- CPU supervision rejects identities without a provable baseline or reap
  relationship. This is conservative and may stop an ordinary subprocess path.
  A process that is neither sampled nor reaped yet is not measured by the live
  probe; final waited accounting does not prove a continuous stage cap.
- GPU binding currently polls the process set. Live PID/start-time validation
  is performed on append, not continuously during polling. The target-server
  handshake and its timeout need production integration coverage.

These are engineering blockers, not world-model research conclusions.

All root historical docs and old Windows/WSL tests copied from v6 are reference
material; they cannot prove this Windows SSH / server-native joint launch ready.
No formal collection/training authorization is presently approved.
