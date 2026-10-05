# W1 G1/G2 Joint Production Acceptance v2

This is an unaccepted joint development snapshot. `RESOURCE_REQUEST.json` is
`NOT_APPROVED` and `runner_ready=false`. The active proposed scope is label
admission followed by G1/G2 world-model training and prediction; GPPO training
and task comparison are excluded from this proposal. See `DEVELOPMENT_STATUS.md`
for inherited evidence. Current changes and acceptance status are in
`ENGINEERING_ACCEPTANCE.md`; `SERVER_ACCEPTANCE_REQUEST.json` requests only
bounded, separately approved server engineering work. No acceptance result is
implied by freezing this directory.

The local integration profile uses synthetic bottom-environment interactions,
CPU model computation, one epoch, and 2/2/8 parents. It is not the formal remote
GPU matrix and its nonzero model, optimizer and checkpoint calls are counted
separately. The only proposed formal entry is `launch_joint_once.py`; there is
no current training authorization or registration identity.

The material below is inherited v6 historical reference, including its old
budgets, counts and entry. It does not describe this joint proposal or establish
that the new server production entry is ready.

## Historical V6 Reference

This is an independent preparation package for one finite, label-only
qualification attempt. `RESOURCE_REQUEST.json` remains `NOT_APPROVED`. This
work did not start an attempt or run the real environment, a model, a
checkpoint or training. Dependency checks execute synthetic tensor witnesses;
these are preparation checks and must be reported separately from the zero
research-model forward count.

v6 carries forward the v5 label contract, fixed parent matrix and global limits.
The collection-only pure transparent utility conversion is isolated from the
policy-training module, native project sources are included in the staged
closure, and CPU snapshots have explicit scopes and sampling timestamps.
The old v5 attempt and its failed export are preserved. The v5 0.274925-second
CPU discrepancy cannot be attributed retrospectively because the necessary
timestamps were not retained; it is not labelled measurement noise. See
`cpu-accounting-erratum-v6.md` and the v6 change report.

The global limits are unchanged: 2,832 environment steps, 8 resets, 200
branches, 753 wall seconds, 551 complete-process CPU seconds, 2 GiB RSS, 512
MiB active storage, and 1 GiB native plus verified-export storage. Model,
checkpoint, forward, optimizer, and task-comparison calls remain zero.

The contract labels physical on-time arrival and expiry separately from host
confirmation for a legal first action followed by the fixed Hungarian
continuation. Host confirmation remains unknown in the current environment.
The frozen zero-radius arrival objective is an exact point target. See
`TASK_OUTCOME_LABEL_CONTRACT-v4.md`, `configuration-identity-audit.md`,
`source-erratum.md`, and `data-reuse-audit.json` for semantics, provenance, and
the historical-source correction.

Run the production collector lifecycle and zero-dynamic accounting tests on
Ubuntu-24.04 with the pinned native CPU Python:

```text
/home/asus/w1-action-relative-auxiliary-runtime-dependency-repair-v1-runtime/bin/python -B -m unittest discover -s . -p 'test_*.py' -v
```

The lifecycle tests use a deterministic fake environment and native Linux
temporary storage. They verify production collection and persistence wiring,
not real label coverage or model/task effects. Accounting tests use synthetic
OS processes and controlled launcher doubles; no real W1 environment is built.

The final package regression run recorded 65 tests with exit code 0. Focused
dependency/launcher checks recorded 38 passing tests, and CPU scope plus
runner checks recorded 19 passing tests. The Windows-entry isolated collector
fixture completed one deliberately reduced parent through production freeze,
native staging, collector, SQLite settlement, label reread, and controlled
export using only a lower-environment fake. It is not the eight-parent
qualification run. A separate pending-ledger fixture exits `technical_stop`
and creates no controlled export.

The remote GPU runtime audit is a preparation record for a future independent
training package. It does not change this package's Windows-to-Ubuntu-24.04 CPU
launch route, zero-model budget or NOT_APPROVED status. A training package is
prepared only after separately approved label qualification establishes valid
task-outcome coverage; no historical dynamic approval transfers here.

The single future launch entry is `launch_once.py`; the read-only Windows to
Ubuntu-24.04 command is in `unique-launch-command.md`. `--preflight-only` checks
package identity and dependencies without staging, creating an attempt, lock,
or SQLite file, reading a token, or starting a worker. A formal run requires a
new explicit approval and an external one-time token bound to the final
package.

Freeze only after all source and report files are final. `freeze_contract.py`
refuses an unmapped Linux-native source path:

```text
python -B freeze_contract.py
python -B freeze_contract.py --verify
```

`hashes.json` covers every payload file except itself and
`execution-manifest.json`; `hashes.json` records the manifest digest. External
authorization binds both outer digests and the resource request. The request
stays `NOT_APPROVED`.
