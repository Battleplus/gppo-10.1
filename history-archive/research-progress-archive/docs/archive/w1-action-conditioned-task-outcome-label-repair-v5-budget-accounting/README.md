# W1 Action-Conditioned Task Outcome Label Qualification v5

This is an independent, frozen preparation package for one finite, label-only
qualification attempt. `RESOURCE_REQUEST.json` remains `NOT_APPROVED`. This
work did not start an attempt or run the real environment, a model, a
checkpoint, a forward pass, or training.

v5 carries forward the label contract and fixed parent matrix from v4. It fixes
resource measurement only: Windows and WSL read-only preflight time is included
in staging; Linux authorization and staging are timed before supervision; the
native process tree CPU is counted once; export settlement is bounded; and the
Windows WSL wait has a timeout. See `budget-accounting-erratum-v5.md` for the
v4 root cause and the one-second CPU reallocation. v4 remains unchanged.

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
OS processes and controlled launcher doubles; no W1 environment is built.

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
