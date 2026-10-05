# WSL runtime dependency repair validation

Date: 2026-09-29

## Root cause and preserved evidence

The sealed attempt `w1-action-relative-auxiliary-production-backend-completion-v1-once`
stopped in `staging_and_zero_step_gate` with
`ModuleNotFoundError: No module named 'torch'`. Its settlement reports an empty
dynamic-call ledger, zero pending calls, zero failed calls, and zero task calls.
The old attempt was not reused or edited.

- old `status.json` SHA-256: `0d17e0595d223a96d0ee452c88bd97f8f68264be9c210d71a5d29a7f8494023a`
- old `settlement.json` SHA-256: `7dc7d33f90ec97cd4f99bfa094a5c808717aa83cca868405090eef46d1e443d3`

## Frozen repair identity

- package: `E:\Z博士\research-plans\w1-action-relative-auxiliary-runtime-dependency-repair-v1`
- attempt: `w1-action-relative-auxiliary-runtime-dependency-repair-v1-once`
- execution manifest SHA-256: `6a74e7ee3896e7ede69834b0bc07d13461065775f0b6903a62e910630ff72e5e`
- hashes SHA-256: `f5fa342ed392d1aacc7ebff1a64ebac4a69a44d6e98cfd98b27b22a23fc77880`
- resource request SHA-256: `2ba32f0d4db23aa62ff8c038771693904c4cab135cfb208d70e628a65e077732`
- request status: `NOT_APPROVED`

All frozen content file hashes were recomputed after the validation attempt;
the mismatch count was zero.

## Runtime identity

- WSL distribution: `Ubuntu-24.04`
- raw interpreter: `/home/asus/w1-action-relative-auxiliary-runtime-dependency-repair-v1-runtime/bin/python`
- `sys.prefix`: `/home/asus/w1-action-relative-auxiliary-runtime-dependency-repair-v1-runtime`
- base interpreter: `/usr/bin/python3.12`
- Python: 3.12.3
- PyTorch: 2.8.0+cpu
- NumPy: 1.26.4
- CUDA build/runtime: absent; `torch.version.cuda=null`, `torch.cuda.is_available=false`, `USE_CUDA=0`
- CPU kernel witness: finite 4x4 matrix multiplication, checksum 3680.0
- dependency lock SHA-256: `725b14265d729fb8ac33cde1c9e5686e4638966d0d35244fed0d703c59d60e56`

A negative probe using `/usr/bin/python3` was rejected with
`PYTHON_PREFIX_MISMATCH:/usr`, proving that the contract does not silently use
the system interpreter.

## Read-only preflight

Command:

```powershell
python E:\Z博士\research-plans\w1-action-relative-auxiliary-runtime-dependency-repair-v1\launch_once.py --authorization-file E:\Z博士\.codex-tmp\w1-action-relative-auxiliary-runtime-dependency-repair-v1.authorization.json --preflight-only
```

Exit code: 0. Linux output reported `preflight_pass`, `staging_started=false`,
and `worker_started=false`. The package contained 54 files before and after;
the complete file-identity comparison reported no change.

## Authorized zero-step validation

Command (the token value remained outside logs and the package):

```powershell
python E:\Z博士\research-plans\w1-action-relative-auxiliary-runtime-dependency-repair-v1\launch_once.py --attempt-token-file E:\Z博士\.codex-tmp\w1-action-relative-auxiliary-runtime-dependency-repair-v1.token --authorization-file E:\Z博士\.codex-tmp\w1-action-relative-auxiliary-runtime-dependency-repair-v1.authorization.json --dependency-only
```

Exit code: 0. The native entry reported:

- worker started: false
- environment constructed: false
- model initialized: false
- checkpoint loaded: false
- training started: false

Every dynamic counter in `zero-step-settlement.json` is zero: environment
steps, resets, candidate branches, rule decisions, model initialization/load,
optimizer updates, batch forwards, model sample evaluations, and checkpoint
writes. No `run-once`, execution lock, SQLite database, checkpoint file, or
checkpoint directory exists in the native attempt.

Native filesystem type is ext2/ext3 as reported by WSL `stat -f`, rather than a
Windows mount. Measured infrastructure use was 4.978429495 wall seconds and
2.111641172 complete-process CPU seconds. The native dependency child used
1.03142 CPU seconds.

## Controlled export

- export path: `E:\Z博士\.codex-exports\w1-action-relative-auxiliary-runtime-dependency-repair-v1-once`
- final files: 61
- final export manifest SHA-256: `5bbd92d7468d33c295e2d76cb7d2dbcddbf6d5fdbdbe8954aa9c4a3c66fd88b1`
- export manifest verification: 61/61 files matched, zero mismatches
- dependency probe SHA-256 in native and export: `ccf3ae726c0d1610c408c07ed39781aab0260d014fd748ea1aeeadd5e9b1a869`
- zero-step settlement SHA-256 in native and export: `98aa5ef921821a47af8a7efcc512e14a52defd6b4bfe1f256bf0cceeb363f03c`

This validates the repaired dependency and cross-system zero-step launch path.
It does not run or validate label collection, model training, prediction gates,
or task comparison. The consumed zero-step attempt must not be reused for a
later dynamic experiment.
