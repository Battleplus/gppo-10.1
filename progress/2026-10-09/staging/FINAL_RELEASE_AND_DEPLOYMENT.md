# W1 v5 Recovery Release and Linux Deployment

Date: 2026-10-08 (Asia/Shanghai)

This report covers release identity `w1-drone-action-consequence-world-model-v5-recovery-v1-linux-release-v2-once`. It does not authorize or start formal research.

## Package comparison

- The original recovery package and both earlier server directories remain preserved. Release v1's failed controlled-acceptance log is retained separately; its callback failed with `NameError: seed` before its schedule accounting completed. Release v2 fixes that acceptance callback in an independent copy.
- The v2 publish tree contains no `__pycache__`, `.pyc`, or `.pyo`. The launcher rejects those paths before inventory verification and omits them when staging a future authorized run.
- All 81 Python source paths are present. Relative to the original recovery package, the only source differences are:
  - `launch_pilot.py`: reject bytecode caches and omit them from staged copies.
  - `package/acceptance/recovery_path_acceptance.py`: keep schedule, checkpoint accounting, and returned schedule metadata inside `scheduled_train`.
- The original package had 40 `.pyc` files; every one has a corresponding source module. No bytecode-only required module was found.
- All 288 window-cache files match byte for byte (25,549,852 compressed bytes). The window manifest, parent split, 8201 binding, 8201 checkpoint, environment configuration, candidate contract, public-prefix contract, and flat-layout contract all match byte for byte. Checkpoint SHA-256: `ebd9894270d6a32edec6892a912d11933c1a266c1a831294388ffb06be52d8b0`.
- `RESOURCE_REQUEST.json`, after normalizing the attempt field, is identical to the prior request. Scientific settings and every resource ceiling are unchanged.

Comparison evidence: `release-comparison.json` in the v2 publish package.

## Linux targeted acceptance

Ran with the existing Linux Python 3.11.16 runtime under CPU0 affinity, one-thread environment settings, `CUDA_VISIBLE_DEVICES` empty, and `-B`/`PYTHONDONTWRITEBYTECODE=1`.

- Staging identity/preflight: pass; bytecode-injection negative check: rejected as required.
- Recovery driver with real `BudgetLedger` and real `StageServer`: pass.
- Input admission: 288 reused windows; 174 complete and 114 `no_opportunity`; collector not called.
- Seed 8201: production checkpoint admission path accepted the bound file/metadata/state/optimizer identity.
- Seeds 8202 and 8203: each scheduled 40 epochs over 58 train windows, batch 8, 320 updates; 640 scheduled updates total.
- Ledger settlement: pass; 1,477 candidate-contract audits, 3 checkpoint-load reservations, 3 checkpoint-write reservations, and 20,560 candidate target/training evaluation reservations each.
- Deliberately under-budget admission: rejected before its operation; ledger had zero pending and failed calls.
- Model forwards: 0; formal environment steps: 0; policy updates: 0; task episodes: 0.

This is a schedule and recovery-wiring acceptance, not model training. The callback replaces the trainer and the three-model restore is intercepted to verify seed bindings; it does not prove the 8202/8203 tensor training or the final strict three-model restore. Those remain formal-run work. The acceptance output does not contain an independent CPU or wall measurement; engineering CPU/wall are recorded as unavailable rather than inferred. Ledger schedule amounts are test reservations, not formal research consumption.

Server acceptance and deployment receipts are under `linux-release-v2-evidence/v2/`.

## Freeze and deployment

- Manifest SHA-256: `d10922e7d439fae4b14cb48dbb62ef40cd139e8c7df141e86260fd0cde07e1a2`
- Hashes SHA-256: `5b72e995e46d214d73725ad59c69cf26f2576854466af69b81991ce8b12cb536`
- Resource Request SHA-256: `5af24201a1235279c79f1b54a24c0f32424961be1e0272b47c536bec10f6b562`
- Frozen identity file SHA-256: `6b0de54349bff46cd935c41c0b3d8896a523129e2b99c4f5ff1ebd1d1d3ba606`
- Server deployment: `/home/user1/w1-pilot/packages/w1-drone-action-consequence-world-model-v5-recovery-v1-linux-release-v2`
- Full inventory verification and deployed Linux `--preflight`: pass. Old recovery deployment and v1 staging remain present; v2 staging was atomically renamed into its new package directory. The deployed tree remains bytecode-free.
- The preflight reports `NOT_APPROVED`, `attempt_unconsumed: true`, and `research_started: false`. No token, authorization file, run directory, or hosting receipt exists for this identity.

## Recovery scope and budget

The formal recovery remains: reuse all 288 windows and model 8201; train 8202 and 8203; strict three-model recovery; unchanged offline prediction gate; only if it passes, train the nine conditional GPPO/critic policies and evaluate 240 episodes. This is not eight-policy reuse and not a G1/8303-only run. No recollection occurs.

Unchanged request totals: 22,752 environment steps (18,432 strategy training + 4,320 evaluation), 640 world-model updates, 1,152 policy updates, 240 episodes, 23,000 wall seconds, and 12,000 CPU seconds. Existing memory, storage, stage, and transport limits remain as frozen in `RESOURCE_REQUEST.json`.

## Unique entry

Preflight:

```sh
/home/user1/w1-runtimes/w1-py31116-torch270cu128-v2/w1-light-repaired-fair-rerun-py31116/bin/python -B /home/user1/w1-pilot/packages/w1-drone-action-consequence-world-model-v5-recovery-v1-linux-release-v2/launch_pilot.py --preflight
```

Formal execution requires a new external authorization bound to the three v2 digests. Its unique entry and authorization path are in the deployed `unique-launch-command.md`. No formal authorization was created during this work.
