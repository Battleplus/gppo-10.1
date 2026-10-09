# v5 Recovery Staging / Preflight Block

Date: 2026-10-08

This is an external staging record. The frozen recovery package, its hashes,
the old v5 deployment, and all prior evidence were left unchanged.

## Identity

- Attempt: `w1-drone-action-consequence-world-model-v5-recovery-v1-once`
- Manifest: `60827735c0bffcae269bd733c54d192d1b55a49729c2022bd63d3478ec124dab`
- Hashes: `071595cea23d106f4dbe97328d698f01da806134212d7406ec1bb6aa0a9d0246`
- Request: `f90cf3c82a157f4017ed69a8d4ece0234dd7fc3f8a2ac277c99771b7f104d9ea`

## Remote staging

- Host: `user1@172.17.27.173` (`ru-server-MS-1`)
- Staging directory: `/home/user1/w1-pilot/staging/w1-recovery-v1-60827735c0bffcae269bd733c54d192d1b55a49729c2022bd63d3478ec124dab`
- Files in staging: 489
- `execution-manifest.json`: SHA-256 matches the Manifest above.
- `hashes.json`: SHA-256 matches the Hashes value above.
- `RESOURCE_REQUEST.json`: SHA-256 matches the Request value above.
- `frozen-identity.json`: binds the Attempt above and records `formal_attempt_created: false`.
- No authorization directory or `package/run-once` directory was present in staging.

## Blocking mismatches

The two files below are listed by the frozen `hashes.json`, but their exact
frozen bytes are unavailable in local archives and on the server:

| Path | Frozen bytes / SHA-256 | Staged bytes / SHA-256 |
|---|---:|---:|
| `package/__pycache__/production_driver.cpython-314.pyc` | 77542 / `fd209dc07ed5a7d76a4eea133302eee603adfd913794dc0fb6d1d02de14fedb9` | 78025 / `4d8bc20a94225af9926983453c01949625a14825ebf85bb71292638146e2fa11` |
| `package/acceptance/__pycache__/recovery_path_acceptance.cpython-314.pyc` | 7085 / `eadcb6679314b1d7a50df80c2c6a9e3d1f2d6fd29a3b650ec57a32840ec04b28` | 8374 / `e56945ce2a1d8034e512a6fd6b7ef39cf60811968865a587bf622270fbd9fcfb` |

The source files match their frozen hashes. The server's old deployment only
contains `cpython-311` bytecode, so it cannot satisfy these `cpython-314`
entries. Removing the entries, changing `hashes.json`, or substituting other
bytecode would change or bypass the frozen identity and was not done.

## Execution decision

- Atomic deployment switch: **not performed**.
- Linux `--preflight`: **not run** because complete file admission failed.
- Token, authorization file, attempt, hosting receipt, and formal run: **not created**.
- Old v5 deployment and evidence: **preserved**.

The recovery package remains blocked until the exact two frozen byte streams
are recovered or a separately authorized new identity is issued.

## Read-only runtime observation

The requested interpreter exists and reports Python 3.11.16, PyTorch
2.7.0+cu128, and NumPy 1.26.0. A direct diagnostic invocation reported CUDA
available and a 32-CPU affinity with 16 Torch threads; this was not a formal
run and no process settings were changed. CPU0/single-thread compliance
therefore remains **not verified** by the blocked preflight.
