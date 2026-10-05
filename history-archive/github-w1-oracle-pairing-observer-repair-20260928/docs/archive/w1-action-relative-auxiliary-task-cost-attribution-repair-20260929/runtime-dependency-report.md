# Runtime dependency repair

The prior attempt stopped at `staging_and_zero_step_gate` because the formal
Ubuntu-24.04 `/usr/bin/python3` could not import `torch`. Its ledger contained
zero dynamic calls and its stopped attempt remains sealed.

The repaired package uses this WSL-native interpreter:

`/home/asus/w1-action-relative-auxiliary-runtime-dependency-repair-v1-runtime/bin/python`

It is a Python 3.12.3 virtual environment on native ext4, with system site
packages visible only for the already-installed scientific runtime libraries.
The installed PyTorch is `2.8.0+cpu`, built with `USE_CUDA=0`; no CUDA runtime is
required. numpy is `1.26.4`. The direct and transitive package versions are in
`runtime-dependency-lock.txt`.

Installation source was the CPU-only PyTorch wheel index
`https://download.pytorch.org/whl/cpu`, performed before freezing and outside
the formal experiment. The formal entry never installs or downloads packages.

The dependency probe imports `torch` and numpy, records the interpreter and
build information, and asserts CPU-only operation. It does not initialize a
model, load a checkpoint, call an environment, or train.

The old stopped evidence remains unchanged:

- `status.json` SHA-256: `0d17e0595d223a96d0ee452c88bd97f8f68264be9c210d71a5d29a7f8494023a`
- `settlement.json` SHA-256: `7dc7d33f90ec97cd4f99bfa094a5c808717aa83cca868405090eef46d1e443d3`
- ledger totals: empty; pending calls: 0; failed calls: 0; task calls: 0

The new zero-step route is `Windows entry -> WSL dependency probe -> native
staging -> native dependency probe -> zero-call settlement -> controlled
export`. It passes `--dependency-only` to `native_launch.py`, which returns
before importing the supervisor or runner. The resulting attempt is evidence
for dependency and launch plumbing only. It does not authorize or establish a
successful learning experiment.
