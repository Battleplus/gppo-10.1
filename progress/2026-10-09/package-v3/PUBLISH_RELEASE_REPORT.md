# Linux Recovery Release Report

This release is an independent publish copy of the recovery package. The
previous recovery identity and the first Linux release identity are
superseded for any future authorization; both packages, the first release's
preflight result, and its controlled-acceptance first error are retained.

The release excludes every `__pycache__` directory and every `.pyc`/`.pyo`
file. `launch_pilot.verify()` rejects such entries if they appear in a staged
publish tree. Python source files and runtime assets remain part of the frozen
inventory.

The Linux controlled acceptance exposed an indentation error in the prior
release's schedule-only callback (`NameError: seed` before callback
completion). This release fixes only that acceptance callback so its 8202 and
8203 update schedule and checkpoint accounting execute inside the callback.
The production trainer and research path are unchanged.

Recovery scope is unchanged: reuse all 288 cached windows and world model
8201; train world models 8202 and 8203; do not recollect. This is not an
eight-policy reuse run and not a G1/8303-only training run. The formal stages
remain data admission, 8201 reuse, 8202/8203 training, three-model recovery,
the unchanged offline gate, conditional nine-policy GPPO/critic training,
240 task episodes, analysis, settlement, and export. Gate failure still
requires zero policy updates and zero task episodes.

Scientific settings, split, seeds, model contracts, and all resource ceilings
are unchanged from the prior recovery request. Only the attempt binding and
published file inventory differ. Linux engineering acceptance is kept
separate from formal research consumption.

## Source and asset comparison

The release audit records exact comparisons against the prior recovery copy:
all `.py` source files, non-bytecode configuration and contract files, all
288 compressed window files, and `world-8201.pt`. Any source differences are
listed explicitly in `release-comparison.json`; bytecode cache removal is
reported separately. This release has two source differences from the
original recovery package: `launch_pilot.py` enforces bytecode rejection and
`package/acceptance/recovery_path_acceptance.py` repairs the test callback.

## Identity

The generated Manifest, Hashes, and Request digests are in
`frozen-identity.json`. A new one-time authorization must bind all three.
This package does not create an authorization, token, attempt directory, or
hosting receipt.
