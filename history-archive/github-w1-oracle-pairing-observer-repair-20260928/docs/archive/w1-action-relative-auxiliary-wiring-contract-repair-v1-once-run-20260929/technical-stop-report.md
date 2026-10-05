# Technical stop report

Attempt: `w1-action-relative-auxiliary-wiring-contract-repair-v1-once`

The approved Windows entry verified the package and staged all 49 package
files into WSL native ext4 at `/home/asus/w1-action-relative-auxiliary-wiring-contract-repair-v1-once`.
The supervisor started the worker process, but the worker stopped before any
environment or model call. `runner.py` compares `RESOURCE_REQUEST.json` with a
hard-coded attempt value `w1-action-relative-auxiliary-wiring-v1-once`; the
approved request uses `w1-action-relative-auxiliary-wiring-contract-repair-v1-once`.
It therefore raised `RuntimeError: frozen request identity changed`.

This is a frozen-entrypoint contract defect. No live repair or retry was made.
The run is classified as `technical_stop`; labels, training, prediction
evaluation, task comparison, and research gates are `not evaluated`.

Observed dynamic usage before the stop:

- environment steps, reset, branches, rule decisions: 0;
- model initialization, batch forward, sample evaluation, optimizer updates,
  checkpoint writes: 0;
- Windows preflight CPU: 0.140625 s;
- Linux launcher CPU: 0.184079185 s;
- supervisor complete-process CPU: 0.208206 s;
- export CPU: 0.089314713 s;
- combined reported CPU: 0.529277123 s;
- combined reported wall: 2.6485621509982593 s;
- staging bytes: 231692; exported payload: 234847 bytes;
- peak RSS upper bound: 55,488,512 bytes.

The controlled export and settlement were verified. The raw traceback,
launcher/supervisor status, staging record, resource history, export manifest,
and hashes are retained in this directory. The stale `launcher-status.json`
snapshot reports `running`, while the later supervisor and settlement records
authoritatively report `stopped`, `returncode=1`, and `worker_relaunched=false`.

No result supports or rejects the A/B prediction gate or the four-arm task
comparison. The approved attempt must remain stopped until a separately
reviewed package repair is authorized.
