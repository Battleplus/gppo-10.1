# Independent luna_worker review

The reviewer executed the documented `--preflight-only` command without a
token. It returned `preflight_pass` and WSL return code 0. The Linux record
reported `staging_started=false` and `worker_started=false`.

Static review confirmed that `--dependency-only` stages the frozen package,
runs `native_launch.py`, returns before the supervisor import, writes zero-call
settlement evidence, and performs controlled export. It cannot reach runner,
environment, model, checkpoint, or training code. The subsequent end-to-end
zero-step execution is recorded separately in `validation-report.md`.

The Windows preflight summary reports worker state as `unknown`, although the
same Linux preflight record reports `false`. A local top-level `__pycache__` is
outside the frozen manifest and was not staged. Neither affects the frozen
identity or zero-step result.
