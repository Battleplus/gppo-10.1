# Independent Review: v3 GPU Smoke

Review scope: static/offline review of the Windows SSH/SFTP system controller and GPU smoke gates, including the development controllers and their frozen `package/` copies, evidence download validation, transport receipt binding, and synthetic G1/G2 test selection. This review updated only this report. No remote worker, authorization, CUDA context, model call, or training operation was started.

## Disposition

Remote acceptance is **blocked before SSH session creation**. `blocking-result.json` records a `TimeoutError` at Paramiko `socket.connect`; both Windows and WSL TCP/22 probes to `172.17.27.173` also timed out after about five seconds (`network-probes.json`). Staging and worker startup remained false. No engineering job, authorization, remote interpreter verification, or GPU work was consumed. `runner_ready` remains false, and production acceptance has not completed.

The prior source findings are resolved in the current reviewed code and offline regressions:

- The system controller applies one 180-second wall budget with a five-second closing reserve, bounds SSH/SFTP operations by the remaining time, and checks controller/remote resource limits. Its frozen package copy contains the same control logic.
- The smoke gate binds both the transport and system receipts to the current manifest, hashes, and request identities. It also checks the transport-receipt digest and verifies the downloaded evidence paths and file hashes. The frozen `package/smoke_controller.py` carries these checks.
- Evidence downloads reject unsafe relative paths and symlinks, and confirm resolved paths stay within the destination. `system-io-regression.json` records seven unsafe paths rejected, with no network or model calls.
- The request requires physical GPU 1 with logical `cuda:0`. The harness's ordered phase list matches the frozen request's test IDs; the static contract test verifies those methods exist and that the pre-phase `W1_SYNTHETIC_TEST_DEVICE` setting matches the request's logical device. Actual use of the target GPU remains unverified.

No unresolved offline source blocker was identified in these reviewed gates. This does not establish behavior on the target remote runtime.

## Offline Evidence

- `offline-regression-final.log`: Windows controller regression, 46 tests passed.
- `local-wsl-system-stdout.log`: local WSL system harness, 9 tests passed on Python 3.12.3, with zero CUDA requested and zero model calls. It does not substitute for the target Python 3.10.12 / PyTorch 2.5.1+cu121 runtime.
- `offline-verification.json`: remote interpreter unverified; CUDA calls 0; model calls 0; Windows regression and local WSL harness both exited successfully.
- No `prior-transport-regression.json` or `final-system-acceptance.json` receipt exists, so the GPU smoke gate has not been satisfied.

The frozen request remains `NOT_APPROVED`; no evidence supports full resource acceptance or a research-effect claim. The only current blocker is network reachability to the configured SSH endpoint. Once connectivity is available, a real system-only acceptance must still verify the target runtime and produce its bound receipts before the smoke gate can proceed.
