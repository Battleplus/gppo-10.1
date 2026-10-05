# 当前 v3 smoke 入口说明

本包唯一GPU入口见unique-launch-command.md；下方继承的研究v2示例
不属于当前可用启动命令。新v3使用真实supervisor/worker完整链。
运行时与隔离导入在metered worker内合并核验，先于CUDA；系统验收
先通过同一wrapper --system-only。预算与profile不改变。

# Bounded Server Acceptance Protocol

This is a separate engineering request. `SERVER_ACCEPTANCE_REQUEST.json` must
remain `NOT_APPROVED`; `RESOURCE_REQUEST.json` remains `NOT_APPROVED` and
`runner_ready` remains `false`. No authorization or result from this workflow
creates a formal research attempt or supports a scientific success claim.

## Local structure check

After the request's four `implementation_sha256` entries have been finalized,
run either command from this directory:

```powershell
$controllerPython = 'E:\Z博士\.runtime-tools\w1-ssh-controller-v1\Scripts\python.exe'
$launcher = 'E:\Z博士\research-plans\w1-task-outcome-g1-g2-joint-production-acceptance-v2\launch_server_acceptance.py'
& $controllerPython -B $launcher --structure-only
& $controllerPython -B $launcher --preflight-only
```

Both flags perform the same local-only checks: the frozen package identity,
request invariants, acceptance-source hashes, launch/runtime bindings, and
joint input contract. They do not read an authorization token, prompt for a
password, import a server runtime, connect over SSH, create a remote path, or
start a worker. A `PENDING_FINAL_HASH` value must fail this check. This check
does not establish that the remote runtime or GPU is available.

## Independent approval

A formal engineering launch requires an externally stored JSON authorization
with exactly these fields:

```json
{
  "schema": "w1-server-acceptance-authorization/1.0.0",
  "status": "APPROVED",
  "engineering_job_id": "w1-joint-production-acceptance-v2-engineering-once",
  "remote_execution_root": "/home/user1/w1-joint-production-acceptance-v2-engineering-once",
  "evidence_directory": "/home/user1/w1-joint-production-acceptance-v2-engineering-evidence",
  "server_acceptance_request_sha256": "<sha256>",
  "execution_manifest_sha256": "<sha256>",
  "hashes_sha256": "<sha256>",
  "authorization_id": "<independent approval id>",
  "one_time_token_sha256": "<sha256 of the separately supplied token>",
  "formal_research_attempt_created": false,
  "automatic_retry": false
}
```

The authorization binds the exact acceptance request, frozen execution
manifest, and `hashes.json`. It must not contain the raw token or any other
credential. The request, identities, and authorization are checked locally
before SSH. The supplied real name is required before the password prompt and
must derive the same server registration identity.

## One-shot launch

Use the fixed command only after a separate approval exists:

```powershell
$controllerPython = 'E:\Z博士\.runtime-tools\w1-ssh-controller-v1\Scripts\python.exe'
$launcher = 'E:\Z博士\research-plans\w1-task-outcome-g1-g2-joint-production-acceptance-v2\launch_server_acceptance.py'
& $controllerPython -B $launcher --authorization-file '<external-authorization.json>' --real-name '<user-supplied-real-name>'
```

The controller verifies the Windows SSH runtime, uses the pinned host key,
runs the read-only remote runtime check, and then checks that both fixed remote
paths are absent. It stops before reading the token if either path exists or
cannot be checked. It never selects another GPU or path. With
`--password-stdin`, provide the SSH password on stdin's first line and the
one-time token on its second line; without that flag, the password is prompted
interactively and stdin supplies the token line. The token is sent only to the
remote isolated entry and is not printed, saved, or registered.

Creation of the fixed execution directory consumes the single engineering
attempt. A staging, runtime, GPU, test, export, accounting, or download failure
is a technical stop. There is no automatic retry, alternate directory, or
reused token. The launcher downloads only files named in the remote evidence
manifest and verifies each byte count and SHA-256 before exposing the local
copy under `runs/<engineering_job_id>`.

On the server, registration is append-only and follows this order:

1. `REGISTERED` with the request, package identities, and full engineering budget.
2. `RUNNING` for the root PID with an empty GPU binding.
3. Exactly one CUDA-context initialization, counted separately from model calls.
4. `RUNNING` with the registered root PID observed on physical GPU 1.
5. Terminal `SUCCEEDED` only after all fixed tests, controlled export, exact call
   totals, evidence manifest, and closing resource upper bound pass. Any prior
   technical failure records `FAILED` or `EXPIRED` where the ledger remains
   writable. An appended terminal event cannot be rolled back; failure to
   write the later receipt or manifest remains a technical stop and must not
   be reported as complete.

## Fixed workload and limits

Only four tests are selected directly; there is no discovery:

1. The one CPU-scope arithmetic case named in the request.
2. `test_supervised_phase_integration`'s short production supervisor and
   authenticated `BudgetLedger` boundary case: three waited synthetic CPU
   children, 0.09 commanded CPU seconds, no research environment or model.
3. The `test_joint_pipeline` CUDA synthetic 2/2/8 one-epoch production
   collection, training, restore, prediction, metrics, ledger, and controlled
   export case.
4. The `test_sequence_world` CPU 1/1/8 one-epoch direct production-world
   checkpoint and prediction regression.

The exact aggregate model counters are 24 initialization/load calls, 150
world-batch forwards, 18 backward calls, 18 optimizer updates, 12 checkpoint
writes, 12 checkpoint loads, 750 sample evaluations, and one CUDA-context
initialization. A count mismatch stops the attempt. The `test_joint_pipeline`
and `test_sequence_world` evidence are kept separate from the supervisor
phase-sync evidence.

The end-to-end request remains capped at 180 seconds wall and 360 complete-
process CPU seconds. It allocates at most 150 wall / 300 CPU seconds to the
server acceptance entry, 30 wall seconds to controller/transport overhead,
30 CPU seconds to Windows controller CPU, and a separate charged 30 CPU
second allowance for remote preflight and SSH/SFTP transport work. The server
allocation includes a 3 CPU second / 5 wall second closing charge. The
controller reserves 2 CPU seconds and 5 wall seconds inside its 30 second
allowance. The launcher checks `server_charged_wall + 30 <= 180`,
`server_charged_cpu + 30 remote-setup allowance + 30 controller allowance
<= 360`, and observed controller overhead plus its close reserve against 30.
No budget is borrowed from the formal research request.

The server starts wall accounting at acceptance-entry startup and counts
process-self plus waited-descendant CPU from process start.
`linux_process_scope.enable_subreaper` and `reap_owned_children` preserve
ownership of child work. Before terminal registration, a 3 CPU second / 5
wall second closing charge is added to the server measurements and checked
against its allocated 300 / 150 caps. The production
`metered_joint_entry.enforce_closing_limits` applies a hard CPU limit and a
five-second alarm during terminal registration and final writes.

Server CPU and RSS evidence is sampled, not an independent kernel hard cap on
every descendant: CPU uses `RUSAGE_SELF + RUSAGE_CHILDREN`, so a child's CPU
is included after it is waited/reaped, and checks run around ledger operations
and fixed phase boundaries. The configured RSS ceiling measures only the root
entry's `/proc/self/status` `VmRSS`/`VmHWM`; it is sampled at meter checks,
excludes descendants, and is not an all-resident process-tree peak. Storage is
scanned at phase/call boundaries and forced before final checks, so transient
peaks between samples are not claimed as continuously observed. Remote preflight
and SSH/SFTP helper CPU is not individually measurable from the Windows
controller; its fixed 30 CPU second
allowance is charged, but is not a verified hard upper bound. Therefore an
end-to-end CPU hard-cap claim remains blocked until that remote transport
scope can be measured or bounded. The launcher preserves any successful
remote test result and downloaded evidence, but reports overall
`technical_stop`, `cpu_scope_unverified: true`, and `runner_ready: false` while
this CPU scope remains unverified. This settlement block is not a model or
collection failure. The existing phase-sync regression covers
short waited children and authenticated stage boundaries; it does not turn
these sampled checks into a formal all-process resource guarantee. A signal
or five-second alarm also cannot guarantee prompt termination during
uninterruptible kernel D-state I/O. Missing final evidence or an unconfirmed
terminal event is a technical stop.

GPU 1 must be the requested physical device, have at least 9 GiB free before
registration, show exclusive ownership at the observed checks, and stay under
the 8 GiB owned-process memory ceiling. The runtime must match the pinned
isolated Python, PyTorch, and CUDA identities. No GPU substitution or
interference with another workload is allowed. A busy GPU or runtime mismatch
stops before synthetic model work.

After CUDA context setup, and at both boundaries of each fixed test phase, the
entry runs the pinned `gpu_snapshot` query and rejects any unowned compute PID
or owned-process memory over 8 GiB. Within each model call, it checks
`torch.cuda.max_memory_allocated(0)` and `max_memory_reserved(0)` before and
after the existing production operation. These allocator high-water reads do
not add a forward pass. The expensive `nvidia-smi` snapshot is phase-boundary
only, not repeated for every model call; transient GPU ownership between those
snapshots is therefore not claimed to be continuously observed.

Passing this acceptance verifies only these bounded synthetic engineering
paths on the observed server. The selected joint test must enter through
`runner.main` on an isolated test copy with an explicit synthetic bottom
boundary; it does not enable an integration boundary in the formal CLI. This
does not validate the formal 24/8/8 research matrix, real simulator collection,
or model effects. The earlier local v6 CPU runtime is not the pinned server
CUDA runtime. The workflow must leave `runner_ready=false`.

## Explicit GPU smoke-test adapter scope

This independent copy is authorized only for a synthetic GPU smoke test. It
uses a new engineering job identity and does not create a formal research
attempt. The adapter skips the fixed registration-file write and authoritative
GPU-exclusive-allocation assertion that belong to the full engineering
acceptance, while retaining GPU 1 selection, free-memory checks, runtime
identity checks, model-call budgets, production collector/training/checkpoint/
prediction/metric/export code, and the 180-second wall / 360-second CPU
limits. Full SSH/SFTP CPU lifetime is reported as unmeasured. The result must
be labeled `GPU_SMOKE_TEST` and cannot set `runner_ready` or support a
research-effect claim.
