# 唯一启动命令（开发草案，尚未冻结）

本包状态为 `NOT_APPROVED`、`runner_ready=false`。当前命令只可用于未来最终冻结包的结构核对；它不构成授权，也不表示此开发包可运行。

Windows → Ubuntu-24.04 的只读身份与运行时预检不读取 token、不暂存、不创建 attempt、锁或 SQLite，也不启动 worker：

```powershell
& "E:\Z博士\.runtime-tools\w1-ssh-controller-v1\Scripts\python.exe" -B "E:\Z博士\research-plans\w1-task-outcome-g1-g2-joint-preparation-v1\launch_joint_once.py" --authorization-file "<external-authorization.json>" --preflight-only --structure-only
```

正式启动模板仅供未来新的明确授权使用。外部授权与一次性 token 必须绑定最终 execution manifest、hashes、RESOURCE_REQUEST、attempt 和预算。`RESOURCE_REQUEST.json` 保持 `NOT_APPROVED`，不得修改冻结输入：

```powershell
$env:W1_EXTERNAL_ATTEMPT_TOKEN = Get-Content -Raw "<new-external-one-time-token-file>"
& "E:\Z博士\.runtime-tools\w1-ssh-controller-v1\Scripts\python.exe" -B "E:\Z博士\research-plans\w1-task-outcome-g1-g2-joint-preparation-v1\launch_joint_once.py" --authorization-file "<approved-external-authorization.json>" --real-name "<user-real-name>"
```

The structure-only command verifies local frozen inputs without connecting to
SSH. It does not validate the remote runtime or consume an approval. Omitting
`--structure-only` also runs the remote read-only probe and needs an SSH connection.
The explicit controller interpreter must match `controller-runtime-contract.json`;
the default Windows Python is not the frozen transport runtime.

The Windows entry derives and validates `name_id` from the supplied real name
before reading a password or opening SSH. An explicit `--name-id` is accepted
only when it equals that derived value; it is not a second user registration
field.

唯一正式入口为包内 `launch_joint_once.py`。它按冻结合同连接 Ubuntu-24.04 原生服务器，再由 Linux 原生 supervisor 启动 `runner.py`。正式启动前，远端固定登记文件追加并回读 `REGISTERED`，然后记录真实 PID/PGID/worker PID 与 GPU；任务结束追加终态。不得使用旧 attempt、dependency-only 运行或替代入口。当前开发包仍禁止正式启动。
