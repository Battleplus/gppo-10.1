# 唯一启动入口与待批准身份

本包状态为 `NOT_APPROVED`、`runner_ready=false`。研究命令是待批准模板，不能在服务器工程验收之前启动；工程验收入口须另获有限授权。最终摘要与实际预检证据存放在包外。

Windows 正式入口的本地结构预检不读取 token、不连接服务器、不暂存、不创建 attempt、锁或 SQLite，也不启动 worker：

```powershell
& "E:\Z博士\.runtime-tools\w1-ssh-controller-v1\Scripts\python.exe" -B "E:\Z博士\research-plans\w1-task-outcome-g1-g2-joint-production-acceptance-v2\launch_joint_once.py" --authorization-file "<external-authorization.json>" --preflight-only --structure-only
```

正式启动模板仅供未来新的明确授权使用。外部授权与一次性 token 必须绑定最终 execution manifest、hashes、RESOURCE_REQUEST、attempt 和预算。`RESOURCE_REQUEST.json` 保持 `NOT_APPROVED`，不得修改冻结输入：

```powershell
$env:W1_EXTERNAL_ATTEMPT_TOKEN = Get-Content -Raw "<new-external-one-time-token-file>"
& "E:\Z博士\.runtime-tools\w1-ssh-controller-v1\Scripts\python.exe" -B "E:\Z博士\research-plans\w1-task-outcome-g1-g2-joint-production-acceptance-v2\launch_joint_once.py" --authorization-file "<approved-external-authorization.json>" --real-name "<user-real-name>"
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

唯一正式研究入口为包内 `launch_joint_once.py`。它按冻结合同连接服务器，调用 `metered_joint_entry.py` 外层寿命计量，再由 `joint_remote_native.py` 和 supervisor 启动 `runner.py`。终态登记由等待式外层在完整内层退出和资源核验后完成。当前 `runner_ready=false`，禁止正式启动。新的工程验收使用独立 `launch_server_acceptance.py`，命令与预算见 `SERVER_ACCEPTANCE_PROTOCOL.md`；它不是正式研究attempt，不消费正式研究授权。旧attempt、dependency-only和其他入口均不可替代本次批准身份。
