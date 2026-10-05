# 唯一启动命令

本包状态为 `NOT_APPROVED`。先做 Windows → Ubuntu-24.04 的只读身份与运行时预检；该命令不读取 token、不暂存、不创建 attempt、锁或 SQLite，也不启动 worker：

```powershell
python -B "E:\Z博士\research-plans\w1-action-conditioned-task-outcome-label-repair-v6-dependency-cpu-settlement\launch_once.py" --authorization-file "E:\Z博士\.codex-private\w1-action-conditioned-task-outcome-label-qualification-v6-dependency-cpu-settlement-once\authorization.json" --preflight-only
```

正式启动模板仅供未来新的明确授权使用。外部授权与一次性 token 必须绑定最终 execution manifest、hashes、RESOURCE_REQUEST、attempt 和预算。`RESOURCE_REQUEST.json` 保持 `NOT_APPROVED`，不得修改冻结输入：

```powershell
python -B "E:\Z博士\research-plans\w1-action-conditioned-task-outcome-label-repair-v6-dependency-cpu-settlement\launch_once.py" --attempt-token-file "<new-external-one-time-token-file>" --authorization-file "E:\Z博士\.codex-private\w1-action-conditioned-task-outcome-label-qualification-v6-dependency-cpu-settlement-once\authorization.json"
```

唯一正式入口为包内 `launch_once.py`。它按冻结合同调用 Ubuntu-24.04，再由 Linux 原生 supervisor 启动 `runner.py`。本 attempt 只采集并核验标签，不加载模型、checkpoint，不做前向或训练。不得使用旧 attempt、dependency-only 运行或替代入口。
