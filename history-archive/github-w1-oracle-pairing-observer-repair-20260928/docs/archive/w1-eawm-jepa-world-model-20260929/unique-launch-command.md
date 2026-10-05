# 唯一启动入口（未授权）

正式研究 backend 尚未完成，因此不存在可用正式启动命令。当前只允许只读预检：

```powershell
python .\production_chain_check.py --preflight-only
```

该命令只读取既有修复后入口和本包合同，不创建 attempt、锁、SQLite、native staging 或 worker，不调用 `reset`/`step`，不导入 checkpoint。

该预检只验证既有基础设施文件身份，不是本包全流程验收，也不消费授权。完成
`BLOCKERS.md` 所列 backend 和 Windows→WSL 替身集成后，才能生成冻结清单、包外授权
与一次性 token 绑定的唯一命令；不能复用旧 attempt 或旧 token。
