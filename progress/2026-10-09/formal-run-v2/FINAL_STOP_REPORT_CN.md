# W1 v5 Recovery v2 正式运行停止结算

日期：2026-10-09（Asia/Shanghai）

## 身份与提交

- Attempt：`w1-drone-action-consequence-world-model-v5-recovery-v1-linux-release-v2-once`
- Manifest：`d10922e7d439fae4b14cb48dbb62ef40cd139e8c7df141e86260fd0cde07e1a2`
- Hashes：`5b72e995e46d214d73725ad59c69cf26f2576854466af69b81991ce8b12cb536`
- Request：`5af24201a1235279c79f1b54a24c0f32424961be1e0272b47c536bec10f6b562`
- 正式入口提交一次并返回 `persistent_handoff`；同一 systemd user service 最终失败退出，`ExecMainStatus=1`。授权身份记为 `consumed_no_retry`。没有重新提交、自动重试或追加预算。

## 停止原因

正式 worker 在 `input_admission` 构造完窗口输入后，准备原子写入 `lean-inputs.json` 时，被真实 `BudgetLedger` 拒绝：`input_admission output_writes limit exceeded`。worker traceback 位于随附 `first-error.json`。这是账本输出写次数上限触发的技术停止，不是模型预测门的科学停止，也不是旧的 `event_signal` / `flat[769]` 错误在本次运行中被重新触发。

监督器首错只写 `Worker exited 1` 且无 traceback；worker 的持久首错收据提供了上述直接原因。监督器记录 `automatic_retry=false`，`pending_calls=0`、`failed_calls=0`。账本已登记 `output_writes=2`，第三次输出请求在执行前被拒绝；完整 `lean-inputs` 没有成功发布。

## 已完成与未进入阶段

正式运行计数为：复用窗口 288，候选合同审计 1,129，账本登记输出写 2。窗口进入了复用和合同审计路径，但由于 lean 输入未能持久化，`input_admission` 未完成准入提交。因此不能将这次运行报告为完整数据准入成功。

停止发生在 `train_worlds` 之前。正式运行没有执行世界模型 forward 或更新；8201 的本次严格复用核验未进入，8202/8203 均未训练。离线预测门、九路 critic/GPPO/PreCo 训练及恢复、240 episodes 评价、独立指标复算均未评价。正式研究环境步、策略更新和任务 episodes 均未发生；没有生成可用于科学结论的预测或任务结果。既有工程验收不计作正式训练。

## 资源结算

- 申请总上限：server wall 23,000 秒、server CPU 12,000 秒。
- controller 收据测得：wall `141.67668371787295` 秒，Linux CPU `138.31047194500002` 秒，退出码 1。
- 导出子阶段：wall `0.5121300686150789` 秒、CPU `0.16786426000001597` 秒；dispatch CPU `0.358226953` 秒。这些为收据中的分项视图，不与总量重复相加。
- Supervisor 峰值 RSS：`642,715,648` bytes；RSS 上界样本 `652,013,568` bytes。记录 artifact bytes：`223,669`。
- 资源范围不完整：controller 明确列出 SSH/client transport、systemd user-manager CPU、controller exit tail 未计量；监督器也标记 `cpu_scope_complete=false`、`cpu_lifetime_settlement_complete=false`。因此上述 CPU/wall 是已测服务器范围，不是完整端到端总成本；未测项保持 unknown，不补零。
- 未触及批准的总 wall/CPU 数值上限；触发的是阶段内 `output_writes` 账本上限。冻结包预算没有更改。

## 保留与归档

服务器原始 run 保留于 `/home/user1/w1-pilot/runs/w1-drone-action-consequence-world-model-v5-recovery-v1-linux-release-v2-once`。本地只读证据副本位于本目录的 `verified-export/`，包含 controller、worker、supervisor 收据和资源历史；各文件 SHA-256 见 `evidence-sha256.txt`。旧数据、历史失败证据、pending 与旧授权记录均未修改。

研究结论：本次方案整体为 `technical_stop`；所有预测效能、任务收益、重大损失、Hungarian 比较、种子/父场景稳定性及完整决策成本结论均为“未评价”。不作风险认证或部署结论。

GitHub 归档独立处理。现有本地归档状态报告没有配置远端 URL，因此没有可核验的 GitHub commit 链接；这不影响本地停止结算。
