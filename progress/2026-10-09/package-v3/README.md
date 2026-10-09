# W1 v5 恢复执行包

独立恢复版本。v5 原包、technical_stop、SQLite/journal、pending 调用、原始导出及旧授权只读保留；本包不重跑 v5 attempt。复用已完成采集窗口与 seed 8201，只补训 8202/8203，再按原协议继续。

本轮仅完成恢复审计、缓存准入接线和冻结预算准备；没有运行恢复入口、世界模型 forward/训练、GPPO/critic 更新或任务评价。当前资源申请 NOT_APPROVED，没有新 token、授权或正式 attempt。

主要制品：RECOVERY_AUDIT.md、RECOVERY_BUDGET.md、recovery-inputs/window-cache-manifest.json、recovery-inputs/reused-world-8201-binding.json、生产 driver 恢复路径和冻结身份。小缓存包含 288 个窗口文件（174 complete、114 no_opportunity），按 train/calibration/development/evaluation 原 split 复用；不含大型原始 JSONL、SQLite 或 snapshot bundle。

唯一入口见 unique-launch-command.md。新批准前只能部署新包并运行无授权服务器 --preflight；旧服务器准入不覆盖 v5。

seed 8201 已核验为 40 epochs、320 updates 的最终固定 epoch checkpoint；seed 8202/8203 各按 58 个 complete train 窗口、40 epochs、batch 8 训练 320 updates。旧 8202 的 155 个已完成更新及一个 pending 更新是历史消耗，不抵扣恢复训练。采集环境步为历史 6,839（恢复采集 0）；恢复新增策略训练环境步 18,432、评价环境步 4,320，共 22,752。结果仍是有限模拟器开发探索，不是独立确认、风险认证或部署结论。GitHub 归档不作为执行前置。
