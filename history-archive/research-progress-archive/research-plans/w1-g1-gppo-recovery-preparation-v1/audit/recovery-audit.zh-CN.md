# G1 GPPO 离线恢复审计

审计输入来自 phase-accounting-repair 导出的 7 个策略 checkpoint、route JSONL、只读 SQLite、冻结配置和资源结算。审计器仅用 Python 标准库；pickle 仅允许原始映射、tensor 元数据 stub 与 Storage 标记，tensor 状态哈希从 ZIP 原始 storage bytes 重算。没有导入 torch、加载模型、创建环境、执行前向或训练。

- 7 个文件对应 G0×3、T×3、G1 seed 8301。完整性（schema、method/seed、文件哈希、state hash、保存账本、2048 steps/128 updates）通过：`True`。
- G1 seed 8301 的配对世界模型是 8201；绑定文件、8201 checkpoint 路径和 SHA-256 与 route 配对一致。
- G1 seed 8302 只有 512 steps、28 个完成更新和 1 个 pending 更新，没有任何中间/最终 G1:8302 checkpoint。writer 仅保存 policy、optimizer 和 route summary，缺 RNG、环境、公开历史及 rollout buffer 等精确续训状态，因此不能从该断点精确续训；公平补跑应从初始 seed 8302 重跑完整路由。seed 8303 也需要完整路由。
- G1:8301 route wall 为 951.758s；G1:8302 partial wall 到账本最后事件为 251.610s。两者均含未归属的 route 间调用间隔；SQLite 只有 wall，不可把间隔说成纯等待或拆成 CPU。
- 两路补跑需要 4096 steps、256 updates。按观测较慢的 G1 wall/step 外推并乘 1.5 安全系数，建议训练估算 3060 wall 秒、约 2160 CPU 秒；相对本次申请 3200 wall / 2600 CPU 秒 cap，余量分别为 140 / 440 秒。CPU 数值由 incomplete stage 的整体 CPU/wall 比近似，不能称为 route 实测。
- 本次申请的 task confirmation cap 为 1200 wall / 1600 CPU 秒，240 episodes：8×3×(9 个策略实例+共享 H)，不是训练步或 PPO rollout 数。训练 SQLite 的 decision component wall proxy 在最大策略步数下约 136.7s；这是下界代理，不含 H 决策、环境、特征与透明评分、账本、日志和阶段开销，不能证明该 cap 充足。没有逐调用 CPU 计时，也不能据此校准 CPU。全局 cap 为 4670 wall / 4390 CPU 秒；按 stage 与 accounting reserve 求和一致：True。

机器结果见 `recovery-audit.json`。重新运行：`python audit_recovery.py`。
