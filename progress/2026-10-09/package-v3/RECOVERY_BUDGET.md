# 恢复预算依据

所有数值是新 attempt 的上限，不是旧 v5 的剩余额度。旧 CPU、wall 和 pending 不抵扣新申请。

## 环境步与更新

| 工作 | 环境步 | 模型/策略更新 | Episode |
|---|---:|---:|---:|
| v5 历史采集（只读复用） | 6,839 已完成 | 0 | 0 |
| 恢复采集 | 0 | 0 | 0 |
| 8201 | 0 | 0（复用 320-update 完整模型） | 0 |
| 8202、8203 | 0 | 640（每 seed 320） | 0 |
| 九路策略训练 | 18,432 | 1,152 GPPO 更新 | 0 |
| 四臂任务评价 | 4,320 | 0 | 240 |
| 新 attempt 合计 | **22,752** | **640 世界模型 + 1,152 策略更新** | **240** |

训练数据有 58 条 complete train 窗口。原方案每 epoch 的 batch 数为 `ceil(58/8)=8`，40 epochs 得每 seed 320 updates。每 epoch 候选数合计 257，因此每个新 seed 有 10,280 candidate evaluations；两 seed 合计 20,560。每 seed 的 target 处理窗口数为 `58*40=2,320`，两 seed 共 4,640 target forwards。旧 8202 的 155 complete updates 和 1 pending update全部记历史，不冲减该数。

## CPU / wall 依据

v5 `world_training_and_restore` 的不完整尾部测得 CPU 1,600.227401 秒、wall 1,605.171824 秒，对应账本已有 475 次 complete updates；其中 8201 的 320 次由完整 checkpoint 独立证明，activity 指向 8202。`1600.227401/475 ≈ 3.37` 秒/update 只作包含数据准备、target forward、账本与阶段开销的粗估。

两路新训练估算约 `640*3.37=2,157` 秒 CPU。训练阶段上限设 3,200 秒 CPU、3,600 秒 wall，分别留约 48% 与 67% 的余量，覆盖输入 tensor 转换、checkpoint 保存/恢复、8201 binding 核验、同步及阶段关闭；硬限仍有效，超限停止，不取消检查。

| 阶段 | CPU 秒上限 | wall 秒上限 | 依据/用途 |
|---|---:|---:|---|
| input_admission | 600 | 900 | 288 个小 gzip 文件的摘要、解压、标签/身份复核、8201 文件准入；不读大 JSONL |
| paired_collection | 0 | 0 | 禁止采集器调用；复用窗口，环境步为 0 |
| world_training_and_restore | 3,200 | 3,600 | 两个 320-update seed 的实测外推加明确余量；严格三模型恢复 |
| offline_prediction | 400 | 600 | 原离线门，三个固定模型；预算保持独立 |
| conditional_policy_training | 5,000 | 12,000 | 原九路 1,152 更新 / 18,432 环境步，沿用冻结资源分项 |
| conditional_task_confirmation | 1,200 | 2,000 | 独立保护 240 episodes / 4,320 环境步 |
| analysis_and_settlement | 500 | 600 | 独立指标复算和资源核对 |
| settlement_and_verified_export | 800 | 2,400 | 最终收据、制品清单、校验导出与 supervisor 收尾 |
| **worker stages subtotal** | **11,700** | **22,100** | 不含 controller/transport reserve |
| **服务器总限额** | **12,000** | **23,000** | 另留 300 CPU 秒及 900 wall 秒给监督器/进程边界 |

单线程 CPU0、GPU 关闭、RSS 4 GiB。外部包上传上限 128 MiB、wall 600/CPU 300 秒；结果检索上限 8 GiB、wall 3,600/CPU 600 秒，单独计量。含传输上限为 wall 27,200 秒、CPU 12,900 秒。网络当前没有可靠实测值，外部传输预算是硬上限而非吞吐预测。

## 完整分项额度

资源申请逐 stage 定义于 `RESOURCE_REQUEST.json`。核心 totals：

- `reused_windows=288`，`candidate_contract_audits=219,277`；这包括恢复输入逐窗口/逐候选准入、训练前两类输入审核、离线、策略及任务阶段审计。
- `environment_steps=22,752`，其中策略训练 18,432、评价 4,320、采集 0；`candidate_branches=0`、`candidate_snapshot_probes=0`、`forced_first_actions=0`。
- `world_model_updates=640`、`training_forwards=640`、`training_candidate_evaluations=20,560`、`target_forwards=4,640`、`target_candidate_evaluations=20,560`。
- `policy_updates=1,152`、`task_episodes=240`、`checkpoint_loads=18`、`checkpoint_writes=11`、`model_initializations=23`。
- 每项 stage cap 和全局 cap 均由正式 SQLite `BudgetLedger` 强制；未知键、超 stage 或总量上限均在操作前拒绝。working/export storage 各 8 GiB，RSS 4 GiB。

v5 已经测得 1,600.227 CPU 秒是不完整旧阶段样本，不作为本申请“已花费”或“剩余预算”。正式研究预算只有新授权通过后才能消费。
