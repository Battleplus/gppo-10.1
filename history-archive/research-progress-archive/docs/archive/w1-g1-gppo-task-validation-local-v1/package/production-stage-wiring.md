# 生产阶段与验收边界

这是新的任务验证准备包，不是一次真实任务实验的结果。

| 阶段 | 正式可达函数 | 底层依赖与计费 | 对应准备证据 |
|---|---|---|---|
| Windows授权与身份 | run_g1_task_validation.main | manifest_contract、worker_contract、task_contract；先校验再launch-intent | test_task_launch_contract实际入口拒绝坏授权；最终Windows只读预检 |
| 原生入口与worker | local_pipeline_controller.main → local_research_entry.main → supervise.main → local_research_worker.main | 固定解释器、原生暂存、socket、一次性身份、CPU0与线程1 | 最终WSL source-preflight；本轮没有启动正式supervisor/worker |
| G1资格及数据身份 | task_contract.verify_task_inputs | tape的真实父场景/结构、配置、G1 checkpoint字节、历史预测资格 | 纯字节校验；准备阶段不反序列化真实G1 checkpoint |
| 九路GPPO训练 | task_pipeline.run_pipeline → NativeRuntimeHooks.train_policy_routes → production_policy.train_policy_routes → NativeRuntimeHooks.route_runner | PublicDecisionAdapter、CausalPublicHistory、原Native GPPO；CallAccounting → SQLite ledger.call | test_g1_policy_preparation与受控生产pipeline；环境构造/step仅用合成边界 |
| 原GPPO更新 | NativeRuntimeHooks.ppo_update_fn → gppo_world.joint_training.ppo_preference_update | 多目标clipped surrogate、偏好相似度、PreCo、Adam；encoder/actor样本、反向和更新分别计费 | 非零先验/部分非法/仅NOOP有限损失与梯度、行为回放log-prob一致性 |
| checkpoint保存/恢复 | NativeRuntimeHooks.checkpoint_writer / policy_loader | 真实torch.save/load合成策略；字节及state摘要；checkpoint_writes/loads | 九路合成策略恢复；真实G1权重仅在获批后的生产loader读取 |
| 四臂任务评价 | NativeRuntimeHooks.evaluate_task_confirmation → production_policy.evaluate_task_confirmation → NativeRuntimeHooks.episode_runner | G0/T/G1配对种子，H调用真正ClassicalSelector(hungarian)；240 episodes；实际reset/step/规则决策计费 | 受控九路＋共享H矩阵；真实任务未运行 |
| 先验与完整决策成本 | DecisionPriorPolicy.select / NativeRuntimeHooks.transparent_scorer / world_predictor | 决策前公开准备、因果历史、特征、单模型全候选batch、先验、GPPO actor、合法mask、动作选择 | 原始trace及分method/parent/repeat/seed成本；假时钟验证，不用混合均值验收G1 |
| 真实效用与生命周期 | NativeRuntimeHooks.episode_runner / task_outcome_labels | 原向量奖励独立保存；物理到达严格早于deadline，确认不晚于deadline；unknown独立掩码 | 残差不改奖励、精确边界、无机会/unknown仍保存已知奖励 |
| 指标及门 | independent_task_recompute.recompute_from_files / evaluate_frozen_gates | 标准库从奖励复算、完整矩阵、配对父场景bootstrap；与生产指标比较 | 合成门只是软件测试，不是模型科研效果 |
| 结算导出 | runner._settlement / BudgetLedger / local_pipeline_controller受控导出 | SQLite逐调用、阶段上限、SELF＋waited children、导出摘要 | 受控pipeline实际结算/复制核验；最终目录只读身份核对 |

生产阶段函数没有由整阶段假函数替换来证明可运行。合成测试缩小策略更新规模并使用合成世界模型，全部实际模型计算另列计数；它不覆盖真实环境分布、真实G1权重加载后的任务表现或正式2048步×9路预算可完成性。

`runner_ready`只表示冻结任务链已通过明确的工程检查、具备申请正式执行的入口，不表示用户已批准、完整资源计量通过或研究成功。Windows/WSL桥CPU、最终写入/退出尾段与共享主机干扰仍有计量缺口；必须与训练完成和真实任务门分别报告。

本轮最终只读预检使用NOT_APPROVED授权模板、无token、无暂存、无worker、无消费。所有最终验证证据在package外保存。正式运行只能在新的明确批准后通过唯一入口执行一次。
