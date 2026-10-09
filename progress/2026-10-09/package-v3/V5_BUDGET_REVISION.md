# v5 独立资源申请

这是新身份的完整申请，不是 v4 已消费预算的余额。原 v3/v4 授权不可复用。

保持全部阶段和总量上限：environment_steps=106416、world_model_updates=1440、policy_updates=1152、task_episodes=240、candidate_snapshot_probes=7200、candidate_contract_audits=226800；服务器 wall=32500 秒、CPU=14900 秒；含传输 wall=36700 秒、CPU=15800 秒。RSS 4 GiB；working 与 export 各 8 GiB。其余字段与冻结 v4 一致，没有扩大研究预算。

candidate_snapshot_probes 的操作定义变为一次已保存公共快照重建与序列化合同核验，不再 deepcopy 并调用环境 observation。同一恢复计费一个 probe；命名字段检查、缓存检查、前缀摘要和跨层比较计入已有 enclosing operation 的实耗，不增加重复计费键。candidate_contract_audits 仍为一条持久身份审计行，新 prefix 摘要作为该行字段保存。

新快照保存一次完整公共状态及紧凑候选记录；window/replay 同样不逐候选重复保存完整公共状态。最多 25 个候选记录，每条五个固定字段（schema 和四个 64 字符摘要），约 9 KiB/决策；18,432 条训练行为的上界增量约 160 MiB，另加 source/header 和审计序列摘要。审计字段最坏按 25 个前缀摘要约 1.8 KiB/行估算，226,800 行约 390 MiB。仍使用既有 1 GiB 合同增量储备及 8 GiB 工作上限，不把估算当全规模实测。

成功的受控完整采集和合同验收为 738 个合成环境步；进程 /usr/bin/time 约 wall 63.48 秒、CPU 35.46 秒、RSS 578924 KiB。实际规模、候选数和服务器负载不同，不能称正式全规模测量。仅按环境步线性外推该样本，83,664 个采集步骤约 4,020 秒 CPU；该粗估包括样本内 tensor/label/audit 验收，且不代表全部正式阶段。采集 CPU 上限仍 5,000 秒，超限必须停止，不能因本估算放宽。G1 推理、训练和九路计算仍受原各自阶段上限约束，不用该样本替代其测量。

中断开发轮和补充受控验证均单列在 engineering-resource-accounting.json；不是正式训练，也不是零工程计算。批 tensor 构造、CPU 序列化和哈希实耗已计入测量；模型初始化、checkpoint 加载和 neural forward 为 0。

正式预测门失败时 policy/critic 更新及任务 episodes 必须为 0。analysis_and_settlement 的 wall 600/CPU 500 与 settlement_and_verified_export 的 wall 2400/CPU 800 独立保留。持久托管、停止、不自动重试和计量缺口沿用已有协议。

v5 当前 NOT_APPROVED；没有正式 token、授权、attempt 或动态研究。部署后须独立执行 v5 无授权服务器 preflight，不能以旧 v4 准入替代。
