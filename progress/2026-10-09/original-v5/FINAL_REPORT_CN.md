# v5 本次执行结论

状态：`technical_stop`；正式提交恰为 1 次，未重试。

本次补充检查通过实际函数调用与摘要一致性证明 prefix_hashes 使用统一 PublicPrefix 序列化器；原错误检查及停止证据保持只读。v5 冻结身份和请求未修改。

环境步 6839；世界模型更新 475 次已完成、1 次 pending；GPPO 更新 0；任务 episodes 0。
服务器 wall 2423.338 秒、Linux CPU 2335.631 秒；全局冻结上限为 32500/14900 秒。总上限内，但阶段 CPU 尾部超过 1,600 秒上限 0.227 秒。完整计量边界见 controller-settlement.json 和 final-result.json，未计量项不补零。

首错：RuntimeError：world_training_and_restore CPU shutdown reserve

冻结 world_training_and_restore 阶段 CPU 上限为 1,600 秒；监督器采样到该阶段 1,600.227 秒并触发 CPU shutdown reserve，未获阶段转换确认。worker 被 SIGTERM，最终 worker ledger 缺失。SQLite 保留 475 次已确认世界模型更新、1 次 pending 更新及其内部 1 次 pending forward；pending 仍计在请求预留量内。该 0.227 秒采样越界是计量/关停粒度带来的已记录阶段超限，不作隐藏修正。这是资源阶段停止，不是候选排序或研究假设的科学负结果。

停止于世界模型训练阶段。独立预测门、GPPO 更新、四臂任务评价及相应预测/任务指标均未评价；G1−G0、G1−T、G1−Hungarian、重大损失与最差场景、在线触发和决策成本未评价。

结果限定开发探索，不作 5% 风险认证或部署结论。碰撞保持 unknown。GitHub 归档单列，未阻断本地执行与交付。
