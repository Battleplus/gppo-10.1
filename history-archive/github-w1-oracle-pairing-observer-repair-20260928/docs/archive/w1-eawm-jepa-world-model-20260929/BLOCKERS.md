# 启动阻塞

当前包不是 `RUNNER_READY`，不得申请或执行动态 attempt。

唯一不可绕过的实现缺口是新的 G0/T/G1/G2/H production backend。已验证的
Windows -> Ubuntu-24.04 -> native staging -> supervisor -> worker -> settlement ->
verified export 基础设施可复用，但其现有 runtime backend 固定服务旧 A/B 标量
剩余效用实验，标签是 Hungarian 续行终局结果，没有本包所需的一步公开状态/
事件/透明残差标签，也没有四个 GPPO 策略训练路由和五臂任务评价。

下一次静态实现必须完成并用底层替身贯通：

1. 生产 collector 在分支前冻结 `W1PublicInput` 和 `PublicHistoryState`，采集一个
   决策间隔的全部合法候选标签，写入 measurement/receipt/valid/continuation 身份；
2. G1/G2 各3 seed 的真实 `train_world_model` 调度、模型选择、checkpoint 身份和计费；
3. 逐候选 G1/G2/透明 trace、父场景宏平均指标及 `world_model_pipeline` 三项预测门；
4. 通过门后 G0/T/G1/G2 各3 seed 的2048步 GPPO 训练，以及312个配对任务 episode；
5. 每臂独立完整决策成本、SQLite 结算和受控导出；
6. 从真实 Windows 入口以底层环境/模型/优化器替身完成全链集成测试。

旧 `production_chain_check.py --preflight-only` 只证明依赖文件和既有入口存在，
不证明上述研究 backend 已接通。完成 backend 后需生成新的 execution manifest、
hashes 和外部授权模板，对最终目录原样只读预检；不能只改摘要。
