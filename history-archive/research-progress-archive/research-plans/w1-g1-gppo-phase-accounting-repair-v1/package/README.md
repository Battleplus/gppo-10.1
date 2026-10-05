> 本副本：阶段CPU协议与异常结算修复。旧包运行已技术停止，旧runner_ready结论撤回。新身份和验收见 PHASE-CPU-REPAIR.md；本轮无正式研究执行。下文继承研究方案不代表本次批准。

> 历史继承文档，仅保留溯源。下文的身份、授权、设备、预算、状态和入口均不适用于本次 G1＋GPPO 任务申请。
> 本次以 G1-GPPO-PROTOCOL.md、README-task-validation.md、RESOURCE_REQUEST.json 和 unique-launch-command.md 为准；未批准正式运行。

# G1＋GPPO 任务验证包

本副本只申请新的 GPPO 四臂任务验证。准备、测试、冻结不授权正式训练。请优先阅读 G1-GPPO-PROTOCOL.md、experiment-matrix.json、g1-model-binding.json、parent-split.json 和 RESOURCE_REQUEST.json。继承的旧实验说明仅作历史溯源，不是当前协议。

唯一入口为 run_g1_task_validation.py；run_local_research.py 已禁用。真实运行依赖全新包外授权和token。当前申请 NOT_APPROVED，proposed attempt 为 w1-g1-gppo-task-validation-local-v1-once。

现有证据支持 G1 相对乐观透明参考的候选排序改善，尚无GPPO收益或超越Hungarian证据。G2负结果保留，本轮不扩训。八任务父场景的跨项目历史未使用状态未证明。

正式授权后按固定九路策略训练→九checkpoint恢复→240任务episodes→独立指标复算→收益与G1成本分别判断→结算与受控导出推进。异常不重试；完整计量限制必须披露。
