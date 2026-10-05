# Luna 独立复核

复核代理 `/root/training_package_review`，模型 gpt-6-luna、max；仅阅读源码和保存制品，没有 SSH、真实环境、模型、checkpoint、前向或更新。

运行时 v2 identity、requirements lock、synthetic-probe 与探针源码的哈希相符，远端依赖修复已有可复用证据。主线程最新 SSH 只读预检通过，两张 RTX 2080 Ti 可见、空闲 10388/10988 MiB，tensor/model/optimizer=0，未启动 staging/worker/attempt。

合成探针仅验证 CPU、cuda:0、cuda:1 的 4×4 matmul；反向和 SGD 只在 cuda:1 各一次。未验证双卡并发、NCCL/DDP、生产模型、checkpoint 或正式训练。独占分配未证明，未来负载可用性及正式训练就绪未知。

v6 数据不能直接输入 v3/v4 学习链：schema/split/continuation 不匹配；真实任务后果标签未进入旧训练目标。新训练副本是 BLOCKED_DRAFT_NOT_SUBMITTABLE，launcher 与 worker 已拒绝正式执行。结论不是训练验收通过，也不授权训练。
