# 恢复接线与预算独立静态审查

结论：在限定的静态审查范围内未发现阻断项。审查未导入 torch、加载/初始化模型、创建环境、执行前向、训练或运行正式任务；没有读取封存确认集或任务结果。没有修改 package 文件。

恢复选择与训练/评估接线一致：[recovery_contract.py](../package/recovery_contract.py:6) 固定复用 G0×3、T×3 与 G1:8301，只训练 G1:8302/8303；恢复合约核对七条最终 route、文件哈希、完整 steps/updates/saves、配对世界模型与 unchanged scientific inputs。调度仍覆盖九个方法/seed 身份；训练入口将七条记录作为 reuse routes 并只运行两条训练 route。管线检查九条 route 的身份、预算、路径与 checkpoint 哈希，再进行 240 个确认 episode。

checkpoint writer 同时保存策略 state 与 optimizer state；task loader 校验文件哈希、route 身份和 state hash 后只恢复策略权重。评估不恢复 optimizer state，新训练 route 使用新建的 Adam。离线审计观察到七份文件都含一个 35 个参数 ID 的 optimizer group，LR 均为 0.0003；每份有 33 个已序列化 optimizer slots，所有可读 step 值均为 128。受限解析器不能把两个未见 slot 的数字 ID 映射到具名参数，因此不据此声称 optimizer 参数逐项完整，也没有证据表明它改变了复用权重或新 route 的恢复行为。

本次申请预算与估算的比较：两条 G1 路由的估算为 3060 wall / 2160 CPU 秒，对应训练 cap 3200 / 2600 秒，余量 140 / 440 秒。CPU 是基于旧 incomplete stage 整体 CPU/wall 比的粗略估算，不是 route 级实测。任务确认 cap 仍为 1200 wall / 1600 CPU 秒、240 episodes；136.708 秒仅是训练日志中策略决策组件的 wall proxy 下界，不包含 H 决策、环境、评分、账本、日志或阶段开销，不能证明 cap 充足。全局 cap 为 4670 wall / 4390 CPU 秒，stage 与 accounting reserve 求和一致。

审计脚本通过标准库受限解析器重算七份 checkpoint state hash，并确认 route/export hash、schema、method/seed、保存账本及 2048 steps/128 updates；结果为完整性通过，且 `torch_imported=false`。父 agent 报告其 WSL 10 项 stdlib contract/merge 测试 exit 0；本审查没有重复执行这些测试。
