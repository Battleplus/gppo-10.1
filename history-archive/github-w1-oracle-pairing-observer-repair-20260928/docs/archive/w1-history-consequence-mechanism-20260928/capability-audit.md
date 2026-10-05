# 已有历史能力与输入边界

源码基准：E:/Z博士/runs/w1-light-repaired-fair-rerun-v2-nativefs-once；精确文件哈希见input-index.json。没有导入rl_adapters、runner或环境/模型模块；只静态读取它们。

| 组件 | 实际已有能力 | 不能作出的结论 |
|---|---|---|
| public_controller.py PublicMemory.observe/get/candidates/submitted | 按测量时刻保留字段、随当前时间更新age、存观测历史、追踪pending与公开continuation、排除占用 | 不能称规则完全无历史；存有history也不代表每个selector读取完整history |
| classical_baselines.py public_edges/ClassicalSelector | nearest/EDF/Hungarian用有效公开位置、deadline及共同历史过滤；每次只提交一个动作 | 它们不等同于无限视野最优控制 |
| public_controller.py PublicPlanner.initial/child | search使用位置age、能量age、pending和continuation推算旅时/可用性 | 不能把age补偿作为新增模型独有能力 |
| rl_adapters.py ScalarRecurrentPolicy.encode/Graph5Encoder._features | PPO和Graph-5策略已有循环表示；原观测含字段age与valid | 未经消融不能认为历史已被充分利用，也不能称原方法无历史 |
| rl_adapters.py ActionConditionedTemporalWorldModel | GRUCell128，动作与关系编码；下一公开状态、奖励、后果、事件头 | 它不是只有当前帧的无记忆预测器 |
| runner.py select (约78–106行) | prepare生成公共输入；policy_hidden持续更新；world_hidden取已执行动作的hidden；commit共同执行过滤 | 原始communication_delta与本步feedback没有作为该select的显式输入，不能擅自补入 |

审计方法：按episode和保存顺序调用纯公开PublicMemory，输入仅保存的observation，随后重放自身已选动作提交的记忆更新；不重跑selector，不计算新动作，不运行环境。feedback和command_submitted仅复制为事后标签。原始通信增量包含未必公开的记录，本提案未用它构造历史特征。

表中提到的模型能力来自静态代码，不等于训练有效性结论。观察日志中的完整当前公开状态不等于真实世界状态；不同轨迹的同时间不能被当成相同状态。

计数口径：367是保存action不等于24的决策数，不自动等于新命令数。774条中总command_lost反馈为13，367行子集内为8；其余可能与NOOP时续租/原生执行过程有关，本轮未做归因。105个FULL多非NOOP窗口与旧报告115个“多合法动作”窗口口径不同，后者可含单分配＋NOOP，不能混用。
