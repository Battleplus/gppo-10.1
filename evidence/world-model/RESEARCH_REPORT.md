# 本机真实 W1：G1/G2 六路训练与预测评价

真实运行已完成，退出码0，状态 `prediction_evaluation_complete`。数据门40/40通过，G1/G2各三个种子训练完成，六个选定checkpoint全部保存并恢复。无技术停止、无补跑；GPPO策略训练和任务对照均为0。历史停止状态与负结果不改判。

结论：G1在本次内部隔离评价中改善了相对冻结透明方法的候选动作排序；G2改善了部分自动事件指标，却没有带来相对G1的排序增量。值得准备G1的独立GPPO任务验证，尚不能称为真实策略收益或实用成本成功。

## 执行身份、迁移与数据来源

- Attempt：`w1-task-outcome-g1-g2-joint-local-v1-once`，已消费，不可重启。
- Manifest：`e08328eeeab23d4cf5d64456f5b7b12d5604b793e56a44963ae233f71f0c590c`。
- Hashes：`4303b91ac78e056a0d4fa187a6ab358e77d215df9fb646b4693f9ec314c8217d`。
- 资源申请：`bd11b07eb6146eb75166d336bc7a31369ece4450e896167be1fcba7ce6fd6d9e`；包内仍为NOT_APPROVED，运行采用本轮外部授权。
- 实际设备：本机RTX 3060 Laptop GPU0，Ubuntu-24.04原生文件系统；Python3.11.16、torch2.7.0+cu128、CUDA12.8、驱动571.96。
- 复用成功本机smoke生产实现；新增本机入口、运行时绑定、暂存/结算/共享GPU限制。研究collector、模型/损失、训练选择、数据门和预测门保持冻结版本；详见migration-audit.json。
- 实际启动：`python -B "E:\Z博士\research-plans\w1-task-outcome-g1-g2-joint-local-v1\package\run_local_research.py"`。此命令仅记录本次已完成执行，不是可再次使用的命令。
- Windows与原生最终目录预检退出0；原生检查token且不消费。运行后182个冻结内容文件及外层身份保持不变，27个导出文件和6个checkpoint摘要全部复核一致。

真实生产W1模拟器构造40次，采集40完整窗口、547候选。训练train-0064..0087（24父场景/341候选），选择train-0088..0095（8/105），评价train-0096..0103（8/101）。烟雾测试合成数据未混入。场景tape摘要 `f2e6f7b2d7590989e305be5d7c6faf685369e15563a7b2fefb18c0468e7aa510`；真实来源与逐分支执行证据保存在world-model-windows.jsonl及environment-construction.jsonl。

任务语义为arrival_to_region / physical_arrival，到达半径0.0。标签与真实效用表示“首动作＋固定hungarian-v1-fixed续行至结束”的结果，不是首命令单独直接完成，也不是GPPO策略价值。决策前公开输入冻结，未来生命周期信息仅进入监督/审计。父场景在本研究内隔离，历史使用排除未证明；不能称为全新外部盲测，评价数据后续分析后亦不能作为新的独立确认集。

## 标签质量与unknown

表内均为有效/正例/负例/unknown，NOOP不具有目标任务。

| 划分 | 窗口/候选 | 物理按时完成 | 明确过期 | 主机确认 |
|---|---:|---:|---:|---:|
| 训练 | 24/341 | 317/269/48/24 | 317/48/269/24 | 306/160/146/35 |
| 模型选择 | 8/105 | 97/87/10/8 | 97/10/87/8 | 92/48/44/13 |
| 预测评价 | 8/101 | 93/77/16/8 | 93/16/77/8 | 89/39/50/12 |

物理完成/过期的unknown均来自NOOP：24/8/8。主机确认额外unknown为未观察到完成通知接收：11/5/4。unknown保留掩码，没有填零或当失败。24个训练父场景均具备物理正例、过期正例及有效主机标签，冻结数据门通过。

## 完整续行收益与候选排序

下面是父场景宏平均；MAE/RMSE是透明分数加学习残差还原后的标量完整续行效用误差，区别于生产指标中的六维outcome误差。独立stdlib分析从逐候选向量奖励重新构造真实效用，复算选择、并列规则、regret、Top-1与冻结门，无模型加载或新增前向。

| 方法 | 效用MAE | 效用RMSE | regret | Top-1 | 成对方向准确率 |
|---|---:|---:|---:|---:|---:|
| transparent | 0.091836 | 0.116226 | 0.234002 | 0/8 | 0.351296 |
| G1 | 0.074824 | 0.097780 | 0.050967 | 4/8 | 0.577775 |
| G2 | 0.075449 | 0.098553 | 0.101352 | 3/8 | 0.585875 |

G1相对透明方法regret改善0.183035（约78.2%），7/8父场景不劣、3/3种子改善，满足原门（改善≥0.005、不劣≥6/8、正向种子≥2/3）。最大父场景占正向改善总和约27.2%，收益不完全依赖一个窗口。

G2相对G1平均regret恶化0.050385，虽7/8不劣，但0/3种子改善，冻结增量门失败。集成差异集中于train-0099：G1 regret0，G2为0.403076；其余七个窗口集成regret相同。不能把事件Brier改善当成排序或调度收益增量。

**基线限制：透明方法在8/8窗口选择NOOP，G1/G2均不选择NOOP。** 此透明方法是冻结的乐观解析代理，不是强Hungarian策略任务对照。因此当前改善可能包含对该代理失配的纠正，尚未证明对强规则或GPPO具有实用增量；不能根据这些窗口新增规则后在同一数据上声称独立验证。

## 种子、模型选择与checkpoint

| 种子 | G1 regret | G2 regret | G1 Top-1 | G2 Top-1 |
|---|---:|---:|---:|---:|
| 8201 | 0.050967 | 0.101432 | 3/8 | 1/8 |
| 8202 | 0.118420 | 0.135078 | 1/8 | 1/8 |
| 8203 | 0.050967 | 0.101352 | 4/8 | 3/8 |

G1三个种子改善方向一致，但regret范围0.050967..0.118420，三种子动作完全一致仅3/8窗口。G2同样仅3/8动作一致，三个配对种子均劣于G1。集成未消除种子差异，也未事后挑选最佳种子。八个评价父场景支持本次方向判断，不足以证明跨分布或多次训练的广泛稳定性。

| 路线 | 实际epoch | 选定epoch | 首epoch总loss | 末epoch总loss |
|---|---:|---:|---:|---:|
| G1/8201 | 20 | 6 | 1.226487 | 0.441220 |
| G1/8202 | 19 | 4 | 1.317915 | 0.427551 |
| G1/8203 | 20 | 14 | 1.268153 | 0.437922 |
| G2/8201 | 20 | 8 | 1.593059 | 0.556428 |
| G2/8202 | 17 | 2 | 1.696081 | 0.555336 |
| G2/8203 | 18 | 3 | 1.639983 | 0.558006 |

模型按冻结选择集规则选定：最低regret，其次outcome MAE，其次较早epoch，采用既有早停。G1/G2损失含项不同，总loss不能用于跨方案优劣判定。六次保存及六次加载在账本完成；metadata与导出文件SHA-256一致。最终审计仅读取checkpoint字节计算摘要，没有再次加载模型。

## 事件监督的独立证据

自动事件集成Brier：G1 0.231531 → G2 0.029816；ECE10-bin 0.409190 → 0.063439；NLL 0.656087 → 0.142562。有效事件头必须分别解释：

| 自动事件头 | 有效 | 正例 | 负例 | unknown |
|---|---:|---:|---:|---:|
| public_field_change | 34 | 34 | 0 | 67 |
| new_measurement_received | 101 | 101 | 0 | 0 |
| continuation_publicly_confirmed | 101 | 89 | 12 | 0 |
| physical_completion_observed | 0 | 0 | 0 | 101 |
| host_confirmation_observed | 0 | 0 | 0 | 101 |

public_field_change和new_measurement_received的有效样本均为正例；physical_completion_observed和host_confirmation_observed完全无有效监督。因此这个自动事件Brier改善不能证明“所选动作会按时完成并得到确认”的预测改善。另设的生命周期horizon头确有有效完成/过期/确认监督，不能与上述即时自动事件头混为一谈。

| 生命周期horizon头 | G1 ensemble Brier | G2 ensemble Brier |
|---|---:|---:|
| physical_on_time_completion | 0.140279 | 0.140022 |
| task_expired | 0.140253 | 0.140810 |
| host_confirmation | 0.145791 | 0.158712 |

物理完成Brier仅微小改善，过期及主机确认Brier变差；当前不支持事件辅助带来一致的任务后果增量。

## 资源结算、完整性与清理

| 计数 | 实际 | 冻结上限 |
|---|---:|---:|
| environment_steps | 5396 | 14160 |
| resets_upper | 40 | 40 |
| branches | 547 | 1000 |
| public_rule_decisions | 4849 | 13160 |
| model_initializations_or_loads | 12 | 12 |
| checkpoint_writes | 6 | 6 |
| checkpoint_loads | 6 | 6 |
| world_batch_forwards | 6432 | 6768 |
| world_sample_evaluations | 90324 | 169200 |
| world_optimizer_updates | 2736 | 2880 |

账本failed0、pending0；反向2736，GPPO更新0、任务对照0。原生controller wall978.273764秒、CPU SELF＋kernel waited CHILDREN535.524980秒；Windows controller CPU0.1875秒另计，已测不重叠CPU合计535.712480秒。supervisor533.072777秒是嵌套分项，不重复累加。RSS峰值上界2,537,992,192字节，PyTorch分配峰值37,822,464字节、reserved峰值41,943,040字节；原生活动138,362,956字节、验证导出128,964,311字节。已测范围均在冻结上限内，各阶段分项见FINAL_RESOURCE_SUMMARY.json。

Windows wall963.911949秒与原生wall978.273764秒相差14.361815秒，来源未确定。保留跨时钟域原值，采用较大值核对3138秒总上限，不校正、不称为噪声。WSL桥接CPU、controller最终写入/退出尾段、桌面GPU逐进程显存未完整测量，CPU为轮询限制、无cgroup硬限制；辅助审计/监控不属于研究进程树。**完整资源验收仍为false**，与训练完成分开报告。任务决策CPU均值10ms、wall p95 50ms未评价。

运行进程已退出，原生只读检查无本任务残留，没有杀其他进程。最终Windows与native冻结文件、导出和checkpoint摘要一致；证据全部放在包外，封存数据未改写。

## 腾讯会议要求对应

| 要求 | 本次实现与依据 | 仍未证明 |
|---|---|---|
| 预测动作后果 | 因果公开历史/图表示＋动作条件JEPA，预测公开状态、任务/能耗残差与生命周期概率；G1标量效用误差降低 | 真实GPPO策略下的多步价值及跨分布泛化 |
| 反事实比较合法候选 | 真实分支首动作＋冻结Hungarian续行，101评价候选；G1 regret/Top-1改善 | 替换续行策略后的收益与强规则优势 |
| 作为RL决策先验 | 已保留候选适配接口；本轮仅离线排序评价 | GPPO实际采样、PreCo更新及真实收益，本轮未运行 |
| 用真实任务收益证明价值 | 本轮保存真实模拟器分支效用，支持候选排序评价 | 策略任务收益≥0.01及10ms/50ms成本标准均未评价 |

## 交付与建议

- 数据：verified-export/run-once/world-model-windows.jsonl；实际配置：environment.json。
- 六路模型：verified-export/run-once/world-model-checkpoints/{G1,G2}/seed-{8201,8202,8203}.pt。
- 逐候选/逐种子/集成预测：verified-export/run-once/prediction-trace.jsonl。
- 原始指标、账本与结算：prediction-metrics.json、budget.sqlite3、resource-settlement.json；独立复算：independent-metric-recalculation.json。
- 导出摘要：verified-export/export-hashes.json；最终身份核验：final-artifact-verification.json。
- 独立复核：postrun-independent-review.md已返回，详见原文；主助手另行进行了身份和指标复算。
- Git本地小型代码/协议/结果归档及实际远端状态见archive-status.json；凭据、token、checkpoint、SQLite和大型原始制品不上GitHub。

下一步仅建议准备另行授权、预注册且保持多目标/偏好/PreCo的G1-GPPO任务验证，纳入公开历史、透明方法和真正Hungarian公平对照及完整决策成本。G2当前无独立排序增量，不因事件Brier优势直接进入任务验证；可先对现有训练/选择数据做离线诊断，不追溯改门槛。当前确认数据已用于分析，新的独立确认需要重新隔离身份。此建议不启动任何新实验。
