# 本次四臂正式运行：技术停止，任务结果未评价

Attempt：`w1-g1-gppo-phase-accounting-repair-v1-once`。本次授权已消费并封存，没有重试、修复后续跑或替代 attempt。

正式入口退出码 **1**。首次异常是 `RuntimeError: conditional_policy_training wall shutdown reserve`，位于冻结 `supervise.py:181` 的阶段 wall 检查。训练阶段尾段实测 wall **3600.830220 秒**，上限 3600 秒；其 CPU **2478.891734 秒**，上限 6000 秒。因此是阶段 wall 停止，不能称为 CPU 超限或世界模型机制失败。SIGTERM 后 worker 返回 `-15`；首次 supervisor traceback 原样保留。

## 实际训练与保存

| 方法／种子 | 已完成环境步 | 已完成更新 | pending 更新 | checkpoint |
| --- | ---: | ---: | ---: | --- |
| G0／8301 | 2048 | 128 | 0 | 已保存 |
| G0／8302 | 2048 | 128 | 0 | 已保存 |
| G0／8303 | 2048 | 128 | 0 | 已保存 |
| T／8301 | 2048 | 128 | 0 | 已保存 |
| T／8302 | 2048 | 128 | 0 | 已保存 |
| T／8303 | 2048 | 128 | 0 | 已保存 |
| G1／8301 | 2048 | 128 | 0 | 已保存 |
| G1／8302 | 512 | 28 | 1 | 未保存，中断 |
| G1／8303 | 0 | 0 | 0 | 未开始 |

七个 checkpoint 字节摘要均与路线记录及导出清单一致。未加载模型来核验；正式策略 checkpoint 恢复阶段未进入，恢复结果为“未评价”。账本的三次 checkpoint 加载是冻结世界模型 8201、8202、8203 的加载，不是策略恢复。完成路线的真实 GPPO 更新及历史回放 trace 保留；各路描述性训练 loss／梯度／偏好相似度／PreCo 指标另存 `formal-result-audit/training-metrics-descriptive.json`。它们不能替代任务收益。

G1-8301 路线记录世界模型冻结、世界模型更新 0。生产总路线结束时的全模型终态摘要没有生成，不能把未执行的全程冻结终验称为通过。G1-8302 的 pending 更新可能已有部分计算，完成程度 unknown，既不算完成，也不填成零。该路内存中尚未持久化的 trace 未补造。

## 用户要求的结果判定

| 要求 | 本次结果 |
| --- | --- |
| G0、T、G1、H 四臂任务效用 | 全部未评价，任务 episodes 为 0 |
| G1 相对 G0、T、H 的效用差 | 未评价 |
| 跨八父场景／三策略种子稳定性、bootstrap | 未评价 |
| 完成、过期、确认、能耗的收益分解 | 未评价 |
| 各方法完整决策 CPU 均值／wall p95 | 全部未评价，没有任务决策样本 |
| 冻结收益门 | 未评价；没有事后判通过或判研究失败 |
| 冻结决策成本门（10ms／50ms） | 未评价；不能用训练 CPU 或空列表代替 |
| 资源验收 | 阶段 wall 失败，完整资源验收未通过 |

本次没有任务 episode、任务候选预测 trace 或 decision-costs 制品，不能生成虚假的零收益、零成本或候选结果。保留此前 G1 排序改善及 G2 无排序增量的独立结果；本次不新增“G1 帮助 GPPO”或“G1 无效”的科学结论。

## 资源、账本和计量边界

| 资源 | 实际记录 | 冻结上限／说明 |
| --- | ---: | --- |
| 环境步 | 14848 完成 | 全局 22752；训练阶段 18432 |
| 策略更新／反向 | 924 完成，1 pending | 全局／训练 1152；pending 保留预留量 |
| 任务 episodes／任务对照调用 | 0／0 | 240／1 |
| 世界模型 batch forward | 2560 完成 | 全局 7440；训练 6144 |
| 世界模型样本评价 | 8495 完成 | 全局 186000；训练 153600 |
| 模型初始化／加载 | 11 完成 | 全局 24；训练 12 |
| checkpoint 写入／加载 | 7／3 完成 | 全局 9／15；训练 9／3 |
| reset | 1073 完成 | 全局 18681 |
| 公开规则决策／世界模型更新／GPU | 0／0／0 | 本次未进入 H 评价；不重新训练世界模型 |
| Windows 正式入口 wall | 3629.399041 秒 | 全局 5070 秒 |
| 原生控制器 wall | 3628.619906 秒 | 与 Windows wall 嵌套，不相加 |
| 原生控制器 SELF＋内核回收 CHILDREN CPU | 2499.036717 秒 | 原生 cap 7760 秒，实测范围内未超限 |
| Windows 控制器 SELF CPU | 0.031250 秒 | 实测；30 秒预留不当作消耗 |
| supervisor 范围 CPU | 2490.217164 秒 | 已含在原生控制器范围，不能重复加 |
| 峰值聚合 RSS | 1279746048 bytes（约 1.192 GiB） | 4 GiB |
| 原生活动存储 | 101491683 bytes | 2 GiB |
| 验证导出存储 | 88338076 bytes | 原生＋导出合计 189829759 bytes，4 GiB 上限 |
| controller 结算／导出阶段 | wall 6.047350 秒、CPU 1.030355 秒 | 各 120 秒／80 秒 |

SQLite 共 **51017** 条记录：**51016 complete，1 pending**。唯一 pending 是 `ppo_update:G1:8302`，预留了 1 次反向、1 次更新和各 64 个 actor／encode 样本；其实际完成量无法确定。完整按状态、阶段和预算上限的核对见 `formal-result-audit/ledger-audit.json`，不能把预留量当作已完成调用。

worker 被终止而没有最终结算，因此 outer fallback 明确标为 **incomplete**，`ledger_snapshot_status=unavailable_worker_final_missing`，缺失值保持 null。后续只读 SQLite 复算已另存，不能追溯改写原 fallback 或宣称 worker 正常结算。外层已保留开放训练阶段 CPU 尾段，其计数是总量的分项视图，不再重复相加。

Windows/WSL 桥 CPU、控制器最终写入／退出尾段仍未测量；无 delegated cgroup，CPU 上限为轮询监控，不能宣称内核硬限制。用户已接受这些启动缺口，但 `full_resource_acceptance=false` 保持不变。包外只读监控和离线审计成本独立记录：24 份保存监控的 SELF CPU 合计 2.465542 秒，另有未保存原始文件的早期监控记录及最终只读核验（0.272522 秒）；离线审计和归档未做完整生命周期精确计量，不并入嵌套研究 CPU，也不报告为零。

## 封存、身份及独立复核

- 启动前核对包、申请、唯一入口、未消费状态及科学／预算不变量，未发现未披露变化。外部授权用标准 JSON 序列化、新 token 绑定；包内申请始终 `NOT_APPROVED`。
- Manifest SHA-256：`9ac065202126484968227469c16634ffbdfd5c30933c73a5c3a407e6c26815f9`。
- Hashes SHA-256：`a4d2be66a4834fe62ee461d0e5ba7255b77b0ee2adef1702b94f41c48fd12ae1`。
- Request SHA-256：`4eda5196e3fc0fce1c69f74b93b0b97c1a1335c655fc4007bce95d823a5f252b`。
- 208 项 manifest 输入及两项身份元文件组成 210 文件冻结包；源包、WSL 原生副本和历史 frozen-archive 身份分别复核。28 项导出 payload 哈希均一致，另有清单自身；payload 已通过本次导出哈希检查，worker 完整结算仍未通过。
- 本次原生路径已消费，最终只读核验未发现本任务残留进程。包外 plaintext token 已定向清除；不声称物理安全擦除。授权／消费证据保留，不归档秘密。
- Luna 独立只读复核了源码、停止原因、七 checkpoint 摘要、账本、incomplete fallback 及导出；没有加载模型、checkpoint 或环境，没有增加动态调用。其结论与本报告一致。

完整原始证据位于 `verified-export/`；复算、checkpoint 身份、逐路状态位于 `formal-result-audit/`。原始数据、账本、checkpoint 和首次异常未修改。本次停止后没有追加研究动态调用或重新启动正式入口。

下一步只能先离线定位为何本机冻结 CPU 条件下训练无法在 3600 秒阶段内完成，再决定是否提出独立修复／新预算申请；本轮没有执行该工作或发起新研究运行。

Git 签名和实际远端状态另见 `formal-git-archive-status.json`；仅核验远端提交后才能称远端归档完成。
