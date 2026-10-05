# 任务预算依据与计量限制

RESOURCE_REQUEST.json为NOT_APPROVED。本文件解释已有申请数字，不授权执行，不改变研究或资源门。

| 项目 | 计算依据 | 上限 |
|---|---|---:|
| 策略环境步 | 3方法×3种子×2048 | 18432 |
| 任务环境步 | 8父场景×3repeat×(3方法×3种子+共享H)×18 | 4320 |
| 策略反向/更新 | 9路×(2048/64)×4 | 各1152 |
| G1训练先验batch forward | 3路×2048，每次全部合法候选 | 6144 |
| G1任务先验batch forward | 8×3×3×18 | 1296 |
| world候选样本 | (6144+1296)×最多25合法候选 | 186000 |
| 训练actor样本 | 9×(2048行为+128×64更新+2048 bootstrap预留) | 110592 |
| 训练encoder样本 | 同上限；bootstrap预留同时覆盖更新后的公开episode隐藏状态重放 | 110592 |
| 任务actor/encoder样本 | 8×3×3方法×3种子×18 | 各3888 |
| 模型初始化/加载 | 训练3 world+9 policy，任务3 world+9 policy | 24 |
| checkpoint读/写 | 训练3冻结world读+9策略写，任务3world+9policy读 | 15读/9写 |
| H规则决策 | 8×3×18 | 432 |

更新后隐藏状态只在每个rollout的四次更新全部完成后，重放当前episode已经消费的公开flat序列一次。下一步当前观测尚未消费，不入重放；episode reset清空。冻结horizon18、interval1意味着每个rollout最多17个已消费状态，32个rollout最多544个encoder样本/route，无额外actor/world调用。保守再计ceil(2048/18)+32=146个horizon/rollout bootstrap，合计690低于已有2048个encoder预留；不需要提高任何数字上限。terminated不bootstrap。若非冻结时间边界被突破，则技术停止，不临时借预算。

总wall5070秒：零步120、策略3600、任务1200、结算/导出120，加跨系统30。总CPU7790秒：80+6000+1600+80，加Windows控制器SELF30。RSS4GiB、活动存储2GiB、原生与校验导出聚合4GiB；GPU、world训练、world优化更新全部为零。reset上限保留早期终止的最坏情况，而不假定所有episode都跑满18步。

逐调用限额由SQLite预约与确认执行，failed/pending不退款。父进程SELF与内核waited CHILDREN覆盖不同进程，不叠加嵌套子进程自己提交的CPU账单。supervisor另记录实际进程树与阶段边界，防止跨阶段借用。

Windows/WSL桥CPU、最终写入和退出尾段、非原子进程轮询及共享CPU干扰仍有已披露缺口，没有声称cgroup硬CPU上限或完整全生命周期计量通过。实际已测wall/CPU/存储/RSS或逐调用越界必须技术停止。训练完成、科研门结果与完整资源验收是独立字段；不以缺口抹去实际完成计算，也不虚报完整验收。

准备合成fixture缩小策略训练步数/更新数，不改变正式请求；全部真实合成计算从其账本、底层调用和测试证据另列，不能报为零。内存返回的合成world-loader边界与实际checkpoint反序列化须分别计数。
