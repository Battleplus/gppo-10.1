# G1＋GPPO 四臂任务验证协议

状态：待批准。此协议取代本副本中继承的历史 smoke、标签采集及 G1/G2 训练说明；历史文件只保留溯源，不提供有效启动入口。

## 研究问题与既有证据

检验在相同 GPPO 框架内加入冻结 G1 动作条件后果先验，是否改善真实 W1 调度收益，并达到相对 Hungarian 的实用增量及决策成本标准。

本轮不重新采集世界模型分支数据、不训练世界模型、不扩训 G2。上次真实数据的 G1 集成 regret 为 0.050967，透明参考为 0.234002，三个种子改善，G2 增量门失败。原结果原样保留。独立离线审计说明：透明参考理想化地将剩余公开任务视为免费完成，主动动作与 NOOP 的任务项相同，前者增加能耗，所以八个窗口严格选择 NOOP。未发现评分算术、mask 或平局错误；这不是胜过强透明调度的证明。事后第一步分数敏感性分析不替代冻结指标、不用来调整本轮系数。

G1 预测公开历史及首候选动作下的六维后果残差、下一公开状态、潜在状态和 horizon 生命周期概率；收益标签对应“首动作＋固定 Hungarian 续行”。GPPO 后续行动不同，因此将这个结果作为候选先验属于待验证迁移假设，不能称为 GPPO 策略价值函数。自动事件头中缺少有效物理完成/确认标签的旧负面限制仍保留。

## 比较与固定路由

| 方法 | 行为策略 | 候选先验 | 策略种子 |
|---|---|---|---|
| G0 | 原多目标 GPPO＋公开历史 | 0 | 8301/8302/8303 |
| T | 同一 GPPO | 0.1×原公开完整续行透明分数 | 8301/8302/8303 |
| G1 | 同一 GPPO | 0.1×(同一透明分数＋配对 G1 残差效用) | 8301/8302/8303 |
| H | 冻结真实 Hungarian | 不调用策略或模型 | 每父场景 repeat 共享一次 |

策略 8301/8302/8303 分别配对世界模型 8201/8202/8203，使用上轮模型选择集选出的 checkpoint，完整 bytes 及 metadata 见 g1-model-binding.json。无集成挑选、无任务结果选择模型或系数。所有 GPPO 路由同架构、同初始化种子、同父场景顺序、同偏好安排、同合法 mask、同奖励和相同更新预算。保留两目标 clipped surrogate、返回值偏好相似度及 PreCo logit 梯度方向；不是普通单标量 PPO。

候选批量预测只使用决策前公开观测及因果历史。策略行为保存的 prior、mask、候选和历史用于回放，更新不能重新算成另一套先验。世界模型 eval、requires_grad=False，训练前后全参数哈希核对。无奖励塑形、想象 PPO、GES 或动态系数。

## 数据与任务语义

实际配置来自 environment-config-contract.json。冻结 arrival_to_region / physical_arrival、到达半径 0.0、single_shot 完成通知及全部环境参数，实例化后先核对实际配置。物理完成、明确过期、主机确认分别记录；未能证实的状态为 unknown/null，不补零。

策略训练父场景固定 train-0064..0087（24），复用世界模型训练身份；任务评价固定 train-0104..0111（8），每父场景 repeat 0/1/2，身份、结构和外生键见 parent-split.json。已核对与本次世界模型训练/选择/分析 train-0064..0103 的场景及结构不重合。整个项目历史未使用情况尚未证明，不能宣称全新独立泛化测试集。评价隔离从本协议冻结起执行，不根据任务结果换场景、补选或追加种子。

每个策略路由训练 2048 环境步，rollout 64，每 rollout 四次更新，共 128 更新；保存最后 checkpoint，无 task-based 选择。训练偏好 [0.2,0.8]/[0.5,0.5]/[0.8,0.2] 按 episode 与种子索引固定轮换，评价偏好 [0.8,0.2]。任务效用完整保存原两维逐步奖励并按 0.99^step×(0.5×p_task×task_reward＋p_energy×energy_reward) 独立复算，与透明评分分开。

评价 240 episodes：8×3×(3 方法×3 策略种子＋1 H)，每 episode 最多 18 步。所有方法在父场景/repeat 使用相同外生随机键，不能逐动作随机重采。terminated 不 bootstrap；truncated 与 rollout 截断使用下一因果公开观测的 critic bootstrap，episode reset 清空历史。无机会不等于效用未知：完整有效 episode 即使仅 NOOP 也保存真实效用；缺失奖励/未完整 episode 保留未知并使覆盖失败。

## 预注册判定

历史 G1 预测资格通过且 G2 未通过后，正式批准才允许跑全部九个策略路由和完整四臂评价。技术错误、非有限数、身份/账本/预算失败立即停止并结算，不修复后续跑。没有运行时选择部分矩阵的分支。

聚合顺序：同父场景/repeat 先平均三个策略种子；再平均三个 repeat；最后八个父场景等权宏平均。H 共享 episode 不复制成三次独立样本。报告每个种子、父场景、repeat 的 paired utility difference，以及收益集中程度。

任务增量要求全部满足：八个父场景和全部 repeat 完整；G1−G0>0；G1−T>0；G1−H≥0.01；相对 T 至少 6/8 父场景不劣；至少 2/3 策略种子相对 T 正收益；以父场景为配对单位、10000 次 bootstrap（8401），G1−T 的 95% 区间下界>0。该 bootstrap 仅描述八父场景分布，不能替代种子不确定性或推广到总体。

成本独立判定：G1 自身完整决策 CPU 均值≤10ms、wall p95≤50ms，且实际有效决策样本数>0。无模型决策为“未评价”，空列表不能通过。G0/T/G1/H 逐条分别保存，A/B混合均值不可用于 G1 验收。计时从公开输入/特征构建前到模型前向、先验融合、actor、mask和动作选择结束；不包含事后trace写入、实际环境执行，后两者另计总成本。计时内的同步SQLite调用预约/确认也保留，H规则调用使用同样的计费边界。这是实际带审计路径的保守完整决策成本，不是纯模型kernel耗时，不为达到门槛扣除账本开销。

所有路由用 CPU0、torch/interop/BLAS线程1、CUDA_VISIBLE_DEVICES为空。复用现有 Python3.11.16 / torch2.7.0+cu128 解释器和锁定依赖，但不初始化CUDA。共享本机CPU干扰如实记录；训练耗时不替代决策计时。上次GPU训练成本与本次CPU策略/任务成本分别披露。

## 记录与工程停止

保存 nine route 训练损失（两个目标、偏好相似度、PreCo、有限梯度、KL）、rollout identity、实际调用账本、九个策略 checkpoint 哈希及恢复核验。每个决策保存 method/seed/parent/repeat/decision_step、输入哈希、因果历史、mask、全部合法候选、透明分数/G1残差/融合prior、policy概率、所选动作、无prior反事实选择与 prior 改变次数、候选数和独立 CPU/wall。反事实只比较同一 actor logits 的有/无prior选择，不增加环境或模型调用。

保存完整 episode vector rewards、真实效用、完成/过期/确认和unknown、能耗、NOOP比率、动作序列、外生键、config与场景身份，逐候选模型 trace 不进入在线私有输入。初始输出与status必须先于SQLite。技术失败保留首次 traceback、pending/failed调用和已完成计算。

本次任务阶段不预算新的反事实环境分支。未选动作的在线预测可以保存，但没有对应真实分支标签，不能计算新的反事实regret或将其冒充独立预测验收；动作排序证据沿用上轮限定条件，任务迁移是否有效由本轮真实episode收益判断。

计量采用原 Windows→WSL 原生控制器、supervisor、SQLite和受控哈希导出。CPU父进程 SELF＋内核 waited CHILDREN 不与嵌套账单相加；监督阶段用已有进程树/边界握手。Windows/WSL桥、最终写入退出尾段和非原子轮询仍有计量限制，完整资源验收不能默认通过。请求含跨系统30秒wall、Windows控制器30秒CPU预留，超限停止，阶段额度不借用。

## 预算和唯一入口

RESOURCE_REQUEST.json 始终 NOT_APPROVED。本轮只准备，不生成实际APPROVED授权或token、不创建正式attempt。预计策略训练＋任务阶段共22752环境步、1152策略更新，世界模型更新0。全部阶段和计数上限以资源文件为准，审批必须绑定其最终SHA-256。

唯一Windows入口 run_g1_task_validation.py。旧 run_local_research.py 已禁用自动授权。正式运行需要包外 external-authorization.json 和新一次性 token（stdin），外部授权绑定 manifest、hashes、request摘要；源码manifest记录内容文件，hashes记录manifest摘要，最外层由外部授权绑定，无循环。

结构预检与正式授权校验分开；最终预检证据放包外，验证前后全部内容文件身份不变。原包、旧attempt与旧负结果不修改。结果只能得出实用成功、局部改善或没有增量之一；未进入阶段为“未评价”。GitHub只归档小型代码、协议、摘要、哈希，不上传checkpoint、token、SQLite或大型原始数据；只有远端commit核验后才称归档完成。
