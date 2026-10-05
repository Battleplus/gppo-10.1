# 七路最终模型复用、两路补训及完整四臂评价

本协议是当前执行协议，覆盖继承文件中“全部九路重训”及旧预算说明；原科学定义仍按 G1-GPPO-PROTOCOL.md 与不变的矩阵执行。状态 NOT_APPROVED，本轮没有生成 token、启动研究 attempt、初始化模型、读取 tensor 为模型或执行前向／训练。

## 来源及恢复判定

来源 attempt：w1-g1-gppo-phase-accounting-repair-v1-once，阶段 wall 停止、worker -15、结算 incomplete，旧证据不修改。必须复用全部七路 protocol-final checkpoint：G0/T 的 8301、8302、8303，G1 的 8301。每路由同一冻结源码／配置生成，完成 2048 steps、128 updates，最后保存调用 complete；checkpoint metadata、state bytes 与来源／导出 hash 交叉核验，见 recovery-contract.json 和包外 audit。不是按任务表现挑模型；本次从未进入任务评价。

G1/8302 只有 512 环境步、28 complete update 和 1 pending，没有中间 checkpoint。未保存完整 RNG、模型/optimizer 当前状态、环境生命周期/通信状态、公共历史/GRU hidden、rollout缓冲、scenario/episode游标和pending完成位置，不能精确续训。该条仅允许重新从8302初始化，按原2048steps/128updates训练；旧512步及28完成＋1未知update全部保留并计入累计资源。8303首次从原种子训练。不能用残缺模型参与评价，不能续跑旧进程／token。

新 attempt：w1-g1-gppo-recovery-v1-once。新包内七个文件是验证后的字节副本，位于 frozen-models/reused-policy/，不依赖旧工作目录挂载。生产路径先核验 recovery_contract，train_policy_routes 在策略初始化、环境交互、优化器创建前跳过七路；仅G1/8302与8303实际训练。两路最终模型保存且完整性通过后，与七路合并为九路路径，统一进入原生产 policy_loader、任务评价、指标复算、冻结门、结算和受控导出。任务模型加载和状态摘要核验预算已单列；本轮未预先执行真实恢复。

## 不变的科学条件

保持 G0、T、G1、Hungarian 四臂；policy seeds8301/8302/8303和配对world seeds8201/8202/8203；GPPO两目标、偏好相似度和PreCo；架构、全部超参数、0.1先验系数、2048steps/128updates、因果历史/mask、奖励、配置及任务场景/外生键不变。不训练世界模型/G2、不扩采、不新增种子、不按模型表现选复用文件。

任务240episodes =8父场景×3repeat×(3策略方法×3seed+共享H)，每episode最多18步，原偏好[0.8,0.2]，CPU0与torch/interop/BLAS线程1。H不是三个独立种子样本。unknown保持null，utility可知不代表宿主确认可知。

冻结收益门：完整八父场景／三repeat；G1-G0>0、G1-T>0、G1-H≥0.01；相对T至少6/8父场景不劣；至少2/3seed相对T正收益；parent配对10000次bootstrap seed8401的95%区间下界>0。G1自身完整决策CPU均值≤10ms、wall p95≤50ms，且有实际模型决策，成本与收益分别判定。计时边界不缩小，训练wall不当作决策成本。未进入阶段写未评价。

## 新申请的分阶段预算

| 阶段 | wall 秒 | 完整 CPU 秒 | 新动态量 |
| --- | ---: | ---: | --- |
| 暂存／零步门 | 120 | 80 | 不运行环境、模型或checkpoint加载 |
| 两路G1补训练 | 3200 | 2600 | 4096环境步、256更新、2个最终checkpoint写入 |
| 完整四臂任务评价 | 1200 | 1600 | 240episodes、最多4320环境步；9策略＋3世界模型恢复 |
| 结算／受控导出 | 120 | 80 | 无新模型训练 |
| 跨系统／Windows预留 | 30 | 30 | 预留不是实测 |
| 总上限 | 4670 | 4390 | 8416环境步、256更新 |

其余完整分项见 RESOURCE_REQUEST.json：15次checkpoint加载、2次写入、17次模型初始化/加载、5392 world batch forward／134800 world样本、28464 actor／encode样本、reset4338、规则决策432、RSS4GiB、活动存储2GiB、原生＋本次验证导出4GiB、GPU0。旧存储独立保留并清单披露，不删旧文件求通过。阶段额度不能互借；任务阶段预算原样保留。

训练时间申请来自同次真实G1路由whole-route（包括未归属间隙）较慢每步速率×4096×1.5，再向上取整为3200秒；CPU没有逐操作计量，以阶段实测比例估算并上调为2600秒，非实测细分。评价尚无正式任务耗时数据，维持原1200wall/1600CPU，不宣称已验证充足；调用次数上界和来自训练决策日志的计时参考在预算依据中分别列出。日志未覆盖的开销不填零。训练不足则停止，不能挪用评价时间或部分跑任务矩阵。

累计资源单独见 cumulative-resource-contract.json。旧完成14848steps＋新cap8416＝23264steps，相比原22752额外512步；旧924complete＋新256＝1180更新，另保留1pending，预留累计上界1181。旧实测wall约3629.4秒＋新4670cap；累计CPU只加不重叠的外层范围，并保留桥接、退出尾段和离线审计计量缺口。旧授权不得迁移，必须新明确授权绑定Manifest/Hashes/Request。

## 一次性启动与失败规则

仅 package/run_g1_task_validation.py 为入口，见 unique-launch-command.md。包外授权和新token须在批准后用标准JSON序列化、回读和绑定检查生成；旧授权不复制。结构预检不会制造批准或消费身份。

七路输入或runtime不匹配立即在worker启动前拒绝；两路补训技术失败、budget、非有限数、checkpoint完整性失败即封存，不自动修复/重试，不评价不完整矩阵。完整评价成功后才分判收益及成本；失败仍结算、保留首次trace/pending、按现有fallback明确incomplete。旧研究结论不追溯改判。

本轮工程证据仅为无模型契约测试、真实生产调度函数的非计算回调测试、最终Windows/WSL结构/运行时预检；不会声称新增真实训练/恢复/四臂端到端验收已通过。静态可申请执行与动态效果验证分开。
