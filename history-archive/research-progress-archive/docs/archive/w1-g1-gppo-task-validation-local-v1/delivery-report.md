# G1＋GPPO 公平任务验证准备交付

正式任务尚未启动。当前 RESOURCE_REQUEST 为 NOT_APPROVED；新 attempt 为
`w1-g1-gppo-task-validation-local-v1-once`。本次工程准备不继承旧世界模型运行授权。

## 现有证据和透明参考

旧真实 W1 的 G1 相对透明完整续行参考宏平均 regret 改善 0.18303516，
三个世界模型种子均改善；G2 未取得排序增量，保留负结果，不扩训。
透明八窗口全选 NOOP 的 101 候选评分已经精确复算：理想续行把任务项
拉平，主动动作增加能耗，NOOP 严格胜出。没有发现符号、mask 或平局
算术错误。这只支持相对该乐观参考的排序优势，尚不支持 GPPO 收益或
胜过 Hungarian。事后 first-step 敏感性不是预注册结论，也不用于调参。
见 transparent-independent-review.md/json。

## 固定任务方案

G0、T、G1 采用同一原 GPPO 架构、原向量奖励、偏好相似度及 PreCo；
Hungarian 为真实冻结规则。T/G1 先验系数固定 0.1，G0 为零。
policy 8301/8302/8303 分别配对冻结 world 8201/8202/8203，最后策略
checkpoint，不按任务收益选模型、种子或系数。G1 标签对应首动作＋
Hungarian 固定续行，迁移为 GPPO 候选先验是本次待验证假设，不能称为
GPPO 价值函数。不改变环境、奖励、合法执行边界或事件头。

训练固定 24 父场景 train-0064..0087；任务为 train-0104..0111 八父场景
各三 repeat，三方法各三种子加共享 H，共 240 episodes。已核与本轮
世界模型开发场景的真实和结构身份不重合；跨项目历史未使用仍未证明。

任务要求 G1−G0>0、G1−T>0、G1−H≥0.01，至少 6/8 父场景相对 T 不劣，
至少 2/3 种子相对 T 正收益，配对父场景 bootstrap95% 下界对 T>0。
G1 自身完整决策 CPU 均值≤10ms、wall p95≤50ms；CPU0/线程1，GPU0预算。
无模型决策为未评价。全部协议和停止条件见 package/G1-GPPO-PROTOCOL.md。

## 实际工程证据及边界

集中生产合同与合成数值回归：34 项，exit 0，
wall 33.923817s，SELF CPU 6.328243s。
完整命令见 preparation-tests.invocation.json，逐项输出见
preparation-tests.stdout.txt。测试期间全部包文件 SHA 保持相同；随后只
给继承文档加历史标识，并将开发缓存移出包原样保留，生产源码未更改。
部分非法动作、仅 NOOP、非零先验下真实 GPPO 更新损失/梯度有限，更新前
行为 log-prob 回放误差≤1e-6。另见 gppo-preparation-validation-evidence-20261004.json；
其 35 项较早测试与本次 34 项集中测试是不同集合，不相加冒充独立验证数。

最终隔离合成驱动器 exit=1，不写成 exit=0：其唯一失败项是验收脚本错误地
要求每方法重放384观测。实际episode reset会清空历史；三个route每方法
共36重放观测，对应SQLite36完成调用/36encoder计费。主AI只读核对全部
一致，另由Luna复核，不改原失败证据、不追加训练。生产流水线本身完成，
原19项驱动检查中的其他18项成立；纠正是包外验收器勘误，非研究门修改。
详见 retained-synthetic-independent-audit.json 和最终witness的原始退出状态。
较早非隔离驱动器原生制品未找回，不能宣称失败原始文件全部保留；来源
及可得输出按其保留记录披露，不把该次缺失证据作为正式验收依据。

受控生产流水线和实际操作计数详见 synthetic-pipeline-final-evidence.json
及完整包外 witness。替身只提供合成环境交互及合成世界模型，策略编排、
原损失、优化、策略 checkpoint 读写、候选预测、240 episode 矩阵、
指标复算、SQLite结算和导出实际运行。正式训练仍是每路2048步/128更新；
合成仅128步/2更新。不读取真实 G1 checkpoint、不运行真实模拟器，不能
把合成收益或合成门状态解释为研究效果。Windows→WSL 正式supervisor/worker
没有以真实授权启动；该部分复用原链并做源码/只读检查，不能称新正式全流程验收。
最终合成账本记录5472环境替身步、18次反向/更新、24次模型初始化/加载
预约、1680次world batch forward、10140候选样本、6273actor/6381encoder样本，
9个真实合成策略checkpoint写/读；账本checkpoint读15中另6次是内存
world替身，不能冒充反序列化真实权重。其他准备测试也有实际计算，累计
完整操作数未测量；不报告整个准备阶段为零。测试账本沿用proposed attempt
名称字段，但执行在独立fixture目录且显式synthetic_test，无正式授权消费；
正式native路径未创建状态由最终WSL预检核验。
最终合成矩阵的 G1 完整决策 CPU 均值56.5483ms、wall p95为159.3178ms，
均超过保留的10ms/50ms实用门。这是该合成工程路径的实际计时预警，
不能写成成本通过，也不能直接当作冻结真实模型的任务收益结论。
正式成本仍未评价；不因此缩小计时边界或放宽门槛。
可测合成进程最后退出前采样wall643.39s、SELF CPU269.98s；这是下界范围，
没有包含最后采样之后的退出尾段或Windows/WSL桥，不能报告完整CPU验收。

Luna 源码、计数、最终字节审查见 luna-final-review.md/json。
纯 ServiceClock 测试证明等于 deadline 的 moving-arrival 路径先过期再异常，
不会得到完成记录。此原生技术边界保持不变、单独勘误，触发时封存停止。

## 冻结身份与原样预检

- Manifest：`219cd82a046ebffd3feb3f1e2f9bc5b158f7ab040273d6baef5b866b01ff2895`
- Hashes：`aea59645b889fda413fa9ef110453622d402019016132b0daf59d5779d6d465f`
- RESOURCE_REQUEST：`85939a3d6211dc86c5db2a56a368b657c42975468b48dadc141e99b7a7e57e8a`

Windows 和 WSL 原样只读预检分别 exit 0 / 0，
完整命令、stdout/stderr、原始输出bytes见 final-*preflight.*。
全部 209 文件前后身份不变，外层摘要复核一致；
frozen-archive/package 是同字节完整副本。证据见 final-delivery-evidence.json。
staging_started=false、worker_started=false、authorization_consumed=false。
结构预检使用 NOT_APPROVED 模板，无 token，不等于正式用户授权验证。
正式唯一命令见 package/unique-launch-command.md；实际获批后才标准JSON
写包外授权并创建新 token，旧已消费授权不会迁移。

## 待批准的一次联合任务预算

环境步 22752；策略更新/反向各1152；
world batch forward7440、world候选样本186000；actor/encoder样本各114480；
模型初始化/加载24，checkpoint读15/写9，reset上限18681，H规则432；
任务240 episodes；wall5070s、CPU7790s、RSS4GiB、活动2GiB、聚合4GiB；
world训练更新、新分支采集、GPU均为零。阶段分项以 RESOURCE_REQUEST.json 为准，
不跨阶段借额度；技术失败或超限封存，不补跑。获批后连续执行完整九路训练
和完整四臂评价，任务收益和 G1 成本独立判断。

Windows/WSL桥CPU、最终写入退出尾段、轮询和共享CPU干扰仍有明确缺口；
full_resource_acceptance不能默认为true。实际可测超限仍停止；训练完成、
资源完整性和科学结论分开报告。

## 归档状态

完整本地冻结副本见 frozen-archive/package；Git只归档小型代码、协议、摘要
和哈希，排除 token、权重、SQLite、tape和大型trace。签名代理不可用；
同身份直接签名探针超时且没有生成签名，未创建未签名提交、未推送。
远端未归档，网络本轮未验证。实际本地暂存与原staged保护结果以
git-archive-status.json为准，不将暂存称为远端完成。

当前可以申请新任务验证，尚无新增正式任务研究结论。

| 腾讯会议要求 | 当前证据 | 本次任务申请将验证 |
|---|---|---|
| 预测动作后果 | G1预测六维后果残差、下一公开状态/潜在状态与horizon生命周期；固定Hungarian续行数据有排序收益 | 换为GPPO续行后，冻结预测先验的迁移价值 |
| 合法候选反事实比较 | 上次逐候选真实分支及三个种子排序证据；透明参考弱点已解释 | 在线批量候选预测和动作改变记录；不伪造未选动作真实标签 |
| 为强化学习提供先验 | 原多目标GPPO采样/回放同prior、mask、因果历史已合成核验 | G1相对同预算G0/T的真实episode收益 |
| 真实任务收益与实用成本 | GPPO真实任务未评价；合成成本门失败 | G1相对Hungarian≥0.01及自身10ms/50ms门 |
| 事件辅助独立增量 | G2已有排序增量负结果，缺有效自动事件头限制保留 | 本轮不扩训或追溯改判G2 |
