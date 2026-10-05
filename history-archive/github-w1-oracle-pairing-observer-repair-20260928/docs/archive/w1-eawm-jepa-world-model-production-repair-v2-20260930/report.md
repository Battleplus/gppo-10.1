# W1 EAWM/JEPA 迁移准备报告

日期：2026-09-30。状态：`PRODUCTION_REPAIRS_STATIC_VERIFIED_NOT_APPROVED`。

## 已完成

- 读取并继承四份 EAWM/JEPA 复用审计与计划；确认 T-03、T-05、J-01、J-02A/B、R-02 已有实现和负结果，不重复包装；
- 新增 W1 专属公开时间/有效性合同、事件目标、25动作 Graph-JEPA、解耦损失和 GPPO 合法候选先验；任务后果头预测透明原效用之外的6维残差，正式先验只使用透明分数加已指定的原效用残差，不直接以不可解释 latent MLP 打分；
- 新增 `gppo_w1_integration.py` 生产候选桥接和 `w1_training.py` 授权训练边界；训练 API 只在未来授权 runner 中执行，当前测试仅验证冻结接口，不调用优化器；
- 新增 `w1_public_adapter.py` 公开输入适配层：显式接收 telemetry sidecar，保留 measurement/receipt 时间、field mask、历史和 continuation identity；不导入或构造环境。生产 Graph-5 转换通过显式 builder 注入，builder 只能读取公开白名单字段；严格评分入口只接受冻结适配结果；
- 基于修复后训练 tape 生成 `parent-split.json`：24训练、8模型选择、8预测确认、8任务确认，按 tape 顺序在观测结果前分配，父场景无重叠；其他历史使用是否存在仍标为未证明。
- 通过19项纯逻辑测试：未来证据拒绝、unknown不填零、事件/主机/物理标签分离、公开 telemetry 时间边界、重复遥测保留、continuation identity 不推断、Graph payload 隔离、graph/mask/哈希绑定、严格生产评分入口、25动作批量、EMA target 无梯度、事件损失开关、零先验恢复、非法mask和预测门阶段阻断；
- 通过既有生产链只读预检：Graph-5/25动作、runtime dependency launcher、WSL Linux guard、`--preflight-only` 均存在，动态调用为0，未创建attempt/worker/SQLite；
- 另按已验证的 Windows 入口执行了一次继承运行时链的 `--preflight-only`：WSL return code 为0，`preflight_pass`，CPU-only torch `2.8.0+cpu`、解释器 `/usr/bin/python3.12`，staging/worker 均未启动；原始输出保存在包外证据目录，未消费新研究授权；
- 生成 `static-audit.json` 和完整包内文件哈希，便于外部授权绑定。
- 生成 `budget-derivation.md`，明确一次性申请的计数公式和当前不能伪造动态额度的最小缺失字段。
- 冻结24/8/8/8父场景、一步候选窗口、G1/G2及GPPO种子/epoch/步数，生成可复算的数值 `RESOURCE_REQUEST.json`；状态仍为 `NOT_APPROVED`。
- 新增 `production_data.py` 的原生 collector：校验训练 tape 原始字节摘要，在候选分支前保存公开输入、telemetry 时间、continuation 和输入哈希，分支只写标签；NOOP 的任务相关标签保持 unknown。
- `runtime_backend.py` 已接入 collector、六条世界模型训练路线、预测确认、十二条策略训练路线和312条任务确认；`runtime_hooks.py` 只封装 native 环境、模型、优化器和 checkpoint 底层调用，测试替身不能替换整个 backend。
- 新增 `test_production_runtime.py`，用底层替身验证正式 adapter 到达各生产阶段；策略种子固定为 8301/8302/8303，task confirmation 按8父场景×3 repeat展开。
- 策略训练路由已改为直接运行冻结世界模型下的原生环境/PPO transition loop；不再调用会联合更新世界模型的旧 `joint_training.run_group()`。G1/G2 的行为先验保留在 `PriorConditionedPolicy` 回放中，路由预算按64步 rollout、每rollout 4次更新计数。
- 按静态审查补修 GPPO replay logits：原生偏好/PPO 损失收到有限未掩码 logits，合法 mask 只作用于采样分布；行为采样时的 prior、mask、候选特征、preference 与 hidden 在回放 transition 中保留。
- 修复独立复核发现的生产缺陷：Graph-JEPA 推理从共享 `public_history.history_vector` 读取因果历史；collector 将 reset、前缀规则决策、候选强制首动作/扫描和父窗口快照分项计费；数组 JSON 规范化优先调用 `tolist()`，支持多元素 NumPy/Torch 张量。
- 外部一次性授权统一由共享合同函数校验。Windows、WSL 暂存和 native 入口分别核对包外 authorization 文件、attempt、预算摘要、manifest/hashes 与 token hash；冻结包不含待审批 token hash。运行时对原生 `gppo_world` 的44个 Python 源文件做集合和摘要完整校验。
- 新增授权链、共享历史预测、真实 collector+SQLite 计费和完整运行时源树身份回归。以上修复不改变研究矩阵、门槛、种子或预算上限。
- 真实一步 `vector_reward` 计算的 `true_utility` 与透明评分分列持久化；预测 regret/oracle 只读 `true_utility`，透明评分只用于预测先验和对照。
- H 任务臂通过原 `ClassicalSelector("hungarian")` 逐决策运行；原向量效用按偏好、缩放和折扣独立累计，原 scalar 环境奖励单独记录。
- 训练 rollout、任务episode、采集窗口和 JEPA 推理使用同一 `CausalPublicHistory`；策略 hidden 跨决策传递并随 episode 重置。terminated 的下一价值为0；truncated 与 rollout cut 使用最终公开状态 bootstrap。仅 NOOP 仍运行 critic 和历史编码。
- 生产 collector 实际调用 `PublicTransition` 与 `build_event_target`。补充首见字段边界：决策后首见且 receipt 晚于决策边界才标新测量正例；receipt 不能证明在决策后时保持 unknown，不再由于其他字段可比较而产生伪负例。物理完成与主机确认仍区分且未知不填零。
- 机会状态按整个任务 episode 累积；复位时没有非 NOOP 机会但后续出现机会的 episode 仍保留真实效用，真正无机会才输出 `utility=null`。
- 首轮行为测试后纳入 `test_review_repairs.py`、授权及生产运行时回归；最终测试数、正式调用计数和合成数值操作以本版本最终 `checks.json` 为准。资源验证器和静态审计共同约束源树身份与环境构造边界。
- 最终复核修复后的测试数、合成调用计数、资源校验与源树身份审计以新 `checks.json` 为准。尚未完成独立测试副本上的 Windows→WSL→supervisor→worker 底层替身全链验收，因此 `runner_ready=false`，不能把本包称为跨系统全链验收通过。

## 仍未证明

真实 W1 标签是否足以训练稳定 JEPA、事件概率是否校准、候选排序 regret 是否下降、GPPO 任务效用是否超过透明基线和 Hungarian，以及 CPU/wall 实用标准，均未评价。当前不能宣称世界模型有任务增量。按冻结分支，G1/T 与 G2/G1 预测门分别记录；只有覆盖和两项门全部通过才进入完整五方法任务矩阵，否则所有政策/任务阶段均未评价。

## 尚未授权事项

代码修复和静态测试已完成，但本轮没有启动真实环境、初始化正式模型、加载 checkpoint
或执行正式训练。合成数值测试调用的微型假 GPPO 与 Graph-JEPA 计数另列于 `checks.json`，不可称为零前向/零更新。尚未用获批的一次性 token 执行正式任务，也没有任何效果或成本结果。必须在外部授权绑定新的清单后再执行；`RESOURCE_REQUEST.json` 继续为
`NOT_APPROVED`。

## 唯一结论

本包完成了静态生产修复、因果公开输入/候选先验、阶段接线、独立 G1/G2 预测门和闭合数值预算；效果仍未评价，不能把静态测试通过写成世界模型有效。当前申请保持 `NOT_APPROVED`，本轮未启动采集、正式训练、checkpoint 加载或任务对照。

## 修复版本 2 状态

新独立目录为 `w1-eawm-jepa-world-model-production-repair-v2`，执行身份为
`w1-eawm-jepa-world-model-production-repair-v2-once`。原预算、研究目标、父场景划分、
G0/T/G1/G2/H 比较、G1/T 与 G2/G1 两项预测门及成本标准保持不变。v1 的归档 manifest 和
hashes 完整通过验证；v2 将单独生成外部授权绑定。

此次依据 Luna 的追加检查补上 native 入口摘要函数导入，并使策略训练、任务评价环境构造
走统一底层环境边界。组件测试验证边界调用；正式环境/模型/优化器/checkpoint 未调用。

生产实现仍有一个准备验收阻断：没有 manifest 绑定的 test-only native entry/worker 可在
不绕过 sealed package 校验的情况下从 Windows→WSL→supervisor 到达生产 worker。现有阶段
mock 不能代替这项测试，所以本版暂不标为 `RUNNER_READY`。这不影响七项审查问题的静态修复
证据，但不能回答真实预测和任务收益是否达标。
