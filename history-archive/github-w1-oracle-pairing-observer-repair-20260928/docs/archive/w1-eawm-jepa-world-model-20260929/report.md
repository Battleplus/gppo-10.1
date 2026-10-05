# W1 EAWM/JEPA 迁移准备报告

日期：2026-09-29。状态：`IMPLEMENTATION_PARTIAL_NOT_RUNNER_READY`。

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

## 仍未证明

真实 W1 标签是否足以训练稳定 JEPA、事件概率是否校准、候选排序 regret 是否下降、GPPO 任务效用是否超过透明基线和 Hungarian，以及 CPU/wall 实用标准，均未评价。当前不能宣称世界模型有任务增量。

## 启动阻塞

已验证的跨系统基础设施可以复用，但现有 production backend 只实现旧 A/B 标量
实验。本包尚未实现真实 collector、六个JEPA训练路由、十二个GPPO训练路由和五臂
任务 backend，也没有从 Windows 入口贯通这些阶段的替身集成证据。详见
`BLOCKERS.md`。因此不生成 execution manifest 或正式启动命令，也不接受动态授权。

## 唯一结论

本包完成了可运行的核心模型、严格公开输入/候选先验、阶段门和数值预算；生产研究 backend 尚未完成，不能启动。当前申请保持 `NOT_APPROVED`，不得采集、训练、加载checkpoint或进入任务对照。
