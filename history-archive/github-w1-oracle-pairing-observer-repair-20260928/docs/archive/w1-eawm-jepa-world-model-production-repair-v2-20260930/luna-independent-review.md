# Luna 独立复核

复核代理：`luna_worker`，模型 `gpt-6-luna`，推理强度 `max`。范围为只读源码/文档审查；未运行环境、模型、checkpoint 或训练，未修改研究源码。

## 核心结论

- 当前 `graph5.py` 是正确的 W1 宿主：5类节点、4 UAV、6任务槽、24候选加 `NOOP=24`。旧 `jepa.py` 的18动作、旧 `jepa_two_stage.py` 的 `NOOP=16`、旧17动作事件和 adapter 不能直接混用。
- 可复用的机制是 EMA、target stop-gradient、历史 GRU、动作端点、事件目标、masked loss、不可变快照和合法 mask；本包将它们绑定到当前25动作合同。
- 当前 `TaskPolicySnapshot`/`EvidenceItem` 没有同时持久化 measurement time、receipt time、age、实体身份和可读 continuation 身份；新标签必须补齐这些字段。commit 观测适合动作后标签，不能误称为动作前或共享随机性的 next 反事实。
- 旧 `data.py` 将拒绝动作映射为17并对缺失 cost 使用默认0，不能进入 W1 unknown/拒绝合同。
- T-03、T-05、J-01、J-02A/B、R-02 已有执行证据和负结果；再次声称首次引入 EMA、事件、latent adapter 或旧动作体系会重复包装历史工作。
- A-D 必须共享父场景、tape、外生随机身份、公开输入、合法 mask、策略预算、seed 与模型选择规则；C/D 唯一主差异应是自动事件监督。每臂成本要独立计量，不能共享均值掩盖额外前向。

## 对本包的复核结论

`public_transition_contract.py`、`public_event_targets.py`、`w1_graph_jepa.py` 和 `gppo_jepa_adapter.py` 覆盖了上述最小新增差异；`protocol.md` 将真实任务运行置于预测门之后，并保留旧实用门槛。真实标签、候选反事实和任务增量仍未评价。

## 第二次边界复核及处置

Luna 对新增 `w1_public_adapter.py` 的只读复核发现：最初版本可被旧候选桥绕过，
Graph builder 能看到完整 observation，continuation policy identity 未绑定，图没有进入
输入身份。最终版本已据此增加严格 `predict_public_and_score` 入口、builder 公开白名单、
21节点/24候选/25动作形状检查、图摘要和 continuation policy identity 绑定；历史编码器
只能接收冻结 payload。旧 `predict_and_score` 保留作低层兼容测试接口，不列为正式生产入口。

## 本轮 production backend 复核（2026-09-30）

复核代理仍为 `luna_worker`（`gpt-6-luna`, `max`），只读检查了
`runtime_backend.py`、`production_data.py`、`production_world.py`、
`production_policy.py` 和 `runtime_hooks.py`；未运行环境、模型、checkpoint 或训练。

- adapter 已实际调用 `ProductionDataCollector.collect`、
  `train_select_world_models`、`evaluate_prediction_confirmation`、
  `train_policy_routes` 和 `evaluate_task_confirmation`，不再保留五个
  `NOT_CONFIGURED` 占位方法；task unit 由8父场景×3 repeat展开，策略 seed 为8301/8302/8303。
- collector 使用训练 tape 原始字节 SHA-256，候选输入在分支前冻结并保留公开 telemetry
  的 measurement/receipt 时间；NOOP 的任务相关标签保持 unknown。
- 复核结论：静态生产路径已接线，但未做动态 Windows→WSL→worker 验收；效果、资源和
  跨系统一次性运行仍未证明，因此不能把本包标为 `RUNNER_READY` 或宣称收益。

本轮另有两个独立复核请求因服务端 503/429 未返回结果；这些请求不计为通过证据。

## 生产修复版本 2 的独立复核

复核代理 `luna_completion_audit`（`gpt-6-luna`, `max`）只读发现：native 入口先前引用
未导入的 `sha256_file`；策略与任务路径直接构造环境，绕过环境底层替身边界；没有实际
test-only native entry/worker。前两项已在 v2 源码修复并增加边界测试，第三项仍阻塞跨系统
生产 worker 验收。代理还确认正式入口必须继续拒绝 `integration_test=true`，不能为测试删掉
此防线。该复核没有调用环境、模型、checkpoint、训练或正式 attempt。
