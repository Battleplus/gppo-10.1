# W1 EAWM/JEPA 复用与实际改动

日期：2026-09-29。状态：`DESIGN_READY_FOR_REVIEW`，未批准动态预算，未启动环境、checkpoint、训练或策略运行。

## 复用边界

| 已审查资产 | 复用内容 | 本包实际处理 |
|---|---|---|
| `gppo_world/jepa.py` | EMA、target stop-gradient、表示防塌缩 | `w1_graph_jepa.py` 改为 Graph-5 节点集合和25动作合同；不复用旧18动作配置 |
| `gppo_world/jepa_two_stage.py` | 节点 token、动作端点、历史条件和冻结顺序 | 保留因果边界；本包首版只实现一步公开目标，不复制旧11-token布局 |
| `gppo_world/events.py`、`event_training.py` | 事件目标和 masked BCE 的机制 | `public_event_targets.py` 仅从公开 receipt/measurement 和明确执行标签生成；事件权重独立开关 |
| `gppo_world/graph5.py` | 4 UAV、3 region、4 target、6 task、4 event、24候选+NOOP | 适配器接收同一25动作语义，不能把旧17动作或旧NOOP=16混入 |
| `gppo_world/joint_gppo.py` | 历史编码、候选批量预测和合法 mask 的接口形状 | 本包不复用其剩余效用头作为 JEPA 标签；GPPO 适配器只加入冻结的候选先验 |
| `gppo_world/gppo_adapter.py` | 版本绑定、disabled/zero/stale 回退原则 | `gppo_jepa_adapter.py` 在 scale=0 时逐元素恢复基础 logits，非法动作永不可选 |
| 已修复 launcher/ledger/exporter | Windows→WSL、native staging、账本、受控导出 | 仅做静态身份和入口可达性核验；本包不创建 attempt、不消费预算 |

## 明确不复用

- 旧17动作权重、数据集、动作编码和NOOP编号；
- 旧 J-01/J-02A/B 的任务收益结论；它们作为负结果和诊断证据保留；
- 旧标量剩余效用标签作为 JEPA 的状态转移真值；
- 当前历史运行中没有落盘的候选逐行预测，不补造为预测证据。

## 本包新增的最小差异

1. 决策前 `PublicSnapshot` 深拷贝及输入哈希，future evidence 在构造时拒绝；
2. `measurement_at`、`received_at`、`age`、known/valid、continuation identity 分字段保存；
3. 动作条件潜在预测、公共状态读出和事件头分离，事件损失可独立关闭；
4. 25动作候选批处理和 GPPO 合法 mask 适配器；
5. 只读生产链核验和一次性预算申请，保持 `NOT_APPROVED`。
