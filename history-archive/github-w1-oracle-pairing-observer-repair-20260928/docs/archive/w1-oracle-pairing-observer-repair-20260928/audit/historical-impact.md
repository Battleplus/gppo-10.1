# 历史证据影响

## 本次7个oracle分支

维持原状态：技术停止，不能用于oracle收益分析。

虽然135个telemetry粗键已证明没有外生fate不一致，14个command/ACK粗键也证明归组无效，但原运行没有保存所有原语调用身份，且冻结技术门确实返回失败。离线审计不能把该门追溯改写为通过，不能从7个分支计算或发布收益上限。

实际消耗继续累计：90环境步、1次reset、84次公开规则决策、7个分支；模型和训练调用为0。封存账本、状态和导出不修改。

## 两父场景通信动态技术验证

不受本次校验器缺陷影响。

该验证直接包装冻结通信原语，保存method、完整随机identity、参数和返回值，再按精确调用地址比较。它没有使用oracle runner的`_message_identity/_message_outcomes`。validation-0000与validation-0007的共同调用均非空，分支独有调用也非空，配对fate mismatch为0。

本审计不扩大其结论：它仍只支持逐包身份、配对随机fate和当时技术接线，不支持任务收益。

## 修复后的完整公平轻量重跑

不受本次oracle校验器缺陷影响。

公平重跑和oracle包使用同一修复环境源码：

- `m10_environment.py`：`0f4615d11604a7cd1cc74c94f0258dfc6cb59748f64a075276ec90941db094b1`
- `m10_communication.py`：`fcaada4fea48446aae509fb97f4b6b8275d5b6b9dda3bcdef00c5c392e6011ce`

误报代码位于oracle专用runner，公平重跑没有进行多分支`communication_delta`粗粒度配对检查，也没有据此改变方法行为。因此现有公平重跑的任务结果、方法排名、预测诊断和成本记录不因本次发现而失效。

这不改变此前研究结论的限制：公平重跑仍是单种子开发比较，FULL没有取得相对强规则的实用增量，历史10ms成本失败继续保留。

## 仍未知的影响

现有oracle日志不足以逐次核验原始allocation command/ACK的完整随机地址。若未来修复oracle runner，需要在新的、单独授权的技术门中直接记录原语调用；本轮不能据此假定新校验器必然动态通过。
