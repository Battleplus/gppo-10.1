# Oracle分支通信配对根因报告

日期：2026-09-28  
性质：零环境步、零模型调用的事后技术审计

## 结论

本次停止是oracle runner配对校验口径造成的误报，现有证据没有显示真实外生随机配对失效。

冻结校验器的原计数已按完整事件序列复现：983个“共同身份”、149个不一致唯一身份、388次分支间不一致比较；149项按link分为telemetry 135、ACK 6、command 8。分析保留了同一身份在单分支内的全部重复记录，没有用字典覆盖发送、接收或重传阶段。

149项全部得到解释：

- 135项telemetry：每个分支保存的发送/丢包起点完全一致。使用冻结`CommunicationProfile`、场景seed 925710000、完整packet identity和相同oracle exogenous key离线复算，135项在所有出现分支中也都与保存的起点一致。这些消息均在`t=13`或`t=14`产生；冻结校验器比较了后续`received`事件是否出现。不同首动作使分支分别在`t=13`、`t=14`或`t=15`终止，较早终止的分支不会再产生相同的后续接收记录。135项的跨分支序列长度全部不同。这是执行后果差异，不是外生扰动差异。
- 14项command/ACK：全部同时包含lease renewal记录，并且同一`command_id`在不同分支对应多个不同动作。冻结校验器仅按`link + command_id + ordinal + attempt`归组，因而把原始分配命令、动作相关命令、renewal发送、renewal接收及renewal ACK合并。它们实际使用的随机地址分别包含`command_id|version|action`、`renewal_id`或`renewal_id|ack|ordinal`，不属于同一个外生随机事件。

第一处偏离位于冻结runner的身份构造，而不是共享环境：

- `runner.py:148-159`优先取`identity/message_id/command_id`，未保留通信原语方法、完整随机地址、renewal identity及命令动作/版本。
- `runner.py:162-177`把发送随机结果与后续`received/expired/stale_or_duplicate`处理组成整个序列。
- `runner.py:189-199`只要该粗粒度键出现在两个分支，就称为共同语义消息，并要求整个事件序列相等。

冻结合同只要求共同消息具有相同的 keyed exogenous outcome；它同时明确允许action-specific消息不同。共享运行时在`m10_environment.py:297-303`使用修复后的完整packet identity，在`m10_communication.py:86-116`按`seed + profile + method + full identity`确定性计算fate。oracle、两父场景技术验证和完整公平重跑的环境源码SHA-256均为`0f4615d11604a7cd1cc74c94f0258dfc6cb59748f64a075276ec90941db094b1`。

## 复现口径

“共同身份”是冻结粗键在至少两个、但不要求全部七个分支中出现。每个分支内部，同一粗键的所有事件按日志顺序保留为tuple，再把各分支完整tuple比较。前缀通信不在branch summary的逐步`communication_delta`中，因此149项不是复制前缀造成的。

这一定义有两类越界：

1. telemetry的`sent/dropped`是通信随机原语结果，之后的`received/expired/stale_or_duplicate`取决于分支是否继续、队列何时冲刷、视图版本及终止时刻。
2. command/ACK日志未记录传给随机原语的完整identity。相同计数器产生的`command_id`不能证明消息语义相同，renewal更有独立`renewal_id`。

所有149项和逐条来源见`message-mismatch-table.csv`。最小实例包括：

- telemetry：`task|task-0|deadline|14|13.000000000`在各分支均以同样fate发送并计划`t=14`到达；`t=13`终止的分支没有后续`received`行，`t=15`终止的分支有。来源见表中branch-summaries第1、2等行的JSON pointer。
- command：`m10-mixed-0-cmd-00002`在首分支动作中分别对应0、1、6、7、18、19和NOOP 24，且之后还出现多个`renewal_id`。将它们称为同一消息不成立。

## 排除与限制

- 没有发现外生key丢失或随机函数非确定性的证据。135项可复算telemetry全部通过。
- 没有发现源快照被分支修改的证据。runner在每个deepcopy前后及分支后验证pickle摘要，并且本次未触发隔离停止；所有telemetry起点也保持共同fate。
- 现有日志不能完整证明command/ACK的真实共同fate，因为原始分配命令没有保存`policy_version`和传给随机原语的完整地址。它们只能确定为“不应按当前粗键比较”，不能追溯声明command/ACK配对门已通过。
- sender、receiver和telemetry payload没有写入本次`communication_delta`。报告中的随机地址、参数和fate复算均明确标为离线重建。
- 本次只定位首单元校验失败，不把7个分支重新包装成有效oracle结果，也不把原技术门追溯改成通过。

## 资源边界

本审计环境实例、reset、step、模型初始化、模型前向和训练更新均为0。本次已封存attempt的90环境步、1次reset、84次规则决策和全部历史成本保持不变；历史10ms成本失败继续有效。
