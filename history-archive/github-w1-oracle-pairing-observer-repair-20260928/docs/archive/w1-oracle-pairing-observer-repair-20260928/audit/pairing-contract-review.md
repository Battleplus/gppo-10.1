# 配对合同与实现对照

## 合同要求

oracle协议的“Pairing and randomness”段要求：同一parent/repeat中的分支共享通信profile、故障tape、前缀、公开状态和exogenous key；只有语义相同的共同消息才必须得到相同的keyed外生结果。动作特有消息允许不同，且必须与共同消息检查分开。

该要求对应的是通信原语输入与返回值：

- telemetry：完整地址、seed、now和profile相同时，比较`dropped/outage/loss/duplicate/jitter`；
- allocation command：比较`command_id|observation version|action`对应的`command_delivered`；
- allocation ACK：使用同一完整command identity调用`ack_delivered`；
- lease renewal：比较`renewal_id`对应的renewal fate；
- renewal ACK：比较`renewal_id|ack|ordinal`对应的ACK fate。

最终是否进入公开记忆、是否过期、是否被版本检查接受、lease执行结果以及分支何时终止，是随机扰动之后的执行结果，不能直接替代外生fate。

## 冻结实现

| 环节 | 冻结实现 | 证据 | 判定 |
|---|---|---|---|
| 随机地址 | `_random_identity`将exogenous key与完整packet identity组合 | `m10_environment.py:297-303` | 符合修复合同 |
| telemetry随机fate | 先调用`communication.telemetry`，再记录`sent/dropped` | `m10_environment.py:680-703` | 可从保存起点复算 |
| telemetry最终处理 | 排队后可能记录`received/expired/stale_or_duplicate` | `m10_environment.py:705-774` | 受分支状态和终止影响 |
| allocation命令 | 随机地址包含`command_id|version|action` | `m10_environment.py:934-946` | 当前校验粗键丢失version/action |
| renewal | 随机地址使用`renewal_id`，接收和ACK另有阶段 | `m10_environment.py:402-441,626-659` | 当前校验粗键错误合并 |
| 冻结归组 | 取`identity/message_id/command_id`并附link/ordinal/attempt | `runner.py:148-159` | 不足以表示通信原语身份 |
| 冻结结果 | 比较所有日志阶段的完整tuple | `runner.py:162-199` | 超出外生配对合同 |

## 149项分解

| 类别 | 唯一身份数 | 外生起点 | 实际差异 | 根因 |
|---|---:|---|---|---|
| telemetry | 135 | 保存记录跨分支一致；冻结函数复算一致 | 后续`received`记录有无不同 | 将最终处理序列误作外生fate |
| command | 8 | 当前粗键不可比较 | 原命令动作不同，并混有多个renewal阶段 | 错误归组 |
| ACK | 6 | 当前粗键不可比较 | 原ACK与renewal ACK混组，动作也不同 | 错误归组 |

14个command/ACK身份全部同时满足：包含lease renewal行、跨分支原始step action不止一个。它们不是“同一个随机地址出现不同结果”。

## 与此前动态技术验证的差异

两父场景技术验证在`execute_once.py:170-194`用wrapper记录每次冻结通信原语收到的精确identity、参数和返回值；在`execute_once.py:365-416`按`(method, full identity)`取交集并比较原语结果。其validation-0000和validation-0007分别覆盖752与692个共同调用，fate mismatch均为0。

oracle runner没有该wrapper，改为从高层`communication_delta`反推随机fate。两条路径的证据对象不同，这正是本次误报产生的原因。

## 缺失字段

本次oracle记录没有保存：

- 通信原语实际收到的完整随机地址；
- allocation command调用时的observation version；
- 统一的sender和receiver；
- telemetry payload值；
- 每次原语调用的原始参数和完整返回值。

telemetry可利用semantic identity、time、seed、profile和源码确定性复算。allocation command/ACK缺少version，不能完整复算。缺字段必须返回证据不足，不能当作通过。
