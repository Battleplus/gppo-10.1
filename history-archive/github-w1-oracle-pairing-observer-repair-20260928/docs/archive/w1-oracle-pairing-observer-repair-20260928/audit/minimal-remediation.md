# 最小修复方向

## 应修改的位置

只修改新的oracle执行包中的配对观察与校验代码，不修改共享环境、通信profile、随机函数、场景、规则、奖励、阈值或历史制品。

最小可靠方案是在oracle runner创建环境后，为`env.communication`安装只记录的wrapper，沿用两父场景动态技术验证已经使用的模式：

1. 对`telemetry`、`command_delivered`、`ack_delivered`和`renewal`记录method、完整identity、全部参数和返回值。
2. 捕获分支快照时保存调用序号；每个分支只比较快照后的调用切片，前缀调用不得重复计入。
3. 共同外生消息键固定为`(method, full identity, normalized parameters)`。同键调用结果必须完全一致。
4. 只在一个分支出现的调用归为action-specific，不进入共同fate判定，但必须报告覆盖数。
5. telemetry的`received/expired/stale_or_duplicate`、命令接受、lease执行结果和最终任务结果另作执行后果日志，不参与外生fate相等门。
6. 同一完整调用身份在单分支重复出现时不得覆盖；若合同不允许重复，明确技术停止；若允许，使用调用ordinal并逐项比较。
7. 缺完整identity、参数或结果时返回`INSUFFICIENT_EVIDENCE`；共同集合为空不得通过。

因此应替换或收窄`runner.py:148-215`的`_message_identity`、`_message_outcomes`和`verify_message_pairing`，并在环境构造/快照路径中加入只记录wrapper。不能仅删除`received`行，也不能为当前149项建立白名单。

## 必需的回归测试

- 同一完整identity、key和参数产生同一fate；
- 分支独有消息不改变其他消息的随机结果；
- allocation命令以`command_id|version|action`区分；
- renewal与原allocation命令分组分离；
- renewal ACK按`renewal_id|ack|ordinal`区分；
- 同一外生fate但最终送达/接受不同不触发外生配对失败；
- 真正的原语返回值不一致会停止；
- 重复调用不被字典覆盖；
- 缺字段返回证据不足；
- 空共同集合不能通过；
- 分支快照后的调用切片不包含前缀记录。

本目录的10项纯逻辑测试覆盖上述核心分类，但没有实例化环境，不能代替未来动态接线门。

## 未来动态证据缺口

若之后另行批准修复验证，最小动态目标应仅为：

- 在一个固定单元中证明wrapper捕获的完整原语调用与环境`communication_log`起点一一对应；
- 共同调用fate一致且集合非空；
- action-specific调用集合非空；
- 分支顺序改变不改变共同调用fate；
- 快照、账本和资源闭合。

本轮不创建修复包、不申请预算、不运行该验证，也不继续剩余23个单元。
