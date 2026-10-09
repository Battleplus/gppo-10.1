# 公共前缀合同

唯一数据结构为 `PublicPrefix`，唯一扁平化实现为 `serialize_public_prefix()`。完整字段、类型、shape、范围和序列化格式见 `PUBLIC_PREFIX_CONTRACT.json`；逐段偏移见 `flat-layout-manifest.json`。

原 v4 的 `flat[769]` 为 event_signal，float32，字节偏移 3076。UAV/Region/Target/Task/Event 图节点共 672 个数，relations 占 96 个数；归一化时间在 768，事件标量在 769。事件 validity 在结构中保存，不额外添加到既有 770 维向量。

原生产者先生成公开图、公开 trigger 和标量事件，再产生 flat；v4 恢复消费者重新观测环境后只覆盖了结构化事件，未更新 flat 缓存。保存的 time=4 公开对象同时具有 event_signal=1 和 flat769=1；失败消费者的 flat769=0。快照公开对象由 canonical JSON 解析，未反序列化正式 pickle、未执行正式环境或模型。内部 trigger 变化的更深原因保持 unknown。

v5 原生产 `_observation()` 只生成命名字段，然后调用同一 serializer；恢复候选从新格式快照包中读取保存的 pre-action 公共字段及绑定记录。读取私有环境快照仅用于恢复分支及严格 canonical 校验，不重新观察该环境，也不从当前/结束状态重算前缀。

生产 NumPy float32 到 JSON float 列表的转换只发生在声明的生产边界。消费者不得用 tuple/list、float64/float32 或 int/float 的隐式转换抹平差异。公共控制器在任何转换前检查原类型；graph/world/replay 输入再核对相同来源及候选绑定。

未绑定候选的原始公开观测使用 candidate_key=null、task_slot=null；进入 collector、label、world input、prediction、replay、GPPO 候选边界后必须使用完整结构化 key。NOOP 的当前目标槽位为 null，其他任务和全局后果保留。数值占位和 unknown 不作为监督负例。

有效 mask 是公共安全执行过滤后的 mask，与原始环境关系特征中的公开许可信息不是同一层。来源摘要覆盖原始公开观测；完整 PublicPrefix 与 candidate context 另绑定最终合法 mask 和因果 history，不能靠替换来源摘要绕过行为证据。

同一来源下先比较命名字段、类型、shape、valid、值、parent/window/time、候选身份及快照身份。批候选先全部验证再生成向量。缓存 flat 按命名 scalar 来源读取比较；769 的失败报告包含 `$.event_signal (flat[769])`、两边值和来源摘要。

原模型尺寸、三种子顺序、透明公共字段、奖励、多目标 GPPO/PreCo、critic-only 接入、推进门及训练评价次数不变。新 checkpoint metadata 必须带 public_prefix_contract；replay 保存同一前缀绑定，old/new log_prob 的已验收行为语义不变。本轮没有运行其概率或更新计算。
