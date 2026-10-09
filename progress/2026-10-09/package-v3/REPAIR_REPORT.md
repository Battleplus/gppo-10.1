# v5 修复与验收

v4 已封存。原首错为 time=4.0 首个候选（action=8、uav_slot=1、task_slot=2）的 `$.flat[769]`，生产者 1.0、消费者 0.0。该字段实际名称为 event_signal、float32、字节偏移 3076。完整 layout 和生成阶段见 flat-layout-manifest.json；首错、来源摘要、保存的公开对象及未推断的内部原因见 v4-first-error-flat769-trace.json。

v5 移除重观测加标量 override 的恢复路径。快照包保存 pre-action 公共状态、context、候选 keys、前缀记录及原环境/adapter/history；恢复只从保存的公共部分生成 PublicPrefix。快照创建时核验环境时间、随机键和历史，读取和恢复时保持 payload/canonical 摘要严格。

native producer、公共控制器、collector、branch snapshot、label builder、graph encoder、world input、prediction、replay 和 GPPO 的更新前校验器均调用同一 serialize_public_prefix，或调用仅委托该函数的 prefix_records/audit helper。graph、dtype、IDs、有效 mask、事件、time、parent/window、snapshot 和候选字段在新向量生成前验证。原 770 维缓存仅作派生值，不能压过命名字段；没有把期望 1 改为 0。

完整受控验收 40 项通过，实际 Driver/BudgetLedger/StageServer，底层仅合成环境：time 2/4/8/12、time=4 首候选、3/4/6 任务、4 个无人机槽位、NOOP、无机会、事件 0/1/unknown、pre/post 混用、类型/shape、候选顺序/缺失/重复/非法/篡改及 replay 更新前拒绝。预测导出仅用明示的合成占位读出，未计算模型预测。

生产 `_observation()` 方法另外在受控遥测桥上验证 24 个组合，未构造正式环境。清除 trigger 后再次观测为 0，保存的 pre-action signal/flat 仍保持 1；增加 3 个消费者转换前类型拒绝（float64、tuple IDs、int event）。最终代码还复用已准入合成缓存核验 16 个窗口、12 个机会窗口和 3 个 time=4 保存恢复，不重新采集。

测试证据不混同版本：40 项主验收在核心接线修复后完成；消费者转换前 guard、缓存检查的非物化比较和 snapshot 时间/随机键/历史绑定由 final producer/cached 补充证据覆盖。最终 cached-type 验收还通过实际 collector 与 Driver.features 确认类型篡改在 jsonable 转换及 flatten 前拒绝，新增环境步及模型调用均为 0。没有为小补丁重复完整合成采集。

开发过程中首轮因缓存摘要逐元素重复计算而主动中断，首错和部分 SQLite 保留；修复为每个缓存一次摘要。缓存补充恢复曾因合成 Fixture 的 __main__ 模块绑定缺失停止；修复只限测试入口导入该相同合成类，不放宽生产快照或模型类型。中断、失败和补充验证都单列计量，没有覆盖旧证据，也没有消费正式身份。

正式 checkpoint 加载、模型初始化、neural forward、世界模型更新、GPPO/critic 更新和任务评价为 0。GPPO 范围仅更新前身份校验器；未执行概率计算或优化器。碰撞 unknown 和合法差结果标签的原规则保留。任何工程通过都不是研究效果、风险认证或任务收益。

新申请保持原各阶段上限，绑定新 identity，不能复用 v4 授权；预算修订见 V5_BUDGET_REVISION.md。新模型 metadata 和 replay 必须包含 PublicPrefix 合同，禁止将旧 checkpoint 当新合同模型。

旧包、旧停止状态、SQLite、原始快照和导出的逐文件 SHA-256 在封存收据中保留并再次核验。冻结前只复制并修改独立 v5；v4 研究指标保持“未评价”。v5 服务器准入尚未执行，本包不能凭旧 preflight 自动获得动态资格。GitHub 归档本轮未执行，不阻断本地交付。
