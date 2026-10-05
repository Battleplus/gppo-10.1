# 四臂实验恢复准备交付

本轮仅离线审计、无模型合同／调度测试及结构／运行时预检。真实环境调用、模型初始化、checkpoint反序列化、前向、训练更新与研究attempt启动均为0。运行时探针导入torch只核对依赖身份；并未创建模型。旧运行及其incomplete结算保持不变。

## 复用与补训

复用G0、T各8301/8302/8303及G1/8301，共七路。每路2048环境步、128更新；文件哈希、ZIP原始参数storage的state哈希、方法／种子／配置、已保存Adam slot step=128、日志与SQLite保存complete交叉一致。每个Adam group有35个参数ID、33个已保存slot、LR=.0003；两个无slot参数未映射到参数名，不能据此判定原因，保留审计限制。评价只恢复完整策略state，不恢复optimizer；两路补训新建Adam，此限制不是恢复阻断。G1/8301绑定原G1/8201；详细身份在package/recovery-contract.json及audit/recovery-audit.json。仅原始字节审计，没有torch.load。

G1/8302只有512步、28个complete更新和1个pending更新，没有任何完整checkpoint，也缺少RNG、环境／通信、历史hidden、rollout及pending精确边界。因此不能精确恢复；提出从原8302种子重新训练2048步／128更新。G1/8303从原种子首次训练相同规模。仅补两路，共4096步、256更新，不重训七路，不按效果挑选复用。

## 耗时与独立阶段申请

旧阶段wall=3600.830s、CPU=2478.892s。账本调用区间wall合计835.592s；事件间未归属间隙2764.789s。确认每次资源预约重复扫描当前stage全部历史行，开销随账本增长且不在操作计时内；透明评分、fsync、调度／等待等也未分项计时，不能将全部间隙归为扫描或JEPA推理。

两路训练按实测较慢G1 whole-route速率0.4914253368s/步外推4096步，再乘1.5，上取整3200秒wall。CPU2600秒是按旧阶段比例估算并上调，非逐路线实测。完整评价预算保持原1200秒wall／1600秒CPU，尚未有正式评价耗时证据；到限停止，不能挪用评价额度补训。

|阶段|wall秒|CPU秒|动态上限|
|---|---:|---:|---|
|暂存及零步门|120|80|无环境、模型或checkpoint加载|
|G1两路补训|3200|2600|4096步、256更新、2次checkpoint写入|
|完整四臂评价|1200|1600|240episodes、4320步，原8父场景×3repeat设计|
|结算与导出|120|80|无新增训练|
|跨系统预留|30|30|预留不冒充实测|
|总上限|4670|4390|8416步、256更新|

其余分项以RESOURCE_REQUEST为准：模型初始化／加载17，checkpoint读15／写2，世界模型batch前向5392、样本134800，actor／encode样本各28464，reset上限4338，规则决策432，RSS4GiB、活动存储2GiB、原生＋验证导出4GiB、GPU0。各阶段不可借额。新申请NOT_APPROVED；没有生成token或正式授权。

旧14848步＋新8416上限=累计23264步，较原22752额外512步；旧924 complete＋新256=1180完成更新上限，另1 pending，累计预留上界1181。旧wall3629.399s＋新上限4670s=8299.399s；已知不重叠CPU加新上限约6889.068s，桥接及退出尾段未测量仍保留，不宣称完整资源验收通过。旧账本与新账本分开，不重复记七路训练为新调用。详见cumulative-resource-contract.json。

## 冻结与工程证据

- 新包：E:\Z博士\research-plans\w1-g1-gppo-recovery-preparation-v1\package
- 新proposed attempt：w1-g1-gppo-recovery-v1-once
- Manifest SHA-256：71b86783224b00f5295778aa39c6880cda9a37d9113f2a7b2941446507d68d33
- Hashes SHA-256：c7970d39e7aab5b43fe36512afb8cf73e008a26fa9b27e6bd36dae422d480b5f
- RESOURCE_REQUEST SHA-256：7f62d4f5d09c935e0e13241e7468097d9f0f54b9a1e245e060a74d379491168d
- 唯一启动命令：package/unique-launch-command.md（只能在新明确批准后生成包外授权及新token，再执行）。

Windows和真实WSL解释器10项无模型合同测试通过；真实生产调度函数以非计算底层回调验证仅训练两路、复用七路并合并九路。它不是checkpoint实际加载或完整动态端到端验证。最终Windows及WSL source预检exit=0，staging_started=false、worker_started=false；前后冻结字节与完整本地归档副本相同。独立Luna复核见audit中单独报告；主AI核对科学合同、接线及字节一致性。

entry_preparation_ready=true；runner_ready=false表示本轮没有实际补训／恢复／评价全链动态证据，不能虚报为已运行。包已具备新预算申请所需入口与静态合同，不需额外研究设计。架构、GPPO多目标／偏好相似度／PreCo、种子、步数、超参数、G1先验、场景与外生键、收益及成本门均保持不变。

## 执行与停止边界

申请覆盖两路完整训练→与七路合并的九策略checkpoint恢复→原四臂240episodes评价→独立指标复算→分判收益及完整决策成本→结算导出。任一技术或预算失败封存，不自动重试、不评价不完整矩阵。阶段与总量均执行上限；复用身份改变在启动前拒绝。

本轮任务结果、G1有效性和正式决策成本均未评价。七路完成与训练超时都不能判断G1有效或无效。最终科学终点仍是四臂效用、跨父场景／种子稳定性、收益分解及CPU均值10ms／wall p95 50ms。Git归档状态单列，暂存不等于远端归档。

下一步需明确批准以上新身份及完整分项预算；本轮到此停止，不启动新动态运行。
