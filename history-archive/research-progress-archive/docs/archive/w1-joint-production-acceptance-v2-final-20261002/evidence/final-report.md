# 联合生产验收 v2 交付

结论：本地受控生产worker链与计量回归通过；服务器新生产路径尚未实际执行。
`runner_ready=false`；研究与工程申请均为`NOT_APPROVED`，正式attempt未创建。

## 关闭证据与边界

- 短命子进程：内核waited累计用量保留后代CPU，活跃树单次求和；子集变动/PID身份不符即停止。
- 退出尾段：真实子进程在最后快照后忙算0.08CPU秒；外层waited结算恢复至少0.07秒，并不重复加内层快照。
- 阶段归属：worker/socket同步确认与quiescent边界；结算阶段和总量使用同一时点快照。
- 异常路径：保留已回收CPU下界、cleanup错误、wall与进程退出情况；缺失范围总量为null、accounting_complete=false。
- 登记handoff绑定attempt/manifest/hashes及实际inner/outer PID；文件读取拒绝测试通过。
- 实际outer.main超时测试：真实子进程timeout→kill/wait/reaper→handoff→失败记录通过；登记写入和收尾硬限制使用底层替身。
- 未覆盖：真实服务器FAILED登记、全部异常组合及D-state/中断fsync；不能保证跨存储原子终态。
- 无delegated cgroup时运行中CPU/RSS仍采样式；Windows收尾按预留计费，不能声称内核硬封顶。
- 历史0.274925秒差值来源未确定；原失败包/结论/证据保持不变。

## 生产链

实际runner.main→worker身份/源码/tape校验→collector→数据门→六路线训练→checkpoint恢复→
逐候选预测→独立复算→SQLite结算→受控导出已经在本地合成副本运行。
替身只位于底层环境；真实模型架构、损失、优化器及文件读写均执行。
矩阵为2/2/8、CPU、1 epoch，与正式24/8/8及服务器CUDA不同，不是科学效果验证。
本轮实际合成模型总计36初始化/加载、234前向、30反向/更新、18次checkpoint写及读、1170样本评价。
集成初次失败揭示合成来源manifest未包含新tape，已修夹具身份而未绕过校验；所有失败输出保留。

## 最终冻结身份

包：`E:\Z博士\research-plans\w1-task-outcome-g1-g2-joint-production-acceptance-v2`

- execution_manifest_sha256: `b895f970c4f969a24654bb2c02e515d5cc41d6f62eb7a7fafa22e5ab72cce907`
- hashes_sha256: `d80329069803ede6b9cd27e75fd3d7100188281d5b2fdc5640cd9741c107af34`
- resource_request_sha256: `54b473d3b765daac00edc5d59449a3619c5d1eb446901a317bcad14026bc5b5a`
- server_acceptance_request_sha256: `2ecf75cf30ef995389ce0c7baad5418cfe1566a433d57894c97d03477b507626`

Windows正式研究入口结构预检、工程入口结构预检及WSL原生诊断副本共享校验均退出0；
最终全文件前后字节相同。预检不证明服务器依赖/资源，也不代表授权通过。

## 待批准服务器验收

工程job：`w1-joint-production-acceptance-v2-engineering-once`。命令和token/登记合同见包内SERVER_ACCEPTANCE_PROTOCOL.md。
总wall 180秒 / CPU 360秒，
GPU1独占，启动时至少9GiB空闲，owned显存8GiB、RSS4GiB、活动512MiB、聚合1GiB。
合成调用24初始化/加载、150前向、18反向/更新、12checkpoint写及读、750样本评价、1CUDA上下文。
不安装依赖、不改共享运行时、不访问封存确认集、不创建正式研究attempt。
需要新的有限授权及使用者真实姓名，不能使用共享账号替代。

当前另有明确阻断：远端预检/SSH/SFTP帮助进程30秒CPU仅为未核验收费预留。
入口对此fail-closed：即使远端生产测试通过也不报告overall complete，保留cpu_scope_unverified。
因此本包可审阅和结构核验，但仍不能宣称全工程验收通过或建议直接正式研究执行。

## 联合研究申请（服务器验收之前不可执行）

研究总量保持14160步/40reset/1000分支，3138wall秒/4595完整CPU秒；GPPO与任务对照调用0。
数据门不通过立即停止学习；通过才自动G1/G2训练与独立预测，不重复阶段审批。

|阶段|wall秒|CPU秒|
|---|---:|---:|
|label_qualification|828|445|
|prediction_confirmation|180|360|
|settlement_and_verified_export|120|80|
|staging_and_zero_step_gate|180|80|
|world_model_training_and_selection|1800|3600|
|Windows/跨系统预留|30|30|

## 归档

小型代码、协议与证据在本地Git目录保留；既有暂存不修改。
ssh-agent为Stopped/Disabled，GitHub只读查询连接重置；远端未归档，未创建未签名提交。
当前不能申请直接正式采集—训练启动。还需闭合远端帮助进程CPU范围，并获得服务器有限验收授权及真实姓名。
服务器验收和计量复核完成后，才能冻结runner_ready身份并提交联合研究执行申请；不增加研究矩阵或模型目标。
