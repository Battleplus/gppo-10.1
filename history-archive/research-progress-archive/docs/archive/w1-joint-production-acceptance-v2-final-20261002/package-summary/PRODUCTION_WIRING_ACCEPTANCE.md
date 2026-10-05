# 生产接线与验收范围

本文件不代表正式实验效果。研究包尚未获批；服务器验收另有独立请求。

|阶段|生产函数|本地实际证据|服务器状态|
|---|---|---|---|
|生产worker入口|runner.main / verify_worker_contract / verify_joint_inputs|测试副本实际执行main；只注入底层环境boundary和测试路径/身份|待独立工程授权|
|身份与运输|launch_joint_once.main / verify_external_authorization / upload_once|合同拒绝与本地结构预检|真实新路径未执行|
|原生寿命结算|metered_joint_entry.main / closure_accounting|真实短进程退出尾段及单次累计计量|新外层入口未执行|
|阶段同步|supervise.main / StageServer / BudgetLedger.select|真实supervisor、worker、socket、三个短CPU子进程|未执行|
|生产采集|joint_pipeline.run_pipeline / ProductionDataCollector.collect|只替换底层环境；实际枚举、分支、严格标签及持久化|待独立工程授权|
|数据门|joint_pipeline.data_admission|无机会停止，模型调用为0；正常数据通过|待独立工程授权|
|G1/G2训练与选择|production_world.train_select_world_models|完整模型、两方案三种子、1 epoch、实际优化|待CUDA验收；正式预算时限尚未证实|
|保存及恢复|_save_checkpoint / _load_selected_models|生产文件读写与哈希核验|待独立工程授权|
|候选预测与指标|evaluate_world_models / recompute_prediction_metrics|真实前向、三种子与集成trace、保存后复算一致|待独立工程授权|
|账本与导出|runner._settlement / controlled_export|SQLite闭合、生产导出回读|待独立工程授权|

本地主流水线为2/2/8父场景、CPU、1 epoch；另一个序列回归为1/1/8，
直接调用production_world，不宣称它经过collector。均使用合成开发输入。
旧正式数据、封存确认集和真实模拟器本轮未使用。

生产worker的CLI仍无任何测试替身参数。直接Python测试调用main的
显式boundary只允许已独立冻结的integration_test合同；默认正式入口
继续拒绝测试合同。数据/训练/评价/结算阶段函数均实际调用，未被替换。

服务器验收将复用这些实际生产路径，只在环境构造边界使用替身。
验收不会安装依赖、修改共享运行时、运行HARL控制命令、创建正式研究attempt。
若身份、运行时、登记、资源或计量条件不能核验，则技术停止、不重试。
本地进程累计结算通过不等于无cgroup的运行中CPU硬限制成立。
