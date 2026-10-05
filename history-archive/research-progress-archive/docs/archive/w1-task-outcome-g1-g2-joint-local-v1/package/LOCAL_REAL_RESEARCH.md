# 本机真实 W1 联合运行

本包由已成功执行的 w1-gpu-smoke-local-v1 的生产源码副本迁移。旧 smoke 制品与旧研究证据不改写；本次正式数据只来自真实 M10Environment，runner.main(boundary=None)。本包保留的 LOCAL_SMOKE、SERVER_ACCEPTANCE 及旧 README/unique-launch 文档是历史实现说明，不构成本次执行入口或预算。当前唯一入口由 launch-contract 和 execution-manifest 绑定 run_local_research.py。

用户已授权本机一次真实采集及条件六路学习；RESOURCE_REQUEST.json 保持 NOT_APPROVED，实际批准由包外标准 JSON 及新随机一次性 token 绑定 attempt、manifest、hashes 与资源申请摘要。秘密只通过 stdin 传递，不写日志。旧授权和 token 不复用。

实际解释器 /home/asus/.venvs/w1-light-repaired-fair-rerun-py31116/bin/python，Python3.11.16、torch2.7.0+cu128、CUDA12.8，Ubuntu-24.04，物理 GPU0、程序 cuda:0。已成功 smoke 的运行时身份原样复用并检查。使用 -I -B、no user-site、清理动态库和隐式 Python 路径、四个计算线程；不安装依赖。

仅修改执行平台、路径和 attempt 身份及必要的本机计量接线。共享桌面 RTX3060 Laptop 6GiB；启动要求至少4.5GiB空闲，PyTorch allocated/reserved 上限4GiB（比原8GiB预算更严格）。没有声称GPU独占、进程完整GPU内存或完整Windows桥接CPU可测。GPU约束延迟到数据门通过后的学习导入，不因数据失败初始化 CUDA。

训练 train-0064..0087，模型选择 train-0088..0095，预测确认 train-0096..0103，父场景身份与 tape 摘要不变；历史使用未被证明排除，不能声称独立全新盲测。固定 arrival_to_region / physical_arrival 与原到达半径0；每个合法首动作＋hungarian-v1-fixed 完整续行收益，unknown null/mask=false，NOOP没有指定任务标签。没有未来内部信息进入决策前输入。

24/8/8 数据门、训练正例和主机确认支持要求、G1相对透明与G2相对G1的预测门均原样保留。门失败不改门、不补场景、不追加种子。数据门通过才导入学习模型，G1/G2各8201/8202/8203，最多20epochs、600训练候选、每路480更新，冻结规则选模型并保存/恢复6个checkpoint。保存完整逐候选逐种子及集成trace，持久化trace独立复算指标。GPPO与任务对照始终为零，不存在本次自动进入GPPO的分支。

全部原全局与阶段调用、wall、CPU、RSS、存储预算不增额：14160环境步、40reset、1000分支、3138wall秒、4595CPU秒、4GiBRSS、2GiB活动/4GiB聚合存储。计量跨阶段通过生产 StageServer/BudgetLedger 认证同步，已消耗暂存费用作为偏移计入 supervisor。CPU结算 native controller SELF＋kernel-waited CHILDREN，nested supervisor/worker只是分项，不重复相加。Windows控制器另报，WSL桥接CPU和最后写入/退出尾段保持缺口，不虚报完整资源验收。阶段/总预算超限或技术失败停止封存，不重跑。

包外保留原样只读预检、全文件身份前后对比、独立复核、迁移差异、运行日志、导出哈希和资源结算。Windows frozen package 不含运行输出；WSL native副本输出位于run-once，清单覆盖的冻结输入前后核验。受控导出仅复制本次输出、日志和结算，逐文件hash与native比对；不把payload复制等同于验收通过。
