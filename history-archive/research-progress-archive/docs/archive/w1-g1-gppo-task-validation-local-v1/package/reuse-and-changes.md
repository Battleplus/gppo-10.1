# 可追溯复用与修改

| 来源 | 复用内容 | 本副本实际改动 |
|---|---|---|
| w1-task-outcome-g1-g2-joint-local-v1/package | 已成功Windows→WSL原生入口、依赖身份、账本、监督、导出、W1GraphJEPA、public_history、透明完整续行公式及真实模拟器 | 新attempt/CPU条件/任务矩阵/只读预检路径；原包不改 |
| 同次真实run的verified-export | 三个按模型选择规则冻结的G1 checkpoint | 仅复制原bytes，sha逐个绑定；不新增训练或挑种子 |
| w1-eawm-jepa-world-model-production-repair-v4-e2e-integration-auth-repair-v1 | production_policy/runtime_hooks里的因果GPPO route与真正Hungarian调用 | G0/T/G1九路、外部G1绑定加载、全时域透明prior、明确实际环境配置、成本与trace、独立结果复算 |
| native/gppo_world/joint_training.py、joint_gppo.py | 双目标GPPO clipped surrogate、偏好相似度、PreCo、向量critic和因果回放 | 原核心算法保留，先验wrapper保留在采样与更新；不调用joint_training.run_group联合训练旧world模型 |

新runner/task_pipeline只执行冻结资格核验→策略训练→任务评价→独立复算→门→结算。不会再走joint_pipeline采集/G1/G2世界模型训练。run_local_research旧自动批准入口在本副本禁用；主入口只能接受包外真实授权。

继承的local_launcher、launch_joint_once、launch_once、wsl_stage_and_launch、native_launch、acceptance_entry、launch_server_acceptance、smoke_supervisor_entry、metered_joint_entry的main以及自动执行的joint_supervisor_entry均在本副本立即拒绝运行；辅助函数保留溯源。避免旧smoke/远端命令与本次唯一Windows入口混淆。原历史包仍原样保留。

independent_task_recompute用Python标准库从原vector rewards核对效用，按完整矩阵核验cost与场景，拒绝非有限数、缺失成本、重复身份和错误效用。其任务门与cost门分别返回状态。所有其他方法成本单列，不能稀释G1成本。

研究方案新增的是任务验证协议和未批准GPPO预算，不是对旧预测评价的事后改判。八个任务场景与本轮world开发分析按真实scenario与structural摘要隔离，但整个项目的历史排除未证明。透明NOOP审计原样另存，旧指标不修改。

准备测试：替身限底层环境交互和合成checkpoint；真正生产策略编排、原多目标损失、优化器、checkpoint写读、指标复算执行。合成初始化、前向、反向、更新与checkpoint操作逐项另列包外，不声称动态计算为零。清单结构预检仅证明身份、运行时及未消费状态，不等于真实任务效果或完整资源验收。
