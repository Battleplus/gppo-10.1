# W1 联合生产验收 v2 副本静态复核

复核对象：`E:\Z博士\research-plans\w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote`，即由 v2 复制出的开发 v3 目录；它不是新的冻结执行包。复核范围限于源码、合同、预算和用户给定的执行计划。本复核未连接服务器、未调用真实 runner、未产生 attempt，也未改包内文件。

## 结论

生产 runner 的静态路径与计划相符：采集全部固定窗口，先执行数据门；失败时在导入训练后端前停止；通过后才训练 G1/G2 六路、恢复 checkpoint 并对确认集预测。本轮没有任何真实数据、训练或预测结果。

当前工程入口即使远端生产链成功，也无法返回 `complete`：控制器将 `remote_transport_cpu_measured` 固定为 `False`，而完成条件明确要求它为真。修复并取得可核验计量证据之前，工程验收不能成为研究放行证明。正式研究的三个入口仍会因冻结 `runner_ready=false` 而拒绝启动。

## 关键发现

### 1. 工程验收完成状态不可达

在 [launch_server_acceptance.py](E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote/launch_server_acceptance.py:258)，`remote_transport_cpu_measured = False` 是常量；[同文件](E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote/launch_server_acceptance.py:165) 的 `_completion_gate` 要求该值为真。因此，不论远端退出码、验收结果、证据下载和预算算术是否通过，最终都只能是 `technical_stop`，`runner_ready` 也固定为 false。

最小修复不是把该值改成 true，而是为远端预检与 SSH/SFTP 运输补上实际、可核验且范围完整的 CPU 计量，并验证不超过已申请的 30 CPU 秒；若无法测量或约束该范围，就保留技术停止。现有协议也明确承认这 30 秒只是收费额度，不是实测值或已证实的硬上界。

### 2. 冻结的 false 标志没有外部证明放行路径

研究启动链有三处直接拒绝：Windows 入口 [launch_joint_once.py](E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote/launch_joint_once.py:198)、远端寿命计量入口 [metered_joint_entry.py](E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote/metered_joint_entry.py:128)，以及 [joint_remote_native.py](E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote/joint_remote_native.py:217)。它们只读 `runner_ready`，没有读取或校验工程验收证明。外部授权字段又采用严格白名单，[manifest_contract.py](E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote/manifest_contract.py:17) 会拒绝未登记的证明字段。

所需的最小放行合同应保持冻结的 `RESOURCE_REQUEST.json`、`launch-contract.json` 与身份摘要不变，也不把 `runner_ready` 改写为 true；另由正式外部授权绑定工程证明清单摘要，并要求本地和远端入口验证同一证明。证明至少要交叉绑定研究执行 manifest/hash、资源请求摘要、工程 job/request 身份、验收 evidence manifest 摘要、终态登记、资源结算和重试状态。三层启动门统一通过同一证明验证函数后才可继续。工程证明中的“合成链通过”不能替代真实研究数据门。

此项描述的是 v2 副本的旧门控状态；新开发目录中的证明实现由父线程另行处理，本复核不以尚未纳入该副本的文件作为新缺陷。

### 3. CPU 完整范围与 GPU 独占需要管理员侧证据

研究 supervisor 在 [supervise.py](E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote/supervise.py:224) 与最终结算 [同文件](E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote/supervise.py:257) 明确记录 `cpu_scope_complete=false`、`cpu_scope_hard_enforcement=false`：当前依赖采样与最终结算，未获 delegated cgroup 时不能在跨过 CPU 上限前由内核阻止作业。工程协议也把 RSS 限于根进程采样，并把远端运输范围列为未核验。若授权上限要求完整进程树 CPU 不超过 360 秒，运行前需要管理员提供可用的 delegated 计量/限制范围；缺少时不能把最终结算或 30 秒预留描述为硬封顶。

工程申请指定物理 GPU 1、至少 9 GiB 空闲显存、至多 8 GiB 分配和独占分配。[acceptance_entry.py](E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote/acceptance_entry.py:781) 会做启动前检查，[supervise.py](E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote/supervise.py:77) 会检查已观察到的 GPU 使用者；这些采样不能单独证明作业期间有固定 GPU 独占分配。工程放行证明需要管理员/调度器提供的 GPU 1 独占分配依据，并保留代码中的进程和显存检查；否则应在合成工作前停止。

### 4. 登记事件顺序正确，但没有“回执后再启动”的控制器握手

工程入口的顺序是 `register()`、记录 CPU `RUNNING`、初始化 CUDA、执行工作负载，见 [acceptance_entry.py](E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote/acceptance_entry.py:689) 和 [同文件](E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote/acceptance_entry.py:818)。研究远端入口也在创建 worker 前登记并追加根进程 `RUNNING`，见 [joint_remote_native.py](E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote/joint_remote_native.py:253) 与 [同文件](E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote/joint_remote_native.py:266)。因此，登记记录先于 GPU/worker 工作。

若计划中的“先报告登记成功，再启动”要求先把成功回执交给用户，现有单次 SSH 命令不满足：控制器在远端命令结束后才读取 stdout，而 `run_registered_once` 在登记后立即继续。最小调整是拆成两步：第一步仅追加并回读 `REGISTERED`，返回 job_id/回执；控制器报告回执后，第二步用同一 job_id、执行身份和一次性授权启动 `RUNNING` 与工作负载。不要让第二步创建另一个工程 job。

## 数据门、训练、预测与预算

[runner.py](E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote/runner.py:198) 调用 [joint_pipeline.py](E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote/joint_pipeline.py:57)。该流水线使用 `ProductionDataCollector` 收集冻结窗口，[data_admission](E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote/joint_pipeline.py:13) 核验注册窗口、完整返回、每窗口有效目标及训练集跨父场景标签支持；门失败时在训练后端导入前返回 `data_gate_stop`。门通过后，[同一流水线](E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote/joint_pipeline.py:69) 才进入模型选择与确认评价。生产训练器按 G1/G2 × 8201/8202/8203 构造六路，最多每路 20 epoch、480 更新，使用 24 个训练窗口及 8 个模型选择窗口；确认预测只使用另外 8 个窗口，并复算已落盘 trace 指标。G1/G2 不含 GPPO，计划的 `gppo_updates=0`、`task_comparison_calls=0` 与请求一致。

预算加总闭合：

- 研究 wall：阶段 3108 秒 + 跨系统预留 30 秒 = 3138 秒；完整 CPU：阶段 4565 秒 + Windows 预留 30 秒 = 4595 秒。
- 研究学习调用：6720+48=6768 次 batch forward；168000+1200=169200 次样本评价；2880 次 backward 与 2880 次 optimizer update；checkpoint 写入/加载各 6 次。
- 工程 wall：服务器 150 秒 + 控制器 30 秒 = 180 秒；CPU：服务器 300 秒 + 远端预检/运输额度 30 秒 + 控制器 30 秒 = 360 秒。工程结算公式的额度算术闭合，但远端 30 秒仍只是 allowance。

加总闭合不等于研究训练时限已由服务器证明。工程请求只覆盖合成 2 train / 2 selection / 8 confirmation、最多 1 epoch、18 次 backward/update 的有限链路；正式研究上限为 24 train / 8 selection、最多 20 epoch、2880 次更新。验收证据若用于放行，还需要用其真实计量结果形成保守的正式规模耗时/CPU 估算，并证明不超过训练与预测各自阶段上限；估算超限或依据不足时，不启动正式研究，也不借用其他阶段额度。

旧记录 `0.274925s` 的差值在 [cpu-accounting-erratum-v6.md](E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote/cpu-accounting-erratum-v6.md:26) 中被正确标为无法追溯归因。不能把它分摊给启动、worker、结算、导出或退出；只有新运行按新边界采集的证据能解释新运行差值。

## 复核边界与当前状态

身份与联网状态按父线程报告记录：源 v2 manifest 的 156 个内容文件和 manifest/hash 摘要已重新核验；目标 v3 是开发副本，不应称为冻结。服务器 TCP 连接在认证前超时，没有远端命令、登记或 attempt。控制器 runtime 导入接口已在当前副本中改为 `from controller_runtime import verify_controller_runtime`；父线程报告的 22 项本地相关测试通过及入口路由替身测试只证明本地接口路径，不证明服务器验收。真实研究调用数仍为 0。

本复核未运行测试，未执行合成调用、SSH、服务器操作或 runner。可进入下一阶段的外部前置条件仍包括：管理员提供完整 CPU 计量/限制范围、固定 GPU 1 独占分配，以及工程入口实测远端运输 CPU 并完成预算内的合成验收；随后必须以绑定至确切执行身份的外部证明解锁冻结 runner，而不能修改旧身份或伪造 `runner_ready`。
