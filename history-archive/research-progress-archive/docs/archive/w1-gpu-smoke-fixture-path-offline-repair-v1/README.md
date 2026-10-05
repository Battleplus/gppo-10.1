# GPU_SMOKE_TEST — 离线夹具路径修复

该目录是开发副本，不是可启动冻结包。没有生成新 attempt、授权或 token；
保留了来源包的旧身份文件，修改后故意未重冻结，正式身份校验会拒绝它。
RESOURCE_REQUEST.json 仍为 NOT_APPROVED，禁止使用它启动。

本次实际失败证据保持在 w1-gpu-smoke-test-bootstrap-repair-v1。
当前训练目标未达成，原工程身份已消费，不能自动重跑。

修复两处测试代码的包外路径：
- test_action_conditioned_task_outcomes.py 的生命周期来源改为包内 native/gppo_world。
- test_runtime_config_contract.py 的 AST 配置读取改为包内 native/gppo_world。

生命周期、模型、采集、训练代码字节均未改动。
离线回归直接执行真实夹具的路径及 runpy 加载语句，验证无历史 runs 目录时
仍可加载、缺失文件时拒绝且不回退工作区、生产源码字节不变、配置路径正确。
未导入整套训练 fixture，未运行生产流水线，不能据此宣称 runner_ready。

总 wall 的原始失败仍为 219.223 秒 > 180 秒，不改写该事实。
开发副本新增 transport_wall_budget.py：连接、预检、上传、bootstrap、远端
计算和下载共用原 180 秒截止时间，保留关闭预留。计时器仅关闭本作业的
SSH 连接，远端入口接收剩余 wall 上限并使用自己的 alarm 和资源计量。
输入文件不变更预算；缩减后的限额只存在运行时副本。
本地密码输入、文件哈希和 OS 退出仍是协作式检查，未证明内核级硬截止。
远端传输 CPU 仍未测量，不声称完整资源验收通过。

上传失败后的身份状态核查失败时记录 unknown，不覆盖首次异常，不允许
重试。修复 main 对 gpu_smoke_test_pass 返回错误退出码的问题。
34 项离线回归通过：8 项 deadline、21 项原授权/计量测试、5 项生产控制器
编排测试。编排实际执行 run_formal、生产上传/回读校验/下载及阶段路由；
仅 SSH/SFTP 边界为内存替身，返回的远端结果是明确的运输测试夹具，
不是模型效果，不证明真实 collector 或六路 GPU 训练完成。
原始首次回归失败也保留；原因是内存 stdin 替身未实现 Paramiko 的 str
写入语义，修复替身后通过。新增回归没有环境、模型、网络或 checkpoint 调用。
测试证据见 wall-regression-passed.log、wall-regression-initial-failure.log。
4 项路径回归再次通过，见 offline-test-evidence.json。

下一次动态测试需要新明确授权及新身份；本目录没有提出或启动替代 attempt。
