# GPU_SMOKE_TEST 夹具与隔离运行修复 v2

这是单次合成工程测试，不是研究实验。旧 bootstrap-repair-v1 包及失败证据不改写。
新身份为 w1-gpu-smoke-test-fixture-runtime-repair-v2-once。

复用原生产 collector、世界模型、训练、checkpoint、预测及复算代码和原合成
2/2/8 父场景 profile。G1/G2 各三个固定种子，最多 1 epoch、50 条训练候选。
原 CPU sequence 回归和计量回归保留；它们的合成计算单独计入原调用上限。
真实研究环境、真实研究训练、GPPO 调用均禁止。

修复 test_action_conditioned_task_outcomes.py 与 test_runtime_config_contract.py 的
包外工作目录推导，直接使用冻结包 native/gppo_world 中的源码。没有替换模型
训练标签为研究数据；本次输入始终是原协议明确批准的合成夹具。

bootstrap 在隔离解释器中实际导入将运行的四条测试及其全部导入依赖，核对
项目模块来源，并运行纯配置/源码一致性检查。审计 hook 拒绝 /home/runs、
/mnt/e 和 Windows 历史工作目录。bootstrap 无环境、checkpoint 或模型计算。
运行时保持原冻结版本，不安装或改变共享环境。

连接、预检、上传、bootstrap、训练和下载共享原 180 秒 wall，保留关闭时间。
只向远端传递缩小后的剩余时间，不扩大冻结资源上限。连接计时器只关闭本
作业 SSH 连接，远端 alarm 限制作业；本地 OS 退出和哈希为协作式检查。
短命子进程计量及原 0.05 秒同范围一致性门不放宽。

冻结 allowlist 改为批量 ZIP 传输。归档与各成员摘要双重核验，解压使用 xb，
禁止覆盖、路径逃逸、额外成员。外部授权仍使用 SFTP wx，回读长度、摘要、
JSON、身份和预算绑定验证后才启动。导出批量传输保留逐文件摘要验证。

RESOURCE_REQUEST 和 SERVER_ACCEPTANCE_REQUEST 保持 NOT_APPROVED；用户本次
明确授权由包外 JSON 与一次性 token 绑定新身份。旧 token 和旧身份不迁移。
固定登记挂载、独占分配证明和完整传输 CPU 计量不是本次前置条件。
GPU1 必须空闲且至少 9 GiB，仍检查实际权限，不触碰 HARL。

完整 SSH/SFTP CPU、后处理进程 CPU 和聚合 RSS 未充分测量时明确写未测量；
不将 GPU_SMOKE_TEST 通过等同于完整资源验收或 runner_ready。
首个动态失败封存，不修复后自动续跑，不生成额外测试身份。
