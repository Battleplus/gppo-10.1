# GPU_SMOKE_TEST v3 工程收尾

本轮代码集中修复、Windows46项回归、真实本地WSL9项系统验收已完成。
目标服务器SSH/TCP22从Windows和WSL均超时：远端运行时和生产系统
验收未完成，GPU测试未启动。不是训练失败或世界模型研究失败。

- 新冻结候选：package/；唯一入口见package/unique-launch-command.md。
- 同源源码/模型/数据/种子/门槛不改；预算与profile不扩。
- 新工程身份未消费，未创建研究attempt/外部运行授权/token。
- 远端无模型系统验收成功并绑定最终身份之前，frozen smoke controller拒绝启动。
- 本地测试不能冒充远端Python3.10/GPU验收；runner_ready=false。
- 网络恢复后仍先系统验收，再只执行一次已授权GPU测试，失败不追加。
- blocking-result.json、network-probes.json和ssh-first-error.txt为真实阻塞证据。
- Git保留签名要求；签名代理停用，远端未归档。
