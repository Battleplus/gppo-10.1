# GPU_SMOKE_TEST v2 — 独立新授权

本次目标是实际完成合成生产 collector → G1/G2 三种子 GPU 训练 → checkpoint
保存/恢复 → 全候选预测 → 指标复算 → 结算和导出。最多 1 epoch、50 条候选。
不运行研究数据或 GPPO，不把合成结果解释为模型研究效果。

唯一命令：

```powershell
& 'E:\Z博士\.runtime-tools\w1-ssh-controller-v1\Scripts\python.exe' -B 'E:\Z博士\diagnostic-work\w1-gpu-smoke-test-fixture-runtime-repair-v2\run_gpu_smoke.py'
```

仅在本机交互终端输入密码。授权文件和 token 在包外，由本次用户明确授权
生成；token 仅保留内存，不写日志。一次性锁防止重复启动。失败不自动重试。
冻结身份见 smoke-package-identity.json，差异见 source-differences.json。
RESOURCE_REQUEST 保持 NOT_APPROVED；预算仍为 wall 180、CPU 360 秒及原全部子上限。
取消的三项前置要求保持取消。共享环境及 HARL 不改动。

最终原样预检和运行证据全部写在 package 外。原包和旧失败证据不改写。
测试数据来源和隔离检查见 package/SMOKE_REPAIR.md。
runtime/transport 的计量缺口如实报告，不宣称完整资源验收或 runner_ready。
