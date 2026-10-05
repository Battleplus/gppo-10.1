# 唯一正式启动命令

当前尚未批准，不执行下面的正式命令。新 attempt：w1-g1-gppo-recovery-v1-once。

先由新的明确用户批准绑定最终 Manifest、Hashes、RESOURCE_REQUEST，再用标准JSON序列化器写入包外 external-authorization.json，并生成从未使用的一次性token。授权文件的status为APPROVED；包内RESOURCE_REQUEST始终NOT_APPROVED。token仅保存包外受控单行文件，不放日志、Git或冻结包。

```powershell
Get-Content -LiteralPath 'E:\Z博士\research-plans\w1-g1-gppo-recovery-preparation-v1\external-token.txt' -Encoding utf8 | & 'C:\Python314\python.exe' -B 'E:\Z博士\research-plans\w1-g1-gppo-recovery-preparation-v1\package\run_g1_task_validation.py' --authorization-file 'E:\Z博士\research-plans\w1-g1-gppo-recovery-preparation-v1\external-authorization.json'
```

只读结构预检（不消费token、不暂存、不启动worker；NOT_APPROVED模板不能证明已正式批准）：

```powershell
& 'C:\Python314\python.exe' -B 'E:\Z博士\research-plans\w1-g1-gppo-recovery-preparation-v1\package\run_g1_task_validation.py' --preflight-only --authorization-file 'E:\Z博士\research-plans\w1-g1-gppo-recovery-preparation-v1\external-authorization.json'
```

授权和token的正式校验会在Windows、native controller、native entry及worker重复执行。旧自动授权入口已禁用。只允许新包/新身份，禁止复用旧已消费授权。Windows launch-intent、native attempt目录及消耗标记均拒绝重启。技术停止不会自动修复续跑。

实际任务成本只在CPU条件下测量。唯一入口自动设置CUDA_VISIBLE_DEVICES为空和BLAS线程1；native supervisor固定CPU0与torch/interop线程1。其余参数与预算由冻结文件确定，不允许命令行临时覆盖。
