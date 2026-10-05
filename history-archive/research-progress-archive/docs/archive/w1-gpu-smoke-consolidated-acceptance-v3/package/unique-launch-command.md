# 单次 GPU_SMOKE_TEST 唯一入口

在同一最终身份的服务器无模型系统验收及只读预检通过后，只执行一次：

```powershell
& 'E:\Z博士\.runtime-tools\w1-ssh-controller-v1\Scripts\python.exe' -B 'E:\Z博士\diagnostic-work\w1-gpu-smoke-consolidated-acceptance-v3\package\smoke_controller.py'
```

工程身份 w1-gpu-smoke-test-consolidated-acceptance-v3-once。
控制器标准序列化新包外授权和token；密码仅getpass/stdin，不写文件。
controller-start.lock创建后不再执行。不复用旧授权，不启动研究attempt。
RESOURCE_REQUEST与SERVER_ACCEPTANCE_REQUEST保持NOT_APPROVED。
原预算180wall/360CPU、服务器150/300、控制器30/30及全部其他上限不变。
本机只是连接/传输；远端使用冻结venv Python和GPU1。
本地只读入口为本目录launch_server_acceptance.py --preflight-only，
不连接、不暂存、不创建锁/SQLite、不消费授权。
