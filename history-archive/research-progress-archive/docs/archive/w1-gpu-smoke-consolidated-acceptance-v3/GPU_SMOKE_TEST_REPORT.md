# GPU_SMOKE_TEST：集中修复后的交付与实际阻塞

训练未启动。首次实际远端异常是 Paramiko socket.connect 的 TimeoutError，SSH 会话尚未建立；最终 TCP/22 复查仍超时。本次不是模型或训练失败，没有研究效果结论。旧包和停止证据保持原样。

| 项目 | 实际结果 |
| --- | --- |
| G1 / G2 完成路数 | 0/3 / 0/3 |
| loss、预测指标、checkpoint 保存/恢复 | 未评价；无新 checkpoint |
| GPU、目标 Python/PyTorch/CUDA | 本次未连接，未核验 |
| 模型初始化/forward/backward/更新/CUDA | 均为 0 |
| Windows 离线回归 | 46 项通过 |
| 本地 WSL 实际系统测试 | 9 项通过，Python 3.12.3；不能替代远端 3.10.12 |
| 最终原样只读预检 | 退出 0，182 个文件前后内容身份一致 |
| 本地 WSL 系统计量 | wall 4.9155539s；SELF+waited CHILDREN CPU 1.513016s |
| 远端完整资源验收 | 未完成；GPU 峰值和远端 CPU 未测量 |
| runner_ready | false |
| 新工程身份 | 未消费；无研究 attempt、外部授权或 token |

集中修复包括标准 JSON/SFTP 排他写入验证、隔离依赖与历史路径检查、短私有 socket、早期及缺失日志、子进程退出与超时、GPU cuda:0 身份、计量边界、部分结果保留和安全哈希导出。模型、标签、种子、研究门槛与预算保持不变。阶段接口详见 package/SYSTEM_WIRING.md；独立 Luna 源码复核详见 independentreview.md。

CPU 历史 0.274925s 差值来源仍未确定；不事后归因、不放宽容差。训练完成与完整计量通过分开记录。用户取消的登记挂载、独占分配证明、SSH/SFTP 全生命周期精确 CPU 计量要求未重新作为阻断项。

最终冻结身份：
- Manifest：23255950aa927aa516150f9e4b2c8b39b0752c1df3dc400aa23cd51af290c47a
- Hashes：d03ea9817204c04f9c38fcb86885d484152e5cf73455552edbf83cebb512a2b2
- 工程申请：749cdaa9bf2ba4093b08021d233b9dbd16938552425647277ffb78f1b64706e3

最终目录实际执行的只读命令：
```powershell
& 'E:\Z博士\.runtime-tools\w1-ssh-controller-v1\Scripts\python.exe' -B 'E:\Z博士\diagnostic-work\w1-gpu-smoke-consolidated-acceptance-v3\package\launch_server_acceptance.py' --preflight-only
```
退出码 0，stdout 为 structure_preflight_pass，stderr 为空；staging_started=false、worker_started=false、formal_authorization_validated=false。完整前后文件身份、命令、输出保存在包外 final-original-preflight.json。验证后未改写包内文件。

网络恢复后仍需先执行最终身份绑定的服务器无模型系统验收，再执行 package/unique-launch-command.md 的唯一 GPU 命令。尚无远端系统通过回执，不能伪造，也不能跳过实际服务器系统接口验收。此次 GPU 单次授权仍未消费，失败后不追加测试。

本地归档为小型源码差异、协议与证据摘要，非完整可执行副本。完整冻结包在本目录 package/。归档复制逐文件核验哈希并保留既有 Git 暂存内容。签名要求保留；ssh-agent Stopped/Disabled，未创建新签名提交，远端未归档。
