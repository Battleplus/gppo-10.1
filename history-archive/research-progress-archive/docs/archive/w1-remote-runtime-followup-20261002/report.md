# 远端运行时复核与训练边界

本轮未运行真实环境、研究模型、checkpoint 或正式训练，未创建正式 attempt。执行位置为远端 `user1@172.17.27.173` 的只读诊断入口；不会把本机 WSL 预检当作远端训练验收。

## 运行时事实

已存在的独立环境为 `/home/user1/.venvs/w1-runtime-v1/bin/python`：Python 3.10.12、PyTorch 2.5.1+cu121、CUDA runtime 12.1、NumPy 1.26.4。依赖由该 venv 的 pip wheels 提供；系统仅提供驱动 libcuda。驱动 535.161.08，两张 RTX 2080 Ti（每卡 11264 MiB）。默认 Python 的 cu126 torch 与系统 nvJitLink 混载导致导入失败；Conda pytorch 环境未安装 torch。这些原始失败已保存在原运行时审计包，本轮没有改动默认环境或重新安装。

锁文件 SHA-256：`d645d8db4f408bdd851bcb81254bc17e37347f8266bf6ac06df789c3adfcf50d`。

运行时身份 v2 SHA-256：`af32685059d7c3107ab6d0d2f34bc0e9e71c118ec8179d394a216798d7b62cb1`。

运行时交付清单 SHA-256：`d5ff5bfffa37934dd96b55fb32d407e817bfbb95a9d38ff4d66d7945a07194d6`。

本轮本地 `python -B freeze_delivery.py --verify` 退出 0，26 个内容文件身份通过；`python -B -m unittest -v test_runtime_preflight` 退出 0，8 项合同回归通过，均不运行张量或模型。

历史合成探针为 CPU、GPU 0、GPU 1 各一次 4×4 张量乘法（checksum 3680），GPU 1 一次反向、一次 SGD 初始化和更新，状态 pass。探针 CPU 1.745344719 秒、wall 2.114238775 秒；这些数字只覆盖探针自身，安装/SSH 开销未测全程，不补造。保存于 `w1-remote-runtime-dependency-repair-v1/synthetic-probe.json`，本轮没有重复该探针。历史探针结束时出现 PyTorch/优化器相关临时搜索目录，来源未确定；最终身份预检强制 -I、-B 和不可用的固定缓存前缀，不夸大为全过程路径已证明。

## 本轮实际 SSH 预检

```text
env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /home/user1/.venvs/w1-runtime-v1/bin/python -I -B -X pycache_prefix=/home/user1/.venvs/w1-runtime-v1/.w1-no-bytecode-cache /home/user1/w1-remote-runtime-dependency-repair-v1/runtime_preflight.py --identity /home/user1/w1-remote-runtime-dependency-repair-v1/runtime-identity-v2.json --identity-sha256 af32685059d7c3107ab6d0d2f34bc0e9e71c118ec8179d394a216798d7b62cb1
```

实际退出码 0，status=pass，tensor_calls=0、model_calls=0、optimizer_updates=0，staging_started=false、worker_started=false、formal_attempt_created=false。它验证源码/原生依赖、解释器、搜索路径及设备/驱动身份，不导入 torch 或分配张量；实际 torch/CUDA 算子兼容性来自上述历史合成探针。

首次 BatchMode 连接因没有可用公钥认证返回 Permission denied，随后使用用户已提供的密码认证成功；未把密码或 token 写入文件、报告或 Git。

## 当前资源

| 项目 | 最新只读观测 |
|---|---|
| GPU 0 | 622 MiB 已用、10388 MiB 空闲、利用率 0% |
| GPU 1 | 20 MiB 已用、10988 MiB 空闲、利用率 0% |
| CPU | 32 逻辑 CPU；负载 56.23 / 53.79 / 49.20 |
| 主机可用内存 | 26,169,651,200 bytes |
| 独占分配 | 未证明 |

资源快照不等于获分配资源。未中止其他用户进程，未申请/启动正式训练。两卡微型算子通过不证明 Graph-JEPA/GPPO 矩阵显存、成本或生产训练闭包合格。

## 标签与训练申请状态

v6 一次性标签 attempt 已执行并停止，不能再次申请同名运行。保存标签覆盖通过（8 窗口、108 候选），但外层 CPU 结算技术停止，不追改停止记录或冻结包。标签数据不是独立确认集。

Luna 复核指出 v6 schema 2.0.0、label_qualification split、hungarian-v1-fixed 与旧训练 loader 的 schema 1.0.0、正式分割、one-step-public-transition-v1 不兼容。旧训练目标也没有消费 task_outcome_target 的路径；不能只换名字或 schema 就声称任务后果监督已接通。

新目录 `w1-eawm-jepa-world-model-training-v1-preparation` 仅为阻塞草稿。其 RESOURCE_REQUEST 保持 NOT_APPROVED；预算数值是继承的 v4 参考，不是已重新推导的可批准申请。候选名称不是正式 attempt，launcher 拒绝非 preflight，worker main 在学习后端导入之前拒绝。旧静态/E2E 文档仅为参考，不覆盖新包远端就绪性。

该阻塞草稿最终只读预检退出 0、全部文件前后摘要相等。证据保存于本报告同目录的 `draft-final-preflight-summary.json`、完整 before/after 身份表及 stdout/stderr。其结构身份 manifest=`800aeb6e68d0053dd7d9b2451a6339726713b73242d4ab69f63627cf329d44be`、hashes=`bb90f44051526a42560a704c6918c78890123bb51bd12afbef793dfe5bd5d89c`，仅用于审计阻塞草稿，不可据此申请正式训练。没有正式运行 token。

Luna 的源码/制品复核见 `luna-review.md`；主线程核验了实际 SSH、依赖合同回归和最终草稿身份，不把代理结论单独当作验收。

## 下一项工作

完成显式任务监督/续行适配和父场景训练/选择/确认隔离，修复生产结算范围，接通独立远端 GPU 入口，再运行受控生产链测试及重新推导预算。此后才提交独立训练申请；当前不申请也不启动训练。

## GitHub

仓库 `research-progress-archive` 保留既有 502 个暂存文件。本轮读取远端 main 成功：`6aa16410b0e69aaceb7dee0b4d260eaf76d03e02`，与本地 HEAD 及 origin/main 一致。commit.gpgsign=true、gpg.format=ssh；Windows ssh-agent 仍 Disabled/Stopped。本轮没有新签名提交、没有 push、没有强推，远端未归档。若恢复代理，需在有权限的 PowerShell 中启动服务并加载已有签名密钥；不要将凭据或大型标签文件上传。
