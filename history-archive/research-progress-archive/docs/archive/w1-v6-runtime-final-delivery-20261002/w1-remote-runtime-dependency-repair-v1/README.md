# 远端运行时修复与合成验收

本包只记录运行时准备，不是正式标签或训练 attempt。真实研究环境、模型初始化、checkpoint、研究数据读取和正式训练均为零。合成操作不称为零：CPU、GPU 0、GPU 1 各一次 4×4 张量乘法；GPU 1 一次反向、一次 SGD 初始化和一次更新。

## 已修复范围

- 主机：`user1@172.17.27.173`，Ubuntu 22.04.2，驱动 `535.161.08`，两张 RTX 2080 Ti。
- 独立解释器：`/home/user1/.venvs/w1-runtime-v1/bin/python`，Python `3.10.12`。
- PyTorch `2.5.1+cu121`，`torch.version.cuda=12.1`，NumPy `1.26.4`。
- CUDA 计算库来自该 venv 的 pip wheel；系统只提供 `libcuda` 驱动。未安装/切换系统 CUDA，未改 Conda 或默认 Python。
- 初始安装来源：`https://download.pytorch.org/whl/cu121`（PyTorch 官方索引及其 NVIDIA/PyPI 依赖链接）。显式固定 `nvidia-nvjitlink-cu12==12.1.105` 时使用 `https://pypi.org/simple`。完整实际版本见 `requirements.lock.txt`；其 SHA-256 为 `d645d8db4f408bdd851bcb81254bc17e37347f8266bf6ac06df789c3adfcf50d`。
- `runtime-identity-v2.json` 固定解释器文件、25 个发行包的版本/RECORD 摘要、44 个关键原生文件、GPU UUID/驱动和字节码缓存读取隔离。该文件 SHA-256 为 `af32685059d7c3107ab6d0d2f34bc0e9e71c118ec8179d394a216798d7b62cb1`。早期 `runtime-identity.json`（SHA `c47d940c236fd49d1d419cd16ae918785d2da2f4a0fb97d52dbf67d21212ca50`）保留，不能代替最新合同。

## 根因和边界

默认 `/usr/bin/python3` 的用户级 torch 是 `2.9.0+cu126`。其 cuSPARSE 需要 `__nvJitLinkGetErrorLog_12_6`，实际却加载系统 CUDA 12.1 的 nvJitLink，导致导入失败。默认环境没有被修复或作为候选执行环境使用。`anaconda3/envs/pytorch` 的 Python 3.9.21 没有 torch。

主机已有 PID 831752 及相关训练进程。只读审计时负载约 `42.69/49.84/48.92`，GPU 0 有约 614 MiB 计算进程。资源快照不等于独占分配；未终止、修改或复用这些进程。合成探针只能证明微小算子的兼容性，不能证明目标训练矩阵显存充足、生产训练闭包完整或计时成本达标。

## 实际验收

合成探针退出码 0；三个结果 checksum 均为 3680，反向与更新有限。探针 wall `2.1142387751024216s`，自身 CPU `1.745344719s`，详见 `synthetic-probe.json`。这些 CPU/wall 是探针自身范围，不是安装与 SSH 全过程结算。安装耗时/CPU 未作精确全过程测量，不补造数字。

正式研究调用为零；合成前向/反向/更新单独计数。未创建正式 attempt、锁、SQLite 或 checkpoint。远端证据目录 `/home/user1/w1-runtime-repair-v1-evidence` 是诊断目录，不是实验输出目录。

实际远端只读预检命令：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /home/user1/.venvs/w1-runtime-v1/bin/python -I -B -X pycache_prefix=/home/user1/.venvs/w1-runtime-v1/.w1-no-bytecode-cache /home/user1/w1-remote-runtime-dependency-repair-v1/runtime_preflight.py --identity /home/user1/w1-remote-runtime-dependency-repair-v1/runtime-identity-v2.json --identity-sha256 af32685059d7c3107ab6d0d2f34bc0e9e71c118ec8179d394a216798d7b62cb1
```

上面的最终目录命令在全部文件冻结、同步后执行，证据另存包外；不通过改写包内记录来追认。此前诊断目录 v2 预检退出码 `0`，输出包含 `staging_started=false`、`worker_started=false`、`tensor_calls=0`、`optimizer_updates=0`，见 `runtime-strict-preflight-v2-correct-binding-console.log`。预检核对外层摘要及同一份解析字节、强制 `-I -B`、不存在的固定 pycache_prefix 和搜索路径，逐文件核对发行包 RECORD 中带摘要的源码/原生内容，并核对原生库、GPU UUID 和驱动。它不导入 torch，不分配张量，不创建运行目录。

更早的严格预检曾发现 `numpy/distutils/__pycache__/conv_template.cpython-310.pyc` 尺寸 8295，与一条历史 RECORD 的 8269 不符；另有一条同路径、无摘要的生成记录。对应 `.py` 的 SHA 与 RECORD 完全一致，缓存的 compiled_filename 指向当前安装位置，但生成步骤来源未确定。原始失败及诊断在 `runtime-strict-preflight-console.log` 和 `numpy-bytecode-diagnosis.jsonl` 中保留。新加载规则禁止读取这些缓存，而非将其摘要改成通过。

一次 v2 预检还因手工抄录的绑定摘要多一个字符被拒绝（`RUNTIME_IDENTITY_OUTER_DIGEST_MISMATCH`），见 `runtime-strict-preflight-v2-console.log`；核对真实 64 位摘要后只读预检通过，未消费运行授权。正式研究仍需单独绑定生产源码和阶段合同；没有由这些预检衍生训练批准。

本地预检回归 `python -B -m unittest -v test_runtime_preflight`：8 项通过。只替换 GPU 查询和依赖枚举边界，覆盖正确身份、关键库/依赖源码篡改、错误解释器、脏 CUDA 搜索路径、额外发行包、缺失隔离参数、异常项目路径、已有缓存前缀拒绝和可直接导入的无摘要字节码拒绝；不声称这些本地测试完成远端 GPU 训练。

只有 `__pycache__` 中确定不可达的生成缓存豁免 RECORD 字节校验；其他 `.pyc` 必须校验摘要或直接拒绝。完整隔离由上述唯一 CLI 预检入口强制；单独导入内部 `verify()` 并非完整入口，不用于生产启动。

合成探针报告的 sys.path 还包含一个 `/tmp/tmp...` 目录。它不包含研究工作区，但该临时目录的具体来源未在当时记录，来源未确定；不能把此证据夸大为“全过程搜索路径只有 venv”。最终只读预检不导入 torch，并强制启动时搜索路径仅含已声明标准库和 venv site-packages。

## 后续条件

v6 标签包继续使用其冻结 Windows→Ubuntu-24.04 CPU 运行时，不引用这个 GPU venv执行标签。它的 `RESOURCE_REQUEST.json` 保持 `NOT_APPROVED`、模型预算 0。远端身份引用只是未来训练主机的可追溯审计。

先单独批准标签资格采集；标签覆盖与归属通过后，再创建独立 G1/G2/GPPO 训练包，固定数据划分、种子、研究目标、门槛、GPU/CPU/存储预算，运行生产底层替身端到端测试，并提交新预算。当前运行时不授权训练，不消费旧 token，也不替代标签验证。

GitHub 网络读取和 fetch 已成功。Windows ssh-agent 为 Disabled/Stopped，`ssh-add -l` 返回代理不可用。签名要求未关闭、未创建未签名提交、未强推。当前为本地归档，远端未归档。
