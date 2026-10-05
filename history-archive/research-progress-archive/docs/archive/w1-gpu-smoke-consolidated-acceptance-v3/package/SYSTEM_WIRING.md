# 入口及历史故障审查

| 环节 | 输入/真实函数 | 路径和依赖 | 超时、输出和异常 |
|---|---|---|---|
| 控制器 | run_gpu_smoke.main、冻结清单和包外授权 | 锁定Windows transport Python、getpass/stdin | 排他start.lock、首异常包外 |
| SSH | connect/execute、pin host key | user1@172.17.27.173 | global180s、先保存stdout/stderr |
| SFTP | upload_once、冻结ZIP | 授权wx/chmod600、标准JSON | 回读字节/摘要/身份、排他拒绝 |
| 导入 | _bootstrap_runtime/isolated_fixture_preflight | 锁定venv -I -B、包native | CUDA前完整校验、历史路径拒绝 |
| supervisor | smoke_supervisor_entry→supervise.main | 隔离-supervision目录、私有短socket | CPU/RSS/wall监控、自有进程清理 |
| worker | worker_bootstrap→smoke_worker→acceptance_entry | PDEATHSIG、GPU1、同一Python | 启动前认证、stdin秘密不记录 |
| collector | BottomBoundary→runner.main→ProductionDataCollector | 仅底层环境合成2/2/8 | 原数据合同、生命周期标签持久化 |
| G1/G2 | production_world训练/选择 | 原完整模型3seed、1epoch50rows | 逐调用guard、实际loss和选择 |
| checkpoint/预测 | 保存/恢复/预测/recompute | 六GPU checkpoint、候选trace | 非有限/身份错停止、独立复算 |
| 结算/导出 | runner._settlement/controlled_export | SQLite、status、sealed allowlist | 首异常、部分计算制品保留 |
| 下载 | download_evidence | nativeZIP→Windows独立目录 | 字节/hash/三副本存储、清理 |

历史故障：授权JSON尾随/字段错误、SFTP x未带写标志、-I导入顺序、
历史绝对夹具路径/依赖闭包、119字节Unix socket、缺失console.log掩盖
首异常、子进程退出、控制器75.55s超过30s阶段上限、结果导出。
v2实际CUDA上下文1次但模型0次。历史0.274925sCPU差值来源未确定。
旧证据保留，新修复与范围计量不追溯改判。

系统验收是真实文件/socket/process/export接口。合成环境只在批准的
smoke计算中使用。旧CPU-only WSL专用测试不可冒称适用于远端CUDA环境；
本轮选用明确无模型的系统测试。正式研究runner_ready仍false。
