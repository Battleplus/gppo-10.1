# Preflight 与针对性验收状态

## 冻结前本地核验

- `launch_pilot.verify()`：通过。manifest、hashes、RESOURCE_REQUEST、package copy、stage namespace、候选合同、public-prefix 合同及固定 seed 摘要一致。
- 输入缓存：288 个 gzip 文件、25,549,852 compressed bytes；SHA-256 与 v5 verified export 清单一致；174 complete、114 no_opportunity；父场景 split 和标签重算无差异。
- seed 8201：文件、metadata、state_dict、optimizer、strict restore 通过；40 epochs、320 updates、训练父场景一致。
- 未执行模型 forward、世界模型训练、GPPO/critic 更新、环境或任务评价。

## 受控验收边界

`package/acceptance/recovery_path_acceptance.py` 使用真实 `BudgetLedger`、生产 `Driver` 和真实 `StageServer`，覆盖缓存准入、禁止采集、8201 binding 复用、8202/8203 各 40 epochs × 8 batches 的更新调度、三 seed restore handoff 以及不足预算的调用前拒绝。调度夹具只计账本和训练形状，不执行模型计算，工程计算不计作研究 forward。

本机 Windows 运行该脚本未进入账本：Windows Python 没有 Unix `resource` 模块；生产入口固定为服务器 Python 3.11 runtime，故本机不能替代服务器 StageServer/socket 验收。首个调用前错误为 `ModuleNotFoundError: No module named 'resource'`，没有生成正式 attempt、token、服务器运行目录或研究结果。必须在服务器 Linux runtime 上运行该针对性验收或 preflight 后，才能将服务器托管/计量标为通过。

## 研究状态

当前 `RESOURCE_REQUEST.json` 为 `NOT_APPROVED`。本文件不构成服务器 preflight 通过，也不授权正式研究。v5 停止证据和旧 pending 保持不变。
