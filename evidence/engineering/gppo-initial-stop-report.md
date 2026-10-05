# G1＋GPPO 正式四臂验证：技术停止封存

本次唯一正式入口确已启动一次，退出码 **1**。停止发生在零步阶段转入
`conditional_policy_training` 的 CPU 握手，尚未进入任何策略训练或任务评价。
attempt `w1-g1-gppo-task-validation-local-v1-once` 已消费并封存，不自动修复续跑。
本次不增加世界模型或 GPPO 有效／无效的研究结论；旧 G1 排序结果及 G2 负结果不变。

## 实际启动与批准身份

```powershell
Get-Content -LiteralPath 'E:\Z博士\research-plans\w1-g1-gppo-task-validation-local-v1\external-token.txt' -Encoding utf8 | & 'C:\Python314\python.exe' -B 'E:\Z博士\research-plans\w1-g1-gppo-task-validation-local-v1\package\run_g1_task_validation.py' --authorization-file 'E:\Z博士\research-plans\w1-g1-gppo-task-validation-local-v1\external-authorization.json'
```

这是已执行、已消费的历史命令，不再是可用启动命令。
启动前用标准 JSON 序列化器写入新的包外授权，回读校验字段、批准摘要和 token 绑定。
批准后 Windows 原样结构预检 exit 0；正式暂存后的 native `--preflight-only --check-token`
实际验证批准 token，exit 0。预检不消费授权，消费发生在 native 正式入口。
完整命令、输出和范围见 approved-launch-preparation-evidence.json、
approved-windows-preflight.*、native-preflight-evidence.json、native-preflight.stdout.txt。

- Manifest：`219cd82a046ebffd3feb3f1e2f9bc5b158f7ab040273d6baef5b866b01ff2895`
- Hashes：`aea59645b889fda413fa9ef110453622d402019016132b0daf59d5779d6d465f`
- Request：`85939a3d6211dc86c5db2a56a368b657c42975468b48dadc141e99b7a7e57e8a`

包内 RESOURCE_REQUEST 保持 NOT_APPROVED；这不是缺少用户批准，正式外部授权已经绑定。
既有合成驱动器的错误断言不在冻结包内，正式入口没有调用它。

## 第一处错误和影响范围

原 console.log 完整 traceback：

```text
runner.py:210 -> task_pipeline.py:298 -> budget_ledger.py:37,70
-> phase_handshake.py:34
RuntimeError: PHASE_ACCOUNTING_REJECTED:PHASE_CPU_BASELINE_DISCONTINUITY
```

supervisor 原始 traceback 在 `phase_handshake.py:141` 拒绝请求。
源码与记录共同表明：

1. `BudgetLedger` 初始本地 CPU 基线为 0；首次 `select(staging_and_zero_step_gate)`
   在 stage 尚为空时写入一次同阶段快照，随后将本地基线更新为 **2.866153s**。
2. 该快照由于 `next_stage == stage` 只记录 `final_same_stage_snapshot`，没有发送 supervisor 握手。
3. `StageServer.last_worker_cpu_end` 仍为 **0**。切换到训练阶段时，worker 发送从
   2.866153s 起算的下一边界，触发基线不连续检查，差异不是超预算或容差噪声。
4. supervisor 按失败路径停止 worker（returncode -15）。worker 开始异常结算的 activity
   已写入，但最终 `resource-settlement.json` 未生成，status.json 留在 `initializing`。
   权威停止状态来自 supervisor-status.json；不能把旧初始化状态解释为仍在运行。

被拒绝请求的原始 socket payload 没有单独留存。2.866153s 来自已持久化的首快照，
基线错位是结合该快照和冻结源码的赋值／调用路径定位，不冒充捕获的拒绝请求原文。

这是首次阶段基线同步的生产接线缺陷。此前无 supervisor 握手的合成生产测试不能
证明这一正式首次切换可用，**之前的 runner_ready=true 工程判定在本次结果后撤回**。
冻结文件保留原字节，勘误只写在包外；本轮未修复源码或重跑。

## 请求的研究答案

| 问题 | 本次结果 |
|---|---|
| G1 相对 G0、透明先验、Hungarian 的 episode 收益 | 未评价 |
| 跨八父场景及三个策略种子的稳定性 | 未评价 |
| 完成、确认、能耗等收益来源 | 未评价 |
| G1 完整 CPU 均值 ≤10ms、wall p95 ≤50ms | 未评价 |
| 腾讯会议的 GPPO 调度价值验证 | 本次技术停止未产生证据 |

九路正式策略均未开始，无新增策略 checkpoint、候选预测或任务成本记录。
本次不是 task_gain_gate_stop 或 cost_gate_stop，也不能记为机制失败。
前期合成成本超标记录继续保留，不能充当本次正式成本测量。

## 实际调用、结算与哈希核验

SQLite 用 `mode=ro` 与 query_only 读取，integrity_check=ok；calls 表 **0 行**，
pending=0、failed=0。环境构造、环境步、模型初始化/加载、world/actor前向、
优化更新、checkpoint加载/写入、任务 episode 均未执行。Python/torch 模块导入与
运行时探针确已发生，不把系统准备耗时报告为零。

| 可测资源 | 本次记录 |
|---|---:|
| Windows 入口记录的 launch wall | 19.4057734s |
| native controller wall | 18.9272547s（嵌套，不与上行相加） |
| native controller SELF＋waited CHILDREN CPU | 10.452967s |
| Windows controller CPU | 0.015625s |
| 不重复相加的已知 CPU 合计 | 10.468592s |
| supervisor 嵌套 CPU | 9.692891s（已含在 native 范围，不能再次相加） |
| supervisor RSS 峰值上界 | 1,047,183,360 bytes |
| native 活动存储 | 13,092,649 bytes |
| 验证导出存储 | 23,900 bytes |
| GPU 使用 | 0 |

控制器可测总量及其暂存／导出分项未超冻结上限。但阶段握手失败、worker
最终结算缺失，完整资源验收 **不通过**。WSL桥CPU、控制器最后写入／退出尾段
未完整计量，CPU上限由采样约束，不声称完整CPU计量或硬 enforcement 通过。
Windows 入口的该计时从身份／授权校验之后开始，未覆盖 PowerShell 启动和此前
入口核验开销；10.468592s 是明确可测范围合计，不是完整生命周期CPU。
失败消耗已记账；剩余额度不移用于新运行。

受控导出共有12个受哈希保护的制品，加 export-hashes.json 本身。全部记录摘要
与导出和 native 源文件一致，未混入研究数据。导出摘要文件SHA-256：
`da363b3839de2008d7f01be9dbe1fa8af628314d8a5367f4affd932f3b7cd5dc`。
209个冻结输入在 Windows 原包、完整冻结归档及 native 暂存目录均匹配批准身份；
只读审计前后导出字节不变，无同一attempt残留进程。

证据：formal-stop-audit.json、launch-evidence.json、native-controller-settlement.json、
verified-export/export-hashes.json、verified-export/console.log、verified-export/supervisor-status.json。
独立 Luna 复核结论和当前 Git 归档状态另存包外文件。Luna 最后一轮曾遇到服务端
502；已保留中断记录，后续只读复核补充了基线错位的源码依据，并明确拒绝请求
原始 payload 未捕获。该复核不包含新的环境或模型调用，也不代表工程修复已通过。

## 后续边界

只支持下一步修复首次同阶段快照与 supervisor 基线的同步，并在不构造环境／模型的
真实 supervisor→worker→ledger 阶段切换路径上回归，同时验证异常结算能够落盘。
本轮不实施修复、不创建新attempt、不恢复已消费授权、不以剩余额度继续实验。
已消费的包外明文token已在本机清理，清理记录不含秘密；消费标记与授权绑定保留。
G1 是否帮助 GPPO 仍待新的工程修复与单独批准的一次正式验证。
