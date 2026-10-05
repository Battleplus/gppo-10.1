# 阶段 CPU 协议修复交付

工程范围通过：此前首次同阶段快照后的训练转换已在真实生产 worker→runner→supervisor→AF_UNIX 路径通过。最终代码经独立 Luna 复核、WSL 受控回归及原样只读预检。工程回放在真实训练函数第一条语句前停止，研究环境、模型、checkpoint 和训练调用均为0。

旧 attempt 已消费、技术停止和资源验收失败结论保留；旧 209 个冻结文件哈希不变，未修改原始失败制品。此修复不形成新的研究结果。

## 根因与最小修复

旧本地同阶段快照将 worker CPU 起点从0推进到2.866153s，却未同步 supervisor，首次训练转换遭严格连续性拒绝。旧异常结算再次握手且 supervisor 立即清理 worker，首错与结算因此缺失。

修复分离阶段起点与累计观察；进入阶段登记CPU0，累计快照不移动起点；真实转换以序列、载荷摘要和双向确认提交。相同重复幂等、冲突拒绝，确认丢失停止。保留1e-6数值容差和 PHASE_CPU_BASELINE_DISCONTINUITY 检查。

worker 在结算前保存首 traceback；异常结算不再次切阶段。所有退出都独立核对开放阶段尾段与精确总量。server已提交与worker已收到确认分开记录；不确定阶段保留null。worker未生成结算时，supervisor明确标注不完整fallback，未知账本不是0。

## 真实生产路径证据

| 用例 | 判定 | 外层 waited CPU(s) | supervisor CPU(s) | 独立尾段(s) |
| --- | --- | ---: | ---: | ---: |
| normal | 通过 | 14.120903 | 13.909788 | 0.347887 |
| duplicate | 通过 | 14.028102 | 13.806230 | 0.453642 |
| entry_rejection | 通过 | 12.864970 | 12.629025 | 12.629025 |
| first_rejection | 通过 | 14.124393 | 13.827982 | 13.827982 |
| drop_final_ack | 通过 | 13.851559 | 13.610680 | 0.353385 |
| delayed_ack_timeout | 通过 | 13.844905 | 13.626915 | 13.626915 |
| exit_before_confirmation | 通过 | 13.696967 | 13.443405 | 13.443405 |
| exit_after_confirmation | 通过 | 13.779288 | 13.542405 | 0.112429 |

这些受控停止的worker退出码1、17或18是预设异常路径；验收驱动器检查预期错误、账本、结算与导出后退出0，不将被测试worker退出码改写为成功。normal用例包含staging连续3次、training连续3次和task阶段1次快照，实际等待3个短子进程。研究阶段函数未替换；包外故障注入限socket响应、受控CPU与退出边界。

开发回归失败原件仍保留：development-2的stage_enter通知校验和隔离子进程导入失败、development-1/stable-1的并发修改字节不一致、旧历史入口测试断言及freeze-driver-schema-error.json。已分别修复通知合同、明确子进程路径与-B、稳定最终字节、验证历史入口拒绝和读取独立审核的实际JSON结构；没有删断言、吞异常或改写旧退出码。

CPU复算按外层kernel waited CHILDREN与supervisor SELF+waited分别保留嵌套口径；已关闭区间加独立开放尾段等于supervisor总量。外层另捕获已知0.03s supervisor退出尾段，内部值不再次加入外层。完整Windows/WSL桥、正式controller最后写入与退出尾段仍未测量，full_resource_acceptance=false；没有声称全生命周期成本验收通过。

## 冻结身份与执行状态

- proposed attempt：`w1-g1-gppo-phase-accounting-repair-v1-once`
- Manifest：`9ac065202126484968227469c16634ffbdfd5c30933c73a5c3a407e6c26815f9`
- Hashes：`a4d2be66a4834fe62ee461d0e5ba7255b77b0ee2adef1702b94f41c48fd12ae1`
- RESOURCE_REQUEST：`4eda5196e3fc0fce1c69f74b93b0b97c1a1335c655fc4007bce95d823a5f252b`
- RESOURCE_REQUEST保持NOT_APPROVED；runner_ready仅表示此次工程阻断已关闭。
- 唯一命令见package/unique-launch-command.md；新正式执行仍需绑定以上身份的新授权与新token。本轮未创建正式token或正式研究attempt。

科学代码、四臂、种子、训练规模、父场景、门槛和预算逐项与旧包核对不变。当前预算仍为22,752环境步、1,152策略更新、240episodes、wall5,070s、CPU7,790s。

全部测试、协议、首错、SQLite与受控导出原件放包外engineering-evidence-final；归档副本与最终冻结字节一致。只读Windows/WSL预检与真实生产工程回放是两类不同证据。

GitHub状态见git-archive-status.json；签名代理未恢复时只能报告本地归档与远端未归档。
