# W1 当前研究进度（2026-10-09，Asia/Shanghai）

**最新已完成的正式运行为恢复 v2 的 technical_stop；后续 v3 为修复准备快照。本次 GitHub 发布没有启动或批准任何训练。**

## 已取得的成果

- 历史动作后果采集完成 288 个窗口：174 complete、114 no_opportunity；实际采集 6,839 环境步，原父场景划分保留。
- 8201 世界模型完成 40 epochs、320 次更新并保存完整 checkpoint。原 v5 共确认 475 次更新，另 1 次更新与其内部 forward 为 pending；8202 没有完整 checkpoint，8203 尚未开始。原运行因世界模型阶段 CPU 达 1,600.227401 秒而停止，未进入预测门。[原 v5 报告](original-v5/FINAL_REPORT_CN.md)
- 恢复包复用这批数据和 8201；不重采，8202/8203 各需从头完成 320 次更新。

## 最新正式停止（恢复 v2）

Attempt：`w1-drone-action-consequence-world-model-v5-recovery-v1-linux-release-v2-once`。提交一次，退出码 1，身份已消费，未自动重试。

首错为 `input_admission output_writes limit exceeded`：准入需写入 research-identity、reused-source-inventory、lean-inputs 三项，但第三次写入在执行前被账本拒绝。288 个窗口已走到复用/审计路径，完整 lean input 未发布，因此不能写成准入阶段成功。

本次世界模型 forward/更新、GPPO/critic 更新和任务 episodes 均为 0。已测服务器 wall 141.676684 秒、Linux CPU 138.310472 秒；SSH、systemd manager 和退出尾段计量不完整。它是工程技术停止，不是世界模型效果的负结果。[正式停止报告](formal-run-v2/FINAL_STOP_REPORT_CN.md)

## 后续修复准备快照

目录 `package-v3/` 为 output_writes 合同修复的源码与文档快照，补充三项准入写入、原子持久化及幂等规则。发布时仅核验了源清单和复制字节；不据此宣称服务器部署、完整验收或新正式运行通过。文档内部分说明沿用历史版本，当前资源数量以本快照 RESOURCE_REQUEST 为准。

Attempt：`w1-drone-action-consequence-world-model-v5-recovery-v1-linux-release-v3-output-writes-contract-repair-once`；申请状态 `NOT_APPROVED`。

| 身份 | SHA-256 |
|---|---|
| Manifest | `6d30ab05cc7015045c7547f2c62c65e0fb0a20325906b7fdcadd562dbbe19f5b` |
| Hashes | `aa5d44b5ca8d7274ac49c2a6dd33baa35e479ce640448e8a607892c953af0627` |
| Request | `5d9c42a564478a5d785b6453fa27f33aabbfc13868674e4ae31f4b90f42fd877` |

恢复申请：22,752 环境步、640 世界模型更新、1,152 策略更新、240 episodes；服务器 wall 23,000 秒、CPU 12,000 秒。上限不是已发生的消耗。

流程仍为数据/8201准入 → 8202/8203训练 → 三模型严格恢复 → 原离线预测门 → 通过后九路critic/GPPO/PreCo与240 episodes → 独立复算和结算。预测门失败不进入后续策略实验；当前新分支预测、任务收益、Hungarian比较和在线成本均未评价。

## 历史科学结论不变

此前冻结特征 GPPO 探索实验完成240 episodes：G1−G0/T为−0.001863，G1−Hungarian为−0.058043，G1 CPU均值32.09ms、wall p95为67.48ms。没有建立额外任务价值或实用优势。此负结果属于上一分支，不能拿来替代当前后果监督分支尚未完成的评价。[历史报告](previous-feature-experiment/研究报告.md)

## 档案范围

本档案是进度和源码证据快照，不是完整可直接执行的部署包。模型二进制、288个压缩窗口、SQLite/journal、原始大型日志、token及授权文件均未上传；省略资产的SHA-256索引见 package-snapshot-verification.json。所有原始报告保留当时文字，包括历史“未归档”，不追溯改写。GitHub发布不改变预算、授权或attempt状态。
