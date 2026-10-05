# 资源申请推导与冻结条件

当前 `RESOURCE_REQUEST.json` 保持 `NOT_APPROVED`。数值由 `parent-split.json`、`experiment-matrix.json` 和既有 W1 2048步策略预算逐项推导；历史余额不抵扣。

## 统一计数公式

设：

- `P_train`, `P_select`, `P_confirm`, `P_task` 为预注册父场景数；
- `R_confirm`, `R_task` 为每父场景外生重复数；
- `W(p,r)` 为该父场景/重复中完整决策窗口数；
- `L(p,r,w)` 为窗口合法候选数（含是否有机会的真实值）；
- `S_wm=3`, `S_policy=3` 为冻结种子数。

则一次申请必须由以下可复算量组成：

```text
confirmation_units = sum(P_confirm * R_confirm)
candidate_branches  = sum(L(p,r,w) for confirmation windows)
prediction_forwards = S_wm * sum(W for prediction parents)
policy_forwards     = S_policy * sum(W for task parents) * number_of_policy_arms
environment_steps   = sum(actual prefix and continuation steps for every approved arm)
resets              = sum(parent/repeat unit resets)
```

每个候选 forward 必须按实际候选数计费；同一窗口的所有合法候选必须作为一个 batch。训练更新、checkpoint、完整 CPU、wall、RSS 和 native/export storage 也必须从实际 batch、epoch、模型数和 trace 字节数推导，而不是沿用旧预算。

## 本次固定矩阵

- 训练24×1、模型选择8×1、预测确认8×3，各单元只取按公开条件确定的最早一个窗口；无机会不补选。
- 每窗口最多18步共享前缀和25个一步候选分支，所以56个单元最多 `56×(18+25)=2408` 环境步、1400分支。
- 采集计费区分 reset（56次）、前缀规则决策（最多1008次）、决策快照（56次）和候选分支（最多1400次）。候选分支同时计为 `forced_first_actions` 与 `candidate_scans`，不再重复计入 `public_rule_decisions`；加上任务确认 Hungarian 的432条规则决策，全局规则决策上限仍为1440。
- 世界模型最大600个训练候选行，batch 32，100 epoch；每模型 `ceil(600/32)×100=1900` update，G1/G2各3 seed共11400。在线/EMA target各一次前向；选择集8窗/epoch/model另计，合计27600 batch forward、840000 sample evaluation。
- 预测确认24窗×6模型=144 batch forward、最多3600候选样本。
- G0/T/G1/G2各3 seed×2048步=24576策略训练环境步、1536 PPO update；G1/G2每步一次候选batch，最多12288 batch/307200候选样本。
- 任务确认四个GPPO方法×3 seed×24单元，加共享Hungarian 24单元，共312 episode、最多5616步。G1/G2最多2592候选batch。

wall/CPU 以已完成 W1 2048步训练和旧候选分支包的实际/申请尺度保守放大，并为Graph-JEPA双前向和312任务episode保留上界：54000秒 wall、108000秒完整CPU。它们是硬上限，不是预计耗时。RSS 4GiB、active 24GiB、native+export 48GiB延续已验证基础设施上限。

## 执行状态与剩余验证

生产 backend 已接入 collector、G1/G2 训练与逐候选评价、预测门、G0/T/G1/G2 策略训练、五方法任务对照、计费和结算。collector 的候选分支计费已与 `branches`、`forced_first_actions`、`candidate_scans` 分别对应，且快照独立计费。当前 `NOT_APPROVED` 仍是唯一资源授权状态，正式命令不可执行。

现有47项静态测试通过，Windows 正式只读入口也已调用 Ubuntu-24.04 的 Linux preflight 并返回0；这没有启动 staging 或 worker。仍缺在独立测试副本上以底层替身贯通 Windows→WSL→supervisor→生产 worker 的单次集成验收。任何后续包内容变更后，必须重新冻结并针对最终目录原样执行只读预检；不得只更新摘要掩盖行为变化。
