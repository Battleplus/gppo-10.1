# 恢复审计

## 身份与保留边界

Attempt：`w1-drone-action-consequence-world-model-v5-recovery-v1-once`。本版本独立于已停止的 v5。v5 冻结包、technical_stop、首错、SQLite、两个 pending 调用、监督器/journal、partial 8202 工作状态及全部导出均只读保留。新授权必须绑定本版本冻结摘要；旧授权不复用。

## v5 实际状态

- 配对采集已完成 288 个窗口：174 `complete`、114 `no_opportunity`，SQLite 完整性为 `ok`。
- 采集环境步为 6,839；策略训练环境步为 0；任务评价环境步为 0。
- 世界模型账本为 475 次 complete update、1 次 pending；另有一个 pending `world_forward`。8201 的 checkpoint 证明完成 320 updates；执行 activity 明确位于 `world:8202`。因此其余 155 complete updates 属于未完成 checkpoint 的 8202 历史工作，另一个 8202 update pending，二者均不计入新训练。
- 阶段 CPU 尾部为 1,600.227401 秒，超过旧阶段 1,600 秒 cap；worker 最终账本不可用、总资源 scope 不完整。该尾部是实测但不完整的阶段下界，不能解释成完整消耗或可用额度。

## 288 窗口准入

恢复包内缓存来自 v5 verified export 的 `package/run-once/windows/*.json.gz`，共 25,549,852 字节。使用导出清单逐文件核对字节数与 SHA-256；读取并解压一次，核对 288 个唯一 `(parent, repeat, window_id)`、96 父场景 registry、父场景 split、candidate key catalog、candidate snapshot/label identities 和 public-prefix contract，并使用原 `admit_window` 重算 complete 标签。结果：

| Split | Complete | No opportunity | Total |
|---|---:|---:|---:|
| train | 58 | 38 | 96 |
| calibration | 31 | 17 | 48 |
| development | 27 | 21 | 48 |
| evaluation | 58 | 38 | 96 |
| Total | 174 | 114 | 288 |

文件摘要失败 0，身份重复/缺失 0，split 错配 0，合同或标签重算差异 0。`no_opportunity` 没有分支或标签，不作为负例。没有替换父场景，也未读取原始大型 JSONL。数据准入在正式入口还会用真实账本再次逐项核验；本机工程审计不是正式授权运行。

## seed 8201 准入

Checkpoint SHA-256：`ebd9894270d6a32edec6892a912d11933c1a266c1a831294388ffb06be52d8b0`（2,896,861 bytes）。本机只读核验结果：

- `torch.load(weights_only=True)`、metadata 摘要、state_dict 摘要、optimizer 摘要通过。
- metadata seed 8201、epochs 40、updates 320、final-fixed-epoch selection；optimizer Adam step 为 320。
- 模型 state strict load、optimizer strict restore 后内部摘要一致。
- metadata 的 24 个 `train_parents` 与冻结 parent registry 中 train 父场景完全一致；58 个 train complete 窗口给出 `40 * ceil(58/8) = 320` updates，候选数为 257/epoch，和 checkpoint 一致。
- 模型与槽位合同为 task capacity 6、`public-action-graph-order`、`w1-structured-candidate/1`、`w1-public-prefix/1`，readout 25、packed dimension 507。

上述恢复测试在本机 PyTorch 2.13.0+cpu 完成。正式服务器仍必须按冻结 runtime 再做完整身份准入，不能以本机测试替代服务器核验。

## 固定恢复路径

1. 输入准入：从包内 288 个 gzip 小缓存核验摘要、split、身份和标签；采集器不可调用。
2. 8201：核对冻结 binding 并复用 checkpoint，不重训。
3. 8202、8203：各用原 seed、初始化、损失、训练父场景和超参数，从头做 40 epochs、batch 8，共 320 次更新。
4. 三模型按原严格 state/optimizer 恢复路径完成 ensemble restore。
5. 原 offline prediction gate；失败则 policy updates=0、task episodes=0。
6. gate 通过才执行原九路 critic/GPPO/PreCo 训练、九个完整 checkpoint 恢复及 240 episode 矩阵。
7. 独立复算、结算和导出。

唯一入口见 `unique-launch-command.md`。新身份授权前不得创建 token、attempt 或正式运行目录。
