# input_admission 输出合同

该阶段有三个正式账本写入：`research-identity.json`、`reused-source-inventory.json`、`lean-inputs.json`。包外诊断副本、阶段协议日志、哈希清单和结算/导出产物不使用该阶段的 `json_output` 额度。

每次新写入先由真实 `BudgetLedger.call(json_output, {output_writes: 1}, ...)` 原子预留，随后由 `durable_atomic_json` 写入同目录临时文件、fsync、原子替换并做写后字节和 SHA-256 校验。写入失败时目标文件不发布，预留不退款，避免重试造成隐式超额。若目标已存在且内容完全相同，则视为幂等观察，不重复收费；内容不同立即拒绝。

正式恢复准入成功的判据是：`lean-inputs.json` 可读且哈希在写入前后相同，包含 288 个窗口身份，其中 174 条 `complete`、114 条 `no_opportunity`，并保留 split、candidate key、labels 和来源摘要；随后账本能够进入下一阶段。专项验收使用真实 Driver、BudgetLedger、StageServer，只读复用缓存，不调用采集器、模型或 GPPO。
