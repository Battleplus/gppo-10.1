# v3 登记握手接线审查

日期：2026-10-03  
范围：只读检查 `w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote` 的登记回执与启动 ACK 接线；未改包内文件、未重跑测试、未连接服务器。

## 接线结论

- 工程验收链由 `launch_server_acceptance.py` 调用 `launch_joint_once.execute_registered`。控制器写入初始 token 后保持 SSH stdin 打开，等待远端登记回执；只有操作者输入与回执 job ID 完全匹配的 `START_ACK:<job_id>`，控制器才写入远端 JSON ACK。正常路径在 ACK 写出后才 `shutdown_write`。代码位置：`launch_joint_once.py:46-82`、`launch_server_acceptance.py:236-244`。
- `registration_start_handshake.py` 核对回执字段集合及 schema、事件、姓名/name ID、登记路径、manifest/hash 摘要和预算摘要；再把 ACK 绑定到同一 job ID 与 manifest/hash 摘要。远端 ACK 未通过前不会执行 RUNNING 登记、CUDA 初始化或工作负载：`registration_start_handshake.py:93-109,144-229,260-324`。
- 两处预算绑定在登记与握手之间一致：工程验收入口为 `{"totals": limits, "calls": limits["calls"]}`；原生联合入口为 `{"totals": request["totals"], "stages": request["stages"]}`。后一入口在握手之后才登记 RUNNING 并创建 worker；握手异常由外层记账入口根据 handoff/结算状态追加终态：`acceptance_entry.py:820-849,923-930`、`joint_remote_native.py:234-278,442-480`。
- 本次静态检查未发现 ACK 前关闭 stdin、回执/ACK 绑定不一致或握手失败遗漏终态的接线问题。工程验收入口把握手异常记为 FAILED；原生入口由外层记账路径完成登记终态。

## 操作与放行限制

- 操作协议文档没有说明回执出现后要输入的精确确认行。实现要求 `START_ACK:<job_id>`；建议在下一次冻结前补入操作说明，避免操作者只提供 token 后 stdin 到 EOF、导致登记后失败。这里是文档缺口，不是已观察到的代码接线错误。
- `RESOURCE_REQUEST.json` 仍为 `NOT_APPROVED` 且 `runner_ready=false`。生产研究数据门和完整 CPU 范围仍未验证；远端只读预检及 SSH/SFTP 运输 CPU 范围仍是未核验项。没有服务器工程验收成功证据。
- `DEVELOPMENT_STATUS.md` 记录的上次连接在 TCP 阶段超时，发生于 SSH 认证和远端命令之前；未创建工程 job 或研究 attempt，也未生成 token。该 v3 副本仍未冻结，且保留 v2 来源身份，不能据此启动。
- 主线程此前报告 42 项本地测试（含 3 项生产接线测试）通过；本审查没有重跑。它们只支持本地行为，不构成远端验收。

## 判定

代码级登记握手接线可供继续审查；生产放行仍阻断。确认本地接线不能替代正式授权、数据门、完整资源范围核验或服务器工程验收。
