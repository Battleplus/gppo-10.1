# 本地准备修复与验收记录（2026-10-02）

本轮完成开发包的本地回归与身份冻结；不是正式实验交付验收。
`RESOURCE_REQUEST.json=NOT_APPROVED`，`runner_ready=false`。
未连接 SSH，未调用真实 W1 环境，未读取封存确认集或研究 checkpoint，
未创建或消费正式 attempt。本轮没有新增世界模型或 GPPO 效果结论。

## 修复和证据范围

- 登记接口核对真实姓名与确定性 name_id；GPU worker 不接受空绑定，
  原生入口等待 runner PID 与请求 GPU 集合匹配，超时记录 EXPIRED。
- 远端入口异常保留 technical_stop、CPU/wall 证据和结算；资源失败
  不能保留 complete 或提前登记 SUCCEEDED。
- CPU 合同采用进程 tick 基线、活跃增量和回收子进程用量，身份
  无法核实时停止。废止以重叠快照最大值证明完整计量的解释。
- 输入来源验证统一使用包内来源证据，核查原清单成员、摘要及
  复用函数 AST，不借用外部工作区补齐传递依赖。
- launcher 参数传递测试将输出隔离在临时目录；此前测试写入开发包
  的 accounting residue 已另存于包外，没有删除历史证据。
- CPU 尾段失败夹具同步 charged-once 字段，确保测试实际触及尾段
  不一致检查。原始失败输出另存，未修改生产门槛。
- 启动说明明确指定冻结 Windows SSH 控制器解释器；默认 Python
  缺少 paramiko 的失败，以及挂载盘被原生文件系统门拒绝的输出均保留。

## 实际验证

最终 Ubuntu-24.04 / CPU 回归：125 项通过，退出码 0。
原始命令与完整 stdout/stderr 在 `final-linux-regression.*`。
测试后只改启动说明与开发状态文档，再执行正式生成器冻结。

生产合成集成使用 collector、验证器、G1/G2 训练和选择、
checkpoint 写入/恢复、逐候选预测、指标复算、SQLite 账本和受控导出。
环境交互使用底层生命周期替身；全网络结构运行于 CPU，1 epoch、
2/2/8 合成父场景，并非 24/8/8 正式矩阵，也不验证真实任务效果。
无机会路径没有模型或 checkpoint 调用。

最终一次完整测试中的两条模型测试如实计数：

| 测试 | 初始化/加载 | batch forward | backward | 更新 | checkpoint 写 | checkpoint 读 | 样本评价 |
|---|---:|---:|---:|---:|---:|---:|---:|
| joint pipeline | 12 | 84 | 12 | 12 | 6 | 6 | 420 |
| sequence world | 12 | 66 | 6 | 6 | 6 | 6 | 330 |
| 合计 | 24 | 150 | 18 | 18 | 12 | 12 | 750 |

这张表是最终完整测试的一次执行，不是全部历史回归总消耗。
较早失败与焦点回归亦有合成计算；没有统一整轮聚合账本，不能
把此表冒充全部历史测试结算，也不能将合成模型调用报告为零。
joint pipeline 的逐调用汇总见 `joint-success.json`，原生合成环境
构造 12、reset 12、环境步 288、分支 60，均为替身，不是真实采集。

Windows 原目录结构预检退出码 0；使用正式入口、冻结控制器解释器
和包外 NOT_APPROVED/null-token 结构绑定，不构成正式授权验证。
WSL 创建临时原生诊断副本，按最终清单逐文件复制并核验身份，
执行生产 `native_launch.py --preflight-only`，退出码 0。它只检查
清单和输入合同，不验证远端 CUDA、SSH 或运行时。
诊断副本不在 proposed attempt 路径中，没有启动 supervisor/worker。

`delivery.json` 保存实际命令、退出码和最终三项摘要；
`native-copy-preflight.json` 保存子命令、复制身份和前后摘要。
原目录 `preflight-before.json` 与 `preflight-after.json` 覆盖全部
202 个文件，差异为空，未创建 attempt、锁、SQLite 或运行输出。
最终验证后没有继续改写包内内容。

## 尚未完成的工程验收

1. 终态登记与最后一次采样后的失败持久化尾段，尚未完整测量。
2. 尚未采样、尚未回收的短子进程不能由当前实时探针完整计量；
   最终回收用量不证明连续阶段上限，身份缺失路径会保守停止。
3. GPU/PID 轮询、真实服务器登记、Windows SSH 生产链与远端 GPU
   测试没有执行；学习时限仍缺 GPU 基准依据。

Luna 两项复核已返回，以上阻断项按实际源码发现保留，没有以
测试通过数量替代正式验收。当前不提交正式运行批准请求。

## 登记和归档

按用户提供的 v1.2 登记说明，服务器工作负载开始前须取得用户
真实姓名，锁定追加固定登记文件并回读，再绑定实际 PID/PGID/GPU。
当前未收到真实姓名，未进行服务器登记或启动。

远端 HEAD 只读核验为 `6aa16410b0e69aaceb7dee0b4d260eaf76d03e02`。
ssh-agent 为 Stopped / Disabled。未尝试新的提交、未关闭签名、未
强推。本轮远端未归档；小型代码和证据另存本地归档副本，原有
暂存内容保留。开发冻结身份不代表已获批的可执行研究版本。
