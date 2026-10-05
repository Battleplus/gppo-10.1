# 公开历史—动作后果机制提案

用户授权：继续完成一页机制提案并及时GitHub存档；不包含新动态实验或训练。

- mechanism-proposal.md：主要交付、具体假设、任务证据、对照与否证。
- capability-audit.md：现有能力，防止把已有循环记忆包装成新增贡献。
- audit.py：可复现离线统计；仅stdlib和冻结纯公开记忆模块。
- summary.json / decision-evidence.jsonl：全部56 episode、774决策、367非NOOP选择，保留来源行号。
- input-index.json：原始日志和代码身份；verification.txt：执行结果。

这是对已观察开发记录的事后描述，不是预测准确性或策略收益实验。没有拟合模型、重新计算规则动作、调用环境、加载checkpoint或创建新预算。

GitHub归档只保存本目录的小型派生文件。原始大型运行日志、模型、SQLite、授权token和凭据未上传；这不是完整运行备份。原始来源通过绝对路径与SHA-256引用。

研究状态：HYPOTHESIS_DEFINED_NOT_VALIDATED。旧首窗口oracle方案仍封存；本提案不是将其改名继续。
