# 启动与验收状态

当前包已完成七项静态生产修复，并修复最终独立复核发现的 history 调用、collector 计费归属、外部 token 传递/校验、运行时源代码身份和数组序列化问题；`RESOURCE_REQUEST.json` 仍为 `NOT_APPROVED`。它不等于动态实验验收；本轮不创建或运行动态 attempt。仅当未来批准绑定最终冻结身份后，才可使用正式入口。冻结脚本只生成包身份和未批准的包外授权模板，不产生运行 token。

生产 backend 已由 `runtime_backend.py` 调用本包的 collector、世界模型、策略和任务
阶段；`runtime_hooks.py` 将 native 依赖限制在环境、模型、优化器和 checkpoint 底层。
策略路由使用冻结世界模型下的原生 transition/PPO 循环，不再绕回会联合更新世界模型的
旧 `joint_training.run_group()`。
静态测试不等于真实动态验收，跨系统一次性生产采集与完整效果运行仍未执行。合成数值回归发生的前向/优化更新与正式动态调用分开记账，不能用 `dynamic_calls=0` 含混表述。

本次零动态授权范围内已完成的行为回归包括原生 GPPO/PreCo 损失路径、collector 标签函数、实际 Hungarian episode runner、causal-history 构造、GAE bootstrap 和 opportunity 汇总。正式生产 backend 已实现并接入下列阶段；现有测试覆盖阶段调用路由和若干真实生产函数，但尚未证明这些部分在一次 Windows→WSL→supervisor→worker 流程中端到端协同。不能把函数级/阶段级替身测试说成完整跨系统验收，也不能把测试替身替换成整体 FakeBackend：

1. 从真实 Windows 入口到原生暂存、supervisor、生产 worker、账本结算和受控导出的跨系统底层替身集成测试。替身只能注入环境、模型、优化器和 checkpoint 边界；测试必须实际运行生产 collector、训练调度、预测复算、策略/任务门控与结算路径。当前尚无此完整证据，因此 `runner_ready=false`；
2. 正式动态授权后的真实标签覆盖、模型训练和预测/任务效果。当前申请尚未批准，因此这些效果和运行成本未评价。

生产阶段实现位置如下：`runtime_backend.py` 调用 `production_data.py` 的标签采集、`production_world.py` 的 G1/G2 训练和逐候选评价、`world_model_pipeline.py` 的预测门、`production_policy.py` 的 GPPO 训练与任务对照；`runtime_hooks.py` 提供环境/模型/优化器/checkpoint 底层边界。`checks.json` 中的静态测试数不是上面第1项跨系统全链替身验收。

`launch_once.py --preflight-only` 已从 Windows 实际调用 Ubuntu-24.04 的只读 Linux preflight，返回码为0；该路径验证包身份、Python/torch 依赖和入口，但没有暂存、创建 attempt 或启动 worker。正式动态前仍须获得绑定最终身份和完整预算的一次性授权。若补做全链替身验收，应在独立测试副本运行，不得使用或消费正式授权，也不得把测试副本结果说成正式运行结果。

Luna 追加复核发现并已在本版本修复两处生产路径缺口：`native_launch.py` 显式导入
`sha256_file`；策略训练和任务评价构造环境时调用 `adapter.boundary.environment`。
构造器边界回归通过。它们仍不能代替完整 Windows→WSL→supervisor→worker 验收。

目前没有经清单绑定的 test-only native entry/worker，因此还不能在不绕过 sealed package
校验的前提下，以底层替身跑通正式 runner 全链。Luna 指出的此阻断仍成立，`runner_ready=false`。
请勿将已有的 adapter/stage mock 测试称为跨系统生产验收。
