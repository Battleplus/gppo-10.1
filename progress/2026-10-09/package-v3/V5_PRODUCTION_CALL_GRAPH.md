# v5 生产调用图

```text
launch_pilot.verify -> server_runtime_contract.verify -> external authorization
  -> systemd managed_host -> supervise -> StageServer -> production_driver.Driver
M10Environment._observation
  -> public_prefix.capture_observation -> serialize_public_prefix
  -> public_controller.public_copy (原类型验证) -> PublicDecisionAdapter
  -> CausalPublicHistory (读取同一 serializer 输出的前 128 维)
Driver.collect -> collect_window -> context_for/catalog
  -> serialize_public_prefix -> compact prefix_records
  -> snapshot_payload (pre-action time / random key / history 校验)
  -> BudgetLedger snapshot_store / restore_candidate / candidate_public_snapshot_probe
  -> saved public state + context/keys/records -> serialize_public_prefix(expected)
  -> branch actions / fixed continuation / lifecycle outcomes
  -> admit_window / labels -> serialize_public_prefix -> label public_prefix record
Driver.audit_window_input -> validate_window (全体候选先验证)
  -> graph5_from_m10_observation -> serialize_public_prefix
  -> prepare_world_input / bind_model_input -> shared candidate audit
  -> world model entry verifies model bindings before neural forward
Driver.features -> build_features -> serialize_public_prefix via prefix_records/audit
  -> frozen ensemble feature path (future authorized research only)
serialize_prediction -> serialize_public_prefix -> prediction public_prefix record
Driver.rollout -> replay.seal -> verify_replay_candidate -> serialize_public_prefix
  -> saved behavior snapshot / old_log_prob reconstruction
native GPPO update -> consequence_replay.verify_update -> same prefix verifier
  -> current adapter/policy new_log_prob -> GPPO/PreCo update
Driver.restore_policy -> nine final model gate -> evaluation tape
  -> task results -> independent recompute -> settlement/export
```

所有候选层复用 candidate_keys.audit 中相同的前缀序列摘要。公共共享 graph 编码用未绑定的 PublicPrefix；candidate model binding 再绑定每一行的结构化 key、parent/window/time 和快照摘要。没有模块自行拼接公共 flat。

本轮受控路径动态执行到真实 GPPO 的更新前身份校验器；没有执行模型 forward、概率计算、GAE 更新或 GPPO 函数。训练、九路恢复和任务评价仅为后续正式调用图，不是本轮通过的研究阶段。
