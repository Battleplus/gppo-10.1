# Stage wiring

| Stage | Production function | Bottom dependency | Accounting | Coverage |
|---|---|---|---|---|
| zero-step | `ProductionRuntimeAdapter.zero_step_gate` | frozen runtime module import/identity checks | no env/model calls | `test_production_backend_completion` |
| reuse audit | `ABRuntimeBackend.audit_reused_labels` | sealed JSONL and audit manifest | data contract only | production backend tests |
| confirmation | legacy `AuthorizedRuntimeBackend._collect_unit` through `collect_confirmation_labels` | environment/reset/step/communication observer | `BudgetLedger.call` | adapter route and WSL integration |
| A/B training | `windowed_training.train_variant` for A/B x seeds | model, optimizer, checkpoint boundaries | per-call ledger entries | six training routes |
| prediction | `PredictionTraceRow`, `write_trace`, `metrics.compare_trace` | model forward boundary | forward/sample entries | trace file and independent metrics |
| task gate | `production_pipeline._gate_passes` | no task call unless both gates pass | task calls start only after gate | pass/fail route tests |
| task comparison | legacy `AuthorizedRuntimeBackend.run_task_comparison` with A/B model map | environment/step/model forward boundaries | episode/decision/forward entries | WSL integration and A/B routing assertion |
| settlement | `ABRuntimeBackend.settle` | durable output and SQLite close | pending/limit checks | settlement assertions |

The formal runner never selects `IntegrationBoundary`; it is only constructed
by the explicitly named WSL integration entry.
