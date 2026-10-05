# Stage Wiring Evidence

| stage | production entry | bottom boundary | evidence |
|---|---|---|---|
| collection | `ProductionDataCollector.collect` | environment constructor/reset/step | `world-model-windows.jsonl`, ledger |
| world model | `train_select_world_models` | model, optimizer, checkpoint | six world checkpoints and training summary |
| prediction | `evaluate_prediction_confirmation` | model forward | prediction trace and recomputed metrics |
| GPPO | `train_policy_routes` | policy model/optimizer/environment/checkpoint | 12 route rows and policy checkpoints |
| task | `evaluate_task_confirmation` | policy/world checkpoint load and environment | 312 episodes, decision costs, H rows |
| settlement | `W1RuntimeBackend.settle` | durable JSON and ledger | settlement and export manifest |

The gate branches are selected by `gate_fixture` only after production
prediction metrics have been generated. The fixture never supplies metrics or
replaces a stage function. Formal dynamic calls remain zero.
