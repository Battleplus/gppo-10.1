from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    split = json.loads((ROOT / "parent-split.json").read_text(encoding="utf-8"))
    repeats = {"train": 1, "model_selection": 1, "prediction_confirmation": 3, "task_confirmation": 3}
    matrix = {
        "schema": "w1-eawm-jepa-experiment-matrix/1.0.0",
        "source_parent_split": "parent-split.json",
        "source_parent_split_schema": split["schema"],
        "selection_rule": split["selection_rule"],
        "world_model_seeds": [8201, 8202, 8203],
        "policy_seeds": [8301, 8302, 8303],
        "methods": ["G0", "T", "G1", "G2", "H"],
        "world_model_variants": {"G1": "event_loss_disabled", "G2": "event_loss_enabled"},
        "world_model_architecture": {
            "node_dim": 32, "relation_dim": 4, "history_dim": 128,
            "hidden_dim": 128, "latent_dim": 64, "action_count": 25,
            "outcome_dim": 6, "event_count": 5, "ema": 0.99,
        },
        "world_model_optimizer": {
            "name": "AdamW", "learning_rate": 1e-3,
            "weight_decay": 1e-5, "gradient_clip_norm": 5.0,
        },
        "world_model_loss": {
            "beta_jepa": 1.0, "beta_state": 1.0,
            "beta_outcome": 1.0, "beta_collapse": 0.04,
            "beta_event_by_variant": {"G1": 0.0, "G2": 1.0},
            "unknown_masking": "required",
        },
        "world_model_model_selection": {
            "primary_metric": "parent_macro_action_selection_regret",
            "early_stopping_patience_epochs": 15,
            "confirmation_data_access": False,
            "rule": "minimum primary metric on the frozen model-selection parents; ties use lower parent-macro absolute outcome MAE, then earlier epoch",
        },
        "transition_horizon": "one native decision interval",
        "candidate_continuation_id": "one-step-public-transition-v1",
        "maximum_windows_per_parent_repeat": 1,
        "maximum_legal_candidates_per_window": 25,
        "maximum_native_steps_per_prefix": 18,
        "policy_training_steps_per_method_seed": 2048,
        "policy_configuration": {
            "optimizer": "Adam", "learning_rate": 3e-4,
            "rollout_steps": 64, "gamma": 0.99, "gae_lambda": 0.95,
            "clip_epsilon": 0.2, "entropy_weight": 0.01,
            "value_weight": 0.5, "gradient_clip_norm": 0.5,
            "maximum_optimizer_updates_per_method_seed": 128,
        },
        "preference_configuration": {
            "training_schedule": [[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]],
            "training_assignment": "episode_index plus policy_seed_index modulo 3",
            "task_confirmation": [0.8, 0.2],
            "discount": 0.99,
            "task_component_scale": 0.5,
            "utility_formula": "discount^step * (task_component_scale * p_task * vector_reward_task + p_energy * vector_reward_energy)",
        },
        "transparent_utility_configuration": {
            "task_score_divisor": 600.0,
            "energy_score_weight": 0.1,
            "initial_total_energy": 36.0,
            "component_order": ["task", "energy"],
            "residual_target": "observed one-step vector component minus transparent component",
        },
        "prior_configuration": {
            "G0": {"transparent_score": False, "learned_residual": False, "prior_scale": 0.0},
            "T": {"transparent_score": True, "learned_residual": False, "prior_scale": 0.1},
            "G1": {"transparent_score": True, "learned_residual": True, "prior_scale": 0.1},
            "G2": {"transparent_score": True, "learned_residual": True, "prior_scale": 0.1},
            "illegal_action_policy": "original mask remains authoritative",
        },
        "world_model_batch_size": 32,
        "world_model_maximum_epochs": 100,
        "world_model_maximum_train_candidate_rows": 600,
        "world_model_maximum_updates_per_variant_seed": 1900,
        "task_episode_max_steps": 18,
        "task_episode_count": 312,
        "hungarian_reuse": "one deterministic episode per parent/repeat; paired to all policy seeds",
        "window_selection_rule": "earliest decision step >=4 with at least two legal non-NOOP candidates; no opportunity is retained and not replaced",
        "repeats": repeats,
        "splits": split["splits"],
        "missing_opportunity": "preserve; no replacement; no zero regret imputation",
        "historical_use_elsewhere": split["historical_use_elsewhere"],
    }
    (ROOT / "experiment-matrix.json").write_text(json.dumps(matrix, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"parents": split["split_counts"], "methods": matrix["methods"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
