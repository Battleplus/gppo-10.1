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
        "transition_horizon": "one native decision interval",
        "candidate_continuation_id": "one-step-public-transition-v1",
        "maximum_windows_per_parent_repeat": 1,
        "maximum_legal_candidates_per_window": 25,
        "maximum_native_steps_per_prefix": 18,
        "policy_training_steps_per_method_seed": 2048,
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
