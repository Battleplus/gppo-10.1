"""Deterministically freeze W1 parent identities without reading outcomes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
TRAIN_TAPES = ROOT / "runs" / "w1-light-repaired-fair-rerun-v2-nativefs-once" / "tapes-train.json"
DEV_TAPES = ROOT / "runs" / "w1-light-repaired-fair-rerun-v2-nativefs-once" / "tapes-light-dev.json"
OLD_SELECTION = ROOT / "research-plans" / "w1-action-relative-auxiliary-launch-wiring-v1" / "new-prediction-parent-selection.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    train_rows = json.loads(TRAIN_TAPES.read_text(encoding="utf-8"))
    dev_rows = json.loads(DEV_TAPES.read_text(encoding="utf-8"))
    old = json.loads(OLD_SELECTION.read_text(encoding="utf-8"))
    excluded = {str(row["parent"]) for row in old["parents"]}
    excluded.update(str(item) for item in old.get("sealed_task_parents", []))
    excluded.update(str(row["parent"]) for row in dev_rows)
    unique = {}
    for row in sorted(train_rows, key=lambda item: int(item["tape_index"])):
        unique.setdefault(str(row["parent"]), row)
    available = [row for parent, row in unique.items() if parent not in excluded]
    counts = {"train": 24, "model_selection": 8, "prediction_confirmation": 8, "task_confirmation": 8}
    cursor = 0
    split = {}
    for name, count in counts.items():
        rows = available[cursor:cursor + count]
        if len(rows) != count:
            raise RuntimeError(f"not enough parents for {name}")
        split[name] = [
            {key: row[key] for key in ("parent", "tape_index", "scenario_sha256", "structural_sha256", "exogenous_key", "generator_split", "condition")}
            for row in rows
        ]
        cursor += count
    parents = [item["parent"] for rows in split.values() for item in rows]
    if len(parents) != len(set(parents)):
        raise RuntimeError("parent split overlap")
    output = {
        "schema": "w1-eawm-jepa-parent-split/1.0.0",
        "selection_rule": "sort frozen train tape by tape_index after excluding the explicitly enumerated old prediction parents, reserved task parents, and all light-dev validation parents; assign contiguous blocks before observing outcomes",
        "source_train_tapes_sha256": sha(TRAIN_TAPES),
        "source_dev_tapes_sha256": sha(DEV_TAPES),
        "excluded_parent_count": len(excluded),
        "old_prior_matrix_parent_count_claimed": int(old.get("excluded_prior_matrix_parent_count", 0)),
        "old_prior_matrix_parent_count_explicitly_excluded": len(old.get("parents", [])),
        "split_counts": counts,
        "repeat_contract": {"train": 1, "model_selection": 1, "prediction_confirmation": 3, "task_confirmation": 3},
        "historical_use_elsewhere": "not proven absent by available manifests",
        "splits": split,
    }
    (PACKAGE / "parent-split.json").write_text(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"excluded": len(excluded), "available": len(available), "split_counts": counts, "overlap": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
