"""Freeze eight fixed label-qualification parents without inspecting outcomes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
ATTEMPT = "w1-action-conditioned-task-outcome-label-qualification-v6-dependency-cpu-settlement-once"
PARENTS = tuple(f"train-{index:04d}" for index in range(56, 64))
TAPE = ROOT / "runs" / "w1-light-repaired-fair-rerun-v2-nativefs-once" / "tapes-train.json"
PRIOR_SPLIT = ROOT / "research-plans" / "w1-eawm-jepa-world-model-production-repair-v4-e2e-integration" / "parent-split.json"
BASELINE_PLAN = ROOT / "research-plans" / "gppo-world-baseline-comparison-preparation-20260925" / "parent-plan.json"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    tape_bytes = TAPE.read_bytes()
    tape_sha = sha(tape_bytes)
    expected_tape_sha = "f2e6f7b2d7590989e305be5d7c6faf685369e15563a7b2fefb18c0468e7aa510"
    if tape_sha != expected_tape_sha:
        raise RuntimeError("FROZEN_SOURCE_TAPE_DIGEST_MISMATCH")
    rows = json.loads(tape_bytes.decode("utf-8"))
    by_parent = {}
    for row in rows:
        if str(row.get("parent")) in PARENTS:
            if str(row.get("parent")) in by_parent:
                raise RuntimeError("SOURCE_PARENT_NOT_UNIQUE")
            by_parent[str(row["parent"])] = row
    if set(by_parent) != set(PARENTS):
        raise RuntimeError("FIXED_PARENT_SET_INCOMPLETE")

    prior = json.loads(PRIOR_SPLIT.read_text(encoding="utf-8"))
    prior_ids = {str(row["parent"]) for split in prior["splits"].values() for row in split}
    if prior_ids.intersection(PARENTS):
        raise RuntimeError("FIXED_PARENT_COLLIDES_WITH_PRIOR_REGISTERED_MATRIX")
    baseline = json.loads(BASELINE_PLAN.read_text(encoding="utf-8"))
    baseline_ids = {str(row["parent"]) for row in baseline.get("rows", [])}
    if not set(PARENTS).issubset(baseline_ids):
        raise RuntimeError("FIXED_PARENT_IDENTITIES_NOT_PRESENT_IN_BASELINE_PLAN")

    parents = []
    for parent in PARENTS:
        row = by_parent[parent]
        if row.get("role") != "train" or row.get("generator_split") != "train" or row.get("condition") != "W1":
            raise RuntimeError("FIXED_PARENT_SOURCE_ROLE_MISMATCH:" + parent)
        parents.append({key: row[key] for key in (
            "parent", "tape_index", "scenario_sha256", "structural_sha256",
            "exogenous_key", "generator_split", "condition",
        )})

    payload = {
        "schema": "w1-action-conditioned-task-outcome-parent-split/1.0.0",
        "attempt": ATTEMPT,
        "selection_rule": "fixed contiguous W1 train tape indices 56-63, selected by identity before new collection outcomes; no outcome screening or substitutions",
        "source_train_tapes_sha256": tape_sha,
        "prior_registered_split_sha256": sha(PRIOR_SPLIT.read_bytes()),
        "historical_use_elsewhere": "not_proven_absent; the baseline parent plan lists these identities, while available attempt-level usage registration is incomplete",
        "split_counts": {"label_qualification": 8},
        "repeats": {"label_qualification": 1},
        "parents": parents,
        "no_opportunity": "retain as missing; do not replace or count as a zero label",
    }
    out = PACKAGE / "parent-split.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"written": str(out), "parents": list(PARENTS), "tape_sha256": tape_sha,
                      "prior_registered_overlap": 0, "historical_unseen_claim": False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
