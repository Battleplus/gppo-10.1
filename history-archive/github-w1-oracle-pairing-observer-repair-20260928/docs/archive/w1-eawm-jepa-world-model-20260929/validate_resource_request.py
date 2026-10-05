from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUMMED = (
    "environment_steps", "resets_upper", "branches", "forced_first_actions",
    "public_rule_decisions", "snapshot_captures", "candidate_scans",
    "world_optimizer_updates", "policy_optimizer_updates", "world_batch_forwards",
    "world_sample_evaluations", "encode_sample_evaluations", "actor_sample_evaluations",
    "model_initializations_or_loads", "checkpoint_writes", "wall_seconds",
    "complete_process_cpu_seconds",
)


def main() -> int:
    request = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
    matrix = json.loads((ROOT / "experiment-matrix.json").read_text(encoding="utf-8"))
    if request["status"] != "NOT_APPROVED" or matrix["methods"] != ["G0", "T", "G1", "G2", "H"]:
        raise ValueError("resource or method identity changed")
    derived = {key: sum(int(stage.get(key, 0)) for stage in request["stages"].values()) for key in SUMMED}
    mismatches = {key: {"derived": value, "declared": request["totals"].get(key)} for key, value in derived.items() if value != request["totals"].get(key, 0)}
    if mismatches:
        raise ValueError("resource totals mismatch: " + json.dumps(mismatches, sort_keys=True))
    if matrix["task_episode_count"] != 312 or request["stages"]["conditional_task_confirmation"]["task_episodes"] != 312:
        raise ValueError("task seed/parent crossing changed")
    print(json.dumps({"passed": True, "status": request["status"], "totals": derived}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
