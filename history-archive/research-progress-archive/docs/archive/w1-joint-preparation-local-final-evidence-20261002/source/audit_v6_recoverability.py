"""Read only the consumed v6 artifacts; never reconstruct unrecorded states."""
import argparse
import hashlib
import json
from pathlib import Path


def inspect(root):
    files = sorted(path for path in root.rglob("*") if path.is_file())
    before = {path.relative_to(root).as_posix(): {"bytes": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files}
    source = root / "run-once" / "world-model-windows.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    candidates = [candidate for row in rows for candidate in row["candidate_branch_audit"]]
    steps = [step for candidate in candidates for step in candidate["post_action_trajectory"]]
    other_output_files = [str(path.relative_to(root)) for path in files
        if "run-once" in path.parts and path != source]
    recovery = {"windows": len(rows), "candidates": len(candidates),
        "trajectory_steps": len(steps),
        "steps_with_vector_reward": sum("vector_reward" in step for step in steps),
        "steps_with_all_task_lifecycle": sum("all_task_lifecycle" in step for step in steps),
        "steps_with_global_counts": sum("counts_after" in step for step in steps),
        "steps_with_global_energy": sum("energy_after" in step for step in steps),
        "candidate_sequence_return_records": sum("sequence_return" in candidate for candidate in candidates),
        "trajectory_key_union": sorted({key for step in steps for key in step}),
        "other_saved_run_outputs": other_output_files,
        "full_return_recoverable": False,
        "reason": "Saved target-task lifecycle/communication does not determine every task's global counts or all-UAV energy at every continuation step; missing values are not reconstructed.",
        "old_prediction_gate_stop_changed": False, "environment_calls": 0,
        "model_calls": 0, "checkpoint_calls": 0}
    after = {path.relative_to(root).as_posix(): {"bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files}
    if before != after:
        raise RuntimeError("SEALED_SOURCE_CHANGED_DURING_AUDIT")
    return {"schema": "w1-saved-return-recoverability/1.0.0", "source": str(root),
            "source_identities_before": before, "source_identities_after": after,
            "all_identities_unchanged": True, "analysis": recovery}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                     sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result["analysis"], ensure_ascii=False))
