"""Regenerate the frozen matrix, budget request, manifests, and hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiment_matrix import build_matrix, derive_budget


ROOT = Path(__file__).resolve().parent
ATTEMPT = "w1-action-outcome-learning-loop-launch-repair-v1-once"
TAPE = Path(r"E:\Z博士\research-plans\w1-light-repaired-fair-rerun-infra-repair-v1\tapes-train.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resource_request(matrix: dict, derived: dict) -> dict:
    counter_names = tuple(derived["active_call_totals"])

    def limits(dynamic: dict, wall: int, cpu: int, artifacts: int) -> dict:
        values = {name: 0 for name in counter_names}
        values.update(dynamic)
        values.update({
            "wall_seconds": wall,
            "complete_process_cpu_seconds": cpu,
            "artifact_bytes": artifacts,
        })
        return values

    dynamic = derived["stage_dynamic_counts"]
    gib = 1024 ** 3
    stage_limits = {
        "staging_and_zero_step_gate": limits({}, 600, 1200, 256 * 1024 ** 2),
        "label_collection": limits(dynamic["label_collection"], 24000, 48000, 16 * gib),
        "supervised_training_and_selection": limits(
            dynamic["supervised_training_and_selection"], 3600, 7200, 18 * gib
        ),
        "independent_prediction_evaluation": limits(
            dynamic["independent_prediction_evaluation"], 600, 1200, 18 * gib
        ),
        "conditional_task_comparison": limits(
            dynamic["conditional_task_comparison"], 1200, 2400, 22 * gib
        ),
        "settlement": limits({}, 600, 1200, 24 * gib),
        "verified_export": limits({}, 600, 1200, 24 * gib),
    }
    supervisor_stages = {
        name: {
            "wall_seconds": values["wall_seconds"],
            "complete_process_cpu_seconds": values["complete_process_cpu_seconds"],
            "artifact_bytes": values["artifact_bytes"],
        }
        for name, values in stage_limits.items()
    }
    return {
        "schema": "w1-action-outcome-learning-loop-request/1.0.0",
        "status": "NOT_APPROVED",
        "attempt": ATTEMPT,
        "purpose": "One finite fixed-continuation label, supervised learning, independent prediction evaluation, and gated one-shot task comparison loop.",
        "derived_dynamic_budget": derived,
        "stage_limits": stage_limits,
        "stages": supervisor_stages,
        "totals": {
            **derived["active_call_totals"],
            "wall_seconds": 31200,
            "complete_process_cpu_seconds": 62400,
            "artifact_bytes": 24 * gib,
            "aggregate_native_plus_verified_export_bytes": 48 * gib,
            "all_resident_rss_bytes": 4 * gib,
        },
        "all_resident_rss_bytes": 4 * gib,
        "shutdown_reserve": {
            "wall_seconds_within_each_dynamic_stage": 60,
            "complete_process_cpu_seconds_within_each_dynamic_stage": 240,
            "artifact_bytes_within_active_limit": 256 * 1024 ** 2,
        },
        "resource_limit_basis": {
            "classification": "conservative technical stop ceilings, not predicted consumption",
            "dynamic_counts": "exact maxima derived from the frozen parent/repeat/candidate/epoch matrix",
            "label_and_task_time": {
                "reference": "completed repaired fair rerun: 6918 environment steps, 3036.79 wall seconds, 3658.30 complete CPU seconds",
                "label_ceiling": "41184 steps with additional snapshot/branch overhead: 24000 wall and 48000 complete CPU seconds",
                "task_ceiling": "1296 steps plus three-seed one-shot inference: 1200 wall and 2400 complete CPU seconds",
            },
            "training_time": "5700 optimizer updates and 483600 maximum sample evaluations: 3600 wall and 7200 complete CPU seconds",
            "storage": {
                "reference": "completed 144-branch oracle archive: 465837556 bytes (about 3.24 MB per branch)",
                "label_ceiling": "2200 branches project about 7.12 GB; 16 GiB allows more than 2x for snapshots, logs, and ledger state",
                "active_ceiling": "24 GiB includes label artifacts, three checkpoints, analysis, settlement, and shutdown reserve",
                "aggregate_ceiling": "48 GiB allows one native active copy and one verified Windows export",
            },
        },
        "launch_repair": {
            "dynamic_limits_changed": False,
            "time_limits_changed": False,
            "storage_limits_changed": False,
            "reason": "Windows-to-WSL startup replaces invalid Windows-side POSIX staging and remains inside existing staging/export infrastructure ceilings",
            "measurement": "Linux artifacts persist combined staging and total timing; total CPU adds Windows preflight, Linux launcher, and supervisor worker-subtree CPU. Windows post-preflight process CPU and the cross-process start gap remain explicitly unavailable; the Windows wrapper separately checks its outer wall interval",
        },
        "progression": {
            "data_contract_failure": "stop before training",
            "prediction_gate_failure": "stop before task comparison",
            "task_comparison": "run only when learned three-seed mean strictly beats both public baselines in parent-macro MAE and beats transparent selected-action regret",
            "task_materiality": {"utility_delta": 0.01, "cpu_mean_seconds": 0.010, "wall_p95_seconds": 0.050},
        },
        "minimum_data_parent_coverage": {"train": 18, "model_selection": 6, "prediction_evaluation": 6},
        "storage_contract": {
            "active_state": "WSL native ext4 only",
            "export": "single per-file SHA-256 verified Windows export after worker settlement",
        },
        "model_contract": {
            "architecture": "827-128-64-1 tanh MLP",
            "primary_target": "remaining frozen utility under hungarian-v1-fixed",
            "training_seeds": [7101, 7102, 7103],
            "batch_size": 64,
            "maximum_epochs": 100,
            "early_stopping_patience": 12,
            "task_use": "three-seed mean once, followed by Hungarian continuation",
        },
        "historical_credit": 0,
        "unused_credit_transfer": False,
        "automatic_retry": False,
        "automatic_extension": False,
    }


def main() -> int:
    matrix = build_matrix(TAPE)
    derived = derive_budget(matrix)
    write_json(ROOT / "experiment-matrix.json", matrix)
    write_json(ROOT / "derived-budget.json", derived)
    write_json(ROOT / "RESOURCE_REQUEST.json", resource_request(matrix, derived))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
