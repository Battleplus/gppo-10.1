"""Run the task pipeline through synthetic bottoms and export its evidence."""
from __future__ import annotations

import argparse
import collections
import importlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import traceback
from typing import Any


PLAN_ROOT = Path(__file__).resolve().parent
SOURCE_PACKAGE = PLAN_ROOT / "package"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return {"type": type(value).__name__}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _world_model_hashes(document: Any) -> dict[tuple[str, int], dict[str, Any]]:
    if not isinstance(document, dict) or not isinstance(document.get("models"), list):
        return {}
    result = {}
    for row in document["models"]:
        if not isinstance(row, dict) or type(row.get("seed")) is not int:
            return {}
        key = (str(row.get("variant")), row["seed"])
        if key in result:
            return {}
        result[key] = row
    return result


def _reload_from_fixture_package(root: Path) -> None:
    module_roots = (
        "runner", "task_pipeline", "runtime_hooks", "production_policy",
        "production_data", "production_world", "w1_graph_jepa", "infra_io",
        "budget_ledger", "independent_task_recompute", "phase_handshake",
        "gppo_world",
    )
    for name in tuple(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in module_roots):
            del sys.modules[name]
    sys.path.insert(0, str(root))


def run(evidence_root: Path, archive_root: Path | None = None) -> dict[str, Any]:
    evidence_root = evidence_root.resolve()
    if evidence_root.exists():
        raise FileExistsError(f"Evidence directory already exists: {evidence_root}")
    evidence_root.mkdir(parents=True)
    sys.path.insert(0, str(SOURCE_PACKAGE))
    fixture_module = importlib.import_module("test_g1_policy_preparation")
    root, matrix, request, boundary = fixture_module.make_pipeline_fixture(evidence_root)
    root = Path(root).resolve()
    output = evidence_root / "run-once"
    _reload_from_fixture_package(root)

    runner = importlib.import_module("runner")
    infra_io = importlib.import_module("infra_io")
    affinity_before = None
    affinity_after = None
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        affinity_before = sorted(os.sched_getaffinity(0))
        if affinity_before:
            os.sched_setaffinity(0, {affinity_before[0]})
            affinity_after = sorted(os.sched_getaffinity(0))
    runner.initialize_output(output, str(matrix["attempt"]))
    ledger = runner.initialize_ledger(output, request)
    ledger.select("staging_and_zero_step_gate")

    torch = importlib.import_module("torch")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    original_torch_load = torch.load
    torch_load_paths: list[str] = []

    def recording_torch_load(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)):
            torch_load_paths.append(str(Path(file).resolve()))
        else:
            torch_load_paths.append(f"non_path:{type(file).__name__}")
        return original_torch_load(file, *args, **kwargs)

    torch.load = recording_torch_load
    result = None
    pipeline_error = None
    settlement = None
    close_error = None
    export = None
    try:
        from task_pipeline import run_pipeline
        result = run_pipeline(root, output, matrix, request, ledger,
                              boundary=boundary, allow_integration=True)
    except BaseException as exc:
        pipeline_error = {"type": type(exc).__name__, "message": str(exc),
                          "traceback": traceback.format_exc()}
    finally:
        torch.load = original_torch_load
        settlement_status = (
            "technical_stop" if pipeline_error or not isinstance(result, dict)
            else str(result.get("status", "technical_stop"))
        )
        settlement = runner._settlement(
            ledger, status=settlement_status, attempt=str(matrix["attempt"]),
            output=output,
            error=None if pipeline_error is None else RuntimeError(pipeline_error["message"]),
        )
        try:
            ledger.close()
        except BaseException as exc:
            close_error = {"type": type(exc).__name__, "message": str(exc)}

    status_is_completed = (
        pipeline_error is None and isinstance(result, dict)
        and settlement is not None
        and settlement.get("ledger_settlement_error") is None
        and close_error is None
    )
    status_document = {
        "schema": "w1-controlled-synthetic-task-pipeline-status/1.0.0",
        "attempt": matrix["attempt"],
        "status": (
            "controlled_synthetic_completed" if status_is_completed
            else "controlled_synthetic_technical_stop"
        ),
        "stage": "controlled_synthetic_task_pipeline",
        "pipeline_status": result.get("status") if isinstance(result, dict) else None,
        "task_gate": result.get("task_gate") if isinstance(result, dict) else None,
        "cost_gate": result.get("cost_gate") if isinstance(result, dict) else None,
        "settlement_status": settlement_status,
        "ledger_settlement_error": (settlement or {}).get("ledger_settlement_error"),
        "ledger_close_error": close_error,
        "pipeline_error": None if pipeline_error is None else {
            "type": pipeline_error["type"], "message": pipeline_error["message"],
        },
        "automatic_retry": False,
    }
    infra_io.durable_atomic_json(output / "status.json", status_document)

    export_root = evidence_root / "verified-export"
    export = infra_io.controlled_export(output, export_root)
    load_path_counts = collections.Counter(torch_load_paths)
    policy_checkpoint_root = (output / "policy-checkpoints").resolve()
    policy_load_paths = []
    non_policy_load_paths = []
    for path in torch_load_paths:
        resolved = Path(path)
        if resolved.is_relative_to(policy_checkpoint_root):
            policy_load_paths.append(path)
        else:
            non_policy_load_paths.append(path)

    episodes_path = output / "task-confirmation.jsonl"
    episodes = _read_jsonl(episodes_path) if episodes_path.is_file() else []
    route_summary_path = output / "policy-training-summary.json"
    route_summary = _read_json(route_summary_path) if route_summary_path.is_file() else {}
    training_route_path = output / "policy-training-routes.jsonl"
    training_route_rows = _read_jsonl(training_route_path) if training_route_path.is_file() else []
    fixture = matrix.get("e2e_fixture", {})
    expected_steps = fixture.get("policy_training_steps_per_method_seed")
    expected_rollout = fixture.get("policy_rollout_steps")
    expected_updates = fixture.get("policy_optimizer_updates_per_method_seed")
    independent_path = output / "independent-task-metrics.json"
    independent = _read_json(independent_path) if independent_path.is_file() else {}
    status_path = output / "task-pipeline-summary.json"
    pipeline_summary = _read_json(status_path) if status_path.is_file() else {}
    final_status = _read_json(output / "status.json")
    training_world_state_path = output / "frozen-world-model-state.json"
    task_world_state_path = output / "frozen-task-world-model-state.json"
    training_world_state = _read_json(training_world_state_path) if training_world_state_path.is_file() else {}
    task_world_state = _read_json(task_world_state_path) if task_world_state_path.is_file() else {}
    training_world_hashes = _world_model_hashes(training_world_state)
    task_world_hashes = _world_model_hashes(task_world_state)
    expected_world_identities = {("G1", seed) for seed in (8201, 8202, 8203)}
    fixture_world_hashes = {
        ("G1", int(seed)): digest
        for seed, digest in getattr(boundary, "world_model_fixture_state_hashes", {}).items()
    }
    world_hashes_match = (
        set(training_world_hashes) == expected_world_identities
        and set(task_world_hashes) == expected_world_identities
        and set(fixture_world_hashes) == expected_world_identities
        and all(
            training_world_hashes[key].get("initial_state_sha256")
            == task_world_hashes[key].get("initial_state_sha256")
            == fixture_world_hashes[key]
            and training_world_hashes[key].get("unchanged") is True
            and task_world_hashes[key].get("unchanged") is True
            for key in expected_world_identities
        )
    )
    boundary_data = boundary.telemetry() if callable(getattr(boundary, "telemetry", None)) else vars(boundary)
    episode_counts = collections.Counter(
        (row.get("method"), row.get("seed")) for row in episodes
    )
    observed_episodes = {
        (row.get("parent"), row.get("repeat"), row.get("method"), row.get("seed"))
        for row in episodes
    }
    expected_episodes = {
        (parent["parent"], repeat, method, seed)
        for parent in matrix["splits"]["task_confirmation"]
        for repeat in range(matrix["repeats"]["task_confirmation"])
        for method in matrix["methods"]
        for seed in ([None] if method == "H" else matrix["policy_seeds"])
    }
    totals = (settlement or {}).get("ledger", {}).get("totals", {})
    with sqlite3.connect(f"file:{(output / 'budget.sqlite3').as_posix()}?mode=ro", uri=True) as connection:
        training_call_rows = connection.execute(
            "SELECT name, amounts, status FROM calls WHERE stage='conditional_policy_training'",
        ).fetchall()
    replay_ledger: dict[str, dict[str, Any]] = {}
    for name, amounts_json, status in training_call_rows:
        if not str(name).endswith("_recurrent_hidden_replay"):
            continue
        amounts = json.loads(amounts_json)
        item = replay_ledger.setdefault(str(name), {"calls": 0, "encode_sample_evaluations": 0,
                                                     "statuses": []})
        item["calls"] += 1
        item["encode_sample_evaluations"] += int(amounts.get("encode_sample_evaluations", 0))
        item["statuses"].append(str(status))
    route_replay = {
        f"{row.get('method')}:{row.get('seed')}": {
            "groups": row.get("summary", {}).get("recurrent_hidden_replay_groups"),
            "observations": row.get("summary", {}).get("recurrent_hidden_replay_observations"),
            "ppo_update_count": row.get("ppo_update_count"),
            "ledger": replay_ledger.get(
                f"{str(row.get('method')).lower()}_recurrent_hidden_replay",
                {"calls": 0, "encode_sample_evaluations": 0, "statuses": []},
            ),
        } for row in training_route_rows
    }
    replay_by_method = {}
    for method in ("G0", "T", "G1"):
        rows = [row for row in training_route_rows if row.get("method") == method]
        ledger_item = replay_ledger.get(
            f"{method.lower()}_recurrent_hidden_replay",
            {"calls": 0, "encode_sample_evaluations": 0, "statuses": []},
        )
        replay_by_method[method] = {
            "route_count": len(rows),
            "observations": sum(
                int(row.get("summary", {}).get("recurrent_hidden_replay_observations", 0))
                for row in rows
            ),
            # Only the currently active episode is replayed after an update.
            # Resets clear it; total training steps are not replay observations.
            "maximum_observations": sum(
                int(row.get("summary", {}).get("recurrent_hidden_replay_groups", 0))
                * (int(matrix["task_episode_max_steps"]) - 1) for row in rows
            ),
            "ledger": ledger_item,
        }

    def boundary_counter(name: str) -> Any:
        value = getattr(boundary, name, None)
        if value is None and isinstance(boundary_data, dict):
            value = boundary_data.get(name)
        return value

    verification = {
        "pipeline_completed": pipeline_error is None and isinstance(result, dict),
        "nine_policy_routes": route_summary.get("route_count") == 9
        and len(route_summary.get("routes", ())) == 9
        and len(training_route_rows) == 9,
        "reduced_route_budgets": all(
            route.get("training_steps") == expected_steps
            and route.get("optimizer_updates") == expected_updates
            for route in route_summary.get("routes", ())
        ) and all(
            route.get("training_steps") == expected_steps
            and route.get("rollout_steps") == expected_rollout
            and route.get("optimizer_updates") == expected_updates
            for route in training_route_rows
        ),
        "two_rollout_hidden_replay_groups_per_route": (
            len(route_replay) == 9
            and all(item["groups"] == expected_steps // expected_rollout == 2
                    for item in route_replay.values())
        ),
        "hidden_replay_observations_match_ledger_encode_charges": all(
            item["route_count"] == len(matrix["policy_seeds"])
            and 0 <= item["observations"] <= item["maximum_observations"]
            and item["ledger"]["calls"] == item["observations"]
            and item["ledger"]["encode_sample_evaluations"] == item["observations"]
            and all(status == "complete" for status in item["ledger"]["statuses"])
            for item in replay_by_method.values()
        ),
        "route_updates_executed": (
            sum(int(row.get("ppo_update_count", 0)) for row in training_route_rows)
            == 9 * expected_updates
        ),
        "complete_frozen_task_matrix": len(episodes) == 240
        and observed_episodes == expected_episodes
        and len(expected_episodes) == matrix["task_episode_count"],
        "eight_parents_three_repeats_all_methods_and_seeds": (
            len({row.get("parent") for row in episodes}) == 8
            and sorted({row.get("repeat") for row in episodes}) == [0, 1, 2]
            and all(episode_counts[(method, seed)] == 24
                    for method in ("G0", "T", "G1")
                    for seed in matrix["policy_seeds"])
            and episode_counts[("H", None)] == 24
        ),
        "zero_real_environment_constructors": boundary_counter(
            "real_environment_constructor_calls",
        ) == 0,
        "synthetic_environment_used": (boundary_counter("synthetic_environment_returns") or 0) > 0,
        "six_in_memory_world_model_substitutions": (
            boundary_counter("world_model_loader_calls") == 6
            and boundary_counter("world_model_fixture_returns") == 6
            and boundary_counter("world_model_loader_operation_calls") == 0
            and boundary_counter("world_model_device") == "cpu"
            and boundary_counter("cuda_initialized") is False
            and boundary_counter("world_model_fixture_state_consistent") is True
        ),
        "training_and_task_world_model_initial_hashes_match": world_hashes_match,
        "nine_policy_checkpoints_deserialized_only": (
            len(torch_load_paths) == 9 and len(policy_load_paths) == 9
            and not non_policy_load_paths
            and boundary_counter("policy_checkpoint_loads") == 9
        ),
        "real_frozen_checkpoint_deserializations_zero": not any(
            "frozen-models" in Path(path).parts for path in torch_load_paths
        ),
        "ledger_settled_with_expected_work": (
            settlement is not None
            and settlement.get("ledger_settlement_error") is None
            and settlement.get("ledger", {}).get("pending_calls") == 0
            and settlement.get("ledger", {}).get("failed_calls") == 0
            and totals.get("policy_optimizer_updates") == 9 * expected_updates
            and totals.get("checkpoint_writes") == 9
            and totals.get("task_episodes") == 240
            and totals.get("task_comparison_calls") == 1
            and totals.get("world_optimizer_updates", 0) == 0
            and totals.get("world_backward_calls", 0) == 0
        ),
        "independent_recomputation_complete": (
            independent.get("episode_count") == 240
            and independent.get("complete_parent_count") == 8
            and independent.get("complete_matrix") is True
        ),
        "settlement_export_verified": bool(export and export.get("verified"))
        and close_error is None
        and (_read_json(export_root / "resource-settlement.json") == settlement)
        and (_read_json(export_root / "status.json") == status_document),
        "final_status_records_actual_outcome": (
            final_status == status_document
            and final_status.get("status") == (
                "controlled_synthetic_completed" if status_is_completed
                else "controlled_synthetic_technical_stop"
            )
            and final_status.get("pipeline_status") == (
                result.get("status") if isinstance(result, dict) else None
            )
            and final_status.get("ledger_close_error") == close_error
        ),
        "single_cpu_thread_affinity_and_cuda_uninitialized": (
            torch.get_num_threads() == 1
            and torch.get_num_interop_threads() == 1
            and affinity_after is not None and len(affinity_after) == 1
            and not torch.cuda.is_initialized()
        ),
    }
    evidence = {
        "schema": "w1-controlled-synthetic-task-pipeline-evidence/1.0.0",
        "attempt": matrix["attempt"],
        "fixture": matrix.get("e2e_fixture"),
        "execution": {
            "task_pipeline_called": True,
            "pipeline_result": _json_safe(result),
            "pipeline_error": pipeline_error,
            "final_status": status_document,
            "runtime_provenance": {
                "python_executable": sys.executable,
                "python_version": sys.version,
                "cwd": os.getcwd(),
                "module_files": {
                    name: getattr(sys.modules.get(name), "__file__", None)
                    for name in ("runner", "task_pipeline", "runtime_hooks",
                                 "production_policy", "production_data", "torch")
                },
                "torch_version": str(torch.__version__),
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
                "cpu_affinity_before": affinity_before,
                "cpu_affinity_after": affinity_after,
                "cpu_count": os.cpu_count(),
                "cuda_initialized": bool(torch.cuda.is_initialized()),
            },
            "environment_boundary": _json_safe(boundary_data),
            "torch_load_call_count": len(torch_load_paths),
            "torch_load_path_counts": dict(sorted(load_path_counts.items())),
            "policy_checkpoint_deserializations": len(policy_load_paths),
            "non_policy_torch_load_paths": non_policy_load_paths,
            "policy_checkpoint_load_paths": policy_load_paths,
            "real_environment_constructor_calls": getattr(
                boundary, "real_environment_constructor_calls", None,
            ),
            "real_frozen_checkpoint_deserializations": sum(
                "frozen-models" in Path(path).parts for path in torch_load_paths
            ),
        },
        "matrix": {
            "training_parent_count": len(matrix["splits"]["train"]),
            "confirmation_parent_count": len(matrix["splits"]["task_confirmation"]),
            "confirmation_repeats": matrix["repeats"]["task_confirmation"],
            "policy_seeds": matrix["policy_seeds"],
            "methods": matrix["methods"],
            "task_episode_count": matrix["task_episode_count"],
            "task_gates": matrix["task_gates"],
            "training_episode_rows": len(episodes),
            "unique_confirmation_parents": len({row.get("parent") for row in episodes}),
            "unique_confirmation_repeats": sorted({row.get("repeat") for row in episodes}),
            "unique_methods": sorted({row.get("method") for row in episodes}),
            "unique_policy_seeds": sorted({row.get("seed") for row in episodes
                                           if row.get("seed") is not None}),
        },
        "training": {
            "route_count": route_summary.get("route_count"),
            "routes": route_summary.get("routes", []),
            "resource_amounts": route_summary.get("resource_amounts", {}),
        },
        "task_summary": pipeline_summary,
        "independent_metrics": {
            "episode_count": independent.get("episode_count"),
            "expected_episode_count": independent.get("expected_episode_count"),
            "complete_parent_count": independent.get("complete_parent_count"),
            "complete_matrix": independent.get("complete_matrix"),
            "decision_costs_by_method": independent.get("decision_costs", {}).get("by_method"),
        },
        "settlement": _json_safe(settlement),
        "frozen_world_model_states": {
            "training": training_world_state,
            "task_confirmation": task_world_state,
            "fixture_initial_state_hashes": getattr(
                boundary, "world_model_fixture_state_hashes", {},
            ),
            "initial_hashes_match": world_hashes_match,
        },
        "ledger_close_error": close_error,
        "export": _json_safe(export),
        "export_manifest_sha256": infra_io.sha256_file(export_root / "export-manifest.json"),
        "exported_settlement_matches": (
            _read_json(export_root / "resource-settlement.json") == settlement
            if (export_root / "resource-settlement.json").is_file() else False
        ),
        "hidden_replay": route_replay,
        "hidden_replay_by_method": replay_by_method,
        "sqlite_hidden_replay_calls": replay_ledger,
        "verification": {"checks": verification, "all_pass": all(verification.values())},
    }
    evidence_path = evidence_root / "controlled-run-evidence.json"
    evidence_path.write_text(json.dumps(evidence, sort_keys=True, indent=2,
                                        ensure_ascii=False, allow_nan=False) + "\n",
                             encoding="utf-8")
    evidence["evidence_path"] = str(evidence_path)
    evidence_path.write_text(json.dumps(evidence, sort_keys=True, indent=2,
                                        ensure_ascii=False, allow_nan=False) + "\n",
                             encoding="utf-8")
    if archive_root is not None:
        archive_root = archive_root.resolve()
        if archive_root.exists():
            raise FileExistsError(f"Archive destination already exists: {archive_root}")
        archive_root.parent.mkdir(parents=True, exist_ok=True)
        source_manifest = infra_io.tree_manifest(evidence_root)
        shutil.copytree(evidence_root, archive_root)
        if infra_io.tree_manifest(archive_root) != source_manifest:
            raise IOError("ARCHIVE_COPY_HASH_VERIFICATION_FAILED")
        evidence["archived_evidence_path"] = str(archive_root)
    if not all(verification.values()):
        failed = sorted(name for name, passed in verification.items() if not passed)
        raise RuntimeError("CONTROLLED_SYNTHETIC_VERIFICATION_FAILED:" + ",".join(failed))
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("--archive-root", type=Path)
    args = parser.parse_args()
    result = run(args.evidence_root, args.archive_root)
    print(json.dumps({
        "evidence_path": result.get("evidence_path"),
        "archived_evidence_path": result.get("archived_evidence_path"),
        "status": result.get("task_summary", {}).get("status"),
        "settlement_status": result.get("settlement", {}).get("status"),
        "export": result.get("export"),
        "verification": result.get("verification"),
    }, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
