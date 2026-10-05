import copy
import hashlib
import hmac
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import traceback
import unittest
from pathlib import Path
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE / "native"))

import production_world  # noqa: E402
from budget_ledger import BudgetLedger  # noqa: E402
from infra_io import controlled_export, durable_atomic_json  # noqa: E402
from test_sequence_data import collect_fixture  # noqa: E402
from w1_graph_jepa import NODE_COUNTS  # noqa: E402


SYNTHETIC_SCENARIO_SHA256 = "a" * 64


def _traceback_text(exc):
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _ledger_counters(path):
    if not path.is_file():
        return None
    connection = None
    try:
        connection = sqlite3.connect(path)
        rows = connection.execute("SELECT amounts,status FROM calls ORDER BY id").fetchall()
        totals, statuses = {}, {}
        for amounts_json, status in rows:
            statuses[status] = statuses.get(status, 0) + 1
            for name, amount in json.loads(amounts_json).items():
                totals[name] = totals.get(name, 0) + int(amount)
        return {"available": True, "totals": totals, "call_status_counts": statuses}
    except BaseException as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if connection is not None:
            connection.close()


def _install_training_progress_capture(output):
    models, routes = {}, {}

    def persist():
        durable_atomic_json(output / "world-model-training-progress.json", {
            "schema": "w1-test-world-model-training-progress/1.0.0",
            "routes": [routes[key] for key in sorted(routes)],
        })

    original_model_for = production_world._model_for
    original_train = production_world.train_world_model
    original_selection = production_world._selection_metrics
    original_save = production_world._save_checkpoint
    original_load = production_world._load_checkpoint

    def capture_model_for(config, seed, device, ledger, boundary):
        model = original_model_for(config, seed, device, ledger, boundary)
        models[id(model)] = {"seed": int(seed)}
        return model

    def capture_train(model, batches, **kwargs):
        losses = original_train(model, batches, **kwargs)
        route = models[id(model)]
        route["variant"] = "G2" if kwargs["event_enabled"] else "G1"
        key = f"{route['variant']}:{route['seed']}"
        progress = routes.setdefault(key, {"variant": route["variant"], "seed": route["seed"],
            "loss_by_epoch": [], "selection_metrics_by_epoch": []})
        progress["loss_by_epoch"].append({"epoch": len(progress["loss_by_epoch"]) + 1, **losses})
        persist()
        return losses

    def capture_selection(model, windows, device, expected_parent_count):
        metrics = original_selection(model, windows, device, expected_parent_count)
        route = models[id(model)]
        key = f"{route['variant']}:{route['seed']}"
        progress = routes[key]
        progress["selection_metrics_by_epoch"].append({
            "epoch": len(progress["loss_by_epoch"]),
            "selection_regret": metrics[0],
            "selection_outcome_mae": metrics[1],
        })
        persist()
        return metrics

    def capture_save(payload, path):
        result = original_save(payload, path)
        metadata = payload["metadata"]
        key = f"{metadata['variant']}:{metadata['seed']}"
        progress = routes.setdefault(key, {"variant": metadata["variant"],
            "seed": int(metadata["seed"]), "loss_by_epoch": [],
            "selection_metrics_by_epoch": []})
        progress.setdefault("checkpoint_writes", []).append({
            "path": Path(path).relative_to(output).as_posix(),
            "architecture": metadata["architecture"],
            "event_loss_enabled": metadata["event_loss_enabled"],
            "selection_regret": metadata["selection_regret"],
            "selection_outcome_mae_tiebreak": metadata["selection_outcome_mae_tiebreak"],
        })
        persist()
        return result

    def capture_load(path, device):
        payload = original_load(path, device)
        metadata = payload["metadata"]
        key = f"{metadata['variant']}:{metadata['seed']}"
        progress = routes.setdefault(key, {"variant": metadata["variant"],
            "seed": int(metadata["seed"]), "loss_by_epoch": [],
            "selection_metrics_by_epoch": []})
        progress.setdefault("checkpoint_restores", []).append({
            "path": Path(path).relative_to(output).as_posix(),
            "event_loss_enabled": metadata["event_loss_enabled"],
            "device": device,
        })
        persist()
        return payload

    patchers = [
        patch.object(production_world, "_model_for", side_effect=capture_model_for),
        patch.object(production_world, "train_world_model", side_effect=capture_train),
        patch.object(production_world, "_selection_metrics", side_effect=capture_selection),
        patch.object(production_world, "_save_checkpoint", side_effect=capture_save),
        patch.object(production_world, "_load_checkpoint", side_effect=capture_load),
    ]
    for item in patchers:
        item.start()
    return lambda: [item.stop() for item in reversed(patchers)]


class DirectBoundary:
    def __init__(self):
        config = {
            "schema": production_world._SYNTHETIC_CONTEXT_SCHEMA,
            "training_window_count": 1,
            "selection_window_count": 1,
            "confirmation_window_count": 8,
            "maximum_epochs": 1,
            "maximum_train_candidate_rows": 25,
        }
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.integration_context = {
            "mode": "synthetic_test",
            "integration_config": config,
            "integration_config_sha256": hashlib.sha256(canonical).hexdigest(),
            "authentication": hmac.new(
                production_world._SYNTHETIC_AUTH_KEY, canonical, hashlib.sha256,
            ).hexdigest(),
        }

    @staticmethod
    def _run(operation, *args, **kwargs):
        return operation(*args, **kwargs)

    model = _run
    optimizer = _run
    checkpoint = _run
    backward = _run


def _synthetic_matrix(candidate_count):
    matrix = json.loads((PACKAGE / "experiment-matrix.json").read_text(encoding="utf-8"))
    split_parents = {
        "train": ["synthetic-train"],
        "model_selection": ["synthetic-selection"],
        "prediction_confirmation": [f"synthetic-confirm-{index}" for index in range(8)],
    }
    matrix["splits"] = {
        split: [{"parent": parent, "scenario_sha256": SYNTHETIC_SCENARIO_SHA256}
                for parent in parents]
        for split, parents in split_parents.items()
    }
    matrix["repeats"] = {split: 1 for split in split_parents}
    matrix["world_model_device"] = "cpu"
    matrix["world_model_training_windows"] = 1
    matrix["world_model_selection_windows"] = 1
    matrix["world_model_confirmation_windows"] = 8
    matrix["world_model_maximum_epochs"] = 1
    matrix["world_model_maximum_train_candidate_rows"] = 25
    matrix["world_model_maximum_updates_per_variant_seed"] = 1
    if candidate_count > matrix["world_model_maximum_train_candidate_rows"]:
        raise AssertionError("synthetic fixture exceeds its authenticated candidate cap")
    return matrix


def _synthetic_records(template, split, parents):
    records = []
    for parent in parents:
        row = copy.deepcopy(template)
        row.pop("_test_environment_record", None)
        row.pop("_test_environment_construction_events", None)
        row.update({
            "split": split,
            "parent": parent,
            "repeat": 0,
            "window_id": f"{split}:{parent}:repeat-0",
            "scenario_sha256": SYNTHETIC_SCENARIO_SHA256,
            "source_exogenous_key": f"synthetic-source|{parent}|repeat-0",
            "runtime_exogenous_key": f"synthetic-runtime|{parent}|repeat-0",
        })
        records.append(row)
    return records


def _shape_fixture_graph_for_model(window):
    # The collector unit-test double uses compact uppercase 4-wide nodes.
    source_current, source_targets = window["current_nodes"], window["target_nodes"]

    def pad(rows):
        return [list(row) + [0.0] * (32 - len(row)) for row in rows]

    window["current_nodes"] = {
        "uav": pad(source_current["UAV"]),
        "region": [[0.0] * 32 for _ in range(NODE_COUNTS["region"])],
        "target": [[0.0] * 32 for _ in range(NODE_COUNTS["target"])],
        "task": pad(source_current["Task"]),
        "event": [[0.0] * 32 for _ in range(NODE_COUNTS["event"])],
    }
    candidate_count = len(window["candidate_ids"])
    zeros = {
        name: [[0.0] * 32 for _ in range(NODE_COUNTS[name])]
        for name in ("region", "target", "event")
    }
    window["target_nodes"] = {
        "uav": [pad(candidate) for candidate in source_targets["UAV"]],
        "region": [copy.deepcopy(zeros["region"]) for _ in range(candidate_count)],
        "target": [copy.deepcopy(zeros["target"]) for _ in range(candidate_count)],
        "task": [pad(candidate) for candidate in source_targets["Task"]],
        "event": [copy.deepcopy(zeros["event"]) for _ in range(candidate_count)],
    }


class SequenceWorldIntegrationTests(unittest.TestCase):
    def test_synthetic_cpu_train_checkpoint_restore_and_confirmation_accounting(self):
        evidence_value = os.environ.get("W1_TEST_EVIDENCE_DIR")
        evidence = Path(evidence_value) if evidence_value else None
        failure = failure_tb = None
        failure_traceback = None
        artifact_error = export_error = None
        output = export_destination = None
        ledger = None
        training_capture_cleanup = None
        trained = predicted = settled = export_result = None
        partial_snapshot = None
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            output = base / "sequence-run"
            export_destination = (evidence / "sequence-world-controlled-export"
                                  if evidence is not None else base / "verified-export")
            try:
                _summary, template, _collector_calls = collect_fixture({"deadline": 6.0, "required_service": 1.0})
                _shape_fixture_graph_for_model(template)
                matrix = _synthetic_matrix(len(template["candidate_ids"]))
                train_rows = _synthetic_records(template, "train", ["synthetic-train"])
                selection_rows = _synthetic_records(template, "model_selection", ["synthetic-selection"])
                confirmation_rows = _synthetic_records(
                    template, "prediction_confirmation",
                    [row["parent"] for row in matrix["splits"]["prediction_confirmation"]],
                )
                request = {"stages": {
                    "world_model_training_and_selection": {
                        "model_initializations_or_loads": 6,
                        "world_optimizer_updates": 6,
                        "world_batch_forwards": 18,
                        "world_sample_evaluations": 90,
                        "world_backward_calls": 6,
                        "checkpoint_writes": 6,
                    },
                    "prediction_confirmation": {
                        "model_initializations_or_loads": 6,
                        "checkpoint_loads": 6,
                        "world_batch_forwards": 48,
                        "world_sample_evaluations": 240,
                    },
                }, "totals": {}}
                ledger = BudgetLedger(output / "budget.sqlite3", request)
                boundary = DirectBoundary()
                training_capture_cleanup = _install_training_progress_capture(output)
                trained = production_world.train_select_world_models(
                    train_rows, selection_rows, matrix, output, ledger, boundary,
                )
                self.assertEqual(len(trained.routes), 6)
                self.assertEqual({route["seed"] for route in trained.routes}, {8201, 8202, 8203})
                self.assertTrue(all(route["device"] == "cpu" for route in trained.routes))
                self.assertTrue(all(
                    route["checkpoint_metadata"]["architecture"] == matrix["world_model_architecture"]
                    and route["event_loss_enabled"] == (route["variant"] == "G2")
                    for route in trained.routes
                ))

                predicted = production_world.evaluate_world_models(
                    confirmation_rows, trained.routes, matrix, output, ledger, boundary,
                )
                recalculated = production_world.recompute_prediction_metrics(
                    predicted.trace_path, expected_parent_count=8, expected_seeds=(8201, 8202, 8203),
                )
                self.assertEqual(predicted.metrics, recalculated)
                self.assertTrue(recalculated["coverage_pass"])
                self.assertFalse(recalculated["gppo_executed"])
                for model in ("G1:ensemble", "G2:ensemble"):
                    self.assertEqual(set(recalculated["horizon_task_outcome_brier_by_head_parent_macro"][model]),
                                     {"physical_on_time_completion", "task_expired", "host_confirmation"})
                    self.assertTrue(all(value is not None for value in
                                        recalculated["horizon_task_outcome_brier_by_head_parent_macro"][model].values()))
                trace_rows = [json.loads(line) for line in predicted.trace_path.read_text(encoding="utf-8").splitlines()]
                noop_rows = [candidate for window in trace_rows for candidate in window["candidate_rows"]
                             if candidate["action_id"] == 24]
                self.assertEqual(len(noop_rows), 8)
                self.assertTrue(all(candidate["horizon_task_outcome_target"] == [None, None, None]
                                    and candidate["horizon_task_outcome_valid"] == [False, False, False]
                                    for candidate in noop_rows))

                settled = ledger.assert_settled()
                totals = settled["totals"]
                self.assertEqual(settled["pending_calls"], 0)
                self.assertEqual(settled["failed_calls"], 0)
                self.assertEqual(totals["model_initializations_or_loads"], 12)
                self.assertEqual(totals["world_optimizer_updates"], 6)
                self.assertEqual(totals["world_backward_calls"], 6)
                self.assertEqual(totals["checkpoint_writes"], 6)
                self.assertEqual(totals["checkpoint_loads"], 6)
                self.assertEqual(totals["world_batch_forwards"], 66)
                self.assertEqual(totals["world_sample_evaluations"], 330)
            except BaseException as exc:
                failure = exc
                failure_tb = exc.__traceback__
                failure_traceback = _traceback_text(exc)
            finally:
                if training_capture_cleanup is not None:
                    try:
                        training_capture_cleanup()
                    except BaseException as exc:
                        if failure is None:
                            failure, failure_tb = exc, exc.__traceback__
                            failure_traceback = _traceback_text(exc)
                if ledger is not None:
                    try:
                        partial_snapshot = ledger.snapshot()
                    except BaseException as exc:
                        if failure is None:
                            failure, failure_tb = exc, exc.__traceback__
                            failure_traceback = _traceback_text(exc)
                    try:
                        ledger.close()
                    except BaseException as exc:
                        export_error = _traceback_text(exc)
                        if failure is None:
                            failure, failure_tb = exc, exc.__traceback__
                            failure_traceback = export_error
                if output is not None and output.is_dir():
                    try:
                        if partial_snapshot is not None:
                            durable_atomic_json(output / "ledger-summary.json", {
                                "snapshot_before_close": partial_snapshot,
                                "call_rows": _ledger_counters(output / "budget.sqlite3"),
                            })
                        export_destination.parent.mkdir(parents=True, exist_ok=True)
                        export_result = controlled_export(output, export_destination)
                        if not export_result.get("verified"):
                            raise AssertionError("sequence CPU controlled export was not verified")
                    except BaseException as exc:
                        export_error = _traceback_text(exc)
                        if failure is None:
                            failure, failure_tb = exc, exc.__traceback__
                            failure_traceback = export_error
                if evidence is not None:
                    try:
                        evidence.mkdir(parents=True, exist_ok=True)
                        if output is not None and not (
                                evidence / "sequence-world-controlled-export" / "EXPORT_COMPLETE.json").is_file():
                            shutil.copytree(output, evidence / "sequence-world-run", dirs_exist_ok=True)
                            for source_name, target_name in (
                                    ("prediction-trace.jsonl", "sequence-world-trace.jsonl"),
                                    ("prediction-metrics.json", "sequence-world-metrics.json")):
                                source = output / source_name
                                if source.is_file():
                                    shutil.copy2(source, evidence / target_name)
                        if (export_destination is not None and export_destination.is_dir()
                                and export_destination.resolve() !=
                                (evidence / "sequence-world-controlled-export").resolve()):
                            shutil.copytree(export_destination,
                                evidence / "sequence-world-controlled-export", dirs_exist_ok=True)
                    except BaseException as exc:
                        artifact_error = _traceback_text(exc)
                        if failure is None:
                            failure, failure_tb = exc, exc.__traceback__
                            failure_traceback = artifact_error
                    try:
                        durable_atomic_json(evidence / "sequence-world-test-diagnostics.json", {
                            "schema": "w1-sequence-world-test-diagnostics/1.0.0",
                            "status": "technical_stop" if failure is not None else "complete",
                            "profile": {"device": "cpu", "training_windows": 1,
                                "selection_windows": 1, "confirmation_windows": 8,
                                "seeds": 3, "maximum_epochs": 1,
                                "maximum_train_candidate_rows": 25},
                            "first_original_traceback": failure_traceback,
                            "export_error": export_error,
                            "artifact_persistence_error": artifact_error,
                            "partial_compute_counters": {
                                "ledger_snapshot": partial_snapshot,
                                "ledger_rows": _ledger_counters(output / "budget.sqlite3")
                                    if output is not None else None,
                            },
                            "actual_routes": None if trained is None else [
                                {key: route.get(key) for key in (
                                    "variant", "seed", "event_loss_enabled", "epochs_trained",
                                    "updates", "selection_regret", "selection_outcome_mae_tiebreak",
                                    "loss_by_epoch", "checkpoint_relative_path", "checkpoint_sha256")}
                                for route in trained.routes
                            ],
                            "artifacts": {
                                "training_summary": bool(output is not None and
                                    (output / "world-model-training-summary.json").is_file()),
                                "training_progress": bool(output is not None and
                                    (output / "world-model-training-progress.json").is_file()),
                                "checkpoints": bool(output is not None and
                                    (output / "world-model-checkpoints").is_dir()),
                                "prediction_trace": bool(output is not None and
                                    (output / "prediction-trace.jsonl").is_file()),
                                "prediction_metrics": bool(output is not None and
                                    (output / "prediction-metrics.json").is_file()),
                                "ledger": bool(output is not None and
                                    (output / "budget.sqlite3").is_file()),
                                "controlled_export": bool(export_destination is not None and
                                    export_destination.is_dir()),
                            },
                        })
                    except BaseException as exc:
                        if failure is None:
                            failure, failure_tb = exc, exc.__traceback__
                            failure_traceback = _traceback_text(exc)
            if failure is not None:
                raise failure.with_traceback(failure_tb)


if __name__ == "__main__":
    unittest.main()
