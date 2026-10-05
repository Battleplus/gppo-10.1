import copy
import hashlib
import hmac
import json
import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE / "native"))

import production_world  # noqa: E402
from budget_ledger import BudgetLedger  # noqa: E402
from test_sequence_data import collect_fixture  # noqa: E402
from w1_graph_jepa import NODE_COUNTS  # noqa: E402


SYNTHETIC_SCENARIO_SHA256 = "a" * 64


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

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            ledger = BudgetLedger(output / "budget.sqlite3", request)
            boundary = DirectBoundary()
            try:
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
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
