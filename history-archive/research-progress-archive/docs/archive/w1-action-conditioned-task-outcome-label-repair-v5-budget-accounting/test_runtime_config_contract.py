from dataclasses import dataclass
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch

from runtime_config_contract import (
    RuntimeConfigError,
    canonical_sha256,
    validate_config_before_collection,
    write_runtime_environment_record,
)
from production_data import ProductionDataCollector, ProductionDataError

PACKAGE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ConfigFixture:
    task_completion_mode: str = "arrival_to_region"
    deadline_basis: str = "physical_arrival"
    decision_interval: float = 1.0
    arrival_radius: float = 0.0
    completion_notice_mode: str = "single_shot"


class RuntimeConfigContractTests(unittest.TestCase):
    def test_frozen_config_matches_actual_m10config_default_fields(self):
        source = PACKAGE.parents[1] / "runs" / "w1-light-repaired-fair-rerun-v2-nativefs-once" / "native" / "gppo_world" / "m10_environment.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        config_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "M10Config"
        )
        actual_defaults = {
            node.target.id: ast.literal_eval(node.value)
            for node in config_class.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        }
        contract = json.loads((PACKAGE / "environment-config-contract.json").read_text(encoding="utf-8"))
        expected = dict(actual_defaults)
        expected.update(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
        self.assertEqual(contract["config"], expected)
        self.assertEqual(contract["config_sha256"], canonical_sha256(expected))
        self.assertEqual(contract["config_sha256"], "5fe6bd877533edf50e8fefc147629d92d4aa5d618c3ab678ea91e2fe57c5d9b2")

    def test_matching_runtime_config_passes_precollection_gate(self):
        config = ConfigFixture()
        payload = {
            "task_completion_mode": "arrival_to_region",
            "deadline_basis": "physical_arrival",
            "decision_interval": 1.0,
            "arrival_radius": 0.0,
            "completion_notice_mode": "single_shot",
        }
        checked = validate_config_before_collection(
            config, payload, expected_sha256=canonical_sha256(payload),
        )
        self.assertEqual(checked["task_completion_mode"], "arrival_to_region")
        self.assertEqual(checked["deadline_basis"], "physical_arrival")

    def test_mode_or_frozen_digest_mismatch_rejected(self):
        config = ConfigFixture()
        payload = {
            "task_completion_mode": "continuous_service_until_deadline",
            "deadline_basis": "physical_service",
            "decision_interval": 1.0,
        }
        with self.assertRaisesRegex(RuntimeConfigError, "EFFECTIVE_ENVIRONMENT_CONFIG_MISMATCH"):
            validate_config_before_collection(
                config, payload, expected_sha256=canonical_sha256(payload),
            )
        with self.assertRaisesRegex(RuntimeConfigError, "FROZEN_ENVIRONMENT_CONFIG_DIGEST_MISMATCH"):
            validate_config_before_collection(config, vars(config), expected_sha256="0" * 64)

    @unittest.skipIf(os.name == "nt", "durable directory fsync is a Linux-native contract")
    def test_environment_record_is_serialized_from_runtime_object(self):
        config = ConfigFixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "environment.json"
            record = write_runtime_environment_record(path, config)
            reread = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record, reread)
        self.assertEqual(reread["config"], vars(config))
        self.assertEqual(reread["config_sha256"], canonical_sha256(vars(config)))

    def test_collector_checks_config_before_constructing_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "task_completion_mode": "continuous_service_until_deadline",
                "deadline_basis": "physical_service",
                "decision_interval": 1.0,
            }
            collector = ProductionDataCollector.__new__(ProductionDataCollector)
            collector.root = root
            collector.output = root
            collector.matrix = {}
            collector.environment_config_contract = {
                "schema": "w1-environment-config-contract/1.0.0",
                "config": payload,
                "config_sha256": canonical_sha256(payload),
            }
            collector.M10Config = lambda **_values: ConfigFixture()
            collector.environment_config = None
            collector._row_for = lambda *_args: {"scenario": {}}
            collector.scenario_from_dict = lambda value: value
            environment_calls = []
            collector.M10Environment = lambda *_args, **_kwargs: environment_calls.append(True)
            fake_policy = SimpleNamespace(transparent_utility_components=lambda *_args, **_kwargs: (0.0, 0.0))
            with patch.dict(sys.modules, {"production_policy": fake_policy}):
                with self.assertRaisesRegex(ProductionDataError, "EFFECTIVE_ENVIRONMENT_CONFIG_MISMATCH"):
                    collector._collect_unit({"parent": "fixture", "scenario_sha256": "a" * 64,
                                             "exogenous_key": "fixture|repeat-0"}, 0, "train")
            self.assertEqual(environment_calls, [])


if __name__ == "__main__":
    unittest.main()
