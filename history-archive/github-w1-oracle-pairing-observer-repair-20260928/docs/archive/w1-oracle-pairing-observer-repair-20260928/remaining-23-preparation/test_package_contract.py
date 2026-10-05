from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
import unittest
from pathlib import Path

import classical_baselines
import public_controller
from infra_io import sha256_file
from public_adapter_bridge import validate_runtime_interface


ROOT = Path(__file__).resolve().parent


def portable_path(raw):
    if len(raw) >= 3 and raw[1:3] == ":\\" and Path("/mnt").is_dir():
        return Path("/mnt") / raw[0].lower() / raw[3:].replace("\\", "/")
    return Path(raw)


class PackageContractTests(unittest.TestCase):
    def test_all_indexed_inputs_match(self):
        index = json.loads((ROOT / "input-index.json").read_text(encoding="utf-8"))
        mismatches = []
        for row in index["inputs"]:
            path = portable_path(row["path"])
            actual = sha256_file(path) if path.is_file() else "MISSING"
            if actual != row["sha256"]:
                mismatches.append((row["role"], row["sha256"], actual))
        self.assertEqual(mismatches, [])

    def test_research_runtime_matches_completed_gate(self):
        identity = json.loads((ROOT / "identity-diff.json").read_text(encoding="utf-8"))
        source = portable_path(identity["completed_gate_package"])
        for relative, expected in identity["unchanged_research_runtime"].items():
            self.assertEqual(sha256_file(ROOT / relative), expected, relative)
            self.assertEqual(sha256_file(source / relative), expected, relative)

    def test_real_public_runtime_interface_signatures(self):
        identity = validate_runtime_interface(public_controller, classical_baselines)
        self.assertEqual(identity["signatures"]["prepare"], "(self, observation)")
        self.assertEqual(identity["signatures"]["decide"], "(self, observation, selector)")
        self.assertEqual(
            str(inspect.signature(classical_baselines.ClassicalSelector)),
            "(method, preference=(0.8, 0.2))",
        )

    def test_environment_runtime_imports_without_instantiation_or_models(self):
        sys.path.insert(0, str(ROOT / "native"))
        try:
            from gppo_world.m10_environment import M10Environment, scenario_from_dict
        finally:
            sys.path.pop(0)
        self.assertIn("exogenous_key", str(inspect.signature(M10Environment)))
        self.assertTrue(callable(scenario_from_dict))
        self.assertNotIn("torch", sys.modules)

    def test_request_is_unapproved_remaining_23_only(self):
        request = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
        self.assertEqual(request["status"], "NOT_APPROVED")
        self.assertEqual(request["attempt"], "w1-action-consequence-oracle-remaining-23-v1-once")
        self.assertEqual(request["totals"]["environment_steps"], 10350)
        self.assertEqual(request["totals"]["resets"], 23)
        self.assertEqual(request["totals"]["public_rule_decisions"], 9798)
        self.assertEqual(request["totals"]["branches"], 575)
        self.assertEqual(request["fixed_matrix"]["new_units"], 23)
        self.assertTrue(request["fixed_matrix"]["reused_unit_must_not_run"])
        for name in (
            "model_initializations_or_loads", "encode_forwards", "actor_forwards",
            "world_forwards", "policy_updates", "world_updates", "offline_predictor_updates",
        ):
            self.assertEqual(request["totals"][name], 0)

    def test_external_token_is_hashed_and_not_stored(self):
        contract = json.loads((ROOT / "launch-contract.json").read_text(encoding="utf-8"))
        self.assertFalse(contract["raw_token_stored_in_package"])
        self.assertEqual(contract["attempt"], "w1-action-consequence-oracle-remaining-23-v1-once")
        self.assertEqual(len(contract["external_token_sha256"]), 64)
        int(contract["external_token_sha256"], 16)
        self.assertTrue(contract["one_shot"])
        self.assertFalse(contract["automatic_retry"])
        self.assertFalse(contract["reused_unit_reexecution_authorized"])

    def test_execution_modules_do_not_import_models_or_training(self):
        prohibited = {"torch", "rl_adapters", "joint_gppo", "joint_training", "consequence_model"}
        for name in (
            "accounting.py", "analyze_oracle.py", "communication_observer.py", "reuse_audit.py",
            "runner.py", "infra_io.py", "launch_once.py", "native_launch.py", "supervise.py",
        ):
            tree = ast.parse((ROOT / name).read_text(encoding="utf-8"), name)
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            self.assertTrue(prohibited.isdisjoint(imported), (name, imported))

    def test_stage_call_totals_and_time_reconcile(self):
        request = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
        names = (
            "environment_steps", "resets", "candidate_scans", "public_rule_decisions",
            "snapshot_captures", "branch_snapshot_copies", "branches", "forced_first_actions",
        )
        for name in names:
            observed = sum(stage[name] for stage in request["stages"].values())
            self.assertEqual(observed, request["active_call_totals"][name])
            self.assertEqual(observed, request["totals"][name])
        self.assertEqual(
            sum(stage["wall_seconds"] for stage in request["stages"].values()),
            request["totals"]["wall_seconds"],
        )
        self.assertEqual(
            sum(stage["complete_process_cpu_seconds"] for stage in request["stages"].values()),
            request["totals"]["complete_process_cpu_seconds"],
        )

    def test_reuse_contract_covers_all_copied_evidence(self):
        contract = json.loads((ROOT / "reuse-contract.json").read_text(encoding="utf-8"))
        copied = sorted(path.name for path in (ROOT / "reused-first-unit").iterdir() if path.is_file())
        self.assertEqual(copied, sorted(contract["files"]))
        for relative, expected in contract["files"].items():
            self.assertEqual(sha256_file(ROOT / "reused-first-unit" / relative), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
