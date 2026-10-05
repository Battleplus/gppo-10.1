from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from dependency_probe import probe
from dependency_probe import _validate_remote_runtime_binding
from freeze_contract import verify_remote_runtime_binding_sources
from verify_runtime_inputs import RuntimeInputError, verify_runtime_inputs


ROOT = Path(__file__).resolve().parent


class DependencyClosureTests(unittest.TestCase):
    def test_collector_import_closure_is_staged_and_policy_module_is_not_needed(self):
        result = probe(expected_root=ROOT)
        self.assertIn("production_data", result["project_module_paths"])
        self.assertIn("gppo_world.joint_training", result["project_module_paths"])
        self.assertNotIn("production_policy", result["project_module_paths"])
        self.assertTrue(all(str(ROOT) in path for path in result["project_module_paths"].values()))

    def test_missing_staged_transitive_module_fails_before_environment_construction(self):
        runtime_inputs = ROOT / "runtime-inputs.json"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copy2(runtime_inputs, root / runtime_inputs.name)
            constructed = []
            with self.assertRaisesRegex(RuntimeInputError, "STAGED_MODULE_DIGEST_MISMATCH"):
                verify_runtime_inputs(root)
            self.assertEqual(constructed, [])

    def test_import_error_is_reported_before_worker_can_construct_environment(self):
        script = """
import builtins, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
real = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'gppo_world.joint_consequence_baseline':
        raise ModuleNotFoundError('injected missing transitive module')
    return real(name, *args, **kwargs)
builtins.__import__ = blocked
constructed = []
try:
    from dependency_probe import probe
    probe(expected_root=root)
except ModuleNotFoundError as exc:
    print(json.dumps({'failed_before_environment': not constructed, 'error': str(exc)}))
else:
    raise SystemExit('missing import was not rejected')
"""
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script, str(ROOT)],
            cwd=ROOT, env=environment, text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('"failed_before_environment": true', completed.stdout)

    def test_preloaded_external_gppo_namespace_is_rejected(self):
        foreign = ModuleType("gppo_world")
        foreign.__file__ = "/opt/foreign-project/gppo_world/__init__.py"
        with patch.dict(sys.modules, {"gppo_world": foreign}):
            with self.assertRaisesRegex(RuntimeError, "PROJECT_IMPORT_OUTSIDE_STAGED_ROOT:gppo_world"):
                probe(expected_root=ROOT)

    def test_probe_rejects_preloaded_external_project_module_before_import(self):
        foreign = ModuleType("production_data")
        foreign.__file__ = "/opt/foreign-project/production_data.py"
        with patch.dict(sys.modules, {"production_data": foreign}):
            with self.assertRaisesRegex(RuntimeError, "PROJECT_IMPORT_OUTSIDE_STAGED_ROOT:production_data"):
                probe(expected_root=ROOT)

    def test_preloaded_external_public_controller_is_rejected(self):
        foreign = ModuleType("public_controller")
        foreign.__file__ = "/opt/foreign-project/public_controller.py"
        with patch.dict(sys.modules, {"public_controller": foreign}):
            with self.assertRaisesRegex(RuntimeError, "PROJECT_IMPORT_OUTSIDE_STAGED_ROOT:public_controller"):
                probe(expected_root=ROOT)

    def test_probe_closes_imports_in_isolated_interpreter(self):
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [
                sys.executable, "-I", "-B", str(ROOT / "dependency_probe.py"),
                "--expected-python", "/home/asus/w1-action-relative-auxiliary-runtime-dependency-repair-v1-runtime/bin/python",
                "--expected-root", str(ROOT), "--json",
            ],
            cwd=ROOT, env=environment, text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["torch_version"], "2.8.0+cpu")
        self.assertEqual(result["synthetic_kernel_matmuls"], 1)
        self.assertEqual(result["synthetic_kernel_backwards"], 0)
        self.assertEqual(result["synthetic_optimizer_updates"], 0)
        self.assertEqual(result["model_forwards"], 0)
        self.assertTrue(all(
            path.startswith(str(ROOT)) for path in result["project_module_paths"].values()
        ))

    def test_launcher_preflight_closes_imports_in_isolated_interpreter(self):
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [
                sys.executable, "-I", "-B", str(ROOT / "launcher_preflight_probe.py"),
                "--authorization-windows-path", r"E:\fixture\authorization.json",
                "--expected-python", "/home/asus/w1-action-relative-auxiliary-runtime-dependency-repair-v1-runtime/bin/python",
            ],
            cwd=ROOT, env=environment, text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["dependency_probe"]["torch_version"], "2.8.0+cpu")
        self.assertEqual(result["dependency_probe"]["synthetic_kernel_matmuls"], 1)
        self.assertEqual(result["dependency_probe"]["synthetic_kernel_backwards"], 0)
        self.assertEqual(result["dependency_probe"]["synthetic_optimizer_updates"], 0)
        self.assertEqual(result["dependency_probe"]["model_forwards"], 0)
        self.assertTrue(all(
            path.startswith(str(ROOT))
            for path in result["dependency_probe"]["project_module_paths"].values()
        ))

    def test_launcher_rejects_preloaded_external_dependency_probe(self):
        script = """
import runpy, sys, types
module = types.ModuleType('dependency_probe')
module.__file__ = '/opt/foreign-project/dependency_probe.py'
sys.modules['dependency_probe'] = module
try:
    runpy.run_path(sys.argv[1], run_name='launcher_probe_test')
except RuntimeError as exc:
    print(str(exc))
else:
    raise SystemExit('external dependency_probe was not rejected')
"""
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "-c", script, str(ROOT / "launcher_preflight_probe.py")],
            cwd=ROOT, env=environment, text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("DEPENDENCY_PROBE_PRELOADED_OUTSIDE_PACKAGE", completed.stdout)

    def test_remote_runtime_binding_checks_frozen_metadata_and_local_source_hashes(self):
        result = _validate_remote_runtime_binding(ROOT)
        self.assertEqual(result["scope"], "future-audit-only")
        self.assertFalse(result["remote_execution_during_v6"])
        self.assertFalse(result["training_authorization"])
        checked = verify_remote_runtime_binding_sources(ROOT)
        self.assertEqual(set(checked), {
            "runtime_identity_v2", "delivery_hashes", "final_readonly_preflight",
        })

    def test_runtime_binding_probe_does_not_open_external_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copy2(ROOT / "remote-runtime-audit-binding.json", root / "remote-runtime-audit-binding.json")
            result = _validate_remote_runtime_binding(root)
        self.assertEqual(result["artifact_count"], 3)

    def test_cpu_runtime_source_paths_and_critical_digests_are_verified(self):
        result = probe(expected_python="/home/asus/w1-action-relative-auxiliary-runtime-dependency-repair-v1-runtime/bin/python",
                       expected_root=ROOT)
        identity = result["runtime_identity"]
        self.assertEqual(identity["packages"]["numpy"]["import_path"],
                         "/usr/lib/python3/dist-packages/numpy/__init__.py")
        self.assertTrue(identity["packages"]["torch"]["import_path"].startswith(
            "/home/asus/w1-action-relative-auxiliary-runtime-dependency-repair-v1-runtime/"))
        self.assertIsNone(identity["cuda_build"])


if __name__ == "__main__":
    unittest.main()
