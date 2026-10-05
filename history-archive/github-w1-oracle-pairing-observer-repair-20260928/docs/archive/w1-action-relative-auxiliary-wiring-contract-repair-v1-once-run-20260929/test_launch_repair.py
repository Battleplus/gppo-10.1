from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OLD = Path(r"E:\Z博士\research-plans\w1-action-outcome-prior-prototype-v1")
WSL = "wsl.exe"
DISTRO = "Ubuntu-24.04"
NATIVE_TEST_PARENT = "/home/asus/w1-launch-repair-integration"
EXPORT_TEST_PARENT = Path(r"E:\Z博士\.codex-tmp\w1 launch repair outputs")
RESEARCH_FILES = (
    "budget_ledger.py",
    "experiment_matrix.py",
    "learning.py",
    "learning_schema.py",
    "one_shot_policy.py",
    "outcome_contract.py",
    "pipeline.py",
    "prediction_evaluation.py",
    "predictor.py",
    "prior_adapter.py",
    "runtime_backend.py",
    "transparent_baselines.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def wsl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [WSL, "--distribution", DISTRO, "--exec", *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=check,
    )


def to_wsl(path: Path) -> str:
    result = wsl("/usr/bin/wslpath", "-a", "-u", str(path.resolve()))
    return result.stdout.decode("utf-8").strip()


def exists_wsl(path: str) -> bool:
    return wsl("/usr/bin/test", "-e", path, check=False).returncode == 0


def read_wsl_json(path: str) -> dict:
    result = wsl("/usr/bin/cat", path)
    return json.loads(result.stdout.decode("utf-8"))


def seal_clone(root: Path, *, attempt: str, token: str, fault: str | None = None) -> tuple[str, Path]:
    manifest = json.loads((ROOT / "execution-manifest.json").read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=False)
    for name in sorted(set(manifest["files"]) | {"execution-manifest.json", "hashes.json"}):
        shutil.copy2(ROOT / name, root / name)
    native = f"{NATIVE_TEST_PARENT}/{attempt}"
    export_windows = EXPORT_TEST_PARENT / attempt
    export_wsl = to_wsl(export_windows)
    contract = json.loads((root / "launch-contract.json").read_text(encoding="utf-8"))
    contract.update({
        "attempt": attempt,
        "external_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "windows_source_root": str(root.resolve()),
        "wsl_source_root": to_wsl(root),
        "native_execution_root": native,
        "windows_export_root": export_wsl,
        "native_entry": "integration_native_launch.py",
        "integration_test": True,
        "integration_fault": fault,
    })
    if fault == "child_start":
        contract["native_child_python"] = "/missing/integration-python"
    write_json(root / "launch-contract.json", contract)
    request = json.loads((root / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
    request["attempt"] = attempt
    write_json(root / "RESOURCE_REQUEST.json", request)
    manifest["attempt"] = attempt
    manifest["files"] = {name: sha256(root / name) for name in manifest["files"]}
    write_json(root / "execution-manifest.json", manifest)
    write_json(root / "hashes.json", {
        "schema": "integration-test-seal/1.0.0",
        "execution_manifest_sha256": sha256(root / "execution-manifest.json"),
    })
    token_file = root.parent / f"{attempt}.token"
    token_file.write_text(token, encoding="utf-8")
    return native, token_file


def invoke(root: Path, token_file: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-B", str(root / "launch_once.py"), "--attempt-token-file", str(token_file)],
        cwd=root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )


class StaticContractTests(unittest.TestCase):
    def test_research_implementation_is_byte_identical(self):
        for name in RESEARCH_FILES:
            self.assertEqual(sha256(ROOT / name), sha256(OLD / name), name)

    def test_windows_entry_has_no_posix_staging_primitive(self):
        source = (ROOT / "launch_once.py").read_text(encoding="utf-8")
        self.assertNotIn("stage_sealed_package(", source)
        self.assertNotIn("controlled_export(", source)
        self.assertNotIn("native_execution_root]).exists", source)
        self.assertIn("shell=False", source)
        self.assertIn("stdin=subprocess.PIPE", source)

    def test_no_environment_or_model_import_in_integration_worker(self):
        source = (ROOT / "integration_fake_worker.py").read_text(encoding="utf-8")
        for forbidden in ("torch", "runtime_backend", "m10_environment", "checkpoint", "train_seed"):
            self.assertNotIn(forbidden, source)

    def test_cross_process_accounting_is_checked_and_exported(self):
        linux = (ROOT / "wsl_stage_and_launch.py").read_text(encoding="utf-8")
        infra = (ROOT / "infra_io.py").read_text(encoding="utf-8")
        self.assertIn("combined_staging_wall", linux)
        self.assertIn("combined_staging_cpu", linux)
        self.assertIn('supervisor_evidence["cpu_seconds"]', linux)
        self.assertIn("combined_budget_cpu", linux)
        self.assertIn("append_verified_export_file(", linux)
        self.assertIn("def append_verified_export_file(", infra)


@unittest.skipUnless(os.name == "nt", "real Windows-to-WSL integration requires Windows")
class RealEntryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = Path(tempfile.mkdtemp(prefix="w1-launch-repair-入口 中文 '单' 空格-", dir=r"E:\Z博士\.codex-tmp"))
        EXPORT_TEST_PARENT.mkdir(parents=True, exist_ok=True)
        wsl("/usr/bin/mkdir", "-p", NATIVE_TEST_PARENT)
        cls.records: dict[str, dict] = {}

    def make_case(self, label: str, fault: str | None = None):
        suffix = uuid.uuid4().hex[:8]
        attempt = f"w1-launch-repair-it-{label}-{suffix}"
        root = self.temp / f"包 {label} 'quoted'"
        token = uuid.uuid4().hex + uuid.uuid4().hex
        native, token_file = seal_clone(root, attempt=attempt, token=token, fault=fault)
        record = {"attempt": attempt, "root": root, "token": token, "token_file": token_file, "native": native,
                  "export": EXPORT_TEST_PARENT / attempt}
        self.records[label] = record
        return record

    def test_01_wrong_token_rejected_before_wsl_staging(self):
        case = self.make_case("wrong-token")
        case["token_file"].write_text("wrong", encoding="utf-8")
        result = invoke(case["root"], case["token_file"])
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(exists_wsl(case["native"]))
        self.assertFalse(case["export"].exists())
        failure = json.loads(result.stderr.decode("utf-8").strip().splitlines()[-1])
        self.assertEqual(failure["stage"], "windows_preflight")
        self.assertFalse(failure["staging_started"])
        self.assertFalse(failure["worker_started"])
        self.assertGreaterEqual(failure["windows_wall_seconds"], 0.0)
        self.assertGreaterEqual(failure["windows_cpu_seconds"], 0.0)

    def test_02_success_repeat_guard_unicode_and_hash_export(self):
        case = self.make_case("success")
        result = invoke(case["root"], case["token_file"])
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        self.assertTrue(exists_wsl(case["native"]))
        self.assertFalse(Path(r"E:\home\asus\w1-launch-repair-integration") .exists())
        complete = json.loads((case["export"] / "EXPORT_COMPLETE.json").read_text(encoding="utf-8"))
        self.assertTrue(complete["verified"])
        self.assertEqual(complete["final_settlement_included"], "export-status.json")
        exported_settlement = json.loads(
            (case["export"] / "export-status.json").read_text(encoding="utf-8")
        )
        self.assertEqual(exported_settlement["status"], "verified")
        self.assertGreaterEqual(exported_settlement["combined_budget_wall_seconds"], 0.0)
        self.assertGreaterEqual(exported_settlement["combined_budget_cpu_seconds"], 0.0)
        self.assertGreaterEqual(exported_settlement["supervisor_complete_process_cpu_seconds"], 0.0)
        supervisor = read_wsl_json(f"{case['native']}/supervisor-status.json")
        self.assertEqual(supervisor["status"], "complete")
        worker = json.loads((case["export"] / "run-once" / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(worker["worker_invocations"], 1)
        self.assertEqual((worker["environment_calls"], worker["model_calls"], worker["training_updates"]), (0, 0, 0))
        self.assertEqual(worker["sqlite_pending"], 0)
        self.assertEqual(worker["sqlite_completed"], 12)
        self.assertGreater(worker["concurrent_reads"], 0)
        second = invoke(case["root"], case["token_file"])
        self.assertNotEqual(second.returncode, 0)
        worker_after = read_wsl_json(f"{case['native']}/run-once/status.json")
        self.assertEqual(worker_after["worker_invocations"], 1)

    def test_03_staging_fsync_failure_is_recorded_before_worker(self):
        case = self.make_case("staging-fsync", "staging_fsync")
        result = invoke(case["root"], case["token_file"])
        self.assertNotEqual(result.returncode, 0)
        fallback = f"{NATIVE_TEST_PARENT}/.{case['attempt']}.launcher-failure.json"
        state = read_wsl_json(fallback)
        self.assertEqual(state["stage"], "staging")
        self.assertFalse(state["worker_started"])
        self.assertFalse(exists_wsl(f"{case['native']}/supervisor-status.json"))

    def test_04_child_start_failure_is_settled(self):
        case = self.make_case("child-start", "child_start")
        result = invoke(case["root"], case["token_file"])
        self.assertNotEqual(result.returncode, 0)
        settlement = read_wsl_json(f"{case['native']}/infrastructure-settlement.json")
        self.assertIn("FileNotFoundError", settlement["launch_error"])
        self.assertFalse(settlement["worker_relaunched"])

    def test_05_export_failure_preserves_native_without_restart(self):
        case = self.make_case("export", "export")
        result = invoke(case["root"], case["token_file"])
        self.assertNotEqual(result.returncode, 0)
        state = read_wsl_json(f"{case['native']}/export-status.json")
        self.assertEqual(state["status"], "failed")
        self.assertFalse(state["worker_relaunched"])
        self.assertFalse(case["export"].exists())
        worker = read_wsl_json(f"{case['native']}/run-once/status.json")
        self.assertEqual(worker["worker_invocations"], 1)

    def test_06_native_interruption_does_not_restart(self):
        case = self.make_case("interrupt", "native_interrupt")
        result = invoke(case["root"], case["token_file"])
        self.assertNotEqual(result.returncode, 0)
        settlement = read_wsl_json(f"{case['native']}/infrastructure-settlement.json")
        self.assertIn("injected Linux launcher interruption", settlement["launcher_interruption"])
        self.assertFalse(settlement["worker_relaunched"])
        supervisor = read_wsl_json(f"{case['native']}/supervisor-status.json")
        self.assertEqual(supervisor["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
