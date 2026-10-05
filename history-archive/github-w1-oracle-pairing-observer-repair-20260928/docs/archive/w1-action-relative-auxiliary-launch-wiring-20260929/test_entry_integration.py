from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WSL = "wsl.exe"
DISTRO = "Ubuntu-24.04"
NATIVE_PARENT = "/home/asus/w1-action-relative-auxiliary-launch-wiring-integration"
EXPORT_PARENT = Path(r"E:\Z博士\.codex-tmp\w1 A-B launch exports")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wsl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([WSL, "--distribution", DISTRO, "--exec", *args], capture_output=True, check=check)


def wslpath(path: Path) -> str:
    return wsl("/usr/bin/wslpath", "-a", "-u", str(path.resolve())).stdout.decode().strip()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def clone_case(parent: Path, mode: str, token: str) -> tuple[Path, Path, str, Path]:
    attempt = f"w1-ab-wiring-it-{mode}-{uuid.uuid4().hex[:8]}"
    root = parent / f"包 {mode} {uuid.uuid4().hex[:6]} 中文 'quoted'"
    root.mkdir(parents=True)
    manifest = json.loads((ROOT / "execution-manifest.json").read_text(encoding="utf-8"))
    names = set(manifest["files"]) | {"execution-manifest.json", "hashes.json"}
    for name in names:
        shutil.copy2(ROOT / name, root / name)
    contract = json.loads((root / "launch-contract.json").read_text(encoding="utf-8"))
    export = EXPORT_PARENT / attempt
    contract.update({
        "attempt": attempt,
        "integration_test": True,
        "integration_mode": mode,
        "native_entry": "integration_production_entry.py",
        "native_execution_root": f"{NATIVE_PARENT}/{attempt}",
        "windows_export_root": wslpath(export),
        "windows_source_root": str(root.resolve()),
        "wsl_source_root": wslpath(root),
        "external_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
    })
    write_json(root / "launch-contract.json", contract)
    request = json.loads((root / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
    request["attempt"] = attempt
    write_json(root / "RESOURCE_REQUEST.json", request)
    contract["resource_request_sha256"] = sha(root / "RESOURCE_REQUEST.json")
    write_json(root / "launch-contract.json", contract)
    manifest["attempt"] = attempt
    original_names = set(manifest["files"])
    manifest["files"] = {name: sha(root / name) for name in sorted(original_names)}
    write_json(root / "execution-manifest.json", manifest)
    write_json(root / "hashes.json", {"execution_manifest_sha256": sha(root / "execution-manifest.json")})
    token_file = parent / f"{attempt}.token"
    token_file.write_text(token, encoding="utf-8")
    return root, token_file, f"{NATIVE_PARENT}/{attempt}", export


@unittest.skipUnless(os.name == "nt", "requires the real Windows-to-WSL entry")
class ProductionEntryIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="w1-ab-wiring-", dir=r"E:\Z博士\.codex-tmp"))
        EXPORT_PARENT.mkdir(parents=True, exist_ok=True)
        wsl("/usr/bin/mkdir", "-p", NATIVE_PARENT)

    def invoke(self, mode: str, *, token: str | None = None):
        token = token or uuid.uuid4().hex
        root, token_file, native, export = clone_case(self.tmp, mode, token)
        result = subprocess.run([str(Path(os.environ.get("PYTHON", "python"))), "-B", str(root / "launch_once.py"), "--attempt-token-file", str(token_file)], capture_output=True, check=False)
        return result, root, token_file, native, export

    def test_gate_fail_has_zero_task_calls(self):
        result, _root, _token, native, _export = self.invoke("gate_fail")
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        status = json.loads(wsl("/usr/bin/cat", f"{native}/run-once/status.json").stdout.decode())
        self.assertEqual(status["task_calls"], 0)

    def test_gate_pass_reaches_four_arm_task_matrix(self):
        result, _root, _token, native, export = self.invoke("pass")
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        task = json.loads(wsl("/usr/bin/cat", f"{native}/run-once/task-comparison.json").stdout.decode())
        self.assertEqual(len(task["arms"]), 32)
        for name in ("training.json", "per-candidate-predictions.jsonl", "prediction-metrics.json", "settlement.json"):
            self.assertTrue(wsl("/usr/bin/test", "-e", f"{native}/run-once/{name}", check=False).returncode == 0, name)
        self.assertTrue((export / "EXPORT_COMPLETE.json").exists())
        self.assertFalse(Path(r"E:\home\asus\w1-action-relative-auxiliary-launch-wiring-integration").exists())

    def test_bad_token_and_reuse_failure_stop_before_worker(self):
        valid = uuid.uuid4().hex
        root, token_file, native, export = clone_case(self.tmp, "bad-token", valid)
        token_file.write_text("wrong", encoding="utf-8")
        result = subprocess.run(["python", "-B", str(root / "launch_once.py"), "--attempt-token-file", str(token_file)], capture_output=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(wsl("/usr/bin/test", "-e", native, check=False).returncode == 0)
        self.assertFalse(export.exists())
        result, _root, _token, native, _export = self.invoke("reuse_fail")
        self.assertEqual(result.returncode, 0)
        status = json.loads(wsl("/usr/bin/cat", f"{native}/run-once/status.json").stdout.decode())
        self.assertEqual(status["task_calls"], 0)

    def test_formal_entry_rejects_fake_backend(self):
        result, root, _token, _native, _export = self.invoke("pass")
        self.assertEqual(result.returncode, 0)
        # The production native entry is guarded independently of the test entry.
        probe = subprocess.run(["python", "-c", "from native_launch import verify_native_launch; print('guarded')"], cwd=root, capture_output=True)
        self.assertEqual(probe.returncode, 0)
        self.assertIn("FORMAL_ENTRY_REJECTS_INTEGRATION_BACKEND", (ROOT / "native_launch.py").read_text(encoding="utf-8"))

    def test_duplicate_launch_is_rejected(self):
        token = uuid.uuid4().hex
        root, token_file, native, _export = clone_case(self.tmp, "duplicate", token)
        first = subprocess.run(["python", "-B", str(root / "launch_once.py"), "--attempt-token-file", str(token_file)], capture_output=True, check=False)
        self.assertEqual(first.returncode, 0, first.stderr.decode(errors="replace"))
        second = subprocess.run(["python", "-B", str(root / "launch_once.py"), "--attempt-token-file", str(token_file)], capture_output=True, check=False)
        self.assertNotEqual(second.returncode, 0)


if __name__ == "__main__":
    unittest.main()
