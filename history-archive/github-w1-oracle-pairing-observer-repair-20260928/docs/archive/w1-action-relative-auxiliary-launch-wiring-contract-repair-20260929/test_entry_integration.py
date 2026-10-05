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

from freeze_wiring import freeze_identity
from manifest_contract import PackageContractError, verify_package

ROOT = Path(__file__).resolve().parent
WSL = "wsl.exe"
DISTRO = "Ubuntu-24.04"
NATIVE_PARENT = "/home/asus/w1-ab-contract-repair-integration"
EXPORT_PARENT = Path(r"E:\Z博士\.codex-tmp\w1 contract repair exports")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wsl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([WSL, "--distribution", DISTRO, "--exec", *args], capture_output=True, check=check)


def wslpath(path: Path) -> str:
    return wsl("/usr/bin/wslpath", "-a", "-u", str(path.resolve())).stdout.decode().strip()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def clone_case(parent: Path, mode: str, token: str, *, formal_entry: bool = False):
    attempt = f"w1-ab-contract-it-{mode}-{uuid.uuid4().hex[:8]}"
    root = parent / f"包 {mode} {uuid.uuid4().hex[:6]} 中文 'quoted'"
    root.mkdir(parents=True)
    manifest = json.loads((ROOT / "execution-manifest.json").read_text(encoding="utf-8"))
    for name in set(manifest["files"]) | {"execution-manifest.json", "hashes.json"}:
        shutil.copy2(ROOT / name, root / name)
    contract = json.loads((root / "launch-contract.json").read_text(encoding="utf-8"))
    export = EXPORT_PARENT / attempt
    contract.update({
        "attempt": attempt,
        "integration_test": True,
        "integration_mode": mode,
        "native_entry": "native_launch.py" if formal_entry else "integration_production_entry.py",
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
    freeze_identity(root, attempt)
    authorization = parent / f"{attempt}.authorization.json"
    write_json(authorization, {
        "schema": "w1-external-launch-authorization/2.0.0",
        "attempt": attempt,
        "execution_manifest_sha256": sha(root / "execution-manifest.json"),
        "hashes_sha256": sha(root / "hashes.json"),
        "resource_request_sha256": sha(root / "RESOURCE_REQUEST.json"),
        "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
    })
    token_file = parent / f"{attempt}.token"
    token_file.write_text(token, encoding="utf-8")
    return root, token_file, authorization, f"{NATIVE_PARENT}/{attempt}", export


@unittest.skipUnless(os.name == "nt", "requires real Windows-to-WSL entry")
class ProductionEntryIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="w1-contract-repair-", dir=r"E:\Z博士\.codex-tmp"))
        EXPORT_PARENT.mkdir(parents=True, exist_ok=True)
        wsl("/usr/bin/mkdir", "-p", NATIVE_PARENT)

    def invoke(self, mode: str, *, formal_entry: bool = False):
        root, token_file, authorization, native, export = clone_case(self.tmp, mode, uuid.uuid4().hex, formal_entry=formal_entry)
        command = ["python", "-B", str(root / "launch_once.py"), "--attempt-token-file", str(token_file), "--authorization-file", str(authorization)]
        return subprocess.run(command, capture_output=True, check=False), root, token_file, authorization, native, export

    def test_preflight_only_uses_formal_path_without_attempt(self):
        root, token_file, authorization, native, export = clone_case(self.tmp, "preflight", uuid.uuid4().hex)
        result = subprocess.run(["python", "-B", str(root / "launch_once.py"), "--authorization-file", str(authorization), "--preflight-only"], capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertFalse(wsl("/usr/bin/test", "-e", native, check=False).returncode == 0)
        self.assertFalse(export.exists())

    def test_gate_fail_has_zero_task_calls(self):
        result, _root, _token, _auth, native, _export = self.invoke("gate_fail")
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        status = json.loads(wsl("/usr/bin/cat", f"{native}/run-once/status.json").stdout.decode())
        self.assertEqual(status["task_calls"], 0)

    def test_gate_pass_reaches_four_arm_task_matrix(self):
        result, _root, _token, _auth, native, export = self.invoke("pass")
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        task = json.loads(wsl("/usr/bin/cat", f"{native}/run-once/task-comparison.json").stdout.decode())
        self.assertEqual(len(task["arms"]), 32)
        self.assertTrue((export / "EXPORT_COMPLETE.json").exists())

    def test_formal_entry_rejects_fake_backend_before_worker(self):
        result, _root, _token, _auth, native, _export = self.invoke("formal-fake", formal_entry=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(wsl("/usr/bin/test", "-e", f"{native}/run-once", check=False).returncode == 0)
        launcher = json.loads(wsl("/usr/bin/cat", f"{native}/launcher-status.json").stdout.decode())
        self.assertFalse(launcher.get("worker_started", False))

    def test_bad_token_and_duplicate_launch_rejected(self):
        root, token_file, authorization, native, export = clone_case(self.tmp, "bad-token", uuid.uuid4().hex)
        token_file.write_text("wrong", encoding="utf-8")
        result = subprocess.run(["python", "-B", str(root / "launch_once.py"), "--attempt-token-file", str(token_file), "--authorization-file", str(authorization)], capture_output=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(wsl("/usr/bin/test", "-e", native, check=False).returncode == 0)
        self.assertFalse(export.exists())
        result, root, token_file, authorization, _native, _export = self.invoke("duplicate")
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        second = subprocess.run(["python", "-B", str(root / "launch_once.py"), "--attempt-token-file", str(token_file), "--authorization-file", str(authorization)], capture_output=True, check=False)
        self.assertNotEqual(second.returncode, 0)


class ContractUnitTests(unittest.TestCase):
    def test_tamper_and_schema_errors_are_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("hashes.json", "execution-manifest.json"):
                shutil.copy2(ROOT / name, root / name)
            shutil.copy2(ROOT / "README.md", root / "README.md")
            with self.assertRaises(PackageContractError):
                verify_package(root)


if __name__ == "__main__":
    unittest.main()
