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


ROOT = Path(__file__).resolve().parent
WSL = "wsl.exe"
DISTRO = "Ubuntu-24.04"
EXPORT_PARENT = Path(r"E:\Z博士\.codex-tmp\w1 production backend exports")


def wsl(*args: str, check: bool = True):
    return subprocess.run([WSL, "--distribution", DISTRO, "--exec", *args], capture_output=True, check=check)


def wslpath(path: Path) -> str:
    return wsl("/usr/bin/wslpath", "-a", "-u", str(path.resolve())).stdout.decode().strip()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


@unittest.skipUnless(os.name == "nt", "requires Windows to WSL")
class WindowsWslProductionChain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="w1-production-chain-", dir=r"E:\Z博士\.codex-tmp"))
        EXPORT_PARENT.mkdir(parents=True, exist_ok=True)

    def test_pass_path_uses_production_backend_and_exports(self):
        attempt = f"w1-production-backend-it-{uuid.uuid4().hex[:8]}"
        native_parent = f"/home/asus/{attempt}-native"
        wsl("/usr/bin/mkdir", "-p", native_parent)
        root = self.tmp / "包 中文 path with spaces 'quoted'"
        root.mkdir()
        manifest = json.loads((ROOT / "execution-manifest.json").read_text(encoding="utf-8"))
        for name in set(manifest["files"]) | {"execution-manifest.json", "hashes.json"}:
            shutil.copy2(ROOT / name, root / name)
        token = uuid.uuid4().hex
        export = EXPORT_PARENT / attempt
        contract = json.loads((root / "launch-contract.json").read_text(encoding="utf-8"))
        contract.update({"attempt": attempt, "integration_test": True, "integration_mode": "pass",
                         "native_entry": "integration_production_entry.py",
                         "native_execution_root": f"{native_parent}/{attempt}",
                         "windows_export_root": wslpath(export),
                         "windows_source_root": str(root.resolve()), "wsl_source_root": wslpath(root),
                         "external_token_sha256": hashlib.sha256(token.encode()).hexdigest()})
        write_json(root / "launch-contract.json", contract)
        request = json.loads((root / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
        request["attempt"] = attempt
        write_json(root / "RESOURCE_REQUEST.json", request)
        contract["resource_request_sha256"] = hashlib.sha256((root / "RESOURCE_REQUEST.json").read_bytes()).hexdigest()
        write_json(root / "launch-contract.json", contract)
        freeze_identity(root, attempt)
        authorization = self.tmp / f"{attempt}.authorization.json"
        write_json(authorization, {"schema": "w1-external-launch-authorization/2.0.0", "attempt": attempt,
                                   "execution_manifest_sha256": hashlib.sha256((root / "execution-manifest.json").read_bytes()).hexdigest(),
                                   "hashes_sha256": hashlib.sha256((root / "hashes.json").read_bytes()).hexdigest(),
                                   "resource_request_sha256": hashlib.sha256((root / "RESOURCE_REQUEST.json").read_bytes()).hexdigest(),
                                   "token_sha256": hashlib.sha256(token.encode()).hexdigest()})
        token_file = self.tmp / f"{attempt}.token"
        token_file.write_text(token, encoding="utf-8")
        result = subprocess.run(["python", "-B", str(root / "launch_once.py"), "--attempt-token-file", str(token_file), "--authorization-file", str(authorization)], capture_output=True, check=False)
        self.assertEqual(
            result.returncode,
            0,
            result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace"),
        )
        native = f"{native_parent}/{attempt}"
        status = json.loads(wsl("/usr/bin/cat", f"{native}/run-once/status.json").stdout.decode())
        self.assertEqual(status["status"], "complete")
        self.assertTrue(status["task_calls"] > 0)
        self.assertTrue((export / "EXPORT_COMPLETE.json").exists())
        self.assertFalse((root / "run-once").exists())


if __name__ == "__main__":
    unittest.main()
