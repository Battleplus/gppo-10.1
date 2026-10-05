from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from manifest_contract import (
    AUTHORIZATION_SCHEMA,
    LAUNCH_CONTRACT_SCHEMA,
    PackageContractError,
    sha256_file,
    verify_external_authorization,
    write_identity_files,
)

ROOT = Path(__file__).resolve().parent
_LAUNCH_SPEC = importlib.util.spec_from_file_location("w1_repair_launch_once", ROOT / "launch_once.py")
_LAUNCH_MODULE = importlib.util.module_from_spec(_LAUNCH_SPEC)
_LAUNCH_SPEC.loader.exec_module(_LAUNCH_MODULE)
build_command = _LAUNCH_MODULE.build_command


class ExternalAuthorizationTests(unittest.TestCase):
    def test_external_one_time_token_is_verified_at_each_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            package = base / "package"
            package.mkdir()
            request = {"schema": "test-request/1", "attempt": "attempt-test", "status": "NOT_APPROVED"}
            request_path = package / "RESOURCE_REQUEST.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            contract = {
                "schema": LAUNCH_CONTRACT_SCHEMA,
                "attempt": "attempt-test",
                "resource_request_sha256": sha256_file(request_path),
            }
            (package / "launch-contract.json").write_text(json.dumps(contract), encoding="utf-8")
            identity = write_identity_files(package, attempt="attempt-test")
            authorization_path = base / "authorization.json"
            authorization = {
                "schema": AUTHORIZATION_SCHEMA,
                "attempt": "attempt-test",
                "execution_manifest_sha256": identity["execution_manifest_sha256"],
                "hashes_sha256": identity["hashes_sha256"],
                "resource_request_sha256": sha256_file(request_path),
                "token_sha256": None,
                "status": "NOT_APPROVED",
            }
            authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
            verify_external_authorization(package, authorization_path, token=None, preflight_only=True)
            with self.assertRaisesRegex(PackageContractError, "APPROVED_AUTHORIZATION_AND_TOKEN_REQUIRED"):
                verify_external_authorization(package, authorization_path, token="test-token", preflight_only=False)

            authorization.update(
                status="APPROVED",
                token_sha256=hashlib.sha256(b"test-token").hexdigest(),
            )
            authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
            verify_external_authorization(package, authorization_path, token="test-token", preflight_only=False)
            with self.assertRaisesRegex(PackageContractError, "EXTERNAL_TOKEN_MISMATCH"):
                verify_external_authorization(package, authorization_path, token="other-token", preflight_only=False)

    def test_windows_command_passes_authorization_path_as_one_argument(self):
        contract = {
            "wsl_executable": "wsl.exe", "wsl_distribution": "Ubuntu-24.04",
            "native_python": "/usr/bin/python3", "linux_stage_entry": "stage.py",
            "native_execution_root": "/home/user/run", "windows_export_root": "/mnt/e/export",
            "attempt": "attempt-test", "argument_probe": "中文 path with spaces 'quoted'",
        }
        command = build_command(contract, "/mnt/e/source", "/mnt/e/private/auth file.json",
                                0.1, 0.2, preflight_only=True, dependency_only=False)
        position = command.index("--authorization-file")
        self.assertEqual(command[position + 1], "/mnt/e/private/auth file.json")
        self.assertEqual(command[position + 2], "--native-root")


if __name__ == "__main__":
    unittest.main()
