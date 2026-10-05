from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

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


def make_authorization_fixture(base: Path) -> tuple[Path, Path, Path, str]:
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
    (package / "payload.txt").write_text("fixture", encoding="utf-8")
    identity = write_identity_files(package, attempt="attempt-test")
    authorization_path = base / "authorization.json"
    token = "fixture-token"
    authorization = {
        "schema": AUTHORIZATION_SCHEMA,
        "attempt": "attempt-test",
        "execution_manifest_sha256": identity["execution_manifest_sha256"],
        "hashes_sha256": identity["hashes_sha256"],
        "resource_request_sha256": sha256_file(request_path),
        "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
        "status": "APPROVED",
    }
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    token_path = base / "attempt-token.txt"
    token_path.write_text(token, encoding="utf-8")
    return package, authorization_path, token_path, token


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

    def test_final_authorization_must_be_one_complete_json_object(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            package, authorization_path, _token_path, token = make_authorization_fixture(base)
            authorization_path.write_text(
                authorization_path.read_text(encoding="utf-8") + "\n{}",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PackageContractError, "AUTHORIZATION_INVALID_JSON"):
                verify_external_authorization(package, authorization_path, token=token, preflight_only=False)

    def test_final_authorization_requires_every_bound_field(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            package, authorization_path, _token_path, token = make_authorization_fixture(base)
            authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
            del authorization["hashes_sha256"]
            authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
            with self.assertRaisesRegex(PackageContractError, "AUTHORIZATION_FIELDS_MISSING: hashes_sha256"):
                verify_external_authorization(package, authorization_path, token=token, preflight_only=False)

    def test_final_authorization_rejects_wrong_attempt_and_field_types(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            package, authorization_path, _token_path, token = make_authorization_fixture(base)
            authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
            authorization["attempt"] = 17
            authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
            with self.assertRaisesRegex(PackageContractError, "AUTHORIZATION_FIELD_TYPE_INVALID: attempt"):
                verify_external_authorization(package, authorization_path, token=token, preflight_only=False)

            authorization["attempt"] = "different-attempt"
            authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
            with self.assertRaisesRegex(PackageContractError, "ATTEMPT_IDENTITY_MISMATCH"):
                verify_external_authorization(package, authorization_path, token=token, preflight_only=False)

    def test_launcher_rejects_bad_identity_and_token_before_wsl_or_staging(self):
        cases = {
            "trailing_content": "AUTHORIZATION_INVALID_JSON",
            "missing_field": "AUTHORIZATION_FIELDS_MISSING",
            "wrong_identity": "ATTEMPT_IDENTITY_MISMATCH",
            "wrong_token": "EXTERNAL_TOKEN_MISMATCH",
        }
        for case, expected in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                package, authorization_path, token_path, token = make_authorization_fixture(base)
                authorization_text = authorization_path.read_text(encoding="utf-8")
                authorization = json.loads(authorization_text)
                supplied_token = token
                if case == "trailing_content":
                    authorization_text += "\n{}"
                    authorization_path.write_text(authorization_text, encoding="utf-8")
                elif case == "missing_field":
                    del authorization["hashes_sha256"]
                    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
                elif case == "wrong_identity":
                    authorization["attempt"] = "wrong-attempt"
                    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
                else:
                    supplied_token = "wrong-token"
                token_path.write_text(supplied_token, encoding="utf-8")

                stderr = io.StringIO()
                argv = [
                    "launch_once.py", "--authorization-file", str(authorization_path),
                    "--attempt-token-file", str(token_path),
                ]
                with patch.object(_LAUNCH_MODULE, "ROOT", package), \
                     patch.object(_LAUNCH_MODULE, "windows_to_wsl") as convert, \
                     patch.object(_LAUNCH_MODULE, "run_dependency_probe") as dependency, \
                     patch.object(_LAUNCH_MODULE, "run_wsl") as worker, \
                     patch("sys.argv", argv), redirect_stderr(stderr):
                    self.assertEqual(_LAUNCH_MODULE.main(), 1)
                result = json.loads(stderr.getvalue())
                self.assertEqual(result["stage"], "windows_preflight")
                self.assertIn(expected, result["error"])
                self.assertFalse(result["staging_started"])
                self.assertFalse(result["worker_started"])
                convert.assert_not_called()
                dependency.assert_not_called()
                worker.assert_not_called()

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

    def test_formal_native_entry_rejects_integration_backend(self):
        spec = importlib.util.spec_from_file_location("v3_formal_native_entry", ROOT / "native_launch.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.require_native_linux_filesystem = lambda _root: None
        module.sha256_file = lambda _path: "d" * 64
        module.probe_runtime_dependencies = lambda **_kwargs: {"status": "pass"}
        verified = {
            "identity": {"manifest": {"attempt": "attempt-integration-test"}},
            "request": {"status": "NOT_APPROVED"},
            "contract": {
                "integration_test": True,
                "attempt": "attempt-integration-test",
                "resource_request_sha256": "d" * 64,
            },
        }
        module.verify_external_authorization = lambda *_args, **_kwargs: verified
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "RESOURCE_REQUEST.json").write_text("{}", encoding="utf-8")
            with patch.dict("os.environ", {
                "W1_EXTERNAL_ATTEMPT_TOKEN": "test-token",
                "W1_EXTERNAL_AUTHORIZATION_FILE": str(root.parent / "test-authorization.json"),
            }, clear=False):
                with self.assertRaisesRegex(RuntimeError, "FORMAL_ENTRY_REJECTS_INTEGRATION_BACKEND"):
                    module.verify_native_launch(root)


if __name__ == "__main__":
    unittest.main()
