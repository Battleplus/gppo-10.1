import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import launch_once

try:
    import resource  # noqa: F401
except ModuleNotFoundError:
    resource_stub = ModuleType("resource")
    resource_stub.RUSAGE_SELF = 0
    resource_stub.RUSAGE_CHILDREN = 1
    resource_stub.getrusage = lambda _who: SimpleNamespace(ru_utime=0.0, ru_stime=0.0)
    sys.modules["resource"] = resource_stub

import native_launch
import wsl_stage_and_launch


class FakePosixPath:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value

    def is_absolute(self):
        return True

    def is_file(self):
        return self.value == "/proc/self/mountinfo"

    def exists(self):
        return False


class RemoteEntryGuardTests(unittest.TestCase):
    def test_windows_legacy_formal_launch_returns_frozen_remote_entry_error(self):
        manifest = {"attempt": "attempt-fixture"}
        contract = {"runtime_mode": "remote_joint"}
        with tempfile.TemporaryDirectory(prefix="w1-remote-launch-") as directory:
            token_path = Path(directory) / "token.txt"
            token_path.write_text("fixture-token", encoding="utf-8")
            with (
                patch.object(launch_once, "verify_authorization", return_value=(manifest, contract, {})),
                patch.object(launch_once, "run_preflight_probe") as run_probe,
                patch.object(launch_once.sys, "argv", [
                    "launch_once.py", "--authorization-file", "unused.json",
                    "--attempt-token-file", str(token_path),
                ]),
                contextlib.redirect_stderr(io.StringIO()) as stderr,
            ):
                self.assertEqual(launch_once.main(), 1)
        self.assertIn("REMOTE_JOINT_REQUIRES_FROZEN_REMOTE_ENTRY", stderr.getvalue())
        run_probe.assert_not_called()

    def test_windows_legacy_preflight_checks_joint_inputs_without_wsl_probe(self):
        manifest = {"attempt": "attempt-fixture"}
        contract = {"runtime_mode": "remote_joint"}
        with (
            patch.object(launch_once, "verify_authorization", return_value=(manifest, contract, {})),
            patch.object(launch_once, "verify_joint_inputs", return_value={"status": "pass"}) as verify_inputs,
            patch.object(launch_once, "run_preflight_probe") as run_probe,
            patch.object(launch_once.sys, "argv", ["launch_once.py", "--authorization-file", "unused.json", "--preflight-only"]),
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):
            self.assertEqual(launch_once.main(), 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["preflight_scope"], "package_identity_and_joint_inputs_only")
        self.assertFalse(result["staging_started"])
        verify_inputs.assert_called_once_with(launch_once.ROOT)
        run_probe.assert_not_called()

    def test_native_legacy_formal_launch_rejects_remote_mode(self):
        with tempfile.TemporaryDirectory(prefix="w1-remote-guard-") as directory:
            root = Path(directory)
            verified = {
                "identity": {"manifest": {"attempt": "attempt-fixture"}},
                "request": {"attempt": "attempt-fixture"},
                "contract": {"attempt": "attempt-fixture", "runtime_mode": "remote_joint"},
            }
            with (
                patch.object(native_launch, "require_native_linux_filesystem"),
                patch.object(native_launch, "verify_external_authorization", return_value=verified),
                patch.dict(os.environ, {"W1_EXTERNAL_AUTHORIZATION_FILE": str(root / "authorization.json")}),
            ):
                with self.assertRaisesRegex(RuntimeError, "REMOTE_JOINT_REQUIRES_FROZEN_REMOTE_ENTRY"):
                    native_launch.verify_native_launch(root)

    def test_native_remote_preflight_is_read_only(self):
        with tempfile.TemporaryDirectory(prefix="w1-remote-preflight-") as directory:
            root = Path(directory)
            verified = {
                "identity": {"manifest": {"attempt": "attempt-fixture"}},
                "request": {"status": "NOT_APPROVED", "attempt": "attempt-fixture"},
                "contract": {
                    "attempt": "attempt-fixture",
                    "runtime_mode": "remote_joint",
                    "resource_request_sha256": "fixture-digest",
                },
            }
            with (
                patch.object(native_launch, "require_native_linux_filesystem"),
                patch.object(native_launch, "verify_external_authorization", return_value=verified),
                patch.object(native_launch, "sha256_file", return_value="fixture-digest"),
                patch.object(native_launch, "probe_runtime_dependencies") as probe,
                patch("joint_inputs.verify_joint_inputs", return_value={"status": "pass"}),
                patch.dict(os.environ, {"W1_EXTERNAL_AUTHORIZATION_FILE": str(root / "authorization.json")}),
            ):
                self.assertEqual(native_launch.verify_native_launch(root, preflight_only=True), "attempt-fixture")
            self.assertFalse((root / "runtime-dependency-probe.json").exists())
            probe.assert_not_called()

            (root / "launch-contract.json").write_text(json.dumps({"runtime_mode": "remote_joint"}), encoding="utf-8")
            relative_accounting = Path(native_launch.NATIVE_LAUNCHER_ACCOUNTING_RELATIVE_PATH)
            with (
                patch.object(native_launch, "ROOT", root),
                patch.object(native_launch, "verify_native_launch", return_value="attempt-fixture"),
                patch.object(native_launch.sys, "argv", ["native_launch.py", "--preflight-only"]),
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                self.assertEqual(native_launch.main(), 0)
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["preflight_scope"], "package_identity_and_joint_inputs_only")
            self.assertFalse((root / "runtime-dependency-probe.json").exists())
            self.assertFalse(root.joinpath(*relative_accounting.parts).exists())

    def test_wsl_legacy_formal_launch_rejects_remote_mode(self):
        verified = {
            "identity": {"manifest": {"attempt": "attempt-fixture"}},
            "request": {"attempt": "attempt-fixture"},
            "contract": {"attempt": "attempt-fixture", "runtime_mode": "remote_joint"},
        }
        args = SimpleNamespace(
            source="/mnt/source", authorization_file="/mnt/authorization.json",
            native_root="/home/native", export_root="/mnt/export",
            preflight_only=False, attempt="attempt-fixture",
        )
        with (
            patch.object(wsl_stage_and_launch.os, "name", "posix"),
            patch.object(wsl_stage_and_launch, "Path", FakePosixPath),
            patch.object(wsl_stage_and_launch, "verify_external_authorization", return_value=verified),
        ):
            with self.assertRaisesRegex(RuntimeError, "REMOTE_JOINT_REQUIRES_FROZEN_REMOTE_ENTRY"):
                wsl_stage_and_launch.validate_linux_contract(args, "token")

    def test_wsl_remote_preflight_checks_joint_inputs_without_dependency_probe(self):
        verified = {
            "identity": {"manifest": {"attempt": "attempt-fixture"}},
            "request": {"attempt": "attempt-fixture"},
            "contract": {"attempt": "attempt-fixture", "runtime_mode": "remote_joint"},
        }
        args = SimpleNamespace(
            source="/mnt/source", authorization_file="/mnt/authorization.json",
            native_root="/home/native", export_root="/mnt/export",
            preflight_only=True, attempt="attempt-fixture",
        )
        with (
            patch.object(wsl_stage_and_launch.os, "name", "posix"),
            patch.object(wsl_stage_and_launch, "Path", FakePosixPath),
            patch.object(wsl_stage_and_launch, "verify_external_authorization", return_value=verified),
            patch("joint_inputs.verify_joint_inputs", return_value={"status": "pass"}) as verify_inputs,
            patch.object(wsl_stage_and_launch, "probe_runtime_dependencies") as probe,
        ):
            result = wsl_stage_and_launch.validate_linux_contract(args, "")
        self.assertEqual(result[-1]["preflight_scope"], "package_identity_and_joint_inputs_only")
        self.assertEqual(result[-1]["joint_inputs"], {"status": "pass"})
        verify_inputs.assert_called_once()
        probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
