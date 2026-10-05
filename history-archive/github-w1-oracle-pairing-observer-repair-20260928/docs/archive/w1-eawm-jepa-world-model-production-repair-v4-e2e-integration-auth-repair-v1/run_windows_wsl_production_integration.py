"""Exercise the real Windows -> WSL -> native supervisor chain with a sealed test copy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from manifest_contract import sha256_file, write_identity_files

ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, text=True, encoding="utf-8", errors="replace",
                          capture_output=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--wsl", default="wsl.exe")
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("WINDOWS_INTEGRATION_ENTRY_REQUIRES_WINDOWS")
    args.evidence_output = args.evidence_output.resolve()
    if args.evidence_output.exists() or args.evidence_output.is_relative_to(ROOT):
        raise RuntimeError("EVIDENCE_OUTPUT_MUST_BE_NEW_AND_OUTSIDE_PACKAGE")
    args.evidence_output.mkdir(parents=True)
    run_id = uuid.uuid4().hex
    attempt = f"w1-eawm-jepa-v4-{run_id}-integration-test"
    with tempfile.TemporaryDirectory(prefix="w1 eawm 集成 ") as temp:
        temp_root = Path(temp)
        package = temp_root / "冻结副本 with spaces"
        package.mkdir()
        for source in ROOT.iterdir():
            if source.is_file() and source.name not in {"execution-manifest.json", "hashes.json"}:
                shutil.copy2(source, package / source.name)
        request_path = package / "RESOURCE_REQUEST.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request["attempt"] = attempt
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        contract_path = package / "launch-contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        source_conv = run([args.wsl, "--distribution", contract["wsl_distribution"], "--exec", "/usr/bin/wslpath", "-a", "-u", str(package)])
        if source_conv.returncode:
            raise RuntimeError("TEST_SOURCE_WSLPATH_FAILED")
        source_wsl = source_conv.stdout.strip()
        evidence_native = f"/home/asus/{attempt}"
        export_dir = temp_root / "controlled export"
        export_dir.mkdir()
        export_conv = run([args.wsl, "--distribution", contract["wsl_distribution"], "--exec", "/usr/bin/wslpath", "-a", "-u", str(export_dir / "export")])
        if export_conv.returncode:
            raise RuntimeError("TEST_EXPORT_WSLPATH_FAILED")
        contract.update({
            "attempt": attempt, "integration_test": True,
            "integration_fault": None,
            "windows_source_root": str(package), "wsl_source_root": source_wsl,
            "native_execution_root": evidence_native,
            "windows_export_root": export_conv.stdout.strip(),
            "native_entry": "integration_native_launch.py",
            "authorization_status": "NOT_APPROVED",
        })
        contract["resource_request_sha256"] = sha256_file(request_path)
        contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        identity = write_identity_files(package, attempt=attempt, status="NOT_APPROVED")
        token = secrets.token_urlsafe(32)
        authorization = {
            "schema": "w1-external-launch-authorization/2.0.0", "attempt": attempt,
            "execution_manifest_sha256": identity["execution_manifest_sha256"],
            "hashes_sha256": identity["hashes_sha256"],
            "resource_request_sha256": sha256_file(request_path),
            "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
            "status": "APPROVED",
        }
        auth_path = temp_root / "test-authorization.json"
        auth_path.write_text(json.dumps(authorization, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        token_path = temp_root / "test-token.txt"
        token_path.write_text(token, encoding="utf-8")
        launcher = run([sys.executable, "-B", str(package / "launch_once.py"),
                        "--attempt-token-file", str(token_path),
                        "--authorization-file", str(auth_path)])
        token_path.unlink()
        package_identity_after = {
            "execution_manifest_sha256": sha256_file(package / "execution-manifest.json"),
            "hashes_sha256": sha256_file(package / "hashes.json"),
        }
        export_path = export_dir / "export"
        exported = sorted(path.name for path in export_path.iterdir()) if export_path.is_dir() else []
        evidence = {
            "schema": "w1-windows-wsl-production-integration/1.0.0",
            "test_attempt": attempt, "formal_attempt_created": False,
            "launcher_command": [sys.executable, "-B", str(package / "launch_once.py"),
                                 "--attempt-token-file", "<ephemeral-test-token-file>",
                                 "--authorization-file", str(auth_path)],
            "windows_launcher_returncode": launcher.returncode,
            "windows_launcher_stdout": launcher.stdout,
            "windows_launcher_stderr": launcher.stderr,
            "sealed_test_copy_identity": identity,
            "sealed_test_copy_identity_after": package_identity_after,
            "test_copy_identity_unchanged": package_identity_after == identity,
            "exported_files": exported,
            "export_exists": export_path.is_dir(),
            "token_persisted": token_path.exists(),
            "status": "pass" if launcher.returncode == 0 and export_path.is_dir() else "fail",
        }
        (args.evidence_output / "windows-wsl-integration.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.evidence_output / "windows-launch-stdout.txt").write_text(launcher.stdout, encoding="utf-8")
        (args.evidence_output / "windows-launch-stderr.txt").write_text(launcher.stderr, encoding="utf-8")
        if export_path.is_dir():
            shutil.copytree(export_path, args.evidence_output / "controlled-export")
    return 0 if evidence["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
