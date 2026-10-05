"""Windows entrypoint and read-only preflight for the sealed package."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from manifest_contract import PackageContractError, verify_external_authorization

ROOT = Path(__file__).resolve().parent


def verify_authorization(root: Path, authorization_file: Path, *, token: str | None, preflight_only: bool) -> tuple[dict, dict, dict]:
    if os.name != "nt":
        raise PackageContractError("WINDOWS_ENTRY_REQUIRES_WINDOWS")
    verified = verify_external_authorization(
        root, authorization_file, token=token, preflight_only=preflight_only,
    )
    return verified["identity"]["manifest"], verified["contract"], verified["request"]


def windows_to_wsl(wsl_exe: str, distribution: str, path: Path) -> str:
    result = subprocess.run([wsl_exe, "--distribution", distribution, "--exec", "/usr/bin/wslpath", "-a", "-u", str(path)], stdin=subprocess.DEVNULL, capture_output=True, check=False)
    if result.returncode:
        raise PackageContractError("WSPATH_CONVERSION_FAILED")
    converted = result.stdout.decode("utf-8", errors="strict").strip()
    if not converted.startswith("/mnt/") or "\x00" in converted:
        raise PackageContractError("WSPATH_RESULT_INVALID")
    return converted


def run_dependency_probe(wsl_exe: str, distribution: str, native_python: str, source_wsl: str) -> dict:
    command = [wsl_exe, "--distribution", distribution, "--exec", native_python, "-B",
               f"{source_wsl}/dependency_probe.py", "--expected-python", native_python, "--json"]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, check=False)
    if result.returncode:
        raise PackageContractError("RUNTIME_DEPENDENCY_PROBE_FAILED")
    try:
        payload = json.loads(result.stdout.decode("utf-8", errors="strict").strip().splitlines()[-1])
    except (UnicodeDecodeError, json.JSONDecodeError, IndexError) as exc:
        raise PackageContractError("RUNTIME_DEPENDENCY_PROBE_OUTPUT_INVALID") from exc
    if payload.get("status") != "pass" or payload.get("model_initialized") or payload.get("checkpoint_loaded") or payload.get("training_started"):
        raise PackageContractError("RUNTIME_DEPENDENCY_PROBE_CONTRACT_FAILED")
    return payload


def build_command(contract: dict, source_wsl: str, authorization_wsl: str,
                  preflight_wall: float, preflight_cpu: float, *,
                  preflight_only: bool, dependency_only: bool) -> list[str]:
    command = [contract["wsl_executable"], "--distribution", contract["wsl_distribution"], "--exec", contract["native_python"], "-B", f"{source_wsl}/{contract['linux_stage_entry']}", "--source", source_wsl, "--authorization-file", authorization_wsl, "--native-root", contract["native_execution_root"], "--export-root", contract["windows_export_root"], "--attempt", contract["attempt"], "--argument-probe", contract["argument_probe"], "--windows-preflight-wall-seconds", repr(preflight_wall), "--windows-preflight-cpu-seconds", repr(preflight_cpu)]
    if preflight_only:
        command.append("--preflight-only")
    if dependency_only:
        command.append("--dependency-only")
    return command


def run_wsl(command: list[str], token: str) -> int:
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=None, stderr=None, shell=False)
    child.communicate(token.encode("utf-8"))
    return int(child.returncode)


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-token-file")
    parser.add_argument("--authorization-file", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dependency-only", action="store_true")
    args = parser.parse_args()
    try:
        token = ""
        if args.preflight_only and args.dependency_only:
            raise PackageContractError("PREFLIGHT_AND_DEPENDENCY_ONLY_ARE_EXCLUSIVE")
        if not args.preflight_only:
            if not args.attempt_token_file:
                raise PackageContractError("TOKEN_FILE_REQUIRED")
            token_path = Path(args.attempt_token_file)
            if not token_path.is_file() or token_path.resolve().is_relative_to(ROOT.resolve()):
                raise PackageContractError("TOKEN_FILE_INVALID")
            token = token_path.read_text(encoding="utf-8").strip()
        manifest, contract, _request = verify_authorization(ROOT, Path(args.authorization_file), token=token, preflight_only=args.preflight_only)
        source_wsl = windows_to_wsl(contract["wsl_executable"], contract["wsl_distribution"], ROOT)
        if source_wsl != contract["wsl_source_root"]:
            raise PackageContractError("WSL_SOURCE_IDENTITY_MISMATCH")
        authorization_wsl = windows_to_wsl(
            contract["wsl_executable"], contract["wsl_distribution"], Path(args.authorization_file),
        )
        dependency = run_dependency_probe(contract["wsl_executable"], contract["wsl_distribution"], contract["native_python"], source_wsl)
        preflight_wall = time.monotonic() - started
        preflight_cpu = time.process_time()
        code = run_wsl(build_command(contract, source_wsl, authorization_wsl, preflight_wall,
                                     preflight_cpu, preflight_only=args.preflight_only,
                                     dependency_only=args.dependency_only), token)
        print(json.dumps({"status": "preflight_pass" if args.preflight_only and code == 0 else "completed", "attempt": manifest["attempt"], "wsl_returncode": code, "dependency_probe": {"torch": dependency.get("torch_version"), "python": dependency.get("python_executable"), "cpu_only": dependency.get("cpu_only")}, "staging_started": False if args.preflight_only else "unknown", "worker_started": False if args.preflight_only or args.dependency_only else "unknown", "elapsed_wall_seconds": time.monotonic() - started}, sort_keys=True))
        return code
    except BaseException as exc:
        print(json.dumps({"schema": "w1-windows-launcher-failure/2.0.0", "status": "technical_stop", "stage": "windows_preflight", "error": f"{type(exc).__name__}: {exc}", "staging_started": False, "worker_started": False, "automatic_retry": False, "elapsed_wall_seconds": time.monotonic() - started}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
