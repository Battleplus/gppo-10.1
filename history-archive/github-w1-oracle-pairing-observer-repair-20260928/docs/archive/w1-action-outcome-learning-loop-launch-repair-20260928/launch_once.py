"""Windows-only one-shot entry that delegates every Linux path operation to WSL."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from infra_io import read_json, sha256_file, verify_manifest


ROOT = Path(__file__).resolve().parent


def verify_source(token: str, root: Path = ROOT) -> tuple[dict, dict, dict]:
    if os.name != "nt":
        raise RuntimeError("launch_once.py is the Windows entry and requires os.name=nt")
    hashes = read_json(root / "hashes.json")
    manifest_path = root / "execution-manifest.json"
    if sha256_file(manifest_path) != hashes["execution_manifest_sha256"]:
        raise RuntimeError("Execution manifest SHA-256 mismatch")
    manifest = read_json(manifest_path)
    verify_manifest(root, manifest["files"])
    contract = read_json(root / "launch-contract.json")
    request = read_json(root / "RESOURCE_REQUEST.json")
    if request.get("status") != "NOT_APPROVED":
        raise RuntimeError("Frozen request status changed")
    if not (manifest.get("attempt") == request.get("attempt") == contract.get("attempt")):
        raise RuntimeError("Attempt identity mismatch")
    if Path(contract["windows_source_root"]).resolve() != root:
        raise RuntimeError("Windows source root identity mismatch")
    supplied = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if not token or not hmac.compare_digest(supplied, contract["external_token_sha256"]):
        raise RuntimeError("Exact external one-shot token mismatch")
    return manifest, contract, request


def windows_to_wsl(wsl_exe: str, distribution: str, path: Path) -> str:
    command = [
        wsl_exe,
        "--distribution",
        distribution,
        "--exec",
        "/usr/bin/wslpath",
        "-a",
        "-u",
        str(path),
    ]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, check=False)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"wslpath failed with {result.returncode}: {message}")
    converted = result.stdout.decode("utf-8", errors="strict").strip()
    if not converted.startswith("/mnt/") or "\x00" in converted:
        raise RuntimeError(f"wslpath returned an invalid mounted source path: {converted!r}")
    return converted


def build_wsl_command(contract: dict, source_wsl: str, preflight_wall: float, preflight_cpu: float) -> list[str]:
    return [
        contract["wsl_executable"],
        "--distribution",
        contract["wsl_distribution"],
        "--exec",
        contract["native_python"],
        "-B",
        f"{source_wsl}/{contract['linux_stage_entry']}",
        "--source",
        source_wsl,
        "--native-root",
        contract["native_execution_root"],
        "--export-root",
        contract["windows_export_root"],
        "--attempt",
        contract["attempt"],
        "--argument-probe",
        contract["argument_probe"],
        "--windows-preflight-wall-seconds",
        repr(preflight_wall),
        "--windows-preflight-cpu-seconds",
        repr(preflight_cpu),
    ]


def run_wsl(command: list[str], token: str) -> tuple[int, str | None]:
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    child = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=None,
        stderr=None,
        shell=False,
        creationflags=flags,
    )
    interruption = None
    try:
        child.communicate(token.encode("utf-8"))
    except KeyboardInterrupt:
        interruption = "Windows launcher interrupted"
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
    return int(child.returncode), interruption


def orchestrate(token: str, root: Path = ROOT) -> int:
    windows_wall_start = time.monotonic()
    windows_cpu_start = time.process_time()
    _manifest, contract, request = verify_source(token, root)
    source_wsl = windows_to_wsl(
        contract["wsl_executable"], contract["wsl_distribution"], root
    )
    if source_wsl != contract["wsl_source_root"]:
        raise RuntimeError("WSL source root identity mismatch")
    preflight_wall = time.monotonic() - windows_wall_start
    preflight_cpu = time.process_time() - windows_cpu_start
    command = build_wsl_command(contract, source_wsl, preflight_wall, preflight_cpu)
    returncode, interruption = run_wsl(command, token)
    total_wall = time.monotonic() - windows_wall_start
    total_cpu = time.process_time() - windows_cpu_start
    outer_wall_pass = total_wall <= request["totals"]["wall_seconds"]
    print(json.dumps({
        "schema": "w1-windows-launcher-console-settlement/1.0.0",
        "attempt": contract["attempt"],
        "wsl_returncode": returncode,
        "interruption": interruption,
        "windows_launcher_wall_seconds": total_wall,
        "windows_launcher_cpu_seconds": total_cpu,
        "outer_wall_limit_seconds": request["totals"]["wall_seconds"],
        "outer_wall_pass": outer_wall_pass,
        "scope": "Windows identity checks, wslpath conversion, WSL process startup and wait",
        "persisted_in_native_artifacts": False,
        "unavailable_after_console_loss": True,
    }, sort_keys=True))
    return 0 if returncode == 0 and interruption is None and outer_wall_pass else 1


def main() -> int:
    failure_wall_start = time.monotonic()
    failure_cpu_start = time.process_time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-token-file", required=True)
    args = parser.parse_args()
    token = ""
    try:
        token_path = Path(args.attempt_token_file)
        if not token_path.is_file() or token_path.resolve().is_relative_to(ROOT):
            raise RuntimeError("External token file is missing or stored inside the package")
        token = token_path.read_text(encoding="utf-8").strip()
        return orchestrate(token)
    except BaseException as exc:
        print(json.dumps({
            "schema": "w1-windows-launcher-failure/1.0.0",
            "status": "technical_stop",
            "stage": "windows_preflight",
            "error": type(exc).__name__ + ": " + str(exc),
            "windows_wall_seconds": time.monotonic() - failure_wall_start,
            "windows_cpu_seconds": time.process_time() - failure_cpu_start,
            "staging_started": False,
            "worker_started": False,
            "automatic_retry": False,
        }, sort_keys=True), file=sys.stderr)
        return 1
    finally:
        token = ""


if __name__ == "__main__":
    raise SystemExit(main())
