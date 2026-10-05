"""Measure the read-only WSL dependency and authorization-path probe."""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

PROCESS_STARTED_WALL = time.monotonic()
PROCESS_STARTED_SELF_CPU = resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime
PROCESS_STARTED_CHILD_CPU = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime

PACKAGE_ROOT = Path(__file__).resolve().parent
_preloaded_probe = sys.modules.get("dependency_probe")
if _preloaded_probe is not None:
    _preloaded_path = getattr(_preloaded_probe, "__file__", None)
    if _preloaded_path is None:
        raise RuntimeError("DEPENDENCY_PROBE_PRELOADED_ORIGIN_UNVERIFIABLE")
    _preloaded_path = Path(_preloaded_path).resolve()
    if not _preloaded_path.is_relative_to(PACKAGE_ROOT) or not _preloaded_path.is_file():
        raise RuntimeError(f"DEPENDENCY_PROBE_PRELOADED_OUTSIDE_PACKAGE:{_preloaded_path}")

sys.path[:] = [
    entry for entry in sys.path
    if Path(entry or ".").resolve() != PACKAGE_ROOT
]
sys.path.insert(0, str(PACKAGE_ROOT))

from dependency_probe import probe


def usage_cpu(who: int) -> float:
    value = resource.getrusage(who)
    return value.ru_utime + value.ru_stime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-windows-path", required=True)
    parser.add_argument("--expected-python", required=True)
    args = parser.parse_args()
    wall_started = PROCESS_STARTED_WALL
    try:
        converted = subprocess.run(
            ["/usr/bin/wslpath", "-a", "-u", args.authorization_windows_path],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )
        if converted.returncode:
            raise RuntimeError("AUTHORIZATION_WSLPATH_CONVERSION_FAILED")
        authorization_wsl = converted.stdout.decode("utf-8", errors="strict").strip()
        if not authorization_wsl.startswith("/mnt/") or "\x00" in authorization_wsl:
            raise RuntimeError("AUTHORIZATION_WSLPATH_RESULT_INVALID")
        dependency = probe(
            expected_python=args.expected_python,
            expected_root=PACKAGE_ROOT,
        )
        payload = {
            "schema": "w1-launcher-preflight-probe/1.0.0",
            "status": "pass",
            "authorization_wsl_path": authorization_wsl,
            "dependency_probe": dependency,
            "process_tree_cpu_seconds": (
                usage_cpu(resource.RUSAGE_SELF) - PROCESS_STARTED_SELF_CPU
                + usage_cpu(resource.RUSAGE_CHILDREN) - PROCESS_STARTED_CHILD_CPU
            ),
            "process_wall_seconds": time.monotonic() - wall_started,
            "model_initialized": False,
            "checkpoint_loaded": False,
            "training_started": False,
        }
        print(json.dumps(payload, sort_keys=True))
        return 0
    except BaseException as exc:
        payload = {
            "schema": "w1-launcher-preflight-probe/1.0.0",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "process_tree_cpu_seconds": (
                usage_cpu(resource.RUSAGE_SELF) - PROCESS_STARTED_SELF_CPU
                + usage_cpu(resource.RUSAGE_CHILDREN) - PROCESS_STARTED_CHILD_CPU
            ),
            "process_wall_seconds": time.monotonic() - wall_started,
            "model_initialized": False,
            "checkpoint_loaded": False,
            "training_started": False,
        }
        print(json.dumps(payload, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
