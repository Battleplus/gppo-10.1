"""Read-only check that the previously repaired production chain is present."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "w1-light-repaired-fair-rerun-v2-nativefs-once"
REPAIR = ROOT / "research-plans" / "w1-action-relative-auxiliary-runtime-dependency-repair-v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_preflight() -> dict:
    required = {
        "run_manifest": RUN / "execution-manifest.json",
        "native_graph5": RUN / "native" / "gppo_world" / "graph5.py",
        "native_joint_gppo": RUN / "native" / "gppo_world" / "joint_gppo.py",
        "native_jepa": RUN / "native" / "gppo_world" / "jepa.py",
        "runtime_dependency_launcher": REPAIR / "launch_once.py",
        "runtime_dependency_wsl": REPAIR / "wsl_stage_and_launch.py",
    }
    present = {name: path.is_file() for name, path in required.items()}
    text_checks = {
        "graph5_25_action": "GRAPH5_ACTION_COUNT = 25" in required["native_graph5"].read_text(encoding="utf-8"),
        "wsl_linux_guard": "real Linux process" in required["runtime_dependency_wsl"].read_text(encoding="utf-8"),
        "preflight_mode": "--preflight-only" in required["runtime_dependency_launcher"].read_text(encoding="utf-8"),
    }
    passed = all(present.values()) and all(text_checks.values())
    return {
        "schema": "w1-production-chain-preflight/1.0.0",
        "passed": passed,
        "dynamic_calls": 0,
        "attempt_created": False,
        "worker_started": False,
        "checkpoint_loaded": False,
        "files": {name: {"path": str(path), "present": present[name], "sha256": sha(path) if present[name] else None} for name, path in required.items()},
        "checks": text_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true", required=True)
    parser.parse_args()
    result = check_preflight()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
