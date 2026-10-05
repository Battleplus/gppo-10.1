"""Integration-only production entry using the explicit fake backend."""
from __future__ import annotations

import json
from pathlib import Path

from native_launch import verify_native_launch
from production_backend import FakeABBackend
from production_pipeline import execute_pipeline
from supervise import main as supervise_main

ROOT = Path(__file__).resolve().parent


def worker() -> int:
    contract = json.loads((ROOT / "launch-contract.json").read_text(encoding="utf-8"))
    mode = str(contract.get("integration_mode", "pass"))
    output = ROOT / "run-once"
    backend = FakeABBackend(ROOT, output, mode=mode)
    result = execute_pipeline(backend, output)
    return 0 if result.status in {"complete", "prediction_gate_stop", "prediction_effect_gate_passed_task_condition_failed", "data_reuse_stop", "confirmation_stop", "technical_stop"} else 1


def main() -> int:
    verify_native_launch(ROOT, allow_integration=True)
    return supervise_main(ROOT, "integration_production_worker.py", sample_interval=0.05)


if __name__ == "__main__":
    raise SystemExit(main())
