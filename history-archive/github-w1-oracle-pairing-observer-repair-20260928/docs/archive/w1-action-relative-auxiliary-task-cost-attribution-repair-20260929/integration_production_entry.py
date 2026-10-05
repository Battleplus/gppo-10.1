"""WSL integration worker: formal backend with bottom-boundary substitutes."""
from __future__ import annotations

import json
from pathlib import Path

from native_launch import verify_native_launch
from production_backend import ABRuntimeBackend
from production_pipeline import execute_pipeline
from runtime_adapter import ProductionRuntimeAdapter
from integration_boundary import IntegrationBoundary
from supervise import main as supervise_main

ROOT = Path(__file__).resolve().parent


def worker() -> int:
    request = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
    matrix = json.loads((ROOT / "experiment-matrix.json").read_text(encoding="utf-8"))
    contract = json.loads((ROOT / "launch-contract.json").read_text(encoding="utf-8"))
    boundary = IntegrationBoundary(str(contract.get("integration_mode", "pass")))
    output = ROOT / "run-once"
    factory = lambda root, out, req, mat, account, _boundary: ProductionRuntimeAdapter(
        root, out, req, mat, account, boundary=boundary)
    backend = ABRuntimeBackend(ROOT, output, request, matrix, adapter_factory=factory)
    try:
        result = execute_pipeline(backend, output)
        allowed = {"complete", "prediction_gate_stop", "prediction_effect_gate_passed_task_condition_failed", "confirmation_stop", "data_reuse_stop"}
        return 0 if result.status in allowed else 1
    finally:
        backend.close()


def main() -> int:
    verify_native_launch(ROOT, allow_integration=True)
    return supervise_main(ROOT, "integration_production_worker.py", sample_interval=0.05)


if __name__ == "__main__":
    raise SystemExit(main())
