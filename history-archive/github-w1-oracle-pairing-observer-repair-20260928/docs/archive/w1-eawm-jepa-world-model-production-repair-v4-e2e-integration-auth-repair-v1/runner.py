"""Formal native worker for the frozen W1 EAWM/JEPA experiment."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from infra_io import durable_atomic_json
from runtime_backend import W1RuntimeBackend
from world_model_pipeline import execute_pipeline


ROOT = Path(__file__).resolve().parent


def main() -> int:
    output = ROOT / "run-once"
    backend = None
    evidence: dict = {}
    try:
        request = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
        matrix = json.loads((ROOT / "experiment-matrix.json").read_text(encoding="utf-8"))
        backend = W1RuntimeBackend(ROOT, output, request, matrix)
        result = execute_pipeline(backend)
        evidence = result.evidence
        durable_atomic_json(output / "status.json", {
            "schema": "w1-eawm-jepa-run-status/1.0.0",
            "status": result.status,
            "policy_training_executed": result.policy_training_executed,
            "task_comparison_executed": result.task_comparison_executed,
            "automatic_retry": False,
        })
        return 0
    except BaseException as exc:
        if backend is not None:
            try:
                backend.settle_exception(exc, evidence)
            except BaseException:
                traceback.print_exc()
        if output.exists():
            try:
                durable_atomic_json(output / "status.json", {
                    "schema": "w1-eawm-jepa-run-status/1.0.0",
                    "status": "technical_stop",
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "automatic_retry": False,
                })
            except BaseException:
                traceback.print_exc()
        traceback.print_exc()
        return 1
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
