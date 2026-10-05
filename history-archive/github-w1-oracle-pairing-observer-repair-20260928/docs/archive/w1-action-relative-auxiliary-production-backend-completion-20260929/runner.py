"""Authorized worker entry for the A/B production pipeline."""
from __future__ import annotations

import json
import signal
import traceback
from pathlib import Path

from production_backend import ABRuntimeBackend
from production_pipeline import execute_pipeline

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "run-once"


def main(*, adapter_factory=None, boundary=None) -> int:
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(RuntimeError("worker interrupted")))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(RuntimeError("worker interrupted")))
    request = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
    matrix = json.loads((ROOT / "experiment-matrix.json").read_text(encoding="utf-8")) if (ROOT / "experiment-matrix.json").is_file() else {}
    if request.get("status") != "NOT_APPROVED" or not isinstance(request.get("attempt"), str) or not request.get("attempt"):
        raise RuntimeError("frozen request identity changed")
    backend = None
    try:
        attempt = request.get("attempt")
        if not isinstance(attempt, str) or not attempt:
            raise RuntimeError("frozen request attempt is missing")
        kwargs = {}
        if adapter_factory is not None:
            kwargs["adapter_factory"] = adapter_factory
        if boundary is not None:
            kwargs["boundary"] = boundary
        backend = ABRuntimeBackend(ROOT, OUT, request, matrix, **kwargs)
        result = execute_pipeline(backend, OUT)
        (OUT / "production-result.json").write_text(json.dumps({"status": result.status, "prediction_gate_passed": result.prediction_gate_passed, "task_stage_executed": result.task_stage_executed}, indent=2) + "\n", encoding="utf-8")
        return 0
    except BaseException as exc:
        try:
            if backend is not None:
                backend.settle("technical_stop", {"error": type(exc).__name__ + ": " + str(exc)})
        except BaseException:
            pass
        if OUT.is_dir():
            (OUT / "status.json").write_text(json.dumps({"status": "technical_stop_no_retry", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "automatic_retry": False}, indent=2) + "\n", encoding="utf-8")
        return 1
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
