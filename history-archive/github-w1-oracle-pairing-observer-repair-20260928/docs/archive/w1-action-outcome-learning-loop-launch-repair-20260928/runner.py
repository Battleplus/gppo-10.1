"""Authorized worker entry for the complete finite learning loop."""

from __future__ import annotations

import json
import signal
import sys
import traceback
from pathlib import Path

from pipeline import execute_pipeline
from runtime_backend import AuthorizedRuntimeBackend


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "run-once"
ATTEMPT = "w1-action-outcome-learning-loop-launch-repair-v1-once"


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stop(signal_number, _frame):
    raise RuntimeError(f"worker received signal {signal_number}")


def main() -> int:
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    request = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
    matrix = json.loads((ROOT / "experiment-matrix.json").read_text(encoding="utf-8"))
    if request.get("status") != "NOT_APPROVED" or request.get("attempt") != ATTEMPT:
        raise RuntimeError("frozen request identity changed")
    backend = AuthorizedRuntimeBackend(ROOT, OUT, request, matrix)
    try:
        result = execute_pipeline(backend, OUT)
        _write(OUT / "status.json", {
            "attempt": ATTEMPT,
            "status": result.status,
            "prediction_gate_passed": result.prediction_gate_passed,
            "task_comparison_executed": result.task_comparison_executed,
            "automatic_retry": False,
            "automatic_extension": False,
        })
        return 0
    except BaseException as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        ledger = backend.ledger.snapshot() if backend.ledger is not None else None
        _write(OUT / "status.json", {
            "attempt": ATTEMPT,
            "status": "technical_stop_no_retry",
            "error": type(exc).__name__ + ": " + str(exc),
            "traceback": traceback.format_exc(),
            "ledger": ledger,
            "automatic_retry": False,
            "automatic_extension": False,
        })
        return 1
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
