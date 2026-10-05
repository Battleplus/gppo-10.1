"""Run the production implementation's synthetic test suite under supervisor."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from infra_io import durable_atomic_json
from runtime_backend import W1RuntimeBackend

ROOT = Path(__file__).resolve().parent


def main() -> int:
    runtime_inputs = json.loads((ROOT / "runtime-inputs.json").read_text(encoding="utf-8"))
    source_root = Path(runtime_inputs["source_run"]["wsl_root"])
    for import_root in (source_root, source_root / "native"):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    request = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
    matrix = json.loads((ROOT / "experiment-matrix.json").read_text(encoding="utf-8"))
    backend = W1RuntimeBackend(ROOT, ROOT / "run-once", request, matrix)
    preflight = backend.zero_step_gate()
    if not preflight.get("pass"):
        backend.settle_exception(RuntimeError("SYNTHETIC_PRODUCTION_ZERO_STEP_FAILED"), {})
        backend.close()
        return 1
    backend.settle("synthetic_integration", {"zero_step_gate": preflight})
    backend.close()
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), str(source_root / "native"), env.get("PYTHONPATH", "")]
    )
    command = [
        sys.executable, "-B", "-m", "unittest", "-v",
        "test_review_repairs.py", "test_launch_authorization.py",
        "test_w1_world_model.py", "test_production_world_metrics.py",
        "test_production_policy.py", "test_production_runtime.py",
    ]
    result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                            capture_output=True, check=False)
    output = result.stdout + result.stderr
    (ROOT / "run-once" / "integration-tests.txt").write_text(output, encoding="utf-8")
    checks = ROOT / "checks.json"
    check_payload = json.loads(checks.read_text(encoding="utf-8")) if checks.is_file() else {}
    marker = next((line for line in output.splitlines() if line.startswith("Ran ")), "")
    evidence = {
        "schema": "w1-production-synthetic-worker/1.0.0",
        "status": "pass" if result.returncode == 0 and marker else "fail",
        "command": command,
        "test_summary": marker,
        "test_returncode": result.returncode,
        "test_output_tail": output[-12000:],
        "formal_dynamic_calls": {
            "environment_steps": 0, "resets": 0, "model_initializations": 0,
            "checkpoint_loads": 0, "training_updates": 0,
        },
        "synthetic_test_operations": {
            "scope": "Production test suite fixtures; not formal experiment calls.",
            "counts_source": "checks.json named fixture counters",
        },
        "production_test_suite_executed_by_supervised_worker": True,
        "production_runtime_backend_zero_step_executed": True,
        "production_sqlite_ledger_initialized_and_settled": True,
        "automatic_retry": False,
    }
    evidence["synthetic_test_operations"].update(check_payload.get("synthetic_test_operations", {}))
    evidence["unit_test_count"] = check_payload.get("test_count")
    evidence["checks_passed"] = result.returncode == 0 and bool(marker)
    durable_atomic_json(ROOT / "run-once" / "integration-worker-evidence.json", evidence)
    if result.returncode:
        sys.stderr.write(output)
    return result.returncode if result.returncode == 0 and bool(marker) else 1


if __name__ == "__main__":
    raise SystemExit(main())
