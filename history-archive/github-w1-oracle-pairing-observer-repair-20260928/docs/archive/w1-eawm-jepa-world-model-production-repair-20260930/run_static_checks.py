from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    python = [sys.executable, "-B"]
    tests = subprocess.run([*python, "-m", "unittest", "-v", "test_review_repairs.py", "test_w1_world_model.py", "test_production_world_metrics.py", "test_production_policy.py", "test_production_runtime.py"], cwd=ROOT, text=True, capture_output=True)
    preflight = subprocess.run([*python, "production_chain_check.py", "--preflight-only"], cwd=ROOT, text=True, capture_output=True)
    audit = subprocess.run([*python, "static_audit.py"], cwd=ROOT, text=True, capture_output=True)
    budget = subprocess.run([*python, "validate_resource_request.py"], cwd=ROOT, text=True, capture_output=True)
    result = {
        "schema": "w1-eawm-jepa-static-checks/1.0.0",
        "formal_dynamic_calls": {"environment": 0, "model_initialization": 0,
                                  "checkpoint_load": 0, "training_update": 0},
        "synthetic_test_operations": {
            "w1_graph_jepa_predict_candidates_forwards": 4,
            "ppo_regression_fake_policy_encode_calls": 2,
            "ppo_regression_fake_policy_evaluate_calls": 3,
            "native_gppo_preco_optimizer_updates": 1,
            "production_policy_tests_tiny_policy_encode_calls": 25161,
            "production_policy_tests_tiny_policy_evaluate_calls": 25161,
            "synthetic_world_loss_fake_online_model_calls": 1,
            "synthetic_world_loss_fake_target_model_calls": 1,
            "synthetic_world_loss_fake_backward_calls": 1,
            "synthetic_world_loss_fake_optimizer_steps": 1,
            "synthetic_test_count_scope": "Named test fixtures and call paths; not a runtime profiler and not a count of formal model calls."
        },
        "tests_exit_code": tests.returncode,
        "preflight_exit_code": preflight.returncode,
        "audit_exit_code": audit.returncode,
        "budget_exit_code": budget.returncode,
        "tests_stdout": tests.stdout,
        "tests_stderr": tests.stderr,
        "preflight_stdout": preflight.stdout,
        "preflight_stderr": preflight.stderr,
        "audit_stdout": audit.stdout,
        "audit_stderr": audit.stderr,
        "budget_stdout": budget.stdout,
        "budget_stderr": budget.stderr,
        "passed": tests.returncode == 0 and preflight.returncode == 0 and audit.returncode == 0 and budget.returncode == 0,
    }
    (ROOT / "checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("stdout") and not key.endswith("stderr")}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
