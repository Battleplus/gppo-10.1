"""Audit a completed synthetic pipeline witness without rerunning it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(evidence_root: Path) -> dict[str, Any]:
    evidence_path = evidence_root / "controlled-run-evidence.json"
    evidence = _read_json(evidence_path)
    original_verification = evidence.get("verification", {})
    original_checks = original_verification.get("checks", {})
    replay_rows = evidence.get("hidden_replay", {})
    ledger_path = evidence_root / "verified-export" / "budget.sqlite3"
    method_replay = {}
    with sqlite3.connect(f"file:{ledger_path.as_posix()}?mode=ro", uri=True) as connection:
        for method in ("G0", "T", "G1"):
            route_rows = {
                name: row for name, row in replay_rows.items()
                if name.startswith(method + ":")
            }
            observations = sum(
                int(row.get("observations", 0)) for row in route_rows.values()
            )
            ledger_rows = connection.execute(
                "SELECT amounts, status FROM calls "
                "WHERE stage='conditional_policy_training' AND name=?",
                (method.lower() + "_recurrent_hidden_replay",),
            ).fetchall()
            ledger_charges = sum(
                int(json.loads(amounts).get("encode_sample_evaluations", 0))
                for amounts, _status in ledger_rows
            )
            statuses = [str(status) for _amounts, status in ledger_rows]
            method_replay[method] = {
                "route_count": len(route_rows),
                "route_groups": [row.get("groups") for row in route_rows.values()],
                "observations": observations,
                "ledger_call_count": len(ledger_rows),
                "ledger_encode_sample_evaluations": ledger_charges,
                "ledger_statuses": statuses,
                "matches": (
                    len(route_rows) == 3
                    and all(row.get("groups") == 2 for row in route_rows.values())
                    and observations > 0
                    and observations == len(ledger_rows) == ledger_charges
                    and all(status == "complete" for status in statuses)
                ),
            }
    checks = {
        "original_pipeline_checks_except_replay_accounting": all(
            passed for name, passed in original_checks.items()
            if name != "hidden_replay_observations_match_ledger_encode_charges"
        ),
        "reported_replay_observations_equal_sqlite_ledger_charges": all(
            row["matches"] for row in method_replay.values()
        ),
        "pipeline_completed_without_pipeline_error": (
            original_checks.get("pipeline_completed") is True
            and evidence.get("execution", {}).get("pipeline_error") is None
        ),
        "only_original_driver_check_failed": (
            sorted(name for name, passed in original_checks.items() if not passed)
            == ["hidden_replay_observations_match_ledger_encode_charges"]
        ),
    }
    passed = all(checks.values())
    audit_result = {
        "schema": "w1-controlled-synthetic-post-run-audit/1.0.0",
        "mode": "read_only_post_run_audit_no_pipeline_rerun",
        "exit_code": 0 if passed else 1,
        "original_driver_exit_code": 1,
        "original_driver_verification": original_verification,
        "checks": checks,
        "method_replay_accounting": method_replay,
        "all_pass": passed,
    }
    evidence["driver_verification_before_post_run_audit"] = original_verification
    evidence["post_run_audit"] = audit_result
    evidence["verification"] = {
        "checks": {
            **original_checks,
            "hidden_replay_observations_match_ledger_encode_charges": checks[
                "reported_replay_observations_equal_sqlite_ledger_charges"
            ],
        },
        "all_pass": passed,
        "verified_by": "read-only post-run audit against SQLite ledger",
    }
    evidence_path.write_text(
        json.dumps(evidence, sort_keys=True, indent=2, ensure_ascii=False,
                   allow_nan=False) + "\n", encoding="utf-8",
    )
    (evidence_root / "post-run-audit.json").write_text(
        json.dumps(audit_result, sort_keys=True, indent=2, ensure_ascii=False,
                   allow_nan=False) + "\n", encoding="utf-8",
    )
    return audit_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_root", type=Path)
    args = parser.parse_args()
    result = audit(args.evidence_root.resolve())
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False,
                     allow_nan=False))
    raise SystemExit(0 if result["all_pass"] else 1)


if __name__ == "__main__":
    main()
