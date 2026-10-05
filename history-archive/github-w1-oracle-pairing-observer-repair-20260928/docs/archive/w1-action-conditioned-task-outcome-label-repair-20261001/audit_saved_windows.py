from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_evidence(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    needles = ("physical_completion=None", "host_confirmation=None", "branch.step", "def _event_target")
    hits = []
    for number, line in enumerate(lines, 1):
        if any(needle in line for needle in needles):
            hits.append({"line": number, "text": line.strip()})
    return {"path": str(path), "sha256": _sha256(path), "hits": hits}


def audit_windows(path: Path) -> dict[str, Any]:
    rows = []
    split_rows: Counter[str] = Counter()
    split_parents: defaultdict[str, set[str]] = defaultdict(set)
    status: Counter[str] = Counter()
    candidates: Counter[str] = Counter()
    event_valid: Counter[str] = Counter()
    outcome_valid: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    termination: Counter[str] = Counter()
    completion_records = 0
    completion_messages = 0
    task_states: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            split = str(row.get("split"))
            split_rows[split] += 1
            split_parents[split].add(str(row.get("parent")))
            status[f"{split}:{row.get('status')}"] += 1
            candidate_rows = row.get("candidate_branch_audit", [])
            candidates[split] += len(candidate_rows)
            for values, masks in zip(row.get("outcome_target", []), row.get("outcome_valid", [])):
                for index, is_valid in enumerate(masks):
                    if is_valid:
                        outcome_valid[f"{split}:outcome_index_{index}"] += 1
            for masks in row.get("event_valid", []):
                for index, is_valid in enumerate(masks):
                    if is_valid:
                        event_valid[f"{split}:event_index_{index}"] += 1
            for candidate in candidate_rows:
                info = candidate.get("branch_info") or {}
                branch_counts = info.get("counts") or {}
                for key in ("completed", "expired", "rejected"):
                    counts[f"{split}:{key}:{branch_counts.get(key, 'missing')}"] += 1
                termination[f"{split}:terminated={bool(info.get('terminated'))}:truncated={bool(info.get('truncated'))}"] += 1
                records = info.get("completion_records")
                if records:
                    completion_records += 1
                for item in info.get("communication_log") or ():
                    if isinstance(item, dict) and "completion" in str(item.get("message_kind", "")):
                        completion_messages += 1
                raw_tasks = info.get("tasks") or {}
                task_values = raw_tasks.values() if isinstance(raw_tasks, dict) else raw_tasks
                for task in task_values:
                    if isinstance(task, dict):
                        task_states[f"{split}:{task.get('state', 'missing')}"] += 1
                    elif isinstance(task, str):
                        task_states[f"{split}:{task}"] += 1
    return {
        "schema": "w1-action-conditioned-task-outcome-label-audit/1.0.0",
        "source": {"path": str(path), "sha256": _sha256(path)},
        "rows_by_split": dict(split_rows),
        "parents_by_split": {key: len(value) for key, value in split_parents.items()},
        "status_by_split": dict(status),
        "candidate_rows_by_split": dict(candidates),
        "outcome_valid_counts": dict(outcome_valid),
        "event_valid_counts": dict(event_valid),
        "legacy_event_dimensions": {
            "0": "public_field_change",
            "1": "new_measurement_received",
            "2": "continuation_publicly_confirmed",
            "3": "physical_completion_observed",
            "4": "host_confirmation_observed",
        },
        "legacy_missing_event_valid_counts": {
            "physical_completion_observed": sum(value for key, value in event_valid.items() if key.endswith("event_index_3")),
            "host_confirmation_observed": sum(value for key, value in event_valid.items() if key.endswith("event_index_4")),
        },
        "branch_count_summaries": dict(counts),
        "termination_summaries": dict(termination),
        "task_state_summaries": dict(task_states),
        "nonempty_completion_record_branches": completion_records,
        "completion_message_records": completion_messages,
        "scope_relation": {
            "all_complete_windows": sum(split_rows.values()),
            "prediction_confirmation_windows": split_rows.get("prediction_confirmation", 0),
            "prediction_confirmation_parents": len(split_parents.get("prediction_confirmation", set())),
            "prediction_confirmation_repeats_per_parent": 3,
        },
        "reconstruction": {
            "physical_completion": "NOT_RECONSTRUCTABLE_FROM_SAVED_EXPORT",
            "host_confirmation": "NOT_RECONSTRUCTABLE_FROM_SAVED_EXPORT",
            "expiry": "NOT_RECONSTRUCTABLE_FROM_SAVED_EXPORT",
            "reason_codes": [
                "COLLECTOR_WRITES_NULL_COMPLETION_FIELDS",
                "SINGLE_POST_ACTION_STEP_ONLY",
                "NO_TERMINAL_OR_TRUNCATED_BRANCHES",
                "NO_COMPLETION_RECORDS_OR_COMPLETION_MESSAGES",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit_windows(args.windows)
    result["source_code_evidence"] = _source_evidence(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
