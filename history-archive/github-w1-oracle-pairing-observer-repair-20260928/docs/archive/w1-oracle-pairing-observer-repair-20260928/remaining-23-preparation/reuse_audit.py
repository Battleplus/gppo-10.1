"""Zero-environment audit for reusing the completed observer-gate unit once."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


class ReuseAuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ReuseAuditError(f"missing reuse input: {path.name}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReuseAuditError(f"invalid JSONL {path.name}:{number}") from exc
        if not isinstance(value, dict):
            raise ReuseAuditError(f"non-object JSONL row {path.name}:{number}")
        rows.append(value)
    return rows


def _fleet_energy(info: dict) -> float:
    values = info.get("energy")
    if not isinstance(values, dict) or not values:
        raise ReuseAuditError("step lacks fleet energy")
    result = sum(float(value) for value in values.values())
    if not math.isfinite(result):
        raise ReuseAuditError("nonfinite fleet energy")
    return result


def _unit(row: dict) -> tuple[str | None, int]:
    return row.get("parent"), int(row.get("repeat", -1))


def audit_reuse(root: Path, reuse_dir: Path) -> dict:
    root = Path(root)
    reuse_dir = Path(reuse_dir)
    contract = json.loads((root / "reuse-contract.json").read_text(encoding="utf-8"))

    for relative, expected in contract["files"].items():
        path = reuse_dir / relative
        actual = sha256_file(path) if path.is_file() else "MISSING"
        if actual != expected:
            raise ReuseAuditError(f"reuse input hash mismatch: {relative}")

    source_manifest = json.loads(
        (reuse_dir / "source-execution-manifest.json").read_text(encoding="utf-8")
    )
    source_hashes = json.loads(
        (reuse_dir / "source-hashes.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("attempt") != contract["source_attempt"]:
        raise ReuseAuditError("source attempt identity mismatch")
    if source_hashes.get("execution_manifest_sha256") != contract["files"]["source-execution-manifest.json"]:
        raise ReuseAuditError("source hashes do not identify the source manifest")
    runtime_hashes = {}
    for relative in contract["runtime_identity_files"]:
        expected = source_manifest.get("files", {}).get(relative)
        if not isinstance(expected, str):
            raise ReuseAuditError(f"source manifest lacks runtime identity: {relative}")
        actual = sha256_file(root / relative)
        if actual != expected:
            raise ReuseAuditError(f"current runtime differs from reused unit: {relative}")
        runtime_hashes[relative] = actual

    gate = json.loads((reuse_dir / "technical-gate.json").read_text(encoding="utf-8"))
    status = json.loads((reuse_dir / "status.json").read_text(encoding="utf-8"))
    expected_unit = (
        contract["source_unit"]["parent"],
        int(contract["source_unit"]["repeat"]),
    )
    if not gate.get("passed") or _unit(gate) != expected_unit:
        raise ReuseAuditError("source technical gate is not a pass for the fixed unit")
    if status.get("status") != "technical_gate_complete":
        raise ReuseAuditError("source unit status is not technical_gate_complete")
    if status.get("ledger", {}).get("call_status_counts", {}).get("unknown", 0):
        raise ReuseAuditError("source unit ledger contains unknown calls")
    if status.get("ledger", {}).get("call_status_counts", {}).get("pending", 0):
        raise ReuseAuditError("source unit ledger contains pending calls")

    candidates = read_jsonl(reuse_dir / "candidate-decisions.jsonl")
    summaries = read_jsonl(reuse_dir / "branch-summaries.jsonl")
    pairing_rows = read_jsonl(reuse_dir / "communication-pairing.jsonl")
    steps = read_jsonl(reuse_dir / "steps.jsonl")
    if len(candidates) != 1 or _unit(candidates[0]) != expected_unit:
        raise ReuseAuditError("source candidate record is not exactly the fixed unit")
    candidate = candidates[0]
    if candidate.get("status") != "OPPORTUNITY":
        raise ReuseAuditError("source fixed unit lacks the frozen opportunity")
    if candidate.get("external_key") != contract["source_unit"]["external_key"]:
        raise ReuseAuditError("source external key mismatch")
    if candidate.get("public_state_sha256") != contract["source_unit"]["expected_public_state_sha256"]:
        raise ReuseAuditError("source public state mismatch")
    actions = [int(value) for value in candidate.get("branch_actions", [])]
    if actions != [int(value) for value in contract["source_unit"]["expected_actions"]]:
        raise ReuseAuditError("source action set mismatch")
    if len(pairing_rows) != 1 or _unit(pairing_rows[0]) != expected_unit:
        raise ReuseAuditError("source pairing record is not exactly the fixed unit")
    pairing = pairing_rows[0]
    if (
        int(pairing.get("common_call_keys", 0)) <= 0
        or int(pairing.get("action_specific_call_keys", 0)) <= 0
        or int(pairing.get("common_result_mismatches", -1)) != 0
        or int(pairing.get("parameter_conflict_count", -1)) != 0
        or int(pairing.get("multiplicity_mismatch_count", -1)) != 0
    ):
        raise ReuseAuditError("source primitive pairing evidence is not reusable")

    decision_step = int(candidate["decision_step"])
    prefix = sorted(
        (row for row in steps if row.get("arm") == "hungarian-prefix"),
        key=lambda row: int(row["actual_step"]),
    )
    if [int(row["actual_step"]) for row in prefix] != list(range(decision_step)):
        raise ReuseAuditError("source prefix step sequence is incomplete")
    initial_counts = {"completed": 0, "expired": 0}
    initial_energy = float(contract["utility"]["fleet_initial_energy"])
    if prefix:
        initial_counts = {
            "completed": int(prefix[-1]["info"]["counts"]["completed"]),
            "expired": int(prefix[-1]["info"]["counts"]["expired"]),
        }
        initial_energy = _fleet_energy(prefix[-1]["info"])

    by_action = {}
    for summary in summaries:
        identity = summary.get("identity", {})
        key = (identity.get("parent"), int(identity.get("repeat", -1)))
        action = int(identity.get("action", -1))
        if key != expected_unit or action in by_action:
            raise ReuseAuditError("duplicate or foreign source branch summary")
        by_action[action] = summary
    if sorted(by_action) != sorted(actions):
        raise ReuseAuditError("source branch summary action set mismatch")

    utility = contract["utility"]
    audited = []
    unknown_total = 0
    for action in actions:
        arm = f"action-{action:02d}"
        rows = sorted(
            (row for row in steps if row.get("arm") == arm),
            key=lambda row: int(row["actual_step"]),
        )
        actual_steps = [int(row["actual_step"]) for row in rows]
        if not rows or actual_steps != list(range(decision_step, decision_step + len(rows))):
            raise ReuseAuditError(f"incomplete source step sequence for action {action}")
        if any(not math.isfinite(float(row.get("reward", math.nan))) for row in rows):
            raise ReuseAuditError(f"nonfinite saved environment reward for action {action}")
        done_positions = [index for index, row in enumerate(rows) if bool(row.get("done"))]
        if done_positions != [len(rows) - 1]:
            raise ReuseAuditError(f"source branch lacks one final done row for action {action}")
        final_info = rows[-1].get("info", {})
        if not final_info.get("terminated") or final_info.get("truncated"):
            raise ReuseAuditError(f"source branch is not native-terminal for action {action}")

        previous_counts = dict(initial_counts)
        previous_energy = initial_energy
        recomputed = 0.0
        for index, row in enumerate(rows):
            info = row.get("info", {})
            counts = info.get("counts", {})
            completed = int(counts.get("completed", -1))
            expired = int(counts.get("expired", -1))
            if completed < previous_counts["completed"] or expired < previous_counts["expired"]:
                raise ReuseAuditError(f"nonmonotone task counts for action {action}")
            energy = _fleet_energy(info)
            success_delta = completed - previous_counts["completed"]
            expired_delta = expired - previous_counts["expired"]
            energy_used = max(0.0, previous_energy - energy)
            vector_task = (success_delta - expired_delta) / float(utility["task_capacity"])
            vector_energy = -energy_used / float(utility["fleet_initial_energy"])
            increment = math.pow(float(utility["gamma"]), index) * (
                float(utility["task_scale"])
                * float(utility["preference"][0])
                * vector_task
                + float(utility["preference"][1]) * vector_energy
            )
            if not math.isfinite(increment):
                raise ReuseAuditError(f"nonfinite utility increment for action {action}")
            recomputed += increment
            previous_counts = {"completed": completed, "expired": expired}
            previous_energy = energy

        summary = by_action[action]
        saved = float(summary.get("discounted_utility", math.nan))
        error = abs(recomputed - saved)
        if not math.isfinite(saved) or error > float(utility["absolute_tolerance"]):
            raise ReuseAuditError(f"utility recomputation mismatch for action {action}")
        tasks = summary.get("tasks")
        if not isinstance(tasks, dict) or len(tasks) != 6:
            raise ReuseAuditError(f"source branch lacks six task labels for action {action}")
        unknown = sum(
            value.get("status") in {"unknown", "censored", "unfinished"}
            for value in tasks.values()
        )
        unknown_total += unknown
        audited.append({
            "action": action,
            "step_rows": len(rows),
            "saved_environment_rewards_all_finite": True,
            "native_terminal": True,
            "truncated": False,
            "recomputed_discounted_utility": recomputed,
            "saved_discounted_utility": saved,
            "absolute_error": error,
            "unknown_task_labels": unknown,
        })

    if unknown_total != 7:
        raise ReuseAuditError("source unknown-label count changed")
    return {
        "schema": "w1-oracle-first-unit-reuse-audit/1.0.0",
        "passed": True,
        "source_attempt": contract["source_attempt"],
        "unit": {"parent": expected_unit[0], "repeat": expected_unit[1]},
        "counted_research_samples": 1,
        "runtime_identity_file_count": len(runtime_hashes),
        "runtime_identity_hashes": runtime_hashes,
        "candidate_actions": actions,
        "branch_count": len(audited),
        "step_record_count": len(steps),
        "utility_recomputations": audited,
        "maximum_absolute_utility_error": max(row["absolute_error"] for row in audited),
        "saved_environment_rewards_all_finite": True,
        "utility_recomputed_from": "saved per-step completed/expired counts and fleet energy under the frozen label contract",
        "raw_environment_scalar_reward_alone_is_the_utility": False,
        "all_branches_native_terminal_and_untruncated": True,
        "unknown_host_confirmation_labels": unknown_total,
        "unknown_labels_affect_utility_recomputation": False,
        "unknown_label_policy": contract["unknown_label_policy"],
        "historical_stopped_oracle_attempt_included_as_research_sample": False,
    }
