"""CPU scope rules shared by the Linux staging and Windows launcher."""
from __future__ import annotations

import math
from pathlib import Path

CPU_SCOPE_SCHEMA = "w1-native-cpu-scope-breakdown/2.0.0"
SAME_SCOPE_TOLERANCE_SECONDS = 0.05
NATIVE_LAUNCHER_ACCOUNTING_RELATIVE_PATH = Path(
    "runtime-output/native-launcher-accounting.json"
)
_IDENTITY_TOLERANCE_SECONDS = 1e-6


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def monitor_cpu_estimate(
    observed_process_ticks: dict[object, int | dict[str, object]],
    clock_ticks: int,
    reaped_children_seconds: float,
) -> dict[str, object]:
    """Reconcile disjoint live deltas with cumulative reaped-child CPU.

    A live process record contains ``first_ticks``, ``last_ticks`` and
    ``live=True``. Its delta is CPU accrued during this monitor scope and is
    disjoint from ``RUSAGE_CHILDREN`` because it has not been reaped. Reaped
    CPU also covers children that were too short-lived to appear in a sample.
    Once an observed identity disappears, the caller cannot prove from these
    inputs whether its last ticks have reached ``RUSAGE_CHILDREN``; that scope
    is therefore marked unverifiable instead of silently double-counting or
    dropping the process.

    Integer values remain accepted as explicit scope deltas for small pure
    fixtures. Production supervision passes structured records with baselines.
    """
    if isinstance(clock_ticks, bool) or not isinstance(clock_ticks, int) or clock_ticks <= 0:
        raise ValueError("CPU_CLOCK_TICKS_INVALID")
    if not _finite_number(reaped_children_seconds) or reaped_children_seconds < 0:
        raise ValueError("REAPED_CHILDREN_CPU_INVALID")
    if not isinstance(observed_process_ticks, dict):
        raise ValueError("OBSERVED_PROCESS_TICKS_INVALID")
    total_live_delta_ticks = 0
    unresolved_identities: list[str] = []
    for identity, record in observed_process_ticks.items():
        if isinstance(record, bool):
            raise ValueError("OBSERVED_PROCESS_TICKS_INVALID")
        if isinstance(record, int):
            if record < 0:
                raise ValueError("OBSERVED_PROCESS_TICKS_INVALID")
            total_live_delta_ticks += record
            continue
        if not isinstance(record, dict):
            raise ValueError("OBSERVED_PROCESS_TICKS_INVALID")
        first = record.get("first_ticks")
        last = record.get("last_ticks")
        live = record.get("live")
        baseline_complete = record.get("baseline_complete", True)
        if (
            isinstance(first, bool) or not isinstance(first, int) or first < 0
            or isinstance(last, bool) or not isinstance(last, int) or last < 0
            or last < first or not isinstance(live, bool)
            or not isinstance(baseline_complete, bool)
        ):
            raise ValueError("OBSERVED_PROCESS_TICKS_INVALID")
        if not baseline_complete:
            unresolved_identities.append(str(identity))
        elif live:
            total_live_delta_ticks += last - first
        else:
            unresolved_identities.append(str(identity))
    observed = total_live_delta_ticks / clock_ticks
    reaped = float(reaped_children_seconds)
    verified = not unresolved_identities
    return {
        "observed_process_tree_seconds": observed,
        "reaped_children_seconds": reaped,
        "charged_seconds": observed + reaped,
        "scope_verified": verified,
        "unresolved_process_identities": unresolved_identities,
        "source": "disjoint(live_process_deltas, reaped_children)" if verified else "unverified(disappeared_process_scope)",
    }


def build_native_cpu_scope_breakdown(
    native_tree_cpu_seconds: float,
    supervisor_status: dict | None,
    launcher_accounting: dict | None,
    process_tree_sample: dict,
) -> dict:
    """Reconcile nested snapshots and locate the CPU tail after the last one.

    The waited native process tree is the complete, inclusive measurement. The
    supervisor and native-launcher snapshots are partial observations inside
    that interval. The positive launcher-snapshot-to-reap delta is attributed
    to settlement and remains part of the inclusive tree charge. Its source is
    known only as this interval; individual operations are not separately
    measured.
    """
    if not _finite_number(native_tree_cpu_seconds) or native_tree_cpu_seconds < 0:
        raise ValueError("NATIVE_PROCESS_TREE_CPU_INVALID")
    tree = float(native_tree_cpu_seconds)
    errors: list[str] = []

    supervisor_cpu = None
    supervisor_sampled = None
    supervisor_started = None
    if isinstance(supervisor_status, dict):
        raw_cpu = supervisor_status.get("cpu_seconds")
        if _finite_number(raw_cpu) and raw_cpu >= 0:
            supervisor_cpu = float(raw_cpu)
        else:
            errors.append("SUPERVISOR_CPU_SNAPSHOT_INVALID")
        supervisor_sampled = supervisor_status.get("cpu_sampled_monotonic")
        supervisor_started = supervisor_status.get("cpu_sampling_started_monotonic")
        own_cpu = supervisor_status.get("cpu_process_self_seconds_cumulative")
        child_cpu = supervisor_status.get("cpu_reaped_children_seconds_cumulative")
        if supervisor_cpu is not None and _finite_number(own_cpu) and _finite_number(child_cpu):
            component_delta = supervisor_cpu - float(own_cpu) - float(child_cpu)
            if abs(component_delta) > SAME_SCOPE_TOLERANCE_SECONDS:
                errors.append("SUPERVISOR_SAME_SCOPE_COMPONENT_MISMATCH")

    launcher_self_cpu = None
    launcher_children_cpu = None
    launcher_cpu = None
    launcher_sampled = None
    launcher_entry = None
    launcher_stage = "unavailable"
    if isinstance(launcher_accounting, dict):
        raw_values = (
            launcher_accounting.get("native_launcher_process_cpu_seconds"),
            launcher_accounting.get("native_launcher_reaped_children_cpu_seconds"),
            launcher_accounting.get("native_launcher_complete_process_cpu_snapshot_seconds"),
        )
        if all(_finite_number(value) and value >= 0 for value in raw_values):
            launcher_self_cpu, launcher_children_cpu, launcher_cpu = map(float, raw_values)
            if abs(launcher_cpu - launcher_self_cpu - launcher_children_cpu) > SAME_SCOPE_TOLERANCE_SECONDS:
                errors.append("NATIVE_LAUNCHER_SAME_SCOPE_COMPONENT_MISMATCH")
            launcher_sampled = launcher_accounting.get("sampled_monotonic")
            launcher_entry = launcher_accounting.get("process_module_entry_monotonic")
            launcher_stage = str(launcher_accounting.get("stage", "unknown"))
        else:
            errors.append("NATIVE_LAUNCHER_CPU_SNAPSHOT_INVALID")

    completion_delta = None
    signed_exit_delta = None
    exit_tail_cpu = None
    if supervisor_cpu is not None and launcher_cpu is not None:
        completion_delta = launcher_cpu - supervisor_cpu
        if completion_delta < -SAME_SCOPE_TOLERANCE_SECONDS:
            errors.append("LAUNCHER_SNAPSHOT_PRECEDES_SUPERVISOR_CPU_SNAPSHOT")
    if launcher_cpu is not None:
        signed_exit_delta = tree - launcher_cpu
        if signed_exit_delta < -SAME_SCOPE_TOLERANCE_SECONDS:
            errors.append("NATIVE_LAUNCHER_SNAPSHOT_EXCEEDS_WAITED_TREE_BY_MORE_THAN_TOLERANCE")
        exit_tail_cpu = max(0.0, signed_exit_delta)

    start = process_tree_sample.get("sample_started_monotonic")
    end = process_tree_sample.get("sample_ended_monotonic")
    for label, value in (("process-tree start", start), ("process-tree end", end)):
        if not _finite_number(value):
            errors.append(f"{label} timestamp unavailable")
    if _finite_number(start) and _finite_number(end) and float(end) < float(start):
        errors.append("NATIVE_PROCESS_TREE_SAMPLE_CLOCK_ORDER_INVALID")

    def check_timestamp(label: str, value: object) -> bool:
        if not _finite_number(value):
            errors.append(f"{label} timestamp unavailable")
            return False
        return True

    if supervisor_cpu is not None:
        start_valid = check_timestamp("supervisor start", supervisor_started)
        sample_valid = check_timestamp("supervisor sample", supervisor_sampled)
        if start_valid and sample_valid:
            if float(supervisor_started) > float(supervisor_sampled):
                errors.append("SUPERVISOR_CPU_SNAPSHOT_CLOCK_ORDER_INVALID")
            if _finite_number(start) and float(supervisor_started) < float(start):
                errors.append("SUPERVISOR_CPU_SAMPLING_START_BEFORE_OUTER_TREE")
            if _finite_number(end) and float(supervisor_sampled) > float(end):
                errors.append("SUPERVISOR_CPU_SNAPSHOT_AFTER_OUTER_TREE")

    if launcher_cpu is not None:
        entry_valid = check_timestamp("native-launcher entry", launcher_entry)
        sample_valid = check_timestamp("native-launcher sample", launcher_sampled)
        if entry_valid and sample_valid:
            if float(launcher_entry) > float(launcher_sampled):
                errors.append("NATIVE_LAUNCHER_CPU_SNAPSHOT_CLOCK_ORDER_INVALID")
            if _finite_number(start) and float(launcher_entry) < float(start):
                errors.append("NATIVE_LAUNCHER_ENTRY_BEFORE_OUTER_TREE")
            if _finite_number(end) and float(launcher_entry) > float(end):
                errors.append("NATIVE_LAUNCHER_ENTRY_AFTER_OUTER_TREE")
            if _finite_number(end) and float(launcher_sampled) > float(end):
                errors.append("NATIVE_LAUNCHER_CPU_SNAPSHOT_AFTER_OUTER_TREE")

    if supervisor_cpu is not None and launcher_cpu is not None:
        required = (start, launcher_entry, supervisor_started, supervisor_sampled, launcher_sampled, end)
        if all(_finite_number(value) for value in required):
            ordered = (
                float(start) <= float(launcher_entry)
                <= float(supervisor_started) <= float(supervisor_sampled)
                <= float(launcher_sampled) <= float(end)
            )
            if not ordered:
                errors.append("NESTED_CPU_SNAPSHOT_TIMESTAMPS_OUT_OF_ORDER")

    if errors:
        status = "failed"
    elif supervisor_cpu is None and launcher_cpu is not None:
        status = "launcher_only_early_exit"
    elif launcher_cpu is None:
        status = "tree_only_timeout_or_early_exit"
    else:
        status = "reconciled"

    return {
        "schema": CPU_SCOPE_SCHEMA,
        "status": status,
        "authoritative_total": "native_process_tree_cpu_seconds",
        "native_process_tree_cpu_seconds": tree,
        "native_process_tree_cpu_seconds_charged_once": tree,
        "native_process_tree_sample": process_tree_sample,
        "supervisor_complete_process_cpu_seconds_nested_snapshot": supervisor_cpu,
        "supervisor_cpu_sampled_monotonic": supervisor_sampled,
        "native_launcher_process_cpu_seconds_nested_component": launcher_self_cpu,
        "native_launcher_reaped_children_cpu_seconds_nested_component": launcher_children_cpu,
        "native_launcher_complete_process_cpu_snapshot_seconds": launcher_cpu,
        "native_launcher_process_module_entry_monotonic": launcher_entry if _finite_number(launcher_entry) else None,
        "native_launcher_cpu_sampled_monotonic": launcher_sampled if _finite_number(launcher_sampled) else None,
        "native_launcher_stage": launcher_stage,
        "supervisor_sample_to_native_launcher_snapshot_cpu_seconds": completion_delta,
        "native_launcher_snapshot_to_parent_reap_cpu_seconds": signed_exit_delta,
        "native_launcher_exit_settlement_cpu_seconds": exit_tail_cpu,
        "native_launcher_exit_settlement_stage": "settlement_and_verified_export" if exit_tail_cpu is not None else "unavailable",
        "native_launcher_exit_settlement_inclusive_in_tree": exit_tail_cpu is not None,
        "native_launcher_snapshot_to_parent_reap_wall_seconds": (
            max(0.0, float(end) - float(launcher_sampled))
            if _finite_number(end) and _finite_number(launcher_sampled) else None
        ),
        "tolerance_seconds_for_same_scope_mismatch_only": SAME_SCOPE_TOLERANCE_SECONDS,
        "scope_definitions": {
            "outer_tree": "WSL staging process RUSAGE_CHILDREN delta after waiting or stopping the native launch process; inclusive of its reaped descendants",
            "supervisor": "native entry RUSAGE_SELF plus reaped RUSAGE_CHILDREN snapshot after worker cleanup; nested diagnostic only",
            "native_launcher": "native entry RUSAGE_SELF plus reaped RUSAGE_CHILDREN snapshot after supervise.main returns; nested diagnostic only",
            "settlement_tail": "measured waited-tree CPU after the native-launcher snapshot through parent reap; attributed to settlement and already included in the one inclusive outer-tree charge; individual operations within this interval are not separately measured",
            "budget_rule": "charge the waited native process tree once; snapshots diagnose ordering and consistency and are never added to it",
        },
        "errors": errors,
    }


def settlement_stage_remaining(
    *, stage_cap: float, export_snapshot: float, exit_tail: float,
) -> float:
    """Return unused settlement allowance after disjoint export work and tail."""
    values = (stage_cap, export_snapshot, exit_tail)
    if any(not _finite_number(value) or value < 0 for value in values):
        raise ValueError("SETTLEMENT_STAGE_CPU_VALUE_INVALID")
    remaining = float(stage_cap) - float(export_snapshot) - float(exit_tail)
    if remaining < 0:
        raise ValueError("SETTLEMENT_STAGE_CPU_CAP_EXCEEDED")
    return remaining


def settlement_wall_remaining(
    *, stage_cap: float, export_snapshot: float, exit_tail: float,
) -> float:
    if any(not _finite_number(value) or value < 0 for value in (stage_cap, export_snapshot, exit_tail)):
        raise ValueError("SETTLEMENT_STAGE_WALL_VALUE_INVALID")
    remaining = float(stage_cap) - float(export_snapshot) - float(exit_tail)
    if remaining < 0:
        raise ValueError("SETTLEMENT_STAGE_WALL_CAP_EXCEEDED")
    return remaining


def summarize_failed_export_cpu_evidence(export_status: dict, *, expected_attempt: str) -> dict:
    """Keep measured CPU scope visible when native export stops or fails."""
    evidence = {
        key: export_status.get(key)
        for key in (
            "native_process_tree_cpu_seconds",
            "supervisor_complete_process_cpu_seconds",
            "native_launcher_cpu_scope_breakdown",
            "label_stage_complete_process_cpu_seconds",
            "settlement_stage_complete_process_cpu_seconds",
            "combined_budget_cpu_upper_bound_seconds",
            "combined_budget_wall_upper_bound_seconds",
        )
    }
    return {
        "status": "technical_stop",
        "attempt_matches": export_status.get("attempt") == expected_attempt,
        "export_status": export_status.get("status"),
        "error": export_status.get("error"),
        "measured_cpu_evidence": evidence,
        "automatic_retry": False,
    }


def validate_exported_cpu_scope(export_status: dict) -> dict:
    """Validate an exported nested-snapshot record and return its tail charge."""
    if not isinstance(export_status, dict):
        raise ValueError("NATIVE_CPU_SCOPE_BREAKDOWN_MISSING")
    scope = export_status.get("native_launcher_cpu_scope_breakdown")
    if not isinstance(scope, dict) or scope.get("schema") != CPU_SCOPE_SCHEMA:
        raise ValueError("NATIVE_CPU_SCOPE_BREAKDOWN_MISSING")
    if scope.get("status") != "reconciled" or scope.get("errors") != []:
        raise ValueError("NATIVE_CPU_SCOPE_BREAKDOWN_NOT_RECONCILED")
    tree = export_status.get("native_process_tree_cpu_seconds")
    supervisor = export_status.get("supervisor_complete_process_cpu_seconds")
    scope_tree = scope.get("native_process_tree_cpu_seconds")
    scope_supervisor = scope.get("supervisor_complete_process_cpu_seconds_nested_snapshot")
    if any(not _finite_number(value) or value < 0 for value in (tree, supervisor, scope_tree, scope_supervisor)):
        raise ValueError("NATIVE_CPU_SCOPE_IDENTITY_VALUE_INVALID")
    if abs(float(scope_tree) - float(tree)) > _IDENTITY_TOLERANCE_SECONDS:
        raise ValueError("NATIVE_CPU_SCOPE_TREE_IDENTITY_MISMATCH")
    if abs(float(scope_supervisor) - float(supervisor)) > _IDENTITY_TOLERANCE_SECONDS:
        raise ValueError("NATIVE_CPU_SCOPE_SUPERVISOR_IDENTITY_MISMATCH")
    charged_once = scope.get("native_process_tree_cpu_seconds_charged_once")
    if not _finite_number(charged_once) or abs(float(charged_once) - float(tree)) > _IDENTITY_TOLERANCE_SECONDS:
        raise ValueError("NATIVE_PROCESS_TREE_NOT_CHARGED_ONCE")

    launcher_self = scope.get("native_launcher_process_cpu_seconds_nested_component")
    launcher_children = scope.get("native_launcher_reaped_children_cpu_seconds_nested_component")
    launcher = scope.get("native_launcher_complete_process_cpu_snapshot_seconds")
    completion = scope.get("supervisor_sample_to_native_launcher_snapshot_cpu_seconds")
    signed_tail = scope.get("native_launcher_snapshot_to_parent_reap_cpu_seconds")
    tail = scope.get("native_launcher_exit_settlement_cpu_seconds")
    values = (launcher_self, launcher_children, launcher, completion, signed_tail, tail)
    if any(not _finite_number(value) for value in values):
        raise ValueError("NATIVE_CPU_SCOPE_COMPONENT_INVALID")
    if min(float(launcher_self), float(launcher_children), float(launcher), float(tail)) < 0:
        raise ValueError("NATIVE_CPU_SCOPE_COMPONENT_NEGATIVE")
    if abs(float(launcher) - float(launcher_self) - float(launcher_children)) > SAME_SCOPE_TOLERANCE_SECONDS:
        raise ValueError("NATIVE_LAUNCHER_SAME_SCOPE_COMPONENT_MISMATCH")
    if float(completion) < -SAME_SCOPE_TOLERANCE_SECONDS or float(signed_tail) < -SAME_SCOPE_TOLERANCE_SECONDS:
        raise ValueError("NATIVE_CPU_SCOPE_SNAPSHOT_MISMATCH")
    if abs(float(tree) - float(supervisor) - float(completion) - float(signed_tail)) > _IDENTITY_TOLERANCE_SECONDS:
        raise ValueError("NATIVE_CPU_SCOPE_DECOMPOSITION_MISMATCH")
    if abs(float(tail) - max(0.0, float(signed_tail))) > _IDENTITY_TOLERANCE_SECONDS:
        raise ValueError("NATIVE_CPU_SCOPE_SETTLEMENT_TAIL_MISMATCH")
    if abs(float(scope.get("tolerance_seconds_for_same_scope_mismatch_only", -1)) - SAME_SCOPE_TOLERANCE_SECONDS) > _IDENTITY_TOLERANCE_SECONDS:
        raise ValueError("NATIVE_CPU_SCOPE_TOLERANCE_IDENTITY_MISMATCH")
    return {"scope": scope, "exit_tail_cpu_seconds": float(tail)}
