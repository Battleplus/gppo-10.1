"""Lifetime accounting parent for the frozen remote entry, including its exit tail."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CLOSING_CPU_RESERVE = 3.0
CLOSING_WALL_RESERVE = 5.0


def self_cpu():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def child_cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def closure_accounting(inner, *, own_cpu, waited_cpu, elapsed_wall, wrapper_wall,
                       request, child_returncode):
    values = (own_cpu, waited_cpu, elapsed_wall, wrapper_wall)
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
        raise RuntimeError("OUTER_CPU_SCOPE_VALUE_INVALID")
    snapshot = inner.get("native_cpu_total_seconds")
    if type(snapshot) not in (int, float) or not math.isfinite(snapshot) or snapshot < 0:
        raise RuntimeError("INNER_CPU_SNAPSHOT_MISSING")
    tail = waited_cpu - snapshot
    if tail < -0.05:
        raise RuntimeError("INNER_CPU_SNAPSHOT_EXCEEDS_WAITED_LIFETIME")
    settlement = inner.get("settlement_stage_complete_process_cpu_seconds")
    settlement_wall = inner.get("settlement_stage_elapsed_wall_seconds")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for value in (settlement, settlement_wall)):
        raise RuntimeError("INNER_SETTLEMENT_SCOPE_MISSING")
    total_upper = waited_cpu + own_cpu + CLOSING_CPU_RESERVE
    settlement_upper = settlement + max(0.0, tail) + own_cpu + CLOSING_CPU_RESERVE
    wall_upper = elapsed_wall + CLOSING_WALL_RESERVE
    settlement_wall_upper = settlement_wall + wrapper_wall + CLOSING_WALL_RESERVE
    cap = request["stages"]["settlement_and_verified_export"]
    passed = (total_upper <= request["totals"]["complete_process_cpu_seconds"]
              and wall_upper <= request["totals"]["wall_seconds"]
              and settlement_upper <= cap["complete_process_cpu_seconds"]
              and settlement_wall_upper <= cap["wall_seconds"])
    return {"schema": "w1-joint-outer-lifetime-accounting/1.0.0",
            "status": "complete" if passed and child_returncode == 0 and inner.get("status") == "complete" else "technical_stop",
            "native_cpu_total_seconds": total_upper, "native_wall_seconds": wall_upper,
            "measured_child_lifetime_cpu_seconds": waited_cpu,
            "inner_cpu_snapshot_seconds_diagnostic_only": snapshot,
            "inner_snapshot_to_reap_tail_cpu_seconds": max(0.0, tail),
            "outer_self_cpu_snapshot_seconds": own_cpu,
            "unmeasured_closing_cpu_reserve_seconds": CLOSING_CPU_RESERVE,
            "unmeasured_closing_wall_reserve_seconds": CLOSING_WALL_RESERVE,
            "settlement_stage_complete_process_cpu_seconds": settlement_upper,
            "settlement_stage_elapsed_wall_seconds": settlement_wall_upper,
            "cpu_charge_rule": "waited child lifetime once plus disjoint parent self plus bounded closing reserve; inner snapshots not added",
            "final_resource_pass": passed, "child_returncode": child_returncode,
            "automatic_retry": False, "research_success": False}


def enforce_closing_limits(own_cpu):
    # Linux checks RLIMIT_CPU in whole seconds. Leave one second below the
    # charged 3s reserve for the check/signal granularity; the parent is single-threaded.
    hard = math.floor(own_cpu) + 2
    existing = resource.getrlimit(resource.RLIMIT_CPU)[1]
    if existing != resource.RLIM_INFINITY:
        hard = min(hard, existing)
    resource.setrlimit(resource.RLIMIT_CPU, (hard, hard))
    signal.signal(signal.SIGALRM, lambda _sig, _frame: os._exit(124))
    signal.setitimer(signal.ITIMER_REAL, CLOSING_WALL_RESERVE)


def failure_accounting(*, attempt, error, waited_cpu, own_cpu, elapsed_wall,
                       child_returncode, timed_out, cleanup):
    return {"attempt": attempt, "status": "technical_stop", "error": error,
            "reaped_child_cpu_delta_lower_bound_seconds": waited_cpu,
            "outer_self_cpu_snapshot_seconds": own_cpu,
            "elapsed_wall_seconds": elapsed_wall, "child_returncode": child_returncode,
            "child_wait_timed_out": timed_out, "cleanup": cleanup,
            "native_cpu_total_seconds": None, "stage_attribution_complete": False,
            "final_resource_pass": False, "accounting_complete": False,
            "registration_terminal_status": "unknown", "automatic_retry": False}


def registration_handoff(root, checked, attempt, child_pid):
    from manifest_contract import read_object
    handoff = read_object(Path(str(root) + "-registration-handoff.json"), "REGISTRATION_HANDOFF")
    if (handoff.get("attempt") != attempt or handoff.get("outer_pid") != os.getpid()
            or handoff.get("inner_pid") != child_pid
            or handoff.get("manifest_sha256") != checked["manifest_sha256"]
            or handoff.get("hashes_sha256") != checked["hashes_sha256"]
            or not isinstance(handoff.get("registration_job_id"), str)
            or not handoff["registration_job_id"]):
        raise RuntimeError("REGISTRATION_HANDOFF_IDENTITY_MISMATCH")
    return handoff["registration_job_id"]


def main():
    raise RuntimeError('HISTORICAL_ENTRY_DISABLED_USE_RUN_G1_TASK_VALIDATION')
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-file", required=True, type=Path)
    parser.add_argument("--real-name", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from registration_identity import derive_name_id
    from manifest_contract import verify_package
    from linux_process_scope import enable_subreaper, reap_owned_children
    from infra_io import durable_atomic_json
    from server_registration import append_terminal

    derive_name_id(args.real_name)
    checked = verify_package(ROOT)
    request = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
    contract = json.loads((ROOT / "launch-contract.json").read_text(encoding="utf-8"))
    if not (request["attempt"] == contract["attempt"] == checked["manifest"]["attempt"]):
        raise RuntimeError("OUTER_ATTEMPT_CROSS_BINDING_MISMATCH")
    if ROOT.as_posix() != contract["native_execution_root"]:
        raise RuntimeError("OUTER_NATIVE_EXECUTION_PATH_MISMATCH")
    if not request["runner_ready"]:
        raise RuntimeError("PRODUCTION_ENGINEERING_ACCEPTANCE_REQUIRED")
    cpu_contract = json.loads((ROOT / "cpu-accounting-contract.json").read_text(encoding="utf-8"))
    if (cpu_contract["outer_closing_cpu_reserve_seconds"] != CLOSING_CPU_RESERVE
            or cpu_contract["outer_closing_wall_reserve_seconds"] != CLOSING_WALL_RESERVE):
        raise RuntimeError("OUTER_CLOSING_RESERVE_CONTRACT_MISMATCH")
    enable_subreaper()
    started = time.monotonic()
    baseline = child_cpu()
    child = None
    timed_out = False
    terminal_registration_status = None
    try:
        env = dict(os.environ, W1_OUTER_ACCOUNTING_PID=str(os.getpid()))
        child = subprocess.Popen(
            [sys.executable, "-I", "-B", str(ROOT / "joint_remote_native.py"),
             "--authorization-file", str(args.authorization_file), "--real-name", args.real_name],
            env=env, cwd=ROOT, stdin=sys.stdin, start_new_session=True)
        try:
            child.wait(timeout=max(0.0, request["totals"]["wall_seconds"] - CLOSING_WALL_RESERVE))
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()
            child.wait(timeout=2.0)
        reap_owned_children(timeout_seconds=2.0)
        waited = child_cpu() - baseline
        path = Path(str(ROOT) + "-final-settlement.json")
        inner = json.loads(path.read_text(encoding="utf-8"))
        wrapper_wall = max(0.0, time.monotonic() - started - inner.get("native_wall_seconds", 0.0))
        own = self_cpu()
        closed = closure_accounting(inner, own_cpu=own, waited_cpu=waited,
                                    elapsed_wall=time.monotonic() - started, wrapper_wall=wrapper_wall,
                                    request=request, child_returncode=child.returncode)
        closed.update(attempt=request["attempt"], inner_settlement=inner,
                      identity={"manifest_sha256": checked["manifest_sha256"], "hashes_sha256": checked["hashes_sha256"]})
        enforce_closing_limits(own)
        durable_atomic_json(path, {**closed, "status": "finalization_pending",
                                  "registration_terminal_status": "pending"})
        registration = inner.get("registration_job_id")
        if registration:
            event = "SUCCEEDED" if closed["status"] == "complete" else inner.get("requested_terminal_event", "FAILED")
            if event == "SUCCEEDED" and closed["status"] != "complete":
                event = "FAILED"
            append_terminal(registration, event, exit_code=0 if event == "SUCCEEDED" else 1,
                            reason_redacted="completed" if event == "SUCCEEDED" else "technical_stop",
                            result_path=str(ROOT) + "-export")
            terminal_registration_status = event
            closed["registration_terminal_event"] = event
            closed["registration_terminal_status"] = event
        else:
            closed.update(status="technical_stop", error="REGISTRATION_JOB_ID_MISSING")
        durable_atomic_json(path, closed)
        print(json.dumps(closed, ensure_ascii=False))
        return 0 if closed["status"] == "complete" else 1
    except BaseException as exc:
        cleanup = {"status": "not_completed"}
        if child is not None and child.poll() is None:
            try:
                child.kill()
                child.wait(timeout=2.0)
            except BaseException as wait_error:
                cleanup["child_wait_error"] = type(wait_error).__name__ + ": " + str(wait_error)
        try:
            cleanup.update(reap_owned_children(timeout_seconds=2.0))
        except BaseException as cleanup_error:
            cleanup.update(status="unresolved", error=type(cleanup_error).__name__ + ": " + str(cleanup_error))
        failure = failure_accounting(attempt=request["attempt"],
            error=type(exc).__name__ + ": " + str(exc), waited_cpu=child_cpu() - baseline,
            own_cpu=self_cpu(), elapsed_wall=time.monotonic() - started,
            child_returncode=None if child is None else child.returncode,
            timed_out=timed_out, cleanup=cleanup)
        failure["registration_terminal_status"] = terminal_registration_status or "unknown"
        failure["terminal_and_final_settlement_atomic"] = False
        enforce_closing_limits(self_cpu())
        if terminal_registration_status is None:
            try:
                job = registration_handoff(ROOT, checked, request["attempt"],
                                           None if child is None else child.pid)
                append_terminal(job, "FAILED", exit_code=1, reason_redacted="technical_stop",
                                result_path=str(ROOT) + "-outer-stop.json")
                failure["registration_terminal_status"] = "FAILED"
            except BaseException as registration_error:
                failure["registration_terminal_error"] = type(registration_error).__name__ + ": " + str(registration_error)
        durable_atomic_json(Path(str(ROOT) + "-outer-stop.json"), failure)
        print(json.dumps(failure, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
