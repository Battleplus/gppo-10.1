"""SQLite call ledger with pending-call preservation and stage hard caps."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from pathlib import Path
from typing import Callable, Mapping


class BudgetError(RuntimeError):
    pass


class BudgetLedger:
    def __init__(self, path: Path, request: Mapping):
        self.request = request
        self.stage = ""
        self.phase_path = path.parent / "worker-phase-accounting.jsonl"
        self.phase_started_cpu = 0.0
        self.phase_started_wall = time.monotonic()
        self.phase_history = []
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY, stage TEXT, name TEXT, amounts TEXT, status TEXT, started REAL, finished REAL, error TEXT)"
        )
        self.connection.commit()

    def select(self, stage: str) -> None:
        if stage not in self.request["stages"]:
            raise BudgetError(f"unknown stage {stage}")
        if stage != self.stage:
            self._finish_phase(next_stage=stage)
        self.stage = stage

    @staticmethod
    def _process_cpu():
        if os.name != "posix":
            return time.process_time(), "self_only_nonproduction_test"
        import resource
        # A stage boundary must not leave a running child charged to the next stage.
        for task in Path("/proc/self/task").iterdir():
            try:
                children = (task / "children").read_text(encoding="ascii").strip()
            except FileNotFoundError:
                continue
            if children:
                raise BudgetError("STAGE_BOUNDARY_HAS_UNWAITED_CHILDREN")
        own = resource.getrusage(resource.RUSAGE_SELF)
        waited = resource.getrusage(resource.RUSAGE_CHILDREN)
        return own.ru_utime + own.ru_stime + waited.ru_utime + waited.ru_stime, "self_plus_waited_descendants"

    def _finish_phase(self, *, next_stage):
        cpu, scope = self._process_cpu()
        now = time.monotonic()
        stage = self.stage or "staging_and_zero_step_gate"
        row = {"stage": stage, "next_stage": next_stage, "pid": os.getpid(),
               "cpu_start_seconds": self.phase_started_cpu, "cpu_end_seconds": cpu,
               "cpu_seconds": cpu - self.phase_started_cpu,
               "wall_start_monotonic": self.phase_started_wall, "wall_end_monotonic": now,
               "wall_seconds": now - self.phase_started_wall, "scope": scope}
        if not all(math.isfinite(row[key]) and row[key] >= 0 for key in ("cpu_seconds", "wall_seconds")):
            raise BudgetError("WORKER_PHASE_ACCOUNTING_NONFINITE_OR_REVERSED")
        self.phase_history.append(row)
        from phase_handshake import request_boundary
        row["supervisor_handshake"] = request_boundary(row) if next_stage != stage else {"scope": "final_same_stage_snapshot"}
        from infra_io import durable_append_jsonl
        durable_append_jsonl(self.phase_path, row)
        self.phase_started_cpu, self.phase_started_wall = cpu, now
        cap = self.request["stages"].get(stage, {})
        for key, measure in (("complete_process_cpu_seconds", "cpu_seconds"), ("wall_seconds", "wall_seconds")):
            consumed = sum(item[measure] for item in self.phase_history if item["stage"] == stage)
            if key in cap and consumed > cap[key]:
                raise BudgetError("WORKER_PHASE_CAP_EXCEEDED:" + stage + ":" + key)

    def _used(self, stage: str, resource: str) -> int:
        total = 0
        for (payload,) in self.connection.execute(
            "SELECT amounts FROM calls WHERE stage=?", (stage,)
        ):
            total += int(json.loads(payload).get(resource, 0))
        return total

    def call(self, name: str, amounts: Mapping[str, int], function: Callable, *args, **kwargs):
        if not self.stage:
            raise BudgetError("ledger stage is not selected")
        limits = self.request["stages"][self.stage]
        for resource, amount in amounts.items():
            if amount < 0 or resource not in limits:
                raise BudgetError(f"invalid or unbudgeted resource {resource}")
            if self._used(self.stage, resource) + amount > int(limits[resource]):
                raise BudgetError(f"{self.stage} {resource} limit exceeded")
        cursor = self.connection.execute(
            "INSERT INTO calls(stage,name,amounts,status,started) VALUES(?,?,?,?,?)",
            (self.stage, name, json.dumps(dict(amounts), sort_keys=True), "pending", time.time()),
        )
        call_id = int(cursor.lastrowid)
        self.connection.commit()
        try:
            result = function(*args, **kwargs)
        except BaseException as exc:
            self.connection.execute(
                "UPDATE calls SET status='failed',finished=?,error=? WHERE id=?",
                (time.time(), type(exc).__name__ + ": " + str(exc), call_id),
            )
            self.connection.commit()
            raise
        self.connection.execute(
            "UPDATE calls SET status='complete',finished=? WHERE id=?", (time.time(), call_id)
        )
        self.connection.commit()
        return result

    def snapshot(self) -> dict:
        pending = self.connection.execute("SELECT COUNT(*) FROM calls WHERE status='pending'").fetchone()[0]
        failed = self.connection.execute("SELECT COUNT(*) FROM calls WHERE status='failed'").fetchone()[0]
        totals: dict[str, int] = {}
        for (payload,) in self.connection.execute("SELECT amounts FROM calls"):
            for resource, amount in json.loads(payload).items():
                totals[resource] = totals.get(resource, 0) + int(amount)
        return {"pending_calls": pending, "failed_calls": failed, "totals": totals}

    def assert_settled(self) -> dict:
        result = self.snapshot()
        if result["pending_calls"]:
            raise BudgetError("ledger has pending calls")
        limits = self.request.get("totals", {})
        for resource, used in result["totals"].items():
            if resource in limits and int(used) > int(limits[resource]):
                raise BudgetError(f"global {resource} limit exceeded")
        return result

    def close(self) -> None:
        try:
            if self.stage:
                self._finish_phase(next_stage="settlement_and_verified_export")
                self.stage = ""
        finally:
            self.connection.close()
