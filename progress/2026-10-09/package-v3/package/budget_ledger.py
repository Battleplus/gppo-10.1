"""SQLite call ledger with pending-call preservation and stage hard caps."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from stage_contract import initial_stage
from pathlib import Path
from typing import Callable, Mapping


class BudgetError(RuntimeError):
    pass


class BudgetLedger:
    def __init__(self, path: Path, request: Mapping):
        self.request = request
        self.stage = ""
        self.phase_path = path.parent / "worker-phase-accounting.jsonl"
        self.protocol_path = path.parent / "worker-phase-protocol.jsonl"
        self.phase_started_cpu = 0.0
        self.phase_started_wall = time.monotonic()
        self.phase_history: list[dict] = []
        self.phase_protocol_sequence = 0
        self._closed = False
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY, stage TEXT, name TEXT, amounts TEXT, status TEXT, started REAL, finished REAL, error TEXT)"
        )
        self.connection.commit()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='reservations'"
            ).fetchone()
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS reservations(stage TEXT,resource TEXT,units INTEGER NOT NULL,PRIMARY KEY(stage,resource))"
            )
            if existing is None:
                for stage, payload in self.connection.execute("SELECT stage,amounts FROM calls").fetchall():
                    for resource, amount in json.loads(payload).items():
                        self._reserve_counter(stage, resource, amount)
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def select(self, stage: str) -> None:
        if self._closed:
            raise BudgetError("ledger is closed")
        if stage not in self.request["stages"]:
            raise BudgetError(f"unknown stage {stage}")
        if not self.stage:
            first = initial_stage(self.request)
            self._record_boundary(
                stage=first,
                next_stage=first,
                event="stage_enter",
            )
            self.stage = first
        if stage == self.stage:
            return
        self._record_boundary(stage=self.stage, next_stage=stage, event="transition")
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

    def _phase_row(self, *, stage: str, next_stage: str, event: str,
                   cpu_start: float, wall_start: float, reason: str | None = None) -> dict:
        cpu, scope = self._process_cpu()
        now = time.monotonic()
        row = {
            "event": event,
            "stage": stage,
            "next_stage": next_stage,
            "pid": os.getpid(),
            "cpu_start_seconds": cpu_start,
            "cpu_end_seconds": cpu,
            "cpu_seconds": cpu - cpu_start,
            "wall_start_monotonic": wall_start,
            "wall_end_monotonic": now,
            "wall_seconds": now - wall_start,
            "scope": scope,
        }
        if reason is not None:
            row["reason"] = reason
        if not all(math.isfinite(row[key]) and row[key] >= 0 for key in ("cpu_seconds", "wall_seconds")):
            raise BudgetError("WORKER_PHASE_ACCOUNTING_NONFINITE_OR_REVERSED")
        return row

    def _completed_phase_amount(self, stage: str, measure: str) -> float:
        return sum(
            float(item[measure])
            for item in self.phase_history
            if item["stage"] == stage and item["event"] == "transition"
        )

    def _check_phase_cap(self, row: Mapping) -> None:
        stage = row["stage"]
        cap = self.request["stages"].get(stage, {})
        prior = self._completed_phase_amount(stage, "cpu_seconds")
        candidate_cpu = prior + float(row["cpu_seconds"])
        if "complete_process_cpu_seconds" in cap and candidate_cpu > cap["complete_process_cpu_seconds"]:
            raise BudgetError("WORKER_PHASE_CAP_EXCEEDED:" + stage + ":complete_process_cpu_seconds")
        if "wall_seconds" in cap:
            prior_wall = self._completed_phase_amount(stage, "wall_seconds")
            if prior_wall + float(row["wall_seconds"]) > cap["wall_seconds"]:
                raise BudgetError("WORKER_PHASE_CAP_EXCEEDED:" + stage + ":wall_seconds")

    def _record_boundary(self, *, stage: str, next_stage: str, event: str) -> dict:
        if event == "stage_enter":
            cpu_start = 0.0
            wall_start = self.phase_started_wall
        else:
            cpu_start = self.phase_started_cpu
            wall_start = self.phase_started_wall
        row = self._phase_row(
            stage=stage,
            next_stage=next_stage,
            event=event,
            cpu_start=cpu_start,
            wall_start=wall_start,
        )
        if event == "stage_enter":
            row["stage_elapsed_cpu_seconds"] = row["cpu_seconds"]
            row["stage_elapsed_wall_seconds"] = row["wall_seconds"]
            row["cpu_seconds"] = 0.0
            row["wall_seconds"] = 0.0
        self._check_phase_cap(row)
        self.phase_protocol_sequence += 1
        row["protocol_sequence"] = self.phase_protocol_sequence
        from phase_handshake import request_boundary

        row["supervisor_handshake"] = request_boundary(row, protocol_path=self.protocol_path)
        from infra_io import durable_append_jsonl

        durable_append_jsonl(self.phase_path, row)
        self.phase_history.append(row)
        if event != "stage_enter":
            self.phase_started_cpu = float(row["cpu_end_seconds"])
            self.phase_started_wall = float(row["wall_end_monotonic"])
        return row

    def snapshot_phase(self, *, reason: str = "same_stage_observation") -> dict | None:
        """Record the open phase so far without a handshake or origin movement."""
        if self._closed:
            raise BudgetError("ledger is closed")
        if not self.stage:
            return None
        row = self._phase_row(
            stage=self.stage,
            next_stage=self.stage,
            event="snapshot",
            cpu_start=self.phase_started_cpu,
            wall_start=self.phase_started_wall,
            reason=reason,
        )
        row["supervisor_handshake"] = {"scope": "no_transition_snapshot"}
        self._check_phase_cap(row)
        from infra_io import durable_append_jsonl

        durable_append_jsonl(self.phase_path, row)
        return row

    def _used(self, stage: str, resource: str) -> int:
        row = self.connection.execute(
            "SELECT units FROM reservations WHERE stage=? AND resource=?", (stage, resource)
        ).fetchone()
        return row[0] if row else 0

    def _total_used(self, resource: str) -> int:
        return self.connection.execute(
            "SELECT COALESCE(SUM(units),0) FROM reservations WHERE resource=?", (resource,)
        ).fetchone()[0]

    def _reserve_counter(self, stage, resource, amount):
        self.connection.execute(
            "INSERT INTO reservations(stage,resource,units) VALUES(?,?,?) "
            "ON CONFLICT(stage,resource) DO UPDATE SET units=units+excluded.units",
            (stage, resource, amount),
        )

    def call(self, name: str, amounts: Mapping[str, int], function: Callable, *args, **kwargs):
        if not self.stage:
            raise BudgetError("ledger stage is not selected")
        limits = self.request["stages"][self.stage]
        # Reservation is atomic across connections. Failed/pending calls retain
        # their charge; total caps must stop operations, not just reject settlement.
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            for resource, amount in amounts.items():
                if (type(amount) is not int or amount < 0 or resource not in limits
                        or resource not in self.request.get("totals", {})):
                    raise BudgetError(f"invalid or unbudgeted resource {resource}")
                if self._used(self.stage, resource) + amount > int(limits[resource]):
                    raise BudgetError(f"{self.stage} {resource} limit exceeded")
                if self._total_used(resource) + amount > int(self.request["totals"][resource]):
                    raise BudgetError(f"global {resource} limit exceeded")
            cursor = self.connection.execute(
                "INSERT INTO calls(stage,name,amounts,status,started) VALUES(?,?,?,?,?)",
                (self.stage, name, json.dumps(dict(amounts), sort_keys=True), "pending", time.time()),
            )
            call_id = int(cursor.lastrowid)
            for resource, amount in amounts.items():
                self._reserve_counter(self.stage, resource, amount)
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise
        try:
            measured_wall = time.monotonic()
            measured_cpu = time.process_time()
            result = function(*args, **kwargs)
        except BaseException as exc:
            self.connection.execute(
                "UPDATE calls SET status='failed',finished=?,error=? WHERE id=?",
                (time.time(), type(exc).__name__ + ": " + str(exc), call_id),
            )
            self.connection.commit()
            from infra_io import durable_append_jsonl
            durable_append_jsonl(self.phase_path.parent / 'operation-measurements.jsonl', {
                'call_id': call_id, 'stage': self.stage, 'name': name, 'amounts': dict(amounts),
                'status': 'failed', 'wall_seconds': time.monotonic() - measured_wall,
                'cpu_seconds': time.process_time() - measured_cpu,
                'scope': 'inclusive; nested operations are non-additive'})
            raise
        from infra_io import durable_append_jsonl
        durable_append_jsonl(self.phase_path.parent / 'operation-measurements.jsonl', {
            'call_id': call_id, 'stage': self.stage, 'name': name, 'amounts': dict(amounts),
            'wall_seconds': time.monotonic() - measured_wall,
            'cpu_seconds': time.process_time() - measured_cpu,
            'status': 'complete', 'scope': 'inclusive; nested operations are non-additive'})
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
        independently_reserved = {}
        for stage, payload in self.connection.execute("SELECT stage,amounts FROM calls"):
            for resource, amount in json.loads(payload).items():
                key = (stage, resource)
                independently_reserved[key] = independently_reserved.get(key, 0) + amount
        projection = {(stage, resource):units for stage, resource, units
                      in self.connection.execute("SELECT stage,resource,units FROM reservations")}
        if projection != independently_reserved:
            raise BudgetError("reservation projection differs from durable call records")
        limits = self.request.get("totals", {})
        for resource, used in result["totals"].items():
            if resource not in limits:
                raise BudgetError(f"global resource missing: {resource}")
            if int(used) > int(limits[resource]):
                raise BudgetError(f"global {resource} limit exceeded")
        return result

    def close(self, *, snapshot: bool = True) -> None:
        if self._closed:
            return
        try:
            if snapshot and self.stage:
                self.snapshot_phase(reason="ledger_close")
        finally:
            self.connection.close()
            self._closed = True
