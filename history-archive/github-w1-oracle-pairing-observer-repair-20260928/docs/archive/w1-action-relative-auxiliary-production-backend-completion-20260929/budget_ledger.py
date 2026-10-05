"""SQLite call ledger with pending-call preservation and stage hard caps."""

from __future__ import annotations

import json
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
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY, stage TEXT, name TEXT, amounts TEXT, status TEXT, started REAL, finished REAL, error TEXT)"
        )
        self.connection.commit()

    def select(self, stage: str) -> None:
        if stage not in self.request["stage_limits"]:
            raise BudgetError(f"unknown stage {stage}")
        self.stage = stage

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
        limits = self.request["stage_limits"][self.stage]
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
        self.connection.close()
