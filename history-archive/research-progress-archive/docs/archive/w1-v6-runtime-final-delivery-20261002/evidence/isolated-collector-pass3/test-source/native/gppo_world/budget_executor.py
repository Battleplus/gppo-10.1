"""SQLite-backed persistent, cross-attempt budget execution."""

from __future__ import annotations

from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import uuid
from typing import Any


class BudgetExhausted(RuntimeError):
    pass


class BudgetStateConflict(RuntimeError):
    pass


class PersistentBudget:
    """Transactional budget ledger shared across attempts and processes."""

    SCHEMA = "persistent-budget/2.0.0"
    DB_SCHEMA = "persistent-budget-sqlite/1.0.0"

    def __init__(self, path: str | Path, *, limits: dict[str, int], attempt_id: str,
                 run_id: str | None = None, run_limits: dict[str, int] | None = None,
                 legacy_run_map: dict[str, str] | None = None,
                 require_existing: bool = False, lock_timeout: float = 5.0):
        supplied = Path(path)
        self.snapshot_path = supplied if supplied.suffix.lower() != ".sqlite3" else supplied.with_suffix(".json")
        self.db_path = supplied if supplied.suffix.lower() == ".sqlite3" else supplied.with_suffix(".sqlite3")
        self.path = self.db_path
        self.limits = {str(k): int(v) for k, v in limits.items()}
        self.attempt_id = str(attempt_id)
        self.run_id = None if run_id is None else str(run_id)
        self.run_limits = None if run_limits is None else {str(k): int(v) for k, v in run_limits.items()}
        self.lock_timeout = float(lock_timeout)
        if require_existing and not self.db_path.exists():
            raise FileNotFoundError(f"required existing budget database not found: {self.db_path}")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.db_path.exists():
            self._validate_existing_limits()
            self._ensure_run_columns(legacy_run_map or {})
        elif self.snapshot_path.exists():
            self._migrate_legacy_json_once()
            self._ensure_run_columns(legacy_run_map or {})
        else:
            self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=self.lock_timeout,
                                     isolation_level=None, check_same_thread=False)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={max(1, int(self.lock_timeout * 1000))}")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @staticmethod
    def _schema_sql() -> tuple[str, ...]:
        return (
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS stages (stage TEXT PRIMARY KEY, limit_amount INTEGER NOT NULL, reserved INTEGER NOT NULL, verified INTEGER NOT NULL, unknown INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS attempts (attempt_id TEXT NOT NULL, stage TEXT NOT NULL, reserved INTEGER NOT NULL, verified INTEGER NOT NULL, unknown INTEGER NOT NULL, run_id TEXT, PRIMARY KEY (attempt_id, stage))",
            "CREATE TABLE IF NOT EXISTS reservations (reservation_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL, stage TEXT NOT NULL, amount INTEGER NOT NULL, status TEXT NOT NULL CHECK(status IN ('pending','verified','unknown')), reservation_index INTEGER, created_at REAL NOT NULL, finished_at REAL, reason TEXT, run_id TEXT)",
            "CREATE TABLE IF NOT EXISTS history (seq INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT NOT NULL, stage TEXT NOT NULL, amount INTEGER NOT NULL, attempt_id TEXT NOT NULL, reservation_id TEXT, reservation_index INTEGER, pid INTEGER, reason TEXT, created_at REAL NOT NULL, run_id TEXT)",
            "CREATE INDEX IF NOT EXISTS idx_history_attempt ON history(attempt_id, seq)",
            "CREATE INDEX IF NOT EXISTS idx_reservations_attempt ON reservations(attempt_id, stage)",
        )

    def _initialize_database(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            for statement in self._schema_sql():
                connection.execute(statement)
            connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", ("schema", self.DB_SCHEMA))
            connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", ("json_schema", self.SCHEMA))
            for stage, limit_amount in self.limits.items():
                connection.execute(
                    "INSERT INTO stages(stage,limit_amount,reserved,verified,unknown) VALUES (?,?,0,0,0)",
                    (stage, limit_amount),
                )
            if self.run_limits is not None:
                connection.execute(
                    "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
                    ("run_limits", json.dumps(self.run_limits, sort_keys=True)),
                )
            connection.commit()

    def _validate_existing_limits(self) -> None:
        with closing(self._connect()) as connection:
            schema = connection.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()
            if schema is None or schema[0] != self.DB_SCHEMA:
                raise ValueError(f"unknown persistent budget database schema: {schema}")
            rows = dict(connection.execute("SELECT stage,limit_amount FROM stages").fetchall())
            if rows != self.limits:
                raise ValueError(f"budget limits changed across attempts: {rows} != {self.limits}")

    def _ensure_run_columns(self, legacy_run_map: dict[str, str]) -> None:
        """Add stable run identity to an already migrated v2 database once."""
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            for table in ("attempts", "reservations", "history"):
                columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
                if "run_id" not in columns:
                    connection.execute(f"ALTER TABLE {table} ADD COLUMN run_id TEXT")
            if self.run_limits is not None:
                connection.execute(
                    "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
                    ("run_limits", json.dumps(self.run_limits, sort_keys=True)),
                )
            if legacy_run_map:
                connection.execute(
                    "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
                    ("legacy_run_map", json.dumps(legacy_run_map, sort_keys=True)),
                )
            mapping_payload = connection.execute(
                "SELECT value FROM metadata WHERE key='legacy_run_map'"
            ).fetchone()
            mapping = legacy_run_map.copy()
            if mapping_payload is not None:
                mapping.update(json.loads(mapping_payload[0]))
            for attempt_id, mapped_run in mapping.items():
                connection.execute("UPDATE attempts SET run_id=? WHERE attempt_id=?", (mapped_run, attempt_id))
                connection.execute("UPDATE reservations SET run_id=? WHERE attempt_id=?", (mapped_run, attempt_id))
                connection.execute("UPDATE history SET run_id=? WHERE attempt_id=?", (mapped_run, attempt_id))
            # Historical rows not covered by an explicit map remain auditable,
            # but cannot silently count toward a new stable run.
            connection.execute(
                "UPDATE attempts SET run_id=COALESCE(run_id,'unmapped:'||attempt_id) WHERE run_id IS NULL"
            )
            connection.execute(
                "UPDATE reservations SET run_id=COALESCE(run_id,'unmapped:'||attempt_id) WHERE run_id IS NULL"
            )
            connection.execute(
                "UPDATE history SET run_id=COALESCE(run_id,'unmapped:'||attempt_id) WHERE run_id IS NULL"
            )
            connection.commit()

    def _effective_run_id(self, run_id: str | None) -> str:
        value = run_id if run_id is not None else (self.run_id or self.attempt_id)
        return str(value)

    def _stored_run_limits(self) -> dict[str, int]:
        if self.run_limits is not None:
            return self.run_limits
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT value FROM metadata WHERE key='run_limits'").fetchone()
            if row is None:
                # Legacy v2 acceptance databases had no run-level contract;
                # keep their historical behavior for read-only compatibility.
                return dict(self.limits)
            return {str(k): int(v) for k, v in json.loads(row[0]).items()}

    def _migrate_legacy_json_once(self) -> None:
        payload = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        if payload.get("schema") != "persistent-budget/1.0.0":
            raise ValueError(f"unsupported legacy budget schema: {payload.get('schema')}")
        if payload.get("limits") != self.limits:
            raise ValueError(f"budget limits changed during migration: {payload.get('limits')} != {self.limits}")
        source_hash = hashlib.sha256(self.snapshot_path.read_bytes()).hexdigest()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            for statement in self._schema_sql():
                connection.execute(statement)
            metadata = {
                "schema": self.DB_SCHEMA,
                "json_schema": self.SCHEMA,
                "legacy_source_path": str(self.snapshot_path),
                "legacy_source_sha256": source_hash,
                "migrated_at": str(time.time()),
            }
            for key, value in metadata.items():
                connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", (key, value))
            for stage, values in payload["stages"].items():
                connection.execute(
                    "INSERT INTO stages(stage,limit_amount,reserved,verified,unknown) VALUES (?,?,?,?,?)",
                    (stage, int(payload["limits"][stage]), int(values["reserved"]), int(values["verified"]), int(values["unknown"])),
                )
            for attempt_id, by_stage in payload.get("attempts", {}).items():
                for stage, values in by_stage.items():
                    connection.execute(
                        "INSERT INTO attempts(attempt_id,stage,reserved,verified,unknown) VALUES (?,?,?,?,?)",
                        (attempt_id, stage, int(values["reserved"]), int(values["verified"]), int(values["unknown"])),
                    )
            reservations: dict[int, str] = {}
            for item in payload.get("history", []):
                index = item.get("reservation_index")
                reservation_id = None if index is None else f"legacy-{int(index)}"
                if item.get("event") == "reserve" and reservation_id is not None:
                    reservations[int(index)] = reservation_id
                    connection.execute(
                        "INSERT INTO reservations(reservation_id,attempt_id,stage,amount,status,reservation_index,created_at) VALUES (?,?,?,?,?,?,?)",
                        (reservation_id, item["attempt_id"], item["stage"], int(item["amount"]), "pending", int(index), time.time()),
                    )
                connection.execute(
                    "INSERT INTO history(event,stage,amount,attempt_id,reservation_id,reservation_index,pid,reason,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (item["event"], item["stage"], int(item["amount"]), item["attempt_id"], reservation_id,
                     None if index is None else int(index), item.get("pid"), item.get("reason"), time.time()),
                )
                if item.get("event") in {"verified", "unknown"} and reservation_id is not None:
                    connection.execute(
                        "UPDATE reservations SET status=?,finished_at=?,reason=? WHERE reservation_id=?",
                        (item["event"], time.time(), item.get("reason"), reservation_id),
                    )
            connection.commit()

    @staticmethod
    def _attempt_row(connection: sqlite3.Connection, attempt_id: str, stage: str) -> tuple[int, int, int]:
        row = connection.execute(
            "SELECT reserved,verified,unknown FROM attempts WHERE attempt_id=? AND stage=?", (attempt_id, stage)
        ).fetchone()
        return (0, 0, 0) if row is None else tuple(int(value) for value in row)

    def reserve(self, stage: str, amount: int = 1, *, run_id: str | None = None) -> dict[str, Any]:
        stage = str(stage)
        amount = int(amount)
        if stage not in self.limits or amount < 1:
            raise ValueError(f"unknown budget stage or amount: {stage}, {amount}")
        run_id_value = self._effective_run_id(run_id)
        run_limits = self._stored_run_limits()
        if stage not in run_limits:
            raise ValueError(f"no per-run limit configured for stage: {stage}")
        reservation_id = uuid.uuid4().hex
        now = time.time()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT reserved,limit_amount FROM stages WHERE stage=?", (stage,)).fetchone()
            if row is None:
                connection.rollback()
                raise ValueError(f"unknown budget stage: {stage}")
            reserved, limit_amount = int(row[0]), int(row[1])
            if reserved + amount > limit_amount:
                connection.rollback()
                raise BudgetExhausted(f"budget exhausted for {stage}: {reserved}+{amount}>{limit_amount}")
            run_reserved = int(connection.execute(
                "SELECT COALESCE(SUM(amount),0) FROM reservations WHERE run_id=? AND stage=?",
                (run_id_value, stage),
            ).fetchone()[0])
            if run_reserved + amount > int(run_limits[stage]):
                connection.rollback()
                raise BudgetExhausted(
                    f"per-run budget exhausted for {run_id_value}/{stage}: "
                    f"{run_reserved}+{amount}>{run_limits[stage]}"
                )
            connection.execute("UPDATE stages SET reserved=reserved+? WHERE stage=?", (amount, stage))
            old_reserved, old_verified, old_unknown = self._attempt_row(connection, self.attempt_id, stage)
            connection.execute(
                "INSERT INTO attempts(attempt_id,stage,reserved,verified,unknown,run_id) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(attempt_id,stage) DO UPDATE SET reserved=excluded.reserved, run_id=excluded.run_id",
                (self.attempt_id, stage, old_reserved + amount, old_verified, old_unknown, run_id_value),
            )
            connection.execute(
                "INSERT INTO reservations(reservation_id,attempt_id,stage,amount,status,created_at,run_id) VALUES (?,?,?,?,?,?,?)",
                (reservation_id, self.attempt_id, stage, amount, "pending", now, run_id_value),
            )
            cursor = connection.execute(
                "INSERT INTO history(event,stage,amount,attempt_id,reservation_id,pid,created_at,run_id) VALUES (?,?,?,?,?,?,?,?)",
                ("reserve", stage, amount, self.attempt_id, reservation_id, os.getpid(), now, run_id_value),
            )
            reservation_index = int(cursor.lastrowid)
            connection.execute("UPDATE reservations SET reservation_index=? WHERE reservation_id=?", (reservation_index, reservation_id))
            connection.commit()
        return {"stage": stage, "amount": amount, "attempt_id": self.attempt_id,
                "reservation_id": reservation_id, "reservation_index": reservation_index}

    def complete(self, token: dict[str, Any]) -> None:
        self._finish(token, "verified")

    def unknown(self, token: dict[str, Any], reason: str) -> None:
        self._finish(token, "unknown", reason=str(reason))

    def _finish(self, token: dict[str, Any], outcome: str, **extra: Any) -> None:
        if outcome not in {"verified", "unknown"}:
            raise ValueError(outcome)
        reservation_id = str(token.get("reservation_id", ""))
        if not reservation_id:
            raise ValueError("reservation token has no reservation_id")
        now = time.time()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT attempt_id,stage,amount,status,reservation_index FROM reservations WHERE reservation_id=?",
                (reservation_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise ValueError(f"unknown reservation_id: {reservation_id}")
            attempt_id, stage, amount, status, reservation_index = row
            if status == outcome:
                connection.commit()
                return
            if status != "pending":
                connection.rollback()
                raise BudgetStateConflict(f"reservation {reservation_id} already {status}, cannot mark {outcome}")
            connection.execute("UPDATE reservations SET status=?,finished_at=?,reason=? WHERE reservation_id=?",
                               (outcome, now, extra.get("reason"), reservation_id))
            connection.execute(f"UPDATE stages SET {outcome}={outcome}+? WHERE stage=?", (int(amount), stage))
            old_reserved, old_verified, old_unknown = self._attempt_row(connection, attempt_id, stage)
            values = {"verified": old_verified, "unknown": old_unknown}
            values[outcome] += int(amount)
            connection.execute("UPDATE attempts SET verified=?,unknown=? WHERE attempt_id=? AND stage=?",
                               (values["verified"], values["unknown"], attempt_id, stage))
            connection.execute(
                "INSERT INTO history(event,stage,amount,attempt_id,reservation_id,reservation_index,pid,reason,created_at,run_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (outcome, stage, int(amount), attempt_id, reservation_id, reservation_index, os.getpid(), extra.get("reason"), now,
                 connection.execute("SELECT run_id FROM reservations WHERE reservation_id=?", (reservation_id,)).fetchone()[0]),
            )
            connection.commit()

    def snapshot(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN")
            stages = {row[0]: {"reserved": int(row[2]), "verified": int(row[3]), "unknown": int(row[4]),
                                "pending": int(row[2]) - int(row[3]) - int(row[4])}
                      for row in connection.execute("SELECT stage,limit_amount,reserved,verified,unknown FROM stages ORDER BY stage")}
            limits = {row[0]: int(row[1]) for row in connection.execute("SELECT stage,limit_amount FROM stages ORDER BY stage")}
            attempts: dict[str, dict[str, dict[str, int]]] = {}
            for attempt_id, stage, reserved, verified, unknown, run_id in connection.execute(
                    "SELECT attempt_id,stage,reserved,verified,unknown,run_id FROM attempts ORDER BY attempt_id,stage"):
                attempts.setdefault(attempt_id, {})[stage] = {
                    "reserved": int(reserved), "verified": int(verified), "unknown": int(unknown),
                    "pending": int(reserved) - int(verified) - int(unknown), "run_id": run_id,
                }
            history = []
            for event, stage, amount, attempt_id, reservation_id, reservation_index, pid, reason, run_id in connection.execute(
                    "SELECT event,stage,amount,attempt_id,reservation_id,reservation_index,pid,reason,run_id FROM history ORDER BY seq"):
                item = {"event": event, "stage": stage, "amount": int(amount), "attempt_id": attempt_id,
                        "reservation_id": reservation_id, "reservation_index": reservation_index, "pid": pid,
                        "run_id": run_id}
                if reason is not None:
                    item["reason"] = reason
                history.append(item)
            connection.commit()
            return {"schema": self.SCHEMA, "db_schema": self.DB_SCHEMA, "db_path": str(self.db_path),
                    "limits": limits, "stages": stages, "attempts": attempts, "history": history}

    def export_snapshot(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.snapshot_path
        target.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self.snapshot(), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        with tempfile.NamedTemporaryFile(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        return target

    def run_totals(self, run_id: str) -> dict[str, dict[str, int]]:
        """Return one run's reserved/verified/unknown/pending totals atomically."""
        with closing(self._connect()) as connection:
            connection.execute("BEGIN")
            rows = connection.execute(
                "SELECT stage, COALESCE(SUM(amount),0), "
                "COALESCE(SUM(CASE WHEN status='verified' THEN amount ELSE 0 END),0), "
                "COALESCE(SUM(CASE WHEN status='unknown' THEN amount ELSE 0 END),0) "
                "FROM reservations WHERE run_id=? GROUP BY stage ORDER BY stage",
                (str(run_id),),
            ).fetchall()
            result = {}
            for stage, reserved, verified, unknown in rows:
                pending = int(reserved) - int(verified) - int(unknown)
                result[stage] = {"reserved": int(reserved), "verified": int(verified),
                                 "unknown": int(unknown), "pending": pending}
            connection.commit()
            return result

    def integrity_check(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            result = connection.execute("PRAGMA integrity_check").fetchone()[0]
            pending = int(connection.execute("SELECT COUNT(*) FROM reservations WHERE status='pending'").fetchone()[0])
            return {"integrity_check": result, "pending_reservations": pending, "db_path": str(self.db_path)}
