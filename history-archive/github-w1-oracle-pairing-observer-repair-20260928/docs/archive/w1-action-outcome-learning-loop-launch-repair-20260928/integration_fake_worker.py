"""Zero-environment integration worker for the real Windows-to-WSL launch path."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "run-once"


def main() -> int:
    contract = json.loads((ROOT / "launch-contract.json").read_text(encoding="utf-8"))
    if contract.get("integration_fault") == "native_interrupt":
        time.sleep(30)
    OUT.mkdir(parents=False, exist_ok=False)
    (OUT / "activity.json").write_text(
        json.dumps({"stage": "staging_and_zero_step_gate", "run": "integration-fake-worker"}) + "\n",
        encoding="utf-8",
    )
    database = OUT / "fake-ledger.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE calls(id INTEGER PRIMARY KEY, state TEXT NOT NULL)")
    connection.commit()
    observed: list[int] = []

    def reader() -> None:
        for _ in range(20):
            with sqlite3.connect(database) as handle:
                observed.append(int(handle.execute("SELECT COUNT(*) FROM calls").fetchone()[0]))
            time.sleep(0.005)

    thread = threading.Thread(target=reader)
    thread.start()
    for index in range(12):
        connection.execute("INSERT INTO calls(state) VALUES (?)", ("pending",))
        connection.commit()
        connection.execute("UPDATE calls SET state='completed' WHERE id=?", (index + 1,))
        connection.commit()
    thread.join()
    pending = int(connection.execute("SELECT COUNT(*) FROM calls WHERE state='pending'").fetchone()[0])
    completed = int(connection.execute("SELECT COUNT(*) FROM calls WHERE state='completed'").fetchone()[0])
    connection.close()
    (OUT / "status.json").write_text(json.dumps({
        "status": "complete",
        "fake_worker": True,
        "environment_calls": 0,
        "model_calls": 0,
        "training_updates": 0,
        "worker_invocations": 1,
        "sqlite_pending": pending,
        "sqlite_completed": completed,
        "concurrent_reads": len(observed),
        "concurrent_read_max": max(observed),
        "automatic_retry": False,
    }, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
