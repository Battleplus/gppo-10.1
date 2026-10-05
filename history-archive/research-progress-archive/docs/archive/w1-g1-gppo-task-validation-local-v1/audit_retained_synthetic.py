"""Read-only verification of retained synthetic accounting; no torch imports."""
import hashlib
import json
from pathlib import Path
import sqlite3

root = Path(__file__).resolve().parent
evidence = root / "controlled-synthetic-task-pipeline-evidence-isolated-20261004"
run = evidence / "run-once"
def read(p):
    return json.loads(p.read_text(encoding="utf-8"))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
routes = [json.loads(line) for line in (run / "policy-training-routes.jsonl").read_text(encoding="utf-8").splitlines()]
with sqlite3.connect((run / "budget.sqlite3").as_uri() + "?mode=ro", uri=True) as connection:
    tables = connection.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall()
    columns = connection.execute("PRAGMA table_info(calls)").fetchall()
    replay = connection.execute("SELECT name,status,amounts FROM calls WHERE name LIKE '%recurrent_hidden_replay%'").fetchall()
by_method = {}
for method in ("G0", "T", "G1"):
    relevant = [r for r in routes if r["method"] == method]
    actual = [r for r in replay if r[0] == method.lower() + "_recurrent_hidden_replay"]
    observations = sum(r["summary"]["recurrent_hidden_replay_observations"] for r in relevant)
    charges = sum(json.loads(r[2])["encode_sample_evaluations"] for r in actual)
    by_method[method] = {"route_count": len(relevant), "replayed_observations": observations,
                         "ledger_calls": len(actual), "ledger_encode_charges": charges,
                         "all_complete": all(r[1] == "complete" for r in actual),
                         "match": observations == charges == len(actual)}
loaded_source = {}
for p in (evidence / "fixture-package").rglob("*.py"):
    relative = p.relative_to(evidence / "fixture-package").as_posix()
    if sha(p) != sha(root / "package" / relative):
        raise RuntimeError("RETAINED_SOURCE_DIFF:" + relative)
    loaded_source[relative] = sha(p)
checkpoints = read(run / "policy-training-summary.json")["routes"]
checkpoint_checks = {}
for row in checkpoints:
    path = run / "policy-checkpoints" / Path(row["checkpoint"]).name
    checkpoint_checks[path.name] = sha(path) == row["checkpoint_sha256"]
value = {"schema": "w1-retained-synthetic-read-only-audit/1.0.0", "imports_torch": False,
         "model_or_environment_calls": 0, "sqlite_open_mode": "ro",
         "per_method_hidden_replay": by_method, "checkpoint_bytes_match": checkpoint_checks,
         "fixture_source_file_count": len(loaded_source), "all_fixture_python_matches_delivery": True,
         "fixture_source_sha256": loaded_source,
         "success": all(v["match"] and v["all_complete"] for v in by_method.values())
                    and all(checkpoint_checks.values())}
(root / "retained-synthetic-independent-audit.json").write_text(
    json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in value.items() if k != "fixture_source_sha256"}, ensure_ascii=False))
