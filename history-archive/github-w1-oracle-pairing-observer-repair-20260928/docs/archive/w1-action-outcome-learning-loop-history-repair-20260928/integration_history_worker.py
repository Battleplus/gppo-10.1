"""Production collector with a controlled fake environment; no real model or training."""

from __future__ import annotations

import json
from pathlib import Path

from history_regression_harness import build_backend
from learning_schema import load_records


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "run-once"


def main() -> int:
    (ROOT / "integration-worker-started.json").write_text(
        json.dumps({"worker": "history-regression", "real_environment_calls": 0}) + "\n",
        encoding="utf-8",
    )
    backend = build_backend(ROOT, OUT)
    data_path = backend.collect_fixed_continuation_labels()
    records = load_records(data_path)
    first = records[0]
    status = {
        "status": "complete",
        "fake_environment": True,
        "real_environment_calls": 0,
        "model_initializations": 0,
        "model_forwards": 0,
        "training_updates": 0,
        "persisted_labels": len(records),
        "first_label_validated": True,
        "first_decision_id": first.decision_id,
        "first_public_state_shared": all(
            row.flat == first.flat
            and row.history == first.history
            and row.legal_actions == first.legal_actions
            for row in records
            if row.decision_id == first.decision_id
        ),
        "ledger": backend.ledger.assert_settled(),
        "automatic_retry": False,
    }
    (OUT / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    backend.ledger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
