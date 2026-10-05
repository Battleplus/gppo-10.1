"""Index and hash the bounded offline diagnostic without experiment calls."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN = Path(r"E:\Z博士\runs\w1-action-outcome-learning-loop-history-repair-v1-once")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    inputs = (
        RUN / "run-once" / "learning-records.jsonl",
        RUN / "run-once" / "data-units.jsonl",
        RUN / "run-once" / "prediction-evaluation.json",
        RUN / "run-once" / "status.json",
        RUN / "run-once" / "settlement.json",
        RUN / "prediction_evaluation.py",
        RUN / "learning.py",
        RUN / "learning_schema.py",
        RUN / "transparent_baselines.py",
        RUN / "experiment-matrix.json",
    )
    write_json(ROOT / "input-index.json", {
        "schema": "w1-prediction-ranking-offline-input-index/1.0.0",
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in inputs
        ],
        "excluded_unread_task_outputs": [
            "conditional task-comparison results (stage was not executed)",
            "unexecuted task-evaluation parent outcomes",
        ],
        "analysis_calls": {
            "environment": 0,
            "reset_or_step": 0,
            "checkpoint_loads": 0,
            "model_initializations": 0,
            "model_forwards": 0,
            "training_updates": 0,
        },
    })

    source = (ROOT / "analyze_ranking.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = sorted({
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    })
    allowed = {"__future__", "collections", "csv", "hashlib", "itertools", "json", "math", "pathlib"}
    checks = json.loads((ROOT / "checks.json").read_text(encoding="utf-8"))
    checks.update({
        "analysis_imports_only_standard_library": set(imports) <= allowed,
        "analysis_has_no_subprocess_execution": "subprocess" not in source,
        "analysis_has_no_torch_import": "import torch" not in source,
        "analysis_has_no_environment_runtime_import": "m10_environment" not in source,
        "sealed_prediction_gate_decision_preserved": (
            json.loads((ROOT / "diagnostic-summary.json").read_text(encoding="utf-8"))
            ["decision_preserved"] == "PREDICTION_GATE_NOT_PASSED"
        ),
    })
    if not all(checks.values()):
        raise RuntimeError(f"freeze check failed: {checks}")
    write_json(ROOT / "checks.json", checks)

    names = (
        "analyze_ranking.py",
        "candidate-evidence.csv",
        "checks.json",
        "diagnostic-report.md",
        "diagnostic-summary.json",
        "evaluation-contract.md",
        "freeze_outputs.py",
        "input-index.json",
        "missing-evidence.md",
        "parent-repeat-regret.csv",
        "parent-summary.csv",
        "window-comparison.csv",
    )
    write_json(ROOT / "hashes.json", {
        "schema": "w1-prediction-ranking-offline-hashes/1.0.0",
        "files": [
            {"path": name, "bytes": (ROOT / name).stat().st_size, "sha256": sha256(ROOT / name)}
            for name in names
        ],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
