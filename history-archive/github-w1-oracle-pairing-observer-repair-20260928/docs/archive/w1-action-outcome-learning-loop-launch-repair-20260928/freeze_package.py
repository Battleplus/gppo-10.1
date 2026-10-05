"""Seal the runner-ready package without executing any experiment calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ATTEMPT = "w1-action-outcome-learning-loop-launch-repair-v1-once"
EXCLUDED = {
    "diagnostic-records.jsonl",
    "public-contexts.jsonl",
    "execution-manifest.json",
    "hashes.json",
    "export-output.txt",
    "test-output.txt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def input_index() -> dict:
    sources = [
        Path(r"E:\Z博士\research-plans\w1-action-outcome-prior-prototype-v1\execution-manifest.json"),
        Path(r"E:\Z博士\runs\w1-action-outcome-learning-loop-v1-once\technical-stop.json"),
        Path(r"E:\Z博士\research-plans\w1-light-repaired-fair-rerun-infra-repair-v1\execution-manifest.json"),
        Path(r"E:\Z博士\research-plans\w1-action-consequence-oracle-remaining-23-preparation-v1\execution-manifest.json"),
        Path(r"E:\Z博士\runs\w1-action-consequence-oracle-remaining-23-v1-once\run-once\analysis.json"),
        ROOT / "diagnostic-records.jsonl",
        ROOT / "public-contexts.jsonl",
    ]
    return {
        "schema": "w1-action-outcome-learning-launch-repair-input-index/1.0.0",
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sources
        ],
        "existing_label_interpretation": {
            "factual_executed_labels": 774,
            "fixed_hungarian_oracle_branches": 144,
            "admitted_to_new_training_split": 0,
            "reason": "development provenance and continuation/split boundaries; labels remain real",
        },
        "preparation_calls": {
            "environment": 0,
            "resets": 0,
            "model_initializations": 0,
            "model_forwards": 0,
            "checkpoint_loads": 0,
            "training_updates": 0,
        },
    }


def included_files() -> list[Path]:
    allowed_suffixes = {".py", ".json", ".md", ".txt"}
    result = []
    for path in ROOT.iterdir():
        if not path.is_file() or path.name in EXCLUDED or path.suffix.lower() not in allowed_suffixes:
            continue
        result.append(path)
    return sorted(result, key=lambda path: path.name)


def main() -> int:
    write_json(ROOT / "input-index.json", input_index())
    files = {path.name: sha256(path) for path in included_files()}
    manifest = {
        "schema": "w1-action-outcome-learning-loop-launch-repair-manifest/1.0.0",
        "attempt": ATTEMPT,
        "status": "RUNNER_READY_NOT_APPROVED",
        "resource_request_status": "NOT_APPROVED",
        "unique_entry": "launch_once.py",
        "wsl_stage_entry": "wsl_stage_and_launch.py",
        "native_entry": "native_launch.py",
        "worker_entry": "runner.py",
        "preparation_consumption": input_index()["preparation_calls"],
        "files": files,
    }
    write_json(ROOT / "execution-manifest.json", manifest)
    manifest_hash = sha256(ROOT / "execution-manifest.json")
    write_json(ROOT / "hashes.json", {
        "schema": "w1-action-outcome-learning-loop-launch-repair-hashes/1.0.0",
        "execution_manifest_sha256": manifest_hash,
        "files": [
            {"path": name, "bytes": (ROOT / name).stat().st_size, "sha256": digest}
            for name, digest in files.items()
        ],
        "excluded_large_local_evidence": [
            {
                "path": name,
                "bytes": (ROOT / name).stat().st_size,
                "sha256": sha256(ROOT / name),
                "github_included": False,
            }
            for name in ("diagnostic-records.jsonl", "public-contexts.jsonl")
        ],
        "nonstaged_test_evidence": (
            {
                "path": "test-output.txt",
                "bytes": (ROOT / "test-output.txt").stat().st_size,
                "sha256": sha256(ROOT / "test-output.txt"),
            }
            if (ROOT / "test-output.txt").is_file()
            else None
        ),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
