"""Freeze the launch-wiring revision without running any dynamic stage."""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ATTEMPT = "w1-action-relative-auxiliary-launch-wiring-v1-once"
TOKEN_FILE = Path(r"E:\Z博士\.codex-tmp\w1-action-relative-auxiliary-launch-wiring-v1.token")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", default=str(TOKEN_FILE))
    args = parser.parse_args()
    token_path = Path(args.token_file)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    if not token_path.exists():
        token_path.write_text(secrets.token_urlsafe(32), encoding="utf-8")
    token_hash = hashlib.sha256(token_path.read_text(encoding="utf-8").strip().encode()).hexdigest()

    request = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
    request["attempt"] = ATTEMPT
    request["status"] = "NOT_APPROVED"
    request["stages"] = request["stage_limits"]
    request["shutdown_reserve"] = {
        "artifact_bytes_within_active_limit": 256 * 1024 ** 2,
        "complete_process_cpu_seconds_within_each_dynamic_stage": 240,
        "wall_seconds_within_each_dynamic_stage": 60,
    }
    request["derived_dynamic_budget"] = json.loads((ROOT / "derived-budget.json").read_text(encoding="utf-8"))
    request["progression"]["task_comparison"] = "run only when the A/B effect gate passes and B is strictly better than transparent history; otherwise task calls remain zero"
    write_json(ROOT / "RESOURCE_REQUEST.json", request)

    matrix = {
        "schema": "w1-action-relative-auxiliary-execution-matrix/1.0.0",
        "groups": {"train": [], "model_selection": [], "prediction_evaluation": [], "task_evaluation": []},
        "data_repeats": {"train": 1, "model_selection": 1, "prediction_evaluation": 3},
        "task_repeats": 3,
        "task_methods": ["hungarian", "transparent_one_shot", "A_one_shot", "B_one_shot"],
        "candidate_contract": {"max_non_noop": 24, "noop_allowed": True, "max_total": 25},
    }
    selection = json.loads((ROOT / "new-prediction-parent-selection.json").read_text(encoding="utf-8"))
    matrix["groups"]["prediction_evaluation"] = selection["parents"]
    write_json(ROOT / "experiment-matrix.json", matrix)

    contract = json.loads((ROOT / "launch-contract.json").read_text(encoding="utf-8"))
    contract["attempt"] = ATTEMPT
    contract["external_token_sha256"] = token_hash
    contract["windows_source_root"] = str(ROOT)
    contract["wsl_source_root"] = "/mnt/e/Z博士/research-plans/w1-action-relative-auxiliary-launch-wiring-v1"
    contract["native_execution_root"] = f"/home/asus/{ATTEMPT}"
    contract["windows_export_root"] = f"/mnt/e/Z博士/.codex-exports/{ATTEMPT}"
    contract["authorization_status"] = "NOT_APPROVED"
    contract["one_shot"] = True
    contract["resource_request_sha256"] = sha256(ROOT / "RESOURCE_REQUEST.json")
    write_json(ROOT / "launch-contract.json", contract)

    excluded = {"hashes.json", "execution-manifest.json"}
    files = []
    for path in sorted(ROOT.iterdir()):
        if not path.is_file() or path.name in excluded or path.name.endswith(".pyc"):
            continue
        files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    hashes = {"schema": "w1-action-relative-auxiliary-launch-wiring-hashes/1.0.0", "files": files}
    write_json(ROOT / "hashes.json", hashes)
    manifest_files = {row["path"]: row["sha256"] for row in files}
    manifest = {"schema": "w1-action-relative-auxiliary-launch-wiring-manifest/1.0.0", "attempt": ATTEMPT, "status": "NOT_APPROVED", "package_dir": str(ROOT), "hashes_sha256": sha256(ROOT / "hashes.json"), "files": manifest_files, "frozen_entrypoint": "launch_once.py", "external_token_env": "W1_ACTION_RELATIVE_AUXILIARY_TOKEN"}
    write_json(ROOT / "execution-manifest.json", manifest)
    print(json.dumps({"attempt": ATTEMPT, "token_file": str(token_path), "execution_manifest_sha256": sha256(ROOT / "execution-manifest.json"), "hashes_sha256": sha256(ROOT / "hashes.json"), "status": "NOT_APPROVED"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
