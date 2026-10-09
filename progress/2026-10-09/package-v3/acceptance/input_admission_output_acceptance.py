"""Targeted Linux acceptance for the input_admission output-write contract.

Uses the real recovery Driver, BudgetLedger and StageServer. The existing
288-window cache is read-only; no collector, model, forward, GPPO or task
evaluation is called.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "package"
OUT = Path(os.environ.get("W1_INPUT_ADMISSION_ACCEPTANCE_OUTPUT",
                         str(ROOT / "acceptance-evidence" / "input-admission-output-v3")))
os.sched_setaffinity(0, {0})
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                  MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                  PYTHONDONTWRITEBYTECODE="1")
import sys
sys.path[:0] = [str(PACKAGE), str(PACKAGE / "native")]

from budget_ledger import BudgetError, BudgetLedger
from infra_io import durable_atomic_json, sha256_file
from linux_process_scope import proc_tree
from phase_handshake import StageServer
from production_driver import Driver


def request_for(*, output_writes: int) -> dict:
    request = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
    resources = (
        "action_samples", "active_storage_bytes", "all_resident_rss_bytes",
        "analysis_calls", "behavior_snapshots", "bootstrap_forwards",
        "candidate_branches", "candidate_contract_audits", "candidate_scans",
        "candidate_snapshot_probes", "checkpoint_loads", "checkpoint_writes",
        "environment_constructions", "environment_resets", "environment_steps",
        "forced_first_actions", "model_initializations", "output_writes",
        "policy_forwards", "policy_updates", "replay_rows", "reused_windows",
        "snapshot_writes", "target_candidate_evaluations", "target_forwards",
        "task_episodes", "training_candidate_evaluations", "training_forwards",
        "update_rows", "world_model_batch_forwards", "world_model_candidate_evaluations",
        "world_model_updates",
    )
    for stage in request["stages"].values():
        for key in resources:
            stage[key] = 0
        stage["all_resident_rss_bytes"] = 4 * 2**30
        stage["active_storage_bytes"] = 512 * 2**20
        stage["wall_seconds"] = 900
        stage["complete_process_cpu_seconds"] = 600
    first = request["stage_order"][0]
    request["stages"][first].update(output_writes=output_writes,
                                    reused_windows=288,
                                    candidate_contract_audits=2000)
    request["totals"] = {key: 0 for key in resources}
    request["totals"].update(output_writes=output_writes,
                              reused_windows=288,
                              candidate_contract_audits=2000,
                              all_resident_rss_bytes=4 * 2**30,
                              active_storage_bytes=512 * 2**20,
                              complete_process_cpu_seconds=600,
                              wall_seconds=900)
    request["attempt"] = "engineering-input-admission-output-v3"
    request["approval_status"] = "ENGINEERING_ONLY"
    return request


class Handshake:
    def __init__(self, output: Path, request: dict):
        self.output, self.request = output, request
        self.errors = []
        self.server = None
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        own = resource.getrusage(resource.RUSAGE_SELF)
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.server = StageServer(
            self.output / "phase.sock", self.request, tree_reader=proc_tree,
            cpu_origin_seconds=(own.ru_utime + own.ru_stime + child.ru_utime + child.ru_stime),
            wall_origin_monotonic=time.monotonic())
        os.environ["W1_PHASE_ACCOUNTING_SOCKET"] = str(self.server.path)

        def serve():
            while not self.stop.is_set():
                try:
                    self.server.process_pending(os.getpid())
                except BaseException as exc:
                    self.errors.append(repr(exc))
                self.stop.wait(.001)

        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()
        return self.server

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=6)
        self.server.close()
        os.environ.pop("W1_PHASE_ACCOUNTING_SOCKET", None)
        durable_atomic_json(self.output / "handshake-result.json",
                            {"errors": self.errors, "rows": self.server.rows})
        if self.errors or self.thread.is_alive():
            raise AssertionError("STAGE_SERVER_HANDSHAKE_FAILURE")


def setup_workspace(name: str, request: dict):
    workspace = OUT / name
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    matrix = json.loads((PACKAGE / "experiment-matrix.json").read_text(encoding="utf-8"))
    matrix.update(controlled=True, attempt=request["attempt"])
    # Driver only needs the existing immutable recovery inputs and environment config.
    durable_atomic_json(workspace / "RESOURCE_REQUEST.json", request)
    durable_atomic_json(workspace / "experiment-matrix.json", matrix)
    (workspace / "recovery-inputs").symlink_to(PACKAGE / "recovery-inputs", target_is_directory=True)
    shutil.copy2(PACKAGE / "parent-split.json", workspace / "parent-split.json")
    shutil.copy2(PACKAGE / "environment-config-contract.json", workspace / "environment-config-contract.json")
    return workspace


def reservations(output: Path):
    import sqlite3
    with sqlite3.connect(output / "ledger.sqlite") as db:
        return db.execute("SELECT stage,resource,units FROM reservations ORDER BY stage,resource").fetchall()


def normal_case():
    request = request_for(output_writes=3)
    workspace = setup_workspace("normal", request)
    output = workspace / "run-once"
    output.mkdir()
    with Handshake(output, request):
        ledger = BudgetLedger(output / "ledger.sqlite", request)
        driver = Driver(workspace, output, json.loads((workspace / "experiment-matrix.json").read_text()), ledger)
        driver.phase("input_admission")
        lean = driver.collect()
        inventory = json.loads((output / "reused-source-inventory.json").read_text())
        lean_payload = (output / "lean-inputs.json").read_bytes()
        assert len(inventory["windows"]) == 288
        assert len(lean) == 174
        assert len(json.loads(lean_payload)["no_opportunity"]) == 114
        before = sha256_file(output / "lean-inputs.json")
        driver.save("lean-inputs.json", json.loads(lean_payload))
        assert sha256_file(output / "lean-inputs.json") == before
        rows = reservations(output)
        assert dict((resource, units) for stage, resource, units in rows
                     if stage == "input_admission")["output_writes"] == 3
        driver.phase("world_training_and_restore")
        ledger.close()
    return {"windows": 288, "complete": 174, "no_opportunity": 114,
            "lean_sha256": before, "next_stage_reached": True,
            "formal_output_writes": 3}


def insufficient_case():
    request = request_for(output_writes=2)
    workspace = setup_workspace("insufficient", request)
    output = workspace / "run-once"
    output.mkdir()
    with Handshake(output, request):
        ledger = BudgetLedger(output / "ledger.sqlite", request)
        driver = Driver(workspace, output, json.loads((workspace / "experiment-matrix.json").read_text()), ledger)
        driver.phase("input_admission")
        driver.save("research-identity.json", {"x": 1})
        driver.save("reused-source-inventory.json", {"x": 2})
        try:
            driver.save("lean-inputs.json", {"x": 3})
        except BudgetError as exc:
            assert not (output / "lean-inputs.json").exists()
            assert "output_writes limit exceeded" in str(exc)
        else:
            raise AssertionError("INSUFFICIENT_OUTPUT_BUDGET_ACCEPTED")
        assert dict((resource, units) for stage, resource, units in reservations(output)
                    if stage == "input_admission")["output_writes"] == 2
        ledger.close()
    return {"rejected_before_write": True, "output_writes_reserved": 2}


def idempotent_case():
    request = request_for(output_writes=3)
    workspace = setup_workspace("idempotent", request)
    output = workspace / "run-once"
    output.mkdir()
    with Handshake(output, request):
        ledger = BudgetLedger(output / "ledger.sqlite", request)
        driver = Driver(workspace, output, json.loads((workspace / "experiment-matrix.json").read_text()), ledger)
        driver.phase("input_admission")
        driver.save("same.json", {"same": True})
        driver.save("same.json", {"same": True})
        assert dict((resource, units) for stage, resource, units in reservations(output)
                    if stage == "input_admission")["output_writes"] == 1
        ledger.close()
    return {"identical_rewrite_not_recharged": True, "output_writes_reserved": 1}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    report = {"pass": True, "scope": "controlled_recovery_cache_only",
              "collector_called": False, "model_forwards": 0,
              "world_model_updates": 0, "policy_updates": 0, "task_episodes": 0,
              "normal": normal_case(), "insufficient": insufficient_case(),
              "idempotent": idempotent_case()}
    durable_atomic_json(OUT / "input-admission-output-acceptance.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
