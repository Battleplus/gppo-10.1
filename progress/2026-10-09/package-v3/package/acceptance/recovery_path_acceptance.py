"""Targeted recovery wiring acceptance; no world-model forward or policy update."""
import copy
import json
import os
import resource
import tempfile
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from budget_ledger import BudgetLedger, BudgetError
from production_driver import Driver
import production_driver
from phase_handshake import StageServer
from linux_process_scope import proc_tree


def main():
    root = Path(__file__).resolve().parents[1]
    matrix = json.loads((root / "experiment-matrix.json").read_text())
    matrix["controlled"] = True
    request = json.loads((root / "RESOURCE_REQUEST.json").read_text())
    result = {"formal_environment_steps": 0, "formal_updates": 0, "formal_attempt_created": False,
              "model_forwards": 0, "policy_updates": 0, "task_episodes": 0}

    with tempfile.TemporaryDirectory(prefix="w1-recovery-acceptance-") as temp:
        output = Path(temp) / "normal"
        output.mkdir(parents=True)
        origin = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        server = StageServer(output / "phase.sock", request, tree_reader=proc_tree,
            cpu_origin_seconds=origin.ru_utime + origin.ru_stime + children.ru_utime + children.ru_stime,
            wall_origin_monotonic=time.monotonic())
        os.environ["W1_PHASE_ACCOUNTING_SOCKET"] = str(server.path)
        stop = threading.Event(); errors = []
        def serve():
            while not stop.is_set():
                try: server.process_pending(os.getpid())
                except BaseException as exc: errors.append(repr(exc))
                stop.wait(.001)
        thread = threading.Thread(target=serve); thread.start()
        ledger = None
        try:
            ledger = BudgetLedger(output / "ledger.sqlite", request)
            driver = Driver(root, output, matrix, ledger)
            driver.phase("input_admission")
            rows = driver.collect()
            result["admitted_complete_windows"] = len(rows)
            result["admitted_no_opportunity_windows"] = len(json.loads((output / "lean-inputs.json").read_text())["no_opportunity"])
            result["collector_used"] = False
            if len(rows) != 174 or result["admitted_no_opportunity_windows"] != 114:
                raise AssertionError("cache admission counts")

            original_train = production_driver.train_final_world
            trained = []
            restored = []
            def scheduled_train(train_rows, seed, active_ledger, checkpoint, *, epochs):
                candidate_count = sum(len(row["labels"]) for row in train_rows)
                updates = 0
                for _epoch in range(epochs):
                    for start in range(0, len(train_rows), 8):
                        batch = train_rows[start:start + 8]
                        candidates = sum(len(row["labels"]) for row in batch)
                        active_ledger.call("acceptance_schedule_only_world_update", {
                            "world_model_updates": 1, "training_forwards": 1,
                            "training_candidate_evaluations": candidates,
                            "target_forwards": len(batch), "target_candidate_evaluations": candidates,
                        }, lambda: None)
                        updates += 1
                trained.append({"seed": seed, "epochs": epochs, "updates": updates,
                                "train_windows": len(train_rows), "candidate_rows_per_epoch": candidate_count})
                active_ledger.call("acceptance_schedule_checkpoint_write", {"checkpoint_writes": 1},
                                   checkpoint.write_bytes, ("controlled-schedule-seed-%d" % seed).encode())
                active_ledger.call("acceptance_schedule_checkpoint_admission", {"checkpoint_loads": 1}, lambda: None)
                return None, {"metadata": {"seed": seed, "epochs": epochs, "updates": updates}}
            production_driver.train_final_world = scheduled_train
            driver.restore_world = lambda bindings: restored.append(sorted(map(int, bindings)))
            try:
                driver.train_worlds(rows)
            finally:
                production_driver.train_final_world = original_train
            if [x["seed"] for x in trained] != [8202, 8203] or any(x["updates"] != 320 for x in trained):
                raise AssertionError("new-seed training schedule")
            if restored != [[8201, 8202, 8203]]:
                raise AssertionError("three-seed restore handoff")
            ledger.select("analysis_and_settlement")
            ledger.select("settlement_and_verified_export")
            settlement = ledger.assert_settled()
            ledger.close(snapshot=False); ledger = None
            result["training_schedule"] = trained
            result["world_bindings_before_restore"] = restored[0]
            result["ledger_schedule_totals"] = settlement["totals"]
            result["normal_path_pass"] = True
        finally:
            if ledger is not None:
                ledger.close(snapshot=False)
            stop.set(); thread.join(timeout=6); server.close()
            os.environ.pop("W1_PHASE_ACCOUNTING_SOCKET", None)
            if errors or thread.is_alive(): raise RuntimeError("STAGE_SERVER_ACCEPTANCE:" + repr(errors))

        # A deliberately insufficient real stage cap must reject before its callable runs.
        low = copy.deepcopy(request)
        low["stages"]["input_admission"]["reused_windows"] = 0
        low_output = Path(temp) / "under_budget"
        low_output.mkdir(parents=True)
        origin = resource.getrusage(resource.RUSAGE_SELF); children = resource.getrusage(resource.RUSAGE_CHILDREN)
        low_server = StageServer(low_output / "phase.sock", low, tree_reader=proc_tree,
            cpu_origin_seconds=origin.ru_utime + origin.ru_stime + children.ru_utime + children.ru_stime,
            wall_origin_monotonic=time.monotonic())
        os.environ["W1_PHASE_ACCOUNTING_SOCKET"] = str(low_server.path)
        low_stop = threading.Event(); low_errors = []
        def serve_low():
            while not low_stop.is_set():
                try: low_server.process_pending(os.getpid())
                except BaseException as exc: low_errors.append(repr(exc))
                low_stop.wait(.001)
        low_thread = threading.Thread(target=serve_low); low_thread.start()
        low_ledger = None
        try:
            low_ledger = BudgetLedger(low_output / "ledger.sqlite", low)
            low_driver = Driver(root, low_output, matrix, low_ledger)
            low_driver.phase("input_admission")
            try:
                low_driver.collect()
            except BudgetError as exc:
                result["under_budget_rejected"] = "limit exceeded" in str(exc)
            else:
                raise AssertionError("under-budget cache admission accepted")
            result["under_budget_ledger"] = low_ledger.snapshot()
            low_ledger.close(snapshot=False); low_ledger = None
        finally:
            if low_ledger is not None: low_ledger.close(snapshot=False)
            low_stop.set(); low_thread.join(timeout=6); low_server.close()
            os.environ.pop("W1_PHASE_ACCOUNTING_SOCKET", None)
            if low_errors or low_thread.is_alive(): raise RuntimeError("LOW_STAGE_SERVER_ACCEPTANCE:" + repr(low_errors))
        if not result["under_budget_rejected"]:
            raise AssertionError("under-budget rejection")

    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
