"""Real Linux processes through production supervisor/ledger, without research work."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import supervise

PACKAGE = Path(__file__).resolve().parent
WORKER = '''
from pathlib import Path
import json, subprocess, sys, time
from budget_ledger import BudgetLedger
root = Path.cwd()
output = root / "run-once"
output.mkdir()
request = json.loads((root / "RESOURCE_REQUEST.json").read_text())
ledger = BudgetLedger(output / "budget.sqlite3", request)
def burn(seconds):
    code = "import time; end=time.process_time()+"+str(seconds)+"\\nwhile time.process_time()<end: pass"
    subprocess.run([sys.executable, "-B", "-c", code], check=True)
ledger.select("label_qualification")
burn(0.03)
ledger.select("world_model_training_and_selection")
burn(0.03)
ledger.select("prediction_confirmation")
burn(0.03)
ledger.select("settlement_and_verified_export")
ledger.close()
'''


@unittest.skipUnless(os.name == "posix", "native Linux production process test")
class SupervisedPhaseIntegrationTests(unittest.TestCase):
    def test_short_subprocesses_and_stage_boundaries_through_real_supervisor(self):
        with tempfile.TemporaryDirectory(prefix="w1-supervised-cpu-") as directory:
            root = Path(directory)
            for name in ("budget_ledger.py", "phase_handshake.py", "infra_io.py"):
                shutil.copyfile(PACKAGE / name, root / name)
            (root / "cpu-fixture.py").write_text(WORKER, encoding="utf-8")
            request = json.loads((PACKAGE / "RESOURCE_REQUEST.json").read_text())
            request.pop("gpu")
            (root / "RESOURCE_REQUEST.json").write_text(json.dumps(request), encoding="utf-8")
            (root / "experiment-matrix.json").write_text('{"world_model_device":"cpu"}', encoding="utf-8")
            allowed = sorted(os.sched_getaffinity(0))
            counter = iter((0, 100))
            def idle():
                tick = next(counter)
                return {cpu: (tick, tick) for cpu in allowed}
            with patch.object(supervise, "machine_guard", return_value={"available_ram_bytes": 8*1024**3, "disk_free_bytes": 16*1024**3}), \
                    patch.object(supervise, "cpu_idle", side_effect=idle), \
                    patch.object(supervise.os, "sched_setaffinity", return_value=None), \
                    patch.object(supervise.os, "nice", return_value=0):
                code = supervise.main(root, "cpu-fixture.py", sample_interval=0.01)
            state = json.loads((root / "supervisor-status.json").read_text())
            if code:
                self.fail(str(state) + "\n" + (root / "console.log").read_text())
            self.assertEqual(state["status"], "complete")
            rows = [json.loads(line) for line in (root / "run-once" / "worker-phase-accounting.jsonl").read_text().splitlines()]
            self.assertEqual([row["stage"] for row in rows], ["staging_and_zero_step_gate", "label_qualification",
                "world_model_training_and_selection", "prediction_confirmation", "settlement_and_verified_export"])
            for row in rows[:-1]:
                self.assertEqual(row["supervisor_handshake"]["status"], "acknowledged")
            for row in rows[1:4]:
                self.assertGreater(row["cpu_seconds"], 0.025)
            ledger = json.loads((root / "supervisor-status.json").read_text())
            self.assertGreater(ledger["cpu_seconds"], sum(row["cpu_seconds"] for row in rows))
            evidence = os.environ.get("W1_TEST_EVIDENCE_DIR")
            if evidence:
                (Path(evidence) / "supervised-phase-integration.json").write_text(json.dumps({
                    "supervisor": state, "worker_phases": rows, "real_environment_calls": 0,
                    "model_calls": 0, "synthetic_cpu_subprocesses": 3,
                    "commanded_cpu_work_seconds": 0.09,
                    "hardware_guard_only_injected": True}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
