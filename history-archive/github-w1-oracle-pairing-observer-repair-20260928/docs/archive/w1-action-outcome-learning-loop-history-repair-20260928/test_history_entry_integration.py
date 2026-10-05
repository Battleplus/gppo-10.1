"""Real Windows-to-WSL integration for the repaired production collector path."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path

from learning_schema import load_records
from test_launch_repair import (
    EXPORT_TEST_PARENT,
    NATIVE_TEST_PARENT,
    exists_wsl,
    invoke,
    read_wsl_json,
    seal_clone,
    sha256,
    wsl,
)


@unittest.skipUnless(os.name == "nt", "real Windows-to-WSL integration requires Windows")
class ProductionCollectorEntryIntegrationTest(unittest.TestCase):
    def test_windows_entry_runs_fake_collector_to_verified_export(self) -> None:
        base = Path(tempfile.mkdtemp(
            prefix="w1-history-入口 中文 '单' 空格-",
            dir=r"E:\Z博士\.codex-tmp",
        ))
        EXPORT_TEST_PARENT.mkdir(parents=True, exist_ok=True)
        wsl("/usr/bin/mkdir", "-p", NATIVE_TEST_PARENT)
        attempt = f"w1-history-collector-it-{uuid.uuid4().hex[:10]}"
        root = base / "历史 修复 'quoted' package"
        token = uuid.uuid4().hex + uuid.uuid4().hex
        native, token_file = seal_clone(
            root,
            attempt=attempt,
            token=token,
            native_entry="integration_history_native_launch.py",
        )
        export = EXPORT_TEST_PARENT / attempt

        result = invoke(root, token_file)
        stderr = result.stderr.decode("utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, stderr)
        self.assertTrue(exists_wsl(native))
        self.assertFalse(Path(r"E:\home\asus\w1-launch-repair-integration").exists())

        complete = json.loads((export / "EXPORT_COMPLETE.json").read_text(encoding="utf-8"))
        self.assertTrue(complete["verified"])
        self.assertEqual(complete["final_settlement_included"], "FINAL_SETTLEMENT.json")
        manifest = json.loads((export / "export-manifest.json").read_text(encoding="utf-8"))
        for relative, identity in manifest["files"].items():
            exported = export / relative
            self.assertTrue(exported.is_file(), relative)
            self.assertEqual(exported.stat().st_size, identity["bytes"], relative)
            self.assertEqual(sha256(exported), identity["sha256"], relative)

        worker = json.loads((export / "run-once" / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(worker["status"], "complete")
        self.assertTrue(worker["fake_environment"])
        self.assertEqual(worker["real_environment_calls"], 0)
        self.assertEqual(worker["model_initializations"], 0)
        self.assertEqual(worker["model_forwards"], 0)
        self.assertEqual(worker["training_updates"], 0)
        self.assertGreater(worker["persisted_labels"], 0)
        self.assertTrue(worker["first_label_validated"])
        self.assertTrue(worker["first_public_state_shared"])
        self.assertEqual(worker["ledger"]["pending_calls"], 0)
        self.assertEqual(worker["ledger"]["failed_calls"], 0)

        records = load_records(export / "run-once" / "learning-records.jsonl")
        self.assertEqual(len(records), worker["persisted_labels"])
        records[0].validate()
        units = [
            json.loads(line)
            for line in (export / "run-once" / "data-units.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertTrue(units)
        self.assertTrue(all(row["status"] == "LABELED" for row in units))
        self.assertEqual(
            len({row["decision_input_sha256"] for row in units}),
            1,
        )

        supervisor = read_wsl_json(f"{native}/supervisor-status.json")
        self.assertEqual(supervisor["status"], "complete")
        self.assertTrue(supervisor["final_resource_pass"])
        final_native = read_wsl_json(f"{native}/FINAL_SETTLEMENT.json")
        final_export = json.loads((export / "FINAL_SETTLEMENT.json").read_text(encoding="utf-8"))
        self.assertEqual(final_native, final_export)
        self.assertEqual(final_export["overall_status"], "complete")
        self.assertEqual(final_export["worker_status"], "complete")
        self.assertEqual(final_export["supervisor_status"], "complete")
        self.assertEqual(final_export["controlled_export_status"], "verified")
        self.assertFalse(final_export["automatic_retry"])
        self.assertFalse(final_export["worker_relaunched"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
