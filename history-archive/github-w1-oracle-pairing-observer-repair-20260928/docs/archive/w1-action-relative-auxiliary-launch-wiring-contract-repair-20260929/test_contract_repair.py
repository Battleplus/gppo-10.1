from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from manifest_contract import PackageContractError, verify_package


ROOT = Path(__file__).resolve().parent
OLD = Path(r"E:\Z博士\research-plans\w1-action-relative-auxiliary-launch-wiring-v1")


class ContractRepairTests(unittest.TestCase):
    def test_old_package_missing_field_is_reproduced(self):
        old_hashes = json.loads((OLD / "hashes.json").read_text(encoding="utf-8"))
        self.assertNotIn("execution_manifest_sha256", old_hashes)

    def test_final_package_uses_shared_contract(self):
        identity = verify_package(ROOT)
        self.assertEqual(identity["manifest"]["schema"], "w1-action-relative-auxiliary-launch-wiring-manifest/2.0.0")
        self.assertEqual(identity["hashes"]["schema"], "w1-action-relative-auxiliary-launch-wiring-hashes/2.0.0")

    def test_schema_and_tamper_errors_are_named(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("hashes.json", "execution-manifest.json"):
                (root / name).write_text((ROOT / name).read_text(encoding="utf-8"), encoding="utf-8")
            (root / "hashes.json").write_text(json.dumps({"files": {}}), encoding="utf-8")
            with self.assertRaises(PackageContractError) as ctx:
                verify_package(root)
            self.assertIn("HASHES_SCHEMA_INVALID", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
