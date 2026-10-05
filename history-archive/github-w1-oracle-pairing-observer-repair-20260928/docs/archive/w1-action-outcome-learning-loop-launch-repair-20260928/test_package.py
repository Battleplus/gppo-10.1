from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SealedPackageTests(unittest.TestCase):
    def test_manifest_and_file_hashes(self):
        hashes = json.loads((ROOT / "hashes.json").read_text(encoding="utf-8"))
        manifest = json.loads((ROOT / "execution-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(sha256(ROOT / "execution-manifest.json"), hashes["execution_manifest_sha256"])
        for relative, expected in manifest["files"].items():
            self.assertEqual(sha256(ROOT / relative), expected, relative)

    def test_launch_is_unapproved_and_token_is_not_packaged(self):
        contract = json.loads((ROOT / "launch-contract.json").read_text(encoding="utf-8"))
        request = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
        manifest = json.loads((ROOT / "execution-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(request["status"], "NOT_APPROVED")
        self.assertEqual(manifest["status"], "RUNNER_READY_NOT_APPROVED")
        self.assertFalse(contract["raw_token_stored_in_package"])
        self.assertEqual(len(contract["external_token_sha256"]), 64)
        serialized = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in ROOT.iterdir() if path.is_file())
        self.assertNotIn("gh" + "p_", serialized)

    def test_large_diagnostics_are_indexed_but_not_staged(self):
        manifest = json.loads((ROOT / "execution-manifest.json").read_text(encoding="utf-8"))
        hashes = json.loads((ROOT / "hashes.json").read_text(encoding="utf-8"))
        self.assertNotIn("diagnostic-records.jsonl", manifest["files"])
        self.assertNotIn("public-contexts.jsonl", manifest["files"])
        self.assertEqual(len(hashes["excluded_large_local_evidence"]), 2)


if __name__ == "__main__":
    unittest.main()
