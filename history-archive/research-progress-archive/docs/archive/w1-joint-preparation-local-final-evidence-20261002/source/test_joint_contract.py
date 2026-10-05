import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from manifest_contract import (AUTHORIZATION_SCHEMA, PackageContractError,
    verify_external_authorization, write_identity_files, sha256_file)
from worker_contract import verify_worker_contract


class JointContractTests(unittest.TestCase):
    def fixture(self, root, *, integration=False):
        request = {"status": "NOT_APPROVED", "attempt": "synthetic-contract-only"}
        contract = {"schema": "w1-eawm-jepa-launch-contract/2.1.0", "attempt": request["attempt"],
                    "integration_test": integration,
                    "resource_request_sha256": "unused"}
        (root / "RESOURCE_REQUEST.json").write_text(json.dumps(request), encoding="utf-8")
        contract["resource_request_sha256"] = sha256_file(root / "RESOURCE_REQUEST.json")
        (root / "launch-contract.json").write_text(json.dumps(contract), encoding="utf-8")
        (root / "experiment-matrix.json").write_text(json.dumps({"attempt": request["attempt"]}), encoding="utf-8")
        identity = write_identity_files(root, attempt=request["attempt"])
        authorization = {"schema": AUTHORIZATION_SCHEMA, "attempt": request["attempt"],
            **identity, "resource_request_sha256": contract["resource_request_sha256"],
            "status": "APPROVED", "token_sha256": hashlib.sha256(b"synthetic-contract-token").hexdigest()}
        return identity, authorization

    def test_authorization_rejections_precede_outputs(self):
        for fault in ("trailing", "missing", "attempt", "budget", "token"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                root = base / "package"
                root.mkdir()
                _, auth = self.fixture(root)
                if fault == "missing":
                    del auth["attempt"]
                if fault == "attempt":
                    auth["attempt"] = "incorrect"
                if fault == "budget":
                    auth["resource_request_sha256"] = "0" * 64
                path = base / "authorization.json"
                path.write_text(json.dumps(auth) + (" {}" if fault == "trailing" else ""), encoding="utf-8")
                with self.assertRaises(PackageContractError):
                    verify_external_authorization(root, path,
                        token="incorrect" if fault == "token" else "synthetic-contract-token", preflight_only=False)
                for name in ("execution.lock", "run-once", "budget.sqlite3"):
                    self.assertFalse((root / name).exists())

    def test_formal_worker_actually_rejects_integration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity, _ = self.fixture(root, integration=True)
            with self.assertRaisesRegex(RuntimeError, "FORMAL_WORKER_REJECTS_TEST_SUBSTITUTES"):
                verify_worker_contract(root, "synthetic-contract-only", identity["execution_manifest_sha256"],
                                       identity["hashes_sha256"])
            self.assertFalse((root / "run-once").exists())
            checked = verify_worker_contract(root, "synthetic-contract-only", identity["execution_manifest_sha256"],
                                            identity["hashes_sha256"], allow_integration=True)
            self.assertTrue(checked["contract"]["integration_test"])

    def test_structural_preflight_preserves_all_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "package"
            root.mkdir()
            _, auth = self.fixture(root)
            auth.update(status="NOT_APPROVED", token_sha256=None)
            path = base / "authorization.json"
            path.write_text(json.dumps(auth), encoding="utf-8")
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            verify_external_authorization(root, path, token=None, preflight_only=True)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
            with self.assertRaises(PackageContractError):
                verify_external_authorization(root, path, token=None, preflight_only=False)


if __name__ == "__main__":
    unittest.main()
