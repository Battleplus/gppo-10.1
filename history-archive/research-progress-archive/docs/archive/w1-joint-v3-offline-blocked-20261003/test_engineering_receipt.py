"""JSON-only rejection tests for the external engineering receipt gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import engineering_receipt as receipt
from manifest_contract import sha256_file, write_identity_files


ROOT = Path(__file__).resolve().parent


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_receipt_fixture(base: Path, *, outer_status: str, runner_ready: bool) -> Path:
    package = base / "package"
    package.mkdir()
    request = {
        "schema": "w1-server-synthetic-acceptance-request/1.0.0",
        "status": "NOT_APPROVED",
        "engineering_job_id": "engineering-fixture-once",
        "formal_research_attempt_created": False,
        "automatic_retry": False,
    }
    write_json(package / "SERVER_ACCEPTANCE_REQUEST.json", request)
    write_json(package / "RESOURCE_REQUEST.json", {
        "status": "NOT_APPROVED", "runner_ready": False, "attempt": "research-fixture-once",
    })
    identity = write_identity_files(package, attempt="research-fixture-once")
    proof = base / "proof"
    proof.mkdir()
    remote = {
        "status": "complete",
        "engineering_job_id": request["engineering_job_id"],
        "formal_research_attempt_created": False,
        "engineering_job_consumed": True,
    }
    outer = {
        "status": outer_status,
        "runner_ready": runner_ready,
        "cpu_scope_unverified": not runner_ready,
        "automatic_retry": False,
        "engineering_job_consumed": True,
        "remote_exit_code": 0,
        "remote_result": remote,
    }
    write_json(proof / "launcher.stdout", outer)
    write_json(proof / "remote.stdout", remote)
    (proof / "evidence").mkdir()
    write_json(proof / "transport-scope.json", {"schema": receipt.TRANSPORT_SCOPE_SCHEMA})
    write_json(proof / "receipt.json", {
        "schema": receipt.RECEIPT_SCHEMA,
        "binding": {
            "execution_manifest_sha256": identity["execution_manifest_sha256"],
            "hashes_sha256": identity["hashes_sha256"],
            "server_acceptance_request_sha256": sha256_file(package / "SERVER_ACCEPTANCE_REQUEST.json"),
            "research_attempt": "research-fixture-once",
            "engineering_job_id": request["engineering_job_id"],
        },
        "artifacts": {
            "launcher_stdout": "launcher.stdout",
            "remote_stdout": "remote.stdout",
            "evidence_directory": "evidence",
            "transport_scope_report": "transport-scope.json",
        },
    })
    return proof / "receipt.json"


class EngineeringReceiptTests(unittest.TestCase):
    def test_missing_external_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            with self.assertRaisesRegex(receipt.EngineeringReceiptError, "RECEIPT_MISSING"):
                receipt.verify_engineering_receipt(ROOT, path)

    def test_receipt_inside_frozen_package_is_rejected(self):
        with self.assertRaisesRegex(receipt.EngineeringReceiptError,
                                    "RECEIPT_MUST_BE_OUTSIDE_PACKAGE"):
            receipt.verify_engineering_receipt(ROOT, ROOT / "not-a-real-receipt.json")

    def test_handwritten_pass_boolean_is_not_an_allowed_receipt_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            write_json(path, {
                "schema": receipt.RECEIPT_SCHEMA,
                "engineering_acceptance_passed": True,
                "runner_ready": True,
            })
            with self.assertRaisesRegex(receipt.EngineeringReceiptError,
                                        "RECEIPT_FIELDS_INVALID"):
                receipt.verify_engineering_receipt(ROOT, path)

    def test_external_complete_claim_with_runner_ready_true_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_receipt_fixture(Path(directory), outer_status="complete", runner_ready=True)
            with self.assertRaisesRegex(receipt.EngineeringReceiptError,
                                        "RUNNER_READY_MUST_REMAIN_FALSE"):
                receipt.verify_engineering_receipt(path.parent.parent / "package", path)

    def test_current_launcher_technical_stop_is_not_an_engineering_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_receipt_fixture(Path(directory), outer_status="technical_stop", runner_ready=False)
            with self.assertRaisesRegex(receipt.EngineeringReceiptError,
                                        "LAUNCHER_STATUS_NOT_COMPLETE"):
                receipt.verify_engineering_receipt(path.parent.parent / "package", path)

    def test_current_transport_scope_report_is_a_hard_block(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scope.json"
            write_json(path, {
                "schema": receipt.TRANSPORT_SCOPE_SCHEMA,
                "status": "partial_unverified",
                "process_tree_scope_status": "verified_complete_at_snapshot",
                "process_tree_cpu_seconds_measured": 1.25,
                "process_tree_cpu_seconds_charged_once": 1.25,
                "process_tree_root_pid": 1212,
                "process_tree_root_start_ticks": 88991,
                "clock_ticks_per_second": 100,
                "remote_transport_scope_status": "incomplete_requires_host_scope",
                "authoritative_remote_transport_cpu_seconds": None,
                "unmeasured_remote_transport_components": [
                    "sshd session/channel process through scope removal",
                    "independent exec and SFTP channels",
                    "meter serialization and process-exit tail",
                ],
                "required_host_admin_condition": "host scope required",
                "cleanup_status": "reaped",
                "remote_transport_cpu_measured": True,
                "pass": True,
            })
            with self.assertRaisesRegex(
                    receipt.EngineeringReceiptError,
                    "REMOTE_TRANSPORT_SCOPE_INCOMPLETE_REQUIRES_HOST_SCOPE"):
                receipt._verify_transport_scope_report(path)

    def test_unrecognized_claimed_complete_scope_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scope.json"
            write_json(path, {
                "schema": receipt.TRANSPORT_SCOPE_SCHEMA,
                "remote_transport_scope_status": "complete",
                "authoritative_remote_transport_cpu_seconds": 0.01,
                "unmeasured_remote_transport_components": [],
            })
            with self.assertRaisesRegex(receipt.EngineeringReceiptError,
                                        "REMOTE_TRANSPORT_SCOPE_STATUS_UNSUPPORTED"):
                receipt._verify_transport_scope_report(path)

    def test_evidence_manifest_is_checked_against_actual_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "evidence"
            evidence.mkdir()
            content = b'{"engineering_job_id":"job-fixture"}\n'
            result = evidence / "acceptance-result.json"
            result.write_bytes(content)
            manifest = {
                "schema": receipt.EVIDENCE_SCHEMA,
                "verified": True,
                "engineering_job_id": "job-fixture",
                "formal_research_attempt_created": False,
                "engineering_job_consumed": True,
                "run_status": "passed",
                "files": {"acceptance-result.json": {
                    "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                }},
            }
            manifest_path = evidence / "acceptance-evidence-manifest.json"
            write_json(manifest_path, manifest)
            result.write_bytes(content + b" ")
            with self.assertRaisesRegex(receipt.EngineeringReceiptError,
                                        "SERVER_EVIDENCE_FILE_BYTES_MISMATCH"):
                receipt._verify_external_evidence_tree(
                    evidence, hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "job-fixture",
                )

    def test_remote_stdout_must_match_manifest_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "evidence"
            evidence.mkdir()
            content = b"acceptance result fixture\n"
            (evidence / "acceptance-result.json").write_bytes(content)
            manifest_path = evidence / "acceptance-evidence-manifest.json"
            write_json(manifest_path, {
                "schema": receipt.EVIDENCE_SCHEMA,
                "verified": True,
                "engineering_job_id": "job-fixture",
                "formal_research_attempt_created": False,
                "engineering_job_consumed": True,
                "run_status": "passed",
                "files": {"acceptance-result.json": {
                    "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                }},
            })
            with self.assertRaisesRegex(receipt.EngineeringReceiptError,
                                        "SERVER_EVIDENCE_MANIFEST_SHA256_MISMATCH"):
                receipt._verify_external_evidence_tree(evidence, "0" * 64, "job-fixture")

    def test_controlled_export_payload_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "evidence"
            export_root = evidence / "joint-controlled-export"
            run = export_root / "run-once"
            run.mkdir(parents=True)
            rows = {
                "run-once/status.json": {
                    "attempt": "synthetic-attempt",
                    "stage": "complete",
                    "status": "prediction_evaluation_complete",
                },
                "run-once/resource-settlement.json": {
                    "attempt": "synthetic-attempt",
                    "status": "prediction_evaluation_complete",
                    "ledger_settlement_error": None,
                    "ledger": {"pending_calls": 0},
                },
                "run-once/label-coverage-summary.json": {
                    "status": "prediction_evaluation_complete",
                },
            }
            identities = {}
            payload_bytes = 0
            for relative, value in rows.items():
                path = export_root / relative
                write_json(path, value)
                content = path.read_bytes()
                identities[relative] = {
                    "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                }
                payload_bytes += len(content)
            export_manifest_path = export_root / "export-manifest.json"
            write_json(export_manifest_path, {
                "schema": "verified-controlled-export/1.0.0",
                "verified": False,
                "files": identities,
            })
            complete_path = export_root / "EXPORT_COMPLETE.json"
            write_json(complete_path, {
                "schema": "controlled-export-completion/1.0.0",
                "verified": False,
                "file_count": len(identities),
                "payload_bytes": payload_bytes,
                "export_manifest_sha256": hashlib.sha256(export_manifest_path.read_bytes()).hexdigest(),
            })
            (run / "status.json").write_bytes(b'{"status":"technical_stop"}\n')
            with self.assertRaisesRegex(receipt.EngineeringReceiptError,
                                        "CONTROLLED_EXPORT_PAYLOAD_BYTES_MISMATCH"):
                receipt._verify_controlled_export(evidence)


if __name__ == "__main__":
    unittest.main()
