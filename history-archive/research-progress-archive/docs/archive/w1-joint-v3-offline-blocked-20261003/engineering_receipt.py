"""Fail-closed verification for an external server engineering receipt.

Hashes establish that the files agree with one another. They do not establish
who produced ordinary files. A valid receipt therefore also requires the
original controller stdout, original remote stdout, the downloaded server
evidence tree, and the transport-scope report to be retained outside the
frozen package. This development-only module is not wired into the frozen
entrypoints, and this version deliberately cannot approve a receipt: the
available transport meter reports an incomplete SSH/SFTP host scope, and no
host-side complete-scope evidence contract exists yet.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any

from manifest_contract import PackageContractError, verify_package


RECEIPT_SCHEMA = "w1-external-engineering-acceptance-receipt/1.0.0"
TRANSPORT_SCOPE_SCHEMA = "w1-remote-transport-cpu-scope/1.0.0"
EVIDENCE_SCHEMA = "w1-server-synthetic-acceptance-evidence/1.0.0"
RESULT_SCHEMA = "w1-server-synthetic-acceptance-result/1.0.0"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
RECEIPT_FIELDS = frozenset({"schema", "binding", "artifacts"})
BINDING_FIELDS = frozenset({
    "execution_manifest_sha256", "hashes_sha256",
    "server_acceptance_request_sha256", "research_attempt", "engineering_job_id",
})
ARTIFACT_FIELDS = frozenset({
    "launcher_stdout", "remote_stdout", "evidence_directory", "transport_scope_report",
})
EXPECTED_CALLS = {
    "model_initializations_or_loads": 24,
    "world_batch_forwards": 150,
    "world_backward_calls": 18,
    "world_optimizer_updates": 18,
    "checkpoint_writes": 12,
    "checkpoint_loads": 12,
    "world_sample_evaluations": 750,
    "cuda_context_initializations": 1,
}


class EngineeringReceiptError(RuntimeError):
    """The external engineering receipt is absent, inconsistent, or incomplete."""


def _fail(code: str) -> None:
    raise EngineeringReceiptError(code)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            _fail("JSON_DUPLICATE_FIELD:" + key)
        output[key] = value
    return output


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                           parse_constant=lambda item: _fail("JSON_NONFINITE_NUMBER:" + item))
    except EngineeringReceiptError:
        raise
    except FileNotFoundError as exc:
        raise EngineeringReceiptError(label + "_MISSING") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EngineeringReceiptError(label + "_INVALID_JSON") from exc
    if not isinstance(value, dict):
        _fail(label + "_OBJECT_REQUIRED")
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_fields(value: dict[str, Any], expected: frozenset[str], label: str) -> None:
    missing = sorted(expected - value.keys())
    unexpected = sorted(value.keys() - expected)
    if missing or unexpected:
        _fail(label + "_FIELDS_INVALID:missing=" + ",".join(missing)
              + ";unexpected=" + ",".join(unexpected))


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _reject_symlink_components(path: Path, stop: Path) -> None:
    current = stop
    for part in path.relative_to(stop).parts:
        current = current / part
        if current.is_symlink():
            _fail("EXTERNAL_EVIDENCE_SYMLINK_NOT_ALLOWED:" + str(current))


def _external_reference(raw: Any, *, label: str, evidence_base: Path,
                        package_root: Path, must_be_directory: bool = False) -> Path:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        _fail(label + "_PATH_INVALID")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        _fail(label + "_PATH_INVALID")
    candidate = evidence_base.joinpath(*relative.parts)
    _reject_symlink_components(candidate, evidence_base)
    resolved = candidate.resolve()
    if not _is_within(resolved, evidence_base) or _is_within(resolved, package_root):
        _fail(label + "_MUST_REMAIN_OUTSIDE_PACKAGE_AND_RECEIPT_ROOT")
    if must_be_directory:
        if not resolved.is_dir():
            _fail(label + "_DIRECTORY_REQUIRED")
    elif not resolved.is_file():
        _fail(label + "_FILE_REQUIRED")
    return resolved


def _last_json_line(path: Path, label: str) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise EngineeringReceiptError(label + "_UNREADABLE") from exc
    last = next((line for line in reversed(lines) if line.strip()), None)
    if last is None:
        _fail(label + "_EMPTY")
    try:
        value = json.loads(last, object_pairs_hook=_unique_object,
                           parse_constant=lambda item: _fail("JSON_NONFINITE_NUMBER:" + item))
    except EngineeringReceiptError:
        raise
    except json.JSONDecodeError as exc:
        raise EngineeringReceiptError(label + "_FINAL_LINE_INVALID_JSON") from exc
    if not isinstance(value, dict):
        _fail(label + "_FINAL_LINE_OBJECT_REQUIRED")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        _fail(label + "_INVALID")
    return float(value)


def _require_bool(value: Any, expected: bool, label: str) -> None:
    if type(value) is not bool or value is not expected:
        _fail(label + "_MISMATCH")


def _exact_counts(value: Any, expected: dict[str, int], label: str) -> bool:
    return (isinstance(value, dict) and set(value) == set(expected)
            and all(type(value[key]) is int and value[key] == count
                    for key, count in expected.items()))


def _verify_package_binding(package_root: Path, receipt: dict[str, Any]) -> tuple[dict, dict, str]:
    _require_fields(receipt, RECEIPT_FIELDS, "RECEIPT")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        _fail("RECEIPT_SCHEMA_INVALID")
    binding = receipt.get("binding")
    artifacts = receipt.get("artifacts")
    if not isinstance(binding, dict) or not isinstance(artifacts, dict):
        _fail("RECEIPT_BINDING_AND_ARTIFACTS_OBJECTS_REQUIRED")
    _require_fields(binding, BINDING_FIELDS, "RECEIPT_BINDING")
    _require_fields(artifacts, ARTIFACT_FIELDS, "RECEIPT_ARTIFACTS")

    try:
        identity = verify_package(package_root)
    except (PackageContractError, OSError) as exc:
        raise EngineeringReceiptError("PACKAGE_IDENTITY_INVALID:" + str(exc)) from exc
    request_path = package_root / "SERVER_ACCEPTANCE_REQUEST.json"
    request = _read_object(request_path, "SERVER_ACCEPTANCE_REQUEST")
    request_digest = _sha256_file(request_path)
    expected = {
        "execution_manifest_sha256": identity["manifest_sha256"],
        "hashes_sha256": identity["hashes_sha256"],
        "server_acceptance_request_sha256": request_digest,
        "research_attempt": identity["manifest"].get("attempt"),
        "engineering_job_id": request.get("engineering_job_id"),
    }
    for key, value in expected.items():
        if not isinstance(value, str) or not value or binding.get(key) != value:
            _fail("RECEIPT_PACKAGE_BINDING_MISMATCH:" + key)
    if (request.get("schema") != "w1-server-synthetic-acceptance-request/1.0.0"
            or request.get("status") != "NOT_APPROVED"
            or request.get("formal_research_attempt_created") is not False
            or request.get("automatic_retry") is not False):
        _fail("SERVER_ACCEPTANCE_REQUEST_NOT_FROZEN_NOT_APPROVED")
    resource_request = _read_object(package_root / "RESOURCE_REQUEST.json", "RESOURCE_REQUEST")
    if (resource_request.get("status") != "NOT_APPROVED"
            or resource_request.get("runner_ready") is not False
            or resource_request.get("attempt") != expected["research_attempt"]):
        _fail("RESEARCH_REQUEST_MUST_REMAIN_NOT_READY")
    return identity, request, request_digest


def _verify_external_evidence_tree(evidence_root: Path, manifest_digest: str,
                                   engineering_job_id: str) -> dict[str, Any]:
    manifest_path = evidence_root / "acceptance-evidence-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        _fail("SERVER_EVIDENCE_MANIFEST_MISSING_OR_INVALID")
    manifest_bytes = manifest_path.read_bytes()
    if _sha256_bytes(manifest_bytes) != manifest_digest:
        _fail("SERVER_EVIDENCE_MANIFEST_SHA256_MISMATCH")
    manifest = _read_object(manifest_path, "SERVER_EVIDENCE_MANIFEST")
    if manifest.get("schema") != EVIDENCE_SCHEMA:
        _fail("SERVER_EVIDENCE_MANIFEST_SCHEMA_INVALID")
    if (manifest.get("engineering_job_id") != engineering_job_id
            or manifest.get("run_status") != "passed"):
        _fail("SERVER_EVIDENCE_MANIFEST_BINDING_MISMATCH")
    _require_bool(manifest.get("formal_research_attempt_created"), False,
                  "SERVER_EVIDENCE_RESEARCH_ATTEMPT_CREATED")
    _require_bool(manifest.get("engineering_job_consumed"), True,
                  "SERVER_EVIDENCE_ENGINEERING_JOB_CONSUMED")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        _fail("SERVER_EVIDENCE_FILES_INVALID")

    observed: set[str] = set()
    for path in sorted(evidence_root.rglob("*")):
        relative = path.relative_to(evidence_root).as_posix()
        if path.is_symlink():
            _fail("SERVER_EVIDENCE_SYMLINK_NOT_ALLOWED:" + relative)
        if path.is_file() and relative != "acceptance-evidence-manifest.json":
            observed.add(relative)
        elif not path.is_file() and not path.is_dir():
            _fail("SERVER_EVIDENCE_NONREGULAR_PATH:" + relative)
    if observed != set(files):
        _fail("SERVER_EVIDENCE_FILE_SET_MISMATCH")

    for relative, metadata in files.items():
        rel = PurePosixPath(relative) if isinstance(relative, str) else PurePosixPath("..")
        if (not isinstance(relative, str) or not relative or "\\" in relative
                or rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts)):
            _fail("SERVER_EVIDENCE_PATH_INVALID")
        if not isinstance(metadata, dict) or set(metadata) != {"bytes", "sha256"}:
            _fail("SERVER_EVIDENCE_FILE_IDENTITY_INVALID:" + relative)
        size = metadata.get("bytes")
        digest = metadata.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            _fail("SERVER_EVIDENCE_BYTE_COUNT_INVALID:" + relative)
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            _fail("SERVER_EVIDENCE_SHA256_INVALID:" + relative)
        path = evidence_root.joinpath(*rel.parts)
        _reject_symlink_components(path, evidence_root)
        if not path.is_file() or path.stat().st_size != size or _sha256_file(path) != digest:
            _fail("SERVER_EVIDENCE_FILE_BYTES_MISMATCH:" + relative)
    return manifest


def _verify_acceptance_result(evidence_root: Path, request: dict[str, Any],
                              package_identity: dict[str, Any], request_digest: str,
                              registration_job_id: Any) -> dict[str, Any]:
    result = _read_object(evidence_root / "acceptance-result.json", "ACCEPTANCE_RESULT")
    if result.get("schema") != RESULT_SCHEMA:
        _fail("ACCEPTANCE_RESULT_SCHEMA_INVALID")
    if result.get("engineering_job_id") != request.get("engineering_job_id"):
        _fail("ACCEPTANCE_RESULT_ENGINEERING_JOB_MISMATCH")
    if result.get("registration_job_id") != registration_job_id:
        _fail("ACCEPTANCE_RESULT_REGISTRATION_JOB_MISMATCH")
    if result.get("run_status") != "passed":
        _fail("ACCEPTANCE_RESULT_NOT_PASSED")
    if result.get("resource_request_status") != "NOT_APPROVED":
        _fail("ACCEPTANCE_RESULT_RESEARCH_REQUEST_STATUS_INVALID")
    for field, expected in (("formal_research_attempt_created", False),
                            ("engineering_job_consumed", True), ("automatic_retry", False),
                            ("research_success", False), ("terminal_registration_completed", True)):
        _require_bool(result.get(field), expected, "ACCEPTANCE_RESULT_" + field.upper())
    if result.get("registration_terminal_event") != "SUCCEEDED":
        _fail("ACCEPTANCE_RESULT_TERMINAL_EVENT_INVALID")
    package_identity = result.get("package_identity")
    if not isinstance(package_identity, dict):
        _fail("ACCEPTANCE_RESULT_PACKAGE_BINDING_MISSING")
    expected_identity = {
        "execution_manifest_sha256": package_identity["manifest_sha256"],
        "hashes_sha256": package_identity["hashes_sha256"],
        "server_acceptance_request_sha256": request_digest,
    }
    if package_identity != expected_identity:
        _fail("ACCEPTANCE_RESULT_PACKAGE_BINDING_MISMATCH")
    if result.get("authorization_id") is None:
        _fail("ACCEPTANCE_RESULT_AUTHORIZATION_ID_MISSING")

    totals = result.get("model_call_totals")
    limits = request.get("limits")
    scope = request.get("scope")
    if not isinstance(limits, dict) or not isinstance(scope, dict):
        _fail("SERVER_ACCEPTANCE_REQUEST_SCOPE_INVALID")
    request_calls = limits.get("calls")
    if not _exact_counts(totals, EXPECTED_CALLS, "MODEL_CALL_TOTALS") or not _exact_counts(
            request_calls, EXPECTED_CALLS, "REQUEST_CALL_LIMITS"):
        _fail("ACCEPTANCE_RESULT_MODEL_CALL_TOTALS_MISMATCH")
    tests = result.get("test_results")
    if not isinstance(tests, list) or len(tests) != 4:
        _fail("ACCEPTANCE_RESULT_TEST_SET_INVALID")
    expected_test_ids = {scope_item.get("test_id") for scope_item in scope.values()
                         if isinstance(scope_item, dict)}
    if len(expected_test_ids) != 4 or not all(isinstance(item, str) for item in expected_test_ids):
        _fail("SERVER_ACCEPTANCE_REQUEST_TEST_SCOPE_INVALID")
    phase_by_test = {scope_item["test_id"]: phase for phase, scope_item in scope.items()}
    actual_test_ids: set[str] = set()
    test_phases: set[str] = set()
    for item in tests:
        if not isinstance(item, dict):
            _fail("ACCEPTANCE_RESULT_TEST_ROW_INVALID")
        test_id = item.get("test_id")
        if not isinstance(test_id, str) or test_id in actual_test_ids:
            _fail("ACCEPTANCE_RESULT_TEST_ID_INVALID")
        actual_test_ids.add(test_id)
        expected_phase = phase_by_test.get(test_id)
        if item.get("phase") != expected_phase or expected_phase in test_phases:
            _fail("ACCEPTANCE_RESULT_TEST_PHASE_INVALID:" + test_id)
        test_phases.add(expected_phase)
        if type(item.get("tests_run")) is not int or item["tests_run"] != 1 or item.get("successful") is not True:
            _fail("ACCEPTANCE_RESULT_TEST_NOT_SUCCESSFUL:" + test_id)
        if any(item.get(key) != 0 for key in ("failures", "errors", "skipped")):
            _fail("ACCEPTANCE_RESULT_TEST_COUNTS_INVALID:" + test_id)
        resources = item.get("resources")
        if not isinstance(resources, dict):
            _fail("ACCEPTANCE_RESULT_TEST_RESOURCES_MISSING:" + test_id)
        wall = _number(resources.get("wall_seconds"), "TEST_WALL_SECONDS")
        cpu = _number(resources.get("complete_process_cpu_seconds"), "TEST_CPU_SECONDS")
        if expected_phase in {"cpu_scope_regression", "supervised_phase_sync"}:
            wall_limit = limits[expected_phase + "_wall_seconds"]
            cpu_limit = limits[expected_phase + "_cpu_seconds"]
        else:
            wall_limit = limits["server_wall_seconds"]
            cpu_limit = limits["server_complete_process_cpu_seconds"]
        if wall > wall_limit or cpu > cpu_limit:
            _fail("ACCEPTANCE_RESULT_TEST_PHASE_BUDGET_EXCEEDED:" + test_id)
    if actual_test_ids != expected_test_ids:
        _fail("ACCEPTANCE_RESULT_TEST_IDS_MISMATCH")

    resource_phases = result.get("resource_phases")
    if not isinstance(resource_phases, list) or len(resource_phases) != 4:
        _fail("ACCEPTANCE_RESULT_RESOURCE_PHASES_INVALID")
    observed_phases: set[str] = set()
    for phase in resource_phases:
        if not isinstance(phase, dict) or phase.get("name") not in phase_by_test.values():
            _fail("ACCEPTANCE_RESULT_RESOURCE_PHASE_INVALID")
        name = phase["name"]
        if name in observed_phases:
            _fail("ACCEPTANCE_RESULT_RESOURCE_PHASE_DUPLICATE")
        observed_phases.add(name)
        wall = _number(phase.get("wall_seconds"), "RESOURCE_PHASE_WALL_SECONDS")
        cpu = _number(phase.get("complete_process_cpu_seconds"), "RESOURCE_PHASE_CPU_SECONDS")
        if name in {"cpu_scope_regression", "supervised_phase_sync"}:
            wall_limit = limits[name + "_wall_seconds"]
            cpu_limit = limits[name + "_cpu_seconds"]
        else:
            wall_limit = limits["server_wall_seconds"]
            cpu_limit = limits["server_complete_process_cpu_seconds"]
        if wall > wall_limit or cpu > cpu_limit:
            _fail("ACCEPTANCE_RESULT_RESOURCE_PHASE_BUDGET_EXCEEDED:" + name)
    if observed_phases != set(phase_by_test.values()):
        _fail("ACCEPTANCE_RESULT_RESOURCE_PHASES_MISMATCH")

    joint = _read_object(evidence_root / "joint-success.json", "JOINT_PIPELINE_EVIDENCE")
    if (joint.get("device") != "cuda"
            or joint.get("status") != "prediction_evaluation_complete"
            or joint.get("real_environment_calls") != 0
            or joint.get("synthetic_environment_constructions") != 12):
        _fail("JOINT_PIPELINE_EVIDENCE_INVALID")
    seq = _read_object(evidence_root / "sequence-world-metrics.json", "SEQUENCE_WORLD_EVIDENCE")
    if seq.get("coverage_pass") is not True or seq.get("gppo_executed") is not False:
        _fail("SEQUENCE_WORLD_EVIDENCE_INVALID")
    supervised = _read_object(evidence_root / "supervised-phase-integration.json", "SUPERVISED_PHASE_EVIDENCE")
    if (supervised.get("supervisor", {}).get("status") != "complete"
            or supervised.get("real_environment_calls") != 0
            or supervised.get("model_calls") != 0
            or supervised.get("synthetic_cpu_subprocesses") != 3
            or supervised.get("commanded_cpu_work_seconds") != 0.09):
        _fail("SUPERVISED_PHASE_EVIDENCE_INVALID")
    return result


def _verify_controlled_export(evidence_root: Path) -> None:
    export_root = evidence_root / "joint-controlled-export"
    completion = _read_object(export_root / "EXPORT_COMPLETE.json", "CONTROLLED_EXPORT_COMPLETION")
    export_manifest_path = export_root / "export-manifest.json"
    export_manifest_bytes = export_manifest_path.read_bytes()
    if completion.get("schema") != "controlled-export-completion/1.0.0":
        _fail("CONTROLLED_EXPORT_COMPLETION_SCHEMA_INVALID")
    if completion.get("export_manifest_sha256") != _sha256_bytes(export_manifest_bytes):
        _fail("CONTROLLED_EXPORT_MANIFEST_SHA256_MISMATCH")
    export_manifest = _read_object(export_manifest_path, "CONTROLLED_EXPORT_MANIFEST")
    if export_manifest.get("schema") != "verified-controlled-export/1.0.0":
        _fail("CONTROLLED_EXPORT_MANIFEST_SCHEMA_INVALID")
    files = export_manifest.get("files")
    if not isinstance(files, dict) or not files:
        _fail("CONTROLLED_EXPORT_PAYLOAD_INVALID")
    observed_payloads: set[str] = set()
    for path in export_root.rglob("*"):
        if path.is_symlink():
            _fail("CONTROLLED_EXPORT_SYMLINK_NOT_ALLOWED")
        if path.is_file():
            relative = path.relative_to(export_root).as_posix()
            if relative not in {"export-manifest.json", "EXPORT_COMPLETE.json"}:
                observed_payloads.add(relative)
    if observed_payloads != set(files):
        _fail("CONTROLLED_EXPORT_PAYLOAD_SET_MISMATCH")
    total_bytes = 0
    for relative, metadata in files.items():
        rel = PurePosixPath(relative) if isinstance(relative, str) else PurePosixPath("..")
        if (not isinstance(relative, str) or not relative or "\\" in relative
                or rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts)):
            _fail("CONTROLLED_EXPORT_PATH_INVALID")
        if not isinstance(metadata, dict) or set(metadata) != {"bytes", "sha256"}:
            _fail("CONTROLLED_EXPORT_IDENTITY_INVALID:" + relative)
        size = metadata.get("bytes")
        digest = metadata.get("sha256")
        if (isinstance(size, bool) or not isinstance(size, int) or size < 0
                or not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None):
            _fail("CONTROLLED_EXPORT_IDENTITY_INVALID:" + relative)
        path = export_root.joinpath(*rel.parts)
        _reject_symlink_components(path, evidence_root)
        if not path.is_file() or path.stat().st_size != size or _sha256_file(path) != digest:
            _fail("CONTROLLED_EXPORT_PAYLOAD_BYTES_MISMATCH:" + relative)
        total_bytes += size
    if (type(completion.get("file_count")) is not int
            or completion.get("file_count") != len(files)
            or completion.get("payload_bytes") != total_bytes):
        _fail("CONTROLLED_EXPORT_COMPLETION_TOTALS_MISMATCH")
    required = {
        "run-once/status.json", "run-once/resource-settlement.json",
        "run-once/label-coverage-summary.json",
    }
    if not required.issubset(files):
        _fail("CONTROLLED_EXPORT_REQUIRED_PAYLOAD_MISSING")
    status = _read_object(export_root / "run-once/status.json", "EXPORTED_RUN_STATUS")
    settlement = _read_object(export_root / "run-once/resource-settlement.json", "EXPORTED_SETTLEMENT")
    summary = _read_object(export_root / "run-once/label-coverage-summary.json", "EXPORTED_LABEL_SUMMARY")
    attempt = status.get("attempt")
    if (status.get("stage") != "complete"
            or status.get("status") != "prediction_evaluation_complete"
            or settlement.get("attempt") != attempt
            or settlement.get("status") != "prediction_evaluation_complete"
            or settlement.get("ledger_settlement_error") is not None
            or summary.get("status") != "prediction_evaluation_complete"):
        _fail("CONTROLLED_EXPORT_PRODUCTION_PAYLOAD_INVALID")
    ledger = settlement.get("ledger")
    if not isinstance(ledger, dict) or ledger.get("pending_calls") != 0:
        _fail("CONTROLLED_EXPORT_LEDGER_NOT_SETTLED")


def _verify_server_resources(resources: Any, request: dict[str, Any]) -> dict[str, float]:
    if not isinstance(resources, dict):
        _fail("SERVER_RESOURCE_SUMMARY_MISSING")
    limits = request["limits"]
    measured = {
        key: _number(resources.get(key), "SERVER_RESOURCE_" + key.upper())
        for key in ("wall_seconds", "complete_process_cpu_seconds",
                    "root_process_rss_high_water_bytes", "active_storage_peak_bytes",
                    "aggregate_storage_peak_bytes", "closing_reserve_wall_seconds",
                    "closing_reserve_cpu_seconds", "charged_upper_wall_seconds",
                    "charged_upper_complete_process_cpu_seconds")
    }
    if measured["wall_seconds"] > limits["server_wall_seconds"]:
        _fail("SERVER_WALL_BUDGET_EXCEEDED")
    if measured["complete_process_cpu_seconds"] > limits["server_complete_process_cpu_seconds"]:
        _fail("SERVER_CPU_BUDGET_EXCEEDED")
    if measured["root_process_rss_high_water_bytes"] > limits["rss_peak_bytes"]:
        _fail("SERVER_RSS_BUDGET_EXCEEDED")
    if measured["active_storage_peak_bytes"] > limits["active_storage_bytes"]:
        _fail("SERVER_ACTIVE_STORAGE_BUDGET_EXCEEDED")
    if measured["aggregate_storage_peak_bytes"] > limits["aggregate_storage_bytes"]:
        _fail("SERVER_AGGREGATE_STORAGE_BUDGET_EXCEEDED")
    reserve_wall = limits["terminalization_reserve_wall_seconds"]
    reserve_cpu = limits["terminalization_reserve_cpu_seconds"]
    if (measured["closing_reserve_wall_seconds"] != reserve_wall
            or measured["closing_reserve_cpu_seconds"] != reserve_cpu
            or not math.isclose(measured["charged_upper_wall_seconds"], measured["wall_seconds"] + reserve_wall,
                                rel_tol=0, abs_tol=1e-9)
            or not math.isclose(measured["charged_upper_complete_process_cpu_seconds"],
                                measured["complete_process_cpu_seconds"] + reserve_cpu,
                                rel_tol=0, abs_tol=1e-9)):
        _fail("SERVER_CLOSING_CHARGE_MISMATCH")
    if (measured["charged_upper_wall_seconds"] > limits["server_wall_seconds"]
            or measured["charged_upper_complete_process_cpu_seconds"] > limits["server_complete_process_cpu_seconds"]):
        _fail("SERVER_CHARGED_UPPER_BUDGET_EXCEEDED")
    return measured


def _verify_controller_settlement(outer: dict[str, Any], remote: dict[str, Any],
                                  server: dict[str, float], request: dict[str, Any]) -> None:
    limits = request["limits"]
    settlement = outer.get("settlement")
    if not isinstance(settlement, dict):
        _fail("CONTROLLER_SETTLEMENT_MISSING")
    controller_wall = _number(settlement.get("controller_elapsed_wall_seconds"), "CONTROLLER_WALL_SECONDS")
    controller_cpu = _number(settlement.get("controller_process_cpu_seconds"), "CONTROLLER_CPU_SECONDS")
    wall_overhead = _number(settlement.get("controller_wall_overhead_seconds"), "CONTROLLER_WALL_OVERHEAD")
    expected_overhead = controller_wall - server["wall_seconds"]
    if expected_overhead < 0 or not math.isclose(wall_overhead, expected_overhead, rel_tol=0, abs_tol=1e-9):
        _fail("CONTROLLER_WALL_OVERHEAD_MISMATCH")
    controller_budget = (
        wall_overhead + limits["controller_closing_wall_reserve_seconds"] <= limits["controller_wall_seconds"]
        and controller_cpu + limits["controller_closing_cpu_reserve_seconds"]
        <= limits["controller_complete_process_cpu_seconds"]
    )
    if not controller_budget:
        _fail("CONTROLLER_BUDGET_EXCEEDED")
    if (server["charged_upper_wall_seconds"] + limits["controller_wall_seconds"]
            > limits["wall_seconds"]):
        _fail("AGGREGATE_WALL_BUDGET_EXCEEDED")
    aggregate_cpu = (server["charged_upper_complete_process_cpu_seconds"]
                     + limits["remote_preflight_and_transport_cpu_allowance_seconds"]
                     + limits["controller_complete_process_cpu_seconds"])
    if aggregate_cpu > limits["complete_process_cpu_seconds"]:
        _fail("AGGREGATE_CPU_BUDGET_EXCEEDED")
    if (settlement.get("combined_wall_charged_upper_seconds") is not None
            and not math.isclose(_number(settlement["combined_wall_charged_upper_seconds"], "COMBINED_WALL"),
                                 server["charged_upper_wall_seconds"] + limits["controller_wall_seconds"],
                                 rel_tol=0, abs_tol=1e-9)):
        _fail("AGGREGATE_WALL_SUM_MISMATCH")
    if (settlement.get("combined_cpu_charged_upper_seconds") is not None
            and not math.isclose(_number(settlement["combined_cpu_charged_upper_seconds"], "COMBINED_CPU"),
                                 aggregate_cpu, rel_tol=0, abs_tol=1e-9)):
        _fail("AGGREGATE_CPU_SUM_MISMATCH")
    if (type(outer.get("remote_exit_code")) is not int or outer.get("remote_exit_code") != 0
            or remote.get("status") != "complete"):
        _fail("REMOTE_ACCEPTANCE_NOT_COMPLETE")


def _verify_transport_scope_report(path: Path) -> None:
    report = _read_object(path, "REMOTE_TRANSPORT_SCOPE_REPORT")
    if report.get("schema") != TRANSPORT_SCOPE_SCHEMA:
        _fail("REMOTE_TRANSPORT_SCOPE_SCHEMA_INVALID")
    scope_status = report.get("remote_transport_scope_status")
    if scope_status == "incomplete_requires_host_scope":
        measured = _number(report.get("process_tree_cpu_seconds_measured"), "PROCESS_TREE_CPU_SECONDS")
        charged = _number(report.get("process_tree_cpu_seconds_charged_once"), "PROCESS_TREE_CPU_CHARGED_ONCE")
        ticks_per_second = report.get("clock_ticks_per_second")
        root_pid = report.get("process_tree_root_pid")
        root_start = report.get("process_tree_root_start_ticks")
        if (not math.isclose(measured, charged, rel_tol=0, abs_tol=1e-9)
                or report.get("process_tree_scope_status") != "verified_complete_at_snapshot"
                or report.get("cleanup_status") != "reaped"
                or isinstance(ticks_per_second, bool) or not isinstance(ticks_per_second, int)
                or ticks_per_second <= 0
                or isinstance(root_pid, bool) or not isinstance(root_pid, int) or root_pid <= 0
                or isinstance(root_start, bool) or not isinstance(root_start, int) or root_start <= 0):
            _fail("PARTIAL_TRANSPORT_PROCESS_TREE_EVIDENCE_INVALID")
        if report.get("authoritative_remote_transport_cpu_seconds") is not None:
            _fail("INCOMPLETE_REMOTE_SCOPE_CLAIMS_AUTHORITATIVE_CPU")
        parts = report.get("unmeasured_remote_transport_components")
        if (not isinstance(parts, list) or len(parts) < 3
                or not any("sshd" in item for item in parts if isinstance(item, str))
                or not any("channel" in item.lower() for item in parts if isinstance(item, str))
                or not isinstance(report.get("required_host_admin_condition"), str)
                or not report["required_host_admin_condition"].strip()):
            _fail("INCOMPLETE_REMOTE_SCOPE_GAPS_MISSING")
        _fail("REMOTE_TRANSPORT_SCOPE_INCOMPLETE_REQUIRES_HOST_SCOPE")
    _fail("REMOTE_TRANSPORT_SCOPE_STATUS_UNSUPPORTED")


def verify_engineering_receipt(package_root: Path, external_receipt_path: Path) -> dict[str, Any]:
    """Validate package-bound remote evidence, failing closed on scope gaps.

    Receipt paths are relative to the receipt's parent directory. Every
    referenced byte remains outside `package_root`; evidence manifests and
    controlled export manifests are checked against bytes on disk. The current
    transport report format explicitly excludes host-owned sshd/channel tails,
    so this API raises `REMOTE_TRANSPORT_SCOPE_INCOMPLETE_REQUIRES_HOST_SCOPE`
    even when all other engineering evidence is structurally consistent.
    """
    root = Path(package_root).resolve()
    receipt_input = Path(external_receipt_path).absolute()
    receipt_path = receipt_input.resolve()
    if _is_within(receipt_input, root) or _is_within(receipt_path, root):
        _fail("RECEIPT_MUST_BE_OUTSIDE_PACKAGE")
    if not receipt_path.is_file():
        _fail("RECEIPT_MISSING")
    receipt = _read_object(receipt_path, "RECEIPT")
    identity, request, request_digest = _verify_package_binding(root, receipt)
    evidence_base = receipt_path.parent.resolve()
    artifacts = receipt["artifacts"]
    launcher_stdout = _external_reference(artifacts["launcher_stdout"], label="LAUNCHER_STDOUT",
                                          evidence_base=evidence_base, package_root=root)
    remote_stdout = _external_reference(artifacts["remote_stdout"], label="REMOTE_STDOUT",
                                        evidence_base=evidence_base, package_root=root)
    evidence_root = _external_reference(artifacts["evidence_directory"], label="SERVER_EVIDENCE",
                                        evidence_base=evidence_base, package_root=root,
                                        must_be_directory=True)
    transport_report = _external_reference(artifacts["transport_scope_report"], label="TRANSPORT_SCOPE_REPORT",
                                           evidence_base=evidence_base, package_root=root)

    outer = _last_json_line(launcher_stdout, "LAUNCHER_STDOUT")
    remote = _last_json_line(remote_stdout, "REMOTE_STDOUT")
    if outer.get("status") != "complete":
        _fail("LAUNCHER_STATUS_NOT_COMPLETE")
    _require_bool(outer.get("runner_ready"), False, "RUNNER_READY_MUST_REMAIN_FALSE")
    _require_bool(outer.get("cpu_scope_unverified"), False, "LAUNCHER_CPU_SCOPE_UNVERIFIED")
    _require_bool(outer.get("automatic_retry"), False, "LAUNCHER_AUTOMATIC_RETRY")
    _require_bool(outer.get("engineering_job_consumed"), True, "LAUNCHER_ENGINEERING_JOB_NOT_CONSUMED")
    if outer.get("remote_result") != remote:
        _fail("REMOTE_STDOUT_AND_LAUNCHER_RESULT_DISAGREE")
    if remote.get("status") != "complete":
        _fail("REMOTE_ACCEPTANCE_NOT_COMPLETE")
    if remote.get("engineering_job_id") != request["engineering_job_id"]:
        _fail("REMOTE_ENGINEERING_JOB_MISMATCH")
    if remote.get("formal_research_attempt_created") is not False:
        _fail("REMOTE_CREATED_FORMAL_RESEARCH_ATTEMPT")
    if remote.get("engineering_job_consumed") is not True:
        _fail("REMOTE_ENGINEERING_JOB_NOT_CONSUMED")
    registration_job_id = remote.get("registration_job_id")
    if not isinstance(registration_job_id, str) or not registration_job_id.strip():
        _fail("REMOTE_REGISTRATION_JOB_ID_MISSING")
    manifest_digest = remote.get("evidence_manifest_sha256")
    if not isinstance(manifest_digest, str) or SHA256_RE.fullmatch(manifest_digest) is None:
        _fail("REMOTE_EVIDENCE_MANIFEST_DIGEST_INVALID")
    evidence_manifest = _verify_external_evidence_tree(evidence_root, manifest_digest,
                                                        request["engineering_job_id"])
    result = _verify_acceptance_result(evidence_root, request, identity, request_digest,
                                       registration_job_id)
    if evidence_manifest.get("files", {}).get("acceptance-result.json", {}).get("sha256") != _sha256_file(
            evidence_root / "acceptance-result.json"):
        _fail("ACCEPTANCE_RESULT_NOT_BOUND_BY_EVIDENCE_MANIFEST")
    _verify_controlled_export(evidence_root)
    server_resources = _verify_server_resources(remote.get("resources"), request)
    _verify_controller_settlement(outer, remote, server_resources, request)

    # No success path is defined until host-owned session/channel accounting is
    # available. The current report is structurally explicit about that gap.
    _verify_transport_scope_report(transport_report)
    _fail("COMPLETE_TRANSPORT_SCOPE_HAS_NO_VALIDATION_CONTRACT")
