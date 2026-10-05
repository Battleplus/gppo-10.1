"""Approval-gated Windows transport for the separate server acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time
import traceback

import acceptance_entry
import launch_joint_once as controller
import controller_runtime
from transport_wall_budget import WallBudget


ROOT = Path(__file__).resolve().parent
REQUEST_PATH = ROOT / "SERVER_ACCEPTANCE_REQUEST.json"
IMPLEMENTATION_FILES = {
    "acceptance_entry.py",
    "launch_server_acceptance.py",
    "test_server_acceptance_entry.py",
    "SERVER_ACCEPTANCE_PROTOCOL.md",
    "bulk_payload_transport.py",
    "isolated_fixture_preflight.py",
    "transport_wall_budget.py",
    "launch_joint_once.py",
}


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("JSON_OBJECT_REQUIRED:" + path.name)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_implementation_identity(request: dict) -> None:
    identities = request.get("implementation_sha256")
    if not isinstance(identities, dict) or set(identities) != IMPLEMENTATION_FILES:
        raise RuntimeError("SERVER_ACCEPTANCE_IMPLEMENTATION_SET_MISMATCH")
    for relative, expected in identities.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise RuntimeError("SERVER_ACCEPTANCE_IMPLEMENTATION_PATH_INVALID:" + relative)
        if (not isinstance(expected, str) or len(expected) != 64
                or any(character not in "0123456789abcdef" for character in expected)
                or _sha256(path) != expected):
            raise RuntimeError("SERVER_ACCEPTANCE_IMPLEMENTATION_DIGEST_MISMATCH:" + relative)


def verify_structure() -> dict:
    request = _read_json(REQUEST_PATH)
    acceptance_entry.validate_request(request)
    identity = controller.verify_package(ROOT)
    verify_implementation_identity(request)
    contract = _read_json(ROOT / "launch-contract.json")
    if (contract.get("runtime_identity_sha256") != request["runtime"]["runtime_identity_sha256"]
            or contract.get("native_python") != request["runtime"]["python_executable"]):
        raise RuntimeError("SERVER_ACCEPTANCE_LAUNCH_RUNTIME_MISMATCH")
    controller.verify_joint_inputs(ROOT)
    return {"request": request, "identity": identity, "contract": contract,
            "request_sha256": _sha256(REQUEST_PATH)}


def validate_approval(path: Path, checked: dict) -> tuple[dict, str]:
    auth_path = path.resolve()
    if auth_path.is_relative_to(ROOT.resolve()) or not auth_path.is_file():
        raise RuntimeError("SERVER_ACCEPTANCE_AUTHORIZATION_MUST_BE_EXTERNAL")
    authorization = _read_json(auth_path)
    authorization_id = acceptance_entry.validate_authorization(
        authorization,
        request=checked["request"],
        request_sha256=checked["request_sha256"],
        manifest_sha256=checked["identity"]["manifest_sha256"],
        hashes_sha256=checked["identity"]["hashes_sha256"],
        token=None,
        require_token=False,
    )
    return authorization, authorization_id


def _remote_contract(checked: dict) -> dict:
    contract = dict(checked["contract"])
    request = checked["request"]
    contract["attempt"] = request["engineering_job_id"]
    contract["native_execution_root"] = request["remote_execution_root"]
    return contract


def remote_target_preflight(client, checked: dict) -> dict:
    request = checked["request"]
    python = checked["contract"]["native_python"]
    paths = [request["remote_execution_root"], request["evidence_directory"]]
    script = (
        "import json,pathlib; paths=" + repr(paths) + "; "
        "state={p:{'exists':pathlib.Path(p).exists(),'symlink':pathlib.Path(p).is_symlink()} for p in paths}; "
        "print(json.dumps(state,sort_keys=True))"
    )
    result = controller.execute(client, [python, "-I", "-B", "-c", script], timeout=30)
    if result["exit_code"]:
        raise RuntimeError("SERVER_ACCEPTANCE_TARGET_PREFLIGHT_FAILED:" + result["stderr"])
    state = json.loads(result["stdout"])
    if any(value["exists"] or value["symlink"] for value in state.values()):
        raise RuntimeError("SERVER_ACCEPTANCE_ENGINEERING_PATH_ALREADY_EXISTS")
    return state


def _stage_acceptance_sources(client, remote_root: str) -> None:
    payloads = ["SERVER_ACCEPTANCE_REQUEST.json", *sorted(IMPLEMENTATION_FILES)]
    with client.open_sftp() as sftp:
        for relative in payloads:
            target = remote_root + "/" + relative
            controller._mkdir_parents(sftp, str(PurePosixPath(target).parent))
            sftp.put(str(ROOT / relative), target)


def _read_one_time_token() -> str:
    token = sys.stdin.readline().rstrip("\r\n")
    if not token:
        raise RuntimeError("SERVER_ACCEPTANCE_ONE_TIME_TOKEN_REQUIRED")
    return token


def verify_uploaded_authorization(client, remote_path: str, local_path: Path,
                                  checked: dict) -> dict:
    local_bytes = local_path.read_bytes()
    with client.open_sftp() as sftp:
        with sftp.open(remote_path, "rb") as stream:
            remote_bytes = stream.read()
    if not remote_bytes or len(remote_bytes) != len(local_bytes):
        raise RuntimeError("UPLOADED_AUTHORIZATION_LENGTH_MISMATCH")
    digest = hashlib.sha256(remote_bytes).hexdigest()
    if digest != hashlib.sha256(local_bytes).hexdigest():
        raise RuntimeError("UPLOADED_AUTHORIZATION_SHA256_MISMATCH")
    # The strict reader rejects trailing JSON, duplicate fields and non-objects.
    authorization = acceptance_entry._read_json(local_path)
    acceptance_entry.validate_authorization(
        authorization, request=checked["request"], request_sha256=checked["request_sha256"],
        manifest_sha256=checked["identity"]["manifest_sha256"],
        hashes_sha256=checked["identity"]["hashes_sha256"], token=None, require_token=False,
    )
    if json.loads(remote_bytes) != authorization:
        raise RuntimeError("UPLOADED_AUTHORIZATION_JSON_MISMATCH")
    return {"bytes": len(remote_bytes), "sha256": digest, "length_verified": True,
            "json_verified": True, "identity_and_budget_binding_verified": True,
            "worker_started": False}


def _local_evidence_path(request: dict) -> Path:
    return ROOT.parents[1] / "runs" / request["engineering_job_id"]


def download_verified_evidence(client, request: dict, expected_manifest_sha256: str) -> dict:
    from bulk_payload_transport import download_evidence
    return download_evidence(client, request, expected_manifest_sha256,
                             destination=_local_evidence_path(request), execute=controller.execute)


def _completion_gate(*, remote_exit_code: int, remote_status: str, evidence_received: bool,
                     controller_budget_pass: bool, server_budget_pass: bool,
                     aggregate_budget_arithmetic_pass: bool,
                     remote_transport_cpu_measured: bool) -> tuple[str, bool]:
    aggregate_budget_pass = (
        aggregate_budget_arithmetic_pass and remote_transport_cpu_measured
    )
    complete = (
        remote_exit_code == 0 and remote_status == "complete" and evidence_received
        and controller_budget_pass and server_budget_pass and aggregate_budget_pass
    )
    return ("complete" if complete else "technical_stop"), aggregate_budget_pass


def run_formal(checked: dict, authorization_path: Path, real_name: str,
               *, password_stdin: bool = False) -> dict:
    controller_started_wall = time.monotonic()
    controller_started_cpu = time.process_time()
    wall_budget = WallBudget(checked["request"]["limits"]["wall_seconds"],
        closing_reserve=checked["request"]["limits"]["controller_closing_wall_reserve_seconds"],
        started=controller_started_wall)
    if not real_name or real_name != real_name.strip():
        raise RuntimeError("REAL_NAME_REQUIRED_BEFORE_SERVER_WORKLOAD")
    name_id = controller.derive_name_id(real_name)
    authorization, authorization_id = validate_approval(authorization_path, checked)
    del authorization
    controller_runtime.verify_controller_runtime()

    password = sys.stdin.readline().rstrip("\r\n") if password_stdin else __import__("getpass").getpass(
        "Remote SSH password: ")
    if not password:
        raise RuntimeError("SSH_PASSWORD_REQUIRED")
    state = {"engineering_job_consumed": False, "token_consumed": False}
    client = None
    token = None
    try:
        with wall_budget.phase("connect"):
            client = controller.connect(checked["contract"], password, wall_budget=wall_budget)
        password = None
        contract = _remote_contract(checked)
        with wall_budget.phase("remote_preflight"):
            controller.remote_readonly_preflight(client, contract)
            remote_state = remote_target_preflight(client, checked)
            if checked['request'].get('smoke_adapter') is True:
                from bulk_payload_transport import gpu_preflight
                remote_state['gpu1_prelaunch'] = gpu_preflight(client, controller.execute)
        token = _read_one_time_token()
        state["token_consumed"] = True
        acceptance_entry.validate_authorization(
            _read_json(authorization_path), request=checked["request"],
            request_sha256=checked["request_sha256"],
            manifest_sha256=checked["identity"]["manifest_sha256"],
            hashes_sha256=checked["identity"]["hashes_sha256"], token=token,
        )

        upload_checked = {**checked, "contract": contract}
        with wall_budget.phase("upload"):
            try:
                from bulk_payload_transport import upload_once
                remote_root, external_authorization = upload_once(
                    client, upload_checked, authorization_path, source_root=ROOT,
                    execute=controller.execute, mkdir_parents=controller._mkdir_parents,
                )
                state["engineering_job_consumed"] = True
            except BaseException:
                # An interrupted mkdir response may hide successful creation.
                # Unknown is terminal and never grants permission to retry.
                state["engineering_job_consumed"] = None
                try:
                    with client.open_sftp() as sftp:
                        sftp.stat(contract["native_execution_root"])
                        state["engineering_job_consumed"] = True
                except FileNotFoundError:
                    state["engineering_job_consumed"] = False
                except BaseException as inspection_error:
                    state["consumption_inspection_error_type"] = type(inspection_error).__name__
                raise
            _stage_acceptance_sources(client, remote_root)
        remote_auth = external_authorization
        if remote_auth != ("/home/user1/.w1-external-authorizations/"
                           + checked["request"]["engineering_job_id"] + ".json"):
            raise RuntimeError("SERVER_ACCEPTANCE_AUTHORIZATION_STAGE_PATH_MISMATCH")
        upload_verification = verify_uploaded_authorization(
            client, remote_auth, authorization_path, checked)
        upload_evidence_path = authorization_path.parent / "upload-verification.json"
        upload_evidence_path.write_text(json.dumps(upload_verification, indent=2) + "\n", encoding="utf-8")
        command = [
            "env", "-u", "LD_LIBRARY_PATH", "-u", "LD_PRELOAD", "-u", "PYTHONPATH", "-u", "PYTHONHOME",
            "CUDA_VISIBLE_DEVICES=1", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4", "OPENBLAS_NUM_THREADS=4",
            "PYTHONNOUSERSITE=1", checked["contract"]["native_python"], "-I", "-B", "-X",
            "pycache_prefix=/home/user1/.venvs/w1-runtime-v1/.w1-no-bytecode-cache",
            remote_root + "/acceptance_entry.py",
            "--authorization-file", remote_auth,
            "--real-name", real_name,
            "--name-id", name_id,
            "--evidence-dir", checked["request"]["evidence_directory"],
        ]
        bootstrap_command = command[:command.index("--authorization-file")] + ["--bootstrap-only"]
        with wall_budget.phase("bootstrap"):
            bootstrap = controller.execute(client, bootstrap_command, timeout=30)
        (authorization_path.parent / "remote-bootstrap.json").write_text(
            json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8")
        if bootstrap["exit_code"] != 0:
            raise RuntimeError("REMOTE_BOOTSTRAP_FAILED:" + bootstrap["stdout"] + bootstrap["stderr"])
        if json.loads(bootstrap["stdout"].splitlines()[-1]).get("status") != "bootstrap_pass":
            raise RuntimeError("REMOTE_BOOTSTRAP_RESULT_INVALID")
        remote_allowance = wall_budget.remote_allowance(
            checked["request"]["limits"]["server_wall_seconds"],
            remote_closing_reserve=checked["request"]["limits"]["terminalization_reserve_wall_seconds"])
        command += ["--remaining-wall-seconds", repr(remote_allowance)]
        with wall_budget.phase("remote_acceptance"):
            result = controller.execute(client, command, timeout=remote_allowance,
                                       input_text=token + "\n")
        token = None
        lines = [line for line in result["stdout"].splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("SERVER_ACCEPTANCE_RESULT_MISSING")
        remote_result = json.loads(lines[-1])
        remote_output = {"exit_code": result["exit_code"], "stdout": result["stdout"],
                         "stderr": result["stderr"]}
        _local_evidence_path(checked["request"]).parent.mkdir(parents=True, exist_ok=True)
        remote_output_path = _local_evidence_path(checked["request"]).with_suffix(".remote-output.json")
        remote_output_path.write_text(json.dumps(remote_output, indent=2) + "\n", encoding="utf-8")
        manifest_digest = remote_result.get("evidence_manifest_sha256")
        evidence = None
        if isinstance(manifest_digest, str) and len(manifest_digest) == 64:
            with wall_budget.phase("download"):
                evidence = download_verified_evidence(client, checked["request"], manifest_digest)
        controller_wall = time.monotonic() - controller_started_wall
        controller_cpu = time.process_time() - controller_started_cpu
        server_resources = remote_result.get("resources") or {}
        server_wall = server_resources.get("wall_seconds")
        server_cpu = server_resources.get("complete_process_cpu_seconds")
        server_wall_upper = server_resources.get("charged_upper_wall_seconds")
        server_cpu_upper = server_resources.get("charged_upper_complete_process_cpu_seconds")
        controller_wall_overhead = (controller_wall - server_wall
                                    if isinstance(server_wall, (int, float)) else None)
        limits = checked["request"]["limits"]
        accounting_values_valid = all(
            isinstance(value, (int, float)) and value >= 0
            for value in (server_wall, server_cpu, server_wall_upper, server_cpu_upper,
                          controller_wall_overhead, controller_cpu)
        )
        remote_transport_cpu_measured = False
        if accounting_values_valid:
            combined_wall_upper = server_wall_upper + limits["controller_wall_seconds"]
            combined_cpu_upper = (server_cpu_upper
                                  + limits["remote_preflight_and_transport_cpu_allowance_seconds"]
                                  + limits["controller_complete_process_cpu_seconds"])
            controller_budget_pass = (
                controller_wall_overhead + limits["controller_closing_wall_reserve_seconds"]
                <= limits["controller_wall_seconds"]
                and controller_cpu + limits["controller_closing_cpu_reserve_seconds"]
                <= limits["controller_complete_process_cpu_seconds"]
            )
            server_budget_pass = (
                server_wall_upper <= limits["server_wall_seconds"]
                and server_cpu_upper <= limits["server_complete_process_cpu_seconds"]
            )
            aggregate_budget_arithmetic_pass = (
                combined_wall_upper <= limits["wall_seconds"]
                and combined_cpu_upper <= limits["complete_process_cpu_seconds"]
            )
        else:
            combined_wall_upper = combined_cpu_upper = None
            controller_budget_pass = server_budget_pass = aggregate_budget_arithmetic_pass = False
        overall_status, aggregate_budget_pass = _completion_gate(
            remote_exit_code=result["exit_code"],
            remote_status=remote_result.get("status"),
            evidence_received=evidence is not None,
            controller_budget_pass=controller_budget_pass,
            server_budget_pass=server_budget_pass,
            aggregate_budget_arithmetic_pass=aggregate_budget_arithmetic_pass,
            remote_transport_cpu_measured=remote_transport_cpu_measured,
        )
        # Smoke authorization waives exact transport CPU acceptance, not measured limits.
        smoke_test_passed = (
            checked["request"].get("smoke_adapter") is True
            and result["exit_code"] == 0 and remote_result.get("status") == "complete"
            and evidence is not None and controller_budget_pass and server_budget_pass
            and aggregate_budget_arithmetic_pass
        )
        if smoke_test_passed:
            overall_status = "gpu_smoke_test_pass"
        settlement = {
            "status": overall_status,
            "gpu_smoke_test_passed": smoke_test_passed,
            "runner_ready": False,
            "cpu_scope_unverified": not remote_transport_cpu_measured,
            "server_wall_seconds": server_wall,
            "server_complete_process_cpu_seconds": server_cpu,
            "server_charged_upper_wall_seconds": server_wall_upper,
            "server_charged_upper_complete_process_cpu_seconds": server_cpu_upper,
            "controller_elapsed_wall_seconds": controller_wall,
            "controller_process_cpu_seconds": controller_cpu,
            "controller_wall_overhead_seconds": controller_wall_overhead,
            "controller_wall_charged_allowance_seconds": limits["controller_wall_seconds"],
            "remote_preflight_and_transport_cpu_charged_allowance_seconds": (
                limits["remote_preflight_and_transport_cpu_allowance_seconds"]
            ),
            "controller_cpu_charged_allowance_seconds": limits["controller_complete_process_cpu_seconds"],
            "combined_wall_charged_upper_seconds": combined_wall_upper,
            "combined_cpu_charged_upper_seconds": combined_cpu_upper,
            "controller_budget_pass": controller_budget_pass,
            "server_budget_pass": server_budget_pass,
            "aggregate_budget_pass": aggregate_budget_pass,
            "aggregate_budget_arithmetic_pass": aggregate_budget_arithmetic_pass,
            "remote_preflight_transport_cpu_individually_measured": remote_transport_cpu_measured,
            "automatic_retry": False,
        }
        return {"status": settlement["status"], "runner_ready": False,
                "cpu_scope_unverified": settlement["cpu_scope_unverified"],
                "remote_exit_code": result["exit_code"], "remote_result": remote_result,
                "remote_target_preflight": remote_state, "local_evidence": evidence,
                "uploaded_authorization_verification": upload_verification,
                "settlement": settlement,
                "engineering_job_consumed": state["engineering_job_consumed"],
                "token_consumed": state["token_consumed"], "automatic_retry": False,
                "authorization_id": authorization_id}
    except BaseException as exc:
        chain = traceback.format_exc()
        if token:
            chain = chain.replace(token, "<redacted>")
        return {"status": "technical_stop", "error_type": type(exc).__name__,
                "exception_chain": chain,
                "engineering_job_id": checked["request"]["engineering_job_id"],
                "runner_ready": False, "cpu_scope_unverified": True,
                "engineering_job_consumed": state["engineering_job_consumed"],
                "consumption_inspection_error_type": state.get("consumption_inspection_error_type"),
                "token_consumed": state["token_consumed"], "automatic_retry": False}
    finally:
        password = None
        token = None
        wall_budget.close()
        if client is not None:
            client.close()
        (authorization_path.parent / "controller-wall-phases.json").write_text(json.dumps({
            "global_wall_limit_seconds": wall_budget.limit,
            "closing_reserve_seconds": wall_budget.closing_reserve,
            "deadline_reached": wall_budget.expired.is_set(),
            "elapsed_wall_seconds": time.monotonic()-controller_started_wall,
            "phases": wall_budget.phases,
            "enforcement_scope": "own SSH connection timer plus remote alarm; local hashing/OS shutdown are cooperative",
        }, indent=2)+"\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure-only", "--preflight-only", dest="structure_only", action="store_true")
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--real-name")
    parser.add_argument("--password-stdin", action="store_true")
    args = parser.parse_args(argv)
    try:
        checked = verify_structure()
        if args.structure_only:
            print(json.dumps({"status": "structure_preflight_pass",
                              "formal_authorization_validated": False,
                              "staging_started": False, "worker_started": False,
                              "authorization_consumed": False,
                              "manifest_sha256": checked["identity"]["manifest_sha256"],
                              "hashes_sha256": checked["identity"]["hashes_sha256"],
                              "server_acceptance_request_sha256": checked["request_sha256"]}, sort_keys=True))
            return 0
        if args.authorization_file is None:
            raise RuntimeError("SERVER_ACCEPTANCE_AUTHORIZATION_FILE_REQUIRED")
        if not args.real_name:
            raise RuntimeError("REAL_NAME_REQUIRED_BEFORE_SERVER_WORKLOAD")
        result = run_formal(checked, args.authorization_file, args.real_name,
                            password_stdin=args.password_stdin)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        if result.get("status") in ("complete", "gpu_smoke_test_pass"):
            return 0
        return 1
    except BaseException as exc:
        print(json.dumps({"status": "rejected", "error_type": type(exc).__name__,
                          "exception_chain": traceback.format_exc(),
                          "engineering_job_consumed": False,
                          "token_consumed": False, "automatic_retry": False}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
