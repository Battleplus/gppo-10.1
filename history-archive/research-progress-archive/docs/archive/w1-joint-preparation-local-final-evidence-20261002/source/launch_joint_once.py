"""Windows SSH controller. Secrets stay in memory; frozen inputs stay immutable."""
from __future__ import annotations
import argparse
import base64
import getpass
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time

from manifest_contract import verify_external_authorization, verify_package, sha256_file
from joint_inputs import verify_joint_inputs
from registration_identity import derive_name_id

ROOT = Path(__file__).resolve().parent
HOST_KEY = "AAAAC3NzaC1lZDI1NTE5AAAAIBc9Q2800Yg0BRYMZXUHlXEj2J4cw5/x9M3yqsac8Sl7"


def connect(contract, password):
    import paramiko
    client = paramiko.SSHClient()
    client.get_host_keys().add(contract["host"], "ssh-ed25519",
        paramiko.Ed25519Key(data=base64.b64decode(HOST_KEY)))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(contract["host"], username=contract["user"], password=password,
                   look_for_keys=False, allow_agent=False, timeout=15, auth_timeout=15,
                   banner_timeout=15)
    return client


def execute(client, argv, *, timeout=180, input_text=None):
    stdin, stdout, stderr = client.exec_command(shlex.join(argv), timeout=timeout)
    if input_text is not None:
        stdin.write(input_text)
        stdin.flush()
    stdin.channel.shutdown_write()
    out, err = stdout.read(), stderr.read()
    code = stdout.channel.recv_exit_status()
    return {"exit_code": code, "stdout": out.decode("utf-8", errors="replace"),
            "stderr": err.decode("utf-8", errors="replace")}


def remote_readonly_preflight(client, contract):
    python = contract["native_python"]
    remote_probe = "/home/user1/w1-remote-runtime-dependency-repair-v1/runtime_preflight.py"
    identity = "/home/user1/w1-remote-runtime-dependency-repair-v1/runtime-identity-v2.json"
    digest_script = ("import hashlib,json,pathlib; paths=" + repr([remote_probe, identity]) +
                     "; print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths}))")
    hashes = execute(client, [python, "-I", "-B", "-c", digest_script], timeout=30)
    if hashes["exit_code"]:
        raise RuntimeError("REMOTE_RUNTIME_SOURCE_UNREADABLE:" + hashes["stderr"])
    actual = json.loads(hashes["stdout"])
    if (actual[remote_probe] != sha256_file(ROOT / "remote_runtime_preflight.py")
            or actual[identity] != contract["runtime_identity_sha256"]):
        raise RuntimeError("REMOTE_RUNTIME_PREFLIGHT_IDENTITY_MISMATCH")
    result = execute(client, ["env", "-u", "LD_LIBRARY_PATH", "-u", "LD_PRELOAD", "-u", "PYTHONPATH", "-u", "PYTHONHOME",
        "PYTHONNOUSERSITE=1", python, "-I", "-B", "-X",
        "pycache_prefix=/home/user1/.venvs/w1-runtime-v1/.w1-no-bytecode-cache",
        remote_probe, "--identity", identity, "--identity-sha256", contract["runtime_identity_sha256"]], timeout=180)
    if result["exit_code"]:
        raise RuntimeError("REMOTE_RUNTIME_PREFLIGHT_FAILED:" + result["stdout"] + result["stderr"])
    consumed = execute(client, [python, "-I", "-B", "-c",
        "import pathlib,json; print(json.dumps({'attempt_path_exists':pathlib.Path(" + repr(contract["native_execution_root"]) + ").exists()}))"], timeout=30)
    if consumed["exit_code"] or json.loads(consumed["stdout"])["attempt_path_exists"]:
        raise RuntimeError("REMOTE_ATTEMPT_CONSUMED_OR_STATE_UNREADABLE")
    return {"runtime": result, "attempt": consumed, "staging_started": False,
            "worker_started": False, "authorization_consumed": False}


def _mkdir_parents(sftp, path):
    from pathlib import PurePosixPath
    parts = PurePosixPath(path).parts
    current = "/"
    for part in parts[1:]:
        current = current.rstrip("/") + "/" + part
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current, mode=0o700)


def upload_once(client, checked, authorization_path):
    """Remote path creation consumes this attempt even if a later copy fails."""
    contract = checked["contract"]
    remote = contract["native_execution_root"]
    with client.open_sftp() as sftp:
        sftp.mkdir(remote, mode=0o700)
        identities = {**checked["identity"]["manifest"]["files"],
                      "execution-manifest.json": checked["identity"]["manifest_sha256"],
                      "hashes.json": checked["identity"]["hashes_sha256"]}
        for relative in sorted(identities):
            target = remote + "/" + relative
            _mkdir_parents(sftp, str(Path(target).parent).replace("\\", "/"))
            sftp.put(str(ROOT / relative), target)
        auth_root = "/home/user1/.w1-external-authorizations"
        _mkdir_parents(sftp, auth_root)
        auth_path = auth_root + "/" + contract["attempt"] + ".json"
        with sftp.open(auth_path, "x") as handle:
            handle.write(Path(authorization_path).read_bytes())
        sftp.chmod(auth_path, 0o600)
    return remote, auth_path


def download_verified_export(client, contract, request):
    """Copy only the sealed manifest allowlist, checking byte counts and hashes."""
    from pathlib import PurePosixPath
    remote_export = contract["native_execution_root"] + "-export"
    destination = ROOT.parents[1] / "runs" / contract["attempt"]
    if destination.exists():
        raise RuntimeError("WINDOWS_EXPORT_PATH_ALREADY_EXISTS_NO_RETRY")
    destination.mkdir(parents=False, exist_ok=False)
    with client.open_sftp() as sftp:
        with sftp.open(remote_export + "/export-manifest.json", "rb") as stream:
            manifest_bytes = stream.read()
        with sftp.open(remote_export + "/EXPORT_COMPLETE.json", "rb") as stream:
            completion_bytes = stream.read()
        manifest, completion = json.loads(manifest_bytes), json.loads(completion_bytes)
        if (manifest.get("verified") is not True or completion.get("verified") is not True
                or hashlib.sha256(manifest_bytes).hexdigest() != completion["export_manifest_sha256"]):
            raise RuntimeError("REMOTE_EXPORT_COMPLETION_BINDING_MISMATCH")
        payload_bytes = 0
        for relative, identity in manifest["files"].items():
            rel = PurePosixPath(relative)
            if rel.is_absolute() or ".." in rel.parts or "\\" in relative:
                raise RuntimeError("REMOTE_EXPORT_PATH_NOT_ALLOWED")
            local = destination.joinpath(*rel.parts)
            local.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote_export + "/" + relative, str(local))
            if local.stat().st_size != identity["bytes"] or sha256_file(local) != identity["sha256"]:
                raise RuntimeError("WINDOWS_EXPORT_IDENTITY_MISMATCH:" + relative)
            payload_bytes += identity["bytes"]
        (destination / "export-manifest.json").write_bytes(manifest_bytes)
        (destination / "EXPORT_COMPLETE.json").write_bytes(completion_bytes)
        with sftp.open(contract["native_execution_root"] + "-final-settlement.json", "rb") as stream:
            final_bytes = stream.read()
        (destination / "remote-final-settlement.json").write_bytes(final_bytes)
    # Three copies now exist: native run, native controlled export, Windows payload.
    if 3 * payload_bytes + len(manifest_bytes) + len(completion_bytes) + len(final_bytes) > request["totals"]["aggregate_native_plus_verified_export_bytes"]:
        raise RuntimeError("AGGREGATE_THREE_COPY_STORAGE_LIMIT_EXCEEDED")
    return {"path": str(destination), "verified_payload_bytes": payload_bytes,
            "remote_final_settlement_sha256": hashlib.sha256(final_bytes).hexdigest(),
            "remote_final_status": json.loads(final_bytes)["status"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-file", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument("--password-stdin", action="store_true", help="automation input; password never logged or stored")
    parser.add_argument("--real-name", help="user-provided Chinese real name or lowercase English real name")
    parser.add_argument("--name-id", help="lowercase English identity generated/provided for server registration")
    args = parser.parse_args()
    # The registration rule requires identity validation before any password
    # prompt, SSH connection, or remote resource probe. Preflight is read-only
    # and deliberately remains usable without a user name.
    if not args.preflight_only and not args.structure_only:
        if not args.real_name:
            raise RuntimeError("REAL_NAME_REQUIRED_BEFORE_SERVER_WORKLOAD")
        generated_name_id = derive_name_id(args.real_name)
        if args.name_id is not None and args.name_id != generated_name_id:
            raise RuntimeError("NAME_ID_DOES_NOT_MATCH_DERIVED_REAL_NAME")
        args.name_id = generated_name_id
    from controller_runtime import verify_controller_runtime
    verify_controller_runtime()
    started_wall, started_cpu = time.monotonic(), time.process_time()
    before = verify_package(ROOT)
    token = os.environ.pop("W1_EXTERNAL_ATTEMPT_TOKEN", None)
    checked = verify_external_authorization(ROOT, args.authorization_file,
        token=token, preflight_only=args.preflight_only)
    if checked["contract"].get("integration_test"):
        raise RuntimeError("FORMAL_ENTRY_REJECTS_INTEGRATION_BACKEND")
    verify_joint_inputs(ROOT)
    if args.structure_only and not args.preflight_only:
        raise RuntimeError("STRUCTURE_ONLY_REQUIRES_PREFLIGHT_ONLY")
    if args.structure_only:
        print(json.dumps({"status": "structure_preflight_pass", "formal_authorization_validated": False,
            "staging_started": False, "worker_started": False, "authorization_consumed": False,
            "manifest_sha256": before["manifest_sha256"], "hashes_sha256": before["hashes_sha256"]}))
        return 0
    password = sys.stdin.readline().rstrip("\r\n") if args.password_stdin else getpass.getpass("Remote SSH password: ")
    client = connect(checked["contract"], password)
    password = None
    try:
        preflight = remote_readonly_preflight(client, checked["contract"])
        if args.preflight_only:
            after = verify_package(ROOT)
            if before["manifest_sha256"] != after["manifest_sha256"] or before["hashes_sha256"] != after["hashes_sha256"]:
                raise RuntimeError("PREFLIGHT_CHANGED_PACKAGE")
            print(json.dumps({"status": "remote_readonly_preflight_pass", "remote": preflight,
                "staging_started": False, "worker_started": False, "authorization_consumed": False,
                "formal_authorization_validated": checked["authorization"]["status"] == "APPROVED",
                "manifest_sha256": after["manifest_sha256"], "hashes_sha256": after["hashes_sha256"]}, ensure_ascii=False))
            return 0
        if not checked["request"].get("runner_ready"):
            raise RuntimeError("JOINT_PRODUCTION_ACCEPTANCE_NOT_COMPLETE")
        if not args.real_name or not args.name_id:
            raise RuntimeError("REAL_NAME_AND_NAME_ID_REQUIRED_BEFORE_SERVER_WORKLOAD")
        remote, authorization = upload_once(client, checked, args.authorization_file)
        result = execute(client, ["env", "-u", "LD_LIBRARY_PATH", "-u", "LD_PRELOAD", "-u", "PYTHONPATH", "-u", "PYTHONHOME",
            "PYTHONNOUSERSITE=1", checked["contract"]["native_python"], "-I", "-B", "-X",
            "pycache_prefix=/home/user1/.venvs/w1-runtime-v1/.w1-no-bytecode-cache",
            remote + "/joint_remote_native.py", "--authorization-file", authorization,
            "--real-name", args.real_name, "--name-id", args.name_id],
            timeout=checked["request"]["totals"]["wall_seconds"], input_text=(token or "") + "\n")
        # Archive failures do not launch a replacement worker or reuse the token.
        exported = download_verified_export(client, checked["contract"], checked["request"])
        remote_final = json.loads((Path(exported["path"]) / "remote-final-settlement.json").read_text(encoding="utf-8"))
        controller_cpu = time.process_time() - started_cpu
        total_wall = time.monotonic() - started_wall
        resource_pass = (controller_cpu <= checked["request"]["accounting_reserves"]["windows_post_preflight_process_cpu_seconds"]
            and controller_cpu + remote_final["native_cpu_total_seconds"] <= checked["request"]["totals"]["complete_process_cpu_seconds"]
            and total_wall <= checked["request"]["totals"]["wall_seconds"])
        settlement = {"status": "complete" if resource_pass and result["exit_code"] == 0 else "technical_stop",
            "controller_cpu_seconds": controller_cpu, "remote_inclusive_cpu_seconds": remote_final["native_cpu_total_seconds"],
            "wall_seconds": total_wall, "resource_pass": resource_pass, "export": exported,
            "automatic_retry": False, "research_success": False}
        (Path(exported["path"]) / "windows-final-settlement.json").write_text(json.dumps(settlement, ensure_ascii=False,
            indent=2, sort_keys=True)+"\n", encoding="utf-8")
        result["settlement"] = settlement
        print(json.dumps(result, ensure_ascii=False))
        return 0 if settlement["status"] == "complete" else 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
