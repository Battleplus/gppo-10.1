"""Actual SSH/SFTP no-model system acceptance; never spends a GPU identity."""
import argparse
import getpass
import hashlib
import json
import re
from pathlib import Path
import sys
import tempfile
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "package"))
import launch_joint_once as connection
import launch_server_acceptance as launcher
from transport_wall_budget import WallBudget

def evidence_target(destination, relative):
    if (not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative
            or relative.startswith('/') or any(part in ('', '.', '..') for part in relative.split('/'))):
        raise RuntimeError('SYSTEM_EVIDENCE_PATH_INVALID')
    path = destination / relative
    if not path.resolve().is_relative_to(destination.resolve()) or path.is_symlink():
        raise RuntimeError('SYSTEM_EVIDENCE_PATH_OUTSIDE_DESTINATION')
    return path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox", required=True)
    parser.add_argument("--final", action="store_true")
    parsed = parser.parse_args()
    sandbox = parsed.sandbox
    match = re.fullmatch(r"/home/user1/w1-gpu-smoke-test-consolidated-acceptance-v3-system-([a-z0-9]{1,32})", sandbox)
    if match is None:
        raise RuntimeError("SYSTEM_SANDBOX_NOT_OWNED")
    suffix = match.group(1)
    checked = launcher.verify_structure()
    password = getpass.getpass("SSH password (not saved): ")
    started = time.monotonic()
    started_cpu = time.process_time()
    wall_budget = WallBudget(checked['request']['limits']['wall_seconds'], closing_reserve=5, started=started)
    client = None
    output = {"classification": "NO_MODEL_SYSTEM_ACCEPTANCE", "formal_attempt_created": False}
    try:
        client = connection.connect(checked["contract"], password, wall_budget=wall_budget)
        password = None
        from bulk_payload_transport import gpu_preflight
        output["gpu"] = gpu_preflight(client, connection.execute)
        body = json.dumps({"nonsecret_transfer_regression": True, "unicode": "标签"}, ensure_ascii=False).encode("utf-8")
        with client.open_sftp() as sftp:
            sftp.mkdir(sandbox, mode=0o700)
            test_path = sandbox + "/nonsecret.json"
            with sftp.open(test_path, "wx") as stream:
                stream.write(body)
            with sftp.open(test_path, "rb") as stream:
                received = stream.read()
            if received != body:
                raise RuntimeError("NONSECRET_TRANSFER_BYTES_MISMATCH")
            try:
                with sftp.open(test_path, "wx") as stream:
                    raise RuntimeError("DUPLICATE_EXCLUSIVE_CREATE_WAS_ACCEPTED")
            except OSError:
                duplicate_rejected = True
            output["transfer"] = {"status": "pass", "bytes": len(body),
                "sha256": hashlib.sha256(received).hexdigest(), "duplicate_rejected": duplicate_rejected,
                "paramiko_open_mode": "wx"}
            with tempfile.TemporaryDirectory() as directory:
                archive = Path(directory) / "package.zip"
                identities = {**checked["identity"]["manifest"]["files"],
                    "execution-manifest.json": checked["identity"]["manifest_sha256"],
                    "hashes.json": checked["identity"]["hashes_sha256"]}
                with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zipped:
                    for name, digest in identities.items():
                        data = (ROOT / "package" / name).read_bytes()
                        if hashlib.sha256(data).hexdigest() != digest:
                            raise RuntimeError("SYSTEM_LOCAL_IDENTITY_CHANGED:" + name)
                        zipped.writestr(name, data)
                sftp.mkdir(sandbox + "/package", mode=0o700)
                sftp.put(str(archive), sandbox + "/package/.frozen-payload.zip")
                archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
        from bulk_payload_transport import EXTRACT_SCRIPT
        extracted = connection.execute(client, [checked["contract"]["native_python"], "-I", "-B", "-c",
            EXTRACT_SCRIPT, sandbox + "/package", archive_hash], timeout=30)
        output["extraction"] = extracted
        if extracted["exit_code"]:
            raise RuntimeError("SYSTEM_EXTRACTION_FAILED:" + extracted["stderr"])
        remaining = wall_budget.remote_allowance(150, remote_closing_reserve=5)
        command = ["env", "-u", "LD_LIBRARY_PATH", "-u", "LD_PRELOAD", "-u", "PYTHONPATH", "-u", "PYTHONHOME",
            "CUDA_VISIBLE_DEVICES=1", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4", "OPENBLAS_NUM_THREADS=4",
            "PYTHONNOUSERSITE=1", checked["contract"]["native_python"], "-I", "-B", "-X",
            "pycache_prefix=/home/user1/.venvs/w1-runtime-v1/.w1-no-bytecode-cache",
            sandbox + "/package/smoke_supervisor_entry.py", "--system-only",
            "--remaining-wall-seconds", repr(remaining), "--evidence-dir", sandbox + "/evidence"]
        output["actual_command"] = command
        output["system"] = connection.execute(client, command, timeout=remaining)
        remote_summary = json.loads(output['system']['stdout'].splitlines()[-1]) if output['system']['stdout'].strip() else {}
        output['remote_summary'] = remote_summary
        # Evidence download is actual SFTP and hash-checked, not a mocked export.
        listing = connection.execute(client, [checked["contract"]["native_python"], "-I", "-B", "-c",
            "import pathlib,json,hashlib,sys; r=pathlib.Path(sys.argv[1]); ps=list(r.rglob('*')); assert not any(p.is_symlink() for p in ps), 'SYSTEM_EVIDENCE_SYMLINK'; print(json.dumps({str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ps if p.is_file()}))",
            sandbox + "/evidence"], timeout=10)
        output["listing"] = listing
        destination = ROOT / ("system-evidence-" + suffix)
        destination.mkdir(exist_ok=False)
        output["download_path"] = str(destination)
        if not listing["exit_code"]:
            files = json.loads(listing["stdout"])
            with client.open_sftp() as sftp:
                for relative, digest in files.items():
                    path = evidence_target(destination, relative)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(sandbox + "/evidence/" + relative, str(path))
                    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                        raise RuntimeError("SYSTEM_EVIDENCE_DOWNLOAD_MISMATCH:" + relative)
            output["download_identities"] = files
        actual = json.loads((destination / "acceptance-result.json").read_text(encoding="utf-8"))
        output["status"] = "pass" if output["system"]["exit_code"] == 0 and actual.get("cuda_initialized") is False and actual.get("status") == "complete" else "failed"
        output["package_identity"] = checked["identity"]["hashes_sha256"]
        output["cuda_initialized"] = actual.get("cuda_initialized")
        elapsed, local_cpu = time.monotonic()-started, time.process_time()-started_cpu
        resources = remote_summary.get('resources') or {}
        remote_wall, remote_cpu = resources.get('wall_seconds'), resources.get('complete_process_cpu_seconds')
        output['resources'] = {'wall_seconds': elapsed, 'controller_cpu_seconds': local_cpu,
            'remote_resources': resources, 'full_ssh_sftp_cpu': None, 'full_resource_acceptance': False}
        if (not isinstance(remote_wall, (int, float)) or not isinstance(remote_cpu, (int, float))
                or elapsed-remote_wall+5 > 30 or local_cpu+2 > 30
                or resources.get('charged_upper_wall_seconds', float('inf')) > 150
                or resources.get('charged_upper_complete_process_cpu_seconds', float('inf')) > 300):
            output['status'] = 'failed'
            output['budget_error'] = 'SYSTEM_RESOURCE_SUBLIMIT_NOT_SATISFIED'
        output['frozen_identity'] = {'manifest_sha256': checked['identity']['manifest_sha256'],
            'hashes_sha256': checked['identity']['hashes_sha256'], 'request_sha256': checked['request_sha256']}
        output['relative_download_path'] = destination.relative_to(ROOT).as_posix()
        if parsed.final and output['status'] == 'pass':
            regression = {**output['transfer'], 'frozen_identity': output['frozen_identity'],
                          'source': 'actual nonsecret SFTP wx, write, readback, duplicate rejection'}
            transfer_path = ROOT / 'prior-transport-regression.json'
            with transfer_path.open('x', encoding='utf-8') as stream:
                json.dump(regression, stream, indent=2)
                stream.write('\n')
            output['transport_evidence_sha256'] = hashlib.sha256(transfer_path.read_bytes()).hexdigest()
            with (ROOT / 'final-system-acceptance.json').open('x', encoding='utf-8') as stream:
                json.dump(output, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
    except BaseException:
        output["status"] = "failed"
        output["exception_chain"] = traceback.format_exc()
    finally:
        password = None
        wall_budget.close()
        if client is not None:
            client.close()
        output["controller_wall_seconds"] = time.monotonic() - started
        name = suffix
        (ROOT / ("system-acceptance-" + name + ".json")).write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "evidence": str(ROOT / ("system-acceptance-" + name + ".json")),
                      "wall": output["controller_wall_seconds"]}))
    return 0 if output["status"] == "pass" else 1

if __name__ == "__main__":
    raise SystemExit(main())
