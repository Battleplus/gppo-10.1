"""One owned supervisor and worker for the entire synthetic GPU pipeline."""
import json
import os
from pathlib import Path
import resource
import runpy
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent

def main():
    raise RuntimeError('HISTORICAL_ENTRY_DISABLED_USE_RUN_G1_TASK_VALIDATION')
    started = time.monotonic()
    # Validate interpreter search paths before project imports, without CUDA.
    probe = runpy.run_path(str(ROOT / "remote_runtime_preflight.py"))
    probe["verify_process_isolation"]()
    sys.path.insert(0, str(ROOT))
    from acceptance_entry import _read_json, _write_json, validate_request, validate_authorization, _manifest_artifacts, _sha256
    from manifest_contract import verify_package
    from supervise import main as supervise
    request = _read_json(ROOT / "SERVER_ACCEPTANCE_REQUEST.json")
    validate_request(request)
    identity = verify_package(ROOT)
    arguments = sys.argv[1:]
    def argument(name):
        return arguments[arguments.index(name) + 1]
    allowance = float(argument("--remaining-wall-seconds"))
    if not 5 < allowance <= request["limits"]["server_wall_seconds"]:
        raise RuntimeError("SMOKE_SUPERVISOR_ALLOWANCE_INVALID")
    system_only = "--system-only" in arguments
    if system_only:
        token = None
        if ROOT == Path(request["remote_execution_root"]):
            raise RuntimeError("SYSTEM_CHECK_CANNOT_USE_GPU_EXECUTION_ROOT")
        evidence_path = Path(argument("--evidence-dir")).resolve()
        if evidence_path != ROOT.parent / "evidence":
            raise RuntimeError("SYSTEM_EVIDENCE_PATH_NOT_ISOLATED")
    else:
        auth_path = Path(argument("--authorization-file")).resolve()
        if auth_path.is_relative_to(ROOT):
            raise RuntimeError("SMOKE_AUTHORIZATION_MUST_BE_EXTERNAL")
        token = sys.stdin.readline()
        validate_authorization(_read_json(auth_path), request=request,
            request_sha256=_sha256(ROOT / "SERVER_ACCEPTANCE_REQUEST.json"),
            manifest_sha256=identity["manifest_sha256"], hashes_sha256=identity["hashes_sha256"],
            token=token.rstrip("\r\n"))
    root = Path(str(ROOT) + ("-system-supervision" if system_only else "-supervision"))
    root.mkdir(mode=0o700, exist_ok=False)
    # Engineering supervision envelope only. Inner acceptance applies every
    # per-call/profile/GPU/storage cap. No research request is edited or spent.
    cap = {"wall_seconds": allowance,
        "complete_process_cpu_seconds": request["limits"]["server_complete_process_cpu_seconds"],
        "active_storage_bytes": request["limits"]["active_storage_bytes"],
        "all_resident_rss_bytes": request["limits"]["rss_peak_bytes"]}
    _write_json(root / "RESOURCE_REQUEST.json", {
        "status": "NOT_APPROVED", "classification": "GPU_SMOKE_TEST",
        "totals": cap, "stages": {"staging_and_zero_step_gate": cap},
        "accounting_reserves": {"cross_system_wall_seconds": 5}})
    _write_json(root / "experiment-matrix.json", {"world_model_device": "cpu",
        "scope": "outer process monitoring; inner worker enforces frozen GPU contract"})
    os.environ["W1_SMOKE_WORKER_ARGUMENTS_JSON"] = json.dumps(arguments)
    try:
        code = supervise(root, str(ROOT / "smoke_worker.py"), worker_input=token,
                         sample_interval=0.25)
    finally:
        token = None
        os.environ.pop("W1_SMOKE_WORKER_ARGUMENTS_JSON", None)
    state = _read_json(root / "supervisor-status.json")
    console = (root / "console.log").read_text(encoding="utf-8", errors="replace") if (root / "console.log").is_file() else ""
    lines = [line for line in console.splitlines() if line.strip()]
    try:
        result = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        result = {"status": "technical_stop", "engineering_job_consumed": True,
                  "first_worker_output": console, "training_completion": "not_completed"}
    evidence = Path(argument("--evidence-dir"))
    own = resource.getrusage(resource.RUSAGE_SELF)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    measured = {"wall_seconds": time.monotonic() - started,
        "complete_process_cpu_seconds": own.ru_utime + own.ru_stime + child.ru_utime + child.ru_stime,
        "scope": "supervisor SELF plus waited CHILDREN once; worker not separately added",
        "tail_scope": "final evidence writes and process exit covered by unchanged 5s wall/3s CPU reserve"}
    resources = result.get("resources") or {}
    resources.update(measured)
    resources["charged_upper_wall_seconds"] = measured["wall_seconds"] + 5
    resources["charged_upper_complete_process_cpu_seconds"] = measured["complete_process_cpu_seconds"] + 3
    result["resources"] = resources
    result["supervisor"] = state
    if code or resources["charged_upper_wall_seconds"] > allowance or resources["charged_upper_complete_process_cpu_seconds"] > cap["complete_process_cpu_seconds"]:
        result["status"] = "technical_stop"
    if evidence.exists():
        _write_json(evidence / "gpu-pipeline-supervisor.json", {"supervisor": state, "resources": measured})
        (evidence / "gpu-pipeline-console.log").write_text(console, encoding="utf-8")
        payload = _read_json(evidence / "acceptance-result.json") if (evidence / "acceptance-result.json").is_file() else {
            "classification": "GPU_SMOKE_TEST", "engineering_job_id": request["engineering_job_id"],
            "run_status": "technical_stop"}
        payload["outer_supervisor"] = state
        payload["outer_resources"] = measured
        if system_only:
            if any(p.is_symlink() for p in evidence.rglob('*')):
                raise RuntimeError('SYSTEM_EVIDENCE_SYMLINK')
            files = {p.relative_to(evidence).as_posix(): {"bytes": p.stat().st_size, "sha256": _sha256(p)}
                     for p in evidence.rglob('*') if p.is_file() and p.name != 'acceptance-evidence-manifest.json'}
            _write_json(evidence / 'acceptance-evidence-manifest.json', {
                'schema': 'w1-no-model-system-evidence/1.0.0', 'verified': True,
                'engineering_job_consumed': False, 'formal_research_attempt_created': False,
                'package_identity': identity['hashes_sha256'], 'files': files})
        else:
            _manifest_artifacts(evidence, payload)
        result["evidence_manifest_sha256"] = _sha256(evidence / "acceptance-evidence-manifest.json")
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") == "complete" else 1

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
