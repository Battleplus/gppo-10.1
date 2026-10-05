import argparse
import json
import os
from pathlib import Path
import secrets
import sys
import time

ROOT = Path(__file__).resolve().parent

def main():
    raise RuntimeError('HISTORICAL_ENTRY_DISABLED_USE_RUN_G1_TASK_VALIDATION')
    started = time.monotonic()
    sys.path.insert(0, str(ROOT))
    import acceptance_entry as acceptance
    from manifest_contract import verify_package
    from local_runtime import verify_runtime, gpu_snapshot
    parsed = argparse.ArgumentParser()
    parsed.add_argument('--preflight-only', action='store_true')
    parsed.add_argument('--system-only', action='store_true')
    args = parsed.parse_args()
    request = acceptance._read_json(ROOT / 'LOCAL_ACCEPTANCE_REQUEST.json')
    if (request['status'] != 'NOT_APPROVED' or request['engineering_job_id'] != 'w1-gpu-smoke-local-v1-once'
            or request['automatic_retry'] is not False or request['research_training'] is not False):
        raise RuntimeError('LOCAL_REQUEST_SCOPE_INVALID')
    if Path(request['remote_execution_root']) != ROOT:
        raise RuntimeError('LOCAL_ROOT_IDENTITY_MISMATCH')
    identity = verify_package(ROOT)
    if identity['manifest']['attempt'] != request['engineering_job_id']:
        raise RuntimeError('LOCAL_ATTEMPT_IDENTITY_MISMATCH')
    runtime = verify_runtime(request)
    if args.preflight_only:
        print(json.dumps({'status': 'structure_preflight_pass', 'manifest': identity['manifest_sha256'],
            'hashes': identity['hashes_sha256'], 'staging_started': False, 'worker_started': False,
            'cuda_initialized': runtime['cuda_initialized'], 'authorization_consumed': False}))
        return 0
    if args.system_only:
        from system_acceptance import run
        return run(ROOT.parent / 'evidence')
    system = acceptance._read_json(ROOT.parent / 'evidence/acceptance-result.json')
    if (system['status'] != 'complete' or system['package_identity'] != identity['hashes_sha256']
            or system['cuda_initialized'] is not False or any(system[key] for key in ('errors', 'failures', 'skipped'))):
        raise RuntimeError('LOCAL_FINAL_SYSTEM_ACCEPTANCE_REQUIRED')
    limits = request['limits']
    if (limits['wall_seconds'] != 180 or limits['complete_process_cpu_seconds'] != 360
            or limits['calls'] != {**acceptance.EXPECTED_COUNTERS, 'cuda_context_initializations': 1}
            or request['gpu']['physical_device'] != 0 or request['gpu']['peak_allocated_memory_bytes'] != 4*1024**3):
        raise RuntimeError('LOCAL_BUDGET_CONTRACT_MISMATCH')
    gpu = gpu_snapshot()
    if gpu['free_bytes'] < request['gpu']['free_memory_minimum_bytes']:
        raise RuntimeError('LOCAL_FREE_GPU_MEMORY_INSUFFICIENT')
    from supervise import main as supervise
    envelope = Path(str(ROOT)+'-supervision')
    envelope.mkdir(mode=0o700, exist_ok=False)
    caps = {'wall_seconds': limits['server_wall_seconds'],
        'complete_process_cpu_seconds': limits['server_complete_process_cpu_seconds'],
        'active_storage_bytes': limits['active_storage_bytes'], 'all_resident_rss_bytes': limits['rss_peak_bytes']}
    acceptance._write_json(envelope / 'RESOURCE_REQUEST.json', {
        'status': 'NOT_APPROVED', 'classification': 'GPU_SMOKE_TEST', 'totals': caps,
        'stages': {'staging_and_zero_step_gate': caps}, 'accounting_reserves': {'cross_system_wall_seconds': 5}})
    acceptance._write_json(envelope / 'experiment-matrix.json', {'world_model_device': 'cpu',
        'scope': 'outer CPU/RSS/process supervision; local worker enforces GPU limits'})
    token = secrets.token_urlsafe(32)
    import hashlib
    authorization = {'schema': acceptance.AUTH_SCHEMA, 'status': 'APPROVED',
        'engineering_job_id': request['engineering_job_id'], 'remote_execution_root': request['remote_execution_root'],
        'evidence_directory': request['evidence_directory'],
        'server_acceptance_request_sha256': acceptance._sha256(ROOT / 'LOCAL_ACCEPTANCE_REQUEST.json'),
        'execution_manifest_sha256': identity['manifest_sha256'], 'hashes_sha256': identity['hashes_sha256'],
        'authorization_id': 'user-authorized-local-gpu-smoke-v1',
        'one_time_token_sha256': hashlib.sha256(token.encode()).hexdigest(),
        'formal_research_attempt_created': False, 'automatic_retry': False}
    auth = ROOT.parent / 'local-external-authorization.json'
    with auth.open('x', encoding='utf-8') as stream:
        json.dump(authorization, stream, indent=2)
    acceptance.validate_authorization(acceptance._read_json(auth), request=request,
        request_sha256=authorization['server_acceptance_request_sha256'],
        manifest_sha256=identity['manifest_sha256'], hashes_sha256=identity['hashes_sha256'], token=token)
    code = supervise(envelope, str(ROOT / 'local_gpu_worker.py'), worker_input=token+'\n', sample_interval=0.1)
    token = None
    state = acceptance._read_json(envelope / 'supervisor-status.json')
    summary = {'classification': 'GPU_SMOKE_TEST', 'returncode': code, 'supervisor': state,
        'wall_seconds': time.monotonic()-started, 'training_result': request['evidence_directory']+'/local-gpu-result.json'}
    acceptance._write_json(ROOT.parent / 'local-supervisor-result.json', summary)
    print(json.dumps(summary), flush=True)
    return code

if __name__ == '__main__':
    raise SystemExit(main())
