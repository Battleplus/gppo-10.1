import json
import os
from pathlib import Path
import runpy
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent

def main():
    started = time.monotonic()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ.pop('W1_PHASE_ACCOUNTING_SOCKET', None)
    sys.path.insert(0, str(ROOT))
    from local_runtime import verify_runtime, gpu_snapshot
    import acceptance_entry as acceptance
    from manifest_contract import verify_package
    request = acceptance._read_json(ROOT / 'LOCAL_ACCEPTANCE_REQUEST.json')
    identity = verify_package(ROOT)
    runtime = verify_runtime(request)
    auth_path = ROOT.parent / 'local-external-authorization.json'
    token = sys.stdin.readline().rstrip('\r\n')
    acceptance.validate_authorization(acceptance._read_json(auth_path), request=request,
        request_sha256=acceptance._sha256(ROOT / 'LOCAL_ACCEPTANCE_REQUEST.json'),
        manifest_sha256=identity['manifest_sha256'], hashes_sha256=identity['hashes_sha256'], token=token)
    token = None
    if ROOT != Path(request['remote_execution_root']):
        raise RuntimeError('LOCAL_FINAL_EXECUTION_ROOT_MISMATCH')
    from isolated_fixture_preflight import verify
    imports = verify()
    initial_gpu = gpu_snapshot()
    if initial_gpu['free_bytes'] < request['gpu']['free_memory_minimum_bytes']:
        raise RuntimeError('LOCAL_FREE_GPU_MEMORY_INSUFFICIENT')
    evidence = Path(request['evidence_directory'])
    evidence.mkdir(mode=0o700, exist_ok=False)
    work = evidence / 'work'
    work.mkdir(mode=0o700)
    meter = acceptance.ResourceMeter(request['limits'], started_wall=started)
    meter.charge_package_root(ROOT)
    meter.bind_paths(evidence, work)
    calls = acceptance.CallBudgetMeter(request['limits']['calls'])
    result = {'classification': 'GPU_SMOKE_TEST', 'platform': 'local-wsl',
        'engineering_job_id': request['engineering_job_id'], 'training_completed': False,
        'full_resource_acceptance': False, 'runtime': runtime, 'module_origins': imports['module_origins'],
        'initial_gpu': initial_gpu, 'identity': {'manifest': identity['manifest_sha256'], 'hashes': identity['hashes_sha256']}}
    def deadline(signum, frame):
        raise RuntimeError('LOCAL_WORKER_WALL_LIMIT')
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, max(0.01, request['limits']['server_wall_seconds'] - (time.monotonic()-started) - 5))
    gpu_meter = None
    try:
        import torch
        calls.record_cuda_context()
        torch.cuda.init()
        if torch.cuda.device_count() != 1:
            raise RuntimeError('LOCAL_LOGICAL_GPU_COUNT_MISMATCH')
        torch.cuda.set_per_process_memory_fraction(request['gpu']['peak_allocated_memory_bytes']/initial_gpu['total_bytes'], 0)
        torch.cuda.reset_peak_memory_stats(0)
        class LocalGpuMeter(acceptance.GpuResourceMeter):
            def check_process_snapshot(self, phase, boundary):
                row = {'phase': phase, 'boundary': boundary, **gpu_snapshot(), **self.check_framework_memory()}
                self.phase_snapshots.append(row)
                return row
            def evidence(self):
                row = super().evidence()
                row['observed_owned_process_memory_peak_bytes'] = None
                row['process_accounting_limitation'] = 'WDDM/WSL unavailable; no exclusivity assertion'
                return row
        gpu_meter = LocalGpuMeter(request['gpu'], None, expected_uuid=initial_gpu['gpu_uuid'])
        gpu_meter.bind_torch(torch)
        results = acceptance._run_bounded_tests(evidence, work, request, meter, calls, gpu_meter)
        torch.cuda.synchronize(0)
        result.update(status='complete', training_completed=True, workload=results,
            device_name=torch.cuda.get_device_name(0))
    except BaseException:
        result.update(status='technical_stop', first_exception=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        progress_files = list((evidence / 'joint-controlled-export').rglob('world-model-training-progress.json'))
        if progress_files:
            progress = acceptance._read_json(progress_files[0])
            routes = progress.get('routes', [])
            result['completed_routes'] = {variant: sum(row.get('variant') == variant and bool(row.get('loss_by_epoch'))
                and bool(row.get('checkpoint_restores')) for row in routes) for variant in ('G1', 'G2')}
            result['training_completed'] = result['completed_routes'] == {'G1': 3, 'G2': 3}
        result['model_call_totals'] = dict(calls.totals)
        result['resource_phases'] = meter.phases
        result['resources'] = meter.sample(force_storage=True)
        if gpu_meter is not None:
            result['gpu_resources'] = gpu_meter.evidence()
        try:
            result['closing_resource_bound'] = meter.check_closing_bound()
            result['measured_limits_pass'] = True
        except BaseException:
            result['measured_limits_pass'] = False
            result['resource_error'] = traceback.format_exc()
            result['status'] = 'technical_stop'
        result['resource_measurement_gaps'] = ['GPU process memory/exclusivity unavailable on WDDM/WSL',
            'final file writes/process exit reserved, not measured here', 'CPU caps sampled, not delegated cgroup hard enforcement']
        acceptance._write_json(evidence / 'local-gpu-result.json', result)
        acceptance._write_json(evidence / 'local-artifact-hashes.json', {
            p.relative_to(evidence).as_posix(): acceptance._sha256(p) for p in evidence.rglob('*')
            if p.is_file() and p.name != 'local-artifact-hashes.json'})
        print(json.dumps({'status': result['status'], 'training_completed': result['training_completed'],
            'result_path': str(evidence / 'local-gpu-result.json'), 'model_call_totals': calls.totals}), flush=True)
    return 0 if result['status'] == 'complete' else 1

if __name__ == '__main__':
    if not sys.flags.isolated:
        os.execv(sys.executable, [sys.executable, '-I', '-B', '-X',
            'pycache_prefix=' + str(Path(sys.prefix) / '.w1-no-bytecode-cache'), str(Path(__file__).resolve())])
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
