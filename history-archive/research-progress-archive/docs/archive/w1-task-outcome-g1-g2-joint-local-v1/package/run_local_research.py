"""Unique Windows launch: fresh local approval, isolated WSL run, verified export."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT.parent

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2)+'\n', encoding='utf-8')
def inventory():
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in ROOT.rglob('*') if p.is_file()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from manifest_contract import verify_package, verify_external_authorization, AUTHORIZATION_SCHEMA
    from worker_contract import verify_worker_contract
    from joint_inputs import verify_joint_inputs
    before = inventory()
    identity = verify_package(ROOT)
    request = json.loads((ROOT/'RESOURCE_REQUEST.json').read_text(encoding='utf-8'))
    contract = json.loads((ROOT/'launch-contract.json').read_text(encoding='utf-8'))
    attempt = request['attempt']
    verify_worker_contract(ROOT, attempt, identity['manifest_sha256'], identity['hashes_sha256'])
    inputs = verify_joint_inputs(ROOT)
    if args.preflight_only:
        evidence = {'status': 'windows_structure_preflight_pass', 'attempt': attempt,
            'identity': {k:identity[k] for k in ('manifest_sha256','hashes_sha256')},
            'parent_roles': inputs['parent_roles'], 'staging_started': False, 'worker_started': False,
            'authorization_validation': 'structure_only; formal token validated at launch and native worker',
            'authorization_consumed': False, 'before': before, 'after': inventory()}
        assert evidence['before'] == evidence['after'], 'FROZEN_PACKAGE_CHANGED'
        write(EVIDENCE/'windows-preflight-evidence.json', evidence)
        print(json.dumps({k:v for k,v in evidence.items() if k not in ('before','after')}, ensure_ascii=False))
        return 0
    if (EVIDENCE/'launch-evidence.json').exists() or (EVIDENCE/'external-authorization.json').exists():
        raise RuntimeError('LOCAL_ATTEMPT_ALREADY_PREPARED_OR_CONSUMED_NO_RETRY')
    started, cpu = time.monotonic(), time.process_time()
    token = secrets.token_urlsafe(40)
    authorization = {'schema': AUTHORIZATION_SCHEMA, 'status': 'APPROVED', 'attempt': attempt,
        'execution_manifest_sha256': identity['manifest_sha256'], 'hashes_sha256': identity['hashes_sha256'],
        'resource_request_sha256': digest(ROOT/'RESOURCE_REQUEST.json'),
        'token_sha256': hashlib.sha256(token.encode()).hexdigest()}
    auth = EVIDENCE/'external-authorization.json'
    with auth.open('x', encoding='utf-8') as stream: json.dump(authorization, stream, indent=2)
    verify_external_authorization(ROOT, auth, token=token, preflight_only=False)
    write(EVIDENCE/'authorization-binding-evidence.json', {
        'source': 'explicit user authorization for one local real W1 joint run, 2026-10-04',
        'attempt': attempt, 'valid_json': True, 'token_validated': True,
        'request_status_unchanged': request['status']=='NOT_APPROVED',
        'bindings': {key:value for key,value in authorization.items() if key != 'token_sha256'}})
    python = contract['native_python']
    source_script = '/mnt/e/Z博士/research-plans/w1-task-outcome-g1-g2-joint-local-v1/package/local_pipeline_controller.py'
    command = ['wsl','-d','Ubuntu-24.04','--','env','-u','LD_LIBRARY_PATH','-u','LD_PRELOAD',
        '-u','PYTHONPATH','-u','PYTHONHOME','PYTHONNOUSERSITE=1','CUDA_VISIBLE_DEVICES=0',
        'OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',
        python,'-I','-B','-X','pycache_prefix='+str(Path(python).parent.parent).replace('\\','/')+'/.w1-no-bytecode-cache',
        source_script]
    result = subprocess.run(command, input=token+'\n', capture_output=True, text=True,
        encoding='utf-8', errors='replace', timeout=request['totals']['wall_seconds']+10)
    token = None
    (EVIDENCE/'launch.stdout.txt').write_text(result.stdout, encoding='utf-8')
    (EVIDENCE/'launch.stderr.txt').write_text(result.stderr, encoding='utf-8')
    after = inventory()
    wall,win_cpu=time.monotonic()-started,time.process_time()-cpu
    limits_pass=(wall<=request['totals']['wall_seconds'] and win_cpu<=request['accounting_reserves']['windows_post_preflight_process_cpu_seconds'])
    evidence = {'attempt': attempt, 'command': command, 'exit_code': result.returncode,
        'wall_seconds': wall, 'windows_controller_cpu_seconds': win_cpu, 'measured_windows_limits_pass':limits_pass,
        'before': before, 'after': after, 'frozen_contents_unchanged': before==after,
        'resource_gaps': ['Windows WSL bridge process CPU not measured; native controller SELF+waited measured separately'],
        'automatic_retry': False}
    write(EVIDENCE/'launch-evidence.json', evidence)
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('before','after')}, ensure_ascii=False))
    print(result.stdout[-5000:]); print(result.stderr[-2000:])
    if before!=after: raise RuntimeError('FROZEN_PACKAGE_CHANGED')
    return result.returncode if limits_pass else 1

if __name__=='__main__': raise SystemExit(main())
