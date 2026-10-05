"""Unique task launch: externally approved JSON and token required."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parent
EVIDENCE=ROOT.parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inventory():return {p.relative_to(ROOT).as_posix():digest(p) for p in ROOT.rglob('*') if p.is_file()}
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--authorization-file')
    args=parser.parse_args();sys.path.insert(0,str(ROOT))
    from manifest_contract import verify_package,verify_external_authorization
    from worker_contract import verify_worker_contract
    from task_contract import verify_task_inputs
    before=inventory();identity=verify_package(ROOT)
    request=json.loads((ROOT/'RESOURCE_REQUEST.json').read_text(encoding='utf-8'))
    contract=json.loads((ROOT/'launch-contract.json').read_text(encoding='utf-8'))
    verify_worker_contract(ROOT,request['attempt'],identity['manifest_sha256'],identity['hashes_sha256'])
    inputs=verify_task_inputs(ROOT)
    if args.preflight_only:
        if args.authorization_file:verify_external_authorization(ROOT,Path(args.authorization_file),token=None,preflight_only=True)
        after=inventory();assert before==after,'FROZEN_PACKAGE_CHANGED'
        value={'status':'windows_structure_preflight_pass','inputs':inputs,'before':before,'after':after,'exit_code':0,
            'staging_started':False,'worker_started':False,'authorization_consumed':False,'authorization_validation':'structure only; NOT_APPROVED is not approval'}
        write(EVIDENCE/'windows-preflight-evidence.json',value)
        print(json.dumps({k:v for k,v in value.items() if k not in ('before','after')},ensure_ascii=False));return 0
    if not args.authorization_file:raise RuntimeError('EXPLICIT_EXTERNAL_AUTHORIZATION_FILE_REQUIRED')
    auth=Path(args.authorization_file).resolve()
    if auth!=(EVIDENCE/'external-authorization.json').resolve():raise RuntimeError('UNIQUE_EXTERNAL_AUTHORIZATION_PATH_REQUIRED')
    token=sys.stdin.readline(4097).rstrip('\r\n')
    if not token or len(token)>4096 or sys.stdin.read(1):raise RuntimeError('TOKEN_INPUT_INVALID')
    verify_external_authorization(ROOT,auth,token=token,preflight_only=False)
    if (EVIDENCE/'launch-evidence.json').exists() or (EVIDENCE/'launch-intent.json').exists():raise RuntimeError('ATTEMPT_ALREADY_LAUNCHED_NO_RETRY')
    started=time.monotonic();cpu=time.process_time()
    command=['wsl','-d','Ubuntu-24.04','--','env','-u','LD_LIBRARY_PATH','-u','LD_PRELOAD','-u','PYTHONPATH','-u','PYTHONHOME',
        'PYTHONNOUSERSITE=1','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
        contract['native_python'],'-I','-B','-X','pycache_prefix=/home/asus/.venvs/w1-light-repaired-fair-rerun-py31116/.w1-no-bytecode-cache',
        contract['native_source_mount']+'/local_pipeline_controller.py']
    intent={'attempt':request['attempt'],'manifest_sha256':identity['manifest_sha256'],'hashes_sha256':identity['hashes_sha256'],'automatic_retry':False}
    with (EVIDENCE/'launch-intent.json').open('x',encoding='utf-8') as handle:
        json.dump(intent,handle,sort_keys=True);handle.write('\n')
    code=1;error=None;stdout='';stderr=''
    try:
        proc=subprocess.run(command,input=token+'\n',capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=request['totals']['wall_seconds'])
        code=proc.returncode;stdout=proc.stdout;stderr=proc.stderr
    except BaseException:
        error=traceback.format_exc();stderr=error
    finally:
        token=None
    (EVIDENCE/'launch.stdout.txt').write_text(stdout,encoding='utf-8');(EVIDENCE/'launch.stderr.txt').write_text(stderr,encoding='utf-8')
    value={'attempt':request['attempt'],'command':command,'exit_code':code,'error':error,'wall_seconds':time.monotonic()-started,
        'windows_controller_cpu_seconds':time.process_time()-cpu,'frozen_contents_unchanged':before==inventory(),
        'automatic_retry':False,'resource_gaps':['WSL bridge CPU and cross-clock reconciliation unmeasured']}
    value['windows_cpu_reserve_pass']=value['windows_controller_cpu_seconds']<=request['accounting_reserves']['windows_post_preflight_process_cpu_seconds']
    value['measured_wall_pass']=value['wall_seconds']<=request['totals']['wall_seconds']
    value['full_resource_acceptance']=False
    if not value['windows_cpu_reserve_pass'] or not value['measured_wall_pass']:
        code=1;value['exit_code']=1;value['error']=(value['error'] or '')+'WINDOWS_MEASURED_BUDGET_EXCEEDED'
    write(EVIDENCE/'launch-evidence.json',value)
    if not value['frozen_contents_unchanged']:raise RuntimeError('FROZEN_PACKAGE_CHANGED')
    return code
if __name__=='__main__':raise SystemExit(main())
