"""Freeze the reviewed repair, preserving all evidence outside package."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT/'package'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def inventory(root):return {p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')


def invoke(name,command):
    started=time.monotonic()
    result=subprocess.run(command,capture_output=True,timeout=120)
    (ROOT/(name+'.stdout.raw')).write_bytes(result.stdout)
    (ROOT/(name+'.stderr.raw')).write_bytes(result.stderr)
    (ROOT/(name+'.stdout.txt')).write_text(result.stdout.decode('utf-8','replace'),encoding='utf-8')
    (ROOT/(name+'.stderr.txt')).write_text(result.stderr.decode('utf-8','replace'),encoding='utf-8')
    value={'command':command,'exit_code':result.returncode,'wall_seconds':time.monotonic()-started,
           'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),
           'stderr_sha256':hashlib.sha256(result.stderr).hexdigest()}
    write(ROOT/(name+'.invocation.json'),value)
    if result.returncode:raise RuntimeError('FINAL_PREFLIGHT_FAILED:'+name)
    return value


def main():
    sys.path.insert(0,str(PACKAGE))
    from manifest_contract import write_identity_files,verify_package,verify_external_authorization
    request_path=PACKAGE/'RESOURCE_REQUEST.json'
    request=json.loads(request_path.read_text(encoding='utf-8'))
    assert request['status']=='NOT_APPROVED'
    tests=json.loads((ROOT/'accounting-regressions-evidence.json').read_text(encoding='utf-8'))
    witness=json.loads((ROOT/'production-handshake-acceptance.json').read_text(encoding='utf-8'))
    review=json.loads((ROOT/'phase-repair-luna-review.json').read_text(encoding='utf-8'))
    if not tests['success'] or not witness['all_cases_pass']:
        raise RuntimeError('PRODUCTION_ACCOUNTING_WITNESS_NOT_PASSED')
    if review.get('blocking_findings') != [] or review.get('ready_for_freeze') is not True:
        raise RuntimeError('INDEPENDENT_REVIEW_HAS_BLOCKERS')
    reviewed=review.get('reviewed_source_sha256',review.get('source_identity',{}).get('reviewed_source_sha256',{}))
    required={'budget_ledger.py','phase_handshake.py','runner.py','supervise.py','local_pipeline_controller.py'}
    if not required.issubset(reviewed):raise RuntimeError('REVIEWED_SOURCE_IDENTITIES_MISSING')
    for name,h in reviewed.items():
        if name.endswith('.py') and sha(PACKAGE/name)!=h:raise RuntimeError('SOURCE_CHANGED_AFTER_INDEPENDENT_REVIEW:'+name)
    for name,h in witness['source_after'].items():
        if name.endswith('.py') and sha(PACKAGE/name)!=h:
            raise RuntimeError('PRODUCTION_SOURCE_DIFFERS_FROM_TESTED:'+name)
    request['runner_ready']=True
    write(request_path,request)
    contract_path=PACKAGE/'launch-contract.json'
    contract=json.loads(contract_path.read_text(encoding='utf-8'))
    for name in ('launch-intent.json','launch-evidence.json','authorization-consumed.json','external-token.txt'):
        if (ROOT/name).exists():raise RuntimeError('PROPOSED_IDENTITY_ALREADY_USED:'+name)
    unused=invoke('proposed-native-identity-unused',[
        'wsl','-d','Ubuntu-24.04','--',contract['native_python'],'-I','-B','-c',
        'import pathlib,sys,json; p=pathlib.Path(sys.argv[1]); present=p.parent.exists(); print(json.dumps({"native_proposed_attempt_path_exists":present})); raise SystemExit(1 if present else 0)',contract['native_execution_root']])
    contract['resource_request_sha256']=sha(request_path)
    write(contract_path,contract)
    identities=write_identity_files(PACKAGE,attempt=request['attempt'],entrypoint='run_g1_task_validation.py')
    auth={'schema':'w1-external-launch-authorization/2.0.0','attempt':request['attempt'],**identities,
          'resource_request_sha256':sha(request_path),'status':'NOT_APPROVED','token_sha256':None}
    auth_path=ROOT/'external-authorization.json'
    if auth_path.exists():raise RuntimeError('DO_NOT_OVERWRITE_PRIOR_AUTHORIZATION')
    write(auth_path,auth)
    verify_external_authorization(PACKAGE,auth_path,token=None,preflight_only=True)
    before=inventory(PACKAGE)
    windows=invoke('final-windows-preflight',[
        'C:/Python314/python.exe','-B',str(PACKAGE/'run_g1_task_validation.py'),
        '--preflight-only','--authorization-file',str(auth_path)])
    mount='/mnt/e/Z博士/research-plans/'+ROOT.name
    wsl=invoke('final-wsl-preflight',[
        'wsl','-d','Ubuntu-24.04','--','env','-u','LD_LIBRARY_PATH','-u','LD_PRELOAD',
        '-u','PYTHONPATH','-u','PYTHONHOME','PYTHONNOUSERSITE=1','CUDA_VISIBLE_DEVICES=',
        'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',contract['native_python'],
        '-I','-B','-X','pycache_prefix=/home/asus/.venvs/w1-light-repaired-fair-rerun-py31116/.w1-no-bytecode-cache',
        mount+'/package/local_research_entry.py','--authorization-file',mount+'/external-authorization.json',
        '--preflight-only','--source-preflight'])
    after=inventory(PACKAGE)
    if before!=after:raise RuntimeError('FROZEN_CONTENT_CHANGED')
    archive=ROOT/'frozen-archive'/'package'
    if archive.exists():raise RuntimeError('FROZEN_ARCHIVE_EXISTS')
    shutil.copytree(PACKAGE,archive)
    if inventory(archive)!=after:raise RuntimeError('FULL_ARCHIVE_DIFFERS')
    verify_package(PACKAGE,expected_manifest_sha256=identities['execution_manifest_sha256'],expected_hashes_sha256=identities['hashes_sha256'])
    verify_package(archive)
    value={'attempt':request['attempt'],**identities,'resource_request_sha256':sha(request_path),
           'before':before,'after':after,'file_count':len(after),'all_files_unchanged':True,
           'archive_identical':True,'windows':windows,'wsl':wsl,'staging_started':False,
           'worker_started':False,'authorization_consumed':False,'authorization_status':'NOT_APPROVED',
           'token_created':False,'full_resource_acceptance':False,
           'proposed_identity_unused_check':unused,
           'scope':'final structural/runtime preflight; engineering witness separate; no formal execution'}
    write(ROOT/'final-delivery-evidence.json',value)
    write(ROOT/'frozen-identity.json',{k:v for k,v in value.items() if k not in ('before','after')})
    print(json.dumps({k:v for k,v in value.items() if k not in ('before','after')},ensure_ascii=False))


if __name__=='__main__':main()
