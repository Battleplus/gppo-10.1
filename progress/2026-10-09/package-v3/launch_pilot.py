"""Unique native-server launch; verification precedes external authorization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
RUNTIME='/home/user1/w1-runtimes/w1-py31116-torch270cu128-v2/w1-light-repaired-fair-rerun-py31116/bin/python'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        while chunk:=stream.read(1024*1024):h.update(chunk)
    return h.hexdigest()
def verify(root):
    for path in root.rglob('*'):
        if (path.is_dir() and path.name == '__pycache__') or (path.is_file() and path.suffix.lower() in {'.pyc', '.pyo'}):
            raise ValueError('BYTECODE_CACHE_IN_PUBLISH_TREE:' + str(path.relative_to(root)))
    identity=read(root/'frozen-identity.json')
    for name,key in [('execution-manifest.json','manifest_sha256'),('hashes.json','hashes_sha256'),('RESOURCE_REQUEST.json','resource_request_sha256')]:
        if sha(root/name)!=identity[key]:raise ValueError('FROZEN_IDENTITY_CHANGED:'+name)
    inventory=read(root/'hashes.json')['files']
    for rel,entry in inventory.items():
        p=(root/rel).resolve()
        p.relative_to(root.resolve())
        if p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:raise ValueError('FROZEN_FILE_CHANGED:'+rel)
    manifest=read(root/'execution-manifest.json')
    if manifest['unique_entry']!='launch_pilot.py' or manifest['formal_fixture_allowed'] is not False:raise ValueError('FORMAL_ENTRY_CONTRACT')
    request=read(root/'RESOURCE_REQUEST.json');matrix=read(root/'package/experiment-matrix.json')
    if sha(root/'package/RESOURCE_REQUEST.json')!=sha(root/'RESOURCE_REQUEST.json'):raise ValueError('RESOURCE_COPY_DISAGREEMENT')
    if matrix.get('controlled') or matrix['world_model_device']!='cpu':raise ValueError('FORMAL_FIXTURE_OR_DEVICE')
    if matrix['attempt']!=identity['attempt'] or request['attempt']!=identity['attempt']:raise ValueError('ATTEMPT_IDENTITY')
    if request['stage_order']!=manifest['stage_order'] or set(request['stage_order'])!=set(request['stages']):raise ValueError('STAGE_NAMESPACE')
    sys.path.insert(0,str(root/'package'))
    from stage_contract import STAGE_ORDER
    if request['stage_order']!=list(STAGE_ORDER):raise ValueError('REUSE_STAGE_ORDER')
    if (matrix.get('research_mode')!='multihead_ranking_critic_combination'
            or matrix.get('world_model_training_enabled') is not True
            or matrix.get('critic_feature_contract',{}).get('packed_dim')!=507
            or matrix.get('phase')!='critic_only'):raise ValueError('CONSEQUENCE_CRITIC_CONTRACT_REQUIRED')
    if (matrix.get('candidate_key_contract') != 'w1-structured-candidate/1'
            or manifest.get('candidate_key_contract') != 'candidate-key-contract.json'
            or read(root/'candidate-key-contract.json').get('schema') != 'w1-structured-candidate/1'):
        raise ValueError('STRUCTURED_CANDIDATE_CONTRACT_REQUIRED')
    if (matrix.get('public_prefix_contract') != 'w1-public-prefix/1'
            or manifest.get('public_prefix_contract') != 'PUBLIC_PREFIX_CONTRACT.json'
            or read(root/'PUBLIC_PREFIX_CONTRACT.json').get('schema') != 'w1-public-prefix/1'
            or read(root/'flat-layout-manifest.json').get('dimension') != 770):
        raise ValueError('UNIFIED_PUBLIC_PREFIX_CONTRACT_REQUIRED')
    if matrix['world_seeds']!=[8201,8202,8203] or matrix['policy_seeds']!=[8301,8302,8303]:raise ValueError('FIXED_SEEDS')
    if request['approval_status']!='NOT_APPROVED':raise ValueError('FROZEN_REQUEST_MUST_NOT_CONTAIN_APPROVAL')
    return identity,request

def approved(path,identity,request):
    if path is None:raise ValueError('EXTERNAL_AUTHORIZATION_REQUIRED; current resource request is NOT_APPROVED')
    p=Path(path).resolve();p.relative_to(p.anchor)
    try:p.relative_to(ROOT.resolve())
    except ValueError:pass
    else:raise ValueError('AUTHORIZATION_MUST_BE_OUTSIDE_FROZEN_PACKAGE')
    approval=read(p)
    if (approval.get('approval_status')!='APPROVED' or approval.get('attempt')!=identity['attempt']
            or any(approval.get(k)!=identity[k] for k in ('manifest_sha256','hashes_sha256','resource_request_sha256'))
            or approval.get('totals')!=request['totals'] or approval.get('stages')!=request['stages']
            or approval.get('external_transport_limits')!=request['external_transport_limits']):
        raise ValueError('EXTERNAL_AUTHORIZATION_BINDING_MISMATCH')
    token=approval.get('one_time_token')
    if not isinstance(token,str) or len(token)<32:raise ValueError('EXTERNAL_ONE_TIME_TOKEN_REQUIRED')
    return hashlib.sha256(token.encode()).hexdigest(),sha(p)

def main(argv=None):
    started=time.monotonic();cpu=time.process_time()
    if os.name=='posix':os.sched_setaffinity(0,{0})
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--managed-execution',action='store_true');parser.add_argument('--authorization');parser.add_argument('--work-root',default='/home/user1/w1-pilot/runs')
    args=parser.parse_args(argv)
    if str(Path(args.work_root))!='/home/user1/w1-pilot/runs':raise ValueError('FROZEN_WORK_ROOT_REQUIRED')
    identity,request=verify(ROOT)
    from server_runtime_contract import verify as verify_runtime
    verify_runtime(ROOT)
    work=Path(args.work_root).resolve()/identity['attempt']
    if work.exists() or (ROOT/'exports'/identity['attempt']).exists():raise ValueError('ATTEMPT_ALREADY_STAGED_OR_CONSUMED')
    if args.preflight:
        print(json.dumps({'status':'preflight_pass','attempt_unconsumed':True,'authorization_status':request['approval_status'],
            'research_started':False,'input_identity':identity},ensure_ascii=False));return 0
    token_sha,auth_sha=approved(args.authorization,identity,request)
    if os.name!='posix' or str(Path(sys.executable))!=RUNTIME:raise ValueError('FROZEN_ISOLATED_SERVER_INTERPRETER_REQUIRED')
    from managed_host import submit,ownership_guard,admit_handoff
    if not args.managed_execution:
        unit=identity['attempt']
        receipt=Path('/home/user1/w1-pilot/hosting')/(unit+'.json')
        result=submit([sys.executable,'-B',str(ROOT/'launch_pilot.py'),'--managed-execution',
                       '--authorization',str(Path(args.authorization).resolve())],unit=unit,receipt=receipt,
                       wall_cap=request['totals']['wall_seconds'],working_directory=ROOT,
                       entry_started_monotonic=started,entry_cpu_started=cpu)
        print(json.dumps({'status':'persistent_handoff','attempt':identity['attempt'],'unit':result['unit'],
                          'receipt':str(receipt),'research_result':'pending','restart':'no'}));return 0
    ownership_guard()
    hosting_receipt=Path('/home/user1/w1-pilot/hosting')/(identity['attempt']+'.json')
    handoff=admit_handoff(hosting_receipt,identity['attempt']+'.service',request['stages'][request['stage_order'][0]]['wall_seconds'])
    dispatch_cpu=float(handoff['dispatch_CPU_seconds'])
    original_started=float(handoff['entry_started_monotonic'])
    started=original_started
    work.parent.mkdir(parents=True,exist_ok=True);work.mkdir(exist_ok=False)
    consumed={'attempt':identity['attempt'],'authorization_sha256':auth_sha,'token_sha256':token_sha,'status':'consumed_no_retry'}
    (work/'consumed-identity.json').write_text(json.dumps(consumed,sort_keys=True)+'\n',encoding='utf-8')
    code=1;first=None
    import resource
    child_start=resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        for name in ('package','execution-manifest.json','hashes.json','RESOURCE_REQUEST.json','frozen-identity.json','launch_pilot.py'):
            src=ROOT/name;dst=work/name
            if src.is_dir():shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pyo'))
            else:shutil.copy2(src,dst)
        # Other frozen documents are copied as bytes so the staged inventory is identical.
        for rel in read(ROOT/'hashes.json')['files']:
            src=ROOT/rel;dst=work/rel
            if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        verify(work)
        elapsed=time.monotonic()-started;used_cpu=time.process_time()-cpu+dispatch_cpu
        stage=request['stages'][request['stage_order'][0]]
        if elapsed>stage['wall_seconds'] or used_cpu>stage['complete_process_cpu_seconds']:raise ValueError('CONTROLLER_STAGING_BUDGET_EXCEEDED')
        env={**os.environ,'W1_LOCAL_STAGING_WALL_SECONDS':str(elapsed),'W1_LOCAL_STAGING_CPU_SECONDS':str(used_cpu),
            'W1_VERIFIED_ATTEMPT':identity['attempt'],'W1_SUPERVISED_PILOT':'1','PYTHONDONTWRITEBYTECODE':'1',
            'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        receipt={'attempt':identity['attempt'],'authorization_sha256':auth_sha,'token_sha256':token_sha,
                 'manifest_sha256':identity['manifest_sha256'],'authorization_verified':True}
        command=[sys.executable,'-B','-c','import supervise; raise SystemExit(supervise.main(runner="production_driver.py",worker_input=__import__("sys").stdin.buffer.read()))']
        from tail_reserve import worker_wall_allowance
        allowance=worker_wall_allowance(request,elapsed)
        if allowance<=0:raise TimeoutError('NO_WORKER_BUDGET_AFTER_EXPORT_RESERVE')
        process=subprocess.Popen(command,cwd=work/'package',env=env,stdin=subprocess.PIPE,start_new_session=True)
        try:
            process.communicate(json.dumps(receipt).encode(),timeout=allowance)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise TimeoutError('CONTROLLER_TOTAL_WALL_STOP')
        code=process.returncode
    except BaseException:
        import traceback
        first=traceback.format_exc()
        (work/'controller-first-error.txt').write_text(first,encoding='utf-8')
    finally:
        destination=ROOT/'exports'/identity['attempt']
        export_start=time.monotonic();export_cpu=time.process_time()
        prior=read(work/'package/supervisor-status.json') if (work/'package/supervisor-status.json').exists() else {}
        tail=prior.get('phase_tail',{})
        if not tail:
            tail=next((r for r in reversed(prior.get('phase_rows',[])) if r.get('stage')=='settlement_and_verified_export'),{})
        cap=request['stages']['settlement_and_verified_export']
        left_wall=min(cap['wall_seconds']-float(tail.get('stage_wall_seconds') or 0),request['totals']['wall_seconds']-30-(export_start-started))
        children=resource.getrusage(resource.RUSAGE_CHILDREN)
        export_child_cpu_start=children.ru_utime+children.ru_stime
        measured_cpu=dispatch_cpu+(time.process_time()-cpu)+(children.ru_utime+children.ru_stime-child_start.ru_utime-child_start.ru_stime)
        left_cpu=min(cap['complete_process_cpu_seconds']-float(tail.get('stage_cpu_seconds') or 0),request['totals']['complete_process_cpu_seconds']-measured_cpu)
        try:
            if left_wall<=0 or left_cpu<=0:raise ValueError('NO_EXPORT_BUDGET_REMAINS; Linux evidence retained')
            exported=subprocess.run([sys.executable,'-B',str(ROOT/'export_artifacts.py'),str(work),str(destination),
                '--wall',str(left_wall),'--cpu',str(left_cpu),'--storage',str(request['controller_limits']['export_destination_bytes'])],
                capture_output=True,text=True,timeout=left_wall)
            if exported.returncode:raise RuntimeError('EXPORT_FAILED:'+exported.stderr)
        except BaseException:
            import traceback
            code=1
            (work/'controller-export-error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        children=resource.getrusage(resource.RUSAGE_CHILDREN)
        total_cpu=dispatch_cpu+time.process_time()-cpu+children.ru_utime+children.ru_stime-child_start.ru_utime-child_start.ru_stime
        export_measured_cpu=time.process_time()-export_cpu+children.ru_utime+children.ru_stime-export_child_cpu_start
        result={'status':'complete' if code==0 else 'stopped','exitcode':code,
            'measured_Linux_CPU_seconds':total_cpu,'wall_seconds':time.monotonic()-started,
            'export_wall_seconds':time.monotonic()-export_start,'export_CPU_seconds':export_measured_cpu,
            'dispatch_CPU_seconds':dispatch_cpu,'hosting_receipt':str(hosting_receipt),
            'CPU_scope':'dispatcher SELF/waited children + controller SELF/waited supervisor/export descendants counted once',
            'resource_scope_complete':False,'unmeasured':['SSH/client transfer outside server research scope','systemd user-manager CPU','controller exit tail'],
            'reserves_are_not_measurements':True,'first_controller_error':first,
            'local_export':str(destination) if destination.exists() else None,'Linux_evidence_retained':str(work)}
        if (result['wall_seconds']>request['totals']['wall_seconds'] or total_cpu>request['totals']['complete_process_cpu_seconds']
                or result['export_wall_seconds']>left_wall or export_measured_cpu>left_cpu):
            code=1;result.update(status='resource_stop',exitcode=1)
        (work/'controller-settlement.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf-8')
        if destination.exists():(destination/'controller-settlement.json').write_bytes((work/'controller-settlement.json').read_bytes())
    return code

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc();raise SystemExit(1)
