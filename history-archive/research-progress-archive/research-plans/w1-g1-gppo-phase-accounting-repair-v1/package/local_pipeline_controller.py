"""Native controller includes staging, waits, settlement and hash-verified export."""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

SOURCE = Path(__file__).resolve().parent
EVIDENCE = SOURCE.parent

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
def size(root): return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
def usage():
    a,b=resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN)
    return {'self_cpu_seconds':a.ru_utime+a.ru_stime, 'waited_children_cpu_seconds':b.ru_utime+b.ru_stime,
            'self_plus_waited_cpu_seconds':a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime}
def identities(root): return {p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file()}

def main():
    started=time.monotonic(); phase_start=started; phase_cpu=usage()['self_plus_waited_cpu_seconds']
    token=sys.stdin.readline().rstrip('\r\n')
    sys.path.insert(0,str(SOURCE))
    from manifest_contract import verify_external_authorization, verify_package
    verified=verify_external_authorization(SOURCE,EVIDENCE/'external-authorization.json',token=token,preflight_only=False)
    request,contract=verified['request'],verified['contract']
    native=Path(contract['native_execution_root']); home=native.parent
    if home.exists(): raise RuntimeError('NATIVE_ATTEMPT_PATH_ALREADY_EXISTS_NO_RETRY')
    home.mkdir(mode=0o700)
    phases=[];error=None;code=1;preflight=None;export=None
    try:
        shutil.copytree(SOURCE,native)
        os.chmod(native,0o700)
        shutil.copy2(EVIDENCE/'external-authorization.json',home/'external-authorization.json')
        os.chmod(home/'external-authorization.json',0o600)
        if identities(SOURCE)!=identities(native): raise RuntimeError('NATIVE_STAGING_IDENTITY_MISMATCH')
        verify_external_authorization(native,home/'external-authorization.json',token=token,preflight_only=False)
        argv=[sys.executable,'-I','-B','-X','pycache_prefix='+str(Path(sys.prefix)/'.w1-no-bytecode-cache'),
            str(native/'local_research_entry.py'),'--authorization-file',str(home/'external-authorization.json')]
        before=identities(native)
        checked=subprocess.run(argv+['--preflight-only','--check-token'],input=token+'\n',capture_output=True,text=True,timeout=90)
        after=identities(native)
        (EVIDENCE/'native-preflight.stdout.txt').write_text(checked.stdout,encoding='utf-8')
        (EVIDENCE/'native-preflight.stderr.txt').write_text(checked.stderr,encoding='utf-8')
        preflight={'command':argv+['--preflight-only','--check-token'],'exit_code':checked.returncode,'before':before,'after':after,
            'frozen_contents_unchanged':before==after,'staging_started':False,'worker_started':False,'authorization_consumed':False}
        write(EVIDENCE/'native-preflight-evidence.json',preflight)
        if checked.returncode or before!=after: raise RuntimeError('FINAL_NATIVE_PREFLIGHT_FAILED')
        cpu=usage()['self_plus_waited_cpu_seconds']
        phases.append({'stage':'staging_and_zero_step_gate','wall_seconds':time.monotonic()-phase_start,'cpu_seconds':cpu-phase_cpu})
        cap=request['stages']['staging_and_zero_step_gate']
        if phases[-1]['wall_seconds']>cap['wall_seconds'] or phases[-1]['cpu_seconds']>cap['complete_process_cpu_seconds']:
            raise RuntimeError('STAGING_BUDGET_EXCEEDED')
        env={**os.environ,'W1_LOCAL_STAGING_WALL_SECONDS':str(phases[-1]['wall_seconds']),
             'W1_LOCAL_STAGING_CPU_SECONDS':str(phases[-1]['cpu_seconds'])}
        remaining=request['totals']['wall_seconds']-(time.monotonic()-started)-request['stages']['settlement_and_verified_export']['wall_seconds']
        with (home/'native-launch.stdout.txt').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True,env=env)
            try: proc.communicate(token+'\n',timeout=max(1,remaining));code=proc.returncode
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
                raise RuntimeError('NATIVE_CONTROLLER_WALL_TIMEOUT_NO_RETRY')
        token=None
    except BaseException:
        error=traceback.format_exc(); (home/'controller-first-error.txt').write_text(error,encoding='utf-8')
    finally:
        phase_start=time.monotonic(); phase_cpu=usage()['self_plus_waited_cpu_seconds']
        try:
            # Frozen source identity is checked even after a stopped worker; outputs are not frozen inputs.
            verify_package(native)
            frozen=verified['identity']['manifest']['files']
            frozen_changed=[name for name,h in frozen.items() if digest(native/name)!=h]
            if frozen_changed:raise RuntimeError('FROZEN_NATIVE_INPUT_CHANGED:'+repr(frozen_changed))
            export=EVIDENCE/'verified-export';export.mkdir(exist_ok=False)
            selected=['run-once','supervisor-status.json','resource-history.jsonl','supervisor-console.log',
                'supervisor-phase-protocol.jsonl','supervisor-first-error.json',
                'outer-incomplete-settlement.json','console.log','worker-initialization-failure.json','local-worker-result.json',
                'native-launch.stdout.txt','controller-first-error.txt','authorization-consumed.json']
            hashes={}
            for name in selected:
                source=(home/name if name in ('native-launch.stdout.txt','controller-first-error.txt','authorization-consumed.json') else native/name)
                if not source.exists():continue
                paths=[source] if source.is_file() else [p for p in source.rglob('*') if p.is_file()]
                for file in paths:
                    if file.is_symlink():raise RuntimeError('EXPORT_SYMLINK_REJECTED')
                    relative=Path(name) if source.is_file() else Path(name)/file.relative_to(source)
                    target=export/relative;target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(file,target)
                    h=digest(file)
                    if digest(target)!=h:raise RuntimeError('EXPORT_DIGEST_MISMATCH:'+str(relative))
                    hashes[relative.as_posix()]=h
            write(export/'export-hashes.json',hashes)
            if size(home)>request['totals']['active_storage_bytes'] or size(home)+size(export)>request['totals']['aggregate_native_plus_verified_export_bytes']:
                raise RuntimeError('FINAL_STORAGE_BUDGET_EXCEEDED')
        except BaseException:
            error=(error or '')+'\nEXPORT:\n'+traceback.format_exc();code=1
        phases.append({'stage':'settlement_and_verified_export','wall_seconds':time.monotonic()-phase_start,
                       'cpu_seconds':usage()['self_plus_waited_cpu_seconds']-phase_cpu})
        final_usage=usage();wall=time.monotonic()-started
        cap=request['stages']['settlement_and_verified_export']
        native_cpu_cap=request['totals']['complete_process_cpu_seconds']-request['accounting_reserves']['windows_post_preflight_process_cpu_seconds']
        measured_pass=(wall<=request['totals']['wall_seconds'] and final_usage['self_plus_waited_cpu_seconds']<=native_cpu_cap
            and phases[-1]['wall_seconds']<=cap['wall_seconds'] and phases[-1]['cpu_seconds']<=cap['complete_process_cpu_seconds'])
        summary={'attempt':request['attempt'],'exit_code':code,'error':error,'wall_seconds':wall,
            **final_usage,'controller_phases':phases,'verified_export':str(export) if export else None,
            'active_native_bytes':size(home), 'verified_export_bytes':size(export) if export and export.exists() else None,
            'measured_controller_limits_pass':measured_pass,'native_cpu_cap_excluding_windows_reserve':native_cpu_cap,'full_resource_acceptance':False,
            'resource_gaps':['Windows WSL bridge CPU not measured','controller final write and exit tail not measured',
                'CPU polled; no delegated cgroup hard enforcement'],
            'cpu_no_double_count':'controller SELF + kernel waited CHILDREN; nested supervisor/worker totals reported only, not added',
            'automatic_retry':False}
        if not measured_pass:code=1;summary['exit_code']=1
        write(EVIDENCE/'native-controller-settlement.json',summary)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
