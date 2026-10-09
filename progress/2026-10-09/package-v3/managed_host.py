"""systemd-owned one-shot hosting. No interactive parent lifetime or restart."""
import json, os, subprocess, time
from pathlib import Path

def ownership_guard():
    if not os.environ.get('INVOCATION_ID') or '.service' not in Path('/proc/self/cgroup').read_text():
        raise RuntimeError('SYSTEMD_PERSISTENT_OWNER_REQUIRED')

def admit_handoff(receipt,unit,stage_wall):
    import math
    for _ in range(50):
        handoff=json.loads(Path(receipt).read_text())
        if handoff.get('state')=='submitted':break
        time.sleep(.1)
    else:raise ValueError('HOSTING_RECEIPT_NOT_FINAL')
    cpu=float(handoff['dispatch_CPU_seconds']);started=float(handoff['entry_started_monotonic'])
    if (handoff.get('unit')!=unit or not math.isfinite(cpu) or not 0<=cpu<=30
        or not math.isfinite(started) or not 0<=time.monotonic()-started<=stage_wall):
        raise ValueError('HANDOFF_ACCOUNTING_INVALID')
    return handoff

def submit(command,*,unit,receipt,wall_cap,working_directory,entry_started_monotonic=None,entry_cpu_started=None):
    import resource
    dispatch_start=time.monotonic() if entry_started_monotonic is None else entry_started_monotonic
    cpu_start=time.process_time() if entry_cpu_started is None else entry_cpu_started
    prior=resource.getrusage(resource.RUSAGE_CHILDREN)
    if not unit.startswith('w1-') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in unit):
        raise ValueError('OWNED_UNIT_NAME')
    linger=subprocess.run(['loginctl','show-user',str(os.getuid()),'-p','Linger','--value'],capture_output=True,text=True,check=True).stdout.strip()
    if linger!='yes':raise RuntimeError('PERSISTENT_USER_MANAGER_LINGER_REQUIRED')
    receipt=Path(receipt);receipt.parent.mkdir(parents=True,exist_ok=True)
    logs=receipt.parent/unit
    logs.mkdir(exist_ok=False)
    # Reserving hosting is independent from authorization consumption; no retries.
    with receipt.open('x',encoding='utf-8') as f:
        json.dump({'unit':unit+'.service','state':'submission_pending','submitted_unix':time.time(),
                   'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                   'restart':'no','formal_token_stored':False,
                   'stdout_file':str(logs/'stdout.log'),'stderr_file':str(logs/'stderr.log')},f);f.flush();os.fsync(f.fileno())
    args=['systemd-run','--user','--unit='+unit,'--property=Restart=no','--property=KillMode=control-group',
          '--property=TimeoutStopSec=30','--property=RuntimeMaxSec='+str(wall_cap+30),
          '--property=CPUAffinity=0','--property=WorkingDirectory='+str(working_directory),
          '--property=StandardOutput=append:'+str(logs/'stdout.log'),
          '--property=StandardError=append:'+str(logs/'stderr.log'),'--property=Nice=10',
          '--setenv=CUDA_VISIBLE_DEVICES=',
          '--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1',
          '--setenv=PYTHONDONTWRITEBYTECODE=1',*map(str,command)]
    result=subprocess.run(args,capture_output=True,text=True,timeout=15)
    child=resource.getrusage(resource.RUSAGE_CHILDREN)
    row=json.loads(receipt.read_text());row.update(state='submitted' if result.returncode==0 else 'submission_failed',
        submission_returncode=result.returncode,submission_stdout=result.stdout,submission_stderr=result.stderr,
        entry_started_monotonic=dispatch_start,dispatch_wall_seconds=time.monotonic()-dispatch_start,
        dispatch_CPU_seconds=time.process_time()-cpu_start+child.ru_utime+child.ru_stime-prior.ru_utime-prior.ru_stime,
        dispatch_CPU_scope='dispatcher SELF and waited systemd-run/loginctl children; manager CPU unavailable')
    temporary=receipt.with_suffix('.pending-json')
    with temporary.open('w',encoding='utf-8') as f:
        f.write(json.dumps(row,sort_keys=True,indent=2)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,receipt)
    if result.returncode:raise RuntimeError('PERSISTENT_SUBMISSION_FAILED:'+result.stderr)
    return row

def observe(unit,work):
    """Read-only: never stop a unit because monitoring ends."""
    fields=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','-p','SubState',
        '-p','MainPID','-p','Result','-p','ExecMainCode','-p','ExecMainStatus'],capture_output=True,text=True,timeout=10)
    row={k:v for line in fields.stdout.splitlines() if '=' in line for k,v in [line.split('=',1)]}
    path=Path(work)/'controller-settlement.json'
    row['controller_final']=json.loads(path.read_text()) if path.exists() else None
    row['termination_attribution']='recorded_by_controller' if path.exists() else 'unknown_until_final_receipt'
    row['no_cleanup_on_monitor_disconnect']=True
    return row
