"""One-shot Linux supervision; race-safe size, fail-closed cleanup, PDEATHSIG."""
import fcntl,json,os,resource,signal,stat,subprocess,sys,time,shutil,traceback
from pathlib import Path
from infra_io import durable_append_jsonl,durable_atomic_json,require_native_linux_filesystem
ROOT=Path(__file__).resolve().parent
MODULE_STARTED=time.monotonic()

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def size(root):
    total=0
    if not root.exists():return 0
    for p in root.rglob('*'):
        try:info=p.stat()
        except FileNotFoundError:continue
        if stat.S_ISREG(info.st_mode):total+=info.st_size
    return total

def artifacts(root):
    total=size(root/'run-once')
    for name in ('console.log','supervisor-status.json','resource-history.jsonl','supervisor-console.log','STOP_REQUESTED.json'):
        try:total+=(root/name).stat().st_size
        except FileNotFoundError:pass
    return total

def proc_tree(root):
    rows={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            data=(p/'stat').read_text();v=data[data.rfind(')')+2:].split();pid=int(p.name)
            rss=int(v[21])*os.sysconf('SC_PAGE_SIZE');peak=rss
            for line in (p/'status').read_text().splitlines():
                if line.startswith('VmHWM:'):peak=int(line.split()[1])*1024
            rows[pid]={'ppid':int(v[1]),'ticks':int(v[11])+int(v[12]),'start':int(v[19]),'rss':rss,'peak':peak}
        except (FileNotFoundError,ProcessLookupError):continue
    found={root}
    while True:
        new=found|{p for p,v in rows.items() if v['ppid'] in found}
        if new==found:break
        found=new
    return {p:rows[p] for p in found if p in rows}

def machine_guard(root,startup=False):
    available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    free=shutil.disk_usage(root).free
    if available<(1280 if startup else 512)*1024**2:raise RuntimeError('Host available RAM guard')
    if free<(8 if startup else 2)*1024**3:raise RuntimeError('Host disk free-space guard')
    return {'available_ram_bytes':available,'disk_free_bytes':free}

def cpu_idle():
    out={}
    for line in Path('/proc/stat').read_text().splitlines():
        v=line.split()
        if v[0].startswith('cpu') and v[0][3:].isdigit():
            n=list(map(int,v[1:9]));out[int(v[0][3:])]=(sum(n),n[3]+n[4])
    return out

def stop_owned_group(worker,grace=30):
    try:os.killpg(worker.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:worker.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:os.killpg(worker.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        worker.wait(timeout=5)
    # Owned descendants must not outlive an exited root worker either.
    try:os.killpg(worker.pid,signal.SIGKILL)
    except ProcessLookupError:pass

def guard(request,stage,run,wall,cpu,disk,starts,run_starts):
    if stage not in request['stages']:raise RuntimeError('Unknown stage '+stage)
    reserve={'wall_seconds':request['shutdown_reserve']['wall_seconds_within_each_dynamic_stage'],'complete_process_cpu_seconds':request['shutdown_reserve']['complete_process_cpu_seconds_within_each_dynamic_stage']}
    checks=[('total',(0,0,0),request['totals'],reserve),(stage,starts[stage],request['stages'][stage],reserve)]
    for name,start,caps,res in checks:
        if wall-start[0]>=caps['wall_seconds']-res['wall_seconds']:raise RuntimeError(name+' wall shutdown reserve')
        if cpu-start[1]>=caps['complete_process_cpu_seconds']-res['complete_process_cpu_seconds']:raise RuntimeError(name+' CPU shutdown reserve')
        if disk-start[2]>=caps['artifact_bytes']:raise RuntimeError(name+' artifact limit')

def cumulative(root,state):
    h=read(root/'HISTORY.json') if (root/'HISTORY.json').exists() else {}
    final=read(root/'run-once/status.json') if (root/'run-once/status.json').exists() else {}
    ledger=final.get('ledger') or {};current=ledger.get('reserved')
    if 'historical_recorded_cumulative' in h:
        prior=h['historical_recorded_cumulative'];counts=prior.get('reserved_consumption',{})
        return {'history':h,'current_reserved':current,'current_call_status_counts':ledger.get('call_status_counts'),'cumulative_reserved_consumption':{k:counts.get(k,0)+current.get(k,0) for k in set(counts)|set(current)} if current is not None else None,'wall_seconds_upper_with_history':prior.get('wall_seconds_upper',0)+state['wall_seconds'],'cpu_seconds_upper_with_history':prior.get('cpu_seconds_upper',0)+state['cpu_seconds'],'historical_pending_and_monitoring_gap_preserved':True,'historical_scope_complete':False,'cumulative_resource_acceptance':False}
    old=h.get('nvme_attempt',{});previous=h.get('previous_attempts',{})
    a=previous.get('cumulative_reserved_consumption',{});b=old.get('reserved',{})
    return {'history':h,'current_reserved':current,'current_call_status_counts':ledger.get('call_status_counts'),'cumulative_reserved_consumption':{k:a.get(k,0)+b.get(k,0)+current.get(k,0) for k in set(a)|set(b)|set(current)} if current is not None else None,'wall_seconds_upper_with_history':previous.get('cumulative_wall_seconds',0)+old.get('wall_seconds_upper',0)+state['wall_seconds'],'cpu_seconds_upper_with_history':previous.get('cumulative_cpu_seconds',0)+old.get('cpu_seconds_upper',0)+state['cpu_seconds'],'historical_pending_and_monitoring_gap_preserved':True,'cumulative_resource_acceptance':False}

def main(root=ROOT,runner='runner.py',status_writer=None,sample_interval=.5):
    root=Path(root)
    require_native_linux_filesystem(root)
    status_writer=status_writer or durable_atomic_json
    if any((root/n).exists() for n in ('run-once','supervisor-status.json','execution.lock')):raise RuntimeError('Previous attempt exists; no retry')
    started=time.monotonic();worker=None;log=None;lock=None;failure=None;peak=0;last_disk=-99;disk=0;ticks={};state={};stage='staging_and_zero_step_gate';run=None;starts={'staging_and_zero_step_gate':(0,0,0)};run_starts={};last_activity=None;limits=None;affinity=[];heartbeat=0
    def interrupted(sig,frame):raise RuntimeError('Supervisor received signal '+str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        if (root/'run-once').exists() or (root/'supervisor-status.json').exists():raise RuntimeError('Previous attempt exists; no retry')
        lock=(root/'execution.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        limits=read(root/'RESOURCE_REQUEST.json');machine_guard(root,True)
        available=sorted(os.sched_getaffinity(0));a=cpu_idle();time.sleep(.25);b=cpu_idle()
        rates={c:(b[c][1]-a[c][1])/max(1,b[c][0]-a[c][0]) for c in available}
        affinity=sorted(available,key=lambda c:(-rates[c],c))[:4]
        if len(affinity)!=4 or any(rates[c]<.5 for c in affinity):raise RuntimeError('Four idle allowed CPU cores not observed')
        os.sched_setaffinity(0,set(affinity));os.nice(10)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','NUMEXPR_NUM_THREADS':'4','PYTHONHASHSEED':'0','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'}
        log=(root/'console.log').open('xb')
        worker=subprocess.Popen([sys.executable,'-B',str(ROOT/'worker_bootstrap.py'),str(os.getpid()),runner],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while worker.poll() is None:
            now=time.monotonic();wall=now-started;tree=proc_tree(os.getpid())
            rss=sum(v['rss'] for v in tree.values());peak=max(peak,sum(v['peak'] for v in tree.values()))
            for pid,v in tree.items():ticks[pid,v['start']]=max(ticks.get((pid,v['start']),0),v['ticks'])
            cpu=sum(ticks.values())/os.sysconf('SC_CLK_TCK')
            p=root/'run-once/activity.json'
            if p.exists():activity=read(p);stage=activity['stage'];run=activity.get('run')
            if now-last_disk>=2:disk=artifacts(root);last_disk=now
            starts.setdefault(stage,(wall,cpu,disk))
            if (stage,run)!=last_activity:
                durable_append_jsonl(root/'resource-history.jsonl',{'stage':stage,'run':run,'wall_seconds_cumulative':wall,'cpu_seconds_cumulative':cpu,'artifact_bytes_cumulative':disk});last_activity=(stage,run)
            machine=machine_guard(root)
            guard(limits,stage,run,wall,cpu,disk,starts,run_starts)
            if peak>=limits['totals']['all_resident_rss_bytes']-128*1024**2:raise RuntimeError('Aggregate RSS shutdown reserve')
            if (root/'STOP_REQUESTED.json').exists():raise RuntimeError('External stop request')
            heartbeat+=1
            state={'status':'running','pid':worker.pid,'supervisor_pid':os.getpid(),'stage':stage,'run':run,'wall_seconds':wall,'cpu_seconds':cpu,'rss_bytes':rss,'peak_rss_bytes':peak,'artifact_bytes':disk,'device':'cpu','affinity':affinity,'heartbeat':heartbeat,'updated_monotonic':now,'stop_reason':None,'automatic_retry':False,**machine}
            status_writer(root/'supervisor-status.json',state);time.sleep(sample_interval)
        worker.wait()
        if worker.returncode:failure='Worker exited '+str(worker.returncode)
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc);traceback.print_exc()
    finally:
        if worker is not None:
            try:stop_owned_group(worker)
            except BaseException as exc:failure=(failure or '')+'; cleanup error '+repr(exc)
        if log is not None:log.close()
        if lock is not None:lock.close()
        # Reaped child process CPU survives exit; monitoring exceptions do not bypass this settlement.
        own=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
        wall=time.monotonic()-started;cpu=own.ru_utime+own.ru_stime+child.ru_utime+child.ru_stime
        peak=max(peak,(own.ru_maxrss+child.ru_maxrss)*1024)
        try:disk=artifacts(root)
        except BaseException as exc:failure=(failure or '')+'; artifact settlement error '+repr(exc)
        state.update(status='complete' if failure is None and worker is not None and worker.returncode==0 else 'stopped',pid=worker.pid if worker else None,supervisor_pid=os.getpid(),stage=stage,run=run,wall_seconds=wall,cpu_seconds=cpu,peak_rss_upper_bound_bytes=peak,artifact_bytes=disk,stop_reason=failure,returncode=worker.returncode if worker else None,automatic_retry=False)
        state['final_resource_pass']=bool(failure is None and limits and wall<=limits['totals']['wall_seconds'] and cpu<=limits['totals']['complete_process_cpu_seconds'] and disk<=limits['totals']['artifact_bytes'] and peak<=limits['totals']['all_resident_rss_bytes'])
        state['updated_monotonic']=time.monotonic();state['heartbeat']=heartbeat+1
        try:
            state['cumulative_cost']=cumulative(root,state);status_writer(root/'supervisor-status.json',state)
        except BaseException as exc:
            traceback.print_exc()
            try:durable_append_jsonl(root/'supervisor-progress.jsonl',{'status':'final_status_write_failed','error':type(exc).__name__+': '+str(exc),'settlement':state})
            except BaseException:traceback.print_exc()
    return 0 if state['status']=='complete' else 1
if __name__=='__main__':sys.exit(main())
