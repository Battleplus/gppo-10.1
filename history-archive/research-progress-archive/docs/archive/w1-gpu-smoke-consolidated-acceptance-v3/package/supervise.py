"""One-shot Linux supervision; race-safe size, fail-closed cleanup, PDEATHSIG."""
import csv,fcntl,json,math,os,resource,signal,stat,subprocess,sys,time,shutil,traceback
from pathlib import Path
from infra_io import durable_append_jsonl,durable_atomic_json,require_native_linux_filesystem
from linux_process_scope import (
    ProcessScopeError,
    enable_subreaper,
    monotonic_cpu_delta,
    proc_tree as _proc_tree,
    reap_owned_children,
    tree_cpu_ticks,
)
from phase_handshake import StageServer
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
    for name in ('console.log','supervisor-status.json','resource-history.jsonl','supervisor-console.log','STOP_REQUESTED.json','worker-initialization-failure.json'):
        try:total+=(root/name).stat().st_size
        except FileNotFoundError:pass
    return total

def proc_tree(root):
    rows=_proc_tree(root)
    for pid,row in rows.items():
        try:
            data=(Path('/proc')/str(pid)/'stat').read_text(encoding='ascii')
            fields=data[data.rfind(')')+2:].split()
            rss=int(fields[21])*os.sysconf('SC_PAGE_SIZE')
            peak=rss
            for line in (Path('/proc')/str(pid)/'status').read_text(encoding='ascii').splitlines():
                if line.startswith('VmHWM:'):peak=int(line.split()[1])*1024
        except (FileNotFoundError,ProcessLookupError):
            rss=peak=0
        row.update(rss=rss,peak=peak)
    return rows

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

def gpu_contract(root,request):
    gpu=request.get('gpu')
    matrix=read(root/'experiment-matrix.json')
    if not isinstance(gpu,dict):
        if str(matrix.get('world_model_device','')).startswith('cuda:'):
            raise RuntimeError('Frozen GPU resource request missing for CUDA matrix')
        return None
    if matrix.get('world_model_device')!=gpu.get('logical_device') or gpu.get('logical_device')!='cuda:0':
        raise RuntimeError('World-model logical GPU device differs from frozen GPU request')
    if type(gpu.get('physical_device')) is not int or gpu['physical_device']<0:raise RuntimeError('Frozen physical GPU index invalid')
    if type(gpu.get('free_memory_minimum_bytes')) is not int or gpu['free_memory_minimum_bytes']<0:raise RuntimeError('Frozen GPU free-memory minimum invalid')
    if type(gpu.get('peak_allocated_memory_bytes')) is not int or gpu['peak_allocated_memory_bytes']<=0:raise RuntimeError('Frozen GPU memory cap invalid')
    if gpu.get('exclusive_allocation_required') is not True:raise RuntimeError('Exclusive GPU allocation is required')
    return gpu

def nvidia_rows(query):
    result=subprocess.run(['nvidia-smi',query,'--format=csv,noheader,nounits'],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False,timeout=10)
    if result.returncode:raise RuntimeError('nvidia-smi query failed: '+result.stderr.strip())
    return [row for row in csv.reader(result.stdout.splitlines(),skipinitialspace=True) if row]

def gpu_snapshot(gpu,pids,require_free_memory=False):
    rows=nvidia_rows('--query-gpu=index,uuid,utilization.gpu,memory.total,memory.free')
    matches=[row for row in rows if len(row)>=5 and row[0].strip().isdigit() and int(row[0].strip())==gpu['physical_device']]
    if len(matches)!=1:raise RuntimeError('Requested physical GPU is unavailable or ambiguous')
    row=matches[0]
    try:
        gpu_uuid=row[1].strip();util=int(row[2].strip());total=int(row[3].strip())*1024**2;free=int(row[4].strip())*1024**2
    except (ValueError,IndexError) as exc:raise RuntimeError('nvidia-smi GPU inventory values invalid') from exc
    if require_free_memory and free<gpu['free_memory_minimum_bytes']:raise RuntimeError('GPU free-memory minimum not met')
    process_rows=nvidia_rows('--query-compute-apps=gpu_uuid,pid,used_memory')
    selected=[];unowned=[]
    for process in process_rows:
        if len(process)<3 or process[0].strip()!=gpu_uuid:continue
        try:pid=int(process[1].strip());used=int(process[2].strip())*1024**2
        except ValueError as exc:raise RuntimeError('nvidia-smi compute-process values invalid') from exc
        selected.append((pid,used))
        if pid not in pids:unowned.append(pid)
    if unowned:raise RuntimeError('Exclusive GPU allocation violated by unowned process')
    owned_bytes=sum(used for _pid,used in selected)
    if owned_bytes>gpu['peak_allocated_memory_bytes']:raise RuntimeError('Owned worker GPU memory cap exceeded')
    return {
        'gpu_physical_device':gpu['physical_device'],
        'gpu_uuid':gpu_uuid,
        'gpu_logical_device':gpu['logical_device'],
        'gpu_utilization_percent':util,
        'gpu_total_memory_bytes':total,
        'gpu_free_memory_bytes':free,
        'gpu_owned_compute_process_pids':sorted(pid for pid,_used in selected),
        'gpu_owned_process_memory_bytes':owned_bytes,
        'gpu_owned_process_memory_current_bytes':owned_bytes,
        'gpu_compute_process_count':len(selected),
        'gpu_exclusive_allocation_observed':True,
    }

def stop_owned_worker(worker,timeout_seconds=30.0):
    if (isinstance(timeout_seconds,bool) or not isinstance(timeout_seconds,(int,float))
            or not 0<=timeout_seconds< float('inf')):raise ValueError('OWNED_WORKER_TIMEOUT_INVALID')
    if worker.poll() is not None:
        worker.wait()
        return
    started=time.monotonic();deadline=started+float(timeout_seconds)
    try:worker.terminate()
    except ProcessLookupError:pass
    term_deadline=started+float(timeout_seconds)*0.75
    try:worker.wait(timeout=max(0.0,min(term_deadline,deadline)-time.monotonic()))
    except subprocess.TimeoutExpired:
        try:worker.kill()
        except ProcessLookupError:pass
        try:worker.wait(timeout=max(0.0,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            if worker.poll() is not None:
                worker.wait()
                return
            raise RuntimeError('OWNED_WORKER_REAP_TIMEOUT')

def guard(request,stage,run,wall,cpu,disk,starts,run_starts):
    if stage not in request['stages']:raise RuntimeError('Unknown stage '+stage)
    reserve={'wall_seconds':0,'complete_process_cpu_seconds':0}
    total_reserve=request.get('accounting_reserves',{}).get('cross_system_wall_seconds',0)
    if (isinstance(total_reserve,bool) or not isinstance(total_reserve,(int,float))
            or not math.isfinite(total_reserve) or total_reserve<0):
        raise RuntimeError('Cross-system wall shutdown reserve invalid')
    checks=[('total',(0,0,0),request['totals'],{'wall_seconds':float(total_reserve),'complete_process_cpu_seconds':0}),
            (stage,starts[stage],request['stages'][stage],reserve)]
    for name,start,caps,res in checks:
        if wall-start[0]>=caps['wall_seconds']-res['wall_seconds']:raise RuntimeError(name+' wall shutdown reserve')
        if cpu-start[1]>=caps['complete_process_cpu_seconds']-res['complete_process_cpu_seconds']:raise RuntimeError(name+' CPU shutdown reserve')
        disk_cap=caps.get('artifact_bytes',caps.get('active_storage_bytes'))
        if disk_cap is None:raise RuntimeError(name+' storage budget missing')
        if disk-start[2]>=disk_cap:raise RuntimeError(name+' artifact limit')

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

def _apply_final_resource_status(state):
    """Prevent a resource-over-limit settlement from retaining success."""
    if not state.get('final_resource_pass') and state.get('status')=='complete':
        state['status']='stopped'
        state['stop_reason']=state.get('stop_reason') or 'FINAL_RESOURCE_LIMIT_EXCEEDED'
    return state

def main(root=ROOT,runner='runner.py',status_writer=None,sample_interval=.5,worker_input=None):
    root=Path(root)
    require_native_linux_filesystem(root)
    status_writer=status_writer or durable_atomic_json
    if any((root/n).exists() for n in ('run-once','supervisor-status.json','execution.lock')):raise RuntimeError('Previous attempt exists; no retry')
    durable_append_jsonl(root/'supervisor-console.log',{'event':'supervisor_starting','pid':os.getpid(),'root':str(root.resolve()),'runner':str(runner),'updated_monotonic':time.monotonic()})
    worker_payload=None
    if worker_input is not None:
        if isinstance(worker_input,str):worker_payload=worker_input.encode('utf-8')
        elif isinstance(worker_input,bytes):worker_payload=worker_input
        else:raise TypeError('WORKER_INPUT_MUST_BE_BYTES_OR_TEXT')
        if len(worker_payload)>4096:raise ValueError('WORKER_INPUT_TOO_LARGE')
    started=time.monotonic();worker=None;log=None;lock=None;phase_server=None;subreaper_enabled=False;failure=None;peak=0;gpu_peak=0;gpu_state={};gpu=None;last_disk=-99;disk=0;state={};stage='staging_and_zero_step_gate';run=None;starts={'staging_and_zero_step_gate':(0,0,0)};run_starts={};last_activity=None;limits=None;affinity=[];heartbeat=0;phase_rows=[];scope_origin_ticks=0;last_scope_ticks=0;latest_boundary_cpu=0.0;clock_ticks=os.sysconf('SC_CLK_TCK');child_cpu_start=None
    def interrupted(sig,frame):raise RuntimeError('Supervisor received signal '+str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        baseline_tree=_proc_tree(os.getpid())
        scope_origin_ticks=tree_cpu_ticks(baseline_tree)
        last_scope_ticks=scope_origin_ticks
        scope_origin_cpu=scope_origin_ticks/clock_ticks
        child_cpu_start=resource.getrusage(resource.RUSAGE_CHILDREN)
        if (root/'run-once').exists() or (root/'supervisor-status.json').exists():raise RuntimeError('Previous attempt exists; no retry')
        lock=(root/'execution.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        enable_subreaper();subreaper_enabled=True
        limits=read(root/'RESOURCE_REQUEST.json')
        bootstrap=ROOT/'worker_bootstrap.py'
        if not bootstrap.is_file():raise RuntimeError('WORKER_BOOTSTRAP_MISSING:'+str(bootstrap))
        phase_server=StageServer(root/'.phase-accounting.sock',limits,tree_reader=_proc_tree,
                                 cpu_origin_seconds=scope_origin_cpu,wall_origin_monotonic=started)
        gpu=gpu_contract(root,limits);machine_guard(root,True)
        if gpu is not None:
            gpu_state=gpu_snapshot(gpu,{os.getpid()},require_free_memory=True);gpu_peak=gpu_state['gpu_owned_process_memory_current_bytes']
        available=sorted(os.sched_getaffinity(0));a=cpu_idle();time.sleep(.25);b=cpu_idle()
        rates={c:(b[c][1]-a[c][1])/max(1,b[c][0]-a[c][0]) for c in available}
        affinity=sorted(available,key=lambda c:(-rates[c],c))[:4]
        if len(affinity)!=4 or any(rates[c]<.5 for c in affinity):raise RuntimeError('Four idle allowed CPU cores not observed')
        os.sched_setaffinity(0,set(affinity));os.nice(10)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu['physical_device']) if gpu else '','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','NUMEXPR_NUM_THREADS':'4','PYTHONHASHSEED':'0','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'}
        if gpu is not None:env.update(W1_WORLD_MODEL_DEVICE=gpu['logical_device'],W1_ALLOCATED_GPU_PHYSICAL_DEVICE=str(gpu['physical_device']))
        env['W1_PHASE_ACCOUNTING_SOCKET']=str(phase_server.path)
        log=(root/'console.log').open('xb')
        worker=subprocess.Popen([sys.executable,'-B',str(bootstrap),str(os.getpid()),runner],cwd=root,env=env,stdin=subprocess.PIPE if worker_payload is not None else subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        if worker_payload is not None:
            try:
                worker.stdin.write(worker_payload);worker.stdin.flush();worker.stdin.close()
            except BaseException as exc:raise RuntimeError('WORKER_INPUT_DELIVERY_FAILED') from exc
        while worker.poll() is None:
            pending=phase_server.process_pending(worker.pid)
            if pending:
                for row in pending:
                    phase_rows.append(row)
                    disk=artifacts(root);last_disk=time.monotonic()
                    stage=phase_server.stage
                    starts[stage]=(max(0.0,row['sampled_monotonic']-started),row['cpu_seconds_cumulative'],disk)
                    latest_boundary_cpu=max(latest_boundary_cpu,row['cpu_seconds_cumulative'])
                    durable_append_jsonl(root/'resource-history.jsonl',{'event':'phase_boundary',**row,'artifact_bytes_cumulative':disk})
            now=time.monotonic();wall=now-started;tree=proc_tree(os.getpid())
            rss=sum(v['rss'] for v in tree.values());peak=max(peak,sum(v['peak'] for v in tree.values()))
            scope_ticks=tree_cpu_ticks(tree)
            monotonic_cpu_delta(last_scope_ticks,scope_ticks,clock_ticks)
            last_scope_ticks=scope_ticks
            observed_tree_cpu=monotonic_cpu_delta(scope_origin_ticks,scope_ticks,clock_ticks)
            cpu=max(observed_tree_cpu,latest_boundary_cpu)
            reaped_usage=resource.getrusage(resource.RUSAGE_CHILDREN)
            reaped_cpu=max(0.0,(reaped_usage.ru_utime+reaped_usage.ru_stime)-(child_cpu_start.ru_utime+child_cpu_start.ru_stime))
            p=root/'run-once/activity.json'
            if p.exists():activity=read(p);run=activity.get('run')
            if now-last_disk>=2:disk=artifacts(root);last_disk=now
            if stage not in starts:raise RuntimeError('CPU_SCOPE_PHASE_START_MISSING:'+stage)
            if (stage,run)!=last_activity:
                durable_append_jsonl(root/'resource-history.jsonl',{'stage':stage,'run':run,'wall_seconds_cumulative':wall,'cpu_seconds_cumulative':cpu,'artifact_bytes_cumulative':disk});last_activity=(stage,run)
            machine=machine_guard(root)
            if gpu is not None:
                gpu_state=gpu_snapshot(gpu,set(tree))
                gpu_peak=max(gpu_peak,gpu_state['gpu_owned_process_memory_current_bytes'])
            guard(limits,stage,run,wall,cpu,disk,starts,run_starts)
            if peak>=limits['totals']['all_resident_rss_bytes']-128*1024**2:raise RuntimeError('Aggregate RSS shutdown reserve')
            if (root/'STOP_REQUESTED.json').exists():raise RuntimeError('External stop request')
            heartbeat+=1
            state={'status':'running','pid':worker.pid,'supervisor_pid':os.getpid(),'stage':stage,'run':run,'wall_seconds':wall,'cpu_seconds':cpu,'rss_bytes':rss,'peak_rss_bytes':peak,'artifact_bytes':disk,'device':gpu['logical_device'] if gpu else 'cpu','affinity':affinity,'heartbeat':heartbeat,'updated_monotonic':now,'stop_reason':None,'automatic_retry':False,'gpu_owned_process_memory_peak_bytes':gpu_peak,'cpu_observed_process_tree_seconds':observed_tree_cpu,'cpu_reaped_children_seconds':max(0.0,reaped_cpu),'cpu_sampled_reaped_overlap_seconds':max(0.0,reaped_cpu),'cpu_scope':'stable owner-verified /proc process-tree sum of utime+stime+cutime+cstime; subreaper catches orphans; authenticated stage boundaries; no delegated cgroup, so caps are polled and can only be rejected at the next sample/final settlement','cpu_estimate_source':'stable_proc_tree_ticks_including_kernel_waited_descendants','cpu_scope_verified':True,'cpu_scope_complete':False,'cpu_scope_hard_enforcement':False,'cpu_scope_limitations':['/proc is not an atomic tree snapshot; unstable scans and nonmonotonic totals fail closed','without a delegated cgroup, CPU cap enforcement is sampled rather than kernel-hard; final RUSAGE settlement rejects overrun'],'phase_rows':phase_rows,**machine,**gpu_state}
            status_writer(root/'supervisor-status.json',state);time.sleep(sample_interval)
        worker.wait()
        if worker.returncode:
            failure='Worker exited '+str(worker.returncode)
            durable_append_jsonl(root/'supervisor-console.log',{'event':'worker_exit','pid':worker.pid,'returncode':worker.returncode,'error':failure,'updated_monotonic':time.monotonic()})
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc)
        trace=traceback.format_exc()
        try:durable_append_jsonl(root/'supervisor-console.log',{'event':'supervisor_exception','error':failure,'traceback':trace,'updated_monotonic':time.monotonic()})
        except BaseException:traceback.print_exc()
        traceback.print_exc()
    finally:
        cleanup_budget=30.0
        if limits is not None:
            reserve=limits.get('accounting_reserves',{}).get('cross_system_wall_seconds',30.0)
            if isinstance(reserve,(int,float)) and not isinstance(reserve,bool) and math.isfinite(reserve) and reserve>=0:
                cleanup_budget=min(cleanup_budget,float(reserve))
            remaining=limits.get('totals',{}).get('wall_seconds')
            if isinstance(remaining,(int,float)) and not isinstance(remaining,bool) and math.isfinite(remaining):
                cleanup_budget=min(cleanup_budget,max(0.0,float(remaining)-(time.monotonic()-started)))
        cleanup_deadline=time.monotonic()+cleanup_budget
        if worker is not None:
            worker_cleanup_budget=max(0.0,cleanup_budget*0.5)
            try:stop_owned_worker(worker,timeout_seconds=max(0.0,min(worker_cleanup_budget,cleanup_deadline-time.monotonic())))
            except BaseException as exc:failure=(failure or '')+'; cleanup error '+repr(exc)
        if subreaper_enabled:
            try:reap_owned_children(timeout_seconds=max(0.0,cleanup_deadline-time.monotonic()))
            except BaseException as exc:failure=(failure or '')+'; owned descendant cleanup error '+repr(exc)
        if phase_server is not None:
            try:phase_server.close()
            except BaseException as exc:failure=(failure or '')+'; phase server cleanup error '+repr(exc)
        if log is not None:log.close()
        if lock is not None:lock.close()
        if gpu is not None:
            try:
                gpu_state=gpu_snapshot(gpu,{os.getpid()})
                gpu_peak=max(gpu_peak,gpu_state['gpu_owned_process_memory_current_bytes'])
            except BaseException as exc:
                failure=(failure or '')+'; GPU settlement sample error '+type(exc).__name__+': '+str(exc)
        # Reaped child process CPU survives exit; monitoring exceptions do not bypass this settlement.
        own=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
        supervisor_cpu_sampled=time.monotonic()
        own_cpu=own.ru_utime+own.ru_stime
        child_cpu=child.ru_utime+child.ru_stime
        wall=supervisor_cpu_sampled-started;cpu=own_cpu+child_cpu
        peak=max(peak,(own.ru_maxrss+child.ru_maxrss)*1024)
        try:durable_append_jsonl(root/'supervisor-console.log',{'event':'supervisor_settlement','status':'complete' if failure is None and worker is not None and worker.returncode==0 else 'stopped','error':failure,'worker_returncode':worker.returncode if worker else None,'updated_monotonic':supervisor_cpu_sampled})
        except BaseException as exc:failure=(failure or '')+'; supervisor log settlement error '+repr(exc)
        try:disk=artifacts(root)
        except BaseException as exc:failure=(failure or '')+'; artifact settlement error '+repr(exc)
        state.update(status='complete' if failure is None and worker is not None and worker.returncode==0 else 'stopped',pid=worker.pid if worker else None,supervisor_pid=os.getpid(),stage=stage,run=run,wall_seconds=wall,cpu_seconds=cpu,cpu_process_self_seconds_cumulative=own_cpu,cpu_reaped_children_seconds_cumulative=child_cpu,cpu_sampling_started_monotonic=started,cpu_sampled_monotonic=supervisor_cpu_sampled,cpu_scope='final RUSAGE_SELF plus waited RUSAGE_CHILDREN after owned descendant cleanup; live stage accounting uses stable owner-verified /proc self+waited ticks and synchronous authenticated boundaries',cpu_scope_complete=False,cpu_lifetime_settlement_complete=bool(subreaper_enabled and failure is None),cpu_scope_hard_enforcement=False,cpu_scope_limitations=['/proc is not an atomic tree snapshot; unstable scans and nonmonotonic totals fail closed','without a delegated cgroup, CPU cap enforcement is sampled rather than kernel-hard; final RUSAGE settlement rejects overrun'],phase_rows=phase_rows,peak_rss_upper_bound_bytes=peak,artifact_bytes=disk,stop_reason=failure,returncode=worker.returncode if worker else None,automatic_retry=False)
        if gpu is not None:
            state.update(device=gpu['logical_device'],gpu_owned_process_memory_peak_bytes=gpu_peak,**gpu_state)
        state['final_resource_pass']=bool(failure is None and limits and wall<=limits['totals']['wall_seconds'] and cpu<=limits['totals']['complete_process_cpu_seconds'] and disk<=limits['totals']['active_storage_bytes'] and peak<=limits['totals']['all_resident_rss_bytes'])
        _apply_final_resource_status(state)
        state['updated_monotonic']=time.monotonic();state['heartbeat']=heartbeat+1
        try:
            state['cumulative_cost']=cumulative(root,state);status_writer(root/'supervisor-status.json',state)
        except BaseException as exc:
            traceback.print_exc()
            try:durable_append_jsonl(root/'supervisor-progress.jsonl',{'status':'final_status_write_failed','error':type(exc).__name__+': '+str(exc),'settlement':state})
            except BaseException:traceback.print_exc()
    return 0 if state['status']=='complete' else 1
if __name__=='__main__':sys.exit(main())
