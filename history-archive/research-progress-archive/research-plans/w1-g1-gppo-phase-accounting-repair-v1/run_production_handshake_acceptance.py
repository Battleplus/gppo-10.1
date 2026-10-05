"""Actual production worker/runner/supervisor, with no-model pretraining trace stop.

All wrappers and fixture identity changes are outside the final source package.
No environment method, training body or checkpoint operation is executed.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import subprocess
import sys
import time

SOURCE = Path(sys.argv[1]).resolve()
EVIDENCE = Path(sys.argv[2]).resolve()
PYTHON = sys.executable
CACHE = str(Path(sys.prefix) / '.w1-no-bytecode-cache')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in root.rglob('*') if p.is_file()}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')


WORKER = r'''
import hashlib,json,os,pathlib,resource,socket,subprocess,sys,threading,time
cache=str(pathlib.Path(sys.prefix)/'.w1-no-bytecode-cache')
if not sys.flags.isolated or not sys.flags.dont_write_bytecode or sys.pycache_prefix!=cache:
    os.execv(sys.executable,[sys.executable,'-I','-B','-X','pycache_prefix='+cache,str(pathlib.Path(__file__).resolve())])
ROOT=pathlib.Path(__file__).parent/'package'
sys.path.insert(0,str(ROOT))
import local_research_worker
from infra_io import durable_atomic_json,durable_append_jsonl
MODE=os.environ['W1_ENGINEERING_MODE']
OUT=ROOT/'run-once'
TARGET=os.environ['W1_PHASE_ACCOUNTING_SOCKET']
SEEN={'pretrain':False,'model_body_entered':False,'initial_stage_snapshots':0,'duplicate_replayed':False}

def burn(seconds):
    started=time.process_time()
    while time.process_time()-started<seconds: pass

def checkpoint(event,**values):
    durable_append_jsonl(OUT/'engineering-events.jsonl',{'event':event,**values})

def trace(frame,event,arg):
    filename=pathlib.Path(frame.f_code.co_filename)
    name=frame.f_code.co_name
    if event=='call' and name=='_send_json' and filename==ROOT/'phase_handshake.py' and MODE=='exit_before_confirmation':
        message=frame.f_locals['value']
        if message.get('action')=='worker_confirm' and message.get('sequence')==2:
            checkpoint('exit_before_worker_confirmation',sequence=2)
            os._exit(18)
    if event=='return' and name=='request_boundary' and filename==ROOT/'phase_handshake.py' and MODE=='duplicate':
        row=frame.f_locals['row']
        if row.get('next_stage')=='conditional_policy_training' and not SEEN['duplicate_replayed'] and isinstance(arg,dict):
            SEEN['duplicate_replayed']=True
            from phase_handshake import request_boundary
            replay=request_boundary(row,protocol_path=OUT/'worker-phase-protocol.jsonl')
            checkpoint('duplicate_replayed',idempotent=replay['idempotent_replay'])
    if event=='call' and name=='_activity' and filename==ROOT/'runner.py' and frame.f_locals.get('stage')=='conditional_policy_training':
        caller=frame.f_back
        ledger=caller.f_locals['ledger']
        for index in range(3):
            burn(.01)
            ledger.snapshot_phase(reason='engineering_staging_snapshot_'+str(index))
        SEEN['initial_stage_snapshots']=3
        checkpoint('initial_staging_snapshots_complete',stage=ledger.stage,phase_cpu_origin=ledger.phase_started_cpu)
    if event=='call' and name=='request_boundary' and filename==ROOT/'phase_handshake.py' and MODE in ('first_rejection','entry_rejection'):
        row=frame.f_locals['row']
        if (MODE=='first_rejection' and row.get('next_stage')=='conditional_policy_training') or (MODE=='entry_rejection' and row.get('event')=='stage_enter'):
            row['cpu_start_seconds']+=.001
            if row.get('event')=='stage_enter':
                row['stage_elapsed_cpu_seconds']=row['cpu_end_seconds']-row['cpu_start_seconds']
            else:
                row['cpu_seconds']=row['cpu_end_seconds']-row['cpu_start_seconds']
            checkpoint('controlled_invalid_boundary_injected')
    if event=='call' and name=='train_policy_routes' and filename==ROOT/'production_policy.py':
        # Stop before any line of production training: the real first phase transition has completed.
        ledger=frame.f_locals['ledger']
        SEEN['pretrain']=True
        checkpoint('first_training_boundary_reached',confirmed_stage=ledger.stage)
        checkpoint('actual_loaded_production_modules',paths={key:str(pathlib.Path(value.__file__).resolve()) for key,value in sys.modules.items() if getattr(value,'__file__',None) and pathlib.Path(value.__file__).resolve().is_relative_to(ROOT.resolve())})
        if MODE=='exit_after_confirmation':
            durable_atomic_json(OUT/'engineering-replay-reached.json',SEEN)
            burn(.04)
            os._exit(17)
        if MODE=='normal':
            for index in range(3):
                ledger.call('controlled_parent_cpu',{},burn,.012)
                ledger.snapshot_phase(reason='engineering_training_snapshot_'+str(index))
            for index in range(3):
                started=resource.getrusage(resource.RUSAGE_CHILDREN)
                subprocess.run([sys.executable,'-I','-B','-c','import time; end=time.process_time()+.015\nwhile time.process_time()<end: pass'],check=True)
                ended=resource.getrusage(resource.RUSAGE_CHILDREN)
                checkpoint('short_child_waited',child_cpu=ended.ru_utime+ended.ru_stime-started.ru_utime-started.ru_stime)
            ledger.select('conditional_task_confirmation')
            ledger.call('controlled_task_cpu',{},burn,.02)
            ledger.snapshot_phase(reason='engineering_task_snapshot')
        durable_atomic_json(OUT/'engineering-replay-reached.json',SEEN)
        raise RuntimeError('ENGINEERING_NO_MODEL_STOP_AFTER_CONFIRMED_TRAINING_BOUNDARY')
    return trace

sys.settrace(trace)
code=local_research_worker.main()
sys.settrace(None)
if OUT.exists():durable_atomic_json(OUT/'engineering-wrapper-end.json',{'code':code,**SEEN})
raise SystemExit(code)
'''


SUPERVISOR = r'''
import hashlib,json,os,pathlib,resource,sys,time
ROOT=pathlib.Path(__file__).parent/'package'
sys.path.insert(0,str(ROOT))
os.sched_setaffinity(0,{0})
begin=resource.getrusage(resource.RUSAGE_SELF)
import supervise
import phase_handshake
original_send=phase_handshake.StageServer._send_response
mode=os.environ['W1_ENGINEERING_MODE']
def fault_send(self,channel,value):
    if value.get('sequence')==2:
        if mode=='drop_final_ack' and value.get('status')=='committed':
            self._record('engineering_drop_final_ack',sequence=2,server_commit_durable=True)
            return
        if mode=='delayed_ack_timeout' and value.get('status')=='accepted':
            self._record('engineering_delayed_accept',sequence=2,delay_seconds=6)
            time.sleep(6)
    return original_send(self,channel,value)
phase_handshake.StageServer._send_response=fault_send
token=sys.stdin.readline().rstrip('\r\n')
code=supervise.main(ROOT,str(ROOT.parent/'engineering_worker.py'),sample_interval=.02,worker_input=token+'\n')
token=None
# Known supervisor post-settlement exit-tail work, captured by the outer waiting parent.
start=time.process_time()
while time.process_time()-start<.03:pass
own=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
value={'code':code,'self_cpu_near_exit':own.ru_utime+own.ru_stime,
 'waited_children_cpu_near_exit':child.ru_utime+child.ru_stime,
 'self_plus_waited_near_exit':own.ru_utime+own.ru_stime+child.ru_utime+child.ru_stime,
 'commanded_supervisor_exit_tail_cpu_seconds':.03}
(ROOT.parent/'supervisor-near-exit.json').write_text(json.dumps(value)+'\n')
raise SystemExit(code)
'''


def main():
    if EVIDENCE.exists():
        raise RuntimeError('ENGINEERING_EVIDENCE_PATH_ALREADY_EXISTS')
    EVIDENCE.mkdir(parents=True)
    source_before = inventory(SOURCE)
    sys.path.insert(0, str(SOURCE))
    from manifest_contract import write_identity_files
    from infra_io import controlled_export
    cases = []
    modes = ['normal', 'duplicate', 'entry_rejection', 'first_rejection', 'drop_final_ack', 'delayed_ack_timeout', 'exit_before_confirmation', 'exit_after_confirmation']
    for mode in modes:
        case = EVIDENCE / mode
        case.mkdir()
        package = case / 'package'
        shutil.copytree(SOURCE, package)
        request_path = package / 'RESOURCE_REQUEST.json'
        request = json.loads(request_path.read_text())
        # Only fixture paths/identity change. Keep all scientific matrix and budget values intact.
        attempt = 'engineering-phase-' + mode
        request['attempt'] = attempt
        write(request_path, request)
        for name in ['experiment-matrix.json', 'parent-split.json']:
            value = json.loads((package / name).read_text())
            value['attempt'] = attempt
            write(package / name, value)
        contract = json.loads((package / 'launch-contract.json').read_text())
        contract['attempt'] = attempt
        contract['native_execution_root'] = str(package)
        contract['resource_request_sha256'] = sha(request_path)
        write(package / 'launch-contract.json', contract)
        identity = write_identity_files(package, attempt=attempt, entrypoint='run_g1_task_validation.py')
        token = 'engineering-fixture-only-' + mode
        authorization = dict(schema='w1-external-launch-authorization/2.0.0',attempt=attempt,
                             **identity,resource_request_sha256=sha(request_path),status='APPROVED',
                             token_sha256=hashlib.sha256(token.encode()).hexdigest())
        write(case / 'fixture-authorization.json', authorization)
        (case / 'engineering_worker.py').write_text(WORKER, encoding='utf-8')
        (case / 'engineering_supervisor.py').write_text(SUPERVISOR, encoding='utf-8')
        env = {**os.environ,'W1_ENGINEERING_MODE':mode,'W1_VERIFIED_ATTEMPT':attempt,
               'W1_VERIFIED_MANIFEST_SHA256':identity['execution_manifest_sha256'],
               'W1_VERIFIED_HASHES_SHA256':identity['hashes_sha256'],
               'W1_EXTERNAL_AUTHORIZATION_FILE':str(case/'fixture-authorization.json'),
               'W1_EXTERNAL_AUTHORIZATION_TOKEN_SHA256':authorization['token_sha256'],
               'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        for name in ['PYTHONPATH','PYTHONHOME','W1_LOCAL_STAGING_WALL_SECONDS','W1_LOCAL_STAGING_CPU_SECONDS']:
            env.pop(name,None)
        argv = [PYTHON,'-I','-B','-X','pycache_prefix='+CACHE,str(case/'engineering_supervisor.py')]
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        start = time.monotonic()
        result = subprocess.run(argv, input=(token+'\n').encode(), capture_output=True,env=env,cwd=case,timeout=45)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        outer_cpu = after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
        (case/'stdout.raw').write_bytes(result.stdout)
        (case/'stderr.raw').write_bytes(result.stderr)
        (case/'stdout.txt').write_text(result.stdout.decode('utf-8','replace'),encoding='utf-8')
        (case/'stderr.txt').write_text(result.stderr.decode('utf-8','replace'),encoding='utf-8')
        state = json.loads((package/'supervisor-status.json').read_text())
        out = package/'run-once'
        phase_path=out/'worker-phase-accounting.jsonl'
        phases = [json.loads(line) for line in phase_path.read_text().splitlines()] if phase_path.exists() else []
        db = sqlite3.connect((out/'budget.sqlite3').as_uri()+'?mode=ro',uri=True)
        ledger_calls = db.execute('SELECT name,amounts,status FROM calls').fetchall();db.close()
        accepted = state.get('phase_rows',[])
        closed = [row for row in accepted if row.get('event') in (None,'transition') and row.get('transition_committed',True)]
        closed_cpu = sum(row['cpu_seconds'] for row in closed)
        tail_rows=[row for row in accepted if row.get('event')=='unconfirmed_stage_tail']
        measured_tail=tail_rows[-1].get('cpu_seconds') if tail_rows else None
        worker_snapshot = next((row for row in reversed(phases) if row.get('scope')=='self_plus_waited_descendants'),None)
        near_exit = json.loads((case/'supervisor-near-exit.json').read_text())
        frozen_inputs = json.loads((package/'execution-manifest.json').read_text())['files']
        source_equal = all(sha(package/n)==h for n,h in source_before.items() if n.endswith('.py'))
        changed_inputs = [n for n,h in frozen_inputs.items() if sha(package/n)!=h]
        exported = controlled_export(out,case/'verified-export')
        checks = {
            'actual_production_worker_runner_and_supervisor': True,
            'first_stage_sequence_passes': mode!='normal' or any(row.get('next_stage')=='conditional_policy_training' for row in closed),
            'frozen_research_inputs_unchanged_during_test': not changed_inputs,
            'production_python_sources_match_final_package':source_equal,
            'worker_research_calls_zero':all(not json.loads(amounts) for _,amounts,_ in ledger_calls),
            'outer_waited_cpu_includes_supervisor_exit_tail':outer_cpu>=state['cpu_seconds']+.025,
            'closed_phase_cpu_not_greater_than_final':closed_cpu<=state['cpu_seconds']+1e-6,
            'independent_phase_sum_and_measured_tail_equal_final': measured_tail is not None and abs(closed_cpu+measured_tail-state['cpu_seconds'])<=1e-6,
            'export_verified':exported['verified'],
            'no_research_attempt_consumed':not (case/'authorization-consumed.json').exists(),
        }
        if mode=='normal':
            checks['training_then_task_transition_committed']=any(row.get('next_stage')=='conditional_task_confirmation' for row in closed)
            checks['worker_failure_settlement_saved']=(out/'resource-settlement.json').exists()
            checks['first_traceback_saved']=(out/'first-error.json').exists()
            engineering_events=[json.loads(line) for line in (out/'engineering-events.jsonl').read_text().splitlines()]
            child_rows=[row for row in engineering_events if row['event']=='short_child_waited']
            child_cpu=sum(row['child_cpu'] for row in child_rows)
            training_close=next(row for row in closed if row.get('stage')=='conditional_policy_training')
            checks['short_children_waited_and_included_once']=len(child_rows)==3 and child_cpu>=.045 and training_close['worker_reported_cpu_seconds']>=child_cpu+.036
            snapshots=[row for row in phases if row.get('event')=='snapshot' and row.get('reason','').startswith('engineering_')]
            for stage_name,expected_count in [('staging_and_zero_step_gate',3),('conditional_policy_training',3),('conditional_task_confirmation',1)]:
                rows=[row for row in snapshots if row['stage']==stage_name]
                checks['snapshot_origin_stable_'+stage_name]=len(rows)==expected_count and len({row['cpu_start_seconds'] for row in rows})==1 and all(rows[i]['cpu_end_seconds']<=rows[i+1]['cpu_end_seconds'] for i in range(len(rows)-1))
        elif mode in ('first_rejection','entry_rejection'):
            checks['baseline_check_not_disabled']='PHASE_CPU_BASELINE_DISCONTINUITY' in (package/'console.log').read_text()
            checks['worker_failure_settlement_saved']=(out/'resource-settlement.json').exists()
            checks['first_traceback_saved']=(out/'first-error.json').exists()
            if mode=='entry_rejection':
                checks['unconfirmed_stage_remains_unknown']=closed_cpu==0 and tail_rows[-1]['stage'] is None and tail_rows[-1]['stage_cpu_seconds'] is None
        elif mode=='exit_after_confirmation':
            checks['worker_exit_after_confirmed_stage']=state['returncode']==17
            checks['outer_incomplete_settlement_saved']=(package/'outer-incomplete-settlement.json').exists()
        elif mode=='exit_before_confirmation':
            checks['worker_exit_before_confirmed_stage']=state['returncode']==18 and not any(row.get('next_stage')=='conditional_policy_training' for row in closed)
            checks['outer_incomplete_settlement_saved']=(package/'outer-incomplete-settlement.json').exists()
        elif mode=='duplicate':
            journal=[json.loads(line) for line in (package/'supervisor-phase-protocol.jsonl').read_text().splitlines()]
            checks['duplicate_is_idempotent']=any(row.get('event')=='duplicate_commit' for row in journal) and sum(row.get('next_stage')=='conditional_policy_training' for row in closed)==1
            checks['worker_failure_settlement_saved']=(out/'resource-settlement.json').exists()
        else:
            checks['ambiguous_confirmation_stops_before_model']=not (out/'engineering-replay-reached.json').exists()
            checks['first_traceback_saved']=(out/'first-error.json').exists()
            checks['worker_failure_settlement_saved']=(out/'resource-settlement.json').exists()
            if mode=='drop_final_ack':
                checks['lost_ack_stage_attribution_is_explicitly_unknown']=state.get('server_committed_stage')=='conditional_policy_training' and state.get('worker_last_acknowledged_stage')=='staging_and_zero_step_gate' and tail_rows[-1]['stage'] is None and tail_rows[-1]['stage_cpu_seconds'] is None
        value={'mode':mode,'argv':argv,'exit_code':result.returncode,'wall_seconds':time.monotonic()-start,
               'outer_kernel_waited_cpu_seconds':outer_cpu,'supervisor_nested_cpu_seconds':state['cpu_seconds'],
               'supervisor_near_exit':near_exit,'closed_phase_cpu_seconds':closed_cpu,
               'unclosed_tail_to_supervisor_cpu_seconds':state['cpu_seconds']-closed_cpu,
               'independently_recorded_unconfirmed_tail_cpu_seconds':measured_tail,
               'outer_startup_and_exit_tail_cpu_seconds':outer_cpu-state['cpu_seconds'],
               'partition_arithmetic_identity_only_not_measurement_proof':abs(closed_cpu+(state['cpu_seconds']-closed_cpu)+(outer_cpu-state['cpu_seconds'])-outer_cpu)<1e-9,
               'stage_rows':accepted,'worker_phase_snapshots':phases,'ledger_calls':ledger_calls,
               'checks':checks,'all_checks_pass':all(checks.values()),'supervisor':state,
               'module_source_sha256':{n:h for n,h in source_before.items() if n.endswith('.py')},
               'environment_calls':0,'model_initializations':0,'model_forwards':0,'optimizer_updates':0,'checkpoint_operations':0,
               'wrapper_scope':'pre-training call trace stop and controlled CPU work; no stage function replaced; fixture authorization only'}
        write(case/'case-evidence.json',value)
        cases.append(value)
    summary={'schema':'w1-production-phase-cpu-acceptance/1.0.0','cases':cases,
             'all_cases_pass':all(c['all_checks_pass'] for c in cases),
             'source_before':source_before,'source_after':inventory(SOURCE),
             'source_unchanged':source_before==inventory(SOURCE),
             'formal_attempt_created':False,'research_model_calls':0,'environment_calls':0,
             'third_party_torch_imported_in_workers':True,'cuda_initialized':False,
             'measurement_scope':'outer kernel waited children includes actual supervisor and all reaped descendants; nested values not added'}
    write(EVIDENCE/'production-handshake-acceptance.json',summary)
    print(json.dumps({'all_cases_pass':summary['all_cases_pass'],'source_unchanged':summary['source_unchanged'],
                      'cases':[{'mode':v['mode'],'checks':v['checks']} for v in cases]},ensure_ascii=False))
    return 0 if summary['all_cases_pass'] and summary['source_unchanged'] else 1


if __name__=='__main__':
    raise SystemExit(main())
