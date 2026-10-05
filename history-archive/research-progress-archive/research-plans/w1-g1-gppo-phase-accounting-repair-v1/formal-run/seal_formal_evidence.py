"""Post-stop evidence summary only; no model deserialization or dynamic calls."""
import collections, hashlib, json, math, sqlite3, time
from pathlib import Path
R=Path(__file__).resolve().parent
O=R/'formal-result-audit'
E=R/'verified-export'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
started=time.perf_counter()
db=sqlite3.connect((E/'run-once/budget.sqlite3').as_uri()+'?mode=ro',uri=True)
calls=db.execute('SELECT stage,name,amounts,status FROM calls').fetchall();db.close()
amounts_by_status={};route_calls=collections.defaultdict(collections.Counter)
for stage,name,amounts,status in calls:
    amounts_by_status.setdefault(status,collections.Counter()).update(json.loads(amounts))
    if name.startswith(('policy_step:','ppo_update:','save_policy_checkpoint:')):
        op,method,seed=name.split(':')
        route_calls[(method,int(seed))][op+'_'+status]+=1
recorded=rows(E/'run-once/policy-training-routes.jsonl')
completed={(x['method'],x['seed']):x for x in recorded}
route_summary=[]
for method in ('G0','T','G1'):
    for seed in (8301,8302,8303):
        x=completed.get((method,seed));counter=route_calls[(method,seed)]
        value={'method':method,'seed':seed,'status':'complete_checkpoint_saved' if x else ('partial_interrupted' if counter else 'not_started'),
               'steps_complete':counter.get('policy_step_complete',0),
               'updates_complete':counter.get('ppo_update_complete',0),
               'updates_pending_unknown':counter.get('ppo_update_pending',0),
               'checkpoint_restoration':'未评价'}
        if x:
            cp=E/'run-once/policy-checkpoints'/Path(x['checkpoint']).name
            actual=digest(cp)
            assert actual==x['checkpoint_sha256']
            value.update({'checkpoint_relative':cp.relative_to(E).as_posix(),'checkpoint_sha256':actual,
                          'world_model_seed':x['world_model_seed'],'world_model_frozen_recorded':x['summary'].get('world_model_frozen'),
                          'summary_fields':sorted(x['summary'])})
        route_summary.append(value)
update_traces=rows(E/'run-once/policy-update-traces.jsonl')
update_metrics=collections.defaultdict(lambda:collections.defaultdict(list))
for x in update_traces:
    route=str(x['method'])+':'+str(x['seed'])
    for k,v in x['metrics'].items():
        if type(v) in (int,float) and math.isfinite(v):update_metrics[route][k].append(v)
training_metric_summary={route:{k:{'n':len(v),'mean':sum(v)/len(v),'last':v[-1],'min':min(v),'max':max(v)} for k,v in data.items()} for route,data in update_metrics.items()}
write(O/'training-metrics-descriptive.json',{'persisted_update_traces':len(update_traces),'by_route':training_metric_summary,
        'limits':'training metrics only; not task effect. Partial route in-memory traces were not persisted; SQLite completion and pending states retained.'})
sup=read(E/'supervisor-status.json');launch=read(R/'launch-evidence.json');native=read(R/'native-controller-settlement.json')
monitors=[]
for p in sorted(R.glob('formal-monitor-*.stdout.json')):
    try:monitors.append({'file':p.name,'cpu_seconds':read(p)['readonly_monitor_self_cpu_seconds']})
    except (ValueError,KeyError):pass
facts={'attempt':launch['attempt'],'status':'technical_stop','exit_code':launch['exit_code'],
       'first_error':read(E/'supervisor-first-error.json'),
       'route_summary':route_summary,'completed_checkpoint_count':len(completed),
       'ledger_call_count':len(calls),'ledger_amounts_by_status':amounts_by_status,
       'task_episodes':0,'task_comparison_calls':0,'task_utility':'未评价',
       'parent_seed_stability':'未评价','completion_expiration_confirmation_energy_comparison':'未评价',
       'decision_costs':{m:{'status':'未评价','cpu_mean_ms':None,'wall_p95_ms':None,'samples':0} for m in ('G0','T','G1','H')},
       'task_gate':'未评价','cost_gate':'未评价',
       'measured_resources':{'launch_wall_seconds':launch['wall_seconds'],'native_controller_wall_seconds':native['wall_seconds'],
            'native_controller_self_plus_waited_cpu_seconds':native['self_plus_waited_cpu_seconds'],
            'windows_controller_self_cpu_seconds':launch['windows_controller_cpu_seconds'],
            'peak_rss_bytes':sup['peak_rss_bytes'],'active_native_bytes':native['active_native_bytes'],
            'verified_export_bytes':native['verified_export_bytes'],'training_stage_tail':sup['phase_rows'][-1]},
       'accounting_limits':{'full_resource_acceptance':False,'phase_stage_resource_pass':sup['phase_stage_resource_pass'],
            'measured_controller_limits_pass':native['measured_controller_limits_pass'],
            'gaps':native['resource_gaps'],'worker_final_settlement_missing':True,
            'settlement_status':'incomplete','cpu_totals_are_nested_not_additive':True,
            'pending_update_may_have_partially_executed':'unknown; never counted as completed or zero'},
       'post_run_monitor_cpu':{'saved_monitor_count':len(monitors),'saved_self_cpu_sum_seconds':sum(x['cpu_seconds'] for x in monitors),
            'records':monitors,'scope':'separate readonly monitor processes, not nested formal CPU; unpersisted monitor output reported separately'},
       'audit_wall_seconds':time.perf_counter()-started,'audit_self_cpu_seconds':time.process_time(),
       'automatic_retry':False,'research_conclusion':'本次未完成任务评价，不能判断G1是否改善GPPO，也不能判断实用成本；不改写此前排序实验。'}
write(O/'sealed-run-facts.json',facts)
print(json.dumps({k:facts[k] for k in ('status','completed_checkpoint_count','ledger_call_count','ledger_amounts_by_status','route_summary')},ensure_ascii=False))
