"""Post-run artifact audit and exact metric recomputation; no torch/model/environment calls."""
import datetime,hashlib,importlib.util,json,math,sqlite3,sys,os
from collections import Counter,defaultdict
from pathlib import Path
R=Path(__file__).resolve().parent;P=R/'package';E=R/'verified-export';O=R/'formal-result-audit'
N=Path(r'\\wsl.localhost\Ubuntu-24.04\home\asus\w1-g1-gppo-recovery-v1-once\package')
if '--native-recomputation' in sys.argv:
    O=O/'native-recomputation'
if os.name=='posix':
    N=Path('/home/asus/w1-g1-gppo-recovery-v1-once/package')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.is_file() else []
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,v):(O/name).write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def mean(v):return sum(v)/len(v) if v else None
def load_module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
assert (R/'launch-evidence.json').is_file(),'FORMAL_ENTRY_NOT_FINISHED'
assert not O.exists(),'AUDIT_DESTINATION_ALREADY_EXISTS'
O.mkdir()
identity=read(R/'frozen-identity.json');request=read(P/'RESOURCE_REQUEST.json');matrix=read(P/'experiment-matrix.json')
before=read(R/'audit/final-frozen-input-hashes.json')
assert {f.relative_to(P).as_posix():sha(f) for f in P.rglob('*') if f.is_file()}==before,'FROZEN_INPUT_CHANGED'
assert sha(P/'execution-manifest.json')==identity['execution_manifest_sha256']
assert sha(P/'hashes.json')==identity['hashes_sha256']
assert sha(P/'RESOURCE_REQUEST.json')==identity['resource_request_sha256']
assert request['status']=='NOT_APPROVED'
export_checks=[]
if (E/'export-hashes.json').is_file():
    exported=read(E/'export-hashes.json')
    for name,digest in exported.items():
        p=E/name
        assert not p.is_symlink() and p.is_file() and sha(p)==digest,'EXPORT_HASH:'+name
        np=(N.parent/name if name.split('/')[0] in ('native-launch.stdout.txt','controller-first-error.txt','authorization-consumed.json') else N/name)
        native_match=np.is_file() and sha(np)==digest
        assert native_match,'NATIVE_EXPORT_IDENTITY:'+name
        export_checks.append({'file':name,'sha256':digest,'native_export_match':True,'bytes':p.stat().st_size})
    actual={p.relative_to(E).as_posix() for p in E.rglob('*') if p.is_file()}-{'export-hashes.json'}
    assert actual==set(exported),'EXPORT_MANIFEST_SET_MISMATCH'
write('export-hash-audit.json',{'verified_payload_count':len(export_checks),'files':export_checks,'frozen_inputs_unchanged':True})
launch=read(R/'launch-evidence.json');controller=read(R/'native-controller-settlement.json') if (R/'native-controller-settlement.json').is_file() else None
supervisor=read(E/'supervisor-status.json') if (E/'supervisor-status.json').is_file() else None
status=read(E/'run-once/status.json') if (E/'run-once/status.json').is_file() else None
settlement=read(E/'run-once/resource-settlement.json') if (E/'run-once/resource-settlement.json').is_file() else None
first=read(E/'run-once/first-error.json') if (E/'run-once/first-error.json').is_file() else (read(E/'supervisor-first-error.json') if (E/'supervisor-first-error.json').is_file() else None)
totals=defaultdict(Counter);stages=defaultdict(lambda:defaultdict(Counter));names=defaultdict(Counter);calls=[]
db=E/'run-once/budget.sqlite3'
if db.is_file():
    with sqlite3.connect('file:'+str(db)+'?mode=ro&immutable=1',uri=True) as c:
        c.execute('PRAGMA query_only=ON')
        for id,stage,name,amounts,call_status,started,finished,error in c.execute('SELECT id,stage,name,amounts,status,started,finished,error FROM calls ORDER BY id'):
            amount=json.loads(amounts);totals[call_status].update(amount);stages[stage][call_status].update(amount);names[call_status][name]+=1
            calls.append({'id':id,'stage':stage,'name':name,'status':call_status,'amounts':amount,'error':error,'started':started,'finished':finished})
ledger={'call_status_counts':dict(Counter(x['status'] for x in calls)),'totals':{k:dict(v) for k,v in totals.items()},
        'stages':{k:{s:dict(a) for s,a in v.items()} for k,v in stages.items()},'pending':[x for x in calls if x['status']=='pending'],
        'failed':[x for x in calls if x['status'] not in ('pending','complete')],'operation_counts':{k:dict(v) for k,v in names.items()}}
reserved=Counter()
for v in totals.values():reserved.update(v)
ledger['reserved_including_pending']=dict(reserved)
ledger['global_count_cap_checks']={k:reserved.get(k,0)<=cap for k,cap in request['totals'].items() if k in reserved}
ledger['stage_count_cap_checks']={stage:{k:sum(a.get(k,0) for a in bystatus.values())<=cap for k,cap in request['stages'][stage].items() if any(k in a for a in bystatus.values())} for stage,bystatus in stages.items()}
write('ledger-audit.json',ledger)
raw=load_module('raw_checkpoint_audit',R/'audit/audit_recovery.py')
reuse=read(P/'recovery-contract.json')['reuse_routes'];new=lines(E/'run-once/policy-training-routes.jsonl')
new_index={(r['method'],r['seed']):r for r in new}
checkpoint_rows=[]
for method in ('G0','T','G1'):
    for seed in (8301,8302,8303):
        rr=next((r for r in reuse if (r['method'],r['seed'])==(method,seed)),None)
        row=rr or new_index.get((method,seed))
        if row is None:
            checkpoint_rows.append({'method':method,'seed':seed,'status':'not_completed','reuse':False});continue
        cp=P/rr['checkpoint_relative_path'] if rr else E/'run-once/policy-checkpoints'/Path(row['checkpoint']).name
        facts=raw.checkpoint_facts(cp)
        assert facts['archive_sha256']==row['checkpoint_sha256'] and facts['state_hash_matches_stored']
        assert facts['method']==method and facts['seed']==seed
        assert facts['recomputed_state_sha256']==row['checkpoint_state_sha256']
        assert facts['summary_environment_steps']==2048 and facts['summary_optimizer_updates']==128
        assert facts['optimizer_step_values']==[128.0] and row['training_steps']==2048 and row['optimizer_updates']==128
        if not rr:
            assert names['complete'].get(f'ppo_update:{method}:{seed}',0)==128
            assert names['complete'].get(f'save_policy_checkpoint:{method}:{seed}',0)==1
        checkpoint_rows.append({'method':method,'seed':seed,'status':'complete','reuse':rr is not None,
                                'file':str(cp),'source_attempt':read(P/'recovery-contract.json')['source_attempt'] if rr else request['attempt'],'facts':facts})
write('nine-checkpoint-audit.json',{'routes':checkpoint_rows,'complete_count':sum(r['status']=='complete' for r in checkpoint_rows),
                                  'reuse_count':sum(r['reuse'] for r in checkpoint_rows),'torch_imported':'torch' in sys.modules})
episodes=lines(E/'run-once/task-confirmation.jsonl');costs=lines(E/'run-once/decision-costs.jsonl')
metrics=None;gates=None;decomposition=None
if len(episodes)==240:
    recompute=load_module('independent_recovery_metrics',P/'independent_task_recompute.py')
    metrics=recompute.recompute_records(episodes,costs,matrix);gates=recompute.evaluate_frozen_gates(matrix,metrics)
    original=read(E/'run-once/independent-task-metrics.json')
    assert metrics==original,'POST_EXPORT_METRICS_DISAGREE'
    assert gates=={k:v for k,v in read(E/'run-once/task-gate.json').items() if k not in ('schema','automatic_retry')},'POST_EXPORT_GATES_DISAGREE'
    write('post-export-independent-metrics.json',metrics);write('post-export-gates.json',gates)
    pref=matrix['preference_configuration']['task_confirmation'];gamma=matrix['preference_configuration']['discount'];scale=matrix['preference_configuration']['task_component_scale']
    def parts(row):
        task=sum(gamma**i*scale*pref[0]*r[0] for i,r in enumerate(row['vector_rewards']))
        energy=sum(gamma**i*pref[1]*r[1] for i,r in enumerate(row['vector_rewards']))
        assert abs(task+energy-row['utility'])<=1e-8
        lifecycle=row.get('lifecycle_outcomes',[])
        labels={}
        for head in ('physical_on_time_completion','host_confirmation'):
            vals=[x[head] for x in lifecycle if head in x]
            labels[head]={'valid':sum(x['valid'] for x in vals),'positive':sum(x['valid'] and x['value'] is True for x in vals),
                          'negative':sum(x['valid'] and x['value'] is False for x in vals),'unknown':sum(not x['valid'] for x in vals),
                          'unknown_reasons':dict(Counter(x.get('reason') for x in vals if not x['valid']))}
        actions=row.get('actions',[])
        return {'task_net_reward_contribution':task,'energy_reward_contribution':energy,'utility':row['utility'],
                'completed_count':(row.get('counts') or {}).get('completed'),'expired_count':(row.get('counts') or {}).get('expired'),
                'energy_used':row.get('energy_used'),'physical_rate':row.get('physical_rate'),'host_rate':row.get('host_rate'),
                'noop_count':sum(a.get('selected_action',a.get('action'))==24 for a in actions),'steps':row['steps'],
                'lifecycle':labels,'terminal_state_counts':dict(Counter(x.get('terminal_state','unknown') for x in lifecycle)),
                'argmax_changed_count':row.get('argmax_changed_count'),
                'sampled_action_changed_count':row.get('sampled_action_changed_count')}
    detail=[{'method':r['method'],'seed':r['seed'],'parent':r['parent'],'repeat':r['repeat'],**parts(r)} for r in episodes]
    summaries={}
    for method in ('G0','T','G1','H'):
        rows=[r for r in detail if r['method']==method]
        keys=('utility','task_net_reward_contribution','energy_reward_contribution','completed_count','expired_count','energy_used','physical_rate','host_rate')
        values={key:mean([r[key] for r in rows if r[key] is not None]) for key in keys}
        values.update({'episode_count':len(rows),'noop_count':sum(r['noop_count'] for r in rows),'decision_count':sum(r['steps'] for r in rows),
                       'argmax_changed_count':sum(r['argmax_changed_count'] or 0 for r in rows),
                       'sampled_action_changed_count':sum(r['sampled_action_changed_count'] or 0 for r in rows),
                       'known_host_rate_episode_count':sum(r['host_rate'] is not None for r in rows)})
        values['lifecycle']={}
        for head in ('physical_on_time_completion','host_confirmation'):
            agg=Counter();reasons=Counter()
            for r in rows:
                agg.update({k:r['lifecycle'][head][k] for k in ('valid','positive','negative','unknown')});reasons.update(r['lifecycle'][head]['unknown_reasons'])
            values['lifecycle'][head]={**dict(agg),'unknown_reasons':dict(reasons)}
        state_counts=Counter()
        for r in rows:state_counts.update(r['terminal_state_counts'])
        values['terminal_state_counts']=dict(state_counts)
        values['expiry_label_note']='No separate task_expired head in task-evaluation lifecycle rows; explicit expiry is reported from terminal_state and final counts.'
        summaries[method]=values
    per_seed={str(seed):{m:mean([r['utility'] for r in episodes if r['method']==m and r['seed']==seed]) for m in ('G0','T','G1')} for seed in (8301,8302,8303)}
    for seed,row in per_seed.items():row['G1_minus_G0']=row['G1']-row['G0'];row['G1_minus_T']=row['G1']-row['T'];row['G1_minus_H']=row['G1']-metrics['parent_macro_utility']['H']
    decomposition={'by_method':summaries,'episode_details':detail,'per_seed_utility':per_seed,
                   'g1_minus_controls_reward_components':{m:{k:summaries['G1'][k]-summaries[m][k] for k in ('utility','task_net_reward_contribution','energy_reward_contribution','completed_count','expired_count','energy_used')} for m in ('G0','T','H')},
                   'reward_formula':'discounted 0.8 * task_component_scale * (new_completed - new_expired)/task_capacity + 0.2 * (-energy_used/fleet_initial_energy)',
                   'host_confirmation_reward_contribution':'not a separate reward term; report valid labels/unknown separately',
                   'completed_vs_expired_discounted_contribution':'not uniquely separated from net vector_rewards here; counts are descriptive, no fabricated per-step attribution'}
    write('task-outcome-decomposition.json',decomposition)
old=read(P/'recovery-contract.json');oldtot=old['old_consumption'];cumulative={}
for k in sorted(set(reserved)|set(oldtot['complete'])|set(oldtot['pending'])):
    cumulative[k]={'old_complete':oldtot['complete'].get(k,0),'old_pending_unknown':oldtot['pending'].get(k,0),
                   'new_complete':totals['complete'].get(k,0),'new_pending_unknown':totals['pending'].get(k,0),
                   'new_reserved':reserved.get(k,0),'cumulative_complete':oldtot['complete'].get(k,0)+totals['complete'].get(k,0),
                   'cumulative_reserved':oldtot['complete'].get(k,0)+oldtot['pending'].get(k,0)+reserved.get(k,0)}
oldresource=read(P/'cumulative-resource-contract.json')
resource={'new_launch':launch,'new_native_controller':controller,'supervisor':supervisor,
          'old_wall_seconds':oldresource['old_windows_launch_wall_seconds'],
          'cumulative_launch_wall_seconds':oldresource['old_windows_launch_wall_seconds']+launch['wall_seconds'],
          'cumulative_cpu_measured_disjoint_scope_seconds':(oldresource['old_native_inclusive_cpu_seconds']+oldresource['old_windows_self_cpu_seconds']+controller['self_plus_waited_cpu_seconds']+launch['windows_controller_cpu_seconds']) if controller else None,
          'cpu_scope_rule':'Add old/new controller SELF+waited CHILDREN and Windows SELF only; do not add supervisor/worker stage CPU again.',
          'resource_gaps':['Windows/WSL bridge CPU','controller final write/exit tail','outside-entry prelaunch and offline audit/monitoring process lifetime not comprehensively measured'],
          'full_resource_acceptance':False,'call_resources':cumulative,'old_pending_update_not_assumed_complete':True}
write('cumulative-resource-audit.json',resource)
facts={'attempt':request['attempt'],'audit_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'launch_exit_code':launch['exit_code'],'worker_status':status,'first_error':first,
       'checkpoint_complete_count':sum(r['status']=='complete' for r in checkpoint_rows),'reused_checkpoint_count':sum(r['reuse'] for r in checkpoint_rows),
       'new_training_checkpoint_count':len(new),'task_episode_records':len(episodes),'decision_cost_records':len(costs),
       'metrics_recomputed_and_matched':metrics is not None,'gates':gates,'verified_export_payload_count':len(export_checks),
       'frozen_inputs_unchanged':True,'audit_model_initializations':0,'audit_model_forwards':0,'audit_environment_calls':0,'torch_imported':'torch' in sys.modules}
write('sealed-run-facts.json',facts)
print(json.dumps({k:v for k,v in facts.items() if k not in ('worker_status','first_error')},ensure_ascii=False))
