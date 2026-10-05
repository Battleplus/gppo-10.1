"""Offline stdlib audit of this consumed run; never load models or alter raw evidence."""
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import time

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / 'package'
EXPORT = ROOT / 'verified-export'
OUTPUT = ROOT / 'formal-result-audit'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(name, value):
    (OUTPUT / name).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                        indent=2, allow_nan=False) + '\n', encoding='utf-8')

def lines(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def macro(rows, field, method, seed=None):
    units = collections.defaultdict(list)
    for row in rows:
        if row['method'] == method and (seed is None or row['seed'] == seed):
            if row.get(field) is not None:
                units[(row['parent'], row['repeat'])].append(row[field])
    parents = collections.defaultdict(list)
    for (parent, _repeat), values in units.items():
        parents[parent].append(sum(values) / len(values))
    return sum(sum(v)/len(v) for v in parents.values())/len(parents) if parents else None

def main():
    started = time.perf_counter()
    OUTPUT.mkdir(exist_ok=False)
    manifest = read(PACKAGE / 'execution-manifest.json')
    frozen_mismatches = [n for n,h in manifest['files'].items() if digest(PACKAGE/n) != h]
    hashes = read(EXPORT / 'export-hashes.json')
    export_mismatches = [n for n,h in hashes.items() if digest(EXPORT/n) != h]
    request = read(PACKAGE / 'RESOURCE_REQUEST.json')
    db = EXPORT / 'run-once' / 'budget.sqlite3'
    con = sqlite3.connect(db.as_uri() + '?mode=ro', uri=True)
    calls = con.execute('SELECT stage,name,amounts,status,error FROM calls').fetchall()
    con.close()
    used, stage_used, status, failure = collections.Counter(), {}, collections.Counter(), []
    for stage, name, raw, state, error in calls:
        amounts = json.loads(raw)
        used.update(amounts)
        stage_used.setdefault(stage, collections.Counter()).update(amounts)
        status[state] += 1
        if state != 'complete':
            failure.append({'stage':stage,'name':name,'status':state,'error':error,'amounts':amounts})
    limits = {'global':{k:{'used':v,'cap':request['totals'].get(k),'within_cap':v<=request['totals'].get(k,-1)} for k,v in used.items()},
              'by_stage':{s:{k:{'used':v,'cap':request['stages'][s].get(k),'within_cap':v<=request['stages'][s].get(k,-1)} for k,v in values.items()} for s,values in stage_used.items()}}
    save('ledger-audit.json', {'call_count':len(calls),'call_statuses':dict(status),'noncomplete_calls':failure,
                             'used':dict(used),'by_stage':stage_used,'limits':limits,
                             'reserved_failed_calls_are_not_claimed_successful':True})
    run = EXPORT / 'run-once'
    value = {'attempt':request['attempt'],'frozen_mismatches':frozen_mismatches,
             'verified_export_files':len(hashes),'export_mismatches':export_mismatches,
             'full_resource_acceptance':False,
             'launch':read(ROOT/'launch-evidence.json'),
             'native_controller':read(ROOT/'native-controller-settlement.json')}
    episode_path = run / 'task-confirmation.jsonl'
    if episode_path.exists() and (run/'independent-task-metrics.json').exists():
        episodes, costs = lines(episode_path), lines(run/'decision-costs.jsonl')
        matrix = read(PACKAGE/'experiment-matrix.json')
        spec = importlib.util.spec_from_file_location('offline_frozen_recompute', PACKAGE/'independent_task_recompute.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        metrics = mod.recompute_records(episodes, costs, matrix)
        recorded = read(run/'independent-task-metrics.json')
        value['independent_metrics_exact_match'] = metrics == recorded
        gates = mod.evaluate_frozen_gates(matrix,metrics)
        value['gates_exact_match'] = all(read(run/'task-gate.json')[k] == v for k,v in gates.items())
        save('independently-recomputed-metrics.json',metrics)
        save('independently-recomputed-gates.json',gates)
        decomposed=[]
        pref=matrix['preference_configuration']['task_confirmation']
        gamma=matrix['preference_configuration']['discount']
        scale=matrix['preference_configuration']['task_component_scale']
        for e in episodes:
            row={k:e[k] for k in ('method','seed','parent','repeat','utility')}
            valid=e.get('utility_valid') is True
            row['task_reward_utility']=sum(gamma**i*scale*pref[0]*r[0] for i,r in enumerate(e['vector_rewards'])) if valid else None
            row['energy_reward_utility']=sum(gamma**i*pref[1]*r[1] for i,r in enumerate(e['vector_rewards'])) if valid else None
            if valid and abs(row['task_reward_utility']+row['energy_reward_utility']-e['utility'])>1e-8:
                raise RuntimeError('DECOMPOSITION_MISMATCH')
            row['energy_used']=e.get('energy_used')
            row['completed_count']=e.get('counts',{}).get('completed')
            row['expired_count']=e.get('counts',{}).get('expired')
            row['noop_count']=sum(a.get('selected_action',a.get('action'))==24 for a in e['actions'])
            row['steps']=e['steps']
            row['prior_argmax_changed_count']=e['argmax_changed_count']
            row['sampled_action_changed_count']=e['sampled_action_changed_count']
            for name in ('physical_on_time_completion','host_confirmation'):
                fields=[l[name] for l in e['lifecycle_outcomes']]
                for label,predicate in [('positive',lambda x:x['valid'] and x['value'] is True),
                                        ('negative',lambda x:x['valid'] and x['value'] is False),
                                        ('unknown',lambda x:not x['valid'])]:
                    row[name+'_'+label]=sum(predicate(x) for x in fields)
            decomposed.append(row)
        summary={m:{f:macro(decomposed,f,m) for f in decomposed[0] if f not in ('method','seed','parent','repeat')} for m in ('G0','T','G1','H')}
        seed_summary={str(s):{m:macro(episodes,'utility',m,s) for m in ('G0','T','G1')} for s in matrix['policy_seeds']}
        for values in seed_summary.values():
            for base in ('G0','T'):
                values['G1_minus_'+base]=values['G1']-values[base] if values['G1'] is not None and values[base] is not None else None
            h=macro(episodes,'utility','H')
            values['G1_minus_H']=values['G1']-h if values['G1'] is not None and h is not None else None
        deltas=metrics['paired_parent_differences']['T']
        positive=sorted((v for v in deltas.values() if v>0),reverse=True)
        concentration={'top_parent_share_of_positive_gain':positive[0]/sum(positive) if positive else None,
                       'top_two_share_of_positive_gain':sum(positive[:2])/sum(positive) if positive else None,
                       'interpretation':'descriptive; no selection or gate changes'}
        save('episode-reward-outcome-decomposition.json', {'rows':decomposed,'parent_equal_weight_macro':summary,
             'utility_by_seed':seed_summary,'gain_concentration':concentration,
             'limits':'Task reward and energy contributions are additive. Completion, expiration and confirmation are descriptive outcomes; no unrecorded causal attribution is invented. H is shared, not independent seed replicates.'})
        value.update({'episode_count':len(episodes),'cost_count':len(costs),'metrics_summary':{
             k:metrics[k] for k in ('parent_macro_utility','g1_minus_g0_macro_utility','g1_minus_t_macro_utility',
             'g1_minus_h_macro_utility','seed_g1_minus_t','bootstrap_g1_minus_t')},'gates':gates,
             'decomposition_summary':summary,'seed_summary':seed_summary,'concentration':concentration})
    else:
        value['task_evaluation']='未评价或未完成；保留首次失败，不补造结果'
    value['audit_wall_seconds']=time.perf_counter()-started
    value['audit_self_cpu_seconds']=time.process_time()
    value['audit_cpu_scope']='separate post-run offline verification, not nested study CPU'
    save('formal-audit-summary.json',value)
    save('audit-artifact-hashes.json',{p.name:digest(p) for p in OUTPUT.iterdir() if p.is_file()})
    print(json.dumps(value,ensure_ascii=False))

if __name__=='__main__':
    main()
