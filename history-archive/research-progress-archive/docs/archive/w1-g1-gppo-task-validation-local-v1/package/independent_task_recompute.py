"""Stdlib-only recomputation from raw rewards/costs, independent of GPPO code."""
import json
import math
import random
from pathlib import Path

METHODS=('G0','T','G1','H')
class TaskEvidenceError(RuntimeError):pass
def finite(value,label):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise TaskEvidenceError('NONFINITE_OR_INVALID:'+label)
    return float(value)
def mean(values):return sum(values)/len(values) if values else None
def rank(values,q):
    values=sorted(values)
    return values[max(0,math.ceil(len(values)*q)-1)] if values else None
def read_lines(path):
    if not path.is_file():raise TaskEvidenceError('EVIDENCE_MISSING:'+path.name)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def recompute_records(episodes,costs,matrix):
    parents=[r['parent'] for r in matrix['splits']['task_confirmation']]
    seeds=matrix['policy_seeds'];repeats=matrix['repeats']['task_confirmation']
    expected={(p,r,m,s) for p in parents for r in range(repeats) for m in METHODS for s in ([None] if m=='H' else seeds)}
    indexed={};reward_checks=[]
    pref=matrix['preference_configuration']['task_confirmation']
    gamma=matrix['preference_configuration']['discount'];scale=matrix['preference_configuration']['task_component_scale']
    parent_spec={r['parent']:r for r in matrix['splits']['task_confirmation']}
    for row in episodes:
        key=(row.get('parent'),row.get('repeat'),row.get('method'),row.get('seed'))
        if key not in expected or key in indexed:raise TaskEvidenceError('EPISODE_IDENTITY_INVALID_OR_DUPLICATE:'+repr(key))
        spec=parent_spec[key[0]]
        expected_key=spec['exogenous_key'].rsplit('|repeat-',1)[0]+'|repeat-'+str(key[1])
        if row.get('exogenous_key')!=expected_key or any(row.get(k)!=spec[k] for k in ('scenario_sha256','structural_sha256')):
            raise TaskEvidenceError('EPISODE_INPUT_IDENTITY_MISMATCH:'+repr(key))
        if row.get('preference')!=pref:raise TaskEvidenceError('EPISODE_PREFERENCE_MISMATCH')
        steps=row.get('steps')
        if type(steps) is not int or not 0<=steps<=matrix['task_episode_max_steps']:raise TaskEvidenceError('EPISODE_STEPS_INVALID')
        # Reward completeness is separate from lifecycle/confirmation validity.
        valid=row.get('utility_valid') is True
        if valid:
            if not (row.get('terminated') is True or row.get('truncated') is True) or (row.get('terminated') is True and row.get('truncated') is True):
                raise TaskEvidenceError('EPISODE_COMPLETION_INVALID')
            rewards=row.get('vector_rewards')
            if rewards is None:rewards=[a.get('vector_reward') for a in row.get('actions',[])]
            if not isinstance(rewards,list) or len(rewards)!=steps:raise TaskEvidenceError('RAW_REWARD_SEQUENCE_MISSING')
            total=0.0
            for i,reward in enumerate(rewards):
                if not isinstance(reward,(list,tuple)) or len(reward)!=2:raise TaskEvidenceError('VECTOR_REWARD_INVALID')
                total+=gamma**i*(scale*pref[0]*finite(reward[0],'task reward')+pref[1]*finite(reward[1],'energy reward'))
            recorded=finite(row.get('utility'),'recorded utility')
            if abs(total-recorded)>1e-8:raise TaskEvidenceError('RAW_REWARD_UTILITY_MISMATCH:'+repr(key))
            reward_checks.append({'key':list(key),'utility':total,'recorded_utility':recorded,'absolute_error':abs(total-recorded)})
        else:
            total=None
            if row.get('utility') is not None:raise TaskEvidenceError('INVALID_EPISODE_HAS_KNOWN_UTILITY')
        indexed[key]=(row,total)
    if set(indexed)!=expected:
        raise TaskEvidenceError('TASK_MATRIX_INCOMPLETE:'+str(len(expected-set(indexed))))
    unit={};parent_results={};complete=[]
    for p in parents:
        for r in range(repeats):
            values={m:[indexed.get((p,r,m,s),(None,None))[1] for s in ([None] if m=='H' else seeds)] for m in METHODS}
            unit[(p,r)]={m:mean(v) if all(x is not None for x in v) else None for m,v in values.items()}
        parent_results[p]={m:mean([unit[(p,r)][m] for r in range(repeats)]) if all(unit[(p,r)][m] is not None for r in range(repeats)) else None for m in METHODS}
        if all(v is not None for v in parent_results[p].values()):complete.append(p)
    macro={m:mean([v[m] for v in parent_results.values() if v[m] is not None]) for m in METHODS}
    differences={base:{p:v['G1']-v[base] for p,v in parent_results.items() if v['G1'] is not None and v[base] is not None} for base in ('G0','T','H')}
    deltas={base:mean(list(v.values())) for base,v in differences.items()}
    seed_deltas={}
    for s in seeds:
        pairs=[indexed.get((p,r,'G1',s),(None,None))[1]-indexed.get((p,r,'T',s),(None,None))[1]
               for p in parents for r in range(repeats)
               if indexed.get((p,r,'G1',s),(None,None))[1] is not None and indexed.get((p,r,'T',s),(None,None))[1] is not None]
        seed_deltas[str(s)]=mean(pairs) if len(pairs)==len(parents)*repeats else None
    boot=None
    if len(complete)==len(parents):
        rng=random.Random(matrix['task_gates']['bootstrap_seed']);v=list(differences['T'].values())
        samples=[mean([v[rng.randrange(len(v))] for _ in v]) for _ in range(matrix['task_gates']['bootstrap_parent_samples'])]
        boot={'unit':'paired parent','n':len(v),'samples':len(samples),'seed':matrix['task_gates']['bootstrap_seed'],
              'lower_95':rank(samples,.025),'upper_95':rank(samples,.975),'percentile_rule':'nearest rank'}
    cost_keys=set();by_method={};by_parent_repeat={}
    for row in costs:
        key=(row.get('parent'),row.get('repeat'),row.get('method'),row.get('seed'))
        ident=(*key,row.get('decision_step'))
        if key not in indexed or ident in cost_keys:raise TaskEvidenceError('COST_IDENTITY_INVALID_OR_DUPLICATE:'+repr(ident))
        if type(row.get('decision_step')) is not int or not 0<=row['decision_step']<indexed[key][0]['steps']:raise TaskEvidenceError('COST_STEP_INVALID')
        if type(row.get('candidate_count')) is not int or not 1<=row['candidate_count']<=25:raise TaskEvidenceError('COST_CANDIDATE_COUNT_INVALID')
        for name in ('cpu_seconds','wall_seconds'):
            if finite(row.get(name),name)<0:raise TaskEvidenceError('COST_NEGATIVE')
        cost_keys.add(ident)
    for key,(row,_) in indexed.items():
        if len([c for c in cost_keys if c[:4]==key])!=row['steps']:raise TaskEvidenceError('DECISION_COST_RECORDS_INCOMPLETE:'+repr(key))
    def summarize(rows):
        active=[r for r in rows if r['candidate_count']>1]
        return {'status':'evaluated' if active else 'not_evaluated','sample_count':len(rows),'active_decision_count':len(active),
                'cpu_mean_ms':1000*mean([r['cpu_seconds'] for r in rows]) if rows else None,
                'wall_p95_ms':1000*rank([r['wall_seconds'] for r in rows],.95) if rows else None,
                'world_model_forwards':sum(r.get('world_model_forwards',0) for r in rows)}
    for m in METHODS:
        rows=[r for r in costs if r['method']==m];by_method[m]=summarize(rows)
        for p in parents:
            for r in range(repeats):by_parent_repeat[f'{m}|{p}|{r}']=summarize([x for x in rows if x['parent']==p and x['repeat']==r])
    return {'schema':'w1-g1-gppo-independent-task-metrics/1.0.0','episode_count':len(episodes),
            'expected_episode_count':len(expected),'complete_parent_count':len(complete),'complete_parents':complete,
            'unknown_episode_count':sum(v[1] is None for v in indexed.values()),
            'lifecycle_unevaluated_episode_count':sum(row.get('outcome_valid') is not True for row,_ in indexed.values()),'complete_matrix':set(indexed)==expected,
            'reward_recomputation':reward_checks,'parent_macro_utility':macro,'per_parent':parent_results,
            'paired_parent_differences':differences,'g1_minus_g0_macro_utility':deltas['G0'],
            'g1_minus_t_macro_utility':deltas['T'],'g1_minus_h_macro_utility':deltas['H'],
            'seed_g1_minus_t':seed_deltas,'parents_g1_no_worse_than_t':sum(x>=0 for x in differences['T'].values()),
            'bootstrap_g1_minus_t':boot,'decision_costs':{'by_method':by_method,'by_method_parent_repeat':by_parent_repeat}}

def evaluate_frozen_gates(matrix,metrics):
    g=matrix['task_gates'];boot=metrics['bootstrap_g1_minus_t']
    tests={'coverage':metrics['complete_matrix'] and metrics['complete_parent_count']>=g['minimum_complete_parents'] and metrics['unknown_episode_count']==0,
           'G1_vs_G0':metrics['g1_minus_g0_macro_utility'] is not None and metrics['g1_minus_g0_macro_utility']>0,
           'G1_vs_T':metrics['g1_minus_t_macro_utility'] is not None and metrics['g1_minus_t_macro_utility']>0,
           'G1_vs_H':metrics['g1_minus_h_macro_utility'] is not None and metrics['g1_minus_h_macro_utility']>=g['g1_minus_h_macro_utility_minimum'],
           'parent_stability':metrics['parents_g1_no_worse_than_t']>=g['minimum_noninferior_parents_vs_t'],
           'seed_stability':sum(x is not None and x>0 for x in metrics['seed_g1_minus_t'].values())>=g['minimum_positive_policy_seeds_vs_t'],
           'parent_bootstrap':boot is not None and boot['lower_95']>0}
    c=metrics['decision_costs']['by_method']['G1']
    cost_evaluated=c['status']=='evaluated' and c['world_model_forwards']>0
    cost=cost_evaluated and c['cpu_mean_ms']<=g['cpu_mean_ms_maximum'] and c['wall_p95_ms']<=g['wall_p95_ms_maximum']
    task=all(tests.values())
    return {'task_gate':{'pass':task,'checks':tests},'cost_gate':{'status':'evaluated' if cost_evaluated else 'not_evaluated','pass':cost,'G1_only':c},
            'status':'task_validation_pass' if task and cost else ('task_gain_gate_stop' if not task else 'cost_gate_stop'),
            'research_success':bool(task and cost)}

def recompute_from_files(output,matrix):
    output=Path(output)
    metrics=recompute_records(read_lines(output/'task-confirmation.jsonl'),read_lines(output/'decision-costs.jsonl'),matrix)
    (output/'independent-task-metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return metrics
