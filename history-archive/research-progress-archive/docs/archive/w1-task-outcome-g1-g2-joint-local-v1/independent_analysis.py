"""Independent stdlib arithmetic on saved trace; zero model/environment calls."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'verified-export/run-once'
rows=[json.loads(line) for line in (RUN/'prediction-trace.jsonl').read_text(encoding='utf-8').splitlines()]
metrics=json.loads((RUN/'prediction-metrics.json').read_text(encoding='utf-8'))
windows={r['parent']:r for r in (json.loads(line) for line in (RUN/'world-model-windows.jsonl').read_text(encoding='utf-8').splitlines())}
methods=['transparent','G1','G2']+[f'{v}_seed_{s}' for v in ('G1','G2') for s in (8201,8202,8203)]
events=('public_field_change','new_measurement_received','continuation_publicly_confirmed','physical_completion_observed','host_confirmation_observed')
heads=('physical_on_time_completion','task_expired','host_confirmation')
regrets={m:{} for m in methods};top1={m:{} for m in methods};errors={m:{} for m in methods};per_window=[]
event_counts={h:{'valid':0,'positive':0,'negative':0,'unknown':0} for h in events}
horizon_counts={h:{'valid':0,'positive':0,'negative':0,'unknown':0} for h in heads}
candidate_total=0
for row in rows:
    parent=row['parent'];cs=row['candidate_rows'];candidate_total+=len(cs)
    assert len(cs)==row['candidate_count'] and len({c['candidate_id'] for c in cs})==len(cs)
    assert row['split']=='prediction_confirmation' and row['continuation_id']=='hungarian-v1-fixed'
    assert row['input_hash']==windows[parent]['input_hash']
    task_w,energy_w=row['scoring_preference'];task_scale=row['task_component_scale']
    utilities=[]
    for c in cs:
        sequence=c['sequence_return'];rv=sequence['vector_reward_sum']
        truth=task_scale*task_w*rv[0]+energy_w*rv[1]
        assert math.isclose(truth,c['true_utility'],rel_tol=0,abs_tol=1e-9)
        assert all(c['outcome_valid'][i] for i in (3,4))
        utilities.append(truth)
        for i,h in enumerate(events):
            mask=c['event_valid'][i];value=c['event_labels'][i]
            if mask:event_counts[h]['valid']+=1;event_counts[h]['positive' if value else 'negative']+=1
            else:assert value is None;event_counts[h]['unknown']+=1
        for i,h in enumerate(heads):
            mask=c['horizon_task_outcome_valid'][i];value=c['horizon_task_outcome_target'][i]
            if mask:horizon_counts[h]['valid']+=1;horizon_counts[h]['positive' if value else 'negative']+=1
            else:assert value is None;horizon_counts[h]['unknown']+=1
        for v in ('G1','G2'):
            assert set(c['predictions'][v])=={'8201','8202','8203','ensemble'}
            for dim in range(6):
                avg=sum(c['predictions'][v][str(s)]['outcome'][dim] for s in (8201,8202,8203))/3
                assert math.isclose(avg,c['predictions'][v]['ensemble']['outcome'][dim],rel_tol=1e-6,abs_tol=1e-6)
    oracle=min(range(len(cs)),key=lambda i:(-utilities[i],cs[i]['action_id'],cs[i]['candidate_id']))
    decisions={}
    for method in methods:
        scores=[]
        for c in cs:
            score=c['transparent_score']
            if method!='transparent':
                variant,_,seed=method.partition('_seed_');out=c['predictions'][variant][seed or 'ensemble']['outcome']
                score+=task_scale*task_w*out[4]+energy_w*out[3]
            assert math.isfinite(score);scores.append(score)
        picked=min(range(len(cs)),key=lambda i:(-scores[i],cs[i]['action_id'],cs[i]['candidate_id']))
        regret=max(0.0,utilities[oracle]-utilities[picked]);action=cs[picked]['action_id']
        assert action==row['decisions'][method]['selected_action']
        assert math.isclose(regret,metrics['parent_macro_regret'][method][parent],rel_tol=0,abs_tol=1e-12)
        regrets[method][parent]=regret;top1[method][parent]=float(picked==oracle)
        assert top1[method][parent]==metrics['parent_macro_top1'][method][parent]
        assert cs[oracle]['action_id']==row['decisions'][method]['oracle_action']
        errors[method][parent]={'mae':sum(abs(a-b) for a,b in zip(scores,utilities))/len(cs),
            'mse':sum((a-b)**2 for a,b in zip(scores,utilities))/len(cs)}
        decisions[method]={'action':action,'true_utility':utilities[picked],'regret':regret}
    per_window.append({'parent':parent,'candidate_count':len(cs),'oracle_action':cs[oracle]['action_id'],'decisions':decisions})
assert len(rows)==8 and {r['parent'] for r in rows}=={f'train-{i:04d}' for i in range(96,104)}
macro={m:{'regret':statistics.mean(regrets[m].values()),'top1':statistics.mean(top1[m].values()),
    'full_continuation_utility_mae':statistics.mean(v['mae'] for v in errors[m].values()),
    'full_continuation_utility_rmse':math.sqrt(statistics.mean(v['mse'] for v in errors[m].values()))} for m in methods}
paired={}
for name,candidate,baseline in [('G1_vs_transparent','G1','transparent'),('G2_vs_G1','G2','G1')]:
    changes={p:regrets[baseline][p]-regrets[candidate][p] for p in regrets[baseline]}
    seed_deltas={str(s):macro[(baseline+'_seed_'+str(s)) if baseline!='transparent' else baseline]['regret']-macro[candidate+'_seed_'+str(s)]['regret'] for s in (8201,8202,8203)}
    frozen=metrics['prediction_gates']['g1_vs_transparent' if candidate=='G1' else 'g2_vs_g1']
    count=sum(v>=0 for v in changes.values());positive_seeds=sum(v>0 for v in seed_deltas.values());gain=statistics.mean(changes.values())
    passed=gain>=frozen['regret_improvement_minimum'] and count>=frozen['noninferior_parent_minimum'] and positive_seeds>=frozen['positive_improvement_seed_minimum']
    assert passed==metrics['worth_gppo_suggestions'][name]['worth_gppo_suggestion']
    positive=sum(max(0.0,v) for v in changes.values())
    paired[name]={'gate_pass':passed,'macro_regret_improvement':gain,'parent_improvements':changes,
        'noninferior_parents':count,'seed_improvements':seed_deltas,'positive_seeds':positive_seeds,
        'largest_parent_fraction_of_positive_improvement':max(changes.values())/positive if positive else None}
seed_stability={v:{'seed_macro_regrets':{str(s):macro[f'{v}_seed_{s}']['regret'] for s in (8201,8202,8203)},
    'regret_range':max(macro[f'{v}_seed_{s}']['regret'] for s in (8201,8202,8203))-min(macro[f'{v}_seed_{s}']['regret'] for s in (8201,8202,8203)),
    'all_seeds_same_action_windows':sum(len({w['decisions'][f'{v}_seed_{s}']['action'] for s in (8201,8202,8203)})==1 for w in per_window)} for v in ('G1','G2')}
event_recalculation={}
for variant in ('G1','G2'):
    for seed in ('8201','8202','8203','ensemble'):
        key=variant+':'+seed
        event_by_parent={};head_by_parent={h:{} for h in heads}
        for row in rows:
            valid_event=[];local_heads={h:[] for h in heads}
            for candidate in row['candidate_rows']:
                prediction=candidate['predictions'][variant][seed]
                for field in ('event_probability','horizon_task_outcome_probability'):
                    assert all(math.isfinite(p) and 0<=p<=1 for p in prediction[field])
                    if seed=='ensemble':
                        for index,p in enumerate(prediction[field]):
                            avg=statistics.mean(candidate['predictions'][variant][s][field][index] for s in ('8201','8202','8203'))
                            assert math.isclose(avg,p,rel_tol=1e-6,abs_tol=1e-6)
                for index,valid in enumerate(candidate['event_valid']):
                    if valid:valid_event.append((prediction['event_probability'][index]-candidate['event_labels'][index])**2)
                for index,valid in enumerate(candidate['horizon_task_outcome_valid']):
                    if valid:local_heads[heads[index]].append((prediction['horizon_task_outcome_probability'][index]-candidate['horizon_task_outcome_target'][index])**2)
            if valid_event:event_by_parent[row['parent']]=statistics.mean(valid_event)
            for head,values in local_heads.items():
                if values:head_by_parent[head][row['parent']]=statistics.mean(values)
        event_value=statistics.mean(event_by_parent.values())
        horizon_values={h:statistics.mean(values.values()) if values else None for h,values in head_by_parent.items()}
        assert math.isclose(event_value,metrics['event_brier_parent_macro'][key],rel_tol=0,abs_tol=1e-12)
        for h,value in horizon_values.items():
            assert math.isclose(value,metrics['horizon_task_outcome_brier_by_head_parent_macro'][key][h],rel_tol=0,abs_tol=1e-12)
        event_recalculation[key]={'automatic_event_brier':event_value,'horizon_brier':horizon_values}
result={'status':'pass','method':'stdlib independent arithmetic; parent macro preserved; no new gate',
    'confirmation_windows':8,'confirmation_candidates':candidate_total,'model_calls':0,'environment_calls':0,
    'all_stored_choices_regret_top1_and_gates_reproduced':True,
    'metrics':macro,'parent_utility_errors':errors,'paired':paired,'seed_stability':seed_stability,
    'independent_event_and_horizon_brier':event_recalculation,
    'automatic_event_counts':event_counts,'horizon_task_outcome_counts':horizon_counts,'per_window':per_window,
    'transparent_noop_windows':sum(w['decisions']['transparent']['action']==24 for w in per_window),
    'trace_sha256':hashlib.sha256((RUN/'prediction-trace.jsonl').read_bytes()).hexdigest(),
    'limits_of_interpretation':['transparent is the frozen optimistic surrogate, not evaluated full Hungarian policy',
        '8 within-study held-out parents, not proven globally pristine','fixed Hungarian continuation is not GPPO policy value',
        'automatic event physical/host heads masked unknown; lifecycle horizon heads are separate valid supervision']}
(ROOT/'independent-metric-recalculation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'pass','metrics':{k:macro[k] for k in ('transparent','G1','G2')},'seed_stability':seed_stability,'events':event_counts,'horizon':horizon_counts},ensure_ascii=False,indent=2))
