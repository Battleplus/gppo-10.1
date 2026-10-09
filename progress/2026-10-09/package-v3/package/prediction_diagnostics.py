"""Masked head metrics, grouped by declared role and parent. No parameter fitting."""
import math
from collections import defaultdict
from statistics import mean
from consequence_contract import BINARY, CONTINUOUS, SEEDS


def evaluate_heads(rows):
    groups = defaultdict(lambda: defaultdict(list))
    unknown = defaultdict(int)
    for r in rows:
        for i, (prediction, label) in enumerate(zip(r['candidate_predictions'], r['labels'])):
            for seed_index, seed in enumerate((*SEEDS, 'ensemble')):
                values = (r['seed_readouts'][seed_index][i] if seed != 'ensemble' else None)
                key = f"{r['role']}:{r['parent']}:{seed}"
                for group, names in (('binary', BINARY), ('continuous', CONTINUOUS)):
                    for j, name in enumerate(names):
                        truth = label[group][name]
                        p = prediction[group][name]
                        metric = f'{group}.{name}'
                        if not truth['valid'] or not p['valid']:
                            unknown[key+':'+metric] += 1
                            continue
                        v = p['value'] if values is None else values[j+(len(BINARY) if group=='continuous' else 0)]
                        t = float(truth['value'])
                        error = (-t*math.log(max(1e-12,v))-(1-t)*math.log(max(1e-12,1-v))
                                 if group=='binary' else abs(v-t))
                        groups[key][metric].append(error)
                by_id = {t['task_id']:t for t in label['per_task']}
                for task in prediction['per_task']:
                    if task['task_id'] is None: continue
                    for j, name in enumerate(('completed_delta','expired_delta')):
                        t = by_id[task['task_id']][name]
                        if t['valid'] and task['valid'][j]:
                            v = task['delta'][j] if values is None else values[len(BINARY)+len(CONTINUOUS)+2*task['slot']+j]
                            groups[key]['per_task.'+name].append(abs(v-t['value']))
    return {'scope':'development_exploration_not_risk_certification',
            'groups':{k:{m:{'mean':mean(v),'valid_count':len(v),'metric':'BCE' if m.startswith('binary') else 'MAE'}
                         for m,v in metrics.items()} for k,metrics in groups.items()},
            'unknown_or_unsupported_counts':dict(unknown),
            'uncertainty':'ensemble dispersion only; not calibrated failure probability'}
