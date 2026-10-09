"""Independent episode/matrix/utility verification; no simulator or torch imports."""
import json
import math
from pathlib import Path
from statistics import mean
from collections import defaultdict


def recompute(output, matrix):
    output=Path(output)
    rows=[json.loads(line) for line in (output/'task-results.jsonl').read_text().splitlines()]
    by_key={(r['parent'],r['arm'],r['seed']):r for r in rows}
    if len(by_key)!=len(rows) or len(rows)!=matrix['task_episodes']: raise ValueError('TASK_RESULT_MATRIX')
    parents=sorted({r['parent'] for r in rows})
    expected={(p,a,s) for p in parents for a in ('G0','T','G1','H') for s in ((None,) if a=='H' else (8301,8302,8303))}
    if set(by_key)!=expected or len(parents)!=24: raise ValueError('TASK_PAIRED_MATRIX')
    for r in rows:
        actual=sum(.99**i*(.4*v[0]+.2*v[1]) for i,v in enumerate(r['vector_rewards']))
        if not math.isfinite(actual) or abs(actual-r['utility'])>1e-7: raise ValueError('UTILITY_RECOMPUTE')
        if len(r['vector_rewards'])!=r['steps'] or not (r['terminated'] or r['truncated']): raise ValueError('TASK_TIMELINE')
        if len(r['reward_evidence'])!=r['steps']: raise ValueError('REWARD_EVIDENCE_MISSING')
        for reward,e in zip(r['vector_rewards'],r['reward_evidence']):
            c,b=e['counts_after'],e['counts_before']
            task=(max(0,c['completed']-b['completed'])-max(0,c['expired']-b['expired']))/6
            energy=-max(0,e['energy_before']-e['energy_after'])/36
            if max(abs(reward[0]-task),abs(reward[1]-energy))>1e-7: raise ValueError('STEP_REWARD_RECOMPUTE')
        # Rewards are checked against final physical counts and energy, independently of reported utility.
        if abs(sum(v[0] for v in r['vector_rewards'])-(r['completed']-r['expired'])/6)>1e-6: raise ValueError('PHYSICAL_REWARD_RECOMPUTE')
        if abs(sum(v[1] for v in r['vector_rewards'])+r['energy_used']/36)>1e-6: raise ValueError('ENERGY_REWARD_RECOMPUTE')
    comparisons={}
    for comparator in ('G0','T','H'):
        pairs=[{'parent':p,'seed':s,'delta':by_key[p,'G1',s]['utility']-by_key[p,comparator,None if comparator=='H' else s]['utility']}
               for p in parents for s in (8301,8302,8303)]
        grouped={p:mean(r['delta'] for r in pairs if r['parent']==p) for p in parents}
        worst=sorted(grouped,key=lambda p:(grouped[p],p))
        comparisons[comparator]={'mean_delta':mean(grouped.values()),'parents':grouped,
            'seeds':{s:mean(r['delta'] for r in pairs if r['seed']==s) for s in (8301,8302,8303)},
            'major_loss_count':sum(r['delta']<=-.1 for r in pairs),'paired_results':pairs,
            'worst_parent':worst[0],'worst_parent_delta':grouped[worst[0]],
            'worst_two_mean_delta':mean(grouped[p] for p in worst[:2]),
            'mean_benefit_pass':utility_improvement_pass(mean(grouped.values()),comparator)}
    arms={}
    for arm in ('G0','T','G1','H'):
        ar=[r for r in rows if r['arm']==arm]; costs=[c for r in ar for c in r['decision_costs']]
        walls=sorted(c['wall_seconds']*1000 for c in costs); cpu=mean(c['cpu_seconds']*1000 for c in costs)
        p95=walls[math.ceil(.95*len(walls))-1]
        arms[arm]={'episodes':len(ar),'decisions':len(costs),'cpu_mean_ms':cpu,'wall_p95_ms':p95,
            'practical_cost_pass':cpu<=10 and p95<=50,
            'triggered_episodes':sum(r['world_triggered_decisions']>0 for r in ar),
            'triggered_decisions':sum(r['world_triggered_decisions'] for r in ar),
            'world_batch_forwards':sum(c['world_forwards'] for c in costs),
            **{f:mean(r[f] for r in ar) for f in ('utility','completed','expired','damaged','energy_used','host_confirmed')},
            'unknown_count':sum(r['unknown_count'] for r in ar),
            'by_seed':{str(s):{f:mean(r[f] for r in ar if r['seed']==s) for f in
                ('utility','completed','expired','damaged','energy_used','host_confirmed')}
                for s in sorted({r['seed'] for r in ar},key=str)},
            'collision':'unknown_simulator_not_implemented'}
    return {'status':'controlled_synthetic_only' if matrix.get('controlled') else 'completed_development_exploration',
        'execution_scope':'controlled_synthetic_only' if matrix.get('controlled') else 'development_exploration','comparisons':comparisons,'arms':arms,
        'parents':parents,'risk_certification':False,'global_independence':'unproven_development_only',
        'support_times':[2.,4.,8.,12.], 'counterfactual_teacher_utility_is_not_on_policy_return':True}


def utility_improvement_pass(delta, comparator):
    if comparator=='H': return delta>=.01
    if comparator in ('G0','T'): return delta>0.
    raise ValueError('UNKNOWN_COMPARATOR')


def report(output, status):
    output=Path(output)
    summary=json.loads((output/'result-summary.json').read_text()) if (output/'result-summary.json').exists() else None
    text=['# 研究结果','',f'状态：`{status}`。','',
          '本方案是多头后果监督、排序训练与 critic 接入的组合实验。旧负结果与停止结论不变；不证明 MSE 是旧失败根因。',
          '碰撞为 unknown；主机确认不等于物理完成。结果限定开发探索，不作 5% 风险认证。','']
    if summary:
        for arm,v in summary['comparisons'].items():
            text.append(f"G1−{arm} 平均效用 {v['mean_delta']:.6f}；重大负增量 {v['major_loss_count']} 次；最差父场景 {v['worst_parent']} ({v['worst_parent_delta']:.6f})；最差两个均值 {v['worst_two_mean_delta']:.6f}。")
        for arm,v in summary['arms'].items():
            text.append(f"{arm}：CPU 均值 {v['cpu_mean_ms']:.3f}ms，wall p95 {v['wall_p95_ms']:.3f}ms；成本门 {v['practical_cost_pass']}。")
    else: text.append('GPPO 任务收益、Hungarian 比较与完整在线成本：未评价。受控夹具结果不得替代研究效果。')
    text += ['','GitHub 归档独立处理。技术错误保留首错、pending 和可测资源下界；不自动重试。']
    (output/'研究报告.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
