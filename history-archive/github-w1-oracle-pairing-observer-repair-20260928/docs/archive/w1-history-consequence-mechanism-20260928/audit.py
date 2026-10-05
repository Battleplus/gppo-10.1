"""Offline descriptive audit; stdlib + frozen public-only memory, no simulator/model."""
import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = Path('E:/Z博士/runs/w1-light-repaired-fair-rerun-v2-nativefs-once')
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))
from public_controller import PublicMemory, public_copy, field, U_FIELDS

def main():
    path = SOURCE / 'run-once/evaluation/steps.jsonl'
    groups = collections.defaultdict(list)
    for line, raw in enumerate(path.open(encoding='utf-8'), 1):
        r = json.loads(raw)
        groups[json.dumps(r['key'], sort_keys=True)].append((line, r))
    stats = collections.defaultdict(collections.Counter)
    evidence = []
    for records in groups.values():
        memory = PublicMemory()
        past = []
        for line, r in records:
            obs = public_copy(r['observation'])
            memory.observe(obs)
            candidates = memory.candidates(obs)
            a = r['action']; method = r['key'][2]
            assert a in candidates
            s = stats[method]
            s['decisions'] += 1
            s['multi_assignment'] += sum(x != 24 for x in candidates) >= 2
            s['feedback:' + r['feedback']] += 1
            if a != 24:
                u, t = divmod(a, 6)
                fields = {name: field(obs['uavs'][u], i, obs['time']) for i, name in enumerate(U_FIELDS)}
                ages = [v['age'] for v in fields.values() if v['known']]
                s['assignment'] += 1
                s['selected_age_gt_1'] += max(ages, default=0) > 1.000001
                s['selected_age_gt_0'] += max(ages, default=0) > 0.000001
                s['xy_different_measurement_times'] += abs(fields['x']['measured_at'] - fields['y']['measured_at']) > 1e-6
                s['assignment_feedback:' + r['feedback']] += 1
                # Always emit every chosen assignment: no outcome-based row selection.
                evidence.append({'source_line': line, 'parent': r['key'][0], 'method': method,
                    'step': r['step'], 'time': obs['time'], 'action': a,
                    'uav': obs['public_entity_ids']['uavs'][u], 'task': obs['public_entity_ids']['tasks'][t],
                    'effective_candidates': list(candidates), 'uav_public_fields': fields,
                    'task_deadline': field(obs['tasks'][t], 2, obs['time']),
                    'public_active': list(memory.active), 'public_pending_before': list(memory.pending),
                    'prior_action_lines': past[-3:],
                    'feedback_after_action_label_only': r['feedback'],
                    'command_submitted_label_only': r['command_submitted']})
            memory.submitted(obs, a)
            past.append({'source_line': line, 'time': obs['time'], 'action': a})
    summary = {'episodes': len(groups), 'decisions': sum(s['decisions'] for s in stats.values()),
        'selected_assignment_rows': len(evidence), 'by_method': dict(stats),
        'interpretation': 'ages and feedback co-occurrence only; not causal or counterfactual evidence',
        'history_input_boundary': 'only prior observations and own submitted actions; current feedback and communication_delta excluded',
        'environment_calls': 0, 'model_calls': 0, 'training_updates': 0}
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (ROOT / 'decision-evidence.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in evidence), encoding='utf-8')
    inputs = [path] + [SOURCE / x for x in ['public_controller.py', 'classical_baselines.py', 'rl_adapters.py', 'runner.py', 'environment.json']]
    (ROOT / 'input-index.json').write_text(json.dumps([{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs], indent=2)+'\n', encoding='utf-8')
    assert len(groups) == 56
    assert summary['decisions'] == 774
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('Verified: 56 episodes; 774 decisions; all chosen actions within reconstructed public candidates.')

if __name__ == '__main__':
    main()
