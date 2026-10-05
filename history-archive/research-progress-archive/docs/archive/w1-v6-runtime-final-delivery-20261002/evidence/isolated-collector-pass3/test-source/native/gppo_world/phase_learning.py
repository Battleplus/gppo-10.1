"""Tensor preparation for factual commit targets and legal branch evaluation."""
import json
from pathlib import Path
import torch
from .data import graph_from_dict,state_vector,COST_NAMES
from .jepa import batch_graphs,select_batch
from .model import evidence_features
from .phase_data import model_input,history_inputs,validate_record


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def prepare_phases(rows,factual_rows=None):
    """Branch rows borrow only the matching origin's factual past inputs."""
    originals=rows if factual_rows is None else factual_rows
    hs=history_inputs(originals)
    history={(h['episode_id'],h['step']):h for h in hs}
    graphs=[[] for _ in range(4)];valid=[];past=[];gaps=[];edges=[];attributes=[];actions=[];evidence=[];targets=[];commits=[];raw=[]
    for row in rows:
        validate_record(row)
        inp=model_input(row)
        h=history[row['episode_id'],row['step']]
        if h['current']['graph']!=inp['graph']:raise ValueError('Branch changed factual pre graph')
        current=graph_from_dict(inp['graph'])
        times=[f['time'] for f in h['history'] if f is not None]
        ds=[0.]+[b-a for a,b in zip(times,times[1:])]
        ds=[0.]*(4-len(ds))+ds
        if any(d<0 for d in ds):raise ValueError('History time reversal')
        pa=[]
        for t,frame in enumerate(h['history']):
            graphs[t].append(current if frame is None else graph_from_dict(frame['graph']))
            pa.append(-1 if frame is None or frame['past_executed_action'] is None else frame['past_executed_action'])
        valid.append(h['valid']);past.append(pa);gaps.append([min(d,240.)/240. for d in ds])
        edges.append(current.candidate_edges)
        attributes.append(current.edge_attr['uav','can_serve','region'])
        actions.append(inp['action'])
        age=sum(max(inp['time']-e['received_at'],0.) for e in inp['evidence'])/max(len(inp['evidence']),1)
        ev=torch.cat([evidence_features(inp['evidence']),torch.tensor([min(age,240.)/240.,inp['decision_duration']/240.])])
        evidence.append(ev)
        commit=graph_from_dict(row['commit']['graph']);commits.append(commit)
        targets.append(torch.cat([state_vector(commit)-state_vector(current),torch.tensor([row['reward']]),
            torch.tensor([row['costs'][name] for name in COST_NAMES])]))
        raw.append(torch.cat([state_vector(current),torch.nn.functional.one_hot(torch.tensor(inp['action']),17).float(),ev]))
    return {'graphs':[batch_graphs(g) for g in graphs], 'valid':torch.tensor(valid), 'past_actions':torch.tensor(past),
        'past_gaps':torch.tensor(gaps),'candidate_edges':torch.stack(edges),'edge_attributes':torch.stack(attributes),
        'action':torch.tensor(actions),'evidence':torch.stack(evidence),'commit':batch_graphs(commits),
        'y':torch.stack(targets),'raw':torch.stack(raw),'rows':rows}


def model_arguments(data,indices):
    return {'graphs':[select_batch(g,indices) for g in data['graphs']],
        **{k:data[k][indices] for k in ('valid','past_actions','past_gaps','candidate_edges','edge_attributes','action','evidence')}}
