"""Node-token graph representation and frozen action-conditioned dynamics.

Model forward APIs contain only observed graphs, past executed actions, current
evidence and candidate action endpoints. Future graphs are external targets.
"""
from __future__ import annotations
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from .registry import FEATURE_REGISTRY
from .data import NODE_ORDER, RELATION_ORDER

COUNTS = {'uav':4, 'region':4, 'target':3}
OFFSETS = {'uav':0, 'region':4, 'target':8}


class NodeEncoder(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.width = width
        self.inputs = nn.ModuleDict({k: nn.Sequential(nn.Linear(d,width),nn.LayerNorm(width),nn.SiLU())
                                    for k,d in FEATURE_REGISTRY.node_dimensions.items()})
        self.messages = nn.ModuleDict({'__'.join(r): nn.Linear(width+d,width)
                                      for r,d in FEATURE_REGISTRY.edge_dimensions.items()})
        self.updates = nn.ModuleDict({k: nn.Sequential(nn.Linear(2*width,width),nn.LayerNorm(width),nn.SiLU()) for k in NODE_ORDER})

    def forward(self, graph):
        hidden = {k:self.inputs[k](graph['nodes'][k]) for k in NODE_ORDER}
        totals = {k:torch.zeros_like(x) for k,x in hidden.items()}
        counts = {k:x.new_zeros((*x.shape[:2],1)) for k,x in hidden.items()}
        for r in RELATION_ORDER:
            source,_,dest = r
            src,dst = graph['edge_index'][r][:,0],graph['edge_index'][r][:,1]
            selected = hidden[source].gather(1,src.unsqueeze(-1).expand(-1,-1,self.width))
            msg = F.silu(self.messages['__'.join(r)](torch.cat([selected,graph['edge_attr'][r]],-1)))
            totals[dest].scatter_add_(1,dst.unsqueeze(-1).expand_as(msg),msg)
            counts[dest].scatter_add_(1,dst.unsqueeze(-1),msg.new_ones((*dst.shape,1)))
        return torch.cat([self.updates[k](torch.cat([hidden[k],totals[k]/counts[k].clamp_min(1)],-1)) for k in NODE_ORDER],1)


def masked_graph(graph, generator):
    """Mask a whole semantic node type, removing all duplicate dynamic fields.

    Masked type features are all removed. Other types lose every event-eligible
    feature (assignment, task, load, tracking, location, etc.). All relation
    attributes are removed in both directions. Only static topology/identity
    and unmasked static fields remain. This conservative mask sacrifices context
    rather than permit a one-hot or reverse-edge shortcut.
    """
    result={part:{k:v.clone() for k,v in values.items()} for part,values in graph.items()}
    batch=next(iter(graph['nodes'].values())).shape[0]
    selected=torch.randint(3,(batch,),generator=generator).to(graph['nodes']['uav'].device)
    token_mask=torch.zeros(batch,11,dtype=torch.bool,device=selected.device)
    for index,k in enumerate(NODE_ORDER):
        eligible=[i for i,f in enumerate(FEATURE_REGISTRY.nodes[k]) if f.event_eligible]
        result['nodes'][k][:,:,eligible]=0
        choose=selected==index
        result['nodes'][k][choose]=0
        token_mask[choose,OFFSETS[k]:OFFSETS[k]+COUNTS[k]]=True
    for value in result['edge_attr'].values():value.zero_()
    return result,token_mask


def spread_penalty(tokens):
    # Variance across samples for the same node identity, not across node types.
    return F.relu(0.1-tokens.var(0,unbiased=False).add(1e-4).sqrt()).mean()


class MaskedPretrainer(nn.Module):
    def __init__(self,width=64):
        super().__init__()
        self.encoder=NodeEncoder(width)
        self.target=deepcopy(self.encoder).requires_grad_(False)
        self.predictor=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))

    def loss(self,graph,generator):
        masked,mask=masked_graph(graph,generator)
        tokens=self.encoder(masked)
        with torch.no_grad():target=self.target(graph)
        return F.mse_loss(self.predictor(tokens)[mask],target[mask])+spread_penalty(self.encoder(graph))

    @torch.no_grad()
    def update_target(self):
        for p,q in zip(self.target.parameters(),self.encoder.parameters()):p.lerp_(q,.01)


class ActionDynamics(nn.Module):
    def __init__(self,width=64,with_action=True):
        super().__init__()
        self.width=width
        self.with_action=with_action
        self.history=nn.GRUCell(width+18+1,width)
        # Two endpoint tokens, endpoint edge attributes, NOOP bit.
        self.action=nn.Sequential(nn.Linear(2*width+6,width),nn.SiLU())
        self.update=nn.Sequential(nn.Linear(4*width+14,width),nn.LayerNorm(width),nn.SiLU(),nn.Linear(width,width))

    def forward(self,tokens,valid,past_actions,past_gaps,candidate_edges,edge_attributes,action,evidence):
        batch,steps,nodes,width=tokens.shape
        history=tokens.new_zeros(batch,width)
        for t in range(steps):
            embedded=F.one_hot(past_actions[:,t].clamp_min(0),18).to(tokens.dtype)
            embedded=embedded*(past_actions[:,t]>=0).unsqueeze(-1)
            inp=torch.cat([tokens[:,t].mean(1),embedded,past_gaps[:,t,None]],-1)
            updated=self.history(inp,history)
            history=torch.where(valid[:,t,None],updated,history)
        current=tokens[:,-1]
        if self.with_action:
            noop=action==16
            index=action.clamp_max(15)
            batch_index=torch.arange(batch,device=current.device)
            endpoints=candidate_edges[batch_index,index]
            u=current[batch_index,endpoints[:,0]]
            r=current[batch_index,endpoints[:,1]+4]
            edge=edge_attributes[batch_index,index]
            payload=torch.cat([u,r,edge],-1).masked_fill(noop[:,None],0)
            conditioning=self.action(torch.cat([payload,noop[:,None].to(tokens.dtype)],-1))
        else:
            # Do not even gather current endpoints in this arm.
            conditioning=tokens.new_zeros(batch,width)
        pooled=current.mean(1)
        context=torch.cat([history,conditioning,pooled,evidence],-1)
        return current+self.update(torch.cat([current,context[:,None,:].expand(-1,nodes,-1)],-1))


class TwoStageModel(nn.Module):
    def __init__(self,width=64,with_action=True,frozen=False):
        super().__init__()
        self.encoder=NodeEncoder(width)
        self.dynamics=ActionDynamics(width,with_action)
        self.frozen=frozen
        if frozen:self.encoder.requires_grad_(False)

    def forward(self,graphs,valid,past_actions,past_gaps,candidate_edges,edge_attributes,action,evidence):
        tokens=torch.stack([self.encoder(g) for g in graphs],1)
        return self.dynamics(tokens,valid,past_actions,past_gaps,candidate_edges,edge_attributes,action,evidence)
