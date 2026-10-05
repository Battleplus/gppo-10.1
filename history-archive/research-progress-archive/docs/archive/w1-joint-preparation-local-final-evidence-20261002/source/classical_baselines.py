"""Public-only nearest/EDF/receding Hungarian and retained search controllers.

No model, environment, NumPy or SciPy import. Each call submits one native
action; a Hungarian assignment is only a plan, never a multi-command shortcut.
"""
from __future__ import annotations
import math
from public_controller import ContractError, PublicMemory, PublicPlanner, public_copy, field, discount, travel_cost

METHODS=('nearest','earliest_deadline','hungarian','search')

def normalized_preference(preference):
    if len(preference)!=2 or not all(math.isfinite(float(x)) and x>=0 for x in preference) or sum(preference)<=0:
        raise ContractError('Invalid preference')
    return tuple(float(x)/sum(preference) for x in preference)

def hungarian(cost):
    """Minimum-cost injection of n rows into m>=n columns; deterministic ties."""
    n=len(cost)
    if n==0:return []
    m=len(cost[0])
    if m<n or any(len(r)!=m or not all(math.isfinite(x) for x in r) for r in cost):
        raise ContractError('Hungarian requires a finite rectangular n<=m matrix')
    u=[0.]*(n+1);v=[0.]*(m+1);p=[0]*(m+1);way=[0]*(m+1)
    for i in range(1,n+1):
        p[0]=i;j0=0;minv=[math.inf]*(m+1);used=[False]*(m+1)
        while True:
            used[j0]=True;i0=p[j0];delta=math.inf;j1=0
            for j in range(1,m+1):
                if used[j]:continue
                cur=cost[i0-1][j-1]-u[i0]-v[j]
                if cur<minv[j]:minv[j]=cur;way[j]=j0
                if minv[j]<delta:delta=minv[j];j1=j
            for j in range(m+1):
                if used[j]:u[p[j]]+=delta;v[j]-=delta
                else:minv[j]-=delta
            j0=j1
            if p[j0]==0:break
        while True:
            j1=way[j0];p[j0]=p[j1];j0=j1
            if j0==0:break
    answer=[-1]*n
    for j in range(1,m+1):
        if p[j]:answer[p[j]-1]=j-1
    return answer

def public_edges(obs,candidates):
    edges={};now=float(obs['time'])
    for a in candidates:
        if a==24:continue
        u,t=divmod(a,6)
        uf=[field(obs['uavs'][u],k,now) for k in (0,1)]
        tf=[field(obs['tasks'][t],k,now) for k in (0,1,2)]
        if not all(f['known'] and f['valid'] for f in uf+tf):
            raise ContractError('Legal assignment lacks public valid position/deadline')
        distance=math.hypot(uf[0]['value']-tf[0]['value'],uf[1]['value']-tf[1]['value'])
        if not math.isfinite(distance):raise ContractError('Nonfinite public distance')
        edges[a]={'distance':distance,'deadline':tf[2]['value']}
    return edges

class PreferenceSearch(PublicPlanner):
    """Original search, exactly original score for p=(.8,.2)."""
    def __init__(self,preference):
        super().__init__(node_cap=5000,horizon=18.)
        self.preference=normalized_preference(preference)
    def score(self,node,jobs,now,enabled):
        if self.preference==(.8,.2):return super().score(node,jobs,now,enabled)
        task_weight=.5*self.preference[0];energy_weight=self.preference[1]
        completed=dict(node.completions);value=0.
        for j,job in jobs.items():
            if j in completed:value+=task_weight/6*discount(completed[j],now)
            elif job.deadline<=self.horizon:value-=task_weight/6*discount(job.deadline,now)
        value-=energy_weight/36*travel_cost(node.travels,now,self.horizon,enabled)
        if not math.isfinite(value):raise ContractError('Nonfinite search score')
        return value

class ClassicalSelector:
    def __init__(self,method,preference=(.8,.2)):
        if method not in METHODS:raise ValueError('Unknown baseline')
        self.method=method;self.preference=normalized_preference(preference)
        self.planner=PreferenceSearch(self.preference) if method=='search' else None
    def choose(self,obs,memory):
        candidates=memory.candidates(obs)
        if self.planner is not None:return self.planner.choose(obs,memory)
        edges=public_edges(obs,candidates)
        if not edges:
            if 24 not in candidates:raise ContractError('Cannot manufacture NOOP')
            return 24,{'method':self.method,'reason':'only_common_noop','matching':[]}
        if self.method=='nearest':
            action=min(edges,key=lambda a:(edges[a]['distance'],a));matching=[action]
        elif self.method=='earliest_deadline':
            action=min(edges,key=lambda a:(edges[a]['deadline'],edges[a]['distance'],a));matching=[action]
        else:
            # First maximize assignment cardinality, then minimize total distance.
            # Distances normalized to [0,1]; one dummy penalty 5 exceeds the
            # largest possible difference in four real-edge costs (<=4).
            scale=max(1.,max(e['distance'] for e in edges.values()))
            costs=[[edges[u*6+t]['distance']/scale if u*6+t in edges else 100. for t in range(6)]+[5.]*4 for u in range(4)]
            assignment=hungarian(costs)
            matching=sorted(u*6+t for u,t in enumerate(assignment) if t<6)
            if not matching or any(a not in edges for a in matching):raise ContractError('Invalid matching')
            # The executor accepts one new command per decision. Recompute all
            # matches next observation; never cache an unsubmitted assignment.
            action=min(matching,key=lambda a:(edges[a]['deadline'],edges[a]['distance'],a))
        if action not in candidates:raise ContractError('Baseline escaped shared filter')
        return action,{'method':self.method,'matching':matching,'selected_public_edge':edges[action]}

class PublicDecisionAdapter:
    """One shared boundary for every classical and future learned selector.

    Learned policies must sample AND train log-probabilities on `effective_mask`,
    not select from the raw mask and silently replace rejected actions.
    """
    def __init__(self):self.reset()
    def reset(self):self.memory=PublicMemory();self._pending_observation=None
    def prepare(self,observation):
        if self._pending_observation is not None:raise ContractError('Uncommitted decision')
        obs=public_copy(observation);self.memory.observe(obs)
        candidates=self.memory.candidates(obs)
        effective_mask=[a in candidates for a in range(25)]
        self._pending_observation=obs
        return obs,effective_mask
    def commit(self,action):
        obs=self._pending_observation
        if obs is None:raise ContractError('No prepared observation')
        if type(action) is not int or action not in self.memory.candidates(obs):raise ContractError('Action outside shared execution filter')
        self.memory.submitted(obs,action);self._pending_observation=None
        return action
    def decide(self,observation,selector):
        obs,effective_mask=self.prepare(observation)
        action,diagnostic=selector.choose(obs,self.memory)
        self.commit(action)
        return {'action':action,'submit_command':True,'effective_mask':effective_mask,'diagnostic':diagnostic}
