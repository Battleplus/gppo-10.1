"""Deterministic public-only controller. No simulator or model imports."""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import heapq
import math

U_FIELDS = ('x', 'y', 'energy', 'alive', 'connected', 'idle')
T_FIELDS = ('x', 'y', 'deadline', 'remaining_service', 'priority', 'pending', 'region_id', 'target_id')
ALLOWED = frozenset(('flat', 'graph', 'uavs', 'tasks', 'mask', 'time', 'version',
                     'public_entity_ids', 'trigger_flags', 'event_signal', 'continuation_actions'))


class ContractError(RuntimeError):
    pass


def public_copy(obs):
    """Take only the native observation whitelist; never forward info/env."""
    def convert(x):
        if hasattr(x, 'tolist'):
            return convert(x.tolist())
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [convert(v) for v in x]
        if x is None or isinstance(x, (str, bool, int, float)):
            return x
        raise ContractError(f'Nonpublic object in observation: {type(x).__name__}')
    out = {k: convert(obs[k]) for k in ALLOWED if k in obs}
    validate(out)
    return out


def validate(obs):
    required = {'uavs', 'tasks', 'mask', 'time', 'public_entity_ids', 'continuation_actions'}
    if not required <= obs.keys() or not obs.keys() <= ALLOWED:
        raise ContractError('Public schema missing fields or contains private keys')
    if len(obs['mask']) != 25 or len(obs['uavs']) != 4 or len(obs['tasks']) != 6:
        raise ContractError('Not native 4x6+NOOP')
    ids = obs['public_entity_ids']
    if len(ids.get('uavs', [])) != 4 or len(set(ids['uavs'])) != 4:
        raise ContractError('UAV identities invalid')
    if len(ids.get('tasks', [])) > 6 or len(set(ids['tasks'])) != len(ids['tasks']):
        raise ContractError('Task identities invalid')
    if not math.isfinite(float(obs['time'])):
        raise ContractError('Nonfinite time')
    for name, width in [('uavs', 24), ('tasks', 32)]:
        for row in obs[name]:
            if len(row) != width or not all(math.isfinite(float(v)) for v in row):
                raise ContractError('Nonfinite/wrong width public row')
            for i in range(0, width, 4):
                if row[i+3] < -1e-6:
                    raise ContractError('Future public measurement')
    for a, legal in enumerate(obs['mask'][:24]):
        if legal and a % 6 >= len(ids['tasks']):
            raise ContractError('Legal undelivered task')
    for a in obs['continuation_actions']:
        if type(a) is not int or not 0 <= a < 24 or a % 6 >= len(ids['tasks']):
            raise ContractError('Invalid public continuation identity')


def field(row, index, now):
    v, known, valid, age = map(float, row[4*index:4*index+4])
    return {'value': v, 'known': known > .5, 'valid': valid > .5,
            'measured_at': now-age, 'age': age}


class PublicMemory:
    """Same instance type and state transitions for both experimental arms."""
    def __init__(self):
        self.fields = {}
        self.pending = []
        self.active = ()
        self.time = -math.inf
        self.history = []

    def observe(self, obs):
        validate(obs)
        now = float(obs['time'])
        if now < self.time or len(self.history) >= 19:
            raise ContractError('History reversed or longer than native horizon')
        self.time = now
        ids = obs['public_entity_ids']
        for key, names in [('uavs', U_FIELDS), ('tasks', T_FIELDS)]:
            for i, entity in enumerate(ids[key]):
                for j, name in enumerate(names):
                    item = field(obs[key][i], j, now)
                    if not item['known']:
                        continue
                    addr = (key, entity, name)
                    old = self.fields.get(addr)
                    if old is None or item['measured_at'] >= old['measured_at'] - 1e-6:
                        self.fields[addr] = item
        self.active = tuple((ids['uavs'][a//6], ids['tasks'][a%6]) for a in obs['continuation_actions'])
        keep = []
        for task, uav, submitted in self.pending:
            idle = self.fields.get(('uavs', uav, 'idle'))
            pending = self.fields.get(('tasks', task, 'pending'))
            fresh_receipts = (idle and pending and idle['measured_at'] > submitted + 1e-6
                              and pending['measured_at'] > submitted + 1e-6)
            if (uav, task) not in self.active and not fresh_receipts and now < submitted + 4:
                keep.append((task, uav, submitted))
        self.pending = keep
        self.history.append(copy.deepcopy(obs))

    def get(self, kind, entity, name):
        row = self.fields.get((kind, entity, name))
        if row is None:
            return None
        return {**row, 'age': max(0., self.time-row['measured_at'])}

    def candidates(self, obs):
        ids = obs['public_entity_ids']
        busy_u = {u for u, t in self.active} | {u for t, u, s in self.pending}
        busy_t = {t for u, t in self.active} | {t for t, u, s in self.pending}
        result = []
        for a, legal in enumerate(obs['mask']):
            if legal and (a == 24 or (ids['uavs'][a//6] not in busy_u and ids['tasks'][a%6] not in busy_t)):
                result.append(a)
        if not result:
            raise ContractError('No legal candidate; do not manufacture NOOP')
        return tuple(result)

    def submitted(self, obs, action):
        if action not in self.candidates(obs):
            raise ContractError('Submitted action not in current common candidates')
        if action != 24:
            ids = obs['public_entity_ids']
            self.pending.append((ids['tasks'][action%6], ids['uavs'][action//6], self.time))


@dataclass(frozen=True)
class Job:
    slot: int
    xy: tuple[float, float]
    deadline: float


@dataclass(frozen=True)
class Vehicle:
    xy: tuple[float, float]
    ready: float
    energy: float
    age: float
    available: bool = True


@dataclass(frozen=True)
class Node:
    g: float
    vehicles: tuple[Vehicle, ...]
    remaining: tuple[int, ...]
    completions: tuple[tuple[int, float], ...]
    travels: tuple[tuple[int, float, float], ...]
    sequence: tuple[tuple[int, float], ...]


def discount(event_time, now):
    return .99 ** max(0, math.ceil(event_time-now)-1)


def travel_cost(travels, now, horizon, energy_budgets):
    """Idle everywhere except predicted travel; disjoint intervals per UAV."""
    result = 0.; left=list(energy_budgets)
    for k in range(max(0, int(math.ceil(horizon-now)))):
        lo, hi = now+k, min(horizon, now+k+1)
        energy=0.
        for u in range(4):
            use=.05*(hi-lo)+sum(.30*max(0.,min(hi,end)-max(lo,start))
                                for uid,start,end in travels if uid==u)
            use=min(left[u],use); left[u]-=use; energy+=use
        result += .99**k * energy
    return result


class PublicPlanner:
    def __init__(self, *, node_cap=5000, horizon=18.):
        if node_cap < 25 or horizon <= 0:
            raise ContractError('Invalid fixed planner limits')
        self.node_cap, self.horizon = node_cap, horizon

    def initial(self, obs, memory):
        now = float(obs['time']); ids = obs['public_entity_ids']
        active = dict(memory.active)
        uncertain_uavs = {u for t, u, s in memory.pending}
        jobs = {}
        for j, entity in enumerate(ids['tasks']):
            row = obs['tasks'][j]
            if all(field(row, k, now)['valid'] for k in range(6)) and row[20] == 1 and entity not in active.values():
                jobs[j] = Job(j, (float(row[0]), float(row[4])), float(row[8]))
        vehicles, travels = [], []
        for u, entity in enumerate(ids['uavs']):
            f = {k: memory.get('uavs', entity, k) for k in U_FIELDS}
            ready = all(f[k] is not None for k in U_FIELDS)
            ready = ready and f['alive']['value'] == 1 and f['connected']['value'] == 1 and entity not in uncertain_uavs
            if not ready:
                vehicles.append(Vehicle((0.,0.), self.horizon, 0., 0., False)); continue
            xy=(f['x']['value'],f['y']['value']); age=max(f['x']['age'],f['y']['age'])
            energy=max(0., f['energy']['value']-.05*f['energy']['age'])
            v = Vehicle(xy, now, energy, age)
            if entity in active:
                target=active[entity]
                x,y = memory.get('tasks',target,'x'),memory.get('tasks',target,'y')
                if x is None or y is None:
                    v=replace(v, available=False)
                else:
                    dest=(x['value'],y['value']); duration=math.dist(xy,dest)+age
                    if .35*duration >= energy:
                        v=replace(v, available=False)
                    else:
                        travels.append((u,now,now+duration))
                        v=Vehicle(dest,now+duration,energy-.35*duration,0.)
            elif f['idle']['value'] != 1:
                v=replace(v,available=False)
            vehicles.append(v)
        # Active tasks add the same predicted completion constant to every root;
        # omitted from J because they cannot affect the selected root. Travel is retained.
        enabled=tuple(v.energy+sum(.35*(end-start) for uid,start,end in travels if uid==u)
                      if v.available else 0. for u,v in enumerate(vehicles))
        return jobs, enabled, Node(now,tuple(vehicles),tuple(sorted(jobs)),(),tuple(travels),())

    def child(self, node, action, jobs, *, root=False):
        if action == 24:
            if node.g >= self.horizon: return None
            return replace(node,g=node.g+1,sequence=node.sequence+((24,node.g),))
        u,j=divmod(action,6)
        if j not in node.remaining: return None
        v=node.vehicles[u]; job=jobs[j]
        dispatch=max(node.g,float(math.ceil(v.ready)))
        if not v.available or dispatch>=self.horizon or (root and dispatch>node.g): return None
        duration=math.dist(v.xy,job.xy)+v.age
        tau=dispatch+duration
        energy=v.energy-.05*max(0.,dispatch-v.ready)
        if tau>min(job.deadline,self.horizon) or energy<=.35*duration: return None
        vv=list(node.vehicles); vv[u]=Vehicle(job.xy,tau,energy-.35*duration,0.)
        return Node(dispatch+1,tuple(vv),tuple(jj for jj in node.remaining if jj!=j),
                    node.completions+((j,tau),),node.travels+((u,dispatch,tau),),node.sequence+((action,dispatch),))

    def score(self,node,jobs,now,enabled):
        completed=dict(node.completions)
        score=0.
        for j, job in jobs.items():
            if j in completed:
                score += .4/6*discount(completed[j],now)
            elif job.deadline<=self.horizon:
                score -= .4/6*discount(job.deadline,now)
        score -= .2/36*travel_cost(node.travels,now,self.horizon,enabled)
        if not math.isfinite(score): raise ContractError('Nonfinite planning score')
        return score

    def choose(self,obs,memory):
        candidates=memory.candidates(obs)
        if candidates==(24,):
            return 24, {'nodes':0,'truncated':False,'reason':'only_public_noop'}
        jobs,enabled,initial=self.initial(obs,memory); now=float(obs['time'])
        queue=[]; serial=0; best=None; nodes=0
        root_diagnostics=[]
        def offer(node):
            nonlocal serial,best
            score=self.score(node,jobs,now,enabled)
            key=node.sequence
            if best is None or score>best[0]+1e-12 or (abs(score-best[0])<=1e-12 and key<best[1]):
                best=(score,key,node)
            heapq.heappush(queue,(-score,key,serial,node)); serial+=1
        for a in candidates:
            node=self.child(initial,a,jobs,root=True)
            root_diagnostics.append({'action':a,'estimated_feasible':node is not None})
            if node is not None: offer(node)
        if not queue:
            raise ContractError('No finite root, including NOOP; cannot swap baseline')
        while queue and nodes<self.node_cap:
            _,_,_,node=heapq.heappop(queue); nodes+=1
            if node.g>=self.horizon or not node.remaining: continue
            # A node is also a complete plan that leaves all remaining jobs unserved.
            for a in [u*6+j for u in range(4) for j in node.remaining]+[24]:
                child=self.child(node,a,jobs)
                if child is not None: offer(child)
        action=best[1][0][0]
        if action not in candidates: raise ContractError('Planner escaped current permission')
        return action, {'nodes':nodes,'truncated':bool(queue),'score':best[0],
                        'sequence':best[1],'roots':root_diagnostics,'reason':'finite_public_plan'}
