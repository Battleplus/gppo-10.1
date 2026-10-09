"""Bottom environment fixture only. Uses no native environment constructor."""
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
import numpy as np


class Status(Enum):
    PENDING='pending'
    COMPLETED='completed'
    EXPIRED='expired'


@dataclass
class Communication:
    name: str='controlled-not-research'


class Environment:
    def __init__(self,spec,config,purpose):
        self.config=config; self.scenario=SimpleNamespace(seed=spec['scenario']['seed'])
        self.communication=Communication(); self._exogenous_key=spec['exogenous_key']
        self.n=spec['scenario'].get('task_count',3); self.noop=spec['scenario'].get('noop_only',False)
        self.equal=spec['scenario'].get('equal_outcomes',False)
        self.full_mask=spec['scenario'].get('full_mask',False)
        self.horizon=3 if purpose=='task_evaluation' else 18
        self.reset()

    def reset(self):
        self.clock=SimpleNamespace(time=0.,tasks={},resources={},log=[])
        self.good=False; self._last_completed=0; self._last_expired=0; self._completion_records={}
        for i in range(self.n):
            self.clock.tasks[f'task-{i}']=SimpleNamespace(state=Status.PENDING,deadline=18.,completed_at=None)
        for i in range(4):
            self.clock.resources[f'uav-{i}']=SimpleNamespace(energy=9.,alive=True,position=(0.,0.))
        return self._observation()

    def _observation(self):
        now=self.clock.time; terminal=now>=self.horizon
        def fields(values): return [x for v in values for x in (float(v),1.,1.,0.)]
        u=np.array([fields((0.,0.,r.energy,float(r.alive),1.,1.)) for r in self.clock.resources.values()],np.float32)
        tasks=np.zeros((6,32),np.float32)
        for i in range(self.n): tasks[i]=fields((float(i),0.,18.,1.,1.,float(not terminal),0.,float(i%4)))
        graph=np.zeros((21,32),np.float32); graph[:4,:24]=u; graph[11:17]=tasks
        relations=np.zeros((4,6,4),np.float32)
        mask=np.zeros(25,bool); mask[24]=True
        if not terminal and not self.noop: mask[0]=mask[1]=True
        if not terminal and self.full_mask: mask[:24]=True
        for i in range(self.n): relations[:,i,:]=[i/10.,1.,0.,0.]
        relations[0,0,2]=mask[0]; relations[0,1,2]=mask[1]
        result = {'uavs':u,'tasks':tasks,'graph':{'node_features':graph,'relations':relations},
            'mask':mask,'time':now,'version':int(now),'public_entity_ids':{'tasks':[f'task-{i}' for i in range(self.n)],
            'uavs':[f'uav-{i}' for i in range(4)]},'continuation_actions':[],'event_signal':0.,
            'event_signal_valid':True,'trigger_flags':{}}
        from public_prefix import capture_observation
        return capture_observation(result, source_object='controlled_environment._observation.result',
                                   random_key=self._exogenous_key)

    def step(self,action):
        if not self._observation()['mask'][action]: raise ValueError('FIXTURE_ILLEGAL_ACTION')
        if action==1: self.good=True
        self.clock.time+=1
        used=.01 if self.equal else .02 if action==1 else .06 if action==0 else .01
        self.clock.resources['uav-0'].energy-=used
        events=[]; done=self.clock.time>=self.horizon
        if done:
            for tid,t in self.clock.tasks.items():
                success=self.good and not self.equal
                t.state=Status.COMPLETED if success else Status.EXPIRED
                t.completed_at=self.clock.time if success else None
                if success:
                    events.append({'kind':'arrival','task':tid,'time':self.clock.time})
                    self._completion_records[tid]={'physical_arrival_before_deadline':True,'host_confirmation_time':None}
            self.clock.log+=events
        counts={'completed':sum(t.state==Status.COMPLETED for t in self.clock.tasks.values()),
                'expired':sum(t.state==Status.EXPIRED for t in self.clock.tasks.values())}
        self._last_completed=counts['completed']; self._last_expired=counts['expired']
        return self._observation(),0.,done,{'counts':counts,'energy':{k:r.energy for k,r in self.clock.resources.items()},
            'feedback':'noop' if action==24 else 'accepted','new_events':events,'communication_delta':[],
            'terminated':False,'truncated':done}


class Factory:
    def __call__(self,spec,config,purpose): return Environment(spec,config,purpose)
