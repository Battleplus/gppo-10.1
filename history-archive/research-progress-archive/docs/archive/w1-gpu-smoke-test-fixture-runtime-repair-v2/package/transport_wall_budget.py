"""One wall deadline for this job's SSH operations; no process signalling."""
from contextlib import contextmanager
import math
import threading
import time


class WallDeadlineExceeded(TimeoutError):
    pass


class WallBudget:
    def __init__(self, limit, *, closing_reserve, started=None, clock=time.monotonic):
        if not math.isfinite(limit) or not 0 < closing_reserve < limit:
            raise ValueError('WALL_BUDGET_INVALID')
        self.clock=clock
        self.started=clock() if started is None else started
        self.limit=limit
        self.closing_reserve=closing_reserve
        self.expired=threading.Event()
        self.phases=[]
        self.client=None
        self.timer=None

    def remaining(self):
        value=self.started+self.limit-self.closing_reserve-self.clock()
        if self.expired.is_set() or value<=0:
            raise WallDeadlineExceeded('GLOBAL_WALL_DEADLINE_REACHED')
        return value

    def timeout(self, requested):
        return min(float(requested),self.remaining())

    def bind(self, client):
        self.client=client
        self.timer=threading.Timer(self.remaining(),self.expire)
        self.timer.daemon=True
        self.timer.start()
        return DeadlineClient(client,self)

    def expire(self):
        self.expired.set()
        if self.client is not None:
            # Only this task's connection: do not signal server/user processes.
            self.client.close()

    def close(self):
        if self.timer is not None:self.timer.cancel()

    @contextmanager
    def phase(self, name):
        self.remaining()
        started=self.clock()
        outcome='error'
        try:
            yield
            self.remaining()
            outcome='complete'
        finally:
            self.phases.append({'phase':name,'wall_seconds':self.clock()-started,
                                'elapsed_wall_seconds':self.clock()-self.started,
                                'outcome':outcome})

    def remote_allowance(self, server_limit, *, remote_closing_reserve):
        allowance=min(server_limit,self.remaining()-remote_closing_reserve)
        if allowance<=remote_closing_reserve:
            raise WallDeadlineExceeded('INSUFFICIENT_WALL_FOR_REMOTE_RUN_AND_CLOSING')
        return allowance


class DeadlineClient:
    def __init__(self,client,budget):self.client,self.budget=client,budget

    def exec_command(self,command,**kwargs):
        kwargs['timeout']=self.budget.timeout(kwargs.get('timeout',180))
        result=self.client.exec_command(command,**kwargs)
        self.budget.remaining()
        return result

    def open_sftp(self):
        self.budget.remaining()
        sftp=self.client.open_sftp()
        self.budget.remaining()
        return DeadlineSFTP(sftp,self.budget)

    def close(self):self.client.close()


class DeadlineSFTP:
    def __init__(self,sftp,budget):self.sftp,self.budget=sftp,budget
    def __enter__(self):return self
    def __exit__(self,*args):self.sftp.close()
    def __getattr__(self,name):
        value=getattr(self.sftp,name)
        if not callable(value):return value
        def guarded(*args,**kwargs):
            self.sftp.get_channel().settimeout(self.budget.remaining())
            result=value(*args,**kwargs)
            self.budget.remaining()
            return result
        return guarded
