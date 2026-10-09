"""Connect a received-state policy view to task execution without truth masks."""
from .task_policy_view import TaskPolicyView
from .task_execution import TaskExecution


class TaskDecisionBridge:
    def __init__(self, view: TaskPolicyView, execution: TaskExecution):
        self.view, self.execution = view, execution
        self._issued = None

    def observe(self):
        snapshot = self.view.snapshot(self.execution.clock.time)
        self.execution.observe_version(snapshot.version)
        self._issued = snapshot
        return snapshot

    def submit(self, action: int, *, version: int, command_id: str):
        """Return execution diagnostics to the transport, never directly to PPO.

        Receipt of this feedback must be separately timestamped by the simulator.
        An accepted proposal still requires a matching received execution ACK.
        """
        current = self.view.snapshot(self.execution.clock.time)
        self.execution.observe_version(current.version)
        if (self._issued is None or self._issued != current or
                type(version) is not int or version != current.version):
            return 'stale_snapshot'
        if type(action) is not int or not 0 <= action < len(current.mask):
            return 'invalid_action'
        if not current.mask[action]:
            return 'masked'
        pair = self.view.resolve(action)
        if pair is None:
            return 'noop'
        task_id, uid = pair
        return self.execution.propose(command_id, task_id, uid, version=version,
                                      proposal_allowed=True)
