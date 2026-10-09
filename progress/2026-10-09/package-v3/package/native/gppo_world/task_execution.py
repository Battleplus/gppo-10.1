"""Task-level command/ACK/lease bridge for the M-10 physical clock.

Execution truth and diagnostic records in this object must not be policy input.
The environment supplies observation versions and a received-state proposal mask.
Unlike the legacy region lease, each lease here owns one service task.
"""
from dataclasses import dataclass
import math

from .service_clock import ServiceClock
from .task_lifecycle import TaskState


@dataclass(frozen=True)
class TaskCommand:
    command_id: str
    task_id: str
    uav_id: str
    version: int
    token: int
    expires_at: float


class TaskExecution:
    def __init__(self, clock: ServiceClock, *, command_ttl: float, lease_ttl: float):
        if not all(math.isfinite(v) and v > 0 for v in (command_ttl, lease_ttl)):
            raise ValueError("Finite positive command and lease TTL required")
        self.clock = clock
        self.command_ttl, self.lease_ttl = command_ttl, lease_ttl
        self.version = 0
        self.commands = {}
        self.status = {}
        self.leases = {}  # command id -> expiry; accepted ownership only
        self.task_tokens = {}
        self._next_token = 0
        self.log = []

    def observe_version(self, version: int):
        """Called on received observation changes, never hidden truth changes."""
        if type(version) is not int or version < self.version:
            raise ValueError("Observation version must be an increasing integer")
        self.version = version

    def _available(self, task_id, uid):
        task = self.clock.tasks.get(task_id)
        resource = self.clock.resources.get(uid)
        if task is None or resource is None:
            return "unknown_identity"
        if task.state != TaskState.PENDING:
            return "task_unavailable"
        if not resource.alive or not resource.connected:
            return "resource_unavailable"
        if resource.energy <= 0:
            return "energy"
        if any(t.assigned_uav == uid for t in self.clock.tasks.values()):
            return "resource_busy"
        return None

    def _record(self, command_id, reason):
        self.log.append({"command_id": command_id, "time": self.clock.time, "result": reason})
        return reason

    def propose(self, command_id: str, task_id: str, uid: str, *, version: int,
                proposal_allowed: bool):
        if not command_id or command_id in self.status:
            return self._record(command_id, "duplicate_or_empty_id")
        self.status[command_id] = "rejected"
        if type(version) is not int or version != self.version:
            return self._record(command_id, "stale")
        if proposal_allowed is not True:
            return self._record(command_id, "masked")
        reason = self._available(task_id, uid)
        if reason:
            return self._record(command_id, reason)
        self._next_token += 1
        command = TaskCommand(command_id, task_id, uid, version, self._next_token,
                              self.clock.time + self.command_ttl)
        self.commands[command_id] = command
        self.status[command_id] = "awaiting_ack"
        self._record(command_id, "awaiting_ack")
        return command

    def acknowledge(self, command_id: str, uid: str, token: int):
        """Only the execution transport may call this after a received ACK."""
        command = self.commands.get(command_id)
        if command is None or self.status[command_id] != "awaiting_ack":
            return self._record(command_id, "inactive_command")
        if uid != command.uav_id or type(token) is not int or token != command.token:
            return self._record(command_id, "ack_identity")
        reason = None
        if self.clock.time >= command.expires_at:
            reason = "ack_timeout"
        elif command.version != self.version:
            reason = "stale"
        elif token <= self.task_tokens.get(command.task_id, 0):
            reason = "fenced"
        else:
            reason = self._available(command.task_id, uid)
        if reason:
            self.status[command_id] = "rejected"
            return self._record(command_id, reason)
        self.clock.assign(command.task_id, uid)
        self.task_tokens[command.task_id] = token
        self.leases[command_id] = self.clock.time + self.lease_ttl
        self.status[command_id] = "executing"
        return self._record(command_id, "accepted")

    def _cleanup(self):
        for cid, expiry in list(self.leases.items()):
            command = self.commands[cid]
            task = self.clock.tasks[command.task_id]
            if task.assigned_uav != command.uav_id:
                self.status[cid] = "completed" if task.state == TaskState.COMPLETED else "revoked"
                del self.leases[cid]
            elif self.clock.time >= expiry:
                task.interrupt(self.clock.time)
                self.status[cid] = "lease_expired"
                del self.leases[cid]
                self._record(cid, "lease_expired")

    def advance(self, end: float):
        if not math.isfinite(end) or end < self.clock.time:
            raise ValueError("Execution time must be finite and monotonic")
        self._cleanup()
        while self.clock.time < end:
            boundary = min([end, *self.leases.values()])
            self.clock.advance(boundary)
            self._cleanup()

    def renew(self, command_id: str, uid: str, token: int):
        self._cleanup()
        command = self.commands.get(command_id)
        if command is None or command_id not in self.leases:
            return self._record(command_id, "inactive_lease")
        if uid != command.uav_id or token != command.token or type(token) is not int:
            return self._record(command_id, "ack_identity")
        resource = self.clock.resources[uid]
        if not resource.alive or not resource.connected or resource.energy <= 0:
            return self._record(command_id, "resource_unavailable")
        self.leases[command_id] = self.clock.time + self.lease_ttl
        return self._record(command_id, "renewed")
