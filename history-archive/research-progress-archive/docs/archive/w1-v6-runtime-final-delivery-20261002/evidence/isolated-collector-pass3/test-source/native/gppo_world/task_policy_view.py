"""Fixed-capacity policy input built exclusively from delivered measurements.

No simulator, truth event schedule or execution object is accepted by this API.
Slot IDs are assigned in delivery order, never in future task/scheduler order.
The mask is a conservative proposal filter, not an execution authorization.
"""
from dataclasses import dataclass
import math

from .telemetry import ReceivedTelemetry, Telemetry


UAV_FIELDS = ('x', 'y', 'energy', 'alive', 'connected', 'idle')
TASK_FIELDS = ('x', 'y', 'deadline', 'remaining_service', 'priority', 'pending', 'region_id', 'target_id')
TASK_REQUIRED_FIELDS = ('x', 'y', 'deadline', 'remaining_service', 'priority', 'pending')
BOOLEAN_FIELDS = frozenset(('alive', 'connected', 'idle', 'pending'))


@dataclass(frozen=True)
class TaskPolicySnapshot:
    time: float
    version: int
    uavs: tuple[tuple[float, ...], ...]
    tasks: tuple[tuple[float, ...], ...]
    mask: tuple[bool, ...]


class TaskPolicyView:
    def __init__(self, uav_ids: tuple[str, ...], *, task_capacity: int, max_age: float):
        if (not uav_ids or len(set(uav_ids)) != len(uav_ids) or
                any(not isinstance(uid, str) or not uid for uid in uav_ids)):
            raise ValueError('Unique nonempty public fleet IDs required')
        if type(task_capacity) is not int or task_capacity <= 0:
            raise ValueError('Positive public task capacity required')
        if not math.isfinite(max_age) or max_age < 0:
            raise ValueError('Finite nonnegative maximum age required')
        self.uav_ids = tuple(uav_ids)
        self.task_capacity, self.max_age = task_capacity, max_age
        self._tasks = []
        self._uavs = ReceivedTelemetry()
        self._task_values = ReceivedTelemetry()
        self._completed_tasks = set()
        self._time = 0.0
        self.version = 0

    @property
    def public_task_ids(self) -> tuple[str, ...]:
        """Stable identities whose task telemetry has been legally delivered.

        The returned ordering is the existing public slot ordering.  It does
        not reveal scheduled-but-undelivered tasks and is intended for
        identity-aligned auxiliary labels, not as a replacement for features.
        """
        return tuple(self._tasks)

    def _advance(self, now):
        if not math.isfinite(now) or now < self._time:
            raise ValueError('Policy time must be monotonic')
        if now != self._time:
            # Age/deadline changes affect visible features and the mask too.
            self.version += 1
            self._time = now

    def receive(self, kind: str, message: Telemetry, now: float):
        fields = UAV_FIELDS if kind == 'uav' else TASK_FIELDS if kind == 'task' else ()
        if message.field not in fields:
            raise ValueError('Field is not in the policy whitelist')
        if message.field in BOOLEAN_FIELDS and message.value not in (0, 1):
            raise ValueError('Boolean measurement must be 0 or 1')
        if message.field in ('energy', 'remaining_service', 'deadline', 'priority') and message.value < 0:
            raise ValueError('Nonnegative measurement required')
        if kind == 'uav' and message.entity not in self.uav_ids:
            raise ValueError('Unknown public fleet member')
        if kind == 'task' and message.entity not in self._tasks and len(self._tasks) >= self.task_capacity:
            raise ValueError('Delivered tasks exceed frozen capacity')
        if message.received_at > now:
            raise ValueError('Future delivery rejected before slot allocation')
        self._advance(now)
        store = self._uavs if kind == 'uav' else self._task_values
        accepted = store.ingest(message, now)
        if accepted:
            if kind == 'task' and message.entity not in self._tasks:
                self._tasks.append(message.entity)
            self.version += 1
        return accepted

    def mark_completed(self, task_id: str, now: float) -> None:
        """Record a legally received completion without exposing simulator truth."""
        if task_id not in self._tasks:
            raise ValueError('Completion for an unknown public task')
        self._advance(now)
        if task_id not in self._completed_tasks:
            self._completed_tasks.add(task_id)
            self.version += 1

    def snapshot(self, now: float) -> TaskPolicySnapshot:
        self._advance(now)

        def read_row(store, entity, fields):
            values = [store.read(entity, field, now, self.max_age) for field in fields]
            # Four channels per field: received value, known, fresh, age.
            row = tuple(v for item in values for v in
                        (item['value'], float(item['known']), float(item['valid']), item['age']))
            return row, {f: v for f, v in zip(fields, values)}

        ur, uv = zip(*(read_row(self._uavs, uid, UAV_FIELDS) for uid in self.uav_ids))
        tr, tv = [], []
        for i in range(self.task_capacity):
            if i < len(self._tasks):
                row, values = read_row(self._task_values, self._tasks[i], TASK_FIELDS)
            else:
                row, values = (0.0,) * (4 * len(TASK_FIELDS)), {}
            tr.append(row)
            tv.append(values)
        mask = []
        for uav in uv:
            u_ok = (all(v['valid'] for v in uav.values()) and uav['energy']['value'] > 0
                    and all(uav[f]['value'] == 1 for f in ('alive', 'connected', 'idle')))
            for task in tv:
                t_ok = (not self._tasks[i] in self._completed_tasks if i < len(self._tasks) else True) and (bool(task) and all(task[field]['valid'] for field in TASK_REQUIRED_FIELDS)
                        and task['pending']['value'] == 1 and task['deadline']['value'] > now
                        and task['remaining_service']['value'] > 0)
                mask.append(bool(u_ok and t_ok))
        return TaskPolicySnapshot(now, self.version, tuple(ur), tuple(tr), tuple(mask) + (True,))

    def resolve(self, action: int):
        """Environment-side mapping; not a substitute for a snapshot mask check."""
        noop = len(self.uav_ids) * self.task_capacity
        if type(action) is not int or not 0 <= action <= noop:
            raise ValueError('Action outside fixed interface')
        if action == noop:
            return None
        u, t = divmod(action, self.task_capacity)
        if t >= len(self._tasks):
            raise ValueError('Undelivered task slot')
        return self._tasks[t], self.uav_ids[u]
