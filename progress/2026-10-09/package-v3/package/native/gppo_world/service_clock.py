"""Event-boundary service accounting, a component of M-10 (not a full env).

Normalized straight-line travel, idle and service energy are accounted here.
The integrating environment owns observation, command gates and charging.
"""
from dataclasses import dataclass
import math
from .task_lifecycle import TaskLifecycle, TaskState


@dataclass
class ServiceResource:
    energy: float
    service_rate: float
    service_power: float
    alive: bool = True
    connected: bool = True
    position: tuple[float, float] = (0.0, 0.0)
    speed: float = 1.0
    travel_power: float = 1.0
    idle_power: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.energy, self.service_rate, self.service_power)):
            raise ValueError("Resource values must be finite")
        if self.energy < 0 or self.service_rate <= 0 or self.service_power <= 0:
            raise ValueError("Invalid resource values")
        if (len(self.position) != 2 or
                not all(math.isfinite(v) for v in (*self.position, self.speed,
                                                  self.travel_power, self.idle_power)) or
                self.speed <= 0 or self.travel_power <= 0 or self.idle_power < 0):
            raise ValueError("Invalid motion parameters")


@dataclass(frozen=True)
class ServiceEvent:
    time: float
    resource: str
    kind: str

    def __post_init__(self):
        if not math.isfinite(self.time) or self.time < 0:
            raise ValueError("Invalid event time")
        if self.kind not in ("damage", "disconnect", "reconnect"):
            raise ValueError("Unsupported service event")


class ServiceClock:
    def __init__(self, tasks: dict[str, TaskLifecycle], resources: dict[str, ServiceResource],
                 events: list[ServiceEvent], *, disconnect_interrupts: bool,
                 task_positions: dict[str, tuple[float, float]] | None = None):
        if any(task.last_time != -math.inf for task in tasks.values()):
            raise ValueError("Clock requires fresh tasks")
        if any(e.resource not in resources for e in events):
            raise ValueError("Unknown event resource")
        # Simultaneous disconnect/reconnect is ambiguous: require a single
        # explicit transition. Damage plus either transition remains valid.
        links = [(e.time, e.resource) for e in events if e.kind != "damage"]
        if len(links) != len(set(links)):
            raise ValueError("Ambiguous simultaneous link transitions")
        self.tasks, self.resources = tasks, resources
        self.task_positions = (dict(task_positions) if task_positions is not None
                               else {key: (0.0, 0.0) for key in tasks})
        if (set(self.task_positions) != set(tasks) or
                any(len(p) != 2 or not all(math.isfinite(x) for x in p)
                    for p in self.task_positions.values())):
            raise ValueError("One finite 2D position per task required")
        self.events = sorted(events, key=lambda e: (e.time, e.resource, e.kind))
        self.disconnect_interrupts = disconnect_interrupts
        self.time = 0.0
        self.cursor = 0
        self.log = []
        self._events_now()
        for task in tasks.values():
            task.advance(0)

    def _events_now(self):
        while self.cursor < len(self.events) and self.events[self.cursor].time == self.time:
            event = self.events[self.cursor]
            resource = self.resources[event.resource]
            if event.kind == "damage":
                resource.alive = False
            else:
                resource.connected = event.kind == "reconnect"
            if not resource.alive or (self.disconnect_interrupts and not resource.connected):
                for task in self.tasks.values():
                    if task.assigned_uav == event.resource:
                        task.interrupt(self.time)
            self.log.append({"kind": event.kind, "resource": event.resource, "time": self.time})
            self.cursor += 1

    def assign(self, task_id: str, resource_id: str):
        resource = self.resources[resource_id]
        if not resource.alive or not resource.connected or resource.energy <= 0:
            raise ValueError("Execution resource unavailable")
        if any(t.assigned_uav == resource_id for t in self.tasks.values()):
            raise ValueError("Resource already occupied")
        self.tasks[task_id].assign(resource_id, self.time)

    def advance(self, end: float):
        if not math.isfinite(end) or end < self.time:
            raise ValueError("Clock cannot regress")
        while self.time < end:
            # A task may be assigned while its UAV is already inside the
            # configured target region. Record that physical completion at
            # the current clock time before calculating the next boundary.
            for task_id, task in self.tasks.items():
                if task.assigned_uav is None or task.completion_mode != "arrival_to_region":
                    continue
                if math.dist(self.resources[task.assigned_uav].position, self.task_positions[task_id]) <= task.completion_radius + 1e-12:
                    uid = task.assigned_uav
                    task.arrive(uid, self.time)
                    self.log.append({"kind": "arrival", "task": task_id, "resource": uid,
                                     "time": self.time, "target": self.task_positions[task_id],
                                     "radius": task.completion_radius})
            stop = end
            occupied = {t.assigned_uav: key for key, t in self.tasks.items()
                        if t.assigned_uav is not None}
            if self.cursor < len(self.events):
                stop = min(stop, self.events[self.cursor].time)
            for task_id, task in self.tasks.items():
                for boundary in (task.arrival, task.deadline):
                    if boundary > self.time:
                        stop = min(stop, boundary)
                if task.assigned_uav is not None:
                    resource = self.resources[task.assigned_uav]
                    distance = math.dist(resource.position, self.task_positions[task_id])
                    radius = float(task.completion_radius) if task.completion_mode == "arrival_to_region" else 0.0
                    moving = distance > radius + 1e-12
                    power = resource.travel_power if moving else resource.service_power
                    duration = (distance / resource.speed if moving else
                                (task.required_service - task.service) / resource.service_rate)
                    if moving and task.completion_mode == "arrival_to_region":
                        duration = max(0.0, (distance - radius) / resource.speed)
                    stop = min(stop, self.time + resource.energy / power, self.time + duration)
            for uid, resource in self.resources.items():
                if uid not in occupied and resource.alive and resource.energy > 0 and resource.idle_power > 0:
                    stop = min(stop, self.time + resource.energy / resource.idle_power)
            if stop <= self.time:
                raise RuntimeError("Non-advancing service boundary")
            for task_id, task in self.tasks.items():
                uid = task.assigned_uav
                if uid is not None:
                    resource = self.resources[uid]
                    if not resource.alive or (self.disconnect_interrupts and not resource.connected):
                        raise RuntimeError("Unavailable resource retained assignment")
                    before_energy, before_service = resource.energy, task.service
                    destination = self.task_positions[task_id]
                    distance = math.dist(resource.position, destination)
                    before_position = resource.position
                    moving = distance > 1e-12
                    if moving:
                        fraction = min(1.0, (stop - self.time) * resource.speed / distance)
                        resource.position = (destination if fraction >= 1.0 - 1e-12 else
                                             tuple(a + fraction * (b - a)
                                                   for a, b in zip(resource.position, destination)))
                        entered = task.completion_mode == "arrival_to_region" and distance - (stop - self.time) * resource.speed <= task.completion_radius + 1e-12
                        if entered:
                            task.arrive(uid, stop)
                            self.log.append({"kind": "arrival", "task": task_id, "resource": uid,
                                             "time": stop, "target": destination,
                                             "radius": task.completion_radius})
                        else:
                            task.advance(stop)
                    else:
                        task.provide_service(uid, self.time, stop, resource.service_rate)
                    used = (stop - self.time) * (resource.travel_power if moving else resource.service_power)
                    if used > before_energy + 1e-9:
                        raise RuntimeError("Service exceeded available energy")
                    resource.energy = max(0.0, before_energy - used)
                    if resource.energy <= 1e-12:
                        resource.energy = 0.0
                        task.interrupt(stop)
                    self.log.append({"kind": "travel" if moving else "service", "task": task_id, "resource": uid,
                                     "start": self.time, "end": stop,
                                     "position_before": before_position, "position_after": resource.position,
                                     "energy_before": before_energy, "energy_after": resource.energy,
                                     "service_before": before_service, "service_after": task.service})
                else:
                    task.advance(stop)
            for uid, resource in self.resources.items():
                if uid not in occupied and resource.alive and resource.energy > 0 and resource.idle_power > 0:
                    before = resource.energy
                    resource.energy = max(0.0, before - (stop - self.time) * resource.idle_power)
                    if resource.energy <= 1e-12:
                        resource.energy = 0.0
                    self.log.append({"kind": "idle", "resource": uid, "start": self.time,
                                     "end": stop, "energy_before": before,
                                     "energy_after": resource.energy})
            self.time = stop
            self._events_now()
