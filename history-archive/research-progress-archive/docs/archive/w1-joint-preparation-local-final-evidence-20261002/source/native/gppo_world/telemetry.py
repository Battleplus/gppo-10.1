"""Received-only telemetry store; scheduling and simulator truth live elsewhere."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Telemetry:
    entity: str
    field: str
    value: float
    measured_at: float
    received_at: float
    sequence: int
    # Stable wire identity for audit/replay.  It is not part of the policy
    # view and is deliberately optional for compatibility with old fixtures.
    message_id: str = ""

    def __post_init__(self):
        if not self.entity or not self.field:
            raise ValueError("Telemetry identity required")
        if not all(math.isfinite(x) for x in (self.value, self.measured_at, self.received_at)):
            raise ValueError("Telemetry must be finite")
        if self.measured_at < 0 or self.received_at < self.measured_at:
            raise ValueError("Invalid telemetry timeline")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("Sequence must be a nonnegative integer")
        if not isinstance(self.message_id, str):
            raise ValueError("Message identity must be a string")


class ReceivedTelemetry:
    def __init__(self):
        self._latest = {}
        self._now = 0.0

    def advance(self, now: float):
        if not math.isfinite(now) or now < self._now:
            raise ValueError("Observation time must be monotonic")
        self._now = now

    def ingest(self, message: Telemetry, now: float) -> bool:
        if message.received_at > now:
            raise ValueError("Future delivery cannot enter received state")
        self.advance(now)
        key = (message.entity, message.field)
        previous = self._latest.get(key)
        if previous is not None:
            if message.sequence <= previous.sequence:
                if message.sequence == previous.sequence and message != previous:
                    raise ValueError("Conflicting duplicate telemetry")
                return False
            if message.measured_at < previous.measured_at:
                raise ValueError("New sequence regresses measurement time")
        self._latest[key] = message
        return True

    def read(self, entity: str, field: str, now: float, max_age: float):
        if not math.isfinite(max_age) or max_age < 0:
            raise ValueError("Finite nonnegative max_age required")
        self.advance(now)
        message = self._latest.get((entity, field))
        if message is None:
            return {"value": 0.0, "known": False, "valid": False, "age": 0.0}
        age = now - message.measured_at
        return {"value": message.value, "known": True,
                "valid": age <= max_age, "age": age}

    def reset(self):
        self._latest.clear()
        self._now = 0.0
