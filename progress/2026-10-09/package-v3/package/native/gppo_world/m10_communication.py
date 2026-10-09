"""Deterministic weak-communication tape primitives for M-10.

The profile models links, not simulator truth. Every packet decision is
addressed by semantic message identity, so adding a packet to one condition
does not shift the random stream used by its paired counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any


@dataclass(frozen=True)
class CommunicationProfile:
    """Frozen link impairment parameters; all times use simulator seconds."""

    name: str = "ideal"
    telemetry_extra_delay: float = 0.0
    telemetry_loss_probability: float = 0.0
    telemetry_duplicate_probability: float = 0.0
    telemetry_reorder_window: float = 0.0
    telemetry_outage_intervals: tuple[tuple[float, float], ...] = ()
    command_loss_probability: float = 0.0
    ack_loss_probability: float = 0.0
    # Lease-renewal requests use the command link's loss budget.  These
    # optional fields model renewal-only transport delay/duplication/reorder;
    # defaults preserve the existing M-10 tapes byte-for-byte.
    renewal_extra_delay: float = 0.0
    renewal_duplicate_probability: float = 0.0
    renewal_reorder_window: float = 0.0

    def __post_init__(self) -> None:
        probabilities = (self.telemetry_loss_probability, self.telemetry_duplicate_probability,
                         self.command_loss_probability, self.ack_loss_probability,
                         self.renewal_duplicate_probability)
        if any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in probabilities):
            raise ValueError("link probabilities must be in [0, 1]")
        if not math.isfinite(self.telemetry_extra_delay) or self.telemetry_extra_delay < 0:
            raise ValueError("telemetry extra delay must be nonnegative")
        if not math.isfinite(self.telemetry_reorder_window) or self.telemetry_reorder_window < 0:
            raise ValueError("reorder window must be nonnegative")
        if (not math.isfinite(self.renewal_extra_delay) or self.renewal_extra_delay < 0 or
                not math.isfinite(self.renewal_reorder_window) or self.renewal_reorder_window < 0):
            raise ValueError("renewal delay/reorder parameters must be nonnegative")
        for start, end in self.telemetry_outage_intervals:
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise ValueError("outage intervals must be finite and increasing")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "telemetry_extra_delay": self.telemetry_extra_delay,
            "telemetry_loss_probability": self.telemetry_loss_probability,
            "telemetry_duplicate_probability": self.telemetry_duplicate_probability,
            "telemetry_reorder_window": self.telemetry_reorder_window,
            "telemetry_outage_intervals": [list(interval) for interval in self.telemetry_outage_intervals],
            "command_loss_probability": self.command_loss_probability,
            "ack_loss_probability": self.ack_loss_probability,
            "renewal_extra_delay": self.renewal_extra_delay,
            "renewal_duplicate_probability": self.renewal_duplicate_probability,
            "renewal_reorder_window": self.renewal_reorder_window,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CommunicationProfile":
        if not payload:
            return cls()
        return cls(
            name=str(payload.get("name", "custom")),
            telemetry_extra_delay=float(payload.get("telemetry_extra_delay", 0.0)),
            telemetry_loss_probability=float(payload.get("telemetry_loss_probability", 0.0)),
            telemetry_duplicate_probability=float(payload.get("telemetry_duplicate_probability", 0.0)),
            telemetry_reorder_window=float(payload.get("telemetry_reorder_window", 0.0)),
            telemetry_outage_intervals=tuple(tuple(float(x) for x in interval)
                                              for interval in payload.get("telemetry_outage_intervals", ())),
            command_loss_probability=float(payload.get("command_loss_probability", 0.0)),
            ack_loss_probability=float(payload.get("ack_loss_probability", 0.0)),
            renewal_extra_delay=float(payload.get("renewal_extra_delay", 0.0)),
            renewal_duplicate_probability=float(payload.get("renewal_duplicate_probability", 0.0)),
            renewal_reorder_window=float(payload.get("renewal_reorder_window", 0.0)),
        )

    def _uniform(self, seed: int, link: str, identity: str) -> float:
        material = f"m10-comm-v1|{seed}|{self.name}|{link}|{identity}".encode("utf-8")
        digest = hashlib.sha256(material).digest()
        return int.from_bytes(digest[:8], "big") / float(2**64)

    def _in_outage(self, now: float) -> bool:
        return any(start <= now < end for start, end in self.telemetry_outage_intervals)

    def telemetry(self, *, seed: int, identity: str, now: float) -> dict[str, Any]:
        random_loss = self._uniform(seed, "telemetry-loss", identity) < self.telemetry_loss_probability
        outage = self._in_outage(now)
        dropped = bool(random_loss or outage)
        jitter = self._uniform(seed, "telemetry-order", identity) * self.telemetry_reorder_window
        duplicate = (not dropped and self._uniform(seed, "telemetry-duplicate", identity)
                     < self.telemetry_duplicate_probability)
        return {"dropped": dropped, "outage": outage, "duplicate": duplicate,
                "jitter": jitter, "loss": random_loss}

    def command_delivered(self, *, seed: int, identity: str) -> bool:
        return self._uniform(seed, "command", identity) >= self.command_loss_probability

    def ack_delivered(self, *, seed: int, identity: str) -> bool:
        return self._uniform(seed, "ack", identity) >= self.ack_loss_probability

    def renewal(self, *, seed: int, identity: str) -> dict[str, Any]:
        """Return deterministic transport fate for one lease renewal request."""
        dropped = self._uniform(seed, "renewal-loss", identity) < self.command_loss_probability
        delay = self.renewal_extra_delay + self._uniform(seed, "renewal-order", identity) * self.renewal_reorder_window
        duplicate = (not dropped and self._uniform(seed, "renewal-duplicate", identity)
                     < self.renewal_duplicate_probability)
        return {"dropped": bool(dropped), "delay": float(delay), "duplicate": bool(duplicate)}


def weak_communication_profile(level: str) -> CommunicationProfile:
    """Return frozen protocol levels used by the pilot and formal tapes."""
    profiles = {
        "ideal": CommunicationProfile(),
        "telemetry-delay": CommunicationProfile(name=level, telemetry_extra_delay=0.5),
        "random-loss": CommunicationProfile(name=level, telemetry_loss_probability=0.20),
        "burst-loss": CommunicationProfile(name=level, telemetry_outage_intervals=((3.0, 5.0), (10.0, 12.0))),
        "reorder": CommunicationProfile(name=level, telemetry_reorder_window=0.75),
        "recovery": CommunicationProfile(name=level, telemetry_outage_intervals=((3.0, 6.0),)),
        "composite": CommunicationProfile(
            name=level, telemetry_extra_delay=0.5, telemetry_loss_probability=0.15,
            telemetry_reorder_window=0.75, telemetry_outage_intervals=((3.0, 5.0), (10.0, 12.0)),
            command_loss_probability=0.05, ack_loss_probability=0.10,
        ),
    }
    if level not in profiles:
        raise ValueError(f"unknown communication level: {level}")
    return profiles[level]


def formal_three_condition_profile(level: str) -> CommunicationProfile:
    """Return the pre-registered formal-matrix link profile.

    Per-episode UAV disconnection is represented in the scenario event tape,
    not as a global link outage.  This profile therefore contains only the
    link-local impairment fields; the scenario generator records the selected
    UAV, start and duration alongside it.
    """
    profiles = {
        "I": CommunicationProfile(name="I-ideal"),
        "W1": CommunicationProfile(
            name="W1-light",
            telemetry_extra_delay=1.0,
            telemetry_loss_probability=0.05,
            command_loss_probability=0.02,
            ack_loss_probability=0.02,
        ),
        "W2": CommunicationProfile(
            name="W2-moderate",
            telemetry_extra_delay=2.0,
            telemetry_loss_probability=0.15,
            command_loss_probability=0.05,
            ack_loss_probability=0.05,
        ),
    }
    if level not in profiles:
        raise ValueError(f"unknown formal communication condition: {level}")
    return profiles[level]


__all__ = ["CommunicationProfile", "weak_communication_profile", "formal_three_condition_profile"]
