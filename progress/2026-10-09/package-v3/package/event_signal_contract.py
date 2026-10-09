"""Frozen pre-action event_signal semantics shared by production boundaries."""
from __future__ import annotations

import math

SCHEMA = "w1-event-signal-contract/1"
FIELD = "event_signal"
VALIDITY_FIELD = "event_signal_valid"
SEMANTICS = "pre_action_public_observation"


def producer_value(trigger_flags):
    if not isinstance(trigger_flags, dict) or any(type(v) is not bool for v in trigger_flags.values()):
        raise ValueError("EVENT_SIGNAL_TRIGGER_FLAGS")
    return {FIELD: float(any(trigger_flags.values())), VALIDITY_FIELD: True}


def validate(observation, *, path="$"):
    if not isinstance(observation, dict):
        raise ValueError("EVENT_SIGNAL_OBSERVATION_TYPE:" + path)
    if VALIDITY_FIELD not in observation or type(observation[VALIDITY_FIELD]) is not bool:
        raise ValueError("EVENT_SIGNAL_VALIDITY_REQUIRED:" + path)
    if not observation[VALIDITY_FIELD]:
        raise ValueError("EVENT_SIGNAL_UNKNOWN:" + path)
    value = observation.get(FIELD)
    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("EVENT_SIGNAL_VALUE_TYPE:" + path)
    if float(value) not in (0.0, 1.0):
        raise ValueError("EVENT_SIGNAL_RANGE:" + path)
    return float(value), True


def align_restored_public(observation, *, expected_event_signal, expected_valid):
    if type(expected_valid) is not bool or not expected_valid:
        raise ValueError("EVENT_SIGNAL_UNKNOWN")
    if type(expected_event_signal) not in (int, float) or isinstance(expected_event_signal, bool):
        raise ValueError("EVENT_SIGNAL_EXPECTED_VALUE_TYPE")
    if float(expected_event_signal) not in (0.0, 1.0):
        raise ValueError("EVENT_SIGNAL_EXPECTED_RANGE")
    out = dict(observation)
    out[FIELD] = float(expected_event_signal)
    out[VALIDITY_FIELD] = True
    return out
