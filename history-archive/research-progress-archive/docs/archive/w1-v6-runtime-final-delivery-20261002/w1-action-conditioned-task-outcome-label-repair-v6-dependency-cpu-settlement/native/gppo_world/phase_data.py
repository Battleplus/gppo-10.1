"""Read-only commit observation and replayable, causal three-phase records.

No copy of the environment's transition logic is implemented here. Observation
occurs at its first event-ingestion call inside a native accepted step, after
reward calculation and the fixed decision clock increment, before new events.
"""
from __future__ import annotations

from collections.abc import Mapping
from collections import deque
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .contracts import snapshot_from_gppo
from .data import graph_from_dict
from .recorder import graph_to_dict


FORMAT = 'gppo-three-phase/1.0.0'
OBSERVER_FIELDS = {'_phase_armed', '_phase_commit', '_phase_capture_count'}


def canonical(value, active=None):
    """Audit full native mutable state, including hidden queues and RNGs.

    Fails closed on unknown types; this is a fingerprint, NOT a restore API.
    Internal truth/RNG values are for offline audit only, never model inputs.
    """
    active = set() if active is None else active
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {'float': repr(value)}
    if isinstance(value, Enum):
        return {'enum': type(value).__qualname__, 'value': canonical(value.value, active)}
    if isinstance(value, np.generic):
        return canonical(value.item(), active)
    if isinstance(value, torch.Tensor):
        return {'tensor': str(value.dtype), 'shape': list(value.shape), 'value': value.detach().cpu().tolist()}
    if isinstance(value, np.ndarray):
        return {'ndarray': str(value.dtype), 'shape': list(value.shape), 'value': value.tolist()}
    if isinstance(value, np.dtype):
        return {'dtype': str(value)}
    if isinstance(value, np.random.Generator):
        return {'generator': canonical(value.bit_generator.state, active)}
    if isinstance(value, Path):
        return str(value)
    oid = id(value)
    if oid in active:
        return {'cycle_type': type(value).__qualname__}
    active.add(oid)
    try:
        if isinstance(value, Mapping):
            return {'mapping': [[canonical(k, active), canonical(v, active)]
                                for k, v in sorted(value.items(), key=lambda item: repr(item[0]))]}
        if isinstance(value, (list, tuple, deque)):
            return [canonical(v, active) for v in value]
        if isinstance(value, (set, frozenset)):
            return {'set': sorted((canonical(v, active) for v in value), key=lambda x: json.dumps(x, sort_keys=True))}
        if is_dataclass(value):
            return {'type': type(value).__qualname__, 'state': canonical({f.name: getattr(value, f.name) for f in fields(value)}, active)}
        if hasattr(value, '__dict__'):
            return {'type': type(value).__qualname__, 'state': canonical(vars(value), active)}
        raise TypeError(f'Unrecognized audit-state type: {type(value)}')
    finally:
        active.remove(oid)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def native_fingerprint(env):
    return digest(canonical({k: v for k, v in vars(env).items() if k not in OBSERVER_FIELDS}))


def rng_fingerprints(env):
    """The environment and detector streams are both audited separately."""
    result = {'environment': digest(canonical(env.rng)), 'gym': digest(canonical(env.np_random))}
    if env.runtime_bridge is not None:
        result['detector'] = digest(canonical(env.runtime_bridge.detector.rng))
    return result


def observed_environment(base_class):
    class PhaseObservedEnv(base_class):
        def _step_current(self, *args, **kwargs):
            self._phase_armed = True
            self._phase_commit = None
            self._phase_capture_count = 0
            try:
                result = super()._step_current(*args, **kwargs)
                if self._phase_capture_count != 1:
                    raise RuntimeError('Expected exactly one pre-ingestion commit observation')
                return result
            finally:
                self._phase_armed = False

        def _ingest_observed_events(self):
            if getattr(self, '_phase_armed', False) and self._phase_commit is None:
                from ppo_allocation.random_event.graph import build_graph_state
                before = native_fingerprint(self)
                graph = graph_to_dict(snapshot_from_gppo(build_graph_state(self)))
                record = {'graph': graph, 'time': float(self.current_time),
                          'action_version': int(self.decision_version),
                          'rng': rng_fingerprints(self), 'event_cursor': int(self.next_event_index)}
                if native_fingerprint(self) != before:
                    raise RuntimeError('Commit observation mutated native environment state')
                self._phase_commit = record
                self._phase_capture_count += 1
            return super()._ingest_observed_events()

    return PhaseObservedEnv


def model_input(record):
    """Explicit allowlist: no target, reward, future delta-time or audit state."""
    pre = record['pre']
    if any(e['received_at'] > pre['time'] for e in pre['evidence']):
        raise ValueError('Future evidence in decision input')
    result = {key: pre[key] for key in ('graph', 'time', 'action_version', 'evidence', 'decision_duration')}
    result['action'] = int(record['action'])
    graph = graph_from_dict(result['graph'])
    if not 0 <= result['action'] < graph.num_actions or not bool(graph.action_mask[result['action']]):
        raise ValueError('Executed/branch action must be legal at its starting state')
    return result


def validate_record(record):
    if record['format'] != FORMAT:
        raise ValueError('Wrong phase contract')
    inp = model_input(record)
    pre, commit, nxt = record['pre'], record['commit'], record['next']
    if abs(commit['time'] - pre['time'] - pre['decision_duration']) > 1e-9:
        raise ValueError('Commit must precede event waiting at the known fixed step duration')
    if nxt['time'] < commit['time'] or commit['action_version'] != pre['action_version'] + 1:
        raise ValueError('Time/action version ordering failure')
    if commit['graph']['graph_version'] != pre['graph']['graph_version'] + 1:
        raise ValueError('Commit graph must have only the native action increment')
    if nxt['graph']['graph_version'] < commit['graph']['graph_version']:
        raise ValueError('Next graph version went backwards')
    if record['execution']['executed_action'] != inp['action'] or not record['execution']['accepted']:
        raise ValueError('Execution did not match factual/branch action')
    for phase in (pre, commit, nxt):
        graph_from_dict(phase['graph'])
    return True


def history_inputs(records, window=4):
    """Causal graph histories ending at each decision, padded with valid bits.

    Past actions are attached only to past frames; the action under evaluation
    remains separate. All branch alternatives share the same factual history.
    """
    if window < 1:
        raise ValueError('History window must be positive')
    episodes = {}
    result = []
    for record in records:
        episode = episodes.setdefault(record['episode_id'], [])
        if record['step'] != len(episode):
            raise ValueError('History requires contiguous factual episode steps')
        current = model_input(record)
        past = episode[-(window - 1):] if window > 1 else []
        frames = [{'graph': r['pre']['graph'], 'time': r['pre']['time'], 'past_executed_action': r['action']}
                  for r in past]
        frames.append({'graph': current['graph'], 'time': current['time'], 'past_executed_action': None})
        valid = [False] * (window - len(frames)) + [True] * len(frames)
        padded = [None] * (window - len(frames)) + frames
        result.append({'current': current, 'history': padded, 'valid': valid,
                       'episode_id': record['episode_id'], 'step': record['step']})
        episode.append(record)
    return result
