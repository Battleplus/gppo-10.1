"""Execute the real observation producer on a controlled telemetry bridge, not an environment."""
import copy
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'package'), str(ROOT / 'package/native')]
os.sched_setaffinity(0, {0})
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
START = time.monotonic()
import torch
import numpy as np
from gppo_world.m10_environment import M10Environment, M10Config
from gppo_world.graph5 import graph5_from_m10_observation
from public_controller import public_copy
from public_prefix import make_public_prefix, serialize_public_prefix
import public_prefix as pp

FORBIDDEN = []
def forbidden(*args, **kwargs):
    FORBIDDEN.append(True)
    raise AssertionError('MODEL_OR_CHECKPOINT_EXECUTION_FORBIDDEN')
torch.load = forbidden
torch.nn.Module._call_impl = forbidden
OUT = Path(os.environ['W1_PRODUCER_ACCEPTANCE_OUTPUT'])
OUT.mkdir(parents=True, exist_ok=False)
rows = []
def fields(values): return [x for v in values for x in (float(v), 1., 1., 0.)]
for count in (3, 4, 6):
    for now in (2., 4., 8., 12.):
        for event in (False, True):
            uavs = [fields((0., 0., 9., 1., 1., 1.)) for _ in range(4)]
            tasks = [fields((float(i), 0., 18., 1., 1., 1., 0., float(i % 4))) if i < count else [0.] * 32 for i in range(6)]
            mask = [False] * 25
            mask[8] = mask[24] = True
            telemetry = SimpleNamespace(uavs=uavs, tasks=tasks, mask=mask, version=4)
            bridge = SimpleNamespace(observe=lambda: telemetry)
            owner = SimpleNamespace(bridge=bridge, config=M10Config(), clock=SimpleNamespace(time=now),
                _public_event_records=[], _trigger_flags={'semantic_event': event},
                uav_ids=tuple('uav-' + str(i) for i in range(4)),
                view=SimpleNamespace(public_task_ids=tuple('task-' + str(i) for i in range(count))),
                _active_action=None, _active_actions={}, _exogenous_key='synthetic-producer-key')
            before = M10Environment._observation(owner, clear_trigger=True)
            saved = copy.deepcopy(before)
            public = public_copy(before)
            prefix = serialize_public_prefix(make_public_prefix(public))
            graph = graph5_from_m10_observation(public)
            after = M10Environment._observation(owner)
            assert saved['event_signal'] == saved['flat'][769] == float(event)
            assert after['event_signal'] == after['flat'][769] == 0.
            assert float(graph.global_features[1]) == float(event)
            assert np.array_equal(prefix['flat'], saved['flat'])
            rows.append({'tasks': count, 'time': now, 'event': event, 'pass': True,
                'saved_flat769': float(saved['flat'][769]), 'reobserved_flat769': float(after['flat'][769]),
                'source_sha256': saved['public_prefix_source']['source_sha256']})
assert not FORBIDDEN
rejections = []
for name, mutate in [
    ('flat_dtype_float64', lambda s: s.update(flat=s['flat'].astype(np.float64))),
    ('task_ids_tuple', lambda s: s['public_entity_ids'].update(tasks=tuple(s['public_entity_ids']['tasks']))),
    ('event_type_int', lambda s: s.update(event_signal=1)),
]:
    state = copy.deepcopy(saved); mutate(state); before = pp.FLATTEN_CALLS
    try: public_copy(state)
    except ValueError as exc:
        assert pp.FLATTEN_CALLS == before
        rejections.append({'name': name, 'error': str(exc), 'before_flatten': True})
    else: raise AssertionError('Consumer normalized a type mismatch: ' + name)
report = {'pass': True, 'controlled_producer_cases': len(rows), 'cases': rows,
    'strict_consumer_rejections': rejections,
    'actual_method': 'M10Environment._observation', 'environment_constructor_called': False,
    'formal_environment_steps': 0, 'neural_forwards': 0, 'checkpoint_loads': 0,
    'world_updates': 0, 'GPPO_updates': 0, 'task_evaluation_episodes': 0,
    'wall_seconds_including_import': time.monotonic() - START, 'CPU_seconds_including_import': time.process_time(),
    'max_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
(OUT / 'producer-acceptance-summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ('pass', 'controlled_producer_cases', 'neural_forwards', 'checkpoint_loads')}))
