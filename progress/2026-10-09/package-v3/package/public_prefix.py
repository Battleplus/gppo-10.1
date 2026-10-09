"""Single typed pre-action prefix and flat producer for all consequence boundaries."""
import copy
from dataclasses import dataclass, fields
import hashlib
import json
import math
import numpy as np

SCHEMA = 'w1-public-prefix/1'
LAYOUT = 'w1-public-flat/1'
COUNTS = (4, 3, 4, 6, 4)
WIDTH = 32
FLAT_DIM = 770
FLATTEN_CALLS = 0
CORE = ('graph', 'uavs', 'tasks', 'time', 'version', 'public_entity_ids',
        'continuation_actions', 'trigger_flags', 'event_signal', 'event_signal_valid')
STATE_FIELDS = set(CORE) | {'mask', 'types', 'continuation_action', 'public_history', 'public_history_schema'}
SOURCE_FIELDS = {'schema', 'phase', 'generation_stage', 'source_object', 'horizon',
                 'node_counts', 'flat_dtype', 'random_key_sha256', 'source_sha256'}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('ascii')


def sha(value):
    return hashlib.sha256(value if type(value) is bytes else encoded(value)).hexdigest()


def native_json(value):
    """Conversion is allowed only at the declared NumPy observation producer boundary."""
    if isinstance(value, np.ndarray):
        if value.dtype not in (np.dtype('float32'), np.dtype('bool')):
            raise ValueError('PUBLIC_PREFIX_NATIVE_DTYPE:' + str(value.dtype))
        return value.tolist()
    if type(value) is dict: return {k: native_json(v) for k, v in value.items()}
    if type(value) in (list, tuple): return [native_json(v) for v in value]
    if value is None or type(value) in (str, bool, int): return value
    if type(value) is float and math.isfinite(value): return value
    raise ValueError('PUBLIC_PREFIX_NATIVE_TYPE:' + type(value).__name__)


def mismatch(path, producer, consumer, producer_source, consumer_source):
    detail = {'field': path, 'producer_value': producer, 'consumer_value': consumer,
              'producer_type': type(producer).__name__, 'consumer_type': type(consumer).__name__,
              'producer_source_sha256': producer_source, 'consumer_source_sha256': consumer_source,
              'rejected_before_flatten': True}
    raise ValueError('PUBLIC_PREFIX_FIELD_MISMATCH:' + json.dumps(detail, sort_keys=True))


def compare(producer, consumer, path='$', producer_source=None, consumer_source=None):
    if type(producer) is not type(consumer):
        mismatch(path, producer, consumer, producer_source, consumer_source)
    if type(producer) is dict:
        if set(producer) != set(consumer):
            mismatch(path + '.fields', sorted(producer), sorted(consumer), producer_source, consumer_source)
        for key in sorted(producer):
            compare(producer[key], consumer[key], path + '.' + key, producer_source, consumer_source)
    elif type(producer) is list:
        if len(producer) != len(consumer):
            mismatch(path + '.shape', [len(producer)], [len(consumer)], producer_source, consumer_source)
        for i, (a, b) in enumerate(zip(producer, consumer)):
            compare(a, b, path + '[' + str(i) + ']', producer_source, consumer_source)
    elif producer != consumer:
        mismatch(path, producer, consumer, producer_source, consumer_source)


def matrix(value, shape, path):
    if type(value) is not list or len(value) != shape[0]: raise ValueError('PUBLIC_PREFIX_SHAPE:' + path)
    if len(shape) > 1:
        for i, row in enumerate(value): matrix(row, shape[1:], path + '[' + str(i) + ']')
    elif any(type(v) is not float or not math.isfinite(v) for v in value):
        raise ValueError('PUBLIC_PREFIX_FLOAT32_JSON_VALUES:' + path)


def source_material(state, source):
    return {'observation': {k: state[k] for k in CORE},
            'header': {k: source[k] for k in SOURCE_FIELDS - {'source_sha256'}}}


def layout_manifest():
    cursor = 0; blocks = []
    for name, count in zip(('uav', 'region', 'target', 'task', 'event'), COUNTS):
        blocks.append({'field': 'graph.node_features.' + name, 'offset': cursor,
            'end_exclusive': cursor + count * WIDTH, 'shape': [count, WIDTH], 'dtype': 'float32',
            'source_object': 'pre_action_public_state.graph.node_features', 'generation_stage': 'pre_action_observation'})
        cursor += count * WIDTH
    blocks.extend([
        {'field': 'graph.relations', 'offset': 672, 'end_exclusive': 768, 'shape': [4, 6, 4], 'dtype': 'float32',
         'source_object': 'pre_action_public_state.graph.relations', 'generation_stage': 'pre_action_observation'},
        {'field': 'time_fraction', 'offset': 768, 'end_exclusive': 769, 'shape': [1], 'dtype': 'float32',
         'source_object': 'time / snapshot_identity.horizon', 'generation_stage': 'serialize_public_prefix'},
        {'field': 'event_signal', 'offset': 769, 'end_exclusive': 770, 'shape': [1], 'dtype': 'float32',
         'source_object': 'pre_action_public_state.event_signal', 'generation_stage': 'pre_action_observation'},
    ])
    return {'schema': LAYOUT, 'dimension': FLAT_DIM, 'dtype': 'float32', 'order': 'C',
            'blocks': blocks, 'index_769': {'field': 'event_signal', 'element_offset': 769,
                'byte_offset': 3076, 'dtype': 'float32', 'validity_field': 'event_signal_valid'}}


@dataclass(frozen=True)
class PublicPrefix:
    pre_action_public_state: dict
    event_signal: float
    event_signal_valid: bool
    candidate_key: dict | None
    time: float
    task_slot: int | None
    legal_mask: list
    snapshot_identity: dict
    flat_cache: object = None

    def payload(self):
        return {'schema': SCHEMA, **{f.name: copy.deepcopy(getattr(self, f.name))
            for f in fields(self) if f.name != 'flat_cache'}}

    @classmethod
    def from_payload(cls, value):
        expected = {'schema'} | {f.name for f in fields(cls) if f.name != 'flat_cache'}
        if type(value) is not dict or set(value) != expected or value['schema'] != SCHEMA:
            raise ValueError('PUBLIC_PREFIX_PAYLOAD_FIELDS')
        return cls(**{k: copy.deepcopy(v) for k, v in value.items() if k != 'schema'})


def make_public_prefix(observation, context=None, candidate_key=None):
    if type(observation) is not dict: raise ValueError('PUBLIC_PREFIX_STATE_DICT')
    if not {'event_signal', 'event_signal_valid', 'time', 'mask'} <= set(observation):
        raise ValueError('PUBLIC_PREFIX_MISSING_FIELD')
    source = observation.get('public_prefix_source')
    if type(source) is not dict or set(source) != SOURCE_FIELDS:
        raise ValueError('PUBLIC_PREFIX_SOURCE_REQUIRED')
    state = {k: copy.deepcopy(v) for k, v in observation.items() if k not in ('flat', 'public_prefix_source')}
    snapshot = dict(source, parent=None if context is None else context['parent'],
                    repeat=None if context is None else context['repeat'],
                    window_id=None if context is None else context['window_id'], time=state['time'],
                    public_sha256=None if context is None else context['public_sha256'],
                    context_prefix_sha256=None if context is None else context['prefix_sha256'])
    return PublicPrefix(state, state['event_signal'], state['event_signal_valid'], copy.deepcopy(candidate_key),
                        state['time'], None if candidate_key is None else candidate_key['task_slot'],
                        copy.deepcopy(state['mask']), snapshot, observation.get('flat'))


def validate_prefix(prefix):
    if type(prefix) is not PublicPrefix: raise ValueError('PUBLIC_PREFIX_TYPE')
    state, identity = prefix.pre_action_public_state, prefix.snapshot_identity
    if type(state) is not dict or set(state) - STATE_FIELDS or not set(CORE) <= set(state):
        raise ValueError('PUBLIC_PREFIX_PUBLIC_FIELDS')
    if type(identity) is not dict or set(identity) != SOURCE_FIELDS | {'parent', 'repeat', 'window_id', 'time', 'public_sha256', 'context_prefix_sha256'}:
        raise ValueError('PUBLIC_PREFIX_SNAPSHOT_FIELDS')
    source = {k: identity[k] for k in SOURCE_FIELDS}
    if source['schema'] != SCHEMA or source['phase'] != 'pre_action' or source['flat_dtype'] != 'float32':
        raise ValueError('PUBLIC_PREFIX_PHASE_OR_DTYPE')
    if type(source['node_counts']) is not list or source['node_counts'] != list(COUNTS):
        raise ValueError('PUBLIC_PREFIX_NODE_LAYOUT')
    if type(source['horizon']) is not float or source['horizon'] != 18.0:
        raise ValueError('PUBLIC_PREFIX_HORIZON')
    if source['generation_stage'] != 'pre_action_observation' or source['source_object'] not in (
        'M10Environment._observation.result', 'controlled_environment._observation.result'):
        raise ValueError('PUBLIC_PREFIX_SOURCE_STAGE')
    for name in ('source_sha256', 'random_key_sha256'):
        value = source[name]
        if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('PUBLIC_PREFIX_SOURCE_DIGEST:' + name)
    from event_signal_contract import validate
    validate({'event_signal': prefix.event_signal, 'event_signal_valid': prefix.event_signal_valid})
    if type(prefix.event_signal) is not float: raise ValueError('PUBLIC_PREFIX_EVENT_FLOAT_REQUIRED')
    for name, value in [('event_signal', prefix.event_signal), ('event_signal_valid', prefix.event_signal_valid),
                        ('time', prefix.time), ('mask', prefix.legal_mask)]:
        compare(state[name], value, '$.' + name, source['source_sha256'], source['source_sha256'])
    if type(prefix.time) is not float or not math.isfinite(prefix.time) or prefix.time < 0:
        raise ValueError('PUBLIC_PREFIX_TIME')
    if type(state['version']) is not int or type(state['trigger_flags']) is not dict or any(
        type(k) is not str or type(v) is not bool for k, v in state['trigger_flags'].items()):
        raise ValueError('PUBLIC_PREFIX_VERSION_OR_FLAGS')
    if type(state['continuation_actions']) is not list or any(type(a) is not int or not 0 <= a < 24 for a in state['continuation_actions']):
        raise ValueError('PUBLIC_PREFIX_CONTINUATION_ACTIONS')
    if 'public_history' in state:
        matrix(state['public_history'], [128], '$.public_history')
        if state.get('public_history_schema') != 'w1-causal-public-history/1.0.0':
            raise ValueError('PUBLIC_PREFIX_HISTORY_SCHEMA')
    compare(identity['time'], prefix.time, '$.snapshot_identity.time')
    if type(prefix.legal_mask) is not list or len(prefix.legal_mask) != 25 or any(type(v) is not bool for v in prefix.legal_mask):
        raise ValueError('PUBLIC_PREFIX_LEGAL_MASK')
    graph = state['graph']
    if type(graph) is not dict or set(graph) != {'node_features', 'relations'}:
        raise ValueError('PUBLIC_PREFIX_GRAPH_FIELDS')
    matrix(graph['node_features'], [21, 32], '$.graph.node_features')
    matrix(graph['relations'], [4, 6, 4], '$.graph.relations')
    matrix(state['uavs'], [4, 24], '$.uavs'); matrix(state['tasks'], [6, 32], '$.tasks')
    compare([r[:24] for r in graph['node_features'][:4]], state['uavs'], '$.uavs.graph_alignment')
    compare(graph['node_features'][11:17], state['tasks'], '$.tasks.graph_alignment')
    ids = state['public_entity_ids']
    if type(ids) is not dict or set(ids) != {'tasks', 'uavs'}: raise ValueError('PUBLIC_PREFIX_ENTITY_FIELDS')
    for name, capacity in [('uavs', 4), ('tasks', 6)]:
        if type(ids[name]) is not list or len(ids[name]) > capacity or any(type(v) is not str or not v for v in ids[name]) or len(set(ids[name])) != len(ids[name]):
            raise ValueError('PUBLIC_PREFIX_ENTITY_IDENTITIES:' + name)
    if len(ids['uavs']) != 4: raise ValueError('PUBLIC_PREFIX_UAV_CAPACITY')
    if any(legal and a % 6 >= len(ids['tasks']) for a, legal in enumerate(prefix.legal_mask[:24])):
        raise ValueError('PUBLIC_PREFIX_ILLEGAL_SLOT')
    actual = sha(source_material(state, source))
    if source['source_sha256'] != actual:
        mismatch('$.snapshot_identity.source_sha256', source['source_sha256'], actual, source['source_sha256'], actual)
    key = prefix.candidate_key
    if key is not None:
        from candidate_keys import KEY_FIELDS
        if type(key) is not dict or set(key) != KEY_FIELDS: raise ValueError('PUBLIC_PREFIX_CANDIDATE_FIELDS')
        for name, expected in [('parent', identity['parent']), ('repeat', identity['repeat']),
            ('window_id', identity['window_id']), ('time', prefix.time), ('public_sha256', identity['public_sha256']),
            ('prefix_sha256', identity['context_prefix_sha256'])]:
            compare(expected, key[name], '$.candidate_key.' + name, source['source_sha256'], actual)
        a = key['action_index']
        if type(a) is not int or not 0 <= a < 25 or not prefix.legal_mask[a]: raise ValueError('PUBLIC_PREFIX_ILLEGAL_CANDIDATE')
        legal = [i for i, v in enumerate(prefix.legal_mask) if v]
        compare(legal.index(a), key['candidate_order'], '$.candidate_key.candidate_order')
        noop = a == 24
        for name, expected in [('noop', noop), ('schema', 'w1-structured-candidate/1'),
            ('uav_slot', None if noop else a // 6), ('task_slot', None if noop else a % 6),
            ('uav_id', None if noop else ids['uavs'][a // 6]), ('task_id', None if noop else ids['tasks'][a % 6])]:
            compare(expected, key[name], '$.candidate_key.' + name)
        compare(key['task_slot'], prefix.task_slot, '$.task_slot')
    elif prefix.task_slot is not None: raise ValueError('PUBLIC_PREFIX_UNBOUND_SLOT')
    return source


def _scalar_sources(prefix):
    graph = prefix.pre_action_public_state['graph']
    for i, row in enumerate(graph['node_features']):
        for j, value in enumerate(row): yield '$.graph.node_features[' + str(i) + '][' + str(j) + ']', value
    for u, tasks in enumerate(graph['relations']):
        for t, row in enumerate(tasks):
            for j, value in enumerate(row): yield f'$.graph.relations[{u}][{t}][{j}]', value
    yield '$.time_fraction', float(np.float32(prefix.time / prefix.snapshot_identity['horizon']))
    yield '$.event_signal', prefix.event_signal


def serialize_public_prefix(prefix, *, expected=None, expected_record=None, validation_only=False):
    """Compare named typed fields and caches before any new flat vector is built."""
    global FLATTEN_CALLS
    if expected is not None:
        compare(expected.payload(), prefix.payload(), producer_source=expected.snapshot_identity['source_sha256'],
                consumer_source=prefix.snapshot_identity['source_sha256'])
    source = validate_prefix(prefix)
    payload = prefix.payload()
    record = {'schema': SCHEMA, 'sha256': sha(payload), 'source_sha256': source['source_sha256'],
              'candidate_key_sha256': sha(prefix.candidate_key), 'snapshot_identity_sha256': sha(prefix.snapshot_identity)}
    if expected_record is not None:
        compare(expected_record, record, '$.serialized_prefix',
                producer_source=expected_record.get('source_sha256'), consumer_source=source['source_sha256'])
    cache = prefix.flat_cache
    if cache is not None:
        if isinstance(cache, np.ndarray):
            if cache.dtype != np.dtype('float32') or cache.shape != (FLAT_DIM,): raise ValueError('PUBLIC_PREFIX_FLAT_CACHE_DTYPE_SHAPE')
            cache = cache.tolist()
        if type(cache) is not list or len(cache) != FLAT_DIM: raise ValueError('PUBLIC_PREFIX_FLAT_CACHE_SHAPE')
        cache_sha256 = sha(cache)
        for index, ((path, value), cached) in enumerate(zip(_scalar_sources(prefix), cache)):
            compare(value, cached, path + f' (flat[{index}])', source['source_sha256'], cache_sha256)
    result = {'schema': SCHEMA, 'payload': payload, 'encoded': encoded(payload), 'sha256': record['sha256'], 'record': record}
    if validation_only:
        return result
    FLATTEN_CALLS += 1
    result['flat'] = np.asarray([value for _, value in _scalar_sources(prefix)], dtype=np.float32)
    return result


def capture_observation(observation, *, source_object, random_key, horizon=18.0):
    """Native producer: assemble a named snapshot, then use the sole serializer."""
    state = native_json(observation)
    source = {'schema': SCHEMA, 'phase': 'pre_action', 'generation_stage': 'pre_action_observation',
        'source_object': source_object, 'horizon': horizon, 'node_counts': list(COUNTS),
        'flat_dtype': 'float32', 'random_key_sha256': sha(random_key), 'source_sha256': ''}
    source['source_sha256'] = sha(source_material(state, source))
    state['public_prefix_source'] = source
    result = serialize_public_prefix(make_public_prefix(state))
    # Native callers retain array types; JSON boundaries use the same saved source.
    state['flat'] = result['flat']
    return state


def prefix_hashes(state, context, keys):
    for key in keys: serialize_public_prefix(make_public_prefix(state, context, key), validation_only=True)
    return [serialize_public_prefix(make_public_prefix(state, context, key))['sha256'] for key in keys]


def prefix_records(state, context, keys):
    for key in keys: serialize_public_prefix(make_public_prefix(state, context, key), validation_only=True)
    return [serialize_public_prefix(make_public_prefix(state, context, key))['record'] for key in keys]
