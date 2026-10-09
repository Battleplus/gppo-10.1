"""Strict JSON candidate identities shared by every production boundary."""
import hashlib
import json
import math

SCHEMA = 'w1-structured-candidate/1'
CONTEXT_SCHEMA = 'w1-candidate-context/1'
CONTINUATION = 'public-joint-opportunity-rule/1'
CONTEXT_FIELDS = {'schema', 'parent', 'repeat', 'window_id', 'time', 'public_sha256',
                  'prefix_sha256', 'continuation_id', 'uav_ids', 'task_ids', 'legal_actions'}
KEY_FIELDS = {'schema', 'parent', 'repeat', 'window_id', 'time', 'public_sha256', 'prefix_sha256',
              'action_index', 'noop', 'candidate_order', 'uav_slot', 'uav_id', 'task_slot', 'task_id'}


def encoded(value):
    def check(x, path):
        if type(x) is dict:
            if any(type(k) is not str for k in x):
                raise ValueError('CANDIDATE_JSON_KEY_TYPE:' + path)
            for k, v in x.items():
                check(v, path + '.' + k)
        elif type(x) is list:
            for i, v in enumerate(x):
                check(v, path + '[' + str(i) + ']')
        elif x is None or type(x) in (str, bool, int):
            pass
        elif type(x) is float and math.isfinite(x):
            pass
        else:
            raise ValueError('CANDIDATE_JSON_TYPE:' + path + ':' + type(x).__name__)
    check(value, '$')
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')


def key_digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def _same(actual, expected, path):
    if type(actual) is not type(expected):
        raise ValueError('CANDIDATE_TYPE:' + path + ':' + type(actual).__name__ + '!=' + type(expected).__name__)
    if type(expected) is dict:
        if set(actual) != set(expected):
            raise ValueError('CANDIDATE_FIELDS:' + path)
        for k in sorted(expected):
            _same(actual[k], expected[k], path + '.' + k)
    elif type(expected) is list:
        if len(actual) != len(expected):
            raise ValueError('CANDIDATE_LENGTH:' + path)
        for i, v in enumerate(expected):
            _same(actual[i], v, path + '[' + str(i) + ']')
    elif actual != expected:
        raise ValueError('CANDIDATE_VALUE:' + path + ':' + repr(actual) + '!=' + repr(expected))


def context_for(parent, repeat, window_id, state):
    from public_prefix import make_public_prefix, serialize_public_prefix
    serialize_public_prefix(make_public_prefix(state), validation_only=True)
    from consequence_contract import task_slots, legal_actions, digest
    if type(parent) is not str or not parent or type(window_id) is not str or not window_id:
        raise ValueError('CANDIDATE_PARENT_WINDOW')
    if type(repeat) is not int or repeat < 0:
        raise ValueError('CANDIDATE_REPEAT')
    if type(state['time']) is not float or not math.isfinite(state['time']) or state['time'] < 0:
        raise ValueError('CANDIDATE_TIME_FLOAT_REQUIRED')
    task_slots(state)
    ids = state['public_entity_ids']
    for field in ('tasks', 'uavs'):
        if type(ids[field]) is not list or any(type(x) is not str or not x for x in ids[field]):
            raise ValueError('CANDIDATE_PUBLIC_IDS_LIST_REQUIRED:' + field)
    prefix = {'parent': parent, 'repeat': repeat, 'window_id': window_id, 'time': state['time'],
              'public_sha256': digest(state), 'continuation_id': CONTINUATION}
    return dict(prefix, schema=CONTEXT_SCHEMA, prefix_sha256=key_digest(prefix),
                uav_ids=ids['uavs'].copy(), task_ids=ids['tasks'].copy(), legal_actions=legal_actions(state['mask']))


def validate_context(context, state):
    if type(context) is not dict or set(context) != CONTEXT_FIELDS:
        raise ValueError('CANDIDATE_CONTEXT_FIELDS')
    expected = context_for(context['parent'], context['repeat'], context['window_id'], state)
    _same(context, expected, 'context')
    return context


def catalog(context, state):
    validate_context(context, state)
    keys = []
    for order, action in enumerate(context['legal_actions']):
        noop = action == 24
        u, t = (None, None) if noop else divmod(action, 6)
        key = {k: context[k] for k in ('parent', 'repeat', 'window_id', 'time', 'public_sha256', 'prefix_sha256')}
        key.update(schema=SCHEMA, action_index=action, noop=noop, candidate_order=order,
                   uav_slot=u, task_slot=t, uav_id=None if noop else context['uav_ids'][u],
                   task_id=None if noop else context['task_ids'][t])
        keys.append(key)
    return keys


def validate_catalog(keys, context, state):
    expected = catalog(context, state)
    _same(keys, expected, 'candidate_keys')
    return [key_digest(k) for k in keys]


def validate_key(key, context, state):
    if type(key) is not dict or set(key) != KEY_FIELDS or type(key.get('candidate_order')) is not int:
        raise ValueError('CANDIDATE_KEY_FIELDS')
    expected = catalog(context, state)
    order = key['candidate_order']
    if not 0 <= order < len(expected):
        raise ValueError('CANDIDATE_ORDER_RANGE')
    _same(key, expected[order], 'candidate_key')
    return key_digest(key)


def audit(layer, keys, context, state):
    from public_prefix import prefix_hashes, SCHEMA as PREFIX_SCHEMA
    hashes = validate_catalog(keys, context, state)
    return {'layer': layer, 'context_sha256': key_digest(context), 'candidate_key_sha256': hashes,
            'candidate_sequence_sha256': key_digest(keys), 'public_prefix_schema': PREFIX_SCHEMA,
            'public_prefix_sha256': prefix_hashes(state, context, keys),
            'public_source_sha256': state['public_prefix_source']['source_sha256']}


def validate_audit(record, keys, context, state, layer):
    _same(record, audit(layer, keys, context, state), 'candidate_audit')


def validate_window(row, *, labels_required=True):
    from public_prefix import make_public_prefix, serialize_public_prefix
    state, context, keys = row['state_before'], row['candidate_context'], row['candidate_keys']
    validate_catalog(keys, context, state)
    saved = row.get('public_prefixes')
    if type(saved) is not list or len(saved) != len(keys): raise ValueError('PUBLIC_PREFIX_WINDOW_RECORDS')
    for key, record in zip(keys, saved):
        serialize_public_prefix(make_public_prefix(state, context, key), expected_record=record, validation_only=True)
    for name in ('parent', 'repeat', 'window_id'):
        _same(row[name], context[name], 'window.' + name)
    if row['status'] == 'no_opportunity':
        if row.get('branches') or row.get('labels'):
            raise ValueError('CANDIDATE_NO_OPPORTUNITY_BRANCHES')
        validate_audit(row['candidate_audits']['collector'], keys, context, state, 'collector')
        return keys
    if row['status'] != 'complete':
        raise ValueError('CANDIDATE_INCOMPLETE_WINDOW')
    branches = row['branches']
    if type(branches) is not list or len(branches) != len(keys):
        raise ValueError('CANDIDATE_BRANCH_COUNT')
    for key, branch in zip(keys, branches):
        _same(branch['candidate_key'], key, 'branch.candidate_key')
        _same(branch['action'], key['action_index'], 'branch.action')
        if branch['candidate_key_sha256'] != key_digest(key) or branch['action'] != key['action_index']:
            raise ValueError('CANDIDATE_BRANCH_KEY_ACTION')
        validate_audit(branch['candidate_snapshot_audit'], keys, context, state, 'branch_snapshot')
    validate_audit(row['candidate_audits']['collector'], keys, context, state, 'collector')
    if labels_required:
        if type(row['labels']) is not list or len(row['labels']) != len(keys):
            raise ValueError('CANDIDATE_LABEL_COUNT')
        for key, label in zip(keys, row['labels']):
            _same(label['candidate_key'], key, 'label.candidate_key')
            if label['candidate_key_sha256'] != key_digest(key):
                raise ValueError('CANDIDATE_LABEL_DIGEST')
            serialize_public_prefix(make_public_prefix(state, context, key), expected_record=label['public_prefix'], validation_only=True)
        validate_audit(row['candidate_audits']['label_lookup'], keys, context, state, 'label_lookup')
    return keys


def tensor_digest(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().contiguous().numpy()
    if not hasattr(value, 'dtype') or not hasattr(value, 'shape') or not hasattr(value, 'tobytes'):
        raise ValueError('CANDIDATE_TENSOR_REQUIRED')
    header = {'dtype': str(value.dtype), 'shape': [int(n) for n in value.shape]}
    return hashlib.sha256(encoded(header) + value.tobytes(order='C')).hexdigest()


def bind_model_input(state, context, keys, nodes, history, actions, relations):
    validate_catalog(keys, context, state)
    if actions.tolist() != context['legal_actions']:
        raise ValueError('CANDIDATE_GRAPH_ACTION_ORDER')
    import torch
    from gppo_world.graph5 import graph5_from_m10_observation
    from public_history import history_vector
    if actions.dtype != torch.long or history.dtype != torch.float32 or relations.dtype != torch.float32 or any(v.dtype != torch.float32 for v in nodes.values()):
        raise ValueError('CANDIDATE_MODEL_INPUT_DTYPE')
    graph = graph5_from_m10_observation(state)
    expected_nodes = {k: v[None].expand(len(keys), -1, -1) for k, v in graph.nodes.items()}
    expected_relations = torch.cat((graph.candidate_features, torch.zeros(1, 4)))[actions.cpu()]
    expected_history = torch.tensor(history_vector(state))[None].expand(len(keys), -1)
    if set(nodes) != set(expected_nodes) or any(not torch.equal(v.cpu(), expected_nodes[k]) for k, v in nodes.items()):
        raise ValueError('CANDIDATE_GRAPH_NODE_MAPPING')
    if not torch.equal(relations.cpu(), expected_relations) or not torch.equal(history.cpu(), expected_history):
        raise ValueError('CANDIDATE_GRAPH_RELATION_HISTORY_MAPPING')
    # Identity is carried beside tensors and never appended to learned public inputs.
    return {'schema': SCHEMA, 'state_before': state, 'context': context, 'keys': keys,
            'audit': audit('world_model_input', keys, context, state),
            'tensor_hashes': {'nodes': {k: tensor_digest(v) for k, v in nodes.items()},
                             'history': tensor_digest(history), 'actions': tensor_digest(actions),
                             'relations': tensor_digest(relations)}}


def verify_model_inputs(bindings, nodes, history, actions, relations):
    if type(bindings) is not list or not bindings:
        raise ValueError('CANDIDATE_MODEL_BINDING_REQUIRED')
    offset = 0
    for binding in bindings:
        n = len(binding['keys'])
        end = offset + n
        ns = {k: v[offset:end] for k, v in nodes.items()}
        expected = bind_model_input(binding['state_before'], binding['context'], binding['keys'],
                                    ns, history[offset:end], actions[offset:end], relations[offset:end])
        _same(binding, expected, 'model_binding')
        offset = end
    if offset != len(actions) or len(history) != offset or len(relations) != offset or any(len(v) != offset for v in nodes.values()):
        raise ValueError('CANDIDATE_MODEL_BATCH_LENGTH')


def feature_binding(keys, context, state, features):
    return {'audit': audit('candidate_features', keys, context, state), 'features_sha256': tensor_digest(features)}


def verify_replay_candidate(transition):
    from public_prefix import make_public_prefix, serialize_public_prefix
    evidence = transition['replay_evidence']
    state, context, keys = evidence['candidate_state_before'], evidence['candidate_context'], evidence['candidate_keys']
    validate_catalog(keys, context, state)
    records = evidence.get('public_prefixes')
    if type(records) is not list or len(records) != len(keys): raise ValueError('PUBLIC_PREFIX_REPLAY_RECORDS')
    for key, record in zip(keys, records):
        serialize_public_prefix(make_public_prefix(state, context, key), expected_record=record, validation_only=True)
    if str(transition['mask'].dtype) != 'bool' or type(transition['action']) is not int or transition['mask'].tolist() != state['mask'] or transition['action'] not in context['legal_actions']:
        raise ValueError('CANDIDATE_REPLAY_MASK_ACTION')
    _same(evidence['candidate_feature_binding'], feature_binding(keys, context, state, transition['candidate_features']), 'replay.feature_binding')
    validate_audit(evidence['candidate_audits']['replay'], keys, context, state, 'replay')
    for layer, record in evidence['candidate_audits'].items():
        validate_audit(record, keys, context, state, layer)
    if evidence['counterfactual_identity']['public_sha256'] != context['public_sha256']:
        raise ValueError('CANDIDATE_COUNTERFACTUAL_PUBLIC_IDENTITY')
    return audit('GPPO', keys, context, state)
