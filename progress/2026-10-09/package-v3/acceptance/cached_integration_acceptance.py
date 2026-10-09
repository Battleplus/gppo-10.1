"""Final consumer validation on the already collected synthetic cache; no recollection."""
import copy
import gzip
import json
import os
from pathlib import Path
import time
import public_prefix_acceptance as harness
from public_prefix_acceptance import Fixture
from public_controller import public_copy
from consequence_replay import verify_update
from candidate_keys import verify_replay_candidate
from consequence_contract import admit_window
import public_prefix as pp
from consequence_collection import snapshot_payload, restore_candidate, state_bytes
from consequence_collection import collect_window

SOURCE = Path(os.environ['W1_SYNTHETIC_PREFIX_CACHE'])
assert json.loads((SOURCE.parent / 'RESOURCE_REQUEST.json').read_text())['approval_status'] == 'ENGINEERING_ONLY'
START = time.monotonic()
FORBIDDEN = []
def forbidden(*args, **kwargs):
    FORBIDDEN.append(True)
    raise AssertionError('NEURAL_EXECUTION_FORBIDDEN')
harness.torch.load = forbidden
harness.torch.nn.Module._call_impl = forbidden
workspace, request, matrix = harness.setup()
for stage in request['stages'].values():
    for key in ('environment_constructions', 'environment_resets', 'environment_steps', 'world_model_batch_forwards', 'world_model_updates', 'policy_updates', 'task_episodes'):
        stage[key] = 0
for key in ('environment_constructions', 'environment_resets', 'environment_steps', 'world_model_batch_forwards', 'world_model_updates', 'policy_updates', 'task_episodes'):
    request['totals'][key] = 0
harness.durable_atomic_json(workspace / 'RESOURCE_REQUEST.json', request)
output = workspace / 'controlled-output'; output.mkdir()
with harness.handshake(output, request) as server:
    ledger = harness.BudgetLedger(output / 'ledger.sqlite', request)
    driver = harness.Driver(workspace, output, matrix, ledger, harness.Factory())
    driver.phase('input_admission'); driver.phase('paired_collection')
    tested = []
    inventory = json.loads((SOURCE / 'source-inventory.json').read_text())['windows']
    for entry in inventory:
        path = SOURCE / entry['path']
        import hashlib
        with path.open('rb') as f: actual = hashlib.file_digest(f, 'sha256').hexdigest()
        assert actual == entry['sha256'] and path.stat().st_size == entry['bytes']
        with gzip.open(path, 'rt') as stream: row = json.load(stream)
        derived = admit_window(row, driver.splits)
        public_copy(row['state_before'])
        if row['status'] == 'no_opportunity': continue
        assert derived == row['labels']
        pieces = driver.audit_window_input(row)
        transition = harness.replay_row(driver, row)
        verify_replay_candidate(transition)
        state, context, keys = row['state_before'], row['candidate_context'], row['candidate_keys']
        typed = copy.deepcopy(state)
        typed['public_entity_ids']['tasks'] = tuple(typed['public_entity_ids']['tasks'])
        harness.rejects('driver_features_tuple_before_conversion', lambda: driver.features(typed, 'T', candidate_context=context, candidate_keys=keys))
        harness.rejects('collector_tuple_before_conversion', lambda: collect_window(None, typed,
            adapter=None, history=None, parent=row['parent'], repeat=0, split=row['split'], window_id=row['window_id'],
            ledger=ledger, snapshot_store=None, evidence_store=None))
        damaged = copy.deepcopy(state); damaged['flat'][769] = 1. - damaged['flat'][769]
        before = pp.FLATTEN_CALLS
        error = harness.rejects('cached_flat769', lambda: public_copy(damaged))
        assert pp.FLATTEN_CALLS == before and '$.event_signal (flat[769])' in error
        damaged_replay = copy.deepcopy(transition)
        damaged_replay['replay_evidence']['candidate_state_before']['flat'][769] = damaged['flat'][769]
        harness.rejects('cached_GPPO_prefix', lambda: verify_update(None, [damaged_replay], harness.torch.device('cpu')))
        tested.append({'parent': row['parent'], 'time': state['time'], 'tasks': len(state['public_entity_ids']['tasks']),
            'pass': True, 'world_input_candidate_count': len(keys), 'flat769_rejection': error})
    import pickle
    candidates = []
    for path in (SOURCE / 'snapshots').glob('*.pickle'):
        payload_bytes = path.read_bytes()
        assert hashlib.sha256(payload_bytes).hexdigest() == path.stem
        # The synthetic fixture was saved by an engineering __main__ entry.
        # Its exact class is imported above; production snapshots use module paths.
        snapshot = pickle.loads(payload_bytes)
        if snapshot['state_before']['time'] == 4.: candidates.append(snapshot)
    assert len(candidates) == 3
    for snapshot in candidates:
        env, adapter, history = snapshot['environment']
        state, context, keys = snapshot['state_before'], snapshot['context'], snapshot['keys']
        payload = snapshot_payload(env, adapter, history, state, context, keys)
        canonical = state_bytes((env, adapter, history))
        restored = ledger.call('cached_time4_restore', {'candidate_branches': 1}, restore_candidate,
            payload, canonical, state, context, keys, keys[0], ledger, driver.record_candidate_audit,
            expected_payload_sha256=harness.digest(payload))
        assert state_bytes(restored) == canonical
        post = copy.deepcopy(state); post['time'] += 1.
        harness.rejects('snapshot_post_time', lambda: snapshot_payload(env, adapter, history, post, context, keys))
        wrong_rng = copy.deepcopy(env); wrong_rng._exogenous_key = 'different-synthetic-key'
        harness.rejects('snapshot_random_key', lambda: snapshot_payload(wrong_rng, adapter, history, state, context, keys))
    snapshot = ledger.assert_settled()
    driver.phase('analysis_and_settlement'); driver.phase('settlement_and_verified_export'); ledger.close()
assert not FORBIDDEN and len(tested) == 12
report = {'pass': True, 'cache_source': str(SOURCE), 'windows_verified': len(inventory),
    'final_time4_snapshot_restores': len(candidates), 'snapshot_time_and_random_key_rejections': True,
    'collector_and_driver_type_guard_before_conversion': True,
    'opportunity_windows': len(tested), 'rows': tested, 'ledger': snapshot,
    'model_initializations': 0, 'checkpoint_loads': 0, 'neural_forwards': 0, 'environment_steps': 0,
    'world_updates': 0, 'GPPO_updates': 0, 'task_episodes': 0,
    'wall_seconds_after_import': time.monotonic() - START, 'CPU_seconds_including_import': time.process_time(),
    'scope': 'final code reuses admitted synthetic cache; no new environment or research data'}
harness.durable_atomic_json(harness.DIAG / 'cached-final-acceptance-summary.json', report)
print(json.dumps({k: report[k] for k in ('pass', 'windows_verified', 'opportunity_windows', 'neural_forwards', 'environment_steps')}))
