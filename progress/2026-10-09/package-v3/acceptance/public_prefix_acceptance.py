"""Production prefix integration; synthetic environment only, no neural execution."""
import ast
import copy
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'package'), str(ROOT / 'package/native'), str(ROOT / 'acceptance')]
if os.environ.get('W1_VERIFIED_ATTEMPT'): raise ValueError('FORMAL_IDENTITY_FORBIDDEN_IN_ACCEPTANCE')
os.sched_setaffinity(0, {0})
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
START = time.monotonic()
import numpy as np
import torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)
from controlled_environment import Environment
from production_driver import Driver
from budget_ledger import BudgetLedger, BudgetError
from phase_handshake import StageServer
from linux_process_scope import proc_tree
from infra_io import durable_atomic_json
from consequence_collection import jsonable, snapshot_payload, state_bytes, restore_candidate
from consequence_contract import digest, task_slots, admit_window
from consequence_training import serialize_prediction
from consequence_model import READOUT_DIM, AUX_DIM
from consequence_policy import ConsequencePolicy
from consequence_replay import verify_update
from candidate_keys import catalog, context_for, audit, feature_binding, verify_replay_candidate, validate_window
import public_prefix as pp

DIAG = Path(os.environ['W1_PUBLIC_PREFIX_ACCEPTANCE_OUTPUT'])
CASES = []
EXPECTED_FAILURES = []


def require(value):
    if not value: raise AssertionError('Required condition failed')


def case(name, function):
    started = time.monotonic(); function()
    CASES.append({'name': name, 'pass': True, 'wall_seconds': time.monotonic() - started})
    print('PASS ' + name, flush=True)


def rejects(name, function, *, no_flatten=True):
    before = pp.FLATTEN_CALLS
    try: function()
    except (ValueError, BudgetError) as exc:
        if no_flatten: require(pp.FLATTEN_CALLS == before)
        EXPECTED_FAILURES.append({'case': name, 'error': str(exc), 'new_flattens': pp.FLATTEN_CALLS - before})
        return str(exc)
    raise AssertionError('Invalid input accepted: ' + name)


class Fixture(Environment):
    def _observation(self):
        state = super()._observation()
        self.observation_calls = getattr(self, 'observation_calls', 0) + 1
        state['mask'][:24] = [False] * 24
        if self.clock.time < self.horizon and not self.noop:
            for u in range(4): state['mask'][u * 6 + (u + int(self.clock.time)) % self.n] = True
        for u in range(4):
            for t in range(6): state['graph']['relations'][u][t][2] = float(state['mask'][u * 6 + t])
        now = self.clock.time
        event = now in (2., 4., 8., 12.) and getattr(self, 'trigger_observed_at', None) != now
        self.trigger_observed_at = now
        state.update(event_signal=float(event), trigger_flags={'public_boundary': event})
        state.pop('flat'); state.pop('public_prefix_source')
        return pp.capture_observation(state, source_object='controlled_environment._observation.result', random_key=self._exogenous_key)


class Factory:
    def __call__(self, spec, config, purpose): return Fixture(spec, config, purpose)


@contextmanager
def handshake(output, request):
    own = resource.getrusage(resource.RUSAGE_SELF); child = resource.getrusage(resource.RUSAGE_CHILDREN)
    server = StageServer(output / 'phase.sock', request, tree_reader=proc_tree,
        cpu_origin_seconds=own.ru_utime + own.ru_stime + child.ru_utime + child.ru_stime,
        wall_origin_monotonic=START)
    os.environ['W1_PHASE_ACCOUNTING_SOCKET'] = str(server.path)
    stop = threading.Event(); errors = []
    def serve():
        while not stop.is_set():
            try: server.process_pending(os.getpid())
            except BaseException as exc: errors.append(repr(exc))
            stop.wait(.001)
    thread = threading.Thread(target=serve); thread.start()
    try: yield server
    finally:
        stop.set(); thread.join(timeout=6); server.close()
        os.environ.pop('W1_PHASE_ACCOUNTING_SOCKET', None)
        durable_atomic_json(output / 'handshake-result.json', {'errors': errors, 'rows': server.rows})
        require(not errors and not thread.is_alive())


def setup():
    DIAG.mkdir(parents=True, exist_ok=True)
    workspace = DIAG / 'controlled-workspace'; workspace.mkdir(exist_ok=False)
    request = json.loads((ROOT / 'RESOURCE_REQUEST.json').read_text())
    allowed = {'environment_constructions': 4, 'environment_resets': 4, 'environment_steps': 7200,
        'candidate_scans': 8000, 'candidate_branches': 400, 'snapshot_writes': 32,
        'forced_first_actions': 400, 'output_writes': 64, 'candidate_snapshot_probes': 401,
        'candidate_contract_audits': 2500}
    for stage, limits in request['stages'].items():
        for key, value in list(limits.items()):
            if type(value) is int: limits[key] = allowed.get(key, 0)
        limits.update(wall_seconds=240, complete_process_cpu_seconds=180,
                      all_resident_rss_bytes=4 * 2**30, active_storage_bytes=512 * 2**20)
    request['totals'] = dict(request['stages']['paired_collection'], wall_seconds=300, complete_process_cpu_seconds=240)
    request.update(attempt='controlled-public-prefix-only', approval_status='ENGINEERING_ONLY')
    specs = []
    for i, count in enumerate((3, 4, 6, 3)):
        scenario = {'seed': 970200 + i, 'task_count': count, 'noop_only': i == 3}
        specs.append({'parent': 'synthetic-prefix-parent-' + str(i), 'repeat': 0, 'split': 'train',
            'role': 'world_and_policy_train', 'scenario': scenario, 'scenario_sha256': digest(scenario),
            'exogenous_key': 'synthetic-prefix-key-' + str(i)})
    matrix = json.loads((ROOT / 'experiment-matrix.json').read_text())
    matrix.update(controlled=True, attempt='engineering-only-no-formal-identity')
    for name, value in [('parent-split.json', {'parents': specs}), ('RESOURCE_REQUEST.json', request),
        ('experiment-matrix.json', matrix), ('evaluation-tape.json', {'parents': []})]:
        durable_atomic_json(workspace / name, value)
    (workspace / 'environment-config-contract.json').write_bytes((ROOT / 'package/environment-config-contract.json').read_bytes())
    return workspace, request, matrix


def replay_row(driver, row):
    state, context, keys = row['state_before'], row['candidate_context'], row['candidate_keys']
    base, aux, valid, mask, available, trace = driver.features(state, 'T', candidate_context=context, candidate_keys=keys)
    pack = SimpleNamespace(arm='T', normalization_mean=torch.zeros(AUX_DIM), normalization_scale=torch.ones(AUX_DIM))
    packed = ConsequencePolicy.pack(pack, base, aux, valid, mask, available).numpy()
    return {'candidate_features': packed, 'mask': mask.numpy(), 'action': keys[0]['action_index'],
        'replay_evidence': {'candidate_state_before': state, 'candidate_context': context,
            'candidate_keys': keys, 'public_prefixes': trace['public_prefixes'],
            'candidate_feature_binding': feature_binding(keys, context, state, packed),
            'candidate_audits': dict(trace['candidate_audits'], replay=audit('replay', keys, context, state)),
            'counterfactual_identity': {'public_sha256': context['public_sha256']}}}


def main():
    workspace, request, matrix = setup(); output = workspace / 'controlled-output'; output.mkdir()
    with handshake(output, request) as server:
        ledger = BudgetLedger(output / 'ledger.sqlite', request)
        driver = Driver(workspace, output, matrix, ledger, Factory())
        driver.phase('input_admission')
        lean = driver.collect()
        case('actual_driver_collect_2_4_8_12_and_3_4_6_tasks', lambda: require(len(lean) == 12 and
            {r['state_before']['time'] for r in lean} == {2., 4., 8., 12.} and
            {len(r['state_before']['public_entity_ids']['tasks']) for r in lean} == {3, 4, 6}))
        inventory = json.loads((output / 'source-inventory.json').read_text())['windows']
        case('no_opportunity_is_normal', lambda: require(sum(r['status'] == 'no_opportunity' for r in inventory) == 4))
        row = next(r for r in lean if r['state_before']['time'] == 4.)
        state, context, keys = row['state_before'], row['candidate_context'], row['candidate_keys']
        case('first_time4_candidate_preserves_event_one_and_flat769', lambda: require(keys[0]['candidate_order'] == 0 and
            state['event_signal'] == state['flat'][769] == 1.0))
        audits = [json.loads(line) for line in (output / 'candidate-identity-audits.jsonl').read_text().splitlines()]
        probes = [r for r in audits if r.get('restoration_public_source')]
        case('all_restores_read_saved_snapshot_without_environment_observation', lambda: require(
            len(probes) == 60 and all(r['environment_observation_calls_during_probe'] == 0 and r['audited_branch_unmodified'] for r in probes)))
        case('all_legal_candidates_including_noop_and_multi_uav_slots', lambda: require(
            all(any(k['noop'] for k in r['candidate_keys']) for r in lean) and
            {k['uav_slot'] for r in lean for k in r['candidate_keys'] if not k['noop']} == {0, 1, 2, 3}))
        parts = driver.audit_window_input(row)
        expected_hashes = [r['sha256'] for r in row['public_prefixes']]
        predictions = [serialize_prediction(k, context, state, task_slots(state), [0.] * READOUT_DIM, [False] * READOUT_DIM) for k in keys]
        case('collector_label_graph_world_input_prediction_share_prefix', lambda: require(
            row['candidate_audits']['collector']['public_prefix_sha256'] == expected_hashes and
            row['candidate_audits']['label_lookup']['public_prefix_sha256'] == expected_hashes and
            parts[-1]['audit']['public_prefix_sha256'] == expected_hashes and
            [l['public_prefix']['sha256'] for l in row['labels']] == expected_hashes and
            [p['public_prefix']['sha256'] for p in predictions] == expected_hashes))
        case('unknown_collision_not_fabricated', lambda: require(not parts[5]['binary'][:, 4].any()))
        transition = replay_row(driver, row)
        case('replay_GPPO_preupdate_verifier_preserves_prefix', lambda: require(verify_replay_candidate(transition)['public_prefix_sha256'] == expected_hashes))
        producer = pp.make_public_prefix(state, context, keys[0])
        for value in (0., 1.):
            event_state = copy.deepcopy(state); event_state.update(event_signal=value, trigger_flags={'public_boundary': bool(value)})
            event_state.pop('flat'); event_state.pop('public_prefix_source')
            event_state = jsonable(pp.capture_observation(event_state, source_object='controlled_environment._observation.result', random_key='synthetic-events'))
            case('valid_event_' + str(int(value)), lambda s=event_state, v=value: require(
                pp.serialize_public_prefix(pp.make_public_prefix(s))['flat'][769] == v))
        for name, mutate in [
            ('event_value', lambda s: s.update(event_signal=0.)),
            ('event_type', lambda s: s.update(event_signal=1)),
            ('event_unknown', lambda s: s.update(event_signal_valid=False)),
            ('event_missing', lambda s: s.pop('event_signal')),
            ('event_bool_type', lambda s: s.update(event_signal=True)),
            ('pre_post_time_mix', lambda s: s.update(time=5.)),
            ('pre_post_phase_mix', lambda s: s['public_prefix_source'].update(phase='post_action')),
            ('graph_shape', lambda s: s['graph']['node_features'].pop()),
            ('flat769_cached_mismatch', lambda s: s['flat'].__setitem__(769, 0.)),
            ('snapshot_identity', lambda s: s['public_prefix_source'].update(source_sha256='0' * 64)),
        ]:
            def reject_state(name=name, mutate=mutate):
                damaged = copy.deepcopy(state); mutate(damaged)
                error = rejects(name, lambda: pp.serialize_public_prefix(pp.make_public_prefix(damaged, context, keys[0]), expected=producer))
                if name == 'flat769_cached_mismatch':
                    require('$.event_signal (flat[769])' in error and 'producer_source_sha256' in error and 'producer_value' in error)
            case(name + '_before_flatten', reject_state)
        for name, mutate in [
            ('parent', lambda k: k.update(parent='other-parent')),
            ('window', lambda k: k.update(window_id='time-8')),
            ('candidate_order', lambda k: k.update(candidate_order=1)),
            ('candidate_prefix', lambda k: k.update(prefix_sha256='0' * 64)),
            ('candidate_slot', lambda k: k.update(task_slot=5)),
            ('candidate_action', lambda k: k.update(action_index=24)),
        ]:
            def reject_key(name=name, mutate=mutate):
                key = copy.deepcopy(keys[0]); mutate(key)
                rejects(name, lambda: pp.serialize_public_prefix(pp.make_public_prefix(state, context, key), expected=producer))
            case(name + '_before_flatten', reject_key)
        for name, mutate in [
            ('candidate_sequence', lambda r: r['candidate_keys'].reverse()),
            ('missing_candidate', lambda r: r['candidate_keys'].pop()),
            ('duplicate_candidate', lambda r: r['candidate_keys'].__setitem__(1, r['candidate_keys'][0])),
            ('prefix_sequence', lambda r: r['public_prefixes'].reverse()),
            ('illegal_action', lambda r: r['candidate_keys'][0].update(action_index=0)),
        ]:
            def reject_window(name=name, mutate=mutate):
                damaged = copy.deepcopy(row); mutate(damaged)
                rejects(name, lambda: driver.audit_window_input(damaged))
            case(name + '_production_input_before_flatten', reject_window)
        for name, mutate in [
            ('mask', lambda r: r['mask'].__setitem__(0, True)),
            ('prefix_order', lambda r: r['replay_evidence']['public_prefixes'].reverse()),
            ('prefix_missing', lambda r: r['replay_evidence'].pop('public_prefixes')),
            ('event', lambda r: r['replay_evidence']['candidate_state_before']['flat'].__setitem__(769, 0.)),
        ]:
            def reject_replay(name=name, mutate=mutate):
                damaged = copy.deepcopy(transition); mutate(damaged)
                rejects('replay_' + name, lambda: verify_update(None, [damaged], torch.device('cpu')))
            case('actual_preupdate_' + name + '_before_flatten', reject_replay)
        # A saved synthetic snapshot can be restored even if the caller's current state advances.
        spec = driver.registry[0]
        payload_path = next((output / 'snapshots').glob('*.pickle'))
        snapshot = pickle.loads(payload_path.read_bytes())
        env, adapter, history = snapshot['environment']
        saved_state, saved_context, saved_keys = snapshot['state_before'], snapshot['context'], snapshot['keys']
        canonical = state_bytes((env, adapter, history)); public_reads = env.observation_calls
        restored = ledger.call('controlled_saved_restore', {'candidate_branches': 1}, restore_candidate,
            payload_path.read_bytes(), canonical, saved_state, saved_context, saved_keys, saved_keys[0], ledger,
            driver.record_candidate_audit, expected_payload_sha256=digest(payload_path.read_bytes()))
        case('saved_restore_never_reads_trigger_or_post_state', lambda: require(
            state_bytes(restored) == canonical and restored[0].observation_calls == public_reads))
        fired = []
        case('real_ledger_rejects_unknown_key', lambda: rejects('unknown_key', lambda:
            ledger.call('forbidden_unknown', {'not_in_contract': 1}, lambda: fired.append(True))))
        case('real_ledger_neural_budget_is_zero', lambda: rejects('neural_budget', lambda:
            ledger.call('forbidden_forward', {'world_model_batch_forwards': 1}, lambda: fired.append(True))))
        require(not fired)
        case('evaluation_remains_locked', lambda: rejects('evaluation_locked', driver.evaluation_specs))
        settled = ledger.assert_settled()
        driver.phase('analysis_and_settlement'); driver.phase('settlement_and_verified_export'); ledger.close()
        require(settled['pending_calls'] == 0 and server.stage == 'settlement_and_verified_export')
    source = (ROOT / 'package/consequence_collection.py').read_text()
    restore = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'restore_candidate')
    case('restore_has_no_environment_observation_or_trigger_read_static', lambda: require(
        not any(isinstance(n, ast.Attribute) and n.attr in ('_observation', '_trigger_flags', 'clock') for n in ast.walk(restore))))
    duration = time.monotonic() - START
    require(duration <= 300 and time.process_time() <= 240)
    summary = {'pass': True, 'tests_passed': len(CASES), 'tests': CASES,
        'actual_driver_and_ledger_and_handshake': True, 'synthetic_environment_only': True,
        'formal_environment_steps': 0, 'formal_checkpoint_loads': 0, 'model_initializations': 0,
        'neural_forwards': 0, 'world_updates': 0, 'GPPO_updates': 0, 'task_evaluation_episodes': 0,
        'synthetic_environment_steps': settled['totals'].get('environment_steps', 0), 'ledger': settled,
        'wall_seconds_including_import': duration, 'CPU_seconds_including_import': time.process_time(),
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'interpreter': sys.executable, 'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__,
        'flatten_calls': pp.FLATTEN_CALLS, 'GPPO_scope': 'actual preupdate identity verifier only; no GPPO or model computation'}
    durable_atomic_json(DIAG / 'acceptance-summary.json', summary)
    durable_atomic_json(DIAG / 'expected-rejections.json', EXPECTED_FAILURES)
    print(json.dumps({k: summary[k] for k in ('pass', 'tests_passed', 'neural_forwards', 'GPPO_updates', 'wall_seconds_including_import')}))


if __name__ == '__main__':
    try: main()
    except BaseException as exc:
        import traceback
        p = DIAG / 'acceptance-first-error.json'
        if not p.exists(): durable_atomic_json(p, {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc(), 'completed_tests': CASES})
        raise
