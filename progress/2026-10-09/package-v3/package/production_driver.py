"""Full consequence experiment. Only the environment factory is replaceable in acceptance."""
import copy
import gzip
import json
import math
import os
import shutil
import time
import traceback
from dataclasses import asdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'native'))
import numpy as np
import torch
from budget_ledger import BudgetLedger
from infra_io import canonical, durable_atomic_json, durable_append_jsonl, sha256_file
from classical_baselines import PublicDecisionAdapter, ClassicalSelector
from consequence_contract import digest, admit_window, opportunity, SEEDS, BINARY, CONTINUOUS
from consequence_collection import collect_window, choose_public, lifecycle, jsonable, state_bytes
from consequence_features import build_features, SUPPORT_TIMES
from consequence_model import ConsequenceJEPA, FrozenEnsemble, READOUT_DIM, AUX_DIM, PACKED_DIM
from consequence_training import train_final_world, admit_checkpoint, serialize_prediction, restore_optimizer_strict, prepare_world_input
from consequence_policy import ConsequencePolicy, FEATURE_ID
import consequence_replay as replay
from public_history import CausalPublicHistory
from offline_metrics import window_metrics, summarize, decide_gate
from gppo_world.m10_environment import M10Config, M10Environment, scenario_from_dict
from gppo_world.joint_gppo import JointGraphPreferencePolicy, JointTrainConfig
from gppo_world.joint_training import _vector_reward, ppo_preference_update, _gae_vector
from candidate_keys import (SCHEMA as CANDIDATE_SCHEMA, context_for, catalog, audit, validate_window,
                            validate_audit, validate_catalog, feature_binding, key_digest)

POLICY_SEEDS = (8301, 8302, 8303)
ARMS = ('G0', 'T', 'G1')


class NativeEnvironmentFactory:
    def __call__(self, spec, config, purpose):
        return M10Environment(config, scenario_from_dict(spec['scenario']), exogenous_key=spec['exogenous_key'])


class Driver:
    def __init__(self, root, output, matrix, ledger, env_factory=None):
        self.root, self.output, self.matrix, self.ledger = Path(root), Path(output), matrix, ledger
        self.env_factory = env_factory or NativeEnvironmentFactory()
        if not matrix.get('controlled') and type(self.env_factory) is not NativeEnvironmentFactory:
            raise ValueError('FORMAL_ENVIRONMENT_SUBSTITUTION_FORBIDDEN')
        self.config = M10Config(**json.loads((self.root/'environment-config-contract.json').read_text())['config'])
        self.registry = json.loads((self.root/'parent-split.json').read_text())['parents']
        self.splits = {s['parent']: s['split'] for s in self.registry}
        self.support = None; self.ensemble = None; self.normalization = None
        self.evaluation_open = False; self.final_policies = {}; self.initial_hashes = {}
        self.prediction_cache = {}; self.last_route = None
        self.last_preparation_cost = (0., 0.)
        self.pending_history_cost = (0., 0.)
        self.active_parent = None; self.active_episode = -1
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()

    def phase(self, stage):
        self.ledger.select(stage)
        self.progress(stage=stage)

    def progress(self, **fields):
        path = self.output/'activity.json'
        old = json.loads(path.read_text()) if path.exists() else {}
        old.update(fields, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                   updated_monotonic=time.monotonic(), formal_attempt=not self.matrix.get('controlled', False))
        durable_atomic_json(path, old)

    def save(self, name, obj):
        """Charge only a real new formal output, then verify the atomic result.

        Re-emitting identical content is an idempotent observation and does not
        consume another output-write unit. A changed existing artifact is a
        contract collision and is rejected before touching the file.
        """
        path = self.output / name
        expected = (canonical(obj) + "\n").encode("utf-8")
        if path.exists():
            actual = path.read_bytes()
            if actual != expected:
                raise ValueError("FORMAL_OUTPUT_IDENTITY_COLLISION:" + str(path.name))
            return {"status": "already_present", "name": name,
                    "bytes": len(actual), "sha256": sha256_file(path)}

        def write_checked():
            durable_atomic_json(path, obj)
            if not path.is_file():
                raise IOError("FORMAL_OUTPUT_NOT_PUBLISHED:" + str(path.name))
            actual_bytes = path.read_bytes()
            actual_sha = sha256_file(path)
            if len(actual_bytes) != len(expected) or actual_bytes != expected:
                raise IOError("FORMAL_OUTPUT_POSTWRITE_MISMATCH:" + str(path.name))
            return {"status": "written", "name": name,
                    "bytes": len(actual_bytes), "sha256": actual_sha}

        return self.ledger.call('json_output', {'output_writes': 1}, write_checked)

    def create_episode(self, spec, purpose):
        self.active_parent = spec['parent']; self.active_episode += 1
        env = self.ledger.call('environment_construct', {'environment_constructions': 1},
                               self.env_factory, spec, self.config, purpose)
        observation = self.ledger.call('environment_reset', {'environment_resets': 1}, env.reset)
        started=time.monotonic();cpu=time.process_time()
        adapter = PublicDecisionAdapter(); history = CausalPublicHistory(); history.append(observation)
        state, mask = adapter.prepare(observation); state['mask'] = mask
        result=history.attach_current(state)
        self.last_preparation_cost=(time.process_time()-cpu,time.monotonic()-started)
        self.pending_history_cost=(0.,0.)
        return env, adapter, history, result

    def append_history(self, history, observation):
        started=time.monotonic();cpu=time.process_time()
        history.append(observation)
        self.pending_history_cost=(time.process_time()-cpu,time.monotonic()-started)

    def advance(self, env, adapter, history, action, *, label):
        adapter.commit(int(action))
        nxt, scalar, done, info = self.ledger.call(label, {'environment_steps': 1}, env.step, int(action))
        self.append_history(history,nxt)
        return nxt, scalar, done, info

    def prepare_next(self, adapter, history, nxt):
        started=time.monotonic();cpu=time.process_time()
        state, mask = adapter.prepare(nxt); state['mask'] = mask
        result=history.attach_current(state)
        history_cpu,history_wall=self.pending_history_cost
        self.last_preparation_cost=(time.process_time()-cpu+history_cpu,time.monotonic()-started+history_wall)
        self.pending_history_cost=(0.,0.)
        return result

    def snapshot_store(self, identity, payload, audit):
        folder = self.output/'snapshots'; folder.mkdir(exist_ok=True)
        p = folder/(identity+'.pickle')
        if p.exists() and p.read_bytes() != payload: raise ValueError('SNAPSHOT_IDENTITY_COLLISION')
        if not p.exists():
            temporary = p.with_suffix('.pending'); temporary.write_bytes(payload); temporary.replace(p)
            durable_atomic_json(p.with_suffix('.json'), audit)

    def compact(self, row):
        validate_window(row)
        return {k: copy.deepcopy(v) for k, v in row.items() if k != 'branches'} | {
            'admitted_lean': True, 'branches': [{'action': b['action'], 'candidate_key': b['candidate_key'],
                'candidate_key_sha256': b['candidate_key_sha256'], 'candidate_snapshot_audit': b['candidate_snapshot_audit'],
                'trace_sha256': b['trace_sha256'], 'snapshot_ref': b['snapshot_ref'],
                'final_public': b['trace'][-1]['public_after']} for b in row['branches']]}

    def record_candidate_audit(self, record):
        return self.ledger.call('candidate_identity_audit', {'candidate_contract_audits': 1},
            durable_append_jsonl, self.output/'candidate-identity-audits.jsonl', record)

    def audit_window_input(self, row):
        """Actual production pre-model entry, also callable by no-forward acceptance."""
        validate_window(row)
        pieces = prepare_world_input(row)
        graph_audit = audit('graph_action_encoding', row['candidate_keys'], row['candidate_context'], row['state_before'])
        self.record_candidate_audit(graph_audit)
        self.record_candidate_audit(pieces[-1]['audit'])
        return pieces

    def collect(self):
        if self.matrix.get('recovery_mode') == 'reuse_verified_windows':
            from consequence_contract import admit_window
            source = self.root/'recovery-inputs'
            manifest = json.loads((source/'window-cache-manifest.json').read_text())
            if manifest.get('schema') != 'w1-recovery-window-cache/1' or manifest.get('window_count') != 288:
                raise ValueError('RECOVERY_CACHE_MANIFEST')
            lean = []
            inventory = []
            cache_files = manifest.get('files', [])
            if len(cache_files) != 288:
                raise ValueError('RECOVERY_CACHE_FILE_COUNT')
            expected = {(s['parent'], f'time-{t:g}') for s in self.registry
                        if s['role'] != 'policy_task_evaluation' for t in SUPPORT_TIMES}
            seen = set()
            for item in cache_files:
                rel = Path(item['path'])
                if rel.is_absolute() or '..' in rel.parts or rel.suffixes[-2:] != ['.json', '.gz']:
                    raise ValueError('RECOVERY_CACHE_PATH')
                path = source/rel
                def load_admit():
                    raw = path.read_bytes()
                    if len(raw) != item['bytes'] or sha256_file(path) != item['sha256']:
                        raise ValueError('RECOVERY_WINDOW_SHA256:' + str(rel))
                    row = json.loads(gzip.decompress(raw))
                    identity = (row['parent'], row['window_id'])
                    if identity != (item['parent'], item['window_id']) or identity in seen:
                        raise ValueError('RECOVERY_WINDOW_IDENTITY:' + str(identity))
                    if len(row.get('candidate_keys', [])) != item['candidate_count']:
                        raise ValueError('RECOVERY_CANDIDATE_COUNT:' + str(identity))
                    if row['split'] != item['split'] or row['status'] != item['status']:
                        raise ValueError('RECOVERY_WINDOW_REGISTRY:' + str(identity))
                    if (row['parent'], row['repeat'], row['window_id']) not in {
                            (x['parent'], 0, f'time-{t:g}') for x in self.registry
                            if x['role'] != 'policy_task_evaluation' for t in SUPPORT_TIMES}:
                        raise ValueError('RECOVERY_WINDOW_PARENT_OR_TIME')
                    validate_window(row)
                    labels = admit_window(row, self.splits)
                    if row['status'] == 'complete' and row.get('labels') != labels:
                        raise ValueError('RECOVERY_LABEL_RECOMPUTE')
                    if row['status'] == 'no_opportunity' and (row.get('branches') or labels):
                        raise ValueError('RECOVERY_NO_OPPORTUNITY_NOT_EMPTY')
                    return row
                row = self.ledger.call('reused_window_admission',
                    {'reused_windows': 1, 'candidate_contract_audits': 1 + item['candidate_count']}, load_admit)
                seen.add((row['parent'], row['window_id']))
                if row['status'] == 'complete':
                    compact = self.compact(row)
                    compact['source'] = {k: item[k] for k in ('parent', 'window_id', 'split', 'status', 'bytes', 'sha256')}
                    lean.append(compact)
                inventory.append({k: item[k] for k in ('parent', 'window_id', 'split', 'status', 'bytes', 'sha256')})
            if seen != expected or len(inventory) != len(expected):
                raise ValueError('RECOVERY_CACHE_COVERAGE')
            self.save('reused-source-inventory.json', {'windows': inventory, 'source_manifest_sha256': sha256_file(source/'window-cache-manifest.json')})
            self.save('lean-inputs.json', {'complete': lean,
                'no_opportunity': [x for x in inventory if x['status'] == 'no_opportunity'],
                'source': 'verified_v5_small_window_cache', 'collector_called': False})
            return lean
        self.phase('paired_collection'); lean = []
        raw_dir = self.output/'windows'; raw_dir.mkdir()
        expected = []
        for spec in self.registry:
            if spec['role'] == 'policy_task_evaluation': continue
            env, adapter, history, state = self.create_episode(spec, 'paired_collection')
            done = False
            for scheduled in SUPPORT_TIMES:
                while state['time'] < scheduled and not done:
                    action = self.ledger.call('prefix_scan', {'candidate_scans': 1}, choose_public, state)
                    nxt, _, done, info = self.advance(env, adapter, history, action, label='prefix_step')
                    state = self.prepare_next(adapter, history, nxt)
                window_id = f'time-{scheduled:g}'
                if done and opportunity(state['mask']): raise ValueError('TERMINAL_STATE_HAS_ALLOCATION_OPPORTUNITY')
                row = collect_window(env, state, adapter=adapter, history=history, parent=spec['parent'], repeat=0,
                    split=spec['split'], window_id=window_id, ledger=self.ledger, snapshot_store=self.snapshot_store,
                    evidence_store=self.record_candidate_audit)
                row['scheduled_time'] = scheduled
                row['terminal_before_scheduled_prefix'] = bool(done and state['time'] < scheduled)
                admitted = admit_window(row, self.splits)
                if row['status'] == 'complete':
                    for b in row['branches']:
                        snap = self.output/'snapshots'/(b['snapshot_ref']+'.pickle')
                        payload = snap.read_bytes()
                        if digest(payload) != b['snapshot_ref']: raise ValueError('SNAPSHOT_FILE_DIGEST')
                        import pickle
                        restored = pickle.loads(payload)
                        if digest(state_bytes(restored['environment'])) != b['prefix_sha256']: raise ValueError('SNAPSHOT_STATE_DIGEST')
                        snapshot_audit = json.loads(snap.with_suffix('.json').read_text())
                        validate_audit(snapshot_audit['candidate_audit'], row['candidate_keys'], row['candidate_context'], row['state_before'], 'branch_snapshot')
                        if snapshot_audit['candidate_keys'] != row['candidate_keys'] or snapshot_audit['candidate_context'] != row['candidate_context']:
                            raise ValueError('SNAPSHOT_CANDIDATE_CONTEXT')
                    if row['labels'] != admitted: raise ValueError('LABEL_RECOMPUTE_MISMATCH')
                    self.audit_window_input(row)
                self.record_candidate_audit(row['candidate_audits']['collector'])
                if row['status'] == 'complete': self.record_candidate_audit(row['candidate_audits']['label_lookup'])
                name = spec['parent']+'--'+window_id
                raw = json.dumps(row, sort_keys=True, allow_nan=False).encode()
                path = raw_dir/(name+'.json.gz'); temporary = path.with_suffix('.pending')
                self.ledger.call('raw_window_write', {'output_writes': 1}, temporary.write_bytes, gzip.compress(raw, mtime=0))
                temporary.replace(path)
                entry = {'parent': spec['parent'], 'window_id': window_id, 'split': spec['split'], 'status': row['status'],
                         'path': str(path.relative_to(self.output)), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
                expected.append(entry)
                if row['status'] == 'complete':
                    compact = self.compact(row); compact['source'] = entry
                    lean.append(compact)
                self.progress(parent=spec['parent'], window=window_id)
        keys = {(r['parent'], r['window_id']) for r in expected}
        target = {(s['parent'], f'time-{t:g}') for s in self.registry if s['role'] != 'policy_task_evaluation' for t in SUPPORT_TIMES}
        if keys != target or len(expected) != len(keys): raise ValueError('MISSING_DUPLICATE_WINDOW')
        self.save('source-inventory.json', {'windows': expected})
        self.save('lean-inputs.json', {'complete': lean, 'no_opportunity': [r for r in expected if r['status']=='no_opportunity']})
        train = [r for r in lean if r['split']=='train']
        parents = {r['parent'] for r in train}
        if len(parents) < math.ceil(sum(s['split']=='train' for s in self.registry)/2):
            raise ScientificStop('data_coverage_gate_stop')
        return lean

    def restore_world(self, binding):
        models = []
        for seed in SEEDS:
            b = binding[str(seed)]; p = self.output/b['path']
            self.ledger.call('world_strict_admission', {'checkpoint_loads': 1}, admit_checkpoint, p, b['sha256'], b['metadata'])
            payload = self.ledger.call('world_restore_load', {'checkpoint_loads': 1}, torch.load, p, map_location='cpu', weights_only=True)
            model = self.ledger.call('world_restore_initialize', {'model_initializations': 1}, ConsequenceJEPA)
            model.load_state_dict(payload['state_dict'], strict=True)
            optimizer = restore_optimizer_strict(model,payload['optimizer'],include_frozen=True)
            if replay.fingerprint(optimizer.state_dict()) != payload['optimizer_sha256']: raise ValueError('WORLD_OPTIMIZER_RESTORE')
            if replay.fingerprint(model.state_dict()) != payload['state_sha256']: raise ValueError('WORLD_RESTORE_STATE')
            models.append(model)
        self.ensemble = FrozenEnsemble(models)
        self.world_identity = replay.fingerprint(self.ensemble.state_dict())
        self.support = torch.tensor(binding[str(SEEDS[0])]['metadata']['support_mask'], dtype=torch.bool)
        for s in SEEDS:
            if binding[str(s)]['metadata']['support_mask'] != self.support.tolist(): raise ValueError('SEED_SUPPORT_IDENTITY')
        self.save('three-world-restore-audit.json', {'pass': True, 'seeds': list(SEEDS), 'world_identity': self.world_identity})

    def train_worlds(self, lean):
        self.phase('world_training_and_restore'); folder = self.output/'world-models'; folder.mkdir()
        train = [r for r in lean if r['split']=='train']; bindings = {}
        for row in lean: self.audit_window_input(row)
        if len(train) != 58:
            raise ValueError('RECOVERY_TRAIN_WINDOW_COUNT:' + str(len(train)))
        if self.matrix.get('recovery_mode') == 'reuse_verified_windows':
            source = self.root/'recovery-inputs'/'world-models'/'world-8201.pt'
            expected = json.loads((self.root/'recovery-inputs'/'reused-world-8201-binding.json').read_text())
            identity = self.ledger.call('reuse_8201_checkpoint_admission',
                {'checkpoint_loads': 1}, admit_checkpoint, source, expected['sha256'], expected['metadata'])
            meta = identity['metadata']
            if identity['state_sha256'] != expected['state_sha256']:
                raise ValueError('RECOVERY_8201_STATE_BINDING')
            if identity['optimizer_sha256'] != expected['optimizer_sha256']:
                raise ValueError('RECOVERY_8201_OPTIMIZER_BINDING')
            if (meta.get('seed') != 8201 or meta.get('epochs') != 40 or meta.get('updates') != 320
                    or meta.get('train_parents') != sorted({r['parent'] for r in train})
                    or meta.get('candidate_key_contract') != CANDIDATE_SCHEMA
                    or meta.get('public_prefix_contract') != 'w1-public-prefix/1'):
                raise ValueError('RECOVERY_8201_FINAL_MODEL_IDENTITY')
            target = folder/'world-8201.pt'
            self.ledger.call('reuse_8201_checkpoint_copy', {'checkpoint_writes': 1}, shutil.copyfile, source, target)
            bindings['8201'] = {'path': str(target.relative_to(self.output)),
                                'sha256': identity['file_sha256'], 'metadata': meta,
                                'reused_from_cache': True}
        for seed in SEEDS:
            if str(seed) in bindings:
                continue
            self.progress(seed=seed, run=f'world:{seed}')
            path = folder/f'world-{seed}.pt'
            _, audit = train_final_world(train, seed, self.ledger, path, epochs=self.matrix['world_epochs'])
            bindings[str(seed)] = {'path': str(path.relative_to(self.output)), 'sha256': sha256_file(path), 'metadata': audit['metadata']}
        self.save('world-bindings.json', bindings); self.restore_world(bindings)

    def assert_frozen(self):
        if self.ensemble is None: raise ValueError('WORLD_REQUIRED')
        if replay.fingerprint(self.ensemble.state_dict()) != self.world_identity or any(p.requires_grad or p.grad is not None for p in self.ensemble.parameters()):
            raise ValueError('FROZEN_WORLD_BOUNDARY')

    def features(self, state, arm, *, candidate_context=None, candidate_keys=None):
        from public_prefix import make_public_prefix, serialize_public_prefix
        serialize_public_prefix(make_public_prefix(state), validation_only=True)
        state = jsonable(state)
        if candidate_context is None:
            if self.active_parent is None: raise ValueError('LIVE_CANDIDATE_PARENT_REQUIRED')
            candidate_context = context_for(self.active_parent, 0,
                f'episode-{self.active_episode}-time-{state["time"]:g}', state)
        if candidate_keys is None: candidate_keys = catalog(candidate_context, state)
        validate_catalog(candidate_keys, candidate_context, state)
        key = (arm, key_digest(candidate_context))
        if key not in self.prediction_cache:
            self.prediction_cache[key] = self.ledger.call('candidate_feature_pipeline', {'candidate_scans': 1},
                build_features, state, self.ensemble if arm=='G1' else None, self.support, arm, self.ledger,
                candidate_context=candidate_context, candidate_keys=candidate_keys)
            for record in self.prediction_cache[key][-1]['candidate_audits'].values(): self.record_candidate_audit(record)
        return self.prediction_cache[key]

    def offline(self, lean):
        self.phase('offline_prediction'); by_role = {}; predictions = []
        train_values = []; train_masks = []
        role = {s['parent']: s['role'] for s in self.registry}
        for row in lean:
            self.audit_window_input(row)
            state = row['state_before']; self.prediction_cache.clear()
            start = time.monotonic(); cpu = time.process_time()
            _, aux, valid, legal, _, trace = self.features(state, 'G1', candidate_context=row['candidate_context'], candidate_keys=row['candidate_keys'])
            costs = {'cpu_seconds': time.process_time()-cpu, 'wall_seconds': time.monotonic()-start}
            actions = torch.where(legal)[0].tolist()
            readouts = trace.get('seed_readouts')
            if readouts is None: raise ValueError('OFFLINE_PREDICTION_MISSING')
            label_by_key = {key_digest(label['candidate_key']): label for label in row['labels']}
            truth = {key['action_index']: label_by_key[key_digest(key)]['continuous']['global_utility_delta']['value'] for key in row['candidate_keys']}
            means = aux[actions, 192:192+READOUT_DIM].tolist(); stds = aux[actions, 192+READOUT_DIM:192+2*READOUT_DIM].tolist()
            utility_index = len(BINARY)+CONTINUOUS.index('global_utility_delta')
            selected_rows = by_role.setdefault(role[row['parent']], {s: [] for s in (*SEEDS, 'ensemble')})
            for seed, pred in [*zip(SEEDS, readouts), ('ensemble', means)]:
                score = {a: pred[i][utility_index] for i, a in enumerate(actions)}
                result = window_metrics(score, truth, actions, row['baseline_action'])
                selected_rows[seed].append(dict(result, parent=row['parent'], window_id=row['window_id']))
            pred_rows = [serialize_prediction(row['candidate_keys'][i], row['candidate_context'], state,
                row['labels'][0]['slots'], means[i], valid[a,192:192+READOUT_DIM].tolist()) for i,a in enumerate(actions)]
            prediction_audit = audit('prediction_export', row['candidate_keys'], row['candidate_context'], state)
            self.record_candidate_audit(prediction_audit)
            predictions.append({'parent': row['parent'], 'role': role[row['parent']], 'window_id': row['window_id'],
                'pair_identity': row['pair_identity'], 'candidate_context': row['candidate_context'],
                'candidate_keys': row['candidate_keys'], 'candidate_audit': prediction_audit,
                'candidate_predictions': pred_rows, 'seed_readouts': readouts,
                'ensemble_std': stds, 'labels': row['labels'], 'costs': costs})
            if row['split']=='train': train_values.append(aux); train_masks.append(valid)
        if not train_values: raise ScientificStop('data_coverage_gate_stop')
        values = torch.stack(train_values); masks = torch.stack(train_masks)
        count = masks.sum((0,1)); safe = torch.where(masks, values, torch.zeros_like(values)); div = count.clamp_min(1)
        avg = safe.sum((0,1))/div; centered = torch.where(masks, values-avg, torch.zeros_like(values))
        scale = (centered.square().sum((0,1))/div).sqrt().clamp_min(1e-6)
        scale = torch.where(count>1, scale, torch.ones_like(scale)); avg = torch.where(count>0,avg,torch.zeros_like(avg))
        self.normalization = (avg, scale)
        self.save('frozen-normalization.json', {'mean':avg.tolist(),'scale':scale.tolist(),'source':'train_only'})
        summary = {r: {str(s): summarize(rows) for s, rows in seeds.items()} for r,seeds in by_role.items()}
        self.save('prediction-cache.json', predictions)
        # Diagnostic BCE/MAE use only valid labels; unknown never supplies a target.
        from prediction_diagnostics import evaluate_heads
        self.save('head-metrics.json', evaluate_heads(predictions))
        self.save('offline-per-parent.json', summary)
        self.save('offline-per-window.json',{r:{str(seed):rows for seed,rows in seeds.items()} for r,seeds in by_role.items()})
        eval_role = 'offline_prediction_gate'; selected = by_role.get(eval_role, {s: [] for s in (*SEEDS,'ensemble')})
        ensemble = summarize(selected['ensemble']); seed_summaries = {s:summarize(selected[s]) for s in SEEDS}
        evaluation_count = sum(s['role']==eval_role for s in self.registry)
        valid_windows = [r for r in lean if role[r['parent']]==eval_role]
        gate = decide_gate(ensemble, seed_summaries, identity_complete=True, independent=True,
            cpu_mean_ms=None, wall_p95_ms=None, opportunity_parent_count=len({r['parent'] for r in valid_windows}),
            valid_label_fraction=(sum(l['continuous']['global_utility_delta']['valid'] for r in valid_windows for l in r['labels']) /
                max(1, sum(len(r['labels']) for r in valid_windows))), expected_parent_count=evaluation_count)
        self.save('prediction-gate.json', gate)
        if not gate['pass']: raise ScientificStop('offline_prediction_gate_stop')
        return gate

    def make_policy(self, arm, seed):
        def create():
            torch.manual_seed(seed)
            p = ConsequencePolicy(JointGraphPreferencePolicy(self.config), arm)
            p.normalization_mean.copy_(self.normalization[0]); p.normalization_scale.copy_(self.normalization[1])
            return p
        policy = self.ledger.call('policy_initialize', {'model_initializations': 1}, create)
        h = replay.fingerprint(policy.state_dict())
        if self.initial_hashes.setdefault(seed,h) != h: raise ValueError('PAIRED_INITIALIZATION')
        return policy

    def evaluate_state(self, policy, state, hidden, preference, *, bootstrap=False):
        # One interface for current value, next value and cut bootstrap. No sampled action is an input.
        pieces = self.features(state, policy.arm)
        base, aux, valid, mask, available, trace = pieces
        packed = policy.pack(base, aux, valid, mask, available)
        trace['candidate_feature_binding'] = feature_binding(trace['candidate_keys'], trace['candidate_context'], jsonable(state), packed)
        def forward():
            with torch.no_grad():
                f, pair, next_hidden = policy.encode(torch.tensor(state['flat'], dtype=torch.float32)[None], hidden)
                ev = policy.evaluate_encoded(f,pair,torch.tensor(preference,dtype=torch.float32)[None],packed[None],mask[None])
            return ev, next_hidden
        ev, next_hidden = self.ledger.call('critic_bootstrap' if bootstrap else 'behavior_forward',
            {'policy_forwards': 1, **({'bootstrap_forwards': 1} if bootstrap else {})}, forward)
        return ev, next_hidden, packed, trace

    def rollout(self, policy, seed, context, count):
        transitions = []; preference = context['preference']
        for _ in range(count):
            state = context['state']; hidden = context['hidden']
            start = time.monotonic(); cpu = time.process_time()
            ev, next_hidden, packed, trace = self.evaluate_state(policy,state,hidden,preference)
            random_key = int(digest({'seed':seed,'episode':context['episode'],'step':context['step']})[:16],16) & ((1<<63)-1)
            def sample():
                torch.manual_seed(random_key); a = int(ev['distribution'].sample()[0]); return a,float(ev['distribution'].log_prob(torch.tensor([a]))[0])
            action, old = self.ledger.call('action_sampling', {'action_samples': 1}, sample)
            nxt, scalar, done, info = self.advance(context['env'],context['adapter'],context['history'],action,label='policy_environment_step')
            reward, _, counts, energy = _vector_reward(info,context['counts'],context['energy'],self.config)
            terminated, truncated = bool(info['terminated']), bool(info['truncated'])
            if bool(done)!=bool(terminated or truncated): raise ValueError('ENVIRONMENT_TERMINATION_CONTRACT')
            next_state = self.prepare_next(context['adapter'],context['history'],nxt)
            if terminated: next_value = np.zeros(2,dtype=np.float32)
            else:
                nv, _, _, _ = self.evaluate_state(policy,next_state,next_hidden,preference,bootstrap=True)
                next_value = nv['critic_values'][0].numpy()
            t = {'obs':np.asarray(state['flat'],np.float32),'policy_hidden_before':hidden.reshape(-1).numpy().copy(),
                'preference':np.asarray(preference,np.float32),'candidate_features':packed.numpy(),
                'mask':np.asarray(state['mask'],bool),'action':action,'old_log_prob':old,
                'old_values':ev['critic_values'][0].numpy(),'next_values':next_value,'vector_reward':reward,
                'terminated':terminated,'truncated':truncated}
            replay.seal(policy,t,{'counterfactual_identity':{'public_sha256':digest(jsonable(state)),
                'continuation_id':'public-joint-opportunity-rule/1','feature_contract':FEATURE_ID},
                'world_model_identity':replay.REGISTRY[policy]['world_hash'],'normalization_identity':replay.normalization_id(policy),
                'hidden_source':'stored_behavior_hidden_per_transition','random_key':random_key,'gate':trace['gate'],
                'task_slots':trace['task_slots'],'feature_contract':FEATURE_ID,
                'candidate_state_before':jsonable(state),'candidate_context':trace['candidate_context'],
                'candidate_keys':trace['candidate_keys'],'candidate_feature_binding':trace['candidate_feature_binding'],
                'public_prefixes':trace['public_prefixes'],
                'candidate_audits':dict(trace['candidate_audits'], replay=audit('replay',trace['candidate_keys'],trace['candidate_context'],jsonable(state)))})
            transitions.append(t)
            durable_append_jsonl(self.output/'behavior-evidence.jsonl',jsonable(t))
            self.progress(seed=seed, run=f'{policy.arm}:{seed}', steps=context['global_steps']+1)
            context.update(state=next_state,hidden=next_hidden.detach(),counts=counts,energy=energy,
                           step=context['step']+1,global_steps=context['global_steps']+1)
            if done:
                context['episode'] += 1
                spec = context['specs'][context['episode'] % len(context['specs'])]
                env,adapter,history,state = self.create_episode(spec,'policy_training')
                preference = self.matrix['training_preferences'][(context['episode']+POLICY_SEEDS.index(seed))%3]
                context.update(env=env,adapter=adapter,history=history,state=state,hidden=torch.zeros(1,1,128),
                    counts={'completed':0,'expired':0},energy=36.,preference=preference,step=0)
                self.prediction_cache.clear()
        return transitions

    def train_policies(self):
        self.phase('conditional_policy_training'); bindings = {}
        specs = [s for s in self.registry if s['split']=='train']
        for arm in ARMS:
            for seed in POLICY_SEEDS:
                self.assert_frozen(); self.prediction_cache.clear()
                policy = self.make_policy(arm,seed)
                optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],lr=3e-4)
                replay.initialize(policy,self.ensemble if arm=='G1' else None,self.ledger)
                env,adapter,history,state = self.create_episode(specs[0],'policy_training')
                context = dict(env=env,adapter=adapter,history=history,state=state,hidden=torch.zeros(1,1,128),
                    counts={'completed':0,'expired':0},energy=36.,preference=self.matrix['training_preferences'][POLICY_SEEDS.index(seed)],
                    specs=specs,episode=0,step=0,global_steps=0)
                updates = 0; config = JointTrainConfig(seed=seed, rollout_steps=self.matrix['rollout_steps'])
                while context['global_steps'] < self.matrix['policy_steps']:
                    count = min(config.rollout_steps,self.matrix['policy_steps']-context['global_steps'])
                    rows = self.rollout(policy,seed,context,count)
                    advantages, returns = _gae_vector(rows,config,torch.device('cpu'))
                    for _ in range(self.matrix['updates_per_rollout']):
                        metrics = self.ledger.call('GPPO_PreCo_update', {'policy_updates':1,'update_rows':len(rows)},
                            ppo_preference_update,policy,rows,optimizer,config,torch.device('cpu'),event_group='B')
                        updates += 1; self.assert_frozen()
                        durable_append_jsonl(self.output/'policy-update-evidence.jsonl',{'arm':arm,'seed':seed,'update':updates,
                            'metrics':metrics,'advantage_mean':advantages.mean(0).tolist(),'returns_mean':returns.mean(0).tolist()})
                        self.progress(updates=updates)
                    replay.release_rollout(policy)
                if updates != self.matrix['policy_updates']: raise ValueError('FINAL_UPDATE_COUNT')
                bindings[f'{arm}/{seed}'] = self.save_policy(policy,optimizer,arm,seed,context['global_steps'],updates)
        self.save('policy-bindings.json',bindings)
        # Nine restorations are an atomic eligibility gate before any task-tape access.
        restored = {key:self.restore_policy(key,b) for key,b in bindings.items()}
        if set(restored) != {f'{a}/{s}' for a in ARMS for s in POLICY_SEEDS}: raise ValueError('NINE_FINAL_MODELS_REQUIRED')
        self.final_policies = restored; self.evaluation_open = True
        self.save('nine-final-restore-audit.json',{'pass':True,'routes':bindings,'evaluation_open':True})

    def save_policy(self,policy,optimizer,arm,seed,steps,updates):
        folder = self.output/'policy-models'; folder.mkdir(exist_ok=True)
        path = folder/f'{arm}-{seed}.pt'
        metadata = {'schema':'consequence-critic-policy/2','arm':arm,'seed':seed,'steps':steps,'updates':updates,
            'feature_contract':FEATURE_ID,'candidate_key_contract':CANDIDATE_SCHEMA,'public_prefix_contract':'w1-public-prefix/1','packed_dim':PACKED_DIM,'task_capacity':6,'phase':'critic_only',
            'world_identity':self.world_identity,'initial_state_sha256':self.initial_hashes[seed],
            'normalization_identity':replay.normalization_id(policy),'optimizer_class':'Adam','lr':3e-4,
            'GPPO_config':asdict(JointTrainConfig(seed=seed,rollout_steps=self.matrix['rollout_steps']))}
        payload = {'metadata':metadata,'state':policy.state_dict(),'optimizer':optimizer.state_dict(),
            'state_sha256':replay.fingerprint(policy.state_dict()),'optimizer_sha256':replay.fingerprint(optimizer.state_dict())}
        self.ledger.call('final_policy_save',{'checkpoint_writes':1},torch.save,payload,path)
        return {'path':str(path.relative_to(self.output)),'sha256':sha256_file(path),'metadata':metadata,
            'state_sha256':payload['state_sha256'],'optimizer_sha256':payload['optimizer_sha256']}

    def restore_policy(self,key,binding):
        arm, seed = key.split('/'); seed = int(seed); path = self.output/binding['path']
        if sha256_file(path) != binding['sha256']: raise ValueError('POLICY_FILE_DIGEST')
        payload = self.ledger.call('strict_policy_load',{'checkpoint_loads':1},torch.load,path,map_location='cpu',weights_only=True)
        if payload['metadata'] != binding['metadata']: raise ValueError('POLICY_METADATA')
        meta = payload['metadata']
        if meta['arm']!=arm or meta['seed']!=seed or meta['steps']!=self.matrix['policy_steps'] or meta['updates']!=self.matrix['policy_updates']:
            raise ValueError('POLICY_FINAL_IDENTITY')
        if meta.get('public_prefix_contract')!='w1-public-prefix/1' or meta.get('candidate_key_contract')!=CANDIDATE_SCHEMA or meta['world_identity']!=self.world_identity or meta['feature_contract']!=FEATURE_ID or meta['packed_dim']!=PACKED_DIM:
            raise ValueError('POLICY_WORLD_FEATURE_IDENTITY')
        if meta['initial_state_sha256']!=self.initial_hashes.get(seed): raise ValueError('POLICY_PAIRED_INITIAL_IDENTITY')
        if replay.fingerprint(payload['state'])!=binding['state_sha256'] or replay.fingerprint(payload['optimizer'])!=binding['optimizer_sha256']:
            raise ValueError('POLICY_STATE_OPTIMIZER_DIGEST')
        policy = self.make_policy(arm,seed); policy.load_state_dict(payload['state'],strict=True)
        optimizer = restore_optimizer_strict(policy,payload['optimizer'])
        if replay.fingerprint(policy.state_dict())!=binding['state_sha256'] or replay.fingerprint(optimizer.state_dict())!=binding['optimizer_sha256']:
            raise ValueError('POLICY_STRICT_RESTORE')
        if replay.normalization_id(policy)!=meta['normalization_identity']: raise ValueError('POLICY_NORMALIZATION_RESTORE')
        policy.eval(); return policy

    def evaluation_specs(self):
        if not self.evaluation_open or len(self.final_policies)!=9: raise ValueError('TASK_EVALUATION_TAPE_LOCKED')
        tape = json.loads((self.root/'evaluation-tape.json').read_text())
        registered = {s['parent']:s for s in self.registry if s['role']=='policy_task_evaluation'}
        for spec in tape['parents']:
            if spec['parent'] not in registered or digest(spec['scenario'])!=registered[spec['parent']]['scenario_sha256']:
                raise ValueError('EVALUATION_TAPE_IDENTITY')
        self.save('evaluation-access.json',{'opened_after_nine_strict_restores':True,'parent_count':len(tape['parents'])})
        return tape['parents']

    def tasks(self):
        self.phase('conditional_task_confirmation'); episodes = []
        for spec in self.evaluation_specs():
            for arm in (*ARMS,'H'):
                for seed in ((None,) if arm=='H' else POLICY_SEEDS):
                    self.assert_frozen(); self.prediction_cache.clear()
                    env,adapter,history,state = self.create_episode(spec,'task_evaluation')
                    hidden = torch.zeros(1,1,128); preference = [.8,.2]; counts={'completed':0,'expired':0}; energy=36.
                    selector=ClassicalSelector('hungarian',preference) if arm=='H' else None
                    policy=self.final_policies[f'{arm}/{seed}'] if arm!='H' else None
                    rewards=[]; costs=[]; actions=[]; reward_evidence=[]; done=False
                    self.ledger.call('task_episode_start',{'task_episodes':1},lambda:None)
                    for step in range(18):
                        start=time.monotonic(); cpu=time.process_time(); model_before=self.ensemble.calls
                        prepared_cpu,prepared_wall=self.last_preparation_cost
                        if arm=='H':
                            action,_ = self.ledger.call('Hungarian_distribution',{'candidate_scans':1},selector.choose,state,adapter.memory)
                        else:
                            ev,hidden,_,trace=self.evaluate_state(policy,state,hidden,preference)
                            random_key=int(digest({'parent':spec['parent'],'seed':seed,'step':step})[:16],16)&((1<<63)-1)
                            def sample(): torch.manual_seed(random_key); return int(ev['distribution'].sample()[0])
                            action=self.ledger.call('task_sampling',{'action_samples':1},sample)
                        self.assert_frozen()
                        # Command safety/commit is part of decision latency; environment stepping is not.
                        adapter.commit(int(action))
                        cost={'parent':spec['parent'],'arm':arm,'seed':seed,'step':step,'cpu_seconds':time.process_time()-cpu+prepared_cpu,
                            'wall_seconds':time.monotonic()-start+prepared_wall,'world_forwards':self.ensemble.calls-model_before,
                            'public_preparation_cpu_seconds':prepared_cpu,'public_preparation_wall_seconds':prepared_wall,
                            'candidate_count':sum(state['mask']),'time':state['time']}
                        costs.append(cost); durable_append_jsonl(self.output/'decision-costs.jsonl',cost)
                        nxt,scalar,done,info=self.ledger.call('task_environment_step',{'environment_steps':1},env.step,int(action))
                        if not done: self.append_history(history,nxt)
                        reward_evidence.append({'counts_before':dict(counts),'energy_before':energy,
                            'counts_after':info['counts'],'energy_after':sum(info['energy'].values()),
                            'terminated':info['terminated'],'truncated':info['truncated']})
                        reward,_,counts,energy=_vector_reward(info,counts,energy,self.config)
                        rewards.append(reward.tolist()); actions.append(int(action))
                        self.progress(run=f'task:{arm}:{seed}',parent=spec['parent'],steps=step+1)
                        if done: break
                        state=self.prepare_next(adapter,history,nxt)
                    if not done: raise ValueError('TASK_EPISODE_INCOMPLETE')
                    utility=sum(.99**i*(.4*r[0]+.2*r[1]) for i,r in enumerate(rewards))
                    result={'parent':spec['parent'],'arm':arm,'seed':seed,'scenario_sha256':spec['scenario_sha256'],
                        'exogenous_key':spec['exogenous_key'],'vector_rewards':rewards,'utility':utility,'steps':len(rewards),
                        'actions':actions,'reward_evidence':reward_evidence,'completed':counts['completed'],'expired':counts['expired'],'energy_used':36.-energy,
                        'damaged':sum(not r.alive for r in env.clock.resources.values()),'lifecycle':lifecycle(env),
                        'collision':None,'terminated':info['terminated'],'truncated':info['truncated'],
                        'world_triggered_decisions':sum(c['world_forwards']>0 for c in costs),
                        'decision_costs':costs, 'reward_accounting':'native_vector_reward_physical_arrival',
                        'host_confirmed':sum(r.get('host_confirmation_time') is not None for r in env._completion_records.values()),
                        'unknown_count':1+sum(not t[f]['valid'] for t in lifecycle(env) for f in
                            ('physical_arrival','on_time','expired','host_confirmation')),
                        'final_energy':energy,'task_capacity':self.config.task_capacity,'initial_energy':36.}
                    self.ledger.call('task_result',{'output_writes':1},durable_append_jsonl,self.output/'task-results.jsonl',result)
                    episodes.append(result)
        if len(episodes) != self.matrix['task_episodes']: raise ValueError('TASK_MATRIX_INCOMPLETE')

    def run(self):
        status='technical_stop'; error=None
        try:
            self.phase('input_admission')
            if len({s['parent'] for s in self.registry})!=len(self.registry): raise ValueError('DUPLICATE_PARENT')
            if len({s['scenario_sha256'] for s in self.registry})!=len(self.registry): raise ValueError('SCENARIO_CROSS_SPLIT')
            self.save('research-identity.json',{'mode':self.matrix['research_mode'],'controlled':self.matrix.get('controlled',False)})
            lean=self.collect(); self.train_worlds(lean); self.offline(lean); self.train_policies(); self.tasks()
            self.phase('analysis_and_settlement')
            from task_analysis import recompute
            summary=self.ledger.call('independent_task_recompute',{'analysis_calls':1},recompute,self.output,self.matrix)
            summary['execution_scope']='controlled_synthetic_only' if self.matrix.get('controlled') else 'development_exploration'
            self.save('result-summary.json',summary)
            status='controlled_engineering_pipeline_pass' if self.matrix.get('controlled') else 'completed_development_exploration'
        except ScientificStop as exc: status=str(exc)
        except BaseException as exc:
            error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
            durable_atomic_json(self.output/'first-error.json',error)
        finally:
            status=self.finish(status,error)
        return 1 if status=='technical_stop' else 0

    def finish(self, status, error):
        primary_outcome=status
        settlement_errors=[]
        settled=None

        def failure(operation, exc):
            nonlocal status,error
            detail={'operation':operation,'type':type(exc).__name__,'message':str(exc),
                    'traceback':traceback.format_exc()}
            settlement_errors.append(detail)
            status='technical_stop'
            if error is None:
                error=detail
                path=self.output/'first-error.json'
                if path.exists(): error=json.loads(path.read_text())
                else: durable_atomic_json(path,error)

        try: self.phase('analysis_and_settlement')
        except BaseException as exc: failure('settlement_phase',exc)
        try:
            settled=self.ledger.assert_settled()
            if settled['failed_calls']: raise RuntimeError('LEDGER_FAILED_CALLS_PRESENT')
        except BaseException as exc:
            failure('ledger_settlement',exc)
            try: settled=self.ledger.snapshot()
            except BaseException as snapshot_exc: failure('ledger_snapshot',snapshot_exc)
        try:
            self.phase('settlement_and_verified_export')
            self.ledger.snapshot_phase(reason='worker_final')
        except BaseException as exc: failure('export_boundary',exc)
        try: self.ledger.close(snapshot=False)
        except BaseException as exc: failure('ledger_close',exc)
        try:
            from task_analysis import report
            report(self.output,status)
        except BaseException as exc: failure('report',exc)
        durable_atomic_json(self.output/'resource-settlement.json',{'status':status,'primary_outcome':primary_outcome,
            'first_error':error,'ledger':settled,'settlement_errors':settlement_errors,
            'execution_scope':'controlled_synthetic_only' if self.matrix.get('controlled') else 'development_exploration',
            'final_resource_scope':'worker_only_not_supervisor_final','wall_seconds_worker':time.monotonic()-self.started,
            'scope':'worker; supervisor totals include it once','unentered_stages':'unevaluated', 'automatic_retry':False})
        durable_atomic_json(self.output/'status.json',{'status':status,'primary_outcome':primary_outcome,'ledger':settled})
        return status


class ScientificStop(RuntimeError): pass


def main():
    import sys
    root=Path(__file__).resolve().parent
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    matrix=json.loads((root/'experiment-matrix.json').read_text())
    if matrix.get('controlled'): raise ValueError('CONTROLLED_FACTORY_ONLY_IN_SEPARATE_ACCEPTANCE_ENTRY')
    receipt=json.loads(sys.stdin.buffer.read())
    if not receipt.get('authorization_verified') or os.environ.get('W1_VERIFIED_ATTEMPT')!=matrix['attempt']:
        raise ValueError('FORMAL_AUTHORIZATION_RECEIPT_REQUIRED')
    output=root/'run-once'; output.mkdir(exist_ok=False)
    ledger=BudgetLedger(output/'ledger.sqlite',json.loads((root/'RESOURCE_REQUEST.json').read_text()))
    return Driver(root,output,matrix,ledger).run()


if __name__=='__main__': raise SystemExit(main())
