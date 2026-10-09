"""New all-legal-candidate collector; fixed public-rule continuation, no old teacher."""
import copy
import enum
import json
import math
import pickle
from dataclasses import asdict
import numpy as np
from consequence_contract import digest, leaf, opportunity, task_slots, admit_window, GAMMA
from candidate_keys import context_for, catalog, audit, validate_key, validate_catalog, key_digest

CONTINUATION_ID = "public-joint-opportunity-rule/1"


def first_difference(a, b, path='$'):
    if type(a) is not type(b):
        return {'field': path, 'expected_type': type(a).__name__, 'actual_type': type(b).__name__,
                'expected': a if type(a) in (str, float, int, bool) else None,
                'actual': b if type(b) in (str, float, int, bool) else None}
    if type(a) is dict:
        if set(a) != set(b): return {'field': path, 'expected_keys': sorted(a), 'actual_keys': sorted(b)}
        for k in sorted(a):
            found = first_difference(a[k], b[k], path + '.' + k)
            if found: return found
    elif type(a) is list:
        if len(a) != len(b): return {'field': path, 'expected_length': len(a), 'actual_length': len(b)}
        for i, (x, y) in enumerate(zip(a, b)):
            found = first_difference(x, y, path + '[' + str(i) + ']')
            if found: return found
    elif a != b:
        return {'field': path, 'expected': a, 'actual': b}
    return None


def snapshot_payload(env, adapter, history, state, context, keys):
    from public_prefix import prefix_records
    if float(env.clock.time) != state['time']:
        raise ValueError('PUBLIC_PREFIX_SNAPSHOT_TIME_MISMATCH')
    from public_prefix import sha as public_sha
    if public_sha(env._exogenous_key) != state['public_prefix_source']['random_key_sha256']:
        raise ValueError('PUBLIC_PREFIX_SNAPSHOT_RANDOM_KEY_MISMATCH')
    if jsonable(history.state) != state.get('public_history'):
        raise ValueError('PUBLIC_PREFIX_SNAPSHOT_HISTORY_MISMATCH')
    prefixes = prefix_records(state, context, keys)
    return pickle.dumps({'schema': 'w1-pre-action-branch-snapshot/1',
        'environment': (env, adapter, history), 'state_before': copy.deepcopy(state),
        'context': copy.deepcopy(context), 'keys': copy.deepcopy(keys), 'public_prefixes': prefixes}, protocol=4)


def restore_candidate(payload, canonical, state, context, keys, key, ledger, evidence_store, *, expected_payload_sha256):
    """Rebuild exclusively from the saved public snapshot; never observe restored env."""
    from public_prefix import make_public_prefix, serialize_public_prefix, compare
    validate_catalog(keys, context, state)
    validate_key(key, context, state)
    if digest(payload) != expected_payload_sha256: raise ValueError('PUBLIC_PREFIX_SNAPSHOT_PAYLOAD_DIGEST')
    snapshot = pickle.loads(payload)
    if type(snapshot) is not dict or set(snapshot) != {'schema', 'environment', 'state_before', 'context', 'keys', 'public_prefixes'} or snapshot['schema'] != 'w1-pre-action-branch-snapshot/1':
        raise ValueError('PUBLIC_PREFIX_BRANCH_SNAPSHOT_SCHEMA')
    branch, adapter, history = snapshot['environment']
    initial = state_bytes((branch, adapter, history))
    canonical_ok = initial == canonical
    def probe():
        compare(context, snapshot['context'], '$.snapshot.context')
        compare(keys, snapshot['keys'], '$.snapshot.candidate_keys')
        public = snapshot['state_before']
        expected = make_public_prefix(state, context, key)
        actual = make_public_prefix(public, snapshot['context'], key)
        serialize_public_prefix(actual, expected=expected)
        serialized = serialize_public_prefix(actual, expected_record=snapshot['public_prefixes'][key['candidate_order']])
        return public, serialized['sha256']
    public, public_prefix_sha256 = ledger.call('candidate_public_snapshot_probe', {'candidate_snapshot_probes': 1}, probe)
    after = state_bytes((branch, adapter, history))
    result = {'candidate_key': key, 'candidate_key_sha256': key_digest(key),
              'candidate_audit': audit('branch_snapshot', keys, context, state),
              'canonical_expected_sha256': digest(canonical), 'canonical_actual_sha256': digest(initial),
              'canonical_equal': canonical_ok, 'public_expected_sha256': digest(state),
              'public_actual_sha256': digest(public), 'public_equal': public == state,
              'first_public_difference': first_difference(state, public),
              'public_prefix_sha256': public_prefix_sha256, 'snapshot_payload_sha256': expected_payload_sha256,
              'restoration_public_source': 'saved_pre_action_snapshot_only',
              'environment_observation_calls_during_probe': 0,
              'audited_branch_unmodified': after == initial}
    evidence_store(result)
    if not canonical_ok: raise ValueError('CANDIDATE_CANONICAL_PREFIX_MISMATCH')
    if after != initial: raise ValueError('CANDIDATE_AUDIT_MUTATED_BRANCH')
    if public != state: raise ValueError('CANDIDATE_PUBLIC_PREFIX_MISMATCH:' + str(result['first_public_difference']))
    return branch, adapter, history


def jsonable(x):
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if hasattr(x, "tolist"): return x.tolist()
    return x


def state_bytes(value):
    seen = {}
    def enc(x):
        if isinstance(x, enum.Enum): return {"enum": type(x).__qualname__, "value": enc(x.value)}
        if x is None or isinstance(x, (str, int, float, bool)): return x
        if isinstance(x, np.generic): return enc(x.item())
        if isinstance(x, bytes): return {"bytes": x.hex()}
        if isinstance(x, np.ndarray): return {"dtype": str(x.dtype), "shape": list(x.shape), "bytes": x.tobytes().hex()}
        key = id(x)
        if key in seen: return {"reference": seen[key]}
        seen[key] = len(seen)
        if isinstance(x, dict): return {"mapping": [[enc(k), enc(v)] for k, v in sorted(x.items(), key=lambda kv: repr(kv[0]))]}
        if isinstance(x, (tuple, list)): return {type(x).__name__: [enc(v) for v in x]}
        if isinstance(x, set): return {"set": [enc(v) for v in sorted(x, key=repr)]}
        if hasattr(x, "__dict__"): return {"type": type(x).__module__ + "." + type(x).__qualname__, "state": enc(vars(x))}
        raise ValueError("UNSERIALIZABLE_PREFIX:" + type(x).__name__)
    # Private lifecycle clocks contain -inf before release; preserve exact IEEE representations.
    return json.dumps(enc(value), sort_keys=True, separators=(",", ":"), allow_nan=True).encode()


def choose_public(observation):
    from gppo_world.graph5 import graph5_from_m10_observation
    from gppo_world.joint_consequence_baseline import select_public_joint_action
    return select_public_joint_action(graph5_from_m10_observation(observation))


def lifecycle(env):
    result = []
    for tid, t in sorted(env.clock.tasks.items()):
        arrivals = [e for e in env.clock.log if e.get("kind") == "arrival" and e.get("task") == tid]
        on_time = t.completed_at is not None and t.completed_at <= t.deadline
        settled = t.state.value in ("completed", "expired")
        record = env._completion_records.get(tid, {})
        confirmed = record.get("host_confirmation_time") is not None
        result.append({"task_id": tid, "deadline": t.deadline, "completed_at": t.completed_at,
            "state": t.state.value, "physical_arrival": leaf(bool(arrivals), True) if arrivals or settled else leaf(reason="horizon_censored"),
            "on_time": leaf(bool(on_time), True) if settled else leaf(reason="horizon_censored"),
            "expired": leaf(t.state.value == "expired", True) if settled else leaf(reason="horizon_censored"),
            "host_confirmation": leaf(True, True) if confirmed else leaf(reason="not_observed_at_branch_end")})
    return result


def collect_window(env, observation, *, adapter, history, parent, repeat, split, window_id, ledger, snapshot_store, evidence_store):
    """Shared production collection hook; acceptance replaces only the environment."""
    from pathlib import Path
    from gppo_world.joint_training import _vector_reward
    from public_prefix import make_public_prefix, serialize_public_prefix
    serialize_public_prefix(make_public_prefix(observation), validation_only=True)
    state = jsonable(observation)
    slots = task_slots(state)
    row = {"parent": parent, "repeat": repeat, "split": split, "window_id": window_id, "state_before": state}
    context = context_for(parent, repeat, window_id, state)
    keys = catalog(context, state)
    from public_prefix import prefix_records
    row['public_prefixes'] = prefix_records(state, context, keys)
    row.update(candidate_context=context, candidate_keys=keys, candidate_audits={'collector': audit('collector', keys, context, state)})
    if not opportunity(state["mask"]):
        row.update(status="no_opportunity", branches=[])
        return row
    restoration = snapshot_payload(env, adapter, history, state, context, keys)
    canonical = state_bytes((env, adapter, history))
    if state_bytes(pickle.loads(restoration)['environment']) != canonical:
        raise ValueError("PREFIX_RESTORE_MISMATCH")
    rng = pickle.dumps({"scheme": "stateless-semantic-message-sha256", "seed": env.scenario.seed,
                        "key": env._exogenous_key, "communication": asdict(env.communication)}, protocol=4)
    identity = {k: row[k] for k in ("parent", "repeat", "split", "window_id")}
    identity.update(public_sha256=digest(state), prefix_sha256=digest(canonical), hidden_sha256=digest(history.state.tobytes()),
                    rng_sha256=digest(rng), exogenous_key=env._exogenous_key, continuation_id=CONTINUATION_ID,
                    sampler_sha256=digest((Path(__file__).parent / "native/gppo_world/m10_communication.py").read_bytes()),
                    environment_sha256=digest(asdict(env.config)))
    # A snapshot is stored once per window, never duplicated in candidate JSONL rows.
    ledger.call("snapshot_store", {"snapshot_writes": 1}, snapshot_store,
                digest(restoration), restoration, {"identity": identity, "rng_hex": rng.hex(), "canonical_hex": canonical.hex(),
                 'candidate_context': context, 'candidate_keys': keys, 'candidate_audit': audit('branch_snapshot', keys, context, state)})
    baseline = ledger.call('baseline_scan', {'candidate_scans': 1}, choose_public, observation)
    branches = []
    for key in keys:
        a = key['action_index']
        branch, branch_adapter, branch_history = ledger.call('restore_candidate', {'candidate_branches': 1},
            restore_candidate, restoration, canonical, state, context, keys, key, ledger, evidence_store,
            expected_payload_sha256=digest(restoration))
        previous_counts = {"completed": env._last_completed, "expired": env._last_expired}
        previous_energy = sum(r.energy for r in env.clock.resources.values())
        initial_energy = previous_energy
        initial_damage = sum(not r.alive for r in env.clock.resources.values())
        initial_log = len(env.clock.log)
        trace, vector_return, done, action, obs = [], np.zeros(2), False, a, copy.deepcopy(observation)
        for index in range(18):
            before = float(obs["time"])
            tasks_before = lifecycle(branch)
            resources_before = {k: {"energy": r.energy, "alive": r.alive, "position": list(r.position)} for k, r in branch.clock.resources.items()}
            branch_adapter.commit(action)
            nxt, scalar_reward, done, info = ledger.call("candidate_step", {"environment_steps": 1,
                **({"forced_first_actions": 1} if index == 0 else {})}, branch.step, action)
            reward, _, previous_counts, previous_energy = _vector_reward(info, previous_counts, previous_energy, branch.config)
            vector_return += GAMMA ** index * reward
            trace.append({"action": action, "mask": jsonable(obs["mask"]), "time_before": before,
                "time_after": float(nxt["time"]), "vector_reward": reward.tolist(), "scalar_reward": float(scalar_reward),
                "lifecycle_before": tasks_before, "lifecycle_after": lifecycle(branch), "resources_before": resources_before,
                "resources_after": {k: {"energy": r.energy, "alive": r.alive, "position": list(r.position)} for k, r in branch.clock.resources.items()},
                "feedback": info["feedback"], "new_events": jsonable(info["new_events"]),
                "communication_delta": jsonable(info["communication_delta"]), "public_after": jsonable(nxt)})
            branch_history.append(nxt)
            public_next, mask_next = branch_adapter.prepare(nxt)
            public_next["mask"] = mask_next
            obs = branch_history.attach_current(public_next)
            if done: break
            action = ledger.call('continuation_scan', {'candidate_scans': 1}, choose_public, obs)
        if not done: raise ValueError("BRANCH_END_NOT_REACHED")
        damaged = sum(not r.alive for r in branch.clock.resources.values()) - initial_damage
        new_damage = any(e.get("kind") == "damage" for e in branch.clock.log[initial_log:])
        # Explicit queue-rejection count, not a claim about unavailable collision physics.
        competing = sum(s["feedback"] == "resource_busy" for s in trace)
        out = {"tasks": lifecycle(branch), "utility": float(.4 * vector_return[0] + .2 * vector_return[1]),
               "energy_used": leaf(initial_energy - previous_energy, True), "battery_remaining": leaf(previous_energy, True),
               "damaged": leaf(new_damage, True), "collision": leaf(reason="simulator_not_implemented"),
               "damage_count": leaf(damaged, True), "competition_count": leaf(competing, True)}
        b = dict(identity, action=a, candidate_key=key, candidate_key_sha256=key_digest(key),
                 candidate_snapshot_audit=audit('branch_snapshot', keys, context, state),
                 target_task_id=key['task_id'],
                 restore_verified=True, snapshot_ref=digest(restoration), trace=trace, trace_sha256=digest(trace),
                 end_condition="terminated" if info["terminated"] else "horizon", outcome=out)
        branches.append(b)
    row.update(status="complete", pair_identity=identity, baseline_action=baseline, branches=branches)
    row["labels"] = admit_window(row, {parent: split})
    row['candidate_audits']['label_lookup'] = audit('label_lookup', keys, context, state)
    return row
