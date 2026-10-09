"""Mandatory pre-update behavior reconstruction; current parameters may change."""
import copy
import hashlib
import json
import weakref
import torch
from consequence_policy import FEATURE_ID
from candidate_keys import verify_replay_candidate, audit, validate_audit

ATOL = 1e-6
REGISTRY = weakref.WeakKeyDictionary()
EVIDENCE_FIELDS = ("counterfactual_identity", "world_model_identity", "normalization_identity",
                   "hidden_source", "random_key", "gate", "task_slots", "feature_contract",
                   'candidate_state_before', 'candidate_context', 'candidate_keys', 'candidate_feature_binding', 'candidate_audits', 'public_prefixes')


def fingerprint(value):
    h = hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            y = x.detach().cpu().contiguous()
            h.update(str((str(y.dtype), tuple(y.shape))).encode())
            h.update(y.numpy().tobytes())
        elif hasattr(x, "tolist"):
            add(torch.as_tensor(x))
        elif isinstance(x, dict):
            for k in sorted(x):
                h.update(json.dumps(k).encode()); add(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(str(len(x)).encode())
            for v in x: add(v)
        else:
            h.update(json.dumps(x, allow_nan=False).encode())
    add(value)
    return h.hexdigest()


def normalization_id(policy):
    return fingerprint({"mean": policy.normalization_mean, "scale": policy.normalization_scale,
                        "feature_contract": FEATURE_ID})


def initialize(policy, frozen_world=None, ledger=None):
    worlds = frozen_world
    if worlds is not None and any(p.requires_grad for p in worlds.parameters()):
        raise ValueError("WORLD_NOT_FROZEN")
    REGISTRY[policy] = {"rows": {}, "snapshots": {}, "world": worlds,
                        "world_hash": fingerprint(worlds.state_dict()) if worlds is not None else "NO_WORLD",
                        "normalization": normalization_id(policy), "verified": 0, 'ledger': ledger}


def guard(policy):
    r = REGISTRY.get(policy)
    if r is None: raise ValueError("REPLAY_NOT_INITIALIZED")
    w = r["world"]
    if w is not None and (any(p.requires_grad or p.grad is not None for p in w.parameters()) or
                          fingerprint(w.state_dict()) != r["world_hash"]):
        raise ValueError("FROZEN_WORLD_CHANGED")
    if normalization_id(policy) != r["normalization"]:
        raise ValueError("NORMALIZATION_CHANGED")
    return r


def record_fields(t):
    names = ("obs", "policy_hidden_before", "preference", "candidate_features", "mask", "action", "old_log_prob",
             "old_values", "next_values", "vector_reward", "terminated", "truncated", "replay_evidence")
    return {k: t[k] for k in names}


def seal(policy, transition, evidence):
    transition['replay_evidence'] = copy.deepcopy(evidence)
    verify_replay_candidate(transition)
    r = guard(policy)
    if any(k not in evidence for k in EVIDENCE_FIELDS):
        raise ValueError("MISSING_BEHAVIOR_EVIDENCE")
    if evidence["feature_contract"] != FEATURE_ID or evidence["world_model_identity"] != r["world_hash"]:
        raise ValueError("SOURCE_IDENTITY")
    if evidence["normalization_identity"] != r["normalization"]:
        raise ValueError("NORMALIZATION_IDENTITY")
    if evidence["hidden_source"] != "stored_behavior_hidden_per_transition":
        raise ValueError("HIDDEN_SEMANTICS")
    if evidence["gate"] not in ("feature_available", "base_fallback", "no_opportunity"):
        raise ValueError("GATE_CONTRACT")
    snapshot_id = fingerprint(policy.state_dict())
    if snapshot_id not in r["snapshots"]:
        operation = lambda: copy.deepcopy(policy).eval()
        r["snapshots"][snapshot_id] = (r['ledger'].call('behavior_snapshot', {'behavior_snapshots': 1}, operation)
                                      if r['ledger'] else operation())
    evidence = copy.deepcopy(evidence)
    evidence["behavior_policy_identity"] = snapshot_id
    evidence["behavior_adapter_identity"] = fingerprint(policy.critic_adapter.state_dict())
    transition["replay_evidence"] = evidence
    row_id = str(len(r["rows"]))
    transition["replay_id"] = row_id
    r["rows"][row_id] = {"digest": fingerprint(record_fields(transition)), "snapshot": snapshot_id}
    # Old probabilities are checked at ingestion as well as before every native update.
    verify_update(policy, [transition], torch.device("cpu"))
    return transition


def verify_update(policy, transitions, device):
    for transition in transitions:
        verify_replay_candidate(transition)
    r = guard(policy)
    if not transitions: raise ValueError("EMPTY_REPLAY")
    from gppo_world.joint_training import _joint_logprob_rows
    for t in transitions:
        row = r["rows"].get(t.get("replay_id"))
        if row is None or fingerprint(record_fields(t)) != row["digest"]:
            raise ValueError("BEHAVIOR_EVIDENCE_CHANGED")
        candidate_audit = verify_replay_candidate(t)
        t['candidate_GPPO_audit'] = candidate_audit
        if r['ledger']:
            from infra_io import durable_append_jsonl
            r['ledger'].call('GPPO_candidate_identity_audit', {'candidate_contract_audits': 1},
                durable_append_jsonl, r['ledger'].phase_path.parent/'candidate-identity-audits.jsonl', candidate_audit)
        snapshot = r["snapshots"][row["snapshot"]]
        if fingerprint(snapshot.state_dict()) != row["snapshot"]:
            raise ValueError("BEHAVIOR_SNAPSHOT_CHANGED")
        operation = lambda: float(_joint_logprob_rows(snapshot, [t], device)[0])
        rebuilt = (r['ledger'].call('behavior_reconstruction', {'replay_rows': 1, 'policy_forwards': 1}, operation)
                   if r['ledger'] else operation())
        if not torch.isfinite(torch.tensor(rebuilt)) or abs(rebuilt - t["old_log_prob"]) > ATOL:
            raise ValueError("OLD_LOGPROB_MISMATCH")
    r["verified"] += 1


def release_rollout(policy):
    r = guard(policy)
    r["rows"].clear(); r["snapshots"].clear()
