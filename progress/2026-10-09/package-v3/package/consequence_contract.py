"""Identity-aware counterfactual labels; no environment/model imports."""
import hashlib
import json
import math
from candidate_keys import validate_key, validate_window, key_digest, audit

SCHEMA = "w1-drone-action-consequence/1"
SEEDS = (8201, 8202, 8203)
BINARY = ("physical_arrival", "on_time", "expired", "damaged", "collision", "host_confirmation")
CONTINUOUS = ("other_completed_delta", "other_expired_delta", "energy_delta", "battery_delta",
              "damage_count_delta", "competition_delta", "global_utility_delta")
SLOT_FIELDS = ("completed_delta", "expired_delta")
GAMMA = .99
PUBLIC_FIELDS = {"flat", "uavs", "tasks", "graph", "mask", "time", "version", "types", "public_entity_ids",
                 "trigger_flags", "event_signal", "event_signal_valid", "continuation_action", "continuation_actions",
                 "public_history", "public_history_schema", "public_prefix_source"}


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def leaf(value=None, valid=False, reason="unknown"):
    if type(valid) is not bool:
        raise ValueError("VALIDITY_TYPE")
    if valid and (type(value) not in (bool, int, float) or not math.isfinite(value)):
        raise ValueError("LABEL_NONFINITE_OR_TYPE")
    return {"value": value if valid else None, "valid": valid,
            "unknown_reason": None if valid else reason}


def legal_actions(mask):
    if not isinstance(mask, list) or len(mask) != 25 or any(type(v) is not bool for v in mask):
        raise ValueError("MASK_CONTRACT")
    result = [a for a, allowed in enumerate(mask) if allowed]
    if not result:
        raise ValueError("NO_LEGAL_ACTION")
    return result


def opportunity(mask):
    actions = legal_actions(mask)
    return len(actions) >= 2 and any(a != 24 for a in actions)


def task_slots(state):
    """Slot order is the public action/graph order, never label-list order."""
    if set(state) - PUBLIC_FIELDS:
        raise ValueError("NONPUBLIC_OR_FUTURE_INPUT_FIELD")
    from public_prefix import make_public_prefix, serialize_public_prefix
    serialize_public_prefix(make_public_prefix(state), validation_only=True)
    from event_signal_contract import validate as validate_event_signal
    validate_event_signal(state, path='$.event_signal')
    ids = state["public_entity_ids"]["tasks"]
    uavs = state["public_entity_ids"]["uavs"]
    if not isinstance(ids, (list, tuple)) or len(ids) > 6:
        raise ValueError("TASK_CAPACITY")
    if any(type(t) is not str or not t for t in ids) or len(set(ids)) != len(ids):
        raise ValueError("TASK_ID_DUPLICATE_OR_INVALID")
    if len(uavs) != 4 or len(set(uavs)) != 4:
        raise ValueError("UAV_SLOTS")
    rows = state["tasks"]
    graph = state["graph"]["node_features"]
    if len(rows) != 6 or len(graph) != 21 or any(len(r) != 32 for r in rows):
        raise ValueError("TASK_GRAPH_SHAPE")
    for slot, row in enumerate(rows):
        if list(row) != list(graph[11 + slot]) or any(not math.isfinite(v) for v in row):
            raise ValueError("TASK_GRAPH_ALIGNMENT")
        if slot >= len(ids) and any(row):
            raise ValueError("UNIDENTIFIED_TASK_NODE")
    actions = legal_actions(state["mask"])
    if any(a != 24 and a % 6 >= len(ids) for a in actions):
        raise ValueError("ACTION_EMPTY_SLOT")
    return list(ids) + [None] * (6 - len(ids))


def outcome_tasks(outcome):
    rows = outcome["tasks"]
    result = {r["task_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("DUPLICATE_LIFECYCLE_ID")
    for r in rows:
        for name in ("physical_arrival", "on_time", "expired", "host_confirmation"):
            f = r[name]
            if type(f.get("valid")) is not bool or (f["valid"] and type(f["value"]) is not bool):
                raise ValueError("BINARY_LABEL_CONTRACT")
            if not f["valid"] and f["value"] is not None:
                raise ValueError("UNKNOWN_NOT_NULL")
    return result


def paired(candidate, baseline, state, expected):
    if expected.get("continuation_id") != "public-joint-opportunity-rule/1":
        raise ValueError("NEW_FIXED_RULE_CONTINUATION_REQUIRED_NOT_GPPO_Q")
    for key in ("parent", "repeat", "window_id", "split", "public_sha256", "prefix_sha256",
                "hidden_sha256", "rng_sha256", "exogenous_key", "sampler_sha256",
                "continuation_id", "environment_sha256"):
        if key not in expected or candidate.get(key) != expected[key] or baseline.get(key) != expected[key]:
            raise ValueError("PAIR_IDENTITY:" + key)
    if expected["public_sha256"] != digest(state):
        raise ValueError("PUBLIC_HASH")
    slots = task_slots(state)
    for b in (candidate, baseline):
        if b.get("restore_verified") is not True:
            raise ValueError("reproducibility_unknown")
        a = b["action"]
        validate_key(b['candidate_key'], expected['candidate_context'], state)
        if type(a) is not int or a not in legal_actions(state["mask"]) or b['candidate_key']['action_index'] != a or b['candidate_key_sha256'] != key_digest(b['candidate_key']):
            raise ValueError("CANDIDATE_IDENTITY")
        if b["target_task_id"] != (None if a == 24 else slots[a % 6]):
            raise ValueError("CANDIDATE_TARGET_SLOT")
        trace = b["trace"]
        if not trace or b["trace_sha256"] != digest(trace) or trace[0]["action"] != a:
            raise ValueError("TRACE_IDENTITY")
        clock = float(state["time"])
        for step in trace:
            if step["time_before"] != clock or step["time_after"] <= clock:
                raise ValueError("TIMELINE_DISCONTINUITY")
            if step["action"] not in legal_actions(step["mask"]):
                raise ValueError("TRACE_ILLEGAL_ACTION")
            clock = step["time_after"]
        if b["end_condition"] not in ("horizon", "terminated"):
            raise ValueError("INCOMPLETE_BRANCH")
        actual = sum(GAMMA ** i * (.4 * s["vector_reward"][0] + .2 * s["vector_reward"][1])
                     for i, s in enumerate(trace))
        if abs(actual - b["outcome"]["utility"]) > 1e-7:
            raise ValueError("UTILITY_RECOMPUTE")
        tasks = outcome_tasks(b["outcome"])
        if not set(t for t in slots if t is not None) <= set(tasks):
            raise ValueError("LIFECYCLE_MISSING_PUBLIC_TASK")
        if trace[-1].get("lifecycle_after") != b["outcome"]["tasks"]:
            raise ValueError("LABEL_LIFECYCLE_UNEXPLAINED")
        for t in tasks.values():
            completed = t.get("completed_at")
            deadline = t["deadline"]
            if t["on_time"]["valid"] and t["on_time"]["value"] != (completed is not None and completed <= deadline):
                raise ValueError("PHYSICAL_COMPLETION_LABEL")
            if t["expired"]["valid"] and t["expired"]["value"] != (t["state"] == "expired"):
                raise ValueError("PHYSICAL_EXPIRY_LABEL")
        if b["outcome"]["collision"]["valid"]:
            raise ValueError("UNIMPLEMENTED_COLLISION_LABEL")
    if set(outcome_tasks(candidate["outcome"])) != set(outcome_tasks(baseline["outcome"])):
        raise ValueError("LIFECYCLE_SET_MISMATCH")


def difference(a, b):
    if not a["valid"] or not b["valid"]:
        return leaf(reason="paired_label_unknown")
    return leaf(float(a["value"]) - float(b["value"]), True)


def labels(candidate, baseline, state, expected):
    from public_prefix import make_public_prefix, serialize_public_prefix
    public_prefix = serialize_public_prefix(make_public_prefix(state, expected['candidate_context'], candidate['candidate_key']))['record']
    paired(candidate, baseline, state, expected)
    c, b = candidate["outcome"], baseline["outcome"]
    ct, bt = outcome_tasks(c), outcome_tasks(b)
    target = candidate["target_task_id"]
    current = {k: ct[target][k] if target else leaf(reason="NOOP_has_no_current_task")
               for k in BINARY[:3] + ("host_confirmation",)}
    current.update({k: c[k] for k in ("damaged", "collision")})
    per_task = [{"task_id": tid, "completed_delta": difference(ct[tid]["on_time"], bt[tid]["on_time"]),
                 "expired_delta": difference(ct[tid]["expired"], bt[tid]["expired"])} for tid in sorted(ct)]
    other = [r for r in per_task if r["task_id"] != target]
    continuous = {}
    for out, field in (("other_completed_delta", "completed_delta"), ("other_expired_delta", "expired_delta")):
        continuous[out] = (leaf(sum(r[field]["value"] for r in other), True)
                           if all(r[field]["valid"] for r in other) else leaf(reason="other_task_unknown"))
    for out, field in (("energy_delta", "energy_used"), ("battery_delta", "battery_remaining"),
                       ("damage_count_delta", "damage_count"), ("competition_delta", "competition_count")):
        continuous[out] = difference(c[field], b[field])
    continuous["global_utility_delta"] = leaf(c["utility"] - b["utility"], True)
    return {"candidate_key": candidate['candidate_key'], 'candidate_key_sha256': candidate['candidate_key_sha256'], 'public_prefix': public_prefix,
            "binary": current, "continuous": continuous, "per_task": per_task,
            "slots": task_slots(state), "source_trace_sha256": candidate["trace_sha256"],
            "baseline_trace_sha256": baseline["trace_sha256"]}


def tensor_targets(label):
    """Unknown/absent placeholders are numerical only, with false masks."""
    values, valid = {}, {}
    for group, names in (("binary", BINARY), ("continuous", CONTINUOUS)):
        fields = [label[group][n] for n in names]
        values[group] = [float(f["value"]) if f["valid"] else 0. for f in fields]
        valid[group] = [f["valid"] for f in fields]
    by_id = {r["task_id"]: r for r in label["per_task"]}
    values["per_task"], valid["per_task"] = [], []
    for tid in label["slots"]:
        row = [leaf(reason="task_absent")] * 2 if tid is None else [by_id[tid][f] for f in SLOT_FIELDS]
        values["per_task"].append([float(f["value"]) if f["valid"] else 0. for f in row])
        valid["per_task"].append([f["valid"] for f in row])
    return values, valid


def admit_window(row, split_by_parent):
    validate_window(row, labels_required=False)
    if row["split"] != split_by_parent[row["parent"]]:
        raise ValueError("PARENT_SPLIT_MISMATCH")
    state = row["state_before"]
    task_slots(state)
    if not opportunity(state["mask"]):
        if row["status"] != "no_opportunity" or row.get("branches"):
            raise ValueError("NO_OPPORTUNITY_STATUS")
        return []
    if row["status"] != "complete":
        raise ValueError("WINDOW_INCOMPLETE")
    branches = row["branches"]
    by_key = {key_digest(b['candidate_key']): b for b in branches}
    if len(by_key) != len(branches) or set(by_key) != {key_digest(k) for k in row['candidate_keys']}:
        raise ValueError("MISSING_DUPLICATE_CANDIDATE")
    baseline_keys = [k for k in row['candidate_keys'] if k['action_index'] == row['baseline_action']]
    if type(row['baseline_action']) is not int or len(baseline_keys) != 1:
        raise ValueError('BASELINE_CANDIDATE_KEY_REQUIRED')
    baseline = by_key[key_digest(baseline_keys[0])]
    expected = dict(row['pair_identity'], candidate_context=row['candidate_context'])
    return [labels(by_key[key_digest(key)], baseline, state, expected) for key in row['candidate_keys']]
