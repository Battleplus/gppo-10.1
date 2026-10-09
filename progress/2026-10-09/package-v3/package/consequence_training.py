"""New data/tensor/checkpoint hooks; fixed final models, never selects a seed."""
import json
import random
from pathlib import Path
import torch
from consequence_contract import admit_window, tensor_targets, digest, SEEDS, BINARY, CONTINUOUS
from consequence_model import ConsequenceJEPA, consequence_loss, WEIGHTS, READOUT_DIM, PACKED_DIM
from consequence_policy import FEATURE_ID
from consequence_replay import fingerprint
from gppo_world.graph5 import graph5_from_m10_observation
from public_history import history_vector, public_state_vector
from candidate_keys import SCHEMA as CANDIDATE_SCHEMA, validate_window, bind_model_input, verify_model_inputs, validate_key, key_digest

EPOCHS = 40
WINDOW_BATCH = 8


def restore_optimizer_strict(model, payload, *, include_frozen=False):
    optimizer=torch.optim.Adam([p for p in model.parameters() if include_frozen or p.requires_grad],lr=3e-4)
    optimizer.load_state_dict(payload)
    for p,state in optimizer.state.items():
        if set(state)!={'step','exp_avg','exp_avg_sq'}: raise ValueError('ADAM_STATE_FIELDS')
        if state['exp_avg'].shape!=p.shape or state['exp_avg_sq'].shape!=p.shape:
            raise ValueError('ADAM_STATE_SHAPE')
        if any(not torch.isfinite(x).all() for x in state.values()): raise ValueError('ADAM_STATE_NONFINITE')
        if state['step'].numel()!=1 or float(state['step'])<0: raise ValueError('ADAM_STEP')
    if fingerprint(optimizer.state_dict())!=fingerprint(payload): raise ValueError('ADAM_STRICT_RESTORE')
    return optimizer


def load_windows(path, split_by_parent):
    """Single streaming read, admission and duplicate rejection before tensors."""
    seen, rows = set(), []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identity = (row["parent"], row["repeat"], row["window_id"])
            if identity in seen: raise ValueError("DUPLICATE_WINDOW")
            seen.add(identity)
            derived = admit_window(row, split_by_parent)
            if row["status"] == "complete":
                if row["labels"] != derived: raise ValueError("SAVED_LABEL_RECOMPUTE")
                rows.append(row)
    return rows, seen


def prepare_world_input(row):
    validate_window(row)
    graph = graph5_from_m10_observation(row["state_before"])
    actions = torch.where(graph.action_mask)[0]
    nodes = {k: v[None].expand(len(actions), -1, -1).contiguous() for k, v in graph.nodes.items()}
    history = torch.tensor(history_vector(row["state_before"]))[None].expand(len(actions), -1).contiguous()
    relation = torch.cat((graph.candidate_features, torch.zeros(1, 4)))[actions]
    labels = row['labels'] if row.get('admitted_lean') else admit_window(row, {row["parent"]: row["split"]})
    values, validity = zip(*(tensor_targets(l) for l in labels))
    targets = {k: torch.tensor([v[k] for v in values], dtype=torch.float32) for k in ("binary", "continuous", "per_task")}
    masks = {k: torch.tensor([v[k] for v in validity], dtype=torch.bool) for k in targets}
    by_key = {key_digest(b['candidate_key']): b for b in row['branches']}
    final = [by_key[key_digest(key)]['final_public'] if row.get('admitted_lean') else
             by_key[key_digest(key)]['trace'][-1]['public_after'] for key in row['candidate_keys']]
    binding = bind_model_input(row['state_before'], row['candidate_context'], row['candidate_keys'], nodes, history, actions, relation)
    verify_model_inputs([binding], nodes, history, actions, relation)
    return nodes, history, actions, relation, targets, masks, final, binding


def window_tensors(row, model):
    nodes, history, actions, relation, targets, masks, final, binding = prepare_world_input(row)
    targets["public"] = torch.tensor([public_state_vector(s).tolist() for s in final])
    masks["public"] = torch.ones_like(targets["public"], dtype=torch.bool)
    masks["public"][:, 0::4] = targets["public"][:, 2::4] > .5
    future_graphs = [graph5_from_m10_observation(s) for s in final]
    future_nodes = {k: torch.stack([g.nodes[k] for g in future_graphs]) for k in nodes}
    targets["latent"] = model.target(future_nodes).detach()
    masks["latent"] = torch.ones_like(targets["latent"], dtype=torch.bool)
    return nodes, history, actions, relation, targets, masks, binding


def support_from_training(train_rows):
    masks = []
    for row in train_rows:
        validate_window(row)
        if row["split"] != "train": raise ValueError("TRAIN_ONLY_SUPPORT_REQUIRED")
        for label in row["labels"]:
            _, v = tensor_targets(label)
            masks.append(v["binary"] + v["continuous"] + [x for slot in v["per_task"] for x in slot])
    if not masks: raise ValueError("NO_TRAIN_LABELS")
    counts = torch.tensor(masks, dtype=torch.int64).sum(0)
    return counts > 0, counts


def train_final_world(train_rows, seed, ledger, output, *, epochs=EPOCHS):
    if seed not in SEEDS or any(r["split"] != "train" for r in train_rows):
        raise ValueError("WORLD_SEED_OR_SPLIT")
    torch.manual_seed(seed)
    model = ledger.call('world_initialize', {'model_initializations': 1}, ConsequenceJEPA)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    order_rng = random.Random(seed)
    updates = 0
    for epoch in range(epochs):
        order = list(range(len(train_rows))); order_rng.shuffle(order)
        for start in range(0, len(order), WINDOW_BATCH):
            rows = [train_rows[i] for i in order[start:start + WINDOW_BATCH]]
            def step():
                tensors = [window_tensors(row, model) for row in rows]
                n = {k: torch.cat([t[0][k] for t in tensors]) for k in tensors[0][0]}
                h, a, rel = [torch.cat([t[i] for t in tensors]) for i in (1, 2, 3)]
                target = {k: torch.cat([t[4][k] for t in tensors]) for k in tensors[0][4]}
                valid = {k: torch.cat([t[5][k] for t in tensors]) for k in tensors[0][5]}
                bindings = [t[6] for t in tensors]
                verify_model_inputs(bindings, n, h, a, rel)
                window_ids = [f'{r["parent"]}/{r["window_id"]}' for r, t in zip(rows, tensors) for _ in t[2]]
                prediction = ledger.call('world_forward', {'training_forwards': 1,
                    'training_candidate_evaluations': len(a)}, model, n, h, a, rel, bindings)
                loss, parts = consequence_loss(prediction, target, valid, window_ids)
                if not torch.isfinite(loss): raise ValueError("WORLD_LOSS_NONFINITE")
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
                optimizer.step(); model.update_target()
                return {k: float(v.detach()) for k, v in parts.items()}
            ledger.call("world_update", {"world_model_updates": 1, 'target_forwards': len(rows),
                'target_candidate_evaluations': sum(len(r['labels']) for r in rows)}, step)
            updates += 1
    supported, counts = support_from_training(train_rows)
    metadata = {"schema": model.format_version, "seed": seed, "epochs": epochs, "updates": updates,
        "selection_rule": "final_fixed_epoch_only", "weights": WEIGHTS, "task_capacity": 6,
        "slot_contract": "public-action-graph-order", "support_mask": supported.tolist(), "support_counts": counts.tolist(),
        "train_parents": sorted({r["parent"] for r in train_rows}), "continuation_id": "public-joint-opportunity-rule/1",
        "feature_contract": FEATURE_ID, "candidate_key_contract": CANDIDATE_SCHEMA, "public_prefix_contract": 'w1-public-prefix/1',
        "readout_dim": READOUT_DIM, "packed_dim": PACKED_DIM}
    payload = {"metadata": metadata, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
               "optimizer_sha256": fingerprint(optimizer.state_dict()), 'metadata_sha256': digest(metadata),
               "state_sha256": fingerprint(model.state_dict())}
    target = Path(output); temporary = target.with_suffix(".pending")
    ledger.call("save_world", {"checkpoint_writes": 1}, torch.save, payload, temporary)
    admitted = ledger.call('world_save_admission', {'checkpoint_loads': 1}, admit_checkpoint,
                           temporary, digest(temporary.read_bytes()), metadata)
    temporary.replace(target)
    model.zero_grad(set_to_none=True)
    return model.requires_grad_(False).eval(), admitted


def admit_checkpoint(path, file_sha256, expected_metadata):
    path = Path(path)
    if digest(path.read_bytes()) != file_sha256: raise ValueError("CHECKPOINT_FILE_SHA256")
    # These are new locally produced tensor/dict checkpoints, no legacy pickle modules.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["metadata"] != expected_metadata:
        issues = [k for k in set(expected_metadata) | set(payload["metadata"]) if expected_metadata.get(k) != payload["metadata"].get(k)]
        raise ValueError("CHECKPOINT_METADATA:" + ",".join(sorted(issues)))
    if fingerprint(payload["state_dict"]) != payload["state_sha256"]: raise ValueError("CHECKPOINT_INTERNAL_STATE_SHA256")
    if digest(payload['metadata']) != payload['metadata_sha256']: raise ValueError('CHECKPOINT_METADATA_DIGEST')
    if fingerprint(payload['optimizer']) != payload['optimizer_sha256']: raise ValueError('CHECKPOINT_OPTIMIZER_DIGEST')
    if payload["metadata"].get('public_prefix_contract') != 'w1-public-prefix/1' or payload["metadata"].get('candidate_key_contract') != CANDIDATE_SCHEMA or payload["metadata"]["task_capacity"] != 6 or payload["metadata"]["packed_dim"] != PACKED_DIM:
        raise ValueError("CHECKPOINT_MODEL_SLOT_CONTRACT")
    return {"file_sha256": file_sha256, "state_sha256": payload["state_sha256"],
            "optimizer_sha256": payload["optimizer_sha256"], "metadata": payload["metadata"]}


def serialize_prediction(candidate_key, context, state, task_ids, predicted, valid):
    validate_key(candidate_key, context, state)
    from public_prefix import make_public_prefix, serialize_public_prefix
    public_prefix = serialize_public_prefix(make_public_prefix(state, context, candidate_key))['record']
    if task_ids != __import__('consequence_contract').task_slots(state) or len(predicted) != READOUT_DIM or len(valid) != READOUT_DIM:
        raise ValueError("PREDICTION_IDENTITY_LAYOUT")
    prefix = len(BINARY) + len(CONTINUOUS)
    return {"action": candidate_key['action_index'], "candidate_key": candidate_key,
        'public_prefix': public_prefix,
        'candidate_key_sha256': key_digest(candidate_key),
        "binary": {k: {"value": float(predicted[i]) if valid[i] else None, "valid": bool(valid[i])} for i, k in enumerate(BINARY)},
        "continuous": {k: {"value": float(predicted[len(BINARY) + i]) if valid[len(BINARY) + i] else None,
            "valid": bool(valid[len(BINARY) + i])} for i, k in enumerate(CONTINUOUS)},
        "per_task": [{"task_id": tid, "slot": s, "exists": tid is not None,
            "delta": [float(predicted[prefix + 2*s + i]) if tid is not None and valid[prefix + 2*s + i] else None for i in (0, 1)],
            "valid": [bool(tid is not None and valid[prefix + 2*s + i]) for i in (0, 1)]} for s, tid in enumerate(task_ids)]}
