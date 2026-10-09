"""Shared public candidate/ensemble interface for offline and online consumers."""
import torch
from consequence_contract import BINARY, CONTINUOUS, task_slots, opportunity, digest
from consequence_model import AUX_DIM, READOUT_DIM
from consequence_policy import FEATURE_ID
from gppo_world.graph5 import graph5_from_m10_observation
from gppo_world.joint_consequence_baseline import public_joint_action_score
from public_history import history_vector
from candidate_keys import validate_catalog, audit, bind_model_input, verify_model_inputs

SUPPORT_TIMES = (2., 4., 8., 12.)


def base_candidates(state):
    graph = graph5_from_m10_observation(state)
    result = torch.zeros(25, 17)
    for a in range(24):
        u, t = divmod(a, 6)
        result[a] = torch.cat((graph.candidate_features[a], graph.nodes["uav"][u, :4],
                               graph.nodes["task"][t, :4], torch.tensor([u / 3., t / 5., 0., state["time"] / 18., float(state["mask"][a])])))
    result[24, -3:] = torch.tensor([1., state["time"] / 18., float(state["mask"][24])])
    return result


def build_features(state, ensemble, supported_readouts, arm, ledger=None, *, candidate_context, candidate_keys):
    """Only public identity/presence/training support controls validity, never future labels."""
    validate_catalog(candidate_keys, candidate_context, state)
    from public_prefix import prefix_records
    slots = task_slots(state)
    base = base_candidates(state)
    aux = torch.zeros(25, AUX_DIM)
    valid = torch.zeros_like(aux, dtype=torch.bool)
    legal = torch.tensor(state["mask"], dtype=torch.bool)
    actions = torch.where(legal)[0]
    available = torch.zeros(25, dtype=torch.bool)
    trace = {"feature_contract": FEATURE_ID, "task_slots": slots,
             "public_input_sha256": digest(state), "support_times": list(SUPPORT_TIMES), "world_model_calls": 0}
    trace.update(candidate_context=candidate_context, candidate_keys=candidate_keys,
                 public_prefixes=prefix_records(state, candidate_context, candidate_keys),
                 candidate_audits={'graph_action_encoding': audit('graph_action_encoding', candidate_keys, candidate_context, state),
                                   'candidate_features': audit('candidate_features', candidate_keys, candidate_context, state)})
    if not opportunity(state["mask"]):
        trace["gate"] = "no_opportunity"
        return base, aux, valid, legal, available, trace
    if float(state["time"]) not in SUPPORT_TIMES:
        trace.update(gate="base_fallback", reason="outside_registered_time_support")
        return base, aux, valid, legal, available, trace
    graph = graph5_from_m10_observation(state)
    transparent = torch.tensor([[float(d["own_on_time_public"]), float(d["opportunity_cost"]), float(d["energy_cost"])]
                               for d in (public_joint_action_score(graph, int(a)) for a in actions)])
    if arm == "G0":
        trace["gate"] = "base_fallback"
        return base, aux, valid, legal, available, trace
    aux[actions, -3:] = transparent; valid[actions, -3:] = True; available[actions] = True
    if arm == "T":
        aux[actions, -3:] = transparent; valid[actions, -3:] = True; available[actions] = True
        trace["gate"] = "feature_available"
        return base, aux, valid, legal, available, trace
    if arm != "G1": raise ValueError("ARM_CONTRACT")
    if ensemble is None or supported_readouts is None:
        trace.update(gate="base_fallback", reason="prediction_unavailable")
        return base, aux, valid, legal, available, trace
    support = torch.as_tensor(supported_readouts, dtype=torch.bool)[None].expand(len(actions), -1).clone()
    if support.shape != (len(actions), READOUT_DIM): raise ValueError("SUPPORT_READOUT_DIM")
    for row, a in enumerate(actions):
        if a == 24:
            support[row, :3] = False; support[row, BINARY.index("host_confirmation")] = False
        for s, tid in enumerate(slots):
            if tid is None:
                start = len(BINARY) + len(CONTINUOUS) + 2*s
                support[row, start:start+2] = False
    nodes = {k: v[None].expand(len(actions), -1, -1).contiguous() for k, v in graph.nodes.items()}
    history = torch.tensor(history_vector(state))[None].expand(len(actions), -1).contiguous()
    relations = torch.cat((graph.candidate_features, torch.zeros(1, 4)))[actions]
    binding = bind_model_input(state, candidate_context, candidate_keys, nodes, history, actions, relations)
    verify_model_inputs([binding], nodes, history, actions, relations)
    trace['candidate_audits']['world_model_input'] = binding['audit']
    def infer(): return ensemble.predict(nodes, history, actions, relations, legal, support, transparent, [binding])
    result = (ledger.call("ensemble_candidate_batch", {"world_model_batch_forwards": 3,
              "world_model_candidate_evaluations": 3*len(actions)}, infer) if ledger else infer())
    x, v, outputs = result
    aux[actions] = x; valid[actions] = v; available[actions] = True
    trace['seed_readouts'] = [__import__('consequence_model').readout(o).tolist() for o in outputs]
    trace.update(gate="feature_available" if available.any() else "base_fallback", world_model_calls=3,
                 seed_order=[8201, 8202, 8203], latent_fusion="ordered_concatenation",
                 uncertainty_interpretation="uncalibrated_ensemble_std_not_failure_probability")
    return base, aux, valid, legal, available, trace
