"""Fresh Graph-JEPA with masked physical heads and utility pairwise ranking."""
import torch
from torch import nn
from torch.nn import functional as F
from vendor.w1_graph_jepa import W1GraphJEPA
from consequence_contract import BINARY, CONTINUOUS, SEEDS
from candidate_keys import verify_model_inputs, key_digest

# Preregistered, never fitted on development/evaluation results.
WEIGHTS = {"latent": 1., "public": .1, "binary": 1., "continuous": 1.,
           "per_task": 1., "ranking": 1., "anticollapse": .01}
READOUT_DIM = len(BINARY) + len(CONTINUOUS) + 12
AUX_DIM = 3 * 64 + READOUT_DIM * 2 + 3
PACKED_DIM = 17 + 2 * AUX_DIM


class ConsequenceJEPA(W1GraphJEPA):
    format_version = "drone-action-consequence-jepa/2-candidate-key"

    def __init__(self):
        super().__init__()
        # Remove obsolete heads so unused old objectives cannot be mistaken for new supervision.
        del self.outcome_head, self.horizon_task_outcome_head, self.event_head
        self.physical_head = nn.Linear(64, len(BINARY))
        self.increment_head = nn.Linear(64, len(CONTINUOUS))
        self.externality_head = nn.Linear(64, 12)

    def forward(self, nodes, history, actions, relations, candidate_bindings=None):
        verify_model_inputs(candidate_bindings, nodes, history, actions, relations)
        graph, h = self.encode(nodes, history)
        z = self.predictor(torch.cat((graph, h, self.action_embedding(actions.long()),
                                     self.relation_embedding(relations)), -1))
        return {'candidate_keys': [key for binding in candidate_bindings for key in binding['keys']],
                'candidate_key_sha256': [key_digest(key) for binding in candidate_bindings for key in binding['keys']],
                "latent": z, "public": self.public_decoder(z), "binary": self.physical_head(z),
                "continuous": self.increment_head(z), "per_task": self.externality_head(z).reshape(-1, 6, 2)}

    def predict_candidates(self, nodes, history, actions, relations, candidate_bindings=None):
        verify_model_inputs(candidate_bindings, nodes, history, actions, relations)
        return self.forward(nodes, history, actions, relations, candidate_bindings)


def masked_loss(prediction, target, valid, kind):
    if prediction.shape != target.shape or valid.shape != target.shape or valid.dtype != torch.bool:
        raise ValueError("TARGET_MASK_SHAPE")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target[valid]).all():
        raise ValueError("VALID_TARGET_NONFINITE")
    if not valid.any():
        return prediction.sum() * 0.
    p, t = prediction[valid], target[valid]
    if kind == "bce":
        if ((t < 0) | (t > 1)).any():
            raise ValueError("BCE_TARGET_RANGE")
        return F.binary_cross_entropy_with_logits(p, t)
    return F.smooth_l1_loss(p, t, beta=1.)


def ranking_loss(predicted, truth, valid, window_ids):
    """Compare only valid candidates in the same paired decision window."""
    if len(window_ids) != len(predicted):
        raise ValueError("RANK_WINDOW_SHAPE")
    losses = []
    for i in range(len(predicted)):
        for j in range(i + 1, len(predicted)):
            if window_ids[i] == window_ids[j] and valid[i] and valid[j] and truth[i] != truth[j]:
                direction = torch.sign(truth[i] - truth[j])
                losses.append(F.softplus(-direction * (predicted[i] - predicted[j])))
    return torch.stack(losses).mean() if losses else predicted.sum() * 0.


def consequence_loss(pred, targets, masks, window_ids):
    parts = {"binary": masked_loss(pred["binary"], targets["binary"], masks["binary"], "bce"),
             "continuous": masked_loss(pred["continuous"], targets["continuous"], masks["continuous"], "huber"),
             "per_task": masked_loss(pred["per_task"], targets["per_task"], masks["per_task"], "huber"),
             "public": masked_loss(pred["public"], targets["public"], masks["public"], "huber"),
             "latent": masked_loss(pred["latent"], targets["latent"].detach(), masks["latent"], "huber")}
    u = CONTINUOUS.index("global_utility_delta")
    parts["ranking"] = ranking_loss(pred["continuous"][:, u], targets["continuous"][:, u], masks["continuous"][:, u], window_ids)
    z = pred["latent"][masks["latent"].all(-1)]
    parts["anticollapse"] = (F.relu(.1 - z.std(0, unbiased=False)).mean() if len(z) > 1 else pred["latent"].sum() * 0.)
    return sum(WEIGHTS[k] * v for k, v in parts.items()), parts


def readout(output):
    return torch.cat((output["binary"].sigmoid(), output["continuous"], output["per_task"].flatten(1)), -1)


class FrozenEnsemble(nn.Module):
    """Only common-unit readouts are averaged; seed latent axes are concatenated."""
    def __init__(self, models, seed_order=SEEDS):
        super().__init__()
        if tuple(seed_order) != SEEDS or len(models) != 3:
            raise ValueError("FIXED_ALL_THREE_SEEDS_REQUIRED")
        self.models = nn.ModuleList(models).requires_grad_(False).eval()
        self.models.zero_grad(set_to_none=True)
        self.seed_order = SEEDS
        self.calls = 0

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def predict(self, nodes, history, actions, relations, legal_mask, support_mask, transparent, candidate_bindings=None):
        verify_model_inputs(candidate_bindings, nodes, history, actions, relations)
        if not opportunity_tensor(legal_mask):
            raise ValueError("OUTSIDE_SUPPORT_MUST_SKIP_PREDICT")
        expected = torch.where(legal_mask)[0]
        if not torch.equal(actions.cpu(), expected.cpu()):
            raise ValueError("LEGAL_CANDIDATE_BATCH_IDENTITY")
        if support_mask.shape != (len(actions), READOUT_DIM) or support_mask.dtype != torch.bool:
            raise ValueError("SUPERVISION_SUPPORT_MASK")
        out = [m.predict_candidates(nodes, history, actions, relations, candidate_bindings) for m in self.models]
        expected_keys = [key for binding in candidate_bindings for key in binding['keys']]
        if any(o['candidate_keys'] != expected_keys or o['candidate_key_sha256'] != [key_digest(k) for k in expected_keys] for o in out):
            raise ValueError('CANDIDATE_PREDICTION_OUTPUT_IDENTITY')
        self.calls += 3
        reads = torch.stack([readout(p) for p in out])
        aux = torch.cat((*[p["latent"] for p in out], reads.mean(0), reads.var(0, unbiased=False).sqrt(), transparent), -1)
        valid = torch.cat((torch.ones((len(actions), 192), dtype=torch.bool, device=aux.device), support_mask.to(aux.device), support_mask.to(aux.device),
                           torch.isfinite(transparent)), -1).to(aux.device)
        return aux.detach(), valid, out


def opportunity_tensor(mask):
    if mask.dtype != torch.bool or mask.shape != (25,) or not mask.any():
        raise ValueError("LEGAL_MASK_TENSOR")
    return int(mask.sum()) >= 2 and bool(mask[:24].any())
