"""Equal-capacity GPPO adapters; first phase changes vector critic only."""
import torch
from torch import nn
from gppo_world.m10_training import masked_distribution
from consequence_model import AUX_DIM, PACKED_DIM

FEATURE_ID = "drone-consequence/critic-common-transparent+latent-8201-8202-8203/2"


class ConsequencePolicy(nn.Module):
    consequence_experiment = True

    def __init__(self, base, arm="G1", phase="critic_only"):
        super().__init__()
        if arm not in ("G0", "T", "G1") or phase != "critic_only":
            raise ValueError("PHASE2_REQUIRES_SEPARATE_PROTOCOL_AND_AUTHORIZATION")
        self.base, self.arm, self.phase = base, arm, phase
        self.critic_adapter = nn.Sequential(nn.Linear(PACKED_DIM, 32), nn.Tanh(), nn.Linear(32, 2))
        self.actor_adapter = nn.Sequential(nn.Linear(PACKED_DIM, 32), nn.Tanh(), nn.Linear(32, 1))
        nn.init.zeros_(self.critic_adapter[-1].weight)
        nn.init.zeros_(self.critic_adapter[-1].bias)
        nn.init.zeros_(self.actor_adapter[-1].weight)
        nn.init.zeros_(self.actor_adapter[-1].bias)
        self.actor_adapter.requires_grad_(False)
        self.register_buffer("normalization_mean", torch.zeros(AUX_DIM))
        self.register_buffer("normalization_scale", torch.ones(AUX_DIM))

    def encode(self, *args, **kwargs):
        return self.base.encode(*args, **kwargs)

    def pack(self, public17, auxiliary, valid, legal_mask, available):
        if public17.shape != (25, 17) or auxiliary.shape != (25, AUX_DIM) or valid.shape != auxiliary.shape:
            raise ValueError("FEATURE_LAYOUT")
        if valid.dtype != torch.bool or legal_mask.dtype != torch.bool or available.dtype != torch.bool:
            raise ValueError("FEATURE_VALIDITY_TYPE")
        if legal_mask.shape != (25,) or available.shape != (25,):
            raise ValueError("AVAILABILITY_LAYOUT")
        if not torch.isfinite(public17).all() or not torch.isfinite(auxiliary[valid]).all():
            raise ValueError("FEATURE_NONFINITE")
        v = valid.clone() & legal_mask[:, None] & available[:, None]
        if self.arm == "G0":
            v.zero_()
        elif self.arm == "T":
            v[:, :-3] = False
        normalized = (auxiliary - self.normalization_mean) / self.normalization_scale
        clean = torch.where(v, normalized, torch.zeros_like(normalized))
        return torch.cat((public17, clean, v.float()), -1).detach()

    def evaluate_encoded(self, features, pair_messages, preference, candidate_features, mask):
        if candidate_features.shape != (features.shape[0], 25, PACKED_DIM):
            raise ValueError("REPLAY_FEATURE_LAYOUT")
        if mask.dtype != torch.bool or not mask.any(-1).all():
            raise ValueError("LEGAL_MASK")
        if not torch.isfinite(candidate_features).all():
            raise ValueError("REPLAY_NONFINITE_FEATURES")
        result = dict(self.base.evaluate_encoded(features, pair_messages, preference, candidate_features[..., :17], mask))
        # Detached behavior WM features, current adapter parameters, real rollout value targets.
        candidate_values = self.critic_adapter(candidate_features.detach())
        pooled = (result["probabilities"].detach()[..., None] * candidate_values).sum(1)
        value = result["critic_values"] + pooled
        result.update(critic_values=value, values=value, state_values=value)
        # Actor logits/distribution deliberately retain the exact base calculation.
        if (result["probabilities"][~mask] != 0).any():
            raise ValueError("ILLEGAL_ACTION_PROBABILITY")
        return result


def features_or_base(ensemble, mask, predict):
    """No predictor call outside opportunity support; missing evidence is fallback."""
    from consequence_model import opportunity_tensor
    if not opportunity_tensor(mask):
        return None, {"gate": "no_opportunity", "world_model_calls": 0}
    if ensemble is None:
        return None, {"gate": "base_fallback", "reason": "prediction_unavailable", "world_model_calls": 0}
    return predict(), {"gate": "feature_available", "world_model_calls": 3}
