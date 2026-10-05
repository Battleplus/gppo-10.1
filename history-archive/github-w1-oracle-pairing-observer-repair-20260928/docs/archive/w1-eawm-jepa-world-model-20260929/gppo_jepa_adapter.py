"""Frozen-world-model candidate prior for the Graph-5 GPPO interface."""
from __future__ import annotations

import torch
from torch import nn


ACTION_COUNT = 25
UTILITY_RESIDUAL_INDEX = 4


class GPPOSafeJEPAAdapter(nn.Module):
    def __init__(self, latent_dim: int = 64, feature_dim: int = 8, *, scale: float = 0.0):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(latent_dim, 32), nn.Tanh(), nn.Linear(32, feature_dim))
        self.score = nn.Linear(feature_dim, 1)
        self.scale = float(scale)

    def forward(self, base_logits: torch.Tensor, predicted_latent: torch.Tensor, legal_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if base_logits.ndim != 2 or base_logits.shape[-1] != ACTION_COUNT:
            raise ValueError("base_logits must be [B,25]")
        if predicted_latent.shape[:2] != base_logits.shape:
            raise ValueError("predicted_latent must be [B,25,D]")
        mask = legal_mask.bool()
        if mask.shape != base_logits.shape or not bool(mask.any(dim=-1).all()):
            raise ValueError("legal mask must be [B,25] with at least one legal action")
        features = self.projection(predicted_latent)
        prior = self.score(features).squeeze(-1) * self.scale
        prior = prior.masked_fill(~mask, 0.0)
        final = (base_logits + prior).masked_fill(~mask, -torch.inf)
        return {"base_logits": base_logits, "prior": prior, "final_logits": final, "features": features}


class TransparentResidualPrior(nn.Module):
    """Interpretable prior: transparent original-utility score plus learned residual."""
    def __init__(self, *, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, base_logits: torch.Tensor, transparent_scores: torch.Tensor, predicted_outcomes: torch.Tensor, legal_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if base_logits.shape != transparent_scores.shape or base_logits.ndim != 2 or base_logits.shape[-1] != ACTION_COUNT:
            raise ValueError("base and transparent scores must be [B,25]")
        if predicted_outcomes.shape[:2] != base_logits.shape or predicted_outcomes.shape[-1] <= UTILITY_RESIDUAL_INDEX:
            raise ValueError("predicted outcomes must be [B,25,K]")
        mask = legal_mask.bool()
        if mask.shape != base_logits.shape or not bool(mask.any(dim=-1).all()):
            raise ValueError("legal mask must be [B,25] with at least one legal action")
        residual = predicted_outcomes[..., UTILITY_RESIDUAL_INDEX]
        consequence = transparent_scores + residual
        prior = (self.scale * consequence).masked_fill(~mask, 0.0)
        final = (base_logits + prior).masked_fill(~mask, -torch.inf)
        return {"base_logits": base_logits, "transparent": transparent_scores, "residual": residual, "prior": prior, "final_logits": final}


def select_action(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[-1] != ACTION_COUNT:
        raise ValueError("logits must be [B,25]")
    if not torch.isfinite(logits.masked_fill(torch.isinf(logits), 0.0)).all():
        raise FloatingPointError("non-finite candidate logits")
    return logits.argmax(dim=-1)


__all__ = ["GPPOSafeJEPAAdapter", "TransparentResidualPrior", "UTILITY_RESIDUAL_INDEX", "select_action"]
