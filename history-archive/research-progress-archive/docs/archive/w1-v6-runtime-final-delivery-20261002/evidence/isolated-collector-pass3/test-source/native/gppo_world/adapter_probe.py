"""Opt-in evaluation observer; never changes model weights or action selection."""
from __future__ import annotations

from typing import Any

import torch

from .gppo_adapter import LatentAugmentedActorCritic


class AdapterProbe:
    """Single-threaded probe comparing the SAME checkpoint's base and adapter.

    Hooks observe the one original forward; there is no second policy inference,
    no second context-store read, and no counterfactual action submission.
    The embedded, co-trained base is NOT an independently trained GPPO control.
    """

    def __init__(self, model: LatentAugmentedActorCritic):
        self.model = model
        self.records: list[dict[str, Any]] = []

    @torch.no_grad()
    def act(self, graph: Any, deterministic: bool = False):
        if self.model.training:
            raise ValueError("AdapterProbe is restricted to eval-mode policies")
        selected = self.model._resolve_context(graph, None)
        captured: dict[str, Any] = {}

        def capture_base(module, args, output):
            captured["base"] = output

        def capture_final(module, args, output):
            captured["final"] = output

        handles = [self.model.base_model.register_forward_hook(capture_base),
                   self.model.register_forward_hook(capture_final)]
        mask_before = graph.action_mask.clone()
        version_before = graph.graph_version
        try:
            result = self.model.act(graph, deterministic=deterministic, context=selected)
        finally:
            for handle in handles:
                handle.remove()
        base_logits, base_value, _ = captured["base"]
        logits, value, diagnostics = captured["final"]
        mask = graph.action_mask
        if not torch.equal(mask_before, mask) or version_before != graph.graph_version:
            raise RuntimeError("Policy/probe mutated graph mask or version")
        p = torch.softmax(base_logits[mask].double(), dim=-1)
        q = torch.softmax(logits[mask].double(), dim=-1)
        used = bool(diagnostics.get("latent_adapter_used", False))
        if used:
            reason = "used"
        elif not self.model.enabled:
            reason = "disabled"
        elif selected is None:
            reason = "missing_context"
        elif selected.fallback_reason:
            reason = selected.fallback_reason
        else:
            reason = "unusable_or_zero_context"
        latent = torch.tensor(selected.latent) if selected is not None else torch.zeros(1)
        self.records.append({
            "graph_version": int(version_before),
            "context_graph_version": None if selected is None else selected.post_graph_version,
            "context_action_version": None if selected is None else selected.post_action_version,
            "context_source_step": None if selected is None else selected.source_step,
            "context_valid": False if selected is None else selected.valid,
            "reason": reason,
            "legal_action_count": int(mask.sum()),
            "selected_action": int(result[0]),
            "embedded_base_argmax": int(base_logits.argmax()),
            "augmented_argmax": int(logits.argmax()),
            "argmax_disagreement": bool(base_logits.argmax() != logits.argmax()),
            "legal_probability_total_variation": float(0.5 * torch.abs(p - q).sum()),
            "legal_actor_residual_l2": float((logits[mask] - base_logits[mask]).norm()),
            "critic_residual": float(value - base_value),
            "latent_l2": float(latent.norm()),
            "latent_abs_max": float(latent.abs().max()),
            "invalid_logits_unchanged": bool(torch.equal(base_logits[~mask], logits[~mask])),
            "mask_and_version_unchanged": True,
        })
        return result
