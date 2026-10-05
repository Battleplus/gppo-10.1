"""Authorized-runner-only training boundary for the W1 world model.

Importing this module does not instantiate an optimizer or execute an update.
The runner must supply already split, causally valid batches.  Future graph
payloads are accepted only as target-side data and never passed to the online
inference API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from event_jepa_losses import jepa_objective
from w1_graph_jepa import W1GraphJEPA


@dataclass(frozen=True)
class WorldModelBatch:
    current_nodes: dict[str, torch.Tensor]
    history: torch.Tensor
    actions: torch.Tensor
    relations: torch.Tensor
    target_nodes: dict[str, torch.Tensor]
    jepa_valid: torch.Tensor
    state_target: torch.Tensor | None = None
    state_valid: torch.Tensor | None = None
    outcome_target: torch.Tensor | None = None
    outcome_valid: torch.Tensor | None = None
    event_labels: torch.Tensor | None = None
    event_valid: torch.Tensor | None = None


def train_world_model(model: W1GraphJEPA, batches: Iterable[WorldModelBatch], *, optimizer: torch.optim.Optimizer, event_enabled: bool, beta_state: float = 1.0, beta_outcome: float = 1.0, beta_event: float = 1.0, beta_collapse: float = 0.04, max_updates: int) -> dict[str, float | int]:
    """Run the explicitly authorized supervised JEPA loop.

    This function is intentionally not called by preparation tests.  It rejects
    an unbounded iterator and reports aggregate losses only; per-candidate
    predictions remain the runner's trace responsibility.
    """
    if max_updates <= 0:
        raise ValueError("max_updates must be positive")
    model.train()
    totals = {"updates": 0, "loss": 0.0, "jepa": 0.0, "state": 0.0, "outcome": 0.0, "event": 0.0, "collapse": 0.0}
    for batch in batches:
        if totals["updates"] >= max_updates:
            break
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.current_nodes, batch.history, batch.actions, batch.relations)
        with torch.no_grad():
            target = model.target(batch.target_nodes)
        target = target.detach()
        if target.shape != output["latent"].shape:
            raise ValueError("target graph rows must align one-to-one with candidate rows")
        total, parts = jepa_objective(
            output["latent"], target, batch.jepa_valid,
            state_prediction=output["public_state"], state_target=batch.state_target, state_valid=batch.state_valid,
            outcome_prediction=output["outcome"], outcome_target=batch.outcome_target, outcome_valid=batch.outcome_valid,
            event_logits=output["event_logits"], event_labels=batch.event_labels, event_valid=batch.event_valid,
            beta_state=beta_state, beta_outcome=beta_outcome,
            beta_event=beta_event if event_enabled else 0.0, beta_collapse=beta_collapse,
        )
        total.backward()
        optimizer.step()
        model.update_target()
        totals["updates"] += 1
        for name in ("loss", "jepa", "state", "outcome", "event", "collapse"):
            value = total if name == "loss" else parts[name]
            totals[name] += float(value.detach())
    if totals["updates"] == 0:
        raise ValueError("no training batches were consumed")
    count = float(totals["updates"])
    return {key: (value / count if key != "updates" else value) for key, value in totals.items()}


def freeze_world_model(model: W1GraphJEPA) -> W1GraphJEPA:
    """Freeze all world-model parameters before GPPO/adapter training."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


__all__ = ["WorldModelBatch", "train_world_model", "freeze_world_model"]
