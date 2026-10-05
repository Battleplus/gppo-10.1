"""Production-facing W1 Graph-5 -> JEPA -> GPPO candidate bridge.

This module has no environment ownership and no training loop.  It consumes a
detached Graph-5 snapshot plus the current public history, performs one batch
prediction for all legal candidates, and returns an auditable logit trace.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import torch

from gppo_jepa_adapter import TransparentResidualPrior
from public_transition_contract import public_hash
from w1_public_adapter import W1PublicInput
from w1_graph_jepa import ACTION_COUNT, W1GraphJEPA, expand_candidate_batch


@dataclass(frozen=True)
class W1CandidateTrace:
    input_hash: str
    continuation_id: str
    active_continuation_ids: tuple[str, ...]
    legal_actions: tuple[int, ...]
    candidate_actions: tuple[int, ...]
    selected_action: int
    base_logits: tuple[float, ...]
    transparent_scores: tuple[float, ...]
    learned_residuals: tuple[float, ...]
    prior_logits: tuple[float, ...]
    final_logits: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_hash": self.input_hash,
            "continuation_id": self.continuation_id,
            "active_continuation_ids": list(self.active_continuation_ids),
            "legal_actions": list(self.legal_actions),
            "candidate_actions": list(self.candidate_actions),
            "selected_action": self.selected_action,
            "base_logits": list(self.base_logits),
            "transparent_scores": list(self.transparent_scores),
            "learned_residuals": list(self.learned_residuals),
            "prior_logits": list(self.prior_logits),
            "final_logits": list(self.final_logits),
        }


def _digest_public(snapshot: Mapping[str, Any]) -> str:
    payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=lambda value: value.tolist() if hasattr(value, "tolist") else value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def graph5_candidate_inputs(snapshot: Any, history: torch.Tensor, *, device: torch.device | str = "cpu") -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract one decision window without reading any future branch fields."""
    if not hasattr(snapshot, "nodes") or not hasattr(snapshot, "candidate_features") or not hasattr(snapshot, "action_mask"):
        raise TypeError("snapshot must expose detached Graph-5 nodes, candidate_features and action_mask")
    nodes = {name: value.to(device).unsqueeze(0) for name, value in snapshot.nodes.items()}
    history = torch.as_tensor(history, dtype=torch.float32, device=device)
    if history.ndim == 1:
        history = history.unsqueeze(0)
    mask = torch.as_tensor(snapshot.action_mask, dtype=torch.bool, device=device).reshape(-1)
    if mask.shape != (ACTION_COUNT,) or not bool(mask.any()):
        raise ValueError("Graph-5 action mask must have 25 actions and one legal action")
    candidate_actions = torch.arange(ACTION_COUNT, dtype=torch.long, device=device)[mask]
    relations = torch.as_tensor(snapshot.candidate_features, dtype=torch.float32, device=device).reshape(24, 4)
    noop = torch.zeros((1, 4), dtype=torch.float32, device=device)
    relations = torch.cat((relations, noop), dim=0)[candidate_actions]
    expanded_nodes, expanded_history = expand_candidate_batch(nodes, history, candidate_actions, relations)
    return expanded_nodes, expanded_history, candidate_actions, relations, mask


@torch.no_grad()
def _predict_and_score(snapshot: Any, history: torch.Tensor, base_logits: torch.Tensor, transparent_scores: torch.Tensor, world_model: W1GraphJEPA, adapter: TransparentResidualPrior, *, continuation_id: str, active_continuation_ids: tuple[str, ...], public_payload: Mapping[str, Any]) -> W1CandidateTrace:
    expanded_nodes, expanded_history, actions, relations, mask = graph5_candidate_inputs(snapshot, history, device=base_logits.device)
    if base_logits.shape not in {(1, ACTION_COUNT), (ACTION_COUNT,)}:
        raise ValueError("base_logits must be [25] or [1,25]")
    base = base_logits.reshape(1, ACTION_COUNT)
    transparent = transparent_scores.reshape(1, ACTION_COUNT)
    predicted = world_model.predict_candidates(expanded_nodes, expanded_history, actions, relations)
    latent = predicted["latent"].reshape(1, actions.shape[0], -1)
    dense_latent = torch.zeros((1, ACTION_COUNT, latent.shape[-1]), dtype=latent.dtype, device=latent.device)
    dense_latent[:, actions] = latent
    outcomes = predicted["outcome"].reshape(1, actions.shape[0], -1)
    dense_outcomes = torch.zeros((1, ACTION_COUNT, outcomes.shape[-1]), dtype=outcomes.dtype, device=outcomes.device)
    dense_outcomes[:, actions] = outcomes
    result = adapter(base, transparent, dense_outcomes, mask.unsqueeze(0))
    selected = int(torch.argmax(result["final_logits"], dim=-1).item())
    if not bool(mask[selected]):
        raise RuntimeError("adapter selected an illegal action")
    if not torch.isfinite(result["base_logits"]).all() or not torch.isfinite(result["prior"]).all():
        raise FloatingPointError("non-finite W1 candidate trace")
    input_hash = _digest_public(public_payload)
    return W1CandidateTrace(
        input_hash=input_hash,
        continuation_id=str(continuation_id),
        active_continuation_ids=tuple(active_continuation_ids),
        legal_actions=tuple(int(item) for item in torch.nonzero(mask, as_tuple=False).flatten().tolist()),
        candidate_actions=tuple(int(item) for item in actions.tolist()),
        selected_action=selected,
        base_logits=tuple(float(item) for item in result["base_logits"].flatten().tolist()),
        transparent_scores=tuple(float(item) for item in result["transparent"].flatten().tolist()),
        learned_residuals=tuple(float(item) for item in result["residual"].flatten().tolist()),
        prior_logits=tuple(float(item) for item in result["prior"].flatten().tolist()),
        final_logits=tuple(float(item) if torch.isfinite(item) else float("-inf") for item in result["final_logits"].flatten()),
    )


@dataclass(frozen=True)
class PublicHistoryState:
    values: tuple[float, ...]
    source_input_hashes: tuple[str, ...]
    state_hash: str

    @classmethod
    def create(cls, values: torch.Tensor, source_input_hashes: tuple[str, ...]) -> "PublicHistoryState":
        tensor = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
        if tensor.shape != (128,) or not torch.isfinite(tensor).all():
            raise ValueError("public history state must contain 128 finite values")
        if any(not isinstance(item, str) or len(item) != 64 for item in source_input_hashes):
            raise ValueError("public history sources must be SHA-256 identities")
        payload = {"values": tensor.tolist(), "source_input_hashes": list(source_input_hashes)}
        return cls(tuple(float(item) for item in tensor.tolist()), tuple(source_input_hashes), public_hash(payload))

    def tensor(self, *, device: torch.device | str) -> torch.Tensor:
        payload = {"values": list(self.values), "source_input_hashes": list(self.source_input_hashes)}
        if public_hash(payload) != self.state_hash:
            raise ValueError("public history identity changed")
        return torch.tensor(self.values, dtype=torch.float32, device=device)


@torch.no_grad()
def predict_public_and_score(
    public_input: W1PublicInput,
    base_logits: torch.Tensor,
    transparent_scores: torch.Tensor,
    world_model: W1GraphJEPA,
    adapter: TransparentResidualPrior,
    history_state: PublicHistoryState,
) -> W1CandidateTrace:
    """Strict production entry requiring the frozen public adapter output."""
    if not isinstance(public_input, W1PublicInput):
        raise TypeError("production scoring requires W1PublicInput")
    public_input.validate_production()
    if not isinstance(history_state, PublicHistoryState):
        raise TypeError("production scoring requires PublicHistoryState")
    if not hasattr(public_input.graph, "nodes"):
        raise TypeError("production graph builder must return a Graph5Snapshot")
    payload = {
        "public": public_input.public.frozen_payload(),
        "graph_hash": public_input.graph_hash,
    }
    history = history_state.tensor(device=base_logits.device)
    expected_history_source = public_hash(list(public_input.public.history))
    if not history_state.source_input_hashes or history_state.source_input_hashes[-1] != expected_history_source:
        raise ValueError("public history state is not bound to the frozen history")
    payload["history_state_hash"] = history_state.state_hash
    trace = _predict_and_score(
        public_input.graph, history, base_logits, transparent_scores, world_model, adapter,
        continuation_id=public_input.continuation_policy_id,
        active_continuation_ids=public_input.public.continuation_ids,
        public_payload=payload,
    )
    expected = public_hash(payload)
    if trace.input_hash != expected:
        raise RuntimeError("candidate trace input identity mismatch")
    return trace


__all__ = ["W1CandidateTrace", "PublicHistoryState", "graph5_candidate_inputs", "predict_public_and_score"]
