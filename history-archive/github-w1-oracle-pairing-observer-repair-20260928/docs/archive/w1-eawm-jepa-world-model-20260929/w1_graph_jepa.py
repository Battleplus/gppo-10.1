"""Minimal W1 Graph-JEPA adapter with 25-action candidate batching.

The implementation is a W1-specific thin layer: EMA and stop-gradient follow
the reviewed JEPA code, while node counts and action encoding follow Graph-5.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F


ACTION_COUNT = 25
NODE_TYPES = ("uav", "region", "target", "task", "event")
NODE_COUNTS = {"uav": 4, "region": 3, "target": 4, "task": 6, "event": 4}


@dataclass(frozen=True)
class W1JEPAConfig:
    node_dim: int = 32
    relation_dim: int = 4
    history_dim: int = 128
    hidden_dim: int = 128
    latent_dim: int = 64
    action_count: int = ACTION_COUNT
    outcome_dim: int = 6
    ema: float = 0.99

    def validate(self) -> None:
        if self.action_count != ACTION_COUNT:
            raise ValueError("W1 action contract is 24 candidates plus NOOP")
        if not 0.0 <= self.ema < 1.0:
            raise ValueError("EMA must lie in [0,1)")


class _GraphEncoder(nn.Module):
    def __init__(self, config: W1JEPAConfig):
        super().__init__()
        self.node = nn.ModuleDict({name: nn.Sequential(nn.Linear(config.node_dim, config.hidden_dim), nn.LayerNorm(config.hidden_dim), nn.SiLU()) for name in NODE_TYPES})
        self.pool = nn.Sequential(nn.Linear(config.hidden_dim * len(NODE_TYPES), config.latent_dim), nn.LayerNorm(config.latent_dim), nn.SiLU())

    def forward(self, nodes: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for name in NODE_TYPES:
            value = nodes[name]
            if value.ndim != 3 or value.shape[1] != NODE_COUNTS[name] or value.shape[-1] != self.node[name][0].in_features:
                raise ValueError(f"nodes[{name}] must be [B,{NODE_COUNTS[name]},{self.node[name][0].in_features}]")
            parts.append(self.node[name](value).mean(dim=1))
        return self.pool(torch.cat(parts, dim=-1))


class W1GraphJEPA(nn.Module):
    format_version = "w1-graph-jepa/1.0.0"

    def __init__(self, config: W1JEPAConfig = W1JEPAConfig(), *, event_count: int = 5):
        super().__init__()
        config.validate()
        self.config = config
        self.event_count = int(event_count)
        self.online = _GraphEncoder(config)
        self.target_encoder = deepcopy(self.online).requires_grad_(False).eval()
        self.history = nn.GRUCell(config.history_dim, config.history_dim)
        self.action_embedding = nn.Embedding(config.action_count, 32)
        self.relation_embedding = nn.Sequential(nn.Linear(config.relation_dim, 32), nn.Tanh())
        self.predictor = nn.Sequential(
            nn.Linear(config.latent_dim + config.history_dim + 32 + 32, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self.public_decoder = nn.Linear(config.latent_dim, config.history_dim)
        self.outcome_head = nn.Linear(config.latent_dim, config.outcome_dim)
        self.event_head = nn.Linear(config.latent_dim, event_count)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    def encode(self, nodes: dict[str, torch.Tensor], history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        graph_latent = self.online(nodes)
        if history.ndim != 2 or history.shape[-1] != self.config.history_dim:
            raise ValueError("history must be [B,history_dim]")
        return graph_latent, self.history(history, history)

    def forward(self, nodes: dict[str, torch.Tensor], history: torch.Tensor, actions: torch.Tensor, relations: torch.Tensor) -> dict[str, torch.Tensor]:
        if actions.ndim != 1 or relations.shape != (actions.shape[0], self.config.relation_dim):
            raise ValueError("candidate actions and relations must be aligned [N]")
        graph_latent, history_latent = self.encode(nodes, history)
        if graph_latent.shape[0] != actions.shape[0]:
            raise ValueError("nodes/history must already be expanded to candidate batch")
        action = self.action_embedding(actions.long())
        relation = self.relation_embedding(relations)
        predicted = self.predictor(torch.cat((graph_latent, history_latent, action, relation), dim=-1))
        return {
            "latent": predicted,
            "public_state": self.public_decoder(predicted),
            "outcome": self.outcome_head(predicted),
            "event_logits": self.event_head(predicted),
            "history_latent": history_latent,
        }

    @torch.no_grad()
    def target(self, nodes: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.target_encoder(nodes)

    @torch.no_grad()
    def update_target(self) -> None:
        for target, online in zip(self.target_encoder.parameters(), self.online.parameters()):
            target.lerp_(online, 1.0 - self.config.ema)
        for target, online in zip(self.target_encoder.buffers(), self.online.buffers()):
            target.copy_(online)

    def predict_candidates(self, nodes: dict[str, torch.Tensor], history: torch.Tensor, candidate_actions: torch.Tensor, relations: torch.Tensor) -> dict[str, torch.Tensor]:
        """Batch all legal candidates; caller restores rows by action identity."""
        if candidate_actions.numel() and (candidate_actions.min() < 0 or candidate_actions.max() >= ACTION_COUNT):
            raise ValueError("candidate action outside W1 contract")
        return self.forward(nodes, history, candidate_actions, relations)


def expand_candidate_batch(nodes: dict[str, torch.Tensor], history: torch.Tensor, actions: torch.Tensor, relations: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Expand one frozen window into one row per candidate for one forward."""
    if history.ndim != 2 or history.shape[0] != 1:
        raise ValueError("window history must be [1,history_dim]")
    if actions.ndim != 1 or relations.shape != (actions.shape[0], 4):
        raise ValueError("candidate actions and relation rows must align")
    if actions.numel() == 0:
        raise ValueError("candidate batch cannot be empty")
    return {name: value.expand(actions.shape[0], -1, -1).contiguous() for name, value in nodes.items()}, history.expand(actions.shape[0], -1).contiguous()


__all__ = ["ACTION_COUNT", "NODE_COUNTS", "W1JEPAConfig", "W1GraphJEPA", "expand_candidate_batch"]
