"""Independent one-step Graph-JEPA. No policy, event preference, or Shadow writes."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .data import STATE_DIM
from .model import HeteroGraphEncoder, WorldModelConfig, RELATIONS, _relation_key
from .registry import FEATURE_REGISTRY, SCHEMA_VERSION


def batch_graphs(graphs):
    """Stack fixed-size snapshots, preserving each sample's own edge indices."""
    return {
        "nodes": {k: torch.stack([g.nodes[k] for g in graphs]) for k in graphs[0].nodes},
        "edge_index": {k: torch.stack([g.edge_index[k] for g in graphs]) for k in RELATIONS},
        "edge_attr": {k: torch.stack([g.edge_attr[k] for g in graphs]) for k in RELATIONS},
    }


def select_batch(graph, indices):
    return {part: {key: value[indices] for key, value in values.items()} for part, values in graph.items()}


def encode_batch(encoder, graph):
    """Vectorized equivalent of HeteroGraphEncoder; no cross-sample edges/pooling."""
    hidden = {k: encoder.encoders[k](v) for k, v in graph["nodes"].items()}
    for messages, updates in zip(encoder.messages, encoder.updates):
        totals = {k: torch.zeros_like(v) for k, v in hidden.items()}
        counts = {k: v.new_zeros((*v.shape[:2], 1)) for k, v in hidden.items()}
        for relation in RELATIONS:
            source, _, destination = relation
            index = graph["edge_index"][relation]
            src, dst = index[:, 0], index[:, 1]
            features = hidden[source].gather(1, src.unsqueeze(-1).expand(-1, -1, hidden[source].shape[-1]))
            message = messages[_relation_key(relation)](torch.cat([features, graph["edge_attr"][relation]], -1))
            totals[destination].scatter_add_(1, dst.unsqueeze(-1).expand_as(message), message)
            counts[destination].scatter_add_(1, dst.unsqueeze(-1), message.new_ones((*dst.shape, 1)))
        hidden = {k: updates[k](torch.cat([v, totals[k] / counts[k].clamp_min(1)], -1)) for k, v in hidden.items()}
    return encoder.pool(torch.cat([hidden[k].mean(1) for k in ("uav", "region", "target")], -1))


@dataclass(frozen=True)
class JEPAConfig:
    hidden_dim: int = 64
    latent_dim: int = 32
    action_dim: int = 18
    evidence_dim: int = 12
    ema: float = 0.99


class GraphJEPA(nn.Module):
    """Future graphs are accepted only by target(), never forward()."""
    FORMAT = "gppo-graph-jepa-exploratory-v1"

    def __init__(self, config=JEPAConfig(), group="action_jepa"):
        super().__init__()
        if group not in {"action_jepa", "no_action_jepa", "supervised_graph"}:
            raise ValueError("unknown experimental group")
        if not 0 <= config.ema < 1:
            raise ValueError("invalid EMA")
        self.config, self.group = config, group
        self.online = HeteroGraphEncoder(WorldModelConfig(hidden_dim=config.hidden_dim, graph_dim=config.latent_dim))
        self.predictor = nn.Sequential(
            nn.Linear(config.latent_dim + config.action_dim + config.evidence_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self.target_encoder = deepcopy(self.online).requires_grad_(False).eval()
        self.decoder = nn.Linear(config.latent_dim, STATE_DIM + 9) if group == "supervised_graph" else None

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    def forward(self, graph, actions, evidence):
        current = encode_batch(self.online, graph)
        action = F.one_hot(actions, self.config.action_dim).to(current.dtype)
        if self.group == "no_action_jepa":
            action = torch.zeros_like(action)
        future = self.predictor(torch.cat([current, action, evidence], -1))
        return current, future

    @torch.no_grad()
    def target(self, future_graph):
        return encode_batch(self.target_encoder, future_graph)

    @torch.no_grad()
    def update_target(self):
        for target, online in zip(self.target_encoder.parameters(), self.online.parameters()):
            target.lerp_(online, 1.0 - self.config.ema)
        for target, online in zip(self.target_encoder.buffers(), self.online.buffers()):
            target.copy_(online)

    def save(self, path, metadata):
        torch.save({"format": self.FORMAT, "config": asdict(self.config), "group": self.group,
                    "registry_sha256": FEATURE_REGISTRY.sha256(), "schema_version": SCHEMA_VERSION,
                    "model": self.state_dict(), "metadata": metadata}, path)

    @classmethod
    def load(cls, path):
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record["format"] != cls.FORMAT or record["registry_sha256"] != FEATURE_REGISTRY.sha256() or record["schema_version"] != SCHEMA_VERSION:
            raise ValueError("JEPA checkpoint contract mismatch")
        model = cls(JEPAConfig(**record["config"]), record["group"])
        model.load_state_dict(record["model"], strict=True)
        return model.eval(), record["metadata"]


def representation_loss(current, predicted, target, *, std_target=1.0, variance_weight=1.0, covariance_weight=0.04):
    """Project-specific VICReg-inspired online variance/covariance regularizers."""
    if current.shape[0] < 2:
        raise ValueError("variance/covariance require at least two samples")
    prediction = F.mse_loss(predicted, target.detach())
    variance = F.relu(std_target - (current.var(0, unbiased=False) + 1e-4).sqrt()).mean()
    centered = current - current.mean(0)
    covariance = centered.T @ centered / (current.shape[0] - 1)
    off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
    decorrelation = off_diagonal.square().sum() / current.shape[1]
    total = prediction + variance_weight * variance + covariance_weight * decorrelation
    return total, {"prediction": prediction, "variance": variance, "covariance": decorrelation}
