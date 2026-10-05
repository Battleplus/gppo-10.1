"""Training and evaluation utilities for the T-02 world-model gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import COST_NAMES, TensorTransition, apply_predicted_delta, state_vector
from .model import FlatGRUWorldModel, GraphWorldModel


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20260902
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    state_weight: float = 5.0
    state_nll_weight: float = 0.05
    change_weight: float = 0.25
    reward_weight: float = 0.5
    cost_weight: float = 0.25
    continuation_weight: float = 0.5
    kl_weight: float = 1e-3
    patience: int = 15


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _gaussian_nll(error: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.exp(-logvar) * error.square() + logvar).mean()


def episode_loss(model: nn.Module, episode: list[TensorTransition], config: TrainingConfig):
    hidden = None
    parts = {"state": [], "state_nll": [], "change": [], "reward": [], "cost": [], "continuation": [], "kl": []}
    for transition in episode:
        outputs, hidden = model.step(
            transition.graph,
            transition.action,
            hidden,
        )
        state_error = transition.target_delta - outputs["state_delta_soft"]
        parts["state"].append(
            F.smooth_l1_loss(outputs["state_delta_soft"], transition.target_delta, beta=0.05)
        )
        parts["state_nll"].append(_gaussian_nll(state_error, outputs["state_logvar"]))
        changed = (transition.target_delta.abs() > 1e-6).to(torch.float32)
        parts["change"].append(
            F.binary_cross_entropy_with_logits(
                outputs["state_change_logit"],
                changed,
                pos_weight=torch.full_like(changed, 4.0),
            )
        )
        reward = torch.tensor(transition.reward, dtype=torch.float32)
        parts["reward"].append(_gaussian_nll(reward - outputs["reward"], outputs["reward_logvar"]))
        parts["cost"].append(
            _gaussian_nll(transition.costs - outputs["costs"], outputs["cost_logvar"])
        )
        continuation = torch.tensor(transition.continuation, dtype=torch.float32)
        parts["continuation"].append(F.binary_cross_entropy_with_logits(outputs["continuation_logit"], continuation))
        if isinstance(model, GraphWorldModel):
            mean, logvar = outputs["z_mean"], outputs["z_logvar"]
            parts["kl"].append(-0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean())
        else:
            parts["kl"].append(torch.zeros((), dtype=torch.float32))
    means = {name: torch.stack(values).mean() for name, values in parts.items()}
    total = (
        config.state_weight * means["state"]
        + config.state_nll_weight * means["state_nll"]
        + config.change_weight * means["change"]
        + config.reward_weight * means["reward"]
        + config.cost_weight * means["cost"]
        + config.continuation_weight * means["continuation"]
        + config.kl_weight * means["kl"]
    )
    return total, means


def train_model(
    model: nn.Module,
    train_episodes: list[list[TensorTransition]],
    validation_episodes: list[list[TensorTransition]],
    config: TrainingConfig,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    seed_everything(config.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = None
    best_score = float("inf")
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    rng = random.Random(config.seed)
    for epoch in range(config.epochs):
        model.train()
        indices = list(range(len(train_episodes)))
        rng.shuffle(indices)
        train_losses = []
        for index in indices:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = episode_loss(model, train_episodes[index], config)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        validation = evaluate_model(model, validation_episodes)
        score = validation["state_mae"] + 0.05 * validation["reward_mae"] + 0.01 * validation["cost_mae"]
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                "validation_score": score,
                **validation,
            }
        )
        if score < best_score - 1e-8:
            best_score = score
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint candidate")
    model.load_state_dict(best_state)
    return model, history, {"best_epoch": best_epoch, "best_validation_score": best_score}


@torch.no_grad()
def calibrate_change_threshold(
    model: nn.Module,
    validation_episodes: list[list[TensorTransition]],
) -> dict[str, Any]:
    """Choose the sparse-delta threshold on validation data only."""

    model.eval()
    raw_deltas: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for episode in validation_episodes:
        hidden = None
        for transition in episode:
            outputs, hidden = model.step(transition.graph, transition.action, hidden, sample=False)
            raw_deltas.append(outputs["state_delta_raw"])
            probabilities.append(outputs["state_change_probability"])
            targets.append(transition.target_delta)
    raw = torch.stack(raw_deltas)
    probability = torch.stack(probabilities)
    target = torch.stack(targets)
    candidates = [round(value, 2) for value in np.linspace(0.05, 0.95, 91)]
    scores = {
        f"{threshold:.2f}": float(
            (target - raw * (probability >= threshold).to(raw.dtype)).abs().mean()
        )
        for threshold in candidates
    }
    selected = min(candidates, key=lambda threshold: (scores[f"{threshold:.2f}"], threshold))
    model.change_threshold.fill_(selected)
    return {
        "selection_split": "validation",
        "selected_threshold": selected,
        "selected_state_mae": scores[f"{selected:.2f}"],
        "candidate_count": len(candidates),
        "curve": scores,
    }


@torch.no_grad()
def prediction_errors(
    model: nn.Module,
    episodes: list[list[TensorTransition]],
) -> dict[str, list[float]]:
    model.eval()
    result = {"state": [], "reward": [], "cost": [], "continuation_brier": [], "continuation_correct": []}
    for episode in episodes:
        hidden = None
        for transition in episode:
            outputs, hidden = model.step(transition.graph, transition.action, hidden, sample=False)
            result["state"].append(float((transition.target_delta - outputs["state_delta"]).abs().mean()))
            result["reward"].append(float(abs(transition.reward - float(outputs["reward"]))))
            result["cost"].append(float((transition.costs - outputs["costs"]).abs().mean()))
            probability = float(torch.sigmoid(outputs["continuation_logit"]))
            result["continuation_brier"].append((probability - transition.continuation) ** 2)
            result["continuation_correct"].append(float((probability >= 0.5) == bool(transition.continuation)))
    return result


@torch.no_grad()
def evaluate_model(model: nn.Module, episodes: list[list[TensorTransition]]) -> dict[str, float]:
    errors = prediction_errors(model, episodes)
    return {
        "state_mae": float(np.mean(errors["state"])),
        "reward_mae": float(np.mean(errors["reward"])),
        "cost_mae": float(np.mean(errors["cost"])),
        "continuation_brier": float(np.mean(errors["continuation_brier"])),
        "continuation_accuracy": float(np.mean(errors["continuation_correct"])),
        "transition_count": len(errors["state"]),
    }


def baseline_metrics(
    train_episodes: list[list[TensorTransition]],
    test_episodes: list[list[TensorTransition]],
) -> dict[str, float]:
    train = [transition for episode in train_episodes for transition in episode]
    test = [transition for episode in test_episodes for transition in episode]
    reward_mean = float(np.mean([item.reward for item in train]))
    cost_mean = torch.stack([item.costs for item in train]).mean(dim=0)
    continuation_mean = float(np.mean([item.continuation for item in train]))
    return {
        "state_mae": float(np.mean([float(item.target_delta.abs().mean()) for item in test])),
        "reward_mae": float(np.mean([abs(item.reward - reward_mean) for item in test])),
        "cost_mae": float(np.mean([float((item.costs - cost_mean).abs().mean()) for item in test])),
        "continuation_brier": float(np.mean([(item.continuation - continuation_mean) ** 2 for item in test])),
        "continuation_accuracy": float(np.mean([(continuation_mean >= 0.5) == bool(item.continuation) for item in test])),
        "transition_count": len(test),
    }


def _paired_action_ablation(
    model: nn.Module,
    episodes: list[list[TensorTransition]],
    *,
    mode: str,
    seed: int = 20260902,
    samples: int = 5000,
) -> dict[str, float]:
    if mode not in {"legal_counterfactual", "no_action"}:
        raise ValueError(f"unsupported action ablation {mode!r}")
    model.eval()
    selector = random.Random(seed)
    metric_names = ("state", "reward", "cost")
    correct_errors = {name: [] for name in metric_names}
    ablated_errors = {name: [] for name in metric_names}
    episode_differences = {name: [] for name in metric_names}
    alternatives: list[int] = []
    skipped = 0
    illegal_alternative_count = 0
    with torch.no_grad():
        for episode in episodes:
            hidden = None
            local_differences = {name: [] for name in metric_names}
            for transition in episode:
                correct_output, next_hidden = model.step(
                    transition.graph, transition.action, hidden, sample=False
                )
                if mode == "no_action":
                    alternative: int | None = None
                else:
                    legal = [
                        int(index)
                        for index in torch.nonzero(transition.graph.action_mask, as_tuple=False).flatten().tolist()
                        if int(index) != transition.action
                    ]
                    if not legal:
                        hidden = next_hidden
                        skipped += 1
                        continue
                    alternative = legal[selector.randrange(len(legal))]
                    alternatives.append(alternative)
                    if not bool(transition.graph.action_mask[alternative].item()):
                        illegal_alternative_count += 1
                ablated_output, _ = model.step(transition.graph, alternative, hidden, sample=False)
                current_correct = {
                    "state": float((transition.target_delta - correct_output["state_delta"]).abs().mean()),
                    "reward": abs(transition.reward - float(correct_output["reward"])),
                    "cost": float((transition.costs - correct_output["costs"]).abs().mean()),
                }
                current_ablated = {
                    "state": float((transition.target_delta - ablated_output["state_delta"]).abs().mean()),
                    "reward": abs(transition.reward - float(ablated_output["reward"])),
                    "cost": float((transition.costs - ablated_output["costs"]).abs().mean()),
                }
                for name in metric_names:
                    correct_errors[name].append(current_correct[name])
                    ablated_errors[name].append(current_ablated[name])
                    local_differences[name].append(current_ablated[name] - current_correct[name])
                hidden = next_hidden
            if local_differences["state"]:
                for name in metric_names:
                    episode_differences[name].append(float(np.mean(local_differences[name])))
    if not episode_differences["state"]:
        raise RuntimeError(f"no evaluable episodes for {mode}")
    rng = np.random.default_rng(seed ^ (0x4E4F4143 if mode == "no_action" else 0x4C454741))
    bootstrap_indices = rng.integers(
        0, len(episode_differences["state"]), size=(samples, len(episode_differences["state"]))
    )
    metrics = {}
    for name in metric_names:
        differences = np.asarray(episode_differences[name])
        bootstrap = differences[bootstrap_indices].mean(axis=1)
        metrics[name] = {
            "correct_mae": float(np.mean(correct_errors[name])),
            "ablated_mae": float(np.mean(ablated_errors[name])),
            "mean_degradation": float(differences.mean()),
            "relative_degradation": float(
                differences.mean() / max(np.mean(correct_errors[name]), 1e-12)
            ),
            "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
            "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
        }
    result = {
        "mode": mode,
        "metrics": metrics,
        "bootstrap_samples": samples,
        "bootstrap_unit": "episode",
        "evaluated_transitions": len(correct_errors["state"]),
        "evaluated_episodes": len(episode_differences["state"]),
        "skipped_without_legal_alternative": skipped,
        "illegal_alternative_count": illegal_alternative_count,
        "unique_legal_alternatives": sorted(set(alternatives)),
    }
    result.update(
        {
            "correct_state_mae": metrics["state"]["correct_mae"],
            "ablated_state_mae": metrics["state"]["ablated_mae"],
            "mean_degradation": metrics["state"]["mean_degradation"],
            "relative_degradation": metrics["state"]["relative_degradation"],
            "bootstrap_ci95_low": metrics["state"]["bootstrap_ci95_low"],
            "bootstrap_ci95_high": metrics["state"]["bootstrap_ci95_high"],
        }
    )
    return result


def paired_bootstrap_action_degradation(
    model: nn.Module,
    episodes: list[list[TensorTransition]],
    *,
    seed: int = 20260902,
    samples: int = 5000,
) -> dict[str, float]:
    return _paired_action_ablation(
        model, episodes, mode="legal_counterfactual", seed=seed, samples=samples
    )


def paired_bootstrap_no_action_degradation(
    model: nn.Module,
    episodes: list[list[TensorTransition]],
    *,
    seed: int = 20260902,
    samples: int = 5000,
) -> dict[str, float]:
    return _paired_action_ablation(model, episodes, mode="no_action", seed=seed, samples=samples)


@torch.no_grad()
def uncertainty_diagnostics(
    model: nn.Module,
    episodes: list[list[TensorTransition]],
) -> dict[str, Any]:
    model.eval()
    errors: list[float] = []
    uncertainties: list[float] = []
    for episode in episodes:
        hidden = None
        for transition in episode:
            outputs, hidden = model.step(
                transition.graph, transition.action, hidden, sample=False
            )
            errors.append(float((transition.target_delta - outputs["state_delta"]).abs().mean()))
            uncertainties.append(float(outputs["state_logvar"].exp().mean()))
    error_array = np.asarray(errors)
    uncertainty_array = np.asarray(uncertainties)
    order = np.argsort(uncertainty_array, kind="stable")
    bins = np.array_split(order, 4)
    quartile_mae = [float(error_array[index].mean()) for index in bins]
    error_rank = np.argsort(np.argsort(error_array, kind="stable"), kind="stable")
    uncertainty_rank = np.argsort(np.argsort(uncertainty_array, kind="stable"), kind="stable")
    spearman = float(np.corrcoef(error_rank, uncertainty_rank)[0, 1]) if len(errors) > 1 else 0.0
    return {
        "spearman_rank_correlation": spearman,
        "uncertainty_quartile_state_mae": quartile_mae,
        "high_vs_low_risk_ratio": quartile_mae[-1] / max(quartile_mae[0], 1e-12),
        "transition_count": len(errors),
    }


@torch.no_grad()
def rollout_errors(
    model: nn.Module,
    episodes: list[list[TensorTransition]],
    horizons: Iterable[int] = (1, 3, 5),
) -> dict[str, dict[str, float]]:
    model.eval()
    result = {}
    for horizon in horizons:
        errors = []
        for episode in episodes:
            for start in range(0, len(episode) - horizon + 1):
                graph = episode[start].graph
                hidden = None
                for offset in range(horizon):
                    transition = episode[start + offset]
                    outputs, hidden = model.step(graph, transition.action, hidden, sample=False)
                    graph = apply_predicted_delta(graph, outputs["state_delta"])
                truth = episode[start + horizon - 1].next_graph
                errors.append(float((state_vector(graph) - state_vector(truth)).abs().mean()))
        result[str(horizon)] = {
            "state_mae": float(np.mean(errors)) if errors else float("nan"),
            "windows": len(errors),
        }
    return result


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def config_dict(config: TrainingConfig) -> dict[str, Any]:
    return asdict(config)
