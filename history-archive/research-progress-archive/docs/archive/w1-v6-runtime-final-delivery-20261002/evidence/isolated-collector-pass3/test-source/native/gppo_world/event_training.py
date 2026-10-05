"""T-03 event-aware training, GES weighting and independent metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import STATE_DIM, TensorTransition
from .events import (
    EVIDENCE_EVENTS,
    MODALITIES,
    NOMINAL_SLOTS,
    ORDINAL_SLOTS,
    STRUCTURAL_SLOT_NAMES,
    EventLabels,
    EventSchema,
    ges_weight,
)
from .model import EventAwareGraphWorldModel
from .training import _gaussian_nll, evaluate_model, seed_everything


@dataclass(frozen=True)
class EventTrainingConfig:
    seed: int = 20260903
    epochs: int = 40
    learning_rate: float = 5e-4
    backbone_learning_rate_multiplier: float = 0.2
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    state_weight: float = 5.0
    state_nll_weight: float = 0.05
    change_weight: float = 0.25
    reward_weight: float = 0.5
    cost_weight: float = 0.25
    continuation_weight: float = 0.5
    kl_weight: float = 1e-3
    # Frozen after the first three-seed validation audit: 0.50 caused a
    # 6% reward-MAE regression on validation for seed 20260905.  The complete
    # pre-calibration run is retained as T-03 failed evidence.
    event_weight: float = 0.25
    event_aware_weight: float = 0.5
    focal_gamma: float = 4.0
    smooth_epsilon: float = 5e-4
    max_base_metric_degradation: float = 0.05
    rare_prevalence_threshold: float = 0.10


VARIANTS = {
    "wm": {"event_enabled": False, "ges_mode": "none"},
    "ea_no_ges": {"event_enabled": True, "ges_mode": "none"},
    "eawm_hard": {"event_enabled": True, "ges_mode": "hard"},
    "eawm_smooth": {"event_enabled": True, "ges_mode": "smooth"},
}


def _modality_targets(labels: EventLabels, modality: str) -> torch.Tensor:
    if modality == "ordinal":
        return (labels.ordinal != 1).to(torch.float32)
    return getattr(labels, modality).to(torch.float32)


def freeze_class_balance(
    labeled_train: list[list[tuple[TensorTransition, EventLabels]]],
) -> dict[str, Any]:
    ordinal_class_counts = torch.zeros(3, dtype=torch.float64)
    slot_positive = {
        "ordinal": torch.zeros(len(ORDINAL_SLOTS), dtype=torch.float64),
        "nominal": torch.zeros(len(NOMINAL_SLOTS), dtype=torch.float64),
        "structural": torch.zeros(len(STRUCTURAL_SLOT_NAMES), dtype=torch.float64),
        "evidence": torch.zeros(len(EVIDENCE_EVENTS), dtype=torch.float64),
    }
    slot_valid = {name: torch.zeros_like(value) for name, value in slot_positive.items()}
    for episode in labeled_train:
        for _, labels in episode:
            ordinal_class_counts += torch.bincount(labels.ordinal, minlength=3).to(torch.float64)
            for name in MODALITIES:
                target = _modality_targets(labels, name)
                valid = labels.valid_mask(name).to(torch.float64)
                slot_positive[name] += target.to(torch.float64) * valid
                slot_valid[name] += valid
    inverse = ordinal_class_counts.sum() / ordinal_class_counts.clamp_min(1.0)
    ordinal_weights = inverse.sqrt()
    ordinal_weights /= ordinal_weights.mean()
    modalities = {}
    for name in MODALITIES:
        prevalence = slot_positive[name] / slot_valid[name].clamp_min(1.0)
        positive = float(slot_positive[name].sum())
        valid = float(slot_valid[name].sum())
        negative = valid - positive
        positive_alpha = min(0.95, max(0.05, negative / max(valid, 1.0)))
        modalities[name] = {
            "positive": int(positive),
            "negative": int(negative),
            "valid": int(valid),
            "prevalence": positive / max(valid, 1.0),
            "positive_alpha": positive_alpha,
            "slot_positive": slot_positive[name].to(torch.int64).tolist(),
            "slot_valid": slot_valid[name].to(torch.int64).tolist(),
            "slot_prevalence": prevalence.tolist(),
        }
    return {
        "source_split": "train",
        "ordinal_class_counts": ordinal_class_counts.to(torch.int64).tolist(),
        "ordinal_class_weights": ordinal_weights.to(torch.float32).tolist(),
        "modalities": modalities,
    }


def _focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    positive_alpha: float,
    gamma: float,
) -> torch.Tensor:
    selected_logits = logits[valid]
    selected_target = target[valid]
    if selected_logits.numel() == 0:
        return torch.zeros((), device=logits.device)
    cross_entropy = F.binary_cross_entropy_with_logits(selected_logits, selected_target, reduction="none")
    probability = torch.sigmoid(selected_logits)
    probability_true = probability * selected_target + (1.0 - probability) * (1.0 - selected_target)
    alpha = positive_alpha * selected_target + (1.0 - positive_alpha) * (1.0 - selected_target)
    return (alpha * (1.0 - probability_true).pow(gamma) * cross_entropy).mean()


def _state_event_weights(
    labels: EventLabels,
    schema: EventSchema,
    ges_mode: str,
    event_aware_weight: float,
    smooth_epsilon: float,
) -> torch.Tensor:
    weights = torch.ones(STATE_DIM, dtype=torch.float32)
    for modality, slots in (("ordinal", ORDINAL_SLOTS), ("nominal", NOMINAL_SLOTS)):
        gate = ges_weight(
            labels.density[modality],
            schema.density_thresholds[modality],
            ges_mode,
            smooth_epsilon,
        )
        # Equation 8 can exceed one for the smooth segmentor; clipping here
        # prevents negative reconstruction weights while preserving its rank.
        effective_gate = min(1.0, gate)
        occurred = labels.event_mask(modality)
        for index, slot in enumerate(slots):
            if not bool(occurred[index]):
                weights[slot.state_index] = max(0.0, 1.0 - event_aware_weight * effective_gate)
    return weights


def event_episode_loss(
    model: EventAwareGraphWorldModel,
    episode: list[tuple[TensorTransition, EventLabels]],
    schema: EventSchema,
    balance: Mapping[str, Any],
    config: EventTrainingConfig,
    *,
    variant: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    settings = VARIANTS[variant]
    hidden = None
    names = (
        "state",
        "state_nll",
        "change",
        "reward",
        "cost",
        "continuation",
        "kl",
        "event_ordinal",
        "event_nominal",
        "event_structural",
        "event_evidence",
    )
    parts: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    ordinal_weights = torch.tensor(balance["ordinal_class_weights"], dtype=torch.float32)
    for transition, labels in episode:
        outputs, hidden = model.step(
            transition.graph, transition.action, hidden, evidence=transition.evidence
        )
        per_state = F.smooth_l1_loss(
            outputs["state_delta_soft"], transition.target_delta, beta=0.05, reduction="none"
        )
        state_weights = _state_event_weights(
            labels,
            schema,
            settings["ges_mode"] if settings["event_enabled"] else "none",
            config.event_aware_weight if settings["event_enabled"] else 0.0,
            config.smooth_epsilon,
        )
        parts["state"].append((per_state * state_weights).sum() / state_weights.sum().clamp_min(1.0))
        state_error = transition.target_delta - outputs["state_delta_soft"]
        parts["state_nll"].append(_gaussian_nll(state_error, outputs["state_logvar"]))
        changed = (transition.target_delta.abs() > 1e-6).to(torch.float32)
        parts["change"].append(
            F.binary_cross_entropy_with_logits(
                outputs["state_change_logit"], changed, pos_weight=torch.full_like(changed, 4.0)
            )
        )
        reward = torch.tensor(transition.reward, dtype=torch.float32)
        parts["reward"].append(_gaussian_nll(reward - outputs["reward"], outputs["reward_logvar"]))
        parts["cost"].append(_gaussian_nll(transition.costs - outputs["costs"], outputs["cost_logvar"]))
        continuation = torch.tensor(transition.continuation, dtype=torch.float32)
        parts["continuation"].append(
            F.binary_cross_entropy_with_logits(outputs["continuation_logit"], continuation)
        )
        mean, logvar = outputs["z_mean"], outputs["z_logvar"]
        parts["kl"].append(-0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean())

        gates = {
            name: ges_weight(
                labels.density[name],
                schema.density_thresholds[name],
                settings["ges_mode"],
                config.smooth_epsilon,
            )
            if settings["event_enabled"]
            else 0.0
            for name in MODALITIES
        }
        parts["event_ordinal"].append(
            gates["ordinal"]
            * F.cross_entropy(outputs["ordinal_event_logits"], labels.ordinal, weight=ordinal_weights)
        )
        for name in ("nominal", "structural", "evidence"):
            parts[f"event_{name}"].append(
                gates[name]
                * _focal_loss(
                    outputs[f"{name}_event_logits"],
                    getattr(labels, name),
                    labels.valid_mask(name),
                    positive_alpha=float(balance["modalities"][name]["positive_alpha"]),
                    gamma=config.focal_gamma,
                )
            )
    means = {name: torch.stack(values).mean() for name, values in parts.items()}
    event_total = sum(means[f"event_{name}"] for name in MODALITIES)
    total = (
        config.state_weight * means["state"]
        + config.state_nll_weight * means["state_nll"]
        + config.change_weight * means["change"]
        + config.reward_weight * means["reward"]
        + config.cost_weight * means["cost"]
        + config.continuation_weight * means["continuation"]
        + config.kl_weight * means["kl"]
        + config.event_weight * event_total
    )
    means["event_total"] = event_total
    return total, means


def _average_precision(target: np.ndarray, score: np.ndarray) -> float:
    positives = int(target.sum())
    if positives == 0:
        return float("nan")
    # Integrate at distinct score thresholds.  Treating tied scores as a
    # single group makes a constant frequency baseline equal prevalence and
    # avoids accidental dependence on dataset row order.
    order = np.argsort(-score, kind="stable")
    ranked_target = target[order]
    ranked_score = score[order]
    boundaries = np.flatnonzero(np.r_[ranked_score[1:] != ranked_score[:-1], True])
    cumulative_true = np.cumsum(ranked_target)
    true_at_boundary = cumulative_true[boundaries]
    predicted_at_boundary = boundaries + 1
    precision_at_boundary = true_at_boundary / predicted_at_boundary
    previous_true = np.r_[0, true_at_boundary[:-1]]
    recall_increment = (true_at_boundary - previous_true) / positives
    return float(np.sum(precision_at_boundary * recall_increment))


def _binary_metrics(target: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    prediction = score >= threshold
    truth = target.astype(bool)
    true_positive = int((prediction & truth).sum())
    false_positive = int((prediction & ~truth).sum())
    false_negative = int((~prediction & truth).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "f1": f1,
        "auprc": _average_precision(target, score),
        "precision": precision,
        "recall": recall,
        "positive_support": int(truth.sum()),
        "negative_support": int((~truth).sum()),
    }


@torch.no_grad()
def evaluate_events(
    model: EventAwareGraphWorldModel,
    labeled_episodes: list[list[tuple[TensorTransition, EventLabels]]],
    balance: Mapping[str, Any],
    *,
    rare_threshold: float = 0.10,
) -> dict[str, Any]:
    model.eval()
    targets = {name: [] for name in MODALITIES}
    scores = {name: [] for name in MODALITIES}
    valid_masks = {name: [] for name in MODALITIES}
    slot_targets = {name: [] for name in MODALITIES}
    slot_scores = {name: [] for name in MODALITIES}
    ordinal_class_truth = []
    ordinal_class_prediction = []
    for episode in labeled_episodes:
        hidden = None
        for transition, labels in episode:
            outputs, hidden = model.step(
                transition.graph,
                transition.action,
                hidden,
                sample=False,
                evidence=transition.evidence,
            )
            ordinal_probability = torch.softmax(outputs["ordinal_event_logits"], dim=-1)
            current_scores = {
                "ordinal": 1.0 - ordinal_probability[:, 1],
                "nominal": torch.sigmoid(outputs["nominal_event_logits"]),
                "structural": torch.sigmoid(outputs["structural_event_logits"]),
                "evidence": torch.sigmoid(outputs["evidence_event_logits"]),
            }
            ordinal_class_truth.extend(labels.ordinal.tolist())
            ordinal_class_prediction.extend(ordinal_probability.argmax(dim=-1).tolist())
            for name in MODALITIES:
                target = _modality_targets(labels, name)
                valid = labels.valid_mask(name)
                targets[name].extend(target[valid].tolist())
                scores[name].extend(current_scores[name][valid].tolist())
                valid_masks[name].append(valid)
                slot_targets[name].append(target)
                slot_scores[name].append(current_scores[name])
    per_modality = {}
    baseline = {}
    rare = {}
    per_slot = {}
    slot_names = {
        "ordinal": [slot.path for slot in ORDINAL_SLOTS],
        "nominal": [slot.path for slot in NOMINAL_SLOTS],
        "structural": list(STRUCTURAL_SLOT_NAMES),
        "evidence": list(EVIDENCE_EVENTS),
    }
    for name in MODALITIES:
        target = np.asarray(targets[name], dtype=np.int64)
        score = np.asarray(scores[name], dtype=np.float64)
        per_modality[name] = _binary_metrics(target, score)
        train_prevalence = float(balance["modalities"][name]["prevalence"])
        baseline_prediction = np.full(len(target), 1.0 if train_prevalence >= 0.5 else 0.0)
        baseline[name] = _binary_metrics(target, baseline_prediction)

        prevalence = np.asarray(balance["modalities"][name]["slot_prevalence"])
        supported = np.asarray(balance["modalities"][name]["slot_positive"]) > 0
        rare_slots = supported & (prevalence <= rare_threshold)
        if slot_targets[name]:
            target_matrix = torch.stack(slot_targets[name]).numpy().astype(bool)
            score_matrix = torch.stack(slot_scores[name]).numpy()
            valid_matrix = torch.stack(valid_masks[name]).numpy().astype(bool)
            per_slot[name] = {}
            for index, slot_name in enumerate(slot_names[name]):
                selected_slot = valid_matrix[:, index]
                slot_target = target_matrix[selected_slot, index].astype(np.int64)
                slot_score = score_matrix[selected_slot, index]
                train_slot_prevalence = float(balance["modalities"][name]["slot_prevalence"][index])
                baseline_slot_score = np.full(len(slot_target), train_slot_prevalence)
                per_slot[name][slot_name] = {
                    "model": _binary_metrics(slot_target, slot_score),
                    "frequency_baseline": _binary_metrics(slot_target, baseline_slot_score),
                    "train_prevalence": train_slot_prevalence,
                    "valid_support": int(len(slot_target)),
                }
            selected = valid_matrix & rare_slots.reshape(1, -1)
            rare_target = target_matrix[selected].astype(np.int64)
            rare_score = score_matrix[selected]
            rare[name] = {
                **(_binary_metrics(rare_target, rare_score) if len(rare_target) else {
                    "f1": float("nan"), "auprc": float("nan"), "precision": float("nan"),
                    "recall": float("nan"), "positive_support": 0, "negative_support": 0,
                }),
                "slot_count": int(rare_slots.sum()),
            }
        else:
            per_slot[name] = {}
            rare[name] = {"slot_count": 0, "positive_support": 0, "recall": float("nan")}
    supported_metrics = [value for value in per_modality.values() if value["positive_support"] > 0]
    supported_baselines = [baseline[name] for name in MODALITIES if per_modality[name]["positive_support"] > 0]
    rare_recalls = [
        value["recall"] for value in rare.values()
        if value.get("positive_support", 0) > 0 and np.isfinite(value.get("recall", float("nan")))
    ]
    baseline_rare_recalls = []
    for name in MODALITIES:
        if rare[name].get("positive_support", 0) > 0:
            baseline_rare_recalls.append(
                1.0 if float(balance["modalities"][name]["prevalence"]) >= 0.5 else 0.0
            )
    class_f1 = []
    ordinal_truth = np.asarray(ordinal_class_truth)
    ordinal_prediction = np.asarray(ordinal_class_prediction)
    for event_class in range(3):
        truth = ordinal_truth == event_class
        prediction = ordinal_prediction == event_class
        tp = int((truth & prediction).sum())
        fp = int((~truth & prediction).sum())
        fn = int((truth & ~prediction).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        class_f1.append(2.0 * precision * recall / max(precision + recall, 1e-12))
    return {
        "per_modality": per_modality,
        "per_slot": per_slot,
        "frequency_baseline": baseline,
        "macro_f1": float(np.mean([value["f1"] for value in supported_metrics])),
        "macro_auprc": float(np.mean([value["auprc"] for value in supported_metrics])),
        "baseline_macro_f1": float(np.mean([value["f1"] for value in supported_baselines])),
        "baseline_macro_auprc": float(np.mean([value["auprc"] for value in supported_baselines])),
        "rare": rare,
        "rare_recall": float(np.mean(rare_recalls)) if rare_recalls else float("nan"),
        "baseline_rare_recall": float(np.mean(baseline_rare_recalls)) if baseline_rare_recalls else float("nan"),
        "ordinal_direction_macro_f1": float(np.mean(class_f1)),
    }


def ges_diagnostics(
    labeled_episodes: list[list[tuple[TensorTransition, EventLabels]]],
    schema: EventSchema,
    mode: str,
    *,
    epsilon: float = 5e-4,
) -> dict[str, Any]:
    result = {}
    for name in MODALITIES:
        labels_for_name = [labels for episode in labeled_episodes for _, labels in episode]
        densities = [labels.density[name] for labels in labels_for_name]
        weights = [ges_weight(value, schema.density_thresholds[name], mode, epsilon) for value in densities]
        active_count = sum(
            int((labels.event_mask(name) & labels.valid_mask(name)).sum())
            for labels in labels_for_name
        )
        eligible_count = sum(int(labels.valid_mask(name).sum()) for labels in labels_for_name)
        result[name] = {
            "density_threshold": schema.density_thresholds[name],
            "mean_density": float(np.mean(densities)),
            "max_density": float(np.max(densities)),
            "mean_weight": float(np.mean(weights)),
            "min_weight": float(np.min(weights)),
            "max_weight": float(np.max(weights)),
            "effective_weight_sum": float(np.sum(weights)),
            "gated_transition_count": int(sum(value == 0.0 for value in weights)),
            "retained_transition_count": int(sum(value > 0.0 for value in weights)),
            "active_count": active_count,
            "eligible_count": eligible_count,
            "transition_count": len(weights),
        }
    return {"mode": mode, "modalities": result}


def train_event_model(
    model: EventAwareGraphWorldModel,
    train_episodes: list[list[tuple[TensorTransition, EventLabels]]],
    validation_episodes: list[list[tuple[TensorTransition, EventLabels]]],
    schema: EventSchema,
    balance: Mapping[str, Any],
    config: EventTrainingConfig,
    *,
    variant: str,
) -> tuple[EventAwareGraphWorldModel, list[dict[str, Any]], dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown T-03 variant {variant!r}")
    seed_everything(config.seed)
    event_prefixes = (
        "evidence_encoder.",
        "event_trunk.",
        "ordinal_event_head.",
        "nominal_event_head.",
        "structural_event_head.",
        "evidence_event_head.",
    )
    event_parameters = []
    backbone_parameters = []
    for name, parameter in model.named_parameters():
        (event_parameters if name.startswith(event_prefixes) else backbone_parameters).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": config.learning_rate * config.backbone_learning_rate_multiplier,
            },
            {"params": event_parameters, "lr": config.learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    best_score = float("inf")
    best_epoch = -1
    history = []
    rng = random.Random(config.seed)
    raw_validation = [[transition for transition, _ in episode] for episode in validation_episodes]
    for epoch in range(config.epochs):
        model.train()
        indices = list(range(len(train_episodes)))
        rng.shuffle(indices)
        train_losses = []
        for index in indices:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = event_episode_loss(
                model, train_episodes[index], schema, balance, config, variant=variant
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        base = evaluate_model(model, raw_validation)
        events = evaluate_events(model, validation_episodes, balance, rare_threshold=config.rare_prevalence_threshold)
        event_penalty = (1.0 - events["macro_f1"]) if VARIANTS[variant]["event_enabled"] else 0.0
        score = base["state_mae"] + 0.05 * base["reward_mae"] + 0.01 * base["cost_mae"] + 0.05 * event_penalty
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                "validation_score": score,
                "validation_base": base,
                "validation_event_macro_f1": events["macro_f1"],
                "validation_event_macro_auprc": events["macro_auprc"],
            }
        )
        if score < best_score - 1e-8:
            best_score = score
            best_epoch = epoch + 1
    if not history:
        raise RuntimeError("event-aware training produced no checkpoint")
    return model, history, {
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "executed_epochs": len(history),
        "fixed_epoch_budget": True,
        "checkpoint_epoch": len(history),
        "selection_rule": "final epoch after equal fixed training budget",
    }


def config_dict(config: EventTrainingConfig) -> dict[str, Any]:
    return asdict(config)
