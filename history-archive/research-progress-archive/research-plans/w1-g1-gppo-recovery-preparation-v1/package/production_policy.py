"""Production GPPO routes and paired W1 task confirmation."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from infra_io import durable_append_jsonl, durable_atomic_json

POLICY_METHODS = ("G0", "T", "G1")
TASK_METHODS = (*POLICY_METHODS, "H")
POLICY_SEEDS = (8301, 8302, 8303)
WORLD_MODEL_VARIANTS = ("G1",)
WORLD_MODEL_SEEDS = (8201, 8202, 8203)
ACTION_COUNT = 25
NOOP_ACTION = 24
BASE_CANDIDATE_FEATURES = 17


def _canonical_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return _canonical_value(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _canonical_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        raise PolicyProductionError("NONFINITE_CANONICAL_INPUT")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise PolicyProductionError(f"CANONICAL_INPUT_TYPE_UNSUPPORTED:{type(value).__name__}")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(_canonical_value(value), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        if not torch.is_tensor(value):
            raise PolicyProductionError("CHECKPOINT_STATE_VALUE_NOT_TENSOR")
        tensor = value.detach().to(device="cpu").contiguous()
        header = json.dumps({"name": str(name), "dtype": str(tensor.dtype),
                             "shape": list(tensor.shape)}, sort_keys=True,
                            separators=(",", ":")).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


class PolicyProductionError(RuntimeError):
    """Frozen policy stage contract violation."""


class BottomBoundary(Protocol):
    def environment(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...
    def model(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...
    def optimizer(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...
    def checkpoint(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...


class DirectBoundary:
    def environment(self, operation, *args, **kwargs): return operation(*args, **kwargs)
    def model(self, operation, *args, **kwargs): return operation(*args, **kwargs)
    def optimizer(self, operation, *args, **kwargs): return operation(*args, **kwargs)
    def checkpoint(self, operation, *args, **kwargs): return operation(*args, **kwargs)


class CallAccounting:
    """Send each low-level operation through the budget ledger and bottom seam."""

    def __init__(self, ledger: Any, boundary: BottomBoundary | None = None):
        self.ledger = ledger
        self.boundary = boundary or DirectBoundary()
        self.amounts: Counter[str] = Counter()
        self.calls: Counter[str] = Counter()

    def call(self, kind: str, name: str, amounts: Mapping[str, int],
             operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if kind not in {"environment", "model", "optimizer", "checkpoint"}:
            raise PolicyProductionError(f"UNKNOWN_BOTTOM_BOUNDARY:{kind}")
        normalized = {str(key): int(value) for key, value in amounts.items()}
        if any(value < 0 for value in normalized.values()):
            raise PolicyProductionError("NEGATIVE_RESOURCE_CHARGE")
        self.calls[name] += 1
        self.amounts.update(normalized)
        boundary_operation = getattr(self.boundary, kind)
        if self.ledger is None:
            return boundary_operation(operation, *args, **kwargs)
        return self.ledger.call(name, normalized, boundary_operation, operation, *args, **kwargs)

    def charge(self, name: str, amounts: Mapping[str, int]) -> None:
        normalized = {str(key): int(value) for key, value in amounts.items()}
        if any(value < 0 for value in normalized.values()):
            raise PolicyProductionError("NEGATIVE_RESOURCE_CHARGE")
        self.calls[name] += 1
        self.amounts.update(normalized)
        if self.ledger is not None:
            self.ledger.call(name, normalized, lambda: None)

    def snapshot(self) -> dict[str, int]:
        return dict(self.amounts)


@dataclass(frozen=True)
class PolicyRoute:
    method: str
    seed: int
    steps: int
    optimizer_updates: int
    rollout_steps: int
    world_model_variant: str | None
    world_model_seed: int | None


@dataclass(frozen=True)
class PolicySchedule:
    seeds: tuple[int, ...]
    steps_per_route: int
    updates_per_route: int
    rollout_steps: int
    updates_per_rollout: int
    policy_configuration: Mapping[str, Any]
    prior_configuration: Mapping[str, Any]
    training_preferences: tuple[tuple[float, float], ...]
    task_preference: tuple[float, float]
    utility_discount: float
    task_component_scale: float

    def routes(self) -> tuple[PolicyRoute, ...]:
        paired_world_seed = dict(zip(self.seeds, WORLD_MODEL_SEEDS))
        return tuple(
            PolicyRoute(method, seed, self.steps_per_route, self.updates_per_route,
                        self.rollout_steps,
                        method if method in WORLD_MODEL_VARIANTS else None,
                        paired_world_seed[seed] if method in WORLD_MODEL_VARIANTS else None)
            for method in POLICY_METHODS for seed in self.seeds
        )

    def training_preference(self, episode_index: int, policy_seed: int) -> tuple[float, float]:
        if episode_index < 0 or policy_seed not in self.seeds:
            raise PolicyProductionError("TRAINING_PREFERENCE_IDENTITY_INVALID")
        index = (episode_index + self.seeds.index(policy_seed)) % len(self.training_preferences)
        return self.training_preferences[index]


def policy_schedule(matrix: Mapping[str, Any], request: Mapping[str, Any]) -> PolicySchedule:
    if matrix.get("schema") != "w1-eawm-jepa-experiment-matrix/1.0.0":
        raise PolicyProductionError("MATRIX_SCHEMA_MISMATCH")
    if tuple(matrix.get("methods", ())) != TASK_METHODS:
        raise PolicyProductionError("POLICY_METHOD_CONTRACT_MISMATCH")
    seeds = tuple(int(seed) for seed in matrix.get("policy_seeds", ()))
    if seeds != POLICY_SEEDS:
        raise PolicyProductionError("POLICY_SEED_CONTRACT_MISMATCH")
    if tuple(int(seed) for seed in matrix.get("world_model_seeds", ())) != WORLD_MODEL_SEEDS:
        raise PolicyProductionError("WORLD_MODEL_SEED_CONTRACT_MISMATCH")
    config, priors = matrix.get("policy_configuration"), matrix.get("prior_configuration")
    preference_config = matrix.get("preference_configuration")
    if not isinstance(config, Mapping) or not isinstance(priors, Mapping):
        raise PolicyProductionError("POLICY_OR_PRIOR_CONFIGURATION_MISSING")
    if set(priors) != set(POLICY_METHODS) | {"illegal_action_policy"}:
        raise PolicyProductionError("POLICY_PRIOR_METHOD_SET_MISMATCH")
    if not isinstance(preference_config, Mapping):
        raise PolicyProductionError("PREFERENCE_CONFIGURATION_MISSING")
    steps = int(matrix.get("policy_training_steps_per_method_seed", 0))
    rollout = int(config.get("rollout_steps", 0))
    updates = int(config.get("maximum_optimizer_updates_per_method_seed", 0))
    fixture = matrix.get("e2e_fixture")
    fixture_enabled = isinstance(fixture, Mapping) and bool(fixture.get("enabled"))
    if fixture_enabled:
        steps = int(fixture.get("policy_training_steps_per_method_seed", steps))
        rollout = int(fixture.get("policy_rollout_steps", rollout))
        updates = int(fixture.get("policy_optimizer_updates_per_method_seed", updates))
    from recovery_contract import training_route_keys
    route_count = len(training_route_keys(matrix))
    stage = request.get("stages", {}).get("conditional_policy_training", {})
    if not fixture_enabled and (steps, rollout, updates) != (2048, 64, 128):
        raise PolicyProductionError("POLICY_TRAINING_BUDGET_MISMATCH")
    if steps % rollout or updates % (steps // rollout):
        raise PolicyProductionError("POLICY_ROLLOUT_UPDATE_SCHEDULE_MISMATCH")
    if int(stage.get("policy_optimizer_updates", -1)) != route_count * updates:
        raise PolicyProductionError("POLICY_UPDATE_LEDGER_BUDGET_MISMATCH")
    expected_priors = {
        "G0": (False, False, 0.0), "T": (True, False, 0.1),
        "G1": (True, True, 0.1),
    }
    for method, expected in expected_priors.items():
        value = priors.get(method)
        if not isinstance(value, Mapping):
            raise PolicyProductionError(f"PRIOR_CONFIGURATION_MISSING:{method}")
        observed = (bool(value.get("transparent_score")), bool(value.get("learned_residual")),
                    float(value.get("prior_scale", float("nan"))))
        if observed != expected:
            raise PolicyProductionError(f"PRIOR_CONFIGURATION_MISMATCH:{method}")
    if priors.get("illegal_action_policy") != "original mask remains authoritative":
        raise PolicyProductionError("ILLEGAL_ACTION_POLICY_MISMATCH")
    for key in ("optimizer", "learning_rate", "gamma", "gae_lambda", "clip_epsilon",
                "entropy_weight", "value_weight", "gradient_clip_norm"):
        if key not in config:
            raise PolicyProductionError(f"POLICY_CONFIGURATION_MISSING:{key}")
    training_preferences = tuple(
        tuple(float(component) for component in preference)
        for preference in preference_config.get("training_schedule", ())
    )
    task_preference = tuple(float(component) for component in preference_config.get("task_confirmation", ()))
    if training_preferences != ((0.2, 0.8), (0.5, 0.5), (0.8, 0.2)):
        raise PolicyProductionError("TRAINING_PREFERENCE_SCHEDULE_MISMATCH")
    if preference_config.get("training_assignment") != "episode_index plus policy_seed_index modulo 3":
        raise PolicyProductionError("TRAINING_PREFERENCE_ASSIGNMENT_MISMATCH")
    if task_preference != (0.8, 0.2):
        raise PolicyProductionError("TASK_PREFERENCE_MISMATCH")
    discount = float(preference_config.get("discount", float("nan")))
    task_scale = float(preference_config.get("task_component_scale", float("nan")))
    if not math.isfinite(discount) or discount != 0.99:
        raise PolicyProductionError("UTILITY_DISCOUNT_MISMATCH")
    if not math.isfinite(task_scale) or task_scale != 0.5:
        raise PolicyProductionError("UTILITY_TASK_SCALE_MISMATCH")
    return PolicySchedule(
        seeds, steps, updates, rollout, updates // (steps // rollout), dict(config),
        {key: dict(value) for key, value in priors.items() if isinstance(value, Mapping)},
        training_preferences, task_preference, discount, task_scale,
    )


def _finite_tensor(value: Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all()):
        raise PolicyProductionError(f"NONFINITE_{name.upper()}")
    return tensor


def transparent_utility_components(diagnostic: Mapping[str, Any], *,
                                   task_capacity: int,
                                   initial_total_energy: float) -> tuple[float, float]:
    """Map the frozen public heuristic into the original vector-reward units."""
    if task_capacity <= 0 or not math.isfinite(initial_total_energy) or initial_total_energy <= 0:
        raise PolicyProductionError("TRANSPARENT_UTILITY_NORMALIZATION_INVALID")
    score = float(diagnostic.get("score", float("nan")))
    energy_cost = float(diagnostic.get("energy_cost", float("nan")))
    if not math.isfinite(score) or not math.isfinite(energy_cost) or energy_cost < 0:
        raise PolicyProductionError("TRANSPARENT_UTILITY_DIAGNOSTIC_INVALID")
    task_component = (score + 0.1 * energy_cost) / (100.0 * task_capacity)
    energy_component = -energy_cost / initial_total_energy
    if not math.isfinite(task_component) or not math.isfinite(energy_component):
        raise PolicyProductionError("TRANSPARENT_UTILITY_COMPONENT_NONFINITE")
    return task_component, energy_component


def augment_candidate_features(candidate_features: Any, prior_logits: Any) -> torch.Tensor:
    """Carry additive prior in PPO transitions so replay uses behavior logits."""
    features = _finite_tensor(candidate_features, name="candidate_features")
    prior = _finite_tensor(prior_logits, name="prior_logits")
    if features.ndim != 3 or features.shape[1:] != (ACTION_COUNT, BASE_CANDIDATE_FEATURES):
        raise PolicyProductionError("BASE_CANDIDATE_FEATURE_SHAPE_MISMATCH")
    if prior.ndim == 1:
        prior = prior.unsqueeze(0)
    if prior.shape != features.shape[:2]:
        raise PolicyProductionError("PRIOR_LOGIT_SHAPE_MISMATCH")
    return torch.cat((features, prior.unsqueeze(-1)), dim=-1)


class PriorConditionedPolicy(nn.Module):
    """Reuse the frozen public GPPO policy and add a non-trainable score prior."""

    def __init__(self, base_policy: nn.Module):
        super().__init__()
        self.base = base_policy

    def encode(self, *args: Any, **kwargs: Any):
        return self.base.encode(*args, **kwargs)

    def evaluate_encoded(self, features: torch.Tensor, pair_messages: torch.Tensor,
                         preference: torch.Tensor, candidate_features: torch.Tensor,
                         mask: torch.Tensor) -> dict[str, Any]:
        if candidate_features.ndim != 3 or candidate_features.shape[1:] != (ACTION_COUNT, BASE_CANDIDATE_FEATURES + 1):
            raise PolicyProductionError("AUGMENTED_CANDIDATE_FEATURE_SHAPE_MISMATCH")
        base_features = candidate_features[..., :BASE_CANDIDATE_FEATURES]
        prior = candidate_features[..., BASE_CANDIDATE_FEATURES]
        legal = mask.to(dtype=torch.bool)
        if legal.shape != (features.shape[0], ACTION_COUNT) or not bool(legal.any(dim=-1).all()):
            raise PolicyProductionError("POLICY_LEGAL_MASK_INVALID")
        if not bool(torch.isfinite(base_features).all()) or not bool(torch.isfinite(prior).all()):
            raise PolicyProductionError("NONFINITE_POLICY_CANDIDATE_INPUT")
        base_result = self.base.evaluate_encoded(features, pair_messages, preference, base_features, legal)
        base_logits = base_result["logits"]
        if base_logits.shape != legal.shape or not bool(torch.isfinite(base_logits[legal]).all()):
            raise PolicyProductionError("BASE_POLICY_LOGITS_INVALID")
        replay_logits = base_logits + prior
        if not bool(torch.isfinite(replay_logits).all()):
            raise PolicyProductionError("REPLAY_POLICY_LOGITS_INVALID")
        final_logits = replay_logits.masked_fill(~legal, -torch.inf)
        if not bool(torch.isfinite(final_logits[legal]).all()):
            raise PolicyProductionError("FINAL_POLICY_LOGITS_INVALID")
        distribution = Categorical(logits=final_logits)
        result = dict(base_result)
        result.update({"base_logits": base_logits, "prior_logits": prior,
                      # GPPO/PreCo losses require finite values for every action.
                      # The behavior distribution remains strictly mask-aware.
                      "logits": replay_logits, "final_logits": final_logits,
                      "distribution": distribution, "probabilities": distribution.probs})
        return result

    def evaluate_observation(self, obs: torch.Tensor, preference: torch.Tensor,
                             candidate_features: torch.Tensor, mask: torch.Tensor,
                             hidden: torch.Tensor | None = None) -> dict[str, Any]:
        features, pair, next_hidden = self.encode(obs, hidden)
        result = self.evaluate_encoded(features, pair, preference, candidate_features, mask)
        result.update({"features": features, "pair_messages": pair, "next_hidden": next_hidden})
        return result

@dataclass(frozen=True)
class PolicyDecision:
    action: int
    log_probability: float
    candidate_features: torch.Tensor
    next_hidden: torch.Tensor | None
    trace: Mapping[str, Any]
    cost: Mapping[str, Any] | None
    values: torch.Tensor | None = None


class DecisionPriorPolicy:
    """Apply the frozen prior to one decision's original legal action set."""

    def __init__(self, *, method: str, seed: int, policy: PriorConditionedPolicy,
                 prior_configuration: Mapping[str, Mapping[str, Any]],
                 input_builder: Callable[[Any], Mapping[str, Any]],
                 transparent_scorer: Callable[[Any, Mapping[str, Any]], Any] | None,
                 world_models: Sequence[Any],
                 world_predictor: Callable[[str, Any, Any, torch.Tensor, torch.Tensor], Any] | None,
                 calls: CallAccounting, parent: str, repeat: int,
                 clock: Any = time, selector: Callable[[Categorical], int] | None = None):
        if method not in POLICY_METHODS:
            raise PolicyProductionError(f"UNKNOWN_POLICY_METHOD:{method}")
        config = prior_configuration.get(method)
        if not isinstance(config, Mapping):
            raise PolicyProductionError(f"PRIOR_CONFIGURATION_MISSING:{method}")
        if bool(config["learned_residual"]) and (not world_models or world_predictor is None):
            raise PolicyProductionError(f"FROZEN_WORLD_MODEL_MISSING:{method}")
        if not bool(config["learned_residual"]) and world_models:
            raise PolicyProductionError(f"UNEXPECTED_WORLD_MODEL:{method}")
        if bool(config["transparent_score"]) and transparent_scorer is None:
            raise PolicyProductionError("TRANSPARENT_SCORER_MISSING")
        self.method = method
        self.seed = int(seed)
        self.policy = policy
        self.prior_config = dict(config)
        self.input_builder = input_builder
        self.transparent_scorer = transparent_scorer
        self.world_models = tuple(world_models)
        self.world_predictor = world_predictor
        self.calls = calls
        self.parent = str(parent)
        self.repeat = int(repeat)
        self.clock = clock
        self.selector_injected = selector is not None
        self.selector = selector or (lambda distribution: int(distribution.sample().item()))
        self.cost_rows: list[dict[str, Any]] = []
        self.decision_count = 0
        self.base_argmax_counts: Counter[int] = Counter()
        self.prior_adjusted_argmax_counts: Counter[int] = Counter()
        self.argmax_changed_count = 0
        self.sampled_action_changed_count = 0
        self.sampled_action_comparison_count = 0

    @staticmethod
    def _residuals(value: Any, expected_actions: torch.Tensor,
                   preference: Any) -> torch.Tensor:
        if not isinstance(value, Mapping):
            raise PolicyProductionError("WORLD_PREDICTION_CANDIDATE_IDENTITIES_MISSING")
        if "candidate_actions" not in value:
            raise PolicyProductionError("WORLD_PREDICTION_CANDIDATE_IDENTITIES_MISSING")
        returned_actions = torch.as_tensor(value["candidate_actions"])
        if returned_actions.ndim != 1 or returned_actions.dtype == torch.bool:
            raise PolicyProductionError("WORLD_PREDICTION_CANDIDATE_IDENTITIES_INVALID")
        if returned_actions.is_floating_point():
            if not bool(torch.isfinite(returned_actions).all()) or not bool((returned_actions == returned_actions.round()).all()):
                raise PolicyProductionError("WORLD_PREDICTION_CANDIDATE_IDENTITIES_INVALID")
        returned_actions = returned_actions.to(dtype=torch.long, device="cpu")
        expected_actions = expected_actions.to(dtype=torch.long, device="cpu").reshape(-1)
        if not torch.equal(returned_actions, expected_actions):
            raise PolicyProductionError("WORLD_PREDICTION_CANDIDATE_IDENTITIES_MISMATCH")

        if "utility_residuals" in value:
            prediction = value["utility_residuals"]
        elif "outcomes" in value:
            outcomes = _finite_tensor(value["outcomes"], name="world_outcomes")
            if outcomes.ndim != 2 or outcomes.shape[0] != len(expected_actions) or outcomes.shape[1] <= 4:
                raise PolicyProductionError("WORLD_OUTCOME_SHAPE_MISMATCH")
            pref = _finite_tensor(preference, name="policy_preference").reshape(-1)
            if pref.shape != (2,):
                raise PolicyProductionError("POLICY_PREFERENCE_SHAPE_MISMATCH")
            prediction = 0.5 * pref[0] * outcomes[:, 4] + pref[1] * outcomes[:, 3]
        else:
            raise PolicyProductionError("WORLD_PREDICTION_MISSING_UTILITY_RESIDUAL")
        result = _finite_tensor(prediction, name="world_utility_residual").reshape(-1)
        if result.shape != (len(expected_actions),):
            raise PolicyProductionError("WORLD_RESIDUAL_SHAPE_MISMATCH")
        dense = torch.zeros(ACTION_COUNT, dtype=result.dtype, device="cpu")
        dense[expected_actions] = result.to(device="cpu")
        return dense

    def _prior(self, observation: Any, inputs: Mapping[str, Any], mask: torch.Tensor) -> tuple[torch.Tensor, int, int, dict[str, Any]]:
        transparent = torch.zeros(ACTION_COUNT, dtype=torch.float32)
        transparent_diagnostics: Mapping[str, Any] = {}
        if self.prior_config["transparent_score"]:
            transparent_result = self.transparent_scorer(observation, inputs)
            if isinstance(transparent_result, Mapping):
                transparent_diagnostics = transparent_result.get("components", {})
                transparent_result = transparent_result.get("scores")
            transparent = _finite_tensor(transparent_result, name="transparent_scores").reshape(-1)
            if transparent.shape != (ACTION_COUNT,):
                raise PolicyProductionError("TRANSPARENT_SCORE_SHAPE_MISMATCH")
        residuals = []
        residual_diagnostics = []
        candidate_actions = torch.nonzero(
            mask.reshape(-1).to(dtype=torch.bool, device="cpu"), as_tuple=False,
        ).flatten().to(dtype=torch.long)
        forwards = 0
        sample_evaluations = 0
        if self.prior_config["learned_residual"]:
            for model in self.world_models:
                raw = self.calls.call(
                    "model", f"{self.method.lower()}_candidate_prior_forward",
                    {"world_batch_forwards": 1, "world_sample_evaluations": int(candidate_actions.numel())},
                    self.world_predictor, self.method, model, observation,
                    candidate_actions, mask.reshape(-1).cpu(),
                )
                residuals.append(self._residuals(raw, candidate_actions, inputs["preference"]))
                model_row: dict[str, Any] = {
                    "world_model_variant": self.method,
                    "world_model_seed": WORLD_MODEL_SEEDS[POLICY_SEEDS.index(self.seed)],
                    "candidate_actions": candidate_actions.tolist(),
                }
                if isinstance(raw, Mapping) and "outcomes" in raw:
                    outcomes = _finite_tensor(raw["outcomes"], name="world_outcomes")
                    if outcomes.shape != (int(candidate_actions.numel()), 6):
                        raise PolicyProductionError("WORLD_OUTCOME_SHAPE_MISMATCH")
                    model_row["raw_six_dimensional_residuals"] = {
                        str(int(action)): vector
                        for action, vector in zip(candidate_actions.tolist(), outcomes.detach().cpu().tolist())
                    }
                    model_row["utility_residuals"] = {
                        str(int(action)): float(residuals[-1][int(action)])
                        for action in candidate_actions.tolist()
                    }
                elif isinstance(raw, Mapping) and "utility_residuals" in raw:
                    raw_residuals = _finite_tensor(raw["utility_residuals"], name="world_utility_residual").reshape(-1)
                    if raw_residuals.shape != (int(candidate_actions.numel()),):
                        raise PolicyProductionError("WORLD_RESIDUAL_SHAPE_MISMATCH")
                    model_row["utility_residuals"] = {
                        str(int(action)): float(value)
                        for action, value in zip(candidate_actions.tolist(), raw_residuals.detach().cpu().tolist())
                    }
                residual_diagnostics.append(model_row)
                forwards += 1
                sample_evaluations += int(candidate_actions.numel())
        score = transparent
        if residuals:
            score = score + torch.stack(residuals).mean(dim=0)
        score = score * float(self.prior_config["prior_scale"])
        legal = mask.reshape(-1).to(dtype=torch.bool, device="cpu")
        if not bool(torch.isfinite(score[legal]).all()):
            raise PolicyProductionError("LEGAL_PRIOR_SCORE_INVALID")
        score[~legal] = 0.0
        diagnostics = {
            "transparent_horizon_components": _canonical_value(transparent_diagnostics),
            "world_model_predictions": residual_diagnostics,
            "residual_utility_by_action": {
                str(int(action)): float(torch.stack(residuals).mean(dim=0)[int(action)])
                for action in candidate_actions.tolist()
            } if residuals else {},
            "final_prior_by_action": {str(int(action)): float(score[int(action)])
                                      for action in candidate_actions.tolist()},
        }
        return score, forwards, sample_evaluations, diagnostics

    def select(self, observation: Any, *, decision_step: int,
               hidden: torch.Tensor | None = None,
               clock_start: tuple[float, float] | None = None) -> PolicyDecision:
        self.decision_count += 1
        if clock_start is None:
            wall_started = self.clock.perf_counter()
            cpu_started = self.clock.process_time()
        else:
            if (not isinstance(clock_start, (tuple, list)) or len(clock_start) != 2
                    or not all(math.isfinite(float(value)) for value in clock_start)):
                raise PolicyProductionError("DECISION_CLOCK_START_INVALID")
            wall_started, cpu_started = map(float, clock_start)
        inputs = self.input_builder(observation)
        required = {"obs_tensor", "preference", "candidate_features", "mask"}
        missing = required.difference(inputs)
        if missing:
            raise PolicyProductionError("DECISION_INPUT_MISSING:" + ",".join(sorted(missing)))
        obs_tensor = torch.as_tensor(inputs["obs_tensor"], dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        preference = torch.as_tensor(inputs["preference"], dtype=torch.float32)
        mask = torch.as_tensor(inputs["mask"], dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != (1, ACTION_COUNT) or not bool(mask[0, NOOP_ACTION]):
            raise PolicyProductionError("DECISION_LEGAL_MASK_INVALID")
        candidate_features = _finite_tensor(inputs["candidate_features"], name="candidate_features")
        if candidate_features.ndim == 2:
            candidate_features = candidate_features.unsqueeze(0)
        if candidate_features.shape != (1, ACTION_COUNT, BASE_CANDIDATE_FEATURES):
            raise PolicyProductionError("BASE_CANDIDATE_FEATURE_SHAPE_MISMATCH")
        candidate_count = int(mask.sum().item())
        with torch.no_grad():
            encoded = self.calls.call(
                "model", f"{self.method.lower()}_policy_encode",
                {"encode_sample_evaluations": 1}, self.policy.base.encode, obs_tensor, hidden,
            )
            if not isinstance(encoded, (tuple, list)) or len(encoded) != 3:
                raise PolicyProductionError("POLICY_ENCODER_OUTPUT_INVALID")
            features, pair_messages, next_hidden = encoded
            prior, world_forwards, world_samples, prior_diagnostics = self._prior(observation, inputs, mask)
            augmented = augment_candidate_features(candidate_features, prior.unsqueeze(0)).to(features.device)
            preference = preference.to(features.device)
            mask = mask.to(features.device)
            evaluated = self.calls.call(
                "model", f"{self.method.lower()}_policy_action_select",
                {"actor_sample_evaluations": 1}, self.policy.evaluate_encoded,
                features, pair_messages, preference, augmented, mask,
            )
        legal = mask[0].to(dtype=torch.bool)
        base_argmax_action = int(evaluated["base_logits"][0].masked_fill(~legal, -torch.inf).argmax().item())
        prior_adjusted_argmax_action = int(evaluated["final_logits"][0].argmax().item())
        prior_changed = base_argmax_action != prior_adjusted_argmax_action
        self.base_argmax_counts[base_argmax_action] += 1
        self.prior_adjusted_argmax_counts[prior_adjusted_argmax_action] += 1
        self.argmax_changed_count += int(prior_changed)
        sampled_action_change_evaluated = not self.selector_injected
        selected_action_without_prior = None
        if sampled_action_change_evaluated:
            pre_selection_rng_state = torch.get_rng_state()
        action = int(self.selector(evaluated["distribution"]))
        if sampled_action_change_evaluated:
            post_selection_rng_state = torch.get_rng_state()
            try:
                torch.set_rng_state(pre_selection_rng_state)
                no_prior_logits = evaluated["base_logits"].masked_fill(~mask, -torch.inf)
                selected_action_without_prior = int(Categorical(logits=no_prior_logits).sample()[0].item())
            finally:
                torch.set_rng_state(post_selection_rng_state)
            self.sampled_action_comparison_count += 1
            self.sampled_action_changed_count += int(selected_action_without_prior != action)
        if action < 0 or action >= ACTION_COUNT or not bool(mask[0, action]):
            raise PolicyProductionError("POLICY_SELECTED_ILLEGAL_ACTION")
        log_probability = float(evaluated["distribution"].log_prob(
            torch.tensor([action], dtype=torch.long, device=features.device),
        )[0].detach().cpu())
        if not math.isfinite(log_probability):
            raise PolicyProductionError("SELECTED_ACTION_LOG_PROBABILITY_INVALID")
        cpu_seconds = float(self.clock.process_time() - cpu_started)
        wall_seconds = float(self.clock.perf_counter() - wall_started)
        if not math.isfinite(cpu_seconds) or not math.isfinite(wall_seconds) or cpu_seconds < 0 or wall_seconds < 0:
            raise PolicyProductionError("DECISION_CLOCK_INVALID")
        legal_actions = torch.nonzero(mask[0], as_tuple=False).flatten().detach().cpu().tolist()
        has_non_noop_candidate = any(int(value) != NOOP_ACTION for value in legal_actions)
        g1_cost_eligible = bool(self.method == "G1" and has_non_noop_candidate and world_forwards > 0)
        route_identity = observation.get("decision_identity", {}) if isinstance(observation, Mapping) else {}
        if not isinstance(route_identity, Mapping):
            raise PolicyProductionError("DECISION_ROUTE_IDENTITY_INVALID")
        history_schema = observation.get("public_history_schema") if isinstance(observation, Mapping) else None
        history_state = observation.get("public_history") if isinstance(observation, Mapping) else None
        history_identity = None if history_state is None else {
            "schema": history_schema, "sha256": canonical_sha256(history_state),
        }
        public_input_sha256 = canonical_sha256({
            "inputs": inputs,
            "public_history_schema": history_schema,
            "public_history": history_state,
            "public_entity_ids": observation.get("public_entity_ids") if isinstance(observation, Mapping) else None,
        })
        entity_ids = observation.get("public_entity_ids", {}) if isinstance(observation, Mapping) else {}
        uav_ids, task_ids = tuple(entity_ids.get("uavs", ())), tuple(entity_ids.get("tasks", ()))
        legal_candidate_identities = []
        for legal_action in legal_actions:
            if int(legal_action) == NOOP_ACTION:
                legal_candidate_identities.append({"action_id": NOOP_ACTION, "candidate_id": "NOOP"})
                continue
            uav_index, task_index = divmod(int(legal_action), 6)
            if uav_index >= len(uav_ids) or task_index >= len(task_ids):
                raise PolicyProductionError("LEGAL_CANDIDATE_PUBLIC_IDENTITY_MISSING")
            legal_candidate_identities.append({
                "action_id": int(legal_action),
                "candidate_id": f"{uav_ids[uav_index]}|{task_ids[task_index]}",
                "uav_id": str(uav_ids[uav_index]), "task_id": str(task_ids[task_index]),
            })
        cost = {
            "method": self.method, "parent": self.parent, "repeat": self.repeat,
            "seed": self.seed, "decision_step": int(decision_step),
            "candidate_count": candidate_count, "cpu_seconds": cpu_seconds,
            "wall_seconds": wall_seconds, "world_model_forwards": world_forwards,
            "world_model_sample_evaluations": world_samples,
            "selected_action": action, "noop_selected": action == NOOP_ACTION,
            "has_non_noop_candidate": has_non_noop_candidate,
            "g1_cost_eligible": g1_cost_eligible,
            "sampled_action_change_evaluated": sampled_action_change_evaluated,
            "policy_sampling_key": route_identity.get("policy_sampling_key"),
            "policy_sampling_seed": route_identity.get("policy_sampling_seed"),
            "cost_scope": "shared public preparation/history + features + world-model forwards + ensemble + policy logits + action selection",
        }
        self.cost_rows.append(cost)
        trace = {
            "method": self.method, "seed": self.seed, "decision_step": int(decision_step),
            "parent": route_identity.get("parent", self.parent),
            "repeat": route_identity.get("repeat", self.repeat),
            "exogenous_key": route_identity.get("exogenous_key"),
            "scenario_id": route_identity.get("scenario_id"),
            "policy_sampling_key": route_identity.get("policy_sampling_key"),
            "policy_sampling_seed": route_identity.get("policy_sampling_seed"),
            "public_input_sha256": public_input_sha256,
            "public_history_identity": history_identity,
            "legal_actions": legal_actions,
            "legal_candidate_identities": legal_candidate_identities,
            "candidate_count": candidate_count,
            "base_logits": evaluated["base_logits"][0].detach().cpu().tolist(),
            "prior_logits": evaluated["prior_logits"][0].detach().cpu().tolist(),
            "final_logits": [float(x) if math.isfinite(float(x)) else None
                             for x in evaluated["final_logits"][0].detach().cpu().tolist()],
            "base_argmax_action": base_argmax_action,
            "prior_adjusted_argmax_action": prior_adjusted_argmax_action,
            "prior_changed": prior_changed,
            "argmax_changed": prior_changed,
            "selected_action_without_prior_same_rng": selected_action_without_prior,
            "sampled_action_change_evaluated": sampled_action_change_evaluated,
            "action_probabilities": evaluated["probabilities"][0].detach().cpu().tolist(),
            "selected_action": action,
            **prior_diagnostics,
        }
        values = evaluated.get("state_values", evaluated.get("critic_values"))
        if values is None:
            raise PolicyProductionError("POLICY_STATE_VALUE_MISSING")
        values = _finite_tensor(values, name="state_values").reshape(-1).detach().cpu()
        if values.shape != (2,):
            raise PolicyProductionError("POLICY_STATE_VALUE_SHAPE_MISMATCH")
        return PolicyDecision(action, log_probability, augmented, next_hidden, trace, cost, values)

    def decision_summary(self) -> dict[str, Any]:
        return {
            "raw_decision_count": self.decision_count,
            "base_argmax_action_counts": {
                str(key): value for key, value in sorted(self.base_argmax_counts.items())
            },
            "prior_adjusted_argmax_action_counts": {
                str(key): value for key, value in sorted(self.prior_adjusted_argmax_counts.items())
            },
            "argmax_changed_count": self.argmax_changed_count,
            "sampled_action_changed_count": self.sampled_action_changed_count,
            "sampled_action_comparison_count": self.sampled_action_comparison_count,
            "sampled_action_comparison_status": (
                "evaluated" if self.sampled_action_comparison_count == self.decision_count
                else "partially_evaluated" if self.sampled_action_comparison_count
                else "not_evaluated"
            ),
        }

    def bootstrap_value(self, observation: Any, *, hidden: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluate V(s,p) at a rollout boundary without selecting an action.

        The frozen native critic is state/preference conditioned and does not
        consume candidate consequences.  This avoids an extra world-model
        forward while preserving the behavior-version bootstrap required by
        GAE.  The call is accounted as one encoder and actor/critic evaluation.
        """
        inputs = self.input_builder(observation)
        required = {"obs_tensor", "preference", "candidate_features", "mask"}
        missing = required.difference(inputs)
        if missing:
            raise PolicyProductionError("BOOTSTRAP_INPUT_MISSING:" + ",".join(sorted(missing)))
        obs_tensor = torch.as_tensor(inputs["obs_tensor"], dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        preference = torch.as_tensor(inputs["preference"], dtype=torch.float32)
        mask = torch.as_tensor(inputs["mask"], dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != (1, ACTION_COUNT) or not bool(mask.any()):
            raise PolicyProductionError("BOOTSTRAP_LEGAL_MASK_INVALID")
        with torch.no_grad():
            encoded = self.calls.call(
                "model", f"{self.method.lower()}_bootstrap_encode",
                {"encode_sample_evaluations": 1}, self.policy.base.encode, obs_tensor, hidden,
            )
            if not isinstance(encoded, (tuple, list)) or len(encoded) != 3:
                raise PolicyProductionError("BOOTSTRAP_ENCODER_OUTPUT_INVALID")
            features, pair_messages, _ = encoded
            base_candidates = torch.zeros(
                (1, ACTION_COUNT, BASE_CANDIDATE_FEATURES),
                dtype=features.dtype, device=features.device,
            )
            augmented = augment_candidate_features(
                base_candidates, torch.zeros((1, ACTION_COUNT), dtype=features.dtype, device=features.device),
            )
            evaluated = self.calls.call(
                "model", f"{self.method.lower()}_bootstrap_value",
                {"actor_sample_evaluations": 1}, self.policy.evaluate_encoded,
                features, pair_messages, preference.to(features.device),
                augmented, mask.to(features.device),
            )
        values = evaluated.get("state_values", evaluated.get("critic_values"))
        if values is None:
            raise PolicyProductionError("BOOTSTRAP_STATE_VALUE_MISSING")
        result = _finite_tensor(values, name="bootstrap_state_values").reshape(-1).detach().cpu()
        if result.shape != (2,):
            raise PolicyProductionError("BOOTSTRAP_STATE_VALUE_SHAPE_MISMATCH")
        return result

@dataclass
class PolicyTrainingContext:
    route: PolicyRoute
    policy: PriorConditionedPolicy
    optimizer: Any
    policy_config: Mapping[str, Any]
    calls: CallAccounting
    decision_policy: DecisionPriorPolicy
    update_policy: Callable[..., Any]
    schedule: PolicySchedule


def _checkpoint_path(output: Path | None, method: str, seed: int) -> str:
    return f"{method}-seed-{seed}" if output is None else str(output / "policy-checkpoints" / f"{method}-seed-{seed}.pt")


def _file_sha256(path: str | Path) -> str | None:
    target = Path(path)
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_policy_routes(*, matrix: Mapping[str, Any], request: Mapping[str, Any], ledger: Any,
                        boundary: BottomBoundary | None,
                        world_model_loader: Callable[[str, int], Any],
                        policy_factory: Callable[[str, int, Mapping[str, Any]], nn.Module],
                        optimizer_factory: Callable[[nn.Module, str, int, Mapping[str, Any]], Any],
                        route_runner: Callable[[PolicyTrainingContext], Mapping[str, Any]],
                        ppo_update_fn: Callable[..., Any], checkpoint_writer: Callable[..., Any],
                        input_builder: Callable[[Any], Mapping[str, Any]],
                        transparent_scorer: Callable[[Any, Mapping[str, Any]], Any] | None,
                        world_predictor: Callable[[str, Any, Any, torch.Tensor, torch.Tensor], Any] | None,
                        output_dir: str | Path | None = None, clock: Any = time,
                        reused_routes: Mapping[tuple[str, int], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Run the frozen 3-policy-method × 3-seed stage through supplied native ops."""
    schedule = policy_schedule(matrix, request)
    calls = CallAccounting(ledger, boundary)
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    frozen_models: dict[tuple[str, int], Any] = {}
    world_model_hashes: dict[tuple[str, int], str] = {}
    for variant in WORLD_MODEL_VARIANTS:
        for wm_seed in WORLD_MODEL_SEEDS:
            model = calls.call(
                "checkpoint", f"load_frozen_world_model:{variant}:{wm_seed}",
                {"checkpoint_loads": 1, "model_initializations_or_loads": 1},
                world_model_loader, variant, wm_seed,
            )
            if not isinstance(model, nn.Module):
                raise PolicyProductionError("WORLD_MODEL_LOADER_MUST_RETURN_MODULE")
            model.eval()
            model.requires_grad_(False)
            if model.training or any(parameter.requires_grad for parameter in model.parameters()):
                raise PolicyProductionError("FROZEN_WORLD_MODEL_STATE_INVALID")
            frozen_models[(variant, wm_seed)] = model
            world_model_hashes[(variant, wm_seed)] = state_dict_sha256(model.state_dict())
    routes: list[dict[str, Any]] = []
    policies: dict[tuple[str, int], PriorConditionedPolicy] = {}
    checkpoint_paths: dict[tuple[str, int], str] = {}
    from recovery_contract import training_route_keys, REUSE_KEYS
    allowed_training = training_route_keys(matrix)
    reuse = dict(reused_routes or {})
    expected_reuse = REUSE_KEYS if 'recovery' in matrix else set()
    if set(reuse) != expected_reuse:
        raise PolicyProductionError('POLICY_REUSE_ROUTE_SET_MISMATCH')
    for route in schedule.routes():
        key = (route.method, route.seed)
        if key in reuse:
            row = dict(reuse[key])
            if _file_sha256(row['checkpoint']) != row['checkpoint_sha256']:
                raise PolicyProductionError('REUSED_CHECKPOINT_CHANGED')
            routes.append(row)
            checkpoint_paths[key] = row['checkpoint']
            if output is not None:
                durable_append_jsonl(output / 'policy-reused-routes.jsonl', row)
            continue
        if key not in allowed_training:
            raise PolicyProductionError('POLICY_UNAPPROVED_TRAIN_ROUTE')
        before = calls.snapshot()
        base = calls.call(
            "model", f"initialize_policy:{route.method}:{route.seed}",
            {"model_initializations_or_loads": 1}, policy_factory,
            route.method, route.seed, schedule.policy_configuration,
        )
        if not isinstance(base, nn.Module):
            raise PolicyProductionError("POLICY_FACTORY_MUST_RETURN_MODULE")
        policy = PriorConditionedPolicy(base)
        optimizer = calls.call(
            "optimizer", f"initialize_policy_optimizer:{route.method}:{route.seed}", {},
            optimizer_factory, policy, route.method, route.seed, schedule.policy_configuration,
        )
        wm = () if route.world_model_variant is None else (
            frozen_models[(route.world_model_variant, int(route.world_model_seed))],
        )
        decision_policy = DecisionPriorPolicy(
            method=route.method, seed=route.seed, policy=policy,
            prior_configuration=schedule.prior_configuration,
            input_builder=input_builder, transparent_scorer=transparent_scorer,
            world_models=wm, world_predictor=world_predictor, calls=calls,
            parent="policy-training", repeat=0, clock=clock,
        )

        def update_policy(transitions: Sequence[Mapping[str, Any]], *args: Any,
                          _route: PolicyRoute = route, _policy: PriorConditionedPolicy = policy,
                          _optimizer: Any = optimizer, **kwargs: Any) -> Any:
            if len(transitions) != _route.rollout_steps:
                raise PolicyProductionError("PPO_UPDATE_BATCH_SIZE_MISMATCH")
            if "device" not in kwargs:
                raise PolicyProductionError("PPO_UPDATE_DEVICE_REQUIRED")
            device = kwargs.pop("device")
            return calls.call(
                "optimizer", f"ppo_update:{_route.method}:{_route.seed}",
                {"policy_optimizer_updates": 1,
                 "policy_backward_calls": 1,
                 "encode_sample_evaluations": len(transitions),
                 "actor_sample_evaluations": len(transitions)},
                ppo_update_fn, _policy, transitions, _optimizer,
                schedule.policy_configuration, device, *args,
                event_group=_route.method, **kwargs,
            )

        context = PolicyTrainingContext(route, policy, optimizer, schedule.policy_configuration,
                                        calls, decision_policy, update_policy, schedule)
        summary = route_runner(context)
        if not isinstance(summary, Mapping):
            raise PolicyProductionError("POLICY_ROUTE_SUMMARY_INVALID")
        current = calls.snapshot()
        if current.get("environment_steps", 0) - before.get("environment_steps", 0) != route.steps:
            raise PolicyProductionError(f"POLICY_ROUTE_STEP_COUNT_MISMATCH:{route.method}:{route.seed}")
        if current.get("policy_optimizer_updates", 0) - before.get("policy_optimizer_updates", 0) != route.optimizer_updates:
            raise PolicyProductionError(f"POLICY_ROUTE_UPDATE_COUNT_MISMATCH:{route.method}:{route.seed}")
        if decision_policy.decision_count != route.steps:
            raise PolicyProductionError(f"POLICY_ROUTE_DECISION_COUNT_MISMATCH:{route.method}:{route.seed}")
        checkpoint = _checkpoint_path(output, route.method, route.seed)
        saved = calls.call(
            "checkpoint", f"save_policy_checkpoint:{route.method}:{route.seed}",
            {"checkpoint_writes": 1}, checkpoint_writer,
            checkpoint, policy, optimizer, route, dict(summary),
        )
        if saved is not None:
            checkpoint = str(saved)
        checkpoint_file_sha256 = _file_sha256(checkpoint)
        checkpoint_state_digest = state_dict_sha256(policy.base.state_dict())
        if checkpoint_file_sha256 is None:
            raise PolicyProductionError("POLICY_CHECKPOINT_FILE_MISSING_AFTER_SAVE")
        row = {
            "method": route.method, "seed": route.seed,
            "world_model_variant": route.world_model_variant,
            "world_model_seed": route.world_model_seed,
            "training_steps": route.steps, "optimizer_updates": route.optimizer_updates,
            "rollout_steps": route.rollout_steps,
            "updates_per_rollout": schedule.updates_per_rollout,
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_file_sha256,
            "checkpoint_state_sha256": checkpoint_state_digest,
            "summary": {key: value for key, value in summary.items()
                        if key not in {"decision_traces", "ppo_update_traces"}},
            "ppo_update_count": len(summary.get("ppo_update_traces", ())),
        }
        routes.append(row)
        policies[(route.method, route.seed)] = policy
        checkpoint_paths[(route.method, route.seed)] = checkpoint
        if output is not None:
            update_traces = summary.get("ppo_update_traces", ())
            if not isinstance(update_traces, Sequence):
                raise PolicyProductionError("PPO_UPDATE_TRACES_INVALID")
            for trace in update_traces:
                if not isinstance(trace, Mapping) or not isinstance(trace.get("metrics"), Mapping):
                    raise PolicyProductionError("PPO_UPDATE_TRACE_INVALID")
                durable_append_jsonl(output / "policy-update-traces.jsonl", trace)
            decision_traces = summary.get("decision_traces", ())
            if not isinstance(decision_traces, Sequence):
                raise PolicyProductionError("POLICY_DECISION_TRACES_INVALID")
            for trace in decision_traces:
                if not isinstance(trace, Mapping):
                    raise PolicyProductionError("POLICY_DECISION_TRACE_INVALID")
                durable_append_jsonl(output / "policy-training-traces.jsonl", trace)
            durable_append_jsonl(output / "policy-training-routes.jsonl", row)
    expected = {(method, seed) for method in POLICY_METHODS for seed in POLICY_SEEDS}
    if set(checkpoint_paths) != expected or set(policies) != allowed_training or len(routes) != 9:
        raise PolicyProductionError("POLICY_TRAINING_ROUTE_SET_MISMATCH")
    final_world_model_hashes = {
        key: state_dict_sha256(model.state_dict()) for key, model in frozen_models.items()
    }
    world_model_evidence = [
        {"variant": variant, "seed": seed,
         "initial_state_sha256": world_model_hashes[(variant, seed)],
         "final_state_sha256": final_world_model_hashes[(variant, seed)],
         "unchanged": world_model_hashes[(variant, seed)] == final_world_model_hashes[(variant, seed)]}
        for variant, seed in sorted(world_model_hashes)
    ]
    if any(not row["unchanged"] for row in world_model_evidence):
        raise PolicyProductionError("FROZEN_WORLD_MODEL_STATE_CHANGED_DURING_POLICY_TRAINING")
    if output is not None:
        durable_atomic_json(output / "frozen-world-model-state.json", {
            "schema": "w1-g1-frozen-world-model-state/1.0.0",
            "models": world_model_evidence,
        })
    return {"routes": routes, "policies": policies, "checkpoints": checkpoint_paths,
            "frozen_world_models": frozen_models, "resource_amounts": calls.snapshot(),
            "frozen_world_model_state": world_model_evidence,
            "route_count": len(routes)}


@dataclass
class TaskEpisodeContext:
    method: str
    seed: int | None
    parent: str
    repeat: int
    exogenous_key: str
    max_steps: int
    policy: PriorConditionedPolicy | None
    decision_policy: DecisionPriorPolicy | None
    calls: CallAccounting
    task_preference: tuple[float, float]
    utility_discount: float
    task_component_scale: float
    clock: Any = time

    def reset_environment(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return self.calls.call("environment", f"task_reset:{self.method}:{self.parent}:{self.repeat}",
                               {"resets_upper": 1}, operation, *args, **kwargs)

    def step_environment(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return self.calls.call("environment", f"task_step:{self.method}:{self.parent}:{self.repeat}",
                               {"environment_steps": 1}, operation, *args, **kwargs)


def _task_units(matrix: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    source_rows = tuple(matrix.get("splits", {}).get("task_confirmation", ()))
    repeats = int(matrix.get("repeats", {}).get("task_confirmation", 0))
    if len(source_rows) != 8 or repeats != 3:
        raise PolicyProductionError("TASK_PARENT_REPEAT_SOURCE_CONTRACT_MISMATCH")
    units = []
    seen_parents = set()
    for source in source_rows:
        parent = str(source.get("parent", ""))
        if not parent or parent in seen_parents:
            raise PolicyProductionError("TASK_PARENT_IDENTITY_INVALID")
        seen_parents.add(parent)
        source_exogenous = source.get("exogenous_key")
        if not isinstance(source_exogenous, str) or "|repeat-" not in source_exogenous:
            raise PolicyProductionError("TASK_EXOGENOUS_IDENTITY_MISSING")
        prefix = source_exogenous.rsplit("|repeat-", 1)[0]
        for repeat in range(repeats):
            units.append({**dict(source), "repeat": repeat,
                          "exogenous_key": f"{prefix}|repeat-{repeat}"})
    if len(units) != 24 or len({(row["parent"], row["repeat"]) for row in units}) != 24:
        raise PolicyProductionError("TASK_PARENT_REPEAT_UNIT_COUNT_MISMATCH")
    return tuple(units)


def _nearest_rank_p95(values: Sequence[float]) -> float:
    if not values:
        raise PolicyProductionError("P95_REQUIRES_VALUES")
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def summarize_decision_costs(cost_rows: Sequence[Mapping[str, Any]],
                             units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"by_method": {}, "by_method_parent_repeat": {}}
    for method in TASK_METHODS:
        selected = [row for row in cost_rows if row.get("method") == method]
        non_noop_opportunities = sum(bool(row.get("has_non_noop_candidate", int(row.get("candidate_count", 1)) > 1))
                                     for row in selected)
        world_forward_decisions = sum(int(row.get("world_model_forwards", 0)) > 0 for row in selected)
        eligible_decisions = sum(bool(row.get("g1_cost_eligible", False)) for row in selected)
        if selected:
            result["by_method"][method] = {
                "status": "evaluated", "sample_count": len(selected),
                "cpu_mean_ms": 1000.0 * sum(float(row["cpu_seconds"]) for row in selected) / len(selected),
                "wall_p95_ms": 1000.0 * _nearest_rank_p95([float(row["wall_seconds"]) for row in selected]),
                "noop_selected_count": sum(bool(row.get("noop_selected", row.get("selected_action") == NOOP_ACTION)) for row in selected),
                "world_model_forwards": sum(int(row.get("world_model_forwards", 0)) for row in selected),
                "world_model_sample_evaluations": sum(int(row.get("world_model_sample_evaluations", 0)) for row in selected),
                "non_noop_candidate_decision_count": non_noop_opportunities,
                "world_model_forward_decision_count": world_forward_decisions,
                "g1_cost_eligible_decision_count": eligible_decisions,
                "g1_cost_eligible": bool(method == "G1" and eligible_decisions > 0),
            }
        else:
            result["by_method"][method] = {"status": "not_evaluated", "sample_count": 0,
                                            "cpu_mean_ms": None, "wall_p95_ms": None,
                                            "noop_selected_count": 0, "world_model_forwards": 0,
                                            "world_model_sample_evaluations": 0,
                                            "non_noop_candidate_decision_count": 0,
                                            "world_model_forward_decision_count": 0,
                                            "g1_cost_eligible_decision_count": 0,
                                            "g1_cost_eligible": False}
        for unit in units:
            parent, repeat = str(unit["parent"]), int(unit["repeat"])
            group = [row for row in selected if row.get("parent") == parent and int(row.get("repeat", -1)) == repeat]
            key = f"{method}|{parent}|{repeat}"
            if group:
                result["by_method_parent_repeat"][key] = {
                    "status": "evaluated", "sample_count": len(group),
                    "cpu_mean_ms": 1000.0 * sum(float(row["cpu_seconds"]) for row in group) / len(group),
                    "wall_p95_ms": 1000.0 * _nearest_rank_p95([float(row["wall_seconds"]) for row in group]),
                    "noop_selected_count": sum(bool(row.get("noop_selected", row.get("selected_action") == NOOP_ACTION)) for row in group),
                    "world_model_forwards": sum(int(row.get("world_model_forwards", 0)) for row in group),
                    "world_model_sample_evaluations": sum(int(row.get("world_model_sample_evaluations", 0)) for row in group),
                    "non_noop_candidate_decision_count": sum(bool(row.get("has_non_noop_candidate", int(row.get("candidate_count", 1)) > 1)) for row in group),
                    "world_model_forward_decision_count": sum(int(row.get("world_model_forwards", 0)) > 0 for row in group),
                    "g1_cost_eligible_decision_count": sum(bool(row.get("g1_cost_eligible", False)) for row in group),
                    "g1_cost_eligible": bool(method == "G1" and any(bool(row.get("g1_cost_eligible", False)) for row in group)),
                }
            else:
                result["by_method_parent_repeat"][key] = {
                    "status": "not_evaluated", "sample_count": 0,
                    "cpu_mean_ms": None, "wall_p95_ms": None,
                    "noop_selected_count": 0, "world_model_forwards": 0,
                    "world_model_sample_evaluations": 0,
                    "non_noop_candidate_decision_count": 0,
                    "world_model_forward_decision_count": 0,
                    "g1_cost_eligible_decision_count": 0,
                    "g1_cost_eligible": False,
                }
    return result


def recompute_task_metrics(episodes: Sequence[Mapping[str, Any]],
                           units: Sequence[Mapping[str, Any]], *,
                           task_preference: Sequence[float] = (0.8, 0.2),
                           utility_discount: float = 0.99,
                           task_component_scale: float = 0.5) -> dict[str, Any]:
    """Recompute paired G1 utility contrasts while keeping unknowns explicit."""
    preference = tuple(float(value) for value in task_preference)
    if (len(preference) != 2 or not all(math.isfinite(value) and value >= 0 for value in preference)
            or abs(sum(preference) - 1.0) > 1e-8 or not math.isfinite(utility_discount)
            or not 0.0 <= utility_discount <= 1.0 or not math.isfinite(task_component_scale)
            or task_component_scale < 0):
        raise PolicyProductionError("TASK_UTILITY_CONFIGURATION_INVALID")
    comparisons = {"G0": "g1_minus_g0", "T": "g1_minus_t", "H": "g1_minus_h"}

    def mean_known(values: Sequence[float | None]) -> tuple[float | None, int]:
        known = [float(value) for value in values if value is not None]
        if any(not math.isfinite(value) for value in known):
            raise PolicyProductionError("TASK_METRIC_VALUE_NONFINITE")
        return (sum(known) / len(known), len(known)) if known else (None, 0)

    unit_rows: list[dict[str, Any]] = []
    expected_routes = {(method, seed) for method in POLICY_METHODS for seed in POLICY_SEEDS}
    for unit in units:
        parent, repeat = str(unit["parent"]), int(unit["repeat"])
        selected = [row for row in episodes
                    if str(row.get("parent")) == parent and int(row.get("repeat", -1)) == repeat]
        route_rows: dict[tuple[str, int | None], Mapping[str, Any]] = {}
        for row in selected:
            method = str(row.get("method"))
            seed = None if method == "H" else int(row.get("seed", -1))
            key = (method, seed)
            if key in route_rows:
                raise PolicyProductionError(f"TASK_METRIC_DUPLICATE_ROUTE:{parent}:{repeat}:{key}")
            route_rows[key] = row
        if set(route_rows) != expected_routes | {("H", None)}:
            raise PolicyProductionError(f"TASK_METRIC_ROUTE_COUNT_MISMATCH:{parent}:{repeat}")

        utility_by_route: dict[tuple[str, int | None], float | None] = {}
        opportunity_counts = {method: 0 for method in TASK_METHODS}
        unknown_outcome_counts = {method: 0 for method in TASK_METHODS}
        unknown_utility_counts = {method: 0 for method in TASK_METHODS}
        utility_recomputation_errors = []
        for key, row in route_rows.items():
            method = key[0]
            opportunity_counts[method] += int(not bool(row.get("opportunity", False)))
            unknown_outcome_counts[method] += int(not bool(row.get("outcome_valid", True)))
            recorded_utility = row.get("utility")
            utility_valid = bool(row.get("utility_valid", recorded_utility is not None))
            if utility_valid:
                if recorded_utility is None or not math.isfinite(float(recorded_utility)):
                    raise PolicyProductionError(f"TASK_UTILITY_INVALID:{method}:{parent}:{repeat}")
                rewards = row.get("vector_rewards")
                if rewards is None:
                    rewards = [item.get("vector_reward") for item in row.get("actions", ())]
                steps = int(row.get("steps", -1))
                if not isinstance(rewards, Sequence) or len(rewards) != steps:
                    raise PolicyProductionError(f"TASK_VECTOR_REWARD_SEQUENCE_INVALID:{method}:{parent}:{repeat}")
                reconstructed = 0.0
                for step, reward in enumerate(rewards):
                    if not isinstance(reward, Sequence) or isinstance(reward, (str, bytes)) or len(reward) != 2:
                        raise PolicyProductionError(f"TASK_VECTOR_REWARD_INVALID:{method}:{parent}:{repeat}:{step}")
                    task_reward, energy_reward = float(reward[0]), float(reward[1])
                    if not math.isfinite(task_reward) or not math.isfinite(energy_reward):
                        raise PolicyProductionError(f"TASK_VECTOR_REWARD_NONFINITE:{method}:{parent}:{repeat}:{step}")
                    reconstructed += utility_discount ** step * (
                        task_component_scale * preference[0] * task_reward
                        + preference[1] * energy_reward
                    )
                error = abs(reconstructed - float(recorded_utility))
                if error > 1e-8:
                    raise PolicyProductionError(f"TASK_REWARD_UTILITY_MISMATCH:{method}:{parent}:{repeat}")
                utility_by_route[key] = reconstructed
                utility_recomputation_errors.append(error)
            else:
                if recorded_utility is not None:
                    raise PolicyProductionError(f"TASK_UNKNOWN_UTILITY_HAS_VALUE:{method}:{parent}:{repeat}")
                utility_by_route[key] = None
                unknown_utility_counts[method] += 1

        seed_rows = []
        for seed in POLICY_SEEDS:
            g1_value = utility_by_route[("G1", seed)]
            seed_comparisons = {}
            for baseline, name in comparisons.items():
                baseline_seed = seed if baseline != "H" else None
                baseline_value = utility_by_route[(baseline, baseline_seed)]
                seed_comparisons[name] = (
                    None if g1_value is None or baseline_value is None
                    else g1_value - baseline_value
                )
            seed_rows.append({
                "seed": seed,
                "utilities": {
                    method: utility_by_route[(method, seed)] for method in POLICY_METHODS
                } | {"H": utility_by_route[("H", None)]},
                "comparisons": seed_comparisons,
            })
        utilities = {}
        for method in POLICY_METHODS:
            values = [utility_by_route[(method, seed)] for seed in POLICY_SEEDS]
            utilities[method], _ = mean_known(values)
        utilities["H"] = utility_by_route[("H", None)]
        comparison_rows = {}
        for name in comparisons.values():
            value, count = mean_known([row["comparisons"][name] for row in seed_rows])
            comparison_rows[name] = {
                "mean": value, "valid_seed_count": count,
                "unknown_seed_count": len(POLICY_SEEDS) - count,
            }
        known_utility_count = sum(value is not None for value in utility_by_route.values())
        unit_rows.append({
            "parent": parent, "repeat": repeat,
            "status": "evaluated" if known_utility_count else "utility_unknown",
            "utilities": utilities, "per_seed": seed_rows,
            "comparisons": comparison_rows,
            "no_opportunity_episode_count_by_method": opportunity_counts,
            "unknown_outcome_count_by_method": unknown_outcome_counts,
            "unknown_utility_count_by_method": unknown_utility_counts,
            "known_utility_episode_count": known_utility_count,
            "utility_recomputation_max_abs_error": max(utility_recomputation_errors, default=0.0),
        })

    parent_rows: list[dict[str, Any]] = []
    for parent in sorted({str(unit["parent"]) for unit in units}):
        parent_units = [row for row in unit_rows if row["parent"] == parent]
        utilities: dict[str, float | None] = {}
        utility_counts = {}
        for method in TASK_METHODS:
            utilities[method], utility_counts[method] = mean_known(
                [row["utilities"][method] for row in parent_units]
            )
        parent_comparisons = {}
        for name in comparisons.values():
            value, count = mean_known([
                row["comparisons"][name]["mean"] for row in parent_units
            ])
            parent_comparisons[name] = {
                "mean": value, "valid_repeat_count": count,
                "unknown_repeat_count": len(parent_units) - count,
            }
        per_seed = {}
        for seed in POLICY_SEEDS:
            per_seed[str(seed)] = {}
            for name in comparisons.values():
                value, count = mean_known([
                    seed_row["comparisons"][name]
                    for row in parent_units for seed_row in row["per_seed"]
                    if seed_row["seed"] == seed
                ])
                per_seed[str(seed)][name] = {
                    "mean": value, "valid_repeat_count": count,
                    "unknown_repeat_count": len(parent_units) - count,
                }
        parent_rows.append({
            "parent": parent,
            "status": "evaluated" if any(item["mean"] is not None for item in parent_comparisons.values()) else "utility_unknown",
            "utilities": utilities, "utility_episode_count_by_method": utility_counts,
            "comparisons": parent_comparisons, "per_seed_comparisons": per_seed,
            "valid_repeat_count": max(
                (item["valid_repeat_count"] for item in parent_comparisons.values()), default=0,
            ),
            "no_opportunity_episode_count_by_method": {
                method: sum(row["no_opportunity_episode_count_by_method"][method] for row in parent_units)
                for method in TASK_METHODS
            },
            "unknown_outcome_count_by_method": {
                method: sum(row["unknown_outcome_count_by_method"][method] for row in parent_units)
                for method in TASK_METHODS
            },
            "unknown_utility_count_by_method": {
                method: sum(row["unknown_utility_count_by_method"][method] for row in parent_units)
                for method in TASK_METHODS
            },
        })

    macro_utility = {}
    for method in TASK_METHODS:
        macro_utility[method], _ = mean_known([row["utilities"][method] for row in parent_rows])
    macro_comparisons = {}
    for name in comparisons.values():
        value, count = mean_known([row["comparisons"][name]["mean"] for row in parent_rows])
        macro_comparisons[name] = {"mean": value, "valid_parent_count": count,
                                   "unknown_parent_count": len(parent_rows) - count}
    seed_macro = {}
    for seed in POLICY_SEEDS:
        seed_macro[str(seed)] = {}
        for name in comparisons.values():
            value, count = mean_known([
                row["per_seed_comparisons"][str(seed)][name]["mean"] for row in parent_rows
            ])
            seed_macro[str(seed)][name] = {
                "mean": value, "valid_parent_count": count,
                "unknown_parent_count": len(parent_rows) - count,
            }
    evaluated_parents = [row for row in parent_rows if row["status"] == "evaluated"]
    return {
        "valid_parent_count": len(evaluated_parents),
        "valid_unit_count": sum(row["status"] == "evaluated" for row in unit_rows),
        "episode_count": len(episodes),
        "unknown_outcome_episode_count": sum(
            not bool(row.get("outcome_valid", True)) for row in episodes
        ),
        "no_opportunity_episode_count": sum(
            not bool(row.get("opportunity", False)) for row in episodes
        ),
        "unknown_utility_episode_count": sum(
            row.get("utility") is None for row in episodes
        ),
        "utility_recomputation_max_abs_error": max(
            (row["utility_recomputation_max_abs_error"] for row in unit_rows), default=0.0,
        ),
        "parent_macro_utility": macro_utility,
        "macro_comparisons": macro_comparisons,
        "per_seed_macro_comparisons": seed_macro,
        "g1_minus_g0_macro_utility": macro_comparisons["g1_minus_g0"]["mean"],
        "g1_minus_t_macro_utility": macro_comparisons["g1_minus_t"]["mean"],
        "g1_minus_h_macro_utility": macro_comparisons["g1_minus_h"]["mean"],
        "parents_g1_no_worse_than_t": sum(
            row["comparisons"]["g1_minus_t"]["mean"] >= 0.0
            for row in evaluated_parents
            if row["comparisons"]["g1_minus_t"]["mean"] is not None
        ),
        "per_parent": parent_rows,
        "per_parent_repeat": unit_rows,
        "aggregation": "mean policy seeds within parent-repeat, mean known repeats within parent, macro mean across parents; unavailable values remain null",
    }

def evaluate_task_confirmation(*, matrix: Mapping[str, Any], request: Mapping[str, Any], ledger: Any,
                               boundary: BottomBoundary | None,
                               policy_loader: Callable[[str, int, str], nn.Module],
                               world_model_loader: Callable[[str, int], Any],
                               episode_runner: Callable[[TaskEpisodeContext], Mapping[str, Any]],
                               input_builder: Callable[[Any], Mapping[str, Any]],
                               transparent_scorer: Callable[[Any, Mapping[str, Any]], Any] | None,
                               world_predictor: Callable[[str, Any, Any, torch.Tensor, torch.Tensor], Any] | None,
                               checkpoints: Mapping[tuple[str, int], str],
                               output_dir: str | Path | None = None, clock: Any = time,
                               selector: Callable[[Categorical], int] | None = None) -> dict[str, Any]:
    """Run 216 paired policy episodes and 24 shared deterministic H episodes."""
    schedule = policy_schedule(matrix, request)
    units = _task_units(matrix)
    max_steps = int(matrix.get("task_episode_max_steps", 0))
    expected_count = len(units) * (len(POLICY_METHODS) * len(schedule.seeds) + 1)
    if max_steps != 18 or int(matrix.get("task_episode_count", -1)) != expected_count:
        raise PolicyProductionError("TASK_EPISODE_CONTRACT_MISMATCH")
    calls = CallAccounting(ledger, boundary)
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    policies: dict[tuple[str, int], PriorConditionedPolicy] = {}
    for method in POLICY_METHODS:
        for seed in schedule.seeds:
            checkpoint = checkpoints.get((method, seed))
            if not checkpoint:
                raise PolicyProductionError(f"POLICY_CHECKPOINT_MISSING:{method}:{seed}")
            base = calls.call(
                "checkpoint", f"load_policy_checkpoint:{method}:{seed}",
                {"checkpoint_loads": 1, "model_initializations_or_loads": 1}, policy_loader,
                method, seed, str(checkpoint),
            )
            if not isinstance(base, nn.Module):
                raise PolicyProductionError("POLICY_LOADER_MUST_RETURN_MODULE")
            policies[(method, seed)] = PriorConditionedPolicy(base)
    task_world_models: dict[tuple[str, int], Any] = {}
    task_world_model_initial_hashes: dict[tuple[str, int], str] = {}
    for variant in WORLD_MODEL_VARIANTS:
        for world_seed in WORLD_MODEL_SEEDS:
            model = calls.call(
                "checkpoint", f"load_task_world_model:{variant}:{world_seed}",
                {"checkpoint_loads": 1, "model_initializations_or_loads": 1},
                world_model_loader, variant, world_seed,
            )
            if not isinstance(model, nn.Module):
                raise PolicyProductionError("WORLD_MODEL_LOADER_MUST_RETURN_MODULE")
            model.eval()
            model.requires_grad_(False)
            if model.training or any(parameter.requires_grad for parameter in model.parameters()):
                raise PolicyProductionError("FROZEN_WORLD_MODEL_STATE_INVALID")
            task_world_models[(variant, world_seed)] = model
            task_world_model_initial_hashes[(variant, world_seed)] = state_dict_sha256(model.state_dict())
    episodes: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    task_count_before = calls.amounts["task_episodes"]

    def run_episode(unit: Mapping[str, Any], method: str, seed: int | None,
                    policy: PriorConditionedPolicy | None, models: Sequence[Any]) -> None:
        parent, repeat = str(unit["parent"]), int(unit["repeat"])
        exogenous_key = str(unit["exogenous_key"])
        decision = None
        if method in POLICY_METHODS:
            decision = DecisionPriorPolicy(
                method=method, seed=int(seed), policy=policy,
                prior_configuration=schedule.prior_configuration,
                input_builder=input_builder, transparent_scorer=transparent_scorer,
                world_models=models, world_predictor=world_predictor,
                calls=calls, parent=parent, repeat=repeat, clock=clock, selector=selector,
            )
        before = calls.snapshot()
        calls.charge("task_episode", {"task_episodes": 1})
        context = TaskEpisodeContext(method, seed, parent, repeat, exogenous_key,
                                     max_steps, policy, decision, calls,
                                     schedule.task_preference, schedule.utility_discount,
                                     schedule.task_component_scale, clock)
        result = episode_runner(context)
        if not isinstance(result, Mapping):
            raise PolicyProductionError("TASK_EPISODE_RESULT_INVALID")
        steps = int(result.get("steps", -1))
        if steps < 0 or steps > max_steps:
            raise PolicyProductionError("TASK_EPISODE_STEP_LIMIT_VIOLATION")
        after = calls.snapshot()
        if after.get("environment_steps", 0) - before.get("environment_steps", 0) != steps:
            raise PolicyProductionError("TASK_EPISODE_STEP_ACCOUNTING_MISMATCH")
        if after.get("resets_upper", 0) - before.get("resets_upper", 0) != 1:
            raise PolicyProductionError("TASK_EPISODE_RESET_ACCOUNTING_MISMATCH")
        opportunity = bool(result.get("opportunity"))
        utility = result.get("utility")
        utility_valid = bool(result.get("utility_valid", utility is not None))
        if utility is not None:
            if not math.isfinite(float(utility)):
                raise PolicyProductionError("TASK_EPISODE_UTILITY_INVALID")
            utility = float(utility)
        elif utility_valid:
            raise PolicyProductionError("TASK_EPISODE_UTILITY_INVALID")
        vector_rewards = result.get("vector_rewards")
        if vector_rewards is None:
            vector_rewards = [item.get("vector_reward") for item in result.get("actions", ())
                              if isinstance(item, Mapping)]
        if not isinstance(vector_rewards, Sequence) or len(vector_rewards) != steps:
            raise PolicyProductionError("TASK_EPISODE_VECTOR_REWARDS_MISSING")
        normalized_rewards = []
        for index, reward in enumerate(vector_rewards):
            if not isinstance(reward, Sequence) or isinstance(reward, (str, bytes)) or len(reward) != 2:
                raise PolicyProductionError(f"TASK_EPISODE_VECTOR_REWARD_INVALID:{index}")
            values = [float(reward[0]), float(reward[1])]
            if not all(math.isfinite(value) for value in values):
                raise PolicyProductionError(f"TASK_EPISODE_VECTOR_REWARD_NONFINITE:{index}")
            normalized_rewards.append(values)
        terminated = result.get("terminated") is True
        truncated = result.get("truncated") is True
        if terminated == truncated:
            raise PolicyProductionError("TASK_EPISODE_COMPLETION_FLAGS_INVALID")
        if utility_valid:
            reconstructed_utility = sum(
                schedule.utility_discount ** index * (
                    schedule.task_component_scale * schedule.task_preference[0] * reward[0]
                    + schedule.task_preference[1] * reward[1]
                ) for index, reward in enumerate(normalized_rewards)
            )
            if utility is None or abs(reconstructed_utility - utility) > 1e-8:
                raise PolicyProductionError("TASK_EPISODE_REWARD_UTILITY_MISMATCH")
            utility = reconstructed_utility
        outcome_valid = bool(result.get("outcome_valid", True))
        lifecycle_outcomes = result.get("lifecycle_outcomes")
        if not isinstance(lifecycle_outcomes, Sequence) or isinstance(lifecycle_outcomes, (str, bytes)) or not lifecycle_outcomes:
            raise PolicyProductionError("TASK_LIFECYCLE_OUTCOMES_MISSING")
        for label in lifecycle_outcomes:
            if not isinstance(label, Mapping):
                raise PolicyProductionError("TASK_LIFECYCLE_OUTCOME_INVALID")
            for name in ("physical_on_time_completion", "host_confirmation"):
                field = label.get(name)
                if not isinstance(field, Mapping) or type(field.get("valid")) is not bool:
                    raise PolicyProductionError(f"TASK_LIFECYCLE_LABEL_INVALID:{name}")
                if field["valid"] and type(field.get("value")) is not bool:
                    raise PolicyProductionError(f"TASK_LIFECYCLE_LABEL_VALUE_INVALID:{name}")
                if not field["valid"] and field.get("value") is not None:
                    raise PolicyProductionError(f"TASK_UNKNOWN_LIFECYCLE_LABEL_HAS_VALUE:{name}")
        if outcome_valid != all(label["physical_on_time_completion"]["valid"] for label in lifecycle_outcomes):
            raise PolicyProductionError("TASK_LIFECYCLE_VALIDITY_MISMATCH")
        preference = tuple(float(value) for value in result.get("preference", ()))
        if preference != schedule.task_preference:
            raise PolicyProductionError("TASK_EPISODE_PREFERENCE_MISMATCH")
        if decision is not None:
            if decision.decision_count != steps or len(decision.cost_rows) != steps:
                raise PolicyProductionError("TASK_POLICY_DECISION_COST_COVERAGE_MISMATCH")
            cost_rows.extend(decision.cost_rows)
            if output is not None:
                for row in decision.cost_rows:
                    durable_append_jsonl(output / "decision-costs.jsonl", row)
        else:
            raw_costs = result.get("decision_costs", ())
            if not isinstance(raw_costs, Sequence) or len(raw_costs) != steps:
                raise PolicyProductionError("TASK_DECISION_COSTS_INVALID")
            for cost in raw_costs:
                if (not isinstance(cost, Mapping) or cost.get("method") != "H"
                        or cost.get("parent") != parent or int(cost.get("repeat", -1)) != repeat
                        or cost.get("seed") is not None):
                    raise PolicyProductionError("HUNGARIAN_DECISION_COST_INVALID")
                cost_rows.append(dict(cost))
                if output is not None:
                    durable_append_jsonl(output / "decision-costs.jsonl", dict(cost))
        episode = {
            "method": method, "seed": seed, "parent": parent, "repeat": repeat,
            "scenario_sha256": unit.get("scenario_sha256"),
            "structural_sha256": unit.get("structural_sha256"),
            "exogenous_key": exogenous_key, "opportunity": opportunity,
            "outcome_valid": outcome_valid,
            "outcome_unknown_reason": result.get("outcome_unknown_reason"),
            "lifecycle_outcomes": list(lifecycle_outcomes),
            "utility_valid": utility_valid,
            "utility": utility, "steps": steps,
            "actions": list(result.get("actions", ())),
            "vector_rewards": normalized_rewards,
            "preference": list(preference),
            "environment_config_sha256": result.get("environment_config_sha256"),
            "scenario_id": result.get("scenario_id", parent),
            "physical_rate": result.get("physical_rate"),
            "host_rate": result.get("host_rate"),
            "counts": result.get("counts"),
            "energy_used": result.get("energy_used"),
            "terminated": terminated,
            "truncated": truncated,
            "decision_count": steps,
            "decision_cost_count": steps,
            "argmax_changed_count": 0 if decision is None else decision.argmax_changed_count,
            "sampled_action_changed_count": 0 if decision is None else decision.sampled_action_changed_count,
            "sampled_action_comparison_count": 0 if decision is None else decision.sampled_action_comparison_count,
            "sampled_action_comparison_status": (
                "not_evaluated" if decision is None
                else decision.decision_summary()["sampled_action_comparison_status"]
            ),
            "policy_sampling_key": result.get("policy_sampling_key"),
            "policy_sampling_seed": result.get("policy_sampling_seed"),
        }
        episodes.append(episode)
        if output is not None:
            for index, trace in enumerate(episode["actions"]):
                durable_append_jsonl(output / "task-decision-traces.jsonl", {
                    "method": method, "seed": seed, "parent": parent,
                    "repeat": repeat, "exogenous_key": exogenous_key,
                    "decision_step": index, "scenario_sha256": unit.get("scenario_sha256"),
                    "structural_sha256": unit.get("structural_sha256"),
                    "environment_config_sha256": episode["environment_config_sha256"],
                    **dict(trace),
                })
            durable_append_jsonl(output / "task-confirmation.jsonl", episode)

    for unit in units:
        for method in POLICY_METHODS:
            for seed in schedule.seeds:
                if method in WORLD_MODEL_VARIANTS:
                    world_seed = WORLD_MODEL_SEEDS[schedule.seeds.index(seed)]
                    models = (task_world_models[(method, world_seed)],)
                else:
                    models = ()
                run_episode(unit, method, seed, policies[(method, seed)], models)
        run_episode(unit, "H", None, None, ())
    expected_keys = {
        (method, seed, str(unit["parent"]), int(unit["repeat"]))
        for method in POLICY_METHODS for seed in schedule.seeds for unit in units
    }
    observed_keys = {(row["method"], row["seed"], row["parent"], row["repeat"])
                     for row in episodes if row["method"] in POLICY_METHODS}
    if expected_keys != observed_keys or len(episodes) != expected_count:
        raise PolicyProductionError("TASK_POLICY_ROUTE_SET_MISMATCH")
    if sum(row["method"] == "H" for row in episodes) != len(units):
        raise PolicyProductionError("HUNGARIAN_SHARED_EPISODE_COUNT_MISMATCH")
    if calls.amounts["task_episodes"] - task_count_before != expected_count:
        raise PolicyProductionError("TASK_EPISODE_LEDGER_COUNT_MISMATCH")
    task_world_model_evidence = [
        {"variant": variant, "seed": seed,
         "initial_state_sha256": task_world_model_initial_hashes[(variant, seed)],
         "final_state_sha256": state_dict_sha256(task_world_models[(variant, seed)].state_dict()),
         "unchanged": task_world_model_initial_hashes[(variant, seed)]
         == state_dict_sha256(task_world_models[(variant, seed)].state_dict())}
        for variant, seed in sorted(task_world_models)
    ]
    if any(not row["unchanged"] for row in task_world_model_evidence):
        raise PolicyProductionError("FROZEN_WORLD_MODEL_STATE_CHANGED_DURING_TASK_EVALUATION")
    summaries = summarize_decision_costs(cost_rows, units)
    metrics = recompute_task_metrics(
        episodes, units, task_preference=schedule.task_preference,
        utility_discount=schedule.utility_discount,
        task_component_scale=schedule.task_component_scale,
    )
    calls.charge("task_comparison", {"task_comparison_calls": 1})
    if output is not None:
        durable_atomic_json(output / "frozen-task-world-model-state.json", {
            "schema": "w1-g1-frozen-task-world-model-state/1.0.0",
            "models": task_world_model_evidence,
        })
        durable_append_jsonl(output / "task-cost-summary.jsonl", summaries)
        durable_atomic_json(output / "task-metrics.json", metrics)
    return {
        "task_episodes": len(episodes), "policy_episodes": len(episodes) - 24,
        "shared_hungarian_episodes": 24, "episodes": episodes,
        "decision_costs": cost_rows, "cost_summary": summaries,
        "metrics": metrics,
        "resource_amounts": calls.snapshot(),
    }


__all__ = [
    "ACTION_COUNT", "NOOP_ACTION", "POLICY_METHODS", "TASK_METHODS", "POLICY_SEEDS",
    "WORLD_MODEL_SEEDS", "WORLD_MODEL_VARIANTS", "CallAccounting",
    "DecisionPriorPolicy", "PolicyDecision", "PolicyProductionError",
    "PolicyRoute", "PolicySchedule", "PolicyTrainingContext",
    "PriorConditionedPolicy", "TaskEpisodeContext", "augment_candidate_features",
    "evaluate_task_confirmation", "policy_schedule", "recompute_task_metrics",
    "summarize_decision_costs", "transparent_utility_components",
    "train_policy_routes",
]



