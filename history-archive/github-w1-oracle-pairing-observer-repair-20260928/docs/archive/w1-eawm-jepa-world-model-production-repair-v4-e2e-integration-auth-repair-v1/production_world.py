"""Production G1/G2 world-model routes and candidate prediction tracing."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
import torch
from public_event_targets import EVENT_NAMES
from w1_graph_jepa import ACTION_COUNT, W1GraphJEPA, W1JEPAConfig
from w1_training import WorldModelBatch, train_world_model

WINDOW_SCHEMA = "w1-world-model-complete-window/1.0.0"
TRACE_SCHEMA = "w1-world-model-prediction-trace/1.0.0"
VARIANTS = ("G1", "G2")
OUTCOME_DIM, EVENT_DIM = 6, len(EVENT_NAMES)
ENERGY_RESIDUAL_INDEX, TASK_RESIDUAL_INDEX = 3, 4
PREDICTION_PREFERENCE = (0.8, 0.2)
TASK_COMPONENT_SCALE = 0.5

class WorldModelContractError(RuntimeError):
    """Persisted data or execution violates the frozen world-model contract."""

class BottomBoundary(Protocol):
    def model(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...
    def optimizer(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...
    def checkpoint(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...

@dataclass(frozen=True)
class Window:
    split: str
    parent: str
    repeat: int
    window_id: str
    status: str
    continuation_id: str
    input_hash: str
    scenario_sha256: str
    raw: Mapping[str, Any]
    @property
    def candidate_count(self) -> int:
        return len(self.raw.get("candidate_ids", ())) if self.status == "complete" else 0

@dataclass(frozen=True)
class TrainingRun:
    routes: tuple[Mapping[str, Any], ...]
    models: Mapping[tuple[str, int], Any]

@dataclass(frozen=True)
class PredictionRun:
    trace_path: Path
    metrics_path: Path
    metrics: Mapping[str, Any]

class _Operation:
    def __init__(self, name: str, fn: Callable[..., Any]):
        self.name, self.fn = name, fn
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

def _account(ledger: Any, boundary: Any, kind: str, name: str, amounts: Mapping[str, int],
             fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return ledger.call(name, dict(amounts), getattr(boundary, kind), _Operation(name, fn), *args, **kwargs)

class _AccountedModel:
    def __init__(self, model: Any, ledger: Any, boundary: Any):
        self._model, self._ledger, self._boundary = model, ledger, boundary
    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        nodes = args[0] if args else kwargs.get("nodes")
        return _account(self._ledger, self._boundary, "model", "world_model.forward",
                        {"world_batch_forwards": 1, "world_sample_evaluations": _node_rows(nodes)}, self._model, *args, **kwargs)
    def train(self, mode: bool = True) -> "_AccountedModel":
        _account(self._ledger, self._boundary, "model", "world_model.train_mode", {}, self._model.train, mode)
        return self
    def eval(self) -> "_AccountedModel":
        _account(self._ledger, self._boundary, "model", "world_model.eval_mode", {}, self._model.eval)
        return self
    def target(self, nodes: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return _account(self._ledger, self._boundary, "model", "world_model.target_forward",
                        {"world_batch_forwards": 1, "world_sample_evaluations": _node_rows(nodes)}, self._model.target, nodes)
    def update_target(self) -> None:
        _account(self._ledger, self._boundary, "model", "world_model.ema_update", {}, self._model.update_target)
    def state_dict(self) -> Mapping[str, torch.Tensor]:
        return _account(self._ledger, self._boundary, "model", "world_model.state_dict", {}, self._model.state_dict)
    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> Any:
        return _account(self._ledger, self._boundary, "model", "world_model.load_state_dict", {}, self._model.load_state_dict, state)

class _AccountedOptimizer:
    def __init__(self, optimizer: Any, ledger: Any, boundary: Any, parameters: Sequence[torch.nn.Parameter], clip_norm: float):
        self._optimizer, self._ledger, self._boundary = optimizer, ledger, boundary
        self._parameters, self._clip_norm = tuple(parameters), float(clip_norm)
    def __getattr__(self, name: str) -> Any:
        return getattr(self._optimizer, name)
    def zero_grad(self, *args: Any, **kwargs: Any) -> Any:
        return _account(self._ledger, self._boundary, "optimizer", "world_optimizer.zero_grad", {}, self._optimizer.zero_grad, *args, **kwargs)
    def step(self, *args: Any, **kwargs: Any) -> Any:
        _account(self._ledger, self._boundary, "optimizer", "world_optimizer.gradient_clip", {},
                 torch.nn.utils.clip_grad_norm_, self._parameters, self._clip_norm)
        return _account(self._ledger, self._boundary, "optimizer", "world_optimizer.step",
                        {"world_optimizer_updates": 1}, self._optimizer.step, *args, **kwargs)

def _fail(message: str) -> None:
    raise WorldModelContractError(message)

def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        _fail(f"{name}:BOOLEAN_IS_NOT_NUMERIC")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WorldModelContractError(f"{name}:NOT_NUMERIC") from exc
    if not math.isfinite(result):
        _fail(f"{name}:NONFINITE")
    return result

def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        _fail(f"{name}:INVALID_SHA256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise WorldModelContractError(f"{name}:INVALID_SHA256") from exc
    return value.lower()

def _tensor(value: Any, name: str, unknown: bool = False) -> torch.Tensor:
    if unknown:
        def convert(item: Any) -> Any:
            if isinstance(item, list):
                return [convert(child) for child in item]
            return math.nan if item is None else _number(item, name)
        value = convert(value)
    try:
        result = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise WorldModelContractError(f"{name}:INVALID_TENSOR") from exc
    if not unknown and not bool(torch.isfinite(result).all()):
        _fail(f"{name}:NONFINITE")
    return result

def _mask(value: Any, name: str) -> torch.Tensor:
    try:
        result = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise WorldModelContractError(f"{name}:INVALID_MASK") from exc
    if result.dtype != torch.bool and not bool(((result == 0) | (result == 1)).all()):
        _fail(f"{name}:MASK_NOT_BOOLEAN")
    return result.bool()

def load_complete_windows(path: Path | str) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        _fail(f"WINDOW_FILE_MISSING:{source}")
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                _fail(f"WINDOW_FILE_BLANK_LINE:{line_no}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WorldModelContractError(f"WINDOW_FILE_INVALID_JSON:{line_no}") from exc
            if not isinstance(row, dict):
                _fail(f"WINDOW_RECORD_NOT_OBJECT:{line_no}")
            rows.append(row)
    if not rows:
        _fail("WINDOW_FILE_EMPTY")
    return rows

def _expected_keys(matrix: Mapping[str, Any], split: str) -> dict[tuple[str, int], Mapping[str, Any]]:
    if split not in ("train", "model_selection", "prediction_confirmation"):
        _fail(f"UNSUPPORTED_WORLD_MODEL_SPLIT:{split}")
    rows = matrix.get("splits", {}).get(split)
    repeats = int(matrix.get("repeats", {}).get(split, 0))
    if not isinstance(rows, list) or repeats <= 0:
        _fail(f"MATRIX_SPLIT_CONTRACT_MISSING:{split}")
    expected = {}
    for row in rows:
        parent = str(row.get("parent", ""))
        if not parent:
            _fail(f"MATRIX_PARENT_MISSING:{split}")
        for repeat in range(repeats):
            key = (parent, repeat)
            if key in expected:
                _fail(f"MATRIX_PARENT_REPEAT_DUPLICATE:{split}:{key}")
            expected[key] = row
    return expected

def validate_windows(records: Sequence[Mapping[str, Any]], matrix: Mapping[str, Any], split: str) -> list[Window]:
    expected = _expected_keys(matrix, split)
    found: dict[tuple[str, int], Window] = {}
    continuation = str(matrix.get("candidate_continuation_id", ""))
    for raw in records:
        if raw.get("schema") != WINDOW_SCHEMA or raw.get("split") != split:
            _fail("WINDOW_SCHEMA_OR_SPLIT_MISMATCH")
        parent = str(raw.get("parent", ""))
        try:
            repeat = int(raw["repeat"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise WorldModelContractError("WINDOW_REPEAT_MISSING_OR_INVALID") from exc
        key = (parent, repeat)
        if key not in expected or key in found:
            _fail(f"WINDOW_PARENT_REPEAT_UNPLANNED_OR_DUPLICATE:{split}:{key}")
        scenario = _digest(raw.get("scenario_sha256"), "scenario_sha256")
        if scenario != str(expected[key].get("scenario_sha256", "")).lower():
            _fail(f"SCENARIO_IDENTITY_MISMATCH:{split}:{parent}")
        status = str(raw.get("status", ""))
        if status not in ("complete", "no_opportunity"):
            _fail(f"WINDOW_STATUS_INVALID:{status}")
        window_id = str(raw.get("window_id", ""))
        if not window_id:
            _fail(f"WINDOW_ID_MISSING:{split}:{parent}")
        continuation_id = str(raw.get("continuation_id", ""))
        if status == "complete" and continuation_id != continuation:
            _fail(f"CONTINUATION_IDENTITY_MISMATCH:{split}:{parent}")
        input_hash = raw.get("input_hash", "")
        if status == "complete":
            input_hash = _digest(input_hash, "input_hash")
        elif input_hash:
            input_hash = _digest(input_hash, "input_hash")
        else:
            input_hash = ""
        window = Window(split, parent, repeat, window_id, status, continuation_id,
                        input_hash, scenario, raw)
        if status == "complete":
            _validate_payload(window, matrix)
        found[key] = window
    missing = set(expected) - set(found)
    if missing:
        _fail(f"WINDOW_COVERAGE_RECORDS_MISSING:{split}:{sorted(missing)}")
    order = list(expected)
    return sorted(found.values(), key=lambda item: order.index((item.parent, item.repeat)))

def _validate_payload(window: Window, matrix: Mapping[str, Any]) -> None:
    raw, count = window.raw, window.candidate_count
    if count <= 0 or count > int(matrix.get("maximum_legal_candidates_per_window", 0)):
        _fail(f"CANDIDATE_COUNT_INVALID:{window.parent}:{window.repeat}")
    ids, actions, scores = raw.get("candidate_ids"), raw.get("actions"), raw.get("transparent_scores")
    if not isinstance(ids, list) or len(set(map(str, ids))) != count:
        _fail("CANDIDATE_IDENTITIES_MISSING_OR_DUPLICATE")
    if not isinstance(actions, list) or len(actions) != count:
        _fail("ACTION_VECTOR_SHAPE_MISMATCH")
    actions = [int(item) for item in actions]
    if len(set(actions)) != count or any(action < 0 or action >= ACTION_COUNT for action in actions):
        _fail("ACTION_VECTOR_INVALID")
    if not isinstance(scores, list) or len(scores) != count:
        _fail("TRANSPARENT_SCORE_SHAPE_MISMATCH")
    for item in scores:
        _number(item, "transparent_score")
    components = raw.get("transparent_components")
    if not isinstance(components, list) or len(components) != count:
        _fail("TRANSPARENT_COMPONENT_SHAPE_MISMATCH")
    for index, component in enumerate(components):
        if not isinstance(component, list) or len(component) != 2:
            _fail("TRANSPARENT_COMPONENT_SHAPE_MISMATCH")
        task, energy = (_number(value, "transparent_component") for value in component)
        derived = TASK_COMPONENT_SCALE * PREDICTION_PREFERENCE[0] * task + PREDICTION_PREFERENCE[1] * energy
        if not math.isclose(derived, float(scores[index]), rel_tol=0.0, abs_tol=1e-8):
            _fail("TRANSPARENT_SCORE_COMPONENT_MISMATCH")
    true_utility = raw.get("true_utility")
    if not isinstance(true_utility, list) or len(true_utility) != count:
        _fail("TRUE_UTILITY_LABEL_SHAPE_MISMATCH")
    for item in true_utility:
        _number(item, "true_utility")
    required = ("current_nodes", "history", "relations", "target_nodes", "jepa_valid",
                "state_target", "state_valid", "outcome_target", "outcome_valid", "event_labels", "event_valid")
    if any(name not in raw for name in required):
        _fail(f"WINDOW_REQUIRED_TARGET_MISSING:{window.parent}:{window.repeat}")
    arch = matrix.get("world_model_architecture", {})
    if not isinstance(raw["current_nodes"], dict) or not raw["current_nodes"]:
        _fail("CURRENT_GRAPH_MISSING")
    for name, value in raw["current_nodes"].items():
        node = _tensor(value, f"current_nodes.{name}")
        if node.ndim == 3 and node.shape[0] == 1:
            node = node[0]
        if node.ndim != 2 or node.shape[-1] != int(arch["node_dim"]) or node.shape[0] <= 0:
            _fail(f"CURRENT_GRAPH_SHAPE_INVALID:{name}")
    history = _tensor(raw["history"], "history")
    if history.numel() != int(arch["history_dim"]):
        _fail("HISTORY_SHAPE_MISMATCH")
    relation = _tensor(raw["relations"], "relations")
    if tuple(relation.shape) != (count, int(arch["relation_dim"])):
        _fail("RELATION_SHAPE_MISMATCH")
    targets = raw["target_nodes"]
    if not isinstance(targets, dict) or set(targets) != set(raw["current_nodes"]):
        _fail("TARGET_GRAPH_NODE_TYPES_MISMATCH")
    for name, value in targets.items():
        target = _tensor(value, f"target_nodes.{name}")
        current = _tensor(raw["current_nodes"][name], f"current_nodes.{name}")
        if current.ndim == 3:
            current = current[0]
        if target.ndim != 3 or target.shape[0] != count or target.shape[1] != current.shape[0] or target.shape[-1] != int(arch["node_dim"]):
            _fail(f"TARGET_GRAPH_SHAPE_INVALID:{name}")
    jepa = _mask(raw["jepa_valid"], "jepa_valid")
    if tuple(jepa.shape) != (count, int(arch["latent_dim"])):
        _fail("JEPA_MASK_SHAPE_MISMATCH")
    _validate_masked(raw["state_target"], raw["state_valid"], (count, int(arch["history_dim"])), "state")
    _validate_masked(raw["outcome_target"], raw["outcome_valid"], (count, OUTCOME_DIM), "outcome")
    _validate_masked(raw["event_labels"], raw["event_valid"], (count, EVENT_DIM), "event")

def _validate_masked(values: Any, mask_raw: Any, shape: tuple[int, int], name: str) -> None:
    target, mask = _tensor(values, f"{name}_target", unknown=True), _mask(mask_raw, f"{name}_valid")
    if tuple(target.shape) != shape or tuple(mask.shape) != shape:
        _fail(f"{name.upper()}_TARGET_OR_MASK_SHAPE_MISMATCH")
    if bool((mask & ~torch.isfinite(target)).any()):
        _fail(f"{name.upper()}_VALID_LABEL_UNKNOWN")

def _node_rows(nodes: Mapping[str, torch.Tensor]) -> int:
    if not isinstance(nodes, Mapping) or not nodes:
        _fail("MODEL_GRAPH_BATCH_MISSING")
    rows = {int(value.shape[0]) for value in nodes.values() if isinstance(value, torch.Tensor) and value.ndim == 3}
    if len(rows) != 1 or any(not isinstance(value, torch.Tensor) or value.ndim != 3 for value in nodes.values()):
        _fail("MODEL_GRAPH_BATCH_SHAPE_MISMATCH")
    return rows.pop()

def _model_config(matrix: Mapping[str, Any]) -> W1JEPAConfig:
    arch = matrix.get("world_model_architecture")
    required = {"action_count": 25, "ema": 0.99, "event_count": EVENT_DIM, "hidden_dim": 128,
                "history_dim": 128, "latent_dim": 64, "node_dim": 32, "outcome_dim": OUTCOME_DIM, "relation_dim": 4}
    if not isinstance(arch, Mapping) or dict(arch) != required:
        _fail("WORLD_MODEL_ARCHITECTURE_CONTRACT_MISMATCH")
    return W1JEPAConfig(**{name: arch[name] for name in (
        "action_count", "ema", "hidden_dim", "history_dim", "latent_dim", "node_dim", "outcome_dim", "relation_dim")})

def _frozen_contract(matrix: Mapping[str, Any]) -> dict[str, Any]:
    seeds = matrix.get("world_model_seeds")
    if not isinstance(seeds, list) or len(seeds) != 3 or len(set(map(int, seeds))) != 3:
        _fail("WORLD_MODEL_SEED_CONTRACT_MISMATCH")
    if matrix.get("world_model_variants") != {"G1": "event_loss_disabled", "G2": "event_loss_enabled"}:
        _fail("WORLD_MODEL_VARIANT_CONTRACT_MISMATCH")
    optimizer = matrix.get("world_model_optimizer", {})
    expected_optimizer = {"gradient_clip_norm": 5.0, "learning_rate": 0.001, "name": "AdamW", "weight_decay": 0.00001}
    if optimizer != expected_optimizer:
        _fail("WORLD_MODEL_OPTIMIZER_CONTRACT_MISMATCH")
    loss = matrix.get("world_model_loss", {})
    expected_loss = {"beta_collapse": 0.04, "beta_event_by_variant": {"G1": 0.0, "G2": 1.0},
                     "beta_jepa": 1.0, "beta_outcome": 1.0, "beta_state": 1.0, "unknown_masking": "required"}
    if loss != expected_loss:
        _fail("WORLD_MODEL_LOSS_CONTRACT_MISMATCH")
    selection = matrix.get("world_model_model_selection", {})
    expected_selection = {"confirmation_data_access": False, "early_stopping_patience_epochs": 15,
        "primary_metric": "parent_macro_action_selection_regret",
        "rule": "minimum primary metric on the frozen model-selection parents; ties use lower parent-macro absolute outcome MAE, then earlier epoch"}
    if selection != expected_selection:
        _fail("WORLD_MODEL_SELECTION_CONTRACT_MISMATCH")
    fixture = matrix.get("e2e_fixture")
    if not (isinstance(fixture, Mapping) and bool(fixture.get("enabled"))):
        if int(matrix.get("world_model_batch_size", 0)) != 32 or int(matrix.get("world_model_maximum_epochs", 0)) != 100:
            _fail("WORLD_MODEL_BATCH_OR_EPOCH_CONTRACT_MISMATCH")
        if int(matrix.get("world_model_maximum_train_candidate_rows", 0)) != 600 or int(matrix.get("world_model_maximum_updates_per_variant_seed", 0)) != 1900:
            _fail("WORLD_MODEL_TRAINING_LIMIT_CONTRACT_MISMATCH")
    else:
        if int(matrix.get("world_model_batch_size", 0)) <= 0 or int(matrix.get("world_model_maximum_epochs", 0)) <= 0:
            _fail("E2E_FIXTURE_WORLD_MODEL_LIMIT_INVALID")
        if int(matrix.get("world_model_maximum_updates_per_variant_seed", 0)) <= 0:
            _fail("E2E_FIXTURE_WORLD_MODEL_UPDATE_LIMIT_INVALID")
    transparent = matrix.get("transparent_utility_configuration")
    if transparent != {
        "task_score_divisor": 600.0, "energy_score_weight": 0.1,
        "initial_total_energy": 36.0, "component_order": ["task", "energy"],
        "residual_target": "observed one-step vector component minus transparent component",
    }:
        _fail("TRANSPARENT_UTILITY_CONFIGURATION_MISMATCH")
    return {"seeds": tuple(map(int, seeds)), "optimizer": optimizer, "loss": loss, "selection": selection}

def _window_batch(window: Window) -> WorldModelBatch:
    if window.status != "complete":
        _fail(f"NO_OPPORTUNITY_HAS_NO_TRAINING_BATCH:{window.parent}:{window.repeat}")
    raw, count = window.raw, window.candidate_count
    current_nodes = {}
    for name, value in raw["current_nodes"].items():
        node = _tensor(value, f"current_nodes.{name}")
        if node.ndim == 3 and node.shape[0] == 1:
            node = node[0]
        if node.ndim != 2:
            _fail(f"CURRENT_NODE_RANK_INVALID:{name}")
        current_nodes[name] = node.unsqueeze(0).expand(count, -1, -1).contiguous()
    history = _tensor(raw["history"], "history").reshape(1, -1).expand(count, -1).contiguous()
    actions = torch.as_tensor(raw["actions"], dtype=torch.long)
    relations = _tensor(raw["relations"], "relations")
    target_nodes = {name: _tensor(value, f"target_nodes.{name}") for name, value in raw["target_nodes"].items()}
    jepa_valid = _mask(raw["jepa_valid"], "jepa_valid")
    state_target = _tensor(raw["state_target"], "state_target", unknown=True)
    state_valid = _mask(raw["state_valid"], "state_valid")
    outcome_target = _tensor(raw["outcome_target"], "outcome_target", unknown=True)
    outcome_valid = _mask(raw["outcome_valid"], "outcome_valid")
    event_labels = _tensor(raw["event_labels"], "event_labels", unknown=True)
    event_valid = _mask(raw["event_valid"], "event_valid")
    for label, target, valid in (("state", state_target, state_valid),
                                 ("outcome", outcome_target, outcome_valid),
                                 ("event", event_labels, event_valid)):
        if bool((valid & ~torch.isfinite(target)).any()):
            _fail(f"{label.upper()}_VALID_LABEL_NONFINITE:{window.parent}:{window.repeat}")
    return WorldModelBatch(current_nodes, history, actions, relations, target_nodes, jepa_valid,
                           state_target, state_valid, outcome_target, outcome_valid, event_labels, event_valid)

def _whole_window_groups(windows: Sequence[Window], maximum_rows: int) -> list[list[Window]]:
    groups, current, rows = [], [], 0
    for window in windows:
        count = window.candidate_count
        if count <= 0 or count > maximum_rows:
            _fail(f"WINDOW_CANNOT_FIT_TRAIN_BATCH:{window.parent}:{window.repeat}")
        if current and rows + count > maximum_rows:
            groups.append(current)
            current, rows = [], 0
        current.append(window)
        rows += count
    if current:
        groups.append(current)
    return groups

def _combine_batches(groups: Sequence[Sequence[Window]]) -> list[WorldModelBatch]:
    result = []
    for group in groups:
        batches = [_window_batch(window) for window in group]
        node_types = set(batches[0].current_nodes)
        if any(set(batch.current_nodes) != node_types for batch in batches):
            _fail("BATCH_NODE_TYPES_DIFFER")
        result.append(WorldModelBatch(
            current_nodes={name: torch.cat([batch.current_nodes[name] for batch in batches]) for name in node_types},
            history=torch.cat([batch.history for batch in batches]),
            actions=torch.cat([batch.actions for batch in batches]),
            relations=torch.cat([batch.relations for batch in batches]),
            target_nodes={name: torch.cat([batch.target_nodes[name] for batch in batches]) for name in node_types},
            jepa_valid=torch.cat([batch.jepa_valid for batch in batches]),
            state_target=torch.cat([batch.state_target for batch in batches]),
            state_valid=torch.cat([batch.state_valid for batch in batches]),
            outcome_target=torch.cat([batch.outcome_target for batch in batches]),
            outcome_valid=torch.cat([batch.outcome_valid for batch in batches]),
            event_labels=torch.cat([batch.event_labels for batch in batches]),
            event_valid=torch.cat([batch.event_valid for batch in batches]),
        ))
    return result

def _pick(scores: Sequence[float], actions: Sequence[int], ids: Sequence[str]) -> int:
    if not scores or len(scores) != len(actions) or len(scores) != len(ids):
        _fail("ACTION_SELECTION_SHAPE_MISMATCH")
    for score in scores:
        _number(score, "action_score")
    index = min(range(len(scores)), key=lambda i: (-float(scores[i]), int(actions[i]), str(ids[i])))
    return int(actions[index])

def _utility_residual(values: Sequence[Any], label: str) -> float:
    if len(values) != OUTCOME_DIM:
        _fail(f"UTILITY_COMPONENT_SHAPE_MISMATCH:{label}")
    return (TASK_COMPONENT_SCALE * PREDICTION_PREFERENCE[0]
            * _number(values[TASK_RESIDUAL_INDEX], f"{label}.task_residual")
            + PREDICTION_PREFERENCE[1]
            * _number(values[ENERGY_RESIDUAL_INDEX], f"{label}.energy_residual"))


def _window_labels(window: Window) -> tuple[list[str], list[int], list[float], torch.Tensor, torch.Tensor]:
    raw, count = window.raw, window.candidate_count
    ids = [str(item) for item in raw["candidate_ids"]]
    actions = [int(item) for item in raw["actions"]]
    transparent = [_number(item, "transparent_score") for item in raw["transparent_scores"]]
    target = _tensor(raw["outcome_target"], "outcome_target", unknown=True)
    valid = _mask(raw["outcome_valid"], "outcome_valid")
    if tuple(target.shape) != (count, OUTCOME_DIM) or tuple(valid.shape) != tuple(target.shape):
        _fail(f"OUTCOME_TARGET_SHAPE_MISMATCH:{window.parent}:{window.repeat}")
    if bool((valid & ~torch.isfinite(target)).any()):
        _fail(f"OUTCOME_TARGET_NONFINITE:{window.parent}:{window.repeat}")
    return ids, actions, transparent, target, valid

def _regret(window: Window, predicted_outcomes: Sequence[Sequence[Any]], label: str) -> dict[str, Any]:
    ids, actions, transparent, target, valid = _window_labels(window)
    count = len(ids)
    if len(predicted_outcomes) != count:
        _fail(f"PREDICTION_CANDIDATE_COUNT_MISMATCH:{label}:{window.window_id}")
    predicted_residual = [_utility_residual(row, label) for row in predicted_outcomes]
    pred_scores = [base + _number(residual, label) for base, residual in zip(transparent, predicted_residual)]
    chosen, transparent_choice = _pick(pred_scores, actions, ids), _pick(transparent, actions, ids)
    utility_valid = [bool(valid[index, ENERGY_RESIDUAL_INDEX]) and bool(valid[index, TASK_RESIDUAL_INDEX])
                     for index in range(count)]
    if not all(utility_valid):
        return {"selected_action": chosen, "regret": None, "oracle_action": None,
                "utility_complete": False, "transparent_selected_action": transparent_choice,
                "transparent_regret": None}
    true_utility = window.raw.get("true_utility")
    if not isinstance(true_utility, list) or len(true_utility) != count:
        _fail(f"TRUE_UTILITY_LABEL_INVALID:{window.window_id}")
    utilities = [_number(value, "true_utility") for value in true_utility]
    oracle = min(range(count), key=lambda i: (-utilities[i], actions[i], ids[i]))
    selected, transparent_idx = actions.index(chosen), actions.index(transparent_choice)
    return {"selected_action": chosen,
            "regret": max(0.0, utilities[oracle] - utilities[selected]),
            "oracle_action": actions[oracle], "utility_complete": True,
            "transparent_selected_action": transparent_choice,
            "transparent_regret": max(0.0, utilities[oracle] - utilities[transparent_idx])}

def _model_for(config: W1JEPAConfig, seed: int, ledger: Any, boundary: Any) -> _AccountedModel:
    torch.manual_seed(seed)
    creator = _Operation("world_model.initialize", lambda: W1GraphJEPA(config, event_count=EVENT_DIM))
    model = ledger.call("world_model.initialize", {"model_initializations_or_loads": 1}, boundary.model, creator)
    if model is None or not callable(model):
        _fail("MODEL_INITIALIZATION_RETURNED_INVALID_OBJECT")
    return _AccountedModel(model, ledger, boundary)

def _optimizer_for(model: _AccountedModel, contract: Mapping[str, Any], ledger: Any, boundary: Any) -> _AccountedOptimizer:
    params = tuple(model.parameters())
    def create() -> Any:
        return torch.optim.AdamW(params, lr=float(contract["learning_rate"]), weight_decay=float(contract["weight_decay"]))
    creator = _Operation("world_optimizer.initialize", create)
    optimizer = ledger.call("world_optimizer.initialize", {}, boundary.optimizer, creator)
    if optimizer is None:
        _fail("OPTIMIZER_INITIALIZATION_RETURNED_NONE")
    return _AccountedOptimizer(optimizer, ledger, boundary, params, float(contract["gradient_clip_norm"]))

def _model_output(output: Any, name: str, shape: tuple[int, ...], label: str) -> torch.Tensor:
    if not isinstance(output, Mapping) or name not in output or not isinstance(output[name], torch.Tensor):
        _fail(f"MODEL_OUTPUT_MISSING:{label}:{name}")
    tensor = output[name].detach()
    if tuple(tensor.shape) != shape:
        _fail(f"MODEL_OUTPUT_SHAPE_MISMATCH:{label}:{name}:{tuple(tensor.shape)}!={shape}")
    if not bool(torch.isfinite(tensor).all()):
        _fail(f"MODEL_OUTPUT_NONFINITE:{label}:{name}")
    return tensor

def _parent_macro(values: Mapping[str, Sequence[float]], name: str) -> tuple[float, dict[str, float]]:
    per_parent = {parent: sum(items) / len(items) for parent, items in values.items() if items}
    if not per_parent:
        _fail(f"MODEL_SELECTION_METRIC_UNAVAILABLE:{name}")
    macro = sum(per_parent.values()) / len(per_parent)
    if not math.isfinite(macro):
        _fail(f"MODEL_SELECTION_METRIC_NONFINITE:{name}")
    return macro, per_parent

def _selection_metrics(model: _AccountedModel, windows: Sequence[Window]) -> tuple[float, float]:
    regret_by_parent: dict[str, list[float]] = {}
    mae_by_parent: dict[str, list[float]] = {}
    model.eval()
    with torch.no_grad():
        for window in windows:
            batch, count = _window_batch(window), window.candidate_count
            output = model(batch.current_nodes, batch.history, batch.actions, batch.relations)
            prediction = _model_output(output, "outcome", (count, OUTCOME_DIM), f"selection:{window.parent}:{window.repeat}")
            result = _regret(window, prediction.cpu().tolist(), "model-selection")
            if not result["utility_complete"] or result["regret"] is None:
                _fail(f"MODEL_SELECTION_UTILITY_LABELS_INCOMPLETE:{window.parent}:{window.repeat}")
            regret_by_parent.setdefault(window.parent, []).append(float(result["regret"]))
            valid = batch.outcome_valid & torch.isfinite(batch.outcome_target)
            if not bool(valid.any()):
                _fail(f"MODEL_SELECTION_OUTCOME_LABELS_EMPTY:{window.parent}:{window.repeat}")
            mae = float((prediction[valid] - batch.outcome_target[valid]).abs().mean())
            if not math.isfinite(mae):
                _fail("MODEL_SELECTION_MAE_NONFINITE")
            mae_by_parent.setdefault(window.parent, []).append(mae)
    if len(regret_by_parent) != 8 or set(regret_by_parent) != set(mae_by_parent):
        _fail("MODEL_SELECTION_REQUIRES_EIGHT_VALID_PARENTS")
    return _parent_macro(regret_by_parent, "action_regret")[0], _parent_macro(mae_by_parent, "outcome_mae")[0]

def _clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {str(name): tensor.detach().cpu().clone() for name, tensor in state.items()}

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _sync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def _save_checkpoint(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_dir(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_dir(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

def _unit_identity(window: Window) -> dict[str, Any]:
    return {"split": window.split, "parent": window.parent, "repeat": window.repeat,
            "window_id": window.window_id, "status": window.status}

def train_select_world_models(records: Sequence[Mapping[str, Any]], matrix: Mapping[str, Any],
                              ledger: Any, boundary: Any, output_dir: Path | str) -> TrainingRun:
    """Train/early-select exactly the six frozen G1/G2 by seed routes."""
    contract = _frozen_contract(matrix)
    if any(row.get("split") not in ("train", "model_selection") for row in records):
        _fail("CONFIRMATION_DATA_PROHIBITED_DURING_WORLD_MODEL_TRAINING")
    train_all = validate_windows([row for row in records if row.get("split") == "train"], matrix, "train")
    selection_all = validate_windows([row for row in records if row.get("split") == "model_selection"], matrix, "model_selection")
    train_windows = [window for window in train_all if window.status == "complete"]
    selection_windows = [window for window in selection_all if window.status == "complete"]
    if not train_windows:
        _fail("TRAINING_WINDOWS_EMPTY")
    if len(selection_windows) != 8 or len({window.parent for window in selection_windows}) != 8:
        _fail("MODEL_SELECTION_WINDOW_COVERAGE_NOT_EIGHT")
    groups, batch_size = [], int(matrix["world_model_batch_size"])
    rows_in_group = 0
    current_group: list[Window] = []
    for window in train_windows:
        if window.candidate_count > batch_size:
            _fail(f"WINDOW_EXCEEDS_FROZEN_BATCH:{window.parent}:{window.repeat}")
        if current_group and rows_in_group + window.candidate_count > batch_size:
            groups.append(current_group)
            current_group, rows_in_group = [], 0
        current_group.append(window)
        rows_in_group += window.candidate_count
    if current_group:
        groups.append(current_group)
    train_rows = sum(window.candidate_count for window in train_windows)
    if train_rows > int(matrix["world_model_maximum_train_candidate_rows"]):
        _fail(f"TRAINING_CANDIDATE_ROW_CAP_EXCEEDED:{train_rows}")
    batches = _combine_batches(groups)
    max_epochs = int(matrix["world_model_maximum_epochs"])
    max_updates = int(matrix["world_model_maximum_updates_per_variant_seed"])
    epochs_budgeted = min(max_epochs, max_updates // len(batches))
    if epochs_budgeted <= 0:
        _fail("WORLD_MODEL_TRAINING_BATCHES_EXCEED_UPDATE_CAP")
    patience = int(contract["selection"]["early_stopping_patience_epochs"])
    config = _model_config(matrix)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ledger.select("world_model_training_and_selection")
    routes: list[dict[str, Any]] = []
    models: dict[tuple[str, int], Any] = {}
    for variant in VARIANTS:
        event_enabled = variant == "G2"
        event_beta = float(contract["loss"]["beta_event_by_variant"][variant])
        for seed in contract["seeds"]:
            model = _model_for(config, seed, ledger, boundary)
            optimizer = _optimizer_for(model, contract["optimizer"], ledger, boundary)
            best_regret = best_mae = None
            best_epoch = None
            best_state = None
            stale, updates = 0, 0
            loss_history: list[dict[str, float | int]] = []
            for epoch in range(1, epochs_budgeted + 1):
                losses = train_world_model(
                    model, batches, optimizer=optimizer, event_enabled=event_enabled,
                    beta_state=float(contract["loss"]["beta_state"]),
                    beta_outcome=float(contract["loss"]["beta_outcome"]),
                    beta_event=event_beta, beta_collapse=float(contract["loss"]["beta_collapse"]),
                    max_updates=len(batches),
                )
                updates += int(losses["updates"])
                if updates > max_updates:
                    _fail(f"WORLD_MODEL_ROUTE_UPDATE_CAP_EXCEEDED:{variant}:{seed}")
                loss_history.append(dict(losses))
                val_regret, val_mae = _selection_metrics(model, selection_windows)
                improved = best_regret is None or val_regret < best_regret or (val_regret == best_regret and val_mae < best_mae)
                if improved:
                    best_regret, best_mae, best_epoch = val_regret, val_mae, epoch
                    best_state, stale = _clone_state(model.state_dict()), 0
                else:
                    stale += 1
                if stale >= patience:
                    break
            if best_state is None or best_epoch is None or best_regret is None or best_mae is None:
                _fail(f"WORLD_MODEL_NO_SELECTED_STATE:{variant}:{seed}")
            model.load_state_dict(best_state)
            relative = Path("world-model-checkpoints") / variant / f"seed-{seed}.pt"
            checkpoint_path = output / relative
            metadata = {
                "schema": "w1-world-model-checkpoint/1.0.0", "variant": variant, "seed": seed,
                "event_loss_enabled": event_enabled, "event_beta": event_beta,
                "architecture": dict(matrix["world_model_architecture"]),
                "training_parents": [window.parent for window in train_windows],
                "model_selection_parents": [window.parent for window in selection_windows],
                "training_candidate_rows": train_rows, "best_epoch": best_epoch,
                "selection_rule": contract["selection"]["rule"],
                "selection_regret": best_regret, "selection_outcome_mae_tiebreak": best_mae,
                "optimizer": dict(contract["optimizer"]), "loss": {
                    "beta_jepa": contract["loss"]["beta_jepa"], "beta_state": contract["loss"]["beta_state"],
                    "beta_outcome": contract["loss"]["beta_outcome"], "beta_event": event_beta,
                    "beta_collapse": contract["loss"]["beta_collapse"]},
                "continuation_id": str(matrix["candidate_continuation_id"]),
            }
            save = _Operation("checkpoint.save", _save_checkpoint)
            ledger.call("world_model.checkpoint_write", {"checkpoint_writes": 1}, boundary.checkpoint,
                        save, {"metadata": metadata, "state_dict": best_state}, checkpoint_path)
            digest = _file_sha256(checkpoint_path)
            routes.append({
                "variant": variant, "seed": seed, "event_loss_enabled": event_enabled,
                "event_beta": event_beta, "epochs_trained": len(loss_history), "updates": updates,
                "train_window_count": len(train_windows), "train_candidate_rows": train_rows,
                "selection_parent_count": 8, "best_epoch": best_epoch,
                "selection_regret": best_regret, "selection_outcome_mae_tiebreak": best_mae,
                "checkpoint_relative_path": relative.as_posix(), "checkpoint_sha256": digest,
                "checkpoint_metadata": metadata, "loss_by_epoch": loss_history,
            })
            models[(variant, seed)] = model
    expected_routes = {(variant, seed) for variant in VARIANTS for seed in contract["seeds"]}
    if len(routes) != 6 or {(item["variant"], item["seed"]) for item in routes} != expected_routes:
        _fail("WORLD_MODEL_ROUTE_SET_INCOMPLETE")
    _write_json(output / "world-model-training-summary.json", {
        "schema": "w1-world-model-training-summary/1.0.0", "routes": routes,
        "preserved_no_opportunity": [_unit_identity(window) for window in train_all + selection_all if window.status == "no_opportunity"],
    })
    return TrainingRun(tuple(routes), models)

def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)

def _load_selected_models(routes: Sequence[Mapping[str, Any]], matrix: Mapping[str, Any],
                          output_dir: Path, ledger: Any, boundary: Any) -> dict[tuple[str, int], _AccountedModel]:
    expected = {(variant, int(seed)) for variant in VARIANTS for seed in matrix["world_model_seeds"]}
    route_map = {}
    for route in routes:
        key = (str(route.get("variant")), int(route.get("seed", -1)))
        if key not in expected or key in route_map:
            _fail(f"CHECKPOINT_ROUTE_DUPLICATE_OR_UNEXPECTED:{key}")
        route_map[key] = route
    if set(route_map) != expected or len(route_map) != 6:
        _fail("CHECKPOINT_ROUTE_SET_MISMATCH")
    config = _model_config(matrix)
    models = {}
    for key in sorted(expected):
        route = route_map[key]
        relative = Path(str(route.get("checkpoint_relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            _fail(f"CHECKPOINT_PATH_INVALID:{key}")
        path = output_dir / relative
        digest = _digest(route.get("checkpoint_sha256"), "checkpoint_sha256")
        if not path.is_file() or _file_sha256(path) != digest:
            _fail(f"CHECKPOINT_DIGEST_MISMATCH:{key}")
        ledger.select("prediction_confirmation")
        model = _model_for(config, key[1], ledger, boundary)
        load = _Operation("checkpoint.load", _load_checkpoint)
        payload = ledger.call("world_model.checkpoint_load", {}, boundary.checkpoint, load, path)
        if not isinstance(payload, Mapping) or set(payload) != {"metadata", "state_dict"}:
            _fail(f"CHECKPOINT_PAYLOAD_INVALID:{key}")
        metadata = payload["metadata"]
        if (metadata != route.get("checkpoint_metadata")
                or metadata.get("schema") != "w1-world-model-checkpoint/1.0.0"
                or metadata.get("variant") != key[0]
                or int(metadata.get("seed", -1)) != key[1]
                or metadata.get("event_loss_enabled") != (key[0] == "G2")
                or metadata.get("continuation_id") != matrix.get("candidate_continuation_id")):
            _fail(f"CHECKPOINT_IDENTITY_MISMATCH:{key}")
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models[key] = model
    return models

def _masked_list(values: torch.Tensor, valid: torch.Tensor) -> list[Any]:
    data, mask = values.detach().cpu().tolist(), valid.detach().cpu().tolist()
    return [float(value) if bool(flag) else None for value, flag in zip(data, mask)]

def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _sync_dir(path.parent)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass

def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True,
                                        separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _sync_dir(path.parent)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass

def _ensemble(values: Sequence[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack([value.detach().to(dtype=torch.float64) for value in values])
    if not bool(torch.isfinite(stacked).all()):
        _fail("ENSEMBLE_INPUT_NONFINITE")
    result = stacked.mean(dim=0)
    if not bool(torch.isfinite(result).all()):
        _fail("ENSEMBLE_OUTPUT_NONFINITE")
    return result

def _prediction_for_window(window: Window, models: Mapping[tuple[str, int], _AccountedModel],
                           matrix: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    batch, count = _window_batch(window), window.candidate_count
    ids, actions, transparent, target, target_valid = _window_labels(window)
    predictions: dict[str, dict[str, Any]] = {variant: {} for variant in VARIANTS}
    with torch.no_grad():
        for variant in VARIANTS:
            for seed in (int(value) for value in matrix["world_model_seeds"]):
                model = models[(variant, seed)]
                model.eval()
                output = model(batch.current_nodes, batch.history, batch.actions, batch.relations)
                label = f"{window.parent}:{window.repeat}:{variant}:{seed}"
                latent = _model_output(output, "latent", (count, int(matrix["world_model_architecture"]["latent_dim"])), label)
                state = _model_output(output, "public_state", (count, int(matrix["world_model_architecture"]["history_dim"])), label)
                outcome = _model_output(output, "outcome", (count, OUTCOME_DIM), label)
                logits = _model_output(output, "event_logits", (count, EVENT_DIM), label)
                probability = torch.sigmoid(logits)
                if not bool(torch.isfinite(probability).all()):
                    _fail(f"MODEL_EVENT_PROBABILITY_NONFINITE:{label}")
                predictions[variant][str(seed)] = {
                    "latent": latent.cpu().tolist(), "public_state": state.cpu().tolist(),
                    "outcome": outcome.cpu().tolist(), "event_probability": probability.cpu().tolist(),
                }
    for variant in VARIANTS:
        for head in ("latent", "public_state", "outcome", "event_probability"):
            stack = [torch.tensor(predictions[variant][str(seed)][head], dtype=torch.float64)
                     for seed in matrix["world_model_seeds"]]
            predictions[variant]["ensemble"] = predictions[variant].get("ensemble", {})
            predictions[variant]["ensemble"][head] = _ensemble(stack).tolist()
    decisions = {"transparent": _regret(window, [[0.0] * OUTCOME_DIM for _ in range(count)], "transparent")}
    for variant in VARIANTS:
        for seed in matrix["world_model_seeds"]:
            outcomes = predictions[variant][str(seed)]["outcome"]
            decisions[f"{variant}_seed_{seed}"] = _regret(window, outcomes, f"{variant}:{seed}")
        outcomes = predictions[variant]["ensemble"]["outcome"]
        decisions[variant] = _regret(window, outcomes, f"{variant}:ensemble")
    rows = []
    state_target, state_valid = batch.state_target, batch.state_valid
    event_labels, event_valid = batch.event_labels, batch.event_valid
    for index, candidate_id in enumerate(ids):
        row_predictions = {}
        for variant in VARIANTS:
            row_predictions[variant] = {
                **{str(seed): {head: predictions[variant][str(seed)][head][index]
                               for head in ("latent", "public_state", "outcome", "event_probability")}
                   for seed in matrix["world_model_seeds"]},
                "ensemble": {head: predictions[variant]["ensemble"][head][index]
                             for head in ("latent", "public_state", "outcome", "event_probability")},
            }
        residual_valid = bool(target_valid[index, ENERGY_RESIDUAL_INDEX]) and bool(target_valid[index, TASK_RESIDUAL_INDEX])
        residual = _utility_residual(target[index].tolist(), "true_residual") if residual_valid else None
        rows.append({
            "candidate_id": candidate_id, "action_id": actions[index], "transparent_score": transparent[index],
            "transparent_components": list(window.raw["transparent_components"][index]),
            "true_utility": _number(window.raw["true_utility"][index], "true_utility") if residual_valid else None,
            "outcome_target": _masked_list(target[index], target_valid[index]),
            "outcome_valid": target_valid[index].cpu().tolist(),
            "state_target": _masked_list(state_target[index], state_valid[index]),
            "state_valid": state_valid[index].cpu().tolist(),
            "event_labels": _masked_list(event_labels[index], event_valid[index]),
            "event_valid": event_valid[index].cpu().tolist(),
            "jepa_valid": batch.jepa_valid[index].cpu().tolist(), "predictions": row_predictions,
        })
    trace_window = {
        "schema": TRACE_SCHEMA, "split": window.split, "parent": window.parent,
        "repeat": window.repeat, "window_id": window.window_id,
        "scenario_sha256": window.scenario_sha256, "continuation_id": window.continuation_id,
        "input_hash": window.input_hash, "candidate_count": count,
        "scoring_preference": list(PREDICTION_PREFERENCE),
        "task_component_scale": TASK_COMPONENT_SCALE,
        "candidate_rows": rows, "decisions": decisions,
    }
    return trace_window, decisions

def evaluate_prediction_confirmation(records: Sequence[Mapping[str, Any]], matrix: Mapping[str, Any],
                                      routes: Sequence[Mapping[str, Any]], ledger: Any,
                                      boundary: Any, output_dir: Path | str) -> PredictionRun:
    """Load selected checkpoints and execute exactly one batch forward per model/window."""
    if any(row.get("split") != "prediction_confirmation" for row in records):
        _fail("NON_CONFIRMATION_DATA_IN_PREDICTION_STAGE")
    windows = validate_windows(records, matrix, "prediction_confirmation")
    complete = [window for window in windows if window.status == "complete"]
    if not complete:
        _fail("PREDICTION_CONFIRMATION_HAS_NO_COMPLETE_WINDOW")
    output = Path(output_dir)
    models = _load_selected_models(routes, matrix, output, ledger, boundary)
    trace_rows = []
    for window in windows:
        if window.status == "no_opportunity":
            trace_rows.append({
                "schema": TRACE_SCHEMA, "split": window.split, "parent": window.parent,
                "repeat": window.repeat, "window_id": window.window_id,
                "scenario_sha256": window.scenario_sha256, "continuation_id": window.continuation_id,
                "input_hash": window.input_hash or None, "candidate_count": 0,
                "candidate_rows": [], "decisions": {}, "status": "no_opportunity",
            })
            continue
        row, _ = _prediction_for_window(window, models, matrix)
        trace_rows.append(row)
    trace_path = output / "prediction-trace.jsonl"
    metrics_path = output / "prediction-metrics.json"
    _atomic_jsonl(trace_path, trace_rows)
    expected_parents = len(matrix["splits"]["prediction_confirmation"])
    metrics = recompute_prediction_metrics(trace_path, expected_parent_count=expected_parents,
                                           seeds=tuple(map(int, matrix["world_model_seeds"])))
    _atomic_json(metrics_path, metrics)
    return PredictionRun(trace_path, metrics_path, metrics)

def _trace_decision(window: Mapping[str, Any], method: str, seeds: Sequence[int]) -> dict[str, Any]:
    candidates = window.get("candidate_rows")
    if not isinstance(candidates, list) or not candidates:
        _fail("TRACE_CANDIDATES_MISSING")
    ids = [str(row.get("candidate_id", "")) for row in candidates]
    actions = [int(row.get("action_id", -1)) for row in candidates]
    transparent = [_number(row.get("transparent_score"), "trace.transparent_score") for row in candidates]
    if len(set(ids)) != len(ids) or len(set(actions)) != len(actions):
        _fail("TRACE_CANDIDATE_IDENTITY_DUPLICATE")
    if method == "transparent":
        predicted_residual = [0.0] * len(candidates)
    else:
        variant, _, seed_label = method.partition("_seed_")
        if variant not in VARIANTS:
            _fail(f"TRACE_METHOD_INVALID:{method}")
        if seed_label and int(seed_label) not in seeds:
            _fail(f"TRACE_SEED_INVALID:{method}")
        key = seed_label if seed_label else "ensemble"
        predicted_residual = []
        for row in candidates:
            try:
                prediction = row["predictions"][variant][key]["outcome"]
                if len(prediction) != OUTCOME_DIM:
                    _fail(f"TRACE_OUTCOME_PREDICTION_SHAPE_INVALID:{method}")
                predicted_residual.append(_utility_residual(prediction, "trace.utility_prediction"))
            except (KeyError, TypeError) as exc:
                raise WorldModelContractError(f"TRACE_PREDICTION_MISSING:{method}") from exc
    chosen = _pick([base + residual for base, residual in zip(transparent, predicted_residual)], actions, ids)
    transparent_choice = _pick(transparent, actions, ids)
    masks = [row.get("outcome_valid") for row in candidates]
    labels = [row.get("outcome_target") for row in candidates]
    if any(not isinstance(mask, list) or len(mask) != OUTCOME_DIM for mask in masks):
        _fail("TRACE_OUTCOME_MASK_SHAPE_INVALID")
    if any(not isinstance(label, list) or len(label) != OUTCOME_DIM for label in labels):
        _fail("TRACE_OUTCOME_LABEL_SHAPE_INVALID")
    if not all(bool(mask[ENERGY_RESIDUAL_INDEX]) and bool(mask[TASK_RESIDUAL_INDEX]) for mask in masks):
        return {"selected_action": chosen, "regret": None, "oracle_action": None,
                "utility_complete": False, "transparent_selected_action": transparent_choice,
                "transparent_regret": None}
    utility = [_number(row.get("true_utility"), "trace.true_utility") for row in candidates]
    oracle = min(range(len(actions)), key=lambda i: (-utility[i], actions[i], ids[i]))
    return {"selected_action": chosen, "regret": max(0.0, utility[oracle] - utility[actions.index(chosen)]),
            "oracle_action": actions[oracle], "utility_complete": True,
            "transparent_selected_action": transparent_choice,
            "transparent_regret": max(0.0, utility[oracle] - utility[actions.index(transparent_choice)])}

def _parent_macro(values: Mapping[str, Sequence[float]], label: str) -> tuple[float | None, dict[str, float]]:
    per_parent = {parent: sum(rows) / len(rows) for parent, rows in values.items() if rows}
    if not per_parent:
        return None, {}
    macro = sum(per_parent.values()) / len(per_parent)
    if not math.isfinite(macro):
        _fail(f"METRIC_NONFINITE:{label}")
    return macro, per_parent

def _parent_rmse(values: Mapping[str, Sequence[float]], label: str) -> tuple[float | None, dict[str, float]]:
    per_parent = {
        parent: math.sqrt(sum(items) / len(items))
        for parent, items in values.items() if items
    }
    if not per_parent:
        return None, {}
    macro = sum(per_parent.values()) / len(per_parent)
    if not math.isfinite(macro):
        _fail(f"METRIC_NONFINITE:{label}")
    return macro, per_parent

def _parent_ece(values: Mapping[str, Sequence[tuple[float, float]]], *, bins: int = 10) -> tuple[float | None, dict[str, float]]:
    per_parent: dict[str, float] = {}
    for parent, pairs in values.items():
        if not pairs:
            continue
        total = len(pairs)
        error = 0.0
        for index in range(bins):
            lower, upper = index / bins, (index + 1) / bins
            bucket = [(probability, label) for probability, label in pairs
                      if lower <= probability < upper or (index == bins - 1 and probability == 1.0)]
            if bucket:
                confidence = sum(probability for probability, _ in bucket) / len(bucket)
                frequency = sum(label for _, label in bucket) / len(bucket)
                error += len(bucket) / total * abs(confidence - frequency)
        per_parent[parent] = error
    if not per_parent:
        return None, {}
    return sum(per_parent.values()) / len(per_parent), per_parent

def _check_decision_agreement(stored: Mapping[str, Any], recomputed: Mapping[str, Any], method: str) -> None:
    if not isinstance(stored, Mapping):
        _fail(f"TRACE_DECISION_MISSING:{method}")
    if stored.get("selected_action") != recomputed.get("selected_action"):
        _fail(f"TRACE_SELECTED_ACTION_MISMATCH:{method}")
    left, right = stored.get("regret"), recomputed.get("regret")
    if left is None or right is None:
        if left is not None or right is not None:
            _fail(f"TRACE_REGRET_MASK_MISMATCH:{method}")
    elif not math.isclose(_number(left, "stored_regret"), _number(right, "computed_regret"), rel_tol=0.0, abs_tol=1e-12):
        _fail(f"TRACE_REGRET_MISMATCH:{method}")

def recompute_prediction_metrics(trace_path: Path | str, *, expected_parent_count: int = 8,
                                 seeds: Sequence[int] = (8201, 8202, 8203)) -> dict[str, Any]:
    """Recompute candidate choices, regret, and masked errors from persisted trace only."""
    rows = []
    with Path(trace_path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WorldModelContractError(f"TRACE_INVALID_JSON:{line_no}") from exc
            if not isinstance(row, dict) or row.get("schema") != TRACE_SCHEMA:
                _fail(f"TRACE_SCHEMA_MISMATCH:{line_no}")
            rows.append(row)
    if not rows:
        _fail("TRACE_EMPTY")
    methods = ["transparent", "G1", "G2"] + [f"{variant}_seed_{seed}" for variant in VARIANTS for seed in seeds]
    regret_rows: dict[str, dict[str, list[float]]] = {name: {} for name in methods}
    top1_rows: dict[str, dict[str, list[float]]] = {name: {} for name in methods}
    outcome_rows: dict[str, dict[str, list[float]]] = {}
    outcome_squared_rows: dict[str, dict[str, list[float]]] = {}
    state_rows: dict[str, dict[str, list[float]]] = {}
    state_squared_rows: dict[str, dict[str, list[float]]] = {}
    event_rows: dict[str, dict[str, list[float]]] = {}
    event_nll_rows: dict[str, dict[str, list[float]]] = {}
    event_calibration_rows: dict[str, dict[str, list[tuple[float, float]]]] = {}
    pairwise_rows: dict[str, dict[str, list[float]]] = {name: {} for name in methods}
    per_window = []
    seen = set()
    for window in rows:
        parent, repeat = str(window.get("parent", "")), int(window.get("repeat", -1))
        key = (parent, repeat)
        if not parent or repeat < 0 or key in seen:
            _fail(f"TRACE_PARENT_REPEAT_INVALID_OR_DUPLICATE:{key}")
        seen.add(key)
        candidates = window.get("candidate_rows")
        if not candidates:
            if window.get("status") != "no_opportunity" or int(window.get("candidate_count", -1)) != 0:
                _fail(f"TRACE_EMPTY_CANDIDATES_NOT_NO_OPPORTUNITY:{key}")
            per_window.append({"parent": parent, "repeat": repeat, "status": "no_opportunity", "selected": {}, "regret": {}})
            continue
        if int(window.get("candidate_count", -1)) != len(candidates):
            _fail(f"TRACE_CANDIDATE_COUNT_MISMATCH:{key}")
        oracle_values = []
        for row in candidates:
            if "predictions" not in row or "outcome_target" not in row or "outcome_valid" not in row:
                _fail(f"TRACE_CANDIDATE_FIELDS_MISSING:{key}")
            mask, target = row["outcome_valid"], row["outcome_target"]
            if len(mask) != OUTCOME_DIM or len(target) != OUTCOME_DIM:
                _fail("TRACE_OUTCOME_LABEL_SHAPE_INVALID")
            if mask[ENERGY_RESIDUAL_INDEX] and mask[TASK_RESIDUAL_INDEX]:
                oracle_values.append(_number(row.get("true_utility"), "true_utility"))
            elif target[ENERGY_RESIDUAL_INDEX] is not None or target[TASK_RESIDUAL_INDEX] is not None:
                _fail("UNKNOWN_UTILITY_COMPONENT_WAS_NOT_MASKED")
        valid_utility = len(oracle_values) == len(candidates)
        oracle_value = max(oracle_values) if valid_utility else None
        decisions = {}
        selected = {}
        for method in methods:
            decision = _trace_decision(window, method, seeds)
            stored_key = method
            _check_decision_agreement(window.get("decisions", {}).get(stored_key), decision, method)
            selected[method] = decision["selected_action"]
            decisions[method] = decision["regret"]
            if decision["regret"] is not None:
                regret_rows[method].setdefault(parent, []).append(float(decision["regret"]))
                top1_rows[method].setdefault(parent, []).append(float(decision["selected_action"] == decision["oracle_action"]))
        if valid_utility and len(candidates) >= 2:
            true_scores = oracle_values
            for method in methods:
                if method == "transparent":
                    predicted_scores = [_number(row["transparent_score"], "transparent_score") for row in candidates]
                else:
                    variant, _, seed_label = method.partition("_seed_")
                    model_key = seed_label or "ensemble"
                    predicted_scores = []
                    for candidate in candidates:
                        outcome = candidate["predictions"][variant][model_key]["outcome"]
                        predicted_scores.append(
                            _number(candidate["transparent_score"], "transparent_score")
                            + _utility_residual(outcome, "utility_prediction")
                        )
                correct = []
                for left in range(len(candidates)):
                    for right in range(left + 1, len(candidates)):
                        true_delta = true_scores[left] - true_scores[right]
                        if true_delta == 0.0:
                            continue
                        predicted_delta = predicted_scores[left] - predicted_scores[right]
                        correct.append(float(predicted_delta * true_delta > 0.0))
                if correct:
                    pairwise_rows[method].setdefault(parent, []).append(sum(correct) / len(correct))
        per_window.append({"parent": parent, "repeat": repeat, "status": "complete",
                           "selected": selected, "regret": decisions, "oracle_utility": oracle_value})
        for variant in VARIANTS:
            for model_key in ["ensemble", *map(str, seeds)]:
                prediction_key = f"{variant}:{model_key}"
                out_error, out_squared, state_error, state_squared = [], [], [], []
                event_error, event_nll, event_pairs = [], [], []
                for candidate in candidates:
                    prediction = candidate["predictions"].get(variant, {}).get(model_key)
                    if not isinstance(prediction, Mapping):
                        _fail(f"TRACE_MODEL_PREDICTION_MISSING:{prediction_key}")
                    for head, target_field, valid_field, bucket in (
                            ("outcome", "outcome_target", "outcome_valid", outcome_rows),
                            ("public_state", "state_target", "state_valid", state_rows)):
                        predicted, target, mask = prediction.get(head), candidate.get(target_field), candidate.get(valid_field)
                        if not isinstance(predicted, list) or not isinstance(target, list) or not isinstance(mask, list) or len(predicted) != len(target) or len(mask) != len(target):
                            _fail(f"TRACE_METRIC_SHAPE_MISMATCH:{prediction_key}:{head}")
                        errors = [_number(predicted[i], f"{head}_prediction") - _number(target[i], f"{head}_target")
                                  for i, valid in enumerate(mask) if valid]
                        if errors:
                            if head == "outcome":
                                out_error.extend(abs(error) for error in errors)
                                out_squared.extend(error * error for error in errors)
                            else:
                                state_error.extend(abs(error) for error in errors)
                                state_squared.extend(error * error for error in errors)
                    labels, valid = candidate.get("event_labels"), candidate.get("event_valid")
                    probabilities = prediction.get("event_probability")
                    if not isinstance(labels, list) or not isinstance(valid, list) or not isinstance(probabilities, list) or len(labels) != EVENT_DIM or len(valid) != EVENT_DIM or len(probabilities) != EVENT_DIM:
                        _fail(f"TRACE_EVENT_SHAPE_INVALID:{prediction_key}")
                    for index, is_valid in enumerate(valid):
                        if is_valid:
                            label = _number(labels[index], "event_label")
                            probability = _number(probabilities[index], "event_probability")
                            if label not in (0.0, 1.0) or not 0 <= probability <= 1:
                                _fail("TRACE_EVENT_VALUE_INVALID")
                            event_error.append((probability - label) ** 2)
                            clipped = min(max(probability, 1e-7), 1.0 - 1e-7)
                            event_nll.append(-(label * math.log(clipped) + (1.0 - label) * math.log(1.0 - clipped)))
                            event_pairs.append((probability, label))
                for bucket, errors in ((outcome_rows, out_error), (outcome_squared_rows, out_squared),
                                       (state_rows, state_error), (state_squared_rows, state_squared),
                                       (event_rows, event_error), (event_nll_rows, event_nll)):
                    if errors:
                        bucket.setdefault(prediction_key, {}).setdefault(parent, []).append(sum(errors) / len(errors))
                if event_pairs:
                    event_calibration_rows.setdefault(prediction_key, {}).setdefault(parent, []).extend(event_pairs)
    parent_regret = {method: {parent: sum(values) / len(values) for parent, values in grouped.items()}
                     for method, grouped in regret_rows.items()}
    parent_top1 = {method: {parent: sum(values) / len(values) for parent, values in grouped.items()}
                   for method, grouped in top1_rows.items()}
    action_parents = set(parent_regret["transparent"]) & set(parent_regret["G1"]) & set(parent_regret["G2"])
    event_parent_1 = _parent_macro(event_rows.get("G1:ensemble", {}), "G1_EVENT_BRIER")[1]
    event_parent_2 = _parent_macro(event_rows.get("G2:ensemble", {}), "G2_EVENT_BRIER")[1]
    common_events = set(event_parent_1) & set(event_parent_2)
    common = action_parents & common_events
    if not common:
        _fail("PREDICTION_GATE_METRICS_HAVE_NO_COMMON_VALID_PARENTS")
    g1_delta = sum(parent_regret["G1"][parent] - parent_regret["transparent"][parent] for parent in action_parents) / len(action_parents)
    g2_delta = sum(parent_regret["G2"][parent] - parent_regret["G1"][parent] for parent in action_parents) / len(action_parents)
    event_delta = sum(event_parent_2[parent] - event_parent_1[parent] for parent in common_events) / len(common_events)
    if any(not math.isfinite(value) for value in (g1_delta, g2_delta, event_delta)):
        _fail("PREDICTION_GATE_METRIC_NONFINITE")
    outcome_macro = {name: _parent_macro(values, name)[0] for name, values in outcome_rows.items()}
    outcome_rmse = {name: _parent_rmse(values, name)[0] for name, values in outcome_squared_rows.items()}
    state_macro = {name: _parent_macro(values, name)[0] for name, values in state_rows.items()}
    state_rmse = {name: _parent_rmse(values, name)[0] for name, values in state_squared_rows.items()}
    event_macro = {name: _parent_macro(values, name)[0] for name, values in event_rows.items()}
    event_nll_macro = {name: _parent_macro(values, name)[0] for name, values in event_nll_rows.items()}
    event_ece_macro = {name: _parent_ece(values)[0] for name, values in event_calibration_rows.items()}
    pairwise_macro = {name: _parent_macro(values, name)[0] for name, values in pairwise_rows.items()}
    per_seed = {method: values for method, values in parent_regret.items() if "_seed_" in method}
    return {
        "schema": "w1-world-model-metrics/1.0.0",
        "regret_definition": "transparent task/energy utility components plus one-step residual outcome[4]/outcome[3] at frozen preference [0.8,0.2]; no fixed-continuation terminal value",
        "selection_tie_rule": "highest score, then lowest action_id, then lexical candidate_id",
        "parent_macro_denominator": len(action_parents), "valid_parent_count": len(common),
        "expected_parent_count": expected_parent_count,
        "g1_minus_transparent_macro_regret": g1_delta,
        "g2_minus_g1_macro_regret": g2_delta,
        "g2_minus_g1_event_brier": event_delta,
        "parents_g1_no_worse_than_transparent": sum(parent_regret["G1"][p] <= parent_regret["transparent"][p] for p in action_parents),
        "parents_g2_no_worse_than_g1": sum(parent_regret["G2"][p] <= parent_regret["G1"][p] for p in action_parents),
        "parent_macro_regret": parent_regret,
        "parent_macro_top1": parent_top1,
        "parent_macro_absolute_outcome_mae": outcome_macro,
        "parent_macro_outcome_rmse": outcome_rmse,
        "parent_macro_public_state_mae": state_macro,
        "parent_macro_public_state_rmse": state_rmse,
        "parent_macro_pairwise_direction_accuracy": pairwise_macro,
        "event_brier_parent_macro": event_macro,
        "event_nll_parent_macro": event_nll_macro,
        "event_ece_10bin_parent_macro": event_ece_macro,
        "seed_parent_macro_regret": per_seed,
        "window_decisions": per_window,
        "coverage_pass": len(common) == expected_parent_count,
    }

__all__ = ["PredictionRun", "TrainingRun", "WINDOW_SCHEMA", "TRACE_SCHEMA",
           "WorldModelContractError", "evaluate_prediction_confirmation",
           "load_complete_windows", "recompute_prediction_metrics",
           "train_select_world_models", "validate_windows"]
