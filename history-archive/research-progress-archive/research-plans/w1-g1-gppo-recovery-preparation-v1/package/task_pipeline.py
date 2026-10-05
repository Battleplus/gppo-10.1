"""Task-only policy training and paired task-confirmation orchestration."""
from __future__ import annotations

import importlib
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Mapping

from infra_io import durable_atomic_json


POLICY_METHODS = ("G0", "T", "G1")
POLICY_SEEDS = (8301, 8302, 8303)
WORLD_SEEDS = {8301: 8201, 8302: 8202, 8303: 8203}
REQUIRED_STAGES = (
    "staging_and_zero_step_gate",
    "conditional_policy_training",
    "conditional_task_confirmation",
    "settlement_and_verified_export",
)


class TaskPipelineError(RuntimeError):
    """Frozen task worker contract violation."""


class _DirectBoundary:
    def environment(self, operation, *args, **kwargs):
        return operation(*args, **kwargs)

    model = environment
    optimizer = environment
    checkpoint = environment


class _RuntimeAdapter:
    def __init__(self, root: Path, matrix: Mapping[str, Any], boundary: Any,
                 output: Path):
        self.root = root
        self.matrix = matrix
        self.boundary = boundary
        self.output = output
        self.policy_checkpoint_hashes: dict[str, str] = {}


def _policy_route_budgets(matrix: Mapping[str, Any], *, allow_integration: bool) -> tuple[int, int, int, bool]:
    configuration = matrix.get("policy_configuration")
    if not isinstance(configuration, Mapping):
        raise TaskPipelineError("TASK_POLICY_CONFIGURATION_INVALID")
    steps = matrix.get("policy_training_steps_per_method_seed")
    rollout = configuration.get("rollout_steps")
    updates = configuration.get("maximum_optimizer_updates_per_method_seed")
    if any(type(value) is not int for value in (steps, rollout, updates)):
        raise TaskPipelineError("TASK_POLICY_BUDGET_INVALID")
    if (steps, rollout, updates) != (2048, 64, 128):
        raise TaskPipelineError("TASK_POLICY_BUDGET_MISMATCH")

    fixture = matrix.get("e2e_fixture")
    if fixture is None:
        fixture = {}
    if not isinstance(fixture, Mapping) or type(fixture.get("enabled", False)) is not bool:
        raise TaskPipelineError("TASK_FIXTURE_CONTRACT_INVALID")
    enabled = fixture.get("enabled", False)
    if not enabled:
        return steps, rollout, updates, False

    if not allow_integration:
        raise TaskPipelineError("TASK_INTEGRATION_FIXTURE_REJECTED")
    if set(fixture) != {
        "enabled", "mode", "policy_training_steps_per_method_seed",
        "policy_rollout_steps", "policy_optimizer_updates_per_method_seed",
    }:
        raise TaskPipelineError("TASK_FIXTURE_CONTRACT_INVALID")
    if fixture.get("mode") != "synthetic_test":
        raise TaskPipelineError("TASK_FIXTURE_MODE_INVALID")
    fixture_steps = fixture.get("policy_training_steps_per_method_seed")
    fixture_rollout = fixture.get("policy_rollout_steps")
    fixture_updates = fixture.get("policy_optimizer_updates_per_method_seed")
    if any(type(value) is not int for value in (fixture_steps, fixture_rollout, fixture_updates)):
        raise TaskPipelineError("TASK_FIXTURE_POLICY_BUDGET_INVALID")
    if (not 0 < fixture_steps < 2048
            or fixture_rollout <= 0
            or fixture_rollout != rollout
            or not 0 < fixture_updates < 128
            or fixture_steps % fixture_rollout
            or fixture_updates % (fixture_steps // fixture_rollout)):
        raise TaskPipelineError("TASK_FIXTURE_POLICY_BUDGET_INVALID")
    return fixture_steps, fixture_rollout, fixture_updates, True


def validate_task_configuration(matrix: Mapping[str, Any], request: Mapping[str, Any], *,
                                allow_integration: bool = False) -> None:
    if matrix.get("schema") != "w1-eawm-jepa-experiment-matrix/1.0.0":
        raise TaskPipelineError("TASK_MATRIX_SCHEMA_MISMATCH")
    if (matrix.get("methods") != ["G0", "T", "G1", "H"]
            or matrix.get("policy_seeds") != list(POLICY_SEEDS)
            or matrix.get("world_model_seeds") != list(WORLD_SEEDS.values())):
        raise TaskPipelineError("TASK_ROUTE_IDENTITY_MISMATCH")
    _route_steps, _rollout_steps, route_updates, _fixture_enabled = _policy_route_budgets(
        matrix, allow_integration=allow_integration,
    )
    if (matrix.get("world_model_device") != "cpu"
            or matrix.get("world_model_training_enabled") is not False
            or matrix.get("g2_training_enabled") is not False):
        raise TaskPipelineError("TASK_WORKER_CPU_SCOPE_MISMATCH")
    if matrix.get("task_episode_count") != 240 or matrix.get("task_episode_max_steps") != 18:
        raise TaskPipelineError("TASK_EPISODE_BUDGET_MISMATCH")
    splits = matrix.get("splits")
    repeats = matrix.get("repeats")
    if not isinstance(splits, Mapping) or not isinstance(repeats, Mapping):
        raise TaskPipelineError("TASK_SPLIT_CONTRACT_INVALID")
    if len(splits.get("train", ())) != 24:
        raise TaskPipelineError("TASK_TRAIN_PARENT_COUNT_MISMATCH")
    if (len(splits.get("task_confirmation", ())) != 8
            or repeats.get("task_confirmation") != 3):
        raise TaskPipelineError("TASK_CONFIRMATION_PARENT_REPEAT_MISMATCH")

    stages = request.get("stages")
    if not isinstance(stages, Mapping) or set(stages) != set(REQUIRED_STAGES):
        raise TaskPipelineError("TASK_STAGE_SET_MISMATCH")
    from recovery_contract import training_route_keys
    train_count = len(training_route_keys(matrix))
    expected_limits = {
        "conditional_policy_training": {
            "policy_optimizer_updates": train_count * route_updates,
            "checkpoint_writes": train_count,
        },
        "conditional_task_confirmation": {"task_episodes": 240,
                                           "task_comparison_calls": 1},
    }
    for stage, resources in expected_limits.items():
        if not isinstance(stages[stage], Mapping):
            raise TaskPipelineError(f"TASK_STAGE_LIMIT_INVALID:{stage}")
        for name, expected in resources.items():
            if stages[stage].get(name) != expected:
                raise TaskPipelineError(f"TASK_STAGE_LIMIT_MISMATCH:{stage}:{name}")
    totals = request.get("totals", {})
    if (totals.get("gpu_devices") != 0
            or totals.get("new_world_data_collection_steps") != 0
            or totals.get("world_model_training_updates") != 0
            or totals.get("world_optimizer_updates") != 0
            or totals.get("world_backward_calls") != 0):
        raise TaskPipelineError("TASK_ONLY_RESOURCE_SCOPE_MISMATCH")

    gates = matrix.get("task_gates")
    expected_gates = {
        "minimum_complete_parents": 8,
        "minimum_complete_repeats_per_parent": 3,
        "g1_minus_g0_macro_utility_strict_positive": True,
        "g1_minus_t_macro_utility_strict_positive": True,
        "g1_minus_h_macro_utility_minimum": 0.01,
        "minimum_noninferior_parents_vs_t": 6,
        "minimum_positive_policy_seeds_vs_t": 2,
        "cpu_mean_ms_maximum": 10,
        "wall_p95_ms_maximum": 50,
        "cost_status_must_be_evaluated": True,
        "bootstrap_parent_samples": 10000,
        "bootstrap_seed": 8401,
        "require_parent_bootstrap_lower_bound_positive_vs_t": True,
    }
    if not isinstance(gates, Mapping) or any(gates.get(key) != value for key, value in expected_gates.items()):
        raise TaskPipelineError("TASK_GATE_CONTRACT_MISMATCH")


def _validated_checkpoints(training: Mapping[str, Any], output: Path, *, root: Path | None = None,
                           expected_steps: int = 2048,
                           expected_updates: int = 128) -> tuple[dict[tuple[str, int], str], dict[str, str]]:
    routes = training.get("routes")
    checkpoints = training.get("checkpoints")
    expected = {(method, seed) for method in POLICY_METHODS for seed in POLICY_SEEDS}
    if (type(expected_steps) is not int or expected_steps <= 0
            or type(expected_updates) is not int or expected_updates <= 0):
        raise TaskPipelineError("POLICY_TRAINING_EXPECTED_BUDGET_INVALID")
    if not isinstance(routes, list) or len(routes) != len(expected) or not isinstance(checkpoints, Mapping):
        raise TaskPipelineError("POLICY_TRAINING_ROUTE_SET_INVALID")

    route_by_key = {}
    for route in routes:
        if not isinstance(route, Mapping):
            raise TaskPipelineError("POLICY_TRAINING_ROUTE_INVALID")
        method, seed = route.get("method"), route.get("seed")
        if method not in POLICY_METHODS or type(seed) is not int:
            raise TaskPipelineError("POLICY_TRAINING_ROUTE_IDENTITY_INVALID")
        key = (method, seed)
        if key not in expected or key in route_by_key:
            raise TaskPipelineError("POLICY_TRAINING_ROUTE_IDENTITY_INVALID")
        if (route.get("training_steps") != expected_steps
                or route.get("optimizer_updates") != expected_updates):
            raise TaskPipelineError(f"POLICY_TRAINING_ROUTE_BUDGET_MISMATCH:{method}:{seed}")
        world_variant = "G1" if method == "G1" else None
        world_seed = WORLD_SEEDS[seed] if method == "G1" else None
        if (route.get("world_model_variant") != world_variant
                or route.get("world_model_seed") != world_seed):
            raise TaskPipelineError(f"POLICY_TRAINING_WORLD_BINDING_MISMATCH:{method}:{seed}")
        route_by_key[key] = route
    if set(route_by_key) != expected or set(checkpoints) != expected:
        raise TaskPipelineError("POLICY_TRAINING_ROUTE_SET_MISMATCH")

    output_root = output.resolve()
    result: dict[tuple[str, int], str] = {}
    digest_by_path: dict[str, str] = {}
    for key in sorted(expected):
        route_path = Path(str(route_by_key[key].get("checkpoint", "")))
        mapped_path = Path(str(checkpoints[key]))
        if route_path != mapped_path:
            raise TaskPipelineError(f"POLICY_CHECKPOINT_ROUTE_MAP_MISMATCH:{key[0]}:{key[1]}")
        resolved = route_path.resolve()
        if root is not None and route_by_key[key].get('reuse') is True:
            from recovery_contract import validate_checkpoint_location
            resolved = validate_checkpoint_location(root, output, route_by_key[key])
        elif not resolved.is_relative_to(output_root) or not resolved.is_file():
            raise TaskPipelineError(f"POLICY_CHECKPOINT_PATH_INVALID:{key[0]}:{key[1]}")
        actual_digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        expected_digest = route_by_key[key].get("checkpoint_sha256")
        if not isinstance(expected_digest, str) or actual_digest != expected_digest:
            raise TaskPipelineError(f"POLICY_CHECKPOINT_DIGEST_MISMATCH:{key[0]}:{key[1]}")
        resolved_text = str(resolved)
        result[key] = resolved_text
        digest_by_path[resolved_text] = actual_digest
    return result, digest_by_path


def _metric_matches(production: Any, independent: Mapping[str, Any]) -> bool:
    if not isinstance(production, Mapping):
        return False
    for key in ("g1_minus_g0_macro_utility", "g1_minus_t_macro_utility",
                "g1_minus_h_macro_utility"):
        if key not in production or key not in independent:
            return False
        left, right = production[key], independent[key]
        if left is None or right is None:
            if left is not right:
                return False
            continue
        try:
            if isinstance(left, bool) or isinstance(right, bool):
                return False
            left_value, right_value = float(left), float(right)
            if not math.isfinite(left_value) or not math.isfinite(right_value):
                return False
            if abs(left_value - right_value) > 1e-8:
                return False
        except (OverflowError, TypeError, ValueError):
            return False
    return True


def _activity(output: Path, stage: str, operation: str) -> None:
    durable_atomic_json(output / "activity.json", {"stage": stage, "run": operation})


def _record_runtime_modules(root: Path, output: Path) -> dict[str, str]:
    names = (
        "runtime_hooks", "production_policy", "production_data", "production_world",
        "w1_graph_jepa", "gppo_world.graph5", "gppo_world.joint_consequence_baseline",
        "gppo_world.joint_gppo", "gppo_world.m10_environment",
    )
    paths = {}
    for name in names:
        module = importlib.import_module(name)
        source = Path(str(getattr(module, "__file__", ""))).resolve()
        try:
            relative = source.relative_to(root).as_posix()
        except ValueError as exc:
            raise TaskPipelineError(f"RUNTIME_MODULE_OUTSIDE_STAGED_PACKAGE:{name}") from exc
        if name.startswith("gppo_world.") and not source.is_relative_to(root / "native"):
            raise TaskPipelineError(f"NATIVE_MODULE_OUTSIDE_STAGED_PACKAGE:{name}")
        paths[name] = str(source)
        paths[name + ".relative"] = relative
    durable_atomic_json(output / "runtime-module-provenance.json", {
        "schema": "w1-task-runtime-module-provenance/1.0.0",
        "modules": paths,
        "capture_point": "before_conditional_policy_training",
        "provenance_probe_deserialized_checkpoint": False,
        "provenance_probe_constructed_environment": False,
    })
    return paths


def run_pipeline(root: Path, output: Path, matrix: Mapping[str, Any],
                 request: Mapping[str, Any], ledger: Any, *, boundary: Any | None = None,
                 clock: Any = time, allow_integration: bool = False) -> dict[str, Any]:
    """Train the nine policy routes, evaluate the frozen task matrix, and gate it."""
    root, output = Path(root).resolve(), Path(output).resolve()
    route_steps, _rollout_steps, route_updates, fixture_enabled = _policy_route_budgets(
        matrix, allow_integration=allow_integration,
    )
    validate_task_configuration(matrix, request, allow_integration=allow_integration)
    if fixture_enabled:
        context = getattr(boundary, "integration_context", {})
        if boundary is None or not isinstance(context, Mapping) or context.get("mode") != "synthetic_test":
            raise TaskPipelineError("SYNTHETIC_TEST_BOUNDARY_REQUIRED")

    from runtime_hooks import NativeRuntimeHooks
    native_boundary = _DirectBoundary() if boundary is None else boundary
    adapter = _RuntimeAdapter(root, matrix, native_boundary, output)
    hooks = NativeRuntimeHooks(adapter)
    _record_runtime_modules(root, output)

    _activity(output, "conditional_policy_training", "policy_route_training")
    ledger.select("conditional_policy_training")
    training = hooks.train_policy_routes(
        matrix=matrix, request=request, ledger=ledger, boundary=boundary,
        output_dir=output, clock=clock,
    )
    if not isinstance(training, Mapping):
        raise TaskPipelineError("POLICY_TRAINING_RESULT_INVALID")
    checkpoints, checkpoint_hashes = _validated_checkpoints(
        training, output, root=root, expected_steps=route_steps, expected_updates=route_updates,
    )
    adapter.policy_checkpoint_hashes = checkpoint_hashes
    durable_atomic_json(output / "policy-training-summary.json", {
        "schema": "w1-g1-gppo-policy-training-summary/1.0.0",
        "route_count": len(checkpoints),
        "routes": [{
            "method": method, "seed": seed,
            "world_model_variant": "G1" if method == "G1" else None,
            "world_model_seed": WORLD_SEEDS[seed] if method == "G1" else None,
            "training_steps": route_steps, "optimizer_updates": route_updates,
            "checkpoint": checkpoints[(method, seed)],
            "checkpoint_sha256": checkpoint_hashes[checkpoints[(method, seed)]],
            "reuse": next(r.get('reuse', False) for r in training['routes'] if (r['method'], r['seed']) == (method, seed)),
            "source_attempt": next(r.get('source_attempt') for r in training['routes'] if (r['method'], r['seed']) == (method, seed)),
        } for method, seed in sorted(checkpoints)],
        "checkpoint_sha256_by_path": checkpoint_hashes,
        "resource_amounts": dict(training.get("resource_amounts", {})),
        "automatic_retry": False,
    })
    del training

    _activity(output, "conditional_task_confirmation", "paired_task_confirmation")
    ledger.select("conditional_task_confirmation")
    evaluation = hooks.evaluate_task_confirmation(
        matrix=matrix, request=request, ledger=ledger, checkpoints=checkpoints,
        boundary=boundary, output_dir=output, clock=clock,
    )
    if not isinstance(evaluation, Mapping):
        raise TaskPipelineError("TASK_CONFIRMATION_RESULT_INVALID")
    expected_episodes = int(request["stages"]["conditional_task_confirmation"]["task_episodes"])
    if (evaluation.get("task_episodes") != expected_episodes
            or len(evaluation.get("episodes", ())) != expected_episodes
            or evaluation.get("shared_hungarian_episodes") != 24):
        raise TaskPipelineError("TASK_CONFIRMATION_EPISODE_COUNT_MISMATCH")
    evaluation_metrics = evaluation.get("metrics")

    from independent_task_recompute import evaluate_frozen_gates, recompute_from_files
    independent = recompute_from_files(output, matrix)
    if not isinstance(independent, Mapping) or not _metric_matches(evaluation_metrics, independent):
        raise TaskPipelineError("TASK_METRIC_IMPLEMENTATION_DISAGREEMENT")
    gate_result = evaluate_frozen_gates(matrix, independent)
    if not isinstance(gate_result, Mapping) or not {
            "status", "task_gate", "cost_gate", "research_success"}.issubset(gate_result):
        raise TaskPipelineError("TASK_GATE_RESULT_INVALID")
    status = gate_result.get("status")
    if status not in {"task_validation_pass", "task_gain_gate_stop", "cost_gate_stop"}:
        raise TaskPipelineError("TASK_GATE_STATUS_INVALID")
    durable_atomic_json(output / "task-gate.json", {
        "schema": "w1-g1-gppo-task-gate/1.0.0",
        **dict(gate_result),
        "automatic_retry": False,
    })
    durable_atomic_json(output / "task-pipeline-summary.json", {
        "schema": "w1-g1-gppo-task-pipeline-summary/1.0.0",
        "status": status,
        "task_episodes": expected_episodes,
        "policy_episodes": evaluation.get("policy_episodes"),
        "shared_hungarian_episodes": evaluation.get("shared_hungarian_episodes"),
        "complete_parent_count": independent.get("complete_parent_count"),
        "task_gate": gate_result["task_gate"],
        "cost_gate": gate_result["cost_gate"],
        "research_success": bool(gate_result["research_success"]),
        "independent_metrics_file": "independent-task-metrics.json",
        "automatic_retry": False,
    })
    return {
        "status": status,
        "task_gate": gate_result["task_gate"],
        "cost_gate": gate_result["cost_gate"],
        "research_success": bool(gate_result["research_success"]),
        "task_comparison_executed": True,
        "task_episodes": expected_episodes,
        "policy_episodes": evaluation.get("policy_episodes"),
        "shared_hungarian_episodes": evaluation.get("shared_hungarian_episodes"),
        "independent_metrics": independent,
        "automatic_retry": False,
    }


__all__ = ["TaskPipelineError", "run_pipeline", "validate_task_configuration"]
