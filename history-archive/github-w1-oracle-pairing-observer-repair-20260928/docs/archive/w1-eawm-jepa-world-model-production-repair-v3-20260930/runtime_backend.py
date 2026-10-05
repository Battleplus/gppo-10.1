"""Production backend for the frozen W1 EAWM/JEPA pipeline."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from budget_ledger import BudgetLedger
from infra_io import durable_append_jsonl, durable_atomic_json


PREDICTION_PARENT_COUNT = 8
JEPA_VARIANTS = ("G1", "G2")
WORLD_MODEL_SEEDS = (8201, 8202, 8203)
GPPO_METHODS = ("G0", "T", "G1", "G2")
POLICY_SEEDS = (8301, 8302, 8303)
TASK_EPISODE_COUNT = 312


class RuntimeBackendError(RuntimeError):
    """Raised when the frozen runtime contract cannot be honored."""


class BottomBoundary(Protocol):
    """Test seam limited to environment, model, optimizer and checkpoint calls."""

    def environment(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...
    def model(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...
    def optimizer(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...
    def checkpoint(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...


class DirectBoundary:
    def environment(self, operation, *args, **kwargs):
        return operation(*args, **kwargs)

    def model(self, operation, *args, **kwargs):
        return operation(*args, **kwargs)

    def optimizer(self, operation, *args, **kwargs):
        return operation(*args, **kwargs)

    def checkpoint(self, operation, *args, **kwargs):
        return operation(*args, **kwargs)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_production_source_tree(source_root: Path, manifest_files: Mapping[str, Any]) -> int:
    package_root = source_root / "native" / "gppo_world"
    if not package_root.is_dir():
        raise RuntimeBackendError("PRODUCTION_SOURCE_TREE_MISSING")
    expected = {
        name for name in manifest_files
        if isinstance(name, str) and name.startswith("native/gppo_world/") and name.endswith(".py")
    }
    actual = {
        path.relative_to(source_root).as_posix()
        for path in package_root.rglob("*.py") if path.is_file()
    }
    if not expected or actual != expected:
        raise RuntimeBackendError("PRODUCTION_SOURCE_TREE_FILE_SET_MISMATCH")
    for relative in sorted(expected):
        path = source_root / Path(relative)
        if _sha256(path) != manifest_files[relative]:
            raise RuntimeBackendError(f"PRODUCTION_SOURCE_TREE_DIGEST_MISMATCH:{relative}")
    return len(expected)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    durable_atomic_json(path, dict(value))


class ProductionRuntimeAdapter:
    """Hash-bound bridge to the repaired native W1 runtime."""

    def __init__(self, root: Path, output: Path, request: Mapping[str, Any],
                 matrix: Mapping[str, Any], ledger: BudgetLedger,
                 boundary: BottomBoundary | None = None,
                 hooks: Any | None = None):
        self.root = Path(root).resolve()
        self.output = Path(output).resolve()
        self.request = dict(request)
        self.matrix = dict(matrix)
        self.ledger = ledger
        self.boundary = boundary or DirectBoundary()
        self.modules: dict[str, Any] = {}
        self.models: dict[str, list[Any]] = {}
        self.phase = "created"
        self.prediction_passed = False
        self.world_training = None
        self.prediction_run = None
        self.policy_training = None
        self.hooks = hooks

    def _verify_runtime(self) -> dict[str, Any]:
        inputs_path = self.root / "runtime-inputs.json"
        if not inputs_path.is_file():
            raise RuntimeBackendError("RUNTIME_INPUTS_MISSING")
        inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
        source = inputs.get("source_run", {})
        source_root = Path(source.get("windows_root" if os.name == "nt" else "wsl_root", ""))
        if not source_root.is_absolute():
            raise RuntimeBackendError("RUNTIME_ROOT_NOT_ABSOLUTE")
        manifest_path = source_root / "execution-manifest.json"
        if not manifest_path.is_file() or _sha256(manifest_path) != source.get("execution_manifest_sha256"):
            raise RuntimeBackendError("RUNTIME_MANIFEST_DIGEST_MISMATCH")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_files = manifest.get("files")
        if not isinstance(manifest_files, dict):
            raise RuntimeBackendError("RUNTIME_MANIFEST_FILE_MAP_INVALID")
        source_tree_file_count = _verify_production_source_tree(source_root, manifest_files)
        expected_modules = inputs.get("production_modules", {})
        if not expected_modules:
            raise RuntimeBackendError("PRODUCTION_MODULE_HASHES_MISSING")
        for relative, expected in expected_modules.items():
            candidate = source_root / Path(relative)
            if not candidate.is_file() or _sha256(candidate) != expected:
                raise RuntimeBackendError(f"PRODUCTION_MODULE_DIGEST_MISMATCH:{relative}")
            if manifest_files.get(relative) != expected:
                raise RuntimeBackendError(f"MANIFEST_MODULE_IDENTITY_MISMATCH:{relative}")
        native_root = source_root / "native"
        if str(native_root) not in sys.path:
            sys.path.insert(0, str(native_root))
        for module_name in ("gppo_world.m10_environment", "gppo_world.graph5",
                            "gppo_world.joint_gppo", "gppo_world.joint_training"):
            self.modules[module_name] = importlib.import_module(module_name)
        return {"runtime_root": str(source_root),
                "execution_manifest_sha256": source["execution_manifest_sha256"],
                "verified_modules": sorted(expected_modules),
                "verified_source_tree_files": source_tree_file_count,
                "resource_request_status": manifest.get("resource_request_status")}

    def zero_step_gate(self) -> dict[str, Any]:
        self._validate_frozen_contract()
        verified = self._verify_runtime()
        self.phase = "zero_step_passed"
        return {"pass": True, "runtime": verified,
                "environment_constructed": False, "model_constructed": False,
                "optimizer_constructed": False, "checkpoint_loaded": False}

    def _validate_frozen_contract(self) -> None:
        if self.request.get("schema") != "w1-eawm-jepa-world-model-resource-request/2.0.0":
            raise RuntimeBackendError("RESOURCE_REQUEST_SCHEMA_MISMATCH")
        if self.request.get("status") != "NOT_APPROVED":
            raise RuntimeBackendError("RESOURCE_REQUEST_IDENTITY_CHANGED")
        if self.matrix.get("schema") != "w1-eawm-jepa-experiment-matrix/1.0.0":
            raise RuntimeBackendError("MATRIX_SCHEMA_MISMATCH")
        if self.matrix.get("task_episode_count") != TASK_EPISODE_COUNT:
            raise RuntimeBackendError("TASK_EPISODE_CONTRACT_MISMATCH")
        if tuple(self.matrix.get("methods", ())) != ("G0", "T", "G1", "G2", "H"):
            raise RuntimeBackendError("METHOD_ROUTE_CONTRACT_MISMATCH")
        expected = self.request.get("parent_split", {})
        split_counts = {name: len(self.matrix.get("splits", {}).get(name, ()))
                        for name in ("train", "model_selection", "prediction_confirmation", "task_confirmation")}
        if split_counts != {"train": 24, "model_selection": 8,
                            "prediction_confirmation": 8, "task_confirmation": 8}:
            raise RuntimeBackendError("PARENT_SPLIT_CONTRACT_MISMATCH")
        if (expected.get("train_parents"), expected.get("model_selection_parents"),
                expected.get("prediction_confirmation_parents"), expected.get("task_confirmation_parents")) != (24, 8, 8, 8):
            raise RuntimeBackendError("REQUEST_PARENT_SPLIT_CONTRACT_MISMATCH")
        routes = [(variant, seed) for variant in JEPA_VARIANTS for seed in WORLD_MODEL_SEEDS]
        if len(routes) != 6 or len(set(routes)) != 6:
            raise RuntimeBackendError("JEPA_ROUTE_CONTRACT_MISMATCH")
        policy_routes = [(method, seed) for method in GPPO_METHODS for seed in POLICY_SEEDS]
        if len(policy_routes) != 12 or len(set(policy_routes)) != 12:
            raise RuntimeBackendError("GPPO_ROUTE_CONTRACT_MISMATCH")
        task_splits = self.matrix.get("splits", {}).get("task_confirmation", ())
        repeats = int(self.matrix.get("repeats", {}).get("task_confirmation", 0))
        unit_keys = {
            (str(row.get("parent")), repeat)
            for row in task_splits for repeat in range(repeats)
        }
        if len(unit_keys) != 24:
            raise RuntimeBackendError("HUNGARIAN_SHARED_UNIT_CONTRACT_MISMATCH")

    def collect_world_model_data(self) -> dict[str, Any]:
        from production_data import ProductionDataCollector
        self.ledger.select("world_model_data_collection")
        collector = ProductionDataCollector(
            self.root, self.output, self.matrix, self.ledger, self.boundary,
        )
        result = collector.collect()
        if not isinstance(result, dict):
            raise RuntimeBackendError("WORLD_MODEL_COLLECTION_RESULT_INVALID")
        self.phase = "world_model_data_collected"
        _write_json(self.output / "world-model-data-summary.json", result)
        return result

    def train_world_models(self) -> dict[str, Any]:
        from production_world import load_complete_windows, train_select_world_models
        windows_path = self.output / "world-model-windows.jsonl"
        records = load_complete_windows(windows_path)
        self.ledger.select("world_model_training_and_selection")
        run = train_select_world_models(
            records, self.matrix, self.ledger, self.boundary,
            self.output,
        )
        self.world_training = run
        self.models = {f"world:{variant}:{seed}": model for (variant, seed), model in run.models.items()}
        result = {"routes": list(run.routes), "model_count": len(run.models),
                  "window_file": str(windows_path)}
        self.phase = "world_models_trained"
        return result

    def evaluate_world_models(self) -> dict[str, Any]:
        if self.world_training is None:
            raise RuntimeBackendError("WORLD_MODEL_EVALUATION_BEFORE_TRAINING")
        from production_world import evaluate_prediction_confirmation, load_complete_windows
        records = load_complete_windows(self.output / "world-model-windows.jsonl")
        confirmation = [row for row in records if row.get("split") == "prediction_confirmation"]
        self.ledger.select("prediction_confirmation")
        run = evaluate_prediction_confirmation(
            confirmation, self.matrix, self.world_training.routes,
            self.ledger, self.boundary, self.output,
        )
        self.prediction_run = run
        result = dict(run.metrics)
        result.update({"trace_path": str(run.trace_path), "metrics_path": str(run.metrics_path)})
        self.phase = "prediction_confirmed"
        return result

    def train_policies(self) -> dict[str, Any]:
        if self.prediction_run is None or not self.prediction_passed:
            raise RuntimeBackendError("POLICY_TRAINING_BEFORE_PREDICTION_GATE")
        if self.hooks is None:
            from runtime_hooks import NativeRuntimeHooks
            self.hooks = NativeRuntimeHooks(self)
        from production_policy import train_policy_routes
        self.ledger.select("conditional_policy_training")
        result = train_policy_routes(
            matrix=self.matrix, request=self.request, ledger=self.ledger,
            boundary=self.boundary, world_model_loader=self.hooks.world_model_loader,
            policy_factory=self.hooks.policy_factory,
            optimizer_factory=self.hooks.optimizer_factory,
            route_runner=self.hooks.route_runner,
            ppo_update_fn=self.hooks.ppo_update_fn,
            checkpoint_writer=self.hooks.checkpoint_writer,
            input_builder=self.hooks.input_builder,
            transparent_scorer=self.hooks.transparent_scorer,
            world_predictor=self.hooks.world_predictor,
            output_dir=self.output,
        )
        self.policy_training = result
        self.models.update({f"policy:{method}:{seed}": model for (method, seed), model in result["policies"].items()})
        self.phase = "policies_trained"
        return {key: value for key, value in result.items() if key not in {"policies", "frozen_world_models"}}

    def evaluate_tasks(self) -> dict[str, Any]:
        if self.policy_training is None:
            raise RuntimeBackendError("TASK_EVALUATION_BEFORE_POLICY_TRAINING")
        if self.hooks is None:
            from runtime_hooks import NativeRuntimeHooks
            self.hooks = NativeRuntimeHooks(self)
        from production_policy import evaluate_task_confirmation
        self.ledger.select("conditional_task_confirmation")
        result = evaluate_task_confirmation(
            matrix=self.matrix, request=self.request, ledger=self.ledger,
            boundary=self.boundary, policy_loader=self.hooks.policy_loader,
            world_model_loader=self.hooks.world_model_loader,
            episode_runner=self.hooks.episode_runner,
            input_builder=self.hooks.input_builder,
            transparent_scorer=self.hooks.transparent_scorer,
            world_predictor=self.hooks.world_predictor,
            checkpoints=self.policy_training["checkpoints"], output_dir=self.output,
        )
        self.phase = "tasks_evaluated"
        return result


class W1RuntimeBackend:
    """Production PipelineBackend with stage accounting and exception settlement."""

    def __init__(self, root: Path, output: Path, request: Mapping[str, Any] | None = None,
                 matrix: Mapping[str, Any] | None = None, *,
                 adapter_factory: Callable[..., ProductionRuntimeAdapter] | None = None,
                 boundary: BottomBoundary | None = None):
        self.root = Path(root).resolve()
        self.output = Path(output).resolve()
        if self.output.exists():
            raise RuntimeBackendError("OUTPUT_ALREADY_EXISTS_NO_RETRY")
        self.output.mkdir(parents=True, exist_ok=False)
        self.request = dict(request) if request is not None else json.loads((self.root / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
        self.matrix = dict(matrix) if matrix is not None else json.loads((self.root / "experiment-matrix.json").read_text(encoding="utf-8"))
        self.ledger = BudgetLedger(self.output / "budget.sqlite3", self.request)
        self.task_calls = 0
        self.settled = False
        self.adapter = (adapter_factory or ProductionRuntimeAdapter)(
            self.root, self.output, self.request, self.matrix, self.ledger, boundary)

    def _stage(self, stage: str) -> None:
        if stage not in self.request.get("stages", {}):
            raise RuntimeBackendError(f"UNKNOWN_STAGE:{stage}")
        self.ledger.select(stage)
        _write_json(self.output / "activity.json", {"stage": stage, "backend": "production"})

    def _invoke(self, stage: str, method: str) -> dict[str, Any]:
        self._stage(stage)
        result = getattr(self.adapter, method)()
        if not isinstance(result, dict):
            raise RuntimeBackendError(f"INVALID_STAGE_RESULT:{method}")
        return result

    def zero_step_gate(self) -> dict[str, Any]:
        return self._invoke("staging_and_zero_step_gate", "zero_step_gate")

    def collect_world_model_data(self) -> dict[str, Any]:
        result = self._invoke("world_model_data_collection", "collect_world_model_data")
        if result.get("pass") and int(result.get("prediction_parent_count", 0)) != PREDICTION_PARENT_COUNT:
            raise RuntimeBackendError("PREDICTION_PARENT_COVERAGE_MISMATCH")
        return result

    def train_world_models(self) -> dict[str, Any]:
        result = self._invoke("world_model_training_and_selection", "train_world_models")
        routes = result.get("routes", ())
        expected = {(variant, seed) for variant in JEPA_VARIANTS for seed in WORLD_MODEL_SEEDS}
        observed = {(str(row.get("variant")), int(row.get("seed"))) for row in routes}
        if observed != expected or len(routes) != 6:
            raise RuntimeBackendError("JEPA_TRAINING_ROUTE_SET_MISMATCH")
        return result

    def evaluate_world_models(self) -> dict[str, Any]:
        result = self._invoke("prediction_confirmation", "evaluate_world_models")
        self.prediction_passed = bool(result.get("gate", {}).get("pass"))
        return result

    def train_policies(self) -> dict[str, Any]:
        if not self.prediction_passed:
            raise RuntimeBackendError("POLICY_TRAINING_BEFORE_PREDICTION_GATE")
        result = self._invoke("conditional_policy_training", "train_policies")
        routes = result.get("routes", ())
        expected = {(method, seed) for method in GPPO_METHODS for seed in POLICY_SEEDS}
        observed = {(str(row.get("method")), int(row.get("seed"))) for row in routes}
        if observed != expected or len(routes) != 12:
            raise RuntimeBackendError("GPPO_TRAINING_ROUTE_SET_MISMATCH")
        return result

    def evaluate_tasks(self) -> dict[str, Any]:
        if not self.prediction_passed or not self.adapter.models:
            raise RuntimeBackendError("TASK_COMPARISON_BEFORE_POLICY_GATE_OR_TRAINING")
        self.task_calls += 1
        result = self._invoke("conditional_task_confirmation", "evaluate_tasks")
        if int(result.get("task_episodes", 0)) != TASK_EPISODE_COUNT:
            raise RuntimeBackendError("TASK_EPISODE_COUNT_MISMATCH")
        return result

    def settle(self, status: str, evidence: dict[str, Any]) -> None:
        if self.settled:
            raise RuntimeBackendError("DUPLICATE_SETTLEMENT")
        self._stage("settlement_and_verified_export")
        snapshot = self.ledger.assert_settled()
        settlement = {"status": status, "task_calls": self.task_calls,
                      "ledger": snapshot, "automatic_retry": False,
                      "automatic_extension": False, "evidence": evidence}
        _write_json(self.output / "settlement.json", settlement)
        self.settled = True

    def settle_exception(self, exc: BaseException, evidence: Mapping[str, Any] | None = None) -> None:
        if self.settled:
            return
        try:
            self._stage("settlement_and_verified_export")
            snapshot = self.ledger.snapshot()
            _write_json(self.output / "settlement.json", {
                "status": "exception_stop", "exception_type": type(exc).__name__,
                "exception": str(exc), "task_calls": self.task_calls,
                "ledger": snapshot, "automatic_retry": False,
                "automatic_extension": False, "evidence": dict(evidence or {}),
            })
        finally:
            self.settled = True

    def close(self) -> None:
        self.ledger.close()


__all__ = ["BottomBoundary", "DirectBoundary", "ProductionRuntimeAdapter",
           "RuntimeBackendError", "W1RuntimeBackend"]
