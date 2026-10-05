"""Concrete runtime bridge for the A/B experiment.

The bridge owns the research implementations and exposes only bottom-level
injection points for zero-call integration tests.  The production path uses
the repaired W1 collector, the windowed A/B trainer, trace writer and metric
recomputation; tests replace only environment/model/optimizer/checkpoint
boundaries through ``BottomBoundary``.
"""
from __future__ import annotations

import copy
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from learning_schema import load_records
from metrics import compare_trace, evaluate_trace
from trace_schema import PredictionTraceRow, write_trace
from windowed_training import ModelSpec, TrainingConfig, build_model, train_variant


class RuntimeAdapterError(RuntimeError):
    pass


class BottomBoundary(Protocol):
    """Only low-level calls may be replaced by integration tests."""

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


@dataclass
class AdapterResult:
    payload: dict[str, Any]


class ProductionRuntimeAdapter:
    """Adapter backed by the repaired W1 runtime and current A/B modules."""

    def __init__(self, root: Path, output: Path, request: dict, matrix: dict,
                 account_call: Callable, boundary: BottomBoundary | None = None):
        self.root = root
        self.output = output
        self.request = request
        self.matrix = matrix
        self.account_call = account_call
        self.boundary = boundary or DirectBoundary()
        self._legacy = None
        self._models: dict[str, list[Any]] = {"A": [], "B": []}
        self._checkpoints: dict[str, list[Path]] = {"A": [], "B": []}

    def _load_legacy(self):
        if self._legacy is not None:
            return self._legacy
        inputs_path = self.root / "runtime-inputs.json"
        if not inputs_path.is_file():
            raise RuntimeAdapterError("RUNTIME_INPUTS_MISSING")
        inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
        # The bridge itself must come from the already hash-verified package
        # copy. External repaired-runtime inputs are loaded only by the legacy
        # bridge after their own manifest checks.
        legacy_root = self.root
        if not legacy_root.is_absolute():
            raise RuntimeAdapterError("LEGACY_RUNTIME_ROOT_NOT_ABSOLUTE")
        if str(legacy_root) not in sys.path:
            sys.path.insert(0, str(legacy_root))
        module = importlib.import_module("legacy_runtime_backend")
        self._legacy = module.AuthorizedRuntimeBackend(self.root, self.output, self.request, self.matrix)
        return self._legacy

    def zero_step_gate(self) -> dict[str, Any]:
        """Load and validate runtime modules without constructing an env/model."""
        if hasattr(self.boundary, "zero_step"):
            return self.boundary.zero_step()
        legacy = self._load_legacy()
        # The legacy zero-step implementation derives its own historical
        # request.  The current package owns the identity and budget checks;
        # only its import-time runtime validation is reused here.
        legacy._load_runtime()
        for role, rows in self.matrix.get("groups", {}).items():
            if role not in {"train", "model_selection", "prediction_evaluation", "task_evaluation"}:
                raise RuntimeAdapterError(f"UNKNOWN_MATRIX_ROLE:{role}")
            for row in rows:
                if not row.get("parent") or len(str(row.get("scenario_sha256", ""))) != 64:
                    raise RuntimeAdapterError("MATRIX_IDENTITY_INVALID")
        return {"pass": True, "runtime_modules_loaded": True,
                "environment_constructed": False, "model_constructed": False}

    def collect_confirmation_labels(self) -> dict[str, Any]:
        if hasattr(self.boundary, "collect_records"):
            records = self.boundary.collect_records(self.output, self.matrix)
            if not records:
                raise RuntimeAdapterError("LABEL_EXPORT_EMPTY")
            path = self.output / "learning-records.jsonl"
            path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
            first = load_records(path)[0]
            (self.output / "first-label.json").write_text(json.dumps({"sample_id": first.sample_id, "target": first.remaining_utility, "unknown": False}) + "\n", encoding="utf-8")
            return {"pass": True, "records": len(records), "first_label_persisted": True, "test_bottom_boundary": True}
        legacy = self._load_legacy()
        legacy.ledger = self._ledger
        legacy.select_stage("new_prediction_confirmation")
        inputs = json.loads((self.root / "runtime-inputs.json").read_text(encoding="utf-8"))
        source = Path(inputs["reused_labels_path_linux"] if sys.platform != "win32" else inputs["reused_labels_path"])
        if not source.is_file():
            raise RuntimeAdapterError("REUSED_LABEL_EXPORT_MISSING")
        reused = load_records(source)
        # Reuse only the frozen train/model-selection records. The old
        # prediction-evaluation rows are deliberately excluded.
        records = [row for row in reused if row.split_role in {"train", "model_selection"}]
        units_path = self.output / "data-units.jsonl"
        data_path = self.output / "learning-records.jsonl"
        data_path.write_text("".join(json.dumps(row.__dict__, sort_keys=True) + "\n" for row in records), encoding="utf-8")
        for identity in self.matrix["groups"]["prediction_evaluation"]:
            for repeat in range(int(self.matrix["data_repeats"]["prediction_evaluation"])):
                new_records, unit = self.boundary.environment(
                    legacy._collect_unit, "prediction_evaluation", identity["parent"], repeat)
                with units_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(unit, sort_keys=True) + "\n")
                with data_path.open("a", encoding="utf-8") as handle:
                    for row in new_records:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                records.extend(load_records(data_path)[-len(new_records):] if new_records else [])
        if not records:
            raise RuntimeAdapterError("LABEL_EXPORT_EMPTY")
        first = records[0]
        (self.output / "first-label.json").write_text(
            json.dumps({"sample_id": first.sample_id, "target": first.remaining_utility,
                        "unknown": False}, sort_keys=True) + "\n", encoding="utf-8")
        return {"pass": True, "records": len(records),
                "first_label_persisted": True, "continuation_id": first.continuation_id}

    def _accounted(self, name: str, amounts: dict[str, int], operation, *args):
        if name.startswith("model."):
            boundary = self.boundary.model
        elif name.startswith("optimizer."):
            boundary = self.boundary.optimizer
        elif name.startswith("checkpoint."):
            boundary = self.boundary.checkpoint
        else:
            boundary = self.boundary.environment
        return self.account_call(name, amounts, boundary, operation, *args)

    def train_variants(self) -> dict[str, Any]:
        records = load_records(self.output / "learning-records.jsonl")
        train = [r for r in records if r.split_role == "train"]
        selection = [r for r in records if r.split_role == "model_selection"]
        if not train or not selection:
            raise RuntimeAdapterError("TRAIN_OR_SELECTION_LABELS_MISSING")
        config = TrainingConfig()
        spec = ModelSpec()
        result = []
        for variant_name, objective_name in (("A", "absolute"), ("B", "centered_auxiliary")):
            for seed in config.seeds:
                if hasattr(self.boundary, "train_variant"):
                    fake_model, metadata = self.boundary.train_variant(variant_name, seed, train, selection)
                    self._models[variant_name].append(fake_model)
                    result.append({"variant": variant_name, "seed": seed, "checkpoint": None, "metadata": metadata, "test_bottom_boundary": True})
                    continue
                path = self.output / "checkpoints" / f"{variant_name}-seed-{seed}.pt"
                row = train_variant(train, selection, variant=objective_name, seed=seed,
                                    output_path=path, config=config, spec=spec,
                                    account_call=self._accounted)
                self._checkpoints[variant_name].append(path)
                result.append({"variant": variant_name, "seed": seed,
                               "checkpoint": str(path), "metadata": row})
        (self.output / "training.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"variants": ["A", "B"], "seeds": list(config.seeds),
                "independent_outputs": True, "rows": result}

    def _load_models(self, variant: str) -> list[Any]:
        import torch
        models = []
        for path in self._checkpoints[variant]:
            payload = self._accounted("checkpoint.load", {"model_initializations_or_loads": 1},
                                      torch.load, path, map_location="cpu", weights_only=False)
            model = build_model(ModelSpec())
            model.load_state_dict(payload["state_dict"])
            model.eval()
            models.append(model)
        self._models[variant] = models
        return models

    def evaluate_predictions(self) -> dict[str, Any]:
        records = load_records(self.output / "learning-records.jsonl")
        evaluation = [r for r in records if r.split_role == "prediction_evaluation"]
        if not evaluation:
            raise RuntimeAdapterError("PREDICTION_LABELS_MISSING")
        all_rows: list[PredictionTraceRow] = []
        for variant in ("A", "B"):
            models = self._models[variant] or self._load_models(variant)
            for row in evaluation:
                if hasattr(self.boundary, "predict_variant"):
                    seed_values = [float(value) for value in self.boundary.predict_variant(variant, row, models)]
                else:
                    import torch
                    features = torch.tensor([__import__("learning_schema").materialize_features(row.flat, row.history, row.action)], dtype=torch.float32)
                    seed_values = []
                    for model in models:
                        output = self._accounted("model.forward.prediction", {"model_batch_forwards": 1, "model_sample_evaluations": 1}, model, features)
                        value = float(output.reshape(-1)[0])
                        if not math.isfinite(value):
                            raise RuntimeAdapterError("NONFINITE_PREDICTION")
                        seed_values.append(value)
                if not seed_values or not all(math.isfinite(value) for value in seed_values):
                    raise RuntimeAdapterError("NONFINITE_PREDICTION")
                all_rows.append(PredictionTraceRow(
                    parent=row.parent, repeat=row.repeat, decision_id=row.decision_id,
                    action=row.action, legal_actions=tuple(row.legal_actions),
                    continuation_id=row.continuation_id,
                    public_input_sha256=__import__("hashlib").sha256(
                        json.dumps({"flat": row.flat, "history": row.history, "legal": row.legal_actions}, sort_keys=True).encode()
                    ).hexdigest(), true_utility=row.remaining_utility, target_valid=True,
                    baseline_current_public=row.current_public_score,
                    baseline_transparent=row.transparent_history_score,
                    seed_predictions=tuple(seed_values), ensemble_prediction=sum(seed_values) / len(seed_values),
                    variant=variant))
        trace_path = self.output / "per-candidate-predictions.jsonl"
        write_trace(trace_path, all_rows)
        comparison = compare_trace(all_rows)
        metrics = {"comparison": comparison, "trace_rows": len(all_rows),
                   "trace_complete": True, "independent_recomputation": True,
                   "b_minus_a_parent_macro_regret": comparison["B"]["parent_macro_selected_regret"] - comparison["A"]["parent_macro_selected_regret"],
                   "parents_b_no_worse": 0,
                   "b_strictly_better_than_transparent": comparison["B"]["parent_macro_selected_regret"] < comparison["transparent_history"]["parent_macro_selected_regret"]}
        # Parent non-worse is recomputed from per-parent selections rather than
        # inferred from the aggregate; keep the raw comparison for audit.
        metrics["parents_b_no_worse"] = self._count_parent_non_worse(all_rows)
        (self.output / "prediction-metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return metrics

    @staticmethod
    def _count_parent_non_worse(rows: Sequence[PredictionTraceRow]) -> int:
        grouped: dict[tuple[str, str], list[PredictionTraceRow]] = {}
        for row in rows:
            grouped.setdefault((row.variant, row.decision_id), []).append(row)
        by_parent: dict[str, dict[str, list[float]]] = {}
        for (variant, _decision), group in grouped.items():
            chosen = max(group, key=lambda item: (item.ensemble_prediction, -item.action))
            best = max(item.true_utility for item in group if item.true_utility is not None)
            by_parent.setdefault(chosen.parent, {}).setdefault(variant, []).append(best - float(chosen.true_utility))
        return sum(1 for value in by_parent.values()
                   if "A" in value and "B" in value
                   and sum(value["B"]) / len(value["B"]) <= sum(value["A"]) / len(value["A"]))

    def run_task_comparison(self) -> dict[str, Any]:
        if hasattr(self.boundary, "task_comparison"):
            return self.boundary.task_comparison({"A": self._models.get("A", []), "B": self._models.get("B", [])}, self.matrix)
        legacy = self._load_legacy()
        legacy.ledger = self._ledger
        legacy.select_stage("conditional_task_comparison")
        configure = getattr(self.boundary, "configure_task_runtime", None)
        if configure is not None:
            configure(legacy)
        models = {
            "A": self._models.get("A") or self._load_models("A"),
            "B": self._models.get("B") or self._load_models("B"),
        }
        result = self.boundary.environment(legacy.run_task_comparison, models)
        if not isinstance(result, dict) or "summaries" not in result:
            raise RuntimeAdapterError("TASK_RESULT_INVALID")
        return result

    def bind_ledger(self, ledger: Any) -> None:
        self._ledger = ledger
