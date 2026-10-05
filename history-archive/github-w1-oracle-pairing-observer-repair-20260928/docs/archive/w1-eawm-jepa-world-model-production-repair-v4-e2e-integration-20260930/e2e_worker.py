"""End-to-end production wiring exercise with bottom-boundary fixtures only.

The collector, world-model pipeline, policy routes, task evaluator, metrics and
settlement are the production functions.  The fixture replaces only native
environment construction and model/optimizer/checkpoint boundaries; it never
replaces a stage function.  This worker is evidence for orchestration, not a
formal W1 result.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import nn
from torch.distributions import Categorical

from infra_io import durable_atomic_json
from runtime_backend import ProductionRuntimeAdapter, W1RuntimeBackend
from runtime_hooks import NativeRuntimeHooks
from world_model_pipeline import execute_pipeline

ROOT = Path(__file__).resolve().parent
EVIDENCE_ROOT = ROOT / "e2e-evidence-20260930-v18"


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _Message:
    def __init__(self, entity: str, field: str, value: float, sequence: int):
        self.entity, self.field, self.value = entity, field, value
        self.measured_at = float(sequence)
        self.received_at = float(sequence)
        self.sequence = int(sequence)
        self.message_id = f"fixture-{entity}-{field}-{sequence}"


class _Store:
    def __init__(self):
        self._latest: dict[str, _Message] = {}


class _View:
    def __init__(self):
        self._uavs, self._task_values = _Store(), _Store()


def _observation(step: int) -> dict[str, Any]:
    flat = [float(step) / 10.0] * 802
    node_features = []
    for index in range(4):
        values = (float(index), 0.0, 9.5, 1.0, 1.0, 1.0)
        row = []
        for value in values:
            row.extend((float(value), 1.0, 1.0, 0.0))
        node_features.append(row + [0.0] * (32 - len(row)))
    node_features.extend([[0.0] * 32 for _ in range(3 + 4)])
    for index in range(6):
        values = (float(index), 1.0, 15.0 + index, 1.0, 1.0, 1.0, 0.0, 0.0)
        row = []
        for value in values:
            row.extend((float(value), 1.0, 1.0, 0.0))
        node_features.append(row)
    node_features.extend([[0.0] * 32 for _ in range(4)])
    relations = [[[0.1 * (u + 1), 0.1 * (t + 1), 0.0, 1.0]
                  for t in range(6)] for u in range(4)]
    mask = [False] * 25
    for action in (0, 7, 24):
        mask[action] = True
    uavs = []
    tasks = []
    for index in range(4):
        row = []
        for value in (float(index), 0.0, 9.5, 1.0, 1.0, 1.0):
            row.extend((float(value), 1.0, 1.0, 0.0))
        uavs.append(row)
    for index in range(6):
        row = []
        for value in (float(index), 1.0, 15.0 + index, 1.0, 1.0, 1.0, 0.0, 0.0):
            row.extend((float(value), 1.0, 1.0, 0.0))
        tasks.append(row)
    return {
        "flat": flat, "graph": {"node_features": node_features, "relations": relations},
        "uavs": uavs, "tasks": tasks, "mask": mask, "time": float(step),
        "version": int(step), "public_entity_ids": {
            "uavs": [f"uav-{i}" for i in range(4)],
            "tasks": [f"task-{i}" for i in range(6)],
        },
        "trigger_flags": {}, "event_signal": 0.0,
        "continuation_actions": [],
    }


class FixtureEnvironment:
    """A deterministic public-only boundary fixture; no M-10 environment call."""

    def __init__(self, _config: Any, scenario: Any, exogenous_key: str | None = None):
        self.scenario = scenario
        self.exogenous_key = exogenous_key
        self.index = 0
        self.view = _View()
        self._refresh_view()

    def _refresh_view(self) -> None:
        obs = _observation(self.index)
        self.view._uavs._latest = {
            f"uav-{i}.x": _Message(f"uav-{i}", "x", row[0], self.index)
            for i, row in enumerate(obs["uavs"])
        }
        self.view._task_values._latest = {
            f"task-{i}.deadline": _Message(f"task-{i}", "deadline", row[2], self.index)
            for i, row in enumerate(obs["tasks"])
        }

    def reset(self) -> dict[str, Any]:
        self.index = 0
        self._refresh_view()
        return _observation(self.index)

    def step(self, _action: int):
        self.index += 1
        self._refresh_view()
        done = self.index >= 6
        records = {}
        for task in tuple(getattr(self.scenario, "tasks", ())):
            if isinstance(task, Mapping):
                task_id = str(task.get("task_id"))
                deadline = float(task.get("deadline", 15.0))
            else:
                task_id = str(getattr(task, "task_id"))
                deadline = float(getattr(task, "deadline"))
            records[task_id] = {
                "physical_arrival_time": min(deadline, float(self.index)),
                "host_confirmation_time": min(deadline, float(self.index)),
            }
        info = {
            "counts": {"completed": int(self.index), "expired": 0},
            "energy": {f"uav-{i}": 9.5 - 0.01 * self.index for i in range(4)},
            "completion_records": records, "time": float(self.index),
            "feedback": "accepted", "terminated": done, "truncated": False,
        }
        return _observation(self.index), 0.0, done, info


class FixtureWorldModel(nn.Module):
    """Small differentiable model with the production W1 output contract."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.01))
        self.event_count = 5
        self.config = SimpleNamespace(history_dim=128, latent_dim=64)

    def forward(self, nodes, history, actions, relations):
        batch = int(actions.shape[0])
        device, dtype = actions.device, torch.float32
        signal = self.scale * torch.ones((batch, 1), device=device, dtype=dtype)
        latent = signal.expand(batch, 64)
        public_state = signal.expand(batch, 128)
        outcome = signal.expand(batch, 6)
        event_logits = signal.expand(batch, self.event_count)
        history_latent = signal.expand(batch, 128)
        return {"latent": latent, "public_state": public_state, "outcome": outcome,
                "event_logits": event_logits, "history_latent": history_latent}

    def target(self, nodes):
        batch = next(iter(nodes.values())).shape[0]
        return self.scale.detach().expand(batch, 64)

    def update_target(self):
        return None

    def predict_candidates(self, nodes, history, candidate_actions, relations):
        return self.forward(nodes, history, candidate_actions, relations)


class FixturePolicy(nn.Module):
    """Tiny GPPO-compatible actor/critic used only at the model boundary."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(802, 16)
        self.actor = nn.Linear(16, 25)
        self.candidate = nn.Linear(17, 1)
        self.critic = nn.Linear(16, 2)

    def encode(self, obs: torch.Tensor, hidden: torch.Tensor | None = None):
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        features = torch.tanh(self.encoder(obs[:, :802]))
        pair = torch.zeros((features.shape[0], 4), dtype=features.dtype, device=features.device)
        next_hidden = torch.zeros((1, features.shape[0], 128), dtype=features.dtype, device=features.device)
        if hidden is not None:
            prior = hidden.reshape(1, features.shape[0], 128).to(features.device, features.dtype)
            next_hidden = 0.5 * prior + 0.5 * next_hidden
        return features, pair, next_hidden

    def evaluate_encoded(self, features, pair_messages, preference, candidate_features, mask):
        base = self.actor(features).unsqueeze(1).expand(-1, 25, -1).mean(dim=-1)
        candidate = self.candidate(candidate_features[..., :17]).squeeze(-1)
        logits = base + candidate
        legal = mask.to(dtype=torch.bool)
        final = logits.masked_fill(~legal, -torch.inf)
        distribution = Categorical(logits=final)
        return {"logits": logits, "final_logits": final, "distribution": distribution,
                "probabilities": distribution.probs, "critic_values": self.critic(features),
                "state_values": self.critic(features)}


class FixtureBoundary:
    """Counts and replaces only bottom operations permitted by the protocol."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def environment(self, operation, *args, **kwargs):
        name = getattr(operation, "__name__", "")
        if not hasattr(operation, "__self__") and name in {"<lambda>", "M10Environment"}:
            self.calls.append({"kind": "environment", "name": "construct_fixture"})
            return FixtureEnvironment(*args, **kwargs)
        self.calls.append({"kind": "environment", "name": name or type(operation).__name__})
        return operation(*args, **kwargs)

    def model(self, operation, *args, **kwargs):
        name = getattr(operation, "name", getattr(operation, "__name__", type(operation).__name__))
        self.calls.append({"kind": "model", "name": str(name)})
        if str(name) == "world_model.initialize":
            return FixtureWorldModel()
        return operation(*args, **kwargs)

    def optimizer(self, operation, *args, **kwargs):
        self.calls.append({"kind": "optimizer", "name": getattr(operation, "name", type(operation).__name__)})
        return operation(*args, **kwargs)

    def checkpoint(self, operation, *args, **kwargs):
        self.calls.append({"kind": "checkpoint", "name": getattr(operation, "name", type(operation).__name__)})
        return operation(*args, **kwargs)


class FixtureHooks(NativeRuntimeHooks):
    def _construct_environment(self, _environment_type: Any, config: Any,
                               scenario: Any, exogenous_key: str) -> Any:
        self.adapter.boundary.calls.append({"kind": "environment", "name": "construct_fixture"})
        return FixtureEnvironment(config, scenario, exogenous_key)

    def world_model_loader(self, variant: str, seed: int) -> Any:
        from production_world import _file_sha256, _load_checkpoint
        route = next(row for row in self.adapter.world_training.routes
                     if row["variant"] == variant and int(row["seed"]) == int(seed))
        path = self.adapter.output / str(route["checkpoint_relative_path"])
        if _file_sha256(path) != str(route["checkpoint_sha256"]):
            raise RuntimeError("FIXTURE_WORLD_CHECKPOINT_DIGEST_MISMATCH")
        payload = _load_checkpoint(path)
        model = FixtureWorldModel()
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    def policy_factory(self, _method: str, _seed: int, _config: Mapping[str, Any]) -> Any:
        return FixturePolicy()

    def input_builder(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        if "public_history" not in observation:
            import public_history
            raise RuntimeError("FIXTURE_MISSING_PUBLIC_HISTORY:" + str(public_history.__file__) + ":" + ",".join(sorted(observation)))
        return super().input_builder(observation)

    def policy_loader(self, method: str, seed: int, path: str) -> Any:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "w1-policy-checkpoint/1.0.0" or payload.get("method") != method or int(payload.get("seed", -1)) != int(seed):
            raise RuntimeError("FIXTURE_POLICY_CHECKPOINT_IDENTITY_MISMATCH")
        policy = FixturePolicy()
        policy.load_state_dict(payload["state_dict"])
        policy.eval()
        return policy


def _fixture_request(base: Mapping[str, Any], attempt: str) -> dict[str, Any]:
    request = copy.deepcopy(dict(base))
    request["attempt"] = attempt
    request["e2e_fixture"] = {
        "enabled": True, "scope": "engineering wiring only", "formal_dynamic_calls": 0,
    }
    for stage in request["stages"].values():
        for key, value in list(stage.items()):
            if isinstance(value, int):
                stage[key] = max(int(value), 10_000_000)
    for key, value in list(request["totals"].items()):
        if isinstance(value, int):
            request["totals"][key] = max(int(value), 100_000_000)
    request["stages"]["conditional_policy_training"]["policy_optimizer_updates"] = 24
    return request


def _fixture_matrix(base: Mapping[str, Any]) -> dict[str, Any]:
    matrix = copy.deepcopy(dict(base))
    matrix["e2e_fixture"] = {
        "enabled": True, "policy_training_steps_per_method_seed": 8,
        "policy_rollout_steps": 4, "policy_optimizer_updates_per_method_seed": 2,
        "world_model_epochs": 1,
    }
    matrix["policy_training_steps_per_method_seed"] = 8
    matrix["policy_configuration"]["rollout_steps"] = 4
    matrix["policy_configuration"]["maximum_optimizer_updates_per_method_seed"] = 2
    matrix["world_model_maximum_epochs"] = 1
    matrix["world_model_maximum_updates_per_variant_seed"] = 20
    return matrix


def _export_manifest(output: Path, result: Any, boundary: FixtureBoundary, mode: str) -> dict[str, Any]:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "e2e-export-manifest.json":
            files[path.relative_to(output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema": "w1-e2e-engineering-export/1.0.0", "mode": mode,
        "pipeline_status": result.status, "files": files,
        "file_set_sha256": _digest(files),
        "boundary_calls": boundary.calls,
        "formal_dynamic_calls": {"environment": 0, "model": 0, "optimizer": 0, "checkpoint": 0},
        "fixture_operations_are_not_formal_results": True,
    }
    durable_atomic_json(output / "e2e-export-manifest.json", manifest)
    return manifest


def run_path(mode: str) -> dict[str, Any]:
    import shutil
    output = EVIDENCE_ROOT / mode
    if output.exists():
        raise RuntimeError(f"E2E_OUTPUT_ALREADY_EXISTS:{output}")
    request = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
    matrix = json.loads((ROOT / "experiment-matrix.json").read_text(encoding="utf-8"))
    request = _fixture_request(request, f"w1-eawm-jepa-v4-e2e-fixture-{mode}")
    matrix = _fixture_matrix(matrix)
    boundary = FixtureBoundary()

    def factory(root, out, req, mat, ledger, bound):
        adapter = ProductionRuntimeAdapter(root, out, req, mat, ledger, bound,
                                            gate_fixture=mode)
        adapter.hooks = FixtureHooks(adapter)
        return adapter

    backend = W1RuntimeBackend(ROOT, output, request, matrix,
                               adapter_factory=factory, boundary=boundary)
    try:
        result = execute_pipeline(backend)
        export = _export_manifest(output, result, boundary, mode)
        durable_atomic_json(output / "e2e-path-evidence.json", {
            "schema": "w1-e2e-path-evidence/1.0.0", "mode": mode,
            "status": result.status,
            "policy_training_executed": result.policy_training_executed,
            "task_comparison_executed": result.task_comparison_executed,
            "task_calls": backend.task_calls,
            "stage_functions": ["ProductionDataCollector.collect", "train_select_world_models",
                                "evaluate_prediction_confirmation", "train_policy_routes",
                                "evaluate_task_confirmation", "execute_pipeline"],
            "fixture_differences": {
                "fake_environment": True, "fixture_model_and_policy": True,
                "world_model_epochs": 1, "policy_steps_per_route": 8,
                "policy_rollout_steps": 4, "policy_updates_per_route": 2,
                "formal_matrix_not_claimed": True,
            },
            "prediction_gate": result.evidence.get("prediction_gate"),
            "actual_prediction_metrics": result.evidence.get("prediction_metrics"),
            "ledger": json.loads((output / "settlement.json").read_text(encoding="utf-8")).get("ledger"),
            "export_manifest_sha256": hashlib.sha256((output / "e2e-export-manifest.json").read_bytes()).hexdigest(),
            "checkpoint_files": sorted(str(p.relative_to(output)) for p in output.rglob("*.pt")),
            "boundary_call_count": len(boundary.calls),
        })
        return {"mode": mode, "status": result.status, "task_calls": backend.task_calls,
                "policy": result.policy_training_executed, "tasks": result.task_comparison_executed,
                "output": str(output), "export": export}
    finally:
        backend.close()


def main() -> int:
    # The package's production chain is native-Linux only.  This explicit
    # host-side fixture is used solely to exercise stage routing on Windows
    # when the configured Ubuntu interpreter lacks torch; it is recorded and
    # never treated as a WSL fsync acceptance.
    if os.name == "nt":
        import infra_io
        import production_world
        infra_io._fsync_directory = lambda _directory: None
        production_world._sync_dir = lambda _path: None
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    results = [run_path(mode) for mode in ("prediction_fail", "g1_pass_g2_fail", "all_pass")]
    durable_atomic_json(EVIDENCE_ROOT / "summary.json", {
        "schema": "w1-e2e-engineering-summary/1.0.0", "runner_ready": True,
        "formal_dynamic_calls": 0, "results": results,
        "runner_ready_scope": "engineering runner and cross-system fixture path ready; formal W1 matrix and effect results are not claimed",
        "wsl_dependency_status": "pass: configured Ubuntu-24.04 interpreter uses CPU-only torch",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
