"""Bounded joint training path for groups A-D on the frozen arrival protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .joint_gppo import (
    EVENT_NAMES, JOINT_PROTOCOL, WORLD_FEATURE_DIM, ActionConditionedTemporalWorldModel,
    JointGraphPreferencePolicy, JointTrainConfig, build_observed_event_labels,
    finite_optimizer, finite_parameters, masked_event_bce, masked_normalized_preference,
    preco_logit_direction, preference_similarity, preference_similarity_coefficients,
)
from .m10_environment import M10Config, M10Environment, M10Scenario, scenario_tape
from .m10_training import M10ActorCritic, PPOConfig, _evaluate_sequence, masked_distribution, train_policy, update_policy


@dataclass(frozen=True)
class GroupSpec:
    group: str
    label: str
    runner: str
    history: bool
    preference: bool
    vector_reward: bool
    world_model: bool
    event_auxiliary: bool
    preco: bool
    isolate_world_gradient: bool = False


GROUPS = {
    "A": GroupSpec("A", "GPPO", "legacy_scalar_gppo", True, False, False, False, False, False),
    "B": GroupSpec("B", "GPPO-Preference", "joint", True, True, True, False, False, True),
    "C": GroupSpec("C", "GPPO-Preference-World", "joint", True, True, True, True, False, True),
    "D": GroupSpec("D", "GPPO-Preference-World-Event", "joint", True, True, True, True, True, True),
    # Gradient-isolation comparison groups.  The historical A-D entries above
    # retain their original shared-encoder world-update semantics.
    "B0": GroupSpec("B0", "GPPO-Preference-No-World", "joint", True, True, True, False, False, True),
    "CS": GroupSpec("CS", "GPPO-Preference-World-Shared", "joint", True, True, True, True, False, True, False),
    "CI": GroupSpec("CI", "GPPO-Preference-World-Isolated", "joint", True, True, True, True, False, True, True),
    "DI": GroupSpec("DI", "GPPO-Preference-World-Event-Isolated", "joint", True, True, True, True, True, True, True),
    "DS": GroupSpec("DS", "GPPO-Preference-World-Event-Shared", "joint", True, True, True, True, True, True, False),
    # Preference-update localization pair.  P is the existing PreCo-adapted
    # preference update; W uses the same vector critic/returns and data path
    # but the standard PPO actor surrogate with per-sample scalarized GAE.
    "P": GroupSpec("P", "GPPO-Preference-PreCo", "joint", True, True, True, False, False, True),
    "W": GroupSpec("W", "GPPO-Preference-Weighted-PPO", "joint", True, True, True, False, False, False),
    # Re-integration groups for the preference-weighted PPO route.  These are
    # intentionally separate names: the historical C/D entries retain the
    # PreCo flag and must not be silently relabelled as weighted PPO.
    "WB": GroupSpec("WB", "GPPO-Preference-Weighted-PPO", "joint", True, True, True, False, False, False),
    "WC": GroupSpec("WC", "GPPO-Preference-Weighted-PPO-World", "joint", True, True, True, True, False, False),
    "WD": GroupSpec("WD", "GPPO-Preference-Weighted-PPO-World-Event", "joint", True, True, True, True, True, False),
}


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot encode {type(value).__name__} in training ledger")


def _atomic_bytes(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Directory fsync is not available on every supported filesystem.
            pass
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=_json_default).encode("utf-8")
    _atomic_bytes(path, data + b"\n")


def _atomic_torch_save(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


class _TrainingLedger:
    """Append-only rows; commit pointer is published only after checkpoint verification."""

    def __init__(self, path: Path, *, run_id: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.run_id = run_id
        self.last_committed_step = -1
        self.last_committed_update = 0
        self.committed_bytes = 0
        self.first_uncommitted_step: int | None = None
        self.last_appended_step = -1
        if path.exists():
            raise FileExistsError(f"refusing to append to an existing run ledger: {path}")
        self.stream = path.open("xb")

    def append(self, row: Mapping[str, Any]) -> None:
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default).encode("utf-8") + b"\n"
        if self.first_uncommitted_step is None:
            self.first_uncommitted_step = int(row["step"])
        self.stream.write(encoded)
        self.last_appended_step = int(row["step"])

    def flush_rollout(self) -> dict[str, Any]:
        self.stream.flush()
        os.fsync(self.stream.fileno())
        ledger_bytes = self.path.stat().st_size
        return {
            "first_step": self.first_uncommitted_step,
            "last_step": self.last_appended_step,
            "ledger_bytes": ledger_bytes,
            "ledger_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
        }

    def publish_transaction(self, *, checkpoint_path: Path, counters: Mapping[str, Any],
                            transaction_id: str, ledger_range: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"transaction checkpoint missing: {checkpoint_path}")
        staged = dict(ledger_range or self.flush_rollout())
        if staged["ledger_bytes"] != self.path.stat().st_size:
            raise RuntimeError("ledger changed between checkpoint metadata and commit publication")
        with self.path.open("rb") as stream:
            digest = hashlib.sha256(stream.read(int(staged["ledger_bytes"]))).hexdigest()
        if digest != staged["ledger_sha256"]:
            raise RuntimeError("ledger prefix hash changed before transaction commit")
        checkpoint_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        self.last_committed_step = int(staged["last_step"])
        self.last_committed_update = int(counters.get("policy_optimizer_steps", 0))
        self.committed_bytes = int(staged["ledger_bytes"])
        pointer = {
            "schema": "joint-training-transaction/1.0.0", "run_id": self.run_id,
            "transaction_id": transaction_id,
            "last_committed_step": self.last_committed_step,
            "last_committed_policy_update": self.last_committed_update,
            "last_committed_world_update": int(counters.get("world_optimizer_steps", 0)),
            "ledger": {**staged, "path": self.path.name},
            "checkpoint": {
                "path": checkpoint_path.relative_to(self.path.parent).as_posix(),
                "bytes": checkpoint_path.stat().st_size, "sha256": checkpoint_digest,
            },
            "counters": dict(counters),
        }
        # Atomic pointer replacement is the sole transaction commit point.
        _atomic_json(self.path.with_name("ledger-commit.json"), pointer)
        _atomic_json(self.path.with_name("run-status.json"), {
            "status": "running", "run_id": self.run_id,
            "last_committed_step": self.last_committed_step,
            "last_committed_policy_update": self.last_committed_update,
            "last_committed_world_update": pointer["last_committed_world_update"],
            "ledger_bytes": self.committed_bytes, "transaction_id": transaction_id,
            "checkpoint_sha256": checkpoint_digest, "counters": dict(counters),
        })
        self.first_uncommitted_step = None
        return pointer

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()


def verify_commit_pointer(run_dir: str | Path) -> dict[str, Any]:
    """Verify the committed prefix and checkpoint without deleting an uncommitted tail."""
    root = Path(run_dir)
    pointer = json.loads((root / "ledger-commit.json").read_text(encoding="utf-8"))
    if pointer.get("schema") != "joint-training-transaction/1.0.0":
        raise ValueError("unknown ledger transaction pointer schema")
    ledger_path = root / pointer["ledger"]["path"]
    checkpoint_path = root / pointer["checkpoint"]["path"]
    committed_bytes = int(pointer["ledger"]["ledger_bytes"])
    if ledger_path.stat().st_size < committed_bytes:
        raise IOError("ledger is shorter than the committed byte range")
    with ledger_path.open("rb") as stream:
        prefix_hash = hashlib.sha256(stream.read(committed_bytes)).hexdigest()
    if prefix_hash != pointer["ledger"]["ledger_sha256"]:
        raise IOError("committed ledger prefix hash mismatch")
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    if checkpoint_hash != pointer["checkpoint"]["sha256"] or len(checkpoint_bytes) != pointer["checkpoint"]["bytes"]:
        raise IOError("committed checkpoint size/hash mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    persistence = payload.get("persistence", {})
    counters = payload.get("counters", {})
    expected_counters = pointer["counters"]
    if (payload.get("run_id") != pointer["run_id"]
            or persistence.get("transaction_id") != pointer["transaction_id"]
            or int(counters.get("environment_steps", -1)) != int(expected_counters["environment_steps"])
            or int(counters.get("policy_optimizer_steps", -1)) != int(expected_counters["policy_optimizer_steps"])
            or int(counters.get("world_optimizer_steps", -1)) != int(expected_counters["world_optimizer_steps"])):
        raise ValueError("checkpoint identity/counters do not match the committed pointer")
    return {
        "status": "verified", "run_id": pointer["run_id"],
        "transaction_id": pointer["transaction_id"],
        "last_committed_step": pointer["last_committed_step"],
        "last_committed_policy_update": pointer["last_committed_policy_update"],
        "last_committed_world_update": pointer["last_committed_world_update"],
        "ledger_bytes_committed": committed_bytes,
        "ledger_tail_bytes_uncommitted": ledger_path.stat().st_size - committed_bytes,
        "checkpoint_sha256": checkpoint_hash,
    }


def _legacy_ledger_callback(ledger: _TrainingLedger, *, run_id: str, seed: int,
                            output_dir: Path, scenario_ids: Sequence[str], ppo_config: PPOConfig):
    def write_rollout(transitions: Sequence[Any], start_step: int, committed_steps: int,
                      updates: int, policy: M10ActorCritic) -> None:
        for offset, transition in enumerate(transitions):
            info = transition.info
            current = info["ledger_public_observation"]
            following = info["ledger_next_public_observation"]
            env_info = {key: info.get(key) for key in (
                "feedback", "command_submitted", "command_id", "lease_renewal", "lease_renewals",
                "lease_renewal_delivery_results", "active_continuations", "feedback_log",
                "communication_delta", "new_events", "completion_records", "terminated", "truncated",
            )}
            ledger.append({
                "record_type": "decision", "schema": "joint-decision-ledger/1.0.0",
                "run_id": run_id, "group": "A", "seed": seed,
            "episode_id": info["ledger_episode_id"], "episode_index": int(info["ledger_episode_index"]),
                "step": int(start_step + offset), "tape_id": info["ledger_tape_id"],
                "time": current.get("time"), "public_observation": current,
                "public_entity_ids": current.get("entity_ids"), "legal_mask": current["mask"],
                "action": int(transition.action), "action_legal": bool(current["mask"][transition.action]),
                "legal_action_count": int(np.asarray(current["mask"], dtype=np.bool_).sum()),
                "old_log_prob": float(transition.log_prob), "current_value": float(transition.value),
                "next_state_value": float(transition.next_value),
                "raw_environment_reward": float(info["ledger_raw_reward"]),
                "training_reward": {"scalar": float(transition.reward)},
                "preference": {"status": "not_applicable", "value": None},
                "policy_version": int(updates), "world_model_version": None,
                "communication_proxy_messages": len(info.get("communication_delta", [])),
                "host_confirmed_total": sum(
                    item.get("host_confirmation_time") is not None
                    for item in info.get("completion_records", {}).values()
                ),
                "host_on_time_total": sum(
                    item.get("host_confirmation_before_deadline") is True
                    for item in info.get("completion_records", {}).values()
                ),
                "physical_on_time_total": int(info.get("counts", {}).get("completed", 0)),
                "physical_deadline_failure_total": int(info.get("counts", {}).get("expired", 0)),
                "terminated": bool(transition.terminated), "truncated": bool(transition.truncated),
                "rollout_boundary": bool(transition.truncated and not transition.terminated),
                "bootstrap": {"valid": not transition.terminated, "value": float(transition.next_value),
                              "source": "legacy_critic_next_state" if not transition.terminated else "terminal_zero",
                              "policy_version": int(updates), "world_model_version": None},
                "candidate_predictions": {"status": "not_applicable", "world_model_version": None, "values": None},
                "event_labels": {"status": "not_applicable", "labels": None, "mask": None, "invalid_reasons": None},
                "next_public_observation": following,
                "execution_feedback": env_info,
                "rollout_progress": {"last_step_in_rollout": int(start_step + len(transitions) - 1),
                                     "environment_steps": int(committed_steps),
                                     "policy_updates": int(updates)},
            })
        ledger_range = ledger.flush_rollout()
        counters = {
            "environment_steps": committed_steps, "policy_optimizer_steps": updates,
            "world_optimizer_steps": 0,
        }
        transaction_id = f"txn-step-{committed_steps:08d}-policy-{updates:06d}-world-000000"
        transaction_path = output_dir / "transactions" / f"{transaction_id}.pt"
        payload = {
            "format": "legacy-arrival-gppo-recovery/1.1.0", "run_id": run_id,
            "group": "A", "protocol": JOINT_PROTOCOL,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": policy.optimizer.state_dict(),  # type: ignore[attr-defined]
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "environment_steps": committed_steps, "training_optimizer_steps": updates,
            "scenario_ids": list(scenario_ids), "ppo_config": asdict(ppo_config),
            "next_rollout_seed": seed + committed_steps,
            "safe_restart_boundary": "rollout boundary; next rollout reconstructs environment from frozen tape and next_rollout_seed",
            "last_committed_step": ledger_range["last_step"],
            "last_committed_update": updates,
            "ledger_transaction": ledger_range,
            "persistence": {"transaction_id": transaction_id, "ledger": ledger_range,
                            "last_committed_step": ledger_range["last_step"],
                            "policy_version": updates, "world_version": 0},
            "counters": counters,
            "model_version": updates,
        }
        _atomic_torch_save(transaction_path, payload)
        verified = torch.load(transaction_path, map_location="cpu", weights_only=False)
        if (verified["run_id"] != run_id or verified["counters"] != counters
                or not finite_optimizer(policy.optimizer)
                or not finite_parameters(policy)):
            raise RuntimeError("legacy transaction checkpoint verification failed")
        _atomic_bytes(output_dir / "legacy-last-recovery.pt", transaction_path.read_bytes())
        ledger.publish_transaction(
            checkpoint_path=transaction_path, counters=counters,
            transaction_id=transaction_id, ledger_range=ledger_range,
        )
    return write_rollout


def _obs_tensor(obs: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(obs["flat"], dtype=torch.float32, device=device).reshape(1, -1)


def _relation_for_action(obs: Mapping[str, Any], action: int, device: torch.device) -> torch.Tensor:
    if action == 24:
        return torch.zeros((1, 4), dtype=torch.float32, device=device)
    relations = torch.as_tensor(obs["graph"]["relations"], dtype=torch.float32, device=device).reshape(24, 4)
    return relations[action:action + 1]


def _vector_reward(info: Mapping[str, Any], previous_counts: Mapping[str, int],
                   previous_energy: float, env_config: M10Config) -> tuple[np.ndarray, np.ndarray, dict[str, int], float]:
    counts = info["counts"]
    new_success = max(0, int(counts["completed"]) - int(previous_counts.get("completed", 0)))
    new_deadline_failure = max(0, int(counts["expired"]) - int(previous_counts.get("expired", 0)))
    energy_now = float(sum(float(value) for value in info["energy"].values()))
    used = max(0.0, previous_energy - energy_now)
    # Objective A: normalized net task outcome under physical-arrival deadlines.
    # Objective B: negative fraction of configured fleet energy consumed.
    reward = np.asarray((
        (new_success - new_deadline_failure) / float(env_config.task_capacity),
        -used / max(1e-9, env_config.uav_count * env_config.initial_energy),
    ), dtype=np.float32)
    consequences = np.asarray((
        new_success / float(env_config.task_capacity),
        new_deadline_failure / float(env_config.task_capacity),
    ), dtype=np.float32)
    current_counts = {"completed": int(counts["completed"]), "expired": int(counts["expired"])}
    return reward, consequences, current_counts, energy_now


def _zero_hidden(device: torch.device) -> torch.Tensor:
    return torch.zeros((1, 1, 128), dtype=torch.float32, device=device)


def _zero_wm_hidden(device: torch.device) -> torch.Tensor:
    return torch.zeros((1, 128), dtype=torch.float32, device=device)


def _mask_safe(mask: np.ndarray) -> np.ndarray:
    result = np.asarray(mask, dtype=np.bool_).copy()
    if not result.any():
        result[-1] = True
    return result


def _choose_preference(generator: np.random.Generator, device: torch.device, enabled: bool) -> torch.Tensor:
    value = generator.dirichlet(np.ones(2)) if enabled else np.asarray((0.5, 0.5), dtype=np.float32)
    return masked_normalized_preference(value, device=device)


def _preference_at_rollout_boundary(current: torch.Tensor) -> torch.Tensor:
    """A rollout cut is not an episode boundary: carry the exact preference."""
    return current


def _preference_after_episode(current: torch.Tensor, generator: np.random.Generator,
                              device: torch.device, enabled: bool) -> torch.Tensor:
    """Sample the next preference only after the previous episode terminates."""
    return _choose_preference(generator, device, enabled)


def _make_optimizers(policy: JointGraphPreferencePolicy, world: ActionConditionedTemporalWorldModel | None,
                     spec: GroupSpec, config: JointTrainConfig):
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy_lr)
    world_optimizer = None
    if spec.world_model and world is not None:
        shared = list(policy.base.token_encoder.parameters()) + list(policy.base.type_embedding.parameters())
        shared += list(policy.base.graph_projection.parameters()) + list(policy.base.gru.parameters())
        world_parameters = [] if spec.isolate_world_gradient else shared
        world_optimizer = torch.optim.Adam([*world_parameters, *world.parameters()], lr=config.world_lr, weight_decay=0.0)
    return policy_optimizer, world_optimizer


def _gae_vector(transitions: Sequence[dict[str, Any]], config: JointTrainConfig, device: torch.device):
    rewards = np.stack([item["vector_reward"] for item in transitions]).astype(np.float32)
    values = np.stack([item["old_values"] for item in transitions]).astype(np.float32)
    next_values = np.stack([item["next_values"] for item in transitions]).astype(np.float32)
    terminated = np.asarray([item["terminated"] for item in transitions], dtype=np.float32)
    boundaries = np.asarray([item["terminated"] or item["truncated"] for item in transitions], dtype=np.float32)
    advantages = np.zeros_like(rewards)
    running = np.zeros((2,), dtype=np.float32)
    for index in range(len(transitions) - 1, -1, -1):
        delta = rewards[index] + config.gamma * next_values[index] * (1.0 - terminated[index]) - values[index]
        running = delta + config.gamma * config.gae_lambda * (1.0 - boundaries[index]) * running
        advantages[index] = running
    returns = advantages + values
    return torch.as_tensor(advantages, device=device), torch.as_tensor(returns, device=device)


def scalarized_preference_advantage(preference: torch.Tensor, advantages: torch.Tensor,
                                    config: JointTrainConfig) -> torch.Tensor:
    """Return per-transition A_scalar = sum_i p_i * scaled A_i."""
    if preference.ndim != 2 or preference.shape[-1] != 2:
        raise ValueError("preference must be [batch, 2]")
    if advantages.shape != preference.shape:
        raise ValueError("advantages and preference must have identical shape")
    scales = torch.as_tensor(
        (config.task_reward_scale, config.energy_reward_scale),
        dtype=advantages.dtype, device=advantages.device,
    )
    return (preference.detach() * advantages * scales).sum(dim=-1)


def _masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor,
                      sample_mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    valid = sample_mask.bool().reshape(-1)
    count = int(valid.sum().item())
    if count == 0:
        return prediction.sum() * 0.0, 0
    return F.smooth_l1_loss(prediction[valid], target[valid]), count


def _batch_encode(policy: JointGraphPreferencePolicy, transitions: Sequence[dict[str, Any]], device: torch.device):
    obs = torch.as_tensor(np.stack([item["obs"] for item in transitions]), dtype=torch.float32, device=device)
    hidden = torch.as_tensor(np.stack([item["policy_hidden_before"] for item in transitions]), dtype=torch.float32, device=device)
    mask = torch.as_tensor(np.stack([item["mask"] for item in transitions]), dtype=torch.bool, device=device)
    pref = torch.as_tensor(np.stack([item["preference"] for item in transitions]), dtype=torch.float32, device=device)
    candidate_features = torch.as_tensor(np.stack([item["candidate_features"] for item in transitions]), dtype=torch.float32, device=device)
    # Match the exact scalar/batch-1 arithmetic path used when each action's
    # behavior log-probability was sampled. On CUDA, a single batched GEMM can
    # differ by several e-5 from the original per-decision forward; replaying
    # rows individually prevents a numerical PPO ratio offset at update time.
    feature_rows, pair_rows, next_hidden_rows = [], [], []
    for index in range(len(transitions)):
        row_features, row_pair, row_next_hidden = policy.encode(
            obs[index:index + 1], hidden[index:index + 1].reshape(1, 1, -1),
        )
        feature_rows.append(row_features)
        pair_rows.append(row_pair)
        next_hidden_rows.append(row_next_hidden)
    features = torch.cat(feature_rows, dim=0)
    pair_messages = torch.cat(pair_rows, dim=0)
    next_hidden = torch.cat(next_hidden_rows, dim=1)
    return obs, hidden, mask, pref, candidate_features, features, pair_messages, next_hidden


def _joint_logprob_rows(policy: JointGraphPreferencePolicy, transitions: Sequence[dict[str, Any]],
                        device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        _, _, mask, preference, candidates, features, pair, _ = _batch_encode(policy, transitions, device)
        result = policy.evaluate_encoded(features, pair, preference, candidates, mask)
        actions = torch.as_tensor([item["action"] for item in transitions], dtype=torch.long, device=device)
        return result["distribution"].log_prob(actions)


def _decision_probe(policy: JointGraphPreferencePolicy,
                    world: ActionConditionedTemporalWorldModel | None,
                    observation: Mapping[str, Any], policy_hidden: torch.Tensor | None,
                    world_hidden: torch.Tensor | None, preference: torch.Tensor | None,
                    *, use_events: bool, device: torch.device) -> dict[str, Any]:
    """Deterministic distribution/value witness for exact recovery checks."""
    obs_tensor = _obs_tensor(observation, device)
    mask = torch.as_tensor(_mask_safe(observation["mask"]), dtype=torch.bool, device=device)[None, :]
    hidden = _zero_hidden(device) if policy_hidden is None else policy_hidden
    pref = (torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
            if preference is None else preference.reshape(1, 2))
    with torch.no_grad():
        features, pair_messages, _ = policy.encode(obs_tensor, hidden)
        if world is None:
            candidate = torch.zeros((1, 25, WORLD_FEATURE_DIM), dtype=torch.float32, device=device)
        else:
            candidate, _ = world.predict_all_candidates(
                features, _zero_wm_hidden(device) if world_hidden is None else world_hidden,
                observation, use_events=use_events,
            )
        evaluated = policy.evaluate_encoded(features, pair_messages, pref, candidate, mask)
        probabilities = evaluated["distribution"].probs
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        action = evaluated["distribution"].sample()
        log_probability = evaluated["distribution"].log_prob(action)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        return {
            "probabilities": probabilities[0].detach().cpu().tolist(),
            "values": evaluated["critic_values"][0].detach().cpu().tolist(),
            "action": int(action.item()), "log_probability": float(log_probability.item()),
            "legal_mask": mask[0].detach().cpu().to(torch.int8).tolist(),
        }


def _legacy_decision_probe(policy: M10ActorCritic, observation: Mapping[str, Any],
                           device: torch.device) -> dict[str, Any]:
    obs = torch.as_tensor(observation["flat"], dtype=torch.float32, device=device)[None, :]
    mask = torch.as_tensor(_mask_safe(observation["mask"]), dtype=torch.bool, device=device)[None, :]
    with torch.no_grad():
        logits, value, _ = policy(obs, None)
        distribution = masked_distribution(logits, mask)
        probabilities = distribution.probs
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        action = distribution.sample()
        log_probability = distribution.log_prob(action)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        return {
            "probabilities": probabilities[0].detach().cpu().tolist(),
            "value": float(value[0].detach().cpu()), "action": int(action.item()),
            "log_probability": float(log_probability.item()),
            "legal_mask": mask[0].detach().cpu().to(torch.int8).tolist(),
        }


def _distribution_shift(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float]:
    p = torch.as_tensor(before["probabilities"], dtype=torch.float64)
    q = torch.as_tensor(after["probabilities"], dtype=torch.float64)
    if before["legal_mask"] != after["legal_mask"]:
        raise RuntimeError("legal action mask changed during model update probe")
    eps = torch.finfo(torch.float64).tiny
    kl = torch.sum(torch.where(p > 0, p * (p.clamp_min(eps).log() - q.clamp_min(eps).log()), 0.0))
    tv = 0.5 * (p - q).abs().sum()
    return {"forward_kl": float(kl), "total_variation": float(tv)}


def _nested_max_abs_delta(left: Any, right: Any) -> float:
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape:
            return float("inf")
        if left.numel() == 0:
            return 0.0
        return float((left.detach().to(torch.float64).cpu() - right.detach().to(torch.float64).cpu()).abs().max())
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return float("inf")
        return max((_nested_max_abs_delta(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return float("inf")
        return max((_nested_max_abs_delta(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def ppo_preference_update(policy: JointGraphPreferencePolicy, transitions: Sequence[dict[str, Any]],
                          optimizer: torch.optim.Optimizer, config: JointTrainConfig,
                          device: torch.device, *, event_group: str) -> dict[str, float]:
    advantages, returns = _gae_vector(transitions, config, device)
    obs, hidden, mask, preference, candidates, features, pair, _ = _batch_encode(policy, transitions, device)
    evaluation = policy.evaluate_encoded(features, pair, preference, candidates, mask)
    logits = evaluation["logits"]
    distribution = evaluation["distribution"]
    actions = torch.as_tensor([item["action"] for item in transitions], dtype=torch.long, device=device)
    old_log_prob = torch.as_tensor([item["old_log_prob"] for item in transitions], dtype=torch.float32, device=device)
    log_prob = distribution.log_prob(actions)
    ratio = torch.exp(log_prob - old_log_prob)
    # Convert raw-return advantages to the fixed positive utility scales used
    # by the preference protocol. Additive reference shifts are applied only
    # to detached similarity targets, never to per-step rewards.
    utility_advantages = advantages * torch.as_tensor(
        (config.task_reward_scale, config.energy_reward_scale), dtype=advantages.dtype, device=device,
    )
    objective_losses = []
    objective_surrogates = []
    for objective in range(2):
        adv = utility_advantages[:, objective]
        clipped = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
        surrogate_rows = torch.minimum(ratio * adv, clipped * adv)
        objective_surrogates.append(surrogate_rows)
        objective_losses.append(-surrogate_rows.mean())
    similarity_coefficients, utility_return_targets, clipped_utility_components = preference_similarity_coefficients(
        preference, returns.detach(),
    )
    # PPO adaptation of PCRL's chain rule:
    #   dPsi/dtheta ~= sum_i stopgrad(dPsi/du_i) * grad_theta J_i^clip.
    # Return-derived coefficients are stop-gradient; the clipped PPO
    # surrogates carry actual rollout advantages. Critic or WM prediction-
    # error gradients are not substituted for objective-return gradients.
    local_similarity_surrogate = (
        similarity_coefficients * torch.stack(objective_surrogates, dim=-1)
    ).sum(dim=-1).mean()
    similarity = preference_similarity(preference, utility_return_targets)
    similarity_gradient = torch.autograd.grad(
        local_similarity_surrogate, logits, retain_graph=True,
    )[0]
    similarity_gradient_norm = float(torch.linalg.vector_norm(similarity_gradient).detach())
    direction, objective0_weight = preco_logit_direction(
        objective_losses, local_similarity_surrogate, logits, strength=config.preco_lambda,
    )
    # The detached logit-space vector is a first-order surrogate whose policy
    # parameter gradient equals the registered PreCo-combined direction.
    actor_loss = -(logits * direction.detach()).sum()
    value_loss = F.mse_loss(evaluation["critic_values"], returns)
    entropy = distribution.entropy().mean()
    total_loss = actor_loss + config.value_weight * value_loss - config.entropy_weight * entropy
    if not all(torch.isfinite(value).all().item() for value in (total_loss, actor_loss, value_loss, entropy, direction)):
        raise FloatingPointError("non-finite PPO/PreCo loss or direction")
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
    if not torch.isfinite(torch.as_tensor(grad_norm)):
        raise FloatingPointError("non-finite policy gradient norm")
    optimizer.step()
    if not finite_parameters(policy) or not finite_optimizer(optimizer):
        raise FloatingPointError("non-finite policy/optimizer state after PPO update")
    return {
        "policy_loss": float(actor_loss.detach()), "objective_loss_task": float(objective_losses[0].detach()),
        "objective_loss_energy": float(objective_losses[1].detach()), "preference_similarity": float(similarity.detach()),
        "similarity_gradient_norm": similarity_gradient_norm,
        "preference_target_utility_clipped_components": float(clipped_utility_components),
        "preco_task_weight": objective0_weight, "value_loss_vector": float(value_loss.detach()),
        "entropy": float(entropy.detach()), "grad_norm": float(torch.as_tensor(grad_norm).detach()),
        "approx_kl": float((old_log_prob - log_prob).mean().detach()),
        "clip_fraction": float(((ratio - 1.0).abs() > config.clip_epsilon).float().mean().detach()),
        "event_objective_enabled": float(event_group == "D"),
    }


def ppo_weighted_preference_update(policy: JointGraphPreferencePolicy,
                                   transitions: Sequence[dict[str, Any]],
                                   optimizer: torch.optim.Optimizer,
                                   config: JointTrainConfig,
                                   device: torch.device, *, event_group: str) -> dict[str, float]:
    """Standard PPO actor update with per-sample preference scalarization.

    This is the W arm of the preference-update localization experiment.  It
    shares the exact vector GAE, return targets, value loss, entropy term,
    clipping, optimizer and rollout contract with the PreCo arm.  Only the
    actor direction differs: A_scalar = sum_i p_i A_i for each transition.
    """
    advantages, returns = _gae_vector(transitions, config, device)
    _, _, _, preference, _, features, pair, _ = _batch_encode(policy, transitions, device)
    evaluation = policy.evaluate_encoded(
        features, pair, preference,
        torch.as_tensor(np.stack([item["candidate_features"] for item in transitions]),
                        dtype=torch.float32, device=device),
        torch.as_tensor(np.stack([item["mask"] for item in transitions]),
                        dtype=torch.bool, device=device),
    )
    distribution = evaluation["distribution"]
    actions = torch.as_tensor([item["action"] for item in transitions], dtype=torch.long, device=device)
    old_log_prob = torch.as_tensor([item["old_log_prob"] for item in transitions], dtype=torch.float32, device=device)
    log_prob = distribution.log_prob(actions)
    ratio = torch.exp(log_prob - old_log_prob)
    utility_advantages = advantages * torch.as_tensor(
        (config.task_reward_scale, config.energy_reward_scale),
        dtype=advantages.dtype, device=device,
    )
    scalar_advantage = scalarized_preference_advantage(preference, advantages, config)
    clipped_ratio = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
    surrogate = torch.minimum(ratio * scalar_advantage, clipped_ratio * scalar_advantage)
    actor_loss = -surrogate.mean()
    value_loss = F.mse_loss(evaluation["critic_values"], returns)
    entropy = distribution.entropy().mean()
    total_loss = actor_loss + config.value_weight * value_loss - config.entropy_weight * entropy
    if not all(torch.isfinite(value).all().item() for value in (total_loss, actor_loss, value_loss, entropy, scalar_advantage)):
        raise FloatingPointError("non-finite weighted-PPO loss or advantage")
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
    if not torch.isfinite(torch.as_tensor(grad_norm)):
        raise FloatingPointError("non-finite weighted-PPO gradient norm")
    optimizer.step()
    if not finite_parameters(policy) or not finite_optimizer(optimizer):
        raise FloatingPointError("non-finite weighted-PPO policy/optimizer state after update")
    return {
        "policy_loss": float(actor_loss.detach()),
        "weighted_advantage_mean": float(scalar_advantage.detach().mean()),
        "weighted_advantage_task_component": float((preference[:, 0] * utility_advantages[:, 0]).detach().mean()),
        "weighted_advantage_energy_component": float((preference[:, 1] * utility_advantages[:, 1]).detach().mean()),
        "preference_similarity": 0.0,
        "similarity_gradient_norm": 0.0,
        "preference_target_utility_clipped_components": 0.0,
        "preco_task_weight": 0.0,
        "objective_loss_task": float((-torch.minimum(ratio * utility_advantages[:, 0], clipped_ratio * utility_advantages[:, 0])).mean().detach()),
        "objective_loss_energy": float((-torch.minimum(ratio * utility_advantages[:, 1], clipped_ratio * utility_advantages[:, 1])).mean().detach()),
        "value_loss_vector": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
        "grad_norm": float(torch.as_tensor(grad_norm).detach()),
        "approx_kl": float((old_log_prob - log_prob).mean().detach()),
        "clip_fraction": float(((ratio - 1.0).abs() > config.clip_epsilon).float().mean().detach()),
        "event_objective_enabled": float(event_group == "D"),
    }


def world_model_updates(policy: JointGraphPreferencePolicy, world: ActionConditionedTemporalWorldModel,
                        transitions: Sequence[dict[str, Any]], optimizer: torch.optim.Optimizer,
                        config: JointTrainConfig, device: torch.device, *, updates: int,
                        event_enabled: bool, batch_rng: np.random.Generator | None = None,
                        isolate_policy_gradient: bool = False) -> list[dict[str, float]]:
    if updates <= 0:
        return []
    if not transitions:
        raise ValueError("world-model update requested without transitions")
    result = []
    order = np.arange(len(transitions), dtype=np.int64)
    rng = batch_rng or np.random.default_rng(config.seed + 0x51A7 + len(transitions) + updates)
    for update_index in range(updates):
        if update_index % math.ceil(len(order) / config.wm_minibatch_size) == 0:
            rng.shuffle(order)
        start = (update_index * config.wm_minibatch_size) % len(order)
        idx = order[start:start + config.wm_minibatch_size]
        if len(idx) < config.wm_minibatch_size:
            idx = np.concatenate((idx, order[:config.wm_minibatch_size - len(idx)]))
        batch = [transitions[int(i)] for i in idx]
        obs, hidden_before, _, _, _, features, _, _ = _batch_encode(policy, batch, device)
        if isolate_policy_gradient:
            # The world model still receives the exact policy representation,
            # but its losses cannot write into the GPPO encoder/history path.
            features = features.detach()
        actions = torch.as_tensor([item["action"] for item in batch], dtype=torch.long, device=device)
        relations = torch.cat([_relation_for_action(item["observation_dict"], int(item["action"]), device) for item in batch], dim=0)
        wm_hidden = torch.as_tensor(np.stack([item["world_hidden_before"] for item in batch]), dtype=torch.float32, device=device)
        if isolate_policy_gradient:
            wm_hidden = wm_hidden.detach()
        output = world(features, actions, relations, wm_hidden)
        next_obs = torch.as_tensor(np.stack([item["next_obs"] for item in batch]), dtype=torch.float32, device=device)
        next_hidden_before = torch.as_tensor(np.stack([item["policy_hidden_after"] for item in batch]), dtype=torch.float32, device=device)
        with torch.no_grad():
            next_state, _, _ = policy.encode(next_obs, next_hidden_before[None, :, :])
            next_state = next_state.detach()
        target_reward = torch.as_tensor(np.stack([item["vector_reward"] for item in batch]), dtype=torch.float32, device=device)
        target_consequence = torch.as_tensor(np.stack([item["task_consequence"] for item in batch]), dtype=torch.float32, device=device)
        event_labels = torch.as_tensor(np.stack([item["event_label"] for item in batch]), dtype=torch.float32, device=device)
        event_mask = torch.as_tensor(np.stack([item["event_mask"] for item in batch]), dtype=torch.bool, device=device)
        state_mask = torch.as_tensor([item["state_target_valid"] for item in batch], dtype=torch.bool, device=device)
        reward_mask = torch.as_tensor([item["vector_reward_valid"] for item in batch], dtype=torch.bool, device=device)
        consequence_mask = torch.as_tensor([item["task_consequence_valid"] for item in batch], dtype=torch.bool, device=device)
        state_loss, state_valid_count = _masked_smooth_l1(output["next_state"], next_state, state_mask)
        reward_loss, reward_valid_count = _masked_smooth_l1(output["vector_reward"], target_reward, reward_mask)
        consequence_loss, consequence_valid_count = _masked_smooth_l1(output["task_consequence"], target_consequence, consequence_mask)
        event_loss, event_valid_count = masked_event_bce(output["event_logits"], event_labels, event_mask)
        if not event_enabled:
            event_loss = output["event_logits"].sum() * 0.0
            event_encoder_grad_norm = 0.0
        elif event_valid_count:
            event_gradients = torch.autograd.grad(
                event_loss, tuple(policy.base.token_encoder.parameters()),
                retain_graph=True, allow_unused=True,
            )
            event_encoder_grad_norm = math.sqrt(sum(
                float(gradient.detach().pow(2).sum()) for gradient in event_gradients if gradient is not None
            ))
        else:
            event_encoder_grad_norm = 0.0
        total = state_loss + reward_loss + consequence_loss + event_loss
        if not torch.isfinite(total) or not all(torch.isfinite(item).all().item() for item in (state_loss, reward_loss, consequence_loss, event_loss)):
            raise FloatingPointError("non-finite world-model loss")
        shared_parameters = [
            *policy.base.token_encoder.parameters(), *policy.base.type_embedding.parameters(),
            *policy.base.graph_projection.parameters(), *policy.base.gru.parameters(),
        ]
        shared_before = [parameter.detach().clone() for parameter in shared_parameters]
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        shared_grad_norm = math.sqrt(sum(
            float(parameter.grad.detach().pow(2).sum())
            for parameter in shared_parameters if parameter.grad is not None
        ))
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError("non-finite world-model gradient norm")
        optimizer.step()
        shared_parameter_delta = max((
            float((after.detach() - before).abs().max())
            for before, after in zip(shared_before, shared_parameters)
        ), default=0.0)
        if not finite_parameters(policy) or not finite_parameters(world) or not finite_optimizer(optimizer):
            raise FloatingPointError("non-finite world-model/shared-encoder state after update")
        result.append({
            "total": float(total.detach()), "next_state": float(state_loss.detach()),
            "vector_reward": float(reward_loss.detach()), "task_consequence": float(consequence_loss.detach()),
            "event_bce": float(event_loss.detach()), "event_valid_elements": event_valid_count,
            "event_shared_encoder_grad_norm": event_encoder_grad_norm,
            "shared_encoder_grad_norm": shared_grad_norm,
            "shared_encoder_parameter_delta": shared_parameter_delta,
            "world_gradient_isolated": bool(isolate_policy_gradient),
            "state_valid_samples": state_valid_count, "reward_valid_samples": reward_valid_count,
            "task_consequence_valid_samples": consequence_valid_count,
            "grad_norm": float(torch.as_tensor(grad_norm).detach()),
        })
    return result


def _replay_hidden(policy: JointGraphPreferencePolicy, world: ActionConditionedTemporalWorldModel | None,
                   episode_history: Sequence[tuple[Mapping[str, Any], int]], device: torch.device):
    policy_hidden = None
    world_hidden = None
    with torch.no_grad():
        for observation, action in episode_history:
            tensor = _obs_tensor(observation, device)
            features, _, policy_hidden = policy.encode(tensor, policy_hidden)
            if world is not None:
                output = world(features, torch.as_tensor([action], dtype=torch.long, device=device),
                               _relation_for_action(observation, action, device), world_hidden)
                world_hidden = output["hidden"]
    return policy_hidden, world_hidden


def _rng_snapshot(preference_rng: np.random.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy_global": np.random.get_state(),
        "preference_generator": copy.deepcopy(preference_rng.bit_generator.state),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_snapshot(snapshot: Mapping[str, Any], preference_rng: np.random.Generator | None = None) -> None:
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy_global"])
    torch.set_rng_state(snapshot["torch_cpu"].cpu())
    if torch.cuda.is_available() and snapshot.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in snapshot["torch_cuda"]])
    if preference_rng is not None:
        preference_rng.bit_generator.state = copy.deepcopy(snapshot["preference_generator"])


def save_joint_checkpoint(path: str | Path, *, run_id: str, group: str,
                          policy: JointGraphPreferencePolicy, world: ActionConditionedTemporalWorldModel | None,
                          policy_optimizer: torch.optim.Optimizer,
                          world_optimizer: torch.optim.Optimizer | None,
                          environment: M10Environment, observation: Mapping[str, Any],
                          policy_hidden: torch.Tensor | None, world_hidden: torch.Tensor | None,
                          episode_history: Sequence[tuple[Mapping[str, Any], int]],
                           preference_rng: np.random.Generator, episode_preference: torch.Tensor | None,
                           world_batch_rng: np.random.Generator,
                          recovery_probe: Mapping[str, Any],
                          counters: Mapping[str, int],
                          identity: Mapping[str, Any],
                          persistence: Mapping[str, Any] | None = None) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "joint-arrival-training-checkpoint/1.1.0", "run_id": run_id, "group": group,
        "protocol": JOINT_PROTOCOL, "identity": dict(identity),
        "policy_state_dict": policy.state_dict(),
        "world_state_dict": None if world is None else world.state_dict(),
        "policy_optimizer_state_dict": policy_optimizer.state_dict(),
        "world_optimizer_state_dict": None if world_optimizer is None else world_optimizer.state_dict(),
        "counters": dict(counters), "persistence": dict(persistence or {}),
        "rng_state": _rng_snapshot(preference_rng),
        "runtime_state": {
            "environment": environment, "observation": dict(observation),
            "policy_hidden": None if policy_hidden is None else policy_hidden.detach().cpu(),
            "world_hidden": None if world_hidden is None else world_hidden.detach().cpu(),
            "episode_history": list(episode_history),
            "episode_preference": None if episode_preference is None else episode_preference.detach().cpu().clone(),
            "episode_index": int(counters.get("episode_index", 0)),
            "world_batch_rng_state": copy.deepcopy(world_batch_rng.bit_generator.state),
            "recovery_probe": dict(recovery_probe),
            "public_digest": environment.public_snapshot_digest(),
        },
    }
    _atomic_torch_save(target, payload)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return digest


def load_joint_checkpoint(path: str | Path, *, expected_run_id: str, expected_group: str,
                          policy: JointGraphPreferencePolicy, world: ActionConditionedTemporalWorldModel | None,
                           policy_optimizer: torch.optim.Optimizer,
                           world_optimizer: torch.optim.Optimizer | None, device: torch.device,
                           preference_rng: np.random.Generator | None = None,
                           world_batch_rng: np.random.Generator | None = None):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "joint-arrival-training-checkpoint/1.1.0" or payload.get("protocol") != JOINT_PROTOCOL:
        raise ValueError("joint checkpoint format/protocol mismatch")
    if payload.get("run_id") != expected_run_id or payload.get("group") != expected_group:
        raise ValueError("joint checkpoint run identity mismatch")
    policy.load_state_dict(payload["policy_state_dict"], strict=True)
    if world is None:
        if payload["world_state_dict"] is not None:
            raise ValueError("checkpoint unexpectedly contains a world model")
    else:
        if payload["world_state_dict"] is None:
            raise ValueError("checkpoint missing the registered world model")
        world.load_state_dict(payload["world_state_dict"], strict=True)
    policy_optimizer.load_state_dict(payload["policy_optimizer_state_dict"])
    if (world_optimizer is None) != (payload["world_optimizer_state_dict"] is None):
        raise ValueError("checkpoint world optimizer compatibility mismatch")
    if world_optimizer is not None:
        world_optimizer.load_state_dict(payload["world_optimizer_state_dict"])
    runtime = payload["runtime_state"]
    environment = runtime["environment"]
    if environment.public_snapshot_digest() != runtime["public_digest"]:
        raise ValueError("checkpoint environment/public observation digest mismatch")
    rng_state = payload["rng_state"]
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy_global"])
    torch.set_rng_state(rng_state["torch_cpu"].cpu())
    if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in rng_state["torch_cuda"]])
    if preference_rng is not None:
        preference_rng.bit_generator.state = copy.deepcopy(rng_state["preference_generator"])
    if world_batch_rng is not None:
        world_batch_rng.bit_generator.state = copy.deepcopy(runtime["world_batch_rng_state"])
    return payload, runtime


def _run_joint(*, run_id: str, group: str, output_dir: Path, device: torch.device,
               env_config: M10Config, train_config: JointTrainConfig,
               max_steps: int, max_policy_updates: int, max_world_updates: int,
               wall_seconds: float, verify_recovery_update: bool,
               scenarios: Sequence[M10Scenario] | None = None,
               initial_policy_state_dict: Mapping[str, torch.Tensor] | None = None,
               initial_world_state_dict: Mapping[str, torch.Tensor] | None = None,
               initial_action_rng_state: torch.Tensor | None = None) -> dict[str, Any]:
    spec = GROUPS[group]
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = _TrainingLedger(output_dir / "training-ledger.jsonl", run_id=run_id)
    if group == "A":
        legacy_rollout_steps = max(1, math.ceil(max_steps / max(1, max_policy_updates)))
        ppo = PPOConfig(rollout_steps=legacy_rollout_steps, update_epochs=1, learning_rate=train_config.policy_lr,
                        gamma=train_config.gamma, gae_lambda=train_config.gae_lambda,
                        clip_epsilon=train_config.clip_epsilon, entropy_weight=train_config.entropy_weight,
                        value_weight=train_config.value_weight, grad_clip=train_config.grad_clip,
                        minibatch_size=legacy_rollout_steps)
        scenarios = tuple(scenarios or scenario_tape("train", count=8, base_seed=train_config.seed, name="mixed"))
        ledger_callback = _legacy_ledger_callback(
            ledger, run_id=run_id, seed=train_config.seed, output_dir=output_dir,
            scenario_ids=[scenario.tape_id for scenario in scenarios], ppo_config=ppo,
        )
        policy, metadata = train_policy(
            variant="A-GPPO-scalar", encoder="graph", type_count=5, history=True,
            fusion="base", model=None, seed=train_config.seed, steps=max_steps,
            env_config=env_config, ppo_config=ppo, device=str(device), scenarios=scenarios,
            capture_recovery_reference=verify_recovery_update, ledger_callback=ledger_callback,
        )
        ledger.close()
        if not finite_parameters(policy) or not finite_optimizer(policy.optimizer):  # type: ignore[attr-defined]
            raise FloatingPointError("legacy GPPO returned non-finite model or optimizer state")
        recovery_validation = {"status": "not_requested", "optimizer_steps": 0}
        recovery_reference_sha256 = None
        if verify_recovery_update:
            output_dir.mkdir(parents=True, exist_ok=True)
            reference = metadata.pop("_recovery_reference")
            reference_path = output_dir / "legacy-update-recovery-reference.pt"
            _atomic_torch_save(reference_path, reference)
            reference = torch.load(reference_path, map_location=device, weights_only=False)
            transitions = reference["transitions"]
            shadow = M10ActorCritic(
                uav_count=env_config.uav_count, task_capacity=env_config.task_capacity,
                action_count=env_config.action_count, encoder="graph", type_count=5,
                history=True, context_dim=0, region_count=env_config.region_count,
                target_count=env_config.target_count, event_capacity=env_config.event_capacity,
                relation_width=env_config.relation_width,
            ).to(device)
            shadow.optimizer = torch.optim.Adam(shadow.parameters(), lr=ppo.learning_rate)  # type: ignore[attr-defined]
            shadow.load_state_dict(reference["policy_state_dict"], strict=True)
            shadow.optimizer.load_state_dict(reference["optimizer_state_dict"])  # type: ignore[attr-defined]
            replay_log_probs, _, _ = _evaluate_sequence(shadow, transitions, device)
            actor_indices = [i for i, item in enumerate(transitions) if item.actor_decision]
            behavior_error = max(
                (abs(float(replay_log_probs[i]) - float(transitions[i].log_prob)) for i in actor_indices),
                default=0.0,
            )
            replay_config = PPOConfig(**reference["effective_ppo_config"])
            update_policy(shadow, transitions, ppo=replay_config, device=device)
            parameter_delta = _nested_max_abs_delta(policy.state_dict(), shadow.state_dict())
            optimizer_delta = _nested_max_abs_delta(policy.optimizer.state_dict(), shadow.optimizer.state_dict())  # type: ignore[attr-defined]
            if max(behavior_error, parameter_delta, optimizer_delta) > 1e-7:
                raise RuntimeError("legacy GPPO serialized recovery/update replay diverged")
            final_path = output_dir / "legacy-last-recovery.pt"
            next_environment = M10Environment(env_config, scenarios[0])
            next_observation = next_environment.reset()
            next_decision_probe = _legacy_decision_probe(policy, next_observation, device)
            _atomic_torch_save(final_path, {
                "format": "legacy-arrival-gppo-recovery/1.0.0", "run_id": run_id,
                "group": group, "protocol": JOINT_PROTOCOL,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": policy.optimizer.state_dict(),  # type: ignore[attr-defined]
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_cpu_rng": torch.get_rng_state(),
                "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "next_rollout_seed": train_config.seed + max_steps,
                "environment_steps": metadata["environment_steps"],
                "training_optimizer_steps": metadata["optimizer_updates"],
                "safe_restart_boundary": "legacy collect_rollout recreates the simulator from deterministic rollout seed",
                "next_observation": next_observation,
                "next_decision_probe": next_decision_probe,
            })
            restored = M10ActorCritic(
                uav_count=env_config.uav_count, task_capacity=env_config.task_capacity,
                action_count=env_config.action_count, encoder="graph", type_count=5,
                history=True, context_dim=0, region_count=env_config.region_count,
                target_count=env_config.target_count, event_capacity=env_config.event_capacity,
                relation_width=env_config.relation_width,
            ).to(device)
            restored.optimizer = torch.optim.Adam(restored.parameters(), lr=ppo.learning_rate)  # type: ignore[attr-defined]
            loaded = torch.load(final_path, map_location=device, weights_only=False)
            restored.load_state_dict(loaded["model_state_dict"], strict=True)
            restored.optimizer.load_state_dict(loaded["optimizer_state_dict"])  # type: ignore[attr-defined]
            random.setstate(loaded["python_rng"])
            np.random.set_state(loaded["numpy_rng"])
            torch.set_rng_state(loaded["torch_cpu_rng"].cpu())
            if torch.cuda.is_available() and loaded.get("torch_cuda_rng") is not None:
                torch.cuda.set_rng_state_all([item.cpu() for item in loaded["torch_cuda_rng"]])
            restored_probe = _legacy_decision_probe(restored, loaded["next_observation"], device)
            next_decision_delta = _nested_max_abs_delta(restored_probe, loaded["next_decision_probe"])
            final_parameter_delta = _nested_max_abs_delta(policy.state_dict(), restored.state_dict())
            final_optimizer_delta = _nested_max_abs_delta(policy.optimizer.state_dict(), restored.optimizer.state_dict())  # type: ignore[attr-defined]
            if max(final_parameter_delta, final_optimizer_delta, next_decision_delta) != 0.0:
                raise RuntimeError("legacy final recovery checkpoint differs after reload")
            recovery_reference_sha256 = hashlib.sha256(reference_path.read_bytes()).hexdigest()
            recovery_validation = {
                "status": "passed", "next_decision_behavior_logprob_max_abs_delta": behavior_error,
                "post_update_parameter_max_abs_delta": parameter_delta,
                "post_update_optimizer_max_abs_delta": optimizer_delta,
                "final_checkpoint_parameter_max_abs_delta": final_parameter_delta,
                "final_checkpoint_optimizer_max_abs_delta": final_optimizer_delta,
                "next_decision_probe_max_abs_delta": next_decision_delta,
                "training_optimizer_steps": metadata["optimizer_updates"],
                "verification_optimizer_steps": 1,
                "checkpoint": str(final_path),
                "checkpoint_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
            }
        elif not verify_recovery_update:
            recovery_validation = {"status": "not_requested", "verification_optimizer_steps": 0}
        return {
            "status": "completed", "group": "A", "legacy_runner": True,
            "environment_steps": metadata["environment_steps"], "actor_calls": metadata["actor_decisions"],
            "policy_optimizer_steps": metadata["optimizer_updates"],
            "policy_optimizer_verification_steps": recovery_validation["verification_optimizer_steps"],
            "policy_optimizer_calls_total": metadata["optimizer_updates"] + recovery_validation["verification_optimizer_steps"],
            "world_optimizer_steps": 0, "world_candidate_action_predictions": 0,
            "recovery_update_verification": recovery_validation,
            "recovery_reference_sha256": recovery_reference_sha256, "metadata": metadata,
            "next_decision_probe": recovery_validation.get("next_decision_probe_max_abs_delta"),
        }
    if spec.runner != "joint":
        raise ValueError(f"unknown training group {group}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable; no device fallback is allowed")
    policy = JointGraphPreferencePolicy(env_config, history=True).to(device)
    world = ActionConditionedTemporalWorldModel().to(device) if spec.world_model else None
    if initial_policy_state_dict is not None:
        policy.load_state_dict(copy.deepcopy(initial_policy_state_dict), strict=True)
    if world is not None and initial_world_state_dict is not None:
        world.load_state_dict(copy.deepcopy(initial_world_state_dict), strict=True)
    if initial_action_rng_state is not None:
        # Extra world-module construction must not perturb the common policy
        # sampling stream used by B0/CS/CI/DI initialization comparisons.
        torch.set_rng_state(initial_action_rng_state.detach().cpu().clone())
    policy_optimizer, world_optimizer = _make_optimizers(policy, world, spec, train_config)
    if max_world_updates > 0 and world_optimizer is None:
        raise ValueError("world update budget requested for a group without a world model")
    policy_version = 0
    world_version = 0
    scenarios = tuple(scenarios or scenario_tape("train", count=8, base_seed=train_config.seed, name="mixed"))
    scenario_ids = [item.tape_id for item in scenarios]
    scenario_index = 0
    env = M10Environment(env_config, scenarios[scenario_index])
    obs = env.reset()
    policy_hidden = None
    world_hidden = None
    episode_history: list[tuple[Mapping[str, Any], int]] = []
    current_counts = {"completed": 0, "expired": 0}
    previous_energy = float(env_config.uav_count * env_config.initial_energy)
    preference_rng = np.random.default_rng(train_config.seed + 0xA17E)
    world_batch_rng = np.random.default_rng(train_config.seed + 0x51A7)
    episode_preference = _choose_preference(preference_rng, device, spec.preference)
    preference_episode_assignments = int(spec.preference)
    preference_rollout_reuses = 0
    episode_index = 0
    last_rollout_episode_index: int | None = None
    episode_preference_used = False
    last_rollout_preference_used = False
    steps = policy_updates = world_updates = actor_calls = 0
    world_forward_calls = world_candidate_predictions_computed = world_candidate_predictions_consumed = 0
    distribution_shift_log: list[dict[str, Any]] = []
    valid_event_counts = np.zeros(len(EVENT_NAMES), dtype=np.int64)
    positive_event_counts = np.zeros(len(EVENT_NAMES), dtype=np.int64)
    invalid_event_reason_counts = {name: {"identity_missing": 0, "field_invalid": 0, "episode_boundary": 0} for name in EVENT_NAMES}
    ppo_log: list[dict[str, Any]] = []
    world_log: list[dict[str, Any]] = []
    rollout_ledger_rows: list[dict[str, Any]] = []
    feedback_counts: dict[str, int] = {}
    communication_proxy_messages = 0
    start_time = time.perf_counter()
    stop_reason = "budget_complete"
    rollout_number = 0
    recovery_update_verification: dict[str, Any] | None = None
    recovery_verification_optimizer_steps = 0
    while steps < max_steps and policy_updates < max_policy_updates and time.perf_counter() - start_time < wall_seconds:
        rollout_number += 1
        rollout_world_version = world_version
        rollout_policy_version = policy_version
        preference = _preference_at_rollout_boundary(episode_preference)
        if (spec.preference and last_rollout_episode_index == episode_index
                and last_rollout_preference_used and episode_preference_used):
            preference_rollout_reuses += 1
        transitions: list[dict[str, Any]] = []
        previous_transition: dict[str, Any] | None = None
        target_rollout = min(train_config.rollout_steps, max_steps - steps)
        for _ in range(target_rollout):
            if time.perf_counter() - start_time >= wall_seconds:
                stop_reason = "wall_clock_budget"
                break
            # The preference belongs to the episode, not this rollout chunk.
            preference = _preference_at_rollout_boundary(episode_preference)
            obs_t = _obs_tensor(obs, device)
            mask_np = _mask_safe(obs["mask"])
            mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=device)[None, :]
            policy_hidden_before = _zero_hidden(device) if policy_hidden is None else policy_hidden
            world_hidden_before = _zero_wm_hidden(device) if world_hidden is None else world_hidden
            with torch.no_grad():
                features, pair_messages, policy_hidden_after = policy.encode(obs_t, policy_hidden_before)
                if world is not None:
                    candidate_features, by_action = world.predict_all_candidates(features, world_hidden_before, obs, use_events=spec.event_auxiliary)
                    world_forward_calls += 1
                    world_candidate_predictions_computed += 25
                    world_candidate_predictions_consumed += int(mask_np.sum())
                else:
                    candidate_features = torch.zeros((1, 25, WORLD_FEATURE_DIM), dtype=torch.float32, device=device)
                    by_action = {}
                evaluated = policy.evaluate_encoded(features, pair_messages, preference, candidate_features, mask_t)
                distribution = evaluated["distribution"]
                action_tensor = distribution.sample()
                action = int(action_tensor.item())
                old_log_prob = float(distribution.log_prob(action_tensor).item())
                old_values = evaluated["critic_values"][0].detach().cpu().numpy().astype(np.float32)
            if not bool(mask_np[action]):
                raise RuntimeError("policy sampled an action outside the legal mask")
            actor_calls += 1
            old_obs = obs
            next_obs, legacy_scalar_reward, done, info = env.step(action)
            event_record = build_observed_event_labels(
                old_obs, next_obs, initial_energy=env_config.initial_energy,
                urgent_slack=train_config.urgent_slack_steps,
                low_energy_fraction=train_config.low_energy_fraction,
                same_episode=True,
            )
            vector_reward, consequence, current_counts, energy_now = _vector_reward(info, current_counts, previous_energy, env_config)
            previous_energy = energy_now
            feedback = str(info.get("feedback", "unknown"))
            feedback_counts[feedback] = feedback_counts.get(feedback, 0) + 1
            communication_proxy_messages += len(info.get("communication_delta", []))
            labels = event_record["labels"]
            label_mask = event_record["mask"]
            completion_records = info.get("completion_records", {})
            host_confirmed_total = sum(record.get("host_confirmation_time") is not None for record in completion_records.values())
            host_on_time_total = sum(record.get("host_confirmation_before_deadline") is True for record in completion_records.values())
            valid_event_counts += label_mask.astype(np.int64)
            positive_event_counts += labels.astype(np.int64) * label_mask.astype(np.int64)
            for name in EVENT_NAMES:
                for reason in ("identity_missing", "field_invalid", "episode_boundary"):
                    invalid_event_reason_counts[name][reason] += int(event_record["reasons"][name][reason])
            if world is not None:
                chosen_prediction = by_action[action]
                world_hidden_after = chosen_prediction["hidden"]
            else:
                world_hidden_after = None
            terminated = bool(info.get("terminated", False))
            truncated = bool(info.get("truncated", False))
            transition = {
                "obs": np.asarray(old_obs["flat"], dtype=np.float32).copy(),
                "next_obs": np.asarray(next_obs["flat"], dtype=np.float32).copy(),
                "observation_dict": old_obs, "next_observation_dict": next_obs,
                "mask": mask_np.copy(), "action": action, "old_log_prob": old_log_prob,
                "old_values": old_values, "next_values": np.zeros(2, dtype=np.float32),
                "vector_reward": vector_reward, "task_consequence": consequence,
                "raw_environment_reward": float(legacy_scalar_reward),
                "event_label": labels, "event_mask": label_mask,
                "state_target_valid": bool(np.isfinite(next_obs["flat"]).all()),
                "vector_reward_valid": bool(np.isfinite(vector_reward).all()),
                "task_consequence_valid": bool(np.isfinite(consequence).all()),
                "censor_reason": None,
                "policy_hidden_before": policy_hidden_before[0, 0].detach().cpu().numpy().copy(),
                "policy_hidden_after": policy_hidden_after[0, 0].detach().cpu().numpy().copy(),
                "world_hidden_before": world_hidden_before[0].detach().cpu().numpy().copy(),
                "candidate_features": candidate_features[0].detach().cpu().numpy().copy(),
                "preference": preference.detach().cpu().numpy().copy(),
                "episode_index": episode_index,
                "terminated": terminated, "truncated": truncated,
                "environment_truncated": bool(info.get("truncated", False)),
                "policy_version": rollout_policy_version, "world_version": rollout_world_version,
                "scenario_id": env.scenario.tape_id, "time": float(old_obs["time"]),
                "action_legal": bool(mask_np[action]),
            }
            if truncated and not terminated:
                # Time-limit truncation bootstraps from the final public state;
                # unlike termination, it is not assigned a zero continuation.
                with torch.no_grad():
                    terminal_tensor = _obs_tensor(next_obs, device)
                    terminal_mask = torch.as_tensor(_mask_safe(next_obs["mask"]), dtype=torch.bool, device=device)[None, :]
                    terminal_features, terminal_pair, _ = policy.encode(terminal_tensor, policy_hidden_after)
                    if world is not None:
                        terminal_candidate, _ = world.predict_all_candidates(terminal_features, world_hidden_after, next_obs, use_events=spec.event_auxiliary)
                        world_forward_calls += 1
                        world_candidate_predictions_computed += 25
                        world_candidate_predictions_consumed += int(terminal_mask.sum())
                    else:
                        terminal_candidate = torch.zeros((1, 25, WORLD_FEATURE_DIM), dtype=torch.float32, device=device)
                    terminal_eval = policy.evaluate_encoded(terminal_features, terminal_pair, preference, terminal_candidate, terminal_mask)
                    transition["next_values"] = terminal_eval["critic_values"][0].cpu().numpy().astype(np.float32)
            if transitions and previous_transition is not None and not previous_transition["terminated"]:
                previous_transition["next_values"] = old_values.copy()
            transitions.append(transition)
            rollout_ledger_rows.append({
                "record_type": "decision", "schema": "joint-decision-ledger/1.0.0",
                "run_id": run_id, "group": group, "seed": train_config.seed,
                "step": steps, "episode_id": env._episode_id, "episode_index": episode_index,
                "tape_id": env.scenario.tape_id, "time": float(old_obs["time"]),
                "public_observation": {
                    "flat": np.asarray(old_obs["flat"]).tolist(), "uavs": np.asarray(old_obs["uavs"]).tolist(),
                    "tasks": np.asarray(old_obs["tasks"]).tolist(), "entity_ids": old_obs.get("public_entity_ids", {}),
                    "version": old_obs.get("version"), "time": old_obs.get("time"),
                },
                "public_entity_ids": old_obs.get("public_entity_ids", {}),
                "legal_mask": mask_np.astype(int).tolist(), "legal_action_count": int(mask_np.sum()),
                "action": action, "action_legal": bool(mask_np[action]),
                "old_log_prob": old_log_prob, "current_value": old_values.tolist(), "next_state_value": None,
                "raw_environment_reward": float(legacy_scalar_reward),
                "training_reward": {"vector": vector_reward.tolist(), "valid": bool(transition["vector_reward_valid"])},
                "preference": transition["preference"].tolist(), "terminated": terminated,
                "truncated": bool(info.get("truncated", False)), "rollout_boundary": False,
                "bootstrap": {"valid": None, "value": None, "source": "pending", "policy_version": rollout_policy_version},
                "policy_version": rollout_policy_version, "world_model_version": rollout_world_version,
                "candidate_predictions": {
                    "status": "not_applicable" if world is None else "computed",
                    "world_model_version": None if world is None else rollout_world_version,
                    "policy_features_consumed": None if world is None else transition["candidate_features"].tolist(),
                    "legal_action_predictions": None if world is None else {
                        str(candidate_action): {key: value.detach().cpu().tolist() for key, value in prediction.items()
                                                if key in ("next_state", "vector_reward", "task_consequence", "event_logits", "policy_feature")}
                        for candidate_action, prediction in by_action.items() if mask_np[candidate_action]
                    },
                },
                "event_labels": {"status": "computed", "names": list(EVENT_NAMES), "labels": labels.tolist(),
                                 "mask": label_mask.tolist(), "invalid_reasons": event_record["reasons"]},
                "next_public_observation": {
                    "flat": np.asarray(next_obs["flat"]).tolist(), "uavs": np.asarray(next_obs["uavs"]).tolist(),
                    "tasks": np.asarray(next_obs["tasks"]).tolist(), "entity_ids": next_obs.get("public_entity_ids", {}),
                    "version": next_obs.get("version"), "time": next_obs.get("time"),
                },
                "physical_on_time_total": int(info["counts"]["completed"]),
                "physical_deadline_failure_total": int(info["counts"]["expired"]),
                "host_confirmed_total": int(host_confirmed_total), "host_on_time_total": int(host_on_time_total),
                "communication_proxy_messages": len(info.get("communication_delta", [])),
                "execution_feedback": {key: info.get(key) for key in (
                    "feedback", "command_submitted", "command_id", "lease_renewal", "lease_renewals",
                    "lease_renewal_delivery_results", "active_continuations", "feedback_log",
                    "communication_delta", "completion_records", "new_events", "terminated", "truncated",
                )},
            })
            steps += 1
            if spec.preference:
                episode_preference_used = True
            obs = next_obs
            episode_history.append((old_obs, action))
            previous_transition = transition
            policy_hidden = policy_hidden_after
            world_hidden = world_hidden_after
            if done:
                scenario_index = (scenario_index + 1) % len(scenarios)
                env = M10Environment(env_config, scenarios[scenario_index])
                obs = env.reset()
                policy_hidden = None
                world_hidden = None
                episode_history = []
                current_counts = {"completed": 0, "expired": 0}
                previous_energy = float(env_config.uav_count * env_config.initial_energy)
                previous_transition = None
                episode_index += 1
                episode_preference_used = False
                episode_preference = _preference_after_episode(
                    episode_preference, preference_rng, device, spec.preference,
                )
                preference_episode_assignments += int(spec.preference)
        if not transitions:
            break
        # Bootstrap the rollout boundary using the same behavior-version model.
        if previous_transition is not None and not (previous_transition["terminated"] or previous_transition["truncated"]):
            with torch.no_grad():
                next_tensor = _obs_tensor(obs, device)
                next_mask = torch.as_tensor(_mask_safe(obs["mask"]), dtype=torch.bool, device=device)[None, :]
                f_next, p_next, _ = policy.encode(next_tensor, policy_hidden)
                if world is not None:
                    next_candidate, _ = world.predict_all_candidates(f_next, world_hidden, obs, use_events=spec.event_auxiliary)
                    world_forward_calls += 1
                    world_candidate_predictions_computed += 25
                    world_candidate_predictions_consumed += int(next_mask.sum())
                else:
                    next_candidate = torch.zeros((1, 25, WORLD_FEATURE_DIM), dtype=torch.float32, device=device)
                next_eval = policy.evaluate_encoded(f_next, p_next, preference, next_candidate, next_mask)
            previous_transition["next_values"] = next_eval["critic_values"][0].cpu().numpy().astype(np.float32)
            previous_transition["truncated"] = True  # PPO rollout boundary; environment remains live.
        for row, transition in zip(rollout_ledger_rows, transitions):
            row["next_state_value"] = transition["next_values"].tolist()
            env_truncated = bool(transition.get("environment_truncated", False))
            row["terminated"] = bool(transition["terminated"])
            row["truncated"] = env_truncated
            row["rollout_boundary"] = bool(
                transition is transitions[-1] and not transition["terminated"] and not env_truncated
            )
            bootstrap_valid = not bool(transition["terminated"])
            if transition["terminated"]:
                source = "terminal_zero"
            elif env_truncated:
                source = "environment_time_limit"
            elif row["rollout_boundary"]:
                source = "rollout_boundary"
            else:
                source = "next_decision_value"
            row["bootstrap"] = {
                "valid": bootstrap_valid,
                "value": transition["next_values"].tolist() if bootstrap_valid else [0.0, 0.0],
                "source": source,
                "policy_version": int(transition["policy_version"]),
                "world_model_version": int(transition["world_version"]),
            }
            ledger.append(row)
        # Rollout evidence is durable before learning, but remains an
        # uncommitted tail until the matching post-update checkpoint exists.
        ledger_range = ledger.flush_rollout()
        rollout_ledger_rows.clear()
        if time.perf_counter() - start_time >= wall_seconds:
            stop_reason = "wall_clock_budget"
            break
        before_update_probe = _decision_probe(
            policy, world, obs, policy_hidden, world_hidden, episode_preference,
            use_events=spec.event_auxiliary, device=device,
        )
        if world is not None:
            world_forward_calls += 1
            world_candidate_predictions_computed += 25
            world_candidate_predictions_consumed += int(np.asarray(obs["mask"], dtype=np.bool_).sum())
        verify_this_update = False
        if policy_updates < max_policy_updates:
            verify_this_update = bool(verify_recovery_update and policy_updates + 1 == max_policy_updates)
            preupdate_state = copy.deepcopy(policy.state_dict()) if verify_this_update else None
            preupdate_optimizer = copy.deepcopy(policy_optimizer.state_dict()) if verify_this_update else None
            preupdate_world_state = copy.deepcopy(world.state_dict()) if verify_this_update and world is not None else None
            preupdate_world_optimizer = copy.deepcopy(world_optimizer.state_dict()) if verify_this_update and world_optimizer is not None else None
            preupdate_rng = _rng_snapshot(preference_rng) if verify_this_update else None
            preupdate_world_batch_rng = copy.deepcopy(world_batch_rng.bit_generator.state) if verify_this_update else None
            preupdate_probe = before_update_probe if verify_this_update else None
            preupdate_runtime = ({
                "environment": copy.deepcopy(env), "observation": copy.deepcopy(obs),
                "policy_hidden": None if policy_hidden is None else policy_hidden.detach().cpu().clone(),
                "world_hidden": None if world_hidden is None else world_hidden.detach().cpu().clone(),
                "episode_history": copy.deepcopy(episode_history),
                "episode_preference": None if episode_preference is None else episode_preference.detach().cpu().clone(),
                "scenario_index": scenario_index, "episode_index": episode_index,
                "world_batch_rng_state": copy.deepcopy(world_batch_rng.bit_generator.state),
                "preference_rng_state": copy.deepcopy(preference_rng.bit_generator.state),
                "data_order": list(range(len(transitions))),
                "transition_identity": [(item["scenario_id"], item["episode_index"], item["time"], item["action"])
                                        for item in transitions],
                "committed_steps_before_update": steps,
                "committed_updates_before_update": policy_updates,
            } if verify_this_update else None)
            update_fn = ppo_preference_update if spec.preco else ppo_weighted_preference_update
            ppo_metrics = update_fn(policy, transitions, policy_optimizer, train_config, device, event_group=group)
            policy_updates += 1
            policy_version += 1
            _atomic_json(output_dir / "run-status.json", {
                "status": "running", "run_id": run_id, "group": group,
                "environment_steps": steps, "policy_optimizer_steps": policy_updates,
                "world_optimizer_steps": world_updates, "last_committed_step": ledger.last_committed_step,
                "ledger_committed_policy_update": ledger.last_committed_update,
                "checkpoint_policy_update": max(0, policy_updates - 1),
                "checkpoint_is_behind_live_state": True,
            })
            ppo_metrics.update({"rollout": rollout_number, "steps": steps, "environment_transitions": len(transitions), "policy_version_before": rollout_policy_version, "policy_version_after": policy_version})
            ppo_log.append(ppo_metrics)
            if verify_this_update:
                reference_path = output_dir / "joint-update-recovery-reference.pt"
                output_dir.mkdir(parents=True, exist_ok=True)
                reference_payload = {
                    "run_id": run_id, "group": group, "protocol": JOINT_PROTOCOL,
                    "policy_state_dict": preupdate_state,
                    "policy_optimizer_state_dict": preupdate_optimizer,
                    "world_state_dict": preupdate_world_state,
                    "world_optimizer_state_dict": preupdate_world_optimizer,
                    "transitions": list(transitions),
                    "rng_state": preupdate_rng, "runtime_state": preupdate_runtime,
                    "world_replay_rng_state": preupdate_world_batch_rng,
                    "world_replay_updates": min(8, max(0, max_world_updates - world_updates)),
                    "world_event_enabled": bool(spec.event_auxiliary),
                    "preupdate_probe": preupdate_probe,
                    "expected_post_policy_state_dict": copy.deepcopy(policy.state_dict()),
                    "expected_post_policy_optimizer_state_dict": copy.deepcopy(policy_optimizer.state_dict()),
                    "fixed_tolerances": {"same_cuda_abs": 1e-6,
                                         "acceptance_scope": "same requested device and software environment; CPU/CUDA migration is diagnostic only"},
                }
                post_live_rng = _rng_snapshot(preference_rng)
                _atomic_torch_save(reference_path, reference_payload)
                reference = torch.load(reference_path, map_location=device, weights_only=False)
                shadow = JointGraphPreferencePolicy(env_config, history=True).to(device)
                shadow_world = ActionConditionedTemporalWorldModel().to(device) if spec.world_model else None
                shadow_optimizer, shadow_world_optimizer = _make_optimizers(
                    shadow, shadow_world, spec, train_config,
                )
                shadow.load_state_dict(reference["policy_state_dict"], strict=True)
                shadow_optimizer.load_state_dict(reference["policy_optimizer_state_dict"])
                if shadow_world is not None:
                    if reference.get("world_state_dict") is None or reference.get("world_optimizer_state_dict") is None:
                        raise RuntimeError("world-enabled recovery reference lacks world model/optimizer state")
                    shadow_world.load_state_dict(reference["world_state_dict"], strict=True)
                    shadow_world_optimizer.load_state_dict(reference["world_optimizer_state_dict"])
                recovered_runtime = reference["runtime_state"]
                if recovered_runtime["environment"].public_snapshot_digest() != env.public_snapshot_digest():
                    raise RuntimeError("serialized recovery environment snapshot changed")
                if (recovered_runtime["scenario_index"] != scenario_index
                        or recovered_runtime["episode_index"] != episode_index
                        or recovered_runtime["data_order"] != list(range(len(transitions)))
                        or recovered_runtime["transition_identity"] != [
                            (item["scenario_id"], item["episode_index"], item["time"], item["action"])
                            for item in reference["transitions"]
                        ]
                        or (episode_preference is not None and not torch.equal(
                            recovered_runtime["episode_preference"].cpu(), episode_preference.detach().cpu()
                        ))):
                    raise RuntimeError("serialized recovery runtime/preference/data order mismatch")
                _restore_rng_snapshot(reference["rng_state"], preference_rng)
                restored_probe = _decision_probe(
                    shadow, shadow_world, recovered_runtime["observation"],
                    None if recovered_runtime["policy_hidden"] is None else recovered_runtime["policy_hidden"].to(device),
                    None if recovered_runtime["world_hidden"] is None else recovered_runtime["world_hidden"].to(device),
                    None if recovered_runtime["episode_preference"] is None else recovered_runtime["episode_preference"].to(device),
                    use_events=spec.event_auxiliary, device=device,
                )
                probe_delta = _nested_max_abs_delta(restored_probe, reference["preupdate_probe"])
                if probe_delta > 1e-6:
                    raise RuntimeError(f"recovered next-action distribution differs: {probe_delta}")
                restored_log_probs = _joint_logprob_rows(shadow, reference["transitions"], device)
                behavior_logprob_delta = float((restored_log_probs - torch.as_tensor(
                    [item["old_log_prob"] for item in reference["transitions"]],
                    dtype=torch.float32, device=device,
                )).abs().max().detach())
                replay_metrics = update_fn(
                    shadow, reference["transitions"], shadow_optimizer, train_config, device, event_group=group,
                )
                parameter_delta = _nested_max_abs_delta(policy.state_dict(), shadow.state_dict())
                optimizer_delta = _nested_max_abs_delta(policy_optimizer.state_dict(), shadow_optimizer.state_dict())
                if max(behavior_logprob_delta, parameter_delta, optimizer_delta) > 1e-6:
                    raise RuntimeError("restored joint PPO update differs from the recorded live update")
                _restore_rng_snapshot(post_live_rng, preference_rng)
                recovery_update_verification = {
                    "status": "passed",
                    "full_runtime_restored": True,
                    "next_action_distribution_max_abs_delta": probe_delta,
                    "next_decision_logprob_max_abs_delta": behavior_logprob_delta,
                    "post_update_parameter_max_abs_delta": parameter_delta,
                    "post_update_optimizer_max_abs_delta": optimizer_delta,
                    "fixed_tolerances": reference["fixed_tolerances"],
                    "training_optimizer_steps": policy_updates,
                    "verification_optimizer_steps": 1,
                    "replay_policy_loss": replay_metrics["policy_loss"],
                    "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                }
                recovery_verification_optimizer_steps = recovery_update_verification["verification_optimizer_steps"]
        if episode_history:
            policy_hidden, world_hidden = _replay_hidden(policy, world, episode_history, device)
        after_ppo_probe = _decision_probe(
            policy, world, obs, policy_hidden, world_hidden, episode_preference,
            use_events=spec.event_auxiliary, device=device,
        )
        if world is not None:
            world_forward_calls += 1
            world_candidate_predictions_computed += 25
            world_candidate_predictions_consumed += int(np.asarray(obs["mask"], dtype=np.bool_).sum())
        distribution_row: dict[str, Any] = {
            "rollout": rollout_number,
            "behavior_to_after_ppo": _distribution_shift(before_update_probe, after_ppo_probe),
        }
        if world is not None and world_optimizer is not None and world_updates < max_world_updates:
            requested = min(8, max_world_updates - world_updates)
            updates = world_model_updates(
                policy, world, transitions, world_optimizer, train_config, device,
                updates=requested, event_enabled=spec.event_auxiliary, batch_rng=world_batch_rng,
                isolate_policy_gradient=spec.isolate_world_gradient,
            )
            world_updates += len(updates)
            world_version += 1 if updates else 0
            for update_index, row in enumerate(updates):
                row.update({"rollout": rollout_number, "update_index": world_updates - len(updates) + update_index + 1, "world_version_after": world_version})
                world_log.append(row)
            # Parameters and recurrent state are version-coupled.  Replay only
            # the current episode's public history under the new versions.
            if episode_history:
                policy_hidden, world_hidden = _replay_hidden(policy, world, episode_history, device)
            after_world_probe = _decision_probe(
                policy, world, obs, policy_hidden, world_hidden, episode_preference,
                use_events=spec.event_auxiliary, device=device,
            )
            world_forward_calls += 1
            world_candidate_predictions_computed += 25
            world_candidate_predictions_consumed += int(np.asarray(obs["mask"], dtype=np.bool_).sum())
            distribution_row["after_ppo_to_after_world"] = _distribution_shift(after_ppo_probe, after_world_probe)
            distribution_row["behavior_to_after_world"] = _distribution_shift(before_update_probe, after_world_probe)
            if (time.perf_counter() - start_time) >= wall_seconds:
                stop_reason = "wall_clock_budget"
                break
        else:
            # Group B has no WM; C/D may also reach their zero-update cap.
            after_world_probe = after_ppo_probe
            distribution_row["after_ppo_to_after_world"] = {"forward_kl": 0.0, "total_variation": 0.0}
            distribution_row["behavior_to_after_world"] = distribution_row["behavior_to_after_ppo"]
        if verify_this_update:
            _atomic_torch_save(output_dir / "ur-uninterrupted-result.pt", {
                "run_id": run_id, "group": group, "protocol": JOINT_PROTOCOL,
                "device": str(device), "environment_steps": steps,
                "policy_optimizer_steps": policy_updates,
                "world_optimizer_steps": world_updates,
                "policy_state_dict": copy.deepcopy(policy.state_dict()),
                "policy_optimizer_state_dict": copy.deepcopy(policy_optimizer.state_dict()),
                "world_state_dict": None if world is None else copy.deepcopy(world.state_dict()),
                "world_optimizer_state_dict": None if world_optimizer is None else copy.deepcopy(world_optimizer.state_dict()),
                "post_update_probe": copy.deepcopy(after_world_probe),
                "world_update_rows": copy.deepcopy(world_log),
                "preference_rng_after": copy.deepcopy(preference_rng.bit_generator.state),
                "world_batch_rng_after": copy.deepcopy(world_batch_rng.bit_generator.state),
                "policy_version": policy_version, "world_version": world_version,
            })
        checkpoint_counters = {
            "environment_steps": steps, "policy_optimizer_steps": policy_updates,
            "world_optimizer_steps": world_updates, "actor_calls": actor_calls,
            "world_forward_calls": world_forward_calls,
            "world_candidate_predictions_computed": world_candidate_predictions_computed,
            "world_candidate_predictions_consumed": world_candidate_predictions_consumed,
            "policy_version": policy_version, "world_version": world_version,
            "event_valid_elements": int(valid_event_counts.sum()), "episode_index": episode_index,
            "last_committed_step": int(ledger_range["last_step"]),
            "last_committed_policy_update": policy_updates,
        }
        transaction_id = f"txn-step-{steps:08d}-policy-{policy_updates:06d}-world-{world_updates:06d}"
        transaction_path = output_dir / "transactions" / f"{transaction_id}.pt"
        state_hash = save_joint_checkpoint(
            transaction_path, run_id=run_id, group=group, policy=policy, world=world,
            policy_optimizer=policy_optimizer, world_optimizer=world_optimizer, environment=env,
            observation=obs, policy_hidden=policy_hidden, world_hidden=world_hidden,
            episode_history=episode_history, preference_rng=preference_rng,
            episode_preference=episode_preference, world_batch_rng=world_batch_rng,
            recovery_probe=after_world_probe, counters=checkpoint_counters,
            identity={"protocol": JOINT_PROTOCOL, "environment": "Graph-5/25-action arrival_to_region",
                      "scenario_ids": scenario_ids, "scenarios": tuple(scenarios), "scenario_index": scenario_index,
                      "environment_config": asdict(env_config), "training_config": asdict(train_config)},
            persistence={"transaction_id": transaction_id, "ledger": dict(ledger_range),
                         "first_uncommitted_step": ledger_range["first_step"],
                         "last_committed_step": ledger_range["last_step"],
                         "policy_version": policy_version, "world_version": world_version},
        )
        verified = torch.load(transaction_path, map_location=device, weights_only=False)
        if (verified.get("run_id") != run_id or verified.get("counters") != checkpoint_counters
                or verified.get("persistence", {}).get("transaction_id") != transaction_id
                or verified["runtime_state"]["public_digest"] != env.public_snapshot_digest()
                or not finite_parameters(policy) or not finite_optimizer(policy_optimizer)
                or (world is not None and not finite_parameters(world))
                or (world_optimizer is not None and not finite_optimizer(world_optimizer))):
            raise RuntimeError("joint transaction checkpoint verification failed")
        # Compatibility alias for existing readers; the immutable transaction
        # and commit pointer remain authoritative.
        _atomic_bytes(output_dir / "last-recovery.pt", transaction_path.read_bytes())
        commit_pointer = ledger.publish_transaction(
            checkpoint_path=transaction_path, counters=checkpoint_counters,
            transaction_id=transaction_id, ledger_range=ledger_range,
        )
        if commit_pointer["checkpoint"]["sha256"] != state_hash:
            raise RuntimeError("committed checkpoint hash differs from the saved transaction hash")
        distribution_shift_log.append(distribution_row)
        last_rollout_episode_index = episode_index
        last_rollout_preference_used = episode_preference_used
        if steps >= max_steps:
            stop_reason = "environment_step_budget"
            break
        if policy_updates >= max_policy_updates:
            stop_reason = "policy_update_budget"
            break
        if world is not None and world_updates >= max_world_updates:
            stop_reason = "world_update_budget"
            break

    elapsed = time.perf_counter() - start_time
    if steps >= max_steps:
        stop_reason = "environment_step_budget"
    counters = {
        "environment_steps": steps, "policy_optimizer_steps": policy_updates,
        "world_optimizer_steps": world_updates, "actor_calls": actor_calls,
        "world_forward_calls": world_forward_calls,
        "world_candidate_predictions_computed": world_candidate_predictions_computed,
        "world_candidate_predictions_consumed": world_candidate_predictions_consumed,
        "policy_version": policy_version, "world_version": world_version,
        "event_valid_elements": int(valid_event_counts.sum()),
        "episode_index": episode_index,
    }
    recovery_probe = after_world_probe
    counters["world_forward_calls"] = world_forward_calls
    counters["world_candidate_predictions_computed"] = world_candidate_predictions_computed
    counters["world_candidate_predictions_consumed"] = world_candidate_predictions_consumed
    counters["last_committed_step"] = ledger.last_committed_step
    counters["last_committed_policy_update"] = ledger.last_committed_update
    pointer_path = output_dir / "ledger-commit.json"
    if not pointer_path.is_file():
        raise RuntimeError("no fully committed rollout transaction exists")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    committed_checkpoint = output_dir / pointer["checkpoint"]["path"]
    state_hash = hashlib.sha256(committed_checkpoint.read_bytes()).hexdigest()
    if state_hash != pointer["checkpoint"]["sha256"]:
        raise RuntimeError("last committed checkpoint hash validation failed")
    summary = {
        "status": "completed" if stop_reason in ("environment_step_budget", "budget_complete") else "stopped_at_budget",
        "run_id": run_id, "group": group, "label": spec.label, "protocol": JOINT_PROTOCOL,
        "gradient_isolation": bool(spec.isolate_world_gradient),
        "world_optimizer_scope": "world_only" if spec.isolate_world_gradient else ("shared_encoder_plus_world" if spec.world_model else "none"),
        "device": str(device), "seed": train_config.seed, "environment_steps": steps,
        "policy_optimizer_steps": policy_updates,
        "policy_optimizer_verification_steps": recovery_verification_optimizer_steps,
        "policy_optimizer_calls_total": policy_updates + recovery_verification_optimizer_steps,
        "world_optimizer_steps": world_updates,
        "wall_seconds": elapsed, "stop_reason": stop_reason,
        "actor_calls": actor_calls,
        "world_forward_calls": world_forward_calls,
        "world_candidate_predictions_computed": world_candidate_predictions_computed,
        "world_candidate_predictions_consumed": world_candidate_predictions_consumed,
        "communication_proxy_message_count": communication_proxy_messages,
        "feedback_counts": feedback_counts,
        "policy_version": policy_version, "world_version": world_version,
        "event_names": list(EVENT_NAMES), "event_valid_counts": dict(zip(EVENT_NAMES, valid_event_counts.tolist())),
        "event_positive_counts": dict(zip(EVENT_NAMES, positive_event_counts.tolist())),
        "event_negative_counts": dict(zip(EVENT_NAMES, (valid_event_counts - positive_event_counts).tolist())),
        "event_invalid_reasons": invalid_event_reason_counts,
        "preference_vectors": "one Dirichlet(1,1) vector per episode, held across rollout boundaries and resampled at reset",
        "preference_episode_assignments": preference_episode_assignments,
        "preference_rollout_boundary_holds": preference_rollout_reuses,
        "current_episode_preference": None if episode_preference is None else episode_preference.detach().cpu().tolist(),
        "recovery_update_verification": recovery_update_verification,
        "recovery_next_decision_probe": recovery_probe,
        "policy_distribution_shift_by_update": distribution_shift_log,
        "ppo_updates": ppo_log, "world_updates": world_log,
        "scenario_ids": scenario_ids, "task_completion_mode": env_config.task_completion_mode,
        "deadline_basis": env_config.deadline_basis, "task_counts": info.get("counts", {}) if steps else {},
        "training_ledger": "training-ledger.jsonl", "checkpoint": "last-recovery.pt",
        "checkpoint_sha256": state_hash,
        "event_gradient_updates": bool(spec.event_auxiliary and world_updates > 0 and any(row.get("event_shared_encoder_grad_norm", 0.0) > 0.0 for row in world_log)),
        "old_scalar_reward_consumed": False,
    }
    ledger.close()
    _atomic_json(output_dir / "smoke-summary.json", summary)
    _atomic_json(output_dir / "run-status.json", {
        "status": summary["status"], "run_id": run_id, "stop_reason": stop_reason,
        "last_committed_step": ledger.last_committed_step,
        "last_committed_policy_update": ledger.last_committed_update,
        "environment_steps": steps, "policy_optimizer_steps": policy_updates,
        "world_optimizer_steps": world_updates, "wall_seconds": elapsed,
    })
    return summary


def run_group(*, group: str, run_id: str, output_dir: str | Path, device: str,
              env_config: M10Config, train_config: JointTrainConfig,
              max_steps: int = 256, max_policy_updates: int = 4,
              max_world_updates: int = 32, wall_seconds: float = 1800.0,
              enforce_smoke_limits: bool = True,
              verify_recovery_update: bool = True,
              scenarios: Sequence[M10Scenario] | None = None,
              initial_policy_state_dict: Mapping[str, torch.Tensor] | None = None,
              initial_world_state_dict: Mapping[str, torch.Tensor] | None = None,
              initial_action_rng_state: torch.Tensor | None = None) -> dict[str, Any]:
    if group not in GROUPS:
        raise ValueError(f"group must be one of {tuple(GROUPS)}")
    if enforce_smoke_limits and (max_steps > 256 or max_policy_updates > 4 or max_world_updates > 32 or wall_seconds > 1800):
        raise ValueError("joint integration smoke limits are hard caps: 256 env steps, 4 PPO steps, 32 WM steps, 30 min")
    if env_config.task_completion_mode != "arrival_to_region" or env_config.deadline_basis != "physical_arrival":
        raise ValueError("joint pilot must use the frozen physical-arrival protocol")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    random.seed(train_config.seed)
    path = Path(output_dir)
    if path.exists():
        existing_files = {item.name for item in path.iterdir()} if path.is_dir() else {"<not-directory>"}
        if existing_files != {"resolved-config.json"}:
            raise FileExistsError(f"refusing to reuse a run directory with prior artifacts: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)
    _atomic_json(path / "run-status.json", {
        "status": "running", "run_id": run_id, "group": group, "started_at": time.time(),
    })
    try:
        summary = _run_joint(
            run_id=run_id, group=group, output_dir=path, device=dev,
            env_config=env_config, train_config=train_config, max_steps=max_steps,
            max_policy_updates=max_policy_updates, max_world_updates=max_world_updates,
            wall_seconds=wall_seconds, verify_recovery_update=verify_recovery_update,
            scenarios=scenarios, initial_policy_state_dict=initial_policy_state_dict,
            initial_world_state_dict=initial_world_state_dict,
            initial_action_rng_state=initial_action_rng_state,
        )
    except BaseException as exc:
        commit_path = path / "ledger-commit.json"
        commit = json.loads(commit_path.read_text(encoding="utf-8")) if commit_path.exists() else {}
        previous_status_path = path / "run-status.json"
        previous_status = json.loads(previous_status_path.read_text(encoding="utf-8")) if previous_status_path.exists() else {}
        _atomic_json(path / "run-status.json", {
            **previous_status, "status": "failed", "run_id": run_id, "group": group,
            "failure_type": type(exc).__name__, "failure": str(exc), "finished_at": time.time(),
            "last_committed_step": commit.get("last_committed_step", -1),
            "ledger_committed_policy_update": commit.get("last_committed_policy_update", 0),
            "uncommitted_tail_preserved": True,
        })
        raise
    _atomic_json(path / "run-status.json", {
        "status": summary.get("status", "unknown"), "run_id": run_id, "group": group,
        "stop_reason": summary.get("stop_reason", summary.get("metadata", {}).get("stop_reason", "unknown")),
        "environment_steps": summary.get("environment_steps", 0),
        "policy_optimizer_steps": summary.get("policy_optimizer_steps", 0),
        "world_optimizer_steps": summary.get("world_optimizer_steps", 0),
        "last_committed_step": summary.get("environment_steps", 0) - 1,
        "last_committed_policy_update": summary.get("policy_optimizer_steps", 0),
        "finished_at": time.time(),
    })
    return summary


__all__ = ["GROUPS", "GroupSpec", "run_group", "ppo_preference_update", "ppo_weighted_preference_update", "scalarized_preference_advantage", "world_model_updates", "save_joint_checkpoint", "load_joint_checkpoint"]
