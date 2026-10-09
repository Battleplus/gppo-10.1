"""Post-action Shadow hook that leaves the baseline GPPO environment intact."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import torch

from .contracts import snapshot_from_gppo
from .gppo_adapter import LatentContextStore, context_from_shadow
from .shadow import ShadowRequest, ShadowRuntime


@dataclass(frozen=True)
class ShadowEnvAudit:
    accepted_transitions: int
    stale_retries: int
    shadow_valid: int
    shadow_fallback: int
    belief_write_count: int
    action_mask_write_count: int
    graph_version_write_count: int
    action_version_write_count: int
    action_submission_count: int
    execution_rejection_count: int
    real_environment_mutation_count: int
    real_belief_mutation_count: int
    real_action_mask_mutation_count: int
    real_version_mutation_count: int
    decision_evidence_items: int


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _confirmed_evidence(env: Any, decision_time: float) -> tuple[Mapping[str, Any], ...]:
    """Export only confirmation records already visible in belief at decision time."""

    bridge = getattr(env, "runtime_bridge", None)
    belief = getattr(getattr(bridge, "adapter", None), "belief", None)
    confirmed = getattr(belief, "confirmed_events", ())
    by_id: dict[str, Mapping[str, Any]] = {}
    for event in confirmed:
        value = event.to_dict()
        received_at = value.get("received_at")
        if received_at is None:
            received_at = value.get("confirmed_at")
        if received_at is None:
            received_at = value.get("occurred_at", decision_time)
        received_at = float(received_at)
        if received_at > decision_time:
            continue
        event_id = str(value.get("event_id", ""))
        by_id[event_id] = {
            "source": str(value.get("source_event", "confirmed-belief")),
            "signal_type": str(value.get("event_type", "UNKNOWN")),
            "received_at": received_at,
            "payload": {
                "event_id": event_id,
                "status": str(value.get("status", "CONFIRMED")),
                "severity": float(value.get("severity", 0.0)),
                "affected_uavs": list(value.get("affected_uavs", ())),
                "affected_regions": list(value.get("affected_regions", ())),
                "affected_targets": list(value.get("affected_targets", ())),
            },
        }
    return tuple(by_id[key] for key in sorted(by_id))


def ensure_noop_action_compatibility(env: Any) -> bool:
    """Install the fixed NOOP index missing in the pinned rejection branch.

    GPPO-8.29@2a9bb9f references ``self.noop_action`` only when an assignment
    command is rejected, but its environment constructor never defines the
    attribute.  The action contract already fixes NOOP to ``action_space.n-1``.
    This shim writes only that immutable topology constant.
    """

    if hasattr(env, "noop_action"):
        return False
    action_space = getattr(env, "action_space", None)
    count = getattr(action_space, "n", None)
    if count is None or int(count) <= 1:
        raise ValueError("cannot derive fixed NOOP action from environment action_space")
    setattr(env, "noop_action", int(count) - 1)
    return True


class PostActionShadowEnv:
    """Delegate execution, then update private latent for the *next* decision.

    The wrapped environment remains the only object allowed to submit an
    action.  Shadow receives a detached pre-action snapshot and the confirmed
    executed action only after ``submit_action`` returns successfully.
    """

    def __init__(
        self,
        env: Any,
        runtime: ShadowRuntime,
        context_store: LatentContextStore,
        *,
        model_variant: str,
    ) -> None:
        self.env = env
        self.noop_compatibility_installed = ensure_noop_action_compatibility(env)
        self.runtime = runtime
        self.context_store = context_store
        self.model_variant = str(model_variant)
        if runtime.model.training or any(
            parameter.requires_grad for parameter in runtime.model.parameters()
        ):
            raise ValueError("PostActionShadowEnv requires an eval-mode, fully frozen world model")
        self._episode_index = -1
        self._episode_step = 0
        self._pending_context: Any | None = None
        self._pending_evidence: tuple[Mapping[str, Any], ...] = ()
        self._counts = {
            "accepted": 0,
            "stale": 0,
            "execution_rejected": 0,
            "env_mutation": 0,
            "belief_mutation": 0,
            "mask_mutation": 0,
            "version_mutation": 0,
            "evidence_items": 0,
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def reset(self, *args: Any, **kwargs: Any):
        result = self.env.reset(*args, **kwargs)
        self._episode_index += 1
        self._episode_step = 0
        self._pending_context = None
        self._pending_evidence = ()
        self.runtime.reset()
        self.context_store.reset(
            graph_version=int(getattr(self.env, "graph_version", 0)),
            action_version=int(getattr(self.env, "decision_version", 0)),
        )
        return result

    def begin_decision(self):
        context = self.env.begin_decision()
        self._pending_context = context
        self._pending_evidence = _confirmed_evidence(
            self.env, float(getattr(self.env, "current_time", self._episode_step))
        )
        self.context_store.prepare_decision(context.graph_version, context.action_version)
        return context

    def _versions(self) -> tuple[int, int]:
        return (
            int(getattr(self.env, "graph_version")),
            int(getattr(self.env, "decision_version")),
        )

    def submit_action(self, submission: Any, *args: Any, **kwargs: Any):
        decision = self._pending_context
        if decision is None:
            raise RuntimeError("begin_decision must precede submit_action")
        decision_time = float(getattr(self.env, "current_time", self._episode_step))
        result = self.env.submit_action(submission, *args, **kwargs)
        info: Mapping[str, Any] = result[-1] if isinstance(result, tuple) else {}
        if bool(info.get("stale_decision", False)):
            self._counts["stale"] += 1
            self._pending_context = None
            self._pending_evidence = ()
            return result

        proposed = int(getattr(submission, "action", submission))
        executed = int(info.get("repaired_action", proposed))
        # The pinned baseline does not export an explicit execution status.
        # A legal edge proposal repaired to NOOP is its fail-closed signal for
        # command/ACK/version rejection.  Do not let that proposal update WM.
        invalid_proposal = bool(info.get("invalid_action", False))
        explicit_accepted = info.get("execution_accepted")
        execution_accepted = (
            bool(explicit_accepted)
            if explicit_accepted is not None
            else not (executed != proposed and not invalid_proposal)
        )
        if not execution_accepted:
            self._counts["execution_rejected"] += 1
            self._episode_step += 1
            self._pending_context = None
            self._pending_evidence = ()
            rejected_info = dict(info)
            rejected_info.update(
                {
                    "execution_accepted": False,
                    "execution_rejected": True,
                    "executed_action": None,
                    "execution_status": "command_rejected_fail_closed_noop",
                }
            )
            return (*result[:-1], rejected_info)
        post_graph_version, post_action_version = self._versions()
        request = ShadowRequest(
            episode_id=f"rollout-{self._episode_index:06d}",
            step=self._episode_step,
            graph=snapshot_from_gppo(decision.graph),
            executed_action=executed,
            evidence=self._pending_evidence,
            action_version=int(decision.action_version),
            decision_time=decision_time,
            execution_accepted=True,
            expected_post_graph_version=post_graph_version,
            expected_post_action_version=post_action_version,
        )
        graph_after = result[0]
        env_hash_before = _canonical_hash(self.env.snapshot())
        bridge = getattr(self.env, "runtime_bridge", None)
        belief_hash_before = (
            bridge.adapter.get_snapshot_hash() if bridge is not None else "NO_BELIEF"
        )
        mask_before = graph_after.action_mask.detach().cpu().clone()
        versions_before = self._versions()
        shadow_result = self.runtime.observe(request, version_reader=self._versions)
        env_hash_after = _canonical_hash(self.env.snapshot())
        belief_hash_after = (
            bridge.adapter.get_snapshot_hash() if bridge is not None else "NO_BELIEF"
        )
        mask_after = result[0].action_mask.detach().cpu()
        versions_after = self._versions()
        self._counts["env_mutation"] += int(env_hash_before != env_hash_after)
        self._counts["belief_mutation"] += int(belief_hash_before != belief_hash_after)
        self._counts["mask_mutation"] += int(not torch.equal(mask_before, mask_after))
        self._counts["version_mutation"] += int(versions_before != versions_after)
        if any(
            self._counts[key]
            for key in ("env_mutation", "belief_mutation", "mask_mutation", "version_mutation")
        ):
            raise RuntimeError("Shadow inference mutated the real GPPO runtime")
        self.context_store.publish(
            context_from_shadow(
                shadow_result,
                model_variant=self.model_variant,
                latent_dim=self.context_store.config.latent_dim,
            )
        )
        self._counts["accepted"] += 1
        self._counts["evidence_items"] += len(self._pending_evidence)
        self._episode_step += 1
        self._pending_context = None
        self._pending_evidence = ()
        accepted_info = dict(info)
        accepted_info.update(
            {
                "execution_accepted": True,
                "execution_rejected": False,
                "executed_action": executed,
                "execution_status": "accepted",
            }
        )
        return (*result[:-1], accepted_info)

    def step(self, action: Any):
        # Legacy calls retain baseline semantics.  T-05 PPO must use the
        # begin_decision -> submit_action path so version binding is available.
        return self.env.step(action)

    def audit(self) -> ShadowEnvAudit:
        counters = self.runtime.counters
        return ShadowEnvAudit(
            accepted_transitions=self._counts["accepted"],
            stale_retries=self._counts["stale"],
            shadow_valid=counters["valid_count"],
            shadow_fallback=counters["fallback_count"],
            belief_write_count=counters["belief_write_count"],
            action_mask_write_count=counters["action_mask_write_count"],
            graph_version_write_count=counters["graph_version_write_count"],
            action_version_write_count=counters["action_version_write_count"],
            action_submission_count=counters["action_submission_count"],
            execution_rejection_count=self._counts["execution_rejected"],
            real_environment_mutation_count=self._counts["env_mutation"],
            real_belief_mutation_count=self._counts["belief_mutation"],
            real_action_mask_mutation_count=self._counts["mask_mutation"],
            real_version_mutation_count=self._counts["version_mutation"],
            decision_evidence_items=self._counts["evidence_items"],
        )


__all__ = [
    "PostActionShadowEnv",
    "ShadowEnvAudit",
    "ensure_noop_action_compatibility",
]
