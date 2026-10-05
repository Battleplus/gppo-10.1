"""Read-only T-04 Shadow runtime for an event-aware world model."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from time import perf_counter_ns
from typing import Any, Callable, Mapping

import torch

from .calibration import ShadowCalibration, input_ood_score, prediction_uncertainty
from .contracts import GraphSnapshot
from .model import EventAwareGraphWorldModel


VersionReader = Callable[[], tuple[int, int]]


def graph_snapshot_sha256(graph: GraphSnapshot) -> str:
    digest = hashlib.sha256()
    for group in (graph.nodes, graph.edge_index, graph.edge_attr):
        for name, value in sorted(group.items(), key=lambda item: str(item[0])):
            digest.update(str(name).encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    digest.update(graph.candidate_edges.numpy().tobytes())
    digest.update(graph.action_mask.numpy().tobytes())
    digest.update(str(graph.graph_version).encode("ascii"))
    return digest.hexdigest()


def _evidence_sha256(evidence: tuple[Mapping[str, Any], ...]) -> str:
    payload = json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _request_sha256(request: "ShadowRequest", graph_hash: str, evidence_hash: str) -> str:
    value = {
        "episode_id": request.episode_id,
        "step": request.step,
        "graph_sha256": graph_hash,
        "evidence_sha256": evidence_hash,
        "executed_action": request.executed_action,
        "execution_accepted": request.execution_accepted,
        "input_graph_version": request.graph.graph_version,
        "input_action_version": request.action_version,
        "expected_post_graph_version": request.expected_post_graph_version,
        "expected_post_action_version": request.expected_post_action_version,
        "decision_time": request.decision_time,
    }
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ShadowRequest:
    episode_id: str
    step: int
    graph: GraphSnapshot
    executed_action: int | None
    evidence: tuple[Mapping[str, Any], ...]
    action_version: int
    decision_time: float
    execution_accepted: bool
    expected_post_graph_version: int
    expected_post_action_version: int

    def __post_init__(self) -> None:
        if not self.episode_id or self.step < 0 or not math.isfinite(self.decision_time):
            raise ValueError("shadow request identity/time is invalid")
        if self.execution_accepted:
            if self.executed_action is None:
                raise ValueError("accepted shadow transition requires executed_action")
            if not 0 <= self.executed_action < self.graph.num_actions:
                raise ValueError("executed_action is outside the input graph action space")
            if not bool(self.graph.action_mask[self.executed_action]):
                raise ValueError("executed_action is illegal in the input action mask")
        elif self.executed_action is not None:
            raise ValueError("rejected execution cannot carry executed_action")
        for item in self.evidence:
            if float(item.get("received_at", self.decision_time)) > self.decision_time:
                raise ValueError("future evidence is forbidden in Shadow input")


@dataclass(frozen=True)
class ShadowResult:
    valid: bool
    fallback_reason: str | None
    episode_id: str
    step: int
    graph_version: int
    action_version: int
    post_graph_version: int
    post_action_version: int
    model_version: str
    latency_ms: float
    ood_score: float
    uncertainty: float | None
    latent: tuple[float, ...]
    input_graph_sha256: str
    input_evidence_sha256: str
    input_request_sha256: str
    history_reset: bool
    state_delta_sha256: str | None


class ShadowRuntime:
    """Runs inference and commits only private latent state after all gates.

    The runtime receives detached :class:`GraphSnapshot` values and has no
    reference to environment mutation APIs, action submission, ACK, lease or
    fencing objects.  Its public result is observational and never an action.
    """

    def __init__(
        self,
        model: EventAwareGraphWorldModel,
        calibration: ShadowCalibration,
        *,
        model_version: str,
    ) -> None:
        self.model = model.eval()
        self.calibration = calibration
        self.model_version = str(model_version)
        self._hidden: torch.Tensor | None = None
        self._last_episode: str | None = None
        self._last_step: int | None = None
        self._records: list[ShadowResult] = []
        self._counters = {
            "inference_count": 0,
            "valid_count": 0,
            "fallback_count": 0,
            "stale_before_count": 0,
            "stale_after_count": 0,
            "timeout_count": 0,
            "ood_count": 0,
            "high_uncertainty_count": 0,
            "exception_count": 0,
            "input_mutation_count": 0,
            "belief_write_count": 0,
            "action_mask_write_count": 0,
            "graph_version_write_count": 0,
            "action_version_write_count": 0,
            "action_submission_count": 0,
        }

    @property
    def records(self) -> tuple[ShadowResult, ...]:
        return tuple(self._records)

    @property
    def counters(self) -> dict[str, int]:
        return dict(self._counters)

    def reset(self) -> None:
        self._hidden = None
        self._last_episode = None
        self._last_step = None

    def _fallback(
        self,
        request: ShadowRequest,
        reason: str,
        started_ns: int,
        graph_hash: str,
        evidence_hash: str,
        ood_score: float,
        *,
        latency_injection_ms: float,
        uncertainty: float | None = None,
        history_reset: bool = False,
    ) -> ShadowResult:
        self._counters["fallback_count"] += 1
        key = f"{reason}_count"
        if key in self._counters:
            self._counters[key] += 1
        result = ShadowResult(
            valid=False,
            fallback_reason=reason,
            episode_id=request.episode_id,
            step=request.step,
            graph_version=request.graph.graph_version,
            action_version=request.action_version,
            post_graph_version=request.expected_post_graph_version,
            post_action_version=request.expected_post_action_version,
            model_version=self.model_version,
            latency_ms=(perf_counter_ns() - started_ns) / 1e6 + latency_injection_ms,
            ood_score=ood_score,
            uncertainty=uncertainty,
            latent=(0.0,) * (self.model.config.hidden_dim + self.model.config.stochastic_dim),
            input_graph_sha256=graph_hash,
            input_evidence_sha256=evidence_hash,
            input_request_sha256=_request_sha256(request, graph_hash, evidence_hash),
            history_reset=history_reset,
            state_delta_sha256=None,
        )
        self._records.append(result)
        return result

    @torch.no_grad()
    def observe(
        self,
        request: ShadowRequest,
        *,
        version_reader: VersionReader | None = None,
        latency_injection_ms: float = 0.0,
        force_exception: bool = False,
    ) -> ShadowResult:
        started = perf_counter_ns()
        self._counters["inference_count"] += 1
        graph_hash = graph_snapshot_sha256(request.graph)
        evidence_hash = _evidence_sha256(request.evidence)
        if not request.execution_accepted:
            return self._fallback(
                request,
                "rejected_execution",
                started,
                graph_hash,
                evidence_hash,
                -1.0,
                latency_injection_ms=latency_injection_ms,
            )
        read_versions = version_reader or (
            lambda: (request.expected_post_graph_version, request.expected_post_action_version)
        )
        before_graph_version, before_action_version = read_versions()
        if (
            before_graph_version != request.expected_post_graph_version
            or before_action_version != request.expected_post_action_version
        ):
            return self._fallback(
                request,
                "stale_before",
                started,
                graph_hash,
                evidence_hash,
                -1.0,
                latency_injection_ms=latency_injection_ms,
            )
        ood_score = input_ood_score(request.graph, self.calibration)
        if ood_score > self.calibration.ood_score_threshold:
            return self._fallback(
                request,
                "ood",
                started,
                graph_hash,
                evidence_hash,
                ood_score,
                latency_injection_ms=latency_injection_ms,
            )
        history_reset = (
            self._last_episode != request.episode_id
            or self._last_step is None
            or request.step != self._last_step + 1
        )
        hidden = None if history_reset else self._hidden
        try:
            if force_exception:
                raise RuntimeError("injected shadow inference failure")
            output, candidate_hidden = self.model.step(
                request.graph,
                request.executed_action,
                hidden,
                sample=False,
                evidence=request.evidence,
            )
        except Exception:
            return self._fallback(
                request,
                "exception",
                started,
                graph_hash,
                evidence_hash,
                ood_score,
                latency_injection_ms=latency_injection_ms,
                history_reset=history_reset,
            )
        after_graph_version, after_action_version = read_versions()
        if (
            after_graph_version != request.expected_post_graph_version
            or after_action_version != request.expected_post_action_version
        ):
            return self._fallback(
                request,
                "stale_after",
                started,
                graph_hash,
                evidence_hash,
                ood_score,
                latency_injection_ms=latency_injection_ms,
                history_reset=history_reset,
            )
        latency_ms = (perf_counter_ns() - started) / 1e6 + latency_injection_ms
        if latency_ms > self.calibration.timeout_ms:
            return self._fallback(
                request,
                "timeout",
                started,
                graph_hash,
                evidence_hash,
                ood_score,
                latency_injection_ms=latency_injection_ms,
                history_reset=history_reset,
            )
        uncertainty = prediction_uncertainty(output, self.calibration)
        if uncertainty > self.calibration.uncertainty_threshold:
            return self._fallback(
                request,
                "high_uncertainty",
                started,
                graph_hash,
                evidence_hash,
                ood_score,
                latency_injection_ms=latency_injection_ms,
                uncertainty=uncertainty,
                history_reset=history_reset,
            )
        if graph_snapshot_sha256(request.graph) != graph_hash or _evidence_sha256(request.evidence) != evidence_hash:
            self._counters["input_mutation_count"] += 1
            return self._fallback(
                request,
                "input_mutation",
                started,
                graph_hash,
                evidence_hash,
                ood_score,
                latency_injection_ms=latency_injection_ms,
                uncertainty=uncertainty,
                history_reset=history_reset,
            )
        self._hidden = candidate_hidden.detach().clone()
        self._last_episode = request.episode_id
        self._last_step = request.step
        self._counters["valid_count"] += 1
        latent_tensor = torch.cat([output["h"], output["z"]]).detach().cpu()
        state_delta_bytes = output["state_delta"].detach().cpu().contiguous().numpy().tobytes()
        result = ShadowResult(
            valid=True,
            fallback_reason=None,
            episode_id=request.episode_id,
            step=request.step,
            graph_version=request.graph.graph_version,
            action_version=request.action_version,
            post_graph_version=request.expected_post_graph_version,
            post_action_version=request.expected_post_action_version,
            model_version=self.model_version,
            latency_ms=latency_ms,
            ood_score=ood_score,
            uncertainty=uncertainty,
            latent=tuple(float(item) for item in latent_tensor),
            input_graph_sha256=graph_hash,
            input_evidence_sha256=evidence_hash,
            input_request_sha256=_request_sha256(request, graph_hash, evidence_hash),
            history_reset=history_reset,
            state_delta_sha256=hashlib.sha256(state_delta_bytes).hexdigest(),
        )
        self._records.append(result)
        return result
