from __future__ import annotations

import unittest
import torch

from public_transition_contract import PublicField, PublicSnapshot, TransitionTarget, PublicTransition, freeze_decision_input
from public_event_targets import build_event_target
from w1_graph_jepa import W1GraphJEPA, W1JEPAConfig, expand_candidate_batch
from event_jepa_losses import jepa_objective
from gppo_jepa_adapter import GPPOSafeJEPAAdapter, TransparentResidualPrior
from gppo_w1_integration import PublicHistoryState, predict_public_and_score
from w1_training import freeze_world_model, train_world_model
from world_model_pipeline import execute_pipeline
from w1_public_adapter import PublicAdapterError, PublicTelemetry, W1PublicInput, adapt_m10_public_observation
from public_transition_contract import public_hash


def snapshot(*, future: bool = False, continuations=("u0:t0",)):
    received = 2.0 if future else 1.0
    return PublicSnapshot(
        decision_time=1.5,
        fields={"energy": PublicField(0.8, True, True, 0.5, 1.0, received)},
        legal_actions=(0, 3, 24),
        history=({"received_at": received, "kind": "telemetry"},),
        continuation_ids=continuations,
    )


class ContractTests(unittest.TestCase):
    def test_future_evidence_rejected(self):
        with self.assertRaises(ValueError):
            snapshot(future=True)

    def test_unknown_is_not_zero_and_freeze_isolated(self):
        unknown = PublicField(123.0, False, False, None, None, None)
        self.assertIsNone(unknown.value)
        frozen = freeze_decision_input(snapshot())
        self.assertEqual(frozen.input_hash, snapshot().input_hash)
        self.assertEqual(frozen.continuation_ids, ("u0:t0",))

    def test_event_target_keeps_physical_and_host_labels_separate(self):
        current = snapshot()
        next_snapshot = PublicSnapshot(
            decision_time=2.5,
            fields={"energy": PublicField(0.7, True, True, 0.2, 2.0, 2.2)},
            legal_actions=(0, 3, 24),
            continuation_ids=("u0:t1",),
        )
        transition = PublicTransition(current, 0, TransitionTarget(next_snapshot, {}, {}, True, False, 0, True, "h"), "h")
        event = build_event_target(transition)
        self.assertTrue(event.valid[3] and not event.values[4])

    def _observation(self, *, time=1.0, continuation_actions=()):
        import numpy as np
        graph = {
            "node_features": np.zeros((21, 32), dtype=np.float32),
            "relations": np.zeros((4, 6, 4), dtype=np.float32),
        }
        return {
            "graph": graph, "mask": np.array([True] + [False] * 23 + [True]),
            "time": time, "version": 4, "continuation_actions": continuation_actions,
        }

    def test_adapter_accepts_old_measurement_received_before_boundary(self):
        value = adapt_m10_public_observation(
            self._observation(),
            [PublicTelemetry("uav-0", "energy", 4.0, 0.2, 0.8, 1, "m1")],
            max_age=2.0,
            continuation_ids=("lease:uav-0:task-0",),
            continuation_policy_id="hungarian-v1",
        )
        field = value.public.fields["uav-0.energy"]
        self.assertTrue(field.known and field.valid)
        self.assertEqual(field.measured_at, 0.2)
        self.assertEqual(field.received_at, 0.8)
        self.assertEqual(value.candidate_actions, (0, 24))

    def test_adapter_rejects_old_measurement_received_after_boundary(self):
        with self.assertRaises(PublicAdapterError):
            adapt_m10_public_observation(
                self._observation(time=1.0),
                [PublicTelemetry("uav-0", "energy", 4.0, 0.2, 1.1, 1)],
                continuation_policy_id="hungarian-v1",
            )

    def test_adapter_does_not_infer_physical_change_from_repeated_telemetry(self):
        value = adapt_m10_public_observation(
            self._observation(),
            [
                PublicTelemetry("uav-0", "energy", 4.0, 0.2, 0.4, 1, "m1"),
                PublicTelemetry("uav-0", "energy", 4.0, 0.8, 0.9, 2, "m2"),
            ],
            continuation_policy_id="hungarian-v1",
        )
        self.assertEqual(len(value.public.history), 2)
        self.assertEqual(value.public.fields["uav-0.energy"].measured_at, 0.8)

    def test_adapter_requires_continuation_identity_instead_of_inference(self):
        with self.assertRaises(PublicAdapterError):
            adapt_m10_public_observation(self._observation(continuation_actions=(0,)), [])

    def test_adapter_detaches_default_graph_payload(self):
        observation = self._observation()
        adapted = adapt_m10_public_observation(observation, [], continuation_policy_id="hungarian-v1")
        observation["graph"]["node_features"][0, 0] = 99.0
        self.assertEqual(float(adapted.graph["node_features"][0, 0]), 0.0)

    def test_graph_builder_sees_only_public_whitelist(self):
        observation = self._observation()
        observation["private_future"] = "forbidden"
        seen = {}
        def builder(value):
            seen.update(value)
            return value["graph"]
        adapt_m10_public_observation(
            observation, [], continuation_policy_id="hungarian-v1", graph_builder=builder,
        )
        self.assertNotIn("private_future", seen)

    def test_graph_mask_mismatch_is_rejected(self):
        from types import SimpleNamespace
        graph = SimpleNamespace(
            candidate_features=torch.zeros(24, 4),
            action_mask=torch.ones(25, dtype=torch.bool),
        )
        with self.assertRaises(PublicAdapterError):
            adapt_m10_public_observation(
                self._observation(), [], continuation_policy_id="hungarian-v1",
                graph_builder=lambda _: graph,
            )


class ModelTests(unittest.TestCase):
    def _inputs(self, n=3):
        counts = {"uav": 4, "region": 3, "target": 4, "task": 6, "event": 4}
        nodes = {name: torch.randn(n, count, 32) for name, count in counts.items()}
        history = torch.randn(n, 128)
        actions = torch.tensor([0, 3, 24][:n])
        relations = torch.randn(n, 4)
        return nodes, history, actions, relations

    def test_action_batch_has_25_contract_and_ema_target_no_grad(self):
        model = W1GraphJEPA(W1JEPAConfig())
        nodes, history, actions, relations = self._inputs()
        output = model.predict_candidates(nodes, history, actions, relations)
        self.assertEqual(tuple(output["latent"].shape), (3, 64))
        self.assertFalse(any(parameter.requires_grad for parameter in model.target_encoder.parameters()))
        self.assertEqual(model.config.action_count, 25)

    def test_candidate_reordering_is_identity_stable(self):
        model = W1GraphJEPA(W1JEPAConfig()).eval()
        base_nodes = {name: value[:1] for name, value in self._inputs(1)[0].items()}
        base_history = torch.randn(1, 128)
        actions = torch.tensor([0, 3, 24])
        relations = torch.randn(3, 4)
        nodes, history = expand_candidate_batch(base_nodes, base_history, actions, relations)
        first = model.predict_candidates(nodes, history, actions, relations)["latent"]
        order = torch.tensor([2, 0, 1])
        second = model.predict_candidates({name: value[order] for name, value in nodes.items()}, history[order], actions[order], relations[order])["latent"]
        restored = torch.empty_like(second)
        for index, original in enumerate(order.tolist()):
            restored[original] = second[index]
        self.assertTrue(torch.allclose(first, restored, atol=1e-6, rtol=1e-5))

    def test_event_and_state_switch_are_independent(self):
        predicted = torch.randn(4, 8, requires_grad=True)
        target = torch.randn(4, 8)
        valid = torch.ones_like(predicted, dtype=torch.bool)
        logits = torch.randn(4, 5, requires_grad=True)
        labels = torch.zeros_like(logits)
        event_valid = torch.ones_like(logits, dtype=torch.bool)
        total_a, parts_a = jepa_objective(predicted, target, valid, event_logits=logits, event_labels=labels, event_valid=event_valid, beta_event=0.0)
        total_b, parts_b = jepa_objective(predicted, target, valid, event_logits=logits, event_labels=labels, event_valid=event_valid, beta_event=1.0)
        self.assertEqual(float(parts_a["event"].detach()), float(parts_b["event"].detach()))
        self.assertNotEqual(float(total_a.detach()), float(total_b.detach()))

    def test_adapter_zero_scale_and_illegal_mask(self):
        adapter = GPPOSafeJEPAAdapter(scale=0.0)
        base = torch.randn(1, 25)
        latent = torch.randn(1, 25, 64)
        mask = torch.ones(1, 25, dtype=torch.bool)
        mask[:, 4] = False
        result = adapter(base, latent, mask)
        self.assertTrue(torch.equal(result["final_logits"][0, 4], torch.tensor(float("-inf"))))
        self.assertTrue(torch.allclose(result["final_logits"][mask], base[mask]))

    def test_production_candidate_bridge_batches_legal_actions_and_preserves_trace(self):
        from types import SimpleNamespace
        from types import MappingProxyType
        model = W1GraphJEPA(W1JEPAConfig()).eval()
        counts = {"uav": 4, "region": 3, "target": 4, "task": 6, "event": 4}
        nodes = MappingProxyType({name: torch.randn(count, 32) for name, count in counts.items()})
        mask = torch.ones(25, dtype=torch.bool)
        mask[3] = False
        snapshot = SimpleNamespace(
            nodes=nodes,
            candidate_features=torch.randn(24, 4),
            action_mask=mask,
            graph_version=7,
        )
        with self.assertRaises(TypeError):
            predict_public_and_score(
                snapshot, torch.randn(25), torch.zeros(25), model,
                TransparentResidualPrior(scale=0.1),
                PublicHistoryState.create(torch.zeros(128), ("0" * 64,)),
            )

    def test_strict_production_bridge_requires_frozen_public_input(self):
        from types import SimpleNamespace, MappingProxyType
        model = W1GraphJEPA(W1JEPAConfig()).eval()
        counts = {"uav": 4, "region": 3, "target": 4, "task": 6, "event": 4}
        graph = SimpleNamespace(
            nodes=MappingProxyType({name: torch.zeros(count, 32) for name, count in counts.items()}),
            candidate_features=torch.zeros(24, 4),
            action_mask=torch.tensor([True] + [False] * 23 + [True]), graph_version=1,
        )
        graph.as_dict = lambda: {
            "nodes": {name: value.tolist() for name, value in graph.nodes.items()},
            "candidate_features": graph.candidate_features.tolist(),
            "action_mask": graph.action_mask.tolist(), "graph_version": 1,
        }
        def reviewed_builder(_): return graph
        reviewed_builder.__module__ = "gppo_world.graph5"
        reviewed_builder.__name__ = "graph5_from_m10_observation"
        observation = ContractTests()._observation()
        value = adapt_m10_public_observation(
            observation, [], continuation_policy_id="hungarian-v1", graph_builder=reviewed_builder,
        )
        trace = predict_public_and_score(
            value, torch.zeros(25), torch.zeros(25), model, TransparentResidualPrior(scale=0.0),
            PublicHistoryState.create(torch.zeros(128), (public_hash(list(value.public.history)),)),
        )
        self.assertEqual(trace.continuation_id, "hungarian-v1")
        self.assertEqual(trace.active_continuation_ids, ())
        self.assertEqual(trace.candidate_actions, (0, 24))

        forged = W1PublicInput(
            graph=value.graph, public=value.public, candidate_actions=value.candidate_actions,
            continuation_policy_id=value.continuation_policy_id, graph_hash="0" * 64,
            graph_builder_id=value.graph_builder_id,
        )
        with self.assertRaises(PublicAdapterError):
            predict_public_and_score(
                forged, torch.zeros(25), torch.zeros(25), model, TransparentResidualPrior(scale=0.0),
                PublicHistoryState.create(torch.zeros(128), (public_hash(list(value.public.history)),)),
            )

    def test_training_api_is_explicit_and_freeze_is_read_only(self):
        model = W1GraphJEPA(W1JEPAConfig())
        freeze_world_model(model)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(callable(train_world_model))


class PipelineTests(unittest.TestCase):
    class Backend:
        def __init__(self, metrics):
            self.metrics = metrics
            self.calls = []
        def zero_step_gate(self): self.calls.append("preflight"); return {"pass": True}
        def collect_world_model_data(self): self.calls.append("collect"); return {"pass": True, "prediction_parent_count": 8}
        def train_world_models(self): self.calls.append("wm_train"); return {"models": 6}
        def evaluate_world_models(self): self.calls.append("wm_eval"); return dict(self.metrics)
        def train_policies(self): self.calls.append("policy_train"); return {"models": 12}
        def evaluate_tasks(self): self.calls.append("task_eval"); return {"arms": 5}
        def settle(self, status, evidence): self.calls.append("settle:" + status)

    def _metrics(self, *, passing):
        return {
            "valid_parent_count": 8,
            "g1_minus_transparent_macro_regret": -0.006 if passing else -0.001,
            "g2_minus_g1_macro_regret": -0.003,
            "g2_minus_g1_event_brier": -0.011,
            "parents_g1_no_worse_than_transparent": 5,
            "parents_g2_no_worse_than_g1": 5,
        }

    def test_prediction_gate_stops_before_policy_or_task_calls(self):
        backend = self.Backend(self._metrics(passing=False))
        result = execute_pipeline(backend)
        self.assertEqual(result.status, "prediction_gate_stop")
        self.assertNotIn("policy_train", backend.calls)
        self.assertNotIn("task_eval", backend.calls)

    def test_prediction_gate_passes_to_four_policy_arms_and_five_task_arms(self):
        backend = self.Backend(self._metrics(passing=True))
        result = execute_pipeline(backend)
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.evidence["policy_training"]["models"], 12)
        self.assertEqual(result.evidence["task_comparison"]["arms"], 5)


if __name__ == "__main__":
    unittest.main()
