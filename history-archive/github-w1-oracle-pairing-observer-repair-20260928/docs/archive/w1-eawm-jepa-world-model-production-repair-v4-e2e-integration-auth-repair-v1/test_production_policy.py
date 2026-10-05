from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from collections import Counter

import torch
from torch import nn
from torch.distributions import Categorical

from production_policy import (
    ACTION_COUNT, NOOP_ACTION, POLICY_METHODS, POLICY_SEEDS, CallAccounting,
    DecisionPriorPolicy, PolicyProductionError, PriorConditionedPolicy,
    evaluate_task_confirmation, policy_schedule, train_policy_routes,
)

ROOT = Path(__file__).resolve().parent
MATRIX = json.loads((ROOT / "experiment-matrix.json").read_text(encoding="utf-8"))
REQUEST = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))


class FakeLedger:
    def __init__(self):
        self.amounts = Counter()
        self.calls = []

    def call(self, name, amounts, function, *args, **kwargs):
        self.calls.append(name)
        self.amounts.update(amounts)
        return function(*args, **kwargs)


class FakeBoundary:
    def __init__(self):
        self.kinds = Counter()

    def _call(self, kind, operation, *args, **kwargs):
        self.kinds[kind] += 1
        return operation(*args, **kwargs)

    def environment(self, operation, *args, **kwargs):
        return self._call("environment", operation, *args, **kwargs)

    def model(self, operation, *args, **kwargs):
        return self._call("model", operation, *args, **kwargs)

    def optimizer(self, operation, *args, **kwargs):
        return self._call("optimizer", operation, *args, **kwargs)

    def checkpoint(self, operation, *args, **kwargs):
        return self._call("checkpoint", operation, *args, **kwargs)


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def encode(self, obs, hidden=None):
        batch = obs.shape[0]
        feature = self.anchor.expand(batch, 128)
        pair = self.anchor.expand(batch, 4, 6, 2)
        next_hidden = self.anchor.expand(1, batch, 128)
        return feature, pair, next_hidden

    def evaluate_encoded(self, features, pair, preference, candidate_features, mask):
        batch = features.shape[0]
        logits = self.anchor.expand(batch, ACTION_COUNT)
        legal = mask.to(torch.bool)
        distribution = Categorical(logits=logits.masked_fill(~legal, -torch.inf))
        values = self.anchor.expand(batch, 2)
        return {"logits": logits, "distribution": distribution,
                "critic_values": values, "state_values": values,
                "probabilities": distribution.probs}


class AdvancingClock:
    def __init__(self):
        self.value = 0.0

    def perf_counter(self):
        self.value += 0.002
        return self.value

    def process_time(self):
        self.value += 0.001
        return self.value


def decision_inputs(observation):
    mask = torch.zeros(ACTION_COUNT, dtype=torch.bool)
    if observation.get("opportunity", True):
        mask[0] = True
        mask[1] = True
    mask[NOOP_ACTION] = True
    return {
        "obs_tensor": torch.zeros((1, 16), dtype=torch.float32),
        "preference": torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        "candidate_features": torch.zeros((1, ACTION_COUNT, 17), dtype=torch.float32),
        "mask": mask.unsqueeze(0),
    }


def transparent_scores(_observation, _inputs):
    scores = torch.zeros(ACTION_COUNT)
    scores[0] = 1.0
    scores[1] = 4.0
    return scores


def world_predictor(_method, model, _observation, actions, _mask):
    result = torch.zeros(actions.numel())
    if 1 in actions.tolist():
        result[actions.tolist().index(1)] = float(model["residual"])
    return {"candidate_actions": actions.clone(), "utility_residuals": result}


class PriorTests(unittest.TestCase):
    def test_frozen_matrix_routes_and_prior_configuration(self):
        schedule = policy_schedule(MATRIX, REQUEST)
        routes = schedule.routes()
        self.assertEqual(len(routes), 12)
        self.assertEqual(tuple(route.seed for route in routes[:3]), POLICY_SEEDS)
        self.assertEqual(schedule.steps_per_route, 2048)
        self.assertEqual(schedule.updates_per_route, 128)
        self.assertEqual(schedule.updates_per_rollout, 4)
        self.assertEqual(MATRIX["prior_configuration"]["T"]["prior_scale"], 0.1)

    def test_direct_prior_changes_logits_and_illegal_action_stays_masked(self):
        wrapper = PriorConditionedPolicy(TinyPolicy())
        features = torch.zeros((1, ACTION_COUNT, 17))
        prior = torch.zeros((1, ACTION_COUNT))
        prior[0, 0] = 1000.0
        prior[0, 1] = 2.0
        augmented = torch.cat((features, prior.unsqueeze(-1)), dim=-1)
        mask = torch.zeros((1, ACTION_COUNT), dtype=torch.bool)
        mask[0, 1] = True
        mask[0, NOOP_ACTION] = True
        result = wrapper.evaluate_encoded(torch.zeros((1, 128)), torch.zeros((1, 4, 6, 2)),
                                          torch.tensor([[0.5, 0.5]]), augmented, mask)
        self.assertEqual(int(result["probabilities"].argmax().item()), 1)
        self.assertEqual(result["probabilities"][0, 0].detach().item(), 0.0)
        self.assertTrue(torch.isneginf(result["final_logits"][0, 0]))
        self.assertEqual(float(result["prior_logits"][0, 1]), 2.0)

    def test_three_model_candidate_prior_is_batched_and_costed(self):
        ledger, boundary = FakeLedger(), FakeBoundary()
        policy = PriorConditionedPolicy(TinyPolicy())
        selection = DecisionPriorPolicy(
            method="G2", seed=8301, policy=policy,
            prior_configuration=MATRIX["prior_configuration"],
            input_builder=decision_inputs, transparent_scorer=transparent_scores,
            world_models=({"residual": 2.0}, {"residual": 4.0}, {"residual": 6.0}),
            world_predictor=world_predictor, calls=CallAccounting(ledger, boundary),
            parent="train-0048", repeat=1, clock=AdvancingClock(),
            selector=lambda distribution: int(distribution.probs.argmax().item()),
        )
        decision = selection.select({"opportunity": True}, decision_step=7)
        self.assertEqual(decision.action, 1)
        self.assertAlmostEqual(decision.trace["prior_logits"][1], 0.8, places=6)
        self.assertEqual(ledger.amounts["world_batch_forwards"], 3)
        self.assertEqual(ledger.amounts["world_sample_evaluations"], 9)
        self.assertEqual(boundary.kinds["model"], 5)
        self.assertEqual(decision.cost["world_model_forwards"], 3)
        self.assertEqual(decision.cost["candidate_count"], 3)
        self.assertIsNotNone(decision.values)
        self.assertEqual(tuple(decision.values.shape), (2,))

    def test_bootstrap_value_uses_critic_without_world_forward_or_action(self):
        ledger, boundary = FakeLedger(), FakeBoundary()
        selection = DecisionPriorPolicy(
            method="G2", seed=8301, policy=PriorConditionedPolicy(TinyPolicy()),
            prior_configuration=MATRIX["prior_configuration"],
            input_builder=decision_inputs, transparent_scorer=transparent_scores,
            world_models=({"residual": 2.0},), world_predictor=world_predictor,
            calls=CallAccounting(ledger, boundary), parent="policy-training", repeat=0,
        )
        value = selection.bootstrap_value({"opportunity": True})
        self.assertEqual(tuple(value.shape), (2,))
        self.assertEqual(selection.decision_count, 0)
        self.assertEqual(ledger.amounts["world_batch_forwards"], 0)
        self.assertEqual(ledger.amounts["encode_sample_evaluations"], 1)
        self.assertEqual(ledger.amounts["actor_sample_evaluations"], 1)

    def test_world_outcome_residual_uses_task_and_energy_preference_components(self):
        outcomes = torch.zeros((2, 6), dtype=torch.float32)
        outcomes[0, 4], outcomes[0, 3] = 2.0, 1.0
        outcomes[1, 4], outcomes[1, 3] = -1.0, 0.5
        dense = DecisionPriorPolicy._residuals(
            {"candidate_actions": torch.tensor([0, 24]), "outcomes": outcomes},
            torch.tensor([0, 24]), torch.tensor([[0.2, 0.8]]),
        )
        self.assertAlmostEqual(float(dense[0]), 1.0, places=6)
        self.assertAlmostEqual(float(dense[24]), 0.3, places=6)

    def test_predictor_receives_only_legal_candidates_and_requires_exact_identity_order(self):
        legal_actions = torch.tensor([0, 1, NOOP_ACTION])
        legal_mask = torch.zeros(ACTION_COUNT, dtype=torch.bool)
        legal_mask[legal_actions] = True
        seen = []

        def run_predictor(response_builder):
            def predictor(_method, _model, _observation, actions, _mask):
                seen.append(actions.clone())
                return response_builder(actions)

            selection = DecisionPriorPolicy(
                method="G1", seed=8301, policy=PriorConditionedPolicy(TinyPolicy()),
                prior_configuration=MATRIX["prior_configuration"],
                input_builder=lambda _observation: {
                    **decision_inputs({"opportunity": True}),
                    "mask": legal_mask.unsqueeze(0),
                },
                transparent_scorer=transparent_scores,
                world_models=({"residual": 1.0},), world_predictor=predictor,
                calls=CallAccounting(FakeLedger(), FakeBoundary()),
                parent="train-0048", repeat=1, clock=AdvancingClock(),
                selector=lambda distribution: int(distribution.probs.argmax().item()),
            )
            return selection.select({"opportunity": True}, decision_step=7)

        decision = run_predictor(lambda actions: {
            "candidate_actions": actions.clone(),
            "utility_residuals": torch.tensor([1.0, 2.0, 3.0]),
        })
        self.assertEqual(seen[-1].tolist(), legal_actions.tolist())
        self.assertAlmostEqual(decision.trace["prior_logits"][0], 0.2, places=6)
        self.assertAlmostEqual(decision.trace["prior_logits"][1], 0.6, places=6)
        self.assertAlmostEqual(decision.trace["prior_logits"][NOOP_ACTION], 0.3, places=6)
        self.assertEqual(decision.cost["world_model_sample_evaluations"], 3)

        bad_responses = (
            ({"candidate_actions": torch.tensor([1, 0, NOOP_ACTION]),
              "utility_residuals": torch.tensor([1.0, 2.0, 3.0])},
             "WORLD_PREDICTION_CANDIDATE_IDENTITIES_MISMATCH"),
            ({"candidate_actions": torch.tensor([0, 0, NOOP_ACTION]),
              "utility_residuals": torch.tensor([1.0, 2.0, 3.0])},
             "WORLD_PREDICTION_CANDIDATE_IDENTITIES_MISMATCH"),
            ({"utility_residuals": torch.tensor([1.0, 2.0, 3.0])},
             "WORLD_PREDICTION_CANDIDATE_IDENTITIES_MISSING"),
            ({"candidate_actions": legal_actions,
              "utility_residuals": torch.tensor([1.0, 2.0])},
             "WORLD_RESIDUAL_SHAPE_MISMATCH"),
            ({"candidate_actions": torch.tensor([0, 1, 1.5]),
              "utility_residuals": torch.tensor([1.0, 2.0, 3.0])},
             "WORLD_PREDICTION_CANDIDATE_IDENTITIES_INVALID"),
        )
        for response, expected_error in bad_responses:
            with self.subTest(expected_error=expected_error, response=response):
                with self.assertRaisesRegex(PolicyProductionError, expected_error):
                    run_predictor(lambda _actions, response=response: response)


class PolicyTrainingTests(unittest.TestCase):
    def test_twelve_routes_use_frozen_configs_and_counted_updates(self):
        ledger, boundary = FakeLedger(), FakeBoundary()
        loaded_world_models, initialized_policies, updated = [], [], Counter()

        def load_world(variant, seed):
            loaded_world_models.append((variant, seed))
            return {"variant": variant, "seed": seed, "residual": 2.0}

        def make_policy(method, seed, config):
            initialized_policies.append((method, seed, config["learning_rate"]))
            return TinyPolicy()

        def make_optimizer(policy, method, seed, config):
            return {"method": method, "seed": seed, "lr": config["learning_rate"]}

        def ppo_update(policy, transitions, optimizer, config, device, *, event_group):
            self.assertIsInstance(policy, PriorConditionedPolicy)
            self.assertEqual(len(transitions), 64)
            updated[(event_group, optimizer["seed"])] += 1
            return {"update": updated[(event_group, optimizer["seed"])]}

        def run_route(context):
            route = context.route
            obs = {"opportunity": False}
            for step in range(route.steps):
                decision = context.decision_policy.select(obs, decision_step=step)
                self.assertEqual(decision.action, NOOP_ACTION)
                context.calls.call("environment", "training_env_step",
                                   {"environment_steps": 1}, lambda: None)
            minibatch = [{"candidate_features": torch.zeros(ACTION_COUNT, 18)} for _ in range(64)]
            device = torch.device("cpu")
            for _ in range(route.optimizer_updates):
                context.update_policy(minibatch, device=device)
            return {"method": route.method, "seed": route.seed}

        result = train_policy_routes(
            matrix=MATRIX, request=REQUEST, ledger=ledger, boundary=boundary,
            world_model_loader=load_world, policy_factory=make_policy,
            optimizer_factory=make_optimizer, route_runner=run_route,
            ppo_update_fn=ppo_update,
            checkpoint_writer=lambda path, *_args: path,
            input_builder=decision_inputs, transparent_scorer=transparent_scores,
            world_predictor=world_predictor,
        )
        self.assertEqual(result["route_count"], 12)
        self.assertEqual(len(loaded_world_models), 6)
        self.assertEqual(len(initialized_policies), 12)
        self.assertEqual(ledger.amounts["policy_optimizer_updates"], 1536)
        self.assertEqual(ledger.amounts["checkpoint_writes"], 12)
        self.assertEqual(ledger.amounts["model_initializations_or_loads"], 18)
        self.assertEqual(boundary.kinds["environment"], 24576)
        self.assertEqual(set(result["policies"]),
                         {(method, seed) for method in POLICY_METHODS for seed in POLICY_SEEDS})
        self.assertNotIn(9101, [route[1] for route in initialized_policies])
        for row in result["routes"]:
            if row["method"] in {"G1", "G2"}:
                self.assertEqual(row["world_model_seed"], {8301: 8201, 8302: 8202, 8303: 8203}[row["seed"]])
            else:
                self.assertIsNone(row["world_model_seed"])


class TaskConfirmationTests(unittest.TestCase):
    def _run(self, *, opportunity=True):
        ledger, boundary = FakeLedger(), FakeBoundary()
        episodes, worlds, policies = [], [], []
        checkpoints = {(method, seed): f"{method}-{seed}.pt"
                       for method in POLICY_METHODS for seed in POLICY_SEEDS}

        def load_policy(method, seed, path):
            policies.append((method, seed, path))
            return TinyPolicy()

        def load_world(variant, seed):
            worlds.append((variant, seed))
            return {"variant": variant, "seed": seed, "residual": float(seed % 7)}

        def run_episode(context):
            context.reset_environment(lambda: None)
            context.step_environment(lambda: None)
            actions = []
            if context.decision_policy is not None:
                decision = context.decision_policy.select(
                    {"opportunity": opportunity}, decision_step=0,
                )
                actions.append(decision.trace)
            episodes.append((context.method, context.seed, context.parent, context.repeat,
                             context.exogenous_key))
            return {"steps": 1, "opportunity": opportunity,
                    "utility": 1.25 if opportunity else None, "actions": actions,
                    "preference": list(context.task_preference),
                    "physical_rate": 0.5, "host_rate": 0.25,
                    "counts": {"completed": 3, "expired": 1}, "energy_used": 2.0,
                    "terminated": True, "truncated": False}

        result = evaluate_task_confirmation(
            matrix=MATRIX, request=REQUEST, ledger=ledger, boundary=boundary,
            policy_loader=load_policy, world_model_loader=load_world,
            episode_runner=run_episode, input_builder=decision_inputs,
            transparent_scorer=transparent_scores, world_predictor=world_predictor,
            checkpoints=checkpoints, clock=AdvancingClock(),
            selector=lambda distribution: int(distribution.probs.argmax().item()),
        )
        return result, ledger, boundary, episodes, worlds, policies

    def test_312_episodes_shared_hungarian_and_attributed_costs(self):
        result, ledger, boundary, episodes, worlds, policies = self._run()
        self.assertEqual(result["task_episodes"], 312)
        self.assertEqual(result["policy_episodes"], 288)
        self.assertEqual(result["shared_hungarian_episodes"], 24)
        self.assertEqual(len([episode for episode in episodes if episode[0] == "H"]), 24)
        self.assertEqual(len(worlds), 6)
        self.assertEqual(len(policies), 12)
        self.assertEqual(ledger.amounts["task_episodes"], 312)
        self.assertEqual(ledger.amounts["model_initializations_or_loads"], 18)
        # Task confirmation pairs one frozen world-model seed with each
        # policy seed; it does not run all three models per decision.
        self.assertEqual(ledger.amounts["world_batch_forwards"], 144)
        self.assertEqual(ledger.amounts["world_sample_evaluations"], 432)
        self.assertEqual(boundary.kinds["environment"], 624)
        self.assertEqual(result["cost_summary"]["by_method"]["G1"]["sample_count"], 72)
        self.assertEqual(result["cost_summary"]["by_method"]["G2"]["sample_count"], 72)
        self.assertEqual(result["metrics"]["valid_parent_count"], 8)
        self.assertAlmostEqual(result["metrics"]["parent_macro_utility"]["G2"], 1.25)
        unit_counts = [value["sample_count"] for key, value in result["cost_summary"]["by_method_parent_repeat"].items()
                       if key.startswith("G2|")]
        self.assertEqual(set(unit_counts), {3})
        for unit in MATRIX["splits"]["task_confirmation"]:
            parent = unit["parent"]
            for repeat in range(3):
                paired = [entry[4] for entry in episodes if entry[2] == parent and entry[3] == repeat]
                self.assertEqual(len(set(paired)), 1)

    def test_no_opportunity_is_not_zero_imputed_and_noop_decision_is_costed(self):
        result, ledger, _boundary, _episodes, _worlds, _policies = self._run(opportunity=False)
        self.assertEqual(result["cost_summary"]["by_method"]["G1"]["status"], "evaluated")
        self.assertEqual(result["cost_summary"]["by_method"]["G1"]["sample_count"], 72)
        self.assertEqual(ledger.amounts["actor_sample_evaluations"], 288)
        self.assertTrue(all(row["utility"] is None and not row["outcome_valid"]
                            for row in result["episodes"]))
        self.assertEqual(result["metrics"]["valid_parent_count"], 0)
        self.assertEqual(ledger.amounts["task_episodes"], 312)


if __name__ == "__main__":
    unittest.main()





