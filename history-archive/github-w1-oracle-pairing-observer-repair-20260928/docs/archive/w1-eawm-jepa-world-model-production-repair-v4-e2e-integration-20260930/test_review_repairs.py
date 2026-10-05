from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parent
SOURCE = Path(r"E:\Z博士\runs\w1-light-repaired-fair-rerun-v2-nativefs-once")
for item in (str(SOURCE), str(SOURCE / "native")):
    if item not in sys.path:
        sys.path.insert(0, item)

from gppo_world.joint_gppo import JointTrainConfig
from gppo_world.joint_training import ppo_preference_update
from production_data import _continuation_ids, _event_target, _outcome_target
from production_policy import PriorConditionedPolicy
from production_world import _trace_decision
from public_history import CausalPublicHistory, history_vector


class _TinyGPPO(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_parameter = nn.Parameter(torch.linspace(-0.2, 0.2, 25))
        self.value_parameter = nn.Parameter(torch.tensor([0.1, -0.1]))
        self.encode_calls = 0
        self.evaluate_calls = 0

    def encode(self, obs, hidden=None):
        self.encode_calls += 1
        batch = obs.shape[0]
        features = self.logit_parameter.sum().expand(batch, 128) * 0.0
        pair = features.new_zeros((batch, 4, 6, 2))
        next_hidden = features.new_zeros((1, batch, 128))
        return features, pair, next_hidden

    def evaluate_encoded(self, features, pair_messages, preference, candidate_features, mask):
        self.evaluate_calls += 1
        batch = features.shape[0]
        logits = self.logit_parameter.expand(batch, -1)
        legal = mask.bool()
        distribution = Categorical(logits=logits.masked_fill(~legal, -torch.inf))
        values = self.value_parameter.expand(batch, -1)
        return {"logits": logits, "distribution": distribution,
                "probabilities": distribution.probs, "state_values": values,
                "critic_values": values}


class _EventEnv:
    def __init__(self, value, measured_at, received_at):
        record = SimpleNamespace(entity="u0", field="x", value=value,
                                 measured_at=measured_at, received_at=received_at,
                                 sequence=1, message_id="m1")
        self.view = SimpleNamespace(
            _uavs=SimpleNamespace(_latest={("u0", "x"): record}),
            _task_values=SimpleNamespace(_latest={}),
        )
        self._active_commands = {"private-command": object()}


def _public_obs(time_value, continuation=()):
    return {
        "time": float(time_value), "mask": [True] + [False] * 23 + [True],
        "public_entity_ids": {"uavs": ["u0", "u1", "u2", "u3"],
                              "tasks": ["t0", "t1", "t2", "t3", "t4", "t5"]},
        "continuation_actions": list(continuation),
    }


class ReviewRepairTests(unittest.TestCase):
    def test_history_distinguishes_same_current_state_and_resets_per_episode(self):
        current = {"flat": [3.0] * 128}
        left, right = CausalPublicHistory(), CausalPublicHistory()
        left.append({"flat": [0.0] * 128})
        left.append(current)
        right.append({"flat": [6.0] * 128})
        right.append(current)
        self.assertNotEqual(history_vector(left.attach_current(current)),
                            history_vector(right.attach_current(current)))
        left.reset()
        left.append(current)
        self.assertEqual(left.observations, 1)
        self.assertEqual(history_vector(left.attach_current(current)), [3.0] * 128)

    def test_public_continuation_ids_do_not_read_private_active_commands(self):
        observation = _public_obs(1.0, continuation=(7, 0))
        self.assertEqual(_continuation_ids(_EventEnv(1.0, 0.0, 1.0), observation),
                         ("u0::t0", "u1::t1"))

    def test_event_targets_use_public_receipt_and_keep_missing_execution_unknown(self):
        current_obs, next_obs = _public_obs(0.0), _public_obs(1.0)
        current_env = _EventEnv(2.0, 0.0, 0.0)
        next_env = _EventEnv(2.0, 0.0, 1.0)
        values, valid = _event_target(current_obs, next_obs, current_env, next_env,
                                      {"feedback": "unknown"}, 0, "fixed-hungarian-v1")
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[1], 1.0)
        self.assertTrue(valid[0] and valid[1])
        self.assertFalse(valid[3] or valid[4])

    def test_first_seen_field_is_not_mislabeled_as_no_new_measurement(self):
        current_obs, next_obs = _public_obs(0.0), _public_obs(1.0)
        current_env = _EventEnv(2.0, 0.0, 0.0)
        next_env = _EventEnv(2.0, 0.0, 0.0)
        next_env.view._uavs._latest[("u1", "new_field")] = SimpleNamespace(
            entity="u1", field="new_field", value=7.0,
            measured_at=0.0, received_at=0.0, sequence=2, message_id="m2",
        )
        values, valid = _event_target(current_obs, next_obs, current_env, next_env,
                                      {"feedback": "unknown"}, 0, "fixed-hungarian-v1")
        self.assertFalse(valid[1])
        self.assertEqual(values[1], 0.0)

        next_env.view._uavs._latest[("u1", "new_field")].received_at = 0.5
        next_env.view._uavs._latest[("u1", "new_field")].measured_at = 0.5
        values, valid = _event_target(current_obs, next_obs, current_env, next_env,
                                      {"feedback": "unknown"}, 0, "fixed-hungarian-v1")
        self.assertTrue(valid[1])
        self.assertEqual(values[1], 1.0)

    def test_unknown_acceptance_is_masked_and_true_utility_is_not_transparent_score(self):
        outcome, valid = _outcome_target(
            0, {"flat": [0.0, 1.0], "continuation_actions": ()},
            {"flat": [1.0, 1.0], "continuation_actions": ()},
            {"feedback": "unknown"}, [0.2, -0.1], (0.1, -0.05),
        )
        self.assertIsNone(outcome[0])
        self.assertFalse(valid[0])

        candidates = [{
            "candidate_id": "a", "action_id": 0, "transparent_score": 2.0,
            "true_utility": 0.25, "outcome_valid": [True] * 6,
            "outcome_target": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "predictions": {"G1": {"ensemble": {"outcome": [0.0] * 6}}},
        }, {
            "candidate_id": "b", "action_id": 1, "transparent_score": 0.0,
            "true_utility": 0.5, "outcome_valid": [True] * 6,
            "outcome_target": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "predictions": {"G1": {"ensemble": {"outcome": [0.0] * 6}}},
        }]
        window = {"candidate_rows": candidates}
        first = _trace_decision(window, "G1", (8201, 8202, 8203))
        for row in candidates:
            row["transparent_score"] += 17.0
        second = _trace_decision(window, "G1", (8201, 8202, 8203))
        self.assertEqual(first["oracle_action"], second["oracle_action"])
        self.assertEqual(first["regret"], second["regret"])

    def test_native_gppo_preco_loss_and_gradient_are_finite_with_partial_masks(self):
        base = _TinyGPPO()
        policy = PriorConditionedPolicy(base)
        transitions = []
        masks = ([True] + [False] * 23 + [True], [False] * 24 + [True])
        actions = (0, 24)
        for index, (mask_values, action) in enumerate(zip(masks, actions)):
            mask = torch.tensor([mask_values], dtype=torch.bool)
            candidates = torch.zeros((1, 25, 18), dtype=torch.float32)
            candidates[0, 0, 17] = 0.2
            candidates[0, 24, 17] = -0.1
            with torch.no_grad():
                evaluated = policy.evaluate_encoded(torch.zeros((1, 128)), torch.zeros((1, 4, 6, 2)),
                                                    torch.tensor([[0.8, 0.2]]), candidates, mask)
            self.assertTrue(torch.isfinite(evaluated["logits"]).all())
            self.assertEqual(float(evaluated["probabilities"][0, 1 if action == 0 else 0]), 0.0)
            transitions.append({
                "obs": [float(index)] * 8,
                "policy_hidden_before": [0.0] * 128,
                "mask": list(mask_values), "preference": [0.8, 0.2],
                "candidate_features": candidates[0].numpy().copy(), "action": action,
                "old_log_prob": float(evaluated["distribution"].log_prob(torch.tensor([action]))[0]),
                "old_values": [0.1, -0.1], "next_values": [0.2, 0.1],
                "vector_reward": [0.4, -0.02], "terminated": index == 1, "truncated": False,
            })
        optimizer = torch.optim.SGD(policy.parameters(), lr=1e-4)
        result = ppo_preference_update(policy, transitions, optimizer,
                                       JointTrainConfig(rollout_steps=2), torch.device("cpu"),
                                       event_group="G2")
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in result.values()))
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                            for parameter in policy.parameters()))
        self.assertEqual(base.encode_calls, 2)
        self.assertEqual(base.evaluate_calls, 3)


if __name__ == "__main__":
    unittest.main()
