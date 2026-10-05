"""Native bottom-boundary hooks for the W1 production adapter.

The adapter owns contracts and stage ordering.  This module owns only the
reviewed native environment/model/optimizer/checkpoint calls that are allowed
to occur after an externally approved run.  Tests replace these methods at the
bottom boundary; they do not replace the adapter or the production stages.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class RuntimeHookError(RuntimeError):
    pass


class NativeRuntimeHooks:
    def __init__(self, adapter: Any):
        self.adapter = adapter
        self.root = Path(adapter.root)
        self.source_root = self._source_root()
        if str(self.source_root) not in __import__("sys").path:
            __import__("sys").path.insert(0, str(self.source_root))
        if str(self.source_root / "native") not in __import__("sys").path:
            __import__("sys").path.insert(0, str(self.source_root / "native"))

    def _source_root(self) -> Path:
        inputs = json.loads((self.root / "runtime-inputs.json").read_text(encoding="utf-8"))
        source = inputs["source_run"]
        key = "wsl_root" if __import__("sys").platform != "win32" else "windows_root"
        path = Path(source[key])
        if not path.is_absolute():
            raise RuntimeHookError("SOURCE_ROOT_NOT_ABSOLUTE")
        return path

    def _native(self) -> tuple[Any, ...]:
        from gppo_world.graph5 import graph5_from_m10_observation
        from gppo_world.joint_consequence_baseline import public_joint_action_score
        from gppo_world.joint_gppo import JointGraphPreferencePolicy, JointTrainConfig
        from gppo_world.m10_environment import M10Config, M10Environment, scenario_from_dict
        return (graph5_from_m10_observation, public_joint_action_score,
                JointGraphPreferencePolicy, JointTrainConfig,
                M10Config, M10Environment, scenario_from_dict)

    def _construct_environment(self, environment_type: Any, config: Any,
                               scenario: Any, exogenous_key: str) -> Any:
        operation = lambda: environment_type(
            config, scenario, exogenous_key=exogenous_key,
        )
        return self.adapter.boundary.environment(operation)

    def _scenario_rows(self) -> list[Mapping[str, Any]]:
        inputs = json.loads((self.root / "runtime-inputs.json").read_text(encoding="utf-8"))
        path = self.source_root / inputs["source_run"]["train_tape_file"]
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise RuntimeHookError("TRAIN_TAPE_EMPTY")
        return rows

    def _row(self, exogenous_key: str) -> Mapping[str, Any]:
        prefix = str(exogenous_key).rsplit("|repeat-", 1)[0]
        source_key = prefix + "|repeat-0"
        rows = [row for row in self._scenario_rows() if str(row.get("exogenous_key")) == source_key]
        if len(rows) != 1:
            raise RuntimeHookError(f"SCENARIO_IDENTITY_NOT_UNIQUE:{source_key}")
        return rows[0]

    def world_model_loader(self, variant: str, seed: int) -> Any:
        from production_world import _file_sha256, _load_checkpoint, _model_config
        from w1_graph_jepa import W1GraphJEPA
        training = self.adapter.world_training
        if training is None:
            raise RuntimeHookError("WORLD_MODEL_LOAD_BEFORE_TRAINING")
        route = next((row for row in training.routes
                      if row["variant"] == variant and int(row["seed"]) == int(seed)), None)
        if route is None:
            raise RuntimeHookError(f"WORLD_MODEL_ROUTE_MISSING:{variant}:{seed}")
        path = self.adapter.output / str(route["checkpoint_relative_path"])
        if _file_sha256(path) != str(route["checkpoint_sha256"]):
            raise RuntimeHookError(f"WORLD_MODEL_CHECKPOINT_DIGEST_MISMATCH:{variant}:{seed}")
        payload = _load_checkpoint(path)
        model = W1GraphJEPA(_model_config(self.adapter.matrix), event_count=5)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    def policy_factory(self, method: str, seed: int, _config: Mapping[str, Any]) -> Any:
        _, _, policy_type, _, m10_config, _, _ = self._native()
        import torch
        torch.manual_seed(int(seed))
        return policy_type(m10_config(), history=True)

    def optimizer_factory(self, policy: Any, _method: str, _seed: int, config: Mapping[str, Any]) -> Any:
        import torch
        return torch.optim.Adam(policy.parameters(), lr=float(config["learning_rate"]))

    def checkpoint_writer(self, path: str, policy: Any, optimizer: Any, route: Any, summary: Mapping[str, Any]) -> str:
        import torch
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # ``train_policy_routes`` passes the prior wrapper, whereas task
        # evaluation restores the base policy before rebuilding that wrapper.
        # Persist the base identity so save/load are inverse operations.
        policy_state = policy.base.state_dict() if hasattr(policy, "base") else policy.state_dict()
        torch.save({"schema": "w1-policy-checkpoint/1.0.0", "method": route.method,
                    "seed": route.seed, "state_dict": policy_state,
                    "optimizer_state_dict": optimizer.state_dict(), "summary": dict(summary)}, target)
        return str(target)

    def policy_loader(self, method: str, seed: int, path: str) -> Any:
        import torch
        _, _, policy_type, _, m10_config, _, _ = self._native()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "w1-policy-checkpoint/1.0.0" or payload.get("method") != method or int(payload.get("seed", -1)) != int(seed):
            raise RuntimeHookError(f"POLICY_CHECKPOINT_IDENTITY_MISMATCH:{method}:{seed}")
        policy = policy_type(m10_config(), history=True)
        policy.load_state_dict(payload["state_dict"])
        policy.eval()
        return policy

    def input_builder(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        import math
        import torch
        from public_history import history_vector
        flat = torch.as_tensor(observation["flat"], dtype=torch.float32).reshape(1, -1)
        relations = torch.as_tensor(observation["graph"]["relations"], dtype=torch.float32).reshape(24, 4)
        preference_value = observation.get("policy_preference")
        if not isinstance(preference_value, (list, tuple)) or len(preference_value) != 2:
            raise RuntimeHookError("POLICY_PREFERENCE_MISSING")
        preference = tuple(float(value) for value in preference_value)
        if not all(math.isfinite(value) for value in preference):
            raise RuntimeHookError("POLICY_PREFERENCE_NONFINITE")
        candidates = torch.zeros((25, 17), dtype=torch.float32)
        candidates[:24, :4] = relations
        return {
            "obs_tensor": flat,
            "world_history": torch.as_tensor(history_vector(observation), dtype=torch.float32).reshape(1, 128),
            "preference": torch.tensor([preference], dtype=torch.float32),
            "candidate_features": candidates,
            "mask": torch.as_tensor(observation["mask"], dtype=torch.bool).reshape(1, 25),
        }

    def transparent_scorer(self, observation: Mapping[str, Any], _inputs: Mapping[str, Any]) -> Any:
        import torch
        from production_policy import transparent_utility_components
        graph_builder, scorer, *_ = self._native()
        graph = graph_builder(observation)
        config = self._native()[4]()
        preference = torch.as_tensor(_inputs["preference"], dtype=torch.float32).reshape(-1)
        if preference.shape != (2,) or not bool(torch.isfinite(preference).all()):
            raise RuntimeHookError("TRANSPARENT_PREFERENCE_INVALID")
        scores = torch.zeros(25, dtype=torch.float32)
        for action, allowed in enumerate(graph.action_mask.tolist()):
            if allowed:
                components = transparent_utility_components(
                    scorer(graph, action), task_capacity=config.task_capacity,
                    initial_total_energy=float(config.uav_count * config.initial_energy),
                )
                scores[action] = 0.5 * float(preference[0]) * components[0] + float(preference[1]) * components[1]
        return scores

    def world_predictor(self, _method: str, model: Any, observation: Mapping[str, Any], actions: Any, _mask: Any) -> Mapping[str, Any]:
        import torch
        from w1_graph_jepa import expand_candidate_batch
        from public_history import history_vector
        graph_builder, *_ = self._native()
        graph = graph_builder(observation)
        action_tensor = torch.as_tensor(actions, dtype=torch.long)
        relation_rows = torch.as_tensor(graph.candidate_features, dtype=torch.float32).reshape(24, 4)
        relations = torch.cat((relation_rows, torch.zeros((1, 4))), dim=0)[action_tensor]
        nodes = {name: value.unsqueeze(0) for name, value in graph.nodes.items()}
        history = torch.as_tensor(history_vector(observation), dtype=torch.float32).reshape(1, 128)
        expanded_nodes, expanded_history = expand_candidate_batch(nodes, history, action_tensor, relations)
        with torch.no_grad():
            output = model.predict_candidates(expanded_nodes, expanded_history, action_tensor, relations)
        return {"candidate_actions": action_tensor, "outcomes": output["outcome"]}

    def ppo_update_fn(self, policy: Any, transitions: Any, optimizer: Any, config: Mapping[str, Any], device: Any, *, event_group: str) -> Mapping[str, Any]:
        _, _, _, config_type, *_ = self._native()
        from gppo_world.joint_training import ppo_preference_update
        train_config = config_type(
            seed=0, rollout_steps=int(config["rollout_steps"]), policy_lr=float(config["learning_rate"]),
            gamma=float(config["gamma"]), gae_lambda=float(config["gae_lambda"]),
            clip_epsilon=float(config["clip_epsilon"]), entropy_weight=float(config["entropy_weight"]),
            value_weight=float(config["value_weight"]), grad_clip=float(config["gradient_clip_norm"]),
        )
        # Keep the PriorConditionedPolicy wrapper in the replay evaluation.
        # Its augmented candidate tensor contains the behavior-time prior;
        # stripping it would recompute a different log-probability for G1/G2.
        return ppo_preference_update(policy, transitions, optimizer, train_config, device, event_group=event_group)

    def route_runner(self, context: Any) -> Mapping[str, Any]:
        """Train one GPPO route with the frozen world model, if any.

        This is deliberately a small production orchestration loop.  The old
        ``joint_training.run_group`` entry point jointly optimized a world
        model, so routing through it would violate the frozen-world-model
        contract even when its world-update budget was set to zero.
        """
        import numpy as np
        import torch
        from classical_baselines import PublicDecisionAdapter
        from gppo_world.joint_training import _mask_safe, _vector_reward
        _, _, _, _, m10_config, environment_type, scenario_from_dict = self._native()

        train_rows = self._scenario_rows()
        scenarios = []
        for row in self.adapter.matrix.get("splits", {}).get("train", ()):
            key = str(row.get("exogenous_key"))
            source = next((item for item in train_rows
                           if str(item.get("exogenous_key")) == key), None)
            if source is None:
                raise RuntimeHookError(f"TRAIN_SCENARIO_NOT_FOUND:{key}")
            scenarios.append((row, source))
        if len(scenarios) != 24:
            raise RuntimeHookError("TRAIN_SCENARIO_COUNT_MISMATCH")

        device = torch.device("cpu")
        transitions = []
        total_updates = 0
        rollout_count = context.route.steps // context.route.rollout_steps
        if rollout_count <= 0 or context.route.steps % context.route.rollout_steps:
            raise RuntimeHookError("POLICY_ROLLOUT_SCHEDULE_INVALID")
        updates_per_rollout = context.route.optimizer_updates // rollout_count
        if updates_per_rollout <= 0 or context.route.optimizer_updates % rollout_count:
            raise RuntimeHookError("POLICY_UPDATE_SCHEDULE_INVALID")
        scenario_index = 0
        episode_index = 0
        env = None
        observation = None
        public_adapter = None
        prepared_decision = None
        policy_hidden = None
        previous_counts = {"completed": 0, "expired": 0}
        previous_energy = float(m10_config().uav_count * m10_config().initial_energy)

        def start_episode(index: int):
            from public_history import CausalPublicHistory
            row, source = scenarios[index % len(scenarios)]
            scenario = scenario_from_dict(source["scenario"])
            runtime_env = self._construct_environment(
                environment_type, m10_config(), scenario, str(row["exogenous_key"]),
            )
            obs = context.calls.call(
                "environment", f"policy_reset:{context.route.method}:{context.route.seed}",
                {"resets_upper": 1}, runtime_env.reset,
            )
            history = CausalPublicHistory()
            history.append(obs)
            return runtime_env, obs, PublicDecisionAdapter(), history

        current_preference = context.schedule.training_preference(episode_index, context.route.seed)
        env, observation, public_adapter, causal_history = start_episode(scenario_index)
        while context.decision_policy.decision_count < context.route.steps:
            if observation is None or public_adapter is None:
                raise RuntimeHookError("POLICY_OBSERVATION_MISSING")
            if prepared_decision is None:
                public_observation, effective_mask = public_adapter.prepare(observation)
                public_observation = causal_history.attach_current(public_observation)
                if "public_history" not in public_observation or "public_history_schema" not in public_observation:
                    raise RuntimeHookError("CAUSAL_HISTORY_ATTACH_FAILED")
            else:
                public_observation, effective_mask = prepared_decision
                prepared_decision = None
            decision_observation = dict(public_observation)
            decision_observation["mask"] = effective_mask
            decision_observation["policy_preference"] = current_preference
            decision = context.decision_policy.select(
                decision_observation, decision_step=context.decision_policy.decision_count,
                hidden=policy_hidden,
            )
            values = decision.values
            if values is None:
                raise RuntimeHookError("POLICY_VALUE_MISSING")
            mask = _mask_safe(np.asarray(effective_mask, dtype=np.bool_))
            if not bool(mask[int(decision.action)]):
                raise RuntimeHookError("POLICY_SELECTED_ILLEGAL_ACTION")
            public_adapter.commit(int(decision.action))
            old_obs = decision_observation
            next_obs, raw_reward, done, info = context.calls.call(
                "environment", f"policy_step:{context.route.method}:{context.route.seed}",
                {"environment_steps": 1}, env.step, int(decision.action),
            )
            # The returned state is appended once before it can be used for
            # bootstrap or the next action; rollout replay stores that exact input.
            causal_history.append(next_obs)
            vector_reward, consequence, previous_counts, previous_energy = _vector_reward(
                info, previous_counts, previous_energy, m10_config(),
            )
            transition = {
                "obs": np.asarray(old_obs["flat"], dtype=np.float32).copy(),
                "next_obs": np.asarray(next_obs["flat"], dtype=np.float32).copy(),
                "observation_dict": old_obs, "next_observation_dict": next_obs,
                "mask": mask.copy(), "action": int(decision.action),
                "old_log_prob": float(decision.log_probability),
                "old_values": values.numpy().astype(np.float32),
                "next_values": np.zeros(2, dtype=np.float32),
                "vector_reward": vector_reward,
                "task_consequence": consequence,
                "raw_environment_reward": float(raw_reward),
                "event_label": np.zeros(5, dtype=np.float32),
                "event_mask": np.zeros(5, dtype=np.bool_),
                "state_target_valid": bool(np.isfinite(next_obs["flat"]).all()),
                "vector_reward_valid": bool(np.isfinite(vector_reward).all()),
                "task_consequence_valid": bool(np.isfinite(consequence).all()),
                "censor_reason": None,
                "policy_hidden_before": np.zeros(128, dtype=np.float32) if policy_hidden is None
                    else policy_hidden.detach().cpu().numpy().reshape(-1).copy(),
                "policy_hidden_after": np.zeros(128, dtype=np.float32) if decision.next_hidden is None
                    else decision.next_hidden.detach().cpu().numpy().reshape(-1).copy(),
                "world_hidden_before": np.zeros(128, dtype=np.float32),
                "candidate_features": decision.candidate_features[0].detach().cpu().numpy().copy(),
                "preference": np.asarray(current_preference, dtype=np.float32),
                "terminated": bool(info.get("terminated", False)),
                "truncated": bool(info.get("truncated", False)),
                "environment_truncated": bool(info.get("truncated", False)),
                "policy_version": total_updates, "world_version": 0,
                "scenario_id": str(getattr(env.scenario, "tape_id", scenarios[scenario_index][0]["parent"])),
                "time": float(old_obs.get("time", 0.0)), "action_legal": True,
            }
            if transition["truncated"] and not transition["terminated"]:
                next_public, next_effective_mask = public_adapter.prepare(next_obs)
                next_decision_observation = dict(next_public)
                next_decision_observation["mask"] = next_effective_mask
                next_decision_observation["policy_preference"] = current_preference
                next_decision_observation = causal_history.attach_current(next_decision_observation)
                transition["next_values"] = context.decision_policy.bootstrap_value(
                    next_decision_observation, hidden=decision.next_hidden,
                ).numpy().astype(np.float32)
                # The pending prepare belongs to the terminal observation and
                # must not be committed because no action is submitted.
                public_adapter.reset()
            if transitions and not transitions[-1]["terminated"] and not transitions[-1]["truncated"]:
                transitions[-1]["next_values"] = transition["old_values"].copy()
            transitions.append(transition)
            observation = next_obs
            policy_hidden = decision.next_hidden
            if done:
                episode_index += 1
                scenario_index = episode_index % len(scenarios)
                current_preference = context.schedule.training_preference(episode_index, context.route.seed)
                env, observation, public_adapter, causal_history = start_episode(scenario_index)
                prepared_decision = None
                policy_hidden = None
                previous_counts = {"completed": 0, "expired": 0}
                cfg = m10_config()
                previous_energy = float(cfg.uav_count * cfg.initial_energy)
            if len(transitions) == context.route.rollout_steps:
                last = transitions[-1]
                if not (last["terminated"] or last["truncated"]):
                    next_public, next_effective_mask = public_adapter.prepare(observation)
                    next_decision_observation = dict(next_public)
                    next_decision_observation["mask"] = next_effective_mask
                    next_decision_observation["policy_preference"] = current_preference
                    next_decision_observation = causal_history.attach_current(next_decision_observation)
                    last["next_values"] = context.decision_policy.bootstrap_value(
                        next_decision_observation, hidden=policy_hidden,
                    ).numpy().astype(np.float32)
                    last["truncated"] = True
                    last["rollout_boundary"] = True
                    # The same prepared public observation becomes the first
                    # decision of the next rollout after the PPO update.
                    # Preserve the causal history attached to the bootstrap
                    # observation when it becomes the next rollout prefix.
                    prepared_decision = (next_decision_observation, next_effective_mask)
                for _ in range(updates_per_rollout):
                    context.update_policy(transitions, device=device)
                    total_updates += 1
                transitions = []
        if transitions:
            raise RuntimeHookError("POLICY_ROLLOUT_REMAINDER")
        return {
            "environment_steps": context.route.steps,
            "policy_optimizer_updates": total_updates,
            "world_model_updates": 0,
            "world_model_frozen": context.route.world_model_variant is not None,
            "route": f"{context.route.method}:{context.route.seed}",
        }

    def episode_runner(self, context: Any) -> Mapping[str, Any]:
        import numpy as np
        import torch
        from public_history import CausalPublicHistory
        from classical_baselines import ClassicalSelector, PublicDecisionAdapter
        from gppo_world.joint_training import _vector_reward
        _, _, _, _, m10_config, environment_type, scenario_from_dict = self._native()
        row = self._row(context.exogenous_key)
        config = m10_config()
        env = self._construct_environment(
            environment_type, config, scenario_from_dict(row["scenario"]), context.exogenous_key,
        )
        observation = context.reset_environment(env.reset)
        public_history = CausalPublicHistory()
        public_history.append(observation)
        opportunity = False
        actions, utility, steps = [], 0.0, 0
        public_adapter = PublicDecisionAdapter()
        hungarian = ClassicalSelector("hungarian", preference=context.task_preference) if context.decision_policy is None else None
        policy_hidden = None
        previous_counts = {"completed": 0, "expired": 0}
        initial_energy = float(config.uav_count * config.initial_energy)
        previous_energy = initial_energy
        final_info = None
        done = False
        while steps < context.max_steps and not done:
            public_observation, effective_mask = public_adapter.prepare(observation)
            public_observation = public_history.attach_current(public_observation)
            opportunity = opportunity or bool(torch.as_tensor(effective_mask, dtype=torch.bool)[:24].any())
            decision_observation = dict(public_observation)
            decision_observation["mask"] = effective_mask
            decision_observation["policy_preference"] = context.task_preference
            if hungarian is not None:
                action, diagnostic = hungarian.choose(decision_observation, public_adapter.memory)
                trace = {"method": "H", "selected_action": action, "diagnostic": diagnostic}
            else:
                decision = context.decision_policy.select(
                    decision_observation, decision_step=steps, hidden=policy_hidden,
                )
                action, trace = decision.action, decision.trace
                policy_hidden = decision.next_hidden
            public_adapter.commit(int(action))
            observation, scalar_reward, done, info = context.step_environment(env.step, int(action))
            public_history.append(observation)
            if bool(done) != (bool(info.get("terminated")) or bool(info.get("truncated"))):
                raise RuntimeHookError("TASK_TERMINATION_MISMATCH")
            if bool(info.get("terminated")) and bool(info.get("truncated")):
                raise RuntimeHookError("TASK_TERMINATION_AMBIGUOUS")
            vector_reward, _consequence, previous_counts, previous_energy = _vector_reward(
                info, previous_counts, previous_energy, config,
            )
            if not bool(np.isfinite(vector_reward).all()):
                raise RuntimeHookError("TASK_VECTOR_REWARD_NONFINITE")
            utility += context.utility_discount ** steps * (
                context.task_component_scale * context.task_preference[0] * float(vector_reward[0])
                + context.task_preference[1] * float(vector_reward[1])
            )
            actions.append({**dict(trace), "preference": list(context.task_preference),
                            "vector_reward": np.asarray(vector_reward, dtype=np.float32).tolist(),
                            "scalar_environment_reward": float(scalar_reward)})
            final_info = info
            steps += 1
        if not done or final_info is None:
            raise RuntimeHookError("TASK_REACHED_NON_NATIVE_TRUNCATION")
        records = final_info.get("completion_records", {})
        tasks = tuple(row["scenario"].get("tasks", ()))
        if not tasks:
            raise RuntimeHookError("TASK_SCENARIO_HAS_NO_TASKS")
        physical = 0
        host = 0
        for task in tasks:
            record = records.get(task["task_id"], {})
            physical_time = record.get("physical_arrival_time")
            host_time = record.get("host_confirmation_time")
            deadline = float(task["deadline"])
            physical += int(physical_time is not None and float(physical_time) <= deadline)
            host += int(host_time is not None and float(host_time) <= deadline
                        and float(host_time) <= float(final_info["time"]))
        return {
            "steps": steps, "opportunity": opportunity,
            "utility": utility if opportunity else None, "actions": actions,
            "preference": list(context.task_preference),
            "physical_rate": physical / len(tasks), "host_rate": host / len(tasks),
            "counts": dict(previous_counts), "energy_used": initial_energy - previous_energy,
            "terminated": bool(final_info.get("terminated")),
            "truncated": bool(final_info.get("truncated")),
        }


__all__ = ["NativeRuntimeHooks", "RuntimeHookError"]
