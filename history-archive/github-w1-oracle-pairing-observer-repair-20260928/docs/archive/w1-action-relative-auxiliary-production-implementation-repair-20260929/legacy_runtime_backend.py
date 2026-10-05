"""Authorized repaired-W1 runtime bridge for the frozen learning pipeline.

Importing this module does not construct an environment or a model. Runtime
dependencies are loaded only after the external launch token and zero-step
identity gate have passed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pickle
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_ledger import BudgetLedger
from experiment_matrix import derive_budget
from learning_schema import CONTINUATION_ID, history_features, materialize_features
from transparent_baselines import current_public_scores, qualifies, transparent_history_scores


HORIZON = 18
PREFERENCE = (0.8, 0.2)
INITIAL_FLEET_ENERGY = 36.0


class TechnicalStop(RuntimeError):
    pass


@dataclass(frozen=True)
class DecisionInputSnapshot:
    decision_time: float
    flat: tuple[float, ...]
    history_features: tuple[float, ...]
    legal_actions: tuple[int, ...]
    current_scores: tuple[tuple[int, float], ...]
    transparent_scores: tuple[tuple[int, float], ...]
    public_input_json: str
    sha256: str

    def current_score(self, action: int) -> float:
        return dict(self.current_scores)[action]

    def transparent_score(self, action: int) -> float:
        return dict(self.transparent_scores)[action]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()


def _snapshot_digest(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def freeze_decision_input(
    adapter,
    public_obs: Mapping[str, Any],
    candidates: Sequence[int],
    current_scores: Mapping[int, float],
    transparent_scores: Mapping[int, float],
) -> DecisionInputSnapshot:
    """Freeze every public feature dependency before any candidate branch runs."""

    if not adapter.memory.history:
        raise TechnicalStop("decision input has no public history")
    if _canonical(adapter.memory.history[-1]) != _canonical(public_obs):
        raise TechnicalStop("current public observation is not the last memory entry")
    flat = public_obs.get("flat")
    if flat is None:
        raise TechnicalStop("complete 770-field public flat input is missing")
    legal_actions = tuple(int(value) for value in candidates)
    if not legal_actions or len(set(legal_actions)) != len(legal_actions):
        raise TechnicalStop("decision input candidate set is empty or duplicated")
    if set(current_scores) != set(legal_actions) or set(transparent_scores) != set(legal_actions):
        raise TechnicalStop("decision input score set differs from legal candidates")
    decision_time = float(public_obs["time"])
    public_history = copy.deepcopy(adapter.memory.history[:-1])
    features = tuple(history_features(public_history, decision_time))
    audit_payload = {
        "decision_time": decision_time,
        "current_public": copy.deepcopy(public_obs),
        "public_history": public_history,
        "legal_actions": list(legal_actions),
    }
    public_input_json = _canonical(audit_payload)
    return DecisionInputSnapshot(
        decision_time=decision_time,
        flat=tuple(float(value) for value in flat),
        history_features=features,
        legal_actions=legal_actions,
        current_scores=tuple((action, float(current_scores[action])) for action in legal_actions),
        transparent_scores=tuple(
            (action, float(transparent_scores[action])) for action in legal_actions
        ),
        public_input_json=public_input_json,
        sha256=hashlib.sha256(public_input_json.encode("utf-8")).hexdigest(),
    )


def _verify_task_shared_fates(parent: str, repeat: int, summaries: Sequence[Mapping]) -> dict:
    """Verify common primitive fate while allowing three identical task arms."""

    by_arm: dict[str, dict[str, list[Any]]] = {}
    base_parameters: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for summary in summaries:
        arm = str(summary.get("identity", {}).get("arm", ""))
        calls = summary.get("primitive_calls")
        if not arm or arm in by_arm or not isinstance(calls, list):
            raise TechnicalStop("task pairing evidence is missing or duplicated")
        grouped: dict[str, list[Any]] = defaultdict(list)
        previous = -1
        for call in calls:
            arguments = call.get("arguments")
            ordinal = int(call.get("ordinal", -1))
            if (
                call.get("branch") != arm
                or call.get("status") != "returned"
                or not isinstance(arguments, dict)
                or not isinstance(arguments.get("identity"), str)
                or "result" not in call
                or ordinal <= previous
            ):
                raise TechnicalStop("invalid task primitive call evidence")
            previous = ordinal
            exact = _canonical({"method": call["method"], "arguments": arguments})
            grouped[exact].append(call["result"])
            base = _canonical({"method": call["method"], "identity": arguments["identity"]})
            parameters = _canonical({key: value for key, value in arguments.items() if key != "identity"})
            base_parameters[base][arm].add(parameters)
        by_arm[arm] = dict(grouped)
    all_keys: dict[str, list[str]] = defaultdict(list)
    for arm, grouped in by_arm.items():
        for key in grouped:
            all_keys[key].append(arm)
    common = {key: arms for key, arms in all_keys.items() if len(arms) >= 2}
    if not common:
        raise TechnicalStop("task pairing has no common primitive calls")
    for key, arms in common.items():
        sequences = [by_arm[arm][key] for arm in arms]
        if len({len(values) for values in sequences}) != 1:
            raise TechnicalStop("task pairing primitive multiplicity differs")
        if len({_canonical(values) for values in sequences}) != 1:
            raise TechnicalStop("task pairing common random fate differs")
    for by_branch in base_parameters.values():
        if len(by_branch) >= 2 and len({item for values in by_branch.values() for item in values}) > 1:
            raise TechnicalStop("task pairing parameters differ for a common identity")
    action_specific = sum(1 for arms in all_keys.values() if len(arms) == 1)
    return {
        "schema": "w1-task-comparison-primitive-pairing/1.0.0",
        "parent": parent,
        "repeat": repeat,
        "arms": sorted(by_arm),
        "common_call_keys": len(common),
        "action_specific_call_keys": action_specific,
        "common_result_mismatches": 0,
        "identical_arm_traffic_allowed": True,
    }


class AuthorizedRuntimeBackend:
    def __init__(self, root: Path, output: Path, request: Mapping, matrix: Mapping):
        self.root = root
        self.output = output
        self.request = request
        self.matrix = matrix
        self.ledger: BudgetLedger | None = None
        self.runtime: dict[str, Any] = {}
        self.tape_by_parent: dict[str, dict] = {}
        self.decision_costs: list[dict[str, float]] = []
        self.bottom_environment_factory = None

    def select_stage(self, stage: str) -> None:
        if self.ledger is None:
            raise TechnicalStop("ledger is unavailable before the zero-step gate")
        self.ledger.select(stage)
        (self.output / "activity.json").write_text(
            json.dumps({"stage": stage}, sort_keys=True) + "\n", encoding="utf-8"
        )

    def account_call(self, name: str, amounts: dict[str, int], function, *args, **kwargs):
        if self.ledger is None:
            raise TechnicalStop("unledgered dynamic call")
        return self.ledger.call(name, amounts, function, *args, **kwargs)

    def _load_runtime(self) -> None:
        if self.runtime:
            return
        inputs = json.loads((self.root / "runtime-inputs.json").read_text(encoding="utf-8"))
        platform = "linux" if os.name == "posix" else "windows"
        paths = inputs["platforms"][platform]
        for package in inputs["packages"]:
            package_root = Path(paths[package["id"]])
            manifest_path = package_root / "execution-manifest.json"
            if _sha256(manifest_path) != package["execution_manifest_sha256"]:
                raise TechnicalStop(f"runtime package manifest mismatch: {package['id']}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for relative, expected in manifest["files"].items():
                candidate = package_root / relative
                if not candidate.is_file() or _sha256(candidate) != expected:
                    raise TechnicalStop(f"runtime package file mismatch: {package['id']}/{relative}")
        native_package = Path(paths["repaired_runtime"])
        oracle_root = Path(paths["oracle_runner"])
        native_root = native_package / "native"
        for path in (str(native_root), str(oracle_root)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from gppo_world.m10_environment import M10Config, M10Environment, scenario_from_dict
        from classical_baselines import ClassicalSelector, PublicDecisionAdapter
        from communication_observer import RecordingCommunication, require_observer, verify_primitive_pairing
        from labeling import label_episode

        config = M10Config(**json.loads((native_package / "environment.json").read_text(encoding="utf-8")))
        tape = json.loads((native_package / "tapes-train.json").read_text(encoding="utf-8"))
        self.tape_by_parent = {row["parent"]: row for row in tape}
        self.runtime = {
            "config": config,
            "environment": M10Environment,
            "scenario_from_dict": scenario_from_dict,
            "selector": ClassicalSelector,
            "adapter": PublicDecisionAdapter,
            "recording_communication": RecordingCommunication,
            "require_observer": require_observer,
            "verify_pairing": verify_primitive_pairing,
            "label_episode": label_episode,
        }

    def zero_step_gate(self) -> dict[str, Any]:
        if self.ledger is not None:
            raise TechnicalStop("zero-step gate cannot be repeated")
        self.ledger = BudgetLedger(self.output / "budget.sqlite3", self.request)
        self.ledger.select("staging_and_zero_step_gate")
        expected = derive_budget(dict(self.matrix))
        if expected != self.request["derived_dynamic_budget"]:
            raise TechnicalStop("resource request is not derived from the frozen matrix")
        group_parents = {
            role: {row["parent"] for row in rows}
            for role, rows in self.matrix["groups"].items()
        }
        roles = list(group_parents)
        for index, left in enumerate(roles):
            for right in roles[index + 1:]:
                if group_parents[left] & group_parents[right]:
                    raise TechnicalStop("parent split overlap")
        self._load_runtime()
        for role, rows in self.matrix["groups"].items():
            for identity in rows:
                source = self.tape_by_parent.get(identity["parent"])
                if source is None or source["scenario_sha256"] != identity["scenario_sha256"]:
                    raise TechnicalStop(f"scenario identity mismatch for {role}/{identity['parent']}")
        return {
            "pass": True,
            "environment_constructed": False,
            "model_constructed": False,
            "parent_groups_disjoint": True,
            "budget_rederived": True,
        }

    def _new_environment(self, parent: str, key: str):
        if self.bottom_environment_factory is not None:
            return self.bottom_environment_factory(parent, key)
        row = self.tape_by_parent[parent]
        env = self.runtime["environment"](
            self.runtime["config"], self.runtime["scenario_from_dict"](row["scenario"]),
            exogenous_key=key,
        )
        env.communication = self.runtime["recording_communication"](env.communication)
        return env

    def _step(self, env, action: int):
        return self.account_call(
            "environment.step", {"environment_steps": 1}, env.step, int(action)
        )

    def _run_label_branch(
        self,
        capture,
        *,
        action: int,
        parent: str,
        repeat: int,
        split_role: str,
        decision_id: str,
    ) -> tuple[dict, dict]:
        self.account_call("branch.register", {"branches": 1}, lambda: None)
        expected = _snapshot_digest(capture)
        copied = self.account_call(
            "branch.deepcopy", {"branch_snapshot_copies": 1}, copy.deepcopy, capture
        )
        env, adapter, decision_input, counts, energy, decision_step, key = copied
        if not isinstance(decision_input, DecisionInputSnapshot):
            raise TechnicalStop("branch lacks a frozen decision input")
        branch_input_digest = decision_input.sha256
        if _snapshot_digest(capture) != expected:
            raise TechnicalStop("source snapshot changed during branch copy")
        observer = self.runtime["require_observer"](env.communication)
        marker = observer.start_branch(f"action-{action:02d}")
        self.account_call(
            "branch.force_first", {"forced_first_actions": 1}, adapter.commit, int(action)
        )
        following, reward, done, info = self._step(env, action)
        rows = [{"step": 0, "action": action, "reward": reward, "done": done, "info": info}]
        current = following
        for local_step, actual_step in enumerate(range(decision_step + 1, HORIZON), 1):
            if done:
                break
            selector = self.runtime["selector"]("hungarian", PREFERENCE)
            decision = self.account_call(
                "hungarian.continuation",
                {"public_rule_decisions": 1},
                adapter.decide,
                current,
                selector,
            )
            current, reward, done, info = self._step(env, int(decision["action"]))
            rows.append({
                "step": local_step,
                "action": int(decision["action"]),
                "reward": reward,
                "done": done,
                "info": info,
            })
        summary = self.runtime["label_episode"](
            rows,
            initial_counts=counts,
            initial_energy=energy,
            scope=f"t{decision_step}_to_native_terminal",
            identity={
                "parent": parent,
                "repeat": repeat,
                "arm": f"action-{action:02d}",
                "action": action,
                "external_key": key,
                "communication_profile": "W1-light",
                "continuation": "hungarian",
                "decision_id": decision_id,
            },
        )
        summary["primitive_calls"] = observer.records_since(marker)
        summary["decision_input_sha256"] = branch_input_digest
        record = {
            "sample_id": f"{decision_id}:action-{action:02d}",
            "parent": parent,
            "repeat": repeat,
            "decision_id": decision_id,
            "split_role": split_role,
            "action": action,
            "legal_actions": list(decision_input.legal_actions),
            "flat": list(decision_input.flat),
            "history": list(decision_input.history_features),
            "remaining_utility": float(summary["discounted_utility"]),
            "current_public_score": decision_input.current_score(action),
            "transparent_history_score": decision_input.transparent_score(action),
            "continuation_id": CONTINUATION_ID,
            "is_true_branch": True,
            "training_eligible": True,
        }
        if _snapshot_digest(capture) != expected:
            raise TechnicalStop("branch contaminated its source snapshot")
        if decision_input.sha256 != branch_input_digest:
            raise TechnicalStop("branch mutated its frozen decision input")
        return record, summary

    def _collect_unit(self, split_role: str, parent: str, repeat: int) -> tuple[list[dict], dict]:
        key = f"w1-action-outcome-learning-v1|{split_role}|{parent}|repeat-{repeat}"
        env = self._new_environment(parent, key)
        observation = self.account_call("environment.reset", {"resets_upper": 1}, env.reset)
        adapter = self.runtime["adapter"]()
        counts = {"completed": 0, "expired": 0}
        energy = INITIAL_FLEET_ENERGY
        for decision_step in range(HORIZON):
            public_obs, _mask = adapter.prepare(observation)
            self.account_call("candidate.scan", {"candidate_scans": 1}, lambda: None)
            candidates = tuple(int(value) for value in adapter.memory.candidates(public_obs))
            qualified, transparent_scores = qualifies(
                public_obs, adapter.memory, minimum_step=4, decision_step=decision_step
            )
            if qualified:
                current_scores = current_public_scores(public_obs, candidates)
                decision_id = f"{split_role}:{parent}:r{repeat}:s{decision_step}"
                decision_input = freeze_decision_input(
                    adapter,
                    public_obs,
                    candidates,
                    current_scores,
                    transparent_scores,
                )
                capture = self.account_call(
                    "snapshot.capture",
                    {"snapshot_captures": 1},
                    lambda: (env, adapter, decision_input, counts, energy, decision_step, key),
                )
                records: list[dict] = []
                summaries: list[dict] = []
                for action in candidates:
                    record, summary = self._run_label_branch(
                        capture,
                        action=action,
                        parent=parent,
                        repeat=repeat,
                        split_role=split_role,
                        decision_id=decision_id,
                    )
                    records.append(record)
                    summaries.append(summary)
                pairing = self.runtime["verify_pairing"](parent, repeat, summaries)
                pairing_pass = (
                    int(pairing.get("common_result_mismatches", 1)) == 0
                    and int(pairing.get("parameter_conflict_count", 1)) == 0
                    and int(pairing.get("multiplicity_mismatch_count", 1)) == 0
                    and int(pairing.get("common_call_keys", 0)) > 0
                    and int(pairing.get("action_specific_call_keys", 0)) > 0
                )
                if not pairing_pass:
                    raise TechnicalStop("communication primitive pairing failed")
                return records, {
                    "status": "LABELED",
                    "split_role": split_role,
                    "parent": parent,
                    "repeat": repeat,
                    "decision_step": decision_step,
                    "candidate_count": len(candidates),
                    "decision_input_sha256": decision_input.sha256,
                    "pairing": pairing,
                    "explanation_labels": summaries,
                }

            selector = self.runtime["selector"]("hungarian", PREFERENCE)
            action, _diagnostic = self.account_call(
                "hungarian.prefix",
                {"public_rule_decisions": 1},
                selector.choose,
                public_obs,
                adapter.memory,
            )
            adapter.commit(int(action))
            observation, _reward, done, info = self._step(env, int(action))
            counts = {"completed": int(info["counts"]["completed"]), "expired": int(info["counts"]["expired"])}
            energy = sum(float(value) for value in info["energy"].values())
            if done:
                break
        return [], {
            "status": "NO_OPPORTUNITY",
            "split_role": split_role,
            "parent": parent,
            "repeat": repeat,
            "reason": "no public task-relevant candidate choice at or after step 4",
        }

    def collect_fixed_continuation_labels(self) -> Path:
        data_path = self.output / "learning-records.jsonl"
        units_path = self.output / "data-units.jsonl"
        coverage: dict[str, set[str]] = defaultdict(set)
        records_by_role: dict[str, int] = defaultdict(int)
        repeats = self.matrix["data_repeats"]
        for role in ("train", "model_selection", "prediction_evaluation"):
            for identity in self.matrix["groups"][role]:
                for repeat in range(int(repeats[role])):
                    records, unit = self._collect_unit(role, identity["parent"], repeat)
                    _append_jsonl(units_path, unit)
                    for record in records:
                        _append_jsonl(data_path, record)
                    if records:
                        coverage[role].add(identity["parent"])
                        records_by_role[role] += len(records)
                    self.ledger.assert_settled()
        minimum_parents = {"train": 18, "model_selection": 6, "prediction_evaluation": 6}
        failures = {
            role: {"observed": len(coverage[role]), "required": required}
            for role, required in minimum_parents.items()
            if len(coverage[role]) < required
        }
        data_status = {
            "parent_coverage": {role: len(value) for role, value in coverage.items()},
            "records": dict(records_by_role),
            "minimum_parent_coverage": minimum_parents,
            "pass": not failures,
            "failures": failures,
        }
        (self.output / "data-gate.json").write_text(
            json.dumps(data_status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if failures:
            raise TechnicalStop(f"data coverage gate failed: {failures}")
        return data_path

    def _model_choice(
        self,
        models: Sequence[object],
        observation: Mapping,
        memory,
        candidates: Sequence[int],
    ) -> tuple[dict[int, float], int]:
        import torch

        if len(models) != 3:
            raise TechnicalStop("task inference requires exactly three frozen seed models")
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        history = history_features(memory.history[:-1], float(observation["time"]))
        features = torch.tensor(
            [materialize_features(observation["flat"], history, action) for action in candidates],
            dtype=torch.float32,
        )
        seed_values: list[list[float]] = []
        with torch.no_grad():
            for model in models:
                output = self.account_call(
                    "model.forward.task",
                    {"model_batch_forwards": 1, "model_sample_evaluations": len(candidates)},
                    model,
                    features,
                )
                values = [float(value) for value in output[:, 0]]
                if len(values) != len(candidates) or not all(math.isfinite(value) for value in values):
                    raise TechnicalStop("task prediction is nonfinite or has the wrong candidate shape")
                seed_values.append(values)
        scores = {
            action: sum(seed_values[seed][index] for seed in range(3)) / 3.0
            for index, action in enumerate(candidates)
        }
        if not all(math.isfinite(value) for value in scores.values()):
            raise TechnicalStop("task ensemble prediction is nonfinite")
        action = max(candidates, key=lambda item: (scores[item], -item))
        self.decision_costs.append({
            "wall_seconds": time.perf_counter() - wall_start,
            "cpu_seconds": time.process_time() - cpu_start,
        })
        return scores, action

    def _task_episode(self, parent: str, repeat: int, method: str, models: Sequence[object]) -> dict:
        key = f"w1-action-outcome-learning-v1|task|{parent}|repeat-{repeat}"
        env = self._new_environment(parent, key)
        observation = self.account_call("environment.reset", {"resets_upper": 1}, env.reset)
        adapter = self.runtime["adapter"]()
        observer = self.runtime["require_observer"](env.communication)
        marker = observer.start_branch(method)
        rows = []
        used = False
        for decision_step in range(HORIZON):
            public_obs, _mask = adapter.prepare(observation)
            self.account_call("candidate.scan.task", {"candidate_scans": 1}, lambda: None)
            candidates = tuple(int(value) for value in adapter.memory.candidates(public_obs))
            qualified, transparent = qualifies(
                public_obs, adapter.memory, minimum_step=4, decision_step=decision_step
            )
            if qualified and not used and method != "hungarian":
                if method == "transparent_one_shot":
                    scores = transparent
                    action = max(candidates, key=lambda item: (float(scores[item]), -item))
                elif method in {"A_one_shot", "B_one_shot", "learned_three_seed_one_shot"}:
                    scores, action = self._model_choice(models, public_obs, adapter.memory, candidates)
                else:
                    raise TechnicalStop(f"unknown task method {method}")
                diagnostic = {"phase": "one_shot", "scores": scores}
                used = True
            else:
                selector = self.runtime["selector"]("hungarian", PREFERENCE)
                action, diagnostic = selector.choose(public_obs, adapter.memory)
            self.account_call("task.public_decision", {"public_rule_decisions": 1}, lambda: None)
            adapter.commit(int(action))
            observation, reward, done, info = self._step(env, int(action))
            rows.append({
                "step": decision_step,
                "action": int(action),
                "reward": reward,
                "done": done,
                "info": info,
                "diagnostic": diagnostic,
            })
            if done:
                break
        summary = self.runtime["label_episode"](
            rows,
            initial_counts={"completed": 0, "expired": 0},
            initial_energy=INITIAL_FLEET_ENERGY,
            scope="full_native_episode",
            identity={
                "parent": parent,
                "repeat": repeat,
                "arm": method,
                "external_key": key,
                "communication_profile": "W1-light",
                "continuation": "hungarian_after_single_intervention",
            },
        )
        summary["primitive_calls"] = observer.records_since(marker)
        summary["one_shot_used"] = used
        return summary

    def run_task_comparison(self, models: Mapping[str, Sequence[object]] | Sequence[object]) -> dict[str, Any]:
        # The frozen matrix has separate A and B one-shot arms.  Keep the
        # legacy single-ensemble form only for older callers, while the
        # production adapter always supplies both named ensembles.
        if isinstance(models, Mapping):
            model_map = {str(key): value for key, value in models.items()}
            for variant in ("A", "B"):
                if len(model_map.get(variant, ())) != 3:
                    raise TechnicalStop(f"task comparison requires three {variant} seed models")
        else:
            if len(models) != 3:
                raise TechnicalStop("task comparison requires the fixed three-seed ensemble")
            model_map = {"A": models, "B": models}
        summaries: list[dict] = []
        pairing_rows: list[dict] = []
        for identity in self.matrix["groups"]["task_evaluation"]:
            parent = identity["parent"]
            for repeat in range(int(self.matrix["task_repeats"])):
                unit = []
                for method in self.matrix["task_methods"]:
                    method_models = model_map.get("A" if method == "A_one_shot" else "B" if method == "B_one_shot" else "B", ())
                    unit.append(self._task_episode(parent, repeat, method, method_models))
                pairing = _verify_task_shared_fates(parent, repeat, unit)
                pairing_rows.append(pairing)
                summaries.extend(unit)
                self.ledger.assert_settled()

        utilities: dict[tuple[str, int, str], float] = {}
        for row in summaries:
            identity = row["identity"]
            utilities[(identity["parent"], int(identity["repeat"]), identity["arm"])] = float(row["discounted_utility"])
        by_parent: dict[str, dict[str, float]] = {}
        for identity in self.matrix["groups"]["task_evaluation"]:
            parent = identity["parent"]
            learned_delta = []
            transparent_delta = []
            for repeat in range(int(self.matrix["task_repeats"])):
                hungarian = utilities[(parent, repeat, "hungarian")]
                learned_delta.append(utilities[(parent, repeat, "B_one_shot")] - hungarian)
                transparent_delta.append(utilities[(parent, repeat, "transparent_one_shot")] - hungarian)
            a_delta = [utilities[(parent, repeat, "A_one_shot")] - utilities[(parent, repeat, "hungarian")] for repeat in range(int(self.matrix["task_repeats"]))]
            by_parent[parent] = {
                "A_minus_hungarian": sum(a_delta) / len(a_delta),
                "B_minus_hungarian": sum(learned_delta) / len(learned_delta),
                "learned_minus_hungarian": sum(learned_delta) / len(learned_delta),
                "transparent_minus_hungarian": sum(transparent_delta) / len(transparent_delta),
            }
        learned_macro = sum(value["learned_minus_hungarian"] for value in by_parent.values()) / len(by_parent)
        cpu_values = sorted(value["cpu_seconds"] for value in self.decision_costs)
        wall_values = sorted(value["wall_seconds"] for value in self.decision_costs)
        wall_p95 = wall_values[min(len(wall_values) - 1, math.ceil(0.95 * len(wall_values)) - 1)] if wall_values else None
        cpu_mean = sum(cpu_values) / len(cpu_values) if cpu_values else None
        result = {
            "schema": "w1-action-outcome-one-shot-task-comparison/1.0.0",
            "task_calls": len(summaries),
            "parent_results": by_parent,
            "learned_minus_hungarian_parent_macro_utility": learned_macro,
            "A_minus_hungarian_parent_macro_utility": sum(value["A_minus_hungarian"] for value in by_parent.values()) / len(by_parent),
            "B_minus_hungarian_parent_macro_utility": sum(value["B_minus_hungarian"] for value in by_parent.values()) / len(by_parent),
            "transparent_minus_hungarian_parent_macro_utility": sum(
                value["transparent_minus_hungarian"] for value in by_parent.values()
            ) / len(by_parent),
            "decision_cost": {
                "learned_calls": len(self.decision_costs),
                "cpu_mean_seconds": cpu_mean,
                "wall_p95_seconds": wall_p95,
            },
            "frozen_gate": {
                "utility_at_least_0.01": learned_macro >= 0.01,
                "cpu_mean_at_most_0.010_seconds": cpu_mean is not None and cpu_mean <= 0.010,
                "wall_p95_at_most_0.050_seconds": wall_p95 is not None and wall_p95 <= 0.050,
            },
            "summaries": summaries,
            "pairing": pairing_rows,
        }
        result["frozen_gate"]["pass"] = all(result["frozen_gate"].values())
        (self.output / "task-comparison.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result

    def settle(self, status: str, details: dict[str, Any]) -> None:
        settlement = {
            "status": status,
            "ledger": self.ledger.assert_settled() if self.ledger is not None else None,
            "automatic_retry": False,
            "automatic_extension": False,
            "historical_negative_results_preserved": True,
            "historical_10ms_cost_failure_preserved": True,
        }
        (self.output / "settlement.json").write_text(
            json.dumps(settlement, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def close(self) -> None:
        if self.ledger is not None:
            self.ledger.close()
