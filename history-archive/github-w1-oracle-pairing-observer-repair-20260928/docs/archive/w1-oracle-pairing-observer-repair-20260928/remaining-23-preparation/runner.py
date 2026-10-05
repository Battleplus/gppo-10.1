"""One-shot repaired W1 action-consequence oracle ceiling runner."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pickle
import signal
import sys
from pathlib import Path

from accounting import AccountingError, Ledger, canonical
from analyze_oracle import analyze
from communication_observer import (
    RecordingCommunication,
    require_observer,
    verify_primitive_pairing,
)
from infra_io import durable_append_jsonl, durable_atomic_json, read_json, verify_manifest
from labeling import label_episode
from reuse_audit import audit_reuse


ATTEMPT = "w1-action-consequence-oracle-remaining-23-v1-once"
HORIZON = 18
PARENTS = tuple(f"validation-{index:04d}" for index in range(8))
REPEATS = (0, 1, 2)
REUSED_UNIT = ("validation-0000", 0)
NOOP = 24
MAX_NONNOOP = 24
MAX_TOTAL_BRANCHES = 25
PREFERENCE = (0.8, 0.2)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "run-once"
REUSE = ROOT / "reused-first-unit"


class TechnicalStop(RuntimeError):
    pass


class InsufficientTechnicalCoverage(TechnicalStop):
    pass


def primitive(value):
    if hasattr(value, "tolist"):
        return primitive(value.tolist())
    if isinstance(value, dict):
        return {str(key): primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [primitive(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("nonfinite or unsupported evidence value")


def json_digest(value) -> str:
    return hashlib.sha256(canonical(primitive(value)).encode("utf-8")).hexdigest()


def snapshot_digest(value) -> str:
    return hashlib.sha256(
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def unit_key(parent: str, repeat: int) -> str:
    return f"{parent}|repeat-{repeat}"


def external_key(parent: str, repeat: int) -> str:
    return f"w1-action-consequence-oracle-ceiling-v1|{parent}|repeat-{repeat}"


def announce(output: Path, stage: str, parent: str | None = None, repeat: int | None = None) -> None:
    durable_atomic_json(
        output / "activity.json",
        {"stage": stage, "parent": parent, "repeat": repeat},
    )


def _candidate_scan(public_obs: dict, memory) -> dict:
    candidates = tuple(int(action) for action in memory.candidates(public_obs))
    if len(candidates) != len(set(candidates)):
        raise TechnicalStop("duplicate effective candidate")
    nonnoop = tuple(action for action in candidates if action != NOOP)
    if len(nonnoop) > MAX_NONNOOP or len(candidates) > MAX_TOTAL_BRANCHES:
        raise TechnicalStop("candidate cap exceeded; truncation prohibited")
    ids = public_obs["public_entity_ids"]
    descriptors = {
        str(action): (
            {"kind": "NOOP"}
            if action == NOOP
            else {
                "kind": "assignment",
                "uav_id": ids["uavs"][action // 6],
                "task_id": ids["tasks"][action % 6],
            }
        )
        for action in candidates
    }
    assignments = {
        (descriptors[str(action)]["uav_id"], descriptors[str(action)]["task_id"])
        for action in nonnoop
    }
    return {
        "candidates": candidates,
        "nonnoop": nonnoop,
        "descriptors": descriptors,
        "preliminary_qualified": len(nonnoop) >= 2 and len(assignments) >= 2,
    }


def _call_reset(ledger: Ledger, env):
    return ledger.call("environment.reset", {"resets": 1}, env.reset)


def _call_step(
    ledger: Ledger,
    env,
    action: int,
    *,
    output: Path,
    stage: str,
    parent: str,
    repeat: int,
    arm: str,
    actual_step: int,
    public_observation: dict,
):
    following, reward, done, info = ledger.call(
        "environment.step",
        {"environment_steps": 1},
        env.step,
        int(action),
    )
    record = {
        "stage": stage,
        "parent": parent,
        "repeat": repeat,
        "arm": arm,
        "actual_step": actual_step,
        "action": int(action),
        "public_observation_sha256": json_digest(public_observation),
        "reward": reward,
        "done": done,
        "info": info,
    }
    durable_append_jsonl(output / "steps.jsonl", primitive(record))
    return following, reward, bool(done), info


def _run_branch(
    capture,
    selection: dict,
    action: int,
    *,
    ledger: Ledger,
    output: Path,
    selector_factory,
) -> dict:
    parent = selection["parent"]
    repeat = int(selection["repeat"])
    stage = selection["stage"]
    arm = f"action-{action:02d}"
    ledger.call("branch.register", {"branches": 1}, lambda: None)
    expected = snapshot_digest(capture)
    copied = ledger.call(
        "branch.deepcopy",
        {"branch_snapshot_copies": 1},
        copy.deepcopy,
        capture,
    )
    env, adapter, public_obs, initial_counts, initial_energy, decision_step, key = copied
    if snapshot_digest(capture) != expected:
        raise TechnicalStop("source snapshot changed during deepcopy")
    if getattr(env, "_exogenous_key", None) != key:
        raise TechnicalStop("branch external key changed")
    observer = require_observer(env.communication)
    call_marker = observer.start_branch(arm)
    ledger.call("branch.force_first", {"forced_first_actions": 1}, adapter.commit, int(action))
    following, reward, done, info = _call_step(
        ledger,
        env,
        action,
        output=output,
        stage=stage,
        parent=parent,
        repeat=repeat,
        arm=arm,
        actual_step=decision_step,
        public_observation=public_obs,
    )
    rows = [{
        "step": 0,
        "action": int(action),
        "reward": reward,
        "done": done,
        "info": info,
    }]
    current = following
    local_step = 1
    for actual_step in range(decision_step + 1, HORIZON):
        if done:
            break
        selector = selector_factory()
        decision = ledger.call(
            "hungarian.continuation",
            {"public_rule_decisions": 1},
            adapter.decide,
            current,
            selector,
        )
        public_for_step = adapter.memory.history[-1]
        current, reward, done, info = _call_step(
            ledger,
            env,
            int(decision["action"]),
            output=output,
            stage=stage,
            parent=parent,
            repeat=repeat,
            arm=arm,
            actual_step=actual_step,
            public_observation=public_for_step,
        )
        rows.append({
            "step": local_step,
            "action": int(decision["action"]),
            "diagnostic": decision["diagnostic"],
            "reward": reward,
            "done": done,
            "info": info,
        })
        local_step += 1
    summary = label_episode(
        rows,
        initial_counts=initial_counts,
        initial_energy=initial_energy,
        scope=f"t{decision_step}_to_native_terminal",
        identity={
            "parent": parent,
            "repeat": repeat,
            "arm": arm,
            "action": int(action),
            "assignment": selection["action_descriptors"][str(action)],
            "external_key": key,
            "communication_profile": "W1-light",
            "continuation": "hungarian",
            "public_state_sha256": selection["public_state_sha256"],
        },
    )
    summary["primitive_calls"] = observer.records_since(call_marker)
    summary["execution_outcomes_preserved_separately"] = True
    durable_append_jsonl(output / "branch-summaries.jsonl", primitive(summary))
    if snapshot_digest(capture) != expected:
        raise TechnicalStop("branch contaminated source snapshot")
    return summary


def execute_unit(
    *,
    parent: str,
    repeat: int,
    scenario: dict,
    stage: str,
    ledger: Ledger,
    output: Path,
    runtime_factory,
    adapter_factory,
    selector_factory,
    initial_fleet_energy: float,
) -> tuple[dict, list[dict], dict | None]:
    key = external_key(parent, repeat)
    ledger.select(stage, unit_key(parent, repeat))
    env = runtime_factory(scenario, key)
    if getattr(env, "_exogenous_key", None) != key:
        raise TechnicalStop("runtime did not retain the frozen exogenous key")
    observation = _call_reset(ledger, env)
    adapter = adapter_factory()
    counts = {"completed": 0, "expired": 0}
    energy = float(initial_fleet_energy)
    summaries: list[dict] = []

    for decision_step in range(HORIZON):
        public_obs, _mask = adapter.prepare(observation)
        scan = ledger.call(
            "candidate.scan",
            {"candidate_scans": 1},
            _candidate_scan,
            public_obs,
            adapter.memory,
        )
        baseline = None
        baseline_diagnostic = None
        if scan["preliminary_qualified"]:
            baseline, baseline_diagnostic = ledger.call(
                "hungarian.opportunity_check",
                {"public_rule_decisions": 1},
                selector_factory().choose,
                public_obs,
                adapter.memory,
            )
        qualified = (
            scan["preliminary_qualified"]
            and baseline in scan["nonnoop"]
        )
        if qualified:
            branch_actions = tuple(scan["nonnoop"]) + (
                (NOOP,) if NOOP in scan["candidates"] else ()
            )
            if len(branch_actions) > MAX_TOTAL_BRANCHES:
                raise TechnicalStop("NOOP would create a branch beyond total cap")
            selection = {
                "parent": parent,
                "repeat": repeat,
                "stage": stage,
                "status": "OPPORTUNITY",
                "decision_step": decision_step,
                "time": float(public_obs["time"]),
                "external_key": key,
                "public_state_sha256": json_digest(public_obs),
                "hungarian_action": int(baseline),
                "hungarian_diagnostic": baseline_diagnostic,
                "effective_legal_actions": list(scan["candidates"]),
                "nonnoop_actions": list(scan["nonnoop"]),
                "branch_actions": list(branch_actions),
                "action_descriptors": scan["descriptors"],
                "primary_oracle_excludes_noop": True,
                "secondary_oracle_includes_noop_if_legal": NOOP in branch_actions,
            }
            durable_append_jsonl(output / "candidate-decisions.jsonl", selection)
            capture = ledger.call(
                "snapshot.capture",
                {"snapshot_captures": 1},
                lambda: (
                    env,
                    adapter,
                    public_obs,
                    counts,
                    energy,
                    decision_step,
                    key,
                ),
            )
            source_digest = snapshot_digest(capture)
            for action in branch_actions:
                summaries.append(
                    _run_branch(
                        capture,
                        selection,
                        int(action),
                        ledger=ledger,
                        output=output,
                        selector_factory=selector_factory,
                    )
                )
                if snapshot_digest(capture) != source_digest:
                    raise TechnicalStop("source snapshot changed between branches")
            pairing = verify_primitive_pairing(parent, repeat, summaries)
            durable_append_jsonl(output / "communication-pairing.jsonl", pairing)
            return selection, summaries, pairing

        if baseline is None:
            baseline, baseline_diagnostic = ledger.call(
                "hungarian.prefix",
                {"public_rule_decisions": 1},
                selector_factory().choose,
                public_obs,
                adapter.memory,
            )
        adapter.commit(int(baseline))
        observation, _reward, done, info = _call_step(
            ledger,
            env,
            int(baseline),
            output=output,
            stage=stage,
            parent=parent,
            repeat=repeat,
            arm="hungarian-prefix",
            actual_step=decision_step,
            public_observation=public_obs,
        )
        counts = {
            "completed": int(info["counts"]["completed"]),
            "expired": int(info["counts"]["expired"]),
        }
        energy_values = info["energy"]
        if not isinstance(energy_values, dict) or not energy_values:
            raise TechnicalStop("prefix step lacks fleet energy")
        energy = sum(float(value) for value in energy_values.values())
        if done:
            break

    selection = {
        "parent": parent,
        "repeat": repeat,
        "stage": stage,
        "status": "NO_OPPORTUNITY",
        "reason": "native_terminal_or_horizon_without_qualifying_public_state",
        "external_key": key,
    }
    durable_append_jsonl(output / "candidate-decisions.jsonl", selection)
    return selection, [], None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def execute_matrix(
    *,
    tape_rows: list[dict],
    output: Path,
    request: dict,
    runtime_factory,
    adapter_factory,
    selector_factory,
    initial_fleet_energy: float,
    reuse_dir: Path = REUSE,
) -> dict:
    if output.exists():
        raise FileExistsError("attempt output exists; retry prohibited")
    output.mkdir(parents=True)
    ledger = Ledger(output / "budget.sqlite3", request)
    by_parent = {row["parent"]: row for row in tape_rows}
    if tuple(by_parent) != PARENTS or len(by_parent) != len(tape_rows):
        raise TechnicalStop("frozen parent order or uniqueness mismatch")
    for row in tape_rows:
        scenario = row["scenario"]
        if (
            row.get("condition") != "W1"
            or scenario.get("communication", {}).get("name") != "W1-light"
        ):
            raise TechnicalStop("non-W1-light tape row")

    try:
        announce(output, "staging_and_zero_step_gate")
        reuse = audit_reuse(ROOT, reuse_dir)
        durable_atomic_json(output / "reuse-audit.json", reuse)

        units = [
            (parent, repeat)
            for parent in PARENTS
            for repeat in REPEATS
            if (parent, repeat) != REUSED_UNIT
        ]
        if len(units) != 23 or REUSED_UNIT in units:
            raise TechnicalStop("remaining-unit enumeration is not exactly 23 units")
        for parent, repeat in units:
            stage = "remaining_fixed_units"
            announce(output, stage, parent, repeat)
            execute_unit(
                parent=parent,
                repeat=repeat,
                scenario=by_parent[parent]["scenario"],
                stage=stage,
                ledger=ledger,
                output=output,
                runtime_factory=runtime_factory,
                adapter_factory=adapter_factory,
                selector_factory=selector_factory,
                initial_fleet_energy=initial_fleet_energy,
            )
            ledger.assert_settled()

        announce(output, "settlement")
        candidate_rows = _read_jsonl(reuse_dir / "candidate-decisions.jsonl") + _read_jsonl(
            output / "candidate-decisions.jsonl"
        )
        summaries = _read_jsonl(reuse_dir / "branch-summaries.jsonl") + _read_jsonl(
            output / "branch-summaries.jsonl"
        )
        pairing_rows = _read_jsonl(reuse_dir / "communication-pairing.jsonl") + _read_jsonl(
            output / "communication-pairing.jsonl"
        )
        analysis = analyze(
            candidate_rows,
            summaries,
            pairing_rows,
            reused_audit=reuse,
        )
        durable_atomic_json(output / "analysis.json", analysis)
        status = {
            "status": "complete",
            "attempt": ATTEMPT,
            "ledger": ledger.assert_settled(),
            "reuse_audit": "reuse-audit.json",
            "analysis": "analysis.json",
            "reused_units": 1,
            "new_units": 23,
            "fixed_matrix_units": 24,
            "first_unit_reexecuted": False,
            "oracle_materiality_evaluated": True,
            "model_initializations_or_loads": 0,
            "model_forwards": 0,
            "training_updates": 0,
            "automatic_retry": False,
            "historical_10ms_cost_failure_preserved": True,
        }
        durable_atomic_json(output / "status.json", status)
        return status
    except BaseException as exc:
        durable_atomic_json(
            output / "status.json",
            {
                "status": "technical_stop_no_retry",
                "attempt": ATTEMPT,
                "error": type(exc).__name__ + ": " + str(exc),
                "ledger": ledger.snapshot(),
                "model_initializations_or_loads": 0,
                "model_forwards": 0,
                "training_updates": 0,
                "automatic_retry": False,
                "historical_10ms_cost_failure_preserved": True,
            },
        )
        raise
    finally:
        ledger.close()


def _signal_stop(sig, _frame):
    raise TechnicalStop("received stop signal " + str(sig))


def main() -> int:
    signal.signal(signal.SIGTERM, _signal_stop)
    signal.signal(signal.SIGINT, _signal_stop)
    manifest = read_json(ROOT / "execution-manifest.json")
    verify_manifest(ROOT, manifest["files"])
    request = read_json(ROOT / "RESOURCE_REQUEST.json")
    if request.get("status") != "NOT_APPROVED":
        raise TechnicalStop("frozen request status changed")
    if request.get("attempt") != ATTEMPT or manifest.get("attempt") != ATTEMPT:
        raise TechnicalStop("attempt identity mismatch")
    if request.get("status") != "NOT_APPROVED":
        raise TechnicalStop("resource request is not frozen as NOT_APPROVED")
    if sys.version_info[:3] != (3, 11, 16):
        raise TechnicalStop("frozen Python runtime mismatch")
    sys.path.insert(0, str(ROOT / "native"))
    import numpy as np
    from gppo_world.m10_environment import M10Config, M10Environment, scenario_from_dict
    from classical_baselines import ClassicalSelector, PublicDecisionAdapter

    if np.__version__ != "1.26.0":
        raise TechnicalStop("frozen NumPy runtime mismatch")
    config = M10Config(**read_json(ROOT / "environment.json"))
    tape_rows = read_json(ROOT / "tapes-light-dev.json")

    def runtime_factory(scenario, key):
        env = M10Environment(config, scenario_from_dict(scenario), exogenous_key=key)
        env.communication = RecordingCommunication(env.communication)
        return env

    def selector_factory():
        return ClassicalSelector("hungarian", PREFERENCE)

    execute_matrix(
        tape_rows=tape_rows,
        output=OUT,
        request=request,
        runtime_factory=runtime_factory,
        adapter_factory=PublicDecisionAdapter,
        selector_factory=selector_factory,
        initial_fleet_energy=float(config.uav_count * config.initial_energy),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
