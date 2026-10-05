"""Transparent observation and pairing of frozen communication primitives."""

from __future__ import annotations

import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


METHODS = ("telemetry", "command_delivered", "ack_delivered", "renewal")


class ObservationError(RuntimeError):
    pass


class PairingMismatch(ObservationError):
    pass


class InsufficientPairingEvidence(ObservationError):
    pass


def primitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [primitive(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported communication evidence type: {type(value).__name__}")


def canonical(value: Any) -> str:
    return json.dumps(primitive(value), sort_keys=True, separators=(",", ":"))


class RecordingCommunication:
    """Record existing primitive calls without changing their execution."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []
        self.branch: str | None = None

    def __getattr__(self, name: str) -> Any:
        delegate = self.__dict__.get("delegate")
        if delegate is None:
            raise AttributeError(name)
        return getattr(delegate, name)

    def __deepcopy__(self, memo: dict[int, Any]) -> "RecordingCommunication":
        copied = type(self)(copy.deepcopy(self.delegate, memo))
        memo[id(self)] = copied
        copied.calls = copy.deepcopy(self.calls, memo)
        copied.branch = self.branch
        return copied

    def start_branch(self, branch: str) -> int:
        if not isinstance(branch, str) or not branch:
            raise ObservationError("branch identity is required")
        self.branch = branch
        return len(self.calls)

    def records_since(self, marker: int) -> list[dict[str, Any]]:
        if not isinstance(marker, int) or marker < 0 or marker > len(self.calls):
            raise ObservationError("invalid communication call marker")
        return copy.deepcopy(self.calls[marker:])

    def _record(self, method: str, kwargs: dict[str, Any], function) -> Any:
        if method not in METHODS:
            raise ObservationError(f"unsupported communication primitive: {method}")
        if "identity" not in kwargs or not isinstance(kwargs["identity"], str):
            raise ObservationError("communication call lacks full identity")
        caller = sys._getframe(2)
        callsite = (
            f"{Path(caller.f_code.co_filename).name}:"
            f"{caller.f_lineno}:{caller.f_code.co_name}"
        )
        del caller
        record = {
            "ordinal": len(self.calls),
            "branch": self.branch,
            "callsite": callsite,
            "method": method,
            "arguments": primitive(kwargs),
        }
        try:
            result = function(**kwargs)
        except BaseException as exc:
            record.update(
                {
                    "status": "raised",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                }
            )
            self.calls.append(record)
            raise
        record.update({"status": "returned", "result": primitive(result)})
        self.calls.append(record)
        return result

    def telemetry(self, **kwargs: Any) -> Any:
        return self._record("telemetry", kwargs, self.delegate.telemetry)

    def command_delivered(self, **kwargs: Any) -> Any:
        return self._record(
            "command_delivered", kwargs, self.delegate.command_delivered
        )

    def ack_delivered(self, **kwargs: Any) -> Any:
        return self._record("ack_delivered", kwargs, self.delegate.ack_delivered)

    def renewal(self, **kwargs: Any) -> Any:
        return self._record("renewal", kwargs, self.delegate.renewal)


def require_observer(value: Any) -> RecordingCommunication:
    if not isinstance(value, RecordingCommunication):
        raise ObservationError("runtime communication observer is not installed")
    return value


def _validated_calls(summary: dict[str, Any]) -> list[dict[str, Any]]:
    arm = str(summary.get("identity", {}).get("arm", ""))
    calls = summary.get("primitive_calls")
    if not arm or not isinstance(calls, list):
        raise InsufficientPairingEvidence("branch call evidence is missing")
    result = []
    previous_ordinal = -1
    for source_index, call in enumerate(calls):
        required = {"ordinal", "branch", "callsite", "method", "arguments", "status"}
        if not isinstance(call, dict) or not required <= call.keys():
            raise InsufficientPairingEvidence("primitive call lacks required fields")
        if call["branch"] != arm or call["method"] not in METHODS:
            raise InsufficientPairingEvidence("primitive call branch or method mismatch")
        arguments = call["arguments"]
        if not isinstance(arguments, dict) or not isinstance(arguments.get("identity"), str):
            raise InsufficientPairingEvidence("primitive call lacks full identity")
        ordinal = int(call["ordinal"])
        if ordinal <= previous_ordinal:
            raise InsufficientPairingEvidence("primitive call order is not strictly increasing")
        previous_ordinal = ordinal
        if call["status"] == "returned" and "result" not in call:
            raise InsufficientPairingEvidence("returned primitive call lacks result")
        if call["status"] == "raised":
            raise ObservationError("communication primitive raised; evidence remains recorded")
        if call["status"] != "returned":
            raise InsufficientPairingEvidence("unknown primitive call status")
        row = copy.deepcopy(call)
        row["source_index"] = source_index
        result.append(row)
    return result


def verify_primitive_pairing(
    parent: str,
    repeat: int,
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(summaries) < 2:
        raise InsufficientPairingEvidence("fewer than two branch summaries")
    by_arm: dict[str, list[dict[str, Any]]] = {}
    by_exact: dict[str, dict[str, list[dict[str, Any]]]] = {}
    base_parameters: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for summary in summaries:
        arm = str(summary["identity"]["arm"])
        if arm in by_arm:
            raise InsufficientPairingEvidence("duplicate branch summary")
        calls = _validated_calls(summary)
        by_arm[arm] = calls
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for call in calls:
            method = str(call["method"])
            arguments = call["arguments"]
            identity = str(arguments["identity"])
            exact_key = canonical({"method": method, "arguments": arguments})
            base_key = canonical({"method": method, "identity": identity})
            parameter_signature = canonical(
                {key: value for key, value in arguments.items() if key != "identity"}
            )
            grouped[exact_key].append(call)
            base_parameters[base_key][arm].add(parameter_signature)
        by_exact[arm] = dict(grouped)

    exact_arms: dict[str, list[str]] = defaultdict(list)
    for arm, grouped in by_exact.items():
        for exact_key in grouped:
            exact_arms[exact_key].append(arm)
    common = {
        key: sorted(arms) for key, arms in exact_arms.items() if len(arms) >= 2
    }
    if not common:
        raise InsufficientPairingEvidence("common primitive call set is empty")

    result_mismatches = []
    multiplicity_mismatches = []
    repeated_call_keys = []
    for key, arms in sorted(common.items()):
        sequences = {
            arm: [call["result"] for call in by_exact[arm][key]] for arm in arms
        }
        counts = {arm: len(values) for arm, values in sequences.items()}
        if max(counts.values()) > 1:
            repeated_call_keys.append({"key": json.loads(key), "counts": counts})
        if len(set(counts.values())) > 1:
            multiplicity_mismatches.append(
                {"key": json.loads(key), "counts": counts}
            )
        observed_results = {
            canonical(value)
            for values in sequences.values()
            for value in values
        }
        if len(observed_results) > 1:
            result_mismatches.append(
                {
                    "key": json.loads(key),
                    "arms": arms,
                    "results": sequences,
                }
            )
    if result_mismatches:
        raise PairingMismatch("common primitive call has inconsistent random fate")

    parameter_conflicts = []
    for base_key, by_branch in sorted(base_parameters.items()):
        if len(by_branch) < 2:
            continue
        signatures = {
            signature for values in by_branch.values() for signature in values
        }
        if len(signatures) > 1:
            parameter_conflicts.append(
                {
                    "base": json.loads(base_key),
                    "parameters_by_arm": {
                        arm: [json.loads(value) for value in sorted(values)]
                        for arm, values in sorted(by_branch.items())
                    },
                }
            )

    action_specific = {
        key: arms[0] for key, arms in exact_arms.items() if len(arms) == 1
    }
    if not action_specific:
        raise InsufficientPairingEvidence(
            "action-specific primitive call set is empty"
        )
    return {
        "schema": "w1-oracle-primitive-pairing/1.0.0",
        "parent": parent,
        "repeat": repeat,
        "branch_count": len(summaries),
        "common_call_keys": len(common),
        "action_specific_call_keys": len(action_specific),
        "common_result_mismatches": 0,
        "parameter_conflict_count": len(parameter_conflicts),
        "parameter_conflicts": parameter_conflicts,
        "multiplicity_mismatch_count": len(multiplicity_mismatches),
        "multiplicity_mismatches": multiplicity_mismatches,
        "repeated_common_call_keys": repeated_call_keys,
        "external_fate_only": True,
        "execution_outcomes_compared_as_fate": False,
        "duplicates_preserved": True,
        "empty_common_set_pass_prohibited": True,
        "empty_action_specific_set_pass_prohibited": True,
    }
