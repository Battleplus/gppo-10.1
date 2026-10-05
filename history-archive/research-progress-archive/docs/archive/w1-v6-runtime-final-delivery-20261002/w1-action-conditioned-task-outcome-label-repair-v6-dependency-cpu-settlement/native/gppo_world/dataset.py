"""Dataset-manifest helpers and strict group-split auditing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SPLITS = ("train", "validation", "test")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_key(episode: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(episode["scenario_id"]), str(episode["tape_id"]), int(episode["seed"]))


def audit_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    episodes = list(manifest.get("episodes", ()))
    groups: dict[tuple[str, str, int], set[str]] = {}
    policies: set[str] = set()
    profiles: set[str] = set()
    transition_total = 0
    errors: list[str] = []
    for episode in episodes:
        split = str(episode.get("split"))
        if split not in SPLITS:
            errors.append(f"unknown split {split!r}")
            continue
        key = group_key(episode)
        groups.setdefault(key, set()).add(split)
        policies.add(str(episode.get("behavior_policy")))
        profiles.add(str(episode.get("scenario_id")))
        transition_total += int(episode.get("transition_count", 0))
    overlaps = {
        "/".join((scenario, tape, str(seed))): sorted(splits)
        for (scenario, tape, seed), splits in groups.items()
        if len(splits) > 1
    }
    if overlaps:
        errors.append("scenario/tape/seed groups cross splits")
    required_policies = {"random_legal", "greedy", "gppo"}
    if policies != required_policies:
        errors.append(f"behavior policies are {sorted(policies)}, expected {sorted(required_policies)}")
    required_profiles = {"normal", "single", "sequential", "overlap", "burst", "long_gap", "weak_comm"}
    missing_profiles = sorted(required_profiles - profiles)
    if missing_profiles:
        errors.append(f"missing profiles: {missing_profiles}")
    files = manifest.get("files", {})
    for split in SPLITS:
        if split not in files:
            errors.append(f"missing dataset file for {split}")
    return {
        "passed": not errors,
        "errors": errors,
        "episode_count": len(episodes),
        "transition_count": transition_total,
        "group_count": len(groups),
        "split_overlap_count": len(overlaps),
        "split_overlaps": overlaps,
        "behavior_policies": sorted(policies),
        "profiles": sorted(profiles),
    }


def canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_manifest_bytes(manifest))
    return output


def split_group_sets(episodes: Iterable[Mapping[str, Any]]) -> dict[str, set[tuple[str, str, int]]]:
    result = {split: set() for split in SPLITS}
    for episode in episodes:
        result[str(episode["split"])].add(group_key(episode))
    return result
