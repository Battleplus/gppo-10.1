"""Frozen feature registry for GPPO-8.29's 4-UAV/4-region/3-target graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping


SCHEMA_VERSION = "gppo-world-transition/0.1.0"
Relation = tuple[str, str, str]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    modality: str
    event_eligible: bool
    normalization: str


@dataclass(frozen=True)
class FeatureRegistry:
    schema_version: str
    nodes: Mapping[str, tuple[FeatureSpec, ...]]
    edges: Mapping[Relation, tuple[FeatureSpec, ...]]

    @property
    def node_dimensions(self) -> Mapping[str, int]:
        return MappingProxyType({name: len(features) for name, features in self.nodes.items()})

    @property
    def edge_dimensions(self) -> Mapping[Relation, int]:
        return MappingProxyType({relation: len(features) for relation, features in self.edges.items()})

    def canonical_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "nodes": {
                name: [asdict(feature) for feature in features]
                for name, features in sorted(self.nodes.items())
            },
            "edges": {
                "/".join(relation): [asdict(feature) for feature in features]
                for relation, features in sorted(self.edges.items())
            },
        }

    def sha256(self) -> str:
        payload = json.dumps(self.canonical_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _f(name: str, modality: str, eligible: bool = True, normalization: str = "identity") -> FeatureSpec:
    return FeatureSpec(name, modality, eligible, normalization)


FEATURE_REGISTRY = FeatureRegistry(
    schema_version=SCHEMA_VERSION,
    nodes=MappingProxyType(
        {
            "uav": (
                _f("alive", "nominal"), _f("sensor_available", "nominal"),
                _f("task_search", "nominal"), _f("task_track", "nominal"), _f("task_idle", "nominal"),
                _f("x", "ordinal", normalization="area_size"), _f("y", "ordinal", normalization="area_size"),
                _f("region_load", "ordinal", normalization="num_regions"),
                _f("target_0", "nominal"), _f("target_1", "nominal"),
                _f("target_2", "nominal"), _f("target_none", "nominal"),
            ),
            "region": (
                _f("center_x", "ordinal", False, "area_size"), _f("center_y", "ordinal", False, "area_size"),
                _f("priority", "ordinal"), _f("workload", "ordinal"), _f("vacancy_duration", "ordinal", True, "horizon"),
                _f("assigned_uav_0", "nominal"), _f("assigned_uav_1", "nominal"),
                _f("assigned_uav_2", "nominal"), _f("assigned_uav_3", "nominal"),
                _f("assigned_none", "nominal"), _f("pending", "nominal"), _f("assignment_legal", "nominal"),
            ),
            "target": (
                _f("type_0", "nominal", False), _f("type_1", "nominal", False),
                _f("discovered", "nominal"), _f("tracked", "nominal"), _f("destroyed", "nominal"),
                _f("x", "ordinal", True, "area_size"), _f("y", "ordinal", True, "area_size"),
                _f("region_0", "nominal", False), _f("region_1", "nominal", False),
                _f("region_2", "nominal", False), _f("region_3", "nominal", False),
                _f("tracker_0", "nominal"), _f("tracker_1", "nominal"),
                _f("tracker_2", "nominal"), _f("tracker_3", "nominal"), _f("tracker_none", "nominal"),
            ),
        }
    ),
    edges=MappingProxyType(
        {
            ("uav", "can_serve", "region"): (
                _f("capable", "nominal"), _f("distance", "ordinal", True, "scene_diagonal"),
                _f("current_assignment", "nominal"), _f("uav_load", "ordinal", True, "num_regions"),
                _f("communication_quality", "ordinal"),
            ),
            ("region", "served_by", "uav"): (
                _f("capable", "nominal"), _f("distance", "ordinal", True, "scene_diagonal"),
                _f("current_assignment", "nominal"), _f("uav_load", "ordinal", True, "num_regions"),
                _f("communication_quality", "ordinal"),
            ),
            ("region", "adjacent", "region"): (_f("adjacent", "structural", False),),
            ("target", "located_in", "region"): (_f("discovered", "nominal"), _f("destroyed", "nominal")),
            ("region", "contains", "target"): (_f("discovered", "nominal"), _f("destroyed", "nominal")),
            ("uav", "tracks", "target"): (_f("tracking", "nominal"), _f("discovered", "nominal")),
            ("target", "tracked_by", "uav"): (_f("tracking", "nominal"), _f("discovered", "nominal")),
            ("uav", "communicates", "uav"): (_f("communication_quality", "ordinal"),),
        }
    ),
)
