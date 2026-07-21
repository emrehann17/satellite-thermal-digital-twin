"""Read-only source-scene provenance and artifact-lineage construction.

Only explicit local producer metadata is consumed.  This module never
initializes Earth Engine, submits exports, or claims that a median/mean
composite selected one source scene.
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from rasterio.warp import transform_geom

from core.seam_audit_v2_config import resolve_product_registry_v2
from core.source_scene_provenance_config import SCHEMA_VERSION, VERSION


SCENE_COLUMNS = [
    "experiment_id", "source_collection", "scene_code", "scene_id",
    "landsat_product_id", "spacecraft_id", "sensor_id", "wrs_path",
    "wrs_row", "path_row", "acquisition_datetime", "acquisition_date",
    "cloud_cover", "cloud_cover_land", "processing_level",
    "collection_category", "collection_number", "scene_footprint_wkt",
    "scene_footprint_crs", "source_metadata_method", "input_role",
    "baseline_year", "role", "metadata_source",
]
_REDUCER_COMPOSITES = {"median", "mean", "weighted_composite"}
_SELECTING_COMPOSITES = {
    "mosaic", "quality_mosaic", "latest_valid", "earliest_valid", "single_scene",
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _hash(prefix: str, value: Any, length: int = 20) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return prefix + hashlib.sha256(payload.encode()).hexdigest()[:length]


def _rings(geometry: dict[str, Any] | None) -> list[list[list[float]]]:
    if not isinstance(geometry, dict):
        return []
    if geometry.get("type") == "Polygon":
        coordinates = geometry.get("coordinates") or []
        return [coordinates[0]] if coordinates else []
    if geometry.get("type") == "MultiPolygon":
        return [polygon[0] for polygon in geometry.get("coordinates", []) if polygon]
    return []


def _geometry_wkt(geometry: dict[str, Any] | None) -> str | None:
    rings = _rings(geometry)
    if not rings:
        return None

    def ring_text(ring: Iterable[Iterable[float]]) -> str:
        return ", ".join(f"{float(point[0]):.12g} {float(point[1]):.12g}" for point in ring)

    if geometry and geometry.get("type") == "Polygon":
        return f"POLYGON (({ring_text(rings[0])}))"
    return "MULTIPOLYGON (" + ", ".join(f"(({ring_text(ring)}))" for ring in rings) + ")"


def _scene_records(value: Any, role: str | None = None) -> list[tuple[dict[str, Any], str | None]]:
    """Extract explicitly declared scene lists without recursive file discovery."""
    result: list[tuple[dict[str, Any], str | None]] = []
    if isinstance(value, list):
        return [(item, role) for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return result
    for key in ("scenes", "source_scenes", "scene_manifest"):
        if isinstance(value.get(key), list):
            result.extend(_scene_records(value[key], role))
    collections = value.get("collections")
    if isinstance(collections, dict):
        for collection_role, collection in collections.items():
            result.extend(_scene_records(collection, str(collection_role)))
    for collection_role in (
        "current_lst", "current_ndvi", "baseline_lst", "baseline_ndvi",
        "baseline_lst_yearly", "baseline_ndvi_yearly",
    ):
        if isinstance(value.get(collection_role), dict):
            result.extend(_scene_records(value[collection_role], collection_role))
    return result


def normalize_scene(raw: dict[str, Any], role: str | None, source: Path) -> dict[str, Any]:
    props = raw.get("properties", raw)
    scene_id = (
        props.get("scene_id") or props.get("LANDSAT_SCENE_ID")
        or props.get("landsat_product_id") or props.get("product_id")
        or props.get("LANDSAT_PRODUCT_ID") or props.get("system:index")
        or raw.get("id")
    )
    wrs_path = props.get("wrs_path", props.get("WRS_PATH"))
    wrs_row = props.get("wrs_row", props.get("WRS_ROW"))
    acquired = (
        props.get("acquisition_datetime") or props.get("system:time_start")
        or props.get("DATE_ACQUIRED") or props.get("acquisition_date")
    )
    if isinstance(acquired, (int, float)):
        acquired = datetime.fromtimestamp(float(acquired) / 1000.0, timezone.utc).isoformat()
    geometry = raw.get("geometry") or props.get("footprint") or raw.get("footprint")
    footprint_crs = (
        props.get("scene_footprint_crs") or props.get("footprint_crs")
        or raw.get("crs") or "EPSG:4326"
    )
    if geometry is not None and str(footprint_crs) != "EPSG:4326":
        try:
            geometry = transform_geom(footprint_crs, "EPSG:4326", geometry)
            footprint_crs = "EPSG:4326"
        except Exception:
            geometry = None
    input_role = props.get("input_role") or props.get("role") or role
    record = {
        "scene_id": str(scene_id) if scene_id is not None else None,
        "source_collection": (
            props.get("source_collection") or props.get("collection_id")
            or props.get("collection") or "LANDSAT/LC08/C02/T1_L2"
        ),
        "landsat_product_id": (
            props.get("landsat_product_id") or props.get("product_id")
            or props.get("LANDSAT_PRODUCT_ID")
        ),
        "spacecraft_id": props.get("spacecraft_id", props.get("SPACECRAFT_ID")),
        "sensor_id": props.get("sensor_id", props.get("SENSOR_ID")),
        "wrs_path": int(wrs_path) if wrs_path not in (None, "") else None,
        "wrs_row": int(wrs_row) if wrs_row not in (None, "") else None,
        "path_row": (
            f"{int(wrs_path):03d}_{int(wrs_row):03d}"
            if wrs_path not in (None, "") and wrs_row not in (None, "") else None
        ),
        "acquisition_datetime": acquired,
        "acquisition_date": str(acquired)[:10] if acquired is not None else None,
        "cloud_cover": props.get("cloud_cover", props.get("CLOUD_COVER")),
        "cloud_cover_land": props.get("cloud_cover_land", props.get("CLOUD_COVER_LAND")),
        "processing_level": props.get("processing_level", props.get("PROCESSING_LEVEL")),
        "collection_category": props.get("collection_category", props.get("COLLECTION_CATEGORY")),
        "collection_number": props.get("collection_number", props.get("COLLECTION_NUMBER")),
        "scene_footprint_wkt": _geometry_wkt(geometry),
        "scene_footprint_crs": str(footprint_crs),
        "source_metadata_method": props.get(
            "source_metadata_method", f"local_metadata:{source.name}",
        ),
        "input_role": input_role,
        "role": input_role,
        "baseline_year": props.get("baseline_year"),
        "metadata_source": str(source),
        "_geometry": geometry,
        "_metadata_source": str(source),
    }
    return record


class ArtifactLineageProvider(ABC):
    """Layout adapter. Selection depends on context layout, never AOI identity."""

    def __init__(self, ctx: dict[str, Any] | None = None):
        self.ctx = ctx or {}

    @abstractmethod
    def metadata_paths(self) -> list[Path]: ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    def resolve_source_metadata(self) -> list[Path]:
        return [path for path in self.metadata_paths() if path.exists()]

    def discover_artifacts(
        self, products: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return resolve_product_registry_v2(self.ctx, products)

    def resolve_reference_grid(self) -> Path | None:
        candidate = Path(self.ctx.get("step8a_output_dir", "")) / "step8a_500m_grid_valid_mask.tif"
        return candidate if candidate.exists() else None


class NamespacedExperimentProvider(ArtifactLineageProvider):
    @property
    def provider_name(self) -> str:
        return "namespaced_experiment"

    def metadata_paths(self) -> list[Path]:
        root = Path(self.ctx["output_root"])
        data = Path(self.ctx["data_root"])
        return [
            root / "source_scene_metadata.json",
            root / "predictor_export_metadata.json",
            data / "source_scene_metadata.json",
        ]


class LegacyExperimentProvider(ArtifactLineageProvider):
    @property
    def provider_name(self) -> str:
        return "legacy_shared_layout"

    def metadata_paths(self) -> list[Path]:
        root = Path(self.ctx["output_root"])
        paths = [root / "source_scene_metadata.json"]
        step4 = self.ctx.get("step4_metadata_path")
        if step4:
            step4_path = Path(step4)
            paths.extend([step4_path, step4_path.with_name("step4_export_metadata.json")])
        return paths


def provider_for_context(ctx: dict[str, Any]) -> ArtifactLineageProvider:
    root = Path(ctx["output_root"])
    data = Path(ctx["data_root"])
    is_namespaced = "experiments" in root.parts and data.is_relative_to(root)
    provider = NamespacedExperimentProvider if is_namespaced else LegacyExperimentProvider
    return provider(ctx)


def collect_scene_manifest(
    provider: ArtifactLineageProvider,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    inputs: list[str] = []
    experiment_id = getattr(provider, "ctx", {}).get("experiment_id")
    for path in provider.metadata_paths():
        if not path.exists():
            continue
        inputs.append(str(path))
        for raw, role in _scene_records(_read_json(path), None):
            record = normalize_scene(raw, role, path)
            record["experiment_id"] = experiment_id
            records.append(record)
    records.sort(key=lambda row: (
        str(row.get("scene_id")), str(row.get("input_role")),
        str(row.get("acquisition_datetime")),
    ))
    identities = sorted({
        row.get("scene_id") or _hash(
            "anonymous_", {key: value for key, value in row.items() if not key.startswith("_")},
        )
        for row in records
    })
    codes = {identity: index + 1 for index, identity in enumerate(identities)}
    for record in records:
        identity = record.get("scene_id") or _hash(
            "anonymous_", {key: value for key, value in record.items() if not key.startswith("_")},
        )
        record["scene_code"] = codes[identity]
    frame = pd.DataFrame(
        [{key: row.get(key) for key in SCENE_COLUMNS} for row in records],
        columns=SCENE_COLUMNS,
    )
    return frame, records, inputs


def footprint_collection(records: list[dict[str, Any]]) -> dict[str, Any]:
    features = []
    grouped: dict[tuple[Any, str], list[dict[str, Any]]] = {}
    for row in records:
        geometry = row.get("_geometry")
        if geometry is None:
            continue
        geometry_key = json.dumps(geometry, sort_keys=True, separators=(",", ":"))
        grouped.setdefault((row.get("scene_code"), geometry_key), []).append(row)
    for (_, _), rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        row = rows[0]
        features.append({
            "type": "Feature",
            "geometry": row["_geometry"],
            "properties": {
                "scene_code": row.get("scene_code"),
                "scene_id": row.get("scene_id"),
                "input_roles": sorted({
                    item.get("input_role") for item in rows
                    if item.get("input_role") is not None
                }),
                "path_row": row.get("path_row"),
                "acquisition_date": row.get("acquisition_date"),
                "source_collection": row.get("source_collection"),
            },
        })
    return {
        "type": "FeatureCollection",
        "name": "real_source_scene_footprints",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": features,
    }


def _segments(geometry: dict[str, Any] | None) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    result = []
    for ring in _rings(geometry):
        points = [(float(point[0]), float(point[1])) for point in ring]
        for left, right in zip(points, points[1:]):
            if left != right:
                result.append((left, right))
    return result


def _cross(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[1] - left[1] * right[0]


def _shared_segment(
    first: tuple[tuple[float, float], tuple[float, float]],
    second: tuple[tuple[float, float], tuple[float, float]],
    tolerance: float = 1e-9,
) -> list[list[float]] | None:
    p0, p1 = first
    q0, q1 = second
    vector = (p1[0] - p0[0], p1[1] - p0[1])
    other = (q1[0] - q0[0], q1[1] - q0[1])
    length2 = vector[0] ** 2 + vector[1] ** 2
    if length2 <= tolerance:
        return None
    scale = max(1.0, math.sqrt(length2), math.hypot(*other))
    if abs(_cross(vector, other)) > tolerance * scale:
        return None
    if abs(_cross(vector, (q0[0] - p0[0], q0[1] - p0[1]))) > tolerance * scale:
        return None
    positions = [
        ((q0[0] - p0[0]) * vector[0] + (q0[1] - p0[1]) * vector[1]) / length2,
        ((q1[0] - p0[0]) * vector[0] + (q1[1] - p0[1]) * vector[1]) / length2,
    ]
    lower = max(0.0, min(positions))
    upper = min(1.0, max(positions))
    if upper - lower <= tolerance:
        return None
    points = [
        [p0[0] + lower * vector[0], p0[1] + lower * vector[1]],
        [p0[0] + upper * vector[0], p0[1] + upper * vector[1]],
    ]
    return points if tuple(points[0]) <= tuple(points[1]) else list(reversed(points))


def scene_boundaries(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Create only verified shared polygon-edge segments from real footprints."""
    features: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, left in enumerate(records):
        if left.get("_geometry") is None:
            continue
        for right in records[index + 1:]:
            if right.get("_geometry") is None or (left.get("scene_code") == right.get("scene_code") and left.get("path_row") == right.get("path_row") and left.get("acquisition_date") == right.get("acquisition_date")):
                continue
            for first in _segments(left["_geometry"]):
                for second in _segments(right["_geometry"]):
                    coords = _shared_segment(first, second)
                    if coords is None:
                        continue
                    path_differs = (
                        left.get("path_row") is not None and right.get("path_row") is not None
                        and left.get("path_row") != right.get("path_row")
                    )
                    date_differs = left.get("acquisition_date") != right.get("acquisition_date")
                    boundary_type = (
                        "path_row_boundary" if path_differs
                        else "acquisition_support_boundary" if date_differs
                        else "scene_coverage_boundary"
                    )
                    scene_codes = sorted([left["scene_code"], right["scene_code"]])
                    roles = sorted({
                        value for value in (left.get("input_role"), right.get("input_role"))
                        if value is not None
                    })
                    geometry_hash = hashlib.sha256(
                        json.dumps(coords, separators=(",", ":")).encode(),
                    ).hexdigest()
                    lineage_id = _hash("lin_", {
                        "type": boundary_type, "scene_codes": scene_codes,
                        "roles": roles,
                    })
                    boundary_id = _hash("bnd_", {
                        "lineage_id": lineage_id, "geometry_hash": geometry_hash,
                    }, 24)
                    if boundary_id in seen:
                        continue
                    seen.add(boundary_id)
                    dx = coords[-1][0] - coords[0][0]
                    dy = coords[-1][1] - coords[0][1]
                    properties = {
                        "boundary_id": boundary_id,
                        "source_boundary_id": boundary_id,
                        "lineage_id": lineage_id,
                        "geometry_hash": geometry_hash,
                        "boundary_type": boundary_type,
                        "boundary_source": "source_scene_provenance",
                        "provider": "source_scene_provenance",
                        "source_product_role": roles,
                        "left_support": {
                            "scene_code": left["scene_code"], "scene_id": left.get("scene_id"),
                            "path_row": left.get("path_row"),
                            "acquisition_date": left.get("acquisition_date"),
                        },
                        "right_support": {
                            "scene_code": right["scene_code"], "scene_id": right.get("scene_id"),
                            "path_row": right.get("path_row"),
                            "acquisition_date": right.get("acquisition_date"),
                        },
                        "orientation": (
                            "vertical" if abs(dy) > abs(dx)
                            else "horizontal" if abs(dx) > abs(dy) else "oblique"
                        ),
                        "native_crs": "EPSG:4326",
                        "verification_status": "verified",
                        "derivation": "exact_shared_source_footprint_edge",
                    }
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": coords},
                        "properties": properties,
                    })
    features.sort(key=lambda feature: feature["properties"]["boundary_id"])
    return {
        "type": "FeatureCollection",
        "name": "verified_source_support_boundaries",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": features,
    }


def _stage_contract(product: dict[str, Any]) -> tuple[str, int]:
    key = product["product_key"]
    if product.get("collection_family"):
        return "yearly_source_composite", 10
    if key in {"current_lst", "current_ndvi"}:
        return "current_source_composite", 10
    if key in {"baseline_lst_mean", "baseline_lst_std", "baseline_ndvi_mean", "baseline_ndvi_std"}:
        return "baseline_aggregate", 20
    if key in {"lst_anomaly", "anomaly_zscore"}:
        return "derived_anomaly", 30
    if key in {"current_tvdi", "tvdi_difference"}:
        return "derived_tvdi", 31
    if key == "downscaled_lst":
        return "step7_downscaled", 40
    if key == "fused_lst":
        return "step7_fused", 41
    return product.get("source_stage", "predictors"), 15


def build_artifact_lineage(
    ctx: dict[str, Any], config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    products, resolutions = resolve_product_registry_v2(
        ctx, config.get("artifact_products") or config.get("artifact_families"),
    )
    resolution_by_key = {row["product_key"]: row for row in resolutions}
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def collection_node(role: str) -> str:
        node_id = f"collection:{role}"
        nodes.setdefault(node_id, {
            "node_id": node_id,
            "artifact_id": node_id, "product_key": role, "stage": "source_scene_support",
            "producer": "source_collection", "semantic_identity": role, "path": None,
            "grid_signature": None, "parent_artifact_ids": [],
            "lineage_available": False, "artifact_order": 0,
        })
        return node_id

    for product in products:
        key = product["product_key"]
        artifact_id = f"artifact:{key}"
        stage, order = _stage_contract(product)
        parents = [f"artifact:{parent}" for parent in product.get("derived_from", [])]
        parents.extend(collection_node(role) for role in product.get("export_families", []))
        resolution = resolution_by_key.get(key, {})
        nodes[artifact_id] = {
            "node_id": artifact_id,
            "artifact_id": artifact_id,
            "product_key": key,
            "stage": stage,
            "producer": product.get("source_stage"),
            "semantic_identity": product.get("semantic_identity"),
            "path": str(product["path"]) if product.get("path") else None,
            "grid_signature": resolution.get("grid_signature"),
            "parent_artifact_ids": sorted(set(parents)),
            "lineage_available": bool(parents),
            "artifact_order": order,
            "resolution_method": product.get("resolution_method"),
            "resolution_status": product.get("resolution_status"),
            "semantic_name_mismatch": False,
        }
        for parent in sorted(set(parents)):
            edges.append({
                "parent_artifact_id": parent,
                "child_artifact_id": artifact_id,
                "transformation": "source_collection" if parent.startswith("collection:") else "derived_from",
                "aggregation": "temporal_composite" if parent.startswith("collection:") else None,
                "resampling": None,
                "masking": "producer_defined",
                "composite_method": None,
            })
        feature = product.get("modeling_feature")
        if feature:
            feature_id = f"modeling_feature:{feature}"
            source_key = product.get("modeling_feature_source_product") or key
            mismatch = bool(key == "lst_anomaly" and source_key == "anomaly_zscore")
            nodes[feature_id] = {
                "node_id": feature_id,
                "artifact_id": feature_id,
                "product_key": feature,
                "stage": "step8a_500m_feature",
                "producer": "step8a",
                "semantic_identity": (
                    product.get("modeling_feature_semantic_identity")
                    or product.get("semantic_identity")
                ),
                "path": str(product.get("modeling_feature_source_path") or "") or None,
                "grid_signature": None,
                "parent_artifact_ids": [f"artifact:{source_key}"],
                "lineage_available": True,
                "artifact_order": 50,
                "resolution_method": product.get("modeling_feature_resolution_method"),
                "resolution_status": (
                    "resolved" if product.get("modeling_feature_available") else "missing"
                ),
                "semantic_name_mismatch": mismatch,
                "semantic_mismatch": mismatch,
                "legacy_column_name": feature if mismatch else None,
                "source_product": source_key,
            }
            edges.append({
                "parent_artifact_id": f"artifact:{source_key}",
                "child_artifact_id": feature_id,
                "transformation": "aggregate_to_canonical_500m",
                "aggregation": "cell_statistics",
                "resampling": "explicit_georeferenced_cell_mapping",
                "masking": "step8a_predictor_validity",
                "composite_method": None,
                "semantic_name_mismatch": mismatch,
                "semantic_mismatch": mismatch,
            })

    for edge in edges:
        parent = edge["parent_artifact_id"]
        if parent not in nodes:
            key = parent.split(":", 1)[1]
            nodes[parent] = {
                "node_id": parent,
                "artifact_id": parent, "product_key": key,
                "stage": "upstream_artifact_unavailable", "producer": None,
                "semantic_identity": key, "path": None, "grid_signature": None,
                "parent_artifact_ids": [], "lineage_available": False,
                "artifact_order": 5, "resolution_status": "missing",
            }
    node_rows = sorted(nodes.values(), key=lambda row: (
        row["artifact_order"], row["artifact_id"],
    ))
    graph = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": ctx["experiment_id"],
        "stage_order_source": "producer_registry_and_lineage_edges",
        "nodes": node_rows,
        "edges": edges,
        "resolutions": resolutions,
    }
    return pd.DataFrame(edges), graph


def _composite_contract(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for role, spec in config["composites"].items():
        method = spec["composite_method"]
        selected_allowed = method in _SELECTING_COMPOSITES
        rows.append({
            "source_product_role": role,
            **spec,
            "selected_scene_id_semantically_valid": selected_allowed,
            "dominant_scene_id_definition": (
                "scene with the largest valid-observation contribution count; "
                "it is not necessarily the source of the reducer output value"
                if method in _REDUCER_COMPOSITES else
                "selected scene under the declared per-pixel selection rule"
            ),
        })
    return rows


def _artifact_scene_lineage(
    manifest: pd.DataFrame, graph: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "scene_code", "scene_id", "input_role", "artifact_id",
        "relationship", "lineage_available",
    ]
    if manifest.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    collection_children: dict[str, list[str]] = {}
    for edge in graph["edges"]:
        if edge["parent_artifact_id"].startswith("collection:"):
            role = edge["parent_artifact_id"].split(":", 1)[1]
            collection_children.setdefault(role, []).append(edge["child_artifact_id"])
    for scene in manifest.to_dict("records"):
        role = str(scene.get("input_role") or "")
        normalized_role = role.removesuffix("_yearly")
        children = {
            artifact_id
            for collection_role, artifact_ids in collection_children.items()
            if collection_role == role
            or collection_role == normalized_role
            or collection_role.startswith(normalized_role + "_")
            for artifact_id in artifact_ids
        }
        for artifact_id in sorted(children):
            rows.append({
                "scene_code": scene["scene_code"], "scene_id": scene["scene_id"],
                "input_role": role, "artifact_id": artifact_id,
                "relationship": "contributes_to", "lineage_available": True,
            })
    return pd.DataFrame(rows, columns=columns)


def build_provenance(ctx: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    provider = provider_for_context(ctx)
    manifest, raw_records, metadata_inputs = collect_scene_manifest(provider)
    footprints = footprint_collection(raw_records)
    boundaries = scene_boundaries(raw_records)
    lineage_edges, graph = build_artifact_lineage(ctx, config)
    artifact_scene = _artifact_scene_lineage(manifest, graph)
    composites = _composite_contract(config)
    mode = config.get("mode", "metadata_only")
    selected_products = [
        "selected_scene_id"
        for row in composites if row["selected_scene_id_semantically_valid"]
    ]
    export_plan = {
        "status": "planned_not_submitted" if mode == "pixel_provenance" else "not_requested",
        "mode": mode,
        "products": config["pixel_provenance_products"] + selected_products,
        "gee_submission_started": False,
        "requires_explicit_user_command": True,
        "command": (
            f"python scripts/run_source_scene_provenance.py --experiment "
            f"{ctx['experiment_id']} --mode pixel_provenance --export-plan-only"
        ),
        "median_mean_selected_scene_id_forbidden": not selected_products,
    }
    if len(boundaries["features"]):
        status = "available"
    elif len(manifest):
        status = "metadata_available_boundary_incomplete"
    else:
        status = "insufficient_boundary_metadata"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "experiment_id": ctx["experiment_id"],
        "provider": provider.provider_name,
        "mode": mode,
        "status": status,
        "scene_count": len(manifest),
        "footprint_count": len(footprints["features"]),
        "boundary_count": len(boundaries["features"]),
        "metadata_inputs": metadata_inputs,
        "boundary_type_contract": config["boundary_types"],
        "missing_evidence": (
            [] if len(boundaries["features"]) else [
                "source scene IDs plus real intersecting/shared footprints or pixel support provenance",
            ]
        ),
        "composites": composites,
        "pixel_provenance_export_plan": export_plan,
        "assessment_complete": status == "available",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return {
        "scene_manifest": manifest,
        "footprints": footprints,
        "boundaries": boundaries,
        "lineage": lineage_edges,
        "artifact_scene_lineage": artifact_scene,
        "graph": graph,
        "summary": summary,
    }


def _write_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Source-scene provenance", "",
        f"- Experiment: `{summary['experiment_id']}`",
        f"- Status: **{summary['status']}**",
        f"- Scenes: {summary['scene_count']}",
        f"- Real footprints: {summary['footprint_count']}",
        f"- Verified support-boundary segments: {summary['boundary_count']}",
        f"- Provider: `{summary['provider']}`",
        "",
        "Median/mean composites do not expose a scientifically valid selected scene ID. "
        "Dominant-scene products, when explicitly exported, describe support frequency only.",
    ]
    if summary["missing_evidence"]:
        lines.extend(["", "## Missing evidence", ""])
        lines.extend(f"- {item}" for item in summary["missing_evidence"])
    return "\n".join(lines) + "\n"


def write_provenance(
    result: dict[str, Any], output_dir: Path, *, force: bool,
) -> dict[str, Any]:
    names = [
        "scene_manifest.parquet", "scene_manifest.json",
        "scene_footprints.geojson", "scene_boundaries.geojson",
        "provenance_summary.json", "provenance_summary.md",
        "artifact_scene_lineage.parquet", "artifact_lineage.json",
        "artifact_lineage_nodes.parquet", "artifact_lineage_edges.parquet",
        "pixel_provenance_export_plan.json", "manifest.json",
    ]
    paths = {name: output_dir / name for name in names}
    if not force and any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Provenance output exists; use --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result["scene_manifest"].to_parquet(paths["scene_manifest.parquet"], index=False)
    paths["scene_manifest.json"].write_text(
        result["scene_manifest"].to_json(orient="records", indent=2),
        encoding="utf-8",
    )
    paths["scene_footprints.geojson"].write_text(
        json.dumps(result["footprints"], indent=2), encoding="utf-8",
    )
    paths["scene_boundaries.geojson"].write_text(
        json.dumps(result["boundaries"], indent=2), encoding="utf-8",
    )
    paths["provenance_summary.json"].write_text(
        json.dumps(result["summary"], indent=2), encoding="utf-8",
    )
    paths["provenance_summary.md"].write_text(
        _write_summary_markdown(result["summary"]), encoding="utf-8",
    )
    result["artifact_scene_lineage"].to_parquet(
        paths["artifact_scene_lineage.parquet"], index=False,
    )
    paths["artifact_lineage.json"].write_text(
        json.dumps(result["graph"], indent=2), encoding="utf-8",
    )
    pd.DataFrame(result["graph"]["nodes"]).to_parquet(
        paths["artifact_lineage_nodes.parquet"], index=False,
    )
    pd.DataFrame(result["graph"]["edges"]).to_parquet(
        paths["artifact_lineage_edges.parquet"], index=False,
    )
    paths["pixel_provenance_export_plan.json"].write_text(
        json.dumps(result["summary"]["pixel_provenance_export_plan"], indent=2),
        encoding="utf-8",
    )
    payload_files = [paths[name] for name in names if name != "manifest.json"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": result["summary"]["experiment_id"],
        "read_only": True,
        "files": [
            {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in payload_files
        ],
    }
    paths["manifest.json"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "ran": True,
        "output_dir": str(output_dir),
        "summary": result["summary"],
        "files": [str(paths[name]) for name in names],
    }

