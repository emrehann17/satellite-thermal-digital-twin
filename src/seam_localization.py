"""Read-only, lineage-aware earliest-stage seam localization V1."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform_bounds, transform_geom
from rasterio.windows import Window, from_bounds

from core.seam_audit_v2_config import resolve_product_registry_v2
from src.seam_audit_v2 import (
    BoundaryRecord, INCOMPLETE, canonical_grid_info, classify_continuous,
    pair_metrics, thresholds_for,
)


DETECTED = {"warn", "fail"}
EVALUABLE = {"pass", "warn", "fail"}
MISSING = {
    "insufficient_artifact", "insufficient_boundary_metadata",
    "insufficient_control_pairs", "insufficient_valid_pairs",
    "insufficient_grid_metadata", "insufficient_lineage_match", "grid_mismatch",
}


def _digest(prefix: str, value: Any, length: int = 24) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return prefix + hashlib.sha256(raw.encode()).hexdigest()[:length]


def _geojson_crs(collection: dict[str, Any]) -> str:
    value = collection.get("crs", {}).get("properties", {}).get("name")
    if not value:
        return "EPSG:4326"
    if str(value).lower().startswith("urn:ogc:def:crs:epsg::"):
        return "EPSG:" + str(value).rsplit(":", 1)[-1]
    return str(value)


def load_boundaries(
    paths: list[Path], boundary_source: str = "source_scene_provenance",
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            collection = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        native_crs = _geojson_crs(collection)
        for feature in collection.get("features", []):
            geometry = feature.get("geometry", {})
            if geometry.get("type") != "LineString" or len(geometry.get("coordinates", [])) < 2:
                continue
            properties = dict(feature.get("properties", {}))
            coordinates = geometry["coordinates"]
            is_manual = boundary_source in {"manual", "manual_diagnostic"}
            properties.setdefault(
                "boundary_source", "manual_diagnostic" if is_manual else boundary_source,
            )
            properties.setdefault(
                "provider", "manual_diagnostic" if is_manual else boundary_source,
            )
            properties.setdefault(
                "boundary_type", "manual_diagnostic" if is_manual else "scene_coverage_boundary",
            )
            properties.setdefault("lineage_id", _digest("lin_", {
                "provider": properties["provider"],
                "type": properties["boundary_type"],
                "coordinates": coordinates,
            }, 20))
            properties.setdefault("boundary_id", _digest("bnd_", {
                "lineage": properties["lineage_id"],
                "coordinates": coordinates,
            }))
            properties.setdefault("source_boundary_id", properties["boundary_id"])
            properties.setdefault("native_crs", native_crs)
            properties.setdefault(
                "verification_status", "diagnostic" if is_manual else "verified",
            )
            properties["geometry"] = geometry
            properties["metadata_source"] = str(path)
            result.append(properties)
    unique = {item["boundary_id"]: item for item in result}
    return [unique[key] for key in sorted(unique)]


def manual_boundary_feature(
    coordinates: Iterable[Iterable[float]], crs: str = "EPSG:4326",
) -> dict[str, Any]:
    points = [[float(point[0]), float(point[1])] for point in coordinates]
    if len(points) < 2:
        raise ValueError("A manual boundary requires at least two coordinates")
    lineage_id = _digest("lin_", {
        "provider": "manual_diagnostic", "coordinates": points,
    }, 20)
    boundary_id = _digest("bnd_", {"lineage_id": lineage_id, "coordinates": points})
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": points},
            "properties": {
                "boundary_id": boundary_id,
                "source_boundary_id": boundary_id,
                "lineage_id": lineage_id,
                "boundary_type": "manual_diagnostic",
                "boundary_source": "manual_diagnostic",
                "provider": "manual_diagnostic",
                "native_crs": crs,
                "verification_status": "diagnostic",
            },
        }],
    }


def inline_manual_boundaries(
    collections: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    result = []
    for collection in collections or []:
        for feature in collection.get("features", []):
            properties = dict(feature.get("properties", {}))
            geometry = feature.get("geometry", {})
            if geometry.get("type") != "LineString":
                continue
            properties["geometry"] = geometry
            properties["metadata_source"] = "inline_cli_coordinates"
            properties.setdefault("native_crs", _geojson_crs(collection))
            properties.setdefault("boundary_source", "manual_diagnostic")
            properties.setdefault("boundary_type", "manual_diagnostic")
            properties.setdefault("provider", "manual_diagnostic")
            properties.setdefault("verification_status", "diagnostic")
            result.append(properties)
    return result


def _coordinates_for_crs(item: dict[str, Any], target_crs: Any) -> list[tuple[float, float]]:
    geometry = item["geometry"]
    source_crs = item.get("native_crs", "EPSG:4326")
    if str(source_crs) != str(target_crs):
        geometry = transform_geom(source_crs, target_crs, geometry)
    return [(float(x), float(y)) for x, y in geometry["coordinates"]]


def boundary_for_raster(
    item: dict[str, Any], src: rasterio.DatasetReader,
) -> BoundaryRecord:
    """Compatibility record; arbitrary lines are measured by pixel-pair sampling."""
    coordinates = _coordinates_for_crs(item, src.crs)
    pixels = [(~src.transform) * point for point in coordinates]
    dc = pixels[-1][0] - pixels[0][0]
    dr = pixels[-1][1] - pixels[0][1]
    orientation = (
        item.get("orientation")
        or ("vertical" if abs(dr) > abs(dc) else "horizontal" if abs(dc) > abs(dr) else "oblique")
    )
    index = start = end = None
    if orientation == "vertical":
        index = int(round(float(np.mean([point[0] for point in pixels]))))
        start = max(0, int(np.floor(min(point[1] for point in pixels))))
        end = min(src.height, int(np.ceil(max(point[1] for point in pixels))))
    elif orientation == "horizontal":
        index = int(round(float(np.mean([point[1] for point in pixels]))))
        start = max(0, int(np.floor(min(point[0] for point in pixels))))
        end = min(src.width, int(np.ceil(max(point[0] for point in pixels))))
    normalized = [[round(x, 10), round(y, 10)] for x, y in coordinates]
    geometry_hash = hashlib.sha256(
        json.dumps(normalized, separators=(",", ":")).encode(),
    ).hexdigest()
    return BoundaryRecord(
        boundary_id=item["boundary_id"],
        lineage_id=item["lineage_id"],
        boundary_type=item.get("boundary_type", "scene_coverage_boundary"),
        provider=item.get("provider", "source_scene_provenance"),
        source_product=item.get("source_product", "source_scene_collection"),
        source_artifact=item.get("source_artifact", item["metadata_source"]),
        metadata_source=item["metadata_source"],
        orientation=orientation,
        geometry_wkt="LINESTRING (" + ", ".join(
            f"{x:.12g} {y:.12g}" for x, y in coordinates
        ) + ")",
        geometry_hash=geometry_hash,
        native_crs=str(src.crs),
        verification_status=item.get("verification_status", "verified"),
        index=index,
        start=start,
        end=end,
    )


def _line_cells(
    coordinates: list[tuple[float, float]], transform: Any,
    width: int, height: int, center_offset: float = 0.0,
) -> list[tuple[int, int]]:
    inverse = ~transform
    pixels = [inverse * point for point in coordinates]
    cells: list[tuple[int, int]] = []
    for start, end in zip(pixels, pixels[1:]):
        dc, dr = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dc, dr)
        if length <= 1e-12:
            continue
        nc, nr = -dr / length, dc / length
        steps = max(1, int(math.ceil(max(abs(dc), abs(dr)) * 2)))
        for position in np.linspace(0.0, 1.0, steps + 1):
            col = start[0] + position * dc + center_offset * nc
            row = start[1] + position * dr + center_offset * nr
            cell = (int(math.floor(row)), int(math.floor(col)))
            if 0 <= cell[0] < height and 0 <= cell[1] < width:
                cells.append(cell)
    return list(dict.fromkeys(cells))


def _pixel_pairs(
    coordinates: list[tuple[float, float]], transform: Any,
    width: int, height: int, distance: int,
    center_offset: float = 0.0, max_pairs: int = 200000,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    inverse = ~transform
    pixels = [inverse * point for point in coordinates]
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for start, end in zip(pixels, pixels[1:]):
        dc, dr = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dc, dr)
        if length <= 1e-12:
            continue
        nc, nr = -dr / length, dc / length
        steps = max(1, int(math.ceil(max(abs(dc), abs(dr)) * 2)))
        for position in np.linspace(0.0, 1.0, steps + 1):
            col = start[0] + position * dc + center_offset * nc
            row = start[1] + position * dr + center_offset * nr
            left = (
                int(math.floor(row - distance * nr)),
                int(math.floor(col - distance * nc)),
            )
            right = (
                int(math.floor(row + distance * nr)),
                int(math.floor(col + distance * nc)),
            )
            if left == right:
                continue
            if all(
                0 <= cell[0] < height and 0 <= cell[1] < width
                for cell in (left, right)
            ):
                pairs.append((left, right))
    pairs = list(dict.fromkeys(pairs))
    if len(pairs) > max_pairs:
        indexes = np.linspace(0, len(pairs) - 1, max_pairs, dtype=int)
        pairs = [pairs[index] for index in indexes]
    return pairs


def _sample_pairs(
    src: rasterio.DatasetReader,
    pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    band: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def sample(cells: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
        coordinates = [xy(src.transform, row, col, offset="center") for row, col in cells]
        values: list[float] = []
        valid: list[bool] = []
        for item in src.sample(coordinates, indexes=band, masked=True):
            value = item[0] if np.ndim(item) else item
            masked = bool(np.ma.getmaskarray(item).reshape(-1)[0])
            number = float(value) if not masked else math.nan
            values.append(number)
            valid.append(not masked and math.isfinite(number))
        return np.asarray(values), np.asarray(valid, dtype=bool)

    left, left_valid = sample([pair[0] for pair in pairs])
    right, right_valid = sample([pair[1] for pair in pairs])
    return left, right, left_valid, right_valid


def _control_offsets(
    item: dict[str, Any], coordinates: list[tuple[float, float]],
    transform: Any, width: int, height: int,
    all_boundary_cells: set[tuple[int, int]], config: dict[str, Any],
) -> list[int]:
    accepted = []
    buffer_pixels = int(config["boundary_buffer_pixels"])
    blocked = {
        (row + row_offset, col + col_offset)
        for row, col in all_boundary_cells
        for row_offset in range(-buffer_pixels, buffer_pixels + 1)
        for col_offset in range(-buffer_pixels, buffer_pixels + 1)
    }
    for magnitude in config["local_control_offsets"]:
        for offset in (-int(magnitude), int(magnitude)):
            cells = _line_cells(coordinates, transform, width, height, offset)
            if not cells:
                continue
            if not any(cell in blocked for cell in cells):
                accepted.append(offset)
    return accepted


def _measure_pairs(
    pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    reader: Any, product: dict[str, Any], config: dict[str, Any],
    control_pairs: list[tuple[int, list[tuple[tuple[int, int], tuple[int, int]]]]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    left, right, left_valid, right_valid = reader(pairs)
    thresholds = thresholds_for(product, config)
    metrics = pair_metrics(
        left, right, left_valid, right_valid, thresholds["large_jump_absolute"],
    )
    valid = left_valid & right_valid
    signed = right[valid] - left[valid]
    controls: list[dict[str, Any]] = []
    control_jumps: list[np.ndarray] = []
    for offset, pairs_for_control in control_pairs:
        ca, cb, cav, cbv = reader(pairs_for_control)
        control_metrics = pair_metrics(
            ca, cb, cav, cbv, thresholds["large_jump_absolute"],
        )
        both = cav & cbv
        if both.any():
            control_jumps.append(np.abs(cb[both] - ca[both]))
        controls.append({"offset_distance": offset, **control_metrics})
    combined = np.concatenate(control_jumps) if control_jumps else np.array([], dtype=float)
    control_status = (
        "available"
        if len(combined) >= int(config["minimum_valid_pairs"])
        else "insufficient_control_pairs"
    )
    control_median = float(np.median(combined)) if len(combined) else None
    control_p95 = float(np.percentile(combined, 95)) if len(combined) else None
    absolute_median = metrics["absolute_jump_median"]
    absolute_p95 = metrics["absolute_jump_p95"]

    def ratio(value: float | None, control: float | None) -> float | None:
        if value is None or control is None:
            return None
        if abs(control) <= 1e-12:
            return 1.0 if abs(value) <= 1e-12 else 1e12
        return float(value / control)

    percentile = (
        float(100.0 * np.mean(combined <= absolute_median))
        if len(combined) and absolute_median is not None else None
    )
    mad = (
        float(np.median(np.abs(combined - np.median(combined))))
        if len(combined) else None
    )
    standardized = (
        float((absolute_median - control_median) / max(1e-12, 1.4826 * mad))
        if absolute_median is not None and control_median is not None and mad is not None
        else None
    )
    row = {
        **metrics,
        "signed_jump_mean": float(np.mean(signed)) if len(signed) else None,
        "signed_jump_median": float(np.median(signed)) if len(signed) else None,
        "control_status": control_status,
        "control_absolute_jump_median": control_median,
        "control_absolute_jump_p95": control_p95,
        "median_jump_ratio": ratio(absolute_median, control_median),
        "p95_jump_ratio": ratio(absolute_p95, control_p95),
        "boundary_percentile_against_controls": percentile,
        "standardized_boundary_effect": standardized,
    }
    row["status"] = classify_continuous(row, thresholds, config)
    profiles = [
        {
            "pair_index": index,
            "left_row": pair[0][0], "left_col": pair[0][1],
            "right_row": pair[1][0], "right_col": pair[1][1],
            "left_value": float(left[index]) if left_valid[index] else None,
            "right_value": float(right[index]) if right_valid[index] else None,
            "signed_jump": (
                float(right[index] - left[index])
                if left_valid[index] and right_valid[index] else None
            ),
            "valid_pair": bool(left_valid[index] and right_valid[index]),
        }
        for index, pair in enumerate(pairs)
    ]
    return row, controls, profiles


def classify_propagation(
    previous: dict[str, Any] | None, current: dict[str, Any],
) -> str:
    if current.get("status") in MISSING or current.get("status") in INCOMPLETE:
        return "insufficient_data"
    if previous is None:
        return "appears_at_this_stage" if current.get("status") in DETECTED else "not_detected"
    if current.get("status") == "pass":
        return "disappears" if previous.get("status") in DETECTED else "not_detected"
    if current.get("status") not in DETECTED:
        return "insufficient_data"
    if previous.get("status") not in DETECTED:
        return "appears_at_this_stage"
    if previous.get("semantic_group") != current.get("semantic_group"):
        return "persists_from_upstream"
    old = previous.get("standardized_boundary_effect")
    new = current.get("standardized_boundary_effect")
    if old is None or new is None:
        return "persists_from_upstream"
    if abs(new) > abs(old) * 1.25:
        return "amplified"
    if abs(new) < abs(old) * 0.75:
        return "attenuated"
    return "persists_from_upstream"


def localize_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (
        row.get("artifact_order", 10**9), 0 if row.get("scale") == "native" else 1,
    ))
    detections = [
        row for row in ordered
        if row.get("status") in DETECTED and row.get("corroborated_detection", True)
    ]
    available = [row for row in ordered if row.get("artifact_available", row.get("status") != "missing")]
    if not detections:
        incomplete = any(row.get("status") in MISSING | {"missing"} for row in ordered)
        return {
            "earliest_stage_status": "insufficient_evidence" if incomplete else "not_detected",
            "localization_status": "insufficient_evidence" if incomplete else "not_detected",
            "earliest_detected_artifact": None,
            "earliest_detected_stage": None,
            "earliest_artifact": None,
            "earliest_possible_artifact": None,
            "earliest_possible_stage": None,
            "latest_possible_artifact": None,
            "latest_possible_stage": None,
            "present_at_first_available_artifact": False,
            "root_cause_upstream_of_available_artifacts": False,
            "upstream_risk": False,
        }
    first = detections[0]
    first_available = available[0] if available else first
    if first.get("artifact_id") == first_available.get("artifact_id"):
        return {
            "earliest_stage_status": "present_at_first_available_artifact",
            "localization_status": "present_at_first_available_artifact",
            "earliest_detected_artifact": first.get("artifact_id"),
            "earliest_detected_stage": first.get("stage"),
            "earliest_artifact": first.get("artifact_id"),
            "earliest_possible_artifact": None,
            "earliest_possible_stage": "upstream_of_first_available_artifact",
            "latest_possible_artifact": first.get("artifact_id"),
            "latest_possible_stage": first.get("stage"),
            "present_at_first_available_artifact": True,
            "root_cause_upstream_of_available_artifacts": True,
            "upstream_risk": True,
        }
    before = [
        row for row in ordered
        if row.get("artifact_order", 0) < first.get("artifact_order", 0)
    ]
    last_pass_order = max(
        (row.get("artifact_order", -1) for row in before if row.get("status") == "pass"),
        default=-1,
    )
    unresolved = [
        row for row in before
        if row.get("artifact_order", -1) > last_pass_order
        and row.get("status") in MISSING | {"missing"}
    ]
    if unresolved:
        lower = unresolved[0]
        return {
            "earliest_stage_status": "bounded_but_not_exact",
            "localization_status": "bounded",
            "earliest_detected_artifact": None,
            "earliest_detected_stage": None,
            "earliest_artifact": None,
            "earliest_possible_artifact": lower.get("artifact_id"),
            "earliest_possible_stage": lower.get("stage"),
            "latest_possible_artifact": first.get("artifact_id"),
            "latest_possible_stage": first.get("stage"),
            "present_at_first_available_artifact": False,
            "root_cause_upstream_of_available_artifacts": False,
            "upstream_risk": False,
        }
    return {
        "earliest_stage_status": "exact",
        "localization_status": "exact",
        "earliest_detected_artifact": first.get("artifact_id"),
        "earliest_detected_stage": first.get("stage"),
        "earliest_artifact": first.get("artifact_id"),
        "earliest_possible_artifact": first.get("artifact_id"),
        "earliest_possible_stage": first.get("stage"),
        "latest_possible_artifact": first.get("artifact_id"),
        "latest_possible_stage": first.get("stage"),
        "present_at_first_available_artifact": False,
        "root_cause_upstream_of_available_artifacts": False,
        "upstream_risk": False,
    }


def visualization_check(
    path: Path, semantic_group: str, config: dict[str, Any],
) -> list[dict[str, Any]]:
    fixed = config["visualization"]["fixed_ranges"].get(
        semantic_group, config["visualization"]["fixed_ranges"]["default"],
    )
    with rasterio.open(path) as src:
        scale = min(1.0, 512 / max(src.width, src.height))
        height = max(1, int(src.height * scale))
        width = max(1, int(src.width * scale))
        values = src.read(1, out_shape=(height, width), masked=True).compressed()
        nodata = src.nodata
    robust = [
        float(np.percentile(values, percentile)) if len(values) else None
        for percentile in config["visualization"]["robust_percentiles"]
    ]
    common = {
        "artifact_path": str(path),
        "colormap": "inferno" if semantic_group == "continuous_temperature" else "viridis",
        "nodata_display": "transparent",
        "per_tile_normalization": False,
        "normalization_scope": "whole_artifact",
        "numeric_evidence": False,
        "nodata_value": nodata,
    }
    return [
        {
            **common, "stretch_method": "fixed_physical_scale",
            "vmin": fixed[0], "vmax": fixed[1],
            "visualization_artifact_suspected": False,
        },
        {
            **common, "stretch_method": "robust_global_scale",
            "vmin": robust[0], "vmax": robust[1],
            "visualization_artifact_suspected": False,
        },
    ]


def visualization_artifact_suspected(
    numeric_status: str, fixed_visible: bool,
    robust_visible: bool, per_tile_visible: bool,
) -> bool:
    return (
        numeric_status == "pass" and per_tile_visible
        and not fixed_visible and not robust_visible
    )


def _canonical_cell_value(
    src: rasterio.DatasetReader, canonical: dict[str, Any],
    row: int, col: int, band: int,
    cache: dict[tuple[int, int], tuple[float, bool]],
) -> tuple[float, bool]:
    key = (row, col)
    if key in cache:
        return cache[key]
    x0, y0 = canonical["transform"] * (col, row)
    x1, y1 = canonical["transform"] * (col + 1, row + 1)
    left, right = min(x0, x1), max(x0, x1)
    bottom, top = min(y0, y1), max(y0, y1)
    if str(src.crs) != str(canonical["crs"]):
        left, bottom, right, top = transform_bounds(
            canonical["crs"], src.crs, left, bottom, right, top,
        )
    try:
        window = from_bounds(left, bottom, right, top, src.transform)
        window = window.round_offsets().round_lengths().intersection(
            Window(0, 0, src.width, src.height),
        )
        array = src.read(band, window=window, masked=True)
        values = np.asarray(array.filled(np.nan), dtype=float)
        valid = ~np.ma.getmaskarray(array) & np.isfinite(values)
        result = (
            (float(np.mean(values[valid])), True)
            if valid.any() else (math.nan, False)
        )
    except Exception:
        result = (math.nan, False)
    cache[key] = result
    return result


def _apply_corroboration(
    metrics: list[dict[str, Any]], config: dict[str, Any],
) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in metrics:
        groups.setdefault((
            str(row.get("lineage_id")), str(row.get("artifact_id")),
            str(row.get("scale")),
        ), []).append(row)
    rules = config["decision_rules"]
    for rows in groups.values():
        evaluable = [row for row in rows if row.get("status") in EVALUABLE]
        failed = [row for row in evaluable if row["status"] == "fail"]
        detected = [row for row in evaluable if row["status"] in DETECTED]
        denominator = len(evaluable)
        fail_fraction = len(failed) / denominator if denominator else 0.0
        detected_fraction = len(detected) / denominator if denominator else 0.0
        fail_ok = (
            len(failed) >= int(rules["min_flagged_segments_fail"])
            and fail_fraction >= float(rules["min_flagged_boundary_fraction_fail"])
        )
        warn_ok = (
            len(detected) >= int(rules["min_flagged_segments_warn"])
            and detected_fraction >= float(rules["min_flagged_boundary_fraction_warn"])
        )
        group_status = "fail" if fail_ok else "warn" if warn_ok else "pass" if denominator else "incomplete"
        for row in rows:
            row["corroborated_status"] = group_status
            row["corroborated_detection"] = (
                row.get("status") in DETECTED and group_status in DETECTED
            )
            row["corroborating_segment_count"] = len(detected)
            row["evaluable_segment_count"] = denominator


def _rerun_stage(stage: str | None) -> str | None:
    if stage in {"current_source_composite", "yearly_source_composite"}:
        return "predictors"
    if stage in {"baseline_aggregate", "derived_anomaly", "derived_tvdi"}:
        return "predictors"
    if stage in {"step7_downscaled", "step7_fused"}:
        return "step7"
    if stage == "step8a_500m_feature":
        return "step8a"
    return None


def run_localization(
    ctx: dict[str, Any], config: dict[str, Any],
    manual_boundaries: list[Path] | None = None,
    manual_boundary_collections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    provenance_root = (
        Path(ctx["output_root"]) / "qa" / "source_scene_provenance" / "v1"
    )
    provenance_boundary_path = provenance_root / "scene_boundaries.geojson"
    boundaries = load_boundaries([provenance_boundary_path])
    boundaries += load_boundaries(manual_boundaries or [], "manual_diagnostic")
    boundaries += inline_manual_boundaries(manual_boundary_collections)
    boundaries = list({item["boundary_id"]: item for item in boundaries}.values())

    graph_path = provenance_root / "artifact_lineage.json"
    graph = (
        json.loads(graph_path.read_text(encoding="utf-8"))
        if graph_path.exists() else {"nodes": [], "edges": []}
    )
    node_by_product = {
        node.get("product_key"): node
        for node in graph.get("nodes", [])
        if str(node.get("artifact_id", "")).startswith("artifact:")
    }
    products, resolutions = resolve_product_registry_v2(
        ctx, config.get("artifact_families"),
    )
    fallback_order = {product["product_key"]: index + 10 for index, product in enumerate(products)}
    products.sort(key=lambda product: (
        node_by_product.get(product["product_key"], {}).get(
            "artifact_order", fallback_order[product["product_key"]],
        ),
        fallback_order[product["product_key"]],
    ))
    canonical, canonical_reason = canonical_grid_info(ctx)
    dataset_path = (
        Path(ctx["step8a_output_dir"]) / "step8a_500m_modeling_dataset.parquet"
    )
    dataset = pd.read_parquet(dataset_path) if dataset_path.exists() else None
    dataset_index = (
        dataset.set_index(["row_500m", "col_500m"])
        if dataset is not None and {"row_500m", "col_500m"} <= set(dataset.columns)
        else None
    )

    metrics: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    visualization: list[dict[str, Any]] = []

    for product_index, product in enumerate(products):
        key = product["product_key"]
        node = node_by_product.get(key, {})
        artifact_id = node.get("artifact_id", f"artifact:{key}")
        artifact_order = node.get("artifact_order", fallback_order[key])
        stage = node.get("stage", product.get("source_stage"))
        parent_ids = node.get("parent_artifact_ids", [])
        common = {
            "artifact_id": artifact_id,
            "artifact_order": artifact_order,
            "product_key": key,
            "stage": stage,
            "parent_artifact_id": parent_ids[0] if parent_ids else None,
            "parent_artifact_ids": parent_ids,
            "semantic_identity": product.get("semantic_identity"),
            "semantic_group": product.get("semantic_group"),
            "artifact_available": product.get("path") is not None,
            "resolution_method": product.get("resolution_method"),
        }
        path = product.get("path")
        if path is None:
            for boundary in boundaries:
                for scale in config["audit_scales"]:
                    metrics.append({
                        **common,
                        "boundary_id": boundary["boundary_id"],
                        "source_boundary_id": boundary["source_boundary_id"],
                        "lineage_id": boundary["lineage_id"],
                        "geometry_hash": boundary.get("geometry_hash"),
                        "boundary_type": boundary["boundary_type"],
                        "boundary_source": boundary["boundary_source"],
                        "verification_status": boundary["verification_status"],
                        "scale": scale,
                        "status": "insufficient_artifact",
                        "reason": product["resolution_status"],
                        "used_by_model": False,
                    })
            continue

        path = Path(path)
        visualization.extend({
            "artifact_id": artifact_id, "product_key": key, **row
        } for row in visualization_check(
            path, product.get("semantic_group", "default"), config,
        ))
        try:
            source = rasterio.open(path)
        except rasterio.RasterioError as exc:
            for boundary in boundaries:
                metrics.append({
                    **common,
                    "boundary_id": boundary["boundary_id"],
                    "source_boundary_id": boundary["source_boundary_id"],
                    "lineage_id": boundary["lineage_id"],
                    "boundary_type": boundary["boundary_type"],
                    "boundary_source": boundary["boundary_source"],
                    "verification_status": boundary["verification_status"],
                    "scale": "native", "status": "grid_mismatch",
                    "reason": str(exc), "used_by_model": False,
                })
            continue

        with source as src:
            transformed = {
                boundary["boundary_id"]: _coordinates_for_crs(boundary, src.crs)
                for boundary in boundaries
            }
            known_cells: set[tuple[int, int]] = set()
            for coordinates in transformed.values():
                known_cells.update(
                    _line_cells(coordinates, src.transform, src.width, src.height),
                )
            for boundary in boundaries:
                coordinates = transformed[boundary["boundary_id"]]
                native_pairs = _pixel_pairs(
                    coordinates, src.transform, src.width, src.height,
                    int(config["boundary_buffer_pixels"]),
                    max_pairs=int(config["max_boundary_pairs"]),
                )
                accepted_offsets = _control_offsets(
                    boundary, coordinates, src.transform, src.width, src.height,
                    known_cells, config,
                )
                native_controls = [
                    (
                        offset,
                        _pixel_pairs(
                            coordinates, src.transform, src.width, src.height,
                            int(config["boundary_buffer_pixels"]),
                            center_offset=offset,
                            max_pairs=int(config["max_boundary_pairs"]),
                        ),
                    )
                    for offset in accepted_offsets
                ]
                reader = lambda pairs, src=src, product=product: _sample_pairs(
                    src, pairs, int(product.get("band_index", 1)),
                )
                row, control_rows, profile_rows = _measure_pairs(
                    native_pairs, reader, product, config, native_controls,
                )
                boundary_common = {
                    **common,
                    "boundary_id": boundary["boundary_id"],
                    "source_boundary_id": boundary["source_boundary_id"],
                    "lineage_id": boundary["lineage_id"],
                    "geometry_hash": boundary.get("geometry_hash") or _digest(
                        "", coordinates, 64,
                    ),
                    "boundary_type": boundary["boundary_type"],
                    "boundary_source": boundary["boundary_source"],
                    "verification_status": boundary["verification_status"],
                    "scale": "native",
                    "native_or_modeling": "native",
                    "coordinate_transform_used": (
                        str(boundary.get("native_crs", "EPSG:4326")) != str(src.crs)
                    ),
                    "used_by_model": False,
                }
                metrics.append({**boundary_common, **row})
                controls.extend({
                    **boundary_common,
                    "control_id": _digest(
                        "ctl_", [artifact_id, boundary["boundary_id"], item["offset_distance"]],
                    ),
                    **item,
                } for item in control_rows)
                profiles.extend({
                    **boundary_common, **item
                } for item in profile_rows)

                if "modeling_500m" not in config["audit_scales"]:
                    continue
                if canonical is None:
                    metrics.append({
                        **boundary_common,
                        "artifact_order": artifact_order + 0.5,
                        "scale": "modeling_500m",
                        "native_or_modeling": "modeling_500m",
                        "status": "insufficient_grid_metadata",
                        "reason": canonical_reason,
                    })
                    continue
                canonical_coordinates = _coordinates_for_crs(
                    boundary, canonical["crs"],
                )
                modeling_pairs = _pixel_pairs(
                    canonical_coordinates, canonical["transform"],
                    canonical["width"], canonical["height"], 1,
                    max_pairs=int(config["max_boundary_pairs"]),
                )
                canonical_known: set[tuple[int, int]] = set()
                for other in boundaries:
                    canonical_known.update(_line_cells(
                        _coordinates_for_crs(other, canonical["crs"]),
                        canonical["transform"], canonical["width"], canonical["height"],
                    ))
                canonical_offsets = _control_offsets(
                    boundary, canonical_coordinates, canonical["transform"],
                    canonical["width"], canonical["height"], canonical_known,
                    {**config, "local_control_offsets": [1, 2]},
                )
                modeling_controls = [
                    (
                        offset,
                        _pixel_pairs(
                            canonical_coordinates, canonical["transform"],
                            canonical["width"], canonical["height"], 1,
                            center_offset=offset,
                            max_pairs=int(config["max_boundary_pairs"]),
                        ),
                    )
                    for offset in canonical_offsets
                ]
                feature = product.get("modeling_feature")
                cache: dict[tuple[int, int], tuple[float, bool]] = {}

                def modeling_reader(
                    pairs: list[tuple[tuple[int, int], tuple[int, int]]],
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                    if dataset_index is not None and feature and feature in dataset_index.columns:
                        left = np.asarray([
                            float(dataset_index[feature].get(pair[0], math.nan))
                            for pair in pairs
                        ])
                        right = np.asarray([
                            float(dataset_index[feature].get(pair[1], math.nan))
                            for pair in pairs
                        ])
                        return left, right, np.isfinite(left), np.isfinite(right)
                    left_values = [
                        _canonical_cell_value(
                            src, canonical, *pair[0],
                            int(product.get("band_index", 1)), cache,
                        )
                        for pair in pairs
                    ]
                    right_values = [
                        _canonical_cell_value(
                            src, canonical, *pair[1],
                            int(product.get("band_index", 1)), cache,
                        )
                        for pair in pairs
                    ]
                    left = np.asarray([item[0] for item in left_values])
                    right = np.asarray([item[0] for item in right_values])
                    return (
                        left, right,
                        np.asarray([item[1] for item in left_values], dtype=bool),
                        np.asarray([item[1] for item in right_values], dtype=bool),
                    )

                mrow, mcontrol_rows, mprofile_rows = _measure_pairs(
                    modeling_pairs, modeling_reader, product, config,
                    modeling_controls,
                )
                model_common = {
                    **boundary_common,
                    "artifact_order": artifact_order + 0.5,
                    "scale": "modeling_500m",
                    "native_or_modeling": "modeling_500m",
                    "coordinate_transform_used": (
                        str(boundary.get("native_crs", "EPSG:4326"))
                        != str(canonical["crs"])
                    ),
                    "used_by_model": bool(
                        dataset_index is not None and feature
                        and feature in dataset_index.columns
                        and product.get("modeling_feature_available")
                    ),
                }
                metrics.append({**model_common, **mrow})
                controls.extend({
                    **model_common,
                    "control_id": _digest(
                        "ctl_", [artifact_id, boundary["boundary_id"], "500m", item["offset_distance"]],
                    ),
                    **item,
                } for item in mcontrol_rows)
                profiles.extend({
                    **model_common, **item
                } for item in mprofile_rows)

    _apply_corroboration(metrics, config)
    traces: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    boundary_by_id = {boundary["boundary_id"]: boundary for boundary in boundaries}
    for boundary_id in sorted({row["boundary_id"] for row in metrics}):
        subset = sorted(
            [row for row in metrics if row["boundary_id"] == boundary_id],
            key=lambda row: (
                row["artifact_order"], 0 if row.get("scale") == "native" else 1,
            ),
        )
        previous = None
        for row in subset:
            row["propagation"] = classify_propagation(previous, row)
            if (
                previous is not None
                and previous.get("semantic_group") == row.get("semantic_group")
                and previous.get("standardized_boundary_effect") not in (None, 0)
                and row.get("standardized_boundary_effect") is not None
            ):
                row["amplification_ratio"] = float(
                    row["standardized_boundary_effect"]
                    / previous["standardized_boundary_effect"]
                )
                row["amplification_class"] = row["propagation"]
            else:
                row["amplification_ratio"] = None
                row["amplification_class"] = "not_comparable"
            if row.get("artifact_available"):
                previous = row
        localized = localize_trace(subset)
        boundary = boundary_by_id.get(boundary_id, {})
        modeling_rows = [
            row for row in subset
            if row.get("scale") == "modeling_500m"
            and row.get("status") in EVALUABLE
        ]
        modeling_status = next(
            (status for status in ("fail", "warn", "pass")
             if any(row["status"] == status for row in modeling_rows)),
            "insufficient_data",
        )
        trace = {
            "source_boundary_id": boundary_id,
            "boundary_id": boundary_id,
            "lineage_id": boundary.get("lineage_id"),
            "boundary_source": boundary.get("boundary_source"),
            "source_scene_left": boundary.get("left_support", {}).get("scene_id"),
            "source_scene_right": boundary.get("right_support", {}).get("scene_id"),
            "path_row_left": boundary.get("left_support", {}).get("path_row"),
            "path_row_right": boundary.get("right_support", {}).get("path_row"),
            **localized,
            "last_detected_artifact": next(
                (row["artifact_id"] for row in reversed(subset)
                 if row.get("status") in DETECTED), None,
            ),
            "modeling_500m_status": modeling_status,
        }
        traces.append(trace)
        for row in subset:
            trace_rows.append({
                **{key: value for key, value in trace.items() if key != "trace"},
                **{key: row.get(key) for key in (
                    "artifact_id", "artifact_order", "product_key", "stage", "scale",
                    "status", "corroborated_status", "corroborated_detection",
                    "propagation", "amplification_ratio", "amplification_class",
                )},
            })

    exact_traces = [
        trace for trace in traces if trace["earliest_stage_status"] == "exact"
    ]
    detections = [
        row for row in metrics
        if row.get("corroborated_detection") and row.get("status") in DETECTED
    ]
    verified = {
        boundary["boundary_id"]: (
            boundary.get("verification_status") == "verified"
            and boundary.get("boundary_source") != "manual_diagnostic"
        )
        for boundary in boundaries
    }
    exact_ids = {trace["boundary_id"] for trace in exact_traces}
    blocker_rows = [
        row for row in metrics
        if row["boundary_id"] in exact_ids
        and verified.get(row["boundary_id"], False)
        and row.get("scale") == "modeling_500m"
        and row.get("corroborated_detection")
        and row.get("used_by_model")
        and row.get("propagation") in {"persists_from_upstream", "amplified"}
    ]
    scientific_blocker = bool(blocker_rows)
    potential_modeling_risk = any(
        row.get("scale") == "modeling_500m"
        and row.get("corroborated_detection")
        and (not verified.get(row["boundary_id"], False) or not exact_ids)
        for row in metrics
    )

    provenance_summary_path = provenance_root / "provenance_summary.json"
    provenance_summary = (
        json.loads(provenance_summary_path.read_text(encoding="utf-8"))
        if provenance_summary_path.exists() else {}
    )
    source_status = provenance_summary.get(
        "status", "insufficient_boundary_metadata",
    )
    incomplete_rows = [
        row for row in metrics if row.get("status") in MISSING
    ]
    assessment_complete = (
        source_status == "available" and bool(boundaries) and not incomplete_rows
    )
    earliest_confirmed = min(
        exact_traces,
        key=lambda trace: next(
            (
                row["artifact_order"] for row in metrics
                if row["artifact_id"] == trace["earliest_detected_artifact"]
            ),
            10**9,
        ),
        default=None,
    )
    first_available_product = min(
        (product for product in products if product.get("path") is not None),
        key=lambda product: node_by_product.get(
            product["product_key"], {},
        ).get("artifact_order", fallback_order[product["product_key"]]),
        default=None,
    )
    first_available = (
        node_by_product.get(first_available_product["product_key"], {}).get(
            "artifact_id", f"artifact:{first_available_product['product_key']}",
        )
        if first_available_product else None
    )
    possible_stages = [
        trace["earliest_possible_stage"] for trace in traces
        if trace.get("earliest_possible_stage")
    ]
    latest_stages = [
        trace["latest_possible_stage"] for trace in traces
        if trace.get("latest_possible_stage")
    ]
    rerun = (
        _rerun_stage(earliest_confirmed["earliest_detected_stage"])
        if earliest_confirmed and scientific_blocker else None
    )
    if source_status != "available":
        recommended_action = "collect_missing_upstream_provenance"
    elif any(trace["present_at_first_available_artifact"] for trace in traces):
        recommended_action = "inspect_first_available_artifact"
    elif potential_modeling_risk:
        recommended_action = "inspect_conditional_modeling_risk"
    elif scientific_blocker:
        recommended_action = f"rerun_from_{rerun}"
    else:
        recommended_action = "no_scientific_rerun_required"
    overall_status = (
        "fail" if scientific_blocker
        else "warn" if detections or not assessment_complete
        else "pass"
    )
    propagating_ids = {
        row["boundary_id"] for row in metrics
        if row.get("scale") == "modeling_500m"
        and row.get("corroborated_detection")
    }
    summary = {
        "experiment_id": ctx["experiment_id"],
        "schema_version": "1.0",
        "overall_status": overall_status,
        "status": overall_status,
        "assessment_complete": assessment_complete,
        "source_scene_provenance_status": source_status,
        "first_available_artifact": first_available,
        "earliest_confirmed_stage": (
            earliest_confirmed["earliest_detected_stage"]
            if earliest_confirmed else None
        ),
        "earliest_possible_stage": possible_stages[0] if possible_stages else None,
        "latest_possible_stage": latest_stages[0] if latest_stages else None,
        "root_cause_upstream_of_available_artifacts": any(
            trace["root_cause_upstream_of_available_artifacts"]
            for trace in traces
        ),
        "boundaries_evaluated": len({
            row["boundary_id"] for row in metrics if row.get("status") in EVALUABLE
        }),
        "boundaries_detected": len({row["boundary_id"] for row in detections}),
        "boundaries_propagating_to_500m": len(propagating_ids),
        "scientific_blocker": scientific_blocker,
        "model_blocker": scientific_blocker,
        "potential_modeling_risk": potential_modeling_risk,
        "recommended_rerun_from_stage": rerun,
        "recommended_action": recommended_action,
        "boundary_results": traces,
        "read_only": True,
        "artifact_order_source": (
            "artifact_lineage_graph" if graph_path.exists()
            else "producer_registry_order"
        ),
        "semantic_mismatch": {
            "modeling_feature": "lst_anomaly_mean",
            "source_product": "anomaly_zscore",
            "semantic_identity": "standardized_lst_anomaly",
            "semantic_name_mismatch": True,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    candidates = [
        trace for trace in traces
        if trace["earliest_stage_status"] in {
            "exact", "bounded_but_not_exact",
            "present_at_first_available_artifact",
        }
    ]
    def frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)

    return {
        "summary": summary,
        "metrics": frame(metrics, [
            "artifact_id", "product_key", "stage", "boundary_id",
            "lineage_id", "scale", "status",
        ]),
        "controls": frame(controls, [
            "artifact_id", "boundary_id", "control_id", "offset_distance",
        ]),
        "profiles": frame(profiles, [
            "artifact_id", "boundary_id", "scale", "pair_index",
            "left_value", "right_value", "signed_jump", "valid_pair",
        ]),
        "trace_rows": frame(trace_rows, [
            "source_boundary_id", "artifact_id", "stage", "scale",
            "status", "propagation",
        ]),
        "traces": traces,
        "candidates": frame(candidates, [
            "boundary_id", "earliest_stage_status",
            "earliest_detected_artifact", "earliest_detected_stage",
        ]),
        "visualization": frame(visualization, [
            "artifact_id", "stretch_method", "vmin", "vmax",
            "per_tile_normalization",
        ]),
        "boundaries": boundaries,
        "artifact_resolution": frame(resolutions, [
            "product_key", "resolution_status", "resolved_path",
        ]),
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Earliest-stage seam localization", "",
        f"- Experiment: `{summary['experiment_id']}`",
        f"- Overall status: **{summary['overall_status']}**",
        f"- Assessment complete: {summary['assessment_complete']}",
        f"- Source-scene provenance: `{summary['source_scene_provenance_status']}`",
        f"- Earliest confirmed stage: `{summary['earliest_confirmed_stage']}`",
        f"- Earliest possible stage: `{summary['earliest_possible_stage']}`",
        f"- Latest possible stage: `{summary['latest_possible_stage']}`",
        f"- Scientific blocker: {summary['scientific_blocker']}",
        f"- Potential modeling risk: {summary['potential_modeling_risk']}",
        f"- Recommended action: `{summary['recommended_action']}`",
        "",
        "Thresholds are initial QA heuristics, not formal significance tests or causal attribution.",
    ]
    return "\n".join(lines) + "\n"


def write_localization(
    result: dict[str, Any], output_dir: Path, force: bool,
) -> dict[str, Any]:
    names = [
        "localization_summary.json", "localization_summary.md",
        "artifact_boundary_metrics.parquet", "boundary_stage_trace.parquet",
        "earliest_stage_candidates.parquet", "visualization_checks.parquet",
        "seam_profiles.parquet", "seam_hotspots.geojson",
        "matched_controls.parquet", "artifact_resolution.parquet", "manifest.json",
    ]
    paths = {name: output_dir / name for name in names}
    if not force and any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Localization output exists; use --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths["localization_summary.json"].write_text(
        json.dumps(result["summary"], indent=2), encoding="utf-8",
    )
    paths["localization_summary.md"].write_text(
        _summary_markdown(result["summary"]), encoding="utf-8",
    )
    result["metrics"].to_parquet(
        paths["artifact_boundary_metrics.parquet"], index=False,
    )
    result["trace_rows"].to_parquet(
        paths["boundary_stage_trace.parquet"], index=False,
    )
    result["candidates"].to_parquet(
        paths["earliest_stage_candidates.parquet"], index=False,
    )
    result["visualization"].to_parquet(
        paths["visualization_checks.parquet"], index=False,
    )
    result["profiles"].to_parquet(paths["seam_profiles.parquet"], index=False)
    result["controls"].to_parquet(paths["matched_controls.parquet"], index=False)
    result["artifact_resolution"].to_parquet(
        paths["artifact_resolution.parquet"], index=False,
    )
    candidate_ids = set(result["candidates"].get("boundary_id", pd.Series(dtype=str)))
    hotspots = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": boundary["geometry"],
                "properties": {
                    key: boundary.get(key) for key in (
                        "boundary_id", "source_boundary_id", "lineage_id",
                        "boundary_type", "boundary_source",
                    )
                },
            }
            for boundary in result["boundaries"]
            if boundary["boundary_id"] in candidate_ids
        ],
    }
    paths["seam_hotspots.geojson"].write_text(
        json.dumps(hotspots, indent=2), encoding="utf-8",
    )
    payload_files = [paths[name] for name in names if name != "manifest.json"]
    manifest = {
        "schema_version": "1.0",
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

