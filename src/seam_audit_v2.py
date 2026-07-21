"""Seam Audit V2 primitives.

All raster access is read-only and window bounded.  Boundaries are explicit
lineage records, controls are matched to an individual boundary, and native to
modeling propagation is joined only by the same stable ``boundary_id``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.warp import transform_bounds, transform_geom
from rasterio.windows import Window, from_bounds

from core.utils.tiling import make_tile_grid


DETECTED = {"warn", "fail"}
INCOMPLETE = {
    "insufficient_artifact", "insufficient_boundary_metadata",
    "insufficient_grid_metadata", "insufficient_lineage_match",
    "insufficient_valid_pairs", "insufficient_control_pairs", "grid_mismatch",
}


@dataclass(frozen=True)
class BoundaryRecord:
    boundary_id: str
    lineage_id: str
    boundary_type: str
    provider: str
    source_product: str
    source_artifact: str
    metadata_source: str
    orientation: str
    geometry_wkt: str
    geometry_hash: str
    native_crs: str
    verification_status: str
    index: int | None = None
    start: int | None = None
    end: int | None = None
    overlap_pixels: float | None = None
    gap_pixels: float | None = None


def _json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def _line_wkt(coords: list[tuple[float, float]]) -> str:
    return "LINESTRING (" + ", ".join(f"{x:.12g} {y:.12g}" for x, y in coords) + ")"


def _coords_from_wkt(wkt: str) -> list[tuple[float, float]]:
    body = wkt[wkt.index("(") + 1:wkt.rindex(")")]
    return [tuple(map(float, point.strip().split())) for point in body.split(",")]  # type: ignore[return-value]


def _stable_boundary(
    *, boundary_type: str, provider: str, source_product: str,
    source_artifact: str, metadata_source: str, orientation: str,
    coords: list[tuple[float, float]], crs: str, verification_status: str = "verified",
    index: int | None = None, start: int | None = None, end: int | None = None,
    overlap_pixels: float | None = None, gap_pixels: float | None = None,
) -> BoundaryRecord:
    normalized = [[round(float(x), 10), round(float(y), 10)] for x, y in coords]
    geometry_hash = hashlib.sha256(json.dumps(normalized, separators=(",", ":")).encode()).hexdigest()
    lineage_payload = f"{provider}|{boundary_type}|{source_product}|{source_artifact}"
    lineage_id = "lin_" + hashlib.sha256(lineage_payload.encode()).hexdigest()[:20]
    boundary_payload = f"{lineage_id}|{orientation}|{geometry_hash}"
    boundary_id = "bnd_" + hashlib.sha256(boundary_payload.encode()).hexdigest()[:24]
    return BoundaryRecord(
        boundary_id, lineage_id, boundary_type, provider, source_product,
        source_artifact, metadata_source, orientation, _line_wkt(coords),
        geometry_hash, crs, verification_status, index, start, end,
        overlap_pixels, gap_pixels,
    )


def boundary_row(boundary: BoundaryRecord) -> dict[str, Any]:
    return asdict(boundary)


def _same_transform(values: Iterable[float], transform: Affine) -> bool:
    vals = list(values)
    return len(vals) >= 6 and np.allclose(vals[:6], list(transform)[:6], rtol=0, atol=1e-10)


def _metadata_grid_status(meta: dict[str, Any], src: rasterio.DatasetReader) -> str:
    if meta.get("raster_shape") and list(meta["raster_shape"]) != [src.height, src.width]:
        return "grid_mismatch"
    if meta.get("crs") and str(meta["crs"]) != str(src.crs):
        return "grid_mismatch"
    if meta.get("transform") and not _same_transform(meta["transform"], src.transform):
        return "grid_mismatch"
    return "verified"


def processing_window_boundaries(
    ctx: dict[str, Any], product: dict[str, Any], src: rasterio.DatasetReader,
) -> tuple[list[BoundaryRecord], str, str | None]:
    """Resolve Step7 inference windows; Step7A is intentionally never read."""
    key = product["product_key"]
    if key == "downscaled_lst":
        meta_path = Path(ctx["step7d_output_dir"]) / "downscaling_prediction_metadata.json"
    elif key == "fused_lst":
        meta_path = Path(ctx["step7e_output_dir"]) / "fused_lst_metadata.json"
    else:
        return [], "not_applicable", "product has no Step7 processing-window lineage"
    meta = _json(meta_path)
    if not meta:
        return [], "insufficient_boundary_metadata", f"exact inference metadata unavailable: {meta_path}"
    status = _metadata_grid_status(meta, src)
    if status != "verified":
        return [], status, "inference metadata grid differs from audited raster"
    tile_size = int(meta.get("tile_size") or meta.get("window_size_pixels") or 0)
    overlap = int(meta.get("overlap_pixels", 0))
    if tile_size <= 0 or overlap < 0:
        return [], "insufficient_boundary_metadata", "exact tile/window parameters unavailable"
    # Deterministic reconstruction is valid because Step7D and Step7E invoke
    # iter_windows, whose sole grid constructor is this same helper.
    grid = make_tile_grid({"width": src.width, "height": src.height}, tile_size, overlap)
    col_indices = sorted({int(t["write_window"][0]) for t in grid["tiles"] if t["write_window"][0]})
    row_indices = sorted({int(t["write_window"][1]) for t in grid["tiles"] if t["write_window"][1]})
    records: list[BoundaryRecord] = []
    for index in col_indices:
        x0, y0 = src.transform * (index, 0)
        x1, y1 = src.transform * (index, src.height)
        records.append(_stable_boundary(
            boundary_type="processing_window", provider="step7_inference_windows",
            source_product=key, source_artifact=str(product["path"]),
            metadata_source=f"producer_metadata+core.utils.tiling.make_tile_grid:{meta_path}",
            orientation="vertical", coords=[(x0, y0), (x1, y1)], crs=str(src.crs),
            index=index, start=0, end=src.height,
        ))
    for index in row_indices:
        x0, y0 = src.transform * (0, index)
        x1, y1 = src.transform * (src.width, index)
        records.append(_stable_boundary(
            boundary_type="processing_window", provider="step7_inference_windows",
            source_product=key, source_artifact=str(product["path"]),
            metadata_source=f"producer_metadata+core.utils.tiling.make_tile_grid:{meta_path}",
            orientation="horizontal", coords=[(x0, y0), (x1, y1)], crs=str(src.crs),
            index=index, start=0, end=src.width,
        ))
    return records, "available" if records else "not_applicable", None


_TILE_RE = re.compile(r"(?:^|_)tile_r(?P<row>\d+)_c(?P<col>\d+)\.tif$", re.IGNORECASE)


def _expand_export_families(ctx: dict[str, Any], product: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for family in product.get("export_families", []):
        if family == "baseline_lst_yearly":
            result.extend(f"baseline_lst_{year}" for year in ctx["baseline_years"])
        elif family == "baseline_ndvi_yearly":
            result.extend(f"baseline_ndvi_{year}" for year in ctx["baseline_years"])
        else:
            result.append(family)
    return result


def export_tile_boundaries(
    ctx: dict[str, Any], product: dict[str, Any], src: rasterio.DatasetReader,
) -> tuple[list[BoundaryRecord], str, str | None]:
    """Build verified shared edges from actual tile raster footprints."""
    records: list[BoundaryRecord] = []
    saw_grid_mismatch = False
    families = _expand_export_families(ctx, product)
    if not families:
        return [], "insufficient_boundary_metadata", "no export family declared"
    metadata_path = Path(ctx["output_root"]) / "predictor_export_metadata.json"
    metadata = _json(metadata_path) or {}
    for family in families:
        tile_dir = Path(ctx["data_root"]) / "_tiles" / family
        indexed: dict[tuple[int, int], dict[str, Any]] = {}
        for path in sorted(tile_dir.glob("*.tif")) if tile_dir.exists() else []:
            match = _TILE_RE.search(path.name)
            if not match:
                continue
            with rasterio.open(path) as tile:
                indexed[(int(match.group("row")), int(match.group("col")))] = {
                    "path": path, "crs": str(tile.crs), "transform": tile.transform,
                    "bounds": tile.bounds, "width": tile.width, "height": tile.height,
                    "nodata": tile.nodata,
                }
        if not indexed:
            continue
        for (row, col), left in sorted(indexed.items()):
            for neighbor_key, orientation in (((row, col + 1), "vertical"), ((row + 1, col), "horizontal")):
                right = indexed.get(neighbor_key)
                if right is None:
                    continue
                if left["crs"] != right["crs"] or left["crs"] != str(src.crs):
                    saw_grid_mismatch = True
                    continue
                ta, tb = left["transform"], right["transform"]
                if not np.allclose([ta.a, ta.b, ta.d, ta.e], [tb.a, tb.b, tb.d, tb.e], atol=1e-12, rtol=0):
                    saw_grid_mismatch = True
                    continue
                col_delta = (tb.c - ta.c) / ta.a if ta.a else math.nan
                row_delta = (tb.f - ta.f) / ta.e if ta.e else math.nan
                if abs(col_delta - round(col_delta)) > 1e-6 or abs(row_delta - round(row_delta)) > 1e-6:
                    saw_grid_mismatch = True
                    continue
                lb, rb = left["bounds"], right["bounds"]
                if orientation == "vertical":
                    signed_gap = rb.left - lb.right
                    tolerance = abs(ta.a) * 1.5
                    along_min, along_max = max(lb.bottom, rb.bottom), min(lb.top, rb.top)
                    if abs(signed_gap) > tolerance or along_max <= along_min:
                        continue
                    x = (lb.right + rb.left) / 2.0
                    coords = [(x, along_min), (x, along_max)]
                    inv = ~src.transform
                    index = int(round((inv * (x, (along_min + along_max) / 2))[0]))
                    start = max(0, int(math.floor((inv * (x, along_max))[1])))
                    end = min(src.height, int(math.ceil((inv * (x, along_min))[1])))
                    pixels = signed_gap / abs(ta.a)
                else:
                    signed_gap = lb.bottom - rb.top
                    tolerance = abs(ta.e) * 1.5
                    along_min, along_max = max(lb.left, rb.left), min(lb.right, rb.right)
                    if abs(signed_gap) > tolerance or along_max <= along_min:
                        continue
                    y = (lb.bottom + rb.top) / 2.0
                    coords = [(along_min, y), (along_max, y)]
                    inv = ~src.transform
                    index = int(round((inv * ((along_min + along_max) / 2, y))[1]))
                    start = max(0, int(math.floor((inv * (along_min, y))[0])))
                    end = min(src.width, int(math.ceil((inv * (along_max, y))[0])))
                    pixels = signed_gap / abs(ta.e)
                manifest_entry = metadata.get("exports", {}).get(family)
                if manifest_entry:
                    source = f"predictor_export_metadata+tile_raster_footprints:{metadata_path}"
                else:
                    source = "inferred_from_filename_and_verified_by_bounds"
                records.append(_stable_boundary(
                    boundary_type="export_tile", provider="export_tile_footprints",
                    source_product=family,
                    source_artifact=f"{left['path']}|{right['path']}", metadata_source=source,
                    orientation=orientation, coords=coords, crs=str(src.crs),
                    index=index, start=start, end=end,
                    overlap_pixels=max(0.0, -pixels), gap_pixels=max(0.0, pixels),
                ))
    if records:
        return records, "available", None
    if saw_grid_mismatch:
        return [], "grid_mismatch", "tile CRS, transform, or pixel grid mismatch"
    return [], "insufficient_boundary_metadata", "no verified adjacent tile footprints"


def source_scene_boundaries(
    ctx: dict[str, Any], product: dict[str, Any], src: rasterio.DatasetReader,
) -> tuple[list[BoundaryRecord], str, str | None]:
    """Consume versioned source-scene LineStrings while preserving stable IDs."""
    boundary_path = Path(ctx["output_root"]) / "qa" / "source_scene_provenance" / "v1" / "scene_boundaries.geojson"
    if boundary_path.exists():
        collection = _json(boundary_path) or {}
        records: list[BoundaryRecord] = []
        for feature in collection.get("features", []):
            geometry = feature.get("geometry", {})
            if geometry.get("type") != "LineString":
                continue
            props = feature.get("properties", {})
            source_crs = props.get("native_crs", "EPSG:4326")
            if str(source_crs) != str(src.crs):
                geometry = transform_geom(source_crs, src.crs, geometry)
            coords = [(float(x), float(y)) for x, y in geometry["coordinates"]]
            pixels = [(~src.transform) * point for point in coords]
            orientation = props.get("orientation") or ("vertical" if abs(coords[-1][1] - coords[0][1]) >= abs(coords[-1][0] - coords[0][0]) else "horizontal")
            if orientation == "vertical":
                index = int(round(float(np.mean([p[0] for p in pixels]))))
                start = max(0, int(math.floor(min(p[1] for p in pixels))))
                end = min(src.height, int(math.ceil(max(p[1] for p in pixels))))
            else:
                index = int(round(float(np.mean([p[1] for p in pixels]))))
                start = max(0, int(math.floor(min(p[0] for p in pixels))))
                end = min(src.width, int(math.ceil(max(p[0] for p in pixels))))
            normalized = [[round(x, 10), round(y, 10)] for x, y in coords]
            geometry_hash = hashlib.sha256(json.dumps(normalized, separators=(",", ":")).encode()).hexdigest()
            records.append(BoundaryRecord(
                boundary_id=props.get("boundary_id") or "bnd_" + geometry_hash[:24],
                lineage_id=props.get("lineage_id") or "lin_" + geometry_hash[:20],
                boundary_type=props.get("boundary_type", "scene_coverage"),
                provider="source_scene_provenance", source_product=product["product_key"],
                source_artifact=str(boundary_path), metadata_source=str(boundary_path),
                orientation=orientation, geometry_wkt=_line_wkt(coords),
                geometry_hash=geometry_hash, native_crs=str(src.crs),
                verification_status=props.get("verification_status", "verified"),
                index=index, start=start, end=end,
            ))
        if records:
            return records, "available", None
        return [], "insufficient_boundary_metadata", f"no usable LineString boundaries in {boundary_path}"
    candidates = [
        Path(ctx["output_root"]) / "provenance" / "scene_provenance.tif",
        Path(ctx["output_root"]) / "provenance" / "source_scene_id.tif",
        Path(ctx["output_root"]) / "provenance" / "scene_boundary.geojson",
        Path(ctx["output_root"]) / "provenance" / "scene_manifest.parquet",
    ]
    found = [p for p in candidates if p.exists()]
    if not found:
        return [], "insufficient_boundary_metadata", "pixel-level source-scene provenance unavailable"
    return [], "insufficient_boundary_metadata", f"provenance artifact found but provider schema is not configured: {found[0]}"


def _masked(src: rasterio.DatasetReader, window: Window, band: int = 1) -> tuple[np.ndarray, np.ndarray]:
    arr = src.read(band, window=window, masked=True)
    values = np.asarray(arr.filled(np.nan), dtype="float64").reshape(-1)
    valid = ~np.ma.getmaskarray(arr).reshape(-1) & np.isfinite(values)
    return values, valid


def read_boundary_pairs(
    src: rasterio.DatasetReader, boundary: BoundaryRecord, band: int = 1,
    buffer_pixels: int = 1, chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if boundary.index is None or boundary.start is None or boundary.end is None:
        empty = np.array([], dtype="float64")
        return empty, empty, empty.astype(bool), empty.astype(bool)
    distance = max(1, int(buffer_pixels))
    av: list[np.ndarray] = []; bv: list[np.ndarray] = []
    am: list[np.ndarray] = []; bm: list[np.ndarray] = []
    if boundary.orientation == "vertical":
        a_index, b_index = boundary.index - distance, boundary.index + distance - 1
        if a_index < 0 or b_index >= src.width:
            empty = np.array([], dtype="float64")
            return empty, empty, empty.astype(bool), empty.astype(bool)
        for start in range(boundary.start, boundary.end, chunk_size):
            length = min(chunk_size, boundary.end - start)
            a, va = _masked(src, Window(a_index, start, 1, length), band)
            b, vb = _masked(src, Window(b_index, start, 1, length), band)
            av.append(a); bv.append(b); am.append(va); bm.append(vb)
    else:
        a_index, b_index = boundary.index - distance, boundary.index + distance - 1
        if a_index < 0 or b_index >= src.height:
            empty = np.array([], dtype="float64")
            return empty, empty, empty.astype(bool), empty.astype(bool)
        for start in range(boundary.start, boundary.end, chunk_size):
            length = min(chunk_size, boundary.end - start)
            a, va = _masked(src, Window(start, a_index, length, 1), band)
            b, vb = _masked(src, Window(start, b_index, length, 1), band)
            av.append(a); bv.append(b); am.append(va); bm.append(vb)
    if not av:
        empty = np.array([], dtype="float64")
        return empty, empty, empty.astype(bool), empty.astype(bool)
    return np.concatenate(av), np.concatenate(bv), np.concatenate(am), np.concatenate(bm)


def pair_metrics(
    a: np.ndarray, b: np.ndarray, av: np.ndarray, bv: np.ndarray,
    large_jump_absolute: float,
) -> dict[str, Any]:
    total = int(min(len(a), len(b)))
    both = av[:total] & bv[:total]
    transitions = av[:total] ^ bv[:total]
    jumps = np.abs(b[:total][both] - a[:total][both])
    result: dict[str, Any] = {
        "valid_pair_count": int(both.sum()), "invalid_pair_count": int(total - both.sum()),
        "nodata_transition_fraction": float(transitions.sum() / total) if total else None,
        "absolute_jump_mean": None, "absolute_jump_median": None,
        "absolute_jump_p90": None, "absolute_jump_p95": None,
        "absolute_jump_max": None, "large_jump_fraction": None,
    }
    if len(jumps):
        result.update({
            "absolute_jump_mean": float(np.mean(jumps)),
            "absolute_jump_median": float(np.median(jumps)),
            "absolute_jump_p90": float(np.percentile(jumps, 90)),
            "absolute_jump_p95": float(np.percentile(jumps, 95)),
            "absolute_jump_max": float(np.max(jumps)),
            "large_jump_fraction": float(np.mean(jumps >= large_jump_absolute)),
        })
    return result


def thresholds_for(product: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config["thresholds"]["default"])
    result.update(config["thresholds"].get(product.get("semantic_group", "default"), {}))
    return result


def _ratio(value: float | None, control: float | None) -> float | None:
    if value is None or control is None:
        return None
    if abs(control) <= 1e-12:
        return 1.0 if abs(value) <= 1e-12 else 1e12
    return float(value / control)


def classify_continuous(
    metrics: dict[str, Any], thresholds: dict[str, Any], config: dict[str, Any],
) -> str:
    if metrics.get("valid_pair_count", 0) < int(config["minimum_valid_pairs"]):
        return "insufficient_valid_pairs"
    if metrics.get("control_status") != "available":
        return "insufficient_control_pairs"
    absolute = metrics.get("absolute_jump_median")
    ratio = metrics.get("median_jump_ratio")
    if absolute is None or ratio is None:
        return "insufficient_control_pairs"
    conjunction = bool(config["decision_rules"].get("require_absolute_and_ratio", True))
    fail_abs = absolute >= thresholds["large_jump_absolute"]
    fail_ratio = ratio >= thresholds["fail_jump_ratio"]
    warn_ratio = ratio >= thresholds["warn_jump_ratio"]
    if (fail_abs and fail_ratio) if conjunction else (fail_abs or fail_ratio):
        return "fail"
    if (fail_abs and warn_ratio) if conjunction else (fail_abs or warn_ratio):
        return "warn"
    return "pass"


def local_control_boundaries(
    src: rasterio.DatasetReader, boundary: BoundaryRecord,
    all_boundaries: list[BoundaryRecord], config: dict[str, Any],
) -> list[tuple[BoundaryRecord, int]]:
    if boundary.index is None:
        return []
    known = {
        item.index for item in all_boundaries
        if item.orientation == boundary.orientation and item.index is not None
    }
    result: list[tuple[BoundaryRecord, int]] = []
    max_offset = int(config["local_control_max_offset"])
    for magnitude in config["local_control_offsets"]:
        magnitude = int(magnitude)
        if magnitude > max_offset:
            continue
        for signed in (-magnitude, magnitude):
            index = boundary.index + signed
            limit = src.width if boundary.orientation == "vertical" else src.height
            if index <= 0 or index >= limit:
                continue
            if any(abs(index - actual) <= int(config["boundary_buffer_pixels"]) for actual in known):
                continue
            coords = _coords_from_wkt(boundary.geometry_wkt)
            if boundary.orientation == "vertical":
                shift = signed * src.transform.a
                coords = [(x + shift, y) for x, y in coords]
            else:
                shift = signed * src.transform.e
                coords = [(x, y + shift) for x, y in coords]
            result.append((_stable_boundary(
                boundary_type="matched_local_control", provider="parallel_local_offset",
                source_product=boundary.source_product, source_artifact=boundary.source_artifact,
                metadata_source="deterministic_local_offset", orientation=boundary.orientation,
                coords=coords, crs=boundary.native_crs, index=index,
                start=boundary.start, end=boundary.end,
            ), signed))
    return result


def measure_native_boundary(
    src: rasterio.DatasetReader, boundary: BoundaryRecord, product: dict[str, Any],
    config: dict[str, Any], all_boundaries: list[BoundaryRecord],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    thresholds = thresholds_for(product, config)
    a, b, av, bv = read_boundary_pairs(
        src, boundary, int(product.get("band_index", 1)), int(config["boundary_buffer_pixels"]),
    )
    metrics = pair_metrics(a, b, av, bv, thresholds["large_jump_absolute"])
    control_rows: list[dict[str, Any]] = []
    control_jumps: list[np.ndarray] = []
    for control, offset in local_control_boundaries(src, boundary, all_boundaries, config):
        ca, cb, cav, cbv = read_boundary_pairs(
            src, control, int(product.get("band_index", 1)), int(config["boundary_buffer_pixels"]),
        )
        cm = pair_metrics(ca, cb, cav, cbv, thresholds["large_jump_absolute"])
        both = cav & cbv
        if both.any():
            control_jumps.append(np.abs(cb[both] - ca[both]))
        control_rows.append({
            "product_key": product["product_key"], "control_id": control.boundary_id,
            "matched_boundary_id": boundary.boundary_id, "offset_distance": offset,
            "orientation": boundary.orientation, "native_or_modeling": "native",
            "mask_class": "matched_validity", **cm,
        })
    combined = np.concatenate(control_jumps) if control_jumps else np.array([], dtype=float)
    control_status = "available" if len(combined) >= int(config["minimum_valid_pairs"]) else "insufficient_control_pairs"
    control_median = float(np.median(combined)) if len(combined) else None
    control_p95 = float(np.percentile(combined, 95)) if len(combined) else None
    row = {
        "product_key": product["product_key"], **boundary_row(boundary),
        "native_or_modeling": "native", "scale": "native",
        "matched_native_boundary_id": boundary.boundary_id,
        "matched_modeling_boundary_id": boundary.boundary_id,
        "control_status": control_status, "coverage_status": "not_applicable",
        "continuous_jump_status": None, **metrics,
        "control_absolute_jump_median": control_median,
        "control_absolute_jump_p95": control_p95,
        "median_jump_ratio": _ratio(metrics["absolute_jump_median"], control_median),
        "p95_jump_ratio": _ratio(metrics["absolute_jump_p95"], control_p95),
    }
    row["status"] = classify_continuous(row, thresholds, config)
    row["continuous_jump_status"] = row["status"]
    return row, control_rows


def canonical_grid_info(ctx: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    stats_path = Path(ctx["step8a_output_dir"]) / "step8a_dataset_stats.json"
    stats = _json(stats_path) or {}
    candidate = stats.get("diagnostic_rasters", {}).get("valid_mask_raster")
    path = Path(candidate) if candidate else Path(ctx["step8a_output_dir"]) / "step8a_500m_grid_valid_mask.tif"
    if not path.exists():
        return None, f"canonical Step8A grid raster unavailable: {path}"
    try:
        with rasterio.open(path) as src:
            return {
                "path": path, "crs": str(src.crs), "transform": src.transform,
                "width": src.width, "height": src.height, "metadata_source": str(stats_path),
            }, None
    except rasterio.RasterioError as exc:
        return None, str(exc)


def map_boundary_to_canonical_pairs(
    boundary: BoundaryRecord, canonical: dict[str, Any],
) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], dict[str, Any]]:
    coords = _coords_from_wkt(boundary.geometry_wkt)
    geometry: dict[str, Any] = {"type": "LineString", "coordinates": coords}
    transformed = str(boundary.native_crs) != str(canonical["crs"])
    if transformed:
        geometry = transform_geom(boundary.native_crs, canonical["crs"], geometry)
    points = [(float(x), float(y)) for x, y in geometry["coordinates"]]
    inv = ~canonical["transform"]
    pixels = [inv * point for point in points]
    orientation = boundary.orientation
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if orientation == "vertical":
        col_f = float(np.mean([p[0] for p in pixels]))
        right_col = math.floor(col_f)
        row_min = max(0, math.floor(min(p[1] for p in pixels)))
        row_max = min(canonical["height"], math.ceil(max(p[1] for p in pixels)))
        if 0 < right_col < canonical["width"]:
            pairs = [((row, right_col - 1), (row, right_col)) for row in range(row_min, row_max)]
    else:
        row_f = float(np.mean([p[1] for p in pixels]))
        lower_row = math.floor(row_f)
        col_min = max(0, math.floor(min(p[0] for p in pixels)))
        col_max = min(canonical["width"], math.ceil(max(p[0] for p in pixels)))
        if 0 < lower_row < canonical["height"]:
            pairs = [((lower_row - 1, col), (lower_row, col)) for col in range(col_min, col_max)]
    native_length = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
    matched_length = len(pairs) * (abs(canonical["transform"].e) if orientation == "vertical" else abs(canonical["transform"].a))
    assertion = {
        "native_boundary_crs": boundary.native_crs,
        "canonical_500m_crs": canonical["crs"],
        "coordinate_transform_used": transformed,
        "matched_500m_pair_count": len(pairs),
        "unmatched_boundary_length": max(0.0, native_length - matched_length),
    }
    return pairs, assertion


def _cell_mean(
    src: rasterio.DatasetReader, canonical: dict[str, Any], row: int, col: int,
    cache: dict[tuple[int, int], tuple[float, bool]], band: int,
) -> tuple[float, bool]:
    key = (row, col)
    if key in cache:
        return cache[key]
    x0, y0 = canonical["transform"] * (col, row)
    x1, y1 = canonical["transform"] * (col + 1, row + 1)
    left, right, bottom, top = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
    if str(src.crs) != str(canonical["crs"]):
        left, bottom, right, top = transform_bounds(canonical["crs"], src.crs, left, bottom, right, top)
    try:
        window = from_bounds(left, bottom, right, top, src.transform).round_offsets().round_lengths()
        full = Window(0, 0, src.width, src.height)
        window = window.intersection(full)
        values, valid = _masked(src, window, band)
        result = (float(np.mean(values[valid])), True) if valid.any() else (math.nan, False)
    except Exception:
        result = (math.nan, False)
    cache[key] = result
    return result


def _pairs_from_dataset(
    pairs: list[tuple[tuple[int, int], tuple[int, int]]], dataset: pd.DataFrame,
    feature: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    series = dataset.set_index(["row_500m", "col_500m"])[feature]
    av: list[float] = []; bv: list[float] = []
    for left, right in pairs:
        av.append(float(series.get(left, math.nan))); bv.append(float(series.get(right, math.nan)))
    a = np.asarray(av); b = np.asarray(bv)
    return a, b, np.isfinite(a), np.isfinite(b)


def _pairs_from_raster(
    src: rasterio.DatasetReader, pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    canonical: dict[str, Any], cache: dict[tuple[int, int], tuple[float, bool]], band: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a: list[float] = []; b: list[float] = []; av: list[bool] = []; bv: list[bool] = []
    for left, right in pairs:
        va, oka = _cell_mean(src, canonical, *left, cache, band)
        vb, okb = _cell_mean(src, canonical, *right, cache, band)
        a.append(va); b.append(vb); av.append(oka); bv.append(okb)
    return np.asarray(a), np.asarray(b), np.asarray(av), np.asarray(bv)


def measure_modeling_boundary(
    src: rasterio.DatasetReader, boundary: BoundaryRecord, product: dict[str, Any],
    config: dict[str, Any], canonical: dict[str, Any] | None,
    dataset: pd.DataFrame | None, cache: dict[tuple[int, int], tuple[float, bool]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = {
        "product_key": product["product_key"], **boundary_row(boundary),
        "native_or_modeling": "modeling_500m", "scale": "modeling_500m",
        "matched_native_boundary_id": boundary.boundary_id,
        "matched_modeling_boundary_id": boundary.boundary_id,
        "coverage_status": "not_applicable", "continuous_jump_status": None,
    }
    if canonical is None:
        return {**base, "status": "insufficient_grid_metadata", "continuous_jump_status": "insufficient_grid_metadata", "control_status": "insufficient_control_pairs"}, []
    pairs, assertion = map_boundary_to_canonical_pairs(boundary, canonical)
    if not pairs:
        return {**base, **assertion, "status": "insufficient_lineage_match", "continuous_jump_status": "insufficient_lineage_match", "control_status": "insufficient_control_pairs"}, []
    feature = product.get("modeling_feature")
    if dataset is not None and feature and feature in dataset.columns:
        a, b, av, bv = _pairs_from_dataset(pairs, dataset, feature)
        aggregation_source = "existing_step8a_dataset"
    else:
        a, b, av, bv = _pairs_from_raster(src, pairs, canonical, cache, int(product.get("band_index", 1)))
        aggregation_source = "audit_only_canonical_cell_aggregation"
    thresholds = thresholds_for(product, config)
    metrics = pair_metrics(a, b, av, bv, thresholds["large_jump_absolute"])
    # Canonical controls are parallel adjacent-cell boundaries, one or two
    # cells away, matched to this exact boundary and orientation.
    controls: list[dict[str, Any]] = []
    jumps: list[np.ndarray] = []
    for offset in (-2, -1, 1, 2):
        shifted = []
        for left, right in pairs:
            if boundary.orientation == "vertical":
                shifted.append(((left[0], left[1] + offset), (right[0], right[1] + offset)))
            else:
                shifted.append(((left[0] + offset, left[1]), (right[0] + offset, right[1])))
        shifted = [
            pair for pair in shifted
            if all(0 <= cell[0] < canonical["height"] and 0 <= cell[1] < canonical["width"] for cell in pair)
        ]
        if not shifted:
            continue
        if dataset is not None and feature and feature in dataset.columns:
            ca, cb, cav, cbv = _pairs_from_dataset(shifted, dataset, feature)
        else:
            ca, cb, cav, cbv = _pairs_from_raster(src, shifted, canonical, cache, int(product.get("band_index", 1)))
        cm = pair_metrics(ca, cb, cav, cbv, thresholds["large_jump_absolute"])
        both = cav & cbv
        if both.any():
            jumps.append(np.abs(cb[both] - ca[both]))
        control_id = "ctl_" + hashlib.sha256(f"{boundary.boundary_id}|model|{offset}".encode()).hexdigest()[:20]
        controls.append({
            "product_key": product["product_key"], "control_id": control_id,
            "matched_boundary_id": boundary.boundary_id, "offset_distance": offset,
            "orientation": boundary.orientation, "native_or_modeling": "modeling_500m",
            "mask_class": "matched_validity", **cm,
        })
    combined = np.concatenate(jumps) if jumps else np.array([], dtype=float)
    control_status = "available" if len(combined) >= int(config["minimum_valid_pairs"]) else "insufficient_control_pairs"
    control_median = float(np.median(combined)) if len(combined) else None
    control_p95 = float(np.percentile(combined, 95)) if len(combined) else None
    row = {
        **base, **assertion, **metrics, "aggregation_source": aggregation_source,
        "control_status": control_status,
        "control_absolute_jump_median": control_median,
        "control_absolute_jump_p95": control_p95,
        "median_jump_ratio": _ratio(metrics["absolute_jump_median"], control_median),
        "p95_jump_ratio": _ratio(metrics["absolute_jump_p95"], control_p95),
    }
    row["status"] = classify_continuous(row, thresholds, config)
    row["continuous_jump_status"] = row["status"]
    return row, controls


def scan_nodata_coverage(
    src: rasterio.DatasetReader, product: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    """Internal adjacency audit; raster perimeter is never in the denominator."""
    transitions = 0
    opportunities = 0
    previous: np.ndarray | None = None
    chunk = int(config.get("nodata_scan_chunk_rows", 256))
    for row in range(0, src.height, chunk):
        height = min(chunk, src.height - row)
        arr = src.read(int(product.get("band_index", 1)), window=Window(0, row, src.width, height), masked=True)
        values = np.asarray(arr.filled(np.nan))
        valid = ~np.ma.getmaskarray(arr) & np.isfinite(values)
        if valid.shape[1] > 1:
            transitions += int((valid[:, 1:] ^ valid[:, :-1]).sum())
            opportunities += int(valid.shape[0] * (valid.shape[1] - 1))
        if valid.shape[0] > 1:
            transitions += int((valid[1:, :] ^ valid[:-1, :]).sum())
            opportunities += int((valid.shape[0] - 1) * valid.shape[1])
        if previous is not None:
            transitions += int((previous ^ valid[0]).sum())
            opportunities += int(valid.shape[1])
        previous = valid[-1]
    fraction = float(transitions / opportunities) if opportunities else None
    threshold = thresholds_for(product, config)["warn_nodata_transition_fraction"]
    coverage = "warn" if fraction is not None and fraction >= threshold else "pass"
    return {
        "product_key": product["product_key"], "boundary_type": "nodata_edge",
        "provider": "raster_internal_nodata", "boundary_id": f"coverage_{product['product_key']}",
        "lineage_id": f"coverage_{product['product_key']}", "geometry_hash": None,
        "native_or_modeling": "native", "scale": "native", "status": coverage,
        "coverage_status": coverage, "continuous_jump_status": "not_applicable",
        "control_status": "not_applicable", "valid_pair_count": 0,
        "invalid_pair_count": transitions, "internal_nodata_transition_count": transitions,
        "internal_adjacency_opportunities": opportunities,
        "nodata_transition_fraction": fraction, "outer_raster_perimeter_excluded": True,
        "boundary_class": "internal_nodata_or_coverage_boundary",
        "matched_native_boundary_id": None, "matched_modeling_boundary_id": None,
    }


def measure_gapfill_transition(
    ctx: dict[str, Any], src: rasterio.DatasetReader, product: dict[str, Any],
    config: dict[str, Any], canonical: dict[str, Any] | None,
    dataset: pd.DataFrame | None, cache: dict[tuple[int, int], tuple[float, bool]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[BoundaryRecord], str, str | None]:
    mask_path = Path(ctx["step7e_output_dir"]) / "fused_lst_source_mask.tif"
    meta_path = Path(ctx["step7e_output_dir"]) / "fused_lst_metadata.json"
    meta = _json(meta_path)
    if not mask_path.exists() or not meta:
        return [], [], [], "insufficient_boundary_metadata", "source mask or producer metadata unavailable"
    codes = meta.get("source_mask_codes", {})
    observed = next((int(k) for k, v in codes.items() if "observed" in str(v).lower()), None)
    gapfill = next((int(k) for k, v in codes.items() if "downscaled" in str(v).lower() or "gap-fill" in str(v).lower()), None)
    if observed is None or gapfill is None:
        return [], [], [], "insufficient_boundary_metadata", "source mask classes unavailable"
    thresholds = thresholds_for(product, config)
    max_pairs = int(config["max_boundary_pairs"])
    ba: list[np.ndarray] = []; bb: list[np.ndarray] = []
    controls_by_class: dict[int, list[np.ndarray]] = {observed: [], gapfill: []}
    retained = 0; first: tuple[str, int, int] | None = None
    with rasterio.open(mask_path) as mask_src:
        if (mask_src.width, mask_src.height, str(mask_src.crs), mask_src.transform) != (src.width, src.height, str(src.crs), src.transform):
            return [], [], [], "grid_mismatch", "fused source mask differs from fused raster grid"
        for row in range(0, src.height, 256):
            height = min(256, src.height - row)
            mask = mask_src.read(1, window=Window(0, row, src.width, height))
            values = src.read(1, window=Window(0, row, src.width, height), masked=True).filled(np.nan)
            vertical = ((mask[:, :-1] == observed) & (mask[:, 1:] == gapfill)) | ((mask[:, :-1] == gapfill) & (mask[:, 1:] == observed))
            horizontal = ((mask[:-1, :] == observed) & (mask[1:, :] == gapfill)) | ((mask[:-1, :] == gapfill) & (mask[1:, :] == observed))
            for orient, change, a_values, b_values in (
                ("vertical", vertical, values[:, :-1], values[:, 1:]),
                ("horizontal", horizontal, values[:-1, :], values[1:, :]),
            ):
                if change.any() and retained < max_pairs:
                    left, right = a_values[change], b_values[change]
                    take = min(len(left), max_pairs - retained)
                    ba.append(left[:take]); bb.append(right[:take]); retained += take
                    if first is None:
                        rr, cc = np.argwhere(change)[0]
                        first = (orient, int(cc) + (1 if orient == "vertical" else 0), row + int(rr) + (1 if orient == "horizontal" else 0))
            # Matched local controls come from normal same-class adjacencies in
            # the very same chunks that contain transition pixels.
            if vertical.any() or horizontal.any():
                for klass in (observed, gapfill):
                    same_v = (mask[:, :-1] == klass) & (mask[:, 1:] == klass)
                    same_h = (mask[:-1, :] == klass) & (mask[1:, :] == klass)
                    for same, left, right in ((same_v, values[:, :-1], values[:, 1:]), (same_h, values[:-1, :], values[1:, :])):
                        if same.any():
                            diffs = np.abs(right[same] - left[same])
                            controls_by_class[klass].append(diffs[np.isfinite(diffs)][:max_pairs])
    if not ba:
        return [], [], [], "not_applicable", None
    a, b = np.concatenate(ba), np.concatenate(bb)
    av, bv = np.isfinite(a), np.isfinite(b)
    metrics = pair_metrics(a, b, av, bv, thresholds["large_jump_absolute"])
    orientation, index, along = first or ("vertical", 1, 0)
    if orientation == "vertical":
        coords = [src.transform * (index, 0), src.transform * (index, src.height)]
        start, end = 0, src.height
    else:
        coords = [src.transform * (0, index), src.transform * (src.width, index)]
        start, end = 0, src.width
    boundary = _stable_boundary(
        boundary_type="observed_gapfill_transition", provider="step7e_source_mask",
        source_product=product["product_key"], source_artifact=str(mask_path),
        metadata_source=str(meta_path), orientation=orientation, coords=coords,
        crs=str(src.crs), index=index, start=start, end=end,
    )
    control_rows: list[dict[str, Any]] = []
    all_control = []
    for klass, chunks in controls_by_class.items():
        values = np.concatenate(chunks)[:max_pairs] if chunks else np.array([])
        all_control.append(values)
        control_rows.append({
            "product_key": product["product_key"],
            "control_id": "ctl_" + hashlib.sha256(f"{boundary.boundary_id}|gap|{klass}".encode()).hexdigest()[:20],
            "matched_boundary_id": boundary.boundary_id, "offset_distance": 1,
            "orientation": "mixed", "native_or_modeling": "native",
            "mask_class": "observed_observed" if klass == observed else "gapfilled_gapfilled",
            "valid_pair_count": len(values),
            "absolute_jump_median": float(np.median(values)) if len(values) else None,
            "absolute_jump_p95": float(np.percentile(values, 95)) if len(values) else None,
        })
    combined = np.concatenate([x for x in all_control if len(x)]) if any(len(x) for x in all_control) else np.array([])
    control_status = "available" if all(len(x) >= int(config["minimum_valid_pairs"]) for x in all_control) else "insufficient_control_pairs"
    control_median = float(np.median(combined)) if len(combined) else None
    control_p95 = float(np.percentile(combined, 95)) if len(combined) else None
    native = {
        "product_key": product["product_key"], **boundary_row(boundary),
        "native_or_modeling": "native", "scale": "native",
        "matched_native_boundary_id": boundary.boundary_id,
        "matched_modeling_boundary_id": boundary.boundary_id,
        "coverage_status": "not_applicable", "control_status": control_status,
        **metrics, "control_absolute_jump_median": control_median,
        "control_absolute_jump_p95": control_p95,
        "median_jump_ratio": _ratio(metrics["absolute_jump_median"], control_median),
        "p95_jump_ratio": _ratio(metrics["absolute_jump_p95"], control_p95),
    }
    native["status"] = classify_continuous(native, thresholds, config)
    native["continuous_jump_status"] = native["status"]
    modeling, modeling_controls = measure_modeling_boundary(src, boundary, product, config, canonical, dataset, cache)
    return [native, modeling], control_rows + modeling_controls, [boundary], "available", None


def same_boundary_propagation(rows: list[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        boundary_id = row.get("boundary_id")
        scale = row.get("native_or_modeling")
        if boundary_id and scale in {"native", "modeling_500m"} and row.get("boundary_type") != "nodata_edge":
            grouped.setdefault(boundary_id, {})[scale] = row
    result: dict[str, str] = {}
    for boundary_id, pair in grouped.items():
        native, modeling = pair.get("native"), pair.get("modeling_500m")
        if not native or not modeling:
            result[boundary_id] = "insufficient_data"
            continue
        ns, ms = native.get("status"), modeling.get("status")
        adequate = (
            native.get("valid_pair_count", 0) > 0 and modeling.get("valid_pair_count", 0) > 0
            and native.get("control_status") == "available" and modeling.get("control_status") == "available"
        )
        if ns in DETECTED and ms in DETECTED and adequate:
            result[boundary_id] = "propagates_to_500m"
        elif ns in DETECTED and ms == "pass":
            result[boundary_id] = "native_only"
        elif ns == "pass" and ms in DETECTED:
            result[boundary_id] = "modeling_only"
        elif ns == "pass" and ms == "pass":
            result[boundary_id] = "not_detected"
        elif native.get("geometry_hash") != modeling.get("geometry_hash"):
            result[boundary_id] = "insufficient_lineage_match"
        else:
            result[boundary_id] = "insufficient_data"
    return result


def summarize_product(
    product: dict[str, Any], rows: list[dict[str, Any]], config: dict[str, Any],
) -> dict[str, Any]:
    propagation = same_boundary_propagation(rows)
    evaluable = [r for r in rows if r.get("status") in {"pass", "warn", "fail"} and r.get("boundary_type") != "nodata_edge"]
    warned = [r for r in evaluable if r["status"] == "warn"]
    failed = [r for r in evaluable if r["status"] == "fail"]
    ratios = [r["median_jump_ratio"] for r in evaluable if r.get("median_jump_ratio") is not None]
    count = len(evaluable)
    warn_fraction = len(warned) / count if count else 0.0
    fail_fraction = len(failed) / count if count else 0.0
    rules = config["decision_rules"]
    corroborated_fail = len(failed) >= int(rules["min_flagged_segments_fail"]) and fail_fraction >= float(rules["min_flagged_boundary_fraction_fail"])
    corroborated_warn = (
        len(warned) + len(failed) >= int(rules["min_flagged_segments_warn"])
        and (warn_fraction + fail_fraction) >= float(rules["min_flagged_boundary_fraction_warn"])
    )
    coverage_statuses = [
        r.get("coverage_status") for r in rows
        if r.get("boundary_type") == "nodata_edge" and r.get("coverage_status") in {"pass", "warn", "fail"}
    ]
    measured_status = (
        "fail" if corroborated_fail else "warn" if corroborated_warn
        else "pass" if count else "fail" if "fail" in coverage_statuses
        else "warn" if "warn" in coverage_statuses else "pass" if coverage_statuses
        else "incomplete"
    )
    native_statuses = [r["status"] for r in rows if r.get("native_or_modeling") == "native"]
    model_statuses = [r["status"] for r in rows if r.get("native_or_modeling") == "modeling_500m"]
    native_evaluable = any(status in {"pass", "warn", "fail"} for status in native_statuses)
    modeling_evaluable = any(status in {"pass", "warn", "fail"} for status in model_statuses)
    propagating = [bid for bid, value in propagation.items() if value == "propagates_to_500m"]
    native_only = [bid for bid, value in propagation.items() if value == "native_only"]
    modeling_only = [bid for bid, value in propagation.items() if value == "modeling_only"]
    optional_not_produced = product.get("resolution_status") == "not_produced_optional"
    assessment_complete = optional_not_produced or not any(r.get("status") in INCOMPLETE for r in rows)
    if optional_not_produced:
        status = "not_produced_optional"
        conclusion_scope = "not_produced_optional"
    elif not native_evaluable and not modeling_evaluable:
        status = "incomplete"
        conclusion_scope = "insufficient_artifact"
    elif native_evaluable and modeling_evaluable:
        status = measured_status
        conclusion_scope = "complete" if assessment_complete else "evaluated_boundaries_only"
    elif modeling_evaluable:
        status = measured_status
        conclusion_scope = "modeling_scale_only"
    else:
        status = measured_status
        conclusion_scope = "native_scale_only"
    propagation_values = set(propagation.values())
    propagation_status = next((
        value for value in (
            "propagates_to_500m", "native_only", "modeling_only", "not_detected",
            "insufficient_lineage_match", "insufficient_data",
        ) if value in propagation_values
    ), "insufficient_data" if not assessment_complete else "not_applicable")
    native_summary = _scale_summary(native_statuses)
    if product.get("native_artifact_path") is None and not optional_not_produced:
        native_summary = "insufficient_artifact"
    return {
        "product_key": product["product_key"],
        "semantic_identity": product.get("semantic_identity"),
        "artifact_kind": product.get("artifact_kind"),
        "path": str(product["path"]) if product.get("path") is not None else None,
        "native_artifact_path": (
            str(product["native_artifact_path"])
            if product.get("native_artifact_path") is not None else None
        ),
        "native_artifact_status": product.get("native_resolution_status"),
        "required_or_optional": product["required_or_optional"], "source_stage": product["source_stage"],
        "modeling_feature": product.get("modeling_feature"),
        "modeling_feature_available": bool(product.get("modeling_feature_available")),
        "modeling_feature_source_product": product.get("modeling_feature_source_product"),
        "modeling_feature_semantic_identity": product.get("modeling_feature_semantic_identity"),
        "status": status, "native_status": native_summary,
        "modeling_500m_status": _scale_summary(model_statuses),
        "propagation": propagation_status,
        "conclusion_scope": conclusion_scope,
        "total_segment_count": len([r for r in rows if r.get("boundary_type") != "nodata_edge"]),
        "evaluable_segment_count": count, "warn_segment_count": len(warned),
        "fail_segment_count": len(failed), "warn_segment_fraction": warn_fraction,
        "fail_segment_fraction": fail_fraction,
        "median_segment_ratio": float(np.median(ratios)) if ratios else None,
        "p95_segment_ratio": float(np.percentile(ratios, 95)) if ratios else None,
        "maximum_segment_ratio": max(ratios) if ratios else None,
        "propagating_boundary_count": len(propagating),
        "native_only_boundary_count": len(native_only),
        "modeling_only_boundary_count": len(modeling_only),
        "assessment_complete": assessment_complete,
        "corroborated_fail": corroborated_fail, "corroborated_warn": corroborated_warn,
        "propagation_by_boundary": propagation,
        "boundary_types_available": sorted({r["boundary_type"] for r in rows if r.get("status") not in INCOMPLETE | {"not_applicable"}}),
        "boundary_types_incomplete": sorted({r["boundary_type"] for r in rows if r.get("status") in INCOMPLETE}),
    }


def _scale_summary(statuses: list[str]) -> str:
    for status in ("fail", "warn", "pass"):
        if status in statuses:
            return status
    incomplete = [status for status in statuses if status in INCOMPLETE]
    if incomplete:
        return incomplete[0] if len(set(incomplete)) == 1 else "incomplete"
    return "not_applicable"


def blocker_and_rerun(
    products: list[dict[str, Any]], summaries: dict[str, dict[str, Any]],
    all_rows: list[dict[str, Any]],
) -> tuple[bool, str | None, str]:
    registry = {p["product_key"]: p for p in products}
    verified: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for key, summary in summaries.items():
        product = registry[key]
        if not product.get("scientific_predictor") or not (summary["corroborated_fail"] or summary["corroborated_warn"]):
            continue
        for row in all_rows:
            if row.get("product_key") != key or row.get("native_or_modeling") != "native":
                continue
            if summary["propagation_by_boundary"].get(row.get("boundary_id")) != "propagates_to_500m":
                continue
            if row.get("verification_status") != "verified" or row.get("status") not in DETECTED:
                continue
            verified.append((product, row))
    if not verified:
        if any(s["modeling_only_boundary_count"] for s in summaries.values()):
            return False, None, "investigate_500m_mapping_or_feature_aggregation"
        if any(not s["assessment_complete"] for s in summaries.values()):
            return False, None, "produce_boundary_provenance"
        if any(s["native_only_boundary_count"] for s in summaries.values()):
            return False, None, "visual_quality_investigation"
        return False, None, "no_rerun_required"
    stages = []
    for product, row in verified:
        if row["boundary_type"] in {"processing_window", "observed_gapfill_transition"}:
            stages.append("step7")
        elif row["boundary_type"] == "export_tile":
            stages.append("predictors")
    rerun = "predictors" if "predictors" in stages else "step7" if stages else None
    return True, rerun, "rerun_verified_propagating_lineage" if rerun else "investigate_verified_seam"
