"""Read-only, bounded-memory seam/discontinuity audit primitives.

This module measures boundary-associated discontinuities.  It never opens a
source raster in write mode and never smooths, resamples, overwrites, or fixes
one.  Modeling-scale fallback aggregation is computed from source windows in
memory and exists only in the QA result.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform_geom
from rasterio.windows import Window


STATUS_RANK = {
    "not_applicable": 0,
    "pass": 1,
    "skipped_missing_optional": 1,
    "insufficient_boundary_metadata": 2,
    "insufficient_valid_pairs": 2,
    "warn": 3,
    "fail": 4,
    "error": 5,
}


class SeamAuditError(RuntimeError):
    """Explicit seam-audit failure (including grid mismatch)."""


@dataclass(frozen=True)
class BoundarySegment:
    boundary_type: str
    boundary_id: str
    orientation: str  # vertical or horizontal
    index: int  # boundary lies between index-1 and index
    start: int
    end: int
    metadata_source: str


def _json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _same_transform(left: Iterable[float], right: rasterio.Affine) -> bool:
    values = list(left)
    return len(values) >= 6 and np.allclose(values[:6], list(right)[:6], rtol=0, atol=1e-12)


def _validate_grid_metadata(meta: dict[str, Any], src: rasterio.DatasetReader) -> None:
    shape = meta.get("raster_shape")
    if shape and list(shape) != [src.height, src.width]:
        raise SeamAuditError(
            f"grid_mismatch: metadata shape={shape}, raster shape={[src.height, src.width]}"
        )
    if meta.get("crs") and str(meta["crs"]) != str(src.crs):
        raise SeamAuditError(f"grid_mismatch: metadata CRS={meta['crs']}, raster CRS={src.crs}")
    if meta.get("transform") and not _same_transform(meta["transform"], src.transform):
        raise SeamAuditError("grid_mismatch: boundary metadata transform differs from raster")


def discover_straight_boundaries(
    ctx: dict[str, Any], product: dict[str, Any], boundary_type: str,
    src: rasterio.DatasetReader,
) -> tuple[list[BoundarySegment], str, str | None]:
    """Return straight manifest/window boundaries and availability metadata."""
    output_root = Path(ctx["output_root"])
    if boundary_type == "processing_window":
        meta_path = Path(ctx["step7a_output_dir"]) / "tiling_test_summary.json"
        meta = _json(meta_path)
        if not meta:
            return [], "insufficient_boundary_metadata", f"missing/unreadable {meta_path}"
        _validate_grid_metadata(meta, src)
        tile_size = int(meta.get("tile_size_pixels") or meta.get("tile_grid", {}).get("tile_size_pixels", 0))
        if tile_size <= 0:
            return [], "insufficient_boundary_metadata", "tile_size_pixels unavailable"
        source = f"manifest:{meta_path}"
        return _regular_grid_segments(src.width, src.height, tile_size, tile_size, boundary_type, source), "available", None

    if boundary_type == "export_tile":
        meta_path = output_root / "predictor_export_metadata.json"
        meta = _json(meta_path)
        key = product.get("export_metadata_key")
        export = (meta or {}).get("exports", {}).get(key) if key else None
        if not export:
            return [], "insufficient_boundary_metadata", (
                f"no export manifest entry for product metadata key {key!r}"
            )
        grid = export.get("tile_grid")
        if not (isinstance(grid, list) and len(grid) == 2 and all(int(v) > 0 for v in grid)):
            return [], "insufficient_boundary_metadata", "export tile_grid unavailable"
        rows, cols = map(int, grid)
        segments: list[BoundarySegment] = []
        for col in range(1, cols):
            idx = round(src.width * col / cols)
            segments.append(BoundarySegment(boundary_type, f"export_v{col}", "vertical", idx, 0, src.height, f"manifest:{meta_path}"))
        for row in range(1, rows):
            idx = round(src.height * row / rows)
            segments.append(BoundarySegment(boundary_type, f"export_h{row}", "horizontal", idx, 0, src.width, f"manifest:{meta_path}"))
        return segments, "available" if segments else "not_applicable", None

    if boundary_type == "source_scene":
        # Future-compatible interface.  A pixel provenance raster can be
        # supplied by experiment config without changing this module.
        configured = ctx.get("seam_audit_source_scene_provenance")
        candidates = [Path(configured)] if configured else [output_root / "provenance" / "source_scene.tif"]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            return [], "insufficient_boundary_metadata", "pixel-level source-scene provenance unavailable"
        return [], "insufficient_boundary_metadata", f"provenance interface found but categorical provider not configured: {found}"

    return [], "not_applicable", f"{boundary_type} is not a straight-boundary provider"


def _regular_grid_segments(
    width: int, height: int, col_step: int, row_step: int,
    boundary_type: str, source: str,
) -> list[BoundarySegment]:
    segments = [
        BoundarySegment(boundary_type, f"window_v{c}", "vertical", c, 0, height, source)
        for c in range(col_step, width, col_step)
    ]
    segments.extend(
        BoundarySegment(boundary_type, f"window_h{r}", "horizontal", r, 0, width, source)
        for r in range(row_step, height, row_step)
    )
    return segments


def _masked_values(src: rasterio.DatasetReader, window: Window, band: int) -> tuple[np.ndarray, np.ndarray]:
    arr = src.read(band, window=window, masked=True)
    values = np.asarray(arr.filled(np.nan), dtype="float64").reshape(-1)
    valid = np.isfinite(values) & ~np.ma.getmaskarray(arr).reshape(-1)
    return values, valid


def read_boundary_pairs(
    src: rasterio.DatasetReader, segment: BoundarySegment, band: int = 1,
    buffer_pixels: int = 1, chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read only the two strips bordering one segment, in bounded chunks."""
    a_values: list[np.ndarray] = []
    b_values: list[np.ndarray] = []
    a_valid: list[np.ndarray] = []
    b_valid: list[np.ndarray] = []
    distance = max(int(buffer_pixels), 1)
    if segment.orientation == "vertical":
        left = segment.index - distance
        right = segment.index + distance - 1
        if left < 0 or right >= src.width:
            return tuple(np.array([], dtype="float64") for _ in range(4))  # type: ignore[return-value]
        for start in range(segment.start, segment.end, chunk_size):
            length = min(chunk_size, segment.end - start)
            av, am = _masked_values(src, Window(left, start, 1, length), band)
            bv, bm = _masked_values(src, Window(right, start, 1, length), band)
            a_values.append(av); b_values.append(bv); a_valid.append(am); b_valid.append(bm)
    else:
        upper = segment.index - distance
        lower = segment.index + distance - 1
        if upper < 0 or lower >= src.height:
            return tuple(np.array([], dtype="float64") for _ in range(4))  # type: ignore[return-value]
        for start in range(segment.start, segment.end, chunk_size):
            length = min(chunk_size, segment.end - start)
            av, am = _masked_values(src, Window(start, upper, length, 1), band)
            bv, bm = _masked_values(src, Window(start, lower, length, 1), band)
            a_values.append(av); b_values.append(bv); a_valid.append(am); b_valid.append(bm)
    if not a_values:
        return tuple(np.array([], dtype="float64") for _ in range(4))  # type: ignore[return-value]
    return np.concatenate(a_values), np.concatenate(b_values), np.concatenate(a_valid), np.concatenate(b_valid)


def _pair_metrics(
    a: np.ndarray, b: np.ndarray, a_valid: np.ndarray, b_valid: np.ndarray,
    large_jump_absolute: float,
) -> dict[str, Any]:
    total = int(len(a))
    both = a_valid & b_valid
    transition = a_valid ^ b_valid
    jumps = (b[both] - a[both]).astype("float64", copy=False)
    absolute = np.abs(jumps)
    result: dict[str, Any] = {
        "valid_pair_count": int(both.sum()),
        "invalid_pair_count": int(total - both.sum()),
        "nodata_transition_fraction": float(transition.sum() / total) if total else None,
    }
    if not len(jumps):
        result.update({
            "signed_jump_mean": None, "signed_jump_median": None,
            "absolute_jump_mean": None, "absolute_jump_median": None,
            "absolute_jump_p90": None, "absolute_jump_p95": None,
            "absolute_jump_max": None, "large_jump_fraction": None,
        })
        return result
    result.update({
        "signed_jump_mean": float(np.mean(jumps)),
        "signed_jump_median": float(np.median(jumps)),
        "absolute_jump_mean": float(np.mean(absolute)),
        "absolute_jump_median": float(np.median(absolute)),
        "absolute_jump_p90": float(np.percentile(absolute, 90)),
        "absolute_jump_p95": float(np.percentile(absolute, 95)),
        "absolute_jump_max": float(np.max(absolute)),
        "large_jump_fraction": float(np.mean(absolute >= large_jump_absolute)),
    })
    return result


def sample_control_pairs(
    src: rasterio.DatasetReader, orientation: str, count: int, rng: np.random.Generator,
    excluded_indices: set[int], band: int = 1, buffer_pixels: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproducibly sample comparable adjacent pairs away from real boundaries."""
    distance = max(int(buffer_pixels), 1)
    av: list[float] = []; bv: list[float] = []
    am: list[bool] = []; bm: list[bool] = []
    attempts = 0
    max_attempts = max(count * 30, 100)
    while len(av) < count and attempts < max_attempts:
        attempts += 1
        if orientation == "vertical":
            boundary = int(rng.integers(distance, max(distance + 1, src.width - distance + 1)))
            if any(abs(boundary - x) <= distance * 2 for x in excluded_indices):
                continue
            row = int(rng.integers(0, src.height))
            w1, w2 = Window(boundary - distance, row, 1, 1), Window(boundary + distance - 1, row, 1, 1)
        else:
            boundary = int(rng.integers(distance, max(distance + 1, src.height - distance + 1)))
            if any(abs(boundary - x) <= distance * 2 for x in excluded_indices):
                continue
            col = int(rng.integers(0, src.width))
            w1, w2 = Window(col, boundary - distance, 1, 1), Window(col, boundary + distance - 1, 1, 1)
        a, va = _masked_values(src, w1, band); b, vb = _masked_values(src, w2, band)
        av.append(float(a[0])); bv.append(float(b[0])); am.append(bool(va[0])); bm.append(bool(vb[0]))
    return np.asarray(av), np.asarray(bv), np.asarray(am), np.asarray(bm)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if abs(denominator) <= 1e-12:
        return 1.0 if abs(numerator) <= 1e-12 else 1e12
    return float(numerator / denominator)


def compare_with_controls(boundary: dict[str, Any], control: dict[str, Any], control_abs: np.ndarray) -> dict[str, Any]:
    median_ratio = _ratio(boundary.get("absolute_jump_median"), control.get("absolute_jump_median"))
    p95_ratio = _ratio(boundary.get("absolute_jump_p95"), control.get("absolute_jump_p95"))
    bmedian = boundary.get("absolute_jump_median")
    percentile = None
    if bmedian is not None and len(control_abs):
        percentile = float(100.0 * np.mean(control_abs <= bmedian))
    return {
        "boundary_absolute_jump_median": bmedian,
        "control_absolute_jump_median": control.get("absolute_jump_median"),
        "boundary_absolute_jump_p95": boundary.get("absolute_jump_p95"),
        "control_absolute_jump_p95": control.get("absolute_jump_p95"),
        "median_jump_ratio": median_ratio,
        "p95_jump_ratio": p95_ratio,
        "boundary_percentile_against_controls": percentile,
        "control_large_jump_fraction": control.get("large_jump_fraction"),
    }


def classify_metrics(metrics: dict[str, Any], thresholds: dict[str, Any], minimum_valid_pairs: int) -> str:
    if metrics.get("valid_pair_count", 0) < minimum_valid_pairs:
        return "insufficient_valid_pairs"
    ratio = metrics.get("median_jump_ratio")
    absolute = metrics.get("absolute_jump_median")
    if ratio is not None and absolute is not None and ratio >= thresholds["fail_jump_ratio"] and absolute >= thresholds["large_jump_absolute"]:
        return "fail"
    if (
        (
            ratio is not None and absolute is not None
            and ratio >= thresholds["warn_jump_ratio"]
            and absolute >= thresholds["large_jump_absolute"]
        )
        or (
            (metrics.get("large_jump_fraction") or 0.0)
            - (metrics.get("control_large_jump_fraction") or 0.0)
        ) >= thresholds["warn_large_jump_fraction"]
        or (metrics.get("nodata_transition_fraction") or 0.0) >= thresholds["warn_nodata_transition_fraction"]
    ):
        return "warn"
    return "pass"


def thresholds_for(product: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config["thresholds"]["default"])
    result.update(config["thresholds"].get(product.get("semantic_group", "default"), {}))
    return result


def measure_segment_native(
    src: rasterio.DatasetReader, segment: BoundarySegment, product: dict[str, Any],
    config: dict[str, Any], rng: np.random.Generator, all_segments: list[BoundarySegment],
) -> tuple[dict[str, Any], dict[str, Any]]:
    thresholds = thresholds_for(product, config)
    a, b, am, bm = read_boundary_pairs(
        src, segment, int(product["band_index"]), int(config["boundary_buffer_pixels"]),
    )
    boundary = _pair_metrics(a, b, am, bm, thresholds["large_jump_absolute"])
    excluded = {s.index for s in all_segments if s.orientation == segment.orientation}
    ca, cb, cam, cbm = sample_control_pairs(
        src, segment.orientation, int(config["control_boundary_count"]), rng,
        excluded, int(product["band_index"]), int(config["boundary_buffer_pixels"]),
    )
    control = _pair_metrics(ca, cb, cam, cbm, thresholds["large_jump_absolute"])
    control_abs = np.abs(cb[cam & cbm] - ca[cam & cbm])
    comparison = compare_with_controls(boundary, control, control_abs)
    metrics = {**asdict(segment), "scale": "native", **boundary, **comparison}
    metrics["status"] = classify_metrics(metrics, thresholds, int(config["minimum_valid_pairs"]))
    control_row = {
        "boundary_type": segment.boundary_type, "boundary_id": segment.boundary_id,
        "orientation": segment.orientation, "scale": "native", **control,
        "random_seed": int(config["random_seed"]),
    }
    return metrics, control_row


def _block_mean(src: rasterio.DatasetReader, row: int, col: int, block: int, band: int) -> tuple[float, bool]:
    row_off, col_off = row * block, col * block
    if row_off >= src.height or col_off >= src.width or row < 0 or col < 0:
        return np.nan, False
    window = Window(col_off, row_off, min(block, src.width - col_off), min(block, src.height - row_off))
    values, valid = _masked_values(src, window, band)
    return (float(np.mean(values[valid])), True) if valid.any() else (np.nan, False)


def _aggregated_boundary_pairs(
    src: rasterio.DatasetReader, segment: BoundarySegment, block: int, band: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, BoundarySegment]:
    model_index = max(1, int(round(segment.index / block)))
    pairs_a: list[float] = []; pairs_b: list[float] = []
    valid_a: list[bool] = []; valid_b: list[bool] = []
    if segment.orientation == "vertical":
        start, end = segment.start // block, math.ceil(segment.end / block)
        for row in range(start, end):
            a, av = _block_mean(src, row, model_index - 1, block, band)
            b, bv = _block_mean(src, row, model_index, block, band)
            pairs_a.append(a); pairs_b.append(b); valid_a.append(av); valid_b.append(bv)
    else:
        start, end = segment.start // block, math.ceil(segment.end / block)
        for col in range(start, end):
            a, av = _block_mean(src, model_index - 1, col, block, band)
            b, bv = _block_mean(src, model_index, col, block, band)
            pairs_a.append(a); pairs_b.append(b); valid_a.append(av); valid_b.append(bv)
    model_segment = BoundarySegment(
        segment.boundary_type, segment.boundary_id, segment.orientation,
        model_index, start, end, segment.metadata_source,
    )
    return np.asarray(pairs_a), np.asarray(pairs_b), np.asarray(valid_a), np.asarray(valid_b), model_segment


def _dataset_boundary_pairs(
    dataset: pd.DataFrame, feature: str, segment: BoundarySegment, block: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if feature not in dataset.columns or not {"row_500m", "col_500m"}.issubset(dataset.columns):
        return None
    index = max(1, int(round(segment.index / block)))
    axis = "col_500m" if segment.orientation == "vertical" else "row_500m"
    along = "row_500m" if segment.orientation == "vertical" else "col_500m"
    left = dataset.loc[dataset[axis] == index - 1, [along, feature]].set_index(along)[feature]
    right = dataset.loc[dataset[axis] == index, [along, feature]].set_index(along)[feature]
    joined = pd.concat([left.rename("a"), right.rename("b")], axis=1)
    a = joined["a"].to_numpy(dtype=float); b = joined["b"].to_numpy(dtype=float)
    return a, b, np.isfinite(a), np.isfinite(b)


def measure_segment_modeling(
    src: rasterio.DatasetReader, segment: BoundarySegment, product: dict[str, Any],
    config: dict[str, Any], rng: np.random.Generator,
    modeling_dataset: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    thresholds = thresholds_for(product, config)
    block = max(1, int(round(float(config["modeling_cell_size_m"]) / float(product["native_resolution"]))))
    dataset_pairs = None
    if modeling_dataset is not None and product.get("modeling_feature"):
        dataset_pairs = _dataset_boundary_pairs(modeling_dataset, product["modeling_feature"], segment, block)
    if dataset_pairs is None:
        a, b, am, bm, model_segment = _aggregated_boundary_pairs(src, segment, block, int(product["band_index"]))
        source = "qa_window_mean_fallback"
    else:
        a, b, am, bm = dataset_pairs
        model_segment = BoundarySegment(
            segment.boundary_type, segment.boundary_id, segment.orientation,
            max(1, int(round(segment.index / block))), segment.start // block,
            math.ceil(segment.end / block), segment.metadata_source,
        )
        source = "existing_step8a_dataset"
    boundary = _pair_metrics(a, b, am, bm, thresholds["large_jump_absolute"])

    # Modeling controls use reproducible, comparable aggregated boundaries.
    model_width = math.ceil(src.width / block); model_height = math.ceil(src.height / block)
    ca: list[float] = []; cb: list[float] = []; cam: list[bool] = []; cbm: list[bool] = []
    count = int(config["control_boundary_count"])
    for _ in range(count):
        if segment.orientation == "vertical" and model_width > 1:
            idx = int(rng.integers(1, model_width)); row = int(rng.integers(0, model_height))
            va, oka = _block_mean(src, row, idx - 1, block, int(product["band_index"]))
            vb, okb = _block_mean(src, row, idx, block, int(product["band_index"]))
        elif segment.orientation == "horizontal" and model_height > 1:
            idx = int(rng.integers(1, model_height)); col = int(rng.integers(0, model_width))
            va, oka = _block_mean(src, idx - 1, col, block, int(product["band_index"]))
            vb, okb = _block_mean(src, idx, col, block, int(product["band_index"]))
        else:
            break
        ca.append(va); cb.append(vb); cam.append(oka); cbm.append(okb)
    caa, cbb, cama, cbma = np.asarray(ca), np.asarray(cb), np.asarray(cam), np.asarray(cbm)
    control = _pair_metrics(caa, cbb, cama, cbma, thresholds["large_jump_absolute"])
    control_abs = np.abs(cbb[cama & cbma] - caa[cama & cbma])
    comparison = compare_with_controls(boundary, control, control_abs)
    metrics = {
        **asdict(model_segment), "scale": "modeling_500m",
        "aggregation_source": source, "aggregation_factor_pixels": block,
        **boundary, **comparison,
    }
    metrics["status"] = classify_metrics(metrics, thresholds, int(config["minimum_valid_pairs"]))
    control_row = {
        "boundary_type": segment.boundary_type, "boundary_id": segment.boundary_id,
        "orientation": segment.orientation, "scale": "modeling_500m", **control,
        "random_seed": int(config["random_seed"]), "aggregation_source": source,
    }
    return metrics, control_row


def scan_nodata_edges(
    src: rasterio.DatasetReader, product: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    """Scan coverage transitions block-by-block; memory is bounded by one block."""
    transitions = 0; edges = 0; first: tuple[str, int, int] | None = None
    band = int(product["band_index"])
    previous_last_row: np.ndarray | None = None
    row_offset = 0
    for _, window in src.block_windows(band):
        # Some TIFF layouts split columns; use full-width row strips instead.
        if window.col_off != 0 or window.width != src.width:
            continue
        arr = src.read(band, window=window, masked=True)
        valid = ~np.ma.getmaskarray(arr) & np.isfinite(np.asarray(arr.filled(np.nan)))
        if valid.shape[1] > 1:
            change = valid[:, 1:] ^ valid[:, :-1]
            n = int(change.sum()); transitions += n; edges += int(change.size)
            if first is None and n:
                rr, cc = np.argwhere(change)[0]; first = ("vertical", row_offset + int(rr), int(cc) + 1)
        if valid.shape[0] > 1:
            change = valid[1:, :] ^ valid[:-1, :]
            n = int(change.sum()); transitions += n; edges += int(change.size)
            if first is None and n:
                rr, cc = np.argwhere(change)[0]; first = ("horizontal", row_offset + int(rr) + 1, int(cc))
        if previous_last_row is not None:
            change = valid[0, :] ^ previous_last_row
            n = int(change.sum()); transitions += n; edges += int(change.size)
            if first is None and n:
                cc = int(np.flatnonzero(change)[0]); first = ("horizontal", row_offset, cc)
        previous_last_row = valid[-1, :]
        row_offset += valid.shape[0]

    # Fallback for tiled TIFFs whose block windows are narrower than width.
    if edges == 0:
        previous_last_row = None
        for row in range(0, src.height, 256):
            height = min(256, src.height - row)
            arr = src.read(band, window=Window(0, row, src.width, height), masked=True)
            valid = ~np.ma.getmaskarray(arr) & np.isfinite(np.asarray(arr.filled(np.nan)))
            vertical = valid[:, 1:] ^ valid[:, :-1]
            horizontal = valid[1:, :] ^ valid[:-1, :]
            transitions += int(vertical.sum() + horizontal.sum())
            edges += int(vertical.size + horizontal.size)
            if first is None and vertical.any():
                rr, cc = np.argwhere(vertical)[0]; first = ("vertical", row + int(rr), int(cc) + 1)
            if first is None and horizontal.any():
                rr, cc = np.argwhere(horizontal)[0]; first = ("horizontal", row + int(rr) + 1, int(cc))
            if previous_last_row is not None:
                change = valid[0, :] ^ previous_last_row
                transitions += int(change.sum()); edges += int(change.size)
            previous_last_row = valid[-1, :]
    fraction = float(transitions / edges) if edges else None
    status = "not_applicable" if transitions == 0 else (
        "warn" if fraction is not None else "insufficient_valid_pairs"
    )
    orientation, index, along = first or ("vertical", 0, 0)
    return {
        "boundary_type": "nodata_edge", "boundary_id": "coverage_transitions",
        "orientation": orientation, "index": index,
        "start": along, "end": along + 1, "metadata_source": "raster_nodata_mask",
        "scale": "native", "status": status, "valid_pair_count": 0,
        "invalid_pair_count": transitions, "nodata_transition_fraction": fraction,
        "coverage_edge_count": transitions,
        "signed_jump_mean": None, "signed_jump_median": None,
        "absolute_jump_mean": None, "absolute_jump_median": None,
        "absolute_jump_p90": None, "absolute_jump_p95": None,
        "absolute_jump_max": None, "large_jump_fraction": None,
        "median_jump_ratio": None, "p95_jump_ratio": None,
        "boundary_percentile_against_controls": None,
    }


def scan_gapfill_transitions(
    src: rasterio.DatasetReader, source_mask_path: Path, product: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    """Measure observed(1)-gapfill(2) transitions without loading full rasters."""
    if not source_mask_path.exists():
        return {
            "boundary_type": "observed_gapfill_transition",
            "boundary_id": "observed_gapfill", "scale": "native",
            "status": "insufficient_boundary_metadata",
            "reason": f"source mask unavailable: {source_mask_path}",
        }
    a_all: list[np.ndarray] = []; b_all: list[np.ndarray] = []
    retained = 0
    max_pairs = int(config.get("max_boundary_pairs", 200000))
    first: tuple[str, int, int] | None = None
    with rasterio.open(source_mask_path) as mask_src:
        if (mask_src.width, mask_src.height, mask_src.crs, mask_src.transform) != (src.width, src.height, src.crs, src.transform):
            raise SeamAuditError("grid_mismatch: fused source mask differs from fused_lst grid")
        for row in range(0, src.height, 256):
            height = min(256, src.height - row)
            mask = mask_src.read(1, window=Window(0, row, src.width, height))
            values = src.read(int(product["band_index"]), window=Window(0, row, src.width, height), masked=True).filled(np.nan)
            vchange = ((mask[:, :-1] == 1) & (mask[:, 1:] == 2)) | ((mask[:, :-1] == 2) & (mask[:, 1:] == 1))
            hchange = ((mask[:-1, :] == 1) & (mask[1:, :] == 2)) | ((mask[:-1, :] == 2) & (mask[1:, :] == 1))
            if vchange.any():
                left, right = values[:, :-1][vchange], values[:, 1:][vchange]
                take = min(len(left), max_pairs - retained)
                if take > 0:
                    a_all.append(left[:take]); b_all.append(right[:take]); retained += take
                if first is None:
                    rr, cc = np.argwhere(vchange)[0]; first = ("vertical", int(cc) + 1, row + int(rr))
            if hchange.any():
                upper, lower = values[:-1, :][hchange], values[1:, :][hchange]
                take = min(len(upper), max_pairs - retained)
                if take > 0:
                    a_all.append(upper[:take]); b_all.append(lower[:take]); retained += take
                if first is None:
                    rr, cc = np.argwhere(hchange)[0]; first = ("horizontal", row + int(rr) + 1, int(cc))
    if not a_all:
        return {
            "boundary_type": "observed_gapfill_transition", "boundary_id": "observed_gapfill",
            "scale": "native", "status": "not_applicable", "transition_pair_count": 0,
        }
    a, b = np.concatenate(a_all), np.concatenate(b_all)
    am, bm = np.isfinite(a), np.isfinite(b)
    thresholds = thresholds_for(product, config)
    metrics = _pair_metrics(a, b, am, bm, thresholds["large_jump_absolute"])
    orientation, index, along = first or ("vertical", 0, 0)
    result = {
        "boundary_type": "observed_gapfill_transition", "boundary_id": "observed_gapfill",
        "orientation": orientation, "index": index, "start": along, "end": along + 1,
        "metadata_source": str(source_mask_path), "scale": "native", **metrics,
        "median_jump_ratio": None, "p95_jump_ratio": None,
        "boundary_percentile_against_controls": None,
    }
    result["status"] = classify_metrics(result, thresholds, int(config["minimum_valid_pairs"]))
    return result


def scan_categorical_boundaries(
    src: rasterio.DatasetReader, provenance_path: Path, product: dict[str, Any],
    config: dict[str, Any], rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Measure product jumps wherever a categorical provenance ID changes.

    The provider works for future source-scene provenance rasters without AOI
    code.  At most ``max_boundary_pairs`` pairs are retained for distribution
    metrics; the raster scan itself remains windowed.
    """
    if not provenance_path.exists():
        return ({
            "boundary_type": "source_scene", "boundary_id": "scene_id_change",
            "scale": "native", "status": "insufficient_boundary_metadata",
            "reason": f"pixel-level source-scene provenance unavailable: {provenance_path}",
        }, None)
    max_pairs = int(config.get("max_boundary_pairs", 200000))
    retained = 0; total_transitions = 0
    a_all: list[np.ndarray] = []; b_all: list[np.ndarray] = []
    first: tuple[str, int, int] | None = None
    vertical_count = 0; horizontal_count = 0
    with rasterio.open(provenance_path) as prov:
        if (prov.width, prov.height, prov.crs, prov.transform) != (src.width, src.height, src.crs, src.transform):
            raise SeamAuditError("grid_mismatch: source-scene provenance differs from product grid")
        for row in range(0, src.height, 256):
            height = min(256, src.height - row)
            ids_ma = prov.read(1, window=Window(0, row, src.width, height), masked=True)
            ids = np.asarray(ids_ma.filled(np.nan), dtype="float64")
            ids_valid = ~np.ma.getmaskarray(ids_ma) & np.isfinite(ids)
            values = src.read(int(product["band_index"]), window=Window(0, row, src.width, height), masked=True).filled(np.nan)
            vchange = ids_valid[:, :-1] & ids_valid[:, 1:] & (ids[:, :-1] != ids[:, 1:])
            hchange = ids_valid[:-1, :] & ids_valid[1:, :] & (ids[:-1, :] != ids[1:, :])
            vertical_count += int(vchange.sum()); horizontal_count += int(hchange.sum())
            total_transitions += int(vchange.sum() + hchange.sum())
            if vchange.any():
                left, right = values[:, :-1][vchange], values[:, 1:][vchange]
                take = min(len(left), max_pairs - retained)
                if take > 0:
                    a_all.append(left[:take]); b_all.append(right[:take]); retained += take
                if first is None:
                    rr, cc = np.argwhere(vchange)[0]; first = ("vertical", int(cc) + 1, row + int(rr))
            if hchange.any():
                upper, lower = values[:-1, :][hchange], values[1:, :][hchange]
                take = min(len(upper), max_pairs - retained)
                if take > 0:
                    a_all.append(upper[:take]); b_all.append(lower[:take]); retained += take
                if first is None:
                    rr, cc = np.argwhere(hchange)[0]; first = ("horizontal", row + int(rr) + 1, int(cc))
    if total_transitions == 0:
        return ({
            "boundary_type": "source_scene", "boundary_id": "scene_id_change",
            "scale": "native", "status": "not_applicable", "transition_pair_count": 0,
            "metadata_source": str(provenance_path),
        }, None)
    a, b = np.concatenate(a_all), np.concatenate(b_all)
    am, bm = np.isfinite(a), np.isfinite(b)
    thresholds = thresholds_for(product, config)
    boundary = _pair_metrics(a, b, am, bm, thresholds["large_jump_absolute"])
    control_count = int(config["control_boundary_count"])
    vertical_controls = round(control_count * vertical_count / total_transitions)
    controls = []
    for orientation, count in (("vertical", vertical_controls), ("horizontal", control_count - vertical_controls)):
        if count:
            controls.append(sample_control_pairs(
                src, orientation, count, rng, set(), int(product["band_index"]),
                int(config["boundary_buffer_pixels"]),
            ))
    ca = np.concatenate([item[0] for item in controls]); cb = np.concatenate([item[1] for item in controls])
    cam = np.concatenate([item[2] for item in controls]); cbm = np.concatenate([item[3] for item in controls])
    control = _pair_metrics(ca, cb, cam, cbm, thresholds["large_jump_absolute"])
    comparison = compare_with_controls(boundary, control, np.abs(cb[cam & cbm] - ca[cam & cbm]))
    orientation, index, along = first or ("vertical", 0, 0)
    result = {
        "boundary_type": "source_scene", "boundary_id": "scene_id_change",
        "orientation": orientation, "index": index, "start": along, "end": along + 1,
        "metadata_source": str(provenance_path), "scale": "native",
        "transition_pair_count": total_transitions, "retained_pair_count": retained,
        **boundary, **comparison,
    }
    result["status"] = classify_metrics(result, thresholds, int(config["minimum_valid_pairs"]))
    control_row = {
        "boundary_type": "source_scene", "boundary_id": "scene_id_change",
        "orientation": "mixed", "scale": "native", **control,
        "random_seed": int(config["random_seed"]),
    }
    return result, control_row


def propagation_status(native_status: str | None, modeling_status: str | None) -> str:
    detected = {"warn", "fail"}
    if native_status in detected and modeling_status in detected:
        return "propagates_to_500m"
    if native_status in detected and modeling_status == "pass":
        return "native_only"
    if native_status == "pass" and modeling_status in detected:
        return "modeling_only"
    if native_status == "pass" and modeling_status == "pass":
        return "not_detected"
    return "insufficient_data"


def aggregate_product_status(rows: list[dict[str, Any]], scale: str) -> str:
    statuses = [r["status"] for r in rows if r.get("scale") == scale]
    if not statuses:
        return "insufficient_boundary_metadata"
    return max(statuses, key=lambda s: STATUS_RANK.get(s, 5))


def segment_geometry(segment_row: dict[str, Any], src: rasterio.DatasetReader) -> dict[str, Any]:
    """Return an EPSG:4326 GeoJSON LineString for a straight segment."""
    orientation = segment_row.get("orientation", "vertical")
    idx = int(segment_row.get("index", 0)); start = int(segment_row.get("start", 0)); end = int(segment_row.get("end", start + 1))
    scale = segment_row.get("scale")
    if scale == "modeling_500m":
        # Geometry is anchored to the corresponding native-grid fraction.
        factor = max(1, int(segment_row.get("aggregation_factor_pixels", 1)))
        idx *= factor; start *= factor; end *= factor
    if orientation == "vertical":
        coords = [xy(src.transform, max(0, start), min(src.width - 1, idx)), xy(src.transform, min(src.height - 1, end), min(src.width - 1, idx))]
    else:
        coords = [xy(src.transform, min(src.height - 1, idx), max(0, start)), xy(src.transform, min(src.height - 1, idx), min(src.width - 1, end))]
    geometry = {"type": "LineString", "coordinates": [[float(x), float(y)] for x, y in coords]}
    if src.crs and str(src.crs) != "EPSG:4326":
        geometry = transform_geom(src.crs, "EPSG:4326", geometry)
    return geometry
