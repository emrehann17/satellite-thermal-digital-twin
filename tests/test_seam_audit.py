"""Synthetic, AOI-independent regression tests for the seam audit."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.experiment_context import build_experiment_context
from core.regions import EXPERIMENTS
from core.seam_audit_config import DEFAULT_SEAM_AUDIT_CONFIG, resolve_product_registry
from src.seam_audit import (
    BoundarySegment,
    SeamAuditError,
    discover_straight_boundaries,
    measure_segment_modeling,
    measure_segment_native,
    propagation_status,
    scan_nodata_edges,
)


def _config() -> dict:
    config = copy.deepcopy(DEFAULT_SEAM_AUDIT_CONFIG)
    config["minimum_valid_pairs"] = 3
    config["control_boundary_count"] = 300
    return config


def _product(group: str = "continuous_temperature") -> dict:
    return {
        "product_key": "synthetic", "band_index": 1,
        "semantic_group": group, "native_resolution": 30.0,
        "modeling_feature": None,
    }


def _memory_raster(values: np.ndarray, nodata: float | None = None):
    mem = MemoryFile()
    profile = {
        "driver": "GTiff", "height": values.shape[0], "width": values.shape[1],
        "count": 1, "dtype": "float32", "crs": "EPSG:4326",
        "transform": from_origin(0, values.shape[0] * 0.00027, 0.00027, 0.00027),
        "nodata": nodata,
    }
    with mem.open(**profile) as dst:
        dst.write(values.astype("float32"), 1)
    return mem


def _measure(values: np.ndarray, orientation: str, index: int):
    segment = BoundarySegment(
        "processing_window", "synthetic_boundary", orientation, index, 0,
        values.shape[0] if orientation == "vertical" else values.shape[1],
        "synthetic_manifest",
    )
    mem = _memory_raster(values)
    with mem.open() as src:
        native, native_control = measure_segment_native(
            src, segment, _product(), _config(), np.random.default_rng(42), [segment],
        )
        modeling, modeling_control = measure_segment_modeling(
            src, segment, _product(), _config(), np.random.default_rng(42), None,
        )
    mem.close()
    return native, native_control, modeling, modeling_control


def test_smooth_gradient_passes_against_comparable_controls():
    row, col = np.indices((68, 68))
    native, _, _, _ = _measure((row + col).astype(float), "vertical", 34)
    assert native["status"] == "pass"
    assert 0.9 <= native["median_jump_ratio"] <= 1.1


def test_vertical_seam_detected():
    values = np.zeros((68, 68), dtype=float)
    values[:, 34:] += 8.0
    native, _, _, _ = _measure(values, "vertical", 34)
    assert native["status"] == "fail"
    assert native["absolute_jump_median"] == 8.0
    assert native["median_jump_ratio"] > 2.0


def test_horizontal_seam_detected():
    values = np.zeros((68, 68), dtype=float)
    values[34:, :] += 8.0
    native, _, _, _ = _measure(values, "horizontal", 34)
    assert native["status"] == "fail"


def test_native_only_thin_seam_is_diluted_at_500m():
    values = np.zeros((68, 68), dtype=float)
    values[:, 34] = 8.0
    native, _, modeling, _ = _measure(values, "vertical", 34)
    assert native["status"] == "fail"
    assert modeling["status"] == "pass"
    assert propagation_status(native["status"], modeling["status"]) == "native_only"


def test_wide_seam_propagates_to_500m():
    values = np.zeros((68, 68), dtype=float)
    values[:, 34:] += 8.0
    native, _, modeling, _ = _measure(values, "vertical", 34)
    assert native["status"] == "fail"
    assert modeling["status"] == "fail"
    assert propagation_status(native["status"], modeling["status"]) == "propagates_to_500m"


def test_nodata_seam_reports_transitions():
    values = np.zeros((32, 32), dtype="float32")
    values[:, 16:] = -9999.0
    mem = _memory_raster(values, nodata=-9999.0)
    with mem.open() as src:
        result = scan_nodata_edges(src, _product(), _config())
    mem.close()
    assert result["status"] == "warn"
    assert result["coverage_edge_count"] > 0
    assert result["nodata_transition_fraction"] > 0


def test_source_scene_without_provenance_is_insufficient_not_pass():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        ctx = {"output_root": root}
        mem = _memory_raster(np.zeros((8, 8)))
        with mem.open() as src:
            segments, status, reason = discover_straight_boundaries(ctx, _product(), "source_scene", src)
        mem.close()
    assert segments == []
    assert status == "insufficient_boundary_metadata"
    assert "provenance" in reason


def _fake_context(root: Path, experiment_id: str = "synthetic_future_aoi_2099") -> dict:
    keys = (
        "step5_output_dir", "step5c_output_dir", "ndvi_current_dir",
        "modis_input_dir", "dem_input_dir", "step7d_output_dir", "step7e_output_dir",
    )
    return {"experiment_id": experiment_id, **{key: root / key for key in keys}}


def test_optional_product_missing_is_explicit_and_nonfatal():
    with TemporaryDirectory() as tmp:
        item = resolve_product_registry(_fake_context(Path(tmp)), ["baseline_ndvi_mean"])[0]
    assert item["required_or_optional"] == "optional"
    assert item["exists"] is False


def test_required_product_missing_is_explicit():
    with TemporaryDirectory() as tmp:
        item = resolve_product_registry(_fake_context(Path(tmp)), ["current_lst"])[0]
    assert item["required_or_optional"] == "required"
    assert item["exists"] is False


def test_future_aoi_uses_registry_without_name_specific_code():
    experiment = {
        "enabled": True, "region_key": "future_geometry", "display_name": "Future AOI",
        "role": "synthetic", "country": "Nowhere",
        "predictor_start_date": "2099-01-01", "predictor_end_date": "2099-02-01",
        "label_start_date": "2099-02-02", "label_end_date": "2099-03-01",
        "baseline_years": [2095, 2096, 2097, 2098],
        "output_namespace": "synthetic_future_aoi_2099",
    }
    with patch.dict(EXPERIMENTS, {"synthetic_future_aoi_2099": experiment}):
        ctx = build_experiment_context("synthetic_future_aoi_2099")
        resolved = resolve_product_registry(ctx, ["current_lst", "fused_lst"])
    assert [item["product_key"] for item in resolved] == ["current_lst", "fused_lst"]
    assert all("synthetic_future_aoi_2099" in str(item["path"]) for item in resolved)


def test_control_sampling_is_deterministic_for_same_seed():
    values = np.zeros((68, 68), dtype=float)
    values[:, 34:] += 8.0
    first = _measure(values, "vertical", 34)
    second = _measure(values, "vertical", 34)
    assert first == second


def test_processing_metadata_grid_mismatch_is_explicit():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        step7a = root / "step7a"; step7a.mkdir()
        (step7a / "tiling_test_summary.json").write_text(json.dumps({
            "raster_shape": [9, 8], "crs": "EPSG:4326",
            "transform": [0.00027, 0, 0, 0, -0.00027, 0.00216],
            "tile_size_pixels": 4,
        }), encoding="utf-8")
        ctx = {"output_root": root, "step7a_output_dir": step7a}
        mem = _memory_raster(np.zeros((8, 8)))
        with mem.open() as src:
            try:
                discover_straight_boundaries(ctx, _product(), "processing_window", src)
            except SeamAuditError as exc:
                assert "grid_mismatch" in str(exc)
            else:
                raise AssertionError("grid mismatch was silently accepted")
        mem.close()
