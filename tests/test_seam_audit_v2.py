"""AOI-independent regression tests for Seam Audit V2 (acceptance A-S)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.experiment_context import build_experiment_context
from core.regions import EXPERIMENTS
from core.seam_audit_v2_config import (
    DEFAULT_SEAM_AUDIT_V2_CONFIG,
    PRODUCT_REGISTRY_V2,
    detect_artifact_identity_conflicts,
    qa_output_dir_v2,
    resolve_product_registry_v2,
)
from scripts import run_seam_audit_v2
from src.seam_audit_v2 import (
    BoundaryRecord,
    blocker_and_rerun,
    classify_continuous,
    export_tile_boundaries,
    local_control_boundaries,
    map_boundary_to_canonical_pairs,
    processing_window_boundaries,
    same_boundary_propagation,
    scan_nodata_coverage,
    summarize_product,
)


def _config() -> dict:
    config = copy.deepcopy(DEFAULT_SEAM_AUDIT_V2_CONFIG)
    config["minimum_valid_pairs"] = 3
    return config


def _write_raster(path: Path, values: np.ndarray, transform=None, nodata=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0],
        count=1, dtype="float32", crs="EPSG:3857",
        transform=transform or from_origin(0, values.shape[0], 1, 1), nodata=nodata,
    ) as dst:
        dst.write(values.astype("float32"), 1)


def _product(path: Path, key: str = "downscaled_lst") -> dict:
    return {
        "product_key": key, "path": path, "band_index": 1,
        "semantic_group": "continuous_temperature", "source_stage": "step7",
        "required_or_optional": "optional", "scientific_predictor": True,
        "export_families": [key],
    }


def _boundary(boundary_id: str = "same", index: int = 4) -> BoundaryRecord:
    return BoundaryRecord(
        boundary_id=boundary_id, lineage_id="lineage", boundary_type="processing_window",
        provider="step7_inference_windows", source_product="downscaled_lst",
        source_artifact="synthetic.tif", metadata_source="synthetic_manifest",
        orientation="vertical", geometry_wkt=f"LINESTRING ({index} 8, {index} 0)",
        geometry_hash="geometry", native_crs="EPSG:3857", verification_status="verified",
        index=index, start=0, end=8,
    )


def _runner_context(root: Path) -> dict:
    return {
        "experiment_id": "future", "region_key": "future", "role": "synthetic",
        "output_root": root, "data_root": root / "data", "baseline_years": [],
        "baseline_input_dir": root / "data" / "landsat_timeseries",
        "ndvi_baseline_dir": root / "data" / "ndvi_timeseries",
        "ndvi_current_dir": root / "data" / "ndvi_current_period",
        "dem_input_dir": root / "data" / "dem", "modis_input_dir": root / "data" / "modis",
        "step5_output_dir": root / "step5", "step5c_output_dir": root / "step5c",
        "step7d_output_dir": root / "step7d", "step7e_output_dir": root / "step7e",
        "step8a_output_dir": root / "step8a",
    }


def test_a_boundary_lineage_isolation():
    assert "processing_window" not in PRODUCT_REGISTRY_V2["elevation"]["boundary_lineage"]
    assert "processing_window" not in PRODUCT_REGISTRY_V2["current_lst"]["boundary_lineage"]
    assert "processing_window" not in PRODUCT_REGISTRY_V2["current_ndvi"]["boundary_lineage"]
    assert "processing_window" in PRODUCT_REGISTRY_V2["downscaled_lst"]["boundary_lineage"]
    assert "processing_window" in PRODUCT_REGISTRY_V2["fused_lst"]["boundary_lineage"]


def test_b_step7a_manifest_is_forbidden_as_processing_fallback():
    with TemporaryDirectory() as tmp:
        root = Path(tmp); raster = root / "downscaled.tif"
        _write_raster(raster, np.zeros((8, 8)))
        step7a = root / "step7a"; step7a.mkdir()
        (step7a / "tiling_test_summary.json").write_text(json.dumps({"tile_size_pixels": 4}))
        ctx = {"step7a_output_dir": step7a, "step7d_output_dir": root / "step7d", "step7e_output_dir": root / "step7e"}
        with rasterio.open(raster) as src:
            boundaries, status, _ = processing_window_boundaries(ctx, _product(raster), src)
    assert boundaries == []
    assert status == "insufficient_boundary_metadata"


def test_c_exact_inference_metadata_reconstructs_windows():
    with TemporaryDirectory() as tmp:
        root = Path(tmp); raster = root / "downscaled.tif"
        transform = from_origin(0, 8, 1, 1)
        _write_raster(raster, np.zeros((8, 8)), transform)
        step7d = root / "step7d"; step7d.mkdir()
        (step7d / "downscaling_prediction_metadata.json").write_text(json.dumps({
            "raster_shape": [8, 8], "crs": "EPSG:3857",
            "transform": list(transform)[:6], "tile_size": 4,
        }))
        ctx = {"step7d_output_dir": step7d, "step7e_output_dir": root / "step7e"}
        with rasterio.open(raster) as src:
            boundaries, status, _ = processing_window_boundaries(ctx, _product(raster), src)
    assert status == "available"
    assert {(b.orientation, b.index) for b in boundaries} == {("vertical", 4), ("horizontal", 4)}
    assert all("step7a" not in b.metadata_source.lower() for b in boundaries)


def test_d_non_integer_shifted_grid_mapping_uses_geometry_not_division():
    boundary = BoundaryRecord(
        "b", "l", "export_tile", "tiles", "family", "tiles", "manifest",
        "vertical", "LINESTRING (995 1000, 995 0)", "g", "EPSG:3857", "verified",
    )
    canonical = {
        "crs": "EPSG:3857", "transform": from_origin(200, 1000, 500, 500),
        "width": 10, "height": 2,
    }
    pairs, assertion = map_boundary_to_canonical_pairs(boundary, canonical)
    assert pairs == [((0, 0), (0, 1)), ((1, 0), (1, 1))]
    assert 33 // 17 == 1  # floor pixel arithmetic is not the mapping contract
    assert round(33 / 17) == 2  # the forbidden approximation would select another edge
    assert assertion["matched_500m_pair_count"] == 2


def test_e_propagation_requires_the_same_boundary_id():
    base = {"valid_pair_count": 10, "control_status": "available", "geometry_hash": "g"}
    different = [
        {**base, "boundary_id": "A", "native_or_modeling": "native", "status": "fail"},
        {**base, "boundary_id": "B", "native_or_modeling": "modeling_500m", "status": "fail"},
    ]
    assert "propagates_to_500m" not in same_boundary_propagation(different).values()
    same = different + [{**base, "boundary_id": "A", "native_or_modeling": "modeling_500m", "status": "fail"}]
    assert same_boundary_propagation(same)["A"] == "propagates_to_500m"


def test_f_modeling_only_never_blocks_or_recommends_rerun():
    product = {"product_key": "x", "scientific_predictor": True}
    summary = {
        "corroborated_fail": True, "corroborated_warn": True,
        "modeling_only_boundary_count": 1, "assessment_complete": True,
        "native_only_boundary_count": 0, "propagation_by_boundary": {"b": "modeling_only"},
    }
    blocker, rerun, action = blocker_and_rerun([product], {"x": summary}, [])
    assert blocker is False and rerun is None
    assert "500m" in action


def test_g_tiny_nodata_fraction_passes():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "tiny.tif"
        values = np.zeros((100, 100), dtype=float); values[50, 50] = -9999
        _write_raster(path, values, nodata=-9999)
        with rasterio.open(path) as src:
            row = scan_nodata_coverage(src, _product(path), _config())
    assert row["nodata_transition_fraction"] < 0.001
    assert row["coverage_status"] == "pass"


def test_h_outer_perimeter_is_excluded():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "valid.tif"; _write_raster(path, np.zeros((10, 10)))
        with rasterio.open(path) as src:
            row = scan_nodata_coverage(src, _product(path), _config())
    assert row["internal_nodata_transition_count"] == 0
    assert row["outer_raster_perimeter_excluded"] is True


def test_i_internal_nodata_holes_warn_above_threshold():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "holes.tif"
        values = np.indices((20, 20)).sum(axis=0).astype(float)
        values[values % 2 == 0] = -9999
        _write_raster(path, values, nodata=-9999)
        with rasterio.open(path) as src:
            row = scan_nodata_coverage(src, _product(path), _config())
    assert row["nodata_transition_fraction"] >= 0.20
    assert row["coverage_status"] == "warn"


def test_j_nodata_zero_valid_pairs_is_not_continuous_warn():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "holes.tif"
        values = np.full((10, 10), -9999.0); _write_raster(path, values, nodata=-9999)
        with rasterio.open(path) as src:
            row = scan_nodata_coverage(src, _product(path), _config())
    assert row["valid_pair_count"] == 0
    assert row["continuous_jump_status"] == "not_applicable"


def test_k_export_provider_uses_actual_2x2_tile_footprints():
    with TemporaryDirectory() as tmp:
        root = Path(tmp); mosaic = root / "mosaic.tif"
        _write_raster(mosaic, np.zeros((4, 4)), from_origin(0, 4, 1, 1))
        tile_root = root / "data" / "_tiles" / "current_lst"
        for row in range(2):
            for col in range(2):
                _write_raster(tile_root / f"x_tile_r{row}_c{col}.tif", np.zeros((2, 2)), from_origin(col * 2, 4 - row * 2, 1, 1))
        ctx = {"data_root": root / "data", "output_root": root, "baseline_years": []}
        with rasterio.open(mosaic) as src:
            boundaries, status, _ = export_tile_boundaries(ctx, _product(mosaic, "current_lst"), src)
    assert status == "available"
    assert len(boundaries) == 4
    assert {b.orientation for b in boundaries} == {"vertical", "horizontal"}


def test_l_half_pixel_tile_offset_is_grid_mismatch():
    with TemporaryDirectory() as tmp:
        root = Path(tmp); mosaic = root / "mosaic.tif"
        _write_raster(mosaic, np.zeros((2, 4)), from_origin(0, 2, 1, 1))
        tile_root = root / "data" / "_tiles" / "current_lst"
        _write_raster(tile_root / "x_tile_r0_c0.tif", np.zeros((2, 2)), from_origin(0, 2, 1, 1))
        _write_raster(tile_root / "x_tile_r0_c1.tif", np.zeros((2, 2)), from_origin(2.5, 2, 1, 1))
        ctx = {"data_root": root / "data", "output_root": root, "baseline_years": []}
        with rasterio.open(mosaic) as src:
            boundaries, status, _ = export_tile_boundaries(ctx, _product(mosaic, "current_lst"), src)
    assert boundaries == [] and status == "grid_mismatch"


def test_m_dynamic_baseline_years_are_resolved_from_context():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        ctx = {
            "experiment_id": "future", "baseline_years": [2097, 2098],
            "output_root": root, "ndvi_baseline_dir": root / "ndvi",
            "baseline_input_dir": root / "lst",
        }
        products, _ = resolve_product_registry_v2(ctx, ["baseline_ndvi_yearly"])
    assert [p["product_key"] for p in products] == ["baseline_ndvi_2097", "baseline_ndvi_2098"]


def test_n_anomaly_zscore_never_maps_to_lst_anomaly_feature():
    assert PRODUCT_REGISTRY_V2["anomaly_zscore"]["modeling_feature"] is None
    assert PRODUCT_REGISTRY_V2["lst_anomaly"]["modeling_feature"] == "lst_anomaly_mean"


def test_o_local_controls_are_parallel_and_bounded():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.tif"; _write_raster(path, np.zeros((20, 20)))
        boundary = _boundary(index=10)
        with rasterio.open(path) as src:
            controls = local_control_boundaries(src, boundary, [boundary], _config())
    assert controls
    assert all(item.orientation == boundary.orientation for item, _ in controls)
    assert all(abs(offset) <= _config()["local_control_max_offset"] for _, offset in controls)


def test_p_missing_controls_cannot_warn_or_fail():
    metrics = {"valid_pair_count": 100, "absolute_jump_median": 99.0, "median_jump_ratio": 99.0, "control_status": "insufficient_control_pairs"}
    assert classify_continuous(metrics, _config()["thresholds"]["default"], _config()) == "insufficient_control_pairs"


def test_q_single_extreme_segment_does_not_make_product_fail():
    product = {**_product(Path("x.tif"), "x"), "path": Path("x.tif")}
    rows = []
    for i in range(20):
        rows.append({
            "boundary_id": str(i), "boundary_type": "export_tile", "native_or_modeling": "native",
            "status": "fail" if i == 0 else "pass", "median_jump_ratio": 10.0 if i == 0 else 1.0,
            "control_status": "available", "valid_pair_count": 100,
        })
    summary = summarize_product(product, rows, _config())
    assert summary["corroborated_fail"] is False
    assert summary["status"] != "fail"


def test_r_v2_run_preserves_v1_bytes():
    with TemporaryDirectory() as tmp:
        root = Path(tmp); v1 = root / "qa" / "seam_audit" / "v1"; v1.mkdir(parents=True)
        artifact = v1 / "manifest.json"; original = b'{"audit_version":"v1"}\n'; artifact.write_bytes(original)
        ctx = {
            "experiment_id": "future", "region_key": "future", "role": "synthetic",
            "output_root": root, "data_root": root / "data", "baseline_years": [],
            "step8a_output_dir": root / "step8a", "step7d_output_dir": root / "step7d",
            "step7e_output_dir": root / "step7e",
        }
        config = copy.deepcopy(DEFAULT_SEAM_AUDIT_V2_CONFIG); config["products"] = []
        with patch.object(run_seam_audit_v2, "build_experiment_context", return_value=ctx), patch.object(
            run_seam_audit_v2, "seam_audit_v2_config", return_value=config,
        ):
            result = run_seam_audit_v2.main("future", products=[], force=True)
        assert result["ran"] is True
        assert artifact.read_bytes() == original
        assert (root / "qa" / "seam_audit" / "v2" / "manifest.json").exists()


def test_s_future_aoi_needs_no_special_branch():
    experiment = {
        "enabled": True, "region_key": "future_geometry", "display_name": "Future AOI",
        "role": "synthetic", "country": "Nowhere",
        "predictor_start_date": "2099-01-01", "predictor_end_date": "2099-02-01",
        "label_start_date": "2099-02-02", "label_end_date": "2099-03-01",
        "baseline_years": [2095, 2096], "output_namespace": "synthetic_future_aoi_2099",
    }
    with patch.dict(EXPERIMENTS, {"synthetic_future_aoi_2099": experiment}):
        ctx = build_experiment_context("synthetic_future_aoi_2099")
        products, _ = resolve_product_registry_v2(ctx, ["current_lst", "baseline_ndvi_yearly"])
    assert [p["product_key"] for p in products] == ["current_lst", "baseline_ndvi_2095", "baseline_ndvi_2096"]
    assert qa_output_dir_v2(ctx).parts[-2:] == ("seam_audit", "v2")


def test_t_semantic_collision_is_rejected_without_explicit_alias():
    shared = Path("/tmp/shared-anomaly.tif")
    products = [
        {"product_key": "lst_anomaly", "semantic_identity": "absolute_lst_anomaly_celsius", "native_artifact_path": shared, "resolution_status": "resolved", "explicit_alias_group": None},
        {"product_key": "anomaly_zscore", "semantic_identity": "standardized_lst_anomaly", "native_artifact_path": shared, "resolution_status": "resolved", "explicit_alias_group": None},
    ]
    conflicts = detect_artifact_identity_conflicts(products)
    assert conflicts[0]["resolution_status"] == "artifact_identity_conflict"
    assert all(p["identity_conflict"] for p in products)


def test_u_same_explicit_alias_group_allows_shared_path():
    shared = Path("/tmp/shared-intentional-alias.tif")
    products = [
        {"product_key": "a", "semantic_identity": "first", "native_artifact_path": shared, "resolution_status": "resolved", "explicit_alias_group": "intentional"},
        {"product_key": "b", "semantic_identity": "second", "native_artifact_path": shared, "resolution_status": "resolved", "explicit_alias_group": "intentional"},
    ]
    assert detect_artifact_identity_conflicts(products) == []
    assert not any(p["identity_conflict"] for p in products)


def test_v_lst_anomaly_never_falls_back_to_anomaly_zscore():
    with TemporaryDirectory() as tmp:
        root = Path(tmp); ctx = _runner_context(root)
        _write_raster(ctx["step5_output_dir"] / "anomaly_zscore.tif", np.zeros((4, 4)))
        products, _ = resolve_product_registry_v2(ctx, ["lst_anomaly", "anomaly_zscore"])
    by_key = {p["product_key"]: p for p in products}
    assert by_key["lst_anomaly"]["path"] is None
    assert by_key["lst_anomaly"]["resolution_status"] == "missing_optional_native_artifact"
    assert by_key["anomaly_zscore"]["path"].name == "anomaly_zscore.tif"


def test_w_missing_native_artifact_keeps_modeling_feature_evaluable():
    with TemporaryDirectory() as tmp:
        root = Path(tmp); source = root / "anomaly_zscore.tif"
        transform = from_origin(0, 8, 1, 1)
        values = np.tile(np.arange(8, dtype=float), (8, 1))
        _write_raster(source, values, transform)
        tile_root = root / "data" / "_tiles" / "current_lst"
        _write_raster(tile_root / "x_tile_r0_c0.tif", values[:, :4], transform)
        _write_raster(tile_root / "x_tile_r0_c1.tif", values[:, 4:], from_origin(4, 8, 1, 1))
        product = copy.deepcopy(PRODUCT_REGISTRY_V2["lst_anomaly"])
        product.update({
            "product_key": "lst_anomaly", "path": None, "native_artifact_path": None,
            "resolution_status": "missing_optional_native_artifact",
            "native_resolution_status": "missing_optional_native_artifact",
            "modeling_feature_available": True, "modeling_feature_source_path": str(source),
            "boundary_lineage": {"export_tile": product["boundary_lineage"]["export_tile"]},
            "export_families": ["current_lst"],
        })
        dataset = pd.DataFrame([
            {"row_500m": row, "col_500m": col, "lst_anomaly_mean": float(col)}
            for row in range(8) for col in range(8)
        ])
        canonical = {"crs": "EPSG:3857", "transform": transform, "width": 8, "height": 8}
        rows, _, _ = run_seam_audit_v2._modeling_feature_only_rows(
            {"data_root": root / "data", "output_root": root, "baseline_years": []},
            product, _config(), canonical, dataset,
        )
        rows.append({"boundary_type": "export_tile", "native_or_modeling": "native", "status": "insufficient_artifact"})
        summary = summarize_product(product, rows, _config())
    assert summary["modeling_500m_status"] == "pass"
    assert summary["native_status"] == "insufficient_artifact"
    assert summary["propagation"] == "insufficient_data"
    assert summary["conclusion_scope"] == "modeling_scale_only"
    assert summary["assessment_complete"] is False


def test_x_missing_native_and_modeling_feature_is_incomplete():
    product = copy.deepcopy(PRODUCT_REGISTRY_V2["lst_anomaly"])
    product.update({
        "product_key": "lst_anomaly", "path": None, "native_artifact_path": None,
        "native_resolution_status": "missing_optional_native_artifact",
        "resolution_status": "missing_optional_native_artifact", "modeling_feature_available": False,
    })
    rows = [
        {"boundary_type": "export_tile", "native_or_modeling": "native", "status": "insufficient_artifact"},
        {"boundary_type": "export_tile", "native_or_modeling": "modeling_500m", "status": "insufficient_artifact"},
    ]
    summary = summarize_product(product, rows, _config())
    assert summary["status"] == "incomplete"
    assert summary["conclusion_scope"] == "insufficient_artifact"


def test_y_optional_not_produced_does_not_poison_overall_completeness():
    with TemporaryDirectory() as tmp:
        root = Path(tmp); ctx = _runner_context(root)
        config = copy.deepcopy(DEFAULT_SEAM_AUDIT_V2_CONFIG)
        config.update({"products": ["baseline_ndvi_mean"], "audit_scales": ["native"]})
        with patch.object(run_seam_audit_v2, "build_experiment_context", return_value=ctx), patch.object(
            run_seam_audit_v2, "seam_audit_v2_config", return_value=config,
        ):
            result = run_seam_audit_v2.main("future", force=True)
    assert result["summary"]["assessment_complete"] is True
    assert result["summary"]["optional_products_not_produced"] == ["baseline_ndvi_mean"]
    assert result["summary"]["products"]["baseline_ndvi_mean"]["conclusion_scope"] == "not_produced_optional"


def test_z_missing_required_product_remains_fail_fast():
    with TemporaryDirectory() as tmp:
        root = Path(tmp); ctx = _runner_context(root)
        config = copy.deepcopy(DEFAULT_SEAM_AUDIT_V2_CONFIG)
        config.update({"products": ["current_lst"], "audit_scales": ["native"]})
        with patch.object(run_seam_audit_v2, "build_experiment_context", return_value=ctx), patch.object(
            run_seam_audit_v2, "seam_audit_v2_config", return_value=config,
        ), pytest.raises(run_seam_audit_v2.SeamAuditV2StageNotReady, match="required product"):
            run_seam_audit_v2.main("future", force=True)


def test_aa_summary_reasons_keep_boundary_provenance_separate():
    with TemporaryDirectory() as tmp:
        root = Path(tmp); ctx = _runner_context(root); transform = from_origin(0, 8, 1, 1)
        values = np.tile(np.arange(8, dtype=float), (8, 1))
        _write_raster(ctx["step5_output_dir"] / "current_period_median_celsius.tif", values, transform)
        tile_root = ctx["data_root"] / "_tiles" / "current_lst"
        _write_raster(tile_root / "x_tile_r0_c0.tif", values[:, :4], transform)
        _write_raster(tile_root / "x_tile_r0_c1.tif", values[:, 4:], from_origin(4, 8, 1, 1))
        config = copy.deepcopy(DEFAULT_SEAM_AUDIT_V2_CONFIG)
        config.update({"products": ["current_lst"], "audit_scales": ["native"], "minimum_valid_pairs": 3})
        with patch.object(run_seam_audit_v2, "build_experiment_context", return_value=ctx), patch.object(
            run_seam_audit_v2, "seam_audit_v2_config", return_value=config,
        ):
            summary = run_seam_audit_v2.main("future", force=True)["summary"]
    assert summary["missing_boundary_provenance"] == ["source_scene"]
    assert summary["assessment_incomplete_reasons"] == ["source_scene_provenance_unavailable"]
    assert summary["optional_products_not_produced"] == []
    assert summary["missing_required_artifacts"] == []
    assert summary["artifact_identity_conflicts"] == []


def test_ab_artifact_resolution_contract_has_required_identity_fields():
    ctx = build_experiment_context("mugla_2021")
    _, rows = resolve_product_registry_v2(ctx, ["lst_anomaly"])
    required = {
        "product_key", "semantic_identity", "artifact_kind", "resolved_path",
        "resolution_status", "resolution_method", "producer_stage", "producer_manifest",
        "explicit_alias_group", "identity_conflict", "conflicting_product_keys",
        "native_artifact_available", "modeling_feature_available",
    }
    assert required <= rows[0].keys()


def test_ac_mugla_lst_anomaly_and_zscore_have_distinct_native_identity():
    ctx = build_experiment_context("mugla_2021")
    products, _ = resolve_product_registry_v2(ctx, ["lst_anomaly", "anomaly_zscore"])
    by_key = {p["product_key"]: p for p in products}
    assert by_key["lst_anomaly"]["path"] is None
    assert by_key["anomaly_zscore"]["path"] is not None
    assert by_key["lst_anomaly"]["path"] != by_key["anomaly_zscore"]["path"]
