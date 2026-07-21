from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.seam_localization_config import DEFAULT_SEAM_LOCALIZATION_CONFIG
from src import seam_localization as localization
from src.seam_localization import (
    boundary_for_raster,
    classify_propagation,
    inline_manual_boundaries,
    load_boundaries,
    localize_trace,
    manual_boundary_feature,
    run_localization,
    visualization_artifact_suspected,
    visualization_check,
    write_localization,
)


def _write_raster(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=values.shape[1],
        height=values.shape[0], count=1, dtype="float32",
        crs="EPSG:3857", transform=from_origin(0, 60, 1, 1),
    ) as dst:
        dst.write(values.astype("float32"), 1)


def _boundary_collection(manual: bool = False) -> dict:
    boundary_source = "manual_diagnostic" if manual else "source_scene_provenance"
    verification = "diagnostic" if manual else "verified"
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[30, 10], [30, 50]],
            },
            "properties": {
                "boundary_id": "scene_boundary_123",
                "source_boundary_id": "scene_boundary_123",
                "lineage_id": "scene_lineage_123",
                "geometry_hash": "geometry_123",
                "boundary_type": (
                    "manual_diagnostic" if manual else "path_row_boundary"
                ),
                "boundary_source": boundary_source,
                "provider": boundary_source,
                "native_crs": "EPSG:3857",
                "verification_status": verification,
            },
        }],
    }


def _config(products: list[str]) -> dict:
    config = copy.deepcopy(DEFAULT_SEAM_LOCALIZATION_CONFIG)
    config.update({
        "artifact_families": products,
        "minimum_valid_pairs": 3,
        "boundary_buffer_pixels": 1,
        "local_control_offsets": [5, 10],
        "local_control_max_offset": 10,
        "max_boundary_pairs": 10000,
    })
    config["decision_rules"].update({
        "min_flagged_segments_warn": 1,
        "min_flagged_segments_fail": 1,
        "min_flagged_boundary_fraction_warn": 0.0,
        "min_flagged_boundary_fraction_fail": 0.0,
    })
    return config


def _context(tmp_path: Path) -> dict:
    return {
        "experiment_id": "synthetic_future_aoi_2099",
        "output_root": tmp_path,
        "step8a_output_dir": tmp_path / "step8a",
    }


def _prepare_provenance(
    tmp_path: Path, nodes: list[dict], manual: bool = False,
) -> list[Path]:
    root = tmp_path / "qa" / "source_scene_provenance" / "v1"
    root.mkdir(parents=True)
    if manual:
        path = tmp_path / "manual.geojson"
        path.write_text(json.dumps(_boundary_collection(True)))
        root.joinpath("provenance_summary.json").write_text(json.dumps({
            "status": "insufficient_boundary_metadata",
        }))
        root.joinpath("artifact_lineage.json").write_text(json.dumps({
            "nodes": nodes, "edges": [],
        }))
        return [path]
    root.joinpath("scene_boundaries.geojson").write_text(json.dumps(
        _boundary_collection(False),
    ))
    root.joinpath("provenance_summary.json").write_text(json.dumps({
        "status": "available",
    }))
    root.joinpath("artifact_lineage.json").write_text(json.dumps({
        "nodes": nodes, "edges": [],
    }))
    return []


def _products(paths: dict[str, Path]) -> list[dict]:
    products = []
    for key, path in paths.items():
        feature = f"{key}_mean"
        products.append({
            "product_key": key,
            "path": path,
            "semantic_identity": key,
            "semantic_group": "continuous_temperature",
            "source_stage": "predictors",
            "band_index": 1,
            "modeling_feature": feature,
            "modeling_feature_available": True,
            "resolution_status": "resolved",
            "resolution_method": "test_fixture",
        })
    return products


def _run_two_stage(
    tmp_path: Path, monkeypatch, *, manual: bool = False,
) -> dict:
    raw = np.zeros((60, 60), dtype=float)
    derived = raw.copy()
    derived[:, 30:] = 5.0
    paths = {"raw": tmp_path / "raw.tif", "derived": tmp_path / "derived.tif"}
    _write_raster(paths["raw"], raw)
    _write_raster(paths["derived"], derived)
    nodes = [
        {
            "artifact_id": "artifact:raw", "product_key": "raw",
            "artifact_order": 10, "stage": "raw_predictor",
            "parent_artifact_ids": [],
        },
        {
            "artifact_id": "artifact:derived", "product_key": "derived",
            "artifact_order": 20, "stage": "derived_anomaly",
            "parent_artifact_ids": ["artifact:raw"],
        },
    ]
    manual_paths = _prepare_provenance(tmp_path, nodes, manual)
    step8a = tmp_path / "step8a"
    step8a.mkdir()
    rows = []
    for row in range(6):
        for col in range(6):
            rows.append({
                "row_500m": row, "col_500m": col,
                "raw_mean": 0.0,
                "derived_mean": 0.0 if col < 3 else 5.0,
            })
    pd.DataFrame(rows).to_parquet(
        step8a / "step8a_500m_modeling_dataset.parquet", index=False,
    )
    canonical = {
        "crs": "EPSG:3857", "transform": from_origin(0, 60, 10, 10),
        "width": 6, "height": 6,
    }
    products = _products(paths)
    monkeypatch.setattr(
        localization, "resolve_product_registry_v2",
        lambda ctx, requested: (products, [
            {
                "product_key": product["product_key"],
                "resolution_status": "resolved",
                "resolved_path": str(product["path"]),
            }
            for product in products
        ]),
    )
    monkeypatch.setattr(
        localization, "canonical_grid_info", lambda ctx: (canonical, None),
    )
    return run_localization(
        _context(tmp_path), _config(["raw", "derived"]), manual_paths,
    )


def test_source_scene_boundary_appears_in_raw_predictor():
    result = localize_trace([
        {
            "artifact_order": 10, "artifact_id": "artifact:raw",
            "stage": "raw_predictor", "status": "fail",
            "artifact_available": True,
        },
    ])
    assert result["earliest_stage_status"] == "present_at_first_available_artifact"
    assert result["root_cause_upstream_of_available_artifacts"] is True


def test_tiles_pass_mosaic_fails_and_predictor_pass_anomaly_fails():
    mosaic = localize_trace([
        {
            "artifact_order": 0, "artifact_id": "tiles", "stage": "raw_export_tile",
            "status": "pass", "artifact_available": True,
        },
        {
            "artifact_order": 1, "artifact_id": "mosaic", "stage": "mosaic",
            "status": "fail", "artifact_available": True,
        },
    ])
    anomaly = localize_trace([
        {
            "artifact_order": 0, "artifact_id": "predictor", "stage": "predictor",
            "status": "pass", "artifact_available": True,
        },
        {
            "artifact_order": 1, "artifact_id": "anomaly", "stage": "derived_anomaly",
            "status": "warn", "artifact_available": True,
        },
    ])
    assert mosaic["earliest_detected_stage"] == "mosaic"
    assert anomaly["earliest_detected_stage"] == "derived_anomaly"


def test_missing_intermediate_stage_produces_bounded_interval():
    result = localize_trace([
        {
            "artifact_order": 0, "artifact_id": "raw", "stage": "raw",
            "status": "pass", "artifact_available": True,
        },
        {
            "artifact_order": 1, "artifact_id": "mosaic", "stage": "mosaic",
            "status": "insufficient_artifact", "artifact_available": False,
        },
        {
            "artifact_order": 2, "artifact_id": "anomaly", "stage": "anomaly",
            "status": "fail", "artifact_available": True,
        },
    ])
    assert result["earliest_stage_status"] == "bounded_but_not_exact"
    assert result["earliest_possible_stage"] == "mosaic"
    assert result["latest_possible_stage"] == "anomaly"


def test_propagation_vocabulary_and_unit_comparability():
    assert classify_propagation(None, {"status": "fail"}) == "appears_at_this_stage"
    assert classify_propagation(
        {"status": "fail", "semantic_group": "temperature", "standardized_boundary_effect": 2},
        {"status": "fail", "semantic_group": "temperature", "standardized_boundary_effect": 3},
    ) == "amplified"
    assert classify_propagation(
        {"status": "fail", "semantic_group": "temperature", "standardized_boundary_effect": 2},
        {"status": "fail", "semantic_group": "ndvi", "standardized_boundary_effect": 9},
    ) == "persists_from_upstream"
    assert classify_propagation(
        {"status": "fail"}, {"status": "pass"},
    ) == "disappears"


def test_verified_seam_propagates_to_500m_and_blocks(monkeypatch, tmp_path):
    result = _run_two_stage(tmp_path, monkeypatch)
    summary = result["summary"]
    assert summary["earliest_confirmed_stage"] == "derived_anomaly"
    assert summary["boundaries_propagating_to_500m"] == 1
    assert summary["scientific_blocker"] is True
    assert summary["recommended_rerun_from_stage"] == "predictors"


def test_manual_boundary_never_becomes_scientific_blocker(monkeypatch, tmp_path):
    result = _run_two_stage(tmp_path, monkeypatch, manual=True)
    assert result["summary"]["source_scene_provenance_status"] == (
        "insufficient_boundary_metadata"
    )
    assert result["summary"]["assessment_complete"] is False
    assert result["summary"]["scientific_blocker"] is False
    assert result["summary"]["potential_modeling_risk"] is True


def test_native_thin_seam_attenuates_at_500m(monkeypatch, tmp_path):
    values = np.zeros((60, 60), dtype=float)
    values[:, 31] = 5.0
    path = tmp_path / "thin.tif"
    _write_raster(path, values)
    nodes = [{
        "artifact_id": "artifact:thin", "product_key": "thin",
        "artifact_order": 10, "stage": "raw_predictor",
        "parent_artifact_ids": [],
    }]
    _prepare_provenance(tmp_path, nodes)
    product = _products({"thin": path})[0]
    monkeypatch.setattr(
        localization, "resolve_product_registry_v2",
        lambda ctx, requested: ([product], [{
            "product_key": "thin", "resolution_status": "resolved",
            "resolved_path": str(path),
        }]),
    )
    monkeypatch.setattr(localization, "canonical_grid_info", lambda ctx: ({
        "crs": "EPSG:3857", "transform": from_origin(0, 60, 10, 10),
        "width": 6, "height": 6,
    }, None))
    result = run_localization(_context(tmp_path), _config(["thin"]))
    native = next(
        row for row in result["metrics"].to_dict("records")
        if row["scale"] == "native"
    )
    modeling = next(
        row for row in result["metrics"].to_dict("records")
        if row["scale"] == "modeling_500m"
    )
    assert native["status"] in {"warn", "fail"}
    assert modeling["status"] == "pass"
    assert result["summary"]["boundaries_propagating_to_500m"] == 0
    assert result["summary"]["scientific_blocker"] is False


def test_grid_reprojection_preserves_boundary_identity(tmp_path):
    path = tmp_path / "grid.tif"
    _write_raster(path, np.zeros((60, 60)))
    boundary = {
        "boundary_id": "stable", "source_boundary_id": "stable",
        "lineage_id": "lineage", "boundary_type": "manual_diagnostic",
        "boundary_source": "manual_diagnostic", "provider": "manual_diagnostic",
        "native_crs": "EPSG:4326", "verification_status": "diagnostic",
        "metadata_source": "fixture",
        "geometry": {
            "type": "LineString",
            "coordinates": [[0.0001, 0.0001], [0.0001, 0.0004]],
        },
    }
    with rasterio.open(path) as src:
        record = boundary_for_raster(boundary, src)
    assert record.boundary_id == "stable"
    assert record.native_crs == "EPSG:3857"


def test_manual_boundary_helper_and_stable_ids(tmp_path):
    collection = manual_boundary_feature([[1, 2], [3, 4]])
    inline = inline_manual_boundaries([collection])
    assert inline[0]["boundary_source"] == "manual_diagnostic"
    path = tmp_path / "manual.geojson"
    path.write_text(json.dumps(collection))
    first = load_boundaries([path], "manual_diagnostic")[0]
    second = load_boundaries([path], "manual_diagnostic")[0]
    assert first["boundary_id"] == second["boundary_id"]


def test_visualization_only_seam_contract(tmp_path):
    path = tmp_path / "visual.tif"
    _write_raster(path, np.arange(3600).reshape(60, 60))
    rows = visualization_check(
        path, "continuous_temperature", DEFAULT_SEAM_LOCALIZATION_CONFIG,
    )
    assert {row["stretch_method"] for row in rows} == {
        "fixed_physical_scale", "robust_global_scale",
    }
    assert all(row["per_tile_normalization"] is False for row in rows)
    assert visualization_artifact_suspected(
        "pass", fixed_visible=False, robust_visible=False, per_tile_visible=True,
    )


def test_localization_output_contract(monkeypatch, tmp_path):
    result = _run_two_stage(tmp_path, monkeypatch)
    written = write_localization(result, tmp_path / "written", force=False)
    required = {
        "localization_summary.json", "localization_summary.md",
        "artifact_boundary_metrics.parquet", "boundary_stage_trace.parquet",
        "earliest_stage_candidates.parquet", "visualization_checks.parquet",
        "seam_profiles.parquet", "seam_hotspots.geojson",
        "matched_controls.parquet", "artifact_resolution.parquet", "manifest.json",
    }
    assert required == {Path(path).name for path in written["files"]}

