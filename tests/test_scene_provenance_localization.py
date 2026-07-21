from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.experiment_context import build_experiment_context
from core.pipeline_orchestrator import STAGE_ORDER, validate_stage_range
from core.seam_localization_config import DEFAULT_SEAM_LOCALIZATION_CONFIG
from scripts.run_seam_localization import main as run_localization
from scripts.run_source_scene_provenance import main as run_provenance
from src.seam_localization import (
    classify_propagation, load_boundaries, localize_trace, visualization_check,
)
from src.source_scene_provenance import (
    ArtifactLineageProvider, build_artifact_lineage, collect_scene_manifest,
    footprint_collection, normalize_scene, provider_for_context, scene_boundaries,
)


class FixtureProvider(ArtifactLineageProvider):
    def __init__(self, path: Path): self.path = path
    @property
    def provider_name(self) -> str: return "fixture"
    def metadata_paths(self) -> list[Path]: return [self.path]


def _scenes(path: Path) -> list[dict]:
    geometry_a = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 2], [0, 2], [0, 0]]]}
    geometry_b = {"type": "Polygon", "coordinates": [[[1, 0], [2, 0], [2, 2], [1, 2], [1, 0]]]}
    raw = [
        {"id": "LC09_B", "geometry": geometry_b, "properties": {"WRS_PATH": 177, "WRS_ROW": 34, "DATE_ACQUIRED": "2021-07-02"}},
        {"id": "LC09_A", "geometry": geometry_a, "properties": {"WRS_PATH": 176, "WRS_ROW": 34, "DATE_ACQUIRED": "2021-07-01"}},
    ]
    path.write_text(json.dumps({"collections": {"current_lst": {"scenes": raw}}}))
    return raw


def test_stage_order_contract():
    assert STAGE_ORDER == ["gate", "predictors", "scene-provenance", "step7", "seam-audit", "seam-localization", "step8"]
    assert validate_stage_range("scene-provenance", "seam-localization") == STAGE_ORDER[2:6]


def test_modern_provenance_dry_run_starts_no_gee():
    result = run_provenance("manavgat_2021", dry_run=True)
    assert result["plan"]["provider"] == "namespaced_experiment"
    assert result["plan"]["gee_submission_started"] is False


def test_legacy_provenance_dry_run_uses_adapter():
    result = run_provenance("kozan_2023", dry_run=True)
    assert result["plan"]["provider"] == "legacy_shared_layout"


def test_pixel_provenance_is_plan_only():
    result = run_provenance("manavgat_2021", dry_run=True, mode="pixel_provenance")
    assert result["plan"]["pixel_provenance"] == "plan_only"
    assert result["plan"]["gee_submission_started"] is False


def test_localization_dry_run_starts_no_model_or_gee():
    result = run_localization("kozan_2023", dry_run=True)
    assert result["plan"]["model_training_started"] is False
    assert result["plan"]["gee_submission_started"] is False


def test_provider_selection_uses_layout_not_name():
    ctx = build_experiment_context("manavgat_2021")
    assert provider_for_context(ctx).provider_name == "namespaced_experiment"


def test_scene_manifest_is_deterministic_and_collision_free(tmp_path):
    source = tmp_path / "source_scene_metadata.json"; _scenes(source)
    first, _, _ = collect_scene_manifest(FixtureProvider(source)); second, _, _ = collect_scene_manifest(FixtureProvider(source))
    assert first["scene_code"].tolist() == [1, 2]
    assert first.to_dict("records") == second.to_dict("records")
    assert first["scene_code"].is_unique


def test_manifest_has_required_provenance_fields(tmp_path):
    source = tmp_path / "source_scene_metadata.json"; _scenes(source)
    frame, _, _ = collect_scene_manifest(FixtureProvider(source))
    assert {"scene_id", "path_row", "acquisition_date", "role", "metadata_source"} <= set(frame.columns)
    assert "selected_scene_id" not in frame.columns


def test_footprints_are_from_real_metadata(tmp_path):
    source = tmp_path / "source_scene_metadata.json"; _scenes(source)
    _, records, _ = collect_scene_manifest(FixtureProvider(source))
    assert len(footprint_collection(records)["features"]) == 2


def test_path_row_boundary_is_stable(tmp_path):
    source = tmp_path / "source_scene_metadata.json"; _scenes(source)
    _, records, _ = collect_scene_manifest(FixtureProvider(source))
    one = scene_boundaries(records); two = scene_boundaries(records)
    assert one == two
    assert one["features"][0]["properties"]["boundary_type"] == "path_row_boundary"


def test_artifact_lineage_records_step8_semantic_mismatch():
    ctx = build_experiment_context("manavgat_2021")
    _, graph = build_artifact_lineage(ctx, {"artifact_products": ["lst_anomaly"]})
    node = next(n for n in graph["nodes"] if n["node_id"] == "modeling_feature:lst_anomaly_mean")
    assert node["source_product"] == "anomaly_zscore"
    assert node["semantic_mismatch"] is True


def test_manual_boundary_ids_are_stable(tmp_path):
    path = tmp_path / "manual.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[0, 0], [0, 1]]}, "properties": {}}]}))
    assert load_boundaries([path], "manual")[0]["boundary_id"] == load_boundaries([path], "manual")[0]["boundary_id"]
    assert load_boundaries([path], "manual")[0]["boundary_source"] == "manual_diagnostic"


def test_exact_earliest_stage():
    rows = [{"artifact_order": 0, "artifact_id": "a", "status": "pass"}, {"artifact_order": 1, "artifact_id": "b", "status": "fail"}]
    result = localize_trace(rows)
    assert result["localization_status"] == "exact" and result["earliest_artifact"] == "b"


def test_partial_legacy_lineage_gives_bounds():
    rows = [{"artifact_order": 0, "artifact_id": "a", "status": "missing"}, {"artifact_order": 1, "artifact_id": "b", "status": "warn"}]
    result = localize_trace(rows)
    assert result["localization_status"] == "present_at_first_available_artifact"
    assert result["latest_possible_artifact"] == "b"
    assert result["root_cause_upstream_of_available_artifacts"] is True


def test_first_available_failure_is_upstream_risk():
    result = localize_trace([{"artifact_order": 2, "artifact_id": "first", "status": "fail"}])
    assert result["present_at_first_available_artifact"] is True
    assert result["upstream_risk"] is True


def test_propagation_classes():
    assert classify_propagation(None, {"status": "fail"}) == "appears_at_this_stage"
    assert classify_propagation({"status": "fail", "semantic_group": "temperature", "standardized_boundary_effect": 2}, {"status": "fail", "semantic_group": "temperature", "standardized_boundary_effect": 3}) == "amplified"
    assert classify_propagation({"status": "fail", "semantic_group": "temperature", "standardized_boundary_effect": 2}, {"status": "fail", "semantic_group": "temperature", "standardized_boundary_effect": 1}) == "attenuated"
    assert classify_propagation({"status": "pass"}, {"status": "pass"}) == "not_detected"


def test_visualization_is_global_and_never_tile_normalized(tmp_path):
    path = tmp_path / "x.tif"
    with rasterio.open(path, "w", driver="GTiff", width=10, height=10, count=1, dtype="float32", crs="EPSG:32636", transform=from_origin(0, 300, 30, 30)) as dst:
        dst.write(np.arange(100, dtype="float32").reshape(1, 10, 10))
    result = visualization_check(path, "continuous_temperature", DEFAULT_SEAM_LOCALIZATION_CONFIG)
    assert len(result) == 2
    assert all(row["per_tile_normalization"] is False for row in result)
    assert all(row["normalization_scope"] == "whole_artifact" for row in result)


def test_missing_metadata_is_not_a_pass(tmp_path):
    frame, records, inputs = collect_scene_manifest(FixtureProvider(tmp_path / "missing.json"))
    assert frame.empty and records == [] and inputs == []
