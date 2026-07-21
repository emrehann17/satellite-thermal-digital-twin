from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.experiment_context import build_experiment_context
from core.source_scene_provenance_config import (
    DEFAULT_SOURCE_SCENE_PROVENANCE_CONFIG,
)
from src.source_scene_provenance import (
    ArtifactLineageProvider,
    _composite_contract,
    build_artifact_lineage,
    collect_scene_manifest,
    footprint_collection,
    provider_for_context,
    scene_boundaries,
    write_provenance,
)


class FixtureProvider(ArtifactLineageProvider):
    def __init__(self, path: Path, experiment_id: str = "synthetic_future_aoi_2099"):
        super().__init__({"experiment_id": experiment_id})
        self.path = path

    @property
    def provider_name(self) -> str:
        return "fixture"

    def metadata_paths(self) -> list[Path]:
        return [self.path]


def _polygon(x0: float, x1: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[
            [x0, 0], [x1, 0], [x1, 2], [x0, 2], [x0, 0],
        ]],
    }


def _write_scenes(path: Path, same_scene_id: bool = False) -> None:
    scenes = [
        {
            "id": "LC09_SHARED" if same_scene_id else "LC09_A",
            "geometry": _polygon(0, 1),
            "properties": {
                "LANDSAT_PRODUCT_ID": "P_A", "WRS_PATH": 176, "WRS_ROW": 34,
                "DATE_ACQUIRED": "2021-07-01", "SPACECRAFT_ID": "LANDSAT_9",
                "SENSOR_ID": "OLI_TIRS",
            },
        },
        {
            "id": "LC09_SHARED" if same_scene_id else "LC09_B",
            "geometry": _polygon(1, 2),
            "properties": {
                "LANDSAT_PRODUCT_ID": "P_B", "WRS_PATH": 177, "WRS_ROW": 34,
                "DATE_ACQUIRED": "2021-07-02", "SPACECRAFT_ID": "LANDSAT_9",
                "SENSOR_ID": "OLI_TIRS",
            },
        },
    ]
    path.write_text(json.dumps({
        "collections": {
            "current_lst": {"scenes": scenes},
            "current_ndvi": {"scenes": [scenes[0]]},
        },
    }))


def test_scene_manifest_schema_and_deterministic_integer_lookup(tmp_path):
    path = tmp_path / "source_scene_metadata.json"
    _write_scenes(path)
    first, _, _ = collect_scene_manifest(FixtureProvider(path))
    second, _, _ = collect_scene_manifest(FixtureProvider(path))
    required = {
        "experiment_id", "source_collection", "scene_code", "scene_id",
        "landsat_product_id", "spacecraft_id", "sensor_id", "wrs_path",
        "wrs_row", "path_row", "acquisition_datetime", "acquisition_date",
        "cloud_cover", "cloud_cover_land", "processing_level",
        "collection_category", "collection_number", "scene_footprint_wkt",
        "scene_footprint_crs", "source_metadata_method", "input_role",
    }
    assert required <= set(first.columns)
    assert first.to_dict("records") == second.to_dict("records")
    assert pd.api.types.is_integer_dtype(first["scene_code"])
    grouped = first.groupby("scene_id")["scene_code"].nunique()
    assert grouped.max() == 1
    assert first[["scene_code", "scene_id"]].drop_duplicates()["scene_code"].is_unique


def test_real_footprints_and_exact_shared_edge_boundary(tmp_path):
    path = tmp_path / "source_scene_metadata.json"
    _write_scenes(path)
    _, records, _ = collect_scene_manifest(FixtureProvider(path))
    footprints = footprint_collection(records)
    boundaries = scene_boundaries(records)
    assert [feature["geometry"] for feature in footprints["features"][:2]] == [
        _polygon(0, 1), _polygon(1, 2),
    ]
    feature = boundaries["features"][0]
    assert feature["geometry"]["coordinates"] == [[1.0, 0.0], [1.0, 2.0]]
    assert feature["properties"]["boundary_type"] == "path_row_boundary"
    assert feature["properties"]["derivation"] == "exact_shared_source_footprint_edge"


def test_path_row_boundary_survives_same_scene_id(tmp_path):
    path = tmp_path / "source_scene_metadata.json"
    _write_scenes(path, same_scene_id=True)
    _, records, _ = collect_scene_manifest(FixtureProvider(path))
    boundaries = scene_boundaries(records)
    assert any(
        feature["properties"]["boundary_type"] == "path_row_boundary"
        for feature in boundaries["features"]
    )


def test_median_composite_forbids_selected_scene_semantics():
    rows = _composite_contract(copy.deepcopy(
        DEFAULT_SOURCE_SCENE_PROVENANCE_CONFIG,
    ))
    assert rows
    assert all(not row["selected_scene_id_semantically_valid"] for row in rows)
    assert all("not necessarily" in row["dominant_scene_id_definition"] for row in rows)


def test_single_scene_composite_allows_selected_scene_semantics():
    config = copy.deepcopy(DEFAULT_SOURCE_SCENE_PROVENANCE_CONFIG)
    config["composites"]["current_lst"].update({
        "composite_method": "single_scene",
        "temporal_reducer": "none",
        "scene_selection_method": "only_scene",
        "per_pixel_selection": True,
    })
    row = next(
        item for item in _composite_contract(config)
        if item["source_product_role"] == "current_lst"
    )
    assert row["selected_scene_id_semantically_valid"] is True


def test_artifact_lineage_has_contract_and_semantic_mismatch():
    ctx = build_experiment_context("manavgat_2021")
    edges, graph = build_artifact_lineage(
        ctx, {"artifact_products": ["current_lst", "lst_anomaly"]},
    )
    required_node = {
        "artifact_id", "product_key", "stage", "path", "producer",
        "semantic_identity", "grid_signature", "parent_artifact_ids",
        "lineage_available",
    }
    required_edge = {
        "parent_artifact_id", "child_artifact_id", "transformation",
        "aggregation", "resampling", "masking", "composite_method",
    }
    assert all(required_node <= set(node) for node in graph["nodes"])
    assert all(required_edge <= set(edge) for edge in graph["edges"])
    mismatch = next(
        node for node in graph["nodes"]
        if node["artifact_id"] == "modeling_feature:lst_anomaly_mean"
    )
    assert mismatch["source_product"] == "anomaly_zscore"
    assert mismatch["semantic_name_mismatch"] is True
    assert not edges.empty


def test_provider_selection_is_layout_not_aoi_name(tmp_path):
    future = {
        "experiment_id": "synthetic_future_aoi_2099",
        "output_root": tmp_path / "outputs" / "experiments" / "future",
        "data_root": tmp_path / "outputs" / "experiments" / "future" / "data",
    }
    assert provider_for_context(future).provider_name == "namespaced_experiment"
    future["data_root"] = tmp_path / "legacy_data"
    assert provider_for_context(future).provider_name == "legacy_shared_layout"


def test_write_provenance_emits_required_contract(tmp_path):
    manifest = pd.DataFrame(columns=[
        "experiment_id", "scene_code", "scene_id", "input_role",
    ])
    summary = {
        "experiment_id": "future", "status": "insufficient_boundary_metadata",
        "scene_count": 0, "footprint_count": 0, "boundary_count": 0,
        "provider": "fixture", "missing_evidence": ["scene metadata"],
        "pixel_provenance_export_plan": {
            "status": "not_requested", "gee_submission_started": False,
        },
    }
    result = {
        "scene_manifest": manifest,
        "footprints": {"type": "FeatureCollection", "features": []},
        "boundaries": {"type": "FeatureCollection", "features": []},
        "artifact_scene_lineage": pd.DataFrame(columns=[
            "scene_code", "scene_id", "input_role", "artifact_id",
        ]),
        "graph": {"nodes": [], "edges": []},
        "summary": summary,
    }
    written = write_provenance(result, tmp_path / "qa", force=False)
    required = {
        "scene_manifest.parquet", "scene_manifest.json",
        "scene_footprints.geojson", "scene_boundaries.geojson",
        "provenance_summary.json", "provenance_summary.md",
        "artifact_scene_lineage.parquet", "artifact_lineage.json",
        "artifact_lineage_nodes.parquet", "artifact_lineage_edges.parquet",
        "pixel_provenance_export_plan.json", "manifest.json",
    }
    assert required == {Path(path).name for path in written["files"]}

