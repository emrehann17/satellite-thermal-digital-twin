"""Configuration contract for experiment-aware source-scene provenance V1."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from core.regions import get_experiment

VERSION = "v1"
SCHEMA_VERSION = "1.0"

DEFAULT_SOURCE_SCENE_PROVENANCE_CONFIG: dict[str, Any] = {
    "enabled": True,
    "mode": "metadata_only",
    "collection_roles": ["current_lst", "current_ndvi", "baseline_lst", "baseline_ndvi"],
    "boundary_types": [
        "path_row_boundary", "acquisition_support_boundary",
        "scene_coverage_boundary", "dominant_scene_boundary",
        "observation_count_boundary", "date_support_boundary",
    ],
    "metadata_inputs": ["source_scene_metadata.json", "predictor_export_metadata.json"],
    "pixel_provenance_products": [
        "observation_count", "scene_count", "dominant_path_row",
        "dominant_scene_id", "dominant_scene_fraction",
        "median_acquisition_date", "acquisition_date_spread_days",
        "path_row_support_count",
    ],
    "composites": {
        "current_lst": {
            "composite_method": "median", "temporal_reducer": "median",
            "masking_rules": "QA_PIXEL cloud/shadow/snow/fill + QA_RADSAT",
            "valid_pixel_definition": "finite QA-clean ST_B10 observation",
            "scene_selection_method": "per_pixel_reducer",
            "per_pixel_selection": False,
        },
        "current_ndvi": {
            "composite_method": "median", "temporal_reducer": "median",
            "masking_rules": "QA_PIXEL cloud/shadow/snow/fill + QA_RADSAT",
            "valid_pixel_definition": "finite QA-clean NDVI in [-1, 1]",
            "scene_selection_method": "per_pixel_reducer",
            "per_pixel_selection": False,
        },
        "baseline_lst": {
            "composite_method": "median", "temporal_reducer": "median",
            "masking_rules": "QA_PIXEL cloud/shadow/snow/fill + QA_RADSAT",
            "valid_pixel_definition": "finite QA-clean ST_B10 observation",
            "scene_selection_method": "per_pixel_reducer",
            "per_pixel_selection": False,
        },
        "baseline_ndvi": {
            "composite_method": "median", "temporal_reducer": "median",
            "masking_rules": "QA_PIXEL cloud/shadow/snow/fill + QA_RADSAT",
            "valid_pixel_definition": "finite QA-clean NDVI in [-1, 1]",
            "scene_selection_method": "per_pixel_reducer",
            "per_pixel_selection": False,
        },
    },
}

def source_scene_provenance_config(experiment_id: str) -> dict[str, Any]:
    config = deepcopy(DEFAULT_SOURCE_SCENE_PROVENANCE_CONFIG)
    override = get_experiment(experiment_id).get("source_scene_provenance", {})
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = deepcopy(value)
    return config

