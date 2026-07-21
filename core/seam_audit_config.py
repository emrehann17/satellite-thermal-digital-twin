"""Config-driven product registry and defaults for the read-only seam audit.

The registry deliberately resolves paths through ``ExperimentContext`` keys.
It contains no AOI or experiment-name branches; an experiment may override the
global defaults by adding a ``seam_audit`` mapping to its registry entry.
Thresholds are QA heuristics, not statistical significance tests.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from core.regions import get_experiment


AUDIT_VERSION = "v1"

DEFAULT_SEAM_AUDIT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "products": [
        "current_lst", "current_ndvi", "baseline_lst_mean",
        "baseline_lst_std", "baseline_ndvi_mean", "baseline_ndvi_std",
        "lst_anomaly", "anomaly_zscore", "current_tvdi",
        "tvdi_difference", "modis_lst_mean", "modis_lst_std",
        "elevation", "slope", "downscaled_lst", "fused_lst",
    ],
    "audit_scales": ["native", "modeling_500m"],
    "boundary_types": [
        "export_tile", "processing_window", "source_scene", "nodata_edge",
        "observed_gapfill_transition",
    ],
    "random_seed": 42,
    "control_boundary_count": 500,
    "minimum_valid_pairs": 100,
    "boundary_buffer_pixels": 1,
    "max_boundary_pairs": 200000,
    "modeling_cell_size_m": 500.0,
    "thresholds": {
        "metadata": {
            "kind": "initial_qa_heuristic",
            "formal_statistical_significance": False,
            "causal_attribution": False,
        },
        "default": {
            "warn_jump_ratio": 1.5,
            "fail_jump_ratio": 2.0,
            "warn_large_jump_fraction": 0.05,
            "warn_nodata_transition_fraction": 0.20,
            "large_jump_absolute": 1.0,
        },
        "continuous_temperature": {"large_jump_absolute": 3.0},
        "ndvi": {"large_jump_absolute": 0.15},
        "tvdi": {"large_jump_absolute": 0.15},
        "terrain": {"large_jump_absolute": 20.0},
    },
}


def _spec(
    context_key: str,
    filename: str,
    *,
    required: bool = False,
    source_stage: str,
    semantic_group: str,
    native_resolution: float = 30.0,
    reference_grid: str = "current_lst",
    export_metadata_key: str | None = None,
    modeling_feature: str | None = None,
    supported_boundary_types: tuple[str, ...] = (
        "export_tile", "processing_window", "source_scene", "nodata_edge",
    ),
) -> dict[str, Any]:
    return {
        "context_key": context_key,
        "filename": filename,
        "required_or_optional": "required" if required else "optional",
        "band_index": 1,
        "source_stage": source_stage,
        "continuous_or_categorical": "continuous",
        "semantic_group": semantic_group,
        "native_resolution": native_resolution,
        "reference_grid": reference_grid,
        "resampling_semantics": "mean_for_modeling_500m_audit_only",
        "audit_enabled": True,
        "supported_boundary_types": list(supported_boundary_types),
        "export_metadata_key": export_metadata_key,
        "modeling_feature": modeling_feature,
    }


# Paths here are filenames relative to an existing ExperimentContext directory.
# A future experiment receives the same behavior solely by joining EXPERIMENTS.
PRODUCT_REGISTRY: dict[str, dict[str, Any]] = {
    "current_lst": _spec(
        "step5_output_dir", "current_period_median_celsius.tif", required=True,
        source_stage="predictors", semantic_group="continuous_temperature",
        export_metadata_key="current_lst", modeling_feature="current_lst_mean",
    ),
    "current_ndvi": _spec(
        "ndvi_current_dir", "current_ndvi_median.tif", required=True,
        source_stage="predictors", semantic_group="ndvi",
        export_metadata_key="current_ndvi", modeling_feature="ndvi_mean",
    ),
    "baseline_lst_mean": _spec(
        "step5_output_dir", "baseline_lst_mean_celsius.tif",
        source_stage="predictors", semantic_group="continuous_temperature",
    ),
    "baseline_lst_std": _spec(
        "step5_output_dir", "baseline_lst_std_celsius.tif",
        source_stage="predictors", semantic_group="continuous_temperature",
    ),
    "baseline_ndvi_mean": _spec(
        "step5c_output_dir", "baseline_ndvi_mean.tif",
        source_stage="predictors", semantic_group="ndvi",
    ),
    "baseline_ndvi_std": _spec(
        "step5c_output_dir", "baseline_ndvi_std.tif",
        source_stage="predictors", semantic_group="ndvi",
    ),
    "lst_anomaly": _spec(
        "step5_output_dir", "lst_anomaly_celsius.tif",
        source_stage="predictors", semantic_group="continuous_temperature",
        modeling_feature="lst_anomaly_mean",
    ),
    "anomaly_zscore": _spec(
        "step5_output_dir", "anomaly_zscore.tif",
        source_stage="predictors", semantic_group="default",
        modeling_feature="lst_anomaly_mean",
    ),
    "current_tvdi": _spec(
        "step5c_output_dir", "current_tvdi.tif",
        source_stage="predictors", semantic_group="tvdi",
        modeling_feature="current_tvdi_mean",
    ),
    "tvdi_difference": _spec(
        "step5c_output_dir", "tvdi_difference.tif",
        source_stage="predictors", semantic_group="tvdi",
        modeling_feature="tvdi_difference_mean",
    ),
    "modis_lst_mean": _spec(
        "modis_input_dir", "modis_lst_mean_celsius.tif",
        source_stage="predictors", semantic_group="continuous_temperature",
        native_resolution=1000.0, reference_grid="native_modis",
        supported_boundary_types=("nodata_edge",),
    ),
    "modis_lst_std": _spec(
        "modis_input_dir", "modis_lst_std_celsius.tif",
        source_stage="predictors", semantic_group="continuous_temperature",
        native_resolution=1000.0, reference_grid="native_modis",
        supported_boundary_types=("nodata_edge",),
    ),
    "elevation": _spec(
        "dem_input_dir", "elevation.tif", source_stage="predictors",
        semantic_group="terrain", export_metadata_key="elevation",
        modeling_feature="elevation_mean",
    ),
    "slope": _spec(
        "dem_input_dir", "slope.tif", source_stage="predictors",
        semantic_group="terrain", export_metadata_key="slope",
        modeling_feature="slope_mean",
    ),
    "downscaled_lst": _spec(
        "step7d_output_dir", "downscaled_lst_celsius.tif",
        source_stage="step7", semantic_group="continuous_temperature",
        modeling_feature="downscaled_lst_mean",
    ),
    "fused_lst": _spec(
        "step7e_output_dir", "fused_lst_celsius.tif",
        source_stage="step7", semantic_group="continuous_temperature",
        modeling_feature="fused_lst_mean",
        supported_boundary_types=(
            "export_tile", "processing_window", "source_scene", "nodata_edge",
            "observed_gapfill_transition",
        ),
    ),
}


def seam_audit_config(experiment_id: str) -> dict[str, Any]:
    """Return defaults recursively updated by an experiment's config block."""
    config = deepcopy(DEFAULT_SEAM_AUDIT_CONFIG)
    override = get_experiment(experiment_id).get("seam_audit", {})
    _deep_update(config, override)
    return config


def _deep_update(target: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


def resolve_product_registry(
    ctx: dict[str, Any], requested_products: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Resolve product paths without filesystem discovery or AOI conditions."""
    requested = requested_products or seam_audit_config(ctx["experiment_id"])["products"]
    unknown = [key for key in requested if key not in PRODUCT_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown seam-audit product key(s): {unknown}")

    resolved: list[dict[str, Any]] = []
    for key in requested:
        item = deepcopy(PRODUCT_REGISTRY[key])
        base = Path(ctx[item.pop("context_key")])
        item["product_key"] = key
        item["path"] = base / item.pop("filename")
        item["exists"] = item["path"].exists()
        resolved.append(item)
    return resolved


def qa_output_dir(ctx: dict[str, Any]) -> Path:
    """Versioned, experiment-isolated QA namespace; does not create it."""
    return Path(ctx["output_root"]) / "qa" / "seam_audit" / AUDIT_VERSION
