"""Configuration for read-only earliest-stage seam localization V1."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from core.regions import get_experiment
from core.seam_audit_v2_config import DEFAULT_SEAM_AUDIT_V2_CONFIG

VERSION = "v1"
SCHEMA_VERSION = "1.0"

DEFAULT_SEAM_LOCALIZATION_CONFIG: dict[str, Any] = {
    "enabled": True,
    "boundary_sources": [
        "source_scene", "path_row", "observation_support", "export_tile",
    ],
    "artifact_families": [
        "current_lst", "current_ndvi", "baseline_lst_yearly",
        "baseline_ndvi_yearly", "baseline_lst_mean", "baseline_lst_std",
        "anomaly_zscore", "current_tvdi", "tvdi_difference",
        "downscaled_lst", "fused_lst",
    ],
    "audit_scales": ["native", "modeling_500m"],
    "minimum_valid_pairs": 100,
    "boundary_buffer_pixels": DEFAULT_SEAM_AUDIT_V2_CONFIG["boundary_buffer_pixels"],
    "local_control_offsets": [5, 10, 20],
    "local_control_max_offset": 20,
    "max_boundary_pairs": DEFAULT_SEAM_AUDIT_V2_CONFIG["max_boundary_pairs"],
    "random_seed": 42,
    "thresholds": deepcopy(DEFAULT_SEAM_AUDIT_V2_CONFIG["thresholds"]),
    "decision_rules": deepcopy(DEFAULT_SEAM_AUDIT_V2_CONFIG["decision_rules"]),
    "visualization": {
        "methods": ["fixed_physical", "robust_global"],
        "robust_percentiles": [2, 98],
        "per_tile_normalization": False,
        "fixed_ranges": {"continuous_temperature": [0.0, 60.0], "ndvi": [-1.0, 1.0], "tvdi": [0.0, 1.0], "default": [-3.0, 3.0], "terrain": [0.0, 3000.0]},
    },
}

def seam_localization_config(experiment_id: str) -> dict[str, Any]:
    config = deepcopy(DEFAULT_SEAM_LOCALIZATION_CONFIG)
    override = get_experiment(experiment_id).get("seam_localization", {})
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = deepcopy(value)
    return config
