"""Boundary-lineage-aware configuration and artifact resolution for seam audit V2.

V1 deliberately remains in :mod:`core.seam_audit_config`.  This module has no
broad boundary defaults: every product declares the provenance that can
actually create a boundary in that product.  Resolution is deterministic and
producer-first; recursive filesystem discovery is intentionally absent.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from core.regions import get_experiment


AUDIT_VERSION = "v2"
SCHEMA_VERSION = "2.1"

DEFAULT_SEAM_AUDIT_V2_CONFIG: dict[str, Any] = {
    "enabled": True,
    "products": [
        "current_lst", "current_ndvi", "baseline_lst_mean", "baseline_lst_std",
        "baseline_ndvi_yearly", "baseline_ndvi_mean", "baseline_ndvi_std",
        "lst_anomaly", "anomaly_zscore", "current_tvdi", "tvdi_difference",
        "modis_lst_mean", "modis_lst_std", "elevation", "slope",
        "downscaled_lst", "fused_lst",
    ],
    "audit_scales": ["native", "modeling_500m"],
    "random_seed": 42,
    "minimum_valid_pairs": 20,
    "boundary_buffer_pixels": 1,
    "local_control_offsets": [3, 5, 8],
    "local_control_max_offset": 12,
    "max_boundary_pairs": 200000,
    "nodata_scan_chunk_rows": 256,
    "thresholds": {
        "metadata": {
            "kind": "initial_qa_heuristic",
            "formal_statistical_significance": False,
            "causal_attribution": False,
        },
        "default": {
            "warn_jump_ratio": 1.5,
            "fail_jump_ratio": 2.0,
            "large_jump_absolute": 1.0,
            "warn_nodata_transition_fraction": 0.20,
        },
        "continuous_temperature": {"large_jump_absolute": 3.0},
        "ndvi": {"large_jump_absolute": 0.15},
        "tvdi": {"large_jump_absolute": 0.15},
        "terrain": {"large_jump_absolute": 20.0},
    },
    "decision_rules": {
        "require_absolute_and_ratio": True,
        "min_flagged_segments_warn": 1,
        "min_flagged_segments_fail": 2,
        "min_flagged_boundary_fraction_warn": 0.05,
        "min_flagged_boundary_fraction_fail": 0.10,
    },
}


def _lineage(**providers: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return providers


def _raster(
    context_key: str,
    filename: str,
    *,
    required: bool,
    source_stage: str,
    semantic_group: str,
    modeling_feature: str | None,
    boundary_lineage: dict[str, dict[str, Any]],
    derived_from: list[str] | None = None,
    producer_manifest: tuple[str, str] | None = None,
    export_families: list[str] | None = None,
    scientific_predictor: bool = False,
    artifact_kind: str = "native_raster",
    pipeline_produces_native_artifact: bool = True,
    modeling_feature_source_product: str | None = None,
    modeling_feature_semantic_identity: str | None = None,
    explicit_alias_group: str | None = None,
) -> dict[str, Any]:
    return {
        "artifact_kind": artifact_kind,
        "context_key": context_key,
        "filename": filename,
        "path_resolver": "producer_manifest_then_context_then_confirmed_code_path",
        "required_or_optional": "required" if required else "optional",
        "native_artifact_required": required,
        "source_stage": source_stage,
        "semantic_group": semantic_group,
        "native_resolution": "artifact_grid",
        "modeling_feature": modeling_feature,
        "boundary_lineage": boundary_lineage,
        "derived_from": derived_from or [],
        "producer_manifest": producer_manifest,
        "export_families": export_families or [],
        "band_index": 1,
        "scientific_predictor": scientific_predictor,
        "pipeline_produces_native_artifact": pipeline_produces_native_artifact,
        "modeling_feature_source_product": modeling_feature_source_product,
        "modeling_feature_semantic_identity": modeling_feature_semantic_identity,
        "explicit_alias_group": explicit_alias_group,
    }


_EXPORT_SOURCE_NODATA = _lineage(
    export_tile={"provider": "export_tile_footprints", "required_metadata": True},
    source_scene={"provider": "source_scene_provenance", "required_metadata": True},
    nodata_edge={"provider": "raster_internal_nodata", "required_metadata": False},
)

PRODUCT_REGISTRY_V2: dict[str, dict[str, Any]] = {
    "current_lst": _raster(
        "step5_output_dir", "current_period_median_celsius.tif", required=True,
        source_stage="predictors", semantic_group="continuous_temperature",
        modeling_feature="current_lst_mean", boundary_lineage=_EXPORT_SOURCE_NODATA,
        export_families=["current_lst"], scientific_predictor=True,
    ),
    "current_ndvi": _raster(
        "ndvi_current_dir", "current_ndvi_median.tif", required=True,
        source_stage="predictors", semantic_group="ndvi", modeling_feature="ndvi_mean",
        boundary_lineage=_EXPORT_SOURCE_NODATA, export_families=["current_ndvi"],
        scientific_predictor=True,
    ),
    "baseline_lst_mean": _raster(
        "step5_output_dir", "baseline_lst_mean_celsius.tif", required=False,
        source_stage="predictors", semantic_group="continuous_temperature",
        modeling_feature=None, boundary_lineage=_EXPORT_SOURCE_NODATA,
        derived_from=["baseline_lst_yearly"], export_families=["baseline_lst_yearly"],
    ),
    "baseline_lst_std": _raster(
        "step5_output_dir", "baseline_lst_std_celsius.tif", required=False,
        source_stage="predictors", semantic_group="continuous_temperature",
        modeling_feature=None, boundary_lineage=_EXPORT_SOURCE_NODATA,
        derived_from=["baseline_lst_yearly"], export_families=["baseline_lst_yearly"],
    ),
    "baseline_ndvi_mean": _raster(
        "step5c_output_dir", "baseline_ndvi_mean.tif", required=False,
        source_stage="predictors", semantic_group="ndvi", modeling_feature=None,
        boundary_lineage=_EXPORT_SOURCE_NODATA,
        derived_from=["baseline_ndvi_yearly"], export_families=["baseline_ndvi_yearly"],
        pipeline_produces_native_artifact=False,
    ),
    "baseline_ndvi_std": _raster(
        "step5c_output_dir", "baseline_ndvi_std.tif", required=False,
        source_stage="predictors", semantic_group="ndvi", modeling_feature=None,
        boundary_lineage=_EXPORT_SOURCE_NODATA,
        derived_from=["baseline_ndvi_yearly"], export_families=["baseline_ndvi_yearly"],
        pipeline_produces_native_artifact=False,
    ),
    # Step8A's legacy ``lst_anomaly`` predictor name is sourced from Step5's
    # standardized anomaly_zscore.tif.  That modeling feature is retained for
    # compatibility, but it is not evidence that a native absolute-Celsius
    # anomaly artifact exists.
    "lst_anomaly": _raster(
        "step5_output_dir", "lst_anomaly_celsius.tif", required=False,
        source_stage="predictors", semantic_group="default",
        modeling_feature="lst_anomaly_mean", boundary_lineage=_EXPORT_SOURCE_NODATA,
        derived_from=["anomaly_zscore"],
        export_families=["current_lst", "baseline_lst_yearly"],
        artifact_kind="derived_modeling_feature",
        pipeline_produces_native_artifact=False,
        modeling_feature_source_product="anomaly_zscore",
        modeling_feature_semantic_identity="mean_standardized_lst_anomaly",
        scientific_predictor=True,
    ),
    "anomaly_zscore": _raster(
        "step5_output_dir", "anomaly_zscore.tif", required=False,
        source_stage="predictors", semantic_group="default", modeling_feature=None,
        boundary_lineage=_EXPORT_SOURCE_NODATA,
        derived_from=["current_lst", "baseline_lst_yearly"],
        export_families=["current_lst", "baseline_lst_yearly"],
    ),
    "current_tvdi": _raster(
        "step5c_output_dir", "current_tvdi.tif", required=False,
        source_stage="predictors", semantic_group="tvdi",
        modeling_feature="current_tvdi_mean", boundary_lineage=_EXPORT_SOURCE_NODATA,
        derived_from=["current_ndvi", "current_lst"],
        export_families=["current_ndvi", "current_lst"], scientific_predictor=True,
    ),
    "tvdi_difference": _raster(
        "step5c_output_dir", "tvdi_difference.tif", required=False,
        source_stage="predictors", semantic_group="tvdi",
        modeling_feature="tvdi_difference_mean", boundary_lineage=_EXPORT_SOURCE_NODATA,
        derived_from=["current_tvdi", "baseline_ndvi_yearly"],
        export_families=["current_ndvi", "baseline_ndvi_yearly"],
        scientific_predictor=True,
    ),
    "modis_lst_mean": _raster(
        "modis_input_dir", "modis_lst_mean_celsius.tif", required=False,
        source_stage="predictors", semantic_group="continuous_temperature",
        modeling_feature=None,
        boundary_lineage=_lineage(nodata_edge={"provider": "raster_internal_nodata", "required_metadata": False}),
    ),
    "modis_lst_std": _raster(
        "modis_input_dir", "modis_lst_std_celsius.tif", required=False,
        source_stage="predictors", semantic_group="continuous_temperature",
        modeling_feature=None,
        boundary_lineage=_lineage(nodata_edge={"provider": "raster_internal_nodata", "required_metadata": False}),
    ),
    "elevation": _raster(
        "dem_input_dir", "elevation.tif", required=False, source_stage="predictors",
        semantic_group="terrain", modeling_feature="elevation_mean",
        boundary_lineage=_EXPORT_SOURCE_NODATA, export_families=["elevation"],
        scientific_predictor=True,
    ),
    "slope": _raster(
        "dem_input_dir", "slope.tif", required=False, source_stage="predictors",
        semantic_group="terrain", modeling_feature="slope_mean",
        boundary_lineage=_EXPORT_SOURCE_NODATA, export_families=["slope"],
        scientific_predictor=True,
    ),
    "downscaled_lst": _raster(
        "step7d_output_dir", "downscaled_lst_celsius.tif", required=False,
        source_stage="step7", semantic_group="continuous_temperature",
        modeling_feature="downscaled_lst_mean",
        boundary_lineage=_lineage(
            processing_window={"provider": "step7_inference_windows", "required_metadata": True},
            nodata_edge={"provider": "raster_internal_nodata", "required_metadata": False},
        ),
        producer_manifest=("step7d_output_dir", "downscaling_prediction_metadata.json"),
        scientific_predictor=True,
    ),
    "fused_lst": _raster(
        "step7e_output_dir", "fused_lst_celsius.tif", required=False,
        source_stage="step7", semantic_group="continuous_temperature",
        modeling_feature="fused_lst_mean",
        boundary_lineage=_lineage(
            processing_window={"provider": "step7_inference_windows", "required_metadata": True},
            observed_gapfill_transition={"provider": "step7e_source_mask", "required_metadata": True},
            nodata_edge={"provider": "raster_internal_nodata", "required_metadata": False},
        ),
        producer_manifest=("step7e_output_dir", "fused_lst_metadata.json"),
        scientific_predictor=True,
    ),
}


_SEMANTIC_IDENTITIES = {
    "current_lst": "current_period_lst_celsius",
    "current_ndvi": "current_period_ndvi",
    "baseline_lst_mean": "baseline_lst_mean_celsius",
    "baseline_lst_std": "baseline_lst_std_celsius",
    "baseline_ndvi_mean": "baseline_ndvi_mean",
    "baseline_ndvi_std": "baseline_ndvi_std",
    "lst_anomaly": "absolute_lst_anomaly_celsius",
    "anomaly_zscore": "standardized_lst_anomaly",
    "current_tvdi": "current_period_tvdi",
    "tvdi_difference": "tvdi_difference",
    "modis_lst_mean": "modis_lst_mean_celsius",
    "modis_lst_std": "modis_lst_std_celsius",
    "elevation": "elevation",
    "slope": "slope",
    "downscaled_lst": "downscaled_lst_celsius",
    "fused_lst": "fused_lst_celsius",
}

for _product_key, _product_spec in PRODUCT_REGISTRY_V2.items():
    _product_spec["semantic_identity"] = _SEMANTIC_IDENTITIES[_product_key]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def seam_audit_v2_config(experiment_id: str) -> dict[str, Any]:
    config = deepcopy(DEFAULT_SEAM_AUDIT_V2_CONFIG)
    override = get_experiment(experiment_id).get("seam_audit_v2", {})
    _deep_update(config, override)
    return config


def _deep_update(target: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


def _grid_signature(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        import rasterio

        with rasterio.open(path) as src:
            return {
                "crs": str(src.crs), "transform": list(src.transform)[:6],
                "width": src.width, "height": src.height, "nodata": src.nodata,
            }
    except Exception:
        return None


def _producer_path(ctx: dict[str, Any], key: str, spec: dict[str, Any]) -> tuple[Path | None, str | None, Path | None]:
    manifest_spec = spec.get("producer_manifest")
    if not manifest_spec:
        if key in {"current_lst", "baseline_lst_mean", "baseline_lst_std", "anomaly_zscore"}:
            manifest_spec = ("step5_output_dir", "step5_metadata.json")
        elif key in {"current_tvdi", "tvdi_difference"}:
            manifest_spec = ("step5c_output_dir", "step5c_metadata.json")
        elif key in {"elevation", "slope"}:
            manifest_spec = ("dem_input_dir", "dem_metadata.json")
        elif key in {"modis_lst_mean", "modis_lst_std"}:
            manifest_spec = ("modis_input_dir", "modis_metadata.json")
        elif key == "current_ndvi":
            manifest_spec = ("output_root", "predictor_export_metadata.json")
        else:
            return None, None, None
    context_key, filename = manifest_spec
    manifest_path = Path(ctx[context_key]) / filename
    metadata = _read_json(manifest_path)
    if not metadata:
        return None, None, manifest_path
    candidate: str | None = None
    if key == "downscaled_lst":
        candidate = metadata.get("output_paths", {}).get("predicted")
    elif key == "fused_lst":
        candidate = metadata.get("output_paths", {}).get("fused")
    elif key in {"current_lst", "baseline_lst_mean", "baseline_lst_std", "anomaly_zscore"}:
        output_key = {
            "current_lst": "current_median", "baseline_lst_mean": "baseline_mean",
            "baseline_lst_std": "baseline_std", "anomaly_zscore": "anomaly_zscore",
        }[key]
        candidate = metadata.get("outputs", {}).get(output_key)
        if candidate and not Path(candidate).is_absolute():
            candidate = str(manifest_path.parent / candidate)
    elif key in {"current_tvdi", "tvdi_difference"}:
        candidate = metadata.get("outputs", {}).get(key)
    elif key in {"elevation", "slope"}:
        candidate = metadata.get("output_files", {}).get(key)
    elif key in {"modis_lst_mean", "modis_lst_std"}:
        candidate = metadata.get("output_files", {}).get("mean" if key.endswith("mean") else "std")
    elif key == "current_ndvi":
        candidate = metadata.get("exports", {}).get("current_ndvi", {}).get("path")
    return (Path(candidate), "producing_stage_manifest", manifest_path) if candidate else (None, None, manifest_path)


def _modeling_feature_resolution(
    ctx: dict[str, Any], spec: dict[str, Any],
) -> tuple[bool, str, Path | None, str | None]:
    """Read Step8A provenance without treating it as native artifact identity."""
    feature = spec.get("modeling_feature")
    if not feature:
        return False, "not_applicable", None, None
    stats_path = Path(ctx["step8a_output_dir"]) / "step8a_dataset_stats.json"
    dataset_path = Path(ctx["step8a_output_dir"]) / "step8a_500m_modeling_dataset.parquet"
    stats = _read_json(stats_path)
    if not stats or not dataset_path.exists():
        return False, "missing_step8a_dataset_or_manifest", stats_path, None
    prefix = feature.removesuffix("_mean")
    valid_count = stats.get("feature_valid_counts", {}).get(prefix)
    source_path = stats.get("predictor_paths", {}).get(prefix)
    available = isinstance(valid_count, (int, float)) and valid_count > 0
    return available, "step8a_dataset_stats", stats_path, source_path


def _missing_resolution_status(spec: dict[str, Any]) -> str:
    if spec["native_artifact_required"]:
        return "missing_required_artifact"
    if not spec.get("pipeline_produces_native_artifact", True):
        if spec.get("artifact_kind") == "derived_modeling_feature":
            return "missing_optional_native_artifact"
        return "not_produced_optional"
    return "missing_expected_artifact"


def _collection_member(ctx: dict[str, Any], family: str, year: int) -> tuple[Path, str, Path | None]:
    metadata_path = Path(ctx["output_root"]) / "predictor_export_metadata.json"
    metadata = _read_json(metadata_path) or {}
    key = f"{family.removesuffix('_yearly')}_{year}"
    candidate = metadata.get("exports", {}).get(key, {}).get("path")
    if candidate:
        return Path(candidate), "predictor_export_metadata", metadata_path
    base_key = "ndvi_baseline_dir" if family == "baseline_ndvi_yearly" else "baseline_input_dir"
    prefix = "ndvi_baseline" if family == "baseline_ndvi_yearly" else f"{ctx['experiment_id']}_landsat_baseline"
    matches = sorted(Path(ctx[base_key]).glob(f"{prefix}_{year}-*.tif"))
    if len(matches) == 1:
        return matches[0], "validated_collection_resolver", None
    expected = Path(ctx[base_key]) / f"{prefix}_{year}-UNRESOLVED.tif"
    return expected, "missing", None


def resolve_product_registry_v2(
    ctx: dict[str, Any], requested_products: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requested = (
        requested_products
        if requested_products is not None
        else seam_audit_v2_config(ctx["experiment_id"])["products"]
    )
    allowed = set(PRODUCT_REGISTRY_V2) | {"baseline_ndvi_yearly", "baseline_lst_yearly"}
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError(f"Unknown seam-audit V2 product key(s): {unknown}")

    products: list[dict[str, Any]] = []
    for requested_key in requested:
        if requested_key in {"baseline_ndvi_yearly", "baseline_lst_yearly"}:
            family = requested_key
            semantic = "ndvi" if "ndvi" in family else "continuous_temperature"
            model_feature = None
            for year in ctx["baseline_years"]:
                path, method, manifest = _collection_member(ctx, family, int(year))
                key = f"{family.removesuffix('_yearly')}_{year}"
                spec = _raster(
                    "ndvi_baseline_dir" if "ndvi" in family else "baseline_input_dir",
                    path.name, required=False, source_stage="predictors",
                    semantic_group=semantic, modeling_feature=model_feature,
                    boundary_lineage=_EXPORT_SOURCE_NODATA, export_families=[key],
                )
                spec.update({
                    "product_key": key, "product_family": family,
                    "collection_family": family, "instance_id": str(year), "year": int(year),
                    "semantic_identity": f"{family.removesuffix('_yearly')}_{year}",
                    "expected_native_artifact_path": path,
                    "path": path if path.exists() else None,
                    "native_artifact_path": path if path.exists() else None,
                    "exists": path.exists(),
                    "resolution_status": "resolved" if path.exists() else "missing_expected_artifact",
                    "native_resolution_status": "resolved" if path.exists() else "missing_expected_artifact",
                    "resolution_method": method if path.exists() else "missing_after_exact_semantic_resolution",
                    "producer_manifest_path": manifest,
                    "modeling_feature_available": False,
                    "modeling_feature_resolution_method": "not_applicable",
                    "modeling_feature_manifest": None,
                    "modeling_feature_source_path": None,
                })
                products.append(spec)
            continue

        spec = deepcopy(PRODUCT_REGISTRY_V2[requested_key])
        producer_path, method, manifest = _producer_path(ctx, requested_key, spec)
        confirmed = Path(ctx[spec["context_key"]]) / spec["filename"]
        if producer_path is not None and producer_path.exists():
            path: Path | None = producer_path
            resolution_method = method or "producing_stage_manifest"
        elif confirmed.exists():
            path = confirmed
            resolution_method = "canonical_path_confirmed_by_producer_code"
        else:
            path = None
            resolution_method = "missing_after_exact_semantic_resolution"
        resolution_status = "resolved" if path is not None else _missing_resolution_status(spec)
        feature_available, feature_method, feature_manifest, feature_source = _modeling_feature_resolution(ctx, spec)
        spec.update({
            "product_key": requested_key,
            "expected_native_artifact_path": producer_path or confirmed,
            "path": path, "native_artifact_path": path,
            "exists": path is not None,
            "resolution_status": resolution_status,
            "native_resolution_status": resolution_status,
            "resolution_method": resolution_method,
            "producer_manifest_path": manifest,
            "modeling_feature_available": feature_available,
            "modeling_feature_resolution_method": feature_method,
            "modeling_feature_manifest": feature_manifest,
            "modeling_feature_source_path": feature_source,
        })
        products.append(spec)
    detect_artifact_identity_conflicts(products)
    resolutions = [_resolution_row(spec) for spec in products]
    return products, resolutions


def detect_artifact_identity_conflicts(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag different semantic identities sharing a path without an explicit alias."""
    by_path: dict[str, list[dict[str, Any]]] = {}
    for product in products:
        product["identity_conflict"] = False
        product["conflicting_product_keys"] = []
        path = product.get("native_artifact_path")
        if path is not None:
            by_path.setdefault(str(Path(path).resolve()), []).append(product)
    conflicts: list[dict[str, Any]] = []
    for resolved_path, group in by_path.items():
        for index, left in enumerate(group):
            for right in group[index + 1:]:
                aliases_match = (
                    left.get("explicit_alias_group") is not None
                    and left.get("explicit_alias_group") == right.get("explicit_alias_group")
                )
                if left.get("semantic_identity") == right.get("semantic_identity") or aliases_match:
                    continue
                left["identity_conflict"] = right["identity_conflict"] = True
                left["conflicting_product_keys"].append(right["product_key"])
                right["conflicting_product_keys"].append(left["product_key"])
                conflicts.append({
                    "resolution_status": "artifact_identity_conflict",
                    "resolved_path": resolved_path,
                    "product_keys": sorted([left["product_key"], right["product_key"]]),
                    "semantic_identities": sorted([left["semantic_identity"], right["semantic_identity"]]),
                    "message": (
                        "Artifact identity conflict: product keys "
                        f"'{left['product_key']}' and '{right['product_key']}' resolved to the same "
                        "file but have different semantic identities. An explicit alias is required "
                        "if this is intentional."
                    ),
                })
    for product in products:
        product["conflicting_product_keys"] = sorted(set(product["conflicting_product_keys"]))
        if product["identity_conflict"]:
            product["resolution_status"] = "artifact_identity_conflict"
    return conflicts


def _resolution_row(spec: dict[str, Any]) -> dict[str, Any]:
    path = spec.get("native_artifact_path")
    return {
        "product_key": spec["product_key"],
        "semantic_identity": spec["semantic_identity"],
        "artifact_kind": spec["artifact_kind"],
        "resolution_status": spec["resolution_status"],
        "resolution_method": spec["resolution_method"],
        "producer_stage": spec["source_stage"],
        "producer_manifest": str(spec.get("producer_manifest_path") or ""),
        "resolved_path": str(path) if path is not None else None,
        "path_exists": path is not None and Path(path).exists(),
        "grid_signature": json.dumps(_grid_signature(Path(path)), sort_keys=True) if path is not None else None,
        "explicit_alias_group": spec.get("explicit_alias_group"),
        "identity_conflict": bool(spec.get("identity_conflict")),
        "conflicting_product_keys": spec.get("conflicting_product_keys", []),
        "native_artifact_available": path is not None,
        "modeling_feature_available": bool(spec.get("modeling_feature_available")),
        "modeling_feature": spec.get("modeling_feature"),
        "modeling_feature_source_product": spec.get("modeling_feature_source_product"),
        "modeling_feature_semantic_identity": spec.get("modeling_feature_semantic_identity"),
        "modeling_feature_source_path": spec.get("modeling_feature_source_path"),
        "derived_from": spec.get("derived_from", []),
        "collection_family": spec.get("collection_family"),
        "instance_id": spec.get("instance_id"),
        "year": spec.get("year"),
        "member_paths": json.dumps([str(path)]) if path is not None and spec.get("collection_family") else None,
    }


def qa_output_dir_v2(ctx: dict[str, Any]) -> Path:
    return Path(ctx["output_root"]) / "qa" / "seam_audit" / AUDIT_VERSION
