#!/usr/bin/env python3
"""Experiment-aware, read-only runner for Seam Audit V2."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_geom

from core.experiment_context import build_experiment_context
from core.io_utils import setup_logger
from core.seam_audit_v2_config import (
    AUDIT_VERSION,
    SCHEMA_VERSION,
    qa_output_dir_v2,
    resolve_product_registry_v2,
    seam_audit_v2_config,
)
from src.seam_audit_v2 import (
    INCOMPLETE,
    blocker_and_rerun,
    boundary_row,
    canonical_grid_info,
    export_tile_boundaries,
    measure_gapfill_transition,
    measure_modeling_boundary,
    measure_native_boundary,
    processing_window_boundaries,
    same_boundary_propagation,
    scan_nodata_coverage,
    source_scene_boundaries,
    summarize_product,
)


log, log_file = setup_logger("seam_audit_v2")


class SeamAuditV2StageNotReady(SystemExit):
    """Configured V2 audit cannot start."""


def _parse_csv(value: str | list[str] | None) -> list[str] | None:
    if value is None or isinstance(value, list):
        return value
    return [item.strip() for item in value.split(",") if item.strip()]


def _source_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _v1_hashes(ctx: dict[str, Any]) -> dict[str, str]:
    root = Path(ctx["output_root"]) / "qa" / "seam_audit" / "v1"
    result: dict[str, str] = {}
    if root.exists():
        for path in sorted(p for p in root.iterdir() if p.is_file()):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[str(path)] = digest.hexdigest()
    return result


def _provider_preview(ctx: dict[str, Any], product: dict[str, Any]) -> dict[str, Any]:
    available: dict[str, Any] = {}
    for boundary_type, lineage in product["boundary_lineage"].items():
        provider = lineage["provider"]
        if provider == "export_tile_footprints":
            families = product.get("export_families", [])
            expanded = []
            for family in families:
                if family.endswith("_yearly"):
                    prefix = family.removesuffix("_yearly")
                    expanded.extend(f"{prefix}_{year}" for year in ctx["baseline_years"])
                else:
                    expanded.append(family)
            paths = [Path(ctx["data_root"]) / "_tiles" / family for family in expanded]
            available[boundary_type] = {
                "provider": provider, "available": any(p.exists() for p in paths),
                "metadata": [str(p) for p in paths],
            }
        elif provider == "step7_inference_windows":
            path = (
                Path(ctx["step7d_output_dir"]) / "downscaling_prediction_metadata.json"
                if product["product_key"] == "downscaled_lst"
                else Path(ctx["step7e_output_dir"]) / "fused_lst_metadata.json"
            )
            available[boundary_type] = {"provider": provider, "available": path.exists(), "metadata": str(path)}
        elif provider == "step7e_source_mask":
            path = Path(ctx["step7e_output_dir"]) / "fused_lst_source_mask.tif"
            available[boundary_type] = {"provider": provider, "available": path.exists(), "metadata": str(path)}
        elif provider == "source_scene_provenance":
            versioned = Path(ctx["output_root"]) / "qa" / "source_scene_provenance" / "v1"
            root = Path(ctx["output_root"]) / "provenance"
            paths = [versioned / "scene_boundaries.geojson", versioned / "scene_manifest.parquet"] + [root / name for name in ("scene_provenance.tif", "source_scene_id.tif", "scene_boundary.geojson", "scene_manifest.parquet")]
            available[boundary_type] = {"provider": provider, "available": any(p.exists() for p in paths), "metadata": [str(p) for p in paths]}
        else:
            available[boundary_type] = {
                "provider": provider, "available": bool(product["exists"]),
                "metadata": str(product["path"]) if product.get("path") is not None else None,
            }
    return available


def build_dry_run_plan(
    experiment_id: str, products: list[str] | None = None,
    scales: list[str] | None = None,
) -> dict[str, Any]:
    ctx = build_experiment_context(experiment_id)
    config = seam_audit_v2_config(experiment_id)
    if products is not None:
        config["products"] = products
    if scales is not None:
        config["audit_scales"] = scales
    registry, resolutions = resolve_product_registry_v2(ctx, config["products"])
    canonical, canonical_reason = canonical_grid_info(ctx)
    providers = {p["product_key"]: _provider_preview(ctx, p) for p in registry}
    collections: dict[str, list[str]] = {}
    for product in registry:
        if product.get("collection_family"):
            collections.setdefault(product["collection_family"], []).append(product["product_key"])
    missing_metadata = sorted({
        boundary_type
        for product_providers in providers.values()
        for boundary_type, status in product_providers.items()
        if not status["available"] and status["provider"] != "raster_internal_nodata"
    })
    return {
        "ran": False, "dry_run": True, "reason": "dry_run",
        "experiment_id": experiment_id, "region_key": ctx["region_key"], "role": ctx["role"],
        "audit_version": AUDIT_VERSION, "schema_version": SCHEMA_VERSION,
        "resolved_products": [
            {
                "product_key": p["product_key"],
                "semantic_identity": p["semantic_identity"],
                "path": str(p["path"]) if p.get("path") is not None else None,
                "exists": p["exists"],
                "artifact_kind": p["artifact_kind"], "required_or_optional": p["required_or_optional"],
                "resolution_status": p["resolution_status"],
                "resolution_method": p["resolution_method"],
                "modeling_feature": p["modeling_feature"],
                "modeling_feature_available": p["modeling_feature_available"],
                "boundary_lineage": p["boundary_lineage"],
            }
            for p in registry
        ],
        "resolved_collections": collections,
        "artifact_resolution": resolutions,
        "artifact_identity_conflicts": [
            row for row in resolutions if row["resolution_status"] == "artifact_identity_conflict"
        ],
        "boundary_providers": providers,
        "missing_boundary_metadata": missing_metadata,
        "canonical_500m_grid": ({**canonical, "path": str(canonical["path"]), "transform": list(canonical["transform"])[:6]} if canonical else None),
        "canonical_500m_grid_reason": canonical_reason,
        "audit_scales": config["audit_scales"],
        "planned_output_namespace": str(qa_output_dir_v2(ctx)),
        "v1_namespace": str(Path(ctx["output_root"]) / "qa" / "seam_audit" / "v1"),
        "safety": "read-only audit; no source raster or V1 artifact will be modified",
    }


def _missing_rows(product: dict[str, Any], scales: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for boundary_type, lineage in product["boundary_lineage"].items():
        for scale in scales:
            if boundary_type == "nodata_edge" and scale == "modeling_500m":
                continue
            rows.append({
                "product_key": product["product_key"], "boundary_type": boundary_type,
                "provider": lineage["provider"], "native_or_modeling": scale,
                "scale": scale, "status": "insufficient_artifact",
                "continuous_jump_status": "insufficient_artifact" if boundary_type != "nodata_edge" else "not_applicable",
                "coverage_status": "insufficient_artifact" if boundary_type == "nodata_edge" else "not_applicable",
                "control_status": "insufficient_control_pairs",
                "reason": str(product.get("expected_native_artifact_path") or "native artifact unavailable"),
                "boundary_id": None, "lineage_id": None, "geometry_hash": None,
                "matched_native_boundary_id": None, "matched_modeling_boundary_id": None,
            })
    return rows


def _modeling_feature_only_rows(
    ctx: dict[str, Any], product: dict[str, Any], config: dict[str, Any],
    canonical: dict[str, Any] | None, dataset: pd.DataFrame | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit an existing Step8A feature without claiming a native product raster."""
    rows: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    feature = product.get("modeling_feature")
    source_path = product.get("modeling_feature_source_path")
    feature_available = (
        dataset is not None and feature and feature in dataset.columns
        and product.get("modeling_feature_available")
        and source_path and Path(source_path).exists()
    )
    if not feature_available:
        rows.extend(_missing_rows(product, ["modeling_500m"]))
        return rows, controls, boundary_rows
    cache: dict[tuple[int, int], tuple[float, bool]] = {}
    with rasterio.open(source_path, "r") as reference:
        for boundary_type, lineage in product["boundary_lineage"].items():
            if boundary_type == "nodata_edge":
                continue
            provider = lineage["provider"]
            if boundary_type == "export_tile":
                boundaries, status, reason = export_tile_boundaries(ctx, product, reference)
            elif boundary_type == "source_scene":
                boundaries, status, reason = source_scene_boundaries(ctx, product, reference)
            else:
                boundaries, status, reason = [], "not_applicable", None
            if status != "available":
                rows.extend(_unavailable_rows(
                    product, boundary_type, provider, status, reason, ["modeling_500m"],
                ))
                continue
            for boundary in boundaries:
                boundary_rows.append(boundary_row(boundary))
                measured, measured_controls = measure_modeling_boundary(
                    reference, boundary, product, config, canonical, dataset, cache,
                )
                rows.append(measured)
                controls.extend(measured_controls)
    return rows, controls, boundary_rows


def _unavailable_rows(
    product: dict[str, Any], boundary_type: str, provider: str,
    status: str, reason: str | None, scales: list[str],
) -> list[dict[str, Any]]:
    result = []
    for scale in scales:
        if boundary_type == "nodata_edge" and scale == "modeling_500m":
            continue
        result.append({
            "product_key": product["product_key"], "boundary_type": boundary_type,
            "provider": provider, "native_or_modeling": scale, "scale": scale,
            "status": status, "reason": reason, "boundary_id": None,
            "lineage_id": None, "geometry_hash": None,
            "matched_native_boundary_id": None, "matched_modeling_boundary_id": None,
            "control_status": "insufficient_control_pairs",
            "coverage_status": status if boundary_type == "nodata_edge" else "not_applicable",
            "continuous_jump_status": "not_applicable" if boundary_type == "nodata_edge" else status,
        })
    return result


def _hotspot(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("status") not in {"warn", "fail"} or not row.get("geometry_wkt"):
        return None
    body = row["geometry_wkt"].split("(", 1)[1].rsplit(")", 1)[0]
    coordinates = [[float(v) for v in point.strip().split()] for point in body.split(",")]
    geometry: dict[str, Any] = {"type": "LineString", "coordinates": coordinates}
    if row.get("native_crs") and row["native_crs"] != "EPSG:4326":
        geometry = transform_geom(row["native_crs"], "EPSG:4326", geometry)
    fields = (
        "product_key", "boundary_id", "lineage_id", "boundary_type", "provider",
        "native_or_modeling", "status", "valid_pair_count", "absolute_jump_median",
        "absolute_jump_p95", "median_jump_ratio", "p95_jump_ratio",
    )
    return {"type": "Feature", "geometry": geometry, "properties": {k: row.get(k) for k in fields}}


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Seam Audit V2 summary", "",
        f"- Experiment: `{summary['experiment_id']}`",
        f"- Overall status: **{summary['overall_status']}**",
        f"- Scientific blocker: `{str(summary['scientific_blocker']).lower()}`",
        f"- Assessment complete: `{str(summary['assessment_complete']).lower()}`",
        f"- Recommended rerun: `{summary['recommended_rerun_from_stage']}`",
        f"- Recommended action: `{summary['recommended_action']}`", "",
        "> Thresholds are initial QA heuristics, not formal statistical significance or causal attribution.",
        "", "## Products", "",
        "| Product | Status | Native | 500 m | Scope | Propagation | Complete |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    for key, item in summary["products"].items():
        lines.append(
            f"| `{key}` | {item['status']} | {item['native_status']} | {item['modeling_500m_status']} | "
            f"{item['conclusion_scope']} | {item['propagation']} | {item['assessment_complete']} |"
        )
    return "\n".join(lines) + "\n"


def _write_outputs(
    output_dir: Path, summary: dict[str, Any], product_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]], control_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]], resolution_rows: list[dict[str, Any]],
    hotspots: list[dict[str, Any]], manifest: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_json": output_dir / "seam_audit_summary.json",
        "summary_markdown": output_dir / "seam_audit_summary.md",
        "product_metrics": output_dir / "product_metrics.parquet",
        "boundary_segment_metrics": output_dir / "boundary_segment_metrics.parquet",
        "control_metrics": output_dir / "control_metrics.parquet",
        "boundary_registry": output_dir / "boundary_registry.parquet",
        "artifact_resolution": output_dir / "artifact_resolution.parquet",
        "hotspots": output_dir / "seam_hotspots.geojson",
        "manifest": output_dir / "manifest.json",
    }
    paths["summary_json"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    paths["summary_markdown"].write_text(_summary_markdown(summary), encoding="utf-8")
    pd.DataFrame(product_rows).to_parquet(paths["product_metrics"], index=False)
    pd.DataFrame(segment_rows).to_parquet(paths["boundary_segment_metrics"], index=False)
    pd.DataFrame(control_rows).to_parquet(paths["control_metrics"], index=False)
    pd.DataFrame(boundary_rows).to_parquet(paths["boundary_registry"], index=False)
    pd.DataFrame(resolution_rows).to_parquet(paths["artifact_resolution"], index=False)
    geojson = {"type": "FeatureCollection", "name": "seam_hotspots_v2", "features": hotspots}
    paths["hotspots"].write_text(json.dumps(geojson, indent=2, default=str), encoding="utf-8")
    manifest["output_paths"] = {key: str(path) for key, path in paths.items()}
    paths["manifest"].write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def main(
    experiment_id: str, dry_run: bool = False, force: bool = False,
    products: list[str] | str | None = None, scales: list[str] | str | None = None,
) -> dict[str, Any]:
    products = _parse_csv(products); scales = _parse_csv(scales)
    plan = build_dry_run_plan(experiment_id, products, scales)
    log.info("[seam-audit-v2] plan: %s", json.dumps(plan, indent=2, default=str))
    if dry_run:
        return plan
    ctx = build_experiment_context(experiment_id)
    config = seam_audit_v2_config(experiment_id)
    if products is not None:
        config["products"] = products
    if scales is not None:
        config["audit_scales"] = scales
    if set(config["audit_scales"]) - {"native", "modeling_500m"}:
        raise SeamAuditV2StageNotReady(f"Unsupported audit scale(s): {config['audit_scales']}")
    output_dir = qa_output_dir_v2(ctx)
    if output_dir.exists() and not force and (output_dir / "seam_audit_summary.json").exists():
        return {"ran": False, "reason": "already_exists", "summary_path": str(output_dir / "seam_audit_summary.json")}
    previous_manifest: dict[str, Any] | None = None
    previous_manifest_path = output_dir / "manifest.json"
    if force and previous_manifest_path.exists():
        try:
            previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_manifest = None
    v1_before = _v1_hashes(ctx)
    registry, resolution_rows = resolve_product_registry_v2(ctx, config["products"])
    identity_conflicts = [p for p in registry if p.get("identity_conflict")]
    if identity_conflicts:
        details = "; ".join(
            f"{p['product_key']} conflicts with {p['conflicting_product_keys']} at {p['path']}"
            for p in identity_conflicts
        )
        raise SeamAuditV2StageNotReady(f"artifact_identity_conflict: {details}")
    required_missing = [p for p in registry if p["resolution_status"] == "missing_required_artifact"]
    if required_missing:
        details = ", ".join(
            f"{p['product_key']}={p['expected_native_artifact_path']}" for p in required_missing
        )
        raise SeamAuditV2StageNotReady(
            f"seam-audit-v2 stage-not-ready; required product(s) missing: {details}"
        )
    canonical, canonical_reason = canonical_grid_info(ctx)
    dataset_path = Path(ctx["step8a_output_dir"]) / "step8a_500m_modeling_dataset.parquet"
    dataset = pd.read_parquet(dataset_path) if dataset_path.exists() and "modeling_500m" in config["audit_scales"] else None
    segment_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    registry_by_id: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}

    for product in registry:
        rows: list[dict[str, Any]] = []
        if not product["exists"]:
            if (
                product.get("artifact_kind") == "derived_modeling_feature"
                and product.get("modeling_feature")
            ):
                if "native" in config["audit_scales"]:
                    rows.extend(_missing_rows(product, ["native"]))
                if "modeling_500m" in config["audit_scales"]:
                    model_rows, model_controls, model_boundaries = _modeling_feature_only_rows(
                        ctx, product, config, canonical, dataset,
                    )
                    rows.extend(model_rows); control_rows.extend(model_controls)
                    for boundary in model_boundaries:
                        registry_by_id[boundary["boundary_id"]] = boundary
            else:
                rows.extend(_missing_rows(product, config["audit_scales"]))
            propagation = same_boundary_propagation(rows)
            for row in rows:
                row["propagation"] = propagation.get(row.get("boundary_id"), "insufficient_data")
            segment_rows.extend(rows)
            summaries[product["product_key"]] = summarize_product(product, rows, config)
            continue
        cell_cache: dict[tuple[int, int], tuple[float, bool]] = {}
        with rasterio.open(product["path"], "r") as src:
            for boundary_type, lineage in product["boundary_lineage"].items():
                provider = lineage["provider"]
                if boundary_type == "nodata_edge":
                    if "native" in config["audit_scales"]:
                        rows.append(scan_nodata_coverage(src, product, config))
                    continue
                if boundary_type == "observed_gapfill_transition":
                    gap_rows, gap_controls, boundaries, status, reason = measure_gapfill_transition(
                        ctx, src, product, config, canonical, dataset, cell_cache,
                    )
                    if status == "available":
                        rows.extend(gap_rows); control_rows.extend(gap_controls)
                        for boundary in boundaries:
                            registry_by_id[boundary.boundary_id] = boundary_row(boundary)
                    else:
                        rows.extend(_unavailable_rows(product, boundary_type, provider, status, reason, config["audit_scales"]))
                    continue
                if boundary_type == "processing_window":
                    boundaries, status, reason = processing_window_boundaries(ctx, product, src)
                elif boundary_type == "export_tile":
                    boundaries, status, reason = export_tile_boundaries(ctx, product, src)
                elif boundary_type == "source_scene":
                    boundaries, status, reason = source_scene_boundaries(ctx, product, src)
                else:
                    boundaries, status, reason = [], "not_applicable", None
                if status != "available":
                    rows.extend(_unavailable_rows(product, boundary_type, provider, status, reason, config["audit_scales"]))
                    continue
                for boundary in boundaries:
                    registry_by_id[boundary.boundary_id] = boundary_row(boundary)
                    if "native" in config["audit_scales"]:
                        native, controls = measure_native_boundary(src, boundary, product, config, boundaries)
                        rows.append(native); control_rows.extend(controls)
                    if "modeling_500m" in config["audit_scales"]:
                        modeling, controls = measure_modeling_boundary(
                            src, boundary, product, config, canonical, dataset, cell_cache,
                        )
                        rows.append(modeling); control_rows.extend(controls)
        propagation = same_boundary_propagation(rows)
        for row in rows:
            row["propagation"] = propagation.get(row.get("boundary_id"), "not_applicable")
        segment_rows.extend(rows)
        summaries[product["product_key"]] = summarize_product(product, rows, config)

    scientific_blocker, rerun, action = blocker_and_rerun(registry, summaries, segment_rows)
    propagating_fail = scientific_blocker and any(s["corroborated_fail"] and s["propagating_boundary_count"] for s in summaries.values())
    propagating_warn = scientific_blocker or any(s["corroborated_warn"] and s["propagating_boundary_count"] for s in summaries.values())
    optional_products_not_produced = sorted(
        p["product_key"] for p in registry if p["resolution_status"] == "not_produced_optional"
    )
    missing_required_artifacts = sorted(
        p["product_key"] for p in registry if p["resolution_status"] == "missing_required_artifact"
    )
    artifact_identity_conflicts = sorted(
        p["product_key"] for p in registry if p.get("identity_conflict")
    )
    product_by_key = {p["product_key"]: p for p in registry}
    boundary_types_with_evidence = {
        row["boundary_type"] for row in segment_rows
        if row.get("status") not in INCOMPLETE | {"not_applicable"}
    }
    missing_boundary_provenance = sorted({
        row["boundary_type"] for row in segment_rows
        if row.get("status") == "insufficient_boundary_metadata"
        and product_by_key.get(row.get("product_key"), {}).get("boundary_lineage", {})
            .get(row.get("boundary_type"), {}).get("required_metadata")
    } - boundary_types_with_evidence)
    assessment_incomplete_reasons: list[str] = []
    assessment_incomplete_reasons.extend(
        f"{boundary}_provenance_unavailable" for boundary in missing_boundary_provenance
    )
    assessment_incomplete_reasons.extend(
        f"missing_required_artifact:{key}" for key in missing_required_artifacts
    )
    assessment_incomplete_reasons.extend(
        f"artifact_identity_conflict:{key}" for key in artifact_identity_conflicts
    )
    if canonical is None and "modeling_500m" in config["audit_scales"]:
        assessment_incomplete_reasons.append("canonical_500m_grid_metadata_unavailable")
    for key, item in summaries.items():
        product = product_by_key[key]
        if item["status"] != "incomplete" or product["resolution_status"] == "not_produced_optional":
            continue
        if (
            product["resolution_status"] in {"missing_expected_artifact", "missing_required_artifact"}
            or (product.get("modeling_feature") and not product.get("modeling_feature_available"))
        ):
            assessment_incomplete_reasons.append(f"evaluable_product_unavailable:{key}")
    assessment_incomplete_reasons = sorted(set(assessment_incomplete_reasons))
    assessment_complete = not assessment_incomplete_reasons
    if propagating_fail:
        overall = "fail"
    elif propagating_warn or not assessment_complete:
        overall = "warn"
    elif summaries:
        overall = "pass"
    else:
        overall = "incomplete"
    available_types = sorted(boundary_types_with_evidence)
    unavailable_types = sorted({
        row["boundary_type"] for row in segment_rows if row.get("status") in INCOMPLETE
    } - set(available_types))
    summary = {
        "experiment_id": experiment_id, "region_key": ctx["region_key"], "role": ctx["role"],
        "audit_version": AUDIT_VERSION, "schema_version": SCHEMA_VERSION,
        "overall_status": overall, "scientific_blocker": scientific_blocker,
        "assessment_complete": assessment_complete,
        "assessment_incomplete": not assessment_complete,
        "assessment_incomplete_reasons": assessment_incomplete_reasons,
        "optional_products_not_produced": optional_products_not_produced,
        "artifact_identity_conflicts": artifact_identity_conflicts,
        "missing_required_artifacts": missing_required_artifacts,
        "missing_boundary_provenance": missing_boundary_provenance,
        "boundary_types_available": available_types,
        "boundary_types_unavailable": unavailable_types,
        "propagating_boundary_count": sum(s["propagating_boundary_count"] for s in summaries.values()),
        "native_only_boundary_count": sum(s["native_only_boundary_count"] for s in summaries.values()),
        "modeling_only_boundary_count": sum(s["modeling_only_boundary_count"] for s in summaries.values()),
        "recommended_rerun_from_stage": rerun, "recommended_action": action,
        "canonical_500m_grid_status": "available" if canonical else "insufficient_grid_metadata",
        "canonical_500m_grid_reason": canonical_reason,
        "decision_semantics": "initial QA heuristic; absolute AND local-control ratio; same-boundary propagation",
        "products": summaries,
    }
    product_rows = [{k: v for k, v in item.items() if k != "propagation_by_boundary"} for item in summaries.values()]
    hotspots = [item for item in (_hotspot(row) for row in segment_rows) if item is not None]
    manifest = {
        "experiment_id": experiment_id, "audit_version": AUDIT_VERSION,
        "schema_version": SCHEMA_VERSION, "engine_commit": _source_commit(),
        "v1_compatibility_note": "V1 code and outputs are retained unchanged; V2 uses an isolated namespace and schema.",
        "created_at": datetime.now(timezone.utc).isoformat(), "config_snapshot": config,
        "random_seed": config["random_seed"], "canonical_grid": (
            {**canonical, "path": str(canonical["path"]), "transform": list(canonical["transform"])[:6]} if canonical else None
        ),
        "software_versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "rasterio": rasterio.__version__,
        },
        "safety": {"source_rasters_opened_read_only": True, "raster_modification": False, "v1_modified": False},
    }
    if force:
        manifest["regenerated_at"] = datetime.now(timezone.utc).isoformat()
        manifest["previous_run"] = {
            "created_at": previous_manifest.get("created_at") if previous_manifest else None,
            "schema_version": previous_manifest.get("schema_version") if previous_manifest else None,
        }
    output_paths = _write_outputs(
        output_dir, summary, product_rows, segment_rows, control_rows,
        list(registry_by_id.values()), resolution_rows, hotspots, manifest,
    )
    v1_after = _v1_hashes(ctx)
    if v1_before != v1_after:
        raise RuntimeError("V1 preservation invariant violated: a V1 artifact changed during V2 run")
    return {"ran": True, "dry_run": False, "summary": summary, "output_paths": output_paths}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only experiment Seam Audit V2")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seam-products", help="Comma-separated V2 product keys")
    parser.add_argument("--seam-scales", help="native,modeling_500m")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    result = main(args.experiment, args.dry_run, args.force, args.seam_products, args.seam_scales)
    print(json.dumps(result, indent=2, default=str))
