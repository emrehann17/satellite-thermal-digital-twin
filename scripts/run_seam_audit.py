#!/usr/bin/env python3
"""Experiment-aware runner for the read-only seam audit stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import rasterio

from core.experiment_context import build_experiment_context
from core.io_utils import setup_logger
from core.regions import get_experiment
from core.seam_audit_config import (
    AUDIT_VERSION,
    qa_output_dir,
    resolve_product_registry,
    seam_audit_config,
)
from src.seam_audit import (
    STATUS_RANK,
    SeamAuditError,
    aggregate_product_status,
    discover_straight_boundaries,
    measure_segment_modeling,
    measure_segment_native,
    propagation_status,
    scan_categorical_boundaries,
    scan_gapfill_transitions,
    scan_nodata_edges,
    segment_geometry,
)

log, log_file = setup_logger("seam_audit")


class SeamAuditStageNotReady(SystemExit):
    """Required configured inputs are not available."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _parse_csv(value: str | list[str] | None) -> list[str] | None:
    if value is None or isinstance(value, list):
        return value
    return [item.strip() for item in value.split(",") if item.strip()]


def build_dry_run_plan(
    experiment_id: str, products: list[str] | None = None,
    scales: list[str] | None = None,
) -> dict[str, Any]:
    ctx = build_experiment_context(experiment_id)
    config = seam_audit_config(experiment_id)
    if products is not None:
        config["products"] = products
    if scales is not None:
        config["audit_scales"] = scales
    registry = resolve_product_registry(ctx, config["products"])
    found = [p["product_key"] for p in registry if p["exists"]]
    missing_optional = [
        p["product_key"] for p in registry
        if not p["exists"] and p["required_or_optional"] == "optional"
    ]
    missing_required = [
        p["product_key"] for p in registry
        if not p["exists"] and p["required_or_optional"] == "required"
    ]
    step7a_metadata = Path(ctx["step7a_output_dir"]) / "tiling_test_summary.json"
    export_metadata = Path(ctx["output_root"]) / "predictor_export_metadata.json"
    source_scene = Path(ctx["output_root"]) / "provenance" / "source_scene.tif"
    source_mask = Path(ctx["step7e_output_dir"]) / "fused_lst_source_mask.tif"
    return {
        "ran": False,
        "dry_run": True,
        "reason": "dry_run",
        "experiment_id": experiment_id,
        "region_key": ctx["region_key"],
        "role": ctx["role"],
        "enabled": bool(config["enabled"]),
        "resolved_products": [
            {
                "product_key": p["product_key"], "path": str(p["path"]),
                "exists": p["exists"], "required_or_optional": p["required_or_optional"],
                "source_stage": p["source_stage"],
            }
            for p in registry
        ],
        "products_found": found,
        "missing_optional_products": missing_optional,
        "required_missing_products": missing_required,
        "available_boundary_metadata": {
            "export_tile": export_metadata.exists(),
            "processing_window": step7a_metadata.exists(),
            "source_scene": source_scene.exists(),
            "nodata_edge": bool(found),
            "observed_gapfill_transition": source_mask.exists(),
        },
        "audit_scales": config["audit_scales"],
        "planned_output_namespace": str(qa_output_dir(ctx)),
        "safety": "read-only audit; no source raster will be modified",
    }


def _worst(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "insufficient_boundary_metadata"
    return max((r.get("status", "error") for r in rows), key=lambda s: STATUS_RANK.get(s, 5))


def _product_metric_rows(product_key: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("boundary_type", "unknown"), row.get("scale", "native"))].append(row)
    result = []
    for (boundary_type, scale), values in sorted(groups.items()):
        medians = [v["absolute_jump_median"] for v in values if v.get("absolute_jump_median") is not None]
        p95s = [v["absolute_jump_p95"] for v in values if v.get("absolute_jump_p95") is not None]
        ratios = [v["median_jump_ratio"] for v in values if v.get("median_jump_ratio") is not None]
        result.append({
            "product_key": product_key,
            "boundary_type": boundary_type,
            "scale": scale,
            "status": _worst(values),
            "segment_count": len(values),
            "valid_pair_count": int(sum(v.get("valid_pair_count", 0) or 0 for v in values)),
            "invalid_pair_count": int(sum(v.get("invalid_pair_count", 0) or 0 for v in values)),
            "absolute_jump_median_max": max(medians) if medians else None,
            "absolute_jump_p95_max": max(p95s) if p95s else None,
            "median_jump_ratio_max": max(ratios) if ratios else None,
        })
    return result


def _stage_rank(stage: str) -> int:
    return {"gate": 0, "predictors": 1, "step7": 2, "seam-audit": 3, "step8": 4, "transfer": 5}.get(stage, 99)


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Seam audit summary", "",
        f"- Experiment: `{summary['experiment_id']}`",
        f"- Audit version: `{summary['audit_version']}`",
        f"- Overall status: **{summary['overall_status']}**",
        f"- Scientific blocker (heuristic QA): `{str(summary['scientific_blocker']).lower()}`",
        f"- Earliest affected stage: `{summary['earliest_affected_stage']}`",
        f"- Recommended rerun from stage: `{summary['recommended_rerun_from_stage']}`",
        "",
        "> This is a boundary-associated discontinuity QA audit. Thresholds are initial heuristics, not formal statistical significance or causal attribution.",
        "", "## Products", "",
        "| Product | Native | Modeling 500 m | Propagation |", "|---|---:|---:|---|",
    ]
    for key, item in summary["products"].items():
        lines.append(f"| `{key}` | {item['native_status']} | {item['modeling_500m_status']} | {item['propagation']} |")
    if summary["boundary_types_unavailable"]:
        lines.extend(["", "## Unavailable boundary metadata", ""])
        lines.extend(f"- `{name}`" for name in summary["boundary_types_unavailable"])
    return "\n".join(lines) + "\n"


def _write_outputs(
    output_dir: Path, summary: dict[str, Any], product_metrics: list[dict[str, Any]],
    segment_metrics: list[dict[str, Any]], control_metrics: list[dict[str, Any]],
    hotspots: list[dict[str, Any]], manifest: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_json": output_dir / "seam_audit_summary.json",
        "summary_markdown": output_dir / "seam_audit_summary.md",
        "product_metrics": output_dir / "product_metrics.parquet",
        "boundary_segment_metrics": output_dir / "boundary_segment_metrics.parquet",
        "control_metrics": output_dir / "control_metrics.parquet",
        "hotspots": output_dir / "seam_hotspots.geojson",
        "manifest": output_dir / "manifest.json",
    }
    paths["summary_json"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    paths["summary_markdown"].write_text(_summary_markdown(summary), encoding="utf-8")
    pd.DataFrame(product_metrics).to_parquet(paths["product_metrics"], index=False)
    pd.DataFrame(segment_metrics).to_parquet(paths["boundary_segment_metrics"], index=False)
    pd.DataFrame(control_metrics).to_parquet(paths["control_metrics"], index=False)
    geojson = {"type": "FeatureCollection", "name": "seam_hotspots", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}, "features": hotspots}
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
    log.info("[seam-audit] plan: %s", json.dumps(plan, indent=2, default=str))
    if dry_run:
        return plan

    ctx = build_experiment_context(experiment_id)
    exp = get_experiment(experiment_id)
    config = seam_audit_config(experiment_id)
    if products is not None:
        config["products"] = products
    if scales is not None:
        config["audit_scales"] = scales
    unknown_scales = set(config["audit_scales"]) - {"native", "modeling_500m"}
    if unknown_scales:
        raise SeamAuditStageNotReady(f"Unsupported seam audit scale(s): {sorted(unknown_scales)}")
    if not config["enabled"]:
        return {"ran": False, "dry_run": False, "reason": "disabled_by_config", "experiment_id": experiment_id}

    output_dir = qa_output_dir(ctx)
    if output_dir.exists() and not force:
        summary_path = output_dir / "seam_audit_summary.json"
        if summary_path.exists():
            return {"ran": False, "reason": "already_exists", "summary_path": str(summary_path)}

    registry = resolve_product_registry(ctx, config["products"])
    required_missing = [p for p in registry if not p["exists"] and p["required_or_optional"] == "required"]
    if required_missing:
        details = ", ".join(f"{p['product_key']}={p['path']}" for p in required_missing)
        raise SeamAuditStageNotReady(f"seam-audit stage-not-ready; required product(s) missing: {details}")

    dataset_path = Path(ctx["step8a_output_dir"]) / "step8a_500m_modeling_dataset.parquet"
    modeling_dataset: pd.DataFrame | None = None
    if "modeling_500m" in config["audit_scales"] and dataset_path.exists():
        modeling_dataset = pd.read_parquet(dataset_path)

    all_segment_rows: list[dict[str, Any]] = []
    all_control_rows: list[dict[str, Any]] = []
    all_product_rows: list[dict[str, Any]] = []
    hotspots: list[dict[str, Any]] = []
    product_summary: dict[str, Any] = {}
    boundary_availability: dict[str, bool] = {key: False for key in config["boundary_types"]}
    boundary_sources: dict[str, set[str]] = defaultdict(set)
    input_paths: list[Path] = []

    for product in registry:
        key = product["product_key"]
        if not product["exists"]:
            missing_rows = []
            for boundary_type in config["boundary_types"]:
                for scale in config["audit_scales"]:
                    missing_rows.append({
                        "product_key": key, "boundary_type": boundary_type, "scale": scale,
                        "status": "skipped_missing_optional", "reason": str(product["path"]),
                    })
            all_segment_rows.extend(missing_rows)
            all_product_rows.extend(_product_metric_rows(key, missing_rows))
            product_summary[key] = {
                "path": str(product["path"]), "required_or_optional": "optional",
                "native_status": "skipped_missing_optional",
                "modeling_500m_status": "skipped_missing_optional",
                "propagation": "insufficient_data", "source_stage": product["source_stage"],
            }
            continue

        input_paths.append(Path(product["path"]))
        product_rows: list[dict[str, Any]] = []
        seed_sequence = np.random.SeedSequence([int(config["random_seed"]), sum(map(ord, key))])
        rng = np.random.default_rng(seed_sequence)
        with rasterio.open(product["path"], "r") as src:
            for boundary_type in config["boundary_types"]:
                if boundary_type not in product["supported_boundary_types"]:
                    row = {"product_key": key, "boundary_type": boundary_type, "scale": "native", "status": "not_applicable"}
                    product_rows.append(row)
                    continue
                try:
                    if boundary_type == "source_scene":
                        configured = config.get("source_scene_provenance_path")
                        provenance = Path(configured) if configured else Path(ctx["output_root"]) / "provenance" / "source_scene.tif"
                        if not provenance.is_absolute():
                            provenance = Path(ctx["output_root"]) / provenance
                        row, control = scan_categorical_boundaries(src, provenance, product, config, rng)
                        product_rows.append({"product_key": key, **row})
                        if control is not None:
                            all_control_rows.append({"product_key": key, **control})
                            boundary_availability[boundary_type] = True
                            boundary_sources[boundary_type].add(str(provenance))
                        if "modeling_500m" in config["audit_scales"]:
                            product_rows.append({
                                "product_key": key, "boundary_type": boundary_type,
                                "boundary_id": "scene_id_change", "scale": "modeling_500m",
                                "status": "insufficient_boundary_metadata",
                                "reason": "modeling-scale categorical scene provenance is unavailable",
                            })
                        continue
                    if boundary_type == "nodata_edge":
                        row = {"product_key": key, **scan_nodata_edges(src, product, config)}
                        product_rows.append(row)
                        boundary_availability[boundary_type] = True
                        boundary_sources[boundary_type].add("raster_nodata_mask")
                        continue
                    if boundary_type == "observed_gapfill_transition":
                        source_mask = Path(ctx["step7e_output_dir"]) / "fused_lst_source_mask.tif"
                        row = {"product_key": key, **scan_gapfill_transitions(src, source_mask, product, config)}
                        product_rows.append(row)
                        if row["status"] != "insufficient_boundary_metadata":
                            boundary_availability[boundary_type] = True
                            boundary_sources[boundary_type].add(str(source_mask))
                        continue

                    segments, availability, reason = discover_straight_boundaries(ctx, product, boundary_type, src)
                    if availability != "available":
                        for scale in config["audit_scales"]:
                            product_rows.append({
                                "product_key": key, "boundary_type": boundary_type,
                                "scale": scale, "status": availability, "reason": reason,
                            })
                        continue
                    boundary_availability[boundary_type] = True
                    for segment in segments:
                        boundary_sources[boundary_type].add(segment.metadata_source)
                        if "native" in config["audit_scales"]:
                            metrics, control = measure_segment_native(src, segment, product, config, rng, segments)
                            product_rows.append({"product_key": key, **metrics})
                            all_control_rows.append({"product_key": key, **control})
                        if "modeling_500m" in config["audit_scales"]:
                            metrics, control = measure_segment_modeling(src, segment, product, config, rng, modeling_dataset)
                            product_rows.append({"product_key": key, **metrics})
                            all_control_rows.append({"product_key": key, **control})
                except SeamAuditError as exc:
                    product_rows.append({
                        "product_key": key, "boundary_type": boundary_type,
                        "scale": "native", "status": "error", "reason": str(exc),
                    })

            for row in product_rows:
                if row.get("status") in {"warn", "fail"} and row.get("index") is not None:
                    geometry = segment_geometry(row, src)
                    props = {
                        field: row.get(field) for field in (
                            "product_key", "boundary_type", "boundary_id", "scale", "status",
                            "valid_pair_count", "absolute_jump_median", "absolute_jump_p95",
                            "median_jump_ratio", "p95_jump_ratio",
                        )
                    }
                    props["experiment_id"] = experiment_id
                    hotspots.append({"type": "Feature", "geometry": geometry, "properties": props})

        all_segment_rows.extend(product_rows)
        all_product_rows.extend(_product_metric_rows(key, product_rows))
        native_status = aggregate_product_status(product_rows, "native")
        modeling_status = aggregate_product_status(product_rows, "modeling_500m")
        product_summary[key] = {
            "path": str(product["path"]),
            "required_or_optional": product["required_or_optional"],
            "source_stage": product["source_stage"],
            "native_status": native_status,
            "modeling_500m_status": modeling_status,
            "propagation": propagation_status(native_status, modeling_status),
        }

    audited = [k for k, v in product_summary.items() if v["native_status"] != "skipped_missing_optional"]
    failed = [k for k, v in product_summary.items() if "fail" in {v["native_status"], v["modeling_500m_status"]}]
    warned = [k for k, v in product_summary.items() if "warn" in {v["native_status"], v["modeling_500m_status"]}]
    overall = _worst(all_segment_rows)
    if overall == "pass" and not any(boundary_availability.values()):
        overall = "insufficient_boundary_metadata"
    affected = [
        v["source_stage"] for v in product_summary.values()
        if v["native_status"] in {"warn", "fail"} or v["modeling_500m_status"] in {"warn", "fail"}
    ]
    earliest = min(affected, key=_stage_rank) if affected else None
    propagated = any(v["propagation"] in {"propagates_to_500m", "modeling_only"} for v in product_summary.values())
    recommended = earliest if propagated else None
    summary = {
        "experiment_id": experiment_id, "region_key": ctx["region_key"], "role": ctx["role"],
        "audit_version": AUDIT_VERSION, "overall_status": overall,
        "products_audited": audited, "products_failed": failed, "products_warned": warned,
        "boundary_types_available": sorted(k for k, v in boundary_availability.items() if v),
        "boundary_types_unavailable": sorted(k for k, v in boundary_availability.items() if not v),
        "earliest_affected_stage": earliest,
        "recommended_rerun_from_stage": recommended,
        "scientific_blocker": bool(failed and propagated),
        "decision_semantics": "initial QA heuristic; not formal significance or causal attribution",
        "products": product_summary,
    }

    input_sha256 = {str(path): _sha256(path) for path in input_paths}
    reference = next((p for p in registry if p["product_key"] == "current_lst" and p["exists"]), None)
    reference_grid = None
    if reference:
        with rasterio.open(reference["path"]) as src:
            reference_grid = {
                "path": str(reference["path"]), "crs": str(src.crs),
                "transform": list(src.transform)[:6], "width": src.width, "height": src.height,
            }
    manifest = {
        "experiment_id": experiment_id, "region_key": ctx["region_key"], "role": exp["role"],
        "audit_version": AUDIT_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": _source_commit(), "config_snapshot": config,
        "random_seed": config["random_seed"], "products_requested": config["products"],
        "products_found": [p["product_key"] for p in registry if p["exists"]],
        "products_missing": [p["product_key"] for p in registry if not p["exists"]],
        "input_paths": [str(path) for path in input_paths], "input_sha256": input_sha256,
        "reference_grid": reference_grid,
        "boundary_sources": {k: sorted(v) for k, v in boundary_sources.items()},
        "thresholds": config["thresholds"],
        "software_versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "rasterio": rasterio.__version__,
        },
        "safety": {"source_rasters_opened_read_only": True, "raster_modification": False},
    }
    output_paths = _write_outputs(output_dir, summary, all_product_rows, all_segment_rows, all_control_rows, hotspots, manifest)
    return {"ran": True, "dry_run": False, "summary": summary, "output_paths": output_paths}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only experiment seam audit")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seam-products", help="Comma-separated product keys")
    parser.add_argument("--seam-scales", help="native,modeling_500m")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    result = main(args.experiment, args.dry_run, args.force, args.seam_products, args.seam_scales)
    print(json.dumps(result, indent=2, default=str))
