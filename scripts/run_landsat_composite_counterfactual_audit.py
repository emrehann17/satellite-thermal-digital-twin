#!/usr/bin/env python3
"""run_landsat_composite_counterfactual_audit.py

Runner for the strictly-diagnostic, NON-DESTRUCTIVE Landsat composite
counterfactual audit (see :mod:`src.landsat_composite_counterfactual_audit`).

It compares the canonical per-scene reducer (``scene_weighted``) against a
same-date daily-composite reducer (``date_balanced``) for current + baseline
Landsat LST and NDVI, exports diagnostic rasters ONLY under
``outputs/diagnostics/landsat_composite_counterfactual/<experiment_id>/``, and
runs a paired boundary-jump audit with a bootstrap interval.

This runner is deliberately STANDALONE. It is NOT wired into scripts/main.py or
core/pipeline_orchestrator.py, never runs Step7-Step10, and never modifies any
canonical Step3/Step5 product.

CLI
    python scripts/run_landsat_composite_counterfactual_audit.py \
        --experiment manavgat_2021 --dry-run
    python scripts/run_landsat_composite_counterfactual_audit.py \
        --experiment manavgat_2021 --run --force
    python scripts/run_landsat_composite_counterfactual_audit.py \
        --experiment manavgat_2021 --run --force --cleanup-tiles

``--dry-run`` performs NO Earth Engine export and writes NO files -- it only
resolves and prints the plan (paths, scene roles, policies). A live run
requires ``--run`` (an explicit opt-in, mirroring run_predictors_only's
export/local-only contract).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.experiment_context import build_experiment_context, get_region, log_context_summary
from core.io_utils import setup_logger

import src.landsat_composite_counterfactual_audit as audit

log, log_file = setup_logger("landsat_composite_counterfactual_audit")


class AuditRunnerError(SystemExit):
    """Fail-fast error for this diagnostic runner."""


# EE export scale/crs (identical to the canonical predictor export).
EXPORT_SCALE = 30


def _plan(ctx: dict) -> dict:
    """Resolve the full plan (paths, rasters, docs) without touching GEE/disk."""
    from core.config import EXPORT_CRS

    root = audit.diagnostic_output_root(ctx["experiment_id"])
    raster_plan = audit.plan_raster_outputs(ctx)
    doc_plan = audit.plan_document_outputs()
    all_paths = audit.plan_all_output_paths(ctx)

    # Namespace safety BEFORE anything else runs.
    audit.assert_diagnostic_namespace_safe(all_paths, ctx["experiment_id"])

    return {
        "experiment_id": ctx["experiment_id"],
        "output_root": str(root),
        "export_crs": EXPORT_CRS,
        "export_scale": EXPORT_SCALE,
        "raster_output_count": len(raster_plan),
        "raster_outputs": {name: str(root / rel) for name, rel in raster_plan.items()},
        "document_outputs": {name: str(root / rel) for name, rel in doc_plan.items()},
        "audit_config": audit.build_audit_config(ctx),
    }


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_scene_manifest_csv(path: Path, metadata: dict) -> None:
    columns = [
        "input_role", "baseline_year", "scene_id", "landsat_product_id",
        "spacecraft_id", "sensor_id", "wrs_path", "wrs_row",
        "acquisition_datetime", "acquisition_date", "cloud_cover",
        "cloud_cover_land", "processing_level", "collection_category",
        "collection_number", "source_collection",
    ]
    rows = []
    for records in metadata["collections"].values():
        for record in records:
            rows.append({key: record.get(key) for key in columns})
    rows.sort(key=lambda r: (str(r["input_role"]), str(r["scene_id"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _run_live(ctx: dict, force: bool, cleanup_tiles: bool, resume: bool = False) -> dict:
    """Execute the full diagnostic: export, canonical gate, boundary audit,
    provenance, and a status-gated summary.

    ``resume`` (mutually exclusive with ``force``) keeps the existing diagnostic
    namespace, re-validates every already-present final GeoTIFF, reuses only the
    ones that pass validation, quarantines invalid ones, and re-exports whatever
    is missing. A ``raster_inventory.partial.json`` checkpoint is written
    atomically after every completed or validated raster.
    """
    import ee  # noqa: F401
    import rasterio  # noqa: F401

    from core.config import EXPORT_CRS, GEE_PROJECT
    from core.gee_utils import init_gee

    experiment_id = ctx["experiment_id"]
    root = audit.diagnostic_output_root(experiment_id)
    raster_plan = audit.plan_raster_outputs(ctx)
    doc_plan = audit.plan_document_outputs()
    audit.assert_diagnostic_namespace_safe(audit.plan_all_output_paths(ctx), experiment_id)

    if force and resume:
        raise AuditRunnerError("--force and --resume are mutually exclusive.")

    # --- FORCE vs RESUME namespace semantics ---
    if root.exists() and any(root.iterdir()):
        if force:
            removed = audit.clear_diagnostic_namespace(experiment_id)
            log.info("[--force] cleared diagnostic namespace: %s", removed)
        elif resume:
            log.info(
                "[--resume] keeping existing diagnostic namespace; every existing "
                "final raster will be re-validated (checkpoint text is not trusted): %s",
                root,
            )
        else:
            raise AuditRunnerError(
                f"Diagnostic output already exists; pass --resume to continue or "
                f"--force to overwrite: {root}"
            )
    root.mkdir(parents=True, exist_ok=True)

    try:
        init_gee(GEE_PROJECT)
    except Exception as exc:  # noqa: BLE001
        raise AuditRunnerError(
            f"GEE init/auth failed: {type(exc).__name__}: {exc}. "
            "Run 'earthengine authenticate'."
        ) from exc

    region = get_region(ctx)

    # --- 1. audit_config + source-scene metadata (explicit, from EE) ---
    _write_json(root / doc_plan["audit_config"], audit.build_audit_config(ctx))
    scene_metadata = audit.build_source_scene_metadata(ctx, region)
    # PREFLIGHT: refuse to launch the raster-export loop with an empty scene list.
    scene_total = audit.assert_scene_list_nonempty(scene_metadata)
    log.info("Source-scene preflight OK: %d scenes", scene_total)
    _write_json(root / doc_plan["source_scene_metadata"], scene_metadata)
    _write_scene_manifest_csv(root / doc_plan["scene_manifest_csv"], scene_metadata)

    # --- 2. export/resume every diagnostic raster with a verifiable nodata tag ---
    checkpoint_path = root / "raster_inventory.partial.json"
    if resume and checkpoint_path.exists():
        try:
            prior = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            log.info("[--resume] prior checkpoint found with %d record(s); it is "
                     "advisory only and every raster is still re-validated.",
                     len(prior.get("rasters", [])))
        except (OSError, json.JSONDecodeError):
            log.warning("[--resume] prior checkpoint unreadable; ignoring it.")

    images = audit.build_ee_images(ctx, region)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    raster_inventory = []
    inventory_by_name = {}
    reference_signature = None
    for name, image in images.items():
        row, reference_signature = _export_or_resume_raster(
            name, image, ctx, root, raster_plan, region, EXPORT_CRS,
            force=force, resume=resume, cleanup_tiles=cleanup_tiles,
            reference_signature=reference_signature, run_stamp=run_stamp,
        )
        raster_inventory.append(row)
        inventory_by_name[name] = row
        # CHECKPOINT after EVERY completed/validated raster (atomic write).
        _write_checkpoint(checkpoint_path, experiment_id, raster_inventory)

    # All required rasters must be present AND validated before anything downstream.
    missing = [name for name in raster_plan if not (root / raster_plan[name]).exists()]
    if missing:
        raise AuditRunnerError(
            f"Cannot proceed to derived/reproduction/boundary stages: "
            f"{len(missing)} required raster(s) still missing after export/resume: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    # --- 3-8. bounded-memory local finalization (derived, gate, provenance,
    #         windowed boundary audit, maps, summary, manifest, run log) ---
    local = _local_finalization(
        ctx, root, raster_plan, doc_plan, scene_metadata,
        raster_inventory, inventory_by_name, mode="live",
    )

    exported_names = [r["name"] for r in raster_inventory if r.get("status") == "exported"]
    resumed_names = [r["name"] for r in raster_inventory if r.get("status") == "resumed_existing"]
    return {
        "experiment_id": experiment_id,
        "ran": True,
        "output_root": str(root),
        "raster_count": len(raster_inventory),
        "rasters_exported": len(exported_names),
        "rasters_resumed": len(resumed_names),
        "map_count": local["map_count"],
        "canonical_reproduction": local["reproduction_status"],
        "final_status": local["final_status"],
    }


def _export_or_resume_raster(
    name, image, ctx, root, raster_plan, region, export_crs,
    *, force, resume, cleanup_tiles, reference_signature, run_stamp, base_dir=None,
):
    """Export one diagnostic raster, or validate-and-reuse an existing one.

    Returns ``(inventory_row, reference_signature)``. Never overwrites an
    existing final raster silently: an invalid one is MOVED to the in-namespace
    quarantine directory and then re-exported. Canonical outputs are never
    touched (every path stays under the diagnostic root).
    """
    from scripts.run_predictors_only import export_image_direct_or_tiled

    if base_dir is None:
        base_dir = audit.PROJECT_ROOT
    experiment_id = ctx["experiment_id"]
    out_path = root / raster_plan[name]
    now = datetime.now(timezone.utc).isoformat()
    tiles_dir = root / "_tiles" / name

    # --- RESUME: validate an already-present final raster before reusing it ---
    if resume and out_path.exists():
        verdict = audit.validate_existing_diagnostic_raster(
            out_path, expected_crs=export_crs, reference_signature=reference_signature,
        )
        if verdict["valid"]:
            if reference_signature is None:
                reference_signature = verdict["signature"]
            nodata_check = audit.validate_nodata_mask(out_path)
            log.info("[%s] resumed_existing (validated) -> %s", name, out_path)
            row = {
                "name": name, "path": str(out_path), "status": "resumed_existing",
                "transport": "resumed_existing", "tile_grid": None, "tile_count": None,
                "estimated_bytes": None, "direct_skipped_preflight": None,
                "alignment_qa": None, "nodata_status": nodata_check["status"],
                "nodata_validation": nodata_check, "timestamp": now,
            }
            return row, reference_signature
        # Invalid: quarantine (MOVE, never delete) then re-export from clean tiles.
        dest = audit.quarantine_invalid_raster(experiment_id, out_path, base_dir=base_dir)
        log.warning("[%s] existing raster invalid (%s); quarantined -> %s",
                    name, verdict["reason"], dest)

    # --- EXPORT (missing raster, quarantined-then-reexport, or --force run) ---
    # In resume mode, export into a CLEAN per-product temporary tile directory so
    # existing intermediate tiles are preserved untouched and an incomplete tile
    # set is never mistaken for a completed raster.
    if resume:
        tiles_dir = root / "_tiles_resume" / run_stamp / name
    helper_force = bool(force)  # resume/clean-dir starts empty; force re-does tiles

    exported = image.unmask(audit.NODATA_SENTINEL)
    result = export_image_direct_or_tiled(
        exported, out_path, region, EXPORT_SCALE, export_crs, name, force=helper_force,
        tiles_dir=tiles_dir, cleanup_tiles=cleanup_tiles, band_count=1,
        run_alignment_qa=True, nodata=audit.NODATA_SENTINEL,
    )
    nodata_check = audit.validate_nodata_mask(result["path"])
    # FAIL FAST: an unverifiable nodata mask means masked/AOI-exterior pixels
    # could be read as physical values -- refuse to continue.
    _require_nodata_ok(nodata_check, name)
    signature = audit.grid_signature(result["path"])
    if reference_signature is None:
        reference_signature = signature
    elif not audit._signatures_match(reference_signature, signature):
        raise AuditRunnerError(
            f"[{name}] freshly exported raster grid does not match the accepted "
            f"diagnostic reference grid (grid_mismatch)."
        )
    log.info("[%s] exported via %s (nodata=%s) -> %s",
             name, result["transport"], nodata_check["status"], out_path)
    row = {
        "name": name, "path": str(result["path"]), "status": "exported",
        "transport": result["transport"], "tile_grid": result.get("tile_grid"),
        "tile_count": result.get("tile_count"),
        "estimated_bytes": result.get("estimated_bytes"),
        "direct_skipped_preflight": result.get("direct_skipped_preflight"),
        "alignment_qa": result.get("alignment_qa"),
        "nodata_status": nodata_check["status"], "nodata_validation": nodata_check,
        "timestamp": now,
    }
    return row, reference_signature


def _write_checkpoint(checkpoint_path: Path, experiment_id: str, raster_inventory: list[dict]) -> None:
    """Atomically persist the running raster inventory after each raster."""
    payload = {
        "audit": audit.DIAGNOSTIC_NAMESPACE,
        "experiment_id": experiment_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "raster_count": len(raster_inventory),
        "rasters": raster_inventory,
    }
    audit.write_json_atomic(checkpoint_path, payload)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _inventory_rows(raster_inventory: list[dict]) -> list[dict]:
    """Flatten inventory (retains all export-tile negative-control fields)."""
    rows = []
    for row in raster_inventory:
        rows.append({
            "name": row["name"],
            "path": row["path"],
            "status": row.get("status"),  # exported | resumed_existing
            "transport": row["transport"],
            "tile_grid": json.dumps(row.get("tile_grid")),
            "tile_count": row.get("tile_count"),
            "estimated_bytes": row.get("estimated_bytes"),
            "direct_skipped_preflight": row.get("direct_skipped_preflight"),
            "alignment_qa": json.dumps(row.get("alignment_qa"), default=str),
            "nodata_status": row.get("nodata_status"),
        })
    return rows


def _require_nodata_ok(nodata_check: dict, name: str) -> None:
    """Fail fast when a freshly exported raster's nodata mask is not verifiable."""
    if nodata_check.get("status") != "ok":
        raise AuditRunnerError(
            f"[{name}] nodata validation failed (status="
            f"{nodata_check.get('status')}); refusing to continue -- masked/AOI "
            f"exterior pixels are not verifiably tagged with NODATA_SENTINEL. "
            f"Detail: {nodata_check}"
        )


# Products carried through the paired boundary audit. Support edges are defined
# once (from current-LST support rasters) and applied identically to all three.
_AUDIT_PRODUCTS = ("current_lst", "current_minus_baseline", "anomaly_zscore")

# The predeclared support edge used for the final-status decision.
_FINAL_STATUS_EDGE = "scene_count_edge"


def compute_final_status(gated: dict, reproduction_status: str) -> str:
    """Predeclared final-claim gate.

    ``supported_reduction`` is emitted ONLY when ALL hold:
      * canonical reproduction passed;
      * current LST shows supported_reduction on the predeclared support edge;
      * current-minus-baseline AND anomaly z-score are DIRECTIONALLY CONSISTENT
        (also supported_reduction on the predeclared support edge).
    Any mix of supported_reduction and supported_increase across products is
    ``contradictory_uncertain``. Export-tile controls are a negative control and
    are never consulted here (they can never create positive evidence).
    """
    if reproduction_status != "pass":
        return "canonical_reproduction_failed"

    def stat(product):
        return gated.get(product, {}).get(_FINAL_STATUS_EDGE, {}).get("status")

    cur = stat("current_lst")
    diff = stat("current_minus_baseline")
    z = stat("anomaly_zscore")
    trio = [cur, diff, z]
    if "supported_increase" in trio and "supported_reduction" in trio:
        return "contradictory_uncertain"
    if cur == diff == z == "supported_reduction":
        return "supported_reduction"
    if all(s == "supported_increase" for s in trio):
        return "supported_increase"
    if all(s == "insufficient_evidence" for s in trio):
        return "insufficient_evidence"
    return "uncertain"


def _product_chain_paths(root: Path, raster_plan: dict, product: str) -> dict:
    """Resolve the scene_weighted / date_balanced raster path for a product."""
    if product == "current_lst":
        return {
            "scene_weighted": root / raster_plan["current_lst_scene_weighted_median"],
            "date_balanced": root / raster_plan["current_lst_date_balanced_median"],
        }
    fname = {
        "current_minus_baseline": "current_minus_baseline_celsius.tif",
        "anomaly_zscore": "anomaly_zscore.tif",
    }[product]
    return {
        chain: root / "derived" / chain / fname for chain in audit.CHAINS
    }


def _resolve_provenance_units(root: Path, raster_plan: dict, provenance_state: str) -> dict | None:
    """Rasterize verified scene_boundaries.geojson onto the exact product grid.

    Only when provenance is available AND the geojson has verified boundary
    features do we produce ``{boundary_id: edge_mask}`` units (preserving
    boundary_id). Otherwise returns None so the audit reports
    insufficient_boundary_metadata explicitly.
    """
    import rasterio

    if provenance_state != "provenance_available":
        return None
    geojson_path = root / "scene_boundaries.geojson"
    if not geojson_path.exists():
        return None
    geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
    with rasterio.open(root / raster_plan["current_lst_scene_weighted_median"]) as src:
        transform, width, height = src.transform, src.width, src.height
    units = audit.rasterize_provenance_boundaries(geojson, transform, width, height)
    return units or None


def _support_paths(root: Path, raster_plan: dict) -> dict:
    """Recorded current-LST support-count raster paths (never read whole here)."""
    return {
        "scene_count_edge": root / raster_plan["current_lst_scene_valid_count"],
        "unique_date_count_edge": root / raster_plan["current_lst_unique_date_valid_count"],
        "same_day_multiplicity_edge": root / raster_plan["current_lst_same_day_multiplicity"],
    }


def _resolve_tile_seam_specs(root: Path, raster_plan: dict, inventory_by_name: dict):
    """Approximate export-tile seam specs (negative control), or None."""
    sw_inv = inventory_by_name.get("current_lst_scene_weighted_median", {})
    db_inv = inventory_by_name.get("current_lst_date_balanced_median", {})
    tile_avail = audit.export_tile_control_availability(sw_inv, db_inv)
    if not tile_avail.get("available"):
        return None
    sig = audit.grid_signature(root / raster_plan["current_lst_scene_weighted_median"])
    specs = audit.export_tile_seam_specs(sig["width"], sig["height"], tile_avail["tile_grid"])
    return specs or None


def _resolve_provenance_code_spec(root: Path, raster_plan: dict, provenance_state: str,
                                  tmp_dir: Path):
    """Build the ONE disk-backed int32 provenance code raster (bounded), or None.

    Replaces the old full-mask-per-boundary_id rasterization: verified boundaries
    are burned once into a windowed code memmap under ``_analysis_tmp``.
    """
    import rasterio

    if provenance_state != "provenance_available":
        return None
    geojson_path = root / "scene_boundaries.geojson"
    if not geojson_path.exists():
        return None
    geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
    with rasterio.open(root / raster_plan["current_lst_scene_weighted_median"]) as src:
        transform, width, height = src.transform, src.width, src.height
    return audit.build_provenance_code_memmap(geojson, transform, width, height, tmp_dir)


def _paired_rows_from_verdicts(product: str, verdicts: dict) -> list[dict]:
    rows = []
    for boundary_type, verdict in verdicts.items():
        rows.append({
            "product": product,
            "boundary_type": boundary_type,
            "status": verdict.get("status"),
            "unit_type": verdict.get("unit_type"),
            "n_units": verdict.get("n_units"),
            "n_pairs": verdict.get("n_pairs"),
            "point_estimate": verdict.get("point_estimate"),
            "interval_low": verdict.get("interval_low"),
            "interval_high": verdict.get("interval_high"),
        })
    return rows


def _audit_one_product(root: Path, raster_plan: dict, product: str, support_paths: dict,
                       tile_seam_specs, provenance_state: str, provenance_code_spec,
                       tmp_dir: Path, resource_log: list | None):
    """Bounded-memory windowed boundary audit for ONE product.

    Returns ``(metric_rows, paired_rows, verdicts)``. Never reads whole product
    rasters into RAM; every boundary type is streamed one at a time.
    """
    paths = _product_chain_paths(root, raster_plan, product)
    # GRID CONTRACT: product rasters must share the support-raster grid.
    audit.assert_same_grid([
        support_paths["scene_count_edge"], paths["scene_weighted"], paths["date_balanced"],
    ])
    result = audit.audit_product_boundaries_windowed(
        product, paths["scene_weighted"], paths["date_balanced"], support_paths,
        tmp_dir=tmp_dir, tile_seam_specs=tile_seam_specs,
        provenance_status=provenance_state, provenance_code_spec=provenance_code_spec,
        resource_log=resource_log,
    )
    verdicts = result["verdicts"]
    return result["metric_rows"], _paired_rows_from_verdicts(product, verdicts), verdicts


def _render_all_maps(root: Path, raster_plan: dict) -> list[str]:
    maps = []
    for name in raster_plan:
        if not name.endswith("_scene_weighted_median"):
            continue
        pair = name[: -len("_scene_weighted_median")]
        db_name = f"{pair}_date_balanced_median"
        if db_name not in raster_plan:
            continue
        maps.extend(audit.render_pair_maps(
            root / raster_plan[name], root / raster_plan[db_name],
            root / "maps", pair_name=pair,
        ))
    return maps


def _run_provenance(ctx: dict, root: Path) -> dict:
    """Run the read-only provenance schema over our explicit metadata file.

    The diagnostic wrote an explicit ``source_scene_metadata.json`` in this
    namespace; provenance therefore has a real scene list and returns an honest
    status rather than insufficient-evidence.
    """
    try:
        from core.source_scene_provenance_config import source_scene_provenance_config
        from src.source_scene_provenance import build_provenance

        prov_ctx = dict(ctx)
        prov_ctx["output_root"] = root
        prov_ctx["data_root"] = root
        config = source_scene_provenance_config(ctx["experiment_id"])
        # METADATA CORRECTION ONLY: rewrite the composite masking-rule text so no
        # generated provenance field repeats the false "QA_PIXEL + QA_RADSAT"
        # claim. Raster computation is untouched and QA_RADSAT is never added.
        config = audit.correct_composite_masking_rules(config)
        result = build_provenance(prov_ctx, config)
        summary = dict(result["summary"])
        # Qualify EXACTLY what the metadata-derived source path/row evidence is.
        summary["source_boundary_evidence_qualification"] = (
            audit.provenance_evidence_qualification(summary.get("mode", "metadata_only"))
        )
        summary["footprints_geojson"] = result["footprints"]
        summary["boundaries_geojson"] = result["boundaries"]
        (root / "scene_footprints.geojson").write_text(
            json.dumps(result["footprints"], indent=2), encoding="utf-8"
        )
        (root / "scene_boundaries.geojson").write_text(
            json.dumps(result["boundaries"], indent=2), encoding="utf-8"
        )
        return {k: v for k, v in summary.items() if not k.endswith("_geojson")}
    except Exception as exc:  # noqa: BLE001
        log.warning("Provenance analysis returned insufficient evidence: %s", exc)
        return {"status": "insufficient_evidence", "reason": str(exc)}


def gate_verdicts(verdicts: dict, reproduction_status: str) -> dict:
    """Apply the canonical-reproduction gate to every paired verdict.

    If reproduction did not pass, NO supported_reduction may be emitted -- every
    such verdict is downgraded to ``canonical_reproduction_failed``. Verified
    source/path-row evidence is also kept separate from support-count evidence.
    """
    gated = {}
    for product, product_verdicts in verdicts.items():
        gated[product] = {}
        for boundary_type, verdict in product_verdicts.items():
            emitted = dict(verdict)
            emitted["raw_status"] = verdict.get("status")
            if reproduction_status != "pass" and verdict.get("status") == "supported_reduction":
                emitted["status"] = "canonical_reproduction_failed"
                emitted["supported_reduction_suppressed"] = True
            gated[product][boundary_type] = emitted
    return gated


def _build_summary(
    ctx, scene_metadata, derived, verdicts, provenance, provenance_state, reproduction,
) -> dict:
    reproduction_status = reproduction["status"]
    gated = gate_verdicts(verdicts, reproduction_status)

    # Separate support-count evidence from verified source/path-row evidence.
    support_count_types = ("scene_count_edge", "unique_date_count_edge", "same_day_multiplicity_edge")
    support_count_evidence = {
        product: {bt: gated[product][bt] for bt in support_count_types if bt in gated[product]}
        for product in gated
    }
    verified_source_evidence = {
        product: gated[product].get("source_scene_path_row")
        for product in gated
    }
    # Verified only when a boundary_id unit was actually sampled for a product.
    verified_source_present = any(
        (v or {}).get("is_verified_source_boundary_evidence") is True
        for v in verified_source_evidence.values()
    )

    final_status = compute_final_status(gated, reproduction_status)

    export_tile_negative_control = {
        product: gated[product].get("export_tile_boundary")
        for product in gated
    }
    provenance_mode = (provenance or {}).get("mode", "metadata_only")
    limitations = audit.build_report_limitations(
        experiment_id=ctx["experiment_id"],
        verified_source_evidence=verified_source_evidence,
        export_tile_negative_control=export_tile_negative_control,
        reproduction_status=reproduction_status,
        verified_source_present=verified_source_present,
        provenance_mode=provenance_mode,
    )

    return {
        "audit": audit.DIAGNOSTIC_NAMESPACE,
        "experiment_id": ctx["experiment_id"],
        "final_status": final_status,
        "final_status_rule": (
            "supported_reduction requires canonical_reproduction==pass AND "
            "directionally consistent supported_reduction on the predeclared "
            f"'{_FINAL_STATUS_EDGE}' for current_lst, current_minus_baseline, "
            "and anomaly_zscore; export-tile controls are excluded and can "
            "never create positive evidence"
        ),
        "canonical_reproduction": {
            "status": reproduction_status,
            "canonical_gate_version": reproduction.get("canonical_gate_version"),
            "decisive_gate": reproduction.get("decisive_gate"),
            "raw_encoding": reproduction.get("raw_encoding"),
            "semantic_checks": reproduction.get("semantic_checks", reproduction.get("checks")),
            "threshold_boundary": reproduction.get("threshold_boundary"),
            "warnings": reproduction.get("warnings"),
            "reduction_tolerance_note": reproduction.get("reduction_tolerance_note"),
            "checks": reproduction.get("checks"),
        },
        "qa_mask": audit.qa_mask_provenance(),
        "step5_policy": audit.step5_policy_snapshot(),
        "scene_count_semantics": scene_metadata["count_semantics"],
        "derived_comparison": {
            chain: {
                "baseline_mean": derived[chain]["baseline_mean"],
                "baseline_std": derived[chain]["baseline_std"],
                "difference": derived[chain]["difference"],
                "anomaly_zscore": derived[chain]["anomaly_zscore"],
            }
            for chain in derived
        },
        "provenance": {
            "state": provenance_state,
            "summary": provenance,
            # Kept True: verified geometric boundary evidence WAS sampled. The
            # qualification states exactly what was verified (metadata-derived
            # source-footprint/path-row boundaries, not pixel-level provenance).
            "is_verified_source_boundary_evidence": verified_source_present,
            "source_boundary_evidence_qualification": (
                audit.provenance_evidence_qualification(provenance_mode)
            ),
        },
        "support_count_boundary_evidence": support_count_evidence,
        "verified_source_path_row_evidence": verified_source_evidence,
        "export_tile_negative_control": export_tile_negative_control,
        "paired_boundary_verdicts": gated,
        "limitations": limitations,
        "claim_boundary": (
            "A supported_reduction status is evidence that daily compositing "
            "lowers support-edge jumps in THIS AOI; it is NOT proof that "
            "same-date duplication is the sole cause, nor a mandate to change "
            "the canonical production reducer. Support-count-edge evidence is "
            "distinct from verified source-scene/path-row boundary evidence, "
            "and export-tile controls are a negative control only."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _summary_markdown(summary: dict) -> str:
    current = summary["paired_boundary_verdicts"].get("current_lst", {})
    lines = [
        f"# Landsat composite counterfactual audit -- {summary['experiment_id']}",
        "",
        f"- **Final status: {summary['final_status']}**",
        f"- Canonical reproduction gate: **{summary['canonical_reproduction']['status']}** "
        f"(decisive gate: {summary['canonical_reproduction'].get('decisive_gate')}, "
        f"schema {summary['canonical_reproduction'].get('canonical_gate_version')})",
        (
            "- Raw GeoTIFF encoding differs (canonical serializes masked pixels as "
            "DN 0 with no nodata tag; diagnostic uses a -9999 sentinel) but is "
            "semantically equivalent after the production Step5 conversion; the raw "
            "files are NOT bitwise identical."
            if (summary['canonical_reproduction'].get('raw_encoding') or {}).get('status')
            == 'difference_semantically_equivalent'
            else "- Raw GeoTIFF encodings agree."
        ),
        (
            "- The 5e-5 Celsius derived-reduction tolerance is a float32 NUMERICAL "
            "reproducibility tolerance, not a scientific effect-size threshold."
        ),
        f"- QA masking actually applied: **{summary['qa_mask']['qa_source']}** "
        f"({audit.QA_MASKING_RULE_TEXT})",
        f"- Provenance state: **{summary['provenance']['state']}** "
        f"(verified source-boundary evidence: "
        f"{summary['provenance']['is_verified_source_boundary_evidence']}; "
        "metadata-derived source-footprint/path-row boundary evidence, not "
        "pixel-level selected-scene provenance)",
        "",
        "## Support-count boundary verdicts (current LST)",
        "",
    ]
    for boundary_type, verdict in current.items():
        lines.append(
            f"- `{boundary_type}`: **{verdict.get('status')}** "
            f"(unit_type={verdict.get('unit_type')}, n_units={verdict.get('n_units')}, "
            f"interval=[{verdict.get('interval_low')}, {verdict.get('interval_high')}])"
        )
    lines += [
        "",
        "`reduction = absolute_jump_scene_weighted - absolute_jump_date_balanced` "
        "(positive => date-balanced compositing reduces the jump).",
        "",
    ]
    lines += audit.limitations_markdown(summary.get("limitations", []))
    lines += [
        "",
        "## Claim boundary",
        "",
        summary["claim_boundary"],
    ]
    if summary["final_status"] == "canonical_reproduction_failed":
        lines += [
            "",
            "> Canonical reproduction did not pass; no supported_reduction claim "
            "is emitted regardless of date_balanced results.",
        ]
    return "\n".join(lines) + "\n"


def _collect_output_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        if (path.is_file() and "_tiles" not in path.parts
                and audit.ANALYSIS_TMP_DIRNAME not in path.parts
                and path.name != "manifest.json"):
            files.append(path)
    return files


# =============================================================================
# LOCAL ANALYSIS CHECKPOINT + FLUSHING RESOURCE LOG (bounded-memory finalize)
# =============================================================================
class _FlushingResourceLog(list):
    """A list whose ``append`` also flushes each record to disk + the logger.

    Passed into the windowed audit as ``resource_log`` so the CURRENTLY RUNNING
    stage is written to ``analysis_run.log`` (and echoed to the console) BEFORE a
    potential crash -- making it obvious which product/boundary was active.
    """

    def __init__(self, path):
        super().__init__()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record) -> None:  # noqa: D401
        super().append(record)
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass
        log.info("[resource] %s", json.dumps(record, default=str))


# Stages whose results are produced by the canonical-reproduction gate logic and
# must be INVALIDATED (recomputed) whenever the gate schema/version changes --
# without rerunning Earth Engine. Every other stage (raster exports, derived
# products, boundary metrics/maps, provenance) is algorithm-versioned
# independently and is still reused across a gate-version bump.
_GATE_VERSIONED_STAGES = frozenset({"reproduction", "report"})

# Stages whose CACHED result carries generated report/provenance text (limitations
# wording, QA masking-rule correction, provenance qualification) and must be
# INVALIDATED whenever the report schema/version changes -- so a --finalize-only
# refresh regenerates only these reports while reusing every valid boundary,
# raster, derived, and map calculation. No Earth Engine work is triggered.
_REPORT_VERSIONED_STAGES = frozenset({"provenance", "report"})


class _AnalysisCheckpoint:
    """Atomic local-analysis checkpoint, SEPARATE from raster_inventory.partial.json.

    Stores per-stage results inline plus the files/inputs each stage depends on.
    A stage is only reused when its referenced files still exist and match their
    recorded byte sizes -- checkpoint text alone is never trusted.

    A ``canonical_gate_version`` is recorded alongside the stages. When the stored
    gate version differs from the current ``audit.CANONICAL_GATE_VERSION``, the
    gate-versioned stages (canonical reproduction + final report) are invalidated
    so they are recomputed under the corrected gate, while valid raster exports
    and boundary results are reused. No Earth Engine work is triggered.
    """

    def __init__(self, root: Path, experiment_id: str):
        self.root = Path(root)
        self.experiment_id = experiment_id
        data = audit.read_analysis_checkpoint(self.root)
        self.stages = data.get("stages", {}) if isinstance(data, dict) else {}
        self.stored_gate_version = (
            data.get("canonical_gate_version") if isinstance(data, dict) else None
        )
        self.stored_report_version = (
            data.get("report_schema_version") if isinstance(data, dict) else None
        )

    def _gate_version_stale(self, key: str) -> bool:
        return (
            key in _GATE_VERSIONED_STAGES
            and self.stored_gate_version != audit.CANONICAL_GATE_VERSION
        )

    def _report_version_stale(self, key: str) -> bool:
        return (
            key in _REPORT_VERSIONED_STAGES
            and self.stored_report_version != audit.REPORT_SCHEMA_VERSION
        )

    def reusable(self, key: str):
        entry = self.stages.get(key)
        if not entry or not entry.get("done"):
            return None
        # A gate schema/version change invalidates only the canonical-gate stages;
        # a report schema/version change invalidates only the report/provenance
        # stages. Boundary/raster/derived/map stages are reused unchanged.
        if self._gate_version_stale(key) or self._report_version_stale(key):
            return None
        if not audit.files_present_and_signed(entry.get("files")):
            return None
        if not audit.files_present_and_signed(entry.get("inputs")):
            return None
        return entry.get("result")

    def mark(self, key: str, *, result=None, files=None, inputs=None) -> None:
        self.stages[key] = {
            "done": True,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "files": files or [],
            "inputs": inputs or [],
            "result": result,
        }
        # Persisting under the CURRENT gate + report versions also upgrades the
        # recorded versions, so later runs see the stages as current.
        self.stored_gate_version = audit.CANONICAL_GATE_VERSION
        self.stored_report_version = audit.REPORT_SCHEMA_VERSION
        audit.write_analysis_checkpoint(self.root, {
            "audit": audit.DIAGNOSTIC_NAMESPACE,
            "experiment_id": self.experiment_id,
            "checkpoint_kind": "local_analysis",
            "canonical_gate_version": audit.CANONICAL_GATE_VERSION,
            "report_schema_version": audit.REPORT_SCHEMA_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "stages": self.stages,
        })


def _file_ref(path: Path) -> dict:
    path = Path(path)
    try:
        return {"path": str(path), "bytes": int(path.stat().st_size)}
    except OSError:
        return {"path": str(path), "bytes": -1}


def _derived_baseline_paths(root: Path, raster_plan: dict, chain: str) -> list[Path]:
    return sorted(
        root / raster_plan[name]
        for name in raster_plan
        if name.startswith("baseline_lst_") and name.endswith(f"_{chain}_median")
    )


def _derived_output_files(root: Path, chain: str) -> list[Path]:
    out_dir = root / "derived" / chain
    return [out_dir / n for n in (
        "baseline_lst_mean_celsius.tif", "baseline_lst_std_celsius.tif",
        "baseline_valid_year_count.tif", "current_minus_baseline_celsius.tif",
        "anomaly_zscore.tif",
    )]


def _derived_stats_from_files(chain: str, out_dir: Path) -> dict:
    """Recompute the derived summary stats from EXISTING derived tifs (read-only).

    Lets finalize reuse valid derived products without overwriting them.
    """
    import numpy as np

    def _stat(path):
        arr = audit.read_masked_array(path, 1)
        finite = arr[np.isfinite(arr)]
        return {"valid_pixels": int(finite.size),
                "mean": float(np.mean(finite)) if finite.size else None}

    out_dir = Path(out_dir)
    return {
        "chain": chain,
        "policy": audit.step5_policy_snapshot(),
        "written": {p.name: str(p) for p in _derived_output_files(out_dir.parent, chain)},
        "baseline_mean": _stat(out_dir / "baseline_lst_mean_celsius.tif"),
        "baseline_std": _stat(out_dir / "baseline_lst_std_celsius.tif"),
        "difference": _stat(out_dir / "current_minus_baseline_celsius.tif"),
        "anomaly_zscore": _stat(out_dir / "anomaly_zscore.tif"),
    }


def _local_finalization(ctx, root, raster_plan, doc_plan, scene_metadata,
                        raster_inventory, inventory_by_name, *, mode: str) -> dict:
    """Bounded-memory local post-export finalization shared by live + finalize.

    Runs the Step5-policy derived comparison, canonical reproduction gate,
    provenance, the WINDOWED boundary audit, PNG maps, and the status-gated
    summary/manifest/run-log -- all with a validate-before-skip local analysis
    checkpoint and a flushing resource log. A failed canonical reproduction gate
    suppresses supported_reduction but never aborts finalization. Temporary
    analysis scratch is cleaned only on success.
    """
    experiment_id = ctx["experiment_id"]
    tmp_dir = root / audit.ANALYSIS_TMP_DIRNAME
    tmp_dir.mkdir(parents=True, exist_ok=True)
    resource_log = _FlushingResourceLog(root / "analysis_run.log")
    checkpoint = _AnalysisCheckpoint(root, experiment_id)
    support_paths = _support_paths(root, raster_plan)

    resource_log.append(audit.stage_resource_record(
        "local_finalization", phase="begin", tmp_dir=tmp_dir, mode=mode))

    # --- 3. derived comparison (Step5 policy) per chain (reuse when valid) ---
    derived = {}
    for chain in audit.CHAINS:
        key = f"derived:{chain}"
        cached = checkpoint.reusable(key)
        out_files = _derived_output_files(root, chain)
        if cached is not None:
            derived[chain] = cached
            continue
        if all(p.exists() for p in out_files):
            # Existing valid derived products: recompute stats read-only (no overwrite).
            derived[chain] = _derived_stats_from_files(chain, root / "derived" / chain)
        else:
            current_count = root / raster_plan[
                f"current_lst_{'scene_valid_count' if chain == audit.CHAIN_SCENE_WEIGHTED else 'unique_date_valid_count'}"
            ]
            derived[chain] = audit.compute_derived_comparison(
                chain,
                baseline_year_median_paths=_derived_baseline_paths(root, raster_plan, chain),
                current_median_celsius_path=root / raster_plan[f"current_lst_{chain}_median"],
                current_count_path=current_count,
                out_dir=(root / "derived" / chain),
            )
        checkpoint.mark(key, result=derived[chain],
                        files=[_file_ref(p) for p in out_files])
        resource_log.append(audit.stage_resource_record(
            "derived", phase="end", tmp_dir=tmp_dir, chain=chain))

    # --- 4. CANONICAL REPRODUCTION GATE (never relaxed; gates support claims) ---
    repro_cached = checkpoint.reusable("reproduction")
    if repro_cached is not None:
        reproduction = repro_cached
    else:
        reproduction = audit.run_canonical_reproduction_gate(ctx, root, raster_plan)
        _write_json(root / doc_plan["canonical_reproduction"], reproduction)
        checkpoint.mark("reproduction", result=reproduction,
                        files=[_file_ref(root / doc_plan["canonical_reproduction"])])
    log.info("Canonical reproduction gate: %s", reproduction["status"])
    if reproduction["status"] == "canonical_reproduction_failed":
        log.info("Canonical reproduction FAILED -> scientific support claims will be "
                 "suppressed, but local finalization continues (non-aborting).")

    # --- 5. provenance (its own status; from existing local metadata) ---
    prov_cached = checkpoint.reusable("provenance")
    if prov_cached is not None:
        provenance = prov_cached
    else:
        provenance = _run_provenance(ctx, root)
        checkpoint.mark("provenance", result=provenance, files=[
            _file_ref(root / "scene_footprints.geojson"),
            _file_ref(root / "scene_boundaries.geojson"),
        ])
    provenance_state = audit.map_provenance_status(provenance)
    resource_log.append(audit.stage_resource_record(
        "provenance", phase="end", tmp_dir=tmp_dir, state=provenance_state))

    # --- 6. WINDOWED boundary audit across required products (checkpointed) ---
    tile_seam_specs = _resolve_tile_seam_specs(root, raster_plan, inventory_by_name)
    provenance_code_spec = None
    boundary_rows: list[dict] = []
    paired_rows: list[dict] = []
    verdicts: dict = {}
    for product in _AUDIT_PRODUCTS:
        key = f"boundary:{product}"
        paths = _product_chain_paths(root, raster_plan, product)
        inputs = [_file_ref(paths["scene_weighted"]), _file_ref(paths["date_balanced"]),
                  _file_ref(support_paths["scene_count_edge"]),
                  _file_ref(support_paths["unique_date_count_edge"]),
                  _file_ref(support_paths["same_day_multiplicity_edge"])]
        cached = checkpoint.reusable(key)
        if cached is not None:
            boundary_rows.extend(cached["metric_rows"])
            paired_rows.extend(cached["paired_rows"])
            verdicts[product] = cached["verdicts"]
            continue
        # Build the ONE provenance code raster lazily (shared across products).
        if provenance_code_spec is None and provenance_state == "provenance_available":
            provenance_code_spec = _resolve_provenance_code_spec(
                root, raster_plan, provenance_state, tmp_dir)
        metric_rows, prod_paired, prod_verdicts = _audit_one_product(
            root, raster_plan, product, support_paths, tile_seam_specs,
            provenance_state, provenance_code_spec, tmp_dir, resource_log,
        )
        boundary_rows.extend(metric_rows)
        paired_rows.extend(prod_paired)
        verdicts[product] = prod_verdicts
        checkpoint.mark(key, inputs=inputs, result={
            "metric_rows": metric_rows, "paired_rows": prod_paired,
            "verdicts": prod_verdicts,
        })
    _write_csv(root / doc_plan["boundary_metrics"], boundary_rows)
    _write_csv(root / doc_plan["paired_boundary_comparison"], paired_rows)

    # --- 7. raster inventory + PNG maps (per-pair checkpointed) ---
    _write_csv(root / doc_plan["raster_inventory"], _inventory_rows(raster_inventory))
    maps: list[str] = []
    for name in raster_plan:
        if not name.endswith("_scene_weighted_median"):
            continue
        pair = name[: -len("_scene_weighted_median")]
        db_name = f"{pair}_date_balanced_median"
        if db_name not in raster_plan:
            continue
        key = f"maps:{pair}"
        cached = checkpoint.reusable(key)
        if cached is not None:
            maps.extend(cached["pngs"])
            continue
        pngs = audit.render_pair_maps(
            root / raster_plan[name], root / raster_plan[db_name],
            root / "maps", pair_name=pair,
        )
        maps.extend(pngs)
        checkpoint.mark(key, result={"pngs": pngs},
                        files=[_file_ref(p) for p in pngs])

    # --- 8. status-gated summary + manifest + run log (final report stage) ---
    summary = _build_summary(
        ctx, scene_metadata, derived, verdicts, provenance, provenance_state, reproduction,
    )
    _write_json(root / doc_plan["counterfactual_summary_json"], summary)
    (root / doc_plan["counterfactual_summary_md"]).write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    resource_log.append(audit.stage_resource_record(
        "local_finalization", phase="end", tmp_dir=tmp_dir, mode=mode,
        final_status=summary["final_status"]))
    (root / doc_plan["run_log"]).write_text(
        f"run_at={datetime.now(timezone.utc).isoformat()}\nlog_file={log_file}\n"
        f"mode={mode}\n"
        f"canonical_reproduction={reproduction['status']}\n"
        f"provenance={provenance_state}\n"
        f"final_status={summary['final_status']}\n"
        f"analysis_resource_log={root / 'analysis_run.log'}\n"
        f"peak_rss_mib={audit.process_rss_mib()}\n",
        encoding="utf-8",
    )

    produced = _collect_output_files(root)
    manifest = audit.build_file_manifest(produced, output_dir=root)
    exported_names = [r["name"] for r in raster_inventory if r.get("status") == "exported"]
    resumed_names = [r["name"] for r in raster_inventory
                     if r.get("status") in ("resumed_existing", "finalize_reused")]
    manifest["raster_status"] = {r["name"]: r.get("status") for r in raster_inventory}
    manifest["raster_status_summary"] = {
        "exported": exported_names, "resumed_existing": resumed_names,
    }
    # Final status carries the decisive gate schema/version so a consumer can tell
    # which canonical-reproduction gate produced it.
    manifest["canonical_gate_version"] = audit.CANONICAL_GATE_VERSION
    manifest["final_status"] = summary["final_status"]
    manifest["canonical_reproduction_status"] = reproduction["status"]
    _write_json(root / doc_plan["manifest"], manifest)

    checkpoint.mark("report", files=[
        _file_ref(root / doc_plan["counterfactual_summary_json"]),
        _file_ref(root / doc_plan["manifest"]),
    ])

    # Clean the temporary analysis scratch only after a fully successful pass.
    removed = audit.clean_analysis_tmp(root)
    if removed:
        log.info("Cleaned analysis scratch: %s", removed)

    return {
        "final_status": summary["final_status"],
        "reproduction_status": reproduction["status"],
        "provenance_state": provenance_state,
        "map_count": len(maps),
    }


def _validate_required_rasters_for_finalize(root: Path, raster_plan: dict, export_crs: str):
    """Validate the required GEE-exported rasters exist and are reusable.

    Returns ``(inventory_rows, inventory_by_name)`` on success. Raises
    AuditRunnerError with an explicit list if any required exported raster is
    missing or invalid. Derived (locally rebuilt) products are NOT required here.
    """
    exported_names = [n for n in raster_plan if raster_plan[n].startswith("rasters/")]
    problems = []
    rows = []
    inventory_by_name = {}
    reference_signature = None
    # Reuse the resume checkpoint's transport/tile_grid metadata when present.
    partial = {}
    partial_path = root / "raster_inventory.partial.json"
    if partial_path.exists():
        try:
            data = json.loads(partial_path.read_text(encoding="utf-8"))
            partial = {r.get("name"): r for r in data.get("rasters", [])}
        except (OSError, json.JSONDecodeError):
            partial = {}

    for name in exported_names:
        out_path = root / raster_plan[name]
        verdict = audit.validate_existing_diagnostic_raster(
            out_path, expected_crs=export_crs, reference_signature=reference_signature,
        )
        if not verdict["valid"]:
            problems.append(f"{name}: {verdict['reason']} ({out_path})")
            continue
        if reference_signature is None:
            reference_signature = verdict["signature"]
        prior = partial.get(name, {})
        row = {
            "name": name, "path": str(out_path), "status": "finalize_reused",
            "transport": prior.get("transport", "finalize_reused"),
            "tile_grid": prior.get("tile_grid"), "tile_count": prior.get("tile_count"),
            "estimated_bytes": prior.get("estimated_bytes"),
            "direct_skipped_preflight": prior.get("direct_skipped_preflight"),
            "alignment_qa": prior.get("alignment_qa"),
            "nodata_status": verdict.get("nodata_status"),
        }
        rows.append(row)
        inventory_by_name[name] = row

    if problems:
        raise AuditRunnerError(
            "[--finalize-only] cannot proceed: "
            f"{len(problems)} required exported raster(s) missing or invalid:\n  - "
            + "\n  - ".join(problems)
        )
    return rows, inventory_by_name


def _run_finalize(ctx: dict) -> dict:
    """Local-only finalization: NO Earth Engine of any kind.

    Uses the existing diagnostic directory, validates all required exported
    rasters, reuses the existing source_scene_metadata.json, rebuilds derived
    products when needed, runs canonical reproduction, provenance, the
    low-memory boundary audit, renders maps, and writes summaries/CSVs/manifest/
    run log. Never imports/initializes Earth Engine and never deletes the
    diagnostic namespace.
    """
    from core.config import EXPORT_CRS

    experiment_id = ctx["experiment_id"]
    root = audit.diagnostic_output_root(experiment_id)
    raster_plan = audit.plan_raster_outputs(ctx)
    doc_plan = audit.plan_document_outputs()
    audit.assert_diagnostic_namespace_safe(audit.plan_all_output_paths(ctx), experiment_id)

    if not root.exists():
        raise AuditRunnerError(
            f"[--finalize-only] diagnostic namespace does not exist: {root}. "
            "Run the export phase first."
        )

    # Reuse the EXISTING explicit source-scene metadata (never re-query EE).
    scene_meta_path = root / doc_plan["source_scene_metadata"]
    if not scene_meta_path.exists():
        raise AuditRunnerError(
            f"[--finalize-only] required {scene_meta_path.name} is missing; "
            "cannot finalize without the existing source-scene metadata."
        )
    scene_metadata = json.loads(scene_meta_path.read_text(encoding="utf-8"))

    # Refresh the deterministic (EE-free) audit config document.
    _write_json(root / doc_plan["audit_config"], audit.build_audit_config(ctx))

    # Validate every required exported raster BEFORE any downstream stage.
    raster_inventory, inventory_by_name = _validate_required_rasters_for_finalize(
        root, raster_plan, EXPORT_CRS)
    log.info("[--finalize-only] validated %d required exported rasters.",
             len(raster_inventory))

    local = _local_finalization(
        ctx, root, raster_plan, doc_plan, scene_metadata,
        raster_inventory, inventory_by_name, mode="finalize_only",
    )
    return {
        "experiment_id": experiment_id,
        "ran": True,
        "mode": "finalize_only",
        "output_root": str(root),
        "raster_count": len(raster_inventory),
        "map_count": local["map_count"],
        "canonical_reproduction": local["reproduction_status"],
        "final_status": local["final_status"],
    }


def main(
    experiment_id: str,
    dry_run: bool = False,
    run: bool = False,
    force: bool = False,
    cleanup_tiles: bool = False,
    resume: bool = False,
    finalize_only: bool = False,
) -> dict:
    if force and resume:
        raise AuditRunnerError("--force and --resume are mutually exclusive.")

    # --finalize-only is a pure local post-export mode: it must never combine
    # with any Earth-Engine / export / namespace-mutating flag.
    if finalize_only:
        incompatible = [flag for flag, on in (
            ("--run", run), ("--force", force), ("--resume", resume),
            ("--cleanup-tiles", cleanup_tiles),
        ) if on]
        if incompatible:
            raise AuditRunnerError(
                "--finalize-only is incompatible with: "
                f"{', '.join(incompatible)}. It performs NO Earth Engine work and "
                "only finalizes existing local exports."
            )

    ctx = build_experiment_context(experiment_id)
    log_context_summary(ctx, log)

    if finalize_only:
        log.info("[--finalize-only] local post-export finalization; NO Earth Engine.")
        return _run_finalize(ctx)

    plan = _plan(ctx)

    if dry_run or not run:
        log.info("[dry-run] Diagnostic output root: %s", plan["output_root"])
        log.info("[dry-run] Planned diagnostic rasters: %d", plan["raster_output_count"])
        for name, path in plan["raster_outputs"].items():
            log.info("[dry-run]   raster %s -> %s", name, path)
        for name, path in plan["document_outputs"].items():
            log.info("[dry-run]   doc    %s -> %s", name, path)
        log.info("[dry-run] No GEE export and no raster writes were performed.")
        return {
            "experiment_id": experiment_id,
            "ran": False,
            "reason": "dry_run" if dry_run else "no_run_flag",
            "plan": plan,
        }

    return _run_live(ctx, force=force, cleanup_tiles=cleanup_tiles, resume=resume)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone diagnostic Landsat composite counterfactual audit "
            "(scene_weighted vs date_balanced). Writes ONLY under "
            "outputs/diagnostics/landsat_composite_counterfactual/<experiment_id>/. "
            "Never modifies canonical Step3/Step5 products; never runs Step7-Step10."
        )
    )
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve and print the plan; perform NO GEE export and NO writes.",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Explicit opt-in to perform the live GEE export + diagnostic.",
    )
    # --force and --resume are mutually exclusive (argparse-enforced).
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--force", action="store_true",
        help="Overwrite existing DIAGNOSTIC outputs only (never canonical).",
    )
    mode.add_argument(
        "--resume", action="store_true",
        help="Keep the existing diagnostic namespace; re-validate every present "
             "final raster, reuse valid ones, quarantine invalid ones, and "
             "export only what is missing. Never deletes the namespace.",
    )
    parser.add_argument(
        "--cleanup-tiles", action="store_true",
        help="Delete intermediate diagnostic tile exports after merge.",
    )
    parser.add_argument(
        "--finalize-only", action="store_true",
        help="Local post-export finalization ONLY: no Earth Engine import/init/"
             "query/export/download. Validates existing exported rasters, rebuilds "
             "derived products when needed, runs the bounded-memory boundary audit, "
             "renders maps, and writes summaries/manifest. Incompatible with --run, "
             "--force, --resume, and --cleanup-tiles; never deletes the namespace.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = main(
        experiment_id=args.experiment,
        dry_run=args.dry_run,
        run=args.run,
        force=args.force,
        cleanup_tiles=args.cleanup_tiles,
        resume=args.resume,
        finalize_only=args.finalize_only,
    )
    print(json.dumps(result, indent=2, default=str))
