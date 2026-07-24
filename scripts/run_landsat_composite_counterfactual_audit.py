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


def _run_live(ctx: dict, force: bool, cleanup_tiles: bool) -> dict:
    """Execute the full diagnostic: export, canonical gate, boundary audit,
    provenance, and a status-gated summary."""
    import ee  # noqa: F401
    import rasterio  # noqa: F401

    from core.config import EXPORT_CRS, GEE_PROJECT
    from core.gee_utils import init_gee
    from scripts.run_predictors_only import export_image_direct_or_tiled

    experiment_id = ctx["experiment_id"]
    root = audit.diagnostic_output_root(experiment_id)
    raster_plan = audit.plan_raster_outputs(ctx)
    doc_plan = audit.plan_document_outputs()
    audit.assert_diagnostic_namespace_safe(audit.plan_all_output_paths(ctx), experiment_id)

    # --- FORCE SEMANTICS: clear ONLY the diagnostic namespace (incl. tiles) ---
    if root.exists() and any(root.iterdir()):
        if not force:
            raise AuditRunnerError(
                f"Diagnostic output already exists; pass --force to overwrite: {root}"
            )
        removed = audit.clear_diagnostic_namespace(experiment_id)
        log.info("[--force] cleared diagnostic namespace: %s", removed)
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

    # --- 2. export every diagnostic raster with an explicit nodata sentinel ---
    images = audit.build_ee_images(ctx, region)
    raster_inventory = []
    inventory_by_name = {}
    for name, image in images.items():
        rel = raster_plan[name]
        out_path = root / rel
        tiles_dir = root / "_tiles" / name
        # Unmask to the sentinel so masked/AOI-exterior pixels carry a verifiable
        # nodata tag through direct AND tiled export/merge (never physical zero).
        exported = image.unmask(audit.NODATA_SENTINEL)
        result = export_image_direct_or_tiled(
            exported, out_path, region, EXPORT_SCALE, EXPORT_CRS, name, force=True,
            tiles_dir=tiles_dir, cleanup_tiles=cleanup_tiles, band_count=1,
            run_alignment_qa=True, nodata=audit.NODATA_SENTINEL,
        )
        nodata_check = audit.validate_nodata_mask(result["path"])
        # FAIL FAST: an unverifiable nodata mask means masked/AOI-exterior pixels
        # could be read as physical values -- refuse to continue.
        _require_nodata_ok(nodata_check, name)
        row = {
            "name": name,
            "path": str(result["path"]),
            "transport": result["transport"],
            "tile_grid": result.get("tile_grid"),
            "tile_count": result.get("tile_count"),
            "estimated_bytes": result.get("estimated_bytes"),
            "direct_skipped_preflight": result.get("direct_skipped_preflight"),
            "alignment_qa": result.get("alignment_qa"),
            "nodata_status": nodata_check["status"],
        }
        raster_inventory.append(row)
        inventory_by_name[name] = row
        log.info("[%s] exported via %s (nodata=%s) -> %s",
                 name, result["transport"], nodata_check["status"], out_path)

    # --- 3. derived comparison (Step5 policy) per chain ---
    derived = {}
    for chain in audit.CHAINS:
        baseline_paths = sorted(
            root / raster_plan[name]
            for name in raster_plan
            if name.startswith("baseline_lst_") and name.endswith(f"_{chain}_median")
        )
        current_median = root / raster_plan[f"current_lst_{chain}_median"]
        current_count = root / raster_plan[
            f"current_lst_{'scene_valid_count' if chain == audit.CHAIN_SCENE_WEIGHTED else 'unique_date_valid_count'}"
        ]
        derived[chain] = audit.compute_derived_comparison(
            chain,
            baseline_year_median_paths=baseline_paths,
            current_median_celsius_path=current_median,
            current_count_path=current_count,
            out_dir=(root / "derived" / chain),
        )

    # --- 4. CANONICAL REPRODUCTION GATE (before interpreting date_balanced) ---
    reproduction = audit.run_canonical_reproduction_gate(ctx, root, raster_plan)
    _write_json(root / doc_plan["canonical_reproduction"], reproduction)
    log.info("Canonical reproduction gate: %s", reproduction["status"])

    # --- 5. provenance (kept as its own status; never faked into evidence) ---
    provenance = _run_provenance(ctx, root)
    provenance_state = audit.map_provenance_status(provenance)

    # --- 6. boundary audit across required boundary types + products ---
    boundary_rows, paired_rows, verdicts = _boundary_audit(
        root, raster_plan, inventory_by_name, provenance_state, doc_plan,
    )
    _write_csv(root / doc_plan["boundary_metrics"], boundary_rows)
    _write_csv(root / doc_plan["paired_boundary_comparison"], paired_rows)

    # --- 7. raster inventory + PNG maps ---
    _write_csv(root / doc_plan["raster_inventory"], _inventory_rows(raster_inventory))
    maps = _render_all_maps(root, raster_plan)

    # --- 8. status-gated summary + manifest + run log ---
    summary = _build_summary(
        ctx, scene_metadata, derived, verdicts, provenance, provenance_state, reproduction,
    )
    _write_json(root / doc_plan["counterfactual_summary_json"], summary)
    (root / doc_plan["counterfactual_summary_md"]).write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    (root / doc_plan["run_log"]).write_text(
        f"run_at={datetime.now(timezone.utc).isoformat()}\nlog_file={log_file}\n"
        f"canonical_reproduction={reproduction['status']}\n"
        f"provenance={provenance_state}\n",
        encoding="utf-8",
    )

    produced = _collect_output_files(root)
    manifest = audit.build_file_manifest(produced, output_dir=root)
    _write_json(root / doc_plan["manifest"], manifest)

    return {
        "experiment_id": experiment_id,
        "ran": True,
        "output_root": str(root),
        "raster_count": len(raster_inventory),
        "map_count": len(maps),
        "canonical_reproduction": reproduction["status"],
        "final_status": summary["final_status"],
    }


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


def _boundary_audit(root: Path, raster_plan: dict, inventory_by_name: dict,
                    provenance_state: str, doc_plan: dict):
    """Run the paired boundary audit for every required product + boundary type."""
    # Shared support-edge geometry from current-LST support rasters.
    swc = audit.read_masked_array(root / raster_plan["current_lst_scene_valid_count"])
    dbc = audit.read_masked_array(root / raster_plan["current_lst_unique_date_valid_count"])
    mult = audit.read_masked_array(root / raster_plan["current_lst_same_day_multiplicity"])
    edge_masks = audit.build_edge_masks(swc, dbc, mult)

    # Export-tile negative control availability (from recorded export grid).
    sw_inv = inventory_by_name.get("current_lst_scene_weighted_median", {})
    db_inv = inventory_by_name.get("current_lst_date_balanced_median", {})
    tile_avail = audit.export_tile_control_availability(sw_inv, db_inv)
    tile_units = None
    if tile_avail.get("available"):
        sig = audit.grid_signature(root / raster_plan["current_lst_scene_weighted_median"])
        tile_units = audit.export_tile_boundary_edge_masks(
            sig["width"], sig["height"], tile_avail["tile_grid"],
        )

    # Verified provenance boundary_id units (rasterized onto the exact grid).
    provenance_units = _resolve_provenance_units(root, raster_plan, provenance_state)

    boundary_rows: list[dict] = []
    paired_rows: list[dict] = []
    all_verdicts: dict = {}
    for product in _AUDIT_PRODUCTS:
        paths = _product_chain_paths(root, raster_plan, product)
        # GRID CONTRACT: product rasters must share the support-raster grid.
        audit.assert_same_grid([
            root / raster_plan["current_lst_scene_valid_count"],
            paths["scene_weighted"], paths["date_balanced"],
        ])
        sw = audit.read_masked_array(paths["scene_weighted"])
        db = audit.read_masked_array(paths["date_balanced"])
        result = audit.audit_product_boundaries(
            product, sw, db, edge_masks,
            tile_units=tile_units, provenance_status=provenance_state,
            provenance_units=provenance_units,
        )
        boundary_rows.extend(result["metric_rows"])
        all_verdicts[product] = result["verdicts"]
        for boundary_type, verdict in result["verdicts"].items():
            paired_rows.append({
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
    return boundary_rows, paired_rows, all_verdicts


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
        result = build_provenance(prov_ctx, config)
        summary = dict(result["summary"])
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

    limitations = []
    if not verified_source_present:
        limitations.append(
            "No verified source-scene/path-row boundary evidence was sampled "
            "(provenance was not available or no boundary_id adjacency masks "
            "intersected the grid). Support-count-edge results are a proxy and "
            "do NOT establish verified path/row-boundary behaviour."
        )
    if reproduction_status != "pass":
        limitations.append(
            "Canonical reproduction did not pass; date_balanced results are not "
            "interpretable and no supported_reduction is emitted."
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
            "is_verified_source_boundary_evidence": verified_source_present,
        },
        "support_count_boundary_evidence": support_count_evidence,
        "verified_source_path_row_evidence": verified_source_evidence,
        "export_tile_negative_control": {
            product: gated[product].get("export_tile_boundary")
            for product in gated
        },
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
        f"- Canonical reproduction gate: **{summary['canonical_reproduction']['status']}**",
        f"- QA masking actually applied: **{summary['qa_mask']['qa_source']}** "
        f"(QA_RADSAT applied: {summary['qa_mask']['qa_radsat_applied']})",
        f"- Provenance state: **{summary['provenance']['state']}** "
        f"(verified source-boundary evidence: "
        f"{summary['provenance']['is_verified_source_boundary_evidence']})",
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
        if path.is_file() and "_tiles" not in path.parts and path.name != "manifest.json":
            files.append(path)
    return files


def main(
    experiment_id: str,
    dry_run: bool = False,
    run: bool = False,
    force: bool = False,
    cleanup_tiles: bool = False,
) -> dict:
    ctx = build_experiment_context(experiment_id)
    log_context_summary(ctx, log)

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

    return _run_live(ctx, force=force, cleanup_tiles=cleanup_tiles)


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
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing DIAGNOSTIC outputs only (never canonical).",
    )
    parser.add_argument(
        "--cleanup-tiles", action="store_true",
        help="Delete intermediate diagnostic tile exports after merge.",
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
    )
    print(json.dumps(result, indent=2, default=str))
