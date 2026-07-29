#!/usr/bin/env python3
"""
scripts/run_landsat_current_support_harmonization.py

Dedicated entry point for the DIAGNOSTIC current-period acquisition-date offset
harmonization counterfactual, run after the completed residual seam attribution
audit whose final status was `current_support_dominant`.

    python scripts/run_landsat_current_support_harmonization.py \
        --experiment manavgat_2021 \
        --dry-run

CONTRACT
--------
    - Exactly one of --dry-run / --run. Default execution is never implied.
    - --resume and --force are mutually exclusive, and both require --run.
    - --dry-run writes nothing, creates no directory, and performs NO Earth
      Engine operation: it does not import, initialise, authenticate or call
      Earth Engine.
    - --force may delete ONLY
      outputs/diagnostics/landsat_current_support_harmonization/<experiment_id>/.
    - The ONLY place a live Earth Engine operation may occur is the isolated
      daily-mosaic export stage, and only under --run when a required daily
      current-period mosaic is not already present locally. Only diagnostic
      daily CURRENT-period rasters are ever exported; no baseline raster is
      ever exported, recomputed or overwritten. Pass --no-earth-engine to
      forbid the export stage outright.
    - Every analysis stage runs inside an Earth Engine guard, so no analysis
      code path can reach Earth Engine even accidentally.
    - Step6, Step7 and Step8 are NEVER run. No label, burned-area product or
      model-performance metric is read.
    - Nothing is smoothed, blended, feathered, interpolated or cosmetically
      altered. The single intervention is a per-acquisition-date additive
      scalar.
    - The experiment can never report that the seam is fixed and never issues a
      production decision.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.landsat_composite_counterfactual_audit as audit
import src.landsat_composite_downstream_ab as ab
import src.landsat_current_support_harmonization as hz
from core.io_utils import setup_logger
from core.config import DAILY_EXPORT_NODATA 

log, log_file = setup_logger("landsat_current_support_harmonization")

#: EE export scale, identical to the canonical predictor export.
EXPORT_SCALE = 30

#: Set once per run so namespace-safety checks in helpers stay rooted.
_EXPERIMENT_FOR_SAFETY = [None]


class HarmonizationRunnerError(SystemExit):
    """Fail-fast CLI error (same convention as the sibling runners)."""


# =============================================================================
# Argument validation
# =============================================================================
def validate_modes(dry_run: bool, run: bool, resume: bool, force: bool) -> None:
    """Exactly one mode; --resume/--force are run-only and mutually exclusive."""
    if dry_run and run:
        raise HarmonizationRunnerError(
            "--dry-run and --run are mutually exclusive; pass exactly one."
        )
    if not dry_run and not run:
        raise HarmonizationRunnerError(
            "one of --dry-run or --run is required. Default execution is never "
            "implied by this runner."
        )
    if resume and force:
        raise HarmonizationRunnerError("--resume and --force are mutually exclusive.")
    if resume and not run:
        raise HarmonizationRunnerError("--resume requires --run.")
    if force and not run:
        raise HarmonizationRunnerError("--force requires --run.")


# =============================================================================
# Dry-run
# =============================================================================
def _print_dry_run(plan: dict) -> None:
    log.info("[dry-run] experiment: %s", plan["experiment"])
    log.info("[dry-run] experiment_id: %s", plan["experiment_id"])
    log.info("[dry-run] reference composite: %s", plan["reference_composite"])
    log.info("[dry-run] candidate composite: %s", plan["candidate_composite"])
    log.info("[dry-run] target products: %s", plan["target_products"])
    log.info("[dry-run] output root: %s", plan["output_root"])

    log.info("[dry-run] --- resolved frozen inputs ---")
    for role, entry in plan["resolved_inputs"].items():
        log.info("[dry-run]   %-44s present=%-5s required=%-5s %s",
                 role, entry["present"], entry["required"], entry["path"])
    if plan["missing_required_inputs"]:
        log.error("[dry-run]   MISSING REQUIRED: %s", plan["missing_required_inputs"])
    if plan["missing_optional_inputs"]:
        for item in plan["missing_optional_inputs"]:
            log.warning("[dry-run]   missing optional: %s", item)

    log.info("[dry-run] --- upstream prerequisites ---")
    state = plan["upstream_prerequisites"]
    for key in ("counterfactual_final_status", "downstream_ab_final_status",
                "residual_seam_final_status", "downstream_ab_reference_reproduction",
                "baseline_invariance", "baseline_invariance_source",
                "prerequisites_met"):
        log.info("[dry-run]   %-42s %s", key, state.get(key))
    if state.get("failures"):
        log.error("[dry-run]   PREREQUISITE FAILURES: %s", state["failures"])

    log.info("[dry-run] --- path/row evidence ---")
    pathrow = plan["pathrow_evidence"]
    log.info("[dry-run]   availability: %s (%s)",
             pathrow.get("availability"), pathrow.get("reason"))
    log.info("[dry-run]   distinct metadata interfaces: %s",
             pathrow.get("interface_count"))
    log.info("[dry-run]   evidence is METADATA-derived, not pixel-level "
             "selected-scene provenance")

    log.info("[dry-run] --- daily current-period mosaic plan ---")
    export_plan = plan["daily_export_plan"]
    if export_plan is None:
        log.error("[dry-run]   unavailable: %s", plan["daily_export_plan_error"])
    else:
        window = export_plan["current_window"]
        log.info("[dry-run]   frozen window: [%s, %s) (%s); last included: %s",
                 window["start_date"], window["end_date"],
                 window["end_semantics"], window["effective_last_included_date"])
        log.info("[dry-run]   source collection: %s", window["source_collection"])
        log.info("[dry-run]   Landsat scaling: %s / %s",
                 window["landsat_scale"], window["landsat_offset"])
        log.info("[dry-run]   unique acquisition dates: %s (from %s scenes)",
                 export_plan["date_count"], export_plan["scene_count"])

        # --- the question this section exists to answer, stated out loud ---
        if export_plan["complete_daily_mosaics_present"]:
            log.info("[dry-run]   COMPLETE DAILY CURRENT-PERIOD MOSAICS ALREADY "
                     "EXIST LOCALLY: YES (%s/%s dates present; hashes are "
                     "re-verified at run time)",
                     export_plan["date_count"], export_plan["date_count"])
        else:
            present = export_plan["date_count"] - len(export_plan["missing_locally"])
            log.warning("[dry-run]   COMPLETE DAILY CURRENT-PERIOD MOSAICS ALREADY "
                        "EXIST LOCALLY: NO (status=%s; %s/%s dates present)",
                        export_plan["daily_mosaic_status"], present,
                        export_plan["date_count"])

        log.info("[dry-run]   --- exact required dates (%s) ---",
                 len(export_plan["required_dates"]))
        for date in export_plan["required_dates"]:
            log.info("[dry-run]     %s", date)

        log.info("[dry-run]   --- exact source-scene inventory ---")
        for item in export_plan["items"]:
            log.info("[dry-run]     %s  scenes=%s  path/rows=%s  "
                     "temporal_observations=%s  present=%s",
                     item["acquisition_date"], item["scene_count"],
                     ",".join(item["path_rows"]), item["temporal_observations"],
                     item["present_locally"])
            for scene_id, product_id, acquired in zip(
                item["scene_ids"], item["landsat_product_ids"],
                item["acquisition_datetimes"],
            ):
                log.info("[dry-run]         scene %-24s product=%-16s acquired=%s",
                         scene_id, product_id, acquired)

        log.info("[dry-run]   --- planned diagnostic download paths ---")
        log.info("[dry-run]     root: %s", export_plan["planned_download_root"])
        for item in export_plan["items"]:
            log.info("[dry-run]     %s -> %s%s",
                     item["acquisition_date"], item["planned_download_path"],
                     "" if not item["present_locally"]
                     else f"  (present, sha256={item['verified_sha256'][:12]})")

        log.info("[dry-run]   missing locally: %s",
                 export_plan["missing_locally"] or "none")
        for key, value in export_plan["export_contract"].items():
            log.info("[dry-run]   contract %-30s %s", key, value)

        if not export_plan["complete_daily_mosaics_present"]:
            log.info("[dry-run]   --- later USER-EXECUTED command required to "
                     "fetch the daily mosaics ---")
            for line in export_plan["fetch_command"].splitlines():
                log.info("[dry-run]     %s", line)
            log.info("[dry-run]     note: %s", export_plan["fetch_command_note"])
            log.info("[dry-run]     the agent does NOT run this command and has "
                     "performed no Earth Engine operation")

        log.info("[dry-run]   Earth Engine touched while building this plan: %s",
                 export_plan["earth_engine_touched_by_this_function"])

    config = plan["configuration"]
    log.info("[dry-run] --- intervention ---")
    for key, value in config["intervention"].items():
        log.info("[dry-run]   %-38s %s", key, value)

    log.info("[dry-run] --- overlap graph ---")
    for key, value in config["overlap_graph"].items():
        log.info("[dry-run]   %-38s %s", key, value)

    log.info("[dry-run] --- offset solution ---")
    for key, value in config["offset_solution"].items():
        log.info("[dry-run]   %-38s %s", key, value)

    log.info("[dry-run] --- support invariance gate ---")
    for key, value in config["support_invariance"].items():
        log.info("[dry-run]   %-38s %s", key, value)

    log.info("[dry-run] --- boundary evaluation (reused semantics) ---")
    boundary = config["boundary_evaluation"]
    log.info("[dry-run]   boundaries: %s", list(boundary["boundaries"]))
    log.info("[dry-run]   required reduction boundaries: %s",
             boundary["required_reduction_boundaries"])
    log.info("[dry-run]   non-boundary control: %s", boundary["nonboundary_control"])
    log.info("[dry-run]   reused from: %s", boundary["reused_from"])
    for item in boundary["reused_semantics"]:
        log.info("[dry-run]     reuse: %s", item)

    log.info("[dry-run] --- bootstrap ---")
    for key, value in config["bootstrap"].items():
        log.info("[dry-run]   %-38s %s", key, value)

    log.info("[dry-run] --- predeclared decision bounds ---")
    for key, value in config["decision_bounds"].items():
        log.info("[dry-run]   %-38s %s", key, value)
    log.info("[dry-run]   decision rule: %s", plan["decision_rule"])
    log.info("[dry-run]   allowed final statuses: %s", plan["allowed_final_statuses"])
    log.info("[dry-run]   forbidden conclusions: %s", plan["forbidden_conclusions"])

    log.info("[dry-run] --- planned stages ---")
    for stage in plan["planned_stages"]:
        log.info("[dry-run]   stage %s", stage)

    log.info("[dry-run] --- expected files ---")
    for name, path in plan["expected_files"].items():
        log.info("[dry-run]   %-58s %s", name, path)

    log.info("[dry-run] --- limitations ---")
    for limitation in plan["limitations"]:
        log.info("[dry-run]   - %s", limitation)

    log.info("[dry-run] writes performed: %s | directories created: %s | "
             "Earth Engine calls: %s | rasters modified: %s | "
             "frozen namespaces touched: %s",
             plan["writes_performed"], plan["directories_created"],
             plan["earth_engine_calls"], plan["rasters_modified"],
             plan["frozen_namespaces_touched"])
    log.info("[dry-run] smoothing applied: %s | spatial interpolation applied: %s | "
             "baseline recomputed: %s | labels used: %s | Step8 metrics used: %s",
             plan["smoothing_applied"], plan["spatial_interpolation_applied"],
             plan["baseline_recomputed"], plan["labels_used"],
             plan["step8_metrics_used"])


# =============================================================================
# ISOLATED Earth Engine export stage
# =============================================================================
def _build_daily_ee_image(region, window: dict, date: str):
    """The daily Celsius mosaic for ONE acquisition date.

    Every ingredient is the frozen one, reused verbatim from the counterfactual
    audit: the same filtered source collection, the same QA_PIXEL mask, the same
    same-day SPATIAL median reducer, and the same Landsat scaling. The Celsius
    affine is applied to the daily mosaic instead of to the temporal median;
    because it is a strictly increasing affine map it commutes with the median,
    which the reference-reproduction gate then verifies numerically.
    """
    import ee

    current = audit._current_window(window["end_date"], int(window["window_days"]))
    base = audit._base_filtered_collection(
        region, current["start_date"], current["end_date"],
        (current["month_start"], current["month_end"]), ["ST_B10", "QA_PIXEL"],
    )
    daily = audit._daily_composite_collection(base, "ST_B10", region)
    image = ee.Image(daily.filter(ee.Filter.eq("export_date", date)).first())
    return (
        image.multiply(audit.LANDSAT_SCALE).add(audit.LANDSAT_OFFSET).subtract(273.15)
        .rename("LST_Celsius").toFloat().clip(region)
    )


def _initialise_earth_engine() -> str:
    """Initialise Earth Engine ONCE, immediately before the first `ee.*` use.

    This is the only Earth Engine initialisation in the whole experiment. It is
    reached exclusively from the isolated export stage, and only when at least
    one daily mosaic actually has to be exported -- never on import, never in
    `--dry-run`, never in tests, never when the mosaics are already present, and
    never in the local-only graph/harmonisation/evaluation stages.

    Authentication is NOT attempted: `ee.Authenticate()` is never called, and
    credentials remain a user/environment responsibility.
    """
    from core.config import GEE_PROJECT
    from core.gee_utils import init_gee

    log.info("[earth_engine] initialising Earth Engine for project %r "
             "(required before any ee.* object is constructed)", GEE_PROJECT)
    try:
        init_gee()
    except Exception as error:                              # noqa: BLE001
        raise HarmonizationRunnerError(
            "GEE initialization failed. The configured project comes from the "
            f"existing GEE_PROJECT setting ({GEE_PROJECT!r} in core.config); this "
            "runner never calls ee.Authenticate() and never changes that "
            "project. NO daily mosaic export occurred and nothing was written "
            "under the diagnostic root. Authenticate in your own environment "
            "(for example `earthengine authenticate`) and re-run. Original "
            f"error: {type(error).__name__}: {error}"
        ) from error
    log.info("[earth_engine] initialised for project %r", GEE_PROJECT)
    return GEE_PROJECT


def _pending_export_items(export_plan: dict, *, force: bool) -> list[dict]:
    """The daily mosaics that genuinely have to be exported.

    Resolved BEFORE Earth Engine is touched, so a run whose mosaics are all
    present locally never initialises Earth Engine at all.
    """
    return [item for item in export_plan["items"]
            if force or not Path(item["output_path"]).exists()]


def _export_daily_mosaics(experiment_id: str, root: Path, export_plan: dict,
                          *, force: bool) -> list[dict]:
    """Export ONLY the missing diagnostic daily current-period mosaics.

    This is the single function in the whole experiment that may perform a live
    Earth Engine operation. It never exports, recomputes or overwrites a
    baseline raster, and it writes only inside the dedicated diagnostic root.

    Earth Engine is initialised here, once, and ONLY when something actually has
    to be exported -- `get_region` builds `ee.Geometry` objects, so it must not
    be reached before initialisation.
    """
    from core.config import EXPORT_CRS
    from core.experiment_context import build_experiment_context, get_region
    from scripts.run_predictors_only import export_image_direct_or_tiled

    window = export_plan["current_window"]
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    pending = _pending_export_items(export_plan, force=force)
    region = None
    if pending:
        # Initialise BEFORE build_experiment_context/get_region: get_region
        # constructs ee.Geometry objects and fails on an uninitialised client.
        _initialise_earth_engine()
        ctx = build_experiment_context(experiment_id)
        region = get_region(ctx)
    else:
        log.info("[daily_mosaic_inventory] every daily mosaic is already present "
                 "locally; Earth Engine is NOT initialised and no export runs")

    inventory: list[dict] = []
    reference_signature = None
    for item in export_plan["items"]:
        date = item["acquisition_date"]
        out_path = Path(item["output_path"])
        hz.assert_namespace_safe([out_path], experiment_id)

        if out_path.exists() and not force:
            signed = audit.sha256_and_size(out_path)
            log.info("[daily %s] present locally (sha256=%s); reused without export",
                     date, signed["sha256"][:12])
            inventory.append(_daily_record(date, out_path, item, "reused_existing"))
            continue

        log.info("[daily %s] exporting %s scene(s) -> %s",
                 date, item["scene_count"], out_path)
        image = _build_daily_ee_image(region, window, date)
        result = export_image_direct_or_tiled(
            image.unmask(DAILY_EXPORT_NODATA,
                         sameFootprint=False),
            out_path,
            region,
            EXPORT_SCALE,
            EXPORT_CRS,
            f"daily_{date}",
            force=bool(force),
            tiles_dir=root / "_tiles" / run_stamp / date,
            cleanup_tiles=True,
            band_count=1, 
            run_alignment_qa=True, 
            nodata=audit.NODATA_SENTINEL,
        )
        nodata_check = audit.validate_nodata_mask(result["path"])
        if nodata_check["status"] not in ("ok", "valid", "pass"):
            raise HarmonizationRunnerError(
                f"[daily {date}] nodata mask could not be verified "
                f"({nodata_check['status']}); refusing to continue, because an "
                "unverifiable mask would let masked pixels be read as physical "
                "temperatures."
            )
        signature = audit.grid_signature(result["path"])
        if reference_signature is None:
            reference_signature = signature
        elif not audit._signatures_match(reference_signature, signature):
            raise HarmonizationRunnerError(
                f"[daily {date}] exported grid does not match the accepted daily "
                "reference grid (grid_mismatch)."
            )
        inventory.append(_daily_record(date, Path(result["path"]), item, "exported",
                                       transport=result.get("transport")))
    return inventory


def _daily_record(date: str, path: Path, item: dict, status: str,
                  transport=None) -> dict:
    signed = audit.sha256_and_size(path)
    return OrderedDict((
        ("acquisition_date", date),
        ("path", str(path)),
        ("status", status),
        ("transport", transport),
        ("scene_count", item["scene_count"]),
        ("scene_ids", list(item["scene_ids"])),
        ("path_rows", list(item["path_rows"])),
        ("temporal_observations", 1),
        ("sha256", signed["sha256"]),
        ("bytes", signed["bytes"]),
        ("recorded_at", datetime.now(timezone.utc).isoformat()),
    ))


def _verify_daily_inventory(root: Path, inventory: list[dict]) -> list[str]:
    """Re-hash every daily mosaic; a changed file invalidates the run."""
    failures = []
    for record in inventory:
        path = Path(record["path"])
        if not path.exists():
            failures.append(f"{record['acquisition_date']}: missing {path}")
            continue
        signed = audit.sha256_and_size(path)
        if record.get("sha256") and signed["sha256"] != record["sha256"]:
            failures.append(
                f"{record['acquisition_date']}: sha256 changed since it was recorded"
            )
    return failures


# =============================================================================
# Live run
# =============================================================================
def _progress_logger(started: float):
    state = {"last": 0.0}

    def logger(stage: str, start: int, stop: int, height: int) -> None:
        now = time.time()
        if now - state["last"] < 15.0 and stop < height:
            return
        state["last"] = now
        log.info("[%s] rows %s-%s / %s | rss=%.1f MiB | elapsed=%.1fs",
                 stage, start, stop, height, audit.process_rss_mib(), now - started)

    return logger


def _run_live(experiment_id: str, *, resume: bool, force: bool,
              allow_earth_engine: bool, force_daily_export: bool = False) -> dict:
    started = time.time()
    hz.assert_supported_experiment(experiment_id)
    _EXPERIMENT_FOR_SAFETY[0] = experiment_id
    root = hz.diagnostic_output_root(experiment_id)
    logger = _progress_logger(started)
    resources: "OrderedDict[str, object]" = OrderedDict()

    if force:
        removed = hz.clear_diagnostic_namespace(experiment_id)
        log.warning("[force] removed ONLY the dedicated diagnostic namespace: %s",
                    removed or "(nothing to remove)")

    # --- stage 1: input validation ----------------------------------------
    plan = hz.build_input_plan(experiment_id)
    hz.assert_required_inputs(plan, experiment_id)
    state = hz.load_upstream_state(experiment_id)
    hz.validate_upstream_state(state)
    grid_contract = hz.assert_grid_contract(plan)
    log.info("[input_validation] grid contract %s over %s rasters",
             grid_contract["status"], grid_contract["raster_count"])

    layout = hz.plan_output_layout(experiment_id)
    hz.assert_namespace_safe(layout.values(), experiment_id)
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)

    config = hz.build_config_snapshot(experiment_id)
    config_path = layout["config"] / "harmonization_config.json"
    hz.write_json_atomic(config_path, config)
    provenance = hz.build_input_provenance(experiment_id)
    provenance_path = root / "input_provenance.json"
    hz.write_json_atomic(provenance_path, provenance)
    hz.write_checkpoint_stage(root, "input_validation", [config_path, provenance_path],
                              {"grid_contract": grid_contract["status"]})

    # --- stage 2: daily mosaic inventory (the ONLY EE-capable stage) ------
    export_plan = hz.build_daily_export_plan(experiment_id)
    inventory_path = layout["daily"] / "daily_inventory.json"
    missing = export_plan["missing_locally"]
    if force_daily_export and not missing:
        missing = list(export_plan["required_dates"])
        log.warning("[daily_mosaic_inventory] --force-daily-export: re-exporting "
                    "all %s diagnostic daily rasters", len(missing))
    if missing and not allow_earth_engine:
        raise HarmonizationRunnerError(
            f"{len(missing)} daily current-period mosaic(s) are not present "
            f"locally ({missing}) and --no-earth-engine forbids the export "
            "stage. Re-run without --no-earth-engine to export exactly these "
            "diagnostic daily rasters, or place them under "
            f"{layout['daily_reference']}."
        )
    if missing:
        log.warning("[daily_mosaic_inventory] %s daily mosaic(s) missing locally; "
                    "entering the ISOLATED Earth Engine export stage: %s",
                    len(missing), missing)
    daily_records = _export_daily_mosaics(
        experiment_id, root, export_plan, force=force or force_daily_export)
    verification_failures = _verify_daily_inventory(root, daily_records)
    if verification_failures:
        raise HarmonizationRunnerError(
            "daily mosaic hash verification failed: " + "; ".join(verification_failures)
        )
    dates = [record["acquisition_date"] for record in daily_records]
    daily_paths = [Path(record["path"]) for record in daily_records]

    # STRICT daily-raster validity contract, BEFORE any science reads them, so a
    # corrupt export is reported as a corrupt export and not as a failed
    # scientific gate.
    height, width = hz.raster_shape(hz.reference_grid_path(experiment_id))
    contract = hz.validate_daily_raster_contract(
        daily_paths, dates, height=height, width=width, logger=logger)
    hz.write_json_atomic(layout["daily"] / "daily_raster_contract.json", contract)
    if not contract["passes"]:
        log.error("[daily_mosaic_inventory] daily-raster contract FAILED (%s): %s",
                  contract["root_cause"], contract["failures"])
        _quarantine_incompatible_dailies(root, daily_paths, contract)
    hz.assert_daily_raster_contract(contract)
    log.info("[daily_mosaic_inventory] daily-raster contract passed (%s)",
             contract["root_cause"])
    hz.write_json_atomic(inventory_path, OrderedDict((
        ("experiment", hz.DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("current_window", export_plan["current_window"]),
        ("export_contract", export_plan["export_contract"]),
        ("date_count", len(daily_records)),
        ("scene_count", export_plan["scene_count"]),
        ("versions", dict(hz.DAILY_CONTRACT_VERSIONS)),
        ("dates", daily_records),
        ("exported_baseline_rasters", 0),
        ("created_at", datetime.now(timezone.utc).isoformat()),
    )))
    hz.write_checkpoint_stage(root, "daily_mosaic_inventory", [inventory_path],
                              {"date_count": len(daily_records)})
    log.info("[daily_mosaic_inventory] %s unique acquisition dates from %s scenes",
             len(daily_records), export_plan["scene_count"])

    # --- every remaining stage is Earth-Engine-free by construction -------
    with ab.EarthEngineGuard():
        return _analyse(experiment_id, root, layout, dates, daily_paths,
                        state=state, config=config, provenance=provenance,
                        export_plan=export_plan, resume=resume, logger=logger,
                        started=started, resources=resources,
                        grid_contract=grid_contract)


def _analyse(experiment_id: str, root: Path, layout, dates, daily_paths, *,
             state, config, provenance, export_plan, resume, logger, started,
             resources, grid_contract) -> dict:
    import numpy as np

    inventory = hz.daily_date_inventory(hz.current_scene_records(experiment_id))

    # --- stage 3: reference reproduction ----------------------------------
    reproduction_path = root / "reference_reproduction.json"
    reproduction = _reuse_reference_reproduction(
        root, reproduction_path, resume=resume,
        daily_paths=daily_paths, dates=dates)
    if reproduction is None:
        reproduction = hz.run_reference_reproduction(
            experiment_id, root, daily_paths, dates, logger=logger)
        hz.write_json_atomic(reproduction_path, reproduction)
        hz.write_checkpoint_stage(root, "reference_reproduction", [reproduction_path],
                                  {"passes": reproduction["passes"]})
    log.info("[reference_reproduction] passes=%s failures=%s",
             reproduction["passes"], reproduction["failures"])

    height, width = hz.raster_shape(hz.reference_grid_path(experiment_id))
    resources["grid"] = {"height": height, "width": width}
    resources["daily_contract_versions"] = dict(hz.DAILY_CONTRACT_VERSIONS)
    resources["rss_mib_after_reproduction"] = audit.process_rss_mib()

    if not reproduction["passes"]:
        return _finalise(
            experiment_id, root, dates, inventory,
            state=state, config=config, provenance=provenance,
            reproduction=reproduction, graph=None, diagnostics=None,
            solution=None, sensitivity=[], invariance=None, changes=None,
            evaluation=None, tradeoff=None, resources=resources,
            started=started, inputs_valid=True, invalid_reasons=[],
            date_valid_counts=None, histograms=None)

    # --- stage 4: overlap graph construction ------------------------------
    store, date_valid_counts = hz.run_overlap_evidence(
        daily_paths, height, width, logger=logger)
    date_entries = OrderedDict(
        (date, {**inventory[date], "valid_pixel_count": date_valid_counts[index]})
        for index, date in enumerate(dates)
    )
    grid_cells = height * width
    graphs = []
    for thresholds in hz.SENSITIVITY_THRESHOLDS:
        graph = hz.build_overlap_graph(
            dates, store,
            min_common_pixels=thresholds["min_common_pixels"],
            min_independent_blocks=thresholds["min_independent_blocks"],
            date_entries=date_entries, grid_cells=grid_cells)
        diagnostics = hz.build_graph_diagnostics(dates, graph, date_entries)
        graphs.append(OrderedDict((("label", thresholds["label"]), ("graph", graph),
                                   ("diagnostics", diagnostics))))
    primary = graphs[0]
    graph, diagnostics = primary["graph"], primary["diagnostics"]

    components_path = layout["graph"] / "graph_components.json"
    diagnostics_path = layout["graph"] / "graph_diagnostics.json"
    hz.write_json_atomic(components_path, OrderedDict((
        ("date_count", diagnostics["date_count"]),
        ("connected", diagnostics["connected"]),
        ("connected_component_count", diagnostics["connected_component_count"]),
        ("components", diagnostics["components"]),
        ("isolated_dates", diagnostics["isolated_dates"]),
        ("drop_policy", diagnostics["drop_policy"]),
    )))
    hz.write_json_atomic(diagnostics_path, diagnostics)
    hz.write_checkpoint_stage(root, "overlap_graph_construction",
                              [components_path, diagnostics_path],
                              {"edge_count": graph["edge_count"],
                               "connected": diagnostics["connected"]})
    log.info("[overlap_graph_construction] %s eligible edges, %s component(s), "
             "connected=%s", graph["edge_count"],
             diagnostics["connected_component_count"], diagnostics["connected"])

    # --- stage 5: graph solution ------------------------------------------
    observation_counts = {d: float(date_valid_counts[i]) for i, d in enumerate(dates)}
    solution = None
    if diagnostics["connected"]:
        solution = hz.solve_date_offsets(dates, graph["edges"], observation_counts)
        log.info("[graph_solution] max|alpha|=%.4f C, weighted mean=%.3e C, "
                 "edge residual RMS=%.4f C, cond=%.3e",
                 solution["max_abs_offset_celsius"],
                 solution["weighted_mean_offset_celsius"],
                 solution["edge_residual_rms_celsius"],
                 solution["graph_condition_number"])
    for entry in graphs:
        entry["solution"] = (
            solution if entry is primary else (
                hz.solve_date_offsets(dates, entry["graph"]["edges"], observation_counts)
                if entry["diagnostics"]["connected"] else None
            )
        )

    nodes_path = layout["graph"] / "date_nodes.csv"
    edges_path = layout["graph"] / "date_edges.csv"
    offsets_path = layout["graph"] / "date_offsets.csv"
    hz.write_csv(nodes_path,
                 hz.date_node_rows(dates, inventory, diagnostics, date_valid_counts),
                 hz.DATE_NODE_COLUMNS)
    hz.write_csv(edges_path, hz.date_edge_rows(graph, solution), hz.DATE_EDGE_COLUMNS)
    hz.write_csv(offsets_path, (solution or {}).get("offsets") or [],
                 hz.DATE_OFFSET_COLUMNS)
    sensitivity_path = layout["tables"] / "date_offset_sensitivity.csv"
    hz.write_csv(sensitivity_path, hz.sensitivity_rows(graphs), hz.SENSITIVITY_COLUMNS)
    hz.write_checkpoint_stage(root, "graph_solution",
                              [nodes_path, edges_path, offsets_path, sensitivity_path],
                              {"solved": solution is not None})

    if solution is None:
        log.error("[graph_solution] the primary graph is NOT connected; no "
                  "harmonized candidate raster will be presented as valid.")
        return _finalise(
            experiment_id, root, dates, inventory,
            state=state, config=config, provenance=provenance,
            reproduction=reproduction, graph=graph, diagnostics=diagnostics,
            solution=None, sensitivity=graphs, invariance=None, changes=None,
            evaluation=None, tradeoff=None, resources=resources,
            started=started, inputs_valid=True, invalid_reasons=[],
            date_valid_counts=date_valid_counts, histograms=None)

    # --- stages 6-9: harmonisation, composite, invariance, derived --------
    harmonised = hz.run_harmonisation(
        experiment_id, root, daily_paths, dates, solution["alpha_by_date"],
        logger=logger)
    invariance = harmonised["support_invariance"]
    changes = harmonised["raster_changes"]
    raster_paths = [Path(p) for p in harmonised["outputs"].values()]
    hz.write_checkpoint_stage(root, "daily_harmonisation",
                              [Path(p) for p in harmonised["harmonized_daily_paths"]])
    hz.write_checkpoint_stage(root, "candidate_composite", raster_paths)
    invariance_path = root / "support_invariance.json"
    hz.write_json_atomic(invariance_path, invariance)
    hz.write_checkpoint_stage(root, "support_invariance", [invariance_path],
                              {"passes": invariance["passes"]})
    changes_path = layout["tables"] / "raster_change_summary.csv"
    hz.write_csv(changes_path, hz.raster_change_rows(changes),
                 hz.raster_change_columns(changes))
    hz.write_checkpoint_stage(root, "derived_products", [changes_path])
    log.info("[support_invariance] passes=%s failed=%s",
             invariance["passes"], invariance["failed_checks"])
    resources["rss_mib_after_harmonisation"] = audit.process_rss_mib()

    if not invariance["passes"]:
        return _finalise(
            experiment_id, root, dates, inventory,
            state=state, config=config, provenance=provenance,
            reproduction=reproduction, graph=graph, diagnostics=diagnostics,
            solution=solution, sensitivity=graphs, invariance=invariance,
            changes=changes, evaluation=None, tradeoff=None, resources=resources,
            started=started, inputs_valid=True, invalid_reasons=[],
            date_valid_counts=date_valid_counts, histograms=None)

    # --- stages 10-11: boundary analysis and bootstrap --------------------
    analysis = hz.run_boundary_analysis(experiment_id, root, logger=logger)
    log.info("[boundary_analysis] %s adjacency pairs; %s dropped for a missing "
             "endpoint", analysis["pair_counts"]["total"],
             analysis["pair_counts"]["dropped_invalid_endpoint"])
    evaluation = hz.evaluate_boundaries(analysis)
    evaluation["pathrow_evidence"] = analysis["pathrow_evidence"]
    tradeoff = hz.nonboundary_tradeoff(evaluation)

    boundary_path = layout["tables"] / "boundary_jump_comparison.csv"
    bootstrap_path = layout["tables"] / "paired_bootstrap_summary.csv"
    tradeoff_path = layout["tables"] / "nonboundary_tradeoff.csv"
    rows = hz.boundary_rows(evaluation)
    hz.write_csv(boundary_path, rows, hz.BOUNDARY_COLUMNS)
    hz.write_csv(bootstrap_path, rows, hz.BOOTSTRAP_COLUMNS)
    hz.write_csv(tradeoff_path, tradeoff["rows"], hz.NONBOUNDARY_COLUMNS)
    hz.write_checkpoint_stage(root, "boundary_analysis", [boundary_path])
    hz.write_checkpoint_stage(root, "bootstrap", [bootstrap_path, tradeoff_path])
    resources["rss_mib_after_boundary_analysis"] = audit.process_rss_mib()
    resources["pair_counts"] = analysis["pair_counts"]
    resources["boundary_pair_counts"] = analysis["boundary_pair_counts"]

    return _finalise(
        experiment_id, root, dates, inventory,
        state=state, config=config, provenance=provenance,
        reproduction=reproduction, graph=graph, diagnostics=diagnostics,
        solution=solution, sensitivity=graphs, invariance=invariance,
        changes=changes, evaluation=evaluation, tradeoff=tradeoff,
        resources=resources, started=started, inputs_valid=True,
        invalid_reasons=[], date_valid_counts=date_valid_counts,
        histograms=analysis["anomaly_jump_histograms"])


def _quarantine_incompatible_dailies(root: Path, daily_paths, contract: dict) -> Path:
    """Move contract-violating daily rasters aside; never delete, never repair.

    The files are evidence of the defect, so they are MOVED inside this
    experiment's own namespace rather than removed. Their physical values cannot
    be repaired locally: only Earth Engine holds the true per-date readings at
    the affected pixels, and masking them would drop pixels the frozen composite
    legitimately carries.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = Path(root) / "daily" / "_incompatible" / stamp
    hz.assert_namespace_safe([quarantine], _EXPERIMENT_FOR_SAFETY[0])
    quarantine.mkdir(parents=True, exist_ok=True)
    for path in daily_paths:
        path = Path(path)
        if path.exists():
            path.replace(quarantine / path.name)
    hz.write_json_atomic(quarantine / "why_incompatible.json", OrderedDict((
        ("root_cause", contract["root_cause"]),
        ("failures", contract["failures"]),
        ("versions", contract["versions"]),
        ("constant_fill", contract["constant_fill"]),
        ("remedy",
         "Re-export ONLY these seven diagnostic daily current-period rasters. "
         "No frozen, baseline or production output is involved."),
        ("quarantined_at", datetime.now(timezone.utc).isoformat()),
    )))
    log.error("[daily_mosaic_inventory] %s incompatible daily raster(s) moved to %s",
              len(list(quarantine.glob("*.tif"))), quarantine)
    return quarantine


def _stale_reproduction_reasons(payload: dict, daily_paths, dates) -> list[str]:
    """Why a recorded reproduction verdict may no longer be trusted.

    Any change to the daily raster bytes, the daily semantic version, the
    reconstruction implementation version or the nodata/mask policy version
    invalidates the verdict.
    """
    reasons: list[str] = []
    recorded_versions = payload.get("versions") or {}
    for key, current in hz.DAILY_CONTRACT_VERSIONS.items():
        recorded = recorded_versions.get(key)
        if recorded != current:
            reasons.append(f"{key} changed ({recorded!r} -> {current!r})")

    recorded_hashes = payload.get("daily_raster_hashes") or {}
    if not recorded_hashes:
        reasons.append("no daily raster hashes were recorded with the verdict")
    else:
        for date, path in zip(dates, daily_paths):
            path = Path(path)
            if not path.exists():
                reasons.append(f"daily raster for {date} is missing")
                continue
            current = hz.sha256_and_size(path)["sha256"]
            if recorded_hashes.get(date) != current:
                reasons.append(f"daily raster for {date} changed on disk")
    return reasons


def _reuse_reference_reproduction(root: Path, reproduction_path: Path, *,
                                  resume: bool, daily_paths=(), dates=()) -> dict | None:
    """Reuse the frozen reproduction verdict under `--resume`, or return None.

    A report-generation-only failure must not re-stream seven daily mosaics and
    re-derive the reproduction verdict. Reuse is allowed ONLY when the recorded
    checkpoint output still validates by size AND sha256, and every reference
    raster the stage produced is still on disk -- otherwise the stage is redone.
    Nothing is recomputed, adjusted or re-decided on the reused path.
    """
    if not resume:
        return None
    if not hz.stage_is_reusable(root, "reference_reproduction"):
        log.info("[reference_reproduction] --resume: no hash-valid checkpoint; "
                 "recomputing from the daily mosaics")
        return None
    payload = hz._read_json(reproduction_path)
    if not payload:
        log.warning("[reference_reproduction] --resume: %s is unreadable; "
                    "recomputing", reproduction_path)
        return None

    # A verdict computed under different daily data or different semantics is
    # STALE. Reusing the failed verdict from before this fix would silently
    # re-emit invalid_reference_reproduction, so every version and every daily
    # hash must match before the verdict may be trusted.
    stale = _stale_reproduction_reasons(payload, daily_paths, dates)
    if stale:
        log.warning("[reference_reproduction] --resume: the recorded verdict is "
                    "STALE and will be recomputed: %s", "; ".join(stale))
        return None
    missing = [name for name, path in (payload.get("outputs") or {}).items()
               if not Path(path).exists()]
    if missing:
        log.warning("[reference_reproduction] --resume: reference raster(s) %s "
                    "are missing; recomputing", missing)
        return None
    log.info("[reference_reproduction] --resume: reusing the hash-validated "
             "verdict from %s (passes=%s); no daily mosaic was re-read and no "
             "scientific value was recomputed",
             reproduction_path, payload.get("passes"))
    return payload


def _empty_evaluation() -> dict:
    """A well-formed but empty evaluation, so a failed gate still reports shape."""
    empty = OrderedDict()
    for product in hz.TARGET_PRODUCTS:
        empty[product] = OrderedDict(
            (boundary, hz.bootstrap_paired_reduction(
                hz.MeanAccumulator(),
                hz.MeanAccumulator() if mode == hz.EVAL_MODE_EXCESS else None,
                hz.MeanAccumulator(),
                hz.MeanAccumulator() if mode == hz.EVAL_MODE_EXCESS else None,
                mode=mode))
            for boundary, mode in hz.EVALUATED_BOUNDARIES.items()
        )
        for boundary in empty[product]:
            empty[product][boundary]["product"] = product
            empty[product][boundary]["boundary"] = boundary
            empty[product][boundary]["units"] = hz.PRODUCT_UNITS[product]
    return OrderedDict((("boundary_reductions", empty),
                        ("matching_diagnostics", OrderedDict()),
                        ("evaluation_modes", dict(hz.EVALUATED_BOUNDARIES)),
                        ("pathrow_evidence", None)))


def _finalise(experiment_id: str, root: Path, dates, inventory, *, state, config,
              provenance, reproduction, graph, diagnostics, solution, sensitivity,
              invariance, changes, evaluation, tradeoff, resources, started,
              inputs_valid, invalid_reasons, date_valid_counts, histograms) -> dict:
    """Maps, decision, reports and manifest. Never alters a scientific metric."""
    layout = hz.plan_output_layout(experiment_id)

    if diagnostics is None:
        diagnostics = OrderedDict((
            ("date_count", len(dates)), ("dates", list(dates)), ("edge_count", 0),
            ("rejected_edge_count", 0), ("connected_component_count", 0),
            ("connected", False), ("components", []), ("degree_per_date", {}),
            ("isolated_dates", list(dates)), ("articulation_nodes", []),
            ("cycle_consistency", {"independent_cycle_count": 0,
                                   "tree_edge_count": 0, "non_tree_edge_count": 0,
                                   "max_abs_closure_error_celsius": None,
                                   "median_abs_closure_error_celsius": None,
                                   "cycles": []}),
            ("dates_dropped", []),
            ("drop_policy", "no date is ever silently dropped"),
        ))
    if graph is None:
        graph = OrderedDict((
            ("min_common_pixels", hz.PRIMARY_MIN_COMMON_PIXELS),
            ("min_independent_blocks", hz.PRIMARY_MIN_INDEPENDENT_BLOCKS),
            ("min_block_common_pixels", hz.MIN_BLOCK_COMMON_PIXELS),
            ("edge_count", 0), ("rejected_edge_count", 0),
            ("edges", []), ("rejected_edges", []),
        ))
    if invariance is None:
        invariance = OrderedDict((("checks", []), ("failed_checks", []),
                                  ("passes", False),
                                  ("purpose", "not reached"),
                                  ("required", {})))
    # ONE canonical schema for the JSON summary, the CSV table and the Markdown
    # renderer. Computed rows pass through untouched; a gate-stopped run gets
    # explicitly not-computed rows carrying the full schema, so no metric is
    # ever silently replaced by zero, null or NaN.
    changes = hz.normalise_raster_changes(
        changes, section="raster_changes",
        reason=(
            "an ordered gate stopped the experiment before a candidate "
            "composite existed: "
            + ("reference reproduction failed"
               if not reproduction.get("passes")
               else "the primary date-overlap graph was not connected")
        ),
    )
    if evaluation is None:
        evaluation = _empty_evaluation()
    if tradeoff is None:
        tradeoff = hz.nonboundary_tradeoff(evaluation)

    # --- maps --------------------------------------------------------------
    map_paths: list[Path] = []
    if solution is not None and invariance.get("passes"):
        try:
            map_paths.extend(hz.render_product_maps(root))
            map_paths.extend(hz.render_support_boundary_maps(experiment_id, root))
            if histograms is not None:
                map_paths.append(hz.render_top_residual_jump_map(root, histograms))
        except Exception as error:                          # noqa: BLE001
            log.warning("[maps] product maps skipped: %s", error)
    try:
        map_paths.extend(hz.render_graph_maps(root, dates, graph, diagnostics, solution))
    except Exception as error:                              # noqa: BLE001
        log.warning("[maps] graph maps skipped: %s", error)
    hz.write_checkpoint_stage(root, "maps", map_paths, {"map_count": len(map_paths)})

    # --- decision ----------------------------------------------------------
    evidence = hz.build_evidence(
        reproduction, diagnostics, solution, invariance, changes, evaluation,
        inputs_valid=inputs_valid, invalid_reasons=invalid_reasons)
    decision = hz.decide_final_status(evidence)

    resources["elapsed_seconds"] = round(time.time() - started, 2)
    resources["rss_mib_final"] = audit.process_rss_mib()
    resources["log_file"] = str(log_file)

    summary = hz.build_summary(
        experiment_id, state=state, config=config, provenance=provenance,
        inventory=inventory, reproduction=reproduction, graph=graph,
        diagnostics=diagnostics, solution=solution, sensitivity=sensitivity,
        invariance=invariance, changes=changes, evaluation=evaluation,
        tradeoff=tradeoff, decision=decision, resources=resources)

    # Validate the assembled summary against the canonical schema BEFORE any
    # renderer touches it, so a drift is an explicit contract failure naming the
    # section, product and missing keys rather than a KeyError inside Markdown.
    hz.validate_raster_change_rows(summary["raster_changes"],
                                   section="harmonization_summary.raster_changes")

    before = json.loads(json.dumps(summary, default=str))
    markdown = hz.render_summary_markdown(summary)
    after = json.loads(json.dumps(summary, default=str))
    if not hz.report_generation_preserves_metrics(before, after):
        raise HarmonizationRunnerError(
            "report generation altered a scientific metric; refusing to write."
        )
    if not hz.summary_forbids_banned_conclusions(summary):
        raise HarmonizationRunnerError(
            "the summary claims a forbidden conclusion; refusing to write."
        )

    summary_path = root / "harmonization_summary.json"
    markdown_path = root / "harmonization_summary.md"
    hz.write_json_atomic(summary_path, summary)
    tmp = markdown_path.parent / f".{markdown_path.name}.tmp"
    tmp.write_text(markdown, encoding="utf-8")
    tmp.replace(markdown_path)

    manifest = hz.build_manifest(experiment_id, root, summary)
    manifest_path = root / "harmonization_manifest.json"
    hz.write_json_atomic(manifest_path, manifest)
    hz.write_checkpoint_stage(root, "reports",
                              [summary_path, markdown_path, manifest_path],
                              {"final_status": summary["final_status"]})

    log.info("[reports] final status: %s", summary["final_status"])
    log.info("[reports] %s", summary["final_status_meaning"])
    log.info("[reports] elapsed=%.1fs rss=%.1f MiB maps=%s files=%s",
             resources["elapsed_seconds"], resources["rss_mib_final"],
             len(map_paths), manifest["file_count"])
    return {
        "experiment_id": experiment_id,
        "ran": True,
        "output_root": str(root),
        "final_status": summary["final_status"],
        "map_count": len(map_paths),
        "file_count": manifest["file_count"],
    }


# =============================================================================
# CLI
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic current-period Landsat acquisition-date offset "
            "harmonization counterfactual. Never changes the production "
            "reducer, never recomputes the baseline, never smooths a raster."
        )
    )
    parser.add_argument("--experiment", required=True,
                        help="experiment id (only manavgat_2021 is supported)")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and print the full plan; write nothing")
    parser.add_argument("--run", action="store_true",
                        help="execute the experiment")
    parser.add_argument("--resume", action="store_true",
                        help="reuse hash-validated completed stages (requires --run)")
    parser.add_argument("--force", action="store_true",
                        help="delete ONLY this experiment's diagnostic namespace "
                             "and rebuild it (requires --run)")
    parser.add_argument("--force-daily-export", action="store_true",
                        help="re-export the seven diagnostic daily current-period "
                             "rasters even when local copies exist (use after a "
                             "daily-raster contract failure); touches no frozen, "
                             "baseline or production output")
    parser.add_argument("--no-earth-engine", action="store_true",
                        help="forbid the isolated daily-mosaic export stage; the "
                             "run then fails if a daily mosaic is missing locally")
    return parser


def main(experiment_id: str | None = None, dry_run: bool = False, run: bool = False,
         resume: bool = False, force: bool = False,
         no_earth_engine: bool = False, force_daily_export: bool = False) -> dict:
    validate_modes(dry_run, run, resume, force)
    if not experiment_id:
        raise HarmonizationRunnerError("--experiment is required.")
    hz.assert_supported_experiment(experiment_id)

    if dry_run:
        # The guard is installed for the dry-run too: it must be IMPOSSIBLE for
        # the planning path to reach Earth Engine.
        with ab.EarthEngineGuard():
            plan = hz.build_dry_run_plan(experiment_id)
        _print_dry_run(plan)
        return plan

    return _run_live(experiment_id, resume=resume, force=force,
                     allow_earth_engine=not no_earth_engine,
                     force_daily_export=force_daily_export)


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(experiment_id=args.experiment, dry_run=args.dry_run, run=args.run,
         resume=args.resume, force=args.force, no_earth_engine=args.no_earth_engine,
         force_daily_export=args.force_daily_export)
