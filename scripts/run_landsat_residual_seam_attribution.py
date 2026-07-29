#!/usr/bin/env python3
"""
scripts/run_landsat_residual_seam_attribution.py

Dedicated entry point for the RESIDUAL SEAM ATTRIBUTION audit of the Manavgat
date-balanced candidate, run after the completed downstream A/B experiment.

    python scripts/run_landsat_residual_seam_attribution.py \
        --experiment manavgat_2021 \
        --dry-run

CONTRACT
--------
    - Exactly one of --dry-run / --run. Default execution is never implied.
    - --resume and --force are mutually exclusive, and both require --run.
    - --dry-run writes nothing and creates no directory.
    - No Earth Engine code path is reachable. The live run additionally installs
      a runtime guard that makes every EE entry point raise.
    - Step5-Step8 are NEVER re-run: every input is a frozen local file.
    - Nothing is smoothed, blended, interpolated or cosmetically altered. The
      audit measures the seam; it never removes it.
    - Everything is written under
      outputs/diagnostics/landsat_residual_seam_attribution/<experiment_id>/.
      The frozen downstream A/B, counterfactual and canonical namespaces are
      read-only inputs and are never modified.
    - The audit can never report that the seam is fixed and never issues a
      production decision.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.landsat_composite_downstream_ab as ab
import src.landsat_residual_seam_attribution as rs
from core.io_utils import setup_logger

log, log_file = setup_logger("landsat_residual_seam_attribution")


class ResidualSeamRunnerError(SystemExit):
    """Fail-fast CLI error (same convention as the other runners)."""


# =============================================================================
# CSV helper
# =============================================================================
def _write_csv(path: Path, rows: list[dict], columns) -> Path:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)
    return path


# =============================================================================
# Argument validation
# =============================================================================
def validate_modes(dry_run: bool, run: bool, resume: bool, force: bool) -> None:
    """Exactly one mode; --resume/--force are run-only and mutually exclusive."""
    if dry_run and run:
        raise ResidualSeamRunnerError(
            "--dry-run and --run are mutually exclusive; pass exactly one."
        )
    if not dry_run and not run:
        raise ResidualSeamRunnerError(
            "one of --dry-run or --run is required. Default execution is never "
            "implied by this runner."
        )
    if resume and force:
        raise ResidualSeamRunnerError("--resume and --force are mutually exclusive.")
    if resume and not run:
        raise ResidualSeamRunnerError("--resume requires --run.")
    if force and not run:
        raise ResidualSeamRunnerError("--force requires --run.")


# =============================================================================
# Dry-run
# =============================================================================
def _print_dry_run(plan: dict) -> None:
    log.info("[dry-run] audit: %s", plan["audit"])
    log.info("[dry-run] experiment: %s", plan["experiment_id"])
    log.info("[dry-run] chain under attribution: %s", plan["chain_under_attribution"])
    log.info("[dry-run] target products: %s", plan["target_products"])

    log.info("[dry-run] --- resolved inputs ---")
    for role, entry in plan["resolved_inputs"].items():
        log.info(
            "[dry-run]   %-42s present=%-5s required=%-5s %s",
            role, entry["present"], entry["required"], entry["path"],
        )
    if plan["missing_required_inputs"]:
        log.error("[dry-run]   MISSING REQUIRED: %s", plan["missing_required_inputs"])
    log.info("[dry-run] --- missing optional provenance inputs ---")
    if plan["missing_optional_provenance_inputs"]:
        for item in plan["missing_optional_provenance_inputs"]:
            log.warning("[dry-run]   missing optional: %s", item)
    else:
        log.info("[dry-run]   none")

    pathrow = plan["pathrow_evidence"]
    log.info("[dry-run] --- path/row evidence ---")
    log.info("[dry-run]   availability: %s", pathrow["availability"])
    log.info("[dry-run]   reason: %s", pathrow["reason"])
    log.info("[dry-run]   distinct metadata interfaces: %s", pathrow["interface_count"])
    for entry in pathrow.get("interfaces") or []:
        log.info("[dry-run]     %-24s %s features",
                 entry["interface_id"], entry["feature_count"])
    log.info("[dry-run]   evidence is METADATA-DERIVED, not pixel-level "
             "selected-scene provenance")

    prereq = plan["upstream_prerequisites"]
    log.info("[dry-run] --- upstream prerequisites ---")
    for key, value in prereq.items():
        log.info("[dry-run]   %-38s %s", key, value)

    log.info("[dry-run] output root: %s", plan["output_root"])

    log.info("[dry-run] --- decomposition formulas ---")
    for product, formula in plan["decomposition_formulas"].items():
        log.info("[dry-run]   %s", product)
        for key, value in formula.items():
            log.info("[dry-run]     %-26s %s", key, value)

    classes = plan["boundary_classes"]
    log.info("[dry-run] --- anomaly reconstruction checks (SEPARATE) ---")
    checks = plan["anomaly_reconstruction_checks"]
    identity = checks["algebraic_identity_check"]
    log.info("[dry-run]   1. algebraic identity check (GATES the audit)")
    log.info("[dry-run]      %s", identity["question"])
    log.info("[dry-run]      computed in: %s", identity["computed_in"])
    log.info("[dry-run]      tolerance absolute: %s | relative: %s",
             identity["tolerance_absolute"], identity["tolerance_relative"])
    log.info("[dry-run]      tolerance policy: %s", identity["tolerance_policy"])
    log.info("[dry-run]      failure meaning: %s", identity["failure_meaning"])
    stored = checks["stored_raster_reproduction_check"]
    log.info("[dry-run]   2. stored-raster reproduction check (DESCRIPTIVE ONLY)")
    log.info("[dry-run]      %s", stored["question"])
    log.info("[dry-run]      compared: %s", stored["compared"])
    log.info("[dry-run]      predeclared tolerance: %s (source: %s)",
             stored["predeclared_tolerance"], stored["predeclared_tolerance_source"])
    log.info("[dry-run]      Step5 reproduction policy bound: %s (source: %s)",
             stored["step5_reproduction_policy_tolerance"],
             stored["step5_reproduction_policy_source"])
    log.info("[dry-run]      reported percentiles: %s", stored["reported_percentiles"])
    log.info("[dry-run]      gates the audit: %s | is a decomposition failure: %s",
             stored["gates_the_audit"], stored["is_decomposition_failure"])
    log.info("[dry-run]      %s", stored["interpretation"])

    log.info("[dry-run] --- boundary classes ---")
    log.info("[dry-run]   raw flags: %s", classes["raw_flags"])
    log.info("[dry-run]   stratified classes: %s", classes["stratified_classes"])
    log.info("[dry-run]   excess-jump boundaries: %s", classes["excess_jump_boundaries"])

    thresholds = plan["thresholds"]
    log.info("[dry-run] --- thresholds and sensitivity epsilons ---")
    for key, value in thresholds["step5"].items():
        log.info("[dry-run]   step5 %-30s %s", key, value)
    log.info("[dry-run]   near-std epsilon (PRIMARY, predeclared): %s",
             thresholds["near_std_threshold_epsilon_primary"])
    log.info("[dry-run]   near-std epsilon sensitivity: %s",
             thresholds["near_std_threshold_epsilon_sensitivity"])
    log.info("[dry-run]   hotspot percentiles (DESCRIPTIVE ONLY): %s",
             thresholds["hotspot_percentiles_descriptive_only"])
    log.info("[dry-run]   dominance share lower bound: %s",
             thresholds["dominance_share_lower_bound"])

    log.info("[dry-run] --- bootstrap configuration ---")
    for key, value in plan["bootstrap_configuration"].items():
        log.info("[dry-run]   %-38s %s", key, value)

    log.info("[dry-run] --- matched-control strategy ---")
    control = plan["matched_control_strategy"]
    log.info("[dry-run]   %s", control["strategy"])
    for key in ("elevation_gradient_bins_m", "slope_gradient_bins_deg", "ndvi_gradient_bins"):
        log.info("[dry-run]   %-38s %s", key, control[key])

    log.info("[dry-run] decision-rule version: %s", plan["decision_rule_version"])
    log.info("[dry-run] allowed final statuses: %s", plan["allowed_final_statuses"])
    log.info("[dry-run] --- planned stages ---")
    for stage in plan["planned_stages"]:
        log.info("[dry-run]   stage %s", stage)
    log.info("[dry-run] --- expected files ---")
    for name, path in plan["expected_files"].items():
        log.info("[dry-run]   %-52s %s", name, path)
    log.info("[dry-run] NO writes, NO directories created, NO Earth Engine call, "
             "NO Step5-Step8 rerun, NO smoothing.")


# =============================================================================
# Live stages
# =============================================================================
def _grid_profile(path: Path) -> dict:
    import rasterio

    with rasterio.open(path) as src:
        profile = dict(src.profile)
    profile.pop("nodata", None)
    profile.pop("dtype", None)
    profile.pop("count", None)
    return profile


def _stage_log(records: list, stage: str, phase: str, started: float | None = None, **extra):
    record = OrderedDict((
        ("stage", stage),
        ("phase", phase),
        ("rss_mib", rs.process_rss_mib()),
    ))
    if started is not None:
        record["elapsed_s"] = round(time.time() - started, 3)
    record.update(extra)
    records.append(record)
    log.info(
        "[stage %s:%s] rss=%s MiB%s%s", stage, phase, record["rss_mib"],
        f" elapsed={record['elapsed_s']}s" if started is not None else "",
        "".join(f" {k}={v}" for k, v in extra.items()),
    )
    return record


def _run_live(experiment_id: str, force: bool, resume: bool) -> dict:
    """Execute the whole attribution audit locally. No Earth Engine, ever."""
    import rasterio

    rs.assert_supported_experiment(experiment_id)
    root = rs.diagnostic_output_root(experiment_id)
    resources: list[dict] = []
    invalid_reasons: list[str] = []

    # --- stage: input_validation -------------------------------------------
    started = time.time()
    _stage_log(resources, "input_validation", "begin")
    state = rs.load_upstream_state(experiment_id)
    rs.validate_upstream_state(state)

    plan = rs.build_input_plan(experiment_id)
    rs.assert_required_inputs(plan, experiment_id)
    grid = rs.assert_grid_contract(plan)
    pathrow_availability = rs.resolve_pathrow_availability(experiment_id)

    if force:
        removed = rs.clear_diagnostic_namespace(experiment_id)
        log.info("[--force] removed ONLY the dedicated audit namespace: %s", removed)

    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)

    config = rs.build_config_snapshot(experiment_id)
    config_path = root / "config" / "residual_seam_config.json"
    rs.assert_namespace_safe([config_path], experiment_id)
    rs.write_json_atomic(config_path, config)

    provenance = rs.build_input_provenance(
        experiment_id, plan, state=state, grid=grid, pathrow=pathrow_availability,
    )
    provenance_path = root / "input_provenance.json"
    rs.assert_namespace_safe([provenance_path], experiment_id)
    rs.write_json_atomic(provenance_path, provenance)
    rs.write_checkpoint_stage(
        root, "input_validation", [config_path, provenance_path],
        {"grid_contract_passed": grid["passed"],
         "pathrow_availability": pathrow_availability["availability"]},
    )
    _stage_log(resources, "input_validation", "end", started,
               rasters=len(provenance["raster_inputs"]))

    with rasterio.open(Path(plan[rs.TARGET_CMB]["path"])) as src:
        height, width = int(src.height), int(src.width)
        transform = src.transform
    grid_profile = _grid_profile(Path(plan[rs.TARGET_CMB]["path"]))

    # --- stage: pair_mask_construction --------------------------------------
    started = time.time()
    _stage_log(resources, "pair_mask_construction", "begin", height=height, width=width)
    pathrow_masks = None
    if pathrow_availability["availability"] == "available":
        geojson = json.loads(
            Path(rs.pathrow_boundary_sources(experiment_id)["scene_boundaries_geojson"])
            .read_text(encoding="utf-8")
        )
        pathrow_masks = rs.rasterize_pathrow_boundaries(geojson, transform, width, height)
        log.info(
            "Rasterized %d metadata-derived path/row interfaces onto the exact grid.",
            len(pathrow_masks["interfaces"]),
        )
    else:
        log.warning(
            "Path/row evidence is %s (%s); that mechanism is reported as "
            "UNAVAILABLE and can never create positive evidence.",
            pathrow_availability["availability"], pathrow_availability["reason"],
        )
    rs.write_checkpoint_stage(
        root, "pair_mask_construction", [],
        {"height": height, "width": width,
         "pathrow_interfaces": len((pathrow_masks or {}).get("interfaces") or {})},
    )
    _stage_log(resources, "pair_mask_construction", "end", started)

    # --- stages: decomposition passes ---------------------------------------
    started = time.time()
    _stage_log(resources, "current_minus_baseline_decomposition", "begin")
    analysis = rs.run_streaming_pass(
        plan, height=height, width=width, pathrow_masks=pathrow_masks, log=log,
    )
    resources.extend(analysis.resource_log)
    total_pairs = int(sum(analysis.pair_counts.values()))
    rs.write_checkpoint_stage(
        root, "current_minus_baseline_decomposition", [],
        {"n_pairs": total_pairs,
         "max_reconstruction_residual": analysis.max_residual[rs.TARGET_CMB]},
    )
    _stage_log(resources, "current_minus_baseline_decomposition", "end", started,
               pairs=total_pairs)

    if analysis.max_residual[rs.TARGET_CMB] > rs.CMB_RECONSTRUCTION_ABS_TOL:
        invalid_reasons.append(
            "current-minus-baseline reconstruction exceeded the documented tolerance "
            f"({analysis.max_residual[rs.TARGET_CMB]} > {rs.CMB_RECONSTRUCTION_ABS_TOL})"
        )
    # CHECK 1 -- the algebraic identity. This is the ONLY anomaly check that can
    # invalidate the audit: a failure means the decomposition itself is wrong.
    identity_check = rs.build_anomaly_identity_check(analysis)
    if not identity_check["passed"]:
        invalid_reasons.append(
            "the anomaly algebraic identity failed for "
            f"{identity_check['n_pairs_exceeding_tolerance']} of "
            f"{identity_check['n_pairs_checked']} pairs (max residual "
            f"{identity_check['max_absolute_residual']}, max residual/tolerance "
            f"{identity_check['max_residual_over_tolerance']}); tolerance policy: "
            f"{identity_check['tolerance_policy']}"
        )

    # CHECK 2 -- stored-raster reproduction. Descriptive: expected float32
    # serialization error is NEVER treated as a decomposition failure, so this
    # never contributes an invalid reason.
    stored_check = rs.build_stored_reproduction_check(analysis)
    log.info(
        "Anomaly identity check: passed=%s (%d pairs, %d exceeding, max residual %.3g)",
        identity_check["passed"], identity_check["n_pairs_checked"],
        identity_check["n_pairs_exceeding_tolerance"],
        identity_check["max_absolute_residual"],
    )
    log.info(
        "Anomaly stored-raster reproduction: status=%s (%d pixels, max abs error "
        "%.3g, predeclared tolerance %.3g); this is float32 serialization error "
        "and is NOT a decomposition failure.",
        stored_check["status"], stored_check["n_pixels_checked"],
        stored_check["max_abs_error"], stored_check["predeclared_tolerance"],
    )
    if stored_check["status"] == "exceeds_step5_reproduction_policy":
        log.warning(
            "Stored anomaly reproduction exceeds the existing Step5 policy bound "
            "%.3g; reported, but it does NOT invalidate the decomposition.",
            stored_check["step5_reproduction_policy_tolerance"],
        )

    rs.write_checkpoint_stage(
        root, "anomaly_decomposition", [],
        {"algebraic_identity_passed": identity_check["passed"],
         "algebraic_identity_max_residual": identity_check["max_absolute_residual"],
         "algebraic_identity_pairs_exceeding":
             identity_check["n_pairs_exceeding_tolerance"],
         "stored_reproduction_status": stored_check["status"],
         "stored_reproduction_max_abs_error": stored_check["max_abs_error"],
         "stored_reproduction_gates_the_audit": False},
    )

    # --- stage: mask_boundary_analysis --------------------------------------
    started = time.time()
    _stage_log(resources, "mask_boundary_analysis", "begin")
    epsilon_rows = rs.compute_epsilon_rows(analysis)
    mask_report = rs.build_mask_report(analysis, epsilon_rows)
    rs.write_checkpoint_stage(root, "mask_boundary_analysis", [])
    _stage_log(resources, "mask_boundary_analysis", "end", started)

    # --- stage: matched_control_analysis ------------------------------------
    started = time.time()
    _stage_log(resources, "matched_control_analysis", "begin")
    excess_rows = rs.compute_excess_rows(analysis)
    excess_report = rs.build_excess_report(excess_rows)
    rs.write_checkpoint_stage(root, "matched_control_analysis", [],
                              {"boundary_definitions": len(excess_rows)})
    _stage_log(resources, "matched_control_analysis", "end", started)

    # --- stage: pathrow_analysis --------------------------------------------
    started = time.time()
    _stage_log(resources, "pathrow_analysis", "begin")
    pathrow_rows = rs.compute_pathrow_rows(analysis, excess_rows)
    pathrow_report = rs.build_pathrow_report(pathrow_availability, analysis, pathrow_rows)
    rs.write_checkpoint_stage(root, "pathrow_analysis", [],
                              {"verdict": pathrow_report.get("verdict")})
    _stage_log(resources, "pathrow_analysis", "end", started,
               verdict=pathrow_report.get("verdict"))

    # --- stage: bootstrap ----------------------------------------------------
    started = time.time()
    _stage_log(resources, "bootstrap", "begin")
    share_intervals = rs.compute_share_intervals(analysis)
    bootstrap_summary = rs.build_bootstrap_summary(
        analysis, share_intervals, excess_rows, epsilon_rows,
    )
    rs.write_checkpoint_stage(root, "bootstrap", [],
                              {"statistics": len(bootstrap_summary["rows"])})
    _stage_log(resources, "bootstrap", "end", started)

    detection = rs.build_detection_report(analysis)
    cmb_report = rs.build_cmb_report(analysis, share_intervals)
    anomaly_report = rs.build_anomaly_report(analysis, share_intervals)

    # --- stage: map_generation ----------------------------------------------
    started = time.time()
    _stage_log(resources, "map_generation", "begin")
    cuts = rs.hotspot_thresholds(analysis)
    if resume and rs.stage_is_reusable(root, "map_generation"):
        log.info("Diagnostic overlays reused from a validated checkpoint.")
        map_result = {"overlap_counts": {}, "written": []}
        overlap_path = root / "maps" / "hotspot_overlap_counts.json"
        overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
        map_result["overlap_counts"] = {
            tuple(k.split("||")): v for k, v in overlap.items()
        }
    else:
        map_result = rs.run_hotspot_and_map_pass(
            plan, root=root, experiment_id=experiment_id, height=height, width=width,
            grid_profile=grid_profile, pathrow_masks=pathrow_masks,
            thresholds=rs.step5_thresholds(), hotspot_cuts=cuts, log=log,
        )
        overlap_path = root / "maps" / "hotspot_overlap_counts.json"
        rs.assert_namespace_safe([overlap_path], experiment_id)
        rs.write_json_atomic(overlap_path, {
            "||".join(k): v for k, v in map_result["overlap_counts"].items()
        })
        rs.write_checkpoint_stage(
            root, "map_generation",
            [Path(p) for p in map_result["written"]] + [overlap_path],
            {"smoothing_applied": False, "resampling_applied": False},
        )
    hotspots = rs.build_hotspot_report(map_result["overlap_counts"], cuts)
    _stage_log(resources, "map_generation", "end", started,
               maps=len(map_result["written"]))

    # --- decision -------------------------------------------------------------
    evidence = rs.build_decision_evidence(
        inputs_valid=not invalid_reasons, invalid_reasons=invalid_reasons,
        share_intervals=share_intervals, excess_rows=excess_rows,
        epsilon_rows=epsilon_rows, mask_report=mask_report,
        pathrow_report=pathrow_report,
    )
    decision = rs.decide_final_status(evidence)

    # --- stage: report_generation ---------------------------------------------
    started = time.time()
    _stage_log(resources, "report_generation", "begin")
    metrics_before = {
        "cmb": cmb_report["by_class"], "anomaly": anomaly_report["by_class"],
        "excess": excess_rows, "epsilon": epsilon_rows,
        "bootstrap": bootstrap_summary["rows"],
    }

    tables = root / "tables"
    written_tables = []
    cmb_rows, cmb_columns = rs.csv_rows_current_minus_baseline(cmb_report)
    written_tables.append(_write_csv(
        tables / "current_minus_baseline_decomposition.csv", cmb_rows, cmb_columns,
    ))
    anomaly_rows, anomaly_columns = rs.csv_rows_anomaly(anomaly_report)
    written_tables.append(_write_csv(
        tables / "anomaly_decomposition.csv", anomaly_rows, anomaly_columns,
    ))
    mask_rows, mask_columns = rs.csv_rows_simple(
        mask_report["by_stratum"], ["stratum"],
    )
    written_tables.append(_write_csv(
        tables / "mask_discontinuity_summary.csv", mask_rows, mask_columns,
    ))
    excess_csv, excess_columns = rs.csv_rows_simple(excess_rows, ["product", "boundary"])
    written_tables.append(_write_csv(
        tables / "boundary_excess_jump.csv", excess_csv, excess_columns,
    ))
    hotspot_csv, hotspot_columns = rs.csv_rows_simple(
        hotspots["rows"], ["product", "hotspot_class", "mechanism"],
    )
    written_tables.append(_write_csv(
        tables / "hotspot_mechanism_overlap.csv", hotspot_csv, hotspot_columns,
    ))
    pathrow_csv, pathrow_columns = rs.csv_rows_simple(pathrow_rows, ["stratum"])
    written_tables.append(_write_csv(
        tables / "pathrow_stratified_test.csv", pathrow_csv, pathrow_columns,
    ))
    bootstrap_csv, bootstrap_columns = rs.csv_rows_simple(
        bootstrap_summary["rows"], ["statistic"],
    )
    written_tables.append(_write_csv(
        tables / "bootstrap_summary.csv", bootstrap_csv, bootstrap_columns,
    ))
    sample_rows = analysis.sample.rows
    sample_columns = list(sample_rows[0].keys()) if sample_rows else ["row_a"]
    written_tables.append(_write_csv(
        tables / "pair_sample.csv", sample_rows, sample_columns,
    ))
    rs.assert_namespace_safe(written_tables, experiment_id)

    resource_summary = OrderedDict((
        ("windows_processed", analysis.windows_processed),
        ("window_rows", rs.WINDOW_ROWS),
        ("peak_rss_mib", rs.process_rss_mib()),
        ("total_pairs", total_pairs),
        ("pair_sample_rows", len(sample_rows)),
        ("pair_sample_seen", analysis.sample.seen),
        ("stage_log", resources),
    ))

    summary = rs.build_summary(
        experiment_id, config=config, provenance=provenance, state=state,
        detection=detection, cmb=cmb_report, anomaly=anomaly_report,
        mask_analysis=mask_report, excess=excess_report, pathrow=pathrow_report,
        bootstrap_summary=bootstrap_summary, hotspots=hotspots, decision=decision,
        resources=resource_summary,
    )
    summary_path = root / "residual_seam_summary.json"
    markdown_path = root / "residual_seam_summary.md"
    rs.assert_namespace_safe([summary_path, markdown_path], experiment_id)
    rs.write_json_atomic(summary_path, summary)
    markdown = rs.render_summary_markdown(summary)
    markdown_path.write_text(markdown, encoding="utf-8")

    metrics_after = {
        "cmb": cmb_report["by_class"], "anomaly": anomaly_report["by_class"],
        "excess": excess_rows, "epsilon": epsilon_rows,
        "bootstrap": bootstrap_summary["rows"],
    }
    if not rs.report_generation_preserves_metrics(metrics_before, metrics_after):
        raise ResidualSeamRunnerError(
            "report generation mutated a scientific metric payload; refusing to finish."
        )
    if not rs.summary_forbids_banned_conclusions(summary):
        raise ResidualSeamRunnerError(
            "the summary contains a forbidden conclusion; refusing to finish."
        )
    if not rs.summary_forbids_banned_conclusions(markdown):
        raise ResidualSeamRunnerError(
            "the markdown report contains a forbidden conclusion; refusing to finish."
        )

    manifest = rs.build_manifest(experiment_id, root, summary)
    manifest_path = root / "residual_seam_manifest.json"
    rs.assert_namespace_safe([manifest_path], experiment_id)
    rs.write_json_atomic(manifest_path, manifest)
    rs.write_checkpoint_stage(
        root, "report_generation",
        [summary_path, markdown_path, manifest_path, *written_tables],
    )
    _stage_log(resources, "report_generation", "end", started,
               tables=len(written_tables))

    log.info("FINAL STATUS: %s", decision["final_status"])
    log.info("%s", decision["meaning"])
    return {
        "experiment_id": experiment_id, "ran": True, "dry_run": False,
        "final_status": decision["final_status"],
        "seam_fixed": False, "production_approved": False,
        "output_root": str(root), "summary_path": str(summary_path),
        "markdown_path": str(markdown_path), "manifest_path": str(manifest_path),
        "total_pairs": total_pairs, "map_count": len(map_result["written"]),
    }


# =============================================================================
# Entry point
# =============================================================================
def main(
    experiment_id: str,
    dry_run: bool = False,
    run: bool = False,
    resume: bool = False,
    force: bool = False,
) -> dict:
    validate_modes(dry_run, run, resume, force)

    if dry_run:
        try:
            plan = rs.build_dry_run_plan(experiment_id)
        except rs.ResidualSeamError as exc:
            raise ResidualSeamRunnerError(str(exc)) from exc
        _print_dry_run(plan)
        return {
            "experiment_id": experiment_id, "ran": False, "dry_run": True,
            "plan": plan, "writes_performed": False,
        }

    try:
        with ab.EarthEngineGuard():
            return _run_live(experiment_id, force=force, resume=resume)
    except (rs.PrerequisiteError, rs.ResidualSeamError, rs.NamespaceSafetyError,
            rs.GridMismatchError) as exc:
        raise ResidualSeamRunnerError(str(exc)) from exc


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic attribution of the residual seam in the Manavgat "
            "date-balanced candidate (current_minus_baseline_celsius and "
            "anomaly_zscore). Local-only: no Earth Engine, no Step5-Step8 rerun, "
            "no smoothing, no production decision. Writes ONLY under "
            "outputs/diagnostics/landsat_residual_seam_attribution/<experiment_id>/."
        )
    )
    parser.add_argument(
        "--experiment", type=str, required=True,
        choices=list(rs.SUPPORTED_EXPERIMENT_IDS),
        help="Experiment id. Only manavgat_2021 is supported in this task.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run", action="store_true",
        help="Resolve and print the plan; write nothing and create no directory.",
    )
    mode.add_argument(
        "--run", action="store_true",
        help="Explicit opt-in to execute the local attribution audit.",
    )
    state = parser.add_mutually_exclusive_group()
    state.add_argument(
        "--resume", action="store_true",
        help="Requires --run. Reuse checkpointed stages whose recorded outputs "
             "still validate by size AND sha256; recompute everything else.",
    )
    state.add_argument(
        "--force", action="store_true",
        help="Requires --run. Delete ONLY the dedicated residual-seam namespace "
             "after a safety check, then rerun. Frozen A/B, counterfactual and "
             "canonical outputs are never touched.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = main(
        experiment_id=args.experiment,
        dry_run=args.dry_run,
        run=args.run,
        resume=args.resume,
        force=args.force,
    )
    print(json.dumps(result, indent=2, default=str))
