#!/usr/bin/env python3
"""
scripts/run_landsat_harmonization_downstream_ab.py

Dedicated entry point for the isolated DOWNSTREAM A/B experiment comparing

    reference: date_balanced_reference
    candidate: overlap_harmonized_date_balanced

    python scripts/run_landsat_harmonization_downstream_ab.py \
        --experiment manavgat_2021 \
        --candidate overlap_harmonized_date_balanced \
        --dry-run

CONTRACT
--------
    - Exactly one of --dry-run / --run. Default execution is never implied.
    - --resume and --force are mutually exclusive, and both require --run.
    - --dry-run writes nothing and creates no directory.
    - --force may delete ONLY
      outputs/diagnostics/landsat_harmonization_downstream_ab/<experiment_id>/.
    - Earth Engine is UNREACHABLE: the whole live run executes inside the shared
      `EarthEngineGuard`, and no Earth Engine symbol is imported here.
    - The frozen harmonization, previous-A/B, counterfactual and canonical
      namespaces are READ-ONLY inputs and are never written.
    - Only production callables are run: Step5, Step5C, Step7A-E, Step8A. No
      production Step5/7/8 code, no `core` configuration, no production reducer
      and no Step7B guard default is modified.
    - The two chains differ ONLY in band 1 of the current-period Landsat LST
      raster. Band 2 (unique-date valid count) must be bitwise identical, which
      is a blocking gate.
    - The experiment never claims a seam is fixed, production approval or
      readiness, non-inferiority, transfer improvement, cross-region
      generalization, or causality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.landsat_composite_downstream_ab as ab
import src.landsat_harmonization_downstream_ab as hab
from core.io_utils import setup_logger

log, log_file = setup_logger("landsat_harmonization_downstream_ab")


class HarmonizationDownstreamABRunnerError(SystemExit):
    """Fail-fast CLI error (same convention as the sibling runners)."""


# =============================================================================
# CSV helpers
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


def _write_dataframe_csv(path: Path, frame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


# =============================================================================
# Argument validation
# =============================================================================
def validate_modes(dry_run: bool, run: bool, resume: bool, force: bool,
                   report_only: bool = False) -> None:
    """Exactly one mode; --resume/--force are run-only and mutually exclusive.

    `--report-only` is a third, read-mostly mode: it re-renders the Markdown and
    manifest from the EXISTING summary JSON and never runs the pipeline.
    """
    if report_only:
        if dry_run or run:
            raise HarmonizationDownstreamABRunnerError(
                "--report-only cannot be combined with --dry-run or --run.")
        if resume or force:
            raise HarmonizationDownstreamABRunnerError(
                "--report-only cannot be combined with --resume or --force.")
        return
    if dry_run and run:
        raise HarmonizationDownstreamABRunnerError(
            "--dry-run and --run are mutually exclusive; pass exactly one."
        )
    if not dry_run and not run:
        raise HarmonizationDownstreamABRunnerError(
            "one of --dry-run, --run or --report-only is required. Default "
            "execution is never implied by this runner."
        )
    if resume and force:
        raise HarmonizationDownstreamABRunnerError(
            "--resume and --force are mutually exclusive."
        )
    if resume and not run:
        raise HarmonizationDownstreamABRunnerError("--resume requires --run.")
    if force and not run:
        raise HarmonizationDownstreamABRunnerError("--force requires --run.")


# =============================================================================
# Dry-run
# =============================================================================
def _print_dry_run(plan: dict) -> None:
    log.info("[dry-run] experiment: %s", plan["experiment"])
    log.info("[dry-run] experiment_id: %s", plan["experiment_id"])
    log.info("[dry-run] reference chain: %s", plan["reference_chain"])
    log.info("[dry-run] candidate chain: %s", plan["candidate_chain"])
    log.info("[dry-run] output root: %s", plan["output_root"])

    log.info("[dry-run] --- reproduction target ---")
    target = plan["reproduction_target"]
    log.info("[dry-run]   namespace: %s", target["namespace"])
    log.info("[dry-run]   chain: %s (side dir: %s)", target["chain"], target["root"])
    log.info("[dry-run]   canonical scene-weighted chain used: %s",
             target["canonical_scene_weighted_used"])

    log.info("[dry-run] --- upstream prerequisites ---")
    state = plan["upstream_prerequisites"]
    for block in ("harmonization", "previous_downstream_ab"):
        log.info("[dry-run]   [%s]", block)
        for key, value in state[block].items():
            if key == "failures":
                continue
            log.info("[dry-run]     %-46s %s", key, value)
    if state["failures"]:
        log.error("[dry-run]   PREREQUISITE FAILURES:")
        for failure in state["failures"]:
            log.error("[dry-run]     %s", failure)
    else:
        log.info("[dry-run]   all prerequisites met")

    log.info("[dry-run] --- inputs ---")
    for role, entry in plan["resolved_inputs"].items():
        log.info("[dry-run]   %-42s shared=%-5s differs=%-5s ref=%s cand=%s",
                 role, entry["shared"], entry["differs_between_chains"],
                 entry["reference_present"], entry["candidate_present"])
    log.info("[dry-run]   roles that differ between chains: %s",
             plan["roles_that_differ_between_chains"])
    log.info("[dry-run]   ONLY current LST differs: %s",
             plan["only_current_lst_differs"])
    log.info("[dry-run]   shared date-balanced baseline source: %s (%s files)",
             plan["shared_baseline_source"], plan["shared_baseline_file_count"])
    if plan["missing_sources"]:
        log.error("[dry-run]   MISSING SOURCES: %s", plan["missing_sources"])

    log.info("[dry-run] --- chain contexts ---")
    for chain, preview in plan["chain_context_preview"].items():
        log.info("[dry-run]   %s", chain)
        for key, value in preview.items():
            log.info("[dry-run]     %-24s %s", key, value)

    log.info("[dry-run] --- current-support invariance gate ---")
    for key, value in plan["current_support_invariance_gate"].items():
        log.info("[dry-run]   %-32s %s", key, value)

    log.info("[dry-run] --- MODIS ---")
    log.info("[dry-run]   attestation issuer: %s (unchanged)",
             plan["modis_attestation_issuer"])
    modis = plan["modis_compatibility"]
    for key in ("required", "mode", "status", "reason"):
        if key in modis:
            log.info("[dry-run]   %-32s %s", key, modis[key])

    comparison = plan["configuration"]["comparison"]
    log.info("[dry-run] --- comparison ---")
    for key, value in comparison.items():
        log.info("[dry-run]   %-38s %s", key, value)

    log.info("[dry-run] --- planned stages ---")
    for stage in plan["planned_stages"]:
        log.info("[dry-run]   stage %s", stage)

    log.info("[dry-run] --- expected files ---")
    for name, path in plan["expected_files"].items():
        log.info("[dry-run]   %-46s %s", name, path)

    log.info("[dry-run] --- decision ---")
    log.info("[dry-run]   allowed final statuses: %s", plan["allowed_final_statuses"])
    log.info("[dry-run]   forbidden conclusions: %s", plan["forbidden_conclusions"])
    log.info("[dry-run]   rule: %s", plan["decision_rule"])

    log.info("[dry-run] --- limitations ---")
    for limitation in plan["limitations"]:
        log.info("[dry-run]   - %s", limitation)

    log.info("[dry-run] writes performed: %s | directories created: %s | "
             "Earth Engine calls: %s | rasters modified: %s | "
             "frozen namespaces touched: %s",
             plan["writes_performed"], plan["directories_created"],
             plan["earth_engine_calls"], plan["rasters_modified"],
             plan["frozen_namespaces_touched"])


# =============================================================================
# Per-chain production stages (production callables ONLY)
# =============================================================================
def _run_step5_chain(ctx: dict, root: Path, chain: str, resume: bool) -> dict:
    """Step5 + Step5C for one chain, through the production callables."""
    import src.step5_preprocess_timeseries as step5
    import src.step5c_tvdi as step5c

    side = hab.CHAIN_SIDE[chain]
    outputs = [
        Path(ctx["step5_output_dir"]) / name for name in (
            "current_period_median_celsius.tif", "baseline_lst_mean_celsius.tif",
            "baseline_lst_std_celsius.tif", "baseline_valid_count.tif",
            "anomaly_zscore.tif", "step5_metadata.json",
        )
    ]
    stage = f"{side}_step5"
    if resume and hab.stage_is_reusable(root, stage):
        log.info("[%s] Step5 reused from a validated checkpoint.", chain)
    else:
        log.info("[%s] Step5 running (namespaced, local-only).", chain)
        step5.run_step5(ctx=ctx)
        hab.write_checkpoint_stage(root, stage, outputs, {"chain": chain})

    derived = (Path(ctx["output_root"]) / hab.DERIVED_SUBDIR
               / "current_minus_baseline_celsius.tif")
    hab.assert_namespace_safe([derived], ctx["experiment_id"])
    hab.build_current_minus_baseline(ctx, derived)

    stage_c = f"{side}_step5c"
    c_outputs = [
        Path(ctx["step5c_output_dir"]) / name for name in (
            "current_tvdi.tif", "tvdi_difference.tif", "step5c_metadata.json",
        )
    ]
    if resume and hab.stage_is_reusable(root, stage_c):
        log.info("[%s] Step5C reused from a validated checkpoint.", chain)
    else:
        log.info("[%s] Step5C running.", chain)
        step5c.run_step5c(ctx=ctx)
        hab.write_checkpoint_stage(root, stage_c, c_outputs, {"chain": chain})

    return {"step5_outputs": [str(p) for p in outputs], "derived": str(derived)}


def _run_step7_chain(ctx: dict, root: Path, chain: str, force: bool, resume: bool,
                     modis_compatibility: dict | None = None) -> dict:
    """Step7A-E for one chain, through the production callables.

    Step7B receives the SAME hash-verified MODIS attestation for both chains, or
    None (the strict default) when the historical path is not required. The
    guard default is not changed and MODIS values are never modified.
    """
    import src.step7a_tiling_infrastructure as step7a
    import src.step7b_prepare_downscaling_dataset as step7b
    import src.step7c_train_downscaling_model as step7c
    import src.step7d_predict_downscaled_lst as step7d
    import src.step7e_fuse_landsat_downscaled_lst as step7e

    side = hab.CHAIN_SIDE[chain]
    step7b_attestation = hab.step7b_compatibility_attestation(modis_compatibility)
    stages = (
        ("step7a", step7a.run_step7a,
         [Path(ctx["step7a_output_dir"]) / "tiling_test_summary.json"], {}),
        ("step7b", step7b.run_step7b,
         [Path(ctx["step7b_output_dir"]) / "downscaling_dataset_stats.json"],
         {"legacy_modis_compatibility": step7b_attestation}),
        ("step7c", step7c.run_step7c,
         [Path(ctx["step7c_output_dir"]) / "downscaling_model_metrics.json"], {}),
        ("step7d", step7d.run_step7d,
         [Path(ctx["step7d_output_dir"]) / "downscaled_lst_celsius.tif"], {}),
        ("step7e", step7e.run_step7e,
         [Path(ctx["step7e_output_dir"]) / "fused_lst_celsius.tif"], {}),
    )
    results: dict = {}
    for name, runner, outputs, extra_kwargs in stages:
        stage = f"{side}_{name}"
        if resume and hab.stage_is_reusable(root, stage, modis_compatibility):
            log.info("[%s] %s reused from a validated checkpoint.", chain, name.upper())
            continue
        if name == "step7b" and step7b_attestation is not None:
            log.warning(
                "[%s] Step7B running under %s (attestation %s). MODIS is read "
                "verbatim: no value, mask, dtype or grid is changed.",
                chain, step7b_attestation.mode, step7b_attestation.attestation_id,
            )
        else:
            log.info("[%s] %s running.", chain, name.upper())
        results[name] = runner(ctx=ctx, force=force, **extra_kwargs)
        hab.write_checkpoint_stage(root, stage, outputs, {"chain": chain},
                                   attestation=modis_compatibility)
    return results


def _run_step8a_chain(ctx: dict, root: Path, chain: str, force: bool, resume: bool,
                      modis_compatibility: dict | None = None):
    """Step8A dataset construction for one chain; returns the dataset frame."""
    import pandas as pd

    import src.step8a_prepare_500m_modeling_dataset as step8a

    side = hab.CHAIN_SIDE[chain]
    parquet = Path(ctx["step8a_output_dir"]) / "step8a_500m_modeling_dataset.parquet"
    csv = parquet.with_suffix(".csv")
    stage = f"{side}_step8a"

    if resume and hab.stage_is_reusable(root, stage, modis_compatibility):
        log.info("[%s] Step8A reused from a validated checkpoint.", chain)
    else:
        log.info("[%s] Step8A running.", chain)
        step8a.run_step8a(ctx=ctx, force=force)
        hab.write_checkpoint_stage(root, stage, [parquet, csv], {"chain": chain},
                                   attestation=modis_compatibility)

    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise HarmonizationDownstreamABRunnerError(
        f"Step8A produced no modelling dataset for {chain}.")


def _load_previous_ab_reference_step8(experiment_id: str):
    """The frozen PREVIOUS-A/B candidate Step8A dataset (read-only).

    This -- not the canonical scene-weighted Step8A -- is the reproduction
    target for this experiment's reference chain.
    """
    import pandas as pd

    parquet = hab.previous_ab_step8_dataset_path(experiment_id)
    csv = parquet.with_suffix(".csv")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise HarmonizationDownstreamABRunnerError(
        f"frozen previous-A/B candidate Step8A dataset not found at {parquet}."
    )


def _reference_reproduction(experiment_id: str, reference_ctx: dict,
                            reference_dataset, root: Path,
                            modis_compatibility: dict | None = None) -> dict:
    """Compare the isolated reference chain to the PREVIOUS A/B candidate chain."""
    from collections import OrderedDict

    checks: "OrderedDict[str, dict]" = OrderedDict()
    for product, tolerance in hab.REPRODUCTION_TOLERANCES.items():
        target = hab.previous_ab_product_path(experiment_id, product)
        if target is None or not Path(target).exists():
            continue
        if product == "baseline_valid_count":
            produced = Path(reference_ctx["step5_output_dir"]) / "baseline_valid_count.tif"
        else:
            produced = hab.product_path(reference_ctx, product,
                                        Path(reference_ctx["output_root"]))
        checks[product] = hab.compare_raster_semantic(
            produced, Path(target), tolerance=tolerance)

    target_dataset = _load_previous_ab_reference_step8(experiment_id)
    step8_check = hab.compare_reference_step8_to_canonical(
        reference_dataset, target_dataset, None, None)
    report = hab.build_reference_reproduction_report(experiment_id, checks, step8_check)

    path = root / "reference_reproduction.json"
    hab.assert_namespace_safe([path], experiment_id)
    hab.write_json_atomic(path, report)
    hab.write_checkpoint_stage(root, "reference_reproduction", [path],
                               attestation=modis_compatibility)
    return report


def _render_comparison_maps(experiment_id: str, reference_ctx: dict,
                            candidate_ctx: dict, out_dir: Path) -> list[str]:
    """Side-by-side maps on a shared stretch, plus a symmetric difference map."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio

    out_dir = Path(out_dir)
    hab.assert_namespace_safe([out_dir], experiment_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for product, entry in hab.compared_raster_products().items():
        if not entry["map"]:
            continue
        ref_path = hab.product_path(reference_ctx, product,
                                    Path(reference_ctx["output_root"]))
        cand_path = hab.product_path(candidate_ctx, product,
                                     Path(candidate_ctx["output_root"]))
        if not ref_path.exists() or not cand_path.exists():
            continue

        product_dir = out_dir / product
        product_dir.mkdir(parents=True, exist_ok=True)
        written.extend(hab.render_pair_maps_for_product(
            ref_path, cand_path, product_dir, product=product))

        with rasterio.open(ref_path) as src:
            a = src.read(1, masked=True).astype("float32").filled(np.nan)
        with rasterio.open(cand_path) as src:
            b = src.read(1, masked=True).astype("float32").filled(np.nan)
        difference = b - a
        finite = difference[np.isfinite(difference)]
        span = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
        span = span if span > 0 else 1.0

        fig, axis = plt.subplots(figsize=(6, 6))
        image = axis.imshow(difference, vmin=-span, vmax=span, cmap="RdBu_r",
                            interpolation="nearest")
        axis.set_title(f"{product}\ncandidate minus reference "
                       f"(symmetric +/-{span:.3g})")
        axis.axis("off")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        path = product_dir / f"{product}__candidate_minus_reference.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))
    return written


# =============================================================================
# Live run
# =============================================================================
def _run_live(experiment_id: str, candidate: str, force: bool,
              resume: bool) -> dict:
    """Execute the whole experiment locally. No Earth Engine, ever."""
    from collections import OrderedDict

    import numpy as np

    root = hab.diagnostic_output_root(experiment_id)

    # --- stage: validate_inputs -------------------------------------------
    state = hab.load_upstream_state(experiment_id)
    hab.validate_upstream_state(state)

    from core.experiment_context import build_experiment_context

    base_ctx = build_experiment_context(experiment_id)
    input_plan = hab.build_input_plan(base_ctx, experiment_id)
    hab.assert_required_frozen_inputs(input_plan, experiment_id)
    if not hab.only_current_lst_differs(input_plan):
        raise HarmonizationDownstreamABRunnerError(
            "more than the current-period LST differs between the chains: "
            f"{hab.differing_roles(input_plan)}"
        )

    if force:
        removed = hab.clear_diagnostic_namespace(experiment_id)
        log.info("[--force] removed ONLY the dedicated namespace: %s", removed)

    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    hab.write_checkpoint_stage(root, "validate_inputs", [], {
        "harmonization_final_status": state["harmonization"]["final_status"],
        "previous_ab_final_status": state["previous_downstream_ab"]["final_status"],
    })

    config = hab.build_config_snapshot(experiment_id, candidate, base_ctx)
    config_path = root / "config" / "harmonization_downstream_ab_config.json"
    hab.assert_namespace_safe([config_path], experiment_id)
    hab.write_json_atomic(config_path, config)

    # --- BLOCKING GATE: current-support invariance (BEFORE Step5) ---------
    support_invariance = hab.check_current_support_invariance(
        Path(input_plan["current_lst"]["reference_count_source"]),
        Path(input_plan["current_lst"]["candidate_count_source"]),
        experiment_id=experiment_id,
    )
    support_path = root / "current_support_invariance.json"
    hab.assert_namespace_safe([support_path], experiment_id)
    hab.write_json_atomic(support_path, support_invariance)
    if not support_invariance["passes"]:
        return _finish_early(
            experiment_id, root, candidate, config, provenance=None,
            support_invariance=support_invariance, reproduction={"status": None},
            status=hab.STATUS_SUPPORT_INVARIANCE_FAILED,
            reason="the reference and candidate current-period unique-date "
                   "support rasters are not bitwise identical "
                   f"(unequal_pixels={support_invariance['unequal_pixels']}, "
                   f"changed_valid_pixels="
                   f"{support_invariance['changed_valid_pixels']})",
        )
    log.info("Current-support invariance gate passed (unequal=0, changed=0, "
             "mask agreement=1.0).")

    # --- stage: materialize_inputs ----------------------------------------
    provenance_path = root / "input_provenance.json"
    if resume and hab.stage_is_reusable(root, "materialize_inputs"):
        log.info("Input bundles reused from a validated checkpoint.")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    else:
        log.info("Materializing the shared bundle and both current-period inputs.")
        provenance = hab.materialize_inputs(input_plan, experiment_id)
        hab.assert_namespace_safe([provenance_path], experiment_id)
        hab.write_json_atomic(provenance_path, provenance)
        materialized: list[Path] = []
        for record in provenance["inputs"]:
            for entry in record["materialized"].values():
                if entry.get("path"):
                    materialized.append(Path(entry["path"]))
        hab.write_checkpoint_stage(
            root, "materialize_inputs",
            sorted({p for p in materialized if p.is_file()}) + [provenance_path])

    if not hab.candidate_modifies_current_lst_only(provenance):
        raise HarmonizationDownstreamABRunnerError(
            "the candidate bundle differs from the reference outside the "
            "current-period LST; this would not be a current-LST-only "
            "intervention."
        )
    if not hab.baselines_shared_between_chains(provenance):
        raise HarmonizationDownstreamABRunnerError(
            "the annual Landsat baselines are not one shared copy; the baseline "
            "model would not be invariant."
        )

    reference_ctx = hab.build_chain_context(experiment_id, hab.CHAIN_REFERENCE)
    candidate_ctx = hab.build_chain_context(experiment_id, hab.CHAIN_CANDIDATE)
    context_check = hab.contexts_share_all_inputs_except_current_period(
        reference_ctx, candidate_ctx)
    if not (context_check["all_shared"] and context_check["current_period_dir_differs"]):
        raise HarmonizationDownstreamABRunnerError(
            f"chain contexts are not correctly shared/differentiated: {context_check}"
        )

    # --- stages: Step5 / Step5C per chain ----------------------------------
    for ctx, chain in ((reference_ctx, hab.CHAIN_REFERENCE),
                       (candidate_ctx, hab.CHAIN_CANDIDATE)):
        _run_step5_chain(ctx, root, chain, resume)

    # --- stage: MODIS compatibility attestation (BEFORE any Step7B) -------
    modis_compatibility = hab.validate_legacy_modis_compatibility(
        experiment_id, provenance,
        OrderedDict((
            (hab.CHAIN_REFERENCE, reference_ctx),
            (hab.CHAIN_CANDIDATE, candidate_ctx),
        )),
    )
    attestation_path = root / "modis_compatibility_attestation.json"
    hab.assert_namespace_safe([attestation_path], experiment_id)
    hab.write_json_atomic(attestation_path, modis_compatibility)
    hab.write_checkpoint_stage(
        root, "modis_compatibility_attestation", [attestation_path],
        {"mode": modis_compatibility["mode"],
         "required": modis_compatibility["required"],
         "status": modis_compatibility["status"],
         "issuer": hab.MODIS_ATTESTATION_ISSUER},
        attestation=modis_compatibility)
    if modis_compatibility["required"]:
        log.warning("MODIS historical compatibility ACTIVE (%s), issued by %s. %s",
                    modis_compatibility["mode"], hab.MODIS_ATTESTATION_ISSUER,
                    modis_compatibility["warning"]["scientific_effect"])
    else:
        log.info("MODIS compatibility not required; the strict Step7B guard applies.")

    # --- stages: Step7 per chain -------------------------------------------
    for ctx, chain in ((reference_ctx, hab.CHAIN_REFERENCE),
                       (candidate_ctx, hab.CHAIN_CANDIDATE)):
        _run_step7_chain(ctx, root, chain, force, resume, modis_compatibility)

    # --- shared-MODIS invariance (BLOCKING GATE) ---------------------------
    shared_modis = hab.check_shared_modis_invariance(
        provenance, reference_ctx, candidate_ctx, modis_compatibility)
    shared_modis_path = root / "shared_modis_invariance.json"
    hab.assert_namespace_safe([shared_modis_path], experiment_id)
    hab.write_json_atomic(shared_modis_path, shared_modis)
    if shared_modis["status"] != "pass":
        return _finish_early(
            experiment_id, root, candidate, config, provenance,
            support_invariance, {"status": None},
            status=hab.STATUS_BASELINE_INVARIANCE_FAILED,
            reason=f"MODIS was not identical across the two chains: "
                   f"{shared_modis['reasons']}",
            modis_compatibility=modis_compatibility,
            shared_modis_invariance=shared_modis,
            technical_failure=hab.TECHNICAL_FAILURE_SHARED_MODIS)

    # --- stage: Step8A dataset construction --------------------------------
    reference_dataset = _run_step8a_chain(
        reference_ctx, root, hab.CHAIN_REFERENCE, force, resume, modis_compatibility)
    candidate_dataset = _run_step8a_chain(
        candidate_ctx, root, hab.CHAIN_CANDIDATE, force, resume, modis_compatibility)

    # --- stage: reference reproduction (BLOCKING GATE) ---------------------
    reproduction = _reference_reproduction(
        experiment_id, reference_ctx, reference_dataset, root, modis_compatibility)
    if reproduction["status"] != "pass":
        return _finish_early(
            experiment_id, root, candidate, config, provenance,
            support_invariance, reproduction,
            status=hab.STATUS_INVALID_REFERENCE,
            reason="the isolated reference chain did not reproduce the frozen "
                   f"{hab.PREVIOUS_AB_REFERENCE_CHAIN} chain within the "
                   "predeclared tolerances",
            modis_compatibility=modis_compatibility,
            shared_modis_invariance=shared_modis)

    # --- stage: population alignment ---------------------------------------
    cohort = hab.build_common_cohort(reference_dataset, candidate_dataset)
    alignment = hab.build_population_alignment(
        experiment_id, cohort, reference_dataset, candidate_dataset)
    alignment_path = root / "population_alignment.json"
    hab.assert_namespace_safe([alignment_path], experiment_id)
    hab.write_json_atomic(alignment_path, alignment)
    hab.write_checkpoint_stage(root, "population_alignment", [alignment_path],
                               attestation=modis_compatibility)
    if alignment["status"] != "ok":
        return _finish_early(
            experiment_id, root, candidate, config, provenance,
            support_invariance, reproduction,
            status=hab.STATUS_POPULATION_REVIEW,
            reason=f"population alignment requires review: "
                   f"{alignment['review_reasons']}",
            alignment=alignment, modis_compatibility=modis_compatibility,
            shared_modis_invariance=shared_modis)

    # --- stage: fold assignment (ONE manifest, reused by both chains) ------
    assignment, reference_cohort, _folds = hab.build_fold_assignment(cohort["reference"])
    candidate_cohort = hab.build_fold_assignment(cohort["candidate"])[1]
    fold_path = root / "fold_assignment.csv"
    hab.assert_namespace_safe([fold_path], experiment_id)
    _write_dataframe_csv(fold_path, assignment)
    hab.write_checkpoint_stage(root, "fold_assignment", [fold_path],
                               attestation=modis_compatibility)

    if not np.array_equal(reference_cohort["spatial_block_id"].to_numpy(),
                          candidate_cohort["spatial_block_id"].to_numpy()):
        raise HarmonizationDownstreamABRunnerError(
            "spatial block ids differ between chains on the common cohort.")

    # --- stage: Step8 models (one per chain, same cohort/folds) ------------
    log.info("Training the reference chain on the common cohort.")
    reference_result = hab.run_chain_model(reference_cohort)
    hab.write_checkpoint_stage(root, "reference_step8_model", [fold_path],
                               attestation=modis_compatibility)
    log.info("Training the candidate chain on the common cohort.")
    candidate_result = hab.run_chain_model(candidate_cohort)
    hab.write_checkpoint_stage(root, "candidate_step8_model", [fold_path],
                               attestation=modis_compatibility)

    fold_check = hab.assert_identical_fold_assignment(
        reference_result["fold_id"], candidate_result["fold_id"],
        assignment["cv_fold"].to_numpy())

    # --- baseline invariance (BLOCKING GATE) -------------------------------
    baseline_invariance = hab.check_baseline_invariance(
        reference_cohort, candidate_cohort, reference_result, candidate_result)
    if baseline_invariance["status"] != "pass":
        return _finish_early(
            experiment_id, root, candidate, config, provenance,
            support_invariance, reproduction,
            status=hab.STATUS_BASELINE_INVARIANCE_FAILED,
            reason="the baseline chain differed despite the intended "
                   "current-LST-only intervention",
            alignment=alignment, baseline_invariance=baseline_invariance,
            modis_compatibility=modis_compatibility,
            shared_modis_invariance=shared_modis)

    # --- stage: paired bootstrap -------------------------------------------
    y = reference_cohort["burned"].astype(int).to_numpy()
    bootstrap = hab.paired_block_bootstrap(
        reference_cohort, y,
        reference_result["oof_prob_baseline"],
        reference_result["oof_prob_thermal"],
        candidate_result["oof_prob_thermal"])
    tables = root / "comparison" / "tables"
    step8_rows = hab.build_step8_metric_rows(
        reference_result, candidate_result, bootstrap["intervals"])
    paired_rows = hab.build_paired_bootstrap_rows(
        reference_result, candidate_result, bootstrap)
    step8_metrics_path = _write_csv(tables / "step8_metrics.csv", step8_rows,
                                    step8_rows[0].keys())
    paired_path = _write_csv(tables / "step8_paired_bootstrap.csv", paired_rows,
                             paired_rows[0].keys())
    oof = hab.build_oof_predictions(reference_cohort, assignment,
                                    reference_result, candidate_result)
    oof_path = _write_dataframe_csv(root / "comparison" / "oof_predictions.csv", oof)
    replicates_path = _write_dataframe_csv(
        tables / "step8_paired_bootstrap_replicates.csv", bootstrap["replicates"])
    hab.write_checkpoint_stage(root, "paired_bootstrap", [
        step8_metrics_path, paired_path, oof_path, replicates_path,
    ], attestation=modis_compatibility)

    # --- stage: raster comparisons -----------------------------------------
    change_rows = []
    for product, threshold in hab.CHANGED_PIXEL_THRESHOLDS.items():
        ref_path = hab.product_path(reference_ctx, product,
                                    Path(reference_ctx["output_root"]))
        cand_path = hab.product_path(candidate_ctx, product,
                                     Path(candidate_ctx["output_root"]))
        if not ref_path.exists() or not cand_path.exists():
            continue
        change_rows.append(hab.compare_raster_change(
            ref_path, cand_path, product=product, changed_threshold=threshold))
    change_path = _write_csv(tables / "raster_change_summary.csv", change_rows,
                             hab.RASTER_CHANGE_COLUMNS)
    map_paths = _render_comparison_maps(
        experiment_id, reference_ctx, candidate_ctx, root / "comparison" / "maps")
    hab.write_checkpoint_stage(root, "raster_comparison", [change_path],
                               attestation=modis_compatibility)

    # --- stage: boundary propagation ---------------------------------------
    boundary = hab.run_boundary_propagation(
        experiment_id, reference_ctx, candidate_ctx, tmp_dir=root / "_analysis_tmp")
    boundary_path = _write_csv(tables / "boundary_propagation.csv", boundary["rows"],
                               hab.BOUNDARY_PROPAGATION_COLUMNS)
    boundary_summary = hab.summarize_boundary_propagation(boundary["verdicts"])
    hab.write_checkpoint_stage(root, "boundary_propagation", [boundary_path],
                               attestation=modis_compatibility)

    # --- decision -----------------------------------------------------------
    intervals = bootstrap["intervals"]
    evidence = {
        "reference_reproduction_status": reproduction["status"],
        "current_support_invariance_status":
            "pass" if support_invariance["passes"] else "fail",
        "shared_modis_invariance_status": shared_modis["status"],
        "shared_modis_invariance_reasons": shared_modis["reasons"],
        "modis_compatibility_required": modis_compatibility["required"],
        "modis_compatibility_attestation_status": modis_compatibility["status"],
        "baseline_invariance_status": baseline_invariance["status"],
        "population_alignment_status": alignment["status"],
        "population_review_reasons": alignment["review_reasons"],
        "key_step5_seam_reduction_supported":
            boundary_summary["key_step5_seam_reduction_supported"],
        "downstream_supported_reduction_products":
            boundary_summary["downstream_supported_reduction_products"],
        "downstream_supported_increase_products":
            boundary_summary["downstream_supported_increase_products"],
        "reference_thermal_support": {
            "roc_auc_interval_above_zero":
                intervals["reference_delta_roc_auc"]["interval_wholly_above_zero"],
            "pr_auc_interval_above_zero":
                intervals["reference_delta_pr_auc"]["interval_wholly_above_zero"],
        },
        "candidate_thermal_support": {
            "roc_auc_interval_above_zero":
                intervals["candidate_delta_roc_auc"]["interval_wholly_above_zero"],
            "pr_auc_interval_above_zero":
                intervals["candidate_delta_pr_auc"]["interval_wholly_above_zero"],
        },
        "paired_intervals": {
            "roc_auc": intervals["paired_delta_roc_auc"],
            "pr_auc": intervals["paired_delta_pr_auc"],
            "brier": intervals["paired_delta_brier"],
        },
    }
    decision = hab.decide_final_status(evidence)

    # --- stage: report generation ------------------------------------------
    metrics_before = {"step8": step8_rows, "paired": paired_rows,
                      "rasters": change_rows, "boundary": boundary["rows"]}
    fold_manifest = {
        "seed": int(assignment["seed"].iloc[0]),
        "n_splits": int(assignment["n_splits"].iloc[0]),
        "block_size_cells": int(assignment["block_size_cells"].iloc[0]),
        "grouping": "spatial_block_id",
        "rows": int(len(assignment)),
        **fold_check,
    }
    summary = hab.build_summary(
        experiment_id, candidate=candidate, config=config, provenance=provenance,
        support_invariance=support_invariance, reproduction=reproduction,
        alignment=alignment, fold_manifest=fold_manifest,
        baseline_invariance=baseline_invariance, raster_change_rows=change_rows,
        boundary_summary=boundary_summary, boundary_result=boundary,
        step8_metric_rows=step8_rows, paired_rows=paired_rows, bootstrap=bootstrap,
        decision=decision,
        source_boundary_evidence=hab.frozen_source_boundary_evidence(experiment_id),
        modis_compatibility=modis_compatibility,
        shared_modis_invariance=shared_modis)

    if not hab.summary_forbids_banned_conclusions(summary):
        raise HarmonizationDownstreamABRunnerError(
            "the summary claims a forbidden conclusion; refusing to write.")

    summary_path = root / "harmonization_downstream_ab_summary.json"
    markdown_path = root / "harmonization_downstream_ab_summary.md"
    hab.assert_namespace_safe([summary_path, markdown_path], experiment_id)
    hab.write_json_atomic(summary_path, summary)
    markdown_path.write_text(hab.render_summary_markdown(summary), encoding="utf-8")

    metrics_after = {"step8": step8_rows, "paired": paired_rows,
                     "rasters": change_rows, "boundary": boundary["rows"]}
    if not hab.report_generation_preserves_metrics(metrics_before, metrics_after):
        raise HarmonizationDownstreamABRunnerError(
            "report generation mutated a scientific metric payload; refusing to "
            "finish.")

    manifest = hab.build_manifest(experiment_id, root, summary)
    manifest_path = root / "harmonization_downstream_ab_manifest.json"
    hab.write_json_atomic(manifest_path, manifest)
    hab.write_checkpoint_stage(root, "report_generation", [
        summary_path, markdown_path, manifest_path,
    ], attestation=modis_compatibility)

    tmp_dir = root / "_analysis_tmp"
    if tmp_dir.exists():
        import shutil

        hab.assert_namespace_safe([tmp_dir], experiment_id)
        shutil.rmtree(tmp_dir)

    log.info("FINAL STATUS: %s", decision["final_status"])
    log.info("%s", decision["meaning"])
    return {
        "experiment_id": experiment_id, "ran": True, "dry_run": False,
        "candidate": candidate, "final_status": decision["final_status"],
        "production_approved": False,
        "output_root": str(root), "summary_path": str(summary_path),
        "markdown_path": str(markdown_path), "manifest_path": str(manifest_path),
        "map_count": len(map_paths),
    }


def _finish_early(experiment_id: str, root: Path, candidate: str, config: dict,
                  provenance: dict | None, support_invariance: dict,
                  reproduction: dict, *, status: str, reason: str,
                  alignment: dict | None = None,
                  baseline_invariance: dict | None = None,
                  modis_compatibility: dict | None = None,
                  shared_modis_invariance: dict | None = None,
                  technical_failure: str | None = None) -> dict:
    """Write the terminating report for a blocking gate failure.

    No candidate scientific conclusion is issued; the summary records exactly
    which gate stopped the experiment.
    """
    from collections import OrderedDict

    provenance = provenance or {}
    grid_gate = provenance.get("raw_current_lst_grid_equality_gate") or {}
    summary = OrderedDict((
        ("experiment", hab.DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("reference_chain", hab.CHAIN_REFERENCE),
        ("candidate_chain", candidate),
        ("report_schema_version", hab.REPORT_SCHEMA_VERSION),
        ("decision_rule_version", hab.DECISION_RULE_VERSION),
        ("final_status", status),
        ("final_status_meaning", hab.FINAL_STATUS_MEANINGS[status]),
        ("seam_fixed", False),
        ("production_approved", False),
        ("production_ready", False),
        ("changes_production_reducer", False),
        ("claims_non_inferiority", False),
        ("claims_transfer_improvement", False),
        ("claims_cross_region_generalization", False),
        ("claims_causality", False),
        ("technical_failure", technical_failure),
        ("warnings", hab.summary_warnings(modis_compatibility)),
        ("modis_compatibility", hab.build_modis_compatibility_report(
            modis_compatibility, shared_modis_invariance)),
        ("modis_attestation_issuer", hab.MODIS_ATTESTATION_ISSUER),
        ("terminating_reason", reason),
        ("candidate_scientific_conclusion_issued", False),
        ("configuration", config),
        ("technical_validity", OrderedDict((
            ("reference_reproduction_status", (reproduction or {}).get("status")),
            ("current_support_invariance_status",
             "pass" if support_invariance.get("passes") else "fail"),
            ("current_support_unequal_pixels",
             support_invariance.get("unequal_pixels")),
            ("current_support_changed_valid_pixels",
             support_invariance.get("changed_valid_pixels")),
            ("current_support_mask_agreement",
             support_invariance.get("mask_agreement")),
            ("baseline_invariance_status", (baseline_invariance or {}).get("status")),
            ("shared_modis_invariance_status",
             (shared_modis_invariance or {}).get("status")),
            ("modis_compatibility_mode",
             (modis_compatibility or {}).get("mode", hab.MODIS_STRICT_MODE)),
            ("modis_compatibility_attestation_status",
             (modis_compatibility or {}).get("status")),
            ("population_alignment_status", (alignment or {}).get("status")),
            ("raw_current_lst_grid_equality_passed", grid_gate.get("passed")),
            ("only_current_lst_differs",
             hab.candidate_modifies_current_lst_only(provenance)
             if provenance else None),
            ("earth_engine_used", False),
        ))),
        ("limitations", hab.required_limitations()),
        ("next_decision", hab.next_decision_text(status)),
    ))
    summary_path = root / "harmonization_downstream_ab_summary.json"
    markdown_path = root / "harmonization_downstream_ab_summary.md"
    hab.assert_namespace_safe([summary_path, markdown_path], experiment_id)
    hab.write_json_atomic(summary_path, summary)
    markdown_path.write_text(
        f"# Harmonization downstream A/B -- {experiment_id}\n\n"
        f"**Final status: `{status}`**\n\n"
        f"> {hab.FINAL_STATUS_MEANINGS[status]}\n\n"
        + (f"Technical failure: `{technical_failure}`\n\n" if technical_failure else "")
        + f"Terminating reason: {reason}\n\n"
        "No candidate scientific conclusion is issued.\n\n"
        "## Limitations\n\n"
        + "\n".join(f"- {item}" for item in hab.required_limitations())
        + f"\n\n## Next decision\n\n{hab.next_decision_text(status)}\n",
        encoding="utf-8",
    )
    manifest = hab.build_manifest(experiment_id, root, summary)
    manifest_path = root / "harmonization_downstream_ab_manifest.json"
    hab.write_json_atomic(manifest_path, manifest)
    hab.write_checkpoint_stage(root, "report_generation", [
        summary_path, markdown_path, manifest_path,
    ], attestation=modis_compatibility)
    log.error("FINAL STATUS: %s -- %s", status, reason)
    return {
        "experiment_id": experiment_id, "ran": True, "dry_run": False,
        "candidate": candidate, "final_status": status,
        "production_approved": False, "output_root": str(root),
        "summary_path": str(summary_path),
    }


# =============================================================================
# Entry point
# =============================================================================
def main(experiment_id: str | None = None,
         candidate: str = hab.CHAIN_CANDIDATE,
         dry_run: bool = False, run: bool = False,
         resume: bool = False, force: bool = False,
         report_only: bool = False) -> dict:
    validate_modes(dry_run, run, resume, force, report_only)
    if not experiment_id:
        raise HarmonizationDownstreamABRunnerError("--experiment is required.")
    hab.assert_supported_experiment(experiment_id)
    hab.assert_supported_candidate(candidate)

    if report_only:
        # Re-render reports from the frozen summary JSON. No model, no raster,
        # no Step5-Step8, no Earth Engine.
        with ab.EarthEngineGuard():
            result = hab.rebuild_reports_from_summary(experiment_id)
        log.info("[report-only] final status (unchanged): %s", result["final_status"])
        log.info("[report-only] scientific sections unchanged: %s",
                 result["scientific_sections_unchanged"])
        log.info("[report-only] markdown: %s", result["markdown_path"])
        log.info("[report-only] manifest: %s", result["manifest_path"])
        relabel = result["map_relabel"]
        log.info("[report-only] map PNGs relabelled: %s (skipped: %s)",
                 len(relabel["renamed"]), len(relabel["skipped"]))
        for entry in relabel["renamed"]:
            log.info("[report-only]   %s -> %s",
                     Path(entry["from"]).name, Path(entry["to"]).name)
        log.info("[report-only] models trained: %s | rasters written: %s | "
                 "pipeline steps run: %s | Earth Engine calls: %s",
                 result["models_trained"], result["rasters_written"],
                 result["pipeline_steps_run"], result["earth_engine_calls"])
        return result

    # The guard is installed for the dry-run too: it must be IMPOSSIBLE for any
    # code path of this runner to reach Earth Engine.
    if dry_run:
        with ab.EarthEngineGuard():
            plan = hab.build_dry_run_plan(experiment_id, candidate)
        _print_dry_run(plan)
        return plan

    with ab.EarthEngineGuard():
        return _run_live(experiment_id, candidate, force, resume)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Isolated downstream A/B for the current-period date-offset "
            "harmonization candidate. Never changes production code, never "
            "issues a production decision, never runs Earth Engine."
        )
    )
    parser.add_argument("--experiment", required=True,
                        help="experiment id (only manavgat_2021 is supported)")
    parser.add_argument("--candidate", default=hab.CHAIN_CANDIDATE,
                        choices=list(hab.SUPPORTED_CANDIDATES),
                        help="candidate chain (only "
                             f"{hab.CHAIN_CANDIDATE} is supported)")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and print the full plan; write nothing")
    parser.add_argument("--run", action="store_true", help="execute the experiment")
    parser.add_argument("--resume", action="store_true",
                        help="reuse validated completed stages (requires --run)")
    parser.add_argument("--force", action="store_true",
                        help="delete ONLY this experiment's namespace and rebuild "
                             "it (requires --run)")
    parser.add_argument("--report-only", action="store_true",
                        help="re-render the Markdown and manifest from the "
                             "existing summary JSON; runs no model, no raster "
                             "and no pipeline step")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(experiment_id=args.experiment, candidate=args.candidate,
         dry_run=args.dry_run, run=args.run, resume=args.resume, force=args.force,
         report_only=args.report_only)
