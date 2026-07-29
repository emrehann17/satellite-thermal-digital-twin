#!/usr/bin/env python3
"""
scripts/run_landsat_composite_downstream_ab.py

Dedicated entry point for the DOWNSTREAM A/B experiment comparing

    A. scene_weighted_reference   (current canonical Landsat compositing)
    B. date_balanced_lst_only     (date-balanced daily compositing, LST only)

    python scripts/run_landsat_composite_downstream_ab.py \
        --experiment manavgat_2021 \
        --candidate date_balanced_lst_only \
        --dry-run

CONTRACT
--------
    - Exactly one of --dry-run / --run. Default execution is never implied.
    - --resume and --force are mutually exclusive, and both require --run.
    - --dry-run writes nothing, creates no directory, runs no model and makes
      no Earth Engine call.
    - No Earth Engine code path is reachable from this runner. The live run
      additionally installs a runtime guard that makes every EE entry point
      raise, so the contract is enforced and not merely asserted.
    - The runner fails clearly for any experiment lacking the required frozen
      inputs; it NEVER silently falls back to a GEE export.
    - Everything is written under
      outputs/diagnostics/landsat_composite_downstream_ab/<experiment_id>/.
      Frozen canonical experiment outputs and the frozen counterfactual audit
      are read-only inputs and are never modified.
    - No scientific computation is re-implemented: Step5/Step5C/Step7A-E/Step8A
      run through their own production callables with a namespaced
      ExperimentContext, and the boundary/bootstrap/model helpers are reused.
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
from core.io_utils import setup_logger

log, log_file = setup_logger("landsat_composite_downstream_ab")


class DownstreamABRunnerError(SystemExit):
    """Fail-fast CLI error (same convention as the other runners)."""


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
def validate_modes(dry_run: bool, run: bool, resume: bool, force: bool) -> None:
    """Exactly one mode; --resume/--force are run-only and mutually exclusive."""
    if dry_run and run:
        raise DownstreamABRunnerError(
            "--dry-run and --run are mutually exclusive; pass exactly one."
        )
    if not dry_run and not run:
        raise DownstreamABRunnerError(
            "one of --dry-run or --run is required. Default execution is never "
            "implied by this runner."
        )
    if resume and force:
        raise DownstreamABRunnerError("--resume and --force are mutually exclusive.")
    if resume and not run:
        raise DownstreamABRunnerError("--resume requires --run.")
    if force and not run:
        raise DownstreamABRunnerError("--force requires --run.")


# =============================================================================
# Dry-run
# =============================================================================
def _print_dry_run(plan: dict) -> None:
    log.info("[dry-run] experiment: %s", plan["experiment_id"])
    log.info("[dry-run] reference chain: %s", plan["reference_chain"])
    log.info("[dry-run] candidate chain: %s", plan["candidate_chain"])

    log.info("[dry-run] --- reference source paths ---")
    for role, path in plan["reference_source_paths"].items():
        log.info("[dry-run]   %-28s %s", role, path)
    log.info("[dry-run] --- candidate source paths ---")
    for role, path in plan["candidate_source_paths"].items():
        log.info("[dry-run]   %-28s %s", role, path)

    prereq = plan["audit_prerequisite_status"]
    log.info("[dry-run] --- audit prerequisite status ---")
    log.info("[dry-run]   source root: %s", prereq["source_root"])
    log.info("[dry-run]   final_status: %s (required: %s)",
             prereq["final_status"], prereq["required_final_status"])
    log.info("[dry-run]   canonical reproduction: %s (required: %s)",
             prereq["canonical_reproduction_status"],
             prereq["required_canonical_reproduction"])
    log.info("[dry-run]   report schema version: %s", prereq["report_schema_version"])
    for name, signature in (prereq["audit_file_hashes"] or {}).items():
        log.info("[dry-run]   hash %-32s %s (%s bytes)",
                 name, signature.get("sha256"), signature.get("bytes"))
    log.info("[dry-run]   prerequisites_met: %s", prereq["prerequisites_met"])
    if plan["missing_sources"]:
        log.warning("[dry-run]   MISSING sources: %s", plan["missing_sources"])

    modis = plan["modis_compatibility"]
    log.info("[dry-run] --- MODIS historical-compatibility check ---")
    log.info("[dry-run]   default mode: %s", modis["default_mode"])
    log.info("[dry-run]   historical compatibility required: %s",
             modis["historical_compatibility_required"])
    log.info("[dry-run]   reason: %s", modis["reason"])
    log.info("[dry-run]   mode that would be used: %s",
             modis["mode_name"] if modis["historical_compatibility_required"]
             else modis["default_mode"])
    log.info("[dry-run]   experiment authorized for compatibility: %s (allowed: %s)",
             modis["experiment_is_authorized"], modis["authorized_experiment_ids"])
    log.info("[dry-run]   zero-fill guard threshold: %s",
             modis["zero_fill_guard_threshold"])
    log.info("[dry-run]   frozen MODIS namespace: %s", modis["frozen_modis_namespace"])
    for name, record in modis["frozen_modis_evidence"].items():
        log.info("[dry-run]   %-24s nodata=%s zero_frac=%.4f sha256=%s",
                 name, record.get("nodata"),
                 float(record.get("exact_zero_fraction") or 0.0),
                 record.get("sha256"))
    log.info("[dry-run]   attestation declaration: %s (present: %s)",
             modis["attestation_declaration_path"],
             modis["attestation_declaration_present"])
    log.info("[dry-run]   frozen Step7B historical evidence confirmed: %s",
             modis["historical_step7b_evidence_confirmed"])
    log.info("[dry-run]   default Step7B guard changed: %s",
             modis["default_step7b_guard_changed"])
    if modis["warning"]:
        log.warning("[dry-run]   WARNING %s: %s",
                    modis["warning"]["code"], modis["warning"]["scientific_effect"])

    log.info("[dry-run] output namespace: %s", plan["output_namespace"])
    log.info("[dry-run] --- planned Step5/Step7/Step8 stages ---")
    for stage in plan["planned_stages"]:
        log.info("[dry-run]   stage %s", stage)
    log.info("[dry-run] --- expected files ---")
    for name, path in plan["expected_files"].items():
        log.info("[dry-run]   %-30s %s", name, path)

    config = plan["configuration"]
    log.info("[dry-run] --- model configuration ---")
    for key, value in config["model"].items():
        log.info("[dry-run]   %-28s %s", key, value)
    log.info("[dry-run] --- spatial-block configuration ---")
    for key, value in config["spatial_blocks"].items():
        log.info("[dry-run]   %-28s %s", key, value)
    log.info("[dry-run] --- bootstrap configuration ---")
    for key, value in config["bootstrap"].items():
        log.info("[dry-run]   %-28s %s", key, value)
    log.info("[dry-run] decision-rule version: %s", plan["decision_rule_version"])
    log.info("[dry-run] allowed final statuses: %s", plan["allowed_final_statuses"])
    log.info("[dry-run] NO writes, NO directories created, NO Earth Engine call, "
             "NO model run.")


# =============================================================================
# Live stages
# =============================================================================
def _run_step5_chain(ctx: dict, root: Path, chain: str, resume: bool) -> dict:
    """Step5 + Step5C for one chain, through the production callables."""
    import src.step5_preprocess_timeseries as step5
    import src.step5c_tvdi as step5c

    side = ab.CHAIN_SIDE[chain]
    outputs = [
        Path(ctx["step5_output_dir"]) / name for name in (
            "current_period_median_celsius.tif", "baseline_lst_mean_celsius.tif",
            "baseline_lst_std_celsius.tif", "baseline_valid_count.tif",
            "anomaly_zscore.tif", "step5_metadata.json",
        )
    ]
    stage = f"{side}_step5"
    if resume and ab.stage_is_reusable(root, stage):
        log.info("[%s] Step5 reused from a validated checkpoint.", chain)
    else:
        log.info("[%s] Step5 running (namespaced, local-only).", chain)
        step5.run_step5(ctx=ctx)
        ab.write_checkpoint_stage(root, stage, outputs, {"chain": chain})

    derived = Path(ctx["output_root"]) / ab.DERIVED_SUBDIR / "current_minus_baseline_celsius.tif"
    ab.assert_downstream_namespace_safe([derived], ctx["experiment_id"])
    ab.build_current_minus_baseline(ctx, derived)

    stage_c = f"{side}_step5c"
    c_outputs = [
        Path(ctx["step5c_output_dir"]) / name for name in (
            "current_tvdi.tif", "tvdi_difference.tif", "step5c_metadata.json",
        )
    ]
    if resume and ab.stage_is_reusable(root, stage_c):
        log.info("[%s] Step5C reused from a validated checkpoint.", chain)
    else:
        log.info("[%s] Step5C running.", chain)
        step5c.run_step5c(ctx=ctx)
        ab.write_checkpoint_stage(root, stage_c, c_outputs, {"chain": chain})

    return {"step5_outputs": [str(p) for p in outputs], "derived": str(derived)}


def _run_step7_chain(
    ctx: dict, root: Path, chain: str, force: bool, resume: bool,
    modis_compatibility: dict | None = None,
) -> dict:
    """Step7A-E for one chain, through the production callables.

    Step7B receives the SAME hash-verified MODIS compatibility attestation for
    both chains, or None (the strict default) when the historical path is not
    required. Nothing else about Step7 changes.
    """
    import src.step7a_tiling_infrastructure as step7a
    import src.step7b_prepare_downscaling_dataset as step7b
    import src.step7c_train_downscaling_model as step7c
    import src.step7d_predict_downscaled_lst as step7d
    import src.step7e_fuse_landsat_downscaled_lst as step7e

    side = ab.CHAIN_SIDE[chain]
    step7b_attestation = ab.step7b_compatibility_attestation(modis_compatibility)
    stages = (
        ("step7a", step7a.run_step7a, [Path(ctx["step7a_output_dir"]) / "tiling_test_summary.json"], {}),
        ("step7b", step7b.run_step7b, [Path(ctx["step7b_output_dir"]) / "downscaling_dataset_stats.json"],
         {"legacy_modis_compatibility": step7b_attestation}),
        ("step7c", step7c.run_step7c, [Path(ctx["step7c_output_dir"]) / "downscaling_model_metrics.json"], {}),
        ("step7d", step7d.run_step7d, [Path(ctx["step7d_output_dir"]) / "downscaled_lst_celsius.tif"], {}),
        ("step7e", step7e.run_step7e, [Path(ctx["step7e_output_dir"]) / "fused_lst_celsius.tif"], {}),
    )
    results: dict = {}
    for name, runner, outputs, extra_kwargs in stages:
        stage = f"{side}_{name}"
        if resume and ab.stage_is_reusable(root, stage, modis_compatibility):
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
        ab.write_checkpoint_stage(
            root, stage, outputs, {"chain": chain}, attestation=modis_compatibility,
        )
    return results


def _run_step8a_chain(
    ctx: dict, root: Path, chain: str, force: bool, resume: bool,
    modis_compatibility: dict | None = None,
):
    """Step8A dataset construction for one chain; returns the dataset frame."""
    import pandas as pd

    import src.step8a_prepare_500m_modeling_dataset as step8a

    side = ab.CHAIN_SIDE[chain]
    parquet = Path(ctx["step8a_output_dir"]) / "step8a_500m_modeling_dataset.parquet"
    csv = parquet.with_suffix(".csv")
    stage = f"{side}_step8a"

    if resume and ab.stage_is_reusable(root, stage, modis_compatibility):
        log.info("[%s] Step8A reused from a validated checkpoint.", chain)
    else:
        log.info("[%s] Step8A running.", chain)
        step8a.run_step8a(ctx=ctx, force=force)
        ab.write_checkpoint_stage(
            root, stage, [parquet, csv], {"chain": chain},
            attestation=modis_compatibility,
        )

    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise DownstreamABRunnerError(f"Step8A produced no modelling dataset for {chain}.")


def _load_canonical_step8(experiment_id: str):
    """Frozen canonical Step8A dataset and Step8B metrics (read-only)."""
    import pandas as pd

    root = ab.canonical_experiment_root(experiment_id)
    parquet = root / "step8a" / "step8a_500m_modeling_dataset.parquet"
    csv = parquet.with_suffix(".csv")
    if parquet.exists():
        dataset = pd.read_parquet(parquet)
    elif csv.exists():
        dataset = pd.read_csv(csv)
    else:
        raise DownstreamABRunnerError(
            f"frozen canonical Step8A dataset not found under {root / 'step8a'}."
        )
    metrics_path = root / "step8b" / "step8b_model_comparison_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    return dataset, metrics


def _reference_reproduction(
    experiment_id: str, reference_ctx: dict, reference_dataset, root: Path,
    modis_compatibility: dict | None = None,
) -> dict:
    """Compare the isolated reference chain to the frozen canonical pipeline."""
    from collections import OrderedDict

    checks: "OrderedDict[str, dict]" = OrderedDict()
    for product, tolerance in ab.REPRODUCTION_TOLERANCES.items():
        canonical = ab.canonical_product_path(experiment_id, product)
        if canonical is None:
            continue
        if product == "baseline_valid_count":
            produced = Path(reference_ctx["step5_output_dir"]) / "baseline_valid_count.tif"
        else:
            produced = ab.product_path(
                reference_ctx, product, Path(reference_ctx["output_root"]),
            )
        checks[product] = ab.compare_raster_semantic(
            produced, canonical, tolerance=tolerance,
        )

    canonical_dataset, canonical_metrics = _load_canonical_step8(experiment_id)
    step8_check = ab.compare_reference_step8_to_canonical(
        reference_dataset, canonical_dataset, None, canonical_metrics,
    )
    report = ab.build_reference_reproduction_report(experiment_id, checks, step8_check)
    path = root / "reference_reproduction.json"
    ab.assert_downstream_namespace_safe([path], experiment_id)
    ab.write_json_atomic(path, report)
    ab.write_checkpoint_stage(
        root, "reference_reproduction", [path], attestation=modis_compatibility,
    )
    return report


def _render_comparison_maps(
    experiment_id: str, reference_ctx: dict, candidate_ctx: dict, out_dir: Path,
) -> list[str]:
    """Compact side-by-side maps with a shared display range, plus the difference.

    Reference and candidate use ONE common stretch (via the audit's
    `render_pair_maps`); the difference map uses a symmetric diverging range.
    Nothing is smoothed -- every panel uses nearest-neighbour rendering.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio

    out_dir = Path(out_dir)
    ab.assert_downstream_namespace_safe([out_dir], experiment_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for product, entry in ab.compared_raster_products().items():
        if not entry["map"]:
            continue
        ref_path = ab.product_path(reference_ctx, product, Path(reference_ctx["output_root"]))
        cand_path = ab.product_path(candidate_ctx, product, Path(candidate_ctx["output_root"]))
        if not ref_path.exists() or not cand_path.exists():
            continue

        product_dir = out_dir / product
        product_dir.mkdir(parents=True, exist_ok=True)
        written.extend(ab.render_pair_maps_for_product(
            ref_path, cand_path, product_dir, product=product,
        ))

        with rasterio.open(ref_path) as src:
            a = src.read(1, masked=True).astype("float32").filled(np.nan)
        with rasterio.open(cand_path) as src:
            b = src.read(1, masked=True).astype("float32").filled(np.nan)
        difference = b - a
        finite = difference[np.isfinite(difference)]
        span = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
        span = span if span > 0 else 1.0

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(difference, vmin=-span, vmax=span, cmap="RdBu_r",
                       interpolation="nearest")
        ax.set_title(f"{product}\ncandidate minus reference (symmetric +/-{span:.3g})")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        path = product_dir / f"{product}__candidate_minus_reference.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))
    return written


def _run_live(
    experiment_id: str, candidate: str, force: bool, resume: bool,
) -> dict:
    """Execute the whole A/B experiment locally. No Earth Engine, ever."""
    from collections import OrderedDict

    import numpy as np

    root = ab.diagnostic_output_root(experiment_id)

    # --- stage: validate_inputs -------------------------------------------
    state = ab.load_source_audit_state(experiment_id)
    ab.validate_source_audit_state(state)

    from core.experiment_context import build_experiment_context

    base_ctx = build_experiment_context(experiment_id)
    input_plan = ab.build_input_plan(base_ctx, experiment_id)
    ab.assert_required_frozen_inputs(input_plan, experiment_id)
    grid_gate = ab.assert_reference_candidate_grid_equality(input_plan)
    log.info("Raw LST grid equality gate passed for %d roles.", len(grid_gate))

    if force:
        removed = ab.clear_diagnostic_namespace(experiment_id)
        log.info("[--force] removed ONLY the dedicated A/B namespace: %s", removed)

    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    ab.write_checkpoint_stage(root, "validate_inputs", [], {
        "source_final_status": state["final_status"],
        "source_canonical_reproduction": state["canonical_reproduction_status"],
    })

    config = ab.build_config_snapshot(experiment_id, candidate, base_ctx)
    config_path = root / "config" / "downstream_ab_config.json"
    ab.assert_downstream_namespace_safe([config_path], experiment_id)
    ab.write_json_atomic(config_path, config)

    # --- stage: materialize_inputs ----------------------------------------
    provenance_path = root / "input_provenance.json"
    if resume and ab.stage_is_reusable(root, "materialize_inputs"):
        log.info("Input bundles reused from a validated checkpoint.")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    else:
        log.info("Materializing isolated reference and candidate input bundles.")
        provenance = ab.materialize_inputs(input_plan, experiment_id)
        ab.assert_downstream_namespace_safe([provenance_path], experiment_id)
        ab.write_json_atomic(provenance_path, provenance)
        materialized = [
            Path(record["materialized_path"]) for record in provenance["inputs"]
        ]
        ab.write_checkpoint_stage(
            root, "materialize_inputs", sorted(set(materialized)) + [provenance_path],
        )

    if not ab.candidate_modifies_lst_only(provenance):
        raise DownstreamABRunnerError(
            "the candidate bundle differs from the reference outside Landsat LST; "
            "this would not be an LST-only intervention."
        )
    if not ab.ndvi_inputs_identical(provenance):
        raise DownstreamABRunnerError(
            "canonical NDVI is not identical between chains; the baseline model "
            "would not be invariant."
        )

    reference_ctx = ab.build_chain_context(experiment_id, ab.CHAIN_REFERENCE)
    candidate_ctx = ab.build_chain_context(experiment_id, ab.CHAIN_CANDIDATE)

    # --- stages: Step5 / Step5C per chain ----------------------------------
    for ctx, chain in ((reference_ctx, ab.CHAIN_REFERENCE), (candidate_ctx, ab.CHAIN_CANDIDATE)):
        _run_step5_chain(ctx, root, chain, resume)

    # --- stage: MODIS compatibility attestation (BEFORE any Step7B) --------
    # Read-only: it hashes and describes frozen files, writes only its own
    # report inside the A/B namespace, and refuses the run outright if any gate
    # fails. No scientific status can be produced from a failed attestation.
    modis_compatibility = ab.validate_legacy_modis_compatibility(
        experiment_id, provenance,
        OrderedDict((
            (ab.CHAIN_REFERENCE, reference_ctx),
            (ab.CHAIN_CANDIDATE, candidate_ctx),
        )),
    )
    attestation_path = root / "modis_compatibility_attestation.json"
    ab.assert_downstream_namespace_safe([attestation_path], experiment_id)
    ab.write_json_atomic(attestation_path, modis_compatibility)
    ab.write_checkpoint_stage(
        root, "modis_compatibility_attestation", [attestation_path],
        {"mode": modis_compatibility["mode"],
         "required": modis_compatibility["required"],
         "status": modis_compatibility["status"]},
        attestation=modis_compatibility,
    )
    if modis_compatibility["required"]:
        log.warning(
            "MODIS historical compatibility ACTIVE (%s). %s",
            modis_compatibility["mode"],
            modis_compatibility["warning"]["scientific_effect"],
        )
    else:
        log.info("MODIS compatibility not required; the strict Step7B guard applies.")

    # --- stages: Step7 per chain -------------------------------------------
    for ctx, chain in ((reference_ctx, ab.CHAIN_REFERENCE), (candidate_ctx, ab.CHAIN_CANDIDATE)):
        _run_step7_chain(ctx, root, chain, force, resume, modis_compatibility)

    # --- shared-MODIS invariance (BLOCKING GATE) ---------------------------
    shared_modis = ab.check_shared_modis_invariance(
        provenance, reference_ctx, candidate_ctx, modis_compatibility,
    )
    shared_modis_path = root / "shared_modis_invariance.json"
    ab.assert_downstream_namespace_safe([shared_modis_path], experiment_id)
    ab.write_json_atomic(shared_modis_path, shared_modis)
    if shared_modis["status"] != "pass":
        return _finish_early(
            experiment_id, root, candidate, config, provenance,
            {"status": None},
            status=ab.STATUS_BASELINE_INVARIANCE_FAILED,
            reason="MODIS was not identical across the two chains: "
                   f"{shared_modis['reasons']}",
            modis_compatibility=modis_compatibility,
            shared_modis_invariance=shared_modis,
            technical_failure=ab.TECHNICAL_FAILURE_SHARED_MODIS,
        )

    # --- stage: dataset construction ---------------------------------------
    reference_dataset = _run_step8a_chain(
        reference_ctx, root, ab.CHAIN_REFERENCE, force, resume, modis_compatibility,
    )
    candidate_dataset = _run_step8a_chain(
        candidate_ctx, root, ab.CHAIN_CANDIDATE, force, resume, modis_compatibility,
    )

    # --- stage: reference reproduction (BLOCKING GATE) ---------------------
    reproduction = _reference_reproduction(
        experiment_id, reference_ctx, reference_dataset, root, modis_compatibility,
    )
    if reproduction["status"] != "pass":
        return _finish_early(
            experiment_id, root, candidate, config, provenance, reproduction,
            status=ab.STATUS_INVALID_REFERENCE,
            reason="the isolated reference chain did not reproduce the frozen "
                   "canonical pipeline within the predeclared tolerances",
            modis_compatibility=modis_compatibility,
            shared_modis_invariance=shared_modis,
        )

    # --- stage: population alignment ---------------------------------------
    cohort = ab.build_common_cohort(reference_dataset, candidate_dataset)
    alignment = ab.build_population_alignment(
        experiment_id, cohort, reference_dataset, candidate_dataset,
    )
    alignment_path = root / "population_alignment.json"
    ab.assert_downstream_namespace_safe([alignment_path], experiment_id)
    ab.write_json_atomic(alignment_path, alignment)
    ab.write_checkpoint_stage(
        root, "population_alignment", [alignment_path],
        attestation=modis_compatibility,
    )

    if alignment["status"] != "ok":
        return _finish_early(
            experiment_id, root, candidate, config, provenance, reproduction,
            status=ab.STATUS_POPULATION_REVIEW,
            reason=f"population alignment requires review: {alignment['review_reasons']}",
            alignment=alignment, modis_compatibility=modis_compatibility,
            shared_modis_invariance=shared_modis,
        )

    # --- stage: fold assignment (ONE manifest, reused by both chains) ------
    assignment, reference_cohort, _folds = ab.build_fold_assignment(cohort["reference"])
    candidate_cohort = ab.build_fold_assignment(cohort["candidate"])[1]
    fold_path = root / "fold_assignment.csv"
    ab.assert_downstream_namespace_safe([fold_path], experiment_id)
    _write_dataframe_csv(fold_path, assignment)
    ab.write_checkpoint_stage(
        root, "fold_assignment", [fold_path], attestation=modis_compatibility,
    )

    if not np.array_equal(
        reference_cohort["spatial_block_id"].to_numpy(),
        candidate_cohort["spatial_block_id"].to_numpy(),
    ):
        raise DownstreamABRunnerError(
            "spatial block ids differ between chains on the common cohort."
        )

    # --- stage: Step8 models (one per chain, same cohort/folds) ------------
    log.info("Training the reference chain on the common cohort.")
    reference_result = ab.run_chain_model(reference_cohort)
    ab.write_checkpoint_stage(
        root, "reference_step8_model", [fold_path], attestation=modis_compatibility,
    )
    log.info("Training the candidate chain on the common cohort.")
    candidate_result = ab.run_chain_model(candidate_cohort)
    ab.write_checkpoint_stage(
        root, "candidate_step8_model", [fold_path], attestation=modis_compatibility,
    )

    fold_check = ab.assert_identical_fold_assignment(
        reference_result["fold_id"], candidate_result["fold_id"],
        assignment["cv_fold"].to_numpy(),
    )

    # --- baseline invariance (BLOCKING GATE) -------------------------------
    baseline_invariance = ab.check_baseline_invariance(
        reference_cohort, candidate_cohort, reference_result, candidate_result,
    )
    if baseline_invariance["status"] != "pass":
        return _finish_early(
            experiment_id, root, candidate, config, provenance, reproduction,
            status=ab.STATUS_BASELINE_INVARIANCE_FAILED,
            reason="the baseline chain differed despite the intended LST-only "
                   "intervention",
            alignment=alignment, baseline_invariance=baseline_invariance,
            modis_compatibility=modis_compatibility,
            shared_modis_invariance=shared_modis,
        )

    # --- stage: paired bootstrap -------------------------------------------
    y = reference_cohort["burned"].astype(int).to_numpy()
    bootstrap = ab.paired_block_bootstrap(
        reference_cohort, y,
        reference_result["oof_prob_baseline"],
        reference_result["oof_prob_thermal"],
        candidate_result["oof_prob_thermal"],
    )
    tables = root / "comparison" / "tables"
    step8_rows = ab.build_step8_metric_rows(
        reference_result, candidate_result, bootstrap["intervals"],
    )
    paired_rows = ab.build_paired_bootstrap_rows(
        reference_result, candidate_result, bootstrap,
    )
    step8_metrics_path = _write_csv(
        tables / "step8_metrics.csv", step8_rows, step8_rows[0].keys(),
    )
    paired_path = _write_csv(
        tables / "step8_paired_bootstrap.csv", paired_rows, paired_rows[0].keys(),
    )
    oof = ab.build_oof_predictions(
        reference_cohort, assignment, reference_result, candidate_result,
    )
    oof_path = _write_dataframe_csv(root / "comparison" / "oof_predictions.csv", oof)
    replicates_path = _write_dataframe_csv(
        tables / "step8_paired_bootstrap_replicates.csv", bootstrap["replicates"],
    )
    ab.write_checkpoint_stage(root, "paired_bootstrap", [
        step8_metrics_path, paired_path, oof_path, replicates_path,
    ], attestation=modis_compatibility)

    # --- stage: raster comparisons -----------------------------------------
    change_rows = []
    for product, threshold in ab.CHANGED_PIXEL_THRESHOLDS.items():
        ref_path = ab.product_path(reference_ctx, product, Path(reference_ctx["output_root"]))
        cand_path = ab.product_path(candidate_ctx, product, Path(candidate_ctx["output_root"]))
        if not ref_path.exists() or not cand_path.exists():
            continue
        change_rows.append(ab.compare_raster_change(
            ref_path, cand_path, product=product, changed_threshold=threshold,
        ))
    change_path = _write_csv(
        tables / "raster_change_summary.csv", change_rows, ab.RASTER_CHANGE_COLUMNS,
    )
    map_paths = _render_comparison_maps(
        experiment_id, reference_ctx, candidate_ctx, root / "comparison" / "maps",
    )
    ab.write_checkpoint_stage(
        root, "raster_comparison", [change_path], attestation=modis_compatibility,
    )

    # --- stage: boundary propagation ---------------------------------------
    boundary = ab.run_boundary_propagation(
        experiment_id, reference_ctx, candidate_ctx,
        tmp_dir=root / "_analysis_tmp",
    )
    boundary_path = _write_csv(
        tables / "boundary_propagation.csv", boundary["rows"],
        ab.BOUNDARY_PROPAGATION_COLUMNS,
    )
    boundary_summary = ab.summarize_boundary_propagation(boundary["verdicts"])
    ab.write_checkpoint_stage(
        root, "boundary_propagation", [boundary_path],
        attestation=modis_compatibility,
    )

    # --- decision -----------------------------------------------------------
    intervals = bootstrap["intervals"]
    evidence = {
        "reference_reproduction_status": reproduction["status"],
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
    decision = ab.decide_final_status(evidence)

    # --- stage: report generation ------------------------------------------
    # Report generation must not alter a single scientific number: the metric
    # payloads are snapshotted before rendering and re-checked afterwards.
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
    summary = ab.build_summary(
        experiment_id, candidate=candidate, config=config, provenance=provenance,
        reproduction=reproduction, alignment=alignment, fold_manifest=fold_manifest,
        baseline_invariance=baseline_invariance, raster_change_rows=change_rows,
        boundary_summary=boundary_summary, boundary_result=boundary,
        step8_metric_rows=step8_rows, paired_rows=paired_rows, bootstrap=bootstrap,
        decision=decision, modis_compatibility=modis_compatibility,
        shared_modis_invariance=shared_modis,
    )
    summary_path = root / "downstream_ab_summary.json"
    markdown_path = root / "downstream_ab_summary.md"
    ab.assert_downstream_namespace_safe([summary_path, markdown_path], experiment_id)
    ab.write_json_atomic(summary_path, summary)
    markdown_path.write_text(ab.render_summary_markdown(summary), encoding="utf-8")

    metrics_after = {"step8": step8_rows, "paired": paired_rows,
                     "rasters": change_rows, "boundary": boundary["rows"]}
    if not ab.report_generation_preserves_metrics(metrics_before, metrics_after):
        raise DownstreamABRunnerError(
            "report generation mutated a scientific metric payload; refusing to finish."
        )

    manifest = ab.build_manifest(experiment_id, root, summary)
    manifest_path = root / "downstream_ab_manifest.json"
    ab.write_json_atomic(manifest_path, manifest)
    ab.write_checkpoint_stage(root, "report_generation", [
        summary_path, markdown_path, manifest_path,
    ], attestation=modis_compatibility)
    audit_tmp = root / "_analysis_tmp"
    if audit_tmp.exists():
        import shutil

        ab.assert_downstream_namespace_safe([audit_tmp], experiment_id)
        shutil.rmtree(audit_tmp)

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


def _finish_early(
    experiment_id: str, root: Path, candidate: str, config: dict, provenance: dict,
    reproduction: dict, *, status: str, reason: str,
    alignment: dict | None = None, baseline_invariance: dict | None = None,
    modis_compatibility: dict | None = None,
    shared_modis_invariance: dict | None = None,
    technical_failure: str | None = None,
) -> dict:
    """Write the terminating report for a blocking gate failure.

    No candidate scientific conclusion is issued; the summary records exactly
    which gate stopped the experiment.
    """
    from collections import OrderedDict

    summary = OrderedDict((
        ("experiment", ab.DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("reference_chain", ab.CHAIN_REFERENCE),
        ("candidate_chain", candidate),
        ("report_schema_version", ab.REPORT_SCHEMA_VERSION),
        ("decision_rule_version", ab.DECISION_RULE_VERSION),
        ("final_status", status),
        ("final_status_meaning", ab.FINAL_STATUS_MEANINGS[status]),
        ("production_approved", False),
        ("changes_production_reducer", False),
        ("technical_failure", technical_failure),
        ("warnings", ab.summary_warnings(modis_compatibility)),
        ("modis_compatibility", ab.build_modis_compatibility_report(
            modis_compatibility, shared_modis_invariance,
        )),
        ("terminating_reason", reason),
        ("candidate_scientific_conclusion_issued", False),
        ("configuration", config),
        ("technical_validity", OrderedDict((
            ("reference_reproduction_status", reproduction.get("status")),
            ("baseline_invariance_status",
             (baseline_invariance or {}).get("status")),
            ("shared_modis_invariance_status",
             (shared_modis_invariance or {}).get("status")),
            ("modis_compatibility_mode",
             (modis_compatibility or {}).get("mode", ab.MODIS_STRICT_MODE)),
            ("modis_compatibility_attestation_status",
             (modis_compatibility or {}).get("status")),
            ("population_alignment_status", (alignment or {}).get("status")),
            ("raw_lst_grid_equality_passed",
             provenance["raw_lst_grid_equality_gate"]["passed"]),
            ("candidate_modifies_lst_only", ab.candidate_modifies_lst_only(provenance)),
            ("canonical_ndvi_identical_between_chains", ab.ndvi_inputs_identical(provenance)),
        ))),
        ("limitations", ab.required_limitations()),
        ("next_decision", ab.next_decision_text(status)),
    ))
    summary_path = root / "downstream_ab_summary.json"
    markdown_path = root / "downstream_ab_summary.md"
    ab.assert_downstream_namespace_safe([summary_path, markdown_path], experiment_id)
    ab.write_json_atomic(summary_path, summary)
    markdown_path.write_text(
        f"# Landsat compositing downstream A/B -- {experiment_id}\n\n"
        f"**Final status: `{status}`**\n\n"
        f"> {ab.FINAL_STATUS_MEANINGS[status]}\n\n"
        + (f"Technical failure: `{technical_failure}`\n\n" if technical_failure else "")
        + f"Terminating reason: {reason}\n\n"
        "No candidate scientific conclusion is issued.\n\n"
        + "\n".join(ab.render_warning_block(summary))
        + "\n## Limitations\n\n"
        + "\n".join(f"- {item}" for item in ab.required_limitations())
        + f"\n\n## Next decision\n\n{ab.next_decision_text(status)}\n",
        encoding="utf-8",
    )
    manifest = ab.build_manifest(experiment_id, root, summary)
    manifest_path = root / "downstream_ab_manifest.json"
    ab.write_json_atomic(manifest_path, manifest)
    ab.write_checkpoint_stage(root, "report_generation", [
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
def main(
    experiment_id: str,
    candidate: str = ab.CHAIN_CANDIDATE,
    dry_run: bool = False,
    run: bool = False,
    resume: bool = False,
    force: bool = False,
) -> dict:
    validate_modes(dry_run, run, resume, force)

    if candidate not in ab.SUPPORTED_CANDIDATES:
        raise DownstreamABRunnerError(
            f"unsupported --candidate {candidate!r}. Supported: "
            f"{list(ab.SUPPORTED_CANDIDATES)}."
        )

    if dry_run:
        try:
            plan = ab.build_dry_run_plan(experiment_id, candidate)
        except ab.DownstreamABError as exc:
            raise DownstreamABRunnerError(str(exc)) from exc
        _print_dry_run(plan)
        return {
            "experiment_id": experiment_id, "ran": False, "dry_run": True,
            "candidate": candidate, "plan": plan, "writes_performed": False,
        }

    try:
        with ab.EarthEngineGuard():
            return _run_live(experiment_id, candidate, force=force, resume=resume)
    except (ab.PrerequisiteError, ab.DownstreamABError, ab.NamespaceSafetyError,
            ab.GridMismatchError) as exc:
        raise DownstreamABRunnerError(str(exc)) from exc


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled downstream A/B experiment for the validated Landsat "
            "compositing counterfactual (scene_weighted_reference vs "
            "date_balanced_lst_only). Diagnostic only: it never changes the "
            "production reducer and never returns a production approval. "
            "Writes ONLY under outputs/diagnostics/"
            "landsat_composite_downstream_ab/<experiment_id>/."
        )
    )
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument(
        "--candidate", type=str, default=ab.CHAIN_CANDIDATE,
        choices=list(ab.SUPPORTED_CANDIDATES),
        help="Candidate compositing chain (LST-only date-balanced compositing).",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run", action="store_true",
        help="Resolve and print the plan; write nothing, create no directory, "
             "run no model, make no Earth Engine call.",
    )
    mode.add_argument(
        "--run", action="store_true",
        help="Explicit opt-in to execute the local A/B experiment.",
    )
    state = parser.add_mutually_exclusive_group()
    state.add_argument(
        "--resume", action="store_true",
        help="Requires --run. Reuse checkpointed stages whose recorded outputs "
             "still validate; recompute everything else.",
    )
    state.add_argument(
        "--force", action="store_true",
        help="Requires --run. Delete ONLY the dedicated A/B diagnostic namespace "
             "after a safety check, then rerun. Frozen experiment and "
             "counterfactual outputs are never touched.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = main(
        experiment_id=args.experiment,
        candidate=args.candidate,
        dry_run=args.dry_run,
        run=args.run,
        resume=args.resume,
        force=args.force,
    )
    print(json.dumps(result, indent=2, default=str))
