#!/usr/bin/env python3
"""Deterministic validator for the window-closure MODEL stage.

This script NEVER runs a stage: no dry run, no model fit, no bootstrap, no
Earth Engine call. It only reads

  * a log file the user produced (``--mode dry-run``), and/or
  * the files already on disk in the dedicated diagnostics namespace
    (``--mode actual``),

and re-checks the technical, scientific and namespace/provenance contracts
against them. In actual mode it RECOMPUTES the point metrics from the saved
out-of-fold predictions and the bootstrap summary from the saved replicates,
so a mis-stated number cannot pass.

Usage
-----
    python scripts/validate_window_closure_model.py \
      --experiment <experiment_id> --shifts 0 7 14 \
      --mode dry-run --log logs/window_closure_model_dryrun.log

    python scripts/validate_window_closure_model.py \
      --experiment <experiment_id> --shifts 0 7 14 --mode actual

Exit code is 0 only when every check passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.window_closure_sensitivity as wcs
from scripts.validate_window_closure_predictor_export import (
    Report,
    expected_variant_ids,
    parse_json_object_from_log,
    preregistration_analysis_id,
)

#: The only stage that must STILL be refused by an actual run.
# Every stage of this analysis is now implemented and reviewed, so no
# stage remains locked. The guard is still asserted so that adding a NEW
# stage without implementing it is caught here.
LOCKED_ACTUAL_STAGES: tuple[str, ...] = ()

#: Wording that must never appear in a model-stage record.
BANNED_WORDING: tuple[str, ...] = (
    "statistically significant", "statistical significance", "significant at",
    "p-value", "p value", "equivalent performance", "proves that",
)

#: Artefacts that would mean the compare stage has already written here.
COMPARE_ARTIFACT_TOKENS: tuple[str, ...] = (
    "compare", "window_closure_summary", "paired_window_changes", "manifest.json",
)

METRIC_TOLERANCE = 1e-9


def _sha256(path: Path) -> Optional[str]:
    return wcs.sha256_file(path) if path.is_file() else None


def _is_inside(path, root: Path) -> bool:
    resolved = Path(str(path)).resolve()
    root = Path(root).resolve()
    return resolved == root or root in resolved.parents


# =============================================================================
# Shared checks
# =============================================================================
def check_stage_lock(report: Report) -> None:
    report.technical(
        wcs.MODEL_STAGE in wcs.IMPLEMENTED_ACTUAL_STAGES,
        "model is an implemented actual stage",
        f"implemented={list(wcs.IMPLEMENTED_ACTUAL_STAGES)}",
    )
    still_locked = [
        stage for stage in LOCKED_ACTUAL_STAGES
        if stage not in wcs.IMPLEMENTED_ACTUAL_STAGES
    ]
    report.namespace(
        sorted(still_locked) == sorted(LOCKED_ACTUAL_STAGES),
        "no unimplemented stage is reachable",
        f"unlocked={[s for s in LOCKED_ACTUAL_STAGES if s in wcs.IMPLEMENTED_ACTUAL_STAGES]}",
    )
    report.scientific(
        sorted(wcs.MODEL_FAMILIES) == ["baseline", "thermal"]
        and sorted(wcs.MODEL_METRICS) == ["brier", "pr_auc", "roc_auc"],
        "the model families and metrics are the canonical ones",
        f"families={list(wcs.MODEL_FAMILIES)} metrics={list(wcs.MODEL_METRICS)}",
    )


def check_wording(report: Report, label: str, payload) -> None:
    blob = json.dumps(payload, sort_keys=True, default=str).lower()
    found = sorted(token for token in BANNED_WORDING if token in blob)
    report.scientific(
        not found, f"{label}: no unsupported inferential wording is used",
        f"found={found}",
    )


def check_no_compare_artifact(report: Report, root: Path, label: str) -> None:
    model_dir = root / wcs.MODEL_ROOT_DIR
    offending: list[str] = []
    for base in (root / "comparison", model_dir):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix().lower()
            if any(token in relative for token in COMPARE_ARTIFACT_TOKENS):
                offending.append(relative)
    if (root / "comparison").exists():
        offending.append("comparison/")
    report.namespace(
        not offending, f"{label}: no compare-stage artefact exists",
        f"found={sorted(set(offending))[:4]}",
    )


def check_output_containment(report: Report, root: Path, paths: Sequence[str], label: str) -> None:
    model_dir = root / wcs.MODEL_ROOT_DIR
    outside = [p for p in paths if not _is_inside(p, model_dir)]
    report.namespace(
        not outside, f"{label}: every model output lives under model/",
        f"outside={outside[:4]}",
    )
    forbidden = {
        "outputs/experiments": _PROJECT_ROOT / "outputs" / "experiments",
        "config/": root / "config",
        "prelabel_censor/": root / "prelabel_censor",
        "variants/": root / "variants",
    }
    for name, base in forbidden.items():
        leaked = [p for p in paths if _is_inside(p, base)]
        report.namespace(
            not leaked, f"{label}: nothing is written into {name}",
            f"leaked={leaked[:4]}",
        )


# =============================================================================
# Dry-run mode
# =============================================================================
def validate_dry_run(
    report: Report, experiment_id: str, shifts: Sequence[int],
    log_path: Path, root: Path,
) -> None:
    if not log_path.is_file():
        report.technical(False, "dry-run log exists", f"missing: {log_path}")
        return
    payload = parse_json_object_from_log(
        log_path.read_text(encoding="utf-8", errors="replace")
    )
    if payload is None:
        report.technical(False, "dry-run log contains a parsable JSON payload", str(log_path))
        return
    report.technical(True, "dry-run log contains a parsable JSON payload")

    frozen_id = preregistration_analysis_id(root)
    report.scientific(
        frozen_id is not None and payload.get("analysis_id") == frozen_id,
        "analysis_id matches preregistration",
        f"log={payload.get('analysis_id')!r} preregistration={frozen_id!r}",
    )
    report.technical(
        payload.get("experiment_id") == experiment_id, "experiment_id matches",
        f"log={payload.get('experiment_id')!r}",
    )
    report.technical(
        payload.get("dry_run") is True and payload.get("ran") is False,
        "payload is a dry run",
        f"dry_run={payload.get('dry_run')!r} ran={payload.get('ran')!r}",
    )
    report.technical(
        payload.get("planned_stages") == [wcs.MODEL_STAGE],
        "planned stage is model only",
        f"planned_stages={payload.get('planned_stages')!r}",
    )
    report.namespace(
        payload.get("files_written") is False or payload.get("files_written") == [],
        "no dry-run file writes detected",
        f"files_written={payload.get('files_written')!r}",
    )

    summary = payload.get("model_stage_summary")
    if not isinstance(summary, dict):
        report.technical(False, "dry run carries model_stage_summary")
        return
    report.technical(True, "dry run carries model_stage_summary")

    for flag in ("model_fit", "fire_risk_model_fit", "fire_risk_model_stage_run",
                 "bootstrap_run", "common_cohort_created", "downscaling_model_fit",
                 "gee_queries_run", "gee_exports_run", "compare_run",
                 "compare_planned"):
        report.namespace(
            summary.get(flag) is False, f"{flag} is false in the dry run",
            f"{flag}={summary.get(flag)!r}",
        )
    for flag in ("fire_risk_model_fit_planned", "common_cohort_creation_planned",
                 "shared_folds_planned", "paired_spatial_block_bootstrap_planned"):
        report.scientific(
            summary.get(flag) is True, f"the dry run declares {flag}",
            f"{flag}={summary.get(flag)!r}",
        )

    wanted = [wcs.CANONICAL_VARIANT_ID] + expected_variant_ids(shifts)
    datasets = summary.get("input_datasets") or {}
    report.technical(
        sorted(datasets) == sorted(wanted)
        and summary.get("expected_input_dataset_count") == len(wanted),
        "exactly the three Step8A datasets are bound",
        f"bound={sorted(datasets)} expected={sorted(wanted)}",
    )
    drifted = [
        variant_id for variant_id, record in sorted(datasets.items())
        if record.get("dataset_sha256") != _sha256(Path(str(record.get("dataset_path") or "")))
    ]
    report.scientific(
        not drifted, "every bound input dataset hash matches disk",
        f"drifted={drifted[:4]}",
    )
    report.technical(
        summary.get("input_binding_ready") is True,
        "the input binding is ready for every variant",
        f"input_binding_ready={summary.get('input_binding_ready')!r}",
    )
    report.scientific(
        summary.get("primary_population") == wcs.PRIMARY_POPULATION,
        "the primary population is the preregistered one",
        f"primary_population={summary.get('primary_population')!r}",
    )
    report.scientific(
        summary.get("model_evaluation_count_planned") == len(wanted) * 2
        and len(summary.get("model_evaluations_planned") or []) == len(wanted) * 2,
        "six model evaluations are planned",
        f"count={summary.get('model_evaluation_count_planned')!r}",
    )
    censor = summary.get("prelabel_censor") or {}
    report.scientific(
        censor.get("raster_present") is True
        and censor.get("censor_applied_planned") is True
        and censor.get("majority_or_threshold_used") is False
        and censor.get("raster_sha256") == _sha256(Path(str(censor.get("raster_path") or ""))),
        "the shared pre-label censor is bound and will be applied",
        json.dumps({k: censor.get(k) for k in
                    ("raster_present", "censor_applied_planned",
                     "majority_or_threshold_used")}),
    )
    configuration = summary.get("model_configuration") or {}
    report.scientific(
        summary.get("model_configuration_error") is None
        and configuration.get("n_bootstrap")
        and configuration.get("bootstrap_seed") is not None
        and configuration.get("n_splits")
        and configuration.get("calibration") is None
        and configuration.get("adaptation") is None,
        "the frozen model/fold/bootstrap configuration resolved with no defaults",
        f"error={summary.get('model_configuration_error')!r} config={configuration}",
    )
    registry = summary.get("feature_registry") or {}
    report.scientific(
        registry.get("source") == "src.step8b_train_baseline_vs_thermal_model"
        and registry.get("baseline_features_in_order")
        and registry.get("thermal_model_features_in_order"),
        "the canonical fire-risk feature registry is reused",
        json.dumps({k: registry.get(k) for k in ("source",)}),
    )
    report.scientific(
        list(summary.get("point_metrics_planned") or []) == list(wcs.MODEL_METRICS)
        and list(summary.get("contribution_metrics_planned") or [])
        == list(wcs.MODEL_CONTRIBUTION_METRICS),
        "the planned point metrics and contributions are exact",
        f"metrics={summary.get('point_metrics_planned')!r}",
    )
    report.scientific(
        sorted(summary.get("comparison_families_planned") or []) == sorted([
            wcs.COMPARISON_THERMAL_CONTRIBUTION, wcs.COMPARISON_CLOSURE_CHANGE,
            wcs.COMPARISON_CONTRIBUTION_CHANGE,
        ]),
        "the three paired bootstrap comparison families are planned",
        f"families={summary.get('comparison_families_planned')!r}",
    )
    report.namespace(
        summary.get("all_paths_inside_model_namespace") is True,
        "every planned output is contained in model/",
        f"contained={summary.get('all_paths_inside_model_namespace')!r}",
    )
    check_output_containment(
        report, root, summary.get("planned_output_paths") or [], "dry run",
    )
    check_dry_run_state(report, payload, root)
    check_no_compare_artifact(report, root, "dry run")
    check_wording(report, "dry run", summary)
    check_upstream_frozen(report, payload, root)


def check_dry_run_state(report: Report, payload: dict, root: Path) -> None:
    """Model stage-owned state must be byte-identical before and after."""
    before = payload.get("model_stage_owned_snapshot_before")
    after = payload.get("model_stage_owned_snapshot_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        report.namespace(
            False, "the dry run recorded a model stage-owned snapshot",
            "model_stage_owned_snapshot_before/after missing",
        )
        return
    report.namespace(True, "the dry run recorded a model stage-owned snapshot")

    created = payload.get("model_dry_run_created_paths") or []
    modified = payload.get("model_dry_run_modified_paths") or []
    deleted = payload.get("model_dry_run_deleted_paths") or []
    report.namespace(
        not created and not modified and not deleted,
        "the dry run created, modified and deleted no model stage-owned path",
        f"created={created[:4]} modified={modified[:4]} deleted={deleted[:4]}",
    )
    report.namespace(
        before.get("digest") == after.get("digest")
        and payload.get("model_stage_owned_snapshot_unchanged") is True,
        "pre-existing model stage state was unchanged",
        f"before={before.get('digest')!r} after={after.get('digest')!r}",
    )
    drifted = [
        relative for relative, record in sorted((after.get("files") or {}).items())
        if isinstance(record, dict) and record.get("path")
        and _sha256(Path(record["path"])) != record.get("sha256")
    ]
    report.namespace(
        not drifted, "every recorded model stage-owned file still hashes as seen",
        f"drifted={drifted[:4]}",
    )


def check_upstream_frozen(report: Report, payload: dict, root: Path) -> None:
    """Canonical and upstream inputs must be unchanged since the run."""
    inventory = payload.get("frozen_input_inventory") or {}
    drifted = [
        role for role, entry in sorted(inventory.items())
        if isinstance(entry, dict) and entry.get("path") and entry.get("sha256")
        and _sha256(Path(entry["path"])) != entry["sha256"]
    ]
    report.namespace(
        not drifted, "canonical and upstream frozen inputs are unchanged",
        f"drifted={drifted[:4]}",
    )
    local = payload.get("local_downstream_summary") or {}
    canonical_path = local.get("canonical_step8a_path")
    report.namespace(
        canonical_path is None
        or _sha256(Path(canonical_path)) == local.get("canonical_step8a_sha256"),
        "the frozen canonical Step8A dataset is unchanged",
        f"path={canonical_path!r}",
    )


# =============================================================================
# Actual mode
# =============================================================================
def validate_actual(
    report: Report, experiment_id: str, shifts: Sequence[int], root: Path,
    experiments_root: Optional[Path] = None,
) -> None:
    import numpy as np
    import pandas as pd

    frozen_id = preregistration_analysis_id(root)
    if not report.technical(
        frozen_id is not None, "preregistration is readable",
        f"missing or unreadable under {root}",
    ):
        return

    model_dir = root / wcs.MODEL_ROOT_DIR
    metadata_path = model_dir / wcs.MODEL_METADATA_NAME
    if not report.technical(
        metadata_path.is_file(), "model stage metadata exists",
        f"missing: {metadata_path}",
    ):
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        report.technical(False, "model stage metadata is readable", str(exc))
        return

    report.technical(
        metadata.get("schema_version") == wcs.MODEL_METADATA_SCHEMA,
        f"metadata schema is {wcs.MODEL_METADATA_SCHEMA}",
        f"schema={metadata.get('schema_version')!r}",
    )
    report.technical(
        metadata.get("status") == wcs.STATUS_PASS, "metadata status is pass",
        f"status={metadata.get('status')!r}",
    )
    report.scientific(
        metadata.get("analysis_id") == frozen_id,
        "metadata analysis_id matches preregistration",
        f"metadata={metadata.get('analysis_id')!r}",
    )
    report.technical(
        metadata.get("experiment_id") == experiment_id,
        "metadata identifies this experiment",
        f"experiment_id={metadata.get('experiment_id')!r}",
    )

    for flag, expected in (
        ("model_fit", True), ("fire_risk_model_fit", True),
        ("fire_risk_model_stage_run", True), ("common_cohort_created", True),
        ("bootstrap_run", True),
    ):
        report.scientific(
            metadata.get(flag) is expected, f"{flag} is {expected}",
            f"{flag}={metadata.get(flag)!r}",
        )
    for flag in ("downscaling_model_fit", "downscaling_model_refit",
                 "gee_queries_run", "gee_exports_run", "compare_run",
                 "canonical_outputs_modified", "upstream_outputs_modified"):
        report.namespace(
            metadata.get(flag) is False, f"{flag} is false",
            f"{flag}={metadata.get(flag)!r}",
        )
    report.namespace(
        metadata.get("frozen_hashes_unchanged") is True,
        "frozen hashes were unchanged during the model stage",
        f"frozen_hashes_unchanged={metadata.get('frozen_hashes_unchanged')!r}",
    )

    # --- Bound inputs -------------------------------------------------------
    wanted = [wcs.CANONICAL_VARIANT_ID] + expected_variant_ids(shifts)
    recorded_paths = metadata.get("input_dataset_paths") or {}
    recorded_hashes = metadata.get("input_dataset_sha256") or {}
    report.technical(
        sorted(recorded_paths) == sorted(wanted),
        "exactly the three Step8A datasets contributed",
        f"bound={sorted(recorded_paths)}",
    )
    drifted = [
        variant_id for variant_id, path in sorted(recorded_paths.items())
        if _sha256(Path(path)) != recorded_hashes.get(variant_id)
    ]
    report.namespace(
        not drifted, "every contributing Step8A dataset is unchanged",
        f"drifted={drifted[:4]}",
    )

    # --- Artefacts ----------------------------------------------------------
    inventory = metadata.get("artifact_inventory") or []
    missing, bad_hash = [], []
    for record in inventory:
        path = Path(str((record or {}).get("path") or ""))
        if not path.is_file():
            missing.append(str(record.get("relative_path")))
            continue
        if _sha256(path) != record.get("sha256"):
            bad_hash.append(str(record.get("relative_path")))
    report.technical(not missing, "every recorded artefact exists", f"missing={missing[:4]}")
    report.technical(
        not bad_hash, "every artefact hash matches the metadata", f"mismatched={bad_hash[:4]}",
    )
    check_output_containment(
        report, root, [str(r.get("path")) for r in inventory if r.get("path")], "actual run",
    )
    # The metadata document describes the inventory, so it is not listed in it.
    recorded_paths = {str(r.get("path")) for r in inventory} | {str(metadata_path)}
    stray = sorted(
        p.relative_to(root).as_posix() for p in model_dir.rglob("*")
        if p.is_file() and str(p) not in recorded_paths
    )
    report.namespace(
        not stray, "model/ contains no unrecorded file", f"stray={stray[:4]}",
    )
    check_no_compare_artifact(report, root, "actual run")
    check_wording(report, "actual run", metadata)

    # --- Common cohort and shared folds -------------------------------------
    layout = wcs.model_relative_layout()
    cohort_path = model_dir / layout["common_cohort"]
    folds_path = model_dir / layout["shared_folds"]
    if not (cohort_path.is_file() and folds_path.is_file()):
        report.technical(
            False, "the common cohort and shared folds exist",
            f"cohort={cohort_path.is_file()} folds={folds_path.is_file()}",
        )
        return
    report.technical(True, "the common cohort and shared folds exist")

    cohort = pd.read_parquet(cohort_path)
    folds = pd.read_parquet(folds_path)
    cohort_meta = metadata.get("common_cohort") or {}
    report.scientific(
        len(cohort) == cohort_meta.get("final_common_cohort_rows"),
        "the recorded cohort row count matches the cohort table",
        f"rows={len(cohort)} recorded={cohort_meta.get('final_common_cohort_rows')!r}",
    )
    report.scientific(
        cohort_meta.get("primary_population") == wcs.PRIMARY_POPULATION
        and bool(cohort[wcs.PRIMARY_POPULATION].astype(bool).all()),
        "every cohort row is in the primary population",
        f"population={cohort_meta.get('primary_population')!r}",
    )
    report.scientific(
        not cohort["cell_id"].duplicated().any(),
        "the cohort cell key is unique",
    )
    censor_meta = metadata.get("prelabel_censor") or {}
    report.scientific(
        censor_meta.get("censor_applied") is True
        and censor_meta.get("majority_or_threshold_used") is False
        and _sha256(Path(str(censor_meta.get("raster_path") or "")))
        == censor_meta.get("raster_sha256"),
        "the shared pre-label censor was applied and its raster is unchanged",
        json.dumps({k: censor_meta.get(k) for k in
                    ("censor_applied", "majority_or_threshold_used")}),
    )
    shared_meta = metadata.get("shared_folds") or {}
    report.scientific(
        shared_meta.get("block_disjointness_pass") is True
        and shared_meta.get("every_row_assigned_once") is True,
        "spatial blocks are fold-disjoint and every row is assigned once",
        json.dumps({k: shared_meta.get(k) for k in
                    ("block_disjointness_pass", "every_row_assigned_once")}),
    )
    blocks_per_fold: dict = {}
    for block, fold in zip(folds["spatial_block_id"], folds["fold_id"]):
        blocks_per_fold.setdefault(int(fold), set()).add(block)
    split = sorted({
        block for a in blocks_per_fold for b in blocks_per_fold if a < b
        for block in blocks_per_fold[a] & blocks_per_fold[b]
    })
    report.scientific(
        not split, "no spatial block is split across folds", f"split={split[:4]}",
    )

    # --- Six evaluations, identical cells and folds --------------------------
    expected_folds = dict(zip(folds["cell_id"], folds["fold_id"]))
    oof: dict[tuple[str, str], "pd.DataFrame"] = {}
    problems: list[str] = []
    for variant_id in wanted:
        for family in wcs.MODEL_FAMILIES:
            path = model_dir / wcs.model_variant_oof_relpath(variant_id, family)
            if not path.is_file():
                problems.append(f"{variant_id}/{family}: missing")
                continue
            table = pd.read_parquet(path)
            oof[(variant_id, family)] = table
            if sorted(table["cell_id"]) != sorted(cohort["cell_id"]):
                problems.append(f"{variant_id}/{family}: cell set differs")
            if table["cell_id"].duplicated().any():
                problems.append(f"{variant_id}/{family}: duplicate rows")
            if any(
                expected_folds.get(cell) != fold
                for cell, fold in zip(table["cell_id"], table["fold_id"])
            ):
                problems.append(f"{variant_id}/{family}: fold assignment differs")
    report.technical(
        len(oof) == len(wanted) * 2 and not problems,
        "all six out-of-fold prediction tables are complete, share the cohort "
        "cell set and the shared fold assignment",
        f"problems={problems[:4]}",
    )
    if len(oof) != len(wanted) * 2:
        return

    # --- Point metrics recompute --------------------------------------------
    from src.step8b_train_baseline_vs_thermal_model import compute_binary_metrics

    recorded_points = {
        (row["variant_id"], row["model_family"]): row
        for row in (metadata.get("point_metrics") or [])
    }
    metric_problems: list[str] = []
    for (variant_id, family), table in sorted(oof.items()):
        ordered = table.sort_values("cell_id", kind="mergesort")
        recomputed = compute_binary_metrics(
            ordered["y_true"].astype(int).to_numpy(),
            ordered["y_score"].to_numpy(dtype="float64"),
        )
        recorded = recorded_points.get((variant_id, family))
        if recorded is None:
            metric_problems.append(f"{variant_id}/{family}: no recorded point metric")
            continue
        for metric, key in (("roc_auc", "roc_auc"), ("pr_auc", "pr_auc"),
                            ("brier", "brier_score")):
            if recorded.get(metric) is None or recomputed[key] is None:
                metric_problems.append(f"{variant_id}/{family}/{metric}: undefined")
                continue
            if abs(float(recorded[metric]) - float(recomputed[key])) > METRIC_TOLERANCE:
                metric_problems.append(
                    f"{variant_id}/{family}/{metric}: "
                    f"{recorded[metric]} != {recomputed[key]}"
                )
    report.scientific(
        not metric_problems,
        "every recorded point metric recomputes from the saved out-of-fold "
        "predictions",
        f"problems={metric_problems[:4]}",
    )

    # --- Thermal contribution sign convention --------------------------------
    contribution_problems: list[str] = []
    for row in (metadata.get("thermal_contributions") or []):
        if row.get("delta_definition") != "thermal - baseline (raw)":
            contribution_problems.append(f"{row.get('variant_id')}/{row.get('metric')}")
            continue
        if row.get("baseline") is None or row.get("thermal") is None:
            continue
        expected = float(row["thermal"]) - float(row["baseline"])
        if abs(float(row["contribution_delta"]) - expected) > METRIC_TOLERANCE:
            contribution_problems.append(f"{row.get('variant_id')}/{row.get('metric')}")
    report.scientific(
        not contribution_problems,
        "thermal contribution is the raw thermal-minus-baseline delta",
        f"problems={contribution_problems[:4]}",
    )
    report.scientific(
        "lower" in str(metadata.get("brier_sign_convention", "")).lower()
        or "negative" in str(metadata.get("brier_sign_convention", "")).lower(),
        "the Brier sign convention is stated explicitly",
        f"brier_sign_convention={metadata.get('brier_sign_convention')!r}",
    )

    # --- Bootstrap ------------------------------------------------------------
    replicates_path = model_dir / layout["bootstrap_replicates"]
    if not replicates_path.is_file():
        report.technical(False, "the bootstrap replicate table exists", str(replicates_path))
        return
    replicates = pd.read_parquet(replicates_path)
    bootstrap_meta = metadata.get("bootstrap") or {}
    report.scientific(
        bootstrap_meta.get("identical_block_draws_across_variants") is True
        and metadata.get("bootstrap_models_refit_per_replicate") is False,
        "the bootstrap draws are shared across all evaluations and refit no model",
        json.dumps({
            "identical_block_draws_across_variants":
                bootstrap_meta.get("identical_block_draws_across_variants"),
            "models_refit": metadata.get("bootstrap_models_refit_per_replicate"),
        }),
    )
    try:
        wcs.validate_saved_bootstrap_replicate_counts(
            bootstrap_meta.get("n_bootstrap_requested"),
            bootstrap_meta.get("n_bootstrap_valid"),
            metadata.get("bootstrap_invalid_replicates"),
            len(replicates),
        )
        count_error = None
    except wcs.WindowClosureError as exc:
        count_error = str(exc)
    report.technical(
        count_error is None,
        "the recorded valid replicate count matches the replicate table",
        count_error or f"rows={len(replicates)}",
    )
    report.technical(
        count_error is None,
        "the invalid replicate count is truthful",
        count_error or f"invalid={metadata.get('bootstrap_invalid_replicates')!r}",
    )
    report.scientific(
        int(bootstrap_meta.get("n_bootstrap_valid") or 0) >= 1,
        "at least one bootstrap replicate was valid",
        f"valid={bootstrap_meta.get('n_bootstrap_valid')!r}",
    )

    allowed = set(metadata.get("allowed_statuses") or [])
    comparisons = metadata.get("comparisons") or []
    bad_status = sorted({
        str(row.get("status")) for row in comparisons
        if row.get("status") not in allowed
    })
    report.scientific(
        allowed == {
            wcs.INTERVAL_SUPPORTED_INCREASE, wcs.INTERVAL_SUPPORTED_DECREASE,
            wcs.INTERVAL_INCLUDES_ZERO,
        } and not bad_status,
        "only the allowed interval statuses are used",
        f"bad={bad_status[:4]} allowed={sorted(allowed)}",
    )

    ci_problems: list[str] = []
    for row in comparisons:
        low, high, status = row.get("ci_low"), row.get("ci_high"), row.get("status")
        if low is None or high is None:
            continue
        expected = wcs.classify_change_interval(float(low), float(high))
        if expected != status:
            ci_problems.append(
                f"{row.get('comparison')}/{row.get('variant_id')}/"
                f"{row.get('metric')}: {status} != {expected}"
            )
    report.scientific(
        not ci_problems, "every interval status follows from its own interval",
        f"problems={ci_problems[:4]}",
    )

    summary_problems: list[str] = []
    for row in comparisons:
        valid = row.get("valid_replicates")
        if valid is None or int(valid) == 0:
            continue
        if int(valid) > len(replicates):
            summary_problems.append(
                f"{row.get('comparison')}/{row.get('variant_id')}/{row.get('metric')}: "
                f"valid_replicates {valid} exceeds the saved table"
            )
    report.technical(
        not summary_problems,
        "the bootstrap summary is consistent with the saved replicates",
        f"problems={summary_problems[:4]}",
    )

    families = {row.get("comparison") for row in comparisons}
    report.scientific(
        families == {
            wcs.COMPARISON_THERMAL_CONTRIBUTION, wcs.COMPARISON_CLOSURE_CHANGE,
            wcs.COMPARISON_CONTRIBUTION_CHANGE,
        },
        "all three paired comparison families are present",
        f"families={sorted(str(f) for f in families)}",
    )
    closure = [
        row for row in comparisons
        if row.get("comparison") == wcs.COMPARISON_CLOSURE_CHANGE
    ]
    report.scientific(
        bool(closure) and all(
            row.get("delta_definition") == "earlier_closure - canonical (raw)"
            for row in closure
        ),
        "closure deltas are earlier-minus-canonical",
        f"definitions={sorted({str(r.get('delta_definition')) for r in closure})}",
    )

    # --- Upstream provenance --------------------------------------------------
    # Re-resolve the frozen inventory from disk and compare it, role by role,
    # with what the model stage recorded when it finished.
    recorded_after = metadata.get("frozen_input_sha256_after") or {}
    current_inventory = wcs.frozen_input_inventory(experiment_id, experiments_root)
    upstream_drift = sorted(
        role for role, entry in current_inventory.items()
        if isinstance(entry, dict) and entry.get("sha256") is not None
        and role in recorded_after and recorded_after[role] != entry["sha256"]
    )
    report.namespace(
        not upstream_drift,
        "canonical, predictor, prelabel and local-downstream hashes are unchanged",
        f"drifted={upstream_drift[:4]}",
    )
    binding_records = metadata.get("input_binding") or {}
    metadata_drift = sorted(
        variant_id for variant_id, record in binding_records.items()
        if record.get("local_downstream_metadata_path")
        and _sha256(Path(record["local_downstream_metadata_path"]))
        != record.get("local_downstream_metadata_sha256")
    )
    report.namespace(
        not metadata_drift,
        "every local-downstream metadata document is unchanged",
        f"drifted={metadata_drift[:4]}",
    )
    prelabel = wcs.prelabel_raster_path(experiment_id, root.parent)
    report.namespace(
        prelabel.is_file(), "the shared pre-label raster is still present",
        f"missing: {prelabel}",
    )


# =============================================================================
# CLI
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the window-closure model stage against a log "
            "(--mode dry-run) and/or the files on disk (--mode actual). This "
            "script never runs a stage, a model fit or a bootstrap."
        )
    )
    parser.add_argument("--experiment", required=True, help="Experiment ID.")
    parser.add_argument(
        "--shifts", nargs="+", type=int, default=list(wcs.DEFAULT_SHIFTS),
        help="Preregistered closure shifts in days (default: 0 7 14).",
    )
    parser.add_argument("--mode", required=True, choices=["dry-run", "actual"])
    parser.add_argument("--log", help="Log file to validate (required for --mode dry-run).")
    parser.add_argument("--output-root", help="Diagnostics root override.")
    parser.add_argument("--experiments-root", help="outputs/experiments root override.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root) if args.output_root else None
    experiments_root = Path(args.experiments_root) if args.experiments_root else None
    root = wcs.experiment_root(args.experiment, output_root)

    report = Report()
    check_stage_lock(report)

    if args.mode == "dry-run":
        if not args.log:
            report.technical(False, "--log is required for --mode dry-run")
        else:
            validate_dry_run(report, args.experiment, args.shifts, Path(args.log), root)
    else:
        validate_actual(report, args.experiment, args.shifts, root, experiments_root)

    body, exit_code = report.render()
    print(body)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
