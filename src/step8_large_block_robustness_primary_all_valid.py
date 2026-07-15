"""Preregistered Step8 large-spatial-block robustness analysis for the
FORMAL Step8B primary population (STEP8B_PRIMARY_POPULATION = "all_valid").

This is a corrective, separately preregistered companion to the existing
burnable_tree_shrub_grass robustness analysis at
outputs/robustness/step8_large_block/manavgat_2021__bejis_2022/ (analysis_id
1759eed5dd1027bdde69413d226502c1b8548e1ad187b44bb4a896d9fe1f8edd). That
existing analysis is FROZEN and untouched by this module -- it remains valid
only as a natural-vegetation (burnable_tree_shrub_grass) sensitivity check,
never as evidence about the formal Step8B primary result.

CODE-PATH DISCIPLINE
=====================
This module does NOT maintain an independent copy of the Step8B OOF
algorithm. It calls, unmodified, the SAME shared functions Step8B's own CLI
path uses:
    - core.config.STEP8B_PRIMARY_POPULATION (population identity, sourced
      from the single source of truth, never hardcoded here)
    - step8b.filter_valid_for_modeling   (valid-row filtering)
    - step8b.build_population_masks      (population filtering)
    - step8b.add_spatial_block_id        (block/group-column construction;
      called with block_size_cells=2 and ALL DEFAULT kwargs for the
      equivalence gate, so it is byte-identical to Step8B's own call)
    - step8b.train_population            (preprocessing construction,
      classifier construction, OOF training loop, OOF coverage, metric
      computation -- called with group_column="large_block_id" and
      strict_folds=True for the 10/20-cell conditions, and with
      group_column="spatial_block_id", strict_folds=False for the 2-cell
      equivalence gate, i.e. Step8B's own default arguments)
    - step8b.build_classifier, step8b.build_pipeline, step8b.compute_binary_metrics
Frozen v1 (step8_large_block_robustness.py) infrastructure that is pure
utility (hashing, canonical JSON, grid validation, interval classification,
paired bootstrap) is reused via import rather than reimplemented. v1's
own analysis code/outputs are never modified.

Only the spatial grouping scale differs across the four scientific
conditions in this analysis; the population is held fixed at the formal
Step8B primary population.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.config import (
    STEP8B_MIN_POSITIVES_PER_POPULATION,
    STEP8B_N_SPLITS,
    STEP8B_PRIMARY_POPULATION,
    STEP8B_RANDOM_SEED,
    STEP8B_SPATIAL_BLOCK_SIZE_CELLS,
    STEP8C_CI_LOWER,
    STEP8C_CI_UPPER,
    STEP8C_N_BOOTSTRAP,
    STEP8C_RANDOM_SEED,
)
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT

# --- Reused, UNMODIFIED, from the frozen v1 large-block module (pure
# utilities / infrastructure -- not the Step8B OOF algorithm). ---
from src.step8_large_block_robustness import (
    EXPECTED_BLOCK_SIZES,
    EXPECTED_EXPERIMENTS,
    NOMINAL_SCALES,
    OUTPUT_ROOT as V1_OUTPUT_ROOT,
    PROTECTED_RELATIVE_PATHS,
    classify_interval,
    classify_joint,
    canonical_json,
    experiment_step8_root,
    hash_protected_inputs as hash_protected_original_inputs,
    paired_large_block_bootstrap,
    protected_paths as protected_original_paths,
    sha256_bytes,
    sha256_file,
    validate_canonical_grid,
    _package_versions,
    _git_commit,
)

# --- Reused, UNMODIFIED-BEHAVIOR-BY-DEFAULT, shared Step8B functions. ---
from src.step8b_train_baseline_vs_thermal_model import (
    BASELINE_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    THERMAL_FEATURES,
    THERMAL_MODEL_FEATURES,
    add_spatial_block_id,
    build_classifier,
    build_population_masks,
    compute_binary_metrics,
    filter_valid_for_modeling,
    train_population,
)

log, log_file = setup_logger("step8_large_block_robustness_primary_all_valid")

ANALYSIS_SCHEMA_VERSION = "step8.large_block_robustness_primary_all_valid.v1"
MODEL_NAME = "random_forest"
MIN_VALID_BOOTSTRAP = 900
PRIMARY_POPULATION = STEP8B_PRIMARY_POPULATION  # sourced, never hardcoded

# The formal Step8B primary population must literally be "all_valid" for
# this analysis to make sense; fail loudly (at import/definition time is too
# early to raise -- checked in scientific_configuration()) if it ever drifts.

OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "robustness" / "step8_large_block_primary_all_valid"
    / "manavgat_2021__bejis_2022"
)

# Frozen existing (v1, burnable_tree_shrub_grass) robustness analysis --
# read-only external reference, protected by hash, never rewritten here.
V1_EXPECTED_ANALYSIS_ID = (
    "1759eed5dd1027bdde69413d226502c1b8548e1ad187b44bb4a896d9fe1f8edd"
)


class Step8RobustnessPrimaryError(SystemExit):
    """Fail-fast error for the primary-population (all_valid) large-block analysis."""


# =============================================================================
# Protection: original Step8 A/B/C/E + existing (v1) robustness outputs
# =============================================================================
def hash_v1_robustness_tree(root: Path | None = None) -> dict[str, str]:
    """SHA-256 of every file under the frozen v1 robustness namespace."""
    root = V1_OUTPUT_ROOT if root is None else root
    if not root.exists():
        raise Step8RobustnessPrimaryError(
            f"Frozen existing (v1) robustness namespace not found: {root}. "
            "This analysis requires the existing burnable_tree_shrub_grass "
            "robustness analysis to already exist and be protected."
        )
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def assert_v1_analysis_id_matches(root: Path | None = None) -> str:
    root = V1_OUTPUT_ROOT if root is None else root
    report_path = root / "step8_large_block_final_report.json"
    if not report_path.is_file():
        raise Step8RobustnessPrimaryError(
            f"Frozen v1 final report missing: {report_path}"
        )
    report = json.loads(report_path.read_text())
    analysis_id = report.get("analysis_id")
    if analysis_id != V1_EXPECTED_ANALYSIS_ID:
        raise Step8RobustnessPrimaryError(
            "Frozen v1 robustness analysis_id does not match the expected, "
            f"protected value. Expected {V1_EXPECTED_ANALYSIS_ID}, found "
            f"{analysis_id}. Refusing to proceed -- the existing analysis "
            "may have been rerun, modified, or is not what this corrective "
            "analysis was authorized against."
        )
    return analysis_id


def assert_hash_dict_unchanged(before: dict[str, str], after: dict[str, str], label: str) -> None:
    changed = sorted(
        name for name, digest in before.items() if after.get(name) != digest
    )
    missing = sorted(name for name in before if name not in after)
    if changed or missing:
        raise Step8RobustnessPrimaryError(
            f"{label} protection failed; changed or missing files: "
            + ", ".join(changed + missing)
        )


def hash_all_protected() -> dict[str, Any]:
    """Hashes BOTH the original Step8A/B/C/E inputs/outputs AND the frozen
    v1 (burnable_tree_shrub_grass) robustness outputs. Both are read-only
    for this module."""
    return {
        "original_step8": hash_protected_original_inputs(),
        "v1_robustness_tree": hash_v1_robustness_tree(),
        "v1_analysis_id": assert_v1_analysis_id_matches(),
    }


def assert_all_protected_unchanged(before: dict[str, Any], after: dict[str, Any]) -> None:
    from src.step8_large_block_robustness import assert_protected_hashes_unchanged
    try:
        assert_protected_hashes_unchanged(before["original_step8"], after["original_step8"])
    except SystemExit as exc:
        raise Step8RobustnessPrimaryError(f"Original Step8 protection failed: {exc}") from exc
    assert_hash_dict_unchanged(
        before["v1_robustness_tree"], after["v1_robustness_tree"], "Existing (v1) robustness"
    )
    if after["v1_analysis_id"] != before["v1_analysis_id"]:
        raise Step8RobustnessPrimaryError(
            "Frozen v1 robustness analysis_id changed during this run."
        )


# =============================================================================
# Original Step8B/Step8C reference provenance for the FORMAL primary
# population (all_valid). Deliberately NOT calling v1's
# original_reference_payload(), which is hardcoded to v1's own
# (burnable_tree_shrub_grass) population -- v1's source file is frozen and
# not modified by this corrective analysis. This is provenance-validation
# glue code (reads two JSON files, checks recorded config against runtime
# config); it is not a copy of the Step8B OOF/model-fitting algorithm.
# =============================================================================
def original_reference_payload(experiment_id: str) -> dict[str, Any]:
    root = experiment_step8_root(experiment_id)
    metrics = json.loads((root / "step8b" / "step8b_model_comparison_metrics.json").read_text())
    bootstrap = json.loads((root / "step8c" / "step8c_bootstrap_metrics.json").read_text())
    spatial = metrics.get("spatial_cv_config", {})
    if spatial.get("spatial_block_size_cells") != STEP8B_SPATIAL_BLOCK_SIZE_CELLS or spatial.get("spatial_block_size_cells") != 2:
        raise Step8RobustnessPrimaryError(f"{experiment_id}: frozen original Step8 block size is not proven to be 2 cells.")
    if (
        spatial.get("method") != "StratifiedGroupKFold"
        or spatial.get("n_splits_requested") != STEP8B_N_SPLITS
        or spatial.get("random_state") != STEP8B_RANDOM_SEED
        or spatial.get("random_split_used") is not False
    ):
        raise Step8RobustnessPrimaryError(f"{experiment_id}: frozen original Step8 CV provenance disagrees with runtime configuration.")
    feature_sets = metrics.get("feature_sets", {})
    if (
        metrics.get("model") != MODEL_NAME
        or feature_sets.get("baseline") != BASELINE_FEATURES
        or feature_sets.get("thermal_additional") != THERMAL_FEATURES
        or feature_sets.get("thermal_model_full") != THERMAL_MODEL_FEATURES
    ):
        raise Step8RobustnessPrimaryError(f"{experiment_id}: frozen Step8B model/feature provenance disagrees with runtime configuration.")
    if (
        bootstrap.get("n_bootstrap_requested") != STEP8C_N_BOOTSTRAP
        or bootstrap.get("random_seed") != STEP8C_RANDOM_SEED
    ):
        raise Step8RobustnessPrimaryError(f"{experiment_id}: frozen Step8C bootstrap provenance disagrees with runtime configuration.")
    point = metrics.get("population_metrics", {}).get(PRIMARY_POPULATION)
    ci = bootstrap.get("bootstrap_ci_by_population", {}).get(PRIMARY_POPULATION)
    if not point or not ci:
        raise Step8RobustnessPrimaryError(f"{experiment_id}: primary-population ('{PRIMARY_POPULATION}') Step8B/Step8C reference is missing.")
    return {"metrics": metrics, "bootstrap": bootstrap, "point": point, "ci": ci}


def frozen_step8b_predictions(experiment_id: str) -> Path:
    return experiment_step8_root(experiment_id) / "step8b" / "step8b_predictions.parquet"


# =============================================================================
# Manifest / preregistration
# =============================================================================
def scientific_configuration(protected: dict[str, Any]) -> dict[str, Any]:
    if PRIMARY_POPULATION != "all_valid" or STEP8B_PRIMARY_POPULATION != "all_valid":
        raise Step8RobustnessPrimaryError(
            "This analysis requires the formal Step8B primary population to "
            f"be 'all_valid'; core.config.STEP8B_PRIMARY_POPULATION is "
            f"'{STEP8B_PRIMARY_POPULATION}'."
        )
    references = {experiment: original_reference_payload(experiment) for experiment in EXPECTED_EXPERIMENTS}
    classifier = build_classifier(MODEL_NAME, STEP8B_RANDOM_SEED)
    model_params = classifier.get_params(deep=False)
    return {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "experiment_ids": list(EXPECTED_EXPERIMENTS),
        "primary_population": PRIMARY_POPULATION,
        "primary_population_source": "core.config.STEP8B_PRIMARY_POPULATION",
        "predefined_block_sizes_cells": list(EXPECTED_BLOCK_SIZES),
        "nominal_scales": [NOMINAL_SCALES[size] for size in EXPECTED_BLOCK_SIZES],
        "fixed_grid_origin": {"row_500m": 0, "col_500m": 0},
        "block_construction": "block_row=floor(row_500m/block_size_cells); block_col=floor(col_500m/block_size_cells); large_block_id=b{size}_r{block_row}_c{block_col}",
        "original_small_block_reference_size_cells": STEP8B_SPATIAL_BLOCK_SIZE_CELLS,
        "baseline_features": list(BASELINE_FEATURES),
        "thermal_additional_features": list(THERMAL_FEATURES),
        "thermal_model_features": list(THERMAL_MODEL_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target_column": TARGET_COLUMN,
        "model_class": type(classifier).__name__,
        "model_name": MODEL_NAME,
        "model_hyperparameters": model_params,
        "preprocessing": {
            "numeric": "SimpleImputer(strategy=median)",
            "categorical": "SimpleImputer(strategy=most_frequent) + OneHotEncoder(handle_unknown=ignore)",
            "fit_scope": "inside each training fold only",
        },
        "cv": {"class": "StratifiedGroupKFold", "n_splits": STEP8B_N_SPLITS, "shuffle": True, "random_state": STEP8B_RANDOM_SEED, "dynamic_fold_reduction": False},
        "bootstrap": {"requested_replicates": STEP8C_N_BOOTSTRAP, "random_state": STEP8C_RANDOM_SEED, "ci_method": f"{STEP8C_CI_LOWER}/{STEP8C_CI_UPPER} percentile", "unit": "corresponding large spatial block", "paired": True, "minimum_valid_replicates": MIN_VALID_BOOTSTRAP},
        "primary_estimands": ["delta_roc_auc=thermal_oof_roc_auc-baseline_oof_roc_auc", "delta_pr_auc=thermal_oof_pr_auc-baseline_oof_pr_auc"],
        "interpretation_rules": {
            "positive": "bootstrap_supported_positive iff percentile CI entirely above zero",
            "uncertain": "uncertain_interval_includes_zero iff percentile CI includes zero",
            "negative": "bootstrap_supported_negative iff percentile CI entirely below zero",
            "joint": "supported_on_both_metrics iff both delta intervals entirely above zero; partial_metric_support iff exactly one is above zero; otherwise not_supported",
            "overall": "claim robustness across both predefined scales and both regions only if all four all_valid conditions support both metrics",
        },
        "two_cell_equivalence_gate": {
            "required_before_fitting": True,
            "probability_tolerance": 1e-12,
            "alignment": "cell_id (never row-position-only)",
        },
        "external_frozen_reference": {
            "v1_robustness_analysis_id": V1_EXPECTED_ANALYSIS_ID,
            "v1_population": "burnable_tree_shrub_grass",
            "v1_output_root": str(V1_OUTPUT_ROOT),
            "note": "read as an external frozen reference only; never recomputed or merged into this analysis's conclusions",
        },
        "protected_original_inputs": protected.get("original_step8"),
        "protected_v1_robustness_tree_file_count": len(protected.get("v1_robustness_tree", {})),
        "package_versions": _package_versions(),
        "verified_original_provenance": {
            experiment: {
                "spatial_block_size_cells": references[experiment]["metrics"]["spatial_cv_config"]["spatial_block_size_cells"],
                "n_splits_requested": references[experiment]["metrics"]["spatial_cv_config"]["n_splits_requested"],
                "model": references[experiment]["metrics"]["model"],
                "bootstrap_requested": references[experiment]["bootstrap"]["n_bootstrap_requested"],
                "bootstrap_seed": references[experiment]["bootstrap"]["random_seed"],
            } for experiment in EXPECTED_EXPERIMENTS
        },
        "prohibited_actions": ["post_hoc_block_selection", "feature_selection", "hyperparameter_tuning", "random_row_split", "probability_calibration", "prediction_inversion", "changing_block_sizes_after_seeing_results"],
    }


def build_manifest(protected: dict[str, Any]) -> dict[str, Any]:
    content = scientific_configuration(protected)
    analysis_id = sha256_bytes(canonical_json(content).encode("utf-8"))
    return {"analysis_id": analysis_id, "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(), "scientific_configuration": content}


def _preregistration_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Step8 Large-Spatial-Block Robustness Preregistration -- FORMAL PRIMARY (all_valid) -- IMMUTABLE",
        "",
        f"- analysis_id: {manifest['analysis_id']}",
        f"- created_at: {manifest['created_at']}",
        f"- experiments: {list(EXPECTED_EXPERIMENTS)}",
        f"- predefined block sizes: {list(EXPECTED_BLOCK_SIZES)}",
        f"- primary population: {PRIMARY_POPULATION} (source: core.config.STEP8B_PRIMARY_POPULATION)",
        f"- external frozen reference (NOT recomputed): v1 analysis_id {V1_EXPECTED_ANALYSIS_ID}, population burnable_tree_shrub_grass",
        "",
        "Only spatial grouping scale changes relative to the formal Step8B "
        "primary-population result. No favorable scale will be selected post hoc.",
    ]
    return "\n".join(lines) + "\n"


def validate_or_write_manifest(output_root: Path, protected: dict[str, Any]) -> dict[str, Any]:
    path = output_root / "step8_large_block_primary_all_valid_preregistration.json"
    md_path = output_root / "step8_large_block_primary_all_valid_preregistration.md"
    expected_content = scientific_configuration(protected)
    expected_id = sha256_bytes(canonical_json(expected_content).encode("utf-8"))
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("analysis_id") != expected_id or existing.get("scientific_configuration") != expected_content:
            raise Step8RobustnessPrimaryError("Existing immutable primary-population preregistration disagrees with runtime scientific configuration.")
        if not md_path.is_file() or md_path.read_text(encoding="utf-8") != _preregistration_markdown(existing):
            raise Step8RobustnessPrimaryError("Existing immutable Markdown preregistration is missing or changed.")
        return existing
    if md_path.exists():
        raise Step8RobustnessPrimaryError("Immutable Markdown preregistration exists without its JSON manifest.")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(protected)
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    md_path.write_text(_preregistration_markdown(manifest), encoding="utf-8")
    return manifest


def assert_downstream_outputs_writable(output_root: Path, force: bool) -> None:
    if force or not output_root.exists():
        return
    immutable = {
        output_root / "step8_large_block_primary_all_valid_preregistration.json",
        output_root / "step8_large_block_primary_all_valid_preregistration.md",
    }
    present = sorted(
        str(candidate) for candidate in output_root.rglob("*")
        if candidate.is_file() and candidate not in immutable
    )
    if present:
        raise Step8RobustnessPrimaryError(
            "Downstream primary-population robustness outputs already exist; use --force to overwrite only those: "
            + ", ".join(present)
        )


# =============================================================================
# 2-CELL REAL-DATA EQUIVALENCE GATE
# =============================================================================
def run_two_cell_equivalence_gate(experiment_id: str, tolerance: float = 1e-12) -> dict[str, Any]:
    """
    Reproduces the frozen original Step8B "all_valid" run via the SAME
    shared code path (filter_valid_for_modeling -> build_population_masks ->
    add_spatial_block_id(df, 2) [default kwargs, byte-identical to Step8B's
    own call] -> train_population(..., group_column="spatial_block_id",
    strict_folds=False) [Step8B's own default arguments]), then compares
    against the frozen outputs, aligned by cell_id.
    """
    step8a_path = experiment_step8_root(experiment_id) / "step8a" / "step8a_500m_modeling_dataset.parquet"
    if not step8a_path.is_file():
        raise Step8RobustnessPrimaryError(f"{experiment_id}: frozen Step8A input not found: {step8a_path}")
    frozen_pred_path = frozen_step8b_predictions(experiment_id)
    if not frozen_pred_path.is_file():
        raise Step8RobustnessPrimaryError(f"{experiment_id}: frozen Step8B predictions not found: {frozen_pred_path}")

    df_raw = pd.read_parquet(step8a_path)
    df_valid = filter_valid_for_modeling(df_raw)
    masks = build_population_masks(df_valid)
    df_pop = df_valid.loc[masks[PRIMARY_POPULATION]].reset_index(drop=True)
    df_pop = add_spatial_block_id(df_pop, STEP8B_SPATIAL_BLOCK_SIZE_CELLS)  # default kwargs only

    result = train_population(
        df_pop, PRIMARY_POPULATION, STEP8B_N_SPLITS, STEP8B_RANDOM_SEED, MODEL_NAME,
        STEP8B_MIN_POSITIVES_PER_POPULATION,
        group_column="spatial_block_id", strict_folds=False,
    )
    if result is None or result.get("skipped"):
        raise Step8RobustnessPrimaryError(
            f"{experiment_id}: shared code path could not reproduce a fitted "
            f"'{PRIMARY_POPULATION}' model for the 2-cell equivalence gate: "
            f"{result.get('reason') if result else 'no result'}"
        )

    fresh = pd.DataFrame({
        "cell_id": df_pop["cell_id"].to_numpy(),
        "row_500m": df_pop["row_500m"].to_numpy(),
        "col_500m": df_pop["col_500m"].to_numpy(),
        "spatial_block_id": df_pop["spatial_block_id"].to_numpy(),
        "burned": df_pop[TARGET_COLUMN].astype(int).to_numpy(),
        "fold_id": result["fold_id"],
        "y_prob_baseline": result["oof_prob_baseline"],
        "y_prob_thermal": result["oof_prob_thermal"],
    })

    frozen = pd.read_parquet(frozen_pred_path)
    frozen_pop = frozen.loc[frozen["population"] == PRIMARY_POPULATION].copy()
    if frozen_pop.empty:
        raise Step8RobustnessPrimaryError(
            f"{experiment_id}: frozen Step8B predictions contain no rows for "
            f"population '{PRIMARY_POPULATION}'."
        )

    merged = fresh.merge(
        frozen_pop[["cell_id", "burned", "fold_id", "spatial_block_id", "y_prob_baseline", "y_prob_thermal"]],
        on="cell_id", how="outer", suffixes=("_fresh", "_frozen"), indicator=True,
    )

    checks: dict[str, Any] = {}
    only_fresh = merged.loc[merged["_merge"] == "left_only", "cell_id"].tolist()
    only_frozen = merged.loc[merged["_merge"] == "right_only", "cell_id"].tolist()
    checks["same_evaluated_cell_ids"] = not only_fresh and not only_frozen
    checks["cell_ids_only_in_fresh"] = only_fresh[:20]
    checks["cell_ids_only_in_frozen"] = only_frozen[:20]
    checks["n_fresh_rows"] = int(len(fresh))
    checks["n_frozen_rows"] = int(len(frozen_pop))
    checks["same_row_count"] = len(fresh) == len(frozen_pop)

    both = merged.loc[merged["_merge"] == "both"].copy()
    checks["n_aligned_rows"] = int(len(both))
    checks["same_labels"] = bool((both["burned_fresh"] == both["burned_frozen"]).all())
    checks["same_fold_assignment_per_cell"] = bool((both["fold_id_fresh"] == both["fold_id_frozen"]).all())
    checks["same_spatial_block_assignment"] = bool(
        (both["spatial_block_id_fresh"].astype(str) == both["spatial_block_id_frozen"].astype(str)).all()
    )

    diff_baseline = (both["y_prob_baseline_fresh"] - both["y_prob_baseline_frozen"]).abs()
    diff_thermal = (both["y_prob_thermal_fresh"] - both["y_prob_thermal_frozen"]).abs()
    max_diff_baseline = float(diff_baseline.max()) if len(diff_baseline) else None
    max_diff_thermal = float(diff_thermal.max()) if len(diff_thermal) else None
    checks["max_abs_probability_difference_baseline"] = max_diff_baseline
    checks["max_abs_probability_difference_thermal"] = max_diff_thermal
    checks["same_baseline_oof_probability_per_cell"] = max_diff_baseline is not None and max_diff_baseline <= tolerance
    checks["same_thermal_oof_probability_per_cell"] = max_diff_thermal is not None and max_diff_thermal <= tolerance

    fresh_metrics_baseline = compute_binary_metrics(both["burned_fresh"].to_numpy(), both["y_prob_baseline_fresh"].to_numpy())
    fresh_metrics_thermal = compute_binary_metrics(both["burned_fresh"].to_numpy(), both["y_prob_thermal_fresh"].to_numpy())
    frozen_metrics_baseline = compute_binary_metrics(both["burned_frozen"].to_numpy(), both["y_prob_baseline_frozen"].to_numpy())
    frozen_metrics_thermal = compute_binary_metrics(both["burned_frozen"].to_numpy(), both["y_prob_thermal_frozen"].to_numpy())

    def _auc_close(a, b, name):
        if a is None or b is None:
            return False
        return abs(a - b) <= 1e-9

    checks["fresh_baseline_roc_auc"] = fresh_metrics_baseline["roc_auc"]
    checks["frozen_baseline_roc_auc"] = frozen_metrics_baseline["roc_auc"]
    checks["same_baseline_roc_auc"] = _auc_close(fresh_metrics_baseline["roc_auc"], frozen_metrics_baseline["roc_auc"], "baseline_roc_auc")
    checks["fresh_thermal_roc_auc"] = fresh_metrics_thermal["roc_auc"]
    checks["frozen_thermal_roc_auc"] = frozen_metrics_thermal["roc_auc"]
    checks["same_thermal_roc_auc"] = _auc_close(fresh_metrics_thermal["roc_auc"], frozen_metrics_thermal["roc_auc"], "thermal_roc_auc")
    checks["fresh_baseline_pr_auc"] = fresh_metrics_baseline["pr_auc"]
    checks["frozen_baseline_pr_auc"] = frozen_metrics_baseline["pr_auc"]
    checks["same_baseline_pr_auc"] = _auc_close(fresh_metrics_baseline["pr_auc"], frozen_metrics_baseline["pr_auc"], "baseline_pr_auc")
    checks["fresh_thermal_pr_auc"] = fresh_metrics_thermal["pr_auc"]
    checks["frozen_thermal_pr_auc"] = frozen_metrics_thermal["pr_auc"]
    checks["same_thermal_pr_auc"] = _auc_close(fresh_metrics_thermal["pr_auc"], frozen_metrics_thermal["pr_auc"], "thermal_pr_auc")

    checks["same_delta_roc_auc"] = (
        checks["same_baseline_roc_auc"] and checks["same_thermal_roc_auc"]
    )
    checks["same_delta_pr_auc"] = (
        checks["same_baseline_pr_auc"] and checks["same_thermal_pr_auc"]
    )

    required = [
        "same_evaluated_cell_ids", "same_row_count", "same_labels",
        "same_fold_assignment_per_cell", "same_spatial_block_assignment",
        "same_baseline_oof_probability_per_cell", "same_thermal_oof_probability_per_cell",
        "same_baseline_roc_auc", "same_thermal_roc_auc",
        "same_delta_roc_auc", "same_delta_pr_auc",
    ]
    passed = all(bool(checks[key]) for key in required)
    checks["experiment_id"] = experiment_id
    checks["tolerance"] = tolerance
    checks["gate_passed"] = passed
    checks["required_checks"] = required
    return checks


def write_equivalence_audit(output_root: Path, analysis_id: str, gate_results: list[dict[str, Any]]) -> dict[str, Any]:
    all_passed = all(g["gate_passed"] for g in gate_results)
    audit = {
        "analysis_id": analysis_id,
        "gate": "step8b_two_cell_equivalence_gate",
        "all_experiments_passed": all_passed,
        "per_experiment": gate_results,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "step8b_two_cell_equivalence_audit.json").write_text(json.dumps(audit, indent=2, default=str) + "\n")
    lines = [
        "# Step8B 2-Cell Real-Data Equivalence Audit",
        "",
        f"- analysis_id: `{analysis_id}`",
        f"- all experiments passed: **{all_passed}**",
        "",
    ]
    for g in gate_results:
        lines.append(f"## {g['experiment_id']}")
        lines.append("")
        lines.append(f"- gate_passed: **{g['gate_passed']}**")
        for key in g["required_checks"]:
            lines.append(f"  - {key}: {g[key]}")
        lines.append(f"  - max_abs_probability_difference_baseline: {g['max_abs_probability_difference_baseline']}")
        lines.append(f"  - max_abs_probability_difference_thermal: {g['max_abs_probability_difference_thermal']}")
        lines.append("")
    (output_root / "step8b_two_cell_equivalence_audit.md").write_text("\n".join(lines) + "\n")
    return audit


# =============================================================================
# LARGE-BLOCK (10/20-cell) CONDITIONS -- all_valid, shared code path
# =============================================================================
def run_large_block_condition(df_all: pd.DataFrame, experiment_id: str, block_size_cells: int, analysis_id: str) -> dict[str, Any]:
    validate_canonical_grid(df_all)
    assigned = add_spatial_block_id(
        df_all, block_size_cells,
        column_name="large_block_id", id_prefix=f"b{int(block_size_cells)}", include_row_col=True,
    )  # before valid/population filtering
    df_valid = filter_valid_for_modeling(assigned)
    masks = build_population_masks(df_valid)
    df_pop = df_valid.loc[masks[PRIMARY_POPULATION]].reset_index(drop=True)
    if df_pop.empty:
        raise Step8RobustnessPrimaryError(f"{experiment_id}: primary population '{PRIMARY_POPULATION}' is empty.")

    result = train_population(
        df_pop, PRIMARY_POPULATION, STEP8B_N_SPLITS, STEP8B_RANDOM_SEED, MODEL_NAME,
        STEP8B_MIN_POSITIVES_PER_POPULATION,
        group_column="large_block_id", strict_folds=True,
    )
    if result is None or result.get("skipped"):
        raise Step8RobustnessPrimaryError(
            f"{experiment_id} block={block_size_cells}: shared code path "
            f"could not fit '{PRIMARY_POPULATION}': {result.get('reason') if result else 'no result'}"
        )

    y = df_pop[TARGET_COLUMN].astype(int).to_numpy()
    predictions = pd.DataFrame({
        "analysis_id": analysis_id, "experiment_id": experiment_id,
        "block_size_cells": int(block_size_cells), "nominal_scale": NOMINAL_SCALES[int(block_size_cells)],
        "cell_id": df_pop["cell_id"].to_numpy(), "row_500m": df_pop["row_500m"].to_numpy(),
        "col_500m": df_pop["col_500m"].to_numpy(), "large_block_id": df_pop["large_block_id"].to_numpy(),
        "burned": y, "fold_id": result["fold_id"],
        "y_prob_baseline": result["oof_prob_baseline"], "y_prob_thermal": result["oof_prob_thermal"],
    })

    metrics = {
        "baseline_roc_auc": result["overall_baseline"]["roc_auc"], "thermal_roc_auc": result["overall_thermal"]["roc_auc"],
        "delta_roc_auc": result["delta_auc"],
        "baseline_pr_auc": result["overall_baseline"]["pr_auc"], "thermal_pr_auc": result["overall_thermal"]["pr_auc"],
        "delta_pr_auc": result["delta_pr_auc"],
    }
    grouped = df_pop.assign(_burned=y).groupby("large_block_id")["_burned"].agg(["size", "sum"])
    positives = grouped["sum"]
    fold_rows = []
    for fold_id, fr in enumerate(result["fold_rows"]):
        fold_rows.append({"fold_id": fold_id, **fr})
    block_audit = {
        "total_modeling_rows": int(len(df_pop)), "positive_rows": int(y.sum()),
        "negative_rows": int((y == 0).sum()), "total_large_blocks": int(len(grouped)),
        "blocks_containing_positives": int((positives > 0).sum()),
        "blocks_containing_negatives": int((positives < grouped["size"]).sum()),
        "number_of_cv_folds": result["n_splits_used"], "folds": fold_rows,
    }
    return {"metrics": metrics, "predictions": predictions, "block_audit": block_audit}


def _condition_output_dir(experiment_id: str, block_size_cells: int, output_root: Path | None = None) -> Path:
    output_root = OUTPUT_ROOT if output_root is None else output_root
    return output_root / experiment_id / f"block_{block_size_cells}_cells"


def write_condition_outputs(output_dir: Path, analysis_id: str, experiment_id: str, block_size: int, result: dict[str, Any], bootstrap: dict[str, Any], replicates: pd.DataFrame, force: bool) -> dict[str, Any]:
    expected = ["block_assignment_audit.json", "step8b_large_block_metrics.json", "step8b_large_block_predictions.parquet", "step8c_large_block_bootstrap_summary.json", "step8c_large_block_bootstrap_replicates.parquet"]
    present = [name for name in expected if (output_dir / name).exists()]
    if present and not force:
        raise Step8RobustnessPrimaryError(f"Downstream primary-population outputs exist in {output_dir}; use --force: {present}")
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {"analysis_id": analysis_id, "experiment_id": experiment_id, "block_size_cells": block_size, "nominal_scale": NOMINAL_SCALES[block_size], "primary_population": PRIMARY_POPULATION}
    audit = {**common, **result["block_audit"]}
    (output_dir / "block_assignment_audit.json").write_text(json.dumps(audit, indent=2, default=str) + "\n")
    classifier = build_classifier(MODEL_NAME, STEP8B_RANDOM_SEED)
    metrics = {**common, **result["metrics"], "feature_sets": {"baseline": BASELINE_FEATURES, "thermal_additional": THERMAL_FEATURES, "thermal_model_full": THERMAL_MODEL_FEATURES}, "model_class": type(classifier).__name__, "model_hyperparameters": classifier.get_params(deep=False), "cv": {"class": "StratifiedGroupKFold", "n_splits": STEP8B_N_SPLITS, "random_state": STEP8B_RANDOM_SEED}}
    (output_dir / "step8b_large_block_metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    result["predictions"].to_parquet(output_dir / "step8b_large_block_predictions.parquet", index=False)
    boot_payload = {**common, **bootstrap}
    (output_dir / "step8c_large_block_bootstrap_summary.json").write_text(json.dumps(boot_payload, indent=2, default=str) + "\n")
    for key, value in common.items():
        replicates[key] = value
    ordered = list(common) + [c for c in replicates.columns if c not in common]
    replicates[ordered].to_parquet(output_dir / "step8c_large_block_bootstrap_replicates.parquet", index=False)
    return {"metrics": metrics, "bootstrap": boot_payload, "audit": audit}


def reference_comparison_row(experiment_id: str, analysis_id: str) -> dict[str, Any]:
    reference = original_reference_payload(experiment_id)
    point, ci = reference["point"], reference["ci"]
    roc_ci, pr_ci = ci["delta_auc_ci95"], ci["delta_pr_auc_ci95"]
    roc_status, pr_status = classify_interval(roc_ci), classify_interval(pr_ci)
    valid = ci["n_bootstrap_successful"]
    stability = "stable" if valid >= MIN_VALID_BOOTSTRAP else "bootstrap_unstable"
    status = classify_joint(roc_status, pr_status) if stability == "stable" else "bootstrap_unstable"
    return {"analysis_id": analysis_id, "experiment": experiment_id, "nominal_block_scale": "approximately_1_km", "block_size_cells": 2, "source_type": "frozen_original_small_block", "primary_population": PRIMARY_POPULATION, "baseline_roc_auc": point["overall_baseline"]["roc_auc"], "thermal_roc_auc": point["overall_thermal"]["roc_auc"], "delta_roc_auc": point["delta_auc"], "delta_roc_auc_ci_low": roc_ci[0], "delta_roc_auc_ci_high": roc_ci[1], "baseline_pr_auc": point["overall_baseline"]["pr_auc"], "thermal_pr_auc": point["overall_thermal"]["pr_auc"], "delta_pr_auc": point["delta_pr_auc"], "delta_pr_auc_ci_low": pr_ci[0], "delta_pr_auc_ci_high": pr_ci[1], "valid_bootstrap_replicates": valid, "bootstrap_stability": stability, "roc_support": roc_status, "pr_support": pr_status, "robustness_status": status}


def large_comparison_row(condition: dict[str, Any], analysis_id: str) -> dict[str, Any]:
    metrics, bootstrap = condition["metrics"], condition["bootstrap"]
    roc = bootstrap["series"]["delta_auc"]; pr = bootstrap["series"]["delta_pr_auc"]
    roc_status = classify_interval([roc["ci_2_5"], roc["ci_97_5"]]); pr_status = classify_interval([pr["ci_2_5"], pr["ci_97_5"]])
    stability = bootstrap["bootstrap_stability"]
    status = classify_joint(roc_status, pr_status) if stability == "stable" else "bootstrap_unstable"
    return {"analysis_id": analysis_id, "experiment": metrics["experiment_id"], "nominal_block_scale": metrics["nominal_scale"], "block_size_cells": metrics["block_size_cells"], "source_type": "new_large_block_robustness_primary_all_valid", "primary_population": PRIMARY_POPULATION, "baseline_roc_auc": metrics["baseline_roc_auc"], "thermal_roc_auc": metrics["thermal_roc_auc"], "delta_roc_auc": metrics["delta_roc_auc"], "delta_roc_auc_ci_low": roc["ci_2_5"], "delta_roc_auc_ci_high": roc["ci_97_5"], "baseline_pr_auc": metrics["baseline_pr_auc"], "thermal_pr_auc": metrics["thermal_pr_auc"], "delta_pr_auc": metrics["delta_pr_auc"], "delta_pr_auc_ci_low": pr["ci_2_5"], "delta_pr_auc_ci_high": pr["ci_97_5"], "valid_bootstrap_replicates": bootstrap["valid_replicates"], "bootstrap_stability": stability, "roc_support": roc_status, "pr_support": pr_status, "robustness_status": status}


def write_final_report(output_root: Path, analysis_id: str, comparison: list[dict[str, Any]], conditions: list[dict[str, Any]], protected_status: str) -> dict[str, Any]:
    large = [row for row in comparison if row["source_type"] == "new_large_block_robustness_primary_all_valid"]
    all_supported = len(large) == 4 and all(row["robustness_status"] == "supported_on_both_metrics" for row in large) and all(c["bootstrap"]["bootstrap_stability"] == "stable" for c in conditions)
    failures = [{"experiment": row["experiment"], "block_size_cells": row["block_size_cells"], "roc_support": row["roc_support"], "pr_support": row["pr_support"], "bootstrap_stability": row["bootstrap_stability"]} for row in large if row["robustness_status"] != "supported_on_both_metrics"]
    statement = (
        "the formal Step8B primary-population (all_valid) thermal contribution "
        "remained bootstrap-supported across both predefined large-block scales "
        "in both wildfire regions"
    ) if all_supported else (
        "the formal Step8B primary-population (all_valid) result was sensitive "
        "to spatial block scale or metric support; unsupported "
        "region-scale-metric conditions are listed explicitly."
    )
    external_v1 = json.loads((V1_OUTPUT_ROOT / "step8_large_block_final_report.json").read_text())
    report = {
        "analysis_id": analysis_id,
        "report_schema_version": ANALYSIS_SCHEMA_VERSION,
        "scope": {
            "formal_step8b_primary_population_robustness": {
                "population": PRIMARY_POPULATION,
                "source": "this analysis",
                "conditions": large,
                "original_small_block_reference": [row for row in comparison if row["source_type"] == "frozen_original_small_block"],
                "overall": {"all_four_conditions_supported_on_both_metrics": all_supported, "statement": statement, "conditions_losing_support": failures},
            },
            "natural_vegetation_sensitivity_robustness": {
                "population": "burnable_tree_shrub_grass",
                "source": "external frozen reference, NOT recomputed here",
                "source_analysis_id": V1_EXPECTED_ANALYSIS_ID,
                "source_output_root": str(V1_OUTPUT_ROOT),
                "overall_predefined_scale_robustness": external_v1.get("overall_predefined_scale_robustness"),
            },
        },
        "block_fold_qa": [c["audit"] for c in conditions],
        "protected_original_step8_hash_check": protected_status,
        "protected_v1_robustness_hash_check": protected_status,
        "claim_boundaries": [
            "spatial autocorrelation was not claimed eliminated",
            "no causal thermal effects",
            "not operational wildfire prediction",
            "no statistical significance or p-values",
            "no proof that residual spatial dependence is absent",
            "not successful cross-region transfer",
            "no best block size was selected",
            "the natural-vegetation (burnable_tree_shrub_grass) figures above are an external frozen reference and must not be read as newly rerun or as evidence about the formal all_valid primary result",
        ],
    }
    (output_root / "step8_large_block_primary_all_valid_final_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")

    primary_rows = [[r["experiment"], r["block_size_cells"], r["nominal_block_scale"], f"{r['delta_roc_auc']:.6f}", f"[{r['delta_roc_auc_ci_low']:.6f}, {r['delta_roc_auc_ci_high']:.6f}]", r["roc_support"], f"{r['delta_pr_auc']:.6f}", f"[{r['delta_pr_auc_ci_low']:.6f}, {r['delta_pr_auc_ci_high']:.6f}]", r["pr_support"], r["robustness_status"]] for r in large]
    def table(headers, rows): return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    lines = [
        "# Step8 Large-Spatial-Block Robustness Report -- FORMAL PRIMARY (all_valid)", "",
        f"- analysis_id: `{analysis_id}`",
        f"- protected original Step8 hash check: **{protected_status}**",
        f"- protected existing (v1) robustness hash check: **{protected_status}**",
        "",
        "## Formal Step8B primary-population (all_valid) robustness -- NEW", "",
    ] + table(["experiment", "block cells", "nominal scale", "delta ROC-AUC", "ROC CI", "ROC support", "delta PR-AUC", "PR CI", "PR support", "joint status"], primary_rows) + [
        "", statement, "",
        "## Natural-vegetation (burnable_tree_shrub_grass) sensitivity robustness -- EXTERNAL FROZEN REFERENCE, NOT recomputed", "",
        f"- source analysis_id: `{V1_EXPECTED_ANALYSIS_ID}`",
        f"- source output root: `{V1_OUTPUT_ROOT}`",
        f"- overall (as originally reported): {report['scope']['natural_vegetation_sensitivity_robustness']['overall_predefined_scale_robustness']}",
        "", "## Claim boundaries", "",
    ] + [f"- {item}" for item in report["claim_boundaries"]]
    (output_root / "step8_large_block_primary_all_valid_final_report.md").write_text("\n".join(lines) + "\n")
    return report


def dry_run_plan(output_root: Path | None = None) -> dict[str, Any]:
    output_root = OUTPUT_ROOT if output_root is None else output_root
    protected = hash_all_protected()
    classifier = build_classifier(MODEL_NAME, STEP8B_RANDOM_SEED)
    references = {experiment: original_reference_payload(experiment) for experiment in EXPECTED_EXPERIMENTS}
    return {
        "ran": False, "dry_run": True,
        "experiments": list(EXPECTED_EXPERIMENTS),
        "block_sizes_cells": list(EXPECTED_BLOCK_SIZES),
        "primary_population": PRIMARY_POPULATION,
        "resolved_step8a_inputs": [str(experiment_step8_root(e) / "step8a" / "step8a_500m_modeling_dataset.parquet") for e in EXPECTED_EXPERIMENTS],
        "output_namespace": str(output_root),
        "external_frozen_reference_analysis_id": V1_EXPECTED_ANALYSIS_ID,
        "model": {"class": type(classifier).__name__, "hyperparameters": classifier.get_params(deep=False)},
        "cv": {"class": "StratifiedGroupKFold", "folds": STEP8B_N_SPLITS, "seed": STEP8B_RANDOM_SEED},
        "original_block_size_verified": {e: references[e]["metrics"]["spatial_cv_config"]["spatial_block_size_cells"] for e in EXPECTED_EXPERIMENTS},
        "two_cell_equivalence_gate_required_first": True,
        "fit_or_bootstrap_performed": False, "files_written": False,
        "protected_original_step8_files": len(protected["original_step8"]),
        "protected_v1_robustness_files": len(protected["v1_robustness_tree"]),
    }


def run_analysis(dry_run: bool = False, force: bool = False, run_large_block_fit: bool = False, output_root: Path | None = None) -> dict[str, Any]:
    """
    run_large_block_fit=False (default): runs everything through the 2-cell
    equivalence gate and returns its results for review, WITHOUT fitting the
    10/20-cell scientific conditions. This mirrors the required validation
    order (equivalence gate reviewed before the scientific run proceeds).
    """
    output_root = OUTPUT_ROOT if output_root is None else output_root
    if dry_run:
        return dry_run_plan(output_root)

    before = hash_all_protected()
    manifest_path = output_root / "step8_large_block_primary_all_valid_preregistration.json"
    if manifest_path.exists():
        manifest = validate_or_write_manifest(output_root, before)
        assert_downstream_outputs_writable(output_root, force)
    else:
        assert_downstream_outputs_writable(output_root, force)
        manifest = validate_or_write_manifest(output_root, before)
    analysis_id = manifest["analysis_id"]

    gate_results = [run_two_cell_equivalence_gate(experiment) for experiment in EXPECTED_EXPERIMENTS]
    audit = write_equivalence_audit(output_root, analysis_id, gate_results)

    after_gate = hash_all_protected()
    assert_all_protected_unchanged(before, after_gate)

    if not audit["all_experiments_passed"] or not run_large_block_fit:
        return {
            "ran": False,
            "analysis_id": analysis_id,
            "equivalence_audit": audit,
            "blocked": not audit["all_experiments_passed"],
            "reason": (
                "2-cell equivalence gate failed; 10/20-cell scientific run "
                "is blocked until reviewed."
            ) if not audit["all_experiments_passed"] else (
                "2-cell equivalence gate passed; awaiting explicit review "
                "before the 10/20-cell scientific run (run_large_block_fit=True)."
            ),
            "protected_hash_check": "passed",
        }

    data = {}
    for experiment in EXPECTED_EXPERIMENTS:
        path = experiment_step8_root(experiment) / "step8a" / "step8a_500m_modeling_dataset.parquet"
        data[experiment] = pd.read_parquet(path)

    conditions = []
    for experiment in EXPECTED_EXPERIMENTS:
        for block_size in EXPECTED_BLOCK_SIZES:
            result = run_large_block_condition(data[experiment], experiment, block_size, analysis_id)
            bootstrap, replicates = paired_large_block_bootstrap(result["predictions"])
            condition = write_condition_outputs(_condition_output_dir(experiment, block_size, output_root), analysis_id, experiment, block_size, result, bootstrap, replicates, force)
            conditions.append(condition)

    comparison = [reference_comparison_row(experiment, analysis_id) for experiment in EXPECTED_EXPERIMENTS] + [large_comparison_row(condition, analysis_id) for condition in conditions]
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison).to_csv(output_root / "step8_large_block_primary_all_valid_comparison.csv", index=False)

    after = hash_all_protected()
    assert_all_protected_unchanged(before, after)
    report = write_final_report(output_root, analysis_id, comparison, conditions, "passed")
    return {
        "ran": True, "analysis_id": analysis_id, "equivalence_audit": audit,
        "conditions": conditions, "report": report, "protected_hash_check": "passed",
    }