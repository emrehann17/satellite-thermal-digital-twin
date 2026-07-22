"""Preregistered Step8 large-spatial-block robustness analysis for a SINGLE
experiment (block sizes 10 and 20 cells, approximately 5 km / 10 km), with a
read-only comparison against that experiment's existing small-block (2-cell)
Step8B/Step8C results.

This module generalizes the pair-based (manavgat_2021 + bejis_2022) large
block robustness analyses at src/step8_large_block_robustness.py (v1,
primary population burnable_tree_shrub_grass) and
src/step8_large_block_robustness_primary_all_valid.py (v2, primary
population all_valid) to an arbitrary single `experiment_id`. Those two
modules are FROZEN and untouched by this file -- their EXPECTED_EXPERIMENTS
tuple hard-codes exactly ("manavgat_2021", "bejis_2022") and cannot run for
any other experiment.

CODE-PATH DISCIPLINE
=====================
This module does NOT maintain an independent copy of the Step8B OOF
algorithm. It calls, unmodified, the SAME shared functions Step8B's own CLI
path uses:
    - step8b.filter_valid_for_modeling   (valid-row filtering)
    - step8b.build_population_masks      (population filtering)
    - step8b.add_spatial_block_id        (block/group-column construction;
      called with an arbitrary block_size_cells and a distinct
      column_name="big_block_id" so it never collides with Step8B's own
      "spatial_block_id" column)
    - step8b.train_population            (preprocessing construction,
      classifier construction, OOF training loop, OOF coverage, metric
      computation -- called with group_column="big_block_id" and
      strict_folds=True)
    - step8b.build_classifier, step8b.build_pipeline, step8b.compute_binary_metrics
Frozen v1 (step8_large_block_robustness.py) infrastructure that is pure
utility (hashing, canonical JSON, grid validation, package/git provenance)
is reused via import rather than reimplemented; v1's own analysis code and
outputs are never modified. v1's paired-bootstrap/interval-classification
helpers are NOT reused verbatim here because this analysis requires a
Brier-score delta in the paired bootstrap (v1's helper drops Brier) and a
distinct decision-vocabulary (supported_positive/uncertain/supported_negative,
retained/partially_retained/not_retained, strongly_robust/moderately_robust/
scale_sensitive/not_robust) requested for this analysis specifically -- see
classify_metric_support/classify_brier_support/classify_overall_support/
classify_final_robustness below.

No AOI-specific branching: every public function takes `experiment_id` as an
ordinary string parameter, and this file must never compare that parameter
against a literal region name anywhere.
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
    PROTECTED_RELATIVE_PATHS,
    canonical_json,
    experiment_step8_root,
    hash_protected_inputs,
    sha256_bytes,
    sha256_file,
    validate_canonical_grid,
    _package_versions,
    _git_commit,
)

# --- Reused, UNMODIFIED-BEHAVIOR-BY-DEFAULT, shared Step8B functions. ---
from src.step8c_spatial_block_bootstrap_uncertainty import compute_metrics as compute_paired_metrics
from src.step8b_train_baseline_vs_thermal_model import (
    BASELINE_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    THERMAL_FEATURES,
    THERMAL_MODEL_FEATURES,
    Step8BError,
    add_spatial_block_id,
    build_classifier,
    build_population_masks,
    filter_valid_for_modeling,
    train_population,
)

log, log_file = setup_logger("step8_big_block_robustness")

ANALYSIS_SCHEMA_VERSION = "step8.big_block_robustness.v2"
MODEL_NAME = "random_forest"
MIN_VALID_BOOTSTRAP = 900
PRIMARY_POPULATION = "burnable_tree_shrub_grass"
DEFAULT_BLOCK_SIZES = (10, 20)
BLOCK_COLUMN = "big_block_id"
EFFECT_MAGNITUDE_STABLE_BAND = 0.05  # +/-5% relative change treated as "stable"

# v1 -> v2 reporting-schema migration notes. This bump changes reporting
# semantics only -- it never re-fits models, rebuilds folds/OOF predictions,
# or reruns bootstrap; the underlying Parquet artifacts from a v1 run remain
# byte-identical and valid inputs to a v2 report regeneration.
MIGRATION_NOTES_V1_TO_V2 = [
    "Added a canonical, positive-is-better Brier field (`brier_improvement` "
    "= baseline_brier - thermal_brier) alongside the deprecated legacy "
    "`delta_brier` field (thermal_brier - baseline_brier, negative-is-better), "
    "which is retained unchanged for backward compatibility.",
    "The previously supplied, incorrect small-block reference figures "
    "remain only a diagnostic cross-check (`reference_metric_mismatch`); "
    "the frozen on-disk Step8B/Step8C artifacts are, and always were, the "
    "sole scientific source of truth for the small-block comparison row.",
    "Split the single robustness verdict into `support_robustness_status` "
    "(whether bootstrap support for a positive thermal contribution is "
    "retained across block scales) and `effect_magnitude_stability_status` "
    "(whether the SIZE of that contribution grows, shrinks, or stays stable "
    "across block scales) -- these are independent questions and were "
    "previously conflated under one field name (`final_robustness_status`).",
    "Renamed the lst_anomaly_mean provenance key `legacy_feature_name` to "
    "`column_name` (value unchanged: 'lst_anomaly_mean'); the dataset column "
    "itself is still never renamed by this analysis.",
    "Added reporting provenance fields (input dataset / metric source / "
    "bootstrap source / OOF predictions paths and SHA-256 hashes, git "
    "commit, created_at, and report_regeneration_only/models_refit/"
    "bootstrap_rerun flags) so a report-only regeneration is distinguishable "
    "from a full fit+bootstrap run in the output artifacts themselves.",
]

# The previously-supplied small-block reference figures for
# burnable_tree_shrub_grass. NEVER read as a scientific input anywhere in
# this module -- used ONLY to compute a diagnostic `reference_metric_mismatch`
# flag against the real, on-disk small-block Step8B/Step8C artifacts, which
# remain the sole scientific source of truth regardless of this flag's value.
EXPECTED_SMALL_BLOCK_REFERENCE = {
    "population": "burnable_tree_shrub_grass",
    "baseline_roc_auc": 0.7398,
    "thermal_roc_auc": 0.8629,
    "delta_roc_auc": 0.1231,
    "delta_roc_auc_ci": [0.1138, 0.1330],
    "delta_pr_auc": 0.2325,
    "delta_pr_auc_ci": [0.2102, 0.2555],
}
REFERENCE_TOLERANCE = 0.005

# lst_anomaly_mean semantic provenance (see docs/seam_audit.md and the Step9G
# semantic_mismatch block for the same note elsewhere in the pipeline).
LST_ANOMALY_SEMANTIC_NOTE = {
    "column_name": "lst_anomaly_mean",
    "semantic_identity": "mean_standardized_lst_anomaly",
    "source_product": "anomaly_zscore",
    "semantic_name_mismatch": True,
    "note": (
        "lst_anomaly_mean is a mean of a standardized (z-scored) LST anomaly "
        "raster, not an absolute-Celsius anomaly. The column name is kept "
        "unchanged in this patch; only this provenance note is added."
    ),
}


class Step8BigBlockRobustnessError(SystemExit):
    """Fail-fast error for the single-experiment big-block robustness analysis."""


# =============================================================================
# Paths / namespacing
# =============================================================================
def experiment_output_root(experiment_id: str) -> Path:
    return (
        PROJECT_ROOT / "outputs" / "experiments" / experiment_id
        / "robustness" / "step8_big_blocks"
    )


def _condition_output_dir(experiment_id: str, block_size_cells: int, output_root: Path | None = None) -> Path:
    output_root = experiment_output_root(experiment_id) if output_root is None else output_root
    return output_root / f"block_{int(block_size_cells)}_cells"


def protected_paths_for_experiment(experiment_id: str) -> dict[str, Path]:
    root = experiment_step8_root(experiment_id)
    return {relative: root / relative for relative in PROTECTED_RELATIVE_PATHS}


def nominal_scale_label(block_size_cells: int) -> str:
    km = int(block_size_cells) * 0.5
    return f"approximately_{km:g}_km"


def validate_block_sizes(block_sizes: list[int] | tuple[int, ...]) -> None:
    if not block_sizes:
        raise Step8BigBlockRobustnessError("At least one block size must be provided.")
    for value in block_sizes:
        if int(value) != value or int(value) <= 0:
            raise Step8BigBlockRobustnessError(f"Block size must be a positive integer: {value}")
        if int(value) <= STEP8B_SPATIAL_BLOCK_SIZE_CELLS:
            raise Step8BigBlockRobustnessError(
                f"Block size {value} must be larger than the existing small-block "
                f"reference size ({STEP8B_SPATIAL_BLOCK_SIZE_CELLS} cells)."
            )


# =============================================================================
# Protection: original Step8A/B/C/E outputs for this experiment (read-only)
# =============================================================================
def hash_all_protected(experiment_id: str) -> dict[str, Any]:
    return hash_protected_inputs(protected_paths_for_experiment(experiment_id))


def assert_all_protected_unchanged(before: dict[str, Any], after: dict[str, Any]) -> None:
    from src.step8_large_block_robustness import assert_protected_hashes_unchanged
    try:
        assert_protected_hashes_unchanged(before, after)
    except SystemExit as exc:
        raise Step8BigBlockRobustnessError(f"Original Step8 protection failed: {exc}") from exc


# =============================================================================
# Canonical grid reference
# =============================================================================
def canonical_grid_reference(experiment_id: str) -> dict[str, Any]:
    root = experiment_step8_root(experiment_id)
    grid_path = root / "step8a" / "step8a_500m_grid_valid_mask.tif"
    if not grid_path.is_file():
        raise Step8BigBlockRobustnessError(f"{experiment_id}: canonical 500m grid raster not found: {grid_path}")
    info: dict[str, Any] = {"path": str(grid_path), "sha256": sha256_file(grid_path)}
    try:
        import rasterio
        with rasterio.open(grid_path) as handle:
            info["width"] = handle.width
            info["height"] = handle.height
            info["crs"] = str(handle.crs)
    except Exception:  # noqa: BLE001
        pass
    return info


# =============================================================================
# Original (small, 2-cell) Step8B/Step8C reference for this experiment
# =============================================================================
def original_small_block_reference(experiment_id: str) -> dict[str, Any]:
    root = experiment_step8_root(experiment_id)
    metrics = json.loads((root / "step8b" / "step8b_model_comparison_metrics.json").read_text())
    bootstrap = json.loads((root / "step8c" / "step8c_bootstrap_metrics.json").read_text())
    spatial = metrics.get("spatial_cv_config", {})
    if spatial.get("spatial_block_size_cells") != STEP8B_SPATIAL_BLOCK_SIZE_CELLS:
        raise Step8BigBlockRobustnessError(
            f"{experiment_id}: frozen original Step8 block size is not proven to be "
            f"{STEP8B_SPATIAL_BLOCK_SIZE_CELLS} cells."
        )
    if (
        spatial.get("method") != "StratifiedGroupKFold"
        or spatial.get("n_splits_requested") != STEP8B_N_SPLITS
        or spatial.get("random_state") != STEP8B_RANDOM_SEED
        or spatial.get("random_split_used") is not False
    ):
        raise Step8BigBlockRobustnessError(
            f"{experiment_id}: frozen original Step8 CV provenance disagrees with runtime configuration."
        )
    feature_sets = metrics.get("feature_sets", {})
    if (
        metrics.get("model") != MODEL_NAME
        or feature_sets.get("baseline") != BASELINE_FEATURES
        or feature_sets.get("thermal_additional") != THERMAL_FEATURES
        or feature_sets.get("thermal_model_full") != THERMAL_MODEL_FEATURES
    ):
        raise Step8BigBlockRobustnessError(
            f"{experiment_id}: frozen Step8B model/feature provenance disagrees with runtime configuration."
        )
    if (
        bootstrap.get("n_bootstrap_requested") != STEP8C_N_BOOTSTRAP
        or bootstrap.get("random_seed") != STEP8C_RANDOM_SEED
    ):
        raise Step8BigBlockRobustnessError(
            f"{experiment_id}: frozen Step8C bootstrap provenance disagrees with runtime configuration."
        )
    point = metrics.get("population_metrics", {}).get(PRIMARY_POPULATION)
    ci = bootstrap.get("bootstrap_ci_by_population", {}).get(PRIMARY_POPULATION)
    if not point or not ci:
        raise Step8BigBlockRobustnessError(
            f"{experiment_id}: primary-population ('{PRIMARY_POPULATION}') Step8B/Step8C reference is missing."
        )

    delta_roc_ci = ci["delta_auc_ci95"]
    delta_pr_ci = ci["delta_pr_auc_ci95"]
    delta_brier_ci = ci.get("delta_brier_ci95")  # legacy convention; may be absent in older artifacts
    actual = {
        "baseline_roc_auc": point["overall_baseline"]["roc_auc"],
        "thermal_roc_auc": point["overall_thermal"]["roc_auc"],
        "delta_roc_auc": point["delta_auc"],
        "delta_roc_auc_ci": list(delta_roc_ci),
        "delta_pr_auc": point["delta_pr_auc"],
        "delta_pr_auc_ci": list(delta_pr_ci),
        # informational only -- EXPECTED_SMALL_BLOCK_REFERENCE never had Brier
        # fields, so these are never part of the mismatch check below. Brier
        # is OPTIONAL in the frozen original artifact (older Step8B/Step8C
        # runs may not have recorded it); never fabricated if absent.
        "baseline_brier": point.get("overall_baseline", {}).get("brier_score"),
        "thermal_brier": point.get("overall_thermal", {}).get("brier_score"),
        "delta_brier": point.get("delta_brier"),
        "delta_brier_ci": list(delta_brier_ci) if delta_brier_ci else None,
    }
    mismatches = []
    for key in ("baseline_roc_auc", "thermal_roc_auc", "delta_roc_auc", "delta_pr_auc"):
        expected_value = EXPECTED_SMALL_BLOCK_REFERENCE[key]
        if abs(actual[key] - expected_value) > REFERENCE_TOLERANCE:
            mismatches.append(key)
    for key in ("delta_roc_auc_ci", "delta_pr_auc_ci"):
        expected_pair = EXPECTED_SMALL_BLOCK_REFERENCE[key]
        actual_pair = actual[key]
        if any(abs(a - e) > REFERENCE_TOLERANCE for a, e in zip(actual_pair, expected_pair)):
            mismatches.append(key)

    return {
        "metrics": metrics,
        "bootstrap": bootstrap,
        "point": point,
        "ci": ci,
        "reference_check": {
            "expected": EXPECTED_SMALL_BLOCK_REFERENCE,
            "actual": actual,
            "tolerance": REFERENCE_TOLERANCE,
            "reference_metric_mismatch": bool(mismatches),
            "mismatched_fields": mismatches,
            "note": (
                "reference_metric_mismatch flags a diagnostic discrepancy "
                "against a previously supplied (incorrect) reference figure "
                "for burnable_tree_shrub_grass. It does NOT invalidate the "
                "frozen Step8B/Step8C disk artifacts read above, which "
                "remain the sole scientific source of truth for this "
                "analysis regardless of this flag's value."
            ),
        },
    }


def frozen_step8b_predictions(experiment_id: str) -> Path:
    return experiment_step8_root(experiment_id) / "step8b" / "step8b_predictions.parquet"


# =============================================================================
# Reporting provenance (never fabricated: only paths/hashes that resolve to
# an existing file are populated; everything else is left null)
# =============================================================================
def reporting_provenance(
    experiment_id: str, block_size_cells: int | None = None, output_root: Path | None = None,
) -> dict[str, Any]:
    root = experiment_step8_root(experiment_id)
    input_path = root / "step8a" / "step8a_500m_modeling_dataset.parquet"
    metric_path = root / "step8b" / "step8b_model_comparison_metrics.json"
    bootstrap_path = root / "step8c" / "step8c_bootstrap_metrics.json"
    provenance: dict[str, Any] = {
        "input_dataset_path": str(input_path) if input_path.is_file() else None,
        "input_dataset_sha256": sha256_file(input_path) if input_path.is_file() else None,
        "metric_source_path": str(metric_path) if metric_path.is_file() else None,
        "metric_source_sha256": sha256_file(metric_path) if metric_path.is_file() else None,
        "bootstrap_source_path": str(bootstrap_path) if bootstrap_path.is_file() else None,
        "bootstrap_source_sha256": sha256_file(bootstrap_path) if bootstrap_path.is_file() else None,
        "oof_predictions_path": None,
        "oof_predictions_sha256": None,
        "git_commit": _git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if block_size_cells is not None:
        resolved_root = experiment_output_root(experiment_id) if output_root is None else output_root
        oof_path = _condition_output_dir(experiment_id, block_size_cells, resolved_root) / "oof_predictions.parquet"
        if oof_path.is_file():
            provenance["oof_predictions_path"] = str(oof_path)
            provenance["oof_predictions_sha256"] = sha256_file(oof_path)
    return provenance


# =============================================================================
# Decision-rule vocabulary (this analysis's own; NOT v1/v2's classify_interval
# / classify_joint vocabulary, which is a different, already-used string set
# for a different frozen analysis family).
# =============================================================================
def classify_metric_support(ci_low: float | None, ci_high: float | None) -> str:
    if ci_low is None or ci_high is None:
        return "unavailable"
    if ci_low > 0:
        return "supported_positive"
    if ci_high < 0:
        return "supported_negative"
    return "uncertain"


def classify_brier_improvement_support(ci_low: float | None, ci_high: float | None) -> str:
    """Canonical Brier decision rule. ci_low/ci_high are bounds of
    brier_improvement = baseline_brier - thermal_brier (POSITIVE = thermal
    model has the lower/better Brier score)."""
    if ci_low is None or ci_high is None:
        return "unavailable"
    if ci_low > 0:
        return "supported_improvement"
    if ci_high < 0:
        return "supported_degradation"
    return "uncertain"


def classify_brier_support(ci_low: float | None, ci_high: float | None) -> str:
    """Deprecated (v1 reporting schema). Takes the LEGACY delta_brier CI
    (thermal_brier - baseline_brier, negative-is-better) and returns the
    generic supported_positive/supported_negative/uncertain vocabulary. No
    function in this module calls this as of the v2 reporting schema; use
    classify_brier_improvement_support (canonical, positive-is-better,
    supported_improvement/uncertain/supported_degradation vocabulary)
    instead. Retained only for readers of v1-schema reports."""
    if ci_low is None or ci_high is None:
        return "unavailable"
    return classify_metric_support(-ci_high, -ci_low)


def brier_improvement_definition() -> str:
    return (
        "brier_improvement = baseline_brier - thermal_brier; POSITIVE values "
        "mean the thermal model has the lower/better Brier score. The "
        "deprecated `delta_brier` field (thermal_brier - baseline_brier) uses "
        "the OPPOSITE sign convention (negative = thermal better) and is kept "
        "only for backward compatibility with v1-schema reports."
    )


def brier_improvement_point_fields(legacy_delta_brier: float | None) -> dict[str, Any]:
    """Point-estimate (no CI) canonical Brier fields, derived from the
    already-computed legacy delta_brier point value -- no new computation."""
    brier_improvement = -legacy_delta_brier if legacy_delta_brier is not None else None
    return {
        "delta_brier": legacy_delta_brier,
        "brier_improvement": brier_improvement,
        "brier_improvement_definition": brier_improvement_definition(),
    }


def brier_improvement_ci_fields(legacy_ci_low: float | None, legacy_ci_high: float | None) -> dict[str, Any]:
    """CI-level canonical Brier fields, derived from the already-computed
    legacy delta_brier CI bounds by sign-flip (and bound-swap) -- no new
    bootstrap computation."""
    ci_low = -legacy_ci_high if legacy_ci_high is not None else None
    ci_high = -legacy_ci_low if legacy_ci_low is not None else None
    return {
        "brier_improvement_ci_low": ci_low,
        "brier_improvement_ci_high": ci_high,
        "brier_improvement_definition": brier_improvement_definition(),
        "brier_support_status": classify_brier_improvement_support(ci_low, ci_high),
    }


def relative_reduction(previous: float | None, current: float | None) -> float | None:
    """Fractional reduction from `previous` to `current`, relative to
    abs(previous). POSITIVE means `current` is SMALLER than `previous` (an
    actual reduction in magnitude); NEGATIVE means `current` is LARGER (an
    increase). Returns None if either value is missing or `previous` is 0
    -- never fabricated."""
    if previous is None or current is None or previous == 0:
        return None
    return (previous - current) / abs(previous)


def compute_effect_magnitude_details(
    delta_roc_small: float | None, delta_roc_10: float | None, delta_roc_20: float | None,
    delta_pr_small: float | None, delta_pr_10: float | None, delta_pr_20: float | None,
) -> dict[str, Any]:
    """Metric-specific relative-reduction fields for delta_roc_auc and
    delta_pr_auc across the small (2-cell) reference and the 10-/20-cell
    conditions. Every value is computed directly from the frozen small-block
    and big-block artifacts' own delta_roc_auc/delta_pr_auc point estimates
    -- never hard-coded. `delta_roc_10`/`delta_roc_20`/`delta_pr_10`/
    `delta_pr_20` are looked up by the literal block sizes 10 and 20 (this
    analysis's predefined scales); if a run used different block sizes, the
    corresponding fields are None rather than fabricated."""
    return {
        "delta_roc_auc_relative_reduction_small_to_10": relative_reduction(delta_roc_small, delta_roc_10),
        "delta_roc_auc_relative_reduction_small_to_20": relative_reduction(delta_roc_small, delta_roc_20),
        "delta_roc_auc_relative_reduction_10_to_20": relative_reduction(delta_roc_10, delta_roc_20),
        "delta_pr_auc_relative_reduction_small_to_10": relative_reduction(delta_pr_small, delta_pr_10),
        "delta_pr_auc_relative_reduction_small_to_20": relative_reduction(delta_pr_small, delta_pr_20),
        "delta_pr_auc_relative_reduction_10_to_20": relative_reduction(delta_pr_10, delta_pr_20),
        "stable_band": EFFECT_MAGNITUDE_STABLE_BAND,
    }


def classify_effect_magnitude_stability(details: dict[str, Any]) -> str:
    """Classifies whether the SIZE of the paired delta_roc_auc effect grows,
    shrinks, stays stable, or moves non-monotonically as the spatial block
    scale increases from the small (2-cell) reference through the 10- and
    20-cell conditions, based on delta_roc_auc's relative reductions. This is
    independent of, and must not be conflated with, whether bootstrap
    SUPPORT for a positive effect is retained (see classify_final_robustness
    / support_robustness_status)."""
    reduction_small_to_10 = details["delta_roc_auc_relative_reduction_small_to_10"]
    reduction_10_to_20 = details["delta_roc_auc_relative_reduction_10_to_20"]
    if reduction_small_to_10 is None or reduction_10_to_20 is None:
        return "unavailable"
    if reduction_small_to_10 >= EFFECT_MAGNITUDE_STABLE_BAND and reduction_10_to_20 >= EFFECT_MAGNITUDE_STABLE_BAND:
        return "decreases_with_block_scale"
    if reduction_small_to_10 <= -EFFECT_MAGNITUDE_STABLE_BAND and reduction_10_to_20 <= -EFFECT_MAGNITUDE_STABLE_BAND:
        return "increases_with_block_scale"
    if abs(reduction_small_to_10) < EFFECT_MAGNITUDE_STABLE_BAND and abs(reduction_10_to_20) < EFFECT_MAGNITUDE_STABLE_BAND:
        return "stable_across_block_scale"
    return "non_monotonic"


def classify_overall_support(roc_status: str, pr_status: str) -> str:
    supported = [roc_status == "supported_positive", pr_status == "supported_positive"]
    if all(supported):
        return "retained"
    if any(supported):
        return "partially_retained"
    return "not_retained"


_FINAL_ROBUSTNESS_TABLE = {
    ("retained", "retained"): "strongly_robust",
    ("retained", "partially_retained"): "moderately_robust",
    ("partially_retained", "retained"): "moderately_robust",
    ("partially_retained", "partially_retained"): "moderately_robust",
    ("retained", "not_retained"): "scale_sensitive",
    ("not_retained", "retained"): "scale_sensitive",
    ("partially_retained", "not_retained"): "scale_sensitive",
    ("not_retained", "partially_retained"): "scale_sensitive",
    ("not_retained", "not_retained"): "not_robust",
}


def classify_final_robustness(status_10: str, status_20: str) -> str:
    return _FINAL_ROBUSTNESS_TABLE.get((status_10, status_20), "not_robust")


# =============================================================================
# 10/20-cell (or arbitrary block-size) conditions
# =============================================================================
def run_big_block_condition(
    df_all: pd.DataFrame, experiment_id: str, block_size_cells: int, analysis_id: str,
) -> dict[str, Any]:
    validate_canonical_grid(df_all)
    assigned = add_spatial_block_id(
        df_all, block_size_cells,
        column_name=BLOCK_COLUMN, id_prefix=f"block{int(block_size_cells)}", include_row_col=True,
    )  # before valid/population filtering
    df_valid = filter_valid_for_modeling(assigned)
    masks = build_population_masks(df_valid)
    df_pop = df_valid.loc[masks[PRIMARY_POPULATION]].reset_index(drop=True)
    if df_pop.empty:
        return {
            "status": "infeasible_with_existing_cv_protocol",
            "reason": f"{experiment_id}: primary population '{PRIMARY_POPULATION}' is empty at block_size={block_size_cells}.",
        }

    try:
        result = train_population(
            df_pop, PRIMARY_POPULATION, STEP8B_N_SPLITS, STEP8B_RANDOM_SEED, MODEL_NAME,
            STEP8B_MIN_POSITIVES_PER_POPULATION,
            group_column=BLOCK_COLUMN, strict_folds=True,
        )
    except Step8BError as exc:
        return {"status": "infeasible_with_existing_cv_protocol", "reason": str(exc)}

    if result is None or result.get("skipped"):
        reason = result.get("reason") if result else "no result"
        return {"status": "infeasible_with_existing_cv_protocol", "reason": reason}

    y = df_pop[TARGET_COLUMN].astype(int).to_numpy()
    predictions = pd.DataFrame({
        "experiment_id": experiment_id,
        "block_size_cells": int(block_size_cells),
        "population": PRIMARY_POPULATION,
        "cell_id": df_pop["cell_id"].to_numpy(),
        "row_500m": df_pop["row_500m"].to_numpy(),
        "col_500m": df_pop["col_500m"].to_numpy(),
        "spatial_block_id": df_pop[BLOCK_COLUMN].to_numpy(),
        "fold_id": result["fold_id"],
        "burned": y,
        "baseline_probability": result["oof_prob_baseline"],
        "thermal_probability": result["oof_prob_thermal"],
        "valid_for_evaluation": True,
        "landcover_dominant": df_pop["landcover_dominant"].to_numpy(),
        "burnable_tree_shrub_grass": df_pop["burnable_tree_shrub_grass"].to_numpy(),
    })

    metrics = {
        "baseline_roc_auc": result["overall_baseline"]["roc_auc"],
        "thermal_roc_auc": result["overall_thermal"]["roc_auc"],
        "delta_roc_auc": result["delta_auc"],
        "baseline_pr_auc": result["overall_baseline"]["pr_auc"],
        "thermal_pr_auc": result["overall_thermal"]["pr_auc"],
        "delta_pr_auc": result["delta_pr_auc"],
        "baseline_brier": result["overall_baseline"]["brier_score"],
        "thermal_brier": result["overall_thermal"]["brier_score"],
        **brier_improvement_point_fields(result["delta_brier"]),
    }

    fold_diagnostics = build_fold_diagnostics(df_pop, result, block_column=BLOCK_COLUMN)
    block_audit = build_block_audit(df_pop, y, fold_diagnostics, block_column=BLOCK_COLUMN)

    return {
        "status": "fitted",
        "metrics": metrics,
        "predictions": predictions,
        "block_audit": block_audit,
        "fold_diagnostics": fold_diagnostics,
        "n_splits_used": result["n_splits_used"],
    }


def build_fold_diagnostics(df_pop: pd.DataFrame, result: dict[str, Any], block_column: str) -> list[dict[str, Any]]:
    """Per-fold train/test row/block/class counts and per-fold ROC/PR/Brier
    (baseline vs thermal), reusing fold_rows already computed inside
    train_population (test-side metrics) and deriving train-side counts and
    the leakage check from the fold_id assignment array. Does not recompute
    any metric already produced by train_population."""
    fold_id = result["fold_id"]
    y = df_pop[TARGET_COLUMN].astype(int).to_numpy()
    blocks = df_pop[block_column].to_numpy()
    diagnostics = []
    for fr in result["fold_rows"]:
        f = fr["fold"]
        test_mask = fold_id == f
        train_mask = (fold_id != f) & (fold_id >= 0)
        train_blocks = set(blocks[train_mask])
        test_blocks = set(blocks[test_mask])
        auc_b, auc_t = fr["auc_baseline"], fr["auc_thermal"]
        pr_b, pr_t = fr["pr_auc_baseline"], fr["pr_auc_thermal"]
        br_b, br_t = fr["brier_baseline"], fr["brier_thermal"]
        diagnostics.append({
            "fold_id": f,
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "train_positive": int(y[train_mask].sum()),
            "test_positive": fr["test_positives"],
            "train_negative": int((y[train_mask] == 0).sum()),
            "test_negative": fr["test_negatives"],
            "train_blocks": len(train_blocks),
            "test_blocks": len(test_blocks),
            "block_overlap": len(train_blocks & test_blocks),
            "baseline_roc_auc": auc_b, "thermal_roc_auc": auc_t,
            "delta_roc_auc": (auc_t - auc_b) if auc_b is not None and auc_t is not None else None,
            "baseline_pr_auc": pr_b, "thermal_pr_auc": pr_t,
            "delta_pr_auc": (pr_t - pr_b) if pr_b is not None and pr_t is not None else None,
            "baseline_brier": br_b, "thermal_brier": br_t,
            "delta_brier": (br_t - br_b) if br_b is not None and br_t is not None else None,
        })
    return diagnostics


def build_block_audit(df_pop: pd.DataFrame, y: np.ndarray, fold_diagnostics: list[dict[str, Any]], block_column: str) -> dict[str, Any]:
    grouped = df_pop.assign(_burned=y).groupby(block_column)["_burned"].agg(["size", "sum"])
    positives = grouped["sum"]
    total_overlap = sum(fr["block_overlap"] for fr in fold_diagnostics)
    return {
        "total_rows": int(len(df_pop)),
        "eligible_rows": int(len(df_pop)),
        "positive_rows": int(y.sum()),
        "negative_rows": int((y == 0).sum()),
        "unique_spatial_blocks": int(len(grouped)),
        "positive_containing_blocks": int((positives > 0).sum()),
        "negative_containing_blocks": int((positives < grouped["size"]).sum()),
        "mixed_class_blocks": int(((positives > 0) & (positives < grouped["size"])).sum()),
        "fold_count": len(fold_diagnostics),
        "train_test_block_leakage_total": int(total_overlap),
        "train_test_block_leakage_free": total_overlap == 0,
        "folds": fold_diagnostics,
    }


# =============================================================================
# Paired bootstrap (this analysis's own -- keeps Brier, unlike v1's helper)
# =============================================================================
def paired_big_block_bootstrap(
    predictions: pd.DataFrame, n_replicates: int = STEP8C_N_BOOTSTRAP, seed: int = STEP8C_RANDOM_SEED,
) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {"spatial_block_id", "burned", "baseline_probability", "thermal_probability"}
    if not required.issubset(predictions.columns):
        raise Step8BigBlockRobustnessError(f"Bootstrap predictions missing columns: {sorted(required - set(predictions.columns))}")

    blocks = predictions["spatial_block_id"].drop_duplicates().to_numpy()
    block_values = predictions["spatial_block_id"].to_numpy()
    indices = {block: np.flatnonzero(block_values == block) for block in blocks}
    rng = np.random.default_rng(seed)

    metric_names = (
        "auc_baseline", "auc_thermal", "delta_auc",
        "pr_auc_baseline", "pr_auc_thermal", "delta_pr_auc",
        "brier_baseline", "brier_thermal", "delta_brier",
    )
    rows: list[dict[str, Any]] = []
    successful: list[dict[str, float]] = []
    for replicate in range(n_replicates):
        sampled = rng.choice(blocks, size=len(blocks), replace=True)
        row_idx = np.concatenate([indices[block] for block in sampled])
        y = predictions.iloc[row_idx]["burned"].to_numpy(dtype=int)
        metrics = compute_paired_metrics(
            y,
            predictions.iloc[row_idx]["baseline_probability"].to_numpy(),
            predictions.iloc[row_idx]["thermal_probability"].to_numpy(),
        )
        base = {"replicate": replicate, "sampled_block_ids_sha256": sha256_bytes("|".join(map(str, sampled)).encode())}
        if metrics is None:
            rows.append({**base, "valid": False, "invalid_reason": "single_class"})
        else:
            kept = {name: metrics[name] for name in metric_names}
            successful.append(kept)
            rows.append({**base, "valid": True, "invalid_reason": None, **kept})

    valid = len(successful)
    summary: dict[str, Any] = {
        "requested_replicates": n_replicates,
        "valid_replicates": valid,
        "invalid_single_class_replicates": n_replicates - valid,
        "invalid_other_replicates": 0,
        "valid_fraction": (valid / n_replicates) if n_replicates else 0.0,
        "bootstrap_stability": "stable" if valid >= MIN_VALID_BOOTSTRAP else "bootstrap_unstable",
        "random_seed": seed,
        "ci_method": f"{STEP8C_CI_LOWER}/{STEP8C_CI_UPPER} percentile",
        "bootstrap_unit": "spatial_block_id (the big block for this analysis)",
        "series": {},
    }
    for name in metric_names:
        values = np.array([item[name] for item in successful], dtype=float)
        summary["series"][name] = {
            "mean": float(np.mean(values)) if valid else None,
            "median": float(np.median(values)) if valid else None,
            "ci_2_5": float(np.percentile(values, STEP8C_CI_LOWER)) if valid else None,
            "ci_97_5": float(np.percentile(values, STEP8C_CI_UPPER)) if valid else None,
        }

    # Canonical, positive-is-better Brier series, derived by sign-flip (and
    # bound-swap) from the legacy delta_brier series above -- NOT a separate
    # bootstrap computation.
    summary["series"]["brier_improvement"] = derive_brier_improvement_series(summary["series"]["delta_brier"])
    return summary, pd.DataFrame(rows)


def derive_brier_improvement_series(legacy_series: dict[str, float | None]) -> dict[str, float | None]:
    """Sign-flips (and swaps the CI bounds of) an already-computed legacy
    delta_brier bootstrap series (thermal-baseline, negative=better) into the
    canonical brier_improvement series (baseline-thermal, positive=better).
    No new bootstrap sampling is performed."""
    return {
        "mean": -legacy_series["mean"] if legacy_series["mean"] is not None else None,
        "median": -legacy_series["median"] if legacy_series["median"] is not None else None,
        "ci_2_5": -legacy_series["ci_97_5"] if legacy_series["ci_97_5"] is not None else None,
        "ci_97_5": -legacy_series["ci_2_5"] if legacy_series["ci_2_5"] is not None else None,
    }


# =============================================================================
# Manifest / preregistration
# =============================================================================
def scientific_configuration(experiment_id: str, block_sizes: list[int], protected: dict[str, Any]) -> dict[str, Any]:
    reference = original_small_block_reference(experiment_id)
    classifier = build_classifier(MODEL_NAME, STEP8B_RANDOM_SEED)
    model_params = classifier.get_params(deep=False)
    grid = canonical_grid_reference(experiment_id)
    return {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "migration_notes_v1_to_v2": list(MIGRATION_NOTES_V1_TO_V2),
        "reporting_provenance": reporting_provenance(experiment_id),
        "experiment_id": experiment_id,
        "primary_population": PRIMARY_POPULATION,
        "block_sizes_cells": [int(b) for b in block_sizes],
        "nominal_scales": {str(b): nominal_scale_label(b) for b in block_sizes},
        "fixed_grid_origin": {"row_500m": 0, "col_500m": 0},
        "block_construction": (
            "block_row=floor(row_500m/block_size_cells); "
            "block_col=floor(col_500m/block_size_cells); "
            "spatial_block_id=block{size}_{block_row}_{block_col} "
            "(via step8b.add_spatial_block_id, unmodified)"
        ),
        "original_small_block_reference_size_cells": STEP8B_SPATIAL_BLOCK_SIZE_CELLS,
        "canonical_grid": grid,
        "baseline_features": list(BASELINE_FEATURES),
        "thermal_additional_features": list(THERMAL_FEATURES),
        "thermal_model_features": list(THERMAL_MODEL_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target_column": TARGET_COLUMN,
        "lst_anomaly_mean_provenance": LST_ANOMALY_SEMANTIC_NOTE,
        "model_class": type(classifier).__name__,
        "model_name": MODEL_NAME,
        "model_hyperparameters": model_params,
        "preprocessing": {
            "numeric": "SimpleImputer(strategy=median)",
            "categorical": "SimpleImputer(strategy=most_frequent) + OneHotEncoder(handle_unknown=ignore)",
            "fit_scope": "inside each training fold only",
        },
        "cv": {
            "class": "StratifiedGroupKFold", "n_splits": STEP8B_N_SPLITS,
            "shuffle": True, "random_state": STEP8B_RANDOM_SEED,
            "dynamic_fold_reduction": False, "strict_folds": True,
        },
        "bootstrap": {
            "requested_replicates": STEP8C_N_BOOTSTRAP, "random_state": STEP8C_RANDOM_SEED,
            "ci_method": f"{STEP8C_CI_LOWER}/{STEP8C_CI_UPPER} percentile",
            "unit": "corresponding big spatial block", "paired": True,
            "minimum_valid_replicates": MIN_VALID_BOOTSTRAP,
        },
        "primary_estimands": [
            "delta_roc_auc = thermal_oof_roc_auc - baseline_oof_roc_auc",
            "delta_pr_auc = thermal_oof_pr_auc - baseline_oof_pr_auc",
            "brier_improvement = baseline_oof_brier - thermal_oof_brier "
            "(canonical; POSITIVE means the thermal model has the lower/better "
            "Brier score); the deprecated delta_brier field uses the opposite "
            "sign convention (thermal - baseline, negative-is-better) and is "
            "retained only for backward compatibility.",
        ],
        "interpretation_rules": {
            "per_metric": {
                "supported_positive": "ROC-AUC/PR-AUC percentile CI entirely above zero",
                "uncertain": "percentile CI includes zero",
                "supported_negative": "ROC-AUC/PR-AUC percentile CI entirely below zero",
            },
            "brier_per_metric": {
                "supported_improvement": "brier_improvement percentile CI entirely above zero",
                "uncertain": "brier_improvement percentile CI includes zero",
                "supported_degradation": "brier_improvement percentile CI entirely below zero",
            },
            "per_block_size_overall": {
                "retained": "both ROC-AUC and PR-AUC delta CIs are supported_positive",
                "partially_retained": "exactly one of ROC-AUC/PR-AUC delta CIs is supported_positive",
                "not_retained": "neither ROC-AUC nor PR-AUC delta CI is supported_positive",
            },
            "support_robustness_status_across_block_sizes": {
                "strongly_robust": "both predefined block sizes are retained",
                "moderately_robust": "one retained and the other partially_retained (or both partially_retained)",
                "scale_sensitive": "support present at one block size and not_retained at the other",
                "not_robust": "neither block size is retained or partially_retained",
            },
            "effect_magnitude_stability_status": {
                "decreases_with_block_scale": "delta_roc_auc_relative_reduction is >= +5% (an actual "
                "reduction) at both the small-to-10-cell and 10-to-20-cell steps",
                "increases_with_block_scale": "delta_roc_auc_relative_reduction is <= -5% (an increase) "
                "at both steps",
                "stable_across_block_scale": "delta_roc_auc_relative_reduction is within +/-5% at both steps",
                "non_monotonic": "the two relative reductions disagree in sign or magnitude band",
                "unavailable": "a required delta_roc_auc value is missing",
                "field_semantics": (
                    "relative_reduction = (previous - current) / abs(previous); POSITIVE means "
                    "the later block scale's delta is SMALLER (an actual reduction), NEGATIVE "
                    "means it is LARGER (an increase). Reported per metric "
                    "(delta_roc_auc_relative_reduction_*, delta_pr_auc_relative_reduction_*) and "
                    "per scale pair (small_to_10, small_to_20, 10_to_20)."
                ),
                "note": (
                    "This status answers a DIFFERENT question than "
                    "support_robustness_status: whether the SIZE of the "
                    "thermal contribution grows/shrinks/stays stable, not "
                    "whether bootstrap support for a positive contribution "
                    "is retained. The two must be reported separately."
                ),
            },
            "no_significance_language": (
                "this analysis never uses the phrase 'statistically significant'; "
                "wording is restricted to bootstrap interval support/uncertain/crossing zero"
            ),
        },
        "reference_check": reference["reference_check"],
        "protected_original_inputs": protected,
        "package_versions": _package_versions(),
        "prohibited_actions": [
            "modify_step8a_modeling_dataset", "regenerate_predictors",
            "rerun_step5_step5c_step7", "regenerate_step8a",
            "overwrite_existing_step8b_step8c_outputs", "add_new_model_type",
            "change_rf_hyperparameters", "feature_subset_tuning",
            "drop_features_based_on_importance", "change_population_definition",
            "random_row_split", "cell_level_bootstrap",
            "model_calibration_or_threshold_optimization",
            "seam_analysis_or_seam_correction", "rename_lst_anomaly_mean_column",
            "run_manavgat_bejis_kozan_or_transfer_experiments",
            "post_hoc_block_selection", "hyperparameter_tuning_after_seeing_results",
        ],
    }


def build_manifest(experiment_id: str, block_sizes: list[int], protected: dict[str, Any]) -> dict[str, Any]:
    content = scientific_configuration(experiment_id, block_sizes, protected)
    analysis_id = sha256_bytes(canonical_json(content).encode("utf-8"))
    return {
        "analysis_id": analysis_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "scientific_configuration": content,
    }


def _preregistration_markdown(manifest: dict[str, Any]) -> str:
    config = manifest["scientific_configuration"]
    lines = [
        "# Step8 Big-Spatial-Block Robustness Preregistration (IMMUTABLE)",
        "",
        f"- analysis_id: {manifest['analysis_id']}",
        f"- created_at: {manifest['created_at']}",
        f"- experiment: {config['experiment_id']}",
        f"- predefined block sizes: {config['block_sizes_cells']}",
        f"- primary population: {config['primary_population']}",
        "",
        "Only spatial grouping scale changes relative to the existing small-block "
        "(2-cell) Step8B/Step8C result. No favorable scale will be selected post hoc.",
    ]
    return "\n".join(lines) + "\n"


def validate_or_write_manifest(output_root: Path, experiment_id: str, block_sizes: list[int], protected: dict[str, Any]) -> dict[str, Any]:
    comparison_dir = output_root / "comparison"
    path = comparison_dir / "manifest.json"
    expected_content = scientific_configuration(experiment_id, block_sizes, protected)
    expected_id = sha256_bytes(canonical_json(expected_content).encode("utf-8"))
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("analysis_id") != expected_id or existing.get("scientific_configuration") != expected_content:
            raise Step8BigBlockRobustnessError(
                "Existing immutable big-block preregistration disagrees with runtime scientific configuration."
            )
        return existing
    comparison_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(experiment_id, block_sizes, protected)
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return manifest


def assert_downstream_outputs_writable(output_root: Path, force: bool) -> None:
    if force or not output_root.exists():
        return
    immutable = {output_root / "comparison" / "manifest.json"}
    present = sorted(
        str(candidate) for candidate in output_root.rglob("*")
        if candidate.is_file() and candidate not in immutable
    )
    if present:
        raise Step8BigBlockRobustnessError(
            "Downstream big-block robustness outputs already exist; use --force to overwrite only those: "
            + ", ".join(present)
        )


# =============================================================================
# Writing per-block-size outputs
# =============================================================================
def write_condition_outputs(
    output_dir: Path, analysis_id: str, experiment_id: str, block_size_cells: int,
    result: dict[str, Any], bootstrap: dict[str, Any], replicates: pd.DataFrame,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "analysis_id": analysis_id, "experiment_id": experiment_id,
        "block_size_cells": int(block_size_cells), "nominal_scale": nominal_scale_label(block_size_cells),
        "primary_population": PRIMARY_POPULATION,
    }

    classifier = build_classifier(MODEL_NAME, STEP8B_RANDOM_SEED)
    metrics = {
        **common, **result["metrics"],
        "n_splits_used": result["n_splits_used"],
        "feature_sets": {"baseline": BASELINE_FEATURES, "thermal_additional": THERMAL_FEATURES, "thermal_model_full": THERMAL_MODEL_FEATURES},
        "model_class": type(classifier).__name__, "model_hyperparameters": classifier.get_params(deep=False),
        "cv": {"class": "StratifiedGroupKFold", "n_splits": STEP8B_N_SPLITS, "random_state": STEP8B_RANDOM_SEED},
    }
    (output_dir / "step8b_metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    (output_dir / "step8b_metrics.md").write_text(_condition_metrics_markdown(metrics), encoding="utf-8")

    result["predictions"].to_parquet(output_dir / "oof_predictions.parquet", index=False)

    fold_rows = pd.DataFrame(result["block_audit"]["folds"])
    for key, value in common.items():
        fold_rows[key] = value
    fold_rows.to_parquet(output_dir / "fold_assignments.parquet", index=False)

    boot_payload = {**common, **bootstrap}
    (output_dir / "bootstrap_summary.json").write_text(json.dumps(boot_payload, indent=2, default=str) + "\n")
    (output_dir / "bootstrap_summary.md").write_text(_condition_bootstrap_markdown(boot_payload), encoding="utf-8")
    for key, value in common.items():
        replicates[key] = value
    ordered = list(common) + [c for c in replicates.columns if c not in common]
    replicates[ordered].to_parquet(output_dir / "bootstrap_replicates.parquet", index=False)

    block_manifest = _build_block_manifest(
        output_dir, common, result["block_audit"], experiment_id,
        report_regeneration_only=False, models_refit=True, bootstrap_rerun=True,
    )
    (output_dir / "block_manifest.json").write_text(json.dumps(block_manifest, indent=2, default=str) + "\n")

    return {"metrics": metrics, "bootstrap": boot_payload, "block_audit": result["block_audit"]}


def _build_block_manifest(
    output_dir: Path, common: dict[str, Any], block_audit: dict[str, Any], experiment_id: str,
    report_regeneration_only: bool, models_refit: bool, bootstrap_rerun: bool,
) -> dict[str, Any]:
    root = experiment_step8_root(experiment_id)
    input_path = root / "step8a" / "step8a_500m_modeling_dataset.parquet"
    # This condition's OWN big-block-specific metric/bootstrap source files
    # (already written to output_dir before this manifest is built) -- NOT
    # the original small-block Step8B/Step8C files. The small-block files
    # are recorded separately below under original_small_block_provenance
    # so the two are never conflated.
    metric_path = output_dir / "step8b_metrics.json"
    bootstrap_path = output_dir / "bootstrap_summary.json"
    oof_path = output_dir / "oof_predictions.parquet"
    small_block_metric_path = root / "step8b" / "step8b_model_comparison_metrics.json"
    small_block_bootstrap_path = root / "step8c" / "step8c_bootstrap_metrics.json"
    return {
        **common,
        "block_audit": {k: v for k, v in block_audit.items() if k != "folds"},
        "protected_original_inputs": list(PROTECTED_RELATIVE_PATHS),
        "input_dataset_path": str(input_path) if input_path.is_file() else None,
        "input_dataset_sha256": sha256_file(input_path) if input_path.is_file() else None,
        "metric_source_path": str(metric_path) if metric_path.is_file() else None,
        "metric_source_sha256": sha256_file(metric_path) if metric_path.is_file() else None,
        "bootstrap_source_path": str(bootstrap_path) if bootstrap_path.is_file() else None,
        "bootstrap_source_sha256": sha256_file(bootstrap_path) if bootstrap_path.is_file() else None,
        "oof_predictions_path": str(oof_path) if oof_path.is_file() else None,
        "oof_predictions_sha256": sha256_file(oof_path) if oof_path.is_file() else None,
        "original_small_block_provenance": {
            "metric_source_path": str(small_block_metric_path) if small_block_metric_path.is_file() else None,
            "metric_source_sha256": sha256_file(small_block_metric_path) if small_block_metric_path.is_file() else None,
            "bootstrap_source_path": str(small_block_bootstrap_path) if small_block_bootstrap_path.is_file() else None,
            "bootstrap_source_sha256": sha256_file(small_block_bootstrap_path) if small_block_bootstrap_path.is_file() else None,
        },
        "git_commit": _git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report_regeneration_only": report_regeneration_only,
        "models_refit": models_refit,
        "bootstrap_rerun": bootstrap_rerun,
    }


def _condition_metrics_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        f"# Step8 Big-Block Metrics -- block_size_cells={metrics['block_size_cells']} ({metrics['nominal_scale']})",
        "",
        f"- experiment: {metrics['experiment_id']}",
        f"- primary population: {metrics['primary_population']}",
        f"- baseline ROC-AUC: {metrics['baseline_roc_auc']:.6f}",
        f"- thermal ROC-AUC: {metrics['thermal_roc_auc']:.6f}",
        f"- delta ROC-AUC: {metrics['delta_roc_auc']:.6f}",
        f"- baseline PR-AUC: {metrics['baseline_pr_auc']:.6f}",
        f"- thermal PR-AUC: {metrics['thermal_pr_auc']:.6f}",
        f"- delta PR-AUC: {metrics['delta_pr_auc']:.6f}",
        f"- baseline Brier: {metrics['baseline_brier']:.6f}",
        f"- thermal Brier: {metrics['thermal_brier']:.6f}",
        f"- brier_improvement (canonical; baseline-thermal, positive=thermal better): {metrics['brier_improvement']:.6f}",
        f"- delta Brier (deprecated legacy; thermal-baseline, negative=thermal better): {metrics['delta_brier']:.6f}",
    ]
    return "\n".join(lines) + "\n"


def _condition_bootstrap_markdown(payload: dict[str, Any]) -> str:
    brier_improvement = payload["series"].get("brier_improvement", {})
    lines = [
        f"# Step8 Big-Block Bootstrap -- block_size_cells={payload['block_size_cells']} ({payload['nominal_scale']})",
        "",
        f"- requested replicates: {payload['requested_replicates']}",
        f"- valid replicates: {payload['valid_replicates']}",
        f"- invalid (single-class) replicates: {payload['invalid_single_class_replicates']}",
        f"- bootstrap stability: {payload['bootstrap_stability']}",
        f"- brier_improvement CI (canonical; positive=thermal better): "
        f"[{brier_improvement.get('ci_2_5')}, {brier_improvement.get('ci_97_5')}]",
    ]
    return "\n".join(lines) + "\n"


# =============================================================================
# Report-only regeneration (reads frozen artifacts from a prior full fit;
# never fits models, builds folds, generates predictions, or samples
# bootstrap replicates; never opens a Parquet file for writing)
# =============================================================================
def load_condition_artifacts(output_dir: Path) -> dict[str, Any]:
    """Reads back a prior full run's per-condition JSON/Parquet artifacts and
    reconstructs the {"metrics":..., "bootstrap":..., "block_audit":...} shape
    needed to regenerate reports, without opening any Parquet file for
    writing. Fails clearly (never fabricates) if the prior run's artifacts
    are not present."""
    metrics_path = output_dir / "step8b_metrics.json"
    bootstrap_path = output_dir / "bootstrap_summary.json"
    manifest_path = output_dir / "block_manifest.json"
    fold_path = output_dir / "fold_assignments.parquet"
    missing = [str(p) for p in (metrics_path, bootstrap_path, manifest_path, fold_path) if not p.is_file()]
    if missing:
        raise Step8BigBlockRobustnessError(
            "Cannot regenerate reports: missing frozen artifact(s) from a "
            "prior full fit+bootstrap run: " + ", ".join(missing)
        )
    metrics = json.loads(metrics_path.read_text())
    bootstrap = json.loads(bootstrap_path.read_text())
    block_manifest = json.loads(manifest_path.read_text())
    common_keys = {"analysis_id", "experiment_id", "block_size_cells", "nominal_scale", "primary_population"}
    fold_frame = pd.read_parquet(fold_path)
    fold_columns = [c for c in fold_frame.columns if c not in common_keys]
    folds = fold_frame[fold_columns].to_dict(orient="records")
    block_audit = {**block_manifest["block_audit"], "folds": folds}
    return {"metrics": metrics, "bootstrap": bootstrap, "block_audit": block_audit}


def write_condition_reports_only(
    output_dir: Path, experiment_id: str, block_size_cells: int, analysis_id: str,
) -> dict[str, Any]:
    """Regenerates step8b_metrics.{json,md}, bootstrap_summary.{json,md}, and
    block_manifest.json for one already-fitted block-size condition, using
    ONLY the frozen JSON/Parquet artifacts already on disk. Never writes
    oof_predictions.parquet, fold_assignments.parquet, or
    bootstrap_replicates.parquet. `analysis_id` is the FROZEN analysis
    identity read from the existing top-level preregistration -- it is
    never recomputed here, and it overrides whatever value the loaded
    per-condition artifacts happen to carry, so a single analysis_id is
    used consistently everywhere in the regenerated reports."""
    loaded = load_condition_artifacts(output_dir)
    old_metrics, old_bootstrap = loaded["metrics"], loaded["bootstrap"]

    common = {
        "analysis_id": analysis_id, "experiment_id": experiment_id,
        "block_size_cells": int(block_size_cells), "nominal_scale": nominal_scale_label(block_size_cells),
        "primary_population": PRIMARY_POPULATION,
    }

    metrics = {
        **old_metrics, "analysis_id": analysis_id,
        **brier_improvement_point_fields(old_metrics.get("delta_brier")),
    }
    (output_dir / "step8b_metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    (output_dir / "step8b_metrics.md").write_text(_condition_metrics_markdown(metrics), encoding="utf-8")

    bootstrap = {**old_bootstrap, "analysis_id": analysis_id}
    bootstrap["series"] = {
        **old_bootstrap["series"],
        "brier_improvement": derive_brier_improvement_series(old_bootstrap["series"]["delta_brier"]),
    }
    (output_dir / "bootstrap_summary.json").write_text(json.dumps(bootstrap, indent=2, default=str) + "\n")
    (output_dir / "bootstrap_summary.md").write_text(_condition_bootstrap_markdown(bootstrap), encoding="utf-8")

    block_manifest = _build_block_manifest(
        output_dir, common, loaded["block_audit"], experiment_id,
        report_regeneration_only=True, models_refit=False, bootstrap_rerun=False,
    )
    (output_dir / "block_manifest.json").write_text(json.dumps(block_manifest, indent=2, default=str) + "\n")

    return {"metrics": metrics, "bootstrap": bootstrap, "block_audit": loaded["block_audit"]}


# =============================================================================
# Comparison rows
# =============================================================================
def reference_small_block_row(experiment_id: str, analysis_id: str) -> dict[str, Any]:
    reference = original_small_block_reference(experiment_id)
    point, ci = reference["point"], reference["ci"]
    roc_ci, pr_ci = ci["delta_auc_ci95"], ci["delta_pr_auc_ci95"]
    brier_ci = ci.get("delta_brier_ci95")  # legacy convention; may be absent in older artifacts
    roc_status = classify_metric_support(roc_ci[0], roc_ci[1])
    pr_status = classify_metric_support(pr_ci[0], pr_ci[1])
    overall = classify_overall_support(roc_status, pr_status)

    # Brier is OPTIONAL in the frozen original small-block artifact. Never
    # calculated or invented, and never read from another run -- if absent,
    # every Brier-derived field is reported as null/unavailable.
    baseline_brier = point.get("overall_baseline", {}).get("brier_score")
    thermal_brier = point.get("overall_thermal", {}).get("brier_score")
    brier_available = baseline_brier is not None and thermal_brier is not None
    if brier_available:
        brier_point = brier_improvement_point_fields(point.get("delta_brier"))
        brier_ci_low = brier_ci[0] if brier_ci else None
        brier_ci_high = brier_ci[1] if brier_ci else None
        brier_ci_derived = brier_improvement_ci_fields(brier_ci_low, brier_ci_high)
        legacy_delta_brier = brier_point["delta_brier"]
        brier_improvement_value = brier_point["brier_improvement"]
        brier_improvement_ci_low = brier_ci_derived["brier_improvement_ci_low"]
        brier_improvement_ci_high = brier_ci_derived["brier_improvement_ci_high"]
        brier_support_status = brier_ci_derived["brier_support_status"]
    else:
        brier_ci_low = None
        brier_ci_high = None
        legacy_delta_brier = None
        brier_improvement_value = None
        brier_improvement_ci_low = None
        brier_improvement_ci_high = None
        brier_support_status = "unavailable_in_frozen_original_artifact"

    return {
        "analysis_id": analysis_id, "experiment": experiment_id,
        "source_type": "frozen_original_small_block", "block_size_cells": STEP8B_SPATIAL_BLOCK_SIZE_CELLS,
        "nominal_block_scale": nominal_scale_label(STEP8B_SPATIAL_BLOCK_SIZE_CELLS),
        "primary_population": PRIMARY_POPULATION,
        "baseline_roc_auc": point["overall_baseline"]["roc_auc"], "thermal_roc_auc": point["overall_thermal"]["roc_auc"],
        "delta_roc_auc": point["delta_auc"], "delta_roc_auc_ci_low": roc_ci[0], "delta_roc_auc_ci_high": roc_ci[1],
        "baseline_pr_auc": point["overall_baseline"]["pr_auc"], "thermal_pr_auc": point["overall_thermal"]["pr_auc"],
        "delta_pr_auc": point["delta_pr_auc"], "delta_pr_auc_ci_low": pr_ci[0], "delta_pr_auc_ci_high": pr_ci[1],
        "baseline_brier": baseline_brier, "thermal_brier": thermal_brier,
        # deprecated legacy fields (thermal-baseline, negative=thermal better)
        "delta_brier": legacy_delta_brier, "delta_brier_ci_low": brier_ci_low, "delta_brier_ci_high": brier_ci_high,
        # canonical fields (baseline-thermal, positive=thermal better)
        "brier_improvement": brier_improvement_value,
        "brier_improvement_ci_low": brier_improvement_ci_low,
        "brier_improvement_ci_high": brier_improvement_ci_high,
        "brier_improvement_definition": brier_improvement_definition(),
        "brier_support_status": brier_support_status,
        "roc_support_status": roc_status, "pr_support_status": pr_status,
        "overall_thermal_support_status": overall,
        "reference_metric_mismatch": reference["reference_check"]["reference_metric_mismatch"],
        "reference_check_expected": reference["reference_check"]["expected"],
        "reference_check_actual": reference["reference_check"]["actual"],
        "reference_check_note": reference["reference_check"]["note"],
    }


def big_block_comparison_row(condition: dict[str, Any], analysis_id: str) -> dict[str, Any]:
    metrics, bootstrap = condition["metrics"], condition["bootstrap"]
    roc = bootstrap["series"]["delta_auc"]; pr = bootstrap["series"]["delta_pr_auc"]
    br_legacy = bootstrap["series"]["delta_brier"]
    br_improvement = bootstrap["series"]["brier_improvement"]
    roc_status = classify_metric_support(roc["ci_2_5"], roc["ci_97_5"])
    pr_status = classify_metric_support(pr["ci_2_5"], pr["ci_97_5"])
    brier_status = classify_brier_improvement_support(br_improvement["ci_2_5"], br_improvement["ci_97_5"])
    overall = classify_overall_support(roc_status, pr_status)
    return {
        "analysis_id": analysis_id, "experiment": metrics["experiment_id"],
        "source_type": "new_big_block_robustness", "block_size_cells": metrics["block_size_cells"],
        "nominal_block_scale": metrics["nominal_scale"], "primary_population": PRIMARY_POPULATION,
        "baseline_roc_auc": metrics["baseline_roc_auc"], "thermal_roc_auc": metrics["thermal_roc_auc"],
        "delta_roc_auc": metrics["delta_roc_auc"], "delta_roc_auc_ci_low": roc["ci_2_5"], "delta_roc_auc_ci_high": roc["ci_97_5"],
        "baseline_pr_auc": metrics["baseline_pr_auc"], "thermal_pr_auc": metrics["thermal_pr_auc"],
        "delta_pr_auc": metrics["delta_pr_auc"], "delta_pr_auc_ci_low": pr["ci_2_5"], "delta_pr_auc_ci_high": pr["ci_97_5"],
        "baseline_brier": metrics["baseline_brier"], "thermal_brier": metrics["thermal_brier"],
        # deprecated legacy fields (thermal-baseline, negative=thermal better)
        "delta_brier": metrics["delta_brier"], "delta_brier_ci_low": br_legacy["ci_2_5"], "delta_brier_ci_high": br_legacy["ci_97_5"],
        # canonical fields (baseline-thermal, positive=thermal better)
        "brier_improvement": metrics["brier_improvement"],
        "brier_improvement_ci_low": br_improvement["ci_2_5"], "brier_improvement_ci_high": br_improvement["ci_97_5"],
        "brier_improvement_definition": brier_improvement_definition(),
        "roc_support_status": roc_status, "pr_support_status": pr_status, "brier_support_status": brier_status,
        "overall_thermal_support_status": overall,
        "bootstrap_stability": bootstrap["bootstrap_stability"],
        "valid_bootstrap_replicates": bootstrap["valid_replicates"],
    }


# =============================================================================
# Final report
# =============================================================================
def _support_and_magnitude_conclusion(support_status: str, magnitude_status: str) -> str:
    """Generates the Markdown conclusion sentence from the two independent
    statuses. Never fabricates a claim beyond what the two computed statuses
    already establish."""
    if support_status == "strongly_robust":
        if magnitude_status == "decreases_with_block_scale":
            return (
                "Positive thermal contribution was retained at both larger "
                "block scales, while the magnitude of the contribution "
                "decreased as spatial separation increased."
            )
        if magnitude_status == "increases_with_block_scale":
            return (
                "Positive thermal contribution was retained at both larger "
                "block scales, and the magnitude of the contribution "
                "increased as spatial separation increased."
            )
        if magnitude_status == "stable_across_block_scale":
            return (
                "Positive thermal contribution was retained at both larger "
                "block scales, and the magnitude of the contribution "
                "remained stable as spatial separation increased."
            )
        return (
            "Positive thermal contribution was retained at both larger "
            "block scales; the trend in the magnitude of the contribution "
            "across scales was non-monotonic or could not be determined."
        )
    if support_status in ("moderately_robust", "scale_sensitive"):
        return (
            "Positive thermal contribution was only partially retained "
            "across the predefined large-block scales; see the "
            "per-block-size support status for which scale(s) lost support."
        )
    return (
        "Positive thermal contribution was not retained at the predefined "
        "large-block scales evaluated."
    )


def write_final_report(
    output_root: Path, analysis_id: str, experiment_id: str, comparison: list[dict[str, Any]],
    conditions: dict[int, dict[str, Any]], infeasible: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    new_rows = [row for row in comparison if row["source_type"] == "new_big_block_robustness"]
    by_block = {row["block_size_cells"]: row["overall_thermal_support_status"] for row in new_rows}
    delta_roc_by_block = {row["block_size_cells"]: row["delta_roc_auc"] for row in new_rows}
    delta_pr_by_block = {row["block_size_cells"]: row["delta_pr_auc"] for row in new_rows}
    block_sizes_sorted = sorted(by_block)
    if len(block_sizes_sorted) == 2:
        support_status = classify_final_robustness(by_block[block_sizes_sorted[0]], by_block[block_sizes_sorted[1]])
    else:
        support_status = "not_robust"

    reference_rows = [row for row in comparison if row["source_type"] == "frozen_original_small_block"]
    delta_roc_small = reference_rows[0]["delta_roc_auc"] if reference_rows else None
    delta_pr_small = reference_rows[0]["delta_pr_auc"] if reference_rows else None
    magnitude_details = compute_effect_magnitude_details(
        delta_roc_small, delta_roc_by_block.get(10), delta_roc_by_block.get(20),
        delta_pr_small, delta_pr_by_block.get(10), delta_pr_by_block.get(20),
    )
    magnitude_status = classify_effect_magnitude_stability(magnitude_details)
    conclusion = _support_and_magnitude_conclusion(support_status, magnitude_status)

    report = {
        "analysis_id": analysis_id,
        "report_schema_version": ANALYSIS_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "primary_population": PRIMARY_POPULATION,
        "block_sizes_evaluated": block_sizes_sorted,
        "per_block_size_status": by_block,
        # support_robustness_status: whether bootstrap SUPPORT for a positive
        # thermal contribution is retained across block scales.
        "support_robustness_status": support_status,
        # effect_magnitude_stability_status: whether the SIZE of that
        # contribution grows/shrinks/stays stable across block scales. This
        # is an independent question from support_robustness_status above.
        "effect_magnitude_stability_status": magnitude_status,
        "effect_magnitude_details": magnitude_details,
        "conclusion": conclusion,
        "infeasible_block_sizes": {str(size): info for size, info in infeasible.items()},
        "original_small_block_reference": reference_rows,
        "new_big_block_conditions": new_rows,
        "block_fold_qa": {str(size): conditions[size]["block_audit"] for size in conditions},
        "reporting_provenance": reporting_provenance(experiment_id),
        "migration_notes_v1_to_v2": list(MIGRATION_NOTES_V1_TO_V2),
        "lst_anomaly_mean_provenance": LST_ANOMALY_SEMANTIC_NOTE,
        "claim_boundaries": [
            "no causal thermal effects", "not operational wildfire prediction",
            "no statistical significance or p-values",
            "no proof that residual spatial dependence is absent",
            "not successful cross-region transfer", "no best block size was selected",
            "absolute AUC decline alone is not treated as failure; the primary "
            "estimand is the paired baseline-vs-thermal delta and its bootstrap interval",
            "support_robustness_status and effect_magnitude_stability_status "
            "answer different questions and must not be conflated",
        ],
    }
    (output_root / "comparison" / "big_block_robustness_summary.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    pd.DataFrame(comparison).to_csv(output_root / "comparison" / "big_block_robustness_table.csv", index=False)

    def table(headers, rows):
        return (
            ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
            + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
        )

    md_rows = [
        [r["source_type"], r["block_size_cells"], r["nominal_block_scale"],
         f"{r['delta_roc_auc']:.6f}", f"[{r['delta_roc_auc_ci_low']:.6f}, {r['delta_roc_auc_ci_high']:.6f}]", r["roc_support_status"],
         f"{r['delta_pr_auc']:.6f}", f"[{r['delta_pr_auc_ci_low']:.6f}, {r['delta_pr_auc_ci_high']:.6f}]", r["pr_support_status"],
         f"{r['brier_improvement']:.6f}" if r.get("brier_improvement") is not None else "n/a",
         r.get("brier_support_status", "n/a"),
         r["overall_thermal_support_status"]]
        for r in comparison
    ]
    lines = [
        f"# Step8 Big-Spatial-Block Robustness Report -- {experiment_id}", "",
        f"- analysis_id: `{analysis_id}`",
        f"- primary population: {PRIMARY_POPULATION}",
        f"- support robustness status: **{support_status}**",
        f"- effect magnitude stability status: **{magnitude_status}**",
        f"- delta ROC-AUC relative reduction (small→10-cell): {magnitude_details['delta_roc_auc_relative_reduction_small_to_10']}",
        f"- delta ROC-AUC relative reduction (small→20-cell): {magnitude_details['delta_roc_auc_relative_reduction_small_to_20']}",
        f"- delta ROC-AUC relative reduction (10→20-cell): {magnitude_details['delta_roc_auc_relative_reduction_10_to_20']}",
        f"- delta PR-AUC relative reduction (small→10-cell): {magnitude_details['delta_pr_auc_relative_reduction_small_to_10']}",
        f"- delta PR-AUC relative reduction (small→20-cell): {magnitude_details['delta_pr_auc_relative_reduction_small_to_20']}",
        f"- delta PR-AUC relative reduction (10→20-cell): {magnitude_details['delta_pr_auc_relative_reduction_10_to_20']}",
        "", conclusion, "",
    ] + table(
        ["source", "block cells", "nominal scale", "delta ROC-AUC", "ROC CI", "ROC support",
         "delta PR-AUC", "PR CI", "PR support", "brier_improvement", "brier support", "overall status"], md_rows,
    ) + ["", "## Claim boundaries", ""] + [f"- {item}" for item in report["claim_boundaries"]]
    (output_root / "comparison" / "big_block_robustness_summary.md").write_text("\n".join(lines) + "\n")
    return report


# =============================================================================
# Dry-run / top-level entry point
# =============================================================================
def dry_run_plan(experiment_id: str, block_sizes: list[int], output_root: Path | None = None) -> dict[str, Any]:
    output_root = experiment_output_root(experiment_id) if output_root is None else output_root
    validate_block_sizes(block_sizes)
    protected = hash_all_protected(experiment_id)
    classifier = build_classifier(MODEL_NAME, STEP8B_RANDOM_SEED)
    reference = original_small_block_reference(experiment_id)
    input_path = experiment_step8_root(experiment_id) / "step8a" / "step8a_500m_modeling_dataset.parquet"
    return {
        "ran": False, "dry_run": True,
        "experiment_id": experiment_id,
        "block_sizes_cells": [int(b) for b in block_sizes],
        "primary_population": PRIMARY_POPULATION,
        "resolved_step8a_input": str(input_path),
        "resolved_step8a_input_sha256": sha256_file(input_path) if input_path.is_file() else None,
        "output_namespace": str(output_root),
        "condition_output_dirs": [str(_condition_output_dir(experiment_id, b, output_root)) for b in block_sizes],
        "model": {"class": type(classifier).__name__, "hyperparameters": classifier.get_params(deep=False)},
        "cv": {"class": "StratifiedGroupKFold", "folds": STEP8B_N_SPLITS, "seed": STEP8B_RANDOM_SEED, "strict_folds": True},
        "bootstrap": {"replicates": STEP8C_N_BOOTSTRAP, "seed": STEP8C_RANDOM_SEED, "ci_method": f"{STEP8C_CI_LOWER}/{STEP8C_CI_UPPER} percentile"},
        "original_small_block_size_verified": reference["metrics"]["spatial_cv_config"]["spatial_block_size_cells"],
        "reference_metric_mismatch": reference["reference_check"]["reference_metric_mismatch"],
        "stages_that_will_run": ["step8b_big_block_spatial_cv", "step8c_big_block_paired_bootstrap", "comparison_report"],
        "stages_that_will_not_run": ["predictors", "step5", "step5c", "step7", "step8a_regeneration"],
        "fit_or_bootstrap_performed": False, "files_written": False,
        "protected_original_step8_files": len(protected),
    }


def regenerate_reports_from_frozen_artifacts(
    experiment_id: str, dry_run: bool = False, output_root: Path | None = None,
) -> dict[str, Any]:
    """Regenerates JSON/Markdown/CSV/manifest reporting artifacts from a
    PRIOR, COMPLETED full fit+bootstrap run's frozen artifacts only.

    This function is the ENTIRE report-only code path and is called before
    -- and instead of -- anything in run_analysis. It never:
      - creates or validates a NEW immutable preregistration
      - computes a fresh scientific_configuration() to compare against one
      - constructs spatial-block folds
      - fits a baseline/thermal model
      - draws a bootstrap replicate

    It requires an existing completed analysis: the top-level
    comparison/manifest.json (the immutable preregistration already
    written by a prior full run) is read as-is and never rewritten here.
    `analysis_id` and `block_sizes_cells` are both taken from that frozen
    manifest -- never from CLI defaults or a freshly computed value -- so
    the regenerated reports carry forward the SAME analysis identity the
    original run established.

    Fails clearly (never fabricates, never falls back to a new analysis) if
    the top-level manifest or any block size's per-condition artifacts are
    missing, listing exactly what is missing.
    """
    output_root = experiment_output_root(experiment_id) if output_root is None else output_root
    manifest_path = output_root / "comparison" / "manifest.json"
    if not manifest_path.is_file():
        raise Step8BigBlockRobustnessError(
            f"Cannot regenerate reports for {experiment_id}: no completed big-block "
            f"analysis found (missing {manifest_path}). regenerate_reports_only "
            "requires an existing preregistration + full fit; run a full analysis "
            "first (without regenerate_reports_only)."
        )
    existing_manifest = json.loads(manifest_path.read_text())
    analysis_id = existing_manifest.get("analysis_id")
    config = existing_manifest.get("scientific_configuration", {})
    block_sizes = [int(b) for b in (config.get("block_sizes_cells") or [])]
    if not analysis_id or not block_sizes:
        raise Step8BigBlockRobustnessError(
            f"Cannot regenerate reports for {experiment_id}: existing preregistration "
            f"at {manifest_path} is missing analysis_id and/or "
            "scientific_configuration.block_sizes_cells."
        )

    condition_dirs = {size: _condition_output_dir(experiment_id, size, output_root) for size in block_sizes}

    if dry_run:
        return {
            "ran": False, "dry_run": True, "report_regeneration_only": True,
            "experiment_id": experiment_id, "analysis_id": analysis_id,
            "block_sizes_cells": block_sizes,
            "preregistration_path": str(manifest_path),
            "condition_output_dirs": {str(size): str(path) for size, path in condition_dirs.items()},
            "fit_or_bootstrap_performed": False, "files_written": False,
        }

    before = hash_all_protected(experiment_id)

    conditions: dict[int, dict[str, Any]] = {}
    missing: dict[int, dict[str, Any]] = {}
    for block_size in block_sizes:
        try:
            conditions[block_size] = write_condition_reports_only(
                condition_dirs[block_size], experiment_id, block_size, analysis_id,
            )
        except Step8BigBlockRobustnessError as exc:
            missing[block_size] = {
                "status": "missing_frozen_artifacts_for_report_regeneration", "reason": str(exc),
            }
            log.warning("block_size=%s: cannot regenerate reports: %s", block_size, exc)

    if not conditions:
        missing_summary = "; ".join(f"block_size={size}: {info['reason']}" for size, info in missing.items())
        raise Step8BigBlockRobustnessError(
            f"regenerate_reports_only found no existing per-block-size artifacts to "
            f"regenerate from for {experiment_id} at block sizes {block_sizes}: "
            f"{missing_summary}. Run a full analysis first (without "
            "regenerate_reports_only) -- this mode never falls back to a new analysis."
        )

    comparison = [reference_small_block_row(experiment_id, analysis_id)]
    comparison += [big_block_comparison_row(conditions[size], analysis_id) for size in sorted(conditions)]

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "comparison").mkdir(parents=True, exist_ok=True)
    report = write_final_report(output_root, analysis_id, experiment_id, comparison, conditions, missing)

    after = hash_all_protected(experiment_id)
    assert_all_protected_unchanged(before, after)

    return {
        "ran": True, "report_regeneration_only": True, "models_refit": False, "bootstrap_rerun": False,
        "analysis_id": analysis_id, "experiment_id": experiment_id,
        "conditions_regenerated": list(conditions.keys()), "missing_block_sizes": list(missing.keys()),
        "report": report, "protected_hash_check": "passed",
    }


def run_analysis(
    experiment_id: str, block_sizes: list[int] | None = None,
    dry_run: bool = False, force: bool = False, output_root: Path | None = None,
) -> dict[str, Any]:
    """Normal (full fit + bootstrap) analysis path. Unchanged immutable
    preregistration validation: report-only regeneration never goes through
    this function at all -- see regenerate_reports_from_frozen_artifacts,
    which is dispatched before this function is ever called."""
    block_sizes = [int(b) for b in (block_sizes or DEFAULT_BLOCK_SIZES)]
    validate_block_sizes(block_sizes)
    output_root = experiment_output_root(experiment_id) if output_root is None else output_root

    if dry_run:
        return dry_run_plan(experiment_id, block_sizes, output_root)

    before = hash_all_protected(experiment_id)
    assert_downstream_outputs_writable(output_root, force)
    manifest = validate_or_write_manifest(output_root, experiment_id, block_sizes, before)
    analysis_id = manifest["analysis_id"]

    input_path = experiment_step8_root(experiment_id) / "step8a" / "step8a_500m_modeling_dataset.parquet"
    df_all = pd.read_parquet(input_path)

    conditions: dict[int, dict[str, Any]] = {}
    infeasible: dict[int, dict[str, Any]] = {}
    for block_size in block_sizes:
        result = run_big_block_condition(df_all, experiment_id, block_size, analysis_id)
        if result["status"] != "fitted":
            infeasible[block_size] = result
            log.warning("block_size=%s infeasible: %s", block_size, result.get("reason"))
            continue
        bootstrap, replicates = paired_big_block_bootstrap(result["predictions"])
        condition = write_condition_outputs(
            _condition_output_dir(experiment_id, block_size, output_root),
            analysis_id, experiment_id, block_size, result, bootstrap, replicates,
        )
        conditions[block_size] = condition

    comparison = [reference_small_block_row(experiment_id, analysis_id)]
    comparison += [big_block_comparison_row(conditions[size], analysis_id) for size in sorted(conditions)]

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "comparison").mkdir(parents=True, exist_ok=True)
    report = write_final_report(output_root, analysis_id, experiment_id, comparison, conditions, infeasible)

    after = hash_all_protected(experiment_id)
    assert_all_protected_unchanged(before, after)

    return {
        "ran": True, "analysis_id": analysis_id, "experiment_id": experiment_id,
        "conditions": list(conditions.keys()), "infeasible_block_sizes": list(infeasible.keys()),
        "report": report, "protected_hash_check": "passed",
    }
