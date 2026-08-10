"""Comparison-series enumeration, orientation arithmetic and row formulas.

This module plans and computes; it never resamples and never fits. The actual
replicate generation reuses the frozen per-AOI
`window_closure_sensitivity.multi_variant_block_bootstrap`, which rescores
stored out-of-fold predictions and structurally cannot refit -- it never
receives a feature matrix.

The frozen representation (see OUTPUT_SCHEMA.md 12.0):

* A **comparison series** is one
  `(comparison_family, variant, model_a, model_b, metric)` tuple.
  There are **27 series per AOI**, which is the measured row count of the
  frozen Manavgat `paired_bootstrap_summary.csv`.
* Orientation is a **column pair, never a row dimension**.

Design reference: docs/multi_region_window_closure_design/OUTPUT_SCHEMA.md
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from src.multi_region_window_closure.contract import (
    ACTUAL_AOIS,
    COMPARISON_CLOSURE_CHANGE,
    COMPARISON_CONTRIBUTION_CHANGE,
    COMPARISON_THERMAL_CONTRIBUTION,
    METRICS,
    METRIC_DIRECTION,
    MODEL_FAMILIES,
    MultiRegionWindowClosureError,
    ORIENTATION_NATURAL,
    ORIENTATION_ORIENTED,
    SHIFTED_VARIANTS,
    SYNTHESIS_AOIS,
    VARIANTS,
    classify_interval,
    interval_excludes_zero,
    orient,
    orient_interval,
    orientation_definition,
    orientations_equal,
)

#: Label used in `model_a`/`model_b` when the series compares a DIFFERENCE
#: rather than two model families. Never NULL -- a literal is required so the
#: primary key is always complete.
CONTRAST_THERMAL_MINUS_BASELINE = "thermal_minus_baseline"


def comparison_series(aoi: str) -> list[dict[str, str]]:
    """The 27 comparison series of one AOI, in deterministic order.

    9  thermal_contribution_within_variant : 3 variants x 3 metrics
    12 closure_change_within_model_family  : 2 shifted x 2 families x 3 metrics
    6  thermal_contribution_change         : 2 shifted x 3 metrics
    """
    series: list[dict[str, str]] = []
    for variant in VARIANTS:
        for metric in METRICS:
            series.append({
                "aoi": aoi,
                "comparison_family": COMPARISON_THERMAL_CONTRIBUTION,
                "variant": variant,
                "model_a": "thermal",
                "model_b": "baseline",
                "metric": metric,
            })
    for variant in SHIFTED_VARIANTS:
        for model in MODEL_FAMILIES:
            for metric in METRICS:
                series.append({
                    "aoi": aoi,
                    "comparison_family": COMPARISON_CLOSURE_CHANGE,
                    "variant": variant,
                    "model_a": model,
                    "model_b": model,
                    "metric": metric,
                })
    for variant in SHIFTED_VARIANTS:
        for metric in METRICS:
            series.append({
                "aoi": aoi,
                "comparison_family": COMPARISON_CONTRIBUTION_CHANGE,
                "variant": variant,
                "model_a": CONTRAST_THERMAL_MINUS_BASELINE,
                "model_b": CONTRAST_THERMAL_MINUS_BASELINE,
                "metric": metric,
            })
    return series


SERIES_PER_AOI = len(comparison_series("_"))


def series_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    """Primary key of one series row."""
    return (
        str(row["aoi"]), str(row["comparison_family"]), str(row["variant"]),
        str(row["model_a"]), str(row["model_b"]), str(row["metric"]),
    )


def assert_series_unique(rows: Sequence[dict[str, Any]]) -> None:
    keys = [series_key(r) for r in rows]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise MultiRegionWindowClosureError(
            f"BLOCKER: DUPLICATE_SERIES -- {duplicates[:4]}."
        )


# =============================================================================
# Row-count formulas
# =============================================================================
def bootstrap_summary_row_count(aois: Sequence[str] = ACTUAL_AOIS) -> int:
    """`A x 27` -- one row per series, both orientations in columns."""
    return len(aois) * SERIES_PER_AOI


def four_region_synthesis_row_count(aois: Sequence[str] = SYNTHESIS_AOIS) -> int:
    """`(A + 1) x 27` -- three actual AOIs plus the read-only reference."""
    return len(aois) * SERIES_PER_AOI


def bootstrap_replicate_row_count(valid_replicates_by_aoi: dict[str, int]) -> int:
    """`sum_aoi (27 x valid_replicates_aoi)`."""
    return sum(SERIES_PER_AOI * int(n) for n in valid_replicates_by_aoi.values())


def metrics_row_count(aois: Sequence[str] = ACTUAL_AOIS) -> int:
    """`A x V x M x |metrics|`."""
    return len(aois) * len(VARIANTS) * len(MODEL_FAMILIES) * len(METRICS)


# =============================================================================
# Draw plan identity
# =============================================================================
def draw_plan_id(aoi: str, seed: int, n_bootstrap: int, block_ids: Sequence[Any]) -> str:
    """Deterministic identity of ONE AOI's bootstrap draw plan.

    Every comparison inside an AOI must share this value -- that is what makes
    the differences paired. Draw plans are never shared ACROSS AOIs, because
    AOIs have different block populations.
    """
    from src.step8_large_block_robustness import canonical_json, sha256_bytes

    payload = {
        "aoi": aoi,
        "seed": int(seed),
        "n_bootstrap": int(n_bootstrap),
        "unit": "spatial_block_id",
        "blocks": sorted(str(b) for b in block_ids),
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def assert_paired_draw_plan(rows: Sequence[dict[str, Any]]) -> None:
    """One draw plan per AOI, shared by every series of that AOI."""
    by_aoi: dict[str, set] = {}
    for row in rows:
        by_aoi.setdefault(str(row["aoi"]), set()).add(str(row.get("draw_plan_hash")))
    for aoi, plans in sorted(by_aoi.items()):
        if len(plans) != 1:
            raise MultiRegionWindowClosureError(
                f"BLOCKER: DRAW_PLAN_MISMATCH -- {aoi} carries {len(plans)} "
                "draw plans; every comparison of one AOI must share one."
            )
    seen: dict[str, str] = {}
    for aoi, plans in sorted(by_aoi.items()):
        plan = next(iter(plans))
        if plan in seen and seen[plan] != aoi:
            raise MultiRegionWindowClosureError(
                f"BLOCKER: DRAW_PLAN_MISMATCH -- {seen[plan]} and {aoi} share "
                "a draw plan; plans are per AOI."
            )
        seen[plan] = aoi


# =============================================================================
# Replicate and summary row construction (pure arithmetic)
# =============================================================================
def replicate_row(
    series: dict[str, str],
    replicate_id: int,
    draw_plan_hash: str,
    estimate_a: float,
    estimate_b: float,
) -> dict[str, Any]:
    """One replicate row, carrying BOTH orientations.

    `difference_natural` is the raw convention `estimate_a - estimate_b`;
    `difference_oriented` applies the metric's improvement orientation.
    """
    metric = series["metric"]
    natural = float(estimate_a) - float(estimate_b)
    return {
        "aoi": series["aoi"],
        "comparison_family": series["comparison_family"],
        "variant": series["variant"],
        "model_a": series["model_a"],
        "model_b": series["model_b"],
        "metric": metric,
        "replicate_id": int(replicate_id),
        "draw_plan_id": draw_plan_hash,
        "estimate_a": float(estimate_a),
        "estimate_b": float(estimate_b),
        "difference_natural": natural,
        "difference_oriented": orient(metric, natural),
        "valid": True,
        "invalid_reason": None,
    }


def percentile_interval(
    values: Sequence[float], ci_lower: float, ci_upper: float,
) -> dict[str, Any]:
    """Percentile interval over the finite replicate values."""
    import numpy as np

    array = np.asarray([v for v in values if v is not None], dtype="float64")
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    return {
        "mean": float(np.mean(array)),
        "ci_low": float(np.percentile(array, ci_lower)),
        "ci_high": float(np.percentile(array, ci_upper)),
        "n": int(array.size),
    }


def summary_row(
    series: dict[str, str],
    point_estimate_natural: float,
    natural_differences: Sequence[float],
    bootstrap_config: dict[str, Any],
    draw_plan_hash: str,
    block_count: int,
    requested_replicates: Optional[int] = None,
) -> dict[str, Any]:
    """One `bootstrap_summary.csv` row: one series, both orientations in columns.

    The oriented interval NEGATES the natural one for a loss metric, which
    REVERSES its endpoints. The swap is done by `orient_interval` and is the
    reason there is no bare `ci_low` column anywhere in this schema.
    """
    metric = series["metric"]
    requested = int(
        requested_replicates
        if requested_replicates is not None
        else bootstrap_config["n_bootstrap"]
    )
    interval = percentile_interval(
        natural_differences,
        bootstrap_config["ci_lower_percentile"],
        bootstrap_config["ci_upper_percentile"],
    )
    valid = int(interval["n"])
    low_n, high_n = interval["ci_low"], interval["ci_high"]
    if low_n is None or high_n is None:
        low_o = high_o = None
        mean_o = None
    else:
        low_o, high_o = orient_interval(metric, low_n, high_n)
        mean_o = orient(metric, interval["mean"])
    point_natural = float(point_estimate_natural)
    return {
        "aoi": series["aoi"],
        "comparison_family": series["comparison_family"],
        "variant": series["variant"],
        "model_a": series["model_a"],
        "model_b": series["model_b"],
        "metric": metric,
        "metric_direction": METRIC_DIRECTION[metric],
        "orientation_natural_definition": orientation_definition(
            series["comparison_family"], metric, ORIENTATION_NATURAL,
        ),
        "orientation_oriented_definition": orientation_definition(
            series["comparison_family"], metric, ORIENTATION_ORIENTED,
        ),
        "point_estimate_natural": point_natural,
        "ci_low_natural": low_n,
        "ci_high_natural": high_n,
        "bootstrap_mean_natural": interval["mean"],
        "point_estimate_oriented": orient(metric, point_natural),
        "ci_low_oriented": low_o,
        "ci_high_oriented": high_o,
        "bootstrap_mean_oriented": mean_o,
        "orientations_equal": orientations_equal(metric),
        "confidence_level": bootstrap_config["confidence_level"],
        "interval_method": bootstrap_config["interval_method"],
        "ci_lower_percentile": bootstrap_config["ci_lower_percentile"],
        "ci_upper_percentile": bootstrap_config["ci_upper_percentile"],
        "requested_replicates": requested,
        "valid_replicates": valid,
        "invalid_replicates": requested - valid,
        "seed": bootstrap_config["seed"],
        "draw_plan_hash": draw_plan_hash,
        "block_count": int(block_count),
        # Orientation-invariant: negating an interval cannot change whether it
        # excludes zero.
        "interval_excludes_zero": interval_excludes_zero(low_n, high_n),
        "interval_status": classify_interval(low_n, high_n),
    }


def assert_summary_row(row: dict[str, Any], tolerance: float = 1e-9) -> None:
    """Fail closed on orientation, accounting or interval inconsistency."""
    metric = row["metric"]
    if row["valid_replicates"] + row["invalid_replicates"] != row["requested_replicates"]:
        raise MultiRegionWindowClosureError(
            "BLOCKER: REPLICATE_ACCOUNTING_UNTRUTHFUL -- "
            f"{row['aoi']}/{row['comparison_family']}/{row['metric']}: "
            f"{row['valid_replicates']} + {row['invalid_replicates']} != "
            f"{row['requested_replicates']}."
        )
    if row["block_count"] < 2:
        raise MultiRegionWindowClosureError(
            f"BLOCKER: INSUFFICIENT_BLOCKS -- {row['aoi']} has "
            f"{row['block_count']} blocks; the paired bootstrap needs >= 2."
        )
    if row["orientations_equal"] != orientations_equal(metric):
        raise MultiRegionWindowClosureError(
            f"BLOCKER: BRIER_ORIENTATION_ERROR -- {row['aoi']}/{metric}: "
            "orientations_equal disagrees with the metric contract."
        )
    expected_point = orient(metric, row["point_estimate_natural"])
    if abs(expected_point - row["point_estimate_oriented"]) > tolerance:
        raise MultiRegionWindowClosureError(
            f"BLOCKER: BRIER_ORIENTATION_ERROR -- {row['aoi']}/{metric}: "
            f"oriented point estimate {row['point_estimate_oriented']} != "
            f"{expected_point}."
        )
    if row["ci_low_natural"] is not None and row["ci_high_natural"] is not None:
        if row["ci_low_natural"] > row["ci_high_natural"]:
            raise MultiRegionWindowClosureError(
                f"BLOCKER: CI_ARITHMETIC_ERROR -- {row['aoi']}/{metric}: "
                "natural interval is inverted."
            )
        low_o, high_o = orient_interval(
            metric, row["ci_low_natural"], row["ci_high_natural"],
        )
        if (abs(low_o - row["ci_low_oriented"]) > tolerance
                or abs(high_o - row["ci_high_oriented"]) > tolerance):
            raise MultiRegionWindowClosureError(
                f"BLOCKER: BRIER_ORIENTATION_ERROR -- {row['aoi']}/{metric}: "
                "oriented interval bounds are not the negated-and-swapped "
                "natural bounds."
            )
        if row["ci_low_oriented"] > row["ci_high_oriented"]:
            raise MultiRegionWindowClosureError(
                f"BLOCKER: CI_ARITHMETIC_ERROR -- {row['aoi']}/{metric}: "
                "oriented interval is inverted; the endpoint swap was skipped."
            )
    status = classify_interval(row["ci_low_natural"], row["ci_high_natural"])
    if row["interval_status"] != status:
        raise MultiRegionWindowClosureError(
            f"BLOCKER: CI_ARITHMETIC_ERROR -- {row['aoi']}/{metric}: "
            f"interval_status {row['interval_status']} != {status}."
        )
    if row["interval_excludes_zero"] != (status != "interval_includes_zero"):
        raise MultiRegionWindowClosureError(
            f"BLOCKER: CI_ARITHMETIC_ERROR -- {row['aoi']}/{metric}: "
            "interval_excludes_zero disagrees with interval_status."
        )


# =============================================================================
# Column contracts
# =============================================================================
BOOTSTRAP_REPLICATE_COLUMNS: tuple[str, ...] = (
    "analysis_id", "aoi", "comparison_family", "variant", "model_a", "model_b",
    "metric", "replicate_id", "draw_plan_id", "estimate_a", "estimate_b",
    "difference_natural", "difference_oriented", "valid", "invalid_reason",
)

BOOTSTRAP_SUMMARY_COLUMNS: tuple[str, ...] = (
    "analysis_id", "aoi", "comparison_family", "variant", "model_a", "model_b",
    "metric", "metric_direction",
    "orientation_natural_definition", "orientation_oriented_definition",
    "point_estimate_natural", "ci_low_natural", "ci_high_natural",
    "bootstrap_mean_natural",
    "point_estimate_oriented", "ci_low_oriented", "ci_high_oriented",
    "bootstrap_mean_oriented", "orientations_equal",
    "confidence_level", "interval_method",
    "ci_lower_percentile", "ci_upper_percentile",
    "requested_replicates", "valid_replicates", "invalid_replicates",
    "seed", "draw_plan_hash", "block_count",
    "interval_excludes_zero", "interval_status",
)

#: Columns that would constitute POOLED inference. None may exist in
#: `four_region_synthesis.csv`.
PROHIBITED_POOLED_COLUMNS: tuple[str, ...] = (
    "pooled_estimate", "pooled_ci_low", "pooled_ci_high",
    "meta_analytic_estimate", "combined_p", "heterogeneity", "i_squared",
    "weight", "n_total_across_aois",
)


def assert_no_pooled_columns(columns: Sequence[str]) -> None:
    found = sorted(set(c.lower() for c in columns) & set(PROHIBITED_POOLED_COLUMNS))
    if found:
        raise MultiRegionWindowClosureError(
            f"BLOCKER: POOLED_INFERENCE_DETECTED -- prohibited column(s) {found}. "
            "Each AOI is analysed separately; there is no pooled estimate."
        )
