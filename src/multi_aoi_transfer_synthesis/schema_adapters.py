"""Per-schema-version parsing of resolved frozen-output payloads into a
normalized internal representation.

This is the ONLY module that knows concrete field names such as
`direction_summaries`, `population_results`, `input_audit`, etc. Every other
module in this package works exclusively with the normalized dict shapes
returned here.

All adapters take the raw parsed JSON (as returned by `resolvers.py`) plus
whatever identifying arguments (experiment id / source id / target id / the
resolver's `record`) are needed to locate the relevant slice of that JSON,
and return a plain dict in the normalized shape documented on each function.
Nothing here reads or writes files.
"""
from __future__ import annotations

import math
from typing import Any, Optional

from src.multi_aoi_transfer_synthesis.resolvers import InputResolutionError, PRIMARY_POPULATION


def _finite_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if math.isfinite(xf) else None


def _require_finite(family: str, name: str, value: Any) -> float:
    v = _finite_or_none(value)
    if v is None:
        raise InputResolutionError(family, f"Required metric '{name}' is missing or non-finite: {value!r}.")
    return v


# =============================================================================
# 1. Step8 within-region.
# =============================================================================
def adapt_step8_within_region(raw: dict, experiment_id: str) -> dict:
    """Normalized shape:
        {experiment_id, primary_population,
         baseline: {roc_auc, pr_auc}, thermal: {roc_auc, pr_auc}}
    """
    pop_metrics = (raw.get("population_metrics") or {}).get(PRIMARY_POPULATION)
    if pop_metrics is None:
        raise InputResolutionError(
            "step8_within_region",
            f"Population '{PRIMARY_POPULATION}' missing from Step8B metrics for '{experiment_id}'.",
        )
    baseline = pop_metrics.get("overall_baseline") or {}
    thermal = pop_metrics.get("overall_thermal") or {}
    return {
        "experiment_id": experiment_id,
        "primary_population": PRIMARY_POPULATION,
        "baseline": {
            "roc_auc": _require_finite("step8_within_region", "overall_baseline.roc_auc", baseline.get("roc_auc")),
            "pr_auc": _require_finite("step8_within_region", "overall_baseline.pr_auc", baseline.get("pr_auc")),
        },
        "thermal": {
            "roc_auc": _require_finite("step8_within_region", "overall_thermal.roc_auc", thermal.get("roc_auc")),
            "pr_auc": _require_finite("step8_within_region", "overall_thermal.pr_auc", thermal.get("pr_auc")),
        },
    }


# =============================================================================
# 2. Large-block robustness.
# =============================================================================
def adapt_large_block_robustness_generic(raw: dict, experiment_id: str) -> dict:
    """Normalized shape (generic per-experiment `step8.big_block_robustness.v2`):
        {available: True, support_status, effect_magnitude_stability_status}
    """
    if raw.get("experiment_id") != experiment_id:
        raise InputResolutionError(
            "large_block_robustness",
            f"experiment_id mismatch in generic big-block robustness report: "
            f"expected '{experiment_id}', found '{raw.get('experiment_id')}'.",
        )
    if raw.get("primary_population") != PRIMARY_POPULATION:
        raise InputResolutionError(
            "large_block_robustness",
            f"Generic big-block robustness report for '{experiment_id}' has "
            f"primary_population='{raw.get('primary_population')}', expected '{PRIMARY_POPULATION}'.",
        )
    return {
        "available": True,
        "support_status": raw.get("support_robustness_status"),
        "effect_magnitude_stability_status": raw.get("effect_magnitude_stability_status"),
        "unavailable_reason": None,
    }


def adapt_large_block_robustness_legacy(payload: dict, experiment_id: str) -> dict:
    """Normalized shape (legacy pair-relative `step8.large_block_robustness*`
    schema). `payload["conditions"]` is the pre-filtered (experiment,
    primary_population) match list produced by the resolver."""
    conditions = payload.get("conditions") or []
    if not conditions:
        raise InputResolutionError(
            "large_block_robustness",
            f"No matching legacy large-block-robustness conditions for '{experiment_id}'.",
        )
    statuses = {c.get("robustness_status") for c in conditions}
    if len(statuses) == 1:
        support_status = next(iter(statuses))
    else:
        support_status = "mixed_across_block_sizes"
    return {
        "available": True,
        "support_status": support_status,
        # The legacy schema does not report an effect-magnitude-stability
        # classification across block sizes; report that explicitly rather
        # than fabricating one.
        "effect_magnitude_stability_status": None,
        "unavailable_reason": None,
    }


def unavailable_large_block_robustness(reason: str) -> dict:
    return {
        "available": False,
        "support_status": None,
        "effect_magnitude_stability_status": None,
        "unavailable_reason": reason,
    }


# =============================================================================
# 3. Step9 raw transfer (step9d preferred; step9b/9c fallback).
# =============================================================================
def _find_step9d_direction(raw: dict, source_id: str, target_id: str) -> dict:
    wanted = f"{source_id}_to_{target_id}"
    for entry in raw.get("direction_summaries", []):
        if entry.get("transfer_direction") == wanted or (
            entry.get("source_experiment_id") == source_id
            and entry.get("target_experiment_id") == target_id
        ):
            return entry
    raise InputResolutionError(
        "step9_transfer",
        f"Direction '{wanted}' not present in Step9D report's direction_summaries.",
    )


def adapt_step9_transfer_step9d(raw: dict, source_id: str, target_id: str) -> dict:
    """Normalized shape:
        {source_experiment_id, target_experiment_id, primary_population,
         target_row_count, target_burned_count,
         model_family_metrics: {
             baseline: {roc_auc_ci_low, roc_auc_ci_high, pr_auc_ci_low, pr_auc_ci_high, brier},
             thermal:  {...},
         }}

    `brier` is the frozen Step9 Brier score (`<family>_metrics.brier_score`),
    populated whenever the frozen Step9 payload carries it -- Step9's raw
    transfer metrics are not gated behind the Step10 Brier preregistration
    protocol, so this is always available when present in the frozen file.
    """
    entry = _find_step9d_direction(raw, source_id, target_id)
    pop_results = (entry.get("population_results") or {}).get(PRIMARY_POPULATION)
    if pop_results is None:
        raise InputResolutionError(
            "step9_transfer",
            f"Population '{PRIMARY_POPULATION}' missing from Step9D direction "
            f"'{source_id}' -> '{target_id}'.",
        )
    transfer = pop_results.get("transfer") or {}
    ci = ((pop_results.get("bootstrap") or {}).get("confidence_intervals")) or {}

    def _ci(name: str) -> tuple[Optional[float], Optional[float]]:
        block = ci.get(name) or {}
        return _finite_or_none(block.get("ci_2_5")), _finite_or_none(block.get("ci_97_5"))

    baseline_roc_ci = _ci("baseline_roc_auc")
    thermal_roc_ci = _ci("thermal_roc_auc")
    baseline_pr_ci = _ci("baseline_pr_auc")
    thermal_pr_ci = _ci("thermal_pr_auc")
    baseline_brier = _finite_or_none((transfer.get("baseline_metrics") or {}).get("brier_score"))
    thermal_brier = _finite_or_none((transfer.get("thermal_metrics") or {}).get("brier_score"))

    return {
        "source_experiment_id": source_id,
        "target_experiment_id": target_id,
        "primary_population": PRIMARY_POPULATION,
        "target_row_count": transfer.get("target_cell_count"),
        "target_burned_count": transfer.get("target_positive_count"),
        "model_family_metrics": {
            "baseline": {"roc_auc_ci_low": baseline_roc_ci[0], "roc_auc_ci_high": baseline_roc_ci[1],
                         "pr_auc_ci_low": baseline_pr_ci[0], "pr_auc_ci_high": baseline_pr_ci[1],
                         "brier": baseline_brier},
            "thermal": {"roc_auc_ci_low": thermal_roc_ci[0], "roc_auc_ci_high": thermal_roc_ci[1],
                        "pr_auc_ci_low": thermal_pr_ci[0], "pr_auc_ci_high": thermal_pr_ci[1],
                        "brier": thermal_brier},
        },
    }


def adapt_step9_transfer_fallback(payload: dict, source_id: str, target_id: str) -> dict:
    """Normalized shape identical to `adapt_step9_transfer_step9d`, built
    from separate Step9B (point metrics) + Step9C (bootstrap CIs) payloads
    when a merged Step9D report is absent."""
    b_raw = payload["step9b"]
    c_raw = payload["step9c"]
    wanted = f"{source_id}_to_{target_id}"

    b_entry = next(
        (r for r in b_raw.get("results", [])
         if r.get("population") == PRIMARY_POPULATION and r.get("transfer_direction") == wanted),
        None,
    )
    if b_entry is None:
        raise InputResolutionError(
            "step9_transfer",
            f"Step9B fallback missing population/direction entry for '{wanted}'.",
        )
    c_entries = [
        r for r in c_raw.get("results", [])
        if r.get("population") == PRIMARY_POPULATION and r.get("transfer_direction") == wanted
    ]
    ci_by_metric: dict[str, dict] = {}
    for r in c_entries:
        metric_name = r.get("metric")
        if metric_name:
            ci_by_metric[metric_name] = r

    def _ci(name: str) -> tuple[Optional[float], Optional[float]]:
        block = ci_by_metric.get(name) or {}
        return _finite_or_none(block.get("ci_2_5")), _finite_or_none(block.get("ci_97_5"))

    baseline_roc_ci = _ci("baseline_roc_auc")
    thermal_roc_ci = _ci("thermal_roc_auc")
    baseline_pr_ci = _ci("baseline_pr_auc")
    thermal_pr_ci = _ci("thermal_pr_auc")
    baseline_brier = _finite_or_none((b_entry.get("baseline_metrics") or {}).get("brier_score"))
    thermal_brier = _finite_or_none((b_entry.get("thermal_metrics") or {}).get("brier_score"))

    return {
        "source_experiment_id": source_id,
        "target_experiment_id": target_id,
        "primary_population": PRIMARY_POPULATION,
        "target_row_count": b_entry.get("target_cell_count"),
        "target_burned_count": b_entry.get("target_positive_count"),
        "model_family_metrics": {
            "baseline": {"roc_auc_ci_low": baseline_roc_ci[0], "roc_auc_ci_high": baseline_roc_ci[1],
                         "pr_auc_ci_low": baseline_pr_ci[0], "pr_auc_ci_high": baseline_pr_ci[1],
                         "brier": baseline_brier},
            "thermal": {"roc_auc_ci_low": thermal_roc_ci[0], "roc_auc_ci_high": thermal_roc_ci[1],
                        "pr_auc_ci_low": thermal_pr_ci[0], "pr_auc_ci_high": thermal_pr_ci[1],
                        "brier": thermal_brier},
        },
    }


# =============================================================================
# 4. Step9E shift audit.
# =============================================================================
def adapt_step9e_shift(raw: dict, source_id: str, target_id: str) -> dict:
    """Normalized shape:
        {source_experiment_id, target_experiment_id,
         numeric_feature_rows: [ ...primary-population rows... ],
         frozen_diagnosis_categories: [ ...part_f_summary.diagnosis_categories,
             pair-global (any direction/model)... ],
         direction_specific_ranking_reversal_rows: [ ...part_d rows for this
             exact ordered direction + PRIMARY_POPULATION... ],
         pair_global_ranking_reversal_suspected: bool | None}

    `part_f_summary` is a single pair-global diagnosis rollup (not
    direction-keyed), already computed by the frozen Step9E protocol -- it is
    read here, never recomputed. `part_d_prediction_distribution_audit` rows
    carry a `transfer_direction` field and are the direction-specific ranking
    -reversal signal; by the time `raw` reaches this function the resolver
    has already filtered that field down to the requested direction only
    (when the artifact carries records for more than one direction).
    """
    rows = [
        r for r in raw.get("part_a_numeric_feature_shift", [])
        if r.get("population") == PRIMARY_POPULATION
        and r.get("source_experiment_id") == source_id
        and r.get("target_experiment_id") == target_id
    ]

    part_f = raw.get("part_f_summary") or {}
    frozen_diagnosis_categories = (
        list(part_f.get("diagnosis_categories") or [])
        if part_f.get("primary_population") == PRIMARY_POPULATION
        else []
    )
    pair_global_ranking_reversal_suspected = (
        part_f.get("ranking_reversal_suspected")
        if part_f.get("primary_population") == PRIMARY_POPULATION
        else None
    )

    requested_direction = f"{source_id}_to_{target_id}"
    direction_specific_ranking_reversal_rows = [
        r for r in raw.get("part_d_prediction_distribution_audit", [])
        if r.get("population") == PRIMARY_POPULATION
        and r.get("transfer_direction") == requested_direction
    ]

    return {
        "source_experiment_id": source_id,
        "target_experiment_id": target_id,
        "numeric_feature_rows": rows,
        "frozen_diagnosis_categories": frozen_diagnosis_categories,
        "direction_specific_ranking_reversal_rows": direction_specific_ranking_reversal_rows,
        "pair_global_ranking_reversal_suspected": pair_global_ranking_reversal_suspected,
    }


# =============================================================================
# 5. Step9G feature-AUC direction-reversal (v1 and v2).
# =============================================================================
def _resolve_experiment_prefix(row: dict, experiment_id: str) -> Optional[str]:
    """Find the field-name prefix used for `experiment_id` in a v1/v2
    direction-reversal-table row. The frozen schema derives the prefix from
    the experiment id in an unspecified, non-fixed way -- observed
    conventions include the full experiment id (e.g. "<experiment_id>_auc"),
    a truncated token (e.g. the experiment id with its trailing "_<suffix>"
    segment stripped, as "<prefix>_auc"), and a generic "source"/"target"
    indirection resolved via the row's own `source_experiment_id` /
    `target_experiment_id` fields (e.g. "source_auc" where
    row["source_experiment_id"] == experiment_id). This function tries all
    of them against the row's actual keys rather than assuming a fixed
    convention."""
    if row.get("source_experiment_id") == experiment_id and f"source_auc" in row:
        return "source"
    if row.get("target_experiment_id") == experiment_id and f"target_auc" in row:
        return "target"

    candidate = experiment_id
    while True:
        if f"{candidate}_auc" in row:
            return candidate
        if "_" not in candidate:
            return None
        candidate = candidate.rsplit("_", 1)[0]


def adapt_step9g_pair(payload: dict, experiment_a: str, experiment_b: str) -> dict:
    """Normalized shape:
        {experiment_a, experiment_b, primary_population,
         features: [ {feature, a: {auc, ci_low, ci_high, direction},
                       b: {...}, point_direction_reversal, reversal_status,
                       step9e_relationship_direction_flag(bool)} ]}
    """
    schema = payload["schema"]
    report = payload["report"]
    primary_population = report.get("primary_population")
    if primary_population != PRIMARY_POPULATION:
        raise InputResolutionError(
            "step9g_feature_reversal",
            f"Step9G report primary_population='{primary_population}', expected '{PRIMARY_POPULATION}'.",
        )

    if schema == "v2":
        table = report.get("per_feature_integration", [])
    else:
        table = report.get("direction_reversal_table", [])

    features = []
    for row in table:
        prefix_a = _resolve_experiment_prefix(row, experiment_a)
        prefix_b = _resolve_experiment_prefix(row, experiment_b)
        if prefix_a is None or prefix_b is None:
            # Row does not carry AUC fields for one of the two requested
            # experiments -- treat as unavailable for this feature rather
            # than guessing.
            features.append({
                "feature": row.get("feature"),
                "a": None, "b": None,
                "point_direction_reversal": None,
                "reversal_status": None,
                "step9e_relationship_direction_flag": None,
            })
            continue

        def _side(prefix: str) -> dict:
            return {
                "auc": _finite_or_none(row.get(f"{prefix}_auc")),
                "ci_low": _finite_or_none(row.get(f"{prefix}_ci_low")),
                "ci_high": _finite_or_none(row.get(f"{prefix}_ci_high")),
                "direction": row.get(f"{prefix}_direction"),
            }

        flag_raw = row.get("step9e_relationship_direction_flag")
        if flag_raw is None:
            flag_raw = row.get("step9e_rank_effect_direction_flip")
        if isinstance(flag_raw, str):
            flag = flag_raw.strip().lower() == "true"
        else:
            flag = bool(flag_raw) if flag_raw is not None else None

        features.append({
            "feature": row.get("feature"),
            "a": _side(prefix_a),
            "b": _side(prefix_b),
            "point_direction_reversal": row.get("point_direction_reversal"),
            "reversal_status": row.get("reversal_status"),
            "step9e_relationship_direction_flag": flag,
        })

    return {
        "experiment_a": experiment_a,
        "experiment_b": experiment_b,
        "primary_population": PRIMARY_POPULATION,
        "features": features,
    }


# =============================================================================
# 6. Step10 raw/adapted transfer pair report.
# =============================================================================
def adapt_step10_pair(raw: dict, experiment_a: str, experiment_b: str) -> dict:
    """Normalized shape:
        {rows: [ {direction, source_experiment_id, target_experiment_id,
                  model_family, adaptation_method, target_row_count,
                  target_burned_count, roc_auc, roc_auc_ci_low,
                  roc_auc_ci_high, pr_auc, pr_auc_ci_low, pr_auc_ci_high,
                  brier_available} ],
         adaptation_differences: [ {direction, model_family, adaptation_method,
                  point_estimate_difference, ci_low, ci_high} ],
         decomposition: [ {direction, model_family, adaptation_method,
                  within_region_target_metric, remaining_transfer_gap,
                  remaining_transfer_gap_ci_low, remaining_transfer_gap_ci_high,
                  residual_gap_support_status_raw} ]}
    """
    rows = []
    for r in raw.get("target_performance", []):
        roc_ci = r.get("roc_auc_bootstrap_95_percentile_ci") or [None, None]
        pr_ci = r.get("pr_auc_bootstrap_95_percentile_ci") or [None, None]
        rows.append({
            "direction": r.get("direction"),
            "model_family": r.get("model_family"),
            "adaptation_method": r.get("adaptation_method"),
            "target_row_count": r.get("target_row_count"),
            "target_burned_count": r.get("target_burned_count"),
            "roc_auc": _finite_or_none(r.get("roc_auc")),
            "roc_auc_ci_low": _finite_or_none(roc_ci[0] if len(roc_ci) > 0 else None),
            "roc_auc_ci_high": _finite_or_none(roc_ci[1] if len(roc_ci) > 1 else None),
            # Frozen chance-level classification for THIS row's own ROC-AUC
            # (raw or adapted) -- never derived here, only read from the
            # immutable Step10 protocol, and never to be conflated with
            # adapted-minus-raw support.
            "roc_auc_chance_status": r.get("roc_auc_chance_status"),
            "pr_auc": _finite_or_none(r.get("pr_auc")),
            "pr_auc_ci_low": _finite_or_none(pr_ci[0] if len(pr_ci) > 0 else None),
            "pr_auc_ci_high": _finite_or_none(pr_ci[1] if len(pr_ci) > 1 else None),
            # Brier is only ever included when the immutable protocol
            # produced it; a null/"unavailable_in_frozen_outputs" value must
            # never be fabricated into a number downstream.
            "brier_available": r.get("brier_availability") not in (
                "unavailable_in_frozen_outputs", None,
            ) and r.get("brier") is not None,
            "brier": _finite_or_none(r.get("brier")),
            "brier_availability_raw": r.get("brier_availability"),
        })

    adaptation_differences = []
    for r in raw.get("paired_adaptation_differences", []):
        comparison = r.get("comparison") or ""
        if not comparison.endswith("_minus_raw_source_only"):
            continue
        adaptation_method = comparison[: -len("_minus_raw_source_only")]
        ci = r.get("paired_bootstrap_95_percentile_ci") or [None, None]
        adaptation_differences.append({
            "direction": r.get("direction"),
            "model_family": r.get("model_family"),
            "adaptation_method": adaptation_method,
            "metric": r.get("metric"),
            "point_estimate_difference": _finite_or_none(r.get("point_estimate_difference")),
            "ci_low": _finite_or_none(ci[0] if len(ci) > 0 else None),
            "ci_high": _finite_or_none(ci[1] if len(ci) > 1 else None),
        })

    decomposition = []
    for r in raw.get("within_transfer_decomposition", []):
        gap_ci = r.get("remaining_gap_95_paired_bootstrap_ci") or [None, None]
        decomposition.append({
            "direction": r.get("direction"),
            "model_family": r.get("model_family"),
            "adaptation_method": r.get("adapted_method"),
            "metric": r.get("metric"),
            "within_region_target_metric": _finite_or_none(r.get("target_within_region_step8b_oof_metric")),
            "remaining_transfer_gap": _finite_or_none(r.get("remaining_transfer_gap")),
            "remaining_transfer_gap_ci_low": _finite_or_none(gap_ci[0] if len(gap_ci) > 0 else None),
            "remaining_transfer_gap_ci_high": _finite_or_none(gap_ci[1] if len(gap_ci) > 1 else None),
        })

    return {
        "rows": rows,
        "adaptation_differences": adaptation_differences,
        "decomposition": decomposition,
    }
