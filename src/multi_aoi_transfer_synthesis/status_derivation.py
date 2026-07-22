"""Pure functions deriving conservative status labels from numeric
values/confidence intervals.

This module owns the two closed status vocabularies used by the multi-AOI
transfer synthesis:

1. Transfer-matrix / adaptation / residual-gap status labels (spec-mandated
   vocabulary, e.g. `raw_roc_and_pr_supported`, `adaptation_supported`,
   `residual_gap_remains`, ...).
2. Step9E shift-category labels (a small closed vocabulary invented here,
   documented below, derived from the `abs_smd_ge_*` / `psi_ge_*` /
   `outside_*_support_ge_*` boolean flags already present in the frozen
   Step9E per-feature audit rows).

No function here reads a file or knows about any AOI name -- everything is
plain numeric/boolean input -> label output.

Banned phrases (never returned by any function here): "partial_transfer_
supported", "successful operational transfer", "universal adaptation
success", "statistical significance", "causal concept shift".
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# Raw transfer status (per direction x model_family, from the raw/source-only
# ROC-AUC and PR-AUC bootstrap CIs).
# ---------------------------------------------------------------------------
RAW_DISCRIMINATION_NOT_SUPPORTED = "raw_discrimination_not_supported"
RAW_ROC_SUPPORTED_PR_UNCERTAIN = "raw_roc_supported_pr_uncertain"
RAW_ROC_AND_PR_SUPPORTED = "raw_roc_and_pr_supported"

RAW_TRANSFER_STATUSES = (
    RAW_DISCRIMINATION_NOT_SUPPORTED,
    RAW_ROC_SUPPORTED_PR_UNCERTAIN,
    RAW_ROC_AND_PR_SUPPORTED,
)


def _is_finite(x: Optional[float]) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _ci_above_chance(ci_low: Optional[float], chance: float = 0.5) -> Optional[bool]:
    """True if a ROC-AUC CI lower bound is strictly above chance (0.5).
    Returns None if the CI bound is missing/non-finite (caller should treat
    that as "unavailable", never silently substitute)."""
    if not _is_finite(ci_low):
        return None
    return ci_low > chance


def _ci_positive(ci_low: Optional[float]) -> Optional[bool]:
    """True if a CI's lower bound is strictly above zero (bootstrap-supported
    positive effect)."""
    if not _is_finite(ci_low):
        return None
    return ci_low > 0.0


def derive_raw_transfer_status(
    roc_auc_ci_low: Optional[float],
    pr_auc_ci_low: Optional[float],
    pr_auc_chance_ci_low: Optional[float] = None,
) -> str:
    """Derive the raw-transfer status for one (direction, model_family) from
    the raw/source-only ROC-AUC bootstrap CI lower bound and a PR-AUC signal.

    ROC-AUC has a well-defined chance level (0.5); PR-AUC's chance level is
    prevalence-dependent, so PR-AUC support here is a conservative proxy:
    "PR-AUC not uncertain" is approximated by the PR-AUC CI lower bound
    being finite (present); this function never fabricates a PR-AUC chance
    threshold that was not supplied by the caller. If `pr_auc_chance_ci_low`
    is supplied (e.g. a paired comparison against a chance baseline), it is
    used instead.
    """
    roc_above_chance = _ci_above_chance(roc_auc_ci_low)
    if roc_above_chance is None or not roc_above_chance:
        return RAW_DISCRIMINATION_NOT_SUPPORTED

    if pr_auc_chance_ci_low is not None:
        pr_supported = _ci_positive(pr_auc_chance_ci_low)
    else:
        pr_supported = _is_finite(pr_auc_ci_low)

    if pr_supported:
        return RAW_ROC_AND_PR_SUPPORTED
    return RAW_ROC_SUPPORTED_PR_UNCERTAIN


# ---------------------------------------------------------------------------
# Adaptation-effect status (adapted_minus_raw paired-bootstrap CI).
# ---------------------------------------------------------------------------
ADAPTATION_SUPPORTED = "adaptation_supported"
ADAPTATION_DEGRADED_TRANSFER = "adaptation_degraded_transfer"
ADAPTATION_EFFECT_UNCERTAIN = "adaptation_effect_uncertain"

ADAPTATION_SUPPORT_STATUSES = (
    ADAPTATION_SUPPORTED,
    ADAPTATION_DEGRADED_TRANSFER,
    ADAPTATION_EFFECT_UNCERTAIN,
)


def derive_adaptation_support_status(
    ci_low: Optional[float], ci_high: Optional[float]
) -> str:
    """Derive whether an adapted-vs-raw paired difference is a supported
    improvement, a supported degradation, or uncertain, from its paired
    bootstrap CI."""
    if not _is_finite(ci_low) or not _is_finite(ci_high):
        return ADAPTATION_EFFECT_UNCERTAIN
    if ci_low > 0.0:
        return ADAPTATION_SUPPORTED
    if ci_high < 0.0:
        return ADAPTATION_DEGRADED_TRANSFER
    return ADAPTATION_EFFECT_UNCERTAIN


# ---------------------------------------------------------------------------
# Residual within-region gap status (remaining_transfer_gap paired CI).
# ---------------------------------------------------------------------------
RESIDUAL_GAP_REMAINS = "residual_gap_remains"
RESIDUAL_GAP_UNCERTAIN = "residual_gap_uncertain"
RESIDUAL_GAP_NOT_SUPPORTED = "residual_gap_not_supported"

RESIDUAL_GAP_STATUSES = (
    RESIDUAL_GAP_REMAINS,
    RESIDUAL_GAP_UNCERTAIN,
    RESIDUAL_GAP_NOT_SUPPORTED,
)


def derive_residual_gap_status(ci_low: Optional[float], ci_high: Optional[float]) -> str:
    """Derive whether a remaining (post-adaptation) within-region-vs-target
    performance gap is bootstrap-supported as remaining, not supported, or
    uncertain."""
    if not _is_finite(ci_low) or not _is_finite(ci_high):
        return RESIDUAL_GAP_UNCERTAIN
    if ci_low > 0.0:
        return RESIDUAL_GAP_REMAINS
    if ci_high < 0.0:
        return RESIDUAL_GAP_NOT_SUPPORTED
    return RESIDUAL_GAP_UNCERTAIN


# ---------------------------------------------------------------------------
# Recovery-pattern status across the two directions of an unordered pair.
# ---------------------------------------------------------------------------
BIDIRECTIONAL_RECOVERY = "bidirectional_recovery"
DIRECTION_DEPENDENT_RECOVERY = "direction_dependent_recovery"
NO_RECOVERY_SUPPORTED = "no_recovery_supported"

RECOVERY_PATTERN_STATUSES = (
    BIDIRECTIONAL_RECOVERY,
    DIRECTION_DEPENDENT_RECOVERY,
    NO_RECOVERY_SUPPORTED,
)


def derive_recovery_pattern_status(direction_statuses: Sequence[str]) -> str:
    """Given the `adaptation_support_status` values for the two directions
    of an unordered pair (for one model_family/adaptation_method), derive
    whether recovery is bidirectional, direction-dependent, or unsupported
    in either direction."""
    supported = [s == ADAPTATION_SUPPORTED for s in direction_statuses]
    if all(supported) and len(supported) > 0:
        return BIDIRECTIONAL_RECOVERY
    if any(supported):
        return DIRECTION_DEPENDENT_RECOVERY
    return NO_RECOVERY_SUPPORTED


# ---------------------------------------------------------------------------
# Step9E shift-category vocabulary (invented here; a small closed set rolled
# up from the per-feature `abs_smd_ge_*` / `psi_ge_*` /
# `outside_*_support_ge_*` boolean flags already present in the frozen
# Step9E numeric-feature-shift rows).
# ---------------------------------------------------------------------------
LARGE_COVARIATE_SHIFT_DETECTED = "large_covariate_shift_detected"
MODERATE_COVARIATE_SHIFT_DETECTED = "moderate_covariate_shift_detected"
DISTRIBUTION_SUPPORT_MISMATCH_DETECTED = "distribution_support_mismatch_detected"
NO_MATERIAL_SHIFT_DETECTED = "no_material_shift_detected"
SHIFT_AUDIT_UNAVAILABLE = "shift_audit_unavailable"

SHIFT_CATEGORY_VOCABULARY = (
    LARGE_COVARIATE_SHIFT_DETECTED,
    MODERATE_COVARIATE_SHIFT_DETECTED,
    DISTRIBUTION_SUPPORT_MISMATCH_DETECTED,
    NO_MATERIAL_SHIFT_DETECTED,
    SHIFT_AUDIT_UNAVAILABLE,
)


def derive_shift_categories(numeric_feature_rows: Sequence[dict]) -> list[str]:
    """Roll up per-feature Step9E numeric-shift boolean flags for one
    (source, target) direction into a small set of shift-category labels.

    Each row in `numeric_feature_rows` is expected to carry (when present)
    the boolean flags `abs_smd_ge_0_5`, `abs_smd_ge_0_2`, `psi_ge_0_25`,
    `psi_ge_0_10`, `outside_source_support_ge_0_10` -- exactly as emitted by
    the frozen Step9E per-feature audit. Missing flags are treated as
    "not triggered" (conservative: never inferred as a large shift from
    absent data).

    Returns a non-empty, order-stable list of category labels.
    """
    if not numeric_feature_rows:
        return [SHIFT_AUDIT_UNAVAILABLE]

    any_large_smd = any(bool(r.get("abs_smd_ge_0_5")) for r in numeric_feature_rows)
    any_moderate_smd = any(bool(r.get("abs_smd_ge_0_2")) for r in numeric_feature_rows)
    any_large_psi = any(bool(r.get("psi_ge_0_25")) for r in numeric_feature_rows)
    any_moderate_psi = any(bool(r.get("psi_ge_0_10")) for r in numeric_feature_rows)
    any_support_mismatch = any(
        bool(r.get("outside_source_support_ge_0_10")) for r in numeric_feature_rows
    )

    categories: list[str] = []
    if any_large_smd or any_large_psi:
        categories.append(LARGE_COVARIATE_SHIFT_DETECTED)
    elif any_moderate_smd or any_moderate_psi:
        categories.append(MODERATE_COVARIATE_SHIFT_DETECTED)

    if any_support_mismatch:
        categories.append(DISTRIBUTION_SUPPORT_MISMATCH_DETECTED)

    if not categories:
        categories.append(NO_MATERIAL_SHIFT_DETECTED)
    return categories


def derive_ranking_reversal_suspected(numeric_feature_rows: Sequence[dict]) -> bool:
    """Conservative boolean: True only if the Step9E audit rows for this
    direction indicate at least one feature with both a large covariate
    shift AND an outside-support-mismatch signal (a pattern consistent with,
    not proof of, a ranking/relationship reversal risk)."""
    if not numeric_feature_rows:
        return False
    for row in numeric_feature_rows:
        large_shift = bool(row.get("abs_smd_ge_0_5")) or bool(row.get("psi_ge_0_25"))
        support_mismatch = bool(row.get("outside_source_support_ge_0_10"))
        if large_shift and support_mismatch:
            return True
    return False


# ---------------------------------------------------------------------------
# Step9E frozen diagnosis categories (`part_f_summary.diagnosis_categories`,
# already computed by the frozen Step9E protocol -- this is the CLOSED
# vocabulary that protocol actually emits, re-declared here only for
# validation/documentation, never recomputed).
# ---------------------------------------------------------------------------
HIGH_SHIFT = "high_shift"
MODERATE_SHIFT = "moderate_shift"
LOW_SHIFT = "low_shift"
PROBABILITY_SCALE_SHIFT = "probability_scale_shift"
RELATIONSHIP_DIRECTION_INSTABILITY = "relationship_direction_instability"
RANKING_REVERSAL_SUSPECTED_DIAGNOSIS = "ranking_reversal_suspected"

FROZEN_SHIFT_DIAGNOSIS_VOCABULARY = (
    HIGH_SHIFT,
    MODERATE_SHIFT,
    LOW_SHIFT,
    PROBABILITY_SCALE_SHIFT,
    RELATIONSHIP_DIRECTION_INSTABILITY,
    RANKING_REVERSAL_SUSPECTED_DIAGNOSIS,
)


def merge_shift_categories(
    locally_derived_categories: Sequence[str], frozen_diagnosis_categories: Sequence[str],
) -> list[str]:
    """Union the locally-derived SMD/PSI/support-mismatch categories (see
    `derive_shift_categories`) with the Step9E-frozen `part_f_summary.
    diagnosis_categories` labels, preserving every distinct diagnosis
    instead of collapsing them into one or two generic buckets. Unknown
    frozen labels are dropped rather than silently reinterpreted, using
    `FROZEN_SHIFT_DIAGNOSIS_VOCABULARY` as an allowlist. Once at least one
    real diagnosis is present, the "nothing found" placeholders
    (`SHIFT_AUDIT_UNAVAILABLE`, `NO_MATERIAL_SHIFT_DETECTED`) are dropped so
    they never sit alongside an actual finding."""
    merged: list[str] = []
    for category in locally_derived_categories:
        if category not in merged:
            merged.append(category)
    for category in frozen_diagnosis_categories or []:
        if category in FROZEN_SHIFT_DIAGNOSIS_VOCABULARY and category not in merged:
            merged.append(category)
    if len(merged) > 1:
        merged = [c for c in merged if c not in (SHIFT_AUDIT_UNAVAILABLE, NO_MATERIAL_SHIFT_DETECTED)]
    return merged


# ---------------------------------------------------------------------------
# Ranking-reversal value/scope (Step9E-frozen, never recomputed). A
# direction-specific signal (from `part_d_prediction_distribution_audit`
# rows, which carry their own `transfer_direction`) is preferred; a
# pair-global signal (`part_f_summary.ranking_reversal_suspected`, which
# summarizes "any direction/model" for the whole pair) is used only when no
# direction-specific record is available, and is tagged with a scope that
# must never be read as proven for one specific ordered direction.
# ---------------------------------------------------------------------------
RANKING_REVERSAL_SCOPE_DIRECTION_SPECIFIC = "direction_specific"
RANKING_REVERSAL_SCOPE_PAIR_GLOBAL = "pair_global_any_direction_or_model"

RANKING_REVERSAL_SCOPES = (
    RANKING_REVERSAL_SCOPE_DIRECTION_SPECIFIC,
    RANKING_REVERSAL_SCOPE_PAIR_GLOBAL,
)


def derive_ranking_reversal(
    direction_specific_rows: Sequence[dict], pair_global_value: Optional[bool],
) -> tuple[Optional[bool], Optional[str]]:
    """Returns (ranking_reversal_value, ranking_reversal_scope). Both are
    None only when neither a direction-specific nor a pair-global signal is
    available -- never fabricated, never defaulted to False."""
    if direction_specific_rows:
        value = any(bool(r.get("target_ranking_reversal_suspected")) for r in direction_specific_rows)
        return value, RANKING_REVERSAL_SCOPE_DIRECTION_SPECIFIC
    if pair_global_value is not None:
        return bool(pair_global_value), RANKING_REVERSAL_SCOPE_PAIR_GLOBAL
    return None, None


# ---------------------------------------------------------------------------
# ROC-AUC chance-level status (Step10-frozen `roc_auc_chance_status`, from
# `classify_chance_status` in step10d_final_report.py -- never recomputed).
# Applies uniformly to every transfer-matrix row (raw AND adapted): a row's
# own ROC-AUC being above chance is a DIFFERENT claim from adapted-minus-raw
# being a supported improvement, and the two must never be conflated.
# ---------------------------------------------------------------------------
ROC_AUC_CHANCE_STATUS_ABOVE = "bootstrap_supported_above_chance"
ROC_AUC_CHANCE_STATUS_BELOW = "bootstrap_supported_below_chance"
ROC_AUC_CHANCE_STATUS_NOT_EXCLUDED = "chance_level_not_excluded"
ROC_AUC_CHANCE_STATUS_UNAVAILABLE = "unavailable"

ROC_AUC_CHANCE_STATUS_VOCABULARY = (
    ROC_AUC_CHANCE_STATUS_ABOVE,
    ROC_AUC_CHANCE_STATUS_BELOW,
    ROC_AUC_CHANCE_STATUS_NOT_EXCLUDED,
    ROC_AUC_CHANCE_STATUS_UNAVAILABLE,
)


def bucket_roc_auc_chance_status(value: Optional[str]) -> str:
    """Validate/normalize a frozen Step10 `roc_auc_chance_status` value
    against its known closed vocabulary; unknown or missing values map to
    `unavailable` rather than being silently reinterpreted."""
    if value in ROC_AUC_CHANCE_STATUS_VOCABULARY:
        return value
    return ROC_AUC_CHANCE_STATUS_UNAVAILABLE


# ---------------------------------------------------------------------------
# Step9G feature-AUC direction-reversal buckets (closed vocabulary already
# defined by the frozen Step9G schema; re-exported here so all downstream
# consumers import the vocabulary from one place).
# ---------------------------------------------------------------------------
NO_DIRECTION_REVERSAL = "no_direction_reversal"
BOOTSTRAP_SUPPORTED_DIRECTION_REVERSAL = "bootstrap_supported_direction_reversal"
POINT_DIRECTION_REVERSAL_INTERVAL_UNCERTAIN = "point_direction_reversal_interval_uncertain"
FEATURE_UNAVAILABLE = "feature_unavailable"

REVERSAL_STATUS_VOCABULARY = (
    NO_DIRECTION_REVERSAL,
    BOOTSTRAP_SUPPORTED_DIRECTION_REVERSAL,
    POINT_DIRECTION_REVERSAL_INTERVAL_UNCERTAIN,
    FEATURE_UNAVAILABLE,
)


def bucket_reversal_status(reversal_status: Optional[str]) -> str:
    """Normalize/validate a frozen Step9G `reversal_status` value into the
    known closed vocabulary; unknown/missing values map to
    `feature_unavailable` rather than being silently reinterpreted (e.g.
    never describe an uncertain point reversal as bootstrap-supported)."""
    if reversal_status in REVERSAL_STATUS_VOCABULARY:
        return reversal_status
    return FEATURE_UNAVAILABLE
