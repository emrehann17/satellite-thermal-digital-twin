"""Frozen contract for the multi-region window-closure sensitivity analysis.

This module is the SINGLE source of the AOI scope, the variant set, the model
families, the metrics, the estimands, the metric orientations and the fit
accounting. No other module in this package re-declares any of them; they are
imported from here so a scope list cannot drift between two files.

Nothing here touches Earth Engine, the filesystem or any model library.

Design references
-----------------
docs/multi_region_window_closure_design/SCIENTIFIC_CONTRACT.md
docs/multi_region_window_closure_design/README.md  (authoritative counts)
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

SCHEMA_VERSION = "multi_region_window_closure.v1"
REGIONAL_SCHEMA_VERSION = "window_closure_region.v1"
SYNTHESIS_SCHEMA_VERSION = "window_closure_synthesis.v1"
SCIENTIFIC_CONTRACT_ID = "window_closure_scientific_contract.v1"
DIAGNOSTIC_NAMESPACE = "multi_region_window_closure"
REGIONAL_DIAGNOSTIC_NAMESPACE = "window_closure_region"
SYNTHESIS_DIAGNOSTIC_NAMESPACE = "window_closure_synthesis"


class MultiRegionWindowClosureError(SystemExit):
    """Fail-closed error. Mirrors `window_closure_sensitivity.WindowClosureError`."""


# =============================================================================
# AOI scope -- the ONLY declaration in the codebase
# =============================================================================
#: The four AOIs this analysis actually runs, in canonical (sorted) order.
ACTUAL_AOIS: tuple[str, ...] = (
    "bejis_2022", "evia_2021_extended", "montiferru_2021", "mugla_2021",
)

#: Read-only reference AOI. Never refit, never re-exported, never overwritten.
REFERENCE_AOI = "manavgat_2021"

# Compatibility-only constants consumed by shared schema/production helpers.
# No synthesis command, orchestrator, validator, or production entry point is
# registered after the per-AOI architecture cleanup.
SYNTHESIS_AOIS: tuple[str, ...] = tuple(sorted(ACTUAL_AOIS + (REFERENCE_AOI,)))

#: Superseded narrow North Evia AOI. Must appear nowhere in this analysis.
EXCLUDED_AOIS: tuple[str, ...] = ("evia_2021",)

# =============================================================================
# Retired standalone per-AOI output namespace
# =============================================================================
#: The historical per-AOI window-closure OUTPUT namespace, retired 2026-08-10
#: when all five AOIs were unified under `window_closure_region`. The Python
#: module of the same name was NOT retired -- it is still the scientific backend
#: behind `ProductionRegionalEngine`. Only the public/manual entry points that
#: would recreate this output root are blocked.
RETIRED_DIAGNOSTIC_NAMESPACE = "window_closure_sensitivity"
RETIRED_CLI_COMMAND = "window-closure-sensitivity"
CURRENT_CLI_COMMAND = "window-closure-region"

RETIRED_CLI_MESSAGE = (
    f"{RETIRED_CLI_COMMAND} is retired.\n"
    f"Use {CURRENT_CLI_COMMAND} for current per-AOI window-closure outputs.\n"
    f"Manavgat is preserved as the read-only reference AOI under the canonical "
    f"regional output family "
    f"outputs/diagnostics/{REGIONAL_DIAGNOSTIC_NAMESPACE}/{REFERENCE_AOI}/.\n"
    f"The retired output namespace outputs/diagnostics/"
    f"{RETIRED_DIAGNOSTIC_NAMESPACE}/ no longer exists and must not be "
    f"recreated; nothing was run and nothing was written.\n"
    f"src/window_closure_sensitivity.py itself is NOT retired -- it remains the "
    f"scientific backend of the regional runner."
)

#: The different-regime control. Carries mandatory framing (see wording.py).
DIFFERENT_REGIME_CONTROL_AOI = "evia_2021_extended"

AOI_ROLE_NEW_ACTUAL = "new_actual"
AOI_ROLE_DIFFERENT_REGIME_CONTROL = "different_regime_control"
AOI_ROLE_READ_ONLY_REFERENCE = "read_only_reference"


def aoi_role(aoi: str) -> str:
    """Role of an AOI inside this analysis. Fails closed on an unknown AOI."""
    if aoi == REFERENCE_AOI:
        return AOI_ROLE_READ_ONLY_REFERENCE
    if aoi == DIFFERENT_REGIME_CONTROL_AOI:
        return AOI_ROLE_DIFFERENT_REGIME_CONTROL
    if aoi in ACTUAL_AOIS:
        return AOI_ROLE_NEW_ACTUAL
    raise MultiRegionWindowClosureError(
        f"'{aoi}' is not part of this analysis. Actual AOIs are "
        f"{list(ACTUAL_AOIS)}; the read-only reference is '{REFERENCE_AOI}'."
    )


def assert_aoi_scope(aois: Optional[Iterable[str]] = None) -> tuple[str, ...]:
    """Exactly the four actual AOIs, in canonical order. Fails closed.

    A subset, a superset, a duplicate or the excluded old Evia AOI are all
    refused here rather than being silently normalised, because a partial AOI
    set must never be able to reach an actual run.
    """
    if aois is None:
        return ACTUAL_AOIS
    requested = [str(a).strip() for a in aois if str(a).strip()]
    if not requested:
        raise MultiRegionWindowClosureError(
            "No AOI was requested. This analysis runs exactly "
            f"{len(ACTUAL_AOIS)} AOIs: {list(ACTUAL_AOIS)}."
        )
    duplicates = sorted({a for a in requested if requested.count(a) > 1})
    if duplicates:
        raise MultiRegionWindowClosureError(
            f"Duplicate AOI(s) requested: {duplicates}."
        )
    excluded = sorted(set(requested) & set(EXCLUDED_AOIS))
    if excluded:
        raise MultiRegionWindowClosureError(
            f"BLOCKER: EXCLUDED_AOI_PRESENT -- {excluded} may not enter this "
            "analysis. The superseded narrow North Evia AOI 'evia_2021' is "
            "excluded from the scope, the export plan, the fit plan, the "
            "bootstrap plan and the four-region synthesis. Use "
            "'evia_2021_extended'."
        )
    if REFERENCE_AOI in requested:
        raise MultiRegionWindowClosureError(
            f"'{REFERENCE_AOI}' is the READ-ONLY reference AOI and is never "
            "recomputed. Remove it from --aois; it is included automatically "
            "in the descriptive synthesis."
        )
    unknown = sorted(set(requested) - set(ACTUAL_AOIS))
    if unknown:
        raise MultiRegionWindowClosureError(
            f"Unknown AOI(s) for this analysis: {unknown}. Expected exactly "
            f"{list(ACTUAL_AOIS)}."
        )
    missing = sorted(set(ACTUAL_AOIS) - set(requested))
    if missing:
        raise MultiRegionWindowClosureError(
            f"BLOCKER: PARTIAL_AOI -- missing {missing}. All "
            f"{len(ACTUAL_AOIS)} AOIs must run together; a subset is never a "
            "valid scope for this analysis."
        )
    return ACTUAL_AOIS


# -----------------------------------------------------------------------------
# Reference-replay scope -- narrow, explicit, fail-closed
# -----------------------------------------------------------------------------
#: Active reference-replay AOI, or None. A context variable rather than a module
#: global so a replay can never leak out of the block that opened it, and so a
#: concurrent normal regional run in another task cannot observe it.
#:
#: This exists for exactly ONE purpose: replaying the already-frozen Manavgat
#: window-closure result through the per-AOI regional architecture in order to
#: compare the two. It admits `REFERENCE_AOI` and nothing else, it never widens
#: `ACTUAL_AOIS`, and it never changes what `aoi_role` reports -- Manavgat stays
#: `read_only_reference` throughout. The production CLI
#: (`scripts/run_window_closure_region.py`) constrains `--experiment` to
#: `ACTUAL_AOIS` via argparse and never opens this scope, so a normal regional
#: run cannot reach it even by accident.
_REFERENCE_REPLAY_AOI: "ContextVar[Optional[str]]" = None  # set below


def _init_reference_replay_scope() -> None:
    global _REFERENCE_REPLAY_AOI
    from contextvars import ContextVar

    _REFERENCE_REPLAY_AOI = ContextVar("multi_region_reference_replay_aoi", default=None)


_init_reference_replay_scope()


def active_reference_replay_aoi() -> Optional[str]:
    """The AOI a reference replay has explicitly opened, or None."""
    return _REFERENCE_REPLAY_AOI.get()


def reference_replay_scope(aoi: str):
    """Context manager admitting ONLY `REFERENCE_AOI` to the regional gate.

    Any other AOI -- including an excluded or unknown one -- is refused here,
    so the scope cannot be repurposed into a general allow-list.
    """
    from contextlib import contextmanager

    @contextmanager
    def _scope():
        value = str(aoi).strip()
        if value != REFERENCE_AOI:
            raise MultiRegionWindowClosureError(
                "BLOCKER: REFERENCE_REPLAY_SCOPE_INVALID -- a reference replay "
                f"may only be opened for '{REFERENCE_AOI}', not '{value}'."
            )
        token = _REFERENCE_REPLAY_AOI.set(REFERENCE_AOI)
        try:
            yield REFERENCE_AOI
        finally:
            _REFERENCE_REPLAY_AOI.reset(token)

    return _scope()


def assert_regional_aoi(aoi: str) -> str:
    """Validate the single-AOI execution scope without coupling AOI statuses."""
    value = str(aoi).strip()
    if value in EXCLUDED_AOIS:
        raise MultiRegionWindowClosureError(
            "BLOCKER: EXCLUDED_AOI_PRESENT -- 'evia_2021' is superseded; "
            "use 'evia_2021_extended'."
        )
    if value == REFERENCE_AOI:
        if active_reference_replay_aoi() == REFERENCE_AOI:
            return value
        raise MultiRegionWindowClosureError(
            f"'{REFERENCE_AOI}' is a read-only synthesis reference and cannot "
            "be used for a regional actual run."
        )
    if value not in ACTUAL_AOIS:
        raise MultiRegionWindowClosureError(
            f"Unknown regional AOI '{value}'. Expected one of {list(ACTUAL_AOIS)}."
        )
    return value


# =============================================================================
# Variants, models, metrics
# =============================================================================
CANONICAL_VARIANT_ID = "canonical"

#: Preregistered closure shifts in days. Imported from the per-AOI module so
#: the two analyses cannot disagree; re-declared here only as a fallback for
#: environments where the heavy module is unavailable.
DEFAULT_SHIFTS: tuple[int, ...] = (0, 7, 14)

VARIANTS: tuple[str, ...] = ("canonical", "close_7d_earlier", "close_14d_earlier")
SHIFTED_VARIANTS: tuple[str, ...] = ("close_7d_earlier", "close_14d_earlier")

MODEL_FAMILIES: tuple[str, ...] = ("baseline", "thermal")
METRICS: tuple[str, ...] = ("roc_auc", "pr_auc", "brier")

PRIMARY_POPULATION = "burnable_tree_shrub_grass"
PRIMARY_MODEL = "random_forest"

N_FOLDS = 5

METRIC_DIRECTION: dict[str, str] = {
    "roc_auc": "higher_is_better",
    "pr_auc": "higher_is_better",
    "brier": "lower_is_better",
}

#: Metrics whose two orientations are equal by definition.
ORIENTATION_SYMMETRIC_METRICS: frozenset = frozenset({"roc_auc", "pr_auc"})


def variant_id(shift_days: int) -> str:
    """`canonical` for shift 0, else `close_<n>d_earlier`. Matches the per-AOI module."""
    shift = int(shift_days)
    return CANONICAL_VARIANT_ID if shift == 0 else f"close_{shift}d_earlier"


def shift_days_of(variant: str) -> int:
    """Inverse of :func:`variant_id`. Fails closed on an unknown variant."""
    if variant == CANONICAL_VARIANT_ID:
        return 0
    for shift in DEFAULT_SHIFTS:
        if shift and variant_id(shift) == variant:
            return int(shift)
    raise MultiRegionWindowClosureError(
        f"Unknown variant '{variant}'. Expected one of {list(VARIANTS)}."
    )


def assert_variant_scope(variants: Sequence[str]) -> tuple[str, ...]:
    """Exactly the three frozen variants. A subset is never valid."""
    got = tuple(variants)
    if sorted(got) != sorted(VARIANTS):
        raise MultiRegionWindowClosureError(
            f"BLOCKER: VARIANT_SET_MISMATCH -- got {list(got)}, expected "
            f"exactly {list(VARIANTS)}."
        )
    return VARIANTS


# =============================================================================
# Estimands and comparison families
# =============================================================================
COMPARISON_THERMAL_CONTRIBUTION = "thermal_contribution_within_variant"
COMPARISON_CLOSURE_CHANGE = "closure_change_within_model_family"
COMPARISON_CONTRIBUTION_CHANGE = "thermal_contribution_change"

COMPARISON_FAMILIES: tuple[str, ...] = (
    COMPARISON_THERMAL_CONTRIBUTION,
    COMPARISON_CLOSURE_CHANGE,
    COMPARISON_CONTRIBUTION_CHANGE,
)

#: The primary estimand. Everything else is secondary or descriptive.
PRIMARY_ESTIMAND = {
    "comparison_family": COMPARISON_THERMAL_CONTRIBUTION,
    "metric": "roc_auc",
    "definition": "thermal ROC-AUC - baseline ROC-AUC",
}

#: `thermal_contribution_change` is an EXISTING component of the frozen
#: Manavgat contract (verified: 6 of the 27 rows of its
#: paired_bootstrap_summary.csv, constant COMPARISON_CONTRIBUTION_CHANGE in
#: src/window_closure_sensitivity.py). It is carried so the new AOIs emit the
#: same artefact set as the reference AOI -- but it is NOT the primary
#: estimand and may never drive a "best" or "optimal" window claim.
ESTIMAND_CLASSIFICATION: dict[str, str] = {
    COMPARISON_THERMAL_CONTRIBUTION: "primary_and_secondary_estimand",
    COMPARISON_CLOSURE_CHANGE: "secondary_estimand",
    COMPARISON_CONTRIBUTION_CHANGE: "additional_descriptive_secondary_estimand",
}

CONTRIBUTION_CHANGE_NOTES: tuple[str, ...] = (
    "additional_descriptive_secondary_estimand",
    "existing_manavgat_contract_component",
    "not the primary estimand",
    "not a new post-hoc selection criterion",
)


# =============================================================================
# Metric orientation -- natural vs oriented, never silently flipped
# =============================================================================
ORIENTATION_NATURAL = "natural"
ORIENTATION_ORIENTED = "oriented"
ORIENTATIONS: tuple[str, ...] = (ORIENTATION_NATURAL, ORIENTATION_ORIENTED)


def orientation_definition(comparison_family: str, metric: str, orientation: str) -> str:
    """Human-readable definition of one orientation of one comparison.

    Emitted alongside every stored value so a Brier number can never appear
    without the sign convention that makes it interpretable.
    """
    if orientation not in ORIENTATIONS:
        raise MultiRegionWindowClosureError(
            f"Unknown orientation '{orientation}'; expected one of {list(ORIENTATIONS)}."
        )
    loss = metric == "brier"
    if comparison_family == COMPARISON_THERMAL_CONTRIBUTION:
        if orientation == ORIENTATION_NATURAL:
            return (
                f"thermal {metric} - baseline {metric}"
                + (" (negative favours thermal)" if loss else " (positive favours thermal)")
            )
        return (
            f"baseline {metric} - thermal {metric} (positive favours thermal)"
            if loss
            else f"thermal {metric} - baseline {metric} (positive favours thermal)"
        )
    if comparison_family == COMPARISON_CLOSURE_CHANGE:
        if orientation == ORIENTATION_NATURAL:
            return (
                f"shifted {metric} - canonical {metric}"
                + (" (negative means the shifted window scored a lower loss)" if loss
                   else " (positive means the shifted window scored higher)")
            )
        return (
            f"canonical {metric} - shifted {metric} "
            "(positive means the shifted window produced a lower Brier)"
            if loss
            else f"shifted {metric} - canonical {metric} "
                 "(positive means the shifted window scored higher)"
        )
    if comparison_family == COMPARISON_CONTRIBUTION_CHANGE:
        base = (
            f"(thermal - baseline)_shifted {metric} - "
            f"(thermal - baseline)_canonical {metric}"
        )
        if orientation == ORIENTATION_NATURAL:
            return base + (" (negative favours the shifted window)" if loss
                           else " (positive favours the shifted window)")
        return (
            f"(baseline - thermal)_canonical {metric} - "
            f"(baseline - thermal)_shifted {metric} "
            "(positive favours the shifted window)"
            if loss else base + " (positive favours the shifted window)"
        )
    raise MultiRegionWindowClosureError(
        f"Unknown comparison family '{comparison_family}'."
    )


def orient(metric: str, difference_natural: float) -> float:
    """Convert a natural difference to its improvement-oriented counterpart.

    ROC-AUC / PR-AUC: identity -- higher is better in both readings.
    Brier:            negation -- it is a loss, so a lower value is better.

    The sign flip is explicit and lives in exactly one function, so it can be
    unit-tested rather than scattered across call sites.
    """
    if metric not in METRICS:
        raise MultiRegionWindowClosureError(
            f"Unknown metric '{metric}'; expected one of {list(METRICS)}."
        )
    if metric in ORIENTATION_SYMMETRIC_METRICS:
        return float(difference_natural)
    return -float(difference_natural)


def orient_interval(
    metric: str, ci_low_natural: float, ci_high_natural: float,
) -> tuple[float, float]:
    """Oriented interval bounds for a natural interval.

    For a loss metric the interval is NEGATED, which REVERSES its endpoints.
    Forgetting the swap silently produces `ci_low > ci_high`, so the swap is
    performed here and asserted.
    """
    low, high = float(ci_low_natural), float(ci_high_natural)
    if low > high:
        raise MultiRegionWindowClosureError(
            f"Natural interval is inverted: [{low}, {high}]."
        )
    if metric in ORIENTATION_SYMMETRIC_METRICS:
        return low, high
    return -high, -low


def orientations_equal(metric: str) -> bool:
    """True when the two orientations coincide by definition."""
    return metric in ORIENTATION_SYMMETRIC_METRICS


# =============================================================================
# Interval classification -- reuses the frozen per-AOI vocabulary
# =============================================================================
INTERVAL_SUPPORTED_INCREASE = "bootstrap_supported_increase"
INTERVAL_SUPPORTED_DECREASE = "bootstrap_supported_decrease"
INTERVAL_INCLUDES_ZERO = "interval_includes_zero"


def classify_interval(ci_low: Optional[float], ci_high: Optional[float]) -> str:
    """Three-way interval status. An interval spanning zero resolves nothing."""
    if ci_low is None or ci_high is None:
        return INTERVAL_INCLUDES_ZERO
    if ci_low > 0:
        return INTERVAL_SUPPORTED_INCREASE
    if ci_high < 0:
        return INTERVAL_SUPPORTED_DECREASE
    return INTERVAL_INCLUDES_ZERO


def interval_excludes_zero(ci_low: Optional[float], ci_high: Optional[float]) -> bool:
    return classify_interval(ci_low, ci_high) != INTERVAL_INCLUDES_ZERO


# =============================================================================
# Fit accounting -- two DISJOINT families with different primary keys
# =============================================================================
#: (aoi, variant, model, fold)
EXPECTED_PRIMARY_ESTIMATOR_FITS = len(ACTUAL_AOIS) * len(VARIANTS) * len(MODEL_FAMILIES) * N_FOLDS

#: (aoi, shifted_variant, downstream_stage_or_model_role) -- one Step7C
#: downscaling model per SHIFTED variant. The canonical arm reuses the frozen
#: production Step7C output and therefore fits nothing.
AUXILIARY_DOWNSCALING_STAGE = "step7c"
EXPECTED_AUXILIARY_DOWNSCALING_FITS = len(ACTUAL_AOIS) * len(SHIFTED_VARIANTS)

EXPECTED_TOTAL_FITS = EXPECTED_PRIMARY_ESTIMATOR_FITS + EXPECTED_AUXILIARY_DOWNSCALING_FITS

# Per-invocation accounting. The legacy set-level constants above remain only
# for reading old design artefacts; new execution code uses these values.
REGIONAL_EXPECTED_PRIMARY_ESTIMATOR_FITS = len(VARIANTS) * len(MODEL_FAMILIES) * N_FOLDS
REGIONAL_EXPECTED_AUXILIARY_DOWNSCALING_FITS = len(SHIFTED_VARIANTS)
REGIONAL_EXPECTED_TOTAL_FITS = (
    REGIONAL_EXPECTED_PRIMARY_ESTIMATOR_FITS
    + REGIONAL_EXPECTED_AUXILIARY_DOWNSCALING_FITS
)

PRIMARY_FIT_KEY: tuple[str, ...] = ("aoi", "variant", "model", "fold")
AUXILIARY_FIT_KEY: tuple[str, ...] = (
    "aoi", "shifted_variant", "downstream_stage_or_model_role",
)


def expected_primary_fit_keys() -> list[tuple[str, str, str, int]]:
    """Every `(aoi, variant, model, fold)` the fit stage must produce."""
    return [
        (aoi, variant, model, fold)
        for aoi in ACTUAL_AOIS
        for variant in VARIANTS
        for model in MODEL_FAMILIES
        for fold in range(N_FOLDS)
    ]


def expected_auxiliary_fit_keys() -> list[tuple[str, str, str]]:
    """Every `(aoi, shifted_variant, stage)` Step7C downscaling fit."""
    return [
        (aoi, variant, AUXILIARY_DOWNSCALING_STAGE)
        for aoi in ACTUAL_AOIS
        for variant in SHIFTED_VARIANTS
    ]


def expected_regional_primary_fit_keys(aoi: str) -> list[tuple[str, str, str, int]]:
    selected = assert_regional_aoi(aoi)
    return [
        (selected, variant, model, fold)
        for variant in VARIANTS
        for model in MODEL_FAMILIES
        for fold in range(N_FOLDS)
    ]


def expected_regional_auxiliary_fit_keys(aoi: str) -> list[tuple[str, str, str]]:
    selected = assert_regional_aoi(aoi)
    return [
        (selected, variant, AUXILIARY_DOWNSCALING_STAGE)
        for variant in SHIFTED_VARIANTS
    ]


def regional_fit_accounting_template() -> dict[str, int]:
    return {
        "expected_primary_estimator_fits": REGIONAL_EXPECTED_PRIMARY_ESTIMATOR_FITS,
        "completed_primary_estimator_fits": 0,
        "duplicate_primary_estimator_fits": 0,
        "missing_primary_estimator_fits": REGIONAL_EXPECTED_PRIMARY_ESTIMATOR_FITS,
        "unexpected_primary_estimator_fits": 0,
        "expected_auxiliary_downscaling_fits": REGIONAL_EXPECTED_AUXILIARY_DOWNSCALING_FITS,
        "completed_auxiliary_downscaling_fits": 0,
        "duplicate_auxiliary_downscaling_fits": 0,
        "missing_auxiliary_downscaling_fits": REGIONAL_EXPECTED_AUXILIARY_DOWNSCALING_FITS,
        "unexpected_auxiliary_downscaling_fits": 0,
        "expected_total_fits": REGIONAL_EXPECTED_TOTAL_FITS,
        "completed_total_fits": 0,
        "failed_fit_attempts": 0,
        "retried_fit_attempts": 0,
    }


def fit_accounting_template() -> dict[str, int]:
    """Zeroed accounting block. A retry is an ATTEMPT, never a logical fit."""
    return {
        "expected_primary_estimator_fits": EXPECTED_PRIMARY_ESTIMATOR_FITS,
        "completed_primary_estimator_fits": 0,
        "duplicate_primary_estimator_fits": 0,
        "missing_primary_estimator_fits": EXPECTED_PRIMARY_ESTIMATOR_FITS,
        "unexpected_primary_estimator_fits": 0,
        "expected_auxiliary_downscaling_fits": EXPECTED_AUXILIARY_DOWNSCALING_FITS,
        "completed_auxiliary_downscaling_fits": 0,
        "duplicate_auxiliary_downscaling_fits": 0,
        "missing_auxiliary_downscaling_fits": EXPECTED_AUXILIARY_DOWNSCALING_FITS,
        "unexpected_auxiliary_downscaling_fits": 0,
        "expected_total_fits": EXPECTED_TOTAL_FITS,
        "completed_total_fits": 0,
        "failed_fit_attempts": 0,
        "retried_fit_attempts": 0,
    }


def reconcile_fit_accounting(accounting: dict[str, Any]) -> list[str]:
    """Structural problems in a fit-accounting block. Empty list means consistent."""
    problems: list[str] = []
    exp_p = int(accounting.get("expected_primary_estimator_fits", -1))
    exp_a = int(accounting.get("expected_auxiliary_downscaling_fits", -1))
    exp_t = int(accounting.get("expected_total_fits", -1))
    if exp_p != EXPECTED_PRIMARY_ESTIMATOR_FITS:
        problems.append(
            f"expected_primary_estimator_fits={exp_p}, must be "
            f"{EXPECTED_PRIMARY_ESTIMATOR_FITS}"
        )
    if exp_a != EXPECTED_AUXILIARY_DOWNSCALING_FITS:
        problems.append(
            f"expected_auxiliary_downscaling_fits={exp_a}, must be "
            f"{EXPECTED_AUXILIARY_DOWNSCALING_FITS}"
        )
    if exp_t != exp_p + exp_a:
        problems.append(f"expected_total_fits={exp_t}, must be {exp_p + exp_a}")
    comp_p = int(accounting.get("completed_primary_estimator_fits", 0))
    comp_a = int(accounting.get("completed_auxiliary_downscaling_fits", 0))
    comp_t = int(accounting.get("completed_total_fits", 0))
    if comp_t != comp_p + comp_a:
        problems.append(
            f"completed_total_fits={comp_t}, must equal "
            f"completed_primary({comp_p}) + completed_auxiliary({comp_a})"
        )
    for field in (
        "duplicate_primary_estimator_fits", "missing_primary_estimator_fits",
        "unexpected_primary_estimator_fits", "duplicate_auxiliary_downscaling_fits",
        "missing_auxiliary_downscaling_fits", "unexpected_auxiliary_downscaling_fits",
    ):
        value = int(accounting.get(field, 0))
        if value < 0:
            problems.append(f"{field}={value} is negative")
    return problems


# =============================================================================
# Frozen model / bootstrap configuration, read from production config
# =============================================================================
def frozen_model_configuration() -> dict[str, Any]:
    """The frozen knobs, read from `core.config`. Fails closed on any absence.

    Deliberately mirrors `window_closure_sensitivity.model_frozen_configuration`
    rather than re-deriving anything: the two analyses must agree by
    construction, and a missing constant must never be silently defaulted.
    """
    import core.config as config

    required = (
        "STEP8B_N_SPLITS", "STEP8B_RANDOM_SEED", "STEP8B_SPATIAL_BLOCK_SIZE_CELLS",
        "STEP8B_MIN_POSITIVES_PER_POPULATION",
        "STEP8C_N_BOOTSTRAP", "STEP8C_RANDOM_SEED",
        "STEP8C_CI_LOWER", "STEP8C_CI_UPPER",
        "SUMMER_MONTH_START", "SUMMER_MONTH_END",
    )
    missing = [name for name in required if getattr(config, name, None) is None]
    if missing:
        raise MultiRegionWindowClosureError(
            f"Frozen model/bootstrap configuration is incomplete: {missing} is "
            "not defined in core.config. This analysis refuses to substitute a "
            "default for a preregistered parameter."
        )
    n_splits = int(config.STEP8B_N_SPLITS)
    if n_splits != N_FOLDS:
        raise MultiRegionWindowClosureError(
            f"core.config.STEP8B_N_SPLITS={n_splits} but this analysis's fit "
            f"accounting is frozen at {N_FOLDS} folds. The fit-count contract "
            "and production config must agree."
        )
    return {
        "model": PRIMARY_MODEL,
        "primary_population": PRIMARY_POPULATION,
        "n_splits": n_splits,
        "fold_random_seed": int(config.STEP8B_RANDOM_SEED),
        "spatial_block_size_cells": int(config.STEP8B_SPATIAL_BLOCK_SIZE_CELLS),
        "min_positives": int(config.STEP8B_MIN_POSITIVES_PER_POPULATION),
        "strict_folds": True,
        "calibration": None,
        "adaptation": None,
        "source": "core.config (frozen)",
    }


def frozen_bootstrap_configuration() -> dict[str, Any]:
    """Bootstrap knobs, read from `core.config`. Percentile method, not BCa."""
    import core.config as config

    lower = float(config.STEP8C_CI_LOWER)
    upper = float(config.STEP8C_CI_UPPER)
    return {
        "unit": "spatial_block_id",
        "n_bootstrap": int(config.STEP8C_N_BOOTSTRAP),
        "seed": int(config.STEP8C_RANDOM_SEED),
        "ci_lower_percentile": lower,
        "ci_upper_percentile": upper,
        "confidence_level": upper - lower,
        "interval_method": "percentile",
        "paired": True,
        "identical_block_draws_across_variants": True,
        "refit_inside_bootstrap": False,
        "minimum_blocks": 2,
        "source": "core.config (frozen)",
    }


def modis_season_policy() -> dict[str, Any]:
    """The FIXED production MODIS calendar-month filter.

    Applied on top of `filterDate` by
    `scripts/prepare_modis_for_step7.py:_build_qc_masked_modis_stack`. Unlike
    the Landsat month filter -- which production derives from the window and
    which therefore never clips -- this one is a constant, so an
    earlier-closing window can legitimately lose its earliest days.
    """
    import core.config as config

    return {
        "modis_policy_id": "core.config.SUMMER_MONTH_6_9",
        "summer_month_start": int(config.SUMMER_MONTH_START),
        "summer_month_end": int(config.SUMMER_MONTH_END),
        "is_fixed": True,
        "source": (
            "core.config SUMMER_MONTH_START / SUMMER_MONTH_END, applied by "
            "scripts.prepare_modis_for_step7._build_qc_masked_modis_stack"
        ),
    }


def feature_registry() -> dict[str, Any]:
    """The canonical fire-risk feature contract, imported, never re-declared."""
    from src.step8b_train_baseline_vs_thermal_model import (
        BASELINE_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN,
        THERMAL_FEATURES, THERMAL_MODEL_FEATURES,
    )

    return {
        "baseline_features_in_order": list(BASELINE_FEATURES),
        "thermal_features_in_order": list(THERMAL_FEATURES),
        "thermal_model_features_in_order": list(THERMAL_MODEL_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target_column": TARGET_COLUMN,
        "feature_union": sorted(set(BASELINE_FEATURES) | set(THERMAL_MODEL_FEATURES)),
        "source": "src.step8b_train_baseline_vs_thermal_model",
    }


# =============================================================================
# Limitations -- mandatory in report.md
# =============================================================================
LIMITATIONS: tuple[str, ...] = (
    "Closing the predictor window earlier is a predictor-timing sensitivity "
    "analysis. It is retrospective and is not an operational forecasting "
    "validation.",
    "Three new AOIs and one reference AOI, each a single season, are not "
    "evidence of general generalisability.",
    "A confidence interval that includes zero is NOT evidence of equivalence; "
    "no equivalence margin is preregistered in this analysis.",
    "Any performance change is consistent with the predictor and "
    "observation-support changes that follow from the closure date; it does "
    "not establish a mechanism.",
    "Landsat and MODIS scene/observation support themselves change with the "
    "closure date, so support differences are part of what is being measured.",
    "The fixed production MODIS calendar-month filter (months 6-9) clips "
    "shifted windows by AOI-specific amounts. The analysis therefore measures "
    "the closure date together with its interaction with that fixed policy, "
    "not the closure date in isolation.",
    "PR-AUC depends on prevalence. Every comparison is made on the same common "
    "cohort within an AOI, so prevalence is fixed there; PR-AUC levels are NOT "
    "comparable between AOIs.",
    "The label window, event dates and gate dates are frozen and identical "
    "across variants; only predictor timing moves.",
    "Results are descriptive. Each AOI is analysed separately; there is no "
    "pooled estimate and no meta-analytic combination.",
    "evia_2021_extended is a high-prevalence different-regime control, not an "
    "equal-prevalence fourth validation region.",
    "No deployment, alerting or future-fire forecasting claim is made or "
    "supported by this analysis.",
)
