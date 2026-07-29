"""
Generic, directed, LABEL-BLIND marginal Area-of-Applicability (AoA) analysis.

Scientific question
-------------------
For an ordered pair (source -> target), is each target predictor value inside
the marginal predictor support that was actually OBSERVED in the source AOI?

"Marginal" is meant literally: every predictor is evaluated on its own, one
dimension at a time. This is NOT a multivariate joint-support / convex-hull /
density-ratio AoA, and it is deliberately not presented as one.

What this module does NOT do
----------------------------
It fits no model, produces no prediction, runs no adaptation, no
cross-validation and no bootstrap. It reads only frozen Step8A predictor
columns.

LABEL FIREWALL
--------------
`burned` -- and every other label/outcome column -- is never loaded, never
read and never used, for any purpose, including row selection. Every parquet
read in this module passes an EXPLICIT `columns=` allow-list built from the
predictor contract plus the grid/population/eligibility columns. Changing the
label values in a target dataset cannot change a single byte of this analysis
output; `tests/test_marginal_area_of_applicability.py` asserts exactly that.

Directionality
--------------
Support is an asymmetric relation: source_min/source_max come from the SOURCE
AOI only, and are applied to the TARGET AOI only. `A__B` and `B__A` are
therefore different analyses with different identities and different output
namespaces. All ordered pairs are generated with itertools.permutations over
the sorted resolved experiment IDs, so the caller's argument order can never
change the result.

Interpretation boundary
-----------------------
Every number here is DESCRIPTIVE. Being inside the source range does not
guarantee transfer success, and being outside it does not prove that it caused
transfer failure. No statistical significance is claimed or computed.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.paths import PROJECT_ROOT

# --- Canonical feature / population contract (single source of truth) -------
from src.step9a_audit_cross_region_inputs import (
    CATEGORICAL_FEATURES,
    FORBIDDEN_MODEL_COLUMNS,
    PRIMARY_POPULATIONS,
    SHARED_THERMAL_MODEL_FEATURES,
    resolve_step8a_dataset_path,
)

# --- Canonical eligibility / experiment resolution --------------------------
from src.burned_pattern_audit import (
    ANALYSIS_ELIGIBLE_COLUMN,
    BURNABLE_MASK_COLUMN,
    PRE_LABEL_EXCLUDED_COLUMN,
    ExperimentResolution,
    dataset_schema_columns,
    resolve_analysis_eligible_mask,
    resolve_experiments,
)

# --- Canonical provenance helpers -------------------------------------------
from src.step8_large_block_robustness import (
    _git_commit,
    canonical_json,
    sha256_bytes,
    sha256_file,
)

# --- Secondary, DESCRIPTIVE-ONLY threshold, reused verbatim -----------------
from src.step9e_distribution_shift_audit import OUTSIDE_SUPPORT_THRESHOLD


SCHEMA_VERSION = "marginal_aoa.v1"
SUPPORT_DEFINITION_ID = "source_observed_range_and_levels_v1"
DIAGNOSTIC_NAMESPACE = "marginal_area_of_applicability"

PRIMARY_POPULATION = PRIMARY_POPULATIONS[0]

GRID_COLUMNS = ("row_500m", "col_500m")

CELL_STATUS_INSIDE = "inside_support"
CELL_STATUS_OUTSIDE = "outside_support"
CELL_STATUS_NOT_ASSESSABLE = "not_assessable"

SUPPORT_STATUS_AVAILABLE = "source_support_available"
SUPPORT_STATUS_UNAVAILABLE = "source_support_unavailable"

MISSING_VALUE_RULES = {
    "target_missing_is_not_outside": (
        "A missing target value is never counted as outside support; it is "
        "reported separately as missing and makes the cell not_assessable "
        "unless some other feature is outside."
    ),
    "fraction_outside_denominator": "target_n_finite",
    "fraction_missing_denominator": "target_n_total",
    "fraction_unseen_level_denominator": "target_n_nonmissing",
    "source_support_unavailable": (
        "When the source population has no finite value (or no non-missing "
        "level) for a feature, bounds are null, no target value is classified "
        "as outside, and every target value for that feature is "
        "not_assessable."
    ),
    "zero_width_source_range": (
        "When source_min == source_max, target values exactly equal to that "
        "value are in support and any other finite value is outside. No "
        "division or normalisation is performed."
    ),
}

CELL_STATUS_PRECEDENCE = (
    "1. outside_support when any assessable feature is outside source support",
    "2. not_assessable when nothing is outside but at least one feature is "
    "missing or its source support is unavailable",
    "3. inside_support when every feature is assessable and none is outside",
)

DESCRIPTIVE_THRESHOLD_PROVENANCE = (
    "OUTSIDE_SUPPORT_THRESHOLD reused verbatim from "
    "src/step9e_distribution_shift_audit.py. Used ONLY as a secondary "
    "descriptive flag. It is not a PASS/FAIL gate, not a significance test "
    "and not evidence of scientific support."
)

LABEL_FIREWALL = {
    "label_columns_loaded": [],
    "burned_column_read": False,
    "burn_date_column_read": False,
    "target_label_used": False,
    "transfer_predictions_read": False,
    "step9_metrics_read": False,
    "rule": (
        "Every parquet read passes an explicit columns= allow-list containing "
        "only predictor, grid, population-mask and eligibility columns. No "
        "label or outcome column is loaded at any point."
    ),
}

LIMITATIONS = (
    "Marginal AoA evaluates each predictor separately, one dimension at a time.",
    "It does NOT measure multivariate joint support (no convex hull, no "
    "density ratio, no Mahalanobis/kNN distance).",
    "It does NOT assess the correlation structure between predictors; a cell "
    "inside every marginal range may still be a joint-space extrapolation.",
    "Being inside the source range does NOT guarantee transfer success.",
    "Being outside the source range does NOT prove that it caused transfer "
    "failure; this analysis establishes no causal relationship.",
    "This analysis uses no target label and computes no performance metric.",
    "No statistical significance is claimed, computed or implied; every "
    "quantity is descriptive.",
    "Support is defined by the exact observed source minimum and maximum; it "
    "is sensitive to single extreme source cells and to source sample size.",
    "Experiments differ in AOI extent and burned prevalence regime; results "
    "should be read as a sensitivity context, never as a ranking of AOI "
    "quality.",
)


class MarginalAoAError(SystemExit):
    """Fail-fast, contract-violating condition."""


# =============================================================================
# Feature contract
# =============================================================================
def categorical_features() -> tuple[str, ...]:
    """Categorical predictors, in canonical contract order."""
    return tuple(f for f in SHARED_THERMAL_MODEL_FEATURES if f in set(CATEGORICAL_FEATURES))


def numeric_features() -> tuple[str, ...]:
    """Every non-categorical predictor, in canonical contract order."""
    categorical = set(CATEGORICAL_FEATURES)
    return tuple(f for f in SHARED_THERMAL_MODEL_FEATURES if f not in categorical)


def all_features() -> tuple[str, ...]:
    return tuple(SHARED_THERMAL_MODEL_FEATURES)


def validate_feature_contract() -> None:
    """Fail fast if a forbidden column ever leaks into the predictor contract.

    `row_500m`/`col_500m` are forbidden as MODEL features but are legitimately
    loaded here as grid keys; they are checked against the feature list only.
    """
    forbidden = sorted(set(all_features()) & set(FORBIDDEN_MODEL_COLUMNS))
    if forbidden:
        raise MarginalAoAError(
            "Forbidden column(s) present in the predictor feature contract: "
            f"{forbidden}. The feature contract must contain predictors only."
        )
    if not numeric_features():
        raise MarginalAoAError("Predictor contract resolved to zero numeric features.")
    overlap = set(numeric_features()) & set(categorical_features())
    if overlap:
        raise MarginalAoAError(
            f"Feature(s) classified as both numeric and categorical: {sorted(overlap)}."
        )


def feature_contract_payload() -> dict[str, Any]:
    return {
        "all_features_in_order": list(all_features()),
        "numeric_features_in_order": list(numeric_features()),
        "categorical_features_in_order": list(categorical_features()),
        "numeric_feature_count": len(numeric_features()),
        "categorical_feature_count": len(categorical_features()),
        "total_feature_count": len(all_features()),
        "contract_source": "src.step9a_audit_cross_region_inputs.SHARED_THERMAL_MODEL_FEATURES",
    }


# =============================================================================
# Label-blind loading and population resolution
# =============================================================================
def load_columns_for(schema_columns: Iterable[str]) -> list[str]:
    """The EXPLICIT parquet allow-list. Never contains a label column.

    Optional columns (`analysis_eligible`, `pre_label_burn_excluded`) are
    included only when the frozen dataset actually carries them, which is how
    experiments without a pre-label exclusion configured are handled without
    branching on any experiment name.
    """
    present = set(schema_columns)
    columns: list[str] = list(all_features())
    columns.extend(GRID_COLUMNS)
    columns.append(BURNABLE_MASK_COLUMN)
    for optional in (ANALYSIS_ELIGIBLE_COLUMN, PRE_LABEL_EXCLUDED_COLUMN):
        if optional in present:
            columns.append(optional)

    ordered = list(dict.fromkeys(columns))
    leaked = sorted(set(ordered) & set(FORBIDDEN_MODEL_COLUMNS) - set(GRID_COLUMNS))
    if leaked:
        raise MarginalAoAError(
            f"Label firewall violation: refusing to read column(s) {leaked}."
        )
    return ordered


def validate_dataset_columns(schema_columns: Iterable[str], experiment_id: str) -> None:
    present = set(schema_columns)
    required = list(all_features()) + list(GRID_COLUMNS) + [BURNABLE_MASK_COLUMN]
    missing = [c for c in required if c not in present]
    if missing:
        raise MarginalAoAError(
            f"'{experiment_id}': frozen Step8A dataset is missing required "
            f"column(s) for the marginal AoA feature contract: {missing}."
        )


def load_population(
    path: Path, experiment_id: str, *, read_parquet=None
) -> pd.DataFrame:
    """Load ONLY the allow-listed columns and reduce to the analysis population.

    `read_parquet` is injectable so tests can capture the exact `columns=`
    argument and prove the label firewall holds. It is resolved at CALL time
    (not bound as a default) so that patching `pd.read_parquet` is observable
    here too -- a default-bound reader would silently escape such a patch.
    """
    read_parquet = pd.read_parquet if read_parquet is None else read_parquet
    if not Path(path).is_file():
        raise MarginalAoAError(
            f"'{experiment_id}': frozen Step8A dataset not found: {path}."
        )
    schema_columns = dataset_schema_columns(Path(path))
    validate_dataset_columns(schema_columns, experiment_id)
    columns = load_columns_for(schema_columns)
    frame = read_parquet(path, columns=columns)
    return resolve_population(frame, experiment_id)


def resolve_population(df: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    """Label-blind population resolver.

    population = canonical analysis eligibility AND valid grid AND primary
    burnable mask. Deliberately NOT
    `src.domain_classifier_audit.resolve_population`, whose REQUIRED_COLUMNS
    contract includes `burned`; no burned/unburned split exists here at all.
    """
    eligible_mask = resolve_analysis_eligible_mask(df)
    valid_grid = df["row_500m"].notna() & df["col_500m"].notna()
    primary_mask = df[BURNABLE_MASK_COLUMN].astype(bool)
    population = df.loc[eligible_mask & valid_grid & primary_mask].copy()

    if population.empty:
        raise MarginalAoAError(
            f"'{experiment_id}': the marginal AoA population is empty "
            f"(analysis_eligible AND valid grid AND {BURNABLE_MASK_COLUMN})."
        )

    key = population[list(GRID_COLUMNS)]
    duplicated = key.duplicated(keep=False)
    if duplicated.any():
        n_dup = int(duplicated.sum())
        raise MarginalAoAError(
            f"'{experiment_id}': {n_dup} duplicate (row_500m, col_500m) grid "
            "cell(s) in the marginal AoA population; the frozen Step8A grid "
            "must be unique."
        )

    population = population.sort_values(list(GRID_COLUMNS), kind="mergesort")
    return population.reset_index(drop=True)


# =============================================================================
# Numeric marginal support
# =============================================================================
def _finite(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    return numeric[np.isfinite(numeric)]


def numeric_feature_support(
    source_values: pd.Series, target_values: pd.Series, feature: str,
) -> dict[str, Any]:
    """Marginal support of one numeric feature for one ordered direction.

    Bounds are INCLUSIVE: source_min <= value <= source_max is in support.
    """
    source_numeric = pd.to_numeric(source_values, errors="coerce").to_numpy(dtype="float64")
    target_numeric = pd.to_numeric(target_values, errors="coerce").to_numpy(dtype="float64")
    source_finite = source_numeric[np.isfinite(source_numeric)]
    target_finite_mask = np.isfinite(target_numeric)
    target_finite = target_numeric[target_finite_mask]

    source_n_total = int(source_numeric.size)
    source_n_finite = int(source_finite.size)
    target_n_total = int(target_numeric.size)
    target_n_finite = int(target_finite.size)

    row: dict[str, Any] = {
        "feature": feature,
        "feature_kind": "numeric",
        "source_n_total": source_n_total,
        "source_n_finite": source_n_finite,
        "source_n_missing": source_n_total - source_n_finite,
        "target_n_total": target_n_total,
        "target_n_finite": target_n_finite,
        "target_n_missing": target_n_total - target_n_finite,
        "fraction_target_missing": (
            (target_n_total - target_n_finite) / target_n_total if target_n_total else None
        ),
    }

    if source_n_finite == 0:
        # No observed source support: nothing can be declared outside it.
        row.update({
            "support_status": SUPPORT_STATUS_UNAVAILABLE,
            "source_min": None,
            "source_max": None,
            "source_range_width": None,
            "target_n_below_source_min": 0,
            "target_n_above_source_max": 0,
            "target_n_in_source_range": 0,
            "target_n_not_assessable": target_n_total,
            "fraction_below_source_min": None,
            "fraction_above_source_max": None,
            "fraction_outside_source_range": None,
            "mean_absolute_exceedance": None,
            "max_absolute_exceedance": None,
            "exceeds_step9e_descriptive_threshold": None,
        })
        return row

    source_min = float(np.min(source_finite))
    source_max = float(np.max(source_finite))

    below_mask = target_finite < source_min
    above_mask = target_finite > source_max
    n_below = int(np.count_nonzero(below_mask))
    n_above = int(np.count_nonzero(above_mask))
    n_in_range = target_n_finite - n_below - n_above

    # In-range cells contribute an exceedance of exactly 0.
    exceedance = np.zeros_like(target_finite, dtype="float64")
    exceedance[below_mask] = source_min - target_finite[below_mask]
    exceedance[above_mask] = target_finite[above_mask] - source_max

    fraction_outside = (
        (n_below + n_above) / target_n_finite if target_n_finite else None
    )
    row.update({
        "support_status": SUPPORT_STATUS_AVAILABLE,
        "source_min": source_min,
        "source_max": source_max,
        "source_range_width": source_max - source_min,
        "target_n_below_source_min": n_below,
        "target_n_above_source_max": n_above,
        "target_n_in_source_range": int(n_in_range),
        "target_n_not_assessable": target_n_total - target_n_finite,
        "fraction_below_source_min": n_below / target_n_finite if target_n_finite else None,
        "fraction_above_source_max": n_above / target_n_finite if target_n_finite else None,
        "fraction_outside_source_range": fraction_outside,
        "mean_absolute_exceedance": float(np.mean(exceedance)) if target_n_finite else None,
        "max_absolute_exceedance": float(np.max(exceedance)) if target_n_finite else None,
        "exceeds_step9e_descriptive_threshold": (
            bool(fraction_outside > OUTSIDE_SUPPORT_THRESHOLD)
            if fraction_outside is not None else None
        ),
    })
    return row


def numeric_outside_mask(
    target_values: pd.Series, source_min: Optional[float], source_max: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    """(outside_mask, missing_mask) for one numeric feature's target values."""
    target_numeric = pd.to_numeric(target_values, errors="coerce").to_numpy(dtype="float64")
    finite_mask = np.isfinite(target_numeric)
    missing_mask = ~finite_mask
    if source_min is None or source_max is None:
        return np.zeros(target_numeric.shape, dtype=bool), missing_mask
    outside = np.zeros(target_numeric.shape, dtype=bool)
    outside[finite_mask] = (
        (target_numeric[finite_mask] < source_min)
        | (target_numeric[finite_mask] > source_max)
    )
    return outside, missing_mask


# =============================================================================
# Categorical marginal support
# =============================================================================
def canonical_level(value: Any) -> Optional[str]:
    """Deterministic, stable string form of a categorical level.

    Integral numerics collapse to their integer form so that a level stored as
    10, 10.0 or "10" never serialises as three different levels.
    """
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        as_float = float(value)
        return str(int(as_float)) if as_float.is_integer() else repr(as_float)
    text = str(value).strip()
    return text or None


def canonical_levels(values: pd.Series) -> list[Optional[str]]:
    return [canonical_level(v) for v in values.tolist()]


def categorical_feature_support(
    source_values: pd.Series, target_values: pd.Series, feature: str,
) -> dict[str, Any]:
    """Marginal support of one categorical feature for one ordered direction."""
    source_levels = canonical_levels(source_values)
    target_levels = canonical_levels(target_values)

    source_nonmissing = [lv for lv in source_levels if lv is not None]
    target_nonmissing = [lv for lv in target_levels if lv is not None]
    source_observed = sorted(set(source_nonmissing))
    target_observed = sorted(set(target_nonmissing))

    source_n_total, target_n_total = len(source_levels), len(target_levels)
    source_n_nonmissing = len(source_nonmissing)
    target_n_nonmissing = len(target_nonmissing)

    row: dict[str, Any] = {
        "feature": feature,
        "feature_kind": "categorical",
        "source_n_total": source_n_total,
        "source_n_nonmissing": source_n_nonmissing,
        "source_n_missing": source_n_total - source_n_nonmissing,
        "source_observed_levels": source_observed,
        "target_n_total": target_n_total,
        "target_n_nonmissing": target_n_nonmissing,
        "target_n_missing": target_n_total - target_n_nonmissing,
        "target_observed_levels": target_observed,
        "fraction_target_missing": (
            (target_n_total - target_n_nonmissing) / target_n_total if target_n_total else None
        ),
    }

    if source_n_nonmissing == 0:
        row.update({
            "support_status": SUPPORT_STATUS_UNAVAILABLE,
            "target_unseen_levels": [],
            "target_n_unseen_level": 0,
            "target_n_not_assessable": target_n_total,
            "fraction_target_unseen_level": None,
            "exceeds_step9e_descriptive_threshold": None,
        })
        return row

    support = set(source_observed)
    unseen_levels = sorted(lv for lv in target_observed if lv not in support)
    n_unseen = sum(1 for lv in target_nonmissing if lv not in support)
    fraction_unseen = n_unseen / target_n_nonmissing if target_n_nonmissing else None
    row.update({
        "support_status": SUPPORT_STATUS_AVAILABLE,
        "target_unseen_levels": unseen_levels,
        "target_n_unseen_level": n_unseen,
        "target_n_not_assessable": target_n_total - target_n_nonmissing,
        "fraction_target_unseen_level": fraction_unseen,
        "exceeds_step9e_descriptive_threshold": (
            bool(fraction_unseen > OUTSIDE_SUPPORT_THRESHOLD)
            if fraction_unseen is not None else None
        ),
    })
    return row


def categorical_outside_mask(
    target_values: pd.Series, source_observed_levels: Optional[list[str]],
) -> tuple[np.ndarray, np.ndarray]:
    """(unseen_mask, missing_mask) for one categorical feature."""
    levels = canonical_levels(target_values)
    missing_mask = np.array([lv is None for lv in levels], dtype=bool)
    if not source_observed_levels:
        return np.zeros(len(levels), dtype=bool), missing_mask
    support = set(source_observed_levels)
    unseen = np.array(
        [lv is not None and lv not in support for lv in levels], dtype=bool
    )
    return unseen, missing_mask


# =============================================================================
# Cell-level classification
# =============================================================================
def build_target_cell_table(
    source_id: str, target_id: str, target_population: pd.DataFrame,
    numeric_rows: list[dict[str, Any]], categorical_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    n_rows = len(target_population)
    numeric_outside = np.zeros(n_rows, dtype="int64")
    categorical_outside = np.zeros(n_rows, dtype="int64")
    missing_count = np.zeros(n_rows, dtype="int64")
    unavailable_count = np.zeros(n_rows, dtype="int64")

    by_feature = {row["feature"]: row for row in numeric_rows}
    for feature in numeric_features():
        row = by_feature[feature]
        outside, missing = numeric_outside_mask(
            target_population[feature], row["source_min"], row["source_max"],
        )
        numeric_outside += outside.astype("int64")
        missing_count += missing.astype("int64")
        if row["support_status"] == SUPPORT_STATUS_UNAVAILABLE:
            # Every target value for this feature is un-assessable, including
            # the ones that are present.
            unavailable_count += (~missing).astype("int64")

    by_feature = {row["feature"]: row for row in categorical_rows}
    for feature in categorical_features():
        row = by_feature[feature]
        outside, missing = categorical_outside_mask(
            target_population[feature], row["source_observed_levels"],
        )
        categorical_outside += outside.astype("int64")
        missing_count += missing.astype("int64")
        if row["support_status"] == SUPPORT_STATUS_UNAVAILABLE:
            unavailable_count += (~missing).astype("int64")

    total_outside = numeric_outside + categorical_outside
    any_outside = total_outside > 0
    any_not_assessable = (missing_count + unavailable_count) > 0

    status = np.where(
        any_outside, CELL_STATUS_OUTSIDE,
        np.where(any_not_assessable, CELL_STATUS_NOT_ASSESSABLE, CELL_STATUS_INSIDE),
    )

    table = pd.DataFrame({
        "source_experiment_id": source_id,
        "target_experiment_id": target_id,
        "row_500m": target_population["row_500m"].to_numpy(),
        "col_500m": target_population["col_500m"].to_numpy(),
        "numeric_features_outside_count": numeric_outside,
        "categorical_features_outside_count": categorical_outside,
        "total_features_outside_count": total_outside,
        "features_missing_count": missing_count,
        "features_source_support_unavailable_count": unavailable_count,
        "any_feature_outside_support": any_outside,
        "any_feature_not_assessable": any_not_assessable,
        "cell_support_status": status,
    })
    return table.sort_values(list(GRID_COLUMNS), kind="mergesort").reset_index(drop=True)


# =============================================================================
# Identity / provenance
# =============================================================================
def scientific_configuration(
    source_id: str, target_id: str, source_sha256: str, target_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "support_definition_id": SUPPORT_DEFINITION_ID,
        "support_definition": (
            "A target value is in support when it lies within the INCLUSIVE "
            "closed interval [source_min, source_max] observed in the source "
            "population (numeric), or when its level was observed non-missing "
            "in the source population (categorical). No quantile support, IQR "
            "fence, z-score or winsorisation is used in v1."
        ),
        "source_experiment_id": source_id,
        "target_experiment_id": target_id,
        "direction": f"{source_id}_to_{target_id}",
        "feature_contract": feature_contract_payload(),
        "primary_population": PRIMARY_POPULATION,
        "eligibility_definition": (
            f"{ANALYSIS_ELIGIBLE_COLUMN} (canonical Step8A pre-label-burn "
            "exclusion, reused verbatim) AND non-null row_500m/col_500m AND "
            f"{BURNABLE_MASK_COLUMN} == True. No burned/unburned split."
        ),
        "source_step8a_sha256": source_sha256,
        "target_step8a_sha256": target_sha256,
        "missing_value_rules": MISSING_VALUE_RULES,
        "cell_status_precedence": list(CELL_STATUS_PRECEDENCE),
        "descriptive_threshold": OUTSIDE_SUPPORT_THRESHOLD,
        "descriptive_threshold_provenance": DESCRIPTIVE_THRESHOLD_PROVENANCE,
    }


def compute_analysis_id(config: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(config).encode("utf-8"))


def comparison_analysis_id(pair_ids: list[str]) -> str:
    return sha256_bytes(
        canonical_json({
            "schema_version": SCHEMA_VERSION,
            "support_definition_id": SUPPORT_DEFINITION_ID,
            "pair_analysis_ids": sorted(pair_ids),
        }).encode("utf-8")
    )


# =============================================================================
# Directed pairs
# =============================================================================
def ordered_pairs(experiment_ids: Iterable[str]) -> list[tuple[str, str]]:
    """Every ordered (source, target) pair, deterministically.

    Sorting first makes the result independent of the caller's argument order;
    permutations (not combinations) keeps `A__B` and `B__A` distinct because
    marginal support is an asymmetric relation.
    """
    unique = sorted(set(experiment_ids))
    if len(unique) < 2:
        raise MarginalAoAError(
            "Marginal AoA needs at least 2 resolved experiments to form a "
            f"directed pair; got {unique}."
        )
    return [(a, b) for a, b in permutations(unique, 2)]


def pair_token(source_id: str, target_id: str) -> str:
    """Direction-preserving namespace token; never sorted."""
    return f"{source_id}__{target_id}"


# =============================================================================
# Paths
# =============================================================================
def diagnostics_root(output_root: Optional[Path] = None) -> Path:
    if output_root is not None:
        return Path(output_root)
    return PROJECT_ROOT / "outputs" / "diagnostics" / DIAGNOSTIC_NAMESPACE


def pair_output_dir(source_id: str, target_id: str, output_root: Optional[Path] = None) -> Path:
    return diagnostics_root(output_root) / "pairs" / pair_token(source_id, target_id)


def comparison_output_dir(output_root: Optional[Path] = None) -> Path:
    return diagnostics_root(output_root) / "comparison"


PAIR_OUTPUT_FILENAMES = (
    "marginal_aoa_numeric_features.csv",
    "marginal_aoa_categorical_features.csv",
    "marginal_aoa_target_cells.parquet",
    "marginal_aoa_summary.json",
    "marginal_aoa_report.md",
    "manifest.json",
)

COMPARISON_OUTPUT_FILENAMES = (
    "multi_aoi_marginal_aoa_comparison.csv",
    "multi_aoi_marginal_aoa_comparison.json",
    "multi_aoi_marginal_aoa_comparison.md",
    "manifest.json",
)


def pair_output_paths(source_id: str, target_id: str, output_root: Optional[Path] = None) -> dict[str, Path]:
    directory = pair_output_dir(source_id, target_id, output_root)
    return {name: directory / name for name in PAIR_OUTPUT_FILENAMES}


def comparison_output_paths(output_root: Optional[Path] = None) -> dict[str, Path]:
    directory = comparison_output_dir(output_root)
    return {name: directory / name for name in COMPARISON_OUTPUT_FILENAMES}


def resolve_dataset_path(experiment_id: str, experiments_root: Optional[Path] = None) -> Path:
    """Canonical Step8A path, honouring the injected experiments root.

    Delegates to the canonical resolver so the on-disk contract is defined in
    exactly one place; `experiments_root` is that resolver's own documented
    dependency-injection point, never a monkeypatched global.
    """
    return resolve_step8a_dataset_path(experiment_id, experiments_root=experiments_root)


# =============================================================================
# Experiment resolution (with injectable experiments root)
# =============================================================================
def resolve_experiment_set(
    experiments: Optional[list[str]] = None,
    all_enabled: bool = False,
    experiments_root: Optional[Path] = None,
) -> ExperimentResolution:
    """Resolve the experiment set, reusing the canonical resolver.

    With the canonical root, `burned_pattern_audit.resolve_experiments` is
    used verbatim. With an injected `experiments_root` its Step8A presence
    check would look in the wrong place, so the same selection semantics are
    applied against the injected root instead -- registry membership,
    duplicate rejection and the explicit/all-enabled distinction are still
    delegated to the canonical helpers.
    """
    if experiments_root is None:
        return resolve_experiments(experiments=experiments, all_enabled=all_enabled)

    from core.pipeline_orchestrator import LEGACY_EXPERIMENT_ID
    from core.regions import get_experiment, list_experiments

    if bool(experiments) == bool(all_enabled):
        raise MarginalAoAError(
            "Exactly one of --experiments or --all-enabled must be given "
            f"(got experiments={experiments!r}, all_enabled={all_enabled!r})."
        )

    if experiments:
        if len(experiments) != len(set(experiments)):
            counts: dict[str, int] = {}
            for entry in experiments:
                counts[entry] = counts.get(entry, 0) + 1
            raise MarginalAoAError(
                "Duplicate --experiments entries are not allowed: "
                f"{sorted(k for k, v in counts.items() if v > 1)}."
            )
        requested = tuple(experiments)
        for experiment_id in requested:
            get_experiment(experiment_id)
            path = resolve_dataset_path(experiment_id, experiments_root)
            if not path.is_file():
                raise MarginalAoAError(
                    f"Missing canonical Step8A dataset for '{experiment_id}': {path}."
                )
        return ExperimentResolution(
            requested_ids=requested, resolved_ids=requested,
            selection_mode="explicit", excluded={},
        )

    enabled = list_experiments(include_disabled=False)
    candidates = tuple(sorted(e for e in enabled if e != LEGACY_EXPERIMENT_ID))
    resolved, excluded = [], {}
    for experiment_id in candidates:
        path = resolve_dataset_path(experiment_id, experiments_root)
        if path.is_file():
            resolved.append(experiment_id)
        else:
            excluded[experiment_id] = f"missing_canonical_step8a_dataset:{path}"
    return ExperimentResolution(
        requested_ids=candidates, resolved_ids=tuple(resolved),
        selection_mode="all_enabled", excluded=excluded,
    )


# =============================================================================
# Pair analysis
# =============================================================================
@dataclass(frozen=True)
class PairResult:
    source_id: str
    target_id: str
    analysis_id: str
    summary: dict[str, Any]
    numeric_rows: list[dict[str, Any]]
    categorical_rows: list[dict[str, Any]]
    cells: pd.DataFrame
    scientific_configuration: dict[str, Any]


def analyse_pair(
    source_id: str, target_id: str,
    source_population: pd.DataFrame, target_population: pd.DataFrame,
    source_sha256: str, target_sha256: str,
) -> PairResult:
    config = scientific_configuration(source_id, target_id, source_sha256, target_sha256)
    analysis_id = compute_analysis_id(config)

    numeric_rows = [
        numeric_feature_support(source_population[f], target_population[f], f)
        for f in numeric_features()
    ]
    categorical_rows = [
        categorical_feature_support(source_population[f], target_population[f], f)
        for f in categorical_features()
    ]
    cells = build_target_cell_table(
        source_id, target_id, target_population, numeric_rows, categorical_rows,
    )

    target_rows = len(target_population)
    counts = cells["cell_support_status"].value_counts()
    n_inside = int(counts.get(CELL_STATUS_INSIDE, 0))
    n_outside = int(counts.get(CELL_STATUS_OUTSIDE, 0))
    n_not_assessable = int(counts.get(CELL_STATUS_NOT_ASSESSABLE, 0))

    fractions_by_feature: list[tuple[str, Optional[float]]] = []
    for row in numeric_rows:
        fractions_by_feature.append((row["feature"], row["fraction_outside_source_range"]))
    for row in categorical_rows:
        fractions_by_feature.append((row["feature"], row["fraction_target_unseen_level"]))

    assessed = [(f, v) for f, v in fractions_by_feature if v is not None]
    features_with_any_outside = sorted(f for f, v in assessed if v > 0)
    features_exceeding = sorted(f for f, v in assessed if v > OUTSIDE_SUPPORT_THRESHOLD)
    maximum_fraction = max((v for _, v in assessed), default=None)
    # Deterministic: fraction descending, then feature name ascending.
    top_features = [
        {"feature": f, "fraction_outside": v}
        for f, v in sorted(assessed, key=lambda item: (-item[1], item[0]))
        if v > 0
    ]

    summary = {
        "schema_version": SCHEMA_VERSION,
        "support_definition_id": SUPPORT_DEFINITION_ID,
        "analysis_id": analysis_id,
        "source_experiment_id": source_id,
        "target_experiment_id": target_id,
        "direction": f"{source_id}_to_{target_id}",
        "primary_population": PRIMARY_POPULATION,
        "source_population_rows": len(source_population),
        "target_population_rows": target_rows,
        "numeric_feature_count": len(numeric_features()),
        "categorical_feature_count": len(categorical_features()),
        "total_feature_count": len(all_features()),
        "target_cells_inside_support": n_inside,
        "target_cells_outside_support": n_outside,
        "target_cells_not_assessable": n_not_assessable,
        "fraction_target_cells_inside_support": n_inside / target_rows if target_rows else None,
        "fraction_target_cells_outside_support": n_outside / target_rows if target_rows else None,
        "fraction_target_cells_not_assessable": n_not_assessable / target_rows if target_rows else None,
        "features_with_any_outside_support": features_with_any_outside,
        "features_exceeding_10pct_outside": features_exceeding,
        "maximum_feature_fraction_outside": maximum_fraction,
        "top_outside_support_features": top_features,
        "descriptive_threshold": OUTSIDE_SUPPORT_THRESHOLD,
        "descriptive_threshold_provenance": DESCRIPTIVE_THRESHOLD_PROVENANCE,
        "label_firewall": dict(LABEL_FIREWALL),
        "limitations": list(LIMITATIONS),
    }

    return PairResult(
        source_id=source_id, target_id=target_id, analysis_id=analysis_id,
        summary=summary, numeric_rows=numeric_rows, categorical_rows=categorical_rows,
        cells=cells, scientific_configuration=config,
    )


# =============================================================================
# Rendering
# =============================================================================
def _json_dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n"


def _levels_cell(levels: Optional[list[str]]) -> str:
    """Deterministic, stable serialisation of a level list inside a CSV cell."""
    return "|".join(levels) if levels else ""


def numeric_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def categorical_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in ("source_observed_levels", "target_observed_levels", "target_unseen_levels"):
        if column in frame.columns:
            frame[column] = frame[column].map(_levels_cell)
    return frame


def render_pair_markdown(result: PairResult) -> str:
    s = result.summary
    lines: list[str] = []
    add = lines.append
    add(f"# Marginal Area of Applicability — `{s['source_experiment_id']} → {s['target_experiment_id']}`")
    add("")
    add(f"- Schema: `{s['schema_version']}`")
    add(f"- Support definition: `{s['support_definition_id']}`")
    add(f"- analysis_id: `{s['analysis_id']}`")
    add(f"- Primary population: `{s['primary_population']}`")
    add(f"- Source population rows: **{s['source_population_rows']}**")
    add(f"- Target population rows: **{s['target_population_rows']}**")
    add(f"- Features: **{s['numeric_feature_count']} numeric + {s['categorical_feature_count']} categorical**")
    add("")
    add("## Target cell support status")
    add("")
    add("| Status | Cells | Fraction |")
    add("|---|---:|---:|")
    for label, count, fraction in (
        ("inside_support", s["target_cells_inside_support"], s["fraction_target_cells_inside_support"]),
        ("outside_support", s["target_cells_outside_support"], s["fraction_target_cells_outside_support"]),
        ("not_assessable", s["target_cells_not_assessable"], s["fraction_target_cells_not_assessable"]),
    ):
        rendered = "—" if fraction is None else f"{fraction:.4f}"
        add(f"| {label} | {count} | {rendered} |")
    add("")
    add("## Per-feature marginal support (numeric)")
    add("")
    add("| Feature | source_min | source_max | below | above | in range | fraction outside |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for row in result.numeric_rows:
        if row["support_status"] == SUPPORT_STATUS_UNAVAILABLE:
            add(f"| {row['feature']} | — | — | — | — | — | source support unavailable |")
            continue
        add(
            f"| {row['feature']} | {row['source_min']:.6g} | {row['source_max']:.6g} | "
            f"{row['target_n_below_source_min']} | {row['target_n_above_source_max']} | "
            f"{row['target_n_in_source_range']} | {row['fraction_outside_source_range']:.4f} |"
        )
    add("")
    add("## Per-feature marginal support (categorical)")
    add("")
    add("| Feature | source levels | target levels | unseen levels | fraction unseen |")
    add("|---|---|---|---|---:|")
    for row in result.categorical_rows:
        fraction = row["fraction_target_unseen_level"]
        rendered = "source support unavailable" if fraction is None else f"{fraction:.4f}"
        add(
            f"| {row['feature']} | {_levels_cell(row['source_observed_levels']) or '—'} | "
            f"{_levels_cell(row['target_observed_levels']) or '—'} | "
            f"{_levels_cell(row['target_unseen_levels']) or '—'} | {rendered} |"
        )
    add("")
    if s["top_outside_support_features"]:
        add("## Features with any target value outside source support")
        add("")
        add("Ordered by fraction outside (descending), then feature name.")
        add("")
        for entry in s["top_outside_support_features"]:
            flag = " (above the descriptive 0.10 reference)" if entry["fraction_outside"] > OUTSIDE_SUPPORT_THRESHOLD else ""
            add(f"- `{entry['feature']}`: {entry['fraction_outside']:.4f}{flag}")
        add("")
    add("## Label firewall")
    add("")
    add("This analysis never loads a label column. `burned` and every other "
        "outcome column are excluded from the parquet read allow-list, so the "
        "result is byte-identical under any change to the target labels.")
    add("")
    add("## Interpretation boundary")
    add("")
    for limitation in LIMITATIONS:
        add(f"- {limitation}")
    add("")
    add(f"The {OUTSIDE_SUPPORT_THRESHOLD} reference is a descriptive marker "
        "reused from the Step9E shift audit. It is not a PASS/FAIL gate, not "
        "a significance test, and not evidence of scientific support.")
    add("")
    return "\n".join(lines)


def render_comparison_markdown(rows: list[dict[str, Any]], analysis_id: str) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Multi-AOI marginal Area of Applicability — directed comparison")
    add("")
    add(f"- Schema: `{SCHEMA_VERSION}`")
    add(f"- Support definition: `{SUPPORT_DEFINITION_ID}`")
    add(f"- comparison analysis_id: `{analysis_id}`")
    add(f"- Directed pairs: **{len(rows)}**")
    add("")
    add("Each row is one ORDERED direction: `source → target` answers a "
        "different question from `target → source`.")
    add("")
    add("| Source | Target | Target rows | inside | outside | not assessable | max feature fraction outside |")
    add("|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        maximum = row["maximum_feature_fraction_outside"]
        rendered = "—" if maximum is None else f"{maximum:.4f}"
        add(
            f"| {row['source_experiment_id']} | {row['target_experiment_id']} | "
            f"{row['target_population_rows']} | {row['target_cells_inside_support']} | "
            f"{row['target_cells_outside_support']} | {row['target_cells_not_assessable']} | "
            f"{rendered} |"
        )
    add("")
    add("## Interpretation boundary")
    add("")
    for limitation in LIMITATIONS:
        add(f"- {limitation}")
    add("")
    return "\n".join(lines)


COMPARISON_COLUMNS = (
    "source_experiment_id",
    "target_experiment_id",
    "direction",
    "analysis_id",
    "primary_population",
    "source_population_rows",
    "target_population_rows",
    "numeric_feature_count",
    "categorical_feature_count",
    "total_feature_count",
    "target_cells_inside_support",
    "target_cells_outside_support",
    "target_cells_not_assessable",
    "fraction_target_cells_inside_support",
    "fraction_target_cells_outside_support",
    "fraction_target_cells_not_assessable",
    "maximum_feature_fraction_outside",
    "features_with_any_outside_support",
    "features_exceeding_10pct_outside",
)


def comparison_rows(results: list[PairResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        s = result.summary
        row = {column: s.get(column) for column in COMPARISON_COLUMNS}
        row["features_with_any_outside_support"] = "|".join(s["features_with_any_outside_support"])
        row["features_exceeding_10pct_outside"] = "|".join(s["features_exceeding_10pct_outside"])
        rows.append(row)
    # Deterministic: by source then target -- direction preserved, never sorted
    # into an unordered pair.
    return sorted(rows, key=lambda r: (r["source_experiment_id"], r["target_experiment_id"]))


# =============================================================================
# Writing
# =============================================================================
def build_pair_manifest(
    result: PairResult, input_paths: dict[str, str], input_sha256: dict[str, str],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "analysis_id": result.analysis_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "schema_version": SCHEMA_VERSION,
        "source_experiment_id": result.source_id,
        "target_experiment_id": result.target_id,
        "input_paths": dict(input_paths),
        "input_sha256": dict(input_sha256),
        "output_paths": {name: str(path) for name, path in sorted(output_paths.items())},
        "scientific_configuration": result.scientific_configuration,
        "label_firewall": dict(LABEL_FIREWALL),
        "limitations": list(LIMITATIONS),
    }


def _existing_analysis_id(manifest_path: Path) -> Optional[str]:
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8")).get("analysis_id")
    except (OSError, json.JSONDecodeError):
        return None


def assert_overwrite_allowed(manifest_path: Path, analysis_id: str, force: bool, label: str) -> bool:
    """Returns True when the existing outputs are an idempotent rerun."""
    existing = _existing_analysis_id(manifest_path)
    if existing is None:
        return False
    if existing == analysis_id:
        return True
    if not force:
        raise MarginalAoAError(
            f"{label}: existing outputs carry a DIFFERENT analysis_id "
            f"({existing} != {analysis_id}). Re-run with --force to overwrite "
            "this marginal AoA namespace (no other namespace is touched)."
        )
    return False


def write_pair_outputs(
    result: PairResult, output_root: Optional[Path],
    input_paths: dict[str, str], input_sha256: dict[str, str], force: bool,
) -> dict[str, Path]:
    paths = pair_output_paths(result.source_id, result.target_id, output_root)
    assert_overwrite_allowed(
        paths["manifest.json"], result.analysis_id, force,
        f"[{pair_token(result.source_id, result.target_id)}]",
    )
    directory = pair_output_dir(result.source_id, result.target_id, output_root)
    directory.mkdir(parents=True, exist_ok=True)

    numeric_dataframe(result.numeric_rows).to_csv(
        paths["marginal_aoa_numeric_features.csv"], index=False
    )
    categorical_dataframe(result.categorical_rows).to_csv(
        paths["marginal_aoa_categorical_features.csv"], index=False
    )
    result.cells.to_parquet(paths["marginal_aoa_target_cells.parquet"], index=False)
    paths["marginal_aoa_summary.json"].write_text(_json_dump(result.summary), encoding="utf-8")
    paths["marginal_aoa_report.md"].write_text(render_pair_markdown(result), encoding="utf-8")
    paths["manifest.json"].write_text(
        _json_dump(build_pair_manifest(result, input_paths, input_sha256, paths)),
        encoding="utf-8",
    )
    return paths


def write_comparison_outputs(
    results: list[PairResult], output_root: Optional[Path],
    input_sha256: dict[str, str], force: bool,
) -> dict[str, Path]:
    rows = comparison_rows(results)
    analysis_id = comparison_analysis_id([r.analysis_id for r in results])
    paths = comparison_output_paths(output_root)
    assert_overwrite_allowed(paths["manifest.json"], analysis_id, force, "[comparison]")
    comparison_output_dir(output_root).mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS)).to_csv(
        paths["multi_aoi_marginal_aoa_comparison.csv"], index=False
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "support_definition_id": SUPPORT_DEFINITION_ID,
        "analysis_id": analysis_id,
        "primary_population": PRIMARY_POPULATION,
        "directed_pair_count": len(rows),
        "directed_pairs": [
            {"source_experiment_id": r["source_experiment_id"],
             "target_experiment_id": r["target_experiment_id"],
             "analysis_id": r["analysis_id"]}
            for r in rows
        ],
        "rows": rows,
        "label_firewall": dict(LABEL_FIREWALL),
        "limitations": list(LIMITATIONS),
    }
    paths["multi_aoi_marginal_aoa_comparison.json"].write_text(_json_dump(payload), encoding="utf-8")
    paths["multi_aoi_marginal_aoa_comparison.md"].write_text(
        render_comparison_markdown(rows, analysis_id), encoding="utf-8"
    )
    paths["manifest.json"].write_text(
        _json_dump({
            "analysis_id": analysis_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "schema_version": SCHEMA_VERSION,
            "pair_analysis_ids": {
                pair_token(r.source_id, r.target_id): r.analysis_id for r in results
            },
            "input_sha256": dict(input_sha256),
            "output_paths": {name: str(path) for name, path in sorted(paths.items())},
            "label_firewall": dict(LABEL_FIREWALL),
            "limitations": list(LIMITATIONS),
        }),
        encoding="utf-8",
    )
    return paths


# =============================================================================
# Public entry point
# =============================================================================
def run_analysis(
    experiments: Optional[list[str]] = None,
    all_enabled: bool = False,
    dry_run: bool = False,
    force: bool = False,
    output_root: Optional[Path] = None,
    experiments_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Run the directed marginal AoA analysis.

    `output_root` / `experiments_root` are explicit dependency-injection
    points: None means the canonical diagnostics / outputs-experiments root.
    Tests pass tmp_path rather than monkeypatching another module's global.
    """
    validate_feature_contract()
    resolution = resolve_experiment_set(
        experiments=experiments, all_enabled=all_enabled, experiments_root=experiments_root,
    )
    resolved_ids = list(resolution.resolved_ids)
    pairs = ordered_pairs(resolved_ids)

    dataset_paths = {
        experiment_id: resolve_dataset_path(experiment_id, experiments_root)
        for experiment_id in resolved_ids
    }
    for experiment_id, path in dataset_paths.items():
        if not path.is_file():
            raise MarginalAoAError(
                f"'{experiment_id}': frozen Step8A dataset not found: {path}."
            )
    input_sha256_before = {
        experiment_id: sha256_file(path) for experiment_id, path in dataset_paths.items()
    }
    input_paths = {
        experiment_id: str(path) for experiment_id, path in dataset_paths.items()
    }

    if dry_run:
        # Column names come from the parquet FOOTER; no row group is read and
        # nothing is written.
        schema_by_experiment = {}
        for experiment_id, path in dataset_paths.items():
            columns = dataset_schema_columns(path)
            validate_dataset_columns(columns, experiment_id)
            schema_by_experiment[experiment_id] = load_columns_for(columns)
        return {
            "ran": False,
            "dry_run": True,
            "schema_version": SCHEMA_VERSION,
            "support_definition_id": SUPPORT_DEFINITION_ID,
            "selection_mode": resolution.selection_mode,
            "requested_experiment_ids": list(resolution.requested_ids),
            "resolved_experiment_ids": resolved_ids,
            "resolved_experiment_count": len(resolved_ids),
            "excluded_experiments": dict(resolution.excluded),
            "directed_pairs": [
                {"source_experiment_id": s, "target_experiment_id": t,
                 "pair_token": pair_token(s, t)}
                for s, t in pairs
            ],
            "directed_pair_count": len(pairs),
            "primary_population": PRIMARY_POPULATION,
            "feature_contract": feature_contract_payload(),
            "columns_to_load": schema_by_experiment,
            "input_paths": input_paths,
            "input_sha256": input_sha256_before,
            "planned_output_paths": {
                pair_token(s, t): {
                    name: str(path)
                    for name, path in pair_output_paths(s, t, output_root).items()
                }
                for s, t in pairs
            },
            "planned_comparison_output_paths": {
                name: str(path) for name, path in comparison_output_paths(output_root).items()
            },
            "files_written": False,
            "model_fit": False,
            "prediction_run": False,
            "bootstrap_run": False,
            "label_firewall": dict(LABEL_FIREWALL),
            "limitations": list(LIMITATIONS),
        }

    populations = {
        experiment_id: load_population(path, experiment_id)
        for experiment_id, path in dataset_paths.items()
    }

    results = [
        analyse_pair(
            source_id, target_id, populations[source_id], populations[target_id],
            input_sha256_before[source_id], input_sha256_before[target_id],
        )
        for source_id, target_id in pairs
    ]

    written: dict[str, dict[str, str]] = {}
    for result in results:
        paths = write_pair_outputs(
            result, output_root,
            input_paths={
                "source_step8a": input_paths[result.source_id],
                "target_step8a": input_paths[result.target_id],
            },
            input_sha256={
                "source_step8a": input_sha256_before[result.source_id],
                "target_step8a": input_sha256_before[result.target_id],
            },
            force=force,
        )
        written[pair_token(result.source_id, result.target_id)] = {
            name: str(path) for name, path in paths.items()
        }
    comparison_paths = write_comparison_outputs(
        results, output_root, input_sha256_before, force,
    )

    # Frozen inputs must be byte-identical afterwards: this analysis is
    # strictly read-only with respect to Step8A.
    input_sha256_after = {
        experiment_id: sha256_file(path) for experiment_id, path in dataset_paths.items()
    }
    drifted = sorted(
        experiment_id for experiment_id, digest in input_sha256_after.items()
        if digest != input_sha256_before[experiment_id]
    )
    if drifted:
        raise MarginalAoAError(
            "Frozen Step8A input(s) changed during the marginal AoA analysis: "
            f"{drifted}. This analysis must be read-only."
        )

    return {
        "ran": True,
        "dry_run": False,
        "schema_version": SCHEMA_VERSION,
        "support_definition_id": SUPPORT_DEFINITION_ID,
        "selection_mode": resolution.selection_mode,
        "resolved_experiment_ids": resolved_ids,
        "resolved_experiment_count": len(resolved_ids),
        "excluded_experiments": dict(resolution.excluded),
        "directed_pair_count": len(pairs),
        "pair_analysis_ids": {
            pair_token(r.source_id, r.target_id): r.analysis_id for r in results
        },
        "comparison_analysis_id": comparison_analysis_id([r.analysis_id for r in results]),
        "primary_population": PRIMARY_POPULATION,
        "feature_contract": feature_contract_payload(),
        "input_sha256_before": input_sha256_before,
        "input_sha256_after": input_sha256_after,
        "output_paths": written,
        "comparison_output_paths": {
            name: str(path) for name, path in comparison_paths.items()
        },
        "files_written": True,
        "model_fit": False,
        "prediction_run": False,
        "bootstrap_run": False,
        "label_firewall": dict(LABEL_FIREWALL),
        "limitations": list(LIMITATIONS),
    }
