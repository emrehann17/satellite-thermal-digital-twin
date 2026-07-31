"""
Marginal Area-of-Applicability COMPLETION analysis.

Implements the three components the advisor asked for and that
`marginal_aoa.v1` does not provide:

    1. an importance-weighted predictor-space dissimilarity (DIRECTED),
    2. a climatic distance (SYMMETRIC),
    3. a geographic distance (SYMMETRIC).

The exact scientific contract is frozen in
`docs/marginal_aoa_completion_design/` (schema `marginal_aoa_completion.v1`,
implementation_blocker_count = 0). Nothing here re-derives that contract; this
module implements it.

DIAGNOSTIC CLASS
----------------
    target-label-blind, source-model-informed

Feature weights come from a Step8B RandomForest that was fitted on the SOURCE
`burned` label, so this analysis MUST NOT be described as "label-blind". The
target label, target burn date and every transfer metric stay outside the
diagnostic entirely:

    target_label_used                              = False
    target_burn_date_used                          = False
    target_transfer_metric_used                    = False
    source_label_used                              = True
    source_label_read_directly_by_completion_module = False

The source label enters only through a frozen importance CSV; this module never
opens a label column.

NORMALISER
----------
The DI denominator is the MEAN PAIRWISE weighted distance over all distinct
source-reference cell pairs -- NOT any nearest-neighbour mean. A
nearest-neighbour denominator on a dense, autocorrelated 500 m grid measures
grid spacing rather than source spread, and would shrink as an AOI is sampled
more densely, making DI incomparable between AOIs of different size. Spatial
folds are used ONLY for the training DI and the threshold.

THRESHOLD
---------
The operative threshold is the upper whisker
`min(max(training_DI), Q3 + 1.5*IQR)`. The 0.95 quantile is reported as a
secondary sensitivity value and never classifies.

INTERPRETATION BOUNDARY
-----------------------
Every number is DESCRIPTIVE. A low DI does not guarantee transfer success and a
high DI does not prove it caused transfer failure. No composite scalar index is
produced; the three components stay in separate columns.
"""
from __future__ import annotations

import csv
import io
import itertools
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

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
)

# --- Canonical eligibility / population resolution --------------------------
from src.burned_pattern_audit import (
    ANALYSIS_ELIGIBLE_COLUMN,
    BURNABLE_MASK_COLUMN,
    PRE_LABEL_EXCLUDED_COLUMN,
    dataset_schema_columns,
    resolve_analysis_eligible_mask,
)

# --- Canonical spatial-block construction -----------------------------------
from src.step8b_train_baseline_vs_thermal_model import add_spatial_block_id

# --- Canonical provenance helpers -------------------------------------------
from src.step8_large_block_robustness import (
    _git_commit,
    canonical_json,
    sha256_bytes,
    sha256_file,
)

# --- Canonical source standardisation (statistics only, NEVER the imputing
#     transform `apply_regionwise_zscore`) ----------------------------------
from core.step10_shared import EPSILON_STD, compute_regionwise_zscore_stats


SCHEMA_VERSION = "marginal_aoa_completion.v1"
DIAGNOSTIC_NAMESPACE = "marginal_aoa_completion"
DIAGNOSTIC_CLASS = "target_label_blind_source_model_informed"

# The four canonical AOIs of this frozen analysis, in sorted order.
CANONICAL_EXPERIMENTS: tuple[str, ...] = (
    "bejis_2022",
    "evia_2021_extended",
    "manavgat_2021",
    "mugla_2021",
)

PRIMARY_POPULATION = PRIMARY_POPULATIONS[0]
GRID_COLUMNS = ("row_500m", "col_500m")

EXPECTED_DIRECTED_PAIRS = 12
EXPECTED_UNORDERED_PAIRS = 6

# Frozen canonical Step8A hashes. A mismatch is fail-closed.
CANONICAL_STEP8A_SHA256: dict[str, str] = {
    "manavgat_2021": "054a1961fc0582a33d36413263668b63074b21ae8b03d12269b6e228787f3439",
    "bejis_2022": "3dec785a7d8e31db2d67ed283546bbfbca1559f56df46663488d0afc24d9e393",
    "mugla_2021": "c4ab107db2207f9f20775ccc0b3bf39381173fd07d4e82f6821ce7f40be7db8e",
    "evia_2021_extended": "bdce859cf482f575d0f273174b157f47efd61779953fdd23d9486c5face5e553",
}

# --- Stages -----------------------------------------------------------------
STAGES: tuple[str, ...] = (
    "plan",
    "climate-export",
    "weighted-predictor-space",
    "climate-distance",
    "geographic-distance",
    "compare",
)
STAGE_CLIMATE_EXPORT = "climate-export"

# --- Feature importance contract --------------------------------------------
IMPORTANCE_METHOD = "impurity_gini_in_sample_whole_population_v1"
IMPORTANCE_METHOD_CLASS = "impurity_gain"
IMPORTANCE_POPULATION = "burnable_tree_shrub_grass"
IMPORTANCE_MODEL = "thermal"
IMPORTANCE_MODEL_ALGORITHM = "RandomForestClassifier"
IMPORTANCE_NUMERIC_PREFIX = "num__"
IMPORTANCE_CATEGORICAL_PREFIX = "cat__landcover_dominant_"
WEIGHT_SUM_TOLERANCE = 1e-9

# --- Distance / normaliser contract -----------------------------------------
NORMALISER_METHOD = "source_pairwise_mean_distance_v1"
WEIGHTED_DISTANCE_FORMULA_ID = "weighted_euclidean_plus_gower_categorical_v1"
CATEGORICAL_POLICY_ID = "weighted_mismatch_penalty_gower_v1"
NEAREST_NEIGHBOUR_METHOD = "exact"

# Documented, fixed chunk sizes. Deterministic; not tuned on any result.
PAIRWISE_CHUNK_SIZE = 2048
NEIGHBOUR_CHUNK_SIZE = 1024

# --- Fold / threshold contract ----------------------------------------------
FOLD_BLOCK_SIZE_CELLS = 10
FOLD_BLOCK_NOMINAL_SCALE = "approximately_5_km"
FOLD_COUNT = 5
FOLD_ASSIGNMENT_METHOD = "sorted_block_round_robin_5_folds"
PRIMARY_THRESHOLD_METHOD = "source_spatial_fold_holdout_di_upper_whisker_v1"
SECONDARY_Q95_METHOD = "source_spatial_fold_holdout_di_q95_v1"
QUANTILE_METHOD = "linear"

# --- Climate contract -------------------------------------------------------
CLIMATE_COLLECTION = "IDAHO_EPSCOR/TERRACLIMATE"
CLIMATE_PERIOD_START = "1991-01-01"
CLIMATE_PERIOD_END = "2020-12-31"
# Earth Engine filterDate is end-exclusive; this is the same closed period.
CLIMATE_PERIOD_END_EXCLUSIVE = "2021-01-01"
CLIMATE_YEARS: tuple[int, ...] = tuple(range(1991, 2021))
CLIMATE_EXPECTED_MONTHS = 360
CLIMATE_SEASON_MONTHS: tuple[int, ...] = (6, 7, 8, 9)
CLIMATE_LAND_MASK = "terraclimate_native_valid_land_support"
CLIMATE_DISTANCE_METRIC = "standardised_euclidean_mediterranean_reference_v1"
CLIMATE_RASTER_FILENAME = "terraclimate_1991_2020_climate_normals.tif"
# Hidden sibling staging raster. The export lands here and is promoted to the
# final path with a single os.replace ONLY after every QA gate passes.
CLIMATE_RASTER_STAGING_FILENAME = (
    ".terraclimate_1991_2020_climate_normals.export.tmp.tif"
)
CLIMATE_EXPORT_METADATA_FILENAME = "climate_export_metadata.json"

# Official Earth Engine TerraClimate band scale factors.
CLIMATE_BAND_SCALE_FACTORS: dict[str, float] = {
    "tmmn": 0.1,
    "tmmx": 0.1,
    "def": 0.1,
    "vpd": 0.01,
    "pr": 1.0,
}
CLIMATE_SOURCE_BANDS: tuple[str, ...] = ("tmmn", "tmmx", "def", "vpd", "pr")

# Exactly four primary climate variables, in fixed order.
CLIMATE_FEATURES: tuple[str, ...] = (
    "annual_mean_temperature_c",
    "annual_precipitation_mm",
    "warm_season_climatic_water_deficit_mm",
    "warm_season_vpd_kpa",
)
CLIMATE_FEATURE_COUNT = len(CLIMATE_FEATURES)

CLIMATE_REFERENCE_WINDOW: dict[str, float] = {
    "lon_min": -10.0, "lat_min": 30.0, "lon_max": 42.0, "lat_max": 47.0,
}

# --- Geographic contract ----------------------------------------------------
GEODESIC_IMPLEMENTATION = "geographiclib_wgs84"
GEOGRAPHIC_DISTANCE_METHOD = "wgs84_geodesic_inverse_km"
CENTROID_DEFINITION = "bbox_centre_planar_epsg4326"

# Canonical AOI bboxes, hard-pinned from core/regions.py so no Earth Engine
# session is needed. `assert_geometry_matches_registry` proves they have not
# drifted from the registry.
CANONICAL_AOI_BBOX: dict[str, tuple[float, float, float, float]] = {
    "bejis_2022": (-1.05, 39.68, -0.35, 40.15),
    "evia_2021_extended": (23.05, 38.55, 23.85, 39.15),
    "manavgat_2021": (31.05, 36.72, 31.85, 37.35),
    "mugla_2021": (27.10, 36.60, 28.90, 37.45),
}

# --- Transfer comparison contract -------------------------------------------
TRANSFER_DECOMPOSITION_RELATIVE = (
    "diagnostics/four_aoi_transfer_decomposition/"
    "bejis_2022__evia_2021_extended__manavgat_2021__mugla_2021/"
    "four_aoi_decomposition.csv"
)
PRIMARY_TRANSFER_COMPARISON = "raw_thermal_roc_auc"
PRIMARY_TRANSFER_SELECTION: dict[str, str] = {
    "model_family": "thermal",
    "transfer_state": "raw",
    "metric": "roc_auc",
}
SECONDARY_TRANSFER_COMPARISONS: tuple[str, ...] = (
    "raw_thermal_pr_auc",
    "thermal_roc_auc_gap",
    "thermal_pr_auc_gap",
    "adapted_thermal_roc_auc_regionwise_zscore",
    "adapted_thermal_pr_auc_regionwise_zscore",
    "adapted_thermal_roc_auc_coral",
    "adapted_thermal_pr_auc_coral",
    "recovered_fraction",
)

# --- Existing marginal_aoa.v1 linkage ---------------------------------------
MARGINAL_AOA_V1_NAMESPACE = "diagnostics/marginal_area_of_applicability"
MARGINAL_AOA_V1_ANALYSIS_ID = (
    "4a5b8c80489933ba501394d237b2f3d41d96c4a62ad6388a5f1264cc6b545dee"
)

TARGET_LABEL_FIREWALL: dict[str, Any] = {
    "target_label_used": False,
    "target_burn_date_used": False,
    "target_transfer_metric_used": False,
    "target_label_columns_loaded": [],
    "rule": (
        "Every parquet read passes an explicit columns= allow-list containing "
        "only predictor, grid, population-mask and eligibility columns. No "
        "target label or outcome column is loaded at any point."
    ),
}

SOURCE_LABEL_POLICY: dict[str, Any] = {
    "source_label_used": True,
    "source_label_read_directly_by_completion_module": False,
    "mechanism": (
        "Feature weights are RandomForest impurity importances from a Step8B "
        "model fitted on the source `burned` label. This module reads the "
        "frozen importance CSV, never a label column."
    ),
    "diagnostic_class": DIAGNOSTIC_CLASS,
    "forbidden_description": "label-blind",
    "required_description": (
        "target-label-blind, source-model-informed diagnostic"
    ),
}

LIMITATIONS: tuple[str, ...] = (
    "The weighting is source-model-informed: feature weights come from a model "
    "fitted on the SOURCE label. This is not a label-free geometric statement.",
    "Impurity importance is computed in-sample and is biased toward continuous "
    "predictors relative to low-cardinality categorical ones.",
    "A low DI does not guarantee transfer success; a high DI does not prove it "
    "caused transfer failure. No causal relationship is established.",
    "The mixed numeric/categorical distance inherits the standard Gower scale "
    "asymmetry: the numeric block is unbounded, the categorical term is "
    "bounded by w_landcover.",
    "The DI scale is set by the mean pairwise source distance, a property of "
    "the source distribution alone; the THRESHOLD, unlike the scale, depends "
    "on the source spatial-fold structure.",
    "DI is a nearest-neighbour distance and says nothing about the DENSITY of "
    "source data near a target cell.",
    "Climatic and geographic distances are symmetric AOI-level values; six "
    "unordered distances from four AOIs describe a pattern, they establish no "
    "relationship with transfer performance.",
    "Nothing here supersedes marginal_aoa.v1: marginal range support and joint "
    "nearest-neighbour dissimilarity answer different questions.",
)


class MarginalAoACompletionError(SystemExit):
    """Fail-fast, contract-violating condition."""


# =============================================================================
# Feature contract
# =============================================================================
def numeric_features() -> tuple[str, ...]:
    return tuple(f for f in SHARED_THERMAL_MODEL_FEATURES if f not in CATEGORICAL_FEATURES)


def categorical_features() -> tuple[str, ...]:
    return tuple(f for f in SHARED_THERMAL_MODEL_FEATURES if f in CATEGORICAL_FEATURES)


def all_features() -> tuple[str, ...]:
    return tuple(SHARED_THERMAL_MODEL_FEATURES)


def categorical_feature() -> str:
    features = categorical_features()
    if len(features) != 1:
        raise MarginalAoACompletionError(
            f"This analysis requires exactly one categorical feature; got {list(features)}."
        )
    return features[0]


def validate_feature_contract() -> None:
    numeric = numeric_features()
    if len(numeric) != 9:
        raise MarginalAoACompletionError(
            f"Expected 9 numeric predictors in the frozen contract; got {len(numeric)}: {list(numeric)}."
        )
    leaked = sorted(set(all_features()) & set(FORBIDDEN_MODEL_COLUMNS))
    if leaked:
        raise MarginalAoACompletionError(
            f"Feature contract leaks forbidden column(s): {leaked}."
        )


def feature_contract_payload() -> dict[str, Any]:
    return {
        "source": "src.step9a_audit_cross_region_inputs.SHARED_THERMAL_MODEL_FEATURES",
        "numeric_features": list(numeric_features()),
        "categorical_features": list(categorical_features()),
        "numeric_feature_count": len(numeric_features()),
        "categorical_feature_count": len(categorical_features()),
        "primary_population": PRIMARY_POPULATION,
    }


# =============================================================================
# Stage validation -- applied BEFORE any prerequisite check
# =============================================================================
def validate_stage_range(from_stage: str, to_stage: str) -> list[str]:
    """Resolve --from-stage/--to-stage into an ordered STAGES sub-sequence.

    Fail-closed on an unknown stage or a reversed range. This runs before any
    prerequisite or filesystem inspection, so a malformed request can never
    reach input resolution.
    """
    if from_stage not in STAGES:
        raise MarginalAoACompletionError(
            f"Unknown --from-stage '{from_stage}'. Valid stages, in order: {list(STAGES)}."
        )
    if to_stage not in STAGES:
        raise MarginalAoACompletionError(
            f"Unknown --to-stage '{to_stage}'. Valid stages, in order: {list(STAGES)}."
        )
    start, end = STAGES.index(from_stage), STAGES.index(to_stage)
    if start > end:
        raise MarginalAoACompletionError(
            f"--from-stage ('{from_stage}') cannot come after --to-stage ('{to_stage}'); "
            f"stage order is {list(STAGES)}."
        )
    return list(STAGES[start:end + 1])


def stage_side_effect_flags(stages: Sequence[str]) -> dict[str, bool]:
    """Only `climate-export` may touch Earth Engine. Everything else is local."""
    runs_export = STAGE_CLIMATE_EXPORT in stages
    return {
        "gee_queries_run": bool(runs_export),
        "gee_exports_run": bool(runs_export),
        "model_fit": False,
        "bootstrap_run": False,
    }


# =============================================================================
# Paths
# =============================================================================
def diagnostics_root(output_root: Optional[Path] = None) -> Path:
    root = Path(output_root) if output_root is not None else PROJECT_ROOT / "outputs"
    return root / "diagnostics" / DIAGNOSTIC_NAMESPACE


def analysis_root(analysis_id: str, output_root: Optional[Path] = None) -> Path:
    return diagnostics_root(output_root) / analysis_id


def canonical_step8a_path(experiment_id: str, experiments_root: Optional[Path] = None) -> Path:
    root = (
        Path(experiments_root) if experiments_root is not None
        else PROJECT_ROOT / "outputs" / "experiments"
    )
    return root / experiment_id / "step8a" / "step8a_500m_modeling_dataset.parquet"


def canonical_importance_path(experiment_id: str, experiments_root: Optional[Path] = None) -> Path:
    root = (
        Path(experiments_root) if experiments_root is not None
        else PROJECT_ROOT / "outputs" / "experiments"
    )
    return root / experiment_id / "step8b" / "step8b_feature_importance.csv"


def transfer_decomposition_path(output_root: Optional[Path] = None) -> Path:
    root = Path(output_root) if output_root is not None else PROJECT_ROOT / "outputs"
    return root / TRANSFER_DECOMPOSITION_RELATIVE


def marginal_aoa_v1_root(output_root: Optional[Path] = None) -> Path:
    root = Path(output_root) if output_root is not None else PROJECT_ROOT / "outputs"
    return root / MARGINAL_AOA_V1_NAMESPACE


def climate_raster_path(analysis_id: str, output_root: Optional[Path] = None) -> Path:
    return analysis_root(analysis_id, output_root) / "climate_distance" / CLIMATE_RASTER_FILENAME


def planned_output_layout() -> dict[str, str]:
    """Every relative path this analysis may write, keyed by role."""
    return {
        "config/preregistration.json": "plan",
        "config/frozen_input_inventory.json": "plan",
        "config/feature_importance_inventory.json": "plan",
        "config/climate_input_inventory.json": "plan",
        "config/geometry_inventory.json": "plan",
        "config/transfer_input_inventory.json": "plan",
        "plan_stage_metadata.json": "plan",
        f"climate_distance/{CLIMATE_RASTER_FILENAME}": "climate-export",
        "climate_distance/climate_export_metadata.json": "climate-export",
        "weighted_predictor_space/source_feature_weights.csv": "weighted-predictor-space",
        "weighted_predictor_space/source_threshold_diagnostics.csv": "weighted-predictor-space",
        "weighted_predictor_space/target_cell_dissimilarity.parquet": "weighted-predictor-space",
        "weighted_predictor_space/directed_pair_summary.csv": "weighted-predictor-space",
        "climate_distance/aoi_climate_vectors.csv": "climate-distance",
        "climate_distance/pairwise_climate_distance.csv": "climate-distance",
        "geographic_distance/aoi_geometry_summary.csv": "geographic-distance",
        "geographic_distance/pairwise_geographic_distance.csv": "geographic-distance",
        "comparison/marginal_diagnostics_with_transfer.csv": "compare",
        "comparison/ranking_summary.csv": "compare",
        "comparison/scientific_summary.md": "compare",
        "completion_metadata.json": "compare",
    }


# =============================================================================
# Pair construction
# =============================================================================
def directed_pairs(experiment_ids: Sequence[str]) -> list[tuple[str, str]]:
    """All ordered source->target pairs over the SORTED ids.

    Sorting first means the caller's argument order can never change the
    result; permutations (not combinations) means A->B and B->A are distinct.
    """
    ordered = sorted(experiment_ids)
    if len(set(ordered)) != len(ordered):
        raise MarginalAoACompletionError(
            f"Duplicate experiment id(s) in {list(experiment_ids)}."
        )
    return [(s, t) for s, t in itertools.permutations(ordered, 2)]


def unordered_pairs(experiment_ids: Sequence[str]) -> list[tuple[str, str]]:
    return list(itertools.combinations(sorted(experiment_ids), 2))


def pair_token(source_id: str, target_id: str) -> str:
    """Never sorted -- the token itself carries the direction."""
    return f"{source_id}__{target_id}"


def direction_token(source_id: str, target_id: str) -> str:
    return f"{source_id}_to_{target_id}"


def resolve_experiments(experiments: Optional[Sequence[str]] = None) -> list[str]:
    """This frozen analysis is defined over exactly four canonical AOIs."""
    if experiments is None:
        return list(CANONICAL_EXPERIMENTS)
    resolved = sorted(experiments)
    if tuple(resolved) != CANONICAL_EXPERIMENTS:
        raise MarginalAoACompletionError(
            f"This frozen analysis requires exactly the canonical experiment set "
            f"{list(CANONICAL_EXPERIMENTS)}; got {resolved}."
        )
    return resolved


# =============================================================================
# Frozen input inventory
# =============================================================================
def build_frozen_input_inventory(
    experiment_ids: Sequence[str], experiments_root: Optional[Path] = None
) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for experiment_id in sorted(experiment_ids):
        path = canonical_step8a_path(experiment_id, experiments_root)
        exists = path.is_file()
        entries[experiment_id] = {
            "experiment_id": experiment_id,
            "path": str(path),
            "exists": exists,
            "sha256": sha256_file(path) if exists else None,
            "expected_sha256": CANONICAL_STEP8A_SHA256.get(experiment_id),
        }
    return entries


def assert_canonical_step8a_hashes(inventory: dict[str, Any], *, strict: bool = True) -> dict[str, Any]:
    """Verify each Step8A dataset against its frozen hash. Fail-closed.

    `strict=False` is used only by synthetic tests, which build their own
    Step8A tree and pin its hashes through `expected_sha256` instead.
    """
    missing = sorted(k for k, v in inventory.items() if not v["exists"])
    if missing:
        raise MarginalAoACompletionError(
            f"Canonical Step8A dataset(s) not found for {missing}. "
            "The completion analysis is read-only against frozen Step8A inputs."
        )
    if not strict:
        return {"verified": False, "reason": "strict_hash_check_disabled_for_injected_root"}

    mismatches = []
    for experiment_id, entry in sorted(inventory.items()):
        expected = entry.get("expected_sha256")
        if expected is None:
            mismatches.append(
                f"{experiment_id}: no frozen expected hash is registered"
            )
        elif entry["sha256"] != expected:
            mismatches.append(
                f"{experiment_id}: expected {expected}, found {entry['sha256']}"
            )
    if mismatches:
        raise MarginalAoACompletionError(
            "Canonical Step8A hash verification FAILED -- refusing to run against "
            "inputs that are not the frozen ones:\n  " + "\n  ".join(mismatches)
        )
    return {"verified": True, "n_verified": len(inventory)}


# =============================================================================
# Feature importance -> weights
# =============================================================================
def expected_importance_rows(levels: Sequence[str]) -> list[str]:
    numeric = [f"{IMPORTANCE_NUMERIC_PREFIX}{f}" for f in numeric_features()]
    categorical = [f"{IMPORTANCE_CATEGORICAL_PREFIX}{lvl}" for lvl in levels]
    return sorted(numeric + categorical)


def read_importance_frame(path: Path, experiment_id: str) -> pd.DataFrame:
    if not Path(path).is_file():
        raise MarginalAoACompletionError(
            f"'{experiment_id}': Step8B feature-importance CSV not found: {path}."
        )
    frame = pd.read_csv(path)
    required = {"population", "model", "feature", "importance"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise MarginalAoACompletionError(
            f"'{experiment_id}': importance CSV {path} is missing column(s) {missing}."
        )
    subset = frame[
        (frame["population"] == IMPORTANCE_POPULATION)
        & (frame["model"] == IMPORTANCE_MODEL)
    ].copy()
    if subset.empty:
        raise MarginalAoACompletionError(
            f"'{experiment_id}': importance CSV {path} has no rows for "
            f"(population={IMPORTANCE_POPULATION}, model={IMPORTANCE_MODEL})."
        )
    return subset


def derive_feature_weights(
    importance_frame: pd.DataFrame, experiment_id: str
) -> dict[str, Any]:
    """Nine numeric weights plus ONE grouped landcover weight.

    The landcover dummy importances are SUMMED into a single group weight, so
    the categorical predictor carries exactly the influence the source model
    gave it, independently of how many levels happened to be observed. That
    K-invariance is what makes a 7-level source comparable with an 8-level one.
    """
    duplicated = importance_frame["feature"].duplicated(keep=False)
    if duplicated.any():
        dups = sorted(importance_frame.loc[duplicated, "feature"].unique())
        raise MarginalAoACompletionError(
            f"'{experiment_id}': duplicate importance row(s) for feature(s) {dups}."
        )

    values = pd.to_numeric(importance_frame["importance"], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype="float64")).all():
        bad = sorted(importance_frame.loc[~np.isfinite(values), "feature"].tolist())
        raise MarginalAoACompletionError(
            f"'{experiment_id}': non-finite importance value(s) for {bad}."
        )
    if (values < 0).any():
        bad = sorted(importance_frame.loc[values < 0, "feature"].tolist())
        raise MarginalAoACompletionError(
            f"'{experiment_id}': NEGATIVE importance value(s) for {bad}. "
            "RandomForest mean-decrease-in-impurity is non-negative by "
            "construction, so this artifact is not what the contract expects. "
            "Failing closed rather than silently clipping."
        )

    lookup = dict(zip(importance_frame["feature"], values.astype(float)))
    observed = sorted(lookup)

    numeric_rows = [f"{IMPORTANCE_NUMERIC_PREFIX}{f}" for f in numeric_features()]
    missing_numeric = [r for r in numeric_rows if r not in lookup]
    if missing_numeric:
        raise MarginalAoACompletionError(
            f"'{experiment_id}': importance artifact is missing required numeric "
            f"row(s) {missing_numeric}."
        )

    dummy_rows = [f for f in observed if f.startswith(IMPORTANCE_CATEGORICAL_PREFIX)]
    if not dummy_rows:
        raise MarginalAoACompletionError(
            f"'{experiment_id}': importance artifact has no "
            f"'{IMPORTANCE_CATEGORICAL_PREFIX}*' row; the categorical predictor "
            "weight cannot be derived."
        )

    unexpected = sorted(set(observed) - set(numeric_rows) - set(dummy_rows))
    if unexpected:
        raise MarginalAoACompletionError(
            f"'{experiment_id}': unexpected importance row(s) {unexpected}; the "
            "expected row set is the 9 num__ features plus the observed "
            "cat__landcover_dominant_* levels."
        )

    raw: dict[str, float] = {f: lookup[f"{IMPORTANCE_NUMERIC_PREFIX}{f}"] for f in numeric_features()}
    dummy_contributions = {
        row[len(IMPORTANCE_CATEGORICAL_PREFIX):]: lookup[row] for row in sorted(dummy_rows)
    }
    raw[categorical_feature()] = float(sum(dummy_contributions.values()))

    total = float(sum(raw.values()))
    if not math.isfinite(total) or total <= 0.0:
        raise MarginalAoACompletionError(
            f"'{experiment_id}': importance values sum to {total}; cannot normalise."
        )
    if abs(total - 1.0) > 1e-6:
        raise MarginalAoACompletionError(
            f"'{experiment_id}': importance rows sum to {total!r}, not 1.0. "
            "sklearn normalises feature_importances_, so this artifact is not "
            "the expected one."
        )

    weights = {feature: value / total for feature, value in raw.items()}
    weight_sum = float(sum(weights.values()))
    if abs(weight_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise MarginalAoACompletionError(
            f"'{experiment_id}': normalised weights sum to {weight_sum!r}, "
            f"outside tolerance {WEIGHT_SUM_TOLERANCE}."
        )

    positive = [f for f, w in weights.items() if w > 0.0]
    entropy = -sum(w * math.log(w) for w in weights.values() if w > 0.0)

    return {
        "experiment_id": experiment_id,
        "weights": weights,
        "raw_importance": raw,
        "dummy_level_contributions": dummy_contributions,
        "observed_levels": sorted(dummy_contributions),
        "renormalisation_factor": 1.0 / total,
        "weight_sum": weight_sum,
        "zero_weight_features": sorted(f for f, w in weights.items() if w == 0.0),
        "n_features_with_positive_weight": len(positive),
        "feature_weight_entropy": entropy,
        "effective_feature_count_perplexity": math.exp(entropy),
        "importance_method": IMPORTANCE_METHOD,
        "importance_population": IMPORTANCE_POPULATION,
        "importance_model": IMPORTANCE_MODEL,
    }


def build_feature_importance_inventory(
    experiment_ids: Sequence[str], experiments_root: Optional[Path] = None
) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for experiment_id in sorted(experiment_ids):
        path = canonical_importance_path(experiment_id, experiments_root)
        exists = path.is_file()
        entries[experiment_id] = {
            "experiment_id": experiment_id,
            "path": str(path),
            "exists": exists,
            "sha256": sha256_file(path) if exists else None,
            "population_filter": IMPORTANCE_POPULATION,
            "model_filter": IMPORTANCE_MODEL,
            "importance_method": IMPORTANCE_METHOD,
            "importance_method_class": IMPORTANCE_METHOD_CLASS,
            "model_algorithm": IMPORTANCE_MODEL_ALGORITHM,
            "source_label_used": True,
            "source_label_read_directly_by_completion_module": False,
        }
    return entries


# =============================================================================
# Population loading -- explicit allow-list, no label column, no imputation
# =============================================================================
def load_columns_for(schema_columns: Iterable[str]) -> list[str]:
    """The EXPLICIT parquet allow-list. Never contains a label column."""
    present = set(schema_columns)
    columns: list[str] = list(all_features())
    columns.extend(GRID_COLUMNS)
    columns.append(BURNABLE_MASK_COLUMN)
    for optional in (ANALYSIS_ELIGIBLE_COLUMN, PRE_LABEL_EXCLUDED_COLUMN):
        if optional in present:
            columns.append(optional)

    ordered = list(dict.fromkeys(columns))
    leaked = sorted((set(ordered) & set(FORBIDDEN_MODEL_COLUMNS)) - set(GRID_COLUMNS))
    if leaked:
        raise MarginalAoACompletionError(
            f"Label firewall violation: refusing to read column(s) {leaked}."
        )
    return ordered


def load_population(
    path: Path, experiment_id: str, *, read_parquet=None
) -> pd.DataFrame:
    """Load ONLY allow-listed columns and reduce to the primary population.

    `read_parquet` is resolved at CALL time so a test patching
    `pandas.read_parquet` observes the exact `columns=` argument.
    """
    read_parquet = pd.read_parquet if read_parquet is None else read_parquet
    if not Path(path).is_file():
        raise MarginalAoACompletionError(
            f"'{experiment_id}': frozen Step8A dataset not found: {path}."
        )
    schema_columns = dataset_schema_columns(Path(path))
    required = list(all_features()) + list(GRID_COLUMNS) + [BURNABLE_MASK_COLUMN]
    missing = [c for c in required if c not in set(schema_columns)]
    if missing:
        raise MarginalAoACompletionError(
            f"'{experiment_id}': frozen Step8A dataset is missing required "
            f"column(s) {missing}."
        )
    columns = load_columns_for(schema_columns)
    frame = read_parquet(path, columns=columns)

    eligible = resolve_analysis_eligible_mask(frame)
    valid_grid = frame["row_500m"].notna() & frame["col_500m"].notna()
    primary = frame[BURNABLE_MASK_COLUMN].astype(bool)
    population = frame.loc[eligible & valid_grid & primary].copy()

    if population.empty:
        raise MarginalAoACompletionError(
            f"'{experiment_id}': the primary population is empty."
        )
    key = population[list(GRID_COLUMNS)]
    if key.duplicated(keep=False).any():
        raise MarginalAoACompletionError(
            f"'{experiment_id}': duplicate (row_500m, col_500m) grid cell(s); "
            "the frozen Step8A grid must be unique."
        )
    population = population.sort_values(list(GRID_COLUMNS), kind="mergesort")
    return population.reset_index(drop=True)


def complete_predictor_mask(frame: pd.DataFrame) -> np.ndarray:
    """True where EVERY predictor is present. No imputation anywhere."""
    mask = np.ones(len(frame), dtype=bool)
    for feature in numeric_features():
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype="float64")
        mask &= np.isfinite(values)
    levels = canonical_levels(frame[categorical_feature()])
    mask &= np.array([lvl is not None for lvl in levels], dtype=bool)
    return mask


def canonical_level(value: Any) -> Optional[str]:
    """One canonical string form per level, so 80 / 80.0 / '80' never split."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return None
        if float(value).is_integer():
            return str(int(float(value)))
        return repr(float(value))
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "<na>"}:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return text
    if math.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text


def canonical_levels(values: pd.Series) -> list[Optional[str]]:
    return [canonical_level(v) for v in values.tolist()]


# =============================================================================
# Source standardisation
# =============================================================================
def build_source_scaling(frame: pd.DataFrame, experiment_id: str) -> dict[str, Any]:
    """Source mean / population SD (ddof=0) with the EPSILON_STD guard.

    Reuses `compute_regionwise_zscore_stats` -- the repository's tested,
    label-blind statistics helper -- but NEVER `apply_regionwise_zscore`,
    which imputes missing values with the region mean. Imputation here would
    either make target coordinates depend on the target distribution or place
    every incomplete target cell at the source centroid.
    """
    numeric = list(numeric_features())
    stats = compute_regionwise_zscore_stats(frame[numeric], numeric)
    scaling: dict[str, Any] = {}
    for feature in numeric:
        entry = stats[feature]
        scaling[feature] = {
            "mean": float(entry["mean"]),
            "scale": float(entry["std"]),
            "raw_std": float(entry["raw_std"]),
            "constant_feature_guard_used": bool(entry["constant_feature_guard_used"]),
            "n_observed": int(entry["n_observed"]),
            "source_scale_method": (
                "constant_guard" if entry["constant_feature_guard_used"]
                else "source_population_sd_ddof0"
            ),
        }
    return scaling


def standardise(frame: pd.DataFrame, scaling: dict[str, Any]) -> np.ndarray:
    """z = (x - source_mean) / source_scale. SOURCE statistics, both sides."""
    numeric = list(numeric_features())
    out = np.empty((len(frame), len(numeric)), dtype="float64")
    for j, feature in enumerate(numeric):
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype="float64")
        entry = scaling[feature]
        out[:, j] = (values - entry["mean"]) / entry["scale"]
    return out


def weighted_coordinates(z: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    """Multiply each standardised column by sqrt(w_j).

    Euclidean distance in this space equals the weighted numeric distance;
    the categorical term is handled separately and is NOT a coordinate.
    """
    numeric = list(numeric_features())
    factors = np.array([math.sqrt(max(weights[f], 0.0)) for f in numeric], dtype="float64")
    return z * factors[None, :]


def encode_levels(levels: Sequence[Optional[str]], vocabulary: dict[str, int]) -> np.ndarray:
    """Integer codes for level identity ONLY.

    The codes are comparison tokens; they are never treated as magnitudes and
    never enter an arithmetic expression. Unknown levels get a unique negative
    code so they mismatch every source level.
    """
    out = np.empty(len(levels), dtype="int64")
    unseen = -1
    for i, level in enumerate(levels):
        if level is None:
            out[i] = -(10 ** 9)
        elif level in vocabulary:
            out[i] = vocabulary[level]
        else:
            out[i] = unseen
            unseen -= 1
    return out


# =============================================================================
# Weighted distance kernels -- exact, deterministic, no sampling
# =============================================================================
def _pair_distances(
    a_coords: np.ndarray, a_codes: np.ndarray,
    b_coords: np.ndarray, b_codes: np.ndarray,
    w_landcover: float,
) -> np.ndarray:
    """Full mixed weighted distance matrix between two blocks.

        d = sqrt( ||za - zb||^2  +  w_landcover * 1[level_a != level_b] )

    The numeric block is already sqrt(w)-scaled, so the squared Euclidean norm
    IS the weighted numeric term.
    """
    diff2 = (
        np.sum(a_coords ** 2, axis=1)[:, None]
        + np.sum(b_coords ** 2, axis=1)[None, :]
        - 2.0 * (a_coords @ b_coords.T)
    )
    np.maximum(diff2, 0.0, out=diff2)
    mismatch = (a_codes[:, None] != b_codes[None, :])
    d2 = diff2 + w_landcover * mismatch
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def source_pairwise_mean_distance(
    coords: np.ndarray, codes: np.ndarray, w_landcover: float,
    *, chunk_size: int = PAIRWISE_CHUNK_SIZE,
) -> dict[str, Any]:
    """Mean distance over every DISTINCT source-reference pair.

    Exact and deterministic: a fixed outer block order, the strict upper
    triangle within a block, all pairs across blocks, float64 accumulation.
    Never materialises an n x n matrix. Self-distance is excluded by
    construction, and the accumulated pair count is verified against the exact
    combinatorial count n*(n-1)/2.
    """
    n = int(coords.shape[0])
    if n < 2:
        raise MarginalAoACompletionError(
            f"The source reference set has {n} row(s); at least 2 are needed for "
            "a pairwise mean distance."
        )
    chunk = max(int(chunk_size), 1)
    starts = list(range(0, n, chunk))

    total = 0.0
    count = 0
    for bi, start_i in enumerate(starts):
        stop_i = min(start_i + chunk, n)
        ci, ki = coords[start_i:stop_i], codes[start_i:stop_i]
        for start_j in starts[bi:]:
            stop_j = min(start_j + chunk, n)
            cj, kj = coords[start_j:stop_j], codes[start_j:stop_j]
            block = _pair_distances(ci, ki, cj, kj, w_landcover)
            if start_i == start_j:
                iu = np.triu_indices(block.shape[0], k=1)
                total += float(block[iu].sum())
                count += int(iu[0].size)
            else:
                total += float(block.sum())
                count += int(block.size)

    expected = n * (n - 1) // 2
    if count != expected:
        raise MarginalAoACompletionError(
            f"Pairwise accumulation visited {count} pairs but the exact "
            f"combinatorial count is {expected}; the chunked traversal is wrong."
        )
    return {
        "source_pairwise_mean_distance": total / expected,
        "n_distinct_source_pairs": expected,
        "accumulated_pair_count": count,
        "chunk_size": chunk,
        "normaliser_method": NORMALISER_METHOD,
        "normaliser_uses_folds": False,
    }


def nearest_distances(
    query_coords: np.ndarray, query_codes: np.ndarray,
    ref_coords: np.ndarray, ref_codes: np.ndarray,
    w_landcover: float, *, chunk_size: int = NEIGHBOUR_CHUNK_SIZE,
    ref_mask: Optional[np.ndarray] = None,
    return_index: bool = False,
):
    """Exact nearest weighted distance from every query row to the reference set.

    `ref_mask` (n_query x n_ref boolean) excludes reference rows per query --
    used for the fold holdout, where a cell may not match its own fold. Exact
    brute force in fixed chunks: no tree, no approximation, no sampling, no
    seed. Ties are broken by the smallest reference index, deterministically.
    """
    n_query = int(query_coords.shape[0])
    n_ref = int(ref_coords.shape[0])
    if n_ref == 0:
        raise MarginalAoACompletionError(
            "Nearest-neighbour query against an EMPTY reference set."
        )
    chunk = max(int(chunk_size), 1)
    best = np.full(n_query, np.inf, dtype="float64")
    best_idx = np.full(n_query, -1, dtype="int64")

    for start in range(0, n_query, chunk):
        stop = min(start + chunk, n_query)
        block = _pair_distances(
            query_coords[start:stop], query_codes[start:stop],
            ref_coords, ref_codes, w_landcover,
        )
        if ref_mask is not None:
            block = np.where(ref_mask[start:stop], block, np.inf)
        idx = np.argmin(block, axis=1)
        vals = block[np.arange(stop - start), idx]
        best[start:stop] = vals
        best_idx[start:stop] = idx

    if not np.isfinite(best).all():
        raise MarginalAoACompletionError(
            "Nearest-neighbour search produced a non-finite distance; at least "
            "one query row had no admissible reference cell."
        )
    return (best, best_idx) if return_index else best


# =============================================================================
# Spatial folds -- label-free, seed-free, deterministic
# =============================================================================
def assign_spatial_folds(
    frame: pd.DataFrame, *, block_size_cells: int = FOLD_BLOCK_SIZE_CELLS,
    fold_count: int = FOLD_COUNT,
) -> dict[str, Any]:
    """Sorted-block round-robin. A block is assigned WHOLLY to one fold.

    Deterministic by inspection: no seed, no shuffle, and no label -- unlike
    Step8B's `fold_id`, which comes from StratifiedGroupKFold and consumes `y`.
    """
    blocked = add_spatial_block_id(frame, block_size_cells)
    block_ids = blocked["spatial_block_id"].astype(str).to_numpy()
    unique_blocks = sorted(set(block_ids.tolist()))
    if len(unique_blocks) < fold_count:
        raise MarginalAoACompletionError(
            f"Only {len(unique_blocks)} spatial block(s) at block_size_cells="
            f"{block_size_cells}; {fold_count} folds cannot be formed."
        )
    block_to_fold = {b: i % fold_count for i, b in enumerate(unique_blocks)}
    folds = np.array([block_to_fold[b] for b in block_ids], dtype="int64")

    sizes = [int((folds == k).sum()) for k in range(fold_count)]
    if min(sizes) == 0:
        raise MarginalAoACompletionError(
            f"Fold assignment produced an empty fold (sizes={sizes})."
        )
    return {
        "fold_of_row": folds,
        "block_id_of_row": block_ids,
        "n_blocks": len(unique_blocks),
        "fold_sizes": sizes,
        "fold_count": fold_count,
        "block_size_cells": block_size_cells,
        "block_nominal_scale": FOLD_BLOCK_NOMINAL_SCALE,
        "fold_assignment_method": FOLD_ASSIGNMENT_METHOD,
        "fold_assignment_reads_label": False,
        "block_split_across_folds": False,
    }


def training_dissimilarity(
    coords: np.ndarray, codes: np.ndarray, folds: np.ndarray,
    w_landcover: float, normaliser: float,
    *, chunk_size: int = NEIGHBOUR_CHUNK_SIZE,
) -> np.ndarray:
    """training_DI(s) = holdout_nearest_distance(s) / source_pairwise_mean_distance.

    The denominator is the PAIRWISE mean, not a holdout statistic, so training
    DI and target DI live on exactly the same scale.
    """
    n = int(coords.shape[0])
    out = np.full(n, np.nan, dtype="float64")
    for fold in sorted(set(folds.tolist())):
        in_fold = folds == fold
        out_fold = ~in_fold
        if not out_fold.any():
            raise MarginalAoACompletionError(
                f"Fold {fold} has no out-of-fold reference cells."
            )
        distances = nearest_distances(
            coords[in_fold], codes[in_fold],
            coords[out_fold], codes[out_fold],
            w_landcover, chunk_size=chunk_size,
        )
        out[in_fold] = distances
    if not np.isfinite(out).all():
        raise MarginalAoACompletionError(
            "Training DI contains a non-finite value."
        )
    return out / normaliser


def upper_whisker_threshold(training_di: np.ndarray) -> dict[str, Any]:
    """Operative threshold = min(max(training_DI), Q3 + 1.5*IQR).

    The whisker adapts to the distribution's shape, and the clamp guarantees
    the threshold never exceeds anything actually observed in the source.
    """
    values = np.asarray(training_di, dtype="float64")
    q1 = float(np.quantile(values, 0.25, method=QUANTILE_METHOD))
    q3 = float(np.quantile(values, 0.75, method=QUANTILE_METHOD))
    iqr = q3 - q1
    unclamped = q3 + 1.5 * iqr
    maximum = float(values.max())
    threshold = min(maximum, unclamped)
    return {
        "training_di_q1": q1,
        "training_di_q3": q3,
        "training_di_iqr": iqr,
        "training_di_q50_threshold": float(np.quantile(values, 0.50, method=QUANTILE_METHOD)),
        "training_di_q90_threshold": float(np.quantile(values, 0.90, method=QUANTILE_METHOD)),
        "training_di_q95_threshold": float(np.quantile(values, 0.95, method=QUANTILE_METHOD)),
        "training_di_q99_threshold": float(np.quantile(values, 0.99, method=QUANTILE_METHOD)),
        "training_di_max_threshold": maximum,
        "upper_whisker_unclamped": unclamped,
        "upper_whisker_clamped_to_max": bool(threshold == maximum and unclamped > maximum),
        "training_di_upper_whisker_threshold": threshold,
        "primary_threshold_method": PRIMARY_THRESHOLD_METHOD,
        "training_di_q95_method": SECONDARY_Q95_METHOD,
        "q95_is_operative": False,
    }


# =============================================================================
# Per-source preparation
# =============================================================================
def prepare_source(
    experiment_id: str, population: pd.DataFrame, weights: dict[str, float],
    *, pairwise_chunk_size: int = PAIRWISE_CHUNK_SIZE,
    neighbour_chunk_size: int = NEIGHBOUR_CHUNK_SIZE,
) -> dict[str, Any]:
    """Reference set, scaling, normaliser, folds, training DI and threshold."""
    complete = complete_predictor_mask(population)
    reference = population.loc[complete].reset_index(drop=True)
    if reference.empty:
        raise MarginalAoACompletionError(
            f"'{experiment_id}': no source row has a complete predictor vector."
        )

    scaling = build_source_scaling(reference, experiment_id)
    coords = weighted_coordinates(standardise(reference, scaling), weights)

    levels = canonical_levels(reference[categorical_feature()])
    vocabulary = {lvl: i for i, lvl in enumerate(sorted({l for l in levels if l is not None}))}
    codes = encode_levels(levels, vocabulary)
    w_landcover = float(weights[categorical_feature()])

    normaliser = source_pairwise_mean_distance(
        coords, codes, w_landcover, chunk_size=pairwise_chunk_size,
    )
    folds = assign_spatial_folds(reference)
    training_di = training_dissimilarity(
        coords, codes, folds["fold_of_row"], w_landcover,
        normaliser["source_pairwise_mean_distance"], chunk_size=neighbour_chunk_size,
    )
    threshold = upper_whisker_threshold(training_di)

    return {
        "experiment_id": experiment_id,
        "reference_frame": reference,
        "coords": coords,
        "codes": codes,
        "level_vocabulary": vocabulary,
        "w_landcover": w_landcover,
        "_weights": dict(weights),
        "scaling": scaling,
        "normaliser": normaliser,
        "folds": folds,
        "training_di": training_di,
        "threshold": threshold,
        "source_rows_total": int(len(population)),
        "source_rows_reference": int(len(reference)),
        "source_rows_excluded_missing": int(len(population) - len(reference)),
    }


def source_threshold_row(prepared: dict[str, Any]) -> dict[str, Any]:
    normaliser, folds, threshold = prepared["normaliser"], prepared["folds"], prepared["threshold"]
    row = {
        "source_experiment": prepared["experiment_id"],
        "source_rows_total": prepared["source_rows_total"],
        "source_rows_reference": prepared["source_rows_reference"],
        "source_rows_excluded_missing": prepared["source_rows_excluded_missing"],
        "n_distinct_source_pairs": normaliser["n_distinct_source_pairs"],
        "accumulated_pair_count": normaliser["accumulated_pair_count"],
        "chunk_size": normaliser["chunk_size"],
        "source_pairwise_mean_distance": normaliser["source_pairwise_mean_distance"],
        "source_distance_normaliser": normaliser["source_pairwise_mean_distance"],
        "normaliser_method": normaliser["normaliser_method"],
        "normaliser_uses_folds": normaliser["normaliser_uses_folds"],
        "holdout_block_size_cells": folds["block_size_cells"],
        "holdout_block_nominal_scale": folds["block_nominal_scale"],
        "holdout_fold_count": folds["fold_count"],
        "fold_assignment_method": folds["fold_assignment_method"],
        "fold_assignment_reads_label": folds["fold_assignment_reads_label"],
        "block_split_across_folds": folds["block_split_across_folds"],
        "n_blocks": folds["n_blocks"],
    }
    row.update(threshold)
    return row


SOURCE_THRESHOLD_COLUMNS: tuple[str, ...] = (
    "source_experiment", "source_rows_total", "source_rows_reference",
    "source_rows_excluded_missing", "n_distinct_source_pairs",
    "accumulated_pair_count", "chunk_size", "source_pairwise_mean_distance",
    "source_distance_normaliser", "normaliser_method", "normaliser_uses_folds",
    "holdout_block_size_cells", "holdout_block_nominal_scale",
    "holdout_fold_count", "fold_assignment_method", "fold_assignment_reads_label",
    "block_split_across_folds", "n_blocks",
    "training_di_q1", "training_di_q3", "training_di_iqr",
    "training_di_q50_threshold", "training_di_q90_threshold",
    "training_di_q95_threshold", "training_di_q99_threshold",
    "training_di_max_threshold", "upper_whisker_unclamped",
    "upper_whisker_clamped_to_max", "training_di_upper_whisker_threshold",
    "primary_threshold_method", "training_di_q95_method", "q95_is_operative",
)

SOURCE_WEIGHT_COLUMNS: tuple[str, ...] = (
    "source_experiment", "feature", "feature_kind", "raw_importance",
    "n_dummy_levels_summed", "dummy_level_contributions", "weight",
    "renormalisation_factor", "is_zero_weight", "source_mean", "source_scale",
    "source_scale_method", "constant_feature_guard_used", "importance_method",
    "importance_population", "importance_model",
)


def source_weight_rows(prepared: dict[str, Any], derived: dict[str, Any]) -> list[dict[str, Any]]:
    scaling = prepared["scaling"]
    categorical = categorical_feature()
    rows: list[dict[str, Any]] = []
    for feature in list(numeric_features()) + [categorical]:
        is_categorical = feature == categorical
        entry = scaling.get(feature, {})
        rows.append({
            "source_experiment": derived["experiment_id"],
            "feature": feature,
            "feature_kind": "categorical" if is_categorical else "numeric",
            "raw_importance": derived["raw_importance"][feature],
            "n_dummy_levels_summed": (
                len(derived["dummy_level_contributions"]) if is_categorical else 0
            ),
            "dummy_level_contributions": (
                canonical_json(derived["dummy_level_contributions"]) if is_categorical else ""
            ),
            "weight": derived["weights"][feature],
            "renormalisation_factor": derived["renormalisation_factor"],
            "is_zero_weight": derived["weights"][feature] == 0.0,
            "source_mean": entry.get("mean"),
            "source_scale": entry.get("scale"),
            "source_scale_method": entry.get("source_scale_method"),
            "constant_feature_guard_used": entry.get("constant_feature_guard_used", False),
            "importance_method": IMPORTANCE_METHOD,
            "importance_population": IMPORTANCE_POPULATION,
            "importance_model": IMPORTANCE_MODEL,
        })
    return rows


# =============================================================================
# Directed pair analysis
# =============================================================================
TARGET_CELL_COLUMNS: tuple[str, ...] = (
    "source_experiment", "target_experiment", "row_500m", "col_500m",
    "weighted_dissimilarity", "nearest_source_row_500m", "nearest_source_col_500m",
    "categorical_mismatch_at_nearest", "n_missing_predictors",
    "cell_weighted_aoa_status", "inside_q95_sensitivity",
)


def analyse_directed_pair(
    prepared: dict[str, Any], target_id: str, target_population: pd.DataFrame,
    *, neighbour_chunk_size: int = NEIGHBOUR_CHUNK_SIZE,
) -> dict[str, Any]:
    """One directed source->target pair. The source side is already frozen."""
    source_id = prepared["experiment_id"]
    if source_id == target_id:
        raise MarginalAoACompletionError(
            f"Self pair '{source_id}' -> '{target_id}' is not a valid direction."
        )

    target_total = int(len(target_population))
    complete = complete_predictor_mask(target_population)
    assessable = target_population.loc[complete].reset_index(drop=True)
    not_assessable = target_population.loc[~complete].reset_index(drop=True)

    n_missing = np.zeros(target_total, dtype="int64")
    for feature in numeric_features():
        values = pd.to_numeric(target_population[feature], errors="coerce").to_numpy(dtype="float64")
        n_missing += (~np.isfinite(values)).astype("int64")
    target_levels_all = canonical_levels(target_population[categorical_feature()])
    n_missing += np.array([lvl is None for lvl in target_levels_all], dtype="int64")

    threshold = prepared["threshold"]["training_di_upper_whisker_threshold"]
    q95_threshold = prepared["threshold"]["training_di_q95_threshold"]
    normaliser = prepared["normaliser"]["source_pairwise_mean_distance"]
    w_landcover = prepared["w_landcover"]

    rows: list[dict[str, Any]] = []
    di_values = np.array([], dtype="float64")
    contributions = {f: 0.0 for f in list(numeric_features()) + [categorical_feature()]}
    unseen_level_count = 0

    if len(assessable) > 0:
        coords = weighted_coordinates(
            standardise(assessable, prepared["scaling"]), _weights_of(prepared)
        )
        levels = canonical_levels(assessable[categorical_feature()])
        codes = encode_levels(levels, prepared["level_vocabulary"])
        unseen_level_count = int(sum(1 for lvl in levels if lvl not in prepared["level_vocabulary"]))

        distances, nearest_idx = nearest_distances(
            coords, codes, prepared["coords"], prepared["codes"], w_landcover,
            chunk_size=neighbour_chunk_size, return_index=True,
        )
        di_values = distances / normaliser

        reference = prepared["reference_frame"]
        source_coords = prepared["coords"]
        source_codes = prepared["codes"]
        numeric = list(numeric_features())
        diff = coords - source_coords[nearest_idx]
        squared = diff ** 2
        for j, feature in enumerate(numeric):
            contributions[feature] = float(squared[:, j].mean())
        mismatch = (codes != source_codes[nearest_idx])
        contributions[categorical_feature()] = float(w_landcover * mismatch.mean())

        inside = di_values <= threshold
        inside_q95 = di_values <= q95_threshold
        for i in range(len(assessable)):
            rows.append({
                "source_experiment": source_id,
                "target_experiment": target_id,
                "row_500m": int(assessable.at[i, "row_500m"]),
                "col_500m": int(assessable.at[i, "col_500m"]),
                "weighted_dissimilarity": float(di_values[i]),
                "nearest_source_row_500m": int(reference.at[int(nearest_idx[i]), "row_500m"]),
                "nearest_source_col_500m": int(reference.at[int(nearest_idx[i]), "col_500m"]),
                "categorical_mismatch_at_nearest": bool(mismatch[i]),
                "n_missing_predictors": 0,
                "cell_weighted_aoa_status": (
                    "inside_weighted_aoa" if bool(inside[i]) else "outside_weighted_aoa"
                ),
                "inside_q95_sensitivity": bool(inside_q95[i]),
            })

    for i in range(len(not_assessable)):
        rows.append({
            "source_experiment": source_id,
            "target_experiment": target_id,
            "row_500m": int(not_assessable.at[i, "row_500m"]),
            "col_500m": int(not_assessable.at[i, "col_500m"]),
            "weighted_dissimilarity": None,
            "nearest_source_row_500m": None,
            "nearest_source_col_500m": None,
            "categorical_mismatch_at_nearest": None,
            "n_missing_predictors": None,
            "cell_weighted_aoa_status": "not_assessable",
            "inside_q95_sensitivity": None,
        })

    missing_lookup = {
        (int(r), int(c)): int(m)
        for r, c, m in zip(
            target_population["row_500m"], target_population["col_500m"], n_missing
        )
    }
    for row in rows:
        row["n_missing_predictors"] = missing_lookup[(row["row_500m"], row["col_500m"])]

    rows.sort(key=lambda r: (r["row_500m"], r["col_500m"]))

    n_inside = int(sum(1 for r in rows if r["cell_weighted_aoa_status"] == "inside_weighted_aoa"))
    n_outside = int(sum(1 for r in rows if r["cell_weighted_aoa_status"] == "outside_weighted_aoa"))
    n_not = int(sum(1 for r in rows if r["cell_weighted_aoa_status"] == "not_assessable"))
    if n_inside + n_outside + n_not != target_total:
        raise MarginalAoACompletionError(
            f"{source_id}->{target_id}: cell status counts "
            f"({n_inside}+{n_outside}+{n_not}) do not sum to the target "
            f"population ({target_total})."
        )

    total_contribution = float(sum(contributions.values()))
    top_features = sorted(
        (
            {
                "feature": f,
                "contribution": contributions[f],
                "share": (contributions[f] / total_contribution) if total_contribution > 0 else 0.0,
            }
            for f in contributions
        ),
        key=lambda e: (-e["contribution"], e["feature"]),
    )

    def _q(values: np.ndarray, q: float) -> Optional[float]:
        return float(np.quantile(values, q, method=QUANTILE_METHOD)) if values.size else None

    return {
        "source_experiment": source_id,
        "target_experiment": target_id,
        "direction": direction_token(source_id, target_id),
        "pair_token": pair_token(source_id, target_id),
        "rows": rows,
        "target_rows": target_total,
        "target_rows_assessable": int(len(assessable)),
        "target_rows_not_assessable": int(len(not_assessable)),
        "target_mean_dissimilarity": float(di_values.mean()) if di_values.size else None,
        "target_median_dissimilarity": _q(di_values, 0.50),
        "target_p90_dissimilarity": _q(di_values, 0.90),
        "target_p95_dissimilarity": _q(di_values, 0.95),
        "target_max_dissimilarity": float(di_values.max()) if di_values.size else None,
        "fraction_inside_weighted_aoa": n_inside / target_total,
        "fraction_outside_weighted_aoa": n_outside / target_total,
        "fraction_not_assessable": n_not / target_total,
        "target_cells_with_unseen_level": unseen_level_count,
        "fraction_target_cells_with_unseen_level": unseen_level_count / target_total,
        "top_weighted_mismatch_features": top_features,
    }


def _weights_of(prepared: dict[str, Any]) -> dict[str, float]:
    return prepared["_weights"]


# =============================================================================
# Climate
# =============================================================================
def climate_contract() -> dict[str, Any]:
    return {
        "collection": CLIMATE_COLLECTION,
        "reference_period": f"{CLIMATE_PERIOD_START}/{CLIMATE_PERIOD_END}",
        "period_start": CLIMATE_PERIOD_START,
        "period_end": CLIMATE_PERIOD_END,
        "expected_month_count": CLIMATE_EXPECTED_MONTHS,
        "years": list(CLIMATE_YEARS),
        "season_months": list(CLIMATE_SEASON_MONTHS),
        "season_months_provenance": (
            "core/config.py SUMMER_MONTH_START=6, SUMMER_MONTH_END=9 -- reused"
        ),
        "source_bands": list(CLIMATE_SOURCE_BANDS),
        "band_scale_factors": dict(CLIMATE_BAND_SCALE_FACTORS),
        "climate_features": list(CLIMATE_FEATURES),
        "climate_feature_count": CLIMATE_FEATURE_COUNT,
        "land_mask": CLIMATE_LAND_MASK,
        "reference_window": dict(CLIMATE_REFERENCE_WINDOW),
        "distance_metric": CLIMATE_DISTANCE_METRIC,
        "scaling_contract": (
            "z-score against valid TerraClimate land pixels of "
            "lon[-10,42] lat[30,47] on the same 1991-2020 climatology, "
            "population SD ddof=0"
        ),
        "export_authorised": True,
        "era5_land_cross_check_in_initial_run": False,
    }


def climate_variable_recipes() -> list[dict[str, Any]]:
    """Exact per-variable aggregation, applied AFTER band scaling."""
    return [
        {
            "field": CLIMATE_FEATURES[0],
            "bands": ["tmmx", "tmmn"],
            "expression": "(scaled_tmmx + scaled_tmmn) / 2",
            "aggregation": "mean over all 360 months of 1991-2020",
            "units": "degC",
        },
        {
            "field": CLIMATE_FEATURES[1],
            "bands": ["pr"],
            "expression": "scaled_pr",
            "aggregation": "sum within each year, then mean of the 30 annual sums",
            "units": "mm",
        },
        {
            "field": CLIMATE_FEATURES[2],
            "bands": ["def"],
            "expression": "scaled_def",
            "aggregation": (
                "sum over June-September within each year, then mean of the "
                "30 yearly warm-season sums"
            ),
            "units": "mm",
        },
        {
            "field": CLIMATE_FEATURES[3],
            "bands": ["vpd"],
            "expression": "scaled_vpd",
            "aggregation": "mean over all June-September months of 1991-2020",
            "units": "kPa",
        },
    ]


def build_climate_input_inventory(
    analysis_id: str, output_root: Optional[Path] = None
) -> dict[str, Any]:
    raster = climate_raster_path(analysis_id, output_root)
    exists = raster.is_file()
    return {
        "contract": climate_contract(),
        "variable_recipes": climate_variable_recipes(),
        "raster_path": str(raster),
        "raster_exists": exists,
        "raster_sha256": sha256_file(raster) if exists else None,
        "climate_status": "available" if exists else "authorised_pending_export",
    }


def assert_climate_month_count(observed: int) -> None:
    if int(observed) != CLIMATE_EXPECTED_MONTHS:
        raise MarginalAoACompletionError(
            f"TerraClimate 1991-2020 must contribute exactly "
            f"{CLIMATE_EXPECTED_MONTHS} monthly observations; the collection "
            f"returned {observed}. Refusing to aggregate an incomplete record."
        )


def read_climate_raster(path: Path) -> dict[str, Any]:
    """Read the frozen four-band climate raster and its georeferencing."""
    if not Path(path).is_file():
        raise MarginalAoACompletionError(
            f"Climate normals raster not found: {path}. Run the "
            "'climate-export' stage first; no proxy is substituted."
        )
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - rasterio is a locked dep
        raise MarginalAoACompletionError(
            "rasterio is required to read the climate normals raster."
        ) from exc

    with rasterio.open(path) as handle:
        if handle.count != CLIMATE_FEATURE_COUNT:
            raise MarginalAoACompletionError(
                f"Climate raster {path} has {handle.count} band(s); the frozen "
                f"contract requires exactly {CLIMATE_FEATURE_COUNT} "
                f"({list(CLIMATE_FEATURES)})."
            )
        data = handle.read(masked=True).astype("float64")
        transform = handle.transform
        crs = str(handle.crs) if handle.crs is not None else None
        height, width = int(handle.height), int(handle.width)
        descriptions = list(handle.descriptions or ())

    values = np.ma.filled(data, np.nan)
    finite = np.isfinite(values).all(axis=0)
    return {
        "values": values,
        "support": finite,
        "transform": transform,
        "crs": crs,
        "height": height,
        "width": width,
        "descriptions": descriptions,
    }


def _pixel_centres(raster: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    transform = raster["transform"]
    rows = np.arange(raster["height"]) + 0.5
    cols = np.arange(raster["width"]) + 0.5
    col_grid, row_grid = np.meshgrid(cols, rows)
    lon = transform.c + transform.a * col_grid + transform.b * row_grid
    lat = transform.f + transform.d * col_grid + transform.e * row_grid
    return lon, lat


def climate_vectors(
    raster: dict[str, Any], experiment_ids: Sequence[str]
) -> dict[str, Any]:
    """Per-AOI climate vector plus the Mediterranean reference statistics.

    Both use the SAME native TerraClimate valid-land support -- one mask
    definition, applied identically, so an AOI summary and the reference
    scaling can never disagree about which pixels are land.
    """
    lon, lat = _pixel_centres(raster)
    support = raster["support"]
    values = raster["values"]

    window = CLIMATE_REFERENCE_WINDOW
    in_window = (
        (lon >= window["lon_min"]) & (lon <= window["lon_max"])
        & (lat >= window["lat_min"]) & (lat <= window["lat_max"])
    )
    reference_mask = support & in_window
    if not reference_mask.any():
        raise MarginalAoACompletionError(
            "The Mediterranean reference window contains no valid TerraClimate "
            "land pixel; the climate scaling cannot be derived."
        )

    reference_stats: dict[str, dict[str, float]] = {}
    for k, feature in enumerate(CLIMATE_FEATURES):
        band = values[k][reference_mask]
        reference_stats[feature] = {
            "ref_mean": float(band.mean()),
            "ref_sd": float(band.std(ddof=0)),
            "n_pixels": int(band.size),
        }
        if reference_stats[feature]["ref_sd"] <= 0.0:
            raise MarginalAoACompletionError(
                f"Reference SD for '{feature}' is {reference_stats[feature]['ref_sd']}; "
                "cannot standardise."
            )

    aoi: dict[str, Any] = {}
    for experiment_id in sorted(experiment_ids):
        lon_min, lat_min, lon_max, lat_max = CANONICAL_AOI_BBOX[experiment_id]
        inside = (
            (lon >= lon_min) & (lon <= lon_max)
            & (lat >= lat_min) & (lat <= lat_max)
        )
        mask = support & inside
        n_pixels = int(mask.sum())
        if n_pixels == 0:
            raise MarginalAoACompletionError(
                f"'{experiment_id}': no valid TerraClimate land pixel centre lies "
                "inside the canonical AOI bbox."
            )
        raw = {f: float(values[k][mask].mean()) for k, f in enumerate(CLIMATE_FEATURES)}
        standardised = {
            f: (raw[f] - reference_stats[f]["ref_mean"]) / reference_stats[f]["ref_sd"]
            for f in CLIMATE_FEATURES
        }
        aoi[experiment_id] = {
            "experiment_id": experiment_id,
            "n_valid_pixels": n_pixels,
            "n_pixels_in_bbox": int(inside.sum()),
            "climate_data_completeness": n_pixels / max(int(inside.sum()), 1),
            "raw": raw,
            "standardised": standardised,
        }
    return {"aoi": aoi, "reference_stats": reference_stats}


def pairwise_climate_distances(vectors: dict[str, Any]) -> list[dict[str, Any]]:
    aoi = vectors["aoi"]
    rows: list[dict[str, Any]] = []
    for a, b in unordered_pairs(sorted(aoi)):
        contributions = {
            f: ((aoi[a]["standardised"][f] - aoi[b]["standardised"][f]) ** 2) / CLIMATE_FEATURE_COUNT
            for f in CLIMATE_FEATURES
        }
        squared = float(sum(contributions.values()))
        distance = math.sqrt(squared)
        if abs(sum(contributions.values()) - squared) > 1e-12:
            raise MarginalAoACompletionError(
                "Climate component contributions do not sum to the squared distance."
            )
        rows.append({
            "experiment_a": a,
            "experiment_b": b,
            "climate_distance": distance,
            "climate_distance_squared": squared,
            "climate_distance_metric": CLIMATE_DISTANCE_METRIC,
            "climate_feature_count": CLIMATE_FEATURE_COUNT,
            "climate_features": canonical_json(list(CLIMATE_FEATURES)),
            "climate_reference_period": f"{CLIMATE_PERIOD_START}/{CLIMATE_PERIOD_END}",
            "climate_season_months": canonical_json(list(CLIMATE_SEASON_MONTHS)),
            "climate_land_mask": CLIMATE_LAND_MASK,
            "climate_source_version": CLIMATE_COLLECTION,
            "climate_component_contributions": canonical_json(contributions),
            "climate_uncertainty": "deterministic_aoi_level_value_no_interval",
        })
    return rows


# =============================================================================
# Geographic
# =============================================================================
def bbox_centre(bbox: Sequence[float]) -> tuple[float, float]:
    lon_min, lat_min, lon_max, lat_max = bbox
    return ((lon_min + lon_max) / 2.0, (lat_min + lat_max) / 2.0)


def geometry_contract_hash(experiment_id: str) -> str:
    bbox = CANONICAL_AOI_BBOX[experiment_id]
    payload = {
        "experiment_id": experiment_id,
        "crs": "EPSG:4326",
        "kind": "bbox",
        "coordinates": [float(v) for v in bbox],
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def assert_geometry_matches_registry() -> dict[str, Any]:
    """Prove the hard-pinned bboxes have not drifted from core/regions.py.

    The registry defines two AOIs as named module constants and two as inline
    bbox literals; only the constants can be read without an Earth Engine
    session, so the literals are checked against the registry source text.
    """
    from core import regions as registry

    checked: dict[str, Any] = {}
    named = {
        "mugla_2021": "MUGLA_AOI_BBOX",
        "evia_2021_extended": "NORTH_EVIA_EXTENDED_AOI_BBOX",
    }
    for experiment_id, constant in named.items():
        expected = tuple(float(v) for v in getattr(registry, constant))
        pinned = tuple(float(v) for v in CANONICAL_AOI_BBOX[experiment_id])
        if expected != pinned:
            raise MarginalAoACompletionError(
                f"'{experiment_id}': pinned bbox {pinned} has drifted from "
                f"core/regions.py {constant} = {expected}."
            )
        checked[experiment_id] = {"source": f"core/regions.py:{constant}", "bbox": list(pinned)}

    # These two AOIs are inline bbox literals inside build_regions() rather
    # than named constants, so the drift check matches the four coordinates in
    # order. The Earth Engine constructor name is deliberately NOT spelled out
    # here: this module must stay free of any Earth Engine token.
    source_text = Path(registry.__file__).read_text(encoding="utf-8").replace(" ", "")
    for experiment_id in ("manavgat_2021", "bejis_2022"):
        pinned = CANONICAL_AOI_BBOX[experiment_id]
        literal = ",".join(repr(float(v)) for v in pinned)
        if literal not in source_text:
            raise MarginalAoACompletionError(
                f"'{experiment_id}': pinned bbox {tuple(pinned)} was not found as "
                f"an inline coordinate literal in core/regions.py; the geometry "
                "contract may have drifted."
            )
        checked[experiment_id] = {
            "source": "core/regions.py inline bbox literal inside build_regions()",
            "bbox": [float(v) for v in pinned],
        }
    return checked


GEOGRAPHICLIB_INSTALL_MESSAGE = (
    "The geographic-distance component requires the 'geographiclib' package, "
    "which is not importable.\n"
    "Install it with:\n"
    "    pip install geographiclib\n"
    "No haversine, pyproj or custom Vincenty fallback is permitted: a spherical "
    "approximation would introduce kilometre-scale error and is not the "
    "preregistered method. Nothing was written."
)


def resolve_geodesic_inverse(geodesic_inverse: Any = None):
    """Return the WGS84 geodesic-inverse callable, or fail closed.

    The default binds GeographicLib's `Geodesic.WGS84.Inverse` lazily, so
    importing this module never requires the package. `geodesic_inverse` is the
    injection point: it takes (lat1, lon1, lat2, lon2) and returns a mapping
    with an "s12" distance in metres -- the GeographicLib contract -- so a test
    can exercise the distance binding and the surrounding arithmetic without
    the package installed. There is no fallback implementation anywhere.
    """
    if geodesic_inverse is not None:
        return geodesic_inverse
    try:
        from geographiclib.geodesic import Geodesic
    except ImportError as exc:
        raise MarginalAoACompletionError(GEOGRAPHICLIB_INSTALL_MESSAGE) from exc
    return Geodesic.WGS84.Inverse


def geodesic_distance_km(
    lon1: float, lat1: float, lon2: float, lat2: float,
    *, geodesic_inverse: Any = None,
) -> float:
    """WGS84 geodesic inverse distance in kilometres, via GeographicLib."""
    inverse = resolve_geodesic_inverse(geodesic_inverse)
    return float(inverse(lat1, lon1, lat2, lon2)["s12"]) / 1000.0


def minimum_boundary_distance_km(
    bbox_a: Sequence[float], bbox_b: Sequence[float],
    *, geodesic_inverse: Any = None,
) -> float:
    """Minimum geodesic distance between two axis-aligned lon/lat rectangles."""
    a_lon_min, a_lat_min, a_lon_max, a_lat_max = bbox_a
    b_lon_min, b_lat_min, b_lon_max, b_lat_max = bbox_b

    lon_overlap = not (a_lon_max < b_lon_min or b_lon_max < a_lon_min)
    lat_overlap = not (a_lat_max < b_lat_min or b_lat_max < a_lat_min)
    if lon_overlap and lat_overlap:
        return 0.0

    candidates: list[float] = []
    lons_a = [a_lon_min, a_lon_max]
    lons_b = [b_lon_min, b_lon_max]
    lats_a = [a_lat_min, a_lat_max]
    lats_b = [b_lat_min, b_lat_max]

    if lon_overlap:
        shared_lon = max(a_lon_min, b_lon_min)
        for la in lats_a:
            for lb in lats_b:
                candidates.append(geodesic_distance_km(
                    shared_lon, la, shared_lon, lb,
                    geodesic_inverse=geodesic_inverse))
    if lat_overlap:
        shared_lat = max(a_lat_min, b_lat_min)
        for oa in lons_a:
            for ob in lons_b:
                candidates.append(geodesic_distance_km(
                    oa, shared_lat, ob, shared_lat,
                    geodesic_inverse=geodesic_inverse))
    for oa in lons_a:
        for la in lats_a:
            for ob in lons_b:
                for lb in lats_b:
                    candidates.append(geodesic_distance_km(
                        oa, la, ob, lb, geodesic_inverse=geodesic_inverse))
    return float(min(candidates))


def build_geometry_inventory(experiment_ids: Sequence[str]) -> dict[str, Any]:
    from core import regions as registry

    entries: dict[str, Any] = {}
    for experiment_id in sorted(experiment_ids):
        bbox = CANONICAL_AOI_BBOX[experiment_id]
        lon, lat = bbox_centre(bbox)
        entries[experiment_id] = {
            "experiment_id": experiment_id,
            "bbox": [float(v) for v in bbox],
            "crs": "EPSG:4326",
            "geometry_kind": "axis_aligned_rectangle",
            "centroid_lon": lon,
            "centroid_lat": lat,
            "centroid_definition": CENTROID_DEFINITION,
            "geometry_contract_hash": geometry_contract_hash(experiment_id),
        }
    return {
        "geometry_source_path": "core/regions.py",
        "geometry_source_sha256": sha256_file(Path(registry.__file__)),
        "centroid_definition": CENTROID_DEFINITION,
        "geodesic_implementation": GEODESIC_IMPLEMENTATION,
        "geographic_distance_method": GEOGRAPHIC_DISTANCE_METHOD,
        "geographic_component_reads_step8a": False,
        "population_centroid_reported": False,
        "aois": entries,
    }


def pairwise_geographic_distances(
    experiment_ids: Sequence[str], *, geodesic_inverse: Any = None,
) -> list[dict[str, Any]]:
    # Resolve once, up front: a missing dependency fails BEFORE any row is
    # built and therefore before anything could be written.
    inverse = resolve_geodesic_inverse(geodesic_inverse)
    rows: list[dict[str, Any]] = []
    for a, b in unordered_pairs(sorted(experiment_ids)):
        bbox_a, bbox_b = CANONICAL_AOI_BBOX[a], CANONICAL_AOI_BBOX[b]
        lon_a, lat_a = bbox_centre(bbox_a)
        lon_b, lat_b = bbox_centre(bbox_b)
        rows.append({
            "experiment_a": a,
            "experiment_b": b,
            "centroid_a_lon": lon_a,
            "centroid_a_lat": lat_a,
            "centroid_b_lon": lon_b,
            "centroid_b_lat": lat_b,
            "centroid_geodesic_distance_km": geodesic_distance_km(
                lon_a, lat_a, lon_b, lat_b, geodesic_inverse=inverse),
            "optional_minimum_boundary_distance_km": minimum_boundary_distance_km(
                bbox_a, bbox_b, geodesic_inverse=inverse),
            "geographic_distance_method": GEOGRAPHIC_DISTANCE_METHOD,
            "centroid_definition": CENTROID_DEFINITION,
            "geodesic_implementation": GEODESIC_IMPLEMENTATION,
            "bbox_a": canonical_json([float(v) for v in bbox_a]),
            "bbox_b": canonical_json([float(v) for v in bbox_b]),
            "geometry_contract_hash_a": geometry_contract_hash(a),
            "geometry_contract_hash_b": geometry_contract_hash(b),
            "geographic_distance_uncertainty": "deterministic_aoi_level_value_no_interval",
        })
    return rows


def symmetric_lookup(rows: Sequence[dict[str, Any]], key: str) -> dict[tuple[str, str], Any]:
    """Both orderings map to the SAME stored value -- an exact copy."""
    lookup: dict[tuple[str, str], Any] = {}
    for row in rows:
        a, b = row["experiment_a"], row["experiment_b"]
        lookup[(a, b)] = row[key]
        lookup[(b, a)] = row[key]
    return lookup


# =============================================================================
# Scientific configuration / analysis identity
# =============================================================================
def scientific_configuration(
    experiment_ids: Sequence[str],
    frozen_inventory: dict[str, Any],
    importance_inventory: dict[str, Any],
    transfer_inventory: dict[str, Any],
) -> dict[str, Any]:
    """Everything the analysis identity binds. INPUTS AND CONTRACT ONLY.

    No output value, no computed statistic and no result ever enters this
    payload, so the analysis_id cannot drift with the numbers it labels.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_namespace": DIAGNOSTIC_NAMESPACE,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "experiments": sorted(experiment_ids),
        "primary_population": PRIMARY_POPULATION,
        "feature_contract": feature_contract_payload(),
        "step8a_inputs": {
            k: {"path": v["path"], "sha256": v["sha256"]}
            for k, v in sorted(frozen_inventory.items())
        },
        "feature_importance_inputs": {
            k: {
                "path": v["path"], "sha256": v["sha256"],
                "population_filter": v["population_filter"],
                "model_filter": v["model_filter"],
                "importance_method": v["importance_method"],
            }
            for k, v in sorted(importance_inventory.items())
        },
        "geometry_contract": {
            experiment_id: {
                "bbox": [float(v) for v in CANONICAL_AOI_BBOX[experiment_id]],
                "crs": "EPSG:4326",
                "centroid_definition": CENTROID_DEFINITION,
                "geometry_contract_hash": geometry_contract_hash(experiment_id),
            }
            for experiment_id in sorted(experiment_ids)
        },
        "climate_contract": climate_contract(),
        "transfer_comparison_input": {
            "path": transfer_inventory["path"],
            "sha256": transfer_inventory["sha256"],
            "primary_selection": dict(PRIMARY_TRANSFER_SELECTION),
            "primary_comparison": PRIMARY_TRANSFER_COMPARISON,
        },
        "weighted_distance": {
            "formula_id": WEIGHTED_DISTANCE_FORMULA_ID,
            "categorical_policy_id": CATEGORICAL_POLICY_ID,
            "nearest_neighbour_method": NEAREST_NEIGHBOUR_METHOD,
            "numeric_scaling_method": "source_mean_and_population_sd_ddof0",
            "imputation": "none",
        },
        "normaliser": {
            "method": NORMALISER_METHOD,
            "uses_folds": False,
            "self_distance_excluded": True,
            "categorical_term_included": True,
        },
        "threshold": {
            "primary_method": PRIMARY_THRESHOLD_METHOD,
            "secondary_q95_method": SECONDARY_Q95_METHOD,
            "q95_is_operative": False,
            "block_size_cells": FOLD_BLOCK_SIZE_CELLS,
            "fold_count": FOLD_COUNT,
            "fold_assignment_method": FOLD_ASSIGNMENT_METHOD,
        },
        "uncertainty_policy": "point_estimate_only",
        "composite_index_produced": False,
        "target_label_firewall": dict(TARGET_LABEL_FIREWALL),
        "source_label_policy": dict(SOURCE_LABEL_POLICY),
    }


def compute_analysis_id(config: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(config).encode("utf-8"))


def build_transfer_inventory(output_root: Optional[Path] = None) -> dict[str, Any]:
    path = transfer_decomposition_path(output_root)
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "sha256": sha256_file(path) if exists else None,
        "primary_selection": dict(PRIMARY_TRANSFER_SELECTION),
        "primary_comparison": PRIMARY_TRANSFER_COMPARISON,
        "secondary_comparisons": list(SECONDARY_TRANSFER_COMPARISONS),
        "read_only": True,
        "read_by": "comparison layer only, after every diagnostic is frozen",
    }


# =============================================================================
# Deterministic document serialisation + atomic writes
# =============================================================================
def _json_document(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n"


def _csv_document(columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(columns), lineterminator="\n", extrasaction="ignore",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column) for column in columns})
    return buffer.getvalue()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def assert_inside_namespace(path: Path, root: Path) -> None:
    resolved, root_resolved = Path(path).resolve(), Path(root).resolve()
    if root_resolved not in resolved.parents and resolved != root_resolved:
        raise MarginalAoACompletionError(
            f"Refusing to write outside the analysis namespace: {resolved} is not "
            f"under {root_resolved}."
        )


def write_documents(root: Path, documents: dict[str, str]) -> list[str]:
    written: list[str] = []
    for relative in sorted(documents):
        target = Path(root) / relative
        assert_inside_namespace(target, root)
        _atomic_write_text(target, documents[relative])
        written.append(relative)
    return written


# =============================================================================
# Directed-pair summary table
# =============================================================================
DIRECTED_PAIR_COLUMNS: tuple[str, ...] = (
    "schema_version", "analysis_id", "source_experiment", "target_experiment",
    "direction", "pair_token", "primary_population",
    "source_step8a_sha256", "target_step8a_sha256", "source_importance_sha256",
    "source_rows", "target_rows", "source_rows_reference",
    "source_rows_excluded_missing", "target_rows_assessable",
    "target_rows_not_assessable",
    "importance_method", "importance_population", "importance_model",
    "n_features_with_positive_weight", "effective_feature_count",
    "feature_weight_entropy", "zero_weight_features",
    "constant_feature_guard_features", "numeric_scaling_method",
    "weighted_distance_formula_id", "categorical_policy_id",
    "source_pairwise_mean_distance", "source_distance_normaliser",
    "normaliser_method", "normaliser_uses_folds", "n_distinct_source_pairs",
    "training_di_upper_whisker_threshold", "primary_threshold_method",
    "training_di_q95_threshold", "training_di_q95_method", "q95_is_operative",
    "training_di_q1", "training_di_q3", "training_di_iqr",
    "training_di_q50_threshold", "training_di_q90_threshold",
    "training_di_q99_threshold", "training_di_max_threshold",
    "upper_whisker_unclamped", "upper_whisker_clamped_to_max",
    "target_mean_dissimilarity", "target_median_dissimilarity",
    "target_p90_dissimilarity", "target_p95_dissimilarity",
    "target_max_dissimilarity",
    "fraction_inside_weighted_aoa", "fraction_outside_weighted_aoa",
    "fraction_not_assessable",
    "target_cells_with_unseen_level", "fraction_target_cells_with_unseen_level",
    "top_weighted_mismatch_features",
    "climate_distance", "climate_distance_metric", "climate_feature_count",
    "climate_features", "climate_reference_period", "climate_land_mask",
    "climate_source_version", "climate_status", "climate_export_authorised",
    "climate_uncertainty",
    "source_centroid_lon", "source_centroid_lat", "target_centroid_lon",
    "target_centroid_lat", "centroid_geodesic_distance_km",
    "optional_minimum_boundary_distance_km", "geographic_distance_method",
    "centroid_definition", "geodesic_implementation",
    "geographic_component_reads_step8a", "population_centroid_reported",
    "geographic_distance_uncertainty",
    "unweighted_analysis_id", "unweighted_support_definition_id",
    "unweighted_fraction_target_cells_inside_support",
    "unweighted_fraction_target_cells_outside_support",
    "unweighted_fraction_target_cells_not_assessable",
    "unweighted_sidecar_path",
    "target_label_used", "target_burn_date_used", "target_transfer_metric_used",
    "source_label_used", "source_label_read_directly_by_completion_module",
    "diagnostic_class", "model_fit", "bootstrap_run", "uncertainty_policy",
)


def unweighted_sidecar(
    source_id: str, target_id: str, output_root: Optional[Path] = None
) -> dict[str, Any]:
    """Read-only echo of the frozen marginal_aoa.v1 result for this pair.

    The authoritative values stay in marginal_aoa.v1; these are copies for
    joining. Absent inputs degrade to nulls rather than failing, because the
    completion analysis does not depend on the older artifact.
    """
    root = marginal_aoa_v1_root(output_root)
    summary = root / "pairs" / pair_token(source_id, target_id) / "marginal_aoa_summary.json"
    record = {
        "unweighted_analysis_id": MARGINAL_AOA_V1_ANALYSIS_ID,
        "unweighted_support_definition_id": None,
        "unweighted_fraction_target_cells_inside_support": None,
        "unweighted_fraction_target_cells_outside_support": None,
        "unweighted_fraction_target_cells_not_assessable": None,
        "unweighted_sidecar_path": str(summary),
    }
    if not summary.is_file():
        return record
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return record
    record.update({
        "unweighted_support_definition_id": payload.get("support_definition_id"),
        "unweighted_fraction_target_cells_inside_support": payload.get(
            "fraction_target_cells_inside_support"),
        "unweighted_fraction_target_cells_outside_support": payload.get(
            "fraction_target_cells_outside_support"),
        "unweighted_fraction_target_cells_not_assessable": payload.get(
            "fraction_target_cells_not_assessable"),
    })
    return record


def build_directed_pair_rows(
    analysis_id: str,
    pair_results: Sequence[dict[str, Any]],
    prepared_sources: dict[str, dict[str, Any]],
    derived_weights: dict[str, dict[str, Any]],
    frozen_inventory: dict[str, Any],
    importance_inventory: dict[str, Any],
    climate_rows: Optional[Sequence[dict[str, Any]]],
    geographic_rows: Optional[Sequence[dict[str, Any]]],
    climate_status: str,
    output_root: Optional[Path] = None,
) -> list[dict[str, Any]]:
    climate_lookup = symmetric_lookup(climate_rows or [], "climate_distance")
    climate_contrib = symmetric_lookup(climate_rows or [], "climate_component_contributions")
    geo_centroid = symmetric_lookup(geographic_rows or [], "centroid_geodesic_distance_km")
    geo_boundary = symmetric_lookup(
        geographic_rows or [], "optional_minimum_boundary_distance_km"
    )

    rows: list[dict[str, Any]] = []
    for result in pair_results:
        source_id = result["source_experiment"]
        target_id = result["target_experiment"]
        prepared = prepared_sources[source_id]
        derived = derived_weights[source_id]
        threshold = prepared["threshold"]
        normaliser = prepared["normaliser"]
        scaling = prepared["scaling"]

        source_lon, source_lat = bbox_centre(CANONICAL_AOI_BBOX[source_id])
        target_lon, target_lat = bbox_centre(CANONICAL_AOI_BBOX[target_id])

        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "analysis_id": analysis_id,
            "source_experiment": source_id,
            "target_experiment": target_id,
            "direction": result["direction"],
            "pair_token": result["pair_token"],
            "primary_population": PRIMARY_POPULATION,
            "source_step8a_sha256": frozen_inventory[source_id]["sha256"],
            "target_step8a_sha256": frozen_inventory[target_id]["sha256"],
            "source_importance_sha256": importance_inventory[source_id]["sha256"],
            "source_rows": prepared["source_rows_total"],
            "target_rows": result["target_rows"],
            "source_rows_reference": prepared["source_rows_reference"],
            "source_rows_excluded_missing": prepared["source_rows_excluded_missing"],
            "target_rows_assessable": result["target_rows_assessable"],
            "target_rows_not_assessable": result["target_rows_not_assessable"],
            "importance_method": IMPORTANCE_METHOD,
            "importance_population": IMPORTANCE_POPULATION,
            "importance_model": IMPORTANCE_MODEL,
            "n_features_with_positive_weight": derived["n_features_with_positive_weight"],
            "effective_feature_count": derived["effective_feature_count_perplexity"],
            "feature_weight_entropy": derived["feature_weight_entropy"],
            "zero_weight_features": canonical_json(derived["zero_weight_features"]),
            "constant_feature_guard_features": canonical_json(sorted(
                f for f, e in scaling.items() if e["constant_feature_guard_used"]
            )),
            "numeric_scaling_method": "source_mean_and_population_sd_ddof0",
            "weighted_distance_formula_id": WEIGHTED_DISTANCE_FORMULA_ID,
            "categorical_policy_id": CATEGORICAL_POLICY_ID,
            "source_pairwise_mean_distance": normaliser["source_pairwise_mean_distance"],
            "source_distance_normaliser": normaliser["source_pairwise_mean_distance"],
            "normaliser_method": normaliser["normaliser_method"],
            "normaliser_uses_folds": normaliser["normaliser_uses_folds"],
            "n_distinct_source_pairs": normaliser["n_distinct_source_pairs"],
            "top_weighted_mismatch_features": canonical_json(
                result["top_weighted_mismatch_features"]
            ),
            "climate_distance": climate_lookup.get((source_id, target_id)),
            "climate_distance_metric": CLIMATE_DISTANCE_METRIC,
            "climate_feature_count": CLIMATE_FEATURE_COUNT,
            "climate_features": canonical_json(list(CLIMATE_FEATURES)),
            "climate_reference_period": f"{CLIMATE_PERIOD_START}/{CLIMATE_PERIOD_END}",
            "climate_land_mask": CLIMATE_LAND_MASK,
            "climate_source_version": CLIMATE_COLLECTION,
            "climate_status": climate_status,
            "climate_export_authorised": True,
            "climate_uncertainty": "deterministic_aoi_level_value_no_interval",
            "climate_component_contributions": climate_contrib.get((source_id, target_id)),
            "source_centroid_lon": source_lon,
            "source_centroid_lat": source_lat,
            "target_centroid_lon": target_lon,
            "target_centroid_lat": target_lat,
            "centroid_geodesic_distance_km": geo_centroid.get((source_id, target_id)),
            "optional_minimum_boundary_distance_km": geo_boundary.get((source_id, target_id)),
            "geographic_distance_method": GEOGRAPHIC_DISTANCE_METHOD,
            "centroid_definition": CENTROID_DEFINITION,
            "geodesic_implementation": GEODESIC_IMPLEMENTATION,
            "geographic_component_reads_step8a": False,
            "population_centroid_reported": False,
            "geographic_distance_uncertainty": "deterministic_aoi_level_value_no_interval",
            "target_label_used": False,
            "target_burn_date_used": False,
            "target_transfer_metric_used": False,
            "source_label_used": True,
            "source_label_read_directly_by_completion_module": False,
            "diagnostic_class": DIAGNOSTIC_CLASS,
            "model_fit": False,
            "bootstrap_run": False,
            "uncertainty_policy": "point_estimate_only",
        }
        for key in (
            "target_mean_dissimilarity", "target_median_dissimilarity",
            "target_p90_dissimilarity", "target_p95_dissimilarity",
            "target_max_dissimilarity", "fraction_inside_weighted_aoa",
            "fraction_outside_weighted_aoa", "fraction_not_assessable",
            "target_cells_with_unseen_level", "fraction_target_cells_with_unseen_level",
        ):
            row[key] = result[key]
        row.update(threshold)
        row.update(unweighted_sidecar(source_id, target_id, output_root))
        rows.append(row)

    rows.sort(key=lambda r: (r["source_experiment"], r["target_experiment"]))
    return rows


# =============================================================================
# Transfer comparison layer -- read-only, runs LAST
# =============================================================================
def _spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    a, b = np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    return _pearson(_rank(a[ok]), _rank(b[ok]))


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype="float64")
    ranks[order] = np.arange(1, len(values) + 1, dtype="float64")
    # average ties so the coefficient is the standard tie-corrected one
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for idx in np.flatnonzero(counts > 1):
        tied = inverse == idx
        ranks[tied] = ranks[tied].mean()
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) < 3:
        return None
    a_c, b_c = a - a.mean(), b - b.mean()
    denominator = math.sqrt(float((a_c ** 2).sum()) * float((b_c ** 2).sum()))
    if denominator == 0.0:
        return None
    return float((a_c * b_c).sum() / denominator)


def _kendall(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    a, b = np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 3:
        return None
    concordant = discordant = tie_a = tie_b = 0
    for i in range(n - 1):
        da = a[i + 1:] - a[i]
        db = b[i + 1:] - b[i]
        product = da * db
        concordant += int((product > 0).sum())
        discordant += int((product < 0).sum())
        tie_a += int(((da == 0) & (db != 0)).sum())
        tie_b += int(((db == 0) & (da != 0)).sum())
    denominator = math.sqrt((concordant + discordant + tie_a) * (concordant + discordant + tie_b))
    if denominator == 0.0:
        return None
    return float((concordant - discordant) / denominator)


TRANSFER_QUANTITIES: tuple[tuple[str, str, str, str], ...] = (
    ("raw_thermal_roc_auc", "thermal", "regionwise_zscore", "roc_auc"),
    ("raw_thermal_pr_auc", "thermal", "regionwise_zscore", "pr_auc"),
    ("thermal_roc_auc_gap", "thermal", "regionwise_zscore", "roc_auc"),
    ("thermal_pr_auc_gap", "thermal", "regionwise_zscore", "pr_auc"),
    ("adapted_thermal_roc_auc_regionwise_zscore", "thermal", "regionwise_zscore", "roc_auc"),
    ("adapted_thermal_pr_auc_regionwise_zscore", "thermal", "regionwise_zscore", "pr_auc"),
    ("adapted_thermal_roc_auc_coral", "thermal", "coral_after_regionwise_zscore", "roc_auc"),
    ("adapted_thermal_pr_auc_coral", "thermal", "coral_after_regionwise_zscore", "pr_auc"),
    ("recovered_fraction", "thermal", "regionwise_zscore", "roc_auc"),
)

_TRANSFER_FIELD: dict[str, str] = {
    "raw_thermal_roc_auc": "raw_auc",
    "raw_thermal_pr_auc": "raw_auc",
    "thermal_roc_auc_gap": "raw_gap",
    "thermal_pr_auc_gap": "raw_gap",
    "adapted_thermal_roc_auc_regionwise_zscore": "adapted_auc",
    "adapted_thermal_pr_auc_regionwise_zscore": "adapted_auc",
    "adapted_thermal_roc_auc_coral": "adapted_auc",
    "adapted_thermal_pr_auc_coral": "adapted_auc",
    "recovered_fraction": "recovered_fraction",
}

DIAGNOSTIC_QUANTITIES: tuple[tuple[str, bool], ...] = (
    ("target_mean_dissimilarity", True),
    ("target_p95_dissimilarity", True),
    ("fraction_inside_weighted_aoa", True),
    ("climate_distance", False),
    ("centroid_geodesic_distance_km", False),
    ("unweighted_fraction_target_cells_inside_support", True),
)


def load_transfer_table(path: Path) -> pd.DataFrame:
    if not Path(path).is_file():
        raise MarginalAoACompletionError(
            f"Transfer decomposition artifact not found: {path}. The comparison "
            "layer is read-only against this frozen input."
        )
    return pd.read_csv(path)


def build_comparison_rows(
    directed_rows: Sequence[dict[str, Any]], transfer: pd.DataFrame
) -> list[dict[str, Any]]:
    """Join the 12 directed rows to the frozen transfer decomposition.

    The PRIMARY ordering is raw thermal ROC-AUC. An adapted metric would
    measure performance after an alignment step that itself removes part of
    the distribution shift the AoA quantifies, so ranking a shift diagnostic
    against it would partially cancel the effect under study.
    """
    rows: list[dict[str, Any]] = []
    for base in directed_rows:
        source_id, target_id = base["source_experiment"], base["target_experiment"]
        record = {
            "source_experiment": source_id,
            "target_experiment": target_id,
            "direction": base["direction"],
            "primary_transfer_comparison": PRIMARY_TRANSFER_COMPARISON,
            "primary_selection": canonical_json(dict(PRIMARY_TRANSFER_SELECTION)),
        }
        for key, _directed in DIAGNOSTIC_QUANTITIES:
            record[key] = base.get(key)

        for name, family, adaptation, metric in TRANSFER_QUANTITIES:
            subset = transfer[
                (transfer["source_experiment_id"] == source_id)
                & (transfer["target_experiment_id"] == target_id)
                & (transfer["model_family"] == family)
                & (transfer["adaptation_method"] == adaptation)
                & (transfer["metric"] == metric)
            ]
            field = _TRANSFER_FIELD[name]
            record[name] = float(subset.iloc[0][field]) if len(subset) and field in subset.columns else None
            if len(subset) > 1 and name in {"raw_thermal_roc_auc", "raw_thermal_pr_auc",
                                            "thermal_roc_auc_gap", "thermal_pr_auc_gap"}:
                # raw_auc/raw_gap are adaptation-independent; assert that.
                values = subset[field].astype(float).round(12).unique()
                if len(values) > 1:
                    raise MarginalAoACompletionError(
                        f"{name} for {source_id}->{target_id} differs across "
                        f"adaptation methods ({values}); it must not."
                    )
        record["within_target_auc"] = None
        subset = transfer[
            (transfer["source_experiment_id"] == source_id)
            & (transfer["target_experiment_id"] == target_id)
            & (transfer["model_family"] == "thermal")
            & (transfer["metric"] == "roc_auc")
        ]
        if len(subset) and "within_target_auc" in subset.columns:
            record["within_target_auc"] = float(subset.iloc[0]["within_target_auc"])
        rows.append(record)
    return rows


def build_ranking_summary(comparison_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Descriptive rank associations only. No p-value, ever."""
    rows: list[dict[str, Any]] = []
    for diagnostic, is_directed in DIAGNOSTIC_QUANTITIES:
        diag_values = [r.get(diagnostic) for r in comparison_rows]
        for name, _f, _a, _m in TRANSFER_QUANTITIES:
            transfer_values = [r.get(name) for r in comparison_rows]
            pairs = [
                (d, t) for d, t in zip(diag_values, transfer_values)
                if d is not None and t is not None
                and np.isfinite(float(d)) and np.isfinite(float(t))
            ]
            if len(pairs) >= 3:
                d_vals = [p[0] for p in pairs]
                t_vals = [p[1] for p in pairs]
                spearman = _spearman(d_vals, t_vals)
                kendall = _kendall(d_vals, t_vals)
                order_d = np.argsort(np.asarray(d_vals, dtype="float64"), kind="mergesort")
                order_t = np.argsort(np.asarray(t_vals, dtype="float64"), kind="mergesort")
                top3 = len(set(order_d[-3:].tolist()) & set(order_t[-3:].tolist()))
                bottom3 = len(set(order_d[:3].tolist()) & set(order_t[:3].tolist()))
            else:
                spearman = kendall = None
                top3 = bottom3 = None
            rows.append({
                "diagnostic": diagnostic,
                "transfer": name,
                "is_primary_comparison": bool(name == PRIMARY_TRANSFER_COMPARISON),
                "diagnostic_is_directed": bool(is_directed),
                "n_pairs": len(pairs),
                "spearman_rho": spearman,
                "kendall_tau": kendall,
                "top3_overlap_count": top3,
                "bottom3_overlap_count": bottom3,
                "interpretation_boundary": (
                    "In this four-AOI set, the diagnostic ordering does or does "
                    "not reproduce the observed raw-transfer ordering."
                ),
            })
    return rows


def render_scientific_summary(
    analysis_id: str,
    directed_rows: Sequence[dict[str, Any]],
    comparison_rows: Sequence[dict[str, Any]],
    ranking_rows: Sequence[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# Marginal AoA Completion — scientific summary",
        "",
        f"- Schema: `{SCHEMA_VERSION}`",
        f"- Analysis ID: `{analysis_id}`",
        f"- Diagnostic class: **{SOURCE_LABEL_POLICY['required_description']}**",
        f"- Directed pairs: {len(directed_rows)}",
        f"- Primary transfer comparison: **{PRIMARY_TRANSFER_COMPARISON}**",
        "",
        "## Component symmetry",
        "",
        "| Component | Symmetry |",
        "|---|---|",
        "| weighted predictor-space dissimilarity | directed |",
        "| climatic distance | symmetric |",
        "| geographic distance | symmetric |",
        "",
        "No composite scalar index is produced; the three components stay in",
        "separate columns.",
        "",
        "## Ordered-pair ranking (primary comparison)",
        "",
        "| Direction | mean DI | fraction inside | raw thermal ROC-AUC |",
        "|---|---:|---:|---:|",
    ]
    lookup = {(r["source_experiment"], r["target_experiment"]): r for r in comparison_rows}
    for row in sorted(
        directed_rows,
        key=lambda r: (
            r["target_mean_dissimilarity"] if r["target_mean_dissimilarity"] is not None else -1.0
        ),
        reverse=True,
    ):
        joined = lookup.get((row["source_experiment"], row["target_experiment"]), {})
        lines.append(
            f"| `{row['direction']}` | {_fmt(row['target_mean_dissimilarity'])} | "
            f"{_fmt(row['fraction_inside_weighted_aoa'])} | "
            f"{_fmt(joined.get(PRIMARY_TRANSFER_COMPARISON))} |"
        )

    primary = [r for r in ranking_rows if r["is_primary_comparison"]]
    lines += [
        "",
        "## Rank association against raw thermal ROC-AUC",
        "",
        "| Diagnostic | directed | n | Spearman rho | Kendall tau |",
        "|---|---|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            f"| `{row['diagnostic']}` | {row['diagnostic_is_directed']} | {row['n_pairs']} | "
            f"{_fmt(row['spearman_rho'])} | {_fmt(row['kendall_tau'])} |"
        )

    bejis = [r for r in directed_rows if r["source_experiment"] == "bejis_2022"]
    lines += [
        "",
        "## Bejís-source audit",
        "",
        f"Bejís is the source in {len(bejis)} directed pair(s). `elevation_mean` is",
        "both the sole driver of its unweighted range violations and its",
        "top-weighted feature, so importance weighting is expected to AMPLIFY,",
        "not offset, the Bejís elevation story. This expectation was recorded",
        "before the run.",
        "",
        "| Direction | mean DI | fraction inside |",
        "|---|---:|---:|",
    ]
    for row in sorted(bejis, key=lambda r: r["target_experiment"]):
        lines.append(
            f"| `{row['direction']}` | {_fmt(row['target_mean_dissimilarity'])} | "
            f"{_fmt(row['fraction_inside_weighted_aoa'])} |"
        )

    lines += [
        "",
        "## Interpretation boundary",
        "",
        "> In this four-AOI set, the diagnostic ordering does or does not",
        "> reproduce the observed raw-transfer ordering.",
        "",
        "No p-value, hypothesis test or confidence interval is computed for any",
        "correlation. Twelve directed pairs built from four non-independent AOIs",
        "support description, not inference, and no universal methodological",
        "claim is made.",
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {item}" for item in LIMITATIONS]
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


# =============================================================================
# Climate export plan (the ONLY stage permitted to touch Earth Engine)
# =============================================================================
def climate_export_plan(analysis_id: str, output_root: Optional[Path] = None) -> dict[str, Any]:
    raster = climate_raster_path(analysis_id, output_root)
    return {
        "stage": STAGE_CLIMATE_EXPORT,
        "collection": CLIMATE_COLLECTION,
        "period_start": CLIMATE_PERIOD_START,
        "period_end": CLIMATE_PERIOD_END,
        "period_end_exclusive": CLIMATE_PERIOD_END_EXCLUSIVE,
        "expected_month_count": CLIMATE_EXPECTED_MONTHS,
        "years": list(CLIMATE_YEARS),
        "season_months": list(CLIMATE_SEASON_MONTHS),
        "source_bands": list(CLIMATE_SOURCE_BANDS),
        "band_scale_factors": dict(CLIMATE_BAND_SCALE_FACTORS),
        "output_bands": list(CLIMATE_FEATURES),
        "variable_recipes": climate_variable_recipes(),
        "region": dict(CLIMATE_REFERENCE_WINDOW),
        "land_support": CLIMATE_LAND_MASK,
        "projection": "TerraClimate native projection and scale",
        "destination": str(raster),
        "metadata_destination": str(raster.parent / "climate_export_metadata.json"),
        "gee_queries_run": True,
        "gee_exports_run": True,
        "export_authorised": True,
    }


LIVE_EARTH_ENGINE_OPT_IN_MESSAGE = (
    "The 'climate-export' stage performs LIVE Earth Engine queries and a live "
    "raster export. It is never started implicitly.\n"
    "To run it, opt in explicitly:\n"
    "    run_analysis(..., allow_earth_engine=True)\n"
    "    python scripts/main.py marginal-aoa-completion "
    "--from-stage climate-export --to-stage climate-export --allow-earth-engine\n"
    "Tests and any non-authorised caller must instead inject a "
    "`climate_export_engine`, which exercises the same code path without "
    "contacting Earth Engine. Nothing was written."
)


def run_climate_export(
    analysis_id: str, *, dry_run: bool, output_root: Optional[Path] = None,
    engine: Any = None, force: bool = False, allow_earth_engine: bool = False,
) -> dict[str, Any]:
    """Perform the authorised TerraClimate 1991-2020 export.

    Live Earth Engine work requires a DELIBERATE opt-in. `engine` is the
    injection point -- tests substitute a fake so this exact code path runs
    without contacting Earth Engine -- and `allow_earth_engine=True` is the
    only way to reach the production engine. Without one of the two this
    raises before any session, query or write, so an Earth Engine export can
    never start as a side effect of a broader range or a test.

    Nothing here ever fabricates a raster: an incomplete or invalid export
    raises rather than leaving a placeholder behind.
    """
    plan = climate_export_plan(analysis_id, output_root)
    if dry_run:
        return {
            "stage": STAGE_CLIMATE_EXPORT, "ran": False, "dry_run": True,
            "plan": plan, "gee_queries_run": False, "gee_exports_run": False,
        }

    if engine is None and not allow_earth_engine:
        raise MarginalAoACompletionError(LIVE_EARTH_ENGINE_OPT_IN_MESSAGE)

    from src.marginal_aoa_climate_export import (
        TerraClimateExportEngine, crs_matches, validate_exported_raster,
        validate_projection,
    )

    engine = TerraClimateExportEngine() if engine is None else engine
    destination = climate_raster_path(analysis_id, output_root)
    root = analysis_root(analysis_id, output_root)
    assert_inside_namespace(destination, root)
    metadata_path = destination.parent / CLIMATE_EXPORT_METADATA_FILENAME

    if destination.is_file() and not force:
        # A final raster with no metadata and no stage marker is a PARTIAL
        # output: it is the residue of an export whose QA never passed. Fail
        # closed -- never overwrite it, never delete it, never auto-retry.
        marker = read_stage_marker(analysis_id, STAGE_CLIMATE_EXPORT, output_root)
        if not metadata_path.is_file() or marker is None:
            raise MarginalAoACompletionError(
                f"PARTIAL climate export detected at {destination}:\n"
                f"  raster present        : True\n"
                f"  export metadata present: {metadata_path.is_file()}\n"
                f"  stage marker present  : {marker is not None}\n"
                "A raster without both its metadata and its stage marker never "
                "passed validation and is not a usable export. Refusing to "
                "overwrite, delete or retry it automatically. Move it aside "
                "deliberately, then re-run this stage."
            )
        # Resume only from a COMPLETE, verified export.
        raster_audit = validate_exported_raster(
            destination, expected_bands=CLIMATE_FEATURES
        )
        return {
            "stage": STAGE_CLIMATE_EXPORT, "ran": False, "dry_run": False,
            "reused_existing_export": True, "plan": plan,
            "raster_audit": raster_audit,
            "raster_sha256": sha256_file(destination),
            "gee_queries_run": False, "gee_exports_run": False,
        }

    session = engine.initialise()

    observed_months = int(engine.monthly_image_count(
        collection_id=CLIMATE_COLLECTION,
        period_start=CLIMATE_PERIOD_START,
        period_end_exclusive=CLIMATE_PERIOD_END_EXCLUSIVE,
    ))
    assert_climate_month_count(observed_months)

    image = engine.build_four_band_image(
        collection_id=CLIMATE_COLLECTION,
        period_start=CLIMATE_PERIOD_START,
        period_end_exclusive=CLIMATE_PERIOD_END_EXCLUSIVE,
        years=CLIMATE_YEARS,
        season_months=CLIMATE_SEASON_MONTHS,
        scale_factors=CLIMATE_BAND_SCALE_FACTORS,
        output_bands=CLIMATE_FEATURES,
    )
    projection = validate_projection(engine.native_projection(CLIMATE_COLLECTION))
    export_crs = projection["export_crs"]
    export_crs_representation = projection["export_crs_representation"]
    region = engine.region(CLIMATE_REFERENCE_WINDOW)

    # Export to a hidden SIBLING temporary raster, never straight to the final
    # production path. The shared exporter promotes to whatever out_path it is
    # given before this module's climate-specific QA runs, so a QA failure
    # against the final path would leave a rejected raster in production.
    staging = destination.parent / CLIMATE_RASTER_STAGING_FILENAME
    assert_inside_namespace(staging, root)
    if staging.exists():
        staging.unlink()

    try:
        transport = engine.export(
            image,
            destination=staging,
            region=region,
            scale=projection["source_projection_nominal_scale"],
            crs=export_crs,
            band_count=CLIMATE_FEATURE_COUNT,
            tiles_dir=destination.parent / "_tiles",
            force=True,
            # The shared exporter's alignment QA compares CRS as strings, which
            # a WKT-only source cannot satisfy after GDAL normalises it on
            # write. Hand it the same tested semantic predicate this module
            # uses for its own QA. Only the CRS comparison changes.
            crs_equivalence_fn=lambda actual, expected: crs_matches(
                actual, expected, export_crs_representation
            ),
        )
        # Climate-specific QA -- band count, CRS, transform, dimensions and a
        # non-empty finite support across all four bands -- on the STAGING file.
        raster_audit = validate_exported_raster(
            staging, expected_bands=CLIMATE_FEATURES,
            expected_crs=export_crs,
            expected_crs_representation=export_crs_representation,
        )
        raster_sha256 = sha256_file(staging)
    except BaseException:
        # Any failure: the final production path is never created.
        if staging.exists():
            staging.unlink()
        raise

    # Every check passed -- promote atomically, then write metadata.
    os.replace(staging, destination)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "stage": STAGE_CLIMATE_EXPORT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "engine": getattr(engine, "name", type(engine).__name__),
        "session": session,
        "collection": CLIMATE_COLLECTION,
        "period": f"{CLIMATE_PERIOD_START}/{CLIMATE_PERIOD_END}",
        "expected_month_count": CLIMATE_EXPECTED_MONTHS,
        "observed_month_count": observed_months,
        "season_months": list(CLIMATE_SEASON_MONTHS),
        "source_bands": list(CLIMATE_SOURCE_BANDS),
        "band_scale_factors": dict(CLIMATE_BAND_SCALE_FACTORS),
        "output_bands": list(CLIMATE_FEATURES),
        "variable_recipes": climate_variable_recipes(),
        "region": dict(CLIMATE_REFERENCE_WINDOW),
        "native_projection": projection,
        "canonical_projection_band": projection["canonical_projection_band"],
        "source_projection_authority_crs": projection["source_projection_authority_crs"],
        "source_projection_wkt": projection["source_projection_wkt"],
        "source_projection_transform": projection["source_projection_transform"],
        "source_projection_nominal_scale": projection["source_projection_nominal_scale"],
        "export_crs": export_crs,
        "export_crs_representation": export_crs_representation,
        "projection_read_method": projection["projection_read_method"],
        "staging_path": str(staging),
        "final_path_promoted_after_qa": True,
        "export_transport": {
            k: v for k, v in (transport or {}).items() if k != "path"
        },
        "raster_path": str(destination),
        "raster_sha256": raster_sha256,
        "raster_audit": raster_audit,
        "gee_queries_run": True,
        "gee_exports_run": True,
    }
    _atomic_write_text(metadata_path, _json_document(metadata))

    return {
        "stage": STAGE_CLIMATE_EXPORT, "ran": True, "dry_run": False,
        "reused_existing_export": False, "plan": plan,
        "observed_month_count": observed_months,
        "raster_audit": raster_audit, "raster_sha256": raster_sha256,
        "gee_queries_run": True, "gee_exports_run": True,
    }


# =============================================================================
# Stage dependency contract -- STRICT
# =============================================================================
# Each stage's own outputs, and the stages it requires to be complete first.
# A required stage is NEVER silently skipped in an actual run: the range stops
# at the first unavailable one, before any downstream write.
STAGE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "plan": (
        "config/preregistration.json",
        "config/frozen_input_inventory.json",
        "config/feature_importance_inventory.json",
        "config/climate_input_inventory.json",
        "config/geometry_inventory.json",
        "config/transfer_input_inventory.json",
        "plan_stage_metadata.json",
    ),
    STAGE_CLIMATE_EXPORT: (
        f"climate_distance/{CLIMATE_RASTER_FILENAME}",
        "climate_distance/climate_export_metadata.json",
    ),
    "weighted-predictor-space": (
        "weighted_predictor_space/source_feature_weights.csv",
        "weighted_predictor_space/source_threshold_diagnostics.csv",
        "weighted_predictor_space/target_cell_dissimilarity.parquet",
        "weighted_predictor_space/weighted_pair_summary.csv",
    ),
    "climate-distance": (
        "climate_distance/aoi_climate_vectors.csv",
        "climate_distance/pairwise_climate_distance.csv",
    ),
    "geographic-distance": (
        "geographic_distance/aoi_geometry_summary.csv",
        "geographic_distance/pairwise_geographic_distance.csv",
    ),
    "compare": (
        "weighted_predictor_space/directed_pair_summary.csv",
        "comparison/marginal_diagnostics_with_transfer.csv",
        "comparison/ranking_summary.csv",
        "comparison/scientific_summary.md",
        "completion_metadata.json",
    ),
}

STAGE_REQUIRES: dict[str, tuple[str, ...]] = {
    "plan": (),
    STAGE_CLIMATE_EXPORT: ("plan",),
    "weighted-predictor-space": ("plan",),
    "climate-distance": ("plan", STAGE_CLIMATE_EXPORT),
    "geographic-distance": ("plan",),
    "compare": (
        "plan", "weighted-predictor-space", "climate-distance", "geographic-distance",
    ),
}

STAGE_MARKER_DIR = "stages"


def dependency_closure(stage: str) -> set[str]:
    """`stage` plus every stage it transitively requires."""
    closure: set[str] = set()
    frontier = [stage]
    while frontier:
        current = frontier.pop()
        if current in closure:
            continue
        closure.add(current)
        frontier.extend(STAGE_REQUIRES[current])
    return closure


def resolve_execution_plan(stages: Sequence[str]) -> list[str]:
    """The stages in the requested range that the RANGE'S TERMINAL STAGE needs.

    A stage that sits inside the range but that the terminal stage does not
    depend on is not part of this run's requirement, so it is not executed and
    is not "skipped" either -- it was never required. Concretely:

        plan -> weighted-predictor-space
            runs {plan, weighted-predictor-space}; the climate raster and
            geographiclib are irrelevant to it and it succeeds without them.

        plan -> compare
            compare requires all three components, and climate-distance
            requires climate-export, so the closure is every stage. An
            unavailable one therefore fails closed -- it is never skipped.
    """
    if not stages:
        return []
    needed = dependency_closure(stages[-1])
    return [stage for stage in stages if stage in needed]


def stage_marker_path(
    analysis_id: str, stage: str, output_root: Optional[Path] = None
) -> Path:
    return analysis_root(analysis_id, output_root) / STAGE_MARKER_DIR / f"{stage}.json"


def read_stage_marker(
    analysis_id: str, stage: str, output_root: Optional[Path] = None
) -> Optional[dict[str, Any]]:
    path = stage_marker_path(analysis_id, stage, output_root)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_stage_marker(
    analysis_id: str, stage: str, output_root: Optional[Path] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Record a stage as complete, with every produced file and its hash.

    This is what `--resume` verifies against: a stage counts as reusable only
    when the marker says PASS, the analysis identity and schema match, and
    every recorded file is still present with the recorded hash.
    """
    root = analysis_root(analysis_id, output_root)
    files: dict[str, str] = {}
    for relative in STAGE_OUTPUTS[stage]:
        path = root / relative
        if not path.is_file():
            raise MarginalAoACompletionError(
                f"Stage '{stage}' claims completion but did not produce {relative}."
            )
        files[relative] = sha256_file(path)

    marker = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "stage": stage,
        "status": "pass",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "requires": list(STAGE_REQUIRES[stage]),
        "files": files,
        **(extra or {}),
    }
    _atomic_write_text(stage_marker_path(analysis_id, stage, output_root),
                       _json_document(marker))
    return marker


def verify_stage_complete(
    analysis_id: str, stage: str, output_root: Optional[Path] = None
) -> dict[str, Any]:
    """Is `stage` complete AND still hash-valid?

    A stage that was skipped has no marker and is therefore never resumable as
    completed. A stage whose outputs have changed since it ran is reported
    invalid rather than quietly reused.
    """
    marker = read_stage_marker(analysis_id, stage, output_root)
    if marker is None:
        return {"complete": False, "reason": "no stage marker", "stage": stage}
    if marker.get("status") != "pass":
        return {"complete": False, "reason": f"status={marker.get('status')!r}", "stage": stage}
    if marker.get("analysis_id") != analysis_id:
        return {"complete": False, "reason": "analysis_id mismatch", "stage": stage}
    if marker.get("schema_version") != SCHEMA_VERSION:
        return {"complete": False, "reason": "schema_version mismatch", "stage": stage}

    root = analysis_root(analysis_id, output_root)
    recorded = marker.get("files", {})
    missing = sorted(r for r in STAGE_OUTPUTS[stage] if r not in recorded)
    if missing:
        raise MarginalAoACompletionError(
            f"Stage marker for '{stage}' does not record required output(s) "
            f"{missing}; the tree is partial. Refusing to reuse or overwrite it."
        )
    for relative, expected in sorted(recorded.items()):
        path = root / relative
        if not path.is_file():
            raise MarginalAoACompletionError(
                f"Stage '{stage}' is marked complete but {relative} is missing; "
                "the tree is partial. Refusing to reuse or overwrite it."
            )
        if sha256_file(path) != expected:
            raise MarginalAoACompletionError(
                f"Stage '{stage}' output {relative} no longer matches its "
                "recorded hash; the tree has been modified since it ran. "
                "Refusing to reuse or overwrite it."
            )
    return {"complete": True, "stage": stage, "files": recorded}


def stage_availability(
    stage: str, *, analysis_id: str, frozen_inventory: dict[str, Any],
    importance_inventory: dict[str, Any], transfer_inventory: dict[str, Any],
    output_root: Optional[Path] = None,
    climate_export_engine: Any = None,
    allow_earth_engine: bool = False,
    geodesic_inverse: Any = None,
) -> dict[str, Any]:
    """Can `stage` actually run right now? External capability included.

    Returns {"available": bool, "missing": [...]}. Nothing here writes.
    """
    missing: list[str] = []

    for required in STAGE_REQUIRES[stage]:
        state = verify_stage_complete(analysis_id, required, output_root)
        if not state["complete"]:
            missing.append(
                f"required stage '{required}' is not complete ({state['reason']})"
            )

    if stage == "plan":
        absent = sorted(k for k, v in frozen_inventory.items() if not v["exists"])
        if absent:
            missing.append(f"canonical Step8A dataset(s) missing for {absent}")
        absent = sorted(k for k, v in importance_inventory.items() if not v["exists"])
        if absent:
            missing.append(f"Step8B feature-importance CSV(s) missing for {absent}")

    elif stage == STAGE_CLIMATE_EXPORT:
        if climate_export_engine is None:
            if not allow_earth_engine:
                missing.append(
                    "live Earth Engine work is not authorised for this run; pass "
                    "allow_earth_engine=True (CLI: --allow-earth-engine) to run "
                    "the authorised export, or inject a climate_export_engine"
                )
            elif not earth_engine_available():
                missing.append(
                    "Earth Engine is not available (the `ee` package cannot be "
                    "imported), so the authorised TerraClimate export cannot run"
                )

    elif stage == "weighted-predictor-space":
        absent = sorted(k for k, v in frozen_inventory.items() if not v["exists"])
        if absent:
            missing.append(f"canonical Step8A dataset(s) missing for {absent}")
        absent = sorted(k for k, v in importance_inventory.items() if not v["exists"])
        if absent:
            missing.append(f"Step8B feature-importance CSV(s) missing for {absent}")

    elif stage == "climate-distance":
        raster = climate_raster_path(analysis_id, output_root)
        if not raster.is_file():
            missing.append(
                f"the frozen four-band TerraClimate raster is absent ({raster})"
            )

    elif stage == "geographic-distance":
        if geodesic_inverse is None and not geographiclib_available():
            missing.append(
                "the 'geographiclib' package is not importable; install it with "
                "`pip install geographiclib` (no haversine, pyproj or Vincenty "
                "fallback is permitted)"
            )

    elif stage == "compare":
        if not transfer_inventory["exists"]:
            missing.append(
                f"the four-AOI transfer decomposition artifact is absent "
                f"({transfer_inventory['path']})"
            )

    return {"stage": stage, "available": not missing, "missing": missing}


def assert_stage_available(availability: dict[str, Any]) -> None:
    if availability["available"]:
        return
    raise MarginalAoACompletionError(
        f"Stage '{availability['stage']}' cannot run and MUST NOT be skipped:\n  - "
        + "\n  - ".join(availability["missing"])
        + "\nThis is a required scientific stage of the completion analysis. "
        "No downstream stage was executed and no output was written. Satisfy "
        "the prerequisite, or request a narrower --to-stage that stops before "
        "this stage."
    )


def _package_available(name: str) -> bool:
    """Availability probe by spec lookup -- never imports the package.

    Using `importlib.util.find_spec` rather than a real import keeps this
    module free of any Earth Engine import, which is what makes the
    `gee_queries_run = False` claim verifiable by inspecting the source.
    """
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def earth_engine_available() -> bool:
    return _package_available("ee")


def geographiclib_available() -> bool:
    return _package_available("geographiclib")


def stage_prerequisites(
    stages: Sequence[str], analysis_id: Optional[str],
    frozen_inventory: dict[str, Any], importance_inventory: dict[str, Any],
    transfer_inventory: dict[str, Any], output_root: Optional[Path] = None,
    climate_export_engine: Any = None, allow_earth_engine: bool = False,
    geodesic_inverse: Any = None,
) -> dict[str, Any]:
    """Report, for a DRY RUN, what each requested stage would need.

    A dry run is a plan inspection: it may legitimately report that the climate
    raster is absent or that geographiclib is unavailable, and stay valid. An
    ACTUAL run does not use this report to decide anything -- it re-checks each
    stage immediately before executing it and fails closed.
    """
    checks: dict[str, Any] = {
        "step8a_inputs_present": all(v["exists"] for v in frozen_inventory.values()),
        "feature_importance_inputs_present": all(
            v["exists"] for v in importance_inventory.values()
        ),
        "climate_raster_present": bool(
            analysis_id and climate_raster_path(analysis_id, output_root).is_file()
        ),
        "transfer_input_present": transfer_inventory["exists"],
        "geographiclib_available": geographiclib_available(),
        "earth_engine_available": earth_engine_available(),
    }

    per_stage: dict[str, Any] = {}
    unavailable: list[str] = []
    for stage in stages:
        try:
            availability = stage_availability(
                stage, analysis_id=analysis_id or "",
                frozen_inventory=frozen_inventory,
                importance_inventory=importance_inventory,
                transfer_inventory=transfer_inventory,
                output_root=output_root,
                climate_export_engine=climate_export_engine,
                allow_earth_engine=allow_earth_engine,
                geodesic_inverse=geodesic_inverse,
            )
        except SystemExit as exc:  # a partial tree surfaced during inspection
            availability = {"stage": stage, "available": False, "missing": [str(exc)]}
        per_stage[stage] = availability
        if not availability["available"]:
            unavailable.append(stage)

    first_unavailable = unavailable[0] if unavailable else None
    return {
        "checks": checks,
        "per_stage": per_stage,
        "unavailable_stages": unavailable,
        "first_unavailable_stage": first_unavailable,
        "executable_prefix": list(stages[:stages.index(first_unavailable)])
        if first_unavailable else list(stages),
        "ready": not unavailable,
    }


def existing_namespace_state(
    analysis_id: str, output_root: Optional[Path] = None
) -> dict[str, Any]:
    root = analysis_root(analysis_id, output_root)
    if not root.exists():
        return {"exists": False, "complete": False, "completed_stages": [], "present": []}
    completed = [
        stage for stage in STAGES
        if (read_stage_marker(analysis_id, stage, output_root) or {}).get("status") == "pass"
    ]
    layout = planned_output_layout()
    return {
        "exists": True,
        "complete": (root / "completion_metadata.json").is_file(),
        "completed_stages": completed,
        "present": sorted(rel for rel in layout if (root / rel).is_file()),
    }


def assert_namespace_writable(
    analysis_id: str, stages: Sequence[str], resume: bool,
    output_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Refuse a silent overwrite; `--resume` verifies and reuses instead.

    A partial or invalid tree fails closed and is never deleted or rewritten
    blindly -- the analysis-ID namespace contract means a genuinely different
    configuration lands in a different directory, so no force flag exists.
    """
    state = existing_namespace_state(analysis_id, output_root)
    if not state["exists"]:
        return {"mode": "fresh", **state}
    if state["complete"] and not resume:
        raise MarginalAoACompletionError(
            f"A COMPLETE result already exists at "
            f"{analysis_root(analysis_id, output_root)} "
            f"(completion_metadata.json present). Refusing to overwrite it "
            "silently. Re-run with resume=True to verify and reuse it; a "
            "different scientific configuration would produce a different "
            "analysis_id and therefore a different namespace."
        )
    if state["complete"] and resume:
        return {"mode": "resume_complete", **state}
    return {"mode": "resume_partial" if resume else "extend", **state}


# =============================================================================
# Orchestration -- STRICT actual semantics
# =============================================================================
def run_analysis(
    experiments: Optional[Sequence[str]] = None,
    *,
    from_stage: str = "plan",
    to_stage: str = "compare",
    dry_run: bool = False,
    resume: bool = False,
    output_root: Optional[Path] = None,
    experiments_root: Optional[Path] = None,
    strict_hashes: bool = True,
    pairwise_chunk_size: int = PAIRWISE_CHUNK_SIZE,
    neighbour_chunk_size: int = NEIGHBOUR_CHUNK_SIZE,
    read_parquet=None,
    climate_export_engine: Any = None,
    geodesic_inverse: Any = None,
    allow_earth_engine: bool = False,
) -> dict[str, Any]:
    """Run the requested stage range.

    Stage validation happens FIRST -- before any prerequisite check, any input
    resolution and any filesystem inspection.

    DRY RUN is a plan inspection: it creates no directory, writes no file,
    contacts no Earth Engine, fits no model and computes no distance. It may
    legitimately report that the climate raster is absent or that geographiclib
    is unavailable and still be a valid plan.

    An ACTUAL run is strict. Every stage in the range is a REQUIRED scientific
    stage: each one re-checks its own prerequisites immediately before running,
    and an unavailable stage raises rather than being skipped. No downstream
    stage executes, no downstream file is written, no component is replaced by
    null columns, and no partial completion metadata is produced.
    """
    validate_feature_contract()
    requested_stages = validate_stage_range(from_stage, to_stage)
    stages = resolve_execution_plan(requested_stages)

    experiment_ids = resolve_experiments(experiments)
    frozen_inventory = build_frozen_input_inventory(experiment_ids, experiments_root)
    importance_inventory = build_feature_importance_inventory(experiment_ids, experiments_root)
    transfer_inventory = build_transfer_inventory(output_root)
    geometry_inventory = build_geometry_inventory(experiment_ids)

    hash_verification: dict[str, Any] = {"verified": False, "reason": "not_checked"}
    if all(v["exists"] for v in frozen_inventory.values()):
        hash_verification = assert_canonical_step8a_hashes(
            frozen_inventory, strict=strict_hashes
        )
    config = scientific_configuration(
        experiment_ids, frozen_inventory, importance_inventory, transfer_inventory,
    )
    analysis_id = compute_analysis_id(config)

    climate_inventory = build_climate_input_inventory(analysis_id, output_root)
    root = analysis_root(analysis_id, output_root)

    prerequisites = stage_prerequisites(
        stages, analysis_id, frozen_inventory, importance_inventory,
        transfer_inventory, output_root,
        climate_export_engine=climate_export_engine,
        allow_earth_engine=allow_earth_engine,
        geodesic_inverse=geodesic_inverse,
    )

    plan_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "experiments": sorted(experiment_ids),
        "primary_population": PRIMARY_POPULATION,
        "stages_requested": requested_stages,
        "stages_to_execute": stages,
        "stage_order": list(STAGES),
        "stage_requires": {k: list(v) for k, v in STAGE_REQUIRES.items()},
        "directed_pair_count": len(directed_pairs(experiment_ids)),
        "unordered_pair_count": len(unordered_pairs(experiment_ids)),
        "output_namespace": str(root),
        "planned_output_layout": planned_output_layout(),
        "canonical_step8a_hash_verification": hash_verification,
        "prerequisites": prerequisites,
        "existing_namespace": existing_namespace_state(analysis_id, output_root),
        "climate_status": climate_inventory["climate_status"],
        "geographiclib_available": geographiclib_available(),
        "geographiclib_dependency_available": geographiclib_available(),
        "earth_engine_available": earth_engine_available(),
        "live_earth_engine_authorised": bool(allow_earth_engine),
        "climate_export_engine_injected": climate_export_engine is not None,
        **stage_side_effect_flags(stages),
    }

    if dry_run:
        return {
            "ran": False, "dry_run": True, "stages_executed": [],
            "files_written": [], **plan_payload,
            "gee_queries_run": False, "gee_exports_run": False,
            "model_fit": False, "bootstrap_run": False,
        }

    namespace_state = assert_namespace_writable(analysis_id, stages, resume, output_root)
    if namespace_state["mode"] == "resume_complete":
        return {
            "ran": False, "dry_run": False, "resumed": True,
            "stages_executed": [], "files_written": [],
            "reused_namespace": str(root), **plan_payload,
            "gee_queries_run": False, "gee_exports_run": False,
            "model_fit": False, "bootstrap_run": False,
        }

    executed: list[str] = []
    reused: list[str] = []
    written: list[str] = []
    export_result: Optional[dict[str, Any]] = None
    weighted: Optional[dict[str, Any]] = None

    def _availability(stage: str) -> dict[str, Any]:
        return stage_availability(
            stage, analysis_id=analysis_id, frozen_inventory=frozen_inventory,
            importance_inventory=importance_inventory,
            transfer_inventory=transfer_inventory, output_root=output_root,
            climate_export_engine=climate_export_engine,
            allow_earth_engine=allow_earth_engine,
            geodesic_inverse=geodesic_inverse,
        )

    for stage in stages:
        if resume:
            state = verify_stage_complete(analysis_id, stage, output_root)
            if state["complete"]:
                reused.append(stage)
                continue

        # Re-checked immediately before execution. An unavailable REQUIRED
        # stage raises here, before any write belonging to it or to anything
        # downstream of it.
        assert_stage_available(_availability(stage))

        if stage == "plan":
            documents = build_plan_documents(
                analysis_id, config, experiment_ids, frozen_inventory,
                importance_inventory, transfer_inventory, geometry_inventory,
                climate_inventory, stages, prerequisites,
            )
            written.extend(write_documents(root, documents))

        elif stage == STAGE_CLIMATE_EXPORT:
            export_result = run_climate_export(
                analysis_id, dry_run=False, output_root=output_root,
                engine=climate_export_engine,
                allow_earth_engine=allow_earth_engine,
            )
            written.extend(STAGE_OUTPUTS[stage])

        elif stage == "weighted-predictor-space":
            weighted = run_weighted_predictor_space(
                analysis_id, experiment_ids, frozen_inventory, importance_inventory,
                experiments_root=experiments_root, output_root=output_root,
                pairwise_chunk_size=pairwise_chunk_size,
                neighbour_chunk_size=neighbour_chunk_size,
                read_parquet=read_parquet,
            )
            written.extend(write_weighted_outputs(
                analysis_id, weighted, experiment_ids, frozen_inventory,
                importance_inventory, output_root,
            ))

        elif stage == "climate-distance":
            climate = run_climate_distance(
                analysis_id, experiment_ids, output_root=output_root
            )
            written.extend(write_documents(root, {
                "climate_distance/aoi_climate_vectors.csv": _csv_document(
                    AOI_CLIMATE_COLUMNS, aoi_climate_rows(climate["vectors"])
                ),
                "climate_distance/pairwise_climate_distance.csv": _csv_document(
                    PAIRWISE_CLIMATE_COLUMNS, climate["rows"]
                ),
            }))

        elif stage == "geographic-distance":
            geographic = run_geographic_distance(
                experiment_ids, geodesic_inverse=geodesic_inverse
            )
            written.extend(write_documents(root, {
                "geographic_distance/aoi_geometry_summary.csv": _csv_document(
                    AOI_GEOMETRY_COLUMNS, aoi_geometry_rows(experiment_ids)
                ),
                "geographic_distance/pairwise_geographic_distance.csv": _csv_document(
                    PAIRWISE_GEOGRAPHIC_COLUMNS, geographic["rows"]
                ),
            }))

        elif stage == "compare":
            written.extend(run_compare_stage(
                analysis_id, config, experiment_ids, frozen_inventory,
                importance_inventory, transfer_inventory, geometry_inventory,
                climate_inventory, executed, output_root, experiments_root,
            ))

        write_stage_marker(analysis_id, stage, output_root)
        executed.append(stage)

    return {
        "ran": bool(executed), "dry_run": False, "resumed": bool(resume),
        "stages_executed": executed,
        "stages_reused": reused,
        "files_written": sorted(set(written)),
        "climate_export": export_result,
        **plan_payload,
        **stage_side_effect_flags(executed),
    }


# =============================================================================
# Restored stage helpers
# =============================================================================
def build_plan_documents(
    analysis_id: str, config: dict[str, Any], experiment_ids: Sequence[str],
    frozen_inventory: dict[str, Any], importance_inventory: dict[str, Any],
    transfer_inventory: dict[str, Any], geometry_inventory: dict[str, Any],
    climate_inventory: dict[str, Any], stages: Sequence[str],
    prerequisites: dict[str, Any],
) -> dict[str, str]:
    created = datetime.now(timezone.utc).isoformat()
    no_side_effects = stage_side_effect_flags(["plan"])
    pairs = directed_pairs(experiment_ids)

    documents: dict[str, str] = {}
    documents["config/preregistration.json"] = _json_document({
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "status": "frozen",
        "created_at_utc": created,
        "git_commit": _git_commit(),
        "stage": "plan",
        "written_by": "src/marginal_aoa_completion.py:run_analysis",
        "scientific_configuration": config,
        "directed_pairs": [list(p) for p in pairs],
        "pair_cardinality": len(pairs),
        "unordered_pair_cardinality": len(unordered_pairs(experiment_ids)),
        "planned_output_layout": planned_output_layout(),
        "stage_requires": {k: list(v) for k, v in STAGE_REQUIRES.items()},
        "limitations": list(LIMITATIONS),
        **no_side_effects,
    })
    documents["config/frozen_input_inventory.json"] = _json_document({
        "analysis_id": analysis_id,
        "schema_version": SCHEMA_VERSION,
        "primary_population": PRIMARY_POPULATION,
        "expected_sha256": dict(CANONICAL_STEP8A_SHA256),
        "inventory": dict(sorted(frozen_inventory.items())),
    })
    documents["config/feature_importance_inventory.json"] = _json_document({
        "analysis_id": analysis_id,
        "schema_version": SCHEMA_VERSION,
        "importance_method": IMPORTANCE_METHOD,
        "importance_method_class": IMPORTANCE_METHOD_CLASS,
        "model_algorithm": IMPORTANCE_MODEL_ALGORITHM,
        "population_filter": IMPORTANCE_POPULATION,
        "model_filter": IMPORTANCE_MODEL,
        "source_label_policy": dict(SOURCE_LABEL_POLICY),
        "mandatory_limitation": (
            "Feature weights come from RandomForest mean-decrease-in-impurity "
            "(Gini) importance, computed on a final model refit on the whole "
            "source population. Impurity importance is in-sample and is biased "
            "toward continuous and high-cardinality predictors relative to "
            "low-cardinality categorical ones. It is a source-model relevance "
            "weighting, not a causal importance estimate. A held-out "
            "permutation importance remains a possible later sensitivity."
        ),
        "inventory": dict(sorted(importance_inventory.items())),
    })
    documents["config/climate_input_inventory.json"] = _json_document({
        "analysis_id": analysis_id, "schema_version": SCHEMA_VERSION,
        **climate_inventory,
    })
    documents["config/geometry_inventory.json"] = _json_document({
        "analysis_id": analysis_id, "schema_version": SCHEMA_VERSION,
        **geometry_inventory,
    })
    documents["config/transfer_input_inventory.json"] = _json_document({
        "analysis_id": analysis_id, "schema_version": SCHEMA_VERSION,
        **transfer_inventory,
    })
    documents["plan_stage_metadata.json"] = _json_document({
        "analysis_id": analysis_id,
        "schema_version": SCHEMA_VERSION,
        "stage": "plan",
        "created_at_utc": created,
        "git_commit": _git_commit(),
        "requested_stages": list(stages),
        "prerequisites": prerequisites,
        "experiments": sorted(experiment_ids),
        "directed_pair_count": len(pairs),
        **no_side_effects,
    })
    return documents


def run_weighted_predictor_space(
    analysis_id: str, experiment_ids: Sequence[str],
    frozen_inventory: dict[str, Any], importance_inventory: dict[str, Any],
    *, experiments_root: Optional[Path] = None, output_root: Optional[Path] = None,
    pairwise_chunk_size: int = PAIRWISE_CHUNK_SIZE,
    neighbour_chunk_size: int = NEIGHBOUR_CHUNK_SIZE,
    read_parquet=None,
) -> dict[str, Any]:
    populations: dict[str, pd.DataFrame] = {}
    derived_weights: dict[str, dict[str, Any]] = {}
    prepared_sources: dict[str, dict[str, Any]] = {}

    for experiment_id in sorted(experiment_ids):
        populations[experiment_id] = load_population(
            Path(frozen_inventory[experiment_id]["path"]), experiment_id,
            read_parquet=read_parquet,
        )
        frame = read_importance_frame(
            Path(importance_inventory[experiment_id]["path"]), experiment_id
        )
        derived_weights[experiment_id] = derive_feature_weights(frame, experiment_id)

    for experiment_id in sorted(experiment_ids):
        prepared_sources[experiment_id] = prepare_source(
            experiment_id, populations[experiment_id],
            derived_weights[experiment_id]["weights"],
            pairwise_chunk_size=pairwise_chunk_size,
            neighbour_chunk_size=neighbour_chunk_size,
        )

    pair_results: list[dict[str, Any]] = []
    for source_id, target_id in directed_pairs(experiment_ids):
        pair_results.append(analyse_directed_pair(
            prepared_sources[source_id], target_id, populations[target_id],
            neighbour_chunk_size=neighbour_chunk_size,
        ))
    if len(pair_results) != EXPECTED_DIRECTED_PAIRS:
        raise MarginalAoACompletionError(
            f"Expected {EXPECTED_DIRECTED_PAIRS} directed pairs; produced "
            f"{len(pair_results)}."
        )
    return {
        "populations": populations,
        "derived_weights": derived_weights,
        "prepared_sources": prepared_sources,
        "pair_results": pair_results,
    }


def run_climate_distance(
    analysis_id: str, experiment_ids: Sequence[str],
    *, output_root: Optional[Path] = None,
) -> dict[str, Any]:
    raster = read_climate_raster(climate_raster_path(analysis_id, output_root))
    vectors = climate_vectors(raster, experiment_ids)
    rows = pairwise_climate_distances(vectors)
    if len(rows) != EXPECTED_UNORDERED_PAIRS:
        raise MarginalAoACompletionError(
            f"Expected {EXPECTED_UNORDERED_PAIRS} unordered climate distances; "
            f"produced {len(rows)}."
        )
    return {"vectors": vectors, "rows": rows, "raster_crs": raster["crs"]}


def run_geographic_distance(
    experiment_ids: Sequence[str], *, geodesic_inverse: Any = None,
) -> dict[str, Any]:
    assert_geometry_matches_registry()
    rows = pairwise_geographic_distances(
        experiment_ids, geodesic_inverse=geodesic_inverse
    )
    if len(rows) != EXPECTED_UNORDERED_PAIRS:
        raise MarginalAoACompletionError(
            f"Expected {EXPECTED_UNORDERED_PAIRS} unordered geographic distances; "
            f"produced {len(rows)}."
        )
    return {"rows": rows}


AOI_CLIMATE_COLUMNS: tuple[str, ...] = (
    ("experiment_id", "n_valid_pixels", "n_pixels_in_bbox", "climate_data_completeness")
    + tuple(CLIMATE_FEATURES)
    + tuple(f"standardised_{f}" for f in CLIMATE_FEATURES)
)

PAIRWISE_CLIMATE_COLUMNS: tuple[str, ...] = (
    "experiment_a", "experiment_b", "climate_distance", "climate_distance_squared",
    "climate_distance_metric", "climate_feature_count", "climate_features",
    "climate_reference_period", "climate_season_months", "climate_land_mask",
    "climate_source_version", "climate_component_contributions", "climate_uncertainty",
)

AOI_GEOMETRY_COLUMNS: tuple[str, ...] = (
    "experiment_id", "lon_min", "lat_min", "lon_max", "lat_max",
    "centroid_lon", "centroid_lat", "centroid_definition",
    "geometry_contract_hash", "crs", "geometry_kind",
)

PAIRWISE_GEOGRAPHIC_COLUMNS: tuple[str, ...] = (
    "experiment_a", "experiment_b", "centroid_a_lon", "centroid_a_lat",
    "centroid_b_lon", "centroid_b_lat", "centroid_geodesic_distance_km",
    "optional_minimum_boundary_distance_km", "geographic_distance_method",
    "centroid_definition", "geodesic_implementation", "bbox_a", "bbox_b",
    "geometry_contract_hash_a", "geometry_contract_hash_b",
    "geographic_distance_uncertainty",
)

COMPARISON_COLUMNS: tuple[str, ...] = (
    ("source_experiment", "target_experiment", "direction",
     "primary_transfer_comparison", "primary_selection", "within_target_auc")
    + tuple(q[0] for q in DIAGNOSTIC_QUANTITIES)
    + tuple(q[0] for q in TRANSFER_QUANTITIES)
)

RANKING_COLUMNS: tuple[str, ...] = (
    "diagnostic", "transfer", "is_primary_comparison", "diagnostic_is_directed",
    "n_pairs", "spearman_rho", "kendall_tau", "top3_overlap_count",
    "bottom3_overlap_count", "interpretation_boundary",
)


def aoi_climate_rows(vectors: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment_id, entry in sorted(vectors["aoi"].items()):
        row = {
            "experiment_id": experiment_id,
            "n_valid_pixels": entry["n_valid_pixels"],
            "n_pixels_in_bbox": entry["n_pixels_in_bbox"],
            "climate_data_completeness": entry["climate_data_completeness"],
        }
        for feature in CLIMATE_FEATURES:
            row[feature] = entry["raw"][feature]
            row[f"standardised_{feature}"] = entry["standardised"][feature]
        rows.append(row)
    return rows


def aoi_geometry_rows(experiment_ids: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment_id in sorted(experiment_ids):
        lon_min, lat_min, lon_max, lat_max = CANONICAL_AOI_BBOX[experiment_id]
        lon, lat = bbox_centre(CANONICAL_AOI_BBOX[experiment_id])
        rows.append({
            "experiment_id": experiment_id,
            "lon_min": lon_min, "lat_min": lat_min,
            "lon_max": lon_max, "lat_max": lat_max,
            "centroid_lon": lon, "centroid_lat": lat,
            "centroid_definition": CENTROID_DEFINITION,
            "geometry_contract_hash": geometry_contract_hash(experiment_id),
            "crs": "EPSG:4326",
            "geometry_kind": "axis_aligned_rectangle",
        })
    return rows



# =============================================================================
# Stage output writers
# =============================================================================
WEIGHTED_PAIR_COLUMNS: tuple[str, ...] = tuple(
    c for c in DIRECTED_PAIR_COLUMNS
    if not c.startswith("climate_")
    and not c.startswith("source_centroid_")
    and not c.startswith("target_centroid_")
    and c not in {
        "centroid_geodesic_distance_km", "optional_minimum_boundary_distance_km",
        "geographic_distance_method", "centroid_definition",
        "geodesic_implementation", "geographic_component_reads_step8a",
        "population_centroid_reported", "geographic_distance_uncertainty",
    }
)


def write_weighted_outputs(
    analysis_id: str, weighted: dict[str, Any], experiment_ids: Sequence[str],
    frozen_inventory: dict[str, Any], importance_inventory: dict[str, Any],
    output_root: Optional[Path] = None,
) -> list[str]:
    """Write the weighted predictor-space block ONLY.

    The integrated 12-row table is assembled by `compare`, once all three
    components exist. This stage never emits a climate or geographic column --
    not even a null one -- so an incomplete run can never be mistaken for a
    complete diagnostic.
    """
    root = analysis_root(analysis_id, output_root)
    weight_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    for experiment_id in sorted(experiment_ids):
        weight_rows.extend(source_weight_rows(
            weighted["prepared_sources"][experiment_id],
            weighted["derived_weights"][experiment_id],
        ))
        threshold_rows.append(source_threshold_row(
            weighted["prepared_sources"][experiment_id]
        ))

    cell_rows: list[dict[str, Any]] = []
    for result in weighted["pair_results"]:
        cell_rows.extend(result["rows"])
    cell_frame = pd.DataFrame(cell_rows, columns=list(TARGET_CELL_COLUMNS))
    cell_frame = cell_frame.sort_values(
        ["source_experiment", "target_experiment", "row_500m", "col_500m"],
        kind="mergesort",
    ).reset_index(drop=True)

    pair_rows = build_directed_pair_rows(
        analysis_id, weighted["pair_results"], weighted["prepared_sources"],
        weighted["derived_weights"], frozen_inventory, importance_inventory,
        None, None, "not_joined_at_this_stage", output_root,
    )

    written = write_documents(root, {
        "weighted_predictor_space/source_feature_weights.csv": _csv_document(
            SOURCE_WEIGHT_COLUMNS, weight_rows
        ),
        "weighted_predictor_space/source_threshold_diagnostics.csv": _csv_document(
            SOURCE_THRESHOLD_COLUMNS, threshold_rows
        ),
        "weighted_predictor_space/weighted_pair_summary.csv": _csv_document(
            WEIGHTED_PAIR_COLUMNS, pair_rows
        ),
    })
    parquet_path = root / "weighted_predictor_space/target_cell_dissimilarity.parquet"
    assert_inside_namespace(parquet_path, root)
    _atomic_write_parquet(parquet_path, cell_frame)
    written.append("weighted_predictor_space/target_cell_dissimilarity.parquet")
    return written


REQUIRED_CLIMATE_COLUMNS: tuple[str, ...] = ("climate_distance",)
REQUIRED_GEOGRAPHIC_COLUMNS: tuple[str, ...] = ("centroid_geodesic_distance_km",)


def assert_no_null_required_components(rows: Sequence[dict[str, Any]]) -> None:
    """No required component may reach the integrated table as a null.

    `compare` runs only after all three components exist, so a null here means
    a join silently failed -- which must fail closed rather than ship a table
    that looks complete.
    """
    problems: list[str] = []
    for row in rows:
        for column in REQUIRED_CLIMATE_COLUMNS + REQUIRED_GEOGRAPHIC_COLUMNS:
            value = row.get(column)
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                problems.append(
                    f"{row.get('source_experiment')}->{row.get('target_experiment')}: "
                    f"{column} is {value!r}"
                )
    if problems:
        raise MarginalAoACompletionError(
            "The integrated directed-pair table has NULL required component "
            "value(s); a required diagnostic did not reach the join:\n  - "
            + "\n  - ".join(problems)
        )


def run_compare_stage(
    analysis_id: str, config: dict[str, Any], experiment_ids: Sequence[str],
    frozen_inventory: dict[str, Any], importance_inventory: dict[str, Any],
    transfer_inventory: dict[str, Any], geometry_inventory: dict[str, Any],
    climate_inventory: dict[str, Any], executed: Sequence[str],
    output_root: Optional[Path] = None, experiments_root: Optional[Path] = None,
) -> list[str]:
    """Assemble the integrated table and the read-only comparison layer.

    Reached only after `plan`, `weighted-predictor-space`, `climate-distance`
    and `geographic-distance` are all complete and hash-valid, so all three
    components are present by construction.
    """
    root = analysis_root(analysis_id, output_root)

    weighted_rows = pd.read_csv(
        root / "weighted_predictor_space" / "weighted_pair_summary.csv"
    ).to_dict("records")
    climate_rows = pd.read_csv(
        root / "climate_distance" / "pairwise_climate_distance.csv"
    ).to_dict("records")
    geographic_rows = pd.read_csv(
        root / "geographic_distance" / "pairwise_geographic_distance.csv"
    ).to_dict("records")

    if len(weighted_rows) != EXPECTED_DIRECTED_PAIRS:
        raise MarginalAoACompletionError(
            f"weighted_pair_summary.csv has {len(weighted_rows)} rows; "
            f"{EXPECTED_DIRECTED_PAIRS} directed pairs are required."
        )
    for name, rows in (("climate", climate_rows), ("geographic", geographic_rows)):
        if len(rows) != EXPECTED_UNORDERED_PAIRS:
            raise MarginalAoACompletionError(
                f"pairwise_{name}_distance.csv has {len(rows)} rows; "
                f"{EXPECTED_UNORDERED_PAIRS} unordered pairs are required."
            )

    climate_lookup = symmetric_lookup(climate_rows, "climate_distance")
    climate_contrib = symmetric_lookup(climate_rows, "climate_component_contributions")
    geo_centroid = symmetric_lookup(geographic_rows, "centroid_geodesic_distance_km")
    geo_boundary = symmetric_lookup(
        geographic_rows, "optional_minimum_boundary_distance_km"
    )

    integrated: list[dict[str, Any]] = []
    for row in weighted_rows:
        source_id, target_id = row["source_experiment"], row["target_experiment"]
        source_lon, source_lat = bbox_centre(CANONICAL_AOI_BBOX[source_id])
        target_lon, target_lat = bbox_centre(CANONICAL_AOI_BBOX[target_id])
        merged = dict(row)
        merged.update({
            "climate_distance": climate_lookup.get((source_id, target_id)),
            "climate_distance_metric": CLIMATE_DISTANCE_METRIC,
            "climate_feature_count": CLIMATE_FEATURE_COUNT,
            "climate_features": canonical_json(list(CLIMATE_FEATURES)),
            "climate_reference_period": f"{CLIMATE_PERIOD_START}/{CLIMATE_PERIOD_END}",
            "climate_land_mask": CLIMATE_LAND_MASK,
            "climate_source_version": CLIMATE_COLLECTION,
            "climate_status": "available",
            "climate_export_authorised": True,
            "climate_uncertainty": "deterministic_aoi_level_value_no_interval",
            "climate_component_contributions": climate_contrib.get((source_id, target_id)),
            "source_centroid_lon": source_lon,
            "source_centroid_lat": source_lat,
            "target_centroid_lon": target_lon,
            "target_centroid_lat": target_lat,
            "centroid_geodesic_distance_km": geo_centroid.get((source_id, target_id)),
            "optional_minimum_boundary_distance_km": geo_boundary.get((source_id, target_id)),
            "geographic_distance_method": GEOGRAPHIC_DISTANCE_METHOD,
            "centroid_definition": CENTROID_DEFINITION,
            "geodesic_implementation": GEODESIC_IMPLEMENTATION,
            "geographic_component_reads_step8a": False,
            "population_centroid_reported": False,
            "geographic_distance_uncertainty": "deterministic_aoi_level_value_no_interval",
        })
        integrated.append(merged)

    assert_no_null_required_components(integrated)

    transfer = load_transfer_table(Path(transfer_inventory["path"]))
    comparison_rows = build_comparison_rows(integrated, transfer)
    ranking_rows = build_ranking_summary(comparison_rows)

    written = write_documents(root, {
        "weighted_predictor_space/directed_pair_summary.csv": _csv_document(
            DIRECTED_PAIR_COLUMNS, integrated
        ),
        "comparison/marginal_diagnostics_with_transfer.csv": _csv_document(
            COMPARISON_COLUMNS, comparison_rows
        ),
        "comparison/ranking_summary.csv": _csv_document(RANKING_COLUMNS, ranking_rows),
        "comparison/scientific_summary.md": render_scientific_summary(
            analysis_id, integrated, comparison_rows, ranking_rows
        ),
    })

    stages_done = sorted(
        {s for s in STAGES if (read_stage_marker(analysis_id, s, output_root) or {}).get("status") == "pass"}
        | set(executed) | {"compare"},
        key=STAGES.index,
    )
    metadata = build_completion_metadata(
        analysis_id, config, experiment_ids, stages_done, frozen_inventory,
        importance_inventory, transfer_inventory, geometry_inventory,
        build_climate_input_inventory(analysis_id, output_root),
        root, output_root, experiments_root,
    )
    _atomic_write_text(root / "completion_metadata.json", _json_document(metadata))
    written.append("completion_metadata.json")
    return written


def build_completion_metadata(
    analysis_id: str, config: dict[str, Any], experiment_ids: Sequence[str],
    executed: Sequence[str], frozen_inventory: dict[str, Any],
    importance_inventory: dict[str, Any], transfer_inventory: dict[str, Any],
    geometry_inventory: dict[str, Any], climate_inventory: dict[str, Any],
    root: Path, output_root: Optional[Path], experiments_root: Optional[Path],
) -> dict[str, Any]:
    required = ("plan", "weighted-predictor-space", "climate-distance",
                "geographic-distance", "compare")
    incomplete = [s for s in required if s not in set(executed)]
    if incomplete:
        raise MarginalAoACompletionError(
            f"Refusing to write completion metadata: required stage(s) "
            f"{incomplete} are not complete. Partial PASS metadata is never "
            "produced."
        )

    # `stages/` holds per-stage run bookkeeping, each entry carrying its own
    # file hashes, and the compare marker is written after this map by
    # construction. It is deliberately excluded: output_sha256 binds the
    # scientific outputs.
    output_hashes = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).parts[0] != STAGE_MARKER_DIR
    }
    v1_root = marginal_aoa_v1_root(output_root)
    v1_hashes = {
        str(path.relative_to(v1_root)): sha256_file(path)
        for path in sorted(v1_root.rglob("*")) if path.is_file()
    } if v1_root.exists() else {}

    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "experiments": sorted(experiment_ids),
        "primary_population": PRIMARY_POPULATION,
        "pair_cardinality": EXPECTED_DIRECTED_PAIRS,
        "scientific_configuration": config,
        "stages_executed": list(executed),
        "required_stages_complete": True,
        "canonical_step8a_hashes": {
            k: v["sha256"] for k, v in sorted(frozen_inventory.items())
        },
        "source_importance_hashes": {
            k: v["sha256"] for k, v in sorted(importance_inventory.items())
        },
        "transfer_input": dict(transfer_inventory),
        "geometry_source": {
            "path": geometry_inventory["geometry_source_path"],
            "sha256": geometry_inventory["geometry_source_sha256"],
        },
        "climate_source": climate_inventory,
        "output_sha256": output_hashes,
        "output_sha256_excludes": [f"{STAGE_MARKER_DIR}/"],
        "marginal_aoa_v1_hashes": v1_hashes,
        "canonical_outputs_modified": False,
        "existing_marginal_aoa_outputs_modified": False,
        "comparison_written_after_diagnostics": True,
        "target_label_firewall": dict(TARGET_LABEL_FIREWALL),
        "source_label_policy": dict(SOURCE_LABEL_POLICY),
        "model_fit": False,
        "bootstrap_run": False,
        "uncertainty_policy": "point_estimate_only",
        "composite_index_produced": False,
        "primary_transfer_comparison": PRIMARY_TRANSFER_COMPARISON,
        "gee_queries_run": STAGE_CLIMATE_EXPORT in executed,
        "gee_exports_run": STAGE_CLIMATE_EXPORT in executed,
        "limitations": list(LIMITATIONS),
    }
