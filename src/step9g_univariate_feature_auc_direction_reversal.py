"""
step9g_univariate_feature_auc_direction_reversal.py

Step9G: preregistered UNIVARIATE feature-AUC direction-reversal diagnostic for
an arbitrary source/target experiment pair. The original Manavgat 2021 <->
Bejis 2022 namespace and frozen results remain compatible.

SCIENTIFIC PURPOSE
------------------
For each predefined numeric predictor, measure whether higher raw feature
values rank burned cells above unburned cells (ROC-AUC with burned as the
positive class), computed SEPARATELY within each region on the SAME primary
population used by frozen Step9E/Step9F/Step10 (burnable_tree_shrub_grass).
Then identify features whose raw-value AUC lies on OPPOSITE sides of 0.5
between the two regions -- i.e. marginal feature-label relationship-direction
instability.

STRICT NON-CLAIMS (enforced in report text): this is a post-hoc marginal
diagnostic. It is NOT causal proof, NOT proof that concept shift is the only
transfer-failure mechanism, NOT successful prediction, NOT target-label
adaptation, NOT prediction inversion, and asserts NO statistical
significance. An AUC below 0.5 is NOT "poor performance" here -- its sign is
the diagnostic quantity, and it is never inverted.

CODE-PATH DISCIPLINE
--------------------
This module does not rerun or modify Step8/Step9A-F/Step10. It only:
  - reads the frozen Step8A 500 m modeling datasets (read-only),
  - reads frozen Step9E/Step9F outputs for integration (read-only, schema
    inspected -- never fabricated),
  - reuses shared infrastructure rather than reimplementing it:
      * block construction: step8b.add_spatial_block_id (floor(row/size),
        floor(col/size), fixed origin (0,0))
      * canonical JSON / SHA-256 / package versions / git commit:
        step8_large_block_robustness helpers
      * path + population helpers: step9a / step9e conventions
Outputs are written ONLY under the new diagnostics namespace.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT

# --- Reused infrastructure (NOT reimplemented). ---
from src.step8_large_block_robustness import (
    canonical_json,
    sha256_bytes,
    sha256_file,
    _git_commit,
    _package_versions,
)
from src.step8b_train_baseline_vs_thermal_model import add_spatial_block_id
from src.step9a_audit_cross_region_inputs import (
    resolve_feature_contract,
    resolve_step8a_dataset_path,
)

log, log_file = setup_logger("step9g_univariate_feature_auc_direction_reversal")

SCHEMA_VERSION = "step9g.univariate_feature_auc_direction_reversal.v1"

SOURCE_ID = "manavgat_2021"
TARGET_ID = "bejis_2022"
EXPERIMENT_IDS = (SOURCE_ID, TARGET_ID)
PAIR_TOKEN = f"{SOURCE_ID}__{TARGET_ID}"

TARGET_COLUMN = "burned"
PRIMARY_POPULATION = "burnable_tree_shrub_grass"

# Exact fixed feature list and order -- never mutated after seeing results.
NUMERIC_FEATURES = (
    "ndvi_mean",
    "elevation_mean",
    "slope_mean",
    "lst_anomaly_mean",
    "current_lst_mean",
    "current_tvdi_mean",
    "tvdi_difference_mean",
    "downscaled_lst_mean",
    "fused_lst_mean",
)
LANDCOVER_COLUMN = "landcover_dominant"
LANDCOVER_EXCLUSION_REASON = (
    "landcover_dominant is categorical; its integer class codes have no "
    "scientifically meaningful scalar ordering, so a raw-code ROC-AUC would "
    "be invalid. Reported as a separate descriptive table instead."
)

BLOCK_SIZE_CELLS = 10
NOMINAL_BLOCK_SCALE = "approximately_5_km"
BLOCK_ORIGIN = (0, 0)

BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 42
CI_LOWER_PCT = 2.5
CI_UPPER_PCT = 97.5
MIN_VALID_REPLICATES = 900

DIRECTION_HIGHER = "higher_values_rank_burned"
DIRECTION_LOWER = "lower_values_rank_burned"
DIRECTION_CHANCE = "exactly_at_chance"
DIRECTION_UNAVAILABLE = "unavailable"

OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "diagnostics"
    / "step9g_univariate_feature_auc_direction_reversal" / PAIR_TOKEN
)


def output_root_for(source_id: str, target_id: str) -> Path:
    return (
        PROJECT_ROOT / "outputs" / "diagnostics"
        / "step9g_univariate_feature_auc_direction_reversal"
        / f"{source_id}__{target_id}"
    )


def _pair_ids(source_id: str, target_id: str) -> tuple[str, str]:
    if source_id == target_id:
        raise Step9GError("--source and --target must be different experiment IDs.")
    return source_id, target_id


def validate_feature_contracts(source_id: str, target_id: str) -> dict[str, Any]:
    source_contract, source_errors = resolve_feature_contract(source_id)
    target_contract, target_errors = resolve_feature_contract(target_id)
    errors = source_errors + target_errors
    if source_contract != target_contract:
        differing = sorted(
            key for key in set(source_contract) | set(target_contract)
            if source_contract.get(key) != target_contract.get(key)
        )
        errors.append(f"Source/target feature contracts differ: {differing}.")
    if errors:
        raise Step9GError(" ".join(errors))
    required = set(NUMERIC_FEATURES) | {LANDCOVER_COLUMN}
    missing = sorted(required - set(source_contract))
    if missing:
        raise Step9GError(f"Step9G feature contract is missing required features: {missing}.")
    return source_contract


class Step9GError(SystemExit):
    """Fail-fast error for Step9G (same convention as other steps)."""


# =============================================================================
# Path resolution (read-only frozen references) + namespacing safety
# =============================================================================
def _cross_region_root(source_id: str, target_id: str) -> Path:
    return PROJECT_ROOT / "outputs" / "cross_region" / f"{source_id}__{target_id}"


def step9e_dir(source_id: str, target_id: str) -> Path:
    return _cross_region_root(source_id, target_id) / "step9e"


def step9f_dir(source_id: str, target_id: str) -> Path:
    return _cross_region_root(source_id, target_id) / "step9f"


def step10_dir(source_id: str, target_id: str) -> Path:
    return _cross_region_root(source_id, target_id) / "step10"


def _assert_output_namespace_isolated(path: Path, output_root: Path = OUTPUT_ROOT) -> None:
    """Every write MUST live under the Step9G diagnostics namespace, never in
    an existing Step9/Step10/experiment directory."""
    resolved = path.resolve()
    allowed_root = output_root.resolve()
    if allowed_root not in resolved.parents and resolved != allowed_root:
        raise Step9GError(
            f"Namespace isolation FAILED: '{path}' is outside the Step9G "
            f"diagnostics namespace '{output_root}'."
        )
    forbidden = ("cross_region", "step9e", "step9f", "step10", "step8b", "step8c")
    parts = set(resolved.parts)
    # 'diagnostics' root is fine; only reject if the path escaped into a
    # frozen analysis tree.
    if "diagnostics" not in parts and (parts & set(forbidden)):
        raise Step9GError(f"Namespace isolation FAILED: '{path}' targets a frozen analysis tree.")


# =============================================================================
# Frozen-input protection (SHA-256 before & after)
# =============================================================================
def _hash_tree(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)): sha256_file(p)
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def protected_paths(
    source_id: str = SOURCE_ID, target_id: str = TARGET_ID,
) -> dict[str, Any]:
    """SHA-256 of every frozen file this analysis must never modify."""
    protected: dict[str, Any] = {"step8a_inputs": {}, "step9e_trees": {}, "step9f_trees": {}, "step10_trees": {}}
    experiment_ids = (source_id, target_id)
    for experiment in experiment_ids:
        path = resolve_step8a_dataset(experiment)
        if not path.is_file():
            raise Step9GError(f"Frozen Step8A input not found for '{experiment}': {path}")
        protected["step8a_inputs"][experiment] = sha256_file(path)
    for src_id, tgt_id in ((source_id, target_id), (target_id, source_id)):
        key = f"{src_id}__{tgt_id}"
        protected["step9e_trees"][key] = _hash_tree(step9e_dir(src_id, tgt_id))
        protected["step9f_trees"][key] = _hash_tree(step9f_dir(src_id, tgt_id))
        protected["step10_trees"][key] = _hash_tree(step10_dir(src_id, tgt_id))
    return protected


def assert_protected_unchanged(before: dict[str, Any], after: dict[str, Any]) -> None:
    if before != after:
        # Find a representative changed key for the error message.
        changed: list[str] = []
        for group in before:
            if before[group] != after.get(group):
                changed.append(group)
        raise Step9GError(
            "Frozen-output protection FAILED; a protected input/reference "
            f"changed during the run (groups: {changed})."
        )


# =============================================================================
# Data loading + population + block assignment
# =============================================================================
def experiments_root() -> Path:
    """`outputs/experiments` root, derived from THIS module's PROJECT_ROOT.

    Every Step8A resolution in this module goes through here and passes the
    result explicitly to the resolver, so monkeypatching `PROJECT_ROOT` on this
    module actually redirects the lookup. Without it, the resolver would fall
    back to `src.step9a_audit_cross_region_inputs`'s own global and reach the
    real repository artefacts.
    """
    return PROJECT_ROOT / "outputs" / "experiments"


def resolve_step8a_dataset(experiment_id: str) -> Path:
    return resolve_step8a_dataset_path(experiment_id, experiments_root=experiments_root())


def load_step8a(experiment_id: str) -> pd.DataFrame:
    path = resolve_step8a_dataset(experiment_id)
    if not path.is_file():
        raise Step9GError(f"Frozen Step8A dataset not found for '{experiment_id}': {path}")
    return pd.read_parquet(path)


def assign_blocks_then_filter(df: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    """Assign the 10-cell block BEFORE any valid/population/complete-case
    filtering, then subset to the primary population's valid rows. Uses the
    shared Step8 block helper (floor(row/10), floor(col/10), origin (0,0))."""
    for col in ("row_500m", "col_500m", TARGET_COLUMN, "valid_for_modeling"):
        if col not in df.columns:
            raise Step9GError(f"{experiment_id}: required column '{col}' missing from Step8A dataset.")
    if PRIMARY_POPULATION not in df.columns:
        raise Step9GError(f"{experiment_id}: primary population column '{PRIMARY_POPULATION}' missing.")

    blocked = add_spatial_block_id(
        df, BLOCK_SIZE_CELLS,
        column_name="large_block_id", id_prefix=f"b{BLOCK_SIZE_CELLS}", include_row_col=True,
    )
    valid = blocked[blocked["valid_for_modeling"] == True]  # noqa: E712
    pop = valid[valid[PRIMARY_POPULATION].astype(bool)].reset_index(drop=True)
    return pop


def validate_population(df_pop: pd.DataFrame, experiment_id: str) -> dict[str, Any]:
    if "cell_id" in df_pop.columns and df_pop["cell_id"].duplicated().any():
        raise Step9GError(f"{experiment_id}: duplicate cell_id in primary population.")
    y = pd.to_numeric(df_pop[TARGET_COLUMN], errors="coerce")
    if not set(pd.unique(y.dropna())).issubset({0, 1}):
        raise Step9GError(f"{experiment_id}: target '{TARGET_COLUMN}' is not binary 0/1.")
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise Step9GError(f"{experiment_id}: primary population lacks both label classes (pos={n_pos}, neg={n_neg}).")
    return {
        "experiment_id": experiment_id,
        "primary_population": PRIMARY_POPULATION,
        "n_rows": int(len(df_pop)),
        "n_burned": n_pos,
        "n_unburned": n_neg,
        "n_large_blocks": int(df_pop["large_block_id"].nunique()),
    }


# =============================================================================
# Univariate AUC (raw feature value as score; complete-case; no imputation)
# =============================================================================
def univariate_feature_stats(df_pop: pd.DataFrame, feature: str) -> dict[str, Any]:
    n_total = int(len(df_pop))
    if feature not in df_pop.columns:
        return {
            "feature": feature, "n_total_population": n_total, "available": False,
            "raw_univariate_auc": None, "signed_rank_effect": None, "direction": DIRECTION_UNAVAILABLE,
        }
    x = pd.to_numeric(df_pop[feature], errors="coerce")
    y = pd.to_numeric(df_pop[TARGET_COLUMN], errors="coerce")
    complete = x.notna() & y.notna()
    xc = x[complete].to_numpy()
    yc = y[complete].to_numpy().astype(int)

    n_complete = int(complete.sum())
    n_missing = n_total - n_complete
    burned_mask = yc == 1
    burned_vals, unburned_vals = xc[burned_mask], xc[~burned_mask]
    n_burned_c, n_unburned_c = int(burned_mask.sum()), int((~burned_mask).sum())

    # missingness by target class (over the full primary population)
    y_full = y.fillna(-1).astype(int)
    n_burned_pop = int((y_full == 1).sum())
    n_unburned_pop = int((y_full == 0).sum())
    missing_burned = int(((~x.notna()) & (y_full == 1)).sum())
    missing_unburned = int(((~x.notna()) & (y_full == 0)).sum())

    out: dict[str, Any] = {
        "feature": feature,
        "available": True,
        "n_total_population": n_total,
        "n_complete_case": n_complete,
        "n_missing": n_missing,
        "n_burned_complete": n_burned_c,
        "n_unburned_complete": n_unburned_c,
        "missing_rate_burned": (missing_burned / n_burned_pop) if n_burned_pop else None,
        "missing_rate_unburned": (missing_unburned / n_unburned_pop) if n_unburned_pop else None,
        "burned_mean": float(np.mean(burned_vals)) if n_burned_c else None,
        "unburned_mean": float(np.mean(unburned_vals)) if n_unburned_c else None,
        "burned_median": float(np.median(burned_vals)) if n_burned_c else None,
        "unburned_median": float(np.median(unburned_vals)) if n_unburned_c else None,
        "raw_univariate_auc": None,
        "signed_rank_effect": None,
        "direction": DIRECTION_UNAVAILABLE,
    }
    if n_burned_c == 0 or n_unburned_c == 0 or len(np.unique(xc)) < 2:
        return out

    auc = float(roc_auc_score(yc, xc))  # RAW value as score; never inverted
    out["raw_univariate_auc"] = auc
    out["signed_rank_effect"] = 2.0 * auc - 1.0
    out["direction"] = _direction_label(auc)
    return out


def _direction_label(auc: float | None) -> str:
    if auc is None:
        return DIRECTION_UNAVAILABLE
    if auc > 0.5:
        return DIRECTION_HIGHER
    if auc < 0.5:
        return DIRECTION_LOWER
    return DIRECTION_CHANCE


# =============================================================================
# Spatial-block bootstrap (whole blocks, with replacement, multiplicity kept)
# =============================================================================
def _block_bootstrap_auc(df_pop: pd.DataFrame, feature: str, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Returns per-replicate AUC series for one region-feature pair. Blocks
    are sampled with replacement; ALL rows of each sampled block are included
    (multiplicity preserved). Within a replicate, only that replicate's
    complete-case rows are scored; a replicate is invalid if those rows lack
    both classes. Point AUC is block-size-independent (full complete-case)."""
    point = univariate_feature_stats(df_pop, feature)
    if not point["available"] or point["raw_univariate_auc"] is None:
        return {"point_auc": None, "replicate_aucs": np.array([]), "valid": 0, "invalid": 0, "stable": False, "unavailable": True}

    rng = np.random.default_rng(seed)
    blocks = df_pop["large_block_id"].to_numpy()
    unique_blocks = pd.unique(blocks)
    n_blocks = len(unique_blocks)
    block_to_rows = {b: np.flatnonzero(blocks == b) for b in unique_blocks}

    x_all = pd.to_numeric(df_pop[feature], errors="coerce").to_numpy()
    y_all = pd.to_numeric(df_pop[TARGET_COLUMN], errors="coerce").to_numpy()

    replicate_aucs: list[float] = []
    invalid = 0
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(unique_blocks, size=n_blocks, replace=True)
        idx = np.concatenate([block_to_rows[b] for b in sampled])
        xr, yr = x_all[idx], y_all[idx]
        cc = ~np.isnan(xr) & ~np.isnan(yr)
        xr, yr = xr[cc], yr[cc].astype(int)
        if len(np.unique(yr)) < 2 or len(np.unique(xr)) < 2:
            invalid += 1
            continue
        replicate_aucs.append(float(roc_auc_score(yr, xr)))

    arr = np.array(replicate_aucs)
    valid = len(arr)
    return {
        "point_auc": point["raw_univariate_auc"],
        "replicate_aucs": arr,
        "valid": valid,
        "invalid": invalid,
        "stable": valid >= MIN_VALID_REPLICATES,
        "unavailable": False,
    }


def _percentile_ci(arr: np.ndarray) -> tuple[float | None, float | None]:
    if arr.size == 0:
        return None, None
    return float(np.percentile(arr, CI_LOWER_PCT)), float(np.percentile(arr, CI_UPPER_PCT))


def _support_status(ci_low: float | None, ci_high: float | None, stable: bool) -> str:
    if not stable or ci_low is None or ci_high is None:
        return "unstable_bootstrap"
    if ci_low > 0.5:
        return "bootstrap_supported_higher_values_rank_burned"
    if ci_high < 0.5:
        return "bootstrap_supported_lower_values_rank_burned"
    return "interval_includes_chance"


# =============================================================================
# Direction-reversal classification + secondary cross-region contrast
# =============================================================================
def _point_direction_reversal(auc_m: float | None, auc_b: float | None) -> bool:
    if auc_m is None or auc_b is None:
        return False
    return (auc_m < 0.5 < auc_b) or (auc_b < 0.5 < auc_m)


def _reversal_status(m: dict[str, Any], b: dict[str, Any]) -> str:
    auc_m, auc_b = m["point_auc"], b["point_auc"]
    if m.get("unavailable") or b.get("unavailable") or auc_m is None or auc_b is None:
        return "unavailable"
    if not m["stable"] or not b["stable"]:
        return "unavailable"
    if not _point_direction_reversal(auc_m, auc_b):
        return "no_direction_reversal"
    m_lo, m_hi = m["ci_low"], m["ci_high"]
    b_lo, b_hi = b["ci_low"], b["ci_high"]
    m_below = m_hi < 0.5
    m_above = m_lo > 0.5
    b_below = b_hi < 0.5
    b_above = b_lo > 0.5
    if (m_below and b_above) or (m_above and b_below):
        return "bootstrap_supported_direction_reversal"
    return "point_direction_reversal_interval_uncertain"


def _contrast_ci(m_arr: np.ndarray, b_arr: np.ndarray) -> tuple[float | None, float | None, float | None]:
    """Secondary contrast target - source. The two regions were bootstrapped
    INDEPENDENTLY; replicate indices are paired only AFTER the independent
    draws exist (element-wise over the common valid count)."""
    if m_arr.size == 0 or b_arr.size == 0:
        return None, None, None
    n = min(m_arr.size, b_arr.size)
    diff = b_arr[:n] - m_arr[:n]
    return float(np.mean(diff)), float(np.percentile(diff, CI_LOWER_PCT)), float(np.percentile(diff, CI_UPPER_PCT))


# =============================================================================
# Landcover descriptive table
# =============================================================================
def landcover_descriptive(df_pop: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    if LANDCOVER_COLUMN not in df_pop.columns:
        return pd.DataFrame()
    total = len(df_pop)
    y = pd.to_numeric(df_pop[TARGET_COLUMN], errors="coerce").fillna(-1).astype(int)
    rows = []
    for code, group in df_pop.groupby(LANDCOVER_COLUMN):
        gy = y.loc[group.index]
        burned = int((gy == 1).sum())
        unburned = int((gy == 0).sum())
        n = int(len(group))
        rows.append({
            "experiment_id": experiment_id,
            "landcover_class_code": code,
            "row_count": n,
            "population_fraction": (n / total) if total else None,
            "burned_count": burned,
            "unburned_count": unburned,
            "burned_prevalence": (burned / n) if n else None,
        })
    return pd.DataFrame(rows)


# =============================================================================
# Frozen Step9E / Step9F integration (read-only; never fabricate fields)
# =============================================================================
def _read_json_if_exists(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None


def step9e_feature_integration(
    source_id: str = SOURCE_ID, target_id: str = TARGET_ID,
) -> pd.DataFrame:
    """Join available Step9E per-feature relationship-direction flags for the
    primary population, for BOTH transfer directions. Only fields that exist
    in the frozen CSV are surfaced; absent fields become None (never invented).
    """
    rows = []
    for src_id, tgt_id in ((source_id, target_id), (target_id, source_id)):
        flips = _read_csv_if_exists(step9e_dir(src_id, tgt_id) / "relationship_direction_flips.csv")
        for feature in NUMERIC_FEATURES:
            record: dict[str, Any] = {
                "feature": feature,
                "step9e_direction": f"{src_id}__{tgt_id}",
                "step9e_available": flips is not None,
            }
            if flips is not None and "population" in flips.columns:
                sub = flips[(flips["feature"] == feature) & (flips["population"] == PRIMARY_POPULATION)]
                if not sub.empty:
                    r = sub.iloc[0]
                    for key in (
                        "mean_direction_flip", "median_direction_flip", "rank_effect_direction_flip",
                        "raw_auc_below_0_5_in_one_region_only", "relationship_flip_score",
                        "source_raw_auc", "target_raw_auc",
                    ):
                        record[f"step9e_{key}"] = r[key] if key in sub.columns else None
            rows.append(record)
    return pd.DataFrame(rows)


def step9f_model_level_integration(
    source_id: str = SOURCE_ID, target_id: str = TARGET_ID,
) -> dict[str, Any]:
    """Step9F is a MODEL/representation-level ranking-reversal experiment, not
    a per-feature diagnostic. Surface its manifest/screening at the model
    level in a SEPARATE section; do not fabricate a per-feature join."""
    payload: dict[str, Any] = {
        "note": (
            "Step9F reports model/representation-level ranking reversal across "
            "feature variants, NOT per-feature univariate direction. It is "
            "presented at the model level only and is never joined per-feature."
        ),
        "directions": {},
    }
    for src_id, tgt_id in ((source_id, target_id), (target_id, source_id)):
        key = f"{src_id}__{tgt_id}"
        manifest = _read_json_if_exists(step9f_dir(src_id, tgt_id) / "step9f_experiment_manifest.json")
        screening = _read_csv_if_exists(step9f_dir(src_id, tgt_id) / "exploratory_candidate_screening.csv")
        payload["directions"][key] = {
            "step9f_available": manifest is not None or screening is not None,
            "manifest_analysis_id": (manifest or {}).get("analysis_id") if manifest else None,
            "screening_variant_count": int(len(screening)) if screening is not None else None,
        }
    return payload


def step10_transfer_summary(
    source_id: str = SOURCE_ID, target_id: str = TARGET_ID,
) -> dict[str, Any]:
    """Read frozen Step10 final report(s) for the integrated interpretation
    (raw vs adapted transfer discrimination). Read-only; fields surfaced only
    if present."""
    summary: dict[str, Any] = {"directions": {}}
    for src_id, tgt_id in ((source_id, target_id), (target_id, source_id)):
        key = f"{src_id}__{tgt_id}"
        report = _read_json_if_exists(step10_dir(src_id, tgt_id) / "step10_final_report.json")
        summary["directions"][key] = {
            "step10_available": report is not None,
            "analysis_id": (report or {}).get("analysis_id") if report else None,
        }
    return summary


# =============================================================================
# Preregistration / analysis_id
# =============================================================================
def scientific_configuration(
    protected: dict[str, Any], source_id: str = SOURCE_ID,
    target_id: str = TARGET_ID, output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    legacy_pair = (source_id, target_id) == (SOURCE_ID, TARGET_ID)
    secondary_contrast = (
        "auc_difference_bejis_minus_manavgat with independent regional draws paired by replicate index post hoc; non-zero contrast alone is NOT a reversal"
        if legacy_pair else
        f"auc_difference_{target_id}_minus_{source_id} with independent regional draws paired by replicate index post hoc; non-zero contrast alone is NOT a reversal"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_ids": [source_id, target_id],
        "primary_population": PRIMARY_POPULATION,
        "target": TARGET_COLUMN,
        "numeric_features_in_order": list(NUMERIC_FEATURES),
        "landcover_handling": {"column": LANDCOVER_COLUMN, "excluded_from_numeric_auc": True, "reason": LANDCOVER_EXCLUSION_REASON},
        "complete_case_policy": "feature-specific complete cases; no imputation; no standardization; no inversion; no winsorization",
        "raw_auc_definition": "roc_auc_score(burned, raw_feature_value); burned is positive class; never max(AUC,1-AUC)",
        "signed_rank_effect_definition": "2*raw_univariate_auc - 1",
        "direction_labels": [DIRECTION_HIGHER, DIRECTION_LOWER, DIRECTION_CHANCE, DIRECTION_UNAVAILABLE],
        "block_construction": "block_row=floor(row_500m/10); block_col=floor(col_500m/10)",
        "block_size_cells": BLOCK_SIZE_CELLS,
        "block_origin": list(BLOCK_ORIGIN),
        "block_assigned_before_filtering": True,
        "nominal_block_scale": NOMINAL_BLOCK_SCALE,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
            "method": "spatial blocks sampled with replacement; all rows per sampled block; multiplicity preserved",
            "ci": f"{CI_LOWER_PCT}/{CI_UPPER_PCT} percentile",
            "invalid_replicate": "sampled complete-case rows contain only one target class",
            "min_valid_replicates": MIN_VALID_REPLICATES,
            "second_block_scale": "prohibited",
        },
        "region_support_statuses": [
            "bootstrap_supported_higher_values_rank_burned",
            "bootstrap_supported_lower_values_rank_burned",
            "interval_includes_chance",
            "unstable_bootstrap",
        ],
        "reversal_statuses": [
            "bootstrap_supported_direction_reversal",
            "point_direction_reversal_interval_uncertain",
            "no_direction_reversal",
            "unavailable",
        ],
        "secondary_contrast": secondary_contrast,
        "step9e_step9f_integration_policy": "read-only; surface only existing fields; Step9F is model-level only; never fabricate per-feature joins",
        "output_namespace": str(output_root),
        "frozen_input_hashes": protected["step8a_inputs"],
        "frozen_reference_file_counts": {
            "step9e": {k: len(v) for k, v in protected["step9e_trees"].items()},
            "step9f": {k: len(v) for k, v in protected["step9f_trees"].items()},
            "step10": {k: len(v) for k, v in protected["step10_trees"].items()},
        },
        "package_versions": _package_versions(),
        "git_commit": _git_commit(),
        "prohibited_actions": [
            "feature_tuning_or_transformation_after_results", "auc_inversion", "prediction_inversion",
            "target_label_adaptation", "second_block_scale", "population_mixing_in_primary_table",
            "hardcoding_or_testing_against_prototype_values",
        ],
    }


def build_manifest(
    protected: dict[str, Any], source_id: str = SOURCE_ID,
    target_id: str = TARGET_ID, output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    config = scientific_configuration(protected, source_id, target_id, output_root)
    analysis_id = sha256_bytes(canonical_json(config).encode("utf-8"))
    return {
        "analysis_id": analysis_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scientific_configuration": config,
    }


def _preregistration_md(manifest: dict[str, Any]) -> str:
    config = manifest["scientific_configuration"]
    lines = [
        "# Step9G Univariate Feature-AUC Direction-Reversal Preregistration -- IMMUTABLE",
        "",
        f"- analysis_id: `{manifest['analysis_id']}`",
        f"- created_at: {manifest['created_at']}",
        f"- experiments: {config['experiment_ids']}",
        f"- primary population: {PRIMARY_POPULATION}",
        f"- target: {TARGET_COLUMN}",
        f"- numeric features (fixed order): {list(NUMERIC_FEATURES)}",
        f"- block size: {BLOCK_SIZE_CELLS} cells ({NOMINAL_BLOCK_SCALE}), origin {tuple(BLOCK_ORIGIN)}",
        f"- bootstrap: {BOOTSTRAP_REPLICATES} replicates, seed {BOOTSTRAP_SEED}, "
        f"{CI_LOWER_PCT}/{CI_UPPER_PCT} percentile CI, min valid {MIN_VALID_REPLICATES}",
        "",
        "Raw feature values are passed directly to ROC-AUC; AUC below 0.5 is a "
        "direction, never inverted or relabeled as poor performance. No feature "
        "list, population, block scale, or status rule changes after results.",
    ]
    return "\n".join(lines) + "\n"


def validate_or_write_preregistration(
    output_root: Path, protected: dict[str, Any], force: bool = False,
    source_id: str = SOURCE_ID, target_id: str = TARGET_ID,
) -> dict[str, Any]:
    json_path = output_root / "step9g_preregistration.json"
    md_path = output_root / "step9g_preregistration.md"
    expected = scientific_configuration(protected, source_id, target_id, output_root)
    expected_id = sha256_bytes(canonical_json(expected).encode("utf-8"))
    if json_path.exists():
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        if existing.get("analysis_id") != expected_id or existing.get("scientific_configuration") != expected:
            raise Step9GError("Existing immutable Step9G preregistration disagrees with runtime configuration; refusing to silently rewrite.")
        if not md_path.is_file() or md_path.read_text(encoding="utf-8") != _preregistration_md(existing):
            raise Step9GError("Existing Step9G Markdown preregistration is missing or changed.")
        return existing
    if md_path.exists():
        raise Step9GError("Step9G Markdown preregistration exists without its JSON manifest.")
    output_root.mkdir(parents=True, exist_ok=True)
    _assert_output_namespace_isolated(json_path, output_root)
    manifest = build_manifest(protected, source_id, target_id, output_root)
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    md_path.write_text(_preregistration_md(manifest), encoding="utf-8")
    return manifest


# =============================================================================
# Plot
# =============================================================================
def _region_output_key(experiment_id: str, source_id: str, target_id: str) -> str:
    if (source_id, target_id) == (SOURCE_ID, TARGET_ID):
        return "manavgat" if experiment_id == source_id else "bejis"
    return experiment_id


def make_direction_plot(
    reversal_rows: list[dict[str, Any]], output_root: Path,
    source_id: str = SOURCE_ID, target_id: str = TARGET_ID,
) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = [r["feature"] for r in reversal_rows]
    y_pos = np.arange(len(features))[::-1]  # frozen order top-to-bottom
    fig, ax = plt.subplots(figsize=(9, 6))
    source_key = _region_output_key(source_id, source_id, target_id)
    target_key = _region_output_key(target_id, source_id, target_id)
    for offset, region, label, color in (
        (0.12, source_key, source_id, "#1f77b4"),
        (-0.12, target_key, target_id, "#d62728"),
    ):
        aucs = [r[f"{region}_auc"] for r in reversal_rows]
        los = [r[f"{region}_ci_low"] for r in reversal_rows]
        his = [r[f"{region}_ci_high"] for r in reversal_rows]
        for i, (yp, auc, lo, hi) in enumerate(zip(y_pos, aucs, los, his)):
            if auc is None:
                continue
            if lo is not None and hi is not None:
                ax.plot([lo, hi], [yp + offset, yp + offset], color=color, alpha=0.6, lw=2)
            ax.plot(auc, yp + offset, "o", color=color, label=label if i == 0 else None)
    for i, r in enumerate(reversal_rows):
        if r["point_direction_reversal"]:
            ax.annotate("reversal", (0.5, y_pos[i]), color="black", fontsize=8, ha="center", va="bottom")
    ax.axvline(0.5, color="gray", linestyle="--", lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel("Raw univariate ROC-AUC (burned as positive; not inverted)")
    ax.set_title(f"Step9G univariate feature-AUC direction ({PRIMARY_POPULATION}); {BLOCK_SIZE_CELLS}-cell block bootstrap CI")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = output_root / "step9g_auc_direction_plot.png"
    _assert_output_namespace_isolated(path, output_root)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# =============================================================================
# Dry run
# =============================================================================
def dry_run(
    source_id: str = SOURCE_ID, target_id: str = TARGET_ID,
    output_root: Path | None = None,
) -> dict[str, Any]:
    _pair_ids(source_id, target_id)
    output_root = output_root_for(source_id, target_id) if output_root is None else output_root
    feature_contract = validate_feature_contracts(source_id, target_id)
    protected = protected_paths(source_id, target_id)
    experiment_ids = (source_id, target_id)
    return {
        "mode": "dry_run",
        "computes_auc": False, "runs_bootstrap": False, "writes_files": False,
        "experiment_ids": list(experiment_ids),
        "primary_population": PRIMARY_POPULATION,
        "numeric_features_in_order": list(NUMERIC_FEATURES),
        "block_size_cells": BLOCK_SIZE_CELLS,
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "resolved_step8a_inputs": {e: str(resolve_step8a_dataset(e)) for e in experiment_ids},
        "feature_contract": feature_contract,
        "frozen_step9e_refs": {f"{s}__{t}": str(step9e_dir(s, t)) for s, t in ((source_id, target_id), (target_id, source_id))},
        "frozen_step9f_refs": {f"{s}__{t}": str(step9f_dir(s, t)) for s, t in ((source_id, target_id), (target_id, source_id))},
        "frozen_step10_refs": {f"{s}__{t}": str(step10_dir(s, t)) for s, t in ((source_id, target_id), (target_id, source_id))},
        "output_namespace": str(output_root),
        "protected_step8a_input_count": len(protected["step8a_inputs"]),
    }


CLAIM_BOUNDARY_TEXT = (
    "Raw cross-region transfer was below chance; unsupervised adaptation "
    "recovered part of the discrimination loss; a large gap to within-region "
    "performance remained. Feature-level univariate AUC direction reversals "
    "indicate that marginal feature-label relationships are not stable across "
    "regions, a pattern consistent with residual concept/relationship shift. "
    "This does NOT prove causality and does NOT establish that concept shift "
    "is the only source of transfer failure; AUC below 0.5 is a direction, "
    "not poor performance, and is never inverted."
)


# =============================================================================
# Main orchestration
# =============================================================================
def run_analysis(
    source_id: str = SOURCE_ID, target_id: str = TARGET_ID,
    dry: bool = False, force: bool = False, output_root: Path | None = None,
) -> dict[str, Any]:
    _pair_ids(source_id, target_id)
    output_root = output_root_for(source_id, target_id) if output_root is None else output_root
    if dry:
        return dry_run(source_id, target_id, output_root)

    final_report_path = output_root / "step9g_final_report.json"
    if final_report_path.is_file() and not force:
        existing = json.loads(final_report_path.read_text(encoding="utf-8"))
        return {
            "ran": False,
            "reason": "pair_outputs_already_exist_use_force",
            "analysis_id": existing.get("analysis_id"),
            "source_experiment_id": source_id,
            "target_experiment_id": target_id,
            "output_root": str(output_root),
        }

    feature_contract = validate_feature_contracts(source_id, target_id)
    before = protected_paths(source_id, target_id)
    manifest = validate_or_write_preregistration(
        output_root, before, force, source_id, target_id,
    )
    analysis_id = manifest["analysis_id"]
    experiment_ids = (source_id, target_id)

    # --- load + population audit ---
    populations: dict[str, pd.DataFrame] = {}
    input_audit: dict[str, Any] = {
        "analysis_id": analysis_id, "regions": {},
        "feature_contract": feature_contract,
    }
    for experiment in experiment_ids:
        pop = assign_blocks_then_filter(load_step8a(experiment), experiment)
        populations[experiment] = pop
        input_audit["regions"][experiment] = validate_population(pop, experiment)

    # --- per-region per-feature stats + bootstrap ---
    auc_rows: list[dict[str, Any]] = []
    replicate_records: list[dict[str, Any]] = []
    boot_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for experiment in experiment_ids:
        pop = populations[experiment]
        for feature in NUMERIC_FEATURES:
            stats = univariate_feature_stats(pop, feature)
            boot = _block_bootstrap_auc(pop, feature)
            ci_low, ci_high = _percentile_ci(boot["replicate_aucs"])
            boot["ci_low"], boot["ci_high"] = ci_low, ci_high
            boot_cache[(experiment, feature)] = boot
            support = _support_status(ci_low, ci_high, boot["stable"])
            auc_rows.append({
                "experiment_id": experiment, **{k: v for k, v in stats.items() if k != "available"},
                "auc_ci_low": ci_low, "auc_ci_high": ci_high,
                "valid_replicates": boot["valid"], "invalid_replicates": boot["invalid"],
                "bootstrap_stable": boot["stable"], "support_status": support,
            })
            for r_idx, auc in enumerate(boot["replicate_aucs"]):
                replicate_records.append({
                    "experiment_id": experiment, "feature": feature,
                    "replicate": r_idx, "auc": auc,
                })

    # --- direction reversal table (per feature, across regions) ---
    reversal_rows: list[dict[str, Any]] = []
    step9e_join = step9e_feature_integration(source_id, target_id)
    pair_token = f"{source_id}__{target_id}"
    source_key = _region_output_key(source_id, source_id, target_id)
    target_key = _region_output_key(target_id, source_id, target_id)
    contrast_key = (
        "auc_difference_bejis_minus_manavgat"
        if (source_id, target_id) == (SOURCE_ID, TARGET_ID)
        else "auc_difference_target_minus_source"
    )
    claim_boundary = (
        CLAIM_BOUNDARY_TEXT
        if (source_id, target_id) == (SOURCE_ID, TARGET_ID)
        else (
            "Feature-level univariate AUC direction reversals indicate that marginal "
            "feature-label relationships are not stable across the selected regions, "
            "a pattern consistent with concept/relationship shift. Target labels are "
            "used only for this diagnostic evaluation. This does NOT prove causality "
            "or establish concept shift as the only transfer-failure mechanism; AUC "
            "below 0.5 is a direction and is never inverted."
        )
    )
    for feature in NUMERIC_FEATURES:
        source_boot = boot_cache[(source_id, feature)]
        target_boot = boot_cache[(target_id, feature)]
        status = _reversal_status(source_boot, target_boot)
        diff, diff_lo, diff_hi = _contrast_ci(
            source_boot["replicate_aucs"], target_boot["replicate_aucs"],
        )
        e_flag = None
        sub = step9e_join[(step9e_join["feature"] == feature) & (step9e_join["step9e_direction"] == pair_token)]
        if not sub.empty and "step9e_rank_effect_direction_flip" in sub.columns:
            e_flag = sub.iloc[0].get("step9e_rank_effect_direction_flip")
        reversal_rows.append({
            "feature": feature,
            "source_experiment_id": source_id, "target_experiment_id": target_id,
            f"{source_key}_auc": source_boot["point_auc"], f"{source_key}_ci_low": source_boot["ci_low"], f"{source_key}_ci_high": source_boot["ci_high"],
            f"{source_key}_direction": _direction_label(source_boot["point_auc"]),
            f"{source_key}_support_status": _support_status(source_boot["ci_low"], source_boot["ci_high"], source_boot["stable"]),
            f"{target_key}_auc": target_boot["point_auc"], f"{target_key}_ci_low": target_boot["ci_low"], f"{target_key}_ci_high": target_boot["ci_high"],
            f"{target_key}_direction": _direction_label(target_boot["point_auc"]),
            f"{target_key}_support_status": _support_status(target_boot["ci_low"], target_boot["ci_high"], target_boot["stable"]),
            contrast_key: diff,
            "auc_difference_ci_low": diff_lo, "auc_difference_ci_high": diff_hi,
            "point_direction_reversal": _point_direction_reversal(source_boot["point_auc"], target_boot["point_auc"]),
            "reversal_status": status,
            "step9e_relationship_direction_flag": e_flag,
            "integrated_interpretation": claim_boundary,
        })

    # --- landcover descriptive ---
    landcover_df = pd.concat(
        [landcover_descriptive(populations[e], e) for e in experiment_ids], ignore_index=True
    ) if all(LANDCOVER_COLUMN in populations[e].columns for e in experiment_ids) else pd.DataFrame()

    # --- write outputs (namespace-isolated) ---
    output_root.mkdir(parents=True, exist_ok=True)

    def _write(name: str, writer) -> Path:
        path = output_root / name
        _assert_output_namespace_isolated(path, output_root)
        writer(path)
        return path

    auc_df = pd.DataFrame(auc_rows)
    reversal_df = pd.DataFrame(reversal_rows)
    replicate_df = pd.DataFrame(replicate_records)

    _write("step9g_input_audit.json", lambda p: p.write_text(json.dumps(input_audit, indent=2, default=str) + "\n"))
    _write("step9g_univariate_auc_by_region.csv", lambda p: auc_df.to_csv(p, index=False))
    _write("step9g_direction_reversal_table.csv", lambda p: reversal_df.to_csv(p, index=False))
    _write("step9g_bootstrap_replicates.parquet", lambda p: replicate_df.to_parquet(p, index=False))
    _write("step9g_step9e_feature_integration.csv", lambda p: step9e_join.to_csv(p, index=False))
    step9f_payload = step9f_model_level_integration(source_id, target_id)
    _write("step9g_step9f_model_level_integration.json", lambda p: p.write_text(json.dumps(step9f_payload, indent=2, default=str) + "\n"))
    if not landcover_df.empty:
        _write("step9g_landcover_descriptive.csv", lambda p: landcover_df.to_csv(p, index=False))

    plot_path = make_direction_plot(reversal_rows, output_root, source_id, target_id)

    supported = [r["feature"] for r in reversal_rows if r["reversal_status"] == "bootstrap_supported_direction_reversal"]
    uncertain = [r["feature"] for r in reversal_rows if r["reversal_status"] == "point_direction_reversal_interval_uncertain"]
    same_dir = [r["feature"] for r in reversal_rows if r["reversal_status"] == "no_direction_reversal"]

    final_report = {
        "analysis_id": analysis_id,
        "schema_version": SCHEMA_VERSION,
        "source_experiment_id": source_id,
        "target_experiment_id": target_id,
        "primary_population": PRIMARY_POPULATION,
        "input_audit": input_audit["regions"],
        "direction_reversal_table": reversal_rows,
        "bootstrap_supported_direction_reversals": supported,
        "point_reversals_interval_uncertain": uncertain,
        "same_direction_features": same_dir,
        "landcover_exclusion_reason": LANDCOVER_EXCLUSION_REASON,
        "step9f_model_level_integration": step9f_payload,
        "step10_transfer_summary": step10_transfer_summary(source_id, target_id),
        "answers": {
            "which_features_reverse": [r["feature"] for r in reversal_rows if r["point_direction_reversal"]],
            "which_reversals_bootstrap_supported": supported,
            "which_features_same_direction": same_dir,
            "thermal_features_consistent_with_step9e": [
                r["feature"] for r in reversal_rows
                if r["step9e_relationship_direction_flag"] and r["point_direction_reversal"]
            ],
        },
        "claim_boundary": claim_boundary,
    }
    _write("step9g_final_report.json", lambda p: p.write_text(json.dumps(final_report, indent=2, default=str) + "\n"))
    _write("step9g_final_report.md", lambda p: p.write_text(
        _final_report_md(final_report, reversal_rows, source_id, target_id), encoding="utf-8",
    ))

    after = protected_paths(source_id, target_id)
    assert_protected_unchanged(before, after)

    return {
        "ran": True, "analysis_id": analysis_id,
        "input_audit": input_audit["regions"],
        "bootstrap_supported_direction_reversals": supported,
        "point_reversals_interval_uncertain": uncertain,
        "same_direction_features": same_dir,
        "protected_hash_check": "passed",
        "plot_path": str(plot_path) if plot_path else None,
        "output_root": str(output_root),
    }


def _final_report_md(
    report: dict[str, Any], reversal_rows: list[dict[str, Any]],
    source_id: str = SOURCE_ID, target_id: str = TARGET_ID,
) -> str:
    def _fmt(v):
        return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
    source_key = _region_output_key(source_id, source_id, target_id)
    target_key = _region_output_key(target_id, source_id, target_id)
    header = [
        "feature", f"{source_key}_auc", f"{source_key}_ci", f"{source_key}_direction",
        f"{target_key}_auc", f"{target_key}_ci", f"{target_key}_direction",
        "direction_reversal_status", "step9_diagnostic_concordance",
    ]
    lines = [
        "# Step9G Univariate Feature-AUC Direction-Reversal -- Final Report",
        "",
        f"- analysis_id: `{report['analysis_id']}`",
        f"- source: `{source_id}`",
        f"- target: `{target_id}`",
        f"- primary population: {PRIMARY_POPULATION}",
        f"- protected frozen inputs/references: unchanged",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for r in reversal_rows:
        lines.append("| " + " | ".join([
            r["feature"], _fmt(r[f"{source_key}_auc"]),
            f"[{_fmt(r[f'{source_key}_ci_low'])}, {_fmt(r[f'{source_key}_ci_high'])}]", r[f"{source_key}_direction"],
            _fmt(r[f"{target_key}_auc"]),
            f"[{_fmt(r[f'{target_key}_ci_low'])}, {_fmt(r[f'{target_key}_ci_high'])}]", r[f"{target_key}_direction"],
            r["reversal_status"], _fmt(r["step9e_relationship_direction_flag"]),
        ]) + " |")
    lines += [
        "",
        f"## Which features reverse direction between {source_id} and {target_id}?",
        f"{report['answers']['which_features_reverse'] or 'none'}",
        "",
        "## Which reversals are bootstrap-supported?",
        f"{report['bootstrap_supported_direction_reversals'] or 'none'}",
        "",
        "## Which features retain the same direction?",
        f"{report['same_direction_features'] or 'none'}",
        "",
        "## Point reversals with uncertain intervals",
        f"{report['point_reversals_interval_uncertain'] or 'none'}",
        "",
        "## Claim boundary",
        "",
        report["claim_boundary"],
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step9G univariate feature-AUC direction-reversal diagnostic.")
    parser.add_argument("--source", required=True, help="Source experiment ID.")
    parser.add_argument("--target", required=True, help="Target experiment ID.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_analysis(
        source_id=args.source, target_id=args.target,
        dry=args.dry_run, force=args.force,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
