"""
step8d_thermal_feature_ablation.py

Step8D: THERMAL FEATURE ABLATION for Step8B's baseline-vs-thermal burned-area
modeling.

Core question:
    Which thermal/dryness feature (or feature group) drives Step8B's
    baseline -> thermal improvement?

For each population, trains baseline + 10 "baseline + thermal-group" models
(11 models total) under the SAME spatial-block CV folds, then ranks the
thermal groups by how much they improve over baseline (primarily by
delta_pr_auc, since burned labels are rare; delta_auc is a secondary
ranking).

IMPORTANT CONSTRAINTS (do not violate):
    - Step8D TRAINS multiple ablation models. It does NOT touch Step5/
      Step5C/Step6/Step7B-E/Step8A/Step8B/Step8C logic.
    - MCD64A1 (`burned`) is the ONLY target. FIRMS is NEVER used.
    - Samples are Step8A's 500 m grid cells, NEVER 30 m pixels.
    - Cross-validation is the SAME spatial-block strategy as Step8B
      (StratifiedGroupKFold over 500 m cell blocks). Random/row-wise
      splitting is never used.
    - No label-derived or provenance column is used as a feature (see
      FORBIDDEN_FEATURE_COLUMNS). lon/lat are only used to build spatial
      blocks, never as model features.
    - By default this produces POINT ESTIMATES ONLY. Spatial-block bootstrap
      CIs are opt-in via --bootstrap (only for the top-K ablation groups per
      population, to keep runtime bounded).

Input (read-only):
    outputs/step8a/step8a_500m_modeling_dataset.parquet (preferred)
    outputs/step8a/step8a_500m_modeling_dataset.csv (fallback)
    outputs/step8b/step8b_model_comparison_metrics.json (optional, only to
        cross-check that this script's "all_thermal" ablation result is
        consistent with Step8B's own baseline-vs-thermal result)

Output:
    outputs/step8d/step8d_ablation_metrics.json
    outputs/step8d/step8d_ablation_fold_metrics.csv
    outputs/step8d/step8d_ablation_predictions.parquet
    outputs/step8d/step8d_ablation_predictions.csv
    outputs/step8d/step8d_ablation_feature_importance.csv
    outputs/step8d/step8d_ablation_summary.md
    outputs/step8d/step8d_ablation_barplot.png                  (optional)
    outputs/step8d/step8d_ablation_delta_auc_by_population.csv   (optional)
    outputs/step8d/step8d_ablation_delta_pr_auc_by_population.csv (optional)

CLI:
    python src/step8d_thermal_feature_ablation.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings as pywarnings
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from core.config import (
    STEP8D_OUTPUT_DIR,
    STEP8D_INPUT_DATASET,
    STEP8D_RANDOM_SEED,
    STEP8D_N_SPLITS,
    STEP8D_SPATIAL_BLOCK_SIZE_CELLS,
    STEP8D_MIN_POSITIVES_PER_POPULATION,
    STEP8D_MIN_MONTH_POSITIVES,
    STEP8D_N_ESTIMATORS,
    STEP8D_BOOTSTRAP_DEFAULT,
    STEP8D_BOOTSTRAP_N,
    STEP8D_BOOTSTRAP_TOP_K,
)
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT

BASE_DIR = PROJECT_ROOT
log, log_file = setup_logger("step8d")

pywarnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class Step8DError(SystemExit):
    """Fail-fast error for Step8D (extends SystemExit like other steps)."""


# =============================================================================
# Target / feature registry
# =============================================================================
TARGET_COLUMN = "burned"
LC_CROPLAND = 40  # ESA WorldCover cropland code (must match Step8A's mapping)

BASELINE_FEATURES = ["ndvi_mean", "elevation_mean", "slope_mean", "landcover_dominant"]
CATEGORICAL_FEATURES = ["landcover_dominant"]

# Thermal ablation groups: name -> additional feature list (always appended
# to BASELINE_FEATURES). "baseline" itself is handled separately (no
# additional features).
THERMAL_GROUPS: dict[str, list[str]] = {
    "lst_anomaly_only": ["lst_anomaly_mean"],
    "current_lst_only": ["current_lst_mean"],
    "tvdi_only": ["current_tvdi_mean"],
    "tvdi_difference_only": ["tvdi_difference_mean"],
    "downscaled_only": ["downscaled_lst_mean"],
    "fused_lst_only": ["fused_lst_mean"],
    "lst_anomaly_group": ["lst_anomaly_mean", "current_lst_mean"],
    "tvdi_group": ["current_tvdi_mean", "tvdi_difference_mean"],
    "fused_downscaled_group": ["downscaled_lst_mean", "fused_lst_mean"],
    "all_thermal": [
        "lst_anomaly_mean", "current_lst_mean", "current_tvdi_mean",
        "tvdi_difference_mean", "downscaled_lst_mean", "fused_lst_mean",
    ],
}
ALL_THERMAL_FEATURES = THERMAL_GROUPS["all_thermal"]
MODEL_NAMES = ["baseline"] + list(THERMAL_GROUPS.keys())  # 11 models total

# Columns that must NEVER be used as model features (label/metadata/leakage
# and source-provenance columns; provenance is diagnostics-only). lon/lat are
# only used to (re)derive spatial_block_id, never as features.
FORBIDDEN_FEATURE_COLUMNS = [
    "burned", "burn_date", "burn_month", "burn_day_of_year", "label_source",
    "cell_id", "row_500m", "col_500m", "valid_for_modeling", "invalid_reason",
    "burn_date_pixel_agreement_fraction", "out_of_window_burndate",
    "lon", "lat", "spatial_block_id", "fold_id", "source_mask_majority",
    "observed_fraction", "gapfilled_fraction", "invalid_source_fraction",
    "valid_30m_fraction",
]

REQUIRED_COLUMNS = list(dict.fromkeys(
    BASELINE_FEATURES + ALL_THERMAL_FEATURES + [
        TARGET_COLUMN, "row_500m", "col_500m", "cell_id", "valid_for_modeling",
        "burn_month", "burnable_tree_shrub_grass", "burnable_tree_shrub",
        "landcover_cropland_fraction",
    ]
))

POPULATION_ORDER = ["all_valid", "cropland_dominant", "burnable_tree_shrub_grass", "burnable_tree_shrub"]
DIAGNOSTIC_ONLY_POPULATIONS = {"burnable_tree_shrub_grass", "burnable_tree_shrub"}
CORE_POPULATIONS = {"all_valid", "cropland_dominant"}

MONTHS = [8, 9, 10]
MONTH_NAMES = {8: "august", 9: "september", 10: "october"}

STEP8B_METRICS_PATH_REL = "outputs/step8b/step8b_model_comparison_metrics.json"
ALL_THERMAL_VS_STEP8B_TOLERANCE = 0.01


# =============================================================================
# Data loading + validation
# =============================================================================
def load_dataset(input_arg: str | None) -> tuple[pd.DataFrame, Path]:
    parquet_path = BASE_DIR / (input_arg or STEP8D_INPUT_DATASET)
    csv_path = BASE_DIR / "outputs" / "step8a" / "step8a_500m_modeling_dataset.csv"

    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            log.info("Step8A veri seti okundu (parquet): %s (%d satir)", parquet_path, len(df))
            return df, parquet_path
        except Exception as exc:  # noqa: BLE001
            log.warning("Parquet okunamadi (%s: %s); CSV fallback deneniyor.", type(exc).__name__, exc)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        log.info("Step8A veri seti okundu (CSV fallback): %s (%d satir)", csv_path, len(df))
        return df, csv_path

    raise Step8DError(
        "Step8A modelleme veri seti bulunamadi. Beklenen: "
        f"{parquet_path} veya {csv_path}. Once Step8A'yi calistirin: "
        "python src/step8a_prepare_500m_modeling_dataset.py --force"
    )


def validate_input(df: pd.DataFrame) -> list[str]:
    warnings_out: list[str] = []

    if TARGET_COLUMN not in df.columns:
        raise Step8DError(f"Girdi veri setinde '{TARGET_COLUMN}' kolonu yok.")

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise Step8DError(
            f"Girdi veri setinde beklenen kolonlar eksik: {missing_required}. "
            "Step8A'yi guncel scriptle yeniden calistirin."
        )

    if "valid_for_modeling" not in df.columns:
        raise Step8DError("valid_for_modeling kolonu yok.")

    df_valid = df[df["valid_for_modeling"] == True]  # noqa: E712
    if len(df_valid) == 0:
        raise Step8DError("valid_for_modeling==True olan hic satir yok.")

    burned_vals = df_valid[TARGET_COLUMN].dropna().unique()
    if len(burned_vals) < 2:
        raise Step8DError(
            f"'{TARGET_COLUMN}' hedefi valid_for_modeling==True populasyonunda "
            f"tek sinif iceriyor ({burned_vals})."
        )

    for feat in ALL_THERMAL_FEATURES:
        if feat not in df.columns:
            continue
        miss_frac = df_valid[feat].isna().mean()
        if miss_frac > 0.3:
            warnings_out.append(
                f"{feat}: valid_for_modeling nufusunda %{miss_frac*100:.1f} "
                "eksik deger (yuksek eksiklik)."
            )

    return warnings_out


def check_no_forbidden_features(feature_list: list[str]) -> None:
    leaked = set(feature_list).intersection(FORBIDDEN_FEATURE_COLUMNS)
    if leaked:
        raise Step8DError(
            f"YASAK etiket/metadata kolonlari ozellik setine sizmis: {leaked}."
        )


# =============================================================================
# Spatial blocks + CV (identical strategy to Step8B)
# =============================================================================
def add_spatial_block_id(df: pd.DataFrame, block_size_cells: int) -> pd.DataFrame:
    df = df.copy()
    block_size_cells = max(int(block_size_cells), 1)
    r_block = (df["row_500m"].astype(int) // block_size_cells).astype(int)
    c_block = (df["col_500m"].astype(int) // block_size_cells).astype(int)
    df["spatial_block_id"] = r_block.astype(str) + "_" + c_block.astype(str)
    return df


def make_spatial_folds(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits_requested: int,
    random_state: int,
    min_positive_folds: int = 2,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """
    Builds spatial-block-grouped, stratified CV folds. NEVER falls back to a
    random (non-grouped) split. Tries n_splits_requested, then 3; fails
    clearly if neither works.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))

    def _try(n_splits: int):
        if n_splits < 2 or n_groups < n_splits:
            return None
        try:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            folds = list(splitter.split(np.zeros(len(y)), y, groups=groups))
        except Exception as exc:  # noqa: BLE001
            log.warning("StratifiedGroupKFold(n_splits=%d) basarisiz: %s", n_splits, exc)
            return None
        folds_with_positive = sum(1 for _, test_idx in folds if y[test_idx].sum() > 0)
        if folds_with_positive < min_positive_folds:
            log.warning(
                "n_splits=%d ile yalnizca %d fold'da pozitif ornek var (gerekli >=%d).",
                n_splits, folds_with_positive, min_positive_folds,
            )
            return None
        return folds

    folds = _try(n_splits_requested)
    if folds is not None:
        return folds, n_splits_requested

    if n_splits_requested != 3:
        folds = _try(3)
        if folds is not None:
            log.warning("n_splits=%d gecerli fold uretemedi; 3 fold'a dusuruldu.", n_splits_requested)
            return folds, 3

    raise Step8DError(
        "Spatial-block CV gecerli fold uretemedi (StratifiedGroupKFold, "
        f"n_splits={n_splits_requested} ve 3 denendi). RANDOM SPLIT'e "
        "DUSULMEYECEK -- bu populasyon icin ablation atlanmali."
    )


# =============================================================================
# Model pipeline
# =============================================================================
def build_classifier(model_name: str, n_estimators: int, random_state: int):
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators, max_depth=None, min_samples_leaf=3,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=random_state, class_weight="balanced")
    raise Step8DError(f"Bilinmeyen model: {model_name}")


def build_pipeline(feature_list: list[str], model_choice: str, n_estimators: int, random_state: int) -> Pipeline:
    check_no_forbidden_features(feature_list)
    numeric_features = [f for f in feature_list if f not in CATEGORICAL_FEATURES]
    categorical_features = [f for f in feature_list if f in CATEGORICAL_FEATURES]

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers)
    clf = build_classifier(model_choice, n_estimators, random_state)
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])


def get_expanded_feature_names(pipeline: Pipeline, feature_list: list[str]) -> list[str]:
    try:
        return list(pipeline.named_steps["preprocess"].get_feature_names_out())
    except Exception:  # noqa: BLE001
        return list(feature_list)


def feature_list_for_model(model_name: str) -> list[str]:
    if model_name == "baseline":
        return list(BASELINE_FEATURES)
    return BASELINE_FEATURES + THERMAL_GROUPS[model_name]


# =============================================================================
# Metrics
# =============================================================================
def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    out: dict = {"positive_count": n_pos, "negative_count": n_neg}
    if n_pos == 0 or n_neg == 0:
        out.update({k: None for k in (
            "roc_auc", "pr_auc", "brier_score", "balanced_accuracy",
            "precision", "recall", "f1",
        )})
        return out
    y_pred = (y_prob >= 0.5).astype(int)
    out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    out["brier_score"] = float(brier_score_loss(y_true, y_prob))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    return out


def safe_delta(a: dict, b: dict, key_map: dict[str, str]) -> dict:
    """delta[out_key] = b[src_key] - a[src_key], or None if either is None."""
    out = {}
    for out_key, src_key in key_map.items():
        va, vb = a.get(src_key), b.get(src_key)
        out[out_key] = (vb - va) if (va is not None and vb is not None) else None
    return out


# =============================================================================
# Per-population, per-model training (shared CV folds across all 11 models)
# =============================================================================
def train_one_model(
    df_pop: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    model_name: str,
    model_choice: str,
    n_estimators: int,
    random_state: int,
) -> dict:
    feature_list = feature_list_for_model(model_name)
    y = df_pop[TARGET_COLUMN].astype(int).to_numpy()
    n_index = len(df_pop)
    oof_prob = np.full(n_index, np.nan)
    fold_id = np.full(n_index, -1, dtype=int)
    fold_rows: list[dict] = []

    for i, (train_idx, test_idx) in enumerate(folds):
        fold_id[test_idx] = i
        X_train, y_train = df_pop.iloc[train_idx], y[train_idx]
        X_test, y_test = df_pop.iloc[test_idx], y[test_idx]

        pipe = build_pipeline(feature_list, model_choice, n_estimators, random_state)
        pipe.fit(X_train[feature_list], y_train)
        prob = pipe.predict_proba(X_test[feature_list])[:, 1]
        oof_prob[test_idx] = prob

        m = compute_binary_metrics(y_test, prob)
        fold_rows.append({
            "model_name": model_name, "fold": i,
            "n_train": len(train_idx), "n_test": len(test_idx),
            "test_positives": int(y_test.sum()), "test_negatives": int((y_test == 0).sum()),
            "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "brier_score": m["brier_score"],
        })

    overall = compute_binary_metrics(y, oof_prob)

    # Monthly (OOF-based, no retraining).
    monthly = {}
    burn_month = df_pop["burn_month"].to_numpy()
    for month in MONTHS:
        pos_mask = (y == 1) & (burn_month == month)
        neg_mask = (y == 0)
        eval_mask = pos_mask | neg_mask
        n_pos_month = int(pos_mask.sum())
        if n_pos_month < STEP8D_MIN_MONTH_POSITIVES:
            monthly[MONTH_NAMES[month]] = {
                "positive_count": n_pos_month, "negative_count": int(neg_mask.sum()),
                "roc_auc": None, "pr_auc": None,
                "warning": f"positives ({n_pos_month}) < min_month_positives ({STEP8D_MIN_MONTH_POSITIVES})",
            }
        else:
            mm = compute_binary_metrics(y[eval_mask], oof_prob[eval_mask])
            monthly[MONTH_NAMES[month]] = {
                "positive_count": n_pos_month, "negative_count": int(neg_mask.sum()),
                "roc_auc": mm["roc_auc"], "pr_auc": mm["pr_auc"],
            }

    # Final refit on the WHOLE population for feature importance.
    final_pipe = build_pipeline(feature_list, model_choice, n_estimators, random_state)
    final_pipe.fit(df_pop[feature_list], y)
    names = get_expanded_feature_names(final_pipe, feature_list)
    importances = getattr(final_pipe.named_steps["clf"], "feature_importances_", None)
    feature_importance_rows = []
    if importances is not None:
        for name, imp in zip(names, importances):
            feature_importance_rows.append({"model_name": model_name, "feature": name, "importance": float(imp)})

    return {
        "model_name": model_name,
        "feature_list": feature_list,
        "overall": overall,
        "monthly": monthly,
        "fold_rows": fold_rows,
        "oof_prob": oof_prob,
        "fold_id": fold_id,
        "feature_importance_rows": feature_importance_rows,
    }


def train_population_ablation(
    df_pop: pd.DataFrame,
    population_name: str,
    n_splits: int,
    random_state: int,
    model_choice: str,
    n_estimators: int,
    min_positives: int,
) -> dict:
    y = df_pop[TARGET_COLUMN].astype(int).to_numpy()
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    if n_pos < min_positives or n_neg < min_positives:
        return {
            "skipped": True,
            "reason": f"insufficient_positives_or_negatives (positives={n_pos}, negatives={n_neg}, min_required={min_positives})",
        }

    groups = df_pop["spatial_block_id"].to_numpy()
    try:
        folds, n_splits_used = make_spatial_folds(y, groups, n_splits, random_state)
    except Step8DError as exc:
        return {"skipped": True, "reason": f"spatial_cv_failed: {exc}"}

    log.info("  %s: %d model egitiliyor (n=%d, burned=%d, folds=%d)...",
              population_name, len(MODEL_NAMES), len(df_pop), n_pos, n_splits_used)

    models: dict[str, dict] = {}
    for model_name in MODEL_NAMES:
        models[model_name] = train_one_model(df_pop, folds, model_name, model_choice, n_estimators, random_state)

    baseline_res = models["baseline"]

    # Ablation comparison: each thermal group vs baseline, SAME folds/population.
    ablation_deltas: dict[str, dict] = {}
    for group_name in THERMAL_GROUPS:
        res = models[group_name]
        delta = safe_delta(baseline_res["overall"], res["overall"], {
            "delta_auc": "roc_auc", "delta_pr_auc": "pr_auc", "delta_brier": "brier_score",
        })
        ablation_deltas[group_name] = {
            "auc_baseline": baseline_res["overall"]["roc_auc"],
            "auc_ablation": res["overall"]["roc_auc"],
            "delta_auc": delta["delta_auc"],
            "pr_auc_baseline": baseline_res["overall"]["pr_auc"],
            "pr_auc_ablation": res["overall"]["pr_auc"],
            "delta_pr_auc": delta["delta_pr_auc"],
            "brier_baseline": baseline_res["overall"]["brier_score"],
            "brier_ablation": res["overall"]["brier_score"],
            "delta_brier": delta["delta_brier"],
        }

    # Rankings (1 = best). Primary: delta_pr_auc desc. Secondary: delta_auc desc.
    pr_sorted = sorted(
        ablation_deltas.items(),
        key=lambda kv: (kv[1]["delta_pr_auc"] if kv[1]["delta_pr_auc"] is not None else -np.inf),
        reverse=True,
    )
    for rank, (group_name, _) in enumerate(pr_sorted, start=1):
        ablation_deltas[group_name]["rank_by_delta_pr_auc"] = rank
    auc_sorted = sorted(
        ablation_deltas.items(),
        key=lambda kv: (kv[1]["delta_auc"] if kv[1]["delta_auc"] is not None else -np.inf),
        reverse=True,
    )
    for rank, (group_name, _) in enumerate(auc_sorted, start=1):
        ablation_deltas[group_name]["rank_by_delta_auc"] = rank

    # Monthly ablation deltas (vs baseline, same month, OOF-based).
    monthly_ablation: dict[str, dict] = {}
    for group_name in THERMAL_GROUPS:
        res = models[group_name]
        monthly_ablation[group_name] = {}
        for month_name in MONTH_NAMES.values():
            base_m = baseline_res["monthly"][month_name]
            abl_m = res["monthly"][month_name]
            if base_m.get("roc_auc") is None or abl_m.get("roc_auc") is None:
                monthly_ablation[group_name][month_name] = {
                    "positive_count": abl_m["positive_count"],
                    "delta_auc": None, "delta_pr_auc": None,
                    "warning": abl_m.get("warning") or base_m.get("warning"),
                }
            else:
                monthly_ablation[group_name][month_name] = {
                    "positive_count": abl_m["positive_count"],
                    "delta_auc": abl_m["roc_auc"] - base_m["roc_auc"],
                    "delta_pr_auc": abl_m["pr_auc"] - base_m["pr_auc"],
                }

    return {
        "skipped": False,
        "n_splits_used": n_splits_used,
        "n_positives": n_pos, "n_negatives": n_neg,
        "models": models,
        "ablation_deltas": ablation_deltas,
        "monthly_ablation": monthly_ablation,
        "ranking_order": [g for g, _ in pr_sorted],
    }


# =============================================================================
# Optional spatial-block bootstrap (top-K groups only)
# =============================================================================
def spatial_block_bootstrap_delta(
    y: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    block_ids: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict:
    valid = pd.notna(block_ids)
    y, prob_a, prob_b, block_ids = y[valid], prob_a[valid], prob_b[valid], block_ids[valid]
    unique_blocks, counts = np.unique(block_ids, return_counts=True)
    block_to_idx = {b: np.where(block_ids == b)[0] for b in unique_blocks}
    n_blocks = len(unique_blocks)

    deltas_auc, deltas_pr = [], []
    n_skipped = 0
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_blocks, size=n_blocks, replace=True)
        idx = np.concatenate([block_to_idx[b] for b in sampled])
        y_s = y[idx]
        if len(np.unique(y_s)) < 2:
            n_skipped += 1
            continue
        auc_a = roc_auc_score(y_s, prob_a[idx])
        auc_b = roc_auc_score(y_s, prob_b[idx])
        pr_a = average_precision_score(y_s, prob_a[idx])
        pr_b = average_precision_score(y_s, prob_b[idx])
        deltas_auc.append(auc_b - auc_a)
        deltas_pr.append(pr_b - pr_a)

    def ci(vals):
        if not vals:
            return {"available": False}
        arr = np.array(vals)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return {
            "available": True, "mean": float(np.mean(arr)),
            "ci95": [float(lo), float(hi)],
            "interpretation": (
                "positive_bootstrap_support" if lo > 0
                else "negative_bootstrap_support" if hi < 0
                else "uncertain"
            ),
        }

    return {
        "n_bootstrap_requested": n_bootstrap,
        "n_bootstrap_successful": len(deltas_auc),
        "n_bootstrap_skipped": n_skipped,
        "delta_auc": ci(deltas_auc),
        "delta_pr_auc": ci(deltas_pr),
    }


# =============================================================================
# Writers
# =============================================================================
def build_predictions_table(df: pd.DataFrame, results: dict, population_masks: dict) -> pd.DataFrame:
    pred_cols = [
        "cell_id", "burned", "burn_month", "landcover_dominant",
        "burnable_tree_shrub_grass", "burnable_tree_shrub",
        "spatial_block_id", "observed_fraction", "gapfilled_fraction", "valid_30m_fraction",
    ]
    frames = []
    for pop_name, res in results.items():
        if res.get("skipped"):
            continue
        df_pop = df.loc[population_masks[pop_name]].reset_index(drop=True)
        base_cols = df_pop[[c for c in pred_cols if c in df_pop.columns]].copy()
        if "landcover_cropland_fraction" in df_pop.columns:
            base_cols["cropland_fraction"] = df_pop["landcover_cropland_fraction"]
        for model_name, model_res in res["models"].items():
            sub = base_cols.copy()
            sub["population"] = pop_name
            sub["model_name"] = model_name
            sub["ablation_group"] = "baseline" if model_name == "baseline" else model_name
            sub["fold_id"] = model_res["fold_id"]
            sub["y_prob"] = model_res["oof_prob"]
            frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=pred_cols + ["population", "model_name", "ablation_group", "fold_id", "y_prob"])
    return pd.concat(frames, ignore_index=True)


def write_fold_metrics_csv(results: dict, output_dir: Path) -> Path:
    rows = []
    for pop_name, res in results.items():
        if res.get("skipped"):
            continue
        for model_name, model_res in res["models"].items():
            for r in model_res["fold_rows"]:
                rows.append({"population": pop_name, **r})
    path = output_dir / "step8d_ablation_fold_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_feature_importance_csv(results: dict, output_dir: Path) -> Path:
    rows = []
    for pop_name, res in results.items():
        if res.get("skipped"):
            continue
        for model_name, model_res in res["models"].items():
            for r in model_res["feature_importance_rows"]:
                rows.append({"population": pop_name, **r})
    path = output_dir / "step8d_ablation_feature_importance.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_delta_csvs(results: dict, output_dir: Path) -> tuple[Path, Path]:
    auc_rows, pr_rows = [], []
    for pop_name, res in results.items():
        if res.get("skipped"):
            continue
        for group_name, d in res["ablation_deltas"].items():
            auc_rows.append({
                "population": pop_name, "ablation_group": group_name,
                "delta_auc": d["delta_auc"], "rank_by_delta_auc": d["rank_by_delta_auc"],
            })
            pr_rows.append({
                "population": pop_name, "ablation_group": group_name,
                "delta_pr_auc": d["delta_pr_auc"], "rank_by_delta_pr_auc": d["rank_by_delta_pr_auc"],
            })
    auc_path = output_dir / "step8d_ablation_delta_auc_by_population.csv"
    pr_path = output_dir / "step8d_ablation_delta_pr_auc_by_population.csv"
    pd.DataFrame(auc_rows).sort_values(["population", "rank_by_delta_auc"]).to_csv(auc_path, index=False)
    pd.DataFrame(pr_rows).sort_values(["population", "rank_by_delta_pr_auc"]).to_csv(pr_path, index=False)
    return auc_path, pr_path


def plot_ablation_barplot(results: dict, output_dir: Path) -> Path | None:
    try:
        populations = [p for p, r in results.items() if not r.get("skipped")]
        if not populations:
            return None
        groups = list(THERMAL_GROUPS.keys())
        fig, axes = plt.subplots(len(populations), 1, figsize=(9, 3.2 * len(populations)), squeeze=False)
        for ax, pop_name in zip(axes[:, 0], populations):
            deltas = [results[pop_name]["ablation_deltas"][g]["delta_pr_auc"] or 0.0 for g in groups]
            colors = ["#2a7f2a" if d >= 0 else "#b03030" for d in deltas]
            ax.barh(groups, deltas, color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(f"{pop_name}: delta_pr_auc vs baseline (thermal ablation)")
            ax.set_xlabel("delta_pr_auc")
        fig.tight_layout()
        path = output_dir / "step8d_ablation_barplot.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
    except Exception as exc:  # noqa: BLE001
        log.warning("Ablation barplot basarisiz: %s", exc)
        return None


def load_step8b_all_thermal_reference() -> dict:
    path = BASE_DIR / STEP8B_METRICS_PATH_REL
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("population_metrics", {})
    except Exception as exc:  # noqa: BLE001
        log.warning("Step8B metrikleri okunamadi (%s); karsilastirma atlaniyor.", exc)
        return {}


def write_stats_json(
    output_dir: Path,
    input_path: Path,
    df: pd.DataFrame,
    results: dict,
    population_masks: dict,
    bootstrap_results: dict,
    feature_missing_counts: dict,
    warnings_list: list[str],
    args_ns: argparse.Namespace,
) -> Path:
    model_metrics_by_population = {}
    ablation_delta_metrics_by_population = {}
    ablation_rankings_by_population = {}
    monthly_ablation_metrics = {}
    skipped_populations = {}

    for pop_name, res in results.items():
        if res.get("skipped"):
            skipped_populations[pop_name] = res.get("reason")
            continue
        model_metrics_by_population[pop_name] = {
            m: {
                "overall": mres["overall"], "monthly": mres["monthly"],
                "n_splits_used": res["n_splits_used"],
            }
            for m, mres in res["models"].items()
        }
        ablation_delta_metrics_by_population[pop_name] = res["ablation_deltas"]
        ablation_rankings_by_population[pop_name] = res["ranking_order"]
        monthly_ablation_metrics[pop_name] = res["monthly_ablation"]

    stats = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "step": "step8d_thermal_feature_ablation",
        "input_dataset_path": str(input_path),
        "row_count": int(len(df)),
        "target_counts": {
            "burned": int((df[TARGET_COLUMN] == 1).sum()),
            "unburned": int((df[TARGET_COLUMN] == 0).sum()),
        },
        "feature_groups": {"baseline": BASELINE_FEATURES, **THERMAL_GROUPS},
        "forbidden_columns_checked": FORBIDDEN_FEATURE_COLUMNS,
        "spatial_cv_config": {
            "method": "StratifiedGroupKFold",
            "groups": "spatial_block_id (row_500m // block, col_500m // block)",
            "spatial_block_size_cells": args_ns.spatial_block_size_cells,
            "n_splits_requested": args_ns.n_splits,
            "random_split_used": False,
            "random_state": STEP8D_RANDOM_SEED,
        },
        "model": args_ns.model, "n_estimators": args_ns.n_estimators,
        "population_counts": {pop: int(mask.sum()) for pop, mask in population_masks.items()},
        "skipped_populations": skipped_populations,
        "model_metrics_by_population": model_metrics_by_population,
        "ablation_delta_metrics_by_population": ablation_delta_metrics_by_population,
        "ablation_rankings_by_population": ablation_rankings_by_population,
        "monthly_ablation_metrics": monthly_ablation_metrics,
        "optional_bootstrap_results": bootstrap_results,
        "bootstrap_enabled": bool(bootstrap_results),
        "feature_missing_counts": feature_missing_counts,
        "warnings": warnings_list,
        "no_firms_label_used": True,
        "no_30m_pixel_samples": True,
        "label_source": "MCD64A1",
        "results_are_point_estimates_unless_bootstrap": True,
    }
    path = output_dir / "step8d_ablation_metrics.json"
    path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    return path


def write_summary_md(
    output_dir: Path,
    results: dict,
    bootstrap_enabled: bool,
    bootstrap_results: dict,
    warnings_list: list[str],
    stats_path: Path,
) -> Path:
    lines = [
        "# Step8D: Thermal Feature Ablation for Step8B Burned-Area Modeling",
        "",
        "## What this step does",
        "",
        "- Step8D identifies **which thermal feature group drives Step8B's** "
        "baseline -> thermal improvement, via ablation: baseline + each of "
        "10 thermal feature groups/singles, trained under the **same "
        "spatial-block CV** as Step8B.",
        "- It uses the **same Step8A 500 m MCD64A1-grid dataset** as Step8B "
        "-- never 30 m pixels.",
        "- **MCD64A1 is the target; FIRMS is never used.**",
        "- **`all_valid` is the primary population; `cropland_dominant` is "
        "the important secondary population.** Burnable-mask populations "
        "are diagnostic-only and skipped if positives < 30.",
        "- **Ranking is primarily by delta PR-AUC** (not delta AUC), because "
        "burned labels are rare and PR-AUC is more sensitive to rare-class "
        "improvement.",
        (
            "- **Results include spatial-block bootstrap CIs** for the "
            "top-K ablation groups per population (`--bootstrap` was used)."
            if bootstrap_enabled else
            "- **Results are POINT ESTIMATES ONLY** (`--bootstrap` was not "
            "used this run; no confidence interval is reported)."
        ),
        "",
        "## Key table: ablation ranking (by delta PR-AUC)",
        "",
    ]
    for pop_name, res in results.items():
        if res.get("skipped"):
            lines.append(f"### {pop_name}: SKIPPED ({res.get('reason')})")
            lines.append("")
            continue
        lines.append(f"### {pop_name} (n_pos={res['n_positives']}, n_neg={res['n_negatives']}, folds={res['n_splits_used']})")
        lines.append("")
        lines.append("| rank (PR-AUC) | ablation_group | delta_auc | delta_pr_auc | delta_brier |")
        lines.append("|---|---|---|---|---|")
        for group_name in res["ranking_order"]:
            d = res["ablation_deltas"][group_name]
            da = f"{d['delta_auc']:+.4f}" if d["delta_auc"] is not None else "n/a"
            dp = f"{d['delta_pr_auc']:+.4f}" if d["delta_pr_auc"] is not None else "n/a"
            db = f"{d['delta_brier']:+.4f}" if d["delta_brier"] is not None else "n/a"
            lines.append(f"| {d['rank_by_delta_pr_auc']} | {group_name} | {da} | {dp} | {db} |")
        lines.append("")

        best_single = None
        best_single_val = -np.inf
        for g in ("lst_anomaly_only", "current_lst_only", "tvdi_only", "tvdi_difference_only", "downscaled_only", "fused_lst_only"):
            d = res["ablation_deltas"][g]
            if d["delta_pr_auc"] is not None and d["delta_pr_auc"] > best_single_val:
                best_single_val, best_single = d["delta_pr_auc"], g
        best_group_name = res["ranking_order"][0] if res["ranking_order"] else None
        all_thermal_rank = res["ablation_deltas"]["all_thermal"]["rank_by_delta_pr_auc"]
        fused_downscaled_delta = res["ablation_deltas"]["fused_downscaled_group"]["delta_pr_auc"]
        simpler_best = max(
            (res["ablation_deltas"][g]["delta_pr_auc"] or -np.inf)
            for g in ("lst_anomaly_group", "tvdi_group")
        )
        fused_adds_value = (
            fused_downscaled_delta is not None and fused_downscaled_delta > simpler_best
        )

        lines.append(f"- **Best single thermal feature** (by delta_pr_auc): `{best_single}`.")
        lines.append(f"- **Best thermal group overall**: `{best_group_name}`.")
        lines.append(
            f"- `all_thermal` rank by delta_pr_auc: **#{all_thermal_rank}** of "
            f"{len(THERMAL_GROUPS)} -- "
            + ("all_thermal IS the best group." if all_thermal_rank == 1 else
               "a smaller thermal group outperforms using all thermal features together.")
        )
        lines.append(
            "- `fused_downscaled_group` "
            + ("ADDS value beyond simpler LST-anomaly/TVDI groups." if fused_adds_value
               else "does NOT clearly add value beyond simpler LST-anomaly/TVDI groups.")
        )
        lines.append("")

    lines.extend(["## Monthly lead-time ablation (existing OOF predictions, no monthly models trained)", ""])
    for pop_name, res in results.items():
        if res.get("skipped"):
            continue
        lines.append(f"- **{pop_name}** (top-ranked group: `{res['ranking_order'][0]}`):")
        top_group = res["ranking_order"][0]
        for month_name, m in res["monthly_ablation"][top_group].items():
            if m.get("delta_pr_auc") is None:
                lines.append(f"  - {month_name}: unavailable ({m.get('warning', 'n/a')})")
            else:
                lines.append(
                    f"  - {month_name} (n_pos={m['positive_count']}): "
                    f"delta_auc=`{m['delta_auc']:+.4f}`, delta_pr_auc=`{m['delta_pr_auc']:+.4f}`"
                )

    if bootstrap_enabled:
        lines.extend(["", "## Bootstrap CI (top-K ablation groups per population)", ""])
        for pop_name, groups in bootstrap_results.items():
            lines.append(f"- **{pop_name}**:")
            for group_name, ci in groups.items():
                if not ci["delta_auc"].get("available"):
                    lines.append(f"  - {group_name}: bootstrap unavailable")
                    continue
                lo, hi = ci["delta_auc"]["ci95"]
                lo_p, hi_p = ci["delta_pr_auc"]["ci95"]
                lines.append(
                    f"  - {group_name}: delta_auc CI95=`[{lo:+.4f}, {hi:+.4f}]` "
                    f"({ci['delta_auc']['interpretation']}); "
                    f"delta_pr_auc CI95=`[{lo_p:+.4f}, {hi_p:+.4f}]` "
                    f"({ci['delta_pr_auc']['interpretation']})"
                )

    lines.extend([
        "",
        "## Feature importance",
        "",
        "- Extracted from the final model refit on the whole population, "
        "per population/model. **Descriptive only -- not causal.**",
        "",
        f"Full metrics: `{stats_path.name}`",
    ])
    if warnings_list:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {w}" for w in warnings_list[:50])
        if len(warnings_list) > 50:
            lines.append(f"- ... ({len(warnings_list) - 50} more, see stats JSON)")

    path = output_dir / "step8d_ablation_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# =============================================================================
# Main
# =============================================================================
def main(
    input_arg: str | None = None,
    output_dir_arg: str = STEP8D_OUTPUT_DIR,
    force: bool = False,
    n_splits: int = STEP8D_N_SPLITS,
    spatial_block_size_cells: int = STEP8D_SPATIAL_BLOCK_SIZE_CELLS,
    min_positives: int = STEP8D_MIN_POSITIVES_PER_POPULATION,
    min_month_positives: int = STEP8D_MIN_MONTH_POSITIVES,
    model_choice: str = "random_forest",
    n_estimators: int = STEP8D_N_ESTIMATORS,
    bootstrap: bool = STEP8D_BOOTSTRAP_DEFAULT,
    n_bootstrap: int = STEP8D_BOOTSTRAP_N,
    top_k_bootstrap: int = STEP8D_BOOTSTRAP_TOP_K,
    random_seed: int = STEP8D_RANDOM_SEED,
) -> dict:
    log.info("=" * 60)
    log.info("STEP 8D BASLIYOR (thermal feature ablation, spatial-block CV)")
    log.info("=" * 60)

    out_dir = BASE_DIR / output_dir_arg
    required_outputs = [
        out_dir / "step8d_ablation_metrics.json",
        out_dir / "step8d_ablation_fold_metrics.csv",
        out_dir / "step8d_ablation_predictions.parquet",
        out_dir / "step8d_ablation_predictions.csv",
        out_dir / "step8d_ablation_feature_importance.csv",
        out_dir / "step8d_ablation_summary.md",
    ]
    if any(p.exists() for p in required_outputs) and not force:
        present = [p.name for p in required_outputs if p.exists()]
        raise Step8DError(
            "Step8D ciktilari zaten var (" + ", ".join(present) + "). Uzerine yazmak icin --force verin."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    df, input_path = load_dataset(input_arg)
    input_warnings = validate_input(df)
    check_no_forbidden_features(BASELINE_FEATURES)
    for g, feats in THERMAL_GROUPS.items():
        check_no_forbidden_features(BASELINE_FEATURES + feats)

    df = df[df["valid_for_modeling"] == True].reset_index(drop=True)  # noqa: E712
    df = add_spatial_block_id(df, spatial_block_size_cells)

    feature_missing_counts = {f: int(df[f].isna().sum()) for f in ALL_THERMAL_FEATURES if f in df.columns}
    log.info("Ozellik eksik-deger sayimlari: %s", feature_missing_counts)

    population_masks = {
        "all_valid": pd.Series(True, index=df.index),
        "cropland_dominant": df["landcover_dominant"] == LC_CROPLAND,
        "burnable_tree_shrub_grass": df["burnable_tree_shrub_grass"].astype(bool),
        "burnable_tree_shrub": df["burnable_tree_shrub"].astype(bool),
    }

    warnings_list = list(input_warnings)
    total_burned = int((df[TARGET_COLUMN] == 1).sum())
    crop_burned = int((df.loc[population_masks["cropland_dominant"], TARGET_COLUMN] == 1).sum())
    if total_burned > 0 and crop_burned / total_burned > 0.9:
        warnings_list.append(
            f"cropland_dominant contains {crop_burned}/{total_burned} "
            f"({crop_burned / total_burned:.1%}) of all burned cells."
        )
    october_pos = int(((df[TARGET_COLUMN] == 1) & (df["burn_month"] == 10)).sum())
    if october_pos < min_month_positives:
        warnings_list.append(
            f"October burn_month positives are low (n={october_pos} < {min_month_positives}); "
            "October ablation metrics may be null."
        )

    # --- Fail-fast: primary population must be feasible ---
    y_all = df.loc[population_masks["all_valid"], TARGET_COLUMN].astype(int).to_numpy()
    if len(np.unique(y_all)) < 2:
        raise Step8DError("all_valid populasyonunda 'burned' tek sinif iceriyor.")
    make_spatial_folds(
        y_all, df.loc[population_masks["all_valid"], "spatial_block_id"].to_numpy(),
        n_splits, random_seed,
    )

    # --- Train per population ---
    results: dict = {}
    for pop_name in POPULATION_ORDER:
        mask = population_masks[pop_name]
        df_pop = df.loc[mask].reset_index(drop=True)
        n_pos = int((df_pop[TARGET_COLUMN] == 1).sum())

        if pop_name in DIAGNOSTIC_ONLY_POPULATIONS and n_pos < min_positives:
            log.info("%s ATLANDI (diagnostic-only, positives=%d < %d).", pop_name, n_pos, min_positives)
            results[pop_name] = {
                "skipped": True,
                "reason": f"diagnostic_only_low_positives (n_pos={n_pos} < {min_positives})",
            }
            warnings_list.append(f"{pop_name} skipped: positives={n_pos} < {min_positives} (diagnostic-only).")
            continue

        log.info("Populasyon: %s (n=%d, burned=%d)", pop_name, len(df_pop), n_pos)
        res = train_population_ablation(
            df_pop, pop_name, n_splits, random_seed, model_choice, n_estimators, min_positives,
        )
        if res.get("skipped"):
            log.warning("%s atlandi: %s", pop_name, res.get("reason"))
        else:
            log.info(
                "%s: en iyi grup (delta_pr_auc)=%s, all_thermal rank=#%d",
                pop_name, res["ranking_order"][0],
                res["ablation_deltas"]["all_thermal"]["rank_by_delta_pr_auc"],
            )
        results[pop_name] = res

    primary = results.get("all_valid")
    if primary is None or primary.get("skipped"):
        raise Step8DError(f"Primer populasyon 'all_valid' egitilemedi: {primary.get('reason') if primary else 'sonuc yok'}")

    # --- Cross-check all_thermal vs Step8B's own thermal result ---
    step8b_ref = load_step8b_all_thermal_reference()
    for pop_name in ("all_valid", "cropland_dominant"):
        res = results.get(pop_name)
        if res is None or res.get("skipped") or pop_name not in step8b_ref:
            continue
        step8b_delta_auc = step8b_ref[pop_name].get("delta_auc")
        step8d_delta_auc = res["ablation_deltas"]["all_thermal"]["delta_auc"]
        if step8b_delta_auc is not None and step8d_delta_auc is not None:
            diff = abs(step8b_delta_auc - step8d_delta_auc)
            if diff > ALL_THERMAL_VS_STEP8B_TOLERANCE:
                warnings_list.append(
                    f"{pop_name}: Step8D's all_thermal delta_auc ({step8d_delta_auc:+.4f}) "
                    f"differs from Step8B's reported delta_auc ({step8b_delta_auc:+.4f}) by "
                    f"{diff:.4f} (> {ALL_THERMAL_VS_STEP8B_TOLERANCE}); check for randomness/"
                    "config drift between the two scripts."
                )

    # --- Optional bootstrap for top-K groups ---
    bootstrap_results: dict = {}
    if bootstrap:
        rng = np.random.default_rng(random_seed)
        for pop_name in ("all_valid", "cropland_dominant"):
            res = results.get(pop_name)
            if res is None or res.get("skipped"):
                continue
            df_pop = df.loc[population_masks[pop_name]].reset_index(drop=True)
            y = df_pop[TARGET_COLUMN].astype(int).to_numpy()
            block_ids = df_pop["spatial_block_id"].to_numpy()
            baseline_prob = res["models"]["baseline"]["oof_prob"]
            top_groups = res["ranking_order"][:top_k_bootstrap]
            bootstrap_results[pop_name] = {}
            for group_name in top_groups:
                log.info("Bootstrap (top-K): %s / %s ...", pop_name, group_name)
                group_prob = res["models"][group_name]["oof_prob"]
                bootstrap_results[pop_name][group_name] = spatial_block_bootstrap_delta(
                    y, baseline_prob, group_prob, block_ids, n_bootstrap, rng,
                )

    # --- Write outputs ---
    predictions_df = build_predictions_table(df, results, population_masks)
    predictions_csv = out_dir / "step8d_ablation_predictions.csv"
    predictions_parquet = out_dir / "step8d_ablation_predictions.parquet"
    predictions_df.to_csv(predictions_csv, index=False)
    parquet_written = False
    try:
        predictions_df.to_parquet(predictions_parquet, index=False)
        parquet_written = True
    except (ImportError, ValueError) as exc:
        log.warning("Predictions parquet yazilamadi: %s (yalniz CSV yazildi)", exc)

    fold_metrics_path = write_fold_metrics_csv(results, out_dir)
    feature_importance_path = write_feature_importance_csv(results, out_dir)
    delta_auc_csv, delta_pr_csv = write_delta_csvs(results, out_dir)
    barplot_path = plot_ablation_barplot(results, out_dir)

    args_ns = argparse.Namespace(
        spatial_block_size_cells=spatial_block_size_cells, n_splits=n_splits,
        model=model_choice, n_estimators=n_estimators,
    )
    stats_path = write_stats_json(
        out_dir, input_path, df, results, population_masks, bootstrap_results,
        feature_missing_counts, warnings_list, args_ns,
    )
    summary_path = write_summary_md(out_dir, results, bootstrap, bootstrap_results, warnings_list, stats_path)

    log.info("Stats: %s", stats_path)
    log.info("Summary: %s", summary_path)
    log.info("Predictions: %s (parquet_written=%s)", predictions_csv, parquet_written)
    log.info("=" * 60)
    log.info("STEP 8D TAMAMLANDI (no FIRMS, no 30m pixels, spatial-block CV, %s)",
              "point estimates only" if not bootstrap else "with bootstrap CI for top-K groups")
    log.info("=" * 60)

    return {
        "stats_path": str(stats_path),
        "summary_path": str(summary_path),
        "predictions_csv": str(predictions_csv),
        "predictions_parquet": str(predictions_parquet) if parquet_written else None,
        "fold_metrics_path": str(fold_metrics_path),
        "feature_importance_path": str(feature_importance_path),
        "delta_auc_csv": str(delta_auc_csv),
        "delta_pr_csv": str(delta_pr_csv),
        "barplot_path": str(barplot_path) if barplot_path else None,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step8D: thermal feature ablation for Step8B burned-area modeling."
    )
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=STEP8D_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-splits", type=int, default=STEP8D_N_SPLITS)
    parser.add_argument("--spatial-block-size-cells", type=int, default=STEP8D_SPATIAL_BLOCK_SIZE_CELLS)
    parser.add_argument("--min-positives", type=int, default=STEP8D_MIN_POSITIVES_PER_POPULATION)
    parser.add_argument("--min-month-positives", type=int, default=STEP8D_MIN_MONTH_POSITIVES)
    parser.add_argument("--model", type=str, default="random_forest", choices=["random_forest", "hist_gradient_boosting"])
    parser.add_argument("--n-estimators", type=int, default=STEP8D_N_ESTIMATORS)
    parser.add_argument("--bootstrap", action="store_true", default=STEP8D_BOOTSTRAP_DEFAULT)
    parser.add_argument("--n-bootstrap", type=int, default=STEP8D_BOOTSTRAP_N)
    parser.add_argument("--top-k-bootstrap", type=int, default=STEP8D_BOOTSTRAP_TOP_K)
    parser.add_argument("--random-seed", type=int, default=STEP8D_RANDOM_SEED)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        input_arg=args.input,
        output_dir_arg=args.output_dir,
        force=args.force,
        n_splits=args.n_splits,
        spatial_block_size_cells=args.spatial_block_size_cells,
        min_positives=args.min_positives,
        min_month_positives=args.min_month_positives,
        model_choice=args.model,
        n_estimators=args.n_estimators,
        bootstrap=args.bootstrap,
        n_bootstrap=args.n_bootstrap,
        top_k_bootstrap=args.top_k_bootstrap,
        random_seed=args.random_seed,
    )   