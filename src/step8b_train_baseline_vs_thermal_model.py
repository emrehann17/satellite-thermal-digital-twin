"""
step8b_train_baseline_vs_thermal_model.py

Step8B: the supervisor-requested DETERMINING EXPERIMENT.

Question:
    Do thermal/dryness features improve burned-area discrimination beyond
    baseline non-thermal features?

Compares, under the SAME spatial-block cross-validation and the SAME
population:
    Model A (baseline / non-thermal): elevation, slope, landcover, NDVI
    Model B (baseline + thermal):     Model A + current TVDI, LST anomaly,
                                       TVDI difference, downscaled/fused LST

Main result: delta_auc = AUC(Model B) - AUC(Model A).

IMPORTANT CONSTRAINTS (do not violate):
    - Step8B TRAINS burned-area models. It does NOT touch Step5/Step5C/Step6/
      Step7B-E/Step8A logic, and does NOT change the MCD64A1 raw BurnDate
      export.
    - MCD64A1 (via Step8A's `burned` column) is the ONLY target. FIRMS is
      NEVER used as a target.
    - Samples are Step8A's 500 m grid cells, NEVER 30 m pixels.
    - Cross-validation is SPATIAL-BLOCK based (StratifiedGroupKFold over
      500 m cell blocks). Random/row-wise splitting is never used.
    - burn_month, burn_date, burned, label_source, and other label/metadata
      columns are excluded from the feature set (see FORBIDDEN_FEATURE_
      COLUMNS). source_mask/observed/gapfilled provenance columns are used
      only for sensitivity diagnostics, never as predictive features.

Input (read-only):
    outputs/step8a/step8a_500m_modeling_dataset.parquet (preferred)
    outputs/step8a/step8a_500m_modeling_dataset.csv (fallback)

Output:
    outputs/step8b/step8b_model_comparison_metrics.json
    outputs/step8b/step8b_fold_metrics.csv
    outputs/step8b/step8b_predictions.parquet
    outputs/step8b/step8b_predictions.csv
    outputs/step8b/step8b_feature_importance.csv
    outputs/step8b/step8b_summary.md
    outputs/step8b/step8b_roc_curves.png            (optional)
    outputs/step8b/step8b_pr_curves.png              (optional)
    outputs/step8b/step8b_delta_auc_by_population.csv (optional)
    outputs/step8b/step8b_monthly_leadtime_metrics.csv (optional)

CLI:
    python src/step8b_train_baseline_vs_thermal_model.py --force
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
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from core.config import (
    STEP8B_OUTPUT_DIR,
    STEP8B_INPUT_DATASET,
    STEP8B_RANDOM_SEED,
    STEP8B_N_SPLITS,
    STEP8B_SPATIAL_BLOCK_SIZE_CELLS,
    STEP8B_MIN_POSITIVES_PER_POPULATION,
    STEP8B_MIN_MONTH_POSITIVES,
    STEP8B_PRIMARY_POPULATION,
)
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT

BASE_DIR = PROJECT_ROOT
log, log_file = setup_logger("step8b")

pywarnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class Step8BError(SystemExit):
    """Fail-fast error for Step8B (extends SystemExit like other steps)."""


# =============================================================================
# Target / feature registry
# =============================================================================
TARGET_COLUMN = "burned"

# ESA WorldCover class code for cropland (must match Step8A's mapping).
LC_CROPLAND = 40

BASELINE_FEATURES = [
    "ndvi_mean",
    "elevation_mean",
    "slope_mean",
    "landcover_dominant",
]
THERMAL_FEATURES = [
    "lst_anomaly_mean",
    "current_lst_mean",
    "current_tvdi_mean",
    "tvdi_difference_mean",
    "downscaled_lst_mean",
    "fused_lst_mean",
]
THERMAL_MODEL_FEATURES = BASELINE_FEATURES + THERMAL_FEATURES
CATEGORICAL_FEATURES = ["landcover_dominant"]

# Columns that must NEVER be used as model features (label/metadata/leakage
# and source-provenance columns; provenance is diagnostics-only).
FORBIDDEN_FEATURE_COLUMNS = [
    "burned",
    "burn_date",
    "burn_month",
    "burn_day_of_year",
    "label_source",
    "burn_date_pixel_agreement_fraction",
    "out_of_window_burndate",
    "cell_id",
    "row_500m",
    "col_500m",
    "valid_for_modeling",
    "invalid_reason",
    "source_mask_majority",
    "observed_fraction",
    "gapfilled_fraction",
    "invalid_source_fraction",
    "lon",
    "lat",
]

REQUIRED_COLUMNS = list(dict.fromkeys(
    BASELINE_FEATURES + THERMAL_FEATURES + [
        TARGET_COLUMN, "row_500m", "col_500m", "lon", "lat", "cell_id",
        "valid_for_modeling", "burn_month", "burnable_tree_shrub_grass",
        "burnable_tree_shrub", "landcover_cropland_fraction",
    ]
))

POPULATION_ORDER = [
    "all_valid",
    "cropland_dominant",
    "burnable_tree_shrub_grass",
    "burnable_tree_shrub",
]
DIAGNOSTIC_ONLY_POPULATIONS = {"burnable_tree_shrub_grass", "burnable_tree_shrub"}

MONTHS = [8, 9, 10]
MONTH_NAMES = {8: "august", 9: "september", 10: "october"}

GAPFILL_SENSITIVITY_THRESHOLDS = [0.25, 0.50]
SENSITIVITY_POPULATIONS = {"all_valid", "cropland_dominant"}


# =============================================================================
# Data loading + validation
# =============================================================================
def load_dataset(input_arg: str | None) -> tuple[pd.DataFrame, Path]:
    parquet_path = BASE_DIR / (input_arg or STEP8B_INPUT_DATASET)
    # CSV fallback: AYNI dizin, .csv uzantisi (parquet_path'in kendi
    # dizininden turetilir -- hardcoded "outputs/step8a" DEGIL, boylece
    # namespaced (Kozan-disi) bir input_arg verildiginde de dogru dizine
    # bakar).
    csv_path = parquet_path.with_suffix(".csv")

    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            log.info("Step8A veri seti okundu (parquet): %s (%d satir)", parquet_path, len(df))
            return df, parquet_path
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Parquet okunamadi (%s: %s); CSV fallback deneniyor.",
                type(exc).__name__, exc,
            )

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        log.info("Step8A veri seti okundu (CSV fallback): %s (%d satir)", csv_path, len(df))
        return df, csv_path

    raise Step8BError(
        "Step8A modelleme veri seti bulunamadi. Beklenen: "
        f"{parquet_path} veya {csv_path}. Once Step8A'yi calistirin: "
        "python src/step8a_prepare_500m_modeling_dataset.py --force"
    )


def validate_input(df: pd.DataFrame) -> list[str]:
    """Fail-fast checks on the loaded Step8A dataset. Returns warnings."""
    warnings_out: list[str] = []

    if TARGET_COLUMN not in df.columns:
        raise Step8BError(
            f"Girdi veri setinde '{TARGET_COLUMN}' kolonu yok. Step8A ciktisi "
            "bozuk veya yanlis dosya verildi."
        )

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise Step8BError(
            f"Girdi veri setinde beklenen kolonlar eksik: {missing_required}. "
            "Step8A'yi guncel scriptle yeniden calistirin."
        )

    if "valid_for_modeling" in df.columns:
        n_before = len(df)
        df_valid = df[df["valid_for_modeling"] == True]  # noqa: E712
        if len(df_valid) == 0:
            raise Step8BError("valid_for_modeling==True olan hic satir yok.")
        if len(df_valid) < n_before:
            log.info(
                "valid_for_modeling==False olan %d satir modelleme "
                "populasyonlarindan haric tutulacak (Step8A'nin kendi "
                "isaretlemesi; Step8B bu satirlari SILMEZ, sadece kullanmaz).",
                n_before - len(df_valid),
            )

    burned_vals = df.loc[df["valid_for_modeling"] == True, TARGET_COLUMN].dropna().unique()  # noqa: E712
    if len(burned_vals) < 2:
        raise Step8BError(
            f"'{TARGET_COLUMN}' hedefi valid_for_modeling==True populasyonunda "
            f"tek sinif iceriyor ({burned_vals}). En az burned=0 ve burned=1 "
            "gerekli (all_valid icin)."
        )

    if "gapfilled_fraction" not in df.columns or df["gapfilled_fraction"].isna().all():
        warnings_out.append(
            "gapfilled_fraction kolonu eksik veya tamamen NaN; gap-fill "
            "sensitivity diagnostics calistirilamayabilir."
        )

    return warnings_out


def check_no_forbidden_features(feature_list: list[str]) -> None:
    leaked = set(feature_list).intersection(FORBIDDEN_FEATURE_COLUMNS)
    if leaked:
        raise Step8BError(
            f"YASAK etiket/metadata kolonlari ozellik setine sizmis: {leaked}. "
            "Bu Step8B'nin temel guvenlik kontrolüdür ve asla gecilmemelidir."
        )


# =============================================================================
# Spatial blocks
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
    random (non-grouped) split. Tries n_splits_requested first; if the
    resulting folds are degenerate (fewer than `min_positive_folds` test
    folds contain a positive), retries with 3 folds; if still degenerate,
    fails clearly.

    Returns (folds, n_splits_used) where folds is a list of (train_idx,
    test_idx) index arrays into the input arrays.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))

    def _try(n_splits: int):
        if n_splits < 2 or n_groups < n_splits:
            return None
        try:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            folds = list(splitter.split(np.zeros(len(y)), y, groups=groups))
        except Exception as exc:  # noqa: BLE001
            log.warning("StratifiedGroupKFold(n_splits=%d) basarisiz: %s", n_splits, exc)
            return None

        folds_with_positive = sum(1 for _, test_idx in folds if y[test_idx].sum() > 0)
        if folds_with_positive < min_positive_folds:
            log.warning(
                "n_splits=%d ile yalnizca %d fold'da pozitif ornek var "
                "(gerekli: >=%d).", n_splits, folds_with_positive, min_positive_folds,
            )
            return None
        return folds

    for n_splits in sorted({n_splits_requested}, reverse=True):
        folds = _try(n_splits)
        if folds is not None:
            return folds, n_splits

    if n_splits_requested != 3:
        folds = _try(3)
        if folds is not None:
            log.warning(
                "n_splits=%d gecerli fold uretemedi; 3 fold'a dusuruldu.",
                n_splits_requested,
            )
            return folds, 3

    raise Step8BError(
        "Spatial-block CV gecerli fold uretemedi (StratifiedGroupKFold, "
        f"n_splits={n_splits_requested} ve 3 denendi). Class imbalance veya "
        "spatial block sayisi cok dusuk olabilir. RANDOM SPLIT'e "
        "DUSULMEYECEK -- bu populasyon icin modelleme atlanmali."
    )


# =============================================================================
# Model pipeline
# =============================================================================
def build_classifier(model_name: str, random_state: int):
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            random_state=random_state, class_weight="balanced",
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise Step8BError(
                "xgboost yuklu degil ve zorunlu degildir. Kurmak icin: "
                "pip install xgboost --break-system-packages. Veya "
                "--model random_forest kullanin."
            ) from exc
        return XGBClassifier(
            n_estimators=300, random_state=random_state, eval_metric="logloss",
        )
    raise Step8BError(f"Bilinmeyen model: {model_name}")


def build_pipeline(feature_list: list[str], model_name: str, random_state: int) -> Pipeline:
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
    clf = build_classifier(model_name, random_state)
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])


def get_expanded_feature_names(pipeline: Pipeline, feature_list: list[str]) -> list[str]:
    try:
        return list(pipeline.named_steps["preprocess"].get_feature_names_out())
    except Exception:  # noqa: BLE001
        return list(feature_list)


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
        out.update({
            "roc_auc": None, "pr_auc": None, "brier_score": None,
            "balanced_accuracy": None, "precision": None, "recall": None,
            "f1": None,
        })
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


def interpret_delta(delta_auc: float | None) -> str:
    if delta_auc is None:
        return "unavailable"
    if delta_auc > 0.01:
        return "thermal_improves"
    if delta_auc < -0.01:
        return "thermal_degrades"
    return "neutral"


# =============================================================================
# Per-population training
# =============================================================================
def train_population(
    df_pop: pd.DataFrame,
    population_name: str,
    n_splits: int,
    random_state: int,
    model_name: str,
    min_positives: int,
) -> dict | None:
    """
    Trains Model A (baseline) and Model B (baseline+thermal) on the SAME
    spatial-block CV folds for this population. Returns a result dict, or
    None if the population must be skipped (with the reason logged/returned
    via the caller's skip bookkeeping).
    """
    y = df_pop[TARGET_COLUMN].astype(int).to_numpy()
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())

    if n_pos < min_positives or n_neg < min_positives:
        return {
            "skipped": True,
            "reason": (
                f"insufficient_positives_or_negatives (positives={n_pos}, "
                f"negatives={n_neg}, min_required={min_positives})"
            ),
        }

    groups = df_pop["spatial_block_id"].to_numpy()
    try:
        folds, n_splits_used = make_spatial_folds(y, groups, n_splits, random_state)
    except Step8BError as exc:
        return {"skipped": True, "reason": f"spatial_cv_failed: {exc}"}

    n_index = len(df_pop)
    oof_prob_baseline = np.full(n_index, np.nan)
    oof_prob_thermal = np.full(n_index, np.nan)
    fold_id = np.full(n_index, -1, dtype=int)

    fold_rows: list[dict] = []

    for i, (train_idx, test_idx) in enumerate(folds):
        fold_id[test_idx] = i
        X_train, y_train = df_pop.iloc[train_idx], y[train_idx]
        X_test, y_test = df_pop.iloc[test_idx], y[test_idx]

        pipe_a = build_pipeline(BASELINE_FEATURES, model_name, random_state)
        pipe_a.fit(X_train[BASELINE_FEATURES], y_train)
        prob_a = pipe_a.predict_proba(X_test[BASELINE_FEATURES])[:, 1]
        oof_prob_baseline[test_idx] = prob_a

        pipe_b = build_pipeline(THERMAL_MODEL_FEATURES, model_name, random_state)
        pipe_b.fit(X_train[THERMAL_MODEL_FEATURES], y_train)
        prob_b = pipe_b.predict_proba(X_test[THERMAL_MODEL_FEATURES])[:, 1]
        oof_prob_thermal[test_idx] = prob_b

        m_a = compute_binary_metrics(y_test, prob_a)
        m_b = compute_binary_metrics(y_test, prob_b)
        fold_rows.append({
            "population": population_name, "fold": i,
            "n_train": len(train_idx), "n_test": len(test_idx),
            "test_positives": int(y_test.sum()), "test_negatives": int((y_test == 0).sum()),
            "auc_baseline": m_a["roc_auc"], "auc_thermal": m_b["roc_auc"],
            "pr_auc_baseline": m_a["pr_auc"], "pr_auc_thermal": m_b["pr_auc"],
            "brier_baseline": m_a["brier_score"], "brier_thermal": m_b["brier_score"],
        })

    overall_baseline = compute_binary_metrics(y, oof_prob_baseline)
    overall_thermal = compute_binary_metrics(y, oof_prob_thermal)

    delta_auc = (
        overall_thermal["roc_auc"] - overall_baseline["roc_auc"]
        if overall_baseline["roc_auc"] is not None and overall_thermal["roc_auc"] is not None
        else None
    )
    delta_pr_auc = (
        overall_thermal["pr_auc"] - overall_baseline["pr_auc"]
        if overall_baseline["pr_auc"] is not None and overall_thermal["pr_auc"] is not None
        else None
    )
    delta_brier = (
        overall_thermal["brier_score"] - overall_baseline["brier_score"]
        if overall_baseline["brier_score"] is not None and overall_thermal["brier_score"] is not None
        else None
    )

    # Monthly lead-time evaluation using the SAME out-of-fold predictions
    # (no separate monthly models are trained).
    monthly = {}
    burn_month = df_pop["burn_month"].to_numpy()
    for month in MONTHS:
        pos_mask = (y == 1) & (burn_month == month)
        neg_mask = (y == 0)
        eval_mask = pos_mask | neg_mask
        n_pos_month = int(pos_mask.sum())
        entry = {"positive_count": n_pos_month, "negative_count": int(neg_mask.sum())}
        if n_pos_month < STEP8B_MIN_MONTH_POSITIVES:
            entry.update({
                "auc_baseline": None, "auc_thermal": None,
                "pr_auc_baseline": None, "pr_auc_thermal": None,
                "delta_auc": None, "delta_pr_auc": None,
                "warning": (
                    f"month {MONTH_NAMES[month]} has too few positives "
                    f"({n_pos_month} < {STEP8B_MIN_MONTH_POSITIVES})"
                ),
            })
        else:
            y_m = y[eval_mask]
            mb = compute_binary_metrics(y_m, oof_prob_baseline[eval_mask])
            mt = compute_binary_metrics(y_m, oof_prob_thermal[eval_mask])
            d_auc = (
                mt["roc_auc"] - mb["roc_auc"]
                if mb["roc_auc"] is not None and mt["roc_auc"] is not None else None
            )
            d_pr = (
                mt["pr_auc"] - mb["pr_auc"]
                if mb["pr_auc"] is not None and mt["pr_auc"] is not None else None
            )
            entry.update({
                "auc_baseline": mb["roc_auc"], "auc_thermal": mt["roc_auc"],
                "pr_auc_baseline": mb["pr_auc"], "pr_auc_thermal": mt["pr_auc"],
                "delta_auc": d_auc, "delta_pr_auc": d_pr,
            })
        monthly[MONTH_NAMES[month]] = entry

    # Final models fit on the WHOLE population for feature importance.
    final_pipe_a = build_pipeline(BASELINE_FEATURES, model_name, random_state)
    final_pipe_a.fit(df_pop[BASELINE_FEATURES], y)
    final_pipe_b = build_pipeline(THERMAL_MODEL_FEATURES, model_name, random_state)
    final_pipe_b.fit(df_pop[THERMAL_MODEL_FEATURES], y)

    feature_importance_rows = []
    for label, pipe, feat_list in (
        ("baseline", final_pipe_a, BASELINE_FEATURES),
        ("thermal", final_pipe_b, THERMAL_MODEL_FEATURES),
    ):
        names = get_expanded_feature_names(pipe, feat_list)
        clf = pipe.named_steps["clf"]
        importances = getattr(clf, "feature_importances_", None)
        if importances is None:
            continue
        for name, imp in zip(names, importances):
            feature_importance_rows.append({
                "population": population_name, "model": label,
                "feature": name, "importance": float(imp),
            })

    return {
        "skipped": False,
        "n_splits_used": n_splits_used,
        "n_positives": n_pos, "n_negatives": n_neg,
        "overall_baseline": overall_baseline,
        "overall_thermal": overall_thermal,
        "delta_auc": delta_auc, "delta_pr_auc": delta_pr_auc, "delta_brier": delta_brier,
        "interpretation": interpret_delta(delta_auc),
        "monthly": monthly,
        "fold_rows": fold_rows,
        "oof_prob_baseline": oof_prob_baseline,
        "oof_prob_thermal": oof_prob_thermal,
        "fold_id": fold_id,
        "feature_importance_rows": feature_importance_rows,
    }


def gapfill_sensitivity(df_pop: pd.DataFrame, y: np.ndarray, oof_baseline: np.ndarray, oof_thermal: np.ndarray) -> list[dict]:
    """
    Evaluates EXISTING out-of-fold predictions on gap-fill-filtered subsets.
    Does NOT retrain any model.
    """
    rows = []
    if "gapfilled_fraction" not in df_pop.columns:
        return rows
    gf = df_pop["gapfilled_fraction"].to_numpy()
    for thr in GAPFILL_SENSITIVITY_THRESHOLDS:
        mask = np.isfinite(gf) & (gf < thr)
        if mask.sum() == 0:
            rows.append({"gapfilled_fraction_lt": thr, "n_cells": 0, "note": "no cells pass filter"})
            continue
        mb = compute_binary_metrics(y[mask], oof_baseline[mask])
        mt = compute_binary_metrics(y[mask], oof_thermal[mask])
        delta = (
            mt["roc_auc"] - mb["roc_auc"]
            if mb["roc_auc"] is not None and mt["roc_auc"] is not None else None
        )
        rows.append({
            "gapfilled_fraction_lt": thr,
            "n_cells": int(mask.sum()),
            "positive_count": mb["positive_count"], "negative_count": mb["negative_count"],
            "auc_baseline": mb["roc_auc"], "auc_thermal": mt["roc_auc"],
            "delta_auc": delta,
        })
    return rows


# =============================================================================
# Plotting (optional outputs)
# =============================================================================
def plot_roc_curves(results: dict, output_dir: Path) -> Path | None:
    try:
        fig, ax = plt.subplots(figsize=(7, 6))
        any_plotted = False
        for pop_name, res in results.items():
            if not isinstance(res, dict) or res.get("skipped"):
                continue
            y = res.get("_y")
            if y is None:
                continue
            for label, prob, style in (
                ("baseline", res["oof_prob_baseline"], "--"),
                ("thermal", res["oof_prob_thermal"], "-"),
            ):
                mask = np.isfinite(prob)
                if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
                    continue
                fpr, tpr, _ = roc_curve(y[mask], prob[mask])
                auc = res["overall_baseline"]["roc_auc"] if label == "baseline" else res["overall_thermal"]["roc_auc"]
                lbl = f"{pop_name} ({label}, AUC={auc:.3f})" if auc is not None else f"{pop_name} ({label})"
                ax.plot(fpr, tpr, style, label=lbl)
                any_plotted = True
        if not any_plotted:
            plt.close(fig)
            return None
        ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Step8B ROC curves (out-of-fold, spatial-block CV)")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        path = output_dir / "step8b_roc_curves.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
    except Exception as exc:  # noqa: BLE001
        log.warning("ROC curve plot basarisiz: %s", exc)
        return None


def plot_pr_curves(results: dict, output_dir: Path) -> Path | None:
    try:
        fig, ax = plt.subplots(figsize=(7, 6))
        any_plotted = False
        for pop_name, res in results.items():
            if not isinstance(res, dict) or res.get("skipped"):
                continue
            y = res.get("_y")
            if y is None:
                continue
            for label, prob, style in (
                ("baseline", res["oof_prob_baseline"], "--"),
                ("thermal", res["oof_prob_thermal"], "-"),
            ):
                mask = np.isfinite(prob)
                if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
                    continue
                precision, recall, _ = precision_recall_curve(y[mask], prob[mask])
                pr_auc = res["overall_baseline"]["pr_auc"] if label == "baseline" else res["overall_thermal"]["pr_auc"]
                lbl = f"{pop_name} ({label}, PR-AUC={pr_auc:.3f})" if pr_auc is not None else f"{pop_name} ({label})"
                ax.plot(recall, precision, style, label=lbl)
                any_plotted = True
        if not any_plotted:
            plt.close(fig)
            return None
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Step8B Precision-Recall curves (out-of-fold, spatial-block CV)")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        path = output_dir / "step8b_pr_curves.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
    except Exception as exc:  # noqa: BLE001
        log.warning("PR curve plot basarisiz: %s", exc)
        return None


# =============================================================================
# Writers
# =============================================================================
def build_predictions_table(df: pd.DataFrame, results: dict, population_masks: dict) -> pd.DataFrame:
    frames = []
    pred_cols = [
        "cell_id", "burned", "burn_month", "landcover_dominant",
        "burnable_tree_shrub_grass", "burnable_tree_shrub",
        "spatial_block_id", "observed_fraction", "gapfilled_fraction",
        "valid_30m_fraction",
    ]
    for pop_name, res in results.items():
        if not isinstance(res, dict) or res.get("skipped"):
            continue
        df_pop = df.loc[population_masks[pop_name]].reset_index(drop=True)
        sub = df_pop[[c for c in pred_cols if c in df_pop.columns]].copy()
        if "landcover_cropland_fraction" in df_pop.columns:
            sub["cropland_fraction"] = df_pop["landcover_cropland_fraction"]
        sub["population"] = pop_name
        sub["fold_id"] = res["fold_id"]
        sub["y_prob_baseline"] = res["oof_prob_baseline"]
        sub["y_prob_thermal"] = res["oof_prob_thermal"]
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=pred_cols + ["population", "fold_id", "y_prob_baseline", "y_prob_thermal"])
    return pd.concat(frames, ignore_index=True)


def write_fold_metrics_csv(results: dict, output_dir: Path) -> Path:
    rows = []
    for pop_name, res in results.items():
        if not isinstance(res, dict) or res.get("skipped"):
            continue
        rows.extend(res["fold_rows"])
    df_folds = pd.DataFrame(rows)
    path = output_dir / "step8b_fold_metrics.csv"
    df_folds.to_csv(path, index=False)
    return path


def write_feature_importance_csv(results: dict, output_dir: Path) -> Path:
    rows = []
    for pop_name, res in results.items():
        if not isinstance(res, dict) or res.get("skipped"):
            continue
        rows.extend(res["feature_importance_rows"])
    df_fi = pd.DataFrame(rows)
    path = output_dir / "step8b_feature_importance.csv"
    df_fi.to_csv(path, index=False)
    return path


def write_delta_auc_by_population_csv(results: dict, output_dir: Path) -> Path:
    rows = []
    for pop_name, res in results.items():
        if not isinstance(res, dict):
            continue
        if res.get("skipped"):
            rows.append({"population": pop_name, "skipped": True, "reason": res.get("reason")})
            continue
        rows.append({
            "population": pop_name, "skipped": False,
            "n_positives": res["n_positives"], "n_negatives": res["n_negatives"],
            "auc_baseline": res["overall_baseline"]["roc_auc"],
            "auc_thermal": res["overall_thermal"]["roc_auc"],
            "delta_auc": res["delta_auc"],
            "pr_auc_baseline": res["overall_baseline"]["pr_auc"],
            "pr_auc_thermal": res["overall_thermal"]["pr_auc"],
            "delta_pr_auc": res["delta_pr_auc"],
            "interpretation": res["interpretation"],
        })
    df_delta = pd.DataFrame(rows)
    path = output_dir / "step8b_delta_auc_by_population.csv"
    df_delta.to_csv(path, index=False)
    return path


def write_monthly_leadtime_csv(results: dict, output_dir: Path) -> Path:
    rows = []
    for pop_name, res in results.items():
        if not isinstance(res, dict) or res.get("skipped"):
            continue
        for month_name, entry in res["monthly"].items():
            rows.append({"population": pop_name, "month": month_name, **entry})
    df_month = pd.DataFrame(rows)
    path = output_dir / "step8b_monthly_leadtime_metrics.csv"
    df_month.to_csv(path, index=False)
    return path


def write_stats_json(
    output_dir: Path,
    input_path: Path,
    df: pd.DataFrame,
    results: dict,
    population_masks: dict,
    sensitivity: dict,
    feature_missing_counts: dict,
    warnings_list: list[str],
    args: argparse.Namespace,
) -> Path:
    population_summaries = {}
    skipped_populations = {}
    for pop_name, res in results.items():
        if not isinstance(res, dict):
            continue
        if res.get("skipped"):
            skipped_populations[pop_name] = res.get("reason")
            continue
        population_summaries[pop_name] = {
            "n_splits_used": res["n_splits_used"],
            "n_positives": res["n_positives"],
            "n_negatives": res["n_negatives"],
            "overall_baseline": res["overall_baseline"],
            "overall_thermal": res["overall_thermal"],
            "delta_auc": res["delta_auc"],
            "delta_pr_auc": res["delta_pr_auc"],
            "delta_brier": res["delta_brier"],
            "interpretation": res["interpretation"],
            "monthly": res["monthly"],
        }

    stats = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "step": "step8b_train_baseline_vs_thermal_model",
        "input_dataset_path": str(input_path),
        "row_count": int(len(df)),
        "target_counts": {
            "burned": int((df[TARGET_COLUMN] == 1).sum()),
            "unburned": int((df[TARGET_COLUMN] == 0).sum()),
        },
        "feature_sets": {
            "baseline": BASELINE_FEATURES,
            "thermal_additional": THERMAL_FEATURES,
            "thermal_model_full": THERMAL_MODEL_FEATURES,
        },
        "forbidden_columns_checked": FORBIDDEN_FEATURE_COLUMNS,
        "spatial_cv_config": {
            "method": "StratifiedGroupKFold",
            "groups": "spatial_block_id (row_500m // block, col_500m // block)",
            "spatial_block_size_cells": args.spatial_block_size_cells,
            "n_splits_requested": args.n_splits,
            "random_split_used": False,
            "random_state": STEP8B_RANDOM_SEED,
        },
        "model": args.model,
        "population_counts": {
            pop: int(mask.sum()) for pop, mask in population_masks.items()
        },
        "skipped_populations": skipped_populations,
        "population_metrics": population_summaries,
        "gapfill_sensitivity": sensitivity,
        "feature_missing_counts": feature_missing_counts,
        "warnings": warnings_list,
        "no_firms_label_used": True,
        "no_30m_pixel_samples": True,
        "label_source": "MCD64A1",
        "primary_population": STEP8B_PRIMARY_POPULATION,
        "no_model_confidence_interval_or_pvalue_reported": True,
        "note_on_significance": (
            "No statistical significance test (confidence interval or "
            "p-value) is implemented for delta_auc/delta_pr_auc in this run. "
            "Differences should be read as point estimates only."
        ),
    }
    path = output_dir / "step8b_model_comparison_metrics.json"
    path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    return path


def write_summary_md(
    output_dir: Path,
    results: dict,
    sensitivity: dict,
    stats_path: Path,
    warnings_list: list[str],
) -> Path:
    lines = [
        "# Step8B: Baseline vs Baseline+Thermal Burned-Area Modeling",
        "",
        "## What this step does",
        "",
        "- Step8B is the **determining experiment** requested by the "
        "supervisor: does adding thermal/dryness features improve "
        "burned-area discrimination beyond baseline non-thermal features?",
        "- It compares **Model A (baseline)**: elevation + slope + landcover "
        "+ NDVI, against **Model B (baseline + thermal)**: Model A + current "
        "TVDI + LST anomaly + TVDI difference + downscaled/fused thermal "
        "features.",
        "- It uses the **Step8A 500 m MCD64A1-grid dataset** -- each sample "
        "is one native ~500 m modeling cell, **never a 30 m pixel**.",
        "- **MCD64A1 is the target label; FIRMS is never used as a target.**",
        "- **Spatial-block cross-validation** (`StratifiedGroupKFold` over "
        "500 m cell blocks) is used throughout; **no random row-wise split "
        "is ever performed.**",
        "- **No statistical significance test is implemented** for delta "
        "metrics in this run -- results are point estimates only.",
        "",
        "## Populations",
        "",
        "- `all_valid` is the **primary analysis**, because the "
        "cropland-excluded burnable mask (`burnable_tree_shrub_grass`) has "
        "too few burned positives to be a reliable primary population.",
        "- `cropland_dominant` is evaluated because the large majority of "
        "burned cells in this AOI/window are cropland-dominant.",
        "- `burnable_tree_shrub_grass` and `burnable_tree_shrub` are "
        "**diagnostic/sensitivity only** and are skipped by default if "
        "burned positives are below the minimum threshold.",
        "",
        "## Lead-time (monthly) evaluation",
        "",
        "- Lead-time is evaluated using **out-of-fold predictions from the "
        "single full Aug-Oct model**, stratified by `burn_month` "
        "(August/September/October) -- **no separate monthly models are "
        "trained.**",
        "",
        "## Key result: delta AUC",
        "",
    ]
    for pop_name in POPULATION_ORDER:
        res = results.get(pop_name)
        if res is None:
            continue
        if res.get("skipped"):
            lines.append(f"- **{pop_name}**: SKIPPED ({res.get('reason')})")
            continue
        auc_a = res["overall_baseline"]["roc_auc"]
        auc_b = res["overall_thermal"]["roc_auc"]
        pr_a = res["overall_baseline"]["pr_auc"]
        pr_b = res["overall_thermal"]["pr_auc"]
        lines.append(
            f"- **{pop_name}** (n_pos={res['n_positives']}, "
            f"n_neg={res['n_negatives']}, folds={res['n_splits_used']}): "
            f"AUC baseline=`{auc_a:.4f}` -> thermal=`{auc_b:.4f}` "
            f"(delta_auc=`{res['delta_auc']:+.4f}`); "
            f"PR-AUC baseline=`{pr_a:.4f}` -> thermal=`{pr_b:.4f}` "
            f"(delta_pr_auc=`{res['delta_pr_auc']:+.4f}`); "
            f"**{res['interpretation']}**"
        )
    lines.extend(["", "## Monthly lead-time metrics", ""])
    for pop_name in POPULATION_ORDER:
        res = results.get(pop_name)
        if res is None or res.get("skipped"):
            continue
        lines.append(f"- **{pop_name}**:")
        for month_name, entry in res["monthly"].items():
            if entry.get("auc_baseline") is None:
                lines.append(f"  - {month_name}: unavailable ({entry.get('warning')})")
            else:
                lines.append(
                    f"  - {month_name} (n_pos={entry['positive_count']}): "
                    f"AUC baseline=`{entry['auc_baseline']:.4f}` -> "
                    f"thermal=`{entry['auc_thermal']:.4f}` "
                    f"(delta_auc=`{entry['delta_auc']:+.4f}`)"
                )
    lines.extend(["", "## Gap-fill sensitivity (existing predictions only, no retraining)", ""])
    for pop_name, rows in sensitivity.items():
        lines.append(f"- **{pop_name}**:")
        for row in rows:
            if row.get("auc_baseline") is None:
                lines.append(
                    f"  - gapfilled_fraction < {row['gapfilled_fraction_lt']}: "
                    f"{row.get('note', 'unavailable')} (n_cells={row.get('n_cells', 0)})"
                )
            else:
                lines.append(
                    f"  - gapfilled_fraction < {row['gapfilled_fraction_lt']} "
                    f"(n={row['n_cells']}): AUC baseline=`{row['auc_baseline']:.4f}` "
                    f"-> thermal=`{row['auc_thermal']:.4f}` "
                    f"(delta_auc=`{row['delta_auc']:+.4f}`)"
                )
    lines.extend([
        "",
        "## Feature importance",
        "",
        "- Feature importance is extracted from the final model refit on the "
        "whole population (per population, per model). **This is "
        "descriptive only -- not a causal attribution.**",
        "",
        f"Full metrics: `{stats_path.name}`",
    ])
    if warnings_list:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {w}" for w in warnings_list[:50])
        if len(warnings_list) > 50:
            lines.append(f"- ... ({len(warnings_list) - 50} more, see stats JSON)")

    path = output_dir / "step8b_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# =============================================================================
# Main
# =============================================================================
def main(
    input_arg: str | None = None,
    output_dir_arg: str = STEP8B_OUTPUT_DIR,
    force: bool = False,
    n_splits: int = STEP8B_N_SPLITS,
    spatial_block_size_cells: int = STEP8B_SPATIAL_BLOCK_SIZE_CELLS,
    min_positives: int = STEP8B_MIN_POSITIVES_PER_POPULATION,
    min_month_positives: int = STEP8B_MIN_MONTH_POSITIVES,
    allow_low_positive_strata: bool = False,
    model_name: str = "random_forest",
) -> dict:
    log.info("=" * 60)
    log.info("STEP 8B BASLIYOR (baseline vs baseline+thermal, spatial-block CV)")
    log.info("=" * 60)

    out_dir = BASE_DIR / output_dir_arg
    required_outputs = [
        out_dir / "step8b_model_comparison_metrics.json",
        out_dir / "step8b_fold_metrics.csv",
        out_dir / "step8b_predictions.parquet",
        out_dir / "step8b_predictions.csv",
        out_dir / "step8b_feature_importance.csv",
        out_dir / "step8b_summary.md",
    ]
    if any(p.exists() for p in required_outputs) and not force:
        present = [p.name for p in required_outputs if p.exists()]
        raise Step8BError(
            "Step8B ciktilari zaten var (" + ", ".join(present)
            + "). Uzerine yazmak icin --force verin."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    df, input_path = load_dataset(input_arg)
    input_warnings = validate_input(df)

    check_no_forbidden_features(BASELINE_FEATURES)
    check_no_forbidden_features(THERMAL_MODEL_FEATURES)

    df = df[df["valid_for_modeling"] == True].reset_index(drop=True)  # noqa: E712
    df = add_spatial_block_id(df, spatial_block_size_cells)

    feature_missing_counts = {
        f: int(df[f].isna().sum()) for f in THERMAL_MODEL_FEATURES if f in df.columns
    }
    log.info("Ozellik eksik-deger sayimlari: %s", feature_missing_counts)

    # --- Population masks ---
    population_masks = {
        "all_valid": pd.Series(True, index=df.index),
        "cropland_dominant": df["landcover_dominant"] == LC_CROPLAND,
        "burnable_tree_shrub_grass": df["burnable_tree_shrub_grass"].astype(bool),
        "burnable_tree_shrub": df["burnable_tree_shrub"].astype(bool),
    }

    warnings_list = list(input_warnings)

    # Cross-checks / warnings about population composition.
    total_burned = int((df[TARGET_COLUMN] == 1).sum())
    crop_burned = int((df.loc[population_masks["cropland_dominant"], TARGET_COLUMN] == 1).sum())
    if total_burned > 0 and crop_burned / total_burned > 0.9:
        warnings_list.append(
            f"cropland_dominant contains {crop_burned}/{total_burned} "
            f"({crop_burned / total_burned:.1%}) of all burned cells; "
            "interpret cropland_dominant results with this in mind."
        )
    for mask_name in ("burnable_tree_shrub_grass", "burnable_tree_shrub"):
        n_pos_mask = int((df.loc[population_masks[mask_name], TARGET_COLUMN] == 1).sum())
        if n_pos_mask < min_positives:
            warnings_list.append(
                f"{mask_name} positives < {min_positives} (n={n_pos_mask}); "
                "diagnostic-only population, skipped by default."
            )

    october_pos = int((
        (df[TARGET_COLUMN] == 1) & (df["burn_month"] == 10)
    ).sum())
    if october_pos < min_month_positives:
        warnings_list.append(
            f"October burn_month positives are low (n={october_pos} < "
            f"{min_month_positives}); October metrics may be null."
        )

    # --- Fail-fast: primary population must have a valid spatial CV plan ---
    y_all = df.loc[population_masks["all_valid"], TARGET_COLUMN].astype(int).to_numpy()
    if len(np.unique(y_all)) < 2:
        raise Step8BError("all_valid populasyonunda 'burned' tek sinif iceriyor.")
    try:
        make_spatial_folds(
            y_all,
            df.loc[population_masks["all_valid"], "spatial_block_id"].to_numpy(),
            n_splits, STEP8B_RANDOM_SEED,
        )
    except Step8BError:
        raise Step8BError(
            "Primer populasyon (all_valid) icin spatial-block CV "
            "olusturulamadi. Step8B durduruluyor (random split'e "
            "DUSULMEYECEK)."
        )

    # --- Train per population ---
    results: dict = {}
    for pop_name in POPULATION_ORDER:
        mask = population_masks[pop_name]
        df_pop = df.loc[mask].reset_index(drop=True)
        n_pos = int((df_pop[TARGET_COLUMN] == 1).sum())

        if pop_name in DIAGNOSTIC_ONLY_POPULATIONS and n_pos < min_positives and not allow_low_positive_strata:
            log.warning(
                "%s atlanyor (diagnostic-only, positives=%d < %d). "
                "--allow-low-positive-strata ile zorlanabilir.",
                pop_name, n_pos, min_positives,
            )
            results[pop_name] = {
                "skipped": True,
                "reason": f"diagnostic_only_low_positives (n_pos={n_pos} < {min_positives})",
            }
            continue

        log.info("Populasyon egitiliyor: %s (n=%d, burned=%d)", pop_name, len(df_pop), n_pos)
        res = train_population(
            df_pop, pop_name, n_splits, STEP8B_RANDOM_SEED, model_name,
            min_positives if not allow_low_positive_strata else 2,
        )
        if res.get("skipped"):
            log.warning("%s atlandi: %s", pop_name, res.get("reason"))
        else:
            res["_y"] = df_pop[TARGET_COLUMN].astype(int).to_numpy()
            log.info(
                "%s: AUC baseline=%.4f thermal=%.4f delta_auc=%+.4f (%s)",
                pop_name, res["overall_baseline"]["roc_auc"],
                res["overall_thermal"]["roc_auc"], res["delta_auc"],
                res["interpretation"],
            )
        results[pop_name] = res

    primary = results.get(STEP8B_PRIMARY_POPULATION)
    if primary is None or primary.get("skipped"):
        raise Step8BError(
            f"Primer populasyon '{STEP8B_PRIMARY_POPULATION}' egitilemedi: "
            f"{primary.get('reason') if primary else 'sonuc yok'}"
        )

    # --- Gap-fill sensitivity (existing OOF predictions only) ---
    sensitivity = {}
    for pop_name in SENSITIVITY_POPULATIONS:
        res = results.get(pop_name)
        if res is None or res.get("skipped"):
            continue
        df_pop = df.loc[population_masks[pop_name]].reset_index(drop=True)
        sensitivity[pop_name] = gapfill_sensitivity(
            df_pop, res["_y"], res["oof_prob_baseline"], res["oof_prob_thermal"]
        )

    # --- Write outputs ---
    predictions_df = build_predictions_table(df, results, population_masks)
    predictions_parquet = out_dir / "step8b_predictions.parquet"
    predictions_csv = out_dir / "step8b_predictions.csv"
    predictions_df.to_csv(predictions_csv, index=False)
    parquet_written = False
    try:
        predictions_df.to_parquet(predictions_parquet, index=False)
        parquet_written = True
    except (ImportError, ValueError) as exc:
        log.warning("Predictions parquet yazilamadi: %s (yalniz CSV yazildi)", exc)

    fold_metrics_path = write_fold_metrics_csv(results, out_dir)
    feature_importance_path = write_feature_importance_csv(results, out_dir)
    delta_auc_path = write_delta_auc_by_population_csv(results, out_dir)
    monthly_path = write_monthly_leadtime_csv(results, out_dir)

    roc_path = plot_roc_curves(results, out_dir)
    pr_path = plot_pr_curves(results, out_dir)

    stats_path = write_stats_json(
        out_dir, input_path, df, results, population_masks, sensitivity,
        feature_missing_counts, warnings_list,
        argparse.Namespace(
            spatial_block_size_cells=spatial_block_size_cells,
            n_splits=n_splits, model=model_name,
        ),
    )
    summary_path = write_summary_md(out_dir, results, sensitivity, stats_path, warnings_list)

    log.info("Stats: %s", stats_path)
    log.info("Summary: %s", summary_path)
    log.info("Predictions: %s (parquet_written=%s)", predictions_csv, parquet_written)
    if roc_path:
        log.info("ROC plot: %s", roc_path)
    if pr_path:
        log.info("PR plot: %s", pr_path)
    log.info("=" * 60)
    log.info("STEP 8B TAMAMLANDI (no FIRMS label used, MCD64A1 target, spatial-block CV)")
    log.info("=" * 60)

    return {
        "stats_path": str(stats_path),
        "summary_path": str(summary_path),
        "predictions_csv": str(predictions_csv),
        "predictions_parquet": str(predictions_parquet) if parquet_written else None,
        "fold_metrics_path": str(fold_metrics_path),
        "feature_importance_path": str(feature_importance_path),
        "delta_auc_path": str(delta_auc_path),
        "monthly_path": str(monthly_path),
    }


def run_step8b(ctx: dict | None = None, force: bool = False, **kwargs) -> dict:
    """
    Step8B: hucre seviyesinde baseline vs. baseline+thermal karsilastirmasini
    egitir (spatial-block CV).

    ctx: None ise (varsayilan) legacy Kozan davranisi BIREBIR korunur --
        outputs/step8b'ye yazar, outputs/step8a/'dan okur. Verilirse
        (Kozan-disi): outputs/experiments/<experiment_id>/step8b'ye yazar,
        outputs/experiments/<experiment_id>/step8a/'dan okur -- Kozan'in
        legacy paylasilan Step8A veri setine ASLA dokunulmaz.
    """
    use_ctx = ctx is not None and not ctx.get("is_kozan")
    output_dir_arg = str(ctx["step8b_output_dir"]) if use_ctx else STEP8B_OUTPUT_DIR
    input_arg = (
        str(ctx["step8a_output_dir"] / "step8a_500m_modeling_dataset.parquet")
        if use_ctx else None
    )
    result = main(input_arg=input_arg, output_dir_arg=output_dir_arg, force=force, **kwargs)
    if ctx is not None:
        result["experiment_id"] = ctx["experiment_id"]
    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step8B: baseline vs baseline+thermal burned-area model "
        "comparison on the Step8A 500 m MCD64A1-grid dataset (spatial-block CV)."
    )
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=STEP8B_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-splits", type=int, default=STEP8B_N_SPLITS)
    parser.add_argument("--spatial-block-size-cells", type=int, default=STEP8B_SPATIAL_BLOCK_SIZE_CELLS)
    parser.add_argument("--min-positives", type=int, default=STEP8B_MIN_POSITIVES_PER_POPULATION)
    parser.add_argument("--min-month-positives", type=int, default=STEP8B_MIN_MONTH_POSITIVES)
    parser.add_argument("--allow-low-positive-strata", action="store_true")
    parser.add_argument(
        "--model", type=str, default="random_forest",
        choices=["random_forest", "hist_gradient_boosting", "xgboost"],
    )
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
        allow_low_positive_strata=args.allow_low_positive_strata,
        model_name=args.model,
    )