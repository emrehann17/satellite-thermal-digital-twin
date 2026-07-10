"""
step7c_train_downscaling_model.py

SAF MODIS -> Landsat LST downscaling modeli eğitir (Step7B veri setini kullanır).

ÖNEMLİ:
    - Step7C bir DOWNSCALING modeli eğitir; FIRE-RISK modeli DEĞİLDİR.
    - MCD64A1 veya FIRMS etiketleri KULLANILMAZ; yanmış alan tahmini YAPILMAZ.
    - Step5 / Step5C / Step6 / Step7B çıktıları DEĞİŞTİRİLMEZ.
    - Target-türevi özellikler (anomaly_zscore, current_tvdi, tvdi_difference,
      modis_context_zscore) LEAKAGE riski nedeniyle özellik setinden HARİÇ tutulur.

Girdi:
    outputs/step7b/downscaling_training_samples.parquet (tercih)
    outputs/step7b/downscaling_training_samples.csv (fallback)

Çıktılar:
    outputs/step7c/downscaling_model.joblib
    outputs/step7c/downscaling_model_metadata.json
    outputs/step7c/downscaling_model_metrics.json
    outputs/step7c/downscaling_model_summary.md
    outputs/step7c/feature_importance.csv
    outputs/step7c/predicted_vs_actual.png
    outputs/step7c/residual_histogram.png
    outputs/step7c/residual_by_feature_summary.csv
    outputs/step7c/per_split_predictions_sample.csv
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from core.config import (
    STEP7B_MIN_TARGET_CELSIUS,
    STEP7B_MAX_TARGET_CELSIUS,
    STEP7C_RANDOM_SEED,
    STEP7C_MODEL_TYPE,
    STEP7C_TEST_SIZE,
    STEP7C_VAL_SIZE,
    STEP7C_SPLIT_MODE,
    STEP7C_SPATIAL_BLOCK_SIZE_PIXELS,
    STEP7C_EXCLUDE_LEAKAGE_FEATURES,
    STEP7C_MAX_TRAIN_SAMPLES,
    STEP7C_FAST_N_ESTIMATORS,
    STEP7C_RF_N_ESTIMATORS,
    STEP7C_RF_MIN_SAMPLES_LEAF,
    STEP7C_OUTPUT_DIR,
)
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT

BASE_DIR = PROJECT_ROOT
OUTPUTS_DIR = BASE_DIR / STEP7C_OUTPUT_DIR

log, log_file = setup_logger("step7c")

TARGET_COLUMN = "landsat_lst_celsius"

# Target'tan türetilebilecek, leakage riski taşıyan özellikler. Bunlar EĞİTİMDE
# KULLANILMAZ; yalnızca metadata'da "excluded_leakage_features" olarak kaydedilir.
LEAKAGE_FEATURES = [
    "anomaly_zscore",
    "current_tvdi",
    "tvdi_difference",
    "modis_context_zscore",
]

# Saf downscaling için güvenli (leakage-free) aday özellikler. Veri setinde
# bulunanlar kullanılır; eksik olanlar sessizce atlanır (metadata'ya yazılır).
SAFE_FEATURE_CANDIDATES = [
    "modis_lst_mean_celsius",
    "modis_lst_std_celsius",
    "ndvi",
    "elevation",
    "slope",
    "landcover",
    "lon",
    "lat",
    "row",
    "col",
    "row_norm",
    "col_norm",
]

PLOT_MAX_POINTS = 50_000


# =============================================================================
# 1. Veri yükleme
# =============================================================================
def load_dataset(ctx: dict | None = None) -> tuple[pd.DataFrame, Path]:
    """Step7B parquet/csv veri setini yükler (parquet tercih edilir).

    ctx: None ise (varsayılan) legacy outputs/step7b/. Verilirse (Kozan-dışı)
        ctx["step7b_output_dir"] -- legacy Kozan yoluna ASLA dokunmaz.
    """
    step7b_dir = ctx["step7b_output_dir"] if ctx is not None else (BASE_DIR / "outputs" / "step7b")
    parquet_path = step7b_dir / "downscaling_training_samples.parquet"
    csv_path = step7b_dir / "downscaling_training_samples.csv"

    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            log.info("Veri seti parquet'ten yüklendi: %s (%d satır)", parquet_path, len(df))
            return df, parquet_path
        except Exception as exc:  # noqa: BLE001
            log.warning("Parquet okunamadı (%s), CSV'ye düşülüyor.", exc)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        log.info("Veri seti CSV'den yüklendi: %s (%d satır)", csv_path, len(df))
        return df, csv_path

    raise SystemExit(
        "Step7B veri seti bulunamadı. Beklenen: "
        f"{parquet_path} veya {csv_path}. Önce Step7B'yi çalıştırın: "
        "python src/step7b_prepare_downscaling_dataset.py"
    )


# =============================================================================
# 2. Özellik mühendisliği + doğrulama (clamp YOK, yalnız DROP)
# =============================================================================
def engineer_and_validate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    """
    row_norm/col_norm ekler, güvenli özellik listesini belirler, geçersiz
    satırları (NaN target/feature, target aralık-dışı) düşürür. CLAMP YAPILMAZ.
    """
    drop_counts = {
        "dropped_nan_target": 0,
        "dropped_invalid_target_range": 0,
        "dropped_nan_required_features": 0,
    }

    if TARGET_COLUMN not in df.columns:
        raise SystemExit(f"Beklenen target kolonu bulunamadı: {TARGET_COLUMN}")

    # row_norm / col_norm (row/max(row), col/max(col))
    if "row" in df.columns and df["row"].max() > 0:
        df = df.copy()
        df["row_norm"] = df["row"] / df["row"].max()
    if "col" in df.columns and df["col"].max() > 0:
        df["col_norm"] = df["col"] / df["col"].max()

    excluded_present = [c for c in LEAKAGE_FEATURES if c in df.columns]
    safe_features = [c for c in SAFE_FEATURE_CANDIDATES if c in df.columns]

    if not STEP7C_EXCLUDE_LEAKAGE_FEATURES:
        safe_features += [c for c in excluded_present if c not in safe_features]
        log.warning(
            "STEP7C_EXCLUDE_LEAKAGE_FEATURES=False: leakage özellikleri özellik "
            "setine dahil edildi. Bu varsayılan/önerilen davranış DEĞİLDİR."
        )

    if not safe_features:
        raise SystemExit("Kullanılabilir güvenli (leakage-free) özellik bulunamadı.")

    n0 = len(df)
    target_nan = df[TARGET_COLUMN].isna()
    drop_counts["dropped_nan_target"] = int(target_nan.sum())
    df = df[~target_nan]

    in_range = (df[TARGET_COLUMN] >= STEP7B_MIN_TARGET_CELSIUS) & (
        df[TARGET_COLUMN] <= STEP7B_MAX_TARGET_CELSIUS
    )
    drop_counts["dropped_invalid_target_range"] = int((~in_range).sum())
    df = df[in_range]

    req_nan_mask = df[safe_features].isna().any(axis=1)
    drop_counts["dropped_nan_required_features"] = int(req_nan_mask.sum())
    df = df[~req_nan_mask]

    log.info(
        "Doğrulama sonrası: %d -> %d satır (target_nan=%d, target_range=%d, "
        "feature_nan=%d).",
        n0, len(df),
        drop_counts["dropped_nan_target"],
        drop_counts["dropped_invalid_target_range"],
        drop_counts["dropped_nan_required_features"],
    )

    if len(df) == 0:
        raise SystemExit("Doğrulama sonrası geçerli örnek kalmadı.")

    return df.reset_index(drop=True), safe_features, {
        **drop_counts,
        "excluded_leakage_features_present": excluded_present,
    }


# =============================================================================
# 3. Grouped split (spatial_block -> modis_pixel_group -> tile_group -> random)
# =============================================================================
def add_spatial_block_id(df: pd.DataFrame, block_size: int) -> pd.DataFrame:
    """
    row/col'dan sağlam bir mekansal blok kimliği (spatial_block_id) üretir.

    spatial_block_row = row // block_size
    spatial_block_col = col // block_size
    spatial_block_id  = "{block_row}_{block_col}"

    Aynı bloktaki (block_size x block_size piksellik alan) tüm örnekler AYNI
    split'te kalır; bu, modis_pixel_id'nin örnek-başına benzersiz çıkabildiği
    durumlarda bile gerçek mekansal ayrım sağlayan sağlam bir yöntemdir.
    """
    if "row" not in df.columns or "col" not in df.columns:
        return df
    df = df.copy()
    block_row = (df["row"] // block_size).astype("int64")
    block_col = (df["col"] // block_size).astype("int64")
    df["spatial_block_row"] = block_row
    df["spatial_block_col"] = block_col
    df["spatial_block_id"] = block_row.astype(str) + "_" + block_col.astype(str)
    return df


def summarize_samples_per_group(df: pd.DataFrame, group_col: str) -> dict:
    """Grup başına örnek sayısı için min/medyan/ortalama/max özet döndürür."""
    counts = df[group_col].value_counts()
    if counts.empty:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": int(counts.min()),
        "median": float(counts.median()),
        "mean": float(counts.mean()),
        "max": int(counts.max()),
    }


def grouped_split(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
    allow_random_split: bool,
    split_mode: str,
    spatial_block_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str | None, dict]:
    """
    Gruplara göre train/val/test ayırır (aynı grup aynı split'te kalır;
    leakage azaltılır).

    split_mode:
        "spatial_block"     -> row//block, col//block bazlı mekansal blok (VARSAYILAN,
                                sağlam; modis_pixel_id'nin örnek-başına benzersiz
                                çıktığı durumlarda bile gerçek gruplama sağlar).
        "modis_pixel_group" -> modis_pixel_id (Step7B çıktısına bağlı; pratikte
                                örnek-başına benzersiz olabilir, bu durumda
                                aşağıdaki sağlamlık kontrolü uyarı verir).
        "tile_group"         -> source_tile_id.
        "random"             -> yalnızca --allow-random-split ile; uyarı verir.
    """
    rng = np.random.default_rng(seed)
    warnings_list: list[str] = []

    group_col = None
    if split_mode == "spatial_block":
        df = add_spatial_block_id(df, spatial_block_size)
        if "spatial_block_id" in df.columns:
            group_col = "spatial_block_id"
    elif split_mode == "modis_pixel_group":
        if "modis_pixel_id" in df.columns and df["modis_pixel_id"].notna().any():
            group_col = "modis_pixel_id"
    elif split_mode == "tile_group":
        if "source_tile_id" in df.columns and df["source_tile_id"].notna().any():
            group_col = "source_tile_id"
    elif split_mode == "random":
        group_col = None
    else:
        raise SystemExit(f"Bilinmeyen split_mode: {split_mode}")

    # split_mode açıkça "random" istenmediyse ama grup kolonu kurulamadıysa,
    # yalnızca --allow-random-split ile rastgele fallback'e izin verilir.
    if group_col is None and split_mode != "random":
        if not allow_random_split:
            raise SystemExit(
                f"'{split_mode}' split modu için grup kolonu kurulamadı "
                "(gerekli row/col/modis_pixel_id/source_tile_id eksik olabilir). "
                "Rastgele piksel split'e düşmek için --allow-random-split verin "
                "(leakage riskini artırır)."
            )
        warnings_list.append(
            f"Group column for split_mode='{split_mode}' unavailable; "
            "falling back to RANDOM pixel split (--allow-random-split). "
            "This increases leakage risk between neighboring pixels."
        )

    samples_per_group = {"min": None, "median": None, "mean": None, "max": None}

    if group_col is None:
        if split_mode == "random" and not allow_random_split:
            raise SystemExit(
                "split_mode='random' seçildi ama --allow-random-split verilmedi. "
                "Rastgele split leakage riskini artırdığı için açıkça onaylanmalıdır."
            )
        if split_mode == "random":
            warnings_list.append(
                "Using RANDOM pixel split (--split random). This does NOT provide "
                "spatial/group separation and increases leakage risk between "
                "neighboring pixels."
            )
        idx = rng.permutation(len(df))
        n_test = int(len(df) * test_size)
        n_val = int(len(df) * val_size)
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]
        split_mode_used = "random_pixel_fallback" if split_mode != "random" else "random"
        group_info = {"train": None, "val": None, "test": None}
    else:
        samples_per_group = summarize_samples_per_group(df, group_col)
        n_groups = df[group_col].nunique()

        # --- Sağlamlık kontrolü (madde 5) ---
        if n_groups == len(df):
            warnings_list.append(
                "Group split is ineffective because each sample is its own group."
            )
            log.warning(
                "Group split is ineffective because each sample is its own group "
                "(group_col=%s, n_groups=%d == n_samples=%d).",
                group_col, n_groups, len(df),
            )
        elif split_mode == "spatial_block" and n_groups >= 0.9 * len(df):
            warnings_list.append(
                f"spatial_block split produced {n_groups} groups for {len(df)} "
                "samples (groups nearly equal to samples); consider increasing "
                "STEP7C_SPATIAL_BLOCK_SIZE_PIXELS for more effective grouping."
            )

        groups = df[group_col].unique()
        rng.shuffle(groups)
        n_test_g = max(1, int(n_groups * test_size))
        n_val_g = max(1, int(n_groups * val_size))
        if n_groups < 10:
            warnings_list.append(
                f"Very few groups ({n_groups}) for '{group_col}' grouped split; "
                "validation/test splits may be unstable."
            )
        test_groups = set(groups[:n_test_g])
        val_groups = set(groups[n_test_g:n_test_g + n_val_g])
        train_groups = set(groups[n_test_g + n_val_g:])

        test_idx = df.index[df[group_col].isin(test_groups)].to_numpy()
        val_idx = df.index[df[group_col].isin(val_groups)].to_numpy()
        train_idx = df.index[df[group_col].isin(train_groups)].to_numpy()
        split_mode_used = split_mode
        group_info = {
            "train": len(train_groups), "val": len(val_groups), "test": len(test_groups),
        }

    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        if len(split_df) == 0:
            warnings_list.append(f"Split '{name}' is empty after grouping.")

    log.info(
        "Split (%s, group_col=%s): train=%d val=%d test=%d (groups: %s) "
        "samples_per_group=%s",
        split_mode_used, group_col, len(train_df), len(val_df), len(test_df),
        group_info, samples_per_group,
    )
    return train_df, val_df, test_df, split_mode_used, group_col, {
        "group_counts": group_info,
        "samples_per_group": samples_per_group,
        "warnings": warnings_list,
    }


# =============================================================================
# 4. Model eğitimi
# =============================================================================
def build_model(model_type: str, fast: bool, seed: int):
    """model_type'a göre regressor kurar (RF varsayılan, sklearn-only)."""
    n_estimators = STEP7C_FAST_N_ESTIMATORS if fast else STEP7C_RF_N_ESTIMATORS

    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=STEP7C_RF_MIN_SAMPLES_LEAF,
            n_jobs=-1,
            random_state=seed,
        )
    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=n_estimators * 2 if fast else 200,
            random_state=seed,
        )
    if model_type == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise SystemExit(
                "xgboost kurulu değil. requirements'a eklemeden --model xgboost "
                "kullanılamaz; sklearn tabanlı random_forest veya "
                "hist_gradient_boosting kullanın."
            ) from exc
        return xgb.XGBRegressor(
            n_estimators=n_estimators if fast else 300,
            random_state=seed,
            n_jobs=-1,
        )
    raise SystemExit(f"Bilinmeyen model_type: {model_type}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE/MAE/R2/bias/medyan mutlak hata/residual std/n döndürür."""
    if len(y_true) == 0:
        return {
            "rmse": None, "mae": None, "r2": None, "bias": None,
            "median_abs_error": None, "residual_std": None, "sample_count": 0,
        }
    residual = y_pred - y_true
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
        "bias": float(np.mean(residual)),
        "median_abs_error": float(median_absolute_error(y_true, y_pred)),
        "residual_std": float(np.std(residual)),
        "sample_count": int(len(y_true)),
    }


def improvement(model_metrics: dict, baseline_metrics: dict) -> dict:
    """Model'in baseline'a göre RMSE/MAE/R2 iyileşmesini hesaplar (%)."""
    out = {}
    for key in ("rmse", "mae"):
        m, b = model_metrics.get(key), baseline_metrics.get(key)
        if m is not None and b not in (None, 0):
            out[f"{key}_improvement_pct"] = float(100.0 * (b - m) / b)
        else:
            out[f"{key}_improvement_pct"] = None
    m_r2, b_r2 = model_metrics.get("r2"), baseline_metrics.get("r2")
    out["r2_improvement"] = (
        float(m_r2 - b_r2) if m_r2 is not None and b_r2 is not None else None
    )
    return out


# =============================================================================
# 5. Feature importance
# =============================================================================
def compute_feature_importance(
    model, feature_names: list[str], val_df: pd.DataFrame, model_type: str
) -> pd.DataFrame:
    """RF/HGB için feature_importances_; yoksa val alt kümesinde permutation importance."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False).reset_index(drop=True)

    try:
        from sklearn.inspection import permutation_importance
        sample = val_df.sample(
            n=min(2000, len(val_df)), random_state=STEP7C_RANDOM_SEED
        ) if len(val_df) > 0 else val_df
        if len(sample) == 0:
            return pd.DataFrame({"feature": feature_names, "importance": np.nan})
        result = permutation_importance(
            model, sample[feature_names], sample[TARGET_COLUMN],
            n_repeats=3, random_state=STEP7C_RANDOM_SEED, n_jobs=-1,
        )
        return pd.DataFrame(
            {"feature": feature_names, "importance": result.importances_mean}
        ).sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("Feature importance hesaplanamadı (%s): %s", model_type, exc)
        return pd.DataFrame({"feature": feature_names, "importance": np.nan})


# =============================================================================
# 6. Plotlar
# =============================================================================
def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    rng = np.random.default_rng(STEP7C_RANDOM_SEED)
    n = len(y_true)
    if n > PLOT_MAX_POINTS:
        idx = rng.choice(n, size=PLOT_MAX_POINTS, replace=False)
        y_true, y_pred = y_true[idx], y_pred[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=4, alpha=0.3, color="#1f77b4")
    lims = [
        min(np.min(y_true), np.min(y_pred)) if n else 0,
        max(np.max(y_true), np.max(y_pred)) if n else 1,
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="1:1")
    ax.set_xlabel("Actual Landsat LST (°C)")
    ax.set_ylabel("Predicted Landsat LST (°C)")
    ax.set_title("Predicted vs Actual (test split)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_residual_histogram(residual: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residual, bins=60, color="#ff7f0e", alpha=0.8)
    ax.axvline(0, color="k", linewidth=1)
    ax.set_xlabel("Residual (prediction - actual), °C")
    ax.set_ylabel("Count")
    ax.set_title("Residual histogram (test split)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def residual_by_feature_summary(
    test_df: pd.DataFrame, residual: np.ndarray, features: list[str]
) -> pd.DataFrame:
    """residual vs elevation / ndvi gibi özellikler için binned özet tablo."""
    rows = []
    for feat in [f for f in ("elevation", "ndvi") if f in features]:
        vals = test_df[feat].to_numpy()
        try:
            bins = pd.qcut(vals, q=min(5, max(1, len(np.unique(vals)))), duplicates="drop")
        except Exception:  # noqa: BLE001
            continue
        tmp = pd.DataFrame({"bin": bins, "residual": residual})
        grouped = tmp.groupby("bin", observed=True)["residual"].agg(
            ["mean", "std", "count"]
        ).reset_index()
        grouped.insert(0, "feature", feat)
        grouped = grouped.rename(
            columns={"mean": "residual_mean", "std": "residual_std", "count": "n"}
        )
        grouped["bin"] = grouped["bin"].astype(str)
        rows.append(grouped)
    if not rows:
        return pd.DataFrame(columns=["feature", "bin", "residual_mean", "residual_std", "n"])
    return pd.concat(rows, ignore_index=True)


# =============================================================================
# 7. Ana akış
# =============================================================================
def _modis_spatial_calibration_note(ctx: dict | None) -> str:
    """
    MODIS mean/std katmanlarinin ne oldugunu (ve ne OLMADIGINI) aciklayan
    metadata notu. Kozan (ctx=None veya ctx["is_kozan"]) icin legacy metin
    BIREBIR korunur (coklu-yil yaz-ortalamasi baseline). Kozan-disi bir
    deney icin (or. manavgat_2021), MODIS aslinda o deneyin PREDICTOR
    penceresi icin tek-sezonluk (single-season) export edildigi icin
    ("coklu-yil baseline" DEGIL -- bkz. scripts/prepare_modis_for_step7.py),
    metin bunu acikca yansitir.
    """
    if ctx is None or ctx.get("is_kozan"):
        return (
            "modis_lst_mean_celsius is a 4-year summer-mean MODIS context layer, "
            "not a current daily MODIS observation. This is a spatial "
            "downscaling/context calibration prototype, not yet daily MODIS "
            "downscaling."
        )
    return (
        "modis_lst_mean_celsius and modis_lst_std_celsius are single-season "
        "MODIS predictor-window summary layers for "
        f"{ctx['predictor_start_date']} -> {ctx['predictor_end_date']}; they "
        "are not multi-year baselines and not daily MODIS products."
    )


def main(
    model_type: str = STEP7C_MODEL_TYPE,
    fast: bool = False,
    max_train_samples: int | None = STEP7C_MAX_TRAIN_SAMPLES,
    force: bool = False,
    allow_random_split: bool = False,
    input_path: str | None = None,
    split_mode: str = STEP7C_SPLIT_MODE,
    spatial_block_size: int = STEP7C_SPATIAL_BLOCK_SIZE_PIXELS,
    ctx: dict | None = None,
) -> dict:
    log.info("=" * 60)
    log.info("STEP 7C BAŞLIYOR (pure MODIS->Landsat LST downscaling model)")
    log.info("=" * 60)

    existing = [
        OUTPUTS_DIR / "downscaling_model.joblib",
        OUTPUTS_DIR / "downscaling_model_metrics.json",
        OUTPUTS_DIR / "downscaling_model_metadata.json",
        OUTPUTS_DIR / "downscaling_model_summary.md",
    ]
    if any(p.exists() for p in existing) and not force:
        present = [p.name for p in existing if p.exists()]
        raise SystemExit(
            "Step7C çıktıları zaten var (" + ", ".join(present)
            + "). Üzerine yazmak için --force verin."
        )
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    warnings_list: list[str] = []

    if input_path:
        df = pd.read_parquet(input_path) if input_path.endswith(".parquet") \
            else pd.read_csv(input_path)
        used_path = Path(input_path)
        log.info("Özel girdi kullanıldı: %s (%d satır)", used_path, len(df))
    else:
        df, used_path = load_dataset()

    df, safe_features, drop_stats = engineer_and_validate(df)

    train_df, val_df, test_df, split_mode_used, group_col, split_info = grouped_split(
        df, STEP7C_TEST_SIZE, STEP7C_VAL_SIZE, STEP7C_RANDOM_SEED, allow_random_split,
        split_mode=split_mode, spatial_block_size=spatial_block_size,
    )
    warnings_list.extend(split_info["warnings"])

    if max_train_samples is not None and len(train_df) > max_train_samples:
        train_df = train_df.sample(
            n=max_train_samples, random_state=STEP7C_RANDOM_SEED
        ).reset_index(drop=True)
        log.info("Train seti max_train_samples=%d ile sınırlandı.", max_train_samples)

    if len(train_df) == 0:
        raise SystemExit("Train split boş; eğitim yapılamaz.")

    X_train, y_train = train_df[safe_features], train_df[TARGET_COLUMN].to_numpy()
    X_val, y_val = val_df[safe_features], val_df[TARGET_COLUMN].to_numpy()
    X_test, y_test = test_df[safe_features], test_df[TARGET_COLUMN].to_numpy()

    log.info(
        "Model eğitimi: type=%s fast=%s features=%s", model_type, fast, safe_features
    )
    model = build_model(model_type, fast, STEP7C_RANDOM_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    pred_train = model.predict(X_train) if len(X_train) else np.array([])
    pred_val = model.predict(X_val) if len(X_val) else np.array([])
    pred_test = model.predict(X_test) if len(X_test) else np.array([])

    metrics_train = compute_metrics(y_train, pred_train)
    metrics_val = compute_metrics(y_val, pred_val)
    metrics_test = compute_metrics(y_test, pred_test)

    if "modis_lst_mean_celsius" in test_df.columns:
        modis_baseline_pred = test_df["modis_lst_mean_celsius"].to_numpy()
        modis_baseline_metrics = compute_metrics(y_test, modis_baseline_pred)
    else:
        modis_baseline_metrics = compute_metrics(np.array([]), np.array([]))
        warnings_list.append("modis_lst_mean_celsius not found; MODIS baseline skipped.")

    train_mean = float(np.mean(y_train)) if len(y_train) else float("nan")
    train_mean_pred = np.full(len(y_test), train_mean)
    train_mean_baseline_metrics = compute_metrics(y_test, train_mean_pred)

    improvement_over_modis = improvement(metrics_test, modis_baseline_metrics)

    if metrics_test.get("rmse") is not None and modis_baseline_metrics.get("rmse") is not None:
        if metrics_test["rmse"] > modis_baseline_metrics["rmse"]:
            warnings_list.append(
                "Model test RMSE is WORSE than the direct MODIS baseline "
                f"(model={metrics_test['rmse']:.3f} vs "
                f"baseline={modis_baseline_metrics['rmse']:.3f}). Reporting honestly."
            )

    if metrics_train.get("rmse") is not None and metrics_test.get("rmse") is not None:
        if metrics_train["rmse"] > 0 and metrics_test["rmse"] > 1.5 * metrics_train["rmse"]:
            warnings_list.append(
                "Train RMSE is much lower than test RMSE "
                f"(train={metrics_train['rmse']:.3f}, test={metrics_test['rmse']:.3f}); "
                "possible overfitting."
            )

    fi_df = compute_feature_importance(model, safe_features, val_df, model_type)
    fi_path = OUTPUTS_DIR / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)

    pva_path = OUTPUTS_DIR / "predicted_vs_actual.png"
    hist_path = OUTPUTS_DIR / "residual_histogram.png"
    if len(y_test):
        plot_predicted_vs_actual(y_test, pred_test, pva_path)
        plot_residual_histogram(pred_test - y_test, hist_path)
    else:
        warnings_list.append("Test split empty; plots not generated.")

    resid_summary_path = OUTPUTS_DIR / "residual_by_feature_summary.csv"
    if len(y_test):
        resid_df = residual_by_feature_summary(test_df, pred_test - y_test, safe_features)
    else:
        resid_df = pd.DataFrame(columns=["feature", "bin", "residual_mean", "residual_std", "n"])
    resid_df.to_csv(resid_summary_path, index=False)

    sample_path = OUTPUTS_DIR / "per_split_predictions_sample.csv"
    _write_prediction_sample(
        sample_path, train_df, val_df, test_df, pred_train, pred_val, pred_test
    )

    model_path = OUTPUTS_DIR / "downscaling_model.joblib"
    joblib.dump({"model": model, "feature_names": safe_features}, model_path)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "input_dataset_path": str(used_path),
        "model_type": model_type,
        "fast_mode": fast,
        "target_column": TARGET_COLUMN,
        "safe_feature_columns": safe_features,
        "excluded_leakage_features": LEAKAGE_FEATURES,
        "excluded_leakage_features_present_in_dataset": drop_stats[
            "excluded_leakage_features_present"
        ],
        "leakage_guard_enabled": bool(STEP7C_EXCLUDE_LEAKAGE_FEATURES),
        "split_mode": split_mode_used,
        "group_column": group_col,
        "train_group_count": split_info["group_counts"].get("train"),
        "val_group_count": split_info["group_counts"].get("val"),
        "test_group_count": split_info["group_counts"].get("test"),
        "samples_per_group": split_info.get("samples_per_group"),
        "spatial_block_size_pixels": spatial_block_size if split_mode_used == "spatial_block" else None,
        "train_sample_count": int(len(train_df)),
        "val_sample_count": int(len(val_df)),
        "test_sample_count": int(len(test_df)),
        "random_seed": STEP7C_RANDOM_SEED,
        "model_params": _model_params(model),
        "sklearn_version": _safe_version("sklearn"),
        "pandas_version": _safe_version("pandas"),
        "python_version": platform.python_version(),
        "no_fire_risk_model_trained": True,
        "no_burned_area_labels_used": True,
        "leakage_guard": True,
        "drop_stats": drop_stats,
        "output_files": {
            "model": str(model_path),
            "feature_importance": str(fi_path),
            "predicted_vs_actual_plot": str(pva_path) if len(y_test) else None,
            "residual_histogram_plot": str(hist_path) if len(y_test) else None,
            "residual_by_feature_summary": str(resid_summary_path),
            "per_split_predictions_sample": str(sample_path),
        },
        "spatial_calibration_note": _modis_spatial_calibration_note(ctx),
        "warnings": warnings_list,
    }
    metadata_path = OUTPUTS_DIR / "downscaling_model_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    metrics_payload = {
        "created_at": datetime.now().isoformat(),
        "train": metrics_train,
        "validation": metrics_val,
        "test": metrics_test,
        "modis_baseline": modis_baseline_metrics,
        "train_mean_baseline": train_mean_baseline_metrics,
        "improvement_over_modis_baseline": improvement_over_modis,
        "warnings": warnings_list,
    }
    metrics_path = OUTPUTS_DIR / "downscaling_model_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary_path = write_summary_markdown(metadata, metrics_payload, fi_df, safe_features)

    log.info("Model: %s", model_path)
    log.info("Metrics: %s", metrics_path)
    log.info("Metadata: %s", metadata_path)
    log.info("Summary: %s", summary_path)
    log.info(
        "Test RMSE=%.3f vs MODIS baseline RMSE=%.3f",
        metrics_test.get("rmse") or float("nan"),
        modis_baseline_metrics.get("rmse") or float("nan"),
    )
    log.info("=" * 60)
    log.info("STEP 7C TAMAMLANDI (no fire-risk model trained)")
    log.info("=" * 60)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metadata_path": str(metadata_path),
        "summary_path": str(summary_path),
        "metrics": metrics_payload,
    }


def _model_params(model) -> dict:
    try:
        params = model.get_params()
        return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                for k, v in params.items()}
    except Exception:  # noqa: BLE001
        return {}


def _safe_version(module_name: str) -> str | None:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", None)
    except ImportError:
        return None


def _write_prediction_sample(
    path: Path,
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    pred_train: np.ndarray, pred_val: np.ndarray, pred_test: np.ndarray,
    max_per_split: int = 5000,
) -> None:
    """Her split için küçük bir örnek tahmin tablosu yazar (opsiyonel çıktı)."""
    frames = []
    for name, split_df, preds in (
        ("train", train_df, pred_train), ("val", val_df, pred_val), ("test", test_df, pred_test),
    ):
        if len(split_df) == 0:
            continue
        n = min(max_per_split, len(split_df))
        idx = np.random.default_rng(STEP7C_RANDOM_SEED).choice(
            len(split_df), size=n, replace=False
        )
        sub = split_df.iloc[idx][["row", "col", "lon", "lat", TARGET_COLUMN]].copy()
        sub["predicted_" + TARGET_COLUMN] = preds[idx]
        sub["split"] = name
        frames.append(sub)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    else:
        pd.DataFrame(
            columns=["row", "col", "lon", "lat", TARGET_COLUMN,
                     "predicted_" + TARGET_COLUMN, "split"]
        ).to_csv(path, index=False)


# =============================================================================
# 8. Summary markdown
# =============================================================================
def write_summary_markdown(
    metadata: dict, metrics: dict, fi_df: pd.DataFrame, safe_features: list[str]
) -> Path:
    def fmt(v, digits=3):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.{digits}f}"
        return str(v)

    def metrics_row(label: str, m: dict) -> str:
        return (
            f"| {label} | {m.get('sample_count')} | {fmt(m.get('rmse'))} | "
            f"{fmt(m.get('mae'))} | {fmt(m.get('r2'))} | {fmt(m.get('bias'))} | "
            f"{fmt(m.get('median_abs_error'))} | {fmt(m.get('residual_std'))} |"
        )

    lines = [
        "# Step7C: Pure MODIS-to-Landsat LST Downscaling Model",
        "",
        "**Step7C trains a pure MODIS-to-Landsat LST downscaling model. "
        "No burned-area or FIRMS labels are used. This is not a fire-risk model.**",
        "",
        f"- Created at: `{metadata['created_at']}`",
        f"- Model type: `{metadata['model_type']}`" + (" (fast mode)" if metadata["fast_mode"] else ""),
        f"- Input dataset: `{metadata['input_dataset_path']}`",
        f"- Target column: `{metadata['target_column']}`",
        "",
        "## Leakage guard",
        "",
        "Target-derived features are **excluded** to prevent leakage: "
        f"`{', '.join(metadata['excluded_leakage_features'])}`.",
        f"- Present in dataset but excluded from training: "
        f"`{', '.join(metadata['excluded_leakage_features_present_in_dataset']) or 'none'}`",
        f"- Selected (safe) features used for training: `{', '.join(safe_features)}`",
        "",
        "## Split",
        "",
        (
            "**This is spatial block validation**: samples are grouped into "
            f"`{metadata.get('spatial_block_size_pixels')}`x"
            f"`{metadata.get('spatial_block_size_pixels')}` pixel spatial blocks "
            "(`spatial_block_id`), and each block is assigned entirely to one "
            "of train/validation/test. This tests generalization to unseen "
            "spatial regions within this AOI/window, not just unseen pixels."
            if metadata["split_mode"] == "spatial_block" else
            f"Split mode: `{metadata['split_mode']}` (group column: "
            f"`{metadata['group_column']}`)."
        ),
        f"- Train samples: `{metadata['train_sample_count']}` "
        f"(groups: `{metadata['train_group_count']}`)",
        f"- Validation samples: `{metadata['val_sample_count']}` "
        f"(groups: `{metadata['val_group_count']}`)",
        f"- Test samples: `{metadata['test_sample_count']}` "
        f"(groups: `{metadata['test_group_count']}`)",
        f"- Samples per group (whole dataset) — min: "
        f"`{fmt(metadata.get('samples_per_group', {}).get('min'), 1)}`, "
        f"median: `{fmt(metadata.get('samples_per_group', {}).get('median'), 1)}`, "
        f"mean: `{fmt(metadata.get('samples_per_group', {}).get('mean'), 2)}`, "
        f"max: `{fmt(metadata.get('samples_per_group', {}).get('max'), 1)}`",
        "",
        "## Metrics",
        "",
        "| Split | N | RMSE | MAE | R2 | Bias | Median AE | Residual Std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        metrics_row("Train", metrics["train"]),
        metrics_row("Validation", metrics["validation"]),
        metrics_row("Test", metrics["test"]),
        metrics_row("MODIS baseline (test)", metrics["modis_baseline"]),
        metrics_row("Train-mean baseline (test)", metrics["train_mean_baseline"]),
        "",
        "## Improvement over MODIS baseline (test)",
        "",
        f"- RMSE improvement: `{fmt(metrics['improvement_over_modis_baseline'].get('rmse_improvement_pct'))}%`",
        f"- MAE improvement: `{fmt(metrics['improvement_over_modis_baseline'].get('mae_improvement_pct'))}%`",
        f"- R2 improvement: `{fmt(metrics['improvement_over_modis_baseline'].get('r2_improvement'))}`",
        "",
        "> " + metadata["spatial_calibration_note"],
        "",
        "## Feature importance",
        "",
        "| Feature | Importance |",
        "| --- | ---: |",
    ]
    for _, row in fi_df.iterrows():
        lines.append(f"| {row['feature']} | {fmt(row['importance'], 5)} |")

    lines.extend([
        "",
        "## Limitations",
        "",
        "- Single AOI (Kozan / East Mediterranean), single current pre-fire window.",
        "- Dataset sampled from one season/window only; not tested across years.",
        "- `modis_lst_mean_celsius` is a multi-year summer-mean context layer, "
        "not a current daily MODIS observation — this is a spatial "
        "downscaling/context calibration prototype, not yet daily MODIS "
        "downscaling/gap-filling.",
        "- No claim of validated generalization beyond this AOI/window.",
    ])

    if metadata.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {w}" for w in metadata["warnings"])

    path = OUTPUTS_DIR / "downscaling_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _resolve_step7b_dataset_path(step7b_dir: Path) -> Path:
    """load_dataset() ile aynı öncelik sırası (parquet -> csv); yalnızca yol döner."""
    parquet_path = step7b_dir / "downscaling_training_samples.parquet"
    csv_path = step7b_dir / "downscaling_training_samples.csv"
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    raise SystemExit(
        f"Step7B veri seti bulunamadı: {parquet_path} veya {csv_path}. "
        "Önce Step7B'yi (namespaced) çalıştırın."
    )


def run_step7c(ctx: dict | None = None, force: bool = False, **kwargs) -> dict:
    """
    Step7C: yalnızca MODIS->Landsat LST downscaling modelini eğitir.

    ctx: None ise (varsayılan) legacy Kozan davranışı BİREBİR korunur --
        outputs/step7b/'den okur, outputs/step7c/'ye yazar. Verilirse
        (Kozan-dışı), ctx["step7b_output_dir"]'den okur (main()'in zaten var
        olan `input_path` parametresi ile -- load_dataset()'in kendisi
        DEĞİŞTİRİLMEDEN), ctx["step7c_output_dir"]'e yazar.

    Bu ADIM, burned-area/fire-risk modeli DEĞİLDİR -- yalnızca saf
    MODIS->Landsat LST downscaling modelidir (bkz. STEP7C_EXCLUDE_LEAKAGE_FEATURES).
    """
    global OUTPUTS_DIR

    use_ctx = ctx is not None and not ctx.get("is_kozan")
    saved = OUTPUTS_DIR
    try:
        input_path = None
        if use_ctx:
            OUTPUTS_DIR = ctx["step7c_output_dir"]
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            input_path = str(_resolve_step7b_dataset_path(ctx["step7b_output_dir"]))
            log.info(
                "[experiment=%s] Step7C ctx override aktif. output_dir=%s, "
                "input_path=%s", ctx["experiment_id"], OUTPUTS_DIR, input_path,
            )
        result = main(force=force, input_path=input_path, ctx=ctx if use_ctx else None, **kwargs)
        if ctx is not None:
            result["experiment_id"] = ctx["experiment_id"]
        return result
    finally:
        OUTPUTS_DIR = saved


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step7C: pure MODIS-to-Landsat LST downscaling model training "
        "(no fire-risk model, no burned-area labels)."
    )
    parser.add_argument(
        "--model", choices=["random_forest", "hist_gradient_boosting", "xgboost"],
        default=STEP7C_MODEL_TYPE,
    )
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=STEP7C_MAX_TRAIN_SAMPLES)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-random-split", action="store_true")
    parser.add_argument(
        "--split",
        choices=["spatial_block", "modis_pixel_group", "tile_group", "random"],
        default=STEP7C_SPLIT_MODE,
        help="Primary train/val/test split strategy (default: spatial_block).",
    )
    parser.add_argument(
        "--spatial-block-size", type=int, default=STEP7C_SPATIAL_BLOCK_SIZE_PIXELS,
        help="Spatial block size in pixels for --split spatial_block.",
    )
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.output_dir:
        OUTPUTS_DIR = Path(args.output_dir)  # noqa: PLW0603
    main(
        model_type=args.model,
        fast=args.fast,
        max_train_samples=args.max_train_samples,
        force=args.force,
        allow_random_split=args.allow_random_split,
        input_path=args.input,
        split_mode=args.split,
        spatial_block_size=args.spatial_block_size,
    )