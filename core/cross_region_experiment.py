"""
core/cross_region_experiment.py

Step9F ("exploratory cross-region feature-representation experiment") ve
gelecekteki benzer cross-region deneyleri icin PAYLASILAN, bilimsel-olmayan
YARDIMCI fonksiyonlari barindirir. Bu modul var olan Step8B/Step9A/Step9B
mantigini YENIDEN UYGULAMAZ -- yalnizca onlarin ZATEN mevcut fonksiyonlarini
(build_pipeline, make_spatial_folds, load_step8a_dataset, population_subset,
compute_metrics_at_threshold) reuse eder ve Step9F'in ihtiyac duydugu (ama
Step9B'de birebir mevcut olmayan) birkac kucuk orkestrasyon fonksiyonunu
(source OOF + esik secimi TEK gecis, region-relative robust transform,
paired spatial-block bootstrap) ekler.

Step9A/Step9B/Step9C/Step9D/Step9E dosyalari bu modulden HICBIR SEKILDE
degistirilmez/yeniden yazilmaz -- yalnizca ithal edilir (import) veya
salt-okunur olarak okunur.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from core.config import (
    STEP8B_MIN_POSITIVES_PER_POPULATION,
    STEP8B_N_SPLITS,
    STEP8B_RANDOM_SEED,
    STEP8B_SPATIAL_BLOCK_SIZE_CELLS,
)
from src.step8b_train_baseline_vs_thermal_model import build_pipeline, make_spatial_folds
from src.step9a_audit_cross_region_inputs import (
    ALL_POPULATIONS,
    FORBIDDEN_MODEL_COLUMNS,
    PRIMARY_POPULATIONS,
    SECONDARY_POPULATIONS,
    SHARED_BASELINE_FEATURES,
    SHARED_THERMAL_FEATURES,
    TARGET_COLUMN,
    cross_region_output_root,
    resolve_step8a_dataset_path,
)
from src.step9b_run_cross_region_transfer import (
    compute_metrics_at_threshold,
    load_step8a_dataset,
    population_subset,
)

EPSILON_IQR = 1e-6
F1_THRESHOLD_GRID = np.linspace(0.05, 0.95, 19)  # step9b:select_threshold_from_source_oof ile AYNI grid

# =============================================================================
# Sabit feature varyantlari (kod icinde TEK YERDE dondurulmus -- prompt'ta
# "the variant family must remain fixed" gereği CLI'dan degistirilemez).
# SHARED_BASELINE_FEATURES / SHARED_THERMAL_FEATURES, Step9A'daki TEK KAYNAK'tan
# (src/step9a_audit_cross_region_inputs.py) alinir; burada YENIDEN TANIMLANMAZ.
# =============================================================================
CATEGORICAL_FEATURES = ["landcover_dominant"]
ORIGINAL_THERMAL_FEATURES = list(SHARED_BASELINE_FEATURES) + list(SHARED_THERMAL_FEATURES)


def _minus(features: list[str], *drop: str) -> list[str]:
    drop_set = set(drop)
    return [f for f in features if f not in drop_set]


FIXED_VARIANTS: dict[str, list[str]] = {
    "original_baseline": list(SHARED_BASELINE_FEATURES),
    "original_thermal": list(ORIGINAL_THERMAL_FEATURES),
    "thermal_without_elevation": _minus(ORIGINAL_THERMAL_FEATURES, "elevation_mean"),
    "thermal_without_absolute_lst": _minus(
        ORIGINAL_THERMAL_FEATURES, "current_lst_mean", "downscaled_lst_mean", "fused_lst_mean",
    ),
    "thermal_without_tvdi_difference": _minus(ORIGINAL_THERMAL_FEATURES, "tvdi_difference_mean"),
    "thermal_without_elevation_or_absolute_lst": _minus(
        ORIGINAL_THERMAL_FEATURES,
        "elevation_mean", "current_lst_mean", "downscaled_lst_mean", "fused_lst_mean",
    ),
    "stable_core": ["ndvi_mean", "slope_mean", "landcover_dominant", "lst_anomaly_mean", "current_tvdi_mean"],
    "stable_core_without_landcover": ["ndvi_mean", "slope_mean", "lst_anomaly_mean", "current_tvdi_mean"],
}

VARIANT_PURPOSE: dict[str, str] = {
    "original_baseline": "Step9 baseline referansini yeniden uretir (reprodüksiyon kontrolü).",
    "original_thermal": "Step9 thermal referansini yeniden uretir (reprodüksiyon kontrolü + Regime A/B ortak referans).",
    "thermal_without_elevation": "Bolgeye-ozgu elevation iliski-yonu tersine donmesine duyarliligi test eder.",
    "thermal_without_absolute_lst": "Mutlak-sicaklik esiklerinin bolgeye-ozgu transfer basarisizligini surukleyip suruklemedigini test eder.",
    "thermal_without_tvdi_difference": "En guclu kayan ve yon-kararsiz dryness-difference feature'ina duyarliligi test eder.",
    "thermal_without_elevation_or_absolute_lst": "Elevation + mutlak LST feature'larinin BIRLIKTE cikarilmasini test eder.",
    "stable_core": "Karsilastirmali olarak daha stabil feature'lari tutan kompakt, hipotez-guduml bir temsil.",
    "stable_core_without_landcover": "Farkli landcover kompozisyonunun transfer istikrarsizligina katkisini test eder.",
}

# Regime B (region-relative) YALNIZCA bu iki varyant icin calistirilir (prompt geregi).
REGIME_B_VARIANTS = ["original_thermal", "stable_core"]

REGIME_A_LABEL = "strict_source_only_inductive_transfer"
# Regime B'nin IKI acik etiketi (prompt geregi HER IKISI de kullanilmali; asla
# "source-only" / "direct transfer" / "unbiased external transfer" DENMEZ).
REGIME_B_LABELS = ["unsupervised_target_covariate_adaptation", "transductive_region_relative_representation"]

PRIMARY_REFERENCE_VARIANT = "original_thermal"
BASELINE_REFERENCE_VARIANT = "original_baseline"

REPRODUCTION_TOLERANCE = {"roc_auc": 1e-6, "pr_auc": 1e-6, "brier_score": 1e-6}


def check_no_forbidden_features(feature_list: list[str]) -> None:
    leaked = set(feature_list).intersection(FORBIDDEN_MODEL_COLUMNS)
    if leaked:
        raise ValueError(f"YASAK kolonlar Step9F feature varyantina sizmis: {leaked}.")


for _variant_name, _feature_list in FIXED_VARIANTS.items():
    check_no_forbidden_features(_feature_list)


# =============================================================================
# Path resolucion (Step9F, cross_region_output_root/step9f altinda) +
# namespacing guvenlik kontrolu
# =============================================================================
def step9f_output_dir(source_id: str, target_id: str) -> Path:
    return cross_region_output_root(source_id, target_id) / "step9f"


def resolve_step9_stage_dir(source_id: str, target_id: str, stage: str) -> Path:
    return cross_region_output_root(source_id, target_id) / stage


def assert_paths_are_safely_namespaced(source_id: str, target_id: str, path: Path) -> None:
    """Step9E ile AYNI konvansiyon: Step9F'in YALNIZCA kendi (source, target)
    ciftinin namespaced dizinlerine okuma/yazma yaptigini dogrular."""
    parts = Path(path).parts
    pair_token = f"{source_id}__{target_id}"
    if "cross_region" in parts and pair_token not in parts:
        raise ValueError(
            f"Namespacing guvenlik kontrolu BASARISIZ: '{path}' beklenen "
            f"'{pair_token}' cross-region dizinine ait degil."
        )
    if "experiments" in parts and source_id not in parts and target_id not in parts:
        raise ValueError(
            f"Namespacing guvenlik kontrolu BASARISIZ: '{path}' ne "
            f"'{source_id}' ne de '{target_id}' deney dizinine ait."
        )


# =============================================================================
# Regime B: region-relative robust normalizasyon
#
# ONEMLI: Bu fonksiyonlar burned/target etiketlerini HICBIR SEKILDE kullanmaz
# -- yalnizca covariate (feature) degerlerini okur. Source istatistikleri
# SADECE source satirlarindan, target istatistikleri SADECE target
# satirlarindan hesaplanir (cross-contamination YOKTUR).
# =============================================================================
def compute_region_robust_stats(df: pd.DataFrame, numeric_features: list[str]) -> dict:
    """Bir bolgenin KENDI 'all_valid' (valid_for_modeling==True) populasyonu
    uzerinden, her numeric feature icin medyan + IQR hesaplar. Etiket (burned)
    HICBIR SEKILDE kullanilmaz/okunmaz.
    """
    valid = df[df["valid_for_modeling"] == True] if "valid_for_modeling" in df.columns else df  # noqa: E712
    stats: dict = {}
    for feature in numeric_features:
        values = pd.to_numeric(valid[feature], errors="coerce").dropna()
        if len(values) == 0:
            stats[feature] = {"median": None, "iqr": None, "zero_iqr_fallback_used": None, "n_used": 0}
            continue
        median = float(values.median())
        q25, q75 = np.percentile(values, [25, 75])
        iqr = float(q75 - q25)
        zero_iqr_fallback = iqr < EPSILON_IQR
        stats[feature] = {
            "median": median,
            "iqr": (1.0 if zero_iqr_fallback else iqr),
            "raw_iqr": iqr,
            "zero_iqr_fallback_used": bool(zero_iqr_fallback),
            "n_used": int(len(values)),
        }
    return stats


def apply_region_robust_transform(df: pd.DataFrame, stats: dict, numeric_features: list[str]) -> pd.DataFrame:
    """`stats` (compute_region_robust_stats ciktisi) ile numeric feature'lari
    robust_value = (value - median) / iqr olarak DONUSTURUR. Kategorik
    feature'lara (landcover_dominant) DOKUNULMAZ (source-fitted one-hot
    encoding pipeline icinde degismeden kalir). Eksik degerler NaN olarak
    KORUNUR (imputer bunlari daha sonra ele alir)."""
    out = df.copy()
    for feature in numeric_features:
        s = stats.get(feature)
        if s is None or s.get("median") is None:
            continue
        median, iqr = s["median"], s["iqr"]
        out[feature] = (pd.to_numeric(out[feature], errors="coerce") - median) / iqr
    return out


# =============================================================================
# Source-only spatial-block OOF: TEK gecis ile hem OOF olasilik dizisini hem
# de (Step9B'nin select_threshold_from_source_oof'uyla AYNI F1-grid mantigini
# kullanarak) esik secimini dondurur. make_spatial_folds (step8b) reuse edilir.
# =============================================================================
def run_source_oof(pipeline_template, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                    n_splits: int = STEP8B_N_SPLITS, random_state: int = STEP8B_RANDOM_SEED) -> dict:
    try:
        folds, n_splits_used = make_spatial_folds(y, groups, n_splits, random_state)
    except SystemExit as exc:
        return {
            "oof_prob": None, "covered_mask": None, "n_splits_used": None,
            "method": "no_cv_possible", "error": str(exc),
        }

    oof_prob = np.full(len(y), np.nan)
    for train_idx, test_idx in folds:
        model = clone(pipeline_template)
        model.fit(X.iloc[train_idx], y[train_idx])
        oof_prob[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]

    covered = ~np.isnan(oof_prob)
    return {
        "oof_prob": oof_prob, "covered_mask": covered, "n_splits_used": n_splits_used,
        "method": "spatial_block_oof", "oof_coverage": int(covered.sum()),
    }


def select_threshold_from_oof_predictions(y: np.ndarray, oof_prob: np.ndarray, covered_mask: np.ndarray) -> tuple[float, dict]:
    """Step9B:select_threshold_from_source_oof ile AYNI F1-grid secim
    mantigini, ONCEDEN hesaplanmis bir OOF olasilik dizisi uzerinde uygular
    (fold'lari YENIDEN egitmez -- run_source_oof zaten TEK gecis yapti)."""
    if covered_mask is None or covered_mask.sum() == 0 or len(np.unique(y[covered_mask])) < 2:
        return 0.5, {"method": "default_insufficient_oof_coverage"}

    from sklearn.metrics import f1_score

    y_oof, p_oof = y[covered_mask], oof_prob[covered_mask]
    f1s = [f1_score(y_oof, (p_oof >= t).astype(int), zero_division=0) for t in F1_THRESHOLD_GRID]
    best_idx = int(np.argmax(f1s))
    return float(F1_THRESHOLD_GRID[best_idx]), {
        "method": "source_oof_f1_optimal",
        "oof_coverage": int(covered_mask.sum()),
        "best_f1_on_source_oof": float(f1s[best_idx]),
    }


# =============================================================================
# Paired target-region spatial-block bootstrap (Step9C'nin mantigiyla AYNI
# algoritma -- block resample-with-replacement -- ama iki KEYFI olasilik
# kolonu [candidate/reference] icin genellenmis).
# =============================================================================
def _metrics_for_bootstrap_sample(y_true: np.ndarray, prob: np.ndarray) -> dict | None:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    if len(np.unique(y_true)) < 2:
        return None
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "brier_score": float(brier_score_loss(y_true, prob)),
    }


def paired_spatial_block_bootstrap(
    df_group: pd.DataFrame, block_col: str, y_col: str,
    candidate_prob_col: str, reference_prob_col: str,
    n_replicates: int, random_state: int, max_attempts_multiplier: int = 5,
) -> pd.DataFrame:
    """Hedef-bolge spatial_block_id'lerini yerine-koyarak (with replacement)
    bootstrap'lar; HER replikada candidate VE reference olasiliklari AYNI
    resample edilmis satir kumesi uzerinde degerlendirilir (paired). Modelleri
    YENIDEN EGITMEZ (mevcut tahminleri yeniden orneklemedir).
    """
    rng = np.random.default_rng(random_state)
    blocks = df_group[block_col].unique()
    n_blocks = len(blocks)
    if n_blocks == 0:
        return pd.DataFrame()

    block_to_indices = {b: df_group.index[df_group[block_col] == b].to_numpy() for b in blocks}

    records = []
    attempts = 0
    max_attempts = n_replicates * max_attempts_multiplier
    while len(records) < n_replicates and attempts < max_attempts:
        attempts += 1
        sampled_blocks = rng.choice(blocks, size=n_blocks, replace=True)
        idx = np.concatenate([block_to_indices[b] for b in sampled_blocks])
        sample = df_group.loc[idx]

        y = sample[y_col].to_numpy()
        m_cand = _metrics_for_bootstrap_sample(y, sample[candidate_prob_col].to_numpy())
        m_ref = _metrics_for_bootstrap_sample(y, sample[reference_prob_col].to_numpy())
        if m_cand is None or m_ref is None:
            continue  # degenerate (tek sinif) replika -- atla, tekrar dene

        records.append({
            "replicate": len(records),
            "candidate_roc_auc": m_cand["roc_auc"], "reference_roc_auc": m_ref["roc_auc"],
            "delta_roc_auc": m_cand["roc_auc"] - m_ref["roc_auc"],
            "candidate_pr_auc": m_cand["pr_auc"], "reference_pr_auc": m_ref["pr_auc"],
            "delta_pr_auc": m_cand["pr_auc"] - m_ref["pr_auc"],
            "candidate_brier": m_cand["brier_score"], "reference_brier": m_ref["brier_score"],
            "delta_brier": m_cand["brier_score"] - m_ref["brier_score"],
        })

    result = pd.DataFrame(records)
    result.attrs["n_valid_replicates"] = len(records)
    result.attrs["n_skipped_replicates"] = attempts - len(records)
    result.attrs["n_attempts"] = attempts
    return result


def bootstrap_support_category(lo: float | None, hi: float | None, higher_is_better: bool) -> str:
    """positive_support / negative_support / uncertain -- p-value DEGILDIR."""
    if lo is None or hi is None:
        return "uncertain"
    if higher_is_better:
        if lo > 0:
            return "positive_support"
        if hi < 0:
            return "negative_support"
        return "uncertain"
    # lower_is_better (Brier): negatif delta = iyilesme
    if hi < 0:
        return "positive_support"
    if lo > 0:
        return "negative_support"
    return "uncertain"