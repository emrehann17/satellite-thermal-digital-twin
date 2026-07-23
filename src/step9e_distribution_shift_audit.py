"""
step9e_distribution_shift_audit.py

Step9E: Manavgat 2021 <-> Bejís 2022 arasindaki cross-region transferin
Step9B'de discrimination'i KORUYAMAMASININ olasi nedenlerini teshis eden
POST-HOC bir dagilim-kaymasi (distribution-shift) ve iliski-kaymasi
(relationship-shift) denetimidir.

BU BIR POST-HOC TANI (DIAGNOSTIC) ANALIZIDIR. Step9E:
    - hicbir modeli YENIDEN EGITMEZ
    - Step9B tahminlerini DEGISTIRMEZ
    - Step9C bootstrap ciktilarini DEGISTIRMEZ
    - raporlanan Step9 sonucunu DEGISTIRMEZ
    - hedef etiketleri, mevcut Step9 modellerini geriye-donuk ayarlamak
      (retroactive tuning) icin KULLANMAZ
    - GEE, Step5, Step7 veya Step8'i YENIDEN CALISTIRMAZ

Step9E, Step9B/Step9C tamamlandiktan SONRA, "cross-region discrimination
neden desteklenmedi?" sorusunu tanimlamaya calisir -- feature dagilim
kaymasi mi, olasilik olcek kaymasi mi, yoksa bolgeye-bagli feature-label
iliski kaymasi mi?

GIRDILER (salt-okunur, degistirilmez):
    outputs/experiments/<manavgat_2021>/step8a/step8a_500m_modeling_dataset.parquet
    outputs/experiments/<bejis_2022>/step8a/step8a_500m_modeling_dataset.parquet
    outputs/cross_region/<source>__<target>/step9b/cross_region_transfer_predictions.parquet
    outputs/cross_region/<source>__<target>/step9b/cross_region_transfer_metrics.json

CIKTILAR:
    outputs/cross_region/<source>__<target>/step9e/distribution_shift_audit.json
    outputs/cross_region/<source>__<target>/step9e/numeric_feature_shift.csv
    outputs/cross_region/<source>__<target>/step9e/categorical_landcover_shift.csv
    outputs/cross_region/<source>__<target>/step9e/label_conditional_feature_relationships.csv
    outputs/cross_region/<source>__<target>/step9e/relationship_direction_flips.csv
    outputs/cross_region/<source>__<target>/step9e/prediction_distribution_audit.csv
    outputs/cross_region/<source>__<target>/step9e/calibration_bins.csv
    outputs/cross_region/<source>__<target>/step9e/distribution_shift_summary.md
    outputs/cross_region/<source>__<target>/step9e/feature_shift_heatmap.png
    outputs/cross_region/<source>__<target>/step9e/top_shifted_feature_distributions.png
    outputs/cross_region/<source>__<target>/step9e/label_conditional_direction_plot.png
    outputs/cross_region/<source>__<target>/step9e/landcover_distribution_comparison.png
    outputs/cross_region/<source>__<target>/step9e/prediction_probability_distributions.png
    outputs/cross_region/<source>__<target>/step9e/calibration_curves.png

GUVENLI IFADE (rapor bunu kullanir): ARTIK STATIK DEGIL. safe_wording, Step9D'nin
    canonical final_cross_region_report.json'undaki `overall_conclusion`
    degerinden (bkz. resolve_safe_wording()) DINAMIK olarak turetilir --
    "transfer_not_supported", "partial_transfer_supported" ve (Step9D'nin
    tam iki-yonlu destek icin urettigi) "bidirectional_transfer_supported"
    (== "transfer_supported" template'i) icin AYRI, dogru ifadeler kullanilir.
    Sabit tek bir "not supported" cumlesi HER ciftte KULLANILMAZ.

ASLA IDDIA ETMEZ: istatistiksel anlamlilik, nedensel aciklama, basarili
operasyonel transfer, veya "duzeltilmis" transfer performansi. Step9E'nin
onerdigi herhangi bir yeni normalizasyon/feature-secim stratejisi, YENI bir
deney olarak degerlendirilmelidir (ayni hedef bolgelerde validate edilip
"unbiased transfer sonucu" olarak sunulamaz).

RAPOR-ONLY REGENERASYON (--report-only): mevcut distribution_shift_audit.json'u
okur, YALNIZCA metadata alanlarini (safe_wording, step9b_predictions_source_path,
step9b_metrics_source_path, step9b_predictions_sha256, step9b_metrics_sha256,
step9d_overall_conclusion, created_at) gunceller; Part A-F'yi YENIDEN
HESAPLAMAZ, hicbir CSV/PNG/Step9A-D dosyasina DOKUNMAZ. Yazmadan once eski/yeni
JSON'u (yalnizca izin verilen metadata alanlari cikarilarak) karsilastirir ve
herhangi bir sayisal/bilimsel alan degismisse FAIL-FAST yapar (bkz.
assert_numeric_sections_unchanged()).

CLI:
    python src/step9e_distribution_shift_audit.py --source manavgat_2021 --target bejis_2022 --force
    python src/step9e_distribution_shift_audit.py --source manavgat_2021 --target evia_2021 --report-only --force
"""

from __future__ import annotations

import argparse
import hashlib
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

from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import roc_auc_score

from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
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

BASE_DIR = PROJECT_ROOT
log, log_file = setup_logger("step9e_distribution_shift_audit")

pywarnings.filterwarnings("ignore", category=RuntimeWarning)

EPSILON = 1e-6


class Step9EError(SystemExit):
    """Fail-fast error for Step9E (diğer step'lerle aynı konvansiyon)."""


# =============================================================================
# Feature registry -- Step9A'daki TEK KAYNAK'tan (SHARED_BASELINE_FEATURES /
# SHARED_THERMAL_FEATURES) turetilir; Step9E burada YENIDEN TANIMLAMAZ, sadece
# numeric/categorical olarak ayirir.
# =============================================================================
CATEGORICAL_FEATURES = ["landcover_dominant"]
NUMERIC_BASELINE_FEATURES = [f for f in SHARED_BASELINE_FEATURES if f not in CATEGORICAL_FEATURES]
NUMERIC_THERMAL_FEATURES = list(SHARED_THERMAL_FEATURES)
NUMERIC_FEATURES = NUMERIC_BASELINE_FEATURES + NUMERIC_THERMAL_FEATURES
ALL_AUDIT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Prompt'ta birebir listelenen "never treat as model features" kolonlari.
# Bunlar veri setinde identifier/label/diagnostic-grouping olarak bulunmalari
# NORMALDIR -- yasak olan onlari audit feature setine (NUMERIC_FEATURES /
# CATEGORICAL_FEATURES) dahil etmektir.
NEVER_AUDIT_AS_FEATURE_COLUMNS = [
    "burned",
    "cell_id",
    "spatial_block_id",
    "row_500m",
    "col_500m",
    "lon",
    "lat",
    "experiment_id",
    "region_key",
    "burn_date",
    "burn_month",
    "observed_fraction",
    "gapfilled_fraction",
]

MODEL_TYPES = ("baseline", "thermal")

N_CALIBRATION_BINS = 10

# Part A / F diagnostic (non-statistical-significance) heuristic thresholds.
SMD_THRESHOLDS = (0.2, 0.5, 0.8)
PSI_THRESHOLDS = (0.10, 0.25)
OUTSIDE_SUPPORT_THRESHOLD = 0.10

# Part D ranking-reversal heuristic thresholds (descriptive only).
RAW_AUC_REVERSAL_CEILING = 0.45
INVERSE_AUC_REVERSAL_FLOOR = 0.55


def _static_self_check() -> None:
    """Denetim feature setlerine yasak kolonlarin sizmadigini dogrular."""
    leaked = set(ALL_AUDIT_FEATURES).intersection(
        set(NEVER_AUDIT_AS_FEATURE_COLUMNS) | set(FORBIDDEN_MODEL_COLUMNS)
    )
    if leaked:
        raise Step9EError(f"YASAK kolonlar Step9E audit feature setine sizmis: {leaked}.")


_static_self_check()


# =============================================================================
# Path resolution + namespacing safety checks
# =============================================================================
def step9e_output_dir(source_id: str, target_id: str) -> Path:
    return cross_region_output_root(source_id, target_id) / "step9e"


def resolve_step9b_predictions_path(source_id: str, target_id: str) -> Path:
    return cross_region_output_root(source_id, target_id) / "step9b" / "cross_region_transfer_predictions.parquet"


def resolve_step9b_metrics_path(source_id: str, target_id: str) -> Path:
    return cross_region_output_root(source_id, target_id) / "step9b" / "cross_region_transfer_metrics.json"


def resolve_step9d_report_path(source_id: str, target_id: str) -> Path:
    return cross_region_output_root(source_id, target_id) / "step9d" / "final_cross_region_report.json"


def _assert_paths_are_safely_namespaced(source_id: str, target_id: str, path: Path) -> None:
    """
    Step9E'nin YALNIZCA kendi (source, target) ciftinin namespaced dizinlerine
    okuma/yazma yaptigini dogrular -- yanlislikla baska bir deney/cift'in
    ciktisina dokunmayi engeller (repo genelindeki namespacing guvenlik
    kontrolleri konvansiyonuyla tutarli).
    """
    parts = path.parts
    pair_token = f"{source_id}__{target_id}"
    if "cross_region" in parts:
        if pair_token not in parts:
            raise Step9EError(
                f"Namespacing guvenlik kontrolu BASARISIZ: '{path}' beklenen "
                f"'{pair_token}' cross-region dizinine ait degil."
            )
    elif "experiments" in parts:
        if source_id not in parts and target_id not in parts:
            raise Step9EError(
                f"Namespacing guvenlik kontrolu BASARISIZ: '{path}' ne "
                f"'{source_id}' ne de '{target_id}' deney dizinine ait."
            )


# =============================================================================
# Veri yukleme (salt-okunur -- hicbir girdi dosyasi bu modulde DEGISTIRILMEZ)
# =============================================================================
def load_step8a_dataset(experiment_id: str, other_id: str) -> pd.DataFrame:
    path = resolve_step8a_dataset_path(experiment_id)
    _assert_paths_are_safely_namespaced(experiment_id, other_id, path)
    if not path.exists():
        raise Step9EError(
            f"'{experiment_id}' icin Step8A veri seti bulunamadi: {path}. "
            "Step9E, Step8A/Step9A-C'yi YENIDEN CALISTIRMAZ; once bu adimlarin "
            "tamamlanmis olmasi gerekir."
        )
    return pd.read_parquet(path)


def load_step9b_predictions(source_id: str, target_id: str) -> pd.DataFrame:
    path = resolve_step9b_predictions_path(source_id, target_id)
    _assert_paths_are_safely_namespaced(source_id, target_id, path)
    if not path.exists():
        raise Step9EError(
            f"Step9B tahmin dosyasi bulunamadi: {path}. Step9E, Step9B'yi "
            "YENIDEN CALISTIRMAZ; once Step9B tamamlanmis olmalidir."
        )
    df = pd.read_parquet(path)
    if df.empty:
        raise Step9EError(f"Step9B tahmin dosyasi bos: {path}.")
    return df


def load_step9b_metrics(source_id: str, target_id: str) -> dict:
    path = resolve_step9b_metrics_path(source_id, target_id)
    _assert_paths_are_safely_namespaced(source_id, target_id, path)
    if not path.exists():
        raise Step9EError(f"Step9B metrics dosyasi bulunamadi: {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_step9d_report(source_id: str, target_id: str) -> dict:
    """Step9D'nin canonical final_cross_region_report.json'unu (salt-okunur)
    yukler -- safe_wording'in turetilecegi TEK dogru kaynak (bkz.
    resolve_step9e_provenance_and_wording())."""
    path = resolve_step9d_report_path(source_id, target_id)
    _assert_paths_are_safely_namespaced(source_id, target_id, path)
    if not path.exists():
        raise Step9EError(
            f"Step9D final raporu bulunamadi: {path}. Step9E, Step9D'yi "
            "YENIDEN CALISTIRMAZ; once Step9D tamamlanmis olmalidir."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# =============================================================================
# Dynamic safe_wording (Step9D overall_conclusion -> report wording)
# =============================================================================
SAFE_WORDING_BY_CONCLUSION = {
    "transfer_not_supported": (
        "Thermal incremental cross-region transfer was not supported in the "
        "original Step9 evaluation. Step9E examines whether feature-distribution "
        "shift, probability-scale shift, or region-dependent feature-label "
        "relationships are consistent with this result."
    ),
    "partial_transfer_supported": (
        "The original Step9 evaluation showed asymmetric or partial cross-region "
        "support for the thermal predictor set. Step9E examines the "
        "feature-distribution, probability-scale, and feature-label relationship "
        "shifts associated with this mixed result."
    ),
    "transfer_supported": (
        "The original Step9 evaluation showed cross-region support for the "
        "thermal predictor set under the evaluated directions and populations. "
        "Step9E examines the remaining feature-distribution, probability-scale, "
        "and feature-label relationship differences."
    ),
}

# src/step9d_build_cross_region_report.py:classify_overall_conclusion() emits
# "bidirectional_transfer_supported" (not literally "transfer_supported") for
# its full/both-directions-supported case. Treated as an alias of the
# "transfer_supported" template -- same underlying claim (cross-region support
# was observed), just Step9D's actual spelling for it.
_CONCLUSION_ALIASES = {
    "bidirectional_transfer_supported": "transfer_supported",
}


def resolve_safe_wording(overall_conclusion: str | None) -> str:
    """Maps Step9D's `overall_conclusion` to the correct report wording.
    Fails fast (rather than falling back to a generic/incorrect sentence) if
    `overall_conclusion` is missing or not a recognized value."""
    if not overall_conclusion:
        raise Step9EError(
            "Step9D final raporunda 'overall_conclusion' cozulemedi/None -- "
            "safe_wording turetilemiyor."
        )
    key = _CONCLUSION_ALIASES.get(overall_conclusion, overall_conclusion)
    wording = SAFE_WORDING_BY_CONCLUSION.get(key)
    if wording is None:
        raise Step9EError(
            f"Step9D overall_conclusion ('{overall_conclusion}') taninmiyor -- "
            "safe_wording turetilemiyor. Bilinen degerler: "
            f"{sorted(SAFE_WORDING_BY_CONCLUSION)} (+ aliases: {_CONCLUSION_ALIASES})."
        )
    return wording


# =============================================================================
# Step9B provenance (predictions vs. metrics -- MUST stay distinct; this is
# the exact bug this fix corrects, see module docstring / task).
# =============================================================================
def _assert_pair_matches(label: str, payload_source: str | None, payload_target: str | None, source_id: str, target_id: str) -> None:
    if payload_source != source_id or payload_target != target_id:
        raise Step9EError(
            f"{label}: source/target ('{payload_source}'/'{payload_target}') "
            f"Step9E ciftiyle ('{source_id}'/'{target_id}') UYUSMUYOR."
        )


def _assert_extension(label: str, path: Path, expected_suffix: str) -> None:
    if path.suffix != expected_suffix:
        raise Step9EError(
            f"{label} beklenen uzantiya sahip degil (beklenen '{expected_suffix}', "
            f"bulunan '{path.suffix}'): {path}"
        )


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_step9e_provenance_and_wording(source_id: str, target_id: str, step9b_metrics: dict) -> dict:
    """
    Single source of truth for the Step9B provenance fields + Step9D-derived
    safe_wording, used by BOTH the full run_shift_audit() and the
    --report-only regeneration path (regenerate_report_only()) -- so the
    predictions/metrics field-swap bug and the static safe_wording bug
    cannot silently reappear in only one of the two code paths.

    Fails fast (Step9EError) when:
        - the Step9B metrics' source/target ids do not match (source_id, target_id);
        - the Step9B predictions or metrics files are missing;
        - the resolved provenance paths do not have the expected extension
          (predictions: .parquet, metrics: .json);
        - the Step9D final report's source/target ids do not match
          (source_id, target_id);
        - the Step9D overall_conclusion is missing/unrecognized.
    """
    _assert_pair_matches(
        "Step9B cross_region_transfer_metrics.json",
        step9b_metrics.get("source_experiment_id"), step9b_metrics.get("target_experiment_id"),
        source_id, target_id,
    )

    predictions_path = resolve_step9b_predictions_path(source_id, target_id)
    metrics_path = resolve_step9b_metrics_path(source_id, target_id)
    if not predictions_path.exists():
        raise Step9EError(
            f"Step9B tahmin dosyasi bulunamadi: {predictions_path}. Step9E, "
            "Step9B'yi YENIDEN CALISTIRMAZ; once Step9B tamamlanmis olmalidir."
        )
    if not metrics_path.exists():
        raise Step9EError(f"Step9B metrics dosyasi bulunamadi: {metrics_path}.")
    _assert_extension("step9b_predictions_source_path", predictions_path, ".parquet")
    _assert_extension("step9b_metrics_source_path", metrics_path, ".json")

    step9d_report = load_step9d_report(source_id, target_id)
    _assert_pair_matches(
        "Step9D final_cross_region_report.json",
        step9d_report.get("source_experiment_id"), step9d_report.get("target_experiment_id"),
        source_id, target_id,
    )
    safe_wording = resolve_safe_wording(step9d_report.get("overall_conclusion"))

    return {
        "safe_wording": safe_wording,
        "step9d_overall_conclusion": step9d_report.get("overall_conclusion"),
        "step9b_predictions_source_path": str(predictions_path),
        "step9b_metrics_source_path": str(metrics_path),
        "step9b_predictions_sha256": _sha256_file(predictions_path),
        "step9b_metrics_sha256": _sha256_file(metrics_path),
    }


def population_subset(df: pd.DataFrame, population: str) -> pd.DataFrame:
    """Step9B'deki (population_subset) ile AYNI mantik -- YENIDEN hesaplama yok,
    Step8A'nin kendi boolean kolonlarini kullanir."""
    valid = df[df["valid_for_modeling"] == True] if "valid_for_modeling" in df.columns else df  # noqa: E712
    if population == "all_valid":
        return valid
    if population not in valid.columns:
        return valid.iloc[0:0]
    return valid[valid[population].astype(bool)]


# =============================================================================
# PART A -- global numeric feature shift
# =============================================================================
def _numeric_describe(series: pd.Series) -> dict:
    row_count = int(len(series))
    valid = pd.to_numeric(series, errors="coerce").dropna()
    valid_count = int(len(valid))
    missing_count = row_count - valid_count
    missing_fraction = (missing_count / row_count) if row_count else None
    if valid_count == 0:
        return {
            "row_count": row_count, "valid_count": 0, "missing_count": missing_count,
            "missing_fraction": missing_fraction, "mean": None, "standard_deviation": None,
            "minimum": None, "q05": None, "q25": None, "median": None, "q75": None,
            "q95": None, "maximum": None, "iqr": None,
        }
    q05, q25, median, q75, q95 = np.percentile(valid, [5, 25, 50, 75, 95])
    return {
        "row_count": row_count, "valid_count": valid_count, "missing_count": missing_count,
        "missing_fraction": missing_fraction, "mean": float(valid.mean()),
        "standard_deviation": float(valid.std(ddof=1)) if valid_count > 1 else 0.0,
        "minimum": float(valid.min()), "q05": float(q05), "q25": float(q25),
        "median": float(median), "q75": float(q75), "q95": float(q95),
        "maximum": float(valid.max()), "iqr": float(q75 - q25),
    }


def _standardized_mean_difference(mean_a, mean_b, std_a, std_b) -> float | None:
    if mean_a is None or mean_b is None or std_a is None or std_b is None:
        return None
    pooled_std = np.sqrt((std_a ** 2 + std_b ** 2) / 2.0)
    if pooled_std < EPSILON:
        return None
    return float((mean_b - mean_a) / pooled_std)


def _robust_standardized_median_difference(median_a, median_b, iqr_a, iqr_b) -> float | None:
    if median_a is None or median_b is None or iqr_a is None or iqr_b is None:
        return None
    pooled_iqr = (iqr_a + iqr_b) / 2.0
    if pooled_iqr < EPSILON:
        # Zero-IQR kacamak: her iki dagilim da (hemen hemen) sabit -- payda
        # sifirsa ve medyanlar da esitse fark 0, degilse yon bilgisiyle None.
        return 0.0 if abs(median_b - median_a) < EPSILON else None
    return float((median_b - median_a) / pooled_iqr)


def _population_stability_index(reference: np.ndarray, comparison: np.ndarray, n_bins: int = 10) -> float | None:
    """PSI, `reference` dagiliminin kendi kantil kutularindan (bins) turetilir;
    `comparison` bu kutulara yerlestirilir. Yon-spesifiktir (reference'i
    degistirmek farkli bir PSI verir)."""
    reference = reference[~np.isnan(reference)]
    comparison = comparison[~np.isnan(comparison)]
    if len(reference) < n_bins or len(comparison) == 0:
        return None

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.percentile(reference, quantiles * 100))
    if len(edges) < 2:
        return None  # reference sabit (degeri hep ayni) -- PSI hesaplanamaz
    edges[0], edges[-1] = -np.inf, np.inf  # dis uclari kapsa

    ref_counts, _ = np.histogram(reference, bins=edges)
    cmp_counts, _ = np.histogram(comparison, bins=edges)

    ref_prop = ref_counts / max(ref_counts.sum(), 1)
    cmp_prop = cmp_counts / max(cmp_counts.sum(), 1)

    ref_prop = np.where(ref_prop <= 0, EPSILON, ref_prop)
    cmp_prop = np.where(cmp_prop <= 0, EPSILON, cmp_prop)

    psi = float(np.sum((cmp_prop - ref_prop) * np.log(cmp_prop / ref_prop)))
    return psi


def _outside_support_fraction(reference: np.ndarray, comparison: np.ndarray) -> dict:
    """`reference`'in q01/q99 disinda kalan `comparison` degerlerinin oranini
    hesaplar (reference-anchored, yon-spesifik)."""
    reference = reference[~np.isnan(reference)]
    comparison = comparison[~np.isnan(comparison)]
    if len(reference) == 0 or len(comparison) == 0:
        return {"fraction_below_reference_q01": None, "fraction_above_reference_q99": None,
                "outside_reference_support_fraction": None}
    q01, q99 = np.percentile(reference, [1, 99])
    below = float((comparison < q01).mean())
    above = float((comparison > q99).mean())
    return {
        "fraction_below_reference_q01": below,
        "fraction_above_reference_q99": above,
        "outside_reference_support_fraction": below + above,
    }


def compute_numeric_feature_shift_row(
    feature: str, population: str, source_id: str, target_id: str,
    source_series: pd.Series, target_series: pd.Series,
) -> dict:
    source_stats = _numeric_describe(source_series)
    target_stats = _numeric_describe(target_series)

    source_vals = pd.to_numeric(source_series, errors="coerce").dropna().to_numpy()
    target_vals = pd.to_numeric(target_series, errors="coerce").dropna().to_numpy()

    row = {"feature": feature, "population": population,
           "source_experiment_id": source_id, "target_experiment_id": target_id}
    for k, v in source_stats.items():
        row[f"source_{k}"] = v
    for k, v in target_stats.items():
        row[f"target_{k}"] = v

    if len(source_vals) == 0 or len(target_vals) == 0:
        row.update({
            "mean_difference": None, "smd": None, "robust_median_difference": None,
            "ks_statistic": None, "wasserstein_distance": None,
            "normalized_wasserstein_by_source_iqr": None,
            "normalized_wasserstein_by_target_iqr": None,
            "psi_source_to_target": None, "psi_target_to_source": None,
            "fraction_target_below_source_q01": None, "fraction_target_above_source_q99": None,
            "outside_source_support_fraction": None,
            "fraction_source_below_target_q01": None, "fraction_source_above_target_q99": None,
            "outside_target_support_fraction": None,
            "abs_smd_ge_0_2": None, "abs_smd_ge_0_5": None, "abs_smd_ge_0_8": None,
            "psi_ge_0_10": None, "psi_ge_0_25": None, "outside_source_support_ge_0_10": None,
        })
        return row

    row["mean_difference"] = float(target_stats["mean"] - source_stats["mean"])
    row["smd"] = _standardized_mean_difference(
        source_stats["mean"], target_stats["mean"], source_stats["standard_deviation"], target_stats["standard_deviation"]
    )
    row["robust_median_difference"] = _robust_standardized_median_difference(
        source_stats["median"], target_stats["median"], source_stats["iqr"], target_stats["iqr"]
    )

    ks_result = ks_2samp(source_vals, target_vals)
    row["ks_statistic"] = float(ks_result.statistic)

    w_dist = float(wasserstein_distance(source_vals, target_vals))
    row["wasserstein_distance"] = w_dist
    src_iqr, tgt_iqr = source_stats["iqr"], target_stats["iqr"]
    row["normalized_wasserstein_by_source_iqr"] = (w_dist / src_iqr) if src_iqr and src_iqr > EPSILON else None
    row["normalized_wasserstein_by_target_iqr"] = (w_dist / tgt_iqr) if tgt_iqr and tgt_iqr > EPSILON else None

    row["psi_source_to_target"] = _population_stability_index(source_vals, target_vals, N_CALIBRATION_BINS)
    row["psi_target_to_source"] = _population_stability_index(target_vals, source_vals, N_CALIBRATION_BINS)

    outside_from_source = _outside_support_fraction(source_vals, target_vals)
    row["fraction_target_below_source_q01"] = outside_from_source["fraction_below_reference_q01"]
    row["fraction_target_above_source_q99"] = outside_from_source["fraction_above_reference_q99"]
    row["outside_source_support_fraction"] = outside_from_source["outside_reference_support_fraction"]

    outside_from_target = _outside_support_fraction(target_vals, source_vals)
    row["fraction_source_below_target_q01"] = outside_from_target["fraction_below_reference_q01"]
    row["fraction_source_above_target_q99"] = outside_from_target["fraction_above_reference_q99"]
    row["outside_target_support_fraction"] = outside_from_target["outside_reference_support_fraction"]

    abs_smd = abs(row["smd"]) if row["smd"] is not None else None
    row["abs_smd_ge_0_2"] = (abs_smd >= SMD_THRESHOLDS[0]) if abs_smd is not None else None
    row["abs_smd_ge_0_5"] = (abs_smd >= SMD_THRESHOLDS[1]) if abs_smd is not None else None
    row["abs_smd_ge_0_8"] = (abs_smd >= SMD_THRESHOLDS[2]) if abs_smd is not None else None

    max_psi = max(
        [p for p in (row["psi_source_to_target"], row["psi_target_to_source"]) if p is not None],
        default=None,
    )
    row["psi_ge_0_10"] = (max_psi >= PSI_THRESHOLDS[0]) if max_psi is not None else None
    row["psi_ge_0_25"] = (max_psi >= PSI_THRESHOLDS[1]) if max_psi is not None else None
    row["outside_source_support_ge_0_10"] = (
        row["outside_source_support_fraction"] >= OUTSIDE_SUPPORT_THRESHOLD
        if row["outside_source_support_fraction"] is not None else None
    )

    return row


def run_part_a_numeric_shift(source_id: str, target_id: str, source_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for population in ALL_POPULATIONS:
        src_pop = population_subset(source_df, population)
        tgt_pop = population_subset(target_df, population)
        for feature in NUMERIC_FEATURES:
            if feature not in src_pop.columns or feature not in tgt_pop.columns:
                log.warning("[Part A] '%s' kolonu bulunamadi (population=%s); atlaniyor.", feature, population)
                continue
            rows.append(compute_numeric_feature_shift_row(
                feature, population, source_id, target_id, src_pop[feature], tgt_pop[feature],
            ))
    return pd.DataFrame(rows)


# =============================================================================
# PART B -- categorical landcover shift
# =============================================================================
def _category_proportions(series: pd.Series) -> pd.Series:
    counts = series.dropna().value_counts()
    if counts.sum() == 0:
        return counts.astype(float)
    return counts / counts.sum()


def _total_variation_distance(prop_a: pd.Series, prop_b: pd.Series) -> float:
    all_categories = sorted(set(prop_a.index) | set(prop_b.index))
    a = np.array([prop_a.get(c, 0.0) for c in all_categories])
    b = np.array([prop_b.get(c, 0.0) for c in all_categories])
    return float(0.5 * np.abs(a - b).sum())


def _jensen_shannon_divergence(prop_a: pd.Series, prop_b: pd.Series) -> float:
    all_categories = sorted(set(prop_a.index) | set(prop_b.index))
    a = np.array([prop_a.get(c, 0.0) for c in all_categories]) + EPSILON
    b = np.array([prop_b.get(c, 0.0) for c in all_categories]) + EPSILON
    a = a / a.sum()
    b = b / b.sum()
    m = 0.5 * (a + b)
    kl_am = float(np.sum(a * np.log(a / m)))
    kl_bm = float(np.sum(b * np.log(b / m)))
    return float(0.5 * kl_am + 0.5 * kl_bm)


def run_part_b_landcover_shift(source_id: str, target_id: str, source_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    feature = CATEGORICAL_FEATURES[0]
    per_category_rows = []
    scalar_summary: dict = {}

    for population in ALL_POPULATIONS:
        src_pop = population_subset(source_df, population)
        tgt_pop = population_subset(target_df, population)
        if feature not in src_pop.columns or feature not in tgt_pop.columns:
            log.warning("[Part B] '%s' kolonu bulunamadi (population=%s); atlaniyor.", feature, population)
            continue

        src_series = src_pop[feature]
        tgt_series = tgt_pop[feature]
        src_counts = src_series.dropna().value_counts()
        tgt_counts = tgt_series.dropna().value_counts()
        src_prop = _category_proportions(src_series)
        tgt_prop = _category_proportions(tgt_series)

        all_categories = sorted(set(src_counts.index) | set(tgt_counts.index))
        for cat in all_categories:
            per_category_rows.append({
                "population": population, "landcover_category": cat,
                "source_experiment_id": source_id, "target_experiment_id": target_id,
                "source_count": int(src_counts.get(cat, 0)), "target_count": int(tgt_counts.get(cat, 0)),
                "source_proportion": float(src_prop.get(cat, 0.0)), "target_proportion": float(tgt_prop.get(cat, 0.0)),
                "unseen_in_source": bool(src_counts.get(cat, 0) == 0 and tgt_counts.get(cat, 0) > 0),
                "unseen_in_target": bool(tgt_counts.get(cat, 0) == 0 and src_counts.get(cat, 0) > 0),
            })

        n_src, n_tgt = int(src_series.notna().sum()), int(tgt_series.notna().sum())
        target_unseen_categories = sorted(set(tgt_counts.index) - set(src_counts.index))
        source_unseen_categories = sorted(set(src_counts.index) - set(tgt_counts.index))
        target_unseen_row_fraction = (
            float(tgt_series.isin(target_unseen_categories).sum() / n_tgt) if n_tgt else None
        )
        source_unseen_row_fraction = (
            float(src_series.isin(source_unseen_categories).sum() / n_src) if n_src else None
        )

        scalar_summary[population] = {
            "source_to_target": {
                "target_categories_unseen_in_source": [str(c) for c in target_unseen_categories],
                "unseen_target_category_row_fraction": target_unseen_row_fraction,
            },
            "target_to_source": {
                "source_categories_unseen_in_target": [str(c) for c in source_unseen_categories],
                "unseen_source_category_row_fraction": source_unseen_row_fraction,
            },
            "total_variation_distance": _total_variation_distance(src_prop, tgt_prop),
            "jensen_shannon_divergence": _jensen_shannon_divergence(src_prop, tgt_prop),
            "source_class_counts": {str(k): int(v) for k, v in src_counts.items()},
            "target_class_counts": {str(k): int(v) for k, v in tgt_counts.items()},
        }

    return pd.DataFrame(per_category_rows), scalar_summary


# =============================================================================
# PART C -- label-conditional relationship shift (descriptive univariate
# association diagnostics; NOT trained models, NOT significance tests)
# =============================================================================
def _univariate_label_relationship(values: pd.Series, burned: pd.Series) -> dict:
    x = pd.to_numeric(values, errors="coerce")
    y = pd.to_numeric(burned, errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask].to_numpy(), y[mask].to_numpy().astype(int)

    burned_vals, unburned_vals = x[y == 1], x[y == 0]
    n_burned, n_unburned = len(burned_vals), len(unburned_vals)

    out = {
        "burned_count": int(n_burned), "unburned_count": int(n_unburned),
        "burned_mean": float(np.mean(burned_vals)) if n_burned else None,
        "unburned_mean": float(np.mean(unburned_vals)) if n_unburned else None,
        "burned_median": float(np.median(burned_vals)) if n_burned else None,
        "unburned_median": float(np.median(unburned_vals)) if n_unburned else None,
        "raw_auc": None, "inverse_auc": None, "rank_biserial_correlation": None,
    }
    if n_burned and n_unburned:
        out["mean_difference"] = out["burned_mean"] - out["unburned_mean"]
        out["median_difference"] = out["burned_median"] - out["unburned_median"]
    else:
        out["mean_difference"], out["median_difference"] = None, None

    if n_burned == 0 or n_unburned == 0 or len(np.unique(x)) < 2:
        return out

    raw_auc = float(roc_auc_score(y, x))
    out["raw_auc"] = raw_auc
    out["inverse_auc"] = float(roc_auc_score(y, -x))
    # Signed rank-biserial correlation, equivalent to 2*AUC - 1 (Mann-Whitney
    # effect-size direction): positive means burned values tend to be higher.
    out["rank_biserial_correlation"] = float(2.0 * raw_auc - 1.0)
    return out


def run_part_c_label_conditional(source_id: str, target_id: str, source_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    region_frames = {source_id: source_df, target_id: target_df}
    for population in ALL_POPULATIONS:
        for region_id, df in region_frames.items():
            pop_df = population_subset(df, population)
            if TARGET_COLUMN not in pop_df.columns:
                continue
            for feature in NUMERIC_FEATURES:
                if feature not in pop_df.columns:
                    log.warning("[Part C] '%s' kolonu bulunamadi (region=%s, population=%s); atlaniyor.", feature, region_id, population)
                    continue
                rel = _univariate_label_relationship(pop_df[feature], pop_df[TARGET_COLUMN])
                rows.append({
                    "feature": feature, "population": population, "region_experiment_id": region_id,
                    **rel,
                })
    return pd.DataFrame(rows)


def run_relationship_direction_flips(source_id: str, target_id: str, label_conditional_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feature, population), group in label_conditional_df.groupby(["feature", "population"]):
        src_row = group[group["region_experiment_id"] == source_id]
        tgt_row = group[group["region_experiment_id"] == target_id]
        if src_row.empty or tgt_row.empty:
            continue
        src_row, tgt_row = src_row.iloc[0], tgt_row.iloc[0]

        def _sign_flip(a, b) -> bool | None:
            if a is None or b is None or pd.isna(a) or pd.isna(b):
                return None
            if a == 0 or b == 0:
                return False
            return bool(np.sign(a) != np.sign(b))

        mean_flip = _sign_flip(src_row["mean_difference"], tgt_row["mean_difference"])
        median_flip = _sign_flip(src_row["median_difference"], tgt_row["median_difference"])
        rank_flip = _sign_flip(src_row["rank_biserial_correlation"], tgt_row["rank_biserial_correlation"])

        src_auc, tgt_auc = src_row["raw_auc"], tgt_row["raw_auc"]
        auc_below_one_region_only = None
        if src_auc is not None and tgt_auc is not None and not pd.isna(src_auc) and not pd.isna(tgt_auc):
            auc_below_one_region_only = bool((src_auc < 0.5) != (tgt_auc < 0.5))

        flip_indicators = [f for f in (mean_flip, median_flip, rank_flip) if f is not None]
        flip_score = int(sum(1 for f in flip_indicators if f)) if flip_indicators else None

        rows.append({
            "feature": feature, "population": population,
            "source_experiment_id": source_id, "target_experiment_id": target_id,
            "source_mean_difference": src_row["mean_difference"], "target_mean_difference": tgt_row["mean_difference"],
            "source_median_difference": src_row["median_difference"], "target_median_difference": tgt_row["median_difference"],
            "source_rank_biserial_correlation": src_row["rank_biserial_correlation"],
            "target_rank_biserial_correlation": tgt_row["rank_biserial_correlation"],
            "source_raw_auc": src_auc, "target_raw_auc": tgt_auc,
            "mean_direction_flip": mean_flip, "median_direction_flip": median_flip,
            "rank_effect_direction_flip": rank_flip,
            "raw_auc_below_0_5_in_one_region_only": auc_below_one_region_only,
            "relationship_flip_score": flip_score,
        })
    return pd.DataFrame(rows)


# =============================================================================
# PART D -- prediction distribution audit (uses EXISTING Step9B predictions;
# does not retrain, does not invert predictions in the official result)
# =============================================================================
def _threshold_lookup(step9b_metrics: dict) -> dict:
    """(transfer_direction, population, model_key) -> threshold_used, Step9B
    metrics.json'un 'results' listesinden derlenir."""
    lookup: dict = {}
    for res in step9b_metrics.get("results", []):
        if res.get("skipped"):
            continue
        direction, population = res.get("transfer_direction"), res.get("population")
        for model_key in MODEL_TYPES:
            m = res.get(f"{model_key}_metrics") or {}
            threshold = m.get("threshold_used")
            if threshold is not None:
                lookup[(direction, population, model_key)] = threshold
    return lookup


def run_part_d_prediction_distribution(predictions_df: pd.DataFrame, step9b_metrics: dict) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score, brier_score_loss

    threshold_lookup = _threshold_lookup(step9b_metrics)
    rows = []
    for (direction, population), group in predictions_df.groupby(["transfer_direction", "population"]):
        y_true = group["burned"].to_numpy()
        prevalence = float(y_true.mean()) if len(y_true) else None

        for model_key in MODEL_TYPES:
            prob_col = f"{model_key}_probability"
            if prob_col not in group.columns:
                continue
            prob = pd.to_numeric(group[prob_col], errors="coerce").to_numpy()
            valid_mask = ~np.isnan(prob)
            prob_valid, y_valid = prob[valid_mask], y_true[valid_mask]
            if len(prob_valid) == 0:
                continue

            threshold = threshold_lookup.get((direction, population, model_key))
            q = np.percentile(prob_valid, [1, 5, 25, 75, 95, 99]) if len(prob_valid) else [None] * 6

            burned_prob = prob_valid[y_valid == 1]
            unburned_prob = prob_valid[y_valid == 0]

            row = {
                "transfer_direction": direction, "population": population, "model": model_key,
                "target_prevalence": prevalence, "n_rows": int(len(prob_valid)),
                "mean_predicted_probability": float(np.mean(prob_valid)),
                "median_predicted_probability": float(np.median(prob_valid)),
                "q01": float(q[0]), "q05": float(q[1]), "q25": float(q[2]),
                "q75": float(q[3]), "q95": float(q[4]), "q99": float(q[5]),
                "mean_probability_burned_rows": float(np.mean(burned_prob)) if len(burned_prob) else None,
                "mean_probability_unburned_rows": float(np.mean(unburned_prob)) if len(unburned_prob) else None,
                "median_probability_burned_rows": float(np.median(burned_prob)) if len(burned_prob) else None,
                "median_probability_unburned_rows": float(np.median(unburned_prob)) if len(unburned_prob) else None,
                "threshold_used": threshold,
            }

            if threshold is not None:
                above = prob_valid >= threshold
                row["fraction_all_rows_above_threshold"] = float(above.mean())
                row["fraction_burned_rows_above_threshold"] = (
                    float((burned_prob >= threshold).mean()) if len(burned_prob) else None
                )
                row["fraction_unburned_rows_above_threshold"] = (
                    float((unburned_prob >= threshold).mean()) if len(unburned_prob) else None
                )
            else:
                row["fraction_all_rows_above_threshold"] = None
                row["fraction_burned_rows_above_threshold"] = None
                row["fraction_unburned_rows_above_threshold"] = None

            if len(np.unique(y_valid)) < 2:
                row.update({
                    "roc_auc": None, "diagnostic_inverse_roc_auc": None, "pr_auc": None,
                    "brier_score": None, "calibration_in_the_large": None,
                })
            else:
                roc_auc = float(roc_auc_score(y_valid, prob_valid))
                row["roc_auc"] = roc_auc
                row["diagnostic_inverse_roc_auc"] = float(roc_auc_score(y_valid, 1.0 - prob_valid))
                row["pr_auc"] = float(average_precision_score(y_valid, prob_valid))
                row["brier_score"] = float(brier_score_loss(y_valid, prob_valid))
                row["calibration_in_the_large"] = float(np.mean(prob_valid) - prevalence)

            row["all_or_nearly_all_predictions_below_threshold"] = (
                bool(row["fraction_all_rows_above_threshold"] is not None and row["fraction_all_rows_above_threshold"] <= 0.01)
            )
            row["mean_probability_far_from_target_prevalence"] = (
                bool(row["calibration_in_the_large"] is not None and abs(row["calibration_in_the_large"]) >= 0.10)
            )
            inv_minus_raw = (
                (row["diagnostic_inverse_roc_auc"] - row["roc_auc"])
                if row["roc_auc"] is not None and row["diagnostic_inverse_roc_auc"] is not None else None
            )
            row["inverse_auc_substantially_above_raw_auc"] = (
                bool(inv_minus_raw is not None and inv_minus_raw >= 0.10)
            )
            row["target_ranking_reversal_suspected"] = bool(
                row["roc_auc"] is not None and row["diagnostic_inverse_roc_auc"] is not None
                and row["roc_auc"] < RAW_AUC_REVERSAL_CEILING
                and row["diagnostic_inverse_roc_auc"] > INVERSE_AUC_REVERSAL_FLOOR
            )

            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# PART E -- calibration tables
# =============================================================================
def _calibration_bins_for_group(y_true: np.ndarray, prob: np.ndarray, n_bins: int = N_CALIBRATION_BINS) -> pd.DataFrame:
    valid_mask = ~np.isnan(prob)
    y_true, prob = y_true[valid_mask], prob[valid_mask]
    if len(prob) == 0:
        return pd.DataFrame()

    actual_n_bins = n_bins
    bin_index = None
    while actual_n_bins >= 1:
        try:
            bin_index, edges = pd.qcut(prob, q=actual_n_bins, retbins=True, duplicates="drop")
            categories = bin_index.categories if hasattr(bin_index, "categories") else bin_index.cat.categories
            n_distinct = categories.size
            if n_distinct >= 1:
                actual_n_bins = n_distinct
                break
        except ValueError:
            pass
        actual_n_bins -= 1

    if bin_index is None or actual_n_bins < 1:
        # Tum olasilik degerleri ayni -- tek bir kutu.
        df = pd.DataFrame({
            "bin_index": [0], "requested_n_bins": [n_bins], "actual_n_bins": [1],
            "predicted_probability_mean": [float(np.mean(prob))],
            "observed_burned_fraction": [float(np.mean(y_true))],
            "row_count": [int(len(prob))], "positive_count": [int(np.sum(y_true))],
        })
        return df

    frame = pd.DataFrame({"prob": prob, "burned": y_true, "bin": bin_index})
    grouped = frame.groupby("bin", observed=True)
    out = grouped.agg(
        predicted_probability_mean=("prob", "mean"),
        observed_burned_fraction=("burned", "mean"),
        row_count=("prob", "size"),
        positive_count=("burned", "sum"),
    ).reset_index(drop=True)
    out.insert(0, "bin_index", range(len(out)))
    out.insert(1, "requested_n_bins", n_bins)
    out.insert(2, "actual_n_bins", len(out))
    out["row_count"] = out["row_count"].astype(int)
    out["positive_count"] = out["positive_count"].astype(int)
    return out


def run_part_e_calibration_bins(predictions_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for (direction, population), group in predictions_df.groupby(["transfer_direction", "population"]):
        y_true = group["burned"].to_numpy()
        for model_key in MODEL_TYPES:
            prob_col = f"{model_key}_probability"
            if prob_col not in group.columns:
                continue
            prob = pd.to_numeric(group[prob_col], errors="coerce").to_numpy()
            bins_df = _calibration_bins_for_group(y_true, prob)
            if bins_df.empty:
                continue
            bins_df.insert(0, "model", model_key)
            bins_df.insert(0, "population", population)
            bins_df.insert(0, "transfer_direction", direction)
            frames.append(bins_df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =============================================================================
# PART F -- summary ranking + cautious diagnosis
# =============================================================================
def run_part_f_summary(
    numeric_shift_df: pd.DataFrame, categorical_summary: dict,
    flips_df: pd.DataFrame, prediction_audit_df: pd.DataFrame,
    primary_population: str,
) -> dict:
    prim = numeric_shift_df[numeric_shift_df["population"] == primary_population].copy()
    prim["abs_smd_rank_value"] = prim["smd"].abs()
    prim["psi_rank_value"] = prim[["psi_source_to_target", "psi_target_to_source"]].max(axis=1)
    prim["norm_wasserstein_rank_value"] = prim["normalized_wasserstein_by_source_iqr"]
    prim["outside_support_rank_value"] = prim["outside_source_support_fraction"]

    def _shift_bucket(abs_smd: float | None) -> str:
        if abs_smd is None:
            return "low_shift"
        if abs_smd >= SMD_THRESHOLDS[2]:
            return "high_shift"
        if abs_smd >= SMD_THRESHOLDS[0]:
            return "moderate_shift"
        return "low_shift"

    prim["shift_category"] = prim["abs_smd_rank_value"].apply(_shift_bucket)

    ranked = prim.sort_values(
        by=["abs_smd_rank_value", "psi_rank_value", "norm_wasserstein_rank_value", "outside_support_rank_value"],
        ascending=False, na_position="last",
    )
    top_shifted = ranked.head(10)[[
        "feature", "smd", "psi_source_to_target", "psi_target_to_source",
        "normalized_wasserstein_by_source_iqr", "outside_source_support_fraction", "shift_category",
    ]].to_dict(orient="records")

    missingness = prim.copy()
    missingness["missingness_gap"] = (missingness["target_missing_fraction"] - missingness["source_missing_fraction"]).abs()
    top_missingness = (
        missingness.sort_values("missingness_gap", ascending=False, na_position="last")
        .head(5)[["feature", "source_missing_fraction", "target_missing_fraction", "missingness_gap"]]
        .to_dict(orient="records")
    )

    landcover_primary = categorical_summary.get(primary_population, {})

    flips_prim = flips_df[flips_df["population"] == primary_population] if not flips_df.empty else flips_df
    flipped_features = (
        flips_prim[flips_prim["relationship_flip_score"].fillna(0) > 0][
            ["feature", "relationship_flip_score", "raw_auc_below_0_5_in_one_region_only"]
        ].to_dict(orient="records")
        if not flips_prim.empty else []
    )

    prob_collapse = []
    ranking_reversal_suspected = False
    if not prediction_audit_df.empty:
        prim_pred = prediction_audit_df[prediction_audit_df["population"] == primary_population]
        prob_collapse = prim_pred[prim_pred["all_or_nearly_all_predictions_below_threshold"] == True][  # noqa: E712
            ["transfer_direction", "model", "fraction_all_rows_above_threshold"]
        ].to_dict(orient="records")
        ranking_reversal_suspected = bool(
            (prim_pred["target_ranking_reversal_suspected"] == True).any()  # noqa: E712
        )

    diagnosis_categories = set()
    if not ranked.empty and (ranked["shift_category"] == "high_shift").any():
        diagnosis_categories.add("high_shift")
    elif not ranked.empty and (ranked["shift_category"] == "moderate_shift").any():
        diagnosis_categories.add("moderate_shift")
    else:
        diagnosis_categories.add("low_shift")
    if flipped_features:
        diagnosis_categories.add("relationship_direction_instability")
    if prob_collapse or (
        not prediction_audit_df.empty
        and prediction_audit_df[prediction_audit_df["population"] == primary_population][
            "mean_probability_far_from_target_prevalence"
        ].fillna(False).any()
    ):
        diagnosis_categories.add("probability_scale_shift")
    if ranking_reversal_suspected:
        diagnosis_categories.add("ranking_reversal_suspected")

    likely_contributors = []
    if "high_shift" in diagnosis_categories or "moderate_shift" in diagnosis_categories:
        likely_contributors.append(
            "Feature distributions differ between regions for one or more shared predictors "
            "(elevated standardized mean difference / PSI / normalized Wasserstein distance)."
        )
    if "relationship_direction_instability" in diagnosis_categories:
        likely_contributors.append(
            "The direction of the association between one or more features and burned status "
            "is not consistent between the two regions (mean/median/rank-effect direction flips)."
        )
    if "probability_scale_shift" in diagnosis_categories:
        likely_contributors.append(
            "Predicted probabilities on the target region are concentrated below the "
            "source-selected threshold or diverge from the target's observed prevalence."
        )
    if "ranking_reversal_suspected" in diagnosis_categories:
        likely_contributors.append(
            "Diagnostic evidence is consistent with (but does not prove) a ranking-orientation "
            "reversal on the target region for at least one model/direction/population."
        )
    if not likely_contributors:
        likely_contributors.append(
            "No individual diagnostic signal in this audit stands out strongly; poor cross-region "
            "discrimination may reflect a combination of smaller shifts, or factors this audit does "
            "not cover."
        )

    return {
        "primary_population": primary_population,
        "top_globally_shifted_features": top_shifted,
        "top_missingness_differences": top_missingness,
        "landcover_differences_primary_population": landcover_primary,
        "features_with_relationship_direction_flip": flipped_features,
        "probability_collapse_below_threshold": prob_collapse,
        "ranking_reversal_suspected": ranking_reversal_suspected,
        "diagnosis_categories": sorted(diagnosis_categories),
        "likely_contributors_to_poor_cross_region_discrimination": likely_contributors,
        "note": (
            "Diagnosis categories are heuristic descriptive buckets, not a binary "
            "scientific PASS/FAIL and not a claim of statistical significance or "
            "causal explanation."
        ),
    }


# =============================================================================
# FIGURES
# =============================================================================
def _safe_savefig(fig, path: Path) -> None:
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=120)
    except Exception as exc:  # noqa: BLE001
        log.warning("Figur yazilamadi (%s): %s", path, exc)
    finally:
        plt.close(fig)


def plot_feature_shift_heatmap(numeric_shift_df: pd.DataFrame, output_dir: Path) -> None:
    if numeric_shift_df.empty:
        return
    pivot = numeric_shift_df.pivot_table(index="feature", columns="population", values="smd", aggfunc="first")
    pivot = pivot.reindex(NUMERIC_FEATURES)
    fig, ax = plt.subplots(figsize=(1.6 * max(len(pivot.columns), 1) + 3, 0.5 * len(pivot) + 2))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Standardized Mean Difference (SMD)")
    ax.set_title("Step9E: Feature Distribution Shift (SMD) by Population")
    _safe_savefig(fig, output_dir / "feature_shift_heatmap.png")


def plot_top_shifted_feature_distributions(
    numeric_shift_df: pd.DataFrame, source_id: str, target_id: str,
    source_df: pd.DataFrame, target_df: pd.DataFrame, primary_population: str, output_dir: Path,
) -> None:
    prim = numeric_shift_df[numeric_shift_df["population"] == primary_population].copy()
    if prim.empty:
        return
    prim["abs_smd"] = prim["smd"].abs()
    top = prim.sort_values("abs_smd", ascending=False, na_position="last").head(4)["feature"].tolist()
    if not top:
        return

    src_pop = population_subset(source_df, primary_population)
    tgt_pop = population_subset(target_df, primary_population)

    fig, axes = plt.subplots(1, len(top), figsize=(4.5 * len(top), 4), squeeze=False)
    for i, feature in enumerate(top):
        ax = axes[0][i]
        s = pd.to_numeric(src_pop[feature], errors="coerce").dropna()
        t = pd.to_numeric(tgt_pop[feature], errors="coerce").dropna()
        ax.hist(s, bins=30, alpha=0.5, density=True, label=source_id)
        ax.hist(t, bins=30, alpha=0.5, density=True, label=target_id)
        ax.set_title(feature, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(f"Step9E: Top Shifted Feature Distributions (population={primary_population})")
    _safe_savefig(fig, output_dir / "top_shifted_feature_distributions.png")


def plot_label_conditional_direction(label_conditional_df: pd.DataFrame, source_id: str, target_id: str, primary_population: str, output_dir: Path) -> None:
    prim = label_conditional_df[label_conditional_df["population"] == primary_population]
    if prim.empty:
        return
    pivot = prim.pivot_table(index="feature", columns="region_experiment_id", values="mean_difference", aggfunc="first")
    pivot = pivot.reindex(NUMERIC_FEATURES)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(pivot) + 2))
    y_pos = np.arange(len(pivot))
    width = 0.35
    if source_id in pivot.columns:
        ax.barh(y_pos - width / 2, pivot[source_id].to_numpy(dtype=float), height=width, label=source_id)
    if target_id in pivot.columns:
        ax.barh(y_pos + width / 2, pivot[target_id].to_numpy(dtype=float), height=width, label=target_id)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("burned-minus-unburned mean difference")
    ax.set_title(f"Step9E: Label-Conditional Mean Difference by Region (population={primary_population})")
    ax.legend()
    _safe_savefig(fig, output_dir / "label_conditional_direction_plot.png")


def plot_landcover_distribution_comparison(categorical_df: pd.DataFrame, source_id: str, target_id: str, primary_population: str, output_dir: Path) -> None:
    prim = categorical_df[categorical_df["population"] == primary_population]
    if prim.empty:
        return
    prim = prim.sort_values("landcover_category")
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(prim)), 4))
    x = np.arange(len(prim))
    width = 0.35
    ax.bar(x - width / 2, prim["source_proportion"], width=width, label=source_id)
    ax.bar(x + width / 2, prim["target_proportion"], width=width, label=target_id)
    ax.set_xticks(x)
    ax.set_xticklabels(prim["landcover_category"].astype(str), rotation=0)
    ax.set_ylabel("proportion")
    ax.set_title(f"Step9E: Landcover Class Proportions (population={primary_population})")
    ax.legend()
    _safe_savefig(fig, output_dir / "landcover_distribution_comparison.png")


def plot_prediction_probability_distributions(predictions_df: pd.DataFrame, primary_population: str, output_dir: Path) -> None:
    prim = predictions_df[predictions_df["population"] == primary_population]
    if prim.empty:
        return
    directions = sorted(prim["transfer_direction"].unique())
    if not directions:
        return
    fig, axes = plt.subplots(len(directions), len(MODEL_TYPES), figsize=(5 * len(MODEL_TYPES), 3.5 * len(directions)), squeeze=False)
    for i, direction in enumerate(directions):
        d = prim[prim["transfer_direction"] == direction]
        for j, model_key in enumerate(MODEL_TYPES):
            ax = axes[i][j]
            col = f"{model_key}_probability"
            if col not in d.columns:
                continue
            burned = d[d["burned"] == 1][col].dropna()
            unburned = d[d["burned"] == 0][col].dropna()
            ax.hist(unburned, bins=25, alpha=0.5, density=True, label="unburned")
            ax.hist(burned, bins=25, alpha=0.5, density=True, label="burned")
            ax.set_title(f"{direction} / {model_key}", fontsize=9)
            ax.legend(fontsize=7)
    fig.suptitle(f"Step9E: Predicted Probability Distributions (population={primary_population})")
    _safe_savefig(fig, output_dir / "prediction_probability_distributions.png")


def plot_calibration_curves(calibration_bins_df: pd.DataFrame, primary_population: str, output_dir: Path) -> None:
    prim = calibration_bins_df[calibration_bins_df["population"] == primary_population] if not calibration_bins_df.empty else calibration_bins_df
    if prim.empty:
        return
    directions = sorted(prim["transfer_direction"].unique())
    if not directions:
        return
    fig, axes = plt.subplots(1, len(directions), figsize=(5 * len(directions), 4.5), squeeze=False)
    for i, direction in enumerate(directions):
        ax = axes[0][i]
        d = prim[prim["transfer_direction"] == direction]
        for model_key in MODEL_TYPES:
            dm = d[d["model"] == model_key].sort_values("predicted_probability_mean")
            if dm.empty:
                continue
            ax.plot(dm["predicted_probability_mean"], dm["observed_burned_fraction"], marker="o", label=model_key)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.7, label="perfect calibration")
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed burned fraction")
        ax.set_title(direction, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"Step9E: Calibration Curves (population={primary_population})")
    _safe_savefig(fig, output_dir / "calibration_curves.png")


# =============================================================================
# Markdown summary
# =============================================================================
# NOTE: safe_wording is NO LONGER a static module constant -- it is resolved
# dynamically per (source, target) pair from Step9D's overall_conclusion
# (see SAFE_WORDING_BY_CONCLUSION / resolve_safe_wording() above) and read
# from `payload["safe_wording"]` below.

INTERPRETATION_RULES = [
    "Step9E is a post-hoc diagnostic analysis.",
    "It does not alter the original cross-region evaluation (Step9A-D outputs are read-only inputs here and are never modified).",
    "Target labels are inspected only to diagnose relationship shift after the transfer evaluation was completed.",
    "Any new normalization or feature-selection strategy suggested by Step9E must be evaluated as a new experiment.",
    "It must not be validated on the same target regions and then described as an unbiased transfer result.",
    "A third independent region or nested evaluation design is required for a stronger follow-up generalization claim.",
]

NEVER_CLAIMS = [
    "statistical significance",
    "causal explanation",
    "successful operational transfer",
    "corrected transfer performance",
]


def write_markdown_summary(payload: dict, output_dir: Path) -> Path:
    p = payload["part_f_summary"]
    lines = [
        "# Step9E: Cross-Region Distribution-Shift and Relationship-Shift Audit",
        "",
        f"- source: `{payload['source_experiment_id']}`",
        f"- target: `{payload['target_experiment_id']}`",
        f"- primary population: `{p['primary_population']}`",
        "",
        "> " + payload["safe_wording"],
        "",
        "## Diagnosis categories",
        "",
        ", ".join(f"`{c}`" for c in p["diagnosis_categories"]) or "(none)",
        "",
        "## Likely contributors to poor cross-region discrimination",
        "",
    ]
    for item in p["likely_contributors_to_poor_cross_region_discrimination"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Top globally shifted features (primary population)", "",
                  "| feature | smd | psi (source->target) | psi (target->source) | norm. wasserstein (source IQR) | outside-source-support fraction | category |",
                  "|---|---|---|---|---|---|---|"])
    for row in p["top_globally_shifted_features"]:
        lines.append(
            f"| {row['feature']} | {row.get('smd')} | {row.get('psi_source_to_target')} | "
            f"{row.get('psi_target_to_source')} | {row.get('normalized_wasserstein_by_source_iqr')} | "
            f"{row.get('outside_source_support_fraction')} | {row.get('shift_category')} |"
        )

    lines.extend(["", "## Strongest missingness differences", "",
                  "| feature | source missing fraction | target missing fraction | gap |",
                  "|---|---|---|---|"])
    for row in p["top_missingness_differences"]:
        lines.append(
            f"| {row['feature']} | {row.get('source_missing_fraction')} | "
            f"{row.get('target_missing_fraction')} | {row.get('missingness_gap')} |"
        )

    lines.extend(["", "## Landcover differences (primary population)", ""])
    lc = p["landcover_differences_primary_population"]
    if lc:
        lines.append(f"- total variation distance: {lc.get('total_variation_distance')}")
        lines.append(f"- Jensen-Shannon divergence: {lc.get('jensen_shannon_divergence')}")
        lines.append(
            f"- target categories unseen in source: {lc.get('source_to_target', {}).get('target_categories_unseen_in_source')}"
        )
        lines.append(
            f"- source categories unseen in target: {lc.get('target_to_source', {}).get('source_categories_unseen_in_target')}"
        )
    else:
        lines.append("(no landcover data for primary population)")

    lines.extend(["", "## Features with a label-relationship direction flip (primary population)", ""])
    if p["features_with_relationship_direction_flip"]:
        lines.append("| feature | relationship_flip_score | raw AUC below 0.5 in one region only |")
        lines.append("|---|---|---|")
        for row in p["features_with_relationship_direction_flip"]:
            lines.append(
                f"| {row['feature']} | {row.get('relationship_flip_score')} | "
                f"{row.get('raw_auc_below_0_5_in_one_region_only')} |"
            )
    else:
        lines.append("(none flagged)")

    lines.extend(["", "## Prediction probability scale", ""])
    lines.append(f"- ranking reversal suspected (any direction/model, primary population): {p['ranking_reversal_suspected']}")
    if p["probability_collapse_below_threshold"]:
        lines.append("- rows/models where predictions collapse below the source-selected threshold:")
        for row in p["probability_collapse_below_threshold"]:
            lines.append(
                f"  - {row['transfer_direction']} / {row['model']}: "
                f"fraction above threshold = {row.get('fraction_all_rows_above_threshold')}"
            )
    else:
        lines.append("- no probability collapse below threshold flagged.")

    lines.extend(["", "## Interpretation rules", ""])
    for rule in INTERPRETATION_RULES:
        lines.append(f"- {rule}")

    lines.extend(["", "## Never claimed by this report", ""])
    for c in NEVER_CLAIMS:
        lines.append(f"- {c}")

    lines.extend(["", "## Scope note", "",
                  "This is a POST-HOC diagnostic audit of the existing Step9B/Step9C "
                  "cross-region transfer evaluation. It does not retrain any model, does "
                  "not modify Step9B predictions or Step9C bootstrap outputs, and does not "
                  "change the reported Step9 conclusion."])

    md_path = output_dir / "distribution_shift_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


# =============================================================================
# Orkestrasyon
# =============================================================================
def planned_output_files(output_dir: Path) -> list[Path]:
    return [
        output_dir / "distribution_shift_audit.json",
        output_dir / "numeric_feature_shift.csv",
        output_dir / "categorical_landcover_shift.csv",
        output_dir / "label_conditional_feature_relationships.csv",
        output_dir / "relationship_direction_flips.csv",
        output_dir / "prediction_distribution_audit.csv",
        output_dir / "calibration_bins.csv",
        output_dir / "distribution_shift_summary.md",
        output_dir / "feature_shift_heatmap.png",
        output_dir / "top_shifted_feature_distributions.png",
        output_dir / "label_conditional_direction_plot.png",
        output_dir / "landcover_distribution_comparison.png",
        output_dir / "prediction_probability_distributions.png",
        output_dir / "calibration_curves.png",
    ]


def run_shift_audit(source_id: str, target_id: str, force: bool = False) -> dict:
    if source_id == target_id:
        raise Step9EError("--source ve --target ayni deney OLAMAZ.")

    output_dir = step9e_output_dir(source_id, target_id)
    json_path = output_dir / "distribution_shift_audit.json"
    if json_path.exists() and not force:
        log.info("Step9E ciktisi zaten var (%s); --force verilmedigi icin atlaniyor.", json_path)
        return json.loads(json_path.read_text(encoding="utf-8"))

    primary_population = PRIMARY_POPULATIONS[0]

    log.info("Step9E: %s <-> %s icin girdiler yukleniyor (salt-okunur)...", source_id, target_id)
    source_df = load_step8a_dataset(source_id, target_id)
    target_df = load_step8a_dataset(target_id, source_id)
    predictions_df = load_step9b_predictions(source_id, target_id)
    step9b_metrics = load_step9b_metrics(source_id, target_id)

    log.info("[Part A] numeric feature shift hesaplaniyor...")
    numeric_shift_df = run_part_a_numeric_shift(source_id, target_id, source_df, target_df)

    log.info("[Part B] categorical landcover shift hesaplaniyor...")
    categorical_df, categorical_summary = run_part_b_landcover_shift(source_id, target_id, source_df, target_df)

    log.info("[Part C] label-conditional relationship shift hesaplaniyor...")
    label_conditional_df = run_part_c_label_conditional(source_id, target_id, source_df, target_df)
    flips_df = run_relationship_direction_flips(source_id, target_id, label_conditional_df)

    log.info("[Part D] prediction distribution audit hesaplaniyor (Step9B tahminleri, salt-okunur)...")
    prediction_audit_df = run_part_d_prediction_distribution(predictions_df, step9b_metrics)

    log.info("[Part E] calibration bin'leri hesaplaniyor...")
    calibration_bins_df = run_part_e_calibration_bins(predictions_df)

    log.info("[Part F] ozet siralama + tanı hazirlaniyor...")
    part_f_summary = run_part_f_summary(
        numeric_shift_df, categorical_summary, flips_df, prediction_audit_df, primary_population,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_shift_df.to_csv(output_dir / "numeric_feature_shift.csv", index=False)
    categorical_df.to_csv(output_dir / "categorical_landcover_shift.csv", index=False)
    label_conditional_df.to_csv(output_dir / "label_conditional_feature_relationships.csv", index=False)
    flips_df.to_csv(output_dir / "relationship_direction_flips.csv", index=False)
    prediction_audit_df.to_csv(output_dir / "prediction_distribution_audit.csv", index=False)
    calibration_bins_df.to_csv(output_dir / "calibration_bins.csv", index=False)

    # Step9B provenance (predictions vs. metrics paths/hashes, kept distinct)
    # + Step9D-derived safe_wording -- SINGLE resolver shared with
    # --report-only (regenerate_report_only()) so the two code paths cannot
    # drift apart.
    provenance_and_wording = resolve_step9e_provenance_and_wording(source_id, target_id, step9b_metrics)

    payload = {
        "source_experiment_id": source_id,
        "target_experiment_id": target_id,
        "audit_type": "post_hoc_distribution_and_relationship_shift_diagnostic",
        **provenance_and_wording,
        "interpretation_rules": INTERPRETATION_RULES,
        "never_claims": NEVER_CLAIMS,
        "numeric_audit_features": NUMERIC_FEATURES,
        "categorical_audit_features": CATEGORICAL_FEATURES,
        "never_audit_as_feature_columns": NEVER_AUDIT_AS_FEATURE_COLUMNS,
        "primary_populations": PRIMARY_POPULATIONS,
        "secondary_populations": SECONDARY_POPULATIONS,
        "populations_evaluated": ALL_POPULATIONS,
        "part_a_numeric_feature_shift": numeric_shift_df.to_dict(orient="records"),
        "part_b_categorical_landcover_shift": categorical_summary,
        "part_c_label_conditional_relationships": label_conditional_df.to_dict(orient="records"),
        "part_c_relationship_direction_flips": flips_df.to_dict(orient="records"),
        "part_d_prediction_distribution_audit": prediction_audit_df.to_dict(orient="records"),
        "part_e_calibration_bins": calibration_bins_df.to_dict(orient="records"),
        "part_f_summary": part_f_summary,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log.info("distribution_shift_audit.json yazildi: %s", json_path)

    write_markdown_summary(payload, output_dir)

    log.info("Figurler ciziliyor...")
    plot_feature_shift_heatmap(numeric_shift_df, output_dir)
    plot_top_shifted_feature_distributions(numeric_shift_df, source_id, target_id, source_df, target_df, primary_population, output_dir)
    plot_label_conditional_direction(label_conditional_df, source_id, target_id, primary_population, output_dir)
    plot_landcover_distribution_comparison(categorical_df, source_id, target_id, primary_population, output_dir)
    plot_prediction_probability_distributions(predictions_df, primary_population, output_dir)
    plot_calibration_curves(calibration_bins_df, primary_population, output_dir)

    log.info(
        "Step9E tamamlandi: diagnosis_categories=%s (%s)",
        part_f_summary["diagnosis_categories"], output_dir,
    )
    return payload


# =============================================================================
# --report-only: metadata/provenance-only regeneration (Part A-F NEVER
# recomputed; no CSV/PNG/Step9A-D file is ever touched here).
# =============================================================================
# The ONLY keys a --report-only regeneration is allowed to change. Everything
# else in the JSON must remain byte-for-byte (value-)identical -- see
# assert_numeric_sections_unchanged().
METADATA_ONLY_FIELDS = frozenset({
    "safe_wording",
    "step9d_overall_conclusion",
    "step9b_predictions_source_path",
    "step9b_metrics_source_path",
    "step9b_predictions_sha256",
    "step9b_metrics_sha256",
    "created_at",
})


def assert_numeric_sections_unchanged(old_payload: dict, new_payload: dict) -> None:
    """
    Fail-fast guard for --report-only: strips ONLY METADATA_ONLY_FIELDS from
    both the old (on-disk) and new (about-to-be-written) payload, then
    requires the remainder to be EXACTLY value-identical (part_a-f numeric
    sections, diagnosis labels/counts, population counts, feature lists,
    AUC/inverse-AUC diagnostics, etc.). Raises Step9EError -- WITHOUT writing
    anything -- if any non-metadata field differs.
    """
    old_stripped = {k: v for k, v in old_payload.items() if k not in METADATA_ONLY_FIELDS}
    new_stripped = {k: v for k, v in new_payload.items() if k not in METADATA_ONLY_FIELDS}
    if old_stripped != new_stripped:
        all_keys = sorted(set(old_stripped) | set(new_stripped))
        diffs = [k for k in all_keys if old_stripped.get(k) != new_stripped.get(k)]
        raise Step9EError(
            "--report-only FAIL-FAST: bir veya daha fazla NUMERIC/bilimsel "
            f"alan degisti (bu, salt metadata guncellemesi olmasi gereken bir "
            f"islemde ASLA olmamali): {diffs}. Hicbir dosya YAZILMADI."
        )


def _markdown_is_stale(md_text: str, old_safe_wording: str | None, new_safe_wording: str) -> bool:
    """True iff the on-disk markdown still contains the OLD safe_wording text
    and that text has actually changed -- i.e. the markdown is genuinely
    stale and needs rewriting. A markdown that already reflects the current
    wording (e.g. a previous --report-only run already fixed it) is left
    untouched."""
    if not old_safe_wording or old_safe_wording == new_safe_wording:
        return False
    return old_safe_wording in md_text


def regenerate_report_only(source_id: str, target_id: str, force: bool = False) -> dict:
    """
    Report-generation-only regeneration of an EXISTING Step9E audit: reads
    the on-disk distribution_shift_audit.json, updates ONLY safe_wording +
    Step9B provenance paths/hashes + created_at (via the SAME resolver
    run_shift_audit() uses), and rewrites the JSON (and, only if it was
    actually stale, the Markdown). Never recomputes Part A-F, never touches
    any CSV/PNG, never touches any Step9A-D artifact.
    """
    if source_id == target_id:
        raise Step9EError("--source ve --target ayni deney OLAMAZ.")

    output_dir = step9e_output_dir(source_id, target_id)
    json_path = output_dir / "distribution_shift_audit.json"
    if not json_path.exists():
        raise Step9EError(
            f"--report-only: mevcut Step9E ciktisi bulunamadi: {json_path}. "
            "--report-only Part A-F'yi YENIDEN HESAPLAMAZ; once tam bir "
            "run_shift_audit() calistirmasi tamamlanmis olmalidir."
        )
    if not force:
        log.info(
            "--report-only: %s zaten var; --force verilmedigi icin atlaniyor.",
            json_path,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))

    old_payload = json.loads(json_path.read_text(encoding="utf-8"))
    _assert_pair_matches(
        "mevcut distribution_shift_audit.json",
        old_payload.get("source_experiment_id"), old_payload.get("target_experiment_id"),
        source_id, target_id,
    )

    step9b_metrics = load_step9b_metrics(source_id, target_id)
    provenance_and_wording = resolve_step9e_provenance_and_wording(source_id, target_id, step9b_metrics)

    new_payload = dict(old_payload)
    new_payload.update(provenance_and_wording)
    new_payload["created_at"] = datetime.now(timezone.utc).isoformat()

    assert_numeric_sections_unchanged(old_payload, new_payload)

    json_path.write_text(json.dumps(new_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log.info("--report-only: distribution_shift_audit.json guncellendi (yalnizca metadata): %s", json_path)

    md_path = output_dir / "distribution_shift_summary.md"
    rewrote_md = False
    if md_path.exists():
        md_text = md_path.read_text(encoding="utf-8")
        if _markdown_is_stale(md_text, old_payload.get("safe_wording"), new_payload["safe_wording"]):
            write_markdown_summary(new_payload, output_dir)
            rewrote_md = True
            log.info("--report-only: distribution_shift_summary.md eski ifadeyi icerdigi icin yeniden yazildi: %s", md_path)
        else:
            log.info("--report-only: distribution_shift_summary.md zaten guncel/stale degil; DOKUNULMADI: %s", md_path)

    log.info(
        "--report-only TAMAMLANDI [%s <-> %s]: safe_wording (Step9D "
        "overall_conclusion=%s uzerinden) ve Step9B provenance alanlari "
        "guncellendi; markdown_rewritten=%s. Hicbir CSV/PNG/Step9A-D "
        "dosyasina DOKUNULMADI.",
        source_id, target_id, new_payload.get("step9d_overall_conclusion"), rewrote_md,
    )
    return new_payload


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step9E: Manavgat<->Bejís cross-region transferinin "
        "POST-HOC dagilim-kaymasi ve iliski-kaymasi denetimi. Hicbir modeli "
        "YENIDEN EGITMEZ, Step9B/9C ciktilarini DEGISTIRMEZ."
    )
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--report-only", action="store_true",
        help="Part A-F'yi YENIDEN HESAPLAMADAN, yalnizca safe_wording + "
        "Step9B provenance alanlarini + created_at'i gunceller (mevcut "
        "distribution_shift_audit.json'un uzerine yazar).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.report_only:
        regenerate_report_only(source_id=args.source, target_id=args.target, force=args.force)
    else:
        run_shift_audit(source_id=args.source, target_id=args.target, force=args.force)