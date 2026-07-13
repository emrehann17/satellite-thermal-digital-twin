"""
core/step10_shared.py

Step10 ("unsupervised self-calibrated cross-region transfer") icin paylasilan,
BILIMSEL OLARAK KRITIK yardimci fonksiyonlar. Step9'un feature listelerini,
model fabrikasini (build_pipeline) ve namespacing guvenlik desenini REUSE
eder; region-wise z-score, CORAL ve N-yollu (N-way) esli spatial-block
bootstrap gibi Step10'a OZGU yeni istatistiksel mantigi barindirir.

TARGET-LABEL FIREWALL (KRITIK):
    Bu modulun `compute_regionwise_zscore_stats`, `apply_regionwise_zscore`,
    `fit_coral_alignment`, `apply_coral` fonksiyonlarindan HICBIRI etiket
    (y / "burned") PARAMETRESI KABUL ETMEZ -- yalnizca feature matrisi (X)
    alirlar. Bu, fonksiyon imzalarinin kendisiyle YAPISAL olarak garanti
    edilir (bkz. tests/test_step10.py: target-label independence testleri).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from core.config import (
    STEP10_BOOTSTRAP_CI_LOWER_PERCENTILE,
    STEP10_BOOTSTRAP_CI_UPPER_PERCENTILE,
    STEP10_BOOTSTRAP_REPLICATES,
    STEP10_CORAL_LAMBDA,
    STEP10_MIN_VALID_BOOTSTRAP_REPLICATES,
    STEP10_RANDOM_STATE,
)
from core.cross_region_experiment import assert_paths_are_safely_namespaced  # REUSE (aynen)
from core.paths import PROJECT_ROOT
from src.step9a_audit_cross_region_inputs import (
    CATEGORICAL_FEATURES,
    FORBIDDEN_MODEL_COLUMNS,
    PRIMARY_POPULATIONS,
    SHARED_BASELINE_FEATURES,
    SHARED_THERMAL_FEATURES,
    SHARED_THERMAL_MODEL_FEATURES,
    TARGET_COLUMN,
    cross_region_output_root,
    resolve_step8a_dataset_path,
)

EPSILON_STD = 1e-12
MODEL_NAME = "random_forest"  # Step8B/Step9B ile AYNI -- DEGISTIRILMEZ
MODEL_FAMILIES = ("baseline", "thermal")
ADAPTATION_METHODS = ("raw_source_only", "regionwise_zscore", "coral_after_regionwise_zscore")
REGIONWISE_ZSCORE_METADATA_CLASS = "unsupervised_target_covariate_adaptation"
PRIMARY_POPULATION = PRIMARY_POPULATIONS[0]

FEATURE_LISTS = {"baseline": list(SHARED_BASELINE_FEATURES), "thermal": list(SHARED_THERMAL_MODEL_FEATURES)}
NUMERIC_FEATURE_POOL = [f for f in SHARED_THERMAL_MODEL_FEATURES if f not in CATEGORICAL_FEATURES]


class Step10Error(SystemExit):
    """Fail-fast error for Step10 (diğer step'lerle aynı konvansiyon)."""


def check_no_forbidden_features(feature_list: list[str]) -> None:
    leaked = set(feature_list).intersection(FORBIDDEN_MODEL_COLUMNS)
    if leaked:
        raise Step10Error(f"YASAK kolonlar Step10 feature listesine sizmis: {leaked}.")


for _fl in FEATURE_LISTS.values():
    check_no_forbidden_features(_fl)


# =============================================================================
# Path resolvers
# =============================================================================
def step10_output_dir(source_id: str, target_id: str) -> Path:
    return cross_region_output_root(source_id, target_id) / "step10"


def resolve_step8b_predictions_path(experiment_id: str) -> Path:
    from src.step9a_audit_cross_region_inputs import get_experiment_output_root  # lazy (ee gerektirir)
    return get_experiment_output_root(experiment_id) / "step8b" / "step8b_predictions.parquet"


def resolve_step8b_metrics_path(experiment_id: str) -> Path:
    from src.step9a_audit_cross_region_inputs import get_experiment_output_root
    return get_experiment_output_root(experiment_id) / "step8b" / "step8b_model_comparison_metrics.json"


def resolve_step9b_metrics_path(source_id: str, target_id: str) -> Path:
    return cross_region_output_root(source_id, target_id) / "step9b" / "cross_region_transfer_metrics.json"


def resolve_step9b_predictions_path(source_id: str, target_id: str) -> Path:
    return cross_region_output_root(source_id, target_id) / "step9b" / "cross_region_transfer_predictions.parquet"


# =============================================================================
# Hashing / canonical serialization (preregistration manifest icin)
# =============================================================================
def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def compute_analysis_id(scientific_config: dict) -> str:
    return hashlib.sha256(canonical_json(scientific_config).encode("utf-8")).hexdigest()


def git_commit_if_available() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def package_versions() -> dict:
    import sklearn
    return {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__}


# =============================================================================
# Region-wise z-score (LABEL-BLIND -- yalnizca X kabul eder, y PARAMETRESI
# YOKTUR). Kaynak istatistikleri yalnizca kaynak X'ten, hedef istatistikleri
# yalnizca hedef X'ten hesaplanir.
# =============================================================================
def compute_regionwise_zscore_stats(X: pd.DataFrame, numeric_features: list[str]) -> dict:
    """SADECE X (feature matrisi) alir. Etiket parametresi YOKTUR -- bu
    fonksiyonun imzasinin kendisi target-label firewall'un bir parcasidir."""
    stats: dict = {}
    for feature in numeric_features:
        values = pd.to_numeric(X[feature], errors="coerce")
        observed = values.dropna()
        mean = float(observed.mean()) if len(observed) else 0.0
        std = float(observed.std(ddof=0)) if len(observed) else 0.0
        constant_guard = std < EPSILON_STD
        stats[feature] = {
            "mean": mean, "std": (1.0 if constant_guard else std), "raw_std": std,
            "constant_feature_guard_used": bool(constant_guard),
            "n_observed": int(len(observed)), "n_missing": int(values.isna().sum()),
        }
    return stats


def apply_regionwise_zscore(X: pd.DataFrame, stats: dict, numeric_features: list[str]) -> pd.DataFrame:
    """SADECE X alir. z = (x - mean) / std; eksik degerler ONCE bolgenin
    kendi ortalamasiyla doldurulur, bu yuzden transform SONRASI eksik
    degerler 0'a esittir (spec geregi). Kategorik (landcover) DOKUNULMAZ."""
    out = X.copy()
    for feature in numeric_features:
        s = stats[feature]
        values = pd.to_numeric(out[feature], errors="coerce")
        filled = values.fillna(s["mean"])  # eksik -> bolge ortalamasi
        out[feature] = (filled - s["mean"]) / s["std"]  # eksikti -> simdi tam olarak 0.0
    return out


# =============================================================================
# CORAL (yalnizca region-wise z-score SONRASI numeric feature'lar uzerinde).
# LABEL-BLIND -- y PARAMETRESI YOKTUR.
# =============================================================================
def _sym_matrix_power(M: np.ndarray, power: float, eps: float = 1e-12) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(M)  # simetrik -> HER ZAMAN reel
    eigvals_clipped = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(eigvals_clipped ** power) @ eigvecs.T


def fit_coral_alignment(
    Xs_z_numeric: np.ndarray, Xt_z_numeric: np.ndarray, lambda_: float = STEP10_CORAL_LAMBDA,
) -> dict:
    """SADECE numeric X matrislerini (region-wise z-score SONRASI) alir. y
    PARAMETRESI YOKTUR. Cs, Ct, A (whitening+recoloring matrisi) ve
    diagnostikleri dondurur."""
    Cs = np.cov(Xs_z_numeric, rowvar=False, ddof=0) + lambda_ * np.eye(Xs_z_numeric.shape[1])
    Ct = np.cov(Xt_z_numeric, rowvar=False, ddof=0) + lambda_ * np.eye(Xt_z_numeric.shape[1])
    Cs = np.atleast_2d(Cs)
    Ct = np.atleast_2d(Ct)

    Cs_inv_sqrt = _sym_matrix_power(Cs, -0.5)
    Ct_sqrt = _sym_matrix_power(Ct, 0.5)
    A = Cs_inv_sqrt @ Ct_sqrt

    if np.iscomplexobj(A) or not np.isfinite(A).all():
        raise Step10Error("CORAL hizalama matrisi (A) kompleks veya sonlu-olmayan degerler icerdi.")

    return {
        "A": A, "Cs": Cs, "Ct": Ct,
        "condition_number_Cs": float(np.linalg.cond(Cs)),
        "condition_number_Ct": float(np.linalg.cond(Ct)),
        "eigenvalue_floor_used": 1e-12,
        "lambda": lambda_,
    }


def apply_coral(Xs_z_numeric: np.ndarray, coral_fit: dict) -> np.ndarray:
    """SADECE kaynak numeric X matrisini alir. y PARAMETRESI YOKTUR. Hedef
    ASLA donusturulmez (Xt_coral = Xt_z, spec geregi) -- bu fonksiyon
    yalnizca KAYNAK icin cagrilir."""
    Xs_coral = Xs_z_numeric @ coral_fit["A"]
    if np.iscomplexobj(Xs_coral) or not np.isfinite(Xs_coral).all():
        raise Step10Error("CORAL-donusturulmus kaynak matrisi kompleks veya sonlu-olmayan degerler icerdi.")
    return np.real(Xs_coral)


# =============================================================================
# Metrikler -- YALNIZCA threshold-free (ROC-AUC, PR-AUC). Step10 esik
# EKLEMEZ, kalibrasyon YAPMAZ.
# =============================================================================
def compute_threshold_free_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return {"roc_auc": None, "pr_auc": None, "positive_count": n_pos, "negative_count": n_neg}
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "positive_count": n_pos, "negative_count": n_neg,
    }


# =============================================================================
# Target-label firewall dogrulama yardimcisi
# =============================================================================
def assert_label_blind(df: pd.DataFrame, context: str = "") -> None:
    """Verilen DataFrame'in HEDEF ETIKETI (`burned`) icermedigini dogrular --
    Step10B'nin fit/adapt/predict cagrilarindan ONCE cagrilir."""
    if TARGET_COLUMN in df.columns:
        raise Step10Error(
            f"Target-label firewall IHLALI ({context}): '{TARGET_COLUMN}' kolonu "
            "label-blind olmasi gereken bir DataFrame'de bulundu."
        )


# =============================================================================
# N-yollu (N-way) esli (paired) hedef-bolge spatial-block bootstrap.
#
# Step9C/Step9F'teki AYNI block-resample-with-replacement ALGORITMASINI
# kullanir (bkz. src/step9c_cross_region_block_bootstrap.py,
# core/cross_region_experiment.py:paired_spatial_block_bootstrap) ancak TEK
# bir replikada BIRDEN FAZLA (N) olasilik serisini AYNI ANDA degerlendirecek
# sekilde genellenmistir -- boylece TUM serilerin (within/raw/zscore/coral x
# baseline/thermal) HER replikada AYNI orneklenmis bloklari kullanmasi ve
# gecersiz (tek-sinif) replikalarin TUM seriler icin BIRLIKTE gecersiz
# sayilmasi saglanir (Step9C/9F'in ikili/pairwise versiyonlarindan farkli
# olarak burada RETRY YOKTUR -- sabit sayida deneme yapilir, gecersizler
# SAYILIR, YERINE YENISI DENENMEZ; spec geregi).
# =============================================================================
def run_n_way_paired_bootstrap(
    df: pd.DataFrame, block_col: str, y_col: str, prob_columns: dict[str, str],
    n_replicates: int = STEP10_BOOTSTRAP_REPLICATES, random_state: int = STEP10_RANDOM_STATE,
) -> dict:
    rng = np.random.default_rng(random_state)
    blocks = df[block_col].unique()
    n_blocks = len(blocks)
    if n_blocks == 0:
        return {"replicates_df": pd.DataFrame(), "n_requested": n_replicates, "n_valid": 0, "n_invalid_single_class": 0}

    block_to_indices = {b: df.index[df[block_col] == b].to_numpy() for b in blocks}

    records = []
    n_invalid = 0
    for i in range(n_replicates):
        sampled_blocks = rng.choice(blocks, size=n_blocks, replace=True)
        idx = np.concatenate([block_to_indices[b] for b in sampled_blocks])
        sample = df.loc[idx]

        y = sample[y_col].to_numpy()
        if len(np.unique(y)) < 2:
            n_invalid += 1
            continue

        row = {"replicate": i}
        for series_name, col in prob_columns.items():
            prob = sample[col].to_numpy()
            row[f"roc_auc__{series_name}"] = float(roc_auc_score(y, prob))
            row[f"pr_auc__{series_name}"] = float(average_precision_score(y, prob))
        records.append(row)

    replicates_df = pd.DataFrame(records)
    return {
        "replicates_df": replicates_df, "n_requested": n_replicates,
        "n_valid": len(records), "n_invalid_single_class": n_invalid,
    }


def percentile_ci(values: pd.Series) -> tuple[float | None, float | None, float | None]:
    values = values.dropna()
    if len(values) == 0:
        return None, None, None
    lo = float(np.percentile(values, STEP10_BOOTSTRAP_CI_LOWER_PERCENTILE))
    hi = float(np.percentile(values, STEP10_BOOTSTRAP_CI_UPPER_PERCENTILE))
    return lo, hi, float(values.mean())


def is_bootstrap_unstable(n_valid: int) -> bool:
    return n_valid < STEP10_MIN_VALID_BOOTSTRAP_REPLICATES