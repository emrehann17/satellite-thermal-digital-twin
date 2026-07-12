"""
step9f_exploratory_transfer_feature_experiment.py

Step9F: Manavgat 2021 <-> Bejís 2022 arasindaki cross-region transfer
basarisizliginin (Step9B) ve Step9E'nin teshis ettigi dagilim/iliski
kaymalarinin, ONCEDEN SABITLENMIS feature altkumeleri ve acikca etiketlenmis
region-relative temsillerle azalip azalmadigini arastiran KESIFSEL
(exploratory), POST-HOC bir deneydir.

BU DENEY DEGILDIR:
    - tarafsiz (unbiased) bir dis (external) validation
    - Step9'un bir DUZELTMESI
    - transfer-safe bir feature setinin KANITI
    - operasyonel yangin tahmininin KANITI

Manavgat ve Bejís hedef etiketleri Step9E sirasinda ZATEN incelenmisti.
Bu yuzden Step9F'in TUM bulgulari yalnizca KESIFSEL hipotez uretimi olarak
tanimlanmalidir. Step9F'ten SONRA secilen herhangi bir aday, ucuncu bagimsiz
bir wildfire bolgesinde degerlendirilmeden ONCE DONDURULMALIDIR (frozen).

Step9F:
    - hicbir modeli Step9A-Step9E ADINA YENIDEN EGITMEZ (kendi modellerini
      egitir, ama Step9A-Step9E dosyalarina HICBIR SEKILDE YAZMAZ)
    - Step9A/B/C/D/E ciktilarini DEGISTIRMEZ (yalnizca salt-okunur girdi/
      provenance olarak okur)
    - Step8A veri setlerini DEGISTIRMEZ
    - GEE'yi YENIDEN CALISTIRMAZ, predictor'lari YENIDEN URETMEZ
    - hedef etiketleri normalizasyon/esik secimi/fit/kalibrasyon icin
      KULLANMAZ (Regime B "region-relative" temsili dahi yalnizca COVARIATE
      istatistiklerini kullanir, etiketleri DEGIL)
    - tahminleri TERS CEVIRMEZ (inverse AUC yalnizca diagnostic'tir)
    - CLI'dan keyfi feature aramasina IZIN VERMEZ (varyant ailesi kodda
      SABITTIR, bkz. core/cross_region_experiment.py:FIXED_VARIANTS)

GIRDILER (salt-okunur):
    outputs/experiments/<source>/step8a/step8a_500m_modeling_dataset.parquet
    outputs/experiments/<target>/step8a/step8a_500m_modeling_dataset.parquet
    outputs/cross_region/<source>__<target>/step9a/  (provenance, opsiyonel)
    outputs/cross_region/<source>__<target>/step9b/  (metrics.json -- ZORUNLU,
        reprodüksiyon kontrolü icin)
    outputs/cross_region/<source>__<target>/step9c/  (provenance, opsiyonel)
    outputs/cross_region/<source>__<target>/step9d/  (provenance, opsiyonel)
    outputs/cross_region/<source>__<target>/step9e/  (provenance/motivasyon,
        opsiyonel -- feature varyantlarini DINAMIK OLARAK SEC MEZ)

CIKTILAR:
    outputs/cross_region/<source>__<target>/step9f/step9f_experiment_manifest.json
    outputs/cross_region/<source>__<target>/step9f/feature_variant_matrix.csv
    outputs/cross_region/<source>__<target>/step9f/source_oof_metrics.csv
    outputs/cross_region/<source>__<target>/step9f/target_transfer_metrics.csv
    outputs/cross_region/<source>__<target>/step9f/target_predictions.parquet
    outputs/cross_region/<source>__<target>/step9f/target_predictions.csv
    outputs/cross_region/<source>__<target>/step9f/paired_metric_deltas.csv
    outputs/cross_region/<source>__<target>/step9f/spatial_block_bootstrap_deltas.json
    outputs/cross_region/<source>__<target>/step9f/spatial_block_bootstrap_deltas.csv
    outputs/cross_region/<source>__<target>/step9f/exploratory_candidate_screening.csv
    outputs/cross_region/<source>__<target>/step9f/step9f_summary.md
    + 8 figür (bkz. modul sonu)

ZORUNLU IFADELER (rapor bunlari icerir):
    "Step9F is a post-hoc exploratory experiment informed by Step9E. Because
    Manavgat and Bejís labels were already inspected during diagnosis,
    improvements observed here cannot be interpreted as unbiased external
    generalization."
    "Region-relative robust normalization uses unlabeled target covariate
    statistics and is therefore an unsupervised transductive adaptation, not
    pure source-only transfer."
    "A candidate selected here must be frozen without further Manavgat/Bejís
    tuning before testing on a third independent wildfire region."

CLI:
    python src/step9f_exploratory_transfer_feature_experiment.py \
        --source manavgat_2021 --target bejis_2022 --reverse --force
"""

from __future__ import annotations

import argparse
import json
import subprocess
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

from sklearn.metrics import roc_auc_score

from core.config import STEP8B_MIN_POSITIVES_PER_POPULATION, STEP8B_N_SPLITS, STEP8B_RANDOM_SEED
from core.cross_region_experiment import (
    BASELINE_REFERENCE_VARIANT,
    FIXED_VARIANTS,
    ORIGINAL_THERMAL_FEATURES,
    PRIMARY_REFERENCE_VARIANT,
    REGIME_A_LABEL,
    REGIME_B_LABELS,
    REGIME_B_VARIANTS,
    REPRODUCTION_TOLERANCE,
    VARIANT_PURPOSE,
    apply_region_robust_transform,
    assert_paths_are_safely_namespaced,
    bootstrap_support_category,
    compute_region_robust_stats,
    paired_spatial_block_bootstrap,
    resolve_step9_stage_dir,
    run_source_oof,
    select_threshold_from_oof_predictions,
    step9f_output_dir,
)
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from src.step8b_train_baseline_vs_thermal_model import build_pipeline
from src.step9a_audit_cross_region_inputs import (
    ALL_POPULATIONS,
    CATEGORICAL_FEATURES as STEP9A_CATEGORICAL_FEATURES,
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

BASE_DIR = PROJECT_ROOT
log, log_file = setup_logger("step9f_exploratory_transfer_feature_experiment")

pywarnings.filterwarnings("ignore", category=RuntimeWarning)

MODEL_NAME = "random_forest"  # Step8B/Step9B ile AYNI model ailesi -- DEGISTIRILMEZ
RANDOM_STATE = STEP8B_RANDOM_SEED  # Step9B ile AYNI sabit seed -- CLI'dan degistirilebilir (--seed) ama VARSAYILAN ayni
N_SPLITS = STEP8B_N_SPLITS
MIN_POSITIVES_PER_POPULATION = STEP8B_MIN_POSITIVES_PER_POPULATION
DEFAULT_BOOTSTRAP_REPLICATES = 1000

# Regime B icin robust-transform edilecek numeric feature havuzu (butun Regime
# B varyantlarinin numeric feature'larinin ustkumesi). Kategorik
# (landcover_dominant) HARIC tutulur -- o, source-fitted one-hot encoding
# olarak KALIR.
REGIME_B_NUMERIC_FEATURE_POOL = [
    f for f in (list(SHARED_BASELINE_FEATURES) + list(SHARED_THERMAL_FEATURES))
    if f not in STEP9A_CATEGORICAL_FEATURES
]

STEP9F_SAFE_WORDING = [
    "Step9F is a post-hoc exploratory experiment informed by Step9E. Because "
    "Manavgat and Bejís labels were already inspected during diagnosis, "
    "improvements observed here cannot be interpreted as unbiased external "
    "generalization.",
    "Region-relative robust normalization uses unlabeled target covariate "
    "statistics and is therefore an unsupervised transductive adaptation, not "
    "pure source-only transfer.",
    "A candidate selected here must be frozen without further Manavgat/Bejís "
    "tuning before testing on a third independent wildfire region.",
]

STEP9F_NEVER_CLAIMS = [
    "transfer-safe features were validated",
    "cross-region generalization was proven",
    "operational prediction succeeded",
    "statistical significance",
    "causal fire-risk relationships",
    "Step9 was corrected",
    "inverse predictions repair the model",
]


class Step9FError(SystemExit):
    """Fail-fast error for Step9F (diğer step'lerle aynı konvansiyon)."""


# =============================================================================
# Provenance -- Step9A-E ciktilarini SALT-OKUNUR olarak referans alir (var
# olup olmadiklarini kaydeder; hicbirini DEGISTIRMEZ/YENIDEN CALISTIRMAZ).
# =============================================================================
def _load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Provenance JSON okunamadi (%s): %s", path, exc)
        return None


def gather_step9_provenance(source_id: str, target_id: str) -> dict:
    provenance: dict = {}
    for stage in ("step9a", "step9b", "step9c", "step9d", "step9e"):
        stage_dir = resolve_step9_stage_dir(source_id, target_id, stage)
        assert_paths_are_safely_namespaced(source_id, target_id, stage_dir)
        provenance[stage] = {
            "dir": str(stage_dir),
            "exists": stage_dir.exists(),
            "files": sorted(p.name for p in stage_dir.glob("*")) if stage_dir.exists() else [],
        }
    return provenance


def resolve_step9b_metrics_path(source_id: str, target_id: str) -> Path:
    return resolve_step9_stage_dir(source_id, target_id, "step9b") / "cross_region_transfer_metrics.json"


def resolve_step9e_audit_path(source_id: str, target_id: str) -> Path:
    return resolve_step9_stage_dir(source_id, target_id, "step9e") / "distribution_shift_audit.json"


def load_step9b_metrics(source_id: str, target_id: str) -> dict:
    """Step9F'in TEK ZORUNLU salt-okunur Step9 girdisi -- reprodüksiyon
    kontrolu icin gereklidir (bkz. verify_reproduction_against_step9b)."""
    path = resolve_step9b_metrics_path(source_id, target_id)
    assert_paths_are_safely_namespaced(source_id, target_id, path)
    if not path.exists():
        raise Step9FError(
            f"Step9B metrics dosyasi bulunamadi: {path}. Step9F, Step9B'yi "
            "YENIDEN CALISTIRMAZ; once Step9B tamamlanmis olmalidir."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _step9b_reference_metrics(step9b_metrics: dict, direction: str, population: str) -> dict | None:
    for res in step9b_metrics.get("results", []):
        if res.get("transfer_direction") == direction and res.get("population") == population and not res.get("skipped"):
            return res
    return None


# =============================================================================
# Veri yukleme + populasyon yeterlilik kontrolu (Step9B ile AYNI kural)
# =============================================================================
def load_experiment_datasets(source_id: str, target_id: str) -> dict[str, pd.DataFrame]:
    log.info("Step9F: %s ve %s icin Step8A veri setleri yukleniyor (salt-okunur)...", source_id, target_id)
    df_source = load_step8a_dataset(source_id)
    df_target = load_step8a_dataset(target_id)
    return {source_id: df_source, target_id: df_target}


def _sufficiency_check(src_pop: pd.DataFrame, tgt_pop: pd.DataFrame) -> tuple[bool, str, dict]:
    n_src_pos = int((src_pop[TARGET_COLUMN] == 1).sum())
    n_src_neg = int((src_pop[TARGET_COLUMN] == 0).sum())
    n_tgt_pos = int((tgt_pop[TARGET_COLUMN] == 1).sum())
    n_tgt_neg = int((tgt_pop[TARGET_COLUMN] == 0).sum())
    counts = {
        "source_positive_count": n_src_pos, "source_negative_count": n_src_neg,
        "target_positive_count": n_tgt_pos, "target_negative_count": n_tgt_neg,
    }
    sufficient = min(n_src_pos, n_src_neg, n_tgt_pos, n_tgt_neg) >= MIN_POSITIVES_PER_POPULATION
    reason = "" if sufficient else (
        f"insufficient_positives_or_negatives ({counts}, min_required={MIN_POSITIVES_PER_POPULATION})"
    )
    return sufficient, reason, counts


# =============================================================================
# Tek bir (direction, population, regime, variant) calistirmasi
# =============================================================================
def _augment_target_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, base: dict) -> dict:
    """compute_metrics_at_threshold (step9b, reuse edilir) ciktisini Step9F'in
    ekstra istedigi diagnostic alanlarla genisletir. Tahminleri TERS CEVIRMEZ
    -- inverse_roc_auc yalnizca diagnostic amaclidir."""
    out = dict(base)
    n_pos, n_neg = out["positive_count"], out["negative_count"]
    prevalence = (n_pos / (n_pos + n_neg)) if (n_pos + n_neg) else None
    out["prevalence"] = prevalence

    above = y_prob >= threshold
    out["fraction_all_rows_above_threshold"] = float(above.mean()) if len(above) else None
    burned_mask, unburned_mask = (y_true == 1), (y_true == 0)
    out["fraction_burned_rows_above_threshold"] = float(above[burned_mask].mean()) if burned_mask.any() else None
    out["fraction_unburned_rows_above_threshold"] = float(above[unburned_mask].mean()) if unburned_mask.any() else None

    out["mean_predicted_probability"] = float(np.mean(y_prob)) if len(y_prob) else None
    out["mean_probability_burned_rows"] = float(np.mean(y_prob[burned_mask])) if burned_mask.any() else None
    out["mean_probability_unburned_rows"] = float(np.mean(y_prob[unburned_mask])) if unburned_mask.any() else None
    out["calibration_in_the_large"] = (
        (out["mean_predicted_probability"] - prevalence) if prevalence is not None else None
    )

    if n_pos and n_neg:
        out["diagnostic_inverse_roc_auc"] = float(roc_auc_score(y_true, 1.0 - y_prob))
    else:
        out["diagnostic_inverse_roc_auc"] = None

    raw_auc = out.get("roc_auc")
    inv_auc = out.get("diagnostic_inverse_roc_auc")
    out["ranking_reversal_suspected"] = bool(
        raw_auc is not None and inv_auc is not None and raw_auc < 0.45 and inv_auc > 0.55
    )
    return out


def run_one_candidate(
    direction: str, population: str, regime: str, variant: str,
    source_id: str, target_id: str,
    src_pop: pd.DataFrame, tgt_pop: pd.DataFrame,
    random_state: int,
) -> dict:
    """Tek bir (direction, population, regime, variant) icin: source-only fit
    (Regime B'de source/target ONCEDEN region-relative transform edilmis
    olarak gelir) + source spatial-block OOF (esik secimi + source metrikleri)
    + target degerlendirmesi. Hicbir target etiketi fit/esik/kalibrasyona
    KARISMAZ."""
    feature_list = FIXED_VARIANTS[variant]

    y_source = src_pop[TARGET_COLUMN].astype(int).to_numpy()
    y_target = tgt_pop[TARGET_COLUMN].astype(int).to_numpy()
    groups_source = src_pop["spatial_block_id"].to_numpy()

    X_source = src_pop[feature_list]
    X_target = tgt_pop[feature_list]

    pipeline = build_pipeline(feature_list, MODEL_NAME, random_state)
    # --- SOURCE-ONLY fit: imputer/encoder istatistikleri + siniflandirici
    # yalnizca kaynak bolgeden (Regime B'de: kaynagin KENDI region-relative
    # transform edilmis degerlerinden) ogrenilir. Target BURADA HIC GORULMEZ. ---
    pipeline.fit(X_source, y_source)

    oof = run_source_oof(pipeline, X_source, y_source, groups_source, N_SPLITS, random_state)
    threshold, threshold_info = select_threshold_from_oof_predictions(
        y_source, oof.get("oof_prob"), oof.get("covered_mask"),
    )

    if oof.get("covered_mask") is not None and oof["covered_mask"].sum() > 0:
        source_oof_metrics = compute_metrics_at_threshold(
            y_source[oof["covered_mask"]], oof["oof_prob"][oof["covered_mask"]], threshold,
        )
    else:
        source_oof_metrics = {
            "positive_count": int((y_source == 1).sum()), "negative_count": int((y_source == 0).sum()),
            "threshold_used": threshold, "roc_auc": None, "pr_auc": None, "brier_score": None,
            "precision": None, "recall": None, "f1": None,
        }
    source_prevalence = float((y_source == 1).mean()) if len(y_source) else None

    target_prob = pipeline.predict_proba(X_target)[:, 1]
    target_metrics_base = compute_metrics_at_threshold(y_target, target_prob, threshold)
    target_metrics = _augment_target_metrics(y_target, target_prob, threshold, target_metrics_base)

    predictions_df = pd.DataFrame({
        "transfer_direction": direction, "population": population, "regime": regime, "variant": variant,
        "target_experiment_id": target_id,
        "target_cell_id": tgt_pop["cell_id"].to_numpy(),
        "target_spatial_block_id": tgt_pop["spatial_block_id"].to_numpy(),
        "burned": y_target,
        "probability": target_prob,
    })

    return {
        "transfer_direction": direction, "population": population, "regime": regime, "variant": variant,
        "source_experiment_id": source_id, "target_experiment_id": target_id,
        "feature_list": feature_list, "feature_count": len(feature_list),
        "skipped": False,
        "source_row_count": int(len(src_pop)), "target_row_count": int(len(tgt_pop)),
        "source_prevalence": source_prevalence,
        "source_oof": {**source_oof_metrics, "n_splits_used": oof.get("n_splits_used"), "oof_coverage": oof.get("oof_coverage")},
        "threshold_info": threshold_info,
        "target_metrics": target_metrics,
        "predictions": predictions_df,
    }


# =============================================================================
# Ana dongu: tum (direction x population x regime x variant) kombinasyonlari
# =============================================================================
def run_all_candidates(source_id: str, target_id: str, datasets: dict[str, pd.DataFrame], random_state: int) -> dict:
    directions = [(source_id, target_id), (target_id, source_id)]

    # Regime B icin region-relative transform edilmis veri setleri, HER
    # (source,target) yonu icin AYRI hesaplanir (istatistikler her zaman o
    # yonun kendi kaynak/hedef bolgesinden gelir).
    regime_b_datasets: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    normalization_stats: dict = {}
    for src_id, tgt_id in directions:
        src_stats = compute_region_robust_stats(datasets[src_id], REGIME_B_NUMERIC_FEATURE_POOL)
        tgt_stats = compute_region_robust_stats(datasets[tgt_id], REGIME_B_NUMERIC_FEATURE_POOL)
        src_transformed = apply_region_robust_transform(datasets[src_id], src_stats, REGIME_B_NUMERIC_FEATURE_POOL)
        tgt_transformed = apply_region_robust_transform(datasets[tgt_id], tgt_stats, REGIME_B_NUMERIC_FEATURE_POOL)
        regime_b_datasets[(src_id, tgt_id)] = {src_id: src_transformed, tgt_id: tgt_transformed}
        normalization_stats[f"{src_id}_to_{tgt_id}"] = {
            "source_stats": {src_id: src_stats}, "target_stats": {tgt_id: tgt_stats},
        }

    all_candidate_results: list[dict] = []
    all_prediction_frames: list[pd.DataFrame] = []
    skip_log: list[dict] = []

    for src_id, tgt_id in directions:
        direction = f"{src_id}_to_{tgt_id}"
        df_source_a, df_target_a = datasets[src_id], datasets[tgt_id]
        df_source_b, df_target_b = regime_b_datasets[(src_id, tgt_id)][src_id], regime_b_datasets[(src_id, tgt_id)][tgt_id]

        for population in ALL_POPULATIONS:
            src_pop_a = population_subset(df_source_a, population)
            tgt_pop_a = population_subset(df_target_a, population)
            sufficient, reason, counts = _sufficiency_check(src_pop_a, tgt_pop_a)
            if not sufficient:
                log.warning("[%s/%s] atlaniyor (tum varyant/regime'ler): %s", direction, population, reason)
                skip_log.append({"transfer_direction": direction, "population": population, "reason": reason, **counts})
                continue

            src_pop_b = population_subset(df_source_b, population)
            tgt_pop_b = population_subset(df_target_b, population)

            # --- Regime A: strict source-only inductive transfer -- TUM sabit varyantlar ---
            for variant in FIXED_VARIANTS:
                log.info("[%s/%s] Regime A calisiyor: variant=%s", direction, population, variant)
                res = run_one_candidate(
                    direction, population, REGIME_A_LABEL, variant,
                    src_id, tgt_id, src_pop_a, tgt_pop_a, random_state,
                )
                preds = res.pop("predictions")
                all_candidate_results.append(res)
                all_prediction_frames.append(preds)

            # --- Regime B: unsupervised region-relative representation -- YALNIZCA 2 varyant ---
            for variant in REGIME_B_VARIANTS:
                log.info("[%s/%s] Regime B calisiyor: variant=%s", direction, population, variant)
                res = run_one_candidate(
                    direction, population, REGIME_B_LABELS[0], variant,
                    src_id, tgt_id, src_pop_b, tgt_pop_b, random_state,
                )
                preds = res.pop("predictions")
                all_candidate_results.append(res)
                all_prediction_frames.append(preds)

    predictions_df = (
        pd.concat(all_prediction_frames, ignore_index=True) if all_prediction_frames
        else pd.DataFrame(columns=[
            "transfer_direction", "population", "regime", "variant", "target_experiment_id",
            "target_cell_id", "target_spatial_block_id", "burned", "probability",
        ])
    )

    return {
        "candidates": all_candidate_results,
        "predictions_df": predictions_df,
        "skip_log": skip_log,
        "normalization_stats": normalization_stats,
    }


# =============================================================================
# Reprodüksiyon kontrolü: Regime A / original_baseline ve original_thermal,
# MEVCUT Step9B metriklerini (tolerans dahilinde) yeniden uretmeli.
# =============================================================================
def verify_reproduction_against_step9b(candidates: list[dict], step9b_metrics: dict) -> dict:
    checks = []
    all_within_tolerance = True
    for cand in candidates:
        if cand["regime"] != REGIME_A_LABEL or cand["variant"] not in (BASELINE_REFERENCE_VARIANT, PRIMARY_REFERENCE_VARIANT):
            continue
        ref = _step9b_reference_metrics(step9b_metrics, cand["transfer_direction"], cand["population"])
        row = {
            "transfer_direction": cand["transfer_direction"], "population": cand["population"],
            "variant": cand["variant"],
        }
        if ref is None:
            row["status"] = "no_step9b_reference_available"
            checks.append(row)
            continue

        step9b_block = ref.get("baseline_metrics" if cand["variant"] == BASELINE_REFERENCE_VARIANT else "thermal_metrics") or {}
        within = True
        for metric_key, tol in REPRODUCTION_TOLERANCE.items():
            mine, theirs = cand["target_metrics"].get(metric_key), step9b_block.get(metric_key)
            if mine is None or theirs is None:
                row[f"{metric_key}_diff"] = None
                continue
            diff = abs(mine - theirs)
            row[f"{metric_key}_diff"], row[f"{metric_key}_tolerance"] = diff, tol
            if diff > tol:
                within = False
        row["status"] = "within_tolerance" if within else "MISMATCH_BEYOND_TOLERANCE"
        row["within_tolerance"] = within
        all_within_tolerance = all_within_tolerance and within
        checks.append(row)
    return {"checks": checks, "all_within_tolerance": bool(all_within_tolerance)}


# =============================================================================
# Paired point-estimate delta'lar (original_thermal VE original_baseline'a
# karsi, Regime A referanslari kullanilarak -- yalnizca AYNI direction+population
# icinde karsilastirilir).
# =============================================================================
def compute_paired_deltas(candidates: list[dict]) -> pd.DataFrame:
    lookup = {(c["transfer_direction"], c["population"], c["regime"], c["variant"]): c for c in candidates}
    rows = []
    for c in candidates:
        row = {
            "transfer_direction": c["transfer_direction"], "population": c["population"],
            "regime": c["regime"], "variant": c["variant"],
        }
        tm = c["target_metrics"]
        for label, ref_variant in ((PRIMARY_REFERENCE_VARIANT, PRIMARY_REFERENCE_VARIANT), (BASELINE_REFERENCE_VARIANT, BASELINE_REFERENCE_VARIANT)):
            ref = lookup.get((c["transfer_direction"], c["population"], REGIME_A_LABEL, ref_variant))
            prefix = f"vs_{label}"
            if ref is None:
                row[f"delta_roc_auc_{prefix}"] = row[f"delta_pr_auc_{prefix}"] = row[f"delta_brier_{prefix}"] = None
                continue
            rtm = ref["target_metrics"]
            row[f"delta_roc_auc_{prefix}"] = (
                (tm["roc_auc"] - rtm["roc_auc"]) if tm.get("roc_auc") is not None and rtm.get("roc_auc") is not None else None
            )
            row[f"delta_pr_auc_{prefix}"] = (
                (tm["pr_auc"] - rtm["pr_auc"]) if tm.get("pr_auc") is not None and rtm.get("pr_auc") is not None else None
            )
            row[f"delta_brier_{prefix}"] = (
                (tm["brier_score"] - rtm["brier_score"]) if tm.get("brier_score") is not None and rtm.get("brier_score") is not None else None
            )
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Esli (paired) hedef-bolge spatial-block bootstrap: her aday, HER ZAMAN
# Regime A / original_thermal'a karsi karsilastirilir (primary reference).
# =============================================================================
def _predictions_wide_for_group(predictions_df: pd.DataFrame, direction: str, population: str) -> pd.DataFrame:
    subset = predictions_df[
        (predictions_df["transfer_direction"] == direction) & (predictions_df["population"] == population)
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["regime_variant"] = subset["regime"] + "__" + subset["variant"]
    wide = subset.pivot_table(
        index=["target_cell_id", "target_spatial_block_id", "burned"],
        columns="regime_variant", values="probability", aggfunc="first",
    ).reset_index()
    return wide


def run_bootstrap_comparisons(predictions_df: pd.DataFrame, n_replicates: int, seed: int) -> dict:
    reference_col = f"{REGIME_A_LABEL}__{PRIMARY_REFERENCE_VARIANT}"
    all_groups: list[dict] = []
    all_sample_frames: list[pd.DataFrame] = []

    if predictions_df.empty:
        return {"groups": all_groups, "samples_df": pd.DataFrame()}

    for (direction, population), _ in predictions_df.groupby(["transfer_direction", "population"]):
        wide = _predictions_wide_for_group(predictions_df, direction, population)
        if wide.empty or reference_col not in wide.columns:
            continue
        candidate_cols = [c for c in wide.columns if c not in ("target_cell_id", "target_spatial_block_id", "burned", reference_col)]

        for cand_col in candidate_cols:
            regime, variant = cand_col.split("__", 1)
            samples = paired_spatial_block_bootstrap(
                wide, "target_spatial_block_id", "burned", cand_col, reference_col, n_replicates, seed,
            )
            n_valid = samples.attrs.get("n_valid_replicates", len(samples))
            n_skipped = samples.attrs.get("n_skipped_replicates", 0)
            if samples.empty:
                all_groups.append({
                    "transfer_direction": direction, "population": population,
                    "regime": regime, "variant": variant,
                    "n_valid_replicates": 0, "n_skipped_replicates": n_skipped,
                    "note": "No successful bootstrap replicates (degenerate target distribution or too few blocks).",
                })
                continue

            samples = samples.copy()
            samples["transfer_direction"], samples["population"] = direction, population
            samples["regime"], samples["variant"] = regime, variant
            all_sample_frames.append(samples)

            def _ci(col: str) -> tuple[float | None, float | None, float | None]:
                vals = samples[col].dropna()
                if len(vals) == 0:
                    return None, None, None
                return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(vals.mean())

            lo_auc, hi_auc, mean_auc = _ci("delta_roc_auc")
            lo_pr, hi_pr, mean_pr = _ci("delta_pr_auc")
            lo_brier, hi_brier, mean_brier = _ci("delta_brier")

            all_groups.append({
                "transfer_direction": direction, "population": population,
                "regime": regime, "variant": variant,
                "n_valid_replicates": int(n_valid), "n_skipped_replicates": int(n_skipped),
                "delta_roc_auc": {
                    "median": float(samples["delta_roc_auc"].median()), "ci_2_5": lo_auc, "ci_97_5": hi_auc,
                    "mean": mean_auc, "support": bootstrap_support_category(lo_auc, hi_auc, higher_is_better=True),
                },
                "delta_pr_auc": {
                    "median": float(samples["delta_pr_auc"].median()), "ci_2_5": lo_pr, "ci_97_5": hi_pr,
                    "mean": mean_pr, "support": bootstrap_support_category(lo_pr, hi_pr, higher_is_better=True),
                },
                "delta_brier": {
                    "median": float(samples["delta_brier"].median()), "ci_2_5": lo_brier, "ci_97_5": hi_brier,
                    "mean": mean_brier, "support": bootstrap_support_category(lo_brier, hi_brier, higher_is_better=False),
                },
            })

    samples_df = pd.concat(all_sample_frames, ignore_index=True) if all_sample_frames else pd.DataFrame()
    return {"groups": all_groups, "samples_df": samples_df}


# =============================================================================
# Kesifsel (exploratory) aday siralama tablosu -- OTOMATIK bir "bilimsel
# kazanan" ILAN ETMEZ; yalnizca betimsel bir tarama (screening) bayragi
# ureten SABIT bir kural uygular (bkz. candidate_for_third_region_freeze).
# =============================================================================
def build_candidate_screening_table(
    candidates: list[dict], paired_deltas_df: pd.DataFrame, bootstrap_groups: list[dict],
) -> pd.DataFrame:
    lookup = {(c["transfer_direction"], c["population"], c["regime"], c["variant"]): c for c in candidates}
    directions = sorted({c["transfer_direction"] for c in candidates})
    primary_population = PRIMARY_POPULATIONS[0]

    paired_lookup = {
        (r["transfer_direction"], r["population"], r["regime"], r["variant"]): r
        for r in paired_deltas_df.to_dict(orient="records")
    } if not paired_deltas_df.empty else {}
    bootstrap_lookup = {
        (g["transfer_direction"], g["population"], g["regime"], g["variant"]): g for g in bootstrap_groups
    }

    ref_variant_key_tpl = lambda d: (d, primary_population, REGIME_A_LABEL, PRIMARY_REFERENCE_VARIANT)  # noqa: E731

    rows = []
    all_regime_variants = (
        [(REGIME_A_LABEL, v) for v in FIXED_VARIANTS] + [(REGIME_B_LABELS[0], v) for v in REGIME_B_VARIANTS]
    )
    for regime, variant in all_regime_variants:
        row: dict = {
            "regime": regime, "variant": variant, "purpose": VARIANT_PURPOSE.get(variant, ""),
            "feature_count": len(FIXED_VARIANTS[variant]),
            "excluded_feature_groups": sorted(set(FIXED_VARIANTS[PRIMARY_REFERENCE_VARIANT]) - set(FIXED_VARIANTS[variant])),
            "regime_is_source_only": bool(regime == REGIME_A_LABEL),
        }

        delta_roc_list, delta_pr_list, delta_brier_list = [], [], []
        reversal_list, source_oof_drop_list = [], []

        for direction in directions:
            key = (direction, primary_population, regime, variant)
            cand = lookup.get(key)
            paired = paired_lookup.get(key)
            boot = bootstrap_lookup.get(key)
            ref_cand = lookup.get(ref_variant_key_tpl(direction))

            d_roc = paired.get("delta_roc_auc_vs_original_thermal") if paired else None
            d_pr = paired.get("delta_pr_auc_vs_original_thermal") if paired else None
            d_brier = paired.get("delta_brier_vs_original_thermal") if paired else None
            reversal = cand["target_metrics"].get("ranking_reversal_suspected") if cand else None
            src_oof_auc = cand["source_oof"].get("roc_auc") if cand else None
            ref_src_oof_auc = ref_cand["source_oof"].get("roc_auc") if ref_cand else None
            src_oof_drop = (
                (ref_src_oof_auc - src_oof_auc) if src_oof_auc is not None and ref_src_oof_auc is not None else None
            )

            row[f"{direction}__delta_roc_auc_vs_original_thermal"] = d_roc
            row[f"{direction}__delta_pr_auc_vs_original_thermal"] = d_pr
            row[f"{direction}__delta_brier_vs_original_thermal"] = d_brier
            row[f"{direction}__bootstrap_support_roc_auc"] = boot.get("delta_roc_auc", {}).get("support") if boot else None
            row[f"{direction}__bootstrap_support_pr_auc"] = boot.get("delta_pr_auc", {}).get("support") if boot else None
            row[f"{direction}__bootstrap_support_brier"] = boot.get("delta_brier", {}).get("support") if boot else None
            row[f"{direction}__ranking_reversal_suspected"] = reversal
            row[f"{direction}__source_oof_roc_auc"] = src_oof_auc
            row[f"{direction}__source_oof_auc_drop_vs_original_thermal"] = src_oof_drop

            if d_roc is not None:
                delta_roc_list.append(d_roc)
            if d_pr is not None:
                delta_pr_list.append(d_pr)
            if d_brier is not None:
                delta_brier_list.append(d_brier)
            if reversal is not None:
                reversal_list.append(reversal)
            if src_oof_drop is not None:
                source_oof_drop_list.append(src_oof_drop)

        n_dirs = len(directions)
        bidirectional_auc_point_improvement = len(delta_roc_list) == n_dirs and all(v > 0 for v in delta_roc_list)
        bidirectional_pr_point_improvement = len(delta_pr_list) == n_dirs and all(v >= 0 for v in delta_pr_list)
        no_major_brier_degradation = len(delta_brier_list) == n_dirs and all(v <= 0.01 for v in delta_brier_list)
        ranking_reversal_resolved_in_both_directions = len(reversal_list) == n_dirs and all(v is False for v in reversal_list)
        source_oof_auc_drop_within_tolerance = len(source_oof_drop_list) == n_dirs and all(v <= 0.05 for v in source_oof_drop_list)

        row["bidirectional_auc_point_improvement"] = bidirectional_auc_point_improvement
        row["bidirectional_pr_point_improvement"] = bidirectional_pr_point_improvement
        row["no_major_brier_degradation"] = no_major_brier_degradation
        row["ranking_reversal_resolved_in_both_directions"] = ranking_reversal_resolved_in_both_directions
        row["source_oof_auc_drop"] = max(source_oof_drop_list) if source_oof_drop_list else None
        row["source_oof_pr_drop"] = None  # PR-AUC drop bilgisi kaynak OOF PR-AUC'tan da turetilebilir; ROC-AUC birincil kriterdir
        row["source_oof_auc_drop_within_tolerance"] = source_oof_auc_drop_within_tolerance

        row["candidate_for_third_region_freeze"] = bool(
            bidirectional_auc_point_improvement and bidirectional_pr_point_improvement
            and no_major_brier_degradation and ranking_reversal_resolved_in_both_directions
            and source_oof_auc_drop_within_tolerance
        )
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Manifest
# =============================================================================
def _git_commit_if_available() -> str | None:
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


def build_manifest(
    source_id: str, target_id: str, seed: int, bootstrap_replicates: int,
    provenance: dict, normalization_stats: dict, reproduction_check: dict,
    skip_log: list[dict], output_dir: Path,
) -> dict:
    return {
        "audit_type": "post_hoc_exploratory_cross_region_feature_representation_experiment",
        "source_experiment_id": source_id, "target_experiment_id": target_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_if_available(),
        "random_seed": seed,
        "model_configuration": {
            "model_name": MODEL_NAME,
            "note": "Step8B/Step9B ile AYNI RandomForestClassifier konfigurasyonu; hiperparametre TUNING YAPILMAZ.",
        },
        "spatial_block_size_cells": None,  # step8b/step9b sabitinden reuse edilir (bkz. core.config)
        "n_splits": N_SPLITS,
        "min_positives_per_population": MIN_POSITIVES_PER_POPULATION,
        "populations": {"primary": PRIMARY_POPULATIONS, "secondary": SECONDARY_POPULATIONS, "all": ALL_POPULATIONS},
        "fixed_feature_variants": FIXED_VARIANTS,
        "variant_purpose": VARIANT_PURPOSE,
        "regime_a_label": REGIME_A_LABEL,
        "regime_a_variants": list(FIXED_VARIANTS.keys()),
        "regime_b_labels": REGIME_B_LABELS,
        "regime_b_variants": REGIME_B_VARIANTS,
        "source_only_vs_target_adaptive_status": {
            REGIME_A_LABEL: "source_only (strict inductive transfer; target covariates and labels NEVER used for fit/threshold/normalization)",
            REGIME_B_LABELS[0]: "target_covariate_adaptive (unsupervised/transductive; target COVARIATE statistics [median/IQR] used for normalization ONLY -- target LABELS never used)",
        },
        "target_label_use_declaration": (
            "Target labels (burned) are used ONLY for post-hoc evaluation metrics "
            "(reported after prediction) in both regimes. They are NEVER used for "
            "feature normalization, imputation, encoding, model fitting, threshold "
            "selection, or calibration in either Regime A or Regime B."
        ),
        "normalization_statistics": normalization_stats,
        "existing_step9_input_paths": provenance,
        "reproduction_tolerance": REPRODUCTION_TOLERANCE,
        "reproduction_check_result": reproduction_check,
        "skipped_direction_population_combinations": skip_log,
        "bootstrap_replicates_requested": bootstrap_replicates,
        "safe_wording": STEP9F_SAFE_WORDING,
        "never_claims": STEP9F_NEVER_CLAIMS,
        "output_dir": str(output_dir),
    }


# =============================================================================
# Figürler
# =============================================================================
def _safe_savefig(fig, path: Path) -> None:
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=120)
    except Exception as exc:  # noqa: BLE001
        log.warning("Figur yazilamadi (%s): %s", path, exc)
    finally:
        plt.close(fig)


def _candidate_label(regime: str, variant: str) -> str:
    tag = "A" if regime == REGIME_A_LABEL else "B"
    return f"[{tag}] {variant}"


def plot_primary_population_metric_comparison(
    candidates: list[dict], metric_key: str, title: str, filename: str, output_dir: Path,
) -> None:
    primary_population = PRIMARY_POPULATIONS[0]
    rows = [c for c in candidates if c["population"] == primary_population]
    if not rows:
        return
    directions = sorted({c["transfer_direction"] for c in rows})
    regime_variants = (
        [(REGIME_A_LABEL, v) for v in FIXED_VARIANTS] + [(REGIME_B_LABELS[0], v) for v in REGIME_B_VARIANTS]
    )
    labels = [_candidate_label(r, v) for r, v in regime_variants]

    fig, axes = plt.subplots(1, len(directions), figsize=(7 * len(directions), 5), squeeze=False)
    for i, direction in enumerate(directions):
        ax = axes[0][i]
        values = []
        for regime, variant in regime_variants:
            match = next((c for c in rows if c["transfer_direction"] == direction and c["regime"] == regime and c["variant"] == variant), None)
            values.append(match["target_metrics"].get(metric_key) if match else None)
        colors = ["#4C72B0" if r == REGIME_A_LABEL else "#DD8452" for r, _ in regime_variants]
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, [v if v is not None else 0 for v in values], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(metric_key)
        ax.set_title(f"{direction}\n(population={primary_population})", fontsize=9)
        ax.invert_yaxis()
    fig.suptitle(f"Step9F: {title} (blue=Regime A strict source-only, orange=Regime B region-relative)")
    _safe_savefig(fig, output_dir / filename)


def plot_bidirectional_delta_heatmap(paired_deltas_df: pd.DataFrame, output_dir: Path) -> None:
    primary_population = PRIMARY_POPULATIONS[0]
    prim = paired_deltas_df[paired_deltas_df["population"] == primary_population]
    if prim.empty:
        return
    prim = prim.copy()
    prim["candidate_label"] = [_candidate_label(r, v) for r, v in zip(prim["regime"], prim["variant"])]
    pivot = prim.pivot_table(index="candidate_label", columns="transfer_direction", values="delta_roc_auc_vs_original_thermal", aggfunc="first")
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(pivot) + 2))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="RdBu", vmin=-0.1, vmax=0.1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="delta ROC-AUC vs original_thermal (Regime A)")
    ax.set_title(f"Step9F: Bidirectional Delta ROC-AUC (population={primary_population})")
    _safe_savefig(fig, output_dir / "bidirectional_delta_heatmap.png")


def plot_source_vs_target_performance(candidates: list[dict], output_dir: Path) -> None:
    primary_population = PRIMARY_POPULATIONS[0]
    rows = [c for c in candidates if c["population"] == primary_population]
    if not rows:
        return
    directions = sorted({c["transfer_direction"] for c in rows})
    fig, axes = plt.subplots(1, len(directions), figsize=(6 * len(directions), 5), squeeze=False)
    for i, direction in enumerate(directions):
        ax = axes[0][i]
        d_rows = [c for c in rows if c["transfer_direction"] == direction]
        for c in d_rows:
            src_auc = c["source_oof"].get("roc_auc")
            tgt_auc = c["target_metrics"].get("roc_auc")
            if src_auc is None or tgt_auc is None:
                continue
            marker = "o" if c["regime"] == REGIME_A_LABEL else "^"
            ax.scatter(src_auc, tgt_auc, marker=marker, s=60)
            ax.annotate(_candidate_label(c["regime"], c["variant"]), (src_auc, tgt_auc), fontsize=6)
        lims = [0, 1]
        ax.plot(lims, lims, "k--", linewidth=0.6, label="source == target")
        ax.set_xlabel("source spatial-block OOF ROC-AUC")
        ax.set_ylabel("target ROC-AUC")
        ax.set_title(f"{direction} (o=Regime A, ^=Regime B)", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"Step9F: Source OOF vs Target Performance (population={primary_population})")
    _safe_savefig(fig, output_dir / "source_vs_target_performance.png")


def plot_ranking_reversal_diagnostic(candidates: list[dict], output_dir: Path) -> None:
    primary_population = PRIMARY_POPULATIONS[0]
    rows = [c for c in candidates if c["population"] == primary_population]
    if not rows:
        return
    directions = sorted({c["transfer_direction"] for c in rows})
    fig, axes = plt.subplots(1, len(directions), figsize=(7 * len(directions), 5), squeeze=False)
    for i, direction in enumerate(directions):
        ax = axes[0][i]
        d_rows = [c for c in rows if c["transfer_direction"] == direction]
        labels = [_candidate_label(c["regime"], c["variant"]) for c in d_rows]
        raw = [c["target_metrics"].get("roc_auc") for c in d_rows]
        inv = [c["target_metrics"].get("diagnostic_inverse_roc_auc") for c in d_rows]
        y_pos = np.arange(len(labels))
        width = 0.35
        ax.barh(y_pos - width / 2, [v if v is not None else 0 for v in raw], height=width, label="raw ROC-AUC")
        ax.barh(y_pos + width / 2, [v if v is not None else 0 for v in inv], height=width, label="diagnostic inverse ROC-AUC")
        for j, c in enumerate(d_rows):
            if c["target_metrics"].get("ranking_reversal_suspected"):
                ax.text(1.01, j, "reversal suspected", fontsize=6, color="red", va="center")
        ax.axvline(0.5, color="black", linewidth=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0, 1.25)
        ax.set_title(direction, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(
        f"Step9F: Ranking-Reversal Diagnostic (population={primary_population}) -- "
        "inverse AUC is DIAGNOSTIC ONLY, predictions are never inverted"
    )
    _safe_savefig(fig, output_dir / "ranking_reversal_diagnostic.png")


def plot_bootstrap_delta_intervals(bootstrap_groups: list[dict], output_dir: Path) -> None:
    primary_population = PRIMARY_POPULATIONS[0]
    rows = [g for g in bootstrap_groups if g.get("population") == primary_population and "delta_roc_auc" in g]
    if not rows:
        return
    directions = sorted({g["transfer_direction"] for g in rows})
    fig, axes = plt.subplots(1, len(directions), figsize=(6 * len(directions), 5), squeeze=False)
    for i, direction in enumerate(directions):
        ax = axes[0][i]
        d_rows = [g for g in rows if g["transfer_direction"] == direction]
        labels = [_candidate_label(g["regime"], g["variant"]) for g in d_rows]
        medians = [g["delta_roc_auc"]["median"] for g in d_rows]
        los = [g["delta_roc_auc"]["ci_2_5"] for g in d_rows]
        his = [g["delta_roc_auc"]["ci_97_5"] for g in d_rows]
        y_pos = np.arange(len(labels))
        for j, (m, lo, hi) in enumerate(zip(medians, los, his)):
            if m is None or lo is None or hi is None:
                continue
            ax.plot([lo, hi], [j, j], color="gray", linewidth=2)
            ax.plot(m, j, "o", color="black")
        ax.axvline(0, color="red", linewidth=0.7, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("delta ROC-AUC vs original_thermal (95% bootstrap interval)")
        ax.set_title(direction, fontsize=9)
    fig.suptitle(f"Step9F: Target-Region Spatial-Block Bootstrap Delta ROC-AUC (population={primary_population})")
    _safe_savefig(fig, output_dir / "bootstrap_delta_intervals.png")


def plot_probability_distribution_comparison(predictions_df: pd.DataFrame, output_dir: Path) -> None:
    primary_population = PRIMARY_POPULATIONS[0]
    prim = predictions_df[predictions_df["population"] == primary_population]
    if prim.empty:
        return
    directions = sorted(prim["transfer_direction"].unique())
    show_combos = [(REGIME_A_LABEL, PRIMARY_REFERENCE_VARIANT), (REGIME_A_LABEL, "stable_core"), (REGIME_B_LABELS[0], PRIMARY_REFERENCE_VARIANT), (REGIME_B_LABELS[0], "stable_core")]
    fig, axes = plt.subplots(len(directions), len(show_combos), figsize=(4 * len(show_combos), 3.5 * len(directions)), squeeze=False)
    for i, direction in enumerate(directions):
        for j, (regime, variant) in enumerate(show_combos):
            ax = axes[i][j]
            d = prim[(prim["transfer_direction"] == direction) & (prim["regime"] == regime) & (prim["variant"] == variant)]
            if d.empty:
                ax.set_visible(False)
                continue
            burned = d[d["burned"] == 1]["probability"]
            unburned = d[d["burned"] == 0]["probability"]
            ax.hist(unburned, bins=20, alpha=0.5, density=True, label="unburned")
            ax.hist(burned, bins=20, alpha=0.5, density=True, label="burned")
            ax.set_title(f"{direction}\n{_candidate_label(regime, variant)}", fontsize=7)
            ax.legend(fontsize=6)
    fig.suptitle(f"Step9F: Predicted Probability Distributions (population={primary_population})")
    _safe_savefig(fig, output_dir / "probability_distribution_comparison.png")


# =============================================================================
# Markdown ozet
# =============================================================================
def write_markdown_summary(
    source_id: str, target_id: str, manifest: dict, screening_df: pd.DataFrame,
    reproduction_check: dict, output_dir: Path,
) -> Path:
    lines = [
        "# Step9F: Exploratory Cross-Region Feature-Representation Experiment",
        "",
        f"- source: `{source_id}`",
        f"- target: `{target_id}`",
        "",
        "## 1. Scope and post-hoc warning",
        "",
    ]
    for w in STEP9F_SAFE_WORDING:
        lines.append(f"> {w}")
        lines.append("")

    lines.extend(["## 2. Fixed variant definitions", "",
                  "| variant | feature_count | purpose |", "|---|---|---|"])
    for variant, features in FIXED_VARIANTS.items():
        lines.append(f"| `{variant}` | {len(features)} | {VARIANT_PURPOSE.get(variant, '')} |")

    lines.extend(["", "## 3. Strict source-only results (Regime A)", "",
                  f"Regime label: `{REGIME_A_LABEL}`. All {len(FIXED_VARIANTS)} fixed variants evaluated. "
                  "Preprocessing (imputation, encoding, model fit) and threshold selection use SOURCE rows only.",
                  ""])

    lines.extend(["## 4. Unsupervised region-relative results (Regime B)", "",
                  f"Regime labels: `{REGIME_B_LABELS[0]}` / `{REGIME_B_LABELS[1]}`. "
                  f"Only variants evaluated: {REGIME_B_VARIANTS}. Source and target covariates are "
                  "independently robust-normalized (median/IQR) using ONLY each region's own unlabeled "
                  "covariates; target labels are never used for normalization.", ""])

    lines.extend(["## 5. Reproduction check against existing Step9B metrics", "",
                  f"- all_within_tolerance: **{reproduction_check.get('all_within_tolerance')}**", ""])
    for row in reproduction_check.get("checks", []):
        lines.append(f"- {row}")

    lines.extend(["", "## 6-8. Source OOF / target transfer / bootstrap details", "",
                  "See `source_oof_metrics.csv`, `target_transfer_metrics.csv`, "
                  "`spatial_block_bootstrap_deltas.{json,csv}` for full numeric detail.", ""])

    lines.extend(["## 9. Exploratory screening table (primary population)", ""])
    if not screening_df.empty:
        cols = [
            "regime", "variant", "feature_count",
            "bidirectional_auc_point_improvement", "bidirectional_pr_point_improvement",
            "no_major_brier_degradation", "ranking_reversal_resolved_in_both_directions",
            "source_oof_auc_drop_within_tolerance", "candidate_for_third_region_freeze",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for _, r in screening_df.iterrows():
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")

    frozen_candidates = screening_df[screening_df["candidate_for_third_region_freeze"] == True] if not screening_df.empty else pd.DataFrame()  # noqa: E712
    lines.extend(["", "## 10. Candidate freezing recommendation", ""])
    if frozen_candidates.empty:
        lines.append(
            "No candidate satisfied the exploratory screening rule "
            "(`candidate_for_third_region_freeze=true`) on the primary population "
            "in both transfer directions. This is reported as-is; the screening "
            "criteria are NOT weakened after inspecting results."
        )
    else:
        lines.append("Candidates flagged by the exploratory screening rule:")
        for _, r in frozen_candidates.iterrows():
            lines.append(f"- `{r['regime']}` / `{r['variant']}`")
    lines.append("")
    lines.append(
        "\"candidate_for_third_region_freeze is an exploratory screening rule created "
        "after inspecting Manavgat and Bejís. It is not evidence of generalization.\""
    )

    lines.extend(["", "## 11. Required third-region validation", "",
                  STEP9F_SAFE_WORDING[2], ""])

    lines.extend(["## 12. Claim limitations", ""])
    for c in STEP9F_NEVER_CLAIMS:
        lines.append(f"- Never claimed: {c}")

    md_path = output_dir / "step9f_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


# =============================================================================
# Orkestrasyon
# =============================================================================
def planned_output_files(output_dir: Path) -> list[Path]:
    return [
        output_dir / "step9f_experiment_manifest.json",
        output_dir / "feature_variant_matrix.csv",
        output_dir / "source_oof_metrics.csv",
        output_dir / "target_transfer_metrics.csv",
        output_dir / "target_predictions.parquet",
        output_dir / "target_predictions.csv",
        output_dir / "paired_metric_deltas.csv",
        output_dir / "spatial_block_bootstrap_deltas.json",
        output_dir / "spatial_block_bootstrap_deltas.csv",
        output_dir / "exploratory_candidate_screening.csv",
        output_dir / "step9f_summary.md",
        output_dir / "primary_population_auc_comparison.png",
        output_dir / "primary_population_pr_auc_comparison.png",
        output_dir / "primary_population_brier_comparison.png",
        output_dir / "bidirectional_delta_heatmap.png",
        output_dir / "source_vs_target_performance.png",
        output_dir / "ranking_reversal_diagnostic.png",
        output_dir / "bootstrap_delta_intervals.png",
        output_dir / "probability_distribution_comparison.png",
    ]


def build_feature_variant_matrix() -> pd.DataFrame:
    all_features = list(dict.fromkeys(ORIGINAL_THERMAL_FEATURES))
    rows = []
    for variant, features in FIXED_VARIANTS.items():
        row = {"variant": variant, "feature_count": len(features), "purpose": VARIANT_PURPOSE.get(variant, "")}
        for f in all_features:
            row[f] = f in features
        row["regime_a_included"] = True
        row["regime_b_included"] = variant in REGIME_B_VARIANTS
        rows.append(row)
    return pd.DataFrame(rows)


def run_step9f(
    source_id: str, target_id: str, force: bool = False,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES, seed: int = RANDOM_STATE,
) -> dict:
    if source_id == target_id:
        raise Step9FError("--source ve --target ayni deney OLAMAZ.")

    output_dir = step9f_output_dir(source_id, target_id)
    manifest_path = output_dir / "step9f_experiment_manifest.json"
    if manifest_path.exists() and not force:
        log.info("Step9F ciktisi zaten var (%s); --force verilmedigi icin atlaniyor.", manifest_path)
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    log.info("Step9F: %s <-> %s icin provenance (Step9A-E, salt-okunur) toplaniyor...", source_id, target_id)
    provenance = gather_step9_provenance(source_id, target_id)
    step9b_metrics = load_step9b_metrics(source_id, target_id)
    step9e_audit_path = resolve_step9e_audit_path(source_id, target_id)
    if not step9e_audit_path.exists():
        log.warning(
            "Step9E audit dosyasi bulunamadi (%s); Step9F yine de calisir "
            "(Step9E yalnizca motivasyon/provenance icin OKUNUR, dinamik feature "
            "secimi icin KULLANILMAZ).", step9e_audit_path,
        )

    datasets = load_experiment_datasets(source_id, target_id)

    log.info("Step9F: tum (direction x population x regime x variant) kombinasyonlari calistiriliyor...")
    run_result = run_all_candidates(source_id, target_id, datasets, seed)
    candidates = run_result["candidates"]
    predictions_df = run_result["predictions_df"]
    skip_log = run_result["skip_log"]
    normalization_stats = run_result["normalization_stats"]

    log.info("Reprodüksiyon kontrolu: Regime A / original_baseline+original_thermal, Step9B metrikleriyle karsilastiriliyor...")
    reproduction_check = verify_reproduction_against_step9b(candidates, step9b_metrics)
    if not reproduction_check["all_within_tolerance"]:
        mismatches = [c for c in reproduction_check["checks"] if c.get("status") == "MISMATCH_BEYOND_TOLERANCE"]
        raise Step9FError(
            "Step9F reprodüksiyon kontrolu BASARISIZ: Regime A / original_baseline "
            "veya original_thermal, mevcut Step9B metriklerini tolerans dahilinde "
            f"yeniden uretemedi. Uyusmazlıklar: {mismatches}"
        )
    log.info("Reprodüksiyon kontrolu basarili: tum karsilastirmalar tolerans dahilinde.")

    paired_deltas_df = compute_paired_deltas(candidates)

    log.info("Hedef-bolge esli (paired) spatial-block bootstrap calistiriliyor (%d replika)...", bootstrap_replicates)
    bootstrap_result = run_bootstrap_comparisons(predictions_df, bootstrap_replicates, seed)

    screening_df = build_candidate_screening_table(candidates, paired_deltas_df, bootstrap_result["groups"])

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- CSV/parquet ciktilari ---
    build_feature_variant_matrix().to_csv(output_dir / "feature_variant_matrix.csv", index=False)

    source_oof_rows = [{
        "transfer_direction": c["transfer_direction"], "population": c["population"],
        "regime": c["regime"], "variant": c["variant"],
        "source_experiment_id": c["source_experiment_id"], "target_experiment_id": c["target_experiment_id"],
        "source_prevalence": c["source_prevalence"], **{f"source_oof_{k}": v for k, v in c["source_oof"].items()},
        "threshold_selection_method": c["threshold_info"].get("method"),
    } for c in candidates]
    pd.DataFrame(source_oof_rows).to_csv(output_dir / "source_oof_metrics.csv", index=False)

    target_rows = [{
        "transfer_direction": c["transfer_direction"], "population": c["population"],
        "regime": c["regime"], "variant": c["variant"], "feature_count": c["feature_count"],
        "target_row_count": c["target_row_count"], **c["target_metrics"],
    } for c in candidates]
    pd.DataFrame(target_rows).to_csv(output_dir / "target_transfer_metrics.csv", index=False)

    predictions_df.to_parquet(output_dir / "target_predictions.parquet", index=False)
    predictions_df.to_csv(output_dir / "target_predictions.csv", index=False)

    paired_deltas_df.to_csv(output_dir / "paired_metric_deltas.csv", index=False)

    bootstrap_json_path = output_dir / "spatial_block_bootstrap_deltas.json"
    bootstrap_json_path.write_text(
        json.dumps({"groups": bootstrap_result["groups"], "n_replicates_requested": bootstrap_replicates,
                    "random_seed": seed}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    bootstrap_result["samples_df"].to_csv(output_dir / "spatial_block_bootstrap_deltas.csv", index=False)

    screening_df.to_csv(output_dir / "exploratory_candidate_screening.csv", index=False)

    manifest = build_manifest(
        source_id, target_id, seed, bootstrap_replicates, provenance,
        normalization_stats, reproduction_check, skip_log, output_dir,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    write_markdown_summary(source_id, target_id, manifest, screening_df, reproduction_check, output_dir)

    log.info("Figurler ciziliyor...")
    plot_primary_population_metric_comparison(candidates, "roc_auc", "Primary Population ROC-AUC", "primary_population_auc_comparison.png", output_dir)
    plot_primary_population_metric_comparison(candidates, "pr_auc", "Primary Population PR-AUC", "primary_population_pr_auc_comparison.png", output_dir)
    plot_primary_population_metric_comparison(candidates, "brier_score", "Primary Population Brier Score", "primary_population_brier_comparison.png", output_dir)
    plot_bidirectional_delta_heatmap(paired_deltas_df, output_dir)
    plot_source_vs_target_performance(candidates, output_dir)
    plot_ranking_reversal_diagnostic(candidates, output_dir)
    plot_bootstrap_delta_intervals(bootstrap_result["groups"], output_dir)
    plot_probability_distribution_comparison(predictions_df, output_dir)

    frozen = screening_df[screening_df["candidate_for_third_region_freeze"] == True] if not screening_df.empty else pd.DataFrame()  # noqa: E712
    log.info(
        "Step9F tamamlandi: %d aday tarandi, %d aday 'candidate_for_third_region_freeze' bayragini aldi (%s).",
        len(screening_df), len(frozen), output_dir,
    )
    return manifest


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step9F: Manavgat<->Bejís cross-region transferi icin "
        "KESIFSEL, POST-HOC feature-temsili deneyi. Step9A-E'yi DEGISTIRMEZ, "
        "tarafsiz dis validation DEGILDIR, transfer-safe feature setinin "
        "KANITI DEGILDIR."
    )
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_step9f(
        source_id=args.source, target_id=args.target, force=args.force,
        bootstrap_replicates=args.bootstrap_replicates, seed=args.seed,
    )