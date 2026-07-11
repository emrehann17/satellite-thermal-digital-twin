"""
step9b_run_cross_region_transfer.py

Step9B: Step8'in burned-area ASSOCIATION modelini (baseline + baseline+
thermal) BIR bolgede (source) egitip, TAMAMEN BAGIMSIZ bir bolgede (target)
test eder. Iki yon calistirilir (or. manavgat_2021->bejis_2022 VE
bejis_2022->manavgat_2021), her yon icin birden fazla populasyon.

KRITIK ON-ISLEME KURALI: TUM on-isleme (numeric median imputation,
kategorik landcover encoder) YALNIZCA KAYNAK (source) bolgeden fit edilir.
Hedef (target) bolgenin etiketleri ONISLEMEYI ETKILEMEZ, esik secimini
ETKILEMEZ, ve fit'e HICBIR sekilde KATILMAZ (pooled source+target fit
YOKTUR, target fine-tuning/calibration YOKTUR, koordinat/region-identity
feature'i YOKTUR).

Model ailesi/hiperparametreler Step8B ile AYNIDIR (dogrudan
src/step8b_train_baseline_vs_thermal_model.py'den reuse edilir) --
karsilastirma tutarli olsun diye.

Bu bir 30 m yangin tahmin modeli DEGILDIR, operasyonel bir yangin tespit
sistemi DEGILDIR, ve Step7 downscaling modelinin kendisini transfer ETMEZ.

CIKTILAR:
    outputs/cross_region/<source>__<target>/step9b/cross_region_transfer_metrics.json
    outputs/cross_region/<source>__<target>/step9b/cross_region_transfer_predictions.parquet
    outputs/cross_region/<source>__<target>/step9b/cross_region_transfer_predictions.csv
    outputs/cross_region/<source>__<target>/step9b/cross_region_transfer_summary.md
    outputs/cross_region/<source>__<target>/step9b/cross_region_roc_curves.png
    outputs/cross_region/<source>__<target>/step9b/cross_region_pr_curves.png
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, precision_recall_curve,
)

from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT

from src.step9a_audit_cross_region_inputs import (
    ALL_POPULATIONS,
    PRIMARY_POPULATIONS,
    SECONDARY_POPULATIONS,
    SHARED_BASELINE_FEATURES,
    SHARED_THERMAL_MODEL_FEATURES,
    TARGET_COLUMN,
    cross_region_output_root,
    resolve_step8a_dataset_path,
)
from src.step8b_train_baseline_vs_thermal_model import (
    build_pipeline,
    add_spatial_block_id,
    make_spatial_folds,
)
from core.config import (
    STEP8B_RANDOM_SEED,
    STEP8B_SPATIAL_BLOCK_SIZE_CELLS,
    STEP8B_N_SPLITS,
    STEP8B_MIN_POSITIVES_PER_POPULATION,
)

BASE_DIR = PROJECT_ROOT
log, log_file = setup_logger("step9b_run_cross_region_transfer")

MODEL_NAME = "random_forest"  # Step8B ile AYNI model ailesi


class Step9BError(SystemExit):
    """Fail-fast error for Step9B (diğer step'lerle aynı konvansiyon)."""


# =============================================================================
# Veri yukleme + populasyon filtreleme
# =============================================================================
def load_step8a_dataset(experiment_id: str) -> pd.DataFrame:
    path = resolve_step8a_dataset_path(experiment_id)
    if not path.exists():
        raise Step9BError(
            f"'{experiment_id}' icin Step8A veri seti bulunamadi: {path}. "
            "Once Step9A audit'i (ve gerekirse Step8A) calistirin."
        )
    df = pd.read_parquet(path)
    df = add_spatial_block_id(df, STEP8B_SPATIAL_BLOCK_SIZE_CELLS)
    return df


def population_subset(df: pd.DataFrame, population: str) -> pd.DataFrame:
    valid = df[df["valid_for_modeling"] == True] if "valid_for_modeling" in df.columns else df  # noqa: E712
    if population == "all_valid":
        return valid
    if population not in valid.columns:
        return valid.iloc[0:0]
    return valid[valid[population].astype(bool)]


# =============================================================================
# Esik secimi: SADECE kaynak bolgenin kendi spatial-block CV OOF tahminleri
# =============================================================================
def select_threshold_from_source_oof(
    pipeline_template, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
) -> tuple[float, dict]:
    """
    Kaynak (source) bolgenin KENDI spatial-block CV out-of-fold (OOF)
    tahminlerinden, F1'i maksimize eden bir esik secer. Hedef (target)
    etiketleri bu fonksiyona HIC verilmez / hic kullanilmaz.
    """
    try:
        folds, n_splits_used = make_spatial_folds(y, groups, STEP8B_N_SPLITS, STEP8B_RANDOM_SEED)
    except SystemExit:
        return 0.5, {"method": "default_no_cv_possible", "n_splits_used": None}

    oof_prob = np.full(len(y), np.nan)
    for train_idx, test_idx in folds:
        model = clone(pipeline_template)
        model.fit(X.iloc[train_idx], y[train_idx])
        oof_prob[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]

    covered = ~np.isnan(oof_prob)
    if covered.sum() == 0 or len(np.unique(y[covered])) < 2:
        return 0.5, {"method": "default_insufficient_oof_coverage", "n_splits_used": n_splits_used}

    y_oof, p_oof = y[covered], oof_prob[covered]
    thresholds = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y_oof, (p_oof >= t).astype(int), zero_division=0) for t in thresholds]
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), {
        "method": "source_oof_f1_optimal", "n_splits_used": n_splits_used,
        "oof_coverage": int(covered.sum()), "best_f1_on_source_oof": float(f1s[best_idx]),
    }


def compute_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    out = {"positive_count": n_pos, "negative_count": n_neg, "threshold_used": float(threshold)}
    if n_pos == 0 or n_neg == 0:
        out.update({k: None for k in (
            "roc_auc", "pr_auc", "brier_score", "balanced_accuracy",
            "precision", "recall", "f1",
        )})
        return out
    y_pred = (y_prob >= threshold).astype(int)
    out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    out["brier_score"] = float(brier_score_loss(y_true, y_prob))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    return out


# =============================================================================
# Tek yon x tek populasyon
# =============================================================================
def run_one_direction_population(
    source_id: str, target_id: str, population: str,
    source_df: pd.DataFrame, target_df: pd.DataFrame,
) -> dict:
    direction = f"{source_id}_to_{target_id}"
    src_pop = population_subset(source_df, population)
    tgt_pop = population_subset(target_df, population)

    n_src_pos, n_src_neg = int((src_pop[TARGET_COLUMN] == 1).sum()), int((src_pop[TARGET_COLUMN] == 0).sum())
    n_tgt_pos, n_tgt_neg = int((tgt_pop[TARGET_COLUMN] == 1).sum()), int((tgt_pop[TARGET_COLUMN] == 0).sum())

    result = {
        "transfer_direction": direction, "population": population,
        "source_experiment_id": source_id, "target_experiment_id": target_id,
        "source_cell_count": int(len(src_pop)), "target_cell_count": int(len(tgt_pop)),
        "source_positive_count": n_src_pos, "source_negative_count": n_src_neg,
        "target_positive_count": n_tgt_pos, "target_negative_count": n_tgt_neg,
        "target_burned_prevalence": (n_tgt_pos / len(tgt_pop)) if len(tgt_pop) else None,
    }

    if (
        n_src_pos < STEP8B_MIN_POSITIVES_PER_POPULATION or n_src_neg < STEP8B_MIN_POSITIVES_PER_POPULATION
        or n_tgt_pos < STEP8B_MIN_POSITIVES_PER_POPULATION or n_tgt_neg < STEP8B_MIN_POSITIVES_PER_POPULATION
    ):
        result["skipped"] = True
        result["reason"] = (
            f"insufficient_positives_or_negatives (source_pos={n_src_pos}, "
            f"source_neg={n_src_neg}, target_pos={n_tgt_pos}, target_neg={n_tgt_neg}, "
            f"min_required={STEP8B_MIN_POSITIVES_PER_POPULATION})"
        )
        log.warning("[%s/%s] atlaniyor: %s", direction, population, result["reason"])
        return result

    result["skipped"] = False
    y_source = src_pop[TARGET_COLUMN].astype(int).to_numpy()
    y_target = tgt_pop[TARGET_COLUMN].astype(int).to_numpy()
    groups_source = src_pop["spatial_block_id"].to_numpy()

    prediction_rows = pd.DataFrame({
        "transfer_direction": direction, "population": population,
        "target_experiment_id": target_id, "target_cell_id": tgt_pop["cell_id"].to_numpy(),
        "target_spatial_block_id": tgt_pop["spatial_block_id"].to_numpy(),
        "burned": y_target,
    })

    model_metrics: dict = {}
    for model_key, feature_list in (("baseline", SHARED_BASELINE_FEATURES), ("thermal", SHARED_THERMAL_MODEL_FEATURES)):
        X_source = src_pop[feature_list]
        X_target = tgt_pop[feature_list]

        pipeline = build_pipeline(feature_list, MODEL_NAME, STEP8B_RANDOM_SEED)
        # --- SOURCE-ONLY fit: imputer/encoder istatistikleri + siniflandirici
        # yalnizca kaynak bolgeden ogrenilir. Target BURADA HIC GORULMEZ. ---
        pipeline.fit(X_source, y_source)

        threshold, threshold_info = select_threshold_from_source_oof(
            pipeline, X_source, y_source, groups_source,
        )

        target_prob = pipeline.predict_proba(X_target)[:, 1]
        prediction_rows[f"{model_key}_probability"] = target_prob

        metrics = compute_metrics_at_threshold(y_target, target_prob, threshold)
        metrics["threshold_selection"] = threshold_info
        metrics["feature_list"] = feature_list
        model_metrics[model_key] = metrics

    b, t = model_metrics["baseline"], model_metrics["thermal"]
    delta = {
        "delta_auc": (t["roc_auc"] - b["roc_auc"]) if (t["roc_auc"] is not None and b["roc_auc"] is not None) else None,
        "delta_pr_auc": (t["pr_auc"] - b["pr_auc"]) if (t["pr_auc"] is not None and b["pr_auc"] is not None) else None,
        "delta_brier": (t["brier_score"] - b["brier_score"]) if (t["brier_score"] is not None and b["brier_score"] is not None) else None,
    }

    result["baseline_metrics"] = b
    result["thermal_metrics"] = t
    result["delta_metrics"] = delta
    result["predictions"] = prediction_rows
    return result


# =============================================================================
# Orkestrasyon: iki yon x populasyonlar
# =============================================================================
def run_transfer(source_id: str, target_id: str, force: bool = False) -> dict:
    output_dir = cross_region_output_root(source_id, target_id) / "step9b"
    metrics_path = output_dir / "cross_region_transfer_metrics.json"
    if metrics_path.exists() and not force:
        log.info("Step9B ciktisi zaten var (%s); --force verilmedigi icin atlaniyor.", metrics_path)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    log.info("Step9B: %s <-> %s icin veri setleri yukleniyor...", source_id, target_id)
    df_source = load_step8a_dataset(source_id)
    df_target = load_step8a_dataset(target_id)

    directions = [(source_id, target_id), (target_id, source_id)]
    dataset_by_id = {source_id: df_source, target_id: df_target}

    all_results = []
    all_prediction_frames = []
    for src_id, tgt_id in directions:
        for population in ALL_POPULATIONS:
            log.info("[%s -> %s] populasyon=%s calisiyor...", src_id, tgt_id, population)
            res = run_one_direction_population(
                src_id, tgt_id, population, dataset_by_id[src_id], dataset_by_id[tgt_id],
            )
            preds = res.pop("predictions", None)
            all_results.append(res)
            if preds is not None:
                all_prediction_frames.append(preds)
            if not res.get("skipped"):
                log.info(
                    "[%s -> %s / %s] delta_auc=%s delta_pr_auc=%s delta_brier=%s",
                    src_id, tgt_id, population,
                    res["delta_metrics"]["delta_auc"], res["delta_metrics"]["delta_pr_auc"],
                    res["delta_metrics"]["delta_brier"],
                )

    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = (
        pd.concat(all_prediction_frames, ignore_index=True) if all_prediction_frames
        else pd.DataFrame(columns=[
            "transfer_direction", "population", "target_experiment_id", "target_cell_id",
            "target_spatial_block_id", "burned", "baseline_probability", "thermal_probability",
        ])
    )
    parquet_path = output_dir / "cross_region_transfer_predictions.parquet"
    csv_path = output_dir / "cross_region_transfer_predictions.csv"
    predictions_df.to_parquet(parquet_path, index=False)
    predictions_df.to_csv(csv_path, index=False)

    metrics_payload = {
        "source_experiment_id": source_id,
        "target_experiment_id": target_id,
        "model_name": MODEL_NAME,
        "random_seed": STEP8B_RANDOM_SEED,
        "spatial_block_size_cells": STEP8B_SPATIAL_BLOCK_SIZE_CELLS,
        "primary_populations": PRIMARY_POPULATIONS,
        "secondary_populations": SECONDARY_POPULATIONS,
        "preprocessing_rule": (
            "All preprocessing (numeric median imputation, categorical "
            "landcover one-hot encoding) is fitted using SOURCE REGION ONLY. "
            "No target-derived imputation values, no pooled source+target "
            "fitting, no target fine-tuning or calibration, no coordinate or "
            "region-identity features."
        ),
        "threshold_selection_rule": (
            "Classification threshold selected using SOURCE REGION spatial-"
            "block CV out-of-fold predictions only (F1-optimal over a grid), "
            "never using target labels."
        ),
        "results": all_results,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log.info("Metrics JSON yazildi: %s", metrics_path)
    log.info("Predictions yazildi: %s, %s", parquet_path, csv_path)

    write_curve_plots(predictions_df, output_dir)
    write_summary_md(metrics_payload, output_dir)

    return metrics_payload


def write_curve_plots(predictions_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Birincil populasyon(lar) icin, her transfer yonu icin baseline/thermal
    ROC ve PR egrilerini (gercek tahmin verisinden) ciker.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        log.warning("matplotlib kullanılamadı, ROC/PR eğrileri atlanıyor: %s", exc)
        return

    if predictions_df is None or predictions_df.empty:
        log.warning("Tahmin verisi bos; ROC/PR eğrileri atlanıyor.")
        return

    subset = predictions_df[predictions_df["population"].isin(PRIMARY_POPULATIONS)]
    if subset.empty:
        subset = predictions_df
    directions = sorted(subset["transfer_direction"].unique())
    if not directions:
        return

    fig_roc, axes_roc = plt.subplots(1, len(directions), figsize=(6 * len(directions), 5), squeeze=False)
    fig_pr, axes_pr = plt.subplots(1, len(directions), figsize=(6 * len(directions), 5), squeeze=False)

    for i, direction in enumerate(directions):
        d = subset[subset["transfer_direction"] == direction]
        y_true = d["burned"].to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        ax_roc, ax_pr = axes_roc[0][i], axes_pr[0][i]
        for model_key, label in (("baseline_probability", "baseline"), ("thermal_probability", "baseline+thermal")):
            if model_key not in d.columns:
                continue
            fpr, tpr, _ = roc_curve(y_true, d[model_key].to_numpy())
            prec, rec, _ = precision_recall_curve(y_true, d[model_key].to_numpy())
            ax_roc.plot(fpr, tpr, label=label)
            ax_pr.plot(rec, prec, label=label)
        ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.7)
        ax_roc.set_title(direction)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(fontsize=8)
        ax_pr.set_title(direction)
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.legend(fontsize=8)

    fig_roc.suptitle("Cross-Region Transfer ROC Curves (target-region evaluation)")
    fig_roc.tight_layout()
    fig_roc.savefig(output_dir / "cross_region_roc_curves.png", dpi=120)
    plt.close(fig_roc)

    fig_pr.suptitle("Cross-Region Transfer PR Curves (target-region evaluation)")
    fig_pr.tight_layout()
    fig_pr.savefig(output_dir / "cross_region_pr_curves.png", dpi=120)
    plt.close(fig_pr)
    log.info("ROC/PR eğrileri yazıldı: %s", output_dir)


def write_summary_md(metrics_payload: dict, output_dir: Path) -> Path:
    lines = [
        "# Cross-Region Transfer Summary",
        "",
        f"- source: `{metrics_payload['source_experiment_id']}`",
        f"- target: `{metrics_payload['target_experiment_id']}`",
        f"- model: `{metrics_payload['model_name']}`",
        "",
        "This evaluates whether the Step8 burned-area **association** model "
        "(baseline vs. baseline+thermal) generalizes across independent "
        "Mediterranean wildfire regions. It is NOT a 30 m fire prediction "
        "model and NOT an operational fire detection system.",
        "",
        metrics_payload["preprocessing_rule"],
        "",
        metrics_payload["threshold_selection_rule"],
        "",
        "## Results",
        "",
        "| direction | population | skipped | target_prevalence | baseline_auc | thermal_auc | delta_auc | delta_pr_auc | delta_brier |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in metrics_payload["results"]:
        if r.get("skipped"):
            lines.append(f"| {r['transfer_direction']} | {r['population']} | yes ({r.get('reason','')}) | - | - | - | - | - | - |")
            continue
        b, t, d = r["baseline_metrics"], r["thermal_metrics"], r["delta_metrics"]
        lines.append(
            f"| {r['transfer_direction']} | {r['population']} | no | "
            f"{r['target_burned_prevalence']:.4f} | {b['roc_auc']:.4f} | {t['roc_auc']:.4f} | "
            f"{d['delta_auc']:.4f} | {d['delta_pr_auc']:.4f} | {d['delta_brier']:.4f} |"
        )
    lines.extend([
        "", "## Scope note", "",
        "Cross-region transfer of the Step8 ~500 m MCD64A1-cell burned-area "
        "association model only. Not a 30 m fire prediction model, not an "
        "operational fire detection system, does not transfer the Step7 "
        "downscaling model itself.",
    ])
    path = output_dir / "cross_region_transfer_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step9B: Step8 burned-area association modelini bir "
        "bolgede egitip bagimsiz bir bolgede test eder (iki yon)."
    )
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_transfer(source_id=args.source, target_id=args.target, force=args.force)