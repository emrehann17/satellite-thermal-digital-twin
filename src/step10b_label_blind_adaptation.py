"""
step10b_label_blind_adaptation.py

Step10B: uc adaptasyon rejiminin (raw_source_only, regionwise_zscore,
coral_after_regionwise_zscore) fit/adapt/predict asamasi.

TARGET-LABEL FIREWALL (KRITIK, BU DOSYANIN VAROLUS SEBEBI):
    Bu modul:
        - kaynak (source) X VE y'ye erisebilir
        - hedef (target) X'e erisebilir
        - hedef (target) y'yi YUKLEMEZ VE ERISEMEZ -- `strip_target_to_label_blind()`
          hedef DataFrame'i, "burned" (ve tum diger yasak/label kolonlarini)
          YAPISAL OLARAK cikararak dondurur; `assert_label_blind()` bunu
          calisma-zamaninda dogrular.
        - etiketsiz (unlabeled) hedef tahminlerini YAZAR (step10_predictions.parquet
          "burned" kolonu ICERMEZ).

`generate_predictions_for_direction()` fonksiyonu, hedef DataFrame'inde
label kolonu HIC OLMADAN cagrilabilir olmalidir (bkz.
tests/test_step10.py -- target-label independence testleri).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from core.config import STEP10_RANDOM_STATE, STEP8B_MIN_POSITIVES_PER_POPULATION
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from core.step10_shared import (
    CATEGORICAL_FEATURES,
    FEATURE_LISTS,
    MODEL_FAMILIES,
    MODEL_NAME,
    PRIMARY_POPULATION,
    Step10Error,
    apply_coral,
    apply_regionwise_zscore,
    assert_label_blind,
    compute_regionwise_zscore_stats,
    fit_coral_alignment,
    step10_output_dir,
)
from src.step8b_train_baseline_vs_thermal_model import build_pipeline
from src.step9a_audit_cross_region_inputs import TARGET_COLUMN
from src.step9b_run_cross_region_transfer import load_step8a_dataset, population_subset

BASE_DIR = PROJECT_ROOT
log, log_file = setup_logger("step10b_label_blind_adaptation")

# target_X icin TUTULACAK kolonlar -- "burned" ve diger tum yasak/label
# kolonlari YAPISAL OLARAK DISLANIR (bkz. strip_target_to_label_blind).
_ID_AND_MASK_COLUMNS = ["cell_id", "spatial_block_id", "valid_for_modeling",
                        "burnable_tree_shrub_grass", "burnable_tree_shrub", "all_valid"]


def _numeric_features(feature_list: list[str]) -> list[str]:
    return [f for f in feature_list if f not in CATEGORICAL_FEATURES]


def strip_target_to_label_blind(df_full: pd.DataFrame) -> pd.DataFrame:
    """Hedef DataFrame'i, YALNIZCA id/mask/feature kolonlarini tutarak
    dondurur. "burned" (ve BurnDate/label_source/vb. tum diger yasak
    kolonlar) bu ciktida ASLA bulunmaz -- yapisal garanti."""
    keep = [c for c in _ID_AND_MASK_COLUMNS if c in df_full.columns]
    keep += [c for c in FEATURE_LISTS["thermal"] if c in df_full.columns and c not in keep]
    stripped = df_full[keep].copy()
    assert_label_blind(stripped, context="strip_target_to_label_blind")
    return stripped


def _prediction_frame(
    direction: str, source_id: str, target_id: str, model_family: str, method: str,
    tgt_pop: pd.DataFrame, prob: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame({
        "direction": direction, "source_experiment": source_id, "target_experiment": target_id,
        "population": PRIMARY_POPULATION, "target_cell_id": tgt_pop["cell_id"].to_numpy(),
        "target_spatial_block_id": tgt_pop["spatial_block_id"].to_numpy(),
        "model_family": model_family, "adaptation_method": method,
        "prediction_probability": prob,
    })


def generate_predictions_for_direction(
    source_df: pd.DataFrame, target_X: pd.DataFrame, source_id: str, target_id: str,
    random_state: int = STEP10_RANDOM_STATE,
) -> tuple[pd.DataFrame, dict]:
    """
    source_df: TAM kaynak veri seti (X + y) -- kaynak etiketleri KULLANILIR.
    target_X: YALNIZCA hedef feature/id/mask kolonlari -- "burned" YOKTUR
              (assert_label_blind ile dogrulanir). Bu fonksiyon hedef
              etiketine hicbir sekilde ERISEMEZ.
    """
    assert_label_blind(target_X, context=f"generate_predictions_for_direction({source_id}->{target_id})")
    direction = f"{source_id}_to_{target_id}"

    src_pop = population_subset(source_df, PRIMARY_POPULATION)
    tgt_pop = population_subset(target_X, PRIMARY_POPULATION)

    n_src_pos = int((src_pop[TARGET_COLUMN] == 1).sum())
    n_src_neg = int((src_pop[TARGET_COLUMN] == 0).sum())
    if n_src_pos < STEP8B_MIN_POSITIVES_PER_POPULATION or n_src_neg < STEP8B_MIN_POSITIVES_PER_POPULATION:
        raise Step10Error(
            f"[{direction}] kaynak populasyonda yetersiz pozitif/negatif "
            f"(pos={n_src_pos}, neg={n_src_neg}, min={STEP8B_MIN_POSITIVES_PER_POPULATION})."
        )
    if len(tgt_pop) == 0:
        raise Step10Error(f"[{direction}] hedef populasyon BOS.")

    y_source = src_pop[TARGET_COLUMN].astype(int).to_numpy()

    prediction_frames = []
    adaptation_stats: dict = {}

    for model_family in MODEL_FAMILIES:
        feature_list = FEATURE_LISTS[model_family]
        numeric_feats = _numeric_features(feature_list)

        X_source_raw = src_pop[feature_list]
        X_target_raw = tgt_pop[feature_list]

        # --- raw_source_only: Step9B'nin AYNI build_pipeline + source-only fit'i ---
        pipeline_raw = build_pipeline(feature_list, MODEL_NAME, random_state)
        pipeline_raw.fit(X_source_raw, y_source)
        prob_raw = pipeline_raw.predict_proba(X_target_raw)[:, 1]
        prediction_frames.append(_prediction_frame(direction, source_id, target_id, model_family, "raw_source_only", tgt_pop, prob_raw))

        # --- regionwise_zscore: LABEL-BLIND, yalnizca X ---
        source_stats = compute_regionwise_zscore_stats(X_source_raw, numeric_feats)
        target_stats = compute_regionwise_zscore_stats(X_target_raw, numeric_feats)
        X_source_z = apply_regionwise_zscore(X_source_raw, source_stats, numeric_feats)
        X_target_z = apply_regionwise_zscore(X_target_raw, target_stats, numeric_feats)

        pipeline_z = build_pipeline(feature_list, MODEL_NAME, random_state)
        pipeline_z.fit(X_source_z, y_source)
        prob_z = pipeline_z.predict_proba(X_target_z)[:, 1]
        prediction_frames.append(_prediction_frame(direction, source_id, target_id, model_family, "regionwise_zscore", tgt_pop, prob_z))

        # --- coral_after_regionwise_zscore: LABEL-BLIND, yalnizca X ---
        coral_fit = fit_coral_alignment(X_source_z[numeric_feats].to_numpy(dtype=float), X_target_z[numeric_feats].to_numpy(dtype=float))
        Xs_coral_numeric = apply_coral(X_source_z[numeric_feats].to_numpy(dtype=float), coral_fit)
        X_source_coral = X_source_z.copy()
        X_source_coral[numeric_feats] = Xs_coral_numeric
        X_target_coral = X_target_z.copy()  # Xt_coral = Xt_z (hedef DEGISTIRILMEZ)

        pipeline_c = build_pipeline(feature_list, MODEL_NAME, random_state)
        pipeline_c.fit(X_source_coral, y_source)
        prob_c = pipeline_c.predict_proba(X_target_coral)[:, 1]
        prediction_frames.append(_prediction_frame(direction, source_id, target_id, model_family, "coral_after_regionwise_zscore", tgt_pop, prob_c))

        adaptation_stats[model_family] = {
            "source_regionwise_zscore_stats": source_stats,
            "target_regionwise_zscore_stats": target_stats,
            "coral_diagnostics": {
                "condition_number_Cs": coral_fit["condition_number_Cs"],
                "condition_number_Ct": coral_fit["condition_number_Ct"],
                "eigenvalue_floor_used": coral_fit["eigenvalue_floor_used"],
                "lambda": coral_fit["lambda"],
                "numeric_feature_order": numeric_feats,
            },
        }

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    assert_label_blind(predictions_df, context="generate_predictions_for_direction:output")
    return predictions_df, adaptation_stats


# =============================================================================
# Orkestrasyon
# =============================================================================
def run_step10b(source_id: str, target_id: str, analysis_id: str, force: bool = False, random_state: int = STEP10_RANDOM_STATE) -> dict:
    output_dir = step10_output_dir(source_id, target_id)
    predictions_path = output_dir / "step10_predictions.parquet"
    adaptation_stats_path = output_dir / "step10_adaptation_statistics.json"

    if predictions_path.exists() and adaptation_stats_path.exists() and not force:
        log.info("Step10B ciktilari zaten var; --force verilmedigi icin atlaniyor.")
        return {
            "predictions_path": predictions_path, "adaptation_statistics_path": adaptation_stats_path,
            "skipped": True,
        }

    log.info("Step10B: %s ve %s icin Step8A veri setleri yukleniyor (kaynak: X+y; hedef: YALNIZCA X)...", source_id, target_id)
    df_a = load_step8a_dataset(source_id)
    df_b = load_step8a_dataset(target_id)

    directions = [(source_id, target_id, df_a, df_b), (target_id, source_id, df_b, df_a)]

    all_predictions, all_adaptation_stats = [], {}
    for src_id, tgt_id, src_df_full, tgt_df_full in directions:
        target_X = strip_target_to_label_blind(tgt_df_full)
        log.info(
            "[%s_to_%s] hedef DataFrame label-blind hale getirildi (kolonlar: %s) -- '%s' YOK.",
            src_id, tgt_id, list(target_X.columns), TARGET_COLUMN,
        )
        preds, stats = generate_predictions_for_direction(src_df_full, target_X, src_id, tgt_id, random_state)
        all_predictions.append(preds)
        all_adaptation_stats[f"{src_id}_to_{tgt_id}"] = stats

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df.insert(0, "analysis_id", analysis_id)
    assert_label_blind(predictions_df, context="run_step10b:final_output")

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_df.to_parquet(predictions_path, index=False)
    adaptation_stats_path.write_text(
        json.dumps({"analysis_id": analysis_id, "by_direction": all_adaptation_stats}, indent=2, default=str),
        encoding="utf-8",
    )
    log.info("Step10B tamamlandi: %s, %s", predictions_path, adaptation_stats_path)
    return {
        "predictions_path": predictions_path, "adaptation_statistics_path": adaptation_stats_path,
        "predictions_df": predictions_df, "adaptation_statistics": all_adaptation_stats, "skipped": False,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step10B: label-blind adaptation/prediction (raw/zscore/CORAL).")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--analysis-id", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=STEP10_RANDOM_STATE)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_step10b(source_id=args.source, target_id=args.target, analysis_id=args.analysis_id, force=args.force, random_state=args.seed)