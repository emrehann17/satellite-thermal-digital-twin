"""
step9c_cross_region_block_bootstrap.py

Step9C: her transfer yonu + degerlendirilen populasyon icin, HEDEF (target)
bolgenin spatial_block_id degerlerini yerine-koyarak (with replacement)
bootstrap'lar ve baseline/thermal ROC-AUC, PR-AUC, Brier skorlari (+
delta'lari) icin %95 percentile guven araliklari hesaplar.

Bu, KLASIK bir p-value DEGILDIR ve sonuc "istatistiksel olarak anlamli"
OLARAK ADLANDIRILMAZ -- yalnizca target-spatial-block bootstrap percentile
araliklaridir.

CIKTILAR:
    outputs/cross_region/<source>__<target>/step9c/cross_region_bootstrap_metrics.json
    outputs/cross_region/<source>__<target>/step9c/cross_region_bootstrap_samples.csv
    outputs/cross_region/<source>__<target>/step9c/cross_region_bootstrap_summary.md
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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from src.step9a_audit_cross_region_inputs import cross_region_output_root

BASE_DIR = PROJECT_ROOT
log, log_file = setup_logger("step9c_cross_region_block_bootstrap")

N_BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_RANDOM_SEED = 42
MAX_ATTEMPTS_MULTIPLIER = 5  # 1000 basarili replikaya ulasmak icin en fazla 5000 deneme


class Step9CError(SystemExit):
    """Fail-fast error for Step9C (diğer step'lerle aynı konvansiyon)."""


def _metrics_for_sample(y_true: np.ndarray, prob: np.ndarray) -> dict | None:
    if len(np.unique(y_true)) < 2:
        return None
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "brier_score": float(brier_score_loss(y_true, prob)),
    }


def bootstrap_one_group(df_group: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    df_group: tek bir (transfer_direction, population) icin TUM hedef-bolge
    satirlari (burned, baseline_probability, thermal_probability,
    target_spatial_block_id).

    Dondurur: her satiri bir basarili replikanin metriklerini iceren bir
    DataFrame (roc_auc/pr_auc/brier_score, baseline + thermal + delta).
    """
    blocks = df_group["target_spatial_block_id"].unique()
    n_blocks = len(blocks)
    if n_blocks == 0:
        return pd.DataFrame()

    # Hizli erisim icin block -> satir indeksleri haritasi
    block_to_indices = {b: df_group.index[df_group["target_spatial_block_id"] == b].to_numpy() for b in blocks}

    records = []
    attempts = 0
    max_attempts = N_BOOTSTRAP_REPLICATES * MAX_ATTEMPTS_MULTIPLIER
    while len(records) < N_BOOTSTRAP_REPLICATES and attempts < max_attempts:
        attempts += 1
        sampled_blocks = rng.choice(blocks, size=n_blocks, replace=True)
        idx = np.concatenate([block_to_indices[b] for b in sampled_blocks])
        sample = df_group.loc[idx]

        y = sample["burned"].to_numpy()
        m_base = _metrics_for_sample(y, sample["baseline_probability"].to_numpy())
        m_therm = _metrics_for_sample(y, sample["thermal_probability"].to_numpy())
        if m_base is None or m_therm is None:
            continue  # degenerate (tek sinif) -- "basarili" sayilmaz, tekrar dene

        records.append({
            "replicate": len(records),
            "baseline_roc_auc": m_base["roc_auc"], "thermal_roc_auc": m_therm["roc_auc"],
            "delta_roc_auc": m_therm["roc_auc"] - m_base["roc_auc"],
            "baseline_pr_auc": m_base["pr_auc"], "thermal_pr_auc": m_therm["pr_auc"],
            "delta_pr_auc": m_therm["pr_auc"] - m_base["pr_auc"],
            "baseline_brier": m_base["brier_score"], "thermal_brier": m_therm["brier_score"],
            "delta_brier": m_therm["brier_score"] - m_base["brier_score"],
        })

    if len(records) < N_BOOTSTRAP_REPLICATES:
        log.warning(
            "Yalnizca %d/%d basarili bootstrap replikasi uretilebildi (max_attempts=%d).",
            len(records), N_BOOTSTRAP_REPLICATES, max_attempts,
        )
    return pd.DataFrame(records)


def _percentile_ci(values: pd.Series) -> tuple[float | None, float | None, float | None]:
    values = values.dropna()
    if len(values) == 0:
        return None, None, None
    lo, hi = float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
    return lo, hi, float(values.mean())


def _interpret_improvement(lo: float | None, hi: float | None) -> str:
    """delta_auc / delta_pr_auc icin: pozitif = iyilesme."""
    if lo is None or hi is None:
        return "uncertain"
    if lo > 0:
        return "positive_bootstrap_support"
    if hi < 0:
        return "negative_bootstrap_support"
    return "uncertain"


def _interpret_brier_improvement(lo: float | None, hi: float | None) -> str:
    """delta_brier icin: NEGATIF = iyilesme (Brier ne kadar dusukse o kadar iyi)."""
    if lo is None or hi is None:
        return "uncertain"
    if hi < 0:
        return "positive_bootstrap_support"
    if lo > 0:
        return "negative_bootstrap_support"
    return "uncertain"


def run_bootstrap(source_id: str, target_id: str, force: bool = False) -> dict:
    step9b_dir = cross_region_output_root(source_id, target_id) / "step9b"
    predictions_path = step9b_dir / "cross_region_transfer_predictions.parquet"
    if not predictions_path.exists():
        raise Step9CError(
            f"Step9B tahmin dosyasi bulunamadi: {predictions_path}. Once Step9B'yi calistirin."
        )

    output_dir = cross_region_output_root(source_id, target_id) / "step9c"
    metrics_path = output_dir / "cross_region_bootstrap_metrics.json"
    if metrics_path.exists() and not force:
        log.info("Step9C ciktisi zaten var (%s); --force verilmedigi icin atlaniyor.", metrics_path)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    predictions_df = pd.read_parquet(predictions_path)
    if predictions_df.empty:
        raise Step9CError(f"Tahmin dosyasi bos: {predictions_path}.")

    rng = np.random.default_rng(BOOTSTRAP_RANDOM_SEED)
    all_groups_result = []
    all_samples_frames = []

    for (direction, population), group in predictions_df.groupby(["transfer_direction", "population"]):
        log.info(
            "[%s / %s] bootstrap basliyor (%d satir, %d benzersiz spatial_block_id)...",
            direction, population, len(group), group["target_spatial_block_id"].nunique(),
        )
        samples = bootstrap_one_group(group.reset_index(drop=True), rng)
        if samples.empty:
            all_groups_result.append({
                "transfer_direction": direction, "population": population,
                "n_successful_replicates": 0,
                "note": "No successful bootstrap replicates (degenerate target distribution or too few blocks).",
            })
            continue

        samples["transfer_direction"] = direction
        samples["population"] = population
        all_samples_frames.append(samples)

        ci = {}
        for metric_key, interp_fn in (
            ("delta_roc_auc", _interpret_improvement),
            ("delta_pr_auc", _interpret_improvement),
            ("delta_brier", _interpret_brier_improvement),
        ):
            lo, hi, mean = _percentile_ci(samples[metric_key])
            ci[metric_key] = {"ci_2_5": lo, "ci_97_5": hi, "mean": mean, "interpretation": interp_fn(lo, hi)}
        for metric_key in (
            "baseline_roc_auc", "thermal_roc_auc", "baseline_pr_auc", "thermal_pr_auc",
            "baseline_brier", "thermal_brier",
        ):
            lo, hi, mean = _percentile_ci(samples[metric_key])
            ci[metric_key] = {"ci_2_5": lo, "ci_97_5": hi, "mean": mean}

        all_groups_result.append({
            "transfer_direction": direction, "population": population,
            "n_successful_replicates": int(len(samples)),
            "n_target_blocks": int(group["target_spatial_block_id"].nunique()),
            "confidence_intervals": ci,
        })
        log.info(
            "[%s / %s] delta_auc CI=[%.4f, %.4f] (%s), delta_pr_auc CI=[%.4f, %.4f] (%s), "
            "delta_brier CI=[%.4f, %.4f] (%s)",
            direction, population,
            ci["delta_roc_auc"]["ci_2_5"] or float("nan"), ci["delta_roc_auc"]["ci_97_5"] or float("nan"),
            ci["delta_roc_auc"]["interpretation"],
            ci["delta_pr_auc"]["ci_2_5"] or float("nan"), ci["delta_pr_auc"]["ci_97_5"] or float("nan"),
            ci["delta_pr_auc"]["interpretation"],
            ci["delta_brier"]["ci_2_5"] or float("nan"), ci["delta_brier"]["ci_97_5"] or float("nan"),
            ci["delta_brier"]["interpretation"],
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    samples_df = pd.concat(all_samples_frames, ignore_index=True) if all_samples_frames else pd.DataFrame()
    samples_path = output_dir / "cross_region_bootstrap_samples.csv"
    samples_df.to_csv(samples_path, index=False)

    payload = {
        "source_experiment_id": source_id,
        "target_experiment_id": target_id,
        "n_bootstrap_replicates_requested": N_BOOTSTRAP_REPLICATES,
        "random_seed": BOOTSTRAP_RANDOM_SEED,
        "method_note": (
            "Percentile 95% confidence intervals from TARGET-REGION spatial-block "
            "bootstrap (blocks resampled with replacement; all rows in a sampled "
            "block are kept together). These are NOT classical p-values and the "
            "result is NOT described as statistically significant."
        ),
        "groups": all_groups_result,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log.info("Bootstrap metrics JSON yazildi: %s", metrics_path)
    log.info("Bootstrap samples CSV yazildi: %s", samples_path)

    write_bootstrap_summary(payload, output_dir)
    return payload


def write_bootstrap_summary(payload: dict, output_dir: Path) -> Path:
    lines = [
        "# Cross-Region Target-Region Spatial-Block Bootstrap",
        "",
        f"- source: `{payload['source_experiment_id']}`",
        f"- target: `{payload['target_experiment_id']}`",
        f"- replicates requested: {payload['n_bootstrap_replicates_requested']}",
        f"- random seed: {payload['random_seed']}",
        "",
        payload["method_note"],
        "",
        "## Results",
        "",
        "| direction | population | n_replicates | delta_auc CI | interp | delta_pr_auc CI | interp | delta_brier CI | interp |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for g in payload["groups"]:
        if g.get("n_successful_replicates", 0) == 0:
            lines.append(f"| {g['transfer_direction']} | {g['population']} | 0 | - | - | - | - | - | - |")
            continue
        ci = g["confidence_intervals"]
        a, p, b = ci["delta_roc_auc"], ci["delta_pr_auc"], ci["delta_brier"]
        lines.append(
            f"| {g['transfer_direction']} | {g['population']} | {g['n_successful_replicates']} | "
            f"[{a['ci_2_5']:.4f}, {a['ci_97_5']:.4f}] | {a['interpretation']} | "
            f"[{p['ci_2_5']:.4f}, {p['ci_97_5']:.4f}] | {p['interpretation']} | "
            f"[{b['ci_2_5']:.4f}, {b['ci_97_5']:.4f}] | {b['interpretation']} |"
        )
    lines.extend([
        "", "## Wording policy", "",
        "- `positive_bootstrap_support`, `uncertain`, `negative_bootstrap_support` only.",
        "- No classical p-values. No claim of statistical significance.",
    ])
    path = output_dir / "cross_region_bootstrap_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step9C: hedef-bolge spatial-block bootstrap ile "
        "cross-region transfer metriklerinin %95 percentile araliklarini hesaplar."
    )
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_bootstrap(source_id=args.source, target_id=args.target, force=args.force)