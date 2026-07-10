"""
step8c_spatial_block_bootstrap_uncertainty.py

Step8C: spatial-block bootstrap UNCERTAINTY/SENSITIVITY analysis for Step8B's
baseline-vs-thermal delta metrics (delta_auc, delta_pr_auc, delta_brier).

WHY:
    Step8B reports point estimates only (no confidence interval or p-value).
    Step8C estimates how much those point estimates would vary under
    resampling, WITHOUT retraining any model, using Step8B's existing
    out-of-fold predictions.

IMPORTANT CONSTRAINTS (do not violate):
    - Step8C does NOT train new models.
    - Step8C does NOT use a random row-level bootstrap. The resampling unit
      is `spatial_block_id` (the same 500 m spatial blocks used for Step8B's
      CV groups) so spatial dependence is respected; if a block is drawn
      more than once, ALL of its rows are repeated that many times.
    - Step8C uses Step8B's existing out-of-fold predictions only.
    - FIRMS is never used. 30 m pixels are never used (samples are Step8A's
      500 m cells, inherited unchanged from Step8B's predictions table).
    - Step8C does NOT claim a formal p-value or classical statistical
      significance. It reports bootstrap interval support only, using the
      wording "bootstrap-supported positive delta" / "point estimate
      positive but CI overlaps zero" (never "statistically significant").

Input (read-only):
    outputs/step8b/step8b_predictions.parquet (preferred)
    outputs/step8b/step8b_predictions.csv (fallback)
    outputs/step8b/step8b_model_comparison_metrics.json (context only)
    outputs/step8a/step8a_500m_modeling_dataset.parquet (only to reconstruct
        spatial_block_id if it is missing from the predictions table)

Output:
    outputs/step8c/step8c_bootstrap_metrics.json
    outputs/step8c/step8c_bootstrap_samples.csv
    outputs/step8c/step8c_summary.md
    outputs/step8c/step8c_delta_auc_distribution.png      (optional)
    outputs/step8c/step8c_delta_pr_auc_distribution.png   (optional)
    outputs/step8c/step8c_monthly_bootstrap_metrics.csv   (optional)

CLI:
    python src/step8c_spatial_block_bootstrap_uncertainty.py --force
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from core.config import (
    STEP8B_SPATIAL_BLOCK_SIZE_CELLS,
    STEP8C_OUTPUT_DIR,
    STEP8C_INPUT_PREDICTIONS,
    STEP8C_N_BOOTSTRAP,
    STEP8C_RANDOM_SEED,
    STEP8C_CI_LOWER,
    STEP8C_CI_UPPER,
    STEP8C_MIN_POSITIVES,
    STEP8C_MIN_MONTH_POSITIVES,
)
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT

BASE_DIR = PROJECT_ROOT
log, log_file = setup_logger("step8c")


class Step8CError(SystemExit):
    """Fail-fast error for Step8C (extends SystemExit like other steps)."""


# =============================================================================
# Constants
# =============================================================================
PRIMARY_POPULATION = "all_valid"
CORE_POPULATIONS = ["all_valid", "cropland_dominant"]
OPTIONAL_DIAGNOSTIC_POPULATIONS = ["burnable_tree_shrub_grass", "burnable_tree_shrub"]

MONTHS = [8, 9, 10]
MONTH_NAMES = {8: "august", 9: "september", 10: "october"}

GAPFILL_FILTERS: list[tuple[str, float | None]] = [
    ("no_filter", None),
    ("lt_0.25", 0.25),
    ("lt_0.50", 0.50),
]

REQUIRED_PREDICTION_COLUMNS = ["burned", "y_prob_baseline", "y_prob_thermal", "population"]

EXPECTED_PREDICTION_COLUMNS = [
    "cell_id", "population", "burned", "burn_month", "spatial_block_id",
    "fold_id", "y_prob_baseline", "y_prob_thermal", "landcover_dominant",
    "burnable_tree_shrub_grass", "burnable_tree_shrub", "observed_fraction",
    "gapfilled_fraction", "valid_30m_fraction",
]


# =============================================================================
# Data loading + validation
# =============================================================================
def load_predictions(input_arg: str | None) -> tuple[pd.DataFrame, Path]:
    parquet_path = BASE_DIR / (input_arg or STEP8C_INPUT_PREDICTIONS)
    csv_path = parquet_path.with_suffix(".csv")

    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            log.info("Step8B tahminleri okundu (parquet): %s (%d satir)", parquet_path, len(df))
            return df, parquet_path
        except Exception as exc:  # noqa: BLE001
            log.warning("Parquet okunamadi (%s: %s); CSV fallback deneniyor.", type(exc).__name__, exc)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        log.info("Step8B tahminleri okundu (CSV fallback): %s (%d satir)", csv_path, len(df))
        return df, csv_path

    raise Step8CError(
        "Step8B tahmin dosyasi bulunamadi. Beklenen: "
        f"{parquet_path} veya {csv_path}. Once Step8B'yi calistirin: "
        "python src/step8b_train_baseline_vs_thermal_model.py --force"
    )


def reconstruct_spatial_block_id(df: pd.DataFrame, step8a_dataset_path: Path | None = None) -> pd.DataFrame:
    """
    Fallback: if spatial_block_id is missing from the predictions table,
    reconstruct it by joining Step8A's dataset on cell_id to recover
    row_500m/col_500m, then re-deriving the same block id Step8B used.
    Fails clearly if this is not possible.

    step8a_dataset_path: verilirse (Kozan-disi deneyler icin, ctx'ten
        turetilir), legacy outputs/step8a/ yerine bu yol kullanilir.
    """
    if "spatial_block_id" in df.columns:
        return df

    log.warning(
        "spatial_block_id tahmin tablosunda yok; Step8A veri setinden "
        "row_500m/col_500m ile yeniden olusturuluyor."
    )
    if "cell_id" not in df.columns:
        raise Step8CError(
            "spatial_block_id yok ve cell_id de yok; yeniden olusturma "
            "imkansiz. Step8B'yi guncel scriptle yeniden calistirin."
        )

    if step8a_dataset_path is not None:
        step8a_path = Path(step8a_dataset_path)
        if not step8a_path.exists():
            step8a_path = step8a_path.with_suffix(".csv")
    else:
        step8a_path = BASE_DIR / "outputs" / "step8a" / "step8a_500m_modeling_dataset.parquet"
        if not step8a_path.exists():
            step8a_path = BASE_DIR / "outputs" / "step8a" / "step8a_500m_modeling_dataset.csv"
    if not step8a_path.exists():
        raise Step8CError(
            "spatial_block_id yeniden olusturulamiyor: Step8A veri seti de "
            f"bulunamadi ({step8a_path}). Step8B'yi guncel scriptle yeniden "
            "calistirin (spatial_block_id iceren tahmin tablosu uretir)."
        )

    step8a_df = (
        pd.read_parquet(step8a_path) if step8a_path.suffix == ".parquet" else pd.read_csv(step8a_path)
    )
    if not {"cell_id", "row_500m", "col_500m"}.issubset(step8a_df.columns):
        raise Step8CError(
            "Step8A veri setinde row_500m/col_500m/cell_id kolonlari yok; "
            "spatial_block_id yeniden olusturulamiyor."
        )

    merged = df.merge(
        step8a_df[["cell_id", "row_500m", "col_500m"]], on="cell_id", how="left"
    )
    if merged["row_500m"].isna().any():
        n_missing = int(merged["row_500m"].isna().sum())
        log.warning(
            "%d satir Step8A ile eslesmedi (cell_id bulunamadi); bu satirlar "
            "spatial_block_id=NaN alacak ve bootstrap disi kalacak.", n_missing,
        )

    block = STEP8B_SPATIAL_BLOCK_SIZE_CELLS
    r_block = (merged["row_500m"] // block).astype("Int64").astype(str)
    c_block = (merged["col_500m"] // block).astype("Int64").astype(str)
    merged["spatial_block_id"] = r_block + "_" + c_block
    merged.loc[merged["row_500m"].isna(), "spatial_block_id"] = np.nan
    return merged.drop(columns=["row_500m", "col_500m"])


def validate_predictions(df: pd.DataFrame) -> list[str]:
    warnings_out: list[str] = []

    missing_required = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in df.columns]
    if missing_required:
        raise Step8CError(
            f"Tahmin tablosunda zorunlu kolonlar eksik: {missing_required}. "
            "Step8B'yi guncel scriptle yeniden calistirin."
        )

    if PRIMARY_POPULATION not in df["population"].unique():
        raise Step8CError(
            f"Birincil populasyon '{PRIMARY_POPULATION}' tahmin tablosunda yok. "
            "Step8B'nin all_valid icin basariyla tamamlanmis olmasi gerekir."
        )

    primary_burned = df.loc[df["population"] == PRIMARY_POPULATION, "burned"].dropna().unique()
    if len(primary_burned) < 2:
        raise Step8CError(
            f"'{PRIMARY_POPULATION}' populasyonunda 'burned' tek sinif "
            f"iceriyor ({primary_burned})."
        )

    if "fold_id" not in df.columns or df["fold_id"].isna().all():
        warnings_out.append(
            "fold_id kolonu yok/tamamen bos; tahminlerin gercekten "
            "out-of-fold oldugu dogrulanamiyor."
        )

    missing_expected = [c for c in EXPECTED_PREDICTION_COLUMNS if c not in df.columns]
    if missing_expected:
        warnings_out.append(
            f"Beklenen (ama zorunlu olmayan) kolonlar eksik: {missing_expected}."
        )

    return warnings_out


# =============================================================================
# Core spatial-block bootstrap
# =============================================================================
def compute_metrics(y: np.ndarray, prob_baseline: np.ndarray, prob_thermal: np.ndarray) -> dict | None:
    """Returns None if the sample has only one class (metrics undefined)."""
    if len(np.unique(y)) < 2:
        return None
    auc_b = float(roc_auc_score(y, prob_baseline))
    auc_t = float(roc_auc_score(y, prob_thermal))
    pr_b = float(average_precision_score(y, prob_baseline))
    pr_t = float(average_precision_score(y, prob_thermal))
    br_b = float(brier_score_loss(y, prob_baseline))
    br_t = float(brier_score_loss(y, prob_thermal))
    return {
        "auc_baseline": auc_b, "auc_thermal": auc_t, "delta_auc": auc_t - auc_b,
        "pr_auc_baseline": pr_b, "pr_auc_thermal": pr_t, "delta_pr_auc": pr_t - pr_b,
        "brier_baseline": br_b, "brier_thermal": br_t, "delta_brier": br_t - br_b,
    }


def build_block_index(df_subset: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    """Maps each unique spatial_block_id to the row-positions (0-based, into
    df_subset) that belong to it. Rows with NaN spatial_block_id are dropped
    (cannot be assigned to a bootstrap block)."""
    valid = df_subset["spatial_block_id"].notna()
    sub = df_subset.loc[valid].reset_index(drop=True)
    unique_blocks = sub["spatial_block_id"].unique()
    block_to_idx = {
        b: sub.index[sub["spatial_block_id"] == b].to_numpy() for b in unique_blocks
    }
    return unique_blocks, block_to_idx, sub


def spatial_block_bootstrap(
    df_subset: pd.DataFrame,
    n_bootstrap: int,
    rng: np.random.Generator,
    population: str,
    analysis_type: str,
    month: str | None = None,
    gapfill_filter: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Runs the spatial-block bootstrap on df_subset (must contain 'burned',
    'y_prob_baseline', 'y_prob_thermal', 'spatial_block_id').

    Returns (sample_rows, successful_metric_dicts). sample_rows is a full
    per-iteration record list (including skipped iterations) for the CSV
    output; successful_metric_dicts is the list of metric dicts used to
    compute percentiles/CI (skipped iterations excluded).
    """
    unique_blocks, block_to_idx, sub = build_block_index(df_subset)
    n_blocks = len(unique_blocks)

    sample_rows: list[dict] = []
    successful: list[dict] = []

    if n_blocks == 0:
        for i in range(n_bootstrap):
            sample_rows.append({
                "population": population, "analysis_type": analysis_type,
                "month": month, "gapfilled_fraction_filter": gapfill_filter,
                "iteration": i, "n_rows": 0, "n_blocks_sampled": 0,
                "positive_count": 0, "negative_count": 0,
                "auc_baseline": None, "auc_thermal": None, "delta_auc": None,
                "pr_auc_baseline": None, "pr_auc_thermal": None, "delta_pr_auc": None,
                "brier_baseline": None, "brier_thermal": None, "delta_brier": None,
                "skipped": True, "skip_reason": "no_spatial_blocks_available",
            })
        return sample_rows, successful

    y_all = sub["burned"].astype(int).to_numpy()
    pb_all = sub["y_prob_baseline"].to_numpy(dtype="float64")
    pt_all = sub["y_prob_thermal"].to_numpy(dtype="float64")

    for i in range(n_bootstrap):
        sampled_blocks = rng.choice(unique_blocks, size=n_blocks, replace=True)
        # Blocks that are drawn more than once contribute their rows that
        # many times (this is the whole point of a block bootstrap -- NOT a
        # random row-level bootstrap).
        idx = np.concatenate([block_to_idx[b] for b in sampled_blocks])

        y = y_all[idx]
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        metrics = compute_metrics(y, pb_all[idx], pt_all[idx])

        row = {
            "population": population, "analysis_type": analysis_type,
            "month": month, "gapfilled_fraction_filter": gapfill_filter,
            "iteration": i, "n_rows": int(len(idx)), "n_blocks_sampled": n_blocks,
            "positive_count": n_pos, "negative_count": n_neg,
        }
        if metrics is None:
            row.update({
                "auc_baseline": None, "auc_thermal": None, "delta_auc": None,
                "pr_auc_baseline": None, "pr_auc_thermal": None, "delta_pr_auc": None,
                "brier_baseline": None, "brier_thermal": None, "delta_brier": None,
                "skipped": True, "skip_reason": "single_class_in_bootstrap_sample",
            })
        else:
            row.update(metrics)
            row.update({"skipped": False, "skip_reason": None})
            successful.append(metrics)
        sample_rows.append(row)

    return sample_rows, successful


def summarize_bootstrap(successful: list[dict], n_requested: int) -> dict:
    """Computes CI (2.5/50/97.5 percentile, mean, std) for each metric, plus
    probability-of-improvement fields and interpretation labels."""
    n_successful = len(successful)
    result: dict = {
        "n_bootstrap_requested": n_requested,
        "n_bootstrap_successful": n_successful,
        "n_bootstrap_skipped": n_requested - n_successful,
    }
    if n_successful == 0:
        result["available"] = False
        return result
    result["available"] = True

    metric_names = [
        "auc_baseline", "auc_thermal", "delta_auc",
        "pr_auc_baseline", "pr_auc_thermal", "delta_pr_auc",
        "brier_baseline", "brier_thermal", "delta_brier",
    ]
    arrays = {m: np.array([s[m] for s in successful], dtype="float64") for m in metric_names}

    for m in metric_names:
        arr = arrays[m]
        p_lo, p_med, p_hi = np.percentile(arr, [STEP8C_CI_LOWER, 50.0, STEP8C_CI_UPPER])
        result[m] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p2_5": float(p_lo),
            "p50": float(p_med),
            "p97_5": float(p_hi),
        }

    result["delta_auc_ci95"] = [result["delta_auc"]["p2_5"], result["delta_auc"]["p97_5"]]
    result["delta_pr_auc_ci95"] = [result["delta_pr_auc"]["p2_5"], result["delta_pr_auc"]["p97_5"]]
    result["delta_brier_ci95"] = [result["delta_brier"]["p2_5"], result["delta_brier"]["p97_5"]]

    result["probability_delta_auc_gt_0"] = float(np.mean(arrays["delta_auc"] > 0))
    result["probability_delta_pr_auc_gt_0"] = float(np.mean(arrays["delta_pr_auc"] > 0))
    result["probability_delta_brier_lt_0"] = float(np.mean(arrays["delta_brier"] < 0))

    result["delta_auc_interpretation"] = interpret_ci(result["delta_auc_ci95"])
    result["delta_pr_auc_interpretation"] = interpret_ci(result["delta_pr_auc_ci95"])
    # Brier: lower is better, so the "improvement" direction is CI entirely < 0.
    result["delta_brier_interpretation"] = interpret_ci(
        [-result["delta_brier_ci95"][1], -result["delta_brier_ci95"][0]]
    )
    return result


def interpret_ci(ci95: list[float]) -> str:
    lo, hi = ci95
    if lo > 0:
        return "positive_bootstrap_support"
    if hi < 0:
        return "negative_bootstrap_support"
    return "uncertain"


def point_estimate_from_predictions(df_subset: pd.DataFrame) -> dict | None:
    """Point estimate computed directly from ALL rows of df_subset (no
    resampling) -- the estimate the bootstrap is quantifying uncertainty
    around. Independent of, but expected to be close to, Step8B's own
    reported point estimate for the same population."""
    y = df_subset["burned"].astype(int).to_numpy()
    return compute_metrics(
        y,
        df_subset["y_prob_baseline"].to_numpy(dtype="float64"),
        df_subset["y_prob_thermal"].to_numpy(dtype="float64"),
    )


# =============================================================================
# Plotting (optional outputs)
# =============================================================================
def plot_delta_distribution(
    bootstrap_by_population: dict[str, list[dict]], metric_key: str, title: str, output_path: Path
) -> Path | None:
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        any_plotted = False
        for pop_name, successful in bootstrap_by_population.items():
            if not successful:
                continue
            vals = [s[metric_key] for s in successful]
            ax.hist(vals, bins=40, alpha=0.5, label=pop_name)
            any_plotted = True
        if not any_plotted:
            plt.close(fig)
            return None
        ax.axvline(0, color="black", linestyle=":", linewidth=1)
        ax.set_xlabel(metric_key)
        ax.set_ylabel("Bootstrap iteration count")
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return output_path
    except Exception as exc:  # noqa: BLE001
        log.warning("Dagilim grafigi basarisiz (%s): %s", output_path.name, exc)
        return None


# =============================================================================
# Main
# =============================================================================
def main(
    input_arg: str | None = None,
    output_dir_arg: str = STEP8C_OUTPUT_DIR,
    force: bool = False,
    n_bootstrap: int = STEP8C_N_BOOTSTRAP,
    random_seed: int = STEP8C_RANDOM_SEED,
    min_positives: int = STEP8C_MIN_POSITIVES,
    min_month_positives: int = STEP8C_MIN_MONTH_POSITIVES,
    ctx: dict | None = None,
) -> dict:
    log.info("=" * 60)
    log.info("STEP 8C BASLIYOR (spatial-block bootstrap uncertainty)")
    log.info("=" * 60)

    out_dir = BASE_DIR / output_dir_arg
    required_outputs = [
        out_dir / "step8c_bootstrap_metrics.json",
        out_dir / "step8c_bootstrap_samples.csv",
        out_dir / "step8c_summary.md",
    ]
    if any(p.exists() for p in required_outputs) and not force:
        present = [p.name for p in required_outputs if p.exists()]
        raise Step8CError(
            "Step8C ciktilari zaten var (" + ", ".join(present)
            + "). Uzerine yazmak icin --force verin."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    df, input_path = load_predictions(input_arg)
    step8a_dataset_path = (
        ctx["step8a_output_dir"] / "step8a_500m_modeling_dataset.parquet"
        if (ctx is not None and not ctx.get("is_kozan"))
        else None
    )
    df = reconstruct_spatial_block_id(df, step8a_dataset_path=step8a_dataset_path)
    warnings_list = validate_predictions(df)

    if "spatial_block_id" not in df.columns or df["spatial_block_id"].isna().all():
        raise Step8CError(
            "spatial_block_id kullanilabilir degil (yok veya tamamen bos); "
            "spatial-block bootstrap calistirilamaz. RANDOM ROW BOOTSTRAP'e "
            "DUSULMEYECEK."
        )

    rng = np.random.default_rng(random_seed)

    # --- Which populations to evaluate ---
    available_populations = list(df["population"].dropna().unique())
    populations_evaluated: list[str] = [p for p in CORE_POPULATIONS if p in available_populations]
    if PRIMARY_POPULATION not in populations_evaluated:
        raise Step8CError(f"'{PRIMARY_POPULATION}' tahmin tablosunda bulunamadi.")

    for pop in OPTIONAL_DIAGNOSTIC_POPULATIONS:
        if pop not in available_populations:
            continue
        n_pos = int((df.loc[df["population"] == pop, "burned"] == 1).sum())
        if n_pos >= min_positives:
            populations_evaluated.append(pop)
            log.info("%s bootstrap'a dahil edildi (positives=%d >= %d).", pop, n_pos, min_positives)
        else:
            log.info(
                "%s bootstrap'a DAHIL EDILMEDI (positives=%d < %d, diagnostic-only).",
                pop, n_pos, min_positives,
            )

    log.info("Degerlendirilecek populasyonlar: %s", populations_evaluated)

    # cross-check warnings mirroring Step8B
    total_burned = int((df.loc[df["population"] == PRIMARY_POPULATION, "burned"] == 1).sum())
    if "cropland_dominant" in populations_evaluated and total_burned > 0:
        crop_burned = int((df.loc[df["population"] == "cropland_dominant", "burned"] == 1).sum())
        if crop_burned / total_burned > 0.9:
            warnings_list.append(
                f"cropland_dominant contains {crop_burned}/{total_burned} "
                f"({crop_burned / total_burned:.1%}) of all_valid burned cells."
            )

    has_gapfill = (
        "gapfilled_fraction" in df.columns and not df["gapfilled_fraction"].isna().all()
    )
    if not has_gapfill:
        warnings_list.append(
            "gapfilled_fraction eksik veya tamamen bos; gap-fill sensitivity "
            "bootstrap ATLANACAK."
        )

    # =========================================================================
    # 1) Overall bootstrap per population
    # =========================================================================
    all_sample_rows: list[dict] = []
    point_estimates: dict[str, dict | None] = {}
    overall_ci: dict[str, dict] = {}
    overall_successful: dict[str, list[dict]] = {}
    skip_ratio_warnings: list[str] = []

    for pop in populations_evaluated:
        df_pop = df[df["population"] == pop]
        point_estimates[pop] = point_estimate_from_predictions(df_pop)

        sample_rows, successful = spatial_block_bootstrap(
            df_pop, n_bootstrap, rng, pop, analysis_type="overall",
        )
        all_sample_rows.extend(sample_rows)
        overall_successful[pop] = successful
        ci = summarize_bootstrap(successful, n_bootstrap)
        overall_ci[pop] = ci
        skipped_frac = ci["n_bootstrap_skipped"] / n_bootstrap if n_bootstrap else 0.0
        if skipped_frac > 0.1:
            skip_ratio_warnings.append(
                f"{pop}: {ci['n_bootstrap_skipped']}/{n_bootstrap} "
                f"({skipped_frac:.1%}) bootstrap ornegi tek-sinif oldugu icin "
                "atlandi."
            )
        log.info(
            "%s (overall): delta_auc CI95=[%.4f, %.4f] (%s); delta_pr_auc CI95=[%.4f, %.4f] (%s)",
            pop, *ci.get("delta_auc_ci95", [np.nan, np.nan]), ci.get("delta_auc_interpretation"),
            *ci.get("delta_pr_auc_ci95", [np.nan, np.nan]), ci.get("delta_pr_auc_interpretation"),
        )
    warnings_list.extend(skip_ratio_warnings)

    # =========================================================================
    # 2) Monthly lead-time bootstrap (existing OOF predictions; no new models)
    # =========================================================================
    monthly_ci: dict[str, dict] = {}
    for pop in populations_evaluated:
        df_pop = df[df["population"] == pop]
        monthly_ci[pop] = {}
        for month in MONTHS:
            month_name = MONTH_NAMES[month]
            pos_mask = (df_pop["burned"] == 1) & (df_pop["burn_month"] == month)
            neg_mask = df_pop["burned"] == 0
            df_month = df_pop.loc[pos_mask | neg_mask]
            n_pos_month = int(pos_mask.sum())

            if n_pos_month < min_month_positives:
                monthly_ci[pop][month_name] = {
                    "available": False, "positive_count": n_pos_month,
                    "reason": f"positives ({n_pos_month}) < min_month_positives ({min_month_positives})",
                }
                if month == 10:
                    warnings_list.append(
                        f"{pop}: October burn_month positives are low "
                        f"(n={n_pos_month} < {min_month_positives}); monthly "
                        "October bootstrap CI unavailable."
                    )
                continue

            sample_rows, successful = spatial_block_bootstrap(
                df_month, n_bootstrap, rng, pop, analysis_type="monthly", month=month_name,
            )
            all_sample_rows.extend(sample_rows)
            ci = summarize_bootstrap(successful, n_bootstrap)
            ci["positive_count"] = n_pos_month
            ci["negative_count"] = int(neg_mask.sum())
            ci["available"] = ci.get("available", False)
            monthly_ci[pop][month_name] = ci

    # =========================================================================
    # 3) Gap-fill sensitivity bootstrap (existing predictions only)
    # =========================================================================
    gapfill_ci: dict[str, dict] = {}
    if has_gapfill:
        sensitivity_populations = [p for p in ("all_valid", "cropland_dominant") if p in populations_evaluated]
        for pop in sensitivity_populations:
            df_pop = df[df["population"] == pop]
            gapfill_ci[pop] = {}
            for label, threshold in GAPFILL_FILTERS:
                if threshold is None:
                    df_filtered = df_pop
                else:
                    gf = df_pop["gapfilled_fraction"]
                    df_filtered = df_pop.loc[gf.notna() & (gf < threshold)]

                if len(df_filtered) == 0:
                    gapfill_ci[pop][label] = {"available": False, "reason": "no_rows_pass_filter"}
                    continue

                sample_rows, successful = spatial_block_bootstrap(
                    df_filtered, n_bootstrap, rng, pop,
                    analysis_type="gapfill_sensitivity", gapfill_filter=label,
                )
                all_sample_rows.extend(sample_rows)
                ci = summarize_bootstrap(successful, n_bootstrap)
                ci["n_rows_after_filter"] = int(len(df_filtered))
                gapfill_ci[pop][label] = ci

    # =========================================================================
    # Write outputs
    # =========================================================================
    samples_df = pd.DataFrame(all_sample_rows)
    samples_path = out_dir / "step8c_bootstrap_samples.csv"
    samples_df.to_csv(samples_path, index=False)

    roc_plot = plot_delta_distribution(
        overall_successful, "delta_auc",
        "Step8C: bootstrap distribution of delta_auc (spatial-block resampling)",
        out_dir / "step8c_delta_auc_distribution.png",
    )
    pr_plot = plot_delta_distribution(
        overall_successful, "delta_pr_auc",
        "Step8C: bootstrap distribution of delta_pr_auc (spatial-block resampling)",
        out_dir / "step8c_delta_pr_auc_distribution.png",
    )

    monthly_csv_rows = [r for r in all_sample_rows if r["analysis_type"] == "monthly"]
    monthly_csv_path = None
    if monthly_csv_rows:
        monthly_csv_path = out_dir / "step8c_monthly_bootstrap_metrics.csv"
        pd.DataFrame(monthly_csv_rows).to_csv(monthly_csv_path, index=False)

    stats = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "step": "step8c_spatial_block_bootstrap_uncertainty",
        "input_predictions_path": str(input_path),
        "n_bootstrap_requested": n_bootstrap,
        "n_bootstrap_successful_by_population": {
            pop: overall_ci[pop]["n_bootstrap_successful"] for pop in populations_evaluated
        },
        "random_seed": random_seed,
        "populations_evaluated": populations_evaluated,
        "overall_point_estimates_from_predictions": point_estimates,
        "bootstrap_ci_by_population": overall_ci,
        "monthly_bootstrap_ci_by_population": monthly_ci,
        "gapfill_sensitivity_bootstrap_ci": gapfill_ci,
        "warnings": warnings_list,
        "no_model_trained": True,
        "no_firms_label_used": True,
        "no_30m_pixel_samples": True,
        "bootstrap_unit": "spatial_block_id",
        "wording_note": (
            "No classical p-value or formal statistical significance is "
            "reported. Interpretation labels ('positive_bootstrap_support', "
            "'uncertain', 'negative_bootstrap_support') describe whether the "
            "95% bootstrap percentile interval excludes zero, not a "
            "hypothesis-test result."
        ),
    }
    stats_path = out_dir / "step8c_bootstrap_metrics.json"
    stats_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")

    summary_path = write_summary_md(
        out_dir, populations_evaluated, point_estimates, overall_ci,
        monthly_ci, gapfill_ci, warnings_list, stats_path,
    )

    log.info("Stats: %s", stats_path)
    log.info("Bootstrap samples: %s", samples_path)
    log.info("Summary: %s", summary_path)
    log.info("=" * 60)
    log.info("STEP 8C TAMAMLANDI (no model trained, spatial-block bootstrap, no FIRMS)")
    log.info("=" * 60)

    return {
        "stats_path": str(stats_path),
        "samples_path": str(samples_path),
        "summary_path": str(summary_path),
        "roc_plot": str(roc_plot) if roc_plot else None,
        "pr_plot": str(pr_plot) if pr_plot else None,
        "monthly_csv": str(monthly_csv_path) if monthly_csv_path else None,
    }


# =============================================================================
# Summary writer
# =============================================================================
def _fmt_ci(ci: dict | None) -> str:
    if ci is None or not ci.get("available"):
        return "unavailable"
    lo, hi = ci["delta_auc_ci95"]
    return f"delta_auc CI95=[{lo:+.4f}, {hi:+.4f}] ({ci['delta_auc_interpretation']})"


def write_summary_md(
    output_dir: Path,
    populations_evaluated: list[str],
    point_estimates: dict,
    overall_ci: dict,
    monthly_ci: dict,
    gapfill_ci: dict,
    warnings_list: list[str],
    stats_path: Path,
) -> Path:
    lines = [
        "# Step8C: Spatial-Block Bootstrap Uncertainty for Step8B Delta Metrics",
        "",
        "## What this step does",
        "",
        "- Step8C estimates **uncertainty** for Step8B's `delta_auc` / "
        "`delta_pr_auc` point estimates using a **spatial-block bootstrap** "
        "over the same 500 m spatial blocks used for Step8B's cross-validation.",
        "- It does **not retrain any model** -- it reuses Step8B's existing "
        "out-of-fold predictions.",
        "- It does **not use 30 m pixels** (samples are Step8A's 500 m cells, "
        "inherited unchanged from Step8B).",
        "- It does **not use FIRMS** anywhere.",
        "- It does **not report a classical p-value or formal statistical "
        "significance** -- only whether the 95% bootstrap percentile "
        "interval excludes zero (\"bootstrap-supported positive delta\") or "
        "not (\"point estimate positive but CI overlaps zero\").",
        "",
        "## Main result",
        "",
    ]
    for pop in ["all_valid", "cropland_dominant"]:
        if pop not in populations_evaluated:
            lines.append(f"- **{pop}**: not evaluated (not present in Step8B predictions).")
            continue
        pe = point_estimates.get(pop)
        ci = overall_ci.get(pop, {})
        if not ci.get("available"):
            lines.append(f"- **{pop}**: bootstrap unavailable (all iterations skipped).")
            continue
        pe_line = (
            f"point estimate delta_auc=`{pe['delta_auc']:+.4f}`, "
            f"delta_pr_auc=`{pe['delta_pr_auc']:+.4f}`" if pe else "point estimate unavailable"
        )
        auc_lo, auc_hi = ci["delta_auc_ci95"]
        pr_lo, pr_hi = ci["delta_pr_auc_ci95"]
        auc_word = (
            "bootstrap-supported positive delta" if ci["delta_auc_interpretation"] == "positive_bootstrap_support"
            else "bootstrap-supported negative delta" if ci["delta_auc_interpretation"] == "negative_bootstrap_support"
            else "point estimate positive but CI overlaps zero" if pe and pe["delta_auc"] > 0
            else "point estimate negative but CI overlaps zero" if pe
            else "uncertain"
        )
        pr_word = (
            "bootstrap-supported positive delta" if ci["delta_pr_auc_interpretation"] == "positive_bootstrap_support"
            else "bootstrap-supported negative delta" if ci["delta_pr_auc_interpretation"] == "negative_bootstrap_support"
            else "point estimate positive but CI overlaps zero" if pe and pe["delta_pr_auc"] > 0
            else "point estimate negative but CI overlaps zero" if pe
            else "uncertain"
        )
        lines.append(
            f"- **{pop}** ({pe_line}): "
            f"delta_auc 95% bootstrap CI=`[{auc_lo:+.4f}, {auc_hi:+.4f}]` -> **{auc_word}**; "
            f"delta_pr_auc 95% bootstrap CI=`[{pr_lo:+.4f}, {pr_hi:+.4f}]` -> **{pr_word}** "
            f"(n_bootstrap_successful={ci['n_bootstrap_successful']})"
        )

    lines.extend(["", "## Monthly lead-time bootstrap CI (existing OOF predictions, no monthly models trained)", ""])
    for pop in populations_evaluated:
        lines.append(f"- **{pop}**:")
        for month_name, ci in monthly_ci.get(pop, {}).items():
            if not ci.get("available"):
                lines.append(f"  - {month_name}: unavailable ({ci.get('reason', 'n/a')})")
            else:
                lo, hi = ci["delta_auc_ci95"]
                lines.append(
                    f"  - {month_name} (n_pos={ci.get('positive_count')}): "
                    f"delta_auc CI95=`[{lo:+.4f}, {hi:+.4f}]` ({ci['delta_auc_interpretation']})"
                )

    lines.extend(["", "## Gap-fill sensitivity bootstrap (existing predictions only, no retraining)", ""])
    if not gapfill_ci:
        lines.append("- Not evaluated (`gapfilled_fraction` missing or empty).")
    for pop, filters in gapfill_ci.items():
        lines.append(f"- **{pop}**:")
        for label, ci in filters.items():
            if not ci.get("available"):
                lines.append(f"  - {label}: unavailable ({ci.get('reason', 'n/a')})")
            else:
                lo, hi = ci["delta_auc_ci95"]
                lines.append(
                    f"  - {label} (n_rows={ci.get('n_rows_after_filter')}): "
                    f"delta_auc CI95=`[{lo:+.4f}, {hi:+.4f}]` ({ci['delta_auc_interpretation']})"
                )

    lines.extend([
        "",
        "## Method",
        "",
        "- Resampling unit: `spatial_block_id` (NOT individual rows). Each "
        "bootstrap iteration draws spatial blocks **with replacement**; if a "
        "block is drawn more than once, all of its rows are repeated that "
        "many times in the bootstrap sample.",
        "- Bootstrap iterations where the resampled data contains only one "
        "class (`burned` all 0 or all 1) are skipped and excluded from the "
        "percentile/CI calculation (but recorded in "
        "`step8c_bootstrap_samples.csv` with `skipped=True`).",
        "",
        f"Full metrics: `{stats_path.name}`",
    ])
    if warnings_list:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {w}" for w in warnings_list[:50])
        if len(warnings_list) > 50:
            lines.append(f"- ... ({len(warnings_list) - 50} more, see stats JSON)")

    path = output_dir / "step8c_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_step8c(ctx: dict | None = None, force: bool = False, **kwargs) -> dict:
    """
    Step8C: Step8B'nin out-of-fold tahminleri uzerinde spatial-block
    bootstrap belirsizlik araligi hesaplar (yeniden egitim YAPMAZ).

    ctx: None ise (varsayilan) legacy Kozan davranisi BIREBIR korunur.
        Verilirse (Kozan-disi): outputs/experiments/<experiment_id>/step8c'ye
        yazar, o deneyin kendi Step8B tahminlerinden okur.
    """
    use_ctx = ctx is not None and not ctx.get("is_kozan")
    output_dir_arg = str(ctx["step8c_output_dir"]) if use_ctx else STEP8C_OUTPUT_DIR
    input_arg = (
        str(ctx["step8b_output_dir"] / "step8b_predictions.parquet")
        if use_ctx else None
    )
    result = main(input_arg=input_arg, output_dir_arg=output_dir_arg, force=force, ctx=ctx, **kwargs)
    if ctx is not None:
        result["experiment_id"] = ctx["experiment_id"]
    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step8C: spatial-block bootstrap uncertainty for Step8B "
        "delta_auc/delta_pr_auc/delta_brier point estimates."
    )
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=STEP8C_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=STEP8C_N_BOOTSTRAP)
    parser.add_argument("--random-seed", type=int, default=STEP8C_RANDOM_SEED)
    parser.add_argument("--min-positives", type=int, default=STEP8C_MIN_POSITIVES)
    parser.add_argument("--min-month-positives", type=int, default=STEP8C_MIN_MONTH_POSITIVES)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        input_arg=args.input,
        output_dir_arg=args.output_dir,
        force=args.force,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.random_seed,
        min_positives=args.min_positives,
        min_month_positives=args.min_month_positives,
    )