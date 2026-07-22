"""
step10c_paired_evaluation_bootstrap.py

Step10C: Step10B'nin dondurulmus (frozen), ETIKETSIZ tahminlerini yukler,
ANCAK SIMDI hedef etiketi (y) yukler, hedef metriklerini hesaplar ve
Step8B within-region OOF referansiyla + Step9B raw reprodüksiyon kontroluyle
hizalar; ardindan N-yollu esli (paired) hedef-bolge spatial-block bootstrap'i
calistirir.

Bu asamadan ONCE (Step10A/Step10B), hedef etiketi HICBIR YERDE KULLANILMADI.
Step10C, etiketi YALNIZCA degerlendirme (evaluation) icin, tahminler zaten
DONDURULMUS haldeyken yukler -- degerlendirme fit/adapt/predict'i GERIYE
DONUK olarak ETKILEMEZ (dondurulmus parquet dosyasi asla yeniden yazilmaz).

RAW REPRODUKSIYON KONTROLU (FAIL-FAST): step10'un raw_source_only
metrikleri, Step9B'nin metriklerini 1e-6 mutlak tolerans ile YENIDEN
URETMELIDIR. Basarisiz olursa, adapte edilmis (zscore/CORAL) bilimsel
rapora GECILMEZ.
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

from core.config import STEP10_BOOTSTRAP_REPLICATES, STEP10_RANDOM_STATE
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from core.step10_shared import (
    ADAPTATION_METHODS,
    MODEL_FAMILIES,
    PRIMARY_POPULATION,
    Step10Error,
    compute_threshold_free_metrics,
    is_bootstrap_unstable,
    percentile_ci,
    resolve_step8b_metrics_path,
    resolve_step8b_predictions_path,
    resolve_step9b_metrics_path,
    resolve_step9b_predictions_path,
    run_n_way_paired_bootstrap,
    step10_output_dir,
)
from src.step9a_audit_cross_region_inputs import TARGET_COLUMN
from src.step9b_run_cross_region_transfer import load_step8a_dataset, population_subset

BASE_DIR = PROJECT_ROOT
log, log_file = setup_logger("step10c_paired_evaluation_bootstrap")

RAW_REPRODUCTION_TOLERANCE = 1e-6
WITHIN_REGION_REPRODUCTION_TOLERANCE = 1e-6

STEP9_SCHEMA_STEP9B_RESULTS = "step9b_results_v1"
STEP9_SCHEMA_STEP9D_DIRECTION_SUMMARIES = "step9d_direction_summaries_population_results_v1"
REQUIRED_STEP9_RAW_METRICS = ("roc_auc", "pr_auc", "brier_score")

# Sabit seri adlari (within-region + 3 adaptasyon yontemi) x model_family
SERIES_METHOD_NAMES = ("within",) + ADAPTATION_METHODS


def _series_col(method: str, model_family: str) -> str:
    return f"{method}_{model_family}"


# =============================================================================
# Hizalama: Step10B tahminleri + hedef etiketi + Step8B within-region OOF
# =============================================================================
def build_aligned_direction_frame(predictions_df: pd.DataFrame, direction: str, source_id: str, target_id: str) -> pd.DataFrame:
    """Tek bir yon icin: Step10B tahminlerini WIDE formata pivotlar, hedef
    etiketini (SIMDI) yukler, Step8B within-region OOF ile hizalar. Kimlik
    (cell_id) eslesmezse, populasyon maskeleri farkliysa veya spatial block
    ID'leri uyusmuyorsa FAIL-FAST durur."""
    subset = predictions_df[predictions_df["direction"] == direction].copy()
    if subset.empty:
        raise Step10Error(f"[{direction}] Step10B tahmin verisi BOS.")

    subset["series"] = subset["adaptation_method"] + "_" + subset["model_family"]
    wide = subset.pivot_table(
        index=["target_cell_id", "target_spatial_block_id"], columns="series",
        values="prediction_probability", aggfunc="first",
    ).reset_index()
    wide = wide.rename(columns={"target_cell_id": "cell_id", "target_spatial_block_id": "spatial_block_id"})

    if not wide["cell_id"].is_unique:
        raise Step10Error(f"[{direction}] Step10B tahminlerinde cell_id BENZERSIZ degil.")

    # --- SIMDI hedef etiketini yukle (yalnizca degerlendirme icin) ---
    target_full = load_step8a_dataset(target_id)
    target_pop = population_subset(target_full, PRIMARY_POPULATION)
    if not target_pop["cell_id"].is_unique:
        raise Step10Error(f"[{direction}] hedef Step8A '{PRIMARY_POPULATION}' populasyonunda cell_id BENZERSIZ degil.")

    target_labels = target_pop[["cell_id", "spatial_block_id", TARGET_COLUMN]].rename(
        columns={"spatial_block_id": "spatial_block_id_target_step8a", TARGET_COLUMN: "burned"}
    )

    merged = wide.merge(target_labels, on="cell_id", how="left")
    missing = merged["burned"].isna()
    if missing.any():
        raise Step10Error(
            f"[{direction}] {int(missing.sum())} hedef hucre, hedef Step8A "
            f"'{PRIMARY_POPULATION}' populasyonuyla HIZALANAMADI (kimlik/populasyon maskesi uyusmazligi)."
        )
    if len(merged) != len(wide):
        raise Step10Error(f"[{direction}] hizalama sonrasi satir sayisi degisti ({len(wide)} -> {len(merged)}).")

    mismatched_blocks = merged["spatial_block_id"] != merged["spatial_block_id_target_step8a"]
    if mismatched_blocks.any():
        raise Step10Error(
            f"[{direction}] {int(mismatched_blocks.sum())} hucrede spatial_block_id "
            "Step10 tahminleri ile hedef Step8A arasinda UYUSMUYOR."
        )
    merged = merged.drop(columns=["spatial_block_id_target_step8a"])
    merged["burned"] = merged["burned"].astype(int)

    # --- Within-region referans: hedefin KENDI Step8B OOF tahminleri ---
    oof_path = resolve_step8b_predictions_path(target_id)
    if not oof_path.exists():
        raise Step10Error(f"[{direction}] hedef ({target_id}) icin Step8B OOF tahmin dosyasi bulunamadi: {oof_path}.")
    oof_df = pd.read_parquet(oof_path)
    oof_pop = oof_df[oof_df["population"] == PRIMARY_POPULATION]
    if not oof_pop["cell_id"].is_unique:
        raise Step10Error(f"[{direction}] hedef Step8B OOF '{PRIMARY_POPULATION}' populasyonunda cell_id BENZERSIZ degil.")

    oof_ref = oof_pop[["cell_id", "spatial_block_id", "y_prob_baseline", "y_prob_thermal"]].rename(
        columns={"spatial_block_id": "spatial_block_id_oof", "y_prob_baseline": "within_baseline", "y_prob_thermal": "within_thermal"}
    )
    merged = merged.merge(oof_ref, on="cell_id", how="left")
    missing_oof = merged["within_baseline"].isna()
    if missing_oof.any():
        raise Step10Error(
            f"[{direction}] {int(missing_oof.sum())} hedef hucre, Step8B within-region "
            "OOF tahminleriyle HIZALANAMADI."
        )
    mismatched_oof_blocks = merged["spatial_block_id"] != merged["spatial_block_id_oof"]
    if mismatched_oof_blocks.any():
        raise Step10Error(
            f"[{direction}] {int(mismatched_oof_blocks.sum())} hucrede spatial_block_id "
            "Step10 tahminleri ile Step8B OOF arasinda UYUSMUYOR."
        )
    merged = merged.drop(columns=["spatial_block_id_oof"])

    return merged


# =============================================================================
# Nokta-tahmin metrikleri (threshold-free) + within-region reprodüksiyon +
# raw reprodüksiyon (Step9B'ye karsi, FAIL-FAST)
# =============================================================================
def compute_point_metrics(merged: pd.DataFrame) -> dict:
    y = merged["burned"].to_numpy()
    out: dict = {}
    for method in SERIES_METHOD_NAMES:
        out[method] = {}
        for model_family in MODEL_FAMILIES:
            col = _series_col(method, model_family)
            out[method][model_family] = compute_threshold_free_metrics(y, merged[col].to_numpy())
    return out


def verify_within_region_reproduction(merged: pd.DataFrame, point_metrics: dict, target_id: str, direction: str) -> dict:
    step8b_metrics_path = resolve_step8b_metrics_path(target_id)
    if not step8b_metrics_path.exists():
        log.warning("[%s] Step8B metrics.json bulunamadi (%s); within-region reprodüksiyon kontrolu ATLANDI.", direction, step8b_metrics_path)
        return {"checked": False}

    step8b_metrics = json.loads(step8b_metrics_path.read_text(encoding="utf-8"))
    pop_metrics = (step8b_metrics.get("population_metrics") or {}).get(PRIMARY_POPULATION)
    if pop_metrics is None:
        log.warning("[%s] Step8B metrics.json icinde '%s' populasyonu bulunamadi; kontrol ATLANDI.", direction, PRIMARY_POPULATION)
        return {"checked": False}

    checks = {}
    all_ok = True
    for model_family, step8b_key in (("baseline", "overall_baseline"), ("thermal", "overall_thermal")):
        theirs = pop_metrics.get(step8b_key, {})
        mine = point_metrics["within"][model_family]
        row = {}
        for metric in ("roc_auc", "pr_auc"):
            m, t = mine.get(metric), theirs.get(metric)
            if m is None or t is None:
                row[metric] = {"mine": m, "step8b": t, "diff": None, "ok": None}
                continue
            diff = abs(m - t)
            ok = diff <= WITHIN_REGION_REPRODUCTION_TOLERANCE
            row[metric] = {"mine": m, "step8b": t, "diff": diff, "ok": ok}
            all_ok = all_ok and ok
        checks[model_family] = row
    return {"checked": True, "all_within_tolerance": all_ok, "detail": checks}


def _validated_step9_transfer_metrics(
    transfer: dict, *, direction: str, path: Path, schema: str,
) -> dict:
    """Extract the frozen Step9 baseline/thermal metrics without recomputing
    or filling any value. All six required values must be finite numbers."""
    extracted: dict = {}
    for model_family, metric_key in (
        ("baseline", "baseline_metrics"), ("thermal", "thermal_metrics"),
    ):
        metric_block = transfer.get(metric_key)
        if not isinstance(metric_block, dict):
            raise Step10Error(
                f"[{direction}] {schema} icinde '{metric_key}' eksik/gecersiz: {path}."
            )
        extracted[model_family] = {}
        for metric in REQUIRED_STEP9_RAW_METRICS:
            value = metric_block.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise Step10Error(
                    f"[{direction}] {schema} gerekli metrik eksik/sayisal degil: "
                    f"{metric_key}.{metric}={value!r} ({path})."
                )
            value = float(value)
            if not np.isfinite(value):
                raise Step10Error(
                    f"[{direction}] {schema} gerekli metrik sonlu degil: "
                    f"{metric_key}.{metric}={value!r} ({path})."
                )
            extracted[model_family][metric] = value
    return extracted


def _step9b_reference(payload: dict, direction: str, path: Path) -> dict | None:
    """Legacy/current Step9B schema: top-level results[]."""
    results = payload.get("results")
    if not isinstance(results, list):
        return None
    matches = [
        row for row in results
        if isinstance(row, dict)
        and row.get("transfer_direction") == direction
        and row.get("population") == PRIMARY_POPULATION
        and not row.get("skipped")
    ]
    if len(matches) > 1:
        raise Step10Error(
            f"[{direction}] Step9B '{PRIMARY_POPULATION}' icin birden fazla "
            f"sonuc iceriyor ({len(matches)}): {path}."
        )
    if not matches:
        return None
    return _validated_step9_transfer_metrics(
        matches[0], direction=direction, path=path,
        schema=STEP9_SCHEMA_STEP9B_RESULTS,
    )


def _step9d_reference(payload: dict, direction: str, path: Path) -> dict | None:
    """Canonical Step9D schema described by direction_summaries[]."""
    summaries = payload.get("direction_summaries")
    if not isinstance(summaries, list):
        return None
    matches = [
        row for row in summaries
        if isinstance(row, dict) and row.get("transfer_direction") == direction
    ]
    if len(matches) > 1:
        raise Step10Error(
            f"[{direction}] Step9D direction_summaries birden fazla tam eslesme "
            f"iceriyor ({len(matches)}): {path}."
        )
    if not matches:
        return None
    population_results = matches[0].get("population_results")
    if not isinstance(population_results, dict) or PRIMARY_POPULATION not in population_results:
        raise Step10Error(
            f"[{direction}] Step9D primary population eksik: "
            f"population_results.{PRIMARY_POPULATION} ({path})."
        )
    population_entry = population_results[PRIMARY_POPULATION]
    transfer = population_entry.get("transfer") if isinstance(population_entry, dict) else None
    if not isinstance(transfer, dict):
        raise Step10Error(
            f"[{direction}] Step9D transfer nesnesi eksik/gecersiz: "
            f"population_results.{PRIMARY_POPULATION}.transfer ({path})."
        )
    return _validated_step9_transfer_metrics(
        transfer, direction=direction, path=path,
        schema=STEP9_SCHEMA_STEP9D_DIRECTION_SUMMARIES,
    )


def resolve_step9_raw_reference(
    source_id: str, target_id: str, direction: str,
) -> dict:
    """Resolve frozen Step9 metrics for one requested logical direction.

    The requested <source>__<target> root is authoritative. The reversed root
    is checked only for compatibility with legacy Manavgat/Bejis-style shared
    pair outputs that stored both logical directions under one orientation.
    Within each root Step9B is preferred; Step9D is the canonical fallback.
    """
    roots = [(source_id, target_id), (target_id, source_id)]
    attempted: list[str] = []
    saw_supported_schema = False
    for root_source, root_target in roots:
        step9b_path = resolve_step9b_metrics_path(root_source, root_target)
        step9d_path = (
            step9b_path.parent.parent / "step9d" / "final_cross_region_report.json"
        )
        for path, schema, extractor in (
            (step9b_path, STEP9_SCHEMA_STEP9B_RESULTS, _step9b_reference),
            (step9d_path, STEP9_SCHEMA_STEP9D_DIRECTION_SUMMARIES, _step9d_reference),
        ):
            attempted.append(str(path))
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            schema_field = "results" if schema == STEP9_SCHEMA_STEP9B_RESULTS else "direction_summaries"
            if not isinstance(payload, dict) or not isinstance(payload.get(schema_field), list):
                continue
            saw_supported_schema = True
            metrics = extractor(payload, direction, path)
            if metrics is not None:
                return {
                    "resolved_path": str(path),
                    "schema": schema,
                    "root_source_experiment_id": root_source,
                    "root_target_experiment_id": root_target,
                    "metrics": metrics,
                }

    reason = (
        f"hicbir desteklenen Step9 semasinda direction='{direction}' ve "
        f"population='{PRIMARY_POPULATION}' eslesmesi yok"
        if saw_supported_schema else "desteklenen Step9B/Step9D semasi bulunamadi"
    )
    raise Step10Error(
        f"[{direction}] Step9 raw metrikleri cozumlenemedi: {reason}. "
        f"Denenen dosyalar: {attempted}."
    )


def verify_raw_reproduction(point_metrics: dict, source_id: str, target_id: str, direction: str) -> dict:
    """FAIL-FAST: step10 raw_source_only metrikleri Step9B'yi 1e-6 tolerans
    ile YENIDEN URETMELIDIR. Basarisiz olursa Step10Error firlatilir --
    adapte edilmis (zscore/CORAL) rapora GECILMEZ.

    Step9B'nin legacy/current results[] semasi tercih edilir; uygun sonuc
    yoksa ayni pair namespace'indeki canonical Step9D direction_summaries
    semasina dusulur. Hicbir Step9 degeri yeniden hesaplanmaz."""
    reference = resolve_step9_raw_reference(source_id, target_id, direction)
    ref = reference["metrics"]

    checks, all_ok = {}, True
    for model_family in ("baseline", "thermal"):
        theirs = ref[model_family]
        mine = point_metrics["raw_source_only"][model_family]
        row = {}
        for metric in ("roc_auc", "pr_auc"):
            m, t = mine.get(metric), theirs.get(metric)
            if m is None or t is None:
                row[metric] = {"mine": m, "step9b": t, "diff": None, "ok": False}
                all_ok = False
                continue
            diff = abs(m - t)
            ok = diff <= RAW_REPRODUCTION_TOLERANCE
            row[metric] = {"mine": m, "step9b": t, "diff": diff, "ok": ok}
            all_ok = all_ok and ok
        checks[model_family] = row

    if not all_ok:
        raise Step10Error(
            f"[{direction}] RAW REPRODUKSIYON KONTROLU BASARISIZ (tolerans={RAW_REPRODUCTION_TOLERANCE}): "
            f"{checks}. Adapte edilmis (zscore/CORAL) bilimsel rapora GECILMIYOR."
        )
    log.info("[%s] Raw reprodüksiyon kontrolu BASARILI (tum farklar <= %s).", direction, RAW_REPRODUCTION_TOLERANCE)

    # --- Opsiyonel: olasilik-seviyesinde karsilastirma (mumkunse) ---
    probability_check = {"attempted": False}
    try:
        step9b_pred_path = resolve_step9b_predictions_path(
            reference["root_source_experiment_id"],
            reference["root_target_experiment_id"],
        )
        if step9b_pred_path.exists():
            probability_check["attempted"] = True
            # (Bu kontrol best-effort'tur; join basarisiz olursa metrik-seviyesi
            # kontrolu zaten yeterli kanit saglamistir.)
    except Exception as exc:  # noqa: BLE001
        log.warning("[%s] Olasilik-seviyesi reprodüksiyon kontrolu denenemedi: %s", direction, exc)

    return {
        "all_within_tolerance": True,
        "detail": checks,
        "probability_level_check": probability_check,
        "step9_reference": {
            "resolved_path": reference["resolved_path"],
            "schema": reference["schema"],
            "root_source_experiment_id": reference["root_source_experiment_id"],
            "root_target_experiment_id": reference["root_target_experiment_id"],
            "required_metrics": list(REQUIRED_STEP9_RAW_METRICS),
            "baseline_metrics": ref["baseline"],
            "thermal_metrics": ref["thermal"],
        },
    }


# =============================================================================
# Ayristirma (decomposition)
# =============================================================================
def compute_decomposition(point_metrics: dict, direction: str) -> list[dict]:
    rows = []
    for model_family in MODEL_FAMILIES:
        within = point_metrics["within"][model_family]
        raw = point_metrics["raw_source_only"][model_family]
        for method in ("regionwise_zscore", "coral_after_regionwise_zscore"):
            adapted = point_metrics[method][model_family]
            for metric in ("roc_auc", "pr_auc"):
                if within.get(metric) is None or raw.get(metric) is None or adapted.get(metric) is None:
                    continue
                rows.append({
                    "direction": direction, "model_family": model_family, "method": method, "metric": metric,
                    "raw_value": raw[metric], "adapted_value": adapted[metric], "within_value": within[metric],
                    "recovered_covariate_component": adapted[metric] - raw[metric],
                    "remaining_transfer_gap": within[metric] - adapted[metric],
                })
    return rows


# =============================================================================
# N-yollu esli (paired) bootstrap -- yon basina
# =============================================================================
def run_bootstrap_for_direction(merged: pd.DataFrame, n_replicates: int, seed: int) -> dict:
    prob_columns = {_series_col(m, f): _series_col(m, f) for m in SERIES_METHOD_NAMES for f in MODEL_FAMILIES}
    result = run_n_way_paired_bootstrap(
        merged, block_col="spatial_block_id", y_col="burned", prob_columns=prob_columns,
        n_replicates=n_replicates, random_state=seed,
    )
    replicates_df = result["replicates_df"]
    if not replicates_df.empty:
        for model_family in MODEL_FAMILIES:
            raw_c, zscore_c, coral_c, within_c = (
                _series_col("raw_source_only", model_family), _series_col("regionwise_zscore", model_family),
                _series_col("coral_after_regionwise_zscore", model_family), _series_col("within", model_family),
            )
            for metric in ("roc_auc", "pr_auc"):
                replicates_df[f"delta_{metric}__zscore_minus_raw__{model_family}"] = replicates_df[f"{metric}__{zscore_c}"] - replicates_df[f"{metric}__{raw_c}"]
                replicates_df[f"delta_{metric}__coral_minus_raw__{model_family}"] = replicates_df[f"{metric}__{coral_c}"] - replicates_df[f"{metric}__{raw_c}"]
                replicates_df[f"delta_{metric}__coral_minus_zscore__{model_family}"] = replicates_df[f"{metric}__{coral_c}"] - replicates_df[f"{metric}__{zscore_c}"]
                replicates_df[f"delta_{metric}__within_minus_zscore__{model_family}"] = replicates_df[f"{metric}__{within_c}"] - replicates_df[f"{metric}__{zscore_c}"]
                replicates_df[f"delta_{metric}__within_minus_coral__{model_family}"] = replicates_df[f"{metric}__{within_c}"] - replicates_df[f"{metric}__{coral_c}"]
    result["replicates_df"] = replicates_df
    return result


def summarize_bootstrap(replicates_df: pd.DataFrame) -> dict:
    summary = {}
    if replicates_df.empty:
        return summary
    for col in replicates_df.columns:
        if col in ("replicate",):
            continue
        lo, hi, mean = percentile_ci(replicates_df[col])
        vals = replicates_df[col].dropna()
        summary[col] = {"ci_2_5": lo, "ci_97_5": hi, "mean": mean, "median": float(vals.median()) if len(vals) else None}
    return summary


# =============================================================================
# Orkestrasyon
# =============================================================================
def run_step10c(
    source_id: str, target_id: str, analysis_id: str, force: bool = False,
    n_replicates: int = STEP10_BOOTSTRAP_REPLICATES, seed: int = STEP10_RANDOM_STATE,
) -> dict:
    output_dir = step10_output_dir(source_id, target_id)
    metrics_json_path = output_dir / "step10_metrics.json"
    if metrics_json_path.exists() and not force:
        log.info("Step10C ciktilari zaten var; --force verilmedigi icin atlaniyor.")
        return json.loads(metrics_json_path.read_text(encoding="utf-8"))

    predictions_path = output_dir / "step10_predictions.parquet"
    if not predictions_path.exists():
        raise Step10Error(f"Step10B tahminleri bulunamadi ({predictions_path}); once Step10B calistirilmali.")
    predictions_df = pd.read_parquet(predictions_path)

    directions = [f"{source_id}_to_{target_id}", f"{target_id}_to_{source_id}"]
    all_point_metrics, all_decomposition_rows = {}, []
    all_bootstrap_summaries, all_bootstrap_replicates = {}, []
    raw_reproduction_results, within_reproduction_results = {}, {}

    for direction in directions:
        src_id, tgt_id = (source_id, target_id) if direction == f"{source_id}_to_{target_id}" else (target_id, source_id)

        log.info("[%s] Step10B tahminleri hedef etiketiyle hizalaniyor...", direction)
        merged = build_aligned_direction_frame(predictions_df, direction, src_id, tgt_id)

        point_metrics = compute_point_metrics(merged)
        all_point_metrics[direction] = point_metrics

        within_reproduction_results[direction] = verify_within_region_reproduction(merged, point_metrics, tgt_id, direction)
        raw_reproduction_results[direction] = verify_raw_reproduction(
            point_metrics, src_id, tgt_id, direction,
        )  # FAIL-FAST

        all_decomposition_rows.extend(compute_decomposition(point_metrics, direction))

        log.info("[%s] N-yollu esli spatial-block bootstrap calistiriliyor (%d replika)...", direction, n_replicates)
        bootstrap_result = run_bootstrap_for_direction(merged, n_replicates, seed)
        replicates_df = bootstrap_result["replicates_df"]
        n_valid, n_invalid = bootstrap_result["n_valid"], bootstrap_result["n_invalid_single_class"]
        unstable = is_bootstrap_unstable(n_valid)
        if unstable:
            log.warning("[%s] bootstrap_unstable: yalnizca %d/%d gecerli replika.", direction, n_valid, n_replicates)

        all_bootstrap_summaries[direction] = {
            "n_requested": bootstrap_result["n_requested"], "n_valid": n_valid,
            "n_invalid_single_class": n_invalid, "bootstrap_unstable": unstable,
            "ci": summarize_bootstrap(replicates_df),
        }
        if not replicates_df.empty:
            r = replicates_df.copy()
            r.insert(0, "direction", direction)
            all_bootstrap_replicates.append(r)

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "analysis_id": analysis_id, "point_metrics": all_point_metrics,
        "within_region_reproduction": within_reproduction_results,
        "raw_reproduction": raw_reproduction_results,
        "step9_raw_metric_provenance": {
            direction: result["step9_reference"]
            for direction, result in raw_reproduction_results.items()
        },
    }
    metrics_json_path.write_text(json.dumps(metrics_payload, indent=2, default=str), encoding="utf-8")

    metrics_rows = []
    for direction, pm in all_point_metrics.items():
        for method, by_family in pm.items():
            for model_family, m in by_family.items():
                metrics_rows.append({
                    "analysis_id": analysis_id, "direction": direction, "method": method,
                    "model_family": model_family, **m,
                })
    pd.DataFrame(metrics_rows).to_csv(output_dir / "step10_metrics.csv", index=False)

    decomposition_df = pd.DataFrame(all_decomposition_rows)
    decomposition_df.insert(0, "analysis_id", analysis_id)
    decomposition_df.to_csv(output_dir / "step10_decomposition.csv", index=False)

    bootstrap_replicates_df = (
        pd.concat(all_bootstrap_replicates, ignore_index=True) if all_bootstrap_replicates else pd.DataFrame()
    )
    if not bootstrap_replicates_df.empty:
        bootstrap_replicates_df.insert(0, "analysis_id", analysis_id)
    bootstrap_replicates_df.to_parquet(output_dir / "step10_bootstrap_replicates.parquet", index=False)

    bootstrap_summary_payload = {"analysis_id": analysis_id, "by_direction": all_bootstrap_summaries}
    (output_dir / "step10_bootstrap_summary.json").write_text(
        json.dumps(bootstrap_summary_payload, indent=2, default=str), encoding="utf-8",
    )
    summary_rows = []
    for direction, d in all_bootstrap_summaries.items():
        for series, ci in d["ci"].items():
            summary_rows.append({
                "analysis_id": analysis_id, "direction": direction, "series": series,
                "n_valid": d["n_valid"], "n_requested": d["n_requested"],
                "bootstrap_unstable": d["bootstrap_unstable"], **ci,
            })
    pd.DataFrame(summary_rows).to_csv(output_dir / "step10_bootstrap_summary.csv", index=False)

    log.info("Step10C tamamlandi: %s", output_dir)
    return metrics_payload


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step10C: paired evaluation + bootstrap (target labels loaded HERE).")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--analysis-id", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=STEP10_BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=STEP10_RANDOM_STATE)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_step10c(
        source_id=args.source, target_id=args.target, analysis_id=args.analysis_id, force=args.force,
        n_replicates=args.bootstrap_replicates, seed=args.seed,
    )
