"""
run_step8_modeling.py

Step0H: deney-farkinda (experiment-aware) Step8 burned-area modelleme
calistiricisi. Manavgat 2021 icin, Kozan'in ZATEN dogrulanmis Step8A-D
mantigini (label-honest ~500 m MCD64A1-cell aggregation, spatial-block CV,
spatial-block bootstrap, thermal feature ablation) DEGISTIRMEDEN reuse eder;
yalnizca girdi/cikti yollarini deney-farkinda hale getirir.

ONEMLI -- Step8B/8C/8D eslesmesi (bu script'in prompt'taki "Step8B/8C/8D"
tanimlarini nasil karsiladigi):
    Mevcut src/step8b_train_baseline_vs_thermal_model.py TEK bir calistirmada
    hem "baseline" (ndvi/elevation/slope/landcover) HEM "baseline+thermal"
    (+ lst_anomaly/current_tvdi/tvdi_difference/downscaled_lst/fused_lst)
    modellerini egitip delta_auc/delta_pr_auc raporlar -- yani prompt'un
    ayri "Step8B baseline" + "Step8C fused-comparison" olarak tanimladigi
    ISI TEK ADIMDA yapar. src/step8c_spatial_block_bootstrap_uncertainty.py
    bu delta metrikler icin spatial-block bootstrap guven araligi
    hesaplar (prompt'un "Step8D" bootstrap karsilastirmasiyla ESLESIR).
    src/step8d_thermal_feature_ablation.py EK olarak hangi termal grubun
    (ozellikle fused_lst) katkiyi ne kadar surdugunu ayristirir.
    Bu script bu ADLANDIRMA FARKINI DEGISTIRMEZ (Kozan'in kod mimarisini
    BOZMADAN reuse etmek icin) -- ama TUM bilimsel ciktilar (baseline vs.
    fused-LST karsilastirmasi, bootstrap-destekli delta, ablation) uretilir.
    Final rapor (bu script'in kendi write_final_report()'u, Kozan'in
    Step8E'sini REUSE ETMEDEN) prompt'un istedigi TUM icerigi
    (dataset/gate/Step7/model/bootstrap/limitations) tek dosyada toplar.

Bu adimlarin urettigi model SAF bir burned-area ASSOCIATION modelidir --
gercek-zamanli yangin tespiti veya operasyonel yangin tahmini DEGILDIR.
Label MCD64A1'in ~500 m native gridindedir; 30 m piksel HICBIR ZAMAN label
olarak kullanilmaz.

CLI:
    python scripts/run_step8_modeling.py --experiment manavgat_2021 --dry-run
    python scripts/run_step8_modeling.py --experiment manavgat_2021 --force
    python scripts/run_step8_modeling.py --experiment manavgat_2021 --force --allow-no-step7
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

from core.experiment_context import build_experiment_context, log_context_summary
from core.io_utils import setup_logger

log, log_file = setup_logger("run_step8_modeling")

BASE_DIR = _PROJECT_ROOT
STEP_ORDER = ["step8a", "step8b", "step8c", "step8d"]

# Leakage/forbidden-feature guard: hicbir predictor adi/yolu bu alt-dizgeleri
# ICERMEMELIDIR (label-derived / post-fire / FIRMS). Step8A'nin
# CONTINUOUS_PREDICTOR_CANDIDATES kaydi zaten bunlari hic icermez; bu, o
# tasarim degismeden kalirsa diye ek bir CALISMA-ZAMANI guvenlik agidir.
_FORBIDDEN_FEATURE_SUBSTRINGS = [
    "burndate", "burned", "mcd64a1", "firms", "post_fire", "postfire",
]

_LEGACY_SHARED_DIRS = [
    (BASE_DIR / "outputs" / "step8a").resolve(),
    (BASE_DIR / "outputs" / "step8b").resolve(),
    (BASE_DIR / "outputs" / "step8c").resolve(),
    (BASE_DIR / "outputs" / "step8d").resolve(),
    (BASE_DIR / "outputs" / "step8e").resolve(),
    (BASE_DIR / "outputs" / "step5").resolve(),
    (BASE_DIR / "outputs" / "step5c").resolve(),
    (BASE_DIR / "outputs" / "step7e").resolve(),
    (BASE_DIR / "outputs" / "validation" / "labels").resolve(),
]


class Step8RunnerError(SystemExit):
    """Fail-fast error for this runner (diğer step'lerle aynı konvansiyon)."""


def _assert_paths_are_safely_namespaced(ctx: dict) -> None:
    experiment_id = ctx["experiment_id"]
    experiments_root = (BASE_DIR / "outputs" / "experiments" / experiment_id).resolve()
    for key in ("step8a_output_dir", "step8b_output_dir", "step8c_output_dir",
                "step8d_output_dir", "step8e_output_dir"):
        resolved = Path(ctx[key]).resolve()
        for legacy_dir in _LEGACY_SHARED_DIRS:
            if resolved == legacy_dir or legacy_dir in resolved.parents:
                raise Step8RunnerError(
                    f"GÜVENLİK İHLALİ: '{experiment_id}' deneyi için '{key}' "
                    f"yolu ({resolved}) legacy paylaşılan dizine ({legacy_dir}) "
                    "düşüyor. İşlem DURDURULDU."
                )
        if resolved != experiments_root and experiments_root not in resolved.parents:
            raise Step8RunnerError(
                f"GÜVENLİK İHLALİ: '{experiment_id}' deneyi için '{key}' yolu "
                f"({resolved}) outputs/experiments/{experiment_id}/ dışında. "
                "İşlem DURDURULDU."
            )


def _assert_no_leakage_features() -> None:
    """CONTINUOUS_PREDICTOR_CANDIDATES icinde yasak alt-dizgiler var mi kontrol eder."""
    import src.step8a_prepare_500m_modeling_dataset as step8a
    for name, info in step8a.CONTINUOUS_PREDICTOR_CANDIDATES.items():
        lname = name.lower()
        for bad in _FORBIDDEN_FEATURE_SUBSTRINGS:
            if bad in lname:
                raise Step8RunnerError(
                    f"GÜVENLİK İHLALİ: predictor adı '{name}' yasaklı alt-dizge "
                    f"'{bad}' içeriyor (label-derived/post-fire/FIRMS). İşlem DURDURULDU."
                )
        for cand in info["candidates"]:
            lcand = cand.lower()
            for bad in _FORBIDDEN_FEATURE_SUBSTRINGS:
                if bad in lcand:
                    raise Step8RunnerError(
                        f"GÜVENLİK İHLALİ: predictor '{name}' yolu '{cand}' yasaklı "
                        f"alt-dizge '{bad}' içeriyor. İşlem DURDURULDU."
                    )


def _label_paths(ctx: dict) -> dict:
    labels_dir = ctx["gate_labels_dir"]
    return {
        "mcd64a1_raw": labels_dir / "mcd64a1_raw.tif",
        "mcd64a1_burned": labels_dir / "mcd64a1_burned.tif",
    }


def _predictor_paths(ctx: dict) -> dict:
    return {
        "current_period_median_celsius": ctx["step5_output_dir"] / "current_period_median_celsius.tif",
        "anomaly_zscore": ctx["step5_output_dir"] / "anomaly_zscore.tif",
        "baseline_lst_mean_celsius": ctx["step5_output_dir"] / "baseline_lst_mean_celsius.tif",
        "baseline_lst_std_celsius": ctx["step5_output_dir"] / "baseline_lst_std_celsius.tif",
        "current_tvdi": ctx["step5c_output_dir"] / "current_tvdi.tif",
        "tvdi_difference": ctx["step5c_output_dir"] / "tvdi_difference.tif",
        "fused_lst_celsius": ctx["step7e_output_dir"] / "fused_lst_celsius.tif",
        "elevation": ctx["dem_input_dir"] / "elevation.tif",
        "slope": ctx["dem_input_dir"] / "slope.tif",
        "landcover_aligned": ctx["landcover_aligned_path"],
    }


def _log_planned_paths(ctx: dict) -> None:
    log.info("  step8a_output_dir: %s", ctx["step8a_output_dir"])
    log.info("  step8b_output_dir: %s", ctx["step8b_output_dir"])
    log.info("  step8c_output_dir: %s", ctx["step8c_output_dir"])
    log.info("  step8d_output_dir: %s", ctx["step8d_output_dir"])
    log.info("  step8e_output_dir: %s", ctx["step8e_output_dir"])


def _dry_run_input_check(ctx: dict, allow_no_step7: bool) -> dict:
    log.info("Beklenen etiket dosyaları (MCD64A1, ~500 m hücre-seviyesi):")
    labels = _label_paths(ctx)
    label_status = {}
    for name, p in labels.items():
        exists = p.exists()
        label_status[name] = exists
        log.info("  %s %s (%s)", "[VAR]" if exists else "[EKSİK]", name, p)
    log.info(
        "  -> Etiketler MCD64A1 hücre-seviyesindedir (gate_level="
        "500m_reconstructed_mcd64a1_cell); 30 m piksel ASLA label olarak KULLANILMAZ."
    )

    log.info("Beklenen predictor rasterları:")
    predictors = _predictor_paths(ctx)
    predictor_status = {}
    for name, p in predictors.items():
        exists = p is not None and Path(p).exists()
        predictor_status[name] = exists
        log.info("  %s %s (%s)", "[VAR]" if exists else "[EKSİK]", name, p)

    fused_available = predictor_status.get("fused_lst_celsius", False)
    if fused_available:
        log.info("  -> Step7 fused LST (fused_lst_celsius.tif) mevcut.")
    elif allow_no_step7:
        log.info(
            "  -> Step7 fused LST EKSİK, ama --allow-no-step7 verildi: gerçek "
            "çalıştırma fused_lst'siz devam edecek (thermal feature seti azalır)."
        )
    else:
        log.info(
            "  -> Step7 fused LST EKSİK ve --allow-no-step7 VERİLMEDİ: gerçek "
            "çalıştırma (--force) FAIL-FAST ile duracak."
        )

    log.info("Planlanan çıktı dizinleri:")
    _log_planned_paths(ctx)

    log.info(
        "  -> Kozan'ın legacy Step8 çıktılarına (outputs/step8a..e) ASLA "
        "yazılmaz/okunmaz (bkz. güvenlik kontrolü)."
    )
    return {"labels": label_status, "predictors": predictor_status, "fused_available": fused_available}


def _enrich_metadata_with_experiment_context(metadata_path: Path, ctx: dict) -> None:
    if ctx is None or not metadata_path.exists():
        return
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(data, dict):
        return
    changed = False
    for key, value in [
        ("experiment_id", ctx["experiment_id"]),
        ("region_key", ctx["region_key"]),
        ("role", ctx["role"]),
        ("predictor_window", [ctx["predictor_start_date"], ctx["predictor_end_date"]]),
        ("label_window", [ctx["label_start_date"], ctx["label_end_date"]]),
        ("baseline_years", ctx["baseline_years"]),
    ]:
        if key not in data:
            data[key] = value
            changed = True
    if changed:
        metadata_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_final_report(ctx: dict, results: dict) -> dict:
    """
    Manavgat icin kompakt Step8E-esdegeri final rapor. Kozan'in
    src/step8e_final_report.py'sini (coklu-populasyon burnable-mask semasi
    Kozan'a ozel oldugu icin) REUSE ETMEZ -- prompt'un istedigi TUM
    icerigi (deney bilgisi, gate sonucu, Step7 kapsama kazanimi, Step8
    hucre sayilari, baseline-vs-fused metrikleri, bootstrap belirsizligi,
    sinirlamalar) dogrudan Step8A-D'nin (reuse edilen, degistirilmemis)
    JSON ciktilarindan okuyup tek bir rapora birlestirir.
    """
    out_dir = ctx["step8e_output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    gate_path = ctx["gate_labels_dir"] / "burned_landcover_gate.json"
    gate = _load_json_if_exists(gate_path) or {}

    step7e_stats = _load_json_if_exists(ctx["step7e_output_dir"] / "fused_lst_stats.json") or {}
    step8a_stats = _load_json_if_exists(ctx["step8a_output_dir"] / "step8a_dataset_stats.json") or {}
    step8b_metrics = _load_json_if_exists(
        ctx["step8b_output_dir"] / "step8b_model_comparison_metrics.json"
    ) or {}
    step8c_metrics = _load_json_if_exists(
        ctx["step8c_output_dir"] / "step8c_bootstrap_metrics.json"
    ) or {}
    step8d_metrics = _load_json_if_exists(
        ctx["step8d_output_dir"] / "step8d_ablation_metrics.json"
    ) or {}

    report = {
        "experiment_id": ctx["experiment_id"],
        "region_key": ctx["region_key"],
        "role": ctx["role"],
        "predictor_window": [ctx["predictor_start_date"], ctx["predictor_end_date"]],
        "label_window": [ctx["label_start_date"], ctx["label_end_date"]],
        "baseline_years": ctx["baseline_years"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gate_result": {
            "decision": gate.get("decision"),
            "reason": gate.get("reason"),
            "burned_count": gate.get("burned_count"),
            "burned_natural_vegetation_fraction": gate.get("burned_natural_vegetation_fraction"),
            "gate_level": gate.get("gate_level"),
        },
        "step7_fused_lst_coverage": {
            "observed_coverage_pct": step7e_stats.get("observed_coverage_pct"),
            "fused_coverage_pct": step7e_stats.get("fused_coverage_pct"),
            "gapfilled_pixels": step7e_stats.get("gapfilled_pixels"),
            "coverage_gain_pct": (
                step7e_stats.get("fused_coverage_pct", 0) - step7e_stats.get("observed_coverage_pct", 0)
                if step7e_stats else None
            ),
        },
        "step8a_dataset": {
            "cell_level": "500m_reconstructed_mcd64a1_cell",
            "no_30m_label_claim": True,
            "total_500m_cells": step8a_stats.get("total_500m_cells"),
            "valid_modeling_cells": step8a_stats.get("valid_modeling_cells"),
            "burned_cell_count": step8a_stats.get("burned_cell_count"),
            "unburned_cell_count": step8a_stats.get("unburned_cell_count"),
            "burned_rate": step8a_stats.get("burned_rate"),
        },
        "step8b_baseline_vs_fused_model": step8b_metrics.get("population_metrics", step8b_metrics),
        "step8c_bootstrap_uncertainty": step8c_metrics,
        "step8d_thermal_ablation": step8d_metrics.get("ablation_results", step8d_metrics),
        "limitations": [
            "Labels are MCD64A1 ~500 m native-grid cells, not 30 m pixels; "
            "no 30 m burned-area prediction is made or implied.",
            "This is pre-fire/association modeling evaluated retrospectively "
            "on a single historical fire season, not real-time or operational "
            "fire detection.",
            "Step7 fused LST (fused_lst_celsius.tif) is a predictor-window "
            "summary/fusion product (single-season MODIS context + observed "
            "Landsat, gap-filled with downscaled LST), not a daily "
            "operational MODIS downscaling/gap-filling product.",
            "Burned labels (MCD64A1) are used only for this Step8 supervised "
            "evaluation; Step7's downscaling model never saw burned-area "
            "labels during training.",
            "Delta metrics (delta_auc, delta_pr_auc) are reported with "
            "spatial-block bootstrap percentile intervals, not classical "
            "p-values; wording is limited to 'measurable improvement' / "
            "'spatial-block-bootstrap-supported improvement', never "
            "'statistically significant'.",
        ],
        "claim_policy": {
            "no_30m_burned_area_prediction_claim": True,
            "no_operational_wildfire_prediction_claim": True,
            "no_fire_detection_claim": True,
        },
        "results_ran": results,
    }

    json_path = out_dir / "final_step8_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def fmt(x):
        return "n/a" if x is None else x

    lines = [
        f"# Step8 Final Report — {ctx['experiment_id']}",
        "",
        f"- Region: `{ctx['region_key']}` | Role: `{ctx['role']}`",
        f"- Predictor window: {ctx['predictor_start_date']} -> {ctx['predictor_end_date']}",
        f"- Label window: {ctx['label_start_date']} -> {ctx['label_end_date']}",
        f"- Baseline years: {ctx['baseline_years']}",
        "",
        "## Burned-landcover gate result",
        "",
        f"- Decision: `{fmt(gate.get('decision'))}`",
        f"- Burned cells: {fmt(gate.get('burned_count'))}",
        f"- Natural vegetation fraction: {fmt(gate.get('burned_natural_vegetation_fraction'))}",
        "",
        "## Step7 fused LST coverage",
        "",
        f"- Observed coverage: {fmt(step7e_stats.get('observed_coverage_pct'))}%",
        f"- Fused coverage: {fmt(step7e_stats.get('fused_coverage_pct'))}%",
        f"- Gap-filled pixels: {fmt(step7e_stats.get('gapfilled_pixels'))}",
        "",
        "## Step8A dataset (500 m reconstructed MCD64A1 cells)",
        "",
        f"- Total cells: {fmt(step8a_stats.get('total_500m_cells'))}",
        f"- Valid modeling cells: {fmt(step8a_stats.get('valid_modeling_cells'))}",
        f"- Burned: {fmt(step8a_stats.get('burned_cell_count'))} | "
        f"Unburned: {fmt(step8a_stats.get('unburned_cell_count'))}",
        f"- Burned rate: {fmt(step8a_stats.get('burned_rate'))}",
        "",
        "## Baseline vs. baseline+fused-thermal model (Step8B)",
        "",
        "See `step8b_model_comparison_metrics.json` for full per-population "
        "AUC/PR-AUC and delta metrics.",
        "",
        "## Spatial-block bootstrap uncertainty (Step8C)",
        "",
        "See `step8c_bootstrap_metrics.json` for percentile confidence "
        "intervals on delta_auc / delta_pr_auc. Bootstrap-percentile-interval "
        "language only — no classical significance claims.",
        "",
        "## Thermal feature ablation (Step8D)",
        "",
        "See `step8d_ablation_metrics.json` for which thermal feature group "
        "(including Step7 fused LST) contributes most to the measurable "
        "improvement.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")

    md_path = out_dir / "final_step8_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    log.info("Final rapor: %s, %s", json_path, md_path)
    return {"json_path": str(json_path), "md_path": str(md_path)}


def _run_kozan_step8(force: bool) -> dict:
    import src.step8a_prepare_500m_modeling_dataset as step8a
    import src.step8b_train_baseline_vs_thermal_model as step8b
    import src.step8c_spatial_block_bootstrap_uncertainty as step8c
    import src.step8d_thermal_feature_ablation as step8d
    import src.step8e_final_report as step8e

    results = {}
    log.info("[kozan_2023] STEP 8A")
    results["step8a"] = step8a.main(force=force)
    log.info("[kozan_2023] STEP 8B")
    results["step8b"] = step8b.main(force=force)
    log.info("[kozan_2023] STEP 8C")
    results["step8c"] = step8c.main(force=force)
    log.info("[kozan_2023] STEP 8D")
    results["step8d"] = step8d.main(force=force)
    log.info("[kozan_2023] STEP 8E")
    results["step8e"] = step8e.main(force=force)
    return results


def _run_experiment_step8(ctx: dict, force: bool) -> dict:
    import src.step8a_prepare_500m_modeling_dataset as step8a
    import src.step8b_train_baseline_vs_thermal_model as step8b
    import src.step8c_spatial_block_bootstrap_uncertainty as step8c
    import src.step8d_thermal_feature_ablation as step8d

    results = {}

    log.info("[%s] STEP 8A (500 m label-honest modeling dataset, namespaced)", ctx["experiment_id"])
    results["step8a"] = step8a.run_step8a(ctx=ctx, force=force)
    _enrich_metadata_with_experiment_context(
        ctx["step8a_output_dir"] / "step8a_dataset_stats.json", ctx,
    )

    log.info(
        "[%s] STEP 8B (baseline vs. baseline+fused-thermal, spatial-block CV, namespaced)",
        ctx["experiment_id"],
    )
    results["step8b"] = step8b.run_step8b(ctx=ctx, force=force)
    _enrich_metadata_with_experiment_context(
        ctx["step8b_output_dir"] / "step8b_model_comparison_metrics.json", ctx,
    )

    log.info("[%s] STEP 8C (spatial-block bootstrap uncertainty, namespaced)", ctx["experiment_id"])
    results["step8c"] = step8c.run_step8c(ctx=ctx, force=force)
    _enrich_metadata_with_experiment_context(
        ctx["step8c_output_dir"] / "step8c_bootstrap_metrics.json", ctx,
    )

    log.info("[%s] STEP 8D (thermal feature ablation, namespaced)", ctx["experiment_id"])
    results["step8d"] = step8d.run_step8d(ctx=ctx, force=force)
    _enrich_metadata_with_experiment_context(
        ctx["step8d_output_dir"] / "step8d_ablation_metrics.json", ctx,
    )

    log.info("[%s] STEP 8E (compact final report, namespaced)", ctx["experiment_id"])
    results["step8e"] = write_final_report(ctx, results)

    return results


def main(
    experiment_id: str = "kozan_2023",
    dry_run: bool = False,
    force: bool = False,
    allow_no_step7: bool = False,
) -> dict:
    ctx = build_experiment_context(experiment_id)
    log_context_summary(ctx, log)

    is_kozan = ctx["is_kozan"]
    if not is_kozan:
        _assert_paths_are_safely_namespaced(ctx)
    _assert_no_leakage_features()

    if dry_run:
        log.info("[dry-run] Planlanan yollar ve girdi durumu:")
        input_status = _dry_run_input_check(ctx, allow_no_step7=allow_no_step7)
        log.info("[dry-run] Hiçbir raster okuma/yazma, hiçbir model eğitimi ÇALIŞTIRILMADI.")
        return {
            "experiment_id": experiment_id, "ran": False, "reason": "dry_run",
            "input_status": input_status,
        }

    if not is_kozan:
        labels = _label_paths(ctx)
        if not labels["mcd64a1_raw"].exists():
            raise Step8RunnerError(
                f"Gerekli MCD64A1 label bulunamadı: {labels['mcd64a1_raw']}. "
                "Önce Step6/Step6B'yi (namespaced) çalıştırın."
            )
        predictors = _predictor_paths(ctx)
        fused_path = predictors["fused_lst_celsius"]
        if not Path(fused_path).exists() and not allow_no_step7:
            raise Step8RunnerError(
                f"Step7 fused LST bulunamadı: {fused_path}. Önce "
                "scripts/run_step7_downscaling_only.py'yi çalıştırın, ya da "
                "yalnızca --allow-no-step7 ile (thermal feature seti azalmış "
                "şekilde) devam edin."
            )

    if is_kozan:
        results = _run_kozan_step8(force=force)
    else:
        results = _run_experiment_step8(ctx, force=force)

    log.info("=" * 60)
    log.info("STEP8 (A-E) ÇALIŞTIRICISI TAMAMLANDI [experiment=%s]", experiment_id)
    log.info("=" * 60)
    return {"experiment_id": experiment_id, "ran": True, "results": results}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step0H: deney-farkında (experiment-aware) Step8A-E "
        "burned-area association modelleme çalıştırıcısı. Label-honest "
        "~500 m MCD64A1 hücre seviyesinde çalışır; 30 m piksel ASLA label "
        "olarak kullanılmaz. kozan_2023 legacy davranışını korur; diğer "
        "deneyler (örn. manavgat_2021) tamamen namespaced çalışır."
    )
    parser.add_argument("--experiment", type=str, default="kozan_2023")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Hiçbir şey çalıştırma; girdi/çıktı yollarını + durumu bas.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Adımların çıktıları zaten varsa üzerine yaz.",
    )
    parser.add_argument(
        "--allow-no-step7", action="store_true",
        help="Step7 fused LST eksikse bile devam et (thermal feature seti azalır).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        experiment_id=args.experiment,
        dry_run=args.dry_run,
        force=args.force,
        allow_no_step7=args.allow_no_step7,
    )