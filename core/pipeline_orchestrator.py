"""
core/pipeline_orchestrator.py

scripts/main.py'nin (deney-farkinda CLI orkestratoru) kullandigi kucuk,
BILIMSEL OLMAYAN orkestrasyon katmani.

Bu modul:
    - asama (stage) tanimlarini ve sirasini tutar (gate -> predictors ->
      step7 -> step8)
    - asama araligini (--from-stage/--to-stage) dogrular
    - bir deney (experiment) icin insan-okunabilir bir "plan" (ExperimentContext
      + asama listesi + predictor modu + output kokleri) uretir
    - her asamayi, KENDI mevcut callable API'sine sahip runner'a
      (scripts/run_label_gate_only.py, scripts/run_predictors_only.py,
      scripts/run_step7_downscaling_only.py, scripts/run_step8_modeling.py)
      dispatch eder
    - legacy (Kozan, Step1-Step8E, Google Drive tabanli) tam pipeline
      calistiricisini barindirir -- bu, ONCEKI scripts/main.py'nin birebir
      AYNI adim sirasi/mantigidir; yalnizca YER DEGISTIRMISTIR (main.py'yi
      ince/thin tutmak icin), scientific mantik DEGISTIRILMEMISTIR.

BU MODUL:
    - hicbir Step'in bilimsel/istatistiksel mantigini YENIDEN UYGULAMAZ
      (yalnizca mevcut callable'lari cagirir).
    - hicbir GEE/egitim islemini kendisi CALISTIRMAZ; tumunu ilgili
      runner'a devreder.
    - dry-run modunda, her asamanin KENDI mevcut dry-run implementasyonunu
      (dry_run=True) cagirir -- ikinci bir divergent dry-run mantigi
      UYGULAMAZ.
"""

from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from core.regions import get_active_experiment, get_experiment_output_root

BASE_DIR = PROJECT_ROOT
log, log_file = setup_logger("pipeline_orchestrator")


class OrchestratorError(SystemExit):
    """Fail-fast error for the orchestrator (diğer step'lerle aynı konvansiyon)."""


# =============================================================================
# Asama (stage) tanimlari
# =============================================================================
STAGE_ORDER = ["gate", "predictors", "step7", "step8"]
PREDICTOR_MODES = ("export", "local-only")

LEGACY_EXPERIMENT_ID = "kozan_2023"
# Su an legacy komut YALNIZCA kozan_2023'u destekler. Repo ileride baska bir
# deneyi acikca "legacy-compatible" olarak tanimlarsa buraya eklenmelidir --
# YALNIZCA repo'nun kendi kaydinda (core/regions.py) acikca boyle bir deney
# varsa (bkz. cmd_legacy dogrulamasi, scripts/main.py).
LEGACY_COMPATIBLE_EXPERIMENT_IDS = (LEGACY_EXPERIMENT_ID,)


def validate_stage_range(from_stage: str, to_stage: str) -> list[str]:
    """--from-stage/--to-stage'i dogrular ve calistirilacak asama listesini
    (STAGE_ORDER alt-dizisi) dondurur.

    Raises:
        OrchestratorError: gecersiz asama adi veya --from-stage,
            --to-stage'ten SONRA ise.
    """
    if from_stage not in STAGE_ORDER:
        raise OrchestratorError(
            f"Gecersiz --from-stage: '{from_stage}'. Gecerli asamalar (sirali): {STAGE_ORDER}."
        )
    if to_stage not in STAGE_ORDER:
        raise OrchestratorError(
            f"Gecersiz --to-stage: '{to_stage}'. Gecerli asamalar (sirali): {STAGE_ORDER}."
        )
    start_idx, end_idx = STAGE_ORDER.index(from_stage), STAGE_ORDER.index(to_stage)
    if start_idx > end_idx:
        raise OrchestratorError(
            f"--from-stage ('{from_stage}') --to-stage'ten ('{to_stage}') SONRA olamaz "
            f"(asama sirasi: {STAGE_ORDER})."
        )
    return STAGE_ORDER[start_idx:end_idx + 1]


# =============================================================================
# Namespace guvenlik kontrolu
# =============================================================================
def _assert_context_is_safely_namespaced(ctx: dict) -> None:
    """Kozan-disi (is_kozan=False) bir ExperimentContext'in TUM girdi/cikti
    yollarinin kendi output_root'u (`outputs/experiments/<experiment_id>/`)
    altinda kaldigini, legacy paylasilan Kozan yollarina (`outputs/step5/`,
    `outputs/validation/labels/` vb.) SIZMADIGINI dogrular.

    kozan_2023 icin BILEREK atlanir (legacy paylasilan yollari kullanmasi
    beklenen/dogru davranistir -- bkz. core/experiment_context.py).
    """
    if ctx.get("is_kozan"):
        return

    output_root = Path(ctx["output_root"])
    if "experiments" not in output_root.parts or ctx["experiment_id"] not in output_root.parts:
        raise OrchestratorError(
            f"Namespacing guvenlik kontrolu BASARISIZ: '{ctx['experiment_id']}' icin "
            f"output_root ('{output_root}') beklenen outputs/experiments/<experiment_id>/ "
            "deseninde degil."
        )

    paths_to_check = [
        ctx.get("data_root"), ctx.get("step5_output_dir"), ctx.get("step5c_output_dir"),
        ctx.get("gate_labels_dir"), ctx.get("step7a_output_dir"), ctx.get("step8a_output_dir"),
    ]
    for p in paths_to_check:
        if p is None:
            continue
        p = Path(p)
        try:
            p.relative_to(output_root)
        except ValueError:
            raise OrchestratorError(
                f"Namespacing guvenlik kontrolu BASARISIZ: '{p}' output_root "
                f"('{output_root}') altinda degil -- Kozan-disi bir deney icin bu "
                "legacy paylasilan bir yola sizinti olabilir."
            )


# =============================================================================
# Deney plani (yalnizca bilgilendirme/dry-run amacli -- hicbir side-effect yok)
# =============================================================================
def describe_experiment_plan(
    experiment_id: str, from_stage: str, to_stage: str,
    predictor_mode: str, export_labels: bool,
) -> dict:
    """Bir 'experiment' calistirmasi icin insan-okunabilir bir plan uretir.

    Hicbir GEE cagrisi yapmaz, hicbir dosya olusturmaz/degistirmez -- yalnizca
    kayit defteri + ExperimentContext'ten bilgi toplar.

    Raises:
        OrchestratorError / ValueError: deney bilinmiyorsa, disabled ise,
            asama araligi gecersizse veya namespace guvenligi ihlal edilmisse.
    """
    from core.experiment_context import build_experiment_context

    if predictor_mode not in PREDICTOR_MODES:
        raise OrchestratorError(f"--predictor-mode su degerlerden biri olmali: {PREDICTOR_MODES}.")

    exp = get_active_experiment(experiment_id)  # ValueError: bilinmeyen/disabled deney
    stages = validate_stage_range(from_stage, to_stage)
    ctx = build_experiment_context(experiment_id)
    _assert_context_is_safely_namespaced(ctx)

    return {
        "experiment_id": experiment_id,
        "display_name": exp["display_name"],
        "role": exp["role"],
        "region_key": exp["region_key"],
        "predictor_window": [exp["predictor_start_date"], exp["predictor_end_date"]],
        "label_window": [exp["label_start_date"], exp["label_end_date"]],
        "baseline_years": exp["baseline_years"],
        "is_kozan": ctx["is_kozan"],
        "output_root": ctx["output_root"],
        "data_root": ctx["data_root"],
        "gate_labels_dir": ctx["gate_labels_dir"],
        "step7_output_dirs": {
            k: ctx[k] for k in (
                "step7a_output_dir", "step7b_output_dir", "step7c_output_dir",
                "step7d_output_dir", "step7e_output_dir",
            )
        },
        "step8_output_dirs": {
            k: ctx[k] for k in (
                "step8a_output_dir", "step8b_output_dir", "step8c_output_dir",
                "step8d_output_dir", "step8e_output_dir",
            )
        },
        "stages": stages,
        "predictor_mode": predictor_mode,
        "export_labels": export_labels,
    }


def log_experiment_plan(plan: dict) -> None:
    """describe_experiment_plan() ciktisini standart, okunabilir formatta loglar."""
    log.info("[plan] experiment_id: %s (%s)", plan["experiment_id"], plan["display_name"])
    log.info("[plan] role: %s | region_key: %s | is_kozan: %s", plan["role"], plan["region_key"], plan["is_kozan"])
    log.info("[plan] predictor window: %s -> %s", *plan["predictor_window"])
    log.info("[plan] label window: %s -> %s", *plan["label_window"])
    log.info("[plan] baseline years: %s", plan["baseline_years"])
    log.info("[plan] stages to run (order): %s", plan["stages"])
    log.info("[plan] predictor_mode: %s", plan["predictor_mode"])
    log.info("[plan] export_labels (gate stage only): %s", plan["export_labels"])
    log.info("[plan] output_root: %s", plan["output_root"])
    log.info("[plan] data_root: %s", plan["data_root"])
    log.info("[plan] gate_labels_dir: %s", plan["gate_labels_dir"])
    for k, v in plan["step7_output_dirs"].items():
        log.info("[plan]   %s: %s", k, v)
    for k, v in plan["step8_output_dirs"].items():
        log.info("[plan]   %s: %s", k, v)


# =============================================================================
# Asama dispatch -- her biri MEVCUT bir runner'in callable API'sini cagirir.
# Hicbir bilimsel/istatistiksel mantik burada YENIDEN UYGULANMAZ.
# =============================================================================
def run_gate_stage(experiment_id: str, dry_run: bool, force: bool, export_labels: bool) -> dict:
    """gate asamasi: Step6A gate-input hazirlama (Kozan-disi) + opsiyonel raw
    MCD64A1 BurnDate export + Step6B burned-landcover gate.

    scripts/run_label_gate_only.py:main() -- reuse edilir, YENIDEN UYGULANMAZ.
    """
    from scripts.run_label_gate_only import main as run_label_gate_only

    return run_label_gate_only(
        experiment_id=experiment_id, dry_run=dry_run,
        skip_export=False, export_labels=export_labels, force=force,
    )


def run_predictors_stage(experiment_id: str, dry_run: bool, force: bool, predictor_mode: str) -> dict:
    """predictors asamasi: experiment-aware direkt/tiled GEE local download
    (--predictor-mode export) veya yalnizca yerel Step5/Step5C isleme
    (--predictor-mode local-only).

    scripts/run_predictors_only.py:main() -- reuse edilir, YENIDEN UYGULANMAZ.
    Kozan-disi deneyler icin bu ASLA legacy Drive zincirini kullanmaz.
    """
    if predictor_mode not in PREDICTOR_MODES:
        raise OrchestratorError(f"--predictor-mode su degerlerden biri olmali: {PREDICTOR_MODES}.")

    from scripts.run_predictors_only import main as run_predictors_only

    return run_predictors_only(
        experiment_id=experiment_id, dry_run=dry_run,
        export=(predictor_mode == "export"), local_only=(predictor_mode == "local-only"),
        force=force,
    )


def run_step7_stage(experiment_id: str, dry_run: bool, force: bool) -> dict:
    """step7 asamasi: experiment-aware Step7A-E (MODIS->Landsat LST
    downscaling + fusion) zinciri.

    scripts/run_step7_downscaling_only.py:main() -- reuse edilir, YENIDEN
    UYGULANMAZ. Step7 modelleri bolgeler arasi transfer/retrain EDILMEZ
    (her deney kendi Step7 zincirini calistirir -- mevcut proje davranisi).
    """
    from scripts.run_step7_downscaling_only import main as run_step7_downscaling_only

    return run_step7_downscaling_only(experiment_id=experiment_id, dry_run=dry_run, force=force)


def run_step8_stage(experiment_id: str, dry_run: bool, force: bool) -> dict:
    """step8 asamasi: experiment-aware Step8A-E (label-honest ~500 m
    MCD64A1-cell burned-area association modelleme) zinciri.

    scripts/run_step8_modeling.py:main() -- reuse edilir, YENIDEN UYGULANMAZ.
    30 m piksel HICBIR ZAMAN label olarak kullanilmaz (Step8A semantigi
    korunur).
    """
    from scripts.run_step8_modeling import main as run_step8_modeling

    return run_step8_modeling(experiment_id=experiment_id, dry_run=dry_run, force=force)


STAGE_DISPATCH: dict[str, Callable] = {
    "gate": run_gate_stage,
    "predictors": run_predictors_stage,
    "step7": run_step7_stage,
    "step8": run_step8_stage,
}


def dispatch_stage(
    stage: str, experiment_id: str, dry_run: bool, force: bool,
    predictor_mode: str, export_labels: bool,
) -> dict:
    """Tek bir asamayi ilgili runner'a dispatch eder.

    Raises:
        OrchestratorError: bilinmeyen asama adi.
        Exception: altta yatan runner'in kendi fail-fast hatasi (YUTULMAZ,
            oldugu gibi caller'a propagate edilir).
    """
    if stage == "gate":
        return run_gate_stage(experiment_id, dry_run, force, export_labels)
    if stage == "predictors":
        return run_predictors_stage(experiment_id, dry_run, force, predictor_mode)
    if stage == "step7":
        return run_step7_stage(experiment_id, dry_run, force)
    if stage == "step8":
        return run_step8_stage(experiment_id, dry_run, force)
    raise OrchestratorError(f"Bilinmeyen asama: '{stage}'. Gecerli asamalar: {STAGE_ORDER}.")


def run_experiment_plan(
    experiment_id: str, from_stage: str, to_stage: str,
    predictor_mode: str, export_labels: bool, dry_run: bool, force: bool,
) -> dict:
    """Tam bir 'experiment' calistirmasini (plan + sirali asama dispatch)
    yurutur.

    dry_run=True ise, HER asama kendi dry_run=True moduyla cagirilir (ikinci
    bir dry-run mantigi UYGULANMAZ) -- hicbir GEE/egitim calismaz, hicbir
    bilimsel cikti dosyasi yazilmaz.

    Bir asama basarisiz olursa, exception oldugu gibi (hangi asamada
    oldugu bilgisiyle) caller'a propagate edilir -- sessizce devam EDILMEZ.
    """
    plan = describe_experiment_plan(experiment_id, from_stage, to_stage, predictor_mode, export_labels)
    log_experiment_plan(plan)

    stage_results: dict[str, dict] = {}
    for stage in plan["stages"]:
        log.info("=" * 70)
        log.info("ASAMA BASLATILIYOR: '%s' [experiment=%s, dry_run=%s]", stage, experiment_id, dry_run)
        log.info("=" * 70)
        try:
            result = dispatch_stage(
                stage, experiment_id, dry_run=dry_run, force=force,
                predictor_mode=predictor_mode, export_labels=export_labels,
            )
        except Exception as exc:  # noqa: BLE001 -- kasitli: logla + fail-fast propagate et
            log.error("ASAMA BASARISIZ: '%s' [experiment=%s]", stage, experiment_id)
            log.error("Orijinal hata: %s: %s", type(exc).__name__, exc)
            log.error(traceback.format_exc())
            raise
        stage_results[stage] = result
        log.info("ASAMA TAMAMLANDI: '%s'", stage)

    if dry_run:
        log.info(
            "[dry-run] Deney plani TAMAMLANDI; hicbir GEE/egitim CALISTIRILMADI, "
            "hicbir bilimsel cikti dosyasi YAZILMADI."
        )
    else:
        log.info("Deney plani TAMAMEN calisti: %s", plan["stages"])

    return {"experiment_id": experiment_id, "dry_run": dry_run, "plan": plan, "stage_results": stage_results}


# =============================================================================
# transfer / shift-audit -- ince sarmalayicilar (Step9A-D / Step9E YENIDEN
# UYGULANMAZ; mevcut scripts/run_cross_region_transfer.py ve
# scripts/run_cross_region_shift_audit.py reuse edilir).
# =============================================================================
def run_transfer_stage(
    source_id: str, target_id: str, reverse: bool, dry_run: bool, force: bool,
) -> dict:
    """scripts/run_cross_region_transfer.py:main() -- Step9A-D."""
    from scripts.run_cross_region_transfer import main as run_cross_region_transfer

    return run_cross_region_transfer(
        source_id=source_id, target_id=target_id, reverse=reverse, dry_run=dry_run, force=force,
    )


def run_shift_audit_stage(source_id: str, target_id: str, dry_run: bool, force: bool) -> dict:
    """scripts/run_cross_region_shift_audit.py:main() -- Step9E (post-hoc,
    Step9A-D'yi DEGISTIRMEZ, hicbir model YENIDEN EGITMEZ)."""
    from scripts.run_cross_region_shift_audit import main as run_cross_region_shift_audit

    return run_cross_region_shift_audit(
        source_id=source_id, target_id=target_id, dry_run=dry_run, force=force,
    )


def run_transfer_explore_stage(
    source_id: str, target_id: str, reverse: bool, dry_run: bool, force: bool,
    bootstrap_replicates: int = 1000, seed: int | None = None,
) -> dict:
    """scripts/run_exploratory_transfer_features.py:main() -- Step9F (kesifsel,
    post-hoc; Step9A-E'yi DEGISTIRMEZ, hicbir modeli onlar ADINA YENIDEN EGITMEZ)."""
    from scripts.run_exploratory_transfer_features import main as run_exploratory_transfer_features

    return run_exploratory_transfer_features(
        source_id=source_id, target_id=target_id, reverse=reverse, dry_run=dry_run,
        force=force, bootstrap_replicates=bootstrap_replicates, seed=seed,
    )


# =============================================================================
# LEGACY: Kozan-only Step1->Step8E (Google Drive tabanli) tam pipeline.
#
# Bu fonksiyon, ONCEKI scripts/main.py'nin `main(force)` govdesiyle
# BIREBIR AYNI adim sirasini/mantigini calistirir -- yalnizca main.py'yi
# ince (thin) tutmak icin buraya TASINMISTIR. Hicbir Step'in bilimsel
# mantigi DEGISTIRILMEMISTIR.
#
# DAVRANIS FARKI (kasitli, "GENERAL MAIN.PY REQUIREMENTS" geregi): eski
# main.py, ust seviyede TUM exception'lari yutup (yalnizca loglayip) sessizce
# basariyla bitiyormus gibi cikiyordu (exit code hep 0). Bu fonksiyon artik
# beklenmeyen bir hatayi YUTMAZ -- (Step6'nin kendi hata-toleransli
# association testi HARIC, bu ozellikle/acikca korunmustur) -- caller'in
# (scripts/main.py:cmd_legacy) net bir hata mesaji + hatali asama bilgisiyle
# exit code != 0 dondurebilmesi icin exception'i propagate eder.
# =============================================================================
def _run_step(step_name: str, step_func: Callable) -> object:
    log.info("=" * 70)
    log.info("%s BAŞLATILIYOR", step_name)
    log.info("=" * 70)
    try:
        result = step_func()
    except Exception as exc:  # noqa: BLE001 -- kasitli: logla + fail-fast propagate et
        log.error("%s BAŞARISIZ: %s: %s", step_name, type(exc).__name__, exc)
        raise
    log.info("=" * 70)
    log.info("%s TAMAMLANDI", step_name)
    log.info("=" * 70)
    return result


def run_legacy_kozan_pipeline(force: bool = False) -> dict:
    """LEGACY (Google Drive tabanli) Step1->Step8E tam pipeline'ini,
    yalnizca kozan_2023 icin, birebir eski main.py sirasiyla calistirir.

    Bu YENI varsayilan yol DEGILDIR; yalnizca `scripts/main.py legacy`
    alt-komutundan cagirilir.
    """
    import src.step1_fetch_modis as step1_fetch_modis
    import src.step2_modis_5year_mean as step2_modis_5year_mean
    import src.step2b_dem as step2b_dem
    import src.step3_landsat_lst as step3_landsat_lst
    import src.step4_export_geotiff as step4_export_geotiff
    import src.step4b_download_drive_export as step4b_download_drive_export
    import src.step5_preprocess_timeseries as step5_preprocess_timeseries
    import src.step5b_diagnostic_report as step5b_diagnostic_report
    import src.step5c_tvdi as step5c_tvdi
    import src.step6_validate_fire_relation as step6_validate_fire_relation
    import src.step6b_burned_landcover_gate as step6b_burned_landcover_gate
    import src.step7a_tiling_infrastructure as step7a_tiling_infrastructure
    import src.step7b_prepare_downscaling_dataset as step7b_prepare_downscaling_dataset
    import src.step7c_train_downscaling_model as step7c_train_downscaling_model
    import src.step7d_predict_downscaled_lst as step7d_predict_downscaled_lst
    import src.step7e_fuse_landsat_downscaled_lst as step7e_fuse_landsat_downscaled_lst
    import src.step8a_prepare_500m_modeling_dataset as step8a_prepare_500m_modeling_dataset
    import src.step8b_train_baseline_vs_thermal_model as step8b_train_baseline_vs_thermal_model
    import src.step8c_spatial_block_bootstrap_uncertainty as step8c_spatial_block_bootstrap_uncertainty
    import src.step8d_thermal_feature_ablation as step8d_thermal_feature_ablation
    import src.step8e_final_report as step8e_final_report

    start_time = datetime.now()

    log.info("#" * 80)
    log.info("LEGACY PIPELINE BAŞLIYOR (STEP1 -> STEP8E, uçtan uca, kozan_2023, Drive-tabanlı)")
    log.info("Başlangıç zamanı: %s", start_time.isoformat())
    log.info("force=%s", force)
    log.info(
        "NOT: Step6 hata-toleranslıdır (basarisiz olursa yalnizca uyarir). "
        "Label export cleanup (raw MCD64A1 BurnDate) ise ZORUNLUDUR; "
        "basarisiz olursa pipeline burada durur (Step8A gecerli DOY etiketi "
        "olmadan calisamaz). Burned-landcover gate DIAGNOSTIC'tir; "
        "cropland_dominated_control sonucu pipeline'i durdurmaz."
    )
    log.info("#" * 80)

    _run_step("STEP 1", step1_fetch_modis.main)
    _run_step("STEP 2", step2_modis_5year_mean.main)
    _run_step("STEP 2B", step2b_dem.main)
    step3_result = _run_step("STEP 3", step3_landsat_lst.main)
    _run_step("STEP 4", lambda: step4_export_geotiff.main(step3_result=step3_result))
    _run_step("STEP 4B", step4b_download_drive_export.main)
    _run_step("STEP 5", step5_preprocess_timeseries.main)
    _run_step("STEP 5C", step5c_tvdi.main)
    _run_step("STEP 5B", step5b_diagnostic_report.main)

    # Step6 burned-area association testi: GEE/veri erisimi basarisiz olursa
    # pipeline'in geri kalani etkilenmesin diye BILEREK hata-toleranslidir
    # (hicbir sonraki adim Step6'nin ciktisina bagimli degildir).
    try:
        _run_step("STEP 6", step6_validate_fire_relation.main)
    except Exception as exc:  # noqa: BLE001 -- kasitli, belgelenmis istisna
        log.warning("STEP 6 atlandı (burned-area validation başarısız): %s", exc)

    # Label export cleanup: ZORUNLUDUR, hata-toleranslı DEĞİLDİR.
    _run_step(
        "LABEL EXPORT CLEANUP (raw MCD64A1 BurnDate)",
        lambda: step6_validate_fire_relation.export_raw_mcd64a1_labels(also_binary=True),
    )

    # Burned-landcover gate: DIAGNOSTIC'tir; cropland_dominated_control
    # sonucu pipeline'i DURDURMAZ.
    _run_step(
        "BURNED-LANDCOVER GATE",
        lambda: step6b_burned_landcover_gate.main(force=force),
    )

    _run_step("STEP 7A", step7a_tiling_infrastructure.main)
    _run_step("STEP 7B", step7b_prepare_downscaling_dataset.main)
    _run_step("STEP 7C", lambda: step7c_train_downscaling_model.main(force=force))
    _run_step("STEP 7D", lambda: step7d_predict_downscaled_lst.main(force=force))
    _run_step("STEP 7E", lambda: step7e_fuse_landsat_downscaled_lst.main(force=force))

    _run_step("STEP 8A", lambda: step8a_prepare_500m_modeling_dataset.main(force=force))
    _run_step("STEP 8B", lambda: step8b_train_baseline_vs_thermal_model.main(force=force))
    _run_step("STEP 8C", lambda: step8c_spatial_block_bootstrap_uncertainty.main(force=force))
    _run_step("STEP 8D", lambda: step8d_thermal_feature_ablation.main(force=force))
    _run_step("STEP 8E", lambda: step8e_final_report.main(force=force))

    end_time = datetime.now()
    log.info("#" * 80)
    log.info("LEGACY PIPELINE TAMAMLANDI (STEP1 -> STEP8E)")
    log.info("Bitiş zamanı: %s", end_time.isoformat())
    log.info("Toplam sure: %s", end_time - start_time)
    log.info("Log dosyası: %s", log_file)
    log.info(
        "Nihai rapor: outputs/step8e/step8e_summary.md, "
        "step8e_summary.json, step8e_results_tables.xlsx, "
        "step8e_key_findings.csv."
    )
    log.info("#" * 80)

    return {
        "ran": True, "experiment_id": LEGACY_EXPERIMENT_ID,
        "start_time": start_time.isoformat(), "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
    }