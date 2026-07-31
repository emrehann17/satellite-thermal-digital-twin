"""
core/pipeline_orchestrator.py

scripts/main.py'nin (deney-farkinda CLI orkestratoru) kullandigi kucuk,
BILIMSEL OLMAYAN orkestrasyon katmani.

Bu modul:
    - asama (stage) tanimlarini ve sirasini tutar (gate -> predictors ->
      step7 -> seam-audit -> step8)
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
STAGE_ORDER = ["gate", "predictors", "scene-provenance", "step7", "seam-audit", "seam-localization", "step8"]
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
        "seam_audit_output_dir": ctx["output_root"] / "qa" / "seam_audit" / "v2",
        "scene_provenance_output_dir": ctx["output_root"] / "qa" / "source_scene_provenance" / "v1",
        "seam_localization_output_dir": ctx["output_root"] / "qa" / "seam_localization" / "v1",
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
    log.info("[plan] seam_audit_output_dir: %s", plan["seam_audit_output_dir"])
    log.info("[plan] scene_provenance_output_dir: %s", plan["scene_provenance_output_dir"])
    log.info("[plan] seam_localization_output_dir: %s", plan["seam_localization_output_dir"])


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


def run_seam_audit_stage(
    experiment_id: str, dry_run: bool, force: bool,
    products: list[str] | str | None = None,
    scales: list[str] | str | None = None,
) -> dict:
    """Read-only Seam Audit V2 stage; V1 remains available as an audit record."""
    from scripts.run_seam_audit_v2 import main as run_seam_audit

    return run_seam_audit(
        experiment_id=experiment_id, dry_run=dry_run, force=force,
        products=products, scales=scales,
    )


def run_scene_provenance_stage(
    experiment_id: str, dry_run: bool, force: bool, mode: str = "metadata_only",
) -> dict:
    """Build local scene metadata and lineage; never submit GEE tasks."""
    from scripts.run_source_scene_provenance import main as runner

    return runner(experiment_id=experiment_id, dry_run=dry_run, force=force, mode=mode)


def run_seam_localization_stage(
    experiment_id: str, dry_run: bool, force: bool,
    manual_boundaries: list[str] | str | None = None,
) -> dict:
    """Track stable boundary identities through producer-ordered artifacts."""
    from scripts.run_seam_localization import main as runner

    return runner(
        experiment_id=experiment_id, dry_run=dry_run, force=force,
        manual_boundaries=manual_boundaries,
    )


STAGE_DISPATCH: dict[str, Callable] = {
    "gate": run_gate_stage,
    "predictors": run_predictors_stage,
    "scene-provenance": run_scene_provenance_stage,
    "step7": run_step7_stage,
    "seam-audit": run_seam_audit_stage,
    "seam-localization": run_seam_localization_stage,
    "step8": run_step8_stage,
}


def dispatch_stage(
    stage: str, experiment_id: str, dry_run: bool, force: bool,
    predictor_mode: str, export_labels: bool,
    seam_products: list[str] | str | None = None,
    seam_scales: list[str] | str | None = None,
    provenance_mode: str = "metadata_only",
    manual_boundaries: list[str] | str | None = None,
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
    if stage == "scene-provenance":
        return run_scene_provenance_stage(experiment_id, dry_run, force, provenance_mode)
    if stage == "step7":
        return run_step7_stage(experiment_id, dry_run, force)
    if stage == "seam-audit":
        return run_seam_audit_stage(experiment_id, dry_run, force, seam_products, seam_scales)
    if stage == "seam-localization":
        return run_seam_localization_stage(experiment_id, dry_run, force, manual_boundaries)
    if stage == "step8":
        return run_step8_stage(experiment_id, dry_run, force)
    raise OrchestratorError(f"Bilinmeyen asama: '{stage}'. Gecerli asamalar: {STAGE_ORDER}.")


def run_experiment_plan(
    experiment_id: str, from_stage: str, to_stage: str,
    predictor_mode: str, export_labels: bool, dry_run: bool, force: bool,
    seam_products: list[str] | str | None = None,
    seam_scales: list[str] | str | None = None,
    provenance_mode: str = "metadata_only",
    manual_boundaries: list[str] | str | None = None,
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
                seam_products=seam_products, seam_scales=seam_scales,
                provenance_mode=provenance_mode,
                manual_boundaries=manual_boundaries,
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


def run_shift_audit_stage(
    source_id: str, target_id: str, dry_run: bool, force: bool,
    report_only: bool = False,
) -> dict:
    """scripts/run_cross_region_shift_audit.py:main() -- Step9E (post-hoc,
    Step9A-D'yi DEGISTIRMEZ, hicbir model YENIDEN EGITMEZ).

    report_only=True: Part A-F'yi YENIDEN HESAPLAMAZ; yalnizca mevcut
    distribution_shift_audit.json'un safe_wording + Step9B provenance
    alanlarini gunceller (bkz.
    src/step9e_distribution_shift_audit.py:regenerate_report_only)."""
    from scripts.run_cross_region_shift_audit import main as run_cross_region_shift_audit

    return run_cross_region_shift_audit(
        source_id=source_id, target_id=target_id, dry_run=dry_run, force=force,
        report_only=report_only,
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


def run_self_cal_transfer_stage(
    source_id: str, target_id: str, reverse: bool, dry_run: bool, force: bool,
    bootstrap_replicates: int = 1000, seed: int | None = None,
    report_only: bool = False,
) -> dict:
    """scripts/run_step10_self_calibrated_transfer.py:main() -- Step10
    (preregistered, target-label-blind unsupervised covariate adaptation:
    raw_source_only/regionwise_zscore/coral_after_regionwise_zscore).
    Step9A-F'yi DEGISTIRMEZ."""
    from scripts.run_step10_self_calibrated_transfer import main as run_step10_self_calibrated_transfer
    from core.config import STEP10_BOOTSTRAP_REPLICATES, STEP10_RANDOM_STATE

    return run_step10_self_calibrated_transfer(
        source_id=source_id, target_id=target_id, reverse=reverse, dry_run=dry_run, force=force,
        bootstrap_replicates=bootstrap_replicates or STEP10_BOOTSTRAP_REPLICATES,
        seed=seed if seed is not None else STEP10_RANDOM_STATE, report_only=report_only,
    )


def run_step8_robustness_stage(
    experiments: list[str], block_sizes_cells: list[int], dry_run: bool, force: bool,
) -> dict:
    """Dispatch the frozen Step8 large-block robustness runner unchanged."""
    from scripts.run_step8_large_block_robustness import main as run_step8_large_block_robustness

    return run_step8_large_block_robustness(
        experiments=experiments,
        block_sizes_cells=block_sizes_cells,
        dry_run=dry_run,
        force=force,
    )


def run_step10_stage(
    source_id: str, target_id: str, reverse: bool, dry_run: bool, force: bool,
    report_only: bool = False, bootstrap_replicates: int = 1000, seed: int | None = None,
) -> dict:
    """Thin alias of run_self_cal_transfer_stage for the user-facing `step10`
    command. Step10 IS the preregistered, target-label-blind self-calibrated
    transfer; target labels are never used for adaptation/normalization/CORAL/
    threshold/calibration. Delegates to the existing runner; no scientific
    logic is duplicated here."""
    return run_self_cal_transfer_stage(
        source_id=source_id, target_id=target_id, reverse=reverse,
        dry_run=dry_run, force=force, bootstrap_replicates=bootstrap_replicates,
        seed=seed, report_only=report_only,
    )


def run_large_block_robustness_stage(
    dry_run: bool, force: bool, run_large_block_fit: bool = False,
) -> dict:
    """Dispatch the FORMAL Step8B primary-population (all_valid) large-block
    robustness runner. The 2-cell equivalence gate is required before any
    10/20-cell fit; without --run-large-block-fit the run stops after the gate.
    Original Step8 outputs are read-only. Delegates to the standalone runner's
    callable; no analysis logic is duplicated here."""
    from scripts.run_step8_large_block_robustness_primary_all_valid import main as run_primary_all_valid

    return run_primary_all_valid(
        dry_run=dry_run, force=force, run_large_block_fit=run_large_block_fit,
    )


def run_step8_big_block_robustness_stage(
    experiment: str, block_sizes: list[int], dry_run: bool, force: bool,
    regenerate_reports_only: bool = False, output_root: Optional[str] = None,
) -> dict:
    """Dispatch the single-experiment Step8 big-spatial-block robustness
    runner unchanged. Unlike run_step8_robustness_stage/
    run_large_block_robustness_stage (both frozen to the manavgat_2021/
    bejis_2022 pair), this stage accepts any experiment_id at runtime -- no
    AOI is hard-coded in the orchestrator or the underlying runner.
    regenerate_reports_only=True reads ONLY the frozen artifacts from a
    prior full run and regenerates JSON/Markdown/CSV/manifest reports --
    it never fits models, builds folds, generates predictions, or samples
    bootstrap replicates."""
    from scripts.run_step8_big_block_robustness import main as run_step8_big_block_robustness

    return run_step8_big_block_robustness(
        experiment=experiment, block_sizes=block_sizes, dry_run=dry_run, force=force,
        regenerate_reports_only=regenerate_reports_only, output_root=output_root,
    )


def run_concept_shift_stage(
    source_id: str, target_id: str, dry_run: bool, force: bool,
    output_root: Optional[str] = None,
) -> dict:
    """Dispatch the completed Step9G univariate feature-AUC direction-reversal
    analysis (population burnable_tree_shrub_grass; raw feature-value ROC-AUC;
    no inversion/normalization/imputation; 10-cell spatial-block bootstrap).
    This COMPUTES metrics. Target labels are not used for any adaptation.
    Delegates to the Step9G module's callable."""
    from src.step9g_univariate_feature_auc_direction_reversal import run_analysis as run_step9g

    return run_step9g(
        source_id=source_id, target_id=target_id,
        dry=dry_run, force=force,
        output_root=Path(output_root) if output_root else None,
    )


def run_concept_shift_integration_stage(
    source_id: str, target_id: str, dry_run: bool, force: bool,
) -> dict:
    """Dispatch the CANONICAL Step9G integration-v2 report layer. This is
    REPORT-ONLY: it reuses the frozen Step9G numeric outputs verbatim (no AUC/
    bootstrap/CI/reversal recomputation) and only corrects how Step9E/9F/10 are
    integrated. Writes under the canonical integration_v2 namespace; original
    Step9G outputs are immutable. Delegates to the v2 module's callable."""
    from src.step9g_integration_correction_v2 import run_correction as run_step9g_integration_v2

    return run_step9g_integration_v2(
        source_id=source_id, target_id=target_id,
        dry=dry_run, force=force,
    )


def run_concept_shift_report_revision_stage(
    source_id: str, target_id: str, dry_run: bool, force: bool,
) -> dict:
    """Dispatch the Step9G v1 final-report REPORT-ONLY semantic revision
    (report schema v2). Never recomputes AUC/CI/bootstrap/reversal_status;
    revises step9g_final_report.json/.md IN PLACE at the same canonical
    path, preserving analysis_id. Distinct from
    run_concept_shift_integration_stage (which corrects Step9E/9F/10
    integration in a separate namespace) -- this corrects
    integrated_interpretation wording and the
    thermal_features_consistent_with_step9e feature-set bug."""
    from src.step9g_report_revision import revise_report

    return revise_report(
        source_id=source_id, target_id=target_id,
        dry_run=dry_run, force=force,
    )


def run_concept_shift_compare_stage(experiments: list[str], dry_run: bool, force: bool) -> dict:
    """Dispatch the generic, REPORT-ONLY multi-experiment Step9G
    univariate-AUC comparison for a caller-supplied explicit set of
    experiment IDs. Reads only already-frozen Step9G v1 pair reports (never
    reruns concept-shift, never recomputes an AUC/CI/bootstrap draw/
    reversal classification). No experiment ID, AOI name, or pair is
    hard-coded anywhere in this dispatch or in the underlying package."""
    from src.step9g_multi_aoi_comparison.build import run_comparison

    try:
        return run_comparison(experiments, dry_run=dry_run, force=force)
    except ValueError as exc:
        # ConsistencyError / ScientificContractError are plain ValueError
        # subclasses; ComparisonError is already a SystemExit and propagates
        # unchanged.
        raise SystemExit(str(exc)) from exc


def run_multi_aoi_transfer_synthesis_stage(
    aois: list[str], dry_run: bool, force: bool, output_root: Optional[str] = None,
) -> dict:
    """Dispatch the generic, REPORT-ONLY multi-AOI transfer synthesis for a
    caller-supplied set of 2-5 AOI experiment IDs. Reads only already-frozen
    Step8/Step9B-G/Step10 outputs (via src/multi_aoi_transfer_synthesis) and
    NEVER runs modeling, adaptation, prediction, or bootstrap code. No AOI
    name, path, or pair is hard-coded anywhere in this dispatch or in the
    underlying package; everything is derived generically from `aois`."""
    from src.multi_aoi_transfer_synthesis.aoi_set import AoiSetError, validate_aoi_set
    from src.multi_aoi_transfer_synthesis.build import SynthesisBuildError, build_synthesis
    from src.multi_aoi_transfer_synthesis import manifest as multi_aoi_manifest
    from src.multi_aoi_transfer_synthesis import render as multi_aoi_render

    try:
        aoi_set = validate_aoi_set(aois)
    except AoiSetError as exc:
        raise SystemExit(str(exc)) from exc

    if output_root:
        out_dir = Path(output_root) / aoi_set.canonical_set_id
    else:
        out_dir = (
            PROJECT_ROOT / "outputs" / "diagnostics"
            / "multi_aoi_transfer_synthesis" / aoi_set.canonical_set_id
        )

    if dry_run:
        synthesis = build_synthesis(list(aois), dry_run=True, output_root=str(out_dir))
        planned_output_paths = {
            name: str(out_dir / name)
            for name in (
                "multi_aoi_transfer_synthesis.json",
                "multi_aoi_transfer_synthesis.md",
                "multi_aoi_transfer_matrix.csv",
                "multi_aoi_feature_stability.csv",
                "multi_aoi_within_region.csv",
                "multi_aoi_manifest.json",
            )
        }
        return {
            "ran": False,
            "dry_run": True,
            "canonical_set_id": synthesis["canonical_set_id"],
            "aois_display_order": synthesis["aois_display_order"],
            "aois_canonical_order": synthesis["aois_canonical_order"],
            "ordered_directions": synthesis["ordered_directions"],
            "unordered_pairs": synthesis["unordered_pairs"],
            "resolved_inputs": synthesis["resolved_inputs"],
            "resolution_errors": synthesis["resolution_errors"],
            "output_root": str(out_dir),
            "planned_output_paths": planned_output_paths,
        }

    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise SystemExit(
            f"Output directory already exists and is non-empty: {out_dir}. "
            "Use --force to overwrite (original frozen Step8/Step9/Step10 "
            "inputs are never modified regardless)."
        )

    try:
        synthesis = build_synthesis(list(aois), dry_run=False, output_root=str(out_dir))
    except SynthesisBuildError as exc:
        raise SystemExit(str(exc)) from exc

    output_paths = multi_aoi_render.render_all(synthesis, out_dir)
    manifest = multi_aoi_manifest.render_manifest(
        synthesis, output_paths, out_dir / "multi_aoi_manifest.json"
    )
    return {
        "ran": True,
        "dry_run": False,
        "canonical_set_id": synthesis["canonical_set_id"],
        "aois_display_order": synthesis["aois_display_order"],
        "aois_canonical_order": synthesis["aois_canonical_order"],
        "output_root": str(out_dir),
        "output_paths": {name: str(path) for name, path in output_paths.items()},
        "manifest_path": str(out_dir / "multi_aoi_manifest.json"),
        "input_families_resolved": manifest["input_families_resolved"],
    }


def run_evia_signed_auc_stage(
    experiment_id: str, bootstrap_replicates: int, seed: int, dry_run: bool, force: bool,
) -> dict:
    """Dispatch the REPORT-ONLY Evia signed-AUC spatial-block bootstrap.

    Reads the frozen Step8A modeling dataset plus the frozen Step9G univariate
    tables; never runs modeling, adaptation, prediction or Step10. Writes only
    under outputs/diagnostics/evia_signed_auc_bootstrap/<experiment_id>/."""
    from src.evia_signed_auc_bootstrap import EviaSignedAucError, run as run_signed_auc

    try:
        return run_signed_auc(
            experiment_id=experiment_id,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
            dry_run=dry_run,
            force=force,
        )
    except EviaSignedAucError as exc:
        raise SystemExit(str(exc)) from exc


def run_transfer_decomposition_stage(
    aois: list[str], bootstrap_replicates: int, seed: int, dry_run: bool, force: bool,
) -> dict:
    """Dispatch the REPORT-ONLY multi-AOI transfer decomposition.

    Recomputes gap/recovery decomposition and its uncertainty from ALREADY
    FROZEN Step8B/Step9B/Step10 artefacts. Never re-runs models, adaptation,
    predictions or the Step10 bootstrap itself."""
    from src.transfer_decomposition import (
        TransferDecompositionError,
        run as run_decomposition,
    )

    try:
        return run_decomposition(
            aois=list(aois),
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
            dry_run=dry_run,
            force=force,
        )
    except TransferDecompositionError as exc:
        raise SystemExit(str(exc)) from exc


def run_burned_pattern_audit_stage(
    experiments: Optional[list[str]], all_enabled: bool, dry_run: bool, force: bool,
) -> dict:
    """Dispatch the generic multi-experiment burned-area spatial-structure
    and descriptive comparison audit (connected components, patch-size
    distribution, elevation, land-cover composition). Read-only against
    Step8A; no AOI name is hard-coded anywhere in this dispatch or in the
    underlying module -- experiment IDs are resolved through
    core.regions/core.pipeline_orchestrator.LEGACY_EXPERIMENT_ID only."""
    from scripts.run_burned_pattern_audit import main as run_burned_pattern_audit

    return run_burned_pattern_audit(
        experiments=experiments, all_enabled=all_enabled, dry_run=dry_run, force=force,
    )


def run_marginal_aoa_stage(
    experiments: Optional[list[str]], all_enabled: bool, dry_run: bool, force: bool,
    output_root: Optional[str] = None, experiments_root: Optional[str] = None,
) -> dict:
    """Dispatch the generic, DIRECTED, LABEL-BLIND marginal
    Area-of-Applicability analysis: for every ordered source->target pair,
    whether target predictor values fall inside the marginal predictor
    support observed in the source AOI.

    Reads only frozen Step8A predictor columns through an explicit
    allow-list -- never `burned` or any other label. Fits no model, produces
    no prediction, runs no adaptation, CV or bootstrap. `source__target` and
    `target__source` are distinct analyses with distinct namespaces. No AOI
    name is hard-coded here or in the underlying module.

    `output_root`/`experiments_root` are programmatic injection points for
    tests and alternative namespaces; None means the canonical roots."""
    from scripts.run_marginal_area_of_applicability import main as run_marginal_aoa

    return run_marginal_aoa(
        experiments=experiments, all_enabled=all_enabled, dry_run=dry_run, force=force,
        output_root=output_root, experiments_root=experiments_root,
    )


def run_marginal_aoa_completion_stage(
    experiments: Optional[list[str]] = None,
    from_stage: str = "plan", to_stage: str = "compare",
    dry_run: bool = False, resume: bool = False, allow_earth_engine: bool = False,
    output_root: Optional[str] = None, experiments_root: Optional[str] = None,
) -> dict:
    """Dispatch the marginal AoA COMPLETION analysis.

    Produces the three components `marginal_aoa.v1` lacks: an
    importance-weighted predictor-space dissimilarity (DIRECTED), a climatic
    distance (SYMMETRIC) and a geographic distance (SYMMETRIC), over 12
    directed pairs.

    The diagnostic is `target-label-blind, source-model-informed`: feature
    weights come from a Step8B model fitted on the SOURCE label, while the
    target label, target burn date and every transfer metric stay outside the
    computation. Fits no model, runs no bootstrap, produces no composite
    scalar index, and writes only under
    outputs/diagnostics/marginal_aoa_completion/<analysis_id>/.

    Only the `climate-export` stage may contact Earth Engine, and only when
    `allow_earth_engine` is explicitly set -- it is never started implicitly.
    Every other stage is local and read-only against frozen inputs.

    `output_root`/`experiments_root` are programmatic injection points for
    tests and alternative namespaces; None means the canonical roots."""
    from scripts.run_marginal_aoa_completion import main as run_completion

    return run_completion(
        experiments=experiments, from_stage=from_stage, to_stage=to_stage,
        dry_run=dry_run, resume=resume, allow_earth_engine=allow_earth_engine,
        output_root=output_root, experiments_root=experiments_root,
    )


def run_coral_lambda_sensitivity_stage(
    from_stage: str = "plan", to_stage: str = "summarize",
    dry_run: bool = False, resume: bool = False, force: bool = False,
    output_root: Optional[str] = None, experiments_root: Optional[str] = None,
) -> dict:
    """Dispatch the local, frozen CORAL additive-ridge sensitivity analysis."""
    from scripts.run_coral_lambda_sensitivity import main as run_sensitivity
    return run_sensitivity(
        from_stage=from_stage, to_stage=to_stage, dry_run=dry_run,
        resume=resume, force=force, output_root=output_root,
        experiments_root=experiments_root,
    )


def run_few_shot_recovery_stage(
    experiments: Optional[list[str]] = None,
    from_stage: str = "plan", to_stage: str = "summarize",
    dry_run: bool = False, resume: bool = False, force: bool = False,
    output_root: Optional[str] = None, experiments_root: Optional[str] = None,
) -> dict:
    """Dispatch the few-shot recovery curve analysis.

    Answers: when k labeled target spatial blocks are supplied, how much of the
    gap between zero-shot raw transfer and the target-only within-region
    ceiling is recovered? Runs over 6 directed pairs of the three primary AOIs
    on population `burnable_tree_shrub_grass`, with budgets
    {0, 1, 2, 4, 8, 16, 32} and 10 deterministic repeats for every k > 0.

    DIAGNOSTIC CLASS: `target_label_supervised_few_shot_adaptation_sensitivity`.
    Target labels ARE used, deliberately, for adaptation and evaluation. This
    is NOT an operational deployment claim, NOT active learning, NOT a causal
    decomposition and NOT target-label-free adaptation (that is Step10).

    Evaluation blocks never enter any training frame; adaptation blocks are
    drawn only from the target training pool of the outer fold. Recovery
    fractions are signed and unclipped, Brier is oriented by negation, and the
    only interval reported is a repeat-based selection interval -- no
    bootstrap, no p-value, no significance claim. `evia_2021_extended` is
    excluded by design. Contacts no Earth Engine and writes only under
    outputs/diagnostics/few_shot_recovery/<analysis_id>/.

    `output_root`/`experiments_root` are programmatic injection points for
    tests and alternative namespaces; None means the canonical roots."""
    from scripts.run_few_shot_recovery import main as run_few_shot

    return run_few_shot(
        experiments=experiments, from_stage=from_stage, to_stage=to_stage,
        dry_run=dry_run, resume=resume, force=force,
        output_root=output_root, experiments_root=experiments_root,
    )


def run_mugla_subsampling_stage(
    experiments: Optional[list[str]] = None,
    from_stage: str = "plan", to_stage: str = "summarize",
    dry_run: bool = False, resume: bool = False, force: bool = False,
    output_root: Optional[str] = None, experiments_root: Optional[str] = None,
) -> dict:
    """Dispatch the Muğla size-matched subsampling sensitivity analysis.

    Answers: when the Muğla modeling population (41,730 primary cells) is
    reduced to exactly Manavgat's cell count (20,511), how do within-region
    performance, Muğla-as-source raw transfer and Muğla-as-target evaluation
    move relative to their full-population references? 20 deterministic
    repeats on population `burnable_tree_shrub_grass`, over the three primary
    AOIs.

    DIAGNOSTIC CLASS: `population_size_matched_subsampling_sensitivity`. This is
    a TOTAL-SAMPLE-SIZE sensitivity analysis and nothing else. Prevalence is
    preserved within integer rounding limits; the Muğla positive count is NOT
    equalised to Manavgat's; no predictor or label structure is altered. It is
    NOT a causal decomposition, NOT proof of a regional effect and NOT an
    operational deployment claim.

    Cells are drawn without replacement from ~5 km (10-cell) block x label
    strata under an integer-exact Hamilton largest-remainder allocation, so the
    allocation and the per-fold composition are identical in every repeat. The
    5-fold spatial fold mapping is inherited unchanged from the frozen
    full-Muğla 10-cell artifact and is never re-optimised. One Muğla-as-source
    fit serves both targets; the Muğla-as-target arm reuses frozen raw-transfer
    predictions and fits nothing. No bootstrap, no threshold selection and no
    probability statement of any kind. `evia_2021` and `evia_2021_extended` are
    excluded by design. Contacts no Earth Engine and writes only under
    outputs/diagnostics/mugla_subsampling/<analysis_id>/.

    `output_root`/`experiments_root` are programmatic injection points for
    tests and alternative namespaces; None means the canonical roots."""
    from scripts.run_mugla_subsampling import main as run_mugla_subsampling

    return run_mugla_subsampling(
        experiments=experiments, from_stage=from_stage, to_stage=to_stage,
        dry_run=dry_run, resume=resume, force=force,
        output_root=output_root, experiments_root=experiments_root,
    )


def run_window_closure_sensitivity_stage(
    experiment_id: str, shifts: Optional[list[int]] = None,
    from_stage: str = "plan", to_stage: str = "compare",
    dry_run: bool = False, force: bool = False, resume: bool = False,
    output_root: Optional[str] = None, experiments_root: Optional[str] = None,
) -> dict:
    """Dispatch the window-closure sensitivity analysis.

    Shifts the canonical predictor window earlier by preregistered day counts
    while preserving its LENGTH, so every predictor derived from that window
    (current Landsat LST/NDVI, each baseline year, current-window MODIS and
    the Step5/5C/7/8A products above them) is rebuilt coherently. The label
    window, DEM, slope, landcover, grid, feature registry, hyper-parameters,
    seed and spatial block definition are all frozen.

    Writes only under outputs/diagnostics/window_closure_sensitivity/; never
    touches the canonical production namespace. Live Earth Engine work
    happens only when an export stage is explicitly selected and dry_run is
    False. No AOI name is hard-coded here or in the underlying module."""
    from scripts.run_window_closure_sensitivity import main as run_window_closure

    return run_window_closure(
        experiment_id=experiment_id, shifts=shifts,
        from_stage=from_stage, to_stage=to_stage,
        dry_run=dry_run, force=force, resume=resume,
        output_root=output_root, experiments_root=experiments_root,
    )


def run_domain_classifier_audit_stage(
    experiments: Optional[list[str]], all_enabled: bool, dry_run: bool, force: bool,
) -> dict:
    """Dispatch the generic multi-experiment pairwise domain-classifier
    (covariate-separability) diagnostic. Read-only against Step8A; all
    unordered pairs among the resolved experiments are generated
    automatically. No AOI name is hard-coded anywhere in this dispatch or
    in the underlying module."""
    from scripts.run_domain_classifier_audit import main as run_domain_classifier_audit

    return run_domain_classifier_audit(
        experiments=experiments, all_enabled=all_enabled, dry_run=dry_run, force=force,
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
