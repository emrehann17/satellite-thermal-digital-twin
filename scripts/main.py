"""
main.py

satellite-thermal-digital-twin: deney-farkında (experiment-aware) CLI
orkestratörü.

Bu script İNCE (thin) tutulur -- hiçbir Step'in bilimsel/istatistiksel
mantığını yeniden uygulamaz. Tüm gerçek iş, mevcut callable API'lere sahip
runner'lara devredilir (bkz. core/pipeline_orchestrator.py):

    experiment   -> gate / predictors / step7 / seam-audit / step8 asama zinciri
                    (core/pipeline_orchestrator.py -> scripts/run_label_gate_only.py,
                    cscripts/run_predictors_only.py,
                    scripts/run_step7_downscaling_only.py,
                    scripts/run_step8_modeling.py)
    transfer     -> Step9A-D cross-region transfer
                    (scripts/run_cross_region_transfer.py)
    shift-audit  -> Step9E post-hoc distribution-shift audit
                    (scripts/run_cross_region_shift_audit.py)
    legacy       -> Kozan-only, Google Drive tabanlı Step1->Step8E tam
                    pipeline (yalnızca geriye dönük uyumluluk/tarihsel
                    Kozan reprodüksiyonu için; YENİ varsayılan yol DEĞİLDİR)

ALT-KOMUT VERİLMEDEN çalıştırma (`python scripts/main.py`) yalnızca yardım
mesajını basar ve HİÇBİR ŞEY ÇALIŞTIRMAZ (ne GEE, ne legacy pipeline).

CLI:
    python scripts/main.py --help

    python scripts/main.py experiment \\
      --experiment manavgat_2021 --from-stage predictors --to-stage step8 \\
      --predictor-mode local-only --dry-run 

    python scripts/main.py experiment \\
      --experiment bejis_2022 --from-stage predictors --to-stage step8 \\
      --predictor-mode export --force

    python scripts/main.py transfer \\
      --source manavgat_2021 --target bejis_2022 --reverse --force

    python scripts/main.py shift-audit \\
      --source manavgat_2021 --target bejis_2022 --force

    python scripts/main.py legacy --experiment kozan_2023 --force
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from core.regions import get_active_experiment
import core.pipeline_orchestrator as orch

BASE_DIR = PROJECT_ROOT
log, log_file = setup_logger("main")


EPILOG_EXAMPLES = """\
örnek komutlar:

  # yardım (hiçbir şey çalıştırmaz)
  python scripts/main.py --help

  # deney dry-run (yalnızca planı + planlanan yolları basar)
  python scripts/main.py experiment \\
    --experiment manavgat_2021 --from-stage predictors --to-stage step8 \\
    --predictor-mode local-only --dry-run

  # direkt/tiled predictor export (GEE, namespaced)
  python scripts/main.py experiment \\
    --experiment bejis_2022 --from-stage predictors --to-stage step8 \\
    --predictor-mode export --force

  # yalnızca yerel yeniden çalıştırma (GEE'ye dokunmaz)
  python scripts/main.py experiment \\
    --experiment manavgat_2021 --from-stage predictors --to-stage step8 \\
    --predictor-mode local-only --force

  # cross-region Step9A-D
  python scripts/main.py transfer \\
    --source manavgat_2021 --target bejis_2022 --reverse --force

  # Step9E post-hoc distribution-shift audit
  python scripts/main.py shift-audit \\
    --source manavgat_2021 --target bejis_2022 --force

  # Step9F kesifsel (exploratory) cross-region feature-representation deneyi
  python scripts/main.py transfer-explore \\
    --source manavgat_2021 --target bejis_2022 --reverse --force

  # Step10 preregistered unsupervised self-calibrated cross-region transfer
  python scripts/main.py self-cal-transfer \\
    --source manavgat_2021 --target bejis_2022 --reverse --force

  # Step10 (user-facing alias; target labels NOT used for adaptation)
  python scripts/main.py step10 \\
    --source manavgat_2021 --target bejis_2022 --reverse --force

  # formal Step8B primary-population (all_valid) large-block robustness
  # (2-cell equivalence gate first; add --run-large-block-fit to fit 10/20 cells)
  python scripts/main.py large-block-robustness --dry-run
  python scripts/main.py large-block-robustness --run-large-block-fit --force

  # single-experiment Step8 big-spatial-block robustness (any experiment_id)
  python scripts/main.py step8-big-block-robustness \\
    --experiment mugla_2021 --block-sizes 10 20 --dry-run
  python scripts/main.py step8-big-block-robustness \\
    --experiment mugla_2021 --block-sizes 10 20 --force

  # Step9G concept/relationship-shift diagnostic (COMPUTES metrics)
  python scripts/main.py concept-shift --source manavgat_2021 --target bejis_2022 --force

  # Step9G canonical integration-v2 report (REPORT-ONLY; recomputes nothing)
  python scripts/main.py concept-shift --source manavgat_2021 --target bejis_2022 --integration-only --force

  # legacy Kozan (Google Drive tabanlı) tam pipeline
  python scripts/main.py legacy --experiment kozan_2023 --force
"""


def _fail(command: str, exc: BaseException) -> int:
    """Ortak hata-raporlama: orijinal hatayı + hangi komutun başarısız
    olduğunu loglar (traceback dahil), kullanıcıya kısa bir özet basar,
    HİÇBİR ŞEYİ SESSİZCE YUTMADAN net bir exit code döner."""
    log.error("KOMUT BAŞARISIZ: '%s'", command)
    log.error("Orijinal hata: %s: %s", type(exc).__name__, exc)
    log.error(traceback.format_exc())
    print(f"\nHATA: '{command}' komutu başarısız oldu.")
    print(f"{type(exc).__name__}: {exc}")
    print(f"Detaylar log dosyasında: {log_file}\n")
    return 1


# =============================================================================
# Alt-komut handler'ları
# =============================================================================
def cmd_experiment(args: argparse.Namespace) -> int:
    try:
        orch.run_experiment_plan(
            experiment_id=args.experiment,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
            predictor_mode=args.predictor_mode,
            export_labels=args.export_labels,
            dry_run=args.dry_run,
            force=args.force,
            seam_products=args.seam_products,
            seam_scales=args.seam_scales,
            provenance_mode=args.provenance_mode,
            manual_boundaries=args.manual_boundaries,
        )
    except SystemExit as exc:
        return _fail("experiment", exc)
    except Exception as exc:  # noqa: BLE001 -- CLI sınırı: logla + net exit code, YUTMA
        return _fail("experiment", exc)
    return 0


def cmd_transfer(args: argparse.Namespace) -> int:
    try:
        result = orch.run_transfer_stage(
            source_id=args.source, target_id=args.target,
            reverse=args.reverse, dry_run=args.dry_run, force=args.force,
        )
    except SystemExit as exc:
        return _fail("transfer", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("transfer", exc)
    log.info("[transfer] tamamlandı: ran=%s", result.get("ran"))
    return 0


def cmd_shift_audit(args: argparse.Namespace) -> int:
    try:
        result = orch.run_shift_audit_stage(
            source_id=args.source, target_id=args.target,
            dry_run=args.dry_run, force=args.force,
            report_only=args.report_only,
        )
    except SystemExit as exc:
        return _fail("shift-audit", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("shift-audit", exc)
    log.info("[shift-audit] tamamlandı: ran=%s", result.get("ran"))
    return 0


def cmd_transfer_explore(args: argparse.Namespace) -> int:
    try:
        result = orch.run_transfer_explore_stage(
            source_id=args.source, target_id=args.target, reverse=args.reverse,
            dry_run=args.dry_run, force=args.force,
            bootstrap_replicates=args.bootstrap_replicates, seed=args.seed,
        )
    except SystemExit as exc:
        return _fail("transfer-explore", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("transfer-explore", exc)
    log.info("[transfer-explore] tamamlandı: ran=%s", result.get("ran"))
    return 0


def cmd_self_cal_transfer(args: argparse.Namespace) -> int:
    try:
        result = orch.run_self_cal_transfer_stage(
            source_id=args.source, target_id=args.target, reverse=args.reverse,
            dry_run=args.dry_run, force=args.force,
            bootstrap_replicates=args.bootstrap_replicates, seed=args.seed,
            report_only=args.report_only,
        )
    except SystemExit as exc:
        return _fail("self-cal-transfer", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("self-cal-transfer", exc)
    log.info("[self-cal-transfer] tamamlandı: ran=%s", result.get("ran"))
    return 0


def cmd_step8_robustness(args: argparse.Namespace) -> int:
    try:
        result = orch.run_step8_robustness_stage(
            experiments=args.experiments,
            block_sizes_cells=args.block_sizes_cells,
            dry_run=args.dry_run,
            force=args.force,
        )
    except SystemExit as exc:
        return _fail("step8-robustness", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("step8-robustness", exc)
    if args.dry_run:
        print(json.dumps(result, indent=2, default=str))
    log.info("[step8-robustness] tamamlandı: ran=%s", result.get("ran"))
    return 0


def cmd_step10(args: argparse.Namespace) -> int:
    try:
        result = orch.run_step10_stage(
            source_id=args.source, target_id=args.target, reverse=args.reverse,
            dry_run=args.dry_run, force=args.force, report_only=args.report_only,
            bootstrap_replicates=args.bootstrap_replicates, seed=args.seed,
        )
    except SystemExit as exc:
        return _fail("step10", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("step10", exc)
    log.info("[step10] tamamlandı: ran=%s", result.get("ran"))
    return 0


def cmd_coral_lambda_sensitivity(args: argparse.Namespace) -> int:
    try:
        result = orch.run_coral_lambda_sensitivity_stage(
            from_stage=args.from_stage, to_stage=args.to_stage,
            dry_run=args.dry_run, resume=args.resume, force=args.force,
            output_root=args.output_root, experiments_root=args.experiments_root,
        )
    except (SystemExit, ValueError, OSError) as exc:
        return _fail("coral-lambda-sensitivity", exc)
    log.info("[coral-lambda-sensitivity] tamamlandı: ran=%s", result.get("ran"))
    return 0


def cmd_large_block_robustness(args: argparse.Namespace) -> int:
    try:
        result = orch.run_large_block_robustness_stage(
            dry_run=args.dry_run, force=args.force,
            run_large_block_fit=args.run_large_block_fit,
        )
    except SystemExit as exc:
        return _fail("large-block-robustness", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("large-block-robustness", exc)
    if args.dry_run:
        print(json.dumps(result, indent=2, default=str))
    log.info("[large-block-robustness] tamamlandı: ran=%s", result.get("ran"))
    return 0


def cmd_step8_big_block_robustness(args: argparse.Namespace) -> int:
    try:
        result = orch.run_step8_big_block_robustness_stage(
            experiment=args.experiment,
            block_sizes=args.block_sizes,
            dry_run=args.dry_run,
            force=args.force,
            regenerate_reports_only=args.regenerate_reports_only,
            output_root=args.output_root,
        )
    except SystemExit as exc:
        return _fail("step8-big-block-robustness", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("step8-big-block-robustness", exc)
    if args.dry_run:
        print(json.dumps(result, indent=2, default=str))
    log.info("[step8-big-block-robustness] tamamlandı: ran=%s", result.get("ran"))
    return 0


def cmd_concept_shift(args: argparse.Namespace) -> int:
    try:
        if args.report_revision:
            result = orch.run_concept_shift_report_revision_stage(
                source_id=args.source, target_id=args.target,
                dry_run=args.dry_run, force=args.force,
            )
        elif args.integration_only:
            result = orch.run_concept_shift_integration_stage(
                source_id=args.source, target_id=args.target,
                dry_run=args.dry_run, force=args.force,
            )
        else:
            result = orch.run_concept_shift_stage(
                source_id=args.source, target_id=args.target,
                dry_run=args.dry_run, force=args.force,
                output_root=args.output_root,
            )
    except SystemExit as exc:
        return _fail("concept-shift", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("concept-shift", exc)
    if args.dry_run:
        print(json.dumps(result, indent=2, default=str))
    log.info(
        "[concept-shift] tamamlandı (report_revision=%s, integration_only=%s): ran=%s",
        args.report_revision, args.integration_only, result.get("ran"),
    )
    return 0


def cmd_concept_shift_compare(args: argparse.Namespace) -> int:
    try:
        result = orch.run_concept_shift_compare_stage(
            experiments=args.experiments, dry_run=args.dry_run, force=args.force,
        )
    except SystemExit as exc:
        return _fail("concept-shift-compare", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("concept-shift-compare", exc)
    if args.dry_run:
        print(json.dumps(result, indent=2, default=str))
    log.info(
        "[concept-shift-compare] tamamlandı: ran=%s resolved=%s",
        result.get("ran"), result.get("resolved_experiment_ids"),
    )
    return 0


def cmd_transfer_synthesis(args: argparse.Namespace) -> int:
    try:
        result = orch.run_multi_aoi_transfer_synthesis_stage(
            aois=args.aoi, dry_run=args.dry_run, force=args.force,
            output_root=args.output_root,
        )
    except SystemExit as exc:
        return _fail("transfer-synthesis", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("transfer-synthesis", exc)
    if args.dry_run:
        print(json.dumps(result, indent=2, default=str))
    log.info("[transfer-synthesis] tamamlandı: ran=%s canonical_set_id=%s", result.get("ran"), result.get("canonical_set_id"))
    return 0


def cmd_evia_signed_auc(args: argparse.Namespace) -> int:
    try:
        result = orch.run_evia_signed_auc_stage(
            experiment_id=args.experiment,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
            dry_run=args.dry_run,
            force=args.force,
        )
    except SystemExit as exc:
        return _fail("evia-signed-auc", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("evia-signed-auc", exc)
    if args.dry_run:
        print(json.dumps(result, indent=2, default=str))
    log.info(
        "[evia-signed-auc] tamamlandı: ran=%s experiment=%s support=%s",
        result.get("ran"), result.get("experiment_id"), result.get("support_counts"),
    )
    return 0


def cmd_transfer_decomposition(args: argparse.Namespace) -> int:
    try:
        result = orch.run_transfer_decomposition_stage(
            aois=args.aoi,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
            dry_run=args.dry_run,
            force=args.force,
        )
    except SystemExit as exc:
        return _fail("transfer-decomposition", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("transfer-decomposition", exc)
    if args.dry_run:
        print(json.dumps(result, indent=2, default=str))
    log.info(
        "[transfer-decomposition] tamamlandı: ran=%s canonical_set_id=%s rows=%s",
        result.get("ran"), result.get("canonical_set_id"), result.get("n_rows"),
    )
    return 0


def cmd_frozen_hash_inventory(args: argparse.Namespace) -> int:
    try:
        from src.provenance_audit import build_frozen_hash_inventory, diff_frozen_hash_inventory

        if args.phase == "diff":
            result = diff_frozen_hash_inventory()
            ok = result["passes"]
        else:
            result = build_frozen_hash_inventory(args.phase)
            ok = True
    except SystemExit as exc:
        return _fail("frozen-hash-inventory", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("frozen-hash-inventory", exc)
    print(json.dumps({k: v for k, v in result.items() if k != "entries"}, indent=2, default=str))
    log.info("[frozen-hash-inventory] tamamlandı: phase=%s", args.phase)
    if args.phase == "diff" and not ok:
        log.error(
            "[frozen-hash-inventory] BEKLENMEYEN degisiklik: %s",
            result.get("unexpected_change_count"),
        )
        return 1
    return 0


def cmd_manavgat_hash_audit(args: argparse.Namespace) -> int:
    try:
        from src.provenance_audit import run_manavgat_single_step8a_hash_audit

        result = run_manavgat_single_step8a_hash_audit(experiment_id=args.experiment)
    except SystemExit as exc:
        return _fail("manavgat-step8a-hash-audit", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("manavgat-step8a-hash-audit", exc)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2, default=str))
    log.info(
        "[manavgat-step8a-hash-audit] status=%s unique_hashes=%s",
        result["status"], result["unique_hash_count"],
    )
    return 0 if result["passes"] else 1


def cmd_old_new_deltas(args: argparse.Namespace) -> int:
    try:
        from src.old_new_metric_deltas import build as build_deltas

        result = build_deltas(aois=args.aoi)
    except SystemExit as exc:
        return _fail("old-new-deltas", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("old-new-deltas", exc)
    print(json.dumps(result, indent=2, default=str))
    log.info("[old-new-deltas] tamamlandı: rows=%s", result.get("row_count"))
    return 0


def cmd_burned_pattern_audit(args: argparse.Namespace) -> int:
    try:
        result = orch.run_burned_pattern_audit_stage(
            experiments=args.experiments,
            all_enabled=args.all_enabled,
            dry_run=args.dry_run,
            force=args.force,
        )
    except SystemExit as exc:
        return _fail("burned-pattern-audit", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("burned-pattern-audit", exc)
    if args.dry_run:
        print(json.dumps(result, indent=2, default=str))
    log.info(
        "[burned-pattern-audit] tamamlandı: ran=%s resolved=%s",
        result.get("ran"), result.get("resolved_experiment_ids"),
    )
    return 0


def cmd_marginal_aoa(args: argparse.Namespace) -> int:
    try:
        result = orch.run_marginal_aoa_stage(
            experiments=args.experiments,
            all_enabled=args.all_enabled,
            dry_run=args.dry_run,
            force=args.force,
        )
    except SystemExit as exc:
        return _fail("marginal-aoa", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("marginal-aoa", exc)
    if args.dry_run:
        print(json.dumps(result, indent=2, default=str))
    log.info(
        "[marginal-aoa] tamamlandı: ran=%s resolved=%s directed_pairs=%s",
        result.get("ran"), result.get("resolved_experiment_ids"),
        result.get("directed_pair_count"),
    )
    return 0


from src.marginal_aoa_completion import STAGES as _AOA_COMPLETION_STAGES
from src.few_shot_recovery import STAGES as _FEW_SHOT_RECOVERY_STAGES
from src.mugla_subsampling import STAGES as _MUGLA_SUBSAMPLING_STAGES
from src.window_closure_sensitivity import (
    DEFAULT_SHIFTS as _WINDOW_CLOSURE_DEFAULT_SHIFTS,
    STAGES as _WINDOW_CLOSURE_STAGES,
)


def cmd_few_shot_recovery(args: argparse.Namespace) -> int:
    try:
        result = orch.run_few_shot_recovery_stage(
            experiments=args.experiments,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
            dry_run=args.dry_run,
            resume=args.resume,
            force=args.force,
            output_root=args.output_root,
            experiments_root=args.experiments_root,
        )
    except SystemExit as exc:
        return _fail("few-shot-recovery", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("few-shot-recovery", exc)
    if args.dry_run:
        print(json.dumps(result, indent=2, default=str))
    log.info(
        "[few-shot-recovery] tamamlandı: ran=%s stages=%s analysis_id=%s",
        result.get("ran"), result.get("stages_executed"), result.get("analysis_id"),
    )
    return 0


def cmd_mugla_subsampling(args: argparse.Namespace) -> int:
    try:
        result = orch.run_mugla_subsampling_stage(
            experiments=args.experiments,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
            dry_run=args.dry_run,
            resume=args.resume,
            force=args.force,
            output_root=args.output_root,
            experiments_root=args.experiments_root,
        )
    except SystemExit as exc:
        return _fail("mugla-subsampling", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("mugla-subsampling", exc)
    if args.dry_run:
        print(json.dumps(result, indent=2, default=str))
    log.info(
        "[mugla-subsampling] tamamlandı: ran=%s stages=%s analysis_id=%s",
        result.get("ran"), result.get("stages_executed"), result.get("analysis_id"),
    )
    return 0


def cmd_marginal_aoa_completion(args: argparse.Namespace) -> int:
    try:
        result = orch.run_marginal_aoa_completion_stage(
            experiments=args.experiments,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
            dry_run=args.dry_run,
            resume=args.resume,
            allow_earth_engine=args.allow_earth_engine,
            output_root=args.output_root,
            experiments_root=args.experiments_root,
        )
    except SystemExit as exc:
        return _fail("marginal-aoa-completion", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("marginal-aoa-completion", exc)
    if args.dry_run:
        print(json.dumps(result, indent=2, default=str))
    log.info(
        "[marginal-aoa-completion] tamamlandı: ran=%s stages=%s analysis_id=%s",
        result.get("ran"), result.get("stages_executed"), result.get("analysis_id"),
    )
    return 0


def cmd_window_closure_sensitivity(args: argparse.Namespace) -> int:
    try:
        result = orch.run_window_closure_sensitivity_stage(
            experiment_id=args.experiment,
            shifts=args.shifts,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
            dry_run=args.dry_run,
            force=args.force,
            resume=args.resume,
        )
    except SystemExit as exc:
        return _fail("window-closure-sensitivity", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("window-closure-sensitivity", exc)
    if args.dry_run:
        print(json.dumps(result, indent=2, default=str))
    log.info(
        "[window-closure-sensitivity] tamamlandı: ran=%s experiment=%s shifts=%s",
        result.get("ran"), args.experiment, args.shifts,
    )
    if result.get("ran"):
        log.info(
            "[window-closure-sensitivity] stages=%s analysis_id=%s "
            "files_written=%s reused=%s",
            result.get("stages_run"), result.get("analysis_id"),
            result.get("files_written_count"), result.get("reused"),
        )
        if result.get("exported_variants") is not None:
            log.info(
                "[window-closure-sensitivity] predictor-export: processed=%s "
                "exported=%s reused=%s roles=%s rasters=%s "
                "canonical_export_attempted=%s",
                result.get("processed_variants"), result.get("exported_variants"),
                result.get("reused_variants"), result.get("logical_roles_produced"),
                result.get("predictor_rasters_produced"),
                result.get("canonical_export_attempted"),
            )
        if result.get("completed_variants") is not None:
            log.info(
                "[window-closure-sensitivity] local-downstream: processed=%s "
                "completed=%s reused=%s artifacts=%s step8a_datasets=%s "
                "canonical_downstream_attempted=%s common_cohort_created=%s "
                "model_fit=%s bootstrap_run=%s",
                result.get("processed_variants"), result.get("completed_variants"),
                result.get("reused_variants"),
                result.get("downstream_artifacts_produced"),
                result.get("step8a_datasets_produced"),
                result.get("canonical_downstream_attempted"),
                result.get("common_cohort_created"),
                result.get("model_fit"), result.get("bootstrap_run"),
            )
        if result.get("model_stage_metadata") is not None:
            metadata = result.get("model_stage_metadata") or {}
            log.info(
                "[window-closure-sensitivity] model: reused=%s evaluations=%s "
                "cohort_rows=%s prevalence=%s folds=%s "
                "fire_risk_model_fit=%s downscaling_model_fit=%s "
                "bootstrap_run=%s compare_run=%s",
                result.get("model_reused"),
                metadata.get("model_evaluation_count"),
                (metadata.get("common_cohort") or {}).get("final_common_cohort_rows"),
                (metadata.get("common_cohort") or {}).get("prevalence"),
                (metadata.get("shared_folds") or {}).get("fold_count"),
                result.get("fire_risk_model_fit"),
                result.get("downscaling_model_fit"),
                result.get("bootstrap_run"), result.get("compare_run"),
            )
        if result.get("compare_stage_metadata") is not None:
            metadata = result.get("compare_stage_metadata") or {}
            log.info(
                "[window-closure-sensitivity] compare: reused=%s "
                "point_metrics=%s contributions=%s bootstrap_rows=%s "
                "evidence=%s model_fit=%s bootstrap_run=%s compare_run=%s",
                result.get("compare_reused"),
                metadata.get("point_metric_row_count"),
                metadata.get("thermal_contribution_row_count"),
                metadata.get("bootstrap_summary_row_count"),
                metadata.get("evidence_status_counts"),
                result.get("model_fit"), result.get("bootstrap_run"),
                result.get("compare_run"),
            )
        for path in result.get("files_written") or []:
            log.info("[window-closure-sensitivity]   %s", path)
    return 0


def cmd_domain_classifier_audit(args: argparse.Namespace) -> int:
    try:
        result = orch.run_domain_classifier_audit_stage(
            experiments=args.experiments,
            all_enabled=args.all_enabled,
            dry_run=args.dry_run,
            force=args.force,
        )
    except SystemExit as exc:
        return _fail("domain-classifier-audit", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("domain-classifier-audit", exc)
    if args.dry_run:
        print(json.dumps(result, indent=2, default=str))
    log.info(
        "[domain-classifier-audit] tamamlandı: ran=%s resolved=%s",
        result.get("ran"), result.get("resolved_experiment_ids"),
    )
    return 0


def cmd_legacy(args: argparse.Namespace) -> int:
    if args.experiment not in orch.LEGACY_COMPATIBLE_EXPERIMENT_IDS:
        log.error(
            "'legacy' komutu yalnızca şu deneyleri destekler: %s (verilen: '%s').",
            orch.LEGACY_COMPATIBLE_EXPERIMENT_IDS, args.experiment,
        )
        print(
            f"\nHATA: 'legacy' komutu yalnızca {orch.LEGACY_COMPATIBLE_EXPERIMENT_IDS} "
            f"destekler (verilen: '{args.experiment}'). Diğer deneyler için 'experiment' "
            "alt-komutunu kullanın.\n"
        )
        return 1

    try:
        exp = get_active_experiment(args.experiment)
    except ValueError as exc:
        return _fail("legacy", exc)

    log.info(
        "[legacy] LEGACY (Google Drive tabanlı) Step1->Step8E pipeline seçildi: %s "
        "(role=%s, region_key=%s). Bu YENİ varsayılan yol DEĞİLDİR; yalnızca geçmiş "
        "Kozan reprodüksiyonu içindir.",
        args.experiment, exp["role"], exp["region_key"],
    )

    if args.dry_run:
        log.info("[legacy] [dry-run] predictor window: %s -> %s", exp["predictor_start_date"], exp["predictor_end_date"])
        log.info("[legacy] [dry-run] label window: %s -> %s", exp["label_start_date"], exp["label_end_date"])
        log.info("[legacy] [dry-run] baseline years: %s", exp["baseline_years"])
        log.info(
            "[legacy] [dry-run] output root (legacy, namespace'siz, PAYLAŞILAN): "
            "outputs/step5, outputs/step7a..e, outputs/step8a..e, outputs/validation/labels"
        )
        log.info(
            "[legacy] [dry-run] Hiçbir GEE/Drive/eğitim işlemi ÇALIŞTIRILMADI, hiçbir "
            "bilimsel çıktı dosyası YAZILMADI."
        )
        return 0

    log.warning(
        "[legacy] LEGACY Drive-tabanlı Step1-Step8E pipeline ÇALIŞTIRILIYOR "
        "(force=%s). Bu varsayılan yol DEĞİLDİR.", args.force,
    )
    try:
        orch.run_legacy_kozan_pipeline(force=args.force)
    except SystemExit as exc:
        return _fail("legacy", exc)
    except Exception as exc:  # noqa: BLE001
        return _fail("legacy", exc)
    return 0


# =============================================================================
# Argparse
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/main.py",
        description=(
            "satellite-thermal-digital-twin: deney-farkında (experiment-aware) "
            "CLI orkestratörü. Alt-komut verilmeden çalıştırma yalnızca bu yardım "
            "mesajını basar; hiçbir GEE/legacy iş akışını SESSİZCE başlatmaz."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG_EXAMPLES,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="{experiment,transfer,shift-audit,transfer-explore,self-cal-transfer,step10,step8-robustness,large-block-robustness,step8-big-block-robustness,concept-shift,concept-shift-compare,transfer-synthesis,evia-signed-auc,transfer-decomposition,frozen-hash-inventory,manavgat-step8a-hash-audit,old-new-deltas,burned-pattern-audit,marginal-aoa,marginal-aoa-completion,few-shot-recovery,window-closure-sensitivity,domain-classifier-audit,legacy}")

    # --- experiment ---
    p_exp = subparsers.add_parser(
        "experiment",
        help="Deney-farkında provenance ve seam-localization içeren aşama zinciri.",
        description=(
            "Bir deney (experiment_id) için gate/predictors/step7/seam-audit/step8 "
            "asamalarını (--from-stage/--to-stage araligi) sırayla çalıştırır. "
            "Her asama, mevcut namespaced runner'lara dispatch edilir; hiçbir "
            "bilimsel mantık burada yeniden uygulanmaz."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "örnekler:\n"
            "  python scripts/main.py experiment --experiment manavgat_2021 "
            "--from-stage predictors --to-stage step8 --predictor-mode local-only --force\n"
            "  python scripts/main.py experiment --experiment bejis_2022 "
            "--from-stage predictors --to-stage step8 --predictor-mode export --force\n"
            "  python scripts/main.py experiment --experiment manavgat_2021 "
            "--from-stage gate --to-stage step8 --predictor-mode local-only --dry-run\n"
            "  python scripts/main.py experiment --experiment mugla_2021 "
            "--from-stage seam-audit --to-stage seam-audit --predictor-mode local-only --dry-run\n"
        ),
    )
    p_exp.add_argument("--experiment", required=True, help="core/regions.py EXPERIMENTS kaydındaki experiment_id (örn. manavgat_2021, bejis_2022, kozan_2023).")
    p_exp.add_argument("--from-stage", required=True, choices=orch.STAGE_ORDER, help=f"Zincirin başlayacağı asama (sıra: {orch.STAGE_ORDER}).")
    p_exp.add_argument("--to-stage", required=True, choices=orch.STAGE_ORDER, help=f"Zincirin biteceği asama (sıra: {orch.STAGE_ORDER}).")
    p_exp.add_argument("--predictor-mode", required=True, choices=list(orch.PREDICTOR_MODES), help="'export': direkt/tiled GEE local download; 'local-only': yalnızca yerel Step5/Step5C (GEE'ye dokunmaz). Yalnızca 'predictors' asaması için anlamlıdır.")
    p_exp.add_argument("--export-labels", action="store_true", help="Yalnızca 'gate' asamasını etkiler: gate'ten önce raw MCD64A1 BurnDate export'unu çalıştırır.")
    p_exp.add_argument("--force", action="store_true", help="Her asamanın çıktıları zaten varsa üzerine yaz.")
    p_exp.add_argument("--dry-run", action="store_true", help="Hiçbir GEE/eğitim çalıştırma; deney planını + her asamanın kendi planlanan yollarını bas.")
    p_exp.add_argument("--seam-products", help="Yalnız seam-audit için virgülle ayrılmış product_key alt kümesi.")
    p_exp.add_argument("--seam-scales", help="Yalnız seam-audit için virgülle ayrılmış ölçekler: native,modeling_500m.")
    p_exp.set_defaults(func=cmd_experiment)

    p_exp.add_argument("--provenance-mode", choices=["metadata_only", "pixel_provenance"], default="metadata_only", help="pixel_provenance yalnız export planı üretir; GEE işi başlatmaz.")
    p_exp.add_argument("--manual-boundaries", action="append", default=[], help="Seam localization için ek tanısal GeoJSON; tekrarlanabilir.")
    # --- transfer ---
    p_transfer = subparsers.add_parser(
        "transfer",
        help="Cross-region Step9A-D transfer değerlendirmesi.",
        description=(
            "scripts/run_cross_region_transfer.py'yi reuse ederek Step9A "
            "(girdi denetimi) -> Step9B (iki yönlü transfer) -> Step9C "
            "(bootstrap) -> Step9D (final rapor) zincirini çalıştırır. "
            "Step9A-D burada yeniden uygulanmaz."
        ),
    )
    p_transfer.add_argument("--source", required=True, help="Kaynak (source) experiment_id.")
    p_transfer.add_argument("--target", required=True, help="Hedef (target) experiment_id.")
    p_transfer.add_argument("--reverse", action="store_true", help="Ters yönü (target->source) de açıkça teyit eder (Step9B zaten her iki yönü de hesaplar).")
    p_transfer.add_argument("--force", action="store_true", help="Her Step9 alt-asamasının çıktıları zaten varsa üzerine yaz.")
    p_transfer.add_argument("--dry-run", action="store_true", help="Hiçbir şey çalıştırma; planlanan yolları/feature setlerini bas.")
    p_transfer.set_defaults(func=cmd_transfer)

    # --- shift-audit ---
    p_shift = subparsers.add_parser(
        "shift-audit",
        help="Step9E post-hoc cross-region distribution-shift audit.",
        description=(
            "scripts/run_cross_region_shift_audit.py'yi reuse ederek Step9E'yi "
            "çalıştırır. Hiçbir modeli yeniden eğitmez, Step9A-D çıktılarını "
            "değiştirmez."
        ),
    )
    p_shift.add_argument("--source", required=True, help="Kaynak (source) experiment_id.")
    p_shift.add_argument("--target", required=True, help="Hedef (target) experiment_id.")
    p_shift.add_argument("--force", action="store_true", help="step9e çıktısı zaten varsa üzerine yaz.")
    p_shift.add_argument("--dry-run", action="store_true", help="Hiçbir hesaplama çalıştırma; planlanan yolları/feature setlerini bas.")
    p_shift.add_argument(
        "--report-only", action="store_true",
        help="Part A-F'yi YENİDEN HESAPLAMADAN, yalnızca safe_wording + Step9B "
        "provenance alanlarını + created_at'i günceller (mevcut bir Step9E "
        "çıktısı gerektirir; --force ile birlikte kullanılmalıdır).",
    )
    p_shift.set_defaults(func=cmd_shift_audit)

    # --- transfer-explore ---
    p_explore = subparsers.add_parser(
        "transfer-explore",
        help="Step9F kesifsel (exploratory) cross-region feature-representation deneyi.",
        description=(
            "scripts/run_exploratory_transfer_features.py'yi reuse ederek Step9F'i "
            "calistirir. Bu KESIFSEL, POST-HOC bir denedir -- tarafsiz dis validation "
            "DEGILDIR, Step9'un duzeltmesi DEGILDIR, transfer-safe feature setinin "
            "KANITI DEGILDIR. Step9A-E dosyalarini DEGISTIRMEZ."
        ),
    )
    p_explore.add_argument("--source", required=True, help="Kaynak (source) experiment_id.")
    p_explore.add_argument("--target", required=True, help="Hedef (target) experiment_id.")
    p_explore.add_argument("--reverse", action="store_true", help="Ters yonu de acikca teyit eder (Step9F zaten her iki yonu de hesaplar).")
    p_explore.add_argument("--force", action="store_true", help="step9f çıktısı zaten varsa üzerine yaz.")
    p_explore.add_argument("--dry-run", action="store_true", help="Hiçbir model fit/değerlendirme çalıştırma; yalnızca planlanan varyantları/rejimleri/yolları bas.")
    p_explore.add_argument("--bootstrap-replicates", type=int, default=1000, help="Hedef-bölge eşli spatial-block bootstrap replika sayısı (varsayılan: 1000).")
    p_explore.add_argument("--seed", type=int, default=None, help="Rastgele seed (varsayılan: Step9B ile AYNI sabit seed).")
    p_explore.set_defaults(func=cmd_transfer_explore)

    # --- self-cal-transfer ---
    p_selfcal = subparsers.add_parser(
        "self-cal-transfer",
        help="Step10 preregistered unsupervised self-calibrated cross-region transfer.",
        description=(
            "scripts/run_step10_self_calibrated_transfer.py'yi reuse ederek "
            "Step10'u (raw_source_only / regionwise_zscore / "
            "coral_after_regionwise_zscore) calistirir. Hedef etiketleri "
            "adaptasyon/fit/esik/kalibrasyon icin ASLA kullanilmaz. Step9A-F "
            "dosyalarini DEGISTIRMEZ."
        ),
    )
    p_selfcal.add_argument("--source", required=True, help="Kaynak (source) experiment_id.")
    p_selfcal.add_argument("--target", required=True, help="Hedef (target) experiment_id.")
    p_selfcal.add_argument("--reverse", action="store_true", help="Ters yönü de açıkça teyit eder (Step10 zaten her iki yönü de hesaplar).")
    p_selfcal.add_argument("--force", action="store_true", help="step10 çıktılarını üzerine yaz (on-kayıt/preregistration HARİÇ -- o asla değiştirilmez).")
    p_selfcal.add_argument("--dry-run", action="store_true", help="Hiçbir fit/adapt/predict/bootstrap çalıştırma; yalnızca dondurulmuş planı bas.")
    p_selfcal.add_argument("--report-only", action="store_true", help="Yalnızca frozen Step10 girdilerinden Step10D JSON/Markdown raporlarını yeniden üret.")
    p_selfcal.add_argument("--bootstrap-replicates", type=int, default=1000, help="Hedef-bölge eşli (N-yollu) spatial-block bootstrap replika sayısı (varsayılan: 1000).")
    p_selfcal.add_argument("--seed", type=int, default=42, help="Rastgele seed (varsayılan: STEP10_RANDOM_STATE=42).")
    p_selfcal.set_defaults(func=cmd_self_cal_transfer)

    # --- step8-robustness ---
    p_step8_robustness = subparsers.add_parser(
        "step8-robustness",
        help="Frozen within-region Step8 result for predefined 10/20-cell blocks.",
        description=(
            "Runs the exact frozen Manavgat/Bejis Step8 scientific plan at both "
            "predefined large-block scales. Original Step8A-E outputs are read-only."
        ),
    )
    p_step8_robustness.add_argument(
        "--experiments", nargs="+", required=True,
        help="Must be exactly: manavgat_2021 bejis_2022 (in this order).",
    )
    p_step8_robustness.add_argument(
        "--block-sizes-cells", nargs="+", required=True, type=int,
        help="Must be exactly: 10 20 (in this order).",
    )
    p_step8_robustness.add_argument(
        "--force", action="store_true",
        help="Overwrite downstream robustness outputs only; never rewrite preregistration or original Step8 outputs.",
    )
    p_step8_robustness.add_argument(
        "--dry-run", action="store_true",
        help="Resolve and print the frozen plan without fitting, bootstrapping, or writing scientific outputs.",
    )
    p_step8_robustness.set_defaults(func=cmd_step8_robustness)

    # --- step10 ---
    p_step10 = subparsers.add_parser(
        "step10",
        help="Step10 preregistered, target-label-blind self-calibrated cross-region transfer (COMPUTES metrics).",
        description=(
            "Scientific purpose: preregistered unsupervised self-calibrated "
            "cross-region transfer (raw_source_only / regionwise_zscore / "
            "coral_after_regionwise_zscore). "
            "Population: burnable_tree_shrub_grass (primary). "
            "Target labels: NOT used for adaptation, normalization, CORAL "
            "fitting, threshold selection, or calibration. "
            "Computes metrics (unless --report-only, which only regenerates the "
            "Step10D report from frozen inputs). Delegates to "
            "scripts/run_step10_self_calibrated_transfer.py; Step9A-F outputs "
            "are read-only. This is the same analysis as `self-cal-transfer`, "
            "exposed under the user-facing name `step10`."
        ),
    )
    p_step10.add_argument("--source", required=True, help="Kaynak (source) experiment_id (örn. manavgat_2021).")
    p_step10.add_argument("--target", required=True, help="Hedef (target) experiment_id (örn. bejis_2022).")
    p_step10.add_argument("--reverse", action="store_true", help="Ters yönü de açıkça teyit eder (Step10 zaten her iki yönü de hesaplar).")
    p_step10.add_argument("--force", action="store_true", help="step10 çıktılarını üzerine yaz (preregistration HARİÇ -- o asla değiştirilmez).")
    p_step10.add_argument("--dry-run", action="store_true", help="Hiçbir fit/adapt/predict/bootstrap çalıştırma; yalnızca dondurulmuş planı bas.")
    p_step10.add_argument("--report-only", action="store_true", help="Yalnızca frozen Step10 girdilerinden Step10D JSON/Markdown raporlarını yeniden üret (metrik HESAPLAMAZ).")
    p_step10.add_argument("--bootstrap-replicates", type=int, default=1000, help="Hedef-bölge eşli spatial-block bootstrap replika sayısı (varsayılan: 1000).")
    p_step10.add_argument("--seed", type=int, default=42, help="Rastgele seed (varsayılan: STEP10_RANDOM_STATE=42).")
    p_step10.set_defaults(func=cmd_step10)

    # --- frozen CORAL lambda sensitivity ---
    p_coral_lambda = subparsers.add_parser(
        "coral-lambda-sensitivity",
        help="Frozen local sensitivity of Step10 CORAL additive covariance ridge.",
    )
    p_coral_lambda.add_argument("--from-stage", choices=("plan", "fit", "bootstrap", "summarize"), default="plan")
    p_coral_lambda.add_argument("--to-stage", choices=("plan", "fit", "bootstrap", "summarize"), default="summarize")
    p_coral_lambda.add_argument("--dry-run", action="store_true")
    p_coral_lambda.add_argument("--resume", action="store_true")
    p_coral_lambda.add_argument("--force", action="store_true")
    p_coral_lambda.add_argument("--output-root")
    p_coral_lambda.add_argument("--experiments-root")
    p_coral_lambda.set_defaults(func=cmd_coral_lambda_sensitivity)

    # --- large-block-robustness ---
    p_large_block = subparsers.add_parser(
        "large-block-robustness",
        help="Formal Step8B primary-population (all_valid) large-block robustness (COMPUTES metrics; gated).",
        description=(
            "Scientific purpose: test whether the formal within-region Step8B "
            "thermal contribution survives predefined larger spatial validation "
            "blocks (10 cells ~5 km, 20 cells ~10 km). "
            "Population: all_valid (the FORMAL Step8B primary population), "
            "distinct from the natural-vegetation burnable_tree_shrub_grass "
            "sensitivity analysis. "
            "Target labels: standard supervised within-region CV (no cross-region "
            "adaptation). Computes metrics. "
            "STEP8B_SPATIAL_BLOCK_SIZE_CELLS stays 2 (the frozen ~1 km reference); "
            "10/20-cell block sizes are passed at RUNTIME. The 2-cell equivalence "
            "gate is REQUIRED before any 10/20-cell fit: without "
            "--run-large-block-fit the run stops after the gate. Original Step8 "
            "outputs are never overwritten. Delegates to "
            "scripts/run_step8_large_block_robustness_primary_all_valid.py."
        ),
    )
    p_large_block.add_argument("--run-large-block-fit", action="store_true", help="Yalnızca 2-cell equivalence gate incelendikten SONRA: 10/20-cell bilimsel koşumu fit et. Bu bayrak olmadan koşum gate'ten sonra durur.")
    p_large_block.add_argument("--force", action="store_true", help="Yalnızca downstream robustness çıktılarını üzerine yaz; preregistration'ı veya orijinal Step8 çıktılarını asla yeniden yazma.")
    p_large_block.add_argument("--dry-run", action="store_true", help="Dondurulmuş planı çöz ve bas; fit/bootstrap yapma, bilimsel çıktı yazma.")
    p_large_block.set_defaults(func=cmd_large_block_robustness)

    # --- step8-big-block-robustness ---
    p_big_block = subparsers.add_parser(
        "step8-big-block-robustness",
        help="Single-experiment Step8 big-spatial-block robustness (10/20 cells; COMPUTES metrics).",
        description=(
            "Scientific purpose: test whether a single experiment's existing "
            "within-region Step8B/Step8C thermal contribution (population "
            "burnable_tree_shrub_grass) survives predefined larger spatial "
            "validation blocks (default 10 cells ~5 km, 20 cells ~10 km), "
            "compared read-only against the existing small-block (2-cell) "
            "result. Reuses step8b.train_population/add_spatial_block_id "
            "unmodified (no independent OOF algorithm). Never regenerates "
            "Step8A or overwrites existing Step8B/C/E outputs -- writes only "
            "under outputs/experiments/<experiment>/robustness/step8_big_blocks/. "
            "Takes --experiment as a plain argument; no AOI is hard-coded. "
            "Delegates to scripts/run_step8_big_block_robustness.py."
        ),
    )
    p_big_block.add_argument("--experiment", required=True, help="core/regions.py EXPERIMENTS kaydındaki experiment_id.")
    p_big_block.add_argument("--block-sizes", nargs="+", type=int, default=[10, 20], help="Büyük blok boyutları (500m hücre cinsinden; varsayılan: 10 20).")
    p_big_block.add_argument("--force", action="store_true", help="Yalnızca downstream robustness çıktılarını üzerine yaz; preregistration'ı veya orijinal Step8 çıktılarını asla yeniden yazma.")
    p_big_block.add_argument("--dry-run", action="store_true", help="Dondurulmuş planı çöz ve bas; fit/bootstrap yapma, bilimsel çıktı yazma.")
    p_big_block.add_argument(
        "--output-root", default=None,
        help=(
            "Sürümlenmiş çıktı namespace'i (örn. outputs/experiments/manavgat_2021/"
            "robustness/step8_big_blocks_v2). Yalnızca NEREYE yazıldığını değiştirir; "
            "population, blok boyutları, bootstrap/seed ve model sözleşmesi AYNI kalır. "
            "Verilmezse varsayılan .../robustness/step8_big_blocks kullanılır ve mevcut "
            "v1/legacy artefactlara asla yazılmaz."
        ),
    )
    p_big_block.add_argument(
        "--regenerate-reports-only", action="store_true",
        help=(
            "Rapor-only mod: yalnızca önceki tam çalıştırmanın dondurulmuş "
            "JSON/Parquet çıktılarını (step8b_metrics.json, "
            "bootstrap_summary.json, fold_assignments.parquet) okur ve "
            "JSON/Markdown/CSV/manifest raporlarını güncel şemayla yeniden "
            "üretir. Model fit, fold oluşturma, tahmin üretimi veya "
            "bootstrap örnekleme ASLA çalıştırılmaz; oof_predictions.parquet, "
            "fold_assignments.parquet ve bootstrap_replicates.parquet asla "
            "yazılmaz."
        ),
    )
    p_big_block.set_defaults(func=cmd_step8_big_block_robustness)

    # --- concept-shift ---
    p_concept = subparsers.add_parser(
        "concept-shift",
        help="Step9G univariate feature-AUC direction-reversal concept/relationship-shift diagnostic.",
        description=(
            "Scientific purpose: per-feature univariate burned ROC-AUC "
            "direction-reversal diagnostic between a selected source/target pair, "
            "as evidence consistent with residual concept/relationship shift "
            "(not causal proof; not the only transfer-failure source). "
            "Population: burnable_tree_shrub_grass (the primary cross-region "
            "transfer population). "
            "Target labels: burned is the ranking target for the univariate AUC; "
            "labels are NOT used for any adaptation/normalization/inversion. "
            "Default (no flag): runs the Step9G numeric analysis -- COMPUTES "
            "metrics. With --integration-only: REPORT-ONLY canonical "
            "integration-v2 that reuses frozen Step9G numbers verbatim and only "
            "corrects Step9E/9F/10 integration (recomputes nothing). "
            "Delegates to the Step9G module and its integration-v2 module."
        ),
    )
    p_concept.add_argument("--source", required=True, help="Source experiment ID.")
    p_concept.add_argument("--target", required=True, help="Target experiment ID.")
    p_concept.add_argument(
        "--integration-only", action="store_true",
        help=(
            "Bilimsel Step9G sayılarını yeniden hesaplama; yalnızca CANONICAL "
            "integration-v2 raporunu üret (frozen Step9G çıktılarını birebir "
            "kullanır). Çıktı: outputs/diagnostics/"
            "step9g_univariate_feature_auc_direction_reversal_integration_v2/"
            "<source>__<target>/."
        ),
    )
    p_concept.add_argument(
        "--report-revision", action="store_true",
        help=(
            "REPORT-ONLY: revise the existing canonical Step9G v1 final "
            "report IN PLACE (step9g_final_report.json/.md) to correct two "
            "semantic issues -- integrated_interpretation repeating one "
            "generic sentence for every row, and "
            "thermal_features_consistent_with_step9e wrongly able to "
            "include elevation_mean -- without recomputing any numerical "
            "result. analysis_id is preserved verbatim. Mutually exclusive "
            "in effect with --integration-only (which corrects a different, "
            "separate Step9E/9F/10 integration namespace)."
        ),
    )
    p_concept.add_argument("--force", action="store_true", help="İlgili çıktılar zaten varsa üzerine yaz (orijinal frozen Step9G sayısal çıktıları HARİÇ -- onlar değişmez).")
    p_concept.add_argument("--dry-run", action="store_true", help="Hiçbir AUC/bootstrap hesaplama, hiçbir dosya yazma; yalnızca planı bas.")
    p_concept.add_argument(
        "--output-root", default=None,
        help=(
            "Sürümlenmiş çıktı namespace'i. Step9G preregistration'ı IMMUTABLE'dır "
            "ve tükettiği Step8A girdilerinin hash'ine bağlıdır; girdiler yeniden "
            "üretildiğinde eski preregistration kaçınılmaz olarak uyuşmaz. Bu bayrak "
            "YENİ bir dizine yazar; eski dizin ve preregistration DEĞİŞTİRİLMEZ, "
            "silinmez. Bilimsel ayarların hiçbiri değişmez."
        ),
    )
    p_concept.set_defaults(func=cmd_concept_shift)

    # --- concept-shift-compare ---
    p_concept_compare = subparsers.add_parser(
        "concept-shift-compare",
        help="Generic, REPORT-ONLY multi-experiment Step9G univariate-AUC comparison.",
        description=(
            "Scientific purpose: synthesize a side-by-side comparison table "
            "from already-frozen canonical Step9G univariate feature-AUC "
            "direction-reversal pair reports (outputs/diagnostics/"
            "step9g_univariate_feature_auc_direction_reversal/<pair_id>/"
            "step9g_final_report.json), for a caller-supplied explicit set "
            "of experiment IDs (--experiments). REPORT-ONLY -- never reruns "
            "concept-shift, never recomputes an AUC/CI/bootstrap draw/"
            "reversal classification. Requires the discovered reports to use "
            "population burnable_tree_shrub_grass, the fixed 9-feature set, "
            "10-cell (~5 km) spatial blocks, and the frozen bootstrap "
            "configuration; cross-validates repeated region-feature results "
            "across reports at 1e-12 tolerance and fails clearly on "
            "disagreement. No experiment ID, AOI name, or pair is "
            "hard-coded: everything is derived from --experiments. Writes "
            "JSON/2 CSV/Markdown/manifest under outputs/diagnostics/"
            "step9g_univariate_feature_auc_direction_reversal/comparison/"
            "<sorted_experiment_ids_joined>/. Delegates to "
            "src/step9g_multi_aoi_comparison/."
        ),
    )
    p_concept_compare.add_argument(
        "--experiments", nargs="+", required=True,
        help="Explicit experiment_id list (core/regions.py registry entries), e.g. --experiments <experiment_id_1> <experiment_id_2> <experiment_id_3>.",
    )
    p_concept_compare.add_argument("--force", action="store_true", help="Overwrite an existing comparison output produced by a different analysis_id.")
    p_concept_compare.add_argument("--dry-run", action="store_true", help="Resolve experiments, discover pair reports, and validate contracts; no file write, no recomputation.")
    p_concept_compare.set_defaults(func=cmd_concept_shift_compare)

    # --- transfer-synthesis ---
    p_transfer_synthesis = subparsers.add_parser(
        "transfer-synthesis",
        help="Generic, REPORT-ONLY multi-AOI (2-5) cross-region transfer synthesis.",
        description=(
            "Scientific purpose: synthesize a generic cross-AOI report from "
            "already-frozen Step8/Step9B-G/Step10 outputs for a caller-supplied "
            "set of 2-5 AOI experiment IDs (--aoi, repeatable). REPORT-ONLY -- "
            "never runs modeling, adaptation, prediction, bootstrap, or any "
            "scientific analysis; only reads frozen JSON/CSV/parquet outputs "
            "and writes a JSON/Markdown report plus 3 CSV tables and a "
            "manifest under outputs/diagnostics/multi_aoi_transfer_synthesis/"
            "<canonical_set_id>/ (or --output-root, if supplied). No AOI name, "
            "path, or pair is hard-coded: directions/pairs are generated "
            "programmatically from the supplied --aoi list. "
            "Delegates to src/multi_aoi_transfer_synthesis/."
        ),
    )
    p_transfer_synthesis.add_argument(
        "--aoi", action="append", required=True, dest="aoi",
        help="AOI experiment ID (core/regions.py EXPERIMENTS kaydı). En az 2, en fazla 5 kez tekrarlanabilir.",
    )
    p_transfer_synthesis.add_argument(
        "--output-root", default=None,
        help="Varsayılan outputs/diagnostics/multi_aoi_transfer_synthesis/ kökü yerine kullanılacak kök dizin.",
    )
    p_transfer_synthesis.add_argument("--force", action="store_true", help="Çıktı dizini zaten doluysa üzerine yaz.")
    p_transfer_synthesis.add_argument("--dry-run", action="store_true", help="Hiçbir dosya yazma; yalnızca çözümlenen plan/girdileri (ve varsa eksik/çelişkili girdileri) bas.")
    p_transfer_synthesis.set_defaults(func=cmd_transfer_synthesis)

    # --- evia-signed-auc ---
    p_signed_auc = subparsers.add_parser(
        "evia-signed-auc",
        help="REPORT-ONLY signed-AUC (2*AUC-1) spatial-block bootstrap for one experiment.",
        description=(
            "Scientific purpose: express each predictor's RAW univariate ROC-AUC "
            "on the signed effect scale signed_auc = 2*AUC-1 and report it with "
            "~5 km spatial-block bootstrap uncertainty, over the canonical "
            "primary population burnable_tree_shrub_grass. The AUC is NEVER "
            "inverted, so a feature whose high values rank UNBURNED cells higher "
            "keeps a negative signed value. The CI is taken from the "
            "REPLICATE-LEVEL signed distribution. REPORT-ONLY -- reads the frozen "
            "Step8A dataset and the frozen Step9G univariate tables, and never "
            "runs modeling, adaptation, prediction or Step10. Writes under "
            "outputs/diagnostics/evia_signed_auc_bootstrap/<experiment_id>/. "
            "Delegates to src/evia_signed_auc_bootstrap.py."
        ),
    )
    p_signed_auc.add_argument(
        "--experiment", default="evia_2021_extended",
        help="Kanonik deney kimliği (varsayılan: evia_2021_extended).",
    )
    p_signed_auc.add_argument(
        "--bootstrap-replicates", type=int, default=1000,
        help="Ön-kayıtlı replicate sayısı; sözleşmeden farklı bir değer reddedilir.",
    )
    p_signed_auc.add_argument(
        "--seed", type=int, default=42,
        help="Ön-kayıtlı seed; sözleşmeden farklı bir değer reddedilir.",
    )
    p_signed_auc.add_argument("--force", action="store_true", help="Çıktı dizini zaten doluysa üzerine yaz.")
    p_signed_auc.add_argument("--dry-run", action="store_true", help="Hiçbir dosya yazma; yalnızca çözümlenen plan/girdileri bas.")
    p_signed_auc.set_defaults(func=cmd_evia_signed_auc)

    # --- transfer-decomposition ---
    p_decomp = subparsers.add_parser(
        "transfer-decomposition",
        help="REPORT-ONLY multi-AOI transfer gap/recovery decomposition from frozen Step10.",
        description=(
            "Scientific purpose: recompute the transfer decomposition "
            "(raw_gap, adaptation_effect, remaining_gap, recovered_fraction, "
            "remaining_fraction) for every ordered direction x model family x "
            "adaptation method, from ALREADY FROZEN Step8B/Step9B/Step10 "
            "artefacts. Negative recovery is reported as such and NEVER clipped "
            "to zero; fractions are suppressed only when raw_gap <= 0. "
            "REPORT-ONLY -- never re-runs models, adaptation, predictions or the "
            "Step10 bootstrap. Writes under "
            "outputs/diagnostics/four_aoi_transfer_decomposition/"
            "<canonical_set_id>/. Delegates to src/transfer_decomposition.py."
        ),
    )
    p_decomp.add_argument(
        "--aoi", action="append", required=True, dest="aoi",
        help="AOI experiment ID (core/regions.py EXPERIMENTS kaydı). Tekrarlanabilir.",
    )
    p_decomp.add_argument(
        "--bootstrap-replicates", type=int, default=1000,
        help="Frozen Step10 replicate sayısı beklentisi; uyuşmazlık raporlanır.",
    )
    p_decomp.add_argument(
        "--seed", type=int, default=42,
        help="Frozen Step10 seed beklentisi; uyuşmazlık raporlanır.",
    )
    p_decomp.add_argument("--force", action="store_true", help="Çıktı dizini zaten doluysa üzerine yaz.")
    p_decomp.add_argument("--dry-run", action="store_true", help="Hiçbir dosya yazma; yalnızca çözümlenen plan/girdileri bas.")
    p_decomp.set_defaults(func=cmd_transfer_decomposition)

    # --- frozen-hash-inventory ---
    p_inventory = subparsers.add_parser(
        "frozen-hash-inventory",
        help="READ-ONLY content-hash inventory of frozen scientific artefacts (before/after/diff).",
        description=(
            "Scientific purpose: record the SHA-256 of every frozen Step8/Step9/"
            "Step9E/Step9G/Step10 and synthesis artefact so that any change -- "
            "including one caused by running the test suite -- is detectable. "
            "Run with --phase before, then again with --phase after, then "
            "--phase diff. The diff separates expected changes (Manavgat-dependent "
            "reruns) from unexpected ones (anything in the Bejis/Mugla/Evia "
            "triangle). --phase diff exits non-zero if any unexpected change or "
            "disappearance is found. Writes only under "
            "outputs/diagnostics/advisor_followup_provenance/."
        ),
    )
    p_inventory.add_argument(
        "--phase", required=True, choices=["before", "after", "diff"],
        help="'before'/'after' yazar; 'diff' ikisini karsilastirir ve beklenmeyen degisiklikte 1 doner.",
    )
    p_inventory.set_defaults(func=cmd_frozen_hash_inventory)

    # --- manavgat-step8a-hash-audit ---
    p_hash_audit = subparsers.add_parser(
        "manavgat-step8a-hash-audit",
        help="READ-ONLY audit: every Manavgat consumer must record ONE Step8A content hash.",
        description=(
            "Scientific purpose: prove that every artefact consuming Manavgat's "
            "Step8A dataset (Step8B-E, versioned large-block robustness, and the "
            "Step9A-D/9E/9G/Step10 chains of all three Manavgat pairs) was built "
            "against a single Step8A content hash. The expected hash is COMPUTED "
            "from the canonical file on disk, never hard-coded. Artefacts that do "
            "not record Step8A provenance are reported as such and force "
            "'incomplete_provenance' -- the hash is never inferred for them. "
            "Exits non-zero unless the unique hash count is exactly 1."
        ),
    )
    p_hash_audit.add_argument(
        "--experiment", default="manavgat_2021",
        help="Denetlenecek deney kimligi (varsayilan: manavgat_2021).",
    )
    p_hash_audit.set_defaults(func=cmd_manavgat_hash_audit)

    # --- old-new-deltas ---
    p_deltas = subparsers.add_parser(
        "old-new-deltas",
        help="READ-ONLY old-vs-new transfer metric delta report around the Manavgat repair.",
        description=(
            "Scientific purpose: compare the preserved pre-repair four-AOI "
            "decomposition against the regenerated one, per direction x model "
            "family x adaptation method x metric. When an old artefact was not "
            "preserved the row is emitted with comparison_available=false and an "
            "explicit unavailable_reason -- values are never invented. Brier is "
            "reported as unavailable because it is not preregistered for the "
            "Step10 adapted models. Writes only under "
            "outputs/diagnostics/advisor_followup_provenance/."
        ),
    )
    p_deltas.add_argument(
        "--aoi", action="append", required=True, dest="aoi",
        help="AOI experiment ID (tekrarlanabilir); yeniden uretilmis decomposition ile ayni kume olmali.",
    )
    p_deltas.set_defaults(func=cmd_old_new_deltas)

    # --- burned-pattern-audit ---
    p_burned_pattern = subparsers.add_parser(
        "burned-pattern-audit",
        help="Generic multi-experiment burned-area spatial-structure and descriptive comparison audit.",
        description=(
            "Scientific purpose: for each resolved experiment, descriptively "
            "reports the number of 8-neighbour spatially connected "
            "burned-cell components, the component/patch cell-count size "
            "distribution, burned-cell elevation distribution, and "
            "burned-cell land-cover composition (reusing the canonical ESA "
            "WorldCover mapping and the existing burnable_tree_shrub_grass "
            "sensitivity-population mask). Connected components are a "
            "spatial fragmentation proxy, NOT a count of independent "
            "wildfire events. Read-only against Step8A; never fits a model, "
            "computes transfer performance, or reruns Step9/Step9G/Step10. "
            "No experiment ID is hard-coded: --experiments takes an "
            "arbitrary explicit list, --all-enabled resolves every enabled, "
            "non-legacy experiment_id from the core/regions.py registry "
            "with a canonical Step8A dataset. Delegates to "
            "scripts/run_burned_pattern_audit.py / src/burned_pattern_audit.py."
        ),
    )
    p_burned_pattern_selection = p_burned_pattern.add_mutually_exclusive_group(required=True)
    p_burned_pattern_selection.add_argument(
        "--experiments", nargs="+", default=None,
        help="Explicit experiment_id list (core/regions.py registry entries), e.g. --experiments <experiment_id_1> <experiment_id_2>.",
    )
    p_burned_pattern_selection.add_argument(
        "--all-enabled", action="store_true",
        help="Resolve every enabled, non-legacy core/regions.py experiment_id that has a canonical Step8A dataset.",
    )
    p_burned_pattern.add_argument("--force", action="store_true", help="Overwrite existing outputs already produced by a different analysis_id.")
    p_burned_pattern.add_argument("--dry-run", action="store_true", help="Resolve experiments/inputs and print the plan; no component computation, no files written.")
    p_burned_pattern.set_defaults(func=cmd_burned_pattern_audit)

    # --- marginal-aoa ---
    p_marginal_aoa = subparsers.add_parser(
        "marginal-aoa",
        help="Generic directed, label-blind marginal Area-of-Applicability analysis.",
        description=(
            "Scientific purpose: for every ORDERED source->target pair among "
            "the resolved experiments, report whether each target predictor "
            "value falls inside the marginal predictor support actually "
            "OBSERVED in the source AOI (inclusive observed min/max for "
            "numeric predictors; observed non-missing levels for categorical "
            "ones). 'Marginal' is literal: each predictor is evaluated on its "
            "own; this is NOT multivariate joint support and does not assess "
            "predictor correlation structure. "
            "LABEL-BLIND: every parquet read uses an explicit allow-list of "
            "predictor, grid, population-mask and eligibility columns -- "
            "'burned' and every other label column are never loaded, so the "
            "result is byte-identical under any change to the target labels. "
            "Fits no model, produces no prediction, runs no adaptation, CV or "
            "bootstrap; read-only against Step8A. Direction matters: "
            "source__target and target__source are distinct analyses with "
            "distinct output namespaces. Results are DESCRIPTIVE -- being "
            "inside the source range does not guarantee transfer success, and "
            "being outside it does not prove it caused transfer failure; no "
            "statistical significance is claimed. No experiment ID is "
            "hard-coded. Delegates to "
            "scripts/run_marginal_area_of_applicability.py / "
            "src/marginal_area_of_applicability.py."
        ),
    )
    p_marginal_aoa_selection = p_marginal_aoa.add_mutually_exclusive_group(required=True)
    p_marginal_aoa_selection.add_argument(
        "--experiments", nargs="+", default=None,
        help="Explicit experiment_id list (core/regions.py registry entries).",
    )
    p_marginal_aoa_selection.add_argument(
        "--all-enabled", action="store_true",
        help="Resolve every enabled, non-legacy core/regions.py experiment_id that has a canonical Step8A dataset.",
    )
    p_marginal_aoa.add_argument("--force", action="store_true", help="Overwrite existing marginal AoA outputs produced by a different analysis_id (this namespace only).")
    p_marginal_aoa.add_argument("--dry-run", action="store_true", help="Resolve inputs/schema, list every directed pair and planned output path; no analysis, no files written.")
    p_marginal_aoa.set_defaults(func=cmd_marginal_aoa)

    # --- marginal-aoa-completion ---
    p_aoa_completion = subparsers.add_parser(
        "marginal-aoa-completion",
        help="Marginal AoA completion: weighted dissimilarity + climatic and geographic distance.",
        description=(
            "Scientific purpose: complete the marginal Area-of-Applicability "
            "request with the three components marginal_aoa.v1 does not "
            "provide -- an importance-weighted predictor-space dissimilarity "
            "(DIRECTED), a climatic distance (SYMMETRIC) and a geographic "
            "distance (SYMMETRIC) -- over exactly 12 directed pairs. "
            "DIAGNOSTIC CLASS: target-label-blind, source-model-informed. "
            "Feature weights come from a Step8B RandomForest fitted on the "
            "SOURCE label, so this is NOT described as label-blind; the target "
            "label, target burn date and every transfer metric stay outside "
            "the computation. The DI denominator is the MEAN PAIRWISE weighted "
            "distance over all distinct source-reference pairs (not a "
            "nearest-neighbour mean), and the operative threshold is the upper "
            "whisker min(max(training_DI), Q3+1.5*IQR); the 0.95 quantile is "
            "reported as a secondary sensitivity value and never classifies. "
            "Fits no model, runs no bootstrap and produces no composite scalar "
            "index. Only the 'climate-export' stage may contact Earth Engine. "
            "Writes exclusively under "
            "outputs/diagnostics/marginal_aoa_completion/<analysis_id>/. "
            "Delegates to scripts/run_marginal_aoa_completion.py / "
            "src/marginal_aoa_completion.py."
        ),
    )
    p_aoa_completion.add_argument(
        "--experiments", nargs="+", default=None,
        help="Experiment IDs; defaults to the four canonical AOIs of this frozen analysis.",
    )
    p_aoa_completion.add_argument(
        "--from-stage", default="plan", choices=list(_AOA_COMPLETION_STAGES),
        help=f"First stage to run. Stage order: {list(_AOA_COMPLETION_STAGES)}.",
    )
    p_aoa_completion.add_argument(
        "--to-stage", default="compare", choices=list(_AOA_COMPLETION_STAGES),
        help="Last stage to run (inclusive).",
    )
    p_aoa_completion.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Print the exact plan and prerequisite state. Creates no directory, "
            "writes no file, contacts no Earth Engine, fits no model and "
            "computes no distance."
        ),
    )
    p_aoa_completion.add_argument(
        "--resume", action="store_true",
        help="Verify and reuse an existing namespace instead of refusing to overwrite it.",
    )
    p_aoa_completion.add_argument(
        "--allow-earth-engine", action="store_true",
        help=(
            "Authorise LIVE Earth Engine queries and the climate export. The "
            "'climate-export' stage is never started implicitly; without this "
            "flag a range that reaches it fails closed."
        ),
    )
    p_aoa_completion.add_argument(
        "--output-root", default=None,
        help="Alternative outputs/ root (dependency injection).",
    )
    p_aoa_completion.add_argument(
        "--experiments-root", default=None,
        help="Alternative outputs/experiments/ root (dependency injection).",
    )
    p_aoa_completion.set_defaults(func=cmd_marginal_aoa_completion)

    # --- few-shot-recovery ---
    p_few_shot = subparsers.add_parser(
        "few-shot-recovery",
        help="Few-shot recovery curve: raw transfer -> k labeled target blocks -> ceiling.",
        description=(
            "Scientific purpose: quantify how much of the gap between zero-shot "
            "raw transfer and the target-only within-region ceiling is recovered "
            "when a limited number of labeled target spatial blocks is supplied, "
            "over exactly 6 directed pairs of the three primary AOIs. "
            "DIAGNOSTIC CLASS: target-label-supervised few-shot adaptation "
            "sensitivity. Target labels ARE used, deliberately -- this is NOT an "
            "operational deployment claim, NOT active learning, NOT a causal "
            "decomposition and NOT target-label-free adaptation (that is Step10). "
            "Population burnable_tree_shrub_grass; 10-cell (~5 km) evaluation "
            "blocks via the canonical large-block utility; strict 5-fold "
            "StratifiedGroupKFold; budgets 0,1,2,4,8,16,32 with 10 deterministic "
            "repeats for every k>0, nested so each budget is a prefix of the same "
            "ordering. Evaluation blocks never enter any training frame. Recovery "
            "fractions are signed and UNCLIPPED, Brier is oriented by negation, "
            "and the only interval reported is a repeat-based SELECTION INTERVAL "
            "-- no bootstrap, no p-value, no significance claim. "
            "evia_2021_extended is excluded by design. Contacts no Earth Engine "
            "and writes exclusively under "
            "outputs/diagnostics/few_shot_recovery/<analysis_id>/. "
            "Delegates to scripts/run_few_shot_recovery.py / "
            "src/few_shot_recovery.py."
        ),
    )
    p_few_shot.add_argument(
        "--experiments", nargs="+", default=None,
        help="Experiment IDs; defaults to the three canonical AOIs of this frozen analysis.",
    )
    p_few_shot.add_argument(
        "--from-stage", default="plan", choices=list(_FEW_SHOT_RECOVERY_STAGES),
        help=f"First stage to run. Stage order: {list(_FEW_SHOT_RECOVERY_STAGES)}.",
    )
    p_few_shot.add_argument(
        "--to-stage", default="summarize", choices=list(_FEW_SHOT_RECOVERY_STAGES),
        help="Last stage to run (inclusive).",
    )
    p_few_shot.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Print the plan, the resolved input hashes, the expected fit budget and "
            "every planned output path. Creates no directory, writes no file and "
            "fits no model."
        ),
    )
    p_few_shot.add_argument(
        "--resume", action="store_true",
        help=(
            "Verify and reuse complete, hash-bound stages and direction partitions. "
            "A partially written direction is never accepted."
        ),
    )
    p_few_shot.add_argument(
        "--force", action="store_true",
        help="Quarantine an existing namespace under _quarantine/ and start fresh. Never deletes.",
    )
    p_few_shot.add_argument(
        "--output-root", default=None,
        help="Alternative outputs/ root (dependency injection).",
    )
    p_few_shot.add_argument(
        "--experiments-root", default=None,
        help="Alternative outputs/experiments/ root (dependency injection).",
    )
    p_few_shot.set_defaults(func=cmd_few_shot_recovery)

    # --- mugla-subsampling ---
    p_mugla_subsampling = subparsers.add_parser(
        "mugla-subsampling",
        help="Mugla size-matched subsampling sensitivity of within/source/target results.",
        description=(
            "Scientific purpose: when the Mugla modeling population (41,730 "
            "primary cells) is reduced to exactly Manavgat's cell count (20,511), "
            "how do (a) within-Mugla 5-fold spatial OOF performance, (b) "
            "Mugla -> Manavgat/Bejis raw transfer with Mugla as SOURCE, and (c) "
            "Manavgat/Bejis -> Mugla evaluation with Mugla as TARGET move "
            "relative to their full-population references? "
            "DIAGNOSTIC CLASS: population-size-matched subsampling sensitivity. "
            "This is a TOTAL-SAMPLE-SIZE sensitivity analysis and nothing else: "
            "prevalence is preserved within integer rounding limits, the Mugla "
            "positive count is NOT equalised to Manavgat's, no predictor or "
            "label structure is altered, and no causal decomposition or "
            "operational deployment claim is made. "
            "Population burnable_tree_shrub_grass; 20 deterministic repeats of "
            "20,511 unique cells drawn without replacement from 10-cell (~5 km) "
            "block x label strata under an integer-exact Hamilton "
            "largest-remainder allocation, so the allocation and the per-fold "
            "composition are identical in every repeat. The 5-fold spatial fold "
            "mapping is INHERITED unchanged from the frozen full-Mugla 10-cell "
            "artifact and is never re-optimised per repeat. One Mugla-as-source "
            "fit serves both targets and the Mugla-as-target arm reuses frozen "
            "raw-transfer predictions, so the whole analysis is 240 unique fits. "
            "Brier is oriented by negation, the only range reported is a "
            "repeat-based SUBSAMPLING INTERVAL, and no bootstrap or probability "
            "statement of any kind is produced. evia_2021 and "
            "evia_2021_extended are excluded by design. Contacts no Earth Engine "
            "and writes exclusively under "
            "outputs/diagnostics/mugla_subsampling/<analysis_id>/. "
            "Delegates to scripts/run_mugla_subsampling.py / "
            "src/mugla_subsampling.py."
        ),
    )
    p_mugla_subsampling.add_argument(
        "--experiments", nargs="+", default=None,
        help="Experiment IDs; defaults to the three canonical AOIs of this frozen analysis.",
    )
    p_mugla_subsampling.add_argument(
        "--from-stage", default="plan", choices=list(_MUGLA_SUBSAMPLING_STAGES),
        help=f"First stage to run. Stage order: {list(_MUGLA_SUBSAMPLING_STAGES)}.",
    )
    p_mugla_subsampling.add_argument(
        "--to-stage", default="summarize", choices=list(_MUGLA_SUBSAMPLING_STAGES),
        help="Last stage to run (inclusive).",
    )
    p_mugla_subsampling.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Print the plan, the resolved input hashes, the expected fit budget and "
            "every planned output path. Creates no directory, writes no file and "
            "fits no model."
        ),
    )
    p_mugla_subsampling.add_argument(
        "--resume", action="store_true",
        help=(
            "Verify and reuse only complete, hash-bound stages. A partially written "
            "stage or arm partition is never accepted."
        ),
    )
    p_mugla_subsampling.add_argument(
        "--force", action="store_true",
        help="Quarantine an existing namespace under _quarantine/ and start fresh. Never deletes.",
    )
    p_mugla_subsampling.add_argument(
        "--output-root", default=None,
        help="Alternative outputs/ root (dependency injection).",
    )
    p_mugla_subsampling.add_argument(
        "--experiments-root", default=None,
        help="Alternative outputs/experiments/ root (dependency injection).",
    )
    p_mugla_subsampling.set_defaults(func=cmd_mugla_subsampling)

    # --- window-closure-sensitivity ---
    p_window_closure = subparsers.add_parser(
        "window-closure-sensitivity",
        help="Predictor window-closure sensitivity of within-AOI baseline/thermal results.",
        description=(
            "Scientific purpose: shift the canonical predictor window EARLIER "
            "by preregistered day counts (default 0/7/14) while preserving its "
            "LENGTH exactly, and measure how the within-AOI baseline and "
            "thermal model results -- and the thermal contribution between "
            "them -- respond to the predictor closure date. Both window ends "
            "move together, so duration is invariant; the label window never "
            "moves. Every predictor derived from the window is rebuilt "
            "coherently (current Landsat LST and NDVI, each baseline year's "
            "LST and NDVI, current-window MODIS mean/std/valid-observation "
            "support, and the Step5/5C/7/8A products above them) -- moving "
            "only current LST while freezing NDVI/baseline/MODIS is refused. "
            "DEM, slope, landcover, grid, feature registry, hyper-parameters, "
            "seed and spatial block definition are held fixed. A single shared "
            "pre-label censoring interval and a single shared spatial fold "
            "assignment are applied identically to every variant, and all "
            "comparisons are made on the exact common cohort. Results are "
            "DESCRIPTIVE predictor-timing sensitivity: closing the window "
            "earlier is not an operational forecasting validation, and an "
            "interval containing zero is not evidence of equivalence. Writes "
            "only under outputs/diagnostics/window_closure_sensitivity/. "
            "Delegates to scripts/run_window_closure_sensitivity.py / "
            "src/window_closure_sensitivity.py."
        ),
    )
    p_window_closure.add_argument("--experiment", required=True, help="Experiment ID (core/regions.py registry entry).")
    p_window_closure.add_argument(
        "--shifts", nargs="+", type=int, default=list(_WINDOW_CLOSURE_DEFAULT_SHIFTS),
        help="Preregistered closure shifts in days (default: 0 7 14). Shifts move the window earlier; they are never window lengths.",
    )
    p_window_closure.add_argument("--from-stage", default="plan", choices=list(_WINDOW_CLOSURE_STAGES))
    p_window_closure.add_argument("--to-stage", default="compare", choices=list(_WINDOW_CLOSURE_STAGES))
    p_window_closure.add_argument("--force", action="store_true", help="Overwrite the dedicated window-closure namespace only; never canonical outputs or another diagnostics namespace.")
    p_window_closure.add_argument("--resume", action="store_true", help="Reuse completed stages whose recorded input hashes still match.")
    p_window_closure.add_argument("--dry-run", action="store_true", help="Print the full plan (windows, censor interval, export roles, planned paths); write no files, run no GEE query/export, no model fit, no bootstrap.")
    p_window_closure.set_defaults(func=cmd_window_closure_sensitivity)

    # --- domain-classifier-audit ---
    p_domain_classifier = subparsers.add_parser(
        "domain-classifier-audit",
        help="Generic multi-experiment pairwise domain-classifier (covariate-separability) diagnostic.",
        description=(
            "Scientific purpose: for every unordered pair among the "
            "resolved experiments, measure how well the two regions can be "
            "distinguished using predictor features alone (never burned "
            "labels, transfer predictions, coordinates, or experiment "
            "identity). The classifier target is region/domain identity, "
            "not burned-area risk. Population: burnable_tree_shrub_grass "
            "(includes both burned and unburned eligible cells; burned "
            "status is used only for input-accounting). Reuses the "
            "canonical Step9 baseline+thermal feature contract, the "
            "spatial-block construction and estimator already used "
            "elsewhere in this repo (RandomForestClassifier, "
            "class_weight=balanced), and canonical eligibility (pre-label "
            "exclusions apply, e.g. mugla_2021). No historical/legacy "
            "domain-classifier result exists anywhere in this repository "
            "(confirmed by exhaustive audit); legacy_comparable_domain_auc "
            "is therefore always null and legacy_method_available is "
            "always false -- spatial_block_domain_auc is the sole primary "
            "result. Does NOT establish causality. No experiment ID is "
            "hard-coded: --experiments takes an arbitrary explicit list, "
            "--all-enabled resolves every enabled, non-legacy "
            "experiment_id with a canonical Step8A dataset. All unordered "
            "pairs are generated automatically. Delegates to "
            "scripts/run_domain_classifier_audit.py / "
            "src/domain_classifier_audit.py."
        ),
    )
    p_domain_classifier_selection = p_domain_classifier.add_mutually_exclusive_group(required=True)
    p_domain_classifier_selection.add_argument(
        "--experiments", nargs="+", default=None,
        help="Explicit experiment_id list (core/regions.py registry entries); all unordered pairs are generated automatically.",
    )
    p_domain_classifier_selection.add_argument(
        "--all-enabled", action="store_true",
        help="Resolve every enabled, non-legacy core/regions.py experiment_id that has a canonical Step8A dataset.",
    )
    p_domain_classifier.add_argument("--force", action="store_true", help="Overwrite existing outputs already produced by a different analysis_id.")
    p_domain_classifier.add_argument("--dry-run", action="store_true", help="Resolve experiments/pairs and print the plan; no model fitting, no files written.")
    p_domain_classifier.set_defaults(func=cmd_domain_classifier_audit)

    # --- legacy ---
    p_legacy = subparsers.add_parser(
        "legacy",
        help="Yalnızca geriye dönük uyumluluk: Kozan-only, Drive-tabanlı Step1->Step8E.",
        description=(
            "Eski Step1-Step4B Google Drive/batch-export iş akışını, YALNIZCA "
            "kozan_2023 için, birebir eski davranışıyla çalıştırır. Bu YENİ "
            "varsayılan yol DEĞİLDİR ve 'experiment' alt-komutundan ÇAĞRILMAZ."
        ),
    )
    p_legacy.add_argument("--experiment", default=orch.LEGACY_EXPERIMENT_ID, help=f"Yalnızca {orch.LEGACY_COMPATIBLE_EXPERIMENT_IDS} desteklenir.")
    p_legacy.add_argument("--force", action="store_true", help="Çıktılar zaten varsa üzerine yaz.")
    p_legacy.add_argument("--dry-run", action="store_true", help="Hiçbir şey çalıştırma; yalnızca deney özetini bas.")
    p_legacy.set_defaults(func=cmd_legacy)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
