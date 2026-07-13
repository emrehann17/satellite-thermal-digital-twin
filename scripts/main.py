"""
main.py

satellite-thermal-digital-twin: deney-farkında (experiment-aware) CLI
orkestratörü.

Bu script İNCE (thin) tutulur -- hiçbir Step'in bilimsel/istatistiksel
mantığını yeniden uygulamaz. Tüm gerçek iş, mevcut callable API'lere sahip
runner'lara devredilir (bkz. core/pipeline_orchestrator.py):

    experiment   -> gate / predictors / step7 / step8 asama zinciri
                    (core/pipeline_orchestrator.py -> scripts/run_label_gate_only.py,
                    scripts/run_predictors_only.py,
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
    subparsers = parser.add_subparsers(dest="command", metavar="{experiment,step8-robustness,transfer,shift-audit,transfer-explore,self-cal-transfer,legacy}")

    # --- experiment ---
    p_exp = subparsers.add_parser(
        "experiment",
        help="Deney-farkında asama zinciri (gate -> predictors -> step7 -> step8).",
        description=(
            "Bir deney (experiment_id) için gate/predictors/step7/step8 "
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
        ),
    )
    p_exp.add_argument("--experiment", required=True, help="core/regions.py EXPERIMENTS kaydındaki experiment_id (örn. manavgat_2021, bejis_2022, kozan_2023).")
    p_exp.add_argument("--from-stage", required=True, choices=orch.STAGE_ORDER, help=f"Zincirin başlayacağı asama (sıra: {orch.STAGE_ORDER}).")
    p_exp.add_argument("--to-stage", required=True, choices=orch.STAGE_ORDER, help=f"Zincirin biteceği asama (sıra: {orch.STAGE_ORDER}).")
    p_exp.add_argument("--predictor-mode", required=True, choices=list(orch.PREDICTOR_MODES), help="'export': direkt/tiled GEE local download; 'local-only': yalnızca yerel Step5/Step5C (GEE'ye dokunmaz). Yalnızca 'predictors' asaması için anlamlıdır.")
    p_exp.add_argument("--export-labels", action="store_true", help="Yalnızca 'gate' asamasını etkiler: gate'ten önce raw MCD64A1 BurnDate export'unu çalıştırır.")
    p_exp.add_argument("--force", action="store_true", help="Her asamanın çıktıları zaten varsa üzerine yaz.")
    p_exp.add_argument("--dry-run", action="store_true", help="Hiçbir GEE/eğitim çalıştırma; deney planını + her asamanın kendi planlanan yollarını bas.")
    p_exp.set_defaults(func=cmd_experiment)

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