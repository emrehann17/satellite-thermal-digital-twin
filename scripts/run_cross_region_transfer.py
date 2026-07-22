"""
run_cross_region_transfer.py

Step9 orkestratoru: Step8'in ~500 m MCD64A1-hucre burned-area ASOSIASYON
modelinin (baseline + baseline+thermal) iki BAGIMSIZ Akdeniz yangin
bolgesi arasinda ne kadar genellendigini degerlendirir.

    1. manavgat_2021 -> bejis_2022 (train source, test target)
    2. bejis_2022 -> manavgat_2021 (train source, test target)

Bu 30 m'lik bir yangin tahmin modeli DEGILDIR, operasyonel bir yangin
tespit sistemi DEGILDIR, ve Step7 downscaling modelinin kendisini
transfer ETMEZ. Her iki bolge de KENDI bagimsiz Step5/Step5C/Step7/Step8A
predictor'larini kullanir (bu script GEE export CALISTIRMAZ, mevcut
Step5-Step8 ciktilarini DEGISTIRMEZ).

ASAMALAR (sirasiyla, hepsi tek (source,target) cifti icin CIFT YONLU
calisir -- bkz. src/step9b_run_cross_region_transfer.py):
    Step9A: girdi uygunluk denetimi (fail-fast)
    Step9B: iki yonlu transfer (source-only fit, target'ta degerlendirme)
    Step9C: hedef-bolge spatial-block bootstrap (%95 percentile CI)
    Step9D: birlesik karsilastirmali final rapor

CIKTI KOKU:
    outputs/cross_region/<source>__<target>/{step9a,step9b,step9c,step9d}/

CLI:
    python scripts/run_cross_region_transfer.py --source manavgat_2021 --target bejis_2022 --reverse --dry-run
    python scripts/run_cross_region_transfer.py --source manavgat_2021 --target bejis_2022 --reverse --force
    python scripts/run_cross_region_transfer.py --source manavgat_2021 --target mugla_2021 --single-direction --dry-run
    python scripts/run_cross_region_transfer.py --source mugla_2021 --target manavgat_2021 --single-direction
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT

log, log_file = setup_logger("run_cross_region_transfer")

BASE_DIR = PROJECT_ROOT


class CrossRegionRunnerError(SystemExit):
    """Fail-fast error for this orchestrator (diğer step'lerle aynı konvansiyon)."""


def _log_dry_run(
    source_id: str, target_id: str, reverse: bool, single_direction: bool,
) -> None:
    from src.step9a_audit_cross_region_inputs import (
        ALL_POPULATIONS,
        FORBIDDEN_MODEL_COLUMNS,
        PRIMARY_POPULATIONS,
        SECONDARY_POPULATIONS,
        SHARED_BASELINE_FEATURES,
        SHARED_THERMAL_FEATURES,
        cross_region_output_root,
        resolve_step8a_dataset_path,
        resolve_gate_path,
        resolve_step8a_stats_path,
    )

    root = cross_region_output_root(source_id, target_id)
    directions = [f"{source_id}_to_{target_id}"]
    if not single_direction:
        directions.append(f"{target_id}_to_{source_id}")

    log.info(
        "[dry-run] source=%s, target=%s, reverse=%s, single_direction=%s",
        source_id, target_id, reverse, single_direction,
    )
    log.info("[dry-run] transfer directions to be evaluated: %s", directions)
    if not reverse and not single_direction:
        log.info(
            "[dry-run] NOT: --reverse verilmedi. Step9B kendi ic tasarimi geregi "
            "YINE DE HER IKI yonu de hesaplar (bkz. modul docstring); --reverse "
            "bayragi bu davranisi acikca teyit etmek icindir."
        )

    log.info("[dry-run] output root: %s", root)
    for sub in ("step9a", "step9b", "step9c", "step9d"):
        log.info("  planned output dir: %s", root / sub)

    log.info("[dry-run] Girdi yollari (her iki bolge icin):")
    for exp_id in (source_id, target_id):
        dataset_path = resolve_step8a_dataset_path(exp_id)
        stats_path = resolve_step8a_stats_path(exp_id)
        gate_path = resolve_gate_path(exp_id)
        log.info(
            "  [%s] step8a dataset: %s (%s)", exp_id, dataset_path,
            "[VAR]" if dataset_path.exists() else "[EKSİK]",
        )
        log.info(
            "  [%s] step8a stats: %s (%s)", exp_id, stats_path,
            "[VAR]" if stats_path.exists() else "[EKSİK]",
        )
        log.info(
            "  [%s] burned-landcover gate: %s (%s)", exp_id, gate_path,
            "[VAR]" if gate_path.exists() else "[EKSİK]",
        )

    log.info("[dry-run] Shared baseline features: %s", SHARED_BASELINE_FEATURES)
    log.info("[dry-run] Shared thermal features: %s", SHARED_THERMAL_FEATURES)
    log.info("[dry-run] Primary populations: %s", PRIMARY_POPULATIONS)
    log.info("[dry-run] Secondary populations: %s", SECONDARY_POPULATIONS)
    log.info("[dry-run] All populations evaluated: %s", ALL_POPULATIONS)
    log.info("[dry-run] Forbidden model columns (checked, never used as features): %s", FORBIDDEN_MODEL_COLUMNS)

    log.info(
        "[dry-run] Hicbir egitim/tahmin/bootstrap CALISTIRILMADI, hicbir "
        "cikti dosyasi yazilmadi (yalnizca bu log)."
    )


def main(
    source_id: str, target_id: str, reverse: bool = False,
    dry_run: bool = False, force: bool = False,
    single_direction: bool = False,
) -> dict:
    if source_id == target_id:
        raise CrossRegionRunnerError("--source ve --target ayni deney OLAMAZ.")

    if dry_run:
        _log_dry_run(source_id, target_id, reverse, single_direction)
        return {"ran": False, "reason": "dry_run"}

    if not reverse and not single_direction:
        log.warning(
            "--reverse verilmedi. Step9B tasarimi geregi HER IKI transfer yonu "
            "da (source->target VE target->source) yine de hesaplanacak -- bkz. "
            "src/step9b_run_cross_region_transfer.py modul docstring."
        )

    from src.step9a_audit_cross_region_inputs import main as run_step9a
    from src.step9b_run_cross_region_transfer import run_transfer as run_step9b
    from src.step9c_cross_region_block_bootstrap import run_bootstrap as run_step9c
    from src.step9d_build_cross_region_report import main as run_step9d

    log.info("=" * 70)
    log.info("STEP9A: cross-region girdi uygunluk denetimi")
    log.info("=" * 70)
    audit_result = run_step9a(source_id=source_id, target_id=target_id, force=force)

    log.info("=" * 70)
    log.info("STEP9B: iki yonlu cross-region transfer")
    log.info("=" * 70)
    transfer_result = run_step9b(
        source_id=source_id, target_id=target_id, force=force,
        bidirectional=not single_direction,
    )

    log.info("=" * 70)
    log.info("STEP9C: hedef-bolge spatial-block bootstrap")
    log.info("=" * 70)
    bootstrap_result = run_step9c(source_id=source_id, target_id=target_id, force=force)

    log.info("=" * 70)
    log.info("STEP9D: birlesik final rapor")
    log.info("=" * 70)
    report_result = run_step9d(source_id=source_id, target_id=target_id, force=force)

    log.info("=" * 70)
    log.info(
        "STEP9 TAMAMLANDI. Genel sonuc: %s", report_result.get("overall_conclusion"),
    )
    log.info("=" * 70)

    return {
        "ran": True,
        "audit": audit_result,
        "transfer": transfer_result,
        "bootstrap": bootstrap_result,
        "report": report_result,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step9: Step8 burned-area association modelinin iki "
        "bagimsiz Akdeniz yangin bolgesi arasinda cross-region transfer "
        "degerlendirmesi. 30 m fire prediction DEGILDIR, operasyonel "
        "yangin tespiti DEGILDIR, Step7 downscaling modelini transfer ETMEZ."
    )
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument(
        "--reverse", action="store_true",
        help="Ters yonu (target->source) de acikca degerlendirmeyi teyit eder "
        "(Step9B zaten her iki yonu de hesaplar; bu bayrak acik teyit icindir).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Hicbir egitim/tahmin/bootstrap calistirma; yalnizca girdi "
        "yollarini, shared/forbidden feature setlerini, populasyonlari ve "
        "planlanan cikti yollarini bas.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Her Step9 alt-asamasinin ciktilari zaten varsa uzerine yaz.",
    )
    parser.add_argument(
        "--single-direction", action="store_true",
        help="Yalnizca --source -> --target yonunu calistir. Bu secenek, her "
        "yonu kendi <source>__<target> namespace'inde tutmak icindir; "
        "verilmezse legacy iki-yonlu Step9 davranisi korunur.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        source_id=args.source, target_id=args.target,
        reverse=args.reverse, dry_run=args.dry_run, force=args.force,
        single_direction=args.single_direction,
    )
