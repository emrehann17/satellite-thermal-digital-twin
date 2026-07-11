"""
run_cross_region_shift_audit.py

Step9E orkestratoru: Manavgat 2021 <-> Bejís 2022 cross-region transferinin
(Step9B/Step9C) discrimination'i neden koruyamadigini teshis eden POST-HOC
dagilim-kaymasi (distribution-shift) ve iliski-kaymasi (relationship-shift)
denetimini calistirir.

Bu bir orkestrasyon script'idir; asil mantik src/step9e_distribution_shift_audit.py
icindedir. Bu script:
    - hicbir modeli YENIDEN EGITMEZ
    - Step9B tahminlerini / Step9C bootstrap ciktilarini DEGISTIRMEZ
    - raporlanan Step9 sonucunu DEGISTIRMEZ
    - GEE, Step5, Step7 veya Step8'i YENIDEN CALISTIRMAZ
    - Step9A-D dosyalarini DEGISTIRMEZ (yalnizca salt-okunur girdi olarak kullanir)

CIKTI KOKU:
    outputs/cross_region/<source>__<target>/step9e/

CLI:
    python scripts/run_cross_region_shift_audit.py --source manavgat_2021 --target bejis_2022 --dry-run
    python scripts/run_cross_region_shift_audit.py --source manavgat_2021 --target bejis_2022 --force
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

log, log_file = setup_logger("run_cross_region_shift_audit")

BASE_DIR = PROJECT_ROOT


class CrossRegionShiftAuditRunnerError(SystemExit):
    """Fail-fast error for this orchestrator (diğer step'lerle aynı konvansiyon)."""


def _log_dry_run(source_id: str, target_id: str) -> None:
    from src.step9a_audit_cross_region_inputs import (
        ALL_POPULATIONS,
        PRIMARY_POPULATIONS,
        SECONDARY_POPULATIONS,
        resolve_step8a_dataset_path,
    )
    from src.step9e_distribution_shift_audit import (
        CATEGORICAL_FEATURES,
        NEVER_AUDIT_AS_FEATURE_COLUMNS,
        NUMERIC_FEATURES,
        planned_output_files,
        resolve_step9b_metrics_path,
        resolve_step9b_predictions_path,
        step9e_output_dir,
    )

    output_dir = step9e_output_dir(source_id, target_id)

    log.info("[dry-run] source=%s, target=%s", source_id, target_id)
    log.info("[dry-run] output directory: %s", output_dir)

    log.info("[dry-run] Girdi yollari (salt-okunur, hicbiri DEGISTIRILMEZ):")
    for exp_id in (source_id, target_id):
        dataset_path = resolve_step8a_dataset_path(exp_id)
        log.info(
            "  [%s] step8a dataset: %s (%s)", exp_id, dataset_path,
            "[VAR]" if dataset_path.exists() else "[EKSİK]",
        )

    predictions_path = resolve_step9b_predictions_path(source_id, target_id)
    metrics_path = resolve_step9b_metrics_path(source_id, target_id)
    log.info(
        "  [step9b] predictions: %s (%s)", predictions_path,
        "[VAR]" if predictions_path.exists() else "[EKSİK]",
    )
    log.info(
        "  [step9b] metrics: %s (%s)", metrics_path,
        "[VAR]" if metrics_path.exists() else "[EKSİK]",
    )

    log.info("[dry-run] Numeric audit features (%d): %s", len(NUMERIC_FEATURES), NUMERIC_FEATURES)
    log.info("[dry-run] Categorical audit features: %s", CATEGORICAL_FEATURES)
    log.info(
        "[dry-run] Never-audit-as-feature columns (identifiers/labels/diagnostic-only, %d): %s",
        len(NEVER_AUDIT_AS_FEATURE_COLUMNS), NEVER_AUDIT_AS_FEATURE_COLUMNS,
    )

    log.info("[dry-run] Primary populations: %s", PRIMARY_POPULATIONS)
    log.info("[dry-run] Secondary populations: %s", SECONDARY_POPULATIONS)
    log.info("[dry-run] All populations evaluated: %s", ALL_POPULATIONS)

    log.info("[dry-run] Planned output files:")
    for path in planned_output_files(output_dir):
        log.info("  %s", path)

    log.info(
        "[dry-run] Hicbir hesaplama (Part A-F) CALISTIRILMADI, hicbir cikti "
        "dosyasi yazilmadi (yalnizca hafif sema/yol kontrolleri + bu log)."
    )


def main(source_id: str, target_id: str, dry_run: bool = False, force: bool = False) -> dict:
    if source_id == target_id:
        raise CrossRegionShiftAuditRunnerError("--source ve --target ayni deney OLAMAZ.")

    if dry_run:
        _log_dry_run(source_id, target_id)
        return {"ran": False, "reason": "dry_run"}

    from src.step9e_distribution_shift_audit import run_shift_audit

    log.info("=" * 70)
    log.info("STEP9E: cross-region distribution-shift + relationship-shift audit")
    log.info("=" * 70)
    result = run_shift_audit(source_id=source_id, target_id=target_id, force=force)

    log.info("=" * 70)
    log.info(
        "STEP9E TAMAMLANDI. Diagnosis categories: %s",
        result.get("part_f_summary", {}).get("diagnosis_categories"),
    )
    log.info("=" * 70)

    return {"ran": True, "result": result}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step9E: Manavgat<->Bejís cross-region transferinin "
        "POST-HOC dagilim-kaymasi/iliski-kaymasi denetimi. Hicbir modeli "
        "YENIDEN EGITMEZ, Step9B/9C ciktilarini DEGISTIRMEZ, Step9 sonucunu "
        "DEGISTIRMEZ."
    )
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Hicbir hesaplama calistirma; yalnizca girdi yollarini, audit "
        "feature setlerini, populasyonlari ve planlanan cikti yollarini bas.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="step9e ciktisi zaten varsa uzerine yaz.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(source_id=args.source, target_id=args.target, dry_run=args.dry_run, force=args.force)