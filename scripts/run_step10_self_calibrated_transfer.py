"""
run_step10_self_calibrated_transfer.py

Step10 orkestratoru: Manavgat 2021 <-> Bejís 2022 arasinda unsupervised
self-calibrated cross-region transfer (raw_source_only, regionwise_zscore,
coral_after_regionwise_zscore) deneyini calistirir (Step10A -> B -> C -> D).

Asil bilimsel mantik src/step10a-d_*.py icindedir; bu script yalnizca
orkestrasyon yapar (mevcut Step9E/9F CLI-wrapper deseniyle AYNI).

CLI:
    python scripts/run_step10_self_calibrated_transfer.py --source manavgat_2021 --target bejis_2022 --reverse --dry-run
    python scripts/run_step10_self_calibrated_transfer.py --source manavgat_2021 --target bejis_2022 --reverse
    python scripts/run_step10_self_calibrated_transfer.py --source manavgat_2021 --target bejis_2022 --reverse --report-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.config import STEP10_BOOTSTRAP_REPLICATES, STEP10_RANDOM_STATE
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT

log, log_file = setup_logger("run_step10_self_calibrated_transfer")

BASE_DIR = PROJECT_ROOT


class Step10RunnerError(SystemExit):
    """Fail-fast error for this orchestrator (diğer step'lerle aynı konvansiyon)."""


def _log_dry_run(source_id: str, target_id: str, reverse: bool, bootstrap_replicates: int, seed: int) -> None:
    from core.step10_shared import (
        ADAPTATION_METHODS, FEATURE_LISTS, MODEL_NAME, PRIMARY_POPULATION,
        resolve_step8b_metrics_path, resolve_step8b_predictions_path,
        resolve_step9b_metrics_path, step10_output_dir,
    )
    from src.step10a_preregistration_and_audit import build_scientific_config, planned_output_files
    from src.step9a_audit_cross_region_inputs import resolve_step8a_dataset_path

    output_dir = step10_output_dir(source_id, target_id)
    directions = [f"{source_id}_to_{target_id}"]
    if reverse:
        directions.append(f"{target_id}_to_{source_id}")

    log.info("[dry-run] source=%s, target=%s, reverse=%s", source_id, target_id, reverse)
    log.info("[dry-run] transfer directions: %s (Step10 HER ZAMAN iki yonu de calistirir)", directions)
    log.info("[dry-run] output root: %s", output_dir)

    log.info("[dry-run] Girdi yollari (salt-okunur):")
    for exp_id in (source_id, target_id):
        step8a_path = resolve_step8a_dataset_path(exp_id)
        step8b_path = resolve_step8b_predictions_path(exp_id)
        step8b_metrics = resolve_step8b_metrics_path(exp_id)
        log.info("  [%s] step8a dataset: %s (%s)", exp_id, step8a_path, "[VAR]" if step8a_path.exists() else "[EKSİK]")
        log.info("  [%s] step8b OOF predictions (within-region referans): %s (%s)", exp_id, step8b_path, "[VAR]" if step8b_path.exists() else "[EKSİK]")
        log.info("  [%s] step8b metrics: %s (%s)", exp_id, step8b_metrics, "[VAR]" if step8b_metrics.exists() else "[EKSİK]")
    step9b_metrics_path = resolve_step9b_metrics_path(source_id, target_id)
    log.info("  [step9b] metrics (ZORUNLU, raw reprodüksiyon kontrolü icin): %s (%s)", step9b_metrics_path, "[VAR]" if step9b_metrics_path.exists() else "[EKSİK]")

    log.info("[dry-run] Primary population: %s", PRIMARY_POPULATION)
    log.info("[dry-run] Adaptation methods (SABIT, 3): %s", ADAPTATION_METHODS)
    log.info("[dry-run] Model families: baseline (%d feature), thermal (%d feature); model=%s (Step8B/Step9B ile AYNI, TUNING YOK)",
              len(FEATURE_LISTS["baseline"]), len(FEATURE_LISTS["thermal"]), MODEL_NAME)
    log.info("[dry-run] Bootstrap: %d replika, seed=%d, target spatial_block_id ile esli (paired)", bootstrap_replicates, seed)

    scientific_config = build_scientific_config(source_id, target_id)
    log.info("[dry-run] Frozen preregistration plani (henuz YAZILMADI):")
    log.info("  primary_estimand: %s", scientific_config["primary_estimand"])
    log.info("  regionwise_zscore: %s", scientific_config["adaptation_methods"]["regionwise_zscore"]["metadata_classification"])
    log.info("  coral lambda: %s", scientific_config["adaptation_methods"]["coral_after_regionwise_zscore"]["lambda"])
    log.info("  threshold_policy: %s", scientific_config["threshold_policy"])

    log.info(
        "[dry-run] TARGET-LABEL FIREWALL: Step10B, hedef DataFrame'i 'burned' "
        "kolonu OLMADAN alir (strip_target_to_label_blind); hedef etiketi "
        "Step10C'ye kadar HICBIR YERDE yuklenmez/kullanilmaz."
    )

    log.info("[dry-run] Planned output files:")
    for path in planned_output_files(output_dir):
        log.info("  %s", path)

    log.info(
        "[dry-run] Hicbir fit/adapt/predict/bootstrap CALISTIRILMADI, hicbir "
        "bilimsel tahmin/cikti dosyasi YAZILMADI (yalnizca hafif sema/yol "
        "kontrolleri + bu log)."
    )


def _run_report_only(source_id: str, target_id: str) -> dict:
    """Dispatch only Step10D using the immutable preregistration analysis_id."""
    import json

    from core.step10_shared import Step10Error, step10_output_dir
    from src.step10d_final_report import run_step10d

    preregistration_path = step10_output_dir(source_id, target_id) / "step10_preregistration.json"
    if not preregistration_path.is_file():
        raise Step10Error(
            f"Report-only mode requires the existing frozen preregistration: {preregistration_path}"
        )
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    analysis_id = preregistration.get("analysis_id")
    if not analysis_id:
        raise Step10Error("Frozen preregistration has no analysis_id.")
    report = run_step10d(
        source_id=source_id, target_id=target_id, analysis_id=analysis_id,
        force=True, report_only_generation=True,
    )
    return {
        "ran": True, "report_only": True, "analysis_id": analysis_id,
        "step10d": report, "scientific_stages_called": [],
    }


def _log_report_only_plan(source_id: str, target_id: str) -> dict:
    from src.step10d_final_report import report_only_plan

    plan = report_only_plan(source_id, target_id)
    log.info("[dry-run][report-only] Step10D only; Step10A/B/C will not be called.")
    for path in plan["read_only_inputs"]:
        log.info("[dry-run][report-only] read protected: %s", path)
    for path in plan["writable_files_if_executed"]:
        log.info("[dry-run][report-only] would write: %s", path)
    log.info("[dry-run][report-only] no files written.")
    return plan


def main(
    source_id: str, target_id: str, reverse: bool = False, dry_run: bool = False,
    force: bool = False, bootstrap_replicates: int = STEP10_BOOTSTRAP_REPLICATES,
    seed: int = STEP10_RANDOM_STATE, report_only: bool = False,
) -> dict:
    if source_id == target_id:
        raise Step10RunnerError("--source ve --target ayni deney OLAMAZ.")

    if dry_run:
        if report_only:
            return {"ran": False, "reason": "dry_run", "report_only": True, "plan": _log_report_only_plan(source_id, target_id)}
        _log_dry_run(source_id, target_id, reverse, bootstrap_replicates, seed)
        return {"ran": False, "reason": "dry_run"}

    if report_only:
        return _run_report_only(source_id, target_id)

    if not reverse:
        log.warning(
            "--reverse verilmedi. Step10 tasarimi geregi HER IKI transfer yonu "
            "da yine de hesaplanacak; --reverse bayragi bu davranisi acikca "
            "teyit icindir."
        )

    from src.step10a_preregistration_and_audit import main as run_step10a
    from src.step10b_label_blind_adaptation import run_step10b
    from src.step10c_paired_evaluation_bootstrap import run_step10c
    from src.step10d_final_report import run_step10d

    log.info("=" * 70)
    log.info("STEP10A: on-kayit (preregistration) + girdi denetimi")
    log.info("=" * 70)
    a_result = run_step10a(source_id=source_id, target_id=target_id, force=force, dry_run=False)
    analysis_id = a_result["analysis_id"]
    log.info("analysis_id = %s", analysis_id)

    log.info("=" * 70)
    log.info("STEP10B: label-blind adaptasyon + tahmin (raw/zscore/CORAL)")
    log.info("=" * 70)
    b_result = run_step10b(source_id=source_id, target_id=target_id, analysis_id=analysis_id, force=force, random_state=seed)

    log.info("=" * 70)
    log.info("STEP10C: esli degerlendirme + reprodüksiyon kontrolu + bootstrap")
    log.info("=" * 70)
    c_result = run_step10c(
        source_id=source_id, target_id=target_id, analysis_id=analysis_id, force=force,
        n_replicates=bootstrap_replicates, seed=seed,
    )

    log.info("=" * 70)
    log.info("STEP10D: final rapor")
    log.info("=" * 70)
    d_result = run_step10d(source_id=source_id, target_id=target_id, analysis_id=analysis_id, force=force)

    log.info("=" * 70)
    log.info("STEP10 TAMAMLANDI. analysis_id=%s", analysis_id)
    log.info("=" * 70)

    return {
        "ran": True, "analysis_id": analysis_id, "step10a": a_result, "step10b_skipped": b_result.get("skipped"),
        "step10c": c_result, "step10d": d_result,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step10: unsupervised self-calibrated cross-region transfer "
        "(preregistered, target-label-blind adaptation, N-yollu esli bootstrap)."
    )
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--reverse", action="store_true", help="Ters yonu de acikca teyit eder (Step10 zaten her iki yonu de hesaplar).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="step10 ciktilari zaten varsa uzerine yaz (on-kayit HARIC -- o asla degistirilmez).")
    parser.add_argument("--report-only", action="store_true", help="Yalnizca frozen Step10 ciktilarindan Step10D final raporlarini yeniden uret.")
    parser.add_argument("--bootstrap-replicates", type=int, default=STEP10_BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=STEP10_RANDOM_STATE)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        source_id=args.source, target_id=args.target, reverse=args.reverse, dry_run=args.dry_run,
        force=args.force, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed,
        report_only=args.report_only,
    )