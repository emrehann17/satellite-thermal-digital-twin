"""
run_exploratory_transfer_features.py

Step9F orkestratoru: Manavgat 2021 <-> Bejís 2022 arasindaki KESIFSEL,
POST-HOC cross-region feature-representation deneyini calistirir.

Bu script bir orkestrasyon script'idir; asil mantik
src/step9f_exploratory_transfer_feature_experiment.py icindedir. Bu script:
    - hicbir modeli Step9A-Step9E ADINA YENIDEN EGITMEZ
    - Step9A/B/C/D/E dosyalarini DEGISTIRMEZ (yalnizca salt-okunur girdi
      olarak kullanir)
    - Step8A veri setlerini DEGISTIRMEZ
    - GEE'yi YENIDEN CALISTIRMAZ
    - keyfi feature aramasina IZIN VERMEZ (varyant ailesi SABITTIR)

CIKTI KOKU:
    outputs/cross_region/<source>__<target>/step9f/

CLI:
    python scripts/run_exploratory_transfer_features.py --source manavgat_2021 --target bejis_2022 --reverse --dry-run
    python scripts/run_exploratory_transfer_features.py --source manavgat_2021 --target bejis_2022 --reverse --force
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

log, log_file = setup_logger("run_exploratory_transfer_features")

BASE_DIR = PROJECT_ROOT


class ExploratoryTransferRunnerError(SystemExit):
    """Fail-fast error for this orchestrator (diğer step'lerle aynı konvansiyon)."""


def _log_dry_run(source_id: str, target_id: str, reverse: bool, bootstrap_replicates: int, seed: int) -> None:
    from core.cross_region_experiment import (
        FIXED_VARIANTS,
        REGIME_A_LABEL,
        REGIME_B_LABELS,
        REGIME_B_VARIANTS,
        FORBIDDEN_MODEL_COLUMNS,
    )
    from src.step9a_audit_cross_region_inputs import (
        ALL_POPULATIONS, PRIMARY_POPULATIONS, SECONDARY_POPULATIONS,
        resolve_step8a_dataset_path,
    )
    from src.step9f_exploratory_transfer_feature_experiment import (
        MODEL_NAME, N_SPLITS,
        gather_step9_provenance, planned_output_files, resolve_step9b_metrics_path,
        resolve_step9e_audit_path, step9f_output_dir,
    )

    output_dir = step9f_output_dir(source_id, target_id)
    directions = [f"{source_id}_to_{target_id}"]
    if reverse:
        directions.append(f"{target_id}_to_{source_id}")

    log.info("[dry-run] source=%s, target=%s, reverse=%s", source_id, target_id, reverse)
    log.info("[dry-run] transfer directions to be evaluated: %s (Step9F HER ZAMAN iki yonu de calistirir)", directions)
    log.info("[dry-run] output root: %s", output_dir)

    log.info("[dry-run] Girdi yollari (salt-okunur):")
    for exp_id in (source_id, target_id):
        dataset_path = resolve_step8a_dataset_path(exp_id)
        log.info("  [%s] step8a dataset: %s (%s)", exp_id, dataset_path, "[VAR]" if dataset_path.exists() else "[EKSİK]")

    step9b_metrics_path = resolve_step9b_metrics_path(source_id, target_id)
    step9e_audit_path = resolve_step9e_audit_path(source_id, target_id)
    log.info(
        "  [step9b] metrics (ZORUNLU, reprodüksiyon kontrolü icin): %s (%s)",
        step9b_metrics_path, "[VAR]" if step9b_metrics_path.exists() else "[EKSİK]",
    )
    log.info(
        "  [step9e] distribution_shift_audit.json (yalnizca provenance/motivasyon, OPSIYONEL): %s (%s)",
        step9e_audit_path, "[VAR]" if step9e_audit_path.exists() else "[EKSİK]",
    )
    provenance = gather_step9_provenance(source_id, target_id)
    for stage, info in provenance.items():
        log.info("  [%s] dizin: %s (%s)", stage, info["dir"], "[VAR]" if info["exists"] else "[EKSİK]")

    log.info("[dry-run] Populations: primary=%s, secondary=%s, all=%s", PRIMARY_POPULATIONS, SECONDARY_POPULATIONS, ALL_POPULATIONS)

    log.info("[dry-run] SABIT feature varyantlari (%d, Regime A):", len(FIXED_VARIANTS))
    for variant, features in FIXED_VARIANTS.items():
        log.info("  %s (%d feature): %s", variant, len(features), features)
    log.info("[dry-run] Regime A label: %s", REGIME_A_LABEL)
    log.info("[dry-run] Regime B labels: %s (YALNIZCA varyantlar: %s)", REGIME_B_LABELS, REGIME_B_VARIANTS)

    log.info("[dry-run] Model konfigurasyonu: %s (Step8B/Step9B ile AYNI, TUNING YOK), n_splits=%d, seed=%d", MODEL_NAME, N_SPLITS, seed)
    log.info("[dry-run] Bootstrap replicate count: %d", bootstrap_replicates)

    log.info("[dry-run] Yasak kolonlar (checked, asla feature olarak KULLANILMAZ, %d): %s", len(FORBIDDEN_MODEL_COLUMNS), FORBIDDEN_MODEL_COLUMNS)
    leaked = set()
    for features in FIXED_VARIANTS.values():
        leaked |= set(features).intersection(FORBIDDEN_MODEL_COLUMNS)
    log.info("[dry-run] Yasak kolon sizintisi kontrolu: %s", "YOK (guvenli)" if not leaked else f"BULUNDU: {leaked}")

    log.info("[dry-run] Planned output files:")
    for path in planned_output_files(output_dir):
        log.info("  %s", path)

    log.info(
        "[dry-run] Hicbir model fit/OOF/target degerlendirmesi/bootstrap CALISTIRILMADI, "
        "hicbir cikti dosyasi yazilmadi (yalnizca hafif sema/yol kontrolleri + bu log)."
    )


def main(
    source_id: str, target_id: str, reverse: bool = False, dry_run: bool = False,
    force: bool = False, bootstrap_replicates: int = 1000, seed: int | None = None,
) -> dict:
    if source_id == target_id:
        raise ExploratoryTransferRunnerError("--source ve --target ayni deney OLAMAZ.")

    from src.step9f_exploratory_transfer_feature_experiment import RANDOM_STATE
    resolved_seed = seed if seed is not None else RANDOM_STATE

    if dry_run:
        _log_dry_run(source_id, target_id, reverse, bootstrap_replicates, resolved_seed)
        return {"ran": False, "reason": "dry_run"}

    if not reverse:
        log.warning(
            "--reverse verilmedi. Step9F tasarimi geregi HER IKI transfer yonu da "
            "yine de hesaplanacak (bkz. src/step9f_exploratory_transfer_feature_experiment.py "
            "modul docstring); --reverse bayragi bu davranisi acikca teyit icindir."
        )

    from src.step9f_exploratory_transfer_feature_experiment import run_step9f

    log.info("=" * 70)
    log.info("STEP9F: kesifsel cross-region feature-representation deneyi")
    log.info("=" * 70)
    result = run_step9f(
        source_id=source_id, target_id=target_id, force=force,
        bootstrap_replicates=bootstrap_replicates, seed=resolved_seed,
    )
    log.info("=" * 70)
    log.info("STEP9F TAMAMLANDI.")
    log.info("=" * 70)
    return {"ran": True, "result": result}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step9F: Manavgat<->Bejís cross-region transferi icin "
        "KESIFSEL, POST-HOC feature-temsili deneyi. Tarafsiz dis validation "
        "DEGILDIR, Step9'un DUZELTMESI DEGILDIR, transfer-safe feature "
        "setinin KANITI DEGILDIR."
    )
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument(
        "--reverse", action="store_true",
        help="Ters yonu (target->source) de acikca teyit eder (Step9F zaten her iki yonu de hesaplar).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Hicbir model fit/degerlendirme calistirma; yalnizca planlanan yollari/varyantlari/rejimleri bas.")
    parser.add_argument("--force", action="store_true", help="step9f ciktisi zaten varsa uzerine yaz.")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000, help="Hedef-bolge esli spatial-block bootstrap replika sayisi (varsayilan: 1000).")
    parser.add_argument("--seed", type=int, default=None, help="Rastgele seed (varsayilan: Step9B ile AYNI sabit seed).")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        source_id=args.source, target_id=args.target, reverse=args.reverse,
        dry_run=args.dry_run, force=args.force,
        bootstrap_replicates=args.bootstrap_replicates, seed=args.seed,
    )