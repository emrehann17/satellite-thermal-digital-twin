"""
run_label_gate_only.py

Step0B: guvenli "gate-only" calistirici.

Modelleme (Step7A-Step8E) oncesinde gerekli MINIMUM label/gate zincirini
calistirir:
    [opsiyonel] raw MCD64A1 BurnDate export (Step6'nin canonical export'u)
    -> Step6B burned-landcover gate

Step1-Step8'in geri kalanini KESINLIKLE CALISTIRMAZ.

ONEMLI KAPSAM SINIRLAMASI (Step0B):
    Step1-Step6 hala core/config.py'deki LEGACY sabitleri (REGION_NAME,
    PREDICTOR_*_DATE, LABEL_*_DATE) kullanir; deney-farkinda (experiment-
    aware) DEGILDIR. Bu yuzden bu script su an SADECE "kozan_2023" icin
    gercekten calisir. "manavgat_2021" (veya baska herhangi bir deney) icin:
        - Step0 ozetini basar,
        - "Step1-Step6 henuz deney-farkinda degil, calistirilamiyor" der,
        - TEMIZ sekilde cikar.
    Bu script HICBIR ZAMAN, Manavgat (veya baska bir deney) etiketi altinda
    sessizce Kozan verisini calistirmaz. Step1-Step6 deney-farkinda hale
    getirildiginde, bu script'in yapisi degismeden manavgat_2021 destegi
    ACILABILIR (bkz. RUNNABLE_EXPERIMENTS asagida).

CLI:
    python scripts/run_label_gate_only.py --experiment kozan_2023 --skip-export --force
    python scripts/run_label_gate_only.py --experiment kozan_2023 --export-labels --force
    python scripts/run_label_gate_only.py --experiment manavgat_2021 --dry-run

Flags:
    --dry-run        Hicbir sey CALISTIRMAZ; yalnizca Step0 ozetini basar.
    --skip-export     Raw BurnDate export'u ATLA (varsayilan davranis);
                      gate mevcut outputs/validation/labels/mcd64a1_raw.tif
                      dosyasini kullanir.
    --export-labels   Gate'ten ONCE raw BurnDate export'unu (GEE) calistir.
    --force           Gate ciktilari zaten varsa uzerine yaz.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.io_utils import setup_logger
from core.regions import get_active_experiment, get_experiment_output_root

log, log_file = setup_logger("run_label_gate_only")

# Step1-Step6 deney-farkinda hale geldikce buraya yeni experiment_id'ler
# eklenir. Su an SADECE kozan_2023.
RUNNABLE_EXPERIMENTS = ("kozan_2023",)


class LabelGateRunnerError(SystemExit):
    """Fail-fast error for this runner (diger step'lerle ayni konvansiyon)."""


def _print_step0_summary(exp: dict, output_root: Path) -> None:
    log.info("[Step0] Active experiment: %s", exp["experiment_id"])
    log.info("[Step0] Display name: %s", exp["display_name"])
    log.info("[Step0] Region: %s", exp["region_key"])
    log.info("[Step0] Role: %s", exp["role"])
    log.info(
        "[Step0] Predictor window: %s -> %s",
        exp["predictor_start_date"], exp["predictor_end_date"],
    )
    log.info(
        "[Step0] Label window: %s -> %s",
        exp["label_start_date"], exp["label_end_date"],
    )
    log.info("[Step0] Baseline years: %s", ", ".join(str(y) for y in exp["baseline_years"]) or "(yok)")
    log.info("[Step0] Output root: %s", output_root)


def main(
    experiment_id: str = "kozan_2023",
    dry_run: bool = False,
    skip_export: bool = False,
    export_labels: bool = False,
    force: bool = False,
) -> dict:
    exp = get_active_experiment(experiment_id)
    output_root = get_experiment_output_root(experiment_id)
    _print_step0_summary(exp, output_root)

    if experiment_id not in RUNNABLE_EXPERIMENTS:
        log.warning(
            "'%s' deneyi henuz calistirilamiyor: Step1-Step6 hala legacy "
            "kozan_2023 config sabitlerini (core/config.py REGION_NAME, "
            "PREDICTOR_*_DATE, LABEL_*_DATE) kullaniyor, deney-farkinda "
            "(experiment-aware) DEGIL. Bu script Kozan verisini bu deney "
            "etiketiyle SESSIZCE CALISTIRMAZ -- temiz sekilde cikiyor. "
            "Su an calistirilabilir deneyler: %s",
            experiment_id, RUNNABLE_EXPERIMENTS,
        )
        return {
            "experiment_id": experiment_id,
            "ran": False,
            "reason": "not_experiment_aware_yet",
            "runnable_experiments": list(RUNNABLE_EXPERIMENTS),
        }

    if dry_run:
        log.info("[dry-run] Export/gate CALISTIRILMADI.")
        return {"experiment_id": experiment_id, "ran": False, "reason": "dry_run"}

    if skip_export and export_labels:
        raise LabelGateRunnerError(
            "--skip-export ve --export-labels birlikte verilemez (celiskili)."
        )
    do_export = export_labels and not skip_export

    export_result = None
    if do_export:
        log.info("Raw MCD64A1 BurnDate export calistiriliyor (--export-labels)...")
        try:
            from src.step6_validate_fire_relation import export_raw_mcd64a1_labels
        except Exception as exc:  # noqa: BLE001
            raise LabelGateRunnerError(
                f"src.step6_validate_fire_relation import edilemedi: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        export_result = export_raw_mcd64a1_labels(also_binary=True)
        log.info("Raw BurnDate export tamamlandi: %s", export_result["raw_path"])
    else:
        log.info(
            "Raw BurnDate export ATLANDI (--skip-export ya da varsayilan). "
            "Gate mevcut outputs/validation/labels/mcd64a1_raw.tif dosyasini "
            "kullanacak (yoksa gate net bir hata verecektir)."
        )

    log.info("Step6B burned-landcover gate calistiriliyor...")
    try:
        from src.step6b_burned_landcover_gate import main as run_gate
    except Exception as exc:  # noqa: BLE001
        raise LabelGateRunnerError(
            f"src.step6b_burned_landcover_gate import edilemedi: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    gate_result = run_gate(force=force)

    log.info(
        "TAMAMLANDI. Gate karari: %s (burned_count=%s). JSON: %s",
        gate_result["decision"], gate_result["burned_count"], gate_result["json_path"],
    )

    return {
        "experiment_id": experiment_id,
        "ran": True,
        "export_result": export_result,
        "gate_result": gate_result,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step0B: modelleme oncesi minimum label/gate zincirini "
        "(opsiyonel raw BurnDate export + Step6B burned-landcover gate) "
        "calistirir. Step1-Step8'in geri kalanini CALISTIRMAZ. Su an "
        f"sadece {RUNNABLE_EXPERIMENTS} gercekten calisir."
    )
    parser.add_argument("--experiment", type=str, default="kozan_2023")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Hicbir sey calistirma; yalnizca Step0 ozetini bas.",
    )
    parser.add_argument(
        "--skip-export", action="store_true",
        help="Raw BurnDate export'unu atla (varsayilan davranis).",
    )
    parser.add_argument(
        "--export-labels", action="store_true",
        help="Gate'ten once raw BurnDate export'unu (GEE) calistir.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Gate ciktilari zaten varsa uzerine yaz.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        experiment_id=args.experiment,
        dry_run=args.dry_run,
        skip_export=args.skip_export,
        export_labels=args.export_labels,
        force=args.force,
    )