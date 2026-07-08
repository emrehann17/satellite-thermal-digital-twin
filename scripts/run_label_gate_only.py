"""
run_label_gate_only.py

Step0C: guvenli, deney-farkinda (experiment-aware) "gate-only" calistirici.

Modelleme (Step7A-Step8E) oncesinde gerekli MINIMUM label/gate zincirini
calistirir:
    [Kozan disi deneyler icin] Step6A gate-input hazirlama (referans grid +
        hizalanmis landcover, namespaced, TERMAL PREDICTOR DEGIL)
    -> [opsiyonel] raw MCD64A1 BurnDate export (Step6'nin canonical export'u)
    -> Step6B burned-landcover gate

Step1-Step8'in geri kalanini (Step2-Step5 termal on-isleme, Step7, Step8)
KESINLIKLE CALISTIRMAZ.

IKI FARKLI DAVRANIS MODU
-------------------------
kozan_2023:
    LEGACY davranis BIREBIR korunur. outputs/validation/labels/ altindaki
    paylasilan dosyalar kullanilir/yazilir (Step6/Step6B'nin varsayilan
    kesif mantigi). Hicbir namespaced yol hesaplanmaz.

kozan_2023 DISINDAKI her deney (or. manavgat_2021):
    TAMAMEN NAMESPACED calisir:
        outputs/experiments/<experiment_id>/gate_inputs/...     (Step6A)
        outputs/experiments/<experiment_id>/validation/labels/... (Step6/Step6B)
    Bu deneyler icin, her calistirmadan ONCE, hesaplanan TUM yollarin
    outputs/experiments/<experiment_id>/ altinda kaldigi VE legacy Kozan
    dizini (outputs/validation/labels/) ile CAKISMADIGI dogrulanir
    (bkz. _assert_paths_are_safely_namespaced). Herhangi bir ihlalde
    LabelGateRunnerError firlatilir -- HICBIR export/gate calismaz.

Bu script HICBIR ZAMAN, bir deney etiketi altinda (or. manavgat_2021)
SESSIZCE Kozan verisini calistirmaz veya Kozan'in paylasilan dosyalarina
yazmaz.

CLI:
    python scripts/run_label_gate_only.py --experiment kozan_2023 --skip-export --force
    python scripts/run_label_gate_only.py --experiment kozan_2023 --export-labels --force
    python scripts/run_label_gate_only.py --experiment manavgat_2021 --dry-run
    python scripts/run_label_gate_only.py --experiment manavgat_2021 --export-labels --force

Flags:
    --dry-run         Hicbir sey CALISTIRMAZ; Step0 ozetini + TUM planlanan
                      yollari basar.
    --skip-export     Raw BurnDate export'unu ATLA (varsayilan davranis);
                      gate mevcut mcd64a1_raw.tif dosyasini kullanir.
    --export-labels   Gate'ten ONCE raw BurnDate export'unu (GEE) calistir.
                      Kozan-disi deneylerde ONCE Step6A gate-input hazirlama
                      da calisir (referans grid + landcover, eksikse veya
                      --force ise).
    --force           Gate (ve Kozan-disi deneylerde gate-input) ciktilari
                      zaten varsa uzerine yaz.
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

BASE_DIR = _PROJECT_ROOT
LEGACY_VALIDATION_LABEL_DIR = (BASE_DIR / "outputs" / "validation" / "labels").resolve()


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


def _legacy_kozan_paths() -> dict:
    return {
        "gate_inputs_dir": None,
        "reference_path": (BASE_DIR / "outputs" / "step5" / "current_period_median_celsius.tif"),
        "landcover_source_path": (BASE_DIR / "data" / "landcover" / "landcover_esa_worldcover_v200.tif"),
        "landcover_aligned_path": (
            BASE_DIR / "data" / "landcover" / "landcover_esa_worldcover_v200_aligned_to_landsat.tif"
        ),
        "metadata_path": None,
        "raw_path": LEGACY_VALIDATION_LABEL_DIR / "mcd64a1_raw.tif",
        "binary_path": LEGACY_VALIDATION_LABEL_DIR / "mcd64a1_burned.tif",
        "gate_output_dir": LEGACY_VALIDATION_LABEL_DIR,
    }


def _namespaced_paths(experiment_id: str) -> dict:
    """Kozan-disi deneyler icin TUM planlanan yollari (namespaced) hesaplar.

    Hicbir sey OLUSTURMAZ/YAZMAZ -- yalnizca yol hesaplar (dry-run ve
    safety-check tarafindan da kullanilir).
    """
    from src.step6a_prepare_gate_inputs import get_gate_input_paths

    gate_inputs = get_gate_input_paths(experiment_id)
    label_dir = get_experiment_output_root(experiment_id) / "validation" / "labels"
    return {
        "gate_inputs_dir": gate_inputs["gate_inputs_dir"],
        "reference_path": gate_inputs["reference_path"],
        "landcover_source_path": gate_inputs["landcover_source_path"],
        "landcover_aligned_path": gate_inputs["landcover_aligned_path"],
        "metadata_path": gate_inputs["metadata_path"],
        "raw_path": label_dir / "mcd64a1_raw.tif",
        "binary_path": label_dir / "mcd64a1_burned.tif",
        "gate_output_dir": label_dir,
    }


def _assert_paths_are_safely_namespaced(experiment_id: str, paths: dict) -> None:
    """
    GUVENLIK KONTROLU (Kozan-disi deneyler icin ZORUNLU):
        1) Hicbir hesaplanan yol legacy outputs/validation/labels/ altina
           DUSMEMELIDIR (Kozan'in paylasilan dosyalarinin uzerine ASLA
           yazilmaz/okunmaz).
        2) Yazilabilir TUM yollar outputs/experiments/<experiment_id>/
           altinda OLMALIDIR.
    Ihlal varsa LabelGateRunnerError firlatir; hicbir export/gate calismaz.
    """
    experiments_root = (BASE_DIR / "outputs" / "experiments" / experiment_id).resolve()

    for name, p in paths.items():
        if p is None:
            continue
        resolved = Path(p).resolve()

        if resolved == LEGACY_VALIDATION_LABEL_DIR or LEGACY_VALIDATION_LABEL_DIR in resolved.parents:
            raise LabelGateRunnerError(
                f"GUVENLIK IHLALI: '{experiment_id}' deneyi icin hesaplanan '{name}' "
                f"yolu ({resolved}) Kozan'in legacy paylasilan dizinine "
                f"(outputs/validation/labels/) dusuyor. Bu deney bu dizine ASLA "
                "yazamaz/okuyamaz. Islem DURDURULDU."
            )
        if resolved != experiments_root and experiments_root not in resolved.parents:
            raise LabelGateRunnerError(
                f"GUVENLIK IHLALI: '{experiment_id}' deneyi icin hesaplanan '{name}' "
                f"yolu ({resolved}) outputs/experiments/{experiment_id}/ disinda. "
                "Islem DURDURULDU."
            )


def _log_planned_paths(paths: dict) -> None:
    log.info("  gate_inputs reference (30 m, termal predictor DEGIL): %s", paths["reference_path"])
    log.info("  gate_inputs landcover (kaynak): %s", paths["landcover_source_path"])
    log.info("  gate_inputs landcover (hizali): %s", paths["landcover_aligned_path"])
    if paths.get("metadata_path"):
        log.info("  gate_inputs metadata: %s", paths["metadata_path"])
    log.info("  raw BurnDate (DOY): %s", paths["raw_path"])
    log.info("  binary mask: %s", paths["binary_path"])
    log.info("  gate ciktilari (json/md/csv): %s/burned_landcover_gate.{json,md,csv}", paths["gate_output_dir"])


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

    is_kozan = (experiment_id == "kozan_2023")
    paths = _legacy_kozan_paths() if is_kozan else _namespaced_paths(experiment_id)

    if not is_kozan:
        _assert_paths_are_safely_namespaced(experiment_id, paths)
        if exp["label_start_date"] is None or exp["label_end_date"] is None:
            raise LabelGateRunnerError(
                f"'{experiment_id}' deneyinin label penceresi tanimsiz "
                "(placeholder/disabled deney olabilir). Gate-only calistirilamaz."
            )

    if dry_run:
        log.info("[dry-run] Planlanan yollar:")
        _log_planned_paths(paths)
        log.info("[dry-run] Hicbir export/gate CALISTIRILMADI.")
        return {"experiment_id": experiment_id, "ran": False, "reason": "dry_run", "planned_paths": {
            k: (str(v) if v is not None else None) for k, v in paths.items()
        }}

    if skip_export and export_labels:
        raise LabelGateRunnerError(
            "--skip-export ve --export-labels birlikte verilemez (celiskili)."
        )
    do_export = export_labels and not skip_export

    gate_inputs_result = None
    if not is_kozan:
        # Kozan-disi deneyler icin Step6B'nin ihtiyac duydugu referans grid +
        # hizalanmis landcover HENUZ yoktur (Step5/Step4 calismadi) --
        # bunlari once Step6A ile (namespaced, termal predictor OLMADAN)
        # hazirlamamiz gerekir.
        log.info("Step6A gate-input hazirlama calistiriliyor (namespaced, termal predictor DEGIL)...")
        try:
            from src.step6a_prepare_gate_inputs import prepare_gate_inputs
        except Exception as exc:  # noqa: BLE001
            raise LabelGateRunnerError(
                f"src.step6a_prepare_gate_inputs import edilemedi: {type(exc).__name__}: {exc}"
            ) from exc
        gate_inputs_result = prepare_gate_inputs(experiment_id, force=force)
        log.info("Step6A tamamlandi: %s", gate_inputs_result)

    export_result = None
    if do_export:
        log.info("Raw MCD64A1 BurnDate export calistiriliyor (--export-labels)...")
        try:
            from src.step6_validate_fire_relation import export_raw_mcd64a1_labels
        except Exception as exc:  # noqa: BLE001
            raise LabelGateRunnerError(
                f"src.step6_validate_fire_relation import edilemedi: {type(exc).__name__}: {exc}"
            ) from exc
        export_result = export_raw_mcd64a1_labels(
            also_binary=True,
            experiment_id=None if is_kozan else experiment_id,
        )
        log.info("Raw BurnDate export tamamlandi: %s", export_result["raw_path"])
    else:
        log.info(
            "Raw BurnDate export ATLANDI (--skip-export ya da varsayilan). "
            "Gate mevcut %s dosyasini kullanacak (yoksa gate net bir hata "
            "verecektir).", paths["raw_path"],
        )

    log.info("Step6B burned-landcover gate calistiriliyor...")
    try:
        from src.step6b_burned_landcover_gate import main as run_gate
    except Exception as exc:  # noqa: BLE001
        raise LabelGateRunnerError(
            f"src.step6b_burned_landcover_gate import edilemedi: {type(exc).__name__}: {exc}"
        ) from exc

    if is_kozan:
        # Legacy davranis BIREBIR: hicbir explicit path verilmez, Step6B
        # kendi varsayilan (outputs/validation/labels + outputs/step5 +
        # data/landcover) kesif mantigini kullanir.
        gate_result = run_gate(force=force)
    else:
        gate_result = run_gate(
            output_dir_arg=str(paths["gate_output_dir"]),
            force=force,
            label_raster_arg=str(paths["raw_path"]),
            reference_30m_arg=str(paths["reference_path"]),
            landcover_path_arg=str(paths["landcover_aligned_path"]),
            label_start=exp["label_start_date"],
            label_end=exp["label_end_date"],
        )

    log.info(
        "TAMAMLANDI. Gate karari: %s (burned_count=%s). JSON: %s",
        gate_result["decision"], gate_result["burned_count"], gate_result["json_path"],
    )

    return {
        "experiment_id": experiment_id,
        "ran": True,
        "gate_inputs_result": gate_inputs_result,
        "export_result": export_result,
        "gate_result": gate_result,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step0C: modelleme oncesi minimum label/gate zincirini "
        "(Kozan-disi deneyler icin namespaced Step6A gate-input hazirlama + "
        "opsiyonel raw BurnDate export + Step6B burned-landcover gate) "
        "calistirir. Step1-Step5/Step7/Step8'in geri kalanini CALISTIRMAZ. "
        "kozan_2023 icin legacy davranis BIREBIR korunur; diger deneyler "
        "TAMAMEN namespaced calisir ve Kozan'in paylasilan dosyalarina ASLA "
        "yazmaz/okumaz."
    )
    parser.add_argument("--experiment", type=str, default="kozan_2023")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Hicbir sey calistirma; Step0 ozetini + planlanan TUM yollari bas.",
    )
    parser.add_argument(
        "--skip-export", action="store_true",
        help="Raw BurnDate export'unu atla (varsayilan davranis).",
    )
    parser.add_argument(
        "--export-labels", action="store_true",
        help="Gate'ten once raw BurnDate export'unu (GEE) calistir "
        "(Kozan-disi deneylerde once Step6A gate-input hazirlama da calisir).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Gate (ve Kozan-disi deneylerde gate-input) ciktilari zaten "
        "varsa uzerine yaz.",
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