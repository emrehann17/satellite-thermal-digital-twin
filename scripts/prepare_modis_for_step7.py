"""
prepare_modis_for_step7.py

Step0F: Step7 (downscaling) icin gerekli MODIS LST mean/std girdisini,
Kozan-disi bir deney icin (or. manavgat_2021) TAMAMEN namespaced sekilde
hazirlar.

Cikti:
    outputs/experiments/<experiment_id>/data/modis/modis_lst_mean_celsius.tif
    outputs/experiments/<experiment_id>/data/modis/modis_lst_std_celsius.tif
    outputs/experiments/<experiment_id>/data/modis/modis_metadata.json

ONEMLI:
    - Kozan'in legacy data/modis/ dizini bu script tarafindan ASLA
      okunmaz/yazilmaz.
    - MODIS urunu/kaynagi ve DN->Celsius donusum mantigi, projenin zaten
      var olan src/step2_modis_5year_mean.py:process_summer_mean()
      fonksiyonu REUSE edilerek elde edilir -- yeni bir MODIS agregasyon
      mantigi YAZILMAZ. process_summer_mean() start/end parametrelerine
      deneyin PREDICTOR PENCERESI (or. Manavgat icin 2021-06-01 ->
      2021-07-27) verilir; yani mean/std, coklu-yil bir "baseline" degil,
      SECILI deneyin predictor penceresi icindeki MODIS gunluk sahnelerinin
      ortalamasi/standart sapmasidir.
    - GEE'nin senkron getPixels boyut limitini asan export'lar icin,
      scripts/run_predictors_only.py:export_image_direct_or_tiled()
      (direct -> 2x2 -> 4x4 -> 6x6 -> 8x8 tiled fallback) REUSE edilir --
      yeni bir tiled export mantigi YAZILMAZ.
    - Bu script yalnizca MODIS girdisini HAZIRLAR; Step7B/7C/7D/7E'yi
      CALISTIRMAZ, Step8'i KESINLIKLE CALISTIRMAZ, hicbir model EGITMEZ.

CLI:
    python scripts/prepare_modis_for_step7.py --experiment manavgat_2021 --dry-run
    python scripts/prepare_modis_for_step7.py --experiment manavgat_2021 --export --force
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.experiment_context import build_experiment_context, get_region, log_context_summary
from core.io_utils import setup_logger

log, log_file = setup_logger("prepare_modis_for_step7")

BASE_DIR = _PROJECT_ROOT
MODIS_EXPORT_SCALE = 1000  # MOD11A1 native ~1 km

_LEGACY_MODIS_DIR = (BASE_DIR / "data" / "modis").resolve()


class ModisPrepError(SystemExit):
    """Fail-fast error for this script (diğer step'lerle aynı konvansiyon)."""


def resolve_modis_output_paths(ctx: dict) -> dict:
    """
    Bir deney icin MODIS cikti yollarini cozer.

    kozan_2023 icin (ctx["is_kozan"]) legacy data/modis/ dondurur (bu
    script'in Kozan icin kullanilmasi ONERILMEZ -- Kozan zaten
    scripts/main.py -> src/step2_modis_5year_mean.py + Step4/4B Drive
    export zinciriyle hazirlanir). Kozan-disi deneyler icin TAMAMEN
    namespaced (outputs/experiments/<experiment_id>/data/modis/) doner.
    """
    if ctx["is_kozan"]:
        modis_dir = BASE_DIR / "data" / "modis"
    else:
        modis_dir = ctx["modis_input_dir"]
    return {
        "modis_dir": modis_dir,
        "mean_path": modis_dir / "modis_lst_mean_celsius.tif",
        "std_path": modis_dir / "modis_lst_std_celsius.tif",
        "metadata_path": modis_dir / "modis_metadata.json",
    }


def _assert_paths_are_safely_namespaced(ctx: dict, paths: dict) -> None:
    """
    GÜVENLİK KONTROLÜ (Kozan-dışı deneyler için ZORUNLU): tüm MODIS çıktı
    yolları outputs/experiments/<experiment_id>/ altında olmalı ve legacy
    Kozan data/modis/ dizinine ASLA düşmemelidir.
    """
    experiment_id = ctx["experiment_id"]
    experiments_root = (BASE_DIR / "outputs" / "experiments" / experiment_id).resolve()

    for key in ("modis_dir", "mean_path", "std_path", "metadata_path"):
        resolved = Path(paths[key]).resolve()
        if resolved == _LEGACY_MODIS_DIR or _LEGACY_MODIS_DIR in resolved.parents:
            raise ModisPrepError(
                f"GÜVENLİK İHLALİ: '{experiment_id}' deneyi için '{key}' yolu "
                f"({resolved}) Kozan'ın legacy paylaşılan MODIS dizinine "
                f"({_LEGACY_MODIS_DIR}) düşüyor. İşlem DURDURULDU."
            )
        if resolved != experiments_root and experiments_root not in resolved.parents:
            raise ModisPrepError(
                f"GÜVENLİK İHLALİ: '{experiment_id}' deneyi için '{key}' yolu "
                f"({resolved}) outputs/experiments/{experiment_id}/ dışında. "
                "İşlem DURDURULDU."
            )


def prepare_modis_for_step7(ctx: dict, force: bool = False) -> dict:
    """
    Secili deney icin MODIS LST mean/std'yi (predictor penceresi icin)
    export eder ve metadata JSON'unu yazar.

    Kozan-disi deneyler icin (ctx["is_kozan"]=False) BASARISIZLIK durumunda
    (GEE erisimi yok, export basarisiz, vb.) HATA FIRLATIR (bu artik Step7
    icin ZORUNLU bir girdi -- bkz. run_step7_downscaling_only.py'nin
    fail-fast MODIS kontrolu). Kozan icin bu fonksiyon KULLANILMAMALIDIR
    (bkz. resolve_modis_output_paths docstring); yine de cagrilirsa
    basarisizlik durumunda da hata fırlatir (best-effort/sessiz yutma YOK).
    """
    paths = resolve_modis_output_paths(ctx)
    if not ctx["is_kozan"]:
        _assert_paths_are_safely_namespaced(ctx, paths)

    mean_path, std_path, metadata_path = paths["mean_path"], paths["std_path"], paths["metadata_path"]

    if mean_path.exists() and std_path.exists() and not force:
        log.info("MODIS zaten mevcut, atlanıyor: %s, %s", mean_path, std_path)
        return {
            "status": "already_exists", "mean_path": str(mean_path), "std_path": str(std_path),
            "metadata_path": str(metadata_path) if metadata_path.exists() else None,
        }

    import ee
    from core.config import EXPORT_CRS, GEE_PROJECT, MODIS_COLLECTION
    from core.gee_utils import init_gee
    import src.step2_modis_5year_mean as step2
    from scripts.run_predictors_only import export_image_direct_or_tiled

    init_gee(GEE_PROJECT)
    region = get_region(ctx)
    paths["modis_dir"].mkdir(parents=True, exist_ok=True)

    log.info(
        "MODIS LST mean/std hesaplanıyor: region=%s, pencere=%s -> %s "
        "(deneyin PREDICTOR penceresi; çoklu-yıl baseline DEĞİL)",
        ctx["region_key"], ctx["predictor_start_date"], ctx["predictor_end_date"],
    )
    summer_image, source_meta = step2.process_summer_mean(
        region, ctx["region_key"],
        ctx["predictor_start_date"], ctx["predictor_end_date"],
    )
    mean_image = summer_image.select("LST_Celsius_SummerMean").rename("modis_lst_mean_celsius")
    std_image = summer_image.select("LST_Celsius_SummerStdDev").rename("modis_lst_std_celsius")

    tiles_dir_mean = paths["modis_dir"] / "_tiles" / "modis_lst_mean"
    tiles_dir_std = paths["modis_dir"] / "_tiles" / "modis_lst_std"

    mean_result = export_image_direct_or_tiled(
        mean_image, mean_path, region, scale=MODIS_EXPORT_SCALE, crs=EXPORT_CRS,
        label="modis_lst_mean", force=force, tiles_dir=tiles_dir_mean,
    )
    std_result = export_image_direct_or_tiled(
        std_image, std_path, region, scale=MODIS_EXPORT_SCALE, crs=EXPORT_CRS,
        label="modis_lst_std", force=force, tiles_dir=tiles_dir_std,
    )
    log.info(
        "MODIS export tamamlandı: mean_transport=%s std_transport=%s",
        mean_result["transport"], std_result["transport"],
    )

    metadata = {
        "experiment_id": ctx["experiment_id"],
        "region_key": ctx["region_key"],
        "predictor_start_date": ctx["predictor_start_date"],
        "predictor_end_date": ctx["predictor_end_date"],
        "modis_product": MODIS_COLLECTION,
        "modis_input_band": source_meta.get("input_band", "LST_Day_1km"),
        "dn_to_celsius_formula": "LST_Day_1km * 0.02 - 273.15",
        "aggregation_window": (
            "experiment predictor window (single-season mean/std over daily "
            "MODIS scenes within the window), NOT a multi-year baseline"
        ),
        "output_files": {
            "mean": str(mean_path),
            "std": str(std_path),
        },
        "scale_meters": MODIS_EXPORT_SCALE,
        "crs": EXPORT_CRS,
        "export_transport": {
            "mean": mean_result["transport"],
            "std": std_result["transport"],
        },
        "tile_grid": {
            "mean": list(mean_result["tile_grid"]) if mean_result["tile_grid"] else None,
            "std": list(std_result["tile_grid"]) if std_result["tile_grid"] else None,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "exported",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("MODIS metadata yazıldı: %s", metadata_path)

    return {
        "status": "exported",
        "mean_path": str(mean_path), "std_path": str(std_path),
        "metadata_path": str(metadata_path),
        "mean_transport": mean_result["transport"], "std_transport": std_result["transport"],
    }


def _log_dry_run(ctx: dict, paths: dict) -> None:
    log.info("[dry-run] experiment_id: %s", ctx["experiment_id"])
    log.info("[dry-run] region_key: %s", ctx["region_key"])
    log.info(
        "[dry-run] predictor window (MODIS mean/std bu pencere için hesaplanacak): %s -> %s",
        ctx["predictor_start_date"], ctx["predictor_end_date"],
    )
    log.info("[dry-run] Planlanan MODIS çıktı dizini: %s", paths["modis_dir"])
    log.info("  %s %s", "[VAR]" if paths["mean_path"].exists() else "[EKSİK]", paths["mean_path"])
    log.info("  %s %s", "[VAR]" if paths["std_path"].exists() else "[EKSİK]", paths["std_path"])
    log.info(
        "  %s %s", "[VAR]" if paths["metadata_path"].exists() else "[EKSİK]", paths["metadata_path"],
    )
    log.info("[dry-run] Hiçbir GEE export/dosya yazma ÇALIŞTIRILMADI.")


def main(experiment_id: str = "manavgat_2021", dry_run: bool = False, export: bool = False, force: bool = False) -> dict:
    ctx = build_experiment_context(experiment_id)
    log_context_summary(ctx, log)

    if ctx["is_kozan"]:
        log.warning(
            "'%s' Kozan'dır -- bu script Kozan-dışı deneyler için tasarlanmıştır "
            "(Kozan kendi legacy MODIS hazırlığını scripts/main.py -> "
            "src/step2_modis_5year_mean.py üzerinden yapar). Yine de devam "
            "ediliyor, ancak çıktılar legacy data/modis/'a gidecektir.",
            experiment_id,
        )

    paths = resolve_modis_output_paths(ctx)
    if not ctx["is_kozan"]:
        _assert_paths_are_safely_namespaced(ctx, paths)

    if dry_run:
        _log_dry_run(ctx, paths)
        return {"experiment_id": experiment_id, "ran": False, "reason": "dry_run"}

    if not export:
        raise ModisPrepError(
            "Ne --export ne --dry-run verildi; hangi modda çalışılacağı belirsiz."
        )

    result = prepare_modis_for_step7(ctx, force=force)
    log.info("TAMAMLANDI: %s", result)
    return {"experiment_id": experiment_id, "ran": True, "result": result}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step0F: Step7 downscaling için gerekli MODIS LST "
        "mean/std girdisini, Kozan-dışı bir deney için (namespaced) "
        "hazırlar. Step7B/7C/7D/7E'yi ÇALIŞTIRMAZ, Step8'i ÇALIŞTIRMAZ, "
        "model EĞİTMEZ."
    )
    parser.add_argument("--experiment", type=str, default="manavgat_2021")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Hiçbir şey çalıştırma; planlanan MODIS çıktı yollarını + var/yok durumunu bas.",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="MODIS LST mean/std'yi GEE'den export eder ve metadata yazar.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="MODIS çıktıları zaten varsa üzerine yaz.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        experiment_id=args.experiment,
        dry_run=args.dry_run,
        export=args.export,
        force=args.force,
    )