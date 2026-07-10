"""
prepare_dem_for_experiment.py

Step0G: Kozan-disi bir deney (or. manavgat_2021) icin, o deneyin KENDI AOI'sini
kapsayan DEM (elevation) ve slope urunlerini, Step5 referans gridine hizali
sekilde, TAMAMEN namespaced olarak hazirlar.

NEDEN GEREKLI:
    Step7B, DEM/slope'u su ana kadar "paylasilan, salt-okunur" (Option B)
    varsayimiyla data/dem/elevation.tif + data/dem/slope.tif'ten okuyordu.
    Ancak bu dosyalar aslinda yalnizca KOZAN'in AOI'sini kapsiyor (Dogu
    Akdeniz/Adana civari); Manavgat (Antalya civari) ile COGRAFI OLARAK HIC
    ORTUSMUYOR. Bu yuzden Step7B'nin hizalama diagnostikleri
    elevation/slope icin overlap_with_target=0 raporluyor ve
    final_sample_count=0 ile fail-fast oluyor (bkz. Step7B'nin kendi
    hizalama/overlap diagnostikleri).

    Cozum: Option B'yi (paylasilan DEM) Manavgat icin BIRAKIYORUZ ve onun
    yerine Manavgat'a OZEL, kendi AOI'sini kapsayan bir DEM/slope cifti
    export ediyoruz -- Kozan'in data/dem/'ine ASLA yazmadan/okumadan.

Cikti:
    outputs/experiments/<experiment_id>/data/dem/elevation.tif
    outputs/experiments/<experiment_id>/data/dem/slope.tif
    outputs/experiments/<experiment_id>/data/dem/dem_metadata.json

Reuse (yeni bir DEM/slope hesaplama mantigi YAZILMAZ):
    - src/step2b_dem.py:prepare_dem_products() -- ayni DEM kaynagi
      (Copernicus DEM GLO-30, fallback USGS SRTMGL1 003) + ee.Terrain.slope
      hesaplamasi (native projeksiyona sabitlenmis elevation uzerinden,
      Step2B'nin kendi "slope fix" mantigiyla BIREBIR AYNI).
    - scripts/run_predictors_only.py:export_image_direct_or_tiled() -- GEE
      boyut limiti asilirsa tiled fallback.

Hizalama:
    Export sonrasi, elevation/slope Step5 referans gridiyle (Manavgat
    current_period_median_celsius.tif) piksel-piksel (CRS/genislik/
    yukseklik/transform) birebir eslesmiyorsa, rasterio.warp.reproject ile
    BILINEAR resampling kullanilarak referans gride hizalanir (sürekli/
    continuous veri oldugu icin bilinear kabul edilebilir -- kategorik
    landcover'daki gibi nearest-neighbor GEREKMEZ).

CLI:
    python scripts/prepare_dem_for_experiment.py --experiment manavgat_2021 --dry-run
    python scripts/prepare_dem_for_experiment.py --experiment manavgat_2021 --export --force
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

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from core.experiment_context import build_experiment_context, get_region, log_context_summary
from core.io_utils import setup_logger

log, log_file = setup_logger("prepare_dem_for_experiment")

BASE_DIR = _PROJECT_ROOT
DEM_EXPORT_SCALE = 30  # Landsat ile aynı grid ölçeği

_LEGACY_DEM_DIR = (BASE_DIR / "data" / "dem").resolve()


class DemPrepError(SystemExit):
    """Fail-fast error for this script (diğer step'lerle aynı konvansiyon)."""


def resolve_dem_output_paths(ctx: dict) -> dict:
    """
    Bir deney icin DEM cikti yollarini cozer.

    kozan_2023 icin (ctx["is_kozan"]) legacy data/dem/ dondurur (bu script
    Kozan icin KULLANILMAMALIDIR -- Kozan zaten scripts/main.py ->
    src/step2b_dem.py + Step4/4B Drive export zinciriyle hazirlanir).
    Kozan-disi deneyler icin TAMAMEN namespaced
    (outputs/experiments/<experiment_id>/data/dem/) doner.
    """
    if ctx["is_kozan"]:
        dem_dir = BASE_DIR / "data" / "dem"
    else:
        dem_dir = ctx["dem_input_dir"]
    return {
        "dem_dir": dem_dir,
        "elevation_path": dem_dir / "elevation.tif",
        "slope_path": dem_dir / "slope.tif",
        "metadata_path": dem_dir / "dem_metadata.json",
    }


def _assert_paths_are_safely_namespaced(ctx: dict, paths: dict) -> None:
    """
    GÜVENLİK KONTROLÜ (Kozan-dışı deneyler için ZORUNLU): tüm DEM çıktı
    yolları outputs/experiments/<experiment_id>/ altında olmalı ve legacy
    Kozan data/dem/ dizinine ASLA düşmemelidir.
    """
    experiment_id = ctx["experiment_id"]
    experiments_root = (BASE_DIR / "outputs" / "experiments" / experiment_id).resolve()

    for key in ("dem_dir", "elevation_path", "slope_path", "metadata_path"):
        resolved = Path(paths[key]).resolve()
        if resolved == _LEGACY_DEM_DIR or _LEGACY_DEM_DIR in resolved.parents:
            raise DemPrepError(
                f"GÜVENLİK İHLALİ: '{experiment_id}' deneyi için '{key}' yolu "
                f"({resolved}) Kozan'ın legacy paylaşılan DEM dizinine "
                f"({_LEGACY_DEM_DIR}) düşüyor. İşlem DURDURULDU."
            )
        if resolved != experiments_root and experiments_root not in resolved.parents:
            raise DemPrepError(
                f"GÜVENLİK İHLALİ: '{experiment_id}' deneyi için '{key}' yolu "
                f"({resolved}) outputs/experiments/{experiment_id}/ dışında. "
                "İşlem DURDURULDU."
            )


def _reproject_to_reference(source_path: Path, reference_path: Path, out_path: Path) -> dict:
    """
    Bir rasterı (elevation veya slope) referans (Step5) gridine BILINEAR
    resampling ile hizalar ve SONUCU HER ZAMAN `out_path`'e yazar.

    ÖNEMLİ (bug fix): Kaynak raster referans gridle zaten birebir eşleşse
    bile (same_grid=True), dosya `out_path`'e KOPYALANIR -- yalnızca
    diagnostics hesaplayıp erken dönmek YETERLİ DEĞİLDİR, çünkü çağıran
    kod (prepare_dem_for_experiment) her zaman `out_path`'in (canonical
    elevation.tif/slope.tif) var olmasını bekler. Önceki implementasyon
    same_grid=True durumunda `out_path`'i HİÇ YAZMIYORDU -- bu da export
    "başarılı" göründüğü halde Step7'nin DEM'i "[EKSİK]" olarak görmesine
    yol açan asıl sebepti.
    """
    with rasterio.open(reference_path) as ref:
        ref_w, ref_h, ref_crs, ref_t = ref.width, ref.height, ref.crs, ref.transform

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(source_path) as src:
        same_grid = (
            src.crs == ref_crs and src.width == ref_w
            and src.height == ref_h and src.transform == ref_t
        )
        if same_grid:
            import shutil
            shutil.copyfile(source_path, out_path)
            arr = src.read(1, masked=True)
            valid = int((~np.ma.getmaskarray(arr)).sum())
            return {
                "aligned": False, "reason": "already_matches_reference_grid_copied",
                "valid_pixel_count": valid, "total_pixel_count": int(ref_w * ref_h),
            }

        src_nodata = src.nodata if src.nodata is not None else float("nan")
        dst = np.full((ref_h, ref_w), src_nodata, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_t, dst_crs=ref_crs,
            dst_nodata=src_nodata,
            resampling=Resampling.bilinear,
        )

    out_profile = {
        "driver": "GTiff", "width": ref_w, "height": ref_h, "count": 1,
        "dtype": "float32", "crs": ref_crs, "transform": ref_t,
        "nodata": src_nodata, "compress": "deflate",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **out_profile) as dst_ds:
        dst_ds.write(dst, 1)

    if isinstance(src_nodata, float) and np.isnan(src_nodata):
        valid_mask = np.isfinite(dst)
    else:
        valid_mask = (dst != src_nodata)
    return {
        "aligned": True, "reason": "reprojected_bilinear_to_reference_grid",
        "valid_pixel_count": int(valid_mask.sum()), "total_pixel_count": int(ref_w * ref_h),
    }


def prepare_dem_for_experiment(ctx: dict, force: bool = False) -> dict:
    """
    Secili deney icin DEM elevation + slope'u export eder, Step5 referans
    gridine hizalar, ve metadata JSON'unu yazar.

    Basarisizlik durumunda (GEE erisimi yok, export basarisiz, referans grid
    bulunamadi, vb.) HATA FIRLATIR -- bu artik Step7 icin ZORUNLU bir girdi.
    """
    paths = resolve_dem_output_paths(ctx)
    if not ctx["is_kozan"]:
        _assert_paths_are_safely_namespaced(ctx, paths)

    elevation_path = paths["elevation_path"]
    slope_path = paths["slope_path"]
    metadata_path = paths["metadata_path"]

    if elevation_path.exists() and slope_path.exists() and not force:
        log.info("DEM zaten mevcut, atlanıyor: %s, %s", elevation_path, slope_path)
        return {
            "status": "already_exists", "elevation_path": str(elevation_path),
            "slope_path": str(slope_path),
            "metadata_path": str(metadata_path) if metadata_path.exists() else None,
        }

    reference_path = ctx["step5_output_dir"] / "current_period_median_celsius.tif"
    if not reference_path.exists():
        raise DemPrepError(
            f"Step5 referans gridi bulunamadı: {reference_path}. Önce Step5'i "
            "(namespaced) çalıştırın."
        )

    import ee
    from core.config import DEM_COLLECTION, DEM_FALLBACK_DATASET, EXPORT_CRS, GEE_PROJECT
    from core.gee_utils import init_gee
    import src.step2b_dem as step2b
    from scripts.run_predictors_only import export_image_direct_or_tiled

    init_gee(GEE_PROJECT)
    region = get_region(ctx)
    paths["dem_dir"].mkdir(parents=True, exist_ok=True)

    log.info(
        "DEM (elevation + slope) hazırlanıyor: experiment=%s, region=%s",
        ctx["experiment_id"], ctx["region_key"],
    )
    dem_image, source_meta = step2b.prepare_dem_products(region, ctx["region_key"])
    elevation_image = dem_image.select("elevation")
    slope_image = dem_image.select("slope")

    raw_dir = paths["dem_dir"] / "_raw"
    raw_elevation_path = raw_dir / "elevation_raw.tif"
    raw_slope_path = raw_dir / "slope_raw.tif"
    tiles_dir_elev = paths["dem_dir"] / "_tiles" / "elevation"
    tiles_dir_slope = paths["dem_dir"] / "_tiles" / "slope"

    elevation_result = export_image_direct_or_tiled(
        elevation_image, raw_elevation_path, region, scale=DEM_EXPORT_SCALE, crs=EXPORT_CRS,
        label="dem_elevation", force=force, tiles_dir=tiles_dir_elev,
    )
    slope_result = export_image_direct_or_tiled(
        slope_image, raw_slope_path, region, scale=DEM_EXPORT_SCALE, crs=EXPORT_CRS,
        label="dem_slope", force=force, tiles_dir=tiles_dir_slope,
    )
    log.info(
        "DEM export tamamlandı: elevation_transport=%s slope_transport=%s",
        elevation_result["transport"], slope_result["transport"],
    )

    log.info("Referans gride (Step5) hizalanıyor (bilinear): %s", reference_path)
    elevation_align = _reproject_to_reference(raw_elevation_path, reference_path, elevation_path)
    slope_align = _reproject_to_reference(raw_slope_path, reference_path, slope_path)
    log.info(
        "Hizalama tamamlandı: elevation valid=%d/%d, slope valid=%d/%d",
        elevation_align["valid_pixel_count"], elevation_align["total_pixel_count"],
        slope_align["valid_pixel_count"], slope_align["total_pixel_count"],
    )

    # FAIL-FAST DOĞRULAMA: hizalama/kopyalama adımından sonra canonical
    # final dosyaların (elevation.tif/slope.tif) GERÇEKTEN diskte var
    # olduğunu doğrula. Bu, "export/hizalama logu başarılı görünüyor ama
    # final dosya aslında yok" tarzı sessiz başarısızlıkları önler --
    # Step7 dry-run'ın [EKSİK] göstermesini BEKLEMEK yerine burada,
    # export sırasında yakalanır.
    final_outputs_exist = elevation_path.exists() and slope_path.exists()
    if not final_outputs_exist:
        raise DemPrepError(
            "DEM export/hizalama tamamlandı ama canonical final dosyalar "
            f"oluşmadı: elevation.tif var={elevation_path.exists()} "
            f"({elevation_path}), slope.tif var={slope_path.exists()} "
            f"({slope_path}). Ham (_raw) dosyalar mevcut olabilir "
            f"({raw_elevation_path}, {raw_slope_path}) ancak Step7 bunları "
            "DEĞİL, yalnızca canonical isimleri kontrol eder."
        )
    log.info(
        "Canonical final DEM dosyaları doğrulandı: elevation=%s (%s), slope=%s (%s)",
        elevation_path.exists(), elevation_path, slope_path.exists(), slope_path,
    )

    with rasterio.open(reference_path) as ref:
        ref_w, ref_h, ref_crs, ref_t = ref.width, ref.height, ref.crs, ref.transform

    metadata = {
        "experiment_id": ctx["experiment_id"],
        "region_key": ctx["region_key"],
        "dem_source": {
            "preferred_dataset": DEM_COLLECTION,
            "fallback_dataset": DEM_FALLBACK_DATASET,
            "used_fallback": source_meta.get("used_fallback"),
            "dataset_used": source_meta.get("dataset"),
            "input_band": source_meta.get("input_band"),
        },
        "slope_method": "ee.Terrain.slope (native DEM projection, per Step2B slope-fix logic)",
        "raw_elevation_path": str(raw_elevation_path),
        "raw_slope_path": str(raw_slope_path),
        "final_elevation_path": str(elevation_path),
        "final_slope_path": str(slope_path),
        "final_outputs_exist": final_outputs_exist,
        "output_files": {
            "elevation": str(elevation_path),
            "slope": str(slope_path),
        },
        "reference_raster": str(reference_path),
        "crs": str(ref_crs),
        "width": ref_w,
        "height": ref_h,
        "transform": [ref_t.a, ref_t.b, ref_t.c, ref_t.d, ref_t.e, ref_t.f],
        "scale_meters": DEM_EXPORT_SCALE,
        "export_transport": {
            "elevation": elevation_result["transport"],
            "slope": slope_result["transport"],
        },
        "alignment": {
            "elevation": elevation_align,
            "slope": slope_align,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "exported",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("DEM metadata yazıldı: %s", metadata_path)

    return {
        "status": "exported",
        "elevation_path": str(elevation_path), "slope_path": str(slope_path),
        "metadata_path": str(metadata_path),
        "elevation_transport": elevation_result["transport"],
        "slope_transport": slope_result["transport"],
    }


def _log_dry_run(ctx: dict, paths: dict) -> None:
    log.info("[dry-run] experiment_id: %s", ctx["experiment_id"])
    log.info("[dry-run] region_key: %s", ctx["region_key"])
    reference_path = ctx["step5_output_dir"] / "current_period_median_celsius.tif"
    log.info(
        "[dry-run] Referans grid (Step5): %s (%s)",
        reference_path, "[VAR]" if reference_path.exists() else "[EKSİK]",
    )
    log.info("[dry-run] Planlanan DEM çıktı dizini: %s", paths["dem_dir"])
    log.info("  %s %s", "[VAR]" if paths["elevation_path"].exists() else "[EKSİK]", paths["elevation_path"])
    log.info("  %s %s", "[VAR]" if paths["slope_path"].exists() else "[EKSİK]", paths["slope_path"])
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
            "(Kozan kendi legacy DEM hazırlığını scripts/main.py -> "
            "src/step2b_dem.py üzerinden yapar). Yine de devam ediliyor, ancak "
            "çıktılar legacy data/dem/'e gidecektir.",
            experiment_id,
        )

    paths = resolve_dem_output_paths(ctx)
    if not ctx["is_kozan"]:
        _assert_paths_are_safely_namespaced(ctx, paths)

    if dry_run:
        _log_dry_run(ctx, paths)
        return {"experiment_id": experiment_id, "ran": False, "reason": "dry_run"}

    if not export:
        raise DemPrepError(
            "Ne --export ne --dry-run verildi; hangi modda çalışılacağı belirsiz."
        )

    result = prepare_dem_for_experiment(ctx, force=force)
    log.info("TAMAMLANDI: %s", result)
    return {"experiment_id": experiment_id, "ran": True, "result": result}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step0G: Kozan-dışı bir deney için (namespaced) DEM "
        "elevation + slope hazırlar, Step5 referans gridine hizalar. "
        "Step7B/7C/7D/7E'yi ÇALIŞTIRMAZ, Step8'i ÇALIŞTIRMAZ, model EĞİTMEZ."
    )
    parser.add_argument("--experiment", type=str, default="manavgat_2021")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Hiçbir şey çalıştırma; planlanan DEM çıktı yollarını + var/yok durumunu bas.",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="DEM elevation/slope'u GEE'den export eder, referans gride hizalar, metadata yazar.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="DEM çıktıları zaten varsa üzerine yaz.",
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