"""
step6a_prepare_gate_inputs.py

Step6A: gate-only girdi hazirlama (experiment-aware).

AMAC
----
Step6B burned-landcover gate'in ihtiyac duydugu MINIMUM iki rasteri, tam
Step3/Step5 termal pipeline'ini CALISTIRMADAN hazirlar:
    1. AOI'yi kaplayan 30 m'lik bir REFERANS grid rasteri.
    2. Ayni AOI icin ESA WorldCover v200 landcover'i (GATE ICIN 30 m'de
       export edilir, native 10 m'de DEGIL -- bkz. asagida) + referans
       gride nearest-neighbor ile hizalanmis kopyasi.

ONEMLI: Referans raster BIR TERMAL PREDICTOR DEGILDIR. Step6B yalnizca grid
GEOMETRISINE (width/height/CRS/transform) ihtiyac duyar -- piksel DEGERLERINI
hic okumaz (gate hesaplamasi label + landcover rasterlarindan yapilir). Bu
yuzden ee.Image.constant(1) ile sabit-degerli, hizli, ucuz bir raster export
edilir. Kozan'in gercek Step5 LST referansindan farkli olarak bu dosya HICBIR
bilimsel/termal anlam tasimaz; sadece gate icin bir grid tanimidir.

WORLDCOVER COZUNURLUGU (bug fix): ESA WorldCover v200'un NATIVE cozunurlugu
~10 m'dir. Genis AOI'ler (or. Bejis: ~78 km x 52 km) icin 10 m'de export
etmek GEE'nin senkron getPixels boyut limitini (~50 MB) asar (bkz. hata
raporu: "Total request size (81561538 bytes) must be <= 50331648 bytes").
Bu SADECE bir gate-diagnostic girdisi oldugu ve zaten referans gride
(30 m) nearest-neighbor ile hizalandigi icin, kaynak export'u da GATE ICIN
30 m'de yapilir (native 10 m degil) -- run_predictors_only.py'nin
export_image_direct_or_tiled() (direct -> 2x2 -> 4x4 -> 6x6 -> 8x8 tiled
fallback) fonksiyonu REUSE edilerek, boyut limiti asilsa bile guvenli
sekilde tamamlanir. Bu, Manavgat icin MODIS/DEM export'larinda zaten
kullanilan AYNI guvenli desendir -- yeni bir tiled export mantigi
YAZILMAZ.

Kategorik landcover ASLA bilinear ile resample edilmez -- hizalama icin
Step8A'nin prepare_aligned_landcover'i (nearest-neighbor) reuse edilir; iki
farkli/divergent hizalama implementasyonu OLUSMAZ.

CIKTILAR (namespaced; legacy Kozan/Manavgat yollarina ASLA yazilmaz):
    outputs/experiments/<experiment_id>/gate_inputs/reference_30m.tif
    outputs/experiments/<experiment_id>/gate_inputs/landcover_esa_worldcover_v200.tif
    outputs/experiments/<experiment_id>/gate_inputs/landcover_esa_worldcover_v200_aligned_to_reference.tif
    outputs/experiments/<experiment_id>/gate_inputs/gate_inputs_metadata.json

CLI:
    python src/step6a_prepare_gate_inputs.py --experiment manavgat_2021
    python src/step6a_prepare_gate_inputs.py --experiment manavgat_2021 --force

GEE ortami gerektirir (earthengine-api + geemap + auth), tipki Step6 gibi.
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

from core.config import EXPORT_CRS, GEE_PROJECT
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from core.regions import (
    get_active_experiment,
    get_experiment_output_root,
    get_region_for_experiment,
)

BASE_DIR = PROJECT_ROOT
log, log_file = setup_logger("step6a_prepare_gate_inputs")

# Gate referans gridinin cozunurlugu: projenin 30 m Landsat konvansiyonuyla
# ayni (Step8A block-size hesabi STEP8A_REFERENCE_PIXEL_SIZE_M=30 varsayar).
GATE_REFERENCE_SCALE_M = 30

# ESA WorldCover v200 (2021) -- projenin Step4 export'uyla ayni kaynak/band.
WORLDCOVER_COLLECTION = "ESA/WorldCover/v200"
WORLDCOVER_BAND = "Map"
# WorldCover'in GERCEK/native cozunurlugu ~10 m'dir (yalnizca metadata /
# dokumantasyon icin kaydedilir).
WORLDCOVER_NATIVE_RESOLUTION_M = 10
# GATE ICIN kullanilan export cozunurlugu: 30 m (native 10 m DEGIL). Bu
# gate-only, diagnostic bir kategorik landcover katmanidir ve zaten referans
# gride (30 m) nearest-neighbor ile hizalanacaktir; 10 m'de export etmek
# genis AOI'lerde GEE'nin senkron boyut limitini asar (bkz. modul docstring).
GATE_LANDCOVER_EXPORT_SCALE_M = GATE_REFERENCE_SCALE_M

# Geriye donuk uyumluluk icin eski isim (bazi cagiran kodlar/testler
# WORLDCOVER_EXPORT_SCALE_M'i referans alabilir) -- ARTIK 30 m'e isaret eder.
WORLDCOVER_EXPORT_SCALE_M = GATE_LANDCOVER_EXPORT_SCALE_M


class Step6AError(SystemExit):
    """Fail-fast error for Step6A (diger step'lerle ayni konvansiyon)."""


def get_gate_inputs_dir(experiment_id: str) -> Path:
    return get_experiment_output_root(experiment_id) / "gate_inputs"


def get_gate_input_paths(experiment_id: str) -> dict:
    """Step6A ciktilarinin (planlanan) yollarini dondurur; hicbir sey yazmaz."""
    d = get_gate_inputs_dir(experiment_id)
    return {
        "gate_inputs_dir": d,
        "reference_path": d / "reference_30m.tif",
        "landcover_source_path": d / "landcover_esa_worldcover_v200.tif",
        "landcover_aligned_path": d / "landcover_esa_worldcover_v200_aligned_to_reference.tif",
        "metadata_path": d / "gate_inputs_metadata.json",
    }


def _init_gee_or_fail() -> None:
    try:
        from core.gee_utils import init_gee
    except Exception as exc:  # noqa: BLE001
        raise Step6AError(
            f"GEE importlari basarisiz (earthengine-api eksik olabilir): "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    try:
        init_gee(GEE_PROJECT)
    except Exception as exc:  # noqa: BLE001
        raise Step6AError(
            f"GEE init/auth basarisiz: {type(exc).__name__}: {exc}. "
            "'earthengine authenticate' calistirin ve GEE_PROJECT'i kontrol edin."
        ) from exc


def _export_image(image, out_path: Path, scale: int, region, crs: str) -> None:
    try:
        import geemap
    except Exception as exc:  # noqa: BLE001
        raise Step6AError(f"geemap import edilemedi: {type(exc).__name__}: {exc}") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Export basliyor -> %s (scale=%dm, crs=%s)", out_path, scale, crs)
    geemap.ee_export_image(
        image,
        filename=str(out_path),
        scale=scale,
        region=region,
        crs=crs,
        file_per_band=False,
    )
    if not out_path.exists():
        raise Step6AError(f"Export dosyasi olusmadi: {out_path}")
    log.info("Yazildi: %s", out_path)


def prepare_gate_inputs(experiment_id: str, force: bool = False) -> dict:
    """
    Step6B icin gerekli referans grid + hizalanmis landcover rasterlarini
    (namespaced) hazirlar ve metadata JSON yazar.

    Ciktilar zaten varsa ve force=False ise yeniden export YAPMAZ (var olan
    dosyalari dogrulayip yollarini dondurur) -- GEE kotasi bosa harcanmaz.
    """
    import ee  # noqa: F401  (init sonrasi kullanilacak)
    import rasterio

    exp = get_active_experiment(experiment_id)
    if exp["label_start_date"] is None or exp["label_end_date"] is None:
        raise Step6AError(
            f"'{experiment_id}' deneyinin label penceresi tanimsiz; gate "
            "girdisi hazirlanamaz (placeholder deney)."
        )

    paths = get_gate_input_paths(experiment_id)
    ref_path = paths["reference_path"]
    lc_source_path = paths["landcover_source_path"]
    lc_aligned_path = paths["landcover_aligned_path"]
    metadata_path = paths["metadata_path"]

    all_exist = ref_path.exists() and lc_aligned_path.exists()
    if all_exist and not force:
        log.info(
            "Gate girdileri zaten mevcut (%s); --force verilmedigi icin "
            "yeniden export YAPILMADI.", paths["gate_inputs_dir"],
        )
        return {**{k: str(v) for k, v in paths.items()}, "created": False}

    _init_gee_or_fail()
    import ee

    region = get_region_for_experiment(experiment_id)

    # --- 1) Sabit-degerli 30 m referans grid (TERMAL PREDICTOR DEGIL) ---
    log.info(
        "[%s] Gate referans gridi export ediliyor (sabit deger; yalnizca "
        "grid geometrisi icin, termal anlami YOK).", experiment_id,
    )
    reference_image = ee.Image.constant(1).rename("reference").toInt16().clip(region)
    _export_image(reference_image, ref_path, GATE_REFERENCE_SCALE_M, region, EXPORT_CRS)

    # --- 2) ESA WorldCover v200 kaynak export'u (GATE ICIN 30 m, native
    # 10 m DEGIL -- bkz. modul docstring). Boyut limiti asilirsa
    # run_predictors_only.py'nin direct-or-tiled fallback'i (2x2 -> 4x4 ->
    # 6x6 -> 8x8) REUSE edilir; yeni bir tiled export mantigi YAZILMAZ.
    log.info(
        "[%s] ESA WorldCover v200 export ediliyor (gate_export_resolution_m=%d, "
        "native=%dm).", experiment_id, GATE_LANDCOVER_EXPORT_SCALE_M,
        WORLDCOVER_NATIVE_RESOLUTION_M,
    )
    worldcover = ee.ImageCollection(WORLDCOVER_COLLECTION).first()
    if worldcover is None:
        raise Step6AError(f"{WORLDCOVER_COLLECTION} koleksiyonu bos dondu.")
    lc_image = worldcover.select(WORLDCOVER_BAND).clip(region)

    from scripts.run_predictors_only import export_image_direct_or_tiled

    lc_tiles_dir = paths["gate_inputs_dir"] / "_tiles" / "landcover_esa_worldcover_v200"
    lc_export_result = export_image_direct_or_tiled(
        lc_image, lc_source_path, region,
        scale=GATE_LANDCOVER_EXPORT_SCALE_M, crs=EXPORT_CRS,
        label="landcover_esa_worldcover_v200", force=force, tiles_dir=lc_tiles_dir,
    )
    log.info(
        "[%s] WorldCover export tamamlandi: transport=%s", experiment_id,
        lc_export_result["transport"],
    )

    # --- 3) Landcover'i referans gride hizala (nearest-neighbor, reuse) ---
    # Step8A'nin prepare_aligned_landcover'i reuse edilir: kategorik veri
    # icin nearest-neighbor; iki divergent hizalama implementasyonu OLMAZ.
    from src.step8a_prepare_500m_modeling_dataset import (
        Step8AError,
        prepare_aligned_landcover,
    )

    try:
        align_result = prepare_aligned_landcover(ref_path, lc_source_path, lc_aligned_path)
    except Step8AError as exc:
        raise Step6AError(str(exc)) from exc
    log.info("Landcover referans gride hizalandi: %s", align_result["path"])

    # --- 4) Metadata JSON ---
    with rasterio.open(ref_path) as src:
        ref_bounds = list(src.bounds)
        ref_crs = str(src.crs)
        ref_shape = [src.width, src.height]

    metadata = {
        "experiment_id": experiment_id,
        "region_key": exp["region_key"],
        "reference_path": str(ref_path),
        "landcover_path": str(lc_source_path),
        "aligned_landcover_path": str(lc_aligned_path),
        "crs": ref_crs,
        "scale_m": GATE_REFERENCE_SCALE_M,
        "reference_shape_wh": ref_shape,
        "aoi_bounds": ref_bounds,
        "label_window": [exp["label_start_date"], exp["label_end_date"]],
        "worldcover_collection": WORLDCOVER_COLLECTION,
        # BUG FIX: WorldCover artik native 10 m'de DEGIL, gate icin 30 m'de
        # export ediliyor (bkz. modul docstring). Her iki deger de acikca
        # kaydedilir, boylece "hangi cozunurluk kullanildi" hicbir zaman
        # belirsiz olmaz.
        "native_worldcover_resolution_m": WORLDCOVER_NATIVE_RESOLUTION_M,
        "gate_export_resolution_m": GATE_LANDCOVER_EXPORT_SCALE_M,
        "worldcover_export_scale_m": GATE_LANDCOVER_EXPORT_SCALE_M,
        "worldcover_export_transport": lc_export_result["transport"],
        "worldcover_export_tile_grid": (
            list(lc_export_result["tile_grid"]) if lc_export_result.get("tile_grid") else None
        ),
        "landcover_alignment_method": "nearest_neighbor_to_reference_grid",
        "alignment_resampling": "nearest",
        "gate_only_landcover": True,
        "not_a_thermal_predictor": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Gate-only reference grid: constant-valued raster used ONLY for "
            "grid geometry (width/height/CRS/transform) by Step6B. It is NOT "
            "a thermal predictor and carries no scientific/LST meaning. "
            "Landcover is exported at gate_export_resolution_m (30 m) rather "
            "than WorldCover's native_worldcover_resolution_m (10 m) because "
            "it is a diagnostic gate input, already re-aligned to the "
            "reference grid with nearest-neighbor resampling; exporting at "
            "native 10 m resolution can exceed GEE's synchronous download "
            "size limit for large AOIs."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Metadata yazildi: %s", metadata_path)

    return {**{k: str(v) for k, v in paths.items()}, "created": True}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step6A: Step6B gate icin minimum girdileri hazirlar -- "
        "sabit-degerli 30 m referans grid (termal predictor DEGIL) ve "
        "referans gride nearest-neighbor hizalanmis ESA WorldCover landcover. "
        "Tum ciktilar outputs/experiments/<experiment_id>/gate_inputs/ altina "
        "yazilir; legacy Kozan yollarina DOKUNULMAZ."
    )
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    prepare_gate_inputs(experiment_id=args.experiment, force=args.force)