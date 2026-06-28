"""
step2b_dem.py

Yapılanlar:
    - GEE bağlantısını başlatmak
    - Çalışma bölgelerini almak
    - DEM kaynağını seçmek (tercih: Copernicus DEM GLO-30, fallback: USGS SRTMGL1 003)
    - elevation ve slope (ee.Terrain.slope) ürünlerini hazırlamak
    - Export yapılandırmasını (sadece) hazırlamak
    - İşlenmiş DEM görüntüsü ve metadata üretmek

NOT:
    Bu adım export yapmaz, indirme yapmaz ve geemap.ee_export_image() çağırmaz.
    GeoTIFF export işlemi Step4'te, indirme Step4b'de yapılır.
    Step2 (MODIS) ile aynı prepare-data + write-metadata lifecycle'ını izler.
"""

import json
from datetime import datetime
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import ee

from core.config import (
    REGION_NAME,
    EXPORT_CRS,
    EXPORT_FOLDER,
    DEM_COLLECTION,
    DEM_COLLECTION_BAND,
    DEM_FALLBACK_DATASET,
    DEM_FALLBACK_BAND,
    DEM_EXPORT,
)
from core.gee_utils import init_gee
from core.regions import build_regions
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT


BASE_DIR = PROJECT_ROOT
OUTPUTS_DIR = BASE_DIR / "outputs" / "step2b"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

log, log_file = setup_logger("step2b")


# =============================================================================
# 1. DEM KAYNAĞINI SEÇME
# =============================================================================
def select_dem_elevation(region: ee.Geometry) -> tuple[ee.Image, ee.Projection, dict]:
    """
    DEM yükseklik (elevation) görüntüsünü ve native projeksiyonunu seçer.

    Önce tercih edilen Copernicus DEM GLO-30 ImageCollection denenir; bant adı
    elevation'a yeniden adlandırılır. Erişilemezse USGS SRTMGL1 003 fallback'ine
    düşülür. Seçilen kaynak metadata'da raporlanır. Export yapılmaz.

    Mosaic'in sabit projeksiyonu olmadığı için slope (komşuluk işlemi) hesabında
    kullanılmak üzere DEM'in native projeksiyonu da döndürülür.
    """
    preferred_error = None

    try:
        log.info("Tercih edilen DEM kaynağı deneniyor: %s", DEM_COLLECTION)
        dem_collection = (
            ee.ImageCollection(DEM_COLLECTION)
            .filterBounds(region)
            .select(DEM_COLLECTION_BAND)
        )
        # Mosaic'in sabit projeksiyonu yoktur; slope gibi komşuluk (neighborhood)
        # işlemleri için DEM'in NATIVE projeksiyonunu (ilk karo) referans alırız.
        native_projection = dem_collection.first().projection()
        log.info(
            "Native projection: %s",
            native_projection.getInfo()
        )
        elevation = (
            dem_collection
            .mosaic()
            .rename("elevation")
        )
        # filterBounds boş dönerse mosaic boş image üretebilir; bant varlığını doğrula.
        band_names = elevation.bandNames().getInfo()
        if "elevation" not in band_names:
            raise ValueError(
                f"{DEM_COLLECTION} mozaiği elevation bandı üretmedi (bands={band_names})."
            )

        log.info("DEM kaynağı seçildi: %s (elevation)", DEM_COLLECTION)
        source_meta = {
            "dataset": DEM_COLLECTION,
            "dataset_type": "ImageCollection",
            "input_band": DEM_COLLECTION_BAND,
            "used_fallback": False,
        }
        return elevation, native_projection, source_meta

    except Exception as exc:  # noqa: BLE001
        preferred_error = str(exc)
        log.warning(
            "Tercih edilen DEM kaynağı kullanılamadı (%s): %s. Fallback'e düşülüyor: %s",
            DEM_COLLECTION,
            preferred_error,
            DEM_FALLBACK_DATASET,
        )

    log.info("Fallback DEM kaynağı kullanılıyor: %s", DEM_FALLBACK_DATASET)
    fallback_image = ee.Image(DEM_FALLBACK_DATASET).select(DEM_FALLBACK_BAND)
    # Tek Image fallback'inin native projeksiyonu zaten tanımlıdır (~30 m).
    native_projection = fallback_image.projection()
    elevation = fallback_image.rename("elevation")

    source_meta = {
        "dataset": DEM_FALLBACK_DATASET,
        "dataset_type": "Image",
        "input_band": DEM_FALLBACK_BAND,
        "used_fallback": True,
        "preferred_error": preferred_error,
    }
    return elevation, native_projection, source_meta


# =============================================================================
# 2. ELEVATION + SLOPE HAZIRLAMA
# =============================================================================
def prepare_dem_products(
    region: ee.Geometry,
    region_name: str,
) -> tuple[ee.Image, dict]:
    """
    DEM elevation ve slope (ee.Terrain.slope) ürünlerini hazırlar.

    Export yapmaz; sadece işlenmiş ee.Image (elevation + slope bantları) ve
    export yapılandırmasını içeren metadata üretir. AOI yapılandırılmış bölgedir.
    """
    log.info(
        "DEM ürünleri hazırlanıyor: region=%s (elevation, slope)",
        region_name,
    )

    elevation, native_projection, source_meta = select_dem_elevation(region)

    # --- SLOPE FIX ---
    # ee.Terrain.slope bir komşuluk (neighborhood) işlemidir ve piksel aralığını
    # bilmek için girişin SABİT bir projeksiyonu olmasını gerektirir. Copernicus
    # DEM .mosaic() sonucu sabit projeksiyon taşımaz (varsayılan WGS84, ~1 derece
    # ölçek). Bu durumda slope, export sırasındaki 30 m reprojection ile birlikte
    # tümüyle maskeli (her piksel NaN) bir bant üretir.
    #
    # Çözüm: slope'u, DEM'in NATIVE projeksiyonuna sabitlenmiş elevation üzerinde
    # hesapla. Böylece gradyan gerçek ~30 m piksel aralığında hesaplanır; reprojeksiyon
    # slope HESAPLANDIKTAN sonra (export aşamasında) uygulanır.
    elevation_for_slope = elevation.setDefaultProjection(native_projection)
    slope = ee.Terrain.slope(elevation_for_slope).rename("slope")

    log.info("Slope istatistikleri kontrol ediliyor...")

    stats = slope.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=region,
        scale=30,
        maxPixels=1e12,
    ).getInfo()

    count = slope.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=region,
        scale=30,
        maxPixels=1e12,
    ).getInfo()

    log.info("Slope stats: %s", stats)
    log.info("Slope count: %s", count)    

    # elevation (değer geçişli, pointwise) DEĞİŞTİRİLMEDİ.
    elevation = elevation.clip(region)
    slope = slope.clip(region)

    dem_image = elevation.addBands(slope)

    elevation_export = DEM_EXPORT["elevation"]
    slope_export = DEM_EXPORT["slope"]
    scale = DEM_EXPORT["scale"]

    metadata = {
        "step": "step2b_dem",
        "region_name": region_name,
        "dataset": source_meta["dataset"],
        "dataset_type": source_meta["dataset_type"],
        "fallback_dataset": DEM_FALLBACK_DATASET,
        "used_fallback": source_meta["used_fallback"],
        "preferred_error": source_meta.get("preferred_error"),
        "input_band": source_meta["input_band"],
        "aoi": region.getInfo(),
        "crs": EXPORT_CRS,
        "export_resolution_m": scale,
        "export_folder": EXPORT_FOLDER,
        "products": {
            "elevation": {
                "band": "elevation",
                "description": elevation_export["description"],
                "file_name_prefix": elevation_export["file_name_prefix"],
                "source": "dem_dataset",
                "unit": "meters",
            },
            "slope": {
                "band": "slope",
                "description": slope_export["description"],
                "file_name_prefix": slope_export["file_name_prefix"],
                "source": "ee.Terrain.slope",
                "unit": "degrees",
            },
        },
        "export_filenames": [
            elevation_export["file_name_prefix"],
            slope_export["file_name_prefix"],
        ],
        "output_bands": ["elevation", "slope"],
        "usage_note": (
            "DEM statik yardımcı predictor olarak hazırlanır (elevation, slope). "
            "Step5+ içinde HENÜZ kullanılmaz; ileride RF/XGBoost MODIS downscaling "
            "için planlanmıştır."
        ),
        "created_at": datetime.now().isoformat(),
        "status": "prepared",
    }

    log.info(
        "DEM görüntüsü hazırlandı (dataset=%s, fallback=%s).",
        source_meta["dataset"],
        source_meta["used_fallback"],
    )
    return dem_image, metadata


# =============================================================================
# 3. METADATA KAYDETME
# =============================================================================
def save_metadata(metadata: dict, filename: str = "step2b_dem_metadata.json") -> Path:
    """Step2B DEM metadata bilgisini JSON olarak kaydeder."""
    output_path = OUTPUTS_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.info("Metadata kaydedildi: %s", output_path)
    return output_path


# =============================================================================
# ANA AKIŞ
# =============================================================================
def main() -> dict:
    log.info("=" * 60)
    log.info("STEP 2B BAŞLIYOR (DEM prepare + metadata)")
    log.info("=" * 60)

    init_gee()
    regions = build_regions()

    dem_image, metadata = prepare_dem_products(
        region=regions[REGION_NAME],
        region_name=REGION_NAME,
    )

    metadata_path = save_metadata(metadata)

    log.info("=" * 60)
    log.info("STEP 2B TAMAMLANDI")
    log.info("Metadata dosyası: %s", metadata_path)
    log.info("Çıktı: elevation + slope bantlı ee.Image (henüz export edilmedi)")
    log.info("Sonraki adım: Step4 DEM export task'larını oluşturur.")
    log.info("=" * 60)

    _ = dem_image  # Şimdilik değişkeni korumak için
    return metadata


if __name__ == "__main__":
    main()