"""
step3_landsat_lst.py

Yapılanlar:
    - GEE bağlantısını başlatmak
    - Çalışma bölgelerini almak
    - Doğu Akdeniz için Landsat 8 Collection 2 Level 2 verisini sorgulamak
    - ST_B10 bandını kullanarak yüzey sıcaklığı üretmek
    - Kelvin -> Celsius dönüşümü yapmak
    - İşlenmiş yüksek çözünürlüklü LST görüntüsü ve metadata üretmek

NOT:
    Bu adım export yapmaz.
    GeoTIFF export işlemi sonraki step'te yapılacaktır.
"""

import json
from datetime import datetime
from pathlib import Path

import ee

from core.config import *
from core.gee_utils import init_gee
from core.regions import build_regions
from core.io_utils import setup_logger


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs" / "step3"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

log, log_file = setup_logger("step3")

# =============================================================================
# 1. LANDSAT LST İŞLEME
# =============================================================================
def process_landsat_lst(
    region: ee.Geometry,
    region_name: str,
    start: str = START_DATE,
    end: str = END_DATE
) -> tuple[ee.Image, dict]:
    """
    Landsat 8 Collection 2 Level 2 veri setini verilen bölge ve tarih aralığına göre filtreler,
    ST_B10 bandını seçer, sıcaklık dönüşümünü uygular ve ortalama LST görüntüsü üretir.

    Dönüş:
        (landsat_lst_image, metadata_dict)
    """
    log.info(f"Landsat LST işleme başlatıldı. Bölge: {region_name}")
    log.info(f"Tarih aralığı: {start} -> {end}")

    base_collection = (
        ee.ImageCollection(LANDSAT_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
    )

    base_count = base_collection.size().getInfo()
    log.info(f"{start} - {end} aralığındaki Landsat görüntü sayısı: {base_count}")
    
    filtered_collection = (
        base_collection
        .filter(ee.Filter.calendarRange(6,9, "month"))
        .select("ST_B10")
    )

    image_count = filtered_collection.size().getInfo()
    log.info(f"{start} - {end} aralığındaki Yaz aylarındaki Landsat görüntü sayısı: {image_count}")

    if image_count == 0:
        raise ValueError(
            f"{region_name} bölgesi için {start} - {end} aralığında yaz aylarında Landsat görüntüsü bulunamadı."
        )

    first_image = filtered_collection.first()
    first_image_date = (
        ee.Date(first_image.get("system:time_start"))
        .format("YYYY-MM-dd")
        .getInfo()
    )

    # Landsat Collection 2 Level 2 Surface Temperature dönüşümü
    # Kelvin = DN * 0.00341802 + 149.0
    # Celsius = Kelvin - 273.15
    landsat_lst = (
        filtered_collection
        .mean()
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)
        .rename("Landsat_LST_Celsius")
        .clip(region)
    )

    metadata = { #buraya filtre atılmaz. burası veri değil veri hakkında bilgi içerir
        "gee_project": GEE_PROJECT,
        "region_name": region_name,
        "collection": LANDSAT_COLLECTION,
        "band": "ST_B10",
        "unit": "Celsius",
        "date_start": start,
        "date_end": end,
        "all_image_count": base_count,
        "filtered_image_count": image_count,
        "first_image_date": first_image_date,
        "resolution": "30m",
        "created_at": datetime.now().isoformat(),
        "status": "processed"
    }

    log.info("Landsat LST işleme başarıyla tamamlandı.")
    return landsat_lst, metadata

# =============================================================================
# 2. LANDSAT TIMESERIES COLLECTION ÜRETME
# =============================================================================
def get_landsat_timeseries_collection(
    region: ee.Geometry,
    region_name: str,
    start: str = START_DATE,
    end: str = END_DATE
) -> tuple[ee.ImageCollection, dict]:
    """
    Step4 tarafından tarih tarih export edilecek Landsat zaman serisi collection'ını hazırlar.

    NOT:
        Bu fonksiyon export yapmaz.
        ST_B10 ve QA_PIXEL bandlarını birlikte döndürür.
    """
    log.info(f"Landsat zaman serisi collection hazırlanıyor. Bölge: {region_name}")
    log.info(f"Tarih aralığı: {start} -> {end}")

    base_collection = (
        ee.ImageCollection(LANDSAT_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
    )

    base_count = base_collection.size().getInfo()

    filtered_collection = (
        base_collection
        .filter(ee.Filter.calendarRange(6, 9, "month"))
        .select(["ST_B10", "QA_PIXEL"])
        .map(lambda image: image.clip(region))
    )

    filtered_count = filtered_collection.size().getInfo()

    if filtered_count == 0:
        raise ValueError(
            f"{region_name} bölgesi için {start} - {end} aralığında yaz aylarına ait Landsat görüntüsü bulunamadı."
        )

    metadata = {
        "gee_project": GEE_PROJECT,
        "region_name": region_name,
        "collection": LANDSAT_COLLECTION,
        "bands": ["ST_B10", "QA_PIXEL"],
        "date_start": start,
        "date_end": end,
        "months_filter": "6-9",
        "all_image_count": base_count,
        "filtered_image_count": filtered_count,
        "status": "timeseries_collection_prepared"
    }

    log.info(f"Tüm Landsat görüntü sayısı: {base_count}")
    log.info(f"Yaz ayları filtreli Landsat görüntü sayısı: {filtered_count}")

    return filtered_collection, metadata


# =============================================================================
# 3. METADATA KAYDETME
# =============================================================================
def save_metadata(metadata: dict, filename: str = "step3_metadata.json") -> Path:
    """
    Step3 metadata bilgisini JSON olarak kaydeder.
    """
    output_path = OUTPUTS_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.info(f"Metadata kaydedildi: {output_path}")
    return output_path


# =============================================================================
# ANA AKIŞ
# =============================================================================
def main() -> None:
    log.info("=" * 60)
    log.info("STEP 3 BAŞLIYOR")
    log.info("=" * 60)

    init_gee()
    regions = build_regions()

    landsat_timeseries, metadata = get_landsat_timeseries_collection(
        region=regions["dogu_akdeniz"],
        region_name="dogu_akdeniz",
        start=START_DATE,
        end=END_DATE
    )

    metadata_path = save_metadata(metadata)

    log.info("=" * 60)
    log.info("STEP 3 TAMAMLANDI")
    log.info(f"Metadata dosyası: {metadata_path}")
    log.info("Hazırlanan çıktı: ee.ImageCollection tipinde yüksek çözünürlüklü Landsat zaman serisi")
    log.info("Sonraki adım: step4")
    log.info("=" * 60)

    _ = landsat_timeseries # Step4'te kullanılmak üzere döndürülen collection burada tutulur


if __name__ == "__main__":
    main()