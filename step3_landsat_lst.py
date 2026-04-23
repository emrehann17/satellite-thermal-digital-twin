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

from core.config import GEE_PROJECT, LANDSAT_COLLECTION, START_DATE, END_DATE
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

    collection = (
        ee.ImageCollection(LANDSAT_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
        .select("ST_B10")
    )

    image_count = collection.size().getInfo()
    log.info(f"Filtre sonrası Landsat görüntü sayısı: {image_count}")

    if image_count == 0:
        raise ValueError(
            f"{region_name} bölgesi için {start} - {end} aralığında Landsat görüntüsü bulunamadı."
        )

    first_image = collection.first()
    first_image_date = (
        ee.Date(first_image.get("system:time_start"))
        .format("YYYY-MM-dd")
        .getInfo()
    )

    # Landsat Collection 2 Level 2 Surface Temperature dönüşümü
    # Kelvin = DN * 0.00341802 + 149.0
    # Celsius = Kelvin - 273.15
    landsat_lst = (
        collection
        .mean()
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)
        .rename("Landsat_LST_Celsius")
        .clip(region)
    )

    metadata = {
        "gee_project": GEE_PROJECT,
        "region_name": region_name,
        "collection": LANDSAT_COLLECTION,
        "band": "ST_B10",
        "unit": "Celsius",
        "date_start": start,
        "date_end": end,
        "months": "6-9",
        "image_count": image_count,
        "first_image_date": first_image_date,
        "resolution": "30m",
        "created_at": datetime.now().isoformat(),
        "status": "processed"
    }

    log.info("Landsat LST işleme başarıyla tamamlandı.")
    return landsat_lst, metadata


# =============================================================================
# 2. METADATA KAYDETME
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

    landsat_lst_image, metadata = process_landsat_lst(
        region=regions["dogu_akdeniz"],
        region_name="dogu_akdeniz",
        start=START_DATE,
        end=END_DATE
    )

    metadata_path = save_metadata(metadata)

    log.info("=" * 60)
    log.info("STEP 3 TAMAMLANDI")
    log.info(f"Metadata dosyası: {metadata_path}")
    log.info("Hazırlanan çıktı: ee.Image tipinde yüksek çözünürlüklü Landsat LST görüntüsü")
    log.info("Sonraki adım: step4")
    log.info("=" * 60)

    _ = landsat_lst_image


if __name__ == "__main__":
    main()