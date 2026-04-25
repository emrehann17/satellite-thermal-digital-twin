"""
step1_gee_setup_and_fetch.py

Yapılanlar:
    - GEE'ye bağlanmak ve bağlantıyı doğrulamak
    - Çalışma bölgelerini tanımlamak
    - MODIS MOD11A1 koleksiyonunu sorgulamak
    - LST_Day_1km bandını Celsius'a çevirmek
    - Sonraki adımlar için temel metadata üretmek
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
OUTPUTS_DIR = BASE_DIR / "outputs" / "step1"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

log, log_file = setup_logger("step1")


# =============================================================================
# 1. MODIS LST KOLEKSİYONU SORGULAMA
# =============================================================================
def fetch_modis_lst(region: ee.Geometry, region_name: str, start: str = START_DATE, end: str = END_DATE) -> tuple[ee.Image, dict]:
    """
    MODIS MOD11A1 koleksiyonunu verilen bölge ve tarih aralığına göre filtreler.
    LST_Day_1km bandını seçer, zaman ortalaması alır ve Celsius'a çevirir.
    """
    log.info(f"MODIS sorgusu başlatıldı. Tarih aralığı: {start} -> {end}")

    collection = (
        ee.ImageCollection(MODIS_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
        .select("LST_Day_1km")
    )

    image_count = collection.size().getInfo()
    log.info(f"Bulunan görüntü sayısı: {image_count}")

    if image_count == 0:
        raise ValueError(f"{start} - {end} tarih aralığında görüntü bulunamadı.")

    first_image = collection.first()

    first_image_date = (
        ee.Date(first_image.get("system:time_start"))
        .format("YYYY-MM-dd")
        .getInfo()
    )

    lst_celsius = (
        collection
        .mean()
        .multiply(0.02)
        .subtract(273.15)
        .rename("LST_Celsius")
        .clip(region)
    )

    metadata = {
        "gee_project": GEE_PROJECT,
        "collection": MODIS_COLLECTION,
        "band": "LST_Day_1km",
        "unit": "Celsius",
        "date_start": start,
        "date_end": end,
        "image_count": image_count,
        "first_image_date": first_image_date,
        "created_at": datetime.now().isoformat()
    }

    log.info("MODIS sorgulama ve dönüşüm tamamlandı.")
    return lst_celsius, metadata


# =============================================================================
# 2. METADATA KAYDETME
# =============================================================================
def save_metadata(metadata: dict, filename: str = "step1_metadata.json") -> Path:
    """
    Metadata bilgisini outputs klasörüne JSON olarak kaydeder.
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
    log.info("STEP 1 BAŞLIYOR")
    log.info("=" * 60)

    # 1) GEE bağlantısı
    init_gee()

    # 2) Bölgeleri oluştur
    regions = build_regions()

    # 3) Doğu Akdeniz için LST görüntüsünü hazırla
    lst_image, metadata = fetch_modis_lst(
        region=regions["dogu_akdeniz"],
        region_name="dogu_akdeniz",
        start=START_DATE,
        end=END_DATE
    )

    # 4) Metadata kaydet
    metadata_path = save_metadata(metadata)

    log.info("=" * 60)
    log.info("STEP 1 TAMAMLANDI")
    log.info(f"Metadata dosyası: {metadata_path}")
    log.info("Hazırlanan çıktı: ee.Image tipinde LST_Celsius görüntüsü")
    log.info("Sonraki adım: export / GeoTIFF üretimi")
    log.info("=" * 60)

    # Şimdilik sadece test amaçlı
    # Sonraki step'te bu image export edilecek
    _ = lst_image


if __name__ == "__main__":
    main()