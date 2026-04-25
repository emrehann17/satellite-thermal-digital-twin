"""
step2_modis_5year_mean.py

Yapılanlar:
    - GEE bağlantısını başlatmak
    - Çalışma bölgelerini almak
    - Doğu Akdeniz için 2019-2023 yaz ayları MODIS LST verisini sorgulamak
    - Zaman ortalaması almak
    - DN -> Celsius dönüşümü yapmak
    - İşlenmiş görüntü ve metadata üretmek
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
OUTPUTS_DIR = BASE_DIR / "outputs" / "step2"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

log, log_file = setup_logger("step2")


# =============================================================================
# 1. 5 YILLIK YAZ LST GÖRÜNTÜSÜNÜ HAZIRLAMA
# =============================================================================
def process_summer_mean(
        region: ee.Geometry,
        region_name: str,
        start: str = START_DATE,
        end: str = END_DATE
    ) -> tuple[ee.Image, dict]:
    """
    2019-2023 arası yaz aylarının (Haziran-Eylül) MODIS LST ortalamasını hesaplar.
    Export yapmaz. Sadece işlenmiş ee.Image üretir.
    """
    log.info(f"Processing 5-year summer mean for region: {region_name}")

    collection = (
        ee.ImageCollection(MODIS_COLLECTION)
        .filterBounds(region)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.calendarRange(6, 9, "month"))
        .select("LST_Day_1km")
    )

    count = collection.size().getInfo()
    log.info(f"Found {count} MODIS images for region '{region_name}'")

    if count == 0:
        raise ValueError(
            f"No MODIS images found for region '{region_name}' in the specified date range."
        )

    first_date = (
        ee.Date(collection.first().get("system:time_start"))
        .format("YYYY-MM-dd")
        .getInfo()
    )

    summer_mean = (
        collection
        .mean()
        .multiply(0.02)
        .subtract(273.15)
        .rename("LST_Celsius_SummerMean")
        .clip(region)
    )

    metadata = {
        "region_name": region_name,
        "collection": MODIS_COLLECTION,
        "band": "LST_Day_1km",
        "unit": "Celsius",
        "date_start": START_DATE,
        "date_end": END_DATE,
        "months": "6-9",
        "image_count": count,
        "first_image_date": first_date,
        "created_at": datetime.now().isoformat(),
        "status": "processed"
    }

    return summer_mean, metadata

# ══════════════════════════════════════════════════════════════════════════════
# 2. METADATA KAYDETME
# ══════════════════════════════════════════════════════════════════════════════
def save_metadata(metadata: dict, filename: str = "step2_metadata.json") -> Path:
    """Step 2 metadata bilgisini JSON olarak kaydeder."""
    output_path = OUTPUTS_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.info(f"Metadata saved to: {output_path}")
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# ANA AKIŞ
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("STEP 2 INITIALIZING...")
    log.info("=" * 60)

    init_gee()
    regions = build_regions()

    mean_image, metadata = process_summer_mean(
        region=regions["dogu_akdeniz"],
        region_name="dogu_akdeniz",
        start=START_DATE,
        end=END_DATE
    )

    metadata_path = save_metadata(metadata)

    log.info("=" * 60)
    log.info("STEP 2 COMPLETED SUCCESSFULLY")
    log.info(f"Metadata file: {metadata_path}")
    log.info("Output: processed ee.Image object (not exported yet)")
    log.info("Next step: anomaly calculation or GeoTIFF export")
    log.info("=" * 60)

    _ = mean_image  # Şimdilik değişkeni korumak için


if __name__ == "__main__":
    main()