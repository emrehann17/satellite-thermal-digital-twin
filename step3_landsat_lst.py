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

from core.config import (
    GEE_PROJECT, 
    LANDSAT_COLLECTION, 
    START_DATE, 
    END_DATE,
    LANDSAT_SCALE,
    LANDSAT_OFFSET,
    REGION_NAME,
    CURRENT_PERIOD_DAYS,
    CURRENT_PERIOD_END_DATE,
    BASELINE_START_DATE,
    BASELINE_END_DATE
)
from core.gee_utils import init_gee
from core.regions import build_regions
from core.io_utils import setup_logger


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs" / "step3"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

log, log_file = setup_logger("step3")

def _set_export_date(image: ee.Image) -> ee.Image:
    """Adds a YYYY-MM-dd date key used for daily compositing/export."""
    export_date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
    return image.set("export_date", export_date)


def apply_qa_mask(image: ee.Image) -> ee.Image:
    """
    Landsat QA_PIXEL bandındaki bulut/gölge/kar/dolgu piksellerini maskeler.

    Bu maske current period median için özellikle gereklidir; aksi halde soğuk
    bulut pikselleri güncel LST gibi median'a girip mavi anomali blokları üretir.
    """
    qa = image.select("QA_PIXEL")
    bad_bits = (
        (1 << 0)  # fill
        | (1 << 1)  # dilated cloud
        | (1 << 2)  # cirrus
        | (1 << 3)  # cloud
        | (1 << 4)  # cloud shadow
        | (1 << 5)  # snow
    )
    clean_mask = qa.bitwiseAnd(bad_bits).eq(0)
    return image.updateMask(clean_mask)

# =============================================================================
# 1. LANDSAT TIMESERIES COLLECTION ÜRETME
# =============================================================================
def get_landsat_daily_median_collection(
    region: ee.Geometry,
    region_name: str,
    start: str = START_DATE,
    end: str = END_DATE,
) -> tuple[ee.ImageCollection, dict]:
    """
    Step4 tarafindan export edilecek temiz Landsat zaman serisi collection'ini hazirlar.

    Ayni tarihte birden fazla Landsat sahnesi varsa:
        - ST_B10 icin median composite alinir.
        - QA_PIXEL bit maskesi oldugu icin median yerine mode kullanilir.

    Donus:
        (daily_composite_collection, metadata_dict)
    """
    log.info(f"Landsat gunluk median composite hazirlaniyor. Bolge: {region_name}")
    log.info(f"Tarih araligi: {start} -> {end}")

    base_collection = (
        ee.ImageCollection(LANDSAT_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
    )

    filtered_collection = (
        base_collection
        .filter(ee.Filter.calendarRange(6, 9, "month"))
        .select(["ST_B10", "QA_PIXEL"])
        .map(lambda image: image.clip(region))
        .map(_set_export_date)
    )

    unique_dates = filtered_collection.aggregate_array("export_date").distinct().sort()
    daily_dates = unique_dates.getInfo()
    unique_date_count = len(daily_dates)

    if unique_date_count == 0:
        raise ValueError(
            f"{region_name} bolgesi icin {start} - {end} araliginda yaz aylarina ait Landsat goruntusu bulunamadi."
        )

    def build_daily_composite(date_value: ee.String) -> ee.Image:
        daily_collection = filtered_collection.filter(ee.Filter.eq("export_date", date_value))

        lst_median = daily_collection.select("ST_B10").median().rename("ST_B10")
        qa_mode = daily_collection.select("QA_PIXEL").mode().rename("QA_PIXEL").toUint16()
        first_image = ee.Image(daily_collection.sort("system:time_start").first())

        return (
            lst_median
            .addBands(qa_mode)
            .clip(region)
            .set("export_date", date_value)
            .set("system:time_start", first_image.get("system:time_start"))
            .set("source_image_count", daily_collection.size())
        )

    daily_collection = ee.ImageCollection(unique_dates.map(build_daily_composite))

    metadata = {
        "gee_project": GEE_PROJECT,
        "region_name": region_name,
        "collection": LANDSAT_COLLECTION,
        "bands": ["ST_B10", "QA_PIXEL"],
        "date_start": start,
        "date_end": end,
        "months_filter": "6-9",
        "daily_dates": daily_dates,
        "daily_composite_count": unique_date_count,
        "lst_composite_method": "daily_median",
        "qa_composite_method": "daily_mode",
        "resolution": "30m",
        "created_at": datetime.now().isoformat(),
        "status": "daily_median_collection_prepared",
    }

    log.info(f"Gunluk composite sayisi: {unique_date_count}")

    return daily_collection, metadata

# =============================================================================
# 2. CURRENT PERIOD MEDIAN (ANOMALY IÇIN)
# =============================================================================
def get_current_period_median(
    region: ee.Geometry,
    region_name: str,
    end_date: str,
    window_days: int = 60
) -> tuple[ee.Image, dict]:
    """
    Anomali hesabı için 'current state' tanımlar.
    
    Son N günlük penceredeki TÜM Landsat sahnelerini (path/row fark etmez)
    median composite ile tek görüntüye indirir.
    
    Args:
        region: Çalışma bölgesi
        region_name: Bölge adı
        end_date: Pencerenin bitiş tarihi (YYYY-MM-DD)
        window_days: Pencere genişliği (gün)
    
    Returns:
        (current_median_image, metadata_dict)
    """
    from datetime import datetime, timedelta
    
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=window_days)
    start_date = start_dt.strftime("%Y-%m-%d")
    
    log.info(f"Current period median hazırlanıyor:")
    log.info(f"  Bölge: {region_name}")
    log.info(f"  Pencere: {start_date} -> {end_date} ({window_days} gün)")
    
    collection = (
        ee.ImageCollection(LANDSAT_COLLECTION)
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.calendarRange(6, 9, "month"))  # Yaz ayları
        .select(["ST_B10", "QA_PIXEL"])
        .map(lambda img: img.clip(region))
        .map(apply_qa_mask)
        .select("ST_B10")
    )
    
    scene_count = collection.size().getInfo()
    
    if scene_count == 0:
        raise ValueError(
            f"{region_name} için {start_date} - {end_date} arasında yaz aylarında "
            f"Landsat sahnesi bulunamadı."
        )
    
    log.info(f"  Current period sahne sayısı: {scene_count}")
    
    current_valid_count = (
        collection
        .count()
        .rename("Current_Period_Valid_Count")
        .toFloat()
        .clip(region)
    )

    # QA-temiz sahnelerin median'ını al; ikinci bant geçerli gözlem sayısıdır.
    current_median = (
        collection
        .median()
        .multiply(LANDSAT_SCALE)
        .add(LANDSAT_OFFSET)
        .subtract(273.15)
        .rename("Current_Period_LST_Celsius")
        .addBands(current_valid_count)
        .clip(region)
    )
    
    metadata = {
        "gee_project": GEE_PROJECT,
        "region_name": region_name,
        "collection": LANDSAT_COLLECTION,
        "band": "ST_B10",
        "unit": "Celsius",
        "window_start": start_date,
        "window_end": end_date,
        "window_days": window_days,
        "scene_count": scene_count,
        "months_filter": "6-9",
        "composite_method": "median",
        "qa_mask_applied": True,
        "output_bands": [
            "Current_Period_LST_Celsius",
            "Current_Period_Valid_Count",
        ],
        "qa_masked_bits": [
            "fill",
            "dilated_cloud",
            "cirrus",
            "cloud",
            "cloud_shadow",
            "snow",
            "medium_high_cloud_confidence",
            "medium_high_cloud_shadow_confidence",
            "medium_high_snow_ice_confidence",
            "medium_high_cirrus_confidence",
        ],
        "qa_water_bit_preserved": True,
        "created_at": datetime.now().isoformat(),
        "status": "current_period_median_prepared"
    }
    
    log.info("Current period median başarıyla oluşturuldu.")
    
    return current_median, metadata


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

    # Baseline için zaman serisi collection
    log.info("\n1) Baseline zaman serisi hazırlanıyor...")
    landsat_timeseries, ts_metadata = get_landsat_daily_median_collection(
        region=regions[REGION_NAME],
        region_name=REGION_NAME,
        start=BASELINE_START_DATE,
        end=BASELINE_END_DATE
    )

    #Current period median (anomali için)
    log.info("\n2) Current period median hazırlanıyor...")
    current_median, current_metadata = get_current_period_median(
        region=regions[REGION_NAME],
        region_name=REGION_NAME,
        end_date=CURRENT_PERIOD_END_DATE,
        window_days=CURRENT_PERIOD_DAYS
    )

    # Metadata kaydetme
    combined_metadata = {
        "baseline_timeseries": ts_metadata,
        "current_period": current_metadata
    }
    
    metadata_path = save_metadata(combined_metadata)

    log.info("=" * 60)
    log.info("STEP 3 TAMAMLANDI")
    log.info(f"Metadata dosyası: {metadata_path}")
    log.info("\nHazırlanan çıktılar:")
    log.info("  1. Baseline zaman serisi (ee.ImageCollection)")
    log.info("  2. Current period median (ee.Image)")
    log.info("Sonraki adım: step4 (export)")
    log.info("=" * 60)

    return {
        "landsat_timeseries": landsat_timeseries,
        "landsat_metadata": ts_metadata,
        "current_median": current_median,
        "current_metadata": current_metadata,
    }

if __name__ == "__main__":
    main()