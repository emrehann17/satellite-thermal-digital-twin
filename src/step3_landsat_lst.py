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
from datetime import datetime, timedelta
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import ee

from core.config import (
    GEE_PROJECT, 
    LANDSAT_COLLECTION, 
    START_DATE, 
    END_DATE,
    LANDSAT_SCALE,
    LANDSAT_OFFSET,
    LANDSAT_SR_SCALE,
    LANDSAT_SR_OFFSET,
    LANDSAT_RED_BAND,
    LANDSAT_NIR_BAND,
    REGION_NAME,
    CURRENT_PERIOD_DAYS,
    CURRENT_PERIOD_END_DATE,
    BASELINE_START_DATE,
    BASELINE_END_DATE,
    ENABLE_NDVI_EXPORT,
)
from core.gee_utils import init_gee
from core.regions import build_regions
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT


BASE_DIR = PROJECT_ROOT
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
    QA_PIXEL water biti özellikle maskelenmez; su kütleleri LST ürününde korunur.
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

    cloud_confidence = qa.rightShift(8).bitwiseAnd(3)
    cloud_shadow_confidence = qa.rightShift(10).bitwiseAnd(3)
    snow_ice_confidence = qa.rightShift(12).bitwiseAnd(3)
    cirrus_confidence = qa.rightShift(14).bitwiseAnd(3)

    clean_mask = (
        qa.bitwiseAnd(bad_bits)
        .eq(0)
        .And(cloud_confidence.lt(2))
        .And(cloud_shadow_confidence.lt(2))
        .And(snow_ice_confidence.lt(2))
        .And(cirrus_confidence.lt(2))
    )
    return image.updateMask(clean_mask)


def add_ndvi_band(image: ee.Image) -> ee.Image:
    """
    Landsat 8 C2L2 yüzey yansıması (SR) bantlarından NDVI bandı üretir.

    SR_B4 (Red) ve SR_B5 (NIR) DN değerleri önce reflectance'a çevrilir:
        reflectance = DN * LANDSAT_SR_SCALE + LANDSAT_SR_OFFSET
    Ardından:
        NDVI = (NIR - Red) / (NIR + Red)

    NDVI bandı QA maskesini miras alır (apply_qa_mask önce çağrılmalı). LST ile
    aynı QA mask / grid / composite mantığını paylaşması için ayrı maske kurulmaz.
    """
    red = (
        image.select(LANDSAT_RED_BAND)
        .multiply(LANDSAT_SR_SCALE)
        .add(LANDSAT_SR_OFFSET)
    )
    nir = (
        image.select(LANDSAT_NIR_BAND)
        .multiply(LANDSAT_SR_SCALE)
        .add(LANDSAT_SR_OFFSET)
    )
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    return image.addBands(ndvi)


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
        masked_daily_collection = daily_collection.map(apply_qa_mask)

        lst_median = masked_daily_collection.select("ST_B10").median().rename("ST_B10")
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
        "lst_composite_method": "qa_masked_daily_median",
        "qa_composite_method": "daily_mode",
        "qa_mask_applied_to_lst": True,
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
        "resolution": "30m",
        "created_at": datetime.now().isoformat(),
        "status": "daily_median_collection_prepared",
    }

    log.info(f"Gunluk composite sayisi: {unique_date_count}")

    return daily_collection, metadata


def get_landsat_baseline_window_median_collection(
    region: ee.Geometry,
    region_name: str,
    current_end_date: str,
    window_days: int,
    baseline_start: str = BASELINE_START_DATE,
    baseline_end: str = BASELINE_END_DATE,
) -> tuple[ee.ImageCollection, dict]:
    """
    Current period ile simetrik geçmiş yıl pencere medianları üretir.

    Örnek:
        current = 2023-07-17 -> 2023-08-31
        baseline = 2019/2020/2021/2022 aynı ay-gün pencereleri

    Her yıl için tek QA-maskeli ST_B10 median composite export edilir. Step5 bu
    yıllık pencere medianları üzerinden baseline mean/std hesaplar.
    """
    current_end_dt = datetime.strptime(current_end_date, "%Y-%m-%d")
    current_start_dt = current_end_dt - timedelta(days=window_days)
    current_year = current_end_dt.year
    baseline_start_year = datetime.strptime(baseline_start, "%Y-%m-%d").year
    baseline_end_year = datetime.strptime(baseline_end, "%Y-%m-%d").year
    baseline_years = [
        year
        for year in range(baseline_start_year, baseline_end_year + 1)
        if year < current_year
    ]

    if not baseline_years:
        raise ValueError(
            "Pencere-simetrik baseline için current yılından önce en az bir baseline yılı gerekir."
        )

    log.info("Pencere-simetrik Landsat baseline hazırlanıyor. Bölge: %s", region_name)
    log.info(
        "Current pencere referansı: %s -> %s (%s gün)",
        current_start_dt.strftime("%m-%d"),
        current_end_dt.strftime("%m-%d"),
        window_days,
    )
    log.info("Baseline yılları: %s", baseline_years)

    image_list = []
    window_records = []

    for year in baseline_years:
        window_start = current_start_dt.replace(year=year)
        window_end = current_end_dt.replace(year=year)
        start_text = window_start.strftime("%Y-%m-%d")
        end_text = window_end.strftime("%Y-%m-%d")

        raw_collection = (
            ee.ImageCollection(LANDSAT_COLLECTION)
            .filterBounds(region)
            .filterDate(start_text, end_text)
            .select(["ST_B10", "QA_PIXEL"])
            .map(lambda image: image.clip(region))
        )
        scene_count = raw_collection.size().getInfo()

        if scene_count == 0:
            log.warning(
                "%s yılı için %s -> %s penceresinde Landsat sahnesi bulunamadı.",
                year,
                start_text,
                end_text,
            )
            continue

        masked_collection = raw_collection.map(apply_qa_mask)
        lst_median = masked_collection.select("ST_B10").median().rename("ST_B10")
        qa_mode = raw_collection.select("QA_PIXEL").mode().rename("QA_PIXEL").toUint16()
        valid_count = (
            masked_collection
            .select("ST_B10")
            .count()
            .rename("Baseline_Window_Valid_Count")
            .toFloat()
        )

        window_image = (
            lst_median
            .addBands(qa_mode)
            .addBands(valid_count)
            .clip(region)
            .set("export_date", end_text)
            .set("window_start", start_text)
            .set("window_end", end_text)
            .set("baseline_year", year)
            .set("source_image_count", scene_count)
        )
        image_list.append(window_image)
        window_records.append(
            {
                "year": year,
                "window_start": start_text,
                "window_end": end_text,
                "scene_count": scene_count,
            }
        )

    if not image_list:
        raise ValueError("Hiçbir baseline yılı için pencere medianı üretilemedi.")

    baseline_collection = ee.ImageCollection(image_list)
    export_dates = [record["window_end"] for record in window_records]

    metadata = {
        "gee_project": GEE_PROJECT,
        "region_name": region_name,
        "collection": LANDSAT_COLLECTION,
        "bands": ["ST_B10", "QA_PIXEL", "Baseline_Window_Valid_Count"],
        "method": "window_symmetric_baseline",
        "current_reference_end_date": current_end_date,
        "window_days": window_days,
        "baseline_years": baseline_years,
        "windows": window_records,
        "daily_dates": export_dates,
        "daily_composite_count": len(export_dates),
        "lst_composite_method": "qa_masked_window_median_per_year",
        "qa_composite_method": "window_mode_per_year",
        "qa_mask_applied_to_lst": True,
        "qa_water_bit_preserved": True,
        "resolution": "30m",
        "created_at": datetime.now().isoformat(),
        "status": "window_symmetric_baseline_collection_prepared",
    }

    log.info("Pencere-simetrik baseline pencere sayısı: %s", len(export_dates))
    return baseline_collection, metadata

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
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=window_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    # Pencere içindeki tüm ayları hesapla; yalnızca penceredeki ayları filtrele.
    # Sabit 6-9 filtresi, pencere yaz dışına taşarsa sıfır sahne döndürürdü.
    window_months = sorted({
        (start_dt.month + i) % 12 or 12
        for i in range((end_dt - start_dt).days + 1)
    })
    month_start = min(window_months)
    month_end = max(window_months)

    log.info(f"Current period median hazırlanıyor:")
    log.info(f"  Bölge: {region_name}")
    log.info(f"  Pencere: {start_date} -> {end_date} ({window_days} gün)")
    log.info(f"  Ay filtresi: {month_start}-{month_end}")

    collection = (
        ee.ImageCollection(LANDSAT_COLLECTION)
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.calendarRange(month_start, month_end, "month"))
        .select(["ST_B10", "QA_PIXEL"])
        .map(lambda img: img.clip(region))
        .map(apply_qa_mask)
        .select("ST_B10")
    )

    scene_count = collection.size().getInfo()

    if scene_count == 0:
        raise ValueError(
            f"{region_name} için {start_date} - {end_date} arasında "
            f"(ay filtresi: {month_start}-{month_end}) Landsat sahnesi bulunamadı."
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
    current_median_lst = (
        collection
        .median()
        .multiply(LANDSAT_SCALE)
        .add(LANDSAT_OFFSET)
        .subtract(273.15)
        .rename("Current_Period_LST_Celsius")
        .toFloat()
    )

    current_median = (
        current_median_lst
        .addBands(current_valid_count.toFloat())
        .toFloat()
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
        "months_filter": f"{month_start}-{month_end}",
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
        "resolution": "30m",
        "created_at": datetime.now().isoformat(),
        "status": "current_period_median_prepared"
    }
    
    log.info("Current period median başarıyla oluşturuldu.")
    
    return current_median, metadata


# =============================================================================
# 2b. NDVI CURRENT PERIOD + BASELINE (LST ile aynı mask/grid/pencere mantığı)
# =============================================================================
def get_current_period_ndvi_median(
    region: ee.Geometry,
    region_name: str,
    end_date: str,
    window_days: int = CURRENT_PERIOD_DAYS,
) -> tuple[ee.Image, dict]:
    """
    Current period için QA-maskeli NDVI median composite üretir.

    LST current period ile birebir aynı pencere, ay filtresi, QA mask ve median
    composite mantığını kullanır; tek fark NDVI bandının seçilmesidir.

    Çıktı bantları:
        - Current_Period_NDVI         : NDVI median
        - Current_Period_NDVI_Valid_Count : QA-temiz gözlem sayısı
    """
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=window_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    window_months = sorted({
        (start_dt.month + i) % 12 or 12
        for i in range((end_dt - start_dt).days + 1)
    })
    month_start = min(window_months)
    month_end = max(window_months)

    log.info("Current period NDVI median hazırlanıyor. Bölge: %s", region_name)
    log.info("  Pencere: %s -> %s (%s gün)", start_date, end_date, window_days)
    log.info("  Ay filtresi: %s-%s", month_start, month_end)

    collection = (
        ee.ImageCollection(LANDSAT_COLLECTION)
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.calendarRange(month_start, month_end, "month"))
        .map(lambda img: img.clip(region))
        .map(apply_qa_mask)
        .map(add_ndvi_band)
        .select("NDVI")
    )

    scene_count = collection.size().getInfo()
    if scene_count == 0:
        raise ValueError(
            f"{region_name} için {start_date} - {end_date} arasında "
            f"(ay filtresi: {month_start}-{month_end}) NDVI sahnesi bulunamadı."
        )

    log.info("  Current period NDVI sahne sayısı: %s", scene_count)

    ndvi_valid_count = (
        collection
        .count()
        .rename("Current_Period_NDVI_Valid_Count")
        .toFloat()
        .clip(region)
    )

    ndvi_median = (
        collection
        .median()
        .rename("Current_Period_NDVI")
        .toFloat()
    )

    current_ndvi = (
        ndvi_median
        .addBands(ndvi_valid_count)
        .toFloat()
        .clip(region)
    )

    metadata = {
        "gee_project": GEE_PROJECT,
        "region_name": region_name,
        "collection": LANDSAT_COLLECTION,
        "product": "NDVI",
        "source_bands": [LANDSAT_RED_BAND, LANDSAT_NIR_BAND],
        "sr_scale": LANDSAT_SR_SCALE,
        "sr_offset": LANDSAT_SR_OFFSET,
        "window_start": start_date,
        "window_end": end_date,
        "window_days": window_days,
        "scene_count": scene_count,
        "months_filter": f"{month_start}-{month_end}",
        "composite_method": "median",
        "qa_mask_applied": True,
        "output_bands": [
            "Current_Period_NDVI",
            "Current_Period_NDVI_Valid_Count",
        ],
        "qa_water_bit_preserved": True,
        "resolution": "30m",
        "created_at": datetime.now().isoformat(),
        "status": "current_period_ndvi_median_prepared",
    }

    log.info("Current period NDVI median başarıyla oluşturuldu.")
    return current_ndvi, metadata


def get_landsat_baseline_window_ndvi_collection(
    region: ee.Geometry,
    region_name: str,
    current_end_date: str,
    window_days: int,
    baseline_start: str = BASELINE_START_DATE,
    baseline_end: str = BASELINE_END_DATE,
) -> tuple[ee.ImageCollection, dict]:
    """
    NDVI için pencere-simetrik baseline median collection'ı üretir.

    LST baseline ile aynı yıl seçimi, pencere ve QA mask mantığını kullanır;
    her baseline yılı için tek QA-maskeli NDVI median composite döner.
    """
    current_end_dt = datetime.strptime(current_end_date, "%Y-%m-%d")
    current_start_dt = current_end_dt - timedelta(days=window_days)
    current_year = current_end_dt.year
    baseline_start_year = datetime.strptime(baseline_start, "%Y-%m-%d").year
    baseline_end_year = datetime.strptime(baseline_end, "%Y-%m-%d").year
    baseline_years = [
        year
        for year in range(baseline_start_year, baseline_end_year + 1)
        if year < current_year
    ]

    if not baseline_years:
        raise ValueError(
            "Pencere-simetrik NDVI baseline için current yılından önce en az bir "
            "baseline yılı gerekir."
        )

    log.info("Pencere-simetrik NDVI baseline hazırlanıyor. Bölge: %s", region_name)
    log.info("Baseline yılları: %s", baseline_years)

    image_list = []
    window_records = []

    for year in baseline_years:
        window_start = current_start_dt.replace(year=year)
        window_end = current_end_dt.replace(year=year)
        start_text = window_start.strftime("%Y-%m-%d")
        end_text = window_end.strftime("%Y-%m-%d")

        raw_collection = (
            ee.ImageCollection(LANDSAT_COLLECTION)
            .filterBounds(region)
            .filterDate(start_text, end_text)
            .map(lambda image: image.clip(region))
        )
        scene_count = raw_collection.size().getInfo()

        if scene_count == 0:
            log.warning(
                "%s yılı için %s -> %s penceresinde NDVI sahnesi bulunamadı.",
                year,
                start_text,
                end_text,
            )
            continue

        masked_collection = raw_collection.map(apply_qa_mask).map(add_ndvi_band)
        ndvi_median = masked_collection.select("NDVI").median().rename("NDVI")
        valid_count = (
            masked_collection
            .select("NDVI")
            .count()
            .rename("Baseline_Window_NDVI_Valid_Count")
            .toFloat()
        )

        window_image = (
            ndvi_median
            .addBands(valid_count)
            .clip(region)
            .set("export_date", end_text)
            .set("window_start", start_text)
            .set("window_end", end_text)
            .set("baseline_year", year)
            .set("source_image_count", scene_count)
        )
        image_list.append(window_image)
        window_records.append(
            {
                "year": year,
                "window_start": start_text,
                "window_end": end_text,
                "scene_count": scene_count,
            }
        )

    if not image_list:
        raise ValueError("Hiçbir baseline yılı için NDVI pencere medianı üretilemedi.")

    baseline_collection = ee.ImageCollection(image_list)
    export_dates = [record["window_end"] for record in window_records]

    metadata = {
        "gee_project": GEE_PROJECT,
        "region_name": region_name,
        "collection": LANDSAT_COLLECTION,
        "product": "NDVI",
        "source_bands": [LANDSAT_RED_BAND, LANDSAT_NIR_BAND],
        "bands": ["NDVI", "Baseline_Window_NDVI_Valid_Count"],
        "method": "window_symmetric_baseline",
        "current_reference_end_date": current_end_date,
        "window_days": window_days,
        "baseline_years": baseline_years,
        "windows": window_records,
        "daily_dates": export_dates,
        "daily_composite_count": len(export_dates),
        "lst_composite_method": "qa_masked_window_median_per_year",
        "qa_mask_applied_to_ndvi": True,
        "qa_water_bit_preserved": True,
        "resolution": "30m",
        "created_at": datetime.now().isoformat(),
        "status": "window_symmetric_ndvi_baseline_collection_prepared",
    }

    log.info("Pencere-simetrik NDVI baseline pencere sayısı: %s", len(export_dates))
    return baseline_collection, metadata


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


def prepare_landsat_anomaly_inputs(
    region: ee.Geometry,
    region_name: str,
    current_end_date: str = CURRENT_PERIOD_END_DATE,
    window_days: int = CURRENT_PERIOD_DAYS,
    baseline_start: str = BASELINE_START_DATE,
    baseline_end: str = BASELINE_END_DATE,
) -> dict:
    """
    Landsat anomaly pipeline girdilerini tek yerde hazırlar.

    Bu helper Step3 ana akışı ve Step4 standalone yolu tarafından ortak kullanılır.
    Böylece baseline yılları, current period penceresi ve bölge seçimi config ile
    aynı kalır; Step4 tek başına çalıştığında eski daily-baseline yoluna düşmez.
    """
    log.info("Pencere-simetrik Landsat anomaly girdileri hazırlanıyor.")
    log.info(
        "Config: region=%s, current_end=%s, window_days=%s, baseline=%s -> %s",
        region_name,
        current_end_date,
        window_days,
        baseline_start,
        baseline_end,
    )

    landsat_timeseries, ts_metadata = get_landsat_baseline_window_median_collection(
        region=region,
        region_name=region_name,
        current_end_date=current_end_date,
        window_days=window_days,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
    )

    current_median, current_metadata = get_current_period_median(
        region=region,
        region_name=region_name,
        end_date=current_end_date,
        window_days=window_days,
    )

    result = {
        "landsat_timeseries": landsat_timeseries,
        "landsat_metadata": ts_metadata,
        "current_median": current_median,
        "current_metadata": current_metadata,
    }

    # NDVI ürünleri (yeni bilimsel yön). LST ile aynı pencere/QA mantığını paylaşır.
    if ENABLE_NDVI_EXPORT:
        ndvi_timeseries, ndvi_ts_metadata = get_landsat_baseline_window_ndvi_collection(
            region=region,
            region_name=region_name,
            current_end_date=current_end_date,
            window_days=window_days,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
        )
        current_ndvi, current_ndvi_metadata = get_current_period_ndvi_median(
            region=region,
            region_name=region_name,
            end_date=current_end_date,
            window_days=window_days,
        )
        result.update(
            {
                "ndvi_timeseries": ndvi_timeseries,
                "ndvi_metadata": ndvi_ts_metadata,
                "current_ndvi": current_ndvi,
                "current_ndvi_metadata": current_ndvi_metadata,
            }
        )

    return result


# =============================================================================
# ANA AKIŞ
# =============================================================================
def main() -> None:
    log.info("=" * 60)
    log.info("STEP 3 BAŞLIYOR")
    log.info("=" * 60)

    init_gee()
    regions = build_regions()

    step3_result = prepare_landsat_anomaly_inputs(
        region=regions[REGION_NAME],
        region_name=REGION_NAME,
        current_end_date=CURRENT_PERIOD_END_DATE,
        window_days=CURRENT_PERIOD_DAYS,
        baseline_start=BASELINE_START_DATE,
        baseline_end=BASELINE_END_DATE,
    )

    # Metadata kaydetme
    combined_metadata = {
        "baseline_timeseries": step3_result["landsat_metadata"],
        "current_period": step3_result["current_metadata"],
    }
    if "ndvi_metadata" in step3_result:
        combined_metadata["ndvi_baseline_timeseries"] = step3_result["ndvi_metadata"]
        combined_metadata["ndvi_current_period"] = step3_result["current_ndvi_metadata"]
    
    metadata_path = save_metadata(combined_metadata)

    log.info("=" * 60)
    log.info("STEP 3 TAMAMLANDI")
    log.info(f"Metadata dosyası: {metadata_path}")
    log.info("\nHazırlanan çıktılar:")
    log.info("  1. Baseline zaman serisi (ee.ImageCollection)")
    log.info("  2. Current period median (ee.Image)")
    log.info("Sonraki adım: step4 (export)")
    log.info("=" * 60)

    return step3_result

if __name__ == "__main__":
    main()