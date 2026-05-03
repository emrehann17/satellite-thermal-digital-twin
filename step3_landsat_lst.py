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

def _set_export_date(image: ee.Image) -> ee.Image:
    """Adds a YYYY-MM-dd date key used for daily compositing/export."""
    export_date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
    return image.set("export_date", export_date)

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
# 2. LANDSAT TIMESERIES COLLECTION ÜRETME
# =============================================================================
def get_landsat_daily_median_collection(
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
    log.info(f"Landsat gunluk median composite hazirlaniyor. Bölge: {region_name}")
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
        .map(_set_export_date)
    )

    filtered_count = filtered_collection.size().getInfo()

    if filtered_count == 0:
        raise ValueError(
            f"{region_name} bölgesi için {start} - {end} aralığında yaz aylarına ait Landsat görüntüsü bulunamadı."
        )
    
    unique_date = filtered_collection.aggregate_array("export_date").distinct().sort()
    unique_date_count = unique_date.size().getInfo()
    daily_date = unique_date.getInfo()
    log.info(f"Yaz aylarında bulunan benzersiz tarih sayısı: {unique_date_count}")

    def build_daily_composite(date_value):
        date_value = ee.String(date_value)

        daily_collection = filtered_collection.filter(
            ee.Filter.eq("export_date", date_value)
            )
        
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
    
    daily_collection = ee.ImageCollection(unique_date.map(build_daily_composite))

    '''
        ST_B10 fiziksel/sayısal bir ölçüm bandı: yüzey sıcaklığı DN değeri. Aynı tarihte birden fazla sahne varsa bu değerleri median ile birleştirmek mantıklı;
        uç değerleri yumuşatır ve tek bir temsil üretir.

        QA_PIXEL ise sıcaklık gibi sürekli bir ölçüm değil, bit maskesi. İçinde “bulut var mı”, “gölge var mı”, “snow var mı”, “fill mi” gibi bayraklar bit bit kodlanır.
        Bu yüzden median almak teknik olarak yanlış olabilir; iki QA kodunun ortanca değeri gerçek bir QA durumu temsil etmeyebilir.
        mode ise aynı gün sahnelerindeki en sık görülen QA kodunu seçer,
        bit maskesi için daha savunulabilir bir basit composite yöntemidir.
    '''
    
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
        "daily_composite_count": unique_date_count,
        "daily_dates": daily_date,
        "lst_composite_method": "median",
        "qa_composite_method": "mode",
        "all_image_count": base_count,
        "created_at": datetime.now().isoformat(),
        "status": "timeseries_collection_prepared"
    }

    log.info(f"Tüm Landsat görüntü sayısı: {base_count}")
    log.info(f"Yaz ayları filtreli Landsat görüntü sayısı: {filtered_count}")
    log.info(f"Gunluk composite sayisi: {unique_date_count}")

    return daily_collection, metadata

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

    landsat_timeseries, metadata = get_landsat_daily_median_collection(
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