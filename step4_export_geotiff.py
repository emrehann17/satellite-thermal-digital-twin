"""
step4_export_geotiff.py

Yapılanlar:
    - GEE bağlantısını başlatmak
    - Çalışma bölgelerini almak
    - Step2'den MODIS 5 yıllık yaz ortalaması LST görüntüsünü üretmek
    - Step3'ten Landsat yüksek çözünürlüklü LST görüntüsünü üretmek
    - Bu görüntüleri Google Drive'a GeoTIFF olarak export etmek
    - Export metadata bilgisini JSON olarak kaydetmek

NOT:
    Bu adım işleme veya görselleştirme yapmaz.
    Sadece Step2 ve Step3 çıktılarını dışa aktarır.
"""
import json
from datetime import datetime
from pathlib import Path

import ee

from core.config import *
from core.gee_utils import init_gee
from core.regions import build_regions
from core.io_utils import setup_logger


from step2_modis_5year_mean import process_summer_mean
from step3_landsat_lst import get_landsat_timeseries_collection, process_landsat_lst


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs" / "step4"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

log, log_file = setup_logger("step4")

# =============================================================================
# 1. GOOGLE DRIVE EXPORT
# =============================================================================

def export_image_to_drive(
    image : ee.Image,
    region : ee.Geometry,
    description : str,
    folder : str,
    file_name_prefix : str,
    scale : int,
    crs : str = "EPSG:4326"
) -> dict:
    

    """
    Verilen ee.Image nesnesini belirtilen bölge,
    ölçek ve koordinat referans sistemine göre Google Drive'a GeoTIFF olarak export eder.
        
    Dönüş: 
        {
            "description": description,
            "folder": folder,
            "file_name_prefix": file_name_prefix,
            "scale": scale,
            "crs": crs,
            "region": region.toGeoJSONString()
        }
    """
    log.info(f"Export görevi hazırlanıyor: {description}")

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=file_name_prefix,
        region = region,
        scale=scale,
        crs=crs,
        maxPixels=1e13,
        fileFormat="GeoTIFF"
    )

    task.start()
    status = task.status()

    log.info(f"Export görevi başlatıldı: {description}")
    log.info(f"Task ID: {status.get('id')}")
    log.info(f"Task state: {status.get('state')}")

    return {
        "description": description,
        "folder": folder,
        "file_name_prefix" : file_name_prefix,
        "scale": scale,
        "crs": crs,
        "file_format": "GeoTIFF",
        "task_id": status.get("id"),
        "task_state": status.get("state"),
        "started_at": datetime.now().isoformat()
    }

# =============================================================================
# 2. LANDSAT TIMESERIES COLLECTION EXPORT
# =============================================================================
def export_landsat_timeseries_lst_and_qa_to_drive(
    collection: ee.ImageCollection,
    region: ee.Geometry,
    folder: str,
    file_prefix: str,
    scale: int = 30,
    crs: str = "EPSG:4326",
    max_exports: int = 20
) -> list[dict]:
    """
    Landsat zaman serisi collection'ındaki her görüntü için
    ST_B10 ve QA_PIXEL bantlarını ayrı GeoTIFF dosyaları olarak export eder.

    Çıktılar:
        - landsat_lst_YYYY-MM-DD.tif
        - landsat_lst_YYYY-MM-DD_qa.tif
    """
    image_count = collection.size().getInfo()
    export_count = min(image_count, max_exports)

    log.info(f"Landsat zaman serisi toplam görüntü sayısı: {image_count}")
    log.info(f"Export edilecek görüntü sayısı: {export_count}")

    limited_collection = (
        collection
        .sort("system:time_start")
        .limit(export_count)
    )

    collection_list = limited_collection.toList(export_count)

    export_metadata = []

    for i in range(export_count):
        image = ee.Image(collection_list.get(i))

        date_text = (
            ee.Date(image.get("system:time_start"))
            .format("YYYY-MM-dd")
            .getInfo()
        )

        lst_image = image.select("ST_B10")
        qa_image = image.select("QA_PIXEL")

        unique_suffix = f"{i+1:03d}"

        lst_description = f"export_{file_prefix}_{date_text}_{unique_suffix}"
        lst_file_name = f"{file_prefix}_{date_text}_{unique_suffix}"

        qa_description = f"export_{file_prefix}_{date_text}_{unique_suffix}_qa"
        qa_file_name = f"{file_prefix}_{date_text}_{unique_suffix}_qa"

        log.info(f"[{i + 1}/{export_count}] LST export başlatılıyor: {lst_file_name}")

        lst_task = ee.batch.Export.image.toDrive(
            image=lst_image,
            description=lst_description,
            folder=folder,
            fileNamePrefix=lst_file_name,
            region=region,
            scale=scale,
            crs=crs,
            maxPixels=1e13,
            fileFormat="GeoTIFF"
        )
        lst_task.start()
        lst_status = lst_task.status()

        log.info(f"[{i + 1}/{export_count}] QA export başlatılıyor: {qa_file_name}")

        qa_task = ee.batch.Export.image.toDrive(
            image=qa_image,
            description=qa_description,
            folder=folder,
            fileNamePrefix=qa_file_name,
            region=region,
            scale=scale,
            crs=crs,
            maxPixels=1e13,
            fileFormat="GeoTIFF"
        )
        qa_task.start()
        qa_status = qa_task.status()

        export_metadata.append({
            "index": i + 1,
            "date": date_text,
            "folder": folder,
            "scale": scale,
            "crs": crs,
            "lst": {
                "band": "ST_B10",
                "description": lst_description,
                "file_name_prefix": lst_file_name,
                "task_id": lst_status.get("id"),
                "task_state": lst_status.get("state")
            },
            "qa": {
                "band": "QA_PIXEL",
                "description": qa_description,
                "file_name_prefix": qa_file_name,
                "task_id": qa_status.get("id"),
                "task_state": qa_status.get("state")
            }
        })

    return export_metadata


# =============================================================================
# 3. METADATA KAYDETME
# =============================================================================
def save_metadata(metadata: dict, filename: str = "step4_metadata.json") -> Path:
    """
    Step4 metadata bilgisini JSON olarak kaydeder.
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
    log.info("STEP 4 BAŞLIYOR")
    log.info("=" * 60)

    init_gee()
    regions = build_regions()
    region = regions["dogu_akdeniz"]

    log.info("Step2 MODIS görüntüsü üretiliyor.")
    modis_image, modis_processing_metadata = process_summer_mean(
        region=region,
        region_name="dogu_akdeniz",
        start=START_DATE,
        end=END_DATE
    )

    """log.info("Step3 Landsat tek görüntü LST ürünü hazırlanıyor.")
    landsat_image, landsat_processing_metadata = process_landsat_lst(
        region=region,
        region_name="dogu_akdeniz",
        start=START_DATE,
        end=END_DATE
    )"""

    log.info("Step3 Landsat zaman serisi koleksiyonu hazırlanıyor.")
    landsat_timeseries_collection, landsat_timeseries_metadata = get_landsat_timeseries_collection(
        region=region,
        region_name="dogu_akdeniz",
        start=START_DATE,
        end=END_DATE
    )

    modis_export_metadata = export_image_to_drive(
        image=modis_image,
        region=region,
        description=MODIS_EXPORT_DESCRIPTION,
        folder=EXPORT_FOLDER,
        file_name_prefix=MODIS_FILE_PREFIX,
        scale=1000
    )

    landsat_timeseries_exports = export_landsat_timeseries_lst_and_qa_to_drive(
        collection=landsat_timeseries_collection,
        region=region,
        folder=EXPORT_FOLDER,
        file_prefix="landsat_lst_dogu_akdeniz",
        max_exports=MAX_LANDSAT_TIMESERIES_EXPORTS
    )

    metadata = {
        "step": "step4_export_geotiff",
        "region_name": "dogu_akdeniz",
        "date_start": START_DATE,
        "date_end": END_DATE,
        "export_folder": EXPORT_FOLDER,
        "max_landsat_timeseries_exports": MAX_LANDSAT_TIMESERIES_EXPORTS,
        "created_at": datetime.now().isoformat(),
        "log_file": str(log_file),
        "modis_processing_metadata": modis_processing_metadata,
        "landsat_timeseries_metadata": landsat_timeseries_metadata,
        "exports": {
            "modis": modis_export_metadata,
            "landsat_timeseries": landsat_timeseries_exports
        },
        "status": "export_tasks_started"
    }
    metadata_path = save_metadata(metadata)

    log.info("=" * 60)
    log.info("STEP 4 TAMAMLANDI")
    log.info(f"Export metadata dosyası: {metadata_path}")
    log.info("Google Drive export görevleri başlatıldı.")
    log.info("=" * 60)

if __name__ == "__main__":
    main()


#LOG DOSYALARI UST USTE BINIYOR VE LOG DOSYASI BOŞ GÖRÜNÜYOR BUNU ÇÖZ