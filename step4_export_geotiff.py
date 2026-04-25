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


from step2_export_5year_mean import process_summer_mean
from step3_landsat_lst import process_landsat_lst


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs" / "step4"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

log, log_file = setup_logger("step4")

# =============================================================================
# 1. GOOGLE DRIVE EXPORT
# =============================================================================

def export_image_to_drive(
    image = ee.Image,
    region = ee.Geometry,
    description = str,
    folder = str,
    file_name_prefix = str,
    scale = int,
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
# 2. METADATA KAYDETME
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

    log.info("Step3 Landsat LST görüntüsü üretiliyor.")
    landsat_image, landsat_processing_metadata = process_landsat_lst(
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

    landsat_export_metadata = export_image_to_drive(
        image=landsat_image,
        region=region,
        description=LANDSAT_EXPORT_DESCRIPTION,
        folder=EXPORT_FOLDER,
        file_name_prefix=LANDSAT_FILE_PREFIX,
        scale=30
    )

    metadata = {
        "step": "step4_export_geotiff",
        "region_name": "dogu_akdeniz",
        "date_start": START_DATE,
        "date_end": END_DATE,
        "export_folder": EXPORT_FOLDER,
        "created_at": datetime.now().isoformat(),
        "modis_processing_metadata": modis_processing_metadata,
        "landsat_processing_metadata": landsat_processing_metadata,
        "exports": {
            "modis": modis_export_metadata,
            "landsat": landsat_export_metadata
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


    