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
import requests
from datetime import datetime
from pathlib import Path

import ee

from core.config import (
    START_DATE,
    END_DATE,
    REGION_NAME,
    EXPORT_FOLDER,
    MAX_LANDSAT_DAILY_EXPORTS,
    EXPORT_CRS,
    MODIS_EXPORT,
    LANDSAT_EXPORT,
    ENABLE_MODIS_EXPORT,
    ENABLE_LANDSAT_EXPORT,
    DOWNLOAD_MODE,
    BASELINE_START_DATE,
    BASELINE_END_DATE,
    CURRENT_PERIOD_DAYS,
    CURRENT_PERIOD_END_DATE,
)

# Auto-drive için ek config
try:
    from core.config import (
        AUTO_DRIVE_DOWNLOAD_DIR,
        AUTO_DRIVE_CHECK_INTERVAL,
        AUTO_DRIVE_TIMEOUT
    )
except ImportError:
    AUTO_DRIVE_DOWNLOAD_DIR = "data"
    AUTO_DRIVE_CHECK_INTERVAL = 30
    AUTO_DRIVE_TIMEOUT = 3600

from core.gee_utils import init_gee
from core.regions import build_regions
from core.io_utils import setup_logger


from step2_modis_5year_mean import process_summer_mean
from step3_landsat_lst import get_landsat_daily_median_collection, get_current_period_median

# Otomatik Drive indirme için
try:
    from core.drive_downloader import TaskPoller, batch_export_and_wait
    DRIVE_DOWNLOADER_AVAILABLE = True
except ImportError:
    DRIVE_DOWNLOADER_AVAILABLE = False
    if DOWNLOAD_MODE == "auto_drive":
        '''log.warning(
            "drive_downloader module not found. "
            "Auto-drive mode will not work. Falling back to 'drive' mode."
        )'''


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs" / "step4"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

log, log_file = setup_logger("step4")

# =============================================================================
# 1. GOOGLE DRIVE EXPORT
# =============================================================================

def export_image_to_drive(
    image: ee.Image,
    region: ee.Geometry,
    description: str,
    folder: str,
    file_name_prefix: str,
    scale: int,
    crs: str = EXPORT_CRS,
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
# 2. LANDSAT DAILY LST+QA EXPORT 
# =============================================================================
def export_landsat_timeseries_lst_and_qa_to_drive(
collection: ee.ImageCollection,
    date_list: list[str],
    region: ee.Geometry,
    folder: str,
    file_prefix: str,
    scale: int = LANDSAT_EXPORT["scale"],
    crs: str = EXPORT_CRS,
    max_exports: int = 20,
    download_mode: str = "drive",
) -> list[dict]:
    """
    Landsat günlük composite collection'ındaki her görüntü için
    ST_B10 ve QA_PIXEL bantlarını export eder.

    download_mode:
        - "drive"      -> Google Drive export (manuel indirme)
        - "direct"     -> ee.Image.getDownloadURL ile doğrudan yerel indirme
        - "auto_drive" -> Drive export + otomatik task polling + indirme talimatı
    """
    if not date_list:
        log.warning("date_list boş geldi. Landsat export başlatılmadı.")
        return []

    export_count = min(len(date_list), max_exports)

    if export_count == 0:
        log.warning("export_count = 0. Landsat export başlatılmadı.")
        return []

    log.info(f"Export edilecek günlük Landsat görüntü sayısı: {export_count}")
    log.info(f"Çalışma modu: {download_mode}")
    
    # Auto-drive modunda drive'a geri dön eğer module yüklü değilse
    if download_mode == "auto_drive" and not DRIVE_DOWNLOADER_AVAILABLE:
        log.warning("drive_downloader bulunamadı, 'drive' moduna geçiliyor")
        download_mode = "drive"

    limited_collection = (
        collection
        .sort("export_date")
        .limit(export_count)
    )

    collection_list = limited_collection.toList(export_count)
    export_metadata = []
    
    # Task listesi (auto_drive için)
    all_tasks = []

    lst_dir, qa_dir = ensure_direct_download_dirs() if download_mode == "direct" else (None, None)

    for i in range(export_count):
        image = ee.Image(collection_list.get(i))
        date_text = date_list[i]

        lst_image = image.select("ST_B10")
        qa_image = image.select("QA_PIXEL")

        lst_description = f"export_{file_prefix}_{date_text}"
        lst_file_name = f"{file_prefix}_{date_text}"

        qa_description = f"export_{file_prefix}_{date_text}_qa"
        qa_file_name = f"{file_prefix}_{date_text}_qa"

        if download_mode in ["drive", "auto_drive"]:
            log.info(f"[{i + 1}/{export_count}] LST Drive export: {lst_file_name}")

            lst_task = ee.batch.Export.image.toDrive(
                image=lst_image,
                description=lst_description,
                folder=folder,
                fileNamePrefix=lst_file_name,
                region=region,
                scale=scale,
                crs=crs,
                maxPixels=1e13,
                fileFormat="GeoTIFF",
            )
            lst_task.start()
            lst_status = lst_task.status()
            
            if download_mode == "auto_drive":
                all_tasks.append(("lst", lst_task, lst_file_name))

            log.info(f"[{i + 1}/{export_count}] QA Drive export: {qa_file_name}")

            qa_task = ee.batch.Export.image.toDrive(
                image=qa_image,
                description=qa_description,
                folder=folder,
                fileNamePrefix=qa_file_name,
                region=region,
                scale=scale,
                crs=crs,
                maxPixels=1e13,
                fileFormat="GeoTIFF",
            )
            qa_task.start()
            qa_status = qa_task.status()
            
            if download_mode == "auto_drive":
                all_tasks.append(("qa", qa_task, qa_file_name))

            export_metadata.append({
                "date": date_text,
                "mode": download_mode,
                "folder": folder,
                "scale": scale,
                "crs": crs,
                "lst": {
                    "band": "ST_B10",
                    "description": lst_description,
                    "file_name_prefix": lst_file_name,
                    "task_id": lst_status.get("id"),
                    "task_state": lst_status.get("state"),
                },
                "qa": {
                    "band": "QA_PIXEL",
                    "description": qa_description,
                    "file_name_prefix": qa_file_name,
                    "task_id": qa_status.get("id"),
                    "task_state": qa_status.get("state"),
                },
            })

        elif download_mode == "direct":
            log.info(f"[{i + 1}/{export_count}] LST direct download: {lst_file_name}")
            lst_meta = download_image_via_url(
                image=lst_image,
                region=region,
                output_path=lst_dir / f"{lst_file_name}.tif",
                scale=scale,
                crs=crs,
            )

            log.info(f"[{i + 1}/{export_count}] QA direct download: {qa_file_name}")
            qa_meta = download_image_via_url(
                image=qa_image,
                region=region,
                output_path=qa_dir / f"{qa_file_name}.tif",
                scale=scale,
                crs=crs,
            )

            export_metadata.append({
                "date": date_text,
                "mode": "direct",
                "scale": scale,
                "crs": crs,
                "lst": {
                    "band": "ST_B10",
                    **lst_meta,
                },
                "qa": {
                    "band": "QA_PIXEL",
                    **qa_meta,
                },
            })

        else:
            raise ValueError(f"Desteklenmeyen download_mode: {download_mode}")
    
    # Auto-drive modunda: Task'lerin tamamlanmasını bekle
    if download_mode == "auto_drive" and all_tasks:
        log.info("\n" + "=" * 60)
        log.info("AUTO-DRIVE: Task polling başlatılıyor...")
        log.info(f"Toplam {len(all_tasks)} task bekleniyor")
        log.info(f"Kontrol aralığı: {AUTO_DRIVE_CHECK_INTERVAL}s")
        log.info(f"Timeout: {AUTO_DRIVE_TIMEOUT}s")
        log.info("=" * 60)
        
        poller = TaskPoller(
            check_interval=AUTO_DRIVE_CHECK_INTERVAL,
            timeout=AUTO_DRIVE_TIMEOUT,
            output_dir=Path(AUTO_DRIVE_DOWNLOAD_DIR)
        )
        
        # Tüm task'leri bekle
        tasks_only = [task for _, task, _ in all_tasks]
        results = poller.wait_for_tasks(tasks_only)
        
        # Metadata'ya sonuçları ekle
        for meta in export_metadata:
            lst_task_id = meta["lst"]["task_id"]
            qa_task_id = meta["qa"]["task_id"]
            
            meta["lst"]["task_completed"] = results.get(lst_task_id, False)
            meta["qa"]["task_completed"] = results.get(qa_task_id, False)
        
        # İndirme talimatları
        successful_count = sum(results.values())
        log.info("\n" + "=" * 60)
        log.info(f"Task polling tamamlandı: {successful_count}/{len(tasks_only)} başarılı")
        log.info("\nÖNEMLİ: Dosyalar Google Drive'a export edildi.")
        log.info(f"Google Drive/{folder}/ klasöründen dosyaları manuel olarak indir:")
        log.info("")
        
        for task_type, task, filename in all_tasks:
            task_id = task.status()['id']
            if results.get(task_id, False):
                if task_type == "lst":
                    target_dir = "data/landsat_timeseries/"
                else:  # qa
                    target_dir = "data/landsat_qa/"
                
                log.info(f"  ✓ {filename}.tif -> {target_dir}")
        
        log.info("\nDrive'dan indirme talimatları:")
        log.info("  1. Google Drive'ı aç")
        log.info(f"  2. {folder}/ klasörüne git")
        log.info("  3. LST dosyalarını -> data/landsat_timeseries/ klasörüne kopyala")
        log.info("  4. QA dosyalarını -> data/landsat_qa/ klasörüne kopyala")
        log.info("=" * 60)

    return export_metadata

# =============================================================================
# 3. DOWNLOAD MODE - GOOGLE DRIVE'DAN DOSYA İNDİRME
# =============================================================================
def ensure_direct_download_dirs() -> tuple[Path, Path]:
    """
    Direct download için yerel klasörleri hazırlar.
    """
    lst_dir = BASE_DIR / "data" / "landsat_timeseries"
    qa_dir = BASE_DIR / "data" / "landsat_qa"

    lst_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    return lst_dir, qa_dir


def download_image_via_url(
    image: ee.Image,
    region: ee.Geometry,
    output_path: Path,
    scale: int,
    crs: str = EXPORT_CRS
) -> dict:
    """
    ee.Image.getDownloadURL ile tek görüntüyü doğrudan indirir.
    Küçük bölgeler ve test amaçlı kullanıma uygundur.
    """
    url = image.getDownloadURL({
        "region": region,
        "scale": scale,
        "crs": crs,
        "format": "GEO_TIFF"
    }
    )

    log.info(f"Direct download başlatılıyor: {output_path.name}")

    response = requests.get(url, timeout=300)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    log.info(f"Direct download tamamlandı: {output_path}")

    return {
        "output_path": str(output_path),
        "scale": scale,
        "crs": crs,
        "download_method": "getDownloadURL",
        "downloaded_at": datetime.now().isoformat()
    }

# =============================================================================
# 4. METADATA KAYDETME
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
    region = regions[REGION_NAME]

    modis_export_metadata = None
    modis_processing_metadata = None

    if ENABLE_MODIS_EXPORT:
        log.info("Step2 MODIS görüntüsü üretiliyor.")
        modis_image, modis_processing_metadata = process_summer_mean(
            region=region,
            region_name=REGION_NAME,
            start=START_DATE,
            end=END_DATE
        )

        modis_export_metadata = export_image_to_drive(
            image=modis_image,
            region=region,
            description=MODIS_EXPORT["description"],
            folder=EXPORT_FOLDER,
            file_name_prefix=MODIS_EXPORT["file_name_prefix"],
            scale=MODIS_EXPORT["scale"]
        )
    else:
        log.info("MODIS export devre dışı bırakıldı.")
        

    landsat_timeseries_exports = []
    landsat_processing_metadata = None
    current_period_export = None
    current_period_metadata = None

    if ENABLE_LANDSAT_EXPORT:
        # 1. Baseline zaman serisi
        log.info("\nBaseline zaman serisi hazırlanıyor...")
        landsat_daily_collection, landsat_processing_metadata = get_landsat_daily_median_collection(
            region=region,
            region_name=REGION_NAME,
            start=BASELINE_START_DATE,
            end=BASELINE_END_DATE
        )

        landsat_timeseries_exports = export_landsat_timeseries_lst_and_qa_to_drive(
            collection=landsat_daily_collection,
            date_list=landsat_processing_metadata["daily_dates"],
            region=region,
            folder=EXPORT_FOLDER,
            file_prefix=LANDSAT_EXPORT["file_name_prefix"],
            scale=LANDSAT_EXPORT["scale"],
            max_exports=MAX_LANDSAT_DAILY_EXPORTS,
            download_mode=DOWNLOAD_MODE
        )
        
        # 2. Current period median (anomali için)
        log.info("\nCurrent period median hazırlanıyor...")
        current_median_image, current_period_metadata = get_current_period_median(
            region=region,
            region_name=REGION_NAME,
            end_date=CURRENT_PERIOD_END_DATE,
            window_days=CURRENT_PERIOD_DAYS
        )
        
        # Current period median'ı export et
        current_period_description = f"export_current_period_median_{CURRENT_PERIOD_DAYS}days"
        current_period_filename = f"landsat_current_period_{CURRENT_PERIOD_DAYS}days"
        
        if DOWNLOAD_MODE == "auto_drive":
            log.info(f"\nCurrent period median export (auto_drive): {current_period_filename}")
            
            # Task'i başlat
            current_task = ee.batch.Export.image.toDrive(
                image=current_median_image,
                description=current_period_description,
                folder=EXPORT_FOLDER,
                fileNamePrefix=current_period_filename,
                region=region,
                scale=LANDSAT_EXPORT["scale"],
                crs=EXPORT_CRS,
                maxPixels=1e13,
                fileFormat="GeoTIFF"
            )
            current_task.start()
            current_status = current_task.status()
            
            # Metadata kaydet
            current_period_export = {
                "description": current_period_description,
                "folder": EXPORT_FOLDER,
                "file_name_prefix": current_period_filename,
                "scale": LANDSAT_EXPORT["scale"],
                "crs": EXPORT_CRS,
                "task_id": current_status.get("id"),
                "task_state": current_status.get("state"),
                "started_at": datetime.now().isoformat()
            }
            
            # Task'in tamamlanmasını bekle
            log.info("\n" + "=" * 60)
            log.info("AUTO-DRIVE: Current period task bekleniyor...")
            log.info("=" * 60)
            
            poller = TaskPoller(
                check_interval=AUTO_DRIVE_CHECK_INTERVAL,
                timeout=AUTO_DRIVE_TIMEOUT,
                output_dir=Path(AUTO_DRIVE_DOWNLOAD_DIR) / "current_period"
            )
            
            success = poller.wait_for_task(current_task)
            current_period_export["task_completed"] = success
            
            if success:
                log.info("\n" + "=" * 60)
                log.info("CURRENT PERIOD EXPORT TAMAMLANDI")
                log.info(f"Google Drive/{EXPORT_FOLDER}/{current_period_filename}.tif")
                log.info("→ data/current_period/ klasörüne manuel olarak kopyala")
                log.info("=" * 60)
        else:
            # Standart drive veya direct export
            current_period_export = export_image_to_drive(
                image=current_median_image,
                region=region,
                description=current_period_description,
                folder=EXPORT_FOLDER,
                file_name_prefix=current_period_filename,
                scale=LANDSAT_EXPORT["scale"]
            )
        
    else:
        log.info("Landsat export devre dışı bırakıldı.")

        
    metadata = {
        "step": "step4_export_geotiff",
        "pipeline_stage": "online",
        "region_name": REGION_NAME,
        "baseline_date_start": BASELINE_START_DATE,
        "baseline_date_end": BASELINE_END_DATE,
        "current_period_end": CURRENT_PERIOD_END_DATE,
        "current_period_days": CURRENT_PERIOD_DAYS,
        "export_folder": EXPORT_FOLDER,
        "download_mode": DOWNLOAD_MODE,
        "created_at": datetime.now().isoformat(),
        "log_file": str(log_file),
        "manual_download_required": DOWNLOAD_MODE == "drive",
        "enabled_exports": {
            "modis": ENABLE_MODIS_EXPORT,
            "landsat": ENABLE_LANDSAT_EXPORT
        },
        "processing": {
            "modis": modis_processing_metadata,
            "landsat_baseline_timeseries": landsat_processing_metadata,
            "landsat_current_period": current_period_metadata
        },
        "exports": {
            "modis": modis_export_metadata,
            "landsat_baseline_timeseries": landsat_timeseries_exports,
            "landsat_current_period": current_period_export
        },
        "notes": {
            "modis_download_mode": "drive_only_for_now",
            "landsat_download_mode": DOWNLOAD_MODE,
            "anomaly_method": "z_score_window_based"
        },
        "status": "direct_download_completed" if DOWNLOAD_MODE == "direct" else "export_tasks_started"
    }
    metadata_path = save_metadata(metadata)

    log.info("=" * 60)
    log.info("STEP 4 TAMAMLANDI")
    log.info(f"Export metadata dosyası: {metadata_path}")

    if DOWNLOAD_MODE == "direct":
        log.info("Landsat dosyaları doğrudan yerel klasörlere indirildi.")
        log.info("Step5 çalıştırılabilir.")
    else:
        log.info("Google Drive export görevleri başlatıldı.")
        log.info("Dosyaları manuel indirip Step5 klasörlerine yerleştirmen gerekiyor.")
        log.info("\nExport edilen dosyalar:")
        log.info("  1. Baseline zaman serisi (ST_B10 + QA_PIXEL)")
        log.info("  2. Current period median (LST Celsius)")

    log.info("=" * 60)

    if DOWNLOAD_MODE == "direct":
        print("\nSTEP 4 tamamlandı. Landsat dosyaları yerel klasörlere indirildi.")
        print("Şimdi Step5'i çalıştırabilirsin:")
        print("python step5_preprocess_timeseries.py\n")
    else:
        print("\nSTEP 4 tamamlandı. Export görevleri Drive'a gönderildi.")
        print("Drive'dan dosyaları indirip Step5 klasörlerine yerleştirmen gerekiyor.\n")

if __name__ == "__main__":
    main()