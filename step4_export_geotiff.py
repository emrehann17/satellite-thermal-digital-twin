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
import time
from datetime import datetime
from pathlib import Path

import ee
import core.config as runtime_config

from core.config import (
    START_DATE,
    END_DATE,
    REGION_NAME,
    EXPORT_FOLDER,
    MAX_LANDSAT_DAILY_EXPORTS,
    EXPORT_CRS,
    MODIS_EXPORT,
    LANDSAT_EXPORT,
    LANDSAT_NDVI_EXPORT,
    ENABLE_MODIS_EXPORT,
    ENABLE_LANDSAT_EXPORT,
    ENABLE_NDVI_EXPORT,
    DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT,
    DRIVE_TASK_POLLING_ENABLED,
    DRIVE_TASK_POLL_INTERVAL_SECONDS,
    DRIVE_TASK_TIMEOUT_SECONDS,
    BASELINE_START_DATE,
    BASELINE_END_DATE,
    CURRENT_PERIOD_DAYS,
    CURRENT_PERIOD_END_DATE,
    SUMMER_MONTH_START,
    SUMMER_MONTH_END,
)

from core.gee_utils import init_gee
from core.regions import build_regions
from core.io_utils import setup_logger


from step2_modis_5year_mean import process_summer_mean
from step3_landsat_lst import prepare_landsat_anomaly_inputs


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs" / "step4"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

log, log_file = setup_logger("step4")
DRIVE_EXPORT_TASKS = []


def register_drive_task(
    task: ee.batch.Task,
    metadata_ref: dict,
    description: str,
    file_name_prefix: str,
    output_group: str,
) -> None:
    """
    Drive export task'ını polling aşamasında takip edebilmek için kaydeder.

    metadata_ref, daha sonra task state alanları güncellensin diye ilgili
    metadata sözlüğüne referans olarak tutulur.
    """
    DRIVE_EXPORT_TASKS.append(
        {
            "task": task,
            "metadata": metadata_ref,
            "description": description,
            "file_name_prefix": file_name_prefix,
            "output_group": output_group,
        }
    )


def get_legacy_download_mode() -> str | None:
    value = getattr(runtime_config, "DOWNLOAD_MODE", None)
    if value is None:
        return None

    normalized = str(value).strip().lower()
    return normalized or None


def log_drive_download_configuration() -> None:
    legacy_download_mode = get_legacy_download_mode()

    log.info(
        "Drive export config: polling=%s, step4b_auto_download=%s",
        DRIVE_TASK_POLLING_ENABLED,
        DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT,
    )

    if legacy_download_mode is not None:
        log.warning(
            "Legacy DOWNLOAD_MODE=%s bulundu. Step4 artik bu alanla davranis secmiyor; "
            "Drive indirme Step4b tarafindan yonetilir.",
            legacy_download_mode,
        )

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

    metadata = {
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
    register_drive_task(
        task=task,
        metadata_ref=metadata,
        description=description,
        file_name_prefix=file_name_prefix,
        output_group="modis" if "modis" in file_name_prefix.lower() else "current_period",
    )
    return metadata

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
    max_exports: int | None = 20,
) -> list[dict]:
    """
    Landsat günlük composite collection'ındaki her görüntü için
    ST_B10 ve QA_PIXEL bantlarını Google Drive'a export eder.
    """
    if not date_list:
        log.warning("date_list boş geldi. Landsat export başlatılmadı.")
        return []

    export_count = len(date_list) if max_exports is None else min(len(date_list), max_exports)

    if export_count == 0:
        log.warning("export_count = 0. Landsat export başlatılmadı.")
        return []

    log.info(f"Export edilecek günlük Landsat görüntü sayısı: {export_count}")
    log.info("Çalışma modu: Drive export + task polling")

    limited_collection = (
        collection
        .sort("export_date")
        .limit(export_count)
    )

    collection_list = limited_collection.toList(export_count)
    export_metadata = []

    for i in range(export_count):
        image = ee.Image(collection_list.get(i))
        date_text = date_list[i]

        lst_image = image.select("ST_B10")
        qa_image = image.select("QA_PIXEL")

        lst_description = f"export_{file_prefix}_{date_text}"
        lst_file_name = f"{file_prefix}_{date_text}"

        qa_description = f"export_{file_prefix}_{date_text}_qa"
        qa_file_name = f"{file_prefix}_{date_text}_qa"

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

        lst_meta = {
            "band": "ST_B10",
            "description": lst_description,
            "file_name_prefix": lst_file_name,
            "task_id": lst_status.get("id"),
            "task_state": lst_status.get("state"),
        }
        qa_meta = {
            "band": "QA_PIXEL",
            "description": qa_description,
            "file_name_prefix": qa_file_name,
            "task_id": qa_status.get("id"),
            "task_state": qa_status.get("state"),
        }

        register_drive_task(
            task=lst_task,
            metadata_ref=lst_meta,
            description=lst_description,
            file_name_prefix=lst_file_name,
            output_group="baseline_lst",
        )
        register_drive_task(
            task=qa_task,
            metadata_ref=qa_meta,
            description=qa_description,
            file_name_prefix=qa_file_name,
            output_group="baseline_qa",
        )

        export_metadata.append({
            "date": date_text,
            "mode": "drive",
            "folder": folder,
            "scale": scale,
            "crs": crs,
            "lst": lst_meta,
            "qa": qa_meta,
        })

    return export_metadata


# =============================================================================
# 2b. NDVI BASELINE TIMESERIES EXPORT (tek bant NDVI median)
# =============================================================================
def export_ndvi_timeseries_to_drive(
    collection: ee.ImageCollection,
    date_list: list[str],
    region: ee.Geometry,
    folder: str,
    file_prefix: str,
    scale: int = LANDSAT_NDVI_EXPORT["scale"],
    crs: str = EXPORT_CRS,
    max_exports: int | None = 20,
) -> list[dict]:
    """
    NDVI pencere-simetrik baseline collection'ındaki her görüntü için NDVI bandını
    Google Drive'a export eder. LST timeseries export ile aynı limit/sort mantığını
    kullanır; QA bandı ayrı export edilmez çünkü NDVI zaten QA-maskeli üretilir.
    """
    if not date_list:
        log.warning("NDVI date_list boş geldi. NDVI export başlatılmadı.")
        return []

    export_count = len(date_list) if max_exports is None else min(len(date_list), max_exports)
    if export_count == 0:
        log.warning("NDVI export_count = 0. NDVI export başlatılmadı.")
        return []

    log.info(f"Export edilecek baseline NDVI görüntü sayısı: {export_count}")

    limited_collection = (
        collection
        .sort("export_date")
        .limit(export_count)
    )
    collection_list = limited_collection.toList(export_count)
    export_metadata = []

    for i in range(export_count):
        image = ee.Image(collection_list.get(i))
        date_text = date_list[i]

        ndvi_image = image.select("NDVI")
        ndvi_description = f"export_{file_prefix}_{date_text}"
        ndvi_file_name = f"{file_prefix}_{date_text}"

        log.info(f"[{i + 1}/{export_count}] NDVI Drive export: {ndvi_file_name}")

        ndvi_task = ee.batch.Export.image.toDrive(
            image=ndvi_image,
            description=ndvi_description,
            folder=folder,
            fileNamePrefix=ndvi_file_name,
            region=region,
            scale=scale,
            crs=crs,
            maxPixels=1e13,
            fileFormat="GeoTIFF",
        )
        ndvi_task.start()
        ndvi_status = ndvi_task.status()

        ndvi_meta = {
            "band": "NDVI",
            "description": ndvi_description,
            "file_name_prefix": ndvi_file_name,
            "task_id": ndvi_status.get("id"),
            "task_state": ndvi_status.get("state"),
        }

        register_drive_task(
            task=ndvi_task,
            metadata_ref=ndvi_meta,
            description=ndvi_description,
            file_name_prefix=ndvi_file_name,
            output_group="baseline_ndvi",
        )

        export_metadata.append({
            "date": date_text,
            "mode": "drive",
            "folder": folder,
            "scale": scale,
            "crs": crs,
            "ndvi": ndvi_meta,
        })

    return export_metadata


# =============================================================================
# 3. DRIVE TASK POLLING VE GEEMAP İNDİRME
# =============================================================================
def poll_drive_export_tasks(
    task_records: list[dict],
    poll_interval_seconds: int = DRIVE_TASK_POLL_INTERVAL_SECONDS,
    timeout_seconds: int = DRIVE_TASK_TIMEOUT_SECONDS,
) -> dict:
    """
    Başlatılmış Earth Engine Drive export task'larını tamamlanana kadar izler.

    Task durumları metadata referanslarına da yazılır. Herhangi bir task FAILED
    veya CANCELLED olursa hata yükseltilir; böylece Step5'e eksik veriyle
    geçilmesi engellenir.
    """
    if not task_records:
        log.info("Polling için kayıtlı Drive export task'ı yok.")
        return {
            "enabled": True,
            "task_count": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "statuses": [],
        }

    started_at = time.monotonic()
    terminal_states = {"COMPLETED", "FAILED", "CANCELLED"}
    statuses_by_id = {}

    log.info("Drive export task polling başlıyor: %s task", len(task_records))

    while True:
        completed_count = 0
        failed_records = []
        cancelled_records = []

        for record in task_records:
            status = record["task"].status()
            state = status.get("state", "UNKNOWN")
            metadata_ref = record["metadata"]

            metadata_ref["task_state"] = state
            metadata_ref["last_checked_at"] = datetime.now().isoformat()
            metadata_ref["error_message"] = status.get("error_message")

            if state in terminal_states:
                metadata_ref["finished_at"] = metadata_ref.get(
                    "finished_at",
                    datetime.now().isoformat(),
                )
                completed_count += 1

            if state == "FAILED":
                failed_records.append(record)
            elif state == "CANCELLED":
                cancelled_records.append(record)

            statuses_by_id[status.get("id") or record["description"]] = {
                "description": record["description"],
                "file_name_prefix": record["file_name_prefix"],
                "output_group": record["output_group"],
                "state": state,
                "error_message": status.get("error_message"),
            }

        log.info(
            "Drive export task durumu: %s/%s tamamlandı",
            completed_count,
            len(task_records),
        )

        if failed_records or cancelled_records:
            details = [
                f"{record['description']}={record['metadata'].get('task_state')}"
                for record in failed_records + cancelled_records
            ]
            raise RuntimeError(
                "Bazı Drive export task'ları başarısız oldu: " + ", ".join(details)
            )

        if completed_count == len(task_records):
            break

        elapsed = time.monotonic() - started_at
        if elapsed >= timeout_seconds:
            raise TimeoutError(
                f"Drive export task polling zaman aşımına uğradı: {timeout_seconds} sn"
            )

        time.sleep(poll_interval_seconds)

    statuses = list(statuses_by_id.values())
    return {
        "enabled": True,
        "task_count": len(task_records),
        "completed": sum(1 for item in statuses if item["state"] == "COMPLETED"),
        "failed": sum(1 for item in statuses if item["state"] == "FAILED"),
        "cancelled": sum(1 for item in statuses if item["state"] == "CANCELLED"),
        "statuses": statuses,
        "completed_at": datetime.now().isoformat(),
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
def main(step3_result: dict | None = None) -> None:
    log.info("=" * 60)
    log.info("STEP 4 BAŞLIYOR")
    log.info("=" * 60)

    DRIVE_EXPORT_TASKS.clear()
    log_drive_download_configuration()
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
            end=END_DATE,
            month_start=SUMMER_MONTH_START,
            month_end=SUMMER_MONTH_END,
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
    ndvi_timeseries_exports = []
    ndvi_processing_metadata = None
    current_ndvi_export = None
    current_ndvi_metadata = None
    drive_task_polling_metadata = {
        "enabled": DRIVE_TASK_POLLING_ENABLED,
        "task_count": 0,
    }
    drive_download_metadata = {
        "enabled": DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT,
    }
    used_external_step3_result = step3_result is not None

    if ENABLE_LANDSAT_EXPORT:
        if step3_result is not None:
            log.info("\nStep3 çıktıları kullanılıyor; Landsat collection tekrar hesaplanmayacak.")
            landsat_daily_collection = step3_result["landsat_timeseries"]
            landsat_processing_metadata = step3_result["landsat_metadata"]
            current_median_image = step3_result["current_median"]
            current_period_metadata = step3_result["current_metadata"]
        else:
            log.info(
                "\nStep3 çıktısı verilmedi; Step4 standalone aynı "
                "pencere-simetrik Step3 helper'ını kullanacak."
            )
            step3_result = prepare_landsat_anomaly_inputs(
                region=region,
                region_name=REGION_NAME,
                current_end_date=CURRENT_PERIOD_END_DATE,
                window_days=CURRENT_PERIOD_DAYS,
                baseline_start=BASELINE_START_DATE,
                baseline_end=BASELINE_END_DATE,
            )
            landsat_daily_collection = step3_result["landsat_timeseries"]
            landsat_processing_metadata = step3_result["landsat_metadata"]
            current_median_image = step3_result["current_median"]
            current_period_metadata = step3_result["current_metadata"]

        landsat_timeseries_exports = export_landsat_timeseries_lst_and_qa_to_drive(
            collection=landsat_daily_collection,
            date_list=landsat_processing_metadata["daily_dates"],
            region=region,
            folder=EXPORT_FOLDER,
            file_prefix=LANDSAT_EXPORT["file_name_prefix"],
            scale=LANDSAT_EXPORT["scale"],
            max_exports=MAX_LANDSAT_DAILY_EXPORTS
        )

        # Current period median'ı export et
        current_period_export = export_image_to_drive(
            image=current_median_image,
            region=region,
            description=f"export_current_period_median_{CURRENT_PERIOD_DAYS}days",
            folder=EXPORT_FOLDER,
            file_name_prefix=f"landsat_current_period_{CURRENT_PERIOD_DAYS}days",
            scale=LANDSAT_EXPORT["scale"]
        )

        # NDVI ürünleri (yeni bilimsel yön): baseline timeseries + current median.
        if ENABLE_NDVI_EXPORT and "ndvi_timeseries" in step3_result:
            log.info("\nNDVI baseline timeseries ve current median export ediliyor.")
            ndvi_collection = step3_result["ndvi_timeseries"]
            ndvi_processing_metadata = step3_result["ndvi_metadata"]
            current_ndvi_image = step3_result["current_ndvi"]
            current_ndvi_metadata = step3_result["current_ndvi_metadata"]

            ndvi_timeseries_exports = export_ndvi_timeseries_to_drive(
                collection=ndvi_collection,
                date_list=ndvi_processing_metadata["daily_dates"],
                region=region,
                folder=EXPORT_FOLDER,
                file_prefix=LANDSAT_NDVI_EXPORT["file_name_prefix"],
                scale=LANDSAT_NDVI_EXPORT["scale"],
                max_exports=MAX_LANDSAT_DAILY_EXPORTS,
            )

            current_ndvi_export = export_image_to_drive(
                image=current_ndvi_image,
                region=region,
                description=f"export_current_ndvi_median_{CURRENT_PERIOD_DAYS}days",
                folder=EXPORT_FOLDER,
                file_name_prefix="current_ndvi_median",
                scale=LANDSAT_NDVI_EXPORT["scale"],
            )
        elif ENABLE_NDVI_EXPORT:
            log.warning(
                "ENABLE_NDVI_EXPORT açık ama step3_result NDVI çıktısı içermiyor; "
                "NDVI export atlandı."
            )

    else:
        log.info("Landsat export devre dışı bırakıldı.")

    if DRIVE_TASK_POLLING_ENABLED and DRIVE_EXPORT_TASKS:
        drive_task_polling_metadata = poll_drive_export_tasks(DRIVE_EXPORT_TASKS)
        drive_download_metadata = {
            "enabled": DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT,
            "attempted": False,
            "downloaded": False,
            "deferred_to": "step4b_download_drive_export",
            "message": (
                "Drive indirme ve yerel klasörlere dağıtma Step4b aşamasına "
                "ayrıldı."
            ),
        }
    elif DRIVE_EXPORT_TASKS:
        log.info("Drive task polling kapalı; export görevleri başlatıldıktan sonra çıkılıyor.")
        drive_task_polling_metadata = {
            "enabled": False,
            "task_count": len(DRIVE_EXPORT_TASKS),
            "reason": "DRIVE_TASK_POLLING_ENABLED=False",
        }
    else:
        log.info("Polling gerektiren Drive export task'ı yok.")

    drive_files_downloaded = bool(drive_download_metadata.get("downloaded"))
    download_required_before_step5 = bool(DRIVE_EXPORT_TASKS)
    drive_download_message = drive_download_metadata.get("message")
        
    metadata = {
        "step": "step4_export_geotiff",
        "pipeline_stage": "online",
        "region_name": REGION_NAME,
        "baseline_date_start": BASELINE_START_DATE,
        "baseline_date_end": BASELINE_END_DATE,
        "current_period_end": CURRENT_PERIOD_END_DATE,
        "current_period_days": CURRENT_PERIOD_DAYS,
        "export_folder": EXPORT_FOLDER,
        "created_at": datetime.now().isoformat(),
        "log_file": str(log_file),
        "download_required_before_step5": download_required_before_step5,
        "enabled_exports": {
            "modis": ENABLE_MODIS_EXPORT,
            "landsat": ENABLE_LANDSAT_EXPORT,
            "ndvi": ENABLE_NDVI_EXPORT
        },
        "processing": {
            "modis": modis_processing_metadata,
            "landsat_baseline_timeseries": landsat_processing_metadata,
            "landsat_current_period": current_period_metadata,
            "ndvi_baseline_timeseries": ndvi_processing_metadata,
            "ndvi_current_period": current_ndvi_metadata
        },
        "exports": {
            "modis": modis_export_metadata,
            "landsat_baseline_timeseries": landsat_timeseries_exports,
            "landsat_current_period": current_period_export,
            "ndvi_baseline_timeseries": ndvi_timeseries_exports,
            "ndvi_current_period": current_ndvi_export
        },
        "drive_task_polling": drive_task_polling_metadata,
        "drive_download": drive_download_metadata,
        "notes": {
            "modis_download_mode": "drive_export_then_step4b_download",
            "landsat_download_mode": "drive_export_with_task_polling_then_step4b_download",
            "anomaly_method": "window_symmetric_landsat_z_score",
            "landsat_generation_source": (
                "step3_result" if used_external_step3_result else "step4_standalone"
            ),
        },
        "status": (
            "export_tasks_completed_download_pending"
            if drive_task_polling_metadata.get("completed") == len(DRIVE_EXPORT_TASKS)
            else "export_tasks_started"
        )
    }
    metadata_path = save_metadata(metadata)

    log.info("=" * 60)
    log.info("STEP 4 TAMAMLANDI")
    log.info(f"Export metadata dosyası: {metadata_path}")

    if DRIVE_TASK_POLLING_ENABLED:
        log.info("Google Drive export görevleri tamamlandı.")
    else:
        log.info("Google Drive export görevleri başlatıldı.")
    if drive_download_message:
        log.info("Drive indirme durumu: %s", drive_download_message)
    log.info("Sonraki adım: Step4b Drive çıktıları indirip Step5 klasörlerine yerleştirir.")
    log.info("\nExport edilen dosyalar:")
    log.info("  1. Baseline zaman serisi (ST_B10 + QA_PIXEL)")
    log.info("  2. Current period median (LST Celsius + valid count)")

    log.info("=" * 60)

    print("\nSTEP 4 tamamlandı. Export görevleri Drive'da tamamlandı veya başlatıldı.")
    if drive_download_message:
        print(f"Drive indirme notu: {drive_download_message}")
    print("Sonraki adım:")
    print("python step4b_download_drive_export.py\n")

if __name__ == "__main__":
    main()