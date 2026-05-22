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
import re
import shutil
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
    ENABLE_MODIS_EXPORT,
    ENABLE_LANDSAT_EXPORT,
    DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT,
    DRIVE_DOWNLOAD_OVERWRITE,
    DRIVE_DOWNLOAD_STAGING_SUBDIR,
    DRIVE_TASK_POLLING_ENABLED,
    DRIVE_TASK_POLL_INTERVAL_SECONDS,
    DRIVE_TASK_TIMEOUT_SECONDS,
    GOOGLE_DRIVE_EXPORT_FOLDER_ID,
    GOOGLE_DRIVE_EXPORT_FOLDER_URL,
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
from step3_landsat_lst import get_landsat_daily_median_collection, get_current_period_median


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
        "Drive download config: polling=%s, auto_download=%s, folder_url=%s, folder_id=%s",
        DRIVE_TASK_POLLING_ENABLED,
        DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT,
        bool(GOOGLE_DRIVE_EXPORT_FOLDER_URL),
        bool(GOOGLE_DRIVE_EXPORT_FOLDER_ID),
    )

    if legacy_download_mode is not None:
        log.warning(
            "Legacy DOWNLOAD_MODE=%s bulundu. Step4 artik bu alanla davranis secmiyor; "
            "otomatik indirme icin DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT=True ve "
            "GOOGLE_DRIVE_EXPORT_FOLDER_URL veya GOOGLE_DRIVE_EXPORT_FOLDER_ID kullanilmali.",
            legacy_download_mode,
        )


def extract_google_drive_folder_id(url_or_id: str | None) -> str | None:
    if not url_or_id:
        return None

    value = str(url_or_id).strip()
    if not value:
        return None

    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", value)
    if match:
        return match.group(1)

    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", value):
        return value

    return None


def resolve_drive_folder_reference() -> tuple[str | None, str | None]:
    folder_id = extract_google_drive_folder_id(GOOGLE_DRIVE_EXPORT_FOLDER_ID)
    if folder_id is None:
        folder_id = extract_google_drive_folder_id(GOOGLE_DRIVE_EXPORT_FOLDER_URL)

    folder_url = None
    if folder_id is not None:
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    elif GOOGLE_DRIVE_EXPORT_FOLDER_URL:
        folder_url = str(GOOGLE_DRIVE_EXPORT_FOLDER_URL).strip()

    return folder_url, folder_id

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


def ensure_step5_data_dirs() -> tuple[Path, Path, Path, Path]:
    """İndirilen rasterlar için yerel veri klasörlerini hazırlar."""
    lst_dir = BASE_DIR / "data" / "landsat_timeseries"
    qa_dir = BASE_DIR / "data" / "landsat_qa"
    current_dir = BASE_DIR / "data" / "current_period"
    modis_dir = BASE_DIR / "data" / "modis"

    lst_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    current_dir.mkdir(parents=True, exist_ok=True)
    modis_dir.mkdir(parents=True, exist_ok=True)

    return lst_dir, qa_dir, current_dir, modis_dir


def copy_with_overwrite_control(source_path: Path, target_path: Path) -> None:
    """Dosyayı hedefe kopyalar; overwrite ayarı kapalıysa mevcut dosyayı korur."""
    if target_path.exists() and not DRIVE_DOWNLOAD_OVERWRITE:
        log.info("Dosya zaten var, atlandı: %s", target_path)
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def is_landsat_qa_export_name(filename: str) -> bool:
    """
    Landsat QA export adını tek parça ve GEE parçalı dosya adlarında tanır.

    Örnekler:
        landsat_lst_dogu_akdeniz_2020-06-01_qa.tif
        landsat_lst_dogu_akdeniz_2020-06-01_qa-0000000000-0000000000.tif
    """
    stem = Path(filename).stem.lower()
    return bool(re.search(r"_qa($|[-_])", stem))


def place_downloaded_drive_tifs(staging_dir: Path) -> dict:
    """
    geemap ile indirilen Drive GeoTIFF dosyalarını yerel veri klasörlerine dağıtır.

    Dosya adı eşleşmeleri:
        *_qa.tif                         -> data/landsat_qa
        landsat_current_period_*.tif     -> data/current_period
        landsat_lst_*.tif                -> data/landsat_timeseries
        modis_*.tif                      -> data/modis
    """
    lst_dir, qa_dir, current_dir, modis_dir = ensure_step5_data_dirs()
    copied = {
        "baseline_lst": [],
        "baseline_qa": [],
        "current_period": [],
        "modis": [],
        "unmatched": [],
    }

    for source_path in sorted(staging_dir.rglob("*.tif")):
        name = source_path.name
        lower_name = name.lower()

        if is_landsat_qa_export_name(name):
            target_path = qa_dir / name
            copied["baseline_qa"].append(str(target_path))
        elif lower_name.startswith("landsat_current_period_"):
            target_path = current_dir / name
            copied["current_period"].append(str(target_path))
        elif lower_name.startswith(MODIS_EXPORT["file_name_prefix"].lower()):
            target_path = modis_dir / name
            copied["modis"].append(str(target_path))
        elif lower_name.startswith(LANDSAT_EXPORT["file_name_prefix"].lower()):
            target_path = lst_dir / name
            copied["baseline_lst"].append(str(target_path))
        else:
            copied["unmatched"].append(str(source_path))
            continue

        copy_with_overwrite_control(source_path, target_path)
        log.info("Drive çıktısı Step5 klasörüne kopyalandı: %s", target_path)

    return copied


def download_drive_exports_with_geemap() -> dict:
    """
    Google Drive export klasörünü geemap.download_folder ile yerel staging alanına indirir.

    Not: geemap/gdown tarafı Drive klasör URL'si veya klasör ID'si ister. Export
    folder adı tek başına indirme için yeterli değildir.
    """
    if not DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT:
        log.warning("Drive otomatik indirme kapali: DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT=False")
        return {
            "enabled": False,
            "attempted": False,
            "downloaded": False,
            "reason": "DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT=False",
            "message": (
                "Otomatik Drive indirme devre disi. "
                "DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT=True yapilmadi."
            ),
        }

    folder_url, folder_id = resolve_drive_folder_reference()

    if not folder_url and not folder_id:
        log.error(
            "Drive otomatik indirme istendi ama GOOGLE_DRIVE_EXPORT_FOLDER_URL/ID bos."
        )
        return {
            "enabled": True,
            "attempted": False,
            "downloaded": False,
            "reason": "missing_google_drive_folder_url_or_id",
            "message": (
                "Otomatik indirme acik ama Drive klasor URL/ID verilmemis. "
                "GOOGLE_DRIVE_EXPORT_FOLDER_URL veya GOOGLE_DRIVE_EXPORT_FOLDER_ID ayarlanmali."
            ),
        }

    try:
        import geemap
    except ImportError as exc:
        raise ImportError(
            "Drive klasörü indirmek için geemap gerekli. "
            "Kurulum: pip install geemap"
        ) from exc

    staging_dir = BASE_DIR / "data" / DRIVE_DOWNLOAD_STAGING_SUBDIR
    staging_dir.mkdir(parents=True, exist_ok=True)

    log.info("Drive klasoru indiriliyor: %s", staging_dir)
    log.info("Drive referansi cozuldu: folder_id=%s, folder_url=%s", folder_id, folder_url)

    download_kwargs = {
        "output": str(staging_dir),
        "remaining_ok": True,
        "quiet": False,
    }
    if folder_id:
        download_kwargs["id"] = folder_id
    elif folder_url:
        download_kwargs["url"] = folder_url

    downloaded_files = geemap.download_folder(**download_kwargs)

    copied = place_downloaded_drive_tifs(staging_dir)

    return {
        "enabled": True,
        "attempted": True,
        "downloaded": True,
        "staging_dir": str(staging_dir),
        "downloaded_files": downloaded_files,
        "copied": copied,
        "downloaded_at": datetime.now().isoformat(),
        "message": "Drive klasoru indirildi ve GeoTIFF dosyalari Step5 klasorlerine kopyalandi.",
        "resolved_folder_id": folder_id,
        "resolved_folder_url": folder_url,
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
            start_date=START_DATE,
            end_date=END_DATE,
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
    drive_task_polling_metadata = {
        "enabled": DRIVE_TASK_POLLING_ENABLED,
        "task_count": 0,
    }
    drive_download_metadata = {
        "enabled": DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT,
    }

    if ENABLE_LANDSAT_EXPORT:
        if step3_result is not None:
            log.info("\nStep3 çıktıları kullanılıyor; Landsat collection tekrar hesaplanmayacak.")
            landsat_daily_collection = step3_result["landsat_timeseries"]
            landsat_processing_metadata = step3_result["landsat_metadata"]
        else:
            log.info("\nStep3 çıktısı verilmedi; baseline zaman serisi Step4 içinde hazırlanıyor.")
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
            max_exports=MAX_LANDSAT_DAILY_EXPORTS
        )
        
        if step3_result is not None:
            log.info("\nStep3 current period median çıktısı kullanılıyor.")
            current_median_image = step3_result["current_median"]
            current_period_metadata = step3_result["current_metadata"]
        else:
            log.info("\nStep3 çıktısı verilmedi; current period median Step4 içinde hazırlanıyor.")
            current_median_image, current_period_metadata = get_current_period_median(
                region=region,
                region_name=REGION_NAME,
                end_date=CURRENT_PERIOD_END_DATE,
                window_days=CURRENT_PERIOD_DAYS
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
        
    else:
        log.info("Landsat export devre dışı bırakıldı.")

    if DRIVE_TASK_POLLING_ENABLED and DRIVE_EXPORT_TASKS:
        drive_task_polling_metadata = poll_drive_export_tasks(DRIVE_EXPORT_TASKS)
        drive_download_metadata = download_drive_exports_with_geemap()
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
    manual_download_required = (
        bool(DRIVE_EXPORT_TASKS)
        and not drive_files_downloaded
    )
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
        "manual_download_required": manual_download_required,
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
        "drive_task_polling": drive_task_polling_metadata,
        "drive_download": drive_download_metadata,
        "notes": {
            "modis_download_mode": "drive_only_for_now",
            "landsat_download_mode": "drive_export_with_task_polling",
            "anomaly_method": "z_score_window_based"
        },
        "status": (
            "drive_download_completed"
            if drive_files_downloaded
            else "export_tasks_completed"
            if drive_task_polling_metadata.get("completed") == len(DRIVE_EXPORT_TASKS)
            else "export_tasks_started"
        )
    }
    metadata_path = save_metadata(metadata)

    log.info("=" * 60)
    log.info("STEP 4 TAMAMLANDI")
    log.info(f"Export metadata dosyası: {metadata_path}")

    if drive_files_downloaded:
        log.info("Drive export dosyaları indirildi ve Step5 klasörlerine yerleştirildi.")
        log.info("Step5 çalıştırılabilir.")
    else:
        if DRIVE_TASK_POLLING_ENABLED:
            log.info("Google Drive export görevleri tamamlandı.")
        else:
            log.info("Google Drive export görevleri başlatıldı.")
        if drive_download_message:
            log.warning("Otomatik indirme durumu: %s", drive_download_message)
        log.info("Dosyaları manuel indirip Step5 klasörlerine yerleştirmen gerekiyor.")
        log.info("\nExport edilen dosyalar:")
        log.info("  1. Baseline zaman serisi (ST_B10 + QA_PIXEL)")
        log.info("  2. Current period median (LST Celsius)")

    log.info("=" * 60)

    if drive_files_downloaded:
        print("\nSTEP 4 tamamlandı. Drive çıktıları indirildi ve Step5 klasörlerine yerleştirildi.")
        print("Şimdi Step5'i çalıştırabilirsin:")
        print("python step5_preprocess_timeseries.py\n")
    else:
        print("\nSTEP 4 tamamlandı. Export görevleri Drive'da tamamlandı.")
        if drive_download_message:
            print(f"Otomatik indirme notu: {drive_download_message}")
        print("Drive'dan dosyaları indirip Step5 klasörlerine yerleştirmen gerekiyor.\n")

if __name__ == "__main__":
    main()