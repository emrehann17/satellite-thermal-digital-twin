"""
step4b_download_drive_exports.py

Step4 ile Step5 arasındaki offline geçiş katmanı.

Yapılanlar:
    - Step4'te tamamlanan Google Drive export dosyalarını indirir.
    - İndirilen GeoTIFF dosyalarını Step5'in beklediği data klasörlerine dağıtır.
    - Download/placement metadata bilgisini ayrı kaydeder.
"""

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from core.config import (
    DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT,
    DRIVE_DOWNLOAD_OVERWRITE,
    DRIVE_DOWNLOAD_STAGING_SUBDIR,
    GOOGLE_DRIVE_EXPORT_FOLDER_ID,
    GOOGLE_DRIVE_EXPORT_FOLDER_URL,
    LANDSAT_EXPORT,
    LANDSAT_NDVI_EXPORT,
    MODIS_EXPORT,
)
from core.io_utils import setup_logger


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs" / "step4b"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

log, log_file = setup_logger("step4b")


def extract_google_drive_folder_id(url_or_id: str | None) -> str | None:
    """Google Drive klasör URL'sinden veya doğrudan ID değerinden klasör ID'sini çıkarır."""
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
    """Config/env değerlerinden Drive klasör URL ve ID referansını çözer."""
    folder_id = extract_google_drive_folder_id(GOOGLE_DRIVE_EXPORT_FOLDER_ID)
    if folder_id is None:
        folder_id = extract_google_drive_folder_id(GOOGLE_DRIVE_EXPORT_FOLDER_URL)

    folder_url = None
    if folder_id is not None:
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    elif GOOGLE_DRIVE_EXPORT_FOLDER_URL:
        folder_url = str(GOOGLE_DRIVE_EXPORT_FOLDER_URL).strip()

    return folder_url, folder_id


def ensure_step5_data_dirs() -> tuple[Path, Path, Path, Path, Path, Path]:
    """İndirilen rasterlar için Step5'in beklediği yerel veri klasörlerini hazırlar."""
    lst_dir = BASE_DIR / "data" / "landsat_timeseries"
    qa_dir = BASE_DIR / "data" / "landsat_qa"
    current_dir = BASE_DIR / "data" / "current_period"
    modis_dir = BASE_DIR / "data" / "modis"
    ndvi_baseline_dir = BASE_DIR / "data" / "ndvi_timeseries"
    ndvi_current_dir = BASE_DIR / "data" / "ndvi_current_period"

    lst_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    current_dir.mkdir(parents=True, exist_ok=True)
    modis_dir.mkdir(parents=True, exist_ok=True)
    ndvi_baseline_dir.mkdir(parents=True, exist_ok=True)
    ndvi_current_dir.mkdir(parents=True, exist_ok=True)

    return lst_dir, qa_dir, current_dir, modis_dir, ndvi_baseline_dir, ndvi_current_dir


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


def classify_downloaded_tif(source_path: Path) -> tuple[str, Path] | None:
    """İndirilen GeoTIFF'i Step5 data klasörlerinden doğru hedefe sınıflandırır."""
    (
        lst_dir,
        qa_dir,
        current_dir,
        modis_dir,
        ndvi_baseline_dir,
        ndvi_current_dir,
    ) = ensure_step5_data_dirs()
    name = source_path.name
    lower_name = name.lower()

    if is_landsat_qa_export_name(name):
        return "baseline_qa", qa_dir / name

    if lower_name.startswith("current_ndvi_median"):
        return "ndvi_current_period", ndvi_current_dir / name

    if lower_name.startswith(LANDSAT_NDVI_EXPORT["file_name_prefix"].lower()):
        return "ndvi_baseline", ndvi_baseline_dir / name

    if lower_name.startswith("landsat_current_period_"):
        return "current_period", current_dir / name

    if lower_name.startswith(MODIS_EXPORT["file_name_prefix"].lower()):
        return "modis", modis_dir / name

    if lower_name.startswith(LANDSAT_EXPORT["file_name_prefix"].lower()):
        return "baseline_lst", lst_dir / name

    return None


def place_downloaded_drive_tifs(staging_dir: Path) -> dict:
    """
    Drive staging klasöründeki GeoTIFF dosyalarını Step5 data klasörlerine dağıtır.

    `drive_exports` klasörü yalnızca staging alanıdır; Step5 doğrudan burayı okumaz.
    Bu fonksiyon dosyaları `data/landsat_timeseries`, `data/landsat_qa`,
    `data/current_period` ve `data/modis` klasörlerine kopyalar.
    """
    copied = {
        "baseline_lst": [],
        "baseline_qa": [],
        "current_period": [],
        "modis": [],
        "ndvi_baseline": [],
        "ndvi_current_period": [],
        "unmatched": [],
    }

    for source_path in sorted(staging_dir.rglob("*.tif")):
        classified = classify_downloaded_tif(source_path)
        if classified is None:
            copied["unmatched"].append(str(source_path))
            continue

        group, target_path = classified
        copy_with_overwrite_control(source_path, target_path)
        copied[group].append(str(target_path))
        log.info("Drive çıktısı Step5 klasörüne kopyalandı: %s", target_path)

    return copied


def download_drive_exports_with_geemap() -> dict:
    """Google Drive export klasörünü indirir ve GeoTIFF dosyalarını Step5 klasörlerine dağıtır."""
    if not DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT:
        log.warning("Drive otomatik indirme kapalı: DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT=False")
        return {
            "enabled": False,
            "attempted": False,
            "downloaded": False,
            "reason": "DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT=False",
            "message": "Otomatik Drive indirme devre dışı.",
        }

    folder_url, folder_id = resolve_drive_folder_reference()

    if not folder_url and not folder_id:
        log.error("Drive otomatik indirme istendi ama GOOGLE_DRIVE_EXPORT_FOLDER_URL/ID boş.")
        return {
            "enabled": True,
            "attempted": False,
            "downloaded": False,
            "reason": "missing_google_drive_folder_url_or_id",
            "message": (
                "Otomatik indirme açık ama Drive klasör URL/ID verilmemiş. "
                "GOOGLE_DRIVE_EXPORT_FOLDER_URL veya GOOGLE_DRIVE_EXPORT_FOLDER_ID ayarlanmalı."
            ),
        }

    try:
        import geemap
    except ImportError as exc:
        raise ImportError("Drive klasörü indirmek için geemap gerekli.") from exc

    staging_dir = BASE_DIR / "data" / DRIVE_DOWNLOAD_STAGING_SUBDIR
    staging_dir.mkdir(parents=True, exist_ok=True)

    log.info("Drive klasörü staging alanına indiriliyor: %s", staging_dir)
    log.info("Drive referansı çözüldü: folder_id=%s, folder_url=%s", folder_id, folder_url)

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
        "message": "Drive klasörü indirildi ve GeoTIFF dosyaları Step5 klasörlerine kopyalandı.",
        "resolved_folder_id": folder_id,
        "resolved_folder_url": folder_url,
    }


def save_metadata(metadata: dict, filename: str = "step4b_metadata.json") -> Path:
    """Step4b download/placement metadata bilgisini JSON olarak kaydeder."""
    output_path = OUTPUTS_DIR / filename
    output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Step4b metadata kaydedildi: %s", output_path)
    return output_path


def main() -> dict:
    """Drive export klasörünü indirir ve GeoTIFF dosyalarını Step5 klasörlerine dağıtır."""
    log.info("=" * 60)
    log.info("STEP 4B BAŞLIYOR (Drive download + local placement)")
    log.info("=" * 60)

    download_metadata = download_drive_exports_with_geemap()

    metadata = {
        "step": "step4b_download_drive_exports",
        "pipeline_stage": "drive_download_and_local_placement",
        "created_at": datetime.now().isoformat(),
        "log_file": str(log_file),
        "drive_download": download_metadata,
        "status": (
            "download_completed"
            if download_metadata.get("downloaded")
            else "download_not_completed"
        ),
    }
    metadata_path = save_metadata(metadata)

    if download_metadata.get("downloaded"):
        log.info("Drive çıktıları indirildi ve Step5 data klasörlerine yerleştirildi.")
        print("\nSTEP 4B tamamlandı. Step5 çalıştırılabilir:")
        print("python step5_preprocess_timeseries.py\n")
    else:
        message = download_metadata.get("message") or download_metadata.get("reason")
        log.warning("Drive indirme tamamlanmadı: %s", message)
        print("\nSTEP 4B tamamlandı ancak indirme yapılmadı.")
        if message:
            print(f"Not: {message}")
        print(f"Metadata: {metadata_path}\n")

    log.info("=" * 60)
    log.info("STEP 4B TAMAMLANDI")
    return metadata


if __name__ == "__main__":
    main()