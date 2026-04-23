import ee 
import urllib.request
import zipfile
import logging
import json
from datetime import datetime
from pathlib import Path

# ── Klasör yapısı ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data" / "step2"
LOGS_DIR   = BASE_DIR / "logs"

DATA_DIR.mkdir(parents=True, exist_ok=True) #if these folders don't exist, create them
LOGS_DIR.mkdir(parents=True, exist_ok=True) #else do nothing

# ── Loglama ────────────────────────────────────────────────────────────────────
log_file = LOGS_DIR / f"step2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()          # terminale de yaz
    ]
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 1. GEE BAŞLATMA
# ══════════════════════════════════════════════════════════════════════════════
def init_gee(project: str = "b7-thermal-digital-twin") -> None:
    """GEE baslatir ve dosyalari dogrular."""
    log.info("GEE initializing...")
    ee.Initialize(project=project)

    # Basit doğrulama: sabit bir sayıyı GEE üzerinden hesapla
    test_val = ee.Number(42).getInfo()
    assert test_val == 42, "GEE connection failed: expected 42, got {test_val}"
    log.info(f"GEE connection OK  →  project: {project}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. ÇALIŞMA BÖLGELERİ
# ══════════════════════════════════════════════════════════════════════════════
def build_regions() -> dict:
    kozan_merkez = ee.Geometry.Point([35.82, 37.45])
    kozan_aoi = kozan_merkez.buffer(50000) # 50 km tampon

    # Batı: Mersin batısı 33.8  Doğu: Hatay 36.7 Güney: sahil 36.0 Kuzey: Toros dağları 38.0
    dogu_akdeniz = ee.Geometry.BBox(33.8, 36.0, 36.7, 38.0)

    regions = {
        "dogu_akdeniz": dogu_akdeniz,
        "kozan_aoi": kozan_aoi
    }    

    for name, geom in regions.items():
        info = geom.getInfo()
        log.info(f"Region '{name}' built: {geom.getInfo()}")
        
    return regions

# ══════════════════════════════════════════════════════════════════════════════
# 3. 5 YILLIK YAZ ORTALAMASI (REDUCTION)
# ══════════════════════════════════════════════════════════════════════════════
def process_summer_mean(region: ee.Geometry, region_name: str) -> ee.Image:
    '''5 yillik (2021-2025) yaz aylarinin (Haziran-Eylül) sicaklik ortalamasini hesaplar.'''
    log.info(f"For past 5 years processing summer mean for region: {region_name}")

    collection = (
        ee.ImageCollection("ECMWF/ERA5/DAILY")
        .filterBounds(region)
        .filterDate("2019-01-01", "2023-12-31")  # 2021-2025 denedim, veri güncel olsun diye ama hata aldım sanırım 2023 sonuna kadar var, 2024'ü denemedim 2025 verisi yok henüz.
        .filter(ee.Filter.calendarRange(6, 9, "month")) # yaz aylariş
    )

    count = collection.size().getInfo()
    log.info(f"Found {count} images for region '{region_name}' in summer months (2021-2025).")

    if count == 0:
        raise ValueError(f"No images found for region '{region_name}' in the specified date range and months.")
    
    #mean of images
    summer_mean = (
        collection.select("LST_Day_1KM")
        .mean()
        .multiply(0.02)
        .clip(region) # to ROI part 
        .subtract(273.15) # Celsius'a çevir
    )

    return summer_mean

def export_image_to_drive(image: ee.Image, region: ee.Geometry, task_name: str) -> None:
    log.info(f"Exporting image to Google Drive with task name: {task_name}")

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=task_name,
        folder="B7_Thermal_Digital_Twin",
        fileNamePrefix=task_name,
        region=region,
        scale=1000, #1km çözünürlükte
        crs = "EPSG:4326",
        maxPixels=1e10
    )

    task.start()
    log.info(f"Export task '{task_name}' started. Check Google Drive for progress.")


def main():
    log.info("=" * 60)
    log.info("STEP 2-4 INITIALIZING...")
    log.info("=" * 60)
    init_gee()

    regions = build_regions()

    mean_image = process_summer_mean(
        regions["dogu_akdeniz"], "dogu_akdeniz")
    
    export_image_to_drive(
        image = mean_image,
        region = regions["dogu_akdeniz"],
        task_name = "dogu_akdeniz_summer_mean_2021_2025"
    )
    log.info("="*80)
    log.info("Step 2 completed successfully. Check logs for details.")

if __name__ == "__main__":
    main()