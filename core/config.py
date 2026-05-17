import os


# =============================================================================
# Google Earth Engine
# =============================================================================

GEE_PROJECT = "b7-thermal-digital-twin"

MODIS_COLLECTION = "MODIS/061/MOD11A1"
LANDSAT_COLLECTION = "LANDSAT/LC08/C02/T1_L2"

START_DATE = "2019-01-01"
END_DATE = "2023-12-31"

REGION_NAME = "kozan_aoi"  # Test için küçük bölge kullan.
EXPORT_CRS = "EPSG:4326"


# =============================================================================
# Google Drive Export / Download
# =============================================================================

EXPORT_FOLDER = "B7_Thermal_Digital_Twin"

# Earth Engine Drive export task'larını tamamlanana kadar izle.
DRIVE_TASK_POLLING_ENABLED = True
DRIVE_TASK_POLL_INTERVAL_SECONDS = 60
DRIVE_TASK_TIMEOUT_SECONDS = 6 * 60 * 60

# Export görevleri tamamlandıktan sonra Drive klasörünü geemap/gdown ile indir.
# Ortam değişkeni verilirse onu kullanır; yoksa test klasörü varsayılanını kullanır.
DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT = True
GOOGLE_DRIVE_EXPORT_FOLDER_URL = os.getenv(
    "GOOGLE_DRIVE_EXPORT_FOLDER_URL",
    "https://drive.google.com/drive/u/0/folders/1eyqH0MpYH46-F5Ao3VQqv834T9mfyYfn",
)
GOOGLE_DRIVE_EXPORT_FOLDER_ID = os.getenv(
    "GOOGLE_DRIVE_EXPORT_FOLDER_ID",
    "1eyqH0MpYH46-F5Ao3VQqv834T9mfyYfn",
)
DRIVE_DOWNLOAD_STAGING_SUBDIR = "drive_exports"
DRIVE_DOWNLOAD_OVERWRITE = True


# =============================================================================
# Export Toggles
# =============================================================================

ENABLE_MODIS_EXPORT = False
ENABLE_LANDSAT_EXPORT = True


# =============================================================================
# MODIS Settings
# =============================================================================

MODIS_EXPORT = {
    "description": "export_modis_lst_5y_summer_mean",
    "file_name_prefix": "modis_lst_dogu_akdeniz_5y_summer_mean",
    "scale": 1000,
}

MODIS_EXPORT_FOLDER = "B7_Thermal_Digital_Twin_MODIS"
MODIS_FILE_PREFIX = MODIS_EXPORT["file_name_prefix"]

SUMMER_MONTH_START = 6
SUMMER_MONTH_END = 9


# =============================================================================
# Landsat Settings
# =============================================================================

LANDSAT_EXPORT = {
    "file_name_prefix": "landsat_lst_dogu_akdeniz",
    "scale": 30,
}

LANDSAT_LST_EXPORT_FOLDER = "B7_Thermal_Digital_Twin_Landsat_Timeseries"
LANDSAT_QA_EXPORT_FOLDER = "B7_Thermal_Digital_Twin_Landsat_QA"

LANDSAT_SCALE = 0.00341802
LANDSAT_OFFSET = 149.0

# Test için sınırlı sayıda günlük composite export edilir.
MAX_LANDSAT_DAILY_EXPORTS = 12


# =============================================================================
# Anomaly / Temporal Window Settings
# =============================================================================

# Baseline dönemi. Test için tek yaz sezonu kullanılıyor.
BASELINE_START_DATE = "2022-06-01"
BASELINE_END_DATE = "2022-09-30"

# Current period anomali hesabında kullanılacak güncel pencere.
CURRENT_PERIOD_DAYS = 30
CURRENT_PERIOD_END_DATE = "2023-08-31"


# =============================================================================
# Step5 Windowed Raster Processing
# =============================================================================

# Her seferinde okunacak raster pencere kenarı (piksel).
# Yaklaşık bellek: sahne_sayısı * STEP5_WINDOW_SIZE^2 * 4 byte.
STEP5_WINDOW_SIZE = 512

# Standart sapma bu eşiğin altındaysa z-score anomali NaN yazılır.
# Bu, sabit piksellerde sonsuz/yanıltıcı z-score üretimini engeller.
STEP5_STD_EPSILON = 1e-6

# Bu eşiğin altında geçerli baseline gözlemi olan piksellerde mean/std/z-score
# hesaplanmaz; piksel NaN bırakılır.
STEP5_MIN_BASELINE_VALID_COUNT = 3

# Windowed akış ana raster çıktıları üretir. Interpolated full NetCDF çıktısı
# büyük veri için tekrar yüksek bellek/disk baskısı yaratabileceği için kapalıdır.
STEP5_WRITE_INTERPOLATED_NETCDF = False