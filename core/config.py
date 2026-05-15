GEE_PROJECT = "b7-thermal-digital-twin"

MODIS_COLLECTION = "MODIS/061/MOD11A1"
LANDSAT_COLLECTION = "LANDSAT/LC08/C02/T1_L2"

START_DATE = "2019-01-01"
END_DATE = "2023-12-31"

EXPORT_FOLDER = "B7_Thermal_Digital_Twin"

# Drive export task polling ayarları
DRIVE_TASK_POLLING_ENABLED = True
DRIVE_TASK_POLL_INTERVAL_SECONDS = 60
DRIVE_TASK_TIMEOUT_SECONDS = 6 * 60 * 60

# Export görevleri tamamlandıktan sonra Drive klasörünü geemap/gdown ile indir.
# Google Drive klasörünün paylaşılabilir URL'si veya klasör ID'si gerekir.
DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT = True
GOOGLE_DRIVE_EXPORT_FOLDER_URL = "https://drive.google.com/drive/u/0/folders/1eyqH0MpYH46-F5Ao3VQqv834T9mfyYfn"
GOOGLE_DRIVE_EXPORT_FOLDER_ID = "1eyqH0MpYH46-F5Ao3VQqv834T9mfyYfn"
DRIVE_DOWNLOAD_STAGING_SUBDIR = "drive_exports"
DRIVE_DOWNLOAD_OVERWRITE = True

MODIS_EXPORT = {
    "description": "export_modis_lst_5y_summer_mean",
    "file_name_prefix": "modis_lst_dogu_akdeniz_5y_summer_mean",
    "scale": 1000,
}

LANDSAT_EXPORT = {
    "file_name_prefix": "landsat_lst_dogu_akdeniz",
    "scale": 30,
}

ENABLE_MODIS_EXPORT = False
ENABLE_LANDSAT_EXPORT = True

LANDSAT_SCALE = 0.00341802
LANDSAT_OFFSET = 149.0

REGION_NAME = "kozan_aoi"  # Test için küçük bölge kullan

MODIS_EXPORT_FOLDER = "B7_Thermal_Digital_Twin_MODIS"
LANDSAT_LST_EXPORT_FOLDER = "B7_Thermal_Digital_Twin_Landsat_Timeseries"
LANDSAT_QA_EXPORT_FOLDER = "B7_Thermal_Digital_Twin_Landsat_QA"

MODIS_FILE_PREFIX = "modis_lst_dogu_akdeniz_5y_summer_mean"

SUMMER_MONTH_START = 6
SUMMER_MONTH_END = 9

MAX_LANDSAT_DAILY_EXPORTS = 5  

EXPORT_CRS = "EPSG:4326"

# Anomali pencere ayarları (TEST İÇİN KÜÇÜK PENCERELER)
BASELINE_START_DATE = "2022-06-01"  # Tek yaz sezonu
BASELINE_END_DATE = "2022-09-30"

# Current period - anomali hesabı için kullanılacak güncel pencere
CURRENT_PERIOD_DAYS = 30  # 30 gün test için yeterli
CURRENT_PERIOD_END_DATE = "2023-08-31"  # 2023 yaz sonu

# Step5 bellek kullanımı ayarları
# Her seferinde okunacak raster pencere kenarı (piksel).
# Yaklaşık bellek: sahne_sayısı * STEP5_WINDOW_SIZE^2 * 4 byte.
STEP5_WINDOW_SIZE = 512

# Standart sapma bu eşiğin altındaysa z-score anomali NaN yazılır.
# Bu, sabit piksellerde sonsuz/yanıltıcı z-score üretimini engeller.
STEP5_STD_EPSILON = 1e-6

# Windowed akış ana raster çıktıları üretir. Interpolated full NetCDF çıktısı
# büyük veri için tekrar yüksek bellek/disk baskısı yaratabileceği için kapalıdır.
STEP5_WRITE_INTERPOLATED_NETCDF = False
