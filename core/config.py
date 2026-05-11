GEE_PROJECT = "b7-thermal-digital-twin"

MODIS_COLLECTION = "MODIS/061/MOD11A1"
LANDSAT_COLLECTION = "LANDSAT/LC08/C02/T1_L2"

START_DATE = "2019-01-01"
END_DATE = "2023-12-31"

EXPORT_FOLDER = "B7_Thermal_Digital_Twin"

MODIS_EXPORT = {
    "description": "export_modis_lst_5y_summer_mean",
    "file_name_prefix": "modis_lst_dogu_akdeniz_5y_summer_mean",
    "scale": 1000,
}

LANDSAT_EXPORT = {
    "file_name_prefix": "landsat_lst_dogu_akdeniz",
    "scale": 30,
}

DOWNLOAD_MODE = "drive"   # "drive" veya "direct"
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

MAX_LANDSAT_DAILY_EXPORTS = 10  # Test için 10 sahne yeterli

EXPORT_CRS = "EPSG:4326"

DIRECT_DOWNLOAD_DIR = "data"
DIRECT_LANDSAT_LST_SUBDIR = "landsat_timeseries"
DIRECT_LANDSAT_QA_SUBDIR = "landsat_qa"

DIRECT_DOWNLOAD_SCALE = 30
DIRECT_DOWNLOAD_MAX_IMAGES = 5

# Anomali pencere ayarları (TEST İÇİN KÜÇÜK PENCERELER)
BASELINE_START_DATE = "2022-06-01"  # Tek yaz sezonu
BASELINE_END_DATE = "2022-09-30"

# Current period - anomali hesabı için kullanılacak güncel pencere
CURRENT_PERIOD_DAYS = 30  # 30 gün test için yeterli
CURRENT_PERIOD_END_DATE = "2023-08-31"  # 2023 yaz sonu

STEP5_WINDOW_SIZE = 512
STEP5_STD_EPSILON = 1e-6  # Sıfır bölmeye karşı küçük bir değermis
STEP5_WRITE_INTERPOLATED_NETCDF = False