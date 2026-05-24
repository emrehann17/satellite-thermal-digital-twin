import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
GOOGLE_DRIVE_EXPORT_FOLDER_URL = os.getenv("GOOGLE_DRIVE_EXPORT_FOLDER_URL", "")
GOOGLE_DRIVE_EXPORT_FOLDER_ID = os.getenv("GOOGLE_DRIVE_EXPORT_FOLDER_ID", "")
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

ENABLE_MODIS_EXPORT = True
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

# Landsat baseline export üst sınırı. Bu değer hedef sayı değildir.
# Pencere-simetrik baseline'da gerçek export sayısı current yılından önceki
# baseline yıl sayısına bağlıdır; örn. 2019-2023 aralığı ve current=2023 ise 4.
# Production için None yapılabilir.
MAX_LANDSAT_DAILY_EXPORTS = 12

EXPORT_CRS = "EPSG:4326"

# Anomali baseline yıl aralığı.
# Step3, current period penceresinin aynı ay-gün aralığını bu yıllara taşır.
# Current yılı baseline'dan hariç tutulur.
BASELINE_START_DATE = "2019-06-01"
BASELINE_END_DATE = "2023-09-30"

# Current period - anomali hesabı için kullanılacak güncel pencere
CURRENT_PERIOD_DAYS = 45
CURRENT_PERIOD_END_DATE = "2023-08-31"

# Step5 bellek kullanımı ayarları
# Her seferinde okunacak raster pencere kenarı (piksel).
# Yaklaşık bellek: sahne_sayısı * STEP5_WINDOW_SIZE^2 * 4 byte.
STEP5_WINDOW_SIZE = 512

# Standart sapma bu eşiğin altındaysa z-score anomali NaN yazılır.
# Bu, düşük örnek sayısı veya yapay sabit piksellerde z-score patlamasını engeller.
STEP5_MIN_BASELINE_STD_CELSIUS = 1.5

# Bu eşiğin altında geçerli baseline gözlemi olan piksellerde mean/std/z-score
# hesaplanmaz; piksel NaN bırakılır.
STEP5_MIN_BASELINE_VALID_COUNT = 3

# Current period median en az bu kadar QA-temiz gözlemden oluşmalıdır.
# Aksi halde tek bulut kaçağı veya tek sahne kaynaklı soğuk pikseller anomaliye girmez.
STEP5_MIN_CURRENT_VALID_COUNT = 3

# Komşu pikseller arasında valid-count farkı bu eşikleri aşıyorsa coverage
# süreksizliği/dikiş riski var kabul edilir ve anomaly güven dışı bırakılır.
STEP5_CURRENT_COUNT_DISCONTINUITY_THRESHOLD = 1
STEP5_BASELINE_COUNT_DISCONTINUITY_THRESHOLD = 1

# Baseline std katmanında komşu pikseller arasında bu eşik kadar ani sıçrama
# varsa z-score paydasında dikiş riski var kabul edilir.
STEP5_BASELINE_STD_DISCONTINUITY_THRESHOLD = 0.5

# Windowed akış ana raster çıktıları üretir. Interpolated full NetCDF çıktısı
# büyük veri için tekrar yüksek bellek/disk baskısı yaratabileceği için kapalıdır.
STEP5_WRITE_INTERPOLATED_NETCDF = False