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

# Landsat NDVI export ayarları. LST ile aynı QA mask / grid / pencere mantığı kullanılır.
LANDSAT_NDVI_EXPORT = {
    "file_name_prefix": "landsat_ndvi_dogu_akdeniz",
    "scale": 30,
}

ENABLE_MODIS_EXPORT = True
ENABLE_LANDSAT_EXPORT = True
ENABLE_MODIS_STEP5_CONTEXT = True

# NDVI ve TVDI ürünleri (yeni bilimsel yön). Mevcut LST anomaly pipeline'ından bağımsız.
ENABLE_NDVI_EXPORT = True
ENABLE_TVDI_STEP5 = True

LANDSAT_SCALE = 0.00341802
LANDSAT_OFFSET = 149.0

# Landsat 8 Collection 2 Level 2 yüzey yansıması (SR) bantları için scale/offset.
# SR_B4 = Red, SR_B5 = NIR. Reflectance = DN * SR_SCALE + SR_OFFSET.
LANDSAT_SR_SCALE = 0.0000275
LANDSAT_SR_OFFSET = -0.2
LANDSAT_RED_BAND = "SR_B4"
LANDSAT_NIR_BAND = "SR_B5"

REGION_NAME = "kozan_aoi"  # Test için küçük bölge kullan

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
STEP5_MIN_BASELINE_STD_CELSIUS = 1.0

# MODIS bağlam z-score hesabında standart sapma bu eşiğin altındaysa sonuç NaN bırakılır.
# MODIS (~1 km) yalnız düşük çözünürlüklü bağlam ürünü olarak kullanılır.
STEP5_MIN_MODIS_STD_CELSIUS = 1.0

# Bu eşiğin altında geçerli baseline gözlemi olan piksellerde mean/std/z-score
# hesaplanmaz; piksel NaN bırakılır.
STEP5_MIN_BASELINE_VALID_COUNT = 3

# Current period median en az bu kadar QA-temiz gözlemden oluşmalıdır.
# Aksi halde tek bulut kaçağı veya tek sahne kaynaklı soğuk pikseller anomaliye girmez.
STEP5_MIN_CURRENT_VALID_COUNT = 2

# Windowed akış ana raster çıktıları üretir. Interpolated full NetCDF çıktısı
# büyük veri için tekrar yüksek bellek/disk baskısı yaratabileceği için kapalıdır.
STEP5_WRITE_INTERPOLATED_NETCDF = False


# =============================================================================
# TVDI (Temperature Vegetation Dryness Index) ayarları
# =============================================================================
# TVDI = (LST - wet_edge) / (dry_edge - wet_edge)
# Her NDVI bin'i için wet_edge = düşük LST percentile, dry_edge = yüksek LST percentile.
# TVDI offline (Step5) katmanında hesaplanır; tiling/windowed akışı bozulmaz.

# NDVI ekseni bu kadar bin'e bölünür. LST-NDVI scatter üçgeni bu bin'lerle örneklenir.
TVDI_NDVI_BIN_COUNT = 20

# Geçerli NDVI aralığı. Bu aralık dışındaki pikseller TVDI'ye girmez.
TVDI_NDVI_MIN = 0.0
TVDI_NDVI_MAX = 1.0

# Wet/dry edge percentile'ları (her NDVI bin'i içindeki LST dağılımından).
TVDI_WET_EDGE_PERCENTILE = 2.0
TVDI_DRY_EDGE_PERCENTILE = 98.0

# Bir NDVI bin'inin edge hesabına katkı vermesi için gereken minimum geçerli piksel.
# Az örnekli bin'ler gürültülü edge üretir; bu bin'ler edge fit'ine alınmaz.
TVDI_MIN_PIXELS_PER_BIN = 30

# dry_edge - wet_edge bu eşiğin altındaysa TVDI NaN bırakılır (sıfıra bölme koruması).
TVDI_MIN_EDGE_SPAN_CELSIUS = 0.5

# Baseline TVDI std bu eşiğin altındaysa tvdi_anomaly_zscore NaN/maskeli bırakılır.
# Çok küçük baseline std, z-score'u yapay olarak şişirir (küçük paydaya bölme).
# Bu yüzden epsilon eklemek yerine düşük-std piksellerini maskeliyoruz.
MIN_TVDI_BASELINE_STD = 0.05

# Geriye dönük uyumluluk için eski isim; MIN_TVDI_BASELINE_STD ile aynı değer.
TVDI_MIN_BASELINE_STD = MIN_TVDI_BASELINE_STD

# Sıfıra bölme koruması için yalnız sayısal güvenlik amacıyla kullanılan çok küçük
# epsilon. Std'yi yapay şişirmek için DEĞİL; maskeleme bundan bağımsız yapılır.
TVDI_ZSCORE_NUMERICAL_EPSILON = 1e-9

# Bu eşiğin altında geçerli baseline TVDI gözlemi olan piksellerde z-score hesaplanmaz.
TVDI_MIN_BASELINE_VALID_COUNT = 3


# =============================================================================
# Burned-area / aktif yangın validation (skeleton — Phase 2'de doldurulacak)
# =============================================================================
# Bu collection'lar henüz ağır validation için kullanılmaz; sadece config/helper
# taslağı olarak tanımlanır. TVDI/dryness katmanı ile yanmış alan çakıştırması
# (ROC/AUC) bir sonraki bilimsel adımdır.
ENABLE_BURNED_AREA_VALIDATION = False

# MODIS yanmış alan (500 m, aylık)
MCD64A1_COLLECTION = "MODIS/061/MCD64A1"
MCD64A1_BURNDATE_BAND = "BurnDate"

# ESA FireCCI51 yanmış alan (250 m, aylık)
FIRECCI51_COLLECTION = "ESA/CCI/FireCCI/5_1"
FIRECCI51_BURNDATE_BAND = "BurnDate"

# FIRMS aktif yangın (MODIS/MCD14ML türevi, günlük)
FIRMS_COLLECTION = "FIRMS"
FIRMS_FIRE_BAND = "T21"


# =============================================================================
# Step6: burned-area association testing (ilk doğrulama)
# =============================================================================
# Step6, predictor rasterlarını (LST anomaly, current TVDI, TVDI difference,
# TVDI z-score) aynı sezon/AOI yanmış alan etiketlerine karşı test eder.
# Bu bir RF/XGBoost modeli DEĞİL; ilk burned-area ilişki (association) testidir.

# Yanmış alan etiketleri için sezon penceresi. Boş bırakılırsa current period
# baseline ile aynı sezon kullanılır. Yangın sezonu genelde current period'dan
# daha geniştir; bu yüzden ayrı tanımlanabilir.
VALIDATION_SEASON_START = "2023-06-01"
VALIDATION_SEASON_END = "2023-10-31"

# FIRMS aktif yangını da bir etiket kaynağı olarak dahil et (opsiyonel).
VALIDATION_INCLUDE_FIRMS = False

# Sınıf dengesizliği: yanmış pikseller genelde çok azdır. Dengeli metrikler için
# yanmayan pikseller bu oranda (burned_count * ratio) rastgele alt-örneklenir.
VALIDATION_BALANCED_UNBURNED_RATIO = 1.0

# Alt-örnekleme tekrarlanabilirliği için sabit seed.
VALIDATION_RANDOM_SEED = 42

# FIRMS aktif yangın parlaklık eşiği (T21, Kelvin). Bu değerin üzerindeki pikseller
# aktif yangın kabul edilir.
VALIDATION_FIRMS_BRIGHTNESS_THRESHOLD = 330.0

# Step6 etiket export çözünürlüğü (predictor grid'ine resample edilmeden önce GEE
# export ölçeği). Predictor 30 m olduğu için etiketler de 30 m'ye resample edilir.
VALIDATION_LABEL_EXPORT_SCALE = 30

# JSON çıktısında tutulacak ROC önizleme noktası üst sınırı. Full ROC array'leri
# (milyonlarca nokta olabilir) JSON'a YAZILMAZ; sadece bu kadar noktaya downsample
# edilmiş bir önizleme tutulur. ROC PNG'si full array'lerle bellekte üretilir.
VALIDATION_MAX_ROC_PREVIEW_POINTS = 500

# --- Validation modu ---
# "same_season": predictor ve label aynı sezon penceresinden (ilk association testi).
# "pre_fire":    predictor window ve label window ayrı (yangın öncesi kuruluk sinyali).
VALIDATION_MODE = "same_season"

# pre_fire modunda predictor ve label pencereleri ayrı tanımlanır.
# Predictor window: yangından ÖNCEKi kuruluk durumu.
# Label window:     sonraki yanmış alan kayıtları.
VALIDATION_PREFIRE_PREDICTOR_START = "2023-06-01"
VALIDATION_PREFIRE_PREDICTOR_END = "2023-07-31"
VALIDATION_PREFIRE_LABEL_START = "2023-08-01"
VALIDATION_PREFIRE_LABEL_END = "2023-10-31"

# FireCCI51 veri kapsamı (yaklaşık). Bu aralık dışındaki label window'ları için
# FireCCI51 Earth Engine'e SORULMADAN skip edilir (2023 verisi yoktur).
FIRECCI51_AVAILABLE_START = "2001-01-01"
FIRECCI51_AVAILABLE_END = "2020-12-31"