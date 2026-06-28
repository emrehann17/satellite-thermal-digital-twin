# Uydu Tabanlı Termal Veri İşleme Prototipi

Bu proje, Google Earth Engine (GEE) üzerinden alınan **MODIS** ve **Landsat** yüzey sıcaklığı verilerini kullanarak Doğu Akdeniz bölgesi için **termal çevre temsili** oluşturan modüler bir prototip sistemdir. Proje; veri sorgulama, sıcaklık işleme, GeoTIFF export, Google Drive üzerinden otomatik dosya alma ve Python ile yerel raster ön işleme adımlarını içeren bir altyapı kurmayı hedeflemektedir.

Bu repo şu anda tamamlanmış bir **3B termal dijital ikiz** veya **yangın riski tahmin sistemi** değildir. Mevcut haliyle, bu hedeflere ilerleyen bir **termal veri işleme ve ön analiz prototipi** olarak değerlendirilmelidir.

---

# Amaç

Projede amaç, uydu tabanlı yüzey sıcaklığı verilerini kullanarak Doğu Akdeniz bölgesine ait:

* geniş alanlı termal referans katmanlar üretmek,
* yüksek çözünürlüklü Landsat sıcaklık verilerini işlemek,
* zaman serisi tabanlı ön işleme akışı kurmak,
* anomali üretimine temel oluşturacak raster çıktılar elde etmek

ve daha sonraki aşamalarda bunları 3B görselleştirme, risk analizi ve karar destek katmanlarına bağlayabilecek bir temel hazırlamaktır.

---

# Proje Kapsamı

Projede şu adımlar yer almaktadır:

* çalışma bölgesinin tanımlanması
* MODIS LST verisinin sorgulanması
* MODIS için 4 yıllık yaz dönemi (2019-2022) baseline üretimi
* baseline için ortalama ve standart sapma hesaplanması
* Landsat yüksek çözünürlüklü LST verisinin hazırlanması
* Landsat günlük composite zaman serisi oluşturulması
* GeoTIFF export
* Drive export task polling
* Google Drive klasöründen otomatik indirme
* raster dosyalarının yerel veri klasörlerine otomatik yerleştirilmesi
* QA tabanlı bulut maskeleme
* Python ile zaman serisi ön işleme
* baseline ve anomaly raster üretimi
* geçerli gözlem sayısı ve düşük güven maskeleri ile anomaly teşhisi
* ileride 2B/3B görselleştirme ve risk analizi

---

# Mevcut Durum

Şu anda tamamlanan veya büyük ölçüde kurulan kısımlar şunlardır:

* GEE bağlantısı ve doğrulama
* çalışma bölgelerinin tanımlanması
* proje yapısının modüler hale getirilmesi (`core/` yapısı)
* ortak ayarların `config.py` altında toplanması
* MODIS veri sorgulama
* MODIS için 4 yıllık yaz dönemi (2019-2022) **ortalama** üretimi
* MODIS için 4 yıllık yaz dönemi (2019-2022) **standart sapma** üretimi
* Landsat veri sorgulama
* Landsat zaman serisi koleksiyonunun hazırlanması
* current period ile aynı takvim penceresine sahip geçmiş yıl Landsat median composite'lerinin hazırlanması
* current period için QA-maskeli günlük composite yığını üzerinden temporal median üretimi
* current period için geçerli günlük composite sayısı bandı üretimi
* Step4'te MODIS ve Landsat exportlarının aç/kapa mantığıyla kontrol edilmesi
* Step4'te Drive export task polling
* Step4b'de Google Drive klasörü indirme ve dosyaların yerel veri klasörlerine yerleştirilmesi
* `main.py` üzerinden Step1 -> Step5 akışının uçtan uca çalıştırılabilmesi
* Step5'in otomatik akış içinde çalıştırılması
* baseline mean raster, baseline std raster, baseline valid count raster, current period median raster, current period valid count raster ve z-score anomaly raster çıktılarının üretilmesi
* düşük baseline gözlem sayısı, düşük baseline standard deviation ve düşük current gözlem sayısı için tanı maskelerinin üretilmesi
* MODIS 4 yıllık yaz mean/std rasterının Step5'te Landsat gridine yeniden örneklenmesi
* MODIS mean/std üzerinden düşük çözünürlüklü bağlam z-score rasterı üretilmesi
* Step5 raster çıktılarının windowed/chunked okuma ile düşük bellek kullanarak oluşturulması
* NetCDF yerine daha hafif raster + metadata odaklı Step5 çıktı yapısına geçilmesi

---

# Şu Anda Gözlenen Sınırlılıklar

Proje çalışıyor olsa da şu anda bazı önemli teknik sınırlılıklar bulunmaktadır:

* Step4 otomatikleşmiş olsa da bunun sağlıklı çalışması için `DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT`, `GOOGLE_DRIVE_EXPORT_FOLDER_URL` veya `GOOGLE_DRIVE_EXPORT_FOLDER_ID` ayarlarının doğru verilmesi gerekir.
* Google Drive klasör URL/ID değerleri repoda hardcode edilmez; `GOOGLE_DRIVE_EXPORT_FOLDER_URL` veya `GOOGLE_DRIVE_EXPORT_FOLDER_ID` environment variable olarak verilmelidir.
* Drive indirme akışı `geemap/gdown` davranışına bağlıdır; klasör URL/ID, paylaşım yetkisi ve Drive erişimi yanlış ise indirme adımı durabilir.
* Doğrudan `ee.Image.getDownloadURL()` indirme yolu kaldırılmıştır; büyük rasterlar için ana akış Drive export, task polling ve geemap/gdown klasör indirmedir.
* Step5 şu ana kadar sınırlı sayıda pencere composite'i ve daha küçük bir bölge ile test edilmiştir. Güncel simetrik baseline ayarında 2019-2022 yılları için aynı current penceresi export edilir.
* Bu nedenle anomaly ve diğer raster çıktılarında veri bulunmayan beyaz alanlar oluşabilmektedir.
* Anomaly üretimi artık tek sahne yaklaşımı yerine belirli bir zaman penceresi içerisindeki QA-maskeli Landsat sahnelerinin median composite çıktısı üzerinden yapılmaktadır.
* Current yüzey artık tüm sahnelerin tek median'ı yerine QA-temiz günlük composite yığınının median'ı olarak üretilmektedir; buna rağmen küçük test pencerelerinde veya sınırlı sahne sayısında hala veri boşlukları oluşabilmektedir.
* Bu boşlukların temel nedeni yetersiz zamansal kapsama, bulut/QA maskelemesi ve Landsat sahnelerinin doğal mekansal kapsama farklılıklarıdır.
* MODIS (~1 km) ve Landsat (~30 m) çözünürlük farkı nedeniyle MODIS bağlam z-score ürünü yüksek çözünürlüklü Landsat anomaly'nin yerine geçmez.
* Baseline-current simetrisi Landsat tarafında pencere bazlı hale getirilmiştir; yine de bu yaklaşımın büyük bölge ve daha fazla tarih penceresi ile doğrulanması gerekir.
* Step5 çıktıları henüz büyük ölçekli veri ile tam doğrulanmış değildir.

---

# Geliştirme Aşamasında Olan Kısımlar

Şu anda aktif olarak geliştirilen / iyileştirilen alanlar:

* daha geniş zaman pencereleri ile veri boşluklarının azaltılması
* current period composite üretiminin büyük bölge üzerinde doğrulanması
* Step5 eşiklerinin daha geniş sahada test edilmesi
* çıktıların sunulabilir görsel paket haline getirilmesi
* büyük bölge için mosaic/VRT tabanlı raster okuma yaklaşımının eklenmesi

---

# Planlanan Çalışmalar

Henüz tamamlanmamış veya ileride geliştirilecek başlıklar:

* 2B çıktıların daha güçlü görselleştirilmesi
* 3B görselleştirme katmanı
* gelişmiş anomali / risk analizi
* karar destek yapısı
* yangın kayıtları ile veri birleştirme
* Jetson / YOLO entegrasyonu
* daha kararlı online/offline pipeline ayrımı
* Step5 sonrası aşamaların yapılandırılması

---

# Veri Kaynakları

## MODIS

* Veri kümesi: `MODIS/061/MOD11A1`
* İçerik: günlük kara yüzey sıcaklığı verisi
* Çözünürlük: yaklaşık 1 km
* Kullanım amacı: geniş alanlı termal baseline üretimi

## Landsat 8

* Veri kümesi: `LANDSAT/LC08/C02/T1_L2`
* İçerik: Surface Temperature (`ST_B10`) ve kalite bandı (`QA_PIXEL`)
* Çözünürlük: yaklaşık 30 m
* Kullanım amacı: yüksek çözünürlüklü termal katman ve zaman serisi işleme

---

# MODIS Baseline Seçiminin Gerekçesi

Projede MODIS verisi, Doğu Akdeniz için düşük çözünürlüklü ama zamansal olarak daha düzenli bir referans termal davranış üretmek amacıyla kullanılmaktadır.

Bu aşamada MODIS tarafında tam zaman serisi yerine 4 yıllık (2019-2022) yaz dönemi ortalaması ve standart sapması alınmıştır.
(Not: The data period includes 2019–2023, but the baseline statistics exclude the current year and currently use 4 historical baseline years: 2019–2022.) Bunun başlıca gerekçeleri şunlardır:

* MODIS, Landsat'a göre daha düzenli gözlem sıklığı sağlar.
* 4 yıllık pencere, tek bir yıla göre daha kararlı bir baseline üretir.
* Yaz aylarına odaklanılması, yüksek sıcaklık davranışını inceleme amacıyla uyumludur.
* Ortalama tek başına yeterli olmadığı için, değişkenliği tanımlamak üzere standart sapma da eklenmiştir.
* Bu yapı, Step5'te düşük çözünürlüklü MODIS bağlam z-score hesabı için kullanılmaktadır.

Bu nedenle MODIS çıktısı, Landsat'ın yerine geçen bir zaman serisi değildir. Step5, MODIS mean/std rasterını Landsat gridine yeniden örnekleyerek ayrı bir `modis_context_zscore.tif` üretir. Ana yüksek çözünürlüklü anomaly ürünü yine Landsat pencere baseline'ına dayalı `anomaly_zscore.tif` çıktısıdır.

---

# Metodoloji

Projede genel akış şu şekildedir:

1. Çalışma bölgesi GEE üzerinde tanımlanır.
2. MODIS verisi kullanılarak 4 yıllık yaz dönemi baseline katmanları hazırlanır.
3. Bu baseline için ortalama ve standart sapma hesaplanır.
4. Landsat verisi aynı bölge için filtrelenir.
5. Current period için belirli bir temporal window tanımlanır.
6. Bu pencerenin aynı ay-gün aralığı geçmiş baseline yıllarına taşınır.
7. Her baseline yılı için QA-maskeli pencere median composite üretilir.
8. Current period için QA-maskeli Landsat yığını üzerinden median yüzey ve geçerli gözlem sayısı bandı üretilir.
9. MODIS ve Landsat rasterları GeoTIFF olarak Google Drive'a export edilir.
10. Export task'ları polling ile tamamlanana kadar izlenir.
11. Drive klasörü otomatik indirilir ve dosyalar yerel veri klasörlerine dağıtılır.
12. QA verisi kullanılarak bulut, gölge, cirrus, kar ve düşük güvenli pikseller maskelenir.
13. Python tarafında zaman serisi kurulup baseline mean/std, valid count, current median, current valid count ve Landsat z-score anomaly rasterları üretilir.
14. MODIS mean/std rasterı Landsat gridine bilinear yeniden örneklenir ve current median ile düşük çözünürlüklü bağlam z-score rasterı üretilir.
15. Düşük baseline gözlem sayısı, düşük baseline std ve düşük current gözlem sayısı maskeleriyle anomaly çıktısı teşhis edilir.

Landsat temel istatistiklerinde zamansal interpolasyon kullanılmaz. Temel ortalama ve standart sapma değerleri yalnızca kalite kontrolünden geçmiş gerçek gözlemlerden hesaplanır; geçerli gözlem eşiğinin altındaki pikseller NaN olarak kalır veya maskelenir.

---

## İnterpolasyon Politikası

Landsat baseline istatistiklerinde zamansal interpolasyon kullanılmaz. Eksik veya yetersiz geçerli gözleme sahip pikseller doldurulmaz; `NaN` olarak bırakılır veya maskelenir. Bu karar, yapay mekansal kapsama üretmemek ve baseline standart sapmasını interpolasyonla yapay olarak düşürmemek için uygulanmıştır.

MODIS rasterları yalnızca düşük çözünürlüklü bağlamsal karşılaştırma için Landsat gridine mekansal olarak yeniden örneklenir. Bu işlem temporal interpolation değildir. MODIS, Landsat baseline boşluklarını doldurmak için kullanılmaz ve ana anomaly baseline olarak yorumlanmaz.

---

# Proje Yapısı

```text
core/
├── config.py
├── drive_downloader.py
├── gee_utils.py
├── io_utils.py
├── paths.py
└── regions.py

src/
├── step1_fetch_modis.py
├── step2_modis_5year_mean.py
├── step2b_dem.py
├── step3_landsat_lst.py
├── step4_export_geotiff.py
├── step4b_download_drive_export.py
├── step5_preprocess_timeseries.py
├── step5b_diagnostic_report.py
├── step5c_tvdi.py
└── step6_validate_fire_relation.py

scripts/
├── main.py
└── standalone_step5.py

```

---

# Step Açıklamaları

## Step 1

GEE bağlantısını başlatır, çalışma bölgelerini tanımlar ve temel MODIS sorgusunu gerçekleştirir.

## Step 2

MODIS verisini kullanarak 4 yıllık yaz dönemi baseline katmanlarını üretir. Bu aşamada ortalama ve standart sapma hesaplanır.

## Step 2B

DEM (statik yardımcı katman) hazırlama adımıdır. MODIS/Landsat ürünleriyle **aynı lifecycle'ı** izler: bu adım yalnızca veriyi hazırlar ve metadata yazar; export veya indirme yapmaz, `geemap.ee_export_image()` çağırmaz.

DEM kaynağı olarak öncelikle **Copernicus DEM GLO-30** denenir; erişilemezse **USGS/SRTMGL1_003** fallback'ine düşülür. Hangi kaynağın kullanıldığı metadata'ya yazılır. İki ürün hazırlanır:

* `elevation` — DEM yükseklik bandı (metre)
* `slope` — `ee.Terrain.slope` ile elevation'dan türetilen eğim (derece)

Yapılandırılmış AOI (`REGION_NAME`) ve `EXPORT_CRS` kullanılır. Çıktı metadata dosyası:

```text
outputs/step2b/step2b_dem_metadata.json
```

Metadata; dataset, fallback dataset, AOI, CRS, export çözünürlüğü, ürünler (elevation/slope), export dosya adları ve oluşturma zamanını içerir.

## Step 3

Landsat baseline ve current period koleksiyonlarını hazırlar. Güncel yöntemde baseline, tüm yaz günlük composite yığını değildir; current period ile aynı ay-gün aralığındaki geçmiş yıl pencerelerinin QA-maskeli median composite'lerinden oluşur.

Örnek:

```text
current  = 2023-07-17 -> 2023-08-31 median
baseline = 2019-07-17 -> 2019-08-31 median
baseline = 2020-07-17 -> 2020-08-31 median
baseline = 2021-07-17 -> 2021-08-31 median
baseline = 2022-07-17 -> 2022-08-31 median
```

Step5 bu geçmiş yıl pencere medianları üzerinden baseline mean/std üretir.

Current period çıktısı iki bantlıdır:

* `Current_Period_LST_Celsius`
* `Current_Period_Valid_Count`

`Current_Period_LST_Celsius`, QA-temiz günlük median composite yığını üzerinden üretilen current median yüzeydir.
`Current_Period_Valid_Count`, pikselin kaç QA-temiz günlük composite tarafından desteklendiğini gösterir.

## Step 4

Online export katmanıdır. MODIS ve Landsat rasterlarını GeoTIFF olarak Google Drive'a export eder ve export task'larını polling ile izler. Bu adım artık Drive klasörünü indirmez; indirme ve yerel klasörlere dağıtma sorumluluğu Step4b'ye ayrılmıştır.

Step4 ayrıca DEM ürünlerini (`elevation`, `slope`) Landsat/MODIS ile **tamamen aynı export mekanizması** (`export_image_to_drive`) üzerinden export eder: aynı Drive klasörü, aynı CRS, aynı region handling, aynı task polling ve aynı overwrite politikası. DEM görüntüsü Step2B helper'ı (`prepare_dem_products`) ile üretilir; ayrı/yeni bir export yolu eklenmez.

Step4 tek başına çalıştırıldığında da Step3 ile aynı pencere-simetrik baseline/current helper'ını kullanır. Böylece standalone export yolu eski günlük baseline mantığına düşmez; `REGION_NAME`, baseline yıl aralığı ve current period ayarları `core/config.py` üzerinden gelir.

## Step 4B

Drive download ve yerel dosya yerleştirme katmanıdır. Step4 tarafından tamamlanan Drive export dosyalarını indirir ve GeoTIFF dosyalarını Step5'in beklediği yerel klasörlere dağıtır.

Yerel klasör yerleştirmesi şu şekildedir:

* QA dosyaları -> `data/landsat_qa`
* Baseline Landsat LST dosyaları -> `data/landsat_timeseries`
* Current period median dosyası -> `data/current_period`
* MODIS export dosyaları -> `data/modis`
* DEM dosyaları -> `data/dem/elevation.tif` ve `data/dem/slope.tif`

DEM indirme, Landsat/MODIS ile aynı akışı kullanır: zaten varsa atlama (`DRIVE_DOWNLOAD_OVERWRITE` kapalıyken mevcut dosya korunur), aksi halde indirme. DEM dosya adları kanonik (`elevation.tif`/`slope.tif`) hale getirilir; GEE parçalı export üretirse tile son ekleri korunarak çakışma önlenir.

## Step 5

Offline raster işleme katmanıdır. Step4b tarafından yerleştirilen GeoTIFF dosyaları okunur, QA tabanlı maskeleme uygulanır ve aşağıdaki çıktılar üretilir:

* baseline mean raster
* baseline standard deviation raster
* baseline valid count raster
* current period median raster
* current period valid count raster
* MODIS mean/std rasterlarının Landsat gridine yeniden örneklenmiş halleri
* MODIS bağlam z-score raster
* z-score anomaly raster
* düşük güven tanı maskeleri
* metadata JSON çıktısı

Step5 artık tüm zamanı bellekte tutan `xarray + full stack` yolu yerine windowed/chunked raster işleme ile çalışmaktadır. Baseline istatistiklerinde lineer zaman interpolasyonu kullanılmaz; yeterli geçerli gözlem olmayan pikseller `NaN` bırakılır. MODIS çıktıları yalnız bağlam ürünü olarak kullanılır; Landsat z-score ürününün yerine geçmez.

## Step 5B

Mevcut Step5 çıktıları üzerinde tanı raporu üretir. Yeni preprocessing yapmaz, anomaly değerlerini değiştirmez ve smoothing/blur uygulamaz. `outputs/diagnostics/summary.md`, `outputs/diagnostics/diagnostic_stats.json` ve PNG görseller üretir.

## main.py

`main.py`, Step1'den Step5'e kadar olan akışı organize biçimde çalıştırır. Güncel akışta Step4 export/polling sürecini tamamlar, Step4b ile Drive çıktılarını indirip yerel klasörlere dağıtır ve Step5 ile Landsat anomaly ve MODIS bağlam ürünlerini üretir.

---

# Step5 Çıktıları

Step5 sonunda aşağıdaki raster ve metadata çıktıları üretilmektedir:

* `baseline_lst_mean_celsius.tif`
* `baseline_lst_std_celsius.tif`
* `baseline_valid_count.tif`
* `low_baseline_count_mask.tif`
* `low_baseline_std_mask.tif`
* `current_period_median_celsius.tif`
* `current_period_valid_count.tif`
* `low_current_count_mask.tif`
* `anomaly_zscore.tif`
* `modis_lst_mean_celsius_resampled.tif`
* `modis_lst_std_celsius_resampled.tif`
* `modis_context_zscore.tif`
* `step5_metadata.json`

Tanı maskelerinde `1` değeri ilgili pikselin o nedenle güven dışı kaldığını gösterir. Bu katmanlar özellikle anomaly rasterındaki doygun mavi alanların kaynağını ayırmak için kullanılır:

* `low_baseline_count_mask.tif`: baseline döneminde yeterli geçerli Landsat gözlemi yoktur.
* `low_baseline_std_mask.tif`: baseline standart sapması z-score için çok düşüktür.
* `low_current_count_mask.tif`: current period median yeterli QA-temiz gözlemden oluşmamıştır.

Güncel z-score formülü:

```text
z_score = (current_median - baseline_mean) / baseline_std
```

Bu hesap yalnızca şu koşullar sağlandığında yapılır:

* baseline geçerli gözlem sayısı `STEP5_MIN_BASELINE_VALID_COUNT` eşiğini geçmelidir.
* baseline standard deviation `STEP5_MIN_BASELINE_STD_CELSIUS` eşiğini geçmelidir.
* current period geçerli gözlem sayısı `STEP5_MIN_CURRENT_VALID_COUNT` eşiğini geçmelidir.

MODIS bağlam z-score formülü:

```text
modis_context_zscore = (current_median - modis_summer_mean_resampled) / modis_summer_std_resampled
```

Bu ürün `STEP5_MIN_MODIS_STD_CELSIUS` eşiğiyle maskelenir ve yalnız düşük çözünürlüklü bağlam katmanı olarak yorumlanmalıdır.

---

# Kurulum

```bash
git clone https://github.com/emrehann17/satellite-thermal-digital-twin.git
cd satellite-thermal-digital-twin

```

```bash
python -m venv .venv

```

Windows:

```bash
.venv\Scripts\activate

```

Linux / macOS:

```bash
source .venv/bin/activate

```

```bash
pip install -r requirements.txt

```

```bash
earthengine authenticate

```

Google Drive otomatik indirme kullanılacaksa Drive klasör referansı environment variable olarak verilmelidir. Bu değerler güvenlik ve repo hijyeni nedeniyle `core/config.py` içine hardcode edilmez.

Windows PowerShell:

```powershell
$env:GOOGLE_DRIVE_EXPORT_FOLDER_ID="DRIVE_KLASOR_ID"
```

Linux / macOS:

```bash
export GOOGLE_DRIVE_EXPORT_FOLDER_ID="DRIVE_KLASOR_ID"
```

Alternatif olarak klasör URL'si kullanılabilir:

```bash
export GOOGLE_DRIVE_EXPORT_FOLDER_URL="https://drive.google.com/drive/folders/DRIVE_KLASOR_ID"
```

---

# Güncel Test Ayarları

`core/config.py` içinde güncel test çalıştırması için öne çıkan ayarlar:

```python
REGION_NAME = "kozan_aoi"
ENABLE_MODIS_EXPORT = True
ENABLE_LANDSAT_EXPORT = True
ENABLE_MODIS_STEP5_CONTEXT = True

MAX_LANDSAT_DAILY_EXPORTS = 12

BASELINE_START_DATE = "2019-06-01"
BASELINE_END_DATE = "2023-09-30"

CURRENT_PERIOD_DAYS = 45
CURRENT_PERIOD_END_DATE = "2023-08-31"

STEP5_MIN_BASELINE_STD_CELSIUS = 1.0
STEP5_MIN_MODIS_STD_CELSIUS = 1.0
STEP5_MIN_BASELINE_VALID_COUNT = 3
STEP5_MIN_CURRENT_VALID_COUNT = 2
```

Bu ayarla beklenen GEE export task sayısı yaklaşık olarak:

* 4 Landsat baseline window LST
* 4 Landsat baseline window QA
* 1 current period raster
* 1 MODIS raster
* toplam yaklaşık 10 task

Production veya daha kapsamlı denemelerde baseline yıl aralığı ve pencere sayısı artırılabilir. Büyük bölgeye geçmeden önce küçük bölge ve kontrollü test önerilir.

---

# Çalıştırma Sırası

## Uçtan Uca Akış

```bash
python scripts/main.py

```

Bu komut güncel akışta sırasıyla şunları yürütür:

* Step1: GEE bağlantısı ve temel veri sorguları
* Step2: MODIS baseline üretimi
* Step3: Landsat günlük composite ve current period hazırlığı
* Step4: Drive export ve task polling
* Step4b: Drive klasörü indirme ve dosya yerleştirme
* Step5: windowed/chunked raster ön işleme, Landsat anomaly ve MODIS bağlam üretimi

## Adım Adım Çalıştırma

```bash
python src/step1_fetch_modis.py
python src/step2_modis_5year_mean.py
python src/step3_landsat_lst.py
python src/step4_export_geotiff.py
python src/step4b_download_drive_export.py
python src/step5_preprocess_timeseries.py
python src/step5b_diagnostic_report.py

```

## Step4 -> Step4b -> Step5 Geçişi

Step4, Drive export task'larını polling ile tamamlanana kadar bekler ve `outputs/step4/step4_metadata.json` dosyasına export listesini yazar. Step4 artık Drive klasörü indirme veya yerel klasörlere dosya dağıtma yapmaz.

Step4b, `DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT=True` yapılır ve `GOOGLE_DRIVE_EXPORT_FOLDER_URL` veya `GOOGLE_DRIVE_EXPORT_FOLDER_ID` verilirse Drive klasörünü geemap/gdown ile indirir. Bu aşamada config ayarları ve Google Drive dosya erişim izinlerini kullanıcının manuel olarak yapması beklenir.

İndirilen GeoTIFF dosyaları otomatik olarak şu klasörlere yerleştirilir:

* QA dosyaları -> `data/landsat_qa`
* Baseline Landsat LST dosyaları -> `data/landsat_timeseries`
* Current period median dosyası -> `data/current_period`
* MODIS export dosyaları -> `data/modis`

Bu yerleştirme tamamlandıktan sonra Step5 aynı veri akışı üzerinden çalışır.

Step5 mümkünse `outputs/step4/step4_metadata.json` içindeki son export listesini kullanarak baseline dosyalarını seçer. Böylece `data/landsat_timeseries` içinde önceki denemelerden kalan GeoTIFF dosyalarının yeni baseline yığınına karışması engellenir. Metadata okunamazsa Step5 klasördeki uygun Landsat LST dosyalarını kullanır ve log'a uyarı yazar.

Current period tarafında Step5, `CURRENT_PERIOD_DAYS` ile eşleşen dosya adını öncelikli seçer. Örneğin `CURRENT_PERIOD_DAYS=45` ise `landsat_current_period_45days*.tif` dosyası beklenir.

## Step5'i Tek Başına Çalıştırma

```bash
python src/step5_preprocess_timeseries.py

```

Bu komut, Step5'i tek başına yeniden çalıştırmak istediğinde kullanılabilir. Normal kullanımda `scripts/main.py` akışı içinde otomatik tetiklenir.

## Step5B Tanı Raporu

```bash
python src/step5b_diagnostic_report.py

```

Bu komut mevcut Step5 rasterlarını okuyarak anomaly histogramı, `|z| > 2` ve `|z| > 3` oranları, valid-count/std maskeleriyle extreme anomaly çakışmaları ve varsa MODIS context agreement özetini üretir. Raster gridleri uyuşmuyorsa karşılaştırmayı atlar ve raporda belirtir.

---

# Örnek Çıktılar

Bu bölümdeki örnek çıktılar geçici olarak devre dışı bırakılmıştır. Mevcut anomaly haritalarında dikiş/artefact doğrulaması ve legend, CRS, ölçek, quality-mask gibi final harita bileşenleri tamamlandığında örnek çıktılar tekrardan eklenecektir.

---

# Çıktılar Hakkında Önemli Notlar

Bu görseller, projenin mevcut geliştirme aşamasını temsil eden erken prototip çıktılarıdır ve aşağıdaki sınırlamaları içermektedir:

* Çıktılar, işlem süresini kısaltmak amacıyla sınırlı sayıda Landsat pencere composite'i ile üretilmiştir.
* Bu nedenle anomaly raster üzerinde veri bulunmayan alanlar oluşabilmektedir.
* Bu boşluklar çoğunlukla yetersiz zamansal kapsama, QA maskelemesi, düşük baseline gözlem sayısı veya düşük current gözlem sayısından kaynaklanmaktadır.
* Current thermal state, QA-temiz günlük Landsat composite yığını üzerinden tanımlanmaktadır. Bu yaklaşım, sahne-bazlı current tanımına göre daha kararlı bir yöntemdir.
* MODIS bağlam z-score rasterı 1 km kaynak veriden üretildiği için 30 m Landsat ürünü gibi yorumlanmamalıdır.
* Anomaly rasterındaki mavi/kırmızı alanlar doğrudan fiziksel yorumlanmamalıdır; `valid_count`, düşük güven maskeleri ve MODIS bağlam ürünü ile birlikte değerlendirilmelidir.

---

# DEM Aşaması (elevation + slope)

DEM (Digital Elevation Model), pipeline'a mevcut mimariyi bozmadan, MODIS/Landsat ürünleriyle **birebir aynı lifecycle** üzerinden eklenmiştir:

```text
Step2B (prepare + metadata)
   -> Step4 (Google Earth Engine export task)
      -> Step4b (Drive export indirme)
```

İki statik ürün üretilir:

* `elevation` — Copernicus DEM GLO-30 (fallback: USGS/SRTMGL1_003), metre
* `slope` — `ee.Terrain.slope` ile elevation'dan türetilir, derece

İndirilen dosyalar:

```text
data/dem/elevation.tif
data/dem/slope.tif
```

## Önemli durum notları

* **DEM artık indiriliyor.** Step2B hazırlar, Step4 export eder, Step4b `data/dem/` altına yerleştirir.
* **DEM henüz Step5 içinde KULLANILMIYOR.** Step5/Step5B/Step5C/Step6 akışı değiştirilmemiştir; DEM bu adımlara hiçbir girdi sağlamaz.
* **DEM, ileride RF/XGBoost tabanlı MODIS downscaling için hazırlanmaktadır.** Bu aşama henüz uygulanmamıştır; DEM şimdilik yalnızca veri olarak hazır bulundurulur.
* **elevation ve slope statik yardımcı (auxiliary) predictor'lardır.** Zamanla değişmezler; gelecekteki downscaling modelinde sabit yardımcı değişkenler olarak kullanılmaları planlanmaktadır.

Bu aşama RF, XGBoost, downscaling veya ESA WorldCover içermez ve mevcut bilimsel çıktıları (LST anomaly, TVDI, burned-area association) etkilemez.

---

# Planlanan İyileştirmeler

* Pencere-simetrik baseline yaklaşımının farklı pencereler ve büyük bölge üzerinde doğrulanması
* Current composite stratejisinin daha geniş sahada test edilmesi
* Büyük bölgelerde parçalı GeoTIFF exportları için mosaic/VRT tabanlı daha sağlam okuma akışının kurulması
* Nihai, doğrulanmış ve daha temiz görsel çıktılarla bu bölümün güncellenmesi

---

# Not

Bu repo şu anda tamamlanmış bir 3B termal dijital ikiz sistemi değildir. Mevcut haliyle, MODIS ve Landsat tabanlı termal veri işleme ve ön analiz akışını kurmaya odaklanan bir prototip çalışmadır.

Şu anki en önemli açık konular:

* anomaly'nin tüm bölgeyi temsil edecek şekilde güçlendirilmesi
* daha fazla Landsat sahnesi ile Step5'in yeniden test edilmesi
* veri boşluklarının azaltılması
* pencere-simetrik baseline yaklaşımının daha fazla test edilmesi
* büyük bölge tiling/parçalı GeoTIFF akışının tam çözülmesi
* Step5 çıktılarının büyük veri ile daha güçlü doğrulanması

Bu eksikler giderildikçe proje daha güçlü 2B/3B termal temsil ve risk analizi katmanlarına doğru genişletilecektir.
---

# Step5C: TVDI-based Dryness Anomaly

Bu bölüm, projeyi çıplak LST (Land Surface Temperature) anomalisinden, LST + NDVI
temelli bir **TVDI (Temperature Vegetation Dryness Index)** / kuruluk göstergesine
doğru nasıl genişlettiğimizi anlatır. Step5C, mevcut LST anomaly pipeline'ının
yerine geçmez; onun yanında çalışan **yeni ve ayrı bir prototip ürün** ailesidir.

## Neden TVDI? (amaç)

Mevcut Step5 LST anomaly ürünü yalnızca **sıcaklık sapmasını** gösterir: bir piksel,
kendi baseline'ına göre ne kadar sıcak/soğuk? Ancak ham yüzey sıcaklığı tek başına
kuruluğu temsil etmekte zayıftır, çünkü sıcaklık bitki örtüsü yoğunluğuna güçlü
bağlıdır — çıplak/seyrek toprak doğal olarak sıcak, yoğun bitki örtüsü doğal olarak
serindir.

TVDI, **LST ile NDVI arasındaki ilişkiyi** kullanarak bu bağımlılığı normalize eder.
Aynı NDVI seviyesindeki pikselleri kendi içinde kıyaslayıp, bir pikselin o bitki
örtüsü sınıfı için ne kadar "kuru/sıcak uç"ta olduğunu ölçer. Bu sayede TVDI, bitki
örtüsü durumuna göre düzeltilmiş bir **termal kuruluk** temsili verir.

Bu fiziksel düzeltme nedeniyle TVDI, çıplak LST z-score'a kıyasla yangınla daha güçlü
bir fiziksel bağ kuran bir **aday kuruluk göstergesi (candidate dryness indicator)**
olabilir. Bu hipotez henüz doğrulanmamıştır; yanmış alan kayıtlarına karşı test
edilmesi gerekir (bkz. *Next validation step*).

## TVDI formülü

Her piksel için, o pikselin NDVI değerinin düştüğü NDVI aralığı (bin) içindeki LST
dağılımının uçları kullanılır:

```
TVDI = (LST - LST_wet_edge) / (LST_dry_edge - LST_wet_edge)
```

* **wet edge** — aynı NDVI aralığındaki **düşük LST sınırı** (nemli/serin uç; düşük
  percentile, örn. p2). Su stresi olmayan, buharlaşma-terlemesi yüksek yüzeyler.
* **dry edge** — aynı NDVI aralığındaki **yüksek LST sınırı** (kuru/sıcak uç; yüksek
  percentile, örn. p98). Su stresi altında, buharlaşma-terlemesi kısıtlı yüzeyler.
* **TVDI → 0**'a yakınsa: daha **nemli** / düşük kuruluk.
* **TVDI → 1**'e yakınsa: daha **kuru** / yüksek kuruluk.

TVDI değerleri `[0, 1]` aralığına clamp edilir. Bir NDVI bin'inde yeterli geçerli
piksel yoksa veya dry/wet edge farkı çok küçükse, o pikseller NaN bırakılır
(temporal interpolation uygulanmaz).

## Step5C çıktıları (prototype / diagnostic outputs)

Aşağıdaki dosyalar `outputs/step5c/` (rasterlar) ve `outputs/diagnostics/`
(görseller) altında üretilir. Görseller şu an **final ürün değil**, prototip /
diagnostic çıktılardır:

Raster çıktıları:
* `current_tvdi.tif` — current period TVDI (kuruluk durumu)
* `baseline_tvdi_mean.tif` — baseline yıllarının TVDI ortalaması
* `baseline_tvdi_std.tif` — baseline TVDI standart sapması
* `tvdi_difference.tif` — **ham fark** ürünü: `current_tvdi - baseline_tvdi_mean`.
  z-score'a göre yorumlanması daha kolay olan tamamlayıcı (companion) ürün.
* `tvdi_anomaly_zscore.tif` — current TVDI'nin baseline'a göre z-score anomalisi
  (düşük baseline std güvenilirlik kontrolünden geçirilmiş; aşağıya bakınız)
* `baseline_tvdi_valid_count.tif` — her piksele kaç baseline yılının geçerli TVDI
  katkısı verdiği
* `current_tvdi_valid_count.tif` — current TVDI'ye katkı veren geçerli gözlem sayısı

Diagnostic görselleri (Step5B üretir):
* `current_tvdi_map.png`
* `baseline_tvdi_mean_map.png`
* `baseline_tvdi_std_map.png`
* `tvdi_difference_map.png`
* `tvdi_anomaly_zscore_map.png`
* `tvdi_anomaly_histogram.png`

## TVDI z-score güvenilirlik kontrolü (baseline std reliability)

TVDI z-score, `(current_tvdi - baseline_tvdi_mean) / baseline_tvdi_std` ile
hesaplanır. Burada bir kırılganlık vardır: **baseline TVDI std çok küçükse**, küçük
bir paydaya bölme nedeniyle z-score yapay olarak şişer ve `|z| > 3` oranı
gerçekçi olmayan seviyelere çıkar.

Bunu ele almak için Step5C, std'yi yapay olarak epsilon ile şişirmek yerine
**düşük-std piksellerini maskeler**:

* Config'de `MIN_TVDI_BASELINE_STD` (varsayılan `0.05`) eşiği tanımlıdır.
* Bir pikselde `baseline_tvdi_std < MIN_TVDI_BASELINE_STD` ise o piksel için
  `tvdi_anomaly_zscore` NaN bırakılır.
* Sıfıra bölme yalnız çok küçük bir sayısal epsilon (`1e-9`) ile önlenir; bu
  epsilon güvenilirlik maskelemesinin yerini tutmaz, sadece sayısal güvenlik içindir.
* Step5C metadata'sı kaç pikselin bu nedenle maskelendiğini
  (`low_tvdi_std_masked_pixel_count`) ve oranını raporlar; Step5B summary'si
  maskeleme öncesi/sonrası `|tvdi_z| > 2` ve `|tvdi_z| > 3` oranlarını yazar.
* `|tvdi_z| > 3` oranı %10'un üzerindeyse Step5B summary'sine bir **instabilite
  uyarısı** eklenir (baseline std çok düşük veya baseline örnek sayısı yetersiz
  olabilir).

Bu nedenle çıplak z-score'a ek olarak `tvdi_difference.tif` daha sağlam ve
yorumlanabilir bir tamamlayıcı ürün olarak sağlanır.

## LST anomaly vs TVDI anomaly farkı

Projede artık iki ayrı anomali ürünü bir arada bulunur:

* `thermal_anomaly_zscore.tif` (Step5) — **sıcaklık anomalisi**. Pikselin baseline'ına
  göre LST sapması. NDVI'ye göre düzeltme yoktur.
* `tvdi_anomaly_zscore.tif` (Step5C) — **NDVI'ye göre normalize edilmiş kuruluk
  anomalisi**. Bitki örtüsü etkisi giderildikten sonra kalan kuruluk sapması.

İkisi farklı şeyleri ölçer ve bu fark kasıtlıdır. Yangın doğrulama aşamasında
**TVDI anomaly, ana aday predictor (candidate predictor)** olarak değerlendirilecek;
LST anomaly ise referans/karşılaştırma tabanı olarak korunacaktır. z-score'a ek
olarak `tvdi_difference.tif` (current_tvdi − baseline_tvdi_mean), düşük baseline std
kaynaklı şişmeden etkilenmediği için daha sağlam bir tamamlayıcı göstergedir.

## Interpretation status

* Step5C çıktıları küçük Kozan AOI üzerinde başarıyla üretildi (*Step5C outputs have
  been generated successfully on the small Kozan AOI*).
* Current TVDI ve baseline TVDI haritaları tutarlı mekansal desenler gösteriyor
  (*coherent spatial patterns*).
* TVDI anomaly **henüz doğrulanmış bir yangın-riski ürünü olarak değerlendirilmiyor**
  (*not yet treated as a validated fire-risk product*). Bu sürüm bir **first
  TVDI-based dryness prototype**'tır.
* Bir sonraki adım, TVDI'nin yanmış alan ve aktif yangın veri kümelerine karşı
  doğrulanmasıdır (*TVDI will be validated against burned-area records*).

# Step6: Burned-Area Association Test (ilk doğrulama)

`step6_validate_fire_relation.py`, predictor rasterlarını aynı sezon/AOI yanmış
alan etiketlerine karşı test eden **ilk burned-area ilişki (association)
testidir**. Bu bir yangın-riski MODELİ değildir ve RF/XGBoost eğitmez.

## Test edilen predictor'lar

* `thermal_anomaly_zscore.tif` (Step5) — LST sıcaklık anomalisi
* `current_tvdi.tif` (Step5C) — sürekli kuruluk göstergesi
* `tvdi_difference.tif` (Step5C) — sürekli fark göstergesi (current − baseline mean)
* `tvdi_anomaly_zscore.tif` (Step5C) — güvenilirlik-filtreli anomali

## Etiket kaynakları (GEE)

* **MCD64A1** — MODIS yanmış alan (500 m, aylık)
* **FireCCI51** — ESA CCI FireCCI 5.1 yanmış alan (250 m, varsa)
* **FIRMS / MCD14ML** — aktif yangın (opsiyonel, `VALIDATION_INCLUDE_FIRMS`)

## Validation modları

Step6 iki mod destekler (`VALIDATION_MODE`):

* **same_season** — predictor rasterları ve yanmış alan etiketleri AYNI sezon
  penceresinden alınır. İlk burned-area ilişki (association) testidir; predictor
  ile label aynı dönemde çakıştırılır.
* **pre_fire** — predictor window ve label window AYRIDIR. Yangından önceki
  kuruluk sinyalinin, sonraki yanmış alanlarla ilişkisini test etmek içindir.
  Varsayılan örnek: predictor window `2023-06-01 -> 2023-07-31`, label window
  `2023-08-01 -> 2023-10-31`.

pre_fire modu bilimsel olarak yalnızca predictor rasterları gerçekten o predictor
window'a göre üretilmişse anlamlıdır. Mevcut Step5/Step5C çıktıları farklı bir
current period'dan geliyorsa summary.md bir uyarı yazar ("Predictor rasters may
not match the configured pre-fire predictor window; regenerate Step5/Step5C ...").

### Pre-fire deneyini çalıştırma

`VALIDATION_MODE = "pre_fire"` yapıldığında `core/config.py`, current period'u
otomatik olarak pre_fire **predictor window**'undan türetir
(`CURRENT_PERIOD_END_DATE` = predictor_end, `CURRENT_PERIOD_DAYS` = pencere günü).
Böylece Step3/Step4/Step5/Step5C çıktıları yangından ÖNCEKi dönemi temsil eder ve
label window'a sızmaz. Config yüklemesi, predictor window'un label window'la
çakışmadığını da doğrular (çakışırsa `ValueError`).

Uçtan uca koşu (yeni klasör yapısına göre):

```bash
# 1. core/config.py içinde:
#      VALIDATION_MODE   = "pre_fire"
#      PREDICTOR_START_DATE = "2023-06-01"
#      PREDICTOR_END_DATE   = "2023-07-31"
#      LABEL_START_DATE     = "2023-08-01"
#      LABEL_END_DATE       = "2023-10-31"
# 2. (isteğe bağlı) same-season çıktılarını korumak için yedekleyin:
cp -r outputs/step5  outputs/step5_sameseason_backup
cp -r outputs/step5c outputs/step5c_sameseason_backup
# 3. çalıştırın:
python scripts/run_prefire_experiment.py
```

Mode seçimi fail-fast'tir: `VALIDATION_MODE` geçersiz bir değer alırsa veya
pre_fire'da predictor/label pencereleri çakışırsa (ve
`VALIDATION_ALLOW_OVERLAPPING_WINDOWS=False` ise) config/Step6 net bir hata verir;
sessizce same_season'a DÜŞMEZ. Step6 başlangıçta seçili modu ve pencereleri loglar
("Running Step6 validation mode: pre_fire" vb.).

Bu script Step3 -> Step4 -> Step4B -> Step5 -> Step5C -> Step6 sırasını predictor
window için çalıştırır. GEE erişimi ve auth gerektirir. Sonuç ilk pre-fire
burned-area association testidir; doğrulanmış yangın-riski modeli değildir.
Varsayılan örnek: predictor window `2023-06-01 -> 2023-07-31`, label window
`2023-08-01 -> 2023-10-31`, AOI `kozan_aoi`, label kaynağı MCD64A1 (FireCCI51
2023 kapsamı dışı olduğu için skip).

## Akış

1. Validation moduna göre predictor ve label pencereleri belirlenir.
2. Yanmış alan etiketleri label window için GEE'den çekilir.
3. Etiketler binary rasterlara çevrilir (burned=1, unburned=0) ve predictor
   grid'ine resample edilir (nearest).
4. NaN predictor pikselleri ve geçersiz etiketler dışlanır.
5. Sınıf dengesizliği için yanmayan pikseller alt-örneklenir; hem **full** hem
   **balanced** metrikler raporlanır.
6. Her predictor için: burned/unburned mean & median, ROC eğrisi, AUC.

LST anomaly predictor dosya adı esnek tanınır: `thermal_anomaly_zscore.tif`
veya `anomaly_zscore.tif` (ilk bulunan kullanılır).

## Çıktılar

* `outputs/validation/validation_summary.md`
* `outputs/validation/validation_stats.json`
* `outputs/validation/roc_curve_comparison.png`
* `outputs/validation/burned_vs_unburned_boxplot.png`
* `outputs/validation/predictor_maps_with_burn_overlay.png` (mümkünse)

## Önemli yorum notları

* `current_tvdi` ve `tvdi_difference` **sürekli kuruluk göstergeleri** olarak ele
  alınır.
* `tvdi_anomaly_zscore` güvenilirlik-filtreli ve yoğun maskelidir; geçerli örnek
  sayısı **ayrı raporlanır** ve sürekli predictor'lardan çok daha küçüktür. AUC
  karşılaştırmasında bu göz önünde bulundurulmalıdır.
* AUC > 0.5, predictor ile yanmış alan arasında pozitif ilişki olduğunu gösterir;
  bu **ön ilişki kanıtıdır**, doğrulanmış bir yangın-riski ürünü değildir.
* Seçili AOI/sezonda hiç yanmış piksel yoksa Step6 net bir hata verir ve AOI'yi
  genişletmeyi veya sezonu değiştirmeyi önerir.

## JSON çıktı formatı

`validation_stats.json` **kompakt özet formatındadır**. Full per-pixel array'ler
ve full ROC array'leri JSON'a YAZILMAZ — ROC eğrileri PNG olarak kaydedilir.
JSON'da her predictor için yalnız özet metrikler (valid_paired_pixels,
burned/unburned count, mean/median, auc_full, auc_balanced, source_file) ve
küçük bir downsample'lı `roc_curve_preview` (≤ `VALIDATION_MAX_ROC_PREVIEW_POINTS`
nokta) tutulur. Bu sayede JSON boyutu milyonlarca pikselde bile birkaç MB'ın
altında kalır.

## Yanmış alan etiket kaynakları

* **MCD64A1** ana label kaynağıdır (500 m, aylık).
* **FireCCI51** yalnız istenen label window dataset kapsamındaysa denenir.
  FireCCI51 yakın yıllarda (örn. 2023) veri döndürmediği için, label window
  kapsam dışındaysa Earth Engine'e SORULMADAN skip edilir ve summary'de
  "FireCCI51 skipped: requested label window outside dataset availability."
  olarak raporlanır. 2023 validation için MCD64A1 ana label kaynağıdır.
* Boş collection / bandsiz image kontrolleri (collection.size, bandNames().size)
  korunur; `.gt()` asla boş image üzerinde çağrılmaz.

## Diğer ileri adımlar (Phase 2+)

* **DEM** (SRTM/Copernicus): yükseklik + eğim hem değişken hem downscaling girdisi.
* **MODIS downscaling**: 1 km MODIS'in NDVI + DEM ile 30 m'ye indirgenmesi.
* **Kayan pencere**: "current state"in tek snapshot yerine sezon boyunca kayan
  pencere + sürekli doğrulama döngüsüne çevrilmesi.
* **RF/XGBoost**: çok-değişkenli yangın duyarlılık modeli (bu association testi
  olumlu sonuç verirse).