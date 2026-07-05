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

# Current pipeline status

## Completed / validated
- Landsat current-period and historical baseline LST products are generated for the Kozan AOI.
- Current year is excluded from the historical baseline to avoid leakage.
- Temporal interpolation was removed; insufficient observations are masked as NaN.
- NDVI export was fixed/re-exported after stale local artifacts were detected.
- Clean NDVI rasters now pass physical range validation: approximately [-1, 1].
- Step4B GeoTIFF validation is manifest/metadata-driven and catches invalid active products. Baseline NDVI files referenced by `outputs/step5c/step5c_metadata.json` (`inputs.baseline_ndvi_files`) are registered as active required products, not legacy.
- DEM elevation and slope products are generated and validated.
- Step7A tiled/windowed raster processing test passed with exact reconstruction: max_abs_difference = 0.0.
- TVDI products are generated with NDVI-binned wet/dry edges.
- MCD64A1 burned-area validation is implemented in pre-fire mode.
- FireCCI51 is skipped for 2023 because the requested period is outside dataset availability.
- Step7B-7E (MODIS downscaling training dataset, model training, raster
  prediction, and observation+downscaled fusion/gap-filling) are implemented.
  None of these train or validate a fire-risk model; MCD64A1/FIRMS labels are
  never used in Step7B-7E.

## Important validation result
- Clean NDVI changed the interpretation of the previous TVDI results.
- Single-index TVDI association with later burned area is weak in this AOI/year.
- In the clean pre-fire run, the strongest single-index TVDI result is only around NDVI 0.6-0.8 `tvdi_difference` AUC ≈ 0.522.
- LST anomaly remains weakly positive, around AUC ≈ 0.53-0.55 depending on population.
- Therefore the current scientific conclusion is **not** "TVDI alone predicts fire risk strongly."
- The correct conclusion is: the NDVI/TVDI pipeline is now technically valid, but single-index predictors are weak; the next step is multi-feature modeling/downscaling.
- This remains an **initial burned-area association experiment**, not a validated fire-risk model. No high-AUC claim is made.

## Next step
- Step7B-7E (MODIS downscaling training dataset, model training, raster
  prediction, observation+downscaled fusion) are implemented; see the
  Step7B/7C/7D/7E sections below.
- Step8A prepares a native ~500 m MCD64A1-grid modeling dataset (fixes the
  30 m label-duplication issue from resampling MCD64A1 onto the predictor
  grid); see the Step8A section below.
- Do not proceed to fire-risk RF/XGBoost (Step8B) until the Step8A dataset
  has been reviewed; class balancing/model training belongs to Step8B, not
  Step8A.

> 3D / Cesium / Jetson / mobile components are later demo/application layers, not current scientific validation results.

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
├── regions.py
├── validation_burned_area.py
└── utils/
    ├── geotiff_validation.py
    └── tiling.py

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
├── step6_validate_fire_relation.py
├── step7a_tiling_infrastructure.py
├── step7b_prepare_downscaling_dataset.py
├── step7c_train_downscaling_model.py
├── step7d_predict_downscaled_lst.py
├── step7e_fuse_landsat_downscaled_lst.py
├── step8a_prepare_500m_modeling_dataset.py
├── step8b_train_baseline_vs_thermal_model.py
├── step8c_spatial_block_bootstrap_uncertainty.py
├── step8d_thermal_feature_ablation.py
└── step8e_final_report.py

scripts/
├── main.py
├── export_mcd64a1_raw_burndate.py
├── run_prefire_experiment.py
└── standalone_step5-6.py

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

`main.py`, **Step1'den Step8E'ye kadar TÜM pipeline'ı** uçtan uca çalıştırır:
GEE erişimi gereken online kısım (Step1-7B: veri çekme, export/polling,
Landsat anomaly/TVDI, Step6 burned-area testi, downscaling eğitim verisi),
ardından yerel/offline devam (Step7C downscaling model eğitimi -> Step7D
downscaled LST tahmini -> Step7E füzyon -> raw MCD64A1 BurnDate export ->
Step8A native 500 m modelleme verisi -> Step8B baseline vs thermal
belirleyici deney -> Step8C spatial-block bootstrap -> Step8D termal
özellik ablation -> Step8E nihai birleşik rapor).

```bash
python scripts/main.py            # ilk çalıştırma
python scripts/main.py --force    # Step7C/7D/7E ve Step8A-8E çıktıları
                                   # zaten varsa üzerine yaz
```

**Hata toleransı:** Step6 (burned-area association testi) hata-toleranslıdır
— GEE/veri erişimi başarısız olursa yalnızca uyarı verir, pipeline'ın geri
kalanı etkilenmez (hiçbir sonraki adım Step6'nın çıktısına bağımlı değildir).
**Raw MCD64A1 BurnDate export ise ZORUNLUDUR ve hata-toleranslı DEĞİLDİR** —
başarısız olursa pipeline burada durur, çünkü Step8A geçerli DOY etiketi
olmadan (yalnızca binary maskeyle) doğru çalışamaz ve zaten kendi içinde net
hata ile durur.

**Üzerine yazma güvenliği:** Step7C/7D/7E ve Step8A-8E, önceki bir
çalıştırmadan kalan çıktılar zaten varsa varsayılan olarak **net hata ile
durur**. `python scripts/main.py --force` ile tüm bu adımlara `force=True`
iletilir.

**Runtime notu:** Step7C (model eğitimi) ve özellikle Step8D (popülasyon
başına 11 model x spatial-block CV) gerçek AOI verisiyle uzun sürebilir
(Step8D tek başına gerçek ~48k satırlık veride tahminen 15-40 dakika). Bu,
tüm zinciri tek komutla çalıştırmanın doğal maliyetidir.

Adımları ayrı ayrı, elle çalıştırmak isterseniz (örn. yalnızca belirli bir
adımı yeniden çalıştırmak için) aşağıdaki komutları da kullanabilirsiniz:

```bash
python src/step7c_train_downscaling_model.py
python src/step7d_predict_downscaled_lst.py
python src/step7e_fuse_landsat_downscaled_lst.py
# Step8A raw MCD64A1 BurnDate rasteri gerektirir (binary maske degil):
python scripts/export_mcd64a1_raw_burndate.py --also-binary
python src/step8a_prepare_500m_modeling_dataset.py --force
python src/step8b_train_baseline_vs_thermal_model.py --force
python src/step8c_spatial_block_bootstrap_uncertainty.py --force
python src/step8d_thermal_feature_ablation.py --force
python src/step8e_final_report.py --force
```

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

VALIDATION_MODE = "pre_fire"
PREDICTOR_START_DATE = "2023-06-01"
PREDICTOR_END_DATE = "2023-07-31"
LABEL_START_DATE = "2023-08-01"
LABEL_END_DATE = "2023-10-31"

CURRENT_PERIOD_DAYS = 60
CURRENT_PERIOD_END_DATE = "2023-07-31"

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
# veya ciktilar zaten varsa (Step7C/7D/7E, Step8A-8E) uzerine yazmak icin:
python scripts/main.py --force

```

Bu komut güncel akışta sırasıyla şunları yürütür:

* Step1: GEE bağlantısı ve temel veri sorguları
* Step2: MODIS baseline üretimi
* Step2B: DEM (elevation/slope) hazırlığı
* Step3: Landsat günlük composite ve current period hazırlığı
* Step4: Drive export ve task polling
* Step4b: Drive klasörü indirme ve dosya yerleştirme
* Step5: windowed/chunked raster ön işleme, Landsat anomaly ve MODIS bağlam üretimi
* Step5C: TVDI hesaplama
* Step5B: tanı raporu
* Step6: burned-area association testi (hata-toleranslı; GEE/veri yoksa akışın geri kalanını durdurmaz)
* Step7A: tiling/windowed altyapı testi
* Step7B: MODIS downscaling eğitim verisetinin hazırlanması
* Step7C: downscaling model eğitimi
* Step7D: Step7C modelinin tam raster gridine uygulanması (downscaled LST)
* Step7E: gözlemlenen + downscaled LST füzyonu/gap-filling
* Raw MCD64A1 BurnDate export (GEE, **zorunlu** — hata-toleranslı değil)
* Step8A: native 500 m MCD64A1-grid modelleme verisetinin hazırlanması
* Step8B: baseline vs baseline+thermal belirleyici deney (spatial-block CV)
* Step8C: Step8B delta metrikleri için spatial-block bootstrap belirsizlik analizi
* Step8D: termal özellik ablation çalışması
* Step8E: Step8A-8D'yi birleştiren nihai bilimsel özet rapor

`scripts/main.py` artık **Step8E'ye kadar** çalışır (önceki sürümlerde
Step7B'de duruyordu). Step6 dışındaki hiçbir adım atlanmaz; Step7C-7E ve
Step8A-8E önceki çıktılar zaten varsa `--force` verilmedikçe net hata ile
durur (bkz. yukarıdaki "main.py" bölümü).

## Adım Adım Çalıştırma

Adımları tek tek, elle çalıştırmak isterseniz (örneğin yalnızca belirli bir
adımı yeniden çalıştırmak için):

```bash
python src/step1_fetch_modis.py
python src/step2_modis_5year_mean.py
python src/step2b_dem.py
python src/step3_landsat_lst.py
python src/step4_export_geotiff.py
python src/step4b_download_drive_export.py
python src/step5_preprocess_timeseries.py
python src/step5c_tvdi.py
python src/step5b_diagnostic_report.py
python src/step6_validate_fire_relation.py
python src/step7a_tiling_infrastructure.py
python src/step7b_prepare_downscaling_dataset.py

# main.py'de de calisir, ama tek tek de calistirilabilir:
python src/step7c_train_downscaling_model.py
python src/step7d_predict_downscaled_lst.py
python src/step7e_fuse_landsat_downscaled_lst.py

# Step8A icin GERCEK raw MCD64A1 BurnDate rasteri gerekir (binary maske DEGIL).
# Step6 yalnizca binary mcd64a1_burned.tif ve (binary) mcd64a1_raw.tif uretir;
# raw DOY degerlerini asagidaki GEE script'i ile export edin (GEE gerekir):
python scripts/export_mcd64a1_raw_burndate.py --also-binary
python src/step8a_prepare_500m_modeling_dataset.py --force

# Step8B: belirleyici deney (baseline vs baseline+thermal, spatial-block CV):
python src/step8b_train_baseline_vs_thermal_model.py --force

# Step8C: Step8B delta metrikleri icin spatial-block bootstrap belirsizlik analizi:
python src/step8c_spatial_block_bootstrap_uncertainty.py --force

# Step8D: hangi termal ozellik/grubun Step8B iyilesmesini surukledigini bulan ablation:
python src/step8d_thermal_feature_ablation.py --force

# Step8E: Step8A-8D'yi tek bilimsel ozet raporunda birlestirir (egitim/hesaplama YOK):
python src/step8e_final_report.py --force

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

* **MCD64A1** — MODIS yanmış alan (500 m, aylık). **Birincil (primary) yanmış-alan etiketidir.**
* **FireCCI51** — ESA CCI FireCCI 5.1 yanmış alan (250 m, varsa). Label window dataset kapsamındaysa MCD64A1 ile birleştirilebilir.
* **FIRMS MODIS + VIIRS** — bağımsız aktif-yangın **cross-check** (opsiyonel, `VALIDATION_INCLUDE_FIRMS=True`).

**MCD64A1 is the primary burned-area label. FIRMS MODIS+VIIRS is an independent active-fire cross-check and is not OR-combined into the primary burned label.** `VALIDATION_INCLUDE_FIRMS=True` yalnızca FIRMS bağımsız cross-check'ini çalıştırır; birincil etiketi (ve dolayısıyla ana ROC/AUC, NDVI strata, popülasyon metrikleri, direction diagnostics) değiştirmez. FIRMS cross-check sonuçları `validation_stats.json` içinde ayrı `firms_crosscheck` anahtarında ve `validation_summary.md` içinde ayrı "FIRMS active-fire cross-check" bölümünde raporlanır. FIRMS koleksiyonları boş/erişilemezse ana MCD64A1 doğrulaması etkilenmez; cross-check `available=false` olarak işaretlenir.

## Validation modları

Step6 iki mod destekler (`VALIDATION_MODE`):

* **pre_fire (PRIMARY / birincil)** — predictor window ve label window AYRIDIR.
  Yangından önceki kuruluk sinyalinin, sonraki yanmış alanlarla ilişkisini test
  eder. Bu, projenin **birincil doğrulama modudur**. Varsayılan örnek: predictor
  window `2023-06-01 -> 2023-07-31`, label window `2023-08-01 -> 2023-10-31`.
* **same_season (SECONDARY / diagnostic)** — predictor rasterları ve yanmış alan
  etiketleri AYNI sezon penceresinden alınır. **İkincil / tanısal (diagnostic)**
  bir moddur; predictor ile label aynı dönemde çakıştığı için nedensel önceliği
  test etmez, yalnızca ilk association kontrolü sağlar.

### Raporlama önceliği (supervisor feedback)

Step6 raporu artık sonuçları bilimsel önemine göre sıralar:

* **Birincil değerlendirme alanı = burnable / vegetated strata.** Ana bölümler
  şu sırayı önceliklendirir: (1) land-cover burnable pikseller (varsa), (2)
  NDVI > 0.3 vejetasyon pikselleri, (3) NDVI-stratified validation.
* **All-pixel validation yalnızca DIAGNOSTIC'tir.** Tüm-piksel AUC'leri,
  burnable vejetasyonu non-burnable sıcak/kuru yüzeylerle karıştırdığı için
  başlık (headline) sonuç olarak KULLANILMAZ; "Diagnostic / confounding check"
  bölümüne taşınmıştır.
* **Anomaly z-score ürünleri** (`thermal_anomaly_zscore`, `tvdi_anomaly_zscore`)
  şu an `current_tvdi` ve `tvdi_difference`'tan **daha zayıftır** ve fazla
  iddialı (overclaim) sunulmamalıdır.
* TVDI **flip edilmez**; all-pixel inversiyonu bir population-mixing artefaktıdır.

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

## Current scientific interpretation

Projenin şu anki bilimsel hikayesi (overclaim olmadan):

* Hikaye **"global anomaly predicts fire" DEĞİLDİR.**
* Hikaye şudur: **burnable / vegetated alanlar içinde, mevcut kuruluk
  (`current_tvdi`) ve TVDI farkı (`tvdi_difference`), sonraki yanmış alanla
  (subsequent burned area) association gösterir.** En belirgin sinyal NDVI 0.6-0.8
  (yoğun vejetasyon) stratumundadır.
* All-pixel TVDI, burnable vejetasyonu non-burnable sıcak/kuru yüzeylerle
  karıştırır; bu yüzden `current_tvdi` global ölçekte **inverted** görünür. Bu bir
  population-mixing artefaktıdır, gerçek bir işaret tersine dönmesi değildir.
* **TVDI flip edilmez.** Düşük all-pixel AUC, ürün semantiğini değiştirmek için
  gerekçe değildir.
* **LST/TVDI anomaly z-score'ları hâlâ zayıftır** ve fazla iddialı
  sunulmamalıdır; `current_tvdi` ve `tvdi_difference`'tan daha düşük performans
  gösterirler.
* Tüm sonuçlar "candidate dryness indicator" / "first burned-area association
  test" seviyesindedir; doğrulanmış yangın-riski modeli değildir.

## Step4B GeoTIFF validation

Step4B, Drive export'larını indirip yerel `data/` klasörlerine yerleştirdikten
(veya dosyalar zaten varsa onları tespit ettikten) sonra her GeoTIFF için
bütünlük (integrity) doğrulaması çalıştırır. Doğrulama mantığı generic
`utils/geotiff_validation.py` modülündedir.

Yakalanan kritik (loud fail) durumlar: dosya yok, okunamayan GeoTIFF,
width/height = 0, CRS/transform yok, tüm pikseller NaN (`finite_count == 0`) ve
ürün-spesifik imkânsız aralık (örn. slope tüm-NaN veya slope max > 200). Bu
sayede daha önce yaşanan **all-NaN slope** rasterı gibi bir hata anında yakalanır
ve pipeline net bir hata ile durur; Step4B rasterı **onarmaz**, yalnızca doğrular
ve raporlar.

Kritik olmayan durumlar uyarı olarak raporlanır ama akış devam eder: nodata
değeri set edilmemiş, sabit raster, beklenenden düşük (ama sıfır olmayan)
finite yüzdesi, küçük extent farkı, eksik opsiyonel ürün.

Doğrulanan ürünler: Landsat LST, NDVI, MODIS, DEM (`data/dem/elevation.tif`,
`data/dem/slope.tif`) ve varsa land-cover. Raporlar:

```text
outputs/step4b/geotiff_validation_summary.json
outputs/step4b/geotiff_validation_summary.md
```

Step4b metadata'sına `geotiff_validation` bölümü eklenir (enabled, summary_path,
products_checked/passed/failed, warnings_count).

CLI bayrakları:

```bash
# yalnızca mevcut dosyaları doğrula (indirme yok) — slope fix sonrası kontrol için
python src/step4b_download_drive_export.py --validation-only
# indir ama doğrulama yapma
python src/step4b_download_drive_export.py --skip-validation
# warning'leri de failure say
python src/step4b_download_drive_export.py --strict-validation
```

## Step7A Tiling infrastructure

Step7A, büyük rasterların **pencere/tile-güvenli** (window-safe) işlenebildiğini
doğrular. Bu altyapı, Kozan AOI'den Doğu Akdeniz'e ölçeklenmeden ve MODIS
downscaling / gap filling'e geçilmeden önce gereklidir. Generic tiling
yardımcıları `utils/tiling.py` içindedir (proje-spesifik ürün hardcode edilmez).

Step7A **model eğitmez** ve **bilimsel ürün üretmez**. Yalnızca bir referans
rasterı (öncelik: DEM elevation → current LST → current TVDI → anomaly zscore)
pencere pencere okuyup tile dosyalarına yazar, tam rasterı yeniden birleştirir
(mosaic) ve orijinaliyle karşılaştırır. Rasterın tamamı belleğe alınmaz;
`rasterio` windows kullanılır.

Çıktılar (yalnızca `outputs/step7a/` altında):

```text
outputs/step7a/tiling_test_reconstructed.tif
outputs/step7a/tiling_test_difference.tif
outputs/step7a/tiling_test_summary.json
outputs/step7a/tiling_test_summary.md
```

Geçme kriteri: aynı CRS / transform / boyut / bounds ve max mutlak fark ≤ tolerans
(NaN/nodata eşitliği doğru ele alınarak). CLI bayrakları: `--reference-raster`,
`--tile-size`, `--overlap`, `--force`, `--tolerance`. Mevcut çıktılar yalnızca
`--force` ile ezilir.

## Step7B MODIS downscaling training dataset

Step7B, MODIS→Landsat LST downscaling için pencere/tile-bazlı bir **eğitim
veriseti** hazırlar. **Step7B model EĞİTMEZ ve fire-risk modeli ÜRETMEZ;** yalnızca
temiz, hizalanmış tabular örnekler üretir. Step5/Step5C/Step6 bilimsel çıktıları
değişmez. Fire-risk RF/XGBoost, MODIS downscaling'den ayrı tutulur.

Amaç (bilimsel): hedef = yüksek çözünürlüklü Landsat LST (Celsius), öznitelikler =
kaba MODIS context + yüksek çözünürlüklü yardımcı değişkenler.

Girdiler:
* **Target**: Landsat LST (`outputs/step5/current_period_median_celsius.tif`, yoksa
  `data/current_period/landsat_current_period_*days.tif` band 1) — `landsat_lst_celsius`.
* **MODIS context**: `data/modis/modis_lst_dogu_akdeniz_4y_summer_mean.tif` (ve varsa
  step5 resampled mean/std/zscore).
* **NDVI**: `data/ndvi_current_period/current_ndvi_median.tif`.
* **DEM**: `data/dem/elevation.tif`, `data/dem/slope.tif`.
* **Land cover** (varsa): `data/landcover/landcover_esa_worldcover_v200.tif` (nearest).
* **Koordinatlar**: lon, lat (+ opsiyonel modis_pixel_row/col/id).
* **Opsiyonel** (varsa; zorunlu değil): `current_tvdi`, `tvdi_difference`,
  `anomaly_zscore`.

Tüm öznitelikler target grid'ine hizalanır/yeniden örneklenir: sürekli rasterlar
bilinear, kategorik (land cover) ve binary maskeler nearest. İşlem `rasterio`
windows ile yapılır; rasterlar aynı anda RAM'e alınmaz. Değerler **clamp edilmez**;
geçersiz/NaN/aralık-dışı satırlar yalnızca **drop** edilir. Örnekleme deterministiktir
(`STEP7B_RANDOM_SEED`); `STEP7B_STRATIFY_BY_MODIS_PIXEL=True` iken tek bir kaba MODIS
pikselinden aşırı örnek alınmaz.

Çıktılar:

```text
outputs/step7b/downscaling_training_samples.parquet   (pyarrow varsa)
outputs/step7b/downscaling_training_samples.csv
outputs/step7b/downscaling_dataset_stats.json
outputs/step7b/downscaling_dataset_summary.md
```

pyarrow yoksa CSV yine yazılır ve metadata'da `parquet_written=false` kaydedilir;
en az bir format başarılı olduğu sürece hata verilmez. CLI: `--max-samples`,
`--tile-size`, `--output-format csv|parquet|both`, `--force`, `--no-optional-tvdi`,
`--no-optional-anomaly`. Mevcut çıktılar yalnız `--force` ile ezilir.

Step7B'den sonraki aşama, bu veriseti üzerinde MODIS downscaling için model
eğitimidir (Step7C / Step8). Fire-risk modellemesi bundan ayrı bir hattır.

## Step7C: MODIS→Landsat LST downscaling model eğitimi

Step7C, Step7B veri setini kullanarak **saf** bir MODIS→Landsat LST downscaling
modeli eğitir. **Step7C'de fire-risk modeli EĞİTİLMEZ; MCD64A1 veya FIRMS
etiketleri KULLANILMAZ.** Step5/Step5C/Step6/Step7B çıktıları değişmez.

**Amaç:** hedef = `landsat_lst_celsius` (Step7B'nin ürettiği yüksek çözünürlüklü
Landsat LST); girdi = MODIS context + NDVI + DEM + land cover + koordinatlar.

**Leakage koruması (kritik):** Target'tan türetilebilecek özellikler eğitim
setinden **hariç tutulur**: `anomaly_zscore`, `current_tvdi`, `tvdi_difference`,
`modis_context_zscore`. Bunlar mevcut Landsat LST'den (veya onu içeren
ürünlerden) hesaplanmış olabileceğinden, downscaling hedefini tahmin etmede
kullanılırsa metrikleri yapay şekilde şişirir (leakage). Bu dört özellik
metadata'da `excluded_leakage_features` altında kayıtlıdır ama eğitime girmez.

**Güvenli (leakage-free) özellikler:** `modis_lst_mean_celsius`,
`modis_lst_std_celsius`, `ndvi`, `elevation`, `slope`, `landcover`, `lon`, `lat`,
`row`, `col`, ve normalize `row_norm`/`col_norm`. Veri setinde bulunmayanlar
sessizce atlanır.

**Split:** Rastgele piksel split birincil doğrulama olarak KULLANILMAZ.
Birincil split (**varsayılan**) `spatial_block`'tur: örnekler `row//BLOCK`,
`col//BLOCK` ile `STEP7C_SPATIAL_BLOCK_SIZE_PIXELS` (varsayılan 64) piksellik
mekansal bloklara (`spatial_block_id`) gruplanır ve her blok tamamen tek bir
split'e (train/val/test) atanır — bu, AOI içindeki **görülmemiş mekansal
bölgelere** genelleme testidir, yalnızca görülmemiş piksellere değil.
`modis_pixel_group` (`modis_pixel_id`) ve `tile_group` (`source_tile_id`)
alternatif modlardır; `modis_pixel_id` Step7B çıktısında **örnek-başına
neredeyse benzersiz çıkabilir** (bu durumda gruplama etkisizdir). Step7C bunu
otomatik tespit eder: `grup_sayısı == örnek_sayısı` ise
*"Group split is ineffective because each sample is its own group."* uyarısı
loglanır ve metadata/summary'ye yazılır (`samples_per_group`: min/median/
mean/max ile birlikte). Grup kolonu kurulamazsa yalnızca `--allow-random-split`
ile rastgele piksel split'e düşülür (net uyarıyla). Gruplar deterministik seed
ile train/val/test = %70/%15/%15 oranında ayrılır. CLI: `--split
spatial_block|modis_pixel_group|tile_group|random`, `--spatial-block-size`.

**Model:** varsayılan `RandomForestRegressor` (sklearn-only, `n_estimators=200,
min_samples_leaf=2`). Opsiyonel `hist_gradient_boosting` (sklearn) ve
`xgboost` (yalnızca kuruluysa; requirements'a otomatik eklenmez, kurulu
değilse net hata verir). `--fast` ile küçük model (`n_estimators=50`) hızlı
test için.

Çıktılar:

```text
outputs/step7c/downscaling_model.joblib
outputs/step7c/downscaling_model_metadata.json
outputs/step7c/downscaling_model_metrics.json
outputs/step7c/downscaling_model_summary.md
outputs/step7c/feature_importance.csv
outputs/step7c/predicted_vs_actual.png
outputs/step7c/residual_histogram.png
outputs/step7c/residual_by_feature_summary.csv
outputs/step7c/per_split_predictions_sample.csv
```

Metrikler (train/val/test): RMSE, MAE, R², bias, medyan mutlak hata, residual
std, örnek sayısı. **MODIS baseline** (doğrudan `modis_lst_mean_celsius`'u
tahmin olarak kullanma) ve **train-mean baseline** ile karşılaştırılır;
MODIS baseline'a göre RMSE/MAE/R² iyileşmesi raporlanır — model baseline'dan
kötüyse bu **gizlenmez**, açıkça uyarı olarak yazılır. Train/test RMSE arasında
büyük fark varsa overfitting uyarısı; gruplu split'te grup sayısı azsa kararsız
split uyarısı verilir.

> `modis_lst_mean_celsius`, güncel günlük bir MODIS gözlemi değil, çok yıllık
> yaz-ortalaması context katmanıdır. Bu nedenle Step7C şu an **mekansal
> downscaling/context kalibrasyonu prototipi**dir, henüz günlük MODIS
> downscaling değildir.

CLI: `--model random_forest|hist_gradient_boosting|xgboost`, `--fast`,
`--max-train-samples`, `--force`, `--allow-random-split`, `--input`,
`--output-dir`. Mevcut çıktılar yalnız `--force` ile ezilir.

Sonraki aşama Step7D: raster tahmin/mozaik üretimi, gap filling ve MODIS
gerçek zamanlı füzyonu. Fire-risk RF/XGBoost, MODIS downscaling hattından
tamamen ayrı tutulur.

## Step7D: Downscaled LST raster tahmin üretimi

Step7D, eğitilmiş Step7C modelini **tam referans raster gridine** uygulayarak
30 m Landsat-benzeri downscaled LST GeoTIFF üretir. **Fire-risk modeli
DEĞİLDİR; MCD64A1 veya FIRMS etiketleri KULLANILMAZ.** Step5/Step5C/Step6/
Step7B/Step7C çıktıları değişmez; Step7C leakage guard'ına uyulur.

**Leakage koruması:** `anomaly_zscore`, `current_tvdi`, `tvdi_difference`,
`modis_context_zscore` rasterları **diskte mevcut olsalar bile hiç açılmaz**
(hard safety net — Step7C'nin `safe_feature_columns` listesinde bu özellikler
bulunursa Step7D işlemi tamamen durdurur).

**Süreç:** model + Step7C metadata yüklenir → leakage guard doğrulanır →
referans grid (`outputs/step5/current_period_median_celsius.tif`, fallback
current-period LST) ve gerekli feature rasterları çözülür → **tüm feature
rasterlarının referans gridle (width/height/CRS/transform) birebir eşleştiği**
doğrulanır (uyuşmazlıkta **sessizce resample edilmez**, net hata verir) →
pencere/tile bazlı (`rasterio` windows, bellek dostu) tahmin üretilir → aynı
kolon sırası (Step7C metadata'sındaki `safe_feature_columns`) ile model
`predict()` çağrılır.

**Çıktılar:**

```text
outputs/step7d/downscaled_lst_celsius.tif
outputs/step7d/downscaled_lst_valid_mask.tif
outputs/step7d/downscaling_prediction_metadata.json
outputs/step7d/downscaling_prediction_stats.json
outputs/step7d/downscaling_prediction_summary.md
outputs/step7d/downscaling_residual_observed_minus_predicted.tif  (gözlem varsa)
outputs/step7d/downscaling_absolute_error.tif                     (gözlem varsa)
```

Tahmin rasterı float32/NaN-nodata, aynı CRS/transform/boyut; geçerli maske
uint8 (1=geçerli, 0=eksik feature). Aralık dışı tahminler (`[-20, 80]` °C
dışı) **clamp edilmez**, yalnızca sayılır ve dürüstçe raporlanır.

**Önemli:** gözlem-örtüşme residual metrikleri (aynı current-period Landsat
LST rasterıyla karşılaştırma) **in-sample/current-window tanı amaçlıdır,
bağımsız doğrulama DEĞİLDİR.** Gerçek model doğrulama referansı Step7C'nin
**spatial_block** test metrikleridir (summary.md'de ayrıca raporlanır).
`modis_lst_mean_celsius` çok-yıllık yaz-ortalaması context katmanıdır, güncel
günlük MODIS gözlemi değildir — Step7D bu nedenle mekansal downscaling/context
kalibrasyonu prototipidir.

CLI: `--model`, `--model-metadata`, `--output-dir`, `--tile-size`, `--force`,
`--no-residual-products`, `--plot`. Mevcut çıktılar yalnız `--force` ile ezilir.

## Step7E: Gözlem + downscaled LST füzyonu (gap-filling)

Step7E, gözlemlenen Landsat current-period LST ile Step7D downscaled LST'yi
**füzyonlar** (birleştirir). **Model EĞİTİLMEZ; fire-risk modeli DEĞİLDİR;
MCD64A1/FIRMS etiketi KULLANILMAZ; bağımsız model doğrulaması DEĞİLDİR.**

**Füzyon kuralı (öncelik sırası, ortalama/blend YOK):**
1. Gözlemlenen Landsat LST geçerliyse (finite, `[-20,80]°C` içinde) →
   **her zaman** kullanılır (`source_mask=1`). Geçerli gözlem pikselleri
   **asla** model tahminiyle değiştirilmez.
2. Gözlem eksik/geçersizse VE Step7D downscaled LST geçerliyse (finite,
   `valid_mask==1`, aynı aralıkta) → yalnızca o zaman gap-fill olarak
   kullanılır (`source_mask=2`).
3. Her ikisi de geçersizse piksel NaN kalır (`source_mask=0`).

Çıktılar:

```text
outputs/step7e/fused_lst_celsius.tif
outputs/step7e/fused_lst_source_mask.tif        (0=invalid, 1=observed, 2=gap-fill)
outputs/step7e/fused_lst_gapfill_amount.tif      (yalnız gap-fill pikselinde değer)
outputs/step7e/fused_lst_metadata.json
outputs/step7e/fused_lst_stats.json
outputs/step7e/fused_lst_summary.md
```

Rapor: gözlem kapsamı, fused kapsamı, kapsam kazanımı, gap-fill piksel sayısı.
`fused_minus_observed_on_overlap` yalnızca bir **sağlamlık kontrolüdür**
(gözlem kullanılan piksellerde tanım gereği sıfır olmalı) — model doğrulama
metriği değildir. Step7D/Step7C limitasyonu (MODIS'in çok-yıllık yaz-ortalaması
context katmanı olduğu, henüz günlük MODIS gap-filling olmadığı) özetlenir.

CLI: `--observed`, `--downscaled`, `--downscaled-mask`, `--output-dir`,
`--tile-size`, `--force`, `--no-diagnostics`, `--plot`. Diğer Step7D/7C gibi,
grid uyumsuzluğunda Step7E **sessizce resample etmez**, net hata verir.

## Step8A: Native 500 m MCD64A1-grid modeling dataset

**Neden (label-resolution honesty):** MCD64A1 yanmış-alan etiketi native
~500 m çözünürlüktedir. Step6, bu etiketi Earth Engine'den doğrudan 30 m
ölçekte (`VALIDATION_LABEL_EXPORT_SCALE`) export ediyordu; bu her native
500 m yanmış hücreyi ~250-300 adet 30 m piksele **çoğaltır** ve piksel
sayılarına/olası confidence-p-value tarzı istatistiklere dayalı yorumları
yanıltıcı hale getirir (pseudo-replication). Step8A, 30 m predictor
rasterlarını MCD64A1'in native gridine agregat ederek bunu düzeltir.

**Step8A NE YAPMAZ:** model **eğitmez**, RF/XGBoost **çalıştırmaz**, nihai
fire-risk doğrulaması **yapmaz**. FIRMS hedef olarak **kullanılmaz**
(tamamen göz ardı edilir). MCD64A1 birincil yanmış-alan etiketi olarak
kalır. Step5/Step5C/Step6/Step7B/Step7C/Step7D/Step7E çıktıları
değiştirilmez (yalnızca salt-okunur girdi olarak kullanılır). TVDI formülü
ve FIRMS semantiği değişmez.

**Native 500 m grid — uygulama notu:** Repoda gerçek native-CRS'li 500 m
MCD64A1 rasteri yerelde saklanmaz (Step6 BurnDate'i GEE'den doğrudan 30 m
ölçekte indirir, bu da native pikseli zaten 30 m'ye çoğaltılmış olarak
export eder). Step8A, native gridi, referans 30 m grid üzerinde
`round(STEP8A_MCD64A1_NATIVE_CELL_SIZE_M / STEP8A_REFERENCE_PIXEL_SIZE_M)`
(varsayılan `round(500/30) = 17`) piksellik kare bloklara ayırarak yaklaşık
olarak yeniden oluşturur ve her bloğun BurnDate etiketini **mode**
(çoğunluk değer) ile tek bir değere indirger — 30 m alt-pikseller bağımsız
örnek olarak sayılmaz. Bu, gerçek MODIS sinüzoidal native gridin birebir
aynısı değildir, ama depoda halihazırda mevcut veriyle duplikasyon
sorununu doğrudan düzeltir.

**MCD64A1 etiket rasteri keşfi (raw tercihli, robust):** Step8A **raw
BurnDate** (gün-of-year değerleri) rasterini güçlü şekilde tercih eder; binary
yanmış maske yalnızca son çare fallback'tir. Sıra: `outputs/validation/labels/
mcd64a1_raw.tif`, `outputs/step6/mcd64a1_raw.tif`, `outputs/**/*mcd64*raw*.tif`
(raw), ardından SON ÇARE olarak binary maske (`mcd64a1_burned.tif` vb.).
`--label-raster` ile açık yol verilebilir. Hiçbiri yoksa net hata ile durur.

**Etiket rasteri denetimi ve fail-fast (ÖNEMLİ):** Agregasyondan ÖNCE seçilen
etiket rasteri incelenir (dtype, nodata, min/max, benzersiz değerler, 0/1/DOY
sayımları — stats JSON'da `label_raster_diagnostics`). "Raw" olduğu iddia
edilen bir raster aslında **binary** ise (yalnızca {0,1} değerleri, ya da tüm
pozitifler 1.0, ya da Ağustos-Ekim DOY aralığında [213-304] hiç değer yok),
Step8A **sıfır-yanmış geçersiz veri seti üretmek yerine** şu hatayla durur:
`"Selected MCD64A1 raw raster does not contain BurnDate DOY values. It appears
to be binary. Re-export raw MCD64A1 BurnDate."`

**DİKKAT — Step6 binary maske üretir:** Step6'nın `export_label_to_grid()`
fonksiyonu `mosaic.gt(0)` (binary) yazar; `_raw` son eki binary image'in ham
*indirmesini* ifade eder, DOY değerlerini değil. Bu yüzden Step8A için gerçek
raw BurnDate'i `scripts/export_mcd64a1_raw_burndate.py` ile ayrıca export edin
(MODIS/061/MCD64A1 `BurnDate` bandı, `BurnDate.gt(0)` DEĞİL):

```bash
python scripts/export_mcd64a1_raw_burndate.py --also-binary
```

Bu script raw DOY rasterini `outputs/validation/labels/mcd64a1_raw.tif` (ve
`--also-binary` ile binary maskeyi `mcd64a1_burned.tif`) olarak yazar ve
export sonrası değerlerin gerçekten DOY olduğunu (yalnızca {0,1} değil) kontrol
eder.

**Etiket kuralı ve modelleme geçerliliği:** Bir 500 m hücre `burned=1` yalnızca
raw BurnDate'i Ağustos 1 - Ekim 31 penceresine düşerse; aksi halde `burned=0`
(BurnDate 0/NaN/masked/pencere-dışı hepsi unburned). Pencere dışı pozitif
BurnDate hücreleri `out_of_window_burndate_cells` olarak sayılır ama bu label
penceresi için unburned kalır. **Modelleme geçerliliği (`valid_for_modeling`)
yalnızca predictor + landcover + baseline özellik uygunluğuna dayanır; ETİKETE
BAĞLI DEĞİLDİR** — unburned hücreler (negatif sınıf) ve tamamen-nodata etiket
blokları veri setinde KALIR, düşürülmez.

**Agregasyon:** Her satır bir native ~500 m hücredir. NDVI, DEM
(elevation/slope), LST anomaly, current TVDI, TVDI difference, Step7D
downscaled LST, Step7E fused LST (+ `fused_lst_source_mask`) her hücre
içindeki 30 m piksellerden mean/median/std/valid_count/valid_fraction ile
özetlenir. Landcover (ESA WorldCover), Step7D ile **aynı** şekilde yalnızca
kategorik istisna olarak nearest-neighbor ile referans gride hizalanır;
başka hiçbir raster **sessizce resample edilmez** — uyumsuzlukta net hata
verilir. Opsiyonel rasterlar (ör. Step7E henüz çalıştırılmamışsa
`fused_lst`) eksikse ilgili sütunlar NaN bırakılır, satırlar **düşürülmez**;
zorunlu rasterlar (NDVI/elevation/slope/referans grid) eksikse net hata ile
durur.

**Burnable maskeler (supervisor talebi):** `burnable_tree_shrub_grass`
(tree cover + shrubland + grassland) ve `burnable_tree_shrub` (tree cover +
shrubland), `STEP8A_BURNABLE_FRACTION_THRESHOLD` (varsayılan 0.5) eşiğiyle.
**Cropland bu maskelerin dışında tutulur**, yalnızca kendi fraction'ı
(`landcover_cropland_fraction`) raporlanır.

**Çıktılar:**

```text
outputs/step8a/step8a_500m_modeling_dataset.parquet
outputs/step8a/step8a_500m_modeling_dataset.csv
outputs/step8a/step8a_dataset_stats.json
outputs/step8a/step8a_dataset_summary.md
outputs/step8a/step8a_500m_grid_burned_label.tif   (diagnostik, 500 m çözünürlük)
outputs/step8a/step8a_500m_grid_valid_mask.tif     (diagnostik, 500 m çözünürlük)
outputs/step8a/step8a_500m_cell_preview.geojson    (diagnostik, ilk 5000 hücre)
outputs/step8a/step8a_label_raster_diagnostics.json (yalnızca etiket denetimi FAIL-FAST ile durursa yazılır)
```

Stats JSON ayrıca `label_raster_diagnostics` (etiket rasterinin dtype/nodata/
min/max/benzersiz değerler/0-1-DOY sayımları), `label_kind`,
`label_source_description`, `burn_month_available` ve
`out_of_window_burndate_cells` alanlarını içerir.

Sütunlar arasında: `cell_id`, `row_500m`, `col_500m`, `lon`, `lat`, `burned`,
`burn_date`, `burn_month`, `burn_day_of_year`, `label_source`,
baseline özellikler (`ndvi_mean`, `elevation_mean`, `slope_mean`,
`landcover_dominant`, landcover fraction'ları), thermal özellikler
(`lst_anomaly_mean`, `current_lst_mean`, `current_tvdi_mean`,
`tvdi_difference_mean`, `downscaled_lst_mean`, `fused_lst_mean`), coverage/
provenance (`valid_30m_pixel_count`, `valid_30m_fraction`,
`observed_fraction`, `gapfilled_fraction`, `invalid_source_fraction`,
`source_mask_majority`) ve `valid_for_modeling`/`invalid_reason`/
`out_of_window_burndate` (Step8B'nin filtreleme yapabilmesi için; geçersiz
veya unburned hücreler dataset'ten **silinmez**, işaretlenir). Sınıf
dengeleme/undersampling burada **yapılmaz** — bu Step8B'ye aittir.

CLI: `--output-dir`, `--force`, `--write-csv`/`--no-write-csv`,
`--write-parquet`/`--no-write-parquet`, `--min-valid-fraction`,
`--burnable-threshold`, `--label-raster`, `--reference-30m`,
`--allow-all-burned` (yalnızca gerçekten çoğunluk-yanmış / düşük-kapsama bir
veri seti bekleniyorsa fail-fast kontrollerini gevşetir). Mevcut çıktılar
yalnız `--force` ile ezilir.

Sonraki aşama Step8B: bu dataset üzerinden **baseline model**
(elevation + slope + landcover + NDVI) ile **thermal model**
(baseline + current TVDI + LST anomaly + TVDI difference + fused thermal)
karşılaştırması; lead-time değerlendirmesi `burn_month` (Ağustos/Eylül/
Ekim) stratalarıyla yapılacaktır.

## Step8B: Baseline vs Baseline+Thermal Burned-Area Modeling (belirleyici deney)

**Bu, süpervizörün istediği BELİRLEYİCİ deneydir:** Termal/kuraklık
özellikleri, baseline (termal olmayan) özelliklerin ötesinde yanmış-alan
ayrımını iyileştiriyor mu? Step8A'nın 500 m MCD64A1-grid veri seti üzerinde,
**aynı spatial-block CV ve aynı popülasyon altında**, **Model A (baseline:**
elevation + slope + landcover + NDVI**)** ile **Model B (baseline+thermal:**
Model A + current TVDI + LST anomaly + TVDI difference + downscaled/fused
LST**)** karşılaştırılır. Ana sonuç: `delta_auc = AUC(Model B) - AUC(Model A)`.

**Kritik kısıtlar:**
- Örnekler Step8A'nın 500 m hücreleridir; **30 m piksel örnek olarak
  KULLANILMAZ.**
- Hedef yalnızca MCD64A1 (`burned`); **FIRMS hedef olarak KULLANILMAZ.**
- **Spatial-block CV** (`StratifiedGroupKFold`, `groups=spatial_block_id`)
  zorunludur; **random split ASLA kullanılmaz.** `spatial_block_id`,
  `row_500m`/`col_500m`'in `STEP8B_SPATIAL_BLOCK_SIZE_CELLS` (varsayılan 2x2
  hücre, ~1 km x 1 km) ile bloklanmasından türetilir.
- `burn_month`, `burn_date`, `burned`, `label_source` ve diğer etiket/metadata
  kolonları; `source_mask_majority`/`observed_fraction`/`gapfilled_fraction`/
  `invalid_source_fraction` (yalnızca duyarlılık tanısında kullanılır);
  `lon`/`lat` (yalnızca spatial block oluşturmak için, varsayılan olarak
  özellik DEĞİLDİR) — hiçbiri özellik setine girmez. Bu, çalışma zamanında
  `check_no_forbidden_features()` ile de doğrulanır.
- Eksik değerler pipeline İÇİNDE (`SimpleImputer`: sayısal=medyan,
  kategorik=en sık) impute edilir; `landcover_dominant` `OneHotEncoder
  (handle_unknown="ignore")` ile, tamamı `ColumnTransformer`/`Pipeline`
  içinde (fold'lar arası sızıntı yok).
- Varsayılan model: `RandomForestClassifier(n_estimators=300,
  min_samples_leaf=3, class_weight="balanced", random_state=42)`.
  `--model hist_gradient_boosting` da desteklenir; `--model xgboost`
  yalnızca xgboost kuruluysa çalışır (zorunlu bağımlılık değildir).

**Popülasyonlar:**
1. `all_valid` — **birincil analiz** (cropland-hariç burnable maskesinde
   yeterli pozitif olmadığı için).
2. `cropland_dominant` — yanmış hücrelerin büyük çoğunluğu cropland-dominant
   olduğu için ayrıca değerlendirilir.
3. `burnable_tree_shrub_grass`, 4. `burnable_tree_shrub` — **yalnızca
   tanı/duyarlılık amaçlı**; pozitif sayısı `STEP8B_MIN_POSITIVES_PER_
   POPULATION` (varsayılan 30) altındaysa **varsayılan olarak atlanır**
   (`--allow-low-positive-strata` ile zorlanabilir, ama alt sınır yine de
   en az 2 pozitif/negatif gerektirir).

**Lead-time (aylık) değerlendirme:** Ayrı aylık modeller **eğitilmez**; tek
Ağustos-Ekim modelinin out-of-fold tahminleri `burn_month` ile Ağustos/
Eylül/Ekim stratalarına bölünerek değerlendirilir (ay-vs-unburned).
Bir ayda pozitif sayısı `STEP8B_MIN_MONTH_POSITIVES` (varsayılan 10)
altındaysa metrik `null` ve uyarı raporlanır.

**Gap-fill duyarlılık tanısı:** `all_valid` ve `cropland_dominant` için,
**yeniden eğitim yapılmadan**, mevcut out-of-fold tahminleri
`gapfilled_fraction < 0.25` ve `< 0.50` alt kümelerinde yeniden değerlendirilir.

**İstatistiksel anlamlılık:** Bu çalıştırmada delta_auc/delta_pr_auc için
**hiçbir güven aralığı veya p-değeri hesaplanmaz** — sonuçlar yalnızca nokta
tahminidir; stats JSON ve summary bunu açıkça belirtir.

**Çıktılar:**

```text
outputs/step8b/step8b_model_comparison_metrics.json
outputs/step8b/step8b_fold_metrics.csv
outputs/step8b/step8b_predictions.parquet
outputs/step8b/step8b_predictions.csv
outputs/step8b/step8b_feature_importance.csv
outputs/step8b/step8b_summary.md
outputs/step8b/step8b_roc_curves.png              (opsiyonel)
outputs/step8b/step8b_pr_curves.png               (opsiyonel)
outputs/step8b/step8b_delta_auc_by_population.csv (opsiyonel)
outputs/step8b/step8b_monthly_leadtime_metrics.csv (opsiyonel)
```

CLI: `--input`, `--output-dir`, `--force`, `--n-splits` (varsayılan 5, geçerli
fold üretilemezse otomatik 3'e düşer, yine olmazsa net hata ile durur),
`--spatial-block-size-cells`, `--min-positives`, `--min-month-positives`,
`--allow-low-positive-strata`, `--model {random_forest,
hist_gradient_boosting, xgboost}`.

Kalite kontrolleri: girdi Step8A veri seti yoksa, `burned` kolonu yoksa,
`all_valid` icin `burned` tek sınıf içeriyorsa, birincil popülasyon için
spatial-block CV kurulamıyorsa, veya yasak bir etiket/metadata kolonu
özellik setine sızmışsa **net hata ile durur** (random split'e ASLA
düşülmez). `cropland_dominant`'ın neredeyse tüm pozitifleri içermesi,
burnable maskelerinin 30'un altında pozitif içermesi, Ekim pozitiflerinin
düşük olması, veya gapfill kolonlarının eksik olması durumlarında uyarı
verir.

## Step8C: Spatial-Block Bootstrap Uncertainty (Step8B belirsizlik analizi)

Step8B yalnızca nokta tahmini (delta_auc, delta_pr_auc) raporlar; güven
aralığı veya p-değeri yoktur. Step8C, **yeni model eğitmeden**, Step8B'nin
mevcut out-of-fold tahminlerini (`outputs/step8b/step8b_predictions.parquet`)
kullanarak bu nokta tahminlerine **spatial-block bootstrap** ile
belirsizlik/duyarlılık analizi ekler.

**Kritik kısıtlar:**
- **Model eğitilmez.** Yalnızca Step8B'nin mevcut tahminleri kullanılır.
- **Bootstrap birimi satır DEĞİL, `spatial_block_id`'dir** (Step8B'nin CV
  gruplarıyla aynı 500 m mekansal bloklar). Her iterasyonda benzersiz
  bloklar yerine koyarak örneklenir; bir blok birden fazla çekilirse TÜM
  satırları o kadar tekrarlanır. **Random satır bootstrap ASLA
  kullanılmaz.**
- `spatial_block_id` tahmin tablosunda yoksa, Step8A veri setinden
  (`row_500m`/`col_500m`, `cell_id` üzerinden join) yeniden oluşturulur;
  bu da mümkün değilse net hata ile durur.
- **FIRMS kullanılmaz; 30 m piksel kullanılmaz** (örnekler Step8B'den
  değişmeden gelen Step8A 500 m hücreleridir).
- **Klasik p-değeri veya "istatistiksel olarak anlamlı" iddiası YAPILMAZ.**
  Yalnızca %95 bootstrap yüzdelik aralığının sıfırı dışlayıp dışlamadığı
  raporlanır: `"bootstrap-supported positive delta"` (aralık tamamen >0)
  veya `"point estimate positive but CI overlaps zero"` (aralık sıfırı
  kapsıyor). Bu ayrım stats JSON, özet ve kodun kendisinde tutarlı şekilde
  vurgulanır.

**Popülasyonlar:** `all_valid` ve `cropland_dominant` her zaman
değerlendirilir. `burnable_tree_shrub_grass`/`burnable_tree_shrub` yalnızca
tahmin tablosunda mevcutsa VE pozitif sayısı `STEP8C_MIN_POSITIVES`
(varsayılan 30) üzerindeyse dahil edilir — Step8B'de zaten atlanmış
popülasyonlar Step8C'de de değerlendirilmez.

**Aylık lead-time:** Ayrı aylık model eğitilmez; mevcut tam-sezon OOF
tahminleri `burn_month`'a göre filtrelenip bootstrap'lanır. Bir ayda pozitif
< `STEP8C_MIN_MONTH_POSITIVES` (varsayılan 10) ise o ay için CI
`unavailable` olarak işaretlenir (Ekim'de tipik olarak beklenir).

**Gap-fill duyarlılığı:** `all_valid` ve `cropland_dominant` için
`no_filter`, `gapfilled_fraction < 0.25`, `< 0.50` alt kümelerinde **yeniden
eğitim yapılmadan** bootstrap tekrarlanır.

**Çıktılar:**

```text
outputs/step8c/step8c_bootstrap_metrics.json
outputs/step8c/step8c_bootstrap_samples.csv
outputs/step8c/step8c_summary.md
outputs/step8c/step8c_delta_auc_distribution.png    (opsiyonel)
outputs/step8c/step8c_delta_pr_auc_distribution.png (opsiyonel)
outputs/step8c/step8c_monthly_bootstrap_metrics.csv (opsiyonel)
```

CLI: `--input`, `--output-dir`, `--force`, `--n-bootstrap` (varsayılan 1000),
`--random-seed`, `--min-positives`, `--min-month-positives`.

Kalite kontrolleri: girdi tahmin dosyası yoksa, `burned`/`y_prob_baseline`/
`y_prob_thermal`/`population` kolonları yoksa, `spatial_block_id` yok ve
yeniden oluşturulamıyorsa, `all_valid` tahmin tablosunda yoksa veya tek
sınıf içeriyorsa **net hata ile durur**. `cropland_dominant`'ın neredeyse
tüm pozitifleri içermesi, Ekim pozitiflerinin düşük olması, `gapfilled_
fraction` eksikliği (bu durumda gap-fill duyarlılığı atlanır), veya
`fold_id` eksikliği (tahminlerin gerçekten out-of-fold olduğu
doğrulanamıyor) durumlarında uyarı verir.

## Step8D: Thermal Feature Ablation (Step8B'nin iyileşmesinin kaynağı)

Step8B, termal özelliklerin baseline'ı iyileştirdiğini gösterdi ama HANGİ
termal özelliğin/grubun bu iyileşmeyi sağladığını söylemiyor. Step8D,
Step8B ile **aynı spatial-block CV**'yi kullanarak, her popülasyon için
`baseline` + 10 termal özellik grubu/tekil özelliği (toplam 11 model)
eğitip, her grubu baseline'a karşı `delta_pr_auc`/`delta_auc`/`delta_brier`
ile karşılaştırır.

**Termal ablation grupları (baseline'a eklenir):**
`lst_anomaly_only`, `current_lst_only`, `tvdi_only`, `tvdi_difference_only`,
`downscaled_only`, `fused_lst_only` (tekil özellikler); `lst_anomaly_group`
(anomaly+current_lst), `tvdi_group` (tvdi+tvdi_difference),
`fused_downscaled_group` (downscaled+fused); `all_thermal` (Step8B'nin
thermal modeliyle birebir aynı özellik seti — çapraz kontrol için).

**Kritik kısıtlar:** Step8B ile birebir aynı — MCD64A1 tek hedef, FIRMS
kullanılmaz, 30 m piksel kullanılmaz, spatial-block CV zorunlu (random
split'e asla düşülmez), yasak etiket/provenance kolonları özellik setine
giremez (`check_no_forbidden_features()` ile doğrulanır).

**Sıralama:** Öncelikle `delta_pr_auc`'a göre (burned nadir olduğu için
PR-AUC daha duyarlı), ikincil olarak `delta_auc`'a göre. Her popülasyon
için her 11 model **AYNI CV fold'larıyla** eğitilir, böylece karşılaştırma
adil olur.

**Popülasyonlar:** `all_valid` birincil, `cropland_dominant` önemli
ikincil; `burnable_tree_shrub_grass`/`burnable_tree_shrub` yalnızca tanı
amaçlı, pozitif < 30 ise atlanır (Step8A/8B ile aynı eşik).

**Aylık lead-time:** Ayrı aylık model eğitilmez; her ablation modelinin
mevcut OOF tahminleri `burn_month`'a göre filtrelenip baseline'a karşı
değerlendirilir.

**Bootstrap (opsiyonel, varsayılan KAPALI):** `--bootstrap` bayrağıyla,
yalnızca her popülasyonun en iyi top-K (varsayılan 3) ablation grubu için
spatial-block bootstrap CI hesaplanır (yavaş olabileceği için varsayılan
kapalı; `--bootstrap` verilmezse çıktı yalnızca **nokta tahminidir**, özet
bunu açıkça belirtir).

**Step8B ile çapraz kontrol:** `all_thermal` sonucu, mevcutsa
`outputs/step8b/step8b_model_comparison_metrics.json`'daki Step8B'nin
kendi thermal modeliyle karşılaştırılır; `delta_auc` farkı 0.01'i aşarsa
uyarı verilir (iki script arasında konfigürasyon kayması/rastgelelik
sinyali).

**Çıktılar:**

```text
outputs/step8d/step8d_ablation_metrics.json
outputs/step8d/step8d_ablation_fold_metrics.csv
outputs/step8d/step8d_ablation_predictions.parquet
outputs/step8d/step8d_ablation_predictions.csv
outputs/step8d/step8d_ablation_feature_importance.csv
outputs/step8d/step8d_ablation_summary.md
outputs/step8d/step8d_ablation_barplot.png                    (opsiyonel)
outputs/step8d/step8d_ablation_delta_auc_by_population.csv    (opsiyonel)
outputs/step8d/step8d_ablation_delta_pr_auc_by_population.csv (opsiyonel)
```

Tahmin tablosu **uzun formattadır** (her hücre × popülasyon × model_name
için bir satır; `y_prob` tek kolon, `model_name`/`ablation_group` ile
hangi model olduğu ayırt edilir) — Step8B'nin geniş formatından (y_prob_
baseline/y_prob_thermal) farklıdır, çünkü burada 11 model karşılaştırılır.

CLI: `--input`, `--output-dir`, `--force`, `--n-splits`, `--spatial-block-
size-cells`, `--min-positives`, `--min-month-positives`, `--model
{random_forest,hist_gradient_boosting}`, `--n-estimators`, `--bootstrap`,
`--n-bootstrap`, `--top-k-bootstrap`, `--random-seed`.

Özet, her popülasyon için şunları açıkça belirtir: en iyi tekil termal
özellik, en iyi termal grup, `all_thermal`'ın daha küçük gruplardan daha
mı iyi olduğu, ve fused/downscaled özelliklerin daha basit LST/TVDI
gruplarının ötesinde katkı sağlayıp sağlamadığı.

## Step8E: Final Burned-Area Modeling Report (Step8A-8D özet paketi)

Step8E, **hiçbir model eğitmez, hiçbir önceki çıktıyı değiştirmez ve hiçbir
istatistiği yeniden hesaplamaz** — yalnızca Step8A/8B/8C/8D'nin stats
JSON'larını okuyup tek bir tutarlı bilimsel özet raporuna birleştirir.
Bir grup diğerinden büyük/küçük diye cümle kurmak gibi salt biçimlendirme
amaçlı karşılaştırmalar dışında hiçbir yeni analiz yapılmaz.

**Girdi (salt okunur):**
- `outputs/step8a/step8a_dataset_stats.json` (**zorunlu**)
- `outputs/step8b/step8b_model_comparison_metrics.json` (**zorunlu**)
- `outputs/step8c/step8c_bootstrap_metrics.json` (opsiyonel — yoksa
  bootstrap bölümü raporda "not available" olarak işaretlenir, çalışma
  durmaz)
- `outputs/step8d/step8d_ablation_metrics.json` (opsiyonel — yoksa ablation
  bölümü "not available" olarak işaretlenir)

**Rapor yapısı:** (1) Step8A-8D pipeline özeti, (2) dataset istatistikleri,
(3) Step8B model karşılaştırma tablosu, (4) Step8C bootstrap özeti (%95 CI +
yorumlama), (5) Step8D ablation sıralaması + en iyi tekil özellik/grup +
TVDI/downscaled/fusion katkı yorumu, (6) aylık lead-time tablosu (Ağustos/
Eylül/Ekim + güven düzeyi), (7) ana bulgular, (8) sınırlamalar, (9) sonraki
adımlar, ve genel sonuç paragrafı. Tüm anlatı cümleleri (en iyi grup hangisi,
`all_thermal` daha mı iyi, fusion katkı sağlıyor mu, genel sonuç metni)
**gerçek yüklenen sayılara göre dinamik olarak üretilir** — sabit şablon
metni değildir; Step8C/8D mevcut değilse ilgili cümleler otomatik olarak
atlanır/"not available" olarak işaretlenir.

**Çıktılar:**

```text
outputs/step8e/step8e_summary.md
outputs/step8e/step8e_summary.json
outputs/step8e/step8e_results_tables.xlsx
outputs/step8e/step8e_key_findings.csv
outputs/step8e/step8e_feature_ranking.csv      (opsiyonel)
outputs/step8e/step8e_monthly_results.csv      (opsiyonel)
outputs/step8e/step8e_population_summary.csv   (opsiyonel)
```

Excel çalışma kitabı (`step8e_results_tables.xlsx`) sayfaları: `dataset`,
`populations`, `model_comparison`, `bootstrap`, `ablation`, `monthly`,
`limitations` (profesyonel Arial font, koyu mavi başlık dolgusu, donmuş
başlık satırı). Değerler doğrudan yazılır (formül değil), çünkü bu bir
finansal model değil, önceden hesaplanmış Step8A-8D sonuçlarının statik bir
anlık görüntüsüdür — "yeniden hesaplama yok, yalnızca birleştirme" ilkesiyle
tutarlıdır.

CLI: `--output-dir`, `--force`.

Kalite kontrolleri: `outputs/step8a/step8a_dataset_stats.json` veya
`outputs/step8b/step8b_model_comparison_metrics.json` yoksa **net hata ile
durur**; Step8C/8D çıktıları eksikse yalnızca uyarır ve ilgili bölümü
"not available" işaretler.

## Diğer ileri adımlar (Phase 2+)

* **DEM** (SRTM/Copernicus): yükseklik + eğim hem değişken hem downscaling girdisi.
* **MODIS downscaling**: 1 km MODIS'in NDVI + DEM ile 30 m'ye indirgenmesi.
* **Kayan pencere**: "current state"in tek snapshot yerine sezon boyunca kayan
  pencere + sürekli doğrulama döngüsüne çevrilmesi.
* **RF/XGBoost**: çok-değişkenli yangın duyarlılık modeli (bu association testi
  olumlu sonuç verirse).