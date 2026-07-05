# Uydu Tabanlı Termal Veri İşleme ve Burned-Area Modelleme Prototipi

## 1. Kısa Özet

Bu proje, Doğu Akdeniz bölgesindeki Kozan AOI üzerinde MODIS, Landsat, NDVI, DEM (elevation/slope), landcover ve MCD64A1 burned-area verilerini işleyen çok adımlı bir uydu tabanlı termal veri işleme pipeline'ıdır. Pipeline, Google Earth Engine (GEE) üzerinden veri export/download işlemleriyle başlar; rasterio tabanlı local preprocessing, GeoTIFF doğrulama, termal anomaly ve dryness ürünleri üretimi ile devam eder; sonunda label-honest bir modeling/validation aşamasında sonlanır. Yani proje yalnızca raster üretimi yapan bir veri işleme hattı değil, aynı zamanda kendi ürettiği verileri istatistiksel olarak test eden bir doğrulama altyapısıdır.

Projenin bilimsel amacı nettir: termal ve dryness kökenli feature'ların (LST anomaly, TVDI, downscaled/fused LST) statik baseline feature'lara (elevation, slope, landcover, NDVI) kıyasla burned-area discrimination'a ölçülebilir bir katkı sağlayıp sağlamadığını test etmek. Bu soruyu cevaplamak için pipeline, Step1'den Step8E'ye kadar ilerleyen ve her adımda bir öncekinin çıktısını doğrulayan bir zincir şeklinde kurgulanmıştır.

Güncel ana bilimsel çıktı Step8 çekirdek deneyidir: MCD64A1'in native ~500 m grid'i üzerinde, thermal-augmented model statik baseline'a göre ölçülebilir ve spatial-block-bootstrap ile desteklenen pozitif bir performans farkı göstermiştir. Bu sonuç, Step8B (model karşılaştırması), Step8C (bootstrap belirsizlik analizi) ve Step8D (ablation) adımlarının birlikte değerlendirilmesiyle elde edilmiştir.

Bunun ötesinde, bu çalışmanın ne olduğu konusunda net olmak önemlidir: bu proje tamamlanmış bir operasyonel yangın erken uyarı sistemi değildir ve tamamlanmış bir 3B dijital ikiz de değildir. Şu anki hâliyle proje; termal preprocessing, MODIS→Landsat downscaling/fusion ve tek-AOI/tek-sezon kapsamında label-honest bir burned-area modeling altyapısıdır. Sonraki adımlar (bkz. Bölüm 14) bu altyapının genelleştirilmesine yöneliktir.

## 2. Bu Proje Ne Değildir?

- **Tamamlanmış bir 3B dijital ikiz değildir.** Şu an yalnızca 2B raster katmanları ve tablo bazlı bir modeling dataset'i üretilmektedir; 3B görselleştirme veya simülasyon katmanı yoktur.
- **Operasyonel bir yangın erken uyarı sistemi değildir.** Henüz multi-year, second-AOI ve near-real-time üretim akışı yoktur; pipeline tek bir AOI ve tek bir sezon için çalıştırılmıştır.
- **Tek başına TVDI'ye dayalı bir fire-risk modeli değildir.** TVDI, Step8'deki multi-feature model içinde yardımcı bir thermal/dryness göstergesi olarak değerlendirilmiştir; tek başına bir karar mekanizması olarak sunulmamaktadır.
- **FIRMS, target olarak kullanılmaz.** FIRMS yalnızca independent bir active-fire cross-check katmanıdır; hiçbir aşamada MCD64A1 ile OR-combine edilerek primary label'a dahil edilmez.
- **30 m çözünürlükte piksel bazlı bir MCD64A1 modeli değildir.** Step8, MCD64A1'in native ~500 m grid'ini kullanır; 30 m predictor'lar bu grid'e block-mode aggregation ile indirgenir.

## 3. Güncel Durum Özeti

| Bölüm | Durum | Kısa açıklama |
|---|---|---|
| Step1-4B | Tamamlandı | GEE veri hazırlığı, export/download, GeoTIFF validation |
| Step5 | Tamamlandı | Landsat LST anomaly |
| Step5C | Tamamlandı | TVDI/dryness ürünleri |
| Step6 | Tamamlandı / diagnostic | İlk burned-area association ve FIRMS cross-check |
| Step7A-E | Tamamlandı | Tiling, MODIS→Landsat downscaling, fusion/gap-fill |
| Step8A | Tamamlandı | Label-honest 500 m MCD64A1 modeling dataset |
| Step8B | Tamamlandı | Baseline vs thermal spatial-block CV |
| Step8C | Tamamlandı | Spatial-block bootstrap uncertainty |
| Step8D | Tamamlandı | Thermal feature ablation |
| Step8E | Tamamlandı | Final Step8 summary report |
| Step9 | Planlandı | Multi-year / second-AOI generalization |

Step8A-E'nin tamamlanmasıyla birlikte projenin mevcut bilimsel çekirdeği (current scientific core) tamamlanmış durumdadır: thermal feature'ların burned-area discrimination'a katkısı, tek bir AOI ve tek bir sezon için ölçülmüş ve belgelenmiştir. Step9 ve sonrası artık yeni bir hipotez testi değil, bu sonucun genelleştirilmesi ve zamanla/coğrafyayla ne kadar dayanıklı olduğunun test edilmesi aşamasıdır.

## 4. Pipeline Mantığı

Pipeline'ın neden bu sırayla ve bu şekilde kurgulandığını anlamak, Step açıklamalarını takip etmeyi kolaylaştırır.

İlk katman tamamen online: GEE üzerinden MODIS, Landsat, DEM ve MCD64A1 verileri export edilir ve Google Drive üzerinden indirilir. Bu katmanın tek görevi, ham veriyi güvenilir ve doğrulanabilir şekilde local diske taşımaktır; herhangi bir model eğitimi veya istatistiksel işlem burada yapılmaz.

İkinci katman local raster preprocessing'dir: LST anomaly (Step5), TVDI/dryness ürünleri (Step5C) ve MODIS→Landsat downscaling/fusion (Step7A-E) bu katmanda üretilir. Bu adımların ortak özelliği, herhangi bir fire label'ı görmeden çalışmalarıdır — yani downscaling modeli veya TVDI hesaplaması, hangi pikselin yanmış olduğunu bilmeden, saf termal/spektral ilişkilerden üretilir. Bu ayrım, sonradan modelin "yanmışlığı ezberlemesi" (leakage) riskini azaltmak için bilinçli olarak korunmuştur.

Üçüncü katman validation/modeling katmanıdır (Step6, Step8A-E). Burada iki temel prensip vardır: **label-resolution honesty** ve **spatial-block CV**. Label-resolution honesty, MCD64A1'in gerçek çözünürlüğünün (~500 m) altında sahte bir hassasiyet üretilmemesi anlamına gelir — 30 m'lik bir predictor grid'i, MCD64A1'in native grid'ine nearest-neighbor ile "büyütülmüş" bir label ile eşleştirilirse, aynı 500 m hücresindeki onlarca 30 m piksel birbirinin kopyası bir label paylaşır; bu da pseudo-replication'a yol açar ve model performansını yapay şekilde şişirir. Step8A bu sorunu, 30 m predictor'ları feature tipine uygun özet istatistiklerle (sürekli feature'lar için mean/median, kategorik feature'lar için mode/fraction) 500 m MCD64A1 grid'ine indirgeyerek çözer: artık her satır gerçekten bağımsız bir 500 m hücreyi temsil eder.

Spatial-block CV kullanılmasının nedeni de benzer bir mantıktan gelir: komşu hücreler birbirine çok benzer environmental koşullara (elevation, landcover, hatta yangın davranışına) sahip olduğu için, random row-wise bir train/test split kullanılırsa aynı yangının komşu hücreleri hem train hem test setine sızabilir. Bu da modelin gerçekte genelleme yapmadan, yalnızca mekânsal yakınlığı ezberleyerek yüksek skor almasına yol açar. Bu yüzden Step8B ve sonrasında `StratifiedGroupKFold`, mekânsal bloklara göre gruplanarak uygulanır; bu, modelin görmediği bir coğrafi bölgede ne kadar iyi genelleyebildiğini daha dürüst şekilde ölçer.

Son olarak, Step8C ve Step8D bu modelleme sonucunun ne kadar güvenilir olduğunu sorgular: Step8C, mevcut out-of-fold tahminleri üzerinden spatial-block bootstrap ile bir belirsizlik aralığı üretirken, Step8D thermal feature'ların hangisinin bu iyileşmeye ne kadar katkı sağladığını ayrıştırır.

## 5. Step Açıklamaları

### Step1-4B: Veri hazırlığı, export ve doğrulama

Bu aşama AOI'nin (Kozan) tanımlanmasıyla başlar; ardından MODIS, Landsat ve DEM verileri GEE üzerinden export edilir ve Google Drive üzerinden local diske indirilir. DEM'den elevation ve slope ürünleri türetilir. İndirilen tüm GeoTIFF dosyaları, metadata-driven bir doğrulama sürecinden geçer (Step4B); bu süreç dosya adlarına değil, `step5c_metadata.json` gibi referans metadata'ya dayanır, böylece aynı isimli veya "(1)" son ekli duplicate dosyalar yanlışlıkla atlanmaz. Bu aşamada herhangi bir model eğitimi yapılmaz; tamamen veri hazırlığı ve doğrulamadır.

### Step5: Landsat LST anomaly

Step5, current period (2023-06-01 → 2023-07-31, predictor window) ile historical baseline (2019-2022, aynı takvim penceresi) arasındaki Landsat yüzey sıcaklığı farkını hesaplar. Label window (2023-08-01 → 2023-10-31) ile predictor window arasında 1 günlük bir temporal lead bırakılmıştır; bu, "pre-fire" bir tahmin senaryosu kurgusudur. Zamansal interpolasyon uygulanmaz — yalnızca gerçek gözlemler kullanılır ve düşük valid-count'a sahip pikseller bir low-confidence maske ile işaretlenir. Çıktı olan `anomaly_zscore.tif`, tek başına bir fire-risk modeli değil, bir thermal anomaly ürünüdür.

### Step5C: TVDI / dryness ürünleri

TVDI (Temperature-Vegetation Dryness Index), LST-NDVI ilişkisinden yararlanarak yüzey kuruluğunu normalize etmeye çalışan bir indekstir. Bu adımda `current_tvdi`, `tvdi_difference` ve `tvdi_anomaly_zscore` ürünleri hesaplanır. `tvdi_anomaly_zscore` için baseline standart sapmasının düşük olduğu durumlarda bir reliability filtresi uygulanır, çünkü düşük std değerleri z-score'u yapay olarak şişirebilir; bu yüzden `current_tvdi` ve `tvdi_difference` daha doğrudan yorumlanabilir kabul edilir. TVDI, tek başına güçlü bir fire-risk predictor'ü olarak sunulmaz; nitekim Step8D ablation sonucunda TVDI grubunun pozitif ama ana sürücü olmadığı görülmüştür.

### Step6: Burned-area association diagnostics

Step6 bir model eğitmez; MCD64A1'i primary burned-area label olarak alıp, önceki adımlarda üretilen tekil indekslerin (özellikle TVDI) bu label ile ne kadar ilişkili olduğunu diagnostic olarak test eder. FIRMS burada da target değildir, yalnızca independent bir active-fire cross-check'tir. Step6'daki bulgular (tek-indeks discrimination'ın zayıf olması) projenin Step7-8'e, yani multi-feature modellemeye geçmesinin motivasyonunu oluşturmuştur. Ana bilimsel iddia artık Step6 üzerinden değil, Step8 üzerinden yapılmaktadır; Step6 tarihsel/diagnostic bir referans olarak repoda kalır.

### Step7A-E: Downscaling ve fusion

Step7A, sonraki adımlarda kullanılacak tiling/windowed işleme altyapısını kurar ve büyük raster'ların bellek sorunu yaşamadan işlenmesini sağlar. Step7B, MODIS→Landsat LST downscaling için temiz bir eğitim dataset'i hazırlar; bu dataset'te herhangi bir fire label bulunmaz. Step7C, bu dataset üzerinde saf bir MODIS→Landsat LST downscaling modeli eğitir (RandomForestRegressor); leakage guard mekanizması, anomaly/TVDI/z-score gibi türetilmiş feature'ların eğitime sızmasını engeller. Step7D, eğitilen modeli tüm raster grid'e windowed tiling ile uygulayarak full-grid downscaled bir LST üretir; grid uyuşmazlığı durumunda sessizce resample etmek yerine hata fırlatır. Step7E, gözlemlenen Landsat LST ile Step7D'nin downscaled çıktısını deterministik ve observed-priority bir mantıkla birleştirir — gözlemlenen pikseller asla üzerine yazılmaz, yalnızca boşluklar downscaled veriyle doldurulur. Step7 serisinin tamamı bir thermal context/fusion ürünüdür; bir fire-risk modeli değildir. Ancak bu adımların ürettiği downscaled/fused LST, Step8'de thermal feature olarak kullanılır.

### Step8A-E: Çekirdek burned-area modelleme deneyi

Bu bölüm, projenin bilimsel çekirdeğini oluşturduğu için diğerlerinden daha detaylı açıklanmaktadır.

**Step8A — Label-honest 500 m modeling dataset:** 30 m çözünürlükteki predictor raster'ları, MCD64A1'in native/reconstructed ~500 m grid'ine spatial aggregation ile indirgenir. Bu aggregation feature tipine göre değişir: sürekli (continuous) feature'lar (örn. elevation, slope, NDVI, LST anomaly) mean/median gibi özet istatistiklerle; kategorik feature'lar (örn. landcover) ise mode veya sınıf-oranı (fraction) gibi özetlerle 500 m grid'ine indirgenir. Sonuçta oluşan tabloda bir satır, gerçekten bağımsız bir 500 m hücreyi temsil eder. Bu adım, gerçek MCD64A1 BurnDate DOY verisini gerektirir; yalnızca binary (0/1) bir burned mask yeterli değildir, çünkü DOY bilgisi olmadan label kalitesi ve zamansal tutarlılık doğrulanamaz.

**Step8B — Baseline vs. thermal model:** Baseline model yalnızca elevation, slope, landcover ve NDVI kullanır. Thermal-augmented model bunlara ek olarak LST anomaly, current LST, current TVDI, TVDI difference ve downscaled/fused LST feature'larını ekler. İki model de spatial-block `StratifiedGroupKFold` CV ile karşılaştırılır; sonuçlar `delta_AUC` ve `delta_PR-AUC` birlikte raporlanır. PR-AUC özellikle önemlidir, çünkü burned sınıfı seyrektir (~%1,12) ve bu dengesiz dağılımda AUC tek başına ayrım gücünü yeterince yansıtmayabilir.

**Step8C — Spatial-block bootstrap uncertainty:** Step8B'nin ürettiği out-of-fold tahminleri yeniden eğitim yapılmadan, spatial-block bootstrap ile yeniden örneklenir. Bu, delta AUC ve delta PR-AUC için bir güven aralığı üretir; bu aralık klasik bir p-value değil, bir bootstrap percentile interval'dır.

**Step8D — Thermal feature ablation:** Thermal feature'lar gruplara ayrılarak (örn. yalnızca downscaled LST, yalnızca LST anomaly, yalnızca TVDI, tüm thermal feature'lar birlikte) hangi grubun performans farkına ne kadar katkı sağladığı test edilir. Sonuç: en iyi genel performans `all_thermal` grubundan gelirken, en iyi tekil feature `downscaled_only`'dir; LST anomaly grubu `all_thermal`'a yakın bir performans sergiler; TVDI grubu pozitif katkı sağlar ama ana sürücü değildir.

**Step8E — Final rapor:** Step8B, Step8C ve Step8D'nin sonuçlarını yeniden eğitim yapmadan tek bir özet raporda birleştirir.

## 6. Ana Bilimsel Sonuçlar

**Dataset özeti:**

| Metrik | Değer |
|---|---|
| Geçerli 500 m hücre | 48.422 |
| Burned | 542 |
| Unburned | 47.920 |
| Burn rate | ≈ %1,12 |
| Label kaynağı | MCD64A1 BurnDate DOY |

**Model karşılaştırması:**

| Popülasyon | AUC baseline | AUC thermal | Delta AUC | PR-AUC baseline | PR-AUC thermal | Delta PR-AUC |
|---|---|---|---|---|---|---|
| all_valid | 0.9673 | 0.9736 | +0.0063 | 0.1821 | 0.2310 | +0.0489 |
| cropland_dominant | 0.9035 | 0.9212 | +0.0177 | 0.1870 | 0.2487 | +0.0617 |

**Bootstrap güven aralıkları:**

| Popülasyon | Delta AUC %95 CI | Delta PR-AUC %95 CI | Yorum |
|---|---|---|---|
| all_valid | [0.0030, 0.0097] | [0.0198, 0.0812] | pozitif bootstrap desteği |
| cropland_dominant | [0.0097, 0.0260] | [0.0329, 0.0950] | pozitif bootstrap desteği |

Not: `cropland_dominant` stratası, 542 burned hücrenin 533'ünü kapsar; yani örneklem büyük ölçüde cropland-dominant araziye aittir.

**Ablation özeti:**
- En iyi genel grup: `all_thermal`
- En iyi tekil feature: `downscaled_only`
- LST anomaly grubu, `all_thermal`'a yakın performans gösteriyor
- TVDI grubu pozitif katkı veriyor ama tek/başat sürücü değil

**Önemli uyarı:** Bu sonuçlar bootstrap percentile interval'dır; klasik bir p-value değildir. "İstatistiksel olarak anlamlı" (statistically significant) veya "significantly improves" ifadeleri kullanılmaz. Doğru ifade: thermal feature'lar, mevcut Step8 deneyinde statik baseline'a göre ölçülebilir ve spatial-block-bootstrap ile desteklenen bir iyileşme sağlamaktadır.

## 7. Örnek Görseller

Şu an README'de final örnek görseller bulunmamaktadır. Önceki prototip haritalar geçici olarak kaldırılmıştır.

Bunun sebebi basittir: eski görseller, mevcut Step8 label-honest modelleme sonucunu doğrudan temsil etmeyebilir. Legend, CRS, ölçek, quality-mask ve final map layout konularında bir QA turu tamamlanmadan bu görsellerin README'de kalması yanıltıcı olabilirdi. Bu yüzden görsel paketi, doğru ve tutarlı bir sunum sağlanana kadar bilinçli olarak boş bırakılmıştır.

### Eklenecek görsel türleri

QA süreci tamamlandığında aşağıdaki ürünlerden örnekler README'ye eklenecektir:

- Landsat LST anomaly haritası
- Current TVDI haritası
- Downscaled LST haritası
- Fused LST haritası
- Step8 prediction / risk-score tarzı diagnostic harita
- Ablation / feature contribution grafiği
- Bootstrap güven aralığı grafiği

### Görsel ekleme politikası

README'ye yalnızca QA'dan geçmiş, doğru legend/scale/CRS bilgisine sahip görseller eklenecektir. Diagnostic amaçlı ara görseller ile final haritalar birbirine karıştırılmayacak, ve herhangi bir görsel başlığı "fire-risk prediction" gibi bir izlenim yaratacak şekilde yazılmayacaktır.

<!-- TODO: Final visualization package will be added after map QA and layout cleanup. -->

## 8. Önemli Label Notu

Step6 döneminde, MCD64A1'den yalnızca binary (0/1) bir `mcd64a1_raw.tif` üretilebiliyordu. Ancak Step8A, gerçek BurnDate DOY bilgisini gerektirir; binary bir mask, label kalitesini ve zamansal tutarlılığı doğrulamak için yeterli değildir. Bu nedenle `scripts/export_mcd64a1_raw_burndate.py` scripti, Step8A'dan önce çalıştırılacak şekilde pipeline'a eklenmiştir. Eğer bu script sonucu üretilen raw dosya yalnızca 0/1 değerleri içeriyorsa, Step8A fail-fast diagnostics ile bu durumu yakalar ve devam etmez. Bu kontrol, projenin genelinde benimsenen label-resolution honesty prensibinin doğrudan bir uygulamasıdır.

```bash
python scripts/export_mcd64a1_raw_burndate.py --also-binary
```

## 9. Kurulum

```bash
git clone <repo-url>
cd satellite-thermal-digital-twin

python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# veya
.venv\Scripts\activate       # Windows

pip install -r requirements.txt
earthengine authenticate
```

Google Drive export/download işlemleri için `.env` dosyasına ihtiyaç vardır. Drive klasör kimlik bilgilerinin (`GOOGLE_DRIVE_EXPORT_FOLDER_URL`, `GOOGLE_DRIVE_EXPORT_FOLDER_ID`) nasıl ayarlanacağı burada uzun uzun anlatılmaz; detaylı adımlar için `SETUP_ENV.md` dosyasına bakınız.

Kısa örnek:

```bash
cp .env.example .env
```

`.env` içinde:

```
GOOGLE_DRIVE_EXPORT_FOLDER_ID=YOUR_FOLDER_ID
GOOGLE_DRIVE_EXPORT_FOLDER_URL=https://drive.google.com/drive/u/0/folders/YOUR_FOLDER_ID
```

`.env.example`, doldurulacak bir şablon olarak kullanılır; gerçek `.env` dosyası repoya commit edilmez.

## 10. Çalıştırma

### Uçtan uca

```bash
python scripts/main.py
python scripts/main.py --force
```

`main.py`, Step1'den Step8E'ye kadar tüm pipeline'ı sırasıyla çalıştırır. Bu akışın bir kısmı (export/download) GEE ve Google Drive erişimi gerektiren online adımlardır; bir kısmı ise (Step5'ten sonrası) tamamen local raster/tablo işlemleridir. Raw MCD64A1 BurnDate export'u Step8A'dan önce mutlaka başarıyla tamamlanmış olmalıdır; aksi halde Step8A doğru çalışmaz. Step6 ise hata-toleranslıdır — başarısız olsa bile pipeline'ın geri kalanını durdurmaz.

### Adım adım

```bash
python src/step5_preprocess_timeseries.py
python src/step5c_tvdi.py
python src/step6_validate_fire_relation.py
python src/step7c_train_downscaling_model.py
python src/step7d_predict_downscaled_lst.py
python src/step7e_fuse_landsat_downscaled_lst.py
python scripts/export_mcd64a1_raw_burndate.py --also-binary
python src/step8a_prepare_500m_modeling_dataset.py --force
python src/step8b_train_baseline_vs_thermal_model.py --force
python src/step8c_spatial_block_bootstrap_uncertainty.py --force
python src/step8d_thermal_feature_ablation.py --force
python src/step8e_final_report.py --force
```

### Force/overwrite davranışı

`--force` bayrağı, önceki Step7C/7D/7E ve Step8A-E çıktılarını ezmek (overwrite) için kullanılır. `--force` verilmediğinde bazı adımlar mevcut çıktılar için fail-fast davranabilir veya yeniden üretimi reddedebilir; bu davranış adıma göre değişebilir. Bu yüzden tekrar çalıştırmalarda `--force` kullanılması önerilir.

## 11. Proje Yapısı

```
core/
  config.py                    # Merkezi konfigürasyon sabitleri
  paths.py                     # Girdi/çıktı yol tanımları
  io_utils.py                  # Ortak okuma/yazma yardımcıları
  validation_burned_area.py    # Burned-area doğrulama mantığı
  utils/                       # Diğer paylaşılan yardımcı modüller

src/
  step5_preprocess_timeseries.py
  step5c_tvdi.py
  step6_validate_fire_relation.py
  step7a ... step7e_*.py
  step8a_prepare_500m_modeling_dataset.py
  step8b_train_baseline_vs_thermal_model.py
  step8c_spatial_block_bootstrap_uncertainty.py
  step8d_thermal_feature_ablation.py
  step8e_final_report.py

scripts/
  main.py                              # Uçtan uca pipeline çalıştırıcı
  export_mcd64a1_raw_burndate.py       # Raw BurnDate DOY export scripti

data/       # Yerel ham/indirilmiş veri klasörleri
outputs/    # Her step'in ürettiği raster, tablo ve rapor çıktıları
```

`core/`, tüm step dosyalarının paylaştığı sabitleri ve yardımcı fonksiyonları barındırır; `src/`, her bir pipeline adımının çalıştırılabilir mantığını içerir; `scripts/`, uçtan uca çalıştırma ve tek seferlik yardımcı export işlemlerini içerir.

## 12. Ana Çıktılar

**Step4B:**
- `outputs/step4b/geotiff_validation_summary.json`
- `outputs/step4b/geotiff_validation_summary.md`

**Step5:**
- `outputs/step5/anomaly_zscore.tif`

**Step5C:**
- `outputs/step5c/current_tvdi.tif`
- `outputs/step5c/tvdi_difference.tif`

**Step7:**
- `outputs/step7d/downscaled_lst_celsius.tif`
- `outputs/step7e/fused_lst_celsius.tif`

**Step8:**
- `outputs/step8a/step8a_500m_modeling_dataset.parquet`
- `outputs/step8b/step8b_model_comparison_metrics.json`
- `outputs/step8b/step8b_predictions.parquet`
- `outputs/step8c/step8c_bootstrap_metrics.json`
- `outputs/step8d/step8d_ablation_metrics.json`
- `outputs/step8e/step8e_summary.md`

## 13. Sınırlamalar

- **Tek AOI:** Sonuçlar Kozan AOI'ye özeldir; başka coğrafyalara doğrudan genellenemez.
- **Tek sezon/yıl:** Analiz 2023 sezonunu kapsar; year-to-year robustness henüz test edilmemiştir.
- **Cropland dominance:** Burned label'ların büyük çoğunluğu cropland-dominant hücrelerde yoğunlaşmıştır; bu yüzden natural vegetation fire behavior için sonuçlar dikkatli yorumlanmalıdır.
- **Cropland-excluded burnable mask:** Bu maske içinde pozitif örnek sayısı azdır, bu da bu strata için istatistiksel gücü sınırlar.
- **Ekim ayı düşük pozitif sayısı:** Ekim'e ait sonuçlar düşük örneklem nedeniyle düşük güvenle yorumlanmalıdır.
- **Günlük MODIS gap-fill yok:** Mevcut fusion mantığı (Step7E) bir defalık, statik bir birleştirmedir; günlük operasyonel bir veri akışı değildir.
- **Multi-year / second-AOI genelleme henüz yapılmadı:** Bu, Step9'un kapsamındadır.
- **3B / operasyonel dijital ikiz katmanı henüz yoktur.**

## 14. Sonraki Adımlar

1. **Multi-year temporal generalization:** Modelin farklı yıllarda ne kadar dayanıklı olduğunu test etmek.
2. **Second AOI:** Sonuçların Kozan dışında başka bir bölgede tekrarlanıp tekrarlanamayacağını doğrulamak.
3. **Daily MODIS thermal context:** Günlük MODIS verisiyle daha sık güncellenen bir thermal context katmanı eklemek.
4. **Visualization QA ve final görsellerin eklenmesi:** Bölüm 7'de listelenen görsel paketini legend/CRS/ölçek kontrolünden geçirip README'ye eklemek.
5. **Operational rolling digital-twin prototipi:** Yukarıdaki adımlar tamamlandıktan sonra, sürekli güncellenen bir operasyonel prototipe geçiş için altyapı kurmak.

## 15. Terminoloji / Claim Policy

Bu README ve proje çıktıları, aşağıdaki ifade politikasına uyar:

- "Fire-risk prediction model validated" denmez.
- "Statistically significant" denmez.
- "TVDI alone predicts fire risk" denmez.
- Doğru ifade şudur: *"Thermal features provide measurable, spatial-block-bootstrap-supported improvement over static baseline features in the current Step8 experiment."*
- FIRMS hiçbir bağlamda bir target olarak sunulmaz.
- MCD64A1'in native ~500 m label çözünürlüğü korunur; hiçbir yerde 30 m piksel bazlı bir label hassasiyeti iddia edilmez.

## Sonuç

Step8 çekirdek deneyi tamamlanmıştır. Termal bilgi, statik baseline feature'ların ötesinde ölçülebilir bir prediktif sinyal eklemektedir; en iyi performans, tekil bir feature'dan değil, thermal feature'ların birlikte kullanılmasından (`all_thermal`) gelmektedir. Bu, henüz operasyonel bir sistem değildir. Bir sonraki öncelik, yeni bir tekil feature'ı cilalamak değil, mevcut sonucun multi-year ve second-AOI kapsamında genelleştirilmesidir.