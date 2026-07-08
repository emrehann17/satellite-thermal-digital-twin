# Uydu Tabanlı Termal Veri İşleme ve Burned-Area Modelleme Prototipi

## 1. Kısa Özet

Bu proje, Doğu Akdeniz bölgesinde MODIS, Landsat, NDVI, DEM (elevation/slope), landcover ve MCD64A1 burned-area verilerini işleyen çok adımlı bir uydu tabanlı termal veri işleme pipeline'ıdır. Pipeline, Google Earth Engine (GEE) üzerinden veri export/download işlemleriyle başlar; rasterio tabanlı local preprocessing, GeoTIFF doğrulama, termal anomaly ve dryness ürünleri üretimi ile devam eder; sonunda label-honest bir modeling/validation aşamasında sonlanır. Yani proje yalnızca raster üretimi yapan bir veri işleme hattı değil, aynı zamanda kendi ürettiği verileri istatistiksel olarak test eden bir doğrulama altyapısıdır.

Projenin bilimsel amacı nettir: termal ve dryness kökenli feature'ların (LST anomaly, TVDI, downscaled/fused LST) statik baseline feature'lara (elevation, slope, landcover, NDVI) kıyasla burned-area discrimination'a ölçülebilir bir katkı sağlayıp sağlamadığını test etmek. Bu soruyu cevaplamak için pipeline, Step1'den Step8E'ye kadar ilerleyen ve her adımda bir öncekinin çıktısını doğrulayan bir zincir şeklinde kurgulanmıştır.

Pipeline ilk olarak Kozan 2023 AOI'sinde çalıştırılmış ve Step8 çekirdek deneyi bu AOI üzerinde tamamlanmıştır: MCD64A1'in native ~500 m grid'i üzerinde, thermal-augmented model statik baseline'a göre ölçülebilir ve spatial-block-bootstrap ile desteklenen pozitif bir performans farkı göstermiştir. Ancak sonrasında yapılan bir supervisor değerlendirmesi, Kozan 2023'teki MCD64A1 burned label'larının büyük çoğunluğunun (533/542 hücre, bkz. Bölüm 5) cropland/anız-yakma kaynaklı olduğunu, doğal bitki örtüsü (orman/makilik) yangını olmadığını ortaya koymuştur. Bu nedenle **Kozan artık projenin doğal-bitki-örtüsü yangın modelinin kanıtı olarak sunulmaz; cropland/anız-dominant bir negative/control AOI olarak konumlandırılmıştır.**

Bu değerlendirmenin ardından proje, tek-AOI'den çoklu-deney (multi-experiment) bir yapıya geçmiştir: **Step0 deney/bölge kayıt defteri** (`core/regions.py`), her deneyi (region + yıl + predictor/label pencereleri + baseline yılları + rol + çıktı namespace'i) tek bir yerden yönetir. Şu an kayıtlı deneyler `kozan_2023` (negative_control, varsayılan) ve `manavgat_2021` (anchor_wildfire), artı ileride kullanılmak üzere disabled `valencia_2022`/`zamora_2022` placeholder'larıdır. Yeni bir AOI'yi modellemeden önce, **Step6B burned-landcover gate**, o AOI'nin MCD64A1-burned hücrelerinin gerçekten doğal bitki örtüsü mü yoksa cropland mı olduğunu 500 m native grid seviyesinde doğrular. Manavgat 2021, bu gate'i `wildfire_candidate_pass` kararıyla geçmiştir (796 burned hücrenin 783'ü tree/shrub/grass dominant) ve artık bir sonraki tam modelleme çalıştırması için **anchor wildfire AOI**'dir — ancak Manavgat için Step1-Step8 tam modelleme henüz çalıştırılmamıştır; şu ana kadar yalnızca AOI uygunluk (suitability) doğrulaması yapılmıştır.

Bunun ötesinde, bu çalışmanın ne olduğu konusunda net olmak önemlidir: bu proje tamamlanmış bir operasyonel yangın erken uyarı sistemi değildir ve tamamlanmış bir 3B dijital ikiz de değildir. Şu anki hâliyle proje; termal preprocessing, MODIS→Landsat downscaling/fusion, label-honest bir burned-area modeling altyapısı (Kozan üzerinde tamamlanmış) ve bu altyapıyı yeni AOI'lere genişletmek için bir deney kaydı + suitability gate katmanıdır. Sonraki adımlar (bkz. Bölüm 16) Manavgat'ın tam modellenmesine ve bu altyapının daha da genelleştirilmesine yöneliktir.

## 2. Bu Proje Ne Değildir?

- **Tamamlanmış bir 3B dijital ikiz değildir.** Şu an yalnızca 2B raster katmanları ve tablo bazlı bir modeling dataset'i üretilmektedir; 3B görselleştirme veya simülasyon katmanı yoktur.
- **Operasyonel bir yangın erken uyarı sistemi değildir.** Henüz multi-year ve near-real-time üretim akışı yoktur.
- **Kozan, doğal-bitki-örtüsü yangın modelinin kanıtı değildir.** Kozan 2023 burned label'ları cropland/anız-yakma dominanttır (bkz. Bölüm 5); Kozan bir negative/control AOI'dir, wildfire validation AOI'si değildir.
- **Manavgat için henüz bir model sonucu yoktur.** Manavgat 2021 yalnızca Step6B burned-landcover gate'ini (AOI suitability) geçmiştir; Step1-Step8 tam modelleme henüz çalıştırılmamıştır.
- **Tek başına TVDI'ye dayalı bir fire-risk modeli değildir.** TVDI, Step8'deki multi-feature model içinde yardımcı bir thermal/dryness göstergesi olarak değerlendirilmiştir; tek başına bir karar mekanizması olarak sunulmamaktadır.
- **FIRMS, target olarak kullanılmaz.** FIRMS yalnızca independent bir active-fire cross-check katmanıdır; hiçbir aşamada MCD64A1 ile OR-combine edilerek primary label'a dahil edilmez.
- **30 m çözünürlükte piksel bazlı bir MCD64A1 modeli değildir.** Step8 (ve Step6B gate), MCD64A1'in native ~500 m grid'ini kullanır; 30 m predictor'lar bu grid'e block-mode aggregation ile indirgenir.

## 3. Güncel Durum Özeti

| Bölüm | Durum | Kısa açıklama |
|---|---|---|
| Step0 | Tamamlandı | Deney/bölge (experiment/region) kayıt defteri ve namespaced çıktı yapısı |
| Step1-4B | Tamamlandı (Kozan) | GEE veri hazırlığı, export/download, GeoTIFF validation |
| Step5 | Tamamlandı (Kozan) | Landsat LST anomaly |
| Step5C | Tamamlandı (Kozan) | TVDI/dryness ürünleri |
| Step6 | Tamamlandı / diagnostic | İlk burned-area association, FIRMS cross-check, canonical raw BurnDate export |
| Step6A | Tamamlandı | Deney-farkında (experiment-aware) gate-only referans grid + landcover hazırlığı |
| Step6B | Tamamlandı | Burned-landcover gate (500 m native grid seviyesinde AOI suitability kontrolü) |
| Kozan gate | Tamamlandı | `cropland_dominated_control` |
| Manavgat gate | Tamamlandı | `wildfire_candidate_pass` |
| Step7A-E | Tamamlandı (Kozan) | Tiling, MODIS→Landsat downscaling, fusion/gap-fill |
| Step8A | Tamamlandı (Kozan) | Label-honest 500 m MCD64A1 modeling dataset |
| Step8B | Tamamlandı (Kozan) | Baseline vs thermal spatial-block CV |
| Step8C | Tamamlandı (Kozan) | Spatial-block bootstrap uncertainty |
| Step8D | Tamamlandı (Kozan) | Thermal feature ablation |
| Step8E | Tamamlandı (Kozan) | Final Step8 summary report |
| Manavgat Step1-Step8 modelleme | Sıradaki | Gate geçti; tam modelleme henüz çalıştırılmadı |
| Valencia/Zamora transfer | Planlandı | Disabled placeholder deneyler; external validation / hard transfer test |

Step8A-E'nin Kozan üzerinde tamamlanmasıyla birlikte projenin metodolojik çekirdeği doğrulanmıştır: thermal feature'ların burned-area discrimination'a katkısı, label-honest bir 500 m grid ve spatial-block CV ile ölçülmüş ve belgelenmiştir. Ancak Kozan'ın cropland-dominant doğası ortaya çıktıktan sonra öncelik, aynı metodolojiyi doğrulanmış bir wildfire AOI'sinde (Manavgat) tekrarlamak olmuştur — bu da önce bir AOI suitability gate'i (Step6B) gerektirmiştir.

## 4. Experiment Registry ve AOI Rolleri

`core/regions.py`, iki ayrı kavramı birbirinden ayırır:

- **region** = yalnızca geometri (AOI).
- **experiment** = region + yıl + predictor penceresi + label penceresi + baseline yılları + rol (`role`) + çıktı namespace'i.

Kayıtlı deneyler:

| experiment_id | role | Durum |
|---|---|---|
| `kozan_2023` | `negative_control` | Enabled, varsayılan (default) deney |
| `manavgat_2021` | `anchor_wildfire` | Enabled; gate geçti, tam modelleme sırada |
| `valencia_2022` | `external_validation` | Disabled placeholder |
| `zamora_2022` | `hard_transfer_test` | Disabled placeholder |

Aktif deney, `core/regions.py:get_active_experiment(experiment_id)` ile çözülür (varsayılan `kozan_2023`, geriye dönük uyumluluk için). Her deneyin çıktıları kendi namespace'i altında toplanır:

```
outputs/experiments/<experiment_id>/
```

Kozan'ın legacy (namespace'siz) çıktı yolları (`outputs/step5/`, `outputs/validation/labels/`, vb.) değişmeden korunur — Step1-Step8 script'leri hâlâ bu legacy yolları kullanır. Kozan-dışı deneyler (şu an yalnızca Manavgat'ın gate-only zinciri) tamamen `outputs/experiments/<experiment_id>/` altında çalışır ve legacy Kozan dosyalarına asla yazmaz/okumaz (bkz. Bölüm 5 ve `docs/manavgat_gate_only.md`).

## 5. Burned-Landcover Gate

Yeni bir AOI'yi modellemeden önce, Step6B burned-landcover gate, o AOI'nin MCD64A1-burned hücrelerinin landcover kompozisyonunu 500 m native grid seviyesinde (`gate_level = 500m_reconstructed_mcd64a1_cell`) özetler ve üç soruya cevap verir: hücreler ağırlıklı olarak doğal bitki örtüsü mü (wildfire candidate), cropland mı (control), yoksa yeterli pozitif örnek yok mu?

**Karar kuralları:**

```
burned_count < 30                                  -> insufficient_burned_positives
tree+shrub+grass fraction >= 0.50                  -> wildfire_candidate_pass
cropland fraction >= 0.50                          -> cropland_dominated_control
aksi halde                                         -> mixed_or_uncertain
```

**Sonuçlar:**

| Experiment | Role | Decision | Burned cells | Tree+shrub+grass | Cropland-dominant | Yorum |
|---|---|---:|---:|---:|---:|---|
| kozan_2023 | negative_control | cropland_dominated_control | 542 | 9 | 533 | Anız/cropland-dominant control |
| manavgat_2021 | anchor_wildfire | wildfire_candidate_pass | 796 | 783 | 2 | Doğal bitki örtüsü wildfire AOI |

Manavgat için detay: `tree_cover=708`, `grassland=74`, `shrubland=1`, `cropland=2`; natural vegetation fraction ≈ %98,37, tree+shrub fraction ≈ %89,07, cropland fraction ≈ %0,25. Raw BurnDate diagnostics (`looks_binary=false`, `count_one=0`, `count_gt_one=179667`, `count_in_label_doy_range=179667`) etiketin gerçek DOY değerleri içerdiğini, binary bir maskeye düşmediğini doğrular.

Bu gate **diagnostic**tir: `cropland_dominated_control` sonucu (Kozan'ın beklenen sonucu) pipeline'ı durdurmaz. Gate yalnızca raw BurnDate binary görünüyorsa, gerekli girdi rasterları eksikse, veya landcover class mapping çözülemiyorsa hata verir. Detaylar için `docs/label_gate.md`, `docs/aoi_refinement.md` ve `docs/manavgat_gate_only.md`.

## 6. Pipeline Mantığı

Pipeline'ın neden bu sırayla ve bu şekilde kurgulandığını anlamak, Step açıklamalarını takip etmeyi kolaylaştırır.

İlk katman tamamen online: GEE üzerinden MODIS, Landsat, DEM ve MCD64A1 verileri export edilir ve Google Drive üzerinden indirilir. Bu katmanın tek görevi, ham veriyi güvenilir ve doğrulanabilir şekilde local diske taşımaktır; herhangi bir model eğitimi veya istatistiksel işlem burada yapılmaz.

İkinci katman local raster preprocessing'dir: LST anomaly (Step5), TVDI/dryness ürünleri (Step5C) ve MODIS→Landsat downscaling/fusion (Step7A-E) bu katmanda üretilir. Bu adımların ortak özelliği, herhangi bir fire label'ı görmeden çalışmalarıdır — yani downscaling modeli veya TVDI hesaplaması, hangi pikselin yanmış olduğunu bilmeden, saf termal/spektral ilişkilerden üretilir. Bu ayrım, sonradan modelin "yanmışlığı ezberlemesi" (leakage) riskini azaltmak için bilinçli olarak korunmuştur.

Üçüncü katman validation/modeling katmanıdır (Step6, Step6B, Step8A-E). Burada iki temel prensip vardır: **label-resolution honesty** ve **spatial-block CV**. Label-resolution honesty, MCD64A1'in gerçek çözünürlüğünün (~500 m) altında sahte bir hassasiyet üretilmemesi anlamına gelir — 30 m'lik bir predictor grid'i, MCD64A1'in native grid'ine nearest-neighbor ile "büyütülmüş" bir label ile eşleştirilirse, aynı 500 m hücresindeki onlarca 30 m piksel birbirinin kopyası bir label paylaşır; bu da pseudo-replication'a yol açar ve model performansını yapay şekilde şişirir. Step8A (ve aynı block/tile mantığını reuse eden Step6B gate) bu sorunu, 30 m predictor'ları feature tipine uygun özet istatistiklerle (sürekli feature'lar için mean/median, kategorik feature'lar için mode/fraction) 500 m MCD64A1 grid'ine indirgeyerek çözer: artık her satır/hücre gerçekten bağımsız bir 500 m hücreyi temsil eder.

Spatial-block CV kullanılmasının nedeni de benzer bir mantıktan gelir: komşu hücreler birbirine çok benzer environmental koşullara (elevation, landcover, hatta yangın davranışına) sahip olduğu için, random row-wise bir train/test split kullanılırsa aynı yangının komşu hücreleri hem train hem test setine sızabilir. Bu da modelin gerçekte genelleme yapmadan, yalnızca mekânsal yakınlığı ezberleyerek yüksek skor almasına yol açar. Bu yüzden Step8B ve sonrasında `StratifiedGroupKFold`, mekânsal bloklara göre gruplanarak uygulanır; bu, modelin görmediği bir coğrafi bölgede ne kadar iyi genelleyebildiğini daha dürüst şekilde ölçer.

Son olarak, Step8C ve Step8D bu modelleme sonucunun ne kadar güvenilir olduğunu sorgular: Step8C, mevcut out-of-fold tahminleri üzerinden spatial-block bootstrap ile bir belirsizlik aralığı üretirken, Step8D thermal feature'ların hangisinin bu iyileşmeye ne kadar katkı sağladığını ayrıştırır.

Bu üçüncü katmanın önüne, yeni bir AOI eklendiğinde (Step0 deney kaydı ile) bir **suitability gate** (Step6A + Step6B) eklenmiştir: tam Step1-Step8 zincirini çalıştırmadan önce, o AOI'nin gerçekten bir wildfire adayı olup olmadığı ucuz ve hızlı bir şekilde doğrulanır.

## 7. Step Açıklamaları

### Step1-4B: Veri hazırlığı, export ve doğrulama

Bu aşama AOI'nin (Kozan) tanımlanmasıyla başlar; ardından MODIS, Landsat ve DEM verileri GEE üzerinden export edilir ve Google Drive üzerinden local diske indirilir. DEM'den elevation ve slope ürünleri türetilir. İndirilen tüm GeoTIFF dosyaları, metadata-driven bir doğrulama sürecinden geçer (Step4B); bu süreç dosya adlarına değil, `step5c_metadata.json` gibi referans metadata'ya dayanır, böylece aynı isimli veya "(1)" son ekli duplicate dosyalar yanlışlıkla atlanmaz. Bu aşamada herhangi bir model eğitimi yapılmaz; tamamen veri hazırlığı ve doğrulamadır.

### Step5: Landsat LST anomaly

Step5, current period (2023-06-01 → 2023-07-31, predictor window) ile historical baseline (2019-2022, aynı takvim penceresi) arasındaki Landsat yüzey sıcaklığı farkını hesaplar. Label window (2023-08-01 → 2023-10-31) ile predictor window arasında 1 günlük bir temporal lead bırakılmıştır; bu, "pre-fire" bir tahmin senaryosu kurgusudur. Zamansal interpolasyon uygulanmaz — yalnızca gerçek gözlemler kullanılır ve düşük valid-count'a sahip pikseller bir low-confidence maske ile işaretlenir. Çıktı olan `anomaly_zscore.tif`, tek başına bir fire-risk modeli değil, bir thermal anomaly ürünüdür.

### Step5C: TVDI / dryness ürünleri

TVDI (Temperature-Vegetation Dryness Index), LST-NDVI ilişkisinden yararlanarak yüzey kuruluğunu normalize etmeye çalışan bir indekstir. Bu adımda `current_tvdi`, `tvdi_difference` ve `tvdi_anomaly_zscore` ürünleri hesaplanır. `tvdi_anomaly_zscore` için baseline standart sapmasının düşük olduğu durumlarda bir reliability filtresi uygulanır, çünkü düşük std değerleri z-score'u yapay olarak şişirebilir; bu yüzden `current_tvdi` ve `tvdi_difference` daha doğrudan yorumlanabilir kabul edilir. TVDI, tek başına güçlü bir fire-risk predictor'ü olarak sunulmaz; nitekim Step8D ablation sonucunda TVDI grubunun pozitif ama ana sürücü olmadığı görülmüştür.

### Step6: Burned-area association diagnostics + canonical raw BurnDate export

Step6 bir model eğitmez; MCD64A1'i primary burned-area label olarak alıp, önceki adımlarda üretilen tekil indekslerin (özellikle TVDI) bu label ile ne kadar ilişkili olduğunu diagnostic olarak test eder. FIRMS burada da target değildir, yalnızca independent bir active-fire cross-check'tir. Step6'daki bulgular (tek-indeks discrimination'ın zayıf olması) projenin Step7-8'e, yani multi-feature modellemeye geçmesinin motivasyonunu oluşturmuştur.

Ayrıca Step6, artık **canonical raw MCD64A1 BurnDate export'unun tek sahibidir** (`export_raw_mcd64a1_labels()`): gerçek BurnDate DOY değerlerini (1..366) `mcd64a1_raw.tif`'e, isteğe bağlı binary maskeyi `mcd64a1_burned.tif`'e yazar. Deney-farkında (experiment-aware) çağrıldığında (bkz. Bölüm 4) bu, Kozan'ın legacy paylaşılan dosyalarına dokunmadan, herhangi bir deneyin (örn. Manavgat) namespaced çıktı dizinine yazabilir. `scripts/export_mcd64a1_raw_burndate.py` artık yalnızca bu fonksiyonu çağıran ince bir CLI sarmalayıcıdır — iki farklı/divergent implementasyon yoktur.

### Step6A: Gate-only girdi hazırlama (deney-farkında)

`src/step6a_prepare_gate_inputs.py`, Step6B gate'in ihtiyaç duyduğu iki minimum rasteri, tam Step3/Step5 termal pipeline'ını çalıştırmadan hazırlar: (1) AOI'yi kaplayan sabit-değerli bir 30 m referans grid rasteri (`ee.Image.constant(1)` — **bir termal predictor değildir**, yalnızca grid geometrisi için), ve (2) ESA WorldCover v200 landcover'ının bu referans gride nearest-neighbor ile hizalanmış kopyası (Step8A'nın `prepare_aligned_landcover()` fonksiyonu reuse edilir). Tüm çıktılar `outputs/experiments/<experiment_id>/gate_inputs/` altına, namespaced olarak yazılır.

### Step6B: Burned-landcover gate

Bkz. Bölüm 5. `src/step6b_burned_landcover_gate.py`, Step8A'nın aynı 500 m block/tile mantığını (`compute_block_size_pixels`, `mode_and_agreement`, ESA WorldCover class mapping) reuse ederek MCD64A1-burned hücrelerin landcover kompozisyonunu özetler ve `wildfire_candidate_pass` / `cropland_dominated_control` / `insufficient_burned_positives` / `mixed_or_uncertain` kararlarından birini verir. `--label-path`/`--reference-path`/`--landcover-path`/`--output-dir` argümanları verilmezse Kozan'ın legacy yollarını kullanır; verilirse (örn. Manavgat'ın namespaced dosyaları) yalnızca onları kullanır.

### Step7A-E: Downscaling ve fusion

Step7A, sonraki adımlarda kullanılacak tiling/windowed işleme altyapısını kurar ve büyük raster'ların bellek sorunu yaşamadan işlenmesini sağlar. Step7B, MODIS→Landsat LST downscaling için temiz bir eğitim dataset'i hazırlar; bu dataset'te herhangi bir fire label bulunmaz. Step7C, bu dataset üzerinde saf bir MODIS→Landsat LST downscaling modeli eğitir (RandomForestRegressor); leakage guard mekanizması, anomaly/TVDI/z-score gibi türetilmiş feature'ların eğitime sızmasını engeller. Step7D, eğitilen modeli tüm raster grid'e windowed tiling ile uygulayarak full-grid downscaled bir LST üretir; grid uyuşmazlığı durumunda sessizce resample etmek yerine hata fırlatır. Step7E, gözlemlenen Landsat LST ile Step7D'nin downscaled çıktısını deterministik ve observed-priority bir mantıkla birleştirir — gözlemlenen pikseller asla üzerine yazılmaz, yalnızca boşluklar downscaled veriyle doldurulur. Step7 serisinin tamamı bir thermal context/fusion ürünüdür; bir fire-risk modeli değildir. Ancak bu adımların ürettiği downscaled/fused LST, Step8'de thermal feature olarak kullanılır.

### Step8A-E: Çekirdek burned-area modelleme deneyi (Kozan üzerinde tamamlandı)

Bu bölüm, projenin bilimsel çekirdeğini oluşturduğu için diğerlerinden daha detaylı açıklanmaktadır. **Bu adımlar şu ana kadar yalnızca Kozan 2023 üzerinde çalıştırılmıştır**; Manavgat için henüz çalıştırılmamıştır (bkz. Bölüm 16, Sonraki Adımlar).

**Step8A — Label-honest 500 m modeling dataset:** 30 m çözünürlükteki predictor raster'ları, MCD64A1'in native/reconstructed ~500 m grid'ine spatial aggregation ile indirgenir. Bu aggregation feature tipine göre değişir: sürekli (continuous) feature'lar (örn. elevation, slope, NDVI, LST anomaly) mean/median gibi özet istatistiklerle; kategorik feature'lar (örn. landcover) ise mode veya sınıf-oranı (fraction) gibi özetlerle 500 m grid'ine indirgenir. Sonuçta oluşan tabloda bir satır, gerçekten bağımsız bir 500 m hücreyi temsil eder. Bu adım, gerçek MCD64A1 BurnDate DOY verisini gerektirir; yalnızca binary (0/1) bir burned mask yeterli değildir, çünkü DOY bilgisi olmadan label kalitesi ve zamansal tutarlılık doğrulanamaz.

**Step8B — Baseline vs. thermal model:** Baseline model yalnızca elevation, slope, landcover ve NDVI kullanır. Thermal-augmented model bunlara ek olarak LST anomaly, current LST, current TVDI, TVDI difference ve downscaled/fused LST feature'larını ekler. İki model de spatial-block `StratifiedGroupKFold` CV ile karşılaştırılır; sonuçlar `delta_AUC` ve `delta_PR-AUC` birlikte raporlanır. PR-AUC özellikle önemlidir, çünkü burned sınıfı seyrektir (~%1,12) ve bu dengesiz dağılımda AUC tek başına ayrım gücünü yeterince yansıtmayabilir.

**Step8C — Spatial-block bootstrap uncertainty:** Step8B'nin ürettiği out-of-fold tahminleri yeniden eğitim yapılmadan, spatial-block bootstrap ile yeniden örneklenir. Bu, delta AUC ve delta PR-AUC için bir güven aralığı üretir; bu aralık klasik bir p-value değil, bir bootstrap percentile interval'dır.

**Step8D — Thermal feature ablation:** Thermal feature'lar gruplara ayrılarak (örn. yalnızca downscaled LST, yalnızca LST anomaly, yalnızca TVDI, tüm thermal feature'lar birlikte) hangi grubun performans farkına ne kadar katkı sağladığı test edilir. Sonuç: en iyi genel performans `all_thermal` grubundan gelirken, en iyi tekil feature `downscaled_only`'dir; LST anomaly grubu `all_thermal`'a yakın bir performans sergiler; TVDI grubu pozitif katkı sağlar ama ana sürücü değildir.

**Step8E — Final rapor:** Step8B, Step8C ve Step8D'nin sonuçlarını yeniden eğitim yapmadan tek bir özet raporda birleştirir.

## 8. Ana Bilimsel Sonuçlar (Kozan 2023)

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

Not: `cropland_dominant` stratası, 542 burned hücrenin 533'ünü kapsar; yani örneklem büyük ölçüde cropland-dominant araziye aittir (bkz. Bölüm 5, Kozan gate kararı: `cropland_dominated_control`).

**Ablation özeti:**
- En iyi genel grup: `all_thermal`
- En iyi tekil feature: `downscaled_only`
- LST anomaly grubu, `all_thermal`'a yakın performans gösteriyor
- TVDI grubu pozitif katkı veriyor ama tek/başat sürücü değil

**Önemli uyarı:** Bu sonuçlar bootstrap percentile interval'dır; klasik bir p-value değildir. "İstatistiksel olarak anlamlı" (statistically significant) veya "significantly improves" ifadeleri kullanılmaz. Doğru ifade: thermal feature'lar, Kozan üzerindeki mevcut Step8 deneyinde statik baseline'a göre ölçülebilir ve spatial-block-bootstrap ile desteklenen bir iyileşme sağlamaktadır. Bu sonuç cropland-dominant bir AOI'de elde edilmiştir; doğal bitki örtüsü yangın davranışına genellenmesi Manavgat modellemesi tamamlanana kadar doğrulanmamıştır.

## 9. Örnek Görseller

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

## 10. Önemli Label Notu

Step6 döneminde, MCD64A1'den yalnızca binary (0/1) bir `mcd64a1_raw.tif` üretilebiliyordu. Ancak Step8A, gerçek BurnDate DOY bilgisini gerektirir; binary bir mask, label kalitesini ve zamansal tutarlılığı doğrulamak için yeterli değildir. Bu sorunu çözmek için **Step6, artık canonical raw MCD64A1 BurnDate export'unun tek sahibidir** (`export_raw_mcd64a1_labels()` fonksiyonu, `src/step6_validate_fire_relation.py` içinde). `scripts/export_mcd64a1_raw_burndate.py` script'i bu fonksiyonu çağıran ince bir CLI sarmalayıcı olarak kalmıştır — iki farklı implementasyon yoktur.

Eğer export edilen raw dosya yalnızca 0/1 değerleri içeriyorsa (binary görünüyorsa), hem **Step8A** hem de **Step6B burned-landcover gate** bunu fail-fast diagnostics ile yakalar ve devam etmez. Bu kontrol, projenin genelinde benimsenen label-resolution honesty prensibinin doğrudan bir uygulamasıdır.

```bash
python scripts/export_mcd64a1_raw_burndate.py --also-binary
```

## 11. Kurulum

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

## 12. Çalıştırma

Tüm örnek komutlarda, kalıcı (shell `tee`) log çıktılarının repo kök dizinine değil `logs/` altına yazıldığına dikkat edin (`logs/` zaten `.gitignore`'dadır). Bu, `core/io_utils.py:setup_logger()` tarafından üretilen kod-seviyesi log dosyalarından (bunlar zaten otomatik olarak `logs/<step_adı>_<zaman_damgası>.log` konumuna yazılır) ayrı, yalnızca terminal çıktısını kalıcı hale getirmek isteyenler içindir.

### Uçtan uca (yalnızca Kozan; Manavgat henüz deney-farkında tam pipeline'a sahip değil)

```bash
python scripts/main.py
python scripts/main.py --force
```

`main.py`, Step1'den Step8E'ye kadar tüm pipeline'ı sırasıyla çalıştırır. Bu akışın bir kısmı (export/download) GEE ve Google Drive erişimi gerektiren online adımlardır; bir kısmı ise (Step5'ten sonrası) tamamen local raster/tablo işlemleridir. Step6'nın hemen ardından, ZORUNLU (hata-toleranslı olmayan) bir "label export cleanup" adımı (raw MCD64A1 BurnDate export) ve ardından diagnostic burned-landcover gate çalışır; Step8A'dan önce raw BurnDate export'unun başarıyla tamamlanmış olması gerekir. Step6'nın kendi association-test adımı ise hata-toleranslıdır — başarısız olsa bile pipeline'ın geri kalanını durdurmaz.

### Deney (experiment) seçimi ile dry-run

```bash
python scripts/main.py --experiment kozan_2023 --dry-run
python scripts/main.py --experiment manavgat_2021 --dry-run
```

`--dry-run`, yalnızca Step0 aktif deney bilgisini (region, pencereler, baseline yılları, çıktı kökü) yazdırır; hiçbir export/pipeline çalıştırmaz. `kozan_2023` dışındaki bir deney ile `--dry-run` olmadan `python scripts/main.py --experiment ...` çalıştırmak fail-fast bir hata verir (Step1-Step8 henüz tam deney-farkında değildir).

### AOI önizleme

```bash
python scripts/preview_experiment_aoi.py --experiment kozan_2023
python scripts/preview_experiment_aoi.py --experiment manavgat_2021
```

Bir deneyin Step0 metadata'sını ve (GEE erişilebilirse) AOI geometri tipini/kaba sınırlarını yazdırır; hiçbir export/pipeline/model çalıştırmaz. GEE initialize edilemezse çökmeden nazikçe uyarı verir.

### Gate-only dry-run'lar

```bash
mkdir -p logs
python scripts/run_label_gate_only.py --experiment kozan_2023 --dry-run 2>&1 | tee logs/kozan_gate_dryrun.log
python scripts/run_label_gate_only.py --experiment manavgat_2021 --dry-run 2>&1 | tee logs/manavgat_gate_dryrun.log
```

Her ikisi de yalnızca Step0 özetini ve TÜM planlanan dosya yollarını (Kozan için legacy yollar, Manavgat için `outputs/experiments/manavgat_2021/...` altında namespaced yollar) yazdırır; hiçbir export/gate çalıştırmaz.

### Manavgat gate-only çalıştırma (gerçek export + gate)

```bash
mkdir -p logs
python scripts/run_label_gate_only.py --experiment manavgat_2021 --export-labels --force 2>&1 | tee logs/manavgat_gate_run.log
```

Bu komut **Step7 veya Step8'i ÇALIŞTIRMAZ**. Yalnızca: (1) Step6A ile gate-only referans grid + hizalanmış landcover'ı hazırlar (namespaced, termal predictor değil), (2) Manavgat'ın label penceresi için raw MCD64A1 BurnDate'i export eder (namespaced), (3) Step6B burned-landcover gate'ini namespaced dosyalarla çalıştırır. Tüm çıktılar `outputs/experiments/manavgat_2021/` altındadır; Kozan'ın legacy paylaşılan dosyalarına asla yazmaz (runtime safety check, `scripts/run_label_gate_only.py:_assert_paths_are_safely_namespaced`).

`--skip-export` (varsayılan davranış, `--export-labels` verilmezse) raw BurnDate export'unu atlar ve gate'in mevcut dosyayı kullanmasını sağlar — thresholds üzerinde iterasyon yaparken GEE kotası harcamamak için kullanışlıdır.

### Manavgat experiment-aware predictor üretimi (Step3-Step5/5C)

```bash
python scripts/run_predictors_only.py --experiment kozan_2023 --dry-run
python scripts/run_predictors_only.py --experiment manavgat_2021 --dry-run
python scripts/run_predictors_only.py --experiment manavgat_2021 --export --force \
  2>&1 | tee logs/manavgat_predictors_export.log
python scripts/run_predictors_only.py --experiment manavgat_2021 --local-only --force \
  2>&1 | tee logs/manavgat_predictors_local.log
```

Bu script **Step7 veya Step8'i ÇALIŞTIRMAZ**, model eğitmez. `kozan_2023` legacy davranışını korur; `manavgat_2021` (ve kozan-dışı her deney) tamamen `outputs/experiments/<experiment_id>/` altında, namespaced çalışır (`core/experiment_context.py:build_experiment_context()`). `--export`, current+baseline Landsat LST/NDVI'yı doğrudan GEE'den yerel diske export eder (Step4/4B'nin Drive zincirini replike etmez) ve ardından Step5/Step5C'yi çalıştırır; `--local-only` bu dosyaların zaten var olduğunu varsayar ve yalnızca Step5/Step5C'yi çalıştırır. Manavgat için `current_period_days`, mevcut `CURRENT_PERIOD_DAYS` konvansiyonuyla (basit tarih farkı) birebir tutarlı şekilde **56** olarak hesaplanır (57 değil — bkz. `docs/experiments.md`).

### Adım adım (Kozan, legacy)

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

`--force` bayrağı, önceki çıktıları ezmek (overwrite) için kullanılır (Step7C/7D/7E, Step8A-E, Step6A gate-input hazırlama, Step6B gate). `--force` verilmediğinde bazı adımlar mevcut çıktılar için fail-fast davranabilir veya yeniden üretimi reddedebilir; bu davranış adıma göre değişebilir. Bu yüzden tekrar çalıştırmalarda `--force` kullanılması önerilir.

## 13. Proje Yapısı

```
core/
  config.py                    # Merkezi konfigürasyon sabitleri
  paths.py                     # Girdi/çıktı yol tanımları
  io_utils.py                  # Ortak okuma/yazma yardımcıları
  regions.py                   # Step0 deney/bölge (experiment/region) kayıt defteri
  experiment_context.py        # Step3-Step5/5C için deney-farkında path/tarih context'i
  validation_burned_area.py    # Burned-area doğrulama mantığı
  utils/                       # Diğer paylaşılan yardımcı modüller

src/
  step5_preprocess_timeseries.py
  step5c_tvdi.py
  step6_validate_fire_relation.py
  step6a_prepare_gate_inputs.py        # Gate-only referans grid + landcover hazırlığı (deney-farkında)
  step6b_burned_landcover_gate.py      # Burned-landcover gate
  step7a ... step7e_*.py
  step8a_prepare_500m_modeling_dataset.py
  step8b_train_baseline_vs_thermal_model.py
  step8c_spatial_block_bootstrap_uncertainty.py
  step8d_thermal_feature_ablation.py
  step8e_final_report.py

scripts/
  main.py                              # Uçtan uca pipeline çalıştırıcı (--experiment, --dry-run destekler)
  export_mcd64a1_raw_burndate.py       # Raw BurnDate DOY export'u için ince CLI sarmalayıcı
  preview_experiment_aoi.py            # Deney metadata + AOI önizleme (export/pipeline çalıştırmaz)
  run_label_gate_only.py               # Gate-only çalıştırıcı (Kozan: legacy; diğerleri: namespaced)
  run_predictors_only.py               # Step3-Step5/5C predictor çalıştırıcı (Kozan: legacy; diğerleri: namespaced)
  check_experiment_registry.py         # Step0 kayıt defteri doğrulama script'i

docs/
  experiments.md                       # Step0 deney/bölge kaydı dokümantasyonu
  label_gate.md                        # Step6 label export cleanup + Step6B gate detayları
  aoi_refinement.md                    # Manavgat AOI geometrisi + gate-only readiness
  manavgat_gate_only.md                # Namespaced Manavgat gate-only akışı

data/       # Yerel ham/indirilmiş veri klasörleri
outputs/    # Her step'in ürettiği raster, tablo ve rapor çıktıları
  experiments/<experiment_id>/         # Deney-farkında (namespaced) çıktılar (şu an: Manavgat gate-only)
logs/       # Runtime log dosyaları (kod-seviyesi + shell tee çıktıları); .gitignore'da
```

`core/`, tüm step dosyalarının paylaştığı sabitleri ve yardımcı fonksiyonları barındırır; `src/`, her bir pipeline adımının çalıştırılabilir mantığını içerir; `scripts/`, uçtan uca çalıştırma ve tek seferlik yardımcı/gate-only işlemlerini içerir; `docs/`, Step0/gate ile ilgili detaylı dokümantasyonu içerir.

## 14. Ana Çıktılar

**Step4B:**
- `outputs/step4b/geotiff_validation_summary.json`
- `outputs/step4b/geotiff_validation_summary.md`

**Step5:**
- `outputs/step5/anomaly_zscore.tif`

**Step5C:**
- `outputs/step5c/current_tvdi.tif`
- `outputs/step5c/tvdi_difference.tif`

**Step6 (canonical label export, Kozan legacy yolu):**
- `outputs/validation/labels/mcd64a1_raw.tif`
- `outputs/validation/labels/mcd64a1_burned.tif`

**Step6B (burned-landcover gate):**
- `outputs/validation/labels/burned_landcover_gate.{json,md,csv}` (Kozan)
- `outputs/experiments/manavgat_2021/validation/labels/burned_landcover_gate.{json,md,csv}` (Manavgat)

**Step7:**
- `outputs/step7d/downscaled_lst_celsius.tif`
- `outputs/step7e/fused_lst_celsius.tif`

**Step8 (Kozan):**
- `outputs/step8a/step8a_500m_modeling_dataset.parquet`
- `outputs/step8b/step8b_model_comparison_metrics.json`
- `outputs/step8b/step8b_predictions.parquet`
- `outputs/step8c/step8c_bootstrap_metrics.json`
- `outputs/step8d/step8d_ablation_metrics.json`
- `outputs/step8e/step8e_summary.md`

## 15. Sınırlamalar

- **Kozan, primary wildfire AOI'si değildir.** Kozan 2023 burned label'larının büyük çoğunluğu (533/542) cropland/anız-yakma kaynaklıdır; Kozan cropland/anız-dominant bir negative/control AOI'dir. Doğal bitki örtüsü yangın davranışı için sonuçlar Kozan'a değil, Manavgat modellemesi tamamlandıktan sonraki bulgulara dayanmalıdır.
- **Manavgat suitability gate'i geçti, ama tam modelleme henüz yapılmadı.** Step6B, Manavgat 2021'in `wildfire_candidate_pass` olduğunu doğrulamıştır; ancak Step1-Step8 tam modelleme zinciri Manavgat için henüz çalıştırılmamıştır. Bir model sonucu iddia edilmemektedir.
- **Tek sezon/yıl (Kozan için):** Kozan analizi 2023 sezonunu kapsar; year-to-year robustness henüz test edilmemiştir.
- **Cropland-excluded burnable mask:** Bu maske içinde pozitif örnek sayısı azdır, bu da bu strata için istatistiksel gücü sınırlar (Kozan).
- **Ekim ayı düşük pozitif sayısı:** Ekim'e ait sonuçlar düşük örneklem nedeniyle düşük güvenle yorumlanmalıdır (Kozan).
- **Günlük MODIS gap-fill yok:** Mevcut fusion mantığı (Step7E) bir defalık, statik bir birleştirmedir; günlük operasyonel bir veri akışı değildir.
- **Multi-year ve uluslararası transfer (Valencia/Zamora) henüz yapılmadı.** Bu deneyler şu an disabled placeholder olarak kayıtlıdır.
- **3B / operasyonel dijital ikiz katmanı henüz yoktur.**

## 16. Sonraki Adımlar

1. **Manavgat Step1-Step5/5C deney-farkında predictor üretimi:** `scripts/run_predictors_only.py` ile Manavgat AOI'si için Landsat LST/NDVI current+baseline export edip termal anomaly + TVDI ürünlerini üretmek (namespaced, experiment-aware; araç hazır — bkz. Bölüm 12 — ancak gerçek GEE export'u henüz çalıştırılmadı).
2. **Manavgat Step7 downscaling/fusion (gerekirse):** Kozan'da kullanılan MODIS→Landsat downscaling/fusion mantığını Manavgat için gerektiği kadar tekrarlamak.
3. **Manavgat Step8A-E modelleme:** Label-honest 500 m dataset'i oluşturup baseline vs. thermal karşılaştırmasını, bootstrap belirsizliğini ve ablation'ı Manavgat için çalıştırmak.
4. **Kozan negative-control vs. Manavgat wildfire anchor karşılaştırması:** İki AOI'nin sonuçlarını yan yana koyup thermal feature katkısının doğal bitki örtüsü yangınında da geçerli olup olmadığını değerlendirmek.
5. **Valencia/Castellón veya Zamora external validation eklemek.**
6. **Sonrasında leave-one-region-out transfer testi.**

## 17. Terminoloji / Claim Policy

Bu README ve proje çıktıları, aşağıdaki ifade politikasına uyar:

- "Fire-risk prediction model validated" denmez.
- "Statistically significant" denmez.
- "TVDI alone predicts fire risk" denmez.
- **Kozan'ın doğal bitki örtüsü (orman/makilik) yangın davranışını doğruladığı iddia edilmez.** Doğru ifade: *"Kozan serves as a cropland/stubble-dominated negative control; Manavgat passed the burned-landcover gate as a natural-vegetation wildfire AOI."*
- **Manavgat için bir model sonucu olduğu iddia edilmez.** Şu ana kadar yalnızca gate/suitability sonucu (`wildfire_candidate_pass`) vardır; Step1-Step8 modelleme sonucu değildir.
- Doğru ifade (Kozan Step8 sonucu için): *"Thermal features provide measurable, spatial-block-bootstrap-supported improvement over static baseline features in the current Step8 experiment (Kozan 2023, cropland-dominant negative-control AOI)."*
- FIRMS hiçbir bağlamda bir target olarak sunulmaz.
- MCD64A1'in native ~500 m label çözünürlüğü korunur; hiçbir yerde 30 m piksel bazlı bir label hassasiyeti iddia edilmez.

## Sonuç

Kozan üzerindeki Step8 çekirdek deneyi tamamlanmıştır ve metodolojinin (label-honest 500 m grid + spatial-block CV + bootstrap + ablation) çalıştığını göstermiştir. Ancak Kozan'ın cropland-dominant doğası ortaya çıktıktan sonra, bu metodolojiyi doğal bitki örtüsü bir wildfire AOI'sinde tekrarlamak önceliklidir. Step0 deney kaydı ve Step6B burned-landcover gate, bu genişlemeyi güvenli bir şekilde yönetmek için eklenmiştir: Manavgat 2021, gate'i `wildfire_candidate_pass` ile geçmiş ve artık bir sonraki anchor wildfire AOI'sidir — ancak bir model sonucu henüz yoktur. Bir sonraki öncelik, yeni bir tekil feature'ı cilalamak değil, Manavgat için Step1-Step8 zincirini çalıştırıp Kozan'ın negative-control bulgusuyla karşılaştırmaktır.