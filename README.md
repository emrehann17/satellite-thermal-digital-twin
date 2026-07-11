# Uydu Tabanlı Termal Veri İşleme ve Burned-Area Modelleme Prototipi

## 1. Kısa Özet

Bu proje, Doğu Akdeniz bölgesinde MODIS, Landsat, NDVI, DEM (elevation/slope), landcover ve MCD64A1 burned-area verilerini işleyen çok adımlı bir uydu tabanlı termal veri işleme pipeline'ıdır. Pipeline, Google Earth Engine (GEE) üzerinden veri hazırlığıyla başlar; rasterio tabanlı local preprocessing, GeoTIFF doğrulama, termal anomaly ve dryness ürünleri üretimi ile devam eder; label-honest bir modeling/validation aşamasında sonlanır ve artık bunun ötesinde **iki bağımsız Akdeniz yangın bölgesi arasında bir cross-region transfer değerlendirmesi (Step9A-D)** ile **bu transferin neden sınırlı kaldığını teşhis eden bir post-hoc dağılım-kayması denetimi (Step9E)** içerir. Yani proje yalnızca raster üretimi yapan bir veri işleme hattı değil, aynı zamanda kendi ürettiği verileri istatistiksel olarak test eden ve genelleme sınırlarını açıkça belgeleyen bir doğrulama altyapısıdır.

Projenin bilimsel amacı nettir: termal ve dryness kökenli feature'ların (LST anomaly, TVDI, downscaled/fused LST) statik baseline feature'lara (elevation, slope, landcover, NDVI) kıyasla burned-area discrimination'a ölçülebilir bir katkı sağlayıp sağlamadığını test etmek — hem tek bir bölge içinde, hem de bölgeler arasında bu katkının genellenip genellenmediğini sınamak.

Pipeline ilk olarak Kozan 2023 AOI'sinde çalıştırılmış ve Step8 çekirdek deneyi bu AOI üzerinde tamamlanmıştır. Ancak sonrasında yapılan bir supervisor değerlendirmesi, Kozan 2023'teki MCD64A1 burned label'larının büyük çoğunluğunun (533/542 hücre, bkz. Bölüm 5) cropland/anız-yakma kaynaklı olduğunu, doğal bitki örtüsü (orman/makilik) yangını olmadığını ortaya koymuştur. Bu nedenle **Kozan, projenin doğal-bitki-örtüsü yangın modelinin kanıtı olarak sunulmaz; cropland/anız-dominant bir negative/control AOI olarak konumlandırılmıştır.**

Bu değerlendirmenin ardından proje, tek-AOI'den çoklu-deney (multi-experiment) bir yapıya geçmiştir: **Step0 deney/bölge kayıt defteri** (`core/regions.py`), her deneyi (region + yıl + predictor/label pencereleri + baseline yılları + rol + çıktı namespace'i) tek bir yerden yönetir. Kayıtlı deneyler: `kozan_2023` (negative_control), `manavgat_2021` (anchor_wildfire, ilk doğal-bitki-örtüsü wildfire AOI'si) ve `bejis_2022` (mediterranean_transfer_wildfire, İspanya'daki karşılaştırılabilir ikinci Akdeniz wildfire AOI'si), artı ileride kullanılmak üzere disabled `zamora_2022` placeholder'ı.

**Manavgat 2021 ve Bejís 2022 için artık tam bir modelleme zinciri tamamlanmıştır:** her iki AOI de Step6B burned-landcover gate'ini `wildfire_candidate_pass` kararıyla geçmiş, deney-farkında (experiment-aware) predictor üretimi, Step7 downscaling/fusion ve Step8A-E burned-area association modellemesi her iki bölge için de bağımsız olarak çalıştırılmıştır. Bunun üzerine, bu iki bağımsız modelin birbirine ne kadar transfer ettiği **Step9A-D** ile değerlendirilmiş, ve bu transferin sınırlı kalmasının olası nedenleri **Step9E post-hoc dağılım-kayması denetimi** ile teşhis edilmiştir (bkz. Bölüm 9 ve 10). Sonuç, aşağıda dikkatle ifade edildiği gibi, **doğrudan cross-region discrimination generalization'ının desteklenmediği**, ancak thermal feature'ların probability-error (Brier) üzerinde tutarlı bir iyileşme sağladığı yönündedir.

Bu çalışmanın ne olduğu konusunda net olmak önemlidir: bu proje tamamlanmış bir operasyonel yangın erken uyarı sistemi değildir ve tamamlanmış bir 3B dijital ikiz de değildir. Şu anki hâliyle proje; termal preprocessing, MODIS→Landsat downscaling/fusion, label-honest bir burned-area modeling altyapısı (üç bağımsız AOI üzerinde tamamlanmış), bu altyapının iki bölge arasında ne kadar genellendiğinin dürüst bir ölçümü, ve bu ölçümün neden sınırlı kaldığını açıklamaya çalışan bir post-hoc teşhis katmanıdır. Sonraki adımlar (bkz. Bölüm 18) transfer-safe bir feature stratejisinin keşifsel olarak tasarlanmasına ve üçüncü bağımsız bir bölgede doğrulanmasına yöneliktir.

## 2. Bu Proje Ne Değildir?

- **Tamamlanmış bir 3B dijital ikiz değildir.** Şu an yalnızca 2B raster katmanları ve tablo bazlı modeling dataset'leri üretilmektedir; 3B görselleştirme veya simülasyon katmanı yoktur.
- **Operasyonel bir yangın erken uyarı sistemi değildir.** Henüz multi-year ve near-real-time üretim akışı yoktur.
- **Kozan, doğal-bitki-örtüsü yangın modelinin kanıtı değildir.** Kozan 2023 burned label'ları cropland/anız-yakma dominanttır (bkz. Bölüm 5); Kozan bir negative/control AOI'dir, wildfire validation AOI'si değildir.
- **Başarılı bir cross-region transfer modeli değildir.** Step9A-D, Manavgat↔Bejís arasında doğrudan cross-region discrimination generalization'ının desteklenmediğini göstermiştir; yalnızca probability-error (Brier) iyileşmesi tutarlıdır (bkz. Bölüm 9). Bu, "başarılı transfer" olarak sunulmaz.
- **Step9E bir düzeltme/iyileştirme değildir.** Step9E, post-hoc bir teşhis analizidir; hiçbir modeli yeniden eğitmez, orijinal Step9 sonucunu değiştirmez ve "corrected transfer performance" iddia etmez (bkz. Bölüm 10).
- **Tek başına TVDI'ye dayalı bir fire-risk modeli değildir.** TVDI, Step8'deki multi-feature model içinde yardımcı bir thermal/dryness göstergesi olarak değerlendirilmiştir; tek başına bir karar mekanizması olarak sunulmamaktadır.
- **FIRMS, target olarak kullanılmaz.** FIRMS yalnızca independent bir active-fire cross-check katmanıdır; hiçbir aşamada MCD64A1 ile OR-combine edilerek primary label'a dahil edilmez.
- **30 m çözünürlükte piksel bazlı bir MCD64A1 modeli değildir.** Step8 (ve Step6B gate, ve Step9A-E), MCD64A1'in native ~500 m grid'ini kullanır; 30 m predictor'lar bu grid'e block-mode aggregation ile indirgenir.

## 3. Güncel Durum Özeti

| Bölüm | Durum | Kısa açıklama |
|---|---|---|
| Step0 | Tamamlandı | Deney/bölge (experiment/region) kayıt defteri ve namespaced çıktı yapısı |
| Step1-4B | Tamamlandı (Kozan) | GEE veri hazırlığı, export/download, GeoTIFF validation (legacy Drive zinciri) |
| Gate pipeline (Step6A+Step6B) | Tamamlandı | Kozan, Manavgat ve Bejís için uygulanabildiği şekilde tamamlandı |
| Kozan gate | Tamamlandı | `cropland_dominated_control` |
| Manavgat gate | Tamamlandı | `wildfire_candidate_pass` |
| Bejís gate | Tamamlandı | `wildfire_candidate_pass` |
| Deney-farkında (experiment-aware) predictors | Tamamlandı | Manavgat ve Bejís için tamamlandı (direkt/tiled local GEE indirme + Step5/Step5C) |
| Step7 (A-E) | Tamamlandı | Kozan (legacy), Manavgat ve Bejís (experiment-aware) için tamamlandı |
| Step8 (A-E) | Tamamlandı | Kozan (legacy), Manavgat ve Bejís (experiment-aware) için tamamlandı |
| Step9A | Tamamlandı | Manavgat↔Bejís cross-region girdi uygunluk denetimi |
| Step9B | Tamamlandı | Manavgat↔Bejís iki yönlü cross-region transfer |
| Step9C | Tamamlandı | Hedef-bölge spatial-block bootstrap güven aralıkları |
| Step9D | Tamamlandı | Birleşik cross-region final raporu |
| Step9E | Tamamlandı | Post-hoc dağılım-kayması / ilişki-kayması denetimi |
| `scripts/main.py` (deney-farkında CLI) | Tamamlandı | `experiment` / `transfer` / `shift-audit` / `legacy` alt-komutları (bkz. Bölüm 14) |
| Üçüncü bağımsız bölge (external validation) | Planlandı | `zamora_2022` disabled placeholder; bkz. Bölüm 18 |

Step8A-E'nin üç bağımsız AOI'de (Kozan negative-control, Manavgat anchor wildfire, Bejís transfer wildfire) tamamlanmasıyla projenin metodolojik çekirdeği doğrulanmıştır: thermal feature'ların burned-area discrimination'a katkısı, label-honest bir 500 m grid ve spatial-block CV ile her bölgede bağımsız olarak ölçülmüştür. Bunun üzerine, iki wildfire AOI'si (Manavgat, Bejís) arasında bu katkının ne kadar transfer ettiği Step9A-D ile test edilmiş, ve sonuç dikkatle yorumlanmıştır: **doğrudan cross-region discrimination generalization'ı desteklenmemiştir**, ancak Brier (probability-error) iyileşmesi tutarlıdır. Step9E, bu sınırlı transferin olası nedenlerini (feature dağılım kayması, olasılık ölçek kayması, bölgeye-bağlı feature-label ilişki kayması) post-hoc olarak teşhis etmiştir.

## 4. Experiment Registry ve AOI Rolleri

`core/regions.py`, iki ayrı kavramı birbirinden ayırır:

- **region** = yalnızca geometri (AOI).
- **experiment** = region + yıl + predictor penceresi + label penceresi + baseline yılları + rol (`role`) + çıktı namespace'i.

Kayıtlı deneyler:

| experiment_id | role | Durum |
|---|---|---|
| `kozan_2023` | `negative_control` | Enabled; cropland/anız-dominant control AOI, legacy Step1-Step8E zinciriyle tamamlandı |
| `manavgat_2021` | `anchor_wildfire` | Enabled; gate geçti, deney-farkında Step7-Step8 tamamlandı |
| `bejis_2022` | `mediterranean_transfer_wildfire` | Enabled; gate geçti, deney-farkında Step7-Step8 tamamlandı; Manavgat ile Step9 cross-region transfer çifti |
| `zamora_2022` | `hard_transfer_test` | Disabled placeholder; üçüncü bağımsız bölge / harder transfer test için |

Aktif deney, `core/regions.py:get_active_experiment(experiment_id)` ile çözülür. Her deneyin çıktıları kendi namespace'i altında toplanır:

```
outputs/experiments/<experiment_id>/
```

Kozan'ın legacy (namespace'siz) çıktı yolları (`outputs/step5/`, `outputs/validation/labels/`, vb.) değişmeden korunur — Kozan hâlâ bu legacy yolları kullanır (`scripts/main.py legacy` alt-komutu, bkz. Bölüm 14). Kozan-dışı deneyler (Manavgat, Bejís) tamamen `outputs/experiments/<experiment_id>/` altında çalışır ve legacy Kozan dosyalarına asla yazmaz/okumaz — bu, her runner'ın kendi çalışma-zamanı namespacing güvenlik kontrolüyle (`_assert_paths_are_safely_namespaced`) ve `scripts/main.py`'nin kendi ek kontrolüyle (`core/pipeline_orchestrator.py:_assert_context_is_safely_namespaced`) doğrulanır.

Cross-region (Step9A-E) çıktıları ayrı bir kökte toplanır:

```
outputs/cross_region/<source_experiment_id>__<target_experiment_id>/{step9a,step9b,step9c,step9d,step9e}/
```

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

| Experiment | Role | Decision | Yorum |
|---|---|---:|---|
| kozan_2023 | negative_control | cropland_dominated_control | Anız/cropland-dominant control (542 burned hücrenin 533'ü cropland) |
| manavgat_2021 | anchor_wildfire | wildfire_candidate_pass | Doğal bitki örtüsü wildfire AOI (796 burned hücrenin 783'ü tree/shrub/grass) |
| bejis_2022 | mediterranean_transfer_wildfire | wildfire_candidate_pass | Manavgat ile karşılaştırılabilir ikinci Akdeniz doğal-bitki-örtüsü wildfire AOI |

Bu gate **diagnostic**tir: `cropland_dominated_control` sonucu (Kozan'ın beklenen sonucu) pipeline'ı durdurmaz. Gate yalnızca raw BurnDate binary görünüyorsa, gerekli girdi rasterları eksikse, veya landcover class mapping çözülemiyorsa hata verir.

## 6. Pipeline Mimarisi: İki Ayrı İş Akışı

Proje artık iki BAĞIMSIZ iş akışı barındırır. Hangisinin kullanılacağı `scripts/main.py`'nin alt-komutuyla (`experiment` vs. `legacy`) belirlenir — hiçbiri diğerini sessizce çağırmaz.

### Deney-farkında (experiment-aware) iş akışı — YENİ varsayılan yol

```
Experiment registry (core/regions.py)
  -> ExperimentContext (core/experiment_context.py)
  -> [opsiyonel] Step6A gate-input hazırlama + Step6B burned-landcover gate
  -> direkt/tiled Earth Engine yerel indirme (Google Drive YOK)
  -> Step5 / Step5C
  -> Step7 (A-E)
  -> Step8 (A-E)
  -> [opsiyonel] Step9 cross-region transfer (Step9A-D)
  -> [opsiyonel] Step9E post-hoc dağılım-kayması denetimi
```

Bu iş akışının önemli özellikleri:

- **Hiçbir GEE batch Export task'i oluşturmaz.** `scripts/run_predictors_only.py --export`, current+baseline Landsat LST/NDVI'yı doğrudan yerel diske indirir (`getPixels`, gerekirse tiled fallback ile); Google Earth Engine "Tasks" panelinde görünen bir export işi YOKTUR.
- **Google Drive'ı ara adım olarak GEREKTİRMEZ.** Legacy Step4/Step4B'nin Drive export/download zincirini replike etmez.
- Manavgat ve Bejís için TÜM girdi/çıktı yolları `outputs/experiments/<experiment_id>/` altında namespaced'dır; legacy paylaşılan yollara (`outputs/step5/`, `outputs/validation/labels/`, vb.) asla yazmaz/okumaz.

### Legacy (Kozan) iş akışı — yalnızca geriye dönük uyumluluk

```
Kozan legacy Step1 (MODIS export)
  -> Step2 (5-yıllık MODIS baseline)
  -> Step3 (Landsat LST GEE hazırlığı)
  -> Step4 (GEE -> Google Drive batch export)
  -> Step4B (Drive -> yerel indirme + metadata-driven doğrulama)
  -> yerel işleme (Step5, Step5C, Step6, Step7, Step8 — legacy paylaşılan yollar)
```

Bu iş akışının önemli özellikleri:

- **Yalnızca `kozan_2023` için desteklenir** (`scripts/main.py legacy --experiment kozan_2023`).
- Legacy script'ler (`src/step1_fetch_modis.py` ... `src/step4b_download_drive_export.py`) SİLİNMEMİŞTİR, olduğu gibi korunur — yalnızca tarihsel Kozan reprodüksiyonu için kullanılır.
- **YENİ varsayılan giriş noktası (`scripts/main.py`, alt-komutsuz veya `experiment` alt-komutuyla) bu iş akışını SESSİZCE çalıştırmaz.** Legacy'ye erişim yalnızca açıkça `legacy` alt-komutuyla mümkündür.
- Legacy çıktı yolları (`outputs/step5/`, `outputs/step7a-e/`, `outputs/step8a-e/`, `outputs/validation/labels/`) DEĞİŞMEDEN korunur.

### Üçüncü katman (her iki iş akışı için ortak metodoloji)

Deney-farkında ve legacy iş akışları farklı veri erişim mekanizmaları kullansa da, aynı **label-resolution honesty** ve **spatial-block CV** prensiplerini paylaşır. Label-resolution honesty, MCD64A1'in gerçek çözünürlüğünün (~500 m) altında sahte bir hassasiyet üretilmemesi anlamına gelir — 30 m'lik bir predictor grid'i, MCD64A1'in native grid'ine nearest-neighbor ile "büyütülmüş" bir label ile eşleştirilirse, aynı 500 m hücresindeki onlarca 30 m piksel birbirinin kopyası bir label paylaşır; bu da pseudo-replication'a yol açar ve model performansını yapay şekilde şişirir. Step8A (ve aynı block/tile mantığını reuse eden Step6B gate) bu sorunu, 30 m predictor'ları feature tipine uygun özet istatistiklerle (sürekli feature'lar için mean/median, kategorik feature'lar için mode/fraction) 500 m MCD64A1 grid'ine indirgeyerek çözer.

Spatial-block CV kullanılmasının nedeni de benzer bir mantıktan gelir: komşu hücreler birbirine çok benzer environmental koşullara sahip olduğu için, random row-wise bir train/test split kullanılırsa aynı yangının komşu hücreleri hem train hem test setine sızabilir. Bu yüzden Step8B ve sonrasında (ve Step9B'nin kendi cross-region transfer değerlendirmesinde) `StratifiedGroupKFold`, mekânsal bloklara göre gruplanarak uygulanır.

Son olarak, Step8C ve Step8D bu modelleme sonucunun ne kadar güvenilir olduğunu sorgular; Step9C ve Step9E de aynı disiplini cross-region seviyesine taşır: Step9C hedef-bölge spatial-block bootstrap ile bir güven aralığı üretirken, Step9E hangi feature'ların/ilişkilerin bölgeler arasında kaydığını post-hoc olarak teşhis eder.

## 7. Step Açıklamaları

### Step1-4B: Veri hazırlığı, export ve doğrulama (yalnızca legacy Kozan iş akışı)

Bu aşama AOI'nin (Kozan) tanımlanmasıyla başlar; ardından MODIS, Landsat ve DEM verileri GEE üzerinden export edilir ve Google Drive üzerinden local diske indirilir. DEM'den elevation ve slope ürünleri türetilir. İndirilen tüm GeoTIFF dosyaları, metadata-driven bir doğrulama sürecinden geçer (Step4B). Bu aşama yalnızca legacy Kozan iş akışına aittir; Manavgat ve Bejís bu zinciri KULLANMAZ (bkz. Bölüm 6).

### Step5: Landsat LST anomaly

Step5, current period (predictor window) ile historical baseline (baseline yılları, aynı takvim penceresi) arasındaki Landsat yüzey sıcaklığı farkını hesaplar. Label window ile predictor window arasında bir temporal lead bırakılmıştır; bu, "pre-fire" bir tahmin senaryosu kurgusudur. Zamansal interpolasyon uygulanmaz — yalnızca gerçek gözlemler kullanılır. Çıktı olan `anomaly_zscore.tif`, tek başına bir fire-risk modeli değil, bir thermal anomaly ürünüdür. Deney-farkında iş akışında bu adım `core/experiment_context.py`'den gelen namespaced yollarla, legacy Kozan'da ise `core/config.py`'nin sabit yollarıyla çalışır.

### Step5C: TVDI / dryness ürünleri

TVDI (Temperature-Vegetation Dryness Index), LST-NDVI ilişkisinden yararlanarak yüzey kuruluğunu normalize etmeye çalışan bir indekstir. `current_tvdi`, `tvdi_difference` ve `tvdi_anomaly_zscore` ürünleri hesaplanır. TVDI, tek başına güçlü bir fire-risk predictor'ü olarak sunulmaz; Step8D ablation sonucunda TVDI grubunun pozitif ama ana sürücü olmadığı, Step9E'de ise `current_tvdi_mean`'in bölgeler arası nispeten daha stabil, ama `tvdi_difference_mean`'in belirgin şekilde kaydığı görülmüştür (bkz. Bölüm 10).

### Step6: Burned-area association diagnostics + canonical raw BurnDate export

Step6 bir model eğitmez; MCD64A1'i primary burned-area label olarak alıp, önceki adımlarda üretilen tekil indekslerin (özellikle TVDI) bu label ile ne kadar ilişkili olduğunu diagnostic olarak test eder. FIRMS burada da target değildir, yalnızca independent bir active-fire cross-check'tir. Ayrıca Step6, canonical raw MCD64A1 BurnDate export'unun tek sahibidir (`export_raw_mcd64a1_labels()`): gerçek BurnDate DOY değerlerini (1..366) `mcd64a1_raw.tif`'e, isteğe bağlı binary maskeyi `mcd64a1_burned.tif`'e yazar. Deney-farkında çağrıldığında bu, Kozan'ın legacy paylaşılan dosyalarına dokunmadan, herhangi bir deneyin namespaced çıktı dizinine yazabilir.

### Step6A: Gate-only girdi hazırlama (deney-farkında)

`src/step6a_prepare_gate_inputs.py`, Step6B gate'in ihtiyaç duyduğu iki minimum rasteri, tam Step3/Step5 termal pipeline'ını çalıştırmadan hazırlar: (1) AOI'yi kaplayan sabit-değerli bir 30 m referans grid rasteri (bir termal predictor DEĞİLDİR, yalnızca grid geometrisi için), ve (2) ESA WorldCover v200 landcover'ının bu referans gride hizalanmış kopyası. Tüm çıktılar `outputs/experiments/<experiment_id>/gate_inputs/` altına, namespaced olarak yazılır.

### Step6B: Burned-landcover gate

Bkz. Bölüm 5. `src/step6b_burned_landcover_gate.py`, Step8A'nın aynı 500 m block/tile mantığını reuse ederek MCD64A1-burned hücrelerin landcover kompozisyonunu özetler ve `wildfire_candidate_pass` / `cropland_dominated_control` / `insufficient_burned_positives` / `mixed_or_uncertain` kararlarından birini verir.

### Step7A-E: Downscaling ve fusion

Step7A, sonraki adımlarda kullanılacak tiling/windowed işleme altyapısını kurar. Step7B, MODIS→Landsat LST downscaling için temiz bir eğitim dataset'i hazırlar; bu dataset'te herhangi bir fire label bulunmaz. Step7C, bu dataset üzerinde saf bir MODIS→Landsat LST downscaling modeli eğitir; leakage guard mekanizması, anomaly/TVDI/z-score gibi türetilmiş feature'ların eğitime sızmasını engeller. Step7D, eğitilen modeli tüm raster grid'e windowed tiling ile uygulayarak full-grid downscaled bir LST üretir. Step7E, gözlemlenen Landsat LST ile Step7D'nin downscaled çıktısını deterministik ve observed-priority bir mantıkla birleştirir. **Her deney kendi bağımsız Step7 zincirini çalıştırır — Step7 modelleri bölgeler arasında transfer/retrain EDİLMEZ**; Manavgat ve Bejís'in her biri kendi MODIS/Landsat verisiyle kendi downscaling modelini eğitir.

### Step8A-E: Çekirdek burned-area modelleme deneyi

**Step8A — Label-honest 500 m modeling dataset:** 30 m çözünürlükteki predictor raster'ları, MCD64A1'in native/reconstructed ~500 m grid'ine spatial aggregation ile indirgenir (sürekli feature'lar için mean/median, kategorik feature'lar için mode/fraction). Bu adım, gerçek MCD64A1 BurnDate DOY verisini gerektirir; yalnızca binary (0/1) bir burned mask yeterli değildir.

**Step8B — Baseline vs. thermal model:** Baseline model yalnızca elevation, slope, landcover ve NDVI kullanır. Thermal-augmented model bunlara ek olarak LST anomaly, current LST, current TVDI, TVDI difference ve downscaled/fused LST feature'larını ekler. İki model de spatial-block `StratifiedGroupKFold` CV ile karşılaştırılır; sonuçlar `delta_AUC` ve `delta_PR-AUC` birlikte raporlanır.

**Step8C — Spatial-block bootstrap uncertainty:** Step8B'nin ürettiği out-of-fold tahminleri, yeniden eğitim yapılmadan, spatial-block bootstrap ile yeniden örneklenir; bu bir bootstrap percentile interval'dır, klasik bir p-value değildir.

**Step8D — Thermal feature ablation:** Thermal feature'lar gruplara ayrılarak hangi grubun performans farkına ne kadar katkı sağladığı test edilir.

**Step8E — Final rapor:** Step8B, Step8C ve Step8D'nin sonuçlarını yeniden eğitim yapmadan tek bir özet raporda birleştirir.

Bu zincir, `kozan_2023` için legacy paylaşılan yollarla, `manavgat_2021` ve `bejis_2022` için ise `scripts/run_step8_modeling.py` (deney-farkında çalıştırıcı, `core/pipeline_orchestrator.py` üzerinden `scripts/main.py experiment` tarafından da dispatch edilir) ile namespaced olarak çalıştırılmıştır. Üç deneyde de aynı kısıtlar korunur: MCD64A1 tek target'tır (FIRMS asla target değildir), 30 m pikseller asla modelleme örneği olarak kullanılmaz, spatial-block CV rastgele satır-bazlı split ile asla değiştirilmez.

### Step9A-D: Cross-region transfer değerlendirmesi (Manavgat ↔ Bejís)

Step9, Step8'in burned-area association modelini (baseline + baseline+thermal) BİR bölgede eğitip TAMAMEN BAĞIMSIZ bir bölgede test eder — hem Manavgat→Bejís hem Bejís→Manavgat yönünde. Tüm ön-işleme (numeric median imputation, kategorik landcover encoder) YALNIZCA kaynak (source) bölgeden fit edilir; hedef (target) bölgenin etiketleri ön-işlemeyi, eşik seçimini veya fit'i hiçbir şekilde etkilemez (pooled source+target fit yoktur, target fine-tuning/calibration yoktur). Sınıflandırma eşiği yalnızca kaynak bölgenin kendi spatial-block CV out-of-fold tahminlerinden seçilir.

- **Step9A** — iki deneyin Step8A veri setlerinin gerçekten karşılaştırılabilir olup olmadığını (shared feature'lar, populasyon yeterliliği, label kaynağı, gate kararı) fail-fast olarak denetler.
- **Step9B** — iki yönlü transferi çalıştırır, hedef bölgede baseline ve thermal model tahminlerini üretir.
- **Step9C** — Step9B'nin tahminlerini yeniden eğitim yapmadan, hedef-bölgenin spatial-block'larını yeniden örnekleyerek (%95 percentile) bootstrap güven aralığı üretir.
- **Step9D** — Step9A-C'nin sonuçlarını birleştirip iki yönlü bir final rapor üretir.

Sonuçlar ve dikkatli yorumlama için bkz. Bölüm 9. Bu **30 m'lik bir yangın tahmin modeli DEĞİLDİR, operasyonel bir yangın tespit sistemi DEĞİLDİR**, ve Step7 downscaling modelinin kendisini transfer ETMEZ.

### Step9E: Post-hoc dağılım-kayması / ilişki-kayması denetimi

Step9E, Step9B/Step9C'nin neden discrimination'ı koruyamadığını POST-HOC olarak teşhis eder. Hiçbir modeli yeniden eğitmez, Step9B tahminlerini/Step9C bootstrap çıktılarını değiştirmez, raporlanan Step9 sonucunu değiştirmez. Detaylar için bkz. Bölüm 10.

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

**Önemli uyarı:** Bu sonuçlar bootstrap percentile interval'dır; klasik bir p-value değildir. "İstatistiksel olarak anlamlı" (statistically significant) veya "significantly improves" ifadeleri kullanılmaz. Doğru ifade: thermal feature'lar, Kozan üzerindeki Step8 deneyinde statik baseline'a göre ölçülebilir ve spatial-block-bootstrap ile desteklenen bir iyileşme sağlamaktadır. Bu sonuç cropland-dominant bir AOI'de elde edilmiştir; doğal bitki örtüsü yangın davranışına genellenmesi, Manavgat ve Bejís'in kendi within-region Step8 sonuçlarına ve Bölüm 9'daki cross-region değerlendirmesine dayanmalıdır.

Manavgat ve Bejís için de aynı Step8A-E zinciri (namespaced, deney-farkında) bağımsız olarak tamamlanmıştır; bu iki bölgenin within-region Step8 çıktıları `outputs/experiments/manavgat_2021/step8*/` ve `outputs/experiments/bejis_2022/step8*/` altında mevcuttur (bkz. Bölüm 16). Bu README, iki wildfire bölgesi arasındaki asıl karşılaştırmayı (Step9) Bölüm 9'da ayrı olarak raporlar; buradaki sayısal özet yalnızca Kozan'ın ilk çekirdek deneyine aittir.

## 9. Cross-Region Transfer Sonuçları (Step9A-D)

Step9, Manavgat 2021 (anchor wildfire) ile Bejís 2022 (mediterranean transfer wildfire) arasında, Step8'in burned-area association modelini (baseline + baseline+thermal) bir bölgede eğitip diğerinde test ederek genelleme kapasitesini ölçer. Birincil (primary) popülasyon `burnable_tree_shrub_grass`'tır.

**Manavgat → Bejís:**

| Metrik | Baseline | Thermal | Delta |
|---|---:|---:|---:|
| ROC-AUC | 0.3322 | 0.3258 | -0.0064 |
| PR-AUC | 0.04928 | 0.04876 | -0.00053 |
| Brier | — | — | -0.03617 |

- Discrimination delta (ROC-AUC, PR-AUC) bootstrap güven aralığı: **uncertain** (sıfırı kapsıyor)
- Brier iyileşmesi: **positive bootstrap support**

**Bejís → Manavgat:**

| Metrik | Baseline | Thermal | Delta |
|---|---:|---:|---:|
| ROC-AUC | 0.4209 | 0.4435 | +0.0226 |
| PR-AUC | 0.03319 | 0.03441 | +0.00122 |
| Brier | — | — | -0.01865 |

- Discrimination delta (ROC-AUC, PR-AUC) bootstrap güven aralığı: **uncertain** (sıfırı kapsıyor)
- Brier iyileşmesi: **positive bootstrap support**

**Sonuç ifadesi (bu README'nin resmi yorumu):**

> "Direct cross-region discrimination generalization was not supported by the current two-region Step9 experiment. Thermal predictors consistently reduced Brier error, but ROC-AUC and PR-AUC improvements were uncertain or negative under target-region spatial-block bootstrap."

Bu, **başarılı bir transfer olarak sunulmaz.** Step9D'nin makine-okunabilir çıktısındaki `overall_conclusion = partial_transfer_supported` kategorisi, tekrar-üretilebilirlik (reproducibility) için olduğu gibi korunur ve DEĞİŞTİRİLMEZ; ancak bu README'nin insan-okunabilir yorumu, probability-error (Brier) iyileşmesini discrimination generalization'dan **bilerek ayırır**: Brier'in tutarlı iyileşmesi, modelin hedef bölgede "daha az yanlış-güvenli" olasılıklar ürettiğini gösterebilir, ama bu tek başına modelin yanma/yanmama ayrımını (ranking/discrimination) hedef bölgede güvenilir şekilde yaptığı anlamına gelmez — nitekim ROC-AUC/PR-AUC delta'ları belirsizdir (Manavgat→Bejís yönünde negatiftir).

Bu sonucun olası nedenleri Bölüm 10'da (Step9E) post-hoc olarak incelenmiştir.

## 10. Step9E: Post-Hoc Distribution-Shift Audit

Step9E, Step9B/Step9C'nin cross-region discrimination'ı neden koruyamadığını teşhis eden **post-hoc** bir analizdir:

- Step9A-D çıktıları yalnızca **salt-okunur girdi** olarak kullanılır; hiçbiri değiştirilmez.
- Hiçbir model yeniden eğitilmez.
- Hedef etiketler, YALNIZCA transfer değerlendirmesi tamamlandıktan SONRA, ilişki-kayması teşhisi için incelenir (modele geri-besleme / retroactive tuning yoktur).

**Diagnosis categories (Manavgat↔Bejís, primary population):**

- `high_shift`
- `probability_scale_shift`
- `ranking_reversal_suspected`
- `relationship_direction_instability`

**Global olarak en çok kayan primary-population feature'ları:**

`tvdi_difference_mean`, `downscaled_lst_mean`, `current_lst_mean`, `fused_lst_mean`, `lst_anomaly_mean`, `slope_mean`

**İlişki-yönü (relationship-direction) bulguları:**

- `elevation_mean`'in burned/unburned ilişkisi Manavgat ile Bejís arasında güçlü şekilde ters dönüyor.
- `current_lst_mean`, `downscaled_lst_mean` ve `fused_lst_mean`'in ilişkileri, primary population'da ters dönüyor.
- `tvdi_difference_mean` ters dönüyor.
- `current_tvdi_mean`, karşılaştırmalı olarak daha stabil.
- `lst_anomaly_mean`, genel olarak tutarlı bir negatif ilişki gösteriyor, ama global ölçeği (scale) güçlü şekilde kayıyor.

**Ranking-reversal örneği (Manavgat → Bejís, thermal model, primary population):**

- raw ROC-AUC ≈ 0.326
- diagnostic inverse ROC-AUC ≈ 0.674
- hedef hücrelerin yaklaşık %0,5'i kaynak-seçilmiş eşiğin üzerinde
- burned hedef hücrelerin **%0'ı** eşiğin üzerinde

Bu, bir **post-hoc ranking-reversal teşhisidir**. **Tahminler resmi sonuçta TERS ÇEVRİLMEZ**; inverse AUC yalnızca diagnostic amaçlıdır ve orijinal transfer sonucunu "onarmaz" (repair etmez).

**Bu bölümün ASLA iddia etmediği şeyler:** istatistiksel anlamlılık, nedensel açıklama, operasyonel yangın tahmini, başarılı cross-region transfer, düzeltilmiş (corrected) transfer performansı, veya her iki bölgenin etiketleri incelendikten sonra AYNI iki bölgede test edilip "unbiased" ilan edilen bir transfer-safe feature seti.

Kaynak kod ve tam çıktılar: `src/step9e_distribution_shift_audit.py`, `scripts/run_cross_region_shift_audit.py`, `outputs/cross_region/<source>__<target>/step9e/`.

## 11. Örnek Görseller

Şu an README'de final örnek görseller bulunmamaktadır. Önceki prototip haritalar geçici olarak kaldırılmıştır.

Bunun sebebi basittir: eski görseller, mevcut Step8/Step9 label-honest modelleme sonucunu doğrudan temsil etmeyebilir. Legend, CRS, ölçek, quality-mask ve final map layout konularında bir QA turu tamamlanmadan bu görsellerin README'de kalması yanıltıcı olabilirdi.

### Eklenecek görsel türleri

QA süreci tamamlandığında aşağıdaki ürünlerden örnekler README'ye eklenecektir:

- Landsat LST anomaly haritası
- Current TVDI haritası
- Downscaled LST haritası
- Fused LST haritası
- Step8 prediction / risk-score tarzı diagnostic harita
- Ablation / feature contribution grafiği
- Bootstrap güven aralığı grafiği
- Step9E feature-shift heatmap ve calibration eğrileri

### Görsel ekleme politikası

README'ye yalnızca QA'dan geçmiş, doğru legend/scale/CRS bilgisine sahip görseller eklenecektir. Diagnostic amaçlı ara görseller ile final haritalar birbirine karıştırılmayacak, ve herhangi bir görsel başlığı "fire-risk prediction" gibi bir izlenim yaratacak şekilde yazılmayacaktır.

<!-- TODO: Final visualization package will be added after map QA and layout cleanup. -->

## 12. Önemli Label Notu

Step6 döneminde, MCD64A1'den yalnızca binary (0/1) bir `mcd64a1_raw.tif` üretilebiliyordu. Ancak Step8A, gerçek BurnDate DOY bilgisini gerektirir; binary bir mask, label kalitesini ve zamansal tutarlılığı doğrulamak için yeterli değildir. Bu sorunu çözmek için Step6, canonical raw MCD64A1 BurnDate export'unun tek sahibidir (`export_raw_mcd64a1_labels()` fonksiyonu, `src/step6_validate_fire_relation.py` içinde). `scripts/export_mcd64a1_raw_burndate.py` script'i bu fonksiyonu çağıran ince bir CLI sarmalayıcı olarak kalmıştır.

Eğer export edilen raw dosya yalnızca 0/1 değerleri içeriyorsa (binary görünüyorsa), hem **Step8A** hem de **Step6B burned-landcover gate** bunu fail-fast diagnostics ile yakalar ve devam etmez.

```bash
python scripts/export_mcd64a1_raw_burndate.py --also-binary
```

## 13. Kurulum

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

Google Drive kimlik bilgileri (`.env`) **yalnızca `legacy` alt-komutu (Kozan'ın Step1-Step4B Drive zinciri) için** gereklidir. Deney-farkında (`experiment`) iş akışı — Manavgat, Bejís ve `--predictor-mode export` ile Kozan-dışı herhangi bir deney — Drive'a hiç dokunmaz; yalnızca `earthengine authenticate` yeterlidir.

Kısa örnek (yalnızca `legacy` alt-komutu kullanılacaksa):

```bash
cp .env.example .env
```

`.env` içinde:

```
GOOGLE_DRIVE_EXPORT_FOLDER_ID=YOUR_FOLDER_ID
GOOGLE_DRIVE_EXPORT_FOLDER_URL=https://drive.google.com/drive/u/0/folders/YOUR_FOLDER_ID
```

`.env.example`, doldurulacak bir şablon olarak kullanılır; gerçek `.env` dosyası repoya commit edilmez.

## 14. Çalıştırma

Tüm örnek komutlarda, kalıcı (shell `tee`) log çıktılarının repo kök dizinine değil `logs/` altına yazıldığına dikkat edin (`logs/` zaten `.gitignore`'dadır). Bu, `core/io_utils.py:setup_logger()` tarafından üretilen kod-seviyesi log dosyalarından (bunlar zaten otomatik olarak `logs/<step_adı>_<zaman_damgası>.log` konumuna yazılır) ayrı, yalnızca terminal çıktısını kalıcı hale getirmek isteyenler içindir.

**`scripts/main.py`, projenin canonical (tek, önerilen) giriş noktasıdır.** Alt-komut verilmeden çalıştırma yalnızca yardımı basar; hiçbir GEE/legacy iş akışını sessizce başlatmaz.

### Yardım

```bash
python scripts/main.py --help
python scripts/main.py experiment --help
python scripts/main.py transfer --help
python scripts/main.py shift-audit --help
python scripts/main.py legacy --help
```

### `experiment` — deney-farkında asama zinciri (gate → predictors → step7 → step8)

```bash
# Yalnızca planı + planlanan yolları/girdi durumunu bas (hiçbir GEE/eğitim çalıştırmaz)
python scripts/main.py experiment \
  --experiment manavgat_2021 --from-stage predictors --to-stage step8 \
  --predictor-mode local-only --dry-run

# Direkt/tiled predictor export (GEE, namespaced, Drive YOK) + Step7 + Step8
python scripts/main.py experiment \
  --experiment bejis_2022 --from-stage predictors --to-stage step8 \
  --predictor-mode export --force

# Yalnızca yerel yeniden çalıştırma (GeoTIFF'ler zaten var; GEE'ye dokunmaz)
python scripts/main.py experiment \
  --experiment manavgat_2021 --from-stage predictors --to-stage step8 \
  --predictor-mode local-only --force

# Gate'ten başlayarak tüm zinciri önizle
python scripts/main.py experiment \
  --experiment manavgat_2021 --from-stage gate --to-stage step8 \
  --predictor-mode local-only --dry-run
```

`--from-stage`/`--to-stage` asama sırası `gate -> predictors -> step7 -> step8`'dir; `--from-stage` `--to-stage`'ten sonra olamaz (fail-fast). `--predictor-mode export`, direkt/tiled GEE yerel indirmeyi açıkça tetikler; `--predictor-mode local-only` GEE/Drive'a HİÇ dokunmaz. `--export-labels` yalnızca `gate` asamasını etkiler. Her asama, `core/pipeline_orchestrator.py` üzerinden kendi mevcut namespaced runner'ına (`scripts/run_label_gate_only.py`, `scripts/run_predictors_only.py`, `scripts/run_step7_downscaling_only.py`, `scripts/run_step8_modeling.py`) dispatch edilir — hiçbir bilimsel mantık `main.py`'de yeniden uygulanmaz.

### `transfer` — cross-region Step9A-D

```bash
python scripts/main.py transfer \
  --source manavgat_2021 --target bejis_2022 --reverse --dry-run

python scripts/main.py transfer \
  --source manavgat_2021 --target bejis_2022 --reverse --force
```

`scripts/run_cross_region_transfer.py`'yi reuse eder; çıktı kökü `outputs/cross_region/<source>__<target>/`.

### `shift-audit` — Step9E post-hoc audit

```bash
python scripts/main.py shift-audit \
  --source manavgat_2021 --target bejis_2022 --dry-run

python scripts/main.py shift-audit \
  --source manavgat_2021 --target bejis_2022 --force
```

`scripts/run_cross_region_shift_audit.py`'yi reuse eder; Step9A-D'yi DEĞİŞTİRMEZ, hiçbir modeli yeniden eğitmez.

### `legacy` — yalnızca Kozan, Drive-tabanlı tam pipeline

```bash
python scripts/main.py legacy --experiment kozan_2023 --dry-run
python scripts/main.py legacy --experiment kozan_2023 --force
```

Bu, eski Step1-Step8E Drive-tabanlı zincirin BİREBİR aynısıdır (yalnızca `core/pipeline_orchestrator.py:run_legacy_kozan_pipeline()`'e taşınmıştır — `scripts/main.py`'yi ince tutmak için). `kozan_2023` dışında bir `--experiment` verilirse net bir hata ile reddedilir; bu deneyler için `experiment` alt-komutunu kullanın.

### Gelişmiş / doğrudan çalıştırıcılar (advanced/direct runners)

`scripts/main.py`, yukarıdaki tüm alt-komutlar için canonical giriş noktasıdır. Ancak her asamanın kendi doğrudan çalıştırıcı script'i de mevcuttur ve tek bir asamayı izole şekilde çalıştırmak/debug etmek için kullanılabilir:

```bash
# Yalnızca gate zinciri
python scripts/run_label_gate_only.py --experiment manavgat_2021 --export-labels --force

# Yalnızca predictor üretimi
python scripts/run_predictors_only.py --experiment bejis_2022 --export --force
python scripts/run_predictors_only.py --experiment manavgat_2021 --local-only --force

# Yalnızca Step7
python scripts/run_step7_downscaling_only.py --experiment manavgat_2021 --force

# Yalnızca Step8
python scripts/run_step8_modeling.py --experiment manavgat_2021 --force

# AOI önizleme (export/pipeline çalıştırmaz)
python scripts/preview_experiment_aoi.py --experiment bejis_2022

# Step0 kayıt defteri doğrulama
python scripts/check_experiment_registry.py

# DEM / MODIS hazırlığı (Kozan-dışı deneyler için gerektiğinde)
python scripts/prepare_dem_for_experiment.py --experiment manavgat_2021 --export --force
python scripts/prepare_modis_for_step7.py --experiment bejis_2022 --export --force
```

### Adım adım (Kozan, legacy, doğrudan step script'leri)

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

`--force` bayrağı, önceki çıktıları ezmek (overwrite) için kullanılır. `--force` verilmediğinde bazı adımlar mevcut çıktılar için fail-fast davranabilir veya yeniden üretimi reddedebilir; bu davranış adıma göre değişebilir. Bu yüzden tekrar çalıştırmalarda `--force` kullanılması önerilir.

## 15. Proje Yapısı

```
core/
  config.py                    # Merkezi konfigürasyon sabitleri (legacy Kozan)
  paths.py                     # Girdi/çıktı yol tanımları
  io_utils.py                  # Ortak okuma/yazma yardımcıları
  regions.py                   # Step0 deney/bölge (experiment/region) kayıt defteri
  experiment_context.py        # Step3-Step5/5C/7/8 için deney-farkında path/tarih context'i
  pipeline_orchestrator.py     # scripts/main.py'nin kullandığı asama sırası/dispatch katmanı
  validation_burned_area.py    # Burned-area doğrulama mantığı
  drive_downloader.py          # Legacy Google Drive indirme yardımcıları
  gee_utils.py                 # Ortak GEE yardımcıları
  utils/                       # geotiff_validation.py, tiling.py

src/
  step1_fetch_modis.py ... step4b_download_drive_export.py   # Legacy Kozan veri hazırlığı
  step5_preprocess_timeseries.py
  step5b_diagnostic_report.py
  step5c_tvdi.py
  step6_validate_fire_relation.py
  step6a_prepare_gate_inputs.py        # Gate-only referans grid + landcover hazırlığı (deney-farkında)
  step6b_burned_landcover_gate.py      # Burned-landcover gate
  step7a ... step7e_*.py               # Tiling, downscaling, fusion
  step8a_prepare_500m_modeling_dataset.py
  step8b_train_baseline_vs_thermal_model.py
  step8c_spatial_block_bootstrap_uncertainty.py
  step8d_thermal_feature_ablation.py
  step8e_final_report.py
  step9a_audit_cross_region_inputs.py       # Cross-region girdi uygunluk denetimi
  step9b_run_cross_region_transfer.py       # İki yönlü cross-region transfer
  step9c_cross_region_block_bootstrap.py    # Hedef-bölge spatial-block bootstrap
  step9d_build_cross_region_report.py       # Birleşik final cross-region raporu
  step9e_distribution_shift_audit.py        # Post-hoc dağılım-kayması/ilişki-kayması denetimi

scripts/
  main.py                              # Canonical deney-farkında CLI (experiment/transfer/shift-audit/legacy)
  export_mcd64a1_raw_burndate.py       # Raw BurnDate DOY export'u için ince CLI sarmalayıcı
  preview_experiment_aoi.py            # Deney metadata + AOI önizleme (export/pipeline çalıştırmaz)
  run_label_gate_only.py               # Gate-only çalıştırıcı (Kozan: legacy; diğerleri: namespaced)
  run_predictors_only.py               # Step3-Step5/5C predictor çalıştırıcı (Kozan: legacy; diğerleri: namespaced)
  run_step7_downscaling_only.py        # Deney-farkında Step7A-E çalıştırıcı
  run_step8_modeling.py                # Deney-farkında Step8A-E çalıştırıcı
  run_cross_region_transfer.py         # Step9A-D orkestratörü
  run_cross_region_shift_audit.py      # Step9E orkestratörü
  prepare_dem_for_experiment.py        # Kozan-dışı deneyler için namespaced DEM hazırlığı
  prepare_modis_for_step7.py           # Kozan-dışı deneyler için namespaced MODIS mean/std hazırlığı
  check_experiment_registry.py         # Step0 kayıt defteri doğrulama script'i
  run_prefire_experiment.py            # Pre-fire config doğrulama yardımcı script'i (legacy Kozan)
  standalone_step5-6.py                # Step5'i tek başına çalıştıran yardımcı script (legacy Kozan)

tests/
  test_pipeline_orchestrator.py        # Asama sırası, namespace güvenliği, dry-run no-execution testleri
  test_main_cli.py                     # scripts/main.py argparse/dispatch testleri

data/       # Yerel ham/indirilmiş veri klasörleri (.gitignore'da)
outputs/    # Her step'in ürettiği raster, tablo ve rapor çıktıları (.gitignore'da)
  experiments/<experiment_id>/         # Deney-farkında (namespaced) çıktılar (Manavgat, Bejís)
  cross_region/<source>__<target>/     # Step9A-E çıktıları
logs/       # Runtime log dosyaları; .gitignore'da
```

`core/`, tüm step dosyalarının paylaştığı sabitleri, yardımcı fonksiyonları ve `scripts/main.py`'nin kullandığı orkestrasyon katmanını barındırır; `src/`, her bir pipeline adımının (Step1-Step9E) çalıştırılabilir mantığını içerir; `scripts/`, uçtan uca çalıştırma, deney-farkında asama çalıştırıcıları ve tek seferlik yardımcı işlemleri içerir; `tests/`, orkestrasyon/CLI mantığı için odaklı unittest testlerini içerir.

## 16. Ana Çıktılar

**Step4B (legacy Kozan):**
- `outputs/step4b/geotiff_validation_summary.json`, `.md`

**Step5 / Step5C:**
- Kozan (legacy): `outputs/step5/anomaly_zscore.tif`, `outputs/step5c/current_tvdi.tif`, `tvdi_difference.tif`
- Manavgat / Bejís (namespaced): `outputs/experiments/<experiment_id>/step5/...`, `outputs/experiments/<experiment_id>/step5c/...`

**Step6B (burned-landcover gate):**
- `outputs/validation/labels/burned_landcover_gate.{json,md,csv}` (Kozan)
- `outputs/experiments/manavgat_2021/validation/labels/burned_landcover_gate.{json,md,csv}` (Manavgat)
- `outputs/experiments/bejis_2022/validation/labels/burned_landcover_gate.{json,md,csv}` (Bejís)

**Step7 / Step8:**
- Kozan (legacy): `outputs/step7d/downscaled_lst_celsius.tif`, `outputs/step7e/fused_lst_celsius.tif`, `outputs/step8a/step8a_500m_modeling_dataset.parquet`, `outputs/step8b/step8b_model_comparison_metrics.json`, `outputs/step8b/step8b_predictions.parquet`, `outputs/step8c/step8c_bootstrap_metrics.json`, `outputs/step8d/step8d_ablation_metrics.json`, `outputs/step8e/step8e_summary.md`
- Manavgat / Bejís (namespaced): aynı dosya adları, `outputs/experiments/<experiment_id>/step7*/` ve `outputs/experiments/<experiment_id>/step8*/` altında

**Step9A-D (cross-region transfer, Manavgat↔Bejís):**
- `outputs/cross_region/manavgat_2021__bejis_2022/step9a/cross_region_input_audit.{json,md}`
- `outputs/cross_region/manavgat_2021__bejis_2022/step9b/cross_region_transfer_metrics.json`, `cross_region_transfer_predictions.parquet`
- `outputs/cross_region/manavgat_2021__bejis_2022/step9c/cross_region_bootstrap_metrics.json`
- `outputs/cross_region/manavgat_2021__bejis_2022/step9d/final_cross_region_report.{json,md}`

**Step9E (post-hoc distribution-shift audit):**
- `outputs/cross_region/manavgat_2021__bejis_2022/step9e/distribution_shift_audit.json`
- `outputs/cross_region/manavgat_2021__bejis_2022/step9e/distribution_shift_summary.md`
- `outputs/cross_region/manavgat_2021__bejis_2022/step9e/numeric_feature_shift.csv`, `categorical_landcover_shift.csv`, `label_conditional_feature_relationships.csv`, `relationship_direction_flips.csv`, `prediction_distribution_audit.csv`, `calibration_bins.csv`
- Figürler: `feature_shift_heatmap.png`, `top_shifted_feature_distributions.png`, `label_conditional_direction_plot.png`, `landcover_distribution_comparison.png`, `prediction_probability_distributions.png`, `calibration_curves.png`

## 17. Sınırlamalar

- **Kozan, primary wildfire AOI'si değildir.** Kozan 2023 burned label'larının büyük çoğunluğu (533/542) cropland/anız-yakma kaynaklıdır; Kozan cropland/anız-dominant bir negative/control AOI'dir.
- **Cross-region discrimination generalization desteklenmemiştir.** Step9A-D, Manavgat↔Bejís arasında ROC-AUC/PR-AUC delta'larının belirsiz (bir yönde negatif) olduğunu göstermiştir; yalnızca Brier (probability-error) iyileşmesi tutarlıdır (bkz. Bölüm 9). Bu sonuç yalnızca İKİ bölge (Manavgat, Bejís) arasında elde edilmiştir.
- **Step9E teşhisi, iki bölgeye özgüdür ve nedensel değildir.** Step9E'nin bulguları (feature dağılım kayması, ilişki-yönü kayması, olasılık ölçek kayması) tanımlayıcı/betimsel diagnostiklerdir; hangi mekanizmanın ASIL nedensel sürücü olduğu kanıtlanmamıştır ve üçüncü bir bölge olmadan doğrulanamaz.
- **Step9E'nin önerdiği herhangi bir düzeltme, henüz test edilmemiş bir hipotezdir.** Region-relative/robust normalizasyon veya feature subset seçimi gibi fikirler, AYNI iki bölgede (etiketleri zaten incelenmiş) test edilip "unbiased" ilan edilemez; bunlar yeni, bağımsız bir deney olarak ele alınmalıdır (bkz. Bölüm 18).
- **Tek sezon/yıl (her üç AOI için):** Her deney tek bir yangın sezonunu kapsar; year-to-year robustness henüz test edilmemiştir.
- **Cropland-excluded burnable mask:** Bu maske içinde pozitif örnek sayısı azdır, bu da bu strata için istatistiksel gücü sınırlar (özellikle Kozan).
- **Günlük MODIS gap-fill yok:** Mevcut fusion mantığı (Step7E) bir defalık, statik bir birleştirmedir; günlük operasyonel bir veri akışı değildir.
- **Üçüncü bağımsız bölge (external validation) henüz yoktur.** `zamora_2022` şu an disabled placeholder olarak kayıtlıdır.
- **3B / operasyonel dijital ikiz katmanı henüz yoktur.**

## 18. Sonraki Adımlar

1. ~~**Deney-farkında `scripts/main.py` orkestrasyonunu tamamlamak ve doğrulamak.**~~ **Tamamlandı** — `experiment` / `transfer` / `shift-audit` / `legacy` alt-komutları, `core/pipeline_orchestrator.py` üzerinden mevcut namespaced runner'ları reuse eder; bkz. Bölüm 14 ve `tests/`.
2. **Açıkça keşifsel (exploratory) bir transfer-safe feature deneyi tasarlamak.** Step9E'nin teşhis ettiği kayan/ters-dönen feature'ları (bkz. Bölüm 10) hesaba katan, ama bu iki bölgede "doğrulanmış" olarak SUNULMAYACAK yeni bir deney.
3. **Region-relative veya robust normalizasyonu YENİ bir deney olarak test etmek.**
4. **Bölgeye-özgü mutlak LST ve elevation proxy'lerini dışlayan feature subset'lerini değerlendirmek.**
5. **Seçilen herhangi bir yaklaşımı, üçüncü bağımsız bir yangın bölgesinde test etmeden ÖNCE dondurmak (freeze).**
6. **Zamora veya başka bağımsız bir bölgeyi external evaluation için eklemek.**
7. **Ancak bundan SONRA, daha güçlü bir genelleme iddiası düşünmek.**
8. **3B/dijital-ikiz sunum çalışmasına, mevcut modeli operasyonel bir erken-uyarı sistemi olarak sunmadan devam etmek.**

## 19. Terminoloji / Claim Policy

Bu README ve proje çıktıları, aşağıdaki ifade politikasına uyar:

**İzin verilen ifadeler:**
- "Within-region thermal contribution was observed" (bkz. Bölüm 8, Kozan Step8 sonucu; Manavgat/Bejís kendi within-region Step8 çıktıları için de geçerlidir).
- "Direct cross-region discrimination was not supported" (bkz. Bölüm 9).
- "Brier error improved" / "probability-error improved" (bkz. Bölüm 9).
- "Step9E found diagnostic evidence consistent with domain shift and region-dependent feature-label relationships" (bkz. Bölüm 10).
- "A third independent region or nested design is needed for a stronger generalization claim" (bkz. Bölüm 17, 18).

**Yasak ifadeler:**
- "Operational wildfire prediction" — proje operasyonel bir sistem değildir (bkz. Bölüm 2).
- "Successful cross-region model" / "successful transfer" — Step9A-D sonucu bu değildir (bkz. Bölüm 9).
- "Statistically significant transfer" — tüm güven aralıkları bootstrap percentile interval'dır, klasik p-value değildir.
- "Causal fire prediction" — hiçbir aşamada nedensellik iddia edilmez.
- "Validated correction" — Step9E bir düzeltme değil, post-hoc bir teşhistir (bkz. Bölüm 10).
- "Transfer-safe feature set proven on Manavgat and Bejís after inspecting both labels" — bu, aynı iki bölgede etiketleri zaten görülmüş bir stratejiyi "kanıtlanmış" ilan etmek anlamına gelir ve yasaktır (bkz. Bölüm 17, 18).

**Ayrıca (önceki politikadan korunan maddeler):**
- "Fire-risk prediction model validated" denmez.
- "TVDI alone predicts fire risk" denmez.
- FIRMS hiçbir bağlamda bir target olarak sunulmaz.
- MCD64A1'in native ~500 m label çözünürlüğü korunur; hiçbir yerde 30 m piksel bazlı bir label hassasiyeti iddia edilmez.
- **Kozan'ın doğal bitki örtüsü (orman/makilik) yangın davranışını doğruladığı iddia edilmez.** Doğru ifade: *"Kozan serves as a cropland/stubble-dominated negative control; Manavgat and Bejís passed the burned-landcover gate as natural-vegetation wildfire AOIs."*

## Sonuç

Kozan üzerindeki Step8 çekirdek deneyi, metodolojinin (label-honest 500 m grid + spatial-block CV + bootstrap + ablation) çalıştığını göstermiştir. Bu metodoloji, doğal bitki örtüsü iki bağımsız wildfire AOI'sinde (Manavgat 2021, Bejís 2022) tekrarlanmış ve her iki bölgede de kendi within-region Step8 sonuçları üretilmiştir. Bunun üzerine, bu iki bölge arasında bir cross-region transfer değerlendirmesi (Step9A-D) yapılmış ve sonuç dikkatle raporlanmıştır: **doğrudan discrimination generalization'ı desteklenmemiştir, ama probability-error (Brier) iyileşmesi tutarlıdır.** Step9E, bu sınırlı transferin olası nedenlerini (feature dağılım kayması, olasılık ölçek kayması, bölgeye-bağlı feature-label ilişki kayması) post-hoc olarak teşhis etmiştir — bu bir düzeltme değil, bir sonraki deneyin nereye odaklanması gerektiğini gösteren bir haritadır.

Proje artık `scripts/main.py` üzerinden tek, canonical bir deney-farkında CLI'a sahiptir (`experiment` / `transfer` / `shift-audit` / `legacy`); legacy Kozan Drive-tabanlı iş akışı korunmuş ama varsayılan yol olmaktan çıkarılmıştır. Bir sonraki öncelik, Step9E'nin teşhis ettiği kayan feature'ları hesaba katan, açıkça keşifsel bir transfer-safe feature deneyi tasarlamak ve bunu üçüncü bağımsız bir bölgede (örn. Zamora) doğrulamaktır — mevcut iki bölgede "kanıtlanmış" ilan etmeden.