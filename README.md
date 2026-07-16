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
| Step0 | Tamamlandı | Deney/bölge kayıt defteri ve namespaced çıktı yapısı |
| Step1-4B | Tamamlandı (Kozan) | GEE veri hazırlığı, export/download, GeoTIFF validation |
| Gate pipeline (Step6A+Step6B) | Tamamlandı | Kozan, Manavgat ve Bejís |
| Kozan gate | Tamamlandı | `cropland_dominated_control` |
| Manavgat gate | Tamamlandı | `wildfire_candidate_pass` |
| Bejís gate | Tamamlandı | `wildfire_candidate_pass` |
| Deney-farkında predictors | Tamamlandı | Manavgat ve Bejís için local GEE indirme + Step5/Step5C |
| Step7 (A-E) | Tamamlandı | Kozan, Manavgat ve Bejís |
| Step8 (A-E) | Tamamlandı | Kozan, Manavgat ve Bejís |
| Step8 predefined large-block robustness | Tamamlandı | Frozen 10/20-hücre tasarımı; dört koşulda ROC-AUC ve PR-AUC delta aralıkları sıfırın üzerinde |
| Step9A-E | Tamamlandı | İki yönlü transfer, paired bootstrap, final rapor ve post-hoc shift audit |
| Step9F | Tamamlandı (kesifsel / post-hoc) | Sabit varyant paneli + region-relative adaptive rejim |
| Step10 | Tamamlandı | Preregistered, target-label-blind z-score/CORAL deneyi ve report-only QA |
| Step8 formal large-block robustness (all_valid) | Tamamlandı | Formal Step8B primary population `all_valid`; 10/20-hücre; 2-hücre equivalence gate ile doğrulandı |
| Step9G concept/relationship-shift | Tamamlandı | Univariate feature-AUC direction-reversal teşhisi (`burnable_tree_shrub_grass`) |
| Step9G canonical integration-v2 | Tamamlandı | Report-only; frozen Step9G sayılarını birebir kullanır, Step9E/9F/10 entegrasyonunu düzeltir |
| `scripts/main.py` canonical CLI | Tamamlandı | `experiment` / `transfer` / `shift-audit` / `transfer-explore` / `self-cal-transfer` / `step10` / `step8-robustness` / `large-block-robustness` / `concept-shift` / `legacy` |

**Bölge kapsamı (region scope):** Şu anda yalnızca `manavgat_2021` ve `bejis_2022` aktif bilimsel deneylerdir; `kozan_2023` ilgili yerlerde negatif kontrol (cropland-dominated control) olarak kullanılır. Zamora ve başka bölgeler şu an **dahil değildir**; bu prototip yeni bir bölge, model, feature araması veya adaptasyon yöntemi eklemez.

Step8A-E üç AOI üzerinde label-honest 500 m grid ve spatial-block CV ile tamamlandı. Manavgat ve Bejís için önceden belirlenmiş 10-hücre (~5 km) ve 20-hücre (~10 km) robustness koşulları da gerçek veride çalıştırıldı: formal Step8B primary population `all_valid` için dört bölge-ölçek koşulunun tamamında hem delta ROC-AUC hem delta PR-AUC percentile aralıkları sıfırın tamamen üzerindeydi. Bu yalnızca predefined within-region spatial-scale robustness bulgusudur; spatial autocorrelation'ın ortadan kalktığı veya cross-region transferin başarılı olduğu anlamına gelmez.

Step9A-D doğrudan cross-region discrimination generalization'ını desteklemedi. Step9E bu sınırlı transferin olası nedenlerini post-hoc teşhis etti. Step10 target-label-blind covariate adaptation'ı değerlendirdi; sonuçları yön/yöntem bazında raporlar ve Step9'u “düzeltilmiş” ilan etmez.

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

Cross-region Step9/Step10 çıktıları ayrı bir kökte toplanır:

```
outputs/cross_region/<source_experiment_id>__<target_experiment_id>/
```

Frozen Step8 large-block robustness çıktıları da orijinal Step8A-E namespace'inden tamamen ayrıdır:

```
outputs/robustness/step8_large_block/manavgat_2021__bejis_2022/
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
  -> [opsiyonel] frozen Step8 predefined large-block robustness
  -> [opsiyonel] Step9 cross-region transfer (Step9A-D)
  -> [opsiyonel] Step9E post-hoc dağılım-kayması denetimi
  -> [opsiyonel] Step9F kesifsel cross-region feature-representation denemesi
  -> [opsiyonel] Step10 preregistered unsupervised self-calibrated transfer
```

Bu iş akışının önemli özellikleri:

- **Hiçbir GEE batch Export task'i oluşturmaz.** `scripts/run_predictors_only.py --export`, current+baseline Landsat LST/NDVI'yı doğrudan yerel diske indirir (`getPixels`, gerekirse tiled fallback ile); Google Earth Engine "Tasks" panelinde görünen bir export işi YOKTUR.
- **Google Drive'ı ara adım olarak GEREKTİRMEZ.** Legacy Step4/Step4B'nin Drive export/download zincirini replike etmez.
- Manavgat ve Bejís için normal deney çıktıları `outputs/experiments/<experiment_id>/` altında namespaced'dır; legacy paylaşılan yollara (`outputs/step5/`, `outputs/validation/labels/`, vb.) asla yazmaz/okumaz.
- Step8 large-block robustness, orijinal Step8A-E dosyalarını salt okunur korur ve yalnızca `outputs/robustness/step8_large_block/...` altına yazar.

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

Son olarak, Step8C ve Step8D bu modelleme sonucunun ne kadar güvenilir olduğunu sorgular. Frozen Step8 large-block analizi aynı within-region soruyu önceden belirlenmiş daha büyük spatial grouping ölçeklerinde tekrar değerlendirir; scale seçimi sonuçlara göre yapılmaz. Step9C ve Step9E ise aynı disiplini cross-region seviyesine taşır: Step9C hedef-bölge spatial-block bootstrap ile bir güven aralığı üretirken, Step9E hangi feature'ların/ilişkilerin bölgeler arasında kaydığını post-hoc teşhis eder.

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

### Step8 predefined large-spatial-block robustness (Manavgat ve Bejís)

Bu frozen analiz, mevcut Step8B bilimsel yapılandırmasında yalnızca spatial grouping ölçeğini değiştirir. Primary population `burnable_tree_shrub_grass`'tır; block size listesi sonuç görülmeden `[10, 20]` olarak dondurulmuştur. Büyük blok kimlikleri canonical Step8A grid'inden, sabit origin ile `row_500m // block_size_cells` ve `col_500m // block_size_cells` kullanılarak population filtresinden önce oluşturulur.

Her iki ölçekte de aynı baseline/thermal feature setleri, preprocessing, RandomForest hiperparametreleri, seed ve strict 5-fold `StratifiedGroupKFold` kullanılır. Uncertainty, aynı large block'ları baseline ve thermal tahminler için birlikte örnekleyen 1000 replikalı paired spatial-block bootstrap ile ölçülür. Immutable preregistration, `analysis_id` ve orijinal Step8A/B/C/E girdilerinin koşu öncesi/sonrası SHA-256 kontrolü analizin koruma katmanıdır. Orijinal 2-hücre Step8 çıktıları yeniden çalıştırılmaz veya üzerine yazılmaz.

### Step9A-D: Cross-region transfer değerlendirmesi (Manavgat ↔ Bejís)

Step9, Step8'in burned-area association modelini (baseline + baseline+thermal) BİR bölgede eğitip TAMAMEN BAĞIMSIZ bir bölgede test eder — hem Manavgat→Bejís hem Bejís→Manavgat yönünde. Tüm ön-işleme (numeric median imputation, kategorik landcover encoder) YALNIZCA kaynak (source) bölgeden fit edilir; hedef (target) bölgenin etiketleri ön-işlemeyi, eşik seçimini veya fit'i hiçbir şekilde etkilemez (pooled source+target fit yoktur, target fine-tuning/calibration yoktur). Sınıflandırma eşiği yalnızca kaynak bölgenin kendi spatial-block CV out-of-fold tahminlerinden seçilir.

- **Step9A** — iki deneyin Step8A veri setlerinin gerçekten karşılaştırılabilir olup olmadığını (shared feature'lar, populasyon yeterliliği, label kaynağı, gate kararı) fail-fast olarak denetler.
- **Step9B** — iki yönlü transferi çalıştırır, hedef bölgede baseline ve thermal model tahminlerini üretir.
- **Step9C** — Step9B'nin tahminlerini yeniden eğitim yapmadan, hedef-bölgenin spatial-block'larını yeniden örnekleyerek (%95 percentile) bootstrap güven aralığı üretir.
- **Step9D** — Step9A-C'nin sonuçlarını birleştirip iki yönlü bir final rapor üretir.

Sonuçlar ve dikkatli yorumlama için bkz. Bölüm 9. Bu **30 m'lik bir yangın tahmin modeli DEĞİLDİR, operasyonel bir yangın tespit sistemi DEĞİLDİR**, ve Step7 downscaling modelinin kendisini transfer ETMEZ.

### Step9E: Post-hoc dağılım-kayması / ilişki-kayması denetimi

Step9E, Step9B/Step9C'nin neden discrimination'ı koruyamadığını POST-HOC olarak teşhis eder. Hiçbir modeli yeniden eğitmez, Step9B tahminlerini/Step9C bootstrap çıktılarını değiştirmez, raporlanan Step9 sonucunu değiştirmez. Detaylar için bkz. Bölüm 10.

## 8. Within-Region Bilimsel Sonuçlar

### Kozan 2023 çekirdek sonucu

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

### Manavgat/Bejís predefined large-block robustness sonucu

Bu bölüm iki AYRI robustness analizini net biçimde ayırır:

- **Formal Step8B primary population: `all_valid`** — supervisor'ın istediği asıl robustness sorusu.
- **Doğal-bitki-örtüsü duyarlılığı (natural-vegetation sensitivity): `burnable_tree_shrub_grass`** — dondurulmuş (frozen) ikincil duyarlılık analizi.

**Neden `STEP8B_SPATIAL_BLOCK_SIZE_CELLS = 2` sabit kalıyor?** Bu değer, dondurulmuş orijinal ~1 km Step8B referansını korur (frozen ~1 km reference). Robustness analizi büyük blok boyutlarını (10 ve 20 hücre) **runtime'da** geçirir; global config asla 10/20'ye çevrilmez. Böylece orijinal Step8 çıktıları değişmeden kalır. 10/20-hücre fit'inden önce **birebir (exact) 2-hücre equivalence** doğrulandı: aynı paylaşılan Step8B kod yolu 2 hücre ile çalıştırıldığında dondurulmuş orijinal çıktıyla hücre-hücre (cell_id hizalı) eşleşti (max olasılık farkı ≤ 1e-12).

- 10 hücre: ~5 km · 20 hücre: ~10 km

#### Formal `all_valid` sonuçları

| Experiment | Block cells | Nominal scale | Delta ROC-AUC | ROC %95 CI | Delta PR-AUC | PR %95 CI |
|---|---:|---|---:|---|---:|---|
| Manavgat 2021 | 10 | ~5 km | +0.053243 | [0.029428, 0.074860] | +0.048781 | [0.016436, 0.093869] |
| Manavgat 2021 | 20 | ~10 km | +0.044433 | [0.014663, 0.077631] | +0.026045 | [0.000626, 0.055532] |
| Bejís 2022 | 10 | ~5 km | +0.055725 | [0.030753, 0.078098] | +0.134358 | [0.059101, 0.226400] |
| Bejís 2022 | 20 | ~10 km | +0.061286 | [0.039680, 0.087105] | +0.076055 | [0.014015, 0.155429] |

Yorum (formal `all_valid`):

- thermal katkısı hem ~5 km hem ~10 km ölçeğinde, her iki bölgede de korundu
- tüm delta ROC-AUC ve delta PR-AUC aralıkları sıfırın üzerinde kaldı
- Manavgat 10 km delta PR-AUC desteği pozitif ama sıfıra çok yakın ([0.000626, 0.055532])
- daha büyük mekansal doğrulama ölçeklerinde mutlak AUC düştü
- spatial autocorrelation'ın ortadan kalktığı iddia edilmez

#### Doğal-bitki-örtüsü duyarlılığı (`burnable_tree_shrub_grass`) — dondurulmuş ikincil referans

- Frozen analysis ID: `1759eed5dd1027bdde69413d226502c1b8548e1ad187b44bb4a896d9fe1f8edd`
- Orijinal Step8 SHA-256 koruma kontrolü: **passed**
- Bu tablo yeniden koşulmadı; formal `all_valid` sonucundan ayrı, dış (external) dondurulmuş bir referans olarak okunmalıdır.

| Experiment | Block cells | Nominal scale | Delta ROC-AUC | ROC %95 CI | ROC support | Delta PR-AUC | PR %95 CI | PR support | Joint status |
|---|---:|---|---:|---|---|---:|---|---|---|
| Manavgat 2021 | 10 | ~5 km | +0.049880 | [0.023384, 0.077238] | bootstrap-supported positive | +0.051296 | [0.015606, 0.097750] | bootstrap-supported positive | supported on both metrics |
| Manavgat 2021 | 20 | ~10 km | +0.048363 | [0.014303, 0.085028] | bootstrap-supported positive | +0.024397 | [0.004694, 0.052712] | bootstrap-supported positive | supported on both metrics |
| Bejís 2022 | 10 | ~5 km | +0.045099 | [0.017780, 0.069003] | bootstrap-supported positive | +0.107823 | [0.040781, 0.199715] | bootstrap-supported positive | supported on both metrics |
| Bejís 2022 | 20 | ~10 km | +0.056610 | [0.031047, 0.089866] | bootstrap-supported positive | +0.083513 | [0.022369, 0.160857] | bootstrap-supported positive | supported on both metrics |

Dört predefined bölge-ölçek koşulunun tamamında hem delta ROC-AUC hem delta PR-AUC percentile aralığı sıfırın tamamen üzerindedir. Frozen raporun izin verdiği sonuç ifadesi: **“thermal contribution remained bootstrap-supported across both predefined large-block scales in both wildfire regions.”** Tüm koşullar stable bootstrap eşiğini geçti (geçerli replika: 1000, 1000, 1000 ve 995).

Bu sonuç bir “en iyi block size” seçimi değildir; 5 km ve 10 km sonuçları tek bir acceptance metriğinde ortalanmamıştır. Spatial autocorrelation'ın ortadan kalktığı, residual spatial dependence bulunmadığı, nedensel thermal etki veya operasyonel yangın tahmini iddia edilmez. Tam frozen rapor: `outputs/robustness/step8_large_block/manavgat_2021__bejis_2022/step8_large_block_final_report.{json,md}`.

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

### Step9F: Kesifsel (exploratory) cross-region feature-representation denemesi

Step9F, Step9E'nin teşhis ettiği dağılım/ilişki kaymalarının, ÖNCEDEN SABİTLENMİŞ feature altkümeleri (`original_baseline`, `original_thermal`, `thermal_without_elevation`, `thermal_without_absolute_lst`, `thermal_without_tvdi_difference`, `thermal_without_elevation_or_absolute_lst`, `stable_core`, `stable_core_without_landcover`) ve açıkça etiketlenmiş bir region-relative temsille azalıp azalmadığını araştıran **kesifsel, post-hoc** bir denemedir.

**Step9F tamamlandı** (post-hoc ve kesifsel). Step9B referans metrikleri 1e-6 tolerans içinde yeniden üretildi. Sekiz strict source-only varyant ve ayrıca iki region-relative adaptive varyant (yalnızca `original_thermal` ve `stable_core`) ayrı ayrı değerlendirildi. Önceden belirlenmiş `candidate_for_third_region_freeze` kriterlerinin **tamamını** geçen bir aday çıkmadı; kriterler sonuç görüldükten sonra gevşetilmedi. Nitel bulgular: mutlak-LST feature'larını çıkarmak iki yöndeki point estimate'ları iyileştirdi ve source performansını büyük ölçüde korudu, ancak ranking reversal'ı tamamen çözmedi; `stable_core` ranking reversal'ı iki yönde çözdü fakat source OOF performansında fazla kayıp oluşturdu; region-relative normalizasyon bazı transfer sorunlarını azalttı ama source performansı/Brier açısından yeni trade-off'lar getirdi. Bu README, gereksiz metrik kalabalığından kaçınmak için burada ham performans sayıları vermez — tam sayısal sonuçlar `outputs/cross_region/<source>__<target>/step9f/` altındadır. Henüz **doğrulanmış (validated) hiçbir transfer-safe temsil yoktur**; bir sonraki bilimsel karar, danışman değerlendirmesine bağlı olarak, dondurulmuş bir üçüncü-bölge dış değerlendirmesidir (bkz. Bölüm 18).

İki ayrı, AÇIKÇA etiketlenmiş rejim çalıştırılır (birbirine karıştırılmaz):

- **Regime A — `strict_source_only_inductive_transfer`**: tüm 8 sabit varyant; ön-işleme/eşik seçimi yalnızca kaynak (source) satırlarından.
- **Regime B — `unsupervised_target_covariate_adaptation` / `transductive_region_relative_representation`**: yalnızca `original_thermal` ve `stable_core` varyantları; her bölgenin kendi (etiketsiz) covariate medyan/IQR istatistikleriyle region-relative normalizasyon. Bu, saf source-only transfer DEĞİLDİR — asla öyle adlandırılmaz.

Step9F, Step9A-E dosyalarını **DEĞİŞTİRMEZ** (yalnızca salt-okunur girdi/provenance olarak okur), Step8A veri setlerini **DEĞİŞTİRMEZ**, hedef etiketleri normalizasyon/fit/eşik seçimi/kalibrasyon için **ASLA KULLANMAZ**, ve tahminleri **TERS ÇEVİRMEZ**. Regime A / `original_baseline` ve `original_thermal`'ın mevcut Step9B metriklerini sayısal tolerans dahilinde (1e-6) yeniden ürettiği doğrulanır; reprodüksiyon başarısız olursa Step9F fail-fast durur.

Herhangi bir aday burada seçilirse, **üçüncü bağımsız bir wildfire bölgesinde test edilmeden önce, Manavgat/Bejís üzerinde daha fazla ayar yapılmadan DONDURULMALIDIR (frozen)**. `candidate_for_third_region_freeze` bayrağı yalnızca kesifsel bir tarama (screening) kuralıdır — genelleme kanıtı DEĞİLDİR.

```bash
python scripts/run_exploratory_transfer_features.py \
  --source manavgat_2021 --target bejis_2022 --reverse --dry-run

python scripts/run_exploratory_transfer_features.py \
  --source manavgat_2021 --target bejis_2022 --reverse --force

python scripts/main.py transfer-explore \
  --source manavgat_2021 --target bejis_2022 --reverse --force
```

Kaynak kod: `src/step9f_exploratory_transfer_feature_experiment.py`, `core/cross_region_experiment.py` (paylaşılan yardımcılar), `scripts/run_exploratory_transfer_features.py`. Çıktılar (deney çalıştırıldığında): `outputs/cross_region/<source>__<target>/step9f/`.

### Step10: Preregistered unsupervised self-calibrated cross-region transfer

Step10 gerçek Manavgat/Bejís verisiyle tamamlandı. Primary transfer population: `burnable_tree_shrub_grass`. Hedef etiketleri adaptasyon, normalizasyon, CORAL fit, eşik seçimi veya kalibrasyon için **kullanılmadı**. Frozen `analysis_id`: `ea075fc3b67796a0e52fd383366d5f9ab45650efd7a4792d60749c08779a17c6`.

Thermal model, hedef (target) ROC-AUC ve PR-AUC ile hedef spatial-block bootstrap %95 percentile CI'ları:

**Manavgat → Bejís:**

| Yöntem | ROC-AUC | ROC %95 CI | PR-AUC | PR %95 CI |
|---|---:|---|---:|---|
| raw | 0.325834 | [0.304515, 0.348912] | 0.048758 | [0.043432, 0.054826] |
| region-wise z-score | 0.477100 | [0.450665, 0.501836] | 0.066667 | [0.058435, 0.076135] |
| CORAL (z-score sonrası) | 0.510540 | [0.483953, 0.534211] | 0.069581 | [0.061217, 0.079083] |

**Bejís → Manavgat:**

| Yöntem | ROC-AUC | ROC %95 CI | PR-AUC | PR %95 CI |
|---|---:|---|---:|---|
| raw | 0.443528 | [0.407765, 0.479941] | 0.034406 | [0.028893, 0.041660] |
| region-wise z-score | 0.457338 | [0.419580, 0.496820] | 0.038513 | [0.032079, 0.046813] |
| CORAL (z-score sonrası) | 0.555310 | [0.527802, 0.582844] | 0.042710 | [0.036622, 0.049805] |

Yorum:

- raw transfer her iki yönde de chance'in (0.5) altındaydı
- basit z-score, prototip beklentisi olan \"her iki yönde >0.5\" sonucunu yeniden üretmedi
- CORAL, yalnızca Bejís → Manavgat yönünde bootstrap-supported olarak chance'i aştı
- adaptasyon iyileşmesi kısmi ve yöne bağlıydı (asimetrik)
- adapted transfer, within-region performansın hâlâ belirgin biçimde altında kaldı

Başarılı genel cross-region transfer iddia edilmez; Step10 operasyonel yangın tahmini, universal CORAL üstünlüğü, probability calibration veya Step9'un \"düzeltilmesi\" olarak sunulmaz.

```bash
# user-facing step10 komutu (self-cal-transfer ile aynı analiz)
python scripts/main.py step10 \
  --source manavgat_2021 --target bejis_2022 --reverse --dry-run

python scripts/main.py step10 \
  --source manavgat_2021 --target bejis_2022 --reverse

# Frozen Step10A-C dosyalarını değiştirmeden yalnızca Step10D raporunu üret
python scripts/main.py step10 \
  --source manavgat_2021 --target bejis_2022 --reverse --report-only
```

Target-label firewall, raw/within reproduction, protected input hash ve analysis-ID consistency kontrolleri final report-only QA'da geçti. Brier değerleri frozen Step10 çıktılarında mevcut değildir ve report-only patch sırasında yeniden hesaplanmamıştır. Tam rapor: `outputs/cross_region/manavgat_2021__bejis_2022/step10/step10_final_report.{json,md}`.

### Step9G: Concept/relationship-shift teşhisi (univariate feature-AUC direction reversal)

Step9G, cross-region transfer başarısızlığı için mekanistik bir post-hoc teşhistir. Primary population: `burnable_tree_shrub_grass`.

Yöntem:

- ham feature değeriyle univariate burned ROC-AUC (raw feature-value; burned = pozitif sınıf)
- inversion yok, normalizasyon yok, imputation yok
- 10-hücre (~5 km) spatial-block bootstrap
- landcover kategorik olduğu için numeric AUC'den çıkarıldı

Bootstrap-supported direction reversal:

- `elevation_mean`
  - Manavgat: 0.374108 [0.289080, 0.471153]
  - Bejís: 0.643282 [0.558315, 0.729006]

CI'si chance'i (0.5) içeren nokta (point) reversal'ları — bunlar bootstrap-supported DEĞİLDİR:

- `current_lst_mean`
- `tvdi_difference_mean`
- `downscaled_lst_mean`
- `fused_lst_mean`

Aynı yön (same-direction) özellikleri:

- `ndvi_mean`
- `slope_mean`
- `lst_anomaly_mean`
- `current_tvdi_mean`

Step9E bağımsız olarak aynı beş ilişki-yönü (relationship-direction) instabilitesini işaretledi (`elevation_mean` + dört LST/TVDI özelliği). Step9F bulguları **model/temsil-seviyesi ranking-reversal** bulgularıdır; per-feature bulgular DEĞİLDİR ve öyle sunulmaz. Kanonik entegrasyon (integration-v2), frozen Step9G sayılarını birebir kullanır; hiçbir AUC/CI/reversal yeniden hesaplanmaz — yalnızca Step9E/9F/10 entegrasyonu düzeltilir ve `elevation_mean` doğru şekilde baseline (thermal değil) olarak sınıflandırılır.

```bash
# Step9G sayısal analizi (metrik HESAPLAR)
python scripts/main.py concept-shift --dry-run
python scripts/main.py concept-shift --force

# canonical integration-v2 raporu (REPORT-ONLY; hiçbir şey yeniden hesaplanmaz)
python scripts/main.py concept-shift --integration-only --force
```

Kanonik çıktı: `outputs/diagnostics/step9g_univariate_feature_auc_direction_reversal_integration_v2/manavgat_2021__bejis_2022/`. Sayısal Step9G analizi: `outputs/diagnostics/step9g_univariate_feature_auc_direction_reversal/manavgat_2021__bejis_2022/`.

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
python scripts/main.py step8-robustness --help
python scripts/main.py transfer --help
python scripts/main.py shift-audit --help
python scripts/main.py transfer-explore --help
python scripts/main.py self-cal-transfer --help
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

### `transfer-explore` — Step9F kesifsel (exploratory) feature-representation denemesi

```bash
python scripts/main.py transfer-explore \
  --source manavgat_2021 --target bejis_2022 --reverse --dry-run

python scripts/main.py transfer-explore \
  --source manavgat_2021 --target bejis_2022 --reverse --force
```

`scripts/run_exploratory_transfer_features.py`'yi reuse eder. Bu **kesifsel, post-hoc** bir denemedir — tarafsız dış validation DEĞİLDİR, Step9'un düzeltmesi DEĞİLDİR. Step9A-E dosyalarını DEĞİŞTİRMEZ. Detaylar için bkz. Bölüm 10.

### `self-cal-transfer` — Step10 preregistered unsupervised self-calibrated transfer

```bash
python scripts/main.py self-cal-transfer \
  --source manavgat_2021 --target bejis_2022 --reverse --dry-run

python scripts/main.py self-cal-transfer \
  --source manavgat_2021 --target bejis_2022 --reverse --force

python scripts/main.py self-cal-transfer \
  --source manavgat_2021 --target bejis_2022 --reverse --report-only
```

`scripts/run_step10_self_calibrated_transfer.py`'yi reuse eder. Hedef etiketleri adaptasyon/fit/eşik/kalibrasyon için ASLA kullanılmaz (target-label firewall). `--force`, on-kayıt (preregistration) dosyasını ASLA değiştirmez. Step9A-F dosyalarını DEĞİŞTİRMEZ. Detaylar için bkz. Bölüm 10.

### `step8-robustness` — frozen predefined large-spatial-block robustness

```bash
# Salt-okunur plan/provenance kontrolü; fit veya bootstrap çalıştırmaz
python scripts/main.py step8-robustness \
  --experiments manavgat_2021 bejis_2022 \
  --block-sizes-cells 10 20 \
  --dry-run

# Gerçek koşu
python scripts/main.py step8-robustness \
  --experiments manavgat_2021 bejis_2022 \
  --block-sizes-cells 10 20
```

Bu analysis version deneyleri tam olarak `manavgat_2021 bejis_2022`, block size listesini tam olarak `10 20` ve primary population'ı `burnable_tree_shrub_grass` olarak sabitler; alternatif veya yeniden sıralanmış değerler fail-fast reddedilir. Original small-block size'ın provenance'da 2 hücre olduğu doğrulanır. Large block'lar canonical `row_500m`/`col_500m` indekslerinden sabit origin ile population filtresinden önce oluşturulur.

`--dry-run` resolved/protected input yollarını, output namespace'ini, feature/model/CV ayarlarını, block reconstruction kaynağını, bootstrap ayarlarını ve preregistration durumunu basar; hiçbir model fit/bootstrap veya scientific output yazımı yapmaz. Gerçek çıktıların tamamı `outputs/robustness/step8_large_block/manavgat_2021__bejis_2022/` altındadır. `--force` yalnızca downstream robustness çıktılarını yenileyebilir; immutable preregistration veya orijinal Step8A-E dosyalarını değiştiremez.

### `large-block-robustness` — formal Step8B primary-population (all_valid) robustness

```bash
# 2-hücre equivalence gate + plan (fit YAPMAZ)
python scripts/main.py large-block-robustness --dry-run

# gate incelendikten SONRA 10/20-hücre bilimsel koşumu fit et
python scripts/main.py large-block-robustness --run-large-block-fit --force
```

`scripts/run_step8_large_block_robustness_primary_all_valid.py`'yi reuse eder. Formal Step8B primary population `all_valid` için 10 (~5 km) ve 20 (~10 km) hücre robustness'ını test eder. `STEP8B_SPATIAL_BLOCK_SIZE_CELLS` global config **2 olarak kalır** (frozen ~1 km referans); büyük blok boyutları runtime'da geçilir. 10/20-hücre fit'i **yalnızca** 2-hücre equivalence gate geçtikten sonra ve `--run-large-block-fit` verildiğinde başlar. Orijinal Step8 çıktıları asla üzerine yazılmaz. Bu, doğal-bitki-örtüsü (`burnable_tree_shrub_grass`) `step8-robustness` komutundan ayrı, formal `all_valid` analizidir.

### `concept-shift` — Step9G concept/relationship-shift teşhisi

```bash
# Step9G sayısal analizi (metrik HESAPLAR)
python scripts/main.py concept-shift --dry-run
python scripts/main.py concept-shift --force

# canonical integration-v2 raporu (REPORT-ONLY; hiçbir şey yeniden hesaplanmaz)
python scripts/main.py concept-shift --integration-only --force
```

Varsayılan (bayraksız) mod Step9G univariate feature-AUC direction-reversal analizini çalıştırır (population `burnable_tree_shrub_grass`; ham feature değeriyle burned ROC-AUC; inversion/normalizasyon/imputation yok; 10-hücre spatial-block bootstrap). `--integration-only` yalnızca kanonik integration-v2 raporunu üretir: frozen Step9G sayıları birebir kullanılır, hiçbir AUC/CI/reversal yeniden hesaplanmaz. Kanonik çıktı `..._integration_v2/` namespace'indedir. Orijinal Step9G sayısal çıktıları değişmez.

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

# Step8 large-block robustness (gerçek koşu; önce canonical --dry-run incelenmelidir)
python scripts/run_step8_large_block_robustness.py \
  --experiments manavgat_2021 bejis_2022 \
  --block-sizes-cells 10 20

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

`--force` bayrağı, ilgili runner'ın izin verdiği downstream çıktıları yenilemek için kullanılır. Step8 robustness ve Step10 gibi preregistered analizlerde immutable manifest/preregistration bu bayraktan özellikle muaftır; runtime scientific configuration mevcut manifestle uyuşmuyorsa koşu fail-fast durur. Orijinal Step8A-E dosyaları `--force` ile dahi robustness runner tarafından yazılamaz.

## 15. Proje Yapısı

```
core/
  config.py                    # Merkezi konfigürasyon sabitleri (legacy Kozan)
  paths.py                     # Girdi/çıktı yol tanımları
  io_utils.py                  # Ortak okuma/yazma yardımcıları
  regions.py                   # Step0 deney/bölge (experiment/region) kayıt defteri
  experiment_context.py        # Step3-Step5/5C/7/8 için deney-farkında path/tarih context'i
  pipeline_orchestrator.py     # scripts/main.py'nin kullandığı asama sırası/dispatch katmanı
  cross_region_experiment.py   # Step9F'in paylaşılan yardımcıları (sabit varyantlar, region-relative transform, esli bootstrap)
  step10_shared.py             # Step10'un paylaşılan yardımcıları (region-wise z-score, CORAL, N-yollu eşli bootstrap, hashing/analysis_id)
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
  step8_large_block_robustness.py        # Frozen 10/20-hücre robustness + paired block bootstrap
  step9a_audit_cross_region_inputs.py       # Cross-region girdi uygunluk denetimi
  step9b_run_cross_region_transfer.py       # İki yönlü cross-region transfer
  step9c_cross_region_block_bootstrap.py    # Hedef-bölge spatial-block bootstrap
  step9d_build_cross_region_report.py       # Birleşik final cross-region raporu
  step9e_distribution_shift_audit.py        # Post-hoc dağılım-kayması/ilişki-kayması denetimi
  step9f_exploratory_transfer_feature_experiment.py  # Kesifsel cross-region feature-representation denemesi
  step10a_preregistration_and_audit.py      # Step10: immutable preregistration + girdi denetimi
  step10b_label_blind_adaptation.py         # Step10: hedef-etiket-körü (label-blind) fit/adapt/predict
  step10c_paired_evaluation_bootstrap.py    # Step10: etiket SIMDI yüklenir; eşli değerlendirme + N-yollu bootstrap
  step10d_final_report.py                   # Step10: yalnızca yorumlama (hesaplama YAPMAZ)

scripts/
  main.py                              # Canonical CLI (experiment/step8-robustness/transfer/shift-audit/transfer-explore/self-cal-transfer/legacy)
  export_mcd64a1_raw_burndate.py       # Raw BurnDate DOY export'u için ince CLI sarmalayıcı
  preview_experiment_aoi.py            # Deney metadata + AOI önizleme (export/pipeline çalıştırmaz)
  run_label_gate_only.py               # Gate-only çalıştırıcı (Kozan: legacy; diğerleri: namespaced)
  run_predictors_only.py               # Step3-Step5/5C predictor çalıştırıcı (Kozan: legacy; diğerleri: namespaced)
  run_step7_downscaling_only.py        # Deney-farkında Step7A-E çalıştırıcı
  run_step8_modeling.py                # Deney-farkında Step8A-E çalıştırıcı
  run_step8_large_block_robustness.py  # Frozen predefined large-block robustness runner
  run_cross_region_transfer.py         # Step9A-D orkestratörü
  run_cross_region_shift_audit.py      # Step9E orkestratörü
  run_exploratory_transfer_features.py # Step9F orkestratörü (kesifsel, post-hoc)
  run_step10_self_calibrated_transfer.py  # Step10 orkestratörü (preregistered, label-blind)
  prepare_dem_for_experiment.py        # Kozan-dışı deneyler için namespaced DEM hazırlığı
  prepare_modis_for_step7.py           # Kozan-dışı deneyler için namespaced MODIS mean/std hazırlığı
  check_experiment_registry.py         # Step0 kayıt defteri doğrulama script'i
  run_prefire_experiment.py            # Pre-fire config doğrulama yardımcı script'i (legacy Kozan)
  standalone_step5-6.py                # Step5'i tek başına çalıştıran yardımcı script (legacy Kozan)

tests/
  test_pipeline_orchestrator.py        # Asama sırası, namespace güvenliği, dry-run no-execution testleri
  test_main_cli.py                     # scripts/main.py argparse/dispatch testleri
  test_step8_large_block_robustness.py # Block/fold/bootstrap/preregistration/protection testleri
  test_step9f.py                       # Step9F: sabit varyantlar, yasak kolon, region-relative etiket-kullanmama, esli bootstrap, reprodüksiyon kontrolü, aday tarama kuralı testleri
  test_step10.py                       # Step10: region-wise z-score/CORAL, target-label firewall, N-yollu eşli bootstrap, preregistration immutability, within-region hizalama, raw reprodüksiyon testleri

data/       # Yerel ham/indirilmiş veri klasörleri (.gitignore'da)
outputs/    # Her step'in ürettiği raster, tablo ve rapor çıktıları (.gitignore'da)
  experiments/<experiment_id>/         # Deney-farkında (namespaced) çıktılar (Manavgat, Bejís)
  cross_region/<source>__<target>/     # Step9/Step10 çıktıları
  robustness/step8_large_block/        # Orijinal Step8'den ayrı frozen robustness çıktıları
logs/       # Runtime log dosyaları; .gitignore'da
```

`core/`, tüm step dosyalarının paylaştığı sabitleri, yardımcı fonksiyonları ve `scripts/main.py`'nin kullandığı orkestrasyon katmanını barındırır; `src/`, Step1-Step10 ve Step8 robustness analizlerinin çalıştırılabilir mantığını içerir; `scripts/`, uçtan uca çalıştırma, deney-farkında asama çalıştırıcıları ve tek seferlik yardımcı işlemleri içerir; `tests/`, orkestrasyon/CLI mantığı için odaklı unittest testlerini içerir.

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

**Step8 formal large-block robustness — primary population `all_valid` (canonical; tamamlandı):**
- `outputs/robustness/step8_large_block_primary_all_valid/manavgat_2021__bejis_2022/step8_large_block_primary_all_valid_preregistration.{json,md}` (immutable)
- `step8b_two_cell_equivalence_audit.{json,md}` (10/20-hücre fit'inden önce 2-hücre birebir eşdeğerlik)
- `manavgat_2021/` ve `bejis_2022/` altında `block_10_cells/` / `block_20_cells/`: block/fold QA, OOF predictions, metrics, paired-block bootstrap
- `step8_large_block_primary_all_valid_comparison.csv`, `step8_large_block_primary_all_valid_final_report.{json,md}`

**Step8 large-block robustness — doğal-bitki-örtüsü duyarlılığı `burnable_tree_shrub_grass` (dondurulmuş ikincil referans):**
- `outputs/robustness/step8_large_block/manavgat_2021__bejis_2022/step8_large_block_preregistration.{json,md}` (immutable)
- `step8_large_block_input_audit.json`
- `manavgat_2021/` ve `bejis_2022/` altında `block_10_cells/` / `block_20_cells/`: block/fold QA, OOF predictions, metrics ve paired-block bootstrap summary/replicates
- `step8_large_block_comparison.csv`
- `step8_large_block_final_report.{json,md}`

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

**Step10 (preregistered unsupervised self-calibrated transfer; tamamlandı):**
- `outputs/cross_region/manavgat_2021__bejis_2022/step10/step10_preregistration.{json,md}` (immutable)
- `outputs/cross_region/manavgat_2021__bejis_2022/step10/step10_input_audit.json`
- `outputs/cross_region/manavgat_2021__bejis_2022/step10/step10_adaptation_statistics.json`, `step10_predictions.parquet` (hedef etiketi İÇERMEZ)
- `outputs/cross_region/manavgat_2021__bejis_2022/step10/step10_metrics.{json,csv}`, `step10_decomposition.csv`
- `outputs/cross_region/manavgat_2021__bejis_2022/step10/step10_bootstrap_replicates.parquet`, `step10_bootstrap_summary.{json,csv}`
- `outputs/cross_region/manavgat_2021__bejis_2022/step10/step10_final_report.{json,md}`

**Step9G concept/relationship-shift — sayısal analiz (canonical numeric; tamamlandı):**
- `outputs/diagnostics/step9g_univariate_feature_auc_direction_reversal/manavgat_2021__bejis_2022/step9g_preregistration.{json,md}` (immutable)
- `step9g_univariate_auc_by_region.csv`, `step9g_direction_reversal_table.csv`, `step9g_bootstrap_replicates.parquet`
- `step9g_landcover_descriptive.csv`, `step9g_step9e_feature_integration.csv`, `step9g_step9f_model_level_integration.json`
- `step9g_final_report.{json,md}`, `step9g_auc_direction_plot.png`

**Step9G integration-v2 (CANONICAL entegrasyon raporu; report-only; tamamlandı):**
- `outputs/diagnostics/step9g_univariate_feature_auc_direction_reversal_integration_v2/manavgat_2021__bejis_2022/step9g_integration_correction_manifest.json`
- `step9g_integration_correction_final_report.{json,md}`, `step9g_integration_correction_per_feature.csv`, `step9g_integration_availability_before_after.csv`
- Not: frozen Step9G sayısal çıktılarını birebir kullanır; hiçbir AUC/CI/reversal yeniden hesaplanmaz.

## 17. Sınırlamalar

**Bilimsel sonuçların özeti (restrained wording):**

- within-region thermal katkısı bootstrap-supported'dır ve predefined daha büyük mekansal bloklarda (10/20 hücre) korunur
- ham (raw) cross-region discrimination transferi başarısız oldu
- unsupervised adaptation kısmi ve asimetrik (yöne bağlı) bir kısmi iyileşme sağladı
- feature-seviyesi ilişki-yönü (relationship-direction) instabilitesi, residual concept/relationship shift ile tutarlıdır
- bu nedensel kanıt değildir
- concept shift'in transfer başarısızlığının TEK kaynağı olduğu kanıtlanmamıştır
- bu bir operasyonel yangın erken-uyarı sistemi değildir

Kullanılmayan ifadeler: \"statistically significant\", \"causal thermal effect\", \"spatial autocorrelation eliminated\", \"successful generalization\", \"operational prediction system\".

- **Step8 large-block sonucu yalnızca iki bölge ve iki predefined ölçek içindir.** Dört koşulda bootstrap support bulunması spatial autocorrelation'ın elimine edildiğini, residual spatial dependence olmadığını veya başka block size'ların eşdeğer davranacağını kanıtlamaz. Elverişli bir ölçek post hoc seçilmemiştir. Formal `all_valid` analizi ile doğal-bitki-örtüsü `burnable_tree_shrub_grass` duyarlılığı ayrı raporlanır; ikincisi dondurulmuş dış referanstır.
- **Step9G univariate direction-reversal teşhisi marjinaldir ve nedensel değildir.** Tek bootstrap-supported reversal `elevation_mean`'dir; dört LST/TVDI point reversal'ının CI'ları chance'i içerir ve bootstrap-supported DEĞİLDİR. Bu, bir Random Forest'ın neden zayıf transfer ettiğini KANITLAMAZ; yalnızca frozen Step9 diagnostikleriyle tutarlı mekanistik bir teşhistir. Step9F bulguları model/temsil-seviyesidir, per-feature değildir.
- **Kozan, primary wildfire AOI'si değildir.** Kozan 2023 burned label'larının büyük çoğunluğu (533/542) cropland/anız-yakma kaynaklıdır; Kozan cropland/anız-dominant bir negative/control AOI'dir.
- **Cross-region discrimination generalization desteklenmemiştir.** Step9A-D, Manavgat↔Bejís arasında ROC-AUC/PR-AUC delta'larının belirsiz (bir yönde negatif) olduğunu göstermiştir; yalnızca Brier (probability-error) iyileşmesi tutarlıdır (bkz. Bölüm 9). Bu sonuç yalnızca İKİ bölge (Manavgat, Bejís) arasında elde edilmiştir.
- **Step9E teşhisi, iki bölgeye özgüdür ve nedensel değildir.** Step9E'nin bulguları (feature dağılım kayması, ilişki-yönü kayması, olasılık ölçek kayması) tanımlayıcı/betimsel diagnostiklerdir; hangi mekanizmanın ASIL nedensel sürücü olduğu kanıtlanmamıştır ve üçüncü bir bölge olmadan doğrulanamaz.
- **Step9E'nin önerdiği herhangi bir düzeltme, henüz test edilmemiş bir hipotezdir.** Region-relative/robust normalizasyon veya feature subset seçimi gibi fikirler, AYNI iki bölgede (etiketleri zaten incelenmiş) test edilip "unbiased" ilan edilemez; bunlar yeni, bağımsız bir deney olarak ele alınmalıdır (bkz. Bölüm 18).
- **Step10 tamamlanmış olsa da iki-bölge diagnostic adaptation deneyidir.** Region-wise z-score/CORAL sonuçları universal yöntem üstünlüğü, probability calibration veya operasyonel transfer kanıtı değildir; bootstrap-supported residual gap yalnızca remaining concept shift veya diğer non-covariate farklarla tutarlıdır, nedensel kanıt değildir.
- **Tek sezon/yıl (her üç AOI için):** Her deney tek bir yangın sezonunu kapsar; year-to-year robustness henüz test edilmemiştir.
- **Cropland-excluded burnable mask:** Bu maske içinde pozitif örnek sayısı azdır, bu da bu strata için istatistiksel gücü sınırlar (özellikle Kozan).
- **Günlük MODIS gap-fill yok:** Mevcut fusion mantığı (Step7E) bir defalık, statik bir birleştirmedir; günlük operasyonel bir veri akışı değildir.
- **Üçüncü bağımsız bölge (external validation) henüz yoktur.** `zamora_2022` şu an disabled placeholder olarak kayıtlıdır.
- **3B / operasyonel dijital ikiz katmanı henüz yoktur.**

## 18. Sonraki Adımlar

1. ~~**Canonical experiment/transfer/robustness orkestrasyonunu ve dry-run/test korumalarını tamamlamak.**~~ **Tamamlandı.**
2. ~~**Manavgat ve Bejís için frozen 10/20-hücre Step8 robustness analizini çalıştırmak ve orijinal Step8 hash korumasını doğrulamak.**~~ **Tamamlandı.**
3. **Herhangi bir yeni model/representation kararını üçüncü bağımsız wildfire bölgesinde test edilmeden önce dondurmak.**
4. **Zamora veya başka bir bağımsız bölgeyi external evaluation için etkinleştirmek; Manavgat/Bejís üzerinde ek post-hoc seçim yapmamak.**
5. **Year-to-year robustness için aynı AOI'lerde ek sezonları önceden belirlenmiş bir tasarımla değerlendirmek.**
6. **Residual spatial dependence'i ayrıca ölçen diagnostics eklemek; mevcut large-block desteğini “spatial autocorrelation eliminated” olarak yeniden adlandırmamak.**
7. **3B/dijital-ikiz sunum çalışmasına, mevcut modelleri operasyonel erken-uyarı sistemi olarak sunmadan devam etmek.**

## 19. Terminoloji / Claim Policy

Bu README ve proje çıktıları, aşağıdaki ifade politikasına uyar:

**İzin verilen ifadeler:**
- "Within-region thermal contribution was observed" (bkz. Bölüm 8).
- "Thermal contribution remained bootstrap-supported across both predefined large-block scales in both wildfire regions" (yalnızca frozen Step8 robustness koşulları için).
- "Direct cross-region discrimination was not supported" (bkz. Bölüm 9).
- "Brier error improved" / "probability-error improved" (bkz. Bölüm 9).
- "Step9E found diagnostic evidence consistent with domain shift and region-dependent feature-label relationships" (bkz. Bölüm 10).
- "A third independent region or nested design is needed for a stronger generalization claim" (bkz. Bölüm 17, 18).
- "Step10 tests unsupervised target-covariate adaptation (region-wise z-score / CORAL) without using target labels for fitting" (bkz. Bölüm 10).
- "A residual performance gap after covariate adaptation is consistent with remaining concept shift" (yalnızca bootstrap-supported ise, bkz. Bölüm 10).

**Yasak ifadeler:**
- "Operational wildfire prediction" — proje operasyonel bir sistem değildir (bkz. Bölüm 2).
- "Successful cross-region model" / "successful transfer" — Step9A-D sonucu bu değildir (bkz. Bölüm 9).
- "Statistically significant transfer/robustness" — tüm güven aralıkları bootstrap percentile interval'dır, klasik p-value değildir.
- "The best block size was selected" / "spatial autocorrelation was eliminated" — iki ölçek de önceden belirlenmiştir; residual dependence'in yokluğu kanıtlanmamıştır.
- "Causal fire prediction" — hiçbir aşamada nedensellik iddia edilmez.
- "Validated correction" — Step9E bir düzeltme değil, post-hoc bir teşhistir (bkz. Bölüm 10).
- "Transfer-safe feature set proven on Manavgat and Bejís after inspecting both labels" — bu, aynı iki bölgede etiketleri zaten görülmüş bir stratejiyi "kanıtlanmış" ilan etmek anlamına gelir ve yasaktır (bkz. Bölüm 17, 18).
- "Step10 is probability calibration" / "Step10 proves operational transfer" / "Step10 corrected Step9" — Step10, hedef etiketlerini adaptasyon için kullanmayan bir unsupervised covariate-adaptation denemesidir; bir kalibrasyon veya düzeltme değildir (bkz. Bölüm 10).
- "CORAL definitively outperforms region-wise standardization" — yalnızca bootstrap CI'si bunu tam olarak desteklerse ("CORAL - z-score" CI'si 0'ın tamamen üzerindeyse) böyle bir iddia yapılabilir; aksi halde "did not show supported improvement" denir.

**Ayrıca (önceki politikadan korunan maddeler):**
- "Fire-risk prediction model validated" denmez.
- "TVDI alone predicts fire risk" denmez.
- FIRMS hiçbir bağlamda bir target olarak sunulmaz.
- MCD64A1'in native ~500 m label çözünürlüğü korunur; hiçbir yerde 30 m piksel bazlı bir label hassasiyeti iddia edilmez.
- **Kozan'ın doğal bitki örtüsü (orman/makilik) yangın davranışını doğruladığı iddia edilmez.** Doğru ifade: *"Kozan serves as a cropland/stubble-dominated negative control; Manavgat and Bejís passed the burned-landcover gate as natural-vegetation wildfire AOIs."*

## Sonuç

Kozan üzerindeki çekirdek Step8 deneyi ile başlayan label-honest 500 m modeling metodolojisi, Manavgat 2021 ve Bejís 2022 üzerinde bağımsız within-region sonuçlar üretti. Frozen large-spatial-block robustness analizi, primary `burnable_tree_shrub_grass` population'ında, önceden belirlenmiş 10-hücre (~5 km) ve 20-hücre (~10 km) koşullarının dördünde de hem delta ROC-AUC hem delta PR-AUC için bootstrap-supported positive aralıklar buldu. Orijinal Step8 dosyalarının SHA-256 koruma kontrolü geçti; hiçbir elverişli ölçek post hoc seçilmedi.

Bu within-region robustness sonucu cross-region transfer sonucu değildir. Step9A-D doğrudan discrimination generalization'ını desteklemedi; Step9E bunu distribution/relationship shift açısından post-hoc teşhis etti. Step10 target-label-blind covariate adaptation ile bazı yön/yöntem kombinasyonlarında recovery gösterdi, ancak residual within-region gap kaldı ve sonuçlar universal CORAL üstünlüğü veya başarılı operasyonel transfer olarak sunulmadı.

Proje artık `scripts/main.py` üzerinden `experiment`, `step8-robustness`, `transfer`, `shift-audit`, `transfer-explore`, `self-cal-transfer` ve `legacy` alt-komutlarını sağlayan canonical bir CLI'a sahiptir. Bilimsel sonraki öncelik, mevcut iki bölge üzerinde daha fazla post-hoc seçim yapmak değil; seçilecek herhangi bir yaklaşımı önce dondurup üçüncü bağımsız bir wildfire bölgesinde ve daha sonra çok-yıllı bir tasarımda değerlendirmektir.