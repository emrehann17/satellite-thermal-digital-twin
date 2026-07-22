# 1. Projenin tek paragrafta özeti

Bu proje, bir yangının başlamasını ya da yayılmasını operasyonel olarak tahmin etmekten ziyade, **etiket döneminden önceki uydu-tabanlı yüzey koşullarının yaklaşık 500 m hücre düzeyinde sonraki yanmış alanla ilişkili bir sıralama sinyali taşıyıp taşımadığını** inceler. Landsat/MODIS termal göstergeleri, NDVI, topoğrafya ve arazi örtüsünden üretilen hücre özellikleri; MCD64A1 `BurnDate` ile tanımlanan ikili yanma etiketiyle karşılaştırılır. Güncel bilimsel katkı şudur: thermal feature grubu Manavgat 2021 ve Bejís 2022 içinde baseline'a ek ayrım gücü verir ve bu kontrast önceden tanımlanmış daha büyük mekânsal bloklarda da sürer; buna karşılık ham cross-region discrimination başarısızdır, unsupervised z-score/CORAL uyarlaması yalnızca kısmi ve yön-asimetrik iyileşme sağlar, univariate analiz de en az bir bootstrap-destekli feature–label yön terslenmesi gösterir. Bu nedenle proje bir nedensellik çalışması, genellenmiş yangın tehlike modeli, gerçek-zamanlı early-warning sistemi veya concept shift'in tek başına kanıtı değildir. Temel tasarım `core/regions.py`, feature/model sözleşmesi `src/step8b_train_baseline_vs_thermal_model.py`, güncel sonuç zinciri ise `outputs/experiments/`, `outputs/cross_region/`, `outputs/robustness/` ve `outputs/diagnostics/` altındadır.

**30 saniyelik anlatım.** Yangından önceki uydu termal koşullarının, yaklaşık 500 m hücrelerde daha sonra yanıp yanmama ile ilişkisini test ettim. Thermal features her iki yangın bölgesinde yerel modele katkı sağladı ve sonuç daha büyük spatial blocks ile korundu. Fakat bir bölgede öğrenilen sıralama diğerine doğrudan taşınmadı; unsupervised adaptation yalnızca kısmi ve asimetrik bir iyileşme verdi. Sonuç, yararlı yerel sinyal ile güçlü bölgesel ilişki değişkenliğinin birlikte bulunduğunu gösteriyor.

**2 dakikalık danışman anlatımı.** Predictor window ile label window'u ayırdım; MCD64A1 `BurnDate` yalnızca sonraki dönemde yanma etiketi üretir. Landsat/MODIS, NDVI, DEM ve land-cover girdileri yaklaşık 500 m modelleme hücresine toplanır. Baseline model `ndvi_mean`, `elevation_mean`, `slope_mean`, `landcover_dominant`; thermal model bunlara altı thermal değişken ekler. Random Forest değerlendirmesi random-row split değil, spatial-block OOF CV ile yapılır; belirsizlik hücre değil block bootstrap ile hesaplanır. Manavgat ve Bejís içinde thermal-minus-baseline ROC/PR artışı güçlüdür ve 5–10 km blok hassasiyetinde de pozitif kalır. Ancak source-only transferde thermal ROC-AUC Manavgat→Bejís için 0.326, ters yönde 0.444'tür. Region-wise z-score ve CORAL bunu kısmen düzeltir; yalnız Bejís→Manavgat CORAL ROC-AUC'si chance'ın üzerinde bootstrap-desteklidir. Step9G, `elevation_mean` ilişkisinin bölgeler arasında bootstrap-destekli terslendiğini, dört thermal feature'da ise belirsiz point reversal bulunduğunu gösterir. Güvenli sonuç: covariate shift tek başına sorunu açıklamıyor; residual relationship/concept shift ile uyumlu, fakat onu tek neden olarak kanıtlamayan bir örüntü var.

**5 dakikalık teknik anlatım.** Deney sözleşmeleri `EXPERIMENTS` registry'sinde tanımlanır (`core/regions.py`). Step7, burned label kullanmadan Landsat LST'yi MODIS/NDVI/topoğrafya/land-cover ile downscale eder; Step7E gözlenen Landsat'a öncelik verip yalnız eksik yerde downscaled değeri kullanarak fused LST üretir. Step8A, 30 m predictors'ı yaklaşık 500 m hücrelere aggregate eder, `BurnDate` label-window DOY aralığını positive sınıf yapar ve label'ı predictor validity'den ayırır. Step8B aynı spatial folds üzerinde baseline ve thermal Random Forest OOF predictions üretir; fold içindeki median/mode imputation ve one-hot encoding yalnız train fold'a fit edilir. Step8C paired block bootstrap ile metric delta CI'ları üretir; Step8D/E ablation ve raporlamayı tamamlar. Confirmatory all-valid large-block analizi config'deki 2-cell referansı değiştirmeden runtime'da 10/20 cells kullanır; exact 2-cell equivalence gate ve protected hashes ile ayrı namespace'e yazar. Step9 source preprocessing/model fitting'i target labels'tan ayırır ve iki yönlü target evaluation yapar. Step9E shift'i post-hoc teşhis eder; Step9F sekiz source-only representation ve iki unlabeled-target adaptive varyantı keşifsel olarak sınar fakat freeze kriterini geçen üçüncü-bölge adayı bulmaz. Step10 region-wise mean/std ve numeric-only CORAL uygular; target covariates kullanılabilir, target labels adaptation sırasında kullanılamaz. Step9G ise natural-vegetation popülasyonunda, 10-cell blocks ve raw univariate AUC ile yön terslenmesini ölçer. Bütün zincir yerel katkının transfer edilebilirlikle aynı şey olmadığını gösterir.

# 2. Projenin tarihsel gelişimi

Projenin geçmişi “tek bir nihai pipeline” olarak değil, her biri önceki sonucun açtığı soruya yanıt veren aşamalar olarak okunmalıdır. Git geçmişi ve dosya evrimi, Mayıs 2026'daki Kozan thermal digital-twin prototipinden Temmuz ortasındaki formal robustness ve concept-shift teşhisine uzanır.

| Dönem / Step | Bilimsel soru | Ana giriş | Ana çıktı | Statü | Bugünkü önemi |
|---|---|---|---|---|---|
| Kozan / legacy Steps 1–6 | Thermal anomaly/TVDI üretilebilir ve burned area ile doğrulanabilir mi? | `scripts/main.py legacy`; `core/config.py` | `outputs/step*/` | Legacy/control | Veri ürünlerinin ve label-honesty kurallarının kökü; güncel wildfire iddiasının ana kanıtı değil |
| Landsat/MODIS anomaly ve TVDI | Current LST, climatology anomaly ve moisture-stress proxy nasıl üretilir? | `src/step4_*`, `src/step5*` | Kozan step4–5 ağaçları | Legacy ama yöntemsel temel | Thermal feature sözlüğünün kaynağı |
| Burned-area validation / Step6 | Thermal göstergeler gerçekten doğal vejetasyon yangınını mı temsil ediyor? | `src/step6_*` | `outputs/step6/labels/` | Kozan için negatif/control bulgu | Kozan'ın cropland-dominated olduğunu ortaya çıkardı |
| Step7 downscaling/fusion | Coarse MODIS bağlamından label-free fine-resolution LST kestirilebilir mi? | `src/step7*` | experiment `step7*` | Current upstream | `downscaled_lst_mean`, `fused_lst_mean` üretir |
| Step8 within-region | Thermal grup baseline'a aynı bölgede ek predictive value sağlıyor mu? | `scripts/main.py experiment`; `src/step8*` | `outputs/experiments/<id>/step8*` | Canonical | Yerel thermal contribution'ın temel kanıtı |
| Step9 raw transfer | Bir bölgede öğrenilen model diğer yangın bölgesinde doğrudan çalışıyor mu? | `scripts/main.py transfer` | `outputs/cross_region/.../step9*` | Canonical, negatif | Raw discrimination transfer'ın başarısızlığını gösterir |
| Step9E shift audit | Başarısızlık distribution/probability/ranking/relationship değişimiyle ilişkili mi? | `scripts/main.py shift-audit` | aynı pair altında `step9e*` | Canonical post-hoc diagnostic | Feature shift ve reversal şüphesini yerelleştirir |
| Step9F representations | Daha stable/source-only ya da unlabeled-target normalized representation transferı kurtarır mı? | `scripts/main.py transfer-explore` | `step9f*` | Exploratory | Hiçbir varyant freeze kriterini geçmedi |
| Step10 self-calibration | Label kullanmadan region-wise normalization ve CORAL ne kadar recovery sağlar? | `scripts/main.py step10` | pair altında `step10*` | Canonical confirmatory pipeline | Kısmi, asimetrik recovery; büyük remaining gap |
| Step8 large-block robustness | Yerel thermal delta daha büyük spatial blocks altında sürüyor mu? | `scripts/main.py large-block-robustness` | `outputs/robustness/...` | Canonical formal all-valid; ayrı sensitivity | Mekânsal bağımlılık hassasiyetini güçlendirir, ortadan kaldırmaz |
| Step9G direction reversal | Hangi tekil feature–label ilişkileri bölgeler arasında yön değiştiriyor? | `scripts/main.py concept-shift` | `outputs/diagnostics/...` | Canonical numeric + integration-v2 | `elevation_mean` için destekli, dört thermal feature için belirsiz point reversal |

İlk Kozan çalışması bir “başarı hikâyesi” olarak taşınmamalıdır: `outputs/step6/labels/burned_landcover_gate.md`, burned pixels'ın %98.34'ünün cropland olduğunu ve gate kararının `cropland_dominated_control` olduğunu bildirir. Bu bulgu Manavgat anchor wildfire ve Bejís transfer wildfire tasarımına geçişi motive eder. Step8'in olumlu within-region sonucu Step9'u; Step9'un anti-predictive transferı Step9E/F ve Step10'u; adaptation sonrasında kalan büyük gap ise Step9G'yi motive etmiştir. Bu sıra post-hoc çalışmaların confirmatory gibi sunulmasını engellemek için önemlidir.

# 3. Bilimsel problem ve deney tasarımı

## Düz dilde varsayımlar

- **Zaman ayrımı:** Model geçmiş/öncü dönemi görür; “yanacak mı?” etiketi daha sonraki label window'dan gelir. Aynı olayın termal etkisini predictor içine sokmak yasaktır.
- **Hedef:** FIRMS sıcak nokta sayısı değil, MCD64A1'de hücrenin label döneminde yanmış olmasıdır. FIRMS bağımsız bağlam/validation olabilir; primary target değildir.
- **Mekân:** Bir piksel satırını bağımsız örnek saymak gerçekçi değildir. Yakın hücreler benzer arazi ve hava koşullarını paylaşır; train/test ayırımı mekânsal gruplarla yapılır.
- **İki ayrı soru:** “Aynı bölgede thermal bilgi işe yarıyor mu?” ile “başka bölgeye taşınıyor mu?” farklı sorulardır.
- **AUC < 0.5:** Modelin tesadüften biraz kötü olması değil, sıralamanın hedef bölgede tersine dönmesi olasılığıdır. Çıktıyı ters çevirmek problemi teşhis etmek yerine target labels'a bakarak yeni bir karar kuralı seçmek olur.
- **İddia sınırı:** Gözlemsel ilişki vardır; neden-sonuç, tüm yangınlara genelleme veya operasyonel alarm performansı gösterilmemiştir.

## Teknik tasarım

Predictor ve label tarihleri `core/regions.py` içindeki `ExperimentConfig` kayıtlarında ayrıdır. `src/step8a_prepare_500m_modeling_dataset.py` MCD64A1 `BurnDate`'i label-window'un DOY aralığında binary positive yapar. Label raster'ındaki non-burned değer, predictor validity maskesini bozmaz; validity, predictors ve land-cover kullanılabilirliğinden gelir. Bu “label honesty”, burned label'ı feature üretimine veya valid-row seçimine geri sızdırmaz.

Modelleme grid'i 500 m nominaldir. 30 m girdiler `round(500/30)=17` pixels, yani yaklaşık 510 m pencerede summarize edilir (`core/config.py`: Step8 aggregation constants). Continuous variables için mean/median/std/count/fraction; `landcover_dominant` için mode ve sınıf fractions üretilir. “500 m” bu nedenle nominal çözünürlüktür; bütün source pixels'ın tam 500 m geometrik eşdeğeri olduğu iddia edilmez.

Random row split kullanılmaz. `src/step8b_train_baseline_vs_thermal_model.py` hücreleri `row_500m // block_size` ve `col_500m // block_size` ile groups'a atar, `StratifiedGroupKFold` ile bütün group'u tek fold'da tutar. Config primary size 2 cells'tir; formal robustness bunu config'i değiştirmeden runtime'da 10 ve 20 cells ile sınar. Preprocessing yalnız train fold'da fit edilir. Step8 paired bootstrap ve Step9/10 target uncertainty, cell bazında değil spatial block bazında resample eder; böylece yakın hücrelerin yapay biçimde bağımsız sayılması azaltılır. Bu, residual spatial autocorrelation'ın yok edildiği anlamına gelmez.

Within-region değerlendirmede aynı bölgedeki OOF predictions kullanılır; her hücre kendisini görmemiş modelce skorlanır. Cross-region'da preprocessing ve model yalnız source ile fit edilir, target'a doğrudan uygulanır. Step10 target **covariates**'ın mean/std/covariance bilgisini adaptation için kullanır; target labels yalnız son değerlendirmede açılır.

Covariate shift, `P(X)`'in; relationship/concept shift ise `P(Y|X)` veya feature–label sıralama ilişkisinin değişmesidir. Step9E dağılım kayması ile yön değişimi şüphesini; Step9G raw univariate AUC ile doğrudan ilişki yönünü inceler. CORAL sonrası büyük remaining gap ve Step9G reversal birlikte residual concept/relationship shift ile **uyumludur**; tek açıklama olduğunu kanıtlamaz.

**Kendini kontrol:** Label window'dan bir LST görüntüsü predictor stack'e girerse hangi leakage oluşur? Aynı spatial block hem train hem test'e dağılırsa CI neden aşırı iyimser olabilir? AUC 0.32'lik target prediction'ı target labels'a bakarak `1-p` yapmak neden dürüst source-only transfer değildir?

# 4. Bölgeler ve popülasyonlar

## Region registry

| ID | Rol / durum | Predictor | Label | Baseline | İzlenebilir counts | Relevance |
|---|---|---|---|---|---|---|
| `kozan_2023` | Enabled `negative_control` | 2023-06-01–07-31 | 2023-08-01–10-31 | 2019–2022 | 48,422 valid; 542 burned; TSG burned 9; tree+shrub burned 1 | Cropland-dominated control; ana wildfire evidence değil |
| `manavgat_2021` | Enabled `anchor_wildfire` | 2021-06-01–07-27 | 2021-07-28–08-31 | 2017–2020 | 24,150 total; 24,087 valid; 796 burned; TSG 784; tree+shrub 706 | Within-region anchor ve transfer yönü 1 |
| `bejis_2022` | Enabled `mediterranean_transfer_wildfire` | 2022-06-15–08-14 | 2022-08-15–09-30 | 2018–2021 | 15,759 total/valid; 1,103 burned; TSG 1,100; tree+shrub 1,032 | Independent Mediterranean transfer wildfire |
| `mugla_2021` | Enabled `same_country_same_year_transfer_wildfire` | 2021-06-01–07-28 | 2021-07-29–09-15 | 2017–2020 | 73,098 total; 73,045 valid; 3,073 burned; TSG burned 2,952; tree+shrub burned 2,797 | `exclude_pre_label_burns=True` (ayrı Bördübet/Marmaris yangını predictor penceresine düşer, leakage olarak hariç tutulur, negatif sayılmaz); Step5–Step8E ve seam-audit/seam-localization çalıştırıldı, Step9/10 cross-region transfer henüz başlamadı |

`zamora_2022`, `core/regions.py`'deki `EXPERIMENTS` registry'sinden tamamen kaldırıldı (bkz. commit `ab8fc5f`); artık çalıştırılabilir veya disabled bir kayıt olarak bile mevcut değil.

Kaynaklar: registry ve tarihler `core/regions.py`; cell/population counts `outputs/step8a/step8a_dataset_stats.json`, `outputs/experiments/manavgat_2021/step8a/step8a_dataset_stats.json`, `outputs/experiments/bejis_2022/step8a/step8a_dataset_stats.json`, `outputs/experiments/mugla_2021/step8a/step8a_dataset_stats.json`; land-cover gate `outputs/step6/labels/burned_landcover_gate.md` ve experiment Step8A stats. `core/regions.py` içinde bir `valencia_2022_aoi` geometry helper bulunur, ancak `EXPERIMENTS` registry'sinde Valencia experiment yoktur. `docs/experiments.md` bunu disabled experiment gibi gösterir; çalıştırılabilirlik için code registry doğrudur.

## Popülasyon sözleşmeleri

- `all_valid`: predictor validity ve land-cover koşullarını geçen bütün hücreler. Formal Step8 large-block primary analizi, ilk Step8B bilimsel sorusunu aynen korumak için bunu kullanır.
- `burnable_tree_shrub_grass` (TSG): land-cover tree, shrub veya grass olan valid hücreler. Step9, Step10 ve Step9G aynı doğal-vejetasyon transfer sorusunu karşılaştırabilmek için bunu primary population yapar.
- `tree+shrub`: daha dar sensitivity population; grass'i dışlar. Counts raporlanabilir, ancak current primary transfer population değildir.

Formal all-valid robustness ile TSG sensitivity birbirinin yerine geçmez: ilkinde estimand “bütün valid landscape”, ikincisinde “burnable natural vegetation”dır. Manavgat all-valid Step8B'de valid unburned count 23,291 iken Step8A'nın 23,354 `unburned` değeri total grid içindeki invalid unburned hücreleri de içerir; bu iki paydayı karıştırmamak gerekir. Cross-region pair'in TSG source/target counts'ı Manavgat için 20,511/784 positive, Bejís için 15,190/1,100 positive'dir.

# 5. Feature setleri

Baseline feature list ve thermal additions `src/step8b_train_baseline_vs_thermal_model.py` içindeki `BASELINE_FEATURES` ve `THERMAL_FEATURES` sabitleridir.

| Feature | Kaynak / anlam / birim | Zaman ve aggregation | Kullanım | Gerçek bulgu / caveat |
|---|---|---|---|---|
| `ndvi_mean` | Landsat NDVI; vejetasyon yeşilliği, unitless | Predictor window composite; 30 m→~500 m mean | Baseline + thermal | Step9G iki bölgede burned için higher AUC direction; cloud/composite ve phenology etkisi |
| `elevation_mean` | DEM, metre | Statik; mean | Baseline + thermal | Step9G'de Manavgat lower, Bejís higher; tek bootstrap-supported reversal; topography region-specific proxy olabilir |
| `slope_mean` | DEM'den slope, degree olarak izlenebilir | Statik; mean | Baseline + thermal | İki bölgede weak higher direction; DEM türetim/resampling etkisi |
| `lst_anomaly_mean` | Landsat current LST minus baseline climatology, °C | Predictor period; mean | Thermal | İki bölgede same lower direction; climatology/data-availability hassasiyeti |
| `current_lst_mean` | Current Landsat surface temperature, °C | Predictor window; mean | Thermal | Step9E flip; Step9G uncertain higher→lower point reversal; absolute climate/season shift |
| `current_tvdi_mean` | NDVI–LST triangle'dan TVDI, unitless | Predictor period; mean | Thermal | İki bölgede same higher direction; sparse NDVI-bin/edge estimation hassasiyeti |
| `tvdi_difference_mean` | Current TVDI minus baseline TVDI, unitless | Predictor vs baseline farkı; mean | Thermal | Step9E high shift/flip; Step9G uncertain lower→higher point reversal |
| `downscaled_lst_mean` | Step7C RF'nin Landsat LST tahmini | Predictor-date context; mean | Thermal | Step9E high shift/flip; Step9G uncertain higher→lower; target Landsat-derived anomaly/TVDI training feature değildir |
| `fused_lst_mean` | Gözlenen Landsat LST varsa onu, yoksa downscaled LST'yi seçen Step7E ürün | Predictor context; mean | Thermal | Step9E high shift/flip; Step9G uncertain higher→lower; generic blend/weighted average değildir |
| `landcover_dominant` | Land-cover categorical class code; modal sınıf | Statik/ürün yılı; ~500 m mode | Baseline + thermal, categorical | One-hot encoded; Step9G scalar AUC'den çıkarılır çünkü integer code ordinal değildir |

Step7C downscaler'ın target'ı Landsat LST'dir; safe predictors MODIS mean/std, NDVI, elevation, slope ve land-cover'dır. Target-derived `lst_anomaly`, TVDI ve `modis_context_zscore` downscaling features'ına alınmaz (`src/step7c_train_downscaling_model.py`). Step7E “fusion”, observed Landsat'a daima öncelik veren fill kuralıdır (`src/step7e_fuse_landsat_downscaled_lst.py`); iki tahmini optimize edilmiş ağırlıklarla harmanlamaz.

Categorical `landcover_dominant` numeric median ile değil, most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")` ile işlenir. Diğer feature'lar fold-local median imputation alır. Legacy Steps 4–6 daha geniş pixel/raster diagnostics ve anomaly/TVDI ara ürünleri içerir; yalnız yukarıdaki on sütun current Step8 model sözleşmesidir.

# 6. Baştan sona veri akışı

## Bir hücrenin yolculuğu

| Geçiş | Input | Üreten kod | Output / önemli alanlar | Leakage safeguard / failure |
|---|---|---|---|---|
| Raw→preprocess | Landsat 8 C2 L2, MODIS 061 MOD11A1/MCD64A1, DEM, land cover | Steps 1–6 ve experiment orchestrator | GeoTIFF/Parquet/JSON ara ürünler | Region/date registry; Earth Engine/data availability ve CRS/grid mismatch başarısızlıkları |
| Thermal predictors | Current/baseline LST, NDVI | `src/step4_*`, `src/step5*`; TVDI config `core/config.py` | anomaly/TVDI rasters | TVDI: 20 NDVI bins, 2/98 percentiles, min 30 pixels/bin; yetersiz bin span failure |
| Downscale/fuse | MODIS context + safe covariates; Landsat LST target | `src/step7c_train_downscaling_model.py`, `src/step7e_fuse_landsat_downscaled_lst.py` | downscaled/fused LST | Burn labels ve target-derived thermal fields downscaler features'ında yok; observed-first fusion |
| ~500 m aggregate | 30 m predictor rasters | `src/step8a_prepare_500m_modeling_dataset.py` | Parquet/CSV: `cell_id`, row/col, lon/lat, feature stats, validity, populations | Minimum valid fraction 0.30; grid/shape/provenance checks |
| Label join | MCD64A1 `BurnDate` | Step8A label helpers | `burned` binary, BurnDate/DOY provenance | Label-window DOY; label predictor validity'yi belirlemez; FIRMS target değil |
| Population | Step8A table | Step8B/Step9 population filters | all-valid veya TSG table | Population ve counts manifest'e yazılır; estimand karışımı failure |
| Blocks/folds | `row_500m`, `col_500m`, label | Step8B spatial split helpers | group/fold IDs | Whole block aynı fold; insufficient class/group durumunda fail, random fallback yok |
| Preprocess/fit | Train fold rows | Step8B model pipeline | Baseline/thermal RF | Imputer/encoder yalnız train fold; aynı folds iki model için |
| Prediction | Test fold veya target region | Step8B/Step9/Step10 | OOF/target probabilities + label + block | Within: OOF; cross-region: source-only fit; Step10 adaptation target-label-free |
| Metric | Predictions | Step8C, Step9/10 evaluators | ROC-AUC, PR-AUC, Brier (Step10 frozen outputs'ta yok), deltas | One-class sample invalid; no threshold tuning on target |
| Uncertainty/report | Prediction rows grouped by block | paired spatial-block bootstrap/report modules | CSV/JSON/Markdown, manifest, analysis_id | Whole-block resampling, seed 42, usually 1,000; hashes/immutable namespace |

```text
Landsat / MODIS / DEM / land cover
                 |
        region + date contract
                 v
  masks, composites, anomaly, TVDI ----> Step7 safe LST downscaling
                 |                              |
                 +--------------+---------------+
                                v
                    ~500 m Step8A cell table
                    [features | valid | label]
                                |
                   fixed population + blocks
                                v
             fold-local preprocessing + RF fitting
                   /                         \
            baseline OOF                 thermal OOF
                   \                         /
                    metrics + paired block bootstrap
                                |
                      reports + manifests + hashes
```

```text
WITHIN-REGION          RAW CROSS-REGION       ADAPTED CROSS-REGION       POST-HOC DIAGNOSTIC
same region            source train only      source/target X stats      frozen predictions/data
spatial OOF       -->  target score       --> z-score / CORAL       --> Step9E shifts
Step8 metrics          Step9 target metrics   Step10 target metrics       Step9F exploration
large-block check      no target-label fit    labels only evaluation      Step9G raw feature AUC
```

Başlıca failure conditions: missing upstream files, incompatible schemas/grids, one-class fold/bootstrap replicate, population count drift, insufficient groups, protected hash mismatch, existing immutable output without authorized regeneration ve target-label firewall ihlalidir. Üretici/caller zinciri `core/pipeline_orchestrator.py` ile CLI `scripts/main.py` üzerinden bağlanır; standalone runners aynı modüllere daha dar giriş sağlar.
# 7. Repository mimarisi

| Dizin/dosya | Sorumluluk | Statü / ilişki |
|---|---|---|
| `core/config.py` | Sensor collections, resolution, TVDI, Step8 defaults ve legacy Kozan ayarları | Ortak config; legacy hardcoding ile experiment bridge birlikte olduğu için dikkatli değiştirilmeli |
| `core/regions.py` | AOI builders, `ExperimentConfig`, `EXPERIMENTS`, enabled/role/date sözleşmesi | Experiment truth source |
| `core/pipeline_orchestrator.py` | Stage planlama, upstream/downstream sırası, experiment namespacing | Canonical orchestration |
| `src/` | Steps 1–10 bilimsel implementasyonları ve report builders | Çoğu canonical; `legacy`/eski Step ağacı Kozan yolu |
| `scripts/main.py` | Subcommand parser ve runner dispatch | Ana CLI truth source |
| `scripts/run_*.py` | Belirli analizleri doğrudan çağıran standalone runners | Desteklenen fakat ana CLI ile kısmen duplicate girişler |
| `tests/` | CLI, orchestrator ve Step8 robustness/Step10/Step9G invariant tests | Kod davranışı kanıtı; bilimsel doğruluğun kendisi değil |
| `outputs/experiments/<id>/` | Region-namespaced within-region ara/final çıktılar | Manavgat/Bejís Step8 canonical |
| `outputs/cross_region/<source>__<target>/` | Step9 raw/audit/exploration ve Step10 pair çıktıları | Frozen/canonical alt ağaca göre ayrılır |
| `outputs/robustness/` | Step8 large-block formal ve sensitivity ağaçları | Ayrı protected namespaces |
| `outputs/diagnostics/` | Step9G numeric ve integration artifacts | Canonical isim dikkatle seçilmeli |
| `docs/`, `README.md` | Kullanıcı anlatımı ve deney notları | Yararlı fakat code/output'tan düşük öncelikli; bazı kısımlar stale |
| `requirements.txt`, `requirements-lock.txt` | Aralık bazlı ve pinlenmiş Python dependencies | Reproducibility başlangıcı; external data/EE state'i pinlemez |
| `old_codes/` | Eski denemeler | Non-canonical; güncel pipeline için kaynak gösterilmemeli |

Önemli modül ilişkileri:

- `src/step8a_prepare_500m_modeling_dataset.py` upstream rasterları cell table'a dönüştürür; orchestrator ve experiment runner çağırır. `cell_id`, grid coordinates, predictor statistics, label/population/provenance üretir.
- `src/step8b_train_baseline_vs_thermal_model.py` feature contracts, spatial folds, sklearn pipelines ve OOF predictions'ın merkezidir. Step8 robustness ve cross-region code aynı model sözleşmesini reuse eder.
- Step8C/D/E modules bootstrap, ablation ve final reporting yapar; input'u Step8B prediction/metric artifacts'tır.
- Step9 modules source-only transfer, Step9E diagnostics ve Step9F exploration'ı ayırır. Bu ayrım target-label-free fitting ile post-hoc target-label analysis arasındaki sınırı korur.
- Step10 modules preregistration, target-label-free transformation/model scoring ve son target evaluation/reporting'i ayrı stages yapar.
- Step8 large-block ve Step9G modules protected hash, manifest ve equivalence checks içeren formel analizlerdir.

## “Nerede düzenlemeliyim?” tablosu

| İş | İlk bakılacak yer | Birlikte kontrol edilecek yer |
|---|---|---|
| Experiment eklemek/değiştirmek | `core/regions.py` | `core/pipeline_orchestrator.py`, CLI/tests, region docs |
| Model feature listesi | `src/step8b_train_baseline_vs_thermal_model.py` | Step9/10/9G contracts ve frozen compatibility |
| Spatial-block davranışı | Step8B split helpers | large-block modules/tests; `core/config.py` default |
| CLI command | `scripts/main.py` | ilgili `scripts/run_*.py`, `tests/test_main_cli.py` |
| Report generation | İlgili Step'in report module'u | JSON/CSV schema, manifests, invariant tests |
| Test eklemek | `tests/test_<step>.py` | Küçük deterministic fixture; frozen path'ten izole temp dir |
| Metric araştırmak | Canonical final JSON/CSV + prediction table | Evaluator/bootstrap module; population/analysis_id |
| Provenance kontrolü | `manifest*.json`, `preregistration*.json`, final report | Input hashes, git/config snapshot, protected output hashes |

# 8. CLI ve çalıştırma modeli

Ana hiyerarşi gerçek `--help` çıktısına göre `experiment`, `transfer`, `shift-audit`, `transfer-explore`, `self-cal-transfer`, `step10`, `step8-robustness`, `large-block-robustness`, `concept-shift`, `legacy` komutlarıdır (`scripts/main.py`). Aşağıdaki örnekler project root'tan çalışır.

```bash
venv/bin/python scripts/main.py experiment \
  --experiment manavgat_2021 --from-stage predictors --to-stage step8 \
  --predictor-mode local-only --dry-run

venv/bin/python scripts/main.py transfer \
  --source manavgat_2021 --target bejis_2022 --reverse --dry-run

venv/bin/python scripts/main.py shift-audit \
  --source manavgat_2021 --target bejis_2022

venv/bin/python scripts/main.py transfer-explore \
  --source manavgat_2021 --target bejis_2022

venv/bin/python scripts/main.py step10 \
  --source manavgat_2021 --target bejis_2022 --report-only

venv/bin/python scripts/main.py large-block-robustness --dry-run
# Yalnız bilinçli, pahalı gerçek çalıştırma:
venv/bin/python scripts/main.py large-block-robustness --run-large-block-fit

venv/bin/python scripts/main.py concept-shift --dry-run
venv/bin/python scripts/main.py concept-shift --integration-only

venv/bin/python scripts/main.py legacy --experiment kozan_2023 --dry-run
```

`experiment` upstream predictors'dan Step8'e kadar expensive işler tetikleyebilir. `transfer` source-only fit ve target scoring yazar. `shift-audit` yeni model değildir. `transfer-explore` target sonuçları daha önce görülmüş exploratory taramadır. `step10`/`self-cal-transfer` aynı self-calibrated aileye iki alias'tır; `--report-only` numeric outputs'u yeniden üretmez. `large-block-robustness` gerçek RF/1,000-bootstrap işini yalnız açık fit flag'iyle yapar. `concept-shift --integration-only` numeric job'ı tekrarlamadan canonical integration-v2 raporunu kurar. `legacy` Kozan yoludur.

`--dry-run` güvenli plan denetimidir. `--force`, yalnız hedef namespace disposable ise güvenlidir; frozen/canonical outputs üzerinde tehlikelidir. Exact flags her zaman `--help` ile kontrol edilmelidir. `STEP8B_SPATIAL_BLOCK_SIZE_CELLS = 2` korunur çünkü ilk Step8B'nin reproduction referansıdır; 10/20 values runtime override ve ayrı robustness namespace'indedir.

**Dokümantasyon çatışması:** `docs/experiments.md` eski syntax ve yalnız Kozan executable anlatısı taşır; command truth `scripts/main.py --help`, experiment truth `core/regions.py` ve tests'tir.

# 9. Step8 within-region modelleme

Step8A upstream rasterları yaklaşık-500 m grid'e getirir; feature stats, `landcover_dominant`, validity/population flags ve label üretir. `burned=0` missing değildir; label validity'yi belirlemez. Step8B aynı spatial folds üzerinde iki pipeline fit eder: baseline (`ndvi_mean`, `elevation_mean`, `slope_mean`, `landcover_dominant`) ve bunlara altı thermal feature ekleyen thermal model.

Numeric columns train-fold median; categorical column train-fold most-frequent + one-hot ile işlenir. RF: 300 trees, `max_depth=None`, `min_samples_leaf=3`, `class_weight="balanced"`, `random_state=42`, `n_jobs=-1` (`src/step8b_train_baseline_vs_thermal_model.py`). Beş `StratifiedGroupKFold` fold'unda test fold'a preprocessing fit edilmez; her satır OOF probability alır. İki model aynı folds'u paylaşır.

ROC-AUC ranking, PR-AUC imbalanced positive retrieval, Brier probability squared error ölçer. Step8C whole-block paired bootstrap ile thermal-minus-baseline CI üretir. CI “statistical significance” değildir. Step8D ablation; Step8E report integration'dır. Absolute performance modelin düzeyini, delta ise thermal grubun baseline üstündeki katkısını ölçer.

| Canonical all-valid | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|
| Manavgat baseline | 0.827667 | 0.122549 | 0.063704 |
| Manavgat thermal | 0.886610 | 0.222185 | 0.043745 |
| Manavgat delta [CI] | +0.058943 [0.048980, 0.068358] | +0.099636 [0.073086, 0.129382] | −0.019959 [−0.021533, −0.018244] |
| Bejís baseline | 0.868831 | 0.309259 | 0.089160 |
| Bejís thermal | 0.917055 | 0.486975 | 0.061661 |
| Bejís delta [CI] | +0.048224 [0.039813, 0.057128] | +0.177716 [0.140051, 0.215839] | −0.027499 [−0.030261, −0.024745] |

TSG sensitivity de aynı yöndedir: Manavgat delta ROC +0.066906 [0.054559, 0.078572], PR +0.096790 [0.070744, 0.122664]; Bejís delta ROC +0.056133 [0.047853, 0.065327], PR +0.195034 [0.155615, 0.231023]. Within-region evidence başka bölgeye portability kanıtı değildir.

# 10. Step8 büyük blok robustness

2-cell referans yaklaşık 1 km groups verir. Formal runner labels, folds, block assignments, predictions ve metrics'i `cell_id` ile hizalayıp tolerance `1e-12` altında exact equivalence ister. Sonra runtime 10 cells ≈5 km ve 20 cells ≈10 km uygular; config'deki 2 değişmez. Primary `all_valid`, sensitivity TSG'dir. Original Step8 ve v1 tree hashes korunur; ayrı namespace, immutable manifest ve analysis ID overwrite/estimand drift'i önler.

| Formal all-valid | Delta ROC-AUC (95% block-bootstrap CI) | Delta PR-AUC (95% CI) |
|---|---:|---:|
| Manavgat 5 km | +0.053243 [0.029428, 0.074860] | +0.048781 [0.016436, 0.093869] |
| Manavgat 10 km | +0.044433 [0.014663, 0.077631] | +0.026045 [0.000626, 0.055532] |
| Bejís 5 km | +0.055725 [0.030753, 0.078098] | +0.134358 [0.059101, 0.226400] |
| Bejís 10 km | +0.061286 [0.039680, 0.087105] | +0.076055 [0.014015, 0.155429] |

Contrast dört kombinasyonda sürer. PR support en güçlü Bejís 5 km, sıfıra en yakın Manavgat 10 km'dir. Bu, residual spatial autocorrelation'ı yok etmez; yalnız daha kaba predefined grouping'e robustness gösterir. Canonical: `outputs/robustness/step8_large_block_primary_all_valid/manavgat_2021__bejis_2022/step8_large_block_primary_all_valid_final_report.md` (`bacb7df…`). Natural sensitivity (`1759eed…`) ayrı estimand'dır.

# 11. Step9 raw cross-region transfer

Step9A pair/population'ı dondurur; Step9B preprocessing, RF ve threshold'u yalnız source data/OOF ile öğrenir; Step9C target covariates'ı skorlar; Step9D target labels'ı yalnız evaluation ve target whole-block bootstrap için açar.

| Yön / model | Target ROC-AUC | PR-AUC | Brier | Source-OOF threshold |
|---|---:|---:|---:|---:|
| Manavgat→Bejís baseline | 0.332187 | 0.049285 | 0.124340 | 0.35 |
| Manavgat→Bejís thermal | 0.325834 | 0.048758 | 0.088173 | 0.50 |
| delta [CI] | −0.006354 [−0.023499, 0.009666] | −0.000527 [−0.001785, 0.000565] | −0.036167 [−0.038713, −0.033809] | — |
| Bejís→Manavgat baseline | 0.420908 | 0.033190 | 0.107849 | 0.45 |
| Bejís→Manavgat thermal | 0.443528 | 0.034406 | 0.089198 | 0.45 |
| delta [CI] | +0.022621 [−0.000449, 0.043834] | +0.001216 [−0.001190, 0.003587] | −0.018651 [−0.021006, −0.016235] | — |

İki yönde thermal raw ROC-AUC <0.5: discrimination transfer başarısızdır ve ordering reversal önemlidir. Brier iyileşmesi prevalence'a yakın probabilities yüzünden olabilir; ranking başarısını kanıtlamaz. Step9D machine label `partial_transfer_supported`, Brier nedeniyle oluşur; “discrimination supported” diye okunmamalıdır.

# 12. Step9E distribution ve relationship-shift audit

Step9E frozen Step9 data/predictions üzerinde (1) covariate distribution shift, (2) probability-scale shift, (3) ranking-reversal suspicion ve (4) feature–label direction flip diagnostics üretir. `tvdi_difference_mean`, `downscaled_lst_mean`, `current_lst_mean`, `fused_lst_mean` high; `lst_anomaly_mean`, `slope_mean` moderate shift'tir. Flip listesi `current_lst_mean`, `downscaled_lst_mean`, `elevation_mean`, `fused_lst_mean`, `tvdi_difference_mean`dir. Manavgat→Bejís thermal score'un source threshold üstünde kalan target fraction'ı yaklaşık 0.005'tir.

Step9E post-hoc diagnostic'tir; model/adaptation değildir. Target outcomes görüldükten sonra hipotez üretir. Step9G aynı yön sorusunu fixed feature list, 10-cell block bootstrap ve no-inversion kuralıyla daha dar confirmatory diagnostic haline getirir.
# 13. Step9F exploratory feature representations

Step9F, Step9/9E sonuçları görüldükten sonra yapılan **exploratory** representation audit'tir. Sekiz strict source-only varyant şunlardır:

| Varyant | Değişiklik / gerekçe |
|---|---|
| `original_baseline` | Thermal yok; referans |
| `original_thermal` | On-feature current thermal referansı |
| `thermal_without_elevation` | Step9E elevation flip şüphesini çıkarır |
| `thermal_without_absolute_lst` | `current_lst`, `downscaled_lst`, `fused_lst` absolute temperature taşıyıcılarını çıkarır |
| `thermal_without_tvdi_difference` | En yüksek shift göstergelerinden fark feature'ını çıkarır |
| `thermal_without_elevation_or_absolute_lst` | İki instability grubunu birlikte çıkarır |
| `stable_core` | Yalnız daha stable görünen küçük feature çekirdeği |
| `stable_core_without_landcover` | Region-specific categorical composition etkisini de çıkarır |

Bunların imputation/preprocessing/model fitting'i yalnız source ile yapılır. Adaptive regime yalnız `original_thermal` ve `stable_core` için ayrıca çalışır; source ve unlabeled target covariates kendi-region median/IQR ile robust normalize edilir. Target labels transformation seçimi veya fitting'de kullanılmaz. Bu iki adaptive varyant strict source-only sonuçlarla aynı tabloda yorumlansa da farklı information regime'dedir.

Beklenti, absolute-LST/elevation gibi unstable bileşenleri çıkarınca reversal azalırken source OOF değerinin makul kalmasıydı. Gerçek sonuçlarda `stable_core` ranking reversal'ı azaltabildi fakat source OOF kaybı yarattı; diğer removals da source/target trade-off'ları gösterdi. Step9B reproduction tolerance `1e-6` içinde doğrulandı. Hiçbir aday predefined third-region freeze screening'i geçmedi; dolayısıyla “transfer-safe final representation” yoktur ve yeni bir üçüncü-bölge confirmatory run başlatılmamıştır. Target outcomes Step9'dan beri görüldüğü için Step9F bulguları model seçimi kanıtı değil, sonraki bağımsız çalışmaya hipotezdir.

# 14. Step10 self-calibrated transfer

Step10 üç thermal yöntemi karşılaştırır:

- **Raw source-only:** Step9'un değişmeden yeniden üretilen pipeline'ı.
- **Region-wise z-score:** Her numeric feature source'ta source mean/std, target'ta target mean/std ile standardize edilir (`ddof=0`). Missing numeric values kendi-region mean ile doldurulduğunda z-space'de 0 olur. `landcover_dominant` standardize edilmez; source-fitted one-hot sözleşmesi korunur.
- **CORAL after z-score:** Numeric source covariance, target covariance'a `lambda=1e-5` regularization ile hizalanır; target numeric matrix değişmeden kalır. Land cover CORAL'a girmez.

Adaptation source ve target **X** statistics kullanabilir; target **y** Step10B transform/model/score sırasında firewall dışındadır. Target labels yalnız Step10C evaluation'da yüklenir. Seed 42 ve 1,000 target-spatial-block bootstrap, raw/z/CORAL'ı aynı replicates içinde paired karşılaştırır. Frozen Step10 outputs Brier raporlamaz; eklenmemelidir.

| Yön / yöntem | Thermal ROC-AUC [95% CI] | Thermal PR-AUC [95% CI] |
|---|---:|---:|
| Manavgat→Bejís raw | 0.325834 [0.304515, 0.348912] | 0.048758 [0.043432, 0.054826] |
| Manavgat→Bejís z-score | 0.477100 [0.450665, 0.501836] | 0.066667 [0.058435, 0.076135] |
| Manavgat→Bejís CORAL | 0.510540 [0.483953, 0.534211] | 0.069581 [0.061217, 0.079083] |
| Bejís→Manavgat raw | 0.443528 [0.407765, 0.479941] | 0.034406 [0.028893, 0.041660] |
| Bejís→Manavgat z-score | 0.457338 [0.419580, 0.496820] | 0.038513 [0.032079, 0.046813] |
| Bejís→Manavgat CORAL | 0.555310 [0.527802, 0.582844] | 0.042710 [0.036622, 0.049805] |

`recovered_covariate_shift = adapted − raw`; `remaining_gap = within-region thermal − adapted` olarak yorumlanır. Manavgat→Bejís CORAL raw'a göre ROC +0.184706 [0.161060, 0.205301], PR +0.020823 [0.016613, 0.025963] recovery sağlar; fakat remaining ROC gap 0.407271, PR gap 0.428227'dir. Bejís→Manavgat CORAL recovery ROC +0.111781 [0.070108, 0.151152], PR +0.008303 [0.002225, 0.013508]; remaining ROC gap 0.314332, PR gap 0.172917'dir. Z-score recovery ilk yönde nettir; ters yönde ROC +0.013810 [−0.015488, 0.044426] belirsiz, PR artışı küçüktür.

Pipeline'da açıkça yeniden üretilen prototype expectation, Bejís→Manavgat CORAL'ın chance üstü discrimination'ıdır. Manavgat→Bejís CORAL point estimate 0.5105 olsa da CI 0.5'i içerir; above-chance desteklenmez. Asimetri, iki region'ın marginal moments ve feature–label ilişkilerinin simetrik olmamasından beklenebilir; iki vaka ile mekanizması belirlenemez. Güvenli sonuç “unsupervised adaptation covariate kaynaklı kaybın bir kısmını, yön-bağımlı biçimde geri aldı”dır; “general transfer başarılı” değildir.

# 15. Step9G univariate feature-AUC direction reversal

Step9G her continuous feature'ı tek başına bir score gibi kullanır. Raw AUC 0.5 üstünde ise burned cells daha yüksek; altında ise daha düşüktür. AUC ters çevrilmez, çünkü yön farkı araştırma nesnesidir. Normalization/imputation yoktur; her feature kendi complete cases'ını kullanır. Population TSG, blocks fixed 10 cells, bootstrap 1,000 ve seed 42'dir; replicate en az 900 valid olmalıdır. `landcover_dominant`, integer kodların ordinal anlamı olmadığı için scalar AUC'den çıkarılır.

| Feature | Manavgat AUC [CI] / yön | Bejís AUC [CI] / yön | Point reversal | Bootstrap support | Step9E |
|---|---|---|---|---|---|
| `ndvi_mean` | 0.6359 [0.5872, 0.6763] higher | 0.5588 [0.4972, 0.6193] higher | Hayır | — | Aynı yön |
| `elevation_mean` | 0.3741 [0.2891, 0.4712] lower | 0.6433 [0.5583, 0.7290] higher | Evet | **Evet** | Flip ile uyumlu |
| `slope_mean` | 0.5310 [0.4228, 0.6417] higher | 0.5208 [0.4393, 0.6053] higher | Hayır | — | Aynı yön/moderate shift |
| `lst_anomaly_mean` | 0.4824 [0.4280, 0.5299] lower | 0.4180 [0.3638, 0.4796] lower | Hayır | — | Aynı yön |
| `current_lst_mean` | 0.5383 [0.4518, 0.6205] higher | 0.4767 [0.4012, 0.5475] lower | Evet | Hayır, uncertain | Flip şüphesiyle uyumlu |
| `current_tvdi_mean` | 0.5520 [0.4602, 0.6411] higher | 0.5173 [0.4290, 0.5952] higher | Hayır | — | Aynı yön |
| `tvdi_difference_mean` | 0.4494 [0.3905, 0.5052] lower | 0.5125 [0.4432, 0.5829] higher | Evet | Hayır, uncertain | Flip şüphesiyle uyumlu |
| `downscaled_lst_mean` | 0.5521 [0.4660, 0.6372] higher | 0.4836 [0.4004, 0.5598] lower | Evet | Hayır, uncertain | Flip şüphesiyle uyumlu |
| `fused_lst_mean` | 0.5401 [0.4543, 0.6219] higher | 0.4806 [0.4039, 0.5514] lower | Evet | Hayır, uncertain | Flip şüphesiyle uyumlu |

Point reversal yalnız iki point estimate'ın 0.5'in farklı tarafında olmasıdır. Bootstrap-supported reversal, her region interval'ının chance yönünden ayrılması ve ters yönlerin resampling altında korunmasıdır; yalnız `elevation_mean` bunu sağlar. Dört thermal feature point reversal'ı bilimsel sinyaldir ama belirsizdir.

Entegrasyon: Step9E beş flip'i post-hoc işaretledi; Step9G elevation'ı güçlendirdi ve dört thermal flip'in uncertainty'sini gösterdi. Step9F feature removal ile reversal–source performance trade-off'u buldu fakat final temsil seçemedi. Step10 marginal/covariance alignment'ın bir kısmını kurtarıp büyük remaining gap bıraktı. Birlikte, covariate shift + residual relationship instability örüntüsünü destekler. Numeric truth `outputs/diagnostics/step9g_univariate_feature_auc_direction_reversal/`; canonical narrative `outputs/diagnostics/step9g_univariate_feature_auc_direction_reversal_integration_v2/manavgat_2021__bejis_2022/step9g_integration_v2_final_report.md` (integration ID `2b7dd…`). Aynı yerdeki eski `step9g_integration_correction_final_report.*` (`1a9b…`) canonical değildir.

# 16. Güncel bilimsel hikâye

| Claim | Destek? / kanıt | Güvenli ifade | Yasak/aşırı ifade |
|---|---|---|---|
| Within-region thermal contribution | Güçlü: Step8 paired deltas, iki region | “Thermal grup aynı-region OOF'da baseline'a ek ayrım ve Brier katkısı verdi.” | “Thermal conditions yangına neden oldu.” |
| Daha büyük bloklarda sürer | Güçlü sensitivity: formal 5/10 km delta CI'lar pozitif | “Kontrast predefined coarser blocks altında korundu.” | “Spatial autocorrelation eliminated.” |
| Raw cross-region discrimination fails | Güçlü negatif: iki raw thermal ROC <0.5 | “Source ordering target'a taşınmadı; anti-predictive raw AUC gözlendi.” | “Model genel olarak wildfire prediction yapıyor.” |
| Adaptation partial/asymmetric recovery | Güçlü paired recovery; chance support yalnız bir CORAL yönünde | “Unlabeled covariate alignment kaybın bir kısmını yön-bağımlı geri aldı.” | “Domain adaptation transfer sorununu çözdü.” |
| Büyük remaining gap | Güçlü: bütün adapted-vs-within comparisons | “Adapted performans within-region düzeyinin çok altında kaldı.” | “Yalnız normalization eksikti.” |
| Feature direction instability | Elevation için güçlü; dört thermal için uncertain | “En az bir supported ve dört uncertain point reversal gözlendi.” | “Bütün thermal physics tersine döndü.” |
| Residual concept/relationship shift | Birleşik pattern ile consistent, mekanizma kanıtı değil | “Sonuçlar covariate shift ötesinde residual relationship shift ile uyumludur.” | “Concept shift kanıtlandı ve tek açıklamadır.” |

Tam zincir: (1) Step8 yerel katkı, (2) formal large-block robustness, (3) Step9 raw failure, (4) Step10 partial/asymmetric recovery, (5) large remaining gap, (6) Step9E/G direction instability, (7) residual concept/relationship shift ile uyum. Sınırlamalar: yalnız iki doğal-vejetasyon wildfire region, observational labels, MCD64A1 resolution/temporal uncertainty, RF-specific representation ve post-hoc diagnostic geçmişi.

# 17. Teknik güvenlik ve leakage denetimi

| Risk | Safeguard | Uygulama/kanıt |
|---|---|---|
| Target-label leakage | Step9/10 adaptation/fitting'e `y_target` verilmez; ayrı evaluation stage | Step10 target-label firewall tests; Step9 source-only runners |
| Random spatial leakage | `StratifiedGroupKFold`, whole block in one fold; random fallback yok | Step8B split helpers; large-block fold tests |
| Preprocessing leakage | imputer/one-hot sklearn pipeline train fold/source'a fit | Step8B model pipeline; fold-local preprocessing tests |
| Threshold leakage | Operating threshold source OOF'tan seçilir | Step9B artifacts; target yalnız evaluation |
| Calibration leakage | Step10 yalnız unlabeled X moments/covariance kullanır | z-score/CORAL implementation ve label-independence tests |
| Output overwriting | Separate namespaces, existing-output guards, `--force` explicit | CLI/runners/manifests |
| Post-hoc favorable selection | Step9F exploratory etiketi; freeze screening ve no selected candidate | Step9F prereg/report |
| Frozen analyses mutation | Protected file/tree hashes, immutable prereg/manifest, analysis_id | formal robustness, Step10, Step9G tests/artifacts |
| Pipeline drift | 2-cell exact equivalence gate, raw/within reproduction checks | large-block ve Step10 reports/tests |

Preregistration, analize başlamadan feature/population/method/metric sözleşmesini dondurur. `analysis_id` bu sözleşmenin kimliğidir. Manifest input/output paths, hashes, counts ve parameters taşır. Hash içerik değişimini yakalar; sadece filename'i korumaz. `--dry-run` planı yazmadan/fit etmeden gösterir. `--force` guard'ı bilinçli aşar ve canonical evidence üzerinde normal çalışma aracı değildir. Equivalence gate yeni runner'ın reference sonucu aynı ürettiğini doğrular; scientific validity'yi tek başına ispatlamaz.

# 18. Test mimarisi

| Modül/konu | Ana tests | Başlıca invariant |
|---|---|---|
| CLI | `tests/test_main_cli.py` | Subcommand routing, aliases, dry-run/flags |
| Orchestration | `tests/test_pipeline_orchestrator.py` | Stage ranges, namespacing, plan |
| Step9F | `tests/test_step9f.py` | Variant contracts, reproduction, target information regime |
| Step10 | `tests/test_step10.py` | z-score/CORAL math, target-label independence/firewall, paired bootstrap, prereg immutability, raw/within reproduction |
| Large-block v1 | `tests/test_step8_large_block_robustness.py` | 2/10/20 plan, deterministic/nested assignments, no fold overlap, OOF coverage, block bootstrap, one-class invalidation, protected hashes |
| Formal all-valid | `tests/test_step8_large_block_robustness_primary_all_valid.py` | all-valid primary, exact 2-cell equivalence, v1 protection, explicit fit flag |
| Step9G numeric | `tests/test_step9g_univariate_feature_auc_direction_reversal.py` | fixed features/population/blocks, raw AUC, no impute/invert, whole-block bootstrap, hashes |
| Step9G integration | integration-correction test modules | canonical joins ve report identities |

Targeted test, değiştirilen contract'ın dosyasını hızlı sınar; full suite regresyonları tarar. Bootstrap tests bütün block'un resample edildiğini ve paired replicates'i; equivalence tests reference identity'yi; protected-hash tests frozen ağacın değişmediğini denetler. Passing tests implementation'ın declared invariant'larını karşılar. Remote satellite data'nın doğruluğunu, CRS'nin bilimsel uygunluğunu, wildfire mekanizmasını, causal claim'i veya iki vakadan generalization'ı kanıtlamaz. Steps 1–8'in birçok heavy geospatial yolu son aşamalar kadar yoğun unit test kapsamına sahip değildir.

# 19. Canonical ve non-canonical çıktılar

| Step | Canonical directory/report | Supporting artifacts | Frozen/regeneration | Duplicate/obsolete |
|---|---|---|---|---|
| Kozan Step6 | `outputs/step6/labels/burned_landcover_gate.md` | JSON/rasters | Legacy evidence; değiştirilmemeli | Kozan ana wildfire result değil |
| Step8 Manavgat | `outputs/experiments/manavgat_2021/step8*/` final reports | Step8A dataset stats, OOF predictions, metrics, bootstrap/ablation tables | Canonical frozen evidence | Ad-hoc rerun yapma |
| Step8 Bejís | `outputs/experiments/bejis_2022/step8*/` | Aynı schema | Canonical frozen | — |
| Step9–10 pair | `outputs/cross_region/manavgat_2021__bejis_2022/` ve reverse artifacts | predictions, metrics, bootstrap, prereg/manifests, Step9E/F/10 reports | Canonical alt-ağaçlar frozen | Step9F exploratory; Step9D label dikkatli yorumlanmalı |
| Formal large block | `outputs/robustness/step8_large_block_primary_all_valid/manavgat_2021__bejis_2022/step8_large_block_primary_all_valid_final_report.md` | per-block metrics/CI, manifest, hashes | **Canonical formal, frozen** | v1/natural ile karıştırma |
| Natural sensitivity | ayrı `outputs/robustness/` v1/sensitivity tree | 2/10/20 TSG results, manifest | Frozen sensitivity | Formal primary değil |
| Step9G numeric | `outputs/diagnostics/step9g_univariate_feature_auc_direction_reversal/` | feature AUC/CI/reversal CSV/JSON, manifest | **Canonical numeric, frozen** | — |
| Step9G integration-v2 | `outputs/diagnostics/step9g_univariate_feature_auc_direction_reversal_integration_v2/manavgat_2021__bejis_2022/step9g_integration_v2_final_report.md` | Step9E/F/10 join tables/JSON | **Canonical integration** | Aynı dir'deki `step9g_integration_correction_final_report.*` eski/non-canonical |

Bir directory adı tek başına canonicality kanıtı değildir: population, analysis ID, report filename ve manifest birlikte okunmalıdır. `old_codes/`, scratch outputs ve README'de anılan fakat registry/output karşılığı olmayan regions bilimsel kaynak sayılmaz.
# 20. Şu an proje hangi noktada?

## Teknik olarak tamamlananlar

- Manavgat ve Bejís experiment-namespaced Step8A–E artifacts; iki-yönlü Step9 raw transfer; Step9E audit; Step9F exploratory representations; Step10 raw/z-score/CORAL; formal all-valid 5/10 km robustness; Step9G numeric ve canonical integration-v2 raporu mevcuttur.
- Ana CLI bu analiz ailelerini route eder; latest scientific analyses için preregistration/manifest/hash/equivalence safeguards ve tests vardır.
- Kozan gate, onu wildfire anchor yerine cropland-dominated control olarak konumlandıracak kadar açıktır.

## Bilimsel olarak desteklenenler

- İki wildfire region içinde thermal-minus-baseline katkısı.
- Aynı kontrastın predefined 5/10 km spatial blocks altında korunması.
- Raw cross-region discrimination'ın iki yönde başarısızlığı.
- Unsupervised adaptation'ın bazı metrics'te recovery sağladığı, fakat bunun asimetrik ve eksik olduğu.
- `elevation_mean` için bootstrap-supported direction reversal; dört thermal feature için uncertain point reversal.

## Desteklenmeyen/negatif bulgular

- General successful cross-region discrimination, universal transfer-safe representation, symmetric above-chance adaptation, causality ve operational early warning desteklenmez.
- Step9F'te third-region freeze kriterini geçen aday yoktur.
- Manavgat→Bejís CORAL point ROC >0.5 olsa da CI chance'ı içerir.

## Yalnız exploratory

- Step9E'nin target-label-informed shift hypotheses ve Step9F variant taraması.
- Step9G bu hipotezlerin bir kısmını daha formal ölçer, fakat yalnız iki region ile universal mechanism kanıtlamaz.

## Bekleyen işler ve danışman kararı

- README/docs command/status drift'i, duplicate runner/artifact naming ve canonical output index'i temizlenebilir; final visualization/communication package README'de eksik görünür.
- Repository kronolojisinden en son istek zinciri formal `all_valid` large-block correction ve Step9G numeric + integration-v2 olarak anlaşılır; ikisi de tamamlanmış artifacts/tests ile temsil edilir. Ancak repository'de danışman toplantısının verbatim kaydı yoktur; “danışmanın en son söylediği tam cümle” doğrulanamaz.
- Yeni region, model, feature veya adaptation deneyi için current repository'de onaylanmış preregistered next question yoktur. Bu nedenle yeni danışman yönlendirmesi gelmeden yeni bilimsel experiment başlatılmamalı; mevcut kanıt zinciri ve claim boundaries danışmanla gözden geçirilmelidir.

# 21. Bilinen sorunlar ve teknik borç

Bu incelemede **critical** düzeyde kanıtı bozmuş aktif hata görülmedi. Aşağıdaki gerçek sorunlar önceliklidir.

| Öncelik | Sorun / kanıt | Risk | En güvenli sonraki eylem |
|---|---|---|---|
| Medium | `docs/experiments.md` current CLI/registry ile çelişiyor; Valencia'yı experiment gibi anıyor | Yanlış command/status ile run | Code/output değiştirmeden docs truth table güncelle |
| Medium | `core/config.py` legacy Kozan hardcoding ve experiment bridge'i birlikte taşır | İki config source'u drift edebilir | Config ownership sözleşmesini test/doc ile netleştir; scientific defaults'u sessizce taşıma |
| Medium | Step9D `partial_transfer_supported` etiketi Brier nedeniyle, fakat ROC/PR transfer failed | Bilimsel overclaim | Report prose/legend'de “probability error only” sınırını zorunlu kıl |
| Medium | Step9G integration-v2 directory içinde eski `step9g_integration_correction_final_report.*` de var | Yanlış report cite edilir | Canonical index/manifest ekle; frozen dosyaları silme/rename etme |
| Medium | Çok büyük modules: ör. Step6 ~2.7k, Step8A ~2.5k, Step8B ~1.4k lines | Review/test izolasyonu zor | Yeni bilimsel sonuç olmadan, invariant tests altında küçük pure helpers'a kademeli refactor |
| Medium | Outputs git dışında/çok büyük ve local external data/EE state'e bağlı | Başka makinede exact reproduction zor | Artifact inventory, checksum bundle ve environment/data access recipe hazırla |
| Medium | Requirements aralıkları ile lock pins zamanla drift edebilir; `.env.example` dokümantasyon iddiası repository state ile doğrulanmalı | Environment kurulumu şaşar | Clean environment smoke test; secrets içermeyen template/path kontrolü |
| Low | `self-cal-transfer` ve `step10`, main dispatch ve standalone runners duplicate user surfaces | Kullanıcı yanlış entry seçebilir | Birini documented canonical alias yap, shared handler'ı koru |
| Low | `old_codes/` ve legacy yollar kolayca current sanılabilir | Non-canonical code cite/run | Canonical map ve directory-level warning; silme kararı ayrı cleanup task |
| Low | README line/path references ve “next third region” önerileri current advisor-hold ile uyumsuz olabilir | Premature experiment | Current status/claim boundary bölümünü güncelle, yeni run yapma |
| Low | Output path conventions farklı analiz ailelerinde uzun ve benzer | İnsan hatası | Read-only output index generator düşün; analysis ID/population göster |

# 22. Projeyi değiştirmek için güvenli çalışma rehberi

1. **Inspect:** clean/dirty state, registry, related source/tests ve canonical manifest'i oku.
2. **Scientific question:** Tek cümle estimand, population, time windows, comparison ve failure criterion yaz.
3. **Preregister:** Features, data information regime, folds/blocks, metrics, bootstrap, outputs ve no-go decisions'ı run'dan önce dondur.
4. **Code:** En küçük değişiklik; frozen path'e yazma; yeni namespace/version kullan.
5. **Targeted tests:** Değişen scientific invariant'ı deterministic fixture ile test et.
6. **Dry-run:** Paths, stages, flags, hashes ve overwrite planını incele.
7. **Real run:** Yalnız prerequisite/freeze onayıyla; log/manifest'i sakla.
8. **Outputs:** Counts, schema, one-class/fold coverage, input hashes ve exact command'i denetle.
9. **Interpretation:** Point metric, block-bootstrap CI, negative results ve claim boundary'yi birlikte yaz.
10. **Commit/docs:** Code, test, prereg/report index ve rationale'ı tek tutarlı change set yap; heavy outputs policy'sine uy.

```bash
# Durum ve diff
git status --short --branch
git diff --check
git diff --stat
git diff -- core/regions.py src/ tests/ scripts/

# Testler
venv/bin/python -m pytest -q tests/test_step10.py
venv/bin/python -m pytest -q tests/test_step9g_univariate_feature_auc_direction_reversal.py
venv/bin/python -m pytest -q

# CLI truth
venv/bin/python scripts/main.py --help
venv/bin/python scripts/main.py step10 --help
venv/bin/python scripts/main.py concept-shift --help

# Output/provenance bulma
find outputs -type f \( -name '*manifest*.json' -o -name '*preregistration*.json' -o -name '*final_report*.md' \) -print
find outputs/diagnostics outputs/robustness -type f -maxdepth 6 -print

# Bir manifest veya protected file hash'i
sha256sum path/to/file
sha256sum -c path/to/checksums.txt   # Böyle bir canonical checksum listesi gerçekten varsa
```

Hash verification manifest'teki exact relative path ve digest'e karşı yapılmalıdır; rastgele yeni bir checksum üretmek yalnız o anın fingerprint'idir. `--force` scratch/disposable versioned output'u bilinçli yenilerken güvenli olabilir. Frozen Step8/9/10, formal robustness veya Step9G üzerinde, eski manifest/analysis ID'yi koruyarak kullanılması tehlikelidir. Önce backup değil, **new namespace + new preregistration** düşünülmelidir.

# 23. Öğrenme planı

## Oturum 1 — Bilimsel tasarım ve bölgeler (90 dakika)

**Oku:** `core/regions.py`, `core/config.py`, `README.md`, `outputs/step6/labels/burned_landcover_gate.md`.

**Güvenli komutlar:** `venv/bin/python scripts/main.py --help`; `venv/bin/python scripts/main.py experiment --experiment manavgat_2021 --from-stage predictors --to-stage step8 --predictor-mode local-only --dry-run`.

**Kavramlar:** predictor/label separation, region roles, MCD64A1 target, 500 m cell, Kozan control.

**Beş kontrol sorusu:** (1) Manavgat predictor neden 27 Temmuz'da biter? (2) Kozan neden ana wildfire evidence değildir? (3) Baseline period ne işe yarar? (4) FIRMS neden target değildir? (5) Valencia neden executable experiment sayılmaz?

**Egzersiz:** Registry'deki üç enabled region'ın date/role tablosunu kağıt üzerinde yeniden kur; hiçbir dosyayı değiştirme.

## Oturum 2 — Feature generation ve Step8A (120 dakika)

**Oku:** `src/step7c_train_downscaling_model.py`, `src/step7e_fuse_landsat_downscaled_lst.py`, `src/step8a_prepare_500m_modeling_dataset.py`, üç `step8a_dataset_stats.json`.

**Komutlar:** `rg -n "BASELINE_FEATURES|THERMAL_FEATURES|BurnDate|cell_id" src core`; Step8A stats için `python -m json.tool` read-only.

**Kavramlar:** aggregation, validity, label honesty, downscaled vs fused LST, populations.

**Sorular:** (1) 500/30 neden 17 cells olur? (2) Label neden validity maskesi değildir? (3) Fused LST blend midir? (4) Downscaler'da hangi target-derived fields yasak? (5) all-valid ve TSG paydayı nasıl değiştirir?

**Egzersiz:** Tek bir `cell_id` satırının feature/label/provenance alanlarını read-only inceleyip lineage notu yaz.

## Oturum 3 — Step8 modeling ve uncertainty (120 dakika)

**Oku:** `src/step8b_train_baseline_vs_thermal_model.py`, Step8C/D/E modules, Manavgat/Bejís Step8 reports.

**Komutlar:** `venv/bin/python -m pytest -q tests/test_pipeline_orchestrator.py`; metric JSON'larını read-only pretty-print.

**Kavramlar:** fold-local preprocessing, group CV, OOF, ROC/PR/Brier, paired delta, block bootstrap.

**Sorular:** (1) Baseline ve thermal neden aynı folds? (2) Median ne zaman fit edilir? (3) PR-AUC neden önemli? (4) Delta ile absolute AUC farkı? (5) CI neden “significance” diye adlandırılmaz?

**Egzersiz:** İki region için thermal-minus-baseline üç metric'i elle bir tabloya aktar ve safe claim yaz.

## Oturum 4 — Step9 transfer ve Step9E (105 dakika)

**Oku:** Step9A–E source modules; pair final reports; Step9E tables.

**Komutlar:** `venv/bin/python scripts/main.py transfer --help`; `venv/bin/python scripts/main.py transfer --source manavgat_2021 --target bejis_2022 --reverse --dry-run`.

**Kavramlar:** source-only fit, source threshold, target firewall, anti-predictive AUC, distribution vs relationship shift.

**Sorular:** (1) Target labels ilk nerede açılır? (2) AUC 0.326 ne söyler? (3) Brier neden iyileşebilir? (4) Step9E neden confirmatory model değil? (5) En yüksek shift features hangileri?

**Egzersiz:** Step9D machine label'ı ve üç metric'i okuyup overclaim içermeyen iki cümle yaz.

## Oturum 5 — Step9F ve Step10 adaptation (120 dakika)

**Oku:** Step9F variant registry/report, Step10 preregistration, transform/evaluation modules ve final report.

**Komutlar:** `venv/bin/python -m pytest -q tests/test_step9f.py tests/test_step10.py`; `venv/bin/python scripts/main.py step10 --help`.

**Kavramlar:** inductive vs unlabeled-target adaptive, z-score information, CORAL, recovery vs remaining gap, asymmetry.

**Sorular:** (1) Target mean/std neden label leakage değildir? (2) Landcover neden CORAL dışında? (3) Hangi CORAL yönü chance üstü? (4) Step9F neden final representation seçmedi? (5) Recovery neden success transfer değildir?

**Egzersiz:** Her direction için raw→CORAL improvement ve within gap'i tek şemada göster.

## Oturum 6 — Large-block robustness ve Step9G (120 dakika)

**Oku:** İlgili formal reports/manifests; large-block ve Step9G test modules; integration-v2 report.

**Komutlar:** `venv/bin/python scripts/main.py large-block-robustness --dry-run`; `venv/bin/python scripts/main.py concept-shift --dry-run`.

**Kavramlar:** exact equivalence, runtime overrides, protected hashes, raw univariate AUC, point vs supported reversal.

**Sorular:** (1) Config neden 2 kaldı? (2) Manavgat 10 km PR alt sınırı ne? (3) Large blocks neyi çözmez? (4) Neden AUC invert edilmez? (5) Hangi feature supported reversal?

**Egzersiz:** Canonical integration-v2 ve eski correction report yollarını ayıran bir source card hazırla.

## Oturum 7 — Architecture, tests, outputs ve gelecek (90 dakika)

**Oku:** `scripts/main.py`, `core/pipeline_orchestrator.py`, `tests/`, requirements files, bu belgenin 17–22. bölümleri.

**Komutlar:** `git status --short --branch`; `git diff --check`; `venv/bin/python -m pytest -q`; manifest/final-report `find` komutları.

**Kavramlar:** source-of-truth order, canonicality, test boundaries, safe change workflow, advisor decision gate.

**Sorular:** (1) Bir output'u canonical yapan nedir? (2) Passing tests neyi kanıtlamaz? (3) `--force` ne zaman tehlikeli? (4) En acil docs drift hangisi? (5) Yeni experiment öncesi hangi karar eksik?

**Egzersiz:** Danışmana sunulacak beş claim'i kanıt yolu ve yasak wording ile prova et.
# 24. Kendimi sınamam için sorular

## Başlangıç — 20 soru

1. Projenin binary target'ı nedir?
2. Native modeling unit yaklaşık kaç metredir?
3. Predictor window ile label window neden ayrıdır?
4. Üç enabled region ve rolleri nelerdir?
5. Kozan neden anchor wildfire değildir?
6. MCD64A1'in hangi alanı target üretir?
7. FIRMS neden primary target değildir?
8. Baseline modelde hangi dört feature vardır?
9. Thermal model kaç ek feature kullanır?
10. `fused_lst_mean` nasıl üretilir?
11. `all_valid` ne demektir?
12. TSG açılımı/amacı nedir?
13. OOF prediction ne demektir?
14. Neden random-row split yasaktır?
15. ROC-AUC neyi ölçer?
16. PR-AUC neden bu projede önemlidir?
17. Brier score'da daha düşük değer ne anlama gelir?
18. Step9'un sorusu Step8'den nasıl farklıdır?
19. Hangi output aileleri robustness ve diagnostics taşır?
20. Projenin current next step'i nedir?

## Orta — 20 soru

1. 30 m data yaklaşık 500 m'ye nasıl aggregate edilir?
2. Burned label neden predictor validity'yi belirlemez?
3. Numeric ve categorical preprocessing nasıl ayrılır?
4. Baseline ve thermal neden aynı folds'u kullanır?
5. Block bootstrap cell bootstrap'tan neden daha uygundur?
6. Absolute performance ile thermal-minus-baseline delta farkı nedir?
7. Formal robustness neden `all_valid`, transfer neden TSG kullanır?
8. Config block size neden 2 kalmıştır?
9. 2-cell equivalence gate neyi korur?
10. Step9 threshold'u nereden seçer?
11. Brier improvement neden successful transfer kanıtı değildir?
12. AUC <0.5 neden bilimsel olarak bilgilendiricidir?
13. Step9E ve Step9G arasındaki fark nedir?
14. Step9F'te strict ve adaptive regime nasıl ayrılır?
15. Target-region mean/std neden Step10'da izinlidir?
16. CORAL hangi columns'a uygulanmaz?
17. `recovered_covariate_shift` ve `remaining_gap` neyi karşılaştırır?
18. Hangi transfer yönünde CORAL chance üstü bootstrap support alır?
19. Point reversal ile bootstrap-supported reversal farkı nedir?
20. Hangi Step9G integration artifact canonical değildir?

## İleri — 20 soru

1. MCD64A1 temporal/spatial uncertainty estimand'ı nasıl etkileyebilir?
2. `StratifiedGroupKFold` neden yine de bütün spatial dependence sorununu çözmez?
3. Fold-local imputation yapılmazsa hangi leakage oluşur?
4. Target label ile probability inversion seçmek neden post-hoc leakage/selection olur?
5. Manavgat all-valid unburned counts arasındaki 23,354/23,291 farkı nedir?
6. Step7 downscaler'a `lst_anomaly` eklemek neden target-derived risk taşır?
7. Observed-first fusion'ın weighted blend'den bilimsel farkı nedir?
8. Source OOF threshold target'ta neden calibration garantisi vermez?
9. ROC ve Brier neden zıt transfer mesajı verebilir?
10. Step9D `partial_transfer_supported` etiketi nasıl yanlış yorumlanabilir?
11. Step9F source performance–reversal trade-off'u neden bağımsız validation gerektirir?
12. Region-wise z-score hangi covariate shift türlerini düzeltemez?
13. CORAL sonrası remaining gap relationship shift ile neden yalnız “uyumludur”?
14. Adaptation recovery neden yönler arasında simetrik olmak zorunda değildir?
15. Step9G feature-specific complete cases karşılaştırmayı nasıl sınırlayabilir?
16. `elevation_mean` reversal causal mechanism midir?
17. Formal large-block CI'ları neden “autocorrelation eliminated” demeye yetmez?
18. Analysis ID, manifest ve protected hash farklı hangi güvenceyi verir?
19. Passing unit tests Earth Engine/data provenance hakkında neyi kanıtlamaz?
20. Üçüncü region deneyi hangi preregistered kararlar olmadan başlatılmamalı?

<details>
<summary><strong>Yanıt anahtarı — önce soruları cevaplayın</strong></summary>

### Başlangıç yanıtları

1. Label window içinde MCD64A1 `BurnDate` ile hücrenin yanmış/yanmamış olması.
2. Nominal 500 m; 30 m grid'de 17 pixels yaklaşık 510 m.
3. Yangın-sonrası sinyalin predictors'a sızmasını önlemek için.
4. Kozan negative control, Manavgat anchor wildfire, Bejís Mediterranean transfer wildfire.
5. Burned pixels %98.34 cropland; gate `cropland_dominated_control`.
6. `BurnDate` ve label-window DOY aralığı.
7. Hedef burned-area extent/timing'dir; hotspot detections farklı sampling/meaning taşır.
8. NDVI, elevation, slope, dominant land cover.
9. Altı.
10. Observed Landsat varsa onu, yoksa downscaled LST'yi seçer.
11. Predictor/land-cover validity koşullarını geçen tüm cells.
12. `burnable_tree_shrub_grass`; doğal/burnable vegetation transfer population.
13. Bir cell'in, o cell'in bulunmadığı train folds ile skorlanması.
14. Yakın cells bağımlıdır; leakage ve iyimser evaluation doğurur.
15. Positive'ların negatives üstünde sıralanma olasılığı/ranking.
16. Burned class az olduğu için positive retrieval'ı prevalence bağlamında gösterir.
17. Probability predictions label'lara daha yakın/daha iyi calibrated error verir.
18. Step8 within-region OOF katkı; Step9 source→başka target portability.
19. `outputs/robustness/` ve `outputs/diagnostics/`.
20. Advisor review; yeni scientific experiment için yeni guidance/preregistration beklemek.

### Orta yanıtları

1. Continuous stats ve land-cover mode/fractions ile 17×17 nominal windows'a.
2. Non-burned gerçek class'tır; label availability feature eligibility'yi etkilerse selection leakage doğar.
3. Numeric median; categorical most-frequent + one-hot, train-fold fit.
4. Paired ve adil delta için.
5. Yakın cells'i bağımsız saymamak ve spatial unit'i birlikte resample etmek için.
6. İlki model düzeyi, ikincisi thermal information'ın baseline üstü incremental katkısı.
7. İlk analiz original landscape estimand'ını; transfer ortak natural-vegetation estimand'ını korur.
8. Canonical 2-cell reference ve exact reproduction korunur; 10/20 runtime sensitivity'dir.
9. Yeni runner'ın labels/folds/predictions/metrics'i değiştirmediğini.
10. Yalnız source OOF predictions'tan.
11. Prevalence/probability scale düzelebilirken ranking <0.5 kalabilir.
12. Target'ta ordering reversal/relationship instability gösterebilir.
13. Step9E post-hoc geniş audit; Step9G preregistered raw univariate block-bootstrap quantification.
14. Strict yalnız source X/y; adaptive ayrıca unlabeled target X statistics kullanır.
15. Unsupervised domain adaptation'dır; labels kullanılmaz.
16. `landcover_dominant` ve diğer categorical encoded fields.
17. Adapted−raw recovery; within−adapted çözülmemiş gap.
18. Bejís→Manavgat CORAL.
19. Point estimates 0.5'in ters tarafları; supported durumda uncertainty intervals da ters yönü korur.
20. `step9g_integration_correction_final_report.*` eski seti.

### İleri yanıtları

1. Coarse/mixed pixels ve burn-date uncertainty misclassification yaratıp feature–label association'ı zayıflatabilir/değiştirebilir.
2. Block size/model residual scale yanlışsa groups arasında dependence kalabilir; yalnız predefined sensitivity'dir.
3. Test-fold distribution train preprocessing parameterına girer.
4. Target outcomes'a göre yeni decision rule seçilir; source-only sözleşmesi bozulur.
5. İlk sayı total grid'deki invalid unburned cells'i de içerir; ikincisi valid modeling rows'dur.
6. Target Landsat LST'den türemiş bilgi target'ı predictors içinde yeniden kullanabilir.
7. Fused ürün gözlemi değiştirmez; yalnız gaps'i model ile doldurur, optimal blend iddiası yoktur.
8. Prevalence/score distribution target'ta değişebilir.
9. Ranking tersken probabilities prevalence'a yakınlaşarak squared error'ı azaltabilir.
10. Brier-only partial support, discrimination/portability success sanılabilir.
11. Varyantlar target sonuçları görüldükten sonra seçildi; selection bias vardır.
12. Nonlinear, conditional veya label relationship shift; higher-order/domain mechanisms.
13. Marginal/covariance alignment başarısızlığı alternatifleri dışlamaz; iki region mekanizma kanıtlamaz.
14. Domain moments ve `P(Y|X)` değişimleri directional'dır; CORAL mapping source→target'tır.
15. Her feature farklı sample/population altkümesinde ölçülebilir; AUC'lar birebir aynı rows olmayabilir.
16. Hayır; proxy/confounding/terrain–fire context farkı olabilir.
17. Residual dependence ölçülmedi ve sıfırlanmadı; yalnız 5/10 km grouping'de contrast sürdü.
18. ID sözleşme kimliği; manifest lineage/parameters; hash byte-level immutability.
19. Remote collection versions, credentials, export completeness ve physical validity'yi kanıtlamaz.
20. Region role, dates, population, frozen representation, primary metric, blocks/bootstrap, success/failure ve no-go selection rules.

</details>

# 25. Sözlü savunma simülasyonu

1. **Bu çalışma tam olarak neyi tahmin ediyor?** 500 m cells'in öncü predictor window koşullarından, daha sonraki label window'da MCD64A1 ile burned olma sıralamasını. Ignition time/location, spread trajectory veya operational alarm üretmiyor.

2. **Neden Random Forest?** Mixed nonlinearities ve numeric+categorical tabular features için sağlam, az tuning gerektiren ortak comparator sağladı; aynı fixed configuration baseline/thermal ve bölgeler arasında kullanıldı. RF'nin en iyi model olduğu veya sonuçların model-independent olduğu iddia edilmiyor.

3. **Neden MCD64A1?** Sistematik BurnDate field ve region-wide gridded label sağlar; FIRMS detections'ın sampling/coverage anlamından farklıdır. Coarse resolution/mixed-pixel ve date uncertainty limitation olarak kalır.

4. **Neden yaklaşık 500 m?** MCD64A1/MODIS ölçeğiyle uyum ve 30 m predictors'ı ortak cell'de summarize etmek için. 17×30 m yaklaşık 510 m'dir; exact universal support değildir.

5. **Neden yalnız iki wildfire region?** Manavgat anchor ve Bejís bağımsız Mediterranean transfer case olarak repository'de complete/frozen data taşır. İki vaka mechanism/generalization kanıtı için azdır; bu yüzden üçüncü region iddiası yok ve yeni advisor guidance bekleniyor.

6. **Prototype ve pipeline z-score sonuçları neden farklı olabilir?** Final pipeline fixed population, source-only contracts, exact preprocessing, label firewall ve target-block bootstrap uygular. Prototype'ın sampling, missing handling, feature ordering veya evaluation details'i birebir aynı olmayabilir; repository exact prototype output'u saklamadığından spesifik sebebi uydurmuyorum. Reproduced expectation CORAL Bejís→Manavgat above-chance'dır; symmetric recovery değildir.

7. **Concept shift kanıtlandı mı?** Hayır. CORAL sonrası remaining gap, Step9E/G direction instability ve raw reversal birlikte residual relationship/concept shift ile uyumlu. Measurement error, unmodeled covariates, event heterogeneity ve model misspecification da olasıdır.

8. **AUC <0.5'i neden ters çevirmediniz?** Ters direction bilimsel bulgudur. Target labels'a bakıp `1-p` seçmek source-only rule'a target-supervised correction ekler ve portability failure'ı gizler.

9. **Brier iyiyse transfer neden başarısız?** Brier probability error, ROC ranking ölçer. Low-prevalence target'ta daha düşük probabilities Brier'ı iyileştirirken positives yine yanlış sıralanabilir; iki raw ROC <0.5'tir.

10. **Neden formal analizde all-valid, transferda natural vegetation?** Formal large-block ilk Step8B all-valid estimand'ını exact reproduce eder. Step9/10/9G ise iki region'ın karşılaştırılabilir burnable TSG population'ını sabitler. Birini diğerinin yerine koymak soruyu değiştirir; TSG ayrıca sensitivity olarak raporlanır.

11. **Large-block robustness autocorrelation'ı bitirdi mi?** Hayır. 5/10 km predefined groups altında thermal delta sürdü; bu daha kaba leakage sensitivity'sidir. Residual correlation doğrudan estimate edilmedi.

12. **Step10 target bilgisi kullanıyor; leakage değil mi?** Unlabeled target covariates'ın mean/std/covariance'sı unsupervised adaptation information regime'inde izinliydi. Target labels transform ve fitting'den firewall ile ayrıldı, yalnız final evaluation'da açıldı. Bu transductive/adaptive setting, strict source-only değildir ve öyle etiketlenir.

13. **Adaptation başarılı mı?** Kısmi: özellikle CORAL raw AUC'yi iki yönde artırdı; chance üstü bootstrap support yalnız Bejís→Manavgat'ta. Within-region gap büyük kaldığı için general transfer success denemez.

14. **Thermal features yangına neden oluyor diyebilir miyiz?** Hayır. Zaman sırası leakage'i azaltır ama confounding, fuel, weather, ignition ve management etkilerini kontrol eden causal design yoktur. “Association/predictive contribution” denir.

15. **Buna early-warning system denebilir mi?** Hayır. Retrospective iki-event evaluation, operational latency, prospective calibration, alert thresholds/costs, deployment monitoring ve external validation yoktur. En fazla pre-event satellite-condition research pipeline'dır.

# 26. Tek sayfalık hızlı referans

| Başlık | Hızlı cevap |
|---|---|
| Amaç | Pre-label satellite/terrain/land-cover conditions ile ~500 m cell'in sonraki MCD64A1 burned label ilişkisini ölçmek |
| Regions | Kozan 2023 control; Manavgat 2021 anchor; Bejís 2022 transfer wildfire |
| Target | Label-window DOY içinde MCD64A1 `BurnDate` positive; FIRMS primary target değil |
| Baseline | `ndvi_mean`, `elevation_mean`, `slope_mean`, `landcover_dominant` |
| Thermal additions | anomaly, current LST/TVDI, TVDI difference, downscaled/fused LST |
| Populations | Step8 formal robustness: `all_valid`; Step9/10/9G: TSG |
| Evaluation | Spatial-group OOF; whole-block paired bootstrap; no random-row split |
| Within result | Manavgat delta ROC +0.0589, PR +0.0996; Bejís +0.0482, +0.1777; Brier iyileşir |
| Large blocks | 5/10 km'de dört region-scale delta ROC/PR CI pozitif; autocorrelation eliminated değil |
| Raw transfer | Thermal ROC 0.3258 Man→Bej, 0.4435 Bej→Man: discrimination fails |
| Step10 | CORAL ROC 0.5105 ve 0.5553; chance support yalnız Bej→Man; large remaining gaps |
| Step9G | `elevation_mean` supported reversal; current/downscaled/fused LST ve TVDI difference uncertain point reversal |
| Güvenli claim | Yerel thermal katkı + coarse-block robustness; raw portability failure; partial/asymmetric recovery; residual relationship shift ile uyum |
| Yasak claim | Causality, spatial dependence eliminated, general successful transfer, operational early warning, concept shift tek açıklama |
| Canonical roots | `outputs/experiments/`, `outputs/cross_region/`, `outputs/robustness/step8_large_block_primary_all_valid/`, canonical Step9G numeric + integration-v2 |

```bash
venv/bin/python scripts/main.py --help
venv/bin/python scripts/main.py experiment --experiment manavgat_2021 --from-stage predictors --to-stage step8 --predictor-mode local-only --dry-run
venv/bin/python scripts/main.py transfer --source manavgat_2021 --target bejis_2022 --reverse --dry-run
venv/bin/python scripts/main.py large-block-robustness --dry-run
venv/bin/python scripts/main.py concept-shift --dry-run
venv/bin/python -m pytest -q
git status --short --branch && git diff --check
```

**Current next step:** Bu evidence/claim-boundary zincirini advisor ile review et; yeni scientific experiment başlatma. İlk study session: `core/regions.py`, `core/config.py`, Kozan land-cover gate ve CLI dry-run ile scientific design/region roles.
