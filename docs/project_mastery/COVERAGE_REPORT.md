# COVERAGE_REPORT.md — Kapsam Raporu

- **Repository commit:** `47452308c37a3e0e4915a9ab88be8bc8a2d5bf80`
- **Üretim zamanı:** 2026-07-23 13:37 UTC
- **İncelenen Python modülü (venv/ ve old_codes/ hariç):** 130 (izlenen + yeni/untracked; parse hatası: 0)
- **Ana el kitabı:** `docs/project_mastery/PROJECT_MASTERY_GUIDE.md` → `docs/PROJECT_MASTERY_GUIDE.pdf`

Bu rapor, el kitabının kapsam iddiasını destekler: her Python modülü, her CLI komutu, her deney, her feature, her çıktı namespace'i ve her bilimsel aşama incelenmiş ve bir handbook bölümüne eşlenmiştir. Hiçbir dosya sessizce atlanmamıştır.

## 1. Tüm Python modülleri ve kapsam durumu

| Modül | LOC | Durum | Kapsandığı bölüm |
|---|---|---|---|
| `core/config.py` | 698 | canonical | Bölüm 19 |
| `core/cross_region_experiment.py` | 328 | canonical | Bölüm 19 |
| `core/drive_downloader.py` | 372 | **SİLİNDİ (2026-08-10 cleanup)** — zero-caller orphan; legacy Drive akışı `src/step4b_download_drive_export.py` üzerinden sürüyor | Bölüm 5.7, 19 (tarihsel) |
| `core/experiment_context.py` | 246 | canonical | Bölüm 19 |
| `core/gee_utils.py` | 9 | canonical | Bölüm 19 |
| `core/io_utils.py` | 35 | canonical | Bölüm 19 |
| `core/paths.py` | 13 | canonical | Bölüm 19 |
| `core/pipeline_orchestrator.py` | 861 | canonical | Bölüm 19 |
| `core/regions.py` | 487 | canonical | Bölüm 19 |
| `core/seam_audit_config.py` | 221 | canonical (QA) | Bölüm 5.7, 9.5, 19 |
| `core/seam_audit_v2_config.py` | 550 | canonical (QA) | Bölüm 5.7, 9.5, 19 |
| `core/seam_localization_config.py` | 51 | canonical (QA) | Bölüm 5.7, 9.5, 19 |
| `core/source_scene_provenance_config.py` | 71 | canonical (QA) | Bölüm 5.7, 9.5, 19 |
| `core/step10_shared.py` | 315 | canonical | Bölüm 19 |
| `core/utils/__init__.py` | 10 | canonical | Bölüm 19 |
| `core/utils/geotiff_validation.py` | 560 | canonical | Bölüm 19 |
| `core/utils/tiling.py` | 333 | canonical | Bölüm 19 |
| `core/validation_burned_area.py` | 323 | canonical | Bölüm 19 |
| `scripts/check_experiment_registry.py` | 228 | canonical | Bölüm 19 |
| `scripts/export_mcd64a1_raw_burndate.py` | 83 | canonical | Bölüm 19 |
| `scripts/main.py` | 930 | canonical | Bölüm 19 |
| `scripts/prepare_dem_for_experiment.py` | 417 | canonical | Bölüm 19 |
| `scripts/prepare_modis_for_step7.py` | 542 | canonical | Bölüm 19 |
| `scripts/preview_experiment_aoi.py` | 143 | canonical | Bölüm 19 |
| `scripts/run_burned_pattern_audit.py` | 57 | canonical | Bölüm 19 |
| `scripts/run_cross_region_shift_audit.py` | 187 | canonical | Bölüm 19 |
| `scripts/run_cross_region_transfer.py` | 226 | canonical | Bölüm 19 |
| `scripts/run_domain_classifier_audit.py` | 59 | canonical | Bölüm 19 |
| `scripts/run_exploratory_transfer_features.py` | 181 | canonical | Bölüm 19 |
| `scripts/run_label_gate_only.py` | 709 | canonical | Bölüm 19 |
| `scripts/run_predictors_only.py` | 1144 | canonical | Bölüm 19 |
| `scripts/run_prefire_experiment.py` | 134 | **SİLİNDİ (2026-08-10 cleanup)** — zero-caller orphan | Bölüm 5.7, 19 (tarihsel) |
| `scripts/run_seam_audit.py` | 453 | canonical (QA) | Bölüm 5.7, 9.5, 19 |
| `scripts/run_seam_audit_v2.py` | 590 | canonical (QA) | Bölüm 5.7, 9.5, 19 |
| `scripts/run_seam_localization.py` | 111 | canonical (QA) | Bölüm 5.7, 9.5, 19 |
| `scripts/run_source_scene_provenance.py` | 97 | canonical (QA) | Bölüm 5.7, 9.5, 19 |
| `scripts/run_step10_self_calibrated_transfer.py` | 224 | canonical | Bölüm 19 |
| `scripts/run_step7_downscaling_only.py` | 597 | canonical | Bölüm 19 |
| `scripts/run_step8_big_block_robustness.py` | 95 | canonical | Bölüm 19 |
| `scripts/run_step8_large_block_robustness.py` | 60 | canonical | Bölüm 19 |
| `scripts/run_step8_large_block_robustness_primary_all_valid.py` | 51 | canonical | Bölüm 19 |
| `scripts/run_step8_modeling.py` | 728 | canonical | Bölüm 19 |
| `scripts/run_step9g_integration_correction_v2.py` | 15 | **SİLİNDİ (2026-08-10 cleanup)** — canonical yol `scripts/main.py concept-shift` | Bölüm 19 (tarihsel) |
| `scripts/run_step9g_univariate_feature_auc_direction_reversal.py` | 15 | **SİLİNDİ (2026-08-10 cleanup)** — canonical yol `scripts/main.py concept-shift` | Bölüm 19 (tarihsel) |
| `scripts/standalone_step5-6.py` | 67 | **SİLİNDİ (2026-08-10 cleanup)** — zero-caller orphan | Bölüm 5.7, 19 (tarihsel) |
| `src/burned_pattern_audit.py` | 1257 | canonical | Bölüm 19 |
| `src/domain_classifier_audit.py` | 820 | canonical | Bölüm 19 |
| `src/multi_aoi_transfer_synthesis/__init__.py` | 27 | canonical | Bölüm 19 |
| `src/multi_aoi_transfer_synthesis/aoi_set.py` | 119 | canonical | Bölüm 19 |
| `src/multi_aoi_transfer_synthesis/build.py` | 599 | canonical | Bölüm 19 |
| `src/multi_aoi_transfer_synthesis/manifest.py` | 68 | canonical | Bölüm 19 |
| `src/multi_aoi_transfer_synthesis/render.py` | 303 | canonical | Bölüm 19 |
| `src/multi_aoi_transfer_synthesis/resolvers.py` | 787 | canonical | Bölüm 19 |
| `src/multi_aoi_transfer_synthesis/schema_adapters.py` | 499 | canonical | Bölüm 19 |
| `src/multi_aoi_transfer_synthesis/status_derivation.py` | 384 | canonical | Bölüm 19 |
| `src/seam_audit.py` | 697 | canonical (QA) | Bölüm 5.7, 9.5, 19 |
| `src/seam_audit_v2.py` | 1014 | canonical (QA) | Bölüm 5.7, 9.5, 19 |
| `src/seam_localization.py` | 1287 | canonical (QA) | Bölüm 5.7, 9.5, 19 |
| `src/source_scene_provenance.py` | 781 | canonical (QA) | Bölüm 5.7, 9.5, 19 |
| `src/step10a_preregistration_and_audit.py` | 366 | canonical | Bölüm 19 |
| `src/step10b_label_blind_adaptation.py` | 240 | canonical | Bölüm 19 |
| `src/step10c_paired_evaluation_bootstrap.py` | 596 | canonical | Bölüm 19 |
| `src/step10d_final_report.py` | 652 | canonical | Bölüm 19 |
| `src/step1_fetch_modis.py` | 153 | legacy | Bölüm 9.2, 19.2 |
| `src/step2_modis_5year_mean.py` | 187 | legacy | Bölüm 9.2, 19.2 |
| `src/step2b_dem.py` | 285 | legacy | Bölüm 9.2, 19.2 |
| `src/step3_landsat_lst.py` | 885 | legacy | Bölüm 9.2, 19.2 |
| `src/step4_export_geotiff.py` | 861 | legacy | Bölüm 9.2, 19.2 |
| `src/step4b_download_drive_export.py` | 1078 | legacy | Bölüm 9.2, 19.2 |
| `src/step5_preprocess_timeseries.py` | 1163 | canonical | Bölüm 19 |
| `src/step5b_diagnostic_report.py` | 1847 | canonical | Bölüm 19 |
| `src/step5c_tvdi.py` | 886 | canonical | Bölüm 19 |
| `src/step6_validate_fire_relation.py` | 2813 | canonical | Bölüm 19 |
| `src/step6a_prepare_gate_inputs.py` | 339 | canonical | Bölüm 19 |
| `src/step6b_burned_landcover_gate.py` | 1210 | canonical | Bölüm 19 |
| `src/step7a_tiling_infrastructure.py` | 364 | canonical | Bölüm 19 |
| `src/step7b_prepare_downscaling_dataset.py` | 1382 | canonical | Bölüm 19 |
| `src/step7c_train_downscaling_model.py` | 1123 | canonical | Bölüm 19 |
| `src/step7d_predict_downscaled_lst.py` | 1253 | canonical | Bölüm 19 |
| `src/step7e_fuse_landsat_downscaled_lst.py` | 872 | canonical | Bölüm 19 |
| `src/step8_big_block_robustness.py` | 1568 | canonical | Bölüm 19 |
| `src/step8_large_block_robustness.py` | 571 | canonical | Bölüm 19 |
| `src/step8_large_block_robustness_primary_all_valid.py` | 813 | canonical | Bölüm 19 |
| `src/step8a_prepare_500m_modeling_dataset.py` | 2794 | canonical | Bölüm 19 |
| `src/step8b_train_baseline_vs_thermal_model.py` | 1380 | canonical | Bölüm 19 |
| `src/step8c_spatial_block_bootstrap_uncertainty.py` | 899 | canonical | Bölüm 19 |
| `src/step8d_thermal_feature_ablation.py` | 1221 | canonical | Bölüm 19 |
| `src/step8e_final_report.py` | 955 | canonical | Bölüm 19 |
| `src/step9a_audit_cross_region_inputs.py` | 569 | canonical | Bölüm 19 |
| `src/step9b_run_cross_region_transfer.py` | 516 | canonical | Bölüm 19 |
| `src/step9c_cross_region_block_bootstrap.py` | 308 | canonical | Bölüm 19 |
| `src/step9d_build_cross_region_report.py` | 328 | canonical | Bölüm 19 |
| `src/step9e_distribution_shift_audit.py` | 1639 | canonical | Bölüm 19 |
| `src/step9f_exploratory_transfer_feature_experiment.py` | 1227 | canonical | Bölüm 19 |
| `src/step9g_integration_correction_v2.py` | 801 | canonical | Bölüm 19 |
| `src/step9g_multi_aoi_comparison/__init__.py` | 19 | canonical | Bölüm 19 |
| `src/step9g_multi_aoi_comparison/build.py` | 348 | canonical | Bölüm 19 |
| `src/step9g_multi_aoi_comparison/consistency.py` | 87 | canonical | Bölüm 19 |
| `src/step9g_multi_aoi_comparison/discovery.py` | 52 | canonical | Bölüm 19 |
| `src/step9g_multi_aoi_comparison/parse.py` | 189 | canonical | Bölüm 19 |
| `src/step9g_multi_aoi_comparison/render.py` | 186 | canonical | Bölüm 19 |
| `src/step9g_report_revision.py` | 243 | canonical | Bölüm 19 |
| `src/step9g_univariate_feature_auc_direction_reversal.py` | 1075 | canonical | Bölüm 19 |

### Test modülleri (test-only)

| Test modülü | LOC | Kapsandığı bölüm |
|---|---|---|
| `tests/test_burned_pattern_audit.py` | 765 | Bölüm 19.4 |
| `tests/test_domain_classifier_audit.py` | 410 | Bölüm 19.4 |
| `tests/test_export_size_safe_tiling.py` | 818 | Bölüm 19.4 |
| `tests/test_main_cli.py` | 386 | Bölüm 19.4 |
| `tests/test_modis_nodata_qa.py` | 371 | Bölüm 19.4 |
| `tests/test_modis_qc_valid_count.py` | 193 | Bölüm 19.4 |
| `tests/test_mugla_2021_gate.py` | 524 | Bölüm 19.4 |
| `tests/test_multi_aoi_transfer_synthesis.py` | 516 | Bölüm 19.4 |
| `tests/test_pipeline_orchestrator.py` | 216 | Bölüm 19.4 |
| `tests/test_scene_provenance_localization.py` | 164 | Bölüm 19.4 |
| `tests/test_seam_audit.py` | 212 | Bölüm 19.4 |
| `tests/test_seam_audit_v2.py` | 476 | Bölüm 19.4 |
| `tests/test_seam_localization.py` | 392 | Bölüm 19.4 |
| `tests/test_source_scene_provenance.py` | 222 | Bölüm 19.4 |
| `tests/test_step10.py` | 665 | Bölüm 19.4 |
| `tests/test_step7c_split_integrity.py` | 179 | Bölüm 19.4 |
| `tests/test_step8_big_block_robustness.py` | 602 | Bölüm 19.4 |
| `tests/test_step8_large_block_robustness.py` | 321 | Bölüm 19.4 |
| `tests/test_step8_large_block_robustness_primary_all_valid.py` | 529 | Bölüm 19.4 |
| `tests/test_step8a_pre_label_exclusion.py` | 264 | Bölüm 19.4 |
| `tests/test_step8e_report_population_accounting.py` (eski ad: `test_step8e_report_fix.py`) | 310 | Bölüm 19.4 |
| `tests/test_step9e_report_wording_and_provenance.py` (eski ad: `test_step9e_report_fix.py`) | 509 | Bölüm 19.4 |
| `tests/test_step9f.py` | 390 | Bölüm 19.4 |
| `tests/test_step9g_integration_correction_v2.py` | 303 | Bölüm 19.4 |
| `tests/test_step9g_multi_aoi_comparison.py` | 317 | Bölüm 19.4 |
| `tests/test_step9g_report_revision.py` | 207 | Bölüm 19.4 |
| `tests/test_step9g_univariate_feature_auc_direction_reversal.py` | 344 | Bölüm 19.4 |

**Toplam:** 103 bilimsel/altyapı modülü + 27 test modülü = 130 modül. Tümü Bölüm 19'da (developer reference) tek tek listelenmiştir.

## 2. CLI komutları ve alias'lar (15)

| Komut | Alias/Not | İşlev | Bölüm |
|---|---|---|---|
| `experiment` | — | gate→...→step8 zinciri | Bölüm 6.2 |
| `transfer` | — | Step9A-D | Bölüm 6.3 |
| `shift-audit` | — | Step9E | Bölüm 6.4 |
| `transfer-explore` | — | Step9F | Bölüm 6.5 |
| `self-cal-transfer` | step10 ile aynı analiz | Step10 | Bölüm 6.6 |
| `step10` | self-cal-transfer alias | Step10 | Bölüm 6.6 |
| `step8-robustness` | — | frozen 10/20 (manavgat+bejis, burnable) | Bölüm 6.7 |
| `large-block-robustness` | — | formal all_valid (gated) | Bölüm 6.8 |
| `step8-big-block-robustness` | — | tek deney big-block | Bölüm 6.9 |
| `concept-shift` | — | Step9G (+integration-only/report-revision) | Bölüm 6.10 |
| `concept-shift-compare` | — | çoklu-AOI Step9G (report-only) | Bölüm 6.11 |
| `transfer-synthesis` | — | çoklu-AOI sentez (report-only) | Bölüm 6.12 |
| `burned-pattern-audit` | — | betimsel geometri | Bölüm 6.13 |
| `domain-classifier-audit` | — | covariate separability | Bölüm 6.14 |
| `legacy` | — | Kozan Drive Step1-8E | Bölüm 6.15 |

## 3. Yapılandırılmış deneyler (registry: 5)

| experiment_id | enabled | rol | Bölüm |
|---|---|---|---|
| `kozan_2023` | True | (bkz. Bölüm 4.2) | Bölüm 4 |
| `manavgat_2021` | True | (bkz. Bölüm 4.2) | Bölüm 4 |
| `bejis_2022` | True | (bkz. Bölüm 4.2) | Bölüm 4 |
| `mugla_2021` | True | (bkz. Bölüm 4.2) | Bölüm 4 |
| `evia_2021` | True | (bkz. Bölüm 4.2) | Bölüm 4 |

Not: README'nin bahsettiği `zamora_2022` registry'de bir DENEY olarak YOKTUR (yalnız `valencia_2022_aoi` bir region_key'dir). Bkz. Bölüm 4.5 discrepancy tablosu.

## 4. Belgelenmiş feature'lar (10)

- `ndvi_mean` — Bölüm 8
- `elevation_mean` — Bölüm 8
- `slope_mean` — Bölüm 8
- `landcover_dominant` — Bölüm 8
- `lst_anomaly_mean` — Bölüm 8
- `current_lst_mean` — Bölüm 8
- `current_tvdi_mean` — Bölüm 8
- `tvdi_difference_mean` — Bölüm 8
- `downscaled_lst_mean` — Bölüm 8
- `fused_lst_mean` — Bölüm 8

## 5. Belgelenmiş çıktı namespace'leri

- `outputs/experiments/<id>/` — Bölüm 16
- `outputs/kozan-legacy/` — Bölüm 16
- `outputs/cross_region/<src>__<tgt>/` — Bölüm 16
- `outputs/robustness/step8_large_block/` — Bölüm 16
- `outputs/robustness/step8_large_block_primary_all_valid/` — Bölüm 16
- `outputs/diagnostics/burned_pattern_audit/` — Bölüm 16
- `outputs/diagnostics/domain_classifier_audit/` — Bölüm 16
- `outputs/diagnostics/multi_aoi_transfer_synthesis/` — Bölüm 16
- `outputs/diagnostics/step9g_univariate_feature_auc_direction_reversal[_integration_v2]/` — Bölüm 16

## 6. Belgelenmiş bilimsel aşamalar

- Step1-4B (legacy) — Bölüm 9-13
- Step5 anomaly — Bölüm 9-13
- Step5C TVDI — Bölüm 9-13
- Step6/6A/6B gate+label — Bölüm 9-13
- Step7A-E downscale/fuse — Bölüm 9-13
- Step8A-E within-region — Bölüm 9-13
- Step8 large/big-block robustness — Bölüm 9-13
- Step9A-D transfer — Bölüm 9-13
- Step9E shift-audit — Bölüm 9-13
- Step9F exploratory — Bölüm 9-13
- Step9G concept-shift — Bölüm 9-13
- Step10A-D self-cal — Bölüm 9-13
- domain-classifier — Bölüm 9-13
- burned-pattern — Bölüm 9-13
- multi-AOI synthesis — Bölüm 9-13

## 7. Bilerek hariç tutulanlar (gerekçeli)

| Yol | Gerekçe |
|---|---|
| `venv/` | Sanal ortam; proje kaynağı değil |
| `old_codes/` | Tarihsel; aktif pipeline'a dahil değil (git: 'stop tracking old_codes') |
| `data/`, `outputs/`, `logs/` (ikili/raster) | `.gitignore`'da; içerik Bölüm 13/16'da özetlenir, tek tek dosya değil |
| `docs/PROJECT_REONBOARDING.md`, diğer `docs/*.md` | Önceki dokümanlar; bu handbook onları DEĞİŞTİRMEZ, yalnız Bölüm 4.5/21.4'te ilişkilendirir |
| `.pytest_cache/`, `__pycache__/` | Üretilen cache |

## 8. Çözülemeyen belirsizlikler / discrepancy'ler

- **README ↔ registry ↔ outputs** (Bölüm 4.5): AOI kapsamı, CLI listesi, Kozan yolları, zamora placeholder. Kanonik: registry+outputs. README bilimsel çerçeve için hâlâ güvenilir.
- **Evia tamamlanma:** son commit 'Evia AOI at %90'; Step10/bazı diagnostic'ler tüm yönlerde yok (Bölüm 13.9).
- **Erken git commit mesajları** ('.', 'geri alma') gerekçesizdir; tam tarihsel motivasyon belirsiz (Bölüm 20.3).

## 9. Olası stale/orphan dosyalar (kanıt vs. tahmin)

- `scripts/run_prefire_experiment.py`, `scripts/standalone_step5-6.py`: canonical CLI'dan çağrılmaz (legacy yardımcı). **2026-08-10 cleanup'ında silindi** — zero-caller oldukları doğrulandı.
- README'de belgelenen `outputs/step5/` vb. legacy yollar: gerçekte `outputs/kozan-legacy/` kullanılıyor (kanıt).
- `core/seam_audit_config.py` (v1) vs `seam_audit_v2_config.py`: v2 aktif; v1 audit kaydı olarak korunur.

## 10. Çözülemeyen kapsam boşlukları

- Raster/GeoTIFF içerik doğrulaması (piksel-seviyesi) yapılmadı; bu salt-okunur doküman görevinin kapsamı dışıdır ve bilimsel çıktıları değiştirmeme kısıtına tabidir. Şema/manifest seviyesinde inceleme yapıldı (Bölüm 16).
- Runtime çağrı-grafiği (dinamik) yerine statik import grafiği kullanıldı; 'kesin ölü kod' iddiaları tahmin olarak işaretlendi.
