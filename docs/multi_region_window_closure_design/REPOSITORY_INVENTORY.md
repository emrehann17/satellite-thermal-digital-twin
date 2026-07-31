# Repository Inventory — read-only audit

Commit audited: `483027a38148319099b20b97f2307d5457c51260` (branch `main`).
Every path below was opened or listed. Anything not found is marked `NOT FOUND`.

---

## 1. Headline finding

**The existing window-closure implementation is already AOI-generic.**

`scripts/run_window_closure_sensitivity.py` takes `--experiment <experiment_id>` and resolves
every AOI-specific value through `core/regions.py` and `core/experiment_context.py`. A
repository-wide search found **no AOI name hard-coded in any control-flow branch** of
`src/window_closure_sensitivity.py` — Manavgat appears only inside frozen prose (limitation
strings) and in the already-written output namespace.

Consequence for this design: the multi-region extension is predominantly an
**orchestration + synthesis + validation** layer over an unchanged scientific core. The
per-AOI scientific engine is reused *as is*.

---

## 2. Core implementation

| # | Path | Symbol | Responsibility | Reuse decision | Change later? | Risk |
|---|---|---|---|---|---|---|
| 1 | `src/window_closure_sensitivity.py` (10,697 lines) | module | Entire per-AOI window-closure analysis: window arithmetic, censoring, export planning, downstream orchestration, cohort, folds, fits, bootstrap, compare, wording guards | **Reuse verbatim** | **No** | Low — changing it would invalidate the Manavgat PASS |
| 1a | ″ | `canonical_window` (287) | Reads predictor/label dates from experiment ctx; asserts `predictor_end < label_start` | Reuse | No | Low |
| 1b | ″ | `build_window_variants` (311) | Both ends shift by `shift`; duration and label invariance asserted | Reuse | No | Low |
| 1c | ″ | `common_prelabel_interval` (357) | `[min(variant starts), label_start−1]`, flag-independent | Reuse | No | Low |
| 1d | ″ | `scientific_configuration` (1044) / `compute_analysis_id` (1127) | Frozen config → deterministic `analysis_id` | Reuse per AOI; **new** outer `analysis_id` for the multi-region set | Additive | Medium — see §6 |
| 1e | ″ | `frozen_input_inventory` (955) | Hashes the 6 required frozen input roles | Reuse | No | Low |
| 1f | ″ | `landsat_export_plan` (629), `modis_export_plan` (778), `static_shared_plan` (824), `prelabel_export_plan` (836) | Per-variant export planning | Reuse | No | Low |
| 1g | ″ | `predictor_artifact_jobs` (2978), `assert_predictor_job_set` (3071), `assert_jobs_inside_variant_namespace` (3128) | Job set + namespace containment | Reuse | No | Low |
| 1h | ″ | `production_predictor_engine` (3646) | The **only** function that imports `ee` | Reuse | No | Low |
| 1i | ″ | `production_local_downstream_engine` (5406) | Step5 → 5C → 7A–7E → 8A per variant | Reuse | No | Low |
| 1j | ″ | `build_model_common_cohort` (7734) | Structure-A exact intersection + invariance gates | Reuse | No | Low |
| 1k | ″ | `build_shared_spatial_folds` (7895) | One shared fold assignment, hashed | Reuse | No | Low |
| 1l | ″ | `fit_variant_models` (7993) | Calls `step8b.train_population`; asserts shared folds | Reuse | No | Low |
| 1m | ″ | `multi_variant_block_bootstrap` (1316) | Paired block bootstrap, no refit | Reuse | No | Low |
| 1n | ″ | `run_analysis` (10225) | Stage driver, dry-run/resume/force | Reuse per AOI | No | Low |
| 1o | ″ | `assert_compare_wording` (9020), `FORBIDDEN_COMPARE_PHRASES` (8863), `assert_no_foreign_factor_wording` (546) | Report wording QA | **Reuse + extend** (see `W6`) | **Yes — additive only** | Medium |

## 3. CLI

| # | Path | Responsibility | Reuse | Change? |
|---|---|---|---|---|
| 2 | `scripts/run_window_closure_sensitivity.py` (80 lines) | Thin per-AOI dispatcher: `--experiment --shifts --from-stage --to-stage --dry-run --force --resume` | **Reuse unchanged** | No |
| — | Multi-AOI CLI | — | **NOT FOUND** — must be added | New: `scripts/run_multi_region_window_closure.py` |

## 4. Config and registry

| # | Path | Content | Reuse | Change? |
|---|---|---|---|---|
| 3 | `core/regions.py:280-465` | `EXPERIMENTS` registry — the **single authoritative source** of every predictor/label date, baseline years, `output_namespace`, `exclude_pre_label_burns`, `pre_label_burn_window` | Reuse read-only | **No** |
| 4 | `core/config.py:554-557` | `STEP8B_RANDOM_SEED=42`, `STEP8B_N_SPLITS=5`, `STEP8B_SPATIAL_BLOCK_SIZE_CELLS=2`, `STEP8B_MIN_POSITIVES_PER_POPULATION=30` | Reuse | No |
| 5 | `core/config.py:574-577` | `STEP8C_N_BOOTSTRAP=1000`, `STEP8C_RANDOM_SEED=42`, `STEP8C_CI_LOWER=2.5`, `STEP8C_CI_UPPER=97.5` | Reuse | No |
| 6 | `core/config.py:107-108` | `SUMMER_MONTH_START=6`, `SUMMER_MONTH_END=9` — **the MODIS season policy** | Reuse | **No** |
| 7 | `core/experiment_context.py` | `build_experiment_context`, `_current_period_days` (line 38) | Reuse | No |
| 8 | `config/legacy_modis_compatibility_attestation.json` | Legacy MODIS attestation | Not used by this analysis | No |
| — | A window-closure JSON/YAML config file | — | **NOT FOUND** — by design. Configuration is code-level constants + the registry; the effective config is *emitted* as `config/preregistration.json`. |

## 5. Schema

| # | Item | Value | Location |
|---|---|---|---|
| 9 | Per-AOI schema | `window_closure_sensitivity.v1` | line 78 |
| 9a | Stage schemas | `window_closure_prelabel_censor.v1` (2116), `window_closure_predictor_export.v1` (2766), `window_closure_local_downstream.v1` (4152), `window_closure_model.v1` (7192), `window_closure_compare.v1` (8844) | as noted |
| — | `multi_region_window_closure.v1` | — | **NOT FOUND** — to be added |
| — | Formal JSON-Schema files | — | **NOT FOUND.** The project's convention is *executable* contracts (assert functions + validators), not declarative schema documents. The new design follows that convention. |

## 6. Validators

| # | Path | Lines | Scope | Reuse |
|---|---|---|---|---|
| 10 | `scripts/validate_window_closure_predictor_export.py` | 813 | Predictor-export stage; **defines `Report`** (line 67) with `technical` / `scientific` / `namespace` categories and the four-verdict `render()` | **Reuse `Report` and helpers** |
| 11 | `scripts/validate_window_closure_local_downstream.py` | 1110 | Local-downstream stage | Reuse per AOI |
| 12 | `scripts/validate_window_closure_model.py` | 826 | Model stage | Reuse per AOI |
| 13 | `scripts/validate_window_closure_compare.py` | 805 | Compare stage; **re-derives every published number** from model artefacts | Reuse per AOI |
| — | Multi-region validator | — | **NOT FOUND** — must be added: `scripts/validate_multi_region_window_closure.py` |

Validator conventions to preserve: `--mode dry-run|actual`, `--experiment`, `--shifts`,
`[PASS]`/`[FAIL]` line stream, four verdicts, exit code 0/1, **never runs a stage**.

## 7. Tests

| # | Path | Lines | Reuse |
|---|---|---|---|
| 14 | `tests/test_window_closure_sensitivity.py` | 4,905 | Reuse; source of fixture patterns |
| 15 | `tests/test_window_closure_local_downstream.py` | 3,287 | Reuse |
| 16 | `tests/test_window_closure_model.py` | 1,418 | Reuse |
| 17 | `tests/test_window_closure_compare.py` | 1,613 | Reuse |
| 18 | `tests/test_evia_2021_extended_registry.py` | — | Registry guard for the extended Evia AOI |
| 19 | `tests/test_mugla_2021_gate.py` | — | Muğla gate contract |
| 20 | `tests/test_step8a_pre_label_exclusion.py` | — | Pre-label exclusion semantics |
| — | Multi-region tests | — | **NOT FOUND** — plan in `IMPLEMENTATION_PLAN.md` §7 |

Injection points that keep tests off Earth Engine (`run_analysis` signature, line 10225):
`output_root`, `experiments_root`, `prelabel_exporter`, `predictor_engine`,
`local_downstream_engine`, `model_configuration_overrides`.

## 8. Manavgat output namespace (read-only reference)

Root: `outputs/diagnostics/window_closure_sensitivity/manavgat_2021/`

| # | Item | Path (relative to root) | Present |
|---|---|---|---|
| 21 | Config | `config/preregistration.json` | ✅ |
| 22 | Input hashes | `config/frozen_input_inventory.json` | ✅ |
| 23 | Window date audit | `config/window_variants.csv` | ✅ |
| 24 | Pre-label censor | `prelabel_censor/{prelabel_burndate.tif, censoring_summary.json, export_plan.json}` | ✅ |
| 25 | Per-variant predictor metadata | `variants/<variant>/predictor_export_metadata.json` | ✅ (7d, 14d) |
| 26 | Per-variant downstream metadata | `variants/<variant>/local_downstream_metadata.json` | ✅ |
| 27 | Cohort inventory | `model/common_cohort/{common_cohort.parquet, common_cohort_metadata.json}` | ✅ |
| 28 | Fold mapping | `model/shared_folds/{shared_spatial_folds.parquet, ..._metadata.json}` | ✅ |
| 29 | Metrics | `model/metrics/{point_metrics.csv, point_metrics.json, thermal_contributions.csv}` | ✅ |
| 30 | OOF predictions | `model/variants/<variant>/<family>/oof_predictions.parquet` (6 files) | ✅ |
| 31 | Fold metrics | `model/variants/<variant>/<family>/fold_metrics.csv` (6 files) | ✅ |
| 32 | Bootstrap replicates | `model/bootstrap/paired_bootstrap_replicates.parquet` | ✅ |
| 33 | Bootstrap summary | `model/bootstrap/{paired_bootstrap_summary.csv, .json}` | ✅ |
| 34 | Stage state | `{model/model_stage_metadata.json, compare/compare_stage_metadata.json}` | ✅ |
| 35 | Compare tables | `compare/tables/*.csv` (6 files) | ✅ |
| 36 | Compare summaries | `compare/summaries/{comparison_summary,provenance_summary,scientific_conclusions}.json` | ✅ |
| 37 | Report | `compare/report/window_closure_comparison.md` | ✅ |
| 38 | Quarantine | `_quarantine/model/20260731T082055Z/…`, `variants/<v>/_quarantine/…` | ✅ — the force/quarantine mechanism is **proven in practice**, not merely designed |
| — | Validator output artefact | — | **NOT FOUND** as a namespace file. Validator results live in `logs/window_closure_*_validation.log` (~50 log files). **Recommendation:** the new design writes `validation_report.json` *into* the namespace. |
| — | `manifest.json` | — | **NOT FOUND.** Provenance is currently spread across per-stage metadata files. **The new design adds a single top-level manifest.** |

## 9. Pipeline components reused by the analysis

| # | Path | Role |
|---|---|---|
| 39 | `src/step3_landsat_lst.py` | `get_current_period_median`, `get_current_period_ndvi_median`, `get_landsat_baseline_window_median_collection`, `get_landsat_baseline_window_ndvi_collection` — **temporal Landsat export** |
| 40 | `scripts/prepare_modis_for_step7.py` | `prepare_modis_for_step7` — **temporal MODIS export**; applies the fixed summer-month filter |
| 41 | `scripts/run_predictors_only.py:672` | `export_image_direct_or_tiled` — direct `getPixels` with 2×2 → 4×4 → 6×6 → 8×8 tiled escalation, atomic `.tmp` + `os.replace`, alignment QA. `DIRECT_EXPORT_SAFE_THRESHOLD_BYTES` ≈ 38.4 MiB (line 101) |
| 42 | `src/step5_preprocess_timeseries.py`, `src/step5c_tvdi.py` | LST baseline stats, TVDI |
| 43 | `src/step7a_tiling_infrastructure.py` … `src/step7e_fuse_landsat_downscaled_lst.py` | Downscaling chain. **`step7c` fits one downscaling model per variant** (single grouped split, one `.fit`, line 718) |
| 44 | `src/step8a_prepare_500m_modeling_dataset.py` | 500 m modelling dataset — **feature assembly**; also applies the pre-label exclusion |
| 45 | `src/step8b_train_baseline_vs_thermal_model.py` | `BASELINE_FEATURES`, `THERMAL_MODEL_FEATURES`, `add_spatial_block_id`, `make_spatial_folds`, `train_population` (line 519) |
| 46 | `src/step8c_spatial_block_bootstrap_uncertainty.py` | `build_block_index`, `compute_metrics`, `summarize_bootstrap` |
| 47 | `src/step6_validate_fire_relation.py` | `export_raw_mcd64a1_prelabel_labels` — pre-label BurnDate raster |
| 48 | `core/utils/tiling.py` | `make_tile_grid`, `iter_windows` |
| 49 | `core/gee_utils.py` | `init_gee` |

## 10. Missing-data policy

Not a separate module. It is **enforced at the cohort gate** (`build_model_common_cohort`,
step 5): any row missing any feature in the baseline ∪ thermal union is dropped, in **every**
variant, before the intersection. No imputation is performed anywhere in this analysis.
Reuse unchanged.

## 11. Reusable multi-AOI orchestration — the key existing asset

| # | Path | Symbol | Why it matters |
|---|---|---|---|
| 50 | `src/multi_aoi_transfer_synthesis/aoi_set.py` | `AoiSet` (frozen dataclass) | Generic, **AOI-name-agnostic** validated selection of 2–5 experiment IDs. Provides `display_order`, `canonical_order` (lexicographic), and `canonical_set_id` (readable slug, hash fallback past 80 chars). `MIN_AOI_COUNT=2`, `MAX_AOI_COUNT=5`. Explicitly documented as knowing "nothing about any specific AOI name or a fixed region count". |
| 51 | `src/multi_aoi_transfer_synthesis/build.py` | `build_synthesis` (109), `_collect` (92) | Dry-run-aware error collection per family/direction |
| 52 | `src/multi_aoi_transfer_synthesis/manifest.py` | — | Manifest construction |
| 53 | `src/multi_aoi_transfer_synthesis/resolvers.py`, `schema_adapters.py` | — | Per-AOI artefact resolution and version-tolerant adaptation |
| 54 | `src/multi_aoi_transfer_synthesis/status_derivation.py`, `render.py` | — | Status derivation and markdown rendering |
| 55 | `src/step9g_multi_aoi_comparison/{discovery,parse,consistency,build,render}.py` | — | A second multi-AOI precedent: discovery → parse → consistency → build → render |

**Reuse decision:** adopt `AoiSet` for the AOI-set identity and canonical ordering of the new
analysis, and mirror the `resolvers → adapters → build → render` layering. This is the single
largest reuse win in the design and is why no new orchestration primitives are proposed.

**Caution (memory-backed):** versioned big-block namespace routing must reach the *generic*
adapter, never the legacy one. Any adapter added for `multi_region_window_closure.v1` must be
registered in `schema_adapters.py` under its own schema key.

## 12. Dependency and provenance audit

| File | Status |
|---|---|
| `requirements.txt` | 12 direct dependencies, all range-pinned |
| `requirements-lock.txt` | 104 exact `==` pins |
| `pytest.ini` | present |
| `.env` | present (not read) |

Cross-check of all 12 direct requirements against the lock:

| Package | Spec | Lock | Consistent |
|---|---|---|---|
| earthengine-api | `>=1.7,<2` | 1.7.34 | ✅ |
| geemap | `>=0.37,<1` | 0.38.3 | ✅ |
| gdown | `>=6,<7` | 6.1.0 | ✅ |
| matplotlib | `>=3.10,<4` | 3.11.0 | ✅ |
| numpy | `>=2.4,<3` | 2.5.1 | ✅ |
| pandas | `>=3.0,<4` | 3.0.3 | ✅ |
| python-dotenv | `>=1.2,<2` | 1.2.2 | ✅ |
| rasterio | `>=1.5,<2` | 1.5.0 | ✅ |
| scikit-learn | `>=1.9,<2` | 1.9.0 | ✅ |
| pyarrow | `>=24,<25` | 24.0.0 | ✅ |
| openpyxl | `>=3.1,<4` | 3.1.5 | ✅ |
| geographiclib | `>=2.0,<3` | 2.1 | ✅ |

**No version conflict exists.** One documentation drift (`W5`): the comment block in
`requirements.txt` states that `geographiclib` is "not installed in this environment yet; the
lock file is deliberately NOT given an entry". The lock **does** contain
`geographiclib==2.1`, so the comment is stale. **Reported only — not fixed in this task**, as
`requirements.txt` is outside the permitted write scope.

`xgboost` is documented as an optional extra, required only for `--model xgboost`. This
analysis is frozen to `random_forest`, so it is not required.

`dependency_lock_hash` in the new manifest is defined as
`sha256(requirements-lock.txt bytes)`.

## 13. Components that must be added

| Component | Kind | Rationale |
|---|---|---|
| `src/multi_region_window_closure/` | new package | Orchestration, synthesis, manifest for the AOI set |
| `scripts/run_multi_region_window_closure.py` | new CLI | Multi-AOI dispatcher over the existing per-AOI runner |
| `scripts/validate_multi_region_window_closure.py` | new validator | The 84 checks in `VALIDATOR_CHECKLIST.md` |
| `tests/test_multi_region_window_closure.py` | new tests | The 26 test groups in `IMPLEMENTATION_PLAN.md` §7 |
| Forbidden-phrase extension | additive constant | The 7 phrases in `W6` |
| `multi_region_window_closure.v1` adapter | registration | In `schema_adapters.py`, generic path |

## 14. Risks identified during inventory

| Risk | Evidence | Mitigation |
|---|---|---|
| Editing `src/window_closure_sensitivity.py` invalidates the Manavgat PASS | The module is the frozen scientific core; Manavgat's `analysis_id` is a hash over its config | **Additive-only** changes; the new package imports it, never edits scientific functions |
| Wording guard is incomplete for the new claims | `FORBIDDEN_COMPARE_PHRASES` lacks 7 task-required phrases | New validator enforces the union (§12.2 of `SCIENTIFIC_CONTRACT.md`) |
| No top-level manifest exists today | `manifest.json` NOT FOUND in the Manavgat namespace | New design mandates one, with explicit self-hash resolution |
| Validator results are not namespace artefacts | They live in `logs/` | New design writes `validation_report.json` into the namespace |
| Muğla's scale (3.0× Manavgat) drives export/tiling escalation | 73,098 vs 24,150 Step8A rows | Sized explicitly in `EXPORT_FEASIBILITY.md` §5–§7 |
| A partially-run AOI could look complete | Stage metadata is per stage, per AOI | Set-level completeness gate: checks `O10`–`O12` |
