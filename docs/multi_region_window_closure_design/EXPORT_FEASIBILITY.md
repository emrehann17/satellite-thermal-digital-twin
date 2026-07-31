# Export Feasibility, Fit / Task / Runtime / Storage Accounting

**No Earth Engine task was created, no export was started, no model was fitted while
producing this document.** Every number below is either derived from frozen code or measured
from artefacts already on disk, and is labelled accordingly.

---

## 1. Correction to a stated assumption — how exports actually happen

The task brief asks for a *Drive target namespace*, *Earth Engine task naming*, *task-ID
provenance* and *duplicate task prevention*. **This pipeline does not use Earth Engine batch
Drive tasks for the predictor export.**

`production_predictor_engine` (`src/window_closure_sensitivity.py:3646`) calls
`scripts.run_predictors_only.export_image_direct_or_tiled` (line 672), which performs a
**synchronous `getPixels` download** straight to a local path — `geemap.ee_export_image`,
escalating to a tiled download when the pre-flight size estimate exceeds
`DIRECT_EXPORT_SAFE_THRESHOLD_BYTES` ≈ 38.4 MiB (`GEE_DIRECT_DOWNLOAD_LIMIT_BYTES × 0.80`).

Consequences, stated rather than papered over:

| Brief's concept | Reality here | Substitute mechanism |
|---|---|---|
| Drive destination folder | None — writes to the local variant namespace | `variants/<variant>/data/**`, containment-asserted by `assert_jobs_inside_variant_namespace` (line 3128) |
| EE task name | None — no task object | `label = f"{variant_id}__{artifact_id}"`, passed to the exporter |
| Task-ID provenance | No task IDs exist | **`artifact_sha256` per raster** in `predictor_export_metadata.json` — a stronger guarantee than a task ID, because it binds content, not a job handle |
| Duplicate-task prevention | Not applicable | `assert_predictor_job_set` (line 3071) refuses duplicate, missing, extra or forbidden artefacts *before* any request |
| Partial-export detection | No Drive polling | Atomic `.tmp` + `os.replace`; `_file_ok` requires the file to exist **and** be > 0 bytes; a silent `geemap` failure that raises nothing is still treated as failure; per-raster alignment QA (CRS, pixel size, bounds, band count) |
| Retry / resume | No task queue | Tiled escalation 2×2 → 4×4 → 6×6 → 8×8; `predictor_variant_is_reusable` (line 3820) gates hash-bound reuse |

The design keeps this mechanism unchanged. `export_plan.csv` therefore records **download
requests**, not batch tasks, and the column is named `request_count` rather than
`task_count`.

Only **one** function in the entire module imports `ee` (`production_predictor_engine`,
line 3652 — `import ee  # Earth Engine enters the process only here`). This is what makes
"GEE only in the export stage" mechanically enforceable rather than a convention.

---

## 2. Static vs temporal feature inventory

**Not decided by name.** The classification is derived by `step8a_predictor_lineage`
(line 5707) from the frozen canonical `step8a_dataset_stats.json:predictor_paths`, which
records the exact raster each predictor family was built from. A source under
`TIMING_DERIVED_SOURCE_DIRS` (`step5`, `step5c`, `step7d`, `step7e`, `data/current_period`,
`data/ndvi_current_period`, `data/landsat_timeseries`, `data/ndvi_timeseries`, `data/modis`)
is timing-derived; a source under `STATIC_SOURCE_DIRS` (`data/dem`) is static. **An
unrecognised source directory raises** rather than defaulting.

This lineage was executed read-only for all four AOIs and is **identical in all four**:

| feature | family | static_or_temporal | window_dependent | source_product | source_code_path | canonical_artifact | reuse_or_recompute | reason | export_required |
|---|---|---|---|---|---|---|---|---|---|
| `ndvi_mean` | NDVI | **temporal** | yes | Landsat NDVI current-window median | `src/step3_landsat_lst.get_current_period_ndvi_median` | `data/ndvi_current_period/current_ndvi_median.tif` | **recompute** | Current-window composite; moves with the closure date | **yes** |
| `current_lst_mean` | LST | **temporal** | yes | Landsat LST current-window median | `src/step3_landsat_lst.get_current_period_median` → `src/step5_preprocess_timeseries` | `step5/current_period_median_celsius.tif` | **recompute** | ″ | **yes** |
| `lst_anomaly_mean` | LST anomaly | **temporal** | yes | Current vs 4 baseline-year windows | `src/step5_preprocess_timeseries` | `step5/anomaly_zscore.tif` | **recompute** | Both current **and** baseline windows shift | **yes** |
| `current_tvdi_mean` | TVDI | **temporal** | yes | LST + NDVI | `src/step5c_tvdi` | `step5c/current_tvdi.tif` | **recompute** | Derived from shifted inputs | no (local) |
| `tvdi_difference_mean` | TVDI | **temporal** | yes | Current vs baseline TVDI | `src/step5c_tvdi` | `step5c/tvdi_difference.tif` | **recompute** | ″ | no (local) |
| `downscaled_lst_mean` | Downscaled LST | **temporal** | yes | Step7C model → Step7D prediction | `src/step7d_predict_downscaled_lst` | `step7d/downscaled_lst_celsius.tif` | **recompute** | Depends on shifted MODIS + Landsat | no (local) |
| `fused_lst_mean` | Fused LST | **temporal** | yes | Landsat ⊕ downscaled | `src/step7e_fuse_landsat_downscaled_lst` | `step7e/fused_lst_celsius.tif` | **recompute** | ″ | no (local) |
| `elevation_mean` | DEM | **static** | no | SRTM/Copernicus DEM | `src/step2b_dem` | `data/dem/elevation.tif` | **reuse read-only** | Window-independent | **no** |
| `slope_mean` | DEM | **static** | no | DEM-derived slope | `src/step2b_dem` | `data/dem/slope.tif` | **reuse read-only** | Window-independent | **no** |
| `landcover_dominant` | Land cover | **static** | no | ESA WorldCover v200, aligned | `src/step6a_prepare_gate_inputs` | `gate_inputs/landcover_esa_worldcover_v200_aligned_to_reference.tif` | **reuse read-only** | Window-independent; frozen input role `landcover_aligned` | **no** |

Additional artefacts held fixed across variants (`STATIC_SHARED_ROLES`, line 171):
`aoi_geometry`, `reference_grid`, `label_window`, `label_raster`, `model_feature_registry`,
`model_hyperparameters`, `random_seed`, `spatial_block_definition`.

`READ_ONLY_SHARED_CONTEXT_KEYS` (line 168) pins `dem_input_dir` and `landcover_aligned_path`
to the **canonical** read-only artefacts, so a variant context physically cannot re-export
them. This is the mechanism that closes the *static raster overwrite* risk.

### 2.1 The canonical variant exports nothing

`predictor_artifact_jobs` (line 2978) **raises** if handed the canonical variant:

> "The canonical variant 'canonical' has no predictor export: it reads the frozen production
> outputs."

So the canonical arm of every AOI reuses the already-verified canonical Step8A dataset — the
artefact whose SHA-256 is the hash anchor. **No canonical re-export is needed or permitted**,
for any of the three new AOIs. This is proven by construction, not argued.

---

## 3. What must be exported

Per **shifted** variant, from `expected_raster_count` (line 2960):

```
landsat_roles  = 2 (current lst, current ndvi) + 2 × 4 baseline years = 10
rasters        = 10 × 2 products + 3 MODIS roles = 23
```

| Group | Roles | Products | Rasters |
|---|---|---|---|
| Current Landsat | `current_lst`, `current_ndvi` | `scene_weighted_median`, `scene_valid_count` | 4 |
| Baseline Landsat | `baseline_{lst,ndvi}_{y1..y4}` (8) | same 2 | 16 |
| MODIS | `modis_lst_mean`, `modis_lst_std`, `modis_valid_observation_count` | — | 3 |
| **Total** | | | **23** |

Verified empirically: `variants/close_7d_earlier/data/` under the Manavgat namespace contains
exactly 23 non-tile `.tif` files matching these names.

Forbidden by hard assertion (`assert_no_forbidden_products`, line 695): `date_balanced_median`
and `date_balanced_minus_scene_weighted` — those belong to the reducer counterfactual, and
allowing them would move the compositing method and the closure date together, making neither
attributable.

Plus **one** pre-label BurnDate raster per AOI (`prelabel_export_plan`, line 836), shared by
all three variants so the censoring cohort is identical across variants by construction.

### 3.1 Scenario totals

```
3 AOI × 2 shifted variants                        =  6 shifted scenarios
6 scenarios × 23 rasters                          = 138 predictor rasters
3 AOI × 1 pre-label raster                        =   3 pre-label rasters
                                                    ---
total logical raster artefacts                      141
```

---

## 4. Download-request accounting

### 4.1 Measured baseline (Manavgat, 24,150 Step8A cells)

From `variants/<v>/data/_tiles/` on disk, identical for both shifted variants:

* **14** Landsat artefacts were tiled, each at **2×2 = 4** tiles
  (4 × `baseline_lst_*__scene_weighted_median`, 4 × `baseline_ndvi_*__scene_valid_count`,
  4 × `baseline_ndvi_*__scene_weighted_median`, `current_ndvi__scene_valid_count`,
  `current_ndvi__scene_weighted_median`)
* **6** Landsat artefacts downloaded directly
* **3** MODIS artefacts via `prepare_modis_for_step7`

```
requests per shifted variant (measured) = 6 + 14×4 + 3 = 65
```

No escalation beyond 2×2 occurred.

### 4.2 Scaling basis

Tiling is driven by estimated raster bytes, which scale with AOI area. **Step8A row count is
used as the area proxy** — it is a measured artefact property, not a geometric guess:

| AOI | Step8A rows | Ratio vs Manavgat |
|---|---|---|
| `manavgat_2021` | 24,150 | 1.00 |
| `bejis_2022` | 15,759 | 0.65 |
| `mugla_2021` | 73,098 | **3.03** |
| `evia_2021_extended` | 22,925 | 0.95 |

### 4.3 Estimates

| Scope | minimum | expected | maximum |
|---|---|---|---|
| **Per shifted variant — `bejis_2022`** | 23 | ~50 | 368 |
| **Per shifted variant — `mugla_2021`** | 23 | ~260 | 368 |
| **Per shifted variant — `evia_2021_extended`** | 23 | ~65 | 368 |
| **Predictor export, all 6 scenarios** | **138** | **~750** | **~2,208** |
| **Pre-label export, 3 AOIs** | 3 | 3 | 12 |
| **TOTAL Earth Engine download requests** | **141** | **~753** | **~2,220** |

Assumptions, stated explicitly:

* **minimum** — every artefact fits under the 38.4 MiB direct threshold; no tiling anywhere.
  Realistic only for Bejís.
* **expected** — Bejís (0.65×) tiles less than Manavgat; Evia-extended (0.95×) matches
  Manavgat's measured 14-of-20 at 2×2; Muğla (3.03×) escalates most artefacts to 4×4
  (≈ 14×16 + 6×4 + 3×4 = 260/variant).
* **maximum** — every one of the 20 Landsat artefacts and all 3 MODIS artefacts escalate to
  4×4 (= 16 tiles): `23 × 16 = 368` per variant. A 6×6/8×8 escalation is available in code but
  was never reached for Manavgat and is judged unlikely; if it occurred, the ceiling would
  rise to `23 × 64 = 1,472` per variant.

**Dominant uncertainty:** the tiling escalation threshold for Muğla. Its 3× area makes the
2×2 → 4×4 boundary the single biggest driver of total request count.

### 4.4 Local downstream requests

**Zero.** `production_local_downstream_engine` (line 5406) imports only Step5/5C/7A–7E/8A and
explicitly notes "none of these modules touches Earth Engine". This is separately asserted by
check `O02`.

Non-GEE local operations, counted separately:

```
local-downstream stage runs = 3 AOI × 2 shifted variants × 8 production stages = 48 stage runs
                              (step5, step5c, step7a, step7b, step7c, step7d, step7e, step8a)
```

---

## 5. Fit accounting

### 5.1 Primary fits — exact formula

```
expected_logical_fits
  = n_AOI × n_variants × n_model_families × n_folds
  = 3     × 3          × 2                × 5
  = 90
```

Structural note: `step8b.train_population` fits **both** families across **all five** folds in
one call, so these 90 estimator fits arise from

```
3 AOI × 3 variants = 9 train_population invocations
```

Both counts must be recorded; they measure different things.

### 5.2 Auxiliary fits — the ones the naive formula misses

`production_local_downstream_engine` runs **Step7C, which trains a downscaling model**.
Inspection of `src/step7c_train_downscaling_model.py` shows a single `grouped_split`
(line 693) followed by exactly one `model.fit(X_train, y_train)` (line 718) — **one fit per
invocation**, no internal cross-validation.

Step7C runs only for **shifted** variants (the canonical arm reuses frozen production
outputs):

```
auxiliary_downscaling_fits = 3 AOI × 2 shifted variants × 1 = 6
```

### 5.3 Fits that do NOT occur

| Candidate | Count | Why zero |
|---|---|---|
| Bootstrap refits | **0** | `multi_variant_block_bootstrap` rescores stored OOF predictions via `step8c.compute_metrics`. Refitting is structurally impossible — the function never receives feature matrices. |
| Full-data descriptive fits | **0** | Not part of the Manavgat contract; none found in the artefacts. |
| Validator audit fits | **0** | The validator re-derives metrics from stored OOF predictions. It never trains. |
| Canonical-variant Step7C fits | **0** | Canonical reuses frozen production Step7C output. |

### 5.4 Totals and accounting fields

```
expected_logical_fits (primary)    = 90
auxiliary_downscaling_fits         =  6
TOTAL logical fits                 = 96
```

`expected_logical_fits` and executed attempts are tracked separately. A retry is an *attempt*,
never a logical fit:

| Field | Definition | PASS condition |
|---|---|---|
| `expected_logical_fits` | 90 (primary) / 96 (incl. auxiliary) | Constant, from the formula |
| `completed_logical_fits` | Distinct `(aoi, variant, model, fold)` with a complete OOF column | `== expected_logical_fits` |
| `duplicate_logical_fits` | Same key produced more than once | `== 0` → else **FAIL** |
| `missing_logical_fits` | Expected key with no result | `== 0` → else **FAIL** |
| `failed_attempts` | Attempts that raised | Recorded; non-zero allowed |
| `retried_attempts` | Attempts repeated after failure | Recorded; non-zero allowed |
| `unexpected_fits` | Key outside the expected cross-product | `== 0` → else **FAIL** |

### 5.5 Bootstrap workload (no fits)

```
per AOI:  1000 replicates × 3 variants × 2 families = 6,000 metric computations
3 AOIs :  3,000 replicate draws, 18,000 metric computations, 0 fits
```

One draw plan per AOI, shared by all comparisons in that AOI, hashed as `draw_plan_hash`.

---

## 6. Runtime estimate

Baseline measured from file mtime spans in the Manavgat namespace. These are **wall-clock
spans of artefact writes** — an underestimate of true stage time — and are labelled as such.

| Stage | Manavgat measured | Basis |
|---|---|---|
| `predictor-export`, per shifted variant | **~25 min** (80 files, 25.4 / 24.5 min) | measured |
| `local-downstream`, per shifted variant | **~4–5 min** write span (86 files); true compute likely 5–15 min incl. Step7C | measured span, model-adjusted |
| `prelabel-export`, per AOI | 124 min *span*, but that includes retries and a quarantine cycle; a clean single MCD64A1 export is minutes | measured, retry-contaminated |
| `model`, per AOI | files written atomically at end (0 min span); fit + 1000 replicates × 3 variants ≈ 5–20 min | model-based |
| `compare`, per AOI | < 1 min | measured |

Scaled by the row-count ratios of §4.2:

| Stage | minimum | expected | maximum |
|---|---|---|---|
| Predictor export (6 shifted variants) | 2.0 h | **3.9 h** | 12 h |
| Local downstream (6 shifted variants) | 0.6 h | **1.5 h** | 4 h |
| Pre-label export (3 AOIs) | 0.2 h | **0.5 h** | 2 h |
| Model (3 AOIs) | 0.3 h | **0.8 h** | 2 h |
| Compare + synthesis (3 AOIs + set) | 0.05 h | **0.1 h** | 0.3 h |
| **TOTAL wall clock** | **~3.2 h** | **~6.8 h** | **~20 h** |

Expected per-AOI export time: Bejís ~18 min/variant, Evia-ext ~24 min/variant,
Muğla ~75 min/variant.

**Assumptions:** single-threaded sequential execution; no Earth Engine quota throttling;
network throughput comparable to the Manavgat run; no re-runs.
**Dominant uncertainty:** Earth Engine responsiveness and Muğla's tiling escalation. The
maximum assumes sustained throttling plus one full retry cycle.

---

## 7. Storage estimate

Measured from the Manavgat namespace (`du -sh`):

```
namespace total                                    19 GB   (includes 10.5 GB quarantine)
variants/close_7d_earlier/  _quarantine 7.0G | downstream 3.5G | data 589M
variants/close_14d_earlier/ _quarantine 3.5G | downstream 3.5G | data 589M
model/ 6.8M   compare/ 196K   config/ 36K   prelabel_censor/ 132K
```

**Retained (non-quarantine) footprint per shifted variant = 3.5 GB downstream + 589 MB data
≈ 4.1 GB.** Within `downstream/`, `step7c/` alone is 2.4 GB — the dominant single component.

Scaled by row-count ratio:

| AOI | ratio | per shifted variant | × 2 variants | + shared |
|---|---|---|---|---|
| `bejis_2022` | 0.65 | 2.7 GB | 5.4 GB | +7 MB |
| `mugla_2021` | 3.03 | 12.4 GB | **24.8 GB** | +7 MB |
| `evia_2021_extended` | 0.95 | 3.9 GB | 7.8 GB | +7 MB |
| **Total retained** | | | **≈ 38 GB** | |

| Quantity | minimum | expected | maximum |
|---|---|---|---|
| Exported raster bytes (`data/`, 6 variants) | 2.0 GB | **3.4 GB** | 6 GB |
| Local downstream bytes (6 variants) | 20 GB | **34 GB** | 55 GB |
| Model + compare + config (3 AOIs) | 20 MB | **25 MB** | 60 MB |
| Set-level synthesis + manifest | 1 MB | **3 MB** | 10 MB |
| **Total analysis storage (retained)** | **~22 GB** | **~38 GB** | **~61 GB** |
| **Temporary peak** (one quarantined generation, as Manavgat actually incurred) | ~30 GB | **~76 GB** | ~140 GB |
| **Final retained after cleanup** | ~22 GB | **~38 GB** | ~61 GB |

**Assumptions:** per-cell storage intensity is the same as Manavgat's; one quarantine
generation for the peak figure; no additional retries.
**Dominant uncertainty:** how many quarantine generations accumulate. Manavgat's namespace
carries 10.5 GB of quarantine against 8.2 GB of retained variant data — i.e. quarantine can
**exceed** live data.

### 7.1 Disk headroom — not a blocker

```
Filesystem  Size  Used  Avail  Use%
/dev/sdd    1007G   73G   884G    8%
```

Expected peak ~76 GB against 884 GB available (≈ 8.6 % of free space). Even the 140 GB
worst case leaves > 700 GB. **`insufficient disk storage` is not a blocker.**

Recommended pre-flight gate anyway: refuse to start the actual run if free space is below
`3 × expected_total` (≈ 120 GB) — check `O14`.

### 7.2 OOF and bootstrap output sizes

```
oof_predictions   rows = Σ_aoi (cohort_rows × 3 variants × 2 families)
                       ≈ (13.5k + 37k + 8.2k) × 6 ≈ 352k rows  → ~15 MB parquet
bootstrap_replicates    3 AOI × 1000 replicates, ~30 numeric columns → ~2 MB parquet
```

Manavgat's entire `model/` tree is 6.8 MB for one AOI, which corroborates these figures.

---

## 8. Stage-level export controls

### 8.1 Dry-run is provably read-only

`run_analysis` (line 10225) does not merely *promise* read-only behaviour — it **proves** it
by bracketing the whole plan with before/after snapshots of every stage-owned path:

```python
stage_owned_before  = snapshot_local_downstream_state(...)   # line 6441
model_state_before  = snapshot_model_state(...)              # line 7333
compare_state_before= snapshot_compare_state(...)            # line 8955
...  build the entire plan ...
result.update(local_downstream_state_diff(before, after))    # line 6512
```

Any created, modified or deleted stage-owned path is **reported**, not inferred. The code
comment states the reasoning precisely: demanding an empty tree "would be a false positive on
any namespace that has been run before".

The dry-run result carries `files_written: false`, `gee_queries_run: false`,
`gee_exports_run: false`, `model_fit: false`, `bootstrap_run: false`. The multi-region
dry-run must aggregate these across AOIs and fail if any is true (checks `O03`–`O04`).

### 8.2 Namespace containment

| Guard | Line | Effect |
|---|---|---|
| `assert_plan_owned_targets` | 1998 | Plan writes only to paths the plan owns |
| `assert_jobs_inside_variant_namespace` | 3128 | Every export job path is inside its variant namespace |
| `assert_local_downstream_owned_targets` | 5377 | Downstream writes stay in-namespace |
| `assert_variant_context_safe` | 490 | Variant context never mutates the registry |
| `assert_frozen_hashes_unchanged` | 1781 | Frozen inputs identical before and after every stage |
| `READ_ONLY_SHARED_CONTEXT_KEYS` | 168 | DEM and land cover physically cannot be re-exported |
| `frozen_input_sha256_before` / `_after` | metadata | Recorded per stage, compared |
| `canonical_outputs_modified` | metadata | Explicit flag, must be `false` |

`canonical_export_attempted` is also recorded and must be `false` — verified present in the
Manavgat `predictor_export_metadata.json`.

### 8.3 Partial-export detection

1. `assert_predictor_job_set` — exact expected artefact set, before any request.
2. `_file_ok` — file exists **and** > 0 bytes; a silent `geemap` failure still counts as a
   failure.
3. Atomic `.tmp` + `os.replace` — an interruption never leaves a half file at the target path.
4. `inspect_predictor_raster` (line 3512) — band count, grid family, pixel size within
   `1e-3` relative tolerance, count-product non-negativity.
5. `artifact_sha256` per raster in the stage metadata.
6. `predictor_variant_is_reusable` (line 3820) — resume only when all hashes still match.

---

## 9. Risk register

| Risk | Likelihood | Impact | Evidence | Mitigation | Pre-implementation gate |
|---|---|---|---|---|---|
| Canonical hash drift | Low | Critical | All 4 verified at design time | Re-verify immediately before the actual run; `frozen_input_sha256_before/after` per stage | Check `C01`–`C05` |
| Date-source ambiguity | **Resolved** | — | Registry is the single source | `event_*`/`gate_*` aliased to the label window with recorded `source_field` | `WINDOW_DATE_AUDIT.md` §1.1 |
| Off-by-one date bug | Low | High | End-exclusivity explicit; duration asserted | Both duration conventions emitted separately | Checks `D02`, `D07` |
| MODIS season policy drift | Low | High | `SUMMER_MONTH_*` are shared globals | Record and assert both values in `config.json` | Check `D08` |
| **MODIS clipping asymmetry across AOIs** | **Certain** | Medium | Bejís 0/0 d vs Manavgat & Muğla 7/14 d, Evia-ext 3/10 d | Per-AOI reporting; explicit limitation; **never corrected** | `W1`/`W2`, limitation 6 |
| AOI-specific hard-coded path | Low | High | None found in the module | Registry-driven throughout; test `T05` | Check `S01`–`S08` |
| Static raster overwrite | **Very low** | Critical | `READ_ONLY_SHARED_CONTEXT_KEYS` pins DEM/land cover to canonical | Structural, not procedural | Check `C04` |
| Temporal export collision | Low | High | Per-variant namespaces | `assert_jobs_inside_variant_namespace` | Check `O07` |
| Drive folder / task-name collision | **N/A** | — | No Drive, no batch tasks | See §1 | — |
| Partial export | Medium | High | Manavgat needed quarantine cycles | Six-layer detection, §8.3 | Check `O11` |
| Variant cohort mismatch | Low | Critical | Intersection is structural | `BLOCKER: VARIANT_COHORT_MISMATCH` | Checks `H01`–`H08` |
| Fold class infeasibility | **Very low** | Critical | All AOIs ≫ `min_positives=30` | `BLOCKER: FOLD_CLASS_INFEASIBILITY` | Checks `F03`–`F05` |
| High Evia prevalence (0.288) | **Certain** | Medium | Measured | Mandatory different-regime framing; no cross-AOI PR-AUC comparison | Checks `Y03`–`Y04` |
| Missing temporal observations | Medium | Medium | Manavgat lost 353–585 rows/variant to feature-union incompleteness | Intersection gate; counts reported per variant | Check `H09` |
| Dependency / version drift | Low | Medium | Lock fully consistent (§12 of inventory) | `dependency_lock_hash` in manifest | Check `C07` |
| Stale resume state | Medium | High | Resume is hash-bound | File existence alone never sufficient | Checks `O05`, `O16` |
| Manifest incompleteness | Low | High | No manifest exists today | New mandatory manifest; every file recorded or the stage fails | Checks `O12`–`O13` |
| **Insufficient disk storage** | **Low** | High | 884 GB free vs ~76 GB peak | Pre-flight free-space gate at 3× expected | Check `O14` |
| Duplicate logical fit | Low | High | Accounting fields §5.4 | `duplicate_logical_fits == 0` | Check `M02` |
| Excess invalid bootstrap replicates | Low | Medium | Manavgat: 0 invalid of 1000 | `valid + invalid == requested`, asserted | Checks `B04`–`B06` |
| Report wording drift | Medium | Medium | Existing guard misses 7 required phrases | Union enforcement | Checks `Y05`–`Y08` |
| **Pooled inference leakage** | Medium | **Critical** | Easy to introduce accidentally in a synthesis table | No column in `four_region_synthesis.csv` may be a function of >1 AOI | Checks `Y01`–`Y02` |

**Unresolved critical risks: none.**
