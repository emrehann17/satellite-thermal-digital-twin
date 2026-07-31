# Output Schema — `multi_region_window_closure.v1`

Namespace:

```
outputs/diagnostics/multi_region_window_closure/<analysis_id>/
```

`<analysis_id>` is a SHA-256 over the canonical-JSON of the set-level frozen configuration,
following `compute_analysis_id` (`src/window_closure_sensitivity.py:1127`). Its inputs are:
schema version, the canonical AOI ordering (`AoiSet.canonical_order`), the shift tuple
`(0, 7, 14)`, the four canonical AOI hashes, the Manavgat reference hash, the frozen model and
bootstrap configuration, and the MODIS season policy values.

Naming convention shared by all tabular outputs: `long` format, one row per fully-qualified
key, no wide reshaping except where explicitly stated.

---

## 0. Symbols used in row-count formulas

| Symbol | Value |
|---|---|
| `A` | number of actual AOIs = **3** |
| `V` | number of variants = **3** |
| `M` | number of model families = **2** |
| `K` | folds per AOI = **5** |
| `R` | requested bootstrap replicates = **1000** |
| `N_a` | common-cohort rows of AOI `a` |
| `S` | shifted variants = **2** |
| `metrics` | `{roc_auc, pr_auc, brier}` = **3** |

---

## 1. File inventory

| File | Stage | Required | Grain |
|---|---|---|---|
| `config.json` | plan | ✅ | one document |
| `input_hashes.json` | plan | ✅ | one document |
| `repository_inventory.json` | plan | ✅ | one document |
| `window_dates.csv` | plan | ✅ | AOI × variant |
| `export_plan.csv` | plan | ✅ | AOI × variant × artefact |
| `cohort_inventory.csv` | cohort-feasibility | ✅ | AOI × variant |
| `fold_mapping.parquet` | cohort-feasibility | ✅ | AOI × cohort row |
| `variant_artifact_index.csv` | export / local-downstream | ✅ | AOI × variant × artefact |
| `metrics.csv` | fit | ✅ | AOI × variant × model × metric |
| `oof_predictions.parquet` | fit | ✅ | AOI × variant × model × row |
| `bootstrap_replicates.parquet` | compare | ✅ | AOI × comparison × replicate |
| `bootstrap_summary.csv` | compare | ✅ | AOI × comparison × variant × metric × orientation |
| `regional_summary.csv` | summarize | ✅ | AOI |
| `four_region_synthesis.csv` | summarize | ✅ | AOI × comparison × metric |
| `summary.json` | summarize | ✅ | one document |
| `report.md` | summarize | ✅ | one document |
| `stages/<stage>.json` | each stage | ✅ | one per stage |
| `manifest.json` | summarize | ✅ | one document |
| `validation_report.json` | validator | ✅ | one document |

Any file present on disk but absent from `manifest.files[]` is a **stray file** and fails the
stage — mirroring the existing compare-stage rule.

---

## 2. `config.json`

**Purpose:** the frozen, hashable configuration. Nothing about the run may be inferred from
anywhere else.
**Grain:** one document. **Primary key:** `analysis_id`.

| Key | Type | Null | Constraint |
|---|---|---|---|
| `analysis_id` | string(64) | no | lowercase hex SHA-256 |
| `schema_version` | string | no | `== "multi_region_window_closure.v1"` |
| `aois_actual` | array[string] | no | **exactly** `["bejis_2022","evia_2021_extended","mugla_2021"]` (canonical order) |
| `aoi_reference` | string | no | `== "manavgat_2021"` |
| `aois_excluded` | array[string] | no | must contain `"evia_2021"` |
| `aoi_set_id` | string | no | `AoiSet.canonical_set_id` |
| `variants` | array[string] | no | `["canonical","close_7d_earlier","close_14d_earlier"]` |
| `shift_days` | array[int] | no | `[0,7,14]` |
| `primary_population` | string | no | `"burnable_tree_shrub_grass"` |
| `model_families` | array[string] | no | `["baseline","thermal"]` |
| `metrics` | array[string] | no | `["roc_auc","pr_auc","brier"]` |
| `model_configuration` | object | no | mirrors `model_frozen_configuration()` |
| `bootstrap_configuration` | object | no | `n_bootstrap`, `seed`, `ci_lower`, `ci_upper`, `unit`, `interval_method="percentile"` |
| `feature_registry` | object | no | mirrors `model_feature_registry()` |
| `modis_season_policy` | object | no | `{summer_month_start: 6, summer_month_end: 9, source: "core.config"}` |
| `canonical_aoi_sha256` | object | no | 4 entries, AOI → hex |
| `git_commit`, `git_branch`, `dirty_worktree` | string/string/bool | no | recorded |
| `limitations` | array[string] | no | ≥ 11 entries (`SCIENTIFIC_CONTRACT.md` §13) |

---

## 3. `input_hashes.json`

**Purpose:** the complete frozen-input provenance for every AOI.
**Grain:** one document. **Primary key:** `analysis_id`.

Shape: `{ aoi → { role → {path, sha256, exists, required, resolved_from} } }` over the six
required roles (`REQUIRED_FROZEN_INPUT_ROLES`): `canonical_step8a`, `dem_elevation`,
`dem_slope`, `landcover_aligned`, `label_raw_burndate`, `label_burned_binary`; plus the
optional `canonical_step8a_stats`.

Constraints: `sha256` is 64 lowercase hex; `exists == true` for every required role;
`aoi ∈ {3 actual} ∪ {manavgat_2021}`; `input_hashes_hash = sha256(canonical_json(document))`.

All six roles were verified present for all four AOIs at design time.

---

## 4. `repository_inventory.json`

**Purpose:** machine-readable record of the code that produced the run.
**Grain:** one document.

| Key | Type | Constraint |
|---|---|---|
| `git_commit`, `git_branch`, `dirty_worktree` | string/string/bool | recorded |
| `python_version` | string | recorded |
| `dependency_lock_hash` | string(64) | `sha256(requirements-lock.txt)` |
| `modules[]` | array | `{path, sha256, role}` for every module in the analysis import graph |
| `reused_components[]` | array | `{path, symbol, purpose}` |

`dirty_worktree == true` is a **WARNING**, not a FAIL — but it must be surfaced in
`summary.json.warnings` and in `report.md`.

---

## 5. `window_dates.csv`

**Purpose:** every exact date, in ISO form, with its inclusivity semantics.
**Grain:** one row per AOI × variant. **Primary key:** `(analysis_id, aoi, variant)`.
**Row count:** `4 × 3 = 12` — the three actual AOIs **plus** the Manavgat reference row, so
the reference contract is auditable in the same table.

| Column | Type | Null | Constraint |
|---|---|---|---|
| `analysis_id` | string(64) | no | constant |
| `aoi` | string | no | ∈ 3 actual ∪ `manavgat_2021`; **never** `evia_2021` |
| `variant` | string | no | ∈ 3 variants |
| `predictor_start`, `predictor_end` | date | no | ISO `YYYY-MM-DD` |
| `predictor_start_inclusive` | bool | no | always `true` |
| `predictor_end_inclusive` | bool | no | always `true` (registry semantics) |
| `earth_engine_filter_start` | date | no | `== predictor_start` |
| `earth_engine_filter_end` | date | no | `== predictor_end` |
| `earth_engine_end_exclusive` | bool | no | always `true` |
| `effective_last_included_date` | date | no | `predictor_end − 1 day` |
| `calendar_duration_days` | int | no | `(end − start).days`; equal for all variants of an AOI |
| `calendar_days_inclusive` | int | no | `calendar_duration_days + 1` |
| `effective_observation_days` | int | no | MODIS effective days after the fixed month filter |
| `modis_clipped_days` | int | no | `calendar_duration_days − effective_observation_days`, ≥ 0 |
| `shift_days` | int | no | ∈ `{0,7,14}`; `0` ⟺ `variant == "canonical"` |
| `lead_days` | int | no | `label_start − predictor_end`; ≥ 1 |
| `label_start`, `label_end` | date | no | identical across variants of an AOI |
| `event_start`, `event_end` | date | no | **alias of the label window** — see `WINDOW_DATE_AUDIT.md` §1.1 |
| `gate_start`, `gate_end` | date | no | **alias of the label window** — same caveat |
| `event_source_field`, `gate_source_field` | string | no | `"EXPERIMENTS[aoi].label_start_date|label_end_date"` — mandatory provenance so the aliasing is never mistaken for a distinct field |
| `prelabel_censor_start`, `prelabel_censor_end` | date | no | constant per AOI |
| `modis_policy_id` | string | no | `"core.config.SUMMER_MONTH_6_9"` |
| `source_config_path` | string | no | `"core/regions.py::EXPERIMENTS"` |
| `source_config_hash` | string(64) | no | `sha256(core/regions.py)` |
| `date_contract_pass` | bool | no | all row-level assertions passed |
| `failure_reason` | string | **yes** | non-null ⟺ `date_contract_pass == false` |

**Uniqueness:** `(aoi, variant)` unique.
**Cross-row invariants:** within an AOI — `calendar_duration_days` constant; `label_*`,
`event_*`, `gate_*` constant; `predictor_start` and `predictor_end` each decrease by exactly
`shift_days` from the canonical row.

The expected content of all 12 rows is tabulated in `WINDOW_DATE_AUDIT.md` §3.

---

## 6. `export_plan.csv`

**Purpose:** every artefact that will be produced or reused, and why.
**Grain:** one row per AOI × variant × artefact.
**Primary key:** `(analysis_id, aoi, variant, artifact_id)`.

**Row count:**
```
predictor artefacts : A × S × 23              = 3 × 2 × 23 = 138
canonical reuse rows: A × 1 × 23              = 3 × 1 × 23 =  69
pre-label rows      : A × 1                   =              3
static reuse rows   : A × |STATIC_SHARED_ROLES| = 3 × 11    = 33
                                                            ---
total                                                        243
```

| Column | Type | Constraint |
|---|---|---|
| `analysis_id`, `aoi`, `variant` | string | as above |
| `artifact_id` | string | e.g. `baseline_lst_2019__scene_weighted_median` |
| `role` | string | `current_lst`, `baseline_ndvi_2019`, `modis_lst_mean`, `dem_elevation`, … |
| `family` | string | `lst` \| `ndvi` \| `modis` \| `static` \| `label` |
| `static_or_temporal` | string | `static` \| `temporal` |
| `window_dependent` | bool | `true` ⟺ `static_or_temporal == "temporal"` |
| `reuse_or_recompute` | string | `reuse` \| `recompute` |
| `export_required` | bool | **must be `false` for every canonical-variant row and every static row** |
| `grid_family` | string | `landsat_30m` \| `modis_1km` |
| `export_scale_m` | int | 30 \| 1000 |
| `start_date`, `end_date` | date | job window |
| `expected_band_count` | int | 1 |
| `is_count_product` | bool | from `COUNT_PRODUCTS` |
| `output_path` | string | must be inside the variant namespace |
| `producer` | string | production function reference |
| `estimated_request_count` | int | 1 if direct, else tiles |
| `transport` | string | `direct` \| `tiled_2x2` \| `tiled_4x4` \| `production_modis` \| `reuse` |
| `reason` | string | free text, non-empty |

**Constraint (canonical never exports):** `variant == "canonical"` ⇒ `export_required == false`
and `reuse_or_recompute == "reuse"`. Enforced structurally by `predictor_artifact_jobs`, which
raises for the canonical variant.

---

## 7. `cohort_inventory.csv`

**Purpose:** full audit of cohort construction, per AOI and variant.
**Grain:** AOI × variant. **Primary key:** `(analysis_id, aoi, variant)`. **Rows:** `3 × 3 = 9`.

| Column | Type | Constraint |
|---|---|---|
| `analysis_id`, `aoi`, `variant` | string | |
| `initial_rows` | int | Step8A row count |
| `removed_not_valid_for_modeling` | int | ≥ 0 |
| `removed_outside_primary_population` | int | ≥ 0 |
| `removed_prelabel_censor` | int | ≥ 0 |
| `removed_missing_required_feature_union` | int | ≥ 0 |
| `removed_variant_only_keys` | int | ≥ 0 |
| `removed_label_mismatch` | int | **must be 0** — a mismatch raises |
| `removed_static_invariance_failure` | int | **must be 0** — a mismatch raises |
| `final_common_cohort_rows` | int | > 0; **identical for all 3 variants of an AOI** |
| `final_positive_rows` | int | ≥ `min_positives` = 30; identical across variants |
| `final_negative_rows` | int | > 0; identical across variants |
| `prevalence` | float | `positives / rows`, in (0,1); identical across variants |
| `cohort_hash` | string(64) | sha256 of the sorted `cell_id` list; **identical for all 3 variants of an AOI** |
| `duplicate_cell_ids` | int | must be 0 |
| `feasibility_pass` | bool | all 14 checks |
| `failure_reason` | string, nullable | non-null ⟺ `feasibility_pass == false` |

**The decisive check:** `cohort_hash` must be byte-identical for the three variants of one AOI,
and **different** between AOIs.

---

## 8. `fold_mapping.parquet`

**Purpose:** the one shared fold assignment per AOI.
**Grain:** AOI × cohort row. **Primary key:** `(analysis_id, aoi, cell_id)`.
**Row count:** `Σ_a N_a` — note this is **per AOI, not per variant**, precisely because one
assignment is shared by all variants.

| Column | Type | Constraint |
|---|---|---|
| `analysis_id`, `aoi` | string | |
| `cell_id` | string/int64 | unique within AOI |
| `grid_id` | string | stable grid identity |
| `block_id` | string/int64 | `spatial_block_id` |
| `fold_id` | int8 | ∈ `[0, 4]`; exactly one per row |
| `y_true` | int8 | ∈ `{0,1}` |
| `cohort_hash` | string(64) | matches `cohort_inventory` |
| `fold_mapping_hash` | string(64) | `assignment_sha256`; constant within AOI |

**Invariants:** every `block_id` lies wholly within one `fold_id`; every fold contains both
classes; no `cell_id` appears twice; `fold_mapping_hash` differs between AOIs.

---

## 9. `variant_artifact_index.csv`

**Purpose:** content-addressed index of every artefact actually produced.
**Grain:** AOI × variant × artefact. **Primary key:** `(analysis_id, aoi, variant, relative_path)`.

Columns: `analysis_id`, `aoi`, `variant`, `stage`, `relative_path`, `artifact_id`, `role`,
`media_type` (from `MEDIA_TYPES`, line 5457), `size_bytes` (> 0), `sha256`, `band_count`,
`pixel_size_x/y`, `crs`, `alignment_qa_pass`, `produced_at_utc`, `transport`.

**Constraint:** every `relative_path` resolves inside
`outputs/diagnostics/multi_region_window_closure/<analysis_id>/`. No path may point into
`outputs/experiments/**` — that would mean a canonical artefact was written.

---

## 10. `metrics.csv`

**Purpose:** pooled out-of-fold point metrics.
**Grain:** AOI × variant × model × metric.
**Primary key:** `(analysis_id, aoi, variant, model, metric)`.
**Row count:** `A × V × M × 3 = 3 × 3 × 2 × 3 = 54`.

| Column | Type | Null | Constraint |
|---|---|---|---|
| `analysis_id` | string(64) | no | |
| `aoi` | string | no | ∈ 3 actual |
| `variant` | string | no | ∈ 3 variants |
| `model` | string | no | `baseline` \| `thermal` |
| `metric` | string | no | `roc_auc` \| `pr_auc` \| `brier` |
| `estimate` | float64 | no | `roc_auc`,`pr_auc` ∈ [0,1]; `brier` ∈ [0,1] |
| `metric_direction` | string | no | `higher_is_better` \| `lower_is_better` |
| `n_rows` | int | no | `== N_a`; constant across variants and models of an AOI |
| `n_positive`, `n_negative` | int | no | both > 0; sum to `n_rows` |
| `prevalence` | float | no | `n_positive / n_rows` |
| `fold_count` | int | no | `== 5` |
| `oof_complete` | bool | no | **must be `true`** |
| `cohort_hash` | string(64) | no | matches `cohort_inventory` |
| `fold_mapping_hash` | string(64) | no | matches `fold_mapping` |
| `prediction_hash` | string(64) | no | sha256 of the ordered `y_score` array |

**Validator obligation:** every `estimate` must be **recomputed** from `oof_predictions.parquet`
and agree to `1e-9` — the same standard the existing compare validator already applies.

---

## 11. `oof_predictions.parquet`

**Purpose:** one out-of-fold prediction per row, model and variant — the substrate for every
metric and every bootstrap replicate.
**Grain:** AOI × variant × model × cohort row.
**Primary key:** `(analysis_id, aoi, variant, model, row_id)`.
**Row count:** `Σ_a (N_a × V × M) = Σ_a (N_a × 6)` ≈ 352,000 (projected).

| Column | Type | Null | Constraint |
|---|---|---|---|
| `analysis_id`, `aoi`, `variant`, `model` | string | no | |
| `row_id` | int64 | no | dense `[0, N_a)` in `cell_id` order |
| `grid_id` | string | no | |
| `cell_id` | string/int64 | no | |
| `block_id` | string/int64 | no | matches `fold_mapping` |
| `fold_id` | int8 | no | matches `fold_mapping` — **identical across variants and models** |
| `y_true` | int8 | no | ∈ `{0,1}`; **identical for a given `(aoi, cell_id)` across all variants and models** |
| `y_score` | float64 | no | ∈ [0,1]; finite |
| `cohort_hash`, `fold_mapping_hash` | string(64) | no | |

**Duplicate rule:** `(aoi, variant, model, cell_id)` is unique. **Coverage rule:** for every
`(aoi, variant, model)` the `cell_id` set equals that AOI's cohort exactly — no row missing,
no row extra. Both are validator FAILs (`M04`, `M05`).

---

## 12. `bootstrap_replicates.parquet`

**Purpose:** replicate-level paired differences. The interval is derived from this file, never
asserted independently.
**Grain:** AOI × comparison × variant × metric × replicate.
**Primary key:** `(analysis_id, aoi, comparison_family, variant, metric, replicate_id)`.

**Row count (valid replicates only):**
```
per AOI: comparisons × metrics × valid_replicates
  thermal_contribution_within_variant : 3 variants × 3 metrics = 9
  closure_change_within_model_family  : 2 shifted × 2 models × 3 metrics = 12
  thermal_contribution_change         : 2 shifted × 3 metrics = 6
                                        -------------------------
                                        27 series × 1000 = 27,000 per AOI
total ≈ 81,000 rows
```

| Column | Type | Null | Constraint |
|---|---|---|---|
| `analysis_id`, `aoi` | string | no | |
| `comparison_family` | string | no | one of the three families |
| `variant` | string | no | the variant the row is *about* |
| `model_a`, `model_b` | string | no | the two members; `NULL` not allowed — use the literal family name |
| `metric` | string | no | ∈ 3 metrics |
| `replicate_id` | int | no | ∈ `[0, R)` |
| `draw_plan_id` | string | no | **identical for every row of one AOI** — this is what makes the differences paired |
| `estimate_a`, `estimate_b` | float64 | no | finite |
| `difference_natural` | float64 | no | `estimate_a − estimate_b` in the raw convention |
| `difference_oriented` | float64 | no | Brier: `−difference_natural`; ROC/PR: `== difference_natural` |
| `valid` | bool | no | `true` for every stored row |
| `invalid_reason` | string | **yes** | `NULL` when `valid` |

**Design note on invalid replicates.** The existing implementation stores **one row per
globally valid draw** and omits invalid draws entirely, with counts carried in the summary
(`validate_saved_bootstrap_replicate_counts`, line 1427). The new schema keeps that
storage rule — `valid` is always `true` in the file — and preserves the accounting through
`requested_replicates`, `valid_replicates` and `invalid_replicates` in `bootstrap_summary.csv`.
The `valid` / `invalid_reason` columns are retained for schema uniformity and forward
compatibility. This is stated so the always-`true` column is not later mistaken for a bug.

**No-refit guarantee:** every replicate is a rescoring of `oof_predictions.parquet` under a
block resample. The file contains no model artefact reference, and the bootstrap function
never receives a feature matrix.

---

## 13. `bootstrap_summary.csv`

**Purpose:** the intervals actually reported.
**Grain:** AOI × comparison × variant × metric × orientation.
**Primary key:** `(analysis_id, aoi, comparison_family, variant, metric, orientation)`.
**Row count:** `3 AOI × 27 series × 2 orientations = 162`.

| Column | Type | Null | Constraint |
|---|---|---|---|
| `analysis_id`, `aoi`, `comparison_family`, `variant`, `metric` | string | no | |
| `orientation` | string | no | `natural` \| `oriented` — **mandatory; never inferred** |
| `orientation_definition` | string | no | e.g. `"baseline_brier − thermal_brier (positive favours thermal)"` |
| `point_estimate` | float64 | no | computed from `metrics.csv`, **not** the bootstrap mean |
| `bootstrap_mean` | float64 | no | mean over valid replicates; reported separately from `point_estimate` |
| `ci_low`, `ci_high` | float64 | no | `ci_low <= ci_high` |
| `confidence_level` | float | no | `== 95.0` |
| `interval_method` | string | no | `"percentile"` |
| `ci_lower_percentile`, `ci_upper_percentile` | float | no | `2.5`, `97.5` |
| `requested_replicates` | int | no | `== 1000` |
| `valid_replicates` | int | no | `== len(replicate rows for the series)` |
| `invalid_replicates` | int | no | `== requested − valid` |
| `seed` | int | no | `== 42` |
| `draw_plan_hash` | string(64) | no | constant within AOI; differs across AOIs |
| `block_count` | int | no | ≥ 2 |
| `interval_excludes_zero` | bool | no | `ci_low > 0 or ci_high < 0` |
| `interval_status` | string | no | `bootstrap_supported_increase` \| `bootstrap_supported_decrease` \| `interval_includes_zero` |

**Consistency rule:** `interval_excludes_zero == (interval_status != "interval_includes_zero")`.
**Arithmetic rule:** `ci_low` and `ci_high` must be reproducible as the 2.5 / 97.5 percentiles
of the matching replicate series to `1e-9`.

---

## 14. `regional_summary.csv`

**Purpose:** one self-contained summary per AOI.
**Grain:** one row per AOI. **Primary key:** `(analysis_id, aoi)`. **Rows: 3.**

Columns: `analysis_id`, `aoi`, `aoi_role` (`new_actual` \| `different_regime_control`),
`display_name`, `country`, `canonical_step8a_sha256`, `predictor_start`, `predictor_end`,
`calendar_duration_days`, `label_start`, `label_end`, `cohort_rows`, `positives`, `negatives`,
`prevalence`, `block_count`, `fold_count`, `cohort_hash`, `fold_mapping_hash`,
`modis_clipped_days_7d`, `modis_clipped_days_14d`,
`thermal_contribution_roc_auc_canonical/_7d/_14d`,
`closure_change_thermal_roc_auc_7d/_14d`,
`interval_status_*` (per reported comparison), `technical_status`, `n_supported_intervals`,
`n_intervals_including_zero`, `regime_note`.

**Constraint:** `aoi == "evia_2021_extended"` ⇒ `aoi_role == "different_regime_control"` and
`regime_note` non-empty containing the mandated framing.
**Constraint:** no column here is a function of any other AOI's data.

---

## 15. `four_region_synthesis.csv`

**Purpose:** side-by-side **descriptive** presentation of the three new AOIs plus the Manavgat
read-only reference.
**Grain:** AOI × comparison_family × variant × metric.
**Primary key:** `(analysis_id, aoi, comparison_family, variant, metric)`.
**Row count:** `4 AOI × 27 series = 108`.

| Column | Type | Constraint |
|---|---|---|
| `analysis_id` | string(64) | |
| `aoi` | string | ∈ 3 actual ∪ `manavgat_2021`; **never** `evia_2021` |
| `aoi_source` | string | `this_analysis` \| `read_only_reference` |
| `reference_artifact_path` | string, nullable | non-null ⟺ `aoi_source == "read_only_reference"` |
| `reference_artifact_sha256` | string(64), nullable | same condition |
| `comparison_family`, `variant`, `metric`, `orientation` | string | |
| `point_estimate`, `ci_low`, `ci_high` | float64 | per-AOI values, copied verbatim |
| `interval_status` | string | per-AOI |
| `prevalence` | float | that AOI's own |
| `regime_class` | string | `equal_prevalence_validation` \| `different_regime_control` |
| `cross_aoi_comparable` | bool | **`false` for `pr_auc` and `brier`**; `true` only for direction-of-`roc_auc` narrative |
| `descriptive_only` | bool | **hard-coded `true` in every row** |

**Prohibited by schema — these columns may not exist:** `pooled_estimate`, `pooled_ci_low`,
`pooled_ci_high`, `meta_analytic_estimate`, `combined_p`, `heterogeneity`, `i_squared`,
`weight`, `n_total_across_aois`, or any aggregate over AOIs.

**Structural rule (check `Y01`):** every value in every row is traceable to exactly one AOI's
artefacts. The validator asserts that no cell was computed from more than one AOI by
re-deriving each row from that AOI's `bootstrap_summary.csv` alone.

Manavgat rows are **copied** from
`outputs/diagnostics/window_closure_sensitivity/manavgat_2021/**` with the source path and
hash recorded. They are never recomputed.

---

## 16. `summary.json`

**Purpose:** machine-readable status, with technical and scientific verdicts strictly
separated.
**Grain:** one document.

```jsonc
{
  "analysis_id": "<64 hex>",
  "schema_version": "multi_region_window_closure.v1",
  "created_at_utc": "<ISO-8601 Z>",

  "technical_status": {
    "overall": "PASS|FAIL",
    "stages": { "<stage>": "PASS|FAIL|SKIP" },
    "checks_total": 84, "checks_passed": 0, "checks_failed": 0, "checks_skipped": 0,
    "required_checks_skipped": 0,          // must be 0 for overall PASS
    "manifest_complete": true,
    "canonical_outputs_modified": false,   // must be false
    "expected_logical_fits": 90, "completed_logical_fits": 90,
    "duplicate_logical_fits": 0, "missing_logical_fits": 0, "unexpected_fits": 0
  },

  "scientific_summary": {
    "inference_unit": "per_aoi",
    "pooled_inference": false,             // must be false
    "per_aoi": {
      "<aoi>": {
        "aoi_role": "new_actual|different_regime_control",
        "prevalence": 0.0,
        "comparisons": [
          { "comparison_family": "...", "variant": "...", "metric": "...",
            "orientation": "natural|oriented",
            "point_estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0,
            "interval_status": "bootstrap_supported_increase|bootstrap_supported_decrease|interval_includes_zero" }
        ]
      }
    },
    "cross_aoi_statement": "descriptive_only"
  },

  "limitations": ["..."],                  // >= 11 entries
  "blockers":    [ { "id": "...", "message": "...", "evidence": "..." } ],
  "warnings":    [ { "id": "...", "message": "...", "evidence": "..." } ]
}
```

**Rules:** `technical_status.overall == "PASS"` requires zero FAILs **and** zero skipped
required checks. `blockers` non-empty ⇒ `technical_status.overall == "FAIL"`. A technical PASS
implies nothing about `scientific_summary`, and vice versa.

---

## 17. `report.md`

Mandatory sections, in order:

1. **Frozen design** — question, population, AOIs, variants, estimands, orientations.
2. **Exact dates** — the full `window_dates.csv` content, all 12 AOI × variant rows.
3. **Cohort** — per-AOI counts, attrition breakdown, `cohort_hash` equality across variants.
4. **Folds** — per-AOI fold and block counts, positives per fold, `fold_mapping_hash`.
5. **Model contract** — estimator, features, hyper-parameters, seeds; explicit statement that
   `step8b.train_population` was called, not reimplemented.
6. **Bootstrap contract** — unit, replicates, seed, percentile method, pairing, no-refit.
7. **AOI-specific results** — one subsection per AOI. Manavgat appears only as reference.
8. **Descriptive four-region synthesis** — with an explicit statement that no pooled estimate
   exists and that PR-AUC and Brier are not comparable across AOIs.
9. **Evia different-regime framing** — the three mandated phrases verbatim, plus the
   prevalence table.
10. **MODIS support asymmetry** — the clipping table from `WINDOW_DATE_AUDIT.md` §4.1 and its
    interpretive consequence.
11. **Limitations** — all ≥ 11 entries.
12. **Technical PASS vs scientific support** — an explicit statement that these are separate
    verdicts.
13. **Provenance** — `analysis_id`, git commit, dependency lock hash, the four canonical AOI
    hashes, the Manavgat reference hash.

Wording QA: the union forbidden list (`SCIENTIFIC_CONTRACT.md` §12.2) plus
`FOREIGN_FACTOR_PHRASES`, applied to the rendered markdown **and** to every prose field of
every JSON artefact — matching how `assert_compare_wording` (line 9020) already walks
`_PROSE_FIELD_TOKENS`.

---

## 18. `stages/<stage>.json`

One per stage: `plan`, `export`, `local-downstream`, `cohort-feasibility`, `fit`, `compare`,
`summarize`.

| Key | Type | Constraint |
|---|---|---|
| `analysis_id`, `schema_version`, `stage` | string | `stage` ∈ the seven |
| `status` | string | `PASS` \| `FAIL` \| `SKIP` |
| `started_at_utc`, `finished_at_utc` | string | ISO-8601 Z |
| `config_hash` | string(64) | must equal `config.json`'s |
| `input_hashes` | object | role → sha256, at stage entry |
| `output_manifest` | array | `{relative_path, sha256, size_bytes}` |
| `frozen_hashes_before`, `frozen_hashes_after` | object | **must be equal** |
| `aois_processed` | array[string] | must be all three for an actual run |
| `variants_processed` | array[string] | must be all three |
| `gee_queries_run`, `gee_exports_run` | bool | `true` only for `export` |
| `model_fit` | bool | `true` only for `fit` (and `local-downstream`, for Step7C) |
| `bootstrap_run` | bool | `true` only for `compare` |
| `resume_eligible` | bool | derived, see §19 |
| `quarantined_paths` | array[string] | populated on `--force` |
| `failure_reason` | string, nullable | non-null ⟺ `status == "FAIL"` |

**Partial-set rule:** if `aois_processed` or `variants_processed` is not the complete set,
`status` may not be `PASS`.

---

## 19. Stage dependency and resume

```
plan → export → local-downstream → cohort-feasibility → fit → compare → summarize
```

| Stage | Inputs | Outputs | Success | Failure | Resume-eligible when |
|---|---|---|---|---|---|
| `plan` | registry, frozen inputs | `config.json`, `input_hashes.json`, `repository_inventory.json`, `window_dates.csv`, `export_plan.csv` | all date contracts pass; 6 frozen roles hash for 3 AOIs | any missing frozen input; hash drift | config + input hashes match |
| `export` | `export_plan.csv` | 138 predictor rasters + 3 pre-label rasters, `variant_artifact_index.csv` | every planned artefact present, > 0 bytes, alignment QA pass, hashed | missing/zero-byte/misaligned artefact | all artefact hashes match the index |
| `local-downstream` | exported rasters + static reuse | per-variant Step5/5C/7A–7E/8A trees | all 8 stages complete per variant; feature contract, static & label invariance pass | any stage raises | all output hashes match |
| `cohort-feasibility` | variant Step8A + censor raster | `cohort_inventory.csv`, `fold_mapping.parquet` | 14 checks pass for all 3 AOIs | any check fails for any AOI | cohort and fold hashes match |
| `fit` | cohort + folds | `metrics.csv`, `oof_predictions.parquet` | 90 logical fits, complete OOF coverage | duplicate/missing/unexpected fit | prediction hashes match |
| `compare` | OOF predictions | `bootstrap_replicates.parquet`, `bootstrap_summary.csv` | paired draws, counts reconcile, intervals reproduce | `valid + invalid != requested` | draw plan hash matches |
| `summarize` | all above | `regional_summary.csv`, `four_region_synthesis.csv`, `summary.json`, `report.md`, `manifest.json` | no pooled column; wording QA passes; manifest complete | any prohibited phrase or pooled column | never (cheap; always re-run) |

**Resume requires all five conditions simultaneously**, mirroring the existing
`*_is_reusable` helpers:

1. recorded stage `status == "PASS"`,
2. `config_hash` matches the current config,
3. every recorded input hash still matches on disk,
4. `output_manifest` is complete and every recorded hash still matches,
5. the outputs satisfy the current validator's structural checks.

**File existence alone is never sufficient.** A partial AOI or partial variant may never be
silently treated as PASS.

---

## 20. `manifest.json`

**Purpose:** the single top-level provenance record. **None exists today** in the Manavgat
namespace — this is a genuine addition.

| Key | Type | Constraint |
|---|---|---|
| `analysis_id` | string(64) | |
| `schema_version` | string | `multi_region_window_closure.v1` |
| `created_at_utc` | string | ISO-8601 Z |
| `git_commit`, `git_branch` | string | |
| `dirty_worktree` | bool | `true` ⇒ warning in `summary.json` |
| `python_version` | string | |
| `dependency_lock_hash` | string(64) | `sha256(requirements-lock.txt)` |
| `config_hash` | string(64) | `sha256(canonical_json(config.json))` |
| `input_hashes_hash` | string(64) | `sha256(canonical_json(input_hashes.json))` |
| `canonical_aoi_hashes` | object | the four Step8A digests |
| `manavgat_reference_hash` | string(64) | `054a1961…f3439` |
| `output_file_count` | int | `== len(files)` |
| `output_total_bytes` | int | `== Σ files[].size_bytes` |
| `files[]` | array | see below |

Each `files[]` entry: `relative_path` (unique, POSIX, inside the namespace), `size_bytes`
(> 0), `sha256` (64 hex), `stage` (∈ the seven), `required` (bool).

### 20.1 Self-hash resolution — stated explicitly

`manifest.json` **does not contain its own hash**. Its integrity is established externally:

* `manifest.json` is excluded from `files[]` by construction.
* Its digest is written to a sibling `manifest.sha256` (a bare 64-hex line) as the final act of
  the `summarize` stage.
* The validator recomputes `sha256(manifest.json)` and compares it to `manifest.sha256`.

This mirrors the existing `COMPARE_CONTROL_FILES` allow-list convention, whose comment
identifies the same hazard: listing a stage-control document in its own `output_artifacts`
"would create a self-hash cycle (the recorded digest is part of the file being digested)".
Exactly one file is exempt, by exact relative path.

### 20.2 Provenance chain

```
canonical AOI artefact (step8a_500m_modeling_dataset.parquet, hash-anchored)
  → temporal source (Landsat / MODIS collections)  |  static source (DEM, land cover)
  → export (variants/<v>/data/**, per-raster sha256)
  → local downstream feature artefact (step5 … step8a, per-file sha256)
  → cohort (cohort_hash)
  → fold mapping (fold_mapping_hash)
  → fit (90 logical fits)
  → OOF predictions (prediction_hash)
  → metric (metrics.csv, recomputable)
  → bootstrap comparison (draw_plan_hash → bootstrap_summary.csv)
  → regional_summary → four_region_synthesis → summary.json / report.md
  → manifest.json (+ manifest.sha256)
```

Every arrow is a recorded hash relationship, so any link can be re-verified independently.

---

## 21. `validation_report.json`

**New relative to the Manavgat precedent**, where validator output lives only in `logs/`.
Written **into** the namespace so validation is part of the artefact set.

| Key | Type | Constraint |
|---|---|---|
| `analysis_id`, `schema_version` | string | |
| `validated_at_utc` | string | ISO-8601 Z |
| `validator_version` | string | |
| `checks[]` | array | `{check_id, description, severity, status, evidence, failure_message}` |
| `technical_status`, `scientific_contract_status`, `namespace_safety_status` | string | `PASS` \| `FAIL` |
| `overall_status` | string | `PASS` \| `FAIL` |

`status` ∈ `{PASS, FAIL, SKIP}` only. `severity` ∈ `{required, advisory}`. A `required` check
with `status == "SKIP"` forces `overall_status = "FAIL"`.

The four-verdict structure matches the existing `Report` class
(`scripts/validate_window_closure_predictor_export.py:67`) and is deliberately not redesigned.
