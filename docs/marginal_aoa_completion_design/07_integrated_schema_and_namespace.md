# 07. Integrated Schema and Namespace

---

## 1. Separate diagnostics, one directed-pair table

The advisor asked for the three components to be produced together. That is not
the same as asking for them to be **combined into one score**, and this design
does not combine them.

**Primary recommendation:**

```
separate diagnostics, one directed-pair table
```

The three quantities are kept as distinct columns:

1. `weighted predictor-space dissimilarity` — directed, source-model-informed
2. `climatic distance` — symmetric, AOI-level
3. `geographic distance` — symmetric, AOI-level

A composite marginal index is **not** produced. It would require an exact
normalisation across three quantities on incomparable scales (a dimensionless
DI ratio, a reference-SD climate distance, and kilometres), a set of component
weights with no derivation available from four AOIs, and a scientific
justification for why those three things should be added at all. None of that
exists, and inventing it would convert three interpretable diagnostics into one
uninterpretable number.

If a composite is later wanted, it requires its own preregistration with exact
normalisation, exact weights and a stated rationale — added as a **new** schema
version, not retrofitted onto this one.

Note the structural mismatch that makes a naive composite actively misleading:
the weighted DI is **directed** (12 distinct values) while the other two are
**symmetric** (6 distinct values each, duplicated across the 12 rows). Averaging a
directed quantity with two symmetric ones would half-symmetrise the result and
destroy exactly the directionality that is the most striking finding in the
existing artifact.

---

## 2. Namespace

```
outputs/diagnostics/marginal_aoa_completion/<analysis_id>/
```

**The existing `outputs/diagnostics/marginal_area_of_applicability/` tree is not
written to, not moved and not modified.**

Note the difference from the existing convention: `marginal_aoa.v1` writes to a
fixed namespace with the `analysis_id` recorded inside the manifest. The proposal
here puts the `analysis_id` in the **path**, following the newer convention used
by `outputs/diagnostics/four_aoi_transfer_decomposition/<canonical_set_id>/`. This
makes a re-run with different inputs land in a different directory instead of
requiring a `--force` overwrite guard. Recorded as decision A-7; the recommendation
is the `analysis_id` path segment.

### Schema name

```
marginal_aoa_completion.v1
```

Assessed against alternatives:

- `marginal_aoa.v2` — **rejected.** It implies a replacement for
  `marginal_aoa.v1`, which it is not: the unweighted range-and-level diagnostic
  remains valid, remains fully label-blind in both directions, and remains the
  authoritative answer to a different question. A `v2` name would invite the
  advisor to treat the older numbers as superseded.
- `weighted_aoa.v1` — rejected as too narrow; the artifact also carries the two
  distance components.
- `marginal_aoa_completion.v1` — **adopted.** It states exactly what it is: the
  completion of the three missing components of advisor item 3.

---

## 3. File layout

```
outputs/diagnostics/marginal_aoa_completion/<analysis_id>/
│
├── config/
│   ├── preregistration.json              frozen before any computation
│   ├── frozen_input_inventory.json       4 Step8A paths + hashes + row counts
│   ├── feature_importance_inventory.json 4 importance CSVs + hashes + method
│   ├── climate_input_inventory.json      collection, version, period, bands, hashes
│   └── geometry_inventory.json           4 bboxes, centres, contract hashes
│
├── weighted_predictor_space/
│   ├── source_feature_weights.csv        4 AOIs × 10 features
│   ├── directed_pair_summary.csv         12 rows — the main table
│   ├── target_cell_dissimilarity.parquet 260 322 rows (see §5)
│   └── source_threshold_diagnostics.csv  4 rows, one per source AOI
│
├── climate_distance/
│   ├── aoi_climate_vectors.csv           4 rows × 4 variables
│   └── pairwise_climate_distance.csv     6 unordered rows
│
├── geographic_distance/
│   ├── aoi_geometry_summary.csv          4 rows
│   └── pairwise_geographic_distance.csv  6 unordered rows
│
├── comparison/                           POST-ANALYSIS LAYER — written last
│   ├── marginal_diagnostics_with_transfer.csv
│   ├── ranking_summary.csv
│   └── scientific_summary.md
│
└── completion_metadata.json              hashes every file above
```

**This directory was not created.** It is a proposal.

Writing order is a contract, not a convenience: `config/` first, then the three
component directories, then `comparison/` last, then `completion_metadata.json`.
The comparison layer physically cannot influence the components because it does
not exist when they are computed.

---

## 4. The main table: `directed_pair_summary.csv`

**Exactly 12 rows.** One per directed pair, `itertools.permutations` over the four
sorted experiment IDs. No diagonal rows, no duplicates, no unordered token.

### Identity and provenance

| Column | Notes |
|---|---|
| `schema_version` | `marginal_aoa_completion.v1` |
| `analysis_id` | sha256 of the canonicalised scientific config |
| `source_experiment` | |
| `target_experiment` | |
| `direction` | `f"{source}_to_{target}"`, matching `marginal_aoa.v1` |
| `pair_token` | `f"{source}__{target}"`, never sorted |
| `primary_population` | `burnable_tree_shrub_grass` |
| `source_step8a_sha256` | pinned |
| `target_step8a_sha256` | pinned |
| `source_importance_sha256` | pinned |
| `git_commit` | |
| `created_at_utc` | |

### Population counts

| Column | Notes |
|---|---|
| `source_rows` | source primary-population rows |
| `target_rows` | target primary-population rows |
| `source_reference_rows` | after missing-predictor exclusion |
| `source_rows_excluded_missing` | |
| `target_rows_assessable` | |
| `target_rows_not_assessable` | |

### Weighted predictor-space dissimilarity

| Column |
|---|
| `importance_method` |
| `importance_population` |
| `importance_model` |
| `n_features_with_positive_weight` |
| `effective_feature_count_perplexity` |
| `feature_weight_entropy` |
| `zero_weight_features` |
| `constant_feature_guard_features` |
| `numeric_scaling_method` |
| `weighted_distance_formula_id` |
| `categorical_policy_id` |
| `source_pairwise_mean_distance` — mean weighted distance over all distinct source reference cell pairs |
| `source_distance_normaliser` — equals `source_pairwise_mean_distance` |
| `normaliser_method` = `"source_pairwise_mean_distance_v1"` |
| `normaliser_uses_folds` = `false` |
| `training_di_upper_whisker_threshold` — **the operative threshold** |
| `primary_threshold_method` = `"source_spatial_fold_holdout_di_upper_whisker_v1"` |
| `training_di_q95_threshold` — secondary, reported, never operative |
| `training_di_q95_method` = `"source_spatial_fold_holdout_di_q95_v1"` |
| `training_di_q50_threshold` / `_q90_` / `_q99_` / `_max_threshold` |
| `training_di_q1` / `training_di_q3` / `training_di_iqr` |
| `target_mean_dissimilarity` |
| `target_median_dissimilarity` |
| `target_p90_dissimilarity` |
| `target_p95_dissimilarity` |
| `target_max_dissimilarity` |
| `fraction_target_cells_inside_weighted_aoa` |
| `fraction_target_cells_outside_weighted_aoa` |
| `fraction_target_cells_not_assessable` |
| `top_weighted_mismatch_features` (JSON list) |
| `target_cells_with_unseen_level` |
| `fraction_target_cells_with_unseen_level` |

### Climate distance (echoed from the symmetric table)

`climate_distance`, `climate_distance_metric`, `climate_features`,
`climate_reference_period`, `climate_season_months`, `climate_scaling_contract`,
`climate_source_version`, `climate_land_mask`, `climate_band_scale_factors`,
`climate_data_completeness`, `climate_component_contributions`,
`climate_export_authorised`, `climate_status`, `climate_uncertainty`.

Fixed values:

```
climate_feature_count      = 4
climate_features           = ["annual_mean_temperature_c",
                              "annual_precipitation_mm",
                              "warm_season_climatic_water_deficit_mm",
                              "warm_season_vpd_kpa"]
climate_reference_period   = "1991-01-01/2020-12-31"
climate_season_months      = [6, 7, 8, 9]
climate_source_version     = "IDAHO_EPSCOR/TERRACLIMATE" + resolved asset version
climate_land_mask          = "terraclimate_native_valid_land_support"
climate_export_authorised  = true
```

The export is authorised but not yet run: until it completes,
`climate_status = "authorised_pending_export"` and the numeric fields are `null`.

### Geographic distance (echoed from the symmetric table)

`source_centroid_lon`, `source_centroid_lat`, `target_centroid_lon`,
`target_centroid_lat`, `centroid_geodesic_distance_km`,
`optional_minimum_boundary_distance_km`, `geographic_distance_method`,
`centroid_definition`, `geodesic_implementation`, `geodesic_package_version`,
`geometry_source_path`, `geometry_source_sha256`,
`source_geometry_contract_hash`, `target_geometry_contract_hash`, `source_bbox`,
`target_bbox`, `geographic_component_reads_step8a`,
`population_centroid_reported`.

Fixed values:

```
centroid_definition               = "bbox_centre_planar_epsg4326"
geodesic_implementation           = "geographiclib_wgs84"
geographic_component_reads_step8a = false
population_centroid_reported      = false
```

**No population-centroid columns appear.** The geographic component reads no
Step8A data of any kind (doc 06 §3).

### Original unweighted support (read-only echo from `marginal_aoa.v1`)

| Column | Source |
|---|---|
| `unweighted_analysis_id` | `4a5b8c80…` |
| `unweighted_pair_analysis_id` | per-pair, from the comparison manifest |
| `unweighted_support_definition_id` | `source_observed_range_and_levels_v1` |
| `unweighted_fraction_target_cells_inside_support` | |
| `unweighted_fraction_target_cells_outside_support` | |
| `unweighted_fraction_target_cells_not_assessable` | |
| `unweighted_maximum_feature_fraction_outside` | |
| `unweighted_top_outside_support_features` | |
| `unweighted_fraction_target_unseen_level` | |
| `unweighted_categorical_sidecar_path` | |

These are copies for joining. The authoritative values stay in
`marginal_aoa.v1`, and validator check 20 asserts both that the echo matches and
that the original files are byte-identical before and after the run.

### Firewall flags

| Column | Required value |
|---|---|
| `target_label_used` | `false` |
| `target_burn_date_used` | `false` |
| `target_transfer_metric_used` | `false` |
| `source_label_used` | **`true`** |
| `source_label_read_directly_by_completion_module` | `false` |
| `diagnostic_class` | `"target_label_blind_source_model_informed"` |
| `model_fitted` | `false` |
| `gee_query_issued` | `false` (`true` only in the separate climate export artifact) |

---

## 5. `target_cell_dissimilarity.parquet`

One row per (directed pair × target cell). Each of the four AOIs appears as the
target of exactly 3 sources, so the exact expected row count is:

```
3 × (20555 + 15190 + 41731 + 9298)  =  3 × 86 774  =  260 322 rows
      manavgat   bejis    mugla    evia_ext
```

Validator check: the row count must equal 260 322 exactly, and the per-pair
group sizes must equal the `target_rows` column of the main table.

Columns:

```
source_experiment, target_experiment, row_500m, col_500m,
weighted_dissimilarity,            (null when not assessable)
nearest_source_row_500m, nearest_source_col_500m,
categorical_mismatch_at_nearest,   (bool)
n_missing_predictors,
cell_weighted_aoa_status           ∈ {inside_weighted_aoa,
                                      outside_weighted_aoa,
                                      not_assessable}
```

`(source_experiment, target_experiment, row_500m, col_500m)` is the exact join key
onto the existing `marginal_aoa_target_cells.parquet`, so a per-cell comparison
of marginal support against joint dissimilarity is possible without recomputing
anything.

Status vocabulary deliberately mirrors `marginal_aoa.v1`'s
`inside_support` / `outside_support` / `not_assessable` triple, with
`_weighted_aoa` suffixes so the two can never be confused in a join.

---

## 6. `source_feature_weights.csv`

4 rows per source AOI × 10 contract features = 40 rows:

```
source_experiment, feature, feature_kind, raw_importance,
n_dummy_levels_summed, dummy_level_contributions, weight,
renormalisation_factor, is_zero_weight, constant_feature_guard_used,
source_mean, source_scale, source_scale_method
```

`dummy_level_contributions` is a JSON map, populated only for
`landcover_dominant`, recording the per-level importances that were summed. This
is what makes the group-normalisation auditable.

---

## 7. `source_threshold_diagnostics.csv`

4 rows, one per source AOI:

```
source_experiment, source_reference_rows, source_rows_excluded_missing,

source_pairwise_mean_distance, source_distance_normaliser,
normaliser_method, normaliser_uses_folds, n_distinct_source_pairs,

holdout_block_size_cells, holdout_fold_count, fold_assignment_method,
fold_assignment_reads_label, n_blocks, min_block_size, max_block_size,

training_di_q1, training_di_q3, training_di_iqr,
training_di_upper_whisker_threshold, primary_threshold_method,
training_di_max_threshold, whisker_clamped_to_max,
training_di_q50_threshold, training_di_q90_threshold,
training_di_q95_threshold, training_di_q95_method, training_di_q99_threshold
```

Two fields carry the design's two most load-bearing claims:

- `normaliser_uses_folds` must be `false` — the normaliser is the mean pairwise
  source distance and is fold-independent by construction.
- `fold_assignment_reads_label` must be `false` — the folds that build the
  training DI and the threshold did not inherit Step8B's label-informed
  `StratifiedGroupKFold` assignment.

`whisker_clamped_to_max` records whether `min(max(training_DI), Q3 + 1.5·IQR)`
selected the maximum rather than the whisker, so the clamp is visible rather than
inferred.

---

## 8. The comparison layer

Written **after** every component artifact exists and is hashed. Separate
directory, separate provenance record, separate metadata block.

`comparison/marginal_diagnostics_with_transfer.csv` — 12 rows, joining the main
table to the frozen transfer artifact:

```
source     outputs/diagnostics/four_aoi_transfer_decomposition/
             bejis_2022__evia_2021_extended__manavgat_2021__mugla_2021/
             four_aoi_decomposition.csv
sha256     6b071b3ef7e93e0ae9d889ccfa98b852f9fc531b110df59e66f460ff2392c0d9
```

That file has 96 rows = 12 directions × 2 model families × 2 adaptation methods ×
2 metrics. A **preregistered selection** collapses it to 12:

```
PRIMARY RANKING COMPARISON

    model_family    = "thermal"
    transfer_state  = "raw"          <- untransformed transfer, NOT an adaptation
    metric          = "roc_auc"
```

**The primary transfer ordering is raw thermal ROC-AUC.** `raw_auc` is the
model's untransformed cross-region performance, and it is the quantity an
applicability diagnostic should be compared against: the AoA describes how far
the target is from the source's support, before any adaptation is applied.

**`regionwise_zscore` ROC-AUC must not be the primary ordering.** An adapted
metric measures performance *after* an alignment step that itself removes part of
the distribution shift the AoA is quantifying. Ranking a shift diagnostic against
a shift-corrected metric would partially cancel the very effect under study, and
would make a weak association look like evidence about the diagnostic rather than
about the adaptation. `raw_auc` is read directly from the decomposition table and
is identical across both `adaptation_method` rows, so the selection is
unambiguous.

`thermal` is chosen because it is the model family the AoA's feature contract
belongs to (`SHARED_THERMAL_MODEL_FEATURES`).

### Secondary comparison block — complete, always reported together

```
raw thermal PR-AUC
thermal ROC-AUC transfer gap        (raw_gap)
thermal PR-AUC transfer gap
adapted thermal ROC-AUC and PR-AUC, for EVERY preregistered adaptation
    (regionwise_zscore, coral_after_regionwise_zscore)
recovered_fraction
```

The secondary block is reported **in full, every time**. Selecting a comparison
after seeing which association is largest is forbidden, and reporting the whole
block is what makes that impossible rather than merely discouraged.

Columns carried across: `within_target_auc`, `raw_auc`, `adapted_auc`, `raw_gap`,
`adaptation_effect`, `remaining_gap`, `recovered_fraction`, its CI bounds,
`chance_level`, `recovery_status`.

### `comparison/ranking_summary.csv`

Descriptive rank associations only, per transfer metric:

```
diagnostic ∈ {weighted_di_mean, weighted_di_p95,
              fraction_inside_weighted_aoa,
              climate_distance, centroid_geodesic_distance_km,
              unweighted_fraction_inside_support}

transfer   ∈ {raw_thermal_roc_auc,        <- PRIMARY
              raw_thermal_pr_auc,
              thermal_roc_auc_gap,
              thermal_pr_auc_gap,
              adapted_thermal_roc_auc_regionwise_zscore,
              adapted_thermal_pr_auc_regionwise_zscore,
              adapted_thermal_roc_auc_coral,
              adapted_thermal_pr_auc_coral,
              recovered_fraction}

is_primary_comparison  (bool; true only for raw_thermal_roc_auc)
spearman_rho, kendall_tau, n_pairs,
top3_overlap_count, bottom3_overlap_count,
diagnostic_is_directed  (bool)
```

An **ordered-pair ranking table** accompanies the correlations: the 12 directed
pairs listed in diagnostic order beside their raw-transfer order, so the reader
can see where the two orderings agree and where they do not without relying on a
single coefficient.

**No p-value, no hypothesis test, no confidence interval on any correlation.**
Twelve directed pairs from four non-independent AOIs cannot support an inferential
claim, and computing a p-value would invite one.

`diagnostic_is_directed` matters: `climate_distance` and
`centroid_geodesic_distance_km` take only 6 distinct values across the 12 rows, so
their rank correlation against a directed transfer metric is structurally
attenuated. That must be stated in the same table, not discovered later.

Two additional preregistered descriptive views:

- **Bejís-source direction audit.** The three worst-supported directions in the
  unweighted diagnostic are all Bejís-source and all driven by `elevation_mean`,
  which is also Bejís's top-weighted feature (0.334). Whether weighting amplifies
  or offsets that is a stated expectation to check, not a discovery.
- **Directional versus symmetric split.** Report the rank association separately
  for the directed diagnostic and for the two symmetric ones, and never pool them
  into a single "the diagnostics rank transfer" claim.

### Permitted and forbidden wording

Permitted:

> In this four-AOI set, the diagnostic ordering does or does not reproduce the
> observed **raw-transfer** ordering.

Forbidden:

> Marginal diagnostics can never rank transfer.

The second is a general claim about a method; 12 directed pairs from four AOIs in
one season cannot establish it. The existing advisor document already adopted the
softer formulation for the unweighted diagnostic (ρ = +0.077) and the same
constraint carries over verbatim.

---

## 9. `completion_metadata.json`

```json
{
  "schema_version": "marginal_aoa_completion.v1",
  "analysis_id": "<sha256 of canonicalised scientific config>",
  "created_at_utc": "...",
  "git_commit": "...",
  "package_versions": { "numpy": "...", "pandas": "...", "scikit-learn": "...", "scipy": "...", "pyarrow": "..." },
  "experiments": ["bejis_2022", "evia_2021_extended", "manavgat_2021", "mugla_2021"],
  "primary_population": "burnable_tree_shrub_grass",
  "pair_cardinality": 12,
  "canonical_step8a_hashes": { "...": "..." },
  "source_importance_hashes": { "...": "..." },
  "geometry_source": { "path": "core/regions.py", "sha256": "980eb5d4...", "commit": "0a3c5fe8..." },
  "climate_source": null,
  "target_label_firewall": { "...": false },
  "source_label_policy": { "source_label_used": true, "...": "..." },
  "model_fitted": false,
  "gee_query_issued": false,
  "bootstrap_performed": false,
  "output_sha256": { "<relative path>": "<sha256>", "...": "..." },
  "original_marginal_aoa_v1_hashes_before": { "...": "..." },
  "original_marginal_aoa_v1_hashes_after":  { "...": "..." },
  "limitations": [ "..." ]
}
```

`original_marginal_aoa_v1_hashes_before` / `_after` are recorded around the run and
must be identical. That is validator check 20 and it is a positive proof of
non-mutation, not merely an absence of a write call.

`output_sha256` must cover **every** file in the namespace including the
`comparison/` directory — validator check 25.

---

## 10. Uncertainty

> **Decision B-11 is RESOLVED: the first production run is point-estimate-only.**

```
uncertainty_policy = "point_estimate_only"
bootstrap_performed = false
```

Every weighted-AoA quantity is descriptive, matching `marginal_aoa.v1`, whose 12
evidence rows all carry `descriptive_no_interval`. This keeps the two artifacts
directly comparable and is the simplest honest first result.

**Climate and geographic distances receive no interval at all**, in this or any
later run:

```
climate_distance_uncertainty     = "deterministic_aoi_level_value_no_interval"
geographic_distance_uncertainty  = "deterministic_aoi_level_value_no_interval"
```

They are deterministic AOI-level values computed from fixed geometry and a fixed
climatology. There is no resampling unit, and manufacturing one would produce a
fabricated interval.

### Deferred: target spatial-block bootstrap

A target spatial-block bootstrap **may be added later, and only through a
separately preregistered sensitivity analysis** — never as an unannounced
addition to this run. The contract below is recorded now so that, if it is ever
run, it is run as specified rather than designed after the fact.

```
block_source                outputs Step8A row_500m/col_500m via add_spatial_block_id()
block_size_cells            10  (≈5 km, same scale as the source fold blocks)
requested_replicates        1000        (STEP8C_N_BOOTSTRAP)
seed                        42          (STEP8C_RANDOM_SEED)
resampled_unit              target spatial blocks, sampled with replacement
source_reference_held_fixed TRUE  — never resampled
feature_weights_held_fixed  TRUE  — never recomputed per replicate
threshold_held_fixed        TRUE  — never recomputed per replicate
normaliser_held_fixed       TRUE  — never recomputed per replicate
target_cells_only_resampled TRUE
invalid_replicate_policy    a replicate with zero assessable target cells is
                            excluded and counted; never silently retried
ci_method                   percentile, 2.5 / 97.5   (STEP8C_CI_LOWER/UPPER)
reported                    n_replicates_requested, n_replicates_valid,
                            n_replicates_invalid
minimum_valid               900         (MIN_VALID_BOOTSTRAP convention)
bootstrapped_quantities     target_mean_dissimilarity, target_median_dissimilarity,
                            target_p90/p95_dissimilarity,
                            fraction_target_cells_inside_weighted_aoa,
                            fraction_target_cells_outside_weighted_aoa
```

Holding the source side fixed is the point: the bootstrap would answer "how
stable is this target's dissimilarity summary under target spatial resampling",
not "how uncertain is the whole pipeline". Resampling the source too would
conflate the two and would change the normaliser and the threshold under every
replicate.

**No bootstrap is part of the completion run, and none was run in this task.**
