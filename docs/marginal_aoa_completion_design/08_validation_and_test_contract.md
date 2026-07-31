# 08. Validation and Test Contract

Two independent layers:

- **Validator** — runs against the produced artifact, after the analysis. Answers
  "is this artifact what it claims to be?"
- **Tests** — run against the implementation, on synthetic fixtures, without any
  real experiment. Answer "does the code do what the contract says?"

Neither was written or executed in this task.

---

## Part 1 — Validator contract

**27 checks** — the original 25 plus **13b** (the primary classification used the
upper whisker) and **22b** (the primary transfer comparison is raw thermal
ROC-AUC), both added when the threshold and comparison contracts were finalised.
Every one must be machine-checkable against the artifact plus the
repository, with no human judgement. Each writes a
`{check_id, status, expected, observed, evidence_path}` record to a
`validation_report.json`, and any `FAIL` aborts publication.

### Provenance and cardinality

**1. Canonical Step8A hashes match.**
Recompute `sha256` of all four
`outputs/experiments/<exp>/step8a/step8a_500m_modeling_dataset.parquet` and
compare against `completion_metadata.json:canonical_step8a_hashes` **and** against
the four literals:

```
manavgat_2021        054a1961fc0582a33d36413263668b63074b21ae8b03d12269b6e228787f3439
bejis_2022           3dec785a7d8e31db2d67ed283546bbfbca1559f56df46663488d0afc24d9e393
mugla_2021           c4ab107db2207f9f20775ccc0b3bf39381173fd07d4e82f6821ce7f40be7db8e
evia_2021_extended   bdce859cf482f575d0f273174b157f47efd61779953fdd23d9486c5face5e553
```

**2. Exactly 12 directed pairs.**
`directed_pair_summary.csv` has 12 rows, and the set of
`(source_experiment, target_experiment)` equals
`set(permutations(sorted(experiments), 2))`.

**3. No duplicate pair.**
`(source_experiment, target_experiment)` is unique across the 12 rows. Also
checked in `target_cell_dissimilarity.parquet` on
`(source, target, row_500m, col_500m)`.

**7. No diagonal or self pairs.**
No row has `source_experiment == target_experiment`, in any output file.

### Directionality and symmetry

**4. Weighted DI is directional.**
For at least one unordered pair,
`target_mean_dissimilarity(A→B) != target_mean_dissimilarity(B→A)`. Additionally,
`source_pairwise_mean_distance`, `source_distance_normaliser` and
`training_di_upper_whisker_threshold` must depend only on the source: for a fixed
source, all three of its rows carry identical values. And no column in the
artifact is a sorted or unordered pair token.

**5. Climate distance is symmetric.**
For every unordered pair, `climate_distance(A→B) == climate_distance(B→A)`
exactly (bitwise float equality, not within a tolerance — the value is copied
from the same symmetric table). Skipped with status `SKIPPED_PENDING_EXPORT` while
`climate_status == "authorised_pending_export"`, and the skip is recorded, never
silently passed.

**6. Geographic distance is symmetric.**
`centroid_geodesic_distance_km(A→B) == centroid_geodesic_distance_km(B→A)`
exactly. Same for `optional_minimum_boundary_distance_km`. Additionally
`geodesic_implementation == "geographiclib_wgs84"`,
`geographic_component_reads_step8a == false`,
`population_centroid_reported == false`, and no column matching
`*population_centroid*` exists in any output file.

### Label firewall

**8. Target labels were not read.**
Three independent pieces of evidence, all required:

- `target_label_used`, `target_burn_date_used`, `target_transfer_metric_used` are
  all `false` in every row and in `completion_metadata.json`.
- No output file contains a column named `burned`, `burn_date`, `burn_month`,
  `burn_day_of_year`, `label_source`, or matching `y_prob_*`.
- The implementation's parquet read sites are inspected: every
  `pd.read_parquet` / `pq.read_table` call passes an explicit `columns=`
  allow-list, and the union of all allow-lists contains no forbidden column.
  This mirrors `test_burned_never_enters_the_parquet_column_list` in the existing
  test suite.

**9. Source importance provenance is recorded.**
`feature_importance_inventory.json` names, per source AOI: the CSV path, its
`sha256`, `population == "burnable_tree_shrub_grass"`,
`model == "thermal"`, `importance_method == "impurity_gini_in_sample_whole_population_v1"`,
`model_algorithm == "RandomForestClassifier"`. The recorded hashes must match the
files on disk. `source_label_used` must be `true` and
`source_label_read_directly_by_completion_module` must be `false`.

**22. Transfer metrics appear only in the comparison layer.**
No file outside `comparison/` contains any of `within_target_auc`, `raw_auc`,
`adapted_auc`, `raw_gap`, `adaptation_effect`, `remaining_gap`,
`recovered_fraction`, `recovery_status`, `chance_level`. And
`completion_metadata.json:output_sha256` shows every `comparison/` file with an
`mtime` later than every component file.

### Weights

**10. Feature weights are finite, non-negative and sum to 1.**
For each source AOI in `source_feature_weights.csv`: 10 rows, every `weight`
finite and `>= 0`, `abs(sum(weight) - 1.0) <= 1e-9`.

**11. Zero-importance and zero-variance policy is truthful.**
`zero_weight_features` equals the actual set `{f : weight == 0}` — not asserted
empty, but asserted **equal to what was observed**. Same for
`constant_feature_guard_features` against `{f : source_scale_method == "constant_guard"}`.
Both are expected empty for these four AOIs (doc 02 §2, doc 03 §3.4), but the
check compares rather than assumes.

**12. Categorical handling is truthful.**
`categorical_policy_id == "weighted_mismatch_penalty_gower_v1"`. The landcover
`weight` equals the sum of `dummy_level_contributions`. The number of dummy levels
summed equals the observed level count for that source AOI (7 for Manavgat and
Bejís, 8 for Muğla and Evia). No output column encodes `landcover_dominant` as a
numeric magnitude.

### Threshold and counts

**13. The normaliser and threshold are source-only, and are the specified ones.**
`source_threshold_diagnostics.csv` has exactly 4 rows, one per source AOI.

Normaliser:
- `normaliser_method == "source_pairwise_mean_distance_v1"`,
- `normaliser_uses_folds == false`,
- `source_distance_normaliser == source_pairwise_mean_distance` exactly,
- `n_distinct_source_pairs == source_reference_rows * (source_reference_rows - 1) / 2`,
  which proves self-distance was excluded and every distinct pair was counted once.
- **No field anywhere describes the normaliser as a holdout nearest-neighbour
  mean.** A validator string scan over all outputs must find no
  `holdout`-derived normaliser token.

Threshold:
- `primary_threshold_method == "source_spatial_fold_holdout_di_upper_whisker_v1"`,
- `training_di_upper_whisker_threshold == min(training_di_max_threshold,
  training_di_q3 + 1.5 * training_di_iqr)`, recomputed by the validator from the
  stored `q3`/`iqr`/`max` fields and required to agree to 1e-12,
- `whisker_clamped_to_max` truthfully records which branch of the `min` won,
- `training_di_q95_threshold` is present and `training_di_q95_method ==
  "source_spatial_fold_holdout_di_q95_v1"`.

Both are identical across all three directed rows sharing a source. A source-only
quantity cannot vary with the target; if it does, target data leaked in.
Additionally `fold_assignment_reads_label` must be `false` and
`fold_assignment_method` must name a label-free rule.

**13b. The primary classification used the upper whisker.**
`fraction_target_cells_inside_weighted_aoa` must equal
`#{x assessable : DI(x) <= training_di_upper_whisker_threshold} / target_n_total`,
recomputed by the validator from `target_cell_dissimilarity.parquet`. Recomputing
the same fraction against `training_di_q95_threshold` must give a **different**
value unless the two thresholds coincide numerically — this is the check that
catches a silent substitution of the secondary threshold for the primary.

**14. Target cell counts and missingness are truthful.**
Per row: `target_rows_assessable + target_rows_not_assessable == target_rows`;
`target_rows` matches the frozen population counts (20555 / 15190 / 41731 / 9298);
the three fractions sum to `1.0` within 1e-12; and the per-pair group sizes in
`target_cell_dissimilarity.parquet` match `target_rows`, with a total of
260 322 rows.

### Climate

**15. Climate variables come from the preregistered source and version.**
`climate_input_inventory.json` names the collection ID, the resolved asset
version, the reference period, the band list, the season months, the band scale
factors **as actually applied**, the land-support rule and the `sha256` of every
exported raster. These must match `preregistration.json`. Required exact values:

```
climate_source_version     starts with "IDAHO_EPSCOR/TERRACLIMATE"
climate_reference_period   == "1991-01-01/2020-12-31"
climate_season_months      == [6, 7, 8, 9]
climate_feature_count      == 4
climate_features           == ["annual_mean_temperature_c",
                               "annual_precipitation_mm",
                               "warm_season_climatic_water_deficit_mm",
                               "warm_season_vpd_kpa"]
climate_land_mask          == "terraclimate_native_valid_land_support"
climate_export_authorised  == true
```

Exactly four variables — the check fails if `warm_season_mean_temperature` or
`warm_season_precipitation` appears, since both were removed as duplicate axes.
No ERA5-Land field may appear in the initial run.

`climate_component_contributions` must have four entries and sum to
`climate_distance²` within 1e-12.

**16. No event-period predictor is used as a climate normal.**
A **path-provenance** assertion, not a naming convention: every input path
recorded for any `climate_*` field must lie under the authorised climate export
directory. Any path under
`outputs/experiments/*/step5/`, `outputs/experiments/*/step5c/`,
`outputs/experiments/*/data/modis/`, `outputs/experiments/*/data/ndvi_timeseries/`,
`outputs/experiments/*/data/landsat_timeseries/` or
`outputs/experiments/*/data/current_period/` is an immediate `FAIL`. This cannot
be bypassed by renaming a column.

### Geometry

**17. The geometry source is canonical.**
`geometry_inventory.json` records `path = "core/regions.py"` with
`sha256 = 980eb5d4cf459ee52bf065f3b2fb2d644fb72449d62c0f8dfba1c58c93396275`,
recomputed at validation time. The four bboxes must equal:

```
bejis_2022           (-1.05, 39.68, -0.35, 40.15)
evia_2021_extended   (23.05, 38.55, 23.85, 39.15)
manavgat_2021        (31.05, 36.72, 31.85, 37.35)
mugla_2021           (27.10, 36.60, 28.90, 37.45)
```

Explicitly assert that `evia_2021_extended` resolved to the **extended** bbox and
not to `evia_2021`'s `(23.12, 38.68, 23.52, 39.08)`.

**18. Kilometre values are recomputable by the stated method.**
The validator independently recomputes `centroid_geodesic_distance_km` from the
`source_bbox` and `target_bbox` columns stored on each row, using the method named
in `geographic_distance_method`, and requires agreement to ≤ 1e-6 km. This is why
the raw bboxes are stored on every row.

### Isolation

**19. Output lives only in the isolated namespace.**
Every path in `output_sha256` starts with
`outputs/diagnostics/marginal_aoa_completion/<analysis_id>/`. A filesystem walk
of the repository confirms no file outside that prefix was created or modified
during the run, comparing a before/after inventory of `(path, mtime, size)`.

**20. `marginal_aoa.v1` artifacts are unchanged.**
`original_marginal_aoa_v1_hashes_before` and `_after` cover all 13 manifests plus
every `marginal_aoa_*` file in the tree, and must be identical maps. Positive
proof, not merely absence of a write.

**21. Canonical Step8A artifacts are unchanged.**
Same before/after hash comparison for the four Step8A parquets. This is check 1
repeated as a mutation guard rather than an identity guard, and it must be run
after the analysis, not only before.

### Truthfulness of the "did not do" flags

**22b. The primary transfer comparison is raw thermal ROC-AUC.**
In `comparison/ranking_summary.csv`, exactly one row set carries
`is_primary_comparison == true`, and for it
`transfer == "raw_thermal_roc_auc"`. The selection triple recorded in
`preregistration.json` must be `model_family="thermal"`,
`transfer_state="raw"`, `metric="roc_auc"`. An adapted metric
(`regionwise_zscore` or `coral_after_regionwise_zscore`) marked as primary is an
immediate `FAIL`. The complete secondary block must be present in full — all
nine transfer quantities listed in doc 07 §8 — so no subset can have been
selected after inspection.

**23. `model_fitted == false` is truthful.**
The implementation module imports no estimator: a static check that
`sklearn.ensemble`, `sklearn.linear_model`, `xgboost`, and `.fit(` do not appear
in it. `sklearn.neighbors` **is** permitted and must be explicitly allow-listed —
it is a data structure, not an estimator. This mirrors
`test_implementation_imports_no_model_or_bootstrap_machinery` in the existing
suite, with the neighbours exemption added and documented.

**24. `gee_query_issued == false` is truthful.**
The implementation imports no `ee` module and issues no network call. The climate
export, when authorised, is a **separate** script with its own metadata and its
own `gee_query_issued = true`; the two must never share a module.

**25. Metadata binds every output hash.**
`output_sha256` covers every file in the namespace, including `comparison/` and
`config/`, with no file present on disk but absent from the map and none absent
from disk but present in the map. Each recorded hash is recomputed and compared.

---

## Part 2 — Test contract

Synthetic fixtures only. No test may read a real experiment, and none may write
outside `tmp_path`. This mirrors the existing
`tests/test_marginal_area_of_applicability.py` conventions, including
`test_no_real_experiment_id_is_hard_coded_in_the_implementation`.

### Distance algebra

| Test | Assertion |
|---|---|
| `test_identical_vectors_give_zero_dissimilarity` | A target cell whose predictors and landcover level equal a source reference cell exactly has `weighted_dissimilarity == 0.0`. |
| `test_larger_weighted_separation_gives_larger_di` | With a fixed weight vector, moving a target cell further along a positively-weighted axis strictly increases DI. |
| `test_zero_weight_feature_cannot_affect_di` | Setting `w_j = 0` and then varying feature `j` arbitrarily — including to extreme values — leaves DI bitwise unchanged. |
| `test_weighted_distance_matches_closed_form` | On a 3-cell fixture, the computed DI equals a hand-written `sqrt(Σ w (Δz)² + w_lc·mismatch)` evaluation. |
| `test_two_query_partition_equals_brute_force` | The per-level KDTree route and the chunked brute-force route agree to ≤ 1e-12 on a randomised fixture. |
| `test_di_is_invariant_to_source_row_order` | Shuffling the source reference rows leaves every DI bitwise unchanged. |

### Normaliser

| Test | Assertion |
|---|---|
| `test_normaliser_equals_mean_pairwise_distance` | On a small fixture the computed `source_pairwise_mean_distance` equals a hand-written mean over all `n(n-1)/2` distinct pairs. |
| `test_normaliser_two_forms_agree` | The all-distinct-pairs form and the mean-of-per-cell-means form agree to ≤ 1e-12. |
| `test_normaliser_excludes_self_distance` | Adding an exact duplicate of an existing source cell changes the normaliser in the way the pairwise formula predicts, and no `d(s,s)=0` term is ever included. |
| `test_normaliser_includes_categorical_term` | Changing one source cell's landcover level, holding numerics fixed, changes the normaliser. |
| `test_normaliser_ignores_folds` | Changing `block_size_cells` or the fold count leaves `source_distance_normaliser` bitwise unchanged. This is the test that pins the correction away from the old holdout-NN definition. |
| `test_normaliser_is_target_independent` | The three directed pairs sharing a source carry a bitwise-identical normaliser; perturbing any target frame does not change it. |
| `test_normaliser_chunking_is_exact` | Two different chunk sizes give bitwise-identical results. |

### Training DI and threshold

| Test | Assertion |
|---|---|
| `test_training_di_uses_pairwise_normaliser` | `training_DI = holdout_nearest_distance / source_pairwise_mean_distance`; the denominator is the pairwise mean, not a holdout statistic. |
| `test_upper_whisker_formula` | `threshold == min(max(training_DI), Q3 + 1.5*IQR)` on a fixture with a known quartile structure. |
| `test_whisker_clamps_to_max_when_compact` | A compact training-DI distribution whose whisker exceeds the observed maximum yields `threshold == max(training_DI)` and `whisker_clamped_to_max == true`. |
| `test_whisker_below_max_when_tailed` | A tailed distribution yields `threshold < max(training_DI)` and `whisker_clamped_to_max == false`. |
| `test_primary_classification_uses_whisker_not_q95` | On a fixture where the whisker and q95 differ, `fraction_target_cells_inside_weighted_aoa` matches the whisker result and not the q95 result. |
| `test_q95_is_reported_but_not_operative` | `training_di_q95_threshold` is present, and no classification field is derived from it. |
| `test_blocks_are_not_split_across_folds` | Every spatial block appears in exactly one fold. |
| `test_fold_assignment_is_label_free_and_deterministic` | Fold assignment reads no label, needs no seed, and is reproduced by sorted-block round-robin. |
| `test_threshold_reads_no_target_frame` | The threshold code path opens no target parquet. |

### Weights

| Test | Assertion |
|---|---|
| `test_weight_sum_must_equal_one` | A weight vector summing to 0.97 raises, and the message names the observed sum. |
| `test_negative_weight_fails_closed` | A negative importance raises rather than being clipped. |
| `test_nan_weight_fails_closed` | A NaN or infinite importance raises. |
| `test_landcover_group_sum_equals_dummy_sum` | The `landcover_dominant` weight equals the sum of its `cat__landcover_dominant_*` rows, for both a 7-level and an 8-level fixture. |
| `test_k_invariance_of_categorical_penalty` | Two fixtures identical except that one has an extra unused landcover level produce identical DI values. This is the test that would have caught one-hot encoding. |
| `test_missing_importance_row_fails_closed` | An importance CSV missing one of the 9 numeric features raises. |
| `test_extra_importance_row_fails_closed` | An importance CSV with an unexpected `num__*` row raises. |

### Zero-variance and missingness

| Test | Assertion |
|---|---|
| `test_zero_variance_feature_policy_is_deterministic` | A constant source feature triggers the `EPSILON_STD` guard, sets `constant_feature_guard_used = true` for that feature, and produces a finite DI — never a division by zero or a NaN. |
| `test_missing_target_predictor_is_not_assessable` | A target cell with one missing predictor gets `cell_weighted_aoa_status == "not_assessable"` and a null DI, and is **not** counted as outside. |
| `test_missing_source_predictor_excludes_reference_cell` | A source cell with a missing predictor is absent from the reference set and is counted in `source_rows_excluded_missing`. |
| `test_no_imputation_occurs` | Changing the value of a *missing* target cell's other features does not cause it to acquire a DI; and no code path calls a mean/median fill. |
| `test_three_fractions_sum_to_one` | `inside + outside + not_assessable == 1.0` within 1e-12 over the full target population. |

### Label firewall

| Test | Assertion |
|---|---|
| `test_reader_ignores_present_target_labels` | A synthetic target parquet **containing** a `burned` column runs successfully and the column never appears in any allow-list. |
| `test_changing_target_labels_cannot_change_output` | Flipping every `burned` value in the target fixture leaves every output file byte-identical. Direct analogue of the existing `test_changing_burned_values_cannot_change_the_output`. |
| `test_transfer_result_cannot_enter_aoa_calculation` | A synthetic transfer CSV placed on disk, then perturbed, leaves every component artifact byte-identical. |
| `test_comparison_layer_cannot_mutate_aoa_artifacts` | Running the comparison layer twice, and running it against perturbed transfer values, leaves all component artifact hashes unchanged. |
| `test_no_output_contains_a_label_column` | No produced CSV or parquet has a forbidden column name. |
| `test_source_label_is_never_read_directly` | The implementation contains no read of `burned` from any Step8A or Step8B parquet; the source label enters only via the importance CSV. |

### Symmetry and directionality

| Test | Assertion |
|---|---|
| `test_reversed_weighted_pair_need_not_be_equal` | With asymmetric synthetic weights and scales, `DI(A→B) != DI(B→A)`, and neither the code nor the schema sorts the pair. |
| `test_reversed_climate_pair_must_be_equal` | `climate_distance(A,B) == climate_distance(B,A)` exactly. |
| `test_reversed_geographic_pair_must_be_equal` | `centroid_geodesic_distance_km(A,B) == centroid_geodesic_distance_km(B,A)` exactly. |
| `test_pair_token_is_never_sorted` | `f"{source}__{target}"` for a source that sorts after the target. |
| `test_threshold_is_constant_across_targets_for_a_source` | The three rows sharing a source carry identical `training_di_upper_whisker_threshold`, `source_pairwise_mean_distance` and `source_distance_normaliser`. |

### Categorical

| Test | Assertion |
|---|---|
| `test_unseen_categorical_level_follows_exact_policy` | A target level absent from the source adds exactly `w_landcover` to the squared distance against every reference cell, is counted in `target_cells_with_unseen_level`, and can never be classified inside on the strength of the categorical term alone. |
| `test_missing_categorical_is_not_unseen` | A missing level yields `not_assessable`, not `unseen`. |
| `test_categorical_mismatch_is_binary` | The penalty for level 10 vs 90 equals the penalty for level 10 vs 20. There is no ordinal structure in WorldCover codes and the code must not invent one. |
| `test_landcover_never_treated_as_numeric` | Feeding levels `10` and `90` versus `10` and `20` produces identical DI values. |

### Cardinality and provenance

| Test | Assertion |
|---|---|
| `test_four_experiments_yield_exactly_twelve_directed_pairs` | Mirrors the existing test of the same name. |
| `test_duplicate_directed_pair_fails` | A fixture producing a repeated pair raises. |
| `test_missing_aoi_fails` | Requesting an unregistered experiment raises with a clear message. |
| `test_changed_canonical_hash_fails` | Perturbing a Step8A fixture after the hash is pinned causes a fail-closed abort, not a silent re-pin. |
| `test_selection_order_changes_neither_pairs_nor_analysis_id` | Mirrors the existing test. |
| `test_identical_rerun_is_deterministic` | Two runs on the same fixture produce byte-identical outputs, including float formatting. |

### Geometry

| Test | Assertion |
|---|---|
| `test_centroid_distance_recomputes_exactly` | The stored `centroid_geodesic_distance_km` is reproduced from the stored bboxes to ≤ 1e-6 km. |
| `test_geodesic_matches_published_reference_pairs` | At least three WGS84 reference pairs with known geodesic distances agree to ≤ 1 mm. |
| `test_geodesic_self_distance_is_zero` | `d(A, A) == 0.0` exactly. |
| `test_geodesic_uses_geographiclib` | The implementation calls `geographiclib.geodesic.Geodesic.WGS84.Inverse`; `geodesic_implementation == "geographiclib_wgs84"` and the resolved package version is recorded. |
| `test_missing_geographiclib_fails_closed` | With the package unavailable, the module raises a clear error. It must **not** fall back to haversine, to a local Vincenty, or to any approximation. |
| `test_no_custom_geodesy_in_the_implementation` | The module contains no hand-rolled Vincenty/haversine formula — no `atan2`-based great-circle expression, no iteration loop over `lambda`. |
| `test_bbox_centre_matches_regions_constants` | The four hard-pinned bboxes in the completion module equal the values in `core/regions.py`. |
| `test_extended_evia_is_not_narrow_evia` | Resolving `evia_2021_extended` yields `(23.05, 38.55, 23.85, 39.15)`, not `(23.12, 38.68, 23.52, 39.08)`. |
| `test_geographic_component_reads_no_step8a` | The geographic code path opens no Step8A parquet, and no `*population_centroid*` column is produced. |

### Isolation

| Test | Assertion |
|---|---|
| `test_no_output_outside_namespace` | A full synthetic run writes only under the analysis namespace; a before/after filesystem walk confirms it. Mirrors `test_run_touches_no_other_namespace`. |
| `test_dry_run_writes_nothing` | Mirrors the existing test. |
| `test_analysis_never_fits_a_model` | No `.fit(` call is reached; `sklearn.neighbors` is allow-listed and every other estimator import is absent. |
| `test_analysis_issues_no_gee_call` | No `ee` import, no network access. |
| `test_climate_fields_null_when_export_absent` | With no climate artifact present, the numeric `climate_*` fields are null and `climate_status == "authorised_pending_export"` — the run must **not** substitute a proxy. |
| `test_exactly_four_climate_variables` | `climate_feature_count == 4` and the feature list matches exactly; the presence of `warm_season_mean_temperature` or `warm_season_precipitation` fails. |
| `test_no_era5_in_initial_run` | No ERA5-Land collection id or field appears in any output of the completion run. |
| `test_climate_component_contributions_sum_to_squared_distance` | The four contributions sum to `climate_distance²` within 1e-12. |
| `test_climate_field_rejects_forbidden_path` | Pointing a `climate_*` input at a `step5/` or `data/modis/` path raises. This is the unit-test counterpart of validator check 16. |

### Composite guard

| Test | Assertion |
|---|---|
| `test_no_bootstrap_is_performed` | `bootstrap_performed == false`; no replicate loop runs and no `*_ci_low`/`*_ci_high` column is produced for any AoA summary. |
| `test_no_composite_index_is_produced` | No output column combines the three components into a single score; the schema contains no `composite_*` or `marginal_aoa_index` field. |

### Comparison layer

| Test | Assertion |
|---|---|
| `test_comparison_selection_is_preregistered` | The `(model_family, transfer_state, metric)` triple used for the join is read from `preregistration.json`, not from a literal in the comparison code. |
| `test_primary_comparison_is_raw_thermal_roc_auc` | Exactly one row set carries `is_primary_comparison == true`, and it is `raw_thermal_roc_auc`. A primary marked on any adapted metric fails. |
| `test_secondary_comparison_block_is_complete` | All nine transfer quantities of doc 07 §8 are present; a missing one fails, so no subset can be selected after inspection. |
| `test_comparison_produces_no_p_value` | No output column matches `p_value`, `pvalue`, `significance` or `ci_` for any correlation quantity. |
| `test_comparison_runs_after_components_exist` | Invoking the comparison layer before the component artifacts exist raises. |

---

## Part 3 — Coverage note

The existing `tests/test_marginal_area_of_applicability.py` has 46 tests. The plan
above specifies **85** distinct tests for the completion module — the growth over
the earlier draft is almost entirely the normaliser, training-DI and threshold
blocks added when those contracts were corrected. Several are direct
analogues of existing tests and should be written by adapting them rather than
from scratch, so that the two modules' guarantees stay visibly parallel:

```
test_four_experiments_yield_exactly_twelve_directed_pairs
test_selection_order_changes_neither_pairs_nor_analysis_id
test_pair_token_is_never_sorted
test_changing_burned_values_cannot_change_the_output
test_no_output_contains_a_label_column
test_dry_run_writes_nothing
test_identical_rerun_is_deterministic
test_run_touches_no_other_namespace
test_no_real_experiment_id_is_hard_coded_in_the_implementation
```

**No test was written in this task.**
