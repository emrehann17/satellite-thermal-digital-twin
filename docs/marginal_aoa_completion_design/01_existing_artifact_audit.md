# 01. Existing Artifact Audit

Read-only audit of everything the completion work will consume. No file below
was modified.

---

## 1. Canonical Step8A hash verification

Recomputed with `sha256sum` at HEAD `19d825bb7dc21459aebe0870828cb25d8fc2a892`.

| Experiment | Expected (task specification) | On disk | Match |
|---|---|---|---|
| `manavgat_2021` | `054a1961fc0582a33d36413263668b63074b21ae8b03d12269b6e228787f3439` | same | ✅ |
| `bejis_2022` | `3dec785a7d8e31db2d67ed283546bbfbca1559f56df46663488d0afc24d9e393` | same | ✅ |
| `mugla_2021` | `c4ab107db2207f9f20775ccc0b3bf39381173fd07d4e82f6821ce7f40be7db8e` | same | ✅ |
| `evia_2021_extended` | `bdce859cf482f575d0f273174b157f47efd61779953fdd23d9486c5face5e553` | same | ✅ |

Paths: `outputs/experiments/<experiment_id>/step8a/step8a_500m_modeling_dataset.parquet`.

The same four hashes appear as the `input_sha256` map in
`outputs/diagnostics/marginal_area_of_applicability/comparison/manifest.json`.
**No provenance mismatch was found. No correction was needed.**

Note for the record: `outputs/` is git-ignored (`.gitignore:8`). Every hash above
is a filesystem fact, not a tracked-object fact. The completion artifact must
therefore pin these hashes itself, exactly as `marginal_aoa.v1` already does.

---

## 2. Existing `marginal_aoa.v1` artifact

```
namespace     outputs/diagnostics/marginal_area_of_applicability/
schema        marginal_aoa.v1
analysis_id   4a5b8c80489933ba501394d237b2f3d41d96c4a62ad6388a5f1264cc6b545dee
created_at    2026-07-29T12:22:29.155698+00:00
git_commit    3d7e663114c265c8cb57039c57658f1edbd1d1b4  (ancestor of HEAD)
support def   source_observed_range_and_levels_v1
```

Layout:

```
comparison/
  manifest.json                            sha256 912fa4ae709bc438bb6bdbc74ff1a58a905f5a33860de350efe9b641facc96b1
  multi_aoi_marginal_aoa_comparison.csv    sha256 ba4e41d37bd78a6c7d321d6db2306affda0c0ed7007f7deae60143beec10f9a2
  multi_aoi_marginal_aoa_comparison.json
  multi_aoi_marginal_aoa_comparison.md
pairs/<source>__<target>/                  12 directories
  marginal_aoa_summary.json
  manifest.json
  marginal_aoa_numeric_features.csv
  marginal_aoa_categorical_features.csv
  marginal_aoa_report.md
  marginal_aoa_target_cells.parquet
```

`marginal_aoa_target_cells.parquet` columns:

```
source_experiment_id, target_experiment_id, row_500m, col_500m,
numeric_features_outside_count, categorical_features_outside_count,
total_features_outside_count, features_missing_count,
features_source_support_unavailable_count, any_feature_outside_support,
any_feature_not_assessable, cell_support_status
```

This per-cell table is the natural join key for the new per-cell weighted
dissimilarity output: `(source_experiment_id, target_experiment_id, row_500m,
col_500m)`.

**Constraint carried forward: none of the above may be rewritten.** The
completion work writes to a separate namespace only.

Implementation module: `src/marginal_area_of_applicability.py`
(sha256 `66b23909a0da492d477d14122c3480f3c2ca000f6bb9079e6b1d9405b36dd8d3`,
last touched by commit `734d621`). Runner:
`scripts/run_marginal_area_of_applicability.py`. Tests:
`tests/test_marginal_area_of_applicability.py`, 46 tests.

---

## 3. Feature and population contract

Single source of truth: `src/step9a_audit_cross_region_inputs.py`
(sha256 `43f93e4fe9676a64dc775f4d4d614056a7b5f8ce98242d3de83ebb416b226ad4`).
`src/marginal_area_of_applicability.py` imports it rather than redefining it, and
the completion module must do the same.

```python
SHARED_BASELINE_FEATURES = ["ndvi_mean", "elevation_mean", "slope_mean",
                            "landcover_dominant"]
SHARED_THERMAL_FEATURES  = ["lst_anomaly_mean", "current_lst_mean",
                            "current_tvdi_mean", "tvdi_difference_mean",
                            "downscaled_lst_mean", "fused_lst_mean"]
SHARED_THERMAL_MODEL_FEATURES = SHARED_BASELINE_FEATURES + SHARED_THERMAL_FEATURES
CATEGORICAL_FEATURES = ["landcover_dominant"]
PRIMARY_POPULATIONS  = ["burnable_tree_shrub_grass"]
```

The nine numeric predictors, in contract order:

```
ndvi_mean, elevation_mean, slope_mean,
lst_anomaly_mean, current_lst_mean, current_tvdi_mean,
tvdi_difference_mean, downscaled_lst_mean, fused_lst_mean
```

`FORBIDDEN_MODEL_COLUMNS` includes `burned`, `burn_date`, `burn_month`,
`burn_day_of_year`, `label_source`, `cell_id`, `row_500m`, `col_500m`, `lon`,
`lat`. Note that `lon`/`lat` are forbidden **as model features**; the geographic
component uses AOI geometry constants, not these columns (see
`06_geographic_distance_design.md`).

---

## 4. Primary-population row counts

Population `burnable_tree_shrub_grass` after `analysis_eligible` filtering, as
recorded by the existing artifact:

| Experiment | AoA population rows | Step8A `burnable_tree_shrub_grass_count` | `valid_modeling_cells` | `total_500m_cells` |
|---|---:|---:|---:|---:|
| `bejis_2022` | 15190 | 15190 | 15759 | 15759 |
| `evia_2021_extended` | 9298 | 9309 | 22906 | 22925 |
| `manavgat_2021` | 20555 | 20555 | 24087 | 24150 |
| `mugla_2021` | 41731 | 41772 | 73045 | 73098 |

The 11-row and 41-row differences for Evia and Muğla are the
`analysis_eligible` / pre-label-burn exclusions; both experiments set
`exclude_pre_label_burns = True`. The completion module must reuse
`resolve_analysis_eligible_mask` from `src/burned_pattern_audit.py` verbatim so
its population is identical to `marginal_aoa.v1`'s. **A validator check must
assert equality against the numbers above.**

---

## 5. Target missingness on the nine numeric predictors

Taken from the existing per-pair `marginal_aoa_numeric_features.csv`; target-side
counts do not depend on the source, so one pair per target AOI is sufficient.

| Feature | manavgat (n=20555) | bejis (n=15190) | mugla (n=41731) | evia_ext (n=9298) |
|---|---:|---:|---:|---:|
| `ndvi_mean` | 4 | 0 | 0 | 0 |
| `elevation_mean` | 0 | 0 | 0 | 0 |
| `slope_mean` | 0 | 0 | 0 | 0 |
| `lst_anomaly_mean` | **380** | **980** | **1341** | **283** |
| `current_lst_mean` | 83 | 532 | 255 | 23 |
| `current_tvdi_mean` | 83 | 532 | 255 | 23 |
| `tvdi_difference_mean` | 86 | 532 | 255 | 25 |
| `downscaled_lst_mean` | 4 | 0 | 0 | 146 |
| `fused_lst_mean` | 4 | 0 | 0 | 0 |

Consequences for the design:

- Missingness is **not** negligible: `lst_anomaly_mean` is missing for 1.4–6.5%
  of the primary population depending on AOI. A nearest-neighbour distance in
  the full 9-dimensional space cannot be computed for those rows.
- **Imputation is forbidden.** `core/step10_shared.apply_regionwise_zscore` fills
  missing values with the region's own mean; reusing that here would make a
  target cell's coordinates depend on the target distribution and would hide the
  missingness. The completion module reuses the *statistics* function but never
  the *imputing* transform. See `03_weighted_dissimilarity_design.md` §3.
- The `marginal_aoa.v1` missing-value rule carries over verbatim: a target cell
  with any missing predictor is `not_assessable`, never `outside`.
- Source reference cells with any missing predictor are **excluded from the
  reference set**, and the excluded count is reported per AOI.

Categorical `landcover_dominant` has **zero** missing values in all four AOIs, in
both roles, in all 12 pairs.

---

## 6. Landcover levels present per AOI

Primary-population observed levels, from the existing categorical CSVs (ESA
WorldCover v200 class codes):

| Experiment | Observed levels | Count |
|---|---|---:|
| `bejis_2022` | 10, 20, 30, 40, 50, 60, 80 | 7 |
| `manavgat_2021` | 10, 20, 30, 40, 50, 60, 80 | 7 |
| `mugla_2021` | 10, 20, 30, 40, 50, 60, 80, 90 | 8 |
| `evia_2021_extended` | 10, 20, 30, 40, 50, 60, 80, 90 | 8 |

The only unseen-level event in the whole 12-pair set is level `90`
(herbaceous wetland) appearing in Muğla or Evia targets when the source is
Manavgat or Bejís. Largest instance: `manavgat_2021 → mugla_2021`, 7 cells,
fraction 0.000168. The categorical support component is therefore numerically
negligible in the current set — which is a reason to keep it as an honest
sidecar and *not* to let an encoding artefact inflate it.

Full-AOI dominant-class counts (all valid cells, not just the burnable
population) show why the geographic centroid choice matters:

| Experiment | `permanent_water` cells | `valid_modeling_cells` | Water share |
|---|---:|---:|---:|
| `bejis_2022` | 21 | 15759 | 0.1% |
| `manavgat_2021` | 2002 | 24087 | 8.3% |
| `mugla_2021` | 28420 | 73045 | 38.9% |
| `evia_2021_extended` | 13215 | 22906 | 57.7% |

For Muğla and Evia the bounding box is mostly sea, so a burnable-population
centroid and a bbox centre are materially different points. See
`06_geographic_distance_design.md` §3.

---

## 7. Spatial block contract

| Constant | Value | Source |
|---|---|---|
| `STEP8B_SPATIAL_BLOCK_SIZE_CELLS` | 2 (≈1 km) | `core/config.py:556` |
| `STEP8B_N_SPLITS` | 5 | `core/config.py:555` |
| `STEP8B_RANDOM_SEED` | 42 | `core/config.py:554` |
| `STEP8C_N_BOOTSTRAP` | 1000 | `core/config.py:574` |
| `STEP8C_RANDOM_SEED` | 42 | `core/config.py:575` |
| `STEP8C_CI_LOWER` / `UPPER` | 2.5 / 97.5 | `core/config.py:576-577` |
| Large-block sizes | 10 (≈5 km), 20 (≈10 km) | `src/step8_large_block_robustness.py:38-39` |

Block construction, fixed origin `(0, 0)`, assigned before population filtering:

```
block_row = floor(row_500m / block_size_cells)
block_col = floor(col_500m / block_size_cells)
block_id  = f"b{size}_r{block_row}_c{block_col}"
```

`src/step8b_train_baseline_vs_thermal_model.py:277` `add_spatial_block_id()` is
the single implementation; the completion module must call it rather than
reimplementing the floor-division.

**Fold provenance warning.** `outputs/experiments/<exp>/step8b/step8b_predictions.parquet`
carries a `fold_id` column, but those folds come from `StratifiedGroupKFold`,
which **consumes `y`**. Reusing that `fold_id` would import source-label
information into the DI normaliser. The design therefore recomputes folds
label-blind; see `03_weighted_dissimilarity_design.md` §6.

`step8b_predictions.parquet` columns (for reference only — the completion module
reads none of them):

```
cell_id, burned, burn_month, landcover_dominant, burnable_tree_shrub_grass,
burnable_tree_shrub, spatial_block_id, observed_fraction, gapfilled_fraction,
valid_30m_fraction, cropland_fraction, population, fold_id,
y_prob_baseline, y_prob_thermal
```

---

## 8. Transfer artifacts available to the post-analysis comparison layer

`outputs/diagnostics/four_aoi_transfer_decomposition/bejis_2022__evia_2021_extended__manavgat_2021__mugla_2021/four_aoi_decomposition.csv`
(sha256 `6b071b3ef7e93e0ae9d889ccfa98b852f9fc531b110df59e66f460ff2392c0d9`),
96 rows × 32 columns = 12 directions × 2 model families × 2 adaptation methods ×
2 metrics.

Columns relevant to the comparison layer:

```
source_experiment_id, target_experiment_id, direction, model_family,
adaptation_method, metric, within_target_auc, raw_auc, adapted_auc,
raw_gap, adaptation_effect, remaining_gap, recovered_fraction,
recovered_fraction_ci_low, recovered_fraction_ci_high, chance_level,
recovery_status
```

`model_family ∈ {baseline, thermal}`, `adaptation_method ∈ {regionwise_zscore,
coral_after_regionwise_zscore}`, `metric ∈ {roc_auc, pr_auc}`. Selecting a single
(family, adaptation, metric) triple collapses this to exactly 12 rows, one per
directed pair. **The selection must be preregistered, not chosen after seeing the
correlations.** See `10_decisions_required.md` decision B-12.

---

## 9. Other artifacts inspected and their role

| Artifact | Verdict |
|---|---|
| `outputs/diagnostics/domain_classifier_audit/` | Domain-classifier AUC for 3 of 6 unordered pairs; Evia absent. The artifact itself disclaims being a geographic distance. **Not usable** as a distance component; may be cited as context only. |
| `outputs/cross_region/*/step9e/distribution_shift_audit.json` | Per-feature marginal shift diagnostics. Source of `OUTSIDE_SUPPORT_THRESHOLD = 0.1`, already reused by `marginal_aoa.v1`. No importance and no distance. |
| `outputs/experiments/*/step8d/step8d_ablation_feature_importance.csv` | Same RandomForest impurity importance, but per ablation *variant* (11 variants). See `02_feature_importance_audit.md` §4. |
| `outputs/experiments/*/step8c/` | Spatial-block bootstrap uncertainty; the resampling contract the target bootstrap option should mirror. |
| `core/step10_shared.py` (sha256 `9688a8db…`) | `compute_regionwise_zscore_stats` — the repository's existing, tested, label-blind source-standardisation contract. Reused for numeric scaling. |
| `outputs/experiments/*/predictor_export_metadata.json` | Export provenance for every predictor raster. Establishes that no climate collection was ever exported. |

---

## 10. Stale-memory correction

A prior session note recorded that the `evia_2021_extended` registry entry was
uncommitted and therefore unresolvable at HEAD. **That is no longer true.**
`core/regions.py` is tracked, clean (`git status --short` reports nothing), and
`git show HEAD:core/regions.py` contains all six `evia_2021_extended`
occurrences. The AOI geometry source of truth is fully resolvable at
`19d825bb7dc21459aebe0870828cb25d8fc2a892`.
