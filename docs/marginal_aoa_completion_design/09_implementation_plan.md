# 09. Implementation Plan

Ordered phases with explicit gating decisions. **Nothing below has been started.**

---

## Phase 0 — Decisions — COMPLETE

All four former blockers are resolved. **`implementation_blocker_count = 0`.**

| ID | Decision | Resolution |
|---|---|---|
| B-1 | Source feature importance | **Accepted:** Step8B RandomForest impurity importance, `(burnable_tree_shrub_grass, thermal)`. No new permutation fit. |
| C-8 | Climate collection, period, variables | **Accepted:** TerraClimate, 1991–2020, months 6–9, **four** variables. No ERA5 cross-check. |
| C-9 | Climate export authorisation | **Authorised.** |
| C-11 | Geodesic route | **Accepted:** the pinned `geographiclib` package. No custom Vincenty. |

Two contracts were also corrected in the same round and are now fixed:

- **DI normaliser** = mean pairwise weighted distance over all distinct source
  reference cell pairs. The earlier holdout-nearest-neighbour mean is withdrawn
  and must not appear anywhere.
- **AOA threshold** = upper whisker `min(max(training_DI), Q3 + 1.5·IQR)`. The
  0.95 quantile is a reported secondary only.

Every remaining decision carries `decision_status = "accepted"` and
`requires_user_confirmation = false` in `preregistration_candidate.json`.

**Remaining Phase 0 work:** freeze `preregistration.json` into the output
namespace and hash it before Phase 2 begins. The `analysis_id` is assigned at
that point and not before.

---

## Phase 1 — Scaffolding, no science

Create the module and its contracts, with no numerical work.

```
src/marginal_aoa_completion.py
scripts/run_marginal_aoa_completion.py
tests/test_marginal_aoa_completion.py
```

`src/marginal_aoa_completion.py` imports its contracts rather than redefining
them, exactly as `src/marginal_area_of_applicability.py` does:

```python
from src.step9a_audit_cross_region_inputs import (
    CATEGORICAL_FEATURES, FORBIDDEN_MODEL_COLUMNS, PRIMARY_POPULATIONS,
    SHARED_THERMAL_MODEL_FEATURES, resolve_step8a_dataset_path,
)
from src.burned_pattern_audit import (
    ANALYSIS_ELIGIBLE_COLUMN, BURNABLE_MASK_COLUMN, PRE_LABEL_EXCLUDED_COLUMN,
    resolve_analysis_eligible_mask, resolve_experiments,
)
from src.step8_large_block_robustness import (
    _git_commit, canonical_json, sha256_bytes, sha256_file,
)
from src.step8b_train_baseline_vs_thermal_model import add_spatial_block_id
from core.step10_shared import compute_regionwise_zscore_stats, EPSILON_STD
```

Note what is deliberately **not** imported: `apply_regionwise_zscore` (it imputes —
doc 03 §3.3), and anything from `step9b`/`step10` that touches transfer metrics.

Deliverables: `SCHEMA_VERSION`, the field contracts, `LABEL_FIREWALL` and
`SOURCE_LABEL_POLICY` dicts, `LIMITATIONS`, a `--dry-run` path that writes
nothing, and the isolation and firewall tests from doc 08 Part 2 passing against
synthetic fixtures.

**Gate:** `test_dry_run_writes_nothing`, `test_no_output_outside_namespace`,
`test_analysis_never_fits_a_model` and `test_analysis_issues_no_gee_call` pass.

---

## Phase 2 — Feature weights

Gated on B-1.

Implement the weight derivation from `02_feature_importance_audit.md` §6:
read the four importance CSVs, filter to
`(burnable_tree_shrub_grass, thermal)`, assert the exact expected row set,
assert non-negativity and finiteness, group-sum the landcover dummies,
renormalise defensively, compute the weight diagnostics.

Writes `weighted_predictor_space/source_feature_weights.csv` and
`config/feature_importance_inventory.json`.

**Gate:** every weight test in doc 08 Part 2 passes, including
`test_k_invariance_of_categorical_penalty`. Validator checks 9, 10, 11, 12 pass.

**Expected output for sanity-checking**, from the audited values (doc 02 §2):
`elevation_mean` should be the largest numeric weight for Manavgat (0.228),
Bejís (0.334) and Muğla (0.173), and `lst_anomaly_mean` the largest for Evia
(0.140). Landcover group weight between 0.009 and 0.106. If the implementation
produces anything materially different, it is reading the wrong rows.

---

## Phase 3 — Source standardisation, normaliser, folds, threshold

Gated on Phase 2. Note the order: the **normaliser comes before the folds**,
because it does not use them.

Per source AOI:

1. Load the source primary population with an explicit `columns=` allow-list.
2. Drop rows with any missing predictor; record `source_rows_excluded_missing`.
3. `compute_regionwise_zscore_stats` on the source only; record mean, scale and
   the constant-guard flag per feature. **Do not** call `apply_regionwise_zscore`.
4. Scale numeric coordinates by `sqrt(w_j)`.
5. **Normaliser (fold-free).** Compute
   `source_pairwise_mean_distance` — the mean weighted distance over all
   `n(n−1)/2` distinct source reference cell pairs, self-distance excluded,
   categorical term included — by deterministic chunked accumulation.
   Set `source_distance_normaliser = source_pairwise_mean_distance`,
   `normaliser_uses_folds = false`, and record `n_distinct_source_pairs`.
6. `add_spatial_block_id(df, block_size_cells=10)`; assert no block is split.
7. Fold assignment: sorted-block round-robin into 5 folds, label-free, seed-free.
8. `holdout_nearest_distance(s)` for every source cell — exact, full mixed
   distance, restricted to cells outside `s`'s fold.
9. `training_DI(s) = holdout_nearest_distance(s) / source_pairwise_mean_distance`.
10. Threshold:
    `training_di_upper_whisker_threshold = min(max(training_DI), Q3 + 1.5·IQR)`,
    `primary_threshold_method = "source_spatial_fold_holdout_di_upper_whisker_v1"`,
    `whisker_clamped_to_max` recorded truthfully.
    Also record `training_di_q1/q3/iqr` and the secondary
    `q50/q90/q95/q99/max` thresholds, with
    `training_di_q95_method = "source_spatial_fold_holdout_di_q95_v1"`.

Writes `weighted_predictor_space/source_threshold_diagnostics.csv`.

**Gate:** validator checks 13 and 13b pass — normaliser and threshold are
identical across a source's three targets, `normaliser_uses_folds` is `false`,
`fold_assignment_reads_label` is `false`, and the whisker recomputes from the
stored `q3`/`iqr`/`max`.

This is the most expensive phase. Two large computations per source AOI, for
Muğla roughly 7.9 × 10⁸ pair evaluations for the normaliser and 1.3 × 10⁹ for the
training DI (doc 03 §8.1). Still minutes, not hours, at d = 9.

---

## Phase 4 — Directed target dissimilarity

Gated on Phase 3.

For each of the 12 directed pairs: standardise the target with the **source**
statistics, compute the exact nearest weighted distance via the two-query
partition, divide by `source_pairwise_mean_distance`, classify against
`training_di_upper_whisker_threshold`, and decompose the nearest-neighbour
contribution per feature.

Writes `weighted_predictor_space/target_cell_dissimilarity.parquet`
(260 322 rows) and the weighted block of
`weighted_predictor_space/directed_pair_summary.csv`.

**Gate:** validator checks 2, 3, 4, 7, 13b, 14 pass. The three fractions sum to
1.0. The per-pair row counts match `target_rows`. The inside fraction was
computed against the upper whisker and not against q95.

**Stated expectation, recorded before the run:** the three Bejís-source
directions should show elevated dissimilarity, because `elevation_mean` is both
the sole driver of their unweighted range violations and Bejís's top-weighted
feature (0.334). If weighting *reduces* their separation, that is a genuine and
reportable surprise — writing the expectation down now is what makes it one.

---

## Phase 5 — Climatic distance

C-8 and C-9 are accepted and the export is **authorised**. This phase is
unblocked; it simply has not been run.

Two separate deliverables, deliberately in separate modules:

```
scripts/export_climate_normals.py     issues GEE calls; gee_query_issued = true
src/marginal_aoa_completion.py        reads the exported rasters only
```

The export script writes `data/climate/terraclimate_normals_1991_2020.tif` per
AOI plus the Mediterranean reference raster, with a
`climate_export_metadata.json` mirroring the existing
`predictor_export_metadata.json` convention.

The completion module then computes the AOI climate vectors over TerraClimate's
**native valid-land support**, the reference-window standardisation
(`lon [-10, 42]`, `lat [30, 47]`, same land support), the 6 pairwise distances and
the four per-variable contributions.

The climate vector has **exactly four** variables:
`annual_mean_temperature_c`, `annual_precipitation_mm`,
`warm_season_climatic_water_deficit_mm`, `warm_season_vpd_kpa`.
No warm-season mean temperature, no warm-season precipitation, no ERA5-Land
cross-check.

Writes `climate_distance/aoi_climate_vectors.csv`,
`climate_distance/pairwise_climate_distance.csv`,
`config/climate_input_inventory.json`.

**Gate:** validator checks 5, 15, 16 pass. Component contributions sum to
`climate_distance²`.

**Until the export actually runs:** every numeric `climate_*` field stays null
with `climate_status = "authorised_pending_export"` and validator check 5 records
`SKIPPED_PENDING_EXPORT`. Do **not** substitute a proxy under any circumstances.
The rest of the package ships regardless — this phase is independent of the
weighted chain.

---

## Phase 6 — Geographic distance

C-11 is accepted: **`geographiclib`**.

Add `geographiclib` to `requirements.txt` and `requirements-lock.txt`.
**Installation is a user-run step and is reviewed separately** — no auto-install,
no `pip install` from any script, and a clear fail-closed error if the package is
absent rather than a fallback to haversine or a local Vincenty.

Then compute the 6 centroid distances and the 6 minimum boundary distances from
`core/regions.py` bbox constants alone. **No population centroid**, and no Step8A
read of any kind in this phase.

Writes `geographic_distance/aoi_geometry_summary.csv`,
`geographic_distance/pairwise_geographic_distance.csv`,
`config/geometry_inventory.json`.

**Gate:** validator checks 6, 17, 18 pass; `geodesic_implementation ==
"geographiclib_wgs84"`; the reference-pair tests agree to ≤ 1 mm; and no
`*population_centroid*` column exists.

This phase is independent of Phases 2–5 and can be built in parallel with them.

---

## Phase 7 — Assembly

Join the three component blocks onto the 12-row `directed_pair_summary.csv`, echo
the `marginal_aoa.v1` unweighted columns, compute the `analysis_id` from the
canonicalised scientific config, and write `completion_metadata.json` with the
before/after hash maps.

**Gate:** validator checks 1, 19, 20, 21, 23, 24, 25 pass. In particular the
before/after hashes of `marginal_aoa.v1` and of the four Step8A parquets must be
identical maps.

---

## Phase 8 — Uncertainty — NOT PART OF THIS RUN

B-11 is resolved: **`uncertainty_policy = "point_estimate_only"`,
`bootstrap_performed = false`.**

The first production run reports point estimates only, matching `marginal_aoa.v1`
and keeping the two artifacts directly comparable. Climate and geographic
distances are deterministic AOI-level values and receive no interval at all.

A target spatial-block bootstrap may be added **later, and only through a
separately preregistered sensitivity analysis** — never as an unannounced
addition. Its contract is already written down in
`07_integrated_schema_and_namespace.md` §10 so that, if it is ever run, it is run
as specified rather than designed after the fact.

---

## Phase 9 — Comparison layer

Written **last**, after every component file exists and is hashed.

Join to `four_aoi_decomposition.csv` on the preregistered primary triple

```
model_family = "thermal",  transfer_state = "raw",  metric = "roc_auc"
```

— **raw thermal ROC-AUC, not `regionwise_zscore`.** Ranking a shift diagnostic
against a shift-corrected metric would partially cancel the effect under study
(doc 07 §8).

Then report the complete secondary block (raw thermal PR-AUC, both ROC-AUC and
PR-AUC transfer gaps, adapted thermal ROC-AUC and PR-AUC for every preregistered
adaptation, and recovered fraction), Spearman and Kendall rank associations, the
ordered-pair ranking table, top/bottom-3 overlap for each diagnostic, the
Bejís-source audit and the directed-versus-symmetric split.

Writes `comparison/marginal_diagnostics_with_transfer.csv`,
`comparison/ranking_summary.csv`, `comparison/scientific_summary.md`.

**Gate:** validator checks 22 and 22b pass — no transfer column appears outside
`comparison/`, every comparison file post-dates every component file, and exactly
one comparison is marked primary and it is `raw_thermal_roc_auc`.

**Constraints, restated because this is where they are easiest to violate:**
no p-value, no hypothesis test, no confidence interval on any correlation; the
permitted wording is "In this four-AOI set, the diagnostic ordering does or does
not reproduce the observed **raw-transfer** ordering"; the forbidden wording is
"Marginal diagnostics can never rank transfer."

---

## Phase 10 — Validator

Implement all 27 checks from doc 08 Part 1, writing
`validation_report.json`. `marginal_aoa.v1` shipped without a dedicated validator
log — `03_marginal_aoa.md` records "Dedicated validator log: **none found**", and
its technical status therefore rests on metadata and hashes rather than on a
validator. **The completion artifact should not repeat that gap.**

---

## Ordering summary

```
Phase 0  decisions                     COMPLETE - 0 blockers
Phase 1  scaffolding                   ──┐
Phase 2  weights                         │
Phase 3  source: normaliser, threshold   ├─ sequential
Phase 4  directed target                 │
Phase 7  assembly                      ──┘
Phase 5  climate (export authorised)      parallel, independent
Phase 6  geographic (geographiclib)       parallel, independent
Phase 8  bootstrap                        NOT IN THIS RUN
Phase 9  comparison                       last
Phase 10 validator                        last
```

No phase is blocked. Phases 5 and 6 are independent of the weighted chain and of
each other. The climate export is authorised but has not been run; until it is,
the climate fields stay null and the other two components ship complete.

---

## Estimated scope

| Deliverable | Rough size |
|---|---|
| `src/marginal_aoa_completion.py` | ~1200–1500 lines, comparable to `src/marginal_area_of_applicability.py` (1368) |
| `scripts/run_marginal_aoa_completion.py` | ~80 lines, mirroring the existing runner |
| `scripts/export_climate_normals.py` | ~250 lines; C-9 approved |
| `tests/test_marginal_aoa_completion.py` | ~1200–1400 lines, 85 tests |
| Validator | ~450 lines, 27 checks |

No new runtime dependency is required for Phases 1–4 and 7–10
(`numpy`, `pandas`, `scipy`, `scikit-learn`, `pyarrow` are all locked). Phase 6
requires **`geographiclib`**, to be added to `requirements.txt` and
`requirements-lock.txt`, with installation reviewed separately as a user-run
step. Phase 5 needs `earthengine-api`, already locked at 1.7.34.
