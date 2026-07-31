# 10. Decisions — Record of Outcomes

> **STATUS: ALL DECISIONS ACCEPTED. `implementation_blocker_count = 0`.**
>
> This document is retained as the decision record — the questions, the options
> weighed, the repository evidence, and the outcome. Nothing here is still open,
> and no decision requires further user confirmation.

12 decisions in three groups:

- **Group A** — resolvable from artifact inspection alone.
- **Group B** — required a scientific preference.
- **Group C** — required new data, a new dependency, or an authorisation.

The four former blockers — **B-1, C-8, C-9, C-11** — are all resolved.

Two contracts were additionally **corrected** in the same round, superseding what
an earlier draft of this document recommended:

| ID | Superseded recommendation | Accepted contract |
|---|---|---|
| **B-5** | mean source *holdout nearest-neighbour* distance | **mean pairwise distance over all distinct source reference cell pairs** |
| **B-6** | 0.95 quantile of training DI as operative threshold | **upper whisker** `min(max(training_DI), Q3 + 1.5·IQR)`; q95 demoted to reported secondary |

Two further changes narrowed earlier proposals: the climate vector dropped from
six variables to **four** (C-8), and the geographic component dropped its
population-centroid secondary entirely (C-10).

---

## Group A — resolvable from artifact inspection

### A-2. Which model family supplies the weights?

**Question.** `baseline` (4 features) or `thermal` (10 features)?

**Options.** `thermal`; `baseline`; both, reported side by side.

**Repository evidence.** The AoA's feature contract is
`SHARED_THERMAL_MODEL_FEATURES` — 9 numeric + `landcover_dominant`. The existing
`marginal_aoa.v1` uses exactly those 10 features. The `baseline` model has only
`ndvi_mean`, `elevation_mean`, `slope_mean`, `landcover_dominant`, so its
importances cannot weight the six thermal predictors at all. `four_aoi_decomposition.csv`
carries both families, so a `baseline` comparison exists downstream.

**ACCEPTED.** `thermal`. It is the only family whose importance vector spans
the AoA feature space.

**Consequence.** Locked to the thermal predictor set. A `baseline`-weighted
variant would require restricting the AoA to 4 features, which is a different
analysis — available later as a sensitivity, not as the primary.

**Implementation blocker:** no. **Status: accepted.**

---

### A-3. Numeric standardisation statistic source

**Question.** Compute source mean/SD freshly, or reuse the repository's existing
helper?

**Options.** Reuse `core/step10_shared.compute_regionwise_zscore_stats`; write a
new implementation.

**Repository evidence.** `compute_regionwise_zscore_stats` is label-blind by
signature (the docstring states the absent `y` parameter is itself part of the
firewall), uses population SD (`ddof=0`), carries an `EPSILON_STD` constant guard,
and is the exact transform Step10's `regionwise_zscore` adaptation uses — so the
transfer results this will be compared against were produced in that coordinate
system.

**ACCEPTED.** Reuse it. Do **not** reuse `apply_regionwise_zscore`, which
imputes missing values with the region mean (doc 03 §3.3).

**Consequence.** One standardisation contract across Step10 and the AoA
completion. The imputation divergence must be documented explicitly, because the
two functions are neighbours in the same module and reusing only one of them is
surprising.

**Implementation blocker:** no. **Status: accepted.**

---

### A-7. Output namespace convention

**Question.** `analysis_id` in the path, or in the manifest only?

**Options.** `outputs/diagnostics/marginal_aoa_completion/<analysis_id>/`;
fixed namespace with a `--force` overwrite guard.

**Repository evidence.** Two conventions coexist. `marginal_aoa.v1` uses a fixed
namespace with `test_different_analysis_id_cannot_overwrite_without_force`.
`four_aoi_transfer_decomposition` uses an identity path segment
(`<canonical_set_id>/`).

**ACCEPTED.** `analysis_id` in the path. A re-run with different inputs lands
somewhere else instead of colliding, which removes the need for a force flag
entirely.

**Consequence.** Consumers must resolve the `analysis_id` to find the artifact.
Mitigated by recording it in the advisor package index, as
`source_artifact_index.csv` already does for the 12 pair IDs.

**Implementation blocker:** no. **Status: accepted.**

---

### A-10a. Geometry contract hash format

**Question.** Exact string format hashed to produce `geometry_contract_hash`.

**Options.** `repr()`-based deterministic string (doc 06 §2); `canonical_json` of
a bbox dict; a rounded fixed-decimal format.

**Repository evidence.** `canonical_json` + `sha256_bytes` already exist in
`src/step8_large_block_robustness.py` and are used by `marginal_aoa.v1` for
`analysis_id`.

**ACCEPTED.** `canonical_json({"experiment_id":…, "crs":"EPSG:4326",
"kind":"bbox", "coordinates":[lon_min, lat_min, lon_max, lat_max]})` then
`sha256_bytes`. Reuses the existing helper rather than inventing a format;
`json.dumps` round-trips these floats exactly.

**Consequence.** None beyond consistency.

**Implementation blocker:** no. **Status: accepted.**

---

### A-10b. How the four bboxes are read

**Question.** Muğla and extended-Evia are named module constants; Manavgat and
Bejís are inline `ee.Geometry.BBox(...)` literals inside `build_regions()`,
unreachable without a GEE session.

**Options.** (i) Hard-pin all four in the completion module with a test asserting
agreement against `core/regions.py`. (ii) Promote the two inline literals to
module-level constants — a production edit. (iii) Call `build_regions()` and
`.getInfo()` — requires GEE auth.

**Repository evidence.** `core/regions.py:33-40` states the constants exist as
plain tuples precisely so "tests (and provenance/hash logic) can verify AOI
coordinates **without** GEE auth". That intent is only half-realised: two of four
AOIs still need a session.

**ACCEPTED.** (i) for this work. (ii) is the better long-term fix and should
be proposed separately — it is a production change and therefore out of scope
here.

**Consequence.** A hard-pinned mapping can drift from `core/regions.py`. Mitigated
by `test_bbox_centre_matches_regions_constants`, which fails loudly on drift.

**Implementation blocker:** no. **Status: accepted.**

---

## Group B — scientific preference required

### B-1. Which source feature importance? — RESOLVED (was a blocker)

**Question.** Accept RandomForest **impurity (Gini)** importance, or authorise a
new **permutation** importance fit?

**Options.**

| | Description | Cost |
|---|---|---|
| (i) | Step8B impurity importance, `(burnable_tree_shrub_grass, thermal)` | None — artifact exists for all four AOIs |
| (ii) | New out-of-fold permutation importance on the existing 5-fold spatial CV | New model fits; ~4 AOIs × 5 folds × 10 features × n_repeats |
| (iii) | Both, impurity primary, permutation as sensitivity | Cost of (ii) plus reporting |

**Repository evidence.**
`src/step8b_train_baseline_vs_thermal_model.py:654-674` — a final model refit on
the **whole population**, `feature_importances_` read directly. That is
mean-decrease-in-impurity, in-sample. `rg -c "permutation_importance"` finds only
`src/step7c_train_downscaling_model.py`, which is the 30 m LST downscaling
regressor — a different model on a different feature space. No SHAP anywhere.
All four impurity artifacts exist, sum to 1.0, and have `min = 0.0`.

**ACCEPTED.** (i), with the mandatory limitation text in
`02_feature_importance_audit.md` §3 attached to every report and to the artifact
metadata.

**Consequence.** The weighting is in-sample and biased toward continuous
predictors relative to the single low-cardinality categorical one. With 9 numeric
features against one 7-to-8-level categorical, this is exactly the configuration
where the bias bites, so the landcover group weights (0.009–0.106) may be
understated. The advisor asked for a Meyer & Pebesma style index; that method is
usually run with permutation importance, and answering with impurity importance
is a real, disclosable deviation.

**Implementation blocker: NO — resolved. Status: accepted.** Option (i) is
adopted; no permutation fit is produced in this run, and a held-out permutation
importance remains available as a later, separately preregistered sensitivity.

---

### B-4. Numeric scale: SD or a robust alternative?

**Question.** Standardise by source SD, or by source IQR/MAD?

**Options.** SD only; SD primary with an IQR sensitivity column; IQR only.

**Repository evidence.** SD is the repository's established contract
(`compute_regionwise_zscore_stats`, `ddof=0`, `EPSILON_STD` guard) and matches the
published AoA construction. But `elevation_mean` and the four LST-family features
are heavy-tailed here — Bejís `elevation_mean` spans `[120.7, 1990.3]` — and the
existing artifact already flags sensitivity to extreme source cells as a known
weakness of its own min/max definition.

**ACCEPTED.** SD only for the primary run. Add the IQR variant **only** if the
advisor asks for a robustness check, and preregister it before running it.

**Consequence.** Accepts SD's sensitivity to extreme source cells, in exchange for
living in the same coordinate system as Step10's adaptation results.

**Implementation blocker:** no. **Status: accepted.**

---

### B-5. DI normaliser construction

**Question.** What denominator, from which fold structure?

**Options.**

| | Denominator | Fold structure |
|---|---|---|
| (i) | Mean source-internal NN distance | none |
| (ii) | Mean source spatial-fold **holdout** NN distance | ≈5 km blocks, 5 folds, label-free round-robin |
| (iii) | Same as (ii) but median | |
| (iv) | Same as (ii) but reusing Step8B `fold_id` | `StratifiedGroupKFold`, **label-informed** |

**Repository evidence.** `step8b_predictions.parquet` carries `fold_id`, but it
comes from `StratifiedGroupKFold`, which consumes `y`. `STEP8B_SPATIAL_BLOCK_SIZE_CELLS = 2`
(≈1 km) is small relative to predictor autocorrelation at 500 m. The repository's
own preregistered large-block scales are 10 and 20 cells (≈5 km, ≈10 km).

**ACCEPTED — and it supersedes the earlier recommendation.** The normaliser is
**not** any nearest-neighbour mean. It is:

```
source_pairwise_mean_distance
    = mean weighted distance over all DISTINCT source reference cell pairs
      (self-distance excluded, categorical term included)

source_distance_normaliser = source_pairwise_mean_distance
```

**Why the earlier holdout-NN recommendation was wrong.** A nearest-neighbour
distance on a dense, autocorrelated 500 m grid measures **grid spacing**, not the
spread of the source distribution. Spatial folds mitigate that but do not remove
it: with ≈5 km blocks a held-out cell's nearest out-of-fold neighbour still sits
just across a block boundary. The resulting denominator shrinks as an AOI is
sampled more densely, which would make DI values incomparable between a
9 298-cell AOI (Evia) and a 41 731-cell AOI (Muğla) purely through sample size.
The mean pairwise distance is a genuine scale statistic — the expected distance
between two randomly drawn source cells — and is essentially insensitive to
sampling density.

The normaliser uses **no folds**, no labels and no target data, and is identical
across a source's three directed rows (`normaliser_uses_folds = false`).

The ≈5 km fold structure is retained, but **only** for the training DI and the
threshold — see B-6.

**Consequence.** (i) would measure grid spacing rather than source spread. (iv)
would push source-label information into the denominator of every DI, which is
avoidable and therefore should be avoided. The ≈5 km choice sets the DI scale;
a different block size gives a different scale, which is why it is preregistered.

**Implementation blocker:** no. **Status: accepted.**

---

### B-6. AOA threshold

**Question.** 0.95 quantile of the source holdout DI, or its maximum?

**Options.** q95; maximum; another quantile.

**Repository evidence.** `marginal_aoa.v1` defines support by the exact observed
source min/max and lists as a limitation that it "is sensitive to single extreme
source cells and to source sample size". A maximum-based threshold would carry
that same fragility into the replacement diagnostic.

**ACCEPTED — and it supersedes the earlier recommendation.** The operative
threshold is the **upper whisker**:

```
training_DI(s)  = holdout_nearest_distance(s) / source_pairwise_mean_distance
upper_whisker   = Q3(training_DI) + 1.5 * IQR(training_DI)
threshold       = min( max(training_DI), upper_whisker )

primary_threshold_method = "source_spatial_fold_holdout_di_upper_whisker_v1"
```

**Why the whisker rather than q95.** The whisker adapts to the *shape* of the
training-DI distribution; a fixed quantile declares ~5% of source holdout cells
outside by construction, whether or not the distribution has a tail at all. The
`min(max(training_DI), …)` clamp additionally guarantees the threshold never
exceeds anything actually observed in the source, which bounds the single-extreme-
cell fragility that `marginal_aoa.v1` flags for its own min/max definition.

**q95 is demoted to a reported secondary** (`training_di_q95_threshold`, method
token `source_spatial_fold_holdout_di_q95_v1`), alongside q50, q90, q99 and the
maximum, plus `training_di_q1`/`q3`/`iqr` so the whisker is recomputable. The
primary inside/outside classification uses the whisker and nothing else, and
validator check 13b asserts exactly that.

**Consequence.** By construction ~5% of source holdout cells fall outside their
own AOA. That is expected and must be stated, or it will read as a defect.

**Implementation blocker:** no. **Status: accepted.**

---

### B-7. Landcover handling

**Question.** One-hot, mismatch penalty, or sidecar only?

**Options.** (A) one-hot with group-normalised importance; (B) weighted mismatch
penalty; (C) sidecar only, excluded from the DI; (D) other.

**Repository evidence.** Level counts differ by source: 7 for Manavgat and Bejís,
8 for Muğla and Evia. Under one-hot with per-dummy weight `w_lc/K`, a mismatch
contributes `2·w_lc/K`, so the penalty depends on **K** — an encoding artefact
that would make a 7-level source and an 8-level source incomparable. Landcover's
group weight ranges from 0.009 (Muğla) to 0.106 (Bejís). Individual dummy levels
hit exactly 0.0 in all four AOIs. Total unseen-level exposure across all 12 pairs
is **7 cells**.

**ACCEPTED.** (B), the Gower categorical term
`w_landcover · 1[ℓ(x) ≠ ℓ(x_i)]`, **plus** (C) — the existing unweighted
unseen-level diagnostic is retained unchanged as a linked sidecar.

**Consequence.** The numeric block is unbounded while the categorical term is
bounded by `w_landcover` — the standard Gower scale asymmetry, which must be
disclosed. Given 7 unseen-level cells in the entire set, the practical effect on
current numbers is negligible.

**Implementation blocker:** no. **Status: accepted.**

---

### B-11. Bootstrap for the weighted summaries?

**Question.** Point estimates only, or target spatial-block bootstrap CIs?

**Options.** (A) point estimate only; (B) target spatial-block bootstrap.

**Repository evidence.** `marginal_aoa.v1` produces no intervals; all 12 evidence
rows carry `descriptive_no_interval`. Step8C/Step10 conventions exist and are
reusable: 1000 replicates, seed 42, percentile 2.5/97.5, `MIN_VALID_BOOTSTRAP = 900`.

**ACCEPTED: (A), point estimate only.** `uncertainty_policy =
"point_estimate_only"`, `bootstrap_performed = false`. This matches
`marginal_aoa.v1`, whose 12 evidence rows all carry `descriptive_no_interval`, and
keeps the two artifacts directly comparable.

(B) is fully specified in `07_integrated_schema_and_namespace.md` §10 and may be
added **later, only through a separately preregistered sensitivity analysis** —
never as an unannounced addition to this run.

**Consequence.** Under (A), no uncertainty statement is available for the weighted
fractions. Under (B), the source reference set, weights, threshold and normaliser
must all be held fixed, or the interval answers a different question.

**Constraint either way:** climate and geographic distances are deterministic
AOI-level values and get **no** interval. Manufacturing one would be fabrication.

**Implementation blocker:** no. **Status: accepted.**

---

### B-12. Which transfer metrics for the ranking comparison?

**Question.** Which of the 96 rows in `four_aoi_decomposition.csv` are joined?

**Options.** A single preregistered `(model_family, adaptation_method, metric)`
triple; all four triples reported together; a triple chosen after inspection.

**Repository evidence.** 96 rows = 12 directions × {baseline, thermal} ×
{regionwise_zscore, coral_after_regionwise_zscore} × {roc_auc, pr_auc}. The
existing advisor document reported `thermal` / `regionwise_zscore` / `roc_auc`
and found ρ = +0.077 against unweighted support.

**ACCEPTED, and the primary is changed.** The primary ranking comparison is

```
model_family = "thermal",  transfer_state = "raw",  metric = "roc_auc"
```

**`regionwise_zscore` ROC-AUC must not be the primary ordering.** An adapted
metric measures performance *after* an alignment step that itself removes part of
the distribution shift the AoA is quantifying; ranking a shift diagnostic against
a shift-corrected metric would partially cancel the effect under study. `raw_auc`
is the untransformed cross-region performance and is the right comparison for a
diagnostic that describes support before any adaptation.

The **complete** secondary block is reported every time — raw thermal PR-AUC,
both ROC-AUC and PR-AUC transfer gaps, adapted thermal ROC-AUC and PR-AUC for
every preregistered adaptation, and recovered fraction. Reporting all of it is
what makes post-hoc selection impossible rather than merely discouraged.

Also reported: Spearman and Kendall rank correlations, an ordered-pair ranking
table, and the Bejís-source audit.

**Consequence.** The third option is forbidden: choosing the triple after seeing
which correlates best is selection on the outcome. Preregistering the primary and
reporting all four removes the degree of freedom entirely.

**Constraint:** no p-values, no hypothesis tests, no confidence intervals on any
correlation. 12 directed pairs from four non-independent AOIs support description,
not inference.

**Implementation blocker:** no. **Status: accepted.**

---

## Group C — new data, dependency or authorisation

### C-8 (variables and period) and C-9 (export authorisation) — BOTH RESOLVED

**Question.** Which climate variables, over which reference period, from which
collection — and is the new export authorised?

**Options.**

| | Source | Period | Verdict |
|---|---|---|---|
| (i) | `IDAHO_EPSCOR/TERRACLIMATE` | 1991–2020 | **ACCEPTED** — ~4.6 km, native `def`/`aet`/`vpd` |
| (ii) | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | 1991–2020 | **Deferred, not in the initial run** — ~11 km is too coarse for these AOIs |
| (iii) | `WORLDCLIM/V1/BIO` | 1960–1990 fixed | Rejected — stale period, no moisture-deficit axis |
| (iv) | No export; report climate as absent | — | Honest fallback |
| (v) | Relabel `baseline_lst_mean` as climate | — | **Forbidden** |

**Repository evidence.** Exhaustive search found **no** climate input:
`rg` for ERA5, WorldClim, TerraClimate, CHIRPS, CHELSA, Köppen, precipitation,
aridity and climate-zone returns nothing outside `venv/`, `old_codes/` and
`docs/`. The complete set of collections ever exported is MODIS MOD11A1, Landsat
8 C02 T1_L2, Copernicus GLO30 DEM, SRTM, ESA WorldCover v200 and MCD64A1.

The closest multi-year artifact, `step5/baseline_lst_mean_celsius.tif`, fails on
six counts: 4 years not 30; **Bejís uses 2018–2021 while the others use
2017–2020**; anniversary windows differ (07-27 / 08-14 / 07-28 / 08-02); cloud
coverage ranges 47.8%–77.3%; it is a single variable; and it is clear-sky surface
temperature, which is confounded with the land cover and vegetation state the AoA
already measures.

The MODIS artifact's own metadata says, verbatim: `"experiment predictor window
(single-season mean/std over daily MODIS scenes within the window), NOT a
multi-year baseline"`.

**ACCEPTED: (i) alone.** Option (ii) is **not** part of the initial run — an
earlier draft proposed it as a preregistered cross-check, and that is withdrawn.
ERA5-Land is recorded only as a possible later sensitivity that would need its own
preregistration. Options (iii), (iv) and (v) are not taken.

**Consequence.** Substituting a 4-year AOI-specific LST composite would be exactly
the mislabelling the task forbids, and the reference-period mismatch alone would
make Bejís's climate distances partly an artefact of using a different four-year
window. That path stays closed: validator check 16 enforces it as a
path-provenance assertion rather than a naming convention.

**Implementation blocker: NO — both resolved. Status: accepted.**

C-8: `IDAHO_EPSCOR/TERRACLIMATE`, 1991-01-01..2020-12-31, warm season months
6–9, and **exactly four** variables — `annual_mean_temperature_c`,
`annual_precipitation_mm`, `warm_season_climatic_water_deficit_mm`,
`warm_season_vpd_kpa`. The six-variable draft is superseded: *warm-season mean
temperature* and *warm-season precipitation* were removed as near-duplicates of
the annual axes beside them, which would have double-weighted the temperature and
precipitation directions in an equally-weighted metric. **No ERA5-Land
cross-check in this run** — recorded as a later sensitivity only.

The spatial mask is TerraClimate's **native valid-land support**, applied
identically to the AOI summaries and to the Mediterranean reference window; the
earlier ESA WorldCover mask proposal is superseded, so there is one mask
definition rather than two that could disagree.

C-9: the export is **authorised**. It has not been run. Until it runs, the
numeric `climate_*` fields stay null with
`climate_status = "authorised_pending_export"`, and no proxy is substituted.

---

### C-10. Geographic centroid definition

**Question.** bbox centre, geometry centroid, population-weighted centroid, or
valid-cell centroid?

**Options.** All four; and whether to report a secondary alongside the primary.

**Repository evidence.** All four AOIs are axis-aligned EPSG:4326 rectangles, so
the planar geometry centroid **is** the bbox centre exactly; a spherical
area-weighted centroid would differ by well under 100 m at these latitudinal
extents. But the bboxes are not mostly land: Muğla is **38.9%** permanent water
and Evia **57.7%** (doc 01 §6), so their burnable populations are materially
displaced from their bbox centres.

**ACCEPTED, and narrowed.** bbox centre
(`centroid_definition = "bbox_centre_planar_epsg4326"`) is the **only** centroid.
The burnable-population centroid that an earlier draft proposed as a secondary
column is **removed entirely**.

Reason for the narrowing: computing a population centroid means opening the
target Step8A frame and reading `lon`/`lat`. Dropping it lets the firewall
statement for this component be unconditional — *the geographic distance reads no
target data of any kind* — instead of carrying an exemption clause for one
secondary column.

```
geographic_component_reads_step8a = false
population_centroid_reported      = false
```

**Consequence.** For Muğla and Evia the primary centroid does not sit where the
modelled cells are. That is the price of a metric that is exactly recomputable
from two committed constants and cannot drift when Step8A is regenerated. The
secondary column exists precisely so the discrepancy is visible rather than
hidden. Note the secondary reads `lon`/`lat` from Step8A — legitimate, but it must
be declared in the firewall block as an allow-listed non-label column.

**Implementation blocker:** no. **Status: accepted.**

---

### C-11. Geodesic implementation — RESOLVED

**Question.** How is a WGS84 geodesic distance computed?

**Options.** (i) implement Vincenty inverse in-repo; (ii) add `pyproj`;
(iii) haversine.

**Repository evidence.** **Neither `pyproj`, `geographiclib`, `geopy` nor
`shapely` is installed or present in `requirements.txt` / `requirements-lock.txt`.**
Verified by inspecting `venv/lib/python*/site-packages/` and grepping the lock
file. Available: `numpy 2.5.1`, `scipy 1.18.0`, `scikit-learn 1.9.0`,
`rasterio 1.5.0`, `pyarrow 24.0.0`.

**ACCEPTED.** (i). ~40 lines of pure numpy, deterministic, no new dependency,
with an iteration cap, a fail-closed `GeodesicError`, and tests against published
reference pairs to ≤ 1 mm. The near-antipodal non-convergence case cannot arise
among four Mediterranean AOIs.

**Consequence.** (iii) is rejected: spherical error reaches ~0.5%, several
kilometres at these distances, for no saving. (ii) adds a compiled PROJ dependency
to a locked environment for six scalar computations — defensible, but heavier than
the problem.

**Implementation blocker: NO — resolved. Status: accepted.**

---

## Summary

**All 17 rows accepted. `implementation_blocker_count = 0`.**

| ID | Decision | Group | Accepted outcome | Was a blocker |
|---|---|---|---|---|
| B-1 | Source feature importance | B | Step8B RandomForest impurity, `(burnable_tree_shrub_grass, thermal)`, with the mandatory limitation | yes → resolved |
| A-2 | Model family | A | `thermal` | no |
| B-1b | Negative-importance policy | B | fail-closed assert; clip-at-zero only if the source changes | no |
| A-3 | Numeric scale source | A | reuse `compute_regionwise_zscore_stats`; never `apply_regionwise_zscore` | no |
| B-4 | SD vs robust scale | B | source SD, `ddof=0`, `EPSILON_STD` guard | no |
| B-5 | DI normaliser | B | **mean pairwise distance over all distinct source pairs** (supersedes holdout-NN) | no |
| B-6 | AOA threshold | B | **upper whisker** `min(max(training_DI), Q3+1.5·IQR)`; q95 secondary | no |
| B-7 | Landcover handling | B | Gower mismatch penalty + retained unweighted sidecar | no |
| C-8 | Climate collection, period, variables | C | TerraClimate 1991–2020, months 6–9, **4 variables**, no ERA5 | yes → resolved |
| C-9 | Climate export authorisation | C | **authorised**; not yet run | yes → resolved |
| C-10 | Centroid definition | C | bbox centre only; **no population centroid** | no |
| C-11 | Geodesic implementation | C | **`geographiclib`**; user-run install, reviewed separately | yes → resolved |
| B-11 | Bootstrap | B | point estimate only; bootstrap deferred to a separate preregistration | no |
| B-12 | Transfer comparison | B | primary = **raw thermal ROC-AUC**; complete secondary block always reported | no |
| A-7 | Namespace convention | A | `analysis_id` in the path | no |
| A-10a | Geometry hash format | A | `canonical_json` + `sha256_bytes` | no |
| A-10b | How bboxes are read | A | hard-pin with a drift test | no |

The table lists 17 rows because three decisions the task numbered as one
(B-1/B-1b, C-8/C-9, A-10a/A-10b) each split into two independently answerable
choices. The headline count of **12 decisions** refers to the task's own
numbering.

### What still needs a human, and what does not

Nothing in this table requires further confirmation. Two items remain **operational
steps** rather than open decisions:

1. **Run the authorised climate export.** Until it runs, the numeric `climate_*`
   fields stay null with `climate_status = "authorised_pending_export"`.
2. **Install `geographiclib`.** The implementation may add it to
   `requirements.txt` and `requirements-lock.txt`, but installation is user-run
   and is reviewed separately.

Neither blocks the design; both are execution steps with a settled contract.
