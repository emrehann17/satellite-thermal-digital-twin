# Marginal AoA Completion — Design Package

**Status:** design complete, **0 implementation blockers**. The scientific
contract is settled. No analysis was run, no model was fitted, no
nearest-neighbour search was performed, no distance was computed, no GEE query
was issued, no production artifact was modified.

**Created:** 2026-08-02T10:22:29Z
**Contract finalised:** 2026-08-02
**Git HEAD at audit time:** `19d825bb7dc21459aebe0870828cb25d8fc2a892` (branch `main`)

---

## What this package is for

The advisor asked for a marginal Area of Applicability index consisting of three
components:

1. an importance-weighted predictor-space dissimilarity,
2. a climatic distance,
3. a geographic distance.

The repository currently contains only an **unweighted** marginal
observed-range-and-level support diagnostic:

```
schema        marginal_aoa.v1
analysis_id   4a5b8c80489933ba501394d237b2f3d41d96c4a62ad6388a5f1264cc6b545dee
support def   source_observed_range_and_levels_v1
pairs         12 directed
population    burnable_tree_shrub_grass
predictors    9 numeric + landcover_dominant (categorical)
```

This package specifies the exact scientific contract for the three missing
components. **All decisions are now settled** — the package is a finalised
contract, not an open question list.

---

## Headline audit findings

| Question | Answer found in the repository |
|---|---|
| Do the four canonical Step8A hashes still match disk? | **Yes, all four.** See `01_existing_artifact_audit.md`. |
| Is there a usable source feature importance for all four AOIs? | **Yes, one and only one kind:** Step8B RandomForest **impurity (Gini)** importance, `population=burnable_tree_shrub_grass`, `model=thermal`, one-hot expanded. |
| Is there permutation, held-out, OOF or SHAP importance anywhere? | **No.** Nothing in `src/`, `scripts/`, `core/` or `tests/` computes one for Step8B. |
| Is there a genuine climate-normal input for the four AOIs? | **No.** No ERA5, WorldClim, TerraClimate, CHIRPS, CHELSA, Köppen, precipitation, humidity or aridity artifact exists anywhere in the repository. This is why the new export was authorised. |
| What is the closest multi-year artifact? | `step5/baseline_lst_mean_celsius.tif` and `step5c/baseline_tvdi_mean.tif` — a **4-year, AOI-specific-window Landsat composite**. It is not a climate normal and is not cross-AOI comparable. |
| Is there a canonical AOI geometry source of truth? | **Yes.** `core/regions.py`, tracked and clean at HEAD. All four AOIs are axis-aligned EPSG:4326 rectangles defined by module-level constants. |
| Is a new GEE export required? | **Yes — for the climatic distance component only, and it is AUTHORISED.** The weighted dissimilarity and the geographic distance need no new data. |
| Is a new dependency required? | **Yes — `geographiclib`, for the geodesic inverse.** Installation is a user-run step, reviewed separately. |

---

## The honesty constraint this design accepts

The proposed weighted diagnostic uses **source-model feature importances**. Those
importances come from a RandomForest that was fitted on the **source** `burned`
label. The new method therefore **must not** be described as label-blind.

The correct description, used verbatim throughout this package and required in
the artifact metadata:

> **target-label-blind, source-model-informed diagnostic**

Recorded fields:

```
target_label_used            = false
target_burn_date_used        = false
target_transfer_metric_used  = false
source_label_used            = true    (via the Step8B source model importances)
```

The existing `marginal_aoa.v1` artifact remains fully label-blind in both
directions and is not touched by any of this.

---

## Reading order

| File | Contents |
|---|---|
| `01_existing_artifact_audit.md` | Hash verification, existing artifact inventory, population and missingness counts, spatial-block contract. |
| `02_feature_importance_audit.md` | Every importance artifact found, its exact provenance, and why only one candidate survives. |
| `03_weighted_dissimilarity_design.md` | Exact formula, scaling, weights, categorical handling, normaliser, threshold, output fields, computational contract. |
| `04_climate_input_audit.md` | What was searched, what exists, and why nothing in the repository is a climate normal. |
| `05_climatic_distance_design.md` | The authorised export: exact collection, version, four variables, reference period, land support, metric. |
| `06_geographic_distance_design.md` | Geometry source of truth, bbox centroid, `geographiclib` geodesic route. |
| `07_integrated_schema_and_namespace.md` | Schema, namespace, file layout, 12-row directed-pair table, transfer comparison layer. |
| `08_validation_and_test_contract.md` | The 27 validator checks and the 85-test synthetic plan. |
| `09_implementation_plan.md` | Ordered implementation phases; Phase 0 is complete. |
| `10_decisions_required.md` | The decision record — every decision, its evidence, and its accepted outcome. |
| `source_artifact_inventory.csv` | Frozen inputs with paths, hashes and status. |
| `candidate_feature_importances.csv` | The importance-artifact audit table. |
| `candidate_climate_inputs.csv` | Every climate candidate, existing and proposed, with suitability verdict. |
| `canonical_geometry_inventory.csv` | AOI bboxes, centres and geometry provenance. |
| `preregistration_candidate.json` | Machine-readable contract; every field carries `decision_status: "accepted"`. |

---

## Note on file count

Section 15 of the task specification lists **16** files (11 Markdown, 4 CSV,
1 JSON); section 18 refers to "the expected 14 files". All 16 files from the
section-15 list were produced. The discrepancy is in the task text, not in the
output.

---

## The settled contract

`10_decisions_required.md` records **12 decisions (17 rows when split)**, and
**all of them are accepted**:

```
implementation_blocker_count = 0
new GEE export required       = yes, climate component only, AUTHORISED
new model fit required        = no
new bootstrap required        = no
new dependency required       = yes, geographiclib (user-run install)
```

The four former blockers and their outcomes:

1. **B-1 — feature importance.** Accepted: Step8B RandomForest **impurity**
   importance, `(burnable_tree_shrub_grass, thermal)`. No new permutation fit;
   held-out permutation importance stays available as a later sensitivity.
2. **C-8 — climate source, period, variables.** Accepted: TerraClimate,
   1991–2020, warm season months 6–9, **four** variables. No ERA5-Land
   cross-check in this run.
3. **C-9 — climate export.** **Authorised.**
4. **C-11 — geodesic route.** Accepted: the pinned **`geographiclib`** package.
   No custom Vincenty, no haversine fallback.

### Two contracts corrected in the same round

| | Superseded | Accepted |
|---|---|---|
| **DI normaliser** | mean source *holdout nearest-neighbour* distance | **mean pairwise weighted distance over all distinct source reference cell pairs**, self-distance excluded, categorical term included, fold-free |
| **AOA threshold** | 0.95 quantile of training DI | **upper whisker** `min(max(training_DI), Q3 + 1.5·IQR)`; q95 demoted to a reported secondary that never classifies |

The normaliser correction matters because a nearest-neighbour denominator on a
dense 500 m grid measures grid spacing rather than source spread, and would make
DI incomparable between a 9 298-cell AOI and a 41 731-cell one. Spatial folds are
retained, but only for the training DI and the threshold.

### Two further narrowings

- The climate vector went from six variables to **four**: *warm-season mean
  temperature* and *warm-season precipitation* were removed as near-duplicates of
  the annual axes beside them.
- The geographic component dropped its population-centroid secondary entirely, so
  it now reads **no Step8A data of any kind**.

### Remaining operational steps (not decisions)

1. Run the authorised climate export.
2. Install `geographiclib` — reviewed separately as a user-run step.

Neither blocks the design.
