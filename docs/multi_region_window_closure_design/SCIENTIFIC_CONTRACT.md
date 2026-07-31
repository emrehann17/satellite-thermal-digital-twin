# Scientific Contract — `multi_region_window_closure.v1`

Every clause below was **verified against `src/window_closure_sensitivity.py`, `core/config.py`,
`core/regions.py` and the frozen Manavgat artefacts**, not copied from prose documentation.
Line references are to the file at commit `483027a`.

---

## 1. Scientific question

> For each AOI independently: when the predictor window is closed 7 and 14 days earlier
> than its canonical closing date — with window length, label window, event dates and gate
> dates all held fixed — is the incremental contribution of the thermal model over the
> baseline model preserved?

This is a **predictor-timing sensitivity** question. It is retrospective and descriptive.
It is not an operational forecasting validation and establishes no causal mechanism.

---

## 2. Population

```
burnable_tree_shrub_grass
```

Source: `src/window_closure_sensitivity.py:81` (`PRIMARY_POPULATION`).
Applied by `build_model_common_cohort` (line 7734) as a boolean column filter, on top of
`valid_for_modeling` and `analysis_eligible`.

---

## 3. AOIs and their roles

| `aoi` | Role | Treatment |
|---|---|---|
| `manavgat_2021` | Anchor / reference | **Read-only.** Never refit, never re-exported, never overwritten. |
| `bejis_2022` | New actual AOI | Full three-variant run |
| `mugla_2021` | New actual AOI | Full three-variant run |
| `evia_2021_extended` | New actual AOI **and different-regime control** | Full three-variant run |
| `evia_2021` | **EXCLUDED** | Must not appear anywhere in the analysis |

---

## 4. Estimands

### 4.1 Primary estimand

For each `(aoi, variant)`:

```
thermal ROC-AUC − baseline ROC-AUC
```

Computed on the AOI's exact common cohort, from pooled out-of-fold predictions.

### 4.2 Secondary estimands

For each `(aoi, variant)`:

```
thermal PR-AUC − baseline PR-AUC
thermal Brier   − baseline Brier
```

And, within each model family, the shifted-vs-canonical changes:

```
shifted thermal  − canonical thermal      (for each of ROC-AUC, PR-AUC, Brier)
shifted baseline − canonical baseline     (for each of ROC-AUC, PR-AUC, Brier)
```

And the change of the contribution itself:

```
(thermal − baseline)_shifted − (thermal − baseline)_canonical
```

### 4.3 Comparison families

These map exactly onto the three families the Manavgat implementation already emits
(`src/window_closure_sensitivity.py:7218-7220`):

| `comparison_family` | Definition | Existing constant |
|---|---|---|
| `thermal_contribution_within_variant` | `thermal − baseline`, inside one variant | `COMPARISON_THERMAL_CONTRIBUTION` |
| `closure_change_within_model_family` | `shifted − canonical`, inside one model family | `COMPARISON_CLOSURE_CHANGE` |
| `thermal_contribution_change` | `(thermal−baseline)_shifted − (thermal−baseline)_canonical` | `COMPARISON_CONTRIBUTION_CHANGE` |

### 4.4 Direction conventions

* ROC-AUC — **higher is better.**
* PR-AUC — **higher is better.** Prevalence-dependent; comparable only *within* an AOI,
  where prevalence is held fixed by the common cohort. **Never compared across AOIs.**
* Brier — **a loss; lower is better.**

---

## 5. Brier orientation contract

The Manavgat implementation records **both** orientations and never flips a sign silently
(`src/window_closure_sensitivity.py:1391-1399`, `BRIER_SIGN_CONVENTION` line 7207). The new
schema must keep them under **distinct, explicit names**:

| Field | Formula | Favourable sign | Existing counterpart |
|---|---|---|---|
| `difference_natural` | `thermal Brier − baseline Brier` | **negative** favours thermal | `delta_brier` |
| `difference_oriented` | `baseline Brier − thermal Brier` | **positive** favours thermal | `brier_improvement` (= `−delta_brier`) |
| shifted-window natural | `shifted Brier − canonical Brier` | negative = shifted has lower loss | `closure_change_within_model_family` on `brier` |
| shifted-window oriented | `canonical Brier − shifted Brier` | **positive** = shifted has lower loss | (new explicit field) |

For ROC-AUC and PR-AUC, `difference_natural == difference_oriented` (higher is better in
both readings); both columns are still emitted so the schema is uniform and no consumer has
to branch on metric name.

**Rule:** every stored, rendered or reported Brier number must carry its `orientation`
alongside it. A number without an orientation label is a validator FAIL (`M06`, `M07`).

---

## 6. Inference unit — strictly per AOI

* Each AOI has its **own** common cohort, its **own** fold assignment, its **own** bootstrap
  draw plan, and its **own** confidence intervals.
* **There is no pooled bootstrap, no pooled point estimate, no pooled confidence interval,
  and no meta-analytic combination** across AOIs.
* `four_region_synthesis.csv` places per-AOI numbers side by side. It contains no column
  that is a function of more than one AOI's data.

---

## 7. Frozen model contract

Verified from `model_frozen_configuration` (line 7387), `model_feature_registry` (line 7436)
and `core/config.py`.

| Element | Value | Source |
|---|---|---|
| Estimator | `random_forest` | `PRIMARY_MODEL` (line 82) |
| Baseline features (in order) | `ndvi_mean`, `elevation_mean`, `slope_mean`, `landcover_dominant` | `src.step8b_train_baseline_vs_thermal_model.BASELINE_FEATURES` |
| Thermal features (in order) | baseline four **+** `lst_anomaly_mean`, `current_lst_mean`, `current_tvdi_mean`, `tvdi_difference_mean`, `downscaled_lst_mean`, `fused_lst_mean` | `THERMAL_MODEL_FEATURES` |
| Categorical features | `landcover_dominant` | `CATEGORICAL_FEATURES` |
| Target | `burned` | `TARGET_COLUMN` |
| `n_splits` | 5 | `core/config.py:555` |
| Fold random seed | 42 | `core/config.py:554` |
| Spatial block size | 2 cells (≈1 km at 500 m grid) | `core/config.py:556` |
| `min_positives` | 30 | `core/config.py:557` |
| `strict_folds` | `True` | `model_frozen_configuration` |
| Calibration | `None` — no calibration is applied | `model_frozen_configuration` |
| Adaptation | `None` — no domain adaptation | `model_frozen_configuration` |
| Class weighting | Whatever `step8b.train_population` already does; **not overridden** | `src/step8b_train_baseline_vs_thermal_model.py` |
| Preprocessing / imputation | Whatever the production pipeline in `train_population` does; **not overridden**. Rows missing any feature-union value are removed by the cohort gate, so no imputation is introduced. | `build_model_common_cohort` |
| Probability output | `train_population`'s out-of-fold `predict_proba` positive-class column | `oof_prob_baseline` / `oof_prob_thermal` |
| Fit/eval separation | Grouped spatial-block CV; each row scored exactly once, out of fold | `build_shared_spatial_folds` (line 7895) |

**The production routine `step8b.train_population` is *called*, never reimplemented**
(`fit_variant_models`, line 7993). No new tuning, no AOI-specific optimisation, no new model
family, no new feature family may be introduced.

`model_frozen_configuration` **fails closed** if any of the eight required config constants
is absent — it refuses to substitute a default.

---

## 8. Frozen cohort contract — Structure A

The Manavgat implementation uses **Structure A: exact complete-case intersection across all
three variants**, verified in `build_model_common_cohort` (line 7734):

Per AOI, in order:

1. Start from each variant's Step8A frame.
2. Keep rows with `valid_for_modeling` **and** `analysis_eligible`.
3. Keep rows in `burnable_tree_shrub_grass`.
4. Remove cells in the shared pre-label censor set (one raster, one interval, all variants).
5. Remove rows missing **any** feature in the baseline ∪ thermal feature union.
6. Take the **exact `cell_id` set intersection** across all three variants.

Then assert, and **fail** rather than silently drop, on:

* identical `cell_id` ordering across variants,
* **label invariance** — same cell carries the same `burned` in every variant,
* **static invariance** — every window-independent column identical to `1e-9` absolute
  tolerance (`STATIC_INVARIANCE_ABS_TOLERANCE`, line 6108),
* no duplicate `cell_id`,
* both classes present.

Manavgat evidence (`model/common_cohort/common_cohort_metadata.json`):

```
initial_rows_by_variant           canonical/7d/14d = 24150 / 24150 / 24150
rows_present_in_all_variants      19406
final_positive_rows               744
final_negative_rows               18662
prevalence                        0.038339
removed_label_mismatch            0     (a failure, never a removal)
removed_static_invariance_failure 0     (a failure, never a removal)
```

The shared pre-label censor interval is derived generically and is **independent** of the
per-experiment `exclude_pre_label_burns` flag (`common_prelabel_interval`, line 357):

```
start = min(variant predictor_start) over all preregistered shifts
end   = label_start − 1 day
```

**Failure mode:** if any two variants of the same AOI do not end on exactly the same row set,
the run must stop with

```
BLOCKER: VARIANT_COHORT_MISMATCH
```

---

## 9. Frozen fold contract

Verified from `build_shared_spatial_folds` (line 7895) and `fit_variant_models` (line 7993).

* **One** fold assignment per AOI, built **once** on the common cohort.
* Blocks: `block_row = row_500m // 2`, `block_col = col_500m // 2`, fixed origin
  (`step8b.add_spatial_block_id`).
* Splits: `step8b.make_spatial_folds(labels, groups, n_splits=5, seed=42, strict=True)`.
* Asserted invariants, each of which raises rather than warns:
  * no row assigned to more than one validation fold,
  * no row left unassigned,
  * no spatial block split across folds,
  * every variant's own derived `fold_id` array is `np.array_equal` to the shared one —
    **this is the mechanism that makes fold reuse enforced, not merely intended.**
* The assignment is hashed: `assignment_sha256` over canonical-JSON of the sorted
  `{cell_id: fold_id}` map.

Manavgat evidence (`model/shared_folds/shared_spatial_folds_metadata.json`):

```
fold_count            5
unique_block_count    5350
rows_per_fold         3880 / 3880 / 3882 / 3883 / 3881
positives_per_fold    148 / 148 / 150 / 150 / 148
negatives_per_fold    3732 / 3732 / 3732 / 3733 / 3733
assignment_sha256     28d79449ed7a63fe2fca4f3e3b35f47ae6cb5b970b0065965b5b61e9dcc81689
```

**Folds are never re-optimised per variant.** Any evaluation fold lacking both classes, or
any AOI whose cohort cannot yield 5 feasible folds, stops the run with

```
BLOCKER: FOLD_CLASS_INFEASIBILITY
```

---

## 10. Frozen bootstrap contract

Verified from `multi_variant_block_bootstrap` (line 1316), `percentile_interval` (line 1412)
and `validate_saved_bootstrap_replicate_counts` (line 1427).

| Element | Value |
|---|---|
| Unit | `spatial_block_id` — the same blocks as the folds |
| Block definition | `step8b.add_spatial_block_id`, block size 2 cells |
| Draw | Resample all unique blocks **with replacement**, `size = len(unique_blocks)` |
| Requested replicates | 1000 (`STEP8C_N_BOOTSTRAP`) |
| Seed | 42 (`STEP8C_RANDOM_SEED`) |
| Interval method | **Percentile**, 2.5 / 97.5 → 95 % (`STEP8C_CI_LOWER` / `STEP8C_CI_UPPER`). Not BCa. |
| Pairing | **One draw per replicate, scored for every variant and both families.** `identical_block_draws_across_variants = True`. Differences are formed at replicate level. |
| Refit | **None.** Replicates rescore stored out-of-fold predictions via `step8c.compute_metrics`. |
| Invalid replicate | A draw whose resampled labels are single-class, or where `compute_metrics` returns `None`. Dropped globally — the artefact holds one row per **globally valid** draw. |
| Minimum valid replicates | 1 (reuses `step8c.summarize_bootstrap`'s existing `available = n_successful > 0` rule; no new threshold is invented) |
| Accounting rule | `valid == len(replicate_table)` and `invalid == requested − valid`, both asserted |
| Minimum blocks | ≥ 2, else hard failure — a row bootstrap is explicitly not an acceptable substitute |

Manavgat evidence (`model/bootstrap/paired_bootstrap_summary.csv`):
`requested_replicates = 1000`, `valid_replicates = 1000`, `invalid_replicates = 0`,
`block_count = 5350`, `confidence_level = 95.0`.

**The draw plan is per AOI.** Every comparison inside one AOI shares that AOI's single draw
plan, hashed as `draw_plan_hash`. Draw plans are never shared across AOIs, because AOIs have
different block populations.

---

## 11. Interpretation rules

### 11.1 Interval classification

Reuse the three existing statuses verbatim (`src/window_closure_sensitivity.py:198-200`):

| Status | Meaning |
|---|---|
| `bootstrap_supported_increase` | The 95 % percentile interval lies entirely above zero |
| `bootstrap_supported_decrease` | The 95 % percentile interval lies entirely below zero |
| `interval_includes_zero` | The interval spans zero — **direction unresolved; uncertainty remains** |

An interval that includes zero is **not** evidence of equivalence. No equivalence margin is
preregistered in this analysis, so none may be claimed.

### 11.2 Mandatory Evia framing

Every artefact, table, summary and report section that names `evia_2021_extended` must
present it as:

```
different-regime control
high-prevalence sensitivity region
not an equal-prevalence fourth validation region
```

Evidence for the framing (measured, not asserted):

| AOI | Cohort prevalence (canonical, complete-case) |
|---|---|
| `manavgat_2021` | 0.0385 |
| `bejis_2022` | 0.0707 |
| `mugla_2021` | 0.0706 |
| **`evia_2021_extended`** | **0.2882** — ~4–7× the others |

Consequences that must be stated, not merely implied:

* PR-AUC baselines differ by construction with prevalence, so Evia's PR-AUC values are not
  comparable in level to the other three.
* Evia-extended is a geographically **extended** counterpart of `evia_2021`, selected from
  place coverage only — never tuned on labels, prevalence, gate outcome or any metric
  (`core/regions.py:447-462`).

### 11.3 Technical PASS vs scientific support

These are **separate verdicts** and must never be merged:

* **Technical status** — artefacts complete, hashes match, arithmetic reproduces, namespace
  safe. Reported by the validator.
* **Scientific summary** — what the bootstrap intervals do and do not support, per AOI, per
  comparison, per metric.

A technical PASS says nothing about the scientific conclusion; a bootstrap-supported result
says nothing about technical integrity. `summary.json` keeps them in different top-level keys
(`technical_status`, `scientific_summary`).

---

## 12. Permitted and prohibited language

### 12.1 Permitted

```
bootstrap-supported
interval excludes zero
interval includes zero
uncertainty remains
point estimate
descriptive
direction-dependent
under this frozen design
```

### 12.2 Prohibited — union of two lists

The new validator must enforce the **union** of the task's prohibitions and the existing
`FORBIDDEN_COMPARE_PHRASES` (`src/window_closure_sensitivity.py:8863`), because neither list
is a superset of the other.

| Phrase | In existing guard | In task list |
|---|---|---|
| `statistically significant` | ✅ | ✅ |
| `significant difference` | ✅ | — |
| `non-significant`, `insignificant` | ✅ | — |
| `p-value`, `p value`, `hypothesis test` | ✅ | — |
| `equivalent`, `equivalence` | ✅ | — |
| `unchanged`, `stable`, `robust` | ✅ | — |
| `statistically proven` | — | ✅ |
| `proven` | — | ✅ |
| `causal` | — | ✅ |
| `operationally validated` | — | ✅ |
| `leakage eliminated` | — | ✅ |
| `optimal` | — | ✅ |
| `best window` | — | ✅ |

Also retained: `FOREIGN_FACTOR_PHRASES` (line 148) — wording inherited from the compositing
counterfactual (`"compositing method is the only"`) must never appear, because this analysis
moves predictor timing, not the reducer.

---

## 13. Limitations (mandatory in `report.md`)

Carried forward from the Manavgat frozen `LIMITATIONS` list and extended for the multi-region
setting:

1. Closing the predictor window earlier is a **predictor-timing sensitivity** analysis. It is
   retrospective and is not an operational forecasting validation.
2. Three new AOIs and one reference AOI, each a single season, are not evidence of general
   generalisability.
3. A confidence interval that includes zero is **not** evidence of equivalence; no
   equivalence margin is preregistered.
4. Any performance change is consistent with the predictor and observation-support changes
   that follow from the closure date; it does not establish a causal mechanism.
5. Landsat and MODIS scene/observation support themselves change with the closure date, so
   support differences are part of what is being measured.
6. The fixed production MODIS calendar-month filter (months 6–9) clips shifted windows by
   **AOI-specific** amounts — Bejís 0/0 days, Evia-extended 3/10 days, Manavgat and Muğla
   7/14 days. The analysis therefore measures the closure date **together with** its
   interaction with that fixed policy, not the closure date in isolation.
7. PR-AUC depends on prevalence. Every comparison is made on the same common cohort *within*
   an AOI, so prevalence is fixed there; PR-AUC levels are **not** comparable *between* AOIs.
8. The label window, event dates and gate dates are frozen and identical across variants;
   only predictor timing moves.
9. Results are descriptive. Each AOI is analysed separately; there is no pooled estimate and
   no meta-analytic combination.
10. `evia_2021_extended` is a high-prevalence different-regime control, not an
    equal-prevalence fourth validation region.
11. No deployment, alerting or future-fire forecasting claim is made or supported.
