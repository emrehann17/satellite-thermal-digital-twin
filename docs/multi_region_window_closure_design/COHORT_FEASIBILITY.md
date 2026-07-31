# Cohort and Fold Feasibility

All counts below were **measured** from the canonical Step8A parquet files on disk, applying
the same filter sequence the frozen `build_model_common_cohort` applies. No number is
extrapolated except where explicitly marked *projected*.

---

## 1. The Manavgat cohort contract — Structure A, confirmed

The task offered two admissible structures:

* **A)** complete-case intersection cohort across the three variants
* **B)** frozen canonical cohort + per-variant fail-closed completeness

**The repository implements A.** `build_model_common_cohort`
(`src/window_closure_sensitivity.py:7734`) computes an exact `cell_id` set intersection over
the three variants' independently-filtered frames. Its docstring is unambiguous:

> "ONE exact common cohort shared by all six evaluations. … the N-way exact intersection with
> its cross-variant label, coordinate and population equality gates … a cell must carry every
> baseline and thermal model feature in EVERY variant, otherwise the six evaluations would not
> be scored on the same rows."

Evidence that it is A and not B: `removed_variant_only_keys` is **non-zero for the canonical
variant** in the Manavgat metadata (752 rows). Under structure B the canonical cohort would be
frozen and could lose nothing. Under A, cells present in canonical but absent from a shifted
variant are dropped from *all* arms — which is exactly what the artefact records.

### 1.1 The filter sequence (frozen, applied identically per AOI)

| # | Step | Failure mode if violated |
|---|---|---|
| 1 | `valid_for_modeling == True` | — |
| 2 | `analysis_eligible == True` | — |
| 3 | `burnable_tree_shrub_grass == True` | — |
| 4 | Remove shared pre-label censored `cell_id`s | — |
| 5 | Remove rows with any NA in the baseline ∪ thermal feature union (10 features) | — |
| 6 | **Exact `cell_id` intersection across all three variants** | `VARIANT_COHORT_MISMATCH` |
| 7 | Assert identical `cell_id` ordering | raises |
| 8 | Assert **label invariance** across variants | raises — never a silent drop |
| 9 | Assert **static invariance** (atol `1e-9`) | raises — never a silent drop |
| 10 | Assert no duplicate `cell_id` | raises |
| 11 | Assert both classes present | raises |

Steps 8 and 9 are the design's strongest guarantee: the metadata fields
`removed_label_mismatch` and `removed_static_invariance_failure` are **structurally always
zero**, because a mismatch raises instead of removing. A non-zero value would itself be a bug.

### 1.2 The shared pre-label censor

`common_prelabel_interval` (line 357), one interval per AOI, one raster, applied identically
to all three variants:

```
start = min(variant predictor_start) over all preregistered shifts   (= canonical_start − 14)
end   = label_start − 1 day
```

Explicitly **independent of** the per-experiment `exclude_pre_label_burns` flag
(`independent_of_exclude_pre_label_burns_flag: true`). Rationale, from the source: shifting
the closure date earlier opens a gap between `predictor_end` and `label_start`; cells burning
in that gap are not label-window positives, and leaving them as negatives would be wrong.

| AOI | Censor interval | Registry `pre_label_burn_window` | Relationship |
|---|---|---|---|
| `manavgat_2021` | 2021-05-18 .. 2021-07-27 | *(absent)* | Analysis-only |
| `bejis_2022` | 2022-06-01 .. 2022-08-14 | *(absent)* | Analysis-only |
| `mugla_2021` | 2021-05-18 .. 2021-07-28 | 2021-06-01 .. 2021-07-28 | **Strict superset** (14 extra days) |
| `evia_2021_extended` | 2021-05-22 .. 2021-08-02 | 2021-06-05 .. 2021-08-02 | **Strict superset** (14 extra days) |

For Muğla and Evia-extended the analysis censor reaches 14 days further back than the
canonical Step8A pre-label exclusion. Cells burning in that extra fortnight are censored here
but were not censored canonically. This is intended generic behaviour and **must be counted
and reported** (`removed_prelabel_censor` per variant), never silently absorbed.

Manavgat measured `removed_prelabel_censor = {canonical: 0, 7d: 0, 14d: 0}` — no cell burned
in its pre-label window. **The new AOIs are expected to be non-zero**, particularly Muğla,
whose registry notes a known Bördübet/Marmaris fire around 2021-06-21..25 inside the predictor
window. That is the expected, correct behaviour of the leakage-safe contract, not an anomaly.

### 1.3 Same approach mandated for the new analysis

Per AOI, `canonical`, `close_7d_earlier` and `close_14d_earlier` are evaluated on **exactly
the same row set and the same `cell_id` values**. If any residual cohort difference remains
inside an AOI, the actual run must not start:

```
BLOCKER: VARIANT_COHORT_MISMATCH
```

Cohorts are **not** shared across AOIs — each AOI has its own, as required by per-AOI
inference.

---

## 2. Measured canonical feasibility, per AOI

Computed read-only from `outputs/experiments/<aoi>/step8a/step8a_500m_modeling_dataset.parquet`
by applying filters 1–3 and 5 (the pre-label censor and the intersection require the shifted
variants, which do not exist yet):

| AOI | Step8A rows | after valid/eligible | after population | **complete-case** | **positives** | **prevalence** |
|---|---|---|---|---|---|---|
| `manavgat_2021` (ref) | 24,150 | 24,087 | 20,511 | 20,158 | 776 | 0.0385 |
| `bejis_2022` | 15,759 | 15,759 | 15,190 | **14,210** | **1,005** | **0.0707** |
| `mugla_2021` | 73,098 | 73,045 | 41,730 | **40,389** | **2,853** | **0.0706** |
| `evia_2021_extended` | 22,925 | 22,906 | 9,298 | **8,877** | **2,558** | **0.2882** |

All ten feature-union columns (`ndvi_mean`, `elevation_mean`, `slope_mean`,
`landcover_dominant`, `lst_anomaly_mean`, `current_lst_mean`, `current_tvdi_mean`,
`tvdi_difference_mean`, `downscaled_lst_mean`, `fused_lst_mean`) are **present in all four
AOI datasets** — no missing column in any AOI.

### 2.1 Projected common-cohort size

Manavgat's attrition from canonical complete-case to final intersection was
`19,406 / 20,158 = 96.3 %` of the complete-case set (and 80.4 % of raw Step8A rows). Applying
the same retention band (90–97 %):

| AOI | complete-case | **projected common cohort** | **projected positives** | ≥ `min_positives` (30) |
|---|---|---|---|---|
| `bejis_2022` | 14,210 | 12,800 – 13,800 | ~900 – 975 | ✅ ~31× margin |
| `mugla_2021` | 40,389 | 36,400 – 39,200 | ~2,570 – 2,770 | ✅ ~88× margin |
| `evia_2021_extended` | 8,877 | 8,000 – 8,600 | ~2,300 – 2,480 | ✅ ✅ ~80× margin |

These are **projections**, explicitly labelled. The authoritative values are produced by the
`cohort-feasibility` gate (§5) and must be re-checked there before any fit.

### 2.2 Evia-extended's population filter is the outlier

Note the population step for Evia-extended: 22,906 → 9,298, i.e. only **41 %** of eligible
cells are burnable tree/shrub/grass, versus 85 % (Manavgat), 96 % (Bejís), 57 % (Muğla). The
extended AOI deliberately adds surrounding terrain and the Limni–Rovies–Agia Anna–Pefki
corridor, much of which is not burnable vegetation. Combined with a large burned area, this is
what produces the 0.288 prevalence. It is a property of the AOI definition — which was chosen
from geographic place coverage only, never tuned on labels or metrics
(`core/regions.py:447-462`) — and not a data defect.

---

## 3. The fourteen feasibility checks

Run as a distinct `cohort-feasibility` stage, **after** local-downstream and **before** any
fit. All must PASS for all three AOIs.

| # | Check | Threshold | Status now |
|---|---|---|---|
| 1 | Canonical cohort row count | > 0 | ✅ measured, §2 |
| 2 | Canonical positive count | ≥ `min_positives` = 30 | ✅ 1,005 / 2,853 / 2,558 |
| 3 | Canonical prevalence | recorded; no threshold | ✅ 0.071 / 0.071 / 0.288 |
| 4 | Temporal feature completeness per shifted variant | recorded per variant | ⏳ after export |
| 5 | Common complete rows across 3 variants | > 0 | ⏳ projected §2.1 |
| 6 | Dropped-row difference between variants | recorded; no silent asymmetry | ⏳ |
| 7 | **Label invariance** | exact equality | structural — raises |
| 8 | **Static predictor invariance** | atol `1e-9` | structural — raises |
| 9 | Grid-cell identity invariance | identical `cell_id` arrays | structural — raises |
| 10 | Duplicate row / grid-ID check | zero duplicates | structural — raises |
| 11 | Both classes in every evaluation fold | strict | ✅ projected, §4 |
| 12 | Minimum positives and blocks | pos ≥ 30; blocks ≥ 2 | ✅ large margins |
| 13 | Evia high-prevalence effect on metrics and folds | documented | ✅ §4.2, §6 |
| 14 | Partial AOI / partial variant risk | none tolerated | ✅ gate design §5 |

---

## 4. Fold feasibility

Contract (`build_shared_spatial_folds`, line 7895): 5 grouped folds, block size 2 cells,
seed 42, `strict=True`, blocks never split across folds, every row assigned exactly once.

### 4.1 Projected block and fold counts

Manavgat: 19,406 cohort rows → 5,350 unique blocks (≈ 3.6 rows/block, since ~2×2-cell blocks
are partly incomplete at cohort boundaries).

| AOI | projected cohort rows | projected blocks | projected positives/fold |
|---|---|---|---|
| `bejis_2022` | ~13,300 | ~3,600 | ~185 |
| `mugla_2021` | ~37,800 | ~10,400 | ~535 |
| `evia_2021_extended` | ~8,300 | ~2,300 | ~478 |

Manavgat's realised per-fold positives were 148–150 with 3,880–3,883 rows/fold, and folds were
near-perfectly balanced. Every projected AOI has **more** positives per fold than Manavgat, so
5-fold feasibility is comfortable everywhere.

Minimum blocks for the paired bootstrap is 2; the smallest projection is ~2,300. Margin is
three orders of magnitude.

### 4.2 Evia-extended's high prevalence — effect on folds and metrics

| Effect | Assessment |
|---|---|
| Fold class feasibility | **Improved.** ~2,400 positives across 5 folds ⇒ ~478/fold. Single-class folds are effectively impossible. |
| ROC-AUC | Well-defined and comparable in *form*; magnitudes still not compared across AOIs. |
| PR-AUC | **Baseline PR-AUC ≈ prevalence ≈ 0.288**, versus ≈ 0.039–0.071 elsewhere. Levels are structurally incomparable across AOIs. Only *within-AOI* differences are interpretable. |
| Brier | Sensitive to base rate; same restriction applies. |
| Bootstrap | Unaffected mechanically; the block resample retains enough positives in every draw, so `invalid_replicates` should stay near zero. |

**No fold infeasibility is anticipated for any AOI.** Should one arise:

```
BLOCKER: FOLD_CLASS_INFEASIBILITY
```

### 4.3 Fold invariance requirements

* Exactly **one** fold mapping per AOI, built on that AOI's common cohort.
* All three variants use it — enforced by `np.array_equal` against every variant's own derived
  `fold_id` in `fit_variant_models` (line 7993), which raises on mismatch.
* Folds are **never** re-optimised per variant.
* `assignment_sha256` is recorded and must be **identical for all three variants of one AOI**,
  and **different between AOIs** (different cohorts ⇒ different assignments). Both directions
  are checked (`F02`, `F06`).

---

## 5. The `cohort-feasibility` gate

A distinct fail-closed stage between `local-downstream` and `fit`.

```
plan → export → local-downstream → [cohort-feasibility] → fit → compare → summarize
```

| Property | Value |
|---|---|
| Inputs | Per-AOI, per-variant Step8A frames; the AOI's pre-label censor raster; `model_feature_registry` |
| Outputs | `cohort_inventory.csv`, `fold_mapping.parquet`, `stages/cohort-feasibility.json` |
| Earth Engine | **Never** |
| Model fitting | **Never** |
| Success | All 14 checks PASS for **all three** AOIs |
| Failure | Any check fails for any AOI ⇒ stage FAIL; `fit` may not start for **any** AOI |
| Resume | Only when config hash, input hashes and output manifest all match |
| Quarantine | On `--force`, into `<namespace>/_quarantine/<timestamp>_<reason>/` |

**All-or-nothing rule:** the gate evaluates the complete set of 3 AOIs × 3 variants. A subset
never passes. If Muğla fails, Bejís and Evia-extended do not proceed either — this prevents a
partial result set from being written and later mistaken for a complete one.

Because the gate is what first materialises the shifted cohorts, it is also the point at which
the projections in §2.1 are replaced by measured values.

---

## 6. Findings and warnings

| ID | Finding | Severity | Handling |
|---|---|---|---|
| `CF-1` | All three new AOIs have ample positives (1,005 / 2,853 / 2,558 vs `min_positives` = 30) | Informational | — |
| `CF-2` | Muğla's cohort is ~2× Manavgat's, driving runtime and storage | Warning | Sized in `EXPORT_FEASIBILITY.md` §6–§7 |
| `CF-3` | Evia-extended prevalence 0.288 ≈ 4–7× the others | **Warning** | Mandatory different-regime framing; no cross-AOI PR-AUC comparison |
| `CF-4` | Evia-extended retains only 41 % of eligible cells as burnable population | Informational | Explained by the deliberately extended AOI; report the count |
| `CF-5` | Muğla and Evia-extended censor intervals are strict supersets of their registry `pre_label_burn_window` | **Warning** | Expected; count and report `removed_prelabel_censor` per variant |
| `CF-6` | Manavgat's `removed_prelabel_censor` was 0; the new AOIs will likely be non-zero | Informational | Do not treat non-zero as an anomaly |
| `CF-7` | All 10 feature-union columns present in all 4 AOI datasets | Informational | Confirms schema alignment |
| `CF-8` | Common-cohort sizes in §2.1 are **projections** | **Warning** | Replaced by measured values at the `cohort-feasibility` gate; never quoted as results |

**Blockers: none.** Cohort and fold feasibility is satisfied for all three new AOIs on
currently available evidence.
