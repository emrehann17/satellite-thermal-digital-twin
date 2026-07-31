# Scientific Contract — `mugla_subsampling.v1`

Every decision below is **frozen**. There is no unresolved scientific choice in
this design. Anything an implementer might otherwise have to decide is decided
here, with the value written out.

---

## 1. Diagnostic class

```
DIAGNOSTIC_CLASS = "population_size_matched_subsampling_sensitivity"
```

This is a *sensitivity* analysis over one manipulated quantity — the total
number of Muğla modeling rows. It is not a causal decomposition, not an
ablation of regional structure, and not an operational deployment study.

## 2. Fixed scope

| Item | Frozen value |
|---|---|
| Primary population | `burnable_tree_shrub_grass` |
| Valid universe | `valid_for_modeling == True` |
| AOIs | `manavgat_2021`, `bejis_2022`, `mugla_2021` |
| Excluded | `evia_2021`, `evia_2021_extended`, `kozan_2023` — hard exclusion, asserted |
| Model families | `baseline`, `thermal` |
| Model | `random_forest` via canonical `build_pipeline` / `build_classifier` |
| Estimator seed | 42 (`STEP8B_RANDOM_SEED`) |
| Primary metric | ROC-AUC |
| Secondary metrics | PR-AUC, Brier score |
| Subsampled region | `mugla_2021` **only** |

**Canonical Step8A hashes — verified 2026-08-03, all three match:**

```
manavgat_2021  054a1961fc0582a33d36413263668b63074b21ae8b03d12269b6e228787f3439
bejis_2022     3dec785a7d8e31db2d67ed283546bbfbca1559f56df46663488d0afc24d9e393
mugla_2021     c4ab107db2207f9f20775ccc0b3bf39381173fd07d4e82f6821ce7f40be7db8e
```

No new model, no new feature, no adaptation, no hyperparameter tuning, no
threshold selection. The canonical feature contract
(`SHARED_BASELINE_FEATURES` = 4 columns, `SHARED_THERMAL_MODEL_FEATURES` = 10
columns) is used unchanged.

## 3. Subsampling contract

### 3.1 Arm identity

Exactly one subsampling arm:

```
arm_id = "size_matched_to_manavgat"
target_sample_size = 20511
n_repeats = 20
```

`20511` is not a parameter to be recomputed at runtime from Manavgat's frame —
it is a frozen constant, and the implementation additionally **asserts** that
Manavgat's primary population equals it. If Manavgat's Step8A ever changes, the
run fails closed rather than silently re-targeting.

### 3.2 Sampling rule

- Without replacement from the Muğla primary population.
- Exactly 20,511 **unique** rows per repeat.
- Each repeat uses an independent deterministic seed.
- Row order of the input frame must not change the selected set.

### 3.3 Spatial strata

```
assign_large_blocks(df, 10)          # 10 × 500 m ≈ 5 km, fixed (0,0) origin
stratum_id = f"{large_block_id}|L{label}"
```

Block assignment happens **before** the valid/population filter, matching the
canonical utility's contract. Strata are the cross product of 10-cell block and
binary label. Observed: 576 blocks, 636 non-empty strata (566 label-0, 70
label-1).

### 3.4 Allocation — Hamilton / largest remainder, integer-exact

For each stratum with capacity `c_s` out of `N_total = 41730`, target
`N = 20511`:

```
floor_s     = (c_s * N) // N_total          # integer arithmetic, no floats
remainder_s = (c_s * N) %  N_total          # exact integer remainder
shortfall   = N - Σ floor_s
```

The `shortfall` extra units go to the strata with the largest `remainder_s`.

**Tie-break (deterministic, and materially exercised):** sort by
`(-remainder_s, stratum_id)` ascending and award the first `shortfall` entries.
`stratum_id` is compared as a Python string. Integer remainders are used
specifically so that the tie set is exact and not a floating-point artefact —
the float and integer formulations were both computed and agree exactly, but
integer is the frozen rule.

Observed at `N = 20511`: `Σ floor_s = 20211`, `shortfall = 300`, 295 strata are
strictly above the cut, and the cut value is shared by **12 strata of capacity
5**, of which the tie-break awards exactly **5**. The tie-break is therefore
not decorative; it decides 5 of 20,511 rows and must be reproduced exactly.

**Guarantees, all verified on the real frame:**

- `Σ alloc_s = 20511` exactly.
- `alloc_s ≤ c_s` for every stratum — no stratum is over-drawn. Maximum
  observed `alloc_s - c_s` is 0; 12 strata (all capacity 1) reach
  `alloc_s == c_s`. This is structural, not luck: `floor_s ≤ c_s` always, and
  `floor_s = c_s` would require `N = N_total`.
- `alloc_s ≥ 1` for every stratum — no stratum is dropped, so all 576 blocks
  and both label classes survive in every repeat.

### 3.5 Prevalence contract

Allocation depends only on capacities, so it is **identical across all 20
repeats**. Consequently:

```
positives per repeat = 1438   (exactly, every repeat)
negatives per repeat = 19073  (exactly, every repeat)
subsample prevalence = 1438 / 20511 = 0.070108722…
full     prevalence  = 2911 / 41730 = 0.069757968…
absolute drift       = +0.00035075
relative drift       = +0.5028 %
```

The exact proportional (non-integer) positive count is 1430.8057; the +7.19
positive surplus is pure largest-remainder rounding across the 70 label-1
strata (each may gain at most one unit).

**Frozen rounding bound, asserted by the run and by the validator:**

```
| subsample_prevalence − full_prevalence | ≤ n_label1_strata / target_sample_size
                                          = 70 / 20511 = 0.003413
```

Observed drift 0.00035075 is well inside the bound. Muğla's prevalence is
"preserved within rounding limits" in exactly this sense and no stronger one.

### 3.6 Within-stratum selection

```
repeat_seed(repeat_id) = int.from_bytes(
    blake2b(f"mugla_subsampling.v1|size_matched_to_manavgat|{repeat_id}".encode(),
            digest_size=8).digest(), "big") % 2**32
```

Within stratum `s` for repeat `r`:

1. Take the stratum's cells, **sorted by `cell_id` ascending** (stable identity
   ordering — this is what makes the result row-order invariant).
2. Permute with `numpy.random.default_rng(stratum_seed)` where
   `stratum_seed = blake2b(f"{SCHEMA_VERSION}|{repeat_id}|{stratum_id}")` reduced
   to 32 bits, so that a stratum's draw depends on the repeat and on its own
   identity and on nothing else — not on iteration order over strata.
3. Take the first `alloc_s` cells. No replacement.

The selected `cell_id` set, its block provenance and its fold assignment are
written to `selected_cells.parquet` and hashed into the `plan` stage marker
**before any model is fitted**.

### 3.7 One sample serves all three arms

For a given `repeat_id`, the *same* 20,511 Muğla cells are used by the
within-region arm, the Muğla-as-source arm and the Muğla-as-target arm. There
is no per-arm resampling.

## 4. Fixed spatial evaluation

The 10-cell outer-fold construction of the few-shot analysis is **not** used
here. This study is compared directly against canonical Step8 / raw-transfer
metrics, so it reuses the existing Step8 spatial-fold contract.

**The full-Muğla block→fold mapping is loaded from a persisted artifact and
inherited unchanged by all 20 repeats.**

| Property | Frozen value |
|---|---|
| Block scale | 10 cells × 500 m, nominal `approximately_5_km` |
| Block column (frozen artifact) | `spatial_block_id`, values `block10_{row}_{col}` |
| Block column (this analysis) | `large_block_id`, values `b10_r{row}_c{col}` |
| Fold column | `fold_id` |
| Fold count | 5 (`STEP8B_N_SPLITS`) |
| Seed | 42 (`STEP8B_RANDOM_SEED`) |
| Splitter | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` |
| Strictness | `strict_folds=True` |
| Persisted artifact | `outputs/experiments/mugla_2021/robustness/step8_big_blocks/block_10_cells/oof_predictions.parquet` (`fold_id` column, 41,730 rows, sha256 `e16e6b18…9df8cd4`) |

The two block-id spellings are a **bijective relabelling of the same
partition** — verified: 576 ↔ 576, `large_block_id → spatial_block_id` and
`spatial_block_id → large_block_id` are both 1:1 on all 41,730 rows. The
implementation joins on `cell_id`, so the spelling difference never matters
numerically; the bijection is asserted anyway.

**Fold rules:**

1. The mapping is loaded once from the artifact, keyed by `cell_id`.
2. Every repeat inherits it. Folds are **never re-optimised per repeat.**
3. No block spans two folds — verified, 0 of 576 blocks span folds.
4. Every fold has both classes on both sides — verified deterministically
   (see `SAMPLING_FEASIBILITY.md` §5); this holds identically in every repeat
   because the allocation is repeat-invariant.
5. Every sampled row receives exactly one OOF prediction.

**Reproduction fallback, if the artifact were ever absent:** rebuild with
`add_spatial_block_id(df_all, 10, column_name="big_block_id", id_prefix="block10", include_row_col=True)`
before the valid/population filter, then `filter_valid_for_modeling`,
`build_population_masks`, `.loc[masks["burnable_tree_shrub_grass"]].reset_index(drop=True)`,
then `make_spatial_folds(y, groups, 5, 42, strict=True)`. This is exactly what
`src/step8_big_block_robustness.run_big_block_condition` (line 542) does. The
implementation must **prefer the artifact** and record which path it took.

## 5. The three arms

### Arm A — within-Muğla size sensitivity

For each repeat × model family: 5-fold spatial OOF fit on the sampled Muğla
frame, using the inherited fold mapping, canonical preprocessing (inside the
`Pipeline`, so imputers/encoder are fitted on training rows only), canonical
feature list, canonical estimator seed.

**Reference:** the frozen full-Muğla 10-cell within-region OOF result.

| Family | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| baseline | 0.6979859420145867 | 0.16368744586018302 | 0.11271834078260733 |
| thermal | 0.7773268638729566 | 0.30192591578806804 | 0.07726667111613451 |

Source: `…/block_10_cells/step8b_metrics.json` (sha256 `a826279f…c50f34d1`).
These values were independently recomputed from the persisted OOF prediction
vectors and reproduce to machine precision.

### Arm B — Muğla as source

For each repeat: fit on the sampled Muğla frame, predict on the **full,
unchanged** target cohorts, threshold-free metrics only.

| Direction | Target rows | Target positives |
|---|---:|---:|
| `mugla_2021_to_manavgat_2021` | 20,511 | 784 |
| `mugla_2021_to_bejis_2022` | 15,190 | 1,100 |

**References** (`outputs/cross_region/<source>__<target>/step9b/cross_region_transfer_metrics.json`,
population `burnable_tree_shrub_grass`):

| Direction | Family | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| mugla → manavgat | baseline | 0.52151982986128 | 0.03783686447417549 | 0.19626287946345441 |
| mugla → manavgat | thermal | 0.40099966584697444 | 0.028738774210290162 | 0.17711744280608588 |
| mugla → bejís | baseline | 0.4507383379572875 | 0.06416586588786181 | 0.13096665141263947 |
| mugla → bejís | thermal | 0.5831912381443964 | 0.08827191688193281 | 0.11079303395449873 |

This arm measures **source training sample size** only. Targets never change.

### Arm C — Muğla as target

Source models are the **full** Manavgat and **full** Bejís models. The target
is the same 20,511 cells selected for that repeat.

**No model is refitted in this arm.** The canonical raw-transfer artifacts
persist a per-cell probability for every Muğla primary cell and both families:

```
outputs/cross_region/manavgat_2021__mugla_2021/step9b/cross_region_transfer_predictions.parquet
outputs/cross_region/bejis_2022__mugla_2021/step9b/cross_region_transfer_predictions.parquet
```

Verified: filtering to `transfer_direction == "<source>_to_mugla_2021"` and
`population == "burnable_tree_shrub_grass"` yields exactly 41,730 rows, zero
duplicate `target_cell_id`, and a `target_cell_id` set **identical** to the
Muğla primary `cell_id` set. Recomputing ROC-AUC / PR-AUC / Brier from those
columns reproduces the stored metrics to machine precision for all four
directions. Subsetting to a repeat's 20,511 cells and recomputing is therefore
exact and requires no fit.

**References:**

| Direction | Family | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| manavgat → mugla | baseline | 0.5079294316533508 | 0.07490551117572594 | 0.11569939725253452 |
| manavgat → mugla | thermal | 0.47015987108700774 | 0.06264647046501597 | 0.10468152081057548 |
| bejís → mugla | baseline | 0.5922389820175834 | 0.09363146017405545 | 0.08987774657814872 |
| bejís → mugla | thermal | 0.6184747489978263 | 0.09253859122869844 | 0.07892590354146907 |

This arm measures **target cohort size and composition** sensitivity. The
source training sample size is unchanged.

## 6. Reference and metric contract

For every arm × direction × family × metric × repeat, emit:

```
full_reference_value
subsample_value
natural_delta  = subsample_value - full_reference_value
oriented_delta
```

Orientation:

```
roc_auc, pr_auc : oriented_delta = subsample - full
brier_score     : oriented_delta = full - subsample
```

Brier is always reported in its natural (lower-is-better) form in the
`*_value` fields; only the *delta* is oriented. After orientation, for every
metric without exception:

```
oriented_delta > 0  →  the subsample result is better
oriented_delta < 0  →  the subsample result is worse
```

Across the 20 repeats, summarise with: `median`, `p2.5`, `p97.5`, `minimum`,
`maximum`. Percentiles use `numpy.percentile(..., method="linear")`.

**Interval naming — mandatory:**

```
subsampling_interval_lower   ( = p2.5 )
subsampling_interval_upper   ( = p97.5 )
```

The strings `confidence interval`, `95% CI`, `ci_2_5`, `ci_97_5`,
`significant`, `significance`, `p-value`, `p_value`, `pvalue`, and the Turkish
`istatistiksel olarak anlamlı` / `anlamlı` are **forbidden** anywhere in the
outputs. A literal token scan enforces this (the same denylist pattern already
used by `few_shot_recovery`).

**Position token.** The location of the full reference relative to the
subsampling range is reported with exactly one descriptive token:

```
below_subsampling_interval    full_reference <  subsampling_interval_lower
inside_subsampling_interval   lower ≤ full_reference ≤ upper
above_subsampling_interval    full_reference >  subsampling_interval_upper
```

Comparison is on the **oriented** scale so the token means the same thing for
Brier as for the AUCs. This token carries no evidential weight and no
significance meaning. It is a description of where one number sits relative to
a range of 20 deterministic re-selections.

## 7. Interpretation contract

**Permitted sentences — these two, verbatim:**

- Full-Muğla reference inside the range:
  > "Observed performance is compatible with sample-size-matched Muğla subsets
  > under this selection design."

- Full-Muğla reference outside the range:
  > "The full-population point estimate differs from the range observed across
  > the deterministic size-matched subsets."

**Forbidden claims — must not appear in `report.md`, `summary.json` or any
derived text:**

- "sample size causes the difference"
- "regional effect is proven"
- "statistically significant"
- "the Muğla difference is eliminated"

**Additional standing limitations, written into `summary.json` as
`limitations`:**

1. The 20 repeats vary only in *which* cells fill each stratum. The stratum
   allocation, the positive count, the per-fold row counts and the fold
   mapping are all identical across repeats. The range therefore describes
   within-stratum selection variability alone — it is narrower than the
   variability of an unconstrained random subsample, and much narrower than
   any sampling distribution of the estimator.
2. The subsample retains all 576 spatial blocks. Spatial extent is *not*
   reduced; only density within each block is. This design deliberately
   isolates count from geographic coverage, so it cannot speak to the effect
   of a smaller AOI.
3. Prevalence is preserved, not equalised. Muğla's positive count (1,438)
   remains far above Manavgat's (784). Any residual difference between regions
   may still be a positive-count difference and this analysis does not
   separate that.
4. Arm C reuses frozen source models. It measures target cohort sensitivity
   under a fixed source, and says nothing about source-side size effects.
5. The final scientific reading must be made from all three arms jointly:
   does within-region performance move; does transfer move when the Muğla
   source shrinks; is the metric ordering preserved across Muğla target
   subsamples. No single arm supports a conclusion on its own.

## 8. Fail-closed conditions

The run aborts, writing nothing, if any of these holds:

- any of the three Step8A sha256 digests mismatches;
- Manavgat primary population ≠ 20,511, or Muğla primary population ≠ 41,730;
- any AOI outside the three primaries is referenced;
- any repeat yields ≠ 20,511 rows, or a duplicate `cell_id`;
- any selected `cell_id` is not in the canonical Muğla primary set;
- `Σ alloc_s ≠ 20511`, or any `alloc_s > c_s`, or any `alloc_s < 1`;
- prevalence drift exceeds the §3.5 bound;
- the inherited fold mapping does not cover every sampled row exactly once;
- any fold has a single class on either side in any repeat;
- any block appears on both sides of a fold;
- an output path resolves outside the analysis namespace;
- a forbidden token appears in any emitted text.
