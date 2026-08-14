# Scientific Contract — `few_shot_recovery.v1`

Every decision below is frozen. Nothing in this document is left open, and
nothing in it may be revised after seeing results.

---

## 1. Question

> When a limited number of labeled spatial blocks from the target region is
> supplied, how much of the performance gap between zero-shot raw transfer and
> the target-only within-region ceiling is recovered?

## 2. Claim boundary

This analysis **is**:

- a supervised adaptation sensitivity analysis, in which target labels are
  deliberately and openly used for adaptation.

This analysis is **not**:

- an operational deployment claim,
- an active-learning analysis (blocks are drawn by a fixed deterministic rule,
  never by model-informed acquisition),
- a causal decomposition (it attributes nothing to covariate shift, concept
  shift or prevalence),
- target-label-free adaptation (that is Step10 and
  `marginal_aoa_completion`; this is its labeled counterpart).

The analysis class string written into every artifact is:

```
target_label_supervised_few_shot_adaptation_sensitivity
```

## 3. Scope

### 3.1 AOIs and directions

Primary AOIs: `manavgat_2021`, `bejis_2022`, `mugla_2021`.

All six directed transfer pairs are analysed:

| # | source | target |
|---|---|---|
| 1 | manavgat_2021 | bejis_2022 |
| 2 | manavgat_2021 | mugla_2021 |
| 3 | bejis_2022 | manavgat_2021 |
| 4 | bejis_2022 | mugla_2021 |
| 5 | mugla_2021 | manavgat_2021 |
| 6 | mugla_2021 | bejis_2022 |

Self-pairs are forbidden. Pair tokens are never sorted; direction is
`<source>_to_<target>`.

### 3.2 Evia exclusion

`evia_2021_extended` is **excluded from the primary run**. It is registered as
a high-prevalence, different-regime sensitivity/control AOI, not an
equal-prevalence primary transfer validation AOI. It may not be added later
without a new preregistration and a new `analysis_id`. The validator fails if
any Evia identifier appears anywhere in the produced artifact.

`evia_2021` (the non-extended registration) is likewise out of scope.

### 3.3 Population

```
valid_for_modeling == True  AND  burnable_tree_shrub_grass == True
```

This is `step9a_audit_cross_region_inputs.PRIMARY_POPULATIONS[0]` and
`step8_large_block_robustness.PRIMARY_POPULATION`. No secondary population is
analysed.

Measured population sizes (this repository, verified):

| AOI | valid rows | population rows | positives | negatives | prevalence |
|---|---:|---:|---:|---:|---:|
| manavgat_2021 | 24 087 | 20 511 | 784 | 19 727 | 0.0382 |
| bejis_2022 | 15 759 | 15 190 | 1 100 | 14 090 | 0.0724 |
| mugla_2021 | 73 045 | 41 730 | 2 911 | 38 819 | 0.0698 |

### 3.4 Frozen inputs

The only data inputs are the three canonical Step8A modeling datasets. Their
SHA-256 digests are mandatory preconditions, verified in this repository on
2026-08-02:

| AOI | path | sha256 |
|---|---|---|
| manavgat_2021 | `outputs/experiments/manavgat_2021/step8a/step8a_500m_modeling_dataset.parquet` | `054a1961fc0582a33d36413263668b63074b21ae8b03d12269b6e228787f3439` |
| bejis_2022 | `outputs/experiments/bejis_2022/step8a/step8a_500m_modeling_dataset.parquet` | `3dec785a7d8e31db2d67ed283546bbfbca1559f56df46663488d0afc24d9e393` |
| mugla_2021 | `outputs/experiments/mugla_2021/step8a/step8a_500m_modeling_dataset.parquet` | `c4ab107db2207f9f20775ccc0b3bf39381173fd07d4e82f6821ce7f40be7db8e` |

All three matched the values supplied in the task specification. Any mismatch
is a hard failure at the `plan` stage, before any fit.

No Earth Engine access occurs at any stage. The module may not import
`core.gee_utils` or `ee`.

---

## 4. Models

### 4.1 Families

- **Primary:** `thermal` — features `SHARED_THERMAL_MODEL_FEATURES`
  (= `BASELINE_FEATURES + THERMAL_FEATURES`, 10 features).
- **Secondary:** `baseline` — features `SHARED_BASELINE_FEATURES`
  (4 features).

Both families are run for every direction, budget, fold and repeat. The
primary/secondary distinction governs reporting emphasis only; the computation
is identical.

### 4.2 Constructor

Reused verbatim, with no new model type and no hyperparameter tuning:

```python
build_pipeline(feature_list, "random_forest", STEP8B_RANDOM_SEED)   # seed 42
```

from `src/step8b_train_baseline_vs_thermal_model.py`. This yields

```
Pipeline([
  ("preprocess", ColumnTransformer([
      ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
      ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["landcover_dominant"]),
  ])),
  ("clf", RandomForestClassifier(n_estimators=300, max_depth=None,
                                 min_samples_leaf=3, class_weight="balanced",
                                 random_state=42, n_jobs=-1)),
])
```

`MODEL_NAME = "random_forest"` matches Step8B, Step9B and Step10.

### 4.3 Existing sample weighting — documented, not invented

**The canonical estimator already applies class weighting.**
`build_classifier("random_forest", …)` passes `class_weight="balanced"`, so
scikit-learn weights each class inversely to its frequency in the fitted
training frame. This is pre-existing canonical behaviour, present in Step8B,
Step9B and Step10.

Its consequence for this analysis is stated here so that it is never mistaken
for a design choice made by this analysis:

> Because the combined few-shot training frame is `source ∪ k target blocks`,
> and `class_weight="balanced"` is computed on that combined frame, the
> effective per-row weights shift slightly as `k` grows. This is the
> unmodified behaviour of the canonical estimator applied to a larger frame.

**No new weighting rule is introduced.** No source/target re-weighting, no
oversampling of the target blocks, no duplication, no `sample_weight`
argument. The few-shot fit is a plain `pipeline.fit(X_combined, y_combined)`.
The validator asserts that no `sample_weight` is ever passed and that the
classifier hyperparameters equal the canonical ones exactly.

### 4.4 The three model conditions

| Condition | Training frame | Preprocessing fitted on | Fits are a function of |
|---|---|---|---|
| `raw` (k=0) | full source population | same frame | direction, family |
| `few_shot` (k>0) | full source population ∪ k target adaptation blocks | same frame | direction, family, fold, budget, repeat |
| `ceiling` | target training pool for this fold (all target blocks outside the outer evaluation fold) | same frame | target, family, fold |

`raw` is fold-independent by construction: the source model never sees the
target, so a single fit per (direction, family) is evaluated against every
outer fold's evaluation blocks. Its per-fold rows are genuine evaluations of
one model, not five refits; `n_fits_performed` records this truthfully.

`ceiling` is source-independent by construction: it is fitted per (target,
family, fold) and reported against both directions that share that target.
30 unique ceiling fits back 60 direction-level rows. This is recorded in the
manifest as `ceiling_fit_sharing: "per_target_fold_family"`.

---

## 5. Leakage-free target evaluation

### 5.1 Block scale — the first forced decision

Step8B's canonical `spatial_block_id` uses
`STEP8B_SPATIAL_BLOCK_SIZE_CELLS = 2` (2 × 500 m ≈ 1 km). The canonical Step8
folds are **not directly suitable** for this analysis, for two measured
reasons:

1. **A 2-cell block is not a unit of labeling effort.** Median rows per
   2-cell block in this population is 4. A budget of "32 labeled blocks" would
   mean roughly 128 cells — the "few-shot label budget" framing would describe
   something that no field campaign would recognise as 32 labeling decisions.
2. **Adjacency.** At 1 km, adaptation blocks selected from the training pool
   are frequently immediately adjacent to evaluation blocks. Formal
   train/test block disjointness would hold, but recovery would be inflated by
   short-range spatial autocorrelation. The repository has already established
   this concern: `step8_large_block_robustness` exists precisely to test
   Step8 conclusions at coarser blocks.

The task specification pre-authorises the fallback: use the existing canonical
spatial-block utility and preserve the ~5 km convention. That convention
already exists in the repository and is adopted **verbatim**:

```python
from src.step8_large_block_robustness import assign_large_blocks
df = assign_large_blocks(df, 10)      # -> column "large_block_id"
```

- `block_row = floor(row_500m / 10)`, `block_col = floor(col_500m / 10)`,
  fixed origin `(0, 0)`;
- `large_block_id = "b10_r{block_row}_c{block_col}"`;
- `NOMINAL_SCALES[10] == "approximately_5_km"`;
- blocks are assigned from the canonical Step8A `row_500m` / `col_500m`
  **before** population filtering, exactly as the frozen robustness analysis
  does.

No ad-hoc row-random split is created anywhere in this analysis.

### 5.2 Outer folds

```python
from src.step8b_train_baseline_vs_thermal_model import make_spatial_folds
folds, n_splits_used = make_spatial_folds(
    y, groups=large_block_id, n_splits_requested=5,
    random_state=42, strict=True,
)
```

`STEP8B_N_SPLITS = 5`, `STEP8B_RANDOM_SEED = 42`. `strict=True` is the mode
the frozen large-block analysis uses; it guarantees, and hard-fails otherwise:

- the fold count is never silently reduced,
- no block appears on both sides of any fold,
- both classes are present on both sides of every fold,
- every row lands in exactly one test fold (full OOF coverage).

**Verified in this repository (2026-08-02):** `strict=True` succeeds with
`n_splits_used = 5` for all three targets; block/fold overlap is 0 and OOF
coverage is exactly 1 for every row of every target. Per-fold counts are
tabulated in `BLOCK_BUDGET_FEASIBILITY.md`.

The outer fold structure depends only on the **target**, not on the source, so
the two directions sharing a target share one fold assignment. This is
required, not incidental: it makes the two directions into that target
paired at the evaluation-block level.

### 5.3 The firewall

For every outer fold of every direction:

- **Evaluation blocks** = the fold's test blocks. They appear in no training
  frame of any condition — not `raw` (which has no target rows at all), not
  `few_shot`, not `ceiling`.
- **Adaptation blocks** are drawn *only* from the target training pool
  (target blocks outside the evaluation fold).
- **All** source rows are available for training in `raw` and `few_shot`;
  `ceiling` uses no source rows.
- **Preprocessing** (median imputation, most-frequent imputation, one-hot
  encoding) is fitted on the training frame of that condition only, inside the
  `Pipeline`, never on a frame containing evaluation rows.
- **Evaluation** is on held-out target blocks only.
- Target labels enter only two places: the `y` vector of a `few_shot` or
  `ceiling` training frame, and the `y_true` of an evaluation. They never
  enter a threshold choice, a feature choice, a hyperparameter, a block
  ordering, or a stopping rule.

There is **no threshold selection anywhere in this analysis.** All three
metrics are threshold-free. Step9B's `select_threshold_from_source_oof` is
deliberately *not* reused: it would add fits and serve no metric here.

---

## 6. Label budgets

### 6.1 Frozen budget set

```
BUDGETS = (0, 1, 2, 4, 8, 16, 32)
```

**All seven budgets are feasible for every direction and every outer fold.**
No budget is dropped. The binding quantity is the size of the target training
pool per fold, whose measured minimum across all 15 (target × fold)
combinations is **139 blocks** (bejis_2022, fold 2) — comfortably above 32.

The full derivation, including the class-composition transition at `k=16`, is
in `BLOCK_BUDGET_FEASIBILITY.md`. Nothing is silently reduced; had a budget
been infeasible it would have been removed from the common set and named here.

### 6.2 Endpoint definitions

- `k = 0` — **raw**: source-only transfer. No target row in the training
  frame. Identical in construction to Step9B's raw source-only transfer,
  evaluated on this analysis's 10-cell evaluation folds.
- `k > 0` — **few-shot**: source population ∪ `k` labeled target adaptation
  blocks, single combined `fit`.
- **ceiling** — target-only: fitted on *all* target training blocks outside
  the outer fold, no source rows. Across the five folds this is exactly the
  canonical within-region OOF procedure at 10-cell blocks.

### 6.3 Ceiling reproduction anchor

The ceiling is not a newly invented quantity. Frozen artifacts already exist
for all three targets at exactly this configuration (10-cell blocks,
`strict` 5-fold, seed 42, population `burnable_tree_shrub_grass`,
`random_forest`):

| target | artifact | frozen baseline ROC-AUC | frozen thermal ROC-AUC |
|---|---|---:|---:|
| manavgat_2021 | `outputs/robustness/step8_large_block/manavgat_2021__bejis_2022/manavgat_2021/block_10_cells/step8b_large_block_metrics.json` | 0.747550 | 0.797430 |
| bejis_2022 | `outputs/robustness/step8_large_block/manavgat_2021__bejis_2022/bejis_2022/block_10_cells/step8b_large_block_metrics.json` | 0.779370 | 0.824469 |
| mugla_2021 | `outputs/experiments/mugla_2021/robustness/step8_big_blocks/block_10_cells/step8b_metrics.json` | 0.697986 | 0.777327 |

The `validate` stage **must** reproduce all three to within `1e-9`. A mismatch
means the fold construction or the model contract has drifted, and is a hard
failure.

The anchors live in two frozen namespaces. The paired large-block robustness
run covered only `manavgat_2021__bejis_2022`; `mugla_2021`'s 10-cell counterpart
comes from its own per-experiment big-block robustness run, whose
`block_manifest.json` binds `input_dataset_sha256` to the same canonical Step8A
digest this analysis gates on. Both are read-only cross-checks: no robustness
artifact is produced, refitted or re-bootstrapped here.

---

## 7. Repeated block selection

### 7.1 Repeats

10 deterministic repeats for every budget with `k > 0`.

`k = 0` and `ceiling` involve no selection randomness and therefore have
exactly **one** deterministic realisation each. They are written with
`repeat_id = 0` and `n_repeats = 1`, and their `selection_interval` bounds
equal their point estimate. Writing 10 identical copies would fabricate an
appearance of sampling variability that does not exist.

### 7.2 Selection is at block level

`k` counts **spatial blocks**, never rows. The number of target rows entering
a few-shot fit is therefore variable (median ≈ 90–100 rows per 10-cell block)
and is recorded per selection as `adaptation_row_count`.

### 7.3 Nested ordering and the class-feasibility rule

For each `(direction, outer_fold, repeat)` **one** total ordering of the
target training pool is constructed, and budget `k` takes its first `k`
entries. Nesting is therefore true by construction: the `k=1` set is inside
`k=2`, inside `k=4`, and so on.

The ordering is built by deterministic tiering, defined **in advance from
repository data** and never revised in response to results:

```
TIER_A  blocks containing BOTH classes   (burned and unburned rows)
TIER_B  blocks containing positives only
TIER_C  blocks containing negatives only

ordering = shuffle(TIER_A, rng) ++ shuffle(TIER_B, rng) ++ shuffle(TIER_C, rng)
```

Each tier is permuted independently with the same derived RNG, then the tiers
are concatenated in the fixed order A, B, C. Blocks are sorted by
`large_block_id` before shuffling so the permutation does not depend on row
order in the parquet.

Properties, all verified against the measured inventory:

- **Every `k ≥ 1` contains at least one positive.** The measured minimum
  `|TIER_A|` over all 15 (target × fold) combinations is 9 > 0, so the first
  selected block is always a both-class block.
- **Up to `k = 8`, every fold fills entirely from positive-containing blocks.**
  The measured minimum `|TIER_A| + |TIER_B|` is 13 ≥ 8.
- **At `k = 16` and `k = 32`, some folds must draw from TIER_C.** This is a
  composition transition, not an infeasibility, and is expected: with only
  13–70 positive-containing blocks per pool, a 32-block budget necessarily
  includes unburned-only terrain. Each row records
  `n_blocks_tier_a`, `n_blocks_tier_b`, `n_blocks_tier_c` so the transition is
  visible in the curve rather than hidden.

The rule "prefer blocks containing both classes as far as available" is
implemented exactly as this fixed tiering. There is no adaptive fallback and
no result-dependent branch.

### 7.4 Seed derivation

Deterministic from `direction + outer_fold + repeat`, and from nothing else:

```python
def selection_seed(source_id, target_id, outer_fold, repeat_id) -> int:
    key = f"{SCHEMA_VERSION}|{source_id}|{target_id}|{outer_fold}|{repeat_id}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32)
```

The seed does **not** depend on `k`: one ordering serves all budgets, which is
what makes nesting exact. It does not depend on the model family either, so
`baseline` and `thermal` see identical adaptation blocks at every
`(fold, repeat, k)` — the two families are paired by construction, and any
difference between them is attributable to features rather than to selection.

The estimator seed stays `STEP8B_RANDOM_SEED = 42` for every fit; only block
selection varies across repeats.

---

## 8. Metrics and recovery

### 8.1 Metrics

Computed by the canonical
`step8b_train_baseline_vs_thermal_model.compute_binary_metrics`:

- **Primary:** `roc_auc` (`sklearn.metrics.roc_auc_score`)
- **Secondary:** `pr_auc` (`average_precision_score`), `brier_score`
  (`brier_score_loss`)

All three are threshold-free. The threshold-dependent fields that
`compute_binary_metrics` also returns (`balanced_accuracy`, `precision`,
`recall`, `f1`) are not part of this contract and are not reported.

### 8.2 Orientation

Recovery arithmetic requires higher-is-better. Brier is lower-is-better, so
the recovery computation uses

```
oriented_brier = -brier_score
```

`roc_auc` and `pr_auc` are used unchanged. Every artifact reports **both**:
the true `brier_score` in its natural sign (for reading) and the oriented
value used in the recovery arithmetic (for auditing). The
`metric_orientation` column carries `higher_is_better` or
`lower_is_better_oriented_by_negation` on every row.

### 8.3 Quantities, per direction × family × metric × budget

```
raw            = point estimate at k = 0
fewshot        = point estimate at this budget
ceiling        = target-only point estimate

absolute_recovery      = fewshot - raw
ceiling_gap            = ceiling - raw
recovery_fraction      = (fewshot - raw) / (ceiling - raw)
```

All computed on **oriented** values.

### 8.4 Recovery-fraction rules — frozen

- **Not clipped.** Values > 1 are kept (few-shot exceeded the ceiling).
- **Not made absolute.** Negative values are kept (few-shot was worse than
  raw).
- **Undefined near a degenerate denominator.** If
  `|ceiling_gap| < 1e-6`, `recovery_fraction` is written as null with
  `recovery_fraction_status = "undefined_degenerate_denominator"`. The
  threshold matches `transfer_decomposition.RATIO_DEGENERATE_THRESHOLD`.
- **Explicit flag when the ceiling does not exceed raw.** If
  `ceiling <= raw`, `ceiling_not_above_raw = true` and
  `recovery_fraction_status = "ceiling_not_above_raw"`. The fraction is still
  computed when the denominator is non-degenerate, and its sign is preserved;
  the flag warns that the fraction is not interpretable as "fraction of a
  recoverable gap".

Status values are exactly: `interpretable`,
`undefined_degenerate_denominator`, `ceiling_not_above_raw`.

For context: at 2-cell blocks the frozen Step10 numbers show raw transfer ROC-AUC
below 0.5 in several directions (e.g. `manavgat_2021_to_bejis_2022` raw thermal
0.3258 against within 0.9178), so `ceiling > raw` is expected to hold widely —
but the flag is contractual, not optional, and is evaluated per row.

---

## 9. Uncertainty wording

### 9.1 Full OOF per repeat

Every repeat produces a **complete target OOF prediction vector**: each target
population row is predicted exactly once, by the model of the outer fold in
whose evaluation set that row lies. Coverage is asserted per
(direction, family, budget, repeat): every row covered exactly once, no NaN.

Metrics are computed at two levels and both are stored:

- `oof` — over the whole assembled OOF vector (primary, reported in the curve),
- `fold` — within each outer fold separately (diagnostic, for fold stability).

### 9.2 Selection interval — the only interval this analysis produces

Across the 10 repeats of a budget:

```
selection_median = median of the 10 repeat OOF metric values
selection_p2_5   = 2.5th percentile of the 10 repeat values
selection_p97_5  = 97.5th percentile of the 10 repeat values
```

with `numpy.percentile(..., method="linear")`.

These are named **`selection_interval`** everywhere — in column names, in
JSON keys, in the report prose and in the figures. The interval describes
**variability across which blocks were selected**, and nothing else. It does
not describe sampling variability of the target population, and it must never
be presented as if it did.

**Forbidden vocabulary,** enforced by the validator across every produced
`.md`, `.json` and `.csv`:

```
"confidence interval", "95% CI", "CI", "significant", "significance",
"statistically significant", "p-value", "p =", "istatistiksel olarak anlamlı",
"anlamlı"
```

`ci_2_5` / `ci_97_5` column names from other analyses are also forbidden here;
this schema uses `selection_p2_5` / `selection_p97_5`.

With 10 repeats, the 2.5th and 97.5th percentiles are interpolated inside the
observed min–max range and are close to the extremes. The report states this
plainly: the interval is a compact summary of 10 observations, not an estimate
of a population quantile.

### 9.3 Bootstrap reuse — investigated, resolved

The task requires investigating whether existing target spatial-block
bootstrap artifacts can be reused for `raw` and `ceiling`, and forbids
designing a new bootstrap if they cannot.

**Raw transfer — NOT reusable.** The candidates are
`outputs/cross_region/<pair>/step9c/…` and
`outputs/cross_region/<pair>/step10/step10_bootstrap_replicates.parquet`.
Both resample `spatial_block_id`, which is the **2-cell** block
(`step10c_paired_evaluation_bootstrap.py` calls
`run_n_way_paired_bootstrap(..., block_col="spatial_block_id", ...)`, and
`step9b_run_cross_region_transfer.load_step8a_dataset` builds that column with
`STEP8B_SPATIAL_BLOCK_SIZE_CELLS`). Resampling 1 km blocks yields a different
resampling unit from this analysis's 10-cell evaluation frame, and the
underlying raw point estimate is computed over the whole target rather than
over 10-cell OOF folds. The two are not genuinely comparable. **No new
raw-transfer bootstrap is designed.** This is a stated limitation.

**Ceiling — partially reusable, and used as a reproduction anchor.**
`outputs/robustness/step8_large_block/manavgat_2021__bejis_2022/<target>/block_10_cells/step8c_large_block_bootstrap_summary.json`
is a paired spatial-block bootstrap over **10-cell** blocks on this exact
population and configuration, and it reports absolute per-family series
(`auc_thermal.ci_2_5 = 0.7376`, `ci_97_5 = 0.8508` for manavgat_2021, for
example). It is genuinely comparable for the ceiling.

Resolution, frozen:

- the ceiling **point estimates** for all three targets must reproduce the
  frozen 10-cell values to `1e-9` (§6.3) — a hard validator check;
- the frozen ceiling bootstrap intervals are **copied into**
  `summary.json` under `external_ceiling_reference`, verbatim, labelled with
  their true source and their true name (`spatial_block_bootstrap_2_5` /
  `_97_5`) and explicitly marked as *not* a selection interval and *not*
  comparable to the raw endpoint;
- `mugla_2021`'s reference is the equivalent bootstrap of its own big-block
  robustness run
  (`outputs/experiments/mugla_2021/robustness/step8_big_blocks/block_10_cells/bootstrap_summary.json`,
  1000 replicates, seed 42, 10-cell blocks), copied in under the same labelling
  rules;
- **no new bootstrap is run anywhere in this analysis.**

### 9.4 No p-values

No hypothesis test, no p-value, no significance claim, no "supported /
unsupported" status derived from an interval excluding a null value. The
recovery curve reports point estimates and selection intervals, and the report
describes them descriptively.

Note the deliberate contrast with `transfer_decomposition.py`, which does emit
statuses such as `supported_recovery_above_chance` from bootstrap intervals.
That vocabulary is **not** imported here, because a 10-repeat selection
interval cannot support it.

---

## 10. Summary of forced changes

| # | Specification default | Frozen decision | Reason |
|---|---|---|---|
| 1 | Reuse canonical Step8 spatial folds | 10-cell (~5 km) blocks via `assign_large_blocks(df, 10)` + `make_spatial_folds(..., strict=True)` | Canonical 2-cell blocks hold a median of 4 cells; a "labeled block" would not be a labeling unit, and adaptation blocks would abut evaluation blocks. The ~5 km fallback was pre-authorised and already exists in the repository. |
| 2 | Reuse existing raw/ceiling bootstrap artifacts | Ceiling reused (point anchor + external reference, 2 of 3 targets); raw **not** reused | Raw bootstraps resample 2-cell blocks — not comparable. No new bootstrap designed; recorded as a limitation. |
| 3 | 10 repeats per budget | 10 repeats for `k > 0`; 1 realisation for `k = 0` and ceiling | Neither endpoint has selection randomness; 10 identical copies would fabricate apparent variability. |

No budget was dropped. No scientific decision is left unresolved.
