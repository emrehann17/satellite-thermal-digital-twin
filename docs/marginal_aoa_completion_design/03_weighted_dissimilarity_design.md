# 03. Importance-Weighted Predictor-Space Dissimilarity — Exact Design

Meyer & Pebesma style nearest-neighbour dissimilarity, adapted to this
repository's frozen contracts. Nothing in this document was executed.

---

## 1. Scientific statement

For a directed pair `source → target`, and for each target cell in the primary
population, measure how far that cell is — in an **importance-weighted,
source-standardised** predictor space — from the nearest cell the source model
actually saw. Normalise that distance by a source-internal reference distance so
the quantity is dimensionless and comparable across pairs. Compare it against a
source-derived threshold to classify the cell as inside or outside the weighted
Area of Applicability.

This is a **joint-space** diagnostic and is therefore complementary to, not a
replacement for, `marginal_aoa.v1`, which is explicitly one-dimension-at-a-time.
The existing artifact's own limitations block says it does "NOT measure
multivariate joint support (no convex hull, no density ratio, no Mahalanobis/kNN
distance)". This design supplies exactly the kNN-distance piece that was named as
missing.

---

## 2. Assessment of the candidate contract in the task specification

The candidate contract given in the task is scientifically sound and is adopted
with **four specific modifications**, each justified below:

| # | Candidate | Adopted? | Modification and reason |
|---|---|---|---|
| 1 | `z = (x - source_mean) / source_scale`, scale = source SD | **Yes** | Adopted, but bound to the repository's existing `compute_regionwise_zscore_stats` (ddof=0, `EPSILON_STD` constant guard) rather than a fresh implementation. |
| 2 | Weights nonnegative, sum to 1 | **Yes** | Adopted unchanged. Negative-importance branch is unreachable with Gini (see doc 02) but must still fail closed. |
| 3 | `d_w = sqrt(Σ_j w_j (z_j(x) - z_j(x_i))²)` | **Yes, extended** | The formula covers numeric features only. A categorical term must be added explicitly — see §5. Silently coding `landcover_dominant` as a numeric class code is forbidden. |
| 4 | Multiply standardised features by `sqrt(w_j)` before NN search | **Yes** | Adopted as the implementation, with a caveat: this equivalence holds only for the numeric block. The categorical term is not a Euclidean coordinate and must be handled separately — see §5.4. |
| 5 | Normalise by source NN distance; prefer spatial-fold holdout | **No — replaced** | The normaliser is the **mean pairwise weighted distance over all distinct source reference cell pairs**, not any nearest-neighbour mean. A nearest-neighbour denominator measures local grid spacing; a pairwise mean measures the actual spread of the source cloud. See §6. Spatial folds are retained but serve only the training DI and the threshold — see §7. |
| 6 | Threshold = 0.95 quantile of source holdout DI | **No — replaced** | The operative threshold is the **upper whisker**, `min(max(training_DI), Q3 + 1.5·IQR)`. The 0.95 quantile is retained as a reported secondary sensitivity value and must never silently replace the primary. See §7. |

The one place where the candidate contract is genuinely incompatible with the
repository is **missing values**. See §3.3.

---

## 3. Numeric standardisation

### 3.1 Statistics

Computed **from the source AOI primary population only**, reusing
`core/step10_shared.compute_regionwise_zscore_stats` verbatim:

```python
mean = float(observed.mean())
std  = float(observed.std(ddof=0))        # population SD
constant_guard = std < EPSILON_STD        # then std := 1.0
```

For each of the 9 numeric features `j`:

```
z_j(x) = (x_j - source_mean_j) / source_scale_j
```

The **same source statistics** are applied to both source reference cells and
target cells. Target statistics are never computed and never used — this is what
makes the transformation directional and keeps the target distribution out of the
coordinate system.

### 3.2 Why source SD, and not IQR or MAD

Three reasons, in order of weight:

1. **It is already the repository's standardisation contract.** Step10's
   adaptation layer (`regionwise_zscore`, `coral_after_regionwise_zscore`) uses
   exactly this transform, and the transfer results the comparison layer will be
   ranked against were produced in that coordinate system. Using a different
   scale here would compare two diagnostics living in two different spaces.
2. It matches the published Meyer & Pebesma construction, which standardises by
   the (training) standard deviation.
3. It is already tested, and has an explicit constant-feature guard.

**The counter-argument is real and must be recorded.** `elevation_mean`,
`current_lst_mean`, `downscaled_lst_mean` and `fused_lst_mean` are heavy-tailed
in these AOIs, and SD is sensitive to a handful of extreme source cells — the
same weakness the existing artifact already flags for min/max support ("sensitive
to single extreme source cells and to source sample size"). SD is less brittle
than min/max but is not robust.

**Recommended handling:** SD as the preregistered primary; a source-IQR variant
(`(x − source_median) / source_IQR`, with `IQR = q75 − q25`) available as a
**preregistered sensitivity column** that is computed only if decision **B-4** is
answered "report both". It must not be run and then chosen after the fact.

### 3.3 Missing values — divergence from `apply_regionwise_zscore`

`core/step10_shared.apply_regionwise_zscore` fills missing values with the
region's own mean before standardising, so post-transform missing values equal
exactly 0.0. **That transform must not be used here.**

Reason: for a target cell, filling with the target mean would make the cell's
coordinates depend on the target distribution, which breaks directionality; and
filling with the source mean would place every incomplete target cell at the
source centroid, which is the single most-supported point in the space — turning
missingness into spurious evidence of support. Either choice is wrong.

**Adopted policy, mirroring `marginal_aoa.v1`'s missing-value rules:**

| Case | Policy |
|---|---|
| Source reference cell with any missing numeric predictor | **Excluded from the reference set.** Count reported as `source_rows_excluded_missing`. |
| Source reference cell with missing `landcover_dominant` | Excluded. (Observed count: 0 in all four AOIs.) |
| Target cell with any missing numeric predictor | **`not_assessable`.** No DI is computed. Never counted as outside. |
| Target cell with missing `landcover_dominant` | `not_assessable`. (Observed count: 0.) |
| Source feature with no finite value at all | Fail closed — cannot occur under the observed data, but must abort rather than silently drop a weighted dimension. |

Expected reference-set attrition, derivable from doc 01 §5 (union of the
per-feature missing sets is at most their sum, and `lst_anomaly_mean` dominates):

| AOI as source | Population rows | Upper bound on excluded rows | Worst-case reference set |
|---|---:|---:|---:|
| `bejis_2022` | 15190 | ≤ 2576 | ≥ 12614 |
| `evia_2021_extended` | 9298 | ≤ 500 | ≥ 8798 |
| `manavgat_2021` | 20555 | ≤ 644 | ≥ 19911 |
| `mugla_2021` | 41731 | ≤ 2106 | ≥ 39625 |

These are upper bounds on attrition because the missing sets overlap heavily
(`current_lst_mean`, `current_tvdi_mean` and `tvdi_difference_mean` are missing on
almost the same cells). The true reference sets will be larger. Every reference
set stays comfortably above any sample-size concern. **The actual counts must be
computed and reported at implementation time, not assumed.**

### 3.4 Zero and near-zero variance

`EPSILON_STD` guard from `core/step10_shared.py` applies: if
`source_std < EPSILON_STD`, the scale is set to `1.0` and
`constant_feature_guard_used = true` is recorded **per feature per source AOI**.

Evidence that this will not trigger: the existing
`marginal_aoa_numeric_features.csv` reports a strictly positive
`source_range_width` for all 9 numeric features in all 12 pairs (e.g. Bejís
`elevation_mean` range `[120.708, 1990.298]`, width 1869.59). The guard is a
fail-safe, not an active branch. **The validator must record the flag truthfully
rather than assert it is always false.**

---

## 4. Feature weights

Derived exactly as specified in `02_feature_importance_audit.md` §6.

```
w = (w_ndvi, w_elevation, w_slope, w_lst_anomaly, w_current_lst,
     w_current_tvdi, w_tvdi_difference, w_downscaled_lst, w_fused_lst,
     w_landcover)

w_j >= 0  for all j
Σ_j w_j = 1   (asserted to 1e-9)
```

### 4.1 Negative-importance policy

Options considered:

| Option | Verdict |
|---|---|
| A. `max(importance, 0)` | **Recommended** if the weighting source ever changes to permutation importance. |
| B. `abs(importance)` | **Rejected.** A negative permutation importance means shuffling the feature *improved* held-out performance. Treating that as strong importance would weight a predictor the model does not usefully use. Wrong sign of information. |
| C. Drop the feature entirely | **Rejected as the default.** Removing a dimension from the distance changes the geometry of the space, not just the weight; a feature with a small negative estimate is better represented as weight zero than as an absent axis. |
| D. Other | Not needed. |

**Adopted for the current configuration:**

```
negative_importance_policy      = "fail_closed_assert_nonnegative"
negative_importance_reachable   = false
justification                   = "RandomForest mean-decrease-in-impurity is
                                   non-negative by construction; verified
                                   min(importance) = 0.0 in all four source AOIs."
fallback_if_source_changes      = "A: clip at zero, renormalise, record the
                                   clipped feature list."
```

The implementation asserts non-negativity and aborts on violation. It does **not**
silently clip, because under the adopted source a negative value would mean the
artifact is not what this design believes it is.

### 4.2 Zero-weight policy

A feature with `w_j = 0` contributes exactly zero to every distance. This is
mathematically correct and must be preserved, but it must also be **visible**:

```
zero_weight_features        : []          (list, per source AOI)
effective_feature_count     : Σ 1[w_j > 0]
```

Evidence: after group summing, **no** feature in the 10-feature contract has zero
weight in any of the four source AOIs. Individual landcover *dummy levels* do hit
exactly zero (`_50` in Manavgat, `_80` in Bejís, `_90` in Muğla and Evia), which
is one more reason the categorical predictor is weighted as a group rather than
per level — see §5.

### 4.3 Reported weight diagnostics

```
feature_weight_entropy   = -Σ_j w_j * log(w_j)          (natural log; 0·log0 := 0)
effective_feature_count  = exp(feature_weight_entropy)   -- perplexity form
```

Both are reported per source AOI. The perplexity form is the more interpretable
"how many features is this weighting effectively using" number; the raw entropy is
kept because it is the quantity the formula is defined on. Note that
`effective_feature_count` is reported twice under two definitions — the count of
strictly-positive weights and the entropy perplexity. **They must be given
distinct field names** to avoid a silent conflict:

```
n_features_with_positive_weight   : integer
effective_feature_count_perplexity: float
```

---

## 5. Categorical `landcover_dominant`

### 5.1 The options

| Option | Assessment |
|---|---|
| A. One-hot encoding with group-normalised importance | **Rejected.** With per-dummy weight `w_lc/K`, two different levels differ in exactly two one-hot coordinates by 1 each, giving a squared contribution of `2·w_lc/K`. The mismatch penalty then depends on **K**, the number of levels that happened to be observed. Manavgat and Bejís have K=7, Muğla and Evia have K=8 — so the same conceptual mismatch would be penalised differently depending on which AOI is the source. That is an encoding artefact, not a scientific signal. |
| B. Weighted categorical mismatch penalty | **Recommended.** See §5.2. |
| C. Categorical-support sidecar, kept out of the weighted DI | Adopted **in addition to** B, not instead of it. See §5.5. |
| D. Other | Not needed. |

### 5.2 Adopted: weighted mismatch penalty (Gower categorical term)

For a target cell `x` with level `ℓ(x)` and a source reference cell `x_i` with
level `ℓ(x_i)`:

```
d_cat²(x, x_i) = w_landcover · 1[ ℓ(x) ≠ ℓ(x_i) ]
```

Full weighted distance:

```
d_w(x, x_i) = sqrt(  Σ_{j ∈ numeric} w_j · ( z_j(x) − z_j(x_i) )²
                   + w_landcover · 1[ ℓ(x) ≠ ℓ(x_i) ]  )
```

Properties, each of which is a reason for the choice:

- **K-invariant.** The penalty is `w_landcover` for any mismatch, regardless of
  how many levels exist. Comparable between a 7-level and an 8-level source.
- **Preserves the source model's landcover weight exactly.** Total categorical
  influence is `w_landcover = Σ cat__landcover_dominant_*`, which is precisely
  what the source RandomForest assigned to the predictor.
- **Symmetric in the level pair**, as a distance term must be.
- **Handles unseen target levels gracefully** — see §5.3.
- It is the standard Gower heterogeneous-distance categorical term, so it is a
  named, citable construction rather than an ad-hoc rule.

**Honest caveat, required in the artifact limitations block:** the numeric block
is unbounded above while the categorical term is bounded by `w_landcover`. A
target cell that is extreme on `elevation_mean` can therefore reach a far larger
DI than one that merely sits on an unseen landcover class. This is the standard
Gower mixed-type caveat. Given the observed data — the *entire* unseen-level
population across all 12 pairs is 7 cells — this asymmetry has essentially no
influence on the current numbers, but it is a property of the metric and is
stated rather than discovered later.

### 5.3 Unseen target level

An unseen level mismatches **every** source reference cell by construction, so it
adds `w_landcover` to every candidate distance and therefore to the nearest one.
It can never be the reason a cell is classified inside support, and it is never
silently ignored. No special-case branch is required.

Recorded fields:

```
target_cells_with_unseen_level          : integer
fraction_target_cells_with_unseen_level : float
unseen_levels                           : sorted list of level codes
```

Expected values from doc 01 §6: non-zero only for the four directions
`{manavgat_2021, bejis_2022} → {mugla_2021, evia_2021_extended}`, and at most 7
cells.

### 5.4 Implementation note on the `sqrt(w)` equivalence

The candidate contract's "multiply standardised features by `sqrt(w_j)` before
nearest-neighbour search" is valid **for the 9 numeric coordinates only**. The
categorical term is not a coordinate and cannot be folded into a Euclidean
tree query.

Exact, deterministic implementation, with no approximation:

```
For each target cell x:
  d_same  = min over source reference cells with ℓ(x_i) == ℓ(x)  of  d_num(x, x_i)
  d_diff  = min over source reference cells with ℓ(x_i) != ℓ(x)  of  d_num(x, x_i)
  d_w(x)  = min( d_same,  sqrt( d_diff² + w_landcover ) )
```

where `d_num` is the Euclidean distance in the `sqrt(w)`-scaled numeric space.
This is **exactly** the minimum of the full mixed distance — the categorical term
takes only two values, so partitioning the reference set by level and taking two
tree queries is an identity, not an approximation. It costs one `KDTree`/`BallTree`
per source level plus one over the full reference set.

If the simpler route is preferred, chunked brute force over the full reference set
computes the same value directly. Both are exact; §8 recommends which to use.

### 5.5 Relationship to the existing categorical diagnostic

The `marginal_aoa.v1` categorical result
(`marginal_aoa_categorical_features.csv`, `target_unseen_levels`,
`fraction_target_unseen_level`) is **kept unchanged and unmoved**. The new
artifact links it as a sidecar by directed pair:

```
weighted_predictor_space/directed_pair_summary.csv
  → column  unweighted_categorical_sidecar_path
  → column  unweighted_fraction_target_unseen_level   (copied value, for joining)
```

The copied value is a read-only echo for convenience. The authoritative number
stays in `marginal_aoa.v1`, and a validator check asserts the echo matches it.

---

## 6. Distance normaliser — mean pairwise source distance

**This replaces an earlier draft that used a holdout nearest-neighbour mean.**
That construction was wrong for this purpose and is no longer part of the design
in any role.

### 6.1 Why a nearest-neighbour denominator is the wrong quantity

A nearest-neighbour distance on a dense, strongly autocorrelated 500 m grid is
dominated by **grid spacing**, not by the spread of the source distribution.
Introducing spatial folds mitigates this but does not remove it: with ≈5 km
blocks, a held-out cell's nearest out-of-fold neighbour still sits just across a
block boundary, so the denominator remains a local-geometry statistic that
shrinks as the AOI is sampled more densely.

That makes it unfit as a **normaliser**, because the normaliser's job is to put
DI on a scale where "1" means something stable about the source cloud. A
denominator that depends on sampling density would make DI values incomparable
between a 9 298-cell AOI (Evia) and a 41 731-cell AOI (Muğla) purely through
sample size.

The mean pairwise distance is a genuine **scale statistic of the source
distribution**: it is the expected weighted distance between two randomly drawn
source cells. It is essentially insensitive to sampling density and it
characterises spread rather than local spacing.

### 6.2 Adopted definition

```
source_pairwise_mean_distance
    = mean of d_w(s_a, s_b) over all DISTINCT unordered pairs
      of source reference cells

source_distance_normaliser = source_pairwise_mean_distance
```

Equivalent, and the form the implementation should follow because it chunks
naturally:

```
for every source reference cell s:
    m(s) = mean of d_w(s, s') over all source reference cells s' != s

source_distance_normaliser = mean( m(s) over all source reference cells s )
```

The two forms are algebraically identical: each unordered pair `{a,b}` is counted
once in the first form and twice — once in `m(a)`, once in `m(b)` — in the
second, and both denominators scale identically. A test must assert they agree to
floating-point tolerance on a small fixture.

Properties, all required:

- **Self-distance is excluded.** `d_w(s, s) = 0` is never included in any mean.
  With `n` reference cells, the divisor is `n(n−1)/2` in the first form and
  `n−1` inside each `m(s)` in the second.
- **The categorical mismatch term is included**, exactly as in every other
  distance in this design.
- **Source-only.** No target cell enters it.
- **No folds.** The fold structure plays no part in the normaliser.
- **No labels**, directly or indirectly through fold assignment.
- **Identical for all three targets of a given source AOI.** This is a validator
  check, and it is what makes DI comparable across the three directions leaving a
  source.
- **Deterministic.** No seed, no sampling, no approximation.

### 6.3 Computation

Exact, by deterministic chunking over source rows — the same chunked brute force
used elsewhere in this design, accumulating a running sum rather than
materialising the full `n × n` matrix.

Cost is `n(n−1)/2` distance evaluations per source AOI:

| Source AOI | Reference cells (upper bound on attrition applied) | Distinct pairs |
|---|---:|---:|
| `evia_2021_extended` | ≥ 8 798 | ≈ 3.9 × 10⁷ |
| `bejis_2022` | ≥ 12 614 | ≈ 8.0 × 10⁷ |
| `manavgat_2021` | ≥ 19 911 | ≈ 2.0 × 10⁸ |
| `mugla_2021` | ≥ 39 625 | ≈ 7.9 × 10⁸ |

At d = 9 with a running accumulator this is a few minutes for the largest AOI and
needs no more memory than one chunk. **Do not precompute it now** — it is
computed once per source AOI during implementation.

### 6.4 Definition of the dissimilarity index

```
DI(x) = d_w_nearest(x) / source_distance_normaliser
```

where `d_w_nearest(x)` is the nearest full mixed weighted distance from target
cell `x` to the **whole** source reference set.

`DI = 0` means the target cell coincides exactly with a source reference cell.
`DI = 1` means the target cell's distance to its nearest source neighbour equals
the average distance between two arbitrary source cells — i.e. the cell is as far
from its closest support as two typical source cells are from each other. Values
well below 1 are therefore expected for well-supported targets, and the scale is
interpretable without reference to the sampling design.

---

## 7. Training DI and the AOA threshold

The spatial folds removed from the normaliser are retained here, and **only**
here, where a held-out construction is genuinely the right one: the threshold
must answer "how large a DI does source data itself produce when it is not in the
reference set", and that requires holding data out.

### 7.1 Source folds

```
1. Assign every source reference cell a spatial block:
     add_spatial_block_id(df, block_size_cells = 10)      -- ≈5 km
   using src/step8b_train_baseline_vs_thermal_model.py:277 verbatim,
   fixed origin (0,0), assigned before population filtering.

2. Assign blocks to K = 5 folds, LABEL-FREE and deterministically:
     blocks sorted lexicographically by block_id
     fold(block) = index_of_block_in_sorted_order  mod  5

   A block is assigned WHOLLY to one fold; a block is never split.
```

**Why ≈5 km blocks.** This is the repository's own preregistered large-block
scale (`src/step8_large_block_robustness.py:38-39`,
`NOMINAL_SCALES = {10: "approximately_5_km", 20: "approximately_10_km"}`). At the
Step8B 1 km scale the held-out neighbour is too close to represent transfer to an
unseen region.

**Why folds are recomputed rather than read from `step8b_predictions.parquet`.**
That file's `fold_id` comes from `StratifiedGroupKFold`, which consumes `y`.
Reusing it would push source-label information into the threshold. The design
already accepts source-label influence through the weights; it should not also
accept it here, where it is entirely avoidable.

**Why sorted-block round-robin.** Fully deterministic, no seed, no label,
verifiable by hand. `GroupKFold` is an acceptable alternative (also
deterministic, also label-free) and is recorded as decision B-5.

### 7.2 Training DI

```
For each source reference cell s in fold k:

    holdout_nearest_distance(s)
        = min over source reference cells OUTSIDE fold k  of  d_w(s, ·)
      (full mixed distance, including the categorical term)

    training_DI(s)
        = holdout_nearest_distance(s) / source_distance_normaliser
```

Note the denominator: the training DI is divided by the **pairwise mean
normaliser of §6**, not by any holdout statistic. Training DI and target DI
therefore live on exactly the same scale, which is what makes comparing a target
cell against a training-DI-derived threshold meaningful.

### 7.3 Primary threshold — upper whisker

```
Q1  = quantile_0.25( training_DI )
Q3  = quantile_0.75( training_DI )
IQR = Q3 - Q1

upper_whisker = Q3 + 1.5 * IQR

source_aoa_threshold = min( max(training_DI), upper_whisker )
```

```
primary_threshold_method = "source_spatial_fold_holdout_di_upper_whisker_v1"
```

**Why the upper whisker.** It is the standard Tukey outlier fence and it is the
construction used in the published AoA method. It has two properties that a bare
quantile does not:

- It **adapts to the shape** of the training-DI distribution rather than fixing
  the outside rate in advance. A quantile threshold declares ~5% of source
  holdout cells outside by construction, regardless of whether the distribution
  has a tail at all.
- The `min(max(training_DI), ...)` clamp means the threshold **never exceeds
  anything actually observed** in the source. When the training-DI distribution
  is compact and the whisker would land beyond the observed maximum, the maximum
  is used, so the threshold stays inside the evidence.

The clamp also bounds the fragility that `marginal_aoa.v1` flags for its own
min/max support definition: the maximum is only reachable when the whisker
exceeds it, and in that case the distribution has no upper tail for a single
extreme cell to define.

### 7.4 Secondary thresholds — reported, never operative

```
training_di_q50_threshold
training_di_q90_threshold
training_di_q95_threshold      method token: source_spatial_fold_holdout_di_q95_v1
training_di_q99_threshold
training_di_max_threshold
```

**The primary inside/outside classification uses the upper whisker and nothing
else.** The q95 value is reported so the sensitivity of the classification to the
threshold rule is visible, and so a reader can reconstruct what a quantile rule
would have given. It must never silently replace the primary, and a validator
check asserts that `fraction_target_cells_inside_weighted_aoa` was computed
against `training_di_upper_whisker_threshold`.

Quantile method: `numpy.quantile(..., method="linear")`, stated in the artifact.

### 7.5 Firewall

**The threshold is derived from source data only.** No target value, no target
quantile, no target label and no transfer metric enters it. A validator check
must assert that the threshold computation reads no target frame at all, and that
the threshold is identical across the three directed rows sharing a source.

---

## 8. Computational contract

### 8.1 Problem size

Reference sets are the source primary populations minus missing-predictor rows
(doc 01 §5, §3.3 above): between ≈8 800 and ≈41 700 rows, in 9 numeric dimensions
plus one categorical.

Worst-case work:

| Task | Largest instance | Distance evaluations |
|---|---|---:|
| Normaliser: all distinct source pairs (§6.3) | `mugla_2021` source, ≈39.6k cells | ≈ 7.9 × 10⁸ |
| Training DI: source holdout NN (§7.2) | `mugla_2021` source, ≈39.6k × ≈31.7k out-of-fold | ≈ 1.3 × 10⁹ |
| Target → source NN (§6.4) | source `mugla_2021` (≈39.6k) × target `manavgat_2021` (20 555) | ≈ 8.1 × 10⁸ |

At d = 9, all three are routine. The nearest-neighbour tasks use exact
`KDTree`/`BallTree` queries; the normaliser uses chunked accumulation and never
materialises an `n × n` matrix. `scipy==1.18.0` and `scikit-learn==1.9.0` are
both already in `requirements-lock.txt`. **No new dependency is required for this
component** — the only new dependency in the whole design is `geographiclib`, for
the geographic component (doc 06 §4).

### 8.2 Adopted contract

| Question | Decision |
|---|---|
| A. All source cells as reference set | **Yes.** |
| B. Deterministic spatially balanced subsample | **Not used.** No subsampling is needed at this data size. |
| C. Exact nearest neighbours | **Yes.** |
| D. Approximate nearest neighbours | **Not used.** |

```
nearest_neighbour_method     = "exact"
reference_set                = "all source primary-population cells with complete predictors"
subsampling                  = "none"
random_seed                  = null        (no stochastic step exists)
approximation_error_audit    = "not_applicable_exact_computation"
```

**Because no subsampling is used, there is no seed, no minimum per-block
representation, no repeat count and no approximation-error audit to specify.**
The task's subsample sub-contract is therefore recorded as not applicable rather
than filled in with placeholder values. If a future AOI makes the exact
computation infeasible, the subsample contract must be preregistered *then*,
before it is used — not pre-authorised here.

### 8.3 Implementation route

Primary: the two-query partition of §5.4, using `sklearn.neighbors.KDTree` on the
`sqrt(w)`-scaled numeric coordinates — one tree over the full reference set, one
tree per source landcover level. Exact by construction.

Fallback, if the per-level trees prove awkward: chunked brute force
(`scipy.spatial.distance.cdist` over row blocks of ~4096 target cells). Also
exact, simpler to verify, slower by a constant factor that is irrelevant at this
size. **A test must assert the two routes agree to floating-point tolerance on a
synthetic fixture.**

Both routes are deterministic: no sampling, no seed, no tie-breaking that affects
the returned *distance* (ties affect which neighbour is returned, never the
minimum distance itself, and only the distance is used).

---

## 9. Output field definitions

All fields below are per **directed pair**, and are exact definitions, not
descriptions.

### 9.1 Target-cell dissimilarity summaries

Computed over **assessable** target cells only — those with complete predictors.
The denominator is stated explicitly in the artifact.

| Field | Definition |
|---|---|
| `target_mean_dissimilarity` | arithmetic mean of `DI(x)` over assessable target cells |
| `target_median_dissimilarity` | 0.50 quantile of `DI(x)` over assessable target cells |
| `target_p90_dissimilarity` | 0.90 quantile |
| `target_p95_dissimilarity` | 0.95 quantile |
| `target_max_dissimilarity` | maximum |

Quantile method: `numpy.quantile(..., method="linear")`, stated in the artifact so
it is reproducible.

### 9.2 Classification fractions

| Field | Definition |
|---|---|
| `fraction_target_cells_inside_weighted_aoa` | `#{x assessable : DI(x) <= training_di_upper_whisker_threshold} / target_n_total` |
| `fraction_target_cells_outside_weighted_aoa` | `#{x assessable : DI(x) >  training_di_upper_whisker_threshold} / target_n_total` |
| `fraction_target_cells_not_assessable` | `#{x with any missing predictor} / target_n_total` |

The classification threshold is the **upper whisker of §7.3** and nothing else.
No secondary threshold enters these three fields.

**The denominator is `target_n_total`, the full primary-population row count**,
not the assessable count. The three fractions therefore sum to exactly 1.0, and a
validator check asserts that to 1e-12. This matches `marginal_aoa.v1`, whose
three fractions also sum to 1 over the full target population — keeping the two
artifacts directly comparable was the reason for the choice.

Note the inequality is `<=` at the threshold: a cell exactly at the threshold is
**inside**. Stated because it is otherwise ambiguous.

### 9.3 Source-side diagnostics

| Field | Definition |
|---|---|
| `source_pairwise_mean_distance` | mean weighted distance over all distinct source reference cell pairs, self-distance excluded (§6.2) |
| `source_distance_normaliser` | **equals** `source_pairwise_mean_distance`; carried under both names so the schema states the role and the quantity separately |
| `training_di_upper_whisker_threshold` | `min(max(training_DI), Q3 + 1.5·IQR)` (§7.3) — **the operative threshold** |
| `primary_threshold_method` | `"source_spatial_fold_holdout_di_upper_whisker_v1"` |
| `training_di_q95_threshold` | 0.95 quantile of `training_DI` — **secondary, reported, never operative** |
| `training_di_q95_method` | `"source_spatial_fold_holdout_di_q95_v1"` |
| `training_di_q50_threshold`, `training_di_q90_threshold`, `training_di_q99_threshold`, `training_di_max_threshold` | secondary, reported |
| `training_di_q1`, `training_di_q3`, `training_di_iqr` | the inputs to the whisker, stored so the threshold is independently recomputable |
| `source_reference_rows` | reference-set size after missing-predictor exclusion |
| `source_rows_excluded_missing` | primary-population rows dropped for missing predictors |
| `source_holdout_fold_count` | 5 |
| `source_holdout_block_size_cells` | 10 |
| `fold_assignment_method` | `"sorted_block_round_robin_5_folds"` |
| `fold_assignment_reads_label` | `false` |
| `normaliser_uses_folds` | `false` — the normaliser is fold-independent by construction |

### 9.4 Weight diagnostics

| Field | Definition |
|---|---|
| `importance_method` | `"impurity_gini_in_sample_whole_population_v1"` |
| `n_features_with_positive_weight` | `Σ_j 1[w_j > 0]` over the 10 contract features |
| `effective_feature_count_perplexity` | `exp(-Σ_j w_j log w_j)` |
| `feature_weight_entropy` | `-Σ_j w_j log w_j`, natural log, `0·log0 := 0` |
| `zero_weight_features` | sorted list, expected empty |
| `constant_feature_guard_features` | sorted list, expected empty |

### 9.5 `top_weighted_mismatch_features`

This needs an exact definition, because "which feature drove the dissimilarity"
is not well defined for a nearest-neighbour distance without one.

**Adopted definition — mean weighted squared contribution at the nearest
neighbour:**

```
For each target cell x with nearest source reference cell x*(x):
    c_j(x) = w_j · ( z_j(x) − z_j(x*(x)) )²          for numeric j
    c_landcover(x) = w_landcover · 1[ ℓ(x) ≠ ℓ(x*(x)) ]

contribution_j = mean over assessable target cells of c_j(x)
share_j        = contribution_j / Σ_k contribution_k
```

`top_weighted_mismatch_features` is the list of `(feature, contribution, share)`
sorted by `contribution` descending, then by `feature` name ascending for a
deterministic tie-break. By construction `Σ_j share_j = 1`, and
`Σ_j contribution_j = mean(DI(x)² ) · normaliser²` — i.e. the decomposition is
exact against the mean squared nearest distance, which a validator check asserts.

This is the weighted analogue of the existing `top_outside_support_features`, and
the two must be reported side by side rather than conflated: the existing one
ranks by *fraction of cells outside a marginal range*, the new one by *share of
squared joint distance*. **Different quantities, different names, both kept.**

Tie-breaking note: `x*(x)` may be non-unique when several source cells sit at the
identical minimum distance. The contribution decomposition then depends on which
is returned. Policy: break ties by the smallest `(row_500m, col_500m)` of the
source cell, applied deterministically, and record
`nearest_neighbour_tie_break = "min_source_row_then_col"`. Exact ties are
vanishingly unlikely in continuous predictors but the rule must exist.

---

## 10. Directionality

`d_w(source → target)` is **not** symmetric, for three independent reasons:

1. `z` uses **source** mean and scale.
2. `w` comes from the **source** model.
3. The nearest-neighbour search is target-cell → source-reference-set.

Therefore `A → B` and `B → A` are different analyses with different weights,
different coordinate systems and different thresholds.

**Enforced in the schema and tests:**

- Every row carries `source_experiment` and `target_experiment` as separate
  columns; no sorted or unordered pair token is ever constructed.
- The pair token is `f"{source}__{target}"`, never sorted — matching the existing
  `test_pair_token_is_never_sorted` convention.
- A test asserts that reversing a synthetic pair with asymmetric weights produces
  a **different** DI, and that the implementation contains no code path that
  averages, sorts or symmetrises the directed pair.
- The 12 rows are generated with `itertools.permutations` over the sorted resolved
  experiment IDs, exactly as `marginal_aoa.v1` does, so caller argument order
  cannot affect the result.

Contrast with the other two components: climatic distance and geographic distance
**are** symmetric and their tests assert equality on reversal. The three
components are deliberately not required to share a symmetry property, and the
artifact records which is which:

```
weighted_predictor_space_dissimilarity  directed
climatic_distance                       symmetric
geographic_distance                     symmetric
```

---

## 11. What this component does not claim

Required limitations block for the weighted artifact:

- The weighting is source-model-informed. Feature weights come from a model
  fitted on the **source** label; this is not a label-free geometric statement
  about the predictor spaces.
- Impurity importance is in-sample and biased toward continuous predictors
  relative to low-cardinality categorical ones (doc 02 §3).
- A low DI does not guarantee transfer success, and a high DI does not prove it
  caused transfer failure. No causal relationship is established.
- The mixed numeric/categorical distance inherits the standard Gower scale
  asymmetry: the numeric block is unbounded, the categorical term is bounded by
  `w_landcover`.
- The DI **scale** is set by the mean pairwise source distance, which is a
  property of the source distribution alone and does not depend on the fold
  structure or on sampling density.
- The **threshold**, unlike the scale, does depend on the source spatial-fold
  structure; a different block size would move it. The ≈5 km choice is
  preregistered, not optimised, and both the whisker inputs (`Q1`, `Q3`, `IQR`)
  and the secondary quantile thresholds are reported so the sensitivity is
  visible.
- DI is a *nearest-neighbour* distance and therefore says nothing about the
  **density** of source data near a target cell. A target cell adjacent to one
  isolated source cell scores as well supported.
- Nothing here supersedes `marginal_aoa.v1`. Marginal range support and joint
  nearest-neighbour dissimilarity answer different questions and can disagree;
  where they do, both are reported.
