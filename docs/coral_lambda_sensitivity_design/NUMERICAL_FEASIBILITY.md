# Numerical Feasibility — measured, not assumed

Every number below was computed on **2026-08-03** by read-only evaluation of
the canonical Step8A frames through the repository's own
`compute_regionwise_zscore_stats` / `apply_regionwise_zscore` and the exact
CORAL algebra of `core/step10_shared.py`. **No model was fitted, no bootstrap
was run, no file was written.**

---

## 1. What was computed

For each of the 4 directions × 2 model families, the source and target frames
were region-wise z-scored exactly as Step10B does, restricted to the numeric
feature columns, and then for every λ in the frozen grid:

- `Cs = cov(Xs_z, ddof=0) + λI`, `Ct = cov(Xt_z, ddof=0) + λI`
- eigenvalues of both, before and after the ridge
- condition numbers before and after
- whether the `_sym_matrix_power` eigenvalue floor (1e-12) actually bound
- `A = Cs^-1/2 · Ct^1/2`, and finiteness of `A`
- `Xs_coral = Xs_z · A`, finiteness, max |value|
- the alignment residual `‖cov(Xs_coral, ddof=0) − cov(Xt_z, ddof=0)‖_F`

Numeric dimensionality: **d = 3** for baseline
(`ndvi_mean, elevation_mean, slope_mean`) and **d = 9** for thermal
(`landcover_dominant` is categorical and excluded).

## 2. Pre-ridge covariance spectra — the decisive result

| Direction | Family | d | n_src | n_tgt | min eig `Cs` | cond `Cs` | min eig `Ct` | cond `Ct` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bejís→muğla | baseline | 3 | 15,190 | 41,730 | 4.639602e-01 | 3.27 | 6.756641e-01 | 2.15 |
| bejís→muğla | thermal | 9 | 15,190 | 41,730 | 7.150158e-03 | 7.18e+02 | 1.713164e-03 | 2.93e+03 |
| muğla→bejís | baseline | 3 | 41,730 | 15,190 | 6.756641e-01 | 2.15 | 4.639602e-01 | 3.27 |
| muğla→bejís | thermal | 9 | 41,730 | 15,190 | 1.713164e-03 | 2.93e+03 | 7.150158e-03 | 7.18e+02 |
| manavgat→muğla | baseline | 3 | 20,511 | 41,730 | 4.268497e-01 | 3.32 | 6.756641e-01 | 2.15 |
| manavgat→muğla | thermal | 9 | 20,511 | 41,730 | 2.958178e-03 | 1.72e+03 | 1.713164e-03 | 2.93e+03 |
| muğla→manavgat | baseline | 3 | 41,730 | 20,511 | 6.756641e-01 | 2.15 | 4.268497e-01 | 3.32 |
| muğla→manavgat | thermal | 9 | 41,730 | 20,511 | 1.713164e-03 | 2.93e+03 | 2.958178e-03 | 1.72e+03 |

**Smallest eigenvalue anywhere in the study: 1.713164 × 10⁻³** (muğla thermal).
**Largest condition number anywhere: 2.934 × 10³.**

Both are extremely benign. For context, `float64` has ~2.2e-16 relative
precision, so a condition number of 2.9e3 costs about 3.5 decimal digits — the
inverse square root is computed with ~12 significant digits of headroom.

## 3. λ = 0 — is it genuinely unregularised here?

**Yes, on these data — and it is measured, not assumed.**

The concern raised in `CORAL_FORMULA_AUDIT.md` §3 is real: `_sym_matrix_power`
clips eigenvalues at `eps = 1e-12` *after* the ridge, so a singular covariance
would be silently floored rather than failing. Across all 72 (direction ×
family × λ) cells:

| Check | Result |
|---|---|
| Eigenvalue floor ever bound (`min eig < 1e-12`) | **never** — 0 of 72 |
| `A` complex or non-finite | **never** — 0 of 72 |
| `Xs_coral` non-finite | **never** — 0 of 72 |
| Minimum margin over the floor | 1.713e-03 / 1e-12 = **1.7 × 10⁹×** |

So at λ=0 the transform is a true unregularised CORAL: `Cs` and `Ct` are
positive definite with a nine-order-of-magnitude margin, and the clip is
inert.

**Expected `numerical_status` for all 72 cells: `pass`.** The other statuses
(`singular_unregularised_covariance`, `nonfinite_matrix_transform`,
`nonfinite_transformed_features`, `model_fit_failure`) remain in the schema as
fail-closed paths and must still be implemented; they are simply not expected
to fire. The run must **record** the floor-activation flag per cell so that the
"λ=0 was genuinely unregularised" claim is evidenced rather than asserted.

This also means: no small-positive fallback is needed, none is permitted, and
the grid does not need to be revised after seeing results.

## 4. λ = 0 remains a *diagnostic*, not a recommendation

Even though it is numerically safe here, λ=0 is reported as an unregularised
diagnostic only. It carries no preference over the canonical λ=1e-5, and the
analysis selects no λ.

## 5. How much does λ actually perturb the alignment?

Thermal, bejís→muğla (the highest-condition direction). `mismatch_F` is
`‖cov(Xs_coral) − cov(Xt_z)‖_F`, i.e. how far CORAL falls short of matching the
target covariance exactly:

| λ | cond `Cs` | cond `Ct` | mismatch_F | max abs transformed |
|---:|---:|---:|---:|---:|
| 0 | 718.359 | 2933.920 | 1.181e-14 | 14.4157 |
| 1e-8 | 718.358 | 2933.903 | 7.572e-08 | 14.4157 |
| 1e-7 | 718.349 | 2933.749 | 7.572e-07 | 14.4157 |
| 1e-6 | 718.259 | 2932.209 | 7.571e-06 | 14.4157 |
| **1e-5** | **717.357** | **2916.899** | **7.564e-05** | **14.4158** |
| 1e-4 | 708.465 | 2772.163 | 7.497e-04 | 14.4167 |
| 1e-3 | 630.341 | 1852.924 | 6.897e-03 | 14.4269 |
| 1e-2 | 300.078 | 429.968 | 4.190e-02 | 14.5529 |
| 1e-1 | 48.870 | 50.399 | 1.560e-01 | 15.5504 |

Reading: at λ=0 CORAL matches the target covariance to machine precision
(1.2e-14). The residual grows essentially linearly in λ over the small-λ
regime. At the canonical λ=1e-5 the condition number moves by 0.14 % and the
mismatch is 7.6e-05 — a genuinely negligible perturbation. Only λ ≥ 1e-2
changes the operator materially (cond falls by more than half; at λ=1e-1 the
mismatch is 0.156 and the transform is being pulled toward the identity).

This is a statement about the **linear algebra only**. It predicts nothing
about the metrics, and no metric expectation is declared anywhere in this
design.

Transformed-feature magnitudes stay bounded (max |value| ≈ 14.4–15.6), so
there is no overflow risk into the RandomForest.

## 6. Canonical reproduction gate — feasibility, and a required correction

### 6.1 Tier 1 — metric layer, exact: **FEASIBLE at ≤ 1e-12** ✅

Recomputing ROC-AUC and PR-AUC directly from the persisted
`step10_predictions.parquet` probability vectors, joined to the canonical
Step8A labels, reproduces the stored `step10_metrics.csv` values for all
4 directions × 2 families × 3 methods:

```
MAX |recomputed − stored|   roc_auc = 5.551e-17     pr_auc = 9.714e-17
```

Both are inside the 1e-12 tolerance by four orders of magnitude. Parquet
round-trips `float64` exactly, so this tier validates the metric layer, the
label join and the cell-coverage contract with no tolerance debate. Brier is
computable on the same vectors with the same exactness.

### 6.2 Tier 2 — refit reproduction: **≤ 1e-12 on metrics is NOT ACHIEVABLE**

This is a genuine blocker against §6 of the task as written, and it is measured
rather than argued.

The two duplicate Step10 artifacts of each pair (§2 of
`REFERENCE_ARTIFACTS.md`) *are* two executions of the identical canonical
pipeline on identical inputs. They differ:

| Quantity | Measured max difference between two canonical runs |
|---|---:|
| prediction probability | **4.441e-16** (≈ 2 ULP) |
| ROC-AUC | **4.867e-08** |
| PR-AUC | **1.618e-08** |

Cause: `RandomForestClassifier(n_jobs=-1)` accumulates per-tree probabilities
under a lock in thread-completion order, so float addition is
non-associative across runs. Those ULP differences flip near-tied ranks, and
one rank flip moves ROC-AUC by exactly `1/(n_pos·n_neg)`:

| Target | `1/(n_pos·n_neg)` |
|---|---:|
| `mugla_2021` (2,911 × 38,819) | 8.849e-09 |
| `bejis_2022` (1,100 × 14,090) | 6.452e-08 |
| `manavgat_2021` (784 × 19,727) | 6.466e-08 |

4.867e-08 ÷ 8.849e-09 ≈ 5.5 flips — fully consistent. **A fresh λ=1e-5 refit
therefore cannot be required to match a stored ROC-AUC to 1e-12; no execution
of the canonical pipeline can.**

### 6.3 Proposed corrected gate — measurement-derived

```
Tier 1  (exact, mandatory)
  metrics recomputed FROM the resolved persisted probability vectors
  must equal the stored step10_metrics.csv values to  <= 1e-12.
  Applies to: 4 directions x 2 families x 3 methods, ROC-AUC and PR-AUC.
  Measured headroom: 5.6e-17 observed vs 1e-12 allowed.

Tier 2  (refit, mandatory)
  a fresh lambda=1e-5 CORAL refit must reproduce the resolved artifact:
    probability vectors   <=  1e-12      (measured envelope 4.44e-16, ~2250x margin)
    ROC-AUC / PR-AUC      <=  1e-06      (measured envelope 4.87e-08,  ~20x margin)
                          AND  <= 8 / (n_pos * n_neg)  for that direction
    Brier                 <=  1e-09      (a smooth mean; no rank sensitivity)
    cell_id coverage       exact set equality, no tolerance
    labels                 exact equality, no tolerance
    probabilities          all finite, no tolerance
```

The dual ROC/PR condition (absolute **and** rank-quantised) is what keeps the
gate honest: `8/(n_pos·n_neg)` is 7.1e-08 for a muğla target and 5.2e-07 for a
manavgat target, so the gate tightens automatically on the larger cohort rather
than hiding behind a single loose constant.

**If Tier 1 or Tier 2 fails, the sensitivity grid must not run.** The 1e-12
probability tolerance requested in §6 of the task is retained and is
comfortably achievable; only the metric tolerance is relaxed, with the
measurement that forces it recorded in `canonical_reproduction.csv`.

### 6.4 Rejected alternative

Setting `n_jobs=1` would make a *fresh* run bitwise reproducible, but the
stored artifacts were produced with `n_jobs=-1`, so it would not make the
fresh run match **them**. It would also change the frozen model contract.
Rejected.

## 7. Expected fit count

| Quantity | Value |
|---|---:|
| Directions | 4 |
| Model families | 2 |
| λ values | 9 |
| **Maximum unique scientific fits** | **4 × 2 × 9 = 72** |

**No reuse is available across λ**, because every λ changes `A` and therefore
the training frame `X_source_coral`. Within a single (direction, family) the
z-score statistics and `Xs_z` / `Xt_z` are λ-independent and are computed once
and reused — but they are not *fits*.

**CORAL transforms must never be shared across directions**, even where the
source region is the same: `A = Cs^-1/2 · Ct^1/2` depends on the **target**
covariance, so `muğla→bejís` and `muğla→manavgat` have different `A` despite a
common source. The fit identity is therefore
`(direction, model_family, lambda_token)` with no reduction, and 72 is both the
maximum and the expected count.

Audit fits (the Tier-2 reproduction refit, 4 directions × 2 families = 8) are
counted and reported **separately** and are not part of the 72.

## 8. Runtime, memory, disk

**Runtime.** Estimated from RandomForest timings already measured in this
repository on the same `build_pipeline` constructor (1.22 s at 16.4 k rows,
2.14 s at 20.5 k rows, same 10-feature frame), scaled by source row count. No
model was fitted for this document.

| Stage | Work | Estimate |
|---|---|---:|
| `plan` | load 3 frames, hash ~40 artifacts, Tier-1 reproduction, z-score stats | ≈ 2 min |
| `fit` — bejís source (15,190 rows) | 18 fits × ~1.2 s | ≈ 22 s |
| `fit` — manavgat source (20,511 rows) | 18 fits × ~2.1 s | ≈ 38 s |
| `fit` — muğla source (41,730 rows) × 2 directions | 36 fits × ~3.5 s | ≈ 126 s |
| `fit` — CORAL algebra | 72 × (9×9 `eigh` + one GEMM) | < 5 s, negligible |
| `fit` — prediction + Brier/ROC/PR | 72 × up to 41,730 rows | ≈ 30 s |
| **fit subtotal** | **72 scientific + 8 audit fits** | **≈ 5–7 min** |
| `bootstrap` | 4 directions × 1,000 replicates × ~66 series × 3 metrics | **≈ 30–45 min** |
| `summarize` | 216 metric rows → summary | < 10 s |
| **total** | | **≈ 40–55 min** |

The bootstrap dominates by an order of magnitude. Each replicate must score
every series on the resampled rows, and ROC-AUC costs an `O(n log n)` sort per
series. Two implementation notes for the fit budget, neither of which changes
any number: sort the resampled label vector once per replicate and reuse the
ordering across series where the metric permits, and hold the probability
matrix as a single contiguous `float64` array rather than a per-series lookup.

**Memory.** Three Step8A frames ≈ 70 MB. The largest per-direction probability
matrix is 41,730 × 66 series × 8 B ≈ 22 MB. One RandomForest (300 trees,
41,730 × ~14 encoded columns) peaks at a few hundred MB during `fit`, and only
one is alive at a time. **Peak RSS ≈ 1.0–1.5 GB.** No chunking needed.

**Disk.**

| Output | Rows | Estimate |
|---|---:|---:|
| `predictions.parquet` (partitioned) | 4 dir × 2 fam × 9 λ × target rows = 2,144,898 | ≈ 27 MB |
| `bootstrap_replicates.parquet` | 4 × 1,000 = 4,000 rows × ~200 cols | ≈ 6 MB |
| `adaptation_statistics.parquet` | 72 | < 1 MB |
| `numerical_diagnostics.csv` | 72 | < 100 KB |
| `metrics.csv` | 4 × 2 × 9 × 3 = 216 | < 100 KB |
| `bootstrap_summary.csv` | ~600 | < 200 KB |
| `sensitivity_summary.csv` | 4 × 2 × 3 = 24 | < 20 KB |
| `canonical_reproduction.csv` | 4 × 2 × 3 × 2 tiers = 48 | < 20 KB |
| JSON + report | — | < 2 MB |
| **total** | | **≈ 35–40 MB** |

Row basis for `predictions.parquet`: per (family, λ) the four directions
contribute 41,730 + 15,190 + 41,730 + 20,511 = 119,161 target rows; × 2
families × 9 λ = 2,144,898. At the ~12.5 B/row density measured on the existing
`step10_predictions.parquet` (341,520 rows in 4.26 MB) that is ≈ 27 MB.

## 9. Feasibility verdict

| Precondition | Status |
|---|---|
| λ semantics unambiguous and located | ✅ ridge on `Cs` and `Ct`, `step10_shared.py:192–193` |
| Canonical λ = 1e-5 located | ✅ `core/config.py:697`, and in every preregistration artifact |
| All three Step8A digests match | ✅ |
| All four directions have a resolvable Step10 artifact | ✅ under the frozen source-first rule |
| Duplicate artifacts explained and resolved | ✅ ULP-level RF non-determinism, rule frozen |
| λ=0 numerically safe | ✅ min eigenvalue 1.7e-3, floor never binds |
| Eigenvalue floor instrumented rather than assumed | ✅ recorded per cell |
| Tier-1 exact reproduction | ✅ 5.6e-17 ≤ 1e-12 |
| Tier-2 refit reproduction at ≤1e-12 on metrics | ❌ **impossible** — corrected gate in §6.3 |
| Brier available from Step10 | ❌ **absent** — recomputed instead, see `SCIENTIFIC_CONTRACT.md` §7.2 |
| Runtime/memory/disk within ordinary bounds | ✅ ≈ 40–55 min, ≈ 1.5 GB, ≈ 40 MB |
