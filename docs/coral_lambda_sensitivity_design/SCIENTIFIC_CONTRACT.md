# Scientific Contract — `coral_lambda_sensitivity.v1`

Every decision below is **frozen**. Thresholds and tokens are declared here,
before any result exists, and may not be changed after seeing results.

---

## 1. Diagnostic class and what this is not

```
DIAGNOSTIC_CLASS = "coral_regularisation_parameter_sensitivity"
```

This asks how sensitive the existing `coral_after_regionwise_zscore` result is
to the CORAL covariance-regularisation parameter λ.

**It is not** hyperparameter selection, not target-label tuning, and not model
selection. Specifically, and bindingly:

- **No "best λ" or "optimal λ" is chosen or named.** There is no argmax over
  target performance anywhere in the code or the report.
- **The canonical λ = 1e-5 is not changed.** `core/config.py:697` is not
  edited, and `STEP10_CORAL_LAMBDA` is never monkey-patched.
- **No existing Step10 artifact or preregistration file is modified**, copied
  into this namespace, or re-run. References are bound by sha256 only.

## 2. Scope

| Item | Frozen value |
|---|---|
| Primary population | `burnable_tree_shrub_grass` |
| Valid universe | `valid_for_modeling == True` |
| AOIs | `manavgat_2021`, `bejis_2022`, `mugla_2021` |
| Excluded | `evia_2021`, `evia_2021_extended`, `kozan_2023` — hard exclusion, asserted |
| Model families | `baseline` (4 features, 3 numeric), `thermal` (10 features, 9 numeric) |
| Estimator | `random_forest` via `build_pipeline`, seed 42 |
| Adaptation method under study | `coral_after_regionwise_zscore` |

**Directions — exactly four:**

| Tier | Direction |
|---|---|
| primary | `bejis_2022_to_mugla_2021` |
| primary | `mugla_2021_to_bejis_2022` |
| secondary | `manavgat_2021_to_mugla_2021` |
| secondary | `mugla_2021_to_manavgat_2021` |

Primary/secondary is a **reporting emphasis only**: both tiers use the same
grid, the same gate, the same bootstrap and the same summarisation. Nothing is
computed for one that is not computed for the other.

`manavgat_2021 ↔ bejis_2022` is **not re-run** and contributes **no row** to
`metrics.csv`, `bootstrap_summary.csv` or `sensitivity_summary.csv`. Its
existing Step10 artifact is inventoried in `repository_inventory.json` for
context and provenance completeness only, behind an explicit direction
allow-list.

**Canonical Step8A hashes — verified 2026-08-03, all three match:**

```
manavgat_2021  054a1961fc0582a33d36413263668b63074b21ae8b03d12269b6e228787f3439
bejis_2022     3dec785a7d8e31db2d67ed283546bbfbca1559f56df46663488d0afc24d9e393
mugla_2021     c4ab107db2207f9f20775ccc0b3bf39381173fd07d4e82f6821ce7f40be7db8e
```

## 3. λ — the frozen definition

λ is the **additive ridge applied to both covariance matrices**, exactly as in
`core/step10_shared.py:192–193`:

```
Cs = cov(Xs_z, rowvar=False, ddof=0) + λ·I_d
Ct = cov(Xt_z, rowvar=False, ddof=0) + λ·I_d
A  = Cs^(-1/2) · Ct^(+1/2)
Xs_coral = Xs_z · A          Xt_coral = Xt_z   (target never transformed)
```

λ is **not** an interpolation strength, a blending coefficient, a model
regulariser, a RandomForest hyperparameter, or a Ledoit–Wolf-style convex
shrinkage. See `CORAL_FORMULA_AUDIT.md` for the exhaustive audit and the
ruling-out of each alternative.

The implementation must pass `lambda_` **explicitly** to
`fit_coral_alignment(...)`. It must never mutate `STEP10_CORAL_LAMBDA`, which
is also read by `step10a_preregistration_and_audit` and could contaminate a
preregistration artifact.

### 3.1 The eigenvalue floor is separate and must be instrumented

`_sym_matrix_power` clips eigenvalues at `eps = 1e-12` *after* the ridge
(`step10_shared.py:183`). This is a pre-existing second regulariser. This
design neither removes nor extends it, but **records per cell whether the clip
actually bound**, so that a λ=0 row can be reported as genuinely unregularised
rather than silently floored. Measured: it never binds on these data
(`NUMERICAL_FEASIBILITY.md` §3).

## 4. Frozen λ grid

```
LAMBDA_GRID = (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
CANONICAL_LAMBDA = 1e-5          # index 4
```

- Numeric ascending, deterministic, exactly 9 values.
- Every value enters the hashed scientific config, hence the `analysis_id`.
- **The grid is not revised after seeing results.**

**λ = 0 rules.**

- It is an unregularised-CORAL diagnostic, nothing more; it is never preferred.
- If the covariance is singular or the transform non-finite, the cell is **not**
  dropped: it is retained with NA metrics and an explicit `numerical_status`.
- **No small positive fallback may be substituted for 0.** Not `1e-12`, not
  `eps`, not anything.

**Token mapping** — floats never appear in a path:

| λ | token |
|---|---|
| 0 | `lambda_0` |
| 1e-8 | `lambda_1e_m8` |
| 1e-7 | `lambda_1e_m7` |
| 1e-6 | `lambda_1e_m6` |
| 1e-5 | `lambda_1e_m5` |
| 1e-4 | `lambda_1e_m4` |
| 1e-3 | `lambda_1e_m3` |
| 1e-2 | `lambda_1e_m2` |
| 1e-1 | `lambda_1e_m1` |

## 5. Held-fixed contract

λ is the **only** thing that varies. Frozen and asserted:

canonical Step8A datasets · primary population · feature contracts
(`SHARED_BASELINE_FEATURES`, `SHARED_THERMAL_MODEL_FEATURES`) · numeric feature
**order** · RandomForest implementation, hyperparameters and seed 42 ·
source/target cohorts · `compute_regionwise_zscore_stats` /
`apply_regionwise_zscore` unchanged · `EPSILON_STD = 1e-12` constant-feature
guard · missing-value handling (region-own-mean fill before standardisation) ·
target-label firewall · prediction/evaluation cohort · target spatial-block
bootstrap plan (1000, seed 42, `spatial_block_id`) · metric implementations ·
threshold policy (there is none — all metrics are threshold-free) · direction
path-resolution rule.

## 6. Target-label firewall

The target label `burned` is:

- **not** used for z-score statistics — `compute_regionwise_zscore_stats(X, …)`
  has no `y` parameter, and that signature is itself part of the firewall;
- **not** used in any covariance — `fit_coral_alignment(Xs, Xt, λ)` has no `y`;
- **not** used in the CORAL transform — `apply_coral(Xs, fit)` has no `y`;
- **not** used to choose λ — λ is a fixed predeclared grid, never selected;
- **not** used in model fitting — only the **source** label `y_source` is;
- loaded **only** at final evaluation and bootstrap scoring.

Enforcement: `assert_label_blind(df, context)` (`step10_shared.py:243`) is
called on every target frame before any adapt/fit/predict call and on the
prediction output, exactly as Step10B does. The sensitivity run adds one more
call site: on the per-λ target frame immediately before `predict_proba`.

## 7. Metrics and estimands

Metrics: **ROC-AUC**, **PR-AUC**, **Brier score**.

Primary estimand: **thermal ROC-AUC sensitivity across λ**. All other
direction × family × metric combinations are reported in full and identically;
"primary" governs report ordering only.

For each direction × family × λ × metric:

```
metric_value
raw_reference_value                 (adaptation_method = raw_source_only, canonical artifact)
zscore_reference_value              (adaptation_method = regionwise_zscore, canonical artifact)
canonical_coral_reference_value     (adaptation_method = coral..., λ=1e-5, canonical artifact)
delta_vs_raw
delta_vs_zscore
delta_vs_canonical_lambda
```

### 7.1 Orientation

```
roc_auc, pr_auc :  delta = candidate − reference                (both natural and oriented)
brier_score     :  natural_delta  = candidate_brier − reference_brier
                   oriented_delta = reference_brier − candidate_brier
```

Brier is always **stored** in its natural lower-is-better form in every
`*_value` field; only the delta is oriented. After orientation, for every
metric without exception:

```
oriented_delta > 0  →  the candidate is better
oriented_delta < 0  →  the candidate is worse
```

### 7.2 Brier is not in the canonical Step10 artifact — declared handling

`core/step10_shared.compute_threshold_free_metrics` returns only `roc_auc` and
`pr_auc`. Neither `step10_metrics.csv` nor `step10_bootstrap_replicates.parquet`
contains a Brier column, for any method. (The only Brier values in the Step10
outputs are Step9B raw-transfer values carried in as provenance.)

Frozen handling:

1. **All three Brier reference values** (`raw`, `zscore`, `canonical_coral`) are
   **recomputed** from the resolved `step10_predictions.parquet` probability
   vectors joined to the canonical Step8A labels. This is exact, not
   approximate: probabilities round-trip through parquet bit-exactly, and the
   same recomputation reproduces the stored ROC-AUC/PR-AUC to 5.6e-17
   (`NUMERICAL_FEASIBILITY.md` §6.1).
2. Every Brier reference row is flagged `reference_source = "recomputed_from_persisted_probabilities"`,
   while ROC-AUC and PR-AUC rows are flagged `"read_from_step10_metrics_csv"`.
   The distinction is visible in `metrics.csv` and checked by the validator.
3. The **Brier bootstrap cannot reuse** `step10_bootstrap_replicates.parquet`,
   which has no Brier columns. New replicates are computed — still with **no
   model refit**, by rescoring persisted probability vectors under the
   reproduced block draws (§8).
4. Brier's canonical-λ reproduction is checked in Tier 1 against its own
   recomputation, not against a stored value that does not exist.

## 8. Bootstrap contract

Reuses the existing Step10 scheme (`run_n_way_paired_bootstrap`,
`step10_shared.py:267`) unchanged in every respect that defines the draws:

| Property | Value |
|---|---|
| Replicates | 1000 |
| Seed | 42 (`STEP10_RANDOM_STATE`) |
| Block column | `spatial_block_id` (target region, 2-cell canonical blocks) |
| Resampling | blocks drawn with replacement, `size = n_blocks` |
| Generator | `np.random.default_rng(42)`, one `rng.choice` per replicate, sequential |
| Model refit | **none** — persisted probability vectors are rescored |
| Invalid replicate | single-class resample: counted, skipped, **no retry** |
| Interval | 2.5 / 97.5 percentiles |

**Shared draws requirement.** Within one target direction, *every* λ and
*every* reference method must be scored on the **same** block draws in the
**same** replicate. The only way to guarantee this with the existing function
is to pass all series to a **single** `run_n_way_paired_bootstrap` call via an
extended `prob_columns` mapping — never one call per λ, which would restart the
generator and destroy pairing. The draws additionally depend on the order of
`df[block_col].unique()`, so the merged frame's row order is frozen and
asserted.

Paired deltas computed per replicate:

```
λ-CORAL − raw
λ-CORAL − z-score
λ-CORAL − canonical λ=1e-5
```

Reported per contrast: point estimate, p2.5, p97.5, valid replicate count,
invalid replicate count.

### 8.1 Permitted wording

```
bootstrap-supported positive       interval lies entirely above 0
bootstrap-supported negative       interval lies entirely below 0
interval includes zero / uncertainty remains
```

**Forbidden anywhere in the outputs:** `statistically significant`,
`significance`, `p-value`, `proven`, `optimal λ`, `best λ`, `confidence
interval`. A literal token scan enforces this.

## 9. Canonical reproduction gate

Two tiers, both mandatory, both run before any grid point other than λ=1e-5 is
computed. Full derivation in `NUMERICAL_FEASIBILITY.md` §6.

```
Tier 1 — metric layer, exact
  metrics recomputed from the resolved persisted probability vectors
  == stored step10_metrics.csv          tolerance <= 1e-12
  (measured: 5.551e-17 roc_auc, 9.714e-17 pr_auc)

Tier 2 — refit reproduction at λ=1e-5
  cell_id coverage                       exact set equality
  labels                                 exact equality
  probabilities finite                   no NaN/inf
  probability vectors                    <= 1e-12
  ROC-AUC, PR-AUC                        <= 1e-06  AND  <= 8/(n_pos·n_neg)
  Brier                                  <= 1e-09
```

**Why the metric tolerance is 1e-06 and not 1e-12.** Two executions of the
*identical* canonical pipeline — the two duplicate Step10 artifacts — differ by
up to **4.867e-08** in ROC-AUC, because `RandomForestClassifier(n_jobs=-1)`
sums per-tree probabilities in thread-scheduling order and the resulting ~1 ULP
probability differences flip near-tied ranks. A 1e-12 metric tolerance is
therefore unattainable by *any* execution, not just by this one. The 1e-12
tolerance requested for **probabilities** is retained and holds with ~2250×
margin. The additional `8/(n_pos·n_neg)` condition ties the gate to the rank
quantum of each direction so it tightens on larger cohorts.

**If either tier fails, the sensitivity grid must not run.** The failing tier,
the observed deviations and the tolerances are written to
`canonical_reproduction.csv` either way.

## 10. Numerical diagnostics

Recorded for every direction × family × λ (72 rows):

`source_covariance_shape`, `target_covariance_shape`,
`min_eigenvalue_Cs_before_ridge`, `min_eigenvalue_Ct_before_ridge`,
`min_eigenvalue_Cs_after_ridge`, `min_eigenvalue_Ct_after_ridge`,
`condition_number_Cs_before`, `condition_number_Cs_after`,
`condition_number_Ct_before`, `condition_number_Ct_after`,
`eigenvalue_floor_bound_Cs`, `eigenvalue_floor_bound_Ct`,
`matrix_sqrt_finite`, `inverse_sqrt_finite`, `coral_transform_finite`,
`transformed_source_finite`, `transformed_covariance_mismatch_frobenius`,
`max_abs_transformed_value`, `prediction_probability_finite`,
`numerical_status`.

```
ALLOWED_NUMERICAL_STATUS = (
    "pass",
    "singular_unregularised_covariance",
    "nonfinite_matrix_transform",
    "nonfinite_transformed_features",
    "model_fit_failure",
)
```

A failing cell is **retained** with NA metrics and its status. It is never
deleted, and it never aborts the grid: the two `Step10Error` raises inside
`fit_coral_alignment` / `apply_coral` are caught per cell and mapped onto these
statuses.

## 11. Sensitivity interpretation — no λ is selected

For each direction × family × metric, report only these descriptive summaries:

`canonical_lambda_value` · `grid_min` · `grid_max` · `grid_range` ·
`max_abs_deviation_from_canonical` · `canonical_rank_within_finite_grid` ·
`sign_pattern_delta_vs_zscore` · `n_lambda_interval_excludes_zero_positive` ·
`n_lambda_interval_excludes_zero_negative` · `n_lambda_interval_includes_zero` ·
`n_numerical_failures`.

### 11.1 Predeclared magnitude thresholds

Applied to `max_abs_deviation_from_canonical`, over the **finite** grid cells
only:

**ROC-AUC and PR-AUC** — both are unit-free quantities on [0, 1]:

```
insensitive_over_grid        <= 0.005
modest_lambda_sensitivity    >  0.005  and  <= 0.020
material_lambda_sensitivity  >  0.020
```

**Brier — scale-aware.** A Brier score is a mean squared error whose natural
scale is set by the target's own prevalence: the constant-prevalence forecast
scores exactly `p(1−p)`. Because our three targets differ in prevalence by
nearly a factor of two, a single absolute Brier threshold would be strict for
Manavgat and lax for Muğla. The threshold is therefore expressed as a fraction
of the target's `p(1−p)`, which puts it on the same relative footing as the
AUC thresholds:

```
deviation_ratio = max_abs_deviation_from_canonical / (p_target · (1 − p_target))

insensitive_over_grid        deviation_ratio <= 0.005
modest_lambda_sensitivity    0.005 < deviation_ratio <= 0.020
material_lambda_sensitivity  deviation_ratio > 0.020
```

Absolute equivalents, frozen from the canonical prevalences:

| Target | p | p(1−p) | insensitive ≤ | modest ≤ |
|---|---:|---:|---:|---:|
| `mugla_2021` | 0.06975797 | 0.06489179 | 3.245e-04 | 1.298e-03 |
| `bejis_2022` | 0.07241606 | 0.06717198 | 3.359e-04 | 1.343e-03 |
| `manavgat_2021` | 0.03822339 | 0.03676236 | 1.838e-04 | 7.352e-04 |

`p_target` is computed from the canonical Step8A primary population of the
**target** region and is itself hashed into the config.

### 11.2 The fourth token

```
numerical_instability_present
```
is assigned whenever `n_numerical_failures > 0` for that direction × family ×
metric, **in addition to** whichever magnitude token applies. It is not
mutually exclusive with the other three.

### 11.3 These tokens are not inferential

The four tokens are descriptive labels for the size of a deviation across a
predeclared grid. They are not test outcomes, carry no error rate, and support
no claim about which λ is correct. Thresholds were fixed in this document
before any sensitivity result existed and **may not be changed afterwards**.

## 12. Fail-closed conditions

The run aborts, writing nothing, if any of these holds:

- any of the three Step8A digests mismatches;
- any resolved Step10 reference artifact digest mismatches its plan-time value;
- an AOI outside the three primaries is referenced, or `manavgat↔bejis` reaches
  a sensitivity output row;
- the λ grid is not exactly the 9 frozen values in ascending order, or λ=0 has
  been substituted;
- the numeric feature order differs from the canonical order;
- `STEP10_CORAL_LAMBDA` has been mutated;
- a target frame containing `burned` reaches any adapt/fit/predict call;
- either reproduction tier fails;
- a direction × family × λ partition is missing;
- the bootstrap was invoked more than once per direction, or the block draws
  differ between series;
- scientific fits exceed 72;
- an output path resolves outside the analysis namespace;
- a forbidden token appears in any emitted text.
