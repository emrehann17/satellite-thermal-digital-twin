# CORAL Formula Audit — exact existing semantics of λ

Everything below was read in this repository on **2026-08-03**. Nothing was
executed that wrote to a production path and no model was fitted.

**Verdict: the λ semantics are unambiguous and fully located. This is NOT a
blocker.** λ is the additive ridge on the covariance matrices, and on nothing
else.

---

## 1. Where the code is

| Item | File | Line | Value / definition |
|---|---|---:|---|
| Matrix square-root helper | `core/step10_shared.py` | 183 | `_sym_matrix_power(M, power, eps=1e-12)` |
| **CORAL fit** | `core/step10_shared.py` | **186** | `fit_coral_alignment(Xs_z_numeric, Xt_z_numeric, lambda_=STEP10_CORAL_LAMBDA)` |
| **λ enters Cs** | `core/step10_shared.py` | **192** | `Cs = np.cov(Xs_z_numeric, rowvar=False, ddof=0) + lambda_ * np.eye(d)` |
| **λ enters Ct** | `core/step10_shared.py` | **193** | `Ct = np.cov(Xt_z_numeric, rowvar=False, ddof=0) + lambda_ * np.eye(d)` |
| Alignment matrix | `core/step10_shared.py` | 197–200 | `A = _sym_matrix_power(Cs, -0.5) @ _sym_matrix_power(Ct, 0.5)` |
| Finiteness guard on `A` | `core/step10_shared.py` | 202–203 | raises `Step10Error` if complex or non-finite |
| λ recorded in diagnostics | `core/step10_shared.py` | 209 | `"lambda": lambda_` |
| **CORAL apply** | `core/step10_shared.py` | **213** | `apply_coral(Xs_z_numeric, coral_fit)` |
| Source transform | `core/step10_shared.py` | 214 | `Xs_coral = Xs_z_numeric @ coral_fit["A"]` |
| Finiteness guard on output | `core/step10_shared.py` | 215–216 | raises `Step10Error` if complex or non-finite |
| **Canonical λ constant** | `core/config.py` | **697** | `STEP10_CORAL_LAMBDA = 1e-5` |
| Call site (fit) | `src/step10b_label_blind_adaptation.py` | 151 | `fit_coral_alignment(X_source_z[numeric_feats].to_numpy(dtype=float), X_target_z[numeric_feats].to_numpy(dtype=float))` |
| Call site (apply) | `src/step10b_label_blind_adaptation.py` | 152 | `apply_coral(X_source_z[numeric_feats].to_numpy(dtype=float), coral_fit)` |
| λ into preregistration | `src/step10a_preregistration_and_audit.py` | 169 | `"lambda": STEP10_CORAL_LAMBDA` |
| λ in the frozen artifact | — | — | `step10_preregistration.json → /scientific_config/adaptation_methods/coral_after_regionwise_zscore/lambda = 1e-05` |

## 2. The exact mathematics, as implemented

Let `Xs_z ∈ R^{n_s×d}` and `Xt_z ∈ R^{n_t×d}` be the **region-wise
z-scored numeric** feature matrices (categorical `landcover_dominant`
excluded — see §5), and `I_d` the identity.

```
Cs = cov(Xs_z, rowvar=False, ddof=0) + λ·I_d          # step10_shared.py:192
Ct = cov(Xt_z, rowvar=False, ddof=0) + λ·I_d          # step10_shared.py:193

Cs = atleast_2d(Cs) ;  Ct = atleast_2d(Ct)            # 194–195

A  = Cs^(-1/2) · Ct^(+1/2)                            # 197–200

Xs_coral = Xs_z · A                                   # 214
Xt_coral = Xt_z                                       # step10b:155 — TARGET NEVER TRANSFORMED
```

This is the standard Sun & Saenko CORAL whitening-then-recolouring map, with
the recolouring taken toward the **target** covariance.

### 2.1 λ semantics — frozen statement

> **λ is the additive ridge (Tikhonov) term applied to BOTH the source and the
> target covariance matrices, immediately after `np.cov` and before any matrix
> square root is taken. It is added as `λ·I_d` to each covariance
> independently, with the same scalar λ. It is nothing else.**

λ is explicitly **NOT**:

| Not this | Why we can rule it out |
|---|---|
| interpolation strength between raw and CORAL | there is no interpolation anywhere; `Xs_coral = Xs_z @ A` is applied wholly |
| a prediction blending coefficient | probabilities are never blended; each method fits its own pipeline and predicts once |
| model regularisation | the estimator is `RandomForestClassifier`; λ never reaches `build_pipeline` |
| a RandomForest hyperparameter | `build_pipeline(feature_list, MODEL_NAME, random_state)` takes no λ |
| a covariance shrinkage coefficient (Ledoit–Wolf style) | there is no convex combination `(1-λ)·C + λ·target`; it is a pure additive ridge, so the trace grows by `λ·d` rather than being preserved |
| the eigenvalue floor | that is a **separate** constant `eps = 1e-12` inside `_sym_matrix_power` (see §3) |

## 3. The eigenvalue floor — a second, independent regulariser

`_sym_matrix_power` (line 183):

```python
def _sym_matrix_power(M, power, eps=1e-12):
    eigvals, eigvecs = np.linalg.eigh(M)      # symmetric -> always real
    eigvals_clipped = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(eigvals_clipped ** power) @ eigvecs.T
```

**This is the single most important subtlety in this analysis.** There is a
`np.clip(eigvals, 1e-12, None)` floor applied *after* the λ ridge. It means:

- λ = 0 does **not** guarantee a truly unregularised CORAL. If the
  unregularised covariance were singular, the floor would silently substitute
  `1e-12` for each non-positive eigenvalue and the transform would come back
  finite — i.e. the implementation already contains exactly the "small positive
  fallback" that §3 of the task forbids this design from adding.
- The floor is a property of the **existing frozen implementation**. This
  design does not remove it (that would change canonical λ=1e-5 behaviour), and
  does not add to it.
- Therefore λ=0 must be **instrumented, not assumed**: the run records whether
  the clip actually bound, so a λ=0 row can be reported as genuinely
  unregularised rather than silently floored.

Measured on the real canonical frames (see `NUMERICAL_FEASIBILITY.md`): the
smallest pre-ridge eigenvalue over all 4 directions × 2 families is
**1.713164 × 10⁻³**, nine orders of magnitude above the floor. **The clip never
binds anywhere on this grid**, so λ=0 is a genuine unregularised diagnostic on
these data. The instrumentation exists to prove that, not to hope it.

`eigenvalue_floor_used: 1e-12` is already recorded in the CORAL diagnostics
(`core/step10_shared.py:208`) and is carried into
`step10_adaptation_statistics.json`.

## 4. Covariance convention

| Property | Value | Evidence |
|---|---|---|
| Estimator | `numpy.cov` | `step10_shared.py:192–193` |
| Orientation | `rowvar=False` — rows are observations, columns are features | same lines |
| Normalisation | `ddof=0` — divides by `n`, the **biased/MLE** convention, **not** `n-1` | same lines |
| Centering | performed internally by `np.cov` (it subtracts the column means) | numpy semantics |
| Explicit de-centering | **none** — no mean is added back after `Xs_z @ A` | `step10_shared.py:214` |
| Mean handling | already ≈ 0 per column because the input is region-wise z-scored | `step10b:148–149` |
| Weights | none | no `aweights`/`fweights` argument |

Because the inputs are z-scored per region, both covariances are effectively
**correlation-like** matrices with unit-ish diagonals — which is why their
eigenvalues sit in the O(10⁻³ … 5) range and λ ≤ 1e-3 is a very small
perturbation.

## 5. Order of operations — the full pipeline for one (direction, family)

From `src/step10b_label_blind_adaptation.py:125–162`:

```
1.  X_source_raw = src_pop[feature_list]              # step10b:130
    X_target_raw = tgt_pop[feature_list]              # step10b:131

2.  source_stats = compute_regionwise_zscore_stats(X_source_raw, numeric_feats)   # :140
    target_stats = compute_regionwise_zscore_stats(X_target_raw, numeric_feats)   # :141
        - per feature: mean, std(ddof=0) over the region's own non-missing values
        - constant-feature guard: if std < EPSILON_STD (1e-12) then std := 1.0
        - LABEL-BLIND by signature: the function takes X only, no y

3.  X_source_z = apply_regionwise_zscore(X_source_raw, source_stats, numeric_feats)  # :142
    X_target_z = apply_regionwise_zscore(X_target_raw, target_stats, numeric_feats)  # :143
        - missing values are filled with the REGION'S OWN mean FIRST,
          so after the transform a previously-missing value is exactly 0.0
        - categorical landcover_dominant is NOT touched

4.  coral_fit = fit_coral_alignment(X_source_z[numeric].to_numpy(float),
                                    X_target_z[numeric].to_numpy(float))          # :151
        <-- λ ENTERS HERE, AND ONLY HERE

5.  Xs_coral_numeric = apply_coral(X_source_z[numeric].to_numpy(float), coral_fit) # :152

6.  X_source_coral = X_source_z.copy()
    X_source_coral[numeric_feats] = Xs_coral_numeric   # :153–154
    X_target_coral = X_target_z.copy()                 # :155  (target unchanged)

7.  pipeline_c = build_pipeline(feature_list, MODEL_NAME, random_state)  # :157
    pipeline_c.fit(X_source_coral, y_source)                            # :158
    prob_c = pipeline_c.predict_proba(X_target_coral)[:, 1]             # :159
```

**So: z-score first, CORAL second, model third.** The model is fitted on the
**CORAL-transformed source in the z-scored space**, with the untouched
categorical column re-attached, and predicts on the **z-scored (not
CORAL-transformed) target**. There is no other representation.

## 6. Feature ordering and dtype

| Property | Value |
|---|---|
| `FEATURE_LISTS["baseline"]` | `SHARED_BASELINE_FEATURES` — `ndvi_mean, elevation_mean, slope_mean, landcover_dominant` (`step10_shared.py:36`) |
| `FEATURE_LISTS["thermal"]` | `SHARED_THERMAL_MODEL_FEATURES` — the 4 baseline + 6 thermal columns (`step10_shared.py:36`) |
| Categorical | `CATEGORICAL_FEATURES = ["landcover_dominant"]`, imported from Step9A |
| Numeric pool | `_numeric_features(feature_list)` = `[f for f in feature_list if f not in CATEGORICAL_FEATURES]` |
| **Numeric d (baseline)** | **3** — `ndvi_mean, elevation_mean, slope_mean` |
| **Numeric d (thermal)** | **9** — the 3 above + the 6 thermal columns |
| Column order | the declared list order, preserved by `X[numeric_feats]`; recorded per direction/family as `coral_diagnostics.numeric_feature_order` |
| dtype into CORAL | `float64` — forced by `.to_numpy(dtype=float)` at `step10b:151–152` |
| Internal dtype | `float64` throughout (`np.cov`, `np.linalg.eigh`, `@`) |

The numeric feature **order is part of the contract**: `A` is a `d×d` matrix in
that basis, so a reordering would silently change the transform. It is frozen
and asserted.

## 7. Where λ does NOT appear

Verified by exhaustive grep over `src/`, `core/`, `scripts/`:

- `STEP10_CORAL_LAMBDA` is referenced in exactly three places:
  `core/step10_shared.py:33` (import) and `:187` (default argument),
  `src/step10a_preregistration_and_audit.py:39` (import) and `:169` (recorded).
- `fit_coral_alignment` is called from exactly one place:
  `src/step10b_label_blind_adaptation.py:151`, **without** an explicit
  `lambda_` argument — so the canonical value flows from the default.
- No other module reads, scales, anneals or overrides λ.

**Implication for implementation:** the sensitivity run must pass `lambda_`
explicitly to `fit_coral_alignment`. It must **not** monkey-patch
`STEP10_CORAL_LAMBDA`, because that constant is also read by
`step10a_preregistration_and_audit` and mutating it could contaminate a
preregistration artifact.

## 8. Numerical guards inventory

| Guard | Location | Behaviour |
|---|---|---|
| `EPSILON_STD = 1e-12` | `step10_shared.py:52` | z-score constant-feature guard: `std := 1.0` when `std < 1e-12`, recorded as `constant_feature_guard_used` |
| Missing-value fill | `apply_regionwise_zscore` (`:169`) | filled with the region's own mean **before** standardising ⇒ post-transform value is exactly `0.0` |
| Eigenvalue floor `eps=1e-12` | `_sym_matrix_power` (`:183`) | `np.clip(eigvals, 1e-12, None)` — see §3 |
| Complex / non-finite `A` | `fit_coral_alignment` (`:202`) | raises `Step10Error` |
| Complex / non-finite `Xs_coral` | `apply_coral` (`:215`) | raises `Step10Error`, then `np.real(...)` |
| Label firewall | `assert_label_blind` (`:243`) | raises if a `burned` column reaches a label-blind frame |

Note that the two `Step10Error` guards **raise** rather than return a status.
The sensitivity run must catch them per (direction, family, λ) cell and convert
them into the predeclared `numerical_status` values, so that a failing λ row is
retained with NA metrics instead of aborting the whole grid.

## 9. Model fitting representation — frozen answer

> The model is fitted on **`X_source_coral`**: the region-wise z-scored source
> frame whose *numeric* columns have been replaced by `Xs_z @ A`, with the raw
> categorical `landcover_dominant` column carried through unchanged. It
> predicts on **`X_target_z`** — the z-scored target, *not* CORAL-transformed.

`build_pipeline(feature_list, "random_forest", 42)` is the same Step8B/Step9B
constructor; the `ColumnTransformer` inside it then applies median imputation
to numerics and one-hot encoding to the categorical, fitted on the training
frame only.
