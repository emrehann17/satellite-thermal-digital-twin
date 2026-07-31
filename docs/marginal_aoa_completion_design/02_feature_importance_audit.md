# 02. Feature-Importance Source Audit

Machine-readable companion: `candidate_feature_importances.csv`.

---

## 1. Search performed

```
rg -c "permutation_importance"  (whole repo, excluding venv/ and old_codes/)
  → src/step7c_train_downscaling_model.py only  (the 30 m LST DOWNSCALING
    regressor — a different model, a different feature space, not the Step8
    burned-area association model; NOT usable here)

rg -i "shap|SHAP"               → no genuine hit; every match was `.shape`
rg -n "importance" src/step8b_train_baseline_vs_thermal_model.py
  → a single production path, quoted in §2
```

There is **no permutation importance, no out-of-fold importance, no held-out
importance, no SHAP importance and no coefficient-magnitude importance** for the
Step8B model in this repository.

---

## 2. The only candidate: Step8B RandomForest impurity importance

Production code, `src/step8b_train_baseline_vs_thermal_model.py:654-674`:

```python
# Final models fit on the WHOLE population for feature importance.
final_pipe_a = build_pipeline(BASELINE_FEATURES, model_name, random_state)
final_pipe_a.fit(df_pop[BASELINE_FEATURES], y)
final_pipe_b = build_pipeline(THERMAL_MODEL_FEATURES, model_name, random_state)
final_pipe_b.fit(df_pop[THERMAL_MODEL_FEATURES], y)

for label, pipe, feat_list in (("baseline", final_pipe_a, BASELINE_FEATURES),
                               ("thermal",  final_pipe_b, THERMAL_MODEL_FEATURES)):
    names = get_expanded_feature_names(pipe, feat_list)
    clf = pipe.named_steps["clf"]
    importances = getattr(clf, "feature_importances_", None)
    ...
```

Pipeline, `src/step8b_train_baseline_vs_thermal_model.py:449-467`:

```
numeric      : SimpleImputer(strategy="median")
categorical  : SimpleImputer(strategy="most_frequent") → OneHotEncoder(handle_unknown="ignore")
classifier   : RandomForestClassifier(n_estimators=300, max_depth=None,
                                      min_samples_leaf=3, class_weight="balanced",
                                      random_state=42, n_jobs=-1)
```

### Exact characterisation

| Property | Value |
|---|---|
| Artifact | `outputs/experiments/<exp>/step8b/step8b_feature_importance.csv` |
| Columns | `population, model, feature, importance` |
| Model family | tree ensemble |
| Model algorithm | `RandomForestClassifier`, `n_estimators=300`, `min_samples_leaf=3`, `class_weight="balanced"`, `random_state=42` — identical across all four AOIs (`model = "random_forest"` in every `step8b_model_comparison_metrics.json`) |
| Feature contract | `SHARED_THERMAL_MODEL_FEATURES` (`model = "thermal"`) or `SHARED_BASELINE_FEATURES` (`model = "baseline"`) |
| **Importance method** | **Mean decrease in impurity (Gini), `sklearn` `feature_importances_`** |
| Fold / repeat structure | **None.** A single final refit on the whole population; the 5-fold spatial CV is used for metrics only, never for importances. |
| In-sample or out-of-fold | **In-sample**, whole-population refit |
| Source labels used | **Yes** — `.fit(X, y)` on the source `burned` column |
| Target labels used | No |
| Normalised | Yes — sums to exactly 1.0 per `(population, model)` by sklearn construction |
| Negative values possible | **No.** Gini importance is non-negative by construction. Verified: `min(importance) = 0.0` in all four AOIs. |
| Categorical representation | **One-hot dummy level**, `cat__landcover_dominant_<code>`, one row per observed level |
| Numeric representation | `num__<feature_name>` |

### Availability across the four AOIs

`population = burnable_tree_shrub_grass`, `model = thermal` — present in all
four:

| Experiment | Rows | Sum | Min | Dummy levels | SHA-256 of the CSV |
|---|---:|---:|---:|---|---|
| `manavgat_2021` | 16 | 1.000000 | 0.0 | 10,20,30,40,50,60,80 | `24a18effd1b7828d48f2d1507dec982111364d637cee2bb884e4b7178aa72ec5` |
| `bejis_2022` | 16 | 1.000000 | 0.0 | 10,20,30,40,50,60,80 | `abc2a3b771d464dfac38597bbe6a0b425e6bec175bad860cc7a954fa005a3d49` |
| `mugla_2021` | 17 | 1.000000 | 0.0 | 10,20,30,40,50,60,80,90 | `9077eb95f928e9d0a575d658c9c03fe5cb8ffa485b04ebddafc08b102451e3ce` |
| `evia_2021_extended` | 17 | 1.000000 | 0.0 | 10,20,30,40,50,60,80,90 | `81995a6088f13b2d115a23be13917f561af264b5bdf1ef98b4fb940a15b6bf5a` |

### The observed importance values

`population = burnable_tree_shrub_grass`, `model = thermal`. Landcover dummies
are shown as their group sum (the value that becomes `w_landcover`).

| Feature | manavgat | bejis | mugla | evia_ext |
|---|---:|---:|---:|---:|
| `ndvi_mean` | 0.11219 | 0.09058 | 0.10402 | 0.11496 |
| `elevation_mean` | **0.22827** | **0.33356** | **0.17258** | 0.12798 |
| `slope_mean` | 0.13262 | 0.06892 | 0.10430 | 0.13290 |
| `lst_anomaly_mean` | 0.08564 | 0.07214 | 0.09573 | **0.14030** |
| `current_lst_mean` | 0.06101 | 0.04934 | 0.07894 | 0.06837 |
| `current_tvdi_mean` | 0.07301 | 0.06229 | 0.08708 | 0.13388 |
| `tvdi_difference_mean` | 0.11150 | 0.06787 | 0.11597 | 0.08143 |
| `downscaled_lst_mean` | 0.10540 | 0.09065 | 0.14597 | 0.11006 |
| `fused_lst_mean` | 0.06213 | 0.05887 | 0.08610 | 0.07870 |
| `landcover_dominant` (group sum) | 0.02824 | 0.10561 | 0.00932 | 0.01141 |

Facts that matter for the design:

- **No negative value anywhere.** The negative-importance policy is a fail-closed
  assertion in this configuration, not an active rule.
- **Exact zeros do occur, but only at dummy level**: `cat__landcover_dominant_50`
  in Manavgat, `cat__landcover_dominant_80` in Bejís, `cat__landcover_dominant_90`
  in Muğla and Evia. After group summing, **no** feature in the 10-feature
  contract has zero weight in any AOI. The zero-weight policy is therefore also
  a fail-closed assertion in this configuration.
- **`elevation_mean` is the top-weighted feature in three of four AOIs**, and by
  a wide margin in Bejís (0.334). This matters: the unweighted diagnostic already
  showed that every one of the three worst-supported directions is Bejís-source
  and driven entirely by `elevation_mean`. Importance weighting will therefore
  **amplify**, not offset, the Bejís elevation story. That is a genuine finding to
  expect, not a bug — but it should be stated in advance so it is not read as a
  post-hoc discovery.
- **Landcover's total weight is small** (0.009–0.106) and Bejís is the outlier at
  0.106, driven almost entirely by `cat__landcover_dominant_20` (shrubland,
  0.0659) — Bejís is the only AOI where shrubland is a major class (3743 cells).
  A categorical encoding that inflates landcover's contribution would distort
  Bejís-source rows specifically.

---

## 3. Accepted primary — third-best on the requested ladder, and adopted knowingly

> **Decision B-1 is ACCEPTED.** The Step8B RandomForest impurity importance is
> the weighting source for this completion run. No new permutation importance is
> produced. B-1 is no longer an implementation blocker.


The task's preference ladder:

| Rank | Requirement | Available? |
|---|---|---|
| 1 | Source-only, out-of-fold or held-out **permutation** importance, all four AOIs | **No** |
| 2 | Same-method verified source-only importance, all four AOIs | Satisfied only by the impurity artifact |
| 3 | Impurity/gain only → use it with an explicit limitation | **This is where the repository sits** |
| 4 | No comparable artifact → new fit needed | Not the case; a comparable artifact exists |

**Accepted primary:**

```
importance_artifact  outputs/experiments/<source>/step8b/step8b_feature_importance.csv
population           burnable_tree_shrub_grass
model                thermal
importance_method    impurity_gini_in_sample_whole_population_v1
decision_status      accepted
implementation_blocker  false
```

**Mandatory limitation text, to be carried in the artifact metadata and the
report verbatim:**

> Feature weights come from RandomForest mean-decrease-in-impurity (Gini)
> importance, computed on a final model refit on the whole source population.
> Impurity importance is computed in-sample rather than on held-out data, and is
> known to be biased toward continuous and high-cardinality predictors relative
> to low-cardinality categorical ones. It is used here as a source-model
> relevance weighting, not as an unbiased estimate of a predictor's causal or
> generalisation importance. A held-out permutation importance would be the
> preferred weighting, does not exist in this repository, and remains available
> as a later sensitivity analysis.

The bias direction is worth stating precisely because it interacts with the
result: nine continuous numeric predictors versus one 7-to-8-level categorical
predictor is exactly the configuration in which Gini importance under-weights the
categorical. The observed landcover group weights (0.009–0.106) may therefore be
understated. This is an argument for keeping the unweighted categorical
unseen-level diagnostic as a **separate sidecar** rather than folding it into a
single number — see `03_weighted_dissimilarity_design.md` §5.

**Deferred sensitivity, not part of this run.** A held-out permutation importance
would require `sklearn.inspection.permutation_importance` on the existing 5-fold
spatial-CV holdouts, per AOI, on the source label. That fits models, so it is out
of scope for the completion run and must be preregistered separately if it is
ever wanted. Recorded as `later_sensitivity`, not as a blocker.

---

## 4. Rejected candidates, and why

### `step8d_ablation_feature_importance.csv`

Present for all four AOIs, columns `population, model_name, feature, importance`.
`model_name` here is an **ablation variant**, not a model family:

```
all_thermal, baseline, current_lst_only, downscaled_only,
fused_downscaled_group, fused_lst_only, lst_anomaly_group, lst_anomaly_only,
tvdi_difference_only, tvdi_group, tvdi_only
```

Row counts differ per AOI (351 / 384 / 479 / 468) because the population set
differs. The values are the *same* RandomForest impurity importance, just
recomputed per ablated feature subset. **Rejected**: no variant uses the full
`SHARED_THERMAL_MODEL_FEATURES` contract under a name that maps cleanly onto the
Step9/Step10 model family, and using an ablated subset would silently change the
feature space the AoA is defined over. `all_thermal` is the closest match but is
a Step8D-internal identity, not the Step8B `thermal` family that Step9/Step10
transfer actually uses.

If the advisor later wants a *robustness* check on the weights, Step8D's
`all_thermal` rows are the natural sensitivity comparison. That is a follow-up,
not the primary.

### `src/step7c_train_downscaling_model.py` permutation importance

**Rejected.** Step7C is the 30 m LST downscaling regressor. Different target,
different feature space, different spatial support. It has nothing to do with the
500 m burned-area association model whose applicability domain is being measured.

### `step8b_model_comparison_metrics.json` fold metrics

Contains per-fold AUC/PR-AUC, not importances. Useful for provenance
(`model = "random_forest"`, `spatial_cv_config.random_state = 42`) but carries no
weight information.

### Domain-classifier audit

`outputs/diagnostics/domain_classifier_audit/` fits a source-vs-target classifier.
Its coefficients/importances would describe *what separates the AOIs*, which is
a different quantity from *what the burned-area model relies on*. Using it as the
AoA weight would make the diagnostic partly circular with respect to the
separation it is meant to measure. **Rejected**, and Evia is absent from it
anyway.

---

## 5. Label-firewall bookkeeping

Required fields in the completion artifact, exactly:

```json
{
  "target_label_firewall": {
    "target_label_used": false,
    "target_burn_date_used": false,
    "target_transfer_metric_used": false,
    "target_columns_loaded": [],
    "rule": "Every parquet read passes an explicit columns= allow-list containing only predictor, grid, population-mask and eligibility columns."
  },
  "source_label_policy": {
    "source_label_used": true,
    "mechanism": "Feature weights are RandomForest impurity importances from a Step8B model fitted on the source `burned` label.",
    "source_label_read_directly_by_completion_module": false,
    "diagnostic_class": "target-label-blind, source-model-informed"
  }
}
```

`source_label_read_directly_by_completion_module` is `false` and must stay `false`: the
completion module reads the **importance CSV**, never a label column. The source
label enters only through a frozen upstream artifact.

**The phrase "label-blind" must not appear unqualified anywhere in the new
artifact, report or advisor text.** The permitted description is
"target-label-blind, source-model-informed diagnostic".

---

## 6. Weight derivation contract (decided here, formalised in doc 03)

```
1. Read step8b_feature_importance.csv for each source AOI.
2. Filter to population == "burnable_tree_shrub_grass" AND model == "thermal".
3. Assert the row set is exactly {num__<f> for the 9 numeric features}
   ∪ {cat__landcover_dominant_<level> for the observed source levels}.
   Any extra or missing row → fail closed.
4. Assert every importance is finite and >= 0. Any negative → fail closed.
5. raw_w[f]            = importance of num__<f>            for the 9 numeric f
   raw_w[landcover]    = Σ importance of cat__landcover_dominant_*
6. Assert Σ raw_w ≈ 1.0 within 1e-9 (sklearn guarantees this; verify, do not assume).
7. w[f] = raw_w[f] / Σ raw_w        (renormalise defensively; a no-op here)
8. Record: raw values, group sum, renormalisation factor, per-level landcover
   contributions, effective_feature_count, feature_weight_entropy.
```

Step 5 is the **group-normalisation** decision: the categorical predictor gets
exactly the weight the source model gave it in total, independent of how many
levels happened to be observed. This is what makes the weight comparable between
a 7-level source (Manavgat, Bejís) and an 8-level source (Muğla, Evia).
