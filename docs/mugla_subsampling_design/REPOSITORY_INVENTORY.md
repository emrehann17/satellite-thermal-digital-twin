# Repository Inventory — exact files and symbols

Everything below was read in this repository on **2026-08-03**. Nothing was
executed that wrote to a production path. Line numbers are as of commit
`19d825b` plus the uncommitted working tree.

Legend: **REUSE** = imported and called unchanged · **PATTERN** = followed as a
template, new code · **REFERENCE** = read as a frozen input or cross-check ·
**NOT USED** = deliberately excluded, with reason.

---

## 1. Canonical Step8 model constructors, features, preprocessing

`src/step8b_train_baseline_vs_thermal_model.py`

| Symbol | Line | Use | Supplies |
|---|---:|---|---|
| `TARGET_COLUMN` | 116 | **REUSE** | `"burned"` |
| `BASELINE_FEATURES` | 121 | **REUSE** | `ndvi_mean, elevation_mean, slope_mean, landcover_dominant` |
| `THERMAL_FEATURES` | 127 | **REUSE** | 6 thermal columns |
| `THERMAL_MODEL_FEATURES` | 135 | **REUSE** | `BASELINE_FEATURES + THERMAL_FEATURES` (10) |
| `CATEGORICAL_FEATURES` | 136 | **REUSE** | `["landcover_dominant"]`, the only one-hot column |
| `FORBIDDEN_FEATURE_COLUMNS` | 140 | **REUSE** | leakage denylist (incl. `burned`, `burn_date`, `cell_id`, `lon`, `lat`) |
| `check_no_forbidden_features` | 265 | **REUSE** | hard failure if a label/metadata column reaches the feature set |
| `add_spatial_block_id(df, n, column_name=, id_prefix=, include_row_col=)` | 277 | **REFERENCE** | the exact call the frozen 10-cell artifact used (`column_name="big_block_id"`, `id_prefix="block10"`); needed only for the fold-mapping reproduction fallback |
| `make_spatial_folds(y, groups, n_splits_requested, random_state, strict=)` | 309 | **REFERENCE** | `StratifiedGroupKFold(5, shuffle=True, random_state=42)`; only invoked in the fallback path, since the mapping is loaded from the persisted artifact |
| `build_classifier(model_name, random_state)` | 420 | **REUSE** (indirect) | `RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)` |
| `build_pipeline(feature_list, model_name, random_state)` | 449 | **REUSE** | the whole `ColumnTransformer` + RF pipeline; the **only** place a model is constructed |
| `compute_binary_metrics(y_true, y_prob)` | 480 | **REUSE** | `roc_auc`, `pr_auc`, `brier_score` (threshold-dependent fields ignored); returns all-`None` when a class is absent |
| `train_population(...)` | 519 | **NOT USED** | its signature fits a whole region on freshly-built folds and also computes monthly lead-time and feature importance; this analysis needs a per-repeat frame with an *inherited* fold vector. Its fit → `predict_proba` → accumulate-OOF sequence is followed as **PATTERN**. |
| `filter_valid_for_modeling(df)` | 1097 | **REFERENCE** | fallback path only |
| `build_population_masks(df)` | 1107 | **REFERENCE** | fallback path only |
| `interpret_delta`, `gapfill_sensitivity`, `plot_*`, `write_*` | — | **NOT USED** | Step8B reporting surface, outside this schema |

`core/config.py`

| Constant | Line | Value | Use |
|---|---:|---|---|
| `STEP8B_RANDOM_SEED` | 554 | `42` | **REUSE** — estimator seed and fold seed |
| `STEP8B_N_SPLITS` | 555 | `5` | **REUSE** — fold count |
| `STEP8B_SPATIAL_BLOCK_SIZE_CELLS` | 556 | `2` | **REFERENCE** — recorded in the manifest as the canonical small-block scale this analysis departs from |
| `STEP8B_MIN_POSITIVES_PER_POPULATION` | 557 | `30` | **REUSE** — precondition; every arm passes with wide margin (smallest positive count anywhere is 280, in a subsample fold) |
| `STEP8A_OUTPUT_DIR` | 524 | `outputs/step8a` | **REFERENCE** |

## 2. Spatial block / ≈5 km utilities

`src/step8_large_block_robustness.py`

| Symbol | Line | Use | Supplies |
|---|---:|---|---|
| `assign_large_blocks(df, block_size_cells)` | 290 | **REUSE**, `block_size_cells=10` | `large_block_id = "b10_r{row//10}_c{col//10}"`, fixed `(0,0)` origin, assigned **before** population filtering. Internally calls `validate_canonical_grid`. |
| `validate_canonical_grid(df)` | ~270 | **REUSE** (indirect) | asserts `cell_id == f"r{row_500m}_c{col_500m}"` and non-negative indices. Verified to pass on the Muğla frame. |
| `EXPECTED_BLOCK_SIZES` | 38 | `(10, 20)` | **REFERENCE** — 10 is registered, so `assign_large_blocks(df, 10)` passes its guard |
| `NOMINAL_SCALES` | 39 | `{10: "approximately_5_km", 20: "approximately_10_km"}` | **REUSE** — the nominal-scale string written into the manifest |
| `PRIMARY_POPULATION` | 40 | `"burnable_tree_shrub_grass"` | **REFERENCE** |
| `sha256_file(path)` | 68 | **REUSE** | input hashing |
| `make_strict_spatial_folds(...)` | 305 | **NOT USED** | this module's runner is restricted to `EXPECTED_EXPERIMENTS = ("manavgat_2021", "bejis_2022")` (line 37) and never produced a Muğla artifact; the Muğla 10-cell artifacts came from the v2 runner instead |

`src/step8_big_block_robustness.py` — **the module that actually produced
Muğla's frozen 10-cell fold mapping.**

| Symbol | Line | Use |
|---|---:|---|
| `ANALYSIS_SCHEMA_VERSION` | 102 | `"step8.big_block_robustness.v2"` — **REFERENCE**, recorded as the provenance of the inherited fold mapping |
| `BLOCK_COLUMN` | 107 | `"big_block_id"` — **REFERENCE**; the persisted artifact spells the same partition as `spatial_block_id` with `block10_{r}_{c}` values |
| `DEFAULT_BLOCK_SIZES` | 106 | `(10, 20)` — **REFERENCE** |
| `run_big_block_condition(...)` | 542 | **PATTERN / REFERENCE** | the exact `add_spatial_block_id(…, 10, column_name="big_block_id", id_prefix="block10", include_row_col=True)` → `filter_valid_for_modeling` → `build_population_masks` → `train_population(…, group_column="big_block_id", strict_folds=True)` sequence. This is the fold-mapping reproduction fallback of `SCIENTIFIC_CONTRACT.md` §4. |

> **Routing note.** `step8_big_blocks_v2` must continue to reach the generic
> orchestrator adapter, never the legacy one. This analysis only *reads* that
> module's artifacts and registers no new orchestrator routing for it.

## 3. Raw transfer implementation

`src/step9b_run_cross_region_transfer.py`

| Symbol | Line | Use | Supplies |
|---|---:|---|---|
| `MODEL_NAME` | 85 | **REUSE** | `"random_forest"` |
| `load_step8a_dataset(experiment_id)` | 95 | **PATTERN** | same load, but this analysis substitutes `assign_large_blocks(df, 10)` for the 2-cell `add_spatial_block_id` |
| `population_subset(df, population)` | 107 | **REUSE** | `valid_for_modeling == True` ∧ named boolean population column — the single definition of "primary population" used everywhere here |
| `run_one_direction_population(...)` | 177 | **PATTERN** | the canonical source-only transfer: `build_pipeline(feature_list, MODEL_NAME, 42)` → `fit(X_source, y_source)` → `predict_proba(X_target)[:, 1]`. Lines 227–236 are the exact construction Arm B replicates. |
| `select_threshold_from_source_oof(...)` | 119 | **NOT USED** | this analysis reports only threshold-free metrics; reusing it would add ~5 fits per direction per repeat for no reported quantity |
| `compute_metrics_at_threshold(...)` | 152 | **NOT USED** | threshold-dependent; superseded by `compute_binary_metrics` |
| `_dataset_provenance(experiment_id)` | 262 | **PATTERN** | shape of the per-experiment provenance block |

`src/step9a_audit_cross_region_inputs.py`

| Symbol | Line | Use |
|---|---:|---|
| `SHARED_BASELINE_FEATURES` | 64 | **REUSE** — the cross-region feature contract, single source of truth |
| `SHARED_THERMAL_FEATURES` | 70 | **REUSE** |
| `SHARED_THERMAL_MODEL_FEATURES` | 78 | **REUSE** |
| `PRIMARY_POPULATIONS` | 128 | **REUSE** — `["burnable_tree_shrub_grass"]` |
| `resolve_step8a_dataset_path(experiment_id)` | 140 | **REUSE** |
| `resolve_step8a_stats_path(experiment_id)` | 171 | **REUSE** |
| `sha256_file(path)` | 196 | **REUSE** |
| `resolve_git_commit()` | 204 | **REUSE** |
| `resolve_feature_contract(experiment_id)` | 231 | **REUSE** — fails closed if an experiment's feature contract drifted |
| `cross_region_output_root(source_id, target_id)` | 136 | **REFERENCE** — used *read-only* to resolve the frozen transfer artifacts; this analysis never writes there |

`core/cross_region_experiment.py`

| Symbol | Line | Use |
|---|---:|---|
| `SHARED_*` feature re-exports | 64–78 (via step9a) | **REUSE** |
| `assert_paths_are_safely_namespaced` | 132 | **PATTERN** — namespace containment |
| `compute_region_robust_stats` / `apply_region_robust_transform` | 157 / 183 | **NOT USED** — that is the Step10 self-calibration path; this analysis is *raw* transfer only |
| `paired_spatial_block_bootstrap` | 263 | **NOT USED** — no bootstrap is run anywhere in this analysis |
| `bootstrap_support_category` | 313 | **NOT USED** — an interval-derived support status is a significance claim in all but name |

## 4. Frozen inputs — the three reference sets

### 4.1 Canonical Step8A datasets (hash gate)

| Experiment | Path | sha256 | Primary rows | Pos | Neg | Prevalence |
|---|---|---|---:|---:|---:|---:|
| manavgat_2021 | `outputs/experiments/manavgat_2021/step8a/step8a_500m_modeling_dataset.parquet` | `054a1961…787f3439` ✅ | 20,511 | 784 | 19,727 | 0.03822339 |
| bejis_2022 | `outputs/experiments/bejis_2022/step8a/step8a_500m_modeling_dataset.parquet` | `3dec785a…c24d9e393` ✅ | 15,190 | 1,100 | 14,090 | 0.07241606 |
| mugla_2021 | `outputs/experiments/mugla_2021/step8a/step8a_500m_modeling_dataset.parquet` | `c4ab107d…0be7db8e` ✅ | 41,730 | 2,911 | 38,819 | 0.06975797 |

All three digests were recomputed and **match the contract exactly**.

### 4.2 Arm A reference — full-Muğla 10-cell within-region OOF

`outputs/experiments/mugla_2021/robustness/step8_big_blocks/block_10_cells/`

| File | sha256 | Bytes | Role |
|---|---|---:|---|
| `step8b_metrics.json` | `a826279f7268c4fe55a6cb4762b17589ab536c86973c730ee5f9647ec50f34d1` | 2,317 | **REFERENCE** — the Arm A reference metric values |
| `oof_predictions.parquet` | `e16e6b18020b745da6e91a9e59664778b7a2887c8227729048fc459bc9df8cd4` | 1,112,509 | **REUSE** — supplies the inherited `cell_id → fold_id` mapping (41,730 rows), and independently reproduces the reference metrics |
| `fold_assignments.parquet` | `e5b2992857b38b1f58ef391be2f894626302b0048de5f15ad63d7f68169957f8` | 15,906 | **REFERENCE** — per-fold row/positive/block counts and `block_overlap == 0` |
| `block_manifest.json` | `13dd42011754586a88848f520e3e9750b12cc44179a090cf5b866bd0860aab58` | 2,687 | **REFERENCE** — binds the whole condition to Muğla Step8A `c4ab107d…`, records 576 blocks, 70 positive-containing blocks, `train_test_block_leakage_free: true` |

`oof_predictions.parquet` columns: `experiment_id, block_size_cells,
population, cell_id, row_500m, col_500m, spatial_block_id, fold_id, burned,
baseline_probability, thermal_probability, valid_for_evaluation,
landcover_dominant, burnable_tree_shrub_grass`.

Verified read-only:
- 41,730 rows, 0 duplicate `cell_id`, `valid_for_evaluation` all `True`;
- fold sizes 8,374 / 8,325 / 8,360 / 8,331 / 8,340;
- **0 of 576 blocks span more than one fold**;
- `large_block_id ↔ spatial_block_id` is a bijection (576 ↔ 576) and the
  `cell_id` sets are identical to the Muğla primary population (outer merge:
  41,730 both, 0 left-only, 0 right-only);
- recomputing metrics from the probability columns reproduces
  `step8b_metrics.json` exactly (baseline ROC 0.6979859420145867, thermal
  0.7773268638729566, etc.).

**`outputs/experiments/mugla_2021/step8b/` (2-cell Step8B) — NOT USED as a
reference.** It is the small-block (2-cell) within-region result at
`population_counts.burnable_tree_shrub_grass = 41730`; its fold contract is a
different block scale from this analysis's strata, and `make_spatial_folds` is
called there with `strict=False`. It is recorded in `config.json` as context
only.

### 4.3 Arms B and C references — canonical raw transfer

Resolution rule: for direction `S → T`, read from
`outputs/cross_region/{S}__{T}/step9b/`.

| Pair directory | `cross_region_transfer_metrics.json` sha256 | `cross_region_transfer_predictions.parquet` sha256 |
|---|---|---|
| `mugla_2021__manavgat_2021` | `1d8d7d532e2ed63e34c452473cdd907d7491b0858c98ba863b6a0a2d8000f199` | `f28cef6c7d60e4c26d5dc5e9071895b2f5e39ef31afadddbc38bbed2dee35c80` |
| `mugla_2021__bejis_2022` | `676c209e8aa95d637ac85a438fa4f13663b37d81f160a3114cefba5c9557f0ee` | `a82f6934babcd658ef601b19a748010ee5e35b69377a28149bd5ba534b2672c0` |
| `manavgat_2021__mugla_2021` | `3bd2a6ecbd8b815969295ce6fcaa87b840b7dc4e6901e7dba50d49e142927fd7` | `0aace925010a14cd50f20ca74913d2a97cab0624853ba8079ff5d563f5b192d8` |
| `bejis_2022__mugla_2021` | `a635143d4e80f3ca863a243f174a958d27cd87837a889c6d01a5981885a50f6c` | `183ce97582414304e6912f1e2e6cbb8036e4562436b5f44fb2b3bcfe755b2624` |

Each `cross_region_transfer_metrics.json` carries `resolved_inputs` pinning the
same canonical Step8A sha256 digests as §4.1, `spatial_block_size_cells: 2`,
`source_only: true`. The references are therefore provenance-bound to the same
frozen inputs this analysis gates on.

## 5. **Can Arm C avoid model refits? — YES, verified**

This was the decisive inventory question. Answer: the persisted raw-transfer
predictions are sufficient; **Arm C requires zero fits.**

`cross_region_transfer_predictions.parquet` columns (written at
`src/step9b_run_cross_region_transfer.py:218`):

```
transfer_direction, population, target_experiment_id, target_cell_id,
target_spatial_block_id, burned, baseline_probability, thermal_probability
```

Verified read-only for both Muğla-as-target directions:

| Check | manavgat → mugla | bejis → mugla |
|---|---|---|
| Rows at `population == "burnable_tree_shrub_grass"` | 41,730 | 41,730 |
| Duplicate `target_cell_id` | 0 | 0 |
| `target_cell_id` set == Muğla primary `cell_id` set | **True** | **True** |
| Positives | 2,911 | 2,911 |
| Metrics recomputed from columns == stored metrics | **exact match** | **exact match** |

The same exact-match check passes for both Muğla-as-source directions
(`mugla → manavgat`, 20,511 rows; `mugla → bejís`, 15,190 rows), confirming
that `compute_binary_metrics` on these columns is the same arithmetic that
produced the stored references.

Because the per-cell probability of a *fixed* source model does not depend on
which other target cells are evaluated alongside it, restricting to a repeat's
20,511 `cell_id`s and recomputing ROC-AUC / PR-AUC / Brier is exact — not an
approximation. Arm C's 4 nominally-required fits are eliminated.

## 6. Metric helpers

| Metric | Function | Origin |
|---|---|---|
| ROC-AUC | `sklearn.metrics.roc_auc_score` | via `compute_binary_metrics` (step8b:480) |
| PR-AUC | `sklearn.metrics.average_precision_score` | via `compute_binary_metrics` (step8b:480) |
| Brier | `sklearn.metrics.brier_score_loss` | via `compute_binary_metrics` (step8b:480) |

`core/step10_shared.compute_threshold_free_metrics` (line 227) is an equivalent
helper — **NOT USED**, because `compute_binary_metrics` is what every reference
value in §4 was computed with.

## 7. Output, provenance, atomic write, resume, quarantine patterns

`src/few_shot_recovery.py` is the closest structural precedent (staged,
hash-gated, repeat-based, interval-summarised, forbidden-token-guarded) and is
the primary **PATTERN** source.

| Symbol | Line | Pattern supplied |
|---|---:|---|
| `SCHEMA_VERSION` / `DIAGNOSTIC_NAMESPACE` / `DIAGNOSTIC_CLASS` | 81–83 | the three-token identity header |
| `STAGES`, `STAGE_REQUIRES` | 202–207 | `("plan", "fit", "summarize")` with prerequisite map |
| `STAGE_OUTPUTS` | 212 | per-stage declared file list |
| `planned_output_layout()` | 397 | `--dry-run` declaration of every writable file |
| `validate_stage_range(from_stage, to_stage)` | 236 | stage-order validation |
| `assert_not_excluded(experiment_id)` | 272 | the Evia exclusion guard, adapted |
| `direction_token(source_id, target_id)` | 305 | never-sorted `f"{source}_to_{target}"` |
| `selection_seed(...)` | 316 | **PATTERN** — `blake2b(key, digest_size=8) % 2**32`; adapted to `repeat_id` + `stratum_id` |
| `fit_identity(...)` | 328 | **PATTERN** — the identity that decides whether two fits are the same fit |
| `expected_unique_fit_count(...)` | 351 | **PATTERN** — declared fit budget cross-checked against the registry |
| `diagnostics_root` / `analysis_root` / `stage_marker_path` | 369 / 374 / 378 | `outputs/diagnostics/<namespace>/<analysis_id>/` |
| `canonical_json` / `compute_analysis_id` | 437 / 443 | deterministic analysis ID from the frozen scientific config |
| `assert_inside_namespace(path, root)` | 469 | **REUSE-as-pattern** — namespace containment on every write |
| `_atomic_write_text` / `_atomic_write_parquet` | 478 / 489 | temp-file + `os.replace` atomic writes |
| `sha256_file` / `sha256_path` | 500 / 508 | file and logical-dataset hashing |
| `_git_commit` / `_package_versions` | 522 / 536 | provenance |
| `build_frozen_input_inventory` | 548 | `input_hashes.json` shape |
| `assert_canonical_step8a_hashes(..., strict=True)` | 575 | the hard hash gate before any computation |
| `FitRegistry` | 740 | **PATTERN** — memoises fit *results* by identity; `fit_count` / `reuse_count` / `identities()` / `accounting()` / `release(prefix)` |
| `fit_and_predict(...)` | 809 | **PATTERN** — the ONLY place a model is fitted: `check_no_forbidden_features` → `build_pipeline` → `fit` → `predict_proba[:, 1]` |
| `metric_orientation` / `oriented` | 828 / 832 | Brier-by-negation orientation |
| `selection_interval(values)` | 839 | **PATTERN** — median + 2.5/97.5 + min/max, `method="linear"`; renamed here to `subsampling_interval_*` |
| `assert_full_oof_coverage(coverage, context)` | 901 | exactly-one-OOF-prediction assertion |
| `write_stage_marker` / `read_stage_marker` / `verify_stage_complete` | 920 / 948 / 959 | resume support with per-file hashes |
| `QUARANTINE_DIRNAME` | 210 | `_quarantine` for partial/invalid partitions |
| forbidden-token denylist | ~193–200 | the literal scan list (`confidence interval`, `p_value`, `anlamlı`, …) |
| `verify_direction_partition` | 990 | partition completeness check, adapted to repeat partitions |
| `N_OUTER_FOLDS`, `FOLD_RANDOM_STATE`, `ESTIMATOR_SEED`, `MODEL_FAMILIES` | 108–114 | constant naming convention |
| its 10-cell **outer-fold construction** | 680 (`build_outer_folds`) | **NOT USED** — the contract explicitly forbids reusing that fold structure; this analysis inherits the Step8 mapping instead |
| `recovery_quantities`, budget/tier logic | 700–900 | **NOT USED** — no budgets, no tiers, no recovery fraction here |

`src/marginal_aoa_completion.py` — secondary **PATTERN** source

| Symbol | Line | Pattern |
|---|---:|---|
| `SCHEMA_VERSION` | 110 | `"<name>.v1"` convention |
| `DIAGNOSTIC_NAMESPACE` / `DIAGNOSTIC_CLASS` | 111–112 | namespace + class declaration |
| `stage_side_effect_flags(stages)` | 391 | which stages may write |
| `diagnostics_root` / `analysis_root` | 405 / 410 | namespace roots |
| `canonical_step8a_path(experiment_id, experiments_root=None)` | 414 | injectable canonical path resolution (lets tests redirect) |
| `build_frozen_input_inventory` | 517 | `input_hashes.json` shape |
| `assert_canonical_step8a_hashes` | 534 | hash gate |
| `planned_output_layout()` | 444 | dry-run layout |
| `assign_spatial_folds(...)` | 1038 | **NOT USED** — its `sorted_block_round_robin` assignment is not the Step8 contract |

`core/step10_shared.py`

| Symbol | Line | Use |
|---|---:|---|
| `check_no_forbidden_features` | 66 | **REUSE** — second leakage guard |
| `canonical_json(obj)` | 114 | **REUSE** — stable JSON for hashing |
| `compute_analysis_id(scientific_config)` | 118 | **REUSE** — deterministic analysis ID |
| `git_commit_if_available()` | 122 | **REUSE** |
| `package_versions()` | 135 | **REUSE** — numpy/pandas/sklearn versions into the manifest |
| `compute_threshold_free_metrics` | 227 | **NOT USED** — see §6 |
| `run_n_way_paired_bootstrap` | 267 | **NOT USED** — no bootstrap |

## 8. CLI / orchestrator patterns

| File | Line | Pattern |
|---|---:|---|
| `scripts/run_few_shot_recovery.py` | — | thin dispatcher: `build_parser()` → `main(...)` → `run_analysis(...)`, no scientific logic in the script. Template for `scripts/run_mugla_subsampling.py`. |
| `scripts/validate_few_shot_recovery.py` | 39 (`class Report`), 634 (`run_validation`), 683 (`main`) | `{check_id, status, expected, observed, evidence_path, note}` rows; `PASS`/`FAIL`/`SKIP`; dry-run vs actual; any FAIL ⇒ overall FAIL; exit code 1 on FAIL. Template for `scripts/validate_mugla_subsampling.py`. |
| `core/pipeline_orchestrator.py::run_few_shot_recovery_stage` | 819 | stage-dispatch signature: `from_stage` / `to_stage` / `dry_run` / `resume` / `output_root` / `experiments_root`, docstring naming the diagnostic class and namespace. Template for `run_mugla_subsampling_stage`. |
| `core/pipeline_orchestrator.py::run_marginal_aoa_completion_stage` | 784 | same shape, second exemplar |
| `core/pipeline_orchestrator.py::_assert_context_is_safely_namespaced` | 93 | namespace containment guard |
| `scripts/main.py` | 507 (`STAGES` import), 514 (`cmd_few_shot_recovery`), 1387–1453 (subparser) | subcommand registration surface; new subcommand `mugla-subsampling` follows it exactly |

## 9. Test fixture patterns

`tests/test_few_shot_recovery.py` and `tests/test_marginal_aoa_completion.py`
are the models. Reused patterns:

- module-level synthetic fixture builders; unit tests never depend on
  production outputs;
- `tmp_path`-injected `output_root` and `experiments_root`, so no test can
  write to a canonical path;
- adversarial `test_*_fails_closed` for every guard;
- invariance tests (`test_di_is_invariant_to_source_row_order`) — adapted here
  directly to the row-order invariance requirement of §3.6;
- exact-count tests (`test_exactly_six_directed_pairs`) → `test_every_repeat_has_exactly_20511_unique_rows`;
- forbidden-token scan tests over emitted text.

`tests/test_step8_big_block_robustness.py` supplies the block-assignment and
strict-fold assertion patterns and the `block10_{r}_{c}` id expectations.
`tests/test_main_cli.py` and `tests/test_pipeline_orchestrator.py` supply the
subcommand and stage-dispatch registration tests.

---

## 10. Reuse summary

**Imported and called unchanged (16 symbols):** `build_pipeline`,
`build_classifier` (indirect), `check_no_forbidden_features` (both copies),
`compute_binary_metrics`, `assign_large_blocks`, `validate_canonical_grid`
(indirect), `population_subset`, `resolve_step8a_dataset_path`,
`resolve_step8a_stats_path`, `resolve_feature_contract`, `sha256_file`,
`resolve_git_commit`, `canonical_json`, `compute_analysis_id`,
`git_commit_if_available`, `package_versions`.

**Constants reused (12):** `TARGET_COLUMN`, `BASELINE_FEATURES`,
`THERMAL_FEATURES`, `THERMAL_MODEL_FEATURES`, `CATEGORICAL_FEATURES`,
`FORBIDDEN_FEATURE_COLUMNS`, `SHARED_*` feature lists, `PRIMARY_POPULATIONS`,
`MODEL_NAME`, `STEP8B_RANDOM_SEED`, `STEP8B_N_SPLITS`,
`STEP8B_MIN_POSITIVES_PER_POPULATION`, `NOMINAL_SCALES`.

**Frozen artifacts read (10 files):** 3 Step8A parquets, 4 block_10_cells
files, 4 pair directories' metrics + predictions (8 files, of which the 4
prediction parquets are also the Arm C data source).

**New logic required (only this):** the integer-exact Hamilton allocation with
its `(-remainder, stratum_id)` tie-break, the per-stratum deterministic
selection, the inherited-fold join, the oriented-delta / subsampling-interval /
position-token arithmetic, and the schema + validator surface. **No new model,
no new preprocessing, no new fold algorithm, no new metric, no bootstrap.**
