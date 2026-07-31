# Repository Inventory — what is reused, with exact symbols

Everything below was read in this repository on 2026-08-02. Nothing was
executed that wrote to a production path.

Legend: **REUSE** = imported and called unchanged · **PATTERN** = followed as a
template, new code · **REFERENCE** = read as frozen input or cross-check ·
**NOT USED** = deliberately excluded, with reason.

---

## 1. Model constructors, features, preprocessing

`src/step8b_train_baseline_vs_thermal_model.py`

| Symbol | Line | Use | Supplies |
|---|---:|---|---|
| `build_pipeline(feature_list, model_name, random_state)` | 449 | **REUSE** | The whole `ColumnTransformer` + `RandomForestClassifier` pipeline. Also calls `check_no_forbidden_features` internally. |
| `build_classifier(model_name, random_state)` | 420 | **REUSE** (indirect) | `RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)`. **Source of the pre-existing class weighting** documented in `SCIENTIFIC_CONTRACT.md` §4.3. |
| `check_no_forbidden_features(feature_list)` | 265 | **REUSE** | Hard failure if a label/metadata column reaches the feature set. |
| `FORBIDDEN_FEATURE_COLUMNS` | 140 | **REUSE** | 19-column leakage denylist incl. `burned`, `burn_date`, `label_source`, `lon`, `lat`, `cell_id`. |
| `BASELINE_FEATURES` | 121 | **REUSE** | `ndvi_mean`, `elevation_mean`, `slope_mean`, `landcover_dominant`. |
| `THERMAL_FEATURES` | 127 | **REUSE** | `lst_anomaly_mean`, `current_lst_mean`, `current_tvdi_mean`, `tvdi_difference_mean`, `downscaled_lst_mean`, `fused_lst_mean`. |
| `THERMAL_MODEL_FEATURES` | 135 | **REUSE** | `BASELINE_FEATURES + THERMAL_FEATURES` (10). |
| `CATEGORICAL_FEATURES` | 136 | **REUSE** | `["landcover_dominant"]` — the only one-hot column. |
| `TARGET_COLUMN` | 116 | **REUSE** | `"burned"`. |
| `add_spatial_block_id(df, block_size_cells, …)` | 277 | **NOT USED** | Would give the 2-cell column; this analysis uses the 10-cell utility instead. Its 10-cell equivalent (`assign_large_blocks`) is used. |
| `make_spatial_folds(y, groups, n_splits_requested, random_state, strict=)` | 309 | **REUSE**, `strict=True` | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` with the four strict guarantees: no fold reduction, no block on both sides, both classes both sides, exact-once OOF coverage. |
| `compute_binary_metrics(y_true, y_prob)` | 480 | **REUSE** | `roc_auc`, `pr_auc`, `brier_score` (+ threshold-dependent fields this analysis ignores). Returns all-`None` when a class is absent. |
| `train_population(...)` | 519 | **NOT USED** | Fits both families on one region's own folds and computes monthly lead-time and feature importance. The few-shot loop needs a source∪target training frame and a per-repeat selection, which this signature cannot express. Its *fit/predict/metric* sequence is followed as **PATTERN** (lines 570–597) so behaviour matches. |
| `interpret_delta`, `gapfill_sensitivity`, `plot_*`, `write_*` | 506–1097 | **NOT USED** | Step8B reporting surface; not part of this schema. |

`core/config.py`

| Constant | Line | Value | Use |
|---|---:|---|---|
| `STEP8B_RANDOM_SEED` | 554 | `42` | **REUSE** — estimator seed for every fit. |
| `STEP8B_N_SPLITS` | 555 | `5` | **REUSE** — outer fold count. |
| `STEP8B_SPATIAL_BLOCK_SIZE_CELLS` | 556 | `2` | **REFERENCE** — recorded in the manifest as the canonical small-block scale this analysis deliberately departs from. |
| `STEP8B_MIN_POSITIVES_PER_POPULATION` | 557 | `30` | **REUSE** — precondition on source and target population counts (all six directions pass with margin). |
| `STEP8A_OUTPUT_DIR` | 524 | `outputs/step8a` | **REFERENCE**. |

## 2. Spatial block / ~5 km utilities

`src/step8_large_block_robustness.py`

| Symbol | Line | Use | Supplies |
|---|---:|---|---|
| `assign_large_blocks(df, block_size_cells)` | 290 | **REUSE**, `block_size_cells=10` | `large_block_id = "b10_r{floor(row_500m/10)}_c{floor(col_500m/10)}"`, fixed origin `(0,0)`, assigned before population filtering. |
| `EXPECTED_BLOCK_SIZES` | 38 | `(10, 20)` | **REFERENCE** — 10 is a registered block size, so `assign_large_blocks(df, 10)` passes its internal guard. |
| `NOMINAL_SCALES` | 39 | `{10: "approximately_5_km", 20: "approximately_10_km"}` | **REUSE** — the nominal-scale string written into the manifest. |
| `PRIMARY_POPULATION` | 40 | `"burnable_tree_shrub_grass"` | **REFERENCE** — identical to this analysis's population, which is why its frozen 10-cell metrics are a valid ceiling anchor. |
| `sha256_file(path)` | 68 | **REUSE** | Input hashing. |
| `experiment_step8_root(experiment_id)` | 87 | **REUSE** | Canonical Step8A path resolution. |
| `run_oof_condition(...)` | 356 | **PATTERN** | Shows the exact `assign_large_blocks` → population filter → `train_population(group_column="large_block_id", strict_folds=True)` sequence that produced the frozen ceiling. Followed so the ceiling reproduces. |

`src/step8_big_block_robustness.py` — **NOT USED**. The v2 generic runner
(`DEFAULT_BLOCK_SIZES = (10, 20)`, line 106) covers the same ground; the v1
frozen module is the one whose artifacts this analysis anchors against. Noted
because the orchestrator routes `step8_big_blocks_v2` through the generic
adapter and this analysis must not disturb that routing.

## 3. Raw transfer implementation

`src/step9b_run_cross_region_transfer.py`

| Symbol | Line | Use | Supplies |
|---|---:|---|---|
| `MODEL_NAME` | 85 | **REUSE** | `"random_forest"`. |
| `run_one_direction_population(...)` | 177 | **PATTERN** | The canonical source-only transfer: build pipeline on the source feature frame, `pipeline.fit(X_source, y_source)`, `predict_proba(X_target)`. Lines 227–236 are the exact `raw` (k=0) construction. |
| `load_step8a_dataset(experiment_id)` | 95 | **PATTERN** | Same load + block assign, but this analysis substitutes `assign_large_blocks(df, 10)` for `add_spatial_block_id(df, 2)`. |
| `population_subset(df, population)` | 107 | **REUSE** | `valid_for_modeling == True` ∧ named boolean population column. |
| `select_threshold_from_source_oof(...)` | 119 | **NOT USED** | This analysis reports only threshold-free metrics; reusing it would add ~5 fits per direction for no reported quantity. |
| `compute_metrics_at_threshold(...)` | 152 | **NOT USED** | Threshold-dependent; superseded by `compute_binary_metrics`. |
| `_dataset_provenance(experiment_id)` | 262 | **PATTERN** | Shape of the per-experiment provenance block (path, sha256, manifest sha256, feature contract). |

`src/step9a_audit_cross_region_inputs.py`

| Symbol | Line | Use |
|---|---:|---|
| `PRIMARY_POPULATIONS` | 128 | **REUSE** — `["burnable_tree_shrub_grass"]`; the single source of truth for the population. |
| `resolve_step8a_dataset_path(experiment_id)` | 140 | **REUSE** |
| `resolve_step8a_stats_path(experiment_id)` | 171 | **REUSE** |
| `sha256_file(path)` | 196 | **REUSE** |
| `resolve_git_commit()` | 204 | **REUSE** |
| `resolve_feature_contract(experiment_id)` | 231 | **REUSE** — fails closed if an experiment's feature contract has drifted. |
| `cross_region_output_root(source_id, target_id)` | 136 | **NOT USED** — this analysis writes only under its own diagnostics namespace. |

`core/cross_region_experiment.py`

| Symbol | Use |
|---|---|
| `SHARED_BASELINE_FEATURES`, `SHARED_THERMAL_FEATURES`, `SHARED_THERMAL_MODEL_FEATURES` | **REUSE** — the cross-region feature contract, re-exported from Step9A's single source. |
| `ALL_POPULATIONS`, `PRIMARY_POPULATIONS`, `SECONDARY_POPULATIONS` | **REUSE** (primary only). |
| `paired_spatial_block_bootstrap` | **NOT USED** — no bootstrap is run. |

## 4. Within-region ceiling / OOF evaluation

| Source | Use |
|---|---|
| `step8b.train_population` OOF loop, lines 570–597 | **PATTERN** — the per-fold fit → `predict_proba` → accumulate-into-OOF-vector sequence. |
| `outputs/robustness/step8_large_block/manavgat_2021__bejis_2022/{manavgat_2021,bejis_2022}/block_10_cells/step8b_large_block_metrics.json` | **REFERENCE** — frozen 10-cell ceiling. manavgat baseline 0.747550 / thermal 0.797430; bejis baseline 0.779370 / thermal 0.824469. Hard reproduction target. |
| `…/block_10_cells/step8c_large_block_bootstrap_summary.json` | **REFERENCE** — frozen 10-cell paired block bootstrap (1000 replicates, seed 42, 2.5/97.5 percentile). Copied verbatim into `summary.json` as `external_ceiling_reference`. |
| `outputs/experiments/<exp>/step8b/step8b_metrics.json` | **NOT USED** — 2-cell within-region OOF; wrong block scale for this evaluation frame. |

## 5. Metric helpers

| Metric | Function | Origin |
|---|---|---|
| ROC-AUC | `sklearn.metrics.roc_auc_score` | via `compute_binary_metrics` (step8b:496) |
| PR-AUC | `sklearn.metrics.average_precision_score` | via `compute_binary_metrics` (step8b:497) |
| Brier | `sklearn.metrics.brier_score_loss` | via `compute_binary_metrics` (step8b:498) |

`core/step10_shared.compute_threshold_free_metrics` (line 227) is an
equivalent threshold-free helper. **NOT USED** — `compute_binary_metrics` is
the older and more widely reused of the two, and is what the ceiling anchor
was computed with.

## 6. Recovery-fraction prior art

`src/transfer_decomposition.py` — **PATTERN**, with three deliberate departures.

| Symbol | Line | Relationship |
|---|---:|---|
| `decompose_direction(...)` | 176 | Defines `raw_gap = within - raw`, `adaptation_effect = adapted - raw`, `recovered_fraction = adaptation_effect / raw_gap`. This analysis's `ceiling_gap`, `absolute_recovery` and `recovery_fraction` are the same three quantities with `adapted → fewshot(k)` and `within → ceiling`. |
| `RATIO_DEGENERATE_THRESHOLD` | 78 | `1e-6` — **REUSE** as this analysis's degenerate-denominator threshold. |
| `interpretable = raw_gap > 0.0` guard | 227 | **DEPART**: that module sets the fraction to `None` whenever `raw_gap <= 0`. This analysis keeps the signed fraction whenever the denominator is non-degenerate and raises `ceiling_not_above_raw` as a flag instead, because the contract requires signed values to be preserved. |
| `CI_LOWER_PCT` / `CI_UPPER_PCT` | 80–81 | **DEPART**: same 2.5/97.5 percentiles, but over 10 repeats and named `selection_p2_5` / `selection_p97_5`, never `ci_*`. |
| `STATUS_ABOVE_CHANCE`, `STATUS_RELATIVE_ONLY`, … | 84–88 | **NOT USED** — interval-derived support statuses are significance claims in all but name. |
| `MODEL_FAMILIES` | 73 | **REUSE** — `("baseline", "thermal")`. |

## 7. Output, provenance and manifest patterns

`src/marginal_aoa_completion.py` — the closest structural precedent
(diagnostics namespace, staged, hash-gated, label-firewalled).

| Symbol | Line | Pattern supplied |
|---|---:|---|
| `SCHEMA_VERSION` | 110 | `"<name>.v1"` string convention. |
| `DIAGNOSTIC_NAMESPACE`, `DIAGNOSTIC_CLASS` | 111–112 | Namespace token + analysis-class declaration. |
| `diagnostics_root()` / `analysis_root(analysis_id)` | 405 / 410 | `outputs/diagnostics/<namespace>/<analysis_id>/`. |
| `canonical_step8a_path(experiment_id, experiments_root=None)` | 414 | Injectable canonical path resolution. |
| `build_frozen_input_inventory(...)` | 517 | `input_hashes.json` shape. |
| `assert_canonical_step8a_hashes(inventory, strict=True)` | 534 | Hard hash gate before any computation. |
| `validate_stage_range(from_stage, to_stage)` | 367 | Stage-order validation. |
| `stage_side_effect_flags(stages)` | 391 | Which stages may write. |
| `planned_output_layout()` | 444 | Dry-run declaration of every file that will be written. |
| `directed_pairs(...)` / `pair_token(...)` / `direction_token(...)` | 474 / 492 / 497 | Never-sorted direction tokens; `EXPECTED_DIRECTED_PAIRS` guard. |
| `FOLD_BLOCK_SIZE_CELLS = 10`, `FOLD_COUNT = 5` | 168–170 | Independent confirmation that ~5 km / 5 folds is an established convention here. Its `FOLD_ASSIGNMENT_METHOD = "sorted_block_round_robin_5_folds"` is **NOT** adopted — this analysis needs stratification, so it uses `make_spatial_folds(..., strict=True)`. |

`core/step10_shared.py`

| Symbol | Line | Pattern |
|---|---:|---|
| `canonical_json(obj)` | 114 | Stable JSON serialisation for hashing. |
| `compute_analysis_id(scientific_config)` | 118 | `analysis_id` = digest of the frozen scientific config. |
| `git_commit_if_available()` | 122 | Provenance. |
| `package_versions()` | 135 | numpy/pandas/sklearn versions into the manifest. |
| `check_no_forbidden_features(feature_list)` | 66 | Second leakage guard. |
| `assert_label_blind(df, context)` | 243 | **NOT USED** — this analysis is deliberately label-*using*; asserting label-blindness would be false. The inverse assertion (labels only in `y` of a training frame or `y_true` of an evaluation) is implemented instead. |
| `run_n_way_paired_bootstrap(...)` | 267 | **NOT USED** — no bootstrap. |

## 8. CLI / orchestrator patterns

| File | Pattern |
|---|---|
| `scripts/run_marginal_aoa_completion.py` | Thin dispatcher: `main(...)` → `run_analysis(...)`, `build_parser()`, no scientific logic in the script. Template for `scripts/run_few_shot_recovery.py`. |
| `scripts/validate_marginal_aoa_completion.py` | `Report` class emitting `{check_id, status, expected, observed, evidence_path, note}`; dry-run vs actual modes; any FAIL ⇒ overall FAIL. Template for `scripts/validate_few_shot_recovery.py`. |
| `core/pipeline_orchestrator.py::run_marginal_aoa_completion_stage` (line 784) | Stage-dispatch signature: `from_stage`/`to_stage`/`dry_run`/`resume`/`output_root`/`experiments_root`, docstring stating the analysis class and the namespace. Template for `run_few_shot_recovery_stage`. |
| `core/pipeline_orchestrator.py::_assert_context_is_safely_namespaced` (line 93) | Namespace containment guard. |
| `scripts/main.py` | Subcommand registration surface. |

## 9. Test fixture patterns

`tests/test_marginal_aoa_completion.py` (2384 lines) is the model. Reused
patterns:

- module-level synthetic fixture builders, no dependence on production outputs
  for unit tests;
- `tmp_path`-injected `output_root` so no test can write to a canonical path;
- adversarial fail-closed tests (`test_*_fails_closed`) for every guard;
- invariance tests (`test_selection_order_does_not_change_pairs`,
  `test_di_is_invariant_to_source_row_order`) — directly adapted here as
  row-order and selection-order invariance tests;
- `test_exactly_twelve_directed_pairs` → `test_exactly_six_directed_pairs`;
- counterfactual leakage tests (`test_changing_target_labels_cannot_change_output`)
  → adapted to the narrower true claim: changing *evaluation-block* labels
  cannot change any training decision or any selected block.

`tests/test_step8_large_block_robustness.py` supplies the block-assignment and
strict-fold assertion patterns.

---

## 10. Reuse summary

**Imported and called unchanged (13 symbols):**
`build_pipeline`, `build_classifier` (indirect), `check_no_forbidden_features`,
`make_spatial_folds`, `compute_binary_metrics`, `assign_large_blocks`,
`population_subset`, `resolve_step8a_dataset_path`,
`resolve_step8a_stats_path`, `resolve_feature_contract`, `sha256_file`,
`resolve_git_commit`, `canonical_json` / `compute_analysis_id`.

**Constants reused (11):** `BASELINE_FEATURES`, `THERMAL_FEATURES`,
`THERMAL_MODEL_FEATURES`, `CATEGORICAL_FEATURES`, `FORBIDDEN_FEATURE_COLUMNS`,
`TARGET_COLUMN`, `MODEL_NAME`, `STEP8B_RANDOM_SEED`, `STEP8B_N_SPLITS`,
`STEP8B_MIN_POSITIVES_PER_POPULATION`, `NOMINAL_SCALES`.

**New logic required (only this):** the nested tiered block-ordering rule
(§7.3 of the contract), the budget loop, the signed unclipped recovery
arithmetic with its three status values, selection-interval summarisation, and
the schema/validator surface. No new model, no new preprocessing, no new fold
algorithm, no new metric, no new bootstrap.
