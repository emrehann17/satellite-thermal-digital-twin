# Output Contract — `few_shot_recovery.v1`

```
SCHEMA_VERSION      = "few_shot_recovery.v1"
DIAGNOSTIC_NAMESPACE = "few_shot_recovery"
DIAGNOSTIC_CLASS     = "target_label_supervised_few_shot_adaptation_sensitivity"
```

Namespace, and the **only** place this analysis may write:

```
outputs/diagnostics/few_shot_recovery/<analysis_id>/
```

`analysis_id` is `compute_analysis_id(scientific_config)` — a digest over the
canonical JSON of the frozen scientific configuration (schema version,
directions, population, budgets, repeat count, block size, fold count, seeds,
feature lists, model hyperparameters, input hashes). Changing any frozen
decision produces a different `analysis_id` and therefore a different
directory; it can never silently overwrite an existing result.

## 1. Stages

| Stage | Reads | Writes | Fits models |
|---|---|---|---|
| `plan` | canonical Step8A datasets (metadata + blocks only) | `config.json`, `input_hashes.json`, `target_block_inventory.csv`, `direction_budget_feasibility.csv`, `selected_blocks.parquet`, `stages/plan.json` | no |
| `fit` | plan artifacts + Step8A datasets | `oof_predictions.parquet`, `repeat_metrics.csv`, `stages/fit.json` | yes |
| `summarize` | `repeat_metrics.csv` | `recovery_curve.csv`, `summary.json`, `report.md`, `figures/*`, `stages/summarize.json` | no |
| `validate` | everything above + frozen ceiling references | `validation_report.json`, `stages/validate.json`, `manifest.json` | no |

Stage order is fixed. `validate_stage_range(from_stage, to_stage)` rejects any
out-of-order or unknown range. `--dry-run` runs `plan`'s checks and emits
`planned_output_layout()` without writing any artifact and without fitting.

Block selection is entirely determined at `plan` and frozen in
`selected_blocks.parquet`; `fit` may not re-derive it. This makes the
selection auditable before a single model is trained.

## 2. Files

```
outputs/diagnostics/few_shot_recovery/<analysis_id>/
├── config.json
├── input_hashes.json
├── target_block_inventory.csv
├── direction_budget_feasibility.csv
├── selected_blocks.parquet
├── oof_predictions/
│   └── direction=<source>_to_<target>/part-0.parquet
├── repeat_metrics.csv
├── recovery_curve.csv
├── summary.json
├── report.md
├── validation_report.json
├── figures/
│   ├── recovery_curve_roc_auc.png
│   ├── recovery_curve_pr_auc.png
│   └── recovery_curve_brier.png
├── stages/
│   ├── plan.json
│   ├── fit.json
│   ├── summarize.json
│   └── validate.json
└── manifest.json
```

`oof_predictions` is written partitioned by direction (six parts) rather than
as one file; see §4.

## 3. Shared provenance columns

Every CSV/parquet row carries these, so no row can be read out of context:

| Column | Type | Notes |
|---|---|---|
| `schema_version` | str | `few_shot_recovery.v1` |
| `analysis_id` | str | |
| `source_experiment` | str | never sorted with target |
| `target_experiment` | str | |
| `direction` | str | `<source>_to_<target>` |
| `population` | str | `burnable_tree_shrub_grass` |
| `source_step8a_sha256` | str | canonical input hash |
| `target_step8a_sha256` | str | canonical input hash |
| `block_size_cells` | int | `10` |
| `block_nominal_scale` | str | `approximately_5_km` |
| `n_outer_folds` | int | `5` |
| `estimator_random_state` | int | `42` |

## 4. `oof_predictions/` — per-row OOF predictions

One row per (direction × condition × budget × repeat × target population
cell). Both model families are columns, not rows, which halves the row count.

| Column | Type | Notes |
|---|---|---|
| *(shared provenance)* | | |
| `condition` | str | `raw` \| `few_shot` \| `ceiling` |
| `budget_blocks` | int16 | `0` for raw; `k` for few_shot; `-1` for ceiling |
| `repeat_id` | int8 | `0` for raw and ceiling; `0..9` for few_shot |
| `outer_fold` | int8 | fold whose model produced this prediction |
| `evaluation_block_id` | str | `b10_r{r}_c{c}` — the cell's evaluation block |
| `cell_id` | int64 | canonical Step8A cell identifier |
| `burned` | int8 | target label, used only as `y_true` |
| `baseline_probability` | float32 | |
| `thermal_probability` | float32 | |
| `selection_key` | str | join key into `selected_blocks.parquet`; null for raw/ceiling |

Row count: `Σ_directions |target_pop| × (1 raw + 1 ceiling + 6 budgets × 10
repeats)` = `154 862 × 62` ≈ **9.60 M rows**. Estimated 250–450 MB as
snappy parquet with float32 probabilities. Partitioning by direction keeps
each part at ~1.3–2.6 M rows so the file can be read one direction at a time.

This file is what discharges the "every repeat must produce a full target OOF
prediction" requirement, and it is what the OOF-coverage validator checks read.

## 5. `selected_blocks.parquet` — adaptation block provenance

One row per selected block: (direction × outer_fold × repeat × budget ×
selected block).

| Column | Type | Notes |
|---|---|---|
| *(shared provenance)* | | |
| `selection_key` | str | `{direction}|{fold}|{repeat}|{k}` |
| `outer_fold` | int8 | |
| `repeat_id` | int8 | `0..9` |
| `budget_blocks` | int16 | `k` |
| `selection_seed` | int64 | derived per (direction, fold, repeat) — identical across `k` |
| `adaptation_block_id` | str | `b10_r{r}_c{c}` |
| `selection_rank` | int16 | position in the nested ordering, `0..k-1` |
| `block_tier` | str | `both_classes` \| `positives_only` \| `negatives_only` |
| `block_row_count` | int32 | cells in this block |
| `block_positive_count` | int32 | burned cells in this block |

Because the ordering is nested, a block at `selection_rank = r` appears in
every budget `k > r` for the same (direction, fold, repeat). Nesting is
therefore checkable directly from `selection_rank`.

## 6. `repeat_metrics.csv` — per-repeat metric values

One row per (direction × family × metric × condition × budget × repeat ×
evaluation level).

| Column | Type | Notes |
|---|---|---|
| *(shared provenance)* | | |
| `model_family` | str | `baseline` \| `thermal` |
| `model_role` | str | `primary` for thermal, `secondary` for baseline |
| `metric` | str | `roc_auc` \| `pr_auc` \| `brier_score` |
| `metric_orientation` | str | `higher_is_better` \| `lower_is_better_oriented_by_negation` |
| `condition` | str | `raw` \| `few_shot` \| `ceiling` |
| `budget_blocks` | int16 | `0` / `k` / `-1` |
| `repeat_id` | int8 | |
| `evaluation_level` | str | `oof` (whole target OOF) \| `fold` |
| `outer_fold` | int8 | `-1` when `evaluation_level == "oof"` |
| `metric_value` | float64 | natural sign (Brier stays positive) |
| `oriented_value` | float64 | `-metric_value` for Brier, else `metric_value` |
| `n_evaluation_rows` | int32 | |
| `n_evaluation_positives` | int32 | |
| `n_evaluation_blocks` | int32 | |
| `n_train_rows` | int32 | training frame size for the fit behind this row |
| `n_train_source_rows` | int32 | `0` for ceiling |
| `n_train_target_rows` | int32 | `0` for raw |
| `adaptation_row_count` | int32 | target rows added by the k blocks; `0` for raw |
| `adaptation_positive_count` | int32 | burned cells among them |
| `n_blocks_tier_a` / `_b` / `_c` | int16 | tier composition of the selection |
| `selection_key` | str | null for raw/ceiling |
| `fit_id` | str | digest identifying the unique fit; shared where fits are shared |

`fit_id` makes fit sharing auditable: the 60 ceiling rows per family/metric
resolve to 30 distinct `fit_id`s, and raw rows across folds share one.

## 7. `recovery_curve.csv` — the primary result

One row per (direction × family × metric × budget). 6 × 2 × 3 × 7 = **252
rows**.

| Column | Type | Notes |
|---|---|---|
| *(shared provenance)* | | |
| `model_family`, `model_role`, `metric`, `metric_orientation` | | as above |
| `budget_blocks` | int16 | `0, 1, 2, 4, 8, 16, 32` |
| `n_repeats` | int8 | `1` at `k=0`, else `10` |
| **point estimates (natural sign)** | | |
| `raw_value` | float64 | k=0 value for this direction/family/metric |
| `fewshot_value` | float64 | repeat **median** at this budget; equals `raw_value` at k=0 |
| `ceiling_value` | float64 | target-only value |
| **oriented values (used in the arithmetic)** | | |
| `raw_oriented`, `fewshot_oriented`, `ceiling_oriented` | float64 | |
| **recovery quantities** | | |
| `absolute_recovery` | float64 | `fewshot_oriented - raw_oriented` |
| `ceiling_gap` | float64 | `ceiling_oriented - raw_oriented` |
| `recovery_fraction` | float64/null | `absolute_recovery / ceiling_gap`, **unclipped, signed** |
| `recovery_fraction_status` | str | `interpretable` \| `undefined_degenerate_denominator` \| `ceiling_not_above_raw` |
| `ceiling_not_above_raw` | bool | `ceiling_oriented <= raw_oriented` |
| **selection interval** | | |
| `selection_median` | float64 | median of the repeat OOF values (natural sign) |
| `selection_p2_5` | float64 | 2.5th percentile over repeats |
| `selection_p97_5` | float64 | 97.5th percentile over repeats |
| `selection_min`, `selection_max` | float64 | observed extremes, so the interval can be read against them |
| `recovery_fraction_selection_median` | float64/null | fraction recomputed per repeat, then median |
| `recovery_fraction_selection_p2_5` | float64/null | |
| `recovery_fraction_selection_p97_5` | float64/null | |
| **composition** | | |
| `mean_adaptation_row_count` | float64 | |
| `mean_adaptation_positive_count` | float64 | |
| `mean_n_blocks_tier_a` / `_b` / `_c` | float64 | shows the k=16/32 transition |

`fewshot_value` at a budget is the **median across repeats**, matching
`selection_median`. Means are not reported as the point estimate; with 10
repeats the median is the more stable summary and it is the one the interval
is built around.

Interval columns at `k = 0` and for the ceiling equal the point estimate
(`n_repeats = 1`), since neither has selection randomness.

## 8. `direction_budget_feasibility.csv`

6 × 5 × 7 = **210 rows**, one per (direction × outer_fold × budget).

| Column | Notes |
|---|---|
| *(shared provenance)*, `outer_fold`, `budget_blocks` | |
| `target_pool_blocks` | training-pool blocks for this fold |
| `pool_tier_a`, `pool_tier_b`, `pool_tier_c` | pool composition |
| `feasible` | bool — `budget_blocks <= target_pool_blocks` |
| `infeasibility_reason` | null when feasible |
| `fills_from_positive_containing_only` | bool |
| `requires_tier_c` | bool |
| `min_selected_positive_blocks` | over the 10 repeats |
| `n_repeats_planned` | `1` at k=0, else `10` |

All 210 rows carry `feasible = true` (see `BLOCK_BUDGET_FEASIBILITY.md`). The
file exists so the claim is auditable rather than asserted, and so that any
future AOI addition surfaces an infeasible budget explicitly.

## 9. `target_block_inventory.csv`

One row per (target × outer_fold), plus one `outer_fold = -1` row per target
carrying the whole-target totals. Columns mirror §3.1 and §3 of
`BLOCK_BUDGET_FEASIBILITY.md`: `total_blocks`, `blocks_with_burned`,
`blocks_unburned_only`, `blocks_both_classes`, `blocks_burned_only`,
`median_rows_per_block`, `pool_blocks`, `eval_blocks`, `eval_rows`,
`eval_positives`, plus `blocks_2_cell_reference` for the comparison table.

## 10. `config.json`

The frozen scientific configuration — the exact object hashed into
`analysis_id`.

```jsonc
{
  "schema_version": "few_shot_recovery.v1",
  "diagnostic_class": "target_label_supervised_few_shot_adaptation_sensitivity",
  "analysis_id": "<digest>",
  "created_at": "<iso8601>",
  "git_commit": "<sha or null>",
  "package_versions": {"numpy": "...", "pandas": "...", "scikit-learn": "..."},

  "primary_experiments": ["manavgat_2021", "bejis_2022", "mugla_2021"],
  "excluded_experiments": {
    "evia_2021_extended": "high_prevalence_different_regime_sensitivity_control_not_equal_prevalence_primary_transfer_validation_aoi",
    "evia_2021": "out_of_scope_for_this_frozen_analysis"
  },
  "directed_pairs": [["manavgat_2021","bejis_2022"], ["manavgat_2021","mugla_2021"],
                     ["bejis_2022","manavgat_2021"], ["bejis_2022","mugla_2021"],
                     ["mugla_2021","manavgat_2021"], ["mugla_2021","bejis_2022"]],
  "expected_directed_pairs": 6,
  "population": "burnable_tree_shrub_grass",
  "valid_universe": "valid_for_modeling == True",

  "model": {
    "name": "random_forest",
    "class": "RandomForestClassifier",
    "hyperparameters": {"n_estimators": 300, "max_depth": null, "min_samples_leaf": 3,
                        "class_weight": "balanced", "random_state": 42, "n_jobs": -1},
    "source": "src/step8b_train_baseline_vs_thermal_model.build_classifier",
    "tuning_performed": false,
    "sample_weight_argument_used": false,
    "pre_existing_class_weighting": {
      "present": true, "mechanism": "class_weight='balanced'",
      "note": "Canonical Step8B/9B/10 behaviour, computed on whichever training frame is fitted. Not introduced by this analysis; no new weighting rule added."
    }
  },
  "families": {"primary": "thermal", "secondary": "baseline"},
  "feature_lists": {"baseline": ["..."], "thermal": ["..."]},
  "preprocessing": {
    "numeric_imputation": "median",
    "categorical_imputation": "most_frequent",
    "categorical_encoding": "one_hot_handle_unknown_ignore",
    "fit_scope": "training_frame_of_each_condition_only"
  },

  "spatial_blocks": {
    "block_size_cells": 10,
    "nominal_scale": "approximately_5_km",
    "utility": "src/step8_large_block_robustness.assign_large_blocks",
    "id_format": "b10_r{block_row}_c{block_col}",
    "origin": [0, 0],
    "assigned_before_population_filtering": true,
    "canonical_small_block_size_cells": 2,
    "departure_reason": "2-cell blocks hold a median of 4 cells; not a unit of labeling effort and adjacent to evaluation blocks. ~5 km fallback pre-authorised and already canonical in this repository."
  },
  "outer_folds": {
    "utility": "src/step8b_train_baseline_vs_thermal_model.make_spatial_folds",
    "splitter": "StratifiedGroupKFold", "n_splits": 5, "shuffle": true,
    "random_state": 42, "strict": true,
    "grouping_column": "large_block_id",
    "depends_on": "target_only"
  },

  "budgets": [0, 1, 2, 4, 8, 16, 32],
  "budgets_dropped": [],
  "n_repeats": 10,
  "n_repeats_raw": 1,
  "n_repeats_ceiling": 1,
  "selection": {
    "unit": "spatial_block",
    "nested": true,
    "tier_order": ["both_classes", "positives_only", "negatives_only"],
    "within_tier": "shuffle_with_derived_rng_after_sorting_by_block_id",
    "seed_derivation": "blake2b(schema|source|target|outer_fold|repeat, digest_size=8) mod 2**32",
    "seed_depends_on_budget": false,
    "seed_depends_on_model_family": false,
    "result_dependent_branching": false
  },

  "metrics": {"primary": "roc_auc", "secondary": ["pr_auc", "brier_score"],
              "helper": "src/step8b_train_baseline_vs_thermal_model.compute_binary_metrics",
              "brier_orientation": "oriented_brier = -brier_score",
              "threshold_selection_performed": false},
  "recovery": {"clipped": false, "absolute_valued": false,
               "degenerate_denominator_threshold": 1e-6,
               "statuses": ["interpretable", "undefined_degenerate_denominator", "ceiling_not_above_raw"]},
  "uncertainty": {"interval_name": "selection_interval",
                  "percentiles": [2.5, 97.5], "percentile_method": "linear",
                  "basis": "variability_across_block_selection_repeats_only",
                  "bootstrap_performed": false, "p_values_produced": false,
                  "forbidden_terms": ["confidence interval", "95% CI", "significant",
                                      "significance", "p-value", "istatistiksel olarak anlamlı"]},
  "earth_engine": {"used": false, "importable": false}
}
```

## 11. `input_hashes.json`

```jsonc
{
  "canonical_step8a": {
    "manavgat_2021": {"path": "...", "sha256": "054a1961...f3439", "expected_sha256": "054a1961...f3439", "match": true,
                      "step8a_manifest_path": "...", "step8a_manifest_sha256": "...", "feature_contract": {...}},
    "bejis_2022":    {"..." : "3dec785a...d9e393"},
    "mugla_2021":    {"..." : "c4ab107d...db7db8e"}
  },
  "external_references": {
    "ceiling_block_10_metrics": {
      "manavgat_2021": {"path": "outputs/robustness/step8_large_block/manavgat_2021__bejis_2022/manavgat_2021/block_10_cells/step8b_large_block_metrics.json", "sha256": "..."},
      "bejis_2022":    {"path": "outputs/robustness/step8_large_block/manavgat_2021__bejis_2022/bejis_2022/block_10_cells/step8b_large_block_metrics.json", "sha256": "..."},
      "mugla_2021": null
    },
    "ceiling_block_10_bootstrap": {"manavgat_2021": {...}, "bejis_2022": {...}, "mugla_2021": null}
  },
  "read_only_assertion": "No path outside outputs/diagnostics/few_shot_recovery/<analysis_id>/ was opened for writing."
}
```

## 12. `summary.json`

```jsonc
{
  "schema_version": "few_shot_recovery.v1",
  "analysis_id": "...",
  "headline": {
    "primary_metric": "roc_auc",
    "primary_family": "thermal",
    "per_direction": [
      {"direction": "manavgat_2021_to_bejis_2022",
       "raw_value": null, "ceiling_value": null, "ceiling_gap": null,
       "budget_curve": [{"budget_blocks": 1, "fewshot_value": null,
                         "selection_p2_5": null, "selection_p97_5": null,
                         "recovery_fraction": null,
                         "recovery_fraction_status": "interpretable"}]}
    ]
  },
  "external_ceiling_reference": {
    "manavgat_2021": {
      "source_path": "outputs/robustness/step8_large_block/.../block_10_cells/step8c_large_block_bootstrap_summary.json",
      "interval_name_in_source": "spatial_block_bootstrap_2_5 / spatial_block_bootstrap_97_5",
      "n_replicates": 1000, "random_seed": 42,
      "auc_thermal": {"lower": 0.7376477521885416, "upper": 0.8508178369043737},
      "auc_baseline": {"lower": 0.7010930834586141, "upper": 0.7948377481309672},
      "is_selection_interval": false,
      "comparable_to_raw_endpoint": false,
      "note": "Frozen 10-cell paired spatial-block bootstrap of the target-only ceiling. Reproduced here verbatim for context. It is NOT a selection interval and there is no comparable interval for the raw endpoint."
    },
    "bejis_2022": {"...": "..."},
    "mugla_2021": {
      "source_path": "outputs/experiments/mugla_2021/robustness/step8_big_blocks/block_10_cells/bootstrap_summary.json",
      "...": "same fields; 1000 replicates, seed 42, 10-cell blocks"
    }
  },
  "ceiling_reproduction": {
    "manavgat_2021": {"family": {"baseline": {"expected": 0.7475502988238435, "observed": null, "abs_diff": null, "tolerance": 1e-9, "match": null},
                                 "thermal":  {"expected": 0.7974298472620660, "observed": null, "abs_diff": null, "tolerance": 1e-9, "match": null}}},
    "bejis_2022":    {"family": {"baseline": {"expected": 0.7793700238725079, "...": "..."},
                                 "thermal":  {"expected": 0.8244685786179753, "...": "..."}}},
    "mugla_2021":    {"family": {"baseline": {"expected": 0.6979859420145867, "...": "..."},
                                 "thermal":  {"expected": 0.7773268638729566, "...": "..."}}}
  },
  "limitations": [
    "Outer evaluation blocks are 10-cell (~5 km), not Step8B's canonical 2-cell blocks. Values are not directly comparable to 2-cell Step8B/Step9B/Step10 numbers.",
    "No bootstrap interval exists for the raw endpoint at this block scale; existing Step9C/Step10 replicates resample 2-cell blocks and are not comparable. No new bootstrap was designed.",
    "The reported interval is a selection interval over 10 repeats. It describes block-selection variability only, is not a confidence interval, and supports no significance claim.",
    "The frozen 10-cell ceiling anchors come from two separate robustness namespaces: the paired large-block run for manavgat_2021 and bejis_2022, and the per-experiment big-block run for mugla_2021. Both are read-only reproduction anchors under the same 10-cell contract; neither was produced or re-run by this analysis.",
    "At k=16 and k=32 some folds must include unburned-only adaptation blocks; the tier composition columns record where.",
    "evia_2021_extended is excluded by design; nothing here describes high-prevalence different-regime transfer."
  ],
  "fit_accounting": {"unique_fits": null, "raw_fits": null, "fewshot_fits": null, "ceiling_fits": null},
  "p_values_produced": false,
  "bootstrap_performed": false,
  "earth_engine_used": false
}
```

## 13. `report.md`

Human-readable. Must contain: the claim boundary verbatim from
`SCIENTIFIC_CONTRACT.md` §2; the three forced decisions; the recovery curve
per direction for the primary metric and family; the secondary metrics; the
Brier orientation note with both signs shown; the tier-composition transition
at k=16/32; and the limitations list. Must contain none of the forbidden
uncertainty terms, and must call every interval a *selection interval*.

## 14. `manifest.json`

Written last, by `validate`. Lists every produced file with size and sha256;
records `analysis_id`, `schema_version`, `git_commit`, `package_versions`,
stage timings, `fit_accounting`, `ceiling_fit_sharing:
"per_target_fold_family"`, the input hash block, and the overall validation
status. A `manifest.json` whose validation status is not `PASS` marks the
directory as not citable.
