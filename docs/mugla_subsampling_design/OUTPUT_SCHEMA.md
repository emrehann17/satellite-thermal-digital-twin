# Output Schema — `mugla_subsampling.v1`

```
SCHEMA_VERSION      = "mugla_subsampling.v1"
DIAGNOSTIC_NAMESPACE = "mugla_subsampling"
DIAGNOSTIC_CLASS     = "population_size_matched_subsampling_sensitivity"
```

**Namespace (the only writable location):**

```
outputs/diagnostics/mugla_subsampling/<analysis_id>/
```

`analysis_id` = `compute_analysis_id(scientific_config)` — the sha256 of the
canonical JSON of the frozen scientific configuration. Identical configuration
⇒ identical directory; any contract change moves the whole run to a new one.
Every write passes `assert_inside_namespace(path, analysis_root)`.

---

## 1. Stages

```
STAGES = ("plan", "fit", "summarize")
STAGE_REQUIRES = {"plan": (), "fit": ("plan",), "summarize": ("plan", "fit")}
```

The validator is a **separate executable** (`scripts/validate_mugla_subsampling.py`)
and is not a stage.

| Stage | Writes | Fits models |
|---|---|---|
| `plan` | `config.json`, `input_hashes.json`, `sampling_inventory.csv`, `stratum_allocation.csv`, `selected_cells.parquet`, `fold_mapping.parquet`, `reference_metrics.csv` | no |
| `fit` | `oof_predictions/`, `repeat_metrics.csv` | yes (240) |
| `summarize` | `subsampling_summary.csv`, `summary.json`, `report.md`, `manifest.json` | no |

The complete cell selection is frozen and hashed at the end of `plan`, before
any model is fitted. `--dry-run` prints `planned_output_layout()` and writes
nothing.

## 2. File layout

```
outputs/diagnostics/mugla_subsampling/<analysis_id>/
├── config.json
├── input_hashes.json
├── sampling_inventory.csv
├── stratum_allocation.csv
├── selected_cells.parquet
├── fold_mapping.parquet
├── reference_metrics.csv
├── oof_predictions/                    # logical dataset, partitioned by arm
│   ├── part-within.parquet
│   ├── part-source.parquet
│   └── part-target.parquet
├── repeat_metrics.csv
├── subsampling_summary.csv
├── summary.json
├── report.md
├── stages/
│   ├── plan.json
│   ├── fit.json
│   └── summarize.json
├── manifest.json
└── _quarantine/                        # only if a partial partition was found
```

`oof_predictions/` is a partitioned logical dataset rather than a single file
so that a failed `fit` can be resumed at partition granularity; a partition
whose hash does not match its stage marker is moved to `_quarantine/` and
rewritten. `sha256_path` hashes the dataset by hashing its sorted member files.

## 3. `config.json`

The object hashed into `analysis_id`. Contains, and contains nothing that is
not scientifically load-bearing:

```jsonc
{
  "schema_version": "mugla_subsampling.v1",
  "diagnostic_class": "population_size_matched_subsampling_sensitivity",
  "experiments": ["manavgat_2021", "bejis_2022", "mugla_2021"],
  "excluded_experiments": ["evia_2021", "evia_2021_extended", "kozan_2023"],
  "subsampled_experiment": "mugla_2021",
  "population": "burnable_tree_shrub_grass",
  "valid_universe": "valid_for_modeling == True",
  "arm_id": "size_matched_to_manavgat",
  "target_sample_size": 20511,
  "n_repeats": 20,
  "sampling": {
    "with_replacement": false,
    "block_size_cells": 10,
    "nominal_scale": "approximately_5_km",
    "block_column": "large_block_id",
    "stratum_definition": "large_block_id x label",
    "allocation_method": "hamilton_largest_remainder_integer_exact",
    "allocation_tie_break": "sort by (-remainder_numerator, stratum_id) ascending",
    "within_stratum_order": "cell_id ascending, then rng permutation",
    "seed_derivation": "blake2b(schema_version|repeat_id|stratum_id, 8) % 2**32"
  },
  "folds": {
    "source": "inherited_from_frozen_full_mugla_artifact",
    "artifact": ".../block_10_cells/oof_predictions.parquet",
    "fold_column": "fold_id",
    "fold_count": 5,
    "splitter": "StratifiedGroupKFold",
    "shuffle": true,
    "random_state": 42,
    "strict_folds": true,
    "reoptimised_per_repeat": false
  },
  "model": {
    "name": "random_forest",
    "constructor": "step8b.build_pipeline",
    "estimator_seed": 42,
    "families": ["baseline", "thermal"]
  },
  "features": {
    "baseline": ["ndvi_mean", "elevation_mean", "slope_mean", "landcover_dominant"],
    "thermal_model_full": ["... 10 columns ..."],
    "categorical": ["landcover_dominant"]
  },
  "metrics": {
    "primary": "roc_auc",
    "secondary": ["pr_auc", "brier_score"],
    "orientation": {
      "roc_auc": "higher_is_better",
      "pr_auc": "higher_is_better",
      "brier_score": "lower_is_better_oriented_by_negation"
    }
  },
  "summarisation": {
    "statistics": ["median", "p2_5", "p97_5", "minimum", "maximum"],
    "percentile_method": "linear",
    "interval_names": ["subsampling_interval_lower", "subsampling_interval_upper"],
    "position_tokens": ["below_subsampling_interval",
                        "inside_subsampling_interval",
                        "above_subsampling_interval"]
  },
  "arms": ["within", "source", "target"],
  "directions": {
    "within": ["mugla_2021"],
    "source": ["mugla_2021_to_manavgat_2021", "mugla_2021_to_bejis_2022"],
    "target": ["manavgat_2021_to_mugla_2021", "bejis_2022_to_mugla_2021"]
  },
  "expected_unique_fits": {"within": 200, "source": 40, "target": 0, "total": 240}
}
```

## 4. `input_hashes.json`

```jsonc
{
  "step8a": {
    "manavgat_2021": {"path": "...", "sha256": "054a1961...", "rows_primary": 20511,
                      "positives": 784, "negatives": 19727},
    "bejis_2022":    {"path": "...", "sha256": "3dec785a...", "rows_primary": 15190,
                      "positives": 1100, "negatives": 14090},
    "mugla_2021":    {"path": "...", "sha256": "c4ab107d...", "rows_primary": 41730,
                      "positives": 2911, "negatives": 38819}
  },
  "within_reference": {
    "metrics_path": ".../block_10_cells/step8b_metrics.json",
    "metrics_sha256": "a826279f...",
    "oof_path": ".../block_10_cells/oof_predictions.parquet",
    "oof_sha256": "e16e6b18...",
    "fold_assignments_sha256": "e5b29928...",
    "block_manifest_sha256": "13dd4201..."
  },
  "transfer_references": {
    "mugla_2021_to_manavgat_2021": {"metrics_sha256": "1d8d7d53...", "predictions_sha256": "f28cef6c..."},
    "mugla_2021_to_bejis_2022":    {"metrics_sha256": "676c209e...", "predictions_sha256": "a82f6934..."},
    "manavgat_2021_to_mugla_2021": {"metrics_sha256": "3bd2a6ec...", "predictions_sha256": "0aace925..."},
    "bejis_2022_to_mugla_2021":    {"metrics_sha256": "a635143d...", "predictions_sha256": "183ce975..."}
  },
  "hash_gate": "strict",
  "git_commit": "...",
  "package_versions": {"numpy": "...", "pandas": "...", "scikit-learn": "..."}
}
```

`assert_canonical_step8a_hashes(..., strict=True)` runs immediately after this
file is built and before anything else.

## 5. `sampling_inventory.csv`

One row per (experiment, role). Population accounting, ~10 rows.

| Column | Type | Notes |
|---|---|---|
| `experiment_id` | str | |
| `role` | str | `subsampled_source_and_target` / `full_target` / `full_source` |
| `subsampled` | bool | `True` only for `mugla_2021` |
| `rows_total`, `rows_valid`, `rows_primary` | int | |
| `positives`, `negatives`, `prevalence` | int/int/float | |
| `n_blocks`, `n_strata` | int | null for non-subsampled experiments |
| `target_sample_size` | int | 20511, null where not applicable |
| `sampled_positives`, `sampled_negatives`, `sampled_prevalence` | int/int/float | |
| `prevalence_absolute_drift`, `prevalence_drift_bound` | float | 0.00035075 / 0.003413 |
| `step8a_sha256` | str | |

## 6. `stratum_allocation.csv`

One row per stratum — **636 rows**. Repeat-invariant, so it is written once.

| Column | Type | Example |
|---|---|---|
| `stratum_id` | str | `b10_r12_c34|L1` |
| `large_block_id` | str | `b10_r12_c34` |
| `large_block_row`, `large_block_col` | int | |
| `label` | int | 0 / 1 |
| `capacity` | int | rows available in the full population |
| `quota_numerator` | int | `capacity * 20511` |
| `floor_allocation` | int | `quota_numerator // 41730` |
| `remainder_numerator` | int | `quota_numerator % 41730` |
| `remainder_rank` | int | rank under `(-remainder_numerator, stratum_id)` |
| `received_remainder_unit` | bool | `remainder_rank < 300` |
| `allocation_count` | int | `floor_allocation + received_remainder_unit` |
| `capacity_headroom` | int | `capacity - allocation_count`, ≥ 0 |
| `fold_id` | int | inherited from the block |

Recomputable from `input_hashes.json` alone — this is what makes the allocation
independently checkable.

## 7. `selected_cells.parquet`

**20 × 20,511 = 410,220 rows.** The frozen selection, written before any fit.
Contains at least the fields named in the contract:

| Column | Type | Notes |
|---|---|---|
| `repeat_id` | int16 | 0 … 19 |
| `cell_id` | str | `r{row_500m}_c{col_500m}` |
| `large_block_id` | str | `b10_r{row//10}_c{col//10}` |
| `label` | int8 | `burned` |
| `fold_id` | int8 | inherited, 0 … 4 |
| `sampling_seed` | int64 | the 32-bit stratum seed actually used for this cell's stratum in this repeat |
| `stratum_id` | str | |
| `allocation_count` | int16 | the stratum's `allocation_count` |
| `mugla_step8a_sha256` | str | `c4ab107d…` (dictionary-encoded) |
| `repeat_seed` | int64 | the repeat-level seed, for auditability |
| `row_500m`, `col_500m` | int32 | canonical grid indices |

Invariants: exactly 20,511 rows per `repeat_id`; no duplicate
`(repeat_id, cell_id)`; every `cell_id` in the canonical Muğla primary set;
`Σ label` = 1,438 per repeat; row counts per `(repeat_id, fold_id)` = 4,111 /
4,096 / 4,107 / 4,096 / 4,101.

## 8. `fold_mapping.parquet`

**41,730 rows** — the inherited full-Muğla mapping, copied into the namespace so
the analysis is self-contained.

| Column | Type |
|---|---|
| `cell_id` | str |
| `large_block_id` | str |
| `frozen_block_id` | str (`block10_{r}_{c}`, as spelled in the source artifact) |
| `fold_id` | int8 |
| `label` | int8 |
| `source_artifact_path` | str |
| `source_artifact_sha256` | str |
| `fold_source` | str — `persisted_artifact` or `reproduced_fallback` |

## 9. `reference_metrics.csv`

The full-population references. 5 references × 2 families × 3 metrics =
**30 rows**.

| Column | Type | Notes |
|---|---|---|
| `arm` | str | `within` / `source` / `target` |
| `direction` | str | `mugla_2021` for within; `S_to_T` otherwise |
| `model_family` | str | `baseline` / `thermal` |
| `metric` | str | `roc_auc` / `pr_auc` / `brier_score` |
| `full_reference_value` | float | |
| `reference_artifact_path` | str | |
| `reference_artifact_sha256` | str | |
| `reference_n_rows`, `reference_n_positives` | int | |
| `recomputed_from_predictions` | float | independent recomputation |
| `recomputation_matches` | bool | must be `True` for all 30 rows |

## 10. `oof_predictions/`

Three partitions, one logical dataset. All carry `repeat_id`, the evaluated
`cell_id`, the true label and both families' probabilities.

**`part-within.parquet`** — 410,220 rows

```
repeat_id, cell_id, fold_id, large_block_id, burned,
baseline_probability, thermal_probability
```
Exactly-once OOF coverage per `repeat_id` is asserted before write.

**`part-source.parquet`** — 714,020 rows

```
repeat_id, direction, target_experiment_id, target_cell_id,
target_spatial_block_id, burned, baseline_probability, thermal_probability
```
`direction ∈ {mugla_2021_to_manavgat_2021, mugla_2021_to_bejis_2022}`;
20,511 + 15,190 rows per repeat. Target cohorts are full and unchanged.

**`part-target.parquet`** — 820,440 rows

```
repeat_id, direction, source_experiment_id, target_cell_id, fold_id,
burned, baseline_probability, thermal_probability,
reused_from_artifact (bool, always True), source_artifact_sha256
```
`direction ∈ {manavgat_2021_to_mugla_2021, bejis_2022_to_mugla_2021}`;
20,511 rows per (repeat, direction), copied from the frozen artifact so the
metrics are recomputable without re-reading `outputs/cross_region/`.

## 11. `repeat_metrics.csv`

**600 rows** = (1 within + 2 source + 2 target directions) × 2 families ×
3 metrics × 20 repeats.

| Column | Type | Notes |
|---|---|---|
| `arm` | str | `within` / `source` / `target` |
| `direction` | str | |
| `model_family` | str | |
| `metric` | str | |
| `repeat_id` | int16 | |
| `full_reference_value` | float | joined from `reference_metrics.csv` |
| `subsample_value` | float | natural scale; Brier stays lower-is-better |
| `natural_delta` | float | `subsample_value - full_reference_value` |
| `oriented_delta` | float | AUCs: `sub - full`; Brier: `full - sub` |
| `metric_orientation` | str | `higher_is_better` / `lower_is_better_oriented_by_negation` |
| `n_eval_rows`, `n_eval_positives` | int | |
| `n_fits_consumed` | int | 5 / 1 / 0 by arm |
| `sample_hash` | str | sha256 of the repeat's sorted `cell_id` list |

## 12. `subsampling_summary.csv`

**30 rows** = `repeat_metrics.csv` collapsed over `repeat_id`.

| Column | Type | Notes |
|---|---|---|
| `arm`, `direction`, `model_family`, `metric` | str | |
| `full_reference_value` | float | |
| `n_repeats_observed` | int | must be 20 |
| `subsample_median` | float | natural scale |
| `subsampling_interval_lower` | float | p2.5, natural scale |
| `subsampling_interval_upper` | float | p97.5, natural scale |
| `subsample_minimum`, `subsample_maximum` | float | |
| `oriented_delta_median` | float | |
| `oriented_delta_interval_lower`, `oriented_delta_interval_upper` | float | |
| `oriented_delta_minimum`, `oriented_delta_maximum` | float | |
| `reference_position` | str | one of the three position tokens |
| `interpretation_sentence` | str | one of the two permitted sentences, verbatim |

Column names `ci_lower` / `ci_upper` / `p_value` are forbidden and scanned for.

## 13. `summary.json`

```jsonc
{
  "schema_version": "mugla_subsampling.v1",
  "analysis_id": "...",
  "arm_id": "size_matched_to_manavgat",
  "target_sample_size": 20511,
  "n_repeats": 20,
  "population_accounting": { "mugla_full": {...}, "mugla_subsample": {...},
                             "manavgat_full": {...}, "bejis_full": {...} },
  "allocation_accounting": {
    "n_blocks": 576, "n_strata": 636, "n_label1_strata": 70,
    "floor_sum": 20211, "shortfall": 300,
    "strata_above_cut": 295, "strata_tied_at_cut": 12, "tie_units_awarded": 5,
    "allocation_sum": 20511, "strata_over_capacity": 0, "strata_zero_allocation": 0,
    "positives_allocated": 1438, "negatives_allocated": 19073,
    "prevalence_full": 0.069757968, "prevalence_subsample": 0.070108722,
    "prevalence_absolute_drift": 0.00035075, "prevalence_drift_bound": 0.003413,
    "prevalence_within_bound": true
  },
  "fold_accounting": {
    "fold_source": "persisted_artifact", "fold_count": 5,
    "blocks_spanning_folds": 0, "reoptimised_per_repeat": false,
    "per_fold_subsample_rows": [4111, 4096, 4107, 4096, 4101],
    "per_fold_subsample_positives": [293, 280, 295, 281, 289],
    "identical_across_repeats": true
  },
  "fit_accounting": {
    "expected": {"within": 200, "source": 40, "target": 0, "total": 240},
    "observed": {"within": 200, "source": 40, "target": 0, "total": 240},
    "reuse_events": 40,
    "contract_upper_bound": 284,
    "reductions": [
      "source arm: fit identity excludes target_id (source model is target-independent) — 80 -> 40",
      "target arm: frozen per-cell raw-transfer predictions reused — 4 -> 0"
    ]
  },
  "headline": [ /* one entry per summary row, primary metric first */ ],
  "three_arm_reading": {
    "within_region_moves": "...", "source_transfer_moves": "...",
    "target_ordering_preserved": "..."
  },
  "limitations": [ /* the five standing limitations of SCIENTIFIC_CONTRACT.md §7 */ ],
  "forbidden_language_scan": {"scanned_files": [...], "hits": 0},
  "no_gee": true,
  "canonical_outputs_unchanged": true
}
```

## 14. `report.md`

Human-readable. Sections: question and scope; frozen sampling contract;
allocation and prevalence; fold contract; the three arms with their reference
values; the summary table; the three-arm joint reading; limitations. It may use
only the two permitted interpretation sentences and the three position tokens.

## 15. `stages/*.json` and `manifest.json`

Stage markers follow `few_shot_recovery.write_stage_marker` (line 920):

```jsonc
{
  "stage": "fit", "analysis_id": "...", "schema_version": "mugla_subsampling.v1",
  "completed_at": "...", "git_commit": "...",
  "files": [{"path": "oof_predictions/part-within.parquet", "bytes": 0, "sha256": "..."}],
  "requires": ["plan"], "requires_satisfied": true
}
```

`manifest.json` lists every produced file with size and sha256, plus the
`config.json` digest, `input_hashes.json` digest, package versions, git commit,
and the fit accounting. It is the citable record.

## 16. Resume and quarantine

`--resume` re-reads each stage marker, re-hashes the files it names, and skips
a stage only when every file matches. A partition present on disk but absent
from or mismatched against the marker is moved to
`_quarantine/<timestamp>/` and recomputed. Nothing outside the analysis root is
ever read for write purposes, and nothing outside it is ever written.
