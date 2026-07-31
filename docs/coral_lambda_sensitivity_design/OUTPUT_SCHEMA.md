# Output Schema — `coral_lambda_sensitivity.v1`

```
SCHEMA_VERSION       = "coral_lambda_sensitivity.v1"
DIAGNOSTIC_NAMESPACE = "coral_lambda_sensitivity"
DIAGNOSTIC_CLASS     = "coral_regularisation_parameter_sensitivity"
```

**Namespace — the only writable location:**

```
outputs/diagnostics/coral_lambda_sensitivity/<analysis_id>/
```

`analysis_id = compute_analysis_id(scientific_config)`. The frozen config
includes all nine λ values, the four directions, both families, the
reproduction tolerances, the interpretation thresholds and the target
prevalences used for the Brier scale — so any contract change moves the whole
run to a new directory. Every write passes
`assert_inside_namespace(path, analysis_root)`.

---

## 1. Stages

```
STAGES = ("plan", "fit", "bootstrap", "summarize")
STAGE_REQUIRES = {
    "plan": (), "fit": ("plan",),
    "bootstrap": ("plan", "fit"), "summarize": ("plan", "fit", "bootstrap"),
}
```

The validator is a **separate executable** and is not a stage.

| Stage | Writes | Fits models |
|---|---|---|
| `plan` | `config.json`, `input_hashes.json`, `repository_inventory.json`, `lambda_grid.csv`, `canonical_reproduction.csv` | Tier-2 gate only: 8 audit fits |
| `fit` | `adaptation_statistics.parquet`, `predictions.parquet/`, `metrics.csv`, `numerical_diagnostics.csv` | yes — 72 scientific fits |
| `bootstrap` | `bootstrap_replicates.parquet`, `bootstrap_summary.csv` | **no** — rescores persisted probabilities |
| `summarize` | `sensitivity_summary.csv`, `summary.json`, `report.md`, `manifest.json` | no |

Both reproduction tiers complete in `plan`. **If the gate fails, `plan` fails
and `fit` never starts.**

## 2. File layout

```
outputs/diagnostics/coral_lambda_sensitivity/<analysis_id>/
├── config.json
├── input_hashes.json
├── repository_inventory.json
├── lambda_grid.csv
├── canonical_reproduction.csv
├── adaptation_statistics.parquet
├── predictions.parquet/                       # logical dataset
│   └── direction=<direction>/
│       └── model_family=<family>/
│           └── lambda_token=<token>/part.parquet
├── metrics.csv
├── numerical_diagnostics.csv
├── bootstrap_replicates.parquet
├── bootstrap_summary.csv
├── sensitivity_summary.csv
├── summary.json
├── report.md
├── stages/{plan,fit,bootstrap,summarize}.json
├── manifest.json
└── _quarantine/                               # only if --force moved a prior run
```

`predictions.parquet` is a **directory** holding a Hive-partitioned dataset —
72 leaf partitions (4 directions × 2 families × 9 λ). The manifest exposes it
as **one** logical dataset, never as loose parts. No float ever appears in a
path: `lambda_token` uses the frozen token table.

## 3. `config.json`

```jsonc
{
  "schema_version": "coral_lambda_sensitivity.v1",
  "analysis_id": "...",
  "diagnostic_class": "coral_regularisation_parameter_sensitivity",
  "scientific_config": {
    "population": "burnable_tree_shrub_grass",
    "valid_universe": "valid_for_modeling == True",
    "experiments": ["manavgat_2021", "bejis_2022", "mugla_2021"],
    "excluded_experiments": {"evia_2021": "...", "evia_2021_extended": "...",
                             "kozan_2023": "..."},
    "directions": {
      "primary":   ["bejis_2022_to_mugla_2021", "mugla_2021_to_bejis_2022"],
      "secondary": ["manavgat_2021_to_mugla_2021", "mugla_2021_to_manavgat_2021"]
    },
    "contextual_only_not_rerun": ["manavgat_2021_to_bejis_2022",
                                  "bejis_2022_to_manavgat_2021"],
    "model_families": ["baseline", "thermal"],
    "adaptation_method_under_study": "coral_after_regionwise_zscore",
    "reference_methods": ["raw_source_only", "regionwise_zscore"],
    "lambda_grid": [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "lambda_tokens": ["lambda_0", "lambda_1e_m8", "lambda_1e_m7", "lambda_1e_m6",
                      "lambda_1e_m5", "lambda_1e_m4", "lambda_1e_m3",
                      "lambda_1e_m2", "lambda_1e_m1"],
    "canonical_lambda": 1e-5,
    "canonical_lambda_index": 4,
    "lambda_semantics": {
      "definition": "additive ridge lambda*I added to BOTH source and target covariance",
      "source": "core/step10_shared.py:192-193",
      "is_interpolation_strength": false,
      "is_prediction_blend": false,
      "is_model_regularisation": false,
      "is_covariance_shrinkage_convex_combination": false,
      "eigenvalue_floor_separate_constant": 1e-12,
      "config_constant_mutated": false
    },
    "coral": {
      "fit_symbol": "core.step10_shared.fit_coral_alignment",
      "apply_symbol": "core.step10_shared.apply_coral",
      "covariance": {"estimator": "numpy.cov", "rowvar": false, "ddof": 0},
      "matrix_power": {"method": "numpy.linalg.eigh", "eigenvalue_floor": 1e-12},
      "order": "regionwise_zscore -> coral -> model_fit",
      "target_transformed": false,
      "fit_representation": "X_source_coral (z-scored source, numeric replaced by Xs_z@A)",
      "predict_representation": "X_target_z (z-scored target, NOT coral-transformed)",
      "numeric_feature_order": {"baseline": ["..."], "thermal": ["..."]},
      "dtype": "float64"
    },
    "model": {"name": "random_forest", "constructor": "step8b.build_pipeline",
              "seed": 42, "hyperparameters": {"...": "..."}},
    "metrics": {
      "list": ["roc_auc", "pr_auc", "brier_score"],
      "primary_estimand": "thermal_roc_auc_sensitivity_across_lambda",
      "orientation": {"roc_auc": "higher_is_better", "pr_auc": "higher_is_better",
                      "brier_score": "lower_is_better_oriented_by_negation"},
      "brier_not_in_step10": true,
      "brier_reference_source": "recomputed_from_persisted_probabilities"
    },
    "reproduction_gate": {
      "tier1_metric_tolerance": 1e-12,
      "tier2_probability_tolerance": 1e-12,
      "tier2_metric_tolerance": 1e-06,
      "tier2_rank_quantum_multiplier": 8,
      "tier2_brier_tolerance": 1e-09,
      "rationale": "two executions of the identical canonical pipeline differ by "
                   "up to 4.867e-08 in ROC-AUC (RandomForest n_jobs=-1)"
    },
    "bootstrap": {"replicates": 1000, "seed": 42, "block_column": "spatial_block_id",
                  "paired": true, "model_refit": false, "retry_on_invalid": false,
                  "percentiles": [2.5, 97.5], "single_call_per_direction": true},
    "interpretation_thresholds": {
      "auc": {"insensitive": 0.005, "modest": 0.020},
      "brier": {"scale": "p_target*(1-p_target)", "insensitive_ratio": 0.005,
                "modest_ratio": 0.020,
                "p_target": {"mugla_2021": 0.06975797, "bejis_2022": 0.07241606,
                             "manavgat_2021": 0.03822339}},
      "frozen_before_results": true
    },
    "expected_scientific_fits": 72,
    "lambda_selection_performed": false
  }
}
```

## 4. `input_hashes.json`

Three Step8A digests (with `expected_sha256`, `match`, row/class counts), then
per direction the resolved Step10 artifact block: `pair_directory`,
`resolution_rule`, and for each of the nine files its `path`, `sha256`, `bytes`;
plus `step10_analysis_id`, `canonical_lambda_in_artifact`,
`source_step8a_sha256`, `target_step8a_sha256`, `target_row_count`,
`prediction_row_count`, `direction_coverage`. Also records the **rejected**
duplicate pair directory and its digest, so the resolution is auditable. Plus
`git_commit`, `package_versions`, `hash_gate: "strict"`.

## 5. `repository_inventory.json`

Machine-readable form of `REPOSITORY_INVENTORY.md`: for every reused symbol its
module, qualified name, line number and a `role` of
`reuse | pattern | reference | not_used`; plus the `manavgat↔bejis` contextual
artifact digests behind `contextual_only: true`.

## 6. `lambda_grid.csv` — 9 rows

`lambda_index`, `lambda_value`, `lambda_token`, `is_canonical`,
`is_unregularised`, `grid_position` (`below_canonical` / `canonical` /
`above_canonical`).

## 7. `canonical_reproduction.csv` — 48 rows

4 directions × 2 families × 3 metrics × 2 tiers.

`tier` (`tier1_exact_from_persisted` / `tier2_refit`), `direction`,
`model_family`, `method`, `metric`, `stored_value`, `reproduced_value`,
`absolute_deviation`, `tolerance`, `rank_quantum` (`1/(n_pos·n_neg)`, null for
Brier), `within_tolerance`, `n_target_rows`, `cell_coverage_exact`,
`labels_exact`, `probabilities_finite`, `max_abs_probability_deviation`,
`gate_status` (`pass` / `fail`), `evidence_path`, `evidence_sha256`.

Tier-2 rows also carry `audit_fit_count` (8 in total, reported separately from
the 72).

## 8. `adaptation_statistics.parquet` — 72 rows

`direction`, `source_experiment`, `target_experiment`, `model_family`,
`lambda_value`, `lambda_token`, `numeric_feature_order` (list),
`numeric_dimension`, `n_source_rows`, `n_target_rows`,
`source_zscore_stats` (struct: per-feature mean/std/raw_std/constant_guard/
n_observed/n_missing), `target_zscore_stats` (same),
`condition_number_Cs`, `condition_number_Ct`, `eigenvalue_floor_used`,
`coral_A_frobenius_norm`, `coral_A_determinant`.

The z-score statistics are **λ-independent** and identical across the nine λ
rows of a (direction, family); they are repeated per row for self-containment
and their invariance is asserted.

## 9. `predictions.parquet/` — 2,144,898 rows across 72 partitions

Per row: `direction`, `source_experiment`, `target_experiment`, `population`,
`model_family`, `lambda_value`, `lambda_token`, `target_cell_id`,
`target_spatial_block_id`, `prediction_probability`.

`burned` is **absent by construction** — `assert_label_blind` is called on this
frame before it is written, exactly as Step10B does.

Invariants: exactly one row per `target_cell_id` within a partition; the
partition's cell-id set equals the canonical target primary population
(41,730 / 15,190 / 41,730 / 20,511 by direction); all probabilities finite,
unless the partition's `numerical_status` is a failure, in which case the
partition is written with NA probabilities and the status recorded.

## 10. `metrics.csv` — 216 rows

4 directions × 2 families × 9 λ × 3 metrics.

`direction`, `direction_tier` (`primary`/`secondary`), `model_family`,
`lambda_value`, `lambda_token`, `metric`, `metric_value`,
`raw_reference_value`, `zscore_reference_value`,
`canonical_coral_reference_value`, `delta_vs_raw`, `delta_vs_zscore`,
`delta_vs_canonical_lambda`, `natural_delta_vs_canonical_lambda`,
`metric_orientation`, `reference_source`
(`read_from_step10_metrics_csv` / `recomputed_from_persisted_probabilities`),
`n_target_rows`, `n_target_positives`, `numerical_status`.

For `roc_auc` / `pr_auc` the `delta_*` columns are already oriented
(`candidate − reference`). For `brier_score`, `delta_*` are the **oriented**
values (`reference − candidate`) and `natural_delta_vs_canonical_lambda` keeps
the natural `candidate − reference`; the `*_value` fields always hold the
natural lower-is-better Brier.

## 11. `numerical_diagnostics.csv` — 72 rows

The full field list of `SCIENTIFIC_CONTRACT.md` §10, plus `direction`,
`model_family`, `lambda_value`, `lambda_token`.

## 12. `bootstrap_replicates.parquet` — 4,000 rows

One row per (direction, replicate). Wide, following the Step10 convention
`{metric}__{series}`:

- series = `raw_source_only_{family}`, `regionwise_zscore_{family}`, and
  `coral_{lambda_token}_{family}` — 2 + 2 + 18 = 22 series;
- metrics = `roc_auc`, `pr_auc`, `brier_score` → 66 metric columns;
- paired delta columns
  `delta_{metric}__coral_{lambda_token}_minus_{raw|zscore|canonical}__{family}`.

Plus `direction`, `replicate`, `n_blocks_drawn`, `valid`.

**All series in one row come from the same block draw**, guaranteed by a single
`run_n_way_paired_bootstrap`-shaped call per direction.

## 13. `bootstrap_summary.csv`

One row per direction × family × λ × metric × contrast
(`vs_raw`, `vs_zscore`, `vs_canonical_lambda`):
`point_estimate`, `percentile_2_5`, `percentile_97_5`, `n_valid_replicates`,
`n_invalid_replicates`, `bootstrap_unstable`, `support_token`
(`bootstrap_supported_positive` / `bootstrap_supported_negative` /
`interval_includes_zero`).

## 14. `sensitivity_summary.csv` — 24 rows

4 directions × 2 families × 3 metrics — the primary result table.

`direction`, `direction_tier`, `model_family`, `metric`,
`canonical_lambda_value`, `grid_min`, `grid_max`, `grid_range`,
`max_abs_deviation_from_canonical`, `deviation_scale` (1.0 for AUCs,
`p(1−p)` for Brier), `deviation_ratio`,
`canonical_rank_within_finite_grid`, `n_finite_lambda`,
`sign_pattern_delta_vs_zscore`,
`n_lambda_interval_excludes_zero_positive`,
`n_lambda_interval_excludes_zero_negative`,
`n_lambda_interval_includes_zero`, `n_numerical_failures`,
`magnitude_token`, `instability_token`.

There is deliberately **no** `best_lambda`, `selected_lambda`, `optimal_lambda`
or `argmax_*` column, and the validator asserts their absence.

## 15. `summary.json` / `report.md`

`summary.json`: identity, the frozen grid, the reproduction gate outcome, fit
accounting (`scientific_fits: 72`, `audit_fits: 8`), numerical-status counts,
the 24 summary rows, the interpretation thresholds as declared, the
limitations, `lambda_selection_performed: false`, `earth_engine_used: false`.

`report.md`: question and scope; the exact λ semantics with file:line; the
frozen grid; the reproduction gate with its measured deviations and the reason
the metric tolerance is 1e-06; the sensitivity table; the bootstrap support
tokens; numerical diagnostics including whether the eigenvalue floor ever
bound; limitations. It may use only the permitted bootstrap wording and the
four interpretation tokens.

## 16. Stage markers and manifest

Markers follow the `mugla_subsampling` shape: `stage`, `analysis_id`,
`schema_version`, `status`, `completed_at_utc`, `git_commit`, `requires`,
per-file `sha256`, plus stage-specific extras (`plan`: gate outcome; `fit`: fit
accounting and per-partition digests; `bootstrap`: valid/invalid counts per
direction; `summarize`: token counts).

`manifest.json` lists every file with size and sha256, exposes
`predictions.parquet` as one logical dataset with its 72 partition names and a
dataset-level digest, and declares `deferred_files: ["stages/summarize.json"]`
(written after the manifest, because that marker hash-binds the manifest).

## 17. Safety

**Dry-run** — read-only: resolves inputs, verifies the three Step8A digests and
the four Step10 reference digests, prints the plan and the planned output
layout. Creates no directory, writes no file, fits no model, runs no bootstrap.

**Resume** — reuses only complete, hash-bound PASS stages. A partial λ grid is
never accepted: `fit` is complete only when all 72 partitions are present and
hash-bound by the marker. A missing direction / family / λ partition is a hard
failure, never a silent skip.

**Force** — quarantines the existing sensitivity namespace under
`_quarantine/<analysis_id>/<timestamp>/` by `shutil.move`. It deletes nothing
and it never touches `outputs/experiments/`, `outputs/cross_region/` or
`outputs/robustness/`.

All writes are atomic (temp file + `os.replace`). No Earth Engine import is
reachable from the module.
