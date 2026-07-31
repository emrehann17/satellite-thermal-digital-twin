# Repository Inventory — exact files and symbols

Read in this repository on **2026-08-03**. Line numbers are as of commit
`19d825b` plus the uncommitted working tree.

Legend: **REUSE** = imported and called unchanged · **PATTERN** = followed as a
template, new code · **REFERENCE** = read as a frozen input or cross-check ·
**NOT USED** = deliberately excluded, with reason.

---

## 1. CORAL and the adaptation stack

`core/step10_shared.py` — the scientific core; nothing here is modified.

| Symbol | Line | Use | Supplies |
|---|---:|---|---|
| `EPSILON_STD` | 52 | **REUSE** | `1e-12` constant-feature guard for the z-score |
| `MODEL_NAME` | 53 | **REUSE** | `"random_forest"` |
| `MODEL_FAMILIES` | 54 | **REUSE** | `("baseline", "thermal")` |
| `ADAPTATION_METHODS` | 55 | **REFERENCE** | `("raw_source_only", "regionwise_zscore", "coral_after_regionwise_zscore")` — the reference method names |
| `PRIMARY_POPULATION` | 58 | **REUSE** | `"burnable_tree_shrub_grass"` |
| `FEATURE_LISTS` | 60 | **REUSE** | baseline 4 / thermal 10 |
| `NUMERIC_FEATURE_POOL` | 61 | **REFERENCE** | thermal numerics; the per-family numeric list is derived with the same rule |
| `Step10Error` | 64 | **REUSE** | caught per cell and mapped to `numerical_status` |
| `check_no_forbidden_features` | 68 | **REUSE** | leakage guard |
| `step10_output_dir(source_id, target_id)` | 80 | **REUSE** | **the direction resolver** — `cross_region/{source}__{target}/step10` |
| `sha256_file` | 89 | **REUSE** | reference hashing |
| `canonical_json` | 114 | **REUSE** | stable serialisation for the analysis id |
| `compute_analysis_id` | 118 | **REUSE** | `analysis_id` from the frozen config |
| `git_commit_if_available` | 122 | **REUSE** | provenance |
| `package_versions` | 135 | **REUSE** | numpy/pandas/sklearn into the manifest |
| `compute_regionwise_zscore_stats(X, numeric_features)` | 145 | **REUSE** | label-blind by signature; mean, `std(ddof=0)`, constant guard, counts |
| `apply_regionwise_zscore(X, stats, numeric_features)` | 163 | **REUSE** | region-own-mean fill **then** standardise; categorical untouched |
| `_sym_matrix_power(M, power, eps=1e-12)` | 183 | **REUSE** | `eigh` → clip at 1e-12 → power; the floor is instrumented, not changed |
| **`fit_coral_alignment(Xs_z, Xt_z, lambda_)`** | **186** | **REUSE, λ passed explicitly** | the only place λ enters; returns `A`, `Cs`, `Ct`, both condition numbers, `eigenvalue_floor_used`, `lambda` |
| **`apply_coral(Xs_z, coral_fit)`** | **213** | **REUSE** | `Xs_z @ A`, finiteness guard, `np.real` |
| `compute_threshold_free_metrics(y_true, y_prob)` | 227 | **REUSE for ROC/PR** | returns `roc_auc`, `pr_auc`, counts. **No Brier** — see §5 |
| `assert_label_blind(df, context)` | 243 | **REUSE** | target-label firewall |
| `run_n_way_paired_bootstrap(df, block_col, y_col, prob_columns, n_replicates, random_state)` | 267 | **REUSE for the draws** | `default_rng(42)`, one `rng.choice` per replicate, all series scored on the same blocks, invalids counted with no retry. Computes ROC/PR only — Brier needs an extension, see §5 |
| `percentile_ci(values)` | 305 | **REUSE** | 2.5 / 97.5 percentiles |
| `is_bootstrap_unstable(n_valid)` | 314 | **REUSE** | `< STEP10_MIN_VALID_BOOTSTRAP_REPLICATES` |
| `resolve_step8b_predictions_path` / `resolve_step8b_metrics_path` | 71 / 76 | **NOT USED** | those feed the within-region ceiling series, which this analysis does not report |
| `resolve_step9b_metrics_path` / `resolve_step9b_predictions_path` | 81 / 85 | **REFERENCE** | cross-check that the resolved `raw_source_only` values match Step9B (they do) |

`core/config.py`

| Constant | Line | Value | Use |
|---|---:|---|---|
| `STEP10_RANDOM_STATE` | 693 | `42` | **REUSE** — estimator seed and bootstrap seed |
| `STEP10_BOOTSTRAP_REPLICATES` | 694 | `1000` | **REUSE** |
| `STEP10_BOOTSTRAP_CI_LOWER_PERCENTILE` | 695 | `2.5` | **REUSE** |
| `STEP10_BOOTSTRAP_CI_UPPER_PERCENTILE` | 696 | `97.5` | **REUSE** |
| **`STEP10_CORAL_LAMBDA`** | **697** | **`1e-5`** | **REFERENCE ONLY — never mutated.** The canonical anchor of the grid |
| `STEP10_MIN_VALID_BOOTSTRAP_REPLICATES` | 698 | `900` | **REUSE** |
| `STEP8B_MIN_POSITIVES_PER_POPULATION` | 557 | `30` | **REUSE** — source population precondition |

## 2. The Step10 stage modules

`src/step10b_label_blind_adaptation.py`

| Symbol | Line | Use |
|---|---:|---|
| `generate_predictions_for_direction(...)` | 95 | **PATTERN** — the canonical order of operations reproduced exactly at λ=1e-5 for the Tier-2 gate. Not called directly: it hard-codes all three methods and the default λ, whereas this analysis needs one method at nine λ values |
| the CORAL block | 150–159 | **PATTERN** — the seven-step sequence transcribed in `CORAL_FORMULA_AUDIT.md` §5 |
| `_numeric_features(feature_list)` | — | **REUSE** — `[f for f in feature_list if f not in CATEGORICAL_FEATURES]` |
| `_prediction_frame(...)` | — | **PATTERN** — the 10-column prediction row shape |
| `run_step10b(...)` | 193 | **NOT USED** — writes into `outputs/cross_region/**`, which this analysis must never touch |

`src/step10a_preregistration_and_audit.py`

| Symbol | Line | Use |
|---|---:|---|
| `STEP10_CORAL_LAMBDA` import | 39 | **REFERENCE** — proof that mutating the constant would contaminate a preregistration |
| the CORAL method block | 162–170 | **REFERENCE** — `"lambda": STEP10_CORAL_LAMBDA` and the prose formula `Xs_coral = Xs_z @ A, Xt_coral = Xt_z` |
| `run_step10a(...)` | — | **NOT USED** — writes a preregistration into the Step10 namespace |

`src/step10c_paired_evaluation_bootstrap.py`

| Symbol | Line | Use |
|---|---:|---|
| `build_aligned_direction_frame(...)` | 77 | **PATTERN** — pivots predictions to wide, joins the **target label for the first time**, asserts cell-id uniqueness and `spatial_block_id` agreement between predictions and Step8A |
| `compute_point_metrics(merged)` | 159 | **PATTERN** |
| `run_bootstrap_for_direction(merged, n, seed)` | 434 | **PATTERN** — builds `prob_columns` and calls `run_n_way_paired_bootstrap` **once** per direction; this is the call shape that must be extended with the λ series rather than duplicated |
| the paired-delta block | 445–452 | **PATTERN** — replicate-level deltas as plain column subtraction |
| `summarize_bootstrap(replicates_df)` | 457 | **PATTERN** |
| `verify_within_region_reproduction` / `verify_raw_reproduction` | 170 / 344 | **PATTERN** — the shape of a reproduction gate that fails closed |
| `resolve_step9_raw_reference(...)` | 293 | **REFERENCE** — the existing precedent for resolving one canonical reference among several candidate files |
| `_series_col(method, model_family)` | 70 | **REUSE** — the `{method}_{family}` series naming that the bootstrap column names are built from |

`src/step10d_final_report.py`

| Symbol | Line | Use |
|---|---:|---|
| the contrast table | 256–257 | **REFERENCE** — `coral_minus_raw`, `coral_minus_zscore` naming reused for the λ contrasts |
| `coral_lambda` in the diagnostics table | 324 | **REFERENCE** — precedent for surfacing λ and both condition numbers in a report table |
| everything else | — | **NOT USED** — Step10 reporting surface |

## 3. Shared Step8/Step9 contract

| Source | Symbol | Line | Use |
|---|---|---:|---|
| `src/step8b_train_baseline_vs_thermal_model.py` | `build_pipeline(feature_list, model_name, random_state)` | 449 | **REUSE** — the only model constructor |
| | `build_classifier` | 420 | **REUSE** (indirect) — RF(300, min_samples_leaf=3, class_weight balanced, `n_jobs=-1`, seed 42) |
| | `add_spatial_block_id(df, 2)` | 277 | **REUSE** — the 2-cell `spatial_block_id` the bootstrap blocks on |
| | `compute_binary_metrics` | 480 | **REUSE for Brier** — supplies `brier_score`, which the Step10 helper does not |
| `src/step9a_audit_cross_region_inputs.py` | `SHARED_BASELINE_FEATURES` | 64 | **REUSE** |
| | `SHARED_THERMAL_FEATURES` | 70 | **REUSE** |
| | `SHARED_THERMAL_MODEL_FEATURES` | 78 | **REUSE** |
| | `CATEGORICAL_FEATURES` | — | **REUSE** — `["landcover_dominant"]` |
| | `PRIMARY_POPULATIONS` | 128 | **REUSE** |
| | `cross_region_output_root(source, target)` | 136 | **REUSE, read-only** — note it does **not** normalise pair order (see `REFERENCE_ARTIFACTS.md` §2) |
| | `resolve_step8a_dataset_path` | 140 | **REUSE** |
| | `resolve_step8a_stats_path` | 171 | **REUSE** |
| `src/step9b_run_cross_region_transfer.py` | `population_subset(df, population)` | 107 | **REUSE** — the single definition of the primary population |

## 4. Metric helpers — where each metric comes from

| Metric | Function | Origin | Note |
|---|---|---|---|
| ROC-AUC | `roc_auc_score` | via `compute_threshold_free_metrics` (`step10_shared.py:227`) | matches the stored Step10 values exactly |
| PR-AUC | `average_precision_score` | same | same |
| **Brier** | `brier_score_loss` | via `step8b.compute_binary_metrics` (`step8b:480`) | **not available from any Step10 helper** — see §5 |

## 5. The two extension points

Everything else is imported unchanged. Exactly two things the existing code
cannot do, both consequences of Step10 being ROC/PR-only:

1. **`compute_threshold_free_metrics` has no Brier.** The sensitivity run pairs
   it with `step8b.compute_binary_metrics` for the Brier column. The two are
   applied to the identical `(y_true, y_prob)` arrays, so no metric is
   recomputed twice by different code paths.
2. **`run_n_way_paired_bootstrap` scores only ROC and PR** (`step10_shared.py:294–296`).
   A Brier bootstrap therefore needs a new scoring loop. The **draw scheme must
   not be reimplemented**: the design requires the new loop to consume the
   *same* `rng.choice` sequence — i.e. one generator, one pass, all series
   (λ grid + raw + z-score) scored inside a single replicate — so that the
   paired-draw guarantee of `SCIENTIFIC_CONTRACT.md` §8 is structural rather
   than hoped for. The cleanest implementation is a local generalisation of
   `run_n_way_paired_bootstrap` that takes a metric callable set; it must be
   verified against the existing function to produce identical ROC/PR
   replicates for the canonical series.

## 6. Output, provenance, atomic-write, resume and quarantine patterns

`src/mugla_subsampling.py` is the closest structural precedent in the
repository (staged, hash-gated, partitioned logical dataset, forbidden-token
scan, fit registry, quarantine-not-delete) and is the primary **PATTERN**
source: `SCHEMA_VERSION` / `DIAGNOSTIC_NAMESPACE` / `DIAGNOSTIC_CLASS`,
`STAGES` / `STAGE_REQUIRES` / `STAGE_OUTPUTS`, `planned_output_layout`,
`validate_stage_range`, `assert_inside_namespace`, `_atomic_write_text` /
`_atomic_write_parquet`, `sha256_file` / `sha256_path`,
`build_frozen_input_inventory`, `assert_canonical_step8a_hashes`,
`write_stage_marker` / `read_stage_marker` / `verify_stage_complete`,
`verify_arm_partition`, `quarantine_namespace`, `scan_forbidden_tokens`,
`build_manifest` with `deferred_files`, and `FitRegistry`.

`src/few_shot_recovery.py` supplies the same patterns one generation earlier
and is the fallback reference.

## 7. CLI / orchestrator patterns

| File | Line | Pattern |
|---|---:|---|
| `scripts/run_mugla_subsampling.py` | — | thin dispatcher: `build_parser()` → `main()` → `run_analysis()`, no scientific logic |
| `scripts/validate_mugla_subsampling.py` | 44 (`Report`), 1160 (`run_validation`) | `{check_id, status, expected, observed, evidence_path, note}`; any FAIL ⇒ exit 1; `--dry-run` / `--deep` modes; excludes its own `validation_report.json` from the artifact scans |
| `core/pipeline_orchestrator.py::run_mugla_subsampling_stage` | 819 | stage-dispatch signature and docstring shape |
| `scripts/main.py` | 507, 514, 1387 | subcommand registration surface |
| `scripts/run_step10_self_calibrated_transfer.py` | 79 | precedent for logging the CORAL λ at run start |

## 8. Test fixture patterns

`tests/test_mugla_subsampling.py` is the model: synthetic Step8A-shaped
fixtures, `tmp_path`-injected `output_root`/`experiments_root`, adversarial
`*_fails_closed` tests for every guard, invariance tests, exact-count tests,
forbidden-token tests, and helper-level unit tests for identity computations.
`tests/test_step10.py` supplies Step10-specific fixtures and the label-firewall
assertions.

## 9. Reuse summary

**Imported and called unchanged (24 symbols):** `fit_coral_alignment`,
`apply_coral`, `_sym_matrix_power`, `compute_regionwise_zscore_stats`,
`apply_regionwise_zscore`, `compute_threshold_free_metrics`,
`compute_binary_metrics`, `assert_label_blind`, `run_n_way_paired_bootstrap`
(for the canonical ROC/PR series), `percentile_ci`, `is_bootstrap_unstable`,
`build_pipeline`, `build_classifier` (indirect), `add_spatial_block_id`,
`population_subset`, `check_no_forbidden_features`, `step10_output_dir`,
`cross_region_output_root`, `resolve_step8a_dataset_path`,
`resolve_step8a_stats_path`, `sha256_file`, `canonical_json`,
`compute_analysis_id`, `package_versions`.

**Constants reused (14):** `EPSILON_STD`, `MODEL_NAME`, `MODEL_FAMILIES`,
`PRIMARY_POPULATION`, `FEATURE_LISTS`, `CATEGORICAL_FEATURES`,
`SHARED_*_FEATURES`, `STEP10_RANDOM_STATE`, `STEP10_BOOTSTRAP_REPLICATES`,
`STEP10_BOOTSTRAP_CI_{LOWER,UPPER}_PERCENTILE`,
`STEP10_MIN_VALID_BOOTSTRAP_REPLICATES`, `STEP8B_MIN_POSITIVES_PER_POPULATION`,
`ADAPTATION_METHODS`. Plus `STEP10_CORAL_LAMBDA` **read but never written**.

**New logic required (only this):** the λ grid and its tokens, the per-cell
`numerical_status` mapping around the two existing `Step10Error` raises, the
Brier column and the Brier-capable bootstrap loop (§5), the two-tier
reproduction gate, the sensitivity summarisation with its predeclared
thresholds, and the schema plus validator surface. **No new model, no new
preprocessing, no new CORAL formula, no new interpolation semantics, no new
bootstrap draw scheme.**
