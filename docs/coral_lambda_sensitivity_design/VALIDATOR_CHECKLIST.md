# Validator Checklist — `scripts/validate_coral_lambda_sensitivity.py`

An independent executable, **not** a stage. It re-derives every claim from the
emitted artifacts plus the frozen canonical inputs, and never imports a cached
intermediate from the run. Each check emits

```json
{"check_id": "...", "status": "PASS|FAIL|SKIP", "expected": ..., "observed": ...,
 "evidence_path": "...", "note": "..."}
```

Any `FAIL` ⇒ overall `FAIL` ⇒ exit code 1. `--dry-run` runs the contract and
frozen-input checks only. `--deep` additionally re-derives one CORAL transform
and one bootstrap draw sequence.

**Two tiers of check.** *Structural* checks always run and are recomputed from
the artifact's own declared config. *Canonical literal* checks (the Step8A
digests, the resolved Step10 digests, the reference metric values) are asserted
against the frozen values in this design; they cannot be skipped, because there
is no non-production mode for this analysis — it is defined entirely on frozen
production artifacts.

The validator must **exclude its own `validation_report.json`** from every
namespace scan (it echoes the forbidden-token denylist and the excluded-AOI
names) and from the manifest-completeness check.

---

## A. Schema and identity

| # | check_id | Expected |
|---|---|---|
| A1 | `schema_version` | `"coral_lambda_sensitivity.v1"` in `config.json`, `summary.json`, `manifest.json` and all four stage markers |
| A2 | `diagnostic_class` | `"coral_regularisation_parameter_sensitivity"` |
| A3 | `analysis_id_deterministic` | `compute_analysis_id(config["scientific_config"])` equals the directory name |
| A4 | `stages_present` | `plan`, `fit`, `bootstrap`, `summarize` markers all present with `status: pass` and satisfied prerequisites |
| A5 | `stage_file_hashes` | every file named in a marker re-hashes to the recorded sha256 |
| A6 | `manifest_complete` | every namespace file appears in `manifest.json` and re-hashes; `predictions.parquet` is exposed as ONE logical dataset with 72 partitions, never as loose files; the only permitted omission is the declared `deferred_files` entry `stages/summarize.json` and the validator's own report |
| A7 | `no_hash_drift` | re-hashing every manifest entry reproduces the recorded digest |

## B. Scope and frozen inputs

| # | check_id | Expected |
|---|---|---|
| B1 | `step8a_hash_manavgat` | `054a1961fc0582a33d36413263668b63074b21ae8b03d12269b6e228787f3439` |
| B2 | `step8a_hash_bejis` | `3dec785a7d8e31db2d67ed283546bbfbca1559f56df46663488d0afc24d9e393` |
| B3 | `step8a_hash_mugla` | `c4ab107db2207f9f20775ccc0b3bf39381173fd07d4e82f6821ce7f40be7db8e` |
| B4 | `hash_gate_strict` | `input_hashes.json.hash_gate == "strict"` |
| B5 | `four_directions_exact` | exactly `{bejis→mugla, mugla→bejis, manavgat→mugla, mugla→manavgat}` in `metrics.csv`, `bootstrap_summary.csv`, `sensitivity_summary.csv` and the prediction partitions — no more, no fewer |
| B6 | `manavgat_bejis_absent_from_results` | `manavgat_2021_to_bejis_2022` and its reverse appear in **no** result row; they may appear only inside `repository_inventory.json` under `contextual_only: true` |
| B7 | `no_evia_participates` | no excluded AOI in `config.experiments` or any direction token. Declared exception: `config.json` may name them as keys of `excluded_experiments` — that is the exclusion record |
| B8 | `no_evia_elsewhere` | the substrings `evia`, `kozan` appear in no other emitted file |
| B9 | `population_counts` | recomputed with `population_subset`: muğla 41,730 / 2,911; bejís 15,190 / 1,100; manavgat 20,511 / 784 |
| B10 | `step10_references_hash_bound` | every resolved Step10 file re-hashes to its `input_hashes.json` value |
| B11 | `direction_resolution_rule` | for direction `S→T` the recorded `pair_directory` is exactly `{S}__{T}`; the rejected duplicate is recorded with its digest |

## C. λ grid

| # | check_id | Expected |
|---|---|---|
| C1 | `nine_lambda_values` | `lambda_grid.csv` has exactly 9 rows; `metrics.csv` has exactly 9 distinct `lambda_value` per direction × family × metric |
| C2 | `lambda_values_exact` | `{0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1}` compared exactly, not approximately |
| C3 | `lambda_ordering_ascending` | `lambda_index` 0…8 is numeric ascending; `lambda_grid.csv` row order matches |
| C4 | `canonical_lambda_present` | `1e-5` is in the grid, `is_canonical` true for exactly one row, `lambda_index == 4` |
| C5 | `lambda_zero_not_replaced` | the row with `lambda_index == 0` has `lambda_value == 0.0` **exactly** (`== 0.0`, not `< 1e-12`); no grid value equals `1e-12` or any other silent fallback; `is_unregularised` true for exactly one row |
| C6 | `lambda_tokens_exact` | the nine tokens match the frozen table; no path anywhere in the namespace contains a `.`-formatted float |
| C7 | `lambda_in_analysis_id` | perturbing any grid value and recomputing `compute_analysis_id` yields a different id (proves the grid is hashed) |
| C8 | `config_constant_not_mutated` | `core.config.STEP10_CORAL_LAMBDA == 1e-5` at validation time; `config.lambda_semantics.config_constant_mutated == false` |

## D. CORAL formula fidelity

| # | check_id | Expected |
|---|---|---|
| D1 | `coral_symbols_reused` | `config.coral.fit_symbol == "core.step10_shared.fit_coral_alignment"` and `apply_symbol == "core.step10_shared.apply_coral"`; both resolve and are the unmodified functions |
| D2 | `covariance_convention` | `numpy.cov`, `rowvar=False`, `ddof=0` recorded and matching `step10_shared.py:192–193` |
| D3 | `lambda_is_additive_ridge` | `config.lambda_semantics.definition` states the additive ridge on **both** covariances; the four `is_*` flags are all `false` |
| D4 | `no_new_interpolation_semantics` | no field named `alpha`, `blend`, `mix`, `interp`, `shrinkage_coefficient`, `strength` anywhere in `config.json`, `metrics.csv` or `sensitivity_summary.csv` |
| D5 | `target_never_transformed` | `config.coral.target_transformed == false`; `predict_representation` is the z-scored target |
| D6 | `order_of_operations` | `"regionwise_zscore -> coral -> model_fit"` |
| D7 | `numeric_feature_order` | matches `[f for f in FEATURE_LISTS[family] if f not in CATEGORICAL_FEATURES]`; d = 3 (baseline) and 9 (thermal) |
| D8 | `eigenvalue_floor_declared` | `eigenvalue_floor: 1e-12` recorded and distinguished from λ |
| D9 | **`coral_transform_recomputable`** (deep) | for one (direction, family, λ) cell, re-deriving `Cs`, `Ct`, `A` from the canonical frames reproduces `adaptation_statistics.parquet`'s `condition_number_Cs/Ct` and `coral_A_frobenius_norm` to ≤ 1e-9 |
| D10 | `zscore_stats_lambda_invariant` | within a (direction, family), the source and target z-score statistics are identical across all nine λ rows |

## E. Target-label firewall

| # | check_id | Expected |
|---|---|---|
| E1 | `predictions_label_blind` | `predictions.parquet` has no `burned` column in any of the 72 partitions |
| E2 | `adaptation_statistics_label_blind` | no label-derived field in `adaptation_statistics.parquet` |
| E3 | `zscore_stats_label_independent` | recomputing the z-score statistics from X alone reproduces the stored values exactly |
| E4 | `lambda_not_label_selected` | the grid is the frozen constant list, identical for every direction and family; no field records a per-direction chosen λ |
| E5 | `labels_only_at_evaluation` | labels enter only `metrics.csv` and `bootstrap_replicates.parquet` derivations; `canonical_reproduction.csv` records `labels_exact` |

## F. Reproduction gate

| # | check_id | Expected |
|---|---|---|
| F1 | `gate_present` | `canonical_reproduction.csv` has 48 rows (4 × 2 × 3 × 2 tiers) |
| F2 | `gate_status_pass` | every row `gate_status == pass`; the `plan` marker records the gate as passed |
| F3 | `tier1_exact` | tier-1 `absolute_deviation ≤ 1e-12` for all ROC-AUC and PR-AUC rows; re-running the tier-1 recomputation from the resolved artifacts reproduces the stored deviations |
| F4 | `tier1_reproduces_step10_metrics` | tier-1 `stored_value` equals the resolved `step10_metrics.csv` value |
| F5 | `tier2_probability_tolerance` | `max_abs_probability_deviation ≤ 1e-12` for all 8 tier-2 (direction, family) cells |
| F6 | `tier2_metric_tolerance` | ROC/PR `absolute_deviation ≤ 1e-06` **and** `≤ 8 × rank_quantum` for that direction; Brier `≤ 1e-09` |
| F7 | `rank_quantum_correct` | `rank_quantum == 1/(n_pos·n_neg)` recomputed per direction: 8.849e-09 (muğla), 6.452e-08 (bejís), 6.466e-08 (manavgat) |
| F8 | `coverage_and_labels_exact` | `cell_coverage_exact` and `labels_exact` true in every row, with no tolerance |
| F9 | `tolerance_rationale_recorded` | `config.reproduction_gate.rationale` records the measured 4.867e-08 canonical-pipeline drift |
| F10 | `audit_fits_separate` | `audit_fit_count` totals 8 and is **not** included in the 72 |

## G. Fit accounting

| # | check_id | Expected |
|---|---|---|
| G1 | `scientific_fits_exactly_72` | `summary.json.fit_accounting.scientific_fits == 72`; the `fit` marker agrees |
| G2 | `fit_identity_complete` | the 72 identities are exactly `(direction, model_family, lambda_token)` over 4 × 2 × 9, no duplicates, none missing |
| G3 | `no_transform_reuse_across_directions` | no fit or CORAL identity is shared between two directions, even where the source region is the same — `A` depends on the target covariance |
| G4 | `no_fit_in_bootstrap` | the `bootstrap` marker records `model_refit: false` and contributes 0 fits |
| G5 | `partitions_complete` | `predictions.parquet` has exactly 72 leaf partitions; every (direction, family, λ) present |
| G6 | `no_silent_skip` | `metrics.csv` has exactly 216 rows; `numerical_diagnostics.csv` exactly 72; `adaptation_statistics.parquet` exactly 72 |

## H. Prediction coverage and metric arithmetic

| # | check_id | Expected |
|---|---|---|
| H1 | `prediction_coverage_exact` | each partition's `target_cell_id` set equals the canonical target primary population — 41,730 / 15,190 / 41,730 / 20,511 by direction; zero duplicates |
| H2 | `probabilities_finite` | no NaN/inf in any partition whose `numerical_status == pass` |
| H3 | `metrics_recomputable` | recomputing ROC/PR/Brier from each partition reproduces `metrics.csv` to ≤ 1e-12 |
| H4 | `auc_delta_arithmetic` | for `roc_auc`/`pr_auc`: `delta_vs_X == metric_value − X_reference_value` to ≤ 1e-12 |
| H5 | `brier_orientation` | for `brier_score`: `delta_vs_X == X_reference_value − metric_value`; and `natural_delta_vs_canonical_lambda == metric_value − canonical_reference` |
| H6 | `brier_natural_scale_stored` | every Brier `*_value` field is in (0, 1) and lower-is-better; no negated Brier is stored in a value field |
| H7 | `orientation_labels` | `metric_orientation` is `lower_is_better_oriented_by_negation` for Brier and `higher_is_better` for the AUCs, in all 216 rows |
| H8 | `reference_source_flags` | ROC/PR reference rows flagged `read_from_step10_metrics_csv`; **all** Brier reference rows flagged `recomputed_from_persisted_probabilities` |
| H9 | `canonical_lambda_row_matches_reference` | at λ=1e-5, `delta_vs_canonical_lambda == 0` within the tier-2 tolerance for every direction × family × metric |
| H10 | `references_match_resolved_artifacts` | `raw_reference_value` and `zscore_reference_value` equal the resolved `step10_metrics.csv` values exactly |

## I. Bootstrap

| # | check_id | Expected |
|---|---|---|
| I1 | `replicates_1000` | 1000 requested per direction; `bootstrap_replicates.parquet` has 4,000 rows |
| I2 | `seed_and_block_column` | seed 42, `spatial_block_id`, recorded in config and the marker |
| I3 | **`same_block_draws_across_lambda`** | within a (direction, replicate), every λ series and every reference series was scored on the identical resampled index set. Verified structurally: the marker records `single_call_per_direction: true`, and (deep) re-deriving `np.random.default_rng(42)` over the frozen block order reproduces the recorded `n_blocks_drawn` sequence |
| I4 | `paired_deltas_are_row_arithmetic` | every `delta_*` replicate column equals the difference of its two series columns in the same row, to ≤ 1e-12 |
| I5 | `invalid_replicate_accounting_truthful` | `n_valid + n_invalid == 1000` per direction; `n_valid` equals the number of rows with `valid == true`; no retry occurred |
| I6 | `bootstrap_unstable_flag` | `bootstrap_unstable == (n_valid < 900)` |
| I7 | `percentiles_recomputable` | `percentile_2_5` / `percentile_97_5` recomputed from the replicate columns match to ≤ 1e-12 |
| I8 | `support_token_correct` | `bootstrap_supported_positive` iff `2.5th > 0`; `bootstrap_supported_negative` iff `97.5th < 0`; `interval_includes_zero` otherwise |
| I9 | `no_model_refit_in_bootstrap` | the bootstrap stage records zero fits |

## J. Numerical diagnostics

| # | check_id | Expected |
|---|---|---|
| J1 | `diagnostics_complete` | 72 rows, one per (direction, family, λ), all required fields non-null |
| J2 | `status_vocabulary` | `numerical_status` ∈ the five allowed values only |
| J3 | `failures_retained` | any row with a failure status is **present** with NA metrics; no (direction, family, λ) cell is missing from `metrics.csv` because it failed |
| J4 | `eigenvalue_floor_instrumented` | `eigenvalue_floor_bound_Cs` / `_Ct` recorded for all 72 rows; when both are false the λ=0 row is reported as genuinely unregularised |
| J5 | `pre_ridge_eigenvalues_recomputable` | (deep) re-deriving `min_eigenvalue_Cs_before_ridge` from the canonical frames reproduces the stored value to ≤ 1e-9; expected minimum over the study ≈ 1.713164e-03 |
| J6 | `post_ridge_monotonic` | within a (direction, family), `min_eigenvalue_*_after_ridge` is non-decreasing in λ and equals `before_ridge + λ` to ≤ 1e-9 |
| J7 | `condition_number_monotonic` | within a (direction, family), the post-ridge condition number is non-increasing in λ |

## K. Interpretation and language

| # | check_id | Expected |
|---|---|---|
| K1 | `no_lambda_selection` | no column, key or sentence named `best_lambda`, `optimal_lambda`, `selected_lambda`, `recommended_lambda`, `argmax*`, `chosen_lambda` anywhere; `summary.json.lambda_selection_performed == false` |
| K2 | `no_forbidden_vocabulary` | case-insensitive scan for `statistically significant`, `significance`, `significant`, `p-value`, `p_value`, `pvalue`, `proven`, `optimal`, `best lambda`, `confidence interval`, `95% ci`, `ci_lower`, `ci_upper`, `istatistiksel olarak anlaml` → **0 hits** (validator's own report excluded) |
| K3 | `interval_wording` | only `bootstrap_supported_positive`, `bootstrap_supported_negative`, `interval_includes_zero` appear as support tokens |
| K4 | `magnitude_token_vocabulary` | `magnitude_token` ∈ {`insensitive_over_grid`, `modest_lambda_sensitivity`, `material_lambda_sensitivity`}; `instability_token` ∈ {`numerical_instability_present`, null} |
| K5 | `thresholds_as_declared` | AUC thresholds 0.005 / 0.020; Brier ratio thresholds 0.005 / 0.020 with `deviation_scale == p(1−p)` of the **target**, using p = 0.06975797 / 0.07241606 / 0.03822339 |
| K6 | `magnitude_token_recomputable` | recomputing the token from `max_abs_deviation_from_canonical`, `deviation_scale` and the declared thresholds reproduces the emitted token for all 24 rows |
| K7 | `instability_token_correct` | `numerical_instability_present` iff `n_numerical_failures > 0`, and is additive to the magnitude token rather than replacing it |
| K8 | `canonical_rank_correct` | `canonical_rank_within_finite_grid` recomputed over the finite cells matches |
| K9 | `summary_row_count` | `sensitivity_summary.csv` has exactly 24 rows |

## L. Process hygiene and containment

| # | check_id | Expected |
|---|---|---|
| L1 | `no_gee` | no `import ee`, `ee.Initialize`, `gee_utils` or `earthengine` reachable from the analysis module; `summary.json.earth_engine_used == false` |
| L2 | `step8a_unchanged` | the three Step8A parquets re-hash to B1–B3 |
| L3 | `step9_unchanged` | every `outputs/cross_region/*/step9b/` file referenced re-hashes to its recorded digest |
| L4 | `step10_unchanged` | all nine files in each of the four resolved Step10 directories re-hash to their `input_hashes.json` digests; the two rejected duplicate directories are unchanged too |
| L5 | `no_step10_artifact_copied` | no file in the sensitivity namespace is byte-identical to a Step10 artifact — references are by digest only |
| L6 | `preregistration_untouched` | every `step10_preregistration.json` re-hashes to its recorded digest |
| L7 | `output_containment` | every manifest path resolves under `outputs/diagnostics/coral_lambda_sensitivity/<analysis_id>/` |
| L8 | `dry_run_writes_nothing` | (dry-run mode) the namespace is not created and no file is written |
| L9 | `resume_rejects_partial_grid` | a `fit` marker missing any of the 72 partition digests reports incomplete |
| L10 | `force_quarantine_only` | `_quarantine/` entries are complete moved namespaces; no file was deleted; no Step9/Step10 path was touched |
| L11 | `atomic_writes` | no `.tmp` residue in the namespace |

---

## Summary

**91 checks across 12 groups.** Coverage of the contract's required checks:

| Contract requirement | Checks |
|---|---|
| schema exact | A1–A2 |
| analysis ID exact | A3, C7 |
| four directions exact | B5, B6 |
| Evia excluded | B7, B8 |
| 9 λ values exact | C1, C2 |
| λ ordering exact | C3 |
| canonical λ=1e-5 included | C4, C8 |
| λ=0 not silently replaced | C5 |
| canonical Step8A hashes exact | B1–B4, L2 |
| Step10 references hash-bound | B10, B11, L4 |
| exact existing CORAL formula reused | D1–D3, D9 |
| no new interpolation semantics | D4 |
| target-label firewall | E1–E5 |
| baseline and thermal complete | B5, G2, G6 |
| maximum 72 scientific fits | G1–G3 |
| canonical λ reproduction gate PASS | F1–F10 |
| prediction coverage exact | H1, H2 |
| metric arithmetic exact | H3, H4, H9, H10 |
| Brier orientation exact | H5–H8 |
| same paired block draws across λ | I3, I4 |
| 1000 bootstrap replicates | I1, I2 |
| invalid replicate accounting truthful | I5, I6 |
| numerical failures retained | J2, J3 |
| no "best λ" selection | K1 |
| no p-values / no significance wording | K2, K3 |
| no GEE | L1 |
| Step8A / Step9 / Step10 unchanged | L2–L6 |
| output only in the dedicated namespace | L7, L10 |
| manifest complete | A6 |
| no hash drift | A5, A7 |
