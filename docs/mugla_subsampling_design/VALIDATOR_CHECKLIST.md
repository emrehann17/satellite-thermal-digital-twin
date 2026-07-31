# Validator Checklist — `scripts/validate_mugla_subsampling.py`

An independent executable, **not** a stage of the run. It re-derives every
claim from the emitted artifacts plus the frozen canonical inputs, and never
imports a cached intermediate from the run. Each check emits

```json
{"check_id": "...", "status": "PASS|FAIL|SKIP", "expected": ..., "observed": ...,
 "evidence_path": "...", "note": "..."}
```

Any `FAIL` ⇒ overall `FAIL` ⇒ exit code 1. `--dry-run` lists the checks it
would run without reading run outputs. Pattern source:
`scripts/validate_few_shot_recovery.py` (`class Report`, line 39;
`run_validation`, line 634).

**Two tiers of check.** *Structural* checks (the allocation arithmetic, the
invariances, the fold and arm contracts, the metric orientation) always run and
are recomputed from the artifact's own declared contract in `config.json`.
*Production inventory literals* — 20,511 / 20 repeats / 1,438 positives / the
per-fold composition / the 636-stratum tie at remainder 19,095 / the full
cohort sizes — are asserted only when `input_hashes.json` pins the real frozen
Muğla Step8A digest `c4ab107d…`; against any other frame they are reported
`SKIPPED` with the reason, because they would be meaningless there. The hash
gate (B1–B4) still FAILs on a non-canonical frame, so a non-production artifact
can never pass overall.

---

## A. Schema and identity

| # | check_id | Expected |
|---|---|---|
| A1 | `schema_version` | `"mugla_subsampling.v1"` in `config.json`, `summary.json`, `manifest.json`, all stage markers |
| A2 | `diagnostic_class` | `"population_size_matched_subsampling_sensitivity"` |
| A3 | `analysis_id_deterministic` | `compute_analysis_id(config["scientific_config"])` recomputed from `config.json` equals the directory name |
| A4 | `stage_markers_present` | `stages/plan.json`, `stages/fit.json`, `stages/summarize.json` all present, `requires_satisfied: true` |
| A5 | `stage_file_hashes` | every file named in a stage marker exists and re-hashes to the recorded sha256 |
| A6 | `manifest_complete` | every file under the analysis root appears in `manifest.json` and re-hashes correctly; no file in `manifest.json` is missing. **One declared exception:** `stages/summarize.json` is written *after* the manifest, because that marker hash-binds `manifest.json`; the manifest lists it under `deferred_files` with the reason, and the marker hashes it instead. The `oof_predictions.parquet` parts are catalogued as one logical dataset, not as loose files. |

## B. Frozen inputs

| # | check_id | Expected |
|---|---|---|
| B1 | `step8a_hash_manavgat` | `054a1961fc0582a33d36413263668b63074b21ae8b03d12269b6e228787f3439` |
| B2 | `step8a_hash_bejis` | `3dec785a7d8e31db2d67ed283546bbfbca1559f56df46663488d0afc24d9e393` |
| B3 | `step8a_hash_mugla` | `c4ab107db2207f9f20775ccc0b3bf39381173fd07d4e82f6821ce7f40be7db8e` |
| B4 | `hash_gate_strict` | `input_hashes.json.hash_gate == "strict"` |
| B5 | `only_three_primary_aois` | `config.experiments` is exactly `["manavgat_2021","bejis_2022","mugla_2021"]`, order included |
| B6 | `no_evia_participates` | no excluded AOI appears among `scientific_config.experiments` or any direction token. **Declared exception:** `config.json` names the excluded AOIs as *keys of `excluded_experiments`* — that is the provenance record of the exclusion and is required, not a leak. |
| B7 | `no_evia_outside_the_exclusion_record` | the substrings `evia`, `kozan` appear in **no** other emitted file — CSVs, parquet string columns, `summary.json`, `report.md`, stage markers |

| B8 | `only_mugla_subsampled` | `config.subsampled_experiment == "mugla_2021"`; `sampling_inventory.csv` has `subsampled == True` for Muğla only; `selected_cells.parquet` has no non-Muğla cell |
| B9 | `population_counts` | Muğla primary 41,730 / 2,911 / 38,819; Manavgat 20,511 / 784 / 19,727; Bejís 15,190 / 1,100 / 14,090 — recomputed from the Step8A parquets with `population_subset` |

## C. Sampling contract

| # | check_id | Expected |
|---|---|---|
| C1 | `repeat_count` | exactly 20 distinct `repeat_id`, values 0…19; `n_repeats_observed == 20` in every summary row |
| C2 | `rows_per_repeat` | exactly **20,511** rows for every `repeat_id` in `selected_cells.parquet` |
| C3 | `unique_within_repeat` | 0 duplicate `(repeat_id, cell_id)` — i.e. no replacement |
| C4 | `no_foreign_rows` | every `cell_id` ∈ the canonical Muğla primary `cell_id` set (recomputed from the Step8A parquet); 0 exceptions |
| C5 | `subset_of_canonical` | the union over repeats is a **proper subset** of the 41,730 (a repeat is not the whole population) |
| C6 | `label_matches_canonical` | every row's `label` equals the canonical `burned` for that `cell_id` — labels are copied, never recomputed |
| C7 | `block_matches_canonical` | every row's `large_block_id` equals `assign_large_blocks(canonical_df, 10)` for that `cell_id` |
| C8 | `repeats_differ` | at least 2 distinct `sample_hash` values across the 20 repeats (the seeds really vary) |

## D. Allocation arithmetic — recomputed from scratch

The validator recomputes the whole allocation from the canonical Muğla frame
and compares it to `stratum_allocation.csv` row by row.

| # | check_id | Expected |
|---|---|---|
| D1 | `strata_recomputable` | 636 strata, 576 blocks, 70 label-1 strata; `stratum_id` set identical |
| D2 | `capacity_recomputable` | every `capacity` matches the recomputed groupby count |
| D3 | `hamilton_floor` | `floor_allocation == (capacity * 20511) // 41730` for all 636 rows; `Σ = 20211` |
| D4 | `hamilton_remainder` | `remainder_numerator == (capacity * 20511) % 41730` for all 636 rows |
| D5 | `hamilton_shortfall` | `shortfall == 300`; exactly 300 rows have `received_remainder_unit == True` |
| D6 | `hamilton_tie_break` | 295 strata strictly above the cut; 12 strata tied at remainder numerator **19,095** (all capacity 5); exactly **5** of them awarded; the 5 are the `stratum_id`-ascending first five of the tied set |
| D7 | `allocation_sum` | `Σ allocation_count == 20511` |
| D8 | `no_over_capacity` | `allocation_count ≤ capacity` for all 636; `max(allocation_count - capacity) == 0` |
| D9 | `no_dropped_stratum` | `min(allocation_count) ≥ 1`; 0 strata with allocation 0 |
| D10 | `allocation_repeat_invariant` | for every repeat, per-stratum selected counts equal `allocation_count` exactly |
| D11 | `all_blocks_present` | all 576 blocks appear in every repeat |

## E. Prevalence contract

| # | check_id | Expected |
|---|---|---|
| E1 | `positives_per_repeat` | exactly **1,438** in every repeat |
| E2 | `negatives_per_repeat` | exactly **19,073** in every repeat |
| E3 | `prevalence_value` | 1438/20511 = 0.070108722… |
| E4 | `prevalence_drift` | \|0.070108722 − 0.069757968\| = 0.00035075 |
| E5 | `prevalence_bound` | drift ≤ 70/20511 = 0.003413 → within bound |
| E6 | `positives_not_equalised` | subsample positives (1,438) ≠ Manavgat positives (784) — the design deliberately does not equalise them |

## F. Determinism and invariance

| # | check_id | Expected |
|---|---|---|
| F1 | `seed_derivation` | recomputing `blake2b(f"{SCHEMA_VERSION}|{repeat_id}|{stratum_id}", digest_size=8) % 2**32` reproduces every `sampling_seed` in `selected_cells.parquet` |
| F2 | `repeat_seeds_distinct` | 20 distinct `repeat_seed` values |
| F3 | `selection_reproducible` | re-running the documented selection for a sampled subset of repeats (all 20 if cheap) reproduces the exact `cell_id` sets |
| F4 | `row_order_invariance` | shuffling the canonical input frame's row order (fixed permutation seed) and re-running the selection yields **identical** `cell_id` sets for every repeat |
| F5 | `stratum_iteration_order_invariance` | permuting the order in which strata are visited yields identical selections (the seed depends on `stratum_id`, not on position) |

## G. Fold contract

| # | check_id | Expected |
|---|---|---|
| G1 | `fold_source_artifact` | `fold_mapping.parquet.source_artifact_sha256 == e16e6b18020b745da6e91a9e59664778b7a2887c8227729048fc459bc9df8cd4`; `fold_source == "persisted_artifact"` |
| G2 | `fold_mapping_complete` | 41,730 rows, one per canonical Muğla primary `cell_id`, no duplicates |
| G3 | `fold_mapping_matches_artifact` | every `(cell_id, fold_id)` equals the frozen artifact's |
| G4 | `same_mapping_all_repeats` | for every repeat, each sampled `cell_id`'s `fold_id` equals the `fold_mapping.parquet` value — one mapping, 20 repeats |
| G5 | `not_reoptimised` | `config.folds.reoptimised_per_repeat == false`; no per-repeat fold artifact exists |
| G6 | `fold_count` | 5 folds, ids 0…4, `splitter == "StratifiedGroupKFold"`, `random_state == 42`, `strict_folds == true` |
| G7 | `no_block_leakage` | for every repeat and fold, `set(train_blocks) ∩ set(test_blocks) == ∅`; recomputed from `part-within.parquet` joined to `large_block_id` |
| G8 | `both_classes_both_sides` | every (repeat, fold): ≥ 1 positive and ≥ 1 negative on the train side and on the test side. Expected per-fold test positives 293 / 280 / 295 / 281 / 289 |
| G9 | `fold_row_counts` | per-repeat fold sizes exactly 4,111 / 4,096 / 4,107 / 4,096 / 4,101 |
| G10 | `complete_oof_coverage` | in `part-within.parquet`, each `(repeat_id, cell_id)` appears exactly once; 20,511 predictions per repeat; no NaN probability |
| G11 | `block_partition_bijection` | `large_block_id ↔ frozen_block_id` is 1:1 across all 576 blocks |

## H. Arm completeness

| # | check_id | Expected |
|---|---|---|
| H1 | `three_arms_present` | `repeat_metrics.csv` contains all of `within`, `source`, `target` |
| H2 | `directions_complete` | within: `mugla_2021`; source: `mugla_2021_to_manavgat_2021`, `mugla_2021_to_bejis_2022`; target: `manavgat_2021_to_mugla_2021`, `bejis_2022_to_mugla_2021` — 5 direction rows, no more, no fewer |
| H3 | `no_silent_skip` | 5 directions × 2 families × 3 metrics × 20 repeats = **600 rows** in `repeat_metrics.csv`, no nulls in `subsample_value`; no `skipped` / `reason` field is populated anywhere |
| H4 | `summary_row_count` | `subsampling_summary.csv` has exactly **30 rows**, one per (direction, family, metric) |
| H5 | `source_arm_targets_full` | in `part-source.parquet`, target rows per repeat = 20,511 (Manavgat) and 15,190 (Bejís); target `cell_id` sets equal the full canonical target primary sets; target positives 784 and 1,100 |
| H6 | `source_arm_target_labels_untouched` | target `burned` values equal the canonical Step8A values, 0 mismatches |
| H7 | `target_arm_sources_full` | `part-target.parquet.reused_from_artifact` is `True` for all rows; `source_artifact_sha256` equals `0aace925…` (Manavgat) / `183ce975…` (Bejís); no target-arm fit identity exists in the fit registry |
| H8 | `target_arm_rows` | 20,511 rows per (repeat, direction); `target_cell_id` set equals that repeat's selected set exactly |
| H9 | `target_arm_probabilities_copied` | every `(cell_id, family)` probability equals the frozen artifact's value bit-for-bit |

## I. References

| # | check_id | Expected |
|---|---|---|
| I1 | `within_reference_values` | baseline 0.6979859420145867 / 0.16368744586018302 / 0.11271834078260733; thermal 0.7773268638729566 / 0.30192591578806804 / 0.07726667111613451 — matching `block_10_cells/step8b_metrics.json` (sha256 `a826279f…`) |
| I2 | `within_reference_recomputed` | recomputing from `block_10_cells/oof_predictions.parquet` reproduces I1 to < 1e-12 |
| I3 | `source_reference_values` | mugla→manavgat baseline 0.52151982986128 / thermal 0.40099966584697444; mugla→bejís baseline 0.4507383379572875 / thermal 0.5831912381443964 (ROC-AUC; PR-AUC and Brier likewise) |
| I4 | `target_reference_values` | manavgat→mugla baseline 0.5079294316533508 / thermal 0.47015987108700774; bejís→mugla baseline 0.5922389820175834 / thermal 0.6184747489978263 |
| I5 | `references_recomputed` | all four transfer references recompute from their `cross_region_transfer_predictions.parquet` to < 1e-12 |
| I6 | `reference_provenance` | every `reference_artifact_sha256` in `reference_metrics.csv` matches the on-disk file; each transfer metrics JSON's `resolved_inputs` pins the same three Step8A digests as B1–B3 |
| I7 | `reference_direction_resolution` | direction `S→T` was read from `outputs/cross_region/{S}__{T}/step9b/`, confirmed by digest |
| I8 | `recomputation_flags` | `reference_metrics.csv.recomputation_matches` is `True` for all 30 rows |

## J. Metric arithmetic

| # | check_id | Expected |
|---|---|---|
| J1 | `subsample_metrics_recomputable` | for every row of `repeat_metrics.csv`, recomputing the metric from the corresponding `oof_predictions/` partition reproduces `subsample_value` to < 1e-12 |
| J2 | `natural_delta` | `natural_delta == subsample_value - full_reference_value` for all 600 rows |
| J3 | `brier_orientation` | for `metric == "brier_score"`: `oriented_delta == full_reference_value - subsample_value`, i.e. `oriented_delta == -natural_delta` |
| J4 | `auc_orientation` | for `roc_auc` and `pr_auc`: `oriented_delta == natural_delta` |
| J5 | `brier_natural_preserved` | `subsample_value` and `full_reference_value` for Brier are the plain lower-is-better scores, all in (0, 1); no negated value is stored in a `*_value` field |
| J6 | `orientation_semantics` | `metric_orientation` is `lower_is_better_oriented_by_negation` for Brier and `higher_is_better` for the AUCs, in all 600 rows |
| J7 | `summary_statistics` | `median`, p2.5, p97.5 (`numpy.percentile(..., method="linear")`), min, max recomputed from the 20 repeat values match `subsampling_summary.csv` to < 1e-12 |
| J8 | `interval_ordering` | `subsampling_interval_lower ≤ subsample_median ≤ subsampling_interval_upper`; `subsample_minimum ≤ lower`; `upper ≤ subsample_maximum` |
| J9 | `position_token_correct` | `reference_position` recomputed on the **oriented** scale equals the emitted token, for all 30 rows |
| J10 | `position_token_vocabulary` | only `below_subsampling_interval`, `inside_subsampling_interval`, `above_subsampling_interval` appear |

## K. Language contract

| # | check_id | Expected |
|---|---|---|
| K1 | `no_forbidden_tokens` | case-insensitive scan of every emitted text file and every parquet string column for: `confidence interval`, `95% ci`, `ci_2_5`, `ci_97_5`, `ci_lower`, `ci_upper`, `significant`, `significance`, `p-value`, `p_value`, `pvalue`, `istatistiksel olarak anlaml`, `anlaml` → **0 hits** |
| K2 | `interval_names_correct` | the columns are named `subsampling_interval_lower` / `subsampling_interval_upper`; no `ci_*` column exists in any CSV |
| K3 | `permitted_sentences_only` | every `interpretation_sentence` is one of the two verbatim sentences of `SCIENTIFIC_CONTRACT.md` §7 |
| K4 | `no_causal_claims` | scan for `sample size causes`, `regional effect is proven`, `difference is eliminated`, `eliminates the` → 0 hits |
| K5 | `limitations_present` | `summary.json.limitations` contains all five standing limitations |
| K6 | `three_arm_reading_present` | `summary.json.three_arm_reading` has all three keys populated — no single-arm conclusion stands alone |

## L. Fit registry and process hygiene

| # | check_id | Expected |
|---|---|---|
| L1 | `fit_identity_count` | `summary.json.fit_accounting.observed.total == 240`; `within == 200`, `source == 40`, `target == 0` |
| L2 | `fit_reuse_events` | `reuse_events == 40` — each source fit referenced exactly twice |
| L3 | `no_target_arm_fits` | no fit identity begins with `target|` |
| L4 | `expected_vs_observed` | `expected == observed` for every key |
| L5 | `source_fit_sharing_is_exact` (deep mode) | independently re-fit one `(repeat_id, family)` source model and assert the two targets' probability vectors agree with the emitted ones to **≤ 1e-12**. These 2 audit fits are reported separately and are not counted in the 240. |

> **Correction, made during implementation (2026-08-03).** This row originally
> demanded *bit-identical* re-fit output. That is unattainable and the design
> was wrong to ask for it: the canonical estimator is
> `RandomForestClassifier(..., n_jobs=-1)`, which averages per-tree
> probabilities in a thread-scheduling-dependent order, so **two fits of the
> very same training frame in the same process already differ by ~1 ULP**
> (measured: 2.22e-16). Bit-identity is therefore not a property of any re-fit,
> shared or not, and the criterion is agreement far below any metric-relevant
> scale. This does not weaken the 80 → 40 reduction — if anything it argues for
> it, since reusing one fit *removes* a source of that floating-point noise
> rather than introducing one. The target arm, which copies stored
> probabilities rather than re-fitting, **is** checked with exact equality
> (H9).
| L6 | `n_fits_consumed_consistent` | `repeat_metrics.csv.n_fits_consumed` is 5 (within) / 1 (source) / 0 (target) and sums consistently with L1 |
| L7 | `no_gee` | no `ee.` / `earthengine` / `gee_utils` import reachable from the analysis module; no GEE call recorded in the manifest |
| L8 | `canonical_outputs_unchanged` | `outputs/experiments/*/step8a/**`, `outputs/experiments/*/step8b/**`, `outputs/experiments/mugla_2021/robustness/**` re-hash to their pre-run digests, including the four files of `REPOSITORY_INVENTORY.md` §4.2 |
| L9 | `transfer_outputs_unchanged` | all eight `outputs/cross_region/*/step9b/` files of §4.3 re-hash to their recorded digests |
| L10 | `output_containment` | every path in `manifest.json` resolves under `outputs/diagnostics/mugla_subsampling/<analysis_id>/`; no file was written anywhere else |
| L11 | `no_quarantine_residue` | `_quarantine/` is absent, or present and reported with a reason; its presence alone is not a FAIL but an unexplained partition is |
| L12 | `resume_idempotent` | re-running with `--resume` after a complete run writes no new bytes and changes no digest |

---

## Summary

**87 checks across 12 groups.** Coverage of the contract's required checks:

| Contract requirement | Checks |
|---|---|
| schema and deterministic analysis ID | A1–A3 |
| canonical Step8A hashes | B1–B4 |
| only three primary AOIs / no Evia | B5–B7 |
| only Muğla subsampled | B8 |
| 20 repeats | C1 |
| every repeat exactly 20,511 unique rows | C2 |
| no replacement | C3 |
| no foreign-region rows | C4 |
| samples are a canonical Muğla subset | C4–C7 |
| label and spatial allocation recomputable | C6, C7, D1–D2 |
| proportional / Hamilton allocation correct | D3–D9 |
| prevalence rounding contract correct | E1–E5 |
| deterministic seeds | F1–F3 |
| row-order invariance | F4–F5 |
| same frozen full-Muğla fold mapping in all repeats | G1–G6 |
| no fold leakage | G7 |
| complete OOF coverage | G10 |
| all three arms complete | H1–H4 |
| source-arm target cohorts full | H5–H6 |
| target-arm source models full | H7–H9 |
| references bound to canonical artifacts | I1–I8 |
| metric arithmetic recomputable | J1–J2, J7 |
| Brier orientation correct | J3–J6 |
| no p-values | K1 |
| interval wording correct | K2–K4 |
| no GEE | L7 |
| canonical outputs unchanged | L8 |
| transfer outputs unchanged | L9 |
| output confined to the namespace | L10 |
| fit registry expected identity count | L1–L6 |
| no silently skipped repeat or direction | H3, C1 |
