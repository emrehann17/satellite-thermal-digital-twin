# Validator Contract — `scripts/validate_few_shot_recovery.py`

42 checks. Every check emits
`{check_id, status, expected, observed, evidence_path, note}` with
`status ∈ {PASS, FAIL, SKIPPED}`. **Any FAIL makes the overall status FAIL**,
and a directory whose `manifest.json` does not record `PASS` is not citable.

Two modes, following `scripts/validate_marginal_aoa_completion.py`:

- **dry-run** — contract, stage-order, prerequisite and plan checks only
  (`FSR-01`–`FSR-08`, `FSR-38`–`FSR-42`). Writes nothing, reads no produced
  artifact.
- **actual** — every check against a produced artifact.

---

## A. Scope and identity

| ID | Check | Method |
|---|---|---|
| FSR-01 | Exactly 6 directed pairs | `config.directed_pairs` has length 6 and equals the frozen list; every produced table has exactly 6 distinct `direction` values |
| FSR-02 | No self-pair | `source_experiment != target_experiment` on every row of every artifact |
| FSR-03 | Direction tokens never sorted | `direction == f"{source}_to_{target}"` on every row; both orderings of each unordered pair are present |
| FSR-04 | Canonical Step8A hashes | `sha256_file()` of each of the three datasets equals the frozen digest in `SCIENTIFIC_CONTRACT.md` §3.4; `input_hashes.json` records `match: true` for all three |
| FSR-05 | Primary population only | `population == "burnable_tree_shrub_grass"` on every row; no secondary population appears anywhere |
| FSR-06 | Evia excluded from the primary analysis | No occurrence of `evia_2021`, `evia_2021_extended` or `evia` in any produced file (CSV values, parquet columns and values, JSON keys and values, `report.md`), except inside the `config.excluded_experiments` block and the corresponding `limitations` entry |
| FSR-07 | Population definition applied | Row counts per target in `oof_predictions` equal the measured population sizes: manavgat 20 511, bejis 15 190, mugla 41 730 |
| FSR-08 | `analysis_id` reproduces | Recomputing `compute_analysis_id(config.scientific_configuration)` reproduces the directory name |

## B. Leakage firewall

| ID | Check | Method |
|---|---|---|
| FSR-09 | **Evaluation/adaptation block overlap = 0** | For every (direction, fold, repeat, budget): `set(selected adaptation_block_id) ∩ set(evaluation_block_id of that fold) == ∅`. Checked over all 6 × 5 × 10 × 6 = 1 800 selections |
| FSR-10 | Adaptation blocks come only from the target training pool | Every `adaptation_block_id` is a target block of that fold's training pool, and is never a source block id |
| FSR-11 | Evaluation labels never entered a training frame | For every fit, the training-frame cell ids and the evaluation-fold cell ids are disjoint. Verified from `selected_blocks` + fold assignment, per `fit_id` |
| FSR-12 | Evaluation labels never entered a training *decision* | `selection_seed` is reproducible from `(schema, source, target, fold, repeat)` alone; block tiering uses only training-pool blocks; no artifact records any quantity derived from evaluation-fold labels other than the reported metrics |
| FSR-13 | Counterfactual label test | Permuting `burned` on evaluation-fold rows of a synthetic fixture changes metric values but leaves `selected_blocks.parquet`, `selection_seed`, tier assignment and every training frame byte-identical |
| FSR-14 | No forbidden feature column | `check_no_forbidden_features` passes for both feature lists; no `FORBIDDEN_FEATURE_COLUMNS` entry appears in either list |
| FSR-15 | Preprocessing fitted on the training frame only | Every fit goes through `build_pipeline(...)`, so imputers/encoder are inside the `Pipeline`; no artifact records a preprocessing statistic computed over a frame containing evaluation rows |
| FSR-16 | No threshold selection | `config.metrics.threshold_selection_performed == false`; no artifact contains a `threshold`, `precision`, `recall`, `f1` or `balanced_accuracy` column |
| FSR-17 | Full OOF coverage | For every (direction, family, condition, budget, repeat): every target population `cell_id` appears exactly once, with no NaN probability. 6 × 62 = 372 coverage assertions |

## C. Selection contract

| ID | Check | Method |
|---|---|---|
| FSR-18 | Nested budgets | For every (direction, fold, repeat), the blocks at budget `k` are a strict subset of those at the next larger budget; equivalently `selection_rank < k` selects exactly the budget-`k` set |
| FSR-19 | Selection is at block level | `budget_blocks == n distinct adaptation_block_id` for every selection; `adaptation_row_count` varies and is never used as the budget |
| FSR-20 | Deterministic seeds | Recomputing `blake2b(...)` reproduces every `selection_seed`; re-running `plan` twice yields a byte-identical `selected_blocks.parquet` |
| FSR-21 | Seed independent of budget and family | For fixed (direction, fold, repeat), `selection_seed` is identical across all budgets; `baseline` and `thermal` reference the same `selection_key` |
| FSR-22 | Tier order respected | Within a selection, sorting by `selection_rank` yields all `both_classes` before any `positives_only` before any `negatives_only` |
| FSR-23 | Every k≥1 selection contains a positive | `adaptation_positive_count > 0` on every selection with `budget_blocks >= 1` (all 1 800) |
| FSR-24 | 10 repeats for every k>0 | Every (direction, fold, budget) with `k > 0` has exactly `repeat_id ∈ {0..9}`; `n_repeats == 10` in `recovery_curve.csv` for those budgets |
| FSR-25 | 1 realisation for k=0 and ceiling | `raw` and `ceiling` rows have `repeat_id == 0` only and `n_repeats == 1`; their `selection_p2_5 == selection_p97_5 == selection_median == point estimate` |
| FSR-26 | Row-order invariance | Shuffling input row order in a fixture leaves `selected_blocks.parquet` unchanged (blocks are sorted by id before shuffling) |

## D. Fold contract

| ID | Check | Method |
|---|---|---|
| FSR-27 | 10-cell blocks via the canonical utility | `large_block_id` values match `assign_large_blocks(df, 10)` exactly; format `b10_r{r}_c{c}`; `block_size_cells == 10` and `nominal_scale == "approximately_5_km"` in config and every row |
| FSR-28 | Strict 5-fold spatial CV | `make_spatial_folds(..., n_splits_requested=5, random_state=42, strict=True)` returned `n_splits_used == 5` for all three targets; no block on both sides of any fold; both classes on both sides |
| FSR-29 | Folds depend on the target only | The fold assignment for a target is identical across the two directions sharing it, verified cell-by-cell |
| FSR-30 | No ad-hoc random split | No artifact records a row-level random split; the only splitter named anywhere is `StratifiedGroupKFold` |

## E. Condition contract

| ID | Check | Method |
|---|---|---|
| FSR-31 | Raw is source-only | Every `raw` fit has `n_train_target_rows == 0` and `budget_blocks == 0`; `n_train_source_rows` equals the full source population size |
| FSR-32 | Ceiling is target-only | Every `ceiling` fit has `n_train_source_rows == 0`; its training blocks are exactly the target training pool of that fold |
| FSR-33 | Few-shot uses the full source population | Every `few_shot` fit has `n_train_source_rows` equal to the full source population and `n_train_rows == n_train_source_rows + adaptation_row_count` |
| FSR-34 | Canonical model contract unchanged | The recorded classifier class and hyperparameters equal `build_classifier("random_forest", 42).get_params(deep=False)` exactly; `tuning_performed == false`; `sample_weight_argument_used == false`; the pre-existing `class_weight="balanced"` is declared |
| FSR-35 | Ceiling reproduces the frozen 10-cell values | `manavgat_2021` baseline 0.7475502988238435 / thermal 0.7974298472620660, `bejis_2022` baseline 0.7793700238725079 / thermal 0.8244685786179753 and `mugla_2021` baseline 0.6979859420145867 / thermal 0.7773268638729566, each to `abs_diff <= 1e-9`. An experiment with no registered frozen anchor → `SKIPPED` with reason `no_frozen_block_10_artifact` |

## F. Metric and recovery contract

| ID | Check | Method |
|---|---|---|
| FSR-36 | Signed recovery, unclipped | `recovery_fraction` is recomputed from the stored oriented values and matches to `1e-12`; no clipping to `[0,1]`, no `abs()`. The check passes only if the recomputation is exact — values outside `[0,1]` and negative values must survive |
| FSR-37 | Brier orientation correct | For `metric == "brier_score"`: `metric_orientation == "lower_is_better_oriented_by_negation"`, `oriented_value == -metric_value`, and the natural-sign `metric_value` is positive. For `roc_auc`/`pr_auc`: `oriented_value == metric_value`. Recovery arithmetic uses only oriented values |
| FSR-38 | Degenerate denominator handled | `|ceiling_gap| < 1e-6` ⇒ `recovery_fraction` null and status `undefined_degenerate_denominator`; never a division result |
| FSR-39 | Ceiling-not-above-raw flagged | `ceiling_oriented <= raw_oriented` ⇒ `ceiling_not_above_raw == true` and status `ceiling_not_above_raw` |

## G. Wording and containment

| ID | Check | Method |
|---|---|---|
| FSR-40 | **No p-values, no significance language** | Case-insensitive scan of every produced `.md`, `.json`, `.csv` and every parquet column name for: `confidence interval`, `95% ci`, `\bci\b`, `ci_2_5`, `ci_97_5`, `significant`, `significance`, `p-value`, `p_value`, `pvalue`, `p =`, `istatistiksel olarak anlamlı`, `anlamlı`. Zero hits required. Interval columns must be named `selection_p2_5` / `selection_p97_5`. The only permitted occurrence is inside `config.uncertainty.forbidden_terms` and the `limitations` sentence that names what the interval is not |
| FSR-41 | No output outside the namespace | Every path in `manifest.json` is under `outputs/diagnostics/few_shot_recovery/<analysis_id>/`; no file outside it has an mtime inside the run window |
| FSR-42 | Frozen production artifacts unchanged, and no GEE | sha256 of the three canonical Step8A datasets, of `outputs/experiments/*/step8b/*`, of `outputs/cross_region/*/step9b/*` and `*/step10/*`, and of `outputs/robustness/step8_large_block/**` are unchanged before and after the run. The module's import graph contains neither `core.gee_utils` nor `ee`; `earth_engine.used == false` |

---

## Mapping to the required checks

| Required | Check IDs |
|---|---|
| 6 directed pairs | FSR-01 |
| no self-pair | FSR-02 |
| canonical hashes | FSR-04 |
| primary population | FSR-05, FSR-07 |
| evaluation/adaptation block overlap = 0 | FSR-09, FSR-10, FSR-11 |
| nested budgets | FSR-18 |
| deterministic seeds | FSR-20, FSR-21, FSR-26 |
| full target OOF coverage | FSR-17 |
| target labels read only for adaptation/training | FSR-11, FSR-12, FSR-13 |
| evaluation label never leaked into a training decision | FSR-12, FSR-13, FSR-15, FSR-16 |
| raw k=0 is source-only | FSR-31 |
| ceiling is target-only | FSR-32 |
| signed recovery unclipped | FSR-36 |
| Brier orientation | FSR-37 |
| 10 repeats | FSR-24, FSR-25 |
| no p-values | FSR-40 |
| Evia outside the primary analysis | FSR-06 |
| canonical Step8A unchanged | FSR-04, FSR-42 |
| transfer outputs unchanged | FSR-42 |
| no GEE | FSR-42 |
| no output outside the namespace | FSR-41 |

Additional checks beyond the required list — FSR-03, FSR-08, FSR-14,
FSR-19, FSR-22, FSR-23, FSR-27 to FSR-30, FSR-33 to FSR-35, FSR-38, FSR-39 —
cover the two forced decisions (block scale, ceiling reproduction), the
class-feasibility guarantee, and the fit-composition accounting.
