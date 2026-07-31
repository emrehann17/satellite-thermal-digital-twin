# Implementation Plan

Build order, files, and the test contract. Nothing here has been executed.

---

## 1. Files to create

| Path | Role | Est. lines |
|---|---|---:|
| `src/mugla_subsampling.py` | the whole analysis: contract constants, sampling, arms, summarisation, staged runner | ~1,500 |
| `scripts/run_mugla_subsampling.py` | thin CLI dispatcher (`build_parser` → `main` → `run_analysis`), no scientific logic | ~90 |
| `scripts/validate_mugla_subsampling.py` | the independent validator of `VALIDATOR_CHECKLIST.md` | ~900 |
| `tests/test_mugla_subsampling.py` | unit + adversarial + invariance tests | ~1,200 |

## 2. Files to modify

| Path | Change |
|---|---|
| `core/pipeline_orchestrator.py` | add `run_mugla_subsampling_stage(...)` following `run_few_shot_recovery_stage` (line 819): same `from_stage` / `to_stage` / `dry_run` / `resume` / `output_root` / `experiments_root` signature, docstring naming the diagnostic class and namespace, delegating to `scripts.run_mugla_subsampling.main` |
| `scripts/main.py` | import `STAGES as _MUGLA_SUBSAMPLING_STAGES`, add `cmd_mugla_subsampling`, register the `mugla-subsampling` subparser alongside the existing `few-shot-recovery` one (lines 1387–1453) |
| `tests/test_main_cli.py` | subcommand registration + argument passthrough test |
| `tests/test_pipeline_orchestrator.py` | stage-dispatch + namespace-containment test |

No production module changes. `step8b`, `step9a`, `step9b`,
`step8_large_block_robustness`, `step8_big_block_robustness` and
`core/step10_shared` are **imported only**. No canonical output is rewritten.

## 3. Module skeleton — `src/mugla_subsampling.py`

```
# --- identity -------------------------------------------------------------
SCHEMA_VERSION, DIAGNOSTIC_NAMESPACE, DIAGNOSTIC_CLASS
STAGES, STAGE_REQUIRES, STAGE_OUTPUTS
class MuglaSubsamplingError(SystemExit)

# --- frozen contract ------------------------------------------------------
EXPERIMENTS, EXCLUDED_EXPERIMENTS, SUBSAMPLED_EXPERIMENT
POPULATION, TARGET_SAMPLE_SIZE = 20511, N_REPEATS = 20
MUGLA_EXPECTED_POPULATION = 41730, MANAVGAT_EXPECTED_POPULATION = 20511
BLOCK_SIZE_CELLS = 10, BLOCK_COLUMN = "large_block_id"
FOLD_COUNT = 5, FOLD_RANDOM_STATE = 42, ESTIMATOR_SEED = 42
MODEL_FAMILIES = ("baseline", "thermal")
METRICS = ("roc_auc", "pr_auc", "brier_score"); LOWER_IS_BETTER = {"brier_score"}
ARM_WITHIN/ARM_SOURCE/ARM_TARGET
SOURCE_DIRECTIONS, TARGET_DIRECTIONS
PREVALENCE_DRIFT_BOUND_NUMERATOR = "n_label1_strata"
FORBIDDEN_TOKENS  # the denylist of SCIENTIFIC_CONTRACT.md §6

# --- paths / provenance ---------------------------------------------------
diagnostics_root, analysis_root, stage_marker_path
canonical_step8a_path(experiment_id, experiments_root=None)
frozen_fold_artifact_path(output_root=None)
transfer_reference_paths(source_id, target_id, output_root=None)
assert_inside_namespace, _atomic_write_text, _atomic_write_parquet
sha256_file, sha256_path, canonical_json, compute_analysis_id
build_frozen_input_inventory, assert_canonical_step8a_hashes(strict=True)

# --- sampling -------------------------------------------------------------
load_primary_population(experiment_id, ...)      # population_subset
assign_strata(frame)                             # assign_large_blocks(df, 10)
stratum_capacity_table(frame)                    # 636 rows
hamilton_allocation(capacities, target_total)    # integer-exact + tie-break
assert_allocation_valid(table, target_total)     # sum / capacity / >=1
prevalence_drift(table, full_prevalence)         # + bound check
repeat_seed(repeat_id) / stratum_seed(repeat_id, stratum_id)
select_repeat(frame, allocation, repeat_id)      # sorted cell_id -> permute -> head
build_selected_cells(frame, allocation, fold_map)  # 410,220 rows

# --- folds ----------------------------------------------------------------
load_frozen_fold_mapping(...)   -> (mapping_df, provenance)   # preferred
reproduce_fold_mapping(...)     -> (mapping_df, provenance)   # fallback
assert_fold_contract(selected, mapping)                       # G7–G11

# --- fits -----------------------------------------------------------------
class FitRegistry                       # after few_shot_recovery:740
fit_identity(arm, repeat_id, family, fold_id=None)
fit_and_predict(train_frame, eval_frame, feature_list)   # the ONLY fit site
expected_unique_fit_count() -> {"within":200,"source":40,"target":0,"total":240}

# --- arms -----------------------------------------------------------------
run_within_arm(...)     # 5-fold inherited-fold OOF, 200 fits
run_source_arm(...)     # 40 fits, each predicting on two full targets
run_target_arm(...)     # 0 fits: subset the frozen transfer predictions

# --- references / metrics -------------------------------------------------
load_within_reference(...) / load_transfer_reference(source_id, target_id)
build_reference_metrics(...)                     # 30 rows, with recomputation
metric_orientation(metric) / oriented_delta(metric, sub, full)
subsampling_interval(values)                     # median/p2.5/p97.5/min/max
reference_position(full_oriented, lower, upper)  # the three tokens
interpretation_sentence(position_token)          # the two permitted sentences

# --- staged runner --------------------------------------------------------
planned_output_layout()
write_stage_marker / read_stage_marker / verify_stage_complete
scan_forbidden_tokens(paths)
run_plan(...) / run_fit(...) / run_summarize(...)
run_analysis(from_stage, to_stage, dry_run, resume, output_root, experiments_root)
```

## 4. Build order

**Step 1 — contract skeleton.** Constants, paths, `planned_output_layout`,
stage validation, `--dry-run`. Tests: dry-run writes nothing; stage order
rejects `fit` before `plan`; namespace containment rejects an escaping path.

**Step 2 — input gate.** `build_frozen_input_inventory` +
`assert_canonical_step8a_hashes(strict=True)` + the population-count assertions
(41,730 / 20,511 / 15,190). Tests: a mutated digest fails closed; a Manavgat
population ≠ 20,511 fails closed; Evia anywhere in the experiment list fails
closed.

**Step 3 — allocation.** `stratum_capacity_table` + `hamilton_allocation` +
`assert_allocation_valid` + `prevalence_drift`. Unit tests use small synthetic
frames with hand-computed allocations, including a constructed exact tie.
Integration test against the real Muğla frame asserts the literals of
`SAMPLING_FEASIBILITY.md` §3–§4: 636 strata, floor sum 20,211, shortfall 300,
12 tied at remainder 19,095, 5 awarded, `Σ alloc = 20,511`, 0 over capacity,
0 dropped, 1,438 positives, drift 0.00035075 within bound 0.003413.

**Step 4 — selection.** `repeat_seed`, `stratum_seed`, `select_repeat`. Tests:
exactly 20,511 unique rows per repeat; per-stratum counts equal
`allocation_count`; **row-order invariance** (shuffle the input, get the same
sets); **stratum-iteration-order invariance**; two repeats differ; a repeat is
reproducible from its seed alone.

**Step 5 — fold inheritance.** `load_frozen_fold_mapping`, the fallback, and
`assert_fold_contract`. Tests: the loaded mapping covers 41,730 cells with 5
folds and 0 span; the per-repeat fold sizes are exactly 4,111 / 4,096 / 4,107 /
4,096 / 4,101 with positives 293 / 280 / 295 / 281 / 289; a synthetic mapping
that puts a block in two folds fails closed; a synthetic single-class fold fails
closed; the fallback reproduces the artifact's mapping on a synthetic frame.

**Step 6 — `plan` stage.** Emits `config.json`, `input_hashes.json`,
`sampling_inventory.csv`, `stratum_allocation.csv`, `selected_cells.parquet`,
`fold_mapping.parquet`, `reference_metrics.csv`, `stages/plan.json`. Test: the
selection is frozen and hashed before any fit — assert no estimator was
constructed during `plan` (monkeypatch `build_pipeline` to raise).

**Step 7 — references.** Load and independently recompute all five references.
Test: recomputation matches the stored JSON to < 1e-12 for all 30 rows; a
tampered reference artifact fails the digest gate.

**Step 8 — Arm C first (cheapest, zero fits).** Subset the frozen transfer
predictions to each repeat's cells. Test: `target_cell_id` set equality;
probabilities bit-identical to the artifact; metric recomputation exact; and
that the whole arm runs with `build_pipeline` monkeypatched to raise —
**proving no fit occurs.**

**Step 9 — Arm B.** 40 fits, each predicting on two full targets. Tests: target
cohorts are the full 20,511 / 15,190 with untouched labels; the registry
records 40 fits and 40 reuse events; the deep-mode equality audit (an
independent re-fit gives bit-identical probabilities).

**Step 10 — Arm A.** 200 fits over inherited folds. Tests: exactly-once OOF
coverage per repeat; no block on both sides of a fold; per-fold counts as in
Step 5.

**Step 11 — `summarize`.** Orientation, intervals, position tokens, permitted
sentences, `summary.json`, `report.md`, `manifest.json`. Tests: Brier
orientation sign; interval ordering; a synthetic case for each of the three
position tokens; forbidden-token scan over every emitted file.

**Step 12 — resume / quarantine.** Tests: interrupting after `plan` and
resuming skips `plan`; a corrupted partition is quarantined and rewritten;
`--resume` on a complete run changes no digest.

**Step 13 — orchestrator + CLI + validator.** Registration tests, then the
validator against a small synthetic end-to-end run in `tmp_path`.

## 5. Test contract

`tests/test_mugla_subsampling.py`, following
`tests/test_few_shot_recovery.py` and `tests/test_marginal_aoa_completion.py`:

- **Synthetic fixtures only** for unit tests — a builder emits a small
  canonical-shaped frame (valid `cell_id = r{row}_c{col}`, `row_500m`,
  `col_500m`, `valid_for_modeling`, `burnable_tree_shrub_grass`, `burned`,
  all 10 feature columns) so nothing depends on production outputs.
- **`tmp_path`-injected `output_root` and `experiments_root`** in every test, so
  no test can write to a canonical path. A guard test asserts that the default
  `output_root` is never used when an override is supplied.
- **`test_*_fails_closed`** for every guard in `SCIENTIFIC_CONTRACT.md` §8.
- **Invariance tests**: `test_row_order_does_not_change_selection`,
  `test_stratum_iteration_order_does_not_change_selection`,
  `test_repeat_is_reproducible_from_seed_alone`.
- **Exact-count tests**: `test_every_repeat_has_exactly_20511_unique_rows`,
  `test_positive_count_is_1438_in_every_repeat`,
  `test_exactly_five_directions_and_600_repeat_rows`,
  `test_expected_unique_fit_count_is_240`.
- **Counterfactual tests**: `test_changing_target_labels_cannot_change_a_selected_cell`
  (Arm B/C target labels are evaluation-only and never influence selection or
  training); `test_target_arm_performs_no_fit`.
- **Language tests**: `test_no_forbidden_tokens_in_any_emitted_file`,
  `test_only_permitted_interpretation_sentences`.
- **Marked slow**: the real-frame integration test that asserts the
  `SAMPLING_FEASIBILITY.md` literals — it reads production parquets read-only
  and writes only to `tmp_path`.

## 6. CLI surface

```
python scripts/main.py mugla-subsampling \
    [--from-stage plan] [--to-stage summarize] \
    [--dry-run] [--resume] \
    [--output-root PATH] [--experiments-root PATH]

python scripts/run_mugla_subsampling.py   # same flags, direct entry point
python scripts/validate_mugla_subsampling.py [--analysis-id ID] [--dry-run] [--deep]
```

`--deep` enables the two source-fit-equality audit fits of check L5.

## 7. Ordering constraint

`plan` must complete and be hashed before `fit` begins, and Arm C must be
implemented before Arms A and B. Arm C is free, exercises the reference and
schema plumbing end to end, and is the strongest early check that the frozen
artifacts are what the design believes they are.

## 8. What is deliberately not built

No bootstrap. No threshold selection. No calibration. No adaptation. No new
model, feature, preprocessing step, fold algorithm or metric. No plotting
beyond what `report.md` needs. No writes to `outputs/experiments/`,
`outputs/cross_region/` or `outputs/robustness/`. No GEE call anywhere in the
module or its imports.
