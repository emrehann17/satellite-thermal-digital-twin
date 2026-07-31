# Implementation Plan

Nothing in this plan has been implemented. No file listed under §1 exists yet.

---

## 1. Files to create

| Path | Role | Est. lines |
|---|---:|---:|
| `src/few_shot_recovery.py` | All scientific logic: config, stage contract, block tiering, budget loop, fit loop, recovery arithmetic, selection intervals, report | ~1 400 |
| `scripts/run_few_shot_recovery.py` | Thin dispatcher + argparse, no scientific logic (pattern: `scripts/run_marginal_aoa_completion.py`) | ~140 |
| `scripts/validate_few_shot_recovery.py` | The 42 validator checks, dry-run and actual modes (pattern: `scripts/validate_marginal_aoa_completion.py`) | ~900 |
| `tests/test_few_shot_recovery.py` | Test contract, §5 | ~1 100 |

## 2. Files to modify

| Path | Change |
|---|---|
| `core/pipeline_orchestrator.py` | Add `run_few_shot_recovery_stage(...)` following the `run_marginal_aoa_completion_stage` signature exactly (`from_stage`/`to_stage`/`dry_run`/`resume`/`output_root`/`experiments_root`), dispatching to `scripts.run_few_shot_recovery.main`. Register through the **generic** stage adapter — do not add a bespoke routing branch. |
| `scripts/main.py` | Register the `few-shot-recovery` subcommand. |
| `tests/test_pipeline_orchestrator.py` | Namespace-containment and dispatch tests for the new stage. |
| `tests/test_main_cli.py` | Subcommand registration test. |

No existing scientific module is edited. `step8b`, `step9b`, `step8_large_block_robustness` and `core/step10_shared` are imported, never modified.

## 3. Module layout — `src/few_shot_recovery.py`

```python
SCHEMA_VERSION       = "few_shot_recovery.v1"
DIAGNOSTIC_NAMESPACE = "few_shot_recovery"
DIAGNOSTIC_CLASS     = "target_label_supervised_few_shot_adaptation_sensitivity"

PRIMARY_EXPERIMENTS  = ("manavgat_2021", "bejis_2022", "mugla_2021")
EXCLUDED_EXPERIMENTS = {"evia_2021_extended": "...", "evia_2021": "..."}
EXPECTED_DIRECTED_PAIRS = 6
POPULATION           = PRIMARY_POPULATIONS[0]          # step9a

BLOCK_SIZE_CELLS     = 10                              # assign_large_blocks
BLOCK_NOMINAL_SCALE  = NOMINAL_SCALES[10]              # "approximately_5_km"
N_OUTER_FOLDS        = STEP8B_N_SPLITS                 # 5
FOLD_RANDOM_STATE    = STEP8B_RANDOM_SEED              # 42
ESTIMATOR_SEED       = STEP8B_RANDOM_SEED              # 42
MODEL_NAME           = "random_forest"

BUDGETS              = (0, 1, 2, 4, 8, 16, 32)
N_REPEATS            = 10
TIER_ORDER           = ("both_classes", "positives_only", "negatives_only")

METRICS              = ("roc_auc", "pr_auc", "brier_score")
LOWER_IS_BETTER      = ("brier_score",)
DEGENERATE_DENOMINATOR_THRESHOLD = 1e-6                # == transfer_decomposition
SELECTION_PCT_LOW, SELECTION_PCT_HIGH = 2.5, 97.5

CANONICAL_STEP8A_SHA256 = {...}                        # the three frozen digests
FROZEN_CEILING_REFERENCE = {...}                       # manavgat, bejis block_10 paths
CEILING_REPRODUCTION_TOLERANCE = 1e-9

STAGES = ("plan", "fit", "summarize", "validate")
```

Function outline:

```
# --- contract / paths ---
validate_stage_range(from_stage, to_stage)
stage_side_effect_flags(stages)
diagnostics_root(output_root=None) / analysis_root(analysis_id, output_root=None)
planned_output_layout()
directed_pairs(experiment_ids) -> 6 pairs        # never sorted, no self-pair
direction_token(source_id, target_id)

# --- inputs ---
load_target_frame(experiment_id, experiments_root=None)
    # resolve_step8a_dataset_path -> read_parquet
    # -> assign_large_blocks(df, 10)          BEFORE population filtering
    # -> population_subset(df, POPULATION)
build_frozen_input_inventory(...) / assert_canonical_step8a_hashes(..., strict=True)

# --- folds (target-only, cached per target) ---
build_outer_folds(frame) -> list[(train_idx, test_idx)]
    # make_spatial_folds(y, large_block_id, 5, 42, strict=True)

# --- selection (plan stage; frozen before any fit) ---
selection_seed(source_id, target_id, outer_fold, repeat_id) -> int
block_tiers(pool_frame) -> (tier_a, tier_b, tier_c)      # sorted by block id
nested_block_ordering(pool_frame, seed) -> list[block_id]
build_selection_table(...) -> selected_blocks DataFrame
build_block_inventory(...) / build_budget_feasibility(...)

# --- fitting (fit stage) ---
fit_and_predict(train_frame, eval_frame, feature_list) -> np.ndarray
    # build_pipeline(feature_list, "random_forest", 42).fit(...).predict_proba(...)[:, 1]
run_direction(source_id, target_id, ...) -> (oof_frame, repeat_metric_rows)

# --- metrics / recovery (summarize stage) ---
oriented(metric_name, value) -> float
recovery_row(raw, fewshot, ceiling, metric) -> dict     # signed, unclipped, 3 statuses
selection_interval(values) -> (median, p2_5, p97_5, min, max)
build_recovery_curve(repeat_metrics) -> DataFrame
build_summary(...) / render_report(...)

# --- validate stage ---
verify_ceiling_reproduction(...) / write_manifest(...)
```

### 3.1 Fit-sharing implementation

Both sharings are memoised on a `fit_id` key, so the accounting in
`repeat_metrics.csv` is literally what happened:

- **raw** — `fit_id = hash(direction, family, "raw")`. Fitted once per
  (direction, family); the source model never sees the target, so it is
  fold-independent. Its predictions over the whole target are computed once
  and sliced per evaluation fold.
- **ceiling** — `fit_id = hash(target, family, fold, "ceiling")`. Fitted once
  per (target, family, fold) and reused by both directions sharing that
  target. 30 unique fits back 60 direction-level rows.
- **few_shot** — `fit_id = hash(direction, family, fold, budget, repeat)`. No
  sharing.

### 3.2 Memory discipline

The full `oof_predictions` table is ~9.6 M rows and must never be held whole.
`fit` processes **one direction at a time**, accumulates that direction's
predictions (~1.3–2.6 M rows), writes
`oof_predictions/direction=<token>/part-0.parquet`, and releases. Peak
resident set is then one direction's predictions plus one fitted forest.

## 4. Fit accounting and runtime budget

### 4.1 Number of model fits

| Condition | Formula | Fits |
|---|---|---:|
| raw | 6 directions × 2 families | 12 |
| few_shot | 6 directions × 2 families × 5 folds × 6 budgets (k>0) × 10 repeats | 3 600 |
| ceiling | 3 targets × 2 families × 5 folds | 30 |
| **Total unique fits** | | **3 642** |

Reported rows exceed unique fits by design (raw is evaluated against 5 folds;
ceiling is reported for 2 directions), and `fit_id` makes the difference
auditable.

### 4.2 Measured fit cost

A single timing probe was run (in-memory only, wrote nothing) on the
worst-case frame — mugla source (41 730 rows) + 32 manavgat adaptation blocks
(2 976 rows) = 44 706 rows:

| family | fit | predict (20 511 rows) |
|---|---:|---:|
| baseline (4 features) | 2.68 s | 0.25 s |
| thermal (10 features) | 3.49 s | 0.26 s |

`RandomForestClassifier` already runs with `n_jobs=-1`, so a single fit
saturates the available cores.

### 4.3 Wall-clock estimate

Scaling roughly linearly in training rows from the probe (~7.8 × 10⁻⁵ s/row
for thermal, ~6.0 × 10⁻⁵ s/row for baseline), and noting that each source
appears in exactly two directions (600 few-shot fits per family per source):

| source | mean train rows | thermal | baseline | few-shot subtotal |
|---|---:|---:|---:|---:|
| manavgat_2021 | ~22 000 | 1.7 s | 1.3 s | 600 × 3.0 s ≈ 1 800 s |
| bejis_2022 | ~16 700 | 1.3 s | 1.0 s | 600 × 2.3 s ≈ 1 380 s |
| mugla_2021 | ~43 200 | 3.4 s | 2.6 s | 600 × 6.0 s ≈ 3 600 s |
| | | | **few-shot total** | **≈ 6 800 s (1.9 h)** |

Plus ceiling (30 fits ≈ 60 s), raw (12 fits ≈ 25 s), and all predictions
(~3 700 predict calls ≈ 200 s), plus parquet serialisation of ~9.6 M rows
(≈ 300 s).

```
ESTIMATED WALL CLOCK:  2.0 – 3.0 hours, single process
ESTIMATED PEAK MEMORY: 2 – 4 GB
ESTIMATED DISK:        300 – 500 MB (dominated by oof_predictions)
```

### 4.4 Risk assessment

**Runtime risk: LOW.** Two to three hours is comparable to existing analyses
in this repository. No mitigation is applied by default — no reduction of
repeats, no subsampling, no smaller forest. If a future run needs to be
faster, the only sanctioned lever is process-level parallelism across the six
directions (they are independent and each writes its own partition), which
changes no number. Reducing `N_REPEATS`, `n_estimators` or the budget set
would change the frozen contract and require a new `analysis_id`.

**Memory risk: LOW**, conditional on the streaming discipline in §3.2. Holding
all six directions' predictions at once would need ~10–15 GB and is the one
implementation mistake that would turn this into a memory problem.

**Disk risk: LOW–MODERATE.** ~9.6 M prediction rows is the largest artifact
this repository would carry. It is required — the contract mandates a full
target OOF prediction per repeat — and is mitigated by float32 probabilities,
snappy compression and per-direction partitioning. Storing only metrics would
be smaller but would make FSR-17 uncheckable.

## 5. Test contract — `tests/test_few_shot_recovery.py`

All tests use synthetic fixtures and `tmp_path`-injected `output_root`; none
reads or writes a canonical path.

**Contract and scope (6):** exactly six directed pairs; no self-pair; direction
tokens never sorted; selection order of the experiment list does not change
the pairs; duplicate experiment fails closed; any Evia identifier fails closed.

**Blocks and folds (6):** `large_block_id` matches `assign_large_blocks(df, 10)`
byte-for-byte; blocks are assigned before population filtering; `strict=True`
folds have zero block overlap; OOF coverage is exactly 1; fold assignment is
identical across the two directions sharing a target; a target with too few
blocks fails closed rather than reducing the fold count.

**Selection (10):** nesting holds for every consecutive budget pair; tier order
is respected; every `k ≥ 1` selection contains a positive; the seed is
reproducible from `(direction, fold, repeat)`; the seed is invariant to budget
and to model family; row-order permutation of the input leaves the selection
unchanged; selection never draws an evaluation block; selection never draws a
source block; two runs of `plan` are byte-identical; a pool smaller than the
budget fails closed.

**Leakage (6):** evaluation cell ids are absent from every training frame;
permuting evaluation-fold labels leaves selection and training frames
identical; `check_no_forbidden_features` rejects a leaked column; no
preprocessing statistic is fitted outside a `Pipeline`; no threshold is ever
computed; raw training frames contain zero target rows.

**Metrics and recovery (10):** `oriented_brier == -brier`; ROC/PR orientation
unchanged; recovery fraction reproduces the closed form; a negative fraction
survives; a fraction > 1 survives; a degenerate denominator yields null plus
the right status; `ceiling <= raw` raises the flag; the selection interval
matches `numpy.percentile(..., method="linear")`; `k=0` and ceiling get
`n_repeats == 1` with degenerate intervals; metric values are `None`-safe when
a fold has one class.

**Model contract (4):** the classifier hyperparameters equal
`build_classifier("random_forest", 42).get_params(deep=False)`; no
`sample_weight` is ever passed; the pre-existing `class_weight="balanced"` is
declared in the config; the feature lists equal the shared cross-region
contract.

**Output and wording (4):** every written path is inside the namespace; the
forbidden-terms scan finds zero hits in a produced fixture artifact;
`manifest.json` lists every produced file with a matching sha256;
`--dry-run` writes nothing and fits nothing.

## 6. Execution order at implementation time

1. `src/few_shot_recovery.py` — constants, contract, path and pair helpers.
2. `tests/…` for §5 contract/blocks/folds; run.
3. Selection logic (tiering, nested ordering, seeds) + `plan` stage artifacts.
4. `tests/…` for selection and leakage; run. **`plan` must be reviewable and
   fully checkable before any fit is written.**
5. `fit` stage with the three conditions and fit-sharing memoisation.
6. `summarize` stage: recovery arithmetic, selection intervals, report.
7. `tests/…` for metrics/recovery/model contract; run.
8. `scripts/validate_few_shot_recovery.py`, all 42 checks; run in dry-run mode.
9. Orchestrator + CLI registration; `tests/test_pipeline_orchestrator.py` and
   `tests/test_main_cli.py`; run the full suite.
10. `--dry-run` end to end, review `planned_output_layout()`.
11. Actual run (2–3 h), then `validate` in actual mode.

Steps 1–10 write nothing outside `src/`, `scripts/`, `tests/` and `docs/`.
Step 11 is the first step that writes into `outputs/`, and it writes only
under `outputs/diagnostics/few_shot_recovery/<analysis_id>/`.

## 7. Preconditions before step 11

- The three canonical Step8A hashes still match `SCIENTIFIC_CONTRACT.md` §3.4.
- The two frozen 10-cell ceiling artifacts are present and unmodified.
- The working tree is committed, so `resolve_git_commit()` records a real
  commit rather than a dirty state.
- The full test suite passes.
