# Implementation Plan

Build order, files and the test contract. **Nothing here has been executed:**
no code was written, no model fitted, no bootstrap run, no artifact produced.

---

## 0. Resolve the two blockers first

Neither is a coding problem; both are contract decisions that must be confirmed
before Step 1, because they change what the gate and the schema assert.

| # | Blocker | Proposed resolution | Where |
|---|---|---|---|
| B-1 | §6's ≤1e-12 metric reproduction tolerance is unattainable — two runs of the identical canonical pipeline differ by up to 4.867e-08 in ROC-AUC | two-tier gate: exact (≤1e-12) from persisted probabilities, plus refit at ≤1e-06 **and** ≤8×rank-quantum; probabilities keep the requested ≤1e-12 | `SCIENTIFIC_CONTRACT.md` §9, `NUMERICAL_FEASIBILITY.md` §6 |
| B-2 | Step10 stores no Brier, for any method, in metrics or in the bootstrap replicates | recompute all Brier references from the persisted probability vectors (exact); compute Brier bootstrap replicates with no refit, under the reproduced draws | `SCIENTIFIC_CONTRACT.md` §7.2 |

If either resolution is rejected, the affected metric or gate tier must be
dropped from scope rather than weakened silently.

## 1. Files to create

| Path | Role | Est. lines |
|---|---|---:|
| `src/coral_lambda_sensitivity.py` | the whole analysis: contract, λ grid, gate, arms, bootstrap, summarisation, staged runner | ~1,700 |
| `scripts/run_coral_lambda_sensitivity.py` | thin CLI dispatcher, no scientific logic | ~110 |
| `scripts/validate_coral_lambda_sensitivity.py` | the 91 checks of `VALIDATOR_CHECKLIST.md` | ~1,100 |
| `tests/test_coral_lambda_sensitivity.py` | unit + adversarial + invariance tests | ~1,300 |

## 2. Files to modify — narrow and additive only

| Path | Change |
|---|---|
| `core/pipeline_orchestrator.py` | add `run_coral_lambda_sensitivity_stage(...)`, same signature shape as `run_mugla_subsampling_stage` (line 819) |
| `scripts/main.py` | import `STAGES`, add `cmd_coral_lambda_sensitivity`, register the `coral-lambda-sensitivity` subparser |
| `tests/test_main_cli.py` | subcommand registration + flag passthrough |
| `tests/test_pipeline_orchestrator.py` | stage dispatch + namespace containment |

**No production module is modified.** `core/step10_shared.py`,
`core/config.py`, `src/step10*.py`, `src/step9*.py` and `src/step8b*.py` are
**imported only**. In particular `STEP10_CORAL_LAMBDA` is read and never
written, and no `outputs/cross_region/**` path is opened for writing.

## 3. Module skeleton — `src/coral_lambda_sensitivity.py`

```
# --- identity -------------------------------------------------------------
SCHEMA_VERSION, DIAGNOSTIC_NAMESPACE, DIAGNOSTIC_CLASS
STAGES, STAGE_REQUIRES, STAGE_OUTPUTS
class CoralLambdaSensitivityError(SystemExit)

# --- frozen contract ------------------------------------------------------
PRIMARY_EXPERIMENTS, EXCLUDED_EXPERIMENTS, EXCLUDED_TOKENS
PRIMARY_DIRECTIONS, SECONDARY_DIRECTIONS, CONTEXTUAL_ONLY_DIRECTIONS
LAMBDA_GRID = (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
CANONICAL_LAMBDA = 1e-5 ; CANONICAL_LAMBDA_INDEX = 4
LAMBDA_TOKENS = {...}                       # float -> "lambda_1e_m5"
lambda_token(value) / lambda_value(token)   # bijective, asserted
METRICS, LOWER_IS_BETTER, ALLOWED_NUMERICAL_STATUS
TIER1_TOLERANCE, TIER2_PROB_TOLERANCE, TIER2_METRIC_TOLERANCE,
TIER2_RANK_QUANTUM_MULTIPLIER, TIER2_BRIER_TOLERANCE
AUC_INSENSITIVE, AUC_MODEST, BRIER_INSENSITIVE_RATIO, BRIER_MODEST_RATIO
MAGNITUDE_TOKENS, INSTABILITY_TOKEN, SUPPORT_TOKENS, FORBIDDEN_TOKENS
CANONICAL_STEP8A_SHA256

# --- paths ----------------------------------------------------------------
diagnostics_root / analysis_root / stage_marker_path
canonical_step8a_path(experiment_id, experiments_root=None)
resolve_step10_reference_dir(source_id, target_id, output_root=None)
    -> outputs/cross_region/{source}__{target}/step10     # frozen rule
rejected_duplicate_dir(source_id, target_id, output_root=None)  # recorded, unused
assert_inside_namespace / _atomic_write_text / _atomic_write_parquet
sha256_file / sha256_path / canonical_json / compute_analysis_id

# --- inputs and gate ------------------------------------------------------
build_frozen_input_inventory(...)  / assert_canonical_step8a_hashes(strict=True)
load_step10_reference(source_id, target_id, ...)      # metrics + predictions + digests
recompute_reference_metrics(predictions, labels)      # exact, supplies Brier
rank_quantum(n_pos, n_neg) -> 1/(n_pos*n_neg)
run_tier1_gate(...)   # metrics from persisted probabilities  vs stored
run_tier2_gate(...)   # fresh lambda=1e-5 refit             vs resolved artifact
assert_gate_passed(rows)

# --- adaptation -----------------------------------------------------------
load_direction_frames(source_id, target_id, ...)      # label-blind target
zscore_pair(X_source, X_target, numeric_feats)        # lambda-independent, cached
coral_cell(Xs_z, Xt_z, numeric_feats, lambda_)        # the ONLY lambda entry point
    -> {A, Cs, Ct, diagnostics, numerical_status}
    wraps fit_coral_alignment / apply_coral, catches Step10Error,
    records whether the 1e-12 eigenvalue floor bound
fit_and_predict(X_source_coral, y_source, X_target_z, feature_list)
fit_identity(direction, model_family, lambda_token)
expected_scientific_fit_count() -> 72

# --- metrics --------------------------------------------------------------
compute_all_metrics(y_true, y_prob)   # ROC/PR via step10 helper, Brier via step8b
metric_orientation / natural_delta / oriented_delta

# --- bootstrap ------------------------------------------------------------
build_series_matrix(direction)         # 22 series x n_target, float64, contiguous
run_paired_bootstrap(direction_frame, series, n=1000, seed=42)
    ONE generator, ONE pass, all series scored per replicate
    verified against run_n_way_paired_bootstrap on the canonical ROC/PR series
summarise_bootstrap(replicates) -> point, p2.5, p97.5, n_valid, n_invalid, token

# --- summarisation --------------------------------------------------------
deviation_scale(metric, target_id)     # 1.0 for AUCs, p(1-p) for Brier
magnitude_token(max_abs_dev, scale)
build_sensitivity_summary(metrics, bootstrap_summary, diagnostics)
scan_forbidden_tokens(root)

# --- staged runner --------------------------------------------------------
planned_output_layout()
write_stage_marker / read_stage_marker / verify_stage_complete
verify_partition(analysis_id, direction, family, lambda_token, ...)
quarantine_namespace
run_plan / run_fit / run_bootstrap / run_summarize
run_analysis(from_stage, to_stage, dry_run, resume, force, output_root, experiments_root)
```

## 4. Build order

**Step 1 — contract skeleton.** Constants, λ token bijection, paths, stage
validation, `planned_output_layout`, `--dry-run`. Tests: dry-run writes
nothing; reversed stage range rejected; token bijection is total and
round-trips; `lambda_token(0.0) == "lambda_0"`; no token contains `.`.

**Step 2 — input gate and direction resolution.** `build_frozen_input_inventory`,
`assert_canonical_step8a_hashes(strict=True)`, `resolve_step10_reference_dir`,
and recording of the rejected duplicate. Tests: mutated digest fails closed;
resolution returns `{S}__{T}` for all four directions; the rejected duplicate is
recorded with a *different* digest; an Evia id fails closed.

**Step 3 — reference loading and exact recomputation (Tier 1).** Load the four
resolved artifacts, recompute ROC/PR/Brier from the persisted probabilities,
compare against `step10_metrics.csv`. Tests: deviations ≤1e-12 on synthetic
fixtures; a tampered probability column fails the tier; Brier rows carry
`recomputed_from_persisted_probabilities`.

**Step 4 — the CORAL cell.** `coral_cell(...)` around the unmodified
`fit_coral_alignment` / `apply_coral`, with the `Step10Error` → status mapping
and floor instrumentation. Tests: λ enters only as the additive ridge (assert
`Cs_returned − cov(Xs_z) == λI` to ≤1e-12); a deliberately singular synthetic
covariance at λ=0 yields `singular_unregularised_covariance` **and** is
retained; the floor-bound flag is true exactly when an eigenvalue is < 1e-12;
`A` for a shared source but different target differs (no cross-direction
reuse).

**Step 5 — Tier 2 gate.** Fresh λ=1e-5 refit per direction × family, compared
against the resolved artifact under the §9 tolerances. Tests: passing case;
a perturbed probability vector fails; the 8 audit fits are counted separately;
gate failure prevents `fit` from starting.

**Step 6 — `plan` stage.** Emits `config.json`, `input_hashes.json`,
`repository_inventory.json`, `lambda_grid.csv`, `canonical_reproduction.csv`.
Test: the λ grid is inside the hashed config (perturb → different
`analysis_id`).

**Step 7 — `fit` stage.** The 72 cells, z-score cached per (direction, family),
72 partitions written. Tests: exactly 72 identities; every partition's cell-id
set equals the canonical target population; `burned` absent from every
partition; z-score statistics identical across the nine λ of a cell; a failing
cell is written with NA metrics rather than dropped.

**Step 8 — `bootstrap` stage.** One call per direction over all 22 series.
Tests: the ROC/PR replicates for the canonical series are **identical** to
`run_n_way_paired_bootstrap` on the same frame and seed (this is the check that
proves the draw scheme was not reimplemented); every series in a replicate uses
the same drawn index set; `n_valid + n_invalid == 1000`; no fit occurs
(monkeypatch `build_pipeline` to raise).

**Step 9 — `summarize` stage.** Deviations, tokens, summary, report, manifest,
forbidden-token scan. Tests: a synthetic case for each of the three magnitude
tokens at both AUC and Brier scales; `numerical_instability_present` is additive;
no `best_lambda`-shaped key exists anywhere; support tokens correct at the
interval boundaries.

**Step 10 — resume / force / orchestrator / CLI / validator.** Tests: a `fit`
marker missing one partition digest reports incomplete; `--force` quarantines
without deleting and touches no Step9/Step10 path; registration tests; the
validator against a small synthetic end-to-end run in `tmp_path`.

## 5. Test contract

`tests/test_coral_lambda_sensitivity.py`, following
`tests/test_mugla_subsampling.py`:

- **Synthetic fixtures only** for unit tests — small Step8A-shaped frames with
  the canonical `cell_id = r{row}_c{col}` identity and all ten feature columns,
  plus synthetic Step10 reference artifacts whose stored metrics are computed
  from the same synthetic probabilities so Tier 1 can pass.
- **`tmp_path`-injected** `output_root` / `experiments_root` in every test.
- **`test_*_fails_closed`** for every guard in `SCIENTIFIC_CONTRACT.md` §12.
- **λ-semantics tests** — the sharpest ones in the suite, because they are what
  stops λ drifting into a different meaning:
  `test_lambda_is_additive_ridge_on_both_covariances`,
  `test_lambda_zero_gives_unridged_covariance`,
  `test_lambda_does_not_reach_build_pipeline`,
  `test_lambda_is_not_an_interpolation_between_raw_and_coral`,
  `test_step10_coral_lambda_constant_is_never_mutated`.
- **Firewall tests**: `test_target_label_absent_from_every_partition`,
  `test_zscore_stats_do_not_change_when_target_labels_are_permuted`.
- **Invariance tests**: `test_zscore_stats_are_lambda_invariant`,
  `test_bootstrap_draws_identical_across_lambda_series`.
- **Exact-count tests**: `test_exactly_72_scientific_fits`,
  `test_exactly_216_metric_rows`, `test_exactly_24_summary_rows`,
  `test_nine_lambda_values_ascending`.
- **Marked slow**: one real-frame integration test asserting the
  `NUMERICAL_FEASIBILITY.md` §2 eigenvalue literals; reads production parquets
  read-only and writes only to `tmp_path`.

Targeted run:

```
PYTHONPATH="$PWD" python -m pytest -q \
  tests/test_coral_lambda_sensitivity.py \
  tests/test_main_cli.py \
  tests/test_pipeline_orchestrator.py --durations=20
```

## 6. CLI surface

```
python scripts/main.py coral-lambda-sensitivity \
    [--from-stage plan] [--to-stage summarize] \
    [--dry-run] [--resume] [--force] \
    [--output-root PATH] [--experiments-root PATH]

python scripts/validate_coral_lambda_sensitivity.py [--analysis-id ID] [--dry-run] [--deep]
```

## 7. Ordering constraints

1. Both gate tiers complete in `plan`; `fit` cannot start otherwise.
2. Step 4 (the CORAL cell) precedes Step 5 (Tier 2), because Tier 2 is the same
   code path at λ=1e-5 — that is what makes the gate meaningful rather than a
   parallel implementation.
3. Step 8's equivalence test against `run_n_way_paired_bootstrap` must pass
   before any bootstrap output is treated as valid.

## 8. What is deliberately not built

No λ selection, ranking-by-performance, or recommendation. No new CORAL
variant, no interpolation parameter, no shrinkage estimator. No change to the
eigenvalue floor. No modification to `core/config.py`. No new model, feature,
preprocessing step or metric implementation beyond wiring `brier_score_loss`
into the existing helper set. No writes outside
`outputs/diagnostics/coral_lambda_sensitivity/`. No Earth Engine call anywhere
in the module or its import graph.
