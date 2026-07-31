# Implementation Plan

**Nothing in this document has been executed.** It describes the work that follows once the
design acceptance gate in `README.md` §8 is cleared.

Guiding principle, and the reason the plan is as small as it is:
**the per-AOI scientific engine already exists, is AOI-generic, and has passed.** The work is
an orchestration, synthesis, manifest and validation layer around
`src/window_closure_sensitivity.py` — **not** a reimplementation of it.

---

## 1. Non-negotiable constraints

| Constraint | Rationale |
|---|---|
| No scientific function in `src/window_closure_sensitivity.py` may be edited | Manavgat's `analysis_id` is a hash over its frozen configuration; editing the module risks invalidating a PASSed result |
| Additive-only changes to that module (new constants, new exported helpers) | Same reason |
| No new model family, adaptation method or feature family | Frozen contract |
| No re-run, move or overwrite of `outputs/experiments/**` | Canonical outputs |
| No re-run of `outputs/diagnostics/window_closure_sensitivity/manavgat_2021/**` | Read-only reference |
| `evia_2021` never enters any code path | Hard exclusion |
| Every write confined to `outputs/diagnostics/multi_region_window_closure/<analysis_id>/` | Namespace isolation |

---

## 2. Components

### 2.1 Reused unchanged

| Component | Role |
|---|---|
| `src/window_closure_sensitivity.py` | The entire per-AOI analysis; called, never edited |
| `scripts/run_window_closure_sensitivity.py` | Per-AOI dispatcher; invoked per AOI |
| `scripts/validate_window_closure_{predictor_export,local_downstream,model,compare}.py` | Per-AOI stage validation; run per AOI before set-level validation |
| `core/regions.py`, `core/experiment_context.py` | Registry and context — read-only |
| `core/config.py` | Frozen model, bootstrap and MODIS-season constants |
| `src/step8b_*`, `src/step8c_*`, `src/step3_*`, `src/step5*`, `src/step7*`, `src/step8a_*`, `src/step6_*` | Production pipeline |
| `scripts/run_predictors_only.export_image_direct_or_tiled` | Export transport |
| `src/multi_aoi_transfer_synthesis/aoi_set.py::AoiSet` | **AOI-set identity, canonical ordering, deterministic set ID** |
| `scripts/validate_window_closure_predictor_export.py::Report` | Validator reporting structure |

### 2.2 To be added

| Path | Contents |
|---|---|
| `src/multi_region_window_closure/__init__.py` | Package exports |
| `src/multi_region_window_closure/config.py` | `SCHEMA_VERSION = "multi_region_window_closure.v1"`, `ACTUAL_AOIS`, `REFERENCE_AOI`, `EXCLUDED_AOIS`, namespace roots, `build_set_configuration`, `compute_set_analysis_id` |
| `src/multi_region_window_closure/scope.py` | `assert_aoi_scope` — exactly 3 actual, Manavgat reference-only, `evia_2021` absent. **Fails closed.** |
| `src/multi_region_window_closure/dates.py` | `build_window_dates_table` — the 12-row table, delegating to `build_window_variants` and `modis_month_filter_transparency` |
| `src/multi_region_window_closure/orchestrate.py` | Per-AOI stage driver over `run_analysis`; all-or-nothing set semantics; resume and force propagation |
| `src/multi_region_window_closure/cohort_gate.py` | The `cohort-feasibility` stage: 14 checks, `cohort_inventory.csv`, `fold_mapping.parquet` |
| `src/multi_region_window_closure/collect.py` | Read per-AOI artefacts → `metrics.csv`, `oof_predictions.parquet`, `bootstrap_replicates.parquet`, `bootstrap_summary.csv`. **Emits both Brier orientations explicitly.** |
| `src/multi_region_window_closure/reference.py` | Read-only loader for the Manavgat namespace; records source path + sha256; **never writes there** |
| `src/multi_region_window_closure/synthesis.py` | `regional_summary.csv`, `four_region_synthesis.csv`. **Structurally incapable of emitting a pooled column** (see §3) |
| `src/multi_region_window_closure/wording.py` | `MULTI_REGION_FORBIDDEN_PHRASES` = union of the three lists; `assert_multi_region_wording` |
| `src/multi_region_window_closure/manifest.py` | `manifest.json` + `manifest.sha256`, with explicit self-hash exclusion |
| `src/multi_region_window_closure/render.py` | `report.md`, all 13 mandated sections |
| `scripts/run_multi_region_window_closure.py` | CLI (§4) |
| `scripts/validate_multi_region_window_closure.py` | The 92 checks |
| `tests/test_multi_region_window_closure.py` | The 26 test groups (§7) |
| Registration in `src/multi_aoi_transfer_synthesis/schema_adapters.py` | `multi_region_window_closure.v1` → **generic** adapter path, never the legacy one |

---

## 3. Two structural decisions worth stating up front

**Pooled inference must be impossible, not merely prohibited.** `synthesis.py` builds
`four_region_synthesis.csv` by iterating AOIs and emitting rows from **one AOI's**
`bootstrap_summary.csv` at a time. No function in the module ever receives two AOIs'
data in the same call. A pooled column would therefore require a new function, not a slip —
and `Y01` re-derives every row from a single AOI to confirm it.

**Brier orientation must be explicit at the point of writing.** `collect.py` emits
`orientation` and `orientation_definition` as mandatory, non-nullable columns. There is no
code path that writes a Brier number without one. This converts `M06`/`M07` from a review
burden into a schema guarantee.

---

## 4. CLI contract

```bash
python scripts/run_multi_region_window_closure.py \
  --aois bejis_2022 mugla_2021 evia_2021_extended \
  --reference-aoi manavgat_2021 \
  --shifts 0 7 14 \
  --from-stage plan --to-stage summarize \
  [--dry-run | --resume | --force]
```

* `--dry-run`, `--resume` and `--force` are mutually exclusive
  (`assert_resume_force_exclusive`, line 1726).
* `--aois` must resolve to exactly the three actual AOIs; any other set is rejected before
  anything else happens.
* `evia_2021` is rejected explicitly with a named error, not silently dropped.
* There is **no** `--skip-aoi` and no partial-set flag. Subsets cannot be run.

---

## 5. Resume semantics

A stage is reused **only** when all five hold:

1. recorded `status == "PASS"`,
2. `config_hash` matches the current set configuration,
3. every recorded input hash still matches on disk,
4. `output_manifest` is complete and every recorded hash still matches,
5. the outputs satisfy the current validator's structural checks.

File existence alone is never sufficient. On drift, the affected stage **and every downstream
stage** re-run. A partial AOI or partial variant is never PASS, so it can never be resumed
from.

---

## 6. Force and quarantine

`--force` affects **only** `outputs/diagnostics/multi_region_window_closure/<analysis_id>/`.

```
<namespace>/_quarantine/<UTC timestamp>_<reason>/<original relative path>
```

* Existing outputs are **moved**, never deleted.
* `outputs/experiments/**` is untouchable.
* `outputs/diagnostics/window_closure_sensitivity/**` is untouchable.
* Every quarantine move is recorded in `stages/<stage>.json.quarantined_paths`.

This mirrors the mechanism already proven in practice by the Manavgat namespace, which carries
`_quarantine/model/20260731T082055Z/` from a real force cycle.

---

## 7. Test plan

No test code was written in this task. Level key: **U** = unit, **I** = integration
(tmp_path + injected engines), **E** = end-to-end dry-run. **No test may reach Earth Engine** —
use the `predictor_engine` / `prelabel_exporter` / `local_downstream_engine` injection points.

| # | Test name | Level | Fixture / input | Assertion | Expected failure mode |
|---|---|---|---|---|---|
| 1 | `test_date_shift_moves_both_ends` | U | each AOI ctx, shifts 0/7/14 | `start' == start − shift` and `end' == end − shift` for all 12 rows | `SHIFT_ARITHMETIC_ERROR` |
| 2 | `test_inclusive_exclusive_boundaries` | U | all 12 rows | `earth_engine_end_exclusive` true; `effective_last_included == end − 1d`; `calendar_days_inclusive == duration + 1` | `DATE_SEMANTICS_INCONSISTENT` |
| 3 | `test_duration_constant_across_variants` | U | all 4 AOIs | 60/60/60, 57/57/57, 58/58/58, 56/56/56 | `WINDOW_DURATION_DRIFT` |
| 4 | `test_label_event_gate_dates_invariant` | U | all 4 AOIs | `label_*`, `event_*`, `gate_*` identical across variants; `*_source_field` non-empty | `LABEL_WINDOW_DRIFT` |
| 5 | `test_aoi_scope_exactly_three_actual` | U | valid and invalid AOI sets | exactly 3 actual + 1 reference accepted; 2 or 4 rejected | `AOI_SET_MISMATCH` |
| 6 | `test_old_evia_excluded` | U | `--aois` including `evia_2021` | rejected by name; no artefact ever contains `evia_2021` (not followed by `_extended`) | `EXCLUDED_AOI_PRESENT` |
| 7 | `test_canonical_hash_drift_detected` | I | tmp copy of a Step8A parquet with one byte changed | run refuses to start | `CANONICAL_HASH_DRIFT` |
| 8 | `test_cohort_equality_across_variants` | I | synthetic 3-variant frames | `cohort_hash` identical; row/grid ID sets identical | `VARIANT_COHORT_MISMATCH` |
| 9 | `test_cohort_mismatch_fails_closed` | I | one variant missing 5 cells | run stops; no fit occurs | `VARIANT_COHORT_MISMATCH` |
| 10 | `test_fold_mapping_invariance` | I | 3 variants, one cohort | one `fold_mapping_hash` per AOI, shared by all variants | `FOLD_MAPPING_DRIFT` |
| 11 | `test_fold_hash_differs_between_aois` | I | 2 synthetic AOIs | hashes differ | `FOLD_HASH_COLLISION` |
| 12 | `test_fold_class_feasibility` | I | cohort with a single-class fold | refuses before fitting | `FOLD_CLASS_INFEASIBILITY` |
| 13 | `test_metric_arithmetic_recompute` | I | known OOF table | ROC-AUC / PR-AUC / Brier recompute to `1e-9` | `METRIC_RECOMPUTE_MISMATCH` |
| 14 | `test_brier_orientation_both_recorded` | U | known baseline/thermal Brier pair | `difference_natural == t − b`; `difference_oriented == b − t`; both rows carry `orientation` | `BRIER_ORIENTATION_ERROR` |
| 15 | `test_brier_orientation_cannot_be_omitted` | U | attempt to write a Brier row without `orientation` | write rejected | schema violation |
| 16 | `test_paired_bootstrap_shares_draw_plan` | I | 3 variants, 20 replicates | one `draw_plan_id` per `(aoi, replicate_id)` across all series | `BOOTSTRAP_NOT_PAIRED` |
| 17 | `test_bootstrap_does_not_refit` | I | spy/mock on `train_population` | call count is 0 during the compare stage | `BOOTSTRAP_REFIT_DETECTED` |
| 18 | `test_replicate_accounting` | I | forced invalid replicates | `valid + invalid == requested`; `valid == len(rows)` | `REPLICATE_ACCOUNTING_UNTRUTHFUL` |
| 19 | `test_duplicate_and_missing_fit_detected` | I | manipulated fit ledger | duplicate and missing both FAIL | `DUPLICATE_LOGICAL_FIT` / `MISSING_LOGICAL_FIT` |
| 20 | `test_expected_fit_count_is_90` | U | 3 AOI × 3 variants × 2 models × 5 folds | `expected_logical_fits == 90`; auxiliary Step7C fits == 6 | `FIT_COUNT_MISMATCH` |
| 21 | `test_dry_run_is_read_only` | E | tmp namespace, snapshot before/after | zero paths created/modified/deleted; all five flags false | `DRY_RUN_NOT_READ_ONLY` |
| 22 | `test_dry_run_touches_no_ee` | E | `ee` import guard raising on use | dry run completes without touching it | `GEE_IN_LOCAL_STAGE` |
| 23 | `test_resume_hash_binding` | I | PASS stage, then mutate one input | stage and all downstream re-run | `UNSAFE_RESUME` |
| 24 | `test_resume_rejects_partial_stage` | I | stage state with 2 of 3 AOIs | resume refuses | `RESUME_FROM_INVALID_STAGE` |
| 25 | `test_force_quarantines_not_deletes` | I | populated namespace + `--force` | files moved under `_quarantine/<ts>_<reason>/`; none deleted | `FORCE_WITHOUT_QUARANTINE` |
| 26 | `test_manifest_integrity_and_self_hash` | I | complete namespace | every file recorded; counts and byte totals match; `manifest.json` excluded from `files[]`; `manifest.sha256` verifies | `MANIFEST_INCOMPLETE` |
| 27 | `test_no_pooled_inference` | U | synthesis output | no prohibited column; every row re-derives from one AOI | `POOLED_INFERENCE_DETECTED` |
| 28 | `test_evia_wording_present` | U | rendered report + summaries | all three mandated phrases present; `aoi_role == "different_regime_control"` | `EVIA_FRAMING_MISSING` |
| 29 | `test_evia_not_framed_as_fourth_validation_region` | U | rendered report | forbidden framings absent; prevalence table present | `EVIA_FRAMING_VIOLATION` |
| 30 | `test_forbidden_language_union` | U | report containing each of the 20 phrases | every one is caught | `FORBIDDEN_LANGUAGE` |
| 31 | `test_permitted_language_not_flagged` | U | report using all 8 permitted phrases | zero false positives | — |
| 32 | `test_partial_aoi_fails` | I | 2 of 3 AOIs complete | no overall PASS | `PARTIAL_AOI` |
| 33 | `test_partial_variant_fails` | I | 2 of 3 variants for one AOI | no overall PASS | `PARTIAL_STAGE` |
| 34 | `test_canonical_overwrite_prevention` | I | attempt a write into `outputs/experiments/**` | refused; canonical hashes unchanged | `CANONICAL_OVERWRITE` |
| 35 | `test_manavgat_namespace_untouched` | I | full run against tmp roots | Manavgat window-closure namespace mtimes and hashes unchanged | `CANONICAL_OVERWRITE` |
| 36 | `test_model_contract_matches_manavgat` | U | frozen config vs Manavgat `preregistration.json` | estimator, both feature lists, seeds, `n_splits`, block size, `min_positives`, `calibration=None`, `adaptation=None` all equal | `MODEL_CONTRACT_DRIFT` |
| 37 | `test_modis_policy_pinned` | U | `config.json` vs `core.config` | `(6, 9)` both places | `MODIS_POLICY_DRIFT` |
| 38 | `test_modis_clipping_per_aoi` | U | all 12 rows | Bejís 0/0/0; Muğla 0/7/14; Evia-ext 0/3/10; Manavgat 0/7/14 | `DATE_SEMANTICS_INCONSISTENT` |
| 39 | `test_canonical_variant_never_exports` | U | `predictor_artifact_jobs(canonical)` | raises; `export_plan.csv` canonical rows have `export_required == false` | `STATIC_ARTIFACT_REGENERATED` |
| 40 | `test_static_features_reused_not_recomputed` | U | `export_plan.csv` | every static row is `reuse` with `export_required == false` | `STATIC_ARTIFACT_REGENERATED` |

The mandated 26 test categories map onto tests 1–34; tests 35–40 are additional coverage the
audit showed to be worth having.

---

## 8. Stage-by-stage implementation order

Each step ends with its tests green before the next begins.

| # | Step | Deliverable | Exit criterion |
|---|---|---|---|
| 1 | Scope and config | `config.py`, `scope.py` | Tests 5, 6, 20, 36, 37 pass |
| 2 | Date table | `dates.py`, `window_dates.csv` | Tests 1–4, 38 pass; the 12 rows match `WINDOW_DATE_AUDIT.md` §3 exactly |
| 3 | Export plan | `export_plan.csv` | Tests 39, 40 pass; 243 rows |
| 4 | Orchestrator (dry-run only) | `orchestrate.py` | Tests 21, 22 pass |
| 5 | Cohort gate | `cohort_gate.py` | Tests 8–12 pass |
| 6 | Collection | `collect.py` | Tests 13–15, 19 pass |
| 7 | Bootstrap collection | replicate/summary writers | Tests 16–18 pass |
| 8 | Reference loader | `reference.py` | Test 35 passes |
| 9 | Synthesis | `synthesis.py` | Tests 27–29 pass |
| 10 | Wording | `wording.py` | Tests 30, 31 pass |
| 11 | Manifest | `manifest.py` | Test 26 passes |
| 12 | Rendering | `render.py` | All 13 report sections present |
| 13 | Resume / force | orchestrator completion | Tests 23–25, 32–34 pass |
| 14 | Validator | `validate_multi_region_window_closure.py` | All 92 checks implemented; runs green on a synthetic complete namespace |

---

## 9. Execution order (after implementation)

### 9.1 Tests

```bash
pytest tests/test_window_closure_sensitivity.py \
       tests/test_window_closure_local_downstream.py \
       tests/test_window_closure_model.py \
       tests/test_window_closure_compare.py \
       tests/test_multi_region_window_closure.py -q
```

Regression on the four existing modules must stay green. **A new failure there means the
frozen core was touched — stop and revert.**

Known and expected (memory-backed): one Evia signed-AUC frozen-input test fails **by design**
until its bootstrap is re-run. That test belongs to a different analysis and does not gate
this work; the numbers it covers are unaffected.

### 9.2 Dry-run

```bash
# per AOI first — cheapest possible falsification of the frozen-input contract
for a in bejis_2022 mugla_2021 evia_2021_extended; do
  python scripts/run_window_closure_sensitivity.py --experiment "$a" --shifts 0 7 14 \
    --from-stage plan --to-stage compare --dry-run \
    | tee "logs/mrwc_dryrun_${a}.log"
done

# then the set
python scripts/run_multi_region_window_closure.py \
  --aois bejis_2022 mugla_2021 evia_2021_extended \
  --reference-aoi manavgat_2021 --shifts 0 7 14 --dry-run \
  | tee logs/multi_region_window_closure_dryrun.log

python scripts/validate_multi_region_window_closure.py \
  --mode dry-run --log logs/multi_region_window_closure_dryrun.log
```

Required before proceeding: `prerequisites_ready: true` for each AOI; all five dry-run flags
false; zero stage-owned path changes; validator overall PASS.

### 9.3 Actual run

Only if every dry-run gate passed **and** free disk ≥ 120 GB (`O14`).

```bash
python scripts/run_multi_region_window_closure.py \
  --aois bejis_2022 mugla_2021 evia_2021_extended \
  --reference-aoi manavgat_2021 --shifts 0 7 14 \
  --from-stage plan --to-stage summarize \
  | tee logs/multi_region_window_closure_actual.log
```

Recommended staging for a ~7 h run — stop and inspect at each boundary:

1. `--to-stage plan` — check `window_dates.csv` against `WINDOW_DATE_AUDIT.md` §3.
2. `--from-stage export --to-stage export` — the long GEE leg (~4 h).
3. `--from-stage local-downstream --to-stage cohort-feasibility` — **the decision point.**
   Replace the projected cohort sizes with measured ones and confirm fold feasibility.
4. `--from-stage fit --to-stage compare`.
5. `--from-stage summarize --to-stage summarize`.

### 9.4 Validation

```bash
# per-AOI stage validators
for a in bejis_2022 mugla_2021 evia_2021_extended; do
  for v in predictor_export local_downstream model compare; do
    python "scripts/validate_window_closure_${v}.py" \
      --experiment "$a" --shifts 0 7 14 --mode actual \
      | tee "logs/mrwc_validate_${a}_${v}.log"
  done
done

# set-level validator
python scripts/validate_multi_region_window_closure.py --mode actual \
  | tee logs/multi_region_window_closure_validation.log
```

### 9.5 Artefact review order

1. `window_dates.csv` — all 12 rows against `WINDOW_DATE_AUDIT.md` §3.
2. `cohort_inventory.csv` — `cohort_hash` identical within each AOI; attrition reconciles;
   `removed_prelabel_censor` non-zero for Muğla / Evia-ext is expected, not an anomaly.
3. `fold_mapping.parquet` — per-fold class counts; hash identical within AOI, different across.
4. `metrics.csv` — 54 rows; spot-recompute two entries by hand.
5. `bootstrap_summary.csv` — 162 rows; `valid + invalid == 1000` everywhere; both orientations
   present for every Brier series.
6. `four_region_synthesis.csv` — **read the header first** and confirm no pooled column.
7. `report.md` — Evia framing verbatim; MODIS asymmetry section present; limitations complete.
8. `manifest.json` + `manifest.sha256` — counts, byte totals, self-hash.
9. `validation_report.json` — 92 checks, zero required FAIL, zero required SKIP.

---

## 10. Rollback and quarantine

| Situation | Action |
|---|---|
| Stage failed mid-run | Fix, then `--resume`. Hash binding re-runs whatever drifted. |
| Output known bad | `--force` → quarantine, then re-run. Nothing is deleted. |
| Config changed | New `analysis_id` ⇒ new namespace. The old one is untouched. |
| Canonical hash drift detected | **Stop.** Do not force. Investigate why a canonical artefact changed. |
| Frozen core accidentally edited | `git checkout -- src/window_closure_sensitivity.py`; re-run the four existing test modules. |
| Disk pressure | Quarantine directories are the safe thing to prune, only after validation PASS. |

Rollback is never `git reset`, `git clean` or a bulk delete. The namespace is the unit of
recovery.

---

## 11. Pre-commit review checklist

- [ ] `git status --short` shows changes **only** under `src/multi_region_window_closure/`, `scripts/run_multi_region_window_closure.py`, `scripts/validate_multi_region_window_closure.py`, `tests/test_multi_region_window_closure.py`, `docs/multi_region_window_closure_design/`, and one additive registration in `src/multi_aoi_transfer_synthesis/schema_adapters.py`
- [ ] `git diff src/window_closure_sensitivity.py` is **empty**, or additive-only and reviewed line by line
- [ ] `git diff core/regions.py core/config.py` is **empty**
- [ ] No file under `outputs/experiments/**` modified
- [ ] No file under `outputs/diagnostics/window_closure_sensitivity/**` modified
- [ ] The four canonical hashes still reproduce
- [ ] All four existing window-closure test modules green
- [ ] New test module green; all 40 tests present
- [ ] Validator reports overall PASS with zero required SKIPs
- [ ] `four_region_synthesis.csv` header contains no pooled/meta-analytic column
- [ ] `report.md` carries the Evia framing verbatim and the MODIS asymmetry section
- [ ] Forbidden-phrase scan clean across `report.md` and every JSON prose field
- [ ] `manifest.json` complete; `manifest.sha256` verifies
- [ ] `evia_2021` absent from every artefact, path and log
- [ ] Commit message states that the Manavgat contract was preserved unchanged and names the reference hash `054a1961…f3439`
