# Validator Checklist — `multi_region_window_closure.v1`

The validator is **fail-closed** and **never runs a stage**: no dry run, no export, no fit, no
bootstrap, no Earth Engine call. It reads artefacts on disk and **re-derives** every published
number, so a mis-stated table cannot pass. This mirrors the standard already set by
`scripts/validate_window_closure_compare.py`.

---

## 1. Semantics

### 1.1 Status values — exactly three

| Status | Meaning |
|---|---|
| `PASS` | The check ran and the condition held |
| `FAIL` | The check ran and the condition did not hold |
| `SKIP` | The check could not run (prerequisite absent) |

No other value is permitted. In particular there is no `WARN` status — advisory findings are
`PASS` with a note recorded in `summary.json.warnings`.

### 1.2 Severity

| Severity | Effect |
|---|---|
| `required` | `FAIL` **or** `SKIP` blocks overall PASS |
| `advisory` | Recorded; does not block overall PASS |

**A required check that is `SKIP` can never yield an overall PASS.** This is the rule that
prevents "we could not check it, so it passed".

### 1.3 Overall PASS rule

```
overall_status = PASS  ⟺  ( no required check is FAIL )
                      AND ( no required check is SKIP )
                      AND ( technical_status == PASS )
                      AND ( scientific_contract_status == PASS )
                      AND ( namespace_safety_status == PASS )
                      AND ( every stage status ∈ {PASS} )
                      AND ( blockers == [] )
```

Three category verdicts are reported separately, reusing the existing `Report` class
(`scripts/validate_window_closure_predictor_export.py:67`):

* **technical** — artefacts, hashes, arithmetic, accounting
* **scientific-contract** — estimands, orientations, framing, wording, no pooling
* **namespace / provenance safety** — containment, canonical integrity, quarantine

A technical PASS is **not** a scientific conclusion, and a bootstrap-supported result is
**not** a technical verdict. They are never merged.

### 1.4 Evidence requirement

Every check emits an `evidence` string naming the **artefact path and field** it read, or the
recomputed value and its expected counterpart. A check with empty evidence is itself a `FAIL`
(meta-check `O15`).

### 1.5 Record shape

```jsonc
{ "check_id": "D02",
  "description": "Predictor window duration is invariant across variants",
  "severity": "required",
  "status": "PASS",
  "evidence": "window_dates.csv: mugla_2021 duration_days = 57/57/57",
  "failure_message": "BLOCKER: WINDOW_DURATION_DRIFT — variant <v> of <aoi> has duration <d>, canonical <d0>" }
```

### 1.6 Coverage

92 checks: **85 check IDs covering all 84 mandated items** (mandated item 75, "dry-run is
read-only", is split into `O03` and `O04` because emptiness and flag state are independently
falsifiable), **plus 7 additional checks** the audit showed to be necessary
(`C06`, `C07`, `H09`, `F06`, `O14`, `O15`, `O16`).

---

## 2. Scope — `S01`–`S08`

| ID | Description | Sev | Evidence source | Failure message |
|---|---|---|---|---|
| `S01` | Exactly three new actual AOIs | required | `config.json.aois_actual`, `regional_summary.csv` row count == 3 | `BLOCKER: AOI_SET_MISMATCH — expected 3 actual AOIs, found <n>` |
| `S02` | `bejis_2022` present with complete artefacts | required | `cohort_inventory.csv`, `metrics.csv` | `BLOCKER: MISSING_AOI — bejis_2022` |
| `S03` | `mugla_2021` present with complete artefacts | required | ″ | `BLOCKER: MISSING_AOI — mugla_2021` |
| `S04` | `evia_2021_extended` present with complete artefacts | required | ″ | `BLOCKER: MISSING_AOI — evia_2021_extended` |
| `S05` | **`evia_2021` appears nowhere** — not in any `aoi` column, not in any path, not in `export_plan.csv`, not in `manifest.files[]` | required | full-text scan of every artefact for the literal `evia_2021` **not** followed by `_extended` | `BLOCKER: EXCLUDED_AOI_PRESENT — evia_2021 found at <location>` |
| `S06` | `manavgat_2021` appears only as read-only reference | required | `four_region_synthesis.csv.aoi_source == "read_only_reference"`; no Manavgat row in `metrics.csv`, `oof_predictions.parquet`, `bootstrap_*` | `BLOCKER: REFERENCE_AOI_RECOMPUTED — manavgat_2021 found in <artifact>` |
| `S07` | Exactly three variants, per AOI | required | distinct `variant` per `aoi` == `{canonical, close_7d_earlier, close_14d_earlier}` | `BLOCKER: VARIANT_SET_MISMATCH — <aoi> has <set>` |
| `S08` | All six shifted scenarios complete (3 AOI × 2 shifted) | required | `export_plan.csv` + `variant_artifact_index.csv` | `BLOCKER: INCOMPLETE_SHIFTED_SCENARIOS — <n>/6` |

`S05` note: the two Evia experiments share byte-identical dates, so the check must key on
`experiment_id` and `output_namespace`, never on dates.

---

## 3. Date contract — `D01`–`D10`

| ID | Description | Sev | Evidence source | Failure message |
|---|---|---|---|---|
| `D01` | Canonical dates match `core/regions.py::EXPERIMENTS` exactly for all 4 AOIs | required | recompute via `build_experiment_context`; compare to `window_dates.csv` canonical rows | `BLOCKER: CANONICAL_DATE_MISMATCH — <aoi>: <expected> vs <found>` |
| `D02` | 7-day shift is exact: both ends moved by exactly 7 | required | `predictor_start/end` of `close_7d_earlier` vs `canonical` | `BLOCKER: SHIFT_ARITHMETIC_ERROR — <aoi> 7d: start Δ<a>, end Δ<b>` |
| `D03` | 14-day shift is exact | required | ″ for `close_14d_earlier` | `BLOCKER: SHIFT_ARITHMETIC_ERROR — <aoi> 14d` |
| `D04` | Predictor duration constant across variants | required | `calendar_duration_days` per AOI: 60/60/60, 57/57/57, 58/58/58 | `BLOCKER: WINDOW_DURATION_DRIFT — <aoi> <variant> <d> vs <d0>` |
| `D05` | Label dates unchanged across variants | required | `label_start`, `label_end` constant per AOI | `BLOCKER: LABEL_WINDOW_DRIFT — <aoi> <variant>` |
| `D06` | Event dates unchanged across variants | required | `event_start`, `event_end` constant; `event_source_field` non-empty | `BLOCKER: EVENT_WINDOW_DRIFT — <aoi> <variant>` |
| `D07` | Gate dates unchanged across variants | required | `gate_start`, `gate_end` constant; `gate_source_field` non-empty | `BLOCKER: GATE_WINDOW_DRIFT — <aoi> <variant>` |
| `D08` | Inclusivity / exclusivity semantics consistent | required | `earth_engine_end_exclusive == true`; `effective_last_included_date == predictor_end − 1d`; `calendar_days_inclusive == calendar_duration_days + 1` for all 12 rows | `BLOCKER: DATE_SEMANTICS_INCONSISTENT — <aoi> <variant>` |
| `D09` | **MODIS season policy unchanged** | required | `config.json.modis_season_policy` == `{6, 9}`; matches live `core.config.SUMMER_MONTH_START/END` | `BLOCKER: MODIS_POLICY_DRIFT — expected 6-9, found <x>-<y>` |
| `D10` | No off-by-one anywhere | required | recompute all 12 rows from `build_window_variants`; byte-compare; assert `lead_days >= 1`; assert no date is 29 Feb | `BLOCKER: DATE_OFF_BY_ONE — <aoi> <variant> <field>` |

`D06`/`D07` additionally assert that `event_*`/`gate_*` equal `label_*`, because the schema
aliases them (`WINDOW_DATE_AUDIT.md` §1.1) — the aliasing is verified, not assumed.

---

## 4. Canonical integrity — `C01`–`C07`

| ID | Description | Sev | Evidence source | Failure message |
|---|---|---|---|---|
| `C01` | All four canonical AOI Step8A hashes correct | required | recompute `sha256` of each `step8a_500m_modeling_dataset.parquet`; compare to `config.json.canonical_aoi_sha256` and the frozen anchors | `BLOCKER: CANONICAL_HASH_DRIFT — <aoi> expected <e>, got <g>` |
| `C02` | Manavgat reference hash correct | required | `manifest.manavgat_reference_hash == 054a1961…f3439` | `BLOCKER: CANONICAL_HASH_DRIFT — manavgat_2021 reference` |
| `C03` | Canonical outputs unmodified | required | every stage's `frozen_hashes_before == frozen_hashes_after`; `canonical_outputs_modified == false` in every stage state | `BLOCKER: CANONICAL_OUTPUT_MODIFIED — <stage> <role>` |
| `C04` | **Static artefacts not unexpectedly regenerated** | required | `export_plan.csv`: every `static_or_temporal == "static"` row has `export_required == false` **and** `reuse_or_recompute == "reuse"`; DEM and land-cover hashes unchanged; no `variant_artifact_index` row writes to `data/dem/**` or `gate_inputs/**` | `BLOCKER: STATIC_ARTIFACT_REGENERATED — <aoi> <role>` |
| `C05` | No input hash drift between stages | required | `stages/*.json` input hashes form a consistent chain | `BLOCKER: INPUT_HASH_DRIFT — <stage> <role>` |
| `C06` | All six required frozen input roles present and hashed for all three actual AOIs | required | `input_hashes.json`; `exists == true` for each of `canonical_step8a`, `dem_elevation`, `dem_slope`, `landcover_aligned`, `label_raw_burndate`, `label_burned_binary` | `BLOCKER: MISSING_FROZEN_INPUT — <aoi> <role>` |
| `C07` | Dependency lock hash recorded and matches the worktree | advisory | `manifest.dependency_lock_hash == sha256(requirements-lock.txt)` | `WARNING: DEPENDENCY_LOCK_DRIFT` |

---

## 5. Cohort — `H01`–`H09`

| ID | Description | Sev | Evidence source | Failure message |
|---|---|---|---|---|
| `H01` | The three variants of an AOI share the exact same cohort | required | `cohort_hash` identical for the 3 rows of each AOI in `cohort_inventory.csv` | `BLOCKER: VARIANT_COHORT_MISMATCH — <aoi>` |
| `H02` | Row-ID sets identical across variants | required | `oof_predictions.parquet`: `set(row_id)` equal for all `(variant, model)` of an AOI | `BLOCKER: VARIANT_COHORT_MISMATCH — <aoi> row_id` |
| `H03` | Grid-ID sets identical across variants | required | ″ for `grid_id` / `cell_id` | `BLOCKER: VARIANT_COHORT_MISMATCH — <aoi> grid_id` |
| `H04` | Label values identical across variants | required | `y_true` per `(aoi, cell_id)` constant over all variants and models | `BLOCKER: LABEL_INVARIANCE_VIOLATED — <aoi> <n> cells` |
| `H05` | Static feature values identical across variants | required | `cohort_inventory.removed_static_invariance_failure == 0` for every row; static invariance recorded PASS in each `local-downstream` stage state | `BLOCKER: STATIC_INVARIANCE_VIOLATED — <aoi> <column>` |
| `H06` | No duplicate rows | required | `cohort_inventory.duplicate_cell_ids == 0`; `(aoi, variant, model, cell_id)` unique in the OOF table | `BLOCKER: DUPLICATE_COHORT_ROW — <aoi> <n>` |
| `H07` | No missing variant | required | each AOI has exactly 3 rows in `cohort_inventory.csv` | `BLOCKER: MISSING_VARIANT — <aoi> <variant>` |
| `H08` | No partial AOI | required | all 3 AOIs complete through `summarize`; `stages/*.aois_processed` is the full set | `BLOCKER: PARTIAL_AOI — <aoi> incomplete at <stage>` |
| `H09` | Per-variant temporal-feature completeness recorded; attrition reconciles | required | `removed_missing_required_feature_union` present and ≥ 0; `initial − Σ(removals) == final_common_cohort_rows` for every row | `BLOCKER: COHORT_ACCOUNTING_INCONSISTENT — <aoi> <variant>` |

`H05` note: `removed_label_mismatch` and `removed_static_invariance_failure` are structurally
always zero, because a mismatch raises rather than removing. A **non-zero** value is itself
evidence of a code defect and must FAIL.

---

## 6. Folds — `F01`–`F06`

| ID | Description | Sev | Evidence source | Failure message |
|---|---|---|---|---|
| `F01` | One fold mapping per AOI, used by all variants | required | `fold_id` per `(aoi, cell_id)` identical across all variants and models in `oof_predictions.parquet` | `BLOCKER: FOLD_MAPPING_DRIFT — <aoi> <variant>` |
| `F02` | `fold_mapping_hash` identical across the variants of an AOI | required | `metrics.csv`, `fold_mapping.parquet` | `BLOCKER: FOLD_HASH_MISMATCH — <aoi>` |
| `F03` | Every required evaluation fold contains both classes | required | recompute per-fold class counts from `oof_predictions.parquet`; all > 0 | `BLOCKER: FOLD_CLASS_INFEASIBILITY — <aoi> fold <k>` |
| `F04` | Exactly one fold assignment per row | required | each `(aoi, cell_id)` has exactly one `fold_id ∈ [0,4]`; no `-1` | `BLOCKER: FOLD_ASSIGNMENT_INCOMPLETE — <aoi> <n> rows` |
| `F05` | Folds not optimised per variant | required | `fold_mapping_hash` constant within AOI; blocks never split across folds | `BLOCKER: FOLD_REOPTIMISED — <aoi> <variant>` |
| `F06` | `fold_mapping_hash` **differs** between AOIs | required | pairwise comparison across the 3 AOIs | `BLOCKER: FOLD_HASH_COLLISION — <aoi_a> == <aoi_b>` |

`F06` guards against an orchestration bug in which one AOI's cohort or fold mapping is
accidentally reused for another — a failure mode that no within-AOI check would catch.

---

## 7. Fits, OOF predictions and metrics — `M01`–`M17`

| ID | Description | Sev | Evidence source | Failure message |
|---|---|---|---|---|
| `M01` | Expected logical fit count correct | required | `summary.technical_status.expected_logical_fits == 3×3×2×5 == 90` | `BLOCKER: FIT_COUNT_MISMATCH — expected 90, recorded <n>` |
| `M02` | No duplicate logical fit | required | `duplicate_logical_fits == 0`; `(aoi,variant,model,fold)` unique | `BLOCKER: DUPLICATE_LOGICAL_FIT — <key>` |
| `M03` | No missing logical fit | required | `missing_logical_fits == 0`; all 90 keys present | `BLOCKER: MISSING_LOGICAL_FIT — <key>` |
| `M04` | No duplicate OOF row | required | `(aoi,variant,model,cell_id)` unique in `oof_predictions.parquet` | `BLOCKER: DUPLICATE_OOF_ROW — <key>` |
| `M05` | OOF coverage complete | required | for each `(aoi,variant,model)` the `cell_id` set equals that AOI's cohort exactly | `BLOCKER: OOF_COVERAGE_INCOMPLETE — <aoi> <variant> <model>: <n> missing` |
| `M06` | **Brier natural orientation correct** | required | `difference_natural == thermal_brier − baseline_brier` to `1e-9`; `orientation == "natural"`; `orientation_definition` non-empty | `BLOCKER: BRIER_ORIENTATION_ERROR — natural, <aoi> <variant>` |
| `M07` | **Brier oriented orientation correct** | required | `difference_oriented == baseline_brier − thermal_brier == −difference_natural` to `1e-9`; `orientation == "oriented"` | `BLOCKER: BRIER_ORIENTATION_ERROR — oriented, <aoi> <variant>` |
| `M08` | No unexpected fit | required | `unexpected_fits == 0`; no key outside the cross-product | `BLOCKER: UNEXPECTED_FIT — <key>` |
| `M09` | Prediction range valid | required | every `y_score ∈ [0,1]` and finite | `BLOCKER: PREDICTION_RANGE_INVALID — <aoi> <variant> <model>` |
| `M10` | `y_true` invariance | required | per `(aoi, cell_id)`, `y_true` constant across all 6 evaluations | `BLOCKER: LABEL_INVARIANCE_VIOLATED — <aoi>` |
| `M11` | Model and preprocessing contract identical to Manavgat | required | `config.model_configuration` and `config.feature_registry` byte-equal to the Manavgat `preregistration.json` counterparts (estimator, 4 baseline features, 10 thermal features, `n_splits=5`, seed 42, block size 2, `min_positives=30`, `calibration=None`, `adaptation=None`) | `BLOCKER: MODEL_CONTRACT_DRIFT — <field>: <manavgat> vs <this>` |
| `M12` | ROC-AUC recomputation matches | required | recompute from `oof_predictions.parquet`; \|Δ\| ≤ `1e-9` | `BLOCKER: METRIC_RECOMPUTE_MISMATCH — roc_auc <aoi> <variant> <model>` |
| `M13` | PR-AUC recomputation matches | required | ″ | `BLOCKER: METRIC_RECOMPUTE_MISMATCH — pr_auc …` |
| `M14` | Brier recomputation matches | required | ″ | `BLOCKER: METRIC_RECOMPUTE_MISMATCH — brier …` |
| `M15` | `thermal − baseline` arithmetic correct | required | recompute from `metrics.csv`; compare to `bootstrap_summary.point_estimate` | `BLOCKER: DIFFERENCE_ARITHMETIC_ERROR — thermal_contribution <aoi> <variant> <metric>` |
| `M16` | `shifted − canonical` arithmetic correct | required | ″ for both closure-change families | `BLOCKER: DIFFERENCE_ARITHMETIC_ERROR — closure_change <aoi> <variant> <metric>` |
| `M17` | Point estimate consistent with OOF rescore | required | `bootstrap_summary.point_estimate` derives from `metrics.csv`, **not** from `bootstrap_mean`; both columns present and distinct | `BLOCKER: POINT_ESTIMATE_INCONSISTENT — <aoi> <comparison> <metric>` |

`M11` is the check that makes "the frozen contract was preserved" falsifiable rather than
aspirational: it compares against the *already-PASSed Manavgat artefact*, not against prose.

---

## 8. Bootstrap — `B01`–`B11`

| ID | Description | Sev | Evidence source | Failure message |
|---|---|---|---|---|
| `B01` | Paired draws used | required | `bootstrap_replicates.parquet`: for a given `(aoi, replicate_id)` all series share one `draw_plan_id` | `BLOCKER: BOOTSTRAP_NOT_PAIRED — <aoi> replicate <r>` |
| `B02` | Same draw plan across comparison members | required | `draw_plan_hash` constant within AOI across all comparison families | `BLOCKER: DRAW_PLAN_MISMATCH — <aoi> <comparison>` |
| `B03` | **No refit inside the bootstrap** | required | `stages/compare.json.model_fit == false`; no model artefact referenced by the compare stage; replicate estimates reproduce from stored OOF scores under the recorded block resample | `BLOCKER: BOOTSTRAP_REFIT_DETECTED — <aoi>` |
| `B04` | Requested replicate count correct | required | `requested_replicates == 1000 == core.config.STEP8C_N_BOOTSTRAP` | `BLOCKER: REPLICATE_COUNT_MISMATCH — requested <n>` |
| `B05` | Valid replicate count correct | required | `valid_replicates == len(replicate rows for that series)` | `BLOCKER: REPLICATE_COUNT_MISMATCH — valid <n> vs <rows>` |
| `B06` | Invalid replicate count correct | required | `invalid_replicates == requested − valid` | `BLOCKER: REPLICATE_COUNT_MISMATCH — invalid <n>` |
| `B07` | `valid + invalid == requested` | required | arithmetic over every summary row | `BLOCKER: REPLICATE_ACCOUNTING_UNTRUTHFUL — <aoi> <series>` |
| `B08` | CI quantile arithmetic correct | required | recompute 2.5 / 97.5 percentiles from the replicate series; \|Δ\| ≤ `1e-9`; `ci_low <= ci_high`; `interval_method == "percentile"` | `BLOCKER: CI_ARITHMETIC_ERROR — <aoi> <series>` |
| `B09` | Seed and draw-plan hash recorded | required | `seed == 42`; `draw_plan_hash` present, 64 hex, differs across AOIs | `BLOCKER: BOOTSTRAP_PROVENANCE_MISSING — <aoi>` |
| `B10` | No missing replicate | required | `replicate_id` covers `[0, valid_replicates)` with no gap for every series | `BLOCKER: MISSING_REPLICATE — <aoi> <series> id <r>` |
| `B11` | No duplicate replicate | required | `(aoi, comparison_family, variant, metric, replicate_id)` unique | `BLOCKER: DUPLICATE_REPLICATE — <key>` |
| — | Block count sufficient | required (folded into `B09`) | `block_count >= 2` for every AOI | `BLOCKER: INSUFFICIENT_BLOCKS — <aoi> <n>` |

---

## 9. Synthesis and wording — `Y01`–`Y08`

| ID | Description | Sev | Evidence source | Failure message |
|---|---|---|---|---|
| `Y01` | **No pooled inference** | required | `four_region_synthesis.csv` contains none of: `pooled_estimate`, `pooled_ci_low`, `pooled_ci_high`, `meta_analytic_estimate`, `combined_p`, `heterogeneity`, `i_squared`, `weight`, `n_total_across_aois`. Every value re-derives from exactly one AOI's `bootstrap_summary.csv`. `summary.scientific_summary.pooled_inference == false`. | `BLOCKER: POOLED_INFERENCE_DETECTED — <column/row>` |
| `Y02` | Four-region synthesis is descriptive | required | every row has `descriptive_only == true`; `cross_aoi_comparable == false` for `pr_auc` and `brier`; `report.md` §8 carries the no-pooling statement | `BLOCKER: SYNTHESIS_NOT_DESCRIPTIVE — <row>` |
| `Y03` | Evia framed as different-regime control | required | `regional_summary.csv`: `evia_2021_extended.aoi_role == "different_regime_control"`; `regime_note` contains `different-regime control`, `high-prevalence sensitivity region`, `not an equal-prevalence fourth validation region`; `report.md` §9 present | `BLOCKER: EVIA_FRAMING_MISSING` |
| `Y04` | Evia **not** presented as an equal-prevalence fourth validation region | required | no artefact describes Evia as `fourth validation region`, `equal-prevalence`, `equivalent AOI`, or lists it without its regime qualifier; the prevalence table (0.0385 / 0.0707 / 0.0706 / 0.2882) is present | `BLOCKER: EVIA_FRAMING_VIOLATION — <location>` |
| `Y05` | Technical PASS separated from scientific support | required | `summary.json` has distinct `technical_status` and `scientific_summary` keys; no field merges them; `report.md` §12 present | `BLOCKER: VERDICT_CONFLATION` |
| `Y06` | No prohibited significance language | required | union scan (§9.1) of `report.md` and every prose field of every JSON artefact | `BLOCKER: FORBIDDEN_LANGUAGE — "<phrase>" at <location>` |
| `Y07` | No causal claim | required | scan for `causal`, `causes`, `because the closure date`, `due to the closure date`, `mechanism` used as a claim | `BLOCKER: CAUSAL_CLAIM — "<phrase>" at <location>` |
| `Y08` | No best/optimal window claim | required | scan for `optimal`, `best window`, `recommended window`, `should close`, `operationally validated`, `leakage eliminated`, `proven` | `BLOCKER: OPTIMALITY_CLAIM — "<phrase>" at <location>` |

### 9.1 The union forbidden list

The existing `FORBIDDEN_COMPARE_PHRASES` (`src/window_closure_sensitivity.py:8863`) is
**necessary but not sufficient** — it lacks seven phrases the task requires. The validator
enforces the union of all three sets:

```
# from FORBIDDEN_COMPARE_PHRASES (existing, keep all)
statistically significant | significant difference | non-significant | insignificant
p-value | p value | hypothesis test | equivalent | equivalence
unchanged | stable | robust

# added by this analysis (currently NOT guarded anywhere)
statistically proven | proven | causal | operationally validated
leakage eliminated | optimal | best window

# from FOREIGN_FACTOR_PHRASES (existing, keep)
compositing method is the only
```

Scanning follows `assert_compare_wording` (line 9020): walk the payload, collect prose from
`_PROSE_FIELD_TOKENS` fields, lower-case, substring match. Applied to `report.md`,
`summary.json`, `regional_summary.csv`, `four_region_synthesis.csv` and every
`stages/*.json`.

**Permitted vocabulary** (must not be flagged): `bootstrap-supported`,
`interval excludes zero`, `interval includes zero`, `uncertainty remains`, `point estimate`,
`descriptive`, `direction-dependent`, `under this frozen design`.

---

## 10. Operational integrity — `O01`–`O16`

| ID | Description | Sev | Evidence source | Failure message |
|---|---|---|---|---|
| `O01` | Earth Engine only in the export stage | required | `stages/*.json`: `gee_queries_run` and `gee_exports_run` are `true` only for `export` | `BLOCKER: GEE_OUTSIDE_EXPORT — <stage>` |
| `O02` | Local stages make no GEE call | required | `local-downstream`, `cohort-feasibility`, `fit`, `compare`, `summarize` all report `gee_* == false`; `ee` is imported only by `production_predictor_engine` | `BLOCKER: GEE_IN_LOCAL_STAGE — <stage>` |
| `O03` | Dry-run wrote nothing | required | dry-run state diff: zero created, modified or deleted stage-owned paths | `BLOCKER: DRY_RUN_NOT_READ_ONLY — <n> paths changed` |
| `O04` | Dry-run flags all false | required | `files_written`, `gee_queries_run`, `gee_exports_run`, `model_fit`, `bootstrap_run` all `false`, aggregated over all AOIs | `BLOCKER: DRY_RUN_SIDE_EFFECT — <flag> true` |
| `O05` | Resume is hash-bound | required | every reused stage records matching `config_hash`, input hashes and output manifest hashes | `BLOCKER: UNSAFE_RESUME — <stage> <role>` |
| `O06` | Force uses quarantine | required | if `--force` was used, `quarantined_paths` non-empty and each matches `<namespace>/_quarantine/<timestamp>_<reason>/`; nothing was deleted | `BLOCKER: FORCE_WITHOUT_QUARANTINE — <stage>` |
| `O07` | Namespace isolation | required | every `manifest.files[].relative_path` resolves inside `outputs/diagnostics/multi_region_window_closure/<analysis_id>/`; no write to `outputs/experiments/**` or another diagnostics namespace | `BLOCKER: NAMESPACE_ESCAPE — <path>` |
| `O08` | No canonical overwrite | required | canonical Step8A, DEM, land-cover and label hashes identical before and after the full run; `window_closure_sensitivity/manavgat_2021/**` untouched (mtime + hash) | `BLOCKER: CANONICAL_OVERWRITE — <path>` |
| `O09` | Manifest complete | required | every file on disk (except `manifest.json` itself and `manifest.sha256`) appears in `files[]`; `output_file_count == len(files)`; `output_total_bytes == Σ size_bytes` | `BLOCKER: MANIFEST_INCOMPLETE — stray file <path>` / `unrecorded <path>` |
| `O10` | All required output files present | required | the 19 entries of `OUTPUT_SCHEMA.md` §1 marked required exist and are > 0 bytes | `BLOCKER: MISSING_REQUIRED_OUTPUT — <file>` |
| `O11` | Output hashes match the manifest | required | recompute `sha256` for every `files[]` entry | `BLOCKER: OUTPUT_HASH_MISMATCH — <path>` |
| `O12` | Stage states complete; no partial AOI or variant | required | all 7 `stages/*.json` exist; each has `status`, `aois_processed` (3), `variants_processed` (3) | `BLOCKER: PARTIAL_STAGE — <stage>: aois <list>` |
| `O13` | No overall PASS while any stage FAILs or is partial | required | `overall_status == PASS` ⇒ every stage `PASS` and every set complete | `BLOCKER: INCONSISTENT_OVERALL_STATUS` |
| `O14` | Free disk space sufficient at actual-run start | required | pre-flight: free bytes ≥ 3 × expected total (≈ 120 GB) | `BLOCKER: INSUFFICIENT_DISK — <avail> < <required>` |
| `O15` | Every check emitted non-empty evidence (meta-check) | required | scan `validation_report.checks[].evidence` | `BLOCKER: EVIDENCE_MISSING — check <id>` |
| `O16` | Resume never reused a stage recorded as FAIL, SKIP or partial | required | cross-check reuse decisions against recorded stage status | `BLOCKER: RESUME_FROM_INVALID_STAGE — <stage> status <s>` |

`O08` note: the Manavgat *window-closure* namespace is as protected as the canonical
experiment outputs. `manavgat_2021` is read-only in both senses.

---

## 11. CLI contract

Following the four existing validators:

```bash
python scripts/validate_multi_region_window_closure.py \
  --analysis-id <id> \
  --mode dry-run --log logs/multi_region_window_closure_dryrun.log

python scripts/validate_multi_region_window_closure.py \
  --analysis-id <id> --mode actual
```

Output: ordered `[PASS]` / `[FAIL]` lines, then

```
TECHNICAL STATUS: PASS|FAIL
SCIENTIFIC-CONTRACT STATUS: PASS|FAIL
NAMESPACE / PROVENANCE SAFETY: PASS|FAIL
OVERALL STATUS: PASS|FAIL
```

Exit code `0` on overall PASS, `1` otherwise. `validation_report.json` is written into the
analysis namespace. The validator itself **never** starts a stage, an export or a fit.

---

## 12. Check-count summary

| Group | IDs | Count | Mandated items covered |
|---|---|---|---|
| Scope | `S01`–`S08` | 8 | 1–8 |
| Date contract | `D01`–`D10` | 10 | 9–18 |
| Canonical integrity | `C01`–`C07` | 7 | 19–23 (+2 added) |
| Cohort | `H01`–`H09` | 9 | 24–31 (+1 added) |
| Folds | `F01`–`F06` | 6 | 32–36 (+1 added) |
| Fits / OOF / metrics | `M01`–`M17` | 17 | 37–53 |
| Bootstrap | `B01`–`B11` | 11 | 54–64 |
| Synthesis / wording | `Y01`–`Y08` | 8 | 65–72 |
| Operational | `O01`–`O16` | 16 | 73–84 (+3 added, one item split) |
| **Total** | | **92** | **84 / 84** |

All 84 mandated checks are covered. `required` severity: 90 of 92 (`C07` and one advisory
block-count note are advisory).
