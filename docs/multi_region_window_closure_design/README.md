# Multi-Region Window-Closure Sensitivity — Design Package

**Schema (planned):** `multi_region_window_closure.v1`
**Namespace (planned):** `outputs/diagnostics/multi_region_window_closure/<analysis_id>/`
**Design date:** 2026-08-04
**Design status:** `DESIGN READY FOR IMPLEMENTATION`
**Stage of work:** DESIGN ONLY — no export, no fit, no bootstrap, no implementation code was produced.

---

## 1. Purpose

Extend the completed and technically-PASSed Manavgat-2021 **window-closure sensitivity**
analysis to three further AOIs under the *same frozen scientific contract*, and answer,
separately for each AOI:

> When the predictor window is closed 7 and 14 days before its canonical closing date,
> is the incremental contribution of the thermal model over the baseline model preserved?

No pooled inference is produced. The four-region view is **descriptive only**.

---

## 2. Scope

### 2.1 New actual AOIs (exactly three)

| AOI | Role in this analysis |
|---|---|
| `bejis_2022` | New actual AOI (Spain, Mediterranean transfer wildfire) |
| `mugla_2021` | New actual AOI (Turkey, same-country same-year transfer wildfire) |
| `evia_2021_extended` | New actual AOI **and different-regime control** (Greece, high prevalence) |

### 2.2 Read-only reference

| AOI | Treatment |
|---|---|
| `manavgat_2021` | Read-only. Never recomputed, never overwritten. Its existing PASSed contract and results are cited as reference. |

### 2.3 Excluded — hard exclusion

| AOI | Reason |
|---|---|
| `evia_2021` | The **superseded narrow** North Evia AOI. Not an analysis target; may not enter the export plan, the fit plan, the bootstrap plan, or the four-region synthesis. `evia_2021_extended` is the canonical Evia entry throughout. |

`evia_2021` remains registered and unchanged in `core/regions.py`. This analysis simply
never reads it. A dedicated validator check enforces its absence.

---

## 3. Frozen decisions (not revisitable at implementation time)

| Element | Frozen value | Verified from |
|---|---|---|
| Population | `burnable_tree_shrub_grass` | `src/window_closure_sensitivity.py:81` |
| Models | `baseline`, `thermal` | `MODEL_FAMILIES`, line 7197 |
| Variants | `canonical`, `close_7d_earlier`, `close_14d_earlier` | `DEFAULT_SHIFTS = (0, 7, 14)`, line 84; `variant_id()`, line 245 |
| Metrics | ROC-AUC, PR-AUC, Brier | `MODEL_METRICS`, line 7200 |
| Estimator | `random_forest` | `PRIMARY_MODEL`, line 82 |
| Folds | 5, spatial-block grouped, seed 42, block size 2 cells | `core/config.py:554-557` |
| Bootstrap | 1000 replicates, seed 42, percentile 2.5/97.5, block-paired, no refit | `core/config.py:574-577`; `multi_variant_block_bootstrap`, line 1316 |
| Cohort rule | Exact `cell_id` intersection across all three variants (Structure A) | `build_model_common_cohort`, line 7734 |
| Fold rule | ONE shared assignment reused by all six evaluations per AOI | `build_shared_spatial_folds`, line 7895 |
| Inference unit | Per AOI. No pooling. | Section 6 of `SCIENTIFIC_CONTRACT.md` |
| Window shift | Both ends move; duration invariant; label/event/gate frozen | `build_window_variants`, line 311 |

---

## 4. Document map

| Document | Contents |
|---|---|
| `README.md` | This file — scope, frozen decisions, status, gate |
| `SCIENTIFIC_CONTRACT.md` | Question, estimands, orientations, model/fold/cohort/bootstrap contracts, Evia framing, prohibited claims |
| `REPOSITORY_INVENTORY.md` | Every relevant source/test/schema/validator/output path, reuse decisions, dependency audit, missing components |
| `WINDOW_DATE_AUDIT.md` | Exact ISO dates for all 4 AOIs × 3 variants, inclusivity semantics, MODIS season policy, off-by-one analysis |
| `EXPORT_FEASIBILITY.md` | Static/temporal feature classification, export plan, request counts, runtime and storage estimates |
| `COHORT_FEASIBILITY.md` | Cohort contract, per-AOI feasibility evidence, fold feasibility, gates |
| `OUTPUT_SCHEMA.md` | Every output file: grain, keys, columns, types, row-count formulas, manifest, stage-state |
| `VALIDATOR_CHECKLIST.md` | All 84 required checks with severity, evidence and failure messages |
| `IMPLEMENTATION_PLAN.md` | Source changes, stage-by-stage order, test order, dry-run/actual/validation order, rollback |

---

## 5. Current design status

**`DESIGN READY FOR IMPLEMENTATION`**

Every item of the Section-24 acceptance gate is satisfied:

| Gate item | Status | Evidence |
|---|---|---|
| Manavgat implementation and artefacts found | PASS | `src/window_closure_sensitivity.py` (10,697 lines), 4 validators, 4 test modules, full output namespace |
| Manavgat frozen contract verified from source | PASS | `SCIENTIFIC_CONTRACT.md` §2–§8, every value cited to a line number or artefact |
| Exact canonical date sources resolved | PASS | `core/regions.py:280-465` `EXPERIMENTS` registry is the single source |
| Exact shifted dates computable for 3 new AOIs | PASS | `WINDOW_DATE_AUDIT.md` §3, all 12 AOI×variant rows |
| Four canonical AOI hashes verifiable | PASS | All four re-verified byte-for-byte — see §6 below |
| Old Evia definitively excluded | PASS | Check `S05` in `VALIDATOR_CHECKLIST.md` |
| Static/temporal feature split complete | PASS | `EXPORT_FEASIBILITY.md` §2 |
| Export plan complete for 6 shifted scenarios | PASS | `EXPORT_FEASIBILITY.md` §4 |
| Common-cohort approach determined | PASS | Structure **A** (exact intersection) — `COHORT_FEASIBILITY.md` §1 |
| Fold feasibility gate defined | PASS | `COHORT_FEASIBILITY.md` §4 |
| Model contract determined | PASS | `SCIENTIFIC_CONTRACT.md` §5 |
| Bootstrap contract determined | PASS | `SCIENTIFIC_CONTRACT.md` §7 |
| Output schema complete | PASS | `OUTPUT_SCHEMA.md` |
| Validator checklist complete | PASS | `VALIDATOR_CHECKLIST.md`, 84 checks |
| Fit/task/runtime/storage accounting complete | PASS | `EXPORT_FEASIBILITY.md` §5–§7 |
| Canonical overwrite risk closed | PASS | `assert_plan_owned_targets`, `assert_jobs_inside_variant_namespace`, frozen-hash before/after guards |
| Dry-run proven read-only | PASS | Snapshot-diff mechanism, `EXPORT_FEASIBILITY.md` §8.1 |
| Resume hash-bound | PASS | `IMPLEMENTATION_PLAN.md` §5 |
| Force quarantine designed | PASS | `IMPLEMENTATION_PLAN.md` §6 |
| Pooled inference prohibited | PASS | Checks `Y01`–`Y02` |
| Evia wording mandated | PASS | Checks `Y03`–`Y04` |
| No unresolved critical blockers | PASS | §7 below |

---

## 6. Canonical hash status — ALL FOUR VERIFIED

The four SHA-256 anchors are the digests of
`outputs/experiments/<aoi>/step8a/step8a_500m_modeling_dataset.parquet`.
Each was recomputed from the artefact on disk and matched exactly.

| AOI | Expected = Recomputed | Match |
|---|---|---|
| `manavgat_2021` | `054a1961fc0582a33d36413263668b63074b21ae8b03d12269b6e228787f3439` | ✅ |
| `bejis_2022` | `3dec785a7d8e31db2d67ed283546bbfbca1559f56df46663488d0afc24d9e393` | ✅ |
| `mugla_2021` | `c4ab107db2207f9f20775ccc0b3bf39381173fd07d4e82f6821ce7f40be7db8e` | ✅ |
| `evia_2021_extended` | `bdce859cf482f575d0f273174b157f47efd61779953fdd23d9486c5face5e553` | ✅ |

The identification is independently corroborated: the Manavgat value appears as
`frozen_input_sha256.canonical_step8a` in the already-frozen
`outputs/diagnostics/window_closure_sensitivity/manavgat_2021/config/frozen_input_inventory.json`,
whose `inventory.canonical_step8a.path` names exactly that parquet file. The hashes are
therefore anchored to a **content digest of a named artefact**, not to a copied literal.

Reproduction command in `WINDOW_DATE_AUDIT.md` §6.

No `CANONICAL_HASH_DRIFT`.

---

## 7. Blocker summary

**Critical blockers: 0.**

Non-blocking warnings carried into implementation (full treatment in
`EXPORT_FEASIBILITY.md` §9 and `COHORT_FEASIBILITY.md` §6):

| ID | Warning | Handling |
|---|---|---|
| `W1` | MODIS fixed summer-month filter (months 6–9) clips shifted windows by AOI-specific amounts (Manavgat 7/14 d, Muğla 7/14 d, Evia-ext 3/10 d, Bejís 0/0 d). | Pre-existing, transparently reported production behaviour. Manavgat's frozen contract already measures the closure date *together with* this interaction. Must be reported per AOI, never silently corrected. |
| `W2` | Bejís is the only AOI with **zero** MODIS clipping, so its shifted variants are not mechanistically identical to the other three. | Report explicitly in `report.md`; do not present the four AOIs as an interchangeable set. |
| `W3` | Muğla is ~3.0× Manavgat's cell count; export and storage dominate the run. | Sized in `EXPORT_FEASIBILITY.md` §5–§7; disk headroom is ample (884 GB free vs ~38 GB expected). |
| `W4` | Evia-extended prevalence is 0.288 vs 0.039–0.071 elsewhere. | Mandatory different-regime framing; PR-AUC is prevalence-dependent and is never compared across AOIs. |
| `W5` | `requirements.txt` comment claims `geographiclib` is absent from the lock; it is present as `geographiclib==2.1`. Stale comment only — no version conflict exists. | Report only. Out of scope for this task; do not edit. |
| `W6` | The existing `FORBIDDEN_COMPARE_PHRASES` guard does **not** ban `proven`, `causal`, `optimal`, `best window`, `operationally validated`, `leakage eliminated`. | The new validator must enforce the **union** of the existing list and the task's list. See `VALIDATOR_CHECKLIST.md` §8. |
| `W7` | Exports are **synchronous `getPixels` downloads**, not Earth Engine *batch Drive tasks*. | The design substitutes request-level provenance for Drive task-ID provenance. See `EXPORT_FEASIBILITY.md` §4.4. |

---

## 8. Gate for moving to implementation

Implementation may begin **only** when all of the following hold at that moment:

1. `git status --short` shows no modification outside `docs/multi_region_window_closure_design/`
   other than the pre-existing unrelated `docs/` churn noted in §9.
2. The four canonical hashes still reproduce (rerun the §6 command).
3. `pytest tests/test_window_closure_sensitivity.py tests/test_window_closure_local_downstream.py tests/test_window_closure_model.py tests/test_window_closure_compare.py`
   is green on the current worktree.
4. A dry-run of the existing per-AOI runner succeeds for each of the three new AOIs
   and reports `prerequisites_ready: true` — this is the cheapest end-to-end proof that
   the frozen-input contract is satisfiable per AOI *before* any orchestration code exists.
5. `IMPLEMENTATION_PLAN.md` §2 component list is reviewed and approved.

---

## 9. Repository hygiene note

At the start of this design task the worktree carried unrelated pre-existing changes: a
modified `.gitignore`, deleted `docs/advisor_results_review_2026-07-31/**` and two deleted
`docs/*.pdf`, plus untracked `docs/Hocaya_Cevap_Maili_2026-08-03.pdf` and
`docs/advisor_final_numerical_package/`. **None of these were touched by this task.**

This design package created exactly one new directory,
`docs/multi_region_window_closure_design/`, and wrote nine markdown files into it. No source
file, config file, canonical output or existing analysis namespace was read-write accessed.

**Commit `0f9f691` was not created by this task.** While the package was being written, an
external commit (author `emrehann17`, message `.`, 2026-08-04 09:36:58 +0300) swept up the
pre-existing worktree changes listed above **and** this `README.md` — the only design document
that existed at that moment. The remaining eight documents are untracked. Nothing was
rewritten, rebased or force-pushed by this task, and the committed `README.md` is byte-identical
to the file on disk.

Consequence for the §8 gate: item 1 should now be read as *"no modification outside
`docs/multi_region_window_closure_design/`"* relative to `0f9f691`, which the current
`git status --short` satisfies.
