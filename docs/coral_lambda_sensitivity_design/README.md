# CORAL λ Sensitivity — design freeze

**Schema:** `coral_lambda_sensitivity.v1`
**Namespace:** `outputs/diagnostics/coral_lambda_sensitivity/<analysis_id>/`
**Status:** design frozen, **two blockers open** (see §Blockers), not implemented, not run.
**Frozen on:** 2026-08-03.

This directory specifies how sensitive the existing Step10
`coral_after_regionwise_zscore` result is to the CORAL covariance
regularisation parameter λ.

Nothing here has been executed: no code was written, no model fitted, no
bootstrap run, no production artifact touched, no GEE call made, no commit
created. Every number quoted was obtained by read-only inspection and
read-only linear algebra on frozen artifacts.

## What this is not

- **Not** hyperparameter selection. No "best λ" or "optimal λ" is chosen, named
  or implied; there is no argmax over target performance anywhere.
- **Not** target-label tuning. λ is a predeclared grid; the target label is
  loaded only at final evaluation and bootstrap scoring.
- **Not** a change to the canonical λ. `core/config.py:697` stays `1e-5` and
  `STEP10_CORAL_LAMBDA` is never mutated.
- **Not** a modification of any Step10 artifact or preregistration file. They
  are bound read-only by sha256.

## The documents

| File | Contents |
|---|---|
| `SCIENTIFIC_CONTRACT.md` | Frozen decisions: scope, λ grid, held-fixed contract, firewall, metric orientation, bootstrap, reproduction gate, predeclared interpretation thresholds. |
| `CORAL_FORMULA_AUDIT.md` | Exhaustive audit of the existing CORAL implementation and the exact meaning of λ, with file:line for every claim. |
| `REPOSITORY_INVENTORY.md` | Exact files and symbols reused, with a REUSE / PATTERN / REFERENCE / NOT USED verdict for each. |
| `REFERENCE_ARTIFACTS.md` | The four resolved Step10 references, their digests, the duplicate-artifact problem and its frozen resolution rule. |
| `NUMERICAL_FEASIBILITY.md` | Measured covariance spectra, λ=0 safety, the reproduction-gate feasibility measurement, fit count, runtime/memory/disk. |
| `OUTPUT_SCHEMA.md` | Every produced file, every column, every stage, plus dry-run/resume/force semantics. |
| `VALIDATOR_CHECKLIST.md` | 91 checks across 12 groups. |
| `IMPLEMENTATION_PLAN.md` | Build order, new files, test contract. |

## Headline frozen facts

| Quantity | Value |
|---|---|
| λ semantics | additive ridge `λ·I` on **both** `Cs` and `Ct`, `core/step10_shared.py:192–193` |
| Canonical λ | `1e-5`, `core/config.py:697` |
| λ grid | `0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1` (9 values, ascending) |
| Directions | 4 (2 primary muğla↔bejís, 2 secondary muğla↔manavgat) |
| Model families | 2 (baseline d=3 numeric, thermal d=9 numeric) |
| **Maximum unique scientific fits** | **72** = 4 × 2 × 9, with no reuse available |
| Audit fits (reproduction gate) | 8, counted separately |
| Bootstrap | 1000 replicates, seed 42, target `spatial_block_id`, no refit |
| Smallest pre-ridge eigenvalue anywhere | **1.713164e-03** — λ=0 is numerically safe |
| Eigenvalue floor (1e-12) ever binding | **never**, 0 of 72 cells |
| Estimated runtime | ≈ 40–55 min (bootstrap dominates) |
| Estimated disk | ≈ 35–40 MB |

## Blockers

Two, both found by measurement, both requiring a contract decision before
implementation starts. They are set out in full in
`NUMERICAL_FEASIBILITY.md` §6 and `SCIENTIFIC_CONTRACT.md` §7.2.

### B-1 — the ≤1e-12 metric reproduction tolerance is unattainable

§6 of the task requires the λ=1e-5 refit to reproduce the stored Step10 metrics
to ≤1e-12. It cannot, and neither can any other execution of the canonical
pipeline.

Both pair orderings of each pair were run, so **two artifacts exist that are
two executions of the identical canonical pipeline on identical inputs**. They
differ by up to **4.867e-08 in ROC-AUC** and 4.441e-16 in probability, because
`RandomForestClassifier(n_jobs=-1)` sums per-tree probabilities in
thread-scheduling order and the resulting ~1 ULP differences flip near-tied
ranks (one flip moves ROC-AUC by exactly `1/(n_pos·n_neg)` = 8.85e-09 for a
muğla target).

**Proposed resolution** — a two-tier gate. Tier 1 recomputes metrics *from the
persisted probability vectors* and demands ≤1e-12; measured deviation
**5.551e-17**, so this holds with four orders of margin and validates the metric
layer exactly. Tier 2 is the refit, at ≤1e-12 on probabilities (the tolerance
the task asked for, which **does** hold) and ≤1e-06 **and** ≤8×rank-quantum on
ROC/PR.

### B-2 — Step10 contains no Brier score

§8 requires Brier for candidate and reference values.
`core/step10_shared.compute_threshold_free_metrics` returns only ROC-AUC and
PR-AUC; neither `step10_metrics.csv` nor `step10_bootstrap_replicates.parquet`
has a Brier column for any method. The only Brier values in the Step10 outputs
are Step9B raw-transfer values carried in as provenance.

**Proposed resolution** — recompute all three Brier references from the
persisted probability vectors (exact, and verified by the fact that the same
recomputation reproduces the stored ROC/PR to 5.6e-17), flag every Brier row
`reference_source = "recomputed_from_persisted_probabilities"`, and compute
Brier bootstrap replicates with no model refit under the reproduced draws.

**The λ semantics themselves are NOT a blocker.** They were located
unambiguously and are frozen in `CORAL_FORMULA_AUDIT.md`.

## One thing to watch during implementation

`_sym_matrix_power` clips eigenvalues at `1e-12` **after** the λ ridge
(`core/step10_shared.py:183`). That is a pre-existing second regulariser, and it
means λ=0 would not by itself guarantee an unregularised CORAL. On these data
the clip never binds — the smallest eigenvalue anywhere is 1.7e-03, nine orders
above the floor — so λ=0 is a genuine unregularised diagnostic here. The run
must **record** the floor-activation flag per cell so that this is evidenced
rather than assumed.

---

CORAL LAMBDA SENSITIVITY DESIGN READY FOR IMPLEMENTATION
