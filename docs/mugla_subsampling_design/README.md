# Muğla Subsampling — design freeze

**Schema:** `mugla_subsampling.v1`
**Namespace:** `outputs/diagnostics/mugla_subsampling/<analysis_id>/`
**Status:** design frozen, not implemented, not run.
**Frozen on:** 2026-08-03.

This directory is the complete, unresolved-decision-free specification of the
Muğla sample-size sensitivity analysis. Nothing in it has been executed: no
model was fitted, no scientific output was produced, no GEE call was made, no
production artifact was touched, no commit was created. Every number quoted
here was obtained by read-only inspection of frozen on-disk artifacts.

## The question

Muğla's primary modeling population is **41,730** cells; Manavgat's is
**20,511**. When the Muğla modeling population is reduced to exactly Manavgat's
cell count, how do these three quantities move relative to their full-Muğla
references?

1. **Within-Muğla** 5-fold spatial OOF performance.
2. **Muğla → Manavgat / Bejís** raw transfer (Muğla as *source*).
3. **Manavgat / Bejís → Muğla** target evaluation (Muğla as *target*).

This is a **total-sample-size sensitivity analysis and nothing else.** Muğla's
prevalence is preserved; Muğla's positive count is *not* equalised to
Manavgat's; no predictor or label structure is altered; no causal
decomposition and no operational deployment claim is made.

## The documents

| File | Contents |
|---|---|
| `SCIENTIFIC_CONTRACT.md` | The frozen scientific decisions: population, arms, subsampling rule, metric orientation, interval vocabulary, interpretation limits. |
| `REPOSITORY_INVENTORY.md` | Exact files and symbols reused, with line numbers and a REUSE / PATTERN / REFERENCE / NOT USED verdict for each. |
| `SAMPLING_FEASIBILITY.md` | The read-only feasibility computation: strata, capacities, Hamilton allocation, prevalence drift, fold composition, runtime/memory/disk. |
| `OUTPUT_SCHEMA.md` | Every produced file, every column, every stage. |
| `VALIDATOR_CHECKLIST.md` | The independent validator's check list with expected values. |
| `IMPLEMENTATION_PLAN.md` | Build order, new files, test plan, fit registry. |

## Headline frozen facts

| Quantity | Value |
|---|---|
| Target sample size | **20,511** rows per repeat |
| Repeats | **20** |
| Muğla full population | 41,730 (2,911 positive / 38,819 negative) |
| Sampling fraction | 20,511 / 41,730 = 0.491528876… |
| Spatial strata | 636 (576 blocks × label, at 10-cell ≈ 5 km) |
| Positives per subsample | **1,438** — identical in every repeat |
| Subsample prevalence | 0.07010872 vs full 0.06975797 (+0.503 % relative) |
| Unique model fits | **240** (200 within + 40 source + 0 target) |
| Estimated runtime | ≈ 7–10 min single-process wall clock |

Both the sampling arithmetic and the fold composition are **fully deterministic
and repeat-invariant**; only *which* cells fill each stratum varies across the
20 repeats. See `SAMPLING_FEASIBILITY.md` §3–§5.

---

MUĞLA SUBSAMPLING DESIGN READY FOR IMPLEMENTATION
