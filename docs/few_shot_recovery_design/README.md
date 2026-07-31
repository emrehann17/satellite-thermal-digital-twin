# Few-Shot Recovery Curve — Design Freeze

**Status:** design frozen, not implemented, not run.
**Date:** 2026-08-02
**Schema:** `few_shot_recovery.v1`
**Namespace:** `outputs/diagnostics/few_shot_recovery/<analysis_id>/`

This directory freezes the design for the few-shot recovery curve analysis
(item 9 of the advisor list). It contains no results. No model was fitted, no
Earth Engine call was made, no production artifact was modified and no commit
was created while producing it.

The prior status record for this item is
`docs/advisor_results_review_2026-07-31/06_few_shot_recovery_status.md`, which
recorded `NOT STARTED` and verified that no artifact, namespace, runner,
registry entry or design document existed. That remains true: this directory is
design only.

## Documents

| File | Contents |
|---|---|
| `SCIENTIFIC_CONTRACT.md` | The question, the claim boundary, and every frozen scientific decision |
| `REPOSITORY_INVENTORY.md` | Exact files and symbols to be reused, with what each supplies |
| `BLOCK_BUDGET_FEASIBILITY.md` | Measured block inventory and the common feasible budget set |
| `OUTPUT_SCHEMA.md` | `few_shot_recovery.v1` file-by-file and column-by-column contract |
| `VALIDATOR_CHECKLIST.md` | The 42 validator checks |
| `IMPLEMENTATION_PLAN.md` | Stage contract, module layout, test contract, runtime budget |

## The analysis in one paragraph

For each of the six directed transfer pairs among `manavgat_2021`,
`bejis_2022` and `mugla_2021`, on population `burnable_tree_shrub_grass`, a
source-trained model is refitted with the full source training population plus
`k` labeled target spatial blocks, for `k ∈ {0, 1, 2, 4, 8, 16, 32}`, and
evaluated only on held-out target blocks. The curve reports what fraction of
the gap between zero-shot raw transfer (`k=0`) and the target-only
within-region ceiling is recovered at each budget. Recovery fraction is signed
and unclipped. Uncertainty is reported as a repeat-based **selection
interval**, never as a confidence interval, and no p-value is produced.

## Two forced decisions, stated up front

Both are consequences of repository data, not preferences. Each is argued in
full in `SCIENTIFIC_CONTRACT.md` §3 and §8.

1. **Outer evaluation blocks are 10-cell (~5 km), not Step8B's canonical
   2-cell (~1 km) blocks.** A 2-cell block holds a median of 4 cells, so a
   "labeled block" would not be a meaningful unit of labeling effort, and
   adaptation blocks would sit immediately adjacent to evaluation blocks. The
   repository already carries a canonical ~5 km convention
   (`step8_large_block_robustness.assign_large_blocks(df, 10)`,
   `NOMINAL_SCALES[10] == "approximately_5_km"`) used with
   `strict_folds=True` on this exact population. That convention is adopted
   verbatim. This is the fallback that the task specification pre-authorised
   for the case where the canonical Step8 folds are not directly suitable.

2. **No new bootstrap is designed, and existing raw-transfer bootstrap
   artifacts are not reused.** Step9C/Step10 bootstrap replicates resample
   2-cell blocks; they are not comparable to a 10-cell evaluation frame. The
   frozen 10-cell ceiling artifacts *are* comparable and are reused as a
   reproduction check (and, for two of three targets, as a ceiling interval).
   The absence of a comparable raw-transfer interval is recorded as an
   explicit limitation rather than repaired with a newly invented bootstrap.

## What this analysis is not

It is not an operational deployment claim, not active learning, not a causal
decomposition, and not target-label-free adaptation. It is a supervised
adaptation sensitivity analysis in which target labels are deliberately used.
